import os
import sys
import time
import numpy as np
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from scipy import ndimage

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget,
    QGroupBox, QSlider, QSpinBox, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QLineEdit, QComboBox, QGridLayout, QProgressBar,
    QStatusBar, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage

import concurrent.futures
import multiprocessing

# ------------------------------------------------------------
#  UTILITY FUNCTIONS (Image Processing Pipeline)
# ------------------------------------------------------------

def apply_perspective_correction(img):
    """Detect the largest quadrilateral and warp to a top-down rectangle."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_quad = None
    max_area_cont = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            if area > max_area_cont:
                max_area_cont = area
                largest_quad = approx
    if largest_quad is not None:
        pts = largest_quad.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return img


def compute_defect_mask(gray, blur_k, offset, morph_it, bright_bias=0, dark_bias=0):
    """
    Build a TWO-SIDED defect mask: pixels that are locally brighter OR darker
    than the surrounding grey powder-bed background.

    This replaces the old single-direction Otsu threshold (which could only
    ever mark ONE polarity as foreground - the reason dark spots on a grey
    background were never detected before). Here we estimate a local
    background via a large Gaussian blur, take the signed difference of each
    real pixel from that local background, and threshold BOTH tails.
    """
    blur_k = int(blur_k)
    if blur_k % 2 == 0:
        blur_k = max(1, blur_k - 1)
    blur_k = max(3, blur_k)
    # Local background estimate (slow-varying illumination / grey powder level)
    background = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    diff = gray.astype(np.int16) - background.astype(np.int16)
    abs_diff = np.abs(diff).astype(np.uint8)
    # Otsu picks a data-driven separation point between "noise" and "real defect"
    otsu_val, _ = cv2.threshold(abs_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # `offset` (guide-recommended ~49-51) is normalized around a neutral value
    # of 50 so the slider scales sensitivity up/down from the Otsu estimate.
    scale = max(0.1, float(offset) / 50.0)
    base_thresh = max(3.0, otsu_val * scale)
    bright_thresh = max(1.0, base_thresh - bright_bias)
    dark_thresh = max(1.0, base_thresh - dark_bias)
    bright_raw = ((diff > bright_thresh).astype(np.uint8)) * 255
    dark_raw = ((diff < -dark_thresh).astype(np.uint8)) * 255
    combined = cv2.bitwise_or(bright_raw, dark_raw)
    if morph_it and int(morph_it) > 0:
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, selem, iterations=int(morph_it))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, selem, iterations=int(morph_it))
        bright_raw = cv2.bitwise_and(bright_raw, combined)
        dark_raw = cv2.bitwise_and(dark_raw, combined)
    return combined, bright_raw, dark_raw


def run_watershed(img, max_dim, blur_k, thresh_offset, morph_it,
                  use_gradient, fg_thresh, min_dist, compactness, connectivity,
                  bright_bias=0, dark_bias=0):
    """Run watershed segmentation on a BGR image, detecting BOTH bright and
    dark defects against the grey background in a single pass."""
    if img is None:
        return None, None, None, None, None, None
    h, w = img.shape[:2]
    if max_dim is not None and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_disp = img.copy()
    gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)

    cleaned_mask, bright_raw, dark_raw = compute_defect_mask(
        gray, blur_k, thresh_offset, morph_it, bright_bias, dark_bias)

    if use_gradient > 0:
        gradient_surface = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        flooding_surface = gradient_surface.astype(np.float32)
        peak_surface = cv2.bitwise_not(gradient_surface)
        peak_surface[cleaned_mask == 0] = 0
        topo = flooding_surface
    else:
        distance_map = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        flooding_surface = -distance_map
        peak_surface = distance_map.copy()
        max_dist_val = distance_map.max()
        if max_dist_val > 0:
            peak_surface[distance_map < (fg_thresh * max_dist_val)] = 0
        topo = distance_map

    if cleaned_mask.any():
        coords = peak_local_max(peak_surface, min_distance=int(min_dist), labels=cleaned_mask)
    else:
        coords = np.empty((0, 2), dtype=int)
    if len(coords) == 0:
        markers = (cleaned_mask > 0).astype(np.int32)
    else:
        peak_mask = np.zeros(cleaned_mask.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(peak_mask)
    conn_footprint = disk(int(connectivity)) if connectivity > 1 else None
    labels = watershed(flooding_surface, markers, mask=cleaned_mask,
                       compactness=float(compactness), connectivity=conn_footprint)
    return img_disp, cleaned_mask, topo, labels, bright_raw, dark_raw


def classify_defects(labels, bright_raw, dark_raw, min_area=50, max_area=10000):
    """
    Assign each watershed-segmented region to 'bright' or 'dark' based on
    which raw polarity mask it overlaps most, then filter by area
    (per the guide: keep only blobs between 50 and 10000 px - real defects,
    not noise specks or oversized background artifacts).
    """
    bright_mask = np.zeros_like(bright_raw, dtype=np.uint8)
    dark_mask = np.zeros_like(dark_raw, dtype=np.uint8)
    num_labels = labels.max() + 1
    for label_id in range(1, num_labels):
        region = (labels == label_id)
        area = int(np.sum(region))
        if area < min_area or area > max_area:
            continue
        bright_overlap = int(np.count_nonzero(bright_raw[region]))
        dark_overlap = int(np.count_nonzero(dark_raw[region]))
        if bright_overlap == 0 and dark_overlap == 0:
            continue
        if bright_overlap >= dark_overlap:
            bright_mask[region] = 255
        else:
            dark_mask[region] = 255
    return bright_mask, dark_mask


def defect_overlay(img_bgr, bright_mask, dark_mask, mode='combined'):
    """
    Return an RGB image with grey background and defects coloured.
    mode: 'white', 'dark', 'combined'
    """
    h, w = img_bgr.shape[:2]
    grey = np.full((h, w, 3), 128, dtype=np.uint8)  # uniform grey background
    if mode == 'white':
        grey[dark_mask > 0] = [0, 0, 0]
        grey[bright_mask > 0] = [0, 0, 255]
    elif mode == 'dark':
        grey[bright_mask > 0] = [255, 255, 255]
        grey[dark_mask > 0] = [0, 0, 255]
    else:  # combined
        combined = cv2.bitwise_or(bright_mask, dark_mask)
        grey[combined > 0] = [0, 0, 255]
    return cv2.cvtColor(grey, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
#  BATCH PROCESSING WORKER (standalone function for multiprocessing)
# ------------------------------------------------------------
def process_single_batch_image(img_path, blur, offset, morph, surf, fgt, mdist, comp, conn,
                               bright_bias, dark_bias, min_area, max_area, max_dim=None):
    """
    Process one image for batch mode; returns (filename, bgr_overlay, error).
    Runs inside a worker process - never raises, always reports failures back
    to the main thread so one bad/corrupt image can't kill the whole batch.
    """
    try:
        # Each worker process must NOT let OpenCV spawn its own internal
        # thread pool - with N worker *processes* already running in
        # parallel, OpenCV's default multi-threading massively oversubscribes
        # the CPU and slows the whole batch down. One thread per process only.
        cv2.setNumThreads(1)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            return os.path.basename(img_path), None, "Could not read image (corrupt or unsupported format)."
        img_corr = apply_perspective_correction(img)
        _, _, _, labels, bright_raw, dark_raw = run_watershed(
            img_corr, max_dim, blur, offset, morph, surf, fgt, mdist, comp, conn,
            bright_bias, dark_bias)
        if labels is None:
            return os.path.basename(img_path), None, "Watershed segmentation failed."
        bmask, dmask = classify_defects(labels, bright_raw, dark_raw, min_area, max_area)
        combined_rgb = defect_overlay(img_corr, bmask, dmask, mode='combined')
        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
        return os.path.basename(img_path), combined_bgr, None
    except Exception as e:
        return os.path.basename(img_path), None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------
#  THREAD CLASSES (top-level)
# ------------------------------------------------------------
class BatchThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, folder, files, out_dir, params):
        super().__init__()
        self.folder = folder
        self.files = files
        self.out_dir = out_dir
        self.params = params

    def run(self):
        start = time.time()
        processed = 0
        failures = []
        args_list = []
        for fname in self.files:
            full_path = os.path.join(self.folder, fname)
            args_list.append((
                full_path,
                self.params['blur'], self.params['offset'], self.params['morph'],
                self.params['surf'], self.params['fgt'], self.params['mdist'],
                self.params['comp'], self.params['conn'],
                self.params['bright'], self.params['dark'],
                self.params['min_area'], self.params['max_area'],
                self.params.get('max_dim')
            ))

        # Cap worker processes: each worker decodes a full-resolution (up to
        # ~15MB) image at once, so uncapped os.cpu_count() workers on a large
        # batch can exhaust RAM and actually slow things down via thrashing.
        cpu = os.cpu_count() or 4
        max_workers = max(1, min(cpu, 8))
        # PNG compression level 3 (default is 6) writes noticeably faster
        # with a negligible size trade-off - matters a lot across 700-800 files.
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_batch_image, *args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                try:
                    fname, bgr_img, err = future.result()
                    if bgr_img is not None:
                        base = os.path.splitext(fname)[0]
                        out_path = os.path.join(self.out_dir, f"{base}_output.png")
                        cv2.imwrite(out_path, bgr_img, png_params)
                    elif err:
                        failures.append(f"{fname}: {err}")
                except Exception as e:
                    failures.append(str(e))
                processed += 1
                self.progress_signal.emit(processed)
        if failures:
            self.error_signal.emit(
                f"{len(failures)} image(s) failed and were skipped:\n" + "\n".join(failures[:20])
            )
        elapsed = time.time() - start
        self.finished_signal.emit(elapsed)


class ExportThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, img_path, out_folder, params):
        super().__init__()
        self.img_path = img_path
        self.out_folder = out_folder
        self.params = params

    def run(self):
        try:
            img = cv2.imread(self.img_path)
            if img is None:
                self.error_signal.emit("Could not read image.")
                return
            img_corr = apply_perspective_correction(img)
            _, _, _, labels, bright_raw, dark_raw = run_watershed(
                img_corr, None,
                self.params['blur'], self.params['offset'], self.params['morph'],
                self.params['surf'], self.params['fgt'], self.params['mdist'],
                self.params['comp'], self.params['conn'],
                self.params['bright'], self.params['dark']
            )
            if labels is None:
                self.error_signal.emit("Watershed failed.")
                return
            bmask, dmask = classify_defects(
                labels, bright_raw, dark_raw,
                self.params['min_area'], self.params['max_area']
            )
            combined_rgb = defect_overlay(img_corr, bmask, dmask, mode='combined')
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
            base = os.path.splitext(os.path.basename(self.img_path))[0]
            out_path = os.path.join(self.out_folder, f"{base}_combined_defects.png")
            cv2.imwrite(out_path, combined_bgr)
            self.finished_signal.emit(out_path)
        except Exception as e:
            self.error_signal.emit(str(e))


# ------------------------------------------------------------
#  MAIN GUI APPLICATION (PyQt6)
# ------------------------------------------------------------
class DefectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Additive Manufacturing Defect Detection System")
        self.setGeometry(100, 100, 1600, 950)

        # State
        self.image_path = None
        self.folder_path = None
        self.mode = "single"   # "single" or "batch"
        self.results = {
            'img': None,      # BGR, perspective-corrected, resized for display
            'mask': None,
            'topo': None,
            'labels': None,
            'bright': None,
            'dark': None,
        }
        self.max_display_dim = 500

        self.init_ui()
        self.connect_slider_signals()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Top bar
        top_bar = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Image Mode", "Batch Processing Mode"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        top_bar.addWidget(QLabel("Mode:"))
        top_bar.addWidget(self.mode_combo)

        self.load_btn = QPushButton("📂 Load Single Image")
        self.load_btn.clicked.connect(self.load_single_image)
        top_bar.addWidget(self.load_btn)

        self.batch_btn = QPushButton("📁 Load Batch Folder")
        self.batch_btn.clicked.connect(self.load_batch_folder)
        top_bar.addWidget(self.batch_btn)

        self.status_label = QLabel("No image loaded.")
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Tab widget (6 views)
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumHeight(500)
        self.tab_labels = []
        for name in ["Input", "Binary Mask", "Topography", "White Spots", "Dark Spots", "Final Segmentation"]:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            label = QLabel("Awaiting data...")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("QLabel { color: #888; font-size: 14px; }")
            layout.addWidget(label)
            self.tab_widget.addTab(tab, name)
            self.tab_labels.append(label)
        main_layout.addWidget(self.tab_widget, stretch=1)

        # Bottom panel
        bottom_panel = QHBoxLayout()

        # Parameter group
        self.param_group = QGroupBox("Parameters (Single Image Mode)")
        param_layout = QGridLayout()
        self.param_group.setLayout(param_layout)

        row = 0
        param_layout.addWidget(QLabel("Gaussian Blur (odd):"), row, 0)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 31)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(15)
        param_layout.addWidget(self.blur_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Otsu Offset:"), row, 0)
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-120, 120)
        self.offset_spin.setValue(50)
        param_layout.addWidget(self.offset_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Morph Iterations:"), row, 0)
        self.morph_spin = QSpinBox()
        self.morph_spin.setRange(0, 15)
        self.morph_spin.setValue(1)
        param_layout.addWidget(self.morph_spin, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Surface Style:"), row, 0)
        self.surf_group = QButtonGroup()
        self.surf_dist = QRadioButton("Distance Transform")
        self.surf_grad = QRadioButton("Edge Gradient (Ridges)")
        self.surf_grad.setChecked(True)
        self.surf_group.addButton(self.surf_dist, 0)
        self.surf_group.addButton(self.surf_grad, 1)
        hbox = QHBoxLayout()
        hbox.addWidget(self.surf_dist)
        hbox.addWidget(self.surf_grad)
        param_layout.addLayout(hbox, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Foreground Ratio:"), row, 0)
        self.fgt_spin = QDoubleSpinBox()
        self.fgt_spin.setRange(0.0, 0.95)
        self.fgt_spin.setSingleStep(0.05)
        self.fgt_spin.setValue(0.5)
        param_layout.addWidget(self.fgt_spin, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Min Distance:"), row, 0)
        self.mdist_spin = QSpinBox()
        self.mdist_spin.setRange(1, 100)
        self.mdist_spin.setValue(5)
        param_layout.addWidget(self.mdist_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Compactness:"), row, 0)
        self.comp_spin = QDoubleSpinBox()
        self.comp_spin.setRange(0.0, 5.0)
        self.comp_spin.setSingleStep(0.05)
        self.comp_spin.setValue(0.0)
        param_layout.addWidget(self.comp_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Connectivity:"), row, 0)
        self.conn_spin = QSpinBox()
        self.conn_spin.setRange(1, 10)
        self.conn_spin.setValue(1)
        param_layout.addWidget(self.conn_spin, row, 1)
        row += 1

        param_layout.addWidget(QLabel("Bright Sensitivity (+):"), row, 0)
        self.bright_spin = QSpinBox()
        self.bright_spin.setRange(-30, 30)
        self.bright_spin.setValue(0)
        self.bright_spin.setToolTip(
            "Bias added to the auto (Otsu) threshold for bright spots.\n"
            "Higher = more sensitive to faint bright defects.")
        param_layout.addWidget(self.bright_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Dark Sensitivity (+):"), row, 0)
        self.dark_spin = QSpinBox()
        self.dark_spin.setRange(-30, 30)
        self.dark_spin.setValue(0)
        self.dark_spin.setToolTip(
            "Bias added to the auto (Otsu) threshold for dark spots.\n"
            "Higher = more sensitive to faint dark defects.")
        param_layout.addWidget(self.dark_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Min Area (px):"), row, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 500)
        self.min_area_spin.setValue(50)
        param_layout.addWidget(self.min_area_spin, row, 1)
        row += 1
        param_layout.addWidget(QLabel("Max Area (px):"), row, 0)
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 50000)
        self.max_area_spin.setSingleStep(500)
        self.max_area_spin.setValue(10000)
        param_layout.addWidget(self.max_area_spin, row, 1)

        bottom_panel.addWidget(self.param_group, stretch=1)

        # Right panel
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.export_btn = QPushButton("💾 Export Final Defect Image")
        self.export_btn.clicked.connect(self.export_single)
        self.export_btn.setEnabled(False)
        right_panel.addWidget(self.export_btn)

        self.batch_status = QLabel("Batch: idle")
        self.batch_status.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        right_panel.addWidget(self.batch_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)

        bottom_panel.addLayout(right_panel)

        main_layout.addLayout(bottom_panel)

    def connect_slider_signals(self):
        widgets = [
            self.blur_spin, self.offset_spin, self.morph_spin,
            self.fgt_spin, self.mdist_spin, self.comp_spin, self.conn_spin,
            self.bright_spin, self.dark_spin, self.min_area_spin, self.max_area_spin
        ]
        for w in widgets:
            w.valueChanged.connect(self.on_parameter_changed)
        self.surf_dist.toggled.connect(self.on_parameter_changed)
        self.surf_grad.toggled.connect(self.on_parameter_changed)

    def on_parameter_changed(self):
        if self.mode == "single" and self.image_path is not None:
            self.process_single_image()

    def on_mode_changed(self, index):
        self.mode = "single" if index == 0 else "batch"
        if self.mode == "single":
            self.param_group.setEnabled(True)
            self.export_btn.setEnabled(self.image_path is not None)
            self.batch_status.setText("Batch: idle")
            self.progress_bar.setVisible(False)
            if self.image_path is not None:
                self.process_single_image()
        else:
            self.param_group.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.batch_status.setText("Batch: ready")

    def load_single_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.pgm)")
        if path:
            self.image_path = path
            self.folder_path = None
            self.status_label.setText(os.path.basename(path))
            self.mode_combo.setCurrentIndex(0)
            self.export_btn.setEnabled(True)
            self.process_single_image()

    def load_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if folder:
            self.folder_path = folder
            self.image_path = None
            self.status_label.setText(f"Batch: {os.path.basename(folder)}")
            self.mode_combo.setCurrentIndex(1)
            self.export_btn.setEnabled(False)
            self.start_batch_processing(folder)

    # ------------------------------------------------------------
    #  SINGLE IMAGE PROCESSING
    # ------------------------------------------------------------
    def process_single_image(self):
        if not self.image_path:
            return
        params = self.get_current_parameters()
        img = cv2.imread(self.image_path)
        if img is None:
            QMessageBox.critical(self, "Error", "Could not read image file.")
            return
        img_corr = apply_perspective_correction(img)
        disp_img, mask, topo, labels, bright_raw, dark_raw = run_watershed(
            img_corr, self.max_display_dim,
            params['blur'], params['offset'], params['morph'],
            params['surf'], params['fgt'], params['mdist'],
            params['comp'], params['conn'],
            params['bright'], params['dark']
        )
        if disp_img is None:
            QMessageBox.critical(self, "Error", "Watershed processing failed.")
            return
        bright_mask, dark_mask = classify_defects(
            labels, bright_raw, dark_raw,
            params['min_area'], params['max_area']
        )
        self.results['img'] = disp_img
        self.results['mask'] = mask
        self.results['topo'] = topo
        self.results['labels'] = labels
        self.results['bright'] = bright_mask
        self.results['dark'] = dark_mask
        self.update_display(params['surf'])

    def get_current_parameters(self):
        return {
            'blur': self.blur_spin.value(),
            'offset': self.offset_spin.value(),
            'morph': self.morph_spin.value(),
            'surf': 1 if self.surf_grad.isChecked() else 0,
            'fgt': self.fgt_spin.value(),
            'mdist': self.mdist_spin.value(),
            'comp': self.comp_spin.value(),
            'conn': self.conn_spin.value(),
            'bright': self.bright_spin.value(),
            'dark': self.dark_spin.value(),
            'min_area': self.min_area_spin.value(),
            'max_area': self.max_area_spin.value(),
        }

    def update_display(self, surf_mode):
        img = self.results['img']
        if img is None:
            return
        # Tab 0: Input
        pixmap = self.cv_to_pixmap(img)
        self.tab_labels[0].setPixmap(pixmap)
        self.tab_labels[0].setScaledContents(True)
        # Tab 1: Binary Mask
        mask = self.results['mask']
        pixmap2 = self.cv_to_pixmap(mask, is_gray=True)
        self.tab_labels[1].setPixmap(pixmap2)
        self.tab_labels[1].setScaledContents(True)
        # Tab 2: Topography
        topo = self.results['topo']
        cmap = cv2.COLORMAP_MAGMA if surf_mode == 1 else cv2.COLORMAP_VIRIDIS
        topo_norm = cv2.normalize(topo, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        topo_color = cv2.applyColorMap(topo_norm, cmap)
        pixmap3 = self.cv_to_pixmap(topo_color)
        self.tab_labels[2].setPixmap(pixmap3)
        self.tab_labels[2].setScaledContents(True)
        # Tabs 3-5
        bright = self.results['bright']
        dark = self.results['dark']
        white_view = defect_overlay(img, bright, dark, mode='white')
        pixmap4 = self.cv_to_pixmap(white_view, is_rgb=True)
        self.tab_labels[3].setPixmap(pixmap4)
        self.tab_labels[3].setScaledContents(True)
        dark_view = defect_overlay(img, bright, dark, mode='dark')
        pixmap5 = self.cv_to_pixmap(dark_view, is_rgb=True)
        self.tab_labels[4].setPixmap(pixmap5)
        self.tab_labels[4].setScaledContents(True)
        combined_view = defect_overlay(img, bright, dark, mode='combined')
        pixmap6 = self.cv_to_pixmap(combined_view, is_rgb=True)
        self.tab_labels[5].setPixmap(pixmap6)
        self.tab_labels[5].setScaledContents(True)

    # ------------------------------------------------------------
    #  BATCH PROCESSING
    # ------------------------------------------------------------
    def start_batch_processing(self, folder):
        # Guide-recommended parametric values, tuned on sample images -
        # used directly for unattended batch runs (no per-image manual tuning).
        params = {
            'blur': 15, 'offset': 50, 'morph': 1, 'surf': 1,
            'fgt': 0.5, 'mdist': 5, 'comp': 0.0, 'conn': 1,
            'bright': 0, 'dark': 0, 'min_area': 50, 'max_area': 10000,
            'max_dim': None,  # full resolution - no downscaling, for max precision
        }
        image_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pgm')
        files = [f for f in os.listdir(folder) if f.lower().endswith(image_ext)]
        if not files:
            QMessageBox.warning(self, "Batch Error", "No image files found in the selected folder.")
            return
        out_dir = os.path.join(folder, "Output images")
        os.makedirs(out_dir, exist_ok=True)

        self.batch_status.setText("Batch: processing...")
        self.batch_status.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)

        self.batch_thread = BatchThread(folder, files, out_dir, params)
        self.batch_thread.progress_signal.connect(self.update_batch_progress)
        self.batch_thread.finished_signal.connect(self.batch_finished)
        self.batch_thread.error_signal.connect(self.batch_error)
        self.batch_thread.start()

    def update_batch_progress(self, value):
        self.progress_bar.setValue(value)

    def batch_finished(self, elapsed):
        self.batch_status.setText(f"Batch: done ({elapsed:.2f}s)")
        self.batch_status.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Batch Complete",
                                f"Processed {self.progress_bar.maximum()} images in {elapsed:.2f} seconds.\n"
                                f"Output saved in 'Output images' folder.")

    def batch_error(self, msg):
        QMessageBox.critical(self, "Batch Error", f"An error occurred: {msg}")

    # ------------------------------------------------------------
    #  EXPORT (Single Image)
    # ------------------------------------------------------------
    def export_single(self):
        if not self.image_path or self.mode != "single":
            QMessageBox.information(self, "Export", "Load a single image first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not folder:
            return
        self.export_btn.setEnabled(False)
        self.export_btn.setText("⏳ Exporting...")
        params = self.get_current_parameters()
        self.export_thread = ExportThread(self.image_path, folder, params)
        self.export_thread.finished_signal.connect(self.export_done)
        self.export_thread.error_signal.connect(self.export_error)
        self.export_thread.start()

    def export_done(self, path):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("💾 Export Final Defect Image")
        QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")

    def export_error(self, msg):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("💾 Export Final Defect Image")
        QMessageBox.critical(self, "Export Error", msg)

    # ------------------------------------------------------------
    #  HELPER: Convert OpenCV image to QPixmap
    # ------------------------------------------------------------
    def cv_to_pixmap(self, img, is_gray=False, is_rgb=False):
        if img is None:
            return QPixmap()
        if is_gray:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape
            bytes_per_line = w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif is_rgb:
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)


# ------------------------------------------------------------
#  MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = DefectDetectionApp()
    window.show()
    sys.exit(app.exec())