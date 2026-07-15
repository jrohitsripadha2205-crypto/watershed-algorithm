import os
import json
import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading  
from concurrent.futures import ThreadPoolExecutor

# --- ENGINE IMPORTS ---
from skimage.segmentation import watershed
from skimage.morphology import disk
from scipy import ndimage

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- HIGH-PRECISION POWDER BED CORE PIPELINE ---
def process_powder_bed_pipeline(image_path, max_dim, blur_k, thresh_offset, morph_it, Use_Gradient, fg_thresh, min_dist, compactness, connectivity):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None, None, None
    
    # Perspective Correction / Flattening
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_warp = cv2.GaussianBlur(gray_full, (5, 5), 0)
    _, warp_thresh = cv2.threshold(blurred_warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(warp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Resize Strategy for responsive live editing
    h, w = img.shape[:2]
    if max_dim is not None and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_disp = img.copy()

    gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)
    
    # Enforce odd blur kernel constraint
    blur_k = int(blur_k)
    if blur_k % 2 == 0:
        blur_k = max(1, blur_k - 1)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    # --- DYNAMIC ADAPTIVE DUAL THRESHOLD IMPLEMENTATION ---
    # Map input entry variables to dynamic segment limits dynamically
    bright_cutoff = np.clip(128 - thresh_offset, 1, 254)
    dark_cutoff = np.clip(128 + thresh_offset, 1, 254)
    
    _, bright_mask = cv2.threshold(blurred, bright_cutoff, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(blurred, dark_cutoff, 255, cv2.THRESH_BINARY_INV)
    
    combined_binary = cv2.bitwise_or(bright_mask, dark_mask)
    
    # Morphological Cleanup
    selem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, selem, iterations=int(morph_it))
    
    # Area Label Filtering (Guide's Size Bounds Rule: 50 to 10000 pixels)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
    filtered_mask = np.zeros_like(cleaned_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 50 <= area <= 10000:
            filtered_mask[labels_im == i] = 255
            
    # Generate Topography
    if Use_Gradient > 0:
        flooding_surface = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))).astype(np.float32)
    else:
        flooding_surface = -cv2.distanceTransform(filtered_mask, cv2.DIST_L2, 5)
        
    # Standard Connected Components Marker Seeding
    ret, markers = cv2.connectedComponents(filtered_mask)
    markers = markers + 1
    unknown = cv2.subtract(cv2.dilate(filtered_mask, selem, iterations=1), filtered_mask)
    markers[unknown == 255] = 0
    
    # Run high-speed Watershed Segmentation
    conn_footprint = disk(int(connectivity)) if connectivity > 1 else None
    segmentation_labels = watershed(flooding_surface, markers, mask=filtered_mask, 
                                    compactness=float(compactness), connectivity=conn_footprint)
    
    # Create the Visual Tab Output Masks (Background Gray, Highlights Red)
    bg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 1. White Defects Only (Vivid Red Overlay)
    white_display = bg_gray.copy()
    white_display[cv2.bitwise_and(filtered_mask, bright_mask) > 0] = [255, 0, 0]
    
    # 2. Dark Defects Only (Vivid Red Overlay)
    dark_display = bg_gray.copy()
    dark_display[cv2.bitwise_and(filtered_mask, dark_mask) > 0] = [255, 0, 0]
    
    # 3. Unified Output (Both white and dark spots highlighted in Vivid Red)
    final_output = bg_gray.copy()
    final_output[filtered_mask > 0] = [255, 0, 0]
    
    return img_disp, filtered_mask, flooding_surface, white_display, dark_display, final_output


# --- COMPONENT SPINBOX ---
class CTkSpinbox(ctk.CTkFrame):
    def __init__(self, parent, from_val, to_val, current_val, is_float=False, resolution=1, linked_var=None, on_update_callback=None):
        super().__init__(parent, fg_color="transparent")
        self.from_val = from_val
        self.to_val = to_val
        self.is_float = is_float
        self.resolution = resolution
        self.linked_var = linked_var
        self.callback = on_update_callback
        
        self.btn_down = ctk.CTkButton(self, text="▼", width=28, height=28, font=("Arial", 10), 
                                      fg_color="#37474f", hover_color="#263238", command=self.decrement)
        self.btn_down.pack(side=tk.LEFT, padx=2)
        
        self.entry = ctk.CTkEntry(self, width=65, height=28, font=('Helvetica', 12), justify='center')
        self.entry.insert(0, f"{current_val:.2f}" if self.is_float else str(int(current_val)))
        self.entry.pack(side=tk.LEFT, padx=2)
        
        self.btn_up = ctk.CTkButton(self, text="▲", width=28, height=28, font=("Arial", 10), 
                                    fg_color="#37474f", hover_color="#263238", command=self.increment)
        self.btn_up.pack(side=tk.LEFT, padx=2)

    def get_value(self):
        try:
            return float(self.entry.get()) if self.is_float else int(float(self.entry.get()))
        except ValueError:
            return self.from_val

    def set_value(self, val):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"{val:.2f}" if self.is_float else str(int(val)))
        if self.linked_var:
            self.linked_var.set(val)

    def increment(self):
        curr = self.get_value()
        new_val = np.clip(curr + self.resolution, self.from_val, self.to_val)
        if self.is_float:
            new_val = round(new_val, 3)
        self.set_value(new_val)
        if self.callback: self.callback()

    def decrement(self):
        curr = self.get_value()
        new_val = np.clip(curr - self.resolution, self.from_val, self.to_val)
        if self.is_float:
            new_val = round(new_val, 3)
        self.set_value(new_val)
        if self.callback: self.callback()


# --- ULTIMATE BATCH INSPECTION WORKSTATION ---
class UltimateWatershedWorkstation(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Additive Manufacturing Powder Bed Inspection Workstation")
        self.geometry("1600x900")
        self.image_path = None
        self.folder_path = None
        self.max_display_dim = 400  
        self.updating_ui = False  

        self.setup_ui_architecture()

    def setup_ui_architecture(self):
        top_bar = ctk.CTkFrame(self, height=60, corner_radius=0)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        load_btn = ctk.CTkButton(top_bar, text="📂 Load Single Image", command=self.load_image_action, 
                                 font=('Helvetica', 12, 'bold'), fg_color="#0288d1", hover_color="#01579b")
        load_btn.pack(side=tk.LEFT, padx=15, pady=10)
        
        load_folder_btn = ctk.CTkButton(top_bar, text="📁 Load Batch Folder", command=self.load_folder_action, 
                                        font=('Helvetica', 12, 'bold'), fg_color="#ff8f00", hover_color="#c67100")
        load_folder_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.path_lbl = ctk.CTkLabel(top_bar, text="No source files loaded.", font=('Helvetica', 12, 'italic'), text_color="#90a4ae")
        self.path_lbl.pack(side=tk.LEFT, padx=15)

        # 6-Column High-Speed Layout Viewport Frame
        self.viewport_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e1e1e")
        self.viewport_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        for i in range(6):
            self.viewport_frame.columnconfigure(i, weight=1, uniform="viewport")
        self.viewport_frame.rowconfigure(1, weight=1)
        
        titles = [
            "1. Raw Input View", 
            "2. Clean Binary Mask", 
            "3. Surface Gradient", 
            "4. White Defects (Red)", 
            "5. Dark Defects (Red)", 
            "6. Unified Output"
        ]
        for i, t in enumerate(titles):
            lbl = ctk.CTkLabel(self.viewport_frame, text=t, font=('Helvetica', 11, 'bold'), text_color="white")
            lbl.grid(row=0, column=i, pady=(10, 5), sticky="ew")

        self.lbl_view1 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view1.grid(row=1, column=0, padx=3, pady=5, sticky="nsew")
        self.lbl_view2 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view2.grid(row=1, column=1, padx=3, pady=5, sticky="nsew")
        self.lbl_view3 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view3.grid(row=1, column=2, padx=3, pady=5, sticky="nsew")
        self.lbl_view4 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view4.grid(row=1, column=3, padx=3, pady=5, sticky="nsew")
        self.lbl_view5 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view5.grid(row=1, column=4, padx=3, pady=5, sticky="nsew")
        self.lbl_view6 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view6.grid(row=1, column=5, padx=3, pady=5, sticky="nsew")

        # Control Panel Options Panel
        control_panel = ctk.CTkFrame(self, height=320)
        control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        self.tabview = ctk.CTkTabview(control_panel, height=280)
        self.tabview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tab_prep = self.tabview.add(" 1. Preprocessing & Binary Filters ")
        self.tab_dist = self.tabview.add(" 2. Topography & Distance Analytics ")
        self.tab_ws   = self.tabview.add(" 3. Watershed Engine Optimization ")
        
        self.build_tabs_layout()
        
        right_bar = ctk.CTkFrame(control_panel, width=240, fg_color="transparent")
        right_bar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.export_btn = ctk.CTkButton(right_bar, text="💾 Export Output\n(Single or Batch Folder)", command=self.start_async_export,
                                   font=('Helvetica', 12, 'bold'), fg_color="#2e7d32", hover_color="#1b5e20")
        self.export_btn.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_parameter_row(self, parent, label_text, from_val, to_val, current_val, row_idx, is_float=False, resolution=1):
        lbl = ctk.CTkLabel(parent, text=label_text, font=('Helvetica', 12), anchor='w')
        lbl.grid(row=row_idx, column=0, sticky='w', pady=8, padx=10)
        
        var = tk.DoubleVar(value=current_val) if is_float else tk.IntVar(value=current_val)
        
        slider = ctk.CTkSlider(parent, from_=from_val, to=to_val, number_of_steps=int((to_val-from_val)/resolution), variable=var)
        slider.grid(row=row_idx, column=1, sticky='ew', padx=20, pady=8)
        
        spinbox = CTkSpinbox(parent, from_val, to_val, current_val, is_float, resolution, linked_var=var, on_update_callback=self.trigger_pipeline_refresh)
        spinbox.grid(row=row_idx, column=2, padx=10, pady=8)
        
        slider.bind("<ButtonRelease-1>", lambda event: self.sync_slider_to_spinbox(var, spinbox))
        spinbox.entry.bind("<Return>", lambda event: self.sync_spinbox_to_slider(spinbox, var, from_val, to_val))
        
        parent.columnconfigure(1, weight=1)
        return var, spinbox

    def sync_slider_to_spinbox(self, var, spinbox):
        if self.updating_ui: return
        self.updating_ui = True
        spinbox.set_value(var.get())
        self.updating_ui = False
        self.trigger_pipeline_refresh()

    def sync_spinbox_to_slider(self, spinbox, var, from_val, to_val):
        if self.updating_ui: return
        self.updating_ui = True
        val = spinbox.get_value()
        if from_val <= val <= to_val:
            var.set(val)
            self.updating_ui = False
            self.trigger_pipeline_refresh()
            return
        else:
            messagebox.showwarning("Out of Bounds", f"Value must be between {from_val} and {to_val}")
        self.updating_ui = False

    def build_tabs_layout(self):
        # Applied your optimal sample baselines directly as starting parameters
        self.v_blur, self.s_blur = self.create_parameter_row(self.tab_prep, "Gaussian Blur Smoothing Filter Kernel (Odd):", 1, 31, 15, 0, resolution=2)
        self.v_thresh, self.s_thresh = self.create_parameter_row(self.tab_prep, "Otsu Dynamic Threshold Offset Adjustment:", -120, 120, 50, 1, resolution=1)
        self.v_morph, self.s_morph = self.create_parameter_row(self.tab_prep, "Morphological Cleanup Operations (Iterations):", 0, 15, 1, 2, resolution=1)
        
        ctk.CTkLabel(self.tab_dist, text="Target Watershed Topography Surface Style:", font=('Helvetica', 12, 'bold'), text_color='white').grid(row=0, column=0, sticky='w', pady=8, padx=10)
        self.v_surface_mode = tk.IntVar(value=1) 
        self.rb_dist = ctk.CTkRadioButton(self.tab_dist, text="Distance Transform (Valleys)", variable=self.v_surface_mode, value=0, command=self.trigger_pipeline_refresh)
        self.rb_dist.grid(row=0, column=1, sticky='w', padx=10)
        self.rb_grad = ctk.CTkRadioButton(self.tab_dist, text="Edge Morphological Gradient (Ridges)", variable=self.v_surface_mode, value=1, command=self.trigger_pipeline_refresh)
        self.rb_grad.grid(row=0, column=2, sticky='w', padx=10)
        
        self.v_fgt, self.s_fgt = self.create_parameter_row(self.tab_dist, "Foreground Peak Detection Cutoff Threshold Ratio:", 0.0, 0.95, 0.50, 1, is_float=True, resolution=0.05)

        self.v_mdist, self.s_mdist = self.create_parameter_row(self.tab_ws, "Peak Marker Minimum Separation Distance Limit:", 1, 100, 5, 0, resolution=1)
        self.v_comp, self.s_comp = self.create_parameter_row(self.tab_ws, "Watershed Compactness Regularization (Shape Factor):", 0.0, 5.0, 0.00, 1, is_float=True, resolution=0.05)
        self.v_conn, self.s_conn = self.create_parameter_row(self.tab_ws, "Engine Connectivity Footprint Neighborhood Metric (Radius):", 1, 10, 1, 2, resolution=1)

    def load_image_action(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Image Formats", "*.jpg *.jpeg *.png *.bmp *.tiff *.pgm")])
        if file_path:
            self.folder_path = None 
            self.image_path = file_path
            self.path_lbl.configure(text=os.path.basename(file_path))
            self.trigger_pipeline_refresh()

    def load_folder_action(self):
        folder_path = filedialog.askdirectory(title="Select Batch Folder of AM Images")
        if folder_path:
            self.image_path = None 
            self.folder_path = folder_path
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            self.path_lbl.configure(text=f"Batch Folder: {os.path.basename(folder_path)} ({len(files)} target images)")
            
            if len(files) > 0:
                self.image_path = os.path.join(folder_path, files[0])
                self.trigger_pipeline_refresh()

    def trigger_pipeline_refresh(self):
        if not self.image_path or self.updating_ui:
            return
            
        blur = int(self.s_blur.get_value())
        offset = int(self.s_thresh.get_value())
        morph = int(self.s_morph.get_value())
        surf_mode = int(self.v_surface_mode.get())
        fgt = float(self.s_fgt.get_value())
        mdist = int(self.s_mdist.get_value())
        comp = float(self.s_comp.get_value())
        conn = int(self.s_conn.get_value())
        
        disp_img, mask, topology, white_disp, dark_disp, final_disp = process_powder_bed_pipeline(
            self.image_path, self.max_display_dim, blur, offset,
            morph, surf_mode, fgt, mdist, comp, conn
        )
        
        if disp_img is not None:
            self.update_native_viewports(disp_img, mask, topology, white_disp, dark_disp, final_disp, surf_mode)

    def convert_to_ctk_image(self, cv_img, is_gray=False, use_cmap=None):
        if use_cmap == "viridis" or use_cmap == "magma":
            norm_topo = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_mapped = cv2.applyColorMap(norm_topo, cv2.COLORMAP_VIRIDIS if use_cmap == "viridis" else cv2.COLORMAP_MAGMA)
            rgb = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)
        elif is_gray:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        h, w = rgb.shape[:2]
        pil_img = Image.fromarray(rgb)
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))

    def update_native_viewports(self, disp_img, mask, topology, white_disp, dark_disp, final_disp, surf_mode):
        ctk_img1 = self.convert_to_ctk_image(disp_img, is_gray=False)
        ctk_img2 = self.convert_to_ctk_image(mask, is_gray=True)
        
        cmap_choice = 'magma' if surf_mode > 0 else 'viridis'
        ctk_img3 = self.convert_to_ctk_image(topology, use_cmap=cmap_choice)
        
        ctk_img4 = self.convert_to_ctk_image(white_disp, is_gray=False)
        ctk_img5 = self.convert_to_ctk_image(dark_disp, is_gray=False)
        ctk_img6 = self.convert_to_ctk_image(final_disp, is_gray=False)
        
        self.lbl_view1.configure(image=ctk_img1, text="")
        self.lbl_view1.image = ctk_img1
        self.lbl_view2.configure(image=ctk_img2, text="")
        self.lbl_view2.image = ctk_img2
        self.lbl_view3.configure(image=ctk_img3, text="")
        self.lbl_view3.image = ctk_img3
        self.lbl_view4.configure(image=ctk_img4, text="")
        self.lbl_view4.image = ctk_img4
        self.lbl_view5.configure(image=ctk_img5, text="")
        self.lbl_view5.image = ctk_img5
        self.lbl_view6.configure(image=ctk_img6, text="")
        self.lbl_view6.image = ctk_img6

    def start_async_export(self):
        if not self.image_path and not self.folder_path:
            messagebox.showwarning("Export Error", "Please load a single image or batch folder first.")
            return
        
        selected_parent_dir = filedialog.askdirectory(title="Select Destination Folder to Export Output Directory")
        if not selected_parent_dir:
            return 
            
        self.export_btn.configure(state="disabled", text="⏳ Running High-Res Batch Pool...")
        
        export_thread = threading.Thread(target=self.background_export_worker, args=(selected_parent_dir,))
        export_thread.daemon = True
        export_thread.start()

    def background_export_worker(self, target_base_path):
        try:
            blur = int(self.s_blur.get_value())
            offset = int(self.s_thresh.get_value())
            morph = int(self.s_morph.get_value())
            surf_mode = int(self.v_surface_mode.get())
            fgt = float(self.s_fgt.get_value())
            mdist = int(self.s_mdist.get_value())
            comp = float(self.s_comp.get_value())
            conn = int(self.s_conn.get_value())
            
            if self.folder_path:
                output_dir = os.path.join(target_base_path, "Output_images")
                os.makedirs(output_dir, exist_ok=True)
                files_to_process = [
                    os.path.join(self.folder_path, f) 
                    for f in os.listdir(self.folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
                ]
            else:
                base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
                output_dir = os.path.join(target_base_path, f"watershed_{base_filename}_outputs")
                os.makedirs(output_dir, exist_ok=True)
                files_to_process = [self.image_path]

            def process_single_file(file_path):
                file_name_only = os.path.splitext(os.path.basename(file_path))[0]
                
                _, full_mask, full_topo, full_white, full_dark, full_final = process_powder_bed_pipeline(
                    file_path, None, blur, offset, morph, surf_mode, fgt, mdist, comp, conn
                )
                
                if full_final is None:
                    return

                if self.folder_path:
                    output_file_name = f"{file_name_only}_output.png"
                    cv2.imwrite(os.path.join(output_dir, output_file_name), cv2.cvtColor(full_final, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(os.path.join(output_dir, "export_1_input.png"), cv2.imread(file_path))
                    cv2.imwrite(os.path.join(output_dir, "export_2_binary_mask.png"), full_mask)
                    
                    topo_norm = cv2.normalize(full_topo, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv_cmap = cv2.COLORMAP_MAGMA if surf_mode > 0 else cv2.COLORMAP_VIRIDIS
                    cv2.imwrite(os.path.join(output_dir, "export_3_topography.png"), cv2.applyColorMap(topo_norm, cv_cmap))
                    cv2.imwrite(os.path.join(output_dir, "export_4_white_defects.png"), cv2.cvtColor(full_white, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, "export_5_dark_defects.png"), cv2.cvtColor(full_dark, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, "export_6_final_segmentation.png"), cv2.cvtColor(full_final, cv2.COLOR_RGB2BGR))
                    
                    export_config = {
                        "algorithm_metadata": "hybrid_opencv_skimage_universal_workstation_pipeline",
                        "source_file_reference": os.path.basename(file_path),
                        "optimized_parameters": {
                            "gaussian_blur_kernel": blur if blur % 2 != 0 else max(1, blur - 1),
                            "otsu_threshold_offset": offset,
                            "morphological_open_iterations": morph,
                            "topography_surface_selection": "Morphological_Gradient" if surf_mode > 0 else "Distance_Transform",
                            "foreground_peak_cutoff_ratio": fgt,
                            "watershed_peak_min_distance": mdist,
                            "watershed_compactness": comp,
                            "watershed_connectivity_radius": conn
                        }
                    }
                    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
                        json.dump(export_config, f, indent=4)

            with ThreadPoolExecutor() as executor:
                executor.map(process_single_file, files_to_process)

            self.after(0, lambda: self.on_export_complete(output_dir))
            
        except Exception as e:
            self.after(0, lambda: self.on_export_failed(str(e)))

    def on_export_complete(self, output_dir):
        self.export_btn.configure(state="normal", text="💾 Export Output\n(Single or Batch Folder)")
        messagebox.showinfo("Export Successful", f"All operations completed safely!\nTarget Directory:\n{output_dir}")

    def on_export_failed(self, error_msg):
        self.export_btn.configure(state="normal", text="💾 Export Output\n(Single or Batch Folder)")
        messagebox.showerror("Export Error", f"An unexpected processing error occurred:\n{error_msg}")


if __name__ == "__main__":
    app = UltimateWatershedWorkstation()
    app.mainloop()