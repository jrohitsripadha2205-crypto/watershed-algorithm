"""
gui.py
------
The main application window for Watershed Segmentation Studio.

Architecture notes:
    * All OpenCV work happens on a background thread (see
      `_trigger_processing` / `_worker`), so the UI never freezes even
      while a parameter is being dragged. Results are marshalled back to
      the main thread via `self.after(0, ...)`, which is the standard,
      thread-safe way to schedule Tkinter/matplotlib updates from a
      worker thread.
    * Slider/dropdown changes are debounced (see config.DEBOUNCE_MS) so
      that dragging a slider doesn't spawn a flood of redundant
      recomputations - only the last value after a short pause triggers
      a refresh.
    * A monotonically increasing request id discards stale results, so
      if the user changes parameters again before a previous background
      computation finishes, the old result is simply ignored when it
      arrives.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, Optional

import cv2
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import config
import processor as proc
from utils import cv2_to_ctk_image, setup_logging
from widgets import LabeledSlider, ThumbnailPanel, Tooltip

logger = setup_logging()


class App(ctk.CTk):
    """Top-level application window."""

    def __init__(self):
        super().__init__()

        self.title(f"{config.APP_NAME}  v{config.APP_VERSION}")
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(config.WINDOW_MIN_WIDTH, config.WINDOW_MIN_HEIGHT)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- state ----
        self.original_full = None       # full-resolution loaded image (BGR)
        self.preview_source = None      # downscaled copy used for live processing
        self.params: Dict[str, Any] = config.default_params()

        self.slider_widgets: Dict[str, LabeledSlider] = {}
        self.dropdown_widgets: Dict[str, ctk.CTkOptionMenu] = {}

        self._debounce_job: Optional[str] = None
        self._request_id = 0
        self._active_ops = 0            # counts in-flight background operations
        self._im_left = None            # matplotlib artist, created lazily
        self._im_right = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_menu_bar()
        self._build_toolbar()
        self._build_tabs()
        self._build_preview_area()
        self._build_status_bar()
        self._bind_shortcuts()

        logger.info("%s v%s started.", config.APP_NAME, config.APP_VERSION)

    # ======================================================================
    # Menu bar (native OS menu - CustomTkinter has no themed menu widget)
    # ======================================================================
    def _build_menu_bar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load Image...", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Result...", command=self.save_result, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Save Configuration...", command=self.save_configuration)
        file_menu.add_command(label="Load Configuration...", command=self.load_configuration)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Reset Parameters", command=self.reset_params, accelerator="Ctrl+R")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Toggle Dark / Light Mode", command=self.toggle_appearance)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menubar)

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self.load_image())
        self.bind("<Control-s>", lambda e: self.save_result())
        self.bind("<Control-r>", lambda e: self.reset_params())
        self.bind("<Control-q>", lambda e: self.on_close())

    # ======================================================================
    # Toolbar
    # ======================================================================
    def _build_toolbar(self):
        bar = ctk.CTkFrame(self, height=48, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ctk.CTkLabel(bar, text=config.APP_NAME,
                     font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=16, pady=8)

        ctk.CTkButton(bar, text="Save Result", width=110, command=self.save_result).pack(
            side="right", padx=6, pady=8)
        ctk.CTkButton(bar, text="Reset Parameters", width=140, command=self.reset_params).pack(
            side="right", padx=6, pady=8)
        ctk.CTkButton(bar, text="Load Image", width=110, command=self.load_image).pack(
            side="right", padx=6, pady=8)

    # ======================================================================
    # Left side: tabbed parameter panel
    # ======================================================================
    def _build_tabs(self):
        self.tabview = ctk.CTkTabview(self, width=360)
        self.tabview.grid(row=1, column=0, sticky="nswe", padx=(10, 5), pady=10)

        tab_preprocess = self.tabview.add("Preprocessing")
        tab_threshold = self.tabview.add("Threshold")
        tab_morph = self.tabview.add("Morphology")
        tab_watershed = self.tabview.add("Watershed")

        self._build_preprocess_tab(tab_preprocess)
        self._build_threshold_tab(tab_threshold)
        self._build_morph_tab(tab_morph)
        self._build_watershed_tab(tab_watershed)

    def _add_sliders(self, tab, specs):
        """Build a LabeledSlider for each spec in `specs`, wire it to
        `_set_param`, and remember it in self.slider_widgets for later
        (reset / load configuration)."""
        for spec in specs:
            slider = LabeledSlider(
                tab, label=spec["label"], frm=spec["min"], to=spec["max"],
                steps=spec["steps"], default=spec["default"], is_float=spec["is_float"],
                tooltip=spec.get("tooltip"),
                on_change=(lambda v, key=spec["key"]: self._set_param(key, v)))
            slider.pack(fill="x", padx=10)
            self.slider_widgets[spec["key"]] = slider

    def _add_dropdown(self, tab, key, label, values, default, command, tooltip=None):
        text_label = ctk.CTkLabel(tab, text=label, font=ctk.CTkFont(weight="bold"))
        text_label.pack(anchor="w", padx=10, pady=(12, 2))
        if tooltip:
            Tooltip(text_label, tooltip)
        menu = ctk.CTkOptionMenu(tab, values=values, command=command)
        menu.set(str(default))
        menu.pack(fill="x", padx=10, pady=(0, 8))
        self.dropdown_widgets[key] = menu
        return menu

    def _build_preprocess_tab(self, tab):
        self._add_dropdown(
            tab, "blur_type", "Blur Type", ["None", "Gaussian", "Median", "Bilateral"],
            self.params["blur_type"], self._on_blur_type_change,
            tooltip="Smooths the image before thresholding, which helps\nreduce noise and spurious segments.")
        self._add_sliders(tab, config.PREPROCESS_SLIDERS)

    def _build_threshold_tab(self, tab):
        self._add_dropdown(
            tab, "threshold_mode", "Threshold Mode", ["Binary", "Binary Inv", "Otsu", "Adaptive"],
            self.params["threshold_mode"], self._on_threshold_mode_change,
            tooltip="How pixels are split into foreground/background.\nOtsu picks the cutoff automatically.")
        self._add_sliders(tab, config.THRESHOLD_SLIDERS)
        ctk.CTkLabel(
            tab, justify="left", text_color="gray60", font=ctk.CTkFont(size=11),
            text=("Note: 'Otsu' ignores Threshold Value; 'Adaptive'\n"
                  "ignores Threshold Value but uses Block Size and C."),
        ).pack(anchor="w", padx=10, pady=(10, 0))

    def _build_morph_tab(self, tab):
        self._add_dropdown(
            tab, "morph_op", "Morphological Operation",
            ["None", "Opening", "Closing", "Erosion", "Dilation"],
            self.params["morph_op"], self._on_morph_op_change,
            tooltip="Cleans up the binary mask: Opening removes small\nspecks, Closing fills small holes.")
        self._add_sliders(tab, config.MORPH_SLIDERS)

    def _build_watershed_tab(self, tab):
        ctk.CTkLabel(tab, text="Watershed Segmentation",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(12, 2))
        self._add_sliders(tab, config.WATERSHED_SLIDERS)
        self._add_dropdown(
            tab, "mask_size", "Distance Transform Mask Size", ["3", "5"],
            self.params["mask_size"], self._on_mask_size_change,
            tooltip="Neighborhood size used by the distance transform\n(OpenCV only supports 3 or 5 for this metric).")

    # ---- dropdown callbacks (CTkOptionMenu passes a plain string) ----
    def _on_blur_type_change(self, value):
        self._set_param("blur_type", value)

    def _on_threshold_mode_change(self, value):
        self._set_param("threshold_mode", value)

    def _on_morph_op_change(self, value):
        self._set_param("morph_op", value)

    def _on_mask_size_change(self, value):
        self._set_param("mask_size", int(value))

    def _set_param(self, key, value):
        self.params[key] = value
        self._schedule_update()

    # ======================================================================
    # Right side: live preview (thumbnails + embedded matplotlib plot)
    # ======================================================================
    def _build_preview_area(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=1, column=1, sticky="nswe", padx=(5, 10), pady=10)
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        strip = ctk.CTkFrame(container, corner_radius=10)
        strip.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        strip.grid_columnconfigure((0, 1), weight=1)

        self.thumb_preprocessed = ThumbnailPanel(strip, "Preprocessed", config.THUMB_SIZE, 180)
        self.thumb_preprocessed.grid(row=0, column=0, sticky="nswe", padx=8, pady=8)

        self.thumb_binary = ThumbnailPanel(strip, "Threshold + Morphology (Binary Mask)", config.THUMB_SIZE, 180)
        self.thumb_binary.grid(row=0, column=1, sticky="nswe", padx=8, pady=8)

        plot_frame = ctk.CTkFrame(container, corner_radius=10)
        plot_frame.grid(row=1, column=0, sticky="nswe")

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(9, 4.5))
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=(0, 5))

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, height=32, corner_radius=0)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.status_label = ctk.CTkLabel(bar, text="Load an image to get started.", anchor="w")
        self.status_label.pack(side="left", padx=12, pady=4)

        self.progress_bar = ctk.CTkProgressBar(bar, width=120, mode="indeterminate")
        # not packed yet - only shown while a background computation is running

        self.timing_label = ctk.CTkLabel(bar, text="", anchor="e")
        self.timing_label.pack(side="right", padx=12, pady=4)

    # ======================================================================
    # Toolbar / menu actions
    # ======================================================================
    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an image", filetypes=config.SUPPORTED_IMAGE_TYPES)
        if not filepath:
            return
        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Unsupported or corrupted image file.")
        except Exception as exc:
            logger.exception("Failed to load image: %s", filepath)
            messagebox.showerror("Load Error", f"Could not load that image:\n{exc}")
            return

        self.original_full = image
        self.preview_source = proc.resize_max_dim(image, config.PREVIEW_MAX_DIM)

        # A newly loaded image can have different dimensions than the
        # previous one - reset the matplotlib artists so they get
        # recreated (with correct axis limits) on the next update instead
        # of reusing stale ones sized for the old image.
        self._im_left = None
        self._im_right = None
        self.ax_left.clear()
        self.ax_right.clear()

        h, w = image.shape[:2]
        self.status_label.configure(
            text=f"Loaded image ({w}x{h}). Adjust any parameter to update the preview live.")
        logger.info("Loaded image: %s (%dx%d)", filepath, w, h)
        self._schedule_update(immediate=True)

    def reset_params(self):
        self.params = config.default_params()
        for key, widget in self.slider_widgets.items():
            widget.set_silent(self.params[key])
        for key, widget in self.dropdown_widgets.items():
            widget.set(str(self.params[key]))
        logger.info("Parameters reset to defaults.")
        self._schedule_update(immediate=True)

    def save_result(self):
        if self.original_full is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save result as", defaultextension=".png",
            filetypes=config.SUPPORTED_SAVE_TYPES)
        if not filepath:
            return

        self.status_label.configure(text="Rendering full-resolution result...")
        self._show_progress(True)

        params_snapshot = dict(self.params)
        image_snapshot = self.original_full

        def worker():
            try:
                preprocessed = self._run_preprocessing(image_snapshot, params_snapshot)
                binary = self._run_threshold_morphology(preprocessed, params_snapshot)
                overlay, _markers, _n = proc.run_watershed(
                    image_snapshot, binary,
                    dilate_iterations=params_snapshot["dilate_iterations"],
                    fg_ratio=params_snapshot["fg_ratio"],
                    mask_size=params_snapshot["mask_size"],
                    boundary_thickness=params_snapshot["boundary_thickness"])
                cv2.imwrite(filepath, overlay)
                self.after(0, lambda: self._on_save_complete(filepath))
            except Exception as exc:
                logger.exception("Failed to save result.")
                self.after(0, lambda: self._on_save_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_save_complete(self, filepath):
        self._show_progress(False)
        self.status_label.configure(text=f"Saved full-resolution result to {filepath}")
        logger.info("Saved result to %s", filepath)

    def _on_save_failed(self, exc):
        self._show_progress(False)
        self.status_label.configure(text="Save failed.")
        messagebox.showerror("Save Error", f"Could not save the result:\n{exc}")

    def save_configuration(self):
        filepath = filedialog.asksaveasfilename(
            title="Save configuration as", defaultextension=".json",
            filetypes=config.SUPPORTED_CONFIG_TYPES)
        if not filepath:
            return
        try:
            config.save_params_to_file(filepath, self.params)
            self.status_label.configure(text=f"Configuration saved to {filepath}")
            logger.info("Saved configuration to %s", filepath)
        except Exception as exc:
            logger.exception("Failed to save configuration.")
            messagebox.showerror("Save Error", f"Could not save configuration:\n{exc}")

    def load_configuration(self):
        filepath = filedialog.askopenfilename(
            title="Load configuration", filetypes=config.SUPPORTED_CONFIG_TYPES)
        if not filepath:
            return
        try:
            self.params = config.load_params_from_file(filepath)
        except Exception as exc:
            logger.exception("Failed to load configuration.")
            messagebox.showerror("Load Error", f"Could not load configuration:\n{exc}")
            return

        for key, widget in self.slider_widgets.items():
            widget.set_silent(self.params[key])
        for key, widget in self.dropdown_widgets.items():
            widget.set(str(self.params[key]))

        self.status_label.configure(text=f"Configuration loaded from {filepath}")
        logger.info("Loaded configuration from %s", filepath)
        self._schedule_update(immediate=True)

    def toggle_appearance(self):
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)

    def show_about(self):
        top = ctk.CTkToplevel(self)
        top.title("About")
        top.geometry("380x240")
        top.resizable(False, False)
        top.transient(self)

        ctk.CTkLabel(top, text=config.APP_NAME, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(24, 4))
        ctk.CTkLabel(top, text=f"Version {config.APP_VERSION}").pack()
        ctk.CTkLabel(top, text=config.APP_AUTHOR, text_color="gray60").pack(pady=(4, 14))
        ctk.CTkLabel(
            top, justify="center",
            text="A real-time watershed image segmentation tool.\n"
                 "Adjust parameters in any tab and watch the\n"
                 "segmentation update immediately.",
        ).pack(pady=(0, 16))
        ctk.CTkButton(top, text="Close", command=top.destroy, width=100).pack()

    def on_close(self):
        logger.info("Shutting down.")
        plt.close(self.fig)
        self.destroy()

    # ======================================================================
    # Pipeline helpers (shared by the live preview and the full-res save)
    # ======================================================================
    @staticmethod
    def _run_preprocessing(image, params):
        out = proc.apply_blur(image, params["blur_type"], params["blur_ksize"])
        out = proc.adjust_brightness_contrast(out, params["brightness"], params["contrast"])
        return out

    @staticmethod
    def _run_threshold_morphology(image, params):
        out = proc.apply_threshold(
            image, params["threshold_mode"], thresh=params["thresh"],
            maxval=255, block_size=params["block_size"], c=params["c"])
        out = proc.apply_morphology(
            out, params["morph_op"], ksize=params["morph_ksize"], iterations=params["morph_iter"])
        return out

    # ======================================================================
    # Debounced, background-threaded live update
    # ======================================================================
    def _schedule_update(self, immediate: bool = False):
        """Debounce parameter changes: wait config.DEBOUNCE_MS after the
        last change before actually recomputing, so a slider being
        dragged doesn't spawn a flood of redundant background threads."""
        if self.preview_source is None:
            return
        if self._debounce_job is not None:
            self.after_cancel(self._debounce_job)
            self._debounce_job = None

        if immediate:
            self._trigger_processing()
        else:
            self._debounce_job = self.after(config.DEBOUNCE_MS, self._trigger_processing)

    def _trigger_processing(self):
        self._debounce_job = None
        if self.preview_source is None:
            return

        self._request_id += 1
        request_id = self._request_id
        params_snapshot = dict(self.params)
        image_snapshot = self.preview_source

        self._show_progress(True)

        threading.Thread(
            target=self._worker, args=(request_id, image_snapshot, params_snapshot),
            daemon=True).start()

    def _worker(self, request_id, image, params):
        """Runs on a background thread - pure OpenCV/NumPy work only, no
        Tkinter or matplotlib calls (those aren't thread-safe)."""
        try:
            start = time.perf_counter()
            preprocessed = self._run_preprocessing(image, params)
            binary = self._run_threshold_morphology(preprocessed, params)
            overlay, markers, num_regions = proc.run_watershed(
                image, binary,
                dilate_iterations=params["dilate_iterations"],
                fg_ratio=params["fg_ratio"],
                mask_size=params["mask_size"],
                boundary_thickness=params["boundary_thickness"])
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result = {
                "preprocessed": preprocessed, "binary": binary, "markers": markers,
                "num_regions": num_regions, "elapsed_ms": elapsed_ms, "source_rgb_image": image,
            }
            self.after(0, lambda: self._apply_result(request_id, result))
        except Exception as exc:
            logger.exception("Processing failed.")
            self.after(0, lambda: self._handle_processing_error(request_id, exc))

    def _apply_result(self, request_id, result):
        if request_id != self._request_id:
            return  # a newer request has since been issued - discard this stale result
        self._show_progress(False)

        ctk_pre = cv2_to_ctk_image(result["preprocessed"], config.THUMB_SIZE)
        self.thumb_preprocessed.update_image(ctk_pre)

        ctk_bin = cv2_to_ctk_image(result["binary"], config.THUMB_SIZE)
        self.thumb_binary.update_image(ctk_bin)

        markers = result["markers"]
        rgb = cv2.cvtColor(result["source_rgb_image"], cv2.COLOR_BGR2RGB)

        if self._im_left is None:
            self._im_left = self.ax_left.imshow(markers, cmap="viridis")
            self._im_right = self.ax_right.imshow(rgb)
            self.ax_right.set_title("Original Image")
        else:
            self._im_left.set_data(markers)
            self._im_left.set_clim(markers.min(), markers.max())
            self._im_right.set_data(rgb)

        self.ax_left.set_title(f"Segmented Regions ({result['num_regions']})")
        self.canvas.draw_idle()

        self.status_label.configure(text=f"Watershed found {result['num_regions']} region(s).")
        self.timing_label.configure(text=f"Processing time: {result['elapsed_ms']:.2f} ms")

    def _handle_processing_error(self, request_id, exc):
        if request_id != self._request_id:
            return
        self._show_progress(False)
        self.status_label.configure(text="Processing failed - see console for details.")
        messagebox.showerror("Processing Error", f"An error occurred while processing:\n{exc}")

    def _show_progress(self, active: bool):
        """Reference-counted so that if two operations overlap (e.g. a
        Save while a live update is still in flight), the bar only
        disappears once every operation has finished."""
        if active:
            self._active_ops += 1
            if self._active_ops == 1:
                self.progress_bar.pack(side="left", padx=(0, 12), pady=4)
                self.progress_bar.start()
        else:
            self._active_ops = max(0, self._active_ops - 1)
            if self._active_ops == 0:
                self.progress_bar.stop()
                self.progress_bar.pack_forget()


def run_app():
    ctk.set_appearance_mode(config.DEFAULT_APPEARANCE)
    ctk.set_default_color_theme(config.DEFAULT_COLOR_THEME)
    app = App()
    app.mainloop()
