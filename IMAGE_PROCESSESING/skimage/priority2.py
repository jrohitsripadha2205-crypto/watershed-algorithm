import os
import json
import cv2
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt  
import matplotlib
import threading  

# --- ENGINE IMPORTS ---
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from scipy import ndimage

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- HIGH-PRECISION UNIVERSAL ENGINE ---
def process_universal_watershed(image_path, max_dim, blur_k, thresh_offset, morph_it, Use_Gradient, fg_thresh, min_dist, compactness, connectivity):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None
    
    h, w = img.shape[:2]
    if max_dim is not None and max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_disp = img.copy()

    gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)
    
    blur_k = int(blur_k)
    if blur_k % 2 == 0:
        blur_k = max(1, blur_k - 1)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    otsu_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if thresh_offset != 0:
        adjusted_val = np.clip(otsu_val + thresh_offset, 1, 254)
        _, binary = cv2.threshold(blurred, adjusted_val, 255, cv2.THRESH_BINARY_INV)
        
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    selem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, selem, iterations=int(morph_it))
    
    if Use_Gradient > 0:
        gradient_surface = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        flooding_surface = gradient_surface.astype(np.float32)
        peak_surface = cv2.bitwise_not(gradient_surface)
        peak_surface[cleaned_mask == 0] = 0
    else:
        distance_map = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        flooding_surface = -distance_map
        peak_surface = distance_map.copy()
        max_dist_val = distance_map.max()
        if max_dist_val > 0:
            peak_surface[distance_map < (fg_thresh * max_dist_val)] = 0

    coords = peak_local_max(peak_surface, min_distance=int(min_dist), labels=cleaned_mask)
    
    if len(coords) == 0:
        markers = (cleaned_mask > 0).astype(np.int32)
    else:
        peak_mask = np.zeros(cleaned_mask.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(peak_mask)
        
    conn_footprint = disk(int(connectivity)) if connectivity > 1 else None
    segmentation_labels = watershed(flooding_surface, markers, mask=cleaned_mask, 
                                    compactness=float(compactness), connectivity=conn_footprint)
    
    display_landscape = flooding_surface if Use_Gradient > 0 else distance_map
    
    return img_disp, cleaned_mask, display_landscape, segmentation_labels


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


# --- ULTIMATE FAST WORKSTATION ARCHITECTURE ---
class UltimateWatershedWorkstation(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Universal Watershed Prototyping Workstation")
        self.geometry("1550x880")
        self.image_path = None
        self.max_display_dim = 400  
        self.updating_ui = False  

        self.setup_ui_architecture()

    def setup_ui_architecture(self):
        top_bar = ctk.CTkFrame(self, height=60, corner_radius=0)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        load_btn = ctk.CTkButton(top_bar, text="📂 Open Image File", command=self.load_image_action, 
                                 font=('Helvetica', 13, 'bold'), fg_color="#0288d1", hover_color="#01579b")
        load_btn.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.path_lbl = ctk.CTkLabel(top_bar, text="No source image loaded.", font=('Helvetica', 12, 'italic'), text_color="#90a4ae")
        self.path_lbl.pack(side=tk.LEFT, padx=5)

        self.viewport_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e1e1e")
        self.viewport_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        for i in range(4):
            self.viewport_frame.columnconfigure(i, weight=1, uniform="viewport")
        self.viewport_frame.rowconfigure(1, weight=1)
        
        titles = ["1. Raw Input View", "2. Clean Binary Mask", "3. Topology Surface Map", "4. Watershed Segments"]
        for i, t in enumerate(titles):
            lbl = ctk.CTkLabel(self.viewport_frame, text=t, font=('Helvetica', 12, 'bold'), text_color="white")
            lbl.grid(row=0, column=i, pady=(10, 5), sticky="ew")

        self.lbl_view1 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view1.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.lbl_view2 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view2.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        self.lbl_view3 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view3.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        
        self.lbl_view4 = ctk.CTkLabel(self.viewport_frame, text="Awaiting Data...", text_color="#616161")
        self.lbl_view4.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")

        # Lower Tabbed Parameter Control Panel
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
        
        self.export_btn = ctk.CTkButton(right_bar, text="💾 Export Project\n(Images & JSON)", command=self.start_async_export,
                                   font=('Helvetica', 13, 'bold'), fg_color="#2e7d32", hover_color="#1b5e20")
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
            # FIXED: Added the explicit processing trigger call here
            self.updating_ui = False
            self.trigger_pipeline_refresh()
            return
        else:
            messagebox.showwarning("Out of Bounds", f"Value must be between {from_val} and {to_val}")
        self.updating_ui = False

    def build_tabs_layout(self):
        self.v_blur, self.s_blur = self.create_parameter_row(self.tab_prep, "Gaussian Blur Smoothing Filter Kernel (Odd):", 1, 31, 5, 0, resolution=2)
        self.v_thresh, self.s_thresh = self.create_parameter_row(self.tab_prep, "Otsu Dynamic Threshold Offset Adjustment:", -120, 120, 14, 1, resolution=1)
        self.v_morph, self.s_morph = self.create_parameter_row(self.tab_prep, "Morphological Cleanup Operations (Iterations):", 0, 15, 3, 2, resolution=1)
        
        ctk.CTkLabel(self.tab_dist, text="Target Watershed Topography Surface Style:", font=('Helvetica', 12, 'bold'), text_color='white').grid(row=0, column=0, sticky='w', pady=8, padx=10)
        self.v_surface_mode = tk.IntVar(value=0)
        self.rb_dist = ctk.CTkRadioButton(self.tab_dist, text="Distance Transform (Valleys)", variable=self.v_surface_mode, value=0, command=self.trigger_pipeline_refresh)
        self.rb_dist.grid(row=0, column=1, sticky='w', padx=10)
        self.rb_grad = ctk.CTkRadioButton(self.tab_dist, text="Edge Morphological Gradient (Ridges)", variable=self.v_surface_mode, value=1, command=self.trigger_pipeline_refresh)
        self.rb_grad.grid(row=0, column=2, sticky='w', padx=10)
        
        self.v_fgt, self.s_fgt = self.create_parameter_row(self.tab_dist, "Foreground Peak Detection Cutoff Threshold Ratio:", 0.0, 0.95, 0.30, 1, is_float=True, resolution=0.05)

        self.v_mdist, self.s_mdist = self.create_parameter_row(self.tab_ws, "Peak Marker Minimum Separation Distance Limit:", 1, 100, 4, 0, resolution=1)
        self.v_comp, self.s_comp = self.create_parameter_row(self.tab_ws, "Watershed Compactness Regularization (Shape Factor):", 0.0, 5.0, 0.00, 1, is_float=True, resolution=0.05)
        self.v_conn, self.s_conn = self.create_parameter_row(self.tab_ws, "Engine Connectivity Footprint Neighborhood Metric (Radius):", 1, 10, 1, 2, resolution=1)

    def load_image_action(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Image Formats", "*.jpg *.jpeg *.png *.bmp *.tiff *.pgm")])
        if file_path:
            self.image_path = file_path
            self.path_lbl.configure(text=os.path.basename(file_path))
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
        
        disp_img, mask, topology, labels = process_universal_watershed(
            self.image_path, self.max_display_dim, blur, offset,
            morph, surf_mode, fgt, mdist, comp, conn
        )
        
        if disp_img is not None:
            self.update_native_viewports(disp_img, mask, topology, labels, surf_mode)

    def convert_to_ctk_image(self, cv_img, is_gray=False, use_cmap=None, labels_mask=None, orig_mask=None):
        if use_cmap == "jet" and labels_mask is not None:
            boundaries = watershed(cv_img if cv_img.ndim==2 else -cv_img, labels_mask, mask=orig_mask, watershed_line=True) == 0
            norm = plt.Normalize(vmin=labels_mask.min(), vmax=labels_mask.max())
            color_mapped = (plt.cm.jet(norm(labels_mask))[:, :, :3] * 255).astype(np.uint8)
            color_mapped[boundaries] = [255, 0, 0] 
            rgb = color_mapped
        elif use_cmap == "viridis" or use_cmap == "magma":
            norm_topo = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cmap = matplotlib.colormaps[use_cmap]
            rgb = (cmap(norm_topo)[:, :, :3] * 255).astype(np.uint8)
        elif is_gray:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        h, w = rgb.shape[:2]
        pil_img = Image.fromarray(rgb)
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))

    def update_native_viewports(self, disp_img, mask, topology, labels, surf_mode):
        ctk_img1 = self.convert_to_ctk_image(disp_img, is_gray=False)
        ctk_img2 = self.convert_to_ctk_image(mask, is_gray=True)
        
        cmap_choice = 'magma' if surf_mode > 0 else 'viridis'
        ctk_img3 = self.convert_to_ctk_image(topology, use_cmap=cmap_choice)
        ctk_img4 = self.convert_to_ctk_image(topology, use_cmap="jet", labels_mask=labels, orig_mask=mask)
        
        self.lbl_view1.configure(image=ctk_img1, text="")
        self.lbl_view1.image = ctk_img1
        self.lbl_view2.configure(image=ctk_img2, text="")
        self.lbl_view2.image = ctk_img2
        self.lbl_view3.configure(image=ctk_img3, text="")
        self.lbl_view3.image = ctk_img3
        self.lbl_view4.configure(image=ctk_img4, text="")
        self.lbl_view4.image = ctk_img4

    # --- MULTI-THREADED ASYNC EXPORT ENGINE WITH EXPLICIT PATH CHOICE ---
    def start_async_export(self):
        if not self.image_path:
            messagebox.showwarning("Export Error", "Please parse a target input picture before logging parameters.")
            return
        
        # FIXED: Explicitly prompts the user where to drop the output folder folder via a Directory Selector Window
        selected_parent_dir = filedialog.askdirectory(title="Select Destination Folder to Save Output Directory")
        if not selected_parent_dir:
            return # User canceled folder selection loop
            
        self.export_btn.configure(state="disabled", text="⏳ Processing High-Res...\n(GUI Active)")
        
        export_thread = threading.Thread(target=self.background_export_worker, args=(selected_parent_dir,))
        export_thread.daemon = True
        export_thread.start()

    def background_export_worker(self, target_base_path):
        try:
            base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
            # Formulates directory exactly inside your desired chosen download or desktop folder path location
            output_dir = os.path.join(target_base_path, f"watershed_{base_filename}_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            blur = int(self.s_blur.get_value())
            offset = int(self.s_thresh.get_value())
            morph = int(self.s_morph.get_value())
            surf_mode = int(self.v_surface_mode.get())
            fgt = float(self.s_fgt.get_value())
            mdist = int(self.s_mdist.get_value())
            comp = float(self.s_comp.get_value())
            conn = int(self.s_conn.get_value())
            
            _, full_mask, full_topo, full_labels = process_universal_watershed(
                self.image_path, None, blur, offset, morph, surf_mode, fgt, mdist, comp, conn
            )
            
            export_config = {
                "algorithm_metadata": "hybrid_opencv_skimage_universal_workstation_pipeline",
                "source_file_reference": os.path.basename(self.image_path),
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
                
            cv2.imwrite(os.path.join(output_dir, "export_1_input.png"), cv2.imread(self.image_path))
            cv2.imwrite(os.path.join(output_dir, "export_2_binary_mask.png"), full_mask)
            
            topo_norm = cv2.normalize(full_topo, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cmap = matplotlib.colormaps['magma' if surf_mode > 0 else 'viridis']
            cv2.imwrite(os.path.join(output_dir, "export_3_topography.png"), (cmap(topo_norm)[:, :, :3] * 255).astype(np.uint8))
            
            boundaries = watershed(full_topo if surf_mode > 0 else -full_topo, full_labels, mask=full_mask, watershed_line=True) == 0
            norm = plt.Normalize(vmin=full_labels.min(), vmax=full_labels.max())
            color_labels = (plt.cm.jet(norm(full_labels))[:, :, :3] * 255).astype(np.uint8)
            color_labels[boundaries] = [255, 0, 0]
            cv2.imwrite(os.path.join(output_dir, "export_4_segmentation.png"), cv2.cvtColor(color_labels, cv2.COLOR_RGB2BGR))
            
            self.after(0, lambda: self.on_export_complete(output_dir))
            
        except Exception as e:
            self.after(0, lambda: self.on_export_failed(str(e)))

    def on_export_complete(self, output_dir):
        self.export_btn.configure(state="normal", text="💾 Export Project\n(Images & JSON)")
        messagebox.showinfo("Export Successful", f"All full-resolution assets logged inside target structure folder:\n{output_dir}")

    def on_export_failed(self, error_msg):
        self.export_btn.configure(state="normal", text="💾 Export Project\n(Images & JSON)")
        messagebox.showerror("Export Error", f"An unexpected processing error occurred:\n{error_msg}")


if __name__ == "__main__":
    app = UltimateWatershedWorkstation()
    app.mainloop()