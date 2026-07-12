import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

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
    
    # 1. Responsive Downscaling for Preview Speed Stability
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_disp = img.copy()

    gray = cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)
    
    # Enforce odd blur kernel structural validation constraints
    blur_k = int(blur_k)
    if blur_k % 2 == 0:
        blur_k = max(1, blur_k - 1)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    # 2. Binary Extraction Loop with Robust Contrast Offset
    otsu_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if thresh_offset != 0:
        adjusted_val = np.clip(otsu_val + thresh_offset, 1, 254)
        _, binary = cv2.threshold(blurred, adjusted_val, 255, cv2.THRESH_BINARY_INV)
        
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 3. Morphological Cleanup
    selem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, selem, iterations=int(morph_it))
    
    # 4. Topology Selection (Distance Topography vs Edge Morphological Gradients)
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

    # 5. High-Resolution Peak Detection Metrics
    coords = peak_local_max(peak_surface, min_distance=int(min_dist), labels=cleaned_mask)
    
    if len(coords) == 0:
        markers = (cleaned_mask > 0).astype(np.int32)
    else:
        peak_mask = np.zeros(cleaned_mask.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(peak_mask)
        
    # 6. Watershed Engine Implementation utilizing customized connectivity footprints
    conn_footprint = disk(int(connectivity)) if connectivity > 1 else None
    segmentation_labels = watershed(flooding_surface, markers, mask=cleaned_mask, 
                                    compactness=float(compactness), connectivity=conn_footprint)
    
    display_landscape = flooding_surface if Use_Gradient > 0 else distance_map
    
    return img_disp, cleaned_mask, display_landscape, segmentation_labels


# --- COMPONENT SPINBOX (Fixed Variable Synchronization) ---
class CTkSpinbox(ctk.CTkFrame):
    def __init__(self, parent, from_val, to_val, current_val, is_float=False, resolution=1, linked_var=None, on_update_callback=None):
        super().__init__(parent, fg_color="transparent")
        self.from_val = from_val
        self.to_val = to_val
        self.is_float = is_float
        self.resolution = resolution
        self.linked_var = linked_var
        self.callback = on_update_callback
        
        # Down Button
        self.btn_down = ctk.CTkButton(self, text="▼", width=28, height=28, font=("Arial", 10), 
                                      fg_color="#37474f", hover_color="#263238", command=self.decrement)
        self.btn_down.pack(side=tk.LEFT, padx=2)
        
        # Central Data Field Entry
        self.entry = ctk.CTkEntry(self, width=65, height=28, font=('Helvetica', 12), justify='center')
        self.entry.insert(0, f"{current_val:.2f}" if self.is_float else str(int(current_val)))
        self.entry.pack(side=tk.LEFT, padx=2)
        
        # Up Button
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


# --- ULTIMATE TABBED WORKSTATION INTERFACE ---
class UltimateWatershedWorkstation(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Universal Watershed Prototyping Workstation")
        self.geometry("1500x900")
        self.image_path = None
        self.max_display_dim = 500  
        self.updating_ui = False  

        self.setup_ui_architecture()

    def setup_ui_architecture(self):
        # Top Toolbar
        top_bar = ctk.CTkFrame(self, height=60, corner_radius=0)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        load_btn = ctk.CTkButton(top_bar, text="📂 Open Image File", command=self.load_image_action, 
                                 font=('Helvetica', 13, 'bold'), fg_color="#0288d1", hover_color="#01579b")
        load_btn.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.path_lbl = ctk.CTkLabel(top_bar, text="No source image loaded.", font=('Helvetica', 12, 'italic'), text_color="#90a4ae")
        self.path_lbl.pack(side=tk.LEFT, padx=5)

        # Plot Canvas Layout Section
        self.plot_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e1e1e")
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.fig, self.axes = plt.subplots(1, 4, figsize=(16, 4))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.init_empty_plots()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Lower Tabbed Parameter Control Panel
        control_panel = ctk.CTkFrame(self, height=320)
        control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        self.tabview = ctk.CTkTabview(control_panel, height=280)
        self.tabview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tab_prep = self.tabview.add(" 1. Preprocessing & Binary Filters ")
        self.tab_dist = self.tabview.add(" 2. Topography & Distance Analytics ")
        self.tab_ws   = self.tabview.add(" 3. Watershed Engine Optimization ")
        
        self.build_tabs_layout()
        
        # Right Control Save Button Panel
        right_bar = ctk.CTkFrame(control_panel, width=240, fg_color="transparent")
        right_bar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        export_btn = ctk.CTkButton(right_bar, text="💾 Export Project\n(Images & JSON)", command=self.export_assets_action,
                                   font=('Helvetica', 13, 'bold'), fg_color="#2e7d32", hover_color="#1b5e20")
        export_btn.pack(fill=tk.BOTH, expand=True, pady=10)

    def init_empty_plots(self):
        titles = ["1. Raw Input View", "2. Clean Binary Mask", "3. Topology Surface Map", "4. Watershed Segments"]
        for ax, title in zip(self.axes, titles):
            ax.clear()
            ax.set_title(title, fontdict={'fontsize': 10, 'weight': 'bold', 'color': 'white'})
            ax.text(0.5, 0.5, "Awaiting Data...", ha='center', va='center', color='#616161')
            ax.axis('off')

    def create_parameter_row(self, parent, label_text, from_val, to_val, current_val, row_idx, is_float=False, resolution=1):
        lbl = ctk.CTkLabel(parent, text=label_text, font=('Helvetica', 12), anchor='w')
        lbl.grid(row=row_idx, column=0, sticky='w', pady=8, padx=10)
        
        var = tk.DoubleVar(value=current_val) if is_float else tk.IntVar(value=current_val)
        
        slider = ctk.CTkSlider(parent, from_=from_val, to=to_val, number_of_steps=int((to_val-from_val)/resolution), variable=var)
        slider.grid(row=row_idx, column=1, sticky='ew', padx=20, pady=8)
        
        # Inject the active tracked variable link directly into the Spinbox layout
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
            self.trigger_pipeline_refresh()
        else:
            messagebox.showwarning("Out of Bounds", f"Value must be between {from_val} and {to_val}")
        self.updating_ui = False

    def build_tabs_layout(self):
        # Tab 1: Preprocessing Elements
        self.v_blur, self.s_blur = self.create_parameter_row(self.tab_prep, "Gaussian Blur Smoothing Filter Kernel (Odd):", 1, 31, 5, 0, resolution=2)
        self.v_thresh, self.s_thresh = self.create_parameter_row(self.tab_prep, "Otsu Dynamic Threshold Offset Adjustment:", -120, 120, 14, 1, resolution=1)
        self.v_morph, self.s_morph = self.create_parameter_row(self.tab_prep, "Morphological Cleanup Operations (Iterations):", 0, 15, 3, 2, resolution=1)
        
        # Tab 2: Topography Configuration
        ctk.CTkLabel(self.tab_dist, text="Target Watershed Topography Surface Style:", font=('Helvetica', 12, 'bold'), text_color='white').grid(row=0, column=0, sticky='w', pady=8, padx=10)
        self.v_surface_mode = tk.IntVar(value=0)
        self.rb_dist = ctk.CTkRadioButton(self.tab_dist, text="Distance Transform (Valleys)", variable=self.v_surface_mode, value=0, command=self.trigger_pipeline_refresh)
        self.rb_dist.grid(row=0, column=1, sticky='w', padx=10)
        self.rb_grad = ctk.CTkRadioButton(self.tab_dist, text="Edge Morphological Gradient (Ridges)", variable=self.v_surface_mode, value=1, command=self.trigger_pipeline_refresh)
        self.rb_grad.grid(row=0, column=2, sticky='w', padx=10)
        
        self.v_fgt, self.s_fgt = self.create_parameter_row(self.tab_dist, "Foreground Peak Detection Cutoff Threshold Ratio:", 0.0, 0.95, 0.30, 1, is_float=True, resolution=0.05)

        # Tab 3: Watershed Engine Specific Tweaks
        self.v_mdist, self.s_mdist = self.create_parameter_row(self.tab_ws, "Peak Marker Minimum Separation Distance Limit:", 1, 100, 4, 0, resolution=1)
        self.v_comp, self.s_comp = self.create_parameter_row(self.tab_ws, "Watershed Compactness Regularization (Shape Factor):", 0.0, 5.0, 0.00, 1, is_float=True, resolution=0.05)
        self.v_conn, self.s_conn = self.create_parameter_row(self.tab_ws, "Engine Connectivity Footprint Neighborhood Metric (Radius):", 1, 10, 1, 2, resolution=1)

    # --- CORE PIPELINE CONTROLLERS ---
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
            self.update_plots_display(disp_img, mask, topology, labels, surf_mode)

    def update_plots_display(self, disp_img, mask, topology, labels, surf_mode):
        self.axes[0].clear()
        self.axes[0].imshow(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))
        self.axes[0].set_title("1. Raw Input View", fontsize=10, weight='bold', color='white')
        self.axes[0].axis('off')
        
        self.axes[1].clear()
        self.axes[1].imshow(mask, cmap='gray')
        self.axes[1].set_title("2. Clean Binary Mask", fontsize=10, weight='bold', color='white')
        self.axes[1].axis('off')
        
        self.axes[2].clear()
        cmap_choice = 'magma' if surf_mode > 0 else 'viridis'
        title_choice = "3. Morph Edge Gradient" if surf_mode > 0 else "3. Distance Surface Map"
        self.axes[2].imshow(topology, cmap=cmap_choice)
        self.axes[2].set_title(title_choice, fontsize=10, weight='bold', color='white')
        self.axes[2].axis('off')
        
        # Overlay crisp red watershed boundary lines directly onto the final preview color map
        self.axes[3].clear()
        # Find explicit boundary lines where neighboring segment IDs do not match
        boundaries = watershed(topology if surf_mode > 0 else -topology, labels, mask=mask, watershed_line=True) == 0
        
        # Color labels using a clean jet map palette
        color_labels = plt.cm.jet(plt.Normalize(vmin=labels.min(), vmax=labels.max())(labels))
        # Inject bright red lines to visually trace active shifting parameters in real-time
        color_labels[boundaries] = [1.0, 0.0, 0.0, 1.0] 
        
        self.axes[3].imshow(color_labels)
        self.axes[3].set_title("4. Watershed Segments", fontsize=10, weight='bold', color='white')
        self.axes[3].axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()

    def export_assets_action(self):
        if not self.image_path:
            messagebox.showwarning("Export Error", "Please parse a target input picture before logging parameters.")
            return
            
        # DYNAMIC NAMING STRATEGY IMPLEMENTATION
        # Extracts raw filename (e.g., 'coins.png' -> 'coins')
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
        output_dir = f"watershed_{base_filename}_outputs"
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
            self.image_path, float('inf'), blur, offset, morph, surf_mode, fgt, mdist, comp, conn
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
        cmap = cv2.COLORMAP_MAGMA if surf_mode > 0 else cv2.COLORMAP_VIRIDIS
        cv2.imwrite(os.path.join(output_dir, "export_3_topography.png"), cv2.applyColorMap(topo_norm, cmap))
        
        labels_norm = cv2.normalize(full_labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(output_dir, "export_4_segmentation.png"), cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET))
        
        messagebox.showinfo("Success", f"All full-resolution assets logged inside target structure folder:\n'{output_dir}/'")

if __name__ == "__main__":
    app = UltimateWatershedWorkstation()
    app.mainloop()