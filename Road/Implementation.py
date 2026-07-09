import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox

# --- CORE PROCESSING PIPELINE ---
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

def run_advanced_pipeline(image_path, blur_kernel, thresh_offset, morph_iter, dist_mask, min_distance, compactness):
    print("Current folder:", os.getcwd())
    print("Script folder:", os.path.dirname(__file__))
    print("Image path:", image_path)
    print("Image exists:", os.path.exists(image_path))
    img = cv2.imread(image_path)
    print("Image loaded:", img is not None)
    if img is None:
        return None, None , None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Dynamic Gaussian Blur (Kernel size must be odd)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # 2. Thresholding with Offset Modifier
    otsu_thresh, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if thresh_offset != 0:
        # Adjust the threshold calculated by Otsu
        _, thresh_img = cv2.threshold(blurred, otsu_thresh + thresh_offset, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Morphological Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=int(morph_iter))
    
    # 4. Distance Transform with dynamic mask size (3 or 5)
    dist_mask_param = 5 if dist_mask >= 4 else 3
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, dist_mask_param)
    
    # 5. Marker Generation
    coordinates = peak_local_max(dist_transform, min_distance=int(min_distance), labels=clean_mask)
    
    if len(coordinates) == 0:
        return img, np.zeros(dist_transform.shape, dtype=int), clean_mask
        
    peak_mask = np.zeros(dist_transform.shape, dtype=bool)
    peak_mask[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(peak_mask)
    
    # 6. Watershed Segmentation
    final_labels = watershed(-dist_transform, markers, mask=clean_mask, compactness=float(compactness))
    return img, final_labels, clean_mask


# --- INTERACTIVE PARAMETER TUNING GUI ---
class AdvancedWatershedApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Advanced End-to-End CV Pipeline Tuner")
        self.image_path = image_path
        
        # Default State Values
        self.p_blur = 5
        self.p_thresh_offset = 0
        self.p_morph_iter = 2
        self.p_dist_mask = 5
        self.p_min_dist = 15
        self.p_compactness = 0.1
        
        # Initial run
        self.orig_img, self.labels, self.binary_mask = run_advanced_pipeline(
            self.image_path, self.p_blur, self.p_thresh_offset, self.p_morph_iter, 
            self.p_dist_mask, self.p_min_dist, self.p_compactness
        )
        
        if self.orig_img is None:
            messagebox.showerror("Error", f"Could not open image: {image_path}")
            self.root.destroy()
            return

        self.build_layout()

    def build_layout(self):
        # Top Display: 3 plots now (Input, Preprocessed Binary Mask, Final Segmentation)
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 4.5))
        self.render_plots()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Bottom Control Grid Panel
        control_frame = tk.LabelFrame(self.root, text=" Global Pipeline Configuration Panel ", font=('Helvetica', 10, 'bold'), padx=15, pady=10)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        # Column 1 Sliders (Preprocessing)
        tk.Label(control_frame, text="Blur Kernel Size (Odd):").grid(row=0, column=0, sticky='w', pady=2)
        self.s_blur = tk.Scale(control_frame, from_=1, to=31, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.s_blur.set(self.p_blur)
        self.s_blur.grid(row=0, column=1, padx=10, sticky="ew")
        
        tk.Label(control_frame, text="Thresh Offset (-100 to 100):").grid(row=1, column=0, sticky='w', pady=2)
        self.s_thresh = tk.Scale(control_frame, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.s_thresh.set(self.p_thresh_offset)
        self.s_thresh.grid(row=1, column=1, padx=10, sticky="ew")
        
        tk.Label(control_frame, text="Morph Open Iterations:").grid(row=2, column=0, sticky='w', pady=2)
        self.s_morph = tk.Scale(control_frame, from_=0, to=10, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.s_morph.set(self.p_morph_iter)
        self.s_morph.grid(row=2, column=1, padx=10, sticky="ew")
        
        # Column 2 Sliders (Distance & Watershed)
        tk.Label(control_frame, text="Distance Mask Size (3 or 5):").grid(row=0, column=2, sticky='w', pady=2, padx=(20, 0))
        self.s_dist_mask = tk.Scale(control_frame, from_=3, to=5, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.s_dist_mask.set(self.p_dist_mask)
        self.s_dist_mask.grid(row=0, column=3, padx=10, sticky="ew")
        
        tk.Label(control_frame, text="Peak Min Distance:").grid(row=1, column=2, sticky='w', pady=2, padx=(20, 0))
        self.s_min_dist = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.s_min_dist.set(self.p_min_dist)
        self.s_min_dist.grid(row=1, column=3, padx=10, sticky="ew")
        
        tk.Label(control_frame, text="Compactness Factor:").grid(row=2, column=2, sticky='w', pady=2, padx=(20, 0))
        self.s_compact = tk.Scale(control_frame, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.s_compact.set(self.p_compactness)
        self.s_compact.grid(row=2, column=3, padx=10, sticky="ew")
        
        # Centered Save Button spanning across the bottom grid rows
        save_button = tk.Button(control_frame, text="Export Comprehensive Parameters Configuration to JSON", 
                                command=self.save_all_to_json, bg="#1b5e20", fg="white", font=('Helvetica', 11, 'bold'), pady=5)
        save_button.grid(row=3, column=0, columnspan=4, pady=15)

    def render_plots(self):
        for ax in self.axes:
            ax.clear()
            
        self.axes[0].imshow(cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB))
        self.axes[0].set_title("1. Input Image")
        self.axes[0].axis('off')
        
        self.axes[1].imshow(self.binary_mask, cmap='gray')
        self.axes[1].set_title("2. Preprocessed Mask")
        self.axes[1].axis('off')
        
        self.axes[2].imshow(self.labels, cmap=plt.cm.jet)
        self.axes[2].set_title("3. Watershed Segmentation")
        self.axes[2].axis('off')

    def update_pipeline(self, event=None):
        # Update state values from sliders
        self.p_blur = int(self.s_blur.get())
        self.p_thresh_offset = int(self.s_thresh.get())
        self.p_morph_iter = int(self.s_morph.get())
        self.p_dist_mask = int(self.s_dist_mask.get())
        self.p_min_dist = int(self.s_min_dist.get())
        self.p_compactness = float(self.s_compact.get())
        
        # Re-execute entire pipeline dynamically
        _, self.labels, self.binary_mask = run_advanced_pipeline(
            self.image_path, self.p_blur, self.p_thresh_offset, self.p_morph_iter, 
            self.p_dist_mask, self.p_min_dist, self.p_compactness
        )
        
        self.render_plots()
        self.canvas.draw()

    def save_all_to_json(self):
        output_dir = "watershed_project_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Package every single slider variable into the JSON configuration profile
        comprehensive_config = {
            "algorithm_metadata": "hybrid_opencv_skimage_advanced_pipeline",
            "parameters": {
                "gaussian_blur_kernel": self.p_blur if self.p_blur % 2 != 0 else self.p_blur + 1,
                "otsu_threshold_offset": self.p_thresh_offset,
                "morphological_open_iterations": self.p_morph_iter,
                "distance_transform_mask_size": 5 if self.p_dist_mask >= 4 else 3,
                "watershed_peak_min_distance": self.p_min_dist,
                "watershed_compactness": self.p_compactness
            }
        }
        
        # 1. Save detailed parameters JSON
        with open(os.path.join(output_dir, "parameters.json"), "w") as f:
            json.dump(comprehensive_config, f, indent=4)
            
        # 2. Save full visual asset trial chain
        cv2.imwrite(os.path.join(output_dir, "input_image.png"), self.orig_img)
        cv2.imwrite(os.path.join(output_dir, "preprocessed_mask.png"), self.binary_mask)
        labels_norm = cv2.normalize(self.labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(output_dir, "output_segmentation.png"), cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET))
        
        messagebox.showinfo("Pipeline Logged", f"All preprocessing and watershed variables exported successfully to '{output_dir}/parameters.json'")

if __name__ == "__main__":
    root = tk.Tk()
    # Pass your testing image here
    image_path = os.path.join(os.path.dirname(__file__), "traffic.png")
    app = AdvancedWatershedApp(root, image_path=image_path)
    root.mainloop()
    