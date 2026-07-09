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

def run_watershed_pipeline(image_path, min_distance, compactness):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # 1. OpenCV Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
    # 2. skimage Marker Detection using custom min_distance
    coordinates = peak_local_max(dist_transform, min_distance=int(min_distance), labels=clean_mask)
    
    # Handle case where no peaks are found to avoid code crash
    if len(coordinates) == 0:
        return img, np.zeros(dist_transform.shape, dtype=int)
        
    peak_mask = np.zeros(dist_transform.shape, dtype=bool)
    peak_mask[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(peak_mask)
    
    # 3. skimage Watershed Segmentation using custom compactness
    final_labels = watershed(-dist_transform, markers, mask=clean_mask, compactness=float(compactness))
    return img, final_labels


# --- INTERACTIVE PARAMETER TUNING GUI ---
class InteractiveWatershedApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Interactive Watershed Parameter Tuner")
        self.image_path = image_path
        
        # Initial starting parameters
        self.current_min_dist = 15
        self.current_compactness = 0.1
        
        # Run initial pipeline execution
        self.orig_img, self.labels = run_watershed_pipeline(
            self.image_path, self.current_min_dist, self.current_compactness
        )
        
        if self.orig_img is None:
            messagebox.showerror("Error", f"Could not open image: {image_path}")
            self.root.destroy()
            return

        self.build_layout()

    def build_layout(self):
        # 1. Top Section: Matplotlib Plot Canvas
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4.5))
        self.render_plots()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 2. Bottom Section: Parameter Controls Configuration Panel
        control_frame = tk.LabelFrame(self.root, text=" Parameter Tuning Panel ", font=('Helvetica', 10, 'bold'), padx=15, pady=10)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        # Slider for min_distance (Integer values)
        tk.Label(control_frame, text="Peak Min Distance (Object Separation):", font=('Helvetica', 10)).grid(row=0, column=0, sticky='w', pady=5)
        self.dist_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.dist_slider.set(self.current_min_dist)
        self.dist_slider.grid(row=0, column=1,sticky="ew", padx=10)
        
        # Slider for compactness (Float values handled by scale increments)
        tk.Label(control_frame, text="Compactness (Boundary Regularity):", font=('Helvetica', 10)).grid(row=1, column=0, sticky='w', pady=5)
        self.compact_slider = tk.Scale(control_frame, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, command=self.update_pipeline)
        self.compact_slider.set(self.current_compactness)
        self.compact_slider.grid(row=1, column=1, sticky="ew", padx=10)
        
        # Functional Save Configuration Button
        save_button = tk.Button(
            control_frame, 
            text="Save Current Configuration (JSON)", 
            command=self.save_parameters_json, 
            bg="#0d47a1", # Deep Tech Blue
            fg="white", 
            font=('Helvetica', 11, 'bold'),
            padx=15,
            pady=6
        )
        save_button.grid(row=2, column=0, columnspan=2, pady=15)

    def render_plots(self):
        """ Clears and redraws subplots based on current segmentations """
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Original Image
        self.axes[0].imshow(cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB))
        self.axes[0].set_title("Input Image")
        self.axes[0].axis('off')
        
        # Dynamic Segmented Label Map
        self.axes[1].imshow(self.labels, cmap=plt.cm.jet)
        self.axes[1].set_title(f"Segmentation (Dist: {self.current_min_dist}, Compact: {self.current_compactness})")
        self.axes[1].axis('off')

    def update_pipeline(self, event=None):
        """ Triggered automatically whenever a slider is moved """
        # Get active slider values
        self.current_min_dist = int(self.dist_slider.get())
        self.current_compactness = float(self.compact_slider.get())
        
        # Re-run pipeline calculation asynchronously on current image state
        _, self.labels = run_watershed_pipeline(
            self.image_path, self.current_min_dist, self.current_compactness
        )
        
        # Redraw matplotlib canvas plot view
        self.render_plots()
        self.canvas.draw()

    def save_parameters_json(self):
        """ Saves exact runtime configurations and asset pipeline files to workspace """
        output_dir = "watershed_project_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Format exact dictionary capturing current values configuration
        config_data = {
            "algorithm": "skimage_watershed_interactive",
            "final_min_distance": self.current_min_dist,
            "final_compactness": self.current_compactness,
            "preprocessing": "Gaussian Blur (5x5) + Morphological Opening",
            "marker_engine": "skimage.feature.peak_local_max"
        }
        
        # 1. Write structured JSON file
        json_path = os.path.join(output_dir, "parameters.json")
        with open(json_path, "w") as json_file:
            json.dump(config_data, json_file, indent=4)
            
        # 2. Write verification image outputs directly
        cv2.imwrite(os.path.join(output_dir, "input_image.png"), self.orig_img)
        labels_norm = cv2.normalize(self.labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(output_dir, "output_segmentation.png"), cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET))
        
        messagebox.showinfo("Saved Successfully", f"Current optimal parameters logged inside JSON file!\nLocation: {json_path}")


if __name__ == "__main__":
    root = tk.Tk()
    # Replace 'coins.jpg' with your target testing image path 
    app = InteractiveWatershedApp(root, image_path=r"C:\Users\user\Downloads\coins.png")
    root.mainloop()