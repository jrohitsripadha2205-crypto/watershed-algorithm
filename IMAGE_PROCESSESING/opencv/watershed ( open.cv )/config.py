"""
config.py
---------
Application-wide constants, default parameters, slider specifications
(used to build the Preprocessing / Threshold / Morphology / Watershed
tabs without repeating boilerplate), and JSON configuration save/load.

Centralizing this here means gui.py stays focused on wiring widgets
together rather than hard-coding numeric ranges throughout.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TypedDict


# --------------------------------------------------------------------------
# Application metadata
# --------------------------------------------------------------------------
APP_NAME = "Watershed Segmentation Studio"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Built with CustomTkinter, OpenCV & Matplotlib"

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 850
WINDOW_MIN_WIDTH = 1150
WINDOW_MIN_HEIGHT = 700

DEFAULT_APPEARANCE = "Dark"        # "Dark" / "Light" / "System"
DEFAULT_COLOR_THEME = "blue"

THUMB_SIZE = 260                   # max px for the intermediate-step thumbnails
PREVIEW_MAX_DIM = 500              # live preview images are downscaled to this, for speed
DEBOUNCE_MS = 120                  # wait this long after the last slider move before recomputing

SUPPORTED_IMAGE_TYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
    ("All files", "*.*"),
]
SUPPORTED_SAVE_TYPES = [
    ("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("All files", "*.*"),
]
SUPPORTED_CONFIG_TYPES = [("JSON files", "*.json"), ("All files", "*.*")]


# --------------------------------------------------------------------------
# Default processing parameters
# --------------------------------------------------------------------------
def default_params() -> Dict[str, Any]:
    """Return a fresh dict of default processing parameters. Called at
    startup and whenever the user chooses Edit > Reset Parameters."""
    return {
        "blur_type": "None", "blur_ksize": 5,
        "brightness": 0, "contrast": 1.0,
        "threshold_mode": "Otsu", "thresh": 127, "block_size": 11, "c": 2,
        "morph_op": "None", "morph_ksize": 3, "morph_iter": 1,
        "dilate_iterations": 3, "fg_ratio": 50, "mask_size": 5, "boundary_thickness": 1,
    }


# --------------------------------------------------------------------------
# Slider specifications, grouped by tab
# --------------------------------------------------------------------------
class SliderSpec(TypedDict):
    key: str
    label: str
    min: float
    max: float
    steps: int
    default: float
    is_float: bool
    tooltip: str


PREPROCESS_SLIDERS: List[SliderSpec] = [
    {"key": "blur_ksize", "label": "Blur Kernel Size", "min": 1, "max": 31, "steps": 15,
     "default": 5, "is_float": False,
     "tooltip": "Size of the blur kernel. Larger values blur more strongly.\nAutomatically snapped to the nearest odd number."},
    {"key": "brightness", "label": "Brightness", "min": -100, "max": 100, "steps": 40,
     "default": 0, "is_float": False,
     "tooltip": "Additive brightness offset applied to every pixel."},
    {"key": "contrast", "label": "Contrast", "min": 0.1, "max": 3.0, "steps": 29,
     "default": 1.0, "is_float": True,
     "tooltip": "Multiplicative contrast gain. 1.0 leaves the image unchanged."},
]

THRESHOLD_SLIDERS: List[SliderSpec] = [
    {"key": "thresh", "label": "Threshold Value", "min": 0, "max": 255, "steps": 255,
     "default": 127, "is_float": False,
     "tooltip": "Cutoff pixel intensity. Only used by Binary / Binary Inv modes."},
    {"key": "block_size", "label": "Adaptive Block Size", "min": 3, "max": 99, "steps": 48,
     "default": 11, "is_float": False,
     "tooltip": "Neighborhood size used by Adaptive thresholding.\nAutomatically snapped to the nearest odd number."},
    {"key": "c", "label": "Adaptive C", "min": -50, "max": 50, "steps": 100,
     "default": 2, "is_float": False,
     "tooltip": "Constant subtracted from the mean in Adaptive thresholding."},
]

MORPH_SLIDERS: List[SliderSpec] = [
    {"key": "morph_ksize", "label": "Kernel Size", "min": 1, "max": 31, "steps": 15,
     "default": 3, "is_float": False,
     "tooltip": "Size of the structuring element used by the morphological operation."},
    {"key": "morph_iter", "label": "Iterations", "min": 1, "max": 10, "steps": 9,
     "default": 1, "is_float": False,
     "tooltip": "How many times the operation is applied in a row."},
]

WATERSHED_SLIDERS: List[SliderSpec] = [
    {"key": "dilate_iterations", "label": "Background Dilate Iterations", "min": 1, "max": 10,
     "steps": 9, "default": 3, "is_float": False,
     "tooltip": "How far the 'sure background' region is expanded outward."},
    {"key": "fg_ratio", "label": "Foreground Threshold Ratio (%)", "min": 1, "max": 99,
     "steps": 98, "default": 50, "is_float": False,
     "tooltip": "Percent of the maximum distance-transform value used to\ndefine the 'sure foreground' region. Higher = smaller, more\nconfident foreground seeds."},
    {"key": "boundary_thickness", "label": "Boundary Thickness", "min": 1, "max": 5,
     "steps": 4, "default": 1, "is_float": False,
     "tooltip": "Thickness (in pixels) of the red boundary lines drawn on the result."},
]


# --------------------------------------------------------------------------
# JSON configuration save / load
# --------------------------------------------------------------------------
def save_params_to_file(filepath: str, params: Dict[str, Any]) -> None:
    """Write the given parameter dict out as pretty-printed JSON, tagged
    with the app name/version for forward-compatibility."""
    payload = {"app_name": APP_NAME, "app_version": APP_VERSION, "params": params}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def load_params_from_file(filepath: str) -> Dict[str, Any]:
    """Read a parameter dict back from a JSON file, filling in any keys
    missing from the file (e.g. saved by an older version) with current
    defaults so loading never crashes on a partially-outdated file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = default_params()
    params.update(data.get("params", {}))
    return params
