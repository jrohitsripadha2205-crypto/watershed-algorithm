"""
processor.py
------------
Pure image-processing functions used by Watershed Segmentation Studio.
No GUI code lives here - every function just takes an image (and some
parameters) in, and returns an image (or a small result tuple) out. That
keeps this module easy to reason about, unit-testable, and reusable
outside the GUI (e.g. from a script or a test suite).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def ensure_odd(value: float) -> int:
    """OpenCV kernel sizes must be odd and >= 1."""
    value = int(round(value))
    if value < 1:
        value = 1
    if value % 2 == 0:
        value += 1
    return value


def to_grayscale(image: Image) -> Image:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: Image, blur_type: str, ksize: float, sigma: float = 0) -> Image:
    """blur_type is one of: 'None', 'Gaussian', 'Median', 'Bilateral'."""
    if blur_type == "None" or ksize <= 1:
        return image
    k = ensure_odd(ksize)
    if blur_type == "Gaussian":
        return cv2.GaussianBlur(image, (k, k), sigma)
    if blur_type == "Median":
        return cv2.medianBlur(image, k)
    if blur_type == "Bilateral":
        return cv2.bilateralFilter(image, k, 75, 75)
    return image


def adjust_brightness_contrast(image: Image, brightness: float = 0, contrast: float = 1.0) -> Image:
    """output = image * contrast + brightness"""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def apply_threshold(image: Image, mode: str, thresh: float = 127, maxval: float = 255,
                     block_size: float = 11, c: float = 2) -> Image:
    """mode is one of: 'Binary', 'Binary Inv', 'Otsu', 'Adaptive'."""
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mode == "Binary":
        _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
    elif mode == "Binary Inv":
        _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)
    elif mode == "Otsu":
        _, out = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "Adaptive":
        block_size = max(3, ensure_odd(block_size))
        out = cv2.adaptiveThreshold(
            gray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c)
    else:
        out = gray
    return out


def apply_morphology(image: Image, op: str, ksize: float = 3, iterations: int = 1) -> Image:
    """op is one of: 'None', 'Opening', 'Closing', 'Erosion', 'Dilation'."""
    if op == "None" or ksize <= 1:
        return image
    k = ensure_odd(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if op == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if op == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if op == "Erosion":
        return cv2.erode(image, kernel, iterations=iterations)
    if op == "Dilation":
        return cv2.dilate(image, kernel, iterations=iterations)
    return image


def run_watershed(original_bgr: Image, binary_image: Image, dilate_iterations: int = 3,
                   fg_ratio: float = 50, mask_size: int = 5,
                   boundary_thickness: int = 1) -> Tuple[Image, NDArray[np.int32], int]:
    """
    Classic marker-based watershed segmentation.

    `binary_image` (an 0/255 mask, e.g. straight from apply_threshold /
    apply_morphology) is used to derive the sure-foreground / sure-background
    regions; the resulting boundaries are drawn onto `original_bgr`.

    Returns:
        overlay      - original_bgr with segment boundaries drawn in red
        markers      - the raw labeled-region array from cv2.watershed
                        (background/foreground blobs each get their own
                        integer label; boundaries are -1)
        num_regions  - how many distinct foreground regions were found
    """
    gray_bin = binary_image
    if len(gray_bin.shape) != 2:
        gray_bin = cv2.cvtColor(gray_bin, cv2.COLOR_BGR2GRAY)
    _, gray_bin = cv2.threshold(gray_bin, 127, 255, cv2.THRESH_BINARY)
    gray_bin = gray_bin.astype(np.uint8)

    # Clean up small noise specks, then build sure-background / sure-foreground
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(gray_bin, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=int(dilate_iterations))

    mask_size = mask_size if mask_size in (3, 5) else 5
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, mask_size)
    max_val = dist.max() if dist.max() > 0 else 1.0
    _, sure_fg = cv2.threshold(dist, (fg_ratio / 100.0) * max_val, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # background isn't 0 (0 = "unknown" for watershed)
    markers[unknown == 255] = 0    # mark unknown region as 0

    if original_bgr.shape[:2] != markers.shape[:2]:
        markers = cv2.resize(
            markers.astype(np.int32), (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    markers = cv2.watershed(original_bgr.copy(), markers.astype(np.int32))

    overlay = original_bgr.copy()
    thickness = int(boundary_thickness)
    boundary_mask = (markers == -1)
    if thickness > 1:
        boundary_mask = cv2.dilate(
            boundary_mask.astype(np.uint8),
            np.ones((thickness, thickness), np.uint8)).astype(bool)
    overlay[boundary_mask] = [0, 0, 255]  # BGR red

    num_regions = max(len(np.unique(markers)) - 2, 0)  # minus background(-1)/unknown(0) labels
    return overlay, markers, num_regions


def resize_max_dim(image: Image, max_dim: int) -> Image:
    """Downscale `image` so its longer side is at most max_dim pixels,
    preserving aspect ratio. Never upscales. Used to keep the live preview
    fast regardless of the source image's resolution."""
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale >= 1.0:
        return image.copy()
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
