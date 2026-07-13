"""
utils.py
--------
Small helper functions shared across the application: application-wide
logging setup, and OpenCV <-> CTkImage conversion for the preview
thumbnails.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the application's shared logger. Safe to call
    more than once (e.g. if a module is re-imported) - it only attaches
    handlers the first time.
    """
    logger = logging.getLogger("watershed_studio")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def cv2_to_ctk_image(cv_image: Optional[np.ndarray], max_size: int):
    """
    Convert a BGR or single-channel grayscale/binary OpenCV image into a
    CTkImage that fits within a max_size x max_size box, preserving
    aspect ratio (never upscaling). Returns None if cv_image is None.

    customtkinter is imported lazily inside this function so that
    processor.py-style pure logic (and any headless test) can import
    this module without requiring a Tk display or customtkinter to be
    installed.
    """
    import customtkinter as ctk

    if cv_image is None:
        return None

    if len(cv_image.shape) == 2:
        pil_img = Image.fromarray(cv_image)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    w, h = pil_img.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    pil_img = pil_img.resize(new_size, Image.LANCZOS)

    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
