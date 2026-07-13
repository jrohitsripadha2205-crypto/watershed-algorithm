"""
widgets.py
----------
Reusable CustomTkinter UI components used to build the Studio's
interface:

    * Tooltip        - a lightweight hover tooltip for any widget
    * LabeledSlider   - a slider with a label, live value readout, and
                        an optional tooltip
    * ThumbnailPanel  - a titled image preview panel for the
                        intermediate-step thumbnails
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class Tooltip:
    """A minimal hover tooltip. Attach it to any CTk/Tk widget:

        Tooltip(some_label, "Explanation text here")
    """

    def __init__(self, widget, text: str, delay_ms: int = 500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: Optional[str] = None
        self._tip_window: Optional[ctk.CTkToplevel] = None

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _event=None):
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self):
        if self._tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

        self._tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass  # not critical if the platform doesn't support this

        ctk.CTkLabel(
            tw, text=self.text, justify="left", fg_color=("gray90", "gray20"),
            corner_radius=6, font=ctk.CTkFont(size=11),
        ).pack(padx=8, pady=5)

    def _hide(self, _event=None):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None


class LabeledSlider(ctk.CTkFrame):
    """
    A CTkSlider with a text label and a live-updating value readout.
    Fires `on_change(value)` whenever the slider moves. An optional
    tooltip can be attached that explains what the parameter does.
    """

    def __init__(self, master, label: str, frm: float, to: float, steps: int,
                 default: float, on_change: Optional[Callable[[float], None]] = None,
                 is_float: bool = False, tooltip: Optional[str] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_change = on_change
        self.is_float = is_float

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x")

        text_label = ctk.CTkLabel(header, text=label, anchor="w")
        text_label.pack(side="left")
        self.value_label = ctk.CTkLabel(header, text=self._fmt(default), anchor="e")
        self.value_label.pack(side="right")

        self.slider = ctk.CTkSlider(
            self, from_=frm, to=to, number_of_steps=steps, command=self._on_move)
        self.slider.set(default)
        self.slider.pack(fill="x", pady=(2, 8))

        if tooltip:
            Tooltip(text_label, tooltip)

    def _fmt(self, value: float) -> str:
        return f"{value:.2f}" if self.is_float else str(int(value))

    def _on_move(self, value: float):
        value = float(value) if self.is_float else int(round(value))
        self.value_label.configure(text=self._fmt(value))
        if self.on_change:
            self.on_change(value)

    def get(self):
        v = self.slider.get()
        return float(v) if self.is_float else int(round(v))

    def set_silent(self, value: float):
        """Update the slider position and label without firing on_change -
        used when loading a saved configuration or resetting to defaults."""
        self.slider.set(value)
        self.value_label.configure(text=self._fmt(value))


class ThumbnailPanel(ctk.CTkFrame):
    """A titled image preview panel used for the intermediate-step
    thumbnails (e.g. Preprocessed, Binary Mask)."""

    def __init__(self, master, title: str, width: int, height: int, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(0, 4))
        self.image_label = ctk.CTkLabel(
            self, text="No image", fg_color=("gray85", "gray20"),
            width=width, height=height, corner_radius=8)
        self.image_label.pack(fill="both", expand=True)
        self._ctk_image = None  # keep a reference to prevent garbage collection

    def update_image(self, ctk_image):
        self._ctk_image = ctk_image
        if ctk_image is None:
            self.image_label.configure(image=None, text="No image")
        else:
            self.image_label.configure(image=ctk_image, text="")
