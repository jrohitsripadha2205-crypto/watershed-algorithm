# Watershed Segmentation Studio

A real-time desktop application for OpenCV **watershed image
segmentation**, built with CustomTkinter, OpenCV, and Matplotlib.

Every parameter — blur, brightness/contrast, threshold, morphology,
and the watershed algorithm itself — lives in its own tab and updates
the preview **immediately** as you move a slider, with no "Run"
button. All the heavy image processing runs on a background thread,
so the interface stays responsive even while you're dragging a
slider around.

---

## Features

- **Tabbed parameter panel**: Preprocessing / Threshold / Morphology / Watershed
- **Live preview**, updated in real time as parameters change:
  - Two thumbnails (Preprocessed image, Binary mask after threshold + morphology)
  - An embedded Matplotlib plot showing the labeled segmented regions
    (colored with a colormap) next to the original image, complete
    with the standard pan/zoom/save toolbar
- **Non-blocking UI**: processing runs on a background thread with
  debounced updates, so the app never freezes while you adjust values
- **Menu bar** with keyboard shortcuts (Load: `Ctrl+O`, Save: `Ctrl+S`,
  Reset: `Ctrl+R`, Quit: `Ctrl+Q`)
- **Tooltips** on every parameter explaining what it does
- **Save full-resolution result** — the live preview works on a
  downscaled copy for speed, but "Save Result" reruns the exact same
  pipeline at full resolution before saving
- **Save / Load Configuration** as JSON, so a parameter setup can be
  restored later or shared with someone else
- **Dark / Light appearance toggle**
- Logging to the console and friendly error dialogs if something goes wrong

---

## Project Structure

```
watershed_pro/
├── main.py            # Entry point
├── gui.py              # Main window: menu, tabs, live preview, threading
├── processor.py         # Pure OpenCV logic (no GUI code) - reusable/testable
├── widgets.py            # Reusable UI components (slider, tooltip, thumbnail)
├── config.py              # App constants, default params, slider specs, JSON I/O
├── utils.py                # Logging setup, OpenCV -> CTkImage conversion
├── requirements.txt         # Python dependencies
└── README.md                 # This file
```

**Why this structure?** `processor.py` contains zero GUI code, so it
can be reused from a script or tested independently. `config.py`
holds every parameter's range/default/tooltip in one place, so
`gui.py` builds the tabs from data instead of repeating boilerplate
for each slider. `gui.py` is the only file that knows about
Tkinter/CustomTkinter/Matplotlib widgets.

---

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

---

## Usage

1. **File > Load Image** (or `Ctrl+O`) to open a photo.
2. Switch between the **Preprocessing**, **Threshold**, **Morphology**,
   and **Watershed** tabs on the left and adjust any slider or
   dropdown — the preview on the right updates automatically after a
   brief pause.
3. The top strip shows the **Preprocessed** image and the resulting
   **Binary Mask**; the plot below shows the final **Segmented
   Regions** (colored by label) next to the **Original Image**. Use
   the toolbar under the plot to pan, zoom, or save just that figure
   as an image.
4. **File > Save Result** (or `Ctrl+S`) re-runs your current settings
   at full resolution and saves the boundary-overlay result to disk.
5. **File > Save Configuration** / **Load Configuration** to store or
   restore your exact parameter setup as a `.json` file.
6. **Edit > Reset Parameters** (or `Ctrl+R`) to return every tab to
   its default values.

### How the Watershed algorithm works

`processor.py`'s `run_watershed()` follows the standard marker-based
approach:

1. Binarize the thresholded/morphology-processed image.
2. Remove small noise specks with morphological opening.
3. Dilate the opened mask to get a "sure background" region.
4. Threshold the distance transform to get a "sure foreground" region.
5. Subtract sure-foreground from sure-background to find the
   "unknown" boundary region.
6. Label the sure-foreground blobs as markers, offset by +1, and mark
   the unknown region as 0.
7. Run `cv2.watershed()` with these markers; the resulting boundaries
   (`-1` pixels) are drawn in red on the result image, and the raw
   labeled-region array is what's shown, colorized, in the left half
   of the embedded plot.

### Performance notes

- The live preview always works on a copy of the image downscaled to
  at most `config.PREVIEW_MAX_DIM` pixels on its longest side, so
  processing stays fast regardless of how large the original file is.
- **Save Result** reruns the full pipeline on the original,
  full-resolution image, so the exported file isn't limited by the
  preview's resolution.
- Parameter changes are debounced (`config.DEBOUNCE_MS`) and run on a
  background thread; a request-id check discards any result that's
  no longer the most recent one, so rapid slider dragging never
  causes flickering or out-of-order updates.
