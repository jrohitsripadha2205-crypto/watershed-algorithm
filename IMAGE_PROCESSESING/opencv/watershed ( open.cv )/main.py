"""
main.py
-------
Entry point for Watershed Segmentation Studio.

Run with:
    python main.py
"""

from utils import setup_logging

# Configure logging before anything else so import-time issues in other
# modules are still captured with a proper format.
setup_logging()

from gui import run_app


if __name__ == "__main__":
    run_app()
