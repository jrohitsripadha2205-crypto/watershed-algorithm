#!/usr/bin/env python3
"""
Application entry point.

Only responsible for launching the app: creating the root Tk
instance and starting the mainloop. No GUI logic, no processing logic,
no image processing lives here -- matching the original single-file
script's ``main()`` / ``_launch_main()`` / ``_launch_login()``
functions, which are relocated verbatim below.

Note on behaviour parity: in the original single-file application,
``SplashScreen`` and ``LoginWindow`` were both defined but neither was
ever actually launched from ``main()`` -- ``main()`` called
``_launch_main()`` directly, which opens ``ImageAnalyzerApp``
(``Application`` here) straight away. That is reproduced exactly here:
``SplashScreen`` and ``LoginWindow`` remain fully defined and
importable (dead code, same as before) but are not wired into the
startup path, so the running application behaves identically to the
original file.
"""

import tkinter as tk

try:
    from .app import Application
    from .ui.login import LoginWindow
    from .ui.dialogs import SplashScreen  # noqa: F401
    from .utils.logging import get_logger
except ImportError:
    from app import Application
    from ui.login import LoginWindow
    from ui.dialogs import SplashScreen
    from utils.logging import get_logger


def _launch_main():
    root = tk.Tk()
    app = Application(root)
    root.mainloop()


def _launch_login():
    root = tk.Tk()
    LoginWindow(root)
    root.mainloop()


def main():
    logger = get_logger()
    logger.info("Application starting")
    try:
        _launch_main()
    finally:
        logger.info("Application closed")


if __name__ == "__main__":
    main()
