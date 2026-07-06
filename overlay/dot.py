"""On-screen gaze dot (MVP overlay).

Per spec, the MVP renders the dot on a fullscreen OpenCV window (opaque,
near-black background) rather than a true transparent desktop overlay.
Migrating to a click-through layered window (PyQt6 or win32) is planned for
a later phase; the DotOverlay interface is kept minimal so the backend can
be swapped without touching callers.

Visual style (project convention): sober and professional - near-black
background, off-white dot with a thin halo, small dim status text. No
decorations.
"""

from __future__ import annotations

import ctypes
import sys

import cv2
import numpy as np

_BG = (18, 18, 18)
_FG = (235, 235, 235)
_DIM = (110, 110, 110)


def get_screen_size() -> tuple[int, int]:
    """Physical screen resolution of the primary monitor (DPI-aware)."""
    if sys.platform == "win32":
        try:  # per-monitor DPI awareness; fall back to system aware
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), \
            int(user32.GetSystemMetrics(1))
    raise RuntimeError("get_screen_size is implemented for Windows only.")


class DotOverlay:
    """Fullscreen window drawing the gaze dot each frame."""

    def __init__(self, screen_size: tuple[int, int],
                 window_name: str = "gazetrack - demo",
                 fullscreen: bool = True):
        self.width, self.height = screen_size
        self.window = window_name
        cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN
                        if fullscreen else cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        self._canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def render(self, x: float | None, y: float | None,
               status: str = "") -> int:
        """Draw one frame. Pass x=y=None when there is no gaze (face lost).

        Returns the pressed key code (-1/255 mask applied) so the caller
        owns the event loop.
        """
        self._canvas[:] = _BG[0]
        if x is not None and y is not None:
            px = int(round(min(max(x, 0), self.width - 1)))
            py = int(round(min(max(y, 0), self.height - 1)))
            cv2.circle(self._canvas, (px, py), 16, (70, 70, 70), 1,
                       cv2.LINE_AA)
            cv2.circle(self._canvas, (px, py), 8, _FG, -1, cv2.LINE_AA)
        else:
            text = "no face detected"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(self._canvas, text,
                        ((self.width - size[0]) // 2, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _DIM, 1, cv2.LINE_AA)
        if status:
            cv2.putText(self._canvas, status, (24, self.height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DIM, 1, cv2.LINE_AA)
        cv2.imshow(self.window, self._canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(self.window)

    def __enter__(self) -> "DotOverlay":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
