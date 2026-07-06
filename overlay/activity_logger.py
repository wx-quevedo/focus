"""Behavior aggregation for the productivity sensor (PHASE-02B).

Consumes the overlay worker's per-frame gaze samples, samples the foreground
window once a second, and counts keyboard/mouse EVENTS - then writes one
aggregated row per INTERVAL_SECONDS to the activity DB.

PRIVACY - NON-NEGOTIABLE:
    The keyboard listener increments a counter on each press. It MUST NEVER
    read, store, log, transmit or otherwise inspect WHICH key was pressed.
    There is deliberately no reference to `key` in the callback body beyond
    counting. This is a keystroke COUNTER, not a keylogger. The same holds
    for mouse: we count clicks/scrolls/throttled-moves, never positions as
    content. Everything stays on this machine.
"""

from __future__ import annotations

import time
from collections import Counter

from PyQt6 import QtCore

from overlay.activity_db import (INTERVAL_SECONDS, STATE_CAMERA_LOST,
                                 ActivityDB)

_MOVE_THROTTLE_S = 0.15   # count at most one mouse move per this window


def zone_of(x: float, y: float, w: int, h: int, cols: int,
            rows: int) -> str | None:
    """Grid cell label 'c<col>r<row>' for an on-screen point, or None when
    the point is off-screen."""
    if not (0 <= x < w and 0 <= y < h):
        return None
    col = min(int(x / w * cols), cols - 1)
    row = min(int(y / h * rows), rows - 1)
    return f"c{col}r{row}"


class ActivityLogger(QtCore.QObject):
    """Aggregates per-frame samples + input counts into DB rows. Lives on the
    UI thread; the DB write per interval is a committed transaction."""

    stateChanged = QtCore.pyqtSignal(bool)   # True = recording, False = paused

    def __init__(self, db: ActivityDB, session_id: int,
                 screen_size: tuple[int, int], grid_cols: int, grid_rows: int,
                 no_titles: bool):
        super().__init__()
        self._db = db
        self._session = session_id
        self._w, self._h = screen_size
        self._cols, self._rows = grid_cols, grid_rows
        self._no_titles = no_titles
        self._paused = False

        self._reset_accumulators()
        # Input counters (written from pynput threads, read/reset here).
        self._kb = 0
        self._mouse = 0
        self._last_move = 0.0
        self._kb_listener = None
        self._mouse_listener = None
        self._start_input_listeners()

        self._win_timer = QtCore.QTimer(self)
        self._win_timer.timeout.connect(self._sample_window)
        self._win_timer.start(1000)
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start(INTERVAL_SECONDS * 1000)

    # --- accumulators ------------------------------------------------------
    def _reset_accumulators(self) -> None:
        self._frames = 0
        self._on_screen = 0
        self._face = 0
        self._zones: Counter = Counter()
        self._states: Counter = Counter()
        self._procs: Counter = Counter()
        self._titles: Counter = Counter()

    # --- per-frame gaze sample (from the worker, UI thread) ----------------
    @QtCore.pyqtSlot(object)
    def on_sample(self, s) -> None:
        if self._paused:
            return
        self._frames += 1
        self._states[s.state] += 1
        if s.face_present:
            self._face += 1
        if s.on_screen:
            self._on_screen += 1
            zone = zone_of(s.x, s.y, self._w, self._h, self._cols, self._rows)
            if zone:
                self._zones[zone] += 1

    # --- foreground window sampling (1 Hz) ---------------------------------
    def _sample_window(self) -> None:
        if self._paused:
            return
        proc, title = self._foreground()
        if proc:
            self._procs[proc] += 1
        if title and not self._no_titles:
            self._titles[title] += 1

    @staticmethod
    def _foreground() -> tuple[str | None, str | None]:
        try:
            import psutil
            import win32gui
            import win32process
        except ImportError:
            return None, None
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None, None
        title = win32gui.GetWindowText(hwnd) or None
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid).name()
        except Exception:
            proc = None
        return proc, title

    # --- input counters (pynput threads) -----------------------------------
    def _start_input_listeners(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError:
            print("WARNING: pynput unavailable; keyboard/mouse counts off.")
            return

        def on_press(key):
            # PRIVACY: count only. The key identity is never read or stored.
            self._kb += 1

        def on_click(x, y, button, pressed):
            if pressed:
                self._mouse += 1

        def on_scroll(x, y, dx, dy):
            self._mouse += 1

        def on_move(x, y):
            now = time.perf_counter()
            if now - self._last_move >= _MOVE_THROTTLE_S:
                self._last_move = now
                self._mouse += 1

        self._kb_listener = keyboard.Listener(on_press=on_press)
        self._kb_listener.daemon = True
        self._kb_listener.start()
        self._mouse_listener = mouse.Listener(on_click=on_click,
                                              on_scroll=on_scroll,
                                              on_move=on_move)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    # --- flush one interval to disk ----------------------------------------
    def _flush(self) -> None:
        if self._paused:
            return
        kb, self._kb = self._kb, 0
        mouse, self._mouse = self._mouse, 0
        frames = self._frames

        if frames == 0:
            # Worker produced nothing this interval (camera reconnecting):
            # record the gap honestly rather than dropping it.
            zone, on_frac, state, face_frac = "off", 0.0, STATE_CAMERA_LOST, 0.0
        else:
            on_frac = self._on_screen / frames
            face_frac = self._face / frames
            state = self._states.most_common(1)[0][0]
            zone = (self._zones.most_common(1)[0][0] if self._zones
                    else "off")
        proc = self._procs.most_common(1)[0][0] if self._procs else None
        title = self._titles.most_common(1)[0][0] if self._titles else None

        self._db.insert_sample(
            self._session, zone, on_frac, state, face_frac, proc, title,
            kb, mouse)
        self._reset_accumulators()

    # --- control -----------------------------------------------------------
    def set_grid(self, cols: int, rows: int) -> None:
        """Live grid change from the settings panel; affects zone attribution
        from the next frame on."""
        self._cols, self._rows = int(cols), int(rows)

    def set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        if paused:
            self._reset_accumulators()   # drop the partial interval
        self.stateChanged.emit(not paused)

    @property
    def paused(self) -> bool:
        return self._paused

    def stop(self) -> None:
        for t in (self._win_timer, self._flush_timer):
            t.stop()
        for lst in (self._kb_listener, self._mouse_listener):
            if lst is not None:
                try:
                    lst.stop()
                except Exception:
                    pass
        self._db.end_session(self._session)
