"""Transparent, click-through gaze overlay (PHASE-02A).

A frameless, translucent, always-on-top window covers the screen; a soft ring
follows the gaze over the user's real applications (Tobii-style). The window is
click-through and never takes focus, so the apps underneath behave normally.

Architecture (this module is a CONSUMER of the existing pipeline; it does not
touch layer 1 or layer 2):
- GazeWorker (QThread): runs the blocking camera + CNN + mapping loop and the
  One-Euro smoother, emitting gazeUpdated(x, y, visible) at camera rate.
- OverlayWindow (QWidget): a fullscreen translucent surface that repaints ONLY
  the dirty region (previous + current cursor rects), so cost stays marginal.
- OverlayApp: window + worker + system tray + global hotkeys.

Efficiency: never a fullscreen 30 fps repaint - update(QRect) clips painting to
a small area around the cursor. Fade is animated by a short timer.

Click-through: Qt's WindowTransparentForInput plus a win32 touch-up that adds
WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE (the NOACTIVATE bit stops
the overlay from ever stealing focus when the user clicks an app beneath it).
Elevated (admin) windows sit at a higher integrity level and will draw OVER the
overlay - a documented Windows limitation, not a bug.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from mapping.smoother import PointSmoother
from overlay.activity_db import (STATE_CAMERA_LOST, STATE_FACE_LOST,
                                 STATE_OFFSCREEN, STATE_OK, STATE_POSE_EXTREME)
from overlay.config import OverlayConfig

# Fixed fallback pose limits (degrees), used only when the loaded calibration
# does not carry the covered head-pose range. Deliberately generous so normal
# calibrated use at the screen edges (head yaw seen out to ~+-39 deg on the
# reference rig) never fades.
FALLBACK_YAW_DEG = 45.0
FALLBACK_PITCH_DEG = 30.0

FADE_MS = 250          # show/hide and confidence fade duration
ANIM_INTERVAL_MS = 16  # ~60 fps fade animation


@dataclass
class CursorStyle:
    diameter: int = 40           # ring outer diameter (px)
    opacity: float = 0.85        # peak opacity of the ring/fill
    color: tuple = (235, 235, 228)  # off-white, neutral


@dataclass
class GazeSample:
    """One per-frame observation handed to the 2B logger."""
    x: float
    y: float
    face_present: bool
    on_screen: bool
    state: str


def _pose_limits(meta: dict) -> tuple[tuple[float, float], tuple[float, float],
                                      bool]:
    """Covered (yaw_min, yaw_max), (pitch_min, pitch_max) from calibration
    metadata, plus whether it came from the calibration (True) or the fixed
    fallback (False). Supports the v3 top-level fields and the v2 diagnostics
    block."""
    diag = meta.get("diagnostics", {}) if isinstance(meta, dict) else {}
    yaw = meta.get("head_yaw_range_deg") or diag.get("head_yaw_range_deg")
    pitch = meta.get("head_pitch_range_deg") or diag.get("head_pitch_range_deg")
    if yaw and pitch and len(yaw) == 2 and len(pitch) == 2:
        return (float(yaw[0]), float(yaw[1])), \
            (float(pitch[0]), float(pitch[1])), True
    return (-FALLBACK_YAW_DEG, FALLBACK_YAW_DEG), \
        (-FALLBACK_PITCH_DEG, FALLBACK_PITCH_DEG), False


class GazeWorker(QtCore.QThread):
    """Runs the gaze pipeline off the UI thread.

    Emits gazeUpdated(x, y, visible) for the cursor and sampled(GazeSample)
    for the 2B logger, and survives camera disconnects by reconnecting with
    backoff (recording the gap as camera_lost meanwhile)."""

    gazeUpdated = QtCore.pyqtSignal(float, float, bool)
    sampled = QtCore.pyqtSignal(object)      # GazeSample, for the logger
    status = QtCore.pyqtSignal(str)

    def __init__(self, pipeline_factory, mapper, screen_size, smoother,
                 pose_margin_deg: float, offscreen_margin: float,
                 yaw_range, pitch_range):
        super().__init__()
        self._pipeline_factory = pipeline_factory
        self._mapper = mapper
        self._w, self._h = screen_size
        self._smoother = smoother
        self._pose_margin = pose_margin_deg
        self._offscreen_px = offscreen_margin * self._w
        self._yaw_lo, self._yaw_hi = yaw_range
        self._pitch_lo, self._pitch_hi = pitch_range
        self._running = True

    def set_domain(self, pose_margin_deg: float,
                   offscreen_margin: float) -> None:
        """Live-update the fade thresholds (settings panel). Plain float
        writes; the run() loop reads them each frame."""
        self._pose_margin = pose_margin_deg
        self._offscreen_px = offscreen_margin * self._w

    def _classify(self, head_yaw: float, head_pitch: float, x: float,
                  y: float) -> tuple[bool, bool, str]:
        """Return (cursor_visible, on_screen, tracking_state)."""
        m = self._pose_margin
        pose_ok = (self._yaw_lo - m <= head_yaw <= self._yaw_hi + m
                   and self._pitch_lo - m <= head_pitch <= self._pitch_hi + m)
        off = max(-x, x - self._w, -y, y - self._h, 0.0)
        on_screen = 0 <= x < self._w and 0 <= y < self._h
        if not pose_ok:
            state = STATE_POSE_EXTREME
        elif off > self._offscreen_px:
            state = STATE_OFFSCREEN
        else:
            state = STATE_OK
        return (pose_ok and off <= self._offscreen_px), on_screen, state

    def _reconnect(self, pipeline, backoff: float) -> bool:
        """Reopen the CAMERA only (keep MediaPipe and the CNN loaded)."""
        try:
            pipeline.camera.release()
        except Exception:
            pass
        time.sleep(backoff)
        try:
            pipeline.camera.open()
            return True
        except Exception:
            return False

    def run(self) -> None:
        from mapping.calibrator import make_features
        pipeline = self._pipeline_factory()
        try:
            pipeline.open()
        except Exception as exc:
            self.status.emit(f"camera error: {exc}; will keep retrying")
        last_x, last_y = self._w / 2.0, self._h / 2.0
        fails, backoff = 0, 1.0
        try:
            while self._running:
                try:
                    st, obs, gaze = pipeline.read_state()
                except Exception:
                    st, obs, gaze = "no_frame", None, None
                now = time.perf_counter()

                if st == "no_frame":
                    fails += 1
                    self.gazeUpdated.emit(last_x, last_y, False)
                    self.sampled.emit(GazeSample(last_x, last_y, False, False,
                                                 STATE_CAMERA_LOST))
                    if fails >= 15:            # ~0.5 s of dead frames
                        self.status.emit("camera lost; reconnecting")
                        if self._reconnect(pipeline, backoff):
                            fails, backoff = 0, 1.0
                            self.status.emit("camera reconnected")
                        else:
                            backoff = min(backoff * 2, 10.0)
                    continue
                fails, backoff = 0, 1.0

                if st == "no_face":
                    self.gazeUpdated.emit(last_x, last_y, False)
                    self.sampled.emit(GazeSample(last_x, last_y, False, False,
                                                 STATE_FACE_LOST))
                    continue

                feats = make_features(gaze.yaw_deg, gaze.pitch_deg,
                                      obs.head_yaw, obs.head_pitch)
                raw_x, raw_y = self._mapper.predict(feats)
                visible, on_screen, state = self._classify(
                    obs.head_yaw, obs.head_pitch, raw_x, raw_y)
                x = min(max(raw_x, 0.0), self._w - 1)
                y = min(max(raw_y, 0.0), self._h - 1)
                sx, sy = self._smoother.filter(x, y, now)
                last_x, last_y = sx, sy
                self.gazeUpdated.emit(sx, sy, visible)
                self.sampled.emit(GazeSample(sx, sy, True, on_screen, state))
        finally:
            try:
                pipeline.close()
            except Exception:
                pass

    def request_stop(self) -> None:
        """Signal the loop to end without blocking the caller (the UI thread
        must not stall here). The caller waits on the thread separately."""
        self._running = False

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


def _draw_cursor(painter: QtGui.QPainter, x: float, y: float, alpha: float,
                 style: CursorStyle) -> None:
    """Soft ring with a faint halo and semitransparent fill."""
    if alpha <= 0.01:
        return
    r = style.diameter / 2.0
    cr, cg, cb = style.color
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    center = QtCore.QPointF(x, y)

    # Halo: a large, very soft radial glow.
    halo = QtGui.QRadialGradient(center, r * 2.2)
    halo.setColorAt(0.0, QtGui.QColor(cr, cg, cb,
                                      int(70 * alpha * style.opacity)))
    halo.setColorAt(1.0, QtGui.QColor(cr, cg, cb, 0))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(halo)
    painter.drawEllipse(center, r * 2.2, r * 2.2)

    # Semitransparent fill.
    painter.setBrush(QtGui.QColor(cr, cg, cb, int(45 * alpha * style.opacity)))
    painter.drawEllipse(center, r, r)

    # Ring stroke.
    pen = QtGui.QPen(QtGui.QColor(cr, cg, cb, int(235 * alpha * style.opacity)))
    pen.setWidthF(max(2.0, style.diameter * 0.06))
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawEllipse(center, r, r)


class OverlayWindow(QtWidgets.QWidget):
    """Fullscreen translucent click-through surface."""

    def __init__(self, screen_size, style: CursorStyle, deadzone_px: float,
                 fade_ms: int = FADE_MS):
        super().__init__()
        self._w, self._h = screen_size
        self._style = style
        self._deadzone = deadzone_px
        self._fade_ms = fade_ms

        self._x = self._w / 2.0
        self._y = self._h / 2.0
        self._alpha = 0.0            # current fade
        self._target_alpha = 0.0     # 1 when visible and shown, else 0
        self._shown = True           # hotkey toggle

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(0, 0, self._w, self._h)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(ANIM_INTERVAL_MS)
        self._last_anim = time.perf_counter()

    # --- geometry helpers --------------------------------------------------
    def _cursor_rect(self, x: float, y: float) -> QtCore.QRect:
        pad = self._style.diameter * 2.4 + 4
        return QtCore.QRect(int(x - pad), int(y - pad),
                            int(2 * pad), int(2 * pad))

    def _repaint_region(self, old_x, old_y, new_x, new_y) -> None:
        self.update(self._cursor_rect(old_x, old_y).united(
            self._cursor_rect(new_x, new_y)))

    # --- slots -------------------------------------------------------------
    @QtCore.pyqtSlot(float, float, bool)
    def on_gaze(self, x: float, y: float, visible: bool) -> None:
        moved = abs(x - self._x) >= self._deadzone \
            or abs(y - self._y) >= self._deadzone
        old_x, old_y = self._x, self._y
        if moved:
            self._x, self._y = x, y
        self._target_alpha = 1.0 if (visible and self._shown) else 0.0
        if moved:
            self._repaint_region(old_x, old_y, self._x, self._y)

    def set_shown(self, shown: bool) -> None:
        self._shown = shown
        if not shown:
            self._target_alpha = 0.0

    def set_style(self, style: CursorStyle) -> None:
        """Live cursor restyle; repaints a region sized for the larger of the
        old/new diameters."""
        pad = max(self._style.diameter, style.diameter) * 2.4 + 4
        self._style = style
        self.update(QtCore.QRect(int(self._x - pad), int(self._y - pad),
                                 int(2 * pad), int(2 * pad)))

    def set_deadzone(self, deadzone_px: float) -> None:
        self._deadzone = deadzone_px

    def set_fade_ms(self, fade_ms: int) -> None:
        self._fade_ms = max(1, fade_ms)

    def toggle_shown(self) -> None:
        self.set_shown(not self._shown)

    def _animate(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_anim
        self._last_anim = now
        if abs(self._alpha - self._target_alpha) < 0.004:
            if self._alpha != self._target_alpha:
                self._alpha = self._target_alpha
                self.update(self._cursor_rect(self._x, self._y))
            return
        step = dt / (self._fade_ms / 1000.0)   # normalized 0..1 per fade_ms
        if self._alpha < self._target_alpha:
            self._alpha = min(self._alpha + step, self._target_alpha)
        else:
            self._alpha = max(self._alpha - step, self._target_alpha)
        self.update(self._cursor_rect(self._x, self._y))

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        _draw_cursor(painter, self._x, self._y, self._alpha, self._style)


def apply_click_through(hwnd: int) -> bool:
    """Add WS_EX_LAYERED | TRANSPARENT | TOPMOST | NOACTIVATE so clicks pass
    through and the overlay never steals focus. Returns True if applied."""
    try:
        import win32con
        import win32gui
    except ImportError:
        return False
    ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    ex |= (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
           | win32con.WS_EX_TOPMOST | win32con.WS_EX_NOACTIVATE)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex)
    return True


def _tray_icon(style: CursorStyle) -> QtGui.QIcon:
    """A ring icon drawn in code (no shipped asset, no emoji)."""
    pm = QtGui.QPixmap(64, 64)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    cr, cg, cb = style.color
    pen = QtGui.QPen(QtGui.QColor(cr, cg, cb, 235))
    pen.setWidth(6)
    p.setPen(pen)
    p.setBrush(QtGui.QColor(cr, cg, cb, 40))
    p.drawEllipse(QtCore.QPointF(32, 32), 20, 20)
    p.end()
    return QtGui.QIcon(pm)


class _HotkeyBridge(QtCore.QObject):
    """Marshals pynput hotkey callbacks (their thread) onto the UI thread."""
    toggle = QtCore.pyqtSignal()
    settings = QtCore.pyqtSignal()
    quit = QtCore.pyqtSignal()


class OverlayApp:
    """Owns the Qt app, overlay window, gaze worker, tray, hotkeys and the
    live settings panel. Settings changes flow: dialog -> apply_config ->
    window/worker/smoother setters + debounced persist to overlay.yaml. The
    pipeline itself is never touched."""

    def __init__(self, pipeline_factory, mapper, screen_size,
                 config: OverlayConfig, use_tray: bool = True,
                 enable_logging: bool = False, db_path=None,
                 no_titles: bool = False, start_hidden: bool = False):
        from overlay.config import save_config
        self._save_config = save_config
        self.app = QtWidgets.QApplication.instance() \
            or QtWidgets.QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.config = config

        yaw_range, pitch_range, from_cal = _pose_limits(mapper.meta or {})
        src = "calibration" if from_cal else "fixed fallback"
        print(f"fade domain        : head yaw {yaw_range[0]:.0f}.."
              f"{yaw_range[1]:.0f}, pitch {pitch_range[0]:.0f}.."
              f"{pitch_range[1]:.0f} deg (+/-{config.pose_margin:.0f} margin) "
              f"[{src}]")
        if not from_cal:
            print("  note: this calibration has no head-pose range; using the "
                  "generous fixed fallback. Recalibrate to use the exact "
                  "covered range.")

        style = CursorStyle(diameter=config.dot_size,
                            opacity=config.dot_opacity, color=config.dot_color)
        self.smoother = PointSmoother(min_cutoff=config.min_cutoff,
                                      beta=config.beta)
        self.window = OverlayWindow(screen_size, style, config.deadzone,
                                    config.fade_ms)
        self.worker = GazeWorker(pipeline_factory, mapper, screen_size,
                                 self.smoother, config.pose_margin,
                                 config.offscreen_margin, yaw_range,
                                 pitch_range)
        self.worker.gazeUpdated.connect(self.window.on_gaze)
        self.worker.status.connect(self._on_status)

        # 2B behavior logging (optional consumer of the same worker).
        self._db = None
        self._logger = None
        if enable_logging:
            from overlay.activity_db import ActivityDB
            from overlay.activity_logger import ActivityLogger
            self._db = ActivityDB(db_path)
            sid = self._db.start_session(
                notes="overlay hidden" if start_hidden else "overlay visible")
            self._logger = ActivityLogger(
                self._db, sid, screen_size, config.grid_cols, config.grid_rows,
                no_titles)
            self.worker.sampled.connect(self._logger.on_sample)
            est = self._db.estimate_size_per_day()
            print(f"logging            : {self._db.path} (SQLite WAL), grid "
                  f"{config.grid_cols}x{config.grid_rows}, window titles "
                  f"{'OFF (--no-titles)' if no_titles else 'on'}")
            print(f"est. DB size       : ~{est['mb_per_day']} MB/day of use "
                  "(5 s intervals)")
        if start_hidden:
            self.window.set_shown(False)

        self._dialog = None
        self._save_timer = QtCore.QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(
            lambda: self._save_config(self.config))

        self._quitting = False
        self._bridge = _HotkeyBridge()
        self._bridge.toggle.connect(self.window.toggle_shown)
        self._bridge.settings.connect(self.show_settings)
        self._bridge.quit.connect(self.quit)
        self._hotkeys = self._start_hotkeys()

        self._tray = self._build_tray(style) if use_tray else None

    # --- live settings -----------------------------------------------------
    def apply_config(self, cfg: OverlayConfig) -> None:
        """Apply a settings change to the running overlay immediately, then
        persist (debounced) to overlay.yaml."""
        self.config = cfg
        self.window.set_style(CursorStyle(diameter=cfg.dot_size,
                                          opacity=cfg.dot_opacity,
                                          color=cfg.dot_color))
        self.window.set_deadzone(cfg.deadzone)
        self.window.set_fade_ms(cfg.fade_ms)
        self.worker.set_domain(cfg.pose_margin, cfg.offscreen_margin)
        self.smoother.set_params(cfg.min_cutoff, cfg.beta)
        if self._logger is not None:
            self._logger.set_grid(cfg.grid_cols, cfg.grid_rows)
        self._save_timer.start(400)   # debounce disk writes during a drag

    def show_settings(self) -> None:
        from overlay.settings_dialog import SettingsDialog
        if self._dialog is None:
            self._dialog = SettingsDialog(self.config)
            self._dialog.configChanged.connect(self.apply_config)
        else:
            self._dialog.set_from_config(self.config)
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def _on_status(self, msg: str) -> None:
        print(f"overlay: {msg}")
        if self._tray is not None:
            self._tray.showMessage("gazetrack", msg,
                                   QtWidgets.QSystemTrayIcon.MessageIcon.Warning,
                                   4000)

    def _start_hotkeys(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("WARNING: pynput not available; global hotkeys disabled.")
            return None
        hk = keyboard.GlobalHotKeys({
            "<ctrl>+<alt>+g": lambda: self._bridge.toggle.emit(),
            "<ctrl>+<alt>+s": lambda: self._bridge.settings.emit(),
            "<ctrl>+<alt>+q": lambda: self._bridge.quit.emit(),
        })
        hk.daemon = True
        hk.start()
        print("hotkeys            : Ctrl+Alt+G show/hide, Ctrl+Alt+S "
              "settings, Ctrl+Alt+Q quit")
        return hk

    def _build_tray(self, style: CursorStyle):
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            print("WARNING: system tray unavailable; running without it.")
            return None
        tray = QtWidgets.QSystemTrayIcon(_tray_icon(style))
        menu = QtWidgets.QMenu()
        act_toggle = menu.addAction("Mostrar / ocultar cursor")
        act_toggle.triggered.connect(self.window.toggle_shown)
        act_settings = menu.addAction("Configuracion")
        act_settings.triggered.connect(self.show_settings)
        if self._logger is not None:
            menu.addSeparator()
            self._act_record = menu.addAction("Pausar registro")
            self._act_record.triggered.connect(self.toggle_recording)
            self._logger.stateChanged.connect(self._on_record_state)
        menu.addSeparator()
        act_quit = menu.addAction("Salir")
        act_quit.triggered.connect(self.quit)
        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda reason: self.window.toggle_shown()
            if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Trigger
            else None)
        tray.setToolTip("gazetrack" + (" - registrando"
                                       if self._logger is not None else ""))
        tray.show()
        return tray

    def toggle_recording(self) -> None:
        if self._logger is not None:
            self._logger.set_paused(not self._logger.paused)

    def _on_record_state(self, recording: bool) -> None:
        if self._tray is not None:
            self._act_record.setText("Pausar registro" if recording
                                     else "Reanudar registro")
            self._tray.setToolTip("gazetrack - "
                                  + ("registrando" if recording
                                     else "registro en pausa"))
        print(f"logging: {'recording' if recording else 'paused'}",
              flush=True)

    def quit(self) -> None:
        # Keep this non-blocking and break the event loop FIRST. In
        # particular, do NOT call pynput's hotkey stop() here: on Windows it
        # can hang when invoked from another thread, which would keep exec()
        # from ever returning. Graceful camera close + hard exit happen in
        # run() after exec() returns.
        if self._quitting:
            return
        self._quitting = True
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_config(self.config)
        self.worker.request_stop()      # signal the loop, do not block here
        # Close the DB cleanly HERE (before the os._exit in run()): end the
        # session and checkpoint the WAL so the hard exit cannot corrupt it.
        if self._logger is not None:
            self._logger.stop()
        if self._db is not None:
            self._db.close()
        if self._dialog is not None:
            self._dialog.close()
        if self._tray is not None:
            self._tray.hide()
        print("overlay: exiting", flush=True)
        self.app.quit()

    def run(self, exit_after: float | None = None) -> int:
        self.window.show()
        applied = apply_click_through(int(self.window.winId()))
        print(f"click-through      : "
              f"{'win32 (layered/transparent/noactivate)' if applied else 'Qt WindowTransparentForInput only'}")
        self.worker.start()
        if exit_after:
            QtCore.QTimer.singleShot(int(exit_after * 1000), self.quit)
        code = self.app.exec()
        # exec() has returned; terminate hard. The global keyboard hook
        # (pynput) and the CUDA/camera native threads do not reliably unwind
        # at interpreter shutdown and would hang the process; os._exit skips
        # that. The OS reclaims the camera handle on process death. (Do NOT
        # QThread.wait here: its PyQt6 int overload did not honor the
        # timeout and blocked forever.)
        sys.stdout.flush()
        os._exit(code if isinstance(code, int) else 0)
