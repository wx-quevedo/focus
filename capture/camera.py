"""Camera capture for the Logitech C920/C922 with locked, verified settings.

Known traps this module defends against (all reproduced on the project's
c922 Pro Stream Webcam):

1. The camera silently falls back to 1280x720 (or 640x480) if the requested
   mode is not applied, so we always read back what the driver delivered.
2. At 1920x1080 the uncompressed YUY2 mode is capped at 5 fps by USB
   bandwidth (it is the ONLY YUY2 1080p mode in the UVC descriptor).
   OpenCV's DSHOW backend selects the media type by resolution alone and
   lands on YUY2 no matter how FOURCC=MJPG is requested; MSMF honors the
   requested fps and negotiates MJPEG internally, so MSMF is the primary
   backend and DSHOW only a fallback. MSMF's FOURCC/focus/exposure
   readbacks are unreliable - measured fps is the ground truth.
3. A long manual exposure also caps the frame rate (~4 fps at 1/4 s), the
   same symptom as YUY2. Runtime properties (FOURCC decoded, FPS, EXPOSURE)
   are printed at every open so the cause is visible.
4. Disabling autofocus freezes the lens WHEREVER IT WAS, which can leave
   the image permanently blurred. The fix is an explicit manual focus
   value: range 0-255 in steps of 5, 0 = infinity, ~60 cm usually lands in
   25-50. Run `python -m capture.camera --focus-sweep` once to find it
   interactively; the chosen value persists to data/camera.yaml and every
   mode (calibrate/demo/record/evaluate) picks it up automatically.

Setting precedence for index/focus/exposure: explicit argument (CLI flag) >
data/camera.yaml > built-in default.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "camera.yaml"

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30

# C920 manual focus: 0-255 in steps of 5, 0 = infinity. A subject at ~60 cm
# usually lands between 25 and 50; find yours with --focus-sweep.
DEFAULT_FOCUS = 35
FOCUS_MIN, FOCUS_MAX, FOCUS_STEP = 0, 255, 5

# DirectShow exposure is log2(seconds): -6 = 1/64 s (~15.6 ms), short enough
# to sustain 30 fps. Values of -4 and longer (e.g. -2 = 1/4 s) cap the frame
# rate and are the second usual cause of a ~5 fps stream.
DEFAULT_EXPOSURE = -6
EXPOSURE_MIN, EXPOSURE_MAX = -11, -1

# DirectShow convention for CAP_PROP_AUTO_EXPOSURE.
_AUTO_EXPOSURE_ON = 0.75
_AUTO_EXPOSURE_OFF = 0.25

_BG = (18, 18, 18)
_FG = (235, 235, 235)
_DIM = (110, 110, 110)


def load_config() -> dict:
    """Read data/camera.yaml; missing or corrupt file yields {}."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        print(f"WARNING: {CONFIG_PATH} is not valid YAML; ignoring it.")
        return {}


def save_config(values: dict) -> None:
    """Merge values into data/camera.yaml."""
    merged = {**load_config(), **values}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(merged, sort_keys=False),
                           encoding="utf-8")


def snap_focus(value: float) -> int:
    """Clamp to the C920 focus range and snap to its step of 5."""
    clamped = min(max(value, FOCUS_MIN), FOCUS_MAX)
    return int(round(clamped / FOCUS_STEP) * FOCUS_STEP)


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "none"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


@dataclass
class CameraStatus:
    """Requested vs. actually applied capture settings."""

    requested_width: int
    requested_height: int
    requested_fps: float
    actual_width: int
    actual_height: int
    actual_fps: float
    fourcc: str
    # Tri-state: True/False, or None when the driver does not report the
    # property back (DirectShow get() returns -1 for several C920 controls
    # even after a successful set()).
    autofocus_off: bool | None
    focus: float
    auto_exposure_off: bool | None
    exposure: float

    @property
    def resolution_ok(self) -> bool:
        return (self.actual_width == self.requested_width
                and self.actual_height == self.requested_height)


class Camera:
    """OpenCV capture wrapper that forces and verifies C920 settings.

    index/focus/exposure resolve as: explicit argument > data/camera.yaml >
    built-in default. Pass None (the default) to use the config file.
    """

    def __init__(self,
                 index: int | None = None,
                 width: int = TARGET_WIDTH,
                 height: int = TARGET_HEIGHT,
                 fps: float = TARGET_FPS,
                 focus: float | None = None,
                 exposure: float | None = None,
                 disable_autofocus: bool = True,
                 disable_auto_exposure: bool = True):
        config = load_config()

        def resolve(explicit, key: str, default):
            if explicit is not None:
                return explicit, "flag"
            if key in config:
                return config[key], "config"
            return default, "default"

        self.index, self._index_src = resolve(index, "index", 0)
        self.index = int(self.index)
        self.width = width
        self.height = height
        self.fps = fps
        focus_value, self._focus_src = resolve(focus, "focus", DEFAULT_FOCUS)
        self.focus = snap_focus(float(focus_value))
        exposure_value, self._exposure_src = resolve(exposure, "exposure",
                                                     DEFAULT_EXPOSURE)
        self.exposure = float(exposure_value)
        self.disable_autofocus = disable_autofocus
        self.disable_auto_exposure = disable_auto_exposure
        self._cap: cv2.VideoCapture | None = None
        self._backend_name = "unknown"

    def _apply_mode(self, cap: cv2.VideoCapture) -> bool:
        """Negotiate MJPG @ WxH @ fps. Returns True if readback confirms."""

        def set_size() -> None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        def set_mjpg() -> None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        def confirmed() -> bool:
            return (_fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)) == "MJPG"
                    and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == self.width
                    and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    == self.height)

        # Order matters and differs between driver versions: try
        # fourcc-first, then size-first with the fourcc re-applied. The
        # negotiation only finalizes when the stream starts, so re-check
        # after grabbing a frame before giving up on each order.
        for order in ((set_mjpg, set_size), (set_size, set_mjpg)):
            for step in order:
                step()
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if confirmed():
                return True
            cap.read()
            if confirmed():
                return True
        return False

    def _apply_controls(self, cap: cv2.VideoCapture) -> None:
        if self.disable_autofocus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        if self.disable_auto_exposure:
            # Both conventions, harmless to apply together: DirectShow
            # expects 0.25 for manual, MSMF expects 0.
            if self._backend_name == "MSMF":
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_OFF)
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def open(self) -> "Camera":
        # Strategy determined empirically on the Logitech C920/C922 UVC
        # family under Windows: OpenCV's DSHOW backend selects the media
        # type by resolution alone and lands on YUY2, which the USB
        # bandwidth caps at ~5 fps at 1080p. MSMF honors the requested
        # fps, negotiates MJPEG internally (its FOURCC readback is blank -
        # trust measured fps) and delivers 1080p30; focus and exposure
        # controls verified to work live through MSMF. DSHOW remains as
        # fallback only.
        cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF,
                               [cv2.CAP_PROP_FRAME_WIDTH, self.width,
                                cv2.CAP_PROP_FRAME_HEIGHT, self.height,
                                cv2.CAP_PROP_FPS, int(round(self.fps))])
        msmf_ok = (cap.isOpened()
                   and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == self.width
                   and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                   == self.height)
        if not msmf_ok:
            cap.release()
            print("WARNING: MSMF could not deliver the requested mode; "
                  "falling back to DirectShow. If measured fps is ~5 at "
                  "1080p the stream is YUY2 (USB bandwidth cap).")
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"Could not open camera index {self.index}. Check "
                    "that the camera is connected and not in use by "
                    "another application.")
            self._apply_mode(cap)  # best-effort MJPG negotiation

        self._cap = cap
        self._backend_name = cap.getBackendName()
        self._apply_controls(cap)

        # Discard warm-up frames so exposure settles, then re-assert the
        # controls in case the backend adjusted them at stream start.
        for _ in range(5):
            cap.read()
        self._apply_controls(cap)

        # Always report the REAL runtime properties (not the requested
        # ones): FOURCC decoded, FPS, EXPOSURE - plus focus and sources.
        st = self.status()
        print(f"camera {self.index} [{self._backend_name}]: "
              f"{st.actual_width}x{st.actual_height} "
              f"fourcc={st.fourcc} fps={st.actual_fps:.1f} "
              f"exposure={st.exposure:.1f} focus={st.focus:.0f}")
        print(f"  settings source: index={self._index_src}, "
              f"focus={self.focus} ({self._focus_src}), "
              f"exposure={self.exposure:g} ({self._exposure_src})"
              + ("" if self._focus_src != "default" else
                 "  [focus not tuned yet - run "
                 "'python -m capture.camera --focus-sweep']"))
        return self

    def read(self):
        """Return the next BGR frame, or None if the grab failed."""
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        ok, frame = self._cap.read()
        return frame if ok else None

    def set_focus(self, value: float) -> int:
        """Set manual focus (snapped to the C920 grid). Returns the value."""
        self.focus = snap_focus(value)
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        return self.focus

    def set_exposure(self, value: float) -> float:
        self.exposure = min(max(value, EXPOSURE_MIN), EXPOSURE_MAX)
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        return self.exposure

    def status(self) -> CameraStatus:
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        cap = self._cap

        def tri_state(prop: int) -> bool | None:
            """True = off/manual, False = auto is ON, None = not reported.

            DirectShow reports CameraControl flags (1 = auto, 2 = manual),
            MSMF reports 0/1 booleans, and OpenCV's DSHOW auto-exposure
            uses 0.25 (manual) / 0.75 (auto); -1 means no readback.
            """
            value = cap.get(prop)
            if value < 0:
                return None
            if value in (1.0, 0.75, 3.0):
                return False
            if value in (0.0, 0.25, 2.0):
                return True
            return None

        return CameraStatus(
            requested_width=self.width,
            requested_height=self.height,
            requested_fps=self.fps,
            actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            actual_fps=cap.get(cv2.CAP_PROP_FPS),
            fourcc=_fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            autofocus_off=tri_state(cv2.CAP_PROP_AUTOFOCUS),
            focus=cap.get(cv2.CAP_PROP_FOCUS),
            auto_exposure_off=tri_state(cv2.CAP_PROP_AUTO_EXPOSURE),
            exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
        )

    def measure_fps(self, duration: float = 1.0) -> float:
        """Measure real capture throughput (driver-reported FPS can lie)."""
        if self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        self._cap.read()  # sync to a frame boundary
        start = time.perf_counter()
        frames = 0
        while time.perf_counter() - start < duration:
            ok, _ = self._cap.read()
            if ok:
                frames += 1
        elapsed = time.perf_counter() - start
        return frames / elapsed if elapsed > 0 else 0.0

    def verify(self, measure: bool = True) -> bool:
        """Print requested vs. actual settings and diagnose low fps."""

        def describe(state: bool | None) -> str:
            if state is None:
                return ("off requested (driver does not report state back; "
                        "verify visually)")
            return "off" if state else "ON (should be off)"

        st = self.status()
        real_fps = self.measure_fps() if measure else None
        fps_low = real_fps is not None and real_fps < self.fps - 3
        print(f"resolution         : requested {st.requested_width}x"
              f"{st.requested_height}, actual {st.actual_width}x"
              f"{st.actual_height}"
              + ("" if st.resolution_ok else "   [MISMATCH]"))
        # FOURCC readback is unreliable: DSHOW reports the converted
        # format and MSMF reports nothing (it negotiates MJPEG
        # internally). Measured fps is the ground truth.
        fourcc_note = ""
        if st.fourcc != "MJPG":
            if self._backend_name == "MSMF":
                fourcc_note = ("   (MSMF negotiates internally; see "
                               "measured fps)")
            elif fps_low:
                fourcc_note = "   [WARN: not MJPG and fps is low]"
        print(f"fourcc (runtime)   : {st.fourcc}{fourcc_note}")
        print(f"fps (driver)       : requested {st.requested_fps:.0f}, "
              f"reported {st.actual_fps:.1f}")
        if real_fps is not None:
            print(f"fps (measured)     : {real_fps:.1f}"
                  + ("" if not fps_low else "   [WARN: low]"))
        if self._backend_name == "MSMF":
            # MSMF applies focus/exposure writes (verified live via
            # sharpness/brightness reaction) but its readbacks are static
            # garbage; report the applied values instead.
            print(f"autofocus          : {describe(st.autofocus_off)}"
                  f", focus applied={self.focus} (MSMF readback "
                  "unreliable)")
            print(f"auto-exposure      : {describe(st.auto_exposure_off)}"
                  f", exposure applied={self.exposure:g} (MSMF readback "
                  "unreliable)")
        else:
            print(f"autofocus          : {describe(st.autofocus_off)}"
                  f", focus={st.focus:.0f} (requested {self.focus})")
            print(f"auto-exposure      : {describe(st.auto_exposure_off)}"
                  f", exposure={st.exposure:.1f} "
                  f"(requested {self.exposure:g})")

        if fps_low:
            if st.fourcc != "MJPG":
                print("DIAGNOSIS: the negotiated stream is likely not "
                      "MJPEG. YUY2 at 1080p is capped at ~5 fps by USB "
                      "bandwidth. Try another USB port, close apps "
                      "holding the camera, or lower the resolution to "
                      "isolate the problem.")
            if st.exposure >= -4:
                seconds = 2.0 ** st.exposure
                print(f"DIAGNOSIS: manual exposure is long "
                      f"({seconds * 1000:.0f} ms); it caps the frame rate "
                      f"at ~{1.0 / seconds:.0f} fps. Use a shorter "
                      "exposure (e.g. --exposure -6, or adjust with E/D "
                      "inside --focus-sweep) and add frontal light.")
        if not st.resolution_ok:
            print("ERROR: camera did not accept "
                  f"{st.requested_width}x{st.requested_height}. The C920 "
                  "falls back silently; check the USB port, close other "
                  "apps using the camera, and confirm the right --index "
                  "if the laptop has a built-in webcam.")
        return st.resolution_ok and not fps_low

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def _sharpness(frame: np.ndarray) -> float:
    """Focus metric: Laplacian variance on the central third of the frame."""
    h, w = frame.shape[:2]
    crop = frame[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def focus_sweep(camera: Camera) -> int:
    """Interactive focus (and exposure) tuning. Returns an exit code.

    Keys: W/S focus +/-5, E/D exposure +1/-1, ENTER saves to
    data/camera.yaml, ESC exits without saving. Center your face in the
    frame and maximize the sharpness value.
    """
    window = "gazetrack - focus sweep"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("focus sweep: W/S focus +/-5, E/D exposure +/-1, "
          "ENTER save, ESC cancel")
    sharp_history: list[float] = []
    frame_times: list[float] = []
    while True:
        frame = camera.read()
        if frame is None:
            print("ERROR: frame grab failed.")
            return 1
        now = time.perf_counter()
        frame_times = [t for t in frame_times if now - t < 2.0] + [now]
        fps = len(frame_times) / 2.0

        sharp = _sharpness(frame)
        sharp_history = (sharp_history + [sharp])[-90:]
        peak = max(sharp_history)

        lines = [
            f"focus     {camera.focus:>4}   (W/S, range {FOCUS_MIN}-"
            f"{FOCUS_MAX}, 0=infinity, ~60cm: 25-50)",
            f"exposure  {camera.exposure:>4g}   (E/D, log2 seconds)",
            f"sharpness {sharp:7.1f}   (session peak {peak:7.1f})",
            f"{fps:5.1f} fps    ENTER save    ESC cancel",
        ]
        cv2.rectangle(frame, (0, 0), (620, 22 * len(lines) + 12),
                      _BG, -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 22 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _FG, 1, cv2.LINE_AA)
        # Sharpness bar relative to the session peak: at the best focus the
        # fill reaches the right edge.
        bar_w = 600
        fill = int(bar_w * (sharp / peak)) if peak > 0 else 0
        cv2.rectangle(frame, (10, 22 * len(lines) + 20),
                      (10 + bar_w, 22 * len(lines) + 34), _DIM, 1)
        cv2.rectangle(frame, (10, 22 * len(lines) + 20),
                      (10 + fill, 22 * len(lines) + 34), _FG, -1)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("w"), ord("W")):
            camera.set_focus(camera.focus + FOCUS_STEP)
        elif key in (ord("s"), ord("S")):
            camera.set_focus(camera.focus - FOCUS_STEP)
        elif key in (ord("e"), ord("E")):
            camera.set_exposure(camera.exposure + 1)
        elif key in (ord("d"), ord("D")):
            camera.set_exposure(camera.exposure - 1)
        elif key == 13:  # ENTER
            save_config({"index": camera.index, "focus": camera.focus,
                         "exposure": camera.exposure})
            print(f"saved to {CONFIG_PATH}: index={camera.index}, "
                  f"focus={camera.focus}, exposure={camera.exposure:g}")
            cv2.destroyWindow(window)
            return 0
        elif key == 27:  # ESC
            print("focus sweep cancelled; nothing saved.")
            cv2.destroyWindow(window)
            return 0


def _preview(camera: Camera) -> None:
    window = "gazetrack - camera check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    frame_times: list[float] = []
    while True:
        frame = camera.read()
        if frame is None:
            print("ERROR: frame grab failed.")
            break
        now = time.perf_counter()
        frame_times = [t for t in frame_times if now - t < 2.0] + [now]
        fps = len(frame_times) / 2.0
        h, w = frame.shape[:2]
        label = f"{w}x{h}  {fps:.1f} fps  (q to quit)"
        cv2.rectangle(frame, (0, 0), (360, 34), _BG, -1)
        cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, _FG, 1, cv2.LINE_AA)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify C920 capture settings, tune focus/exposure "
                    "(--focus-sweep), or preview the stream.")
    parser.add_argument("--index", type=int, default=None,
                        help="camera index (default: data/camera.yaml, "
                             "else 0)")
    parser.add_argument("--focus", type=float, default=None,
                        help=f"manual focus {FOCUS_MIN}-{FOCUS_MAX} "
                             "(default: data/camera.yaml, else "
                             f"{DEFAULT_FOCUS})")
    parser.add_argument("--exposure", type=float, default=None,
                        help="manual exposure, log2 seconds (default: "
                             f"data/camera.yaml, else {DEFAULT_EXPOSURE})")
    parser.add_argument("--keep-auto-exposure", action="store_true",
                        help="leave auto-exposure enabled (debug only)")
    parser.add_argument("--focus-sweep", action="store_true",
                        help="interactive focus/exposure tuning; saves to "
                             "data/camera.yaml")
    parser.add_argument("--preview", action="store_true",
                        help="show a live preview window after verification")
    args = parser.parse_args()

    camera = Camera(index=args.index, focus=args.focus,
                    exposure=args.exposure,
                    disable_auto_exposure=not args.keep_auto_exposure)
    try:
        camera.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        ok = camera.verify()
        if args.focus_sweep:
            return focus_sweep(camera)
        if args.preview:
            _preview(camera)
        return 0 if ok else 1
    finally:
        camera.release()


if __name__ == "__main__":
    sys.exit(main())
