"""gazetrack CLI: calibrate / demo / record / evaluate.

This is the only module that wires layer 1 (gaze direction: camera, face
mesh, ETH-XGaze on CUDA) to layer 2 (screen mapping: polynomial regression,
smoothing, overlay). The layers themselves never import each other.

Typical flow: calibrate -> demo -> evaluate. Recalibrate whenever lighting
or sitting position changes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np

from capture.camera import DEFAULT_EXPOSURE, DEFAULT_FOCUS, Camera
from gaze.eth_xgaze import EthXGazeEstimator
from gaze.mediapipe_head import HeadTracker
from mapping.calibrator import (GazeMapper, make_features, run_calibration)
from mapping.smoother import PointSmoother
from overlay import screen as screen_cfg
from overlay.dot import DotOverlay, get_screen_size

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "calibration.yaml"
DATA_DIR = PROJECT_ROOT / "data"


class GazePipeline:
    """Layer-1 sample source shared by every subcommand."""

    def __init__(self, args: argparse.Namespace):
        self.camera = Camera(
            index=args.index, focus=args.focus, exposure=args.exposure,
            disable_auto_exposure=not args.keep_auto_exposure)
        self.tracker = HeadTracker()
        self.estimator = EthXGazeEstimator(device=args.device)

    def open(self) -> "GazePipeline":
        self.camera.open()
        if not self.camera.verify():
            print("WARNING: camera is not in the expected mode; results "
                  "will be degraded.")
        return self

    def sample(self):
        """One frame through layer 1. Returns (obs, gaze) or None."""
        frame = self.camera.read()
        if frame is None:
            return None
        obs = self.tracker.process(frame)
        if obs is None:
            return None
        h, w = frame.shape[:2]
        gaze = self.estimator.estimate(frame, obs,
                                       self.tracker.camera_matrix(w, h))
        return obs, gaze

    def features(self):
        """One frame -> layer-2 feature vector, or None if no face."""
        result = self.sample()
        if result is None:
            return None
        obs, gaze = result
        return make_features(gaze.yaw_deg, gaze.pitch_deg, obs.head_yaw,
                             obs.head_pitch, obs.face_center)

    def close(self) -> None:
        self.camera.release()
        self.tracker.close()

    def __enter__(self) -> "GazePipeline":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def _clamp(value: float, upper: int) -> float:
    return min(max(value, 0.0), float(upper - 1))


def _trace_calibration(cal_path: Path) -> None:
    """Print which calibration was loaded and warn if it predates the last
    camera or screen-geometry change (a stale calibration silently produces
    wrong results; this line is how you tell)."""
    import yaml

    created = "unknown"
    try:
        data = yaml.safe_load(cal_path.read_text(encoding="utf-8"))
        created = (data or {}).get("created", "unknown")
    except (OSError, yaml.YAMLError):
        pass
    print(f"loaded calibration: {cal_path}  (created {created})")

    try:
        cal_mtime = cal_path.stat().st_mtime
    except OSError:
        return
    for name in ("camera.yaml", "screen.yaml"):
        cfg = DATA_DIR / name
        if cfg.exists() and cfg.stat().st_mtime > cal_mtime:
            print(f"WARNING: data/{name} was modified AFTER this "
                  "calibration. Camera or screen geometry changed since you "
                  "calibrated; recalibrate for valid results.")


def _report_geometry(geo: screen_cfg.ScreenGeometry) -> None:
    """Print the detected resolution and resolved physical geometry."""
    print(f"screen: {geo.width_px}x{geo.height_px} px (detected)")
    if geo.configured:
        tall = (f", {geo.height_cm:g} cm tall" if geo.height_cm else "")
        print(f"geometry: {geo.width_cm:g} cm wide{tall}, viewing distance "
              f"{geo.distance_cm:g} cm  (width_cm from "
              f"{geo.sources['width_cm']}, distance_cm from "
              f"{geo.sources['distance_cm']})")
    else:
        missing = [k for k in ("width_cm", "distance_cm")
                   if geo.sources[k] == "unset"]
        print("geometry: NOT CONFIGURED (" + ", ".join(missing) + " unset). "
              "cm and deg will be unavailable; px and %-of-width stay valid. "
              "Set it with: python main.py screen --screen-width-cm <W> "
              "--distance-cm <D>")
    for warning in geo.warnings():
        print(f"WARNING: {warning}")


def _resolve_geometry(args: argparse.Namespace,
                      persist: bool) -> screen_cfg.ScreenGeometry:
    """Resolve geometry (flag > screen.yaml > unset), persist any flags."""
    geo = screen_cfg.resolve(args.screen_width_cm, args.screen_height_cm,
                             args.distance_cm)
    if persist and any(v is not None for v in (args.screen_width_cm,
                                               args.screen_height_cm,
                                               args.distance_cm)):
        screen_cfg.save_config({"width_cm": args.screen_width_cm,
                                "height_cm": args.screen_height_cm,
                                "distance_cm": args.distance_cm})
        print(f"(geometry flags saved to {screen_cfg.CONFIG_PATH})")
    _report_geometry(geo)
    return geo


def cmd_calibrate(args: argparse.Namespace) -> int:
    geo = _resolve_geometry(args, persist=True)
    with GazePipeline(args) as pipeline:
        result = run_calibration(
            sample_provider=pipeline.features,
            screen_size=(geo.width_px, geo.height_px),
            output_path=Path(args.calibration),
            screen_width_cm=geo.width_cm,
            screen_height_cm=geo.height_cm,
            distance_cm=geo.distance_cm)
    return 0 if result is not None else 1


def cmd_screen(args: argparse.Namespace) -> int:
    provided = any(v is not None for v in (args.screen_width_cm,
                                           args.screen_height_cm,
                                           args.distance_cm))
    if provided:
        screen_cfg.save_config({"width_cm": args.screen_width_cm,
                                "height_cm": args.screen_height_cm,
                                "distance_cm": args.distance_cm})
        print(f"saved to {screen_cfg.CONFIG_PATH}")
    geo = screen_cfg.resolve()
    _report_geometry(geo)
    if not provided and not geo.configured:
        print("Tip: python main.py screen --screen-width-cm 120 "
              "--screen-height-cm 34 --distance-cm 70")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    mapper = GazeMapper.load(Path(args.calibration))
    _trace_calibration(Path(args.calibration))
    screen = get_screen_size()
    print(f"screen: {screen[0]}x{screen[1]} px (detected)")
    smoother = PointSmoother(min_cutoff=args.min_cutoff, beta=args.beta)
    with GazePipeline(args) as pipeline, \
            DotOverlay(screen) as overlay:
        while True:
            features = pipeline.features()
            if features is None:
                key = overlay.render(None, None, "q to quit")
            else:
                raw_x, raw_y = mapper.predict(features)
                x, y = smoother.filter(_clamp(raw_x, screen[0]),
                                       _clamp(raw_y, screen[1]))
                key = overlay.render(x, y, "q to quit")
            if key in (ord("q"), 27):
                return 0


def cmd_record(args: argparse.Namespace) -> int:
    import cv2

    mapper = GazeMapper.load(Path(args.calibration))
    _trace_calibration(Path(args.calibration))
    screen = get_screen_size()
    smoother = PointSmoother(min_cutoff=args.min_cutoff, beta=args.beta)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"session_{stamp}.csv"
    columns = ["timestamp", "t_monotonic", "gaze_yaw_deg", "gaze_pitch_deg",
               "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
               "face_x_mm", "face_y_mm", "face_z_mm",
               "screen_x_raw", "screen_y_raw",
               "screen_x_smooth", "screen_y_smooth"]

    window = "gazetrack - recording"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 420, 120)
    rows = 0
    started = time.perf_counter()
    print(f"recording to {out_path} (q or ESC in the status window stops)")
    with GazePipeline(args) as pipeline, \
            open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        while True:
            result = pipeline.sample()
            now = time.perf_counter()
            if result is not None:
                obs, gaze = result
                features = make_features(gaze.yaw_deg, gaze.pitch_deg,
                                         obs.head_yaw, obs.head_pitch,
                                         obs.face_center)
                raw_x, raw_y = mapper.predict(features)
                raw_x = _clamp(raw_x, screen[0])
                raw_y = _clamp(raw_y, screen[1])
                smooth_x, smooth_y = smoother.filter(raw_x, raw_y, now)
                writer.writerow([
                    _dt.datetime.now().isoformat(timespec="milliseconds"),
                    f"{now:.4f}", f"{gaze.yaw_deg:.3f}",
                    f"{gaze.pitch_deg:.3f}", f"{obs.head_yaw:.3f}",
                    f"{obs.head_pitch:.3f}", f"{obs.head_roll:.3f}",
                    f"{obs.face_center[0]:.1f}", f"{obs.face_center[1]:.1f}",
                    f"{obs.face_center[2]:.1f}", f"{raw_x:.1f}",
                    f"{raw_y:.1f}", f"{smooth_x:.1f}", f"{smooth_y:.1f}"])
                rows += 1

            panel = np.full((120, 420, 3), 18, dtype=np.uint8)
            elapsed = now - started
            lines = [f"recording  {elapsed:6.1f} s   {rows} samples",
                     "face: ok" if result is not None else "face: lost",
                     "q to stop"]
            for i, line in enumerate(lines):
                cv2.putText(panel, line, (16, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235),
                            1, cv2.LINE_AA)
            cv2.imshow(window, panel)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    cv2.destroyWindow(window)
    print(f"saved {rows} samples to {out_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from eval.grid import run_evaluation

    mapper = GazeMapper.load(Path(args.calibration))
    _trace_calibration(Path(args.calibration))
    geo = _resolve_geometry(args, persist=True)
    screen = (geo.width_px, geo.height_px)

    def predict_provider():
        features = pipeline.features()
        if features is None:
            return None
        # RAW prediction, NOT clamped: evaluate measures the true model
        # output so a runaway magnitude cannot hide behind the screen clamp.
        x, y = mapper.predict(features)
        return float(x), float(y)

    with GazePipeline(args) as pipeline:
        report = run_evaluation(
            predict_provider=predict_provider,
            screen_size=screen,
            screen_width_cm=geo.width_cm,
            distance_cm=geo.distance_cm,
            screen_height_cm=geo.height_cm,
            output_dir=DATA_DIR)
    return 0 if report is not None else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gazetrack",
        description="Webcam gaze tracking: on-screen dot that follows "
                    "the user's gaze (calibrate -> demo -> evaluate).")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        # index/focus/exposure default to None so data/camera.yaml (written
        # by 'python -m capture.camera --focus-sweep') takes effect; an
        # explicit flag always wins.
        p.add_argument("--index", type=int, default=None,
                       help="camera index (default: data/camera.yaml, "
                            "else 0)")
        p.add_argument("--device", default="cuda",
                       help="inference device (default: cuda)")
        p.add_argument("--focus", type=float, default=None,
                       help="C920 manual focus 0-255, step 5 (default: "
                            f"data/camera.yaml, else {DEFAULT_FOCUS})")
        p.add_argument("--exposure", type=float, default=None,
                       help="manual exposure, log2 seconds (default: "
                            f"data/camera.yaml, else {DEFAULT_EXPOSURE})")
        p.add_argument("--keep-auto-exposure", action="store_true",
                       help="leave auto-exposure on (debug only)")
        p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                       help="calibration YAML path")

    def geometry(p: argparse.ArgumentParser) -> None:
        # No silent cm default: geometry comes from data/screen.yaml (set
        # once via 'python main.py screen ...'); a flag overrides and is
        # persisted. Unset geometry is reported, never guessed.
        p.add_argument("--screen-width-cm", type=float, default=None,
                       help="physical screen width in cm "
                            "(default: data/screen.yaml)")
        p.add_argument("--screen-height-cm", type=float, default=None,
                       help="physical screen height in cm "
                            "(default: data/screen.yaml)")
        p.add_argument("--distance-cm", type=float, default=None,
                       help="eye-to-screen distance in cm "
                            "(default: data/screen.yaml)")

    def smoothing(p: argparse.ArgumentParser) -> None:
        p.add_argument("--min-cutoff", type=float, default=1.0,
                       help="One-Euro min cutoff Hz (lower = steadier)")
        p.add_argument("--beta", type=float, default=0.007,
                       help="One-Euro speed coefficient (higher = snappier)")

    p = sub.add_parser("screen",
                       help="show or set physical screen geometry "
                            "(data/screen.yaml)")
    geometry(p)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("calibrate",
                       help="run the 33-target calibration routine")
    common(p); geometry(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("demo", help="live gaze dot (requires calibration)")
    common(p); smoothing(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("record",
                       help="record a gaze session to CSV with timestamps")
    common(p); smoothing(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("evaluate",
                       help="5x5 grid accuracy report (angular and cm, "
                            "by zone)")
    common(p); geometry(p)
    p.set_defaults(func=cmd_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
