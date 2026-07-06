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
from mapping.pursuit import DEFAULT_SPEED_PX_S, SERPENTINE_PASSES
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

    def read_state(self):
        """One frame with an explicit status for robustness:
        ('no_frame', None, None) on a failed grab (camera gone),
        ('no_face', None, None) when no face is detected,
        ('ok', obs, gaze) otherwise. Lets the overlay worker tell a camera
        disconnect apart from the user simply looking away."""
        frame = self.camera.read()
        if frame is None:
            return "no_frame", None, None
        obs = self.tracker.process(frame)
        if obs is None:
            return "no_face", None, None
        h, w = frame.shape[:2]
        gaze = self.estimator.estimate(frame, obs,
                                       self.tracker.camera_matrix(w, h))
        return "ok", obs, gaze

    def features(self):
        """One frame -> layer-2 feature vector, or None if no face."""
        result = self.sample()
        if result is None:
            return None
        obs, gaze = result
        return make_features(gaze.yaw_deg, gaze.pitch_deg, obs.head_yaw,
                             obs.head_pitch)

    def features_ts(self):
        """(features | None, grab_time) for latency-sensitive sampling.

        grab_time is stamped at the camera grab, BEFORE MediaPipe/CNN, so
        smooth-pursuit lag reflects physiological + display latency and not
        this pipeline's own processing time.
        """
        frame = self.camera.read()
        grab_t = self.camera.last_grab_time
        if frame is None:
            return None, grab_t
        obs = self.tracker.process(frame)
        if obs is None:
            return None, grab_t
        h, w = frame.shape[:2]
        gaze = self.estimator.estimate(frame, obs,
                                       self.tracker.camera_matrix(w, h))
        return make_features(gaze.yaw_deg, gaze.pitch_deg, obs.head_yaw,
                             obs.head_pitch), grab_t

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
    if args.pursuit:
        # PHASE-01B (c)/(d): combined session. Discrete grid FIRST (fresh,
        # high-quality anchors), then smooth pursuit (dense coverage), then
        # a degree/alpha model chosen by block-CV. --pursuit-csv reuses a
        # prior pursuit stream instead of recollecting it.
        from mapping.calibrator import collect_grid, fit_combined
        from mapping.pursuit import load_pursuit_result, run_pursuit
        screen = (geo.width_px, geo.height_px)
        with GazePipeline(args) as pipeline:
            print("=== combined calibration: discrete grid, then pursuit ===")
            grid = collect_grid(pipeline.features, screen)
            if grid is None:
                return 1
            if args.pursuit_csv:
                pursuit = load_pursuit_result(
                    Path(args.pursuit_csv), screen_size=screen,
                    px_per_cm=geo.px_per_cm, distance_cm=geo.distance_cm)
            else:
                pursuit = run_pursuit(
                    sample_provider=pipeline.features_ts, screen_size=screen,
                    px_per_cm=geo.px_per_cm, distance_cm=geo.distance_cm,
                    output_dir=DATA_DIR, speed_px_s=args.pursuit_speed,
                    passes=args.pursuit_passes, report=False)
            if pursuit is None:
                return 1
        result = fit_combined(
            grid, pursuit.features, pursuit.targets_px, pursuit.weights,
            screen, Path(args.calibration),
            pursuit_meta={"best_lag_ms": pursuit.best_lag_ms,
                          "discard": pursuit.discard.as_dict()},
            screen_width_cm=geo.width_cm, screen_height_cm=geo.height_cm,
            distance_cm=geo.distance_cm)
        return 0 if result is not None else 1
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


def cmd_record_background(args: argparse.Namespace) -> int:
    """PHASE-02B: overlay (optional) + behavior logging to SQLite, in the
    tray. All local; keyboard/mouse are per-interval COUNTS only."""
    from overlay.config import resolve as resolve_overlay
    from overlay.gaze_overlay import OverlayApp

    mapper = GazeMapper.load(Path(args.calibration))
    _trace_calibration(Path(args.calibration))
    screen = get_screen_size()
    print(f"screen: {screen[0]}x{screen[1]} px (detected)")
    cfg = resolve_overlay({})   # overlay.yaml or defaults; no per-run overrides
    db_path = Path(args.db) if args.db else (DATA_DIR / "activity.db")
    app = OverlayApp(
        pipeline_factory=lambda: GazePipeline(args),
        mapper=mapper, screen_size=screen, config=cfg,
        use_tray=not args.no_tray, enable_logging=True, db_path=db_path,
        no_titles=args.no_titles, start_hidden=args.hide_overlay)
    return app.run(exit_after=args.exit_after)


def cmd_record(args: argparse.Namespace) -> int:
    if getattr(args, "background", False):
        return cmd_record_background(args)
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
                                         obs.head_yaw, obs.head_pitch)
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


def _parse_color(text: str) -> tuple:
    """'r,g,b' (0-255) -> tuple; falls back to neutral off-white."""
    try:
        parts = tuple(int(v) for v in text.split(","))
        if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
            return parts
    except ValueError:
        pass
    print(f"WARNING: could not parse --dot-color '{text}'; using default.")
    return (235, 235, 228)


def cmd_overlay(args: argparse.Namespace) -> int:
    from overlay.config import resolve as resolve_overlay
    from overlay.gaze_overlay import OverlayApp

    mapper = GazeMapper.load(Path(args.calibration))
    _trace_calibration(Path(args.calibration))
    screen = get_screen_size()
    print(f"screen: {screen[0]}x{screen[1]} px (detected)")

    # Precedence flag > overlay.yaml > default; None means the flag was absent.
    color = _parse_color(args.dot_color) if args.dot_color else None
    cfg = resolve_overlay({
        "dot_size": args.dot_size, "dot_opacity": args.dot_opacity,
        "dot_color": color, "deadzone": args.deadzone,
        "min_cutoff": args.min_cutoff, "beta": args.beta,
        "pose_margin": args.pose_margin,
        "offscreen_margin": args.offscreen_margin, "fade_ms": args.fade_ms})
    app = OverlayApp(
        pipeline_factory=lambda: GazePipeline(args),
        mapper=mapper, screen_size=screen, config=cfg,
        use_tray=not args.no_tray)
    return app.run(exit_after=args.exit_after)


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
    p.add_argument("--pursuit", action="store_true",
                   help="combined calibration: discrete grid + smooth "
                        "pursuit, degree/alpha chosen by block-CV")
    p.add_argument("--pursuit-csv", default=None,
                   help="reuse a saved pursuit stream (data/pursuit_*.csv) "
                        "instead of recollecting it")
    p.add_argument("--pursuit-speed", type=float, default=DEFAULT_SPEED_PX_S,
                   help="pursuit ball speed in px/s "
                        f"(default: {DEFAULT_SPEED_PX_S:.0f}; ~8-9 deg/s on "
                        "the reference rig, deliberately slow)")
    p.add_argument("--pursuit-passes", type=int, default=SERPENTINE_PASSES,
                   help="serpentine horizontal passes "
                        f"(default: {SERPENTINE_PASSES})")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("demo", help="live gaze dot (requires calibration)")
    common(p); smoothing(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("overlay",
                       help="transparent click-through gaze overlay over "
                            "your real apps (Tobii-style)")
    common(p)
    # All tunables default None so precedence is flag > data/overlay.yaml >
    # built-in default (resolved in overlay.config). They are also editable
    # live from the settings panel (Ctrl+Alt+S).
    p.add_argument("--dot-size", type=int, default=None,
                   help="ring outer diameter in px (default: 40)")
    p.add_argument("--dot-opacity", type=float, default=None,
                   help="peak ring opacity 0-1 (default: 0.85)")
    p.add_argument("--dot-color", default=None,
                   help="ring color as r,g,b 0-255 (default neutral)")
    p.add_argument("--deadzone", type=float, default=None,
                   help="cursor holds still under this px move (default: 12)")
    p.add_argument("--min-cutoff", type=float, default=None,
                   help="One-Euro min cutoff Hz (default: 1.0)")
    p.add_argument("--beta", type=float, default=None,
                   help="One-Euro speed coefficient (default: 0.007)")
    p.add_argument("--pose-margin", type=float, default=None,
                   help="degrees beyond the calibrated head-pose range "
                        "before the cursor fades (default: 5)")
    p.add_argument("--offscreen-margin", type=float, default=None,
                   help="fade when the prediction is this fraction of screen "
                        "width outside the screen (default: 0.10)")
    p.add_argument("--fade-ms", type=int, default=None,
                   help="fade in/out duration in ms (default: 250)")
    p.add_argument("--no-tray", action="store_true",
                   help="do not create a system-tray icon")
    p.add_argument("--exit-after", type=float, default=None,
                   help="auto-quit after N seconds (measurement/testing)")
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("record",
                       help="record a gaze session to CSV, or "
                            "--background for behavior logging (2B)")
    common(p); smoothing(p)
    p.add_argument("--background", action="store_true",
                   help="PHASE-02B: run the overlay + behavior logging to "
                        "SQLite, in the system tray")
    p.add_argument("--no-titles", action="store_true",
                   help="privacy: log the process only, never window titles")
    p.add_argument("--hide-overlay", action="store_true",
                   help="run with the gaze cursor hidden (logging still runs)")
    p.add_argument("--db", default=None,
                   help="activity DB path (default: data/activity.db)")
    p.add_argument("--no-tray", action="store_true",
                   help="run without a system-tray icon")
    p.add_argument("--exit-after", type=float, default=None,
                   help="auto-quit after N seconds (measurement/testing)")
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
