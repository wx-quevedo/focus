"""Smooth-pursuit calibration data collection (layer 2, additive to the grid).

A single ball travels a fixed polyline (serpentine + perimeter + two
diagonals) at constant speed while gaze is sampled every frame. This yields
one to two orders of magnitude more (features -> screen) pairs than the 33
discrete targets, which is the main data lever of PHASE-01B.

Design notes:
- Both directions of each axis are covered so the residual latency bias
  averages out (the serpentine sweeps x both ways; the perimeter and the two
  diagonals sweep y both ways).
- SPEED: the default 350 px/s is DELIBERATELY comfortable. On this rig
  (32 px/cm, 75 cm viewing distance) that is only ~8-9 deg/s of angular
  velocity, well below the ~30 deg/s ceiling where smooth pursuit breaks into
  catch-up saccades. Do NOT "correct" the speed upward believing the target
  is 30 deg/s - the target is slow-and-smooth so the eye tracks cleanly.
- LATENCY: the eye lags the target by ~100-150 ms. Each gaze sample at time t
  is paired with the target position evaluated analytically at (t - lag) via
  the trajectory's parametric formula, NOT the nearest recorded frame (with a
  25 ms lag grid and 33 ms frames, nearest-frame matching would alias). The
  lag is chosen from a 0-250 ms grid by block-CV error.

Collection (the OpenCV UI) is separated from processing (pure filtering +
lag selection) so the processing can be tested headless.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from mapping.calibrator import FEATURE_NAMES, block_cv_error_px

DEFAULT_SPEED_PX_S = 350.0
SERPENTINE_PASSES = 6
TRAJECTORY_MARGIN = 0.06

# Discard the first REENGAGE_S after each trajectory direction change: the eye
# needs to re-lock onto the new heading (a catch-up saccade), which is not
# smooth pursuit.
REENGAGE_S = 0.30

# Saccade rejection (combined threshold): discard a sample whose gaze angular
# speed exceeds max(3x the target's angular speed, an 80 deg/s absolute
# floor). The floor matters because on slow trajectory stretches 3x the target
# speed would be so small it filters nothing.
SACCADE_TARGET_MULT = 3.0
SACCADE_FLOOR_DEG_S = 80.0

# Latency search grid and the fixed model used only to score lags (the final
# degree/alpha are chosen later, in the model-selection step). The grid runs
# to 450 ms: total lag = physiological pursuit (~100-150 ms) + display and
# residual pipeline latency. On the reference rig the minimum sat at 250 ms
# with the post-processing timestamp; grab-time stamping should lower it.
LAG_GRID_MS = list(range(0, 451, 25))
LAG_SEARCH_DEGREE = 2
LAG_SEARCH_ALPHA = 1.0
CV_BLOCKS = 5

_BG = (18, 18, 18)
_FG = (235, 235, 235)
_DIM = (110, 110, 110)


class Trajectory:
    """Constant-speed polyline with an exact parametric position(t)."""

    def __init__(self, waypoints: np.ndarray, speed_px_s: float):
        pts = [waypoints[0]]
        for p in waypoints[1:]:
            if np.hypot(*(np.asarray(p) - pts[-1])) > 1e-6:
                pts.append(np.asarray(p, dtype=float))
        self.waypoints = np.asarray(pts, dtype=float)
        self.speed = speed_px_s
        seg = np.diff(self.waypoints, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum_time = np.concatenate([[0.0], np.cumsum(seg_len)]) / speed_px_s
        self.total_duration = float(self.cum_time[-1])
        # Interior vertices are the direction-change events.
        self.vertex_times = self.cum_time[1:-1].copy()

    def position_at(self, t: float) -> np.ndarray:
        t = min(max(t, 0.0), self.total_duration)
        i = int(np.searchsorted(self.cum_time, t, side="right") - 1)
        i = min(max(i, 0), len(self.waypoints) - 2)
        t0, t1 = self.cum_time[i], self.cum_time[i + 1]
        frac = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return self.waypoints[i] + frac * (self.waypoints[i + 1]
                                           - self.waypoints[i])


def build_trajectory(width: int, height: int, speed_px_s: float,
                     passes: int = SERPENTINE_PASSES,
                     margin: float = TRAJECTORY_MARGIN) -> Trajectory:
    x0, x1 = margin * width, (1 - margin) * width
    y0, y1 = margin * height, (1 - margin) * height
    wp: list[tuple[float, float]] = []

    # Serpentine horizontal sweep, top to bottom, alternating x direction.
    for i, y in enumerate(np.linspace(y0, y1, passes)):
        wp += ([(x0, y), (x1, y)] if i % 2 == 0 else [(x1, y), (x0, y)])

    # Perimeter: covers all four edges; the left edge is walked upward, giving
    # the vertical axis a second direction.
    wp += [(x0, y1), (x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # Two diagonals, one descending and one ascending.
    wp += [(x0, y0), (x1, y1)]
    wp += [(x0, y1), (x1, y0)]

    return Trajectory(np.array(wp, dtype=float), speed_px_s)


@dataclass
class RawSample:
    t: float
    features: np.ndarray | None  # v2 feature vector, or None on blink/no-face
    ball_x: float
    ball_y: float


@dataclass
class DiscardStats:
    total: int
    kept: int
    blink: int
    saccade: int
    reengage: int

    def as_dict(self) -> dict:
        def pct(n):
            return round(100.0 * n / self.total, 1) if self.total else 0.0
        return {"total": self.total, "kept": self.kept,
                "kept_pct": pct(self.kept),
                "blink": self.blink, "blink_pct": pct(self.blink),
                "saccade": self.saccade, "saccade_pct": pct(self.saccade),
                "reengage": self.reengage, "reengage_pct": pct(self.reengage)}


@dataclass
class PursuitResult:
    features: np.ndarray       # (N, 4) kept feature vectors, time-ordered
    targets_px: np.ndarray     # (N, 2) ball position at (t - best_lag)
    weights: np.ndarray        # (N,) all ones (relative weighting is step c)
    sample_times: np.ndarray   # (N,) acquisition times
    best_lag_ms: int
    cv_error_px: float
    lag_scores: dict           # lag_ms -> block-CV error px
    discard: DiscardStats
    saccade_threshold_deg_s: float


def _gaze_velocity_deg_s(raw: list[RawSample]) -> dict[int, float]:
    """Gaze angular speed (deg/s) per non-blink sample vs the previous
    non-blink sample. A blink breaks the chain (no velocity across it)."""
    vel: dict[int, float] = {}
    prev: tuple[float, float, float] | None = None
    for i, s in enumerate(raw):
        if s.features is None:
            prev = None
            continue
        yaw, pitch = float(s.features[0]), float(s.features[1])
        if prev is not None:
            pt, pyaw, ppitch = prev
            dt = s.t - pt
            if dt > 1e-4:
                vel[i] = np.hypot(yaw - pyaw, pitch - ppitch) / dt
        prev = (s.t, yaw, pitch)
    return vel


def process_pursuit(raw: list[RawSample], trajectory: Trajectory,
                    px_per_cm: float | None, distance_cm: float | None,
                    speed_px_s: float,
                    lag_grid_ms: list[int] = LAG_GRID_MS) -> PursuitResult:
    """Filter the raw stream (blink / saccade / re-engagement), pick the
    latency lag by block-CV, and return the kept pursuit dataset."""
    if px_per_cm and distance_cm:
        target_ang_deg_s = np.degrees((speed_px_s / px_per_cm) / distance_cm)
    else:
        target_ang_deg_s = 0.0
    sac_thresh = max(SACCADE_TARGET_MULT * target_ang_deg_s,
                     SACCADE_FLOOR_DEG_S)

    gaze_vel = _gaze_velocity_deg_s(raw)
    vertex_times = trajectory.vertex_times

    kept: list[int] = []
    blink = saccade = reengage = 0
    for i, s in enumerate(raw):
        if s.features is None:
            blink += 1
            continue
        if vertex_times.size and np.any((s.t - vertex_times >= 0)
                                        & (s.t - vertex_times < REENGAGE_S)):
            reengage += 1
            continue
        v = gaze_vel.get(i)
        if v is not None and v > sac_thresh:
            saccade += 1
            continue
        kept.append(i)

    feats = np.array([raw[i].features for i in kept], dtype=float) \
        if kept else np.empty((0, len(FEATURE_NAMES)))
    times = np.array([raw[i].t for i in kept], dtype=float)
    weights = np.ones(len(kept))

    lag_scores: dict[int, float] = {}
    if len(kept) >= 2 * CV_BLOCKS:
        for lag_ms in lag_grid_ms:
            lag = lag_ms / 1000.0
            tgt = np.array([trajectory.position_at(t - lag) for t in times])
            lag_scores[lag_ms] = block_cv_error_px(
                feats, tgt, weights, LAG_SEARCH_DEGREE, LAG_SEARCH_ALPHA,
                CV_BLOCKS)
    best_lag_ms = min(lag_scores, key=lag_scores.get) if lag_scores else 0
    best_targets = np.array([trajectory.position_at(t - best_lag_ms / 1000.0)
                             for t in times]) if len(times) else \
        np.empty((0, 2))

    return PursuitResult(
        features=feats, targets_px=best_targets, weights=weights,
        sample_times=times, best_lag_ms=best_lag_ms,
        cv_error_px=lag_scores.get(best_lag_ms, float("inf")),
        lag_scores=lag_scores,
        discard=DiscardStats(total=len(raw), kept=len(kept), blink=blink,
                             saccade=saccade, reengage=reengage),
        saccade_threshold_deg_s=float(sac_thresh))


def _wait_for_start(window: str, screen_w: int, screen_h: int) -> bool:
    canvas = np.full((screen_h, screen_w, 3), _BG[0], dtype=np.uint8)
    lines = [
        "Smooth pursuit calibration",
        "",
        "Follow the moving ball with your eyes and head, naturally,",
        "as if watching something drift across the screen.",
        "Track it smoothly - do not jump ahead or look away.",
        "Keep the same seating position you will use afterwards.",
        "",
        "SPACE to start    ESC to abort",
    ]
    y = screen_h // 2 - 100
    for line in lines:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(canvas, line, ((screen_w - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _FG, 1, cv2.LINE_AA)
        y += 34
    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            return True
        if key == 27:
            return False


def collect_pursuit(sample_provider, screen_size: tuple[int, int],
                    trajectory: Trajectory) -> list[RawSample] | None:
    """Run the moving-ball UI and sample gaze each frame. Returns the raw
    stream, or None if the user aborts.

    sample_provider() must return (features | None, grab_time), where
    grab_time is a perf_counter timestamp taken AT the camera grab (see
    GazePipeline.features_ts). Sample time is grab_time - start so the
    latency estimate excludes this loop's processing time.
    """
    screen_w, screen_h = screen_size
    window = "gazetrack - pursuit calibration"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    raw: list[RawSample] = []
    try:
        if not _wait_for_start(window, screen_w, screen_h):
            print("Pursuit collection aborted.")
            return None
        start = time.perf_counter()
        while True:
            features, grab_t = sample_provider()  # blocks at camera rate
            t = grab_t - start
            if t >= trajectory.total_duration:
                break
            bx, by = trajectory.position_at(t)
            raw.append(RawSample(t, features, float(bx), float(by)))

            canvas = np.full((screen_h, screen_w, 3), _BG[0], dtype=np.uint8)
            center = (int(round(bx)), int(round(by)))
            cv2.circle(canvas, center, 16, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.circle(canvas, center, 9, _FG, -1, cv2.LINE_AA)
            if features is None:
                cv2.circle(canvas, center, 20, (60, 60, 220), 1, cv2.LINE_AA)
            progress = 100.0 * t / trajectory.total_duration
            cv2.putText(canvas, f"tracking {progress:4.0f}%    ESC to abort",
                        (24, screen_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        _DIM, 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            if (cv2.waitKey(1) & 0xFF) == 27:
                print("Pursuit collection aborted.")
                return None
    finally:
        cv2.destroyWindow(window)
    return raw


def save_raw_csv(raw: list[RawSample], kept_times: set[float],
                 path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", *FEATURE_NAMES, "ball_x", "ball_y", "kept"])
        for s in raw:
            feat = s.features if s.features is not None else [""] * 4
            writer.writerow([f"{s.t:.4f}", *[
                (f"{v:.4f}" if v != "" else "") for v in feat],
                f"{s.ball_x:.1f}", f"{s.ball_y:.1f}",
                1 if (s.features is not None and s.t in kept_times) else 0])


def _report(result: PursuitResult, trajectory: Trajectory, screen_w: int,
            speed_px_s: float, px_per_cm: float | None,
            distance_cm: float | None, note: bool = True) -> None:
    d = result.discard
    have_geo = bool(px_per_cm and distance_cm)
    ang = np.degrees((speed_px_s / px_per_cm) / distance_cm) \
        if have_geo else None

    def as_extra(err_px: float) -> str:
        parts = [f"{err_px / screen_w * 100:.2f}% width"]
        if have_geo:
            parts.append(f"~{np.degrees(np.arctan2(err_px / px_per_cm, distance_cm)):.2f} deg")
        return " (" + ", ".join(parts) + ")"

    def pct(n: int) -> float:
        return 100.0 * n / max(d.total, 1)

    print("\npursuit collection (phase b) - collection only, not yet "
          "integrated" if note else "\npursuit summary (reused stream)")
    print(f"trajectory         : {trajectory.total_duration:.0f} s, "
          f"{len(trajectory.waypoints)} waypoints, {speed_px_s:.0f} px/s"
          + (f" (~{ang:.1f} deg/s)" if ang is not None else ""))
    print(f"raw samples        : {d.total}")
    print(f"discarded          : blink {d.blink} ({pct(d.blink):.1f}%), "
          f"saccade {d.saccade} ({pct(d.saccade):.1f}%), "
          f"re-engagement {d.reengage} ({pct(d.reengage):.1f}%)")
    print(f"kept               : {d.kept} ({pct(d.kept):.1f}%)")
    print(f"saccade threshold  : {result.saccade_threshold_deg_s:.0f} deg/s "
          "(max of 3x target, 80 floor)")
    if result.lag_scores:
        print(f"latency lag (CV)   : {result.best_lag_ms} ms "
              f"(grid {LAG_GRID_MS[0]}-{LAG_GRID_MS[-1]} ms)")
        print(f"pursuit block-CV   : {result.cv_error_px:.1f} px"
              f"{as_extra(result.cv_error_px)}  (pursuit-only, degree "
              f"{LAG_SEARCH_DEGREE}, block {CV_BLOCKS}-fold)")
        scan = "  ".join(f"{k}:{v:.0f}" for k, v in
                         sorted(result.lag_scores.items()))
        print(f"lag scan (px)      : {scan}")
    else:
        print("latency lag (CV)   : not computed (too few kept samples)")
    if note:
        print("\nNOTE: phase (b) is collection only. The pursuit data is "
              "saved but not yet combined with the discrete grid or used to "
              "fit a calibration; that is the next step.")


def save_pursuit_meta(path: Path, screen_size: tuple[int, int],
                      speed_px_s: float, passes: int,
                      px_per_cm: float | None,
                      distance_cm: float | None) -> None:
    """Sidecar with the exact trajectory parameters so the CSV can be replayed
    (the analytical trajectory must be reconstructed to re-fit or re-select
    the lag)."""
    path.write_text(yaml.safe_dump({
        "screen_w": screen_size[0], "screen_h": screen_size[1],
        "speed_px_s": speed_px_s, "passes": passes,
        "px_per_cm": px_per_cm, "distance_cm": distance_cm,
    }, sort_keys=False), encoding="utf-8")


def load_pursuit_result(csv_path: Path,
                        screen_size: tuple[int, int] | None = None,
                        px_per_cm: float | None = None,
                        distance_cm: float | None = None
                        ) -> PursuitResult | None:
    """Rebuild a PursuitResult from a saved CSV (reuse without recollecting).

    Trajectory parameters come from the sidecar meta if present, else from the
    passed geometry plus the module defaults (with a warning). Re-runs the
    quality filtering and lag selection so a wider lag grid takes effect.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: pursuit CSV not found: {csv_path}")
        return None
    meta_path = csv_path.with_suffix(".meta.yaml")
    if meta_path.exists():
        m = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        sw, sh = int(m["screen_w"]), int(m["screen_h"])
        speed, passes = float(m["speed_px_s"]), int(m["passes"])
        px_per_cm = m.get("px_per_cm", px_per_cm)
        distance_cm = m.get("distance_cm", distance_cm)
    else:
        if screen_size is None:
            print("ERROR: no sidecar meta and no screen_size; cannot rebuild "
                  "the trajectory.")
            return None
        sw, sh = screen_size
        speed, passes = DEFAULT_SPEED_PX_S, SERPENTINE_PASSES
        print(f"WARNING: no sidecar meta for {csv_path.name}; assuming "
              f"speed {speed:.0f} px/s, {passes} passes, {sw}x{sh}. "
              "Verify these match the collection.")

    trajectory = build_trajectory(sw, sh, speed, passes)
    raw: list[RawSample] = []
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if r["gaze_yaw"] == "":
            feats = None
        else:
            feats = np.array([float(r["gaze_yaw"]), float(r["gaze_pitch"]),
                              float(r["head_yaw"]), float(r["head_pitch"])])
        raw.append(RawSample(float(r["t"]), feats, float(r["ball_x"]),
                             float(r["ball_y"])))
    result = process_pursuit(raw, trajectory, px_per_cm, distance_cm, speed)
    _report(result, trajectory, sw, speed, px_per_cm, distance_cm, note=False)
    print(f"reused pursuit stream from {csv_path}")
    return result


def run_pursuit(sample_provider, screen_size: tuple[int, int],
                px_per_cm: float | None, distance_cm: float | None,
                output_dir: Path, speed_px_s: float = DEFAULT_SPEED_PX_S,
                passes: int = SERPENTINE_PASSES,
                report: bool = True) -> PursuitResult | None:
    """Orchestrate a pursuit collection: build trajectory, collect, process,
    persist (CSV + sidecar meta). Returns the result, or None on abort.
    When report is False the standalone phase-b summary is suppressed (the
    combined calibration prints its own)."""
    screen_w, screen_h = screen_size
    trajectory = build_trajectory(screen_w, screen_h, speed_px_s, passes)
    print(f"pursuit trajectory : {trajectory.total_duration:.0f} s at "
          f"{speed_px_s:.0f} px/s ({len(trajectory.waypoints)} waypoints)")
    if not (px_per_cm and distance_cm):
        print("WARNING: screen geometry not configured; saccade rejection "
              "uses only the 80 deg/s floor. Set 'python main.py screen ...' "
              "for the target-relative threshold.")

    raw = collect_pursuit(sample_provider, screen_size, trajectory)
    if raw is None:
        return None

    result = process_pursuit(raw, trajectory, px_per_cm, distance_cm,
                             speed_px_s)
    if report:
        _report(result, trajectory, screen_w, speed_px_s, px_per_cm,
                distance_cm)

    kept_times = set(result.sample_times.tolist())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = Path(output_dir) / f"pursuit_{stamp}.csv"
    save_raw_csv(raw, kept_times, out_path)
    save_pursuit_meta(out_path.with_suffix(".meta.yaml"), screen_size,
                      speed_px_s, passes, px_per_cm, distance_cm)
    print(f"raw pursuit stream saved to {out_path}")
    return result
