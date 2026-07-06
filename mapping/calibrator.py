"""Screen-mapping calibration (layer 2).

Learns (gaze yaw/pitch, head pose, face position) -> (screen_x, screen_y)
with a second-degree polynomial regression (scikit-learn PolynomialFeatures
+ Ridge), weighted so edge and corner calibration points count more than
central ones (webcam gaze degrades toward the periphery; the extra weight
mitigates it).

Layer separation: this module consumes layer-1 OUTPUTS only (angles in
degrees, positions in millimetres). It never touches camera frames or the
CNN. The interactive routine receives a `sample_provider` callable that the
caller (main.py) wires to the layer-1 pipeline.

Calibration protocol (per spec):
- 25-point 5x5 grid plus 8 extra points tighter to the 4 edges and
  4 corners (33 targets total).
- Per target: ~0.6 s settle (animated shrinking ring), then ~0.4 s of
  samples, MAD-based outlier rejection, average of the inliers.
- Edge weighting via `sample_weight` in the regression.
- Result persisted to YAML (plain lists, no pickle).
"""

from __future__ import annotations

import datetime as _dt
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURE_NAMES = ["gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch",
                 "face_x", "face_y", "face_z"]

# Grid margins as a fraction of the screen dimension.
GRID_MARGIN = 0.08     # 5x5 grid
EXTRA_MARGIN = 0.025   # extra corner/edge targets, tighter to the border

# Edge weighting for the regression.
WEIGHT_CENTER = 1.0    # inner grid points
WEIGHT_BORDER = 1.5    # outer ring of the 5x5 grid
WEIGHT_EXTRA = 2.0     # extra corner/edge targets

SETTLE_SECONDS = 0.6
COLLECT_SECONDS = 0.4
MIN_INLIERS = 6
MAX_POINT_RETRIES = 2

_BG = (18, 18, 18)
_FG = (235, 235, 235)
_DIM = (110, 110, 110)


def make_features(gaze_yaw: float, gaze_pitch: float, head_yaw: float,
                  head_pitch: float, face_center: np.ndarray) -> np.ndarray:
    """Canonical layer-2 feature vector. Order must match FEATURE_NAMES."""
    return np.array([gaze_yaw, gaze_pitch, head_yaw, head_pitch,
                     face_center[0], face_center[1], face_center[2]],
                    dtype=np.float64)


@dataclass
class CalibrationTarget:
    x: float
    y: float
    weight: float
    kind: str  # "grid-center" | "grid-border" | "extra-corner" | "extra-edge"


def build_targets(width: int, height: int) -> list[CalibrationTarget]:
    """25-point 5x5 grid + 8 extra edge/corner targets (33 total)."""
    targets: list[CalibrationTarget] = []

    xs = np.linspace(GRID_MARGIN, 1 - GRID_MARGIN, 5)
    ys = np.linspace(GRID_MARGIN, 1 - GRID_MARGIN, 5)
    for iy, ny in enumerate(ys):
        for ix, nx in enumerate(xs):
            border = ix in (0, 4) or iy in (0, 4)
            targets.append(CalibrationTarget(
                x=nx * width, y=ny * height,
                weight=WEIGHT_BORDER if border else WEIGHT_CENTER,
                kind="grid-border" if border else "grid-center"))

    m = EXTRA_MARGIN
    for nx, ny in [(m, m), (1 - m, m), (m, 1 - m), (1 - m, 1 - m)]:
        targets.append(CalibrationTarget(
            x=nx * width, y=ny * height, weight=WEIGHT_EXTRA,
            kind="extra-corner"))
    for nx, ny in [(0.5, m), (0.5, 1 - m), (m, 0.5), (1 - m, 0.5)]:
        targets.append(CalibrationTarget(
            x=nx * width, y=ny * height, weight=WEIGHT_EXTRA,
            kind="extra-edge"))
    return targets


def reject_outliers(samples: np.ndarray) -> np.ndarray:
    """Drop frames whose gaze yaw or pitch deviates > 2.5 robust sigmas.

    samples: (n, n_features) with gaze_yaw/gaze_pitch in columns 0 and 1.
    Returns the inlier subset.
    """
    keep = np.ones(len(samples), dtype=bool)
    for col in (0, 1):
        values = samples[:, col]
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        sigma = 1.4826 * mad
        if sigma > 1e-9:
            keep &= np.abs(values - median) <= 2.5 * sigma
    return samples[keep]


class GazeMapper:
    """Degree-2 polynomial Ridge mapping features -> screen pixels."""

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha
        self._scaler = StandardScaler()
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)
        self._ridge = Ridge(alpha=alpha)
        self._fitted = False
        self.meta: dict = {}

    def fit(self, features: np.ndarray, targets_px: np.ndarray,
            sample_weight: np.ndarray) -> "GazeMapper":
        scaled = self._scaler.fit_transform(features)
        expanded = self._poly.fit_transform(scaled)
        self._ridge.fit(expanded, targets_px, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """features: (n_features,) or (n, n_features) -> (n, 2) or (2,)."""
        if not self._fitted:
            raise RuntimeError("GazeMapper is not fitted/loaded.")
        single = features.ndim == 1
        array = np.atleast_2d(features)
        out = self._ridge.predict(
            self._poly.transform(self._scaler.transform(array)))
        return out[0] if single else out

    def residuals_px(self, features: np.ndarray,
                     targets_px: np.ndarray) -> np.ndarray:
        """Per-point Euclidean residual in pixels."""
        return np.linalg.norm(self.predict(features) - targets_px, axis=1)

    def save(self, path: Path, extra_meta: dict | None = None) -> None:
        if not self._fitted:
            raise RuntimeError("Nothing to save: mapper is not fitted.")
        payload = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": {
                "degree": self.degree,
                "alpha": self.alpha,
                "feature_names": FEATURE_NAMES,
                "scaler_mean": self._scaler.mean_.tolist(),
                "scaler_scale": self._scaler.scale_.tolist(),
                "coef": self._ridge.coef_.tolist(),
                "intercept": np.atleast_1d(
                    self._ridge.intercept_).tolist(),
            },
            "meta": {**self.meta, **(extra_meta or {})},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GazeMapper":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {path}. Run "
                "'python main.py calibrate' first.")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = payload["model"]
        if model["feature_names"] != FEATURE_NAMES:
            raise ValueError(
                f"Calibration {path} was created with features "
                f"{model['feature_names']}, current code expects "
                f"{FEATURE_NAMES}. Recalibrate.")
        mapper = cls(degree=model["degree"], alpha=model["alpha"])
        n = len(FEATURE_NAMES)
        mapper._scaler.mean_ = np.array(model["scaler_mean"])
        mapper._scaler.scale_ = np.array(model["scaler_scale"])
        mapper._scaler.var_ = mapper._scaler.scale_ ** 2
        mapper._scaler.n_features_in_ = n
        mapper._poly.fit(np.zeros((1, n)))
        mapper._ridge.coef_ = np.array(model["coef"])
        mapper._ridge.intercept_ = np.array(model["intercept"])
        mapper._fitted = True
        mapper.meta = payload.get("meta", {})
        return mapper


@dataclass
class CalibrationResult:
    mapper: GazeMapper
    n_targets: int
    n_used: int
    failed_targets: int
    residual_mean_px: float
    residual_p95_px: float
    residual_mean_deg: float | None = None


def _draw_target(canvas: np.ndarray, x: float, y: float,
                 ring_radius: float | None, collecting: bool) -> None:
    center = (int(round(x)), int(round(y)))
    if ring_radius is not None:
        cv2.circle(canvas, center, int(ring_radius), _DIM, 1, cv2.LINE_AA)
    color = _FG if collecting else _DIM
    cv2.circle(canvas, center, 6, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, 2, _BG, -1, cv2.LINE_AA)


def _draw_status(canvas: np.ndarray, text: str) -> None:
    cv2.putText(canvas, text, (24, canvas.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DIM, 1, cv2.LINE_AA)


# Shown while the ring contracts. On a wide monitor, reaching the edges with
# the eyes alone is impossible and unnatural, so head motion is expected and
# is a mapping feature; what matters is that the pose is consistent between
# calibration and use, and steady during each capture.
_PROTOCOL_LINES = [
    "Look at each point naturally - move your head and eyes as you",
    "normally would while working. Hold steady while the ring contracts.",
]


def _draw_protocol_banner(canvas: np.ndarray) -> None:
    screen_w = canvas.shape[1]
    y = 40
    for line in _PROTOCOL_LINES:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(canvas, line, ((screen_w - size[0]) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _DIM, 1, cv2.LINE_AA)
        y += 28


def _wait_for_start(window: str, screen_w: int, screen_h: int) -> bool:
    canvas = np.full((screen_h, screen_w, 3), _BG[0], dtype=np.uint8)
    lines = [
        "Calibration",
        "",
        "Look at each point naturally - move your head and eyes as you",
        "normally would while working. Do NOT hold your head rigidly.",
        "Hold your position steady while the ring contracts.",
        "Keep the same seating position you will use afterwards.",
        "",
        "33 targets, about one second each.",
        "",
        "SPACE to start    ESC to abort",
    ]
    y = screen_h // 2 - 110
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


# A steady fixation scatters gaze by ~1 deg; above this, the user was moving
# during the capture (head/eyes not held) and that target is suspect.
HIGH_VARIANCE_DEG = 2.5


def _print_diagnostics(target_px: list, kinds: list, frame_stds: list,
                       residuals: np.ndarray, features: np.ndarray) -> dict:
    """Per-target residual, head-pose coverage and motion warnings.

    Returns the structured diagnostics for persistence.
    """
    median_res = float(np.median(residuals))
    high_res = max(2.5 * median_res, 40.0)  # px; garbage-target threshold

    rows = []
    for i, (xy, kind, fstd, res) in enumerate(
            zip(target_px, kinds, frame_stds, residuals)):
        flags = []
        if res > high_res:
            flags.append("HIGH-RESIDUAL")
        if fstd > HIGH_VARIANCE_DEG:
            flags.append("MOVED")
        rows.append({"x": round(xy[0]), "y": round(xy[1]), "kind": kind,
                     "residual_px": round(float(res), 1),
                     "frame_std_deg": round(fstd, 2),
                     "flags": " ".join(flags)})

    print("\nper-target diagnostics (sorted by residual):")
    print("   x     y   kind          resid_px  frame_std_deg  flags")
    print("-" * 62)
    for r in sorted(rows, key=lambda r: r["residual_px"], reverse=True):
        print(f"{r['x']:>5} {r['y']:>5}   {r['kind']:<13} "
              f"{r['residual_px']:>8.1f}  {r['frame_std_deg']:>12.2f}  "
              f"{r['flags']}")

    head_yaw = features[:, 2]
    head_pitch = features[:, 3]
    yaw_span = float(head_yaw.max() - head_yaw.min())
    pitch_span = float(head_pitch.max() - head_pitch.min())
    print("-" * 62)
    print(f"head yaw covered   : [{head_yaw.min():+.1f}, "
          f"{head_yaw.max():+.1f}] deg  (span {yaw_span:.1f})")
    print(f"head pitch covered : [{head_pitch.min():+.1f}, "
          f"{head_pitch.max():+.1f}] deg  (span {pitch_span:.1f})")

    moved = [r for r in rows if "MOVED" in r["flags"]]
    garbage = [r for r in rows if "HIGH-RESIDUAL" in r["flags"]]
    if moved:
        print(f"WARNING: {len(moved)} target(s) had high inter-frame "
              f"variance (> {HIGH_VARIANCE_DEG:.1f} deg) - you were moving "
              "during capture. Consider recalibrating those calmly.")
    if garbage:
        print(f"WARNING: {len(garbage)} target(s) fit poorly "
              f"(> {high_res:.0f} px residual) - possible bad points; if "
              "many, recalibrate.")
    if not moved and not garbage:
        print("no per-target anomalies detected.")

    return {
        "per_target": rows,
        "head_yaw_range_deg": [round(float(head_yaw.min()), 1),
                               round(float(head_yaw.max()), 1)],
        "head_pitch_range_deg": [round(float(head_pitch.min()), 1),
                                 round(float(head_pitch.max()), 1)],
        "targets_high_variance": len(moved),
        "targets_high_residual": len(garbage),
    }


def run_calibration(sample_provider,
                    screen_size: tuple[int, int],
                    output_path: Path,
                    screen_width_cm: float | None = None,
                    screen_height_cm: float | None = None,
                    distance_cm: float | None = None,
                    seed: int = 42) -> CalibrationResult | None:
    """Interactive calibration. Returns None if the user aborts.

    sample_provider: callable() -> feature vector (FEATURE_NAMES order) or
    None when no face is visible. It is called once per UI frame and is
    expected to block at camera rate.
    """
    screen_w, screen_h = screen_size
    targets = build_targets(screen_w, screen_h)
    rng = random.Random(seed)
    queue: list[tuple[CalibrationTarget, int]] = \
        [(t, 0) for t in rng.sample(targets, len(targets))]

    window = "gazetrack - calibration"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    try:
        if not _wait_for_start(window, screen_w, screen_h):
            print("Calibration aborted.")
            return None

        rows: list[np.ndarray] = []
        target_px: list[list[float]] = []
        weights: list[float] = []
        kinds: list[str] = []
        frame_stds: list[float] = []  # per-target inter-frame gaze std (deg)
        failed = 0
        done = 0
        total = len(queue)

        while queue:
            target, retries = queue.pop(0)
            phase_start = time.perf_counter()
            samples: list[np.ndarray] = []
            collecting = False
            aborted = False

            while True:
                elapsed = time.perf_counter() - phase_start
                canvas = np.full((screen_h, screen_w, 3), _BG[0],
                                 dtype=np.uint8)
                if not collecting and elapsed >= SETTLE_SECONDS:
                    collecting = True
                    phase_start = time.perf_counter()
                    elapsed = 0.0

                sample = sample_provider()
                if collecting and sample is not None:
                    samples.append(sample)

                ring = None if collecting else \
                    22 - (22 - 8) * (elapsed / SETTLE_SECONDS)
                _draw_protocol_banner(canvas)
                _draw_target(canvas, target.x, target.y, ring, collecting)
                _draw_status(canvas,
                             f"target {done + 1}/{total}    ESC to abort")
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    aborted = True
                    break
                if collecting and elapsed >= COLLECT_SECONDS:
                    break

            if aborted:
                print("Calibration aborted.")
                return None

            inliers = reject_outliers(np.array(samples)) if samples \
                else np.empty((0, len(FEATURE_NAMES)))
            if len(inliers) >= MIN_INLIERS:
                rows.append(inliers.mean(axis=0))
                target_px.append([target.x, target.y])
                weights.append(target.weight)
                kinds.append(target.kind)
                # Inter-frame gaze scatter during the ~0.4 s capture: high
                # values mean the user was moving (not fixating steadily).
                frame_stds.append(float(np.hypot(inliers[:, 0].std(),
                                                 inliers[:, 1].std())))
                done += 1
            elif retries < MAX_POINT_RETRIES:
                queue.append((target, retries + 1))
            else:
                failed += 1
                done += 1
                print(f"WARNING: target at ({target.x:.0f}, {target.y:.0f}) "
                      "skipped after retries (face not stable/visible).")
    finally:
        cv2.destroyWindow(window)

    if len(rows) < 12:
        print(f"ERROR: only {len(rows)} usable targets out of "
              f"{len(targets)}; not enough to fit a stable mapping. "
              "Check lighting and face visibility, then recalibrate.")
        return None

    features = np.vstack(rows)
    targets_arr = np.array(target_px)
    weight_arr = np.array(weights)

    mapper = GazeMapper()
    mapper.fit(features, targets_arr, weight_arr)
    residuals = mapper.residuals_px(features, targets_arr)
    mean_px = float(residuals.mean())
    p95_px = float(np.percentile(residuals, 95))

    mean_deg = None
    if screen_width_cm and distance_cm:
        px_per_cm = screen_w / screen_width_cm
        mean_deg = float(np.degrees(
            np.arctan2(mean_px / px_per_cm, distance_cm)))

    print(f"\ntargets used       : {len(rows)}/{len(targets)}"
          + (f"  ({failed} failed)" if failed else ""))
    print(f"fit residual (train): mean {mean_px:.1f} px, "
          f"p95 {p95_px:.1f} px"
          + (f", ~{mean_deg:.2f} deg" if mean_deg else
             " (deg n/a: screen geometry not configured)"))

    diagnostics = _print_diagnostics(target_px, kinds, frame_stds,
                                     residuals, features)

    meta = {
        "screen": {"width_px": screen_w, "height_px": screen_h,
                   "width_cm": screen_width_cm,
                   "height_cm": screen_height_cm,
                   "distance_cm": distance_cm},
        "targets_total": len(targets),
        "targets_used": len(rows),
        "targets_failed": failed,
        "residual_mean_px": round(mean_px, 2),
        "residual_p95_px": round(p95_px, 2),
        "residual_mean_deg": round(mean_deg, 3) if mean_deg else None,
        "diagnostics": diagnostics,
    }
    mapper.meta = meta
    mapper.save(Path(output_path))
    print(f"\nCalibration saved to {output_path}")
    print("Note: training residual understates real error; run "
          "'python main.py evaluate' for an honest measurement.")

    return CalibrationResult(
        mapper=mapper, n_targets=len(targets), n_used=len(rows),
        failed_targets=failed, residual_mean_px=mean_px,
        residual_p95_px=p95_px, residual_mean_deg=mean_deg)
