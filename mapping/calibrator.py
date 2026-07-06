"""Screen-mapping calibration (layer 2).

Learns (gaze yaw/pitch, head yaw/pitch) -> (screen_x, screen_y) with a
second-degree polynomial regression (scikit-learn PolynomialFeatures +
Ridge), weighted so edge and corner calibration points count more than
central ones (webcam gaze degrades toward the periphery; the extra weight
mitigates it). Ridge alpha is chosen by leave-one-out cross-validation.
Absolute face position was dropped from the features - see FEATURE_NAMES.

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

# Feature vector: gaze angles + head pose only. Absolute face position
# (face_x/y/z) was REMOVED after the 2026-07-05 evaluation: with it the
# degree-2 map had 35 polynomial terms for 33 calibration samples and leaned
# on large, unstable coefficients over face position (e.g. face_x coef ~-218
# with a 6.9 mm std), so normal between-session head drift produced runaway
# predictions that the screen clamp folded onto the edges (center error worse
# than corners). Four inputs give 14 degree-2 terms vs 33 samples (ratio
# 0.42), a healthy regime. Bumping FEATURE_VERSION invalidates old
# calibrations on load.
FEATURE_NAMES = ["gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch"]
# v1 = 7-feature vector (face_x/y/z); v2 = 4-feature grid-only; v3 = combined
# grid + smooth-pursuit calibration (adds lag/degree/pursuit metadata).
FEATURE_VERSION = 3

# Alpha grid searched by cross-validation (log scale, 0.1 .. 1000).
ALPHA_GRID = np.logspace(-1, 3, 13)

# Combined calibration: polynomial degrees compared by block-CV, and how much
# a discrete grid target (high-quality multi-frame average) outweighs a single
# pursuit sample.
MODEL_DEGREES = (2, 3)
DISCRETE_WEIGHT_MULT = 20.0

# Guardrail: warn when polynomial terms / samples exceeds this. Above ~0.5
# the fit starts to interpolate and extrapolate wildly (the bug above sat at
# 35/33 ~ 1.06). Trips if features or the degree grow without more samples.
MAX_FEATURE_SAMPLE_RATIO = 0.5

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
                  head_pitch: float) -> np.ndarray:
    """Canonical layer-2 feature vector. Order must match FEATURE_NAMES.

    Absolute face position is intentionally excluded (see FEATURE_NAMES).
    """
    return np.array([gaze_yaw, gaze_pitch, head_yaw, head_pitch],
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


def _loocv_error_px(design: np.ndarray, targets_px: np.ndarray,
                    sample_weight: np.ndarray, alpha: float) -> float:
    """Mean leave-one-out Euclidean error (px) for Ridge at a given alpha.

    Explicit LOO (33 refits) rather than RidgeCV's internal GCV: it handles
    the 2D target and the edge sample_weight exactly and yields an honest
    error in pixels, which the in-sample training residual does not.
    """
    n = len(design)
    idx = np.arange(n)
    errors = np.empty(n)
    for i in range(n):
        keep = idx != i
        ridge = Ridge(alpha=alpha)
        ridge.fit(design[keep], targets_px[keep],
                  sample_weight=sample_weight[keep])
        pred = ridge.predict(design[i:i + 1])[0]
        errors[i] = np.hypot(pred[0] - targets_px[i, 0],
                             pred[1] - targets_px[i, 1])
    return float(errors.mean())


def block_kfold_indices(n: int, k: int = 5):
    """Yield (train_idx, val_idx) for k-fold over CONTIGUOUS blocks.

    Smooth-pursuit samples are strongly autocorrelated in time (consecutive
    frames are ~33 ms apart on the same eye trajectory). A random k-fold would
    put near-identical neighbours in both train and validation and report an
    optimistically low CV error, which would destroy CV as an honest error
    predictor. Contiguous temporal blocks keep each validation fold
    independent of its training data. Callers MUST pass samples in
    acquisition order.
    """
    bounds = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        val = np.arange(lo, hi)
        train = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        if len(train) == 0:
            continue
        yield train, val


def block_cv_error_px(features: np.ndarray, targets_px: np.ndarray,
                      sample_weight: np.ndarray, degree: int, alpha: float,
                      k: int = 5) -> float:
    """Mean Euclidean px error under block k-fold CV (samples in time order).

    Scaler and polynomial expansion are fit on each training fold only to
    avoid leakage. Used to choose the pursuit lag and, later, the model
    degree/alpha.
    """
    all_err: list[float] = []
    for train, val in block_kfold_indices(len(features), k):
        scaler = StandardScaler().fit(features[train])
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_tr = poly.fit_transform(scaler.transform(features[train]))
        x_va = poly.transform(scaler.transform(features[val]))
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_tr, targets_px[train], sample_weight=sample_weight[train])
        pred = ridge.predict(x_va)
        all_err.extend(np.hypot(pred[:, 0] - targets_px[val, 0],
                                pred[:, 1] - targets_px[val, 1]))
    return float(np.mean(all_err)) if all_err else float("inf")


class GazeMapper:
    """Degree-2 polynomial Ridge mapping features -> screen pixels.

    Ridge alpha is chosen by leave-one-out cross-validation over ALPHA_GRID
    unless a fixed alpha is passed (used only when reloading from YAML).
    """

    def __init__(self, degree: int = 2, alpha: float | None = None):
        self.degree = degree
        self.alpha = alpha           # resolved by LOOCV in fit() if None
        self.loocv_error_px: float | None = None
        self._scaler = StandardScaler()
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)
        self._ridge = Ridge(alpha=alpha if alpha is not None else 1.0)
        self._fitted = False
        self.meta: dict = {}

    @property
    def n_poly_features(self) -> int:
        return int(self._poly.n_output_features_)

    def fit(self, features: np.ndarray, targets_px: np.ndarray,
            sample_weight: np.ndarray) -> "GazeMapper":
        scaled = self._scaler.fit_transform(features)
        design = self._poly.fit_transform(scaled)
        if self.alpha is None:
            scores = {a: _loocv_error_px(design, targets_px, sample_weight, a)
                      for a in ALPHA_GRID}
            self.alpha = float(min(scores, key=scores.get))
            self.loocv_error_px = scores[self.alpha]
        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(design, targets_px, sample_weight=sample_weight)
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
                "feature_version": FEATURE_VERSION,
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
        version = model.get("feature_version", 1)
        if version != FEATURE_VERSION or model["feature_names"] != \
                FEATURE_NAMES:
            raise ValueError(
                f"Calibration {path} uses an outdated feature vector "
                f"(version {version}: {model['feature_names']}); current "
                f"code expects version {FEATURE_VERSION}: {FEATURE_NAMES}. "
                "The feature set changed, so old calibrations are invalid - "
                "run 'python main.py calibrate' to recalibrate.")
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
    loocv_error_px: float | None = None
    cv_error_px: float | None = None  # block-CV error (combined calibration)


@dataclass
class GridData:
    """Discrete-grid collection result (features/targets/weights + QA)."""
    features: np.ndarray       # (n, 4) averaged feature vector per target
    targets_px: np.ndarray     # (n, 2)
    weights: np.ndarray        # (n,) edge weights
    kinds: list                # per-target kind
    frame_stds: list           # per-target inter-frame gaze std (deg)
    targets_total: int
    targets_used: int
    targets_failed: int


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


def collect_grid(sample_provider, screen_size: tuple[int, int],
                 seed: int = 42) -> GridData | None:
    """Interactive discrete-grid collection (no fitting). Returns None if the
    user aborts or too few targets are usable.

    sample_provider: callable() -> feature vector (FEATURE_NAMES order) or
    None when no face is visible. Called once per UI frame; blocks at camera
    rate.
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

    return GridData(
        features=np.vstack(rows), targets_px=np.array(target_px),
        weights=np.array(weights), kinds=kinds, frame_stds=frame_stds,
        targets_total=len(targets), targets_used=len(rows),
        targets_failed=failed)


def run_calibration(sample_provider,
                    screen_size: tuple[int, int],
                    output_path: Path,
                    screen_width_cm: float | None = None,
                    screen_height_cm: float | None = None,
                    distance_cm: float | None = None,
                    seed: int = 42) -> CalibrationResult | None:
    """Grid-only calibration: collect the 33 discrete targets and fit a
    LOOCV-tuned degree-2 map. Returns None if the user aborts."""
    screen_w, screen_h = screen_size
    grid = collect_grid(sample_provider, screen_size, seed)
    if grid is None:
        return None

    features = grid.features
    targets_arr = grid.targets_px
    weight_arr = grid.weights
    target_px = grid.targets_px.tolist()
    kinds = grid.kinds
    frame_stds = grid.frame_stds
    failed = grid.targets_failed
    targets = list(range(grid.targets_total))  # only len() is used below

    mapper = GazeMapper()
    mapper.fit(features, targets_arr, weight_arr)
    residuals = mapper.residuals_px(features, targets_arr)
    mean_px = float(residuals.mean())
    p95_px = float(np.percentile(residuals, 95))
    loocv_px = mapper.loocv_error_px

    def to_deg(px: float) -> float | None:
        if not (screen_width_cm and distance_cm):
            return None
        px_per_cm = screen_w / screen_width_cm
        return float(np.degrees(np.arctan2(px / px_per_cm, distance_cm)))

    mean_deg = to_deg(mean_px)
    loocv_deg = to_deg(loocv_px)

    n_samples = grid.targets_used
    ratio = mapper.n_poly_features / n_samples

    print(f"\ntargets used       : {n_samples}/{len(targets)}"
          + (f"  ({failed} failed)" if failed else ""))
    print(f"ridge alpha (LOOCV) : {mapper.alpha:g}  "
          f"(searched {ALPHA_GRID[0]:g}..{ALPHA_GRID[-1]:g})")
    print(f"LOOCV error (honest): mean {loocv_px:.1f} px"
          + (f", ~{loocv_deg:.2f} deg" if loocv_deg else "")
          + "  <- expected real error")
    print(f"fit residual (train): mean {mean_px:.1f} px, "
          f"p95 {p95_px:.1f} px"
          + (f", ~{mean_deg:.2f} deg" if mean_deg else
             " (deg n/a: screen geometry not configured)")
          + "  (in-sample; optimistic)")
    print(f"model size         : {mapper.n_poly_features} polynomial terms "
          f"for {n_samples} samples (ratio {ratio:.2f})")
    if ratio > MAX_FEATURE_SAMPLE_RATIO:
        print(f"WARNING: features/samples ratio {ratio:.2f} exceeds "
              f"{MAX_FEATURE_SAMPLE_RATIO:.2f}. The map is under-determined "
              "and will overfit (in-sample residual low, real error high). "
              "Reduce features or the polynomial degree, or add samples.")

    diagnostics = _print_diagnostics(target_px, kinds, frame_stds,
                                     residuals, features)

    meta = {
        "screen": {"width_px": screen_w, "height_px": screen_h,
                   "width_cm": screen_width_cm,
                   "height_cm": screen_height_cm,
                   "distance_cm": distance_cm},
        "targets_total": len(targets),
        "targets_used": n_samples,
        "targets_failed": failed,
        "alpha": mapper.alpha,
        "loocv_error_px": round(loocv_px, 2),
        "loocv_error_deg": round(loocv_deg, 3) if loocv_deg else None,
        "residual_mean_px": round(mean_px, 2),
        "residual_p95_px": round(p95_px, 2),
        "residual_mean_deg": round(mean_deg, 3) if mean_deg else None,
        "poly_features": mapper.n_poly_features,
        "feature_sample_ratio": round(ratio, 3),
        "diagnostics": diagnostics,
    }
    mapper.meta = meta
    mapper.save(Path(output_path))
    print(f"\nCalibration saved to {output_path}")
    print("Note: trust the LOOCV error, not the training residual, as the "
          "predictor of real accuracy; run 'python main.py evaluate' to "
          "confirm on held-out targets.")

    return CalibrationResult(
        mapper=mapper, n_targets=len(targets), n_used=n_samples,
        failed_targets=failed, residual_mean_px=mean_px,
        residual_p95_px=p95_px, residual_mean_deg=mean_deg,
        loocv_error_px=loocv_px)


# --- Combined grid + pursuit calibration (PHASE-01B c/d) -------------------

def _combined_block_cv(feats_g, tgt_g, w_g, feats_p, tgt_p, w_p,
                       degree: int, alpha: float, k: int = 5) -> float:
    """Block-CV error (px) for the combined model, validated on held-out
    pursuit BLOCKS. The discrete grid anchors are always in the training fold
    (few, high-quality); each fold holds out one contiguous pursuit segment,
    so the score reflects spatial generalization without the autocorrelation
    leak a random fold would introduce."""
    all_err: list[float] = []
    for train_p, val_p in block_kfold_indices(len(feats_p), k):
        x = np.vstack([feats_g, feats_p[train_p]])
        y = np.vstack([tgt_g, tgt_p[train_p]])
        w = np.concatenate([w_g, w_p[train_p]])
        scaler = StandardScaler().fit(x)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_tr = poly.fit_transform(scaler.transform(x))
        x_va = poly.transform(scaler.transform(feats_p[val_p]))
        ridge = Ridge(alpha=alpha).fit(x_tr, y, sample_weight=w)
        pred = ridge.predict(x_va)
        all_err.extend(np.hypot(pred[:, 0] - tgt_p[val_p, 0],
                                pred[:, 1] - tgt_p[val_p, 1]))
    return float(np.mean(all_err)) if all_err else float("inf")


def select_model_block_cv(feats_g, tgt_g, w_g, feats_p, tgt_p, w_p,
                          degrees=MODEL_DEGREES, alphas=ALPHA_GRID,
                          k: int = 5) -> dict:
    """Choose polynomial degree and alpha by block-CV over the combined data.
    Returns {best_degree, best_alpha, cv_error_px, per_degree: {deg: (alpha,
    err)}}."""
    per_degree: dict = {}
    for degree in degrees:
        best_a, best_e = None, float("inf")
        for alpha in alphas:
            err = _combined_block_cv(feats_g, tgt_g, w_g, feats_p, tgt_p, w_p,
                                     degree, alpha, k)
            if err < best_e:
                best_a, best_e = float(alpha), err
        per_degree[degree] = (best_a, best_e)
    best_degree = min(per_degree, key=lambda d: per_degree[d][1])
    return {"best_degree": best_degree,
            "best_alpha": per_degree[best_degree][0],
            "cv_error_px": per_degree[best_degree][1],
            "per_degree": per_degree}


def _region_residual(features: np.ndarray, targets_px: np.ndarray,
                     mapper: "GazeMapper", screen_w: int,
                     screen_h: int) -> dict:
    """In-sample mean residual per 3x3 screen region (by target position), to
    watch for lateral degradation in the fitted map."""
    pred = mapper.predict(features)
    err = np.hypot(pred[:, 0] - targets_px[:, 0], pred[:, 1] - targets_px[:, 1])
    cols = ("left", "center", "right")
    rows = ("top", "middle", "bottom")

    def third(v, size):
        return min(int(v / (size / 3.0)), 2) if size > 0 else 0

    out: dict = {}
    for r in range(3):
        for c in range(3):
            mask = np.array([third(t[0], screen_w) == c
                             and third(t[1], screen_h) == r
                             for t in targets_px])
            if mask.any():
                out[f"{rows[r]}-{cols[c]}"] = {
                    "samples": int(mask.sum()),
                    "mean_px": round(float(err[mask].mean()), 1)}
    return out


def fit_combined(grid: GridData, feats_p: np.ndarray, tgt_p: np.ndarray,
                 w_p: np.ndarray, screen_size: tuple[int, int],
                 output_path: Path, pursuit_meta: dict,
                 screen_width_cm: float | None = None,
                 screen_height_cm: float | None = None,
                 distance_cm: float | None = None) -> CalibrationResult | None:
    """Combine the discrete grid with pursuit samples, select degree/alpha by
    block-CV, fit the final model, report and save (v3)."""
    screen_w, screen_h = screen_size

    # Discrete anchors weigh DISCRETE_WEIGHT_MULT pursuit samples each, on top
    # of their edge weights.
    feats_g = grid.features
    tgt_g = grid.targets_px
    w_g = grid.weights * DISCRETE_WEIGHT_MULT

    selection = select_model_block_cv(feats_g, tgt_g, w_g, feats_p, tgt_p, w_p)
    degree = selection["best_degree"]
    alpha = selection["best_alpha"]
    cv_px = selection["cv_error_px"]

    feats_all = np.vstack([feats_g, feats_p])
    tgt_all = np.vstack([tgt_g, tgt_p])
    w_all = np.concatenate([w_g, w_p])
    mapper = GazeMapper(degree=degree, alpha=alpha)
    mapper.fit(feats_all, tgt_all, w_all)

    n_total = len(feats_all)
    ratio = mapper.n_poly_features / n_total
    region = _region_residual(feats_all, tgt_all, mapper, screen_w, screen_h)

    def to_deg(px: float) -> float | None:
        if not (screen_width_cm and distance_cm):
            return None
        px_per_cm = screen_w / screen_width_cm
        return float(np.degrees(np.arctan2(px / px_per_cm, distance_cm)))

    cv_deg = to_deg(cv_px)

    print(f"\ncombined calibration")
    print(f"samples            : {grid.targets_used} discrete "
          f"(weight x{DISCRETE_WEIGHT_MULT:g}) + {len(feats_p)} pursuit "
          f"= {n_total} total")
    print(f"latency lag         : {pursuit_meta.get('best_lag_ms', '?')} ms")
    pd = selection["per_degree"]
    for deg in sorted(pd):
        a, e = pd[deg]
        star = "  <- chosen" if deg == degree else ""
        print(f"degree {deg}           : block-CV {e:.1f} px "
              f"(alpha {a:g}){star}")
    print(f"model              : degree {degree}, alpha {alpha:g}, "
          f"{mapper.n_poly_features} terms for {n_total} samples "
          f"(ratio {ratio:.3f})")
    print(f"block-CV error      : {cv_px:.1f} px"
          + (f", ~{cv_deg:.2f} deg" if cv_deg else "")
          + f" ({cv_px / screen_w * 100:.2f}% width)  <- expected real error")
    if ratio > MAX_FEATURE_SAMPLE_RATIO:
        print(f"WARNING: features/samples ratio {ratio:.3f} exceeds "
              f"{MAX_FEATURE_SAMPLE_RATIO:.2f} (overfit risk).")

    print("\nregion residual (3x3, in-sample mean px):")
    for r in ("top", "middle", "bottom"):
        cells = []
        for c in ("left", "center", "right"):
            s = region.get(f"{r}-{c}")
            cells.append(f"{s['mean_px']:>8.0f}" if s else f"{'-':>8}")
        print(f"  {r:<7}" + "".join(cells))

    meta = {
        "screen": {"width_px": screen_w, "height_px": screen_h,
                   "width_cm": screen_width_cm,
                   "height_cm": screen_height_cm,
                   "distance_cm": distance_cm},
        "source": "grid+pursuit",
        "discrete_samples": grid.targets_used,
        "discrete_weight_mult": DISCRETE_WEIGHT_MULT,
        "pursuit_samples": int(len(feats_p)),
        "pursuit_lag_ms": pursuit_meta.get("best_lag_ms"),
        "pursuit_discard": pursuit_meta.get("discard"),
        "degree": degree,
        "alpha": alpha,
        "cv_method": "block-kfold-temporal",
        "cv_error_px": round(cv_px, 2),
        "cv_error_deg": round(cv_deg, 3) if cv_deg else None,
        "degree_cv": {int(d): round(pd[d][1], 2) for d in pd},
        "poly_features": mapper.n_poly_features,
        "feature_sample_ratio": round(ratio, 4),
        # Consumer metadata (not used in fitting): the covered head-pose range,
        # so the overlay can fade only OUTSIDE the range the map was trained on
        # instead of a fixed threshold.
        "head_yaw_range_deg": [round(float(feats_all[:, 2].min()), 1),
                               round(float(feats_all[:, 2].max()), 1)],
        "head_pitch_range_deg": [round(float(feats_all[:, 3].min()), 1),
                                 round(float(feats_all[:, 3].max()), 1)],
        "region_residual": region,
    }
    mapper.meta = meta
    mapper.save(Path(output_path))
    print(f"\nCalibration saved to {output_path}")
    print("Trust the block-CV error as the real-accuracy predictor; run "
          "'python main.py evaluate' to confirm on the held-out grid.")

    return CalibrationResult(
        mapper=mapper, n_targets=grid.targets_total,
        n_used=grid.targets_used, failed_targets=grid.targets_failed,
        residual_mean_px=cv_px, residual_p95_px=cv_px,
        residual_mean_deg=cv_deg, cv_error_px=cv_px)
