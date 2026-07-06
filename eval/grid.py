"""Accuracy evaluation on a 5x5 grid (phase success metric).

Shows 25 targets (same margins as the calibration grid, but this is a held
out measurement: the user looks at each target and we record what the FULL
pipeline - layer 1 + calibrated layer 2, no smoothing - predicts). Errors
are reported by zone:

- corners: the 4 grid corners
- edges:   the 12 remaining boundary targets
- center:  the 9 inner targets

Pixels and percentage-of-screen-width are always reported (the latter is the
metric comparable across monitors). Centimetres and degrees are reported
only when the physical screen geometry is configured (data/screen.yaml); an
unconfigured run is marked as such rather than emitting wrong cm/deg.

Peripheral degradation is expected and inherent to webcam gaze; the goal
here is to quantify it, not to hide it. Phase target: mean error <= 3.5 deg
in the center zone.

Angular conversion is exact only at the screen center (small-angle
approximation elsewhere); the deviation is negligible for this report.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

GRID_MARGIN = 0.08
SETTLE_SECONDS = 0.8
COLLECT_SECONDS = 0.5
MIN_FRAMES = 6

_BG = (18, 18, 18)
_FG = (235, 235, 235)
_DIM = (110, 110, 110)


@dataclass
class EvalTarget:
    x: float
    y: float
    zone: str  # "center" | "edge" | "corner"


@dataclass
class EvalMeasurement:
    target: EvalTarget
    error_px: float
    jitter_px: float  # std of per-frame predictions around their mean
    n_frames: int
    pred_x: float = 0.0  # mean RAW predicted x (before any screen clamp)
    pred_y: float = 0.0
    oob_frac: float = 0.0  # fraction of raw frames predicted off-screen


def build_targets(width: int, height: int) -> list[EvalTarget]:
    xs = np.linspace(GRID_MARGIN, 1 - GRID_MARGIN, 5)
    ys = np.linspace(GRID_MARGIN, 1 - GRID_MARGIN, 5)
    targets = []
    for iy, ny in enumerate(ys):
        for ix, nx in enumerate(xs):
            on_x = ix in (0, 4)
            on_y = iy in (0, 4)
            zone = "corner" if on_x and on_y else \
                "edge" if on_x or on_y else "center"
            targets.append(EvalTarget(nx * width, ny * height, zone))
    return targets


def _zone_stats(measurements: list[EvalMeasurement], screen_w: int,
                px_per_cm: float | None, distance_cm: float | None) -> dict:
    """Per-zone error. px and %-of-width are always computed; cm and deg
    only when the physical geometry is configured."""
    have_geo = px_per_cm is not None and distance_cm is not None

    def to_deg(err_px: float) -> float:
        return float(np.degrees(np.arctan2(err_px / px_per_cm, distance_cm)))

    stats: dict = {}
    for zone in ("center", "edge", "corner", "all"):
        errors = [m.error_px for m in measurements
                  if zone == "all" or m.target.zone == zone]
        if not errors:
            continue
        arr = np.array(errors)
        mean = float(arr.mean())
        entry = {
            "targets": len(arr),
            "mean_px": round(mean, 1),
            # Percentage of screen width: the metric comparable across
            # monitors regardless of physical size.
            "mean_pct": round(mean / screen_w * 100, 2),
            "p95_px": round(float(np.percentile(arr, 95)), 1),
            "worst_px": round(float(arr.max()), 1),
        }
        if have_geo:
            entry.update({
                "mean_cm": round(mean / px_per_cm, 2),
                "mean_deg": round(to_deg(mean), 2),
                "p95_deg": round(to_deg(np.percentile(arr, 95)), 2),
                "worst_deg": round(to_deg(arr.max()), 2),
            })
        stats[zone] = entry
    return stats


def _print_evidence_table(measurements: list[EvalMeasurement],
                          screen_w: int, screen_h: int) -> dict:
    """Per-target target vs RAW prediction and out-of-bounds accounting.

    A high off-screen fraction with large errors is the signature of a
    runaway-magnitude model whose predictions are only kept on-screen by the
    clamp - which then makes edge/corner targets look deceptively accurate.
    """
    print("\nevidence: target vs RAW predicted (before any screen clamp)")
    print("  x_tgt  y_tgt |  x_pred  y_pred |  err_px  zone     off-screen")
    print("-" * 64)
    for m in measurements:
        px_oob = not (0 <= m.pred_x <= screen_w)
        py_oob = not (0 <= m.pred_y <= screen_h)
        mark = ""
        if px_oob or py_oob:
            mark = "OFF(" + ("x" if px_oob else "") + ("y" if py_oob else "") \
                + ")"
        print(f"  {m.target.x:>5.0f}  {m.target.y:>5.0f} |  "
              f"{m.pred_x:>6.0f}  {m.pred_y:>6.0f} |  {m.error_px:>6.0f}  "
              f"{m.target.zone:<7}  {mark}")
    print("-" * 64)

    targets_off = sum(1 for m in measurements
                      if not (0 <= m.pred_x <= screen_w)
                      or not (0 <= m.pred_y <= screen_h))
    frame_oob = float(np.mean([m.oob_frac for m in measurements]))
    print(f"targets whose mean raw prediction is off-screen: "
          f"{targets_off}/{len(measurements)}")
    print(f"per-frame predictions off-screen: {frame_oob * 100:.0f}%")
    if targets_off > len(measurements) * 0.25:
        print("DIAGNOSIS: many raw predictions land off-screen -> the model "
              "magnitude is running away and the screen clamp is masking it. "
              "This is a feature-scale / model-conditioning problem, not a "
              "gaze (layer 1) problem.")
    return {"targets_off_screen": targets_off,
            "frame_off_screen_frac": round(frame_oob, 4)}


_REGION_COLS = ("left", "center", "right")
_REGION_ROWS = ("top", "middle", "bottom")
# Success criterion (PHASE-01B): no region may exceed this multiple of the
# median region error. It is how we confirm the right-side degradation is
# gone after the physical camera re-centering.
REGION_FLAG_RATIO = 1.5


def _region_of(value: float, size: int) -> int:
    """0/1/2 third index; guards the exact-edge case at value == size."""
    return min(int(value / (size / 3.0)), 2) if size > 0 else 0


def _region_stats(measurements: list[EvalMeasurement], screen_w: int,
                  screen_h: int) -> tuple[dict, float]:
    """Mean error over a 3x3 spatial grid (col by x third, row by y third),
    to localize regional degradation the center/edge/corner zones average
    away. Returns (stats, median_region_mean_px)."""
    buckets: dict[tuple[int, int], list[float]] = {}
    for m in measurements:
        col = _region_of(m.target.x, screen_w)
        row = _region_of(m.target.y, screen_h)
        buckets.setdefault((row, col), []).append(m.error_px)

    stats: dict = {}
    for (row, col), errs in buckets.items():
        arr = np.array(errs)
        stats[f"{_REGION_ROWS[row]}-{_REGION_COLS[col]}"] = {
            "row": row, "col": col, "targets": len(arr),
            "mean_px": round(float(arr.mean()), 1),
            "mean_pct": round(float(arr.mean()) / screen_w * 100, 2),
        }
    means = [s["mean_px"] for s in stats.values()]
    median = float(np.median(means)) if means else 0.0
    for s in stats.values():
        s["over_threshold"] = bool(median > 0
                                   and s["mean_px"] > REGION_FLAG_RATIO * median)
    return stats, median


def _print_region_grid(stats: dict, median: float) -> list[str]:
    """Print the 3x3 region error grid; return the flagged region names."""
    by_cell = {(s["row"], s["col"]): (name, s) for name, s in stats.items()}
    print(f"\nregion error (3x3, mean px / % of width; median "
          f"{median:.0f} px; [!] > {REGION_FLAG_RATIO:g}x median):")
    print("  " + f"{'':<7}" + "".join(f"{c:>18}" for c in _REGION_COLS))
    flagged: list[str] = []
    for row in range(3):
        cells = []
        for col in range(3):
            hit = by_cell.get((row, col))
            if hit is None:
                cells.append(f"{'-':>18}")
                continue
            name, s = hit
            if s["over_threshold"]:
                flagged.append(name)
            text = (f"{s['mean_px']:.0f} ({s['mean_pct']:.1f}%)"
                    + (" [!]" if s["over_threshold"] else ""))
            cells.append(f"{text:>18}")
        print("  " + f"{_REGION_ROWS[row]:<7}" + "".join(cells))
    if flagged:
        print(f"WARNING: region(s) over {REGION_FLAG_RATIO:g}x the median "
              f"error: {', '.join(flagged)}. Localized degradation persists "
              "(check camera centering, side lighting, or extreme-angle "
              "gaze).")
    else:
        print("no region exceeds the threshold: error is spatially even.")
    return flagged


def _print_report(stats: dict, jitter_mean_px: float, jitter_pct: float,
                  skipped: int, have_geo: bool) -> None:
    print()
    if have_geo:
        header = ("zone     targets   mean px   mean %    mean cm   "
                  "mean deg   p95 deg   worst deg")
    else:
        header = "zone     targets   mean px   mean %    p95 px   worst px"
    print(header)
    print("-" * len(header))
    for zone in ("center", "edge", "corner", "all"):
        if zone not in stats:
            continue
        s = stats[zone]
        if have_geo:
            print(f"{zone:<8} {s['targets']:>7}   {s['mean_px']:>7.1f}   "
                  f"{s['mean_pct']:>6.2f}   {s['mean_cm']:>7.2f}   "
                  f"{s['mean_deg']:>8.2f}   {s['p95_deg']:>7.2f}   "
                  f"{s['worst_deg']:>9.2f}")
        else:
            print(f"{zone:<8} {s['targets']:>7}   {s['mean_px']:>7.1f}   "
                  f"{s['mean_pct']:>6.2f}   {s['p95_px']:>6.1f}   "
                  f"{s['worst_px']:>8.1f}")
    print("-" * len(header))
    print(f"per-frame jitter (unsmoothed): {jitter_mean_px:.1f} px mean "
          f"({jitter_pct:.2f}% of width)")
    if skipped:
        print(f"WARNING: {skipped} target(s) skipped (face lost).")
    if have_geo:
        center = stats.get("center", {}).get("mean_deg")
        if center is not None:
            goal = "met" if center <= 3.5 else "NOT met"
            print(f"phase target (center mean <= 3.5 deg): {goal} "
                  f"({center:.2f} deg)")
    else:
        center_px = stats.get("center", {}).get("mean_px")
        center_pct = stats.get("center", {}).get("mean_pct")
        print("phase target in deg needs screen geometry (not configured). "
              f"Center mean = {center_px:.0f} px ({center_pct:.2f}% of "
              "width). Set geometry with 'python main.py screen "
              "--screen-width-cm <W> --distance-cm <D>' and re-run for "
              "cm/deg.")


def run_evaluation(predict_provider,
                   screen_size: tuple[int, int],
                   screen_width_cm: float | None,
                   distance_cm: float | None,
                   output_dir: Path,
                   screen_height_cm: float | None = None) -> dict | None:
    """Interactive evaluation. Returns the report dict, or None on abort.

    predict_provider: callable() -> (screen_x, screen_y) RAW predicted by the
    calibrated pipeline for the current frame (NOT clamped to the screen), or
    None when no face. Called once per UI frame, blocking at camera rate.
    Error is measured on the raw prediction so a runaway model cannot hide
    behind the screen clamp (which folds huge predictions onto the edges and
    makes corner targets look accurate).

    If screen_width_cm / distance_cm are None the report omits cm and deg
    (they would be meaningless) but still reports px and %-of-width, and
    marks itself as geometry-not-configured.
    """
    screen_w, screen_h = screen_size
    have_geo = screen_width_cm is not None and distance_cm is not None
    px_per_cm = (screen_w / screen_width_cm) if have_geo else None
    if not have_geo:
        print("WARNING: screen geometry is not configured. cm and deg "
              "cannot be computed and will be omitted; px and %-of-width "
              "are reported. Set it with 'python main.py screen "
              "--screen-width-cm <W> --distance-cm <D>'.")
    targets = build_targets(screen_w, screen_h)

    window = "gazetrack - evaluation"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    measurements: list[EvalMeasurement] = []
    skipped = 0
    try:
        for i, target in enumerate(targets):
            phase_start = time.perf_counter()
            collecting = False
            predictions: list[tuple[float, float]] = []
            while True:
                elapsed = time.perf_counter() - phase_start
                if not collecting and elapsed >= SETTLE_SECONDS:
                    collecting = True
                    phase_start = time.perf_counter()
                    elapsed = 0.0

                prediction = predict_provider()
                if collecting and prediction is not None:
                    predictions.append(prediction)

                canvas = np.full((screen_h, screen_w, 3), _BG[0],
                                 dtype=np.uint8)
                center = (int(target.x), int(target.y))
                if not collecting:
                    ring = int(22 - 14 * (elapsed / SETTLE_SECONDS))
                    cv2.circle(canvas, center, ring, _DIM, 1, cv2.LINE_AA)
                cv2.circle(canvas, center, 6,
                           _FG if collecting else _DIM, -1, cv2.LINE_AA)
                cv2.circle(canvas, center, 2, _BG, -1, cv2.LINE_AA)
                cv2.putText(canvas,
                            f"target {i + 1}/{len(targets)}    ESC to abort",
                            (24, screen_h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, _DIM, 1, cv2.LINE_AA)
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("Evaluation aborted.")
                    return None
                if collecting and elapsed >= COLLECT_SECONDS:
                    break

            if len(predictions) < MIN_FRAMES:
                skipped += 1
                continue
            arr = np.array(predictions)
            mean_pred = arr.mean(axis=0)
            error = float(np.hypot(mean_pred[0] - target.x,
                                   mean_pred[1] - target.y))
            jitter = float(np.linalg.norm(arr - mean_pred, axis=1).mean())
            oob = ((arr[:, 0] < 0) | (arr[:, 0] > screen_w)
                   | (arr[:, 1] < 0) | (arr[:, 1] > screen_h))
            measurements.append(EvalMeasurement(
                target, error, jitter, len(arr),
                pred_x=float(mean_pred[0]), pred_y=float(mean_pred[1]),
                oob_frac=float(oob.mean())))
    finally:
        cv2.destroyWindow(window)

    if not measurements:
        print("ERROR: no measurements collected.")
        return None

    stats = _zone_stats(measurements, screen_w, px_per_cm, distance_cm)
    region_stats, region_median = _region_stats(measurements, screen_w,
                                                screen_h)
    jitter_mean = float(np.mean([m.jitter_px for m in measurements]))
    jitter_pct = jitter_mean / screen_w * 100
    oob_stats = _print_evidence_table(measurements, screen_w, screen_h)
    _print_report(stats, jitter_mean, jitter_pct, skipped, have_geo)
    flagged = _print_region_grid(region_stats, region_median)

    report = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "geometry_configured": have_geo,
        "screen": {"width_px": screen_w, "height_px": screen_h,
                   "width_cm": screen_width_cm,
                   "height_cm": screen_height_cm,
                   "distance_cm": distance_cm},
        "zones": stats,
        "regions": region_stats,
        "region_median_px": round(region_median, 1),
        "regions_over_threshold": flagged,
        "jitter_mean_px": round(jitter_mean, 1),
        "jitter_mean_pct": round(jitter_pct, 2),
        "out_of_bounds": oob_stats,
        "targets_skipped": skipped,
        "measurements": [
            {"x": round(m.target.x), "y": round(m.target.y),
             "zone": m.target.zone, "error_px": round(m.error_px, 1),
             "pred_x_raw": round(m.pred_x), "pred_y_raw": round(m.pred_y),
             "oob_frac": round(m.oob_frac, 3),
             "jitter_px": round(m.jitter_px, 1), "frames": m.n_frames}
            for m in measurements
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"eval_{stamp}.yaml"
    out_path.write_text(yaml.safe_dump(report, sort_keys=False),
                        encoding="utf-8")
    print(f"report saved to {out_path}")
    return report
