"""Persistent physical screen geometry (cm) and resolution detection.

Physical geometry is REQUIRED to turn an on-screen pixel error into cm or
degrees. Without it those numbers are meaningless: a 900 px mean error is
~2.3 cm on a 49" ultrawide (~120 cm / 5120 px) but ~6 cm on a 15" laptop
panel. The pixel error and the error as a percentage of screen width are the
only figures that are valid regardless of geometry.

Values persist to data/screen.yaml so calibrate / demo / evaluate all read
the same geometry automatically. There is deliberately NO built-in cm
default: an unconfigured run is reported as unconfigured, never silently
evaluated against a wrong guess. Precedence: explicit flag > screen.yaml >
unset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from overlay.dot import get_screen_size

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "screen.yaml"

# Sanity band for a desktop setup; outside it the value is probably a typo
# (e.g. mm entered as cm, or inches).
PLAUSIBLE_WIDTH_CM = (20.0, 200.0)
PLAUSIBLE_DISTANCE_CM = (30.0, 150.0)


@dataclass
class ScreenGeometry:
    width_px: int
    height_px: int
    width_cm: float | None
    height_cm: float | None
    distance_cm: float | None
    sources: dict  # per field: "flag" | "config" | "unset"

    @property
    def configured(self) -> bool:
        """True when cm and distance are known (cm/deg conversions valid)."""
        return self.width_cm is not None and self.distance_cm is not None

    @property
    def px_per_cm(self) -> float | None:
        if not self.width_cm:
            return None
        return self.width_px / self.width_cm

    def warnings(self) -> list[str]:
        out: list[str] = []
        lo, hi = PLAUSIBLE_WIDTH_CM
        if self.width_cm is not None and not (lo <= self.width_cm <= hi):
            out.append(f"screen width {self.width_cm:g} cm is outside the "
                       f"plausible {lo:g}-{hi:g} cm band; check the value.")
        lo, hi = PLAUSIBLE_DISTANCE_CM
        if self.distance_cm is not None and not (lo <= self.distance_cm
                                                 <= hi):
            out.append(f"viewing distance {self.distance_cm:g} cm is outside "
                       f"the plausible {lo:g}-{hi:g} cm band; check it.")
        return out


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        print(f"WARNING: {CONFIG_PATH} is not valid YAML; ignoring it.")
        return {}


def save_config(values: dict) -> None:
    """Merge non-None values into data/screen.yaml."""
    clean = {k: v for k, v in values.items() if v is not None}
    merged = {**load_config(), **clean}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(merged, sort_keys=False),
                           encoding="utf-8")


def resolve(width_cm: float | None = None,
            height_cm: float | None = None,
            distance_cm: float | None = None) -> ScreenGeometry:
    """Resolve geometry with precedence flag > screen.yaml > unset."""
    width_px, height_px = get_screen_size()
    cfg = load_config()

    def pick(flag, key):
        if flag is not None:
            return float(flag), "flag"
        if cfg.get(key) is not None:
            return float(cfg[key]), "config"
        return None, "unset"

    wcm, ws = pick(width_cm, "width_cm")
    hcm, hs = pick(height_cm, "height_cm")
    dcm, ds = pick(distance_cm, "distance_cm")
    return ScreenGeometry(
        width_px=width_px, height_px=height_px,
        width_cm=wcm, height_cm=hcm, distance_cm=dcm,
        sources={"width_cm": ws, "height_cm": hs, "distance_cm": ds})
