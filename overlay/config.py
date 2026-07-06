"""Persistent overlay settings (PHASE-02A.1).

Tunables for the gaze cursor, smoothing and fade, editable live from the
settings panel and persisted to data/overlay.yaml. Precedence follows the
project pattern: explicit CLI flag > overlay.yaml > built-in default.

The config is a plain dataclass so 2B/2C can extend it (a logging section, an
alarms section) without changing the resolution logic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "overlay.yaml"


@dataclass
class OverlayConfig:
    # Cursor
    dot_size: int = 40             # ring outer diameter, px
    dot_opacity: float = 0.85      # peak opacity 0-1
    dot_color: tuple = (235, 235, 228)  # neutral off-white
    deadzone: float = 12.0         # px of movement held still
    # Smoothing (One-Euro)
    min_cutoff: float = 1.0        # Hz; lower = steadier, more lag
    beta: float = 0.007            # speed coefficient; higher = snappier
    # Fade
    pose_margin: float = 5.0       # deg beyond calibrated head range
    offscreen_margin: float = 0.10  # fraction of width off-screen
    fade_ms: int = 250             # fade in/out duration
    # Logging (2B): gaze-zone grid over the screen
    grid_cols: int = 6
    grid_rows: int = 3


DEFAULTS = OverlayConfig()


def _to_yaml(cfg: OverlayConfig) -> dict:
    return {
        "dot_size": int(cfg.dot_size),
        "dot_opacity": float(cfg.dot_opacity),
        "dot_color": list(cfg.dot_color),
        "deadzone": float(cfg.deadzone),
        "min_cutoff": float(cfg.min_cutoff),
        "beta": float(cfg.beta),
        "pose_margin": float(cfg.pose_margin),
        "offscreen_margin": float(cfg.offscreen_margin),
        "fade_ms": int(cfg.fade_ms),
        "grid_cols": int(cfg.grid_cols),
        "grid_rows": int(cfg.grid_rows),
    }


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        print(f"WARNING: {CONFIG_PATH} is not valid YAML; ignoring it.")
        return {}


def save_config(cfg: OverlayConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(_to_yaml(cfg), sort_keys=False),
                           encoding="utf-8")


def _coerce(name: str, value):
    if name == "dot_color":
        return tuple(int(v) for v in value)
    if name in ("dot_size", "fade_ms", "grid_cols", "grid_rows"):
        return int(value)
    return float(value)


def resolve(overrides: dict | None = None) -> OverlayConfig:
    """Build the effective config: flag override > overlay.yaml > default.

    overrides maps field name -> CLI value (or None when the flag was absent).
    """
    overrides = overrides or {}
    stored = load_config()
    values = {}
    for f in fields(OverlayConfig):
        if overrides.get(f.name) is not None:
            values[f.name] = _coerce(f.name, overrides[f.name])
        elif f.name in stored and stored[f.name] is not None:
            values[f.name] = _coerce(f.name, stored[f.name])
        else:
            values[f.name] = getattr(DEFAULTS, f.name)
    return replace(DEFAULTS, **values)
