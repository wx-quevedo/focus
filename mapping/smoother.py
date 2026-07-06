"""One-Euro filter for the on-screen dot (layer 2 post-processing).

Chosen over Kalman per spec: simpler (no motion model to tune), and its
speed-adaptive cutoff matches gaze behaviour well - strong smoothing during
fixations (low speed, jitter dominated) and low lag during saccades (high
speed). This filter is what makes the dot feel stable even though the raw
per-frame error is larger.

Reference: Casiez, Roussel, Vogel - "1-euro filter" (CHI 2012).

Tuning intuition (values in screen pixels at ~30 Hz):
- min_cutoff: lower = steadier fixations but more lag. Start at 1.0 Hz.
- beta: higher = faster catch-up on saccades. Start at 0.007.
"""

from __future__ import annotations

import math
import time


class _LowPass:
    def __init__(self) -> None:
        self._value: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar One-Euro filter with explicit timestamps."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: float | None = None

    def filter(self, value: float, timestamp: float | None = None) -> float:
        t = time.perf_counter() if timestamp is None else timestamp
        if self._last_t is None or t <= self._last_t:
            self._last_t = t
            self._x.apply(value, 1.0)
            self._dx.apply(0.0, 1.0)
            return value
        dt = t - self._last_t
        self._last_t = t

        prev = self._x.value
        raw_speed = (value - prev) / dt
        speed = self._dx.apply(raw_speed, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(speed)
        return self._x.apply(value, _alpha(cutoff, dt))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_t = None


class PointSmoother:
    """One-Euro filter for a 2D screen point."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0):
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float,
               timestamp: float | None = None) -> tuple[float, float]:
        t = time.perf_counter() if timestamp is None else timestamp
        return self._fx.filter(x, t), self._fy.filter(y, t)

    def set_params(self, min_cutoff: float, beta: float) -> None:
        """Live-update the filter tuning (used by the settings panel). Only
        changes the cutoff/beta; the filter state and math are untouched."""
        for f in (self._fx, self._fy):
            f.min_cutoff = min_cutoff
            f.beta = beta

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
