"""Shared 120 Hz evidence state for measured final-wheel decay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WheelDecayStatus:
    eligible: bool
    passed: bool
    maximum_abs_velocity_rad_s: float
    stable_for_s: float
    stable_since_s: float | None


class WheelDecayDebounce:
    """Require one uninterrupted in-envelope suffix after zero-wheel readback."""

    def __init__(self) -> None:
        self._stable_since_s: float | None = None

    def reset(self) -> None:
        self._stable_since_s = None

    def update(
        self,
        *,
        sim_time_s: float,
        measured_velocity_rad_s: Sequence[float],
        commanded_velocity_rad_s: Sequence[float],
        threshold_rad_s: float,
        debounce_s: float,
    ) -> WheelDecayStatus:
        measured = _finite_four(measured_velocity_rad_s, "measured wheel velocity")
        commanded = _finite_four(commanded_velocity_rad_s, "commanded wheel velocity")
        now = float(sim_time_s)
        threshold = float(threshold_rad_s)
        debounce = float(debounce_s)
        if not math.isfinite(now):
            raise ValueError("wheel-decay time must be finite")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("wheel-decay threshold must be finite and nonnegative")
        if not math.isfinite(debounce) or debounce <= 0.0:
            raise ValueError("wheel-decay debounce must be finite and positive")

        maximum = max(abs(value) for value in measured)
        eligible = all(abs(value) <= 1.0e-9 for value in commanded)
        if not eligible or maximum > threshold + 1.0e-12:
            self._stable_since_s = None
        elif self._stable_since_s is None:
            self._stable_since_s = now
        stable_for = (
            0.0
            if self._stable_since_s is None
            else max(0.0, now - self._stable_since_s)
        )
        return WheelDecayStatus(
            eligible=eligible,
            passed=eligible
            and maximum <= threshold + 1.0e-12
            and stable_for + 1.0e-12 >= debounce,
            maximum_abs_velocity_rad_s=maximum,
            stable_for_s=stable_for,
            stable_since_s=self._stable_since_s,
        )


def _finite_four(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if len(result) != 4 or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain four finite values")
    return result
