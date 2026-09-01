"""Single-retry, reference-bounded feedback correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from wlr50_clean.reference.motion_contract import MotionPhase

from .motion_executor import ACTION_COUNT, FeedbackCorrection, MAX_CORRECTION_FRACTION


@dataclass(frozen=True)
class RecoveryPlan:
    correction: FeedbackCorrection
    reason: str
    blocked_guard: str


class RecoveryPlanner:
    """Accept optional fractional sensor feedback and enforce the hard bound.

    Sensing may expose ``recovery_correction_fractions`` as a 12-vector, a
    channel-name mapping, or implement ``recovery_correction``.  Fractions are
    clipped, never interpreted as raw joint targets, and are applied only as a
    scaling of the compact reference excursion.
    """

    def __init__(self, full12_order: Sequence[str]) -> None:
        if len(tuple(full12_order)) != ACTION_COUNT:
            raise ValueError("recovery planner requires the full12 channel order")
        self._order = tuple(str(item) for item in full12_order)

    def plan(
        self,
        *,
        phase: MotionPhase,
        observation: Any,
        blocked_guard: str,
    ) -> RecoveryPlan:
        provider = getattr(observation, "recovery_correction", None)
        if callable(provider):
            raw = provider(phase.state_id, blocked_guard)
        elif isinstance(observation, Mapping):
            raw = observation.get("recovery_correction_fractions")
        else:
            raw = getattr(observation, "recovery_correction_fractions", None)
        fractions = self._fractions(raw)
        return RecoveryPlan(
            correction=FeedbackCorrection(fractions),
            reason="one reference-bounded feedback retry",
            blocked_guard=blocked_guard,
        )

    def _fractions(self, raw: Any) -> tuple[float, ...]:
        if raw is None:
            values = (0.0,) * ACTION_COUNT
        elif isinstance(raw, Mapping):
            values = tuple(float(raw.get(channel, 0.0)) for channel in self._order)
        else:
            values = tuple(float(item) for item in raw)
            if len(values) != ACTION_COUNT:
                raise ValueError("recovery feedback must contain 12 fractions")
        return tuple(
            min(max(value, -MAX_CORRECTION_FRACTION), MAX_CORRECTION_FRACTION)
            for value in values
        )

