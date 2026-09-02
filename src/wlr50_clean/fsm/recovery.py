"""Single-retry, reference-bounded feedback correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from wlr50_clean.reference.motion_contract import MotionPhase

from .motion_executor import (
    ACTION_COUNT,
    SERVO_COUNT,
    FeedbackCorrection,
    MAX_CORRECTION_FRACTION,
)


_FINAL_POSE_GUARD = "final_joint_pose_compatible"
_MISSING = object()


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
        blocker_evidence: Any = None,
    ) -> RecoveryPlan:
        provider = getattr(observation, "recovery_correction", None)
        if callable(provider):
            raw = provider(phase.state_id, blocked_guard)
        elif isinstance(observation, Mapping):
            raw = observation.get("recovery_correction_fractions", _MISSING)
        else:
            raw = getattr(observation, "recovery_correction_fractions", _MISSING)

        reason = "one reference-bounded feedback retry"
        if raw is _MISSING:
            live_fractions = self._live_final_pose_fractions(
                phase=phase,
                blocked_guard=blocked_guard,
                blocker_evidence=blocker_evidence,
            )
            if live_fractions is None:
                fractions = self._fractions(None)
            else:
                fractions = live_fractions
                reason = "live final-pose corridor correction"
        else:
            # A sensing-layer provider remains authoritative, including an
            # explicit None (the established request for a zero correction).
            fractions = self._fractions(raw)
        return RecoveryPlan(
            correction=FeedbackCorrection(fractions),
            reason=reason,
            blocked_guard=blocked_guard,
        )

    def _live_final_pose_fractions(
        self,
        *,
        phase: MotionPhase,
        blocked_guard: str,
        blocker_evidence: Any,
    ) -> tuple[float, ...] | None:
        """Derive one bounded P13 retry from the live final-pose error.

        ``error_deg`` is actual minus the v010 measured endpoint, so its
        negation is exactly the desired endpoint shift.  Only reported
        out-of-corridor servo channels participate; wheels, zero-delta
        channels, missing evidence, and already-compatible channels stay zero.
        """

        if phase.state_id != "P13" or blocked_guard != _FINAL_POSE_GUARD:
            return None
        if isinstance(blocker_evidence, Mapping):
            evidence_value = blocker_evidence.get("value", blocker_evidence)
        else:
            evidence_value = getattr(blocker_evidence, "value", None)
        if not isinstance(evidence_value, Mapping):
            return None

        values = [0.0] * ACTION_COUNT
        active_channels = set(phase.active_channels)
        for index, channel in enumerate(self._order[:SERVO_COUNT]):
            if channel not in active_channels or channel not in evidence_value:
                continue
            channel_evidence = evidence_value[channel]
            if not isinstance(channel_evidence, Mapping):
                raise ValueError(
                    f"{channel}: final-pose recovery evidence must be a mapping"
                )
            try:
                error_deg = float(channel_evidence["error_deg"])
                limit_deg = float(channel_evidence["limit_deg"])
                delta_deg = float(phase.delta_full12[index])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{channel}: invalid final-pose recovery evidence"
                ) from exc
            if not math.isfinite(error_deg) or not math.isfinite(limit_deg):
                raise ValueError(
                    f"{channel}: final-pose recovery evidence must be finite"
                )
            if limit_deg < 0.0:
                raise ValueError(
                    f"{channel}: final-pose recovery limit must be non-negative"
                )
            if not math.isfinite(delta_deg):
                raise ValueError(f"{channel}: phase delta must be finite")
            if (
                abs(delta_deg) <= 1e-12
                or abs(error_deg) <= limit_deg + 1e-12
            ):
                continue
            # reference_actual_endpoint - actual == -error_deg
            values[index] = -error_deg / delta_deg
        return self._fractions(values)

    def _fractions(self, raw: Any) -> tuple[float, ...]:
        if raw is None:
            values = (0.0,) * ACTION_COUNT
        elif isinstance(raw, Mapping):
            values = tuple(float(raw.get(channel, 0.0)) for channel in self._order)
        else:
            values = tuple(float(item) for item in raw)
            if len(values) != ACTION_COUNT:
                raise ValueError("recovery feedback must contain 12 fractions")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("recovery feedback must contain only finite fractions")
        return tuple(
            min(max(value, -MAX_CORRECTION_FRACTION), MAX_CORRECTION_FRACTION)
            for value in values
        )
