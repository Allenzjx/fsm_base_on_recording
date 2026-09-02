"""Small reference-bounded drive corrections triggered by live phase evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from wlr50_clean.reference.motion_contract import DriveFeedbackSpec


ACTION_COUNT = 12
SERVO_COUNT = 8
PHYSICS_HZ = 120.0
MAX_CUMULATIVE_CORRECTION_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class DriveFeedback:
    """One canonical post-native correction request and its audit evidence.

    ``bias_full12`` follows the contract's mixed-unit action order: degrees for
    the eight servo channels and rad/s for the four wheel channels.
    """

    bias_full12: tuple[float, ...]
    active: bool
    just_triggered: bool
    tick_index: int | None
    trigger_tick: int | None
    observed_deg: float | None
    reference_deg: float | None
    peak_fraction_of_reference: float
    cumulative_fraction_of_reference: float
    logical_bias_rad_s: float
    reference_wheel_integral_rad: float
    additional_wheel_integral_rad: float
    probe_channel: str | None
    probe_channel_index: int | None
    correction_channel: str | None
    correction_channel_index: int | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "wlr50_clean.drive_feedback.v1",
            "bias_full12": list(self.bias_full12),
            "active": self.active,
            "just_triggered": self.just_triggered,
            "tick_index": self.tick_index,
            "trigger_tick": self.trigger_tick,
            "observed_deg": self.observed_deg,
            "reference_deg": self.reference_deg,
            "peak_fraction_of_reference": self.peak_fraction_of_reference,
            "cumulative_fraction_of_reference": self.cumulative_fraction_of_reference,
            "logical_bias_rad_s": self.logical_bias_rad_s,
            "reference_wheel_integral_rad": self.reference_wheel_integral_rad,
            "additional_wheel_integral_rad": self.additional_wheel_integral_rad,
            "probe_channel": self.probe_channel,
            "probe_channel_index": self.probe_channel_index,
            "correction_channel": self.correction_channel,
            "correction_channel_index": self.correction_channel_index,
            "reason": self.reason,
        }


class ReferenceBoundedDriveFeedback:
    """Latch a direct late-P09 RRK deficit and continue causal wheel carry.

    Two locked pre-endpoint rear-right-knee samples arm an eight-tick extension
    of the existing front-left wheel target.  The extension ends before the
    next decision and the signed P10 velocity guard remains authoritative.
    """

    def __init__(self) -> None:
        self._state_id: str | None = None
        self._trigger_tick: int | None = None
        self._trigger_consumed = False
        self._consecutive_lag_samples = 0

    @property
    def trigger_latched(self) -> bool:
        return self._trigger_tick is not None

    def update(
        self,
        *,
        state_id: str,
        motion_tick_index: int | None,
        actual_full12: Sequence[float] | None,
        spec: DriveFeedbackSpec | None,
    ) -> DriveFeedback:
        tick = None if motion_tick_index is None else int(motion_tick_index)
        if state_id != self._state_id:
            self._state_id = state_id
            self._trigger_tick = None
            self._trigger_consumed = False
            self._consecutive_lag_samples = 0
        if tick == 0:
            self._trigger_tick = None
            self._consecutive_lag_samples = 0

        if spec is not None:
            _validate_runtime_spec(spec, state_id=state_id)

        actual = _full12_or_none(actual_full12)
        just_triggered = False
        channel_index = None if spec is None else spec.probe_channel_index
        observed = (
            None if actual is None or channel_index is None else actual[channel_index]
        )
        probe_by_tick = (
            {}
            if spec is None
            else {
                probe.motion_tick: probe.reference_actual_deg
                for probe in spec.probe_samples
            }
        )
        reference_probe = probe_by_tick.get(tick)
        if spec is not None and reference_probe is not None:
            lag = (
                None
                if observed is None or reference_probe is None
                else reference_probe - observed
            )
            if lag is not None and lag + 1.0e-12 >= spec.lag_threshold_deg:
                self._consecutive_lag_samples += 1
            else:
                self._consecutive_lag_samples = 0
            if (
                not self._trigger_consumed
                and tick == spec.probe_samples[-1].motion_tick
                and self._consecutive_lag_samples
                >= spec.required_consecutive_samples
            ):
                self._trigger_tick = tick
                self._trigger_consumed = True
                just_triggered = True

        active = bool(
            spec is not None
            and tick is not None
            and self._trigger_tick is not None
            and spec.first_bias_tick <= tick <= spec.last_bias_tick
        )
        bias = [0.0] * ACTION_COUNT
        if active and spec is not None:
            bias[spec.correction_channel_index] = spec.logical_bias_rad_s

        cumulative_fraction = (
            0.0 if spec is None else spec.cumulative_fraction_of_reference
        )
        if cumulative_fraction > MAX_CUMULATIVE_CORRECTION_FRACTION + 1.0e-12:
            raise RuntimeError(f"{state_id} drive feedback exceeds the 15% budget")
        triggered = spec is not None and self._trigger_tick is not None
        return DriveFeedback(
            bias_full12=tuple(bias),
            active=active,
            just_triggered=just_triggered,
            tick_index=tick,
            trigger_tick=self._trigger_tick if triggered else None,
            observed_deg=observed,
            reference_deg=(
                reference_probe if spec is not None else None
            ),
            # A wheel-target continuation adds no new peak-amplitude budget;
            # only its additional signed time integral is bounded.
            peak_fraction_of_reference=0.0,
            cumulative_fraction_of_reference=(
                cumulative_fraction if triggered else 0.0
            ),
            logical_bias_rad_s=(
                0.0 if spec is None else spec.logical_bias_rad_s
            ),
            reference_wheel_integral_rad=(
                0.0 if spec is None else spec.reference_wheel_integral_rad
            ),
            additional_wheel_integral_rad=(
                0.0 if spec is None else spec.additional_wheel_integral_rad
            ),
            probe_channel=spec.probe_channel if spec is not None else None,
            probe_channel_index=(
                spec.probe_channel_index if spec is not None else None
            ),
            correction_channel=(
                spec.correction_channel if spec is not None else None
            ),
            correction_channel_index=(
                spec.correction_channel_index if spec is not None else None
            ),
            reason=(
                f"live {state_id} {spec.probe_channel} pre-endpoint deficit "
                f"requests {spec.correction_channel} verify-tail wheel carry"
                if triggered and spec is not None
                else "no live reference-corridor deficit latched"
            ),
        )


def _validate_runtime_spec(spec: DriveFeedbackSpec, *, state_id: str) -> None:
    values = (
        spec.logical_bias_rad_s,
        spec.reference_wheel_integral_rad,
        spec.additional_wheel_integral_rad,
        spec.cumulative_fraction_of_reference,
    )
    active_ticks = spec.last_bias_tick - spec.first_bias_tick + 1
    expected_integral = spec.logical_bias_rad_s * active_ticks / PHYSICS_HZ
    reference = abs(spec.reference_wheel_integral_rad)
    expected_fraction = (
        math.inf if reference == 0.0 else abs(expected_integral) / reference
    )
    if (
        spec.kind != "verify_tail_wheel_carry_alignment"
        or spec.correction_channel_index < SERVO_COUNT
        or spec.correction_channel_index >= ACTION_COUNT
        or any(not math.isfinite(value) for value in values)
        or active_ticks <= 0
        or spec.logical_bias_rad_s * spec.reference_wheel_integral_rad <= 0.0
        or abs(expected_integral - spec.additional_wheel_integral_rad) > 1.0e-12
        or abs(expected_fraction - spec.cumulative_fraction_of_reference) > 1.0e-12
        or expected_fraction > MAX_CUMULATIVE_CORRECTION_FRACTION + 1.0e-12
    ):
        raise RuntimeError(f"{state_id} drive feedback is not a valid wheel carry")


def _full12_or_none(values: Sequence[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(result) != ACTION_COUNT or any(not math.isfinite(value) for value in result):
        return None
    return result
