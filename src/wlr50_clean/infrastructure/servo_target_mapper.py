"""Mature command-space servo target shaping, isolated from Isaac APIs.

This is a narrow clean derivation of the 120 Hz requested-to-applied mapping in
the hash-locked mature ``sim_robot_adapter.py``.  It preserves the successful
Recording's fixed 150 deg/s slew and bounded load-error correction without
changing actuator properties, logical FSM commands, or physical state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .command_batch import (
    PHYSICS_DT_S,
    SERVO_COMMAND_SIGN,
    SERVO_ORDER,
    SERVO_REFERENCE_VELOCITY_DEG_S,
    servo_limits_deg,
)


SERVO_TRACKING_CONVERGENCE_BAND_DEG = 0.75
SERVO_TRACKING_COMPENSATION_GAIN = 8.0
SERVO_TRACKING_COMPENSATION_MAX_DEG = 10.0
SERVO_TRACKING_FEEDBACK_INTERVAL_TICKS = 4
# A feedback segment may end on a wheel-only waypoint while its sampled
# high-gain correction is on the opposite side of the nominal target.  Keeping
# a large transient bias forever makes a sub-degree physical perturbation turn
# into a cross-state pose error.  The two-degree floor is the contract's joint
# endpoint allowance; smaller mature carry-over biases remain untouched.
SERVO_STALE_BIAS_FLOOR_DEG = 2.0


class ServoTargetMapperError(ValueError):
    """Raised before an invalid target mapping can reach the articulation."""


@dataclass(frozen=True, slots=True)
class ServoTargetMapping:
    requested_command_deg: tuple[float, ...]
    applied_drive_command_deg: tuple[float, ...]
    tracking_compensation_deg: tuple[float, ...]
    nominal_target_reached: tuple[bool, ...]
    tracking_active: tuple[bool, ...]
    feedback_sample_tick: int
    feedback_sampled: bool


class ServoTargetMapper:
    """Stateful UI-command-space to applied-drive target mapper.

    Requested commands are canonical degrees relative to the captured standing
    pose.  Applied commands persist across scheduler segment boundaries, so an
    ended segment freezes its current bounded tracking bias exactly as in the
    successful mature Recording environment.
    """

    def __init__(
        self,
        standing_pose_deg: Mapping[str, float],
        *,
        physics_dt_s: float = PHYSICS_DT_S,
        servo_rate_deg_s: float = SERVO_REFERENCE_VELOCITY_DEG_S,
        tracking_gain: float = SERVO_TRACKING_COMPENSATION_GAIN,
        tracking_limit_deg: float = SERVO_TRACKING_COMPENSATION_MAX_DEG,
        feedback_interval_ticks: int = SERVO_TRACKING_FEEDBACK_INTERVAL_TICKS,
    ) -> None:
        if set(standing_pose_deg) != set(SERVO_ORDER):
            raise ServoTargetMapperError("standing pose keys must match all eight servos")
        self.standing_pose_deg = {
            name: _finite(standing_pose_deg[name], f"standing_pose_deg[{name}]")
            for name in SERVO_ORDER
        }
        self.physics_dt_s = _positive(physics_dt_s, "physics_dt_s")
        self.servo_rate_deg_s = _positive(servo_rate_deg_s, "servo_rate_deg_s")
        self.maximum_delta_deg = self.physics_dt_s * self.servo_rate_deg_s
        self.tracking_gain = _nonnegative(tracking_gain, "tracking_gain")
        self.tracking_limit_deg = _nonnegative(
            tracking_limit_deg, "tracking_limit_deg"
        )
        try:
            self.feedback_interval_ticks = int(feedback_interval_ticks)
        except (TypeError, ValueError) as exc:
            raise ServoTargetMapperError(
                "feedback_interval_ticks must be a positive integer"
            ) from exc
        if self.feedback_interval_ticks <= 0:
            raise ServoTargetMapperError(
                "feedback_interval_ticks must be a positive integer"
            )

        self._requested = {name: 0.0 for name in SERVO_ORDER}
        self._applied = {name: 0.0 for name in SERVO_ORDER}
        self._nominal_reached = {name: True for name in SERVO_ORDER}
        self._compensation = {name: 0.0 for name in SERVO_ORDER}
        self._tracking_active = {name: False for name in SERVO_ORDER}
        self._retiring_stale_bias = {name: False for name in SERVO_ORDER}
        self._feedback_tick = 0

    @property
    def feedback_tick(self) -> int:
        return self._feedback_tick

    def advance(
        self,
        requested_command_deg: Sequence[float],
        measured_physical_rad: Sequence[float],
        *,
        tracking_servo_names: Sequence[str] = (),
    ) -> ServoTargetMapping:
        """Advance one physics tick and return all eight applied drive targets."""

        requested = _servo_values(requested_command_deg, "requested_command_deg")
        measured_rad = _servo_values(measured_physical_rad, "measured_physical_rad")
        tracking_names = tuple(str(name) for name in tracking_servo_names)
        if len(set(tracking_names)) != len(tracking_names):
            raise ServoTargetMapperError("tracking_servo_names contains duplicates")
        unknown = sorted(set(tracking_names) - set(SERVO_ORDER))
        if unknown:
            raise ServoTargetMapperError(f"unknown tracking servos: {unknown}")
        scheduled_tracking = set(tracking_names)

        measured_deg = {
            name: math.degrees(value)
            for name, value in zip(SERVO_ORDER, measured_rad, strict=True)
        }
        ended_tracking = {
            name
            for name in SERVO_ORDER
            if self._tracking_active[name] and name not in scheduled_tracking
        }

        # A segment end normally preserves the mature carry-over bias.  The
        # sole exception is a bias larger than the contract's two-degree joint
        # floor when the measured joint has already converged to nominal.  That
        # combination is a sampled feedback-phase artifact, not load support;
        # retire it at the same 150 deg/s target slew instead of freezing it
        # into later wheel-only phases.
        for name in SERVO_ORDER:
            if name not in scheduled_tracking:
                self._tracking_active[name] = False
            if name in ended_tracking and self._nominal_reached[name]:
                nominal_physical_deg = (
                    self.standing_pose_deg[name]
                    + SERVO_COMMAND_SIGN[name] * self._requested[name]
                )
                actual_error_deg = nominal_physical_deg - measured_deg[name]
                if (
                    abs(actual_error_deg)
                    <= SERVO_TRACKING_CONVERGENCE_BAND_DEG + 1.0e-12
                    and abs(self._compensation[name])
                    > SERVO_STALE_BIAS_FLOOR_DEG + 1.0e-12
                ):
                    self._retiring_stale_bias[name] = True
        for name, raw_value in zip(SERVO_ORDER, requested, strict=True):
            lower, upper = servo_limits_deg(name)
            value = _clamp(raw_value, lower, upper)
            if not math.isclose(
                value,
                self._requested[name],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                self._nominal_reached[name] = False
                self._compensation[name] = 0.0
                self._retiring_stale_bias[name] = False
            self._requested[name] = value
        for name in scheduled_tracking:
            self._tracking_active[name] = True
            self._retiring_stale_bias[name] = False

        feedback_names = {
            name
            for name in SERVO_ORDER
            if self._nominal_reached[name] and self._tracking_active[name]
        }
        sample_tick = self._feedback_tick
        sample_feedback = sample_tick % self.feedback_interval_ticks == 0
        self._feedback_tick += 1

        for name in SERVO_ORDER:
            requested_value = self._requested[name]
            applied = self._applied[name]
            if not self._nominal_reached[name]:
                delta = requested_value - applied
                if abs(delta) <= self.maximum_delta_deg:
                    applied = requested_value
                    self._nominal_reached[name] = True
                else:
                    applied += math.copysign(self.maximum_delta_deg, delta)
                self._compensation[name] = 0.0
            elif name in feedback_names and sample_feedback:
                nominal_physical_deg = (
                    self.standing_pose_deg[name]
                    + SERVO_COMMAND_SIGN[name] * requested_value
                )
                actual_error_deg = nominal_physical_deg - measured_deg[name]
                correction = tracking_correction_step(
                    actual_error_deg=actual_error_deg,
                    command_sign=SERVO_COMMAND_SIGN[name],
                    previous_correction_deg=self._compensation[name],
                    maximum_delta_deg=self.maximum_delta_deg,
                    gain=self.tracking_gain,
                    limit_deg=self.tracking_limit_deg,
                )
                lower, upper = servo_limits_deg(name)
                applied = _clamp(requested_value + correction, lower, upper)
                self._compensation[name] = applied - requested_value
            elif self._retiring_stale_bias[name]:
                previous = self._compensation[name]
                correction = math.copysign(
                    max(0.0, abs(previous) - self.maximum_delta_deg),
                    previous,
                )
                lower, upper = servo_limits_deg(name)
                applied = _clamp(requested_value + correction, lower, upper)
                self._compensation[name] = applied - requested_value
                if abs(self._compensation[name]) <= 1.0e-12:
                    self._compensation[name] = 0.0
                    self._retiring_stale_bias[name] = False
            lower, upper = servo_limits_deg(name)
            self._applied[name] = _clamp(applied, lower, upper)

        return ServoTargetMapping(
            requested_command_deg=tuple(self._requested[name] for name in SERVO_ORDER),
            applied_drive_command_deg=tuple(self._applied[name] for name in SERVO_ORDER),
            tracking_compensation_deg=tuple(
                self._compensation[name] for name in SERVO_ORDER
            ),
            nominal_target_reached=tuple(
                self._nominal_reached[name] for name in SERVO_ORDER
            ),
            tracking_active=tuple(
                self._tracking_active[name] for name in SERVO_ORDER
            ),
            feedback_sample_tick=sample_tick,
            feedback_sampled=sample_feedback and bool(feedback_names),
        )


def tracking_correction_step(
    *,
    actual_error_deg: float,
    command_sign: float,
    previous_correction_deg: float,
    maximum_delta_deg: float,
    gain: float = SERVO_TRACKING_COMPENSATION_GAIN,
    limit_deg: float = SERVO_TRACKING_COMPENSATION_MAX_DEG,
) -> float:
    """Advance the mature gain/clamp/slew tracking law by one feedback sample."""

    error = _finite(actual_error_deg, "actual_error_deg")
    sign = _finite(command_sign, "command_sign")
    if math.isclose(sign, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ServoTargetMapperError("command_sign cannot be zero")
    previous = _finite(previous_correction_deg, "previous_correction_deg")
    maximum_delta = _nonnegative(maximum_delta_deg, "maximum_delta_deg")
    correction_limit = _nonnegative(limit_deg, "limit_deg")
    desired = _clamp(
        (error / sign) * _nonnegative(gain, "gain"),
        -correction_limit,
        correction_limit,
    )
    delta = _clamp(desired - previous, -maximum_delta, maximum_delta)
    return _clamp(previous + delta, -correction_limit, correction_limit)


def _servo_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ServoTargetMapperError(f"{label} must be numeric") from exc
    if len(result) != len(SERVO_ORDER):
        raise ServoTargetMapperError(f"{label} must contain eight values")
    if any(not math.isfinite(value) for value in result):
        raise ServoTargetMapperError(f"{label} contains a non-finite value")
    return result


def _positive(value: float, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ServoTargetMapperError(f"{label} must be positive")
    return result


def _nonnegative(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ServoTargetMapperError(f"{label} cannot be negative")
    return result


def _finite(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ServoTargetMapperError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ServoTargetMapperError(f"{label} must be finite")
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))
