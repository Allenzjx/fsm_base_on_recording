"""Small, explicit ±15 percent conformance calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


RELATIVE_LIMIT = 0.15
JOINT_FLOOR_DEG = 2.0
SERVO_VELOCITY_FLOOR_DEG_S = 1.0
WHEEL_VELOCITY_FLOOR_RAD_S = 0.05
WHEEL_INTEGRAL_FLOOR_RAD = 0.05


def allowed_error(reference: float, *, absolute_floor: float) -> float:
    return max(float(absolute_floor), RELATIVE_LIMIT * abs(float(reference)))


def error_percent(actual: float, reference: float, *, absolute_floor: float) -> float:
    denominator = max(abs(float(reference)), float(absolute_floor) / RELATIVE_LIMIT)
    return 100.0 * abs(float(actual) - float(reference)) / denominator


def within_contract(actual: float, reference: float, *, absolute_floor: float) -> bool:
    return abs(float(actual) - float(reference)) <= allowed_error(
        reference, absolute_floor=absolute_floor
    ) + 1e-12


def duration_within_contract(actual_s: float, reference_s: float) -> bool:
    if reference_s <= 0.0:
        return abs(actual_s) <= 1e-12
    return 0.85 * reference_s <= actual_s <= 1.15 * reference_s


def trajectory_rmse(
    actual: Sequence[float],
    reference: Sequence[float],
    *,
    actual_start: float = 0.0,
    reference_start: float = 0.0,
    reference_delta: float | None = None,
    scale_floor: float = 2.0,
) -> tuple[float, float]:
    if len(actual) != len(reference) or not actual:
        raise ValueError("trajectory vectors must be non-empty and equally sized")
    squared = [
        (
            (float(left) - float(actual_start))
            - (float(right) - float(reference_start))
        )
        ** 2
        for left, right in zip(actual, reference)
    ]
    rmse = math.sqrt(sum(squared) / len(squared))
    if reference_delta is None:
        reference_delta = float(reference[-1]) - float(reference[0])
    scale = max(abs(float(reference_delta)), scale_floor)
    return rmse, 100.0 * rmse / scale


@dataclass(frozen=True)
class ChannelConformance:
    commanded_endpoint_error_percent: float
    actual_endpoint_error_percent: float
    endpoint_error_percent: float
    commanded_delta_error_percent: float
    actual_delta_error_percent: float
    delta_error_percent: float
    duration_error_percent: float
    velocity_error_percent: float
    commanded_wheel_integral_error_percent: float
    actual_wheel_integral_error_percent: float
    wheel_integral_error_percent: float
    within_15_percent: bool


def channel_conformance(
    *,
    reference_command_end: float,
    fsm_command_end: float,
    reference_actual_end: float,
    fsm_actual_end: float,
    reference_command_delta: float,
    fsm_command_delta: float,
    reference_actual_delta: float,
    fsm_actual_delta: float,
    reference_duration: float,
    actual_duration: float,
    reference_velocity: float,
    actual_velocity: float,
    reference_command_wheel_integral: float = 0.0,
    fsm_command_wheel_integral: float = 0.0,
    reference_actual_wheel_integral: float = 0.0,
    fsm_actual_wheel_integral: float = 0.0,
    wheel_channel: bool = False,
) -> ChannelConformance:
    endpoint_floor = WHEEL_VELOCITY_FLOOR_RAD_S if wheel_channel else JOINT_FLOOR_DEG
    delta_floor = endpoint_floor
    velocity_floor = (
        WHEEL_VELOCITY_FLOOR_RAD_S
        if wheel_channel
        else SERVO_VELOCITY_FLOOR_DEG_S
    )
    endpoint_allowance = allowed_error(
        reference_command_delta, absolute_floor=endpoint_floor
    )
    command_endpoint_error = abs(fsm_command_end - reference_command_end)
    actual_endpoint_error = abs(fsm_actual_end - reference_actual_end)
    command_endpoint_ok = command_endpoint_error <= endpoint_allowance + 1e-12
    actual_endpoint_ok = actual_endpoint_error <= endpoint_allowance + 1e-12
    command_delta_ok = within_contract(
        fsm_command_delta,
        reference_command_delta,
        absolute_floor=delta_floor,
    )
    actual_delta_error = abs(fsm_actual_delta - reference_actual_delta)
    actual_delta_ok = actual_delta_error <= allowed_error(
        reference_command_delta, absolute_floor=delta_floor
    ) + 1e-12
    duration_ok = duration_within_contract(actual_duration, reference_duration)
    velocity_ok = within_contract(
        actual_velocity, reference_velocity, absolute_floor=velocity_floor
    )
    command_integral_ok = True
    command_integral_error = 0.0
    actual_integral_error = 0.0
    if wheel_channel:
        # Wheel entries are velocity targets and measured wheel velocities, not
        # joint positions.  Position-style endpoint/delta comparisons therefore
        # remain available only as diagnostics and never gate conformance.
        command_endpoint_ok = True
        actual_endpoint_ok = True
        command_delta_ok = True
        actual_delta_ok = True
        command_integral_ok = within_contract(
            fsm_command_wheel_integral,
            reference_command_wheel_integral,
            absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
        )
        command_integral_error = error_percent(
            fsm_command_wheel_integral,
            reference_command_wheel_integral,
            absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
        )
        actual_integral_error = error_percent(
            fsm_actual_wheel_integral,
            reference_actual_wheel_integral,
            absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
        )
    return ChannelConformance(
        commanded_endpoint_error_percent=100.0
        * command_endpoint_error
        / max(abs(reference_command_delta), endpoint_floor / RELATIVE_LIMIT),
        actual_endpoint_error_percent=100.0
        * actual_endpoint_error
        / max(abs(reference_command_delta), endpoint_floor / RELATIVE_LIMIT),
        endpoint_error_percent=(
            0.0
            if wheel_channel
            else 100.0
            * max(command_endpoint_error, actual_endpoint_error)
            / max(abs(reference_command_delta), endpoint_floor / RELATIVE_LIMIT)
        ),
        commanded_delta_error_percent=error_percent(
            fsm_command_delta,
            reference_command_delta,
            absolute_floor=delta_floor,
        ),
        actual_delta_error_percent=100.0
        * actual_delta_error
        / max(abs(reference_command_delta), delta_floor / RELATIVE_LIMIT),
        delta_error_percent=(
            0.0
            if wheel_channel
            else max(
                error_percent(
                    fsm_command_delta,
                    reference_command_delta,
                    absolute_floor=delta_floor,
                ),
                100.0
                * actual_delta_error
                / max(abs(reference_command_delta), delta_floor / RELATIVE_LIMIT),
            )
        ),
        duration_error_percent=(
            100.0 * abs(actual_duration - reference_duration) / reference_duration
            if reference_duration > 0.0
            else 0.0
        ),
        velocity_error_percent=error_percent(
            actual_velocity, reference_velocity, absolute_floor=velocity_floor
        ),
        commanded_wheel_integral_error_percent=command_integral_error,
        actual_wheel_integral_error_percent=actual_integral_error,
        wheel_integral_error_percent=command_integral_error,
        within_15_percent=(
            command_endpoint_ok
            and actual_endpoint_ok
            and command_delta_ok
            and actual_delta_ok
            and duration_ok
            and velocity_ok
            and command_integral_ok
        ),
    )
