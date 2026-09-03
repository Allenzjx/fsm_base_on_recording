"""Small, explicit active/legacy conformance calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from wlr50_clean.conformance_policy import ConformancePolicy, get_conformance_policy


_POLICY = get_conformance_policy()
RELATIVE_LIMIT = _POLICY.active_fraction
LEGACY_RELATIVE_LIMIT = _POLICY.legacy_fraction
JOINT_FLOOR_DEG = _POLICY.floor("joint_endpoint_delta").absolute_allowance
SERVO_VELOCITY_FLOOR_DEG_S = _POLICY.floor("servo_velocity").absolute_allowance
WHEEL_VELOCITY_FLOOR_RAD_S = _POLICY.floor("wheel_velocity").absolute_allowance
WHEEL_INTEGRAL_FLOOR_RAD = _POLICY.floor("wheel_integral").absolute_allowance


def allowed_error(
    reference: float, *, absolute_floor: float, fraction: float | None = None
) -> float:
    selected = RELATIVE_LIMIT if fraction is None else float(fraction)
    if not 0.0 < selected < 1.0:
        raise ValueError("conformance fraction must be between zero and one")
    return max(float(absolute_floor), selected * abs(float(reference)))


def error_percent(
    actual: float,
    reference: float,
    *,
    absolute_floor: float,
    fraction: float | None = None,
) -> float:
    selected = RELATIVE_LIMIT if fraction is None else float(fraction)
    if not 0.0 < selected < 1.0:
        raise ValueError("conformance fraction must be between zero and one")
    denominator = max(abs(float(reference)), float(absolute_floor) / selected)
    return 100.0 * abs(float(actual) - float(reference)) / denominator


def within_contract(
    actual: float,
    reference: float,
    *,
    absolute_floor: float,
    fraction: float | None = None,
) -> bool:
    return abs(float(actual) - float(reference)) <= allowed_error(
        reference, absolute_floor=absolute_floor, fraction=fraction
    ) + 1e-12


def duration_within_contract(
    actual_s: float, reference_s: float, *, fraction: float | None = None
) -> bool:
    if reference_s <= 0.0:
        return abs(actual_s) <= 1e-12
    selected = RELATIVE_LIMIT if fraction is None else float(fraction)
    return (1.0 - selected) * reference_s <= actual_s <= (
        1.0 + selected
    ) * reference_s


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
    within_30_percent: bool
    within_active_tolerance: bool


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
    policy: ConformancePolicy | None = None,
) -> ChannelConformance:
    selected_policy = _POLICY if policy is None else policy
    active_fraction = selected_policy.active_fraction
    legacy_fraction = selected_policy.legacy_fraction
    endpoint_floor = WHEEL_VELOCITY_FLOOR_RAD_S if wheel_channel else JOINT_FLOOR_DEG
    delta_floor = endpoint_floor
    velocity_floor = (
        WHEEL_VELOCITY_FLOOR_RAD_S
        if wheel_channel
        else SERVO_VELOCITY_FLOOR_DEG_S
    )
    endpoint_allowance = allowed_error(
        reference_command_delta,
        absolute_floor=endpoint_floor,
        fraction=active_fraction,
    )
    command_endpoint_error = abs(fsm_command_end - reference_command_end)
    actual_endpoint_error = abs(fsm_actual_end - reference_actual_end)
    command_endpoint_ok = command_endpoint_error <= endpoint_allowance + 1e-12
    actual_endpoint_ok = actual_endpoint_error <= endpoint_allowance + 1e-12
    command_delta_ok = within_contract(
        fsm_command_delta,
        reference_command_delta,
        absolute_floor=delta_floor,
        fraction=active_fraction,
    )
    actual_delta_error = abs(fsm_actual_delta - reference_actual_delta)
    actual_delta_ok = actual_delta_error <= allowed_error(
        reference_command_delta,
        absolute_floor=delta_floor,
        fraction=active_fraction,
    ) + 1e-12
    duration_ok = duration_within_contract(
        actual_duration, reference_duration, fraction=active_fraction
    )
    velocity_ok = within_contract(
        actual_velocity,
        reference_velocity,
        absolute_floor=velocity_floor,
        fraction=active_fraction,
    )
    command_integral_ok = True
    actual_integral_ok = True
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
            fraction=active_fraction,
        )
        actual_integral_ok = within_contract(
            fsm_actual_wheel_integral,
            reference_actual_wheel_integral,
            absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
            fraction=active_fraction,
        )
        command_integral_error = error_percent(
            fsm_command_wheel_integral,
            reference_command_wheel_integral,
            absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
            fraction=active_fraction,
        )
        actual_integral_error = error_percent(
            fsm_actual_wheel_integral,
            reference_actual_wheel_integral,
            absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
            fraction=active_fraction,
        )
    active_ok = (
        command_endpoint_ok
        and actual_endpoint_ok
        and command_delta_ok
        and actual_delta_ok
        and duration_ok
        and velocity_ok
        and command_integral_ok
        and actual_integral_ok
    )
    legacy_endpoint_allowance = allowed_error(
        reference_command_delta,
        absolute_floor=endpoint_floor,
        fraction=legacy_fraction,
    )
    legacy_ok = (
        (wheel_channel or command_endpoint_error <= legacy_endpoint_allowance + 1e-12)
        and (wheel_channel or actual_endpoint_error <= legacy_endpoint_allowance + 1e-12)
        and (
            wheel_channel
            or within_contract(
                fsm_command_delta,
                reference_command_delta,
                absolute_floor=delta_floor,
                fraction=legacy_fraction,
            )
        )
        and (
            wheel_channel
            or actual_delta_error
            <= allowed_error(
                reference_command_delta,
                absolute_floor=delta_floor,
                fraction=legacy_fraction,
            )
            + 1e-12
        )
        and duration_within_contract(
            actual_duration, reference_duration, fraction=legacy_fraction
        )
        and within_contract(
            actual_velocity,
            reference_velocity,
            absolute_floor=velocity_floor,
            fraction=legacy_fraction,
        )
        and (
            not wheel_channel
            or within_contract(
                fsm_command_wheel_integral,
                reference_command_wheel_integral,
                absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
                fraction=legacy_fraction,
            )
        )
        and (
            not wheel_channel
            or within_contract(
                fsm_actual_wheel_integral,
                reference_actual_wheel_integral,
                absolute_floor=WHEEL_INTEGRAL_FLOOR_RAD,
                fraction=legacy_fraction,
            )
        )
    )
    return ChannelConformance(
        commanded_endpoint_error_percent=100.0
        * command_endpoint_error
        / max(abs(reference_command_delta), endpoint_floor / active_fraction),
        actual_endpoint_error_percent=100.0
        * actual_endpoint_error
        / max(abs(reference_command_delta), endpoint_floor / active_fraction),
        endpoint_error_percent=(
            0.0
            if wheel_channel
            else 100.0
            * max(command_endpoint_error, actual_endpoint_error)
            / max(abs(reference_command_delta), endpoint_floor / active_fraction)
        ),
        commanded_delta_error_percent=error_percent(
            fsm_command_delta,
            reference_command_delta,
            absolute_floor=delta_floor,
            fraction=active_fraction,
        ),
        actual_delta_error_percent=100.0
        * actual_delta_error
        / max(abs(reference_command_delta), delta_floor / active_fraction),
        delta_error_percent=(
            0.0
            if wheel_channel
            else max(
                error_percent(
                    fsm_command_delta,
                    reference_command_delta,
                    absolute_floor=delta_floor,
                    fraction=active_fraction,
                ),
                100.0
                * actual_delta_error
                / max(abs(reference_command_delta), delta_floor / active_fraction),
            )
        ),
        duration_error_percent=(
            100.0 * abs(actual_duration - reference_duration) / reference_duration
            if reference_duration > 0.0
            else 0.0
        ),
        velocity_error_percent=error_percent(
            actual_velocity,
            reference_velocity,
            absolute_floor=velocity_floor,
            fraction=active_fraction,
        ),
        commanded_wheel_integral_error_percent=command_integral_error,
        actual_wheel_integral_error_percent=actual_integral_error,
        wheel_integral_error_percent=max(
            command_integral_error, actual_integral_error
        ),
        within_15_percent=legacy_ok,
        within_30_percent=active_ok,
        within_active_tolerance=active_ok,
    )
