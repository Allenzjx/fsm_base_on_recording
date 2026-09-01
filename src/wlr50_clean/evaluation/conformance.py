"""Offline per-channel conformance metrics for one compact-contract trial."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from wlr50_clean.reference.similarity import (
    SERVO_VELOCITY_FLOOR_DEG_S,
    WHEEL_VELOCITY_FLOOR_RAD_S,
    channel_conformance,
    error_percent,
    trajectory_rmse,
    within_contract,
)

from .comparison import PHASE_IDS


SIMILARITY_COLUMNS = (
    "phase", "state", "channel",
    "reference_start", "reference_end", "reference_delta",
    "reference_active_duration", "reference_average_velocity",
    "reference_peak_velocity", "reference_wheel_integral",
    "fsm_start", "fsm_command_end", "fsm_actual_end", "fsm_delta",
    "fsm_active_duration", "fsm_average_velocity", "fsm_peak_velocity",
    "fsm_wheel_integral", "endpoint_error_percent", "delta_error_percent",
    "duration_error_percent", "velocity_error_percent",
    "wheel_integral_error_percent", "within_15_percent",
    "reference_actual_end", "reference_actual_delta",
    "reference_actual_wheel_integral", "fsm_command_delta",
    "fsm_actual_delta", "fsm_command_wheel_integral",
    "fsm_actual_wheel_integral", "command_endpoint_error_percent",
    "actual_endpoint_error_percent", "command_delta_error_percent",
    "actual_delta_error_percent", "average_velocity_error_percent",
    "peak_velocity_error_percent", "command_wheel_integral_error_percent",
    "actual_wheel_integral_error_percent", "trajectory_rmse",
    "trajectory_rmse_percent", "active_sample_count",
    "reference_measured_average_velocity", "fsm_measured_average_velocity",
    "measured_average_velocity_error_percent", "reference_measured_peak_velocity",
    "fsm_measured_peak_velocity", "measured_peak_velocity_error_percent",
)


PHYSICS_DT_S = 1.0 / 120.0


class TrialAnalysisError(ValueError):
    """Run evidence is missing, contradictory, or outside the contract."""


def _number(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] is not None:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def _time(row: Mapping[str, Any]) -> float:
    value = _number(row, "simulation_time_s", "sim_time_s", "time_s")
    if value is None:
        raise TrialAnalysisError("evidence row has no finite simulation time")
    return value


def _phase(row: Mapping[str, Any]) -> str:
    return str(row.get("state_id") or row.get("phase") or "")


def _vector(row: Mapping[str, Any], *names: str) -> tuple[float, ...] | None:
    for name in names:
        raw = row.get(name)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            try:
                values = tuple(float(item) for item in raw)
            except (TypeError, ValueError):
                continue
            if len(values) == 12 and all(math.isfinite(item) for item in values):
                return values
    return None


def _phase_windows(transitions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for phase in PHASE_IDS:
        rows = sorted((row for row in transitions if _phase(row) == phase), key=_time)
        done = [row for row in rows if str(row.get("to_lifecycle")) == "DONE"]
        if len(done) != 1:
            raise TrialAnalysisError(f"{phase}: expected exactly one DONE transition")
        complete = _time(done[0])
        starts = [
            row for row in rows
            if str(row.get("to_lifecycle")) == "EXECUTE_MOTION" and _time(row) <= complete
        ]
        if not starts:
            raise TrialAnalysisError(f"{phase}: missing EXECUTE_MOTION transition")
        # The first execution is the normal action; later ones are RECOVERY and
        # cannot make nominal duration/delta appear closer to the contract.
        start = _time(starts[0])
        verifies = [
            row for row in rows
            if str(row.get("to_lifecycle")) == "VERIFY_RESULT" and start <= _time(row) <= complete
        ]
        if not verifies:
            raise TrialAnalysisError(f"{phase}: missing VERIFY_RESULT transition")
        end = _time(verifies[0])
        if not start < end <= complete + 1.0e-9:
            raise TrialAnalysisError(f"{phase}: invalid motion/verification clock")
        result[phase] = {
            "entry_time_s": min((_time(row) for row in rows), default=start),
            "motion_start_s": start, "motion_end_s": end,
            "completion_time_s": complete, "active_duration_s": end - start,
        }
    completion = [result[phase]["completion_time_s"] for phase in PHASE_IDS]
    if any(right + 1.0e-9 < left for left, right in zip(completion, completion[1:])):
        raise TrialAnalysisError("P01--P13 completion order is not monotonic")
    for previous, current in zip(PHASE_IDS, PHASE_IDS[1:]):
        if result[current]["motion_start_s"] + 1.0e-9 < result[previous]["completion_time_s"]:
            raise TrialAnalysisError(f"{current}: motion began before {previous} completed")
    return result


def _samples(
    rows: Sequence[Mapping[str, Any]], phase: str, start: float, end: float
) -> list[Mapping[str, Any]]:
    selected = []
    for row in rows:
        when = _time(row)
        row_phase = _phase(row)
        if start - 1.0e-9 <= when <= end + 1.0e-9 and (not row_phase or row_phase == phase):
            selected.append(row)
    return sorted(selected, key=_time)


def _command_entry_vector(
    commands: Sequence[Mapping[str, Any]], start: float, fallback: Sequence[float]
) -> tuple[float, ...]:
    prior = sorted((row for row in commands if _time(row) < start - 1.0e-9), key=_time)
    if prior:
        value = _vector(prior[-1], "full12", "command_full12", "commanded_full12")
        if value is not None:
            return value
    return tuple(float(item) for item in fallback)


def _velocity(row: Mapping[str, Any], channel: str, index: int) -> float | None:
    collection_name = "joints" if index < 8 else "wheels"
    value_name = "velocity_deg_s" if index < 8 else "velocity_rad_s"
    collection = row.get(collection_name)
    if isinstance(collection, Mapping) and isinstance(collection.get(channel), Mapping):
        value = _number(collection[channel], value_name, "velocity")
        if value is not None:
            return value
    values = _vector(row, "velocity_full12", "actual_velocity_full12")
    return None if values is None else values[index]


def _integral(
    rows: Sequence[Mapping[str, Any]], index: int, names: Sequence[str], *,
    trapezoid: bool = False, absolute: bool = False,
) -> float:
    if len(rows) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(rows, rows[1:]):
        vector = _vector(left, *names)
        if vector is None:
            raise TrialAnalysisError("full12 evidence is missing while integrating a wheel")
        value = abs(vector[index]) if absolute else vector[index]
        if trapezoid:
            right_vector = _vector(right, *names)
            if right_vector is None:
                raise TrialAnalysisError("full12 evidence is missing while integrating a wheel")
            right_value = abs(right_vector[index]) if absolute else right_vector[index]
            value = 0.5 * (value + right_value)
        total += value * (_time(right) - _time(left))
    return total


def _interpolate(progress: Sequence[float], values: Sequence[float], query: float) -> float:
    if query <= progress[0]:
        return values[0]
    if query >= progress[-1]:
        return values[-1]
    for index in range(1, len(progress)):
        if query <= progress[index]:
            width = progress[index] - progress[index - 1]
            fraction = 0.0 if width <= 0.0 else (query - progress[index - 1]) / width
            return values[index - 1] + fraction * (values[index] - values[index - 1])
    return values[-1]


def _trajectory_error(
    phase: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], index: int,
    start: float, end: float,
) -> tuple[float, float]:
    if index >= 8 or len(observations) < 2:
        return 0.0, 0.0
    reference_rows = phase.get("reference_actual", {}).get("trajectory_samples_normalized", [])
    if not isinstance(reference_rows, list) or len(reference_rows) < 2:
        raise TrialAnalysisError(f"{phase.get('state_id')}: reference trajectory is missing")
    reference_progress = [float(row["progress"]) for row in reference_rows]
    reference_values = [float(row["actual_full12"][index]) for row in reference_rows]
    actual_progress = [max(0.0, min(1.0, (_time(row) - start) / (end - start))) for row in observations]
    actual_values = []
    for row in observations:
        vector = _vector(row, "actual_full12")
        if vector is None:
            raise TrialAnalysisError("observation is missing actual_full12")
        actual_values.append(vector[index])
    aligned_actual = [_interpolate(actual_progress, actual_values, item) for item in reference_progress]
    return trajectory_rmse(
        aligned_actual, reference_values, actual_start=aligned_actual[0],
        reference_start=reference_values[0],
        reference_delta=float(phase["delta_full12"][index]),
    )


def _similarity_rows(
    contract: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]], windows: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    order = tuple(str(item) for item in contract.get("full12_order", ()))
    phases = contract.get("phases")
    if len(order) != 12 or not isinstance(phases, list) or len(phases) != 13:
        raise TrialAnalysisError("compact contract does not contain full12/P01--P13")
    result: list[dict[str, Any]] = []
    for phase in phases:
        phase_id = str(phase.get("state_id"))
        clock = windows[phase_id]
        start, end = float(clock["motion_start_s"]), float(clock["motion_end_s"])
        completion = float(clock["completion_time_s"])
        command_rows = _samples(commands, phase_id, start, end)
        # Runtime observations are emitted after each 120 Hz physics step.  The
        # sample immediately after motion_end is therefore the physical response
        # to the final nominal command and belongs to the measured phase window.
        active_observations = _samples(
            observations, phase_id, start, end + PHYSICS_DT_S
        )
        result_observations = _samples(observations, phase_id, start, completion)
        if len(command_rows) < 2 or len(active_observations) < 2 or not result_observations:
            raise TrialAnalysisError(f"{phase_id}: insufficient command/observation samples")
        command_start = _command_entry_vector(commands, start, phase["start_full12"])
        command_end = _vector(command_rows[-1], "full12", "command_full12", "commanded_full12")
        actual_start = _vector(active_observations[0], "actual_full12")
        actual_end = _vector(result_observations[-1], "actual_full12")
        if any(item is None for item in (command_end, actual_start, actual_end)):
            raise TrialAnalysisError(f"{phase_id}: full12 endpoint evidence is missing")
        assert command_end is not None and actual_start is not None and actual_end is not None
        reference_actual = phase.get("reference_actual", {})
        reference_result = phase.get("reference_result_observation", {})
        for channel in tuple(str(item) for item in phase.get("active_channels", ())):
            if channel not in order:
                raise TrialAnalysisError(f"{phase_id}: unknown active channel {channel}")
            index = order.index(channel)
            wheel = index >= 8
            velocities = [
                abs(value) for row in active_observations
                if (value := _velocity(row, channel, index)) is not None
            ]
            if not velocities:
                raise TrialAnalysisError(f"{phase_id}/{channel}: measured velocity is missing")
            measured_fsm_average = sum(velocities) / len(velocities)
            measured_fsm_peak = max(velocities)
            reference_phase_averages = reference_actual.get(
                "phase_average_abs_velocity",
                # Backward compatibility for synthetic/legacy test contracts;
                # extracted v010 contracts always carry the literal phase mean.
                reference_actual.get("active_window_average_abs_velocity", {}),
            )
            if not isinstance(reference_phase_averages, Mapping):
                raise TrialAnalysisError(
                    f"{phase_id}/{channel}: reference phase velocity is missing"
                )
            measured_ref_average = float(reference_phase_averages[channel])
            measured_ref_peak = float(reference_actual["peak_abs_velocity"][channel])
            command_metrics = phase.get("command_metrics", {})
            if wheel:
                try:
                    ref_average = float(
                        command_metrics[
                            "wheel_time_weighted_average_abs_target_rad_s"
                        ][channel]
                    )
                    ref_peak = float(
                        command_metrics["wheel_peak_abs_target_rad_s"][channel]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise TrialAnalysisError(
                        f"{phase_id}/{channel}: wheel target metrics are missing"
                    ) from exc
                command_abs_integral = _integral(
                    command_rows,
                    index,
                    ("full12", "command_full12", "commanded_full12"),
                    absolute=True,
                )
                fsm_average = command_abs_integral / max(
                    float(clock["active_duration_s"]), 1.0e-12
                )
                fsm_peak = max(
                    abs(vector[index])
                    for row in command_rows
                    if (
                        vector := _vector(
                            row,
                            "full12",
                            "command_full12",
                            "commanded_full12",
                        )
                    )
                    is not None
                )
            else:
                ref_average = measured_ref_average
                ref_peak = measured_ref_peak
                fsm_average = measured_fsm_average
                fsm_peak = measured_fsm_peak
            ref_command_integral = float(phase["command_metrics"]["wheel_integral_rad"][channel]) if wheel else 0.0
            ref_actual_integral = float(reference_actual["actual_wheel_integral_rad"][channel]) if wheel else 0.0
            fsm_command_integral = _integral(command_rows, index, ("full12", "command_full12", "commanded_full12")) if wheel else 0.0
            fsm_actual_integral = _integral(active_observations, index, ("actual_full12",), trapezoid=True) if wheel else 0.0
            command_delta = command_end[index] - command_start[index]
            actual_delta = actual_end[index] - actual_start[index]
            reference_actual_end = float(reference_result["actual_end_full12"][index])
            reference_actual_delta = float(reference_result["actual_delta_from_motion_start_full12"][index])
            conformance = channel_conformance(
                reference_command_end=float(phase["end_full12"][index]), fsm_command_end=command_end[index],
                reference_actual_end=reference_actual_end, fsm_actual_end=actual_end[index],
                reference_command_delta=float(phase["delta_full12"][index]), fsm_command_delta=command_delta,
                reference_actual_delta=reference_actual_delta, fsm_actual_delta=actual_delta,
                reference_duration=float(phase["active_duration_s"]), actual_duration=float(clock["active_duration_s"]),
                reference_velocity=ref_average, actual_velocity=fsm_average,
                reference_command_wheel_integral=ref_command_integral,
                fsm_command_wheel_integral=fsm_command_integral,
                reference_actual_wheel_integral=ref_actual_integral,
                fsm_actual_wheel_integral=fsm_actual_integral, wheel_channel=wheel,
            )
            velocity_floor = (
                WHEEL_VELOCITY_FLOOR_RAD_S
                if wheel
                else SERVO_VELOCITY_FLOOR_DEG_S
            )
            peak_error = error_percent(fsm_peak, ref_peak, absolute_floor=velocity_floor)
            peak_ok = within_contract(fsm_peak, ref_peak, absolute_floor=velocity_floor)
            measured_average_error = error_percent(
                measured_fsm_average,
                measured_ref_average,
                absolute_floor=velocity_floor,
            )
            measured_peak_error = error_percent(
                measured_fsm_peak,
                measured_ref_peak,
                absolute_floor=velocity_floor,
            )
            rmse, rmse_percent = _trajectory_error(phase, active_observations, index, start, end)
            within = conformance.within_15_percent and peak_ok and rmse_percent <= 15.0 + 1.0e-9
            result.append({
                "phase": phase_id, "state": str(phase.get("state_name", phase_id)), "channel": channel,
                "reference_start": phase["start_full12"][index], "reference_end": phase["end_full12"][index],
                "reference_delta": phase["delta_full12"][index], "reference_active_duration": phase["active_duration_s"],
                "reference_average_velocity": ref_average, "reference_peak_velocity": ref_peak,
                "reference_wheel_integral": ref_command_integral, "fsm_start": command_start[index],
                "fsm_command_end": command_end[index], "fsm_actual_end": actual_end[index],
                "fsm_delta": actual_delta, "fsm_active_duration": clock["active_duration_s"],
                "fsm_average_velocity": fsm_average, "fsm_peak_velocity": fsm_peak,
                "fsm_wheel_integral": fsm_actual_integral,
                "endpoint_error_percent": conformance.endpoint_error_percent,
                "delta_error_percent": conformance.delta_error_percent,
                "duration_error_percent": conformance.duration_error_percent,
                "velocity_error_percent": max(conformance.velocity_error_percent, peak_error),
                "wheel_integral_error_percent": conformance.wheel_integral_error_percent,
                "within_15_percent": within, "reference_actual_end": reference_actual_end,
                "reference_actual_delta": reference_actual_delta,
                "reference_actual_wheel_integral": ref_actual_integral,
                "fsm_command_delta": command_delta, "fsm_actual_delta": actual_delta,
                "fsm_command_wheel_integral": fsm_command_integral,
                "fsm_actual_wheel_integral": fsm_actual_integral,
                "command_endpoint_error_percent": conformance.commanded_endpoint_error_percent,
                "actual_endpoint_error_percent": conformance.actual_endpoint_error_percent,
                "command_delta_error_percent": conformance.commanded_delta_error_percent,
                "actual_delta_error_percent": conformance.actual_delta_error_percent,
                "average_velocity_error_percent": conformance.velocity_error_percent,
                "peak_velocity_error_percent": peak_error,
                "command_wheel_integral_error_percent": conformance.commanded_wheel_integral_error_percent,
                "actual_wheel_integral_error_percent": conformance.actual_wheel_integral_error_percent,
                "trajectory_rmse": rmse, "trajectory_rmse_percent": rmse_percent,
                "active_sample_count": len(active_observations),
                "reference_measured_average_velocity": measured_ref_average,
                "fsm_measured_average_velocity": measured_fsm_average,
                "measured_average_velocity_error_percent": measured_average_error,
                "reference_measured_peak_velocity": measured_ref_peak,
                "fsm_measured_peak_velocity": measured_fsm_peak,
                "measured_peak_velocity_error_percent": measured_peak_error,
            })
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "maximum_endpoint_error_percent": "endpoint_error_percent",
        "maximum_delta_error_percent": "delta_error_percent",
        "maximum_duration_error_percent": "duration_error_percent",
        "maximum_velocity_error_percent": "velocity_error_percent",
        "maximum_wheel_integral_error_percent": "wheel_integral_error_percent",
        "maximum_trajectory_rmse_percent": "trajectory_rmse_percent",
    }
    result = {name: max((float(row[field]) for row in rows), default=math.inf) for name, field in fields.items()}
    result.update(
        conformance_row_count=len(rows), phase_coverage=sorted({str(row["phase"]) for row in rows}),
        all_normal_states_within_15_percent=bool(rows) and all(bool(row["within_15_percent"]) for row in rows),
    )
    return result
