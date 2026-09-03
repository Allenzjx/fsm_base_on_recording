from __future__ import annotations

from typing import Any

import pytest

from wlr50_clean.evaluation.comparison import PHASE_IDS
from wlr50_clean.evaluation.conformance import _similarity_rows, _summary
from wlr50_clean.reference.similarity import channel_conformance


ORDER = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)


def _vector(index: int, value: float) -> list[float]:
    result = [0.0] * 12
    result[index] = value
    return result


def _synthetic_evidence(*, wheel: bool) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]
]:
    channel_index = 8 if wheel else 0
    channel = ORDER[channel_index]
    phases: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    windows: dict[str, dict[str, float]] = {}
    for phase_index, phase_id in enumerate(PHASE_IDS):
        motion_start = 2.0 * phase_index
        motion_end = motion_start + 1.0
        completion = motion_end + 0.1
        windows[phase_id] = {
            "entry_time_s": motion_start,
            "motion_start_s": motion_start,
            "motion_end_s": motion_end,
            "completion_time_s": completion,
            "active_duration_s": 1.0,
        }
        if wheel:
            start_value = end_value = 0.3
            actual_values = (100.0, -100.0, 100.0, -100.0)
            command_values = (0.3, 0.3, 0.3)
            reference_phase_average = 1.0
            reference_peak = 1.0
            reference_actual_end = 0.0
            reference_actual_delta = 0.0
            reference_actual_integral = 10.0
            command_metrics = {
                "wheel_time_weighted_average_abs_target_rad_s": {channel: 0.3},
                "wheel_peak_abs_target_rad_s": {channel: 0.3},
                "wheel_integral_rad": {channel: 0.3},
            }
        else:
            start_value = 10.0 * phase_index
            end_value = start_value + 10.0
            actual_values = (start_value, start_value + 5.0, end_value, end_value)
            command_values = (start_value, start_value + 5.0, end_value)
            reference_phase_average = 10.0
            reference_peak = 37.0
            reference_actual_end = end_value
            reference_actual_delta = 10.0
            reference_actual_integral = 0.0
            command_metrics = {
                "servo_average_target_path_velocity_deg_s": {channel: 10.0},
                "servo_velocity_limit_deg_s": {channel: 10.0},
                "wheel_integral_rad": {},
            }
        start_full12 = _vector(channel_index, start_value)
        end_full12 = _vector(channel_index, end_value)
        phases.append(
            {
                "state_id": phase_id,
                "state_name": phase_id,
                "active_duration_s": 1.0,
                "start_full12": start_full12,
                "end_full12": end_full12,
                "delta_full12": _vector(channel_index, end_value - start_value),
                "active_channels": [channel],
                "command_metrics": command_metrics,
                "reference_actual": {
                    "phase_average_abs_velocity": {
                        channel: reference_phase_average
                    },
                    # Deliberately incompatible: the evaluator must use the
                    # whole-phase metric above, not this response-only metric.
                    "active_window_average_abs_velocity": {channel: 99.0},
                    "peak_abs_velocity": {channel: reference_peak},
                    "actual_wheel_integral_rad": (
                        {channel: reference_actual_integral} if wheel else {}
                    ),
                    "trajectory_samples_normalized": [
                        {"progress": 0.0, "actual_full12": start_full12},
                        {
                            "progress": 0.5,
                            "actual_full12": _vector(
                                channel_index, 0.5 * (start_value + end_value)
                            ),
                        },
                        {"progress": 1.0, "actual_full12": end_full12},
                    ],
                },
                "reference_result_observation": {
                    "actual_end_full12": _vector(
                        channel_index, reference_actual_end
                    ),
                    "actual_delta_from_motion_start_full12": _vector(
                        channel_index, reference_actual_delta
                    ),
                },
            }
        )
        for offset, value in zip((0.0, 0.5, 1.0), command_values):
            commands.append(
                {
                    "state_id": phase_id,
                    "simulation_time_s": motion_start + offset,
                    "full12": _vector(channel_index, value),
                }
            )
        measured_velocities = (100.0, -100.0, 100.0, -100.0) if wheel else (1.0, 1.0, 1.0, 37.0)
        for offset, actual, velocity in zip(
            (0.0, 0.5, 1.0, 1.0 + 1.0 / 120.0),
            actual_values,
            measured_velocities,
        ):
            observations.append(
                {
                    "state_id": phase_id,
                    "simulation_time_s": motion_start + offset,
                    "actual_full12": _vector(channel_index, actual),
                    "velocity_full12": _vector(channel_index, velocity),
                }
            )
        observations.append(
            {
                "state_id": phase_id,
                "simulation_time_s": completion,
                "actual_full12": _vector(
                    channel_index, 200.0 if wheel else end_value
                ),
                "velocity_full12": _vector(channel_index, 0.0),
            }
        )
    contract = {
        "full12_order": list(ORDER),
        "wheel_radius_m": 0.04998999834060672,
        "phases": phases,
    }
    return contract, observations, commands, windows


def test_servo_velocity_uses_whole_phase_and_post_end_physics_sample() -> None:
    contract, observations, commands, windows = _synthetic_evidence(wheel=False)
    rows = _similarity_rows(contract, observations, commands, windows)

    assert len(rows) == 13
    assert all(row["active_sample_count"] == 4 for row in rows)
    assert all(row["fsm_average_velocity"] == pytest.approx(10.0) for row in rows)
    assert all(row["reference_average_velocity"] == pytest.approx(10.0) for row in rows)
    summary = _summary(rows)
    assert all(row["command_average_velocity_error_percent"] == 0.0 for row in rows)
    assert all(row["command_peak_velocity_error_percent"] == 0.0 for row in rows)
    assert summary["all_normal_states_within_15_percent"] is True
    assert summary["all_normal_states_within_30_percent"] is True
    assert summary["all_normal_states_within_active_tolerance"] is True


def test_wheels_gate_target_metrics_and_actual_integral() -> None:
    contract, observations, commands, windows = _synthetic_evidence(wheel=True)
    rows = _similarity_rows(contract, observations, commands, windows)

    assert _summary(rows)["all_normal_states_within_15_percent"] is False
    assert all(row["fsm_average_velocity"] == pytest.approx(0.3) for row in rows)
    assert all(row["fsm_peak_velocity"] == pytest.approx(0.3) for row in rows)
    assert all(row["fsm_measured_peak_velocity"] == pytest.approx(100.0) for row in rows)
    assert all(row["endpoint_error_percent"] == 0.0 for row in rows)
    assert all(row["delta_error_percent"] == 0.0 for row in rows)
    assert all(row["actual_endpoint_error_percent"] > 15.0 for row in rows)
    assert all(row["actual_wheel_integral_error_percent"] > 15.0 for row in rows)
    assert all(row["wheel_integral_error_percent"] > 15.0 for row in rows)
    assert all(row["reference_wheel_integral"] == pytest.approx(0.3) for row in rows)
    assert all(row["fsm_wheel_integral"] == pytest.approx(0.3) for row in rows)
    assert all(
        row["wheel_integral_gate_basis"] == "max(command_target,measured_actual)"
        for row in rows
    )


def test_wheel_command_integral_and_one_degree_servo_velocity_floor_gate() -> None:
    wheel = channel_conformance(
        reference_command_end=0.0,
        fsm_command_end=99.0,
        reference_actual_end=0.0,
        fsm_actual_end=99.0,
        reference_command_delta=0.0,
        fsm_command_delta=99.0,
        reference_actual_delta=0.0,
        fsm_actual_delta=99.0,
        reference_duration=1.0,
        actual_duration=1.0,
        reference_velocity=0.3,
        actual_velocity=0.3,
        reference_command_wheel_integral=0.3,
        fsm_command_wheel_integral=0.6,
        reference_actual_wheel_integral=0.0,
        fsm_actual_wheel_integral=99.0,
        wheel_channel=True,
    )
    assert wheel.within_15_percent is False
    assert wheel.endpoint_error_percent == 0.0
    assert wheel.delta_error_percent == 0.0
    assert wheel.wheel_integral_error_percent > 15.0

    servo = channel_conformance(
        reference_command_end=10.0,
        fsm_command_end=10.0,
        reference_actual_end=10.0,
        fsm_actual_end=10.0,
        reference_command_delta=10.0,
        fsm_command_delta=10.0,
        reference_actual_delta=10.0,
        fsm_actual_delta=10.0,
        reference_duration=1.0,
        actual_duration=1.0,
        reference_velocity=0.0,
        actual_velocity=1.1,
    )
    assert servo.within_15_percent is False


def test_active_thirty_percent_and_legacy_fifteen_percent_flags_are_independent() -> None:
    result = channel_conformance(
        reference_command_end=10.0,
        fsm_command_end=10.0,
        reference_actual_end=10.0,
        fsm_actual_end=10.0,
        reference_command_delta=10.0,
        fsm_command_delta=10.0,
        reference_actual_delta=10.0,
        fsm_actual_delta=10.0,
        reference_duration=1.0,
        actual_duration=1.0,
        reference_velocity=10.0,
        actual_velocity=12.0,
    )

    assert result.velocity_error_percent == pytest.approx(20.0)
    assert result.within_15_percent is False
    assert result.within_30_percent is True
    assert result.within_active_tolerance is True
