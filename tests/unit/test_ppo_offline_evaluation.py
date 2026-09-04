from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.evaluation import (
    LiveRunEvaluation,
    OfflineEvaluationError,
    ResidualActivityCalibration,
    TerminationSummary,
    evaluate_live_run,
    paired_baseline_candidate_promotion,
    summarize_reward_contributions,
)
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES, STATE_IDS
from wlr50_clean.ppo.stability_metrics import (
    LOWER_IS_BETTER,
    compare_phase_metrics,
)


def _quaternion_from_pitch(pitch: float) -> list[float]:
    return [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _make_run(
    root: Path,
    *,
    result: str = "INCOMPLETE_CONTROLLER_BLOCKED",
    residual_value: float = 0.02,
) -> Path:
    root.mkdir()
    physics_hz = 10.0
    observations: list[dict[str, object]] = []
    commands: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    phase_times: dict[str, object] = {}
    for phase_index, phase in enumerate(STATE_IDS):
        entry = phase_index * 0.4
        phase_times[phase] = {
            "entry_time_s": entry,
            "motion_start_s": entry,
            "motion_end_s": entry + 0.3,
            "verify_start_s": entry + 0.3,
            "completion_time_s": entry + 0.4,
        }
        transitions.append(
            {
                "sim_time_s": entry,
                "state_id": phase,
                "from_lifecycle": "WAIT_ENTRY",
                "to_lifecycle": "EXECUTE_MOTION",
            }
        )
        for local_tick in range(4):
            tick = phase_index * 4 + local_tick
            time_s = tick / physics_hz
            raw_pitch = 0.10 if tick < 2 else 0.10 + 0.01 * math.sin(tick)
            contacts: dict[str, object] = {}
            wheels: dict[str, object] = {}
            bodies: dict[str, object] = {}
            for wheel_index, wheel_name in enumerate(WHEEL_ORDER):
                body_name = wheel_name.replace("_ankle", "_wheel")
                force = 10.0
                if phase in {"P03", "P12"} and local_tick == 2:
                    force = 20.0
                wheels[wheel_name] = {
                    "body_name": body_name,
                    "velocity_rad_s": 0.0,
                    "bottom_w_m": [0.6, 0.0, 0.06],
                    "geometry_verified": True,
                }
                contacts[body_name] = {
                    "ground": {
                        "normal_force_n": force,
                        "pair_verified": True,
                    },
                    "obstacle": {
                        "normal_force_n": 0.0,
                        "pair_verified": True,
                    },
                }
                bodies[body_name] = {"linear_velocity_w_m_s": [0.0, 0.0, 0.0]}
            observations.append(
                {
                    "schema": "wlr50_clean.live_observation.v1",
                    "physics_tick": tick,
                    "simulation_time_s": time_s,
                    "all_finite": True,
                    "base": {
                        "orientation_wxyz": _quaternion_from_pitch(raw_pitch),
                        "linear_velocity_w_m_s": [0.0, 0.0, 0.0],
                        "angular_velocity_w_rad_s": [0.01, 0.02, 0.0],
                    },
                    "joints": {
                        name: {"position_deg": 0.0} for name in SERVO_ORDER
                    },
                    "wheels": wheels,
                    "contacts": contacts,
                    "bodies": bodies,
                    "obstacle": {"top_z_m": 0.05},
                    "body_collision": {"detected": False},
                    "guards": {
                        "wheel_only_climb_detected": {"passed": False},
                        "physics_explosion_or_fall": {"passed": False},
                    },
                    "measured_wheel_velocity_rad_s": [0.0, 0.0, 0.0, 0.0],
                }
            )
            p13_wheels = (
                [0.1, 0.1, 0.1, 0.1]
                if phase == "P13" and local_tick < 2
                else [0.0, 0.0, 0.0, 0.0]
            )
            applied = [0.0] * 8 + p13_wheels
            commands.append(
                {
                    "control_physics_tick": tick,
                    "sim_time_s": time_s,
                    "state_id": phase,
                    "lifecycle": "EXECUTE_MOTION",
                    "nominal_full12": applied,
                    "residual_full12": [residual_value] * 12,
                    "applied_full12": applied,
                }
            )
    duration = commands[-1]["sim_time_s"]
    manifest = {
        "schema": "wlr50_clean.trial_manifest.v1",
        "trial_id": root.name,
        "result": result,
        "reason": "synthetic test termination",
        "physics_hz": physics_hz,
        "decision_hz": 2.5,
        "phase_times": phase_times,
        "analysis_checks": {"task_result_success": result == "SUCCESS"},
        "conformance": {
            "recovery_count": 0,
            "measured_wheel_velocity_decay_threshold_rad_s": 0.1,
        },
        "success_evidence": {
            "completed_macro_phases": list(STATE_IDS),
            "body_collision": False,
            "wheel_only_climb": False,
            "duration_s": duration,
            "runtime_raw_recording_access": False,
        },
    }
    task_events = [
        {
            "event": "TRIAL_TERMINATION",
            "result": result,
            "state_id": "P13",
            "sim_time_s": duration,
            "reason": "synthetic test termination",
            "details": {"failed_checks": [] if result == "SUCCESS" else ["task_result"]},
        }
    ]
    (root / "trial_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_jsonl(root / "observation_120hz.jsonl", observations)
    _write_jsonl(root / "full12_commands_120hz.jsonl", commands)
    _write_jsonl(root / "state_transitions.jsonl", transitions)
    _write_jsonl(root / "task_events.jsonl", task_events)
    return root


def _residual_calibration() -> ResidualActivityCalibration:
    return ResidualActivityCalibration(
        phase_scale_full12={phase: (1.0,) * 12 for phase in STATE_IDS},
        numeric_noise_floor_full12=(0.001,) * 12,
        quantization_floor_full12=(0.001,) * 12,
    )


def test_live_stream_ingestion_is_read_only_calibrated_and_truthful(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "trial_043_shape")
    before = {
        path.name: path.read_bytes()
        for path in run.iterdir()
        if path.is_file()
    }
    evaluated = evaluate_live_run(
        run,
        seed=7,
        residual_calibration=_residual_calibration(),
        calibration_window_s=0.1,
        wheel_stop_hold_s=0.2,
    )
    after = {
        path.name: path.read_bytes()
        for path in run.iterdir()
        if path.is_file()
    }
    assert after == before
    assert evaluated.calibration.raw_reference_pitch_rad == pytest.approx(0.10)
    assert evaluated.orientation_diagnostics[0].raw_pitch_rad == pytest.approx(0.10)
    assert evaluated.orientation_diagnostics[0].calibrated_pitch_error_rad == pytest.approx(0.0)
    assert evaluated.calibration.quality_passed
    assert len(evaluated.phase_rows) == 13
    assert all("settling_time_s" in row for row in evaluated.phase_rows)
    assert all("phase_entry_action_jump_rms" in row for row in evaluated.phase_rows)
    assert {row["phase"] for row in evaluated.residual_activity_rows} == set(STATE_IDS)
    assert evaluated.episode_row["wheel_stop_completed"] is True
    assert evaluated.episode_row["reward_contributions_available"] is False
    assert evaluated.episode_row["total_reward"] is None
    # P01-P13 completion is not silently relabeled as physical task success.
    assert evaluated.termination.completed_p01_p13 is True
    assert evaluated.termination.result == "INCOMPLETE_CONTROLLER_BLOCKED"
    assert evaluated.termination.task_success is False


def test_live_stream_prefers_explicit_pre_action_backend_calibration(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "explicit_calibration")
    manifest_path = run / "trial_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_pitch = 0.25
    manifest["ppo_calibration"] = {
        "source": "backend_reset_test_window",
        "quality_passed": True,
        "level_reference_orientation_wxyz": _quaternion_from_pitch(reference_pitch),
        "home_joint_positions_deg8": [1.0] * 8,
        "wheel_normal_force_baseline_n4": [9.0] * 4,
        "sample_count": 30,
        "window_start_s": -0.25,
        "window_end_s": 0.0,
        "maximum_linear_speed_m_s": 0.01,
        "maximum_angular_speed_rad_s": 0.02,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evaluated = evaluate_live_run(
        run,
        seed=7,
        residual_calibration=_residual_calibration(),
        calibration_window_s=0.1,
        wheel_stop_hold_s=0.2,
    )

    assert evaluated.calibration.source == "backend_reset_test_window"
    assert evaluated.calibration.raw_reference_pitch_rad == pytest.approx(reference_pitch)
    assert evaluated.calibration.home_joint_positions_deg8 == pytest.approx((1.0,) * 8)
    assert evaluated.calibration.wheel_normal_force_baseline_n4 == pytest.approx((9.0,) * 4)
    assert evaluated.calibration.quality_passed is True


def test_reward_contributions_require_v2_families_and_preserve_events() -> None:
    record = {
        "state_id": "P03",
        "weighted_dense": {name: 0.1 for name in DENSE_FAMILIES},
        "event_components": {
            "phase_completion": 1.0,
            "final_success": 0.0,
            "task_failure": 0.0,
            "safety_abort": 0.0,
        },
    }
    row = summarize_reward_contributions([record])[0]
    assert row["decision_count"] == 1
    assert row["event_reward_sum"] == 1.0
    assert row["total_reward_sum"] == pytest.approx(1.5)
    with pytest.raises(OfflineEvaluationError, match="legacy rewards are not relabeled"):
        summarize_reward_contributions(
            [{"state_id": "P03", "reward_components_t": {"forward_progress": 1.0}}]
        )


def test_stability_comparison_has_the_settling_metric_it_scores(tmp_path: Path) -> None:
    evaluated = evaluate_live_run(
        _make_run(tmp_path / "success", result="SUCCESS"),
        seed=3,
        residual_calibration=_residual_calibration(),
        calibration_window_s=0.1,
        wheel_stop_hold_s=0.2,
    )
    rows = compare_phase_metrics(evaluated.phase_rows, evaluated.phase_rows)
    assert len(rows) == 13
    assert "settling_time_s_improvement_fraction" in rows[0]
    assert all(metric in evaluated.phase_rows[0] for metric in LOWER_IS_BETTER)


def _promotion_fixture(
    base: LiveRunEvaluation, *, seed: int, candidate: bool
) -> LiveRunEvaluation:
    multiplier = 0.90 if candidate else 1.0
    phase_rows = []
    for phase in STATE_IDS:
        row: dict[str, object] = {"phase": phase, "sample_count": 10}
        row.update({name: multiplier for name in LOWER_IS_BETTER})
        phase_rows.append(row)
    episode = dict(base.episode_row)
    episode.update(
        {
            "seed": seed,
            "overall_pitch_rate_rms_rad_s": multiplier,
            "placement_contact_impulse_n_s": multiplier,
            "home_recovery_action_jerk_rms": multiplier,
        }
    )
    termination = TerminationSummary(
        trial_id=f"{'candidate' if candidate else 'baseline'}_{seed}",
        result="SUCCESS",
        reason="test success",
        final_state_id="P13",
        duration_s=100.0,
        completed_phases=STATE_IDS,
        completed_p01_p13=True,
        task_success=True,
        body_collision=False,
        wheel_only_climb=False,
        physics_explosion_or_fall=False,
        safety_abort=False,
        runtime_recording_access_count=0,
        recovery_count=0,
        failed_checks=(),
    )
    residual_rows = tuple(
        {"phase": phase, "nonzero": candidate} for phase in STATE_IDS
    )
    return replace(
        base,
        seed=seed,
        phase_rows=tuple(phase_rows),
        episode_row=episode,
        termination=termination,
        residual_activity_rows=residual_rows,
        residual_activity_evaluated=True,
    )


def test_paired_promotion_requires_matched_evidence_and_all_gates(tmp_path: Path) -> None:
    source = evaluate_live_run(
        _make_run(tmp_path / "source", result="SUCCESS"),
        seed=0,
        residual_calibration=_residual_calibration(),
        calibration_window_s=0.1,
        wheel_stop_hold_s=0.2,
    )
    baseline = [_promotion_fixture(source, seed=seed, candidate=False) for seed in range(5)]
    candidate = [_promotion_fixture(source, seed=seed, candidate=True) for seed in range(5)]
    comparison = paired_baseline_candidate_promotion(
        baseline, candidate, frozen_hashes_unchanged=True
    )
    assert comparison.promotion.promoted is True
    assert comparison.promotion.checks["fall_or_physics_explosion_zero"] is True
    assert comparison.promotion.checks["priority_phases_have_real_residual"] is True
    assert comparison.overall_pitch_rate_improvement_fraction == pytest.approx(0.10)

    unsafe = list(candidate)
    unsafe[0] = replace(
        unsafe[0],
        termination=replace(
            unsafe[0].termination, physics_explosion_or_fall=True, task_success=False
        ),
    )
    rejected = paired_baseline_candidate_promotion(
        baseline, unsafe, frozen_hashes_unchanged=True
    )
    assert rejected.promotion.promoted is False
    assert rejected.promotion.checks["fall_or_physics_explosion_zero"] is False
