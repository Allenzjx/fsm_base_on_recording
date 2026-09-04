from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wlr50_clean.infrastructure.command_batch import FULL12_ORDER
from wlr50_clean.ppo import evaluation_artifacts as subject
from wlr50_clean.ppo.evaluation import (
    LiveRunCalibration,
    LiveRunEvaluation,
    ResidualActivityCalibration,
    TerminationSummary,
)
from wlr50_clean.ppo.phase_action_masks_v2 import DEFAULT_PHASE_ACTION_CONFIG_V2
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES
from wlr50_clean.ppo.stability_metrics import LOWER_IS_BETTER, PHASE_IDS


def _simple_calibration() -> ResidualActivityCalibration:
    return ResidualActivityCalibration(
        phase_scale_full12={phase: (1.0,) * 12 for phase in PHASE_IDS},
        numeric_noise_floor_full12=(math.ulp(1.0),) * 12,
        quantization_floor_full12=(1.0e-6,) * 12,
    )


def _fake_evaluation(
    directory: Path,
    *,
    seed: int,
    multiplier: float,
    residual_nonzero: bool,
    body_collision: bool = False,
) -> LiveRunEvaluation:
    directory.mkdir(parents=True, exist_ok=True)
    phase_rows = []
    for phase_index, phase in enumerate(PHASE_IDS, start=1):
        row: dict[str, object] = {
            "phase": phase,
            "sample_count": 120,
            "duration_s": 1.0 + phase_index / 100.0,
            "pitch_rms_rad": multiplier * 0.08,
            "roll_rms_rad": multiplier * 0.06,
            "pitch_rate_p95_abs_rad_s": multiplier * 0.12,
            "pitch_rate_peak_abs_rad_s": multiplier * 0.20,
            "roll_rate_peak_abs_rad_s": multiplier * 0.18,
            "wheel_slip_integral": multiplier * 0.03,
            "active_leg_min_clearance_m": 0.02,
            "home_pose_error_rms_deg": multiplier * 1.0,
            "phase_completion_observed": True,
        }
        row.update({name: multiplier for name in LOWER_IS_BETTER})
        phase_rows.append(row)
    reward_rows = tuple(
        {
            "phase": phase,
            "decision_count": 10,
            **{f"{name}_sum": -multiplier for name in DENSE_FAMILIES},
            **{f"{name}_mean": -multiplier / 10.0 for name in DENSE_FAMILIES},
            "event_reward_sum": 1.0,
            "total_reward_sum": 1.0 - len(DENSE_FAMILIES) * multiplier,
        }
        for phase in PHASE_IDS
    )
    residual_rows = tuple(
        {
            "phase": phase,
            "normalized_residual_rms": 0.05 if residual_nonzero else 0.0,
            "normalized_residual_peak": 0.10 if residual_nonzero else 0.0,
            "active_channel_count": 12 if residual_nonzero else 0,
            "residual_duration_s": 0.5 if residual_nonzero else 0.0,
            "nonzero": residual_nonzero,
        }
        for phase in PHASE_IDS
    )
    termination = TerminationSummary(
        trial_id=f"trial_{seed}_{'candidate' if residual_nonzero else 'baseline'}",
        result="SUCCESS",
        reason="synthetic success",
        final_state_id="P13",
        duration_s=100.0,
        completed_phases=PHASE_IDS,
        completed_p01_p13=True,
        task_success=True,
        body_collision=body_collision,
        wheel_only_climb=False,
        physics_explosion_or_fall=False,
        safety_abort=False,
        runtime_recording_access_count=0,
        recovery_count=0,
        failed_checks=(),
    )
    level = LiveRunCalibration(
        level_reference_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        raw_reference_roll_rad=0.0,
        raw_reference_pitch_rad=0.0,
        raw_reference_yaw_rad=0.0,
        home_joint_positions_deg8=(0.0,) * 8,
        wheel_normal_force_baseline_n4=(10.0,) * 4,
        window_start_s=-0.25,
        window_end_s=0.0,
        sample_count=30,
        maximum_linear_speed_m_s=0.0,
        maximum_angular_speed_rad_s=0.0,
        quality_passed=True,
        source="synthetic_reset_window",
    )
    episode_row = {
        "trial_id": termination.trial_id,
        "seed": seed,
        "task_result": "SUCCESS",
        "task_success": True,
        "completed_p01_p13": True,
        "body_collision": body_collision,
        "wheel_only_climb": False,
        "physics_explosion_or_fall": False,
        "safety_abort": False,
        "duration_s": 100.0,
        "overall_pitch_rate_rms_rad_s": multiplier,
        "overall_roll_rate_rms_rad_s": multiplier,
        "placement_contact_impulse_n_s": multiplier,
        "home_recovery_action_jerk_rms": multiplier,
        "total_reward": 13.0 * (1.0 - len(DENSE_FAMILIES) * multiplier),
        "runtime_recording_access_count": 0,
    }
    return LiveRunEvaluation(
        run_directory=directory.resolve(),
        seed=seed,
        calibration=level,
        stability_samples=(),
        orientation_diagnostics=(),
        phase_rows=tuple(phase_rows),
        episode_row=episode_row,
        termination=termination,
        residual_activity_rows=residual_rows,
        residual_activity_evaluated=True,
        reward_contribution_rows=reward_rows,
        reward_contributions_available=True,
    )


def _paired_runs(tmp_path: Path) -> tuple[list[LiveRunEvaluation], list[LiveRunEvaluation]]:
    baseline = [
        _fake_evaluation(
            tmp_path / f"baseline_{seed}",
            seed=seed,
            multiplier=1.0,
            residual_nonzero=False,
        )
        for seed in range(2001, 2006)
    ]
    candidate = [
        _fake_evaluation(
            tmp_path / f"candidate_{seed}",
            seed=seed,
            multiplier=0.90,
            residual_nonzero=True,
        )
        for seed in reversed(range(2001, 2006))
    ]
    return baseline, candidate


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _canonical_baseline_dirs(tmp_path: Path) -> list[Path]:
    directories = []
    for seed in subject.BASELINE_VALIDATION_SEEDS:
        directory = tmp_path / f"canonical_baseline_{seed}"
        directory.mkdir()
        for name in subject.CANONICAL_EPISODE_FILES:
            (directory / name).write_text(
                json.dumps({"seed": seed, "stream": name}) + "\n",
                encoding="utf-8",
            )
        directories.append(directory)
    return directories


def test_versioned_activity_calibration_uses_config_and_environment_evidence() -> None:
    evidence = subject.build_versioned_residual_activity_calibration()

    assert tuple(evidence.calibration.phase_scale_full12) == PHASE_IDS
    assert evidence.calibration.phase_scale_full12["P08"][9:11] == (0.0, 0.0)
    assert evidence.servo_command_quantization_floor_deg == pytest.approx(
        math.degrees(1.0e-5)
    )
    assert evidence.wheel_command_quantization_floor_rad_s == pytest.approx(
        float(np.spacing(np.float32(evidence.wheel_velocity_limit_rad_s)))
    )
    assert evidence.calibration.quantization_floor_full12[:8] == pytest.approx(
        (math.degrees(1.0e-5),) * 8
    )
    assert evidence.calibration.numeric_noise_floor_full12[0] == math.ulp(3.0)
    assert evidence.phase_action_config_sha256 == subject.sha256_file(
        evidence.phase_action_config
    )
    payload = evidence.as_dict()
    assert payload["schema"] == subject.RESIDUAL_CALIBRATION_SCHEMA
    assert payload["activity_threshold_formula"].startswith("max(")
    assert payload["full12_order"] == list(FULL12_ORDER)


def test_versioned_activity_calibration_rejects_environment_abi_drift(
    tmp_path: Path,
) -> None:
    source = json.loads(subject.DEFAULT_ENVIRONMENT_LOCK.read_text(encoding="utf-8"))
    source["canonical_action_order_full12"] = list(reversed(FULL12_ORDER))
    changed = tmp_path / "environment_lock.json"
    changed.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(subject.EvaluationArtifactError, match="Full12 order"):
        subject.build_versioned_residual_activity_calibration(
            phase_action_config=DEFAULT_PHASE_ACTION_CONFIG_V2,
            environment_lock=changed,
        )


def test_canonical_sequence_is_read_only_and_routes_reward_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = []
    originals: dict[Path, dict[str, bytes]] = {}
    for seed in (7, 9):
        directory = tmp_path / f"episode_{seed}"
        directory.mkdir()
        (directory / "sentinel.bin").write_bytes(f"immutable-{seed}".encode())
        (directory / subject.DEFAULT_REWARD_STREAM_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
        directories.append(directory)
        originals[directory] = {
            path.name: path.read_bytes() for path in directory.iterdir()
        }
    calls: list[tuple[Path, int, Path]] = []

    def fake_evaluate(directory, *, seed, residual_calibration, reward_stream_path, **options):
        calls.append((Path(directory), seed, Path(reward_stream_path)))
        assert residual_calibration is calibration
        assert options == {"wheel_stop_hold_s": 0.25}
        return _fake_evaluation(
            Path(directory),
            seed=seed,
            multiplier=0.9,
            residual_nonzero=True,
        )

    calibration = _simple_calibration()
    monkeypatch.setattr(subject, "evaluate_live_run", fake_evaluate)
    evaluated = subject.evaluate_canonical_episode_dirs(
        directories,
        seeds=(7, 9),
        residual_calibration=calibration,
        evaluation_options={"wheel_stop_hold_s": 0.25},
    )

    assert [run.seed for run in evaluated] == [7, 9]
    assert [call[2].name for call in calls] == ["reward_15hz.jsonl"] * 2
    for directory in directories:
        assert {
            path.name: path.read_bytes() for path in directory.iterdir()
        } == originals[directory]


def test_canonical_sequence_rejects_duplicate_seed_and_incomplete_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        (directory / subject.DEFAULT_REWARD_STREAM_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )
    with pytest.raises(subject.EvaluationArtifactError, match="seeds must be unique"):
        subject.evaluate_canonical_episode_dirs(
            (first, second),
            seeds=(3, 3),
            residual_calibration=_simple_calibration(),
        )

    complete = _fake_evaluation(
        first, seed=3, multiplier=1.0, residual_nonzero=True
    )
    incomplete = replace(complete, phase_rows=complete.phase_rows[:-1])
    monkeypatch.setattr(subject, "evaluate_live_run", lambda *args, **kwargs: incomplete)
    with pytest.raises(subject.EvaluationArtifactError, match="ordered P01-P13 exactly"):
        subject.evaluate_canonical_episode_dirs(
            (first,), seeds=(3,), residual_calibration=_simple_calibration()
        )


def test_canonical_sequence_detects_input_mutation_during_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "episode"
    directory.mkdir()
    reward = directory / subject.DEFAULT_REWARD_STREAM_FILENAME
    reward.write_text("{}\n", encoding="utf-8")

    def mutating_evaluate(*args, **kwargs):
        (directory / "forbidden-output.txt").write_text("mutation", encoding="utf-8")
        return _fake_evaluation(
            directory, seed=4, multiplier=1.0, residual_nonzero=True
        )

    monkeypatch.setattr(subject, "evaluate_live_run", mutating_evaluate)
    with pytest.raises(subject.EvaluationArtifactError, match="changed during read-only"):
        subject.evaluate_canonical_episode_dirs(
            (directory,), seeds=(4,), residual_calibration=_simple_calibration()
        )


def test_paired_export_is_complete_sorted_and_byte_idempotent(tmp_path: Path) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    checkpoint = tmp_path / "checkpoint_best_validation.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    output = tmp_path / "metrics"

    paths = subject.export_paired_evaluation_artifacts(
        output,
        baseline_runs=baseline,
        candidate_runs=candidate,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name="checkpoint_best_validation",
        candidate_checkpoint_path=checkpoint,
        residual_calibration_evidence=subject.build_versioned_residual_activity_calibration(),
    )

    assert len(_csv_rows(paths.baseline_episode_metrics)) == 5
    assert len(_csv_rows(paths.baseline_phase_metrics)) == 5 * 13
    assert len(_csv_rows(paths.candidate_episode_metrics)) == 5
    assert len(_csv_rows(paths.candidate_phase_metrics)) == 5 * 13
    assert len(_csv_rows(paths.residual_activity_by_phase)) == 5 * 13
    assert len(_csv_rows(paths.reward_contribution_by_phase)) == 5 * 13
    assert len(_csv_rows(paths.termination_summary)) == 10
    phase_rows = _csv_rows(paths.phase_metric_comparison)
    assert [row["phase"] for row in phase_rows] == list(PHASE_IDS)
    assert float(phase_rows[0]["primary_phase_score_improvement_fraction"]) == pytest.approx(
        0.10
    )
    checkpoints = _csv_rows(paths.checkpoint_comparison)
    assert [row["role"] for row in checkpoints] == ["baseline", "candidate"]
    assert json.loads(checkpoints[1]["paired_seeds"]) == list(range(2001, 2006))
    assert checkpoints[1]["checkpoint_sha256"] == subject.sha256_file(checkpoint)

    decision = json.loads(paths.promotion_decision.read_text(encoding="utf-8"))
    assert decision["promotion"]["promoted"] is True
    assert decision["first_failed_gate"] is None
    assert decision["paired_seeds"] == list(range(2001, 2006))
    assert decision["residual_activity_calibration"]["schema"] == (
        subject.RESIDUAL_CALIBRATION_SCHEMA
    )

    published_paths = [Path(value) for value in paths.as_dict().values()]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in published_paths
        if path.is_file()
    }
    repeated = subject.export_paired_evaluation_artifacts(
        output,
        baseline_runs=baseline,
        candidate_runs=candidate,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name="checkpoint_best_validation",
        candidate_checkpoint_path=checkpoint,
        residual_calibration_evidence=subject.build_versioned_residual_activity_calibration(),
    )
    assert repeated == paths
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    } == before


def test_baseline_only_export_evaluates_five_canonical_dirs_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = _canonical_baseline_dirs(tmp_path)
    calls = []

    def fake_evaluate(
        directory, *, seed, residual_calibration, reward_stream_path, **options
    ):
        calls.append(
            (
                Path(directory),
                seed,
                Path(reward_stream_path),
                residual_calibration,
                options,
            )
        )
        return _fake_evaluation(
            Path(directory),
            seed=seed,
            multiplier=1.0,
            residual_nonzero=False,
        )

    monkeypatch.setattr(subject, "evaluate_live_run", fake_evaluate)
    output = tmp_path / "outputs" / "ppo_phase_v1" / "metrics"
    calibration = subject.build_versioned_residual_activity_calibration()
    paths = subject.export_baseline_evaluation_artifacts(
        output,
        episode_directories=directories,
        seeds=subject.BASELINE_VALIDATION_SEEDS,
        residual_calibration_evidence=calibration,
    )

    assert paths.episode_metrics == output / "fsm_baseline_episode_metrics.csv"
    assert paths.phase_metrics == output / "fsm_baseline_phase_metrics.csv"
    assert paths.manifest == output / "fsm_baseline_evaluation_manifest.json"
    assert sorted(path.name for path in output.iterdir()) == [
        "fsm_baseline_episode_metrics.csv",
        "fsm_baseline_evaluation_manifest.json",
        "fsm_baseline_phase_metrics.csv",
    ]
    assert len(_csv_rows(paths.episode_metrics)) == 5
    assert len(_csv_rows(paths.phase_metrics)) == 65
    assert [call[1] for call in calls] == list(subject.BASELINE_VALIDATION_SEEDS)
    assert all(call[2].name == "reward_15hz.jsonl" for call in calls)
    assert all(call[3] is calibration.calibration for call in calls)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["candidate_required"] is False
    assert manifest["validation_seeds"] == list(subject.BASELINE_VALIDATION_SEEDS)
    assert manifest["all_p01_p13_complete"] is True
    assert manifest["all_authoritative_success"] is True
    assert manifest["all_zero_residual"] is True
    assert len(manifest["source_episodes"]) == 5
    assert manifest["artifacts"]["episode_metrics"]["sha256"] == subject.sha256_file(
        paths.episode_metrics
    )
    assert manifest["artifacts"]["phase_metrics"]["sha256"] == subject.sha256_file(
        paths.phase_metrics
    )

    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (paths.episode_metrics, paths.phase_metrics, paths.manifest)
    }
    repeated = subject.export_baseline_evaluation_artifacts(
        output,
        episode_directories=directories,
        seeds=subject.BASELINE_VALIDATION_SEEDS,
        residual_calibration_evidence=calibration,
    )
    assert repeated == paths
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    } == before


def test_baseline_only_export_fails_closed_on_incomplete_phase_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = _canonical_baseline_dirs(tmp_path)

    def fake_evaluate(directory, *, seed, **kwargs):
        complete = _fake_evaluation(
            Path(directory),
            seed=seed,
            multiplier=1.0,
            residual_nonzero=False,
        )
        if seed == 2003:
            return replace(complete, phase_rows=complete.phase_rows[:-1])
        return complete

    monkeypatch.setattr(subject, "evaluate_live_run", fake_evaluate)
    output = tmp_path / "metrics"
    with pytest.raises(subject.EvaluationArtifactError, match="ordered P01-P13 exactly"):
        subject.export_baseline_evaluation_artifacts(
            output,
            episode_directories=directories,
            seeds=subject.BASELINE_VALIDATION_SEEDS,
            residual_calibration_evidence=subject.build_versioned_residual_activity_calibration(),
        )
    assert not output.exists()


def test_baseline_only_export_preflights_conflict_without_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = _canonical_baseline_dirs(tmp_path)
    monkeypatch.setattr(
        subject,
        "evaluate_live_run",
        lambda directory, *, seed, **kwargs: _fake_evaluation(
            Path(directory), seed=seed, multiplier=1.0, residual_nonzero=False
        ),
    )
    output = tmp_path / "metrics"
    output.mkdir()
    conflict = output / subject.BASELINE_PHASE_FILENAME
    conflict.write_bytes(b"unrelated prior baseline\n")

    with pytest.raises(subject.EvaluationArtifactError, match="non-identical"):
        subject.export_baseline_evaluation_artifacts(
            output,
            episode_directories=directories,
            seeds=subject.BASELINE_VALIDATION_SEEDS,
            residual_calibration_evidence=subject.build_versioned_residual_activity_calibration(),
        )
    assert list(output.iterdir()) == [conflict]


def test_promotion_json_preserves_the_exact_first_failed_gate(tmp_path: Path) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    unsafe = list(candidate)
    unsafe[0] = replace(
        unsafe[0],
        termination=replace(unsafe[0].termination, body_collision=True),
    )
    paths = subject.export_paired_evaluation_artifacts(
        tmp_path / "failed-metrics",
        baseline_runs=baseline,
        candidate_runs=unsafe,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name="unsafe_candidate",
    )
    payload = json.loads(paths.promotion_decision.read_text(encoding="utf-8"))

    assert payload["promotion"]["promoted"] is False
    assert payload["first_failed_gate"] == "body_collision_zero"
    assert payload["promotion"]["first_failed_gate"] == "body_collision_zero"
    first_failed = next(
        row["gate"]
        for row in payload["checks_in_evaluation_order"]
        if not row["passed"]
    )
    assert first_failed == "body_collision_zero"


def test_export_rejects_unmatched_or_duplicate_seeds_before_writing(tmp_path: Path) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    mismatched = list(candidate)
    mismatched[0] = replace(mismatched[0], seed=9999)
    output = tmp_path / "not-created"
    with pytest.raises(subject.EvaluationArtifactError, match="not matched"):
        subject.export_paired_evaluation_artifacts(
            output,
            baseline_runs=baseline,
            candidate_runs=mismatched,
            frozen_hashes_unchanged=True,
            candidate_checkpoint_name="candidate",
        )
    assert not output.exists()

    duplicate = list(candidate)
    duplicate[0] = replace(duplicate[0], seed=duplicate[1].seed)
    with pytest.raises(subject.EvaluationArtifactError, match="seeds must be unique"):
        subject.export_paired_evaluation_artifacts(
            output,
            baseline_runs=baseline,
            candidate_runs=duplicate,
            frozen_hashes_unchanged=True,
            candidate_checkpoint_name="candidate",
        )
    assert not output.exists()


def test_export_preflights_nonidentical_existing_file_before_any_creation(
    tmp_path: Path,
) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    output = tmp_path / "conflict"
    output.mkdir()
    conflict = output / subject.BASELINE_PHASE_FILENAME
    conflict.write_bytes(b"unrelated prior evidence\n")

    with pytest.raises(subject.EvaluationArtifactError, match="non-identical"):
        subject.export_paired_evaluation_artifacts(
            output,
            baseline_runs=baseline,
            candidate_runs=candidate,
            frozen_hashes_unchanged=True,
            candidate_checkpoint_name="candidate",
        )

    assert list(output.iterdir()) == [conflict]
    assert conflict.read_bytes() == b"unrelated prior evidence\n"


def test_export_requires_all_candidate_reward_and_residual_phase_rows(
    tmp_path: Path,
) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    missing_reward = list(candidate)
    missing_reward[0] = replace(
        missing_reward[0], reward_contribution_rows=missing_reward[0].reward_contribution_rows[:-1]
    )
    with pytest.raises(subject.EvaluationArtifactError, match="reward contributions"):
        subject.export_paired_evaluation_artifacts(
            tmp_path / "missing-reward",
            baseline_runs=baseline,
            candidate_runs=missing_reward,
            frozen_hashes_unchanged=True,
            candidate_checkpoint_name="candidate",
        )

    missing_residual = list(candidate)
    missing_residual[0] = replace(
        missing_residual[0], residual_activity_rows=missing_residual[0].residual_activity_rows[:-1]
    )
    with pytest.raises(subject.EvaluationArtifactError, match="residual activity"):
        subject.export_paired_evaluation_artifacts(
            tmp_path / "missing-residual",
            baseline_runs=baseline,
            candidate_runs=missing_residual,
            frozen_hashes_unchanged=True,
            candidate_checkpoint_name="candidate",
        )
