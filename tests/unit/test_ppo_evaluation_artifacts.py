from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wlr50_clean.infrastructure.command_batch import FULL12_ORDER
from wlr50_clean.ppo import checkpoint_promotion
from wlr50_clean.ppo import evaluation_artifacts as subject
from wlr50_clean.ppo import final_reporting
from wlr50_clean.ppo.evaluation import (
    LiveRunCalibration,
    LiveRunEvaluation,
    ResidualActivityCalibration,
    TerminationSummary,
)
from wlr50_clean.ppo.phase_action_masks_v2 import DEFAULT_PHASE_ACTION_CONFIG_V2
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES
from wlr50_clean.ppo.phase_snapshots import (
    phase_snapshot_bundle_file_hashes,
    validated_phase_snapshot_bundle_record,
)
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
            "residual_high_frequency_fraction": multiplier * 0.01,
            "applied_high_frequency_fraction": multiplier * 0.02,
            "residual_spectrum_normalization": "phase_scale_full12",
            "residual_spectral_energy_fraction_0p0_0p5_hz": 0.10,
            "residual_spectral_energy_fraction_0p5_1p0_hz": 0.20,
            "residual_spectral_energy_fraction_1p0_2p0_hz": 0.30,
            "residual_spectral_energy_fraction_2p0_3p0_hz": 0.20,
            "residual_spectral_energy_fraction_3p0_nyquist_hz": 0.20,
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
            tmp_path
            / f"baseline_worker_{seed}"
            / f"episode_000_seed_{seed}",
            seed=seed,
            multiplier=1.0,
            residual_nonzero=False,
        )
        for seed in range(2001, 2006)
    ]
    candidate = [
        _fake_evaluation(
            tmp_path
            / f"candidate_worker_{seed}"
            / f"episode_000_seed_{seed}",
            seed=seed,
            multiplier=0.90,
            residual_nonzero=True,
        )
        for seed in reversed(range(2001, 2006))
    ]
    return baseline, candidate


def _aggregate_binding(
    tmp_path: Path,
    *,
    role: str,
    runs: list[LiveRunEvaluation],
    checkpoint: Path | None = None,
    checkpoint_manifest: Path | None = None,
) -> dict[str, object]:
    selected_runs = sorted(runs, key=lambda run: int(run.seed))
    source = tmp_path / f".{role}_aggregate_source.json"
    canonical_dirs = [str(run.run_directory.resolve()) for run in selected_runs]
    worker_dirs = [str(Path(path).resolve().parent) for path in canonical_dirs]
    source_payload = {
        "schema": "wlr50_clean.fresh_process_episode_batch.v1",
        "role": "baseline" if role == "baseline" else "candidate",
        "seed_set": "validation",
        "seeds": [int(run.seed) for run in selected_runs],
        "canonical_episode_dirs": canonical_dirs,
        "workers": [{"run_dir": path} for path in worker_dirs],
        "passed": True,
        "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
        "checkpoint_sha256": (
            subject.sha256_file(checkpoint) if checkpoint is not None else None
        ),
    }
    source.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record = {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": subject.sha256_file(source),
    }
    records = [record]
    binding: dict[str, object] = {
        "schema": "wlr50_clean.validation_aggregate_binding.v1",
        "path": record["path"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "role": role,
        "physical_passed": role == "baseline" or all(
            run.termination.task_success
            and run.termination.completed_p01_p13
            and not run.termination.body_collision
            and not run.termination.wheel_only_climb
            and not run.termination.physics_explosion_or_fall
            and not run.termination.safety_abort
            and run.termination.duration_s <= 200.0
            and run.termination.runtime_recording_access_count == 0
            for run in selected_runs
        ),
        "seeds": [int(run.seed) for run in selected_runs],
        "worker_run_dirs": worker_dirs,
        "canonical_episode_dirs": canonical_dirs,
        "source_file_records": records,
        "source_file_set_sha256": hashlib.sha256(
            json.dumps(
                records, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
    }
    if checkpoint is not None:
        manifest = checkpoint_manifest
        if manifest is None:
            manifest = tmp_path / ".candidate_aggregate_checkpoint_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
        binding.update(
            checkpoint_path=str(checkpoint.resolve()),
            checkpoint_sha256=subject.sha256_file(checkpoint),
            checkpoint_manifest_path=str(manifest.resolve()),
            checkpoint_manifest_sha256=subject.sha256_file(manifest),
        )
    return binding


def _paired_binding_kwargs(
    tmp_path: Path,
    baseline: list[LiveRunEvaluation],
    candidate: list[LiveRunEvaluation],
    checkpoint: Path | None = None,
    checkpoint_manifest: Path | None = None,
) -> dict[str, object]:
    return {
        "baseline_evaluation_aggregate": _aggregate_binding(
            tmp_path, role="baseline", runs=baseline
        ),
        "candidate_validation_aggregate": _aggregate_binding(
            tmp_path,
            role="candidate",
            runs=candidate,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
        ),
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _trust_synthetic_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {
        "schema": "wlr50_clean.committed_runtime_identity.v1",
        "git_commit": "0" * 40,
        "file_count": 1,
        "aggregate_sha256": "1" * 64,
        "content_sha256": "3" * 64,
        "files": [{"path": "pyproject.toml", "bytes": 1, "sha256": "2" * 64}],
    }

    def validate(run_dir, _run_manifest, **_kwargs):
        path = Path(run_dir) / "run_manifest.json"
        record = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": subject.sha256_file(path),
        }
        return record, record, identity

    monkeypatch.setattr(subject, "_validate_worker_runtime_identity", validate)
    monkeypatch.setattr(
        subject, "_validate_current_committed_runtime_identity", lambda _identity: None
    )
    monkeypatch.setattr(
        final_reporting,
        "_validate_current_committed_runtime_identity",
        lambda _identity: None,
    )


def _write_final_lifecycle_aggregate(
    tmp_path: Path,
    *,
    role: str,
    checkpoint_bytes: bytes | None,
    failed_seed: int | None = None,
    promotion_decision: Path | None = None,
) -> Path:
    worker_role = "baseline" if role == "pure_fsm" else "candidate"
    checkpoint_names = {
        "checkpoint_initial": "checkpoint_initial_zero_residual.pt",
        "checkpoint_smoke": "checkpoint_smoke.pt",
        "checkpoint_best": "checkpoint_best_validation.pt",
        "checkpoint_improved": "checkpoint_improved.pt",
    }
    checkpoint = None
    manifest = None
    if checkpoint_bytes is not None:
        creation_identity = (
            tmp_path
            / "synthetic-creation-run"
            / "committed_runtime_identity.before.json"
        )
        creation_identity.parent.mkdir(parents=True, exist_ok=True)
        if not creation_identity.exists():
            creation_identity.write_text("{}\n", encoding="utf-8")
        checkpoint = tmp_path / "checkpoints" / checkpoint_names[role]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(checkpoint_bytes)
        manifest = checkpoint.with_name(f"{checkpoint.stem}_manifest.json")
        manifest_payload = {
            "schema": "wlr50_clean.phase_residual_checkpoint_manifest.v1",
            "training_seed": 1001,
            "source_git_commit": "0" * 40,
            "committed_runtime_content_sha256": "3" * 64,
            "creation_runtime_identity_sha256": subject.sha256_file(
                creation_identity
            ),
            "creation_runtime_identity_path": str(creation_identity.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": subject.sha256_file(checkpoint),
        }
        if promotion_decision is not None:
            manifest_payload.update(
                {
                    "promotion_decision": str(promotion_decision.resolve()),
                    "promotion_decision_sha256": subject.sha256_file(
                        promotion_decision
                    ),
                }
            )
        manifest.write_text(
            json.dumps(
                manifest_payload,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    worker_dirs: list[Path] = []
    for seed in subject.BASELINE_VALIDATION_SEEDS:
        run_dir = tmp_path / f"{role}_workers" / f"worker_{seed}"
        episode_dir = run_dir / f"episode_000_seed_{seed}"
        episode_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"lifecycle": "SUCCEEDED", "exit_code": 0}) + "\n",
            encoding="utf-8",
        )
        for name in subject.CANONICAL_EPISODE_FILES:
            payload = (
                {
                    "seed": seed,
                    "action_projection_audit": {
                        "exact_pair_contact_contract_valid": True,
                    },
                }
                if name == "trial_manifest.json"
                else {}
            )
            (episode_dir / name).write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
        physical_failure = seed == failed_seed
        episode = {
            "seed": seed,
            "task_success": not physical_failure,
            "duration_s": 100.0,
            "body_collision": physical_failure,
            "wheel_only_climb": False,
            "safety_abort": False,
            "under_maximum_duration": True,
            "recording_runtime_access_count": 0,
            "in_episode_root_write_count": 0,
            "canonical_episode_dir": str(episode_dir.resolve()),
        }
        worker_passed = not physical_failure
        if checkpoint is None:
            result = {
                "schema": "wlr50_clean.live_residual_gate.v1",
                "mode": "zero",
                "episode_count": 1,
                "passed": worker_passed,
                "episodes": [episode],
            }
            result_name = "acceptance.json"
        else:
            assert manifest is not None
            result = {
                "schema": "wlr50_clean.ppo_checkpoint_evaluation.v1",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": subject.sha256_file(checkpoint),
                "checkpoint_manifest": str(manifest.resolve()),
                "checkpoint_manifest_sha256": subject.sha256_file(manifest),
                "fresh_process_single_episode": True,
                "vec_env_step_called": False,
                "deterministic_mean_policy": True,
                "episode_count": 1,
                "passed": worker_passed,
                "episodes": [episode],
            }
            result_name = "checkpoint_evaluation.json"
        (run_dir / result_name).write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        worker_dirs.append(run_dir)

    batch = subject.collect_fresh_process_episode_workers(
        worker_dirs,
        seeds=subject.BASELINE_VALIDATION_SEEDS,
        role=worker_role,
        checkpoint_path=checkpoint,
    )
    episodes = [dict(row) for row in batch.episode_rows]
    worker_gate_pass_count = sum(
        row.get("worker_gate_passed") is True for row in batch.worker_rows
    )
    passed = bool(
        all(row["task_success"] is True for row in episodes)
        and all(row["body_collision"] is False for row in episodes)
        and all(row["wheel_only_climb"] is False for row in episodes)
        and all(row["safety_abort"] is False for row in episodes)
        and all(row["under_maximum_duration"] is True for row in episodes)
        and all(row["recording_runtime_access_count"] == 0 for row in episodes)
        and all(row["in_episode_root_write_count"] == 0 for row in episodes)
        and worker_gate_pass_count == len(subject.BASELINE_VALIDATION_SEEDS)
    )
    aggregate = {
        "schema": "wlr50_clean.fresh_process_episode_batch.v1",
        "role": worker_role,
        "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "checkpoint_sha256": (
            None if checkpoint is None else subject.sha256_file(checkpoint)
        ),
        "seed_set": "validation",
        "seeds": list(batch.seeds),
        "canonical_episode_dirs": [str(path) for path in batch.canonical_episode_dirs],
        "fresh_process_per_episode": True,
        "deterministic_evaluation": True,
        "deterministic_mean_policy": True if checkpoint is not None else None,
        "pure_fsm_zero_residual": True if checkpoint is None else None,
        "episode_count": 5,
        "success_count": sum(row["task_success"] is True for row in episodes),
        "body_collision_count": sum(row["body_collision"] is True for row in episodes),
        "wheel_only_climb_count": 0,
        "safety_abort_count": 0,
        "all_under_maximum_duration": True,
        "passed": passed,
        "worker_gate_pass_count": worker_gate_pass_count,
        "workers": [dict(row) for row in batch.worker_rows],
        "episodes": episodes,
    }
    aggregate_path = tmp_path / f"{role}_aggregate" / "evaluation_aggregate.json"
    aggregate_path.parent.mkdir()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate_path


def _final_lifecycle_aggregates(tmp_path: Path) -> dict[str, Path]:
    promoted_bytes = b"promoted best-validation checkpoint"
    terminal_checkpoint = tmp_path / "training" / "terminal_checkpoint.pt"
    terminal_checkpoint.parent.mkdir(parents=True)
    terminal_checkpoint.write_bytes(promoted_bytes)
    cadence_decision = tmp_path / "cadence" / "promotion_decision.json"
    cadence_decision.parent.mkdir(parents=True)
    cadence_decision.write_text(
        json.dumps(
            {
                "candidate_checkpoint_path": str(terminal_checkpoint.resolve()),
                "candidate_checkpoint_sha256": subject.sha256_file(
                    terminal_checkpoint
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "pure_fsm": _write_final_lifecycle_aggregate(
            tmp_path, role="pure_fsm", checkpoint_bytes=None
        ),
        "checkpoint_initial": _write_final_lifecycle_aggregate(
            tmp_path, role="checkpoint_initial", checkpoint_bytes=b"initial checkpoint"
        ),
        "checkpoint_smoke": _write_final_lifecycle_aggregate(
            tmp_path,
            role="checkpoint_smoke",
            checkpoint_bytes=b"smoke checkpoint",
            failed_seed=2003,
        ),
        "checkpoint_best": _write_final_lifecycle_aggregate(
            tmp_path,
            role="checkpoint_best",
            checkpoint_bytes=promoted_bytes,
            promotion_decision=cadence_decision,
        ),
        "checkpoint_improved": _write_final_lifecycle_aggregate(
            tmp_path,
            role="checkpoint_improved",
            checkpoint_bytes=promoted_bytes,
            promotion_decision=cadence_decision,
        ),
    }


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
    incomplete = replace(
        complete,
        phase_rows=complete.phase_rows[:-1],
        residual_activity_rows=complete.residual_activity_rows[:-1],
        reward_contribution_rows=complete.reward_contribution_rows[:-1],
    )
    monkeypatch.setattr(subject, "evaluate_live_run", lambda *args, **kwargs: incomplete)
    with pytest.raises(subject.EvaluationArtifactError, match="ordered P01-P13 exactly"):
        subject.evaluate_canonical_episode_dirs(
            (first,), seeds=(3,), residual_calibration=_simple_calibration()
        )

    accepted = subject.evaluate_canonical_episode_dirs(
        (first,),
        seeds=(3,),
        residual_calibration=_simple_calibration(),
        require_complete_phase_sequence=False,
    )
    assert tuple(row["phase"] for row in accepted[0].phase_rows) == PHASE_IDS[:-1]

    out_of_order = replace(incomplete, phase_rows=tuple(reversed(incomplete.phase_rows)))
    monkeypatch.setattr(subject, "evaluate_live_run", lambda *args, **kwargs: out_of_order)
    with pytest.raises(subject.EvaluationArtifactError, match="contiguous canonical phase prefix"):
        subject.evaluate_canonical_episode_dirs(
            (first,),
            seeds=(3,),
            residual_calibration=_simple_calibration(),
            require_complete_phase_sequence=False,
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
        **_paired_binding_kwargs(tmp_path, baseline, candidate, checkpoint),
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
        **_paired_binding_kwargs(tmp_path, baseline, candidate, checkpoint),
    )
    assert repeated == paths
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    } == before


def test_exported_promotion_decision_is_accepted_by_checkpoint_promotion(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    baseline, candidate = _paired_runs(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    snapshot_bundle = validated_phase_snapshot_bundle_record(
        checkpoint_promotion.DEFAULT_PHASE_SNAPSHOT_ROOT
    )
    infos = {
        "schema": checkpoint_promotion.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "full-episode",
        "training_seed": 1001,
        "global_policy_decisions": 100_000,
        "source_git_commit": "a" * 40,
        "committed_runtime_content_sha256": "b" * 64,
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "files": {
            "training.yaml": "f" * 64,
            **phase_snapshot_bundle_file_hashes(snapshot_bundle),
        },
        "controller_hash": "a" * 64,
        "environment_hash": "b" * 64,
        "observation_schema_hash": "c" * 64,
        "action_schema_hash": "d" * 64,
        "reward_config_hash": "e" * 64,
        "phase_snapshot_manifest": snapshot_bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": snapshot_bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": snapshot_bundle["bundle_sha256"],
        "phase_snapshot_bundle": snapshot_bundle,
    }
    torch.save({"infos": infos}, checkpoint)
    checkpoint_manifest = tmp_path / "candidate_manifest.json"
    checkpoint_manifest.write_text(
        json.dumps(
            {
                **infos,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": subject.sha256_file(checkpoint),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths = subject.export_paired_evaluation_artifacts(
        tmp_path / "metrics",
        baseline_runs=baseline,
        candidate_runs=candidate,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name="candidate",
        candidate_checkpoint_path=checkpoint,
        residual_calibration_evidence=subject.build_versioned_residual_activity_calibration(),
        **_paired_binding_kwargs(
            tmp_path,
            baseline,
            candidate,
            checkpoint,
            checkpoint_manifest,
        ),
    )

    exported = json.loads(paths.promotion_decision.read_text(encoding="utf-8"))
    exported_gate_order = tuple(
        row["gate"] for row in exported["checks_in_evaluation_order"]
    )
    assert exported_gate_order == checkpoint_promotion.REQUIRED_PROMOTION_GATES
    assert set(exported["promotion"]["checks"]) == set(
        checkpoint_promotion.REQUIRED_PROMOTION_GATES
    )

    promoted = checkpoint_promotion.promote_best_validation_checkpoint(
        promotion_decision_path=paths.promotion_decision,
        candidate_checkpoint_path=checkpoint,
        candidate_manifest_path=checkpoint_manifest,
        output_root=tmp_path / "publication",
    )
    assert promoted.best_checkpoint.read_bytes() == checkpoint.read_bytes()


def test_initial_checkpoint_requires_exact_zero_full12_actor_output() -> None:
    torch = pytest.importorskip("torch")
    import io

    payload = {
        "actor_state_dict": {
            "actor.0.weight": torch.ones((32, 125)),
            "actor.0.bias": torch.zeros(32),
            "actor.output.weight": torch.zeros((12, 32)),
            "actor.output.bias": torch.zeros(12),
        },
        "infos": {},
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    subject._require_initial_zero_actor_output(
        stream.getvalue(), {"residual_dimension": 12}
    )

    payload["actor_state_dict"]["actor.output.bias"][3] = 1.0e-12
    stream = io.BytesIO()
    torch.save(payload, stream)
    with pytest.raises(subject.EvaluationArtifactError, match="exact zero"):
        subject._require_initial_zero_actor_output(
            stream.getvalue(), {"residual_dimension": 12}
        )


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
        **_paired_binding_kwargs(tmp_path, baseline, unsafe),
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


def test_partial_candidate_batch_publishes_strict_failed_decision(
    tmp_path: Path,
) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    failed = list(candidate)
    phase_prefix = PHASE_IDS[:5]
    failed[0] = replace(
        failed[0],
        termination=replace(
            failed[0].termination,
            result="BODY_COLLISION",
            reason="synthetic early physical failure",
            final_state_id=phase_prefix[-1],
            completed_phases=phase_prefix[:-1],
            completed_p01_p13=False,
            task_success=False,
            body_collision=True,
            failed_checks=("body_collision",),
        ),
        episode_row={
            **failed[0].episode_row,
            "task_result": "BODY_COLLISION",
            "task_success": False,
            "completed_p01_p13": False,
            "body_collision": True,
            "home_recovery_action_jerk_rms": None,
        },
        phase_rows=failed[0].phase_rows[: len(phase_prefix)],
        residual_activity_rows=failed[0].residual_activity_rows[: len(phase_prefix)],
        reward_contribution_rows=failed[0].reward_contribution_rows[: len(phase_prefix)],
    )

    paths = subject.export_paired_evaluation_artifacts(
        tmp_path / "partial-failed-metrics",
        baseline_runs=baseline,
        candidate_runs=failed,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name="failed_candidate",
        **_paired_binding_kwargs(tmp_path, baseline, failed),
    )
    payload = json.loads(paths.promotion_decision.read_text(encoding="utf-8"))

    assert payload["candidate_validation_aggregate"]["physical_passed"] is False
    assert payload["promotion"]["promoted"] is False
    assert payload["first_failed_gate"] == "p01_p13_completed"
    assert tuple(
        row["gate"] for row in payload["checks_in_evaluation_order"]
    ) == checkpoint_promotion.REQUIRED_PROMOTION_GATES
    comparison_rows = _csv_rows(paths.phase_metric_comparison)
    assert len(comparison_rows) == len(PHASE_IDS)
    assert all(row["comparison_available"] == "False" for row in comparison_rows)
    assert comparison_rows[-1]["candidate_phase_observed_in_all_runs"] == "False"
    assert len(_csv_rows(paths.candidate_phase_metrics)) == 4 * len(PHASE_IDS) + 5


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
            **_paired_binding_kwargs(tmp_path, baseline, mismatched),
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
            **_paired_binding_kwargs(tmp_path, baseline, duplicate),
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
            **_paired_binding_kwargs(tmp_path, baseline, candidate),
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
            **_paired_binding_kwargs(tmp_path, baseline, missing_reward),
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
            **_paired_binding_kwargs(tmp_path, baseline, missing_residual),
        )


def test_final_lifecycle_export_has_exact_five_roles_and_improved_only_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregates = _final_lifecycle_aggregates(tmp_path)
    multipliers = {
        "pure_fsm": 1.0,
        "checkpoint_initial": 1.0,
        "checkpoint_smoke": 0.96,
        "checkpoint_best": 0.91,
        "checkpoint_improved": 0.90,
    }

    def evaluate(directories, *, seeds, **_kwargs):
        role = Path(directories[0]).parent.parent.name.removesuffix("_workers")
        runs = []
        for directory, seed in zip(directories, seeds, strict=True):
            run = _fake_evaluation(
                Path(directory),
                seed=seed,
                multiplier=multipliers[role],
                residual_nonzero=role not in {"pure_fsm", "checkpoint_initial"},
            )
            if role == "checkpoint_smoke" and seed == 2003:
                partial_phases = PHASE_IDS[:10]
                termination = replace(
                    run.termination,
                    result="BODY_COLLISION",
                    reason="synthetic intermediate physical failure",
                    final_state_id="P10",
                    completed_phases=partial_phases[:9],
                    completed_p01_p13=False,
                    task_success=False,
                    body_collision=True,
                    failed_checks=("body_collision",),
                )
                episode_row = {
                    **run.episode_row,
                    "task_result": termination.result,
                    "task_success": False,
                    "completed_p01_p13": False,
                    "body_collision": True,
                    "home_recovery_action_jerk_rms": None,
                }
                run = replace(
                    run,
                    termination=termination,
                    episode_row=episode_row,
                    phase_rows=run.phase_rows[:10],
                    residual_activity_rows=run.residual_activity_rows[:10],
                    reward_contribution_rows=run.reward_contribution_rows[:10],
                )
            runs.append(run)
        return tuple(runs)

    monkeypatch.setattr(subject, "evaluate_canonical_episode_dirs", evaluate)
    monkeypatch.setattr(
        subject, "_validate_checkpoint_role_contract", lambda **_kwargs: ()
    )
    _trust_synthetic_runtime_identity(monkeypatch)
    paths = subject.export_final_lifecycle_evaluation_artifacts(
        tmp_path / "metrics",
        pure_fsm_aggregate=aggregates["pure_fsm"],
        checkpoint_initial_aggregate=aggregates["checkpoint_initial"],
        checkpoint_smoke_aggregate=aggregates["checkpoint_smoke"],
        checkpoint_best_aggregate=aggregates["checkpoint_best"],
        checkpoint_improved_aggregate=aggregates["checkpoint_improved"],
        frozen_hashes_unchanged=True,
        residual_calibration_evidence=subject.build_versioned_residual_activity_calibration(),
    )

    checkpoint_rows = _csv_rows(paths.checkpoint_comparison)
    assert [row["role"] for row in checkpoint_rows] == list(
        subject.FINAL_LIFECYCLE_ROLES
    )
    assert [row["checkpoint"] for row in checkpoint_rows] == list(
        subject.FINAL_LIFECYCLE_ROLES
    )
    assert checkpoint_rows[0]["checkpoint_path"] == ""
    assert all(
        len(row["evaluation_aggregate_sha256"]) == 64 for row in checkpoint_rows
    )
    assert checkpoint_rows[3]["checkpoint_sha256"] == checkpoint_rows[4][
        "checkpoint_sha256"
    ]
    termination_rows = _csv_rows(paths.termination_summary)
    assert len(termination_rows) == 25
    termination_roles = [row["checkpoint"] for row in termination_rows]
    assert termination_roles == [
        role for role in subject.FINAL_LIFECYCLE_ROLES for _ in range(5)
    ]
    smoke_failure = next(
        row
        for row in termination_rows
        if row["checkpoint"] == "checkpoint_smoke" and row["seed"] == "2003"
    )
    assert smoke_failure["task_success"] == "False"
    assert smoke_failure["body_collision"] == "True"
    assert checkpoint_rows[2]["task_success_count"] == "4"
    assert checkpoint_rows[2]["mean_home_recovery_action_jerk_rms"] == ""
    phase_rows = _csv_rows(paths.phase_metric_comparison)
    assert len(phase_rows) == 13
    assert [row["phase"] for row in phase_rows] == list(PHASE_IDS)
    assert {row["baseline_checkpoint"] for row in phase_rows} == {"pure_fsm"}
    assert {row["improved_checkpoint"] for row in phase_rows} == {
        "checkpoint_improved"
    }
    assert {
        row["checkpoint"] for row in _csv_rows(paths.residual_activity_by_phase)
    } == {"checkpoint_improved"}
    assert {
        row["checkpoint"] for row in _csv_rows(paths.reward_contribution_by_phase)
    } == {"checkpoint_improved"}
    promotion = json.loads(paths.promotion_decision.read_text(encoding="utf-8"))
    assert promotion["bundle_kind"] == subject.FINAL_LIFECYCLE_BUNDLE_KIND
    assert promotion["final_lifecycle_roles"] == list(subject.FINAL_LIFECYCLE_ROLES)
    assert promotion["promotion"]["promoted"] is True
    pure_evidence = subject.validate_final_lifecycle_aggregate_evidence(
        aggregates["pure_fsm"], role="pure_fsm"
    )
    baseline_bundle = subject.export_baseline_evaluation_artifacts(
        paths.output_directory,
        episode_directories=pure_evidence.canonical_episode_dirs,
        seeds=pure_evidence.seeds,
        residual_calibration_evidence=(
            subject.build_versioned_residual_activity_calibration()
        ),
        baseline_name="pure_fsm",
    )
    assert baseline_bundle.manifest.is_file()

    orchestration_path = tmp_path / "training_orchestration_manifest.json"
    orchestration_path.write_text("{}\n", encoding="utf-8")
    initial_record = promotion["final_lifecycle_evidence"]["checkpoint_initial"]
    improved_record = promotion["final_lifecycle_evidence"]["checkpoint_improved"]
    smoke_record = promotion["final_lifecycle_evidence"]["checkpoint_smoke"]
    improved_manifest_payload = json.loads(
        Path(improved_record["checkpoint_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    cadence_decision_path = Path(improved_manifest_payload["promotion_decision"])
    cadence_decision = json.loads(
        cadence_decision_path.read_text(encoding="utf-8")
    )
    cadence_checkpoint_path = Path(
        cadence_decision["candidate_checkpoint_path"]
    )
    cadence_checkpoint_record = {
        "path": str(cadence_checkpoint_path.resolve()),
        "bytes": cadence_checkpoint_path.stat().st_size,
        "sha256": subject.sha256_file(cadence_checkpoint_path),
    }
    cadence_decision_record = {
        "path": str(cadence_decision_path.resolve()),
        "bytes": cadence_decision_path.stat().st_size,
        "sha256": subject.sha256_file(cadence_decision_path),
    }
    creation_identity_path = Path(
        initial_record["creation_runtime_identity_path"]
    )
    monkeypatch.setattr(
        final_reporting,
        "validate_training_orchestration_manifest",
        lambda *_args, **_kwargs: {
            "path": str(orchestration_path.resolve()),
            "bytes": orchestration_path.stat().st_size,
            "sha256": subject.sha256_file(orchestration_path),
            "payload": {
                "schema": final_reporting.TRAINING_ORCHESTRATION_SCHEMA,
                "valid": True,
                "status": "PROMOTION_FOUND",
                "training_seed": 1001,
                "git_commit": "0" * 40,
                "initial_checkpoint": {
                    "path": initial_record["checkpoint_path"],
                    "sha256": initial_record["checkpoint_sha256"],
                    "manifest_path": initial_record["checkpoint_manifest_path"],
                    "manifest_sha256": initial_record[
                        "checkpoint_manifest_sha256"
                    ],
                },
                "smoke_checkpoint": {
                    "path": smoke_record["checkpoint_path"],
                    "sha256": smoke_record["checkpoint_sha256"],
                    "manifest_path": smoke_record["checkpoint_manifest_path"],
                    "manifest_sha256": smoke_record[
                        "checkpoint_manifest_sha256"
                    ],
                },
                "canonical_smoke_checkpoint": {
                    "path": smoke_record["checkpoint_path"],
                    "sha256": smoke_record["checkpoint_sha256"],
                    "manifest_path": smoke_record["checkpoint_manifest_path"],
                    "manifest_sha256": smoke_record[
                        "checkpoint_manifest_sha256"
                    ],
                },
                "terminal": {
                    "chunk_index": 0,
                    "checkpoint": cadence_checkpoint_record,
                },
                "chunks": [
                    {
                        "stage": "smoke",
                        "training": {
                            "immutable_history_checkpoint": {
                                "path": promotion["final_lifecycle_evidence"][
                                    "checkpoint_smoke"
                                ]["checkpoint_path"],
                                "sha256": promotion["final_lifecycle_evidence"][
                                    "checkpoint_smoke"
                                ]["checkpoint_sha256"],
                            }
                        },
                    }
                ],
                "promotion_decisions": [
                        {
                            "promoted": True,
                            "record": cadence_decision_record,
                            "bound_chunk_index": 0,
                            "candidate_checkpoint": cadence_checkpoint_record,
                        }
                ],
            },
            "source_file_records": [
                {
                    "path": str(creation_identity_path.resolve()),
                    "bytes": creation_identity_path.stat().st_size,
                    "sha256": subject.sha256_file(creation_identity_path),
                }
            ],
            "status": "PROMOTION_FOUND",
            "valid": True,
        },
    )
    evidence = final_reporting._load_evidence(
        paths.output_directory,
        phase_objectives_config=final_reporting.DEFAULT_PHASE_OBJECTIVES_PATH,
        phase_action_config=final_reporting.DEFAULT_PHASE_ACTION_CONFIG_V2,
        reward_config=final_reporting.DEFAULT_REWARD_PATH_V2,
        reward_migration_config=final_reporting.DEFAULT_MIGRATION_PATH,
        training_orchestration_manifest=orchestration_path,
    )
    assert evidence.final_lifecycle is True
    assert evidence.checkpoint_roles == subject.FINAL_LIFECYCLE_ROLES
    training_report = final_reporting._training_report(evidence)
    improvement_report = final_reporting._improvement_report(evidence)
    assert all(f"| {role} |" in training_report for role in subject.FINAL_LIFECYCLE_ROLES)
    assert all(f"| {role} |" in improvement_report for role in subject.FINAL_LIFECYCLE_ROLES)
    reporting_paths = final_reporting.generate_final_reporting_bundle(
        paths.output_directory,
        tmp_path / "final-reporting",
        training_orchestration_manifest=orchestration_path,
    )
    assert all(path.is_file() for path in reporting_paths.files())


def test_final_lifecycle_export_rejects_seed_label_and_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregates = _final_lifecycle_aggregates(tmp_path)
    monkeypatch.setattr(
        subject,
        "evaluate_canonical_episode_dirs",
        lambda directories, *, seeds, **_kwargs: tuple(
            _fake_evaluation(
                Path(directory),
                seed=seed,
                multiplier=0.9,
                residual_nonzero=("pure_fsm_workers" not in str(directory)),
            )
            for directory, seed in zip(directories, seeds, strict=True)
        ),
    )
    monkeypatch.setattr(
        subject, "_validate_checkpoint_role_contract", lambda **_kwargs: ()
    )
    _trust_synthetic_runtime_identity(monkeypatch)
    tampered = json.loads(aggregates["checkpoint_smoke"].read_text(encoding="utf-8"))
    tampered["seeds"][-1] = 9999
    aggregates["checkpoint_smoke"].write_text(
        json.dumps(tampered) + "\n", encoding="utf-8"
    )
    with pytest.raises(subject.EvaluationArtifactError, match="validation seeds"):
        subject.export_final_lifecycle_evaluation_artifacts(
            tmp_path / "seed-failure",
            pure_fsm_aggregate=aggregates["pure_fsm"],
            checkpoint_initial_aggregate=aggregates["checkpoint_initial"],
            checkpoint_smoke_aggregate=aggregates["checkpoint_smoke"],
            checkpoint_best_aggregate=aggregates["checkpoint_best"],
            checkpoint_improved_aggregate=aggregates["checkpoint_improved"],
            frozen_hashes_unchanged=True,
        )
    assert not (tmp_path / "seed-failure").exists()

    aggregates = _final_lifecycle_aggregates(tmp_path / "manifest-case")
    improved_payload = json.loads(
        aggregates["checkpoint_improved"].read_text(encoding="utf-8")
    )
    improved_result = Path(improved_payload["workers"][0]["worker_result"])
    result_payload = json.loads(improved_result.read_text(encoding="utf-8"))
    Path(result_payload["checkpoint_manifest"]).write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(subject.EvaluationArtifactError, match="manifest SHA-256 is stale"):
        subject.export_final_lifecycle_evaluation_artifacts(
            tmp_path / "manifest-failure",
            pure_fsm_aggregate=aggregates["pure_fsm"],
            checkpoint_initial_aggregate=aggregates["checkpoint_initial"],
            checkpoint_smoke_aggregate=aggregates["checkpoint_smoke"],
            checkpoint_best_aggregate=aggregates["checkpoint_best"],
            checkpoint_improved_aggregate=aggregates["checkpoint_improved"],
            frozen_hashes_unchanged=True,
        )
    assert not (tmp_path / "manifest-failure").exists()

    aggregates = _final_lifecycle_aggregates(tmp_path / "label-case")
    with pytest.raises(subject.EvaluationArtifactError, match="checkpoint_initial must bind"):
        subject.export_final_lifecycle_evaluation_artifacts(
            tmp_path / "label-failure",
            pure_fsm_aggregate=aggregates["pure_fsm"],
            checkpoint_initial_aggregate=aggregates["checkpoint_smoke"],
            checkpoint_smoke_aggregate=aggregates["checkpoint_initial"],
            checkpoint_best_aggregate=aggregates["checkpoint_best"],
            checkpoint_improved_aggregate=aggregates["checkpoint_improved"],
            frozen_hashes_unchanged=True,
        )
    assert not (tmp_path / "label-failure").exists()


def test_final_lifecycle_only_allows_physical_failure_for_intermediate_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject, "_validate_checkpoint_role_contract", lambda **_kwargs: ()
    )
    _trust_synthetic_runtime_identity(monkeypatch)
    initial = _write_final_lifecycle_aggregate(
        tmp_path / "initial",
        role="checkpoint_initial",
        checkpoint_bytes=b"initial checkpoint",
        failed_seed=2002,
    )
    evidence = subject._final_lifecycle_aggregate_evidence(
        initial, role="checkpoint_initial"
    )
    assert evidence.role == "checkpoint_initial"

    best = _write_final_lifecycle_aggregate(
        tmp_path / "best",
        role="checkpoint_best",
        checkpoint_bytes=b"best checkpoint",
        failed_seed=2002,
    )
    with pytest.raises(subject.EvaluationArtifactError, match="must pass every"):
        subject._final_lifecycle_aggregate_evidence(best, role="checkpoint_best")


def test_final_reporting_rejects_legacy_two_role_bundle_by_default(
    tmp_path: Path,
) -> None:
    baseline, candidate = _paired_runs(tmp_path)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    metrics = subject.export_paired_evaluation_artifacts(
        tmp_path / "legacy-metrics",
        baseline_runs=baseline,
        candidate_runs=candidate,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name="candidate",
        candidate_checkpoint_path=checkpoint,
        **_paired_binding_kwargs(tmp_path, baseline, candidate, checkpoint),
    )
    with pytest.raises(final_reporting.FinalReportingError, match="five-role"):
        final_reporting.generate_final_reporting_bundle(
            metrics.output_directory,
            tmp_path / "must-not-be-final",
            training_orchestration_manifest=tmp_path
            / "training_orchestration_manifest.json",
        )
    assert not (tmp_path / "must-not-be-final").exists()

    paths = final_reporting.generate_nonfinal_two_role_reporting_bundle_for_testing(
        metrics.output_directory, tmp_path / "explicit-nonfinal"
    )
    assert all(path.is_file() for path in paths.files())
