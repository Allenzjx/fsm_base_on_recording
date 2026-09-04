from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wlr50_clean.ppo import artifacts
from wlr50_clean.ppo import checkpoint_promotion as checkpoint
from wlr50_clean.ppo import finalization as subject
from wlr50_clean.ppo.evaluation_artifacts import (
    BASELINE_EPISODE_FILENAME,
    BASELINE_EVALUATION_MANIFEST_FILENAME,
    BASELINE_PHASE_FILENAME,
    CANONICAL_EPISODE_FILES,
    CANDIDATE_EPISODE_FILENAME,
    CANDIDATE_PHASE_FILENAME,
    CHECKPOINT_COMPARISON_FILENAME,
    PHASE_COMPARISON_FILENAME,
    PROMOTION_DECISION_FILENAME,
    RESIDUAL_ACTIVITY_FILENAME,
    REWARD_CONTRIBUTION_FILENAME,
    TERMINATION_SUMMARY_FILENAME,
)
from wlr50_clean.ppo.final_reporting import PLOT_FILENAMES, REPORT_FILENAMES
from wlr50_clean.ppo.video_artifacts import (
    COMPARISON_VIDEO_NAME,
    DIAGNOSTIC_VIDEO_NAME,
    FSM_VIDEO_NAME,
    PPO_VIDEO_NAME,
    VIDEO_CHECKSUM_NAME,
    VIDEO_VALIDATION_NAME,
)


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def _record(path: Path, *, relative_to: Path | None = None) -> dict:
    source = path.resolve()
    display = (
        source.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(source)
    )
    return {
        "path": display,
        "bytes": source.stat().st_size,
        "sha256": artifacts.sha256_file(source),
    }


def _make_training_runs(tmp_path: Path, output: Path) -> tuple[list[Path], Path, Path]:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1001\n", encoding="utf-8")
    checkpoint_inputs = {}
    for name in (
        "fsm_states.yaml",
        "environment_lock.json",
        "ppo_observation_schema_v2.json",
        "ppo_phase_action_masks_v2.yaml",
        "ppo_reward_v2.yaml",
    ):
        path = tmp_path / "checkpoint-inputs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        checkpoint_inputs[name] = path
    initial = tmp_path / "checkpoint_initial.pt"
    initial.write_bytes(b"initial-zero-policy")
    history_root = tmp_path / "training-history"
    history_root.mkdir()
    last_path = output / "checkpoints" / "checkpoint_last.pt"
    last_path.parent.mkdir(parents=True)
    previous = initial
    cumulative = 0
    run_dirs: list[Path] = []
    terminal_manifest: Path | None = None

    for index, stage in enumerate(subject.REQUIRED_TRAINING_STAGES, 1):
        decisions = 8
        cumulative += decisions
        run_dir = tmp_path / "runs" / f"train-{index}-{stage}"
        run_dir.mkdir(parents=True)
        started = _json(run_dir / "run_manifest.started.json", {"started": True})
        stdout = run_dir / "stdout.log"
        stderr = run_dir / "stderr.log"
        stdout.write_text("training complete\n", encoding="utf-8")
        stderr.write_bytes(b"")
        frozen_entry = {
            "path": "config.yaml",
            "expected_sha256": artifacts.sha256_file(config),
            "actual_sha256": artifacts.sha256_file(config),
            "exists": True,
            "valid": True,
        }
        _json(
            run_dir / "frozen_hashes.before.json",
            {"passed": True, "mismatches": [], "entries": [frozen_entry]},
        )
        _json(
            run_dir / "frozen_hashes.after.json",
            {"passed": True, "mismatches": [], "entries": [frozen_entry]},
        )

        history = history_root / f"checkpoint_{stage}_{cumulative}.pt"
        history.write_bytes(f"checkpoint-after-{stage}".encode("ascii"))
        history_manifest = _json(
            history.with_name(history.stem + "_manifest.json"),
            {
                "schema": checkpoint.CHECKPOINT_MANIFEST_SCHEMA,
                "stage": stage,
                "global_policy_decisions": cumulative,
                "actor_observation_dimension": 125,
                "critic_observation_dimension": 125,
                "residual_dimension": 12,
                "physics_hz": 120.0,
                "decision_hz": 15.0,
                "files": {
                    str(path.resolve()): artifacts.sha256_file(path)
                    for path in (config, *checkpoint_inputs.values())
                },
                "controller_hash": artifacts.sha256_file(
                    checkpoint_inputs["fsm_states.yaml"]
                ),
                "environment_hash": artifacts.sha256_file(
                    checkpoint_inputs["environment_lock.json"]
                ),
                "observation_schema_hash": artifacts.sha256_file(
                    checkpoint_inputs["ppo_observation_schema_v2.json"]
                ),
                "action_schema_hash": artifacts.sha256_file(
                    checkpoint_inputs["ppo_phase_action_masks_v2.yaml"]
                ),
                "reward_config_hash": artifacts.sha256_file(
                    checkpoint_inputs["ppo_reward_v2.yaml"]
                ),
                "resume_checkpoint": str(previous.resolve()),
                "resume_checkpoint_sha256": artifacts.sha256_file(previous),
                "resume_global_policy_decisions": cumulative - decisions,
                "checkpoint_path": str(history.resolve()),
                "checkpoint_sha256": artifacts.sha256_file(history),
            },
        )
        terminal_manifest = history_manifest
        last_path.write_bytes(history.read_bytes())
        training = {
            "schema": subject.TRAINING_RESULT_SCHEMA,
            "stage": stage,
            "requested_policy_decisions": decisions,
            "stage_policy_decisions": decisions,
            "global_policy_decisions": cumulative,
            "resume_checkpoint": str(previous.resolve()),
            "resume_checkpoint_sha256": artifacts.sha256_file(previous),
            "iterations": 1,
            "num_envs": 1,
            "rollout_length": decisions,
            "environment_contract": {"physics_hz": 120, "decision_hz": 15},
            "training_telemetry": {
                "policy_decision_count": decisions,
                "reward_telemetry_complete": True,
            },
            "wall_time_s": 1.0,
            "checkpoint_last": str(last_path.resolve()),
            "checkpoint_sha256": artifacts.sha256_file(history),
            "immutable_history_checkpoint": str(history.resolve()),
            "save_load_round_trip": True,
            "round_trip_infos": {"global_policy_decisions": cumulative},
            "runner_config": {"seed": 1001},
        }
        _json(run_dir / "training_result.json", training)
        lifecycle = {
            "schema": artifacts.RUN_MANIFEST_SCHEMA,
            "lifecycle": "SUCCEEDED",
            "exit_code": 0,
            "immutable_run_directory": True,
            "run_kind": "train",
            "run_dir": str(run_dir.resolve()),
            "project_root": str(tmp_path.resolve()),
            "identity": {"training_stage": stage},
            "configs": [_record(config, relative_to=tmp_path)],
            "started_manifest": _record(started, relative_to=run_dir),
            "logs": {
                "stdout.log": _record(stdout, relative_to=run_dir),
                "stderr.log": _record(stderr, relative_to=run_dir),
            },
        }
        _json(run_dir / "run_manifest.json", lifecycle)
        run_dirs.append(run_dir.resolve())
        previous = history

    assert terminal_manifest is not None
    return run_dirs, previous.resolve(), terminal_manifest


def _episode(seed: int, directory: Path) -> dict:
    return {
        "seed": seed,
        "task_success": True,
        "termination_reason": "SUCCESS",
        "duration_s": 12.0,
        "body_collision": False,
        "wheel_only_climb": False,
        "safety_abort": False,
        "under_maximum_duration": True,
        "recording_runtime_access_count": 0,
        "in_episode_root_write_count": 0,
        "canonical_episode_dir": str(directory.resolve()),
    }


def _make_batch(
    tmp_path: Path,
    *,
    name: str,
    role: str,
    seeds: tuple[int, ...],
    seed_set: str,
    checkpoint_path: Path | None = None,
    checkpoint_manifest: Path | None = None,
    finalized: bool = False,
) -> Path:
    workers = []
    episodes = []
    batch_root = tmp_path / name.removesuffix(".json")
    for seed in seeds:
        run_dir = batch_root / f"worker-{seed}"
        episode_dir = run_dir / f"episode_000_seed_{seed}"
        episode_dir.mkdir(parents=True)
        for filename in CANONICAL_EPISODE_FILES:
            if filename != "trial_manifest.json":
                (episode_dir / filename).write_text(
                    "" if filename == "state_transitions.jsonl" else "{}\n",
                    encoding="utf-8",
                )
        episode = _episode(seed, episode_dir)
        trial = _json(
            episode_dir / "trial_manifest.json",
            {
                "schema": "wlr50_clean.ppo_live_trial_manifest.v1",
                "seed": seed,
                "result": "SUCCESS",
                "success_evidence": {
                    "p01_p13_completed": True,
                    "body_collision": False,
                    "wheel_only_climb": False,
                    "duration_s": 12.0,
                },
            },
        )
        lifecycle = _json(
            run_dir / "run_manifest.json", {"lifecycle": "SUCCEEDED", "exit_code": 0}
        )
        worker_result = {
            "schema": (
                "wlr50_clean.live_residual_gate.v1"
                if role == "baseline"
                else "wlr50_clean.ppo_checkpoint_evaluation.v1"
            ),
            "episode_count": 1,
            "success_count": 1,
            "passed": True,
            "episodes": [episode],
        }
        result_name = "acceptance.json" if role == "baseline" else "checkpoint_evaluation.json"
        result = _json(run_dir / result_name, worker_result)
        workers.append(
            {
                "role": role,
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "run_manifest_sha256": artifacts.sha256_file(lifecycle),
                "worker_result": str(result),
                "worker_result_sha256": artifacts.sha256_file(result),
                "worker_gate_passed": True,
                "canonical_episode_dir": str(episode_dir.resolve()),
                "trial_manifest_sha256": artifacts.sha256_file(trial),
            }
        )
        episodes.append(episode)

    payload = {
        "schema": subject.FRESH_PROCESS_BATCH_SCHEMA,
        "role": role,
        "seed_set": seed_set,
        "seeds": list(seeds),
        "canonical_episode_dirs": [row["canonical_episode_dir"] for row in episodes],
        "fresh_process_per_episode": True,
        "deterministic_evaluation": True,
        "deterministic_mean_policy": True if role == "candidate" else None,
        "pure_fsm_zero_residual": True if role == "baseline" else None,
        "episode_count": len(seeds),
        "success_count": len(seeds),
        "body_collision_count": 0,
        "wheel_only_climb_count": 0,
        "safety_abort_count": 0,
        "all_under_maximum_duration": True,
        "passed": True,
        "worker_gate_pass_count": len(seeds),
        "workers": workers,
        "episodes": episodes,
    }
    if checkpoint_path is not None:
        payload.update(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": artifacts.sha256_file(checkpoint_path),
            }
        )
    if checkpoint_manifest is not None:
        payload.update(
            {
                "checkpoint_manifest": str(checkpoint_manifest.resolve()),
                "checkpoint_manifest_sha256": artifacts.sha256_file(checkpoint_manifest),
            }
        )
    if finalized:
        payload.update(
            {
                "finalized": True,
                "frozen_hashes_unchanged": True,
                "hash_gates": {
                    gate: True for gate in checkpoint.REQUIRED_LOCKED_TEST_HASH_GATES
                },
            }
        )
    return _json(tmp_path / name, payload)


def _make_fixture(tmp_path: Path, *, promoted: bool = True) -> dict:
    output = tmp_path / "outputs" / "ppo_phase_v1"
    output.mkdir(parents=True)
    run_dirs, candidate, candidate_manifest = _make_training_runs(tmp_path, output)

    baseline_aggregate = _make_batch(
        tmp_path,
        name="fsm_baseline_evaluation_aggregate.json",
        role="baseline",
        seeds=checkpoint.VALIDATION_SEEDS,
        seed_set="validation",
    )
    validation_aggregate = _make_batch(
        tmp_path,
        name="checkpoint_evaluation_validation.json",
        role="candidate",
        seeds=checkpoint.VALIDATION_SEEDS,
        seed_set="validation",
        checkpoint_path=candidate,
    )

    metrics = output / "metrics"
    metrics.mkdir()
    baseline_episode = metrics / BASELINE_EPISODE_FILENAME
    baseline_phase = metrics / BASELINE_PHASE_FILENAME
    baseline_episode.write_text("seed,task_success\n2001,True\n", encoding="utf-8")
    baseline_phase.write_text("phase,pitch_rate_rms_rad_s\nP01,0.1\n", encoding="utf-8")
    baseline_payload = json.loads(baseline_aggregate.read_text(encoding="utf-8"))
    source_episodes = []
    for seed, directory in zip(
        checkpoint.VALIDATION_SEEDS,
        baseline_payload["canonical_episode_dirs"],
        strict=True,
    ):
        trial = Path(directory) / "trial_manifest.json"
        source_episodes.append(
            {
                "seed": seed,
                "canonical_episode_dir": directory,
                "files": [
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": artifacts.sha256_file(path),
                    }
                    for path in (Path(directory) / name for name in CANONICAL_EPISODE_FILES)
                ],
            }
        )
    baseline_manifest = _json(
        metrics / BASELINE_EVALUATION_MANIFEST_FILENAME,
        {
            "schema": subject.BASELINE_METRIC_MANIFEST_SCHEMA,
            "baseline": "pure_fsm",
            "candidate_required": False,
            "episode_count": 5,
            "validation_seeds": list(checkpoint.VALIDATION_SEEDS),
            "all_p01_p13_complete": True,
            "all_authoritative_success": True,
            "all_zero_residual": True,
            "source_episodes": source_episodes,
            "artifacts": {
                "episode_metrics": _record(baseline_episode),
                "phase_metrics": _record(baseline_phase),
                "manifest": str((metrics / BASELINE_EVALUATION_MANIFEST_FILENAME).resolve()),
            },
        },
    )

    validation_names = (
        BASELINE_EPISODE_FILENAME,
        BASELINE_PHASE_FILENAME,
        CANDIDATE_EPISODE_FILENAME,
        CANDIDATE_PHASE_FILENAME,
        CHECKPOINT_COMPARISON_FILENAME,
        PHASE_COMPARISON_FILENAME,
        RESIDUAL_ACTIVITY_FILENAME,
        REWARD_CONTRIBUTION_FILENAME,
        TERMINATION_SUMMARY_FILENAME,
    )
    for name in validation_names[2:]:
        (metrics / name).write_text("evidence\npass\n", encoding="utf-8")
    checks = {gate: promoted for gate in checkpoint.REQUIRED_PROMOTION_GATES}
    promotion_path = metrics / PROMOTION_DECISION_FILENAME
    promotion_payload = {
        "schema": checkpoint.PROMOTION_DECISION_SCHEMA,
        "baseline_checkpoint": "pure_fsm",
        "candidate_checkpoint": "candidate",
        "candidate_checkpoint_path": str(candidate),
        "candidate_checkpoint_sha256": artifacts.sha256_file(candidate),
        "paired_seeds": list(checkpoint.VALIDATION_SEEDS),
        "paired_episode_count": 5,
        "minimum_paired_seeds": 5,
        "frozen_hashes_unchanged": True,
        "promotion": {
            "promoted": promoted,
            "first_failed_gate": None if promoted else "body_collision_zero",
            "checks": checks,
            "global_stability_improvement_fraction": 0.08,
            "improved_priority_phase_count": 4,
        },
        "first_failed_gate": None if promoted else "body_collision_zero",
        "checks_in_evaluation_order": [
            {"gate": gate, "passed": value} for gate, value in checks.items()
        ],
        "artifacts": {
            "baseline_episode_metrics": str((metrics / BASELINE_EPISODE_FILENAME).resolve()),
            "baseline_phase_metrics": str((metrics / BASELINE_PHASE_FILENAME).resolve()),
            "candidate_episode_metrics": str((metrics / CANDIDATE_EPISODE_FILENAME).resolve()),
            "candidate_phase_metrics": str((metrics / CANDIDATE_PHASE_FILENAME).resolve()),
            "checkpoint_comparison": str((metrics / CHECKPOINT_COMPARISON_FILENAME).resolve()),
            "phase_metric_comparison": str((metrics / PHASE_COMPARISON_FILENAME).resolve()),
            "residual_activity_by_phase": str((metrics / RESIDUAL_ACTIVITY_FILENAME).resolve()),
            "reward_contribution_by_phase": str((metrics / REWARD_CONTRIBUTION_FILENAME).resolve()),
            "termination_summary": str((metrics / TERMINATION_SUMMARY_FILENAME).resolve()),
            "promotion_decision": str(promotion_path.resolve()),
        },
    }
    promotion = _json(promotion_path, promotion_payload)

    checkpoints = output / "checkpoints"
    best = checkpoints / checkpoint.BEST_CHECKPOINT_NAME
    improved = checkpoints / checkpoint.IMPROVED_CHECKPOINT_NAME
    best.write_bytes(candidate.read_bytes())
    improved.write_bytes(candidate.read_bytes())
    trained_contract = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    best_manifest = _json(
        checkpoints / checkpoint.BEST_MANIFEST_NAME,
        {
            **trained_contract,
            "publication_role": "best_validation",
            "validation_promotion_authorized": True,
            "locked_test_authorized": False,
            "promotion_decision": str(promotion),
            "promotion_decision_sha256": artifacts.sha256_file(promotion),
            "checkpoint_path": str(best.resolve()),
            "checkpoint_sha256": artifacts.sha256_file(best),
        },
    )
    validation_promotion = _json(
        output / "manifests" / checkpoint.VALIDATION_PROMOTION_MANIFEST_NAME,
        {
            "schema": checkpoint.CHECKPOINT_VALIDATION_PROMOTION_SCHEMA,
            "valid": True,
            "status": "PROMOTED_VALIDATION",
            "promotion_scope": "best_validation_only",
            "improved_checkpoint_authorized": False,
            "filename_inference_used": False,
            "source_checkpoint": str(candidate),
            "source_checkpoint_sha256": artifacts.sha256_file(candidate),
            "source_manifest": str(candidate_manifest),
            "source_manifest_sha256": artifacts.sha256_file(candidate_manifest),
            "promotion_decision": str(promotion),
            "promotion_decision_sha256": artifacts.sha256_file(promotion),
            "validation_seeds": list(checkpoint.VALIDATION_SEEDS),
            "promotion": promotion_payload["promotion"],
            "published_best_validation": {
                "path": str(best.resolve()),
                "sha256": artifacts.sha256_file(best),
                "manifest": str(best_manifest),
                "manifest_sha256": artifacts.sha256_file(best_manifest),
            },
        },
    )
    locked = _make_batch(
        tmp_path,
        name="checkpoint_evaluation_locked_test.json",
        role="candidate",
        seeds=checkpoint.LOCKED_TEST_SEEDS,
        seed_set="locked-test",
        checkpoint_path=best,
        checkpoint_manifest=best_manifest,
        finalized=True,
    )
    improved_manifest = _json(
        checkpoints / checkpoint.IMPROVED_MANIFEST_NAME,
        {
            **trained_contract,
            "publication_role": "improved",
            "validation_promotion_authorized": True,
            "locked_test_authorized": True,
            "promotion_authorized": True,
            "source_best_validation_checkpoint": str(best.resolve()),
            "source_best_validation_checkpoint_sha256": artifacts.sha256_file(best),
            "source_best_validation_manifest": str(best_manifest),
            "source_best_validation_manifest_sha256": artifacts.sha256_file(best_manifest),
            "promotion_decision": str(promotion),
            "promotion_decision_sha256": artifacts.sha256_file(promotion),
            "validation_promotion_manifest": str(validation_promotion),
            "validation_promotion_manifest_sha256": artifacts.sha256_file(validation_promotion),
            "locked_test_aggregate": str(locked),
            "locked_test_aggregate_sha256": artifacts.sha256_file(locked),
            "checkpoint_path": str(improved.resolve()),
            "checkpoint_sha256": artifacts.sha256_file(improved),
        },
    )
    final_promotion = _json(
        output / "manifests" / checkpoint.PROMOTION_MANIFEST_NAME,
        {
            "schema": checkpoint.CHECKPOINT_IMPROVED_PROMOTION_SCHEMA,
            "valid": True,
            "status": "PROMOTED_IMPROVED",
            "two_stage_promotion": True,
            "validation_decision_alone_cannot_authorize_improved": True,
            "filename_inference_used": False,
            "validation_promotion_manifest": str(validation_promotion),
            "validation_promotion_manifest_sha256": artifacts.sha256_file(validation_promotion),
            "validation_promotion": json.loads(validation_promotion.read_text(encoding="utf-8")),
            "locked_test_aggregate": str(locked),
            "locked_test_aggregate_sha256": artifacts.sha256_file(locked),
            "published_checkpoints": {
                "best_validation": {
                    "path": str(best.resolve()),
                    "sha256": artifacts.sha256_file(best),
                    "manifest": str(best_manifest),
                    "manifest_sha256": artifacts.sha256_file(best_manifest),
                },
                "improved": {
                    "path": str(improved.resolve()),
                    "sha256": artifacts.sha256_file(improved),
                    "manifest": str(improved_manifest),
                    "manifest_sha256": artifacts.sha256_file(improved_manifest),
                },
            },
            "byte_identical_best_and_improved": True,
            "immutable_no_overwrite": True,
        },
    )

    torchscript_actor = checkpoints / checkpoint.TORCHSCRIPT_ACTOR_NAME
    torchscript_actor.write_bytes(b"synthetic-inference-only-torchscript")
    inference_manifest = _json(
        output / "manifests" / checkpoint.INFERENCE_EXPORT_MANIFEST_NAME,
        {
            "schema": checkpoint.INFERENCE_EXPORT_SCHEMA,
            "valid": True,
            "status": "PASS",
            "inference_only": True,
            "deterministic_mean_policy": True,
            "contains_critic": False,
            "contains_optimizer": False,
            "contains_rollout_state": False,
            "contains_stochastic_sampler": False,
            "source_checkpoint": str(improved.resolve()),
            "source_checkpoint_sha256": artifacts.sha256_file(improved),
            "source_manifest": str(improved_manifest),
            "source_manifest_sha256": artifacts.sha256_file(improved_manifest),
            "torchscript": {
                "valid": True,
                "status": "PASS",
                "supported": True,
                "path": str(torchscript_actor.resolve()),
                "sha256": artifacts.sha256_file(torchscript_actor),
                "bytes": torchscript_actor.stat().st_size,
                "reloaded": True,
                "finite": True,
                "deterministic": True,
                "equivalent_to_loaded_runner": True,
            },
            "onnx": {
                "status": "UNSUPPORTED",
                "supported": False,
                "reason": "synthetic test runtime",
                "path": None,
            },
        },
    )

    videos_dir = output / "videos"
    videos_dir.mkdir()
    video_names = {
        "fsm_baseline": FSM_VIDEO_NAME,
        "ppo_improved": PPO_VIDEO_NAME,
        "comparison": COMPARISON_VIDEO_NAME,
        "ppo_diagnostic": DIAGNOSTIC_VIDEO_NAME,
    }
    video_rows = {}
    video_paths = []
    for key, name in video_names.items():
        path = videos_dir / name
        path.write_bytes(f"synthetic-{key}-h264".encode("ascii"))
        video_paths.append(path)
        video_rows[key] = {
            "valid": True,
            "status": "PASS",
            "path": str(path.resolve()),
            "sha256": artifacts.sha256_file(path),
            "bytes": path.stat().st_size,
            "duration_s": 12.0,
            "fps": 15.0,
            "frame_count": 180,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "pix_fmt": "yuv420p",
            "full_decode": True,
            "timestamps_monotonic": True,
            "monotonic": True,
            "stitched": False,
            "speed_modified": False,
            "source_checkpoint_sha256": (
                "not_applicable" if key == "fsm_baseline" else artifacts.sha256_file(improved)
            ),
        }
    diagnostic = output / "manifests" / "ppo_improved_diagnostic.ass"
    diagnostic.parent.mkdir(exist_ok=True)
    diagnostic.write_text("[Script Info]\n", encoding="utf-8")
    source_episode_rows = {}
    for role in ("fsm", "ppo"):
        directory = tmp_path / "video-sources" / role
        directory.mkdir(parents=True)
        source_manifest = _json(directory / "trial_manifest.json", {"role": role})
        ledger = directory / "viewport_capture_ledger.jsonl"
        trace = directory / "policy_trace.jsonl"
        raw_video = directory / "viewport_120hz.mp4"
        ledger.write_text("{}\n", encoding="utf-8")
        trace.write_text("{}\n", encoding="utf-8")
        raw_video.write_bytes(f"raw-{role}-video".encode("ascii"))
        source_episode_rows[role] = {
            "directory": str(directory.resolve()),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": artifacts.sha256_file(source_manifest),
            "viewport_ledger": str(ledger.resolve()),
            "viewport_ledger_sha256": artifacts.sha256_file(ledger),
            "policy_trace": str(trace.resolve()),
            "policy_trace_sha256": artifacts.sha256_file(trace),
            "raw_video": str(raw_video.resolve()),
            "raw_video_sha256": artifacts.sha256_file(raw_video),
        }
    source_episode_rows["ppo"].update(
        {
            "checkpoint": str(improved.resolve()),
            "checkpoint_sha256": artifacts.sha256_file(improved),
        }
    )
    video_checksum = output / "manifests" / VIDEO_CHECKSUM_NAME
    video_validation = _json(
        output / "manifests" / VIDEO_VALIDATION_NAME,
        {
            "schema": subject.FINAL_VIDEO_SCHEMA,
            "valid": True,
            "status": "PASS",
            "immutable_no_overwrite": True,
            "maximum_duration_s": 200.0,
            "pair_evidence": {
                "same_seed": True,
                "same_live_environment_contract": True,
                "same_obstacle_contract": True,
                "same_camera": True,
                "same_resolution": True,
                "same_initial_state": True,
            },
            "source_episodes": source_episode_rows,
            "videos": video_rows,
            "diagnostic_ass": {
                "path": str(diagnostic.resolve()),
                "sha256": artifacts.sha256_file(diagnostic),
            },
            "video_checksum_manifest": str(video_checksum.resolve()),
        },
    )
    artifacts.write_checksum_manifest(
        [*video_paths, video_validation, diagnostic], video_checksum, root=output
    )

    reports_dir = output / "reports"
    plots_dir = output / "plots"
    reports_dir.mkdir()
    plots_dir.mkdir()
    reports = []
    plots = []
    for name in REPORT_FILENAMES:
        path = reports_dir / name
        path.write_text(f"# {name}\n\nEvidence-backed report.\n", encoding="utf-8")
        reports.append(path)
    for name in PLOT_FILENAMES:
        path = plots_dir / name
        path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-plot")
        plots.append(path)

    return {
        "output_root": output,
        "training_run_dirs": run_dirs,
        "baseline_aggregate_path": baseline_aggregate,
        "baseline_metric_paths": [baseline_episode, baseline_phase, baseline_manifest],
        "validation_aggregate_path": validation_aggregate,
        "promotion_decision_path": promotion,
        "locked_test_aggregate_path": locked,
        "checkpoint_manifest_paths": [
            candidate_manifest,
            best_manifest,
            validation_promotion,
            improved_manifest,
            final_promotion,
            inference_manifest,
        ],
        "video_validation_path": video_validation,
        "video_checksum_path": video_checksum,
        "report_paths": reports,
        "plot_paths": plots,
    }


def test_finalization_validates_every_layer_and_is_byte_idempotent(tmp_path: Path) -> None:
    inputs = _make_fixture(tmp_path)
    video_checksum_before = inputs["video_checksum_path"].read_bytes()
    result = subject.finalize_ppo_phase_delivery(**inputs)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (result.training_manifest, result.evaluation_manifest, result.checksums)
    }

    repeated = subject.finalize_ppo_phase_delivery(**inputs)

    assert repeated == result
    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime
    training = json.loads(result.training_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(result.evaluation_manifest.read_text(encoding="utf-8"))
    assert training["status"] == "PASS"
    assert training["stage_sequence"] == list(subject.REQUIRED_TRAINING_STAGES)
    assert evaluation["improvement_claim_authorized"] is True
    assert evaluation["success_inferred_from_filename"] is False
    verification = artifacts.verify_checksum_manifest(
        result.checksums, root=inputs["output_root"]
    )
    assert verification["valid"] is True
    expected_inventory = {
        path.resolve().relative_to(inputs["output_root"].resolve()).as_posix()
        for path in inputs["output_root"].rglob("*")
        if path.is_file() and path.resolve() != result.checksums.resolve()
    }
    assert {row["path"] for row in verification["entries"]} == expected_inventory
    assert inputs["video_checksum_path"].read_bytes() == video_checksum_before
    checksum_text = result.checksums.read_text(encoding="utf-8")
    assert f"manifests/{VIDEO_CHECKSUM_NAME}" in checksum_text
    assert "manifests/training_manifest.json" in checksum_text
    assert "manifests/evaluation_manifest.json" in checksum_text


def test_failed_promotion_content_cannot_be_overridden_by_names(tmp_path: Path) -> None:
    inputs = _make_fixture(tmp_path, promoted=False)
    manifests = inputs["output_root"] / "manifests"

    with pytest.raises(subject.FinalizationError, match="did not pass every"):
        subject.finalize_ppo_phase_delivery(**inputs)

    assert not (manifests / "training_manifest.json").exists()
    assert not (manifests / "evaluation_manifest.json").exists()
    assert not (manifests / "checksums.sha256").exists()


def test_video_tampering_is_detected_before_any_final_output(tmp_path: Path) -> None:
    inputs = _make_fixture(tmp_path)
    video = inputs["output_root"] / "videos" / PPO_VIDEO_NAME
    video.write_bytes(b"tampered")

    with pytest.raises(subject.FinalizationError, match="SHA-256 mismatch"):
        subject.finalize_ppo_phase_delivery(**inputs)

    manifests = inputs["output_root"] / "manifests"
    assert not (manifests / "training_manifest.json").exists()
    assert not (manifests / "evaluation_manifest.json").exists()
    assert not (manifests / "checksums.sha256").exists()


def test_conflicting_final_manifest_preflights_without_partial_write(tmp_path: Path) -> None:
    inputs = _make_fixture(tmp_path)
    conflict = inputs["output_root"] / "manifests" / "evaluation_manifest.json"
    conflict.write_bytes(b"prior immutable evidence")

    with pytest.raises(subject.FinalizationError, match="refusing to overwrite"):
        subject.finalize_ppo_phase_delivery(**inputs)

    assert conflict.read_bytes() == b"prior immutable evidence"
    assert not (conflict.parent / "training_manifest.json").exists()
    assert not (conflict.parent / "checksums.sha256").exists()


def test_publication_failure_rolls_back_files_created_by_this_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _make_fixture(tmp_path)
    real_atomic = subject._atomic_bytes
    calls = 0

    def fail_second(path: Path, payload: bytes, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise artifacts.ArtifactError("injected publication failure")
        return real_atomic(path, payload, **kwargs)

    monkeypatch.setattr(subject, "_atomic_bytes", fail_second)
    with pytest.raises(subject.FinalizationError, match="injected"):
        subject.finalize_ppo_phase_delivery(**inputs)

    manifests = inputs["output_root"] / "manifests"
    assert not (manifests / "training_manifest.json").exists()
    assert not (manifests / "evaluation_manifest.json").exists()
    assert not (manifests / "checksums.sha256").exists()
