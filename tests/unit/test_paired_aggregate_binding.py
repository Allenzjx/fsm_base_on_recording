from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wlr50_clean.ppo import artifacts
from wlr50_clean.ppo import paired_aggregate_binding as subject
from wlr50_clean.ppo import training_orchestration
from wlr50_clean.ppo.evaluation_artifacts import (
    CANONICAL_EPISODE_FILES,
    FreshProcessEpisodeBatch,
)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _json(path: Path, value: object) -> Path:
    return _write(
        path,
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _record(path: Path, relative: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _finalize_run(
    root: Path,
    run_dir: Path,
    *,
    run_kind: str,
    stage: str,
    subcommand: str,
    artifacts: tuple[str, ...],
) -> None:
    identity = {
        "seed": 2001,
        "environment_count": 1,
        "git_commit": "a" * 40,
        "training_stage": stage,
    }
    started = {
        "schema": subject.RUN_SCHEMA,
        "lifecycle": "STARTED",
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "project_root": str(root.resolve()),
        "run_kind": run_kind,
        "entrypoint": "wlr50_clean.ppo.cli",
        "subcommand": subcommand,
        "identity": identity,
    }
    started_path = _json(run_dir / "run_manifest.started.json", started)
    stdout = _write(run_dir / "stdout.log", b"")
    stderr = _write(run_dir / "stderr.log", b"")
    final = {
        **started,
        "lifecycle": "SUCCEEDED",
        "exit_code": 0,
        "immutable_run_directory": True,
        "started_manifest": _record(started_path, "run_manifest.started.json"),
        "logs": {
            "stdout.log": _record(stdout, "stdout.log"),
            "stderr.log": _record(stderr, "stderr.log"),
        },
        "artifacts": {
            name: _record(run_dir / name, name) for name in artifacts
        },
    }
    _json(run_dir / "run_manifest.json", final)


def _fixture(
    tmp_path: Path, *, passed: bool = True
) -> tuple[Path, Path, Path, FreshProcessEpisodeBatch, Path]:
    root = tmp_path.resolve()
    checkpoint = _write(root / "outputs" / "candidate.pt", b"checkpoint")
    checkpoint_manifest = _json(root / "outputs" / "candidate_manifest.json", {})
    worker_rows = []
    episode_rows = []
    episode_dirs = []
    for seed in subject.VALIDATION_SEEDS:
        run_dir = (
            root
            / "runs"
            / "ppo_phase_v1"
            / "validation-checkpoint-evaluation"
            / f"worker-{seed}"
        )
        episode_dir = run_dir / f"episode_000_seed_{seed}"
        for name in CANONICAL_EPISODE_FILES:
            _write(episode_dir / name, b"" if name == "state_transitions.jsonl" else b"{}\n")
        result = _json(
            run_dir / "checkpoint_evaluation.json",
            {
                "checkpoint_manifest": str(checkpoint_manifest.resolve()),
                "checkpoint_manifest_sha256": hashlib.sha256(
                    checkpoint_manifest.read_bytes()
                ).hexdigest(),
            },
        )
        runtime = _json(
            run_dir / "committed_runtime_identity.before.json",
            {"git_commit": "a" * 40},
        )
        _write(run_dir / "committed_runtime_identity.after.json", runtime.read_bytes())
        before = _json(run_dir / "frozen_hashes.before.json", {"passed": True, "mismatches": []})
        _write(run_dir / "frozen_hashes.after.json", before.read_bytes())
        artifact_names = (
            "checkpoint_evaluation.json",
            "committed_runtime_identity.before.json",
            "committed_runtime_identity.after.json",
            "frozen_hashes.before.json",
            "frozen_hashes.after.json",
            *(f"{episode_dir.name}/{name}" for name in CANONICAL_EPISODE_FILES),
        )
        _finalize_run(
            root,
            run_dir,
            run_kind="validation-checkpoint-evaluation",
            stage="checkpoint-evaluation-validation-fresh-process",
            subcommand="evaluate",
            artifacts=artifact_names,
        )
        run_manifest = run_dir / "run_manifest.json"
        row = {
            "role": "candidate",
            "seed": seed,
            "run_dir": str(run_dir.resolve()),
            "canonical_episode_dir": str(episode_dir.resolve()),
            "run_manifest_sha256": hashlib.sha256(
                run_manifest.read_bytes()
            ).hexdigest(),
            "worker_result": str(result.resolve()),
            "worker_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
            "trial_manifest_sha256": hashlib.sha256(
                (episode_dir / "trial_manifest.json").read_bytes()
            ).hexdigest(),
            "worker_gate_passed": passed,
        }
        episode = {
            "seed": seed,
            "task_success": passed,
            "body_collision": False,
            "wheel_only_climb": False,
            "safety_abort": False,
            "under_maximum_duration": True,
            "recording_runtime_access_count": 0,
            "in_episode_root_write_count": 0,
        }
        worker_rows.append(row)
        episode_rows.append(episode)
        episode_dirs.append(episode_dir.resolve())
    batch = FreshProcessEpisodeBatch(
        role="candidate",
        seeds=subject.VALIDATION_SEEDS,
        canonical_episode_dirs=tuple(episode_dirs),
        episode_rows=tuple(episode_rows),
        worker_rows=tuple(worker_rows),
    )
    aggregate_dir = (
        root
        / "runs"
        / "ppo_phase_v1"
        / "validation-checkpoint-evaluation-batch"
        / "aggregate"
    )
    aggregate = _json(
        aggregate_dir / "checkpoint_evaluation_aggregate.json",
        {
            "schema": subject.AGGREGATE_SCHEMA,
            "role": "candidate",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed_set": "validation",
            "seeds": list(subject.VALIDATION_SEEDS),
            "canonical_episode_dirs": [str(value) for value in episode_dirs],
            "fresh_process_per_episode": True,
            "deterministic_evaluation": True,
            "deterministic_mean_policy": True,
            "episode_count": 5,
            "success_count": 5 if passed else 0,
            "body_collision_count": 0,
            "wheel_only_climb_count": 0,
            "safety_abort_count": 0,
            "all_under_maximum_duration": True,
            "passed": passed,
            "worker_gate_pass_count": 5 if passed else 0,
            "workers": worker_rows,
            "episodes": episode_rows,
        },
    )
    runtime = _json(
        aggregate_dir / "committed_runtime_identity.before.json",
        {"git_commit": "a" * 40},
    )
    _write(aggregate_dir / "committed_runtime_identity.after.json", runtime.read_bytes())
    frozen = _json(
        aggregate_dir / "frozen_hashes.before.json", {"passed": True, "mismatches": []}
    )
    _write(aggregate_dir / "frozen_hashes.after.json", frozen.read_bytes())
    _finalize_run(
        root,
        aggregate_dir,
        run_kind="validation-checkpoint-evaluation-batch",
        stage="checkpoint-evaluation-validation-aggregate",
        subcommand="aggregate-evaluations",
        artifacts=(
            aggregate.name,
            "committed_runtime_identity.before.json",
            "committed_runtime_identity.after.json",
            "frozen_hashes.before.json",
            "frozen_hashes.after.json",
        ),
    )
    return aggregate, checkpoint, checkpoint_manifest, batch, episode_dirs[0] / "observation_120hz.jsonl"


def _trust_central_run_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        training_orchestration, "_validate_finalized_run", lambda *args, **kwargs: {}
    )


def test_baseline_worker_consumer_matches_real_reserve_run_canonical_kind(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")
    reservation = artifacts.reserve_run(
        project_root=tmp_path,
        run_kind="baseline_fsm_eval",
        config_paths=(config,),
        seed=2001,
        environment_count=1,
        training_stage="baseline-fsm-eval-fresh-process",
        git_commit="a" * 40,
    )
    started = json.loads(reservation.started_manifest.read_text(encoding="utf-8"))

    assert subject.BASELINE_WORKER_RUN_KIND == "baseline-fsm-eval"
    assert reservation.run_dir.parent.name == subject.BASELINE_WORKER_RUN_KIND
    assert started["run_kind"] == subject.BASELINE_WORKER_RUN_KIND


def test_candidate_failure_is_complete_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregate, checkpoint, manifest, batch, _ = _fixture(tmp_path, passed=False)
    _trust_central_run_validator(monkeypatch)
    monkeypatch.setattr(subject, "collect_fresh_process_episode_workers", lambda *a, **k: batch)
    captured = subject.capture_validation_aggregate(
        aggregate,
        role="candidate",
        expected_checkpoint_path=checkpoint,
        expected_checkpoint_manifest_path=manifest,
        project_root=tmp_path,
    )
    record = captured.as_record()
    assert record["physical_passed"] is False
    assert record["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record["source_file_records"] == sorted(
        record["source_file_records"], key=lambda row: row["path"]
    )


def test_mutation_during_worker_reconstruction_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregate, checkpoint, manifest, batch, mutable = _fixture(tmp_path)
    _trust_central_run_validator(monkeypatch)

    def mutate(*args, **kwargs):
        mutable.write_bytes(b"changed\n")
        return batch

    monkeypatch.setattr(subject, "collect_fresh_process_episode_workers", mutate)
    with pytest.raises(subject.PairedAggregateBindingError, match="during reconstruction"):
        subject.capture_validation_aggregate(
            aggregate,
            role="candidate",
            expected_checkpoint_path=checkpoint,
            expected_checkpoint_manifest_path=manifest,
            project_root=tmp_path,
        )
