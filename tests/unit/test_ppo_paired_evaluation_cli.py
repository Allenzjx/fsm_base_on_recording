from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import (
    artifacts,
    checkpoint_promotion,
    cli,
    evaluation_artifacts,
    paired_aggregate_binding,
)
from wlr50_clean.ppo.phase_snapshots import (
    phase_snapshot_bundle_file_hashes,
    validated_phase_snapshot_bundle_record,
)


def _checkpoint_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "candidate.pt"
    manifest = tmp_path / "candidate_manifest.json"
    hash_paths = {
        "controller_hash": cli.PROJECT_ROOT / "configs" / "fsm_states.yaml",
        "environment_hash": cli.PROJECT_ROOT / "configs" / "environment_lock.json",
        "observation_schema_hash": (
            cli.PROJECT_ROOT / "configs" / "ppo_observation_schema_v2.json"
        ),
        "action_schema_hash": (
            cli.PROJECT_ROOT / "configs" / "ppo_phase_action_masks_v2.yaml"
        ),
        "reward_config_hash": cli.PROJECT_ROOT / "configs" / "ppo_reward_v2.yaml",
    }
    snapshot_bundle = validated_phase_snapshot_bundle_record(
        cli.DEFAULT_PHASE_SNAPSHOT_ROOT
    )
    infos = {
        "schema": checkpoint_promotion.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "full-episode",
        "training_seed": 1001,
        "global_policy_decisions": 100_000,
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "files": phase_snapshot_bundle_file_hashes(snapshot_bundle),
        **{field: cli._sha256(path) for field, path in hash_paths.items()},
        "phase_snapshot_manifest": snapshot_bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": snapshot_bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": snapshot_bundle["bundle_sha256"],
        "phase_snapshot_bundle": snapshot_bundle,
    }
    torch.save({"infos": infos}, checkpoint)
    manifest.write_text(
        json.dumps(
            {
                **infos,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": cli._sha256(checkpoint),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, manifest


def _episode_dirs(tmp_path: Path, role: str) -> tuple[Path, ...]:
    result = []
    for seed in range(2001, 2006):
        directory = tmp_path / role / f"episode_seed_{seed}"
        directory.mkdir(parents=True)
        result.append(directory)
    return tuple(result)


def _args(
    tmp_path: Path,
    *,
    baseline_dirs: tuple[Path, ...],
    candidate_dirs: tuple[Path, ...],
    checkpoint: Path,
    manifest: Path,
) -> SimpleNamespace:
    baseline_aggregate = tmp_path / "aggregates" / "baseline" / "baseline_aggregate.json"
    candidate_aggregate = (
        tmp_path / "aggregates" / "candidate" / "candidate_aggregate.json"
    )
    baseline_aggregate.parent.mkdir(parents=True)
    candidate_aggregate.parent.mkdir(parents=True)
    baseline_aggregate.write_text("{}\n", encoding="utf-8")
    candidate_aggregate.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        seed_set="validation",
        seed=2001,
        episode_count=5,
        baseline_episode_dir=list(baseline_dirs),
        candidate_episode_dir=list(candidate_dirs),
        candidate_checkpoint=checkpoint,
        candidate_manifest=manifest,
        baseline_aggregate=baseline_aggregate,
        candidate_validation_aggregate=candidate_aggregate,
        metrics_output_dir=tmp_path / "explicit-metrics",
    )


class _FakeAggregateCapture:
    def __init__(
        self,
        *,
        role: str,
        path: Path,
        directories: tuple[Path, ...],
        checkpoint: Path | None = None,
        manifest: Path | None = None,
    ) -> None:
        self.role = role
        self.aggregate_path = path.resolve()
        self.batch = SimpleNamespace(
            canonical_episode_dirs=directories,
            worker_rows=tuple({"run_dir": str(value.parent)} for value in directories),
        )
        self._checkpoint = checkpoint
        self._manifest = manifest
        self.unchanged_calls = 0

    def assert_unchanged(self) -> None:
        self.unchanged_calls += 1

    def as_record(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": paired_aggregate_binding.SCHEMA,
            "role": self.role,
            "path": str(self.aggregate_path),
        }
        if self._checkpoint is not None:
            result.update(
                checkpoint_path=str(self._checkpoint.resolve()),
                checkpoint_manifest_path=str(self._manifest.resolve()),
            )
        return result


def _patch_aggregate_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: SimpleNamespace,
    baseline_dirs: tuple[Path, ...],
    candidate_dirs: tuple[Path, ...],
) -> tuple[_FakeAggregateCapture, _FakeAggregateCapture]:
    baseline = _FakeAggregateCapture(
        role="baseline",
        path=args.baseline_aggregate,
        directories=baseline_dirs,
    )
    candidate = _FakeAggregateCapture(
        role="candidate",
        path=args.candidate_validation_aggregate,
        directories=candidate_dirs,
        checkpoint=args.candidate_checkpoint,
        manifest=args.candidate_manifest,
    )

    def capture(path, *, role, **kwargs):
        selected = baseline if role == "baseline" else candidate
        assert Path(path).resolve() == selected.aggregate_path
        return selected

    monkeypatch.setattr(paired_aggregate_binding, "capture_validation_aggregate", capture)
    return baseline, candidate


def _passing_frozen_audit() -> dict[str, object]:
    return {
        "passed": True,
        "mismatches": [],
        "frozen_manifest_sha256": "f" * 64,
        "entries": [
            {
                "path": "configs/fsm_states.yaml",
                "expected_sha256": "a" * 64,
                "actual_sha256": "a" * 64,
                "exists": True,
                "valid": True,
            }
        ],
    }


def test_paired_export_is_offline_and_wires_exact_role_separated_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "export-paired-evaluation" not in cli.LIVE_COMMANDS
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    baseline_dirs = _episode_dirs(tmp_path, "baseline")
    candidate_dirs = _episode_dirs(tmp_path, "candidate")
    args = _args(
        tmp_path,
        baseline_dirs=baseline_dirs,
        candidate_dirs=candidate_dirs,
        checkpoint=checkpoint,
        manifest=manifest,
    )
    baseline_capture, candidate_capture = _patch_aggregate_capture(
        monkeypatch,
        args=args,
        baseline_dirs=baseline_dirs,
        candidate_dirs=candidate_dirs,
    )
    audit_calls = []
    monkeypatch.setattr(
        artifacts,
        "verify_frozen_hashes",
        lambda **kwargs: audit_calls.append(kwargs) or _passing_frozen_audit(),
    )
    calibration = object()
    monkeypatch.setattr(
        evaluation_artifacts,
        "build_versioned_residual_activity_calibration",
        lambda: calibration,
    )
    evaluation_calls = []

    def fake_evaluate(
        directories,
        *,
        seeds,
        residual_calibration,
        require_complete_phase_sequence=True,
    ):
        evaluation_calls.append(
            (
                tuple(directories),
                tuple(seeds),
                residual_calibration,
                require_complete_phase_sequence,
            )
        )
        return tuple(f"run-{len(evaluation_calls)}-{seed}" for seed in seeds)

    monkeypatch.setattr(
        evaluation_artifacts,
        "evaluate_canonical_episode_dirs",
        fake_evaluate,
    )
    export_calls = []
    expected_artifacts = {
        "promotion_decision": str(args.metrics_output_dir / "promotion_decision.json")
    }

    def fake_export(output, **kwargs):
        export_calls.append((Path(output), kwargs))
        return SimpleNamespace(as_dict=lambda: expected_artifacts)

    monkeypatch.setattr(
        evaluation_artifacts,
        "export_paired_evaluation_artifacts",
        fake_export,
    )

    assert cli._export_paired_evaluation(args) == 0
    assert len(audit_calls) == 2
    assert evaluation_calls == [
        (baseline_dirs, tuple(range(2001, 2006)), calibration, True),
        (candidate_dirs, tuple(range(2001, 2006)), calibration, False),
    ]
    assert len(export_calls) == 1
    output, export_kwargs = export_calls[0]
    assert output == args.metrics_output_dir.resolve()
    assert export_kwargs["baseline_runs"] == tuple(
        f"run-1-{seed}" for seed in range(2001, 2006)
    )
    assert export_kwargs["candidate_runs"] == tuple(
        f"run-2-{seed}" for seed in range(2001, 2006)
    )
    assert export_kwargs["candidate_checkpoint_path"] == checkpoint.resolve()
    assert export_kwargs["candidate_checkpoint_name"] == checkpoint.stem
    assert export_kwargs["baseline_evaluation_aggregate"] == baseline_capture.as_record()
    assert (
        export_kwargs["candidate_validation_aggregate"]
        == candidate_capture.as_record()
    )
    assert export_kwargs["frozen_hashes_unchanged"] is True
    assert export_kwargs["residual_calibration_evidence"] is calibration
    result = json.loads(capsys.readouterr().out)
    assert result["offline"] is True
    assert result["validation_seeds"] == list(range(2001, 2006))
    assert result["artifacts"] == expected_artifacts
    assert not args.metrics_output_dir.exists()
    assert baseline_capture.unchanged_calls == 4
    assert candidate_capture.unchanged_calls == 4


@pytest.mark.parametrize(
    ("seed_set", "seed", "episode_count"),
    (("locked-test", 2001, 5), ("validation", 3001, 5), ("validation", 2001, 4)),
)
def test_paired_export_rejects_noncanonical_validation_seed_contract(
    tmp_path: Path, seed_set: str, seed: int, episode_count: int
) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    args = _args(
        tmp_path,
        baseline_dirs=_episode_dirs(tmp_path, "baseline"),
        candidate_dirs=_episode_dirs(tmp_path, "candidate"),
        checkpoint=checkpoint,
        manifest=manifest,
    )
    args.seed_set = seed_set
    args.seed = seed
    args.episode_count = episode_count

    with pytest.raises(cli.CliError, match="validation seeds 2001-2005"):
        cli._export_paired_evaluation(args)
    assert not args.metrics_output_dir.exists()


def test_paired_export_requires_exactly_five_directories_per_role(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    baseline_dirs = _episode_dirs(tmp_path, "baseline")
    args = _args(
        tmp_path,
        baseline_dirs=baseline_dirs[:-1],
        candidate_dirs=_episode_dirs(tmp_path, "candidate"),
        checkpoint=checkpoint,
        manifest=manifest,
    )

    with pytest.raises(cli.CliError, match="exactly five baseline and five candidate"):
        cli._export_paired_evaluation(args)
    assert not args.metrics_output_dir.exists()


def test_paired_export_binds_checkpoint_bytes_to_sidecar_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    checkpoint.write_bytes(b"changed after sidecar publication")
    args = _args(
        tmp_path,
        baseline_dirs=_episode_dirs(tmp_path, "baseline"),
        candidate_dirs=_episode_dirs(tmp_path, "candidate"),
        checkpoint=checkpoint,
        manifest=manifest,
    )
    monkeypatch.setattr(
        artifacts,
        "verify_frozen_hashes",
        lambda **kwargs: pytest.fail("frozen audit must follow checkpoint validation"),
    )
    monkeypatch.setattr(
        evaluation_artifacts,
        "evaluate_canonical_episode_dirs",
        lambda *args, **kwargs: pytest.fail("episodes must not be evaluated"),
    )

    with pytest.raises(
        checkpoint_promotion.CheckpointPromotionError,
        match="checkpoint (cannot be safely decoded|bytes do not match)",
    ):
        cli._export_paired_evaluation(args)
    assert not args.metrics_output_dir.exists()


def test_paired_export_requires_current_frozen_hash_pass_before_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint, manifest = _checkpoint_and_manifest(tmp_path)
    args = _args(
        tmp_path,
        baseline_dirs=_episode_dirs(tmp_path, "baseline"),
        candidate_dirs=_episode_dirs(tmp_path, "candidate"),
        checkpoint=checkpoint,
        manifest=manifest,
    )
    _patch_aggregate_capture(
        monkeypatch,
        args=args,
        baseline_dirs=tuple(args.baseline_episode_dir),
        candidate_dirs=tuple(args.candidate_episode_dir),
    )
    monkeypatch.setattr(
        artifacts,
        "verify_frozen_hashes",
        lambda **kwargs: {
            **_passing_frozen_audit(),
            "passed": False,
            "mismatches": ["configs/fsm_states.yaml"],
        },
    )
    monkeypatch.setattr(
        evaluation_artifacts,
        "evaluate_canonical_episode_dirs",
        lambda *args, **kwargs: pytest.fail("episodes must not be evaluated"),
    )

    with pytest.raises(cli.CliError, match="frozen FSM hash audit did not pass"):
        cli._export_paired_evaluation(args)
    assert not args.metrics_output_dir.exists()
