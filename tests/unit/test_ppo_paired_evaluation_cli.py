from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import artifacts, checkpoint_promotion, cli, evaluation_artifacts


def _checkpoint_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"paired evaluation checkpoint")
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
    manifest.write_text(
        json.dumps(
            {
                "schema": checkpoint_promotion.CHECKPOINT_MANIFEST_SCHEMA,
                "stage": "full-episode",
                "training_seed": 1001,
                "global_policy_decisions": 100_000,
                "actor_observation_dimension": 125,
                "critic_observation_dimension": 125,
                "residual_dimension": 12,
                "physics_hz": 120.0,
                "decision_hz": 15.0,
                **{field: cli._sha256(path) for field, path in hash_paths.items()},
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
    return SimpleNamespace(
        seed_set="validation",
        seed=2001,
        episode_count=5,
        baseline_episode_dir=list(baseline_dirs),
        candidate_episode_dir=list(candidate_dirs),
        candidate_checkpoint=checkpoint,
        candidate_manifest=manifest,
        metrics_output_dir=tmp_path / "explicit-metrics",
    )


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

    def fake_evaluate(directories, *, seeds, residual_calibration):
        evaluation_calls.append(
            (tuple(directories), tuple(seeds), residual_calibration)
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
        (baseline_dirs, tuple(range(2001, 2006)), calibration),
        (candidate_dirs, tuple(range(2001, 2006)), calibration),
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
    assert export_kwargs["frozen_hashes_unchanged"] is True
    assert export_kwargs["residual_calibration_evidence"] is calibration
    result = json.loads(capsys.readouterr().out)
    assert result["offline"] is True
    assert result["validation_seeds"] == list(range(2001, 2006))
    assert result["artifacts"] == expected_artifacts
    assert not args.metrics_output_dir.exists()


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
        match="checkpoint bytes do not match",
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
