from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli, rl_library_wrapper


@pytest.mark.parametrize(
    ("requested_stage", "resume_stage"),
    [
        ("smoke", "initial_zero_residual"),
        ("smoke", "smoke"),
        ("phase-curriculum", "smoke"),
        ("phase-curriculum", "phase-curriculum"),
        ("full-episode", "phase-curriculum"),
        ("full-episode", "full-episode"),
    ],
)
def test_training_resume_stage_chain_accepts_only_adjacent_or_same_stage(
    requested_stage: str, resume_stage: str
) -> None:
    result = cli._validate_training_resume_stage(
        SimpleNamespace(stage=resume_stage), requested_stage=requested_stage
    )

    assert result == {
        "requested_stage": requested_stage,
        "resume_stage": resume_stage,
        "stage_chain_valid": True,
    }


@pytest.mark.parametrize(
    ("requested_stage", "resume_stage"),
    [
        ("phase-curriculum", "initial_zero_residual"),
        ("full-episode", "smoke"),
        ("smoke", "full-episode"),
        ("mild-randomization", "phase-curriculum"),
    ],
)
def test_training_resume_stage_chain_rejects_skips_and_reversal(
    requested_stage: str, resume_stage: str
) -> None:
    with pytest.raises(cli.CliError, match="cannot resume"):
        cli._validate_training_resume_stage(
            SimpleNamespace(stage=resume_stage), requested_stage=requested_stage
        )


def test_mild_randomization_rejects_unpromoted_full_episode_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ordinary_full.pt"
    checkpoint.write_bytes(b"ordinary-full")
    manifest = tmp_path / "ordinary_full_manifest.json"
    manifest.write_text(
        json.dumps({"stage": "full-episode"}, sort_keys=True), encoding="utf-8"
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    provenance = SimpleNamespace(
        stage="full-episode",
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        manifest_path=manifest,
        manifest_sha256=manifest_hash,
    )

    with pytest.raises(cli.CliError, match="promoted improved"):
        cli._validate_training_resume_stage(
            provenance, requested_stage="mild-randomization"
        )


def test_train_validates_loaded_resume_provenance_before_learning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "outputs"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"checkpoint")
    learned: list[bool] = []
    runner = SimpleNamespace(
        learn=lambda **kwargs: learned.append(True),
    )
    env = SimpleNamespace(num_envs=1)
    profile = SimpleNamespace(
        budgets={"smoke": 16, "phase_curriculum": 16, "full_episode": 3200},
        rollout_length=8,
        benchmark_env_counts=(1,),
        decision_hz=15.0,
        timeout_s=200.0,
        deterministic_validation_interval=16,
    )

    monkeypatch.setattr(cli, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: (profile, env, runner, {}),
    )
    monkeypatch.setattr(
        cli,
        "_checkpoint_manifest_payload",
        lambda *args, **kwargs: {"schema": rl_library_wrapper.CHECKPOINT_MANIFEST_SCHEMA},
    )
    monkeypatch.setattr(rl_library_wrapper, "load_training_profile", lambda path: profile)
    monkeypatch.setattr(
        rl_library_wrapper,
        "initialize_zero_mean_actor",
        lambda loaded_runner: None,
    )

    def save_checkpoint(loaded_runner, path, *, manifest):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"initial")
        sidecar = target.with_name(target.stem + "_manifest.json")
        sidecar.write_text("{}", encoding="utf-8")
        return target, sidecar

    monkeypatch.setattr(rl_library_wrapper, "save_checkpoint_with_manifest", save_checkpoint)
    loaded_infos = {"global_policy_decisions": 99}
    runtime_capture = object()
    monkeypatch.setattr(
        cli,
        "_pin_live_checkpoint",
        lambda *args, **kwargs: runtime_capture,
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_checkpoint_round_trip",
        lambda loaded_runner, path, *, captured_bundle: (
            loaded_infos
            if captured_bundle is runtime_capture
            else pytest.fail("trainer did not use its pinned checkpoint capture")
        ),
    )

    seen: list[tuple[Path, object]] = []

    def reject_unbound_checkpoint(path, infos, **kwargs):
        seen.append((Path(path), infos))
        raise rl_library_wrapper.RlLibraryConfigurationError("unbound resume evidence")

    monkeypatch.setattr(
        rl_library_wrapper,
        "validate_resume_checkpoint_provenance",
        reject_unbound_checkpoint,
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "seed_training_rngs",
        lambda seed: {"seed": seed},
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "optimizer_learning_rate",
        lambda loaded_runner: 3.0e-4,
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "capture_training_rng_state",
        lambda *, seed: {"schema": "test", "seed": seed},
    )
    args = SimpleNamespace(
        seed=1001,
        num_envs=1,
        training_config=tmp_path / "training.yaml",
        interface_config=tmp_path / "interface.yaml",
        policy_decisions=16,
        stage="smoke",
        checkpoint=resume,
        checkpoint_manifest=None,
        run_dir=run_dir,
        _soft_reset_acceptance_evidence={"valid": True},
    )

    with pytest.raises(
        rl_library_wrapper.RlLibraryConfigurationError,
        match="unbound resume evidence",
    ):
        cli._train(args, object())

    assert seen == [(resume.resolve(), loaded_infos)]
    assert learned == []


def test_non_smoke_training_requires_explicit_checkpoint_before_runner_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed: list[bool] = []
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: constructed.append(True),
    )
    args = SimpleNamespace(
        stage="full-episode",
        checkpoint=None,
        checkpoint_manifest=None,
    )

    with pytest.raises(cli.CliError, match="requires an explicit --checkpoint"):
        cli._train(args, object())

    assert constructed == []
