from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo.reward_v2 import load_reward_v2_config
from wlr50_clean.ppo.rl_library_wrapper import (
    CHECKPOINT_RUNTIME_CONTRACT_FIELDS,
    CHECKPOINT_MANIFEST_SCHEMA,
    DEFAULT_TRAINING_PATH,
    RlLibraryConfigurationError,
    assert_supported_rsl_runtime,
    build_rsl_runner_config,
    capture_training_rng_state,
    entropy_coefficient_at_policy_decision,
    initialize_zero_mean_actor,
    iterations_for_policy_decisions,
    learn_with_entropy_schedule,
    linear_entropy_schedule,
    load_checkpoint_round_trip,
    load_training_profile,
    planned_entropy_anneal_policy_decisions,
    restore_training_rng_state,
    save_checkpoint_with_manifest,
    seed_training_rngs,
    sha256_file,
    synchronize_loaded_optimizer_learning_rate,
    validate_resume_checkpoint_provenance,
    zero_mean_actor_output_layer_verified,
)


def _resume_checkpoint(tmp_path, *, step=12_345):
    checkpoint = tmp_path / "model_12.pt"
    checkpoint.write_bytes(b"deterministic fake RSL checkpoint bytes")
    snapshot_bundle = {
        "schema": "wlr50_clean.ppo_phase_snapshot_bundle_record.v1",
        "snapshot_root": str((tmp_path / "snapshots").resolve()),
        "manifest_path": str((tmp_path / "snapshots" / "manifest.json").resolve()),
        "manifest_sha256": "7" * 64,
        "phase_count": 13,
        "snapshots": [
            {
                "phase": f"P{index:02d}",
                "snapshot_path": str(
                    (tmp_path / "snapshots" / f"P{index:02d}" / "snapshot.json").resolve()
                ),
                "checksum_path": str(
                    (tmp_path / "snapshots" / f"P{index:02d}" / "snapshot.sha256").resolve()
                ),
                "file_sha256": f"{index:064x}",
                "state_sha256": f"{index + 13:064x}",
                "checksum_file_sha256": f"{index + 26:064x}",
            }
            for index in range(1, 14)
        ],
        "bundle_sha256": "8" * 64,
        "source_trial": "success",
    }
    infos = {
        "schema": "wlr50_clean.phase_residual_checkpoint_manifest.v1",
        "stage": "full_episode_100k",
        "training_seed": 1001,
        "global_policy_decisions": step,
        "source_git_commit": "a" * 40,
        "committed_runtime_content_sha256": "b" * 64,
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "files": {"training.yaml": "1" * 64},
        "controller_hash": "2" * 64,
        "environment_hash": "3" * 64,
        "observation_schema_hash": "4" * 64,
        "action_schema_hash": "5" * 64,
        "reward_config_hash": "6" * 64,
        "phase_snapshot_manifest": snapshot_bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": snapshot_bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": snapshot_bundle["bundle_sha256"],
        "phase_snapshot_bundle": snapshot_bundle,
    }
    manifest = {
        **infos,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return checkpoint, infos, sidecar, manifest


def test_profile_has_disjoint_splits_and_locked_timing():
    profile = load_training_profile()
    assert profile.physics_hz == 120.0
    assert profile.decision_hz == 15.0
    assert profile.ticks_per_decision == 8
    assert set(profile.seed_train).isdisjoint(profile.seed_validation)
    assert set(profile.seed_train).isdisjoint(profile.seed_locked_test)
    assert sum(profile.phase_sampling.values()) == pytest.approx(1.0)
    assert profile.phase_curriculum_max_decisions == 64
    assert profile.phase_curriculum_reset_cycle_samples == 128
    assert profile.phase_curriculum_occupancy_tolerance == pytest.approx(0.02)
    assert tuple(profile.phase_curriculum_baseline_decisions) == tuple(
        f"P{index:02d}" for index in range(1, 14)
    )
    assert profile.entropy_start == pytest.approx(0.005)
    assert profile.entropy_end == pytest.approx(0.001)
    assert profile.deterministic_validation_interval == 10_000
    assert profile.early_stop_when_promotion_gate_passes is True
    assert set(profile.budgets) == {
        "smoke",
        "phase_curriculum",
        "full_episode",
        "mild_randomization",
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("deterministic_validation_interval", 0, "positive integer"),
        ("deterministic_validation_interval", True, "positive integer"),
        ("early_stop_when_promotion_gate_passes", 1, "strict boolean"),
        ("smoke", True, "strict integers"),
    ),
)
def test_profile_rejects_non_strict_budget_controls(
    tmp_path, key, value, message
) -> None:
    payload = yaml.safe_load(DEFAULT_TRAINING_PATH.read_text(encoding="utf-8"))
    payload["budgets_policy_decisions"][key] = value
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RlLibraryConfigurationError, match=message):
        load_training_profile(path)


def test_profile_rejects_extra_or_missing_budget_keys(tmp_path) -> None:
    payload = yaml.safe_load(DEFAULT_TRAINING_PATH.read_text(encoding="utf-8"))
    payload["budgets_policy_decisions"]["legacy_dead_field"] = 123
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RlLibraryConfigurationError, match="exactly four stages"):
        load_training_profile(path)


def test_reward_potential_discount_matches_ppo_discount():
    assert load_reward_v2_config().progress_gamma == load_training_profile().gamma


def test_native_rsl_config_exposes_required_ppo_controls():
    profile = load_training_profile()
    cfg = build_rsl_runner_config(profile, seed=profile.seed_train[0], max_iterations=7)
    assert cfg["actor"]["hidden_dims"] == [256, 256]
    assert cfg["critic"]["hidden_dims"] == [256, 256]
    assert cfg["actor"]["distribution_cfg"]["init_std"] == pytest.approx(0.15)
    assert cfg["algorithm"]["gamma"] == pytest.approx(0.995)
    assert cfg["algorithm"]["lam"] == pytest.approx(0.95)
    assert cfg["algorithm"]["clip_param"] == pytest.approx(0.2)
    assert cfg["algorithm"]["use_clipped_value_loss"] is True
    assert cfg["algorithm"]["max_grad_norm"] == pytest.approx(1.0)
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]}


def test_entropy_schedule_is_applied_before_every_native_ppo_update() -> None:
    class _Algorithm:
        entropy_coef = -1.0

        def __init__(self) -> None:
            self.observed: list[float] = []

        def update(self) -> None:
            self.observed.append(self.entropy_coef)

    class _Runner:
        def __init__(self) -> None:
            self.alg = _Algorithm()

        def learn(self, *, num_learning_iterations, init_at_random_ep_len) -> None:
            assert init_at_random_ep_len is False
            for _ in range(num_learning_iterations):
                self.alg.update()

    runner = _Runner()
    original_update_function = runner.alg.update.__func__

    applied = learn_with_entropy_schedule(
        runner,
        num_learning_iterations=5,
        entropy_start=0.005,
        entropy_end=0.001,
    )

    assert applied == pytest.approx((0.005, 0.004, 0.003, 0.002, 0.001))
    assert runner.alg.observed == pytest.approx(applied)
    assert runner.alg.update.__func__ is original_update_function


def test_entropy_schedule_validates_update_count_and_endpoints() -> None:
    assert linear_entropy_schedule(0.005, 0.001, num_updates=1) == (0.005,)
    with pytest.raises(RlLibraryConfigurationError, match="at least one update"):
        linear_entropy_schedule(0.005, 0.001, num_updates=0)
    with pytest.raises(RlLibraryConfigurationError, match="non-negative"):
        linear_entropy_schedule(-0.1, 0.001, num_updates=2)


def test_entropy_window_is_continuous_across_policy_decision_chunks() -> None:
    profile = load_training_profile()
    planned = planned_entropy_anneal_policy_decisions(profile)
    first_end = entropy_coefficient_at_policy_decision(
        profile.entropy_start,
        profile.entropy_end,
        global_policy_decision=10_240,
        planned_policy_decisions=planned,
    )
    second_start = entropy_coefficient_at_policy_decision(
        profile.entropy_start,
        profile.entropy_end,
        global_policy_decision=10_240,
        planned_policy_decisions=planned,
    )

    assert planned == 210_000
    assert first_end == pytest.approx(second_start)
    assert second_start < profile.entropy_start
    assert entropy_coefficient_at_policy_decision(
        profile.entropy_start,
        profile.entropy_end,
        global_policy_decision=planned + 1,
        planned_policy_decisions=planned,
    ) == pytest.approx(profile.entropy_end)


def test_policy_decision_budget_rounds_up_by_env_and_rollout():
    assert iterations_for_policy_decisions(10_000, num_envs=8, rollout_length=128) == 10
    with pytest.raises(RlLibraryConfigurationError):
        iterations_for_policy_decisions(0, num_envs=8, rollout_length=128)


def test_zero_mean_initializer_zeros_only_actor_output_layer():
    torch = pytest.importorskip("torch")
    actor = SimpleNamespace(mlp=torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ELU(), torch.nn.Linear(8, 12)))
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor))
    before = actor.mlp[0].weight.detach().clone()
    initialize_zero_mean_actor(runner)
    assert torch.equal(actor.mlp[0].weight, before)
    assert torch.count_nonzero(actor.mlp[-1].weight) == 0
    assert torch.count_nonzero(actor.mlp[-1].bias) == 0
    assert torch.equal(actor.mlp(torch.randn(3, 5)), torch.zeros(3, 12))
    assert zero_mean_actor_output_layer_verified(runner) is True
    with torch.no_grad():
        actor.mlp[-1].bias[0] = 1.0
    assert zero_mean_actor_output_layer_verified(runner) is False


def test_installed_rsl_matches_isaac_lab_pin():
    assert assert_supported_rsl_runtime() == "5.0.1"


def test_resume_provenance_binds_loaded_infos_sidecar_and_checkpoint_bytes(tmp_path):
    checkpoint, infos, sidecar, _ = _resume_checkpoint(tmp_path)

    result = validate_resume_checkpoint_provenance(
        checkpoint,
        infos,
        expected_global_policy_decisions=12_345,
    )

    assert result.checkpoint_path == checkpoint.resolve()
    assert result.checkpoint_sha256 == sha256_file(checkpoint)
    assert result.manifest_path == sidecar.resolve()
    assert result.manifest_sha256 == sha256_file(sidecar)
    assert result.global_policy_decisions == 12_345
    assert result.stage == "full_episode_100k"
    assert result.checkpoint_infos_match_manifest is True


def test_resume_provenance_rejects_self_consistent_stale_runtime_contract(tmp_path):
    checkpoint, infos, _, _ = _resume_checkpoint(tmp_path)
    expected = {field: infos[field] for field in CHECKPOINT_RUNTIME_CONTRACT_FIELDS}
    expected["reward_config_hash"] = "f" * 64

    with pytest.raises(
        RlLibraryConfigurationError,
        match="runtime contract differs for 'reward_config_hash'",
    ):
        validate_resume_checkpoint_provenance(
            checkpoint,
            infos,
            expected_runtime_contract=expected,
        )


@pytest.mark.parametrize(
    "field", ("source_git_commit", "committed_runtime_content_sha256")
)
def test_resume_provenance_rejects_different_committed_runtime_identity(
    tmp_path, field
):
    checkpoint, infos, _, _ = _resume_checkpoint(tmp_path)
    expected = {name: infos[name] for name in CHECKPOINT_RUNTIME_CONTRACT_FIELDS}
    expected[field] = "f" * (40 if field == "source_git_commit" else 64)

    with pytest.raises(
        RlLibraryConfigurationError,
        match=f"runtime contract differs for '{field}'",
    ):
        validate_resume_checkpoint_provenance(
            checkpoint,
            infos,
            expected_runtime_contract=expected,
        )


def test_resume_provenance_rejects_changed_snapshot_bundle_contract(tmp_path):
    checkpoint, infos, _, _ = _resume_checkpoint(tmp_path)
    expected = {field: infos[field] for field in CHECKPOINT_RUNTIME_CONTRACT_FIELDS}
    expected["phase_snapshot_bundle_sha256"] = "f" * 64

    with pytest.raises(
        RlLibraryConfigurationError,
        match="runtime contract differs for 'phase_snapshot_bundle_sha256'",
    ):
        validate_resume_checkpoint_provenance(
            checkpoint,
            infos,
            expected_runtime_contract=expected,
        )


def test_training_rng_checkpoint_round_trip_continues_without_replaying_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("PYTHONHASHSEED", "1001")
    seed_training_rngs(1001)
    first_torch = torch.rand(4)
    first_numpy = np.random.random(4)
    state = capture_training_rng_state(seed=1001)
    continued_torch = torch.rand(4)
    continued_numpy = np.random.random(4)

    seed_training_rngs(1001)
    assert torch.equal(torch.rand(4), first_torch)
    assert np.array_equal(np.random.random(4), first_numpy)
    restored = restore_training_rng_state(state, expected_seed=1001)

    assert restored["torch_cpu_restored"] is True
    assert torch.equal(torch.rand(4), continued_torch)
    assert np.array_equal(np.random.random(4), continued_numpy)
    assert not torch.equal(continued_torch, first_torch)


def test_loaded_optimizer_learning_rate_resynchronizes_rsl_adaptive_scalar() -> None:
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            learning_rate=3.0e-4,
            optimizer=SimpleNamespace(param_groups=[{"lr": 2.0e-4}]),
        )
    )

    rate = synchronize_loaded_optimizer_learning_rate(
        runner, expected=2.0e-4
    )

    assert rate == pytest.approx(2.0e-4)
    assert runner.alg.learning_rate == pytest.approx(2.0e-4)


def test_saved_checkpoint_manifest_is_directly_resume_validatable(tmp_path):
    class _Runner:
        def __init__(self):
            self.infos = None

        def save(self, path, *, infos):
            self.infos = dict(infos)
            with open(path, "wb") as stream:
                stream.write(b"fake RSL payload containing infos")

    runner = _Runner()
    infos = {
        "schema": "wlr50_clean.phase_residual_checkpoint_manifest.v1",
        "stage": "phase-curriculum",
        "global_policy_decisions": 10_000,
    }
    checkpoint, sidecar = save_checkpoint_with_manifest(
        runner,
        tmp_path / "phase.pt",
        manifest=infos,
    )

    result = validate_resume_checkpoint_provenance(checkpoint, runner.infos)

    assert result.manifest_path == sidecar
    assert result.global_policy_decisions == 10_000


def test_checkpoint_loader_does_not_coerce_non_mapping_infos(tmp_path):
    checkpoint = tmp_path / "bad_infos.pt"
    checkpoint.write_bytes(b"checkpoint")
    runner = SimpleNamespace(
        device="cpu",
        load=lambda *args, **kwargs: [("schema", CHECKPOINT_MANIFEST_SCHEMA)],
    )

    with pytest.raises(RlLibraryConfigurationError, match="infos must be a mapping"):
        load_checkpoint_round_trip(runner, checkpoint)


@pytest.mark.parametrize("step", [True, -1, 1.0, "1", None])
def test_resume_provenance_rejects_non_integer_or_negative_infos_step(tmp_path, step):
    checkpoint, infos, _, _ = _resume_checkpoint(tmp_path)
    infos["global_policy_decisions"] = step

    with pytest.raises(
        RlLibraryConfigurationError,
        match="checkpoint infos global_policy_decisions",
    ):
        validate_resume_checkpoint_provenance(checkpoint, infos)


@pytest.mark.parametrize("infos", [None, {}, {"schema": "wrong"}])
def test_resume_provenance_rejects_missing_or_wrong_infos_schema(tmp_path, infos):
    checkpoint, _, _, _ = _resume_checkpoint(tmp_path)

    with pytest.raises(RlLibraryConfigurationError, match="infos"):
        validate_resume_checkpoint_provenance(checkpoint, infos)


def test_resume_provenance_requires_default_or_explicit_sidecar(tmp_path):
    checkpoint, infos, sidecar, _ = _resume_checkpoint(tmp_path)
    sidecar.unlink()

    with pytest.raises(RlLibraryConfigurationError, match="manifest is missing"):
        validate_resume_checkpoint_provenance(checkpoint, infos)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema", "wrong", "manifest has the wrong schema"),
        ("global_policy_decisions", 12_346, "global_policy_decisions disagree"),
        (
            "global_policy_decisions",
            True,
            "manifest global_policy_decisions",
        ),
        ("stage", "different_stage", "manifest stage disagree"),
        ("training_seed", 1002, "disagree for 'training_seed'"),
        ("checkpoint_path", "different.pt", "different checkpoint path"),
        ("checkpoint_sha256", "0" * 64, "bytes do not match"),
    ],
)
def test_resume_provenance_rejects_sidecar_mismatches(
    tmp_path, field, replacement, message
):
    checkpoint, infos, sidecar, manifest = _resume_checkpoint(tmp_path)
    manifest[field] = replacement
    sidecar.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(RlLibraryConfigurationError, match=message):
        validate_resume_checkpoint_provenance(checkpoint, infos)


def test_resume_provenance_rejects_mutated_checkpoint_bytes(tmp_path):
    checkpoint, infos, _, _ = _resume_checkpoint(tmp_path)
    checkpoint.write_bytes(b"mutated checkpoint")

    with pytest.raises(RlLibraryConfigurationError, match="bytes do not match"):
        validate_resume_checkpoint_provenance(checkpoint, infos)


@pytest.mark.parametrize("expected", [12_346, True, -1])
def test_resume_provenance_strictly_validates_expected_global_step(tmp_path, expected):
    checkpoint, infos, _, _ = _resume_checkpoint(tmp_path)

    with pytest.raises(RlLibraryConfigurationError, match="global_policy_decisions"):
        validate_resume_checkpoint_provenance(
            checkpoint,
            infos,
            expected_global_policy_decisions=expected,
        )


def test_resume_provenance_accepts_an_explicit_sidecar_path(tmp_path):
    checkpoint, infos, sidecar, _ = _resume_checkpoint(tmp_path)
    explicit = tmp_path / "explicit_provenance.json"
    sidecar.rename(explicit)

    result = validate_resume_checkpoint_provenance(
        checkpoint,
        infos,
        manifest_path=explicit,
    )

    assert result.manifest_path == explicit.resolve()
