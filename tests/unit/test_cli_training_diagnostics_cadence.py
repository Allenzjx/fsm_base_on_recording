from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli, rl_library_wrapper


def _telemetry(*, missing=False, success=False, peers=7, decisions=25600):
    phases = {f"P{index:02d}": 0 if missing and index > 2 else 1 for index in range(1, 14)}
    phases["P01"] += decisions - sum(phases.values())
    families = ("phase_task_progress", "body_stability", "contact_motion_quality", "control_smoothness", "residual_regularization")
    return {
        "policy_decision_count": decisions,
        "phase_decision_counts": phases,
        "reward_telemetry_complete": True,
        "reward_dominance_within_limits": True,
        "reward_family_absolute_sums_by_phase": {phase: {family: 0.2 for family in families} for phase in phases},
        "phase_curriculum_occupancy_within_tolerance": False,
        "phase_curriculum_occupancy_violations": ["P08"],
        "authoritative_completed_episode_count": 1,
        "authoritative_terminal_reason_counts": {"SUCCESS" if success else "CONTROLLER_BLOCKED": 1},
        "authoritative_success_count": int(success),
        "vector_batch_reset_peer_count": peers,
        "completed_sample_count": peers + 1,
    }


@pytest.mark.parametrize("missing,success", [(True, False), (False, False), (False, True)])
def test_valid_training_outcomes_are_diagnostics_not_checkpoint_veto(missing, success):
    result = cli._validate_training_telemetry(_telemetry(missing=missing, success=success), stage="full-episode", expected_policy_decisions=25600)
    assert result["all_phases_visited"] is not missing
    assert result["stochastic_full_episode_success_observed"] is success
    assert result["authoritative_success_count"] == int(success)
    assert result["vector_batch_reset_peer_count"] == 7
    assert ("FULL_EPISODE_PHASE_COVERAGE_INCOMPLETE" in result["diagnostic_warnings"]) is missing
    assert ("NO_STOCHASTIC_FULL_EPISODE_SUCCESS_OBSERVED" in result["diagnostic_warnings"]) is not success
    assert result["qualification_requires_deterministic_evaluation"] is True
    assert result["checkpoint_promotion_claimed"] is False


@pytest.mark.parametrize("field,value", [
    ("policy_decision_count", 25599),
    ("reward_telemetry_complete", False),
    ("reward_dominance_within_limits", False),
    ("authoritative_success_count", 1),
    ("completed_sample_count", 1),
    ("phase_decision_counts", {"P01": 25600}),
])
def test_training_integrity_failures_remain_hard_errors(field, value):
    telemetry = _telemetry(missing=True)
    telemetry[field] = value
    with pytest.raises(cli.CliError):
        cli._validate_training_telemetry(telemetry, stage="full-episode", expected_policy_decisions=25600)


def test_phase_curriculum_balance_is_still_mandatory():
    with pytest.raises(cli.CliError, match="occupancy gate"):
        cli._validate_training_telemetry(_telemetry(), stage="phase-curriculum", expected_policy_decisions=25600)


@pytest.mark.parametrize("stage,num_envs,budget,iterations", [
    ("smoke", 8, 10000, 10), ("phase-curriculum", 1, 10000, 79),
    ("full-episode", 8, 25000, 25), ("full-episode", 16, 50000, 25),
    ("full-episode", 32, 100000, 25),
])
def test_cli_uses_profile_derived_stage_cadence(stage, num_envs, budget, iterations):
    profile = rl_library_wrapper.load_training_profile()
    record, actual_budget, actual_iterations = cli._training_chunk_cadence(profile, stage=stage, num_envs=num_envs, requested_policy_decisions=None)
    assert (actual_budget, actual_iterations) == (budget, iterations)
    assert record["requested_policy_decisions_per_chunk"] == budget
    assert record["actual_policy_decisions_per_chunk"] == iterations * num_envs * profile.rollout_length
    assert record["base_validation_interval_policy_decisions"] == 10000
    if stage == "full-episode":
        assert record["policy_decisions_per_env_per_chunk"] == 3200
        assert record["full_window_covers_episode_timeout"] is True


def test_cli_rejects_old_too_short_full_chunk():
    with pytest.raises(cli.CliError, match="cadence"):
        cli._training_chunk_cadence(rl_library_wrapper.load_training_profile(), stage="full-episode", num_envs=8, requested_policy_decisions=10000)


@pytest.mark.parametrize("budget,expected_code", [(10000, 2), (25000, 0)])
def test_main_checks_cadence_before_live_dispatch(tmp_path, monkeypatch, budget, expected_code):
    for name in ("_validate_common", "_capture_runtime_phase_contracts", "_validated_runtime_snapshot_bundle", "_require_training_phase_effective_entry_holdout", "_require_training_phase_zero_residual_rollout", "_require_training_soft_reset_acceptance", "_require_training_vector_benchmark_acceptance"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    dispatched = []
    monkeypatch.setattr(cli, "_dispatch_live", lambda args: dispatched.append(args.policy_decisions) or 0)
    result = cli.main(["train", "--run-dir", str(tmp_path / "run"), "--seed", "1001", "--num-envs", "8", "--stage", "full-episode", "--checkpoint", str(tmp_path / "candidate.pt"), "--policy-decisions", str(budget)])
    assert result == expected_code
    assert dispatched == ([] if expected_code == 2 else [25000])


def test_failed_training_outcomes_are_saved_reloaded_and_reported(tmp_path, monkeypatch):
    profile = rl_library_wrapper.load_training_profile()
    output = tmp_path / "outputs"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resume = tmp_path / "resume.pt"
    initial_infos = {"stage": "phase-curriculum", "global_policy_decisions": 10000, "training_seed": 1001, "optimizer_learning_rate": 0.0003}
    resume.write_text(json.dumps(initial_infos), encoding="utf-8")
    resume.with_name("resume_manifest.json").write_text(json.dumps(initial_infos), encoding="utf-8")
    bundle = SimpleNamespace(as_record=lambda: {"test_bundle": True})
    monkeypatch.setattr(cli, "OUTPUT_ROOT", output)
    monkeypatch.setattr(cli, "_pinned_runtime_phase_contracts", lambda args: (bundle, bundle))
    for name in ("_revalidate_pinned_phase_contracts", "_revalidate_training_phase_effective_entry_holdout", "_revalidate_training_phase_zero_residual_rollout"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    for name in ("_inherit_training_phase_effective_entry_holdout", "_inherit_training_phase_zero_residual_rollout"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_current_checkpoint_runtime_contract", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "_checkpoint_manifest_payload", lambda args, *, global_step, stage, **kwargs: {"schema": rl_library_wrapper.CHECKPOINT_MANIFEST_SCHEMA, "global_policy_decisions": global_step, "stage": stage, "training_seed": args.seed})

    class Runner:
        device = "cpu"
        def save(self, path, *, infos):
            Path(path).write_text(json.dumps(infos), encoding="utf-8")

    runner = Runner()
    env = SimpleNamespace(num_envs=8, cfg={}, training_telemetry=lambda: _telemetry(missing=True))
    monkeypatch.setattr(cli, "_construct_live_runner", lambda *args, **kwargs: (profile, env, runner, {}))
    monkeypatch.setattr(rl_library_wrapper, "seed_training_rngs", lambda seed: {"seed": seed})
    monkeypatch.setattr(rl_library_wrapper, "capture_training_rng_state", lambda *, seed: {"seed": seed})
    monkeypatch.setattr(rl_library_wrapper, "restore_training_rng_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(rl_library_wrapper, "optimizer_learning_rate", lambda runner: 0.0003)
    monkeypatch.setattr(rl_library_wrapper, "learn_with_entropy_schedule", lambda *args, **kwargs: [0.001] * kwargs["num_learning_iterations"])
    monkeypatch.setattr(rl_library_wrapper, "load_checkpoint_round_trip", lambda runner, path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(rl_library_wrapper, "validate_resume_checkpoint_provenance", lambda path, infos, **kwargs: SimpleNamespace(stage=infos["stage"], global_policy_decisions=infos["global_policy_decisions"]))

    def capture(args, checkpoint, manifest, **kwargs):
        path = Path(checkpoint)
        return SimpleNamespace(private_checkpoint_path=path, checkpoint_sha256=cli._sha256(path), manifest_payload=json.loads(Path(manifest).read_text(encoding="utf-8")))
    monkeypatch.setattr(cli, "_pin_live_checkpoint", capture)
    args = SimpleNamespace(stage="full-episode", seed=1001, num_envs=8, checkpoint=resume, checkpoint_manifest=None, policy_decisions=25000, training_config=profile.path, run_dir=run_dir, _vector_benchmark_acceptance_evidence={"passed": True})
    assert cli._train(args, object()) == 0
    result = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["immutable_history_checkpoint_manifest"]).read_text(encoding="utf-8"))
    infos = json.loads(Path(result["immutable_history_checkpoint"]).read_text(encoding="utf-8"))
    for key in ("training_cadence", "training_outcome_diagnostics", "training_telemetry"):
        assert result[key] == manifest[key] == infos[key] == result["round_trip_infos"][key]
    assert result["deterministic_validation_interval"] == 25000
    assert result["save_load_round_trip"] is True
    assert result["training_outcome_diagnostics"]["authoritative_success_count"] == 0
    assert result["training_outcome_diagnostics"]["missing_phases"] == [f"P{i:02d}" for i in range(3, 14)]
    assert Path(result["checkpoint_last"]).is_file()
