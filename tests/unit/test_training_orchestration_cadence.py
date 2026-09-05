"""Real chunk consumer tests; only finalized process-lifecycle discovery is stubbed."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo import training_orchestration as subject


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _consumer_inputs(tmp_path, monkeypatch, n=8):
    root = tmp_path.resolve()
    config = root / "profile.yaml"
    config.write_bytes(subject.DEFAULT_TRAINING_CONFIG.read_bytes())
    snapshot = subject._snapshot(config, label="profile", cache={})
    cadence = subject.derive_stage_cadence(stage="full-episode", num_envs=n,
                                         **subject._load_profile(snapshot)["cadence_inputs"])
    count = cadence["actual_policy_decisions_per_chunk"]
    phases = tuple(f"P{i:02d}" for i in range(1, 14))
    families = ("phase_task_progress", "body_stability", "contact_motion_quality", "control_smoothness", "residual_regularization")
    telemetry = {
        "policy_decision_count": count, "reward_telemetry_complete": True,
        "phase_decision_counts": {p: count if p == "P01" else 0 for p in phases},
        "reward_family_absolute_sums_by_phase": {p: {f: 0.0 for f in families} for p in phases},
        "reward_dominance_within_limits": True, "authoritative_completed_episode_count": 0,
        "authoritative_terminal_reason_counts": {}, "authoritative_success_count": 0,
        "vector_batch_reset_peer_count": 0, "completed_sample_count": 0,
    }
    outcome = cli._validate_training_telemetry(telemetry, stage="full-episode", expected_policy_decisions=count)
    initial = root / "initial.pt"
    initial.write_bytes(b"initial")
    initial_hash = subject._snapshot(initial, label="initial", cache={}).sha256
    checkpoint = root / "checkpoint_full.pt"
    checkpoint.write_bytes(b"trained")
    checkpoint_hash = subject._snapshot(checkpoint, label="checkpoint", cache={}).sha256
    shared = dict(stage="full-episode", stage_policy_decisions=count, global_policy_decisions=10_000 + count,
                  resume_global_policy_decisions=10_000, resume_checkpoint=str(initial), resume_checkpoint_sha256=initial_hash,
                  training_cadence=cadence, training_telemetry=telemetry, training_outcome_diagnostics=outcome)
    result = dict(copy.deepcopy(shared), schema=subject.TRAINING_RESULT_SCHEMA,
                  requested_policy_decisions=cadence["requested_policy_decisions_per_chunk"], iterations=25, num_envs=n,
                  rollout_length=128, ppo_batch_policy_decisions=n * 128,
                  rounding_overrun_policy_decisions=count - cadence["requested_policy_decisions_per_chunk"],
                  budget_accounting_basis="requested_policy_decisions", deterministic_validation_interval=cadence["requested_policy_decisions_per_chunk"],
                  early_stop_when_promotion_gate_passes=True, save_load_round_trip=True,
                  round_trip_infos=copy.deepcopy(shared), wall_time_s=1.0,
                  immutable_history_checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  checkpoint_last=str(root / "checkpoint_last.pt"))
    metadata = dict(copy.deepcopy(shared), schema=subject.CHECKPOINT_MANIFEST_SCHEMA,
                    checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                    source_git_commit="a" * 40, committed_runtime_content_sha256="b" * 64,
                    creation_runtime_identity_path=str(root / "committed_runtime_identity.before.json"),
                    creation_runtime_identity_sha256="c" * 64, files={str(config): snapshot.sha256})
    result_path = root / "training_result.json"
    manifest_path = root / "checkpoint_full_manifest.json"
    def save():
        _write(result_path, result)
        _write(manifest_path, metadata)
    save()
    now = datetime.now(timezone.utc)
    run = dict(directory=root, identity={"training_stage": "full-episode", "environment_count": n, "git_commit": "a" * 40},
               configs=[{"path": "profile.yaml", "bytes": snapshot.size, "sha256": snapshot.sha256}],
               committed_runtime_identity_before_payload={"content_sha256": "b" * 64},
               committed_runtime_identities=[{"sha256": "c" * 64}], frozen_audits=[], run_manifest={},
               completed_at=now, started_at=now)
    monkeypatch.setattr(subject, "_validate_finalized_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(subject, "_required_artifact", lambda *args, **kwargs: (result_path, json.loads(result_path.read_text())))
    def validate():
        save()
        return subject._validate_training_chunk(root, project_root=root, training_config=snapshot, interval=10_000, cache={})
    return result, metadata, validate


@pytest.mark.parametrize("n", [8, 16, 32])
def test_real_consumer_accepts_sorted_json_and_reports_unsuccessful_training_honestly(tmp_path, monkeypatch, n):
    _, _, validate = _consumer_inputs(tmp_path, monkeypatch, n)
    checked = validate()
    assert checked["iterations"] == 25
    assert checked["training_cadence"]["policy_decisions_per_env_per_chunk"] == 3200
    assert checked["training_outcome_diagnostics"]["diagnostic_warnings"] == [
        "FULL_EPISODE_PHASE_COVERAGE_INCOMPLETE", "NO_STOCHASTIC_FULL_EPISODE_SUCCESS_OBSERVED"]
    assert checked["training_outcome_diagnostics"]["checkpoint_promotion_claimed"] is False


@pytest.mark.parametrize("location", ["result", "roundtrip", "manifest"])
@pytest.mark.parametrize("field", ["training_cadence", "training_outcome_diagnostics", "training_telemetry"])
def test_every_checkpoint_evidence_copy_is_bound(tmp_path, monkeypatch, location, field):
    result, manifest, validate = _consumer_inputs(tmp_path, monkeypatch)
    target = result if location == "result" else result["round_trip_infos"] if location == "roundtrip" else manifest
    target.pop(field)
    with pytest.raises(subject.TrainingOrchestrationError):
        validate()


def test_identical_false_diagnostic_in_all_three_copies_is_recomputed_and_rejected(tmp_path, monkeypatch):
    result, manifest, validate = _consumer_inputs(tmp_path, monkeypatch)
    for target in (result, result["round_trip_infos"], manifest):
        target["training_outcome_diagnostics"]["checkpoint_promotion_claimed"] = True
    with pytest.raises(subject.TrainingOrchestrationError, match="diagnostics differ"):
        validate()


@pytest.mark.parametrize("mutation", ["reward_key", "partition", "peer_success", "reward_dominance"])
def test_structural_training_integrity_is_still_hard_required(tmp_path, monkeypatch, mutation):
    result, _, validate = _consumer_inputs(tmp_path, monkeypatch)
    telemetry = result["training_telemetry"]
    if mutation == "reward_key":
        telemetry["reward_family_absolute_sums_by_phase"]["P01"]["extra"] = 0
    elif mutation == "partition":
        telemetry["phase_decision_counts"]["P13"] = 1
    elif mutation == "peer_success":
        telemetry["authoritative_success_count"] = 1
        telemetry["vector_batch_reset_peer_count"] = 1
        telemetry["completed_sample_count"] = 1
    else:
        telemetry["reward_dominance_within_limits"] = False
    with pytest.raises(subject.TrainingOrchestrationError):
        validate()


def test_old_full_interval_declaration_cannot_pass_under_new_metadata(tmp_path, monkeypatch):
    result, _, validate = _consumer_inputs(tmp_path, monkeypatch)
    result["deterministic_validation_interval"] = 10_000
    with pytest.raises(subject.TrainingOrchestrationError, match="accounting/control"):
        validate()
