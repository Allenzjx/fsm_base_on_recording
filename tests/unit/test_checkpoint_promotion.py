from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import checkpoint_promotion as subject
from wlr50_clean.ppo import cli


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _checkpoint_evidence(tmp_path: Path, *, name: str = "candidate.pt") -> tuple[Path, Path]:
    checkpoint = tmp_path / name
    checkpoint.write_bytes(b"actual trained RSL checkpoint bytes")
    digest = subject.sha256_file(checkpoint)
    manifest = _json(
        tmp_path / "candidate_manifest.json",
        {
            "schema": subject.CHECKPOINT_MANIFEST_SCHEMA,
            "stage": "full-episode",
            "training_seed": 1001,
            "global_policy_decisions": 100_000,
            "actor_observation_dimension": 125,
            "critic_observation_dimension": 125,
            "residual_dimension": 12,
            "physics_hz": 120.0,
            "decision_hz": 15.0,
            "files": {"training.yaml": "f" * 64},
            "controller_hash": "a" * 64,
            "environment_hash": "b" * 64,
            "observation_schema_hash": "c" * 64,
            "action_schema_hash": "d" * 64,
            "reward_config_hash": "e" * 64,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
        },
    )
    return checkpoint, manifest


def _promotion_decision(
    tmp_path: Path,
    checkpoint: Path,
    *,
    promoted: bool = True,
    failed_gate: str | None = None,
) -> Path:
    checks = {gate: True for gate in subject.REQUIRED_PROMOTION_GATES}
    if failed_gate is not None:
        checks[failed_gate] = False
    return _json(
        tmp_path / "promotion_decision.json",
        {
            "schema": subject.PROMOTION_DECISION_SCHEMA,
            "baseline_checkpoint": "pure_fsm",
            "candidate_checkpoint": "arbitrary_evaluated_candidate_label",
            "candidate_checkpoint_path": str(checkpoint),
            "candidate_checkpoint_sha256": subject.sha256_file(checkpoint),
            "paired_seeds": [2001, 2002, 2003, 2004, 2005],
            "paired_episode_count": 5,
            "minimum_paired_seeds": 5,
            "frozen_hashes_unchanged": True,
            "promotion": {
                "promoted": promoted,
                "first_failed_gate": failed_gate,
                "checks": checks,
                "global_stability_improvement_fraction": 0.075,
                "improved_priority_phase_count": 4,
            },
            "first_failed_gate": failed_gate,
            "checks_in_evaluation_order": [
                {"gate": gate, "passed": value} for gate, value in checks.items()
            ],
        },
    )


def test_required_promotion_gates_match_complete_evaluation_schema() -> None:
    assert len(subject.REQUIRED_PROMOTION_GATES) == 18
    assert subject.REQUIRED_PROMOTION_GATES[-4:] == (
        "level_calibration_quality_passed",
        "residual_activity_calibrated",
        "priority_phases_have_real_residual",
        "at_least_10_phases_have_real_residual",
    )


def _promote_best(tmp_path: Path, *, name: str = "candidate.pt"):
    checkpoint, manifest = _checkpoint_evidence(tmp_path, name=name)
    decision = _promotion_decision(tmp_path, checkpoint)
    artifacts = subject.promote_best_validation_checkpoint(
        promotion_decision_path=decision,
        candidate_checkpoint_path=checkpoint,
        candidate_manifest_path=manifest,
        output_root=tmp_path / "output",
    )
    return checkpoint, manifest, decision, artifacts


def _locked_test_aggregate(
    tmp_path: Path,
    *,
    checkpoint: Path,
    checkpoint_manifest: Path,
    finalized: bool = True,
    failing_seed: int | None = None,
    name: str = "checkpoint_evaluation_aggregate.json",
) -> Path:
    checkpoint_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    workers = []
    episodes = []
    for seed in subject.LOCKED_TEST_SEEDS:
        worker_dir = tmp_path / "locked-workers" / f"worker-{seed}"
        episode_dir = worker_dir / f"episode_000_seed_{seed}"
        episode_dir.mkdir(parents=True)
        success = seed != failing_seed
        episode = {
            "seed": seed,
            "task_success": success,
            "termination_reason": "SUCCESS" if success else "TASK_FAILURE",
            "duration_s": 12.0,
            "body_collision": False,
            "wheel_only_climb": False,
            "safety_abort": False,
            "under_maximum_duration": True,
            "recording_runtime_access_count": 0,
            "in_episode_root_write_count": 0,
            "canonical_episode_dir": str(episode_dir.resolve()),
        }
        trial = _json(
            episode_dir / "trial_manifest.json",
            {
                "schema": "wlr50_clean.ppo_live_trial_manifest.v1",
                "seed": seed,
                "result": "SUCCESS" if success else "TASK_FAILURE",
                "success_evidence": {
                    "p01_p13_completed": success,
                    "body_collision": False,
                    "wheel_only_climb": False,
                    "duration_s": 12.0,
                },
            },
        )
        result = _json(
            worker_dir / "checkpoint_evaluation.json",
            {
                "schema": "wlr50_clean.ppo_checkpoint_evaluation.v1",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": subject.sha256_file(checkpoint),
                "fresh_process_single_episode": True,
                "vec_env_step_called": False,
                "deterministic_mean_policy": True,
                "episode_count": 1,
                "success_count": int(success),
                "passed": success,
                "episodes": [episode],
                "checkpoint_infos": {
                    field: checkpoint_payload[field]
                    for field in subject.FROZEN_HASH_FIELDS
                },
            },
        )
        lifecycle = _json(
            worker_dir / "run_manifest.json",
            {
                "lifecycle": "SUCCEEDED",
                "exit_code": 0,
                "configs": [
                    {
                        "path": relative_path,
                        "sha256": checkpoint_payload[field],
                    }
                    for field, relative_path in (
                        (
                            "observation_schema_hash",
                            "configs/ppo_observation_schema_v2.json",
                        ),
                        (
                            "action_schema_hash",
                            "configs/ppo_phase_action_masks_v2.yaml",
                        ),
                        ("reward_config_hash", "configs/ppo_reward_v2.yaml"),
                    )
                ],
            },
        )
        frozen_manifest_sha256 = "f" * 64
        frozen_audit = {
            "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "passed": True,
            "mismatches": [],
            "entries": [
                {
                    "path": relative_path,
                    "expected_sha256": checkpoint_payload[field],
                    "actual_sha256": checkpoint_payload[field],
                    "exists": True,
                    "valid": True,
                }
                for field, relative_path in (
                    ("controller_hash", "configs/fsm_states.yaml"),
                    ("environment_hash", "configs/environment_lock.json"),
                )
            ],
        }
        _json(worker_dir / "frozen_hashes.before.json", frozen_audit)
        _json(worker_dir / "frozen_hashes.after.json", frozen_audit)
        workers.append(
            {
                "role": "candidate",
                "seed": seed,
                "run_dir": str(worker_dir.resolve()),
                "run_manifest_sha256": subject.sha256_file(lifecycle),
                "worker_result": str(result.resolve()),
                "worker_result_sha256": subject.sha256_file(result),
                "worker_gate_passed": success,
                "canonical_episode_dir": str(episode_dir.resolve()),
                "trial_manifest_sha256": subject.sha256_file(trial),
            }
        )
        episodes.append(episode)

    all_success = failing_seed is None
    aggregate = {
        "schema": subject.LOCKED_TEST_AGGREGATE_SCHEMA,
        "role": "candidate",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": subject.sha256_file(checkpoint),
        "seed_set": "locked-test",
        "seeds": list(subject.LOCKED_TEST_SEEDS),
        "fresh_process_per_episode": True,
        "deterministic_evaluation": True,
        "deterministic_mean_policy": True,
        "episode_count": 5,
        "success_count": 5 if all_success else 4,
        "body_collision_count": 0,
        "wheel_only_climb_count": 0,
        "safety_abort_count": 0,
        "all_under_maximum_duration": True,
        "passed": all_success,
        "worker_gate_pass_count": 5 if all_success else 4,
        "canonical_episode_dirs": [
            row["canonical_episode_dir"] for row in episodes
        ],
        "workers": workers,
        "episodes": episodes,
    }
    finalized_payload = dict(
        subject.finalize_locked_test_aggregate_payload(
            aggregate,
            checkpoint_path=checkpoint,
            checkpoint_manifest_path=checkpoint_manifest,
        )
    )
    if not finalized:
        finalized_payload["finalized"] = False
    return _json(tmp_path / name, finalized_payload)


def test_validation_decision_publishes_only_best_validation(tmp_path: Path) -> None:
    checkpoint, _, _, artifacts = _promote_best(
        tmp_path, name="unremarkable-candidate.bin"
    )

    assert artifacts.best_checkpoint.name == "checkpoint_best_validation.pt"
    assert artifacts.best_checkpoint.read_bytes() == checkpoint.read_bytes()
    assert not (artifacts.best_checkpoint.parent / subject.IMPROVED_CHECKPOINT_NAME).exists()
    assert not (artifacts.best_checkpoint.parent / subject.IMPROVED_MANIFEST_NAME).exists()
    best = json.loads(artifacts.best_manifest.read_text(encoding="utf-8"))
    promotion = json.loads(
        artifacts.validation_promotion_manifest.read_text(encoding="utf-8")
    )
    assert best["publication_role"] == "best_validation"
    assert best["validation_promotion_authorized"] is True
    assert best["locked_test_authorized"] is False
    assert promotion["promotion_scope"] == "best_validation_only"
    assert promotion["improved_checkpoint_authorized"] is False
    assert promotion["filename_inference_used"] is False
    assert promotion["validation_seeds"] == list(subject.VALIDATION_SEEDS)


def test_offline_cli_wires_validation_then_locked_test_publication(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint_evidence(tmp_path)
    decision = _promotion_decision(tmp_path, checkpoint)
    output = tmp_path / "output"
    validation_run = tmp_path / "validation-publication-run"
    validation_run.mkdir()
    assert cli._promote_best_validation(
        SimpleNamespace(
            promotion_decision=decision,
            candidate_checkpoint=checkpoint,
            candidate_manifest=manifest,
            output_root=output,
            run_dir=validation_run,
        )
    ) == 0
    assert not (output / "checkpoints" / subject.IMPROVED_CHECKPOINT_NAME).exists()

    best_checkpoint = output / "checkpoints" / subject.BEST_CHECKPOINT_NAME
    best_manifest = output / "checkpoints" / subject.BEST_MANIFEST_NAME
    validation_evidence = output / "manifests" / subject.VALIDATION_PROMOTION_MANIFEST_NAME
    aggregate = _locked_test_aggregate(
        tmp_path,
        checkpoint=best_checkpoint,
        checkpoint_manifest=best_manifest,
    )
    improved_run = tmp_path / "improved-publication-run"
    improved_run.mkdir()
    assert cli._promote_improved(
        SimpleNamespace(
            promotion_decision=decision,
            locked_test_aggregate=aggregate,
            best_validation_checkpoint=best_checkpoint,
            best_validation_manifest=best_manifest,
            validation_promotion_manifest=validation_evidence,
            output_root=output,
            run_dir=improved_run,
        )
    ) == 0
    assert (output / "checkpoints" / subject.IMPROVED_CHECKPOINT_NAME).is_file()
    assert (validation_run / "best_validation_publication.json").is_file()
    assert (improved_run / "improved_checkpoint_publication.json").is_file()
    assert "promote-best-validation" not in cli.LIVE_COMMANDS
    assert "promote-improved" not in cli.LIVE_COMMANDS


def test_improved_filename_cannot_override_failed_decision(tmp_path: Path) -> None:
    checkpoint, manifest = _checkpoint_evidence(
        tmp_path, name="checkpoint_super_improved_best.pt"
    )
    decision = _promotion_decision(
        tmp_path,
        checkpoint,
        promoted=False,
        failed_gate="body_collision_zero",
    )
    output = tmp_path / "blocked"

    with pytest.raises(subject.CheckpointPromotionError, match="did not authorize"):
        subject.promote_best_validation_checkpoint(
            promotion_decision_path=decision,
            candidate_checkpoint_path=checkpoint,
            candidate_manifest_path=manifest,
            output_root=output,
        )
    assert not output.exists()


def test_promotion_rejects_incomplete_gate_set_and_checkpoint_hash_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint_evidence(tmp_path)
    decision = _promotion_decision(tmp_path, checkpoint)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["promotion"]["checks"].pop("wheel_only_climb_zero")
    payload["checks_in_evaluation_order"] = payload["checks_in_evaluation_order"][:-1]
    _json(decision, payload)

    with pytest.raises(subject.CheckpointPromotionError, match="complete gate set"):
        subject.promote_best_validation_checkpoint(
            promotion_decision_path=decision,
            candidate_checkpoint_path=checkpoint,
            candidate_manifest_path=manifest,
            output_root=tmp_path / "missing-gate",
        )

    decision = _promotion_decision(tmp_path, checkpoint)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["candidate_checkpoint_sha256"] = "0" * 64
    _json(decision, payload)
    with pytest.raises(subject.CheckpointPromotionError, match="hash mismatch"):
        subject.promote_best_validation_checkpoint(
            promotion_decision_path=decision,
            candidate_checkpoint_path=checkpoint,
            candidate_manifest_path=manifest,
            output_root=tmp_path / "bad-hash",
        )


def test_validation_promotion_preflights_destinations_and_never_overwrites(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint_evidence(tmp_path)
    decision = _promotion_decision(tmp_path, checkpoint)
    output = tmp_path / "output"
    conflict = output / "checkpoints" / subject.BEST_CHECKPOINT_NAME
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"prior immutable checkpoint")

    with pytest.raises(subject.CheckpointPromotionError, match="refusing to overwrite"):
        subject.promote_best_validation_checkpoint(
            promotion_decision_path=decision,
            candidate_checkpoint_path=checkpoint,
            candidate_manifest_path=manifest,
            output_root=output,
        )
    assert conflict.read_bytes() == b"prior immutable checkpoint"
    assert list(conflict.parent.iterdir()) == [conflict]
    assert not (output / "manifests").exists()


def test_validation_bundle_rolls_back_if_a_later_atomic_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest = _checkpoint_evidence(tmp_path)
    decision = _promotion_decision(tmp_path, checkpoint)
    output = tmp_path / "output"
    real_publish = subject._publish_no_replace
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subject.CheckpointPromotionError("injected atomic publication failure")
        real_publish(source, destination)

    monkeypatch.setattr(subject, "_publish_no_replace", fail_second)
    with pytest.raises(subject.CheckpointPromotionError, match="injected"):
        subject.promote_best_validation_checkpoint(
            promotion_decision_path=decision,
            candidate_checkpoint_path=checkpoint,
            candidate_manifest_path=manifest,
            output_root=output,
        )

    assert not (output / "checkpoints" / subject.BEST_CHECKPOINT_NAME).exists()
    assert not (output / "checkpoints" / subject.BEST_MANIFEST_NAME).exists()
    assert not (
        output / "manifests" / subject.VALIDATION_PROMOTION_MANIFEST_NAME
    ).exists()


def test_improved_requires_finalized_locked_test_and_publishes_second_stage(
    tmp_path: Path,
) -> None:
    candidate, _, decision, validation = _promote_best(tmp_path)
    aggregate = _locked_test_aggregate(
        tmp_path,
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
    )

    improved = subject.promote_improved_checkpoint(
        promotion_decision_path=decision,
        locked_test_aggregate_path=aggregate,
        best_validation_checkpoint_path=validation.best_checkpoint,
        best_validation_manifest_path=validation.best_manifest,
        validation_promotion_manifest_path=validation.validation_promotion_manifest,
        output_root=tmp_path / "output",
    )

    assert improved.improved_checkpoint.read_bytes() == candidate.read_bytes()
    assert subject.sha256_file(improved.improved_checkpoint) == subject.sha256_file(
        validation.best_checkpoint
    )
    manifest = json.loads(improved.improved_manifest.read_text(encoding="utf-8"))
    evidence = json.loads(improved.promotion_manifest.read_text(encoding="utf-8"))
    assert manifest["publication_role"] == "improved"
    assert manifest["locked_test_authorized"] is True
    assert manifest["locked_test"]["seeds"] == list(subject.LOCKED_TEST_SEEDS)
    assert evidence["two_stage_promotion"] is True
    assert evidence["validation_decision_alone_cannot_authorize_improved"] is True
    assert evidence["published_checkpoints"]["improved"]["manifest_sha256"] == (
        subject.sha256_file(improved.improved_manifest)
    )


def test_improved_rejects_unfinalized_failed_or_hash_tampered_locked_test(
    tmp_path: Path,
) -> None:
    _, _, decision, validation = _promote_best(tmp_path)
    output = tmp_path / "output"
    unfinalized = _locked_test_aggregate(
        tmp_path,
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
        finalized=False,
        name="unfinalized.json",
    )
    with pytest.raises(subject.CheckpointPromotionError, match="finalized=true"):
        subject.promote_improved_checkpoint(
            promotion_decision_path=decision,
            locked_test_aggregate_path=unfinalized,
            best_validation_checkpoint_path=validation.best_checkpoint,
            best_validation_manifest_path=validation.best_manifest,
            validation_promotion_manifest_path=validation.validation_promotion_manifest,
            output_root=output,
        )

    failed = _locked_test_aggregate(
        tmp_path / "failed",
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
        failing_seed=3004,
    )
    with pytest.raises(subject.CheckpointPromotionError, match="passed=true"):
        subject.promote_improved_checkpoint(
            promotion_decision_path=decision,
            locked_test_aggregate_path=failed,
            best_validation_checkpoint_path=validation.best_checkpoint,
            best_validation_manifest_path=validation.best_manifest,
            validation_promotion_manifest_path=validation.validation_promotion_manifest,
            output_root=output,
        )

    tampered = _locked_test_aggregate(
        tmp_path / "tampered",
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
    )
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    Path(payload["workers"][2]["worker_result"]).write_text(
        "tampered after aggregate\n", encoding="utf-8"
    )
    with pytest.raises(subject.CheckpointPromotionError, match="does not match its file"):
        subject.promote_improved_checkpoint(
            promotion_decision_path=decision,
            locked_test_aggregate_path=tampered,
            best_validation_checkpoint_path=validation.best_checkpoint,
            best_validation_manifest_path=validation.best_manifest,
            validation_promotion_manifest_path=validation.validation_promotion_manifest,
            output_root=output,
        )
    assert not (output / "checkpoints" / subject.IMPROVED_CHECKPOINT_NAME).exists()


@pytest.mark.parametrize(
    ("gate", "message"),
    (
        ("success", "task_success=true"),
        ("body_collision", "body_collision=false"),
        ("safety_abort", "safety_abort=false"),
        ("duration", "outside"),
        ("hash", "failed hash gate"),
    ),
)
def test_improved_rechecks_each_locked_test_gate(
    tmp_path: Path, gate: str, message: str
) -> None:
    _, _, decision, validation = _promote_best(tmp_path)
    aggregate = _locked_test_aggregate(
        tmp_path,
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
    )
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    if gate == "success":
        payload["episodes"][0]["task_success"] = False
    elif gate == "body_collision":
        payload["episodes"][0]["body_collision"] = True
    elif gate == "safety_abort":
        payload["episodes"][0]["safety_abort"] = True
    elif gate == "duration":
        payload["episodes"][0]["duration_s"] = 201.0
    else:
        payload["hash_gates"]["controller_hash_unchanged"] = False
    _json(aggregate, payload)

    with pytest.raises(subject.CheckpointPromotionError, match=message):
        subject.promote_improved_checkpoint(
            promotion_decision_path=decision,
            locked_test_aggregate_path=aggregate,
            best_validation_checkpoint_path=validation.best_checkpoint,
            best_validation_manifest_path=validation.best_manifest,
            validation_promotion_manifest_path=validation.validation_promotion_manifest,
            output_root=tmp_path / "output",
        )
    assert not (
        tmp_path / "output" / "checkpoints" / subject.IMPROVED_CHECKPOINT_NAME
    ).exists()


def test_locked_test_finalizer_recomputes_worker_config_hash_gates(
    tmp_path: Path,
) -> None:
    _, _, _, validation = _promote_best(tmp_path)
    aggregate = _locked_test_aggregate(
        tmp_path,
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
    )
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    run_manifest_path = Path(payload["workers"][0]["run_dir"]) / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["configs"][0]["sha256"] = "0" * 64
    _json(run_manifest_path, run_manifest)
    payload["workers"][0]["run_manifest_sha256"] = subject.sha256_file(
        run_manifest_path
    )

    finalized = subject.finalize_locked_test_aggregate_payload(
        payload,
        checkpoint_path=validation.best_checkpoint,
        checkpoint_manifest_path=validation.best_manifest,
    )

    assert finalized["finalized"] is True
    assert finalized["worker_artifact_hashes_recomputed"] is True
    assert finalized["hash_gates"]["observation_schema_hash_unchanged"] is False
    assert finalized["frozen_hashes_unchanged"] is False
    assert finalized["passed"] is False


def test_improved_publication_never_overwrites(tmp_path: Path) -> None:
    _, _, decision, validation = _promote_best(tmp_path)
    aggregate = _locked_test_aggregate(
        tmp_path,
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
    )
    conflict = tmp_path / "output" / "checkpoints" / subject.IMPROVED_CHECKPOINT_NAME
    conflict.write_bytes(b"prior immutable improved checkpoint")

    with pytest.raises(subject.CheckpointPromotionError, match="refusing to overwrite"):
        subject.promote_improved_checkpoint(
            promotion_decision_path=decision,
            locked_test_aggregate_path=aggregate,
            best_validation_checkpoint_path=validation.best_checkpoint,
            best_validation_manifest_path=validation.best_manifest,
            validation_promotion_manifest_path=validation.validation_promotion_manifest,
            output_root=tmp_path / "output",
        )
    assert conflict.read_bytes() == b"prior immutable improved checkpoint"
    assert not (tmp_path / "output" / "manifests" / subject.PROMOTION_MANIFEST_NAME).exists()


def test_loaded_runner_exports_verified_inference_only_actor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(125, 16),
                torch.nn.ELU(),
                torch.nn.Linear(16, 12),
                torch.nn.Tanh(),
            )

        def forward(self, observations, stochastic_output: bool = False):
            assert stochastic_output is False
            return self.mlp(observations)

        def as_jit(self):
            return copy.deepcopy(self.mlp)

    class TinyRunner:
        device = "cpu"

        def __init__(self) -> None:
            torch.manual_seed(123)
            self.policy = TinyPolicy().eval()

        def get_inference_policy(self, device="cpu"):
            return self.policy.to(device)

    _, _, decision, validation = _promote_best(tmp_path)
    aggregate = _locked_test_aggregate(
        tmp_path,
        checkpoint=validation.best_checkpoint,
        checkpoint_manifest=validation.best_manifest,
    )
    improved = subject.promote_improved_checkpoint(
        promotion_decision_path=decision,
        locked_test_aggregate_path=aggregate,
        best_validation_checkpoint_path=validation.best_checkpoint,
        best_validation_manifest_path=validation.best_manifest,
        validation_promotion_manifest_path=validation.validation_promotion_manifest,
        output_root=tmp_path / "output",
    )
    output = tmp_path / "export"
    artifacts = subject.export_inference_actor(
        TinyRunner(),
        source_checkpoint_path=improved.improved_checkpoint,
        source_manifest_path=improved.improved_manifest,
        output_root=output,
    )

    assert artifacts.torchscript_actor.is_file()
    assert artifacts.torchscript_actor.name == "policy_improved_actor.pt"
    reloaded = torch.jit.load(str(artifacts.torchscript_actor)).eval()
    assert tuple(reloaded(torch.zeros(2, 125)).shape) == (2, 12)
    evidence = artifacts.evidence
    assert evidence["inference_only"] is True
    assert evidence["contains_critic"] is False
    assert evidence["contains_optimizer"] is False
    assert evidence["contains_stochastic_sampler"] is False
    assert evidence["torchscript"]["valid"] is True
    assert evidence["torchscript"]["equivalent_to_loaded_runner"] is True
    assert evidence["torchscript"]["maximum_absolute_error"] <= 1.0e-6
    if evidence["onnx"]["supported"]:
        assert artifacts.onnx_actor is not None and artifacts.onnx_actor.is_file()
        assert artifacts.onnx_actor.name == "policy_improved_actor.onnx"
        assert evidence["onnx"]["valid"] is True
        assert evidence["onnx"]["status"] == "PASS"
        assert evidence["onnx"]["equivalent_to_loaded_runner"] is True
        assert evidence["onnx"]["maximum_absolute_error"] <= 1.0e-6
    else:
        assert artifacts.onnx_actor is None
        assert evidence["onnx"]["status"] == "UNSUPPORTED"

    with pytest.raises(subject.CheckpointPromotionError, match="refusing to overwrite"):
        subject.export_inference_actor(
            TinyRunner(),
            source_checkpoint_path=improved.improved_checkpoint,
            source_manifest_path=improved.improved_manifest,
            output_root=output,
        )


def test_actor_export_rejects_checkpoint_without_locked_test_promotion(
    tmp_path: Path,
) -> None:
    checkpoint, manifest = _checkpoint_evidence(
        tmp_path, name="checkpoint_improved_by_filename_only.pt"
    )
    with pytest.raises(subject.CheckpointPromotionError, match="two-stage promoted"):
        subject.export_inference_actor(
            object(),
            source_checkpoint_path=checkpoint,
            source_manifest_path=manifest,
            output_root=tmp_path / "export",
        )
    assert not (tmp_path / "export").exists()


def test_actor_export_cli_loads_real_runner_without_stepping_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import rl_library_wrapper

    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runner = object()
    construction = {}

    def fake_construct(args, simulation_app, **kwargs):
        construction.update(kwargs)
        construction["simulation_app"] = simulation_app
        return object(), object(), runner, {}

    monkeypatch.setattr(cli, "_construct_live_runner", fake_construct)
    monkeypatch.setattr(
        cli,
        "_current_checkpoint_runtime_contract",
        lambda args: {
            field: manifest[field]
            for field in rl_library_wrapper.CHECKPOINT_RUNTIME_CONTRACT_FIELDS
        },
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_checkpoint_round_trip",
        lambda loaded_runner, loaded_checkpoint: {
            key: manifest[key]
            for key in (
                "schema",
                "stage",
                "training_seed",
                "global_policy_decisions",
                *rl_library_wrapper.CHECKPOINT_RUNTIME_CONTRACT_FIELDS,
            )
        },
    )
    exported = SimpleNamespace(
        torchscript_actor=tmp_path / "output" / "checkpoints" / "policy_improved_actor.pt",
        onnx_actor=tmp_path / "output" / "checkpoints" / "policy_improved_actor.onnx",
        export_manifest=tmp_path / "output" / "manifests" / "inference_actor_export_manifest.json",
    )
    monkeypatch.setattr(subject, "export_inference_actor", lambda *args, **kwargs: exported)
    run_dir = tmp_path / "actor-export-run"
    run_dir.mkdir()
    args = SimpleNamespace(
        num_envs=1,
        deterministic=True,
        checkpoint=checkpoint,
        checkpoint_manifest=manifest_path,
        output_root=tmp_path / "output",
        run_dir=run_dir,
        seed=4001,
    )

    assert cli._export_inference_actor(args, object()) == 0
    assert construction["max_iterations"] == 1
    assert construction["reset_seeds"] == (4001,)
    assert construction["collect_trace"] is False
    result = json.loads((run_dir / "inference_actor_export.json").read_text())
    assert result["live_rsl_runner_loaded"] is True
    assert result["episode_stepped"] is False
    assert result["torchscript_actor"].endswith("policy_improved_actor.pt")
