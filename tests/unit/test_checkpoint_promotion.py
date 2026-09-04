from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import checkpoint_promotion as subject
from wlr50_clean.ppo import cli
from wlr50_clean.ppo import phase_effective_entry_holdout as holdout_subject
from wlr50_clean.ppo import phase_zero_residual_rollout as rollout_subject
from wlr50_clean.ppo.phase_effective_entry import (
    capture_validated_effective_phase_entry_contract,
)
from wlr50_clean.ppo.phase_snapshots import (
    capture_validated_phase_snapshot_bundle,
    phase_snapshot_bundle_file_hashes,
)


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _holdout_evidence(tmp_path: Path, effective_pin: object) -> dict[str, object]:
    acceptance_path = _json(
        tmp_path / "holdout" / holdout_subject.OUTPUT_FILENAME,
        {
            "status": "PASSED",
            "passed": True,
            "source_git_commit": "a" * 40,
            "phase_effective_entry_contract_sha256": effective_pin.contract_sha256,
        },
    ).resolve()
    run_manifest = _json(
        acceptance_path.parent / "run_manifest.json", {"lifecycle": "SUCCEEDED"}
    ).resolve()
    return {
        "schema": holdout_subject.TRAINING_EVIDENCE_SCHEMA,
        "path": str(acceptance_path),
        "sha256": subject.sha256_file(acceptance_path),
        "phase_effective_entry_contract_sha256": effective_pin.contract_sha256,
        "phase_snapshot_bundle_sha256": effective_pin.phase_snapshot_bundle_sha256,
        "source_git_commit": "a" * 40,
        "backend_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": subject.sha256_file(run_manifest),
        "acceptance": json.loads(acceptance_path.read_text(encoding="utf-8")),
        "passed": True,
    }


def _rollout_evidence(
    tmp_path: Path,
    snapshot_pin: object,
    effective_pin: object,
    holdout: dict[str, object],
) -> dict[str, object]:
    evidence_path = _json(
        tmp_path / "rollout" / rollout_subject.ARTIFACT_FILENAME,
        {"passed": True, "seed": 1003},
    ).resolve()
    run_manifest = _json(
        evidence_path.parent / "run_manifest.json", {"lifecycle": "SUCCEEDED"}
    ).resolve()
    return {
        "schema": rollout_subject.TRAINING_EVIDENCE_SCHEMA,
        "path": str(evidence_path),
        "sha256": subject.sha256_file(evidence_path),
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": subject.sha256_file(run_manifest),
        "contract_binding": rollout_subject.build_contract_binding(
            snapshot_pin, effective_pin, holdout
        ),
        "seed": 1003,
        "passed": True,
    }


@pytest.fixture(autouse=True)
def _validate_test_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(path, *, effective_entry_contract, **_kwargs):
        selected = Path(path).resolve()
        try:
            acceptance = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise holdout_subject.PhaseEffectiveEntryHoldoutError(
                "test holdout cannot be read"
            ) from exc
        if acceptance.get("passed") is not True or acceptance.get("status") != "PASSED":
            raise holdout_subject.PhaseEffectiveEntryHoldoutError(
                "test holdout did not pass"
            )
        run_manifest = selected.parent / "run_manifest.json"
        return {
            "schema": holdout_subject.TRAINING_EVIDENCE_SCHEMA,
            "path": str(selected),
            "sha256": subject.sha256_file(selected),
            "phase_effective_entry_contract_sha256": (
                effective_entry_contract.contract_sha256
            ),
            "phase_snapshot_bundle_sha256": (
                effective_entry_contract.phase_snapshot_bundle_sha256
            ),
            "source_git_commit": acceptance["source_git_commit"],
            "backend_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "run_manifest": str(run_manifest.resolve()),
            "run_manifest_sha256": subject.sha256_file(run_manifest),
            "acceptance": acceptance,
            "passed": True,
        }

    monkeypatch.setattr(
        holdout_subject,
        "validate_phase_effective_entry_holdout_acceptance",
        validate,
    )

    def validate_rollout(path, *, expected_contract_binding, **_kwargs):
        selected = Path(path).resolve()
        try:
            payload = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise rollout_subject.PhaseZeroResidualRolloutError(
                "test rollout cannot be read"
            ) from exc
        if payload.get("passed") is not True:
            raise rollout_subject.PhaseZeroResidualRolloutError(
                "test rollout did not pass"
            )
        run_manifest = selected.parent / "run_manifest.json"
        return {
            "schema": rollout_subject.TRAINING_EVIDENCE_SCHEMA,
            "path": str(selected),
            "sha256": subject.sha256_file(selected),
            "run_manifest": str(run_manifest.resolve()),
            "run_manifest_sha256": subject.sha256_file(run_manifest),
            "contract_binding": dict(expected_contract_binding),
            "seed": payload["seed"],
            "passed": True,
        }

    monkeypatch.setattr(
        rollout_subject,
        "validate_phase_zero_residual_rollout_evidence",
        validate_rollout,
    )


def _checkpoint_infos(tmp_path: Path, *, stage: str = "full-episode") -> dict:
    snapshot_pin = capture_validated_phase_snapshot_bundle(
        cli.DEFAULT_PHASE_SNAPSHOT_ROOT,
        canonical_root=cli.DEFAULT_PHASE_SNAPSHOT_ROOT,
    )
    snapshot_bundle = snapshot_pin.as_record()
    effective_pin = capture_validated_effective_phase_entry_contract(
        expected_snapshot_bundle=snapshot_pin,
    )
    infos = {
        "schema": subject.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": stage,
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
            **effective_pin.file_hashes(),
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
        "phase_effective_entry_contract_path": str(effective_pin.contract_path),
        "phase_effective_entry_contract_file_sha256": effective_pin.file_sha256,
        "phase_effective_entry_contract_sidecar_path": str(effective_pin.sidecar_path),
        "phase_effective_entry_contract_sidecar_sha256": (
            effective_pin.sidecar_file_sha256
        ),
        "phase_effective_entry_contract_sha256": effective_pin.contract_sha256,
        "phase_effective_entry_contract": effective_pin.as_record(),
    }
    if stage not in subject._HOLDOUT_OPTIONAL_CHECKPOINT_STAGES:
        holdout = _holdout_evidence(tmp_path, effective_pin)
        rollout = _rollout_evidence(
            tmp_path, snapshot_pin, effective_pin, holdout
        )
        infos.update(
            {
                "phase_effective_entry_holdout_acceptance_path": holdout["path"],
                "phase_effective_entry_holdout_acceptance_sha256": holdout[
                    "sha256"
                ],
                "phase_effective_entry_holdout_contract_sha256": holdout[
                    "phase_effective_entry_contract_sha256"
                ],
                "phase_effective_entry_holdout_source_git_commit": holdout[
                    "source_git_commit"
                ],
                "phase_effective_entry_holdout_acceptance": holdout["acceptance"],
                "phase_effective_entry_holdout_evidence": holdout,
                "phase_effective_entry_holdout_files": {
                    holdout["path"]: holdout["sha256"],
                    holdout["run_manifest"]: holdout["run_manifest_sha256"],
                },
                "phase_zero_residual_rollout_evidence_path": rollout["path"],
                "phase_zero_residual_rollout_evidence_sha256": rollout["sha256"],
                "phase_zero_residual_rollout_run_manifest_path": rollout[
                    "run_manifest"
                ],
                "phase_zero_residual_rollout_run_manifest_sha256": rollout[
                    "run_manifest_sha256"
                ],
                "phase_zero_residual_rollout_evidence": rollout,
                "phase_zero_residual_rollout_files": {
                    rollout["path"]: rollout["sha256"],
                    rollout["run_manifest"]: rollout["run_manifest_sha256"],
                },
            }
        )
    return infos


def _checkpoint_evidence(
    tmp_path: Path,
    *,
    name: str = "candidate.pt",
    stage: str = "full-episode",
) -> tuple[Path, Path]:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / name
    infos = _checkpoint_infos(tmp_path, stage=stage)
    torch.save({"infos": infos}, checkpoint)
    digest = subject.sha256_file(checkpoint)
    manifest = _json(
        tmp_path / "candidate_manifest.json",
        {
            **infos,
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
    def aggregate_binding(role: str) -> dict[str, object]:
        worker_role = "baseline" if role == "baseline" else "candidate"
        worker_dirs = [
            str((tmp_path / f"{role}_worker_{seed}").resolve())
            for seed in subject.VALIDATION_SEEDS
        ]
        episode_dirs = [
            str((Path(directory) / f"episode_000_seed_{seed}").resolve())
            for directory, seed in zip(
                worker_dirs, subject.VALIDATION_SEEDS, strict=True
            )
        ]
        aggregate_path = tmp_path / f"{role}_aggregate.json"
        aggregate_payload = {
            "schema": "wlr50_clean.fresh_process_episode_batch.v1",
            "role": worker_role,
            "seed_set": "validation",
            "seeds": list(subject.VALIDATION_SEEDS),
            "canonical_episode_dirs": episode_dirs,
            "workers": [
                {"run_dir": directory}
                for directory in worker_dirs
            ],
            "checkpoint": str(checkpoint.resolve()) if role == "candidate" else None,
            "checkpoint_sha256": (
                subject.sha256_file(checkpoint) if role == "candidate" else None
            ),
            "passed": True,
        }
        _json(aggregate_path, aggregate_payload)
        record = {
            "path": str(aggregate_path.resolve()),
            "bytes": aggregate_path.stat().st_size,
            "sha256": subject.sha256_file(aggregate_path),
        }
        records = [record]
        binding: dict[str, object] = {
            "schema": "wlr50_clean.validation_aggregate_binding.v1",
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "role": role,
            "physical_passed": True,
            "seeds": list(subject.VALIDATION_SEEDS),
            "worker_run_dirs": worker_dirs,
            "canonical_episode_dirs": episode_dirs,
            "source_file_records": records,
            "source_file_set_sha256": hashlib.sha256(
                json.dumps(
                    records,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
        if role == "candidate":
            manifest = tmp_path / "candidate_manifest.json"
            binding.update(
                checkpoint_path=str(checkpoint.resolve()),
                checkpoint_sha256=subject.sha256_file(checkpoint),
                checkpoint_manifest_path=str(manifest.resolve()),
                checkpoint_manifest_sha256=subject.sha256_file(manifest),
            )
        return binding

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
            "baseline_evaluation_aggregate": aggregate_binding("baseline"),
            "candidate_validation_aggregate": aggregate_binding("candidate"),
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


def test_holdout_contract_is_required_for_full_checkpoint_infos(
    tmp_path: Path,
) -> None:
    infos = _checkpoint_infos(tmp_path)
    for field in subject._PHASE_EFFECTIVE_ENTRY_HOLDOUT_FIELDS:
        infos.pop(field)

    with pytest.raises(subject.CheckpointPromotionError, match="holdout acceptance"):
        subject._validate_checkpoint_snapshot_contract(infos, infos)


def test_holdout_contract_matches_manifest_embedded_infos_and_current_file(
    tmp_path: Path,
) -> None:
    infos = _checkpoint_infos(tmp_path)
    subject._validate_checkpoint_snapshot_contract(infos, infos)

    forged_manifest = dict(infos)
    forged_manifest["phase_effective_entry_holdout_acceptance_sha256"] = "0" * 64
    with pytest.raises(
        subject.CheckpointPromotionError,
        match="manifest and embedded infos disagree",
    ):
        subject._validate_checkpoint_snapshot_contract(forged_manifest, infos)

    acceptance = Path(infos["phase_effective_entry_holdout_acceptance_path"])
    acceptance.write_bytes(acceptance.read_bytes() + b" ")
    with pytest.raises(
        subject.CheckpointPromotionError,
        match="current validated proof",
    ):
        subject._validate_checkpoint_snapshot_contract(infos, infos)


@pytest.mark.parametrize("stage", ("smoke", "initial_zero_residual"))
def test_holdout_contract_remains_optional_for_initial_and_smoke_checkpoint_infos(
    tmp_path: Path, stage: str
) -> None:
    infos = _checkpoint_infos(tmp_path, stage=stage)
    subject._validate_checkpoint_snapshot_contract(infos, infos)
    assert not any(
        field in infos for field in subject._PHASE_EFFECTIVE_ENTRY_HOLDOUT_FIELDS
    )
    assert not any(
        field in infos for field in subject._PHASE_ZERO_RESIDUAL_ROLLOUT_FIELDS
    )


def test_zero_residual_rollout_is_required_for_full_checkpoint_infos(
    tmp_path: Path,
) -> None:
    infos = _checkpoint_infos(tmp_path)
    for field in subject._PHASE_ZERO_RESIDUAL_ROLLOUT_FIELDS:
        infos.pop(field)

    with pytest.raises(
        subject.CheckpointPromotionError, match="phase zero-residual rollout"
    ):
        subject._validate_checkpoint_snapshot_contract(infos, infos)


def test_zero_residual_rollout_matches_infos_and_current_files(
    tmp_path: Path,
) -> None:
    infos = _checkpoint_infos(tmp_path)
    subject._validate_checkpoint_snapshot_contract(infos, infos)

    forged_manifest = copy.deepcopy(infos)
    forged_manifest["phase_zero_residual_rollout_run_manifest_sha256"] = "0" * 64
    with pytest.raises(
        subject.CheckpointPromotionError,
        match="manifest and embedded infos disagree on phase zero-residual rollout",
    ):
        subject._validate_checkpoint_snapshot_contract(forged_manifest, infos)

    evidence = Path(infos["phase_zero_residual_rollout_evidence_path"])
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(
        subject.CheckpointPromotionError,
        match="phase zero-residual rollout differs from the current validated proof",
    ):
        subject._validate_checkpoint_snapshot_contract(infos, infos)


def test_offline_checkpoint_validator_rejects_sidecar_retrofit_on_old_checkpoint(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_infos = {
        key: value
        for key, value in manifest.items()
        if not key.startswith("phase_snapshot_")
        and key not in {"checkpoint_path", "checkpoint_sha256"}
    }
    old_infos["files"] = {"training.yaml": "f" * 64}
    torch.save({"infos": old_infos}, checkpoint)
    manifest["checkpoint_sha256"] = subject.sha256_file(checkpoint)
    _json(manifest_path, manifest)

    with pytest.raises(
        subject.CheckpointPromotionError,
        match="embedded infos differ from snapshot-bound sidecar",
    ):
        subject.validate_checkpoint_artifact_provenance(checkpoint, manifest_path)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("stage", "smoke"),
        ("training_seed", 1002),
        ("global_policy_decisions", 100_001),
        ("controller_hash", "9" * 64),
    ),
)
def test_offline_checkpoint_validator_rejects_forged_sidecar_training_provenance(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = forged_value
    _json(manifest_path, manifest)

    with pytest.raises(
        subject.CheckpointPromotionError,
        match="embedded infos differ from snapshot-bound sidecar",
    ):
        subject.validate_checkpoint_artifact_provenance(checkpoint, manifest_path)


def test_offline_checkpoint_validator_requires_all_27_snapshot_file_hashes(
    tmp_path: Path,
) -> None:
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_path = manifest["phase_snapshot_bundle"]["snapshots"][0][
        "snapshot_path"
    ]
    manifest["files"].pop(required_path)
    _json(manifest_path, manifest)

    with pytest.raises(
        subject.CheckpointPromotionError,
        match="phase reset contract files",
    ):
        subject.validate_checkpoint_artifact_provenance(checkpoint, manifest_path)


def test_full_checkpoint_requires_holdout_in_manifest_and_embedded_infos(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in subject._PHASE_EFFECTIVE_ENTRY_HOLDOUT_FIELDS:
        payload["infos"].pop(field)
        manifest.pop(field)
    torch.save(payload, checkpoint)
    manifest["checkpoint_sha256"] = subject.sha256_file(checkpoint)
    _json(manifest_path, manifest)

    with pytest.raises(subject.CheckpointPromotionError, match="holdout acceptance"):
        subject.validate_checkpoint_artifact_provenance(checkpoint, manifest_path)


def test_smoke_checkpoint_remains_compatible_without_holdout(tmp_path: Path) -> None:
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path, stage="smoke")

    evidence = subject.validate_checkpoint_artifact_provenance(
        checkpoint, manifest_path
    )

    assert evidence.manifest["stage"] == "smoke"
    assert not any(
        field in evidence.manifest
        for field in subject._PHASE_EFFECTIVE_ENTRY_HOLDOUT_FIELDS
    )


def test_checkpoint_revalidates_current_holdout_file_hash(tmp_path: Path) -> None:
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = Path(manifest["phase_effective_entry_holdout_acceptance_path"])
    acceptance.write_bytes(acceptance.read_bytes() + b" ")

    with pytest.raises(
        subject.CheckpointPromotionError,
        match="current validated proof",
    ):
        subject.validate_checkpoint_artifact_provenance(checkpoint, manifest_path)


@pytest.mark.parametrize("linked_source", ("checkpoint", "manifest"))
def test_offline_checkpoint_validator_rejects_redirected_inputs(
    tmp_path: Path, linked_source: str
) -> None:
    checkpoint, manifest_path = _checkpoint_evidence(tmp_path)
    target = checkpoint if linked_source == "checkpoint" else manifest_path
    link = tmp_path / f"linked-{target.name}"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    selected_checkpoint = link if linked_source == "checkpoint" else checkpoint
    selected_manifest = link if linked_source == "manifest" else manifest_path
    with pytest.raises(subject.CheckpointPromotionError, match="symlink|redirect"):
        subject.validate_checkpoint_artifact_provenance(
            selected_checkpoint, selected_manifest
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
    assert all(field in best for field in subject._PHASE_SNAPSHOT_CONTRACT_FIELDS)
    assert set(phase_snapshot_bundle_file_hashes(best["phase_snapshot_bundle"])) <= set(
        best["files"]
    )
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
    assert all(field in manifest for field in subject._PHASE_SNAPSHOT_CONTRACT_FIELDS)
    assert set(
        phase_snapshot_bundle_file_hashes(manifest["phase_snapshot_bundle"])
    ) <= set(manifest["files"])
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
    from wlr50_clean.ppo.checkpoint_runtime_capture import capture_checkpoint_bundle

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
    capture_run = tmp_path / "capture-run"
    capture_run.mkdir()
    with capture_checkpoint_bundle(
        improved.improved_checkpoint,
        improved.improved_manifest,
        run_directory=capture_run,
        purpose="unit-inference-export",
    ) as captured:
        artifacts = subject.export_inference_actor(
            TinyRunner(),
            source_checkpoint_path=improved.improved_checkpoint,
            source_manifest_path=improved.improved_manifest,
            output_root=output,
            captured_bundle=captured,
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
    assert evidence["verification_input_float32"]["shape"] == [4, 125]
    assert evidence["verification_input_sha256"] == (
        evidence["verification_input_float32"]["sha256"]
    )
    assert evidence["loaded_runner_reference_output_float32"]["shape"] == [4, 12]
    assert evidence["loaded_runner_reference_output_sha256"] == (
        evidence["loaded_runner_reference_output_float32"]["sha256"]
    )
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
        lambda args, **kwargs: {
            field: manifest[field]
            for field in rl_library_wrapper.CHECKPOINT_RUNTIME_CONTRACT_FIELDS
        },
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_checkpoint_round_trip",
        lambda loaded_runner, loaded_checkpoint, *, captured_bundle: dict(
            captured_bundle.embedded_infos
        ),
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
        _checkpoint_runtime_capture_stack=ExitStack(),
    )

    try:
        assert cli._export_inference_actor(args, object()) == 0
    finally:
        args._checkpoint_runtime_capture_stack.close()
    assert construction["max_iterations"] == 1
    assert construction["reset_seeds"] == (4001,)
    assert construction["collect_trace"] is False
    result = json.loads((run_dir / "inference_actor_export.json").read_text())
    assert result["live_rsl_runner_loaded"] is True
    assert result["episode_stepped"] is False
    assert result["torchscript_actor"].endswith("policy_improved_actor.pt")
