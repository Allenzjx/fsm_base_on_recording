"""Fail-closed checkpoint promotion and inference-only actor export.

Best-validation promotion is authorized solely by the machine-readable paired
validation decision.  The final improved name additionally requires a
separate, finalized five-seed locked-test aggregate.  Names such as ``best``
or ``improved`` carry no authority.  Every source is hash-bound before an
immutable publication is created.

The export helper accepts an already-loaded RSL-RL runner.  It exports only
the deterministic actor head, not the optimizer, critic, rollout state, or
action distribution sampler, and independently reloads each supported format
for shape, finiteness, determinism, and numerical-equivalence checks.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_json, sha256_file


PROMOTION_DECISION_SCHEMA = "wlr50_clean.ppo_evaluation_artifacts.v1"
CHECKPOINT_MANIFEST_SCHEMA = "wlr50_clean.phase_residual_checkpoint_manifest.v1"
CHECKPOINT_VALIDATION_PROMOTION_SCHEMA = (
    "wlr50_clean.ppo_checkpoint_validation_promotion.v1"
)
CHECKPOINT_IMPROVED_PROMOTION_SCHEMA = "wlr50_clean.ppo_checkpoint_improved_promotion.v1"
INFERENCE_EXPORT_SCHEMA = "wlr50_clean.ppo_inference_actor_export.v1"
LOCKED_TEST_AGGREGATE_SCHEMA = "wlr50_clean.fresh_process_episode_batch.v1"

BEST_CHECKPOINT_NAME = "checkpoint_best_validation.pt"
IMPROVED_CHECKPOINT_NAME = "checkpoint_improved.pt"
BEST_MANIFEST_NAME = "checkpoint_best_validation_manifest.json"
IMPROVED_MANIFEST_NAME = "checkpoint_improved_manifest.json"
VALIDATION_PROMOTION_MANIFEST_NAME = (
    "checkpoint_best_validation_promotion_manifest.json"
)
PROMOTION_MANIFEST_NAME = "checkpoint_promotion_manifest.json"
TORCHSCRIPT_ACTOR_NAME = "policy_improved_actor.pt"
ONNX_ACTOR_NAME = "policy_improved_actor.onnx"
INFERENCE_EXPORT_MANIFEST_NAME = "inference_actor_export_manifest.json"

REQUIRED_PROMOTION_GATES = (
    "p01_p13_completed",
    "task_success_rate_not_below_fsm",
    "body_collision_zero",
    "wheel_only_climb_zero",
    "fall_or_physics_explosion_zero",
    "safety_abort_zero",
    "duration_each_under_200_s",
    "duration_not_over_fsm_by_15pct",
    "frozen_hashes_unchanged",
    "recording_runtime_access_zero",
    "global_stability_improvement_at_least_5pct",
    "at_least_4_of_5_priority_phases_improve",
    "no_priority_phase_degrades_over_10pct",
    "one_visual_key_metric_gate",
    "level_calibration_quality_passed",
    "residual_activity_calibrated",
    "priority_phases_have_real_residual",
    "at_least_10_phases_have_real_residual",
)

VALIDATION_SEEDS = (2001, 2002, 2003, 2004, 2005)
LOCKED_TEST_SEEDS = (3001, 3002, 3003, 3004, 3005)
FROZEN_HASH_FIELDS = (
    "controller_hash",
    "environment_hash",
    "observation_schema_hash",
    "action_schema_hash",
    "reward_config_hash",
)
REQUIRED_LOCKED_TEST_HASH_GATES = (
    "checkpoint_matches_best_validation",
    "checkpoint_manifest_matches_best_validation",
    "controller_hash_unchanged",
    "environment_hash_unchanged",
    "observation_schema_hash_unchanged",
    "action_schema_hash_unchanged",
    "reward_config_hash_unchanged",
    "worker_artifact_hashes_verified",
)
_RUN_CONFIG_PATH_BY_HASH_FIELD = {
    "observation_schema_hash": "configs/ppo_observation_schema_v2.json",
    "action_schema_hash": "configs/ppo_phase_action_masks_v2.yaml",
    "reward_config_hash": "configs/ppo_reward_v2.yaml",
}
_FROZEN_PATH_BY_HASH_FIELD = {
    "controller_hash": "configs/fsm_states.yaml",
    "environment_hash": "configs/environment_lock.json",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CheckpointPromotionError(RuntimeError):
    """Checkpoint evidence is incomplete, inconsistent, or not promotable."""


@dataclass(frozen=True, slots=True)
class ValidationPromotionArtifacts:
    best_checkpoint: Path
    best_manifest: Path
    validation_promotion_manifest: Path


@dataclass(frozen=True, slots=True)
class ImprovedPromotionArtifacts:
    improved_checkpoint: Path
    improved_manifest: Path
    promotion_manifest: Path


@dataclass(frozen=True, slots=True)
class InferenceActorArtifacts:
    torchscript_actor: Path
    onnx_actor: Path | None
    export_manifest: Path
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CheckpointArtifactProvenance:
    """Offline binding between checkpoint bytes and their immutable sidecar."""

    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class _CheckpointEvidence:
    checkpoint: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    observation_dimension: int
    action_dimension: int


@dataclass(frozen=True, slots=True)
class _LockedTestEvidence:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    worker_artifact_sha256: tuple[Mapping[str, str], ...]


def _load_json_snapshot(
    path: Path | str, *, label: str
) -> tuple[Path, Mapping[str, Any], str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise CheckpointPromotionError(f"{label} is missing: {source}")
    try:
        encoded = source.read_bytes()
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointPromotionError(f"{label} is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(value, Mapping):
        raise CheckpointPromotionError(f"{label} must contain a JSON object")
    return source, value, hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path | str, *, label: str) -> Mapping[str, Any]:
    return _load_json_snapshot(path, label=label)[1]


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise CheckpointPromotionError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise CheckpointPromotionError(f"{label} is not finite")
    return result


def _hash(value: Any, *, label: str) -> str:
    result = str(value or "").lower()
    if _SHA256.fullmatch(result) is None:
        raise CheckpointPromotionError(f"{label} is not a SHA-256 digest")
    return result


def _validate_checkpoint(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
) -> _CheckpointEvidence:
    checkpoint = Path(checkpoint_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise CheckpointPromotionError(f"candidate checkpoint is missing or empty: {checkpoint}")
    manifest_file, manifest, manifest_hash = _load_json_snapshot(
        manifest_file, label="candidate checkpoint manifest"
    )
    if manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise CheckpointPromotionError("candidate checkpoint manifest has the wrong schema")
    declared_path = Path(str(manifest.get("checkpoint_path", ""))).resolve()
    if declared_path != checkpoint:
        raise CheckpointPromotionError("checkpoint manifest is bound to a different file")
    actual_hash = sha256_file(checkpoint)
    if _hash(manifest.get("checkpoint_sha256"), label="manifest checkpoint hash") != actual_hash:
        raise CheckpointPromotionError("checkpoint bytes do not match the checkpoint manifest")
    try:
        observation_dimension = int(manifest["actor_observation_dimension"])
        critic_dimension = int(manifest["critic_observation_dimension"])
        action_dimension = int(manifest["residual_dimension"])
        global_decisions = int(manifest["global_policy_decisions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointPromotionError("checkpoint manifest dimensions/budget are incomplete") from exc
    if observation_dimension != 125 or critic_dimension != 125 or action_dimension != 12:
        raise CheckpointPromotionError("checkpoint manifest does not describe the 125D-to-Full12 policy")
    if global_decisions <= 0 or str(manifest.get("stage", "")) == "initial_zero_residual":
        raise CheckpointPromotionError("initial/untrained checkpoints cannot be promoted")
    if not math.isclose(
        _finite(manifest.get("physics_hz"), label="checkpoint physics_hz"),
        120.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        _finite(manifest.get("decision_hz"), label="checkpoint decision_hz"),
        15.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CheckpointPromotionError("checkpoint manifest timing is not 120/15 Hz")
    for key in FROZEN_HASH_FIELDS:
        _hash(manifest.get(key), label=f"checkpoint {key}")
    return _CheckpointEvidence(
        checkpoint=checkpoint,
        checkpoint_sha256=actual_hash,
        manifest_path=manifest_file,
        manifest_sha256=manifest_hash,
        manifest=manifest,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
    )


def validate_checkpoint_artifact_provenance(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
) -> CheckpointArtifactProvenance:
    """Validate an offline checkpoint/sidecar pair without loading Isaac or Torch.

    This is the offline counterpart to ``validate_resume_checkpoint_provenance``:
    it cannot inspect RSL's embedded ``infos`` without loading the runner, but it
    strictly binds the actual checkpoint bytes to the declared sidecar path and
    SHA-256 while enforcing the complete phase-residual policy ABI.
    """

    evidence = _validate_checkpoint(checkpoint_path, manifest_path)
    return CheckpointArtifactProvenance(
        checkpoint_path=evidence.checkpoint,
        checkpoint_sha256=evidence.checkpoint_sha256,
        manifest_path=evidence.manifest_path,
        manifest_sha256=evidence.manifest_sha256,
        manifest=dict(evidence.manifest),
    )


def _validate_promotion_decision(
    decision_path: Path | str,
    checkpoint: _CheckpointEvidence,
) -> tuple[Path, Mapping[str, Any], str]:
    path, decision, decision_hash = _load_json_snapshot(
        decision_path, label="promotion decision"
    )
    if decision.get("schema") != PROMOTION_DECISION_SCHEMA:
        raise CheckpointPromotionError("promotion decision has the wrong schema")
    if decision.get("baseline_checkpoint") != "pure_fsm":
        raise CheckpointPromotionError("promotion decision baseline is not pure_fsm")
    promotion = decision.get("promotion")
    if not isinstance(promotion, Mapping):
        raise CheckpointPromotionError("promotion decision omits its promotion object")
    if promotion.get("promoted") is not True:
        raise CheckpointPromotionError("paired evaluation did not authorize checkpoint promotion")
    if promotion.get("first_failed_gate") is not None or decision.get("first_failed_gate") is not None:
        raise CheckpointPromotionError("promoted decision still names a failed gate")
    checks = promotion.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_PROMOTION_GATES):
        raise CheckpointPromotionError("promotion decision does not contain the complete gate set")
    if any(checks[gate] is not True for gate in REQUIRED_PROMOTION_GATES):
        raise CheckpointPromotionError("promotion decision contains a failed or non-boolean gate")
    ordered = decision.get("checks_in_evaluation_order")
    if not isinstance(ordered, list) or len(ordered) != len(REQUIRED_PROMOTION_GATES):
        raise CheckpointPromotionError("promotion decision lacks ordered gate evidence")
    ordered_names: list[str] = []
    for row in ordered:
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise CheckpointPromotionError("ordered promotion gate evidence contains a failure")
        ordered_names.append(str(row.get("gate", "")))
    if tuple(ordered_names) != REQUIRED_PROMOTION_GATES:
        raise CheckpointPromotionError("promotion gates are missing, duplicated, or reordered")

    improvement = _finite(
        promotion.get("global_stability_improvement_fraction"),
        label="global stability improvement",
    )
    try:
        improved_priority = int(promotion["improved_priority_phase_count"])
        paired_count = int(decision["paired_episode_count"])
        minimum_count = int(decision["minimum_paired_seeds"])
        paired_seeds = tuple(int(seed) for seed in decision["paired_seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointPromotionError("promotion episode/phase evidence is incomplete") from exc
    if improvement < 0.05 or improved_priority < 4:
        raise CheckpointPromotionError("promotion summary is below the required improvement threshold")
    if (
        minimum_count != len(VALIDATION_SEEDS)
        or paired_count != len(VALIDATION_SEEDS)
        or paired_seeds != VALIDATION_SEEDS
    ):
        raise CheckpointPromotionError(
            "best-validation promotion requires exactly validation seeds 2001-2005"
        )
    if decision.get("frozen_hashes_unchanged") is not True:
        raise CheckpointPromotionError("promotion decision does not preserve frozen FSM hashes")

    declared_path = Path(str(decision.get("candidate_checkpoint_path", ""))).resolve()
    if declared_path != checkpoint.checkpoint:
        raise CheckpointPromotionError("promotion decision is bound to a different checkpoint")
    if _hash(
        decision.get("candidate_checkpoint_sha256"), label="promotion candidate hash"
    ) != checkpoint.checkpoint_sha256:
        raise CheckpointPromotionError("promotion decision checkpoint hash mismatch")
    return path, decision, decision_hash


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _publish_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise CheckpointPromotionError(f"refusing to overwrite artifact: {destination}") from exc
    except OSError as exc:
        raise CheckpointPromotionError(f"cannot atomically publish artifact: {destination}") from exc


def _publish_bundle_no_replace(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Publish each file atomically and roll back files from a partial bundle."""

    published: list[Path] = []
    try:
        for source, destination in pairs:
            _publish_no_replace(source, destination)
            published.append(destination)
    except Exception:
        cleanup_failures: list[str] = []
        for destination in reversed(published):
            try:
                destination.unlink()
            except OSError as exc:
                cleanup_failures.append(f"{destination}: {exc}")
        if cleanup_failures:
            raise CheckpointPromotionError(
                "checkpoint publication failed and rollback was incomplete: "
                + "; ".join(cleanup_failures)
            )
        raise


def _records_by_path(value: Any, *, label: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise CheckpointPromotionError(f"{label} must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise CheckpointPromotionError(f"{label} contains a non-object row")
        path = str(row.get("path", "")).replace("\\", "/")
        if not path or path in records:
            raise CheckpointPromotionError(f"{label} paths must be non-empty and unique")
        records[path] = row
    return records


def _frozen_audit_hashes(
    audit: Mapping[str, Any], *, label: str
) -> Mapping[str, str]:
    if (
        audit.get("schema") != "wlr50_clean.frozen_fsm_hash_audit.v1"
        or audit.get("passed") is not True
        or audit.get("mismatches") != []
    ):
        raise CheckpointPromotionError(f"{label} did not pass")
    rows = _records_by_path(audit.get("entries"), label=f"{label} entries")
    result: dict[str, str] = {}
    for field, relative_path in _FROZEN_PATH_BY_HASH_FIELD.items():
        row = rows.get(relative_path)
        if (
            not isinstance(row, Mapping)
            or row.get("exists") is not True
            or row.get("valid") is not True
        ):
            raise CheckpointPromotionError(f"{label} omits valid {relative_path}")
        expected = _hash(row.get("expected_sha256"), label=f"{label} expected hash")
        actual = _hash(row.get("actual_sha256"), label=f"{label} actual hash")
        if expected != actual:
            raise CheckpointPromotionError(f"{label} hash mismatch for {relative_path}")
        result[field] = actual
    return result


def finalize_locked_test_aggregate_payload(
    aggregate_payload: Mapping[str, Any],
    *,
    checkpoint_path: Path | str,
    checkpoint_manifest_path: Path | str,
) -> Mapping[str, Any]:
    """Add independently recomputed hash gates to a completed locked-test batch.

    This function is deliberately offline.  It follows each aggregate worker's
    immutable paths, re-hashes its lifecycle/result/trial and frozen-hash audit
    files, and compares runtime config provenance with the promoted validation
    checkpoint manifest.  Physical failures remain publishable evidence with
    ``passed=false``; malformed or unbound provenance raises.
    """

    payload = copy.deepcopy(dict(aggregate_payload))
    if payload.get("schema") != LOCKED_TEST_AGGREGATE_SCHEMA:
        raise CheckpointPromotionError("locked-test aggregate has the wrong schema")
    if payload.get("role") != "candidate" or payload.get("seed_set") != "locked-test":
        raise CheckpointPromotionError("only a candidate locked-test aggregate can be finalized")
    checkpoint = _validate_checkpoint(checkpoint_path, checkpoint_manifest_path)
    if (
        checkpoint.manifest.get("publication_role") != "best_validation"
        or checkpoint.manifest.get("validation_promotion_authorized") is not True
        or checkpoint.manifest.get("locked_test_authorized") is not False
    ):
        raise CheckpointPromotionError(
            "locked-test aggregation requires the validation-promoted best checkpoint"
        )
    if (
        Path(str(payload.get("checkpoint", ""))).resolve() != checkpoint.checkpoint
        or _hash(payload.get("checkpoint_sha256"), label="aggregate checkpoint hash")
        != checkpoint.checkpoint_sha256
    ):
        raise CheckpointPromotionError("aggregate checkpoint differs from best_validation")
    try:
        seeds = tuple(int(seed) for seed in payload.get("seeds", ()))
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError("locked-test aggregate seeds are invalid") from exc
    if seeds != LOCKED_TEST_SEEDS:
        raise CheckpointPromotionError("locked-test finalization requires seeds 3001-3005")
    workers = payload.get("workers")
    episodes = payload.get("episodes")
    canonical_episode_dirs = payload.get("canonical_episode_dirs")
    if (
        not isinstance(workers, list)
        or not isinstance(episodes, list)
        or not isinstance(canonical_episode_dirs, list)
        or len(workers) != len(LOCKED_TEST_SEEDS)
        or len(episodes) != len(LOCKED_TEST_SEEDS)
        or len(canonical_episode_dirs) != len(LOCKED_TEST_SEEDS)
    ):
        raise CheckpointPromotionError(
            "locked-test finalization requires five workers and five episodes"
        )
    for key in (
        "fresh_process_per_episode",
        "deterministic_evaluation",
        "deterministic_mean_policy",
    ):
        _require_boolean(payload, key, True, "locked-test aggregate")

    success_count = 0
    body_collision_count = 0
    wheel_only_climb_count = 0
    safety_abort_count = 0
    all_under_duration = True
    runtime_clean = True
    for seed, episode in zip(LOCKED_TEST_SEEDS, episodes, strict=True):
        if not isinstance(episode, Mapping):
            raise CheckpointPromotionError(f"locked-test episode {seed} is malformed")
        try:
            episode_seed = int(episode.get("seed", -1))
            duration = float(episode.get("duration_s", math.nan))
            recording_reads = int(episode.get("recording_runtime_access_count", -1))
            root_writes = int(episode.get("in_episode_root_write_count", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointPromotionError(
                f"locked-test episode {seed} gate values are invalid"
            ) from exc
        if episode_seed != seed:
            raise CheckpointPromotionError(f"locked-test episode {seed} seed is invalid")
        success_count += episode.get("task_success") is True
        body_collision_count += episode.get("body_collision") is True
        wheel_only_climb_count += episode.get("wheel_only_climb") is True
        safety_abort_count += episode.get("safety_abort") is True
        all_under_duration = bool(
            all_under_duration
            and episode.get("under_maximum_duration") is True
            and math.isfinite(duration)
            and 0.0 < duration <= 200.0
        )
        runtime_clean = bool(
            runtime_clean and recording_reads == 0 and root_writes == 0
        )

    field_matches: dict[str, list[bool]] = {
        field: [] for field in FROZEN_HASH_FIELDS
    }
    artifact_hashes_verified: list[bool] = []
    enriched_workers: list[Mapping[str, Any]] = []
    frozen_manifest_hash: str | None = None
    for index, (seed, raw_worker) in enumerate(
        zip(LOCKED_TEST_SEEDS, workers, strict=True)
    ):
        if not isinstance(raw_worker, Mapping):
            raise CheckpointPromotionError(f"locked-test worker {seed} identity is invalid")
        try:
            worker_seed = int(raw_worker.get("seed", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointPromotionError(
                f"locked-test worker {seed} identity is invalid"
            ) from exc
        if worker_seed != seed or raw_worker.get("role") != "candidate":
            raise CheckpointPromotionError(f"locked-test worker {seed} identity is invalid")
        worker = dict(raw_worker)
        run_dir = Path(str(worker.get("run_dir", ""))).resolve()
        result_path = Path(str(worker.get("worker_result", ""))).resolve()
        episode_dir = Path(str(worker.get("canonical_episode_dir", ""))).resolve()
        paths = {
            "run_manifest_sha256": run_dir / "run_manifest.json",
            "worker_result_sha256": result_path,
            "trial_manifest_sha256": episode_dir / "trial_manifest.json",
            "frozen_hashes_before_sha256": run_dir / "frozen_hashes.before.json",
            "frozen_hashes_after_sha256": run_dir / "frozen_hashes.after.json",
        }
        if (
            result_path != run_dir / "checkpoint_evaluation.json"
            or episode_dir.parent != run_dir
            or Path(str(canonical_episode_dirs[index])).resolve() != episode_dir
            or Path(str(episodes[index].get("canonical_episode_dir", ""))).resolve()
            != episode_dir
        ):
            raise CheckpointPromotionError(f"locked-test worker {seed} paths are invalid")
        recomputed: dict[str, str] = {}
        for key, evidence_path in paths.items():
            if not evidence_path.is_file():
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} evidence is missing: {evidence_path}"
                )
            actual_hash = sha256_file(evidence_path)
            if key in worker and _hash(
                worker[key], label=f"worker {seed} declared {key}"
            ) != actual_hash:
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} declared {key} is stale"
                )
            worker[key] = actual_hash
            recomputed[key] = actual_hash
        worker["frozen_hashes_before"] = str(paths["frozen_hashes_before_sha256"])
        worker["frozen_hashes_after"] = str(paths["frozen_hashes_after_sha256"])
        worker["all_artifact_hashes_recomputed"] = True
        artifact_hashes_verified.append(True)

        _, lifecycle, lifecycle_hash = _load_json_snapshot(
            paths["run_manifest_sha256"], label=f"worker {seed} lifecycle"
        )
        if lifecycle_hash != recomputed["run_manifest_sha256"]:
            raise CheckpointPromotionError(f"locked-test worker {seed} lifecycle changed")
        if lifecycle.get("lifecycle") != "SUCCEEDED" or lifecycle.get("exit_code") != 0:
            raise CheckpointPromotionError(f"locked-test worker {seed} was not finalized")
        configs = _records_by_path(
            lifecycle.get("configs"), label=f"worker {seed} config records"
        )
        for field, relative_path in _RUN_CONFIG_PATH_BY_HASH_FIELD.items():
            row = configs.get(relative_path)
            actual = None if row is None else row.get("sha256")
            field_matches[field].append(
                isinstance(actual, str)
                and _SHA256.fullmatch(actual.lower()) is not None
                and actual.lower() == str(checkpoint.manifest[field]).lower()
            )

        _, result, result_hash = _load_json_snapshot(
            result_path, label=f"worker {seed} evaluation"
        )
        if result_hash != recomputed["worker_result_sha256"]:
            raise CheckpointPromotionError(f"locked-test worker {seed} result changed")
        infos = result.get("checkpoint_infos")
        if not isinstance(infos, Mapping):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} omits checkpoint provenance"
            )
        for field in FROZEN_HASH_FIELDS:
            field_matches[field].append(
                str(infos.get(field, "")).lower()
                == str(checkpoint.manifest[field]).lower()
            )

        audit_hashes: list[Mapping[str, str]] = []
        for label, audit_path, hash_key in (
            (
                f"worker {seed} frozen pre-audit",
                paths["frozen_hashes_before_sha256"],
                "frozen_hashes_before_sha256",
            ),
            (
                f"worker {seed} frozen post-audit",
                paths["frozen_hashes_after_sha256"],
                "frozen_hashes_after_sha256",
            ),
        ):
            _, audit, audit_file_hash = _load_json_snapshot(audit_path, label=label)
            if audit_file_hash != recomputed[hash_key]:
                raise CheckpointPromotionError(f"{label} changed during finalization")
            audit_manifest_hash = _hash(
                audit.get("frozen_manifest_sha256"),
                label=f"{label} frozen manifest hash",
            )
            if frozen_manifest_hash is None:
                frozen_manifest_hash = audit_manifest_hash
            elif frozen_manifest_hash != audit_manifest_hash:
                raise CheckpointPromotionError(
                    "locked-test workers used different frozen hash manifests"
                )
            audit_hashes.append(_frozen_audit_hashes(audit, label=label))
        for field in _FROZEN_PATH_BY_HASH_FIELD:
            for audit_hash in audit_hashes:
                field_matches[field].append(
                    audit_hash[field] == str(checkpoint.manifest[field]).lower()
                )
        enriched_workers.append(worker)

    field_gates = {
        f"{field.removesuffix('_hash')}_hash_unchanged": all(field_matches[field])
        for field in FROZEN_HASH_FIELDS
    }
    # Keep the externally documented gate names stable.
    field_gates = {
        "controller_hash_unchanged": field_gates["controller_hash_unchanged"],
        "environment_hash_unchanged": field_gates["environment_hash_unchanged"],
        "observation_schema_hash_unchanged": field_gates[
            "observation_schema_hash_unchanged"
        ],
        "action_schema_hash_unchanged": field_gates["action_schema_hash_unchanged"],
        "reward_config_hash_unchanged": field_gates["reward_config_hash_unchanged"],
    }
    hash_gates = {
        "checkpoint_matches_best_validation": True,
        "checkpoint_manifest_matches_best_validation": True,
        **field_gates,
        "worker_artifact_hashes_verified": all(artifact_hashes_verified),
    }
    if set(hash_gates) != set(REQUIRED_LOCKED_TEST_HASH_GATES):
        raise CheckpointPromotionError("internal locked-test hash gate set is incomplete")
    worker_gate_pass_count = sum(
        worker.get("worker_gate_passed") is True for worker in enriched_workers
    )
    physical_gates_passed = bool(
        success_count == len(LOCKED_TEST_SEEDS)
        and body_collision_count == 0
        and wheel_only_climb_count == 0
        and safety_abort_count == 0
        and all_under_duration
        and runtime_clean
        and worker_gate_pass_count == len(LOCKED_TEST_SEEDS)
    )
    payload.update(
        {
            "finalized": True,
            "checkpoint_manifest": str(checkpoint.manifest_path),
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "frozen_hashes_unchanged": all(field_gates.values()),
            "frozen_hashes": {
                field: str(checkpoint.manifest[field]).lower()
                for field in FROZEN_HASH_FIELDS
            },
            "frozen_manifest_sha256": frozen_manifest_hash,
            "hash_gates": hash_gates,
            "workers": enriched_workers,
            "worker_artifact_hash_algorithm": "sha256",
            "worker_artifact_hashes_recomputed": True,
            "episode_count": len(episodes),
            "success_count": success_count,
            "body_collision_count": body_collision_count,
            "wheel_only_climb_count": wheel_only_climb_count,
            "safety_abort_count": safety_abort_count,
            "all_under_maximum_duration": all_under_duration,
            "worker_gate_pass_count": worker_gate_pass_count,
            "physical_gates_passed": physical_gates_passed,
            "passed": physical_gates_passed and all(hash_gates.values()),
        }
    )
    return payload


def promote_best_validation_checkpoint(
    *,
    promotion_decision_path: Path | str,
    candidate_checkpoint_path: Path | str,
    candidate_manifest_path: Path | str,
    output_root: Path | str,
) -> ValidationPromotionArtifacts:
    """Publish only the validation-selected checkpoint.

    A validation decision can never create ``checkpoint_improved.pt``.  That
    name is reserved for :func:`promote_improved_checkpoint`, after a separate
    finalized locked-test aggregate has passed.
    """

    checkpoint = _validate_checkpoint(candidate_checkpoint_path, candidate_manifest_path)
    decision_path, decision, decision_hash = _validate_promotion_decision(
        promotion_decision_path, checkpoint
    )
    root = Path(output_root).resolve()
    checkpoint_dir = root / "checkpoints"
    manifest_dir = root / "manifests"
    artifacts = ValidationPromotionArtifacts(
        best_checkpoint=checkpoint_dir / BEST_CHECKPOINT_NAME,
        best_manifest=checkpoint_dir / BEST_MANIFEST_NAME,
        validation_promotion_manifest=(
            manifest_dir / VALIDATION_PROMOTION_MANIFEST_NAME
        ),
    )
    destinations = (
        artifacts.best_checkpoint,
        artifacts.best_manifest,
        artifacts.validation_promotion_manifest,
        checkpoint_dir / IMPROVED_CHECKPOINT_NAME,
        checkpoint_dir / IMPROVED_MANIFEST_NAME,
        manifest_dir / PROMOTION_MANIFEST_NAME,
    )
    conflict = next((path for path in destinations if path.exists()), None)
    if conflict is not None:
        raise CheckpointPromotionError(f"refusing to overwrite artifact: {conflict}")

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".validation-promotion-", dir=str(root)))
    try:
        stage_checkpoints = staging / "checkpoints"
        stage_manifests = staging / "manifests"
        stage_checkpoints.mkdir(parents=True)
        stage_manifests.mkdir(parents=True)
        stage_best = stage_checkpoints / BEST_CHECKPOINT_NAME
        _copy_file(checkpoint.checkpoint, stage_best)
        if sha256_file(stage_best) != checkpoint.checkpoint_sha256:
            raise CheckpointPromotionError("staged checkpoint copy failed hash verification")

        stage_best_manifest = stage_checkpoints / BEST_MANIFEST_NAME
        atomic_write_json(
            stage_best_manifest,
            {
                **dict(checkpoint.manifest),
                "source_checkpoint_path": str(checkpoint.checkpoint),
                "source_checkpoint_sha256": checkpoint.checkpoint_sha256,
                "source_checkpoint_manifest": str(checkpoint.manifest_path),
                "source_checkpoint_manifest_sha256": checkpoint.manifest_sha256,
                "promotion_decision": str(decision_path),
                "promotion_decision_sha256": decision_hash,
                "validation_seeds": list(VALIDATION_SEEDS),
                "validation_promotion": dict(decision["promotion"]),
                "validation_promotion_authorized": True,
                "locked_test_authorized": False,
                "publication_role": "best_validation",
                "checkpoint_path": str(artifacts.best_checkpoint),
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            },
        )
        best_manifest_hash = sha256_file(stage_best_manifest)
        validation_payload = {
            "schema": CHECKPOINT_VALIDATION_PROMOTION_SCHEMA,
            "valid": True,
            "status": "PROMOTED_VALIDATION",
            "promotion_scope": "best_validation_only",
            "improved_checkpoint_authorized": False,
            "promotion_authority": "promotion_decision.json content and SHA-256 binding",
            "filename_inference_used": False,
            "source_checkpoint": str(checkpoint.checkpoint),
            "source_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "source_manifest": str(checkpoint.manifest_path),
            "source_manifest_sha256": checkpoint.manifest_sha256,
            "promotion_decision": str(decision_path),
            "promotion_decision_sha256": decision_hash,
            "validation_seeds": list(VALIDATION_SEEDS),
            "promotion": dict(decision["promotion"]),
            "published_best_validation": {
                "path": str(artifacts.best_checkpoint),
                "sha256": checkpoint.checkpoint_sha256,
                "manifest": str(artifacts.best_manifest),
                "manifest_sha256": best_manifest_hash,
            },
            "byte_identical_to_source": True,
            "immutable_no_overwrite": True,
        }
        stage_validation_manifest = stage_manifests / VALIDATION_PROMOTION_MANIFEST_NAME
        atomic_write_json(stage_validation_manifest, validation_payload)

        if (
            sha256_file(checkpoint.checkpoint) != checkpoint.checkpoint_sha256
            or sha256_file(checkpoint.manifest_path) != checkpoint.manifest_sha256
            or sha256_file(decision_path) != decision_hash
        ):
            raise CheckpointPromotionError("validation promotion input changed during publication")
        _publish_bundle_no_replace(
            (
                (stage_best, artifacts.best_checkpoint),
                (stage_best_manifest, artifacts.best_manifest),
                (stage_validation_manifest, artifacts.validation_promotion_manifest),
            )
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if (
        sha256_file(artifacts.best_checkpoint) != checkpoint.checkpoint_sha256
        or sha256_file(artifacts.best_manifest)
        != validation_payload["published_best_validation"]["manifest_sha256"]
    ):
        raise CheckpointPromotionError("published best-validation artifact hash mismatch")
    return artifacts


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_boolean(payload: Mapping[str, Any], key: str, expected: bool, label: str) -> None:
    if payload.get(key) is not expected:
        raise CheckpointPromotionError(f"{label} requires {key}={str(expected).lower()}")


def _validate_best_validation_source(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    validation_promotion_manifest_path: Path | str,
    promotion_decision_path: Path | str,
) -> tuple[_CheckpointEvidence, Path, Mapping[str, Any], str]:
    checkpoint = _validate_checkpoint(checkpoint_path, manifest_path)
    if checkpoint.manifest.get("publication_role") != "best_validation":
        raise CheckpointPromotionError("locked-test promotion source is not best_validation")
    _require_boolean(
        checkpoint.manifest,
        "validation_promotion_authorized",
        True,
        "best-validation manifest",
    )
    _require_boolean(
        checkpoint.manifest,
        "locked_test_authorized",
        False,
        "best-validation manifest",
    )
    validation_path, validation, validation_hash = _load_json_snapshot(
        validation_promotion_manifest_path,
        label="best-validation promotion manifest",
    )
    if (
        validation.get("schema") != CHECKPOINT_VALIDATION_PROMOTION_SCHEMA
        or validation.get("valid") is not True
        or validation.get("status") != "PROMOTED_VALIDATION"
        or validation.get("promotion_scope") != "best_validation_only"
        or validation.get("improved_checkpoint_authorized") is not False
        or validation.get("filename_inference_used") is not False
    ):
        raise CheckpointPromotionError("best-validation promotion manifest is invalid")
    try:
        validation_seeds = tuple(int(seed) for seed in validation.get("validation_seeds", ()))
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError("best-validation seed evidence is invalid") from exc
    if validation_seeds != VALIDATION_SEEDS:
        raise CheckpointPromotionError("best-validation evidence lacks seeds 2001-2005")
    promotion = validation.get("promotion")
    if not isinstance(promotion, Mapping) or promotion.get("promoted") is not True:
        raise CheckpointPromotionError("best-validation evidence does not prove promotion")
    checks = promotion.get("checks")
    try:
        improved_priority_count = int(
            promotion.get("improved_priority_phase_count", -1)
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError(
            "best-validation improved-priority count is invalid"
        ) from exc
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(REQUIRED_PROMOTION_GATES)
        or any(checks[gate] is not True for gate in REQUIRED_PROMOTION_GATES)
        or promotion.get("first_failed_gate") is not None
        or _finite(
            promotion.get("global_stability_improvement_fraction"),
            label="validation stability improvement",
        )
        < 0.05
        or improved_priority_count < 4
    ):
        raise CheckpointPromotionError("best-validation promotion gate evidence is invalid")
    if (
        checkpoint.manifest.get("validation_seeds") != list(VALIDATION_SEEDS)
        or checkpoint.manifest.get("validation_promotion") != promotion
    ):
        raise CheckpointPromotionError(
            "best-validation checkpoint manifest and promotion evidence differ"
        )
    if (
        Path(str(validation.get("promotion_decision", ""))).resolve()
        != Path(str(checkpoint.manifest.get("promotion_decision", ""))).resolve()
        or _hash(
            validation.get("promotion_decision_sha256"),
            label="validation promotion decision hash",
        )
        != _hash(
            checkpoint.manifest.get("promotion_decision_sha256"),
            label="best manifest promotion decision hash",
        )
    ):
        raise CheckpointPromotionError("best-validation decision provenance is inconsistent")
    if _hash(
        validation.get("source_checkpoint_sha256"),
        label="validation source checkpoint hash",
    ) != checkpoint.checkpoint_sha256:
        raise CheckpointPromotionError("validation source and published checkpoint differ")
    decision_path, decision, decision_hash = _load_json_snapshot(
        promotion_decision_path, label="promotion decision"
    )
    if (
        decision_path
        != Path(str(validation.get("promotion_decision", ""))).resolve()
        or decision_hash
        != _hash(
            validation.get("promotion_decision_sha256"),
            label="validation promotion decision hash",
        )
        or decision.get("schema") != PROMOTION_DECISION_SCHEMA
        or decision.get("baseline_checkpoint") != "pure_fsm"
        or decision.get("paired_seeds") != list(VALIDATION_SEEDS)
        or decision.get("promotion") != promotion
        or _hash(
            decision.get("candidate_checkpoint_sha256"),
            label="decision candidate checkpoint hash",
        )
        != checkpoint.checkpoint_sha256
    ):
        raise CheckpointPromotionError(
            "best-validation evidence does not match the supplied promotion decision"
        )
    published = validation.get("published_best_validation")
    if not isinstance(published, Mapping):
        raise CheckpointPromotionError("best-validation publication evidence is missing")
    if Path(str(published.get("path", ""))).resolve() != checkpoint.checkpoint:
        raise CheckpointPromotionError("validation evidence names a different best checkpoint")
    if _hash(published.get("sha256"), label="best-validation checkpoint hash") != (
        checkpoint.checkpoint_sha256
    ):
        raise CheckpointPromotionError("validation evidence checkpoint hash mismatch")
    if Path(str(published.get("manifest", ""))).resolve() != checkpoint.manifest_path:
        raise CheckpointPromotionError("validation evidence names a different best manifest")
    if _hash(
        published.get("manifest_sha256"), label="best-validation manifest hash"
    ) != checkpoint.manifest_sha256:
        raise CheckpointPromotionError("validation evidence manifest hash mismatch")
    return checkpoint, validation_path, validation, validation_hash


def _validate_locked_episode(
    episode: Mapping[str, Any], *, seed: int, label: str
) -> None:
    try:
        episode_seed = int(episode.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError(f"{label} seed is invalid") from exc
    if episode_seed != seed:
        raise CheckpointPromotionError(f"{label} seed differs from locked-test seed {seed}")
    for key, expected in (
        ("task_success", True),
        ("body_collision", False),
        ("wheel_only_climb", False),
        ("safety_abort", False),
        ("under_maximum_duration", True),
    ):
        _require_boolean(episode, key, expected, label)
    duration = _finite(episode.get("duration_s"), label=f"{label} duration_s")
    if duration <= 0.0 or duration > 200.0:
        raise CheckpointPromotionError(f"{label} duration is outside (0, 200] seconds")
    for key in ("recording_runtime_access_count", "in_episode_root_write_count"):
        try:
            count = int(episode.get(key, -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointPromotionError(f"{label} {key} is invalid") from exc
        if count != 0:
            raise CheckpointPromotionError(f"{label} requires {key}=0")


def _validate_locked_test_aggregate(
    aggregate_path: Path | str,
    checkpoint: _CheckpointEvidence,
) -> _LockedTestEvidence:
    path, payload, aggregate_hash = _load_json_snapshot(
        aggregate_path, label="locked-test aggregate"
    )
    if payload.get("schema") != LOCKED_TEST_AGGREGATE_SCHEMA:
        raise CheckpointPromotionError("locked-test aggregate has the wrong schema")
    if payload.get("role") != "candidate" or payload.get("seed_set") != "locked-test":
        raise CheckpointPromotionError("aggregate is not a candidate locked-test result")
    for key in (
        "finalized",
        "fresh_process_per_episode",
        "deterministic_evaluation",
        "deterministic_mean_policy",
        "passed",
        "all_under_maximum_duration",
        "physical_gates_passed",
        "frozen_hashes_unchanged",
        "worker_artifact_hashes_recomputed",
    ):
        _require_boolean(payload, key, True, "locked-test aggregate")
    try:
        seeds = tuple(int(seed) for seed in payload.get("seeds", ()))
        episode_count = int(payload.get("episode_count", -1))
        success_count = int(payload.get("success_count", -1))
        body_collision_count = int(payload.get("body_collision_count", -1))
        wheel_only_climb_count = int(payload.get("wheel_only_climb_count", -1))
        safety_abort_count = int(payload.get("safety_abort_count", -1))
        worker_gate_pass_count = int(payload.get("worker_gate_pass_count", -1))
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError("locked-test aggregate counts are invalid") from exc
    if seeds != LOCKED_TEST_SEEDS or episode_count != len(LOCKED_TEST_SEEDS):
        raise CheckpointPromotionError("improved promotion requires locked-test seeds 3001-3005")
    if (
        success_count != len(LOCKED_TEST_SEEDS)
        or body_collision_count != 0
        or wheel_only_climb_count != 0
        or safety_abort_count != 0
        or worker_gate_pass_count != len(LOCKED_TEST_SEEDS)
    ):
        raise CheckpointPromotionError("locked-test success or safety aggregate gate failed")

    if Path(str(payload.get("checkpoint", ""))).resolve() != checkpoint.checkpoint:
        raise CheckpointPromotionError("locked-test aggregate used a different checkpoint")
    if _hash(payload.get("checkpoint_sha256"), label="locked-test checkpoint hash") != (
        checkpoint.checkpoint_sha256
    ):
        raise CheckpointPromotionError("locked-test checkpoint hash mismatch")
    if Path(str(payload.get("checkpoint_manifest", ""))).resolve() != checkpoint.manifest_path:
        raise CheckpointPromotionError("locked-test aggregate names a different manifest")
    if _hash(
        payload.get("checkpoint_manifest_sha256"),
        label="locked-test checkpoint manifest hash",
    ) != checkpoint.manifest_sha256:
        raise CheckpointPromotionError("locked-test checkpoint manifest hash mismatch")

    frozen_hashes = payload.get("frozen_hashes")
    if not isinstance(frozen_hashes, Mapping):
        raise CheckpointPromotionError("locked-test aggregate omits frozen hash evidence")
    for field in FROZEN_HASH_FIELDS:
        locked_hash = _hash(frozen_hashes.get(field), label=f"locked-test {field}")
        if locked_hash != _hash(checkpoint.manifest.get(field), label=f"checkpoint {field}"):
            raise CheckpointPromotionError(f"locked-test {field} differs from validation")
    hash_gates = payload.get("hash_gates")
    if not isinstance(hash_gates, Mapping) or set(hash_gates) != set(
        REQUIRED_LOCKED_TEST_HASH_GATES
    ):
        raise CheckpointPromotionError("locked-test aggregate lacks the complete hash gate set")
    if any(hash_gates[gate] is not True for gate in REQUIRED_LOCKED_TEST_HASH_GATES):
        raise CheckpointPromotionError("locked-test aggregate contains a failed hash gate")

    episodes = payload.get("episodes")
    workers = payload.get("workers")
    canonical_episode_dirs = payload.get("canonical_episode_dirs")
    if (
        not isinstance(episodes, list)
        or not isinstance(workers, list)
        or not isinstance(canonical_episode_dirs, list)
        or len(episodes) != len(LOCKED_TEST_SEEDS)
        or len(workers) != len(LOCKED_TEST_SEEDS)
        or len(canonical_episode_dirs) != len(LOCKED_TEST_SEEDS)
    ):
        raise CheckpointPromotionError("locked-test aggregate must contain five workers/episodes")

    verified_hashes: list[Mapping[str, str]] = []
    for seed, episode, worker in zip(LOCKED_TEST_SEEDS, episodes, workers, strict=True):
        if not isinstance(episode, Mapping) or not isinstance(worker, Mapping):
            raise CheckpointPromotionError("locked-test worker/episode evidence is malformed")
        _validate_locked_episode(episode, seed=seed, label=f"locked-test episode {seed}")
        try:
            worker_seed = int(worker.get("seed", -1))
        except (TypeError, ValueError) as exc:
            raise CheckpointPromotionError("locked-test worker seed is invalid") from exc
        if worker_seed != seed or worker.get("role") != "candidate":
            raise CheckpointPromotionError(f"locked-test worker {seed} identity is invalid")
        _require_boolean(worker, "worker_gate_passed", True, f"locked-test worker {seed}")

        run_dir = Path(str(worker.get("run_dir", ""))).resolve()
        result_path = Path(str(worker.get("worker_result", ""))).resolve()
        episode_dir = Path(str(worker.get("canonical_episode_dir", ""))).resolve()
        if (
            result_path != run_dir / "checkpoint_evaluation.json"
            or episode_dir.parent != run_dir
            or not _path_within(episode_dir, run_dir)
        ):
            raise CheckpointPromotionError(f"locked-test worker {seed} paths are invalid")
        aggregate_episode_dir = Path(
            str(episode.get("canonical_episode_dir", ""))
        ).resolve()
        canonical_episode_dir = Path(str(canonical_episode_dirs[len(verified_hashes)])).resolve()
        if aggregate_episode_dir != episode_dir or canonical_episode_dir != episode_dir:
            raise CheckpointPromotionError(
                f"locked-test worker {seed} episode provenance mismatch"
            )
        run_manifest_path = run_dir / "run_manifest.json"
        trial_manifest_path = episode_dir / "trial_manifest.json"
        frozen_before_path = run_dir / "frozen_hashes.before.json"
        frozen_after_path = run_dir / "frozen_hashes.after.json"
        if (
            Path(str(worker.get("frozen_hashes_before", ""))).resolve()
            != frozen_before_path
            or Path(str(worker.get("frozen_hashes_after", ""))).resolve()
            != frozen_after_path
            or worker.get("all_artifact_hashes_recomputed") is not True
        ):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} frozen-audit provenance is invalid"
            )
        file_specs = (
            ("run_manifest_sha256", run_manifest_path),
            ("worker_result_sha256", result_path),
            ("trial_manifest_sha256", trial_manifest_path),
            ("frozen_hashes_before_sha256", frozen_before_path),
            ("frozen_hashes_after_sha256", frozen_after_path),
        )
        row_hashes: dict[str, str] = {}
        for key, evidence_path in file_specs:
            expected_hash = _hash(worker.get(key), label=f"worker {seed} {key}")
            if not evidence_path.is_file() or sha256_file(evidence_path) != expected_hash:
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} {key} does not match its file"
                )
            row_hashes[key] = expected_hash

        _, lifecycle, lifecycle_hash = _load_json_snapshot(
            run_manifest_path, label=f"worker {seed} run manifest"
        )
        if lifecycle_hash != row_hashes["run_manifest_sha256"]:
            raise CheckpointPromotionError(
                f"locked-test worker {seed} run manifest changed during verification"
            )
        if lifecycle.get("lifecycle") != "SUCCEEDED" or lifecycle.get("exit_code") != 0:
            raise CheckpointPromotionError(f"locked-test worker {seed} was not finalized")
        configs = _records_by_path(
            lifecycle.get("configs"), label=f"worker {seed} config records"
        )
        for field, relative_path in _RUN_CONFIG_PATH_BY_HASH_FIELD.items():
            config = configs.get(relative_path)
            if not isinstance(config, Mapping) or _hash(
                config.get("sha256"), label=f"worker {seed} {relative_path} hash"
            ) != _hash(checkpoint.manifest.get(field), label=f"checkpoint {field}"):
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} runtime {field} changed"
                )
        _, result, result_hash = _load_json_snapshot(
            result_path, label=f"worker {seed} evaluation result"
        )
        if result_hash != row_hashes["worker_result_sha256"]:
            raise CheckpointPromotionError(
                f"locked-test worker {seed} result changed during verification"
            )
        if (
            result.get("schema") != "wlr50_clean.ppo_checkpoint_evaluation.v1"
            or result.get("fresh_process_single_episode") is not True
            or result.get("vec_env_step_called") is not False
            or result.get("deterministic_mean_policy") is not True
            or result.get("episode_count") != 1
            or result.get("passed") is not True
        ):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} evaluation contract is invalid"
            )
        if (
            Path(str(result.get("checkpoint", ""))).resolve() != checkpoint.checkpoint
            or _hash(result.get("checkpoint_sha256"), label=f"worker {seed} checkpoint hash")
            != checkpoint.checkpoint_sha256
        ):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} used a different checkpoint"
            )
        worker_episodes = result.get("episodes")
        if not isinstance(worker_episodes, list) or len(worker_episodes) != 1:
            raise CheckpointPromotionError(
                f"locked-test worker {seed} must contain one episode"
            )
        worker_episode = worker_episodes[0]
        if not isinstance(worker_episode, Mapping):
            raise CheckpointPromotionError(f"locked-test worker {seed} episode is invalid")
        _validate_locked_episode(
            worker_episode, seed=seed, label=f"locked-test worker episode {seed}"
        )
        for key in (
            "task_success",
            "body_collision",
            "wheel_only_climb",
            "safety_abort",
            "under_maximum_duration",
            "duration_s",
            "recording_runtime_access_count",
            "in_episode_root_write_count",
        ):
            if worker_episode.get(key) != episode.get(key):
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} aggregate episode differs at {key}"
                )
        infos = result.get("checkpoint_infos")
        if not isinstance(infos, Mapping):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} omits checkpoint hash provenance"
            )
        for field in FROZEN_HASH_FIELDS:
            if _hash(infos.get(field), label=f"worker {seed} {field}") != _hash(
                checkpoint.manifest.get(field), label=f"checkpoint {field}"
            ):
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} {field} changed"
                )

        aggregate_frozen_manifest_hash = _hash(
            payload.get("frozen_manifest_sha256"),
            label="locked-test frozen manifest hash",
        )
        for audit_label, audit_path, audit_hash_key in (
            (
                f"worker {seed} frozen pre-audit",
                frozen_before_path,
                "frozen_hashes_before_sha256",
            ),
            (
                f"worker {seed} frozen post-audit",
                frozen_after_path,
                "frozen_hashes_after_sha256",
            ),
        ):
            _, audit, audit_hash = _load_json_snapshot(audit_path, label=audit_label)
            if audit_hash != row_hashes[audit_hash_key] or _hash(
                audit.get("frozen_manifest_sha256"),
                label=f"{audit_label} frozen manifest hash",
            ) != aggregate_frozen_manifest_hash:
                raise CheckpointPromotionError(
                    f"locked-test worker {seed} frozen audit hash changed"
                )
            audit_frozen = _frozen_audit_hashes(audit, label=audit_label)
            for field in _FROZEN_PATH_BY_HASH_FIELD:
                if audit_frozen[field] != _hash(
                    checkpoint.manifest.get(field), label=f"checkpoint {field}"
                ):
                    raise CheckpointPromotionError(
                        f"locked-test worker {seed} frozen {field} changed"
                    )

        _, trial, trial_hash = _load_json_snapshot(
            trial_manifest_path, label=f"worker {seed} trial manifest"
        )
        if trial_hash != row_hashes["trial_manifest_sha256"]:
            raise CheckpointPromotionError(
                f"locked-test worker {seed} trial changed during verification"
            )
        if (
            trial.get("schema") != "wlr50_clean.ppo_live_trial_manifest.v1"
            or int(trial.get("seed", -1)) != seed
            or trial.get("result") != "SUCCESS"
        ):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} trial was not a finalized success"
            )
        success = trial.get("success_evidence")
        trial_duration = _finite(
            success.get("duration_s"), label=f"worker {seed} trial duration"
        ) if isinstance(success, Mapping) else math.nan
        if (
            not isinstance(success, Mapping)
            or success.get("p01_p13_completed") is not True
            or success.get("body_collision") is not False
            or success.get("wheel_only_climb") is not False
            or trial_duration <= 0.0
            or trial_duration > 200.0
            or not math.isclose(
                trial_duration,
                _finite(episode.get("duration_s"), label=f"episode {seed} duration"),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise CheckpointPromotionError(
                f"locked-test worker {seed} trial success/safety evidence failed"
            )
        verified_hashes.append(row_hashes)

    return _LockedTestEvidence(
        path=path,
        sha256=aggregate_hash,
        payload=payload,
        worker_artifact_sha256=tuple(verified_hashes),
    )


def promote_improved_checkpoint(
    *,
    promotion_decision_path: Path | str,
    locked_test_aggregate_path: Path | str,
    best_validation_checkpoint_path: Path | str,
    best_validation_manifest_path: Path | str,
    validation_promotion_manifest_path: Path | str,
    output_root: Path | str,
) -> ImprovedPromotionArtifacts:
    """Publish ``checkpoint_improved.pt`` after the independent locked test."""

    checkpoint, validation_path, validation, validation_hash = (
        _validate_best_validation_source(
            best_validation_checkpoint_path,
            best_validation_manifest_path,
            validation_promotion_manifest_path,
            promotion_decision_path,
        )
    )
    decision_path = Path(str(validation["promotion_decision"])).resolve()
    decision_hash = _hash(
        validation["promotion_decision_sha256"], label="promotion decision hash"
    )
    locked = _validate_locked_test_aggregate(locked_test_aggregate_path, checkpoint)
    root = Path(output_root).resolve()
    checkpoint_dir = root / "checkpoints"
    manifest_dir = root / "manifests"
    if (
        checkpoint.checkpoint != checkpoint_dir / BEST_CHECKPOINT_NAME
        or checkpoint.manifest_path != checkpoint_dir / BEST_MANIFEST_NAME
        or validation_path != manifest_dir / VALIDATION_PROMOTION_MANIFEST_NAME
    ):
        raise CheckpointPromotionError(
            "improved promotion inputs are not the canonical best-validation artifacts "
            "under output_root"
        )
    artifacts = ImprovedPromotionArtifacts(
        improved_checkpoint=checkpoint_dir / IMPROVED_CHECKPOINT_NAME,
        improved_manifest=checkpoint_dir / IMPROVED_MANIFEST_NAME,
        promotion_manifest=manifest_dir / PROMOTION_MANIFEST_NAME,
    )
    destinations = (
        artifacts.improved_checkpoint,
        artifacts.improved_manifest,
        artifacts.promotion_manifest,
    )
    conflict = next((path for path in destinations if path.exists()), None)
    if conflict is not None:
        raise CheckpointPromotionError(f"refusing to overwrite artifact: {conflict}")

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".improved-promotion-", dir=str(root)))
    try:
        stage_checkpoints = staging / "checkpoints"
        stage_manifests = staging / "manifests"
        stage_checkpoints.mkdir(parents=True)
        stage_manifests.mkdir(parents=True)
        stage_improved = stage_checkpoints / IMPROVED_CHECKPOINT_NAME
        _copy_file(checkpoint.checkpoint, stage_improved)
        if sha256_file(stage_improved) != checkpoint.checkpoint_sha256:
            raise CheckpointPromotionError("staged improved checkpoint hash mismatch")

        stage_improved_manifest = stage_checkpoints / IMPROVED_MANIFEST_NAME
        locked_summary = {
            "aggregate": str(locked.path),
            "aggregate_sha256": locked.sha256,
            "seeds": list(LOCKED_TEST_SEEDS),
            "success_count": len(LOCKED_TEST_SEEDS),
            "body_collision_count": 0,
            "wheel_only_climb_count": 0,
            "safety_abort_count": 0,
            "all_under_maximum_duration": True,
            "hash_gates": dict(locked.payload["hash_gates"]),
            "worker_artifact_sha256": [dict(row) for row in locked.worker_artifact_sha256],
        }
        atomic_write_json(
            stage_improved_manifest,
            {
                **dict(checkpoint.manifest),
                "source_best_validation_checkpoint": str(checkpoint.checkpoint),
                "source_best_validation_checkpoint_sha256": checkpoint.checkpoint_sha256,
                "source_best_validation_manifest": str(checkpoint.manifest_path),
                "source_best_validation_manifest_sha256": checkpoint.manifest_sha256,
                "validation_promotion_manifest": str(validation_path),
                "validation_promotion_manifest_sha256": validation_hash,
                "promotion_decision": str(decision_path),
                "promotion_decision_sha256": decision_hash,
                "locked_test_aggregate": str(locked.path),
                "locked_test_aggregate_sha256": locked.sha256,
                "validation_promotion_authorized": True,
                "locked_test_authorized": True,
                "promotion_authorized": True,
                "publication_role": "improved",
                "checkpoint_path": str(artifacts.improved_checkpoint),
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "locked_test": locked_summary,
            },
        )
        improved_manifest_hash = sha256_file(stage_improved_manifest)
        final_payload = {
            "schema": CHECKPOINT_IMPROVED_PROMOTION_SCHEMA,
            "valid": True,
            "status": "PROMOTED_IMPROVED",
            "two_stage_promotion": True,
            "validation_decision_alone_cannot_authorize_improved": True,
            "filename_inference_used": False,
            "validation_promotion_manifest": str(validation_path),
            "validation_promotion_manifest_sha256": validation_hash,
            "promotion_decision": str(decision_path),
            "promotion_decision_sha256": decision_hash,
            "validation_promotion": dict(validation),
            "locked_test_aggregate": str(locked.path),
            "locked_test_aggregate_sha256": locked.sha256,
            "locked_test": locked_summary,
            "published_checkpoints": {
                "best_validation": {
                    "path": str(checkpoint.checkpoint),
                    "sha256": checkpoint.checkpoint_sha256,
                    "manifest": str(checkpoint.manifest_path),
                    "manifest_sha256": checkpoint.manifest_sha256,
                },
                "improved": {
                    "path": str(artifacts.improved_checkpoint),
                    "sha256": checkpoint.checkpoint_sha256,
                    "manifest": str(artifacts.improved_manifest),
                    "manifest_sha256": improved_manifest_hash,
                },
            },
            "byte_identical_best_and_improved": True,
            "immutable_no_overwrite": True,
        }
        stage_promotion_manifest = stage_manifests / PROMOTION_MANIFEST_NAME
        atomic_write_json(stage_promotion_manifest, final_payload)

        locked_recheck = _validate_locked_test_aggregate(locked.path, checkpoint)
        if (
            sha256_file(checkpoint.checkpoint) != checkpoint.checkpoint_sha256
            or sha256_file(checkpoint.manifest_path) != checkpoint.manifest_sha256
            or sha256_file(validation_path) != validation_hash
            or sha256_file(decision_path) != decision_hash
            or locked_recheck.sha256 != locked.sha256
            or locked_recheck.worker_artifact_sha256
            != locked.worker_artifact_sha256
        ):
            raise CheckpointPromotionError("improved promotion input changed during publication")
        _publish_bundle_no_replace(
            (
                (stage_improved, artifacts.improved_checkpoint),
                (stage_improved_manifest, artifacts.improved_manifest),
                (stage_promotion_manifest, artifacts.promotion_manifest),
            )
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if (
        sha256_file(artifacts.improved_checkpoint) != checkpoint.checkpoint_sha256
        or sha256_file(artifacts.improved_manifest)
        != final_payload["published_checkpoints"]["improved"]["manifest_sha256"]
    ):
        raise CheckpointPromotionError("published improved artifact hash mismatch")
    return artifacts


def _runner_reference(policy: Any, observations: Any) -> Any:
    """Call the original deterministic RSL policy with its TensorDict ABI."""

    groups = tuple(getattr(policy, "obs_groups", ()))
    if groups:
        if groups != ("policy",):
            raise CheckpointPromotionError(
                f"inference export requires the single policy observation group, found {groups}"
            )
        try:
            from tensordict import TensorDict  # type: ignore
        except ImportError as exc:
            raise CheckpointPromotionError("RSL TensorDict runtime is unavailable") from exc
        policy_input = TensorDict(
            {"policy": observations}, batch_size=[int(observations.shape[0])]
        )
    else:
        policy_input = observations
    try:
        return policy(policy_input, stochastic_output=False)
    except TypeError:
        return policy(policy_input)


def _tensor_checks(
    value: Any,
    repeated: Any,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> None:
    try:
        shape = tuple(int(part) for part in value.shape)
    except Exception as exc:
        raise CheckpointPromotionError(f"{label} did not return a tensor") from exc
    if shape != expected_shape:
        raise CheckpointPromotionError(
            f"{label} returned shape {shape}, expected {expected_shape}"
        )
    import torch  # type: ignore

    if not bool(torch.isfinite(value).all().item()):
        raise CheckpointPromotionError(f"{label} returned NaN or infinity")
    if not torch.equal(value, repeated):
        raise CheckpointPromotionError(f"{label} is not deterministic")


def _maximum_tensor_error(left: Any, right: Any) -> float:
    import torch  # type: ignore

    return float(torch.max(torch.abs(left - right)).item()) if left.numel() else 0.0


def _onnx_available() -> tuple[bool, str]:
    if importlib.util.find_spec("onnx") is None:
        return False, "onnx package is not installed"
    try:
        import onnx  # type: ignore

        from onnx.reference import ReferenceEvaluator  # type: ignore  # noqa: F401

        return True, str(onnx.__version__)
    except Exception as exc:
        return False, f"ONNX validation runtime unavailable: {type(exc).__name__}: {exc}"


def export_inference_actor(
    runner: Any,
    *,
    source_checkpoint_path: Path | str,
    source_manifest_path: Path | str,
    output_root: Path | str,
    batch_size: int = 4,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-5,
) -> InferenceActorArtifacts:
    """Export and independently verify the deterministic inference-only actor."""

    checkpoint = _validate_checkpoint(source_checkpoint_path, source_manifest_path)
    if (
        checkpoint.manifest.get("publication_role") != "improved"
        or checkpoint.manifest.get("validation_promotion_authorized") is not True
        or checkpoint.manifest.get("locked_test_authorized") is not True
        or checkpoint.manifest.get("promotion_authorized") is not True
    ):
        raise CheckpointPromotionError(
            "inference export requires the two-stage promoted improved checkpoint"
        )
    if int(batch_size) <= 0:
        raise CheckpointPromotionError("inference export batch_size must be positive")
    atol = _finite(absolute_tolerance, label="absolute export tolerance")
    rtol = _finite(relative_tolerance, label="relative export tolerance")
    if atol < 0.0 or rtol < 0.0:
        raise CheckpointPromotionError("inference export tolerances must be non-negative")
    root = Path(output_root).resolve()
    torchscript_path = root / "checkpoints" / TORCHSCRIPT_ACTOR_NAME
    onnx_path = root / "checkpoints" / ONNX_ACTOR_NAME
    manifest_path = root / "manifests" / INFERENCE_EXPORT_MANIFEST_NAME
    onnx_supported, onnx_support_detail = _onnx_available()
    # Even an unavailable ONNX runtime cannot make a stale actor with the
    # canonical name safe.  Treat every possible output as immutable state.
    expected_destinations = [torchscript_path, onnx_path, manifest_path]
    conflict = next((path for path in expected_destinations if path.exists()), None)
    if conflict is not None:
        raise CheckpointPromotionError(f"refusing to overwrite artifact: {conflict}")

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise CheckpointPromotionError("PyTorch is required for inference actor export") from exc
    runner_device = str(getattr(runner, "device", "cpu"))
    try:
        policy = runner.get_inference_policy(device=runner_device)
    except Exception as exc:
        raise CheckpointPromotionError("cannot obtain the loaded RSL inference policy") from exc
    if not callable(getattr(policy, "as_jit", None)):
        raise CheckpointPromotionError("loaded RSL policy does not expose as_jit()")
    policy.eval()

    count = int(batch_size) * checkpoint.observation_dimension
    sample_cpu = torch.linspace(-1.0, 1.0, steps=count, dtype=torch.float32).reshape(
        int(batch_size), checkpoint.observation_dimension
    )
    try:
        sample_runner = sample_cpu.to(runner_device)
        with torch.inference_mode():
            reference = _runner_reference(policy, sample_runner)
            repeated_reference = _runner_reference(policy, sample_runner)
        _tensor_checks(
            reference,
            repeated_reference,
            expected_shape=(int(batch_size), checkpoint.action_dimension),
            label="loaded RSL deterministic actor",
        )
        reference_cpu = reference.detach().to("cpu")
        actor = copy.deepcopy(policy.as_jit()).to("cpu").eval()
        with torch.inference_mode():
            eager_actor = actor(sample_cpu)
            repeated_eager = actor(sample_cpu)
        _tensor_checks(
            eager_actor,
            repeated_eager,
            expected_shape=(int(batch_size), checkpoint.action_dimension),
            label="inference-only eager actor",
        )
        if not torch.allclose(eager_actor, reference_cpu, atol=atol, rtol=rtol):
            raise CheckpointPromotionError("inference-only actor differs from loaded RSL policy")
    except CheckpointPromotionError:
        raise
    except Exception as exc:
        raise CheckpointPromotionError(
            f"cannot evaluate loaded deterministic actor: {type(exc).__name__}: {exc}"
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".inference-export-", dir=str(root)))
    try:
        stage_checkpoints = staging / "checkpoints"
        stage_manifests = staging / "manifests"
        stage_checkpoints.mkdir(parents=True)
        stage_manifests.mkdir(parents=True)
        stage_torchscript = stage_checkpoints / TORCHSCRIPT_ACTOR_NAME
        try:
            scripted = torch.jit.script(actor)
            torch.jit.save(scripted, str(stage_torchscript))
            reloaded = torch.jit.load(str(stage_torchscript), map_location="cpu").eval()
            with torch.inference_mode():
                jit_output = reloaded(sample_cpu)
                jit_repeated = reloaded(sample_cpu)
            _tensor_checks(
                jit_output,
                jit_repeated,
                expected_shape=(int(batch_size), checkpoint.action_dimension),
                label="reloaded TorchScript actor",
            )
            if not torch.allclose(jit_output, reference_cpu, atol=atol, rtol=rtol):
                raise CheckpointPromotionError("TorchScript round trip differs from loaded RSL policy")
        except CheckpointPromotionError:
            raise
        except Exception as exc:
            raise CheckpointPromotionError(
                f"TorchScript export/round trip failed: {type(exc).__name__}: {exc}"
            ) from exc

        torchscript_evidence: dict[str, Any] = {
            "valid": True,
            "status": "PASS",
            "supported": True,
            "path": str(torchscript_path),
            "sha256": sha256_file(stage_torchscript),
            "bytes": stage_torchscript.stat().st_size,
            "reloaded": True,
            "output_shape": list(jit_output.shape),
            "finite": True,
            "deterministic": True,
            "equivalent_to_loaded_runner": True,
            "maximum_absolute_error": _maximum_tensor_error(jit_output, reference_cpu),
        }

        stage_onnx: Path | None = None
        onnx_evidence: dict[str, Any]
        if onnx_supported:
            stage_onnx = stage_checkpoints / ONNX_ACTOR_NAME
            try:
                import numpy as np
                import onnx  # type: ignore
                from onnx.reference import ReferenceEvaluator  # type: ignore

                torch.onnx.export(
                    actor,
                    sample_cpu,
                    str(stage_onnx),
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=["observations"],
                    output_names=["residual_actions"],
                    dynamic_axes={
                        "observations": {0: "batch"},
                        "residual_actions": {0: "batch"},
                    },
                    dynamo=False,
                )
                model = onnx.load(str(stage_onnx))
                onnx.checker.check_model(model, full_check=True)
                evaluator = ReferenceEvaluator(model)
                numpy_input = sample_cpu.detach().numpy()
                onnx_output = np.asarray(
                    evaluator.run(["residual_actions"], {"observations": numpy_input})[0]
                )
                onnx_repeated = np.asarray(
                    evaluator.run(["residual_actions"], {"observations": numpy_input})[0]
                )
                expected_shape = (int(batch_size), checkpoint.action_dimension)
                if tuple(onnx_output.shape) != expected_shape:
                    raise CheckpointPromotionError(
                        f"ONNX actor returned shape {onnx_output.shape}, expected {expected_shape}"
                    )
                if not bool(np.isfinite(onnx_output).all()):
                    raise CheckpointPromotionError("ONNX actor returned NaN or infinity")
                if not bool(np.array_equal(onnx_output, onnx_repeated)):
                    raise CheckpointPromotionError("ONNX actor is not deterministic")
                reference_numpy = reference_cpu.detach().numpy()
                if not bool(np.allclose(onnx_output, reference_numpy, atol=atol, rtol=rtol)):
                    raise CheckpointPromotionError("ONNX round trip differs from loaded RSL policy")
                onnx_evidence = {
                    "valid": True,
                    "status": "PASS",
                    "supported": True,
                    "support_runtime": f"onnx {onnx_support_detail} ReferenceEvaluator",
                    "path": str(onnx_path),
                    "sha256": sha256_file(stage_onnx),
                    "bytes": stage_onnx.stat().st_size,
                    "opset": 18,
                    "checker_full_check": True,
                    "reference_evaluator_round_trip": True,
                    "output_shape": list(onnx_output.shape),
                    "finite": True,
                    "deterministic": True,
                    "equivalent_to_loaded_runner": True,
                    "maximum_absolute_error": float(
                        np.max(np.abs(onnx_output - reference_numpy))
                    ),
                }
            except CheckpointPromotionError:
                raise
            except Exception as exc:
                raise CheckpointPromotionError(
                    f"ONNX export/round trip failed: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            onnx_evidence = {
                "status": "UNSUPPORTED",
                "supported": False,
                "reason": onnx_support_detail,
                "path": None,
            }

        parameter_count = sum(int(parameter.numel()) for parameter in actor.parameters())
        evidence = {
            "schema": INFERENCE_EXPORT_SCHEMA,
            "valid": True,
            "status": "PASS",
            "inference_only": True,
            "deterministic_mean_policy": True,
            "contains_critic": False,
            "contains_optimizer": False,
            "contains_rollout_state": False,
            "contains_stochastic_sampler": False,
            "source_checkpoint": str(checkpoint.checkpoint),
            "source_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "source_manifest": str(checkpoint.manifest_path),
            "source_manifest_sha256": checkpoint.manifest_sha256,
            "observation_dimension": checkpoint.observation_dimension,
            "residual_action_dimension": checkpoint.action_dimension,
            "test_batch_size": int(batch_size),
            "test_input": "deterministic linspace[-1,1] float32",
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
            "loaded_runner_output_shape": list(reference_cpu.shape),
            "loaded_runner_output_finite": True,
            "loaded_runner_deterministic": True,
            "actor_parameter_count": parameter_count,
            "torchscript": torchscript_evidence,
            "onnx": onnx_evidence,
            "immutable_no_overwrite": True,
        }
        stage_manifest = stage_manifests / INFERENCE_EXPORT_MANIFEST_NAME
        atomic_write_json(stage_manifest, evidence)
        if (
            sha256_file(checkpoint.checkpoint) != checkpoint.checkpoint_sha256
            or sha256_file(checkpoint.manifest_path) != checkpoint.manifest_sha256
        ):
            raise CheckpointPromotionError("checkpoint changed during inference export")
        pairs = [
            (stage_torchscript, torchscript_path),
            (stage_manifest, manifest_path),
        ]
        if stage_onnx is not None:
            pairs.insert(1, (stage_onnx, onnx_path))
        _publish_bundle_no_replace(pairs)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    published_evidence = _load_json(manifest_path, label="inference export manifest")
    if sha256_file(torchscript_path) != published_evidence["torchscript"]["sha256"]:
        raise CheckpointPromotionError("published TorchScript actor hash mismatch")
    published_onnx = onnx_path if onnx_supported else None
    if published_onnx is not None and (
        sha256_file(published_onnx) != published_evidence["onnx"]["sha256"]
    ):
        raise CheckpointPromotionError("published ONNX actor hash mismatch")
    return InferenceActorArtifacts(
        torchscript_actor=torchscript_path,
        onnx_actor=published_onnx,
        export_manifest=manifest_path,
        evidence=published_evidence,
    )


__all__ = [
    "BEST_CHECKPOINT_NAME",
    "BEST_MANIFEST_NAME",
    "CHECKPOINT_IMPROVED_PROMOTION_SCHEMA",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CHECKPOINT_VALIDATION_PROMOTION_SCHEMA",
    "CheckpointArtifactProvenance",
    "CheckpointPromotionError",
    "FROZEN_HASH_FIELDS",
    "IMPROVED_CHECKPOINT_NAME",
    "IMPROVED_MANIFEST_NAME",
    "ImprovedPromotionArtifacts",
    "INFERENCE_EXPORT_MANIFEST_NAME",
    "InferenceActorArtifacts",
    "LOCKED_TEST_AGGREGATE_SCHEMA",
    "LOCKED_TEST_SEEDS",
    "ONNX_ACTOR_NAME",
    "PROMOTION_MANIFEST_NAME",
    "PROMOTION_DECISION_SCHEMA",
    "REQUIRED_LOCKED_TEST_HASH_GATES",
    "REQUIRED_PROMOTION_GATES",
    "TORCHSCRIPT_ACTOR_NAME",
    "VALIDATION_PROMOTION_MANIFEST_NAME",
    "VALIDATION_SEEDS",
    "ValidationPromotionArtifacts",
    "export_inference_actor",
    "finalize_locked_test_aggregate_payload",
    "promote_best_validation_checkpoint",
    "promote_improved_checkpoint",
    "validate_checkpoint_artifact_provenance",
]
