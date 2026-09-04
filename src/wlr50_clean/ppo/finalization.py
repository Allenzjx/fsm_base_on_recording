"""Fail-closed provenance validation and immutable final-delivery manifests.

This module is deliberately offline.  It does not run Isaac, train a policy,
evaluate an episode, promote a checkpoint, or render a video.  It only accepts
explicit evidence paths, re-hashes every referenced artifact, validates the
success state recorded *inside* those artifacts, and then publishes the two
final provenance manifests and one whole-delivery checksum manifest.

Publication is idempotent for byte-identical results.  A conflicting existing
file aborts before anything is written, and a later publication failure rolls
back every file created by the current call.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import (
    RUN_MANIFEST_SCHEMA,
    ArtifactError,
    _atomic_bytes,
    sha256_file,
    verify_checksum_manifest,
)
from .checkpoint_promotion import (
    CHECKPOINT_IMPROVED_PROMOTION_SCHEMA,
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_VALIDATION_PROMOTION_SCHEMA,
    FROZEN_HASH_FIELDS,
    IMPROVED_CHECKPOINT_NAME,
    INFERENCE_EXPORT_SCHEMA,
    LOCKED_TEST_SEEDS,
    PROMOTION_DECISION_SCHEMA,
    REQUIRED_LOCKED_TEST_HASH_GATES,
    REQUIRED_PROMOTION_GATES,
    VALIDATION_SEEDS,
)
from .evaluation_artifacts import (
    BASELINE_EPISODE_FILENAME,
    BASELINE_EVALUATION_MANIFEST_FILENAME,
    BASELINE_PHASE_FILENAME,
    CANONICAL_EPISODE_FILES,
    CANDIDATE_EPISODE_FILENAME,
    CANDIDATE_PHASE_FILENAME,
    CHECKPOINT_COMPARISON_FILENAME,
    PHASE_COMPARISON_FILENAME,
    PROMOTION_DECISION_FILENAME,
    RESIDUAL_ACTIVITY_FILENAME,
    REWARD_CONTRIBUTION_FILENAME,
    TERMINATION_SUMMARY_FILENAME,
    NONEMPTY_CANONICAL_EPISODE_FILES,
)
from .final_reporting import PLOT_FILENAMES, REPORT_FILENAMES
from .video_artifacts import (
    COMPARISON_VIDEO_NAME,
    DIAGNOSTIC_VIDEO_NAME,
    FSM_VIDEO_NAME,
    PPO_VIDEO_NAME,
    VIDEO_CHECKSUM_NAME,
    VIDEO_VALIDATION_NAME,
)


TRAINING_MANIFEST_SCHEMA = "wlr50_clean.ppo_final_training_manifest.v1"
EVALUATION_MANIFEST_SCHEMA = "wlr50_clean.ppo_final_evaluation_manifest.v1"
TRAINING_RESULT_SCHEMA = "wlr50_clean.ppo_training_run.v1"
FRESH_PROCESS_BATCH_SCHEMA = "wlr50_clean.fresh_process_episode_batch.v1"
BASELINE_METRIC_MANIFEST_SCHEMA = "wlr50_clean.fsm_baseline_evaluation.v1"
FINAL_VIDEO_SCHEMA = "wlr50_clean.ppo_final_videos.v1"

FINAL_STAGE_ORDER = (
    "smoke",
    "phase-curriculum",
    "full-episode",
    "mild-randomization",
)
REQUIRED_TRAINING_STAGES = FINAL_STAGE_ORDER[:3]
_REQUIRED_VALIDATION_FILES = (
    BASELINE_EPISODE_FILENAME,
    BASELINE_PHASE_FILENAME,
    CANDIDATE_EPISODE_FILENAME,
    CANDIDATE_PHASE_FILENAME,
    CHECKPOINT_COMPARISON_FILENAME,
    PHASE_COMPARISON_FILENAME,
    RESIDUAL_ACTIVITY_FILENAME,
    REWARD_CONTRIBUTION_FILENAME,
    TERMINATION_SUMMARY_FILENAME,
    PROMOTION_DECISION_FILENAME,
)
_REQUIRED_BASELINE_FILES = (
    BASELINE_EPISODE_FILENAME,
    BASELINE_PHASE_FILENAME,
    BASELINE_EVALUATION_MANIFEST_FILENAME,
)
_REQUIRED_VIDEO_KEYS = (
    "fsm_baseline",
    "ppo_improved",
    "comparison",
    "ppo_diagnostic",
)
_REQUIRED_VIDEO_NAMES = (
    FSM_VIDEO_NAME,
    PPO_VIDEO_NAME,
    COMPARISON_VIDEO_NAME,
    DIAGNOSTIC_VIDEO_NAME,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_HASH_FILE_NAMES = {
    "controller_hash": "fsm_states.yaml",
    "environment_hash": "environment_lock.json",
    "observation_schema_hash": "ppo_observation_schema_v2.json",
    "action_schema_hash": "ppo_phase_action_masks_v2.yaml",
    "reward_config_hash": "ppo_reward_v2.yaml",
}


class FinalizationError(RuntimeError):
    """Final evidence is missing, inconsistent, unsuccessful, or mutable."""


def _fail_closed(function: Any) -> Any:
    """Normalize malformed-evidence failures to the module's public error."""

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except FinalizationError:
            raise
        except (ArtifactError, KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
            raise FinalizationError(
                f"malformed or unstable finalization evidence: {type(exc).__name__}: {exc}"
            ) from exc

    return guarded


@dataclass(frozen=True, slots=True)
class FinalizationPaths:
    output_root: Path
    training_manifest: Path
    evaluation_manifest: Path
    checksums: Path


@dataclass(frozen=True, slots=True)
class _JsonSnapshot:
    path: Path
    payload: Mapping[str, Any]
    sha256: str
    size: int


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"final manifest is not JSON serializable: {exc}") from exc


def _load_json(path: Path | str, *, label: str) -> _JsonSnapshot:
    source = Path(path).resolve()
    if not source.is_file():
        raise FinalizationError(f"{label} is missing: {source}")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{label} is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(payload, Mapping):
        raise FinalizationError(f"{label} must contain a JSON object: {source}")
    return _JsonSnapshot(
        source,
        payload,
        hashlib.sha256(raw).hexdigest(),
        len(raw),
    )


def _require_hash(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if _SHA256.fullmatch(digest) is None:
        raise FinalizationError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _path(value: Any, *, base: Path, label: str) -> Path:
    if value is None or not str(value).strip():
        raise FinalizationError(f"{label} path is missing")
    raw = Path(str(value))
    result = (base / raw).resolve() if not raw.is_absolute() else raw.resolve()
    return result


def _within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise FinalizationError(f"{label} escapes output_root: {path}") from exc


def _file_record(
    path: Path | str, *, root: Path | None = None, allow_empty: bool = False
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file() or (source.stat().st_size <= 0 and not allow_empty):
        raise FinalizationError(f"required artifact is missing or empty: {source}")
    if source.is_symlink():
        raise FinalizationError(f"final evidence may not be a symbolic link: {source}")
    if root is not None:
        _within(source, root, label="final artifact")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _snapshot_record(snapshot: _JsonSnapshot) -> dict[str, Any]:
    return {
        "path": str(snapshot.path),
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _declared_file(
    payload: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    base: Path,
    label: str,
    bytes_key: str | None = None,
    allow_empty: bool = False,
) -> tuple[Path, dict[str, Any]]:
    source = _path(payload.get(path_key), base=base, label=label)
    record = _file_record(source, allow_empty=allow_empty)
    expected = _require_hash(payload.get(hash_key), label=f"{label} declared hash")
    if record["sha256"] != expected:
        raise FinalizationError(f"{label} SHA-256 mismatch: {source}")
    if bytes_key is not None:
        try:
            declared_size = int(payload.get(bytes_key, -1))
        except (TypeError, ValueError) as exc:
            raise FinalizationError(f"{label} byte count is invalid") from exc
        if declared_size != record["bytes"]:
            raise FinalizationError(f"{label} byte count mismatch: {source}")
    return source, record


def _require_true(payload: Mapping[str, Any], key: str, *, label: str) -> None:
    if payload.get(key) is not True:
        raise FinalizationError(f"{label} requires {key}=true")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise FinalizationError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise FinalizationError(f"{label} must be a positive integer")
    return result


def _finite_positive(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FinalizationError(f"{label} must be finite and positive")
    return result


def _validate_record(
    record: Mapping[str, Any], *, base: Path, label: str, allow_empty: bool = False
) -> dict[str, Any]:
    _, verified = _declared_file(
        record,
        path_key="path",
        hash_key="sha256",
        bytes_key="bytes",
        base=base,
        label=label,
        allow_empty=allow_empty,
    )
    return verified


def _validate_checkpoint_contract(
    payload: Mapping[str, Any], *, base: Path, label: str
) -> None:
    try:
        actor_dimension = int(payload.get("actor_observation_dimension", -1))
        critic_dimension = int(payload.get("critic_observation_dimension", -1))
        residual_dimension = int(payload.get("residual_dimension", -1))
        global_decisions = int(payload.get("global_policy_decisions", -1))
        physics_hz = float(payload.get("physics_hz", math.nan))
        decision_hz = float(payload.get("decision_hz", math.nan))
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} policy/timing contract is invalid") from exc
    if (
        actor_dimension != 125
        or critic_dimension != 125
        or residual_dimension != 12
        or global_decisions < 0
        or not math.isclose(physics_hz, 120.0, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(decision_hz, 15.0, rel_tol=0.0, abs_tol=1.0e-12)
        or not str(payload.get("stage", "")).strip()
    ):
        raise FinalizationError(f"{label} does not describe the 125D-to-Full12 120/15 Hz policy")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise FinalizationError(f"{label} has no checkpoint input-file inventory")
    verified_by_name: dict[str, str] = {}
    for raw_path, raw_hash in files.items():
        source = _path(raw_path, base=base, label=f"{label} input file")
        expected = _require_hash(raw_hash, label=f"{label} input file hash")
        if not source.is_file() or sha256_file(source) != expected:
            raise FinalizationError(f"{label} input file SHA-256 mismatch: {source}")
        if source.name in verified_by_name and verified_by_name[source.name] != expected:
            raise FinalizationError(f"{label} has ambiguous input files named {source.name}")
        verified_by_name[source.name] = expected
    for key in FROZEN_HASH_FIELDS:
        declared = _require_hash(payload.get(key), label=f"{label} {key}")
        expected_name = _CHECKPOINT_HASH_FILE_NAMES[key]
        if verified_by_name.get(expected_name) != declared:
            raise FinalizationError(
                f"{label} {key} is not bound to its {expected_name} input file"
            )


def _validate_training_run(run_dir: Path | str) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise FinalizationError(f"training run directory is missing or unsafe: {directory}")
    lifecycle = _load_json(directory / "run_manifest.json", label="training lifecycle")
    result = _load_json(directory / "training_result.json", label="training result")
    run = lifecycle.payload
    training = result.payload
    if (
        run.get("schema") != RUN_MANIFEST_SCHEMA
        or run.get("lifecycle") != "SUCCEEDED"
        or run.get("exit_code") != 0
        or run.get("immutable_run_directory") is not True
        or _path(run.get("run_dir"), base=directory, label="training run") != directory
    ):
        raise FinalizationError(f"training run did not finalize successfully: {directory}")
    if training.get("schema") != TRAINING_RESULT_SCHEMA:
        raise FinalizationError(f"training result schema is invalid: {result.path}")
    stage = str(training.get("stage", ""))
    if stage not in FINAL_STAGE_ORDER:
        raise FinalizationError(f"training result has an unsupported stage: {stage!r}")
    identity = run.get("identity")
    if not isinstance(identity, Mapping) or identity.get("training_stage") != stage:
        raise FinalizationError(f"training run identity and result stage differ: {directory}")
    if run.get("run_kind") != "train":
        raise FinalizationError(f"training provenance has run_kind={run.get('run_kind')!r}")
    _require_true(training, "save_load_round_trip", label=f"training stage {stage}")
    requested = _positive_int(
        training.get("requested_policy_decisions"), label=f"{stage} requested decisions"
    )
    stage_decisions = _positive_int(
        training.get("stage_policy_decisions"), label=f"{stage} stage decisions"
    )
    global_decisions = _positive_int(
        training.get("global_policy_decisions"), label=f"{stage} global decisions"
    )
    iterations = _positive_int(training.get("iterations"), label=f"{stage} iterations")
    num_envs = _positive_int(training.get("num_envs"), label=f"{stage} num_envs")
    rollout = _positive_int(
        training.get("rollout_length"), label=f"{stage} rollout length"
    )
    if stage_decisions != iterations * num_envs * rollout or requested > stage_decisions:
        raise FinalizationError(f"training policy-decision accounting is invalid: {directory}")
    telemetry = training.get("training_telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("reward_telemetry_complete") is not True
        or int(telemetry.get("policy_decision_count", -1)) != stage_decisions
    ):
        raise FinalizationError(f"training telemetry is incomplete: {directory}")
    if not isinstance(training.get("environment_contract"), Mapping) or not isinstance(
        training.get("runner_config"), Mapping
    ):
        raise FinalizationError(f"training environment/runner contract is missing: {directory}")
    _finite_positive(training.get("wall_time_s"), label=f"{stage} wall time")

    resume_path, resume = _declared_file(
        training,
        path_key="resume_checkpoint",
        hash_key="resume_checkpoint_sha256",
        base=directory,
        label=f"{stage} resume checkpoint",
    )
    history_path = _path(
        training.get("immutable_history_checkpoint"),
        base=directory,
        label=f"{stage} immutable history checkpoint",
    )
    history = _file_record(history_path)
    expected_checkpoint_hash = _require_hash(
        training.get("checkpoint_sha256"), label=f"{stage} checkpoint hash"
    )
    if history["sha256"] != expected_checkpoint_hash:
        raise FinalizationError(f"{stage} immutable checkpoint SHA-256 mismatch")
    last_path = _path(
        training.get("checkpoint_last"), base=directory, label=f"{stage} checkpoint_last"
    )
    if not last_path.is_file():
        raise FinalizationError(f"{stage} checkpoint_last is missing: {last_path}")

    history_manifest_path = history_path.with_name(history_path.stem + "_manifest.json")
    history_manifest = _load_json(history_manifest_path, label=f"{stage} checkpoint manifest")
    checkpoint_payload = history_manifest.payload
    if (
        checkpoint_payload.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or checkpoint_payload.get("stage") != stage
        or int(checkpoint_payload.get("global_policy_decisions", -1)) != global_decisions
        or _path(
            checkpoint_payload.get("checkpoint_path"),
            base=history_manifest.path.parent,
            label=f"{stage} checkpoint manifest",
        )
        != history_path
        or _require_hash(
            checkpoint_payload.get("checkpoint_sha256"),
            label=f"{stage} checkpoint manifest hash",
        )
        != history["sha256"]
    ):
        raise FinalizationError(f"{stage} immutable checkpoint manifest is inconsistent")
    _validate_checkpoint_contract(
        checkpoint_payload, base=history_manifest.path.parent, label=f"{stage} checkpoint"
    )
    resume_manifest_path = _path(
        checkpoint_payload.get("resume_checkpoint"),
        base=history_manifest.path.parent,
        label=f"{stage} checkpoint resume source",
    )
    resume_manifest_hash = _require_hash(
        checkpoint_payload.get("resume_checkpoint_sha256"),
        label=f"{stage} checkpoint resume hash",
    )
    try:
        resume_global = int(checkpoint_payload.get("resume_global_policy_decisions", -1))
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{stage} resume decision count is invalid") from exc
    if (
        resume_manifest_path != resume_path
        or resume_manifest_hash != resume["sha256"]
        or resume_global < 0
        or global_decisions != resume_global + stage_decisions
    ):
        raise FinalizationError(f"{stage} checkpoint resume provenance is inconsistent")

    started = run.get("started_manifest")
    if not isinstance(started, Mapping):
        raise FinalizationError(f"training lifecycle omits started_manifest: {directory}")
    _validate_record(started, base=directory, label=f"{stage} started manifest")
    logs = run.get("logs")
    if not isinstance(logs, Mapping):
        raise FinalizationError(f"training lifecycle omits log provenance: {directory}")
    for name in ("stdout.log", "stderr.log"):
        row = logs.get(name)
        if row is not None:
            if not isinstance(row, Mapping):
                raise FinalizationError(f"training {name} record is invalid: {directory}")
            _validate_record(
                row, base=directory, label=f"{stage} {name}", allow_empty=True
            )

    project_root = _path(run.get("project_root"), base=directory, label="project root")
    configs = run.get("configs")
    if not isinstance(configs, Sequence) or isinstance(configs, (str, bytes)) or not configs:
        raise FinalizationError(f"training run has no versioned configs: {directory}")
    verified_configs: list[dict[str, Any]] = []
    for index, row in enumerate(configs):
        if not isinstance(row, Mapping):
            raise FinalizationError(f"training config record {index} is invalid")
        verified_configs.append(
            _validate_record(row, base=project_root, label=f"{stage} config {index}")
        )

    frozen_records: list[dict[str, Any]] = []
    frozen_entry_sets: list[tuple[tuple[str, str, str], ...]] = []
    for name in ("frozen_hashes.before.json", "frozen_hashes.after.json"):
        audit = _load_json(directory / name, label=f"{stage} frozen hash audit")
        if audit.payload.get("passed") is not True or audit.payload.get("mismatches") not in (
            [],
            (),
        ):
            raise FinalizationError(f"{stage} frozen hash audit failed: {audit.path}")
        entries = audit.payload.get("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
            raise FinalizationError(f"{stage} frozen hash audit has no checked files")
        checked: list[tuple[str, str, str]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise FinalizationError(f"{stage} frozen hash audit entry is invalid")
            expected = _require_hash(
                entry.get("expected_sha256"), label=f"{stage} frozen expected hash"
            )
            actual = _require_hash(
                entry.get("actual_sha256"), label=f"{stage} frozen actual hash"
            )
            relative = str(entry.get("path", ""))
            if (
                not relative
                or expected != actual
                or entry.get("exists") is not True
                or entry.get("valid") is not True
            ):
                raise FinalizationError(f"{stage} frozen hash audit entry failed")
            frozen_source = _path(
                relative, base=project_root, label=f"{stage} frozen file"
            )
            if not frozen_source.is_file() or sha256_file(frozen_source) != actual:
                raise FinalizationError(
                    f"{stage} frozen file changed after its audit: {frozen_source}"
                )
            checked.append((relative, expected, actual))
        frozen_entry_sets.append(tuple(checked))
        frozen_records.append(_snapshot_record(audit))
    if frozen_entry_sets[0] != frozen_entry_sets[1]:
        raise FinalizationError(f"{stage} before/after frozen hash evidence differs")

    return {
        "stage": stage,
        "run_directory": str(directory),
        "run_manifest": _snapshot_record(lifecycle),
        "training_result": _snapshot_record(result),
        "requested_policy_decisions": requested,
        "stage_policy_decisions": stage_decisions,
        "global_policy_decisions": global_decisions,
        "num_envs": num_envs,
        "iterations": iterations,
        "rollout_length": rollout,
        "resume_checkpoint": resume,
        "immutable_history_checkpoint": history,
        "checkpoint_manifest": _snapshot_record(history_manifest),
        "checkpoint_last_path": str(last_path),
        "configs": verified_configs,
        "frozen_hash_audits": frozen_records,
        "save_load_round_trip": True,
        "reward_telemetry_complete": True,
    }


def _validate_training_runs(run_dirs: Sequence[Path | str]) -> tuple[dict[str, Any], ...]:
    directories = tuple(Path(value).resolve() for value in run_dirs)
    if not directories or len(set(directories)) != len(directories):
        raise FinalizationError("training_run_dirs must be a non-empty unique sequence")
    records = tuple(_validate_training_run(path) for path in directories)
    ranks = tuple(FINAL_STAGE_ORDER.index(str(row["stage"])) for row in records)
    if tuple(sorted(ranks)) != ranks:
        raise FinalizationError("training stages are not supplied in curriculum order")
    observed = {str(row["stage"]) for row in records}
    missing = tuple(stage for stage in REQUIRED_TRAINING_STAGES if stage not in observed)
    if missing:
        raise FinalizationError(f"training evidence is missing required stages: {missing}")
    for previous, current in zip(records, records[1:]):
        if int(current["global_policy_decisions"]) <= int(previous["global_policy_decisions"]):
            raise FinalizationError("training global policy decisions are not increasing")
        if current["resume_checkpoint"]["sha256"] != previous[
            "immutable_history_checkpoint"
        ]["sha256"]:
            raise FinalizationError("training checkpoint resume chain is broken")
        if int(current["global_policy_decisions"]) != int(
            previous["global_policy_decisions"]
        ) + int(current["stage_policy_decisions"]):
            raise FinalizationError("training cumulative policy-decision chain is invalid")
    terminal = records[-1]
    if sha256_file(Path(str(terminal["checkpoint_last_path"]))) != terminal[
        "immutable_history_checkpoint"
    ]["sha256"]:
        raise FinalizationError("checkpoint_last is not the terminal training checkpoint")
    return records


def _safe_episode(row: Mapping[str, Any], *, seed: int, label: str) -> None:
    try:
        actual_seed = int(row.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} seed is invalid") from exc
    if actual_seed != seed:
        raise FinalizationError(f"{label} seed differs from {seed}")
    for key, expected in (
        ("task_success", True),
        ("body_collision", False),
        ("wheel_only_climb", False),
        ("safety_abort", False),
        ("under_maximum_duration", True),
    ):
        if row.get(key) is not expected:
            raise FinalizationError(f"{label} requires {key}={expected!r}")
    if "recording_runtime_access_count" in row and int(
        row.get("recording_runtime_access_count", -1)
    ) != 0:
        raise FinalizationError(f"{label} accessed Recording at runtime")
    if "in_episode_root_write_count" in row and int(
        row.get("in_episode_root_write_count", -1)
    ) != 0:
        raise FinalizationError(f"{label} performed an in-episode root write")


def _validate_worker(row: Mapping[str, Any], *, role: str, seed: int) -> dict[str, Any]:
    label = f"{role} worker {seed}"
    if row.get("role") != role or int(row.get("seed", -1)) != seed:
        raise FinalizationError(f"{label} role/seed binding is invalid")
    if row.get("worker_gate_passed") is not True:
        raise FinalizationError(f"{label} did not pass its worker gate")
    run_dir = _path(row.get("run_dir"), base=Path.cwd(), label=label)
    if not run_dir.is_dir():
        raise FinalizationError(f"{label} run directory is missing: {run_dir}")
    lifecycle = _load_json(run_dir / "run_manifest.json", label=f"{label} lifecycle")
    if lifecycle.payload.get("lifecycle") != "SUCCEEDED" or lifecycle.payload.get(
        "exit_code"
    ) != 0:
        raise FinalizationError(f"{label} lifecycle did not succeed")
    if lifecycle.sha256 != _require_hash(
        row.get("run_manifest_sha256"), label=f"{label} lifecycle hash"
    ):
        raise FinalizationError(f"{label} lifecycle SHA-256 mismatch")
    result_path, _ = _declared_file(
        row,
        path_key="worker_result",
        hash_key="worker_result_sha256",
        base=run_dir,
        label=f"{label} result",
    )
    result = _load_json(result_path, label=f"{label} result")
    if result.payload.get("passed") is not True or result.payload.get("episode_count") != 1:
        raise FinalizationError(f"{label} result is not a passing single episode")
    canonical = _path(
        row.get("canonical_episode_dir"), base=run_dir, label=f"{label} episode"
    )
    try:
        canonical.relative_to(run_dir)
    except ValueError as exc:
        raise FinalizationError(f"{label} canonical episode escapes its run") from exc
    trial = _load_json(canonical / "trial_manifest.json", label=f"{label} trial")
    if trial.sha256 != _require_hash(
        row.get("trial_manifest_sha256"), label=f"{label} trial hash"
    ):
        raise FinalizationError(f"{label} trial SHA-256 mismatch")
    if trial.payload.get("result") != "SUCCESS" or int(
        trial.payload.get("seed", -1)
    ) != seed:
        raise FinalizationError(f"{label} trial is not an authoritative success")
    canonical_files: list[dict[str, Any]] = []
    for name in CANONICAL_EPISODE_FILES:
        canonical_files.append(
            _file_record(
                canonical / name,
                allow_empty=name not in NONEMPTY_CANONICAL_EPISODE_FILES,
            )
        )
    return {
        "role": role,
        "seed": seed,
        "run_directory": str(run_dir),
        "run_manifest": _snapshot_record(lifecycle),
        "worker_result": _snapshot_record(result),
        "canonical_episode_directory": str(canonical),
        "canonical_episode_files": canonical_files,
    }


def _validate_batch(
    path: Path | str,
    *,
    role: str,
    seed_set: str,
    seeds: Sequence[int],
) -> tuple[_JsonSnapshot, str | None, tuple[dict[str, Any], ...]]:
    snapshot = _load_json(path, label=f"{role} {seed_set} aggregate")
    payload = snapshot.payload
    expected_seeds = tuple(int(seed) for seed in seeds)
    try:
        actual_seeds = tuple(int(seed) for seed in payload.get("seeds", ()))
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{role} aggregate seeds are invalid") from exc
    if (
        payload.get("schema") != FRESH_PROCESS_BATCH_SCHEMA
        or payload.get("role") != role
        or payload.get("seed_set") != seed_set
        or actual_seeds != expected_seeds
        or payload.get("fresh_process_per_episode") is not True
        or payload.get("deterministic_evaluation") is not True
        or payload.get("passed") is not True
        or int(payload.get("episode_count", -1)) != len(expected_seeds)
        or int(payload.get("success_count", -1)) != len(expected_seeds)
        or int(payload.get("body_collision_count", -1)) != 0
        or int(payload.get("wheel_only_climb_count", -1)) != 0
        or int(payload.get("safety_abort_count", -1)) != 0
        or payload.get("all_under_maximum_duration") is not True
        or int(payload.get("worker_gate_pass_count", -1)) != len(expected_seeds)
    ):
        raise FinalizationError(f"{role} {seed_set} aggregate is not a complete success")
    if role == "baseline":
        if payload.get("pure_fsm_zero_residual") is not True:
            raise FinalizationError("baseline aggregate is not authoritative pure zero residual")
        checkpoint_hash = None
    else:
        if payload.get("deterministic_mean_policy") is not True:
            raise FinalizationError("candidate aggregate is not deterministic mean-policy")
        checkpoint_path, checkpoint = _declared_file(
            payload,
            path_key="checkpoint",
            hash_key="checkpoint_sha256",
            base=snapshot.path.parent,
            label=f"candidate {seed_set} checkpoint",
        )
        del checkpoint_path
        checkpoint_hash = str(checkpoint["sha256"])
    episodes = payload.get("episodes")
    workers = payload.get("workers")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise FinalizationError(f"{role} aggregate episodes are missing")
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)):
        raise FinalizationError(f"{role} aggregate workers are missing")
    if len(episodes) != len(expected_seeds) or len(workers) != len(expected_seeds):
        raise FinalizationError(f"{role} aggregate worker/episode count is incomplete")
    worker_evidence: list[dict[str, Any]] = []
    for seed, episode, worker in zip(expected_seeds, episodes, workers, strict=True):
        if not isinstance(episode, Mapping) or not isinstance(worker, Mapping):
            raise FinalizationError(f"{role} aggregate row is not an object")
        _safe_episode(episode, seed=seed, label=f"{role} episode {seed}")
        worker_evidence.append(_validate_worker(worker, role=role, seed=seed))
    return snapshot, checkpoint_hash, tuple(worker_evidence)


def _require_named_paths(
    paths: Sequence[Path | str],
    *,
    required_names: Sequence[str],
    root: Path,
    label: str,
) -> dict[str, Path]:
    values = tuple(Path(value).resolve() for value in paths)
    if len(values) != len(set(values)):
        raise FinalizationError(f"{label} paths are duplicated")
    by_name = {path.name: path for path in values}
    if len(by_name) != len(values) or set(by_name) != set(required_names):
        raise FinalizationError(
            f"{label} must contain exactly {tuple(required_names)}, got {tuple(sorted(by_name))}"
        )
    for path in values:
        _file_record(path, root=root)
    return by_name


def _validate_baseline_metrics(
    paths: Sequence[Path | str], *, root: Path, aggregate: _JsonSnapshot
) -> tuple[dict[str, Any], ...]:
    named = _require_named_paths(
        paths, required_names=_REQUIRED_BASELINE_FILES, root=root, label="baseline metrics"
    )
    manifest = _load_json(
        named[BASELINE_EVALUATION_MANIFEST_FILENAME], label="baseline metrics manifest"
    )
    payload = manifest.payload
    if (
        payload.get("schema") != BASELINE_METRIC_MANIFEST_SCHEMA
        or payload.get("validation_seeds") != list(VALIDATION_SEEDS)
        or payload.get("episode_count") != len(VALIDATION_SEEDS)
        or payload.get("all_p01_p13_complete") is not True
        or payload.get("all_authoritative_success") is not True
        or payload.get("all_zero_residual") is not True
        or payload.get("candidate_required") is not False
    ):
        raise FinalizationError("baseline metric manifest is incomplete or unsuccessful")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FinalizationError("baseline metric manifest omits artifact records")
    for key, filename in (
        ("episode_metrics", BASELINE_EPISODE_FILENAME),
        ("phase_metrics", BASELINE_PHASE_FILENAME),
    ):
        record = artifacts.get(key)
        if not isinstance(record, Mapping):
            raise FinalizationError(f"baseline metric manifest omits {key}")
        declared_path, _ = _declared_file(
            record,
            path_key="path",
            hash_key="sha256",
            bytes_key="bytes",
            base=manifest.path.parent,
            label=f"baseline {key}",
        )
        if declared_path != named[filename]:
            raise FinalizationError(f"baseline {key} path differs from explicit input")
    if _path(
        artifacts.get("manifest"), base=manifest.path.parent, label="baseline manifest"
    ) != manifest.path:
        raise FinalizationError("baseline metrics manifest does not name itself exactly")

    sources = payload.get("source_episodes")
    if not isinstance(sources, Sequence) or len(sources) != len(VALIDATION_SEEDS):
        raise FinalizationError("baseline metric source episode evidence is incomplete")
    aggregate_dirs = tuple(
        Path(str(value)).resolve()
        for value in aggregate.payload.get("canonical_episode_dirs", ())
    )
    source_dirs: list[Path] = []
    for seed, source in zip(VALIDATION_SEEDS, sources, strict=True):
        if not isinstance(source, Mapping) or int(source.get("seed", -1)) != seed:
            raise FinalizationError("baseline metric source seed order is invalid")
        directory = _path(
            source.get("canonical_episode_dir"),
            base=manifest.path.parent,
            label=f"baseline source {seed}",
        )
        source_dirs.append(directory)
        records = source.get("files")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise FinalizationError(f"baseline source {seed} file inventory is empty")
        names = tuple(
            str(record.get("name", "")) if isinstance(record, Mapping) else ""
            for record in records
        )
        if names != CANONICAL_EPISODE_FILES:
            raise FinalizationError(
                f"baseline source {seed} canonical file inventory is incomplete"
            )
        for record in records:
            if not isinstance(record, Mapping) or not str(record.get("name", "")):
                raise FinalizationError(f"baseline source {seed} file record is invalid")
            _declared_file(
                {**record, "path": str(record["name"])},
                path_key="path",
                hash_key="sha256",
                bytes_key="bytes",
                base=directory,
                label=f"baseline source {seed} {record['name']}",
                allow_empty=str(record["name"]) not in NONEMPTY_CANONICAL_EPISODE_FILES,
            )
    if tuple(source_dirs) != aggregate_dirs:
        raise FinalizationError("baseline aggregate and metric source episodes differ")
    return tuple(_file_record(named[name], root=root) for name in _REQUIRED_BASELINE_FILES)


def _validate_promotion(
    path: Path | str, *, root: Path, validation_checkpoint_hash: str
) -> tuple[_JsonSnapshot, tuple[dict[str, Any], ...]]:
    snapshot = _load_json(path, label="validation promotion decision")
    payload = snapshot.payload
    if payload.get("schema") != PROMOTION_DECISION_SCHEMA:
        raise FinalizationError("promotion decision schema is invalid")
    if payload.get("paired_seeds") != list(VALIDATION_SEEDS) or int(
        payload.get("paired_episode_count", -1)
    ) != len(VALIDATION_SEEDS):
        raise FinalizationError("promotion decision lacks validation seeds 2001-2005")
    if payload.get("frozen_hashes_unchanged") is not True:
        raise FinalizationError("promotion decision reports changed frozen files")
    promotion = payload.get("promotion")
    if not isinstance(promotion, Mapping):
        raise FinalizationError("promotion decision has no promotion object")
    checks = promotion.get("checks")
    improvement = _finite_positive(
        promotion.get("global_stability_improvement_fraction"),
        label="promotion global stability improvement",
    )
    priority_count = _positive_int(
        promotion.get("improved_priority_phase_count"),
        label="promotion improved priority phase count",
    )
    if (
        promotion.get("promoted") is not True
        or promotion.get("first_failed_gate") is not None
        or payload.get("first_failed_gate") is not None
        or not isinstance(checks, Mapping)
        or set(checks) != set(REQUIRED_PROMOTION_GATES)
        or any(checks.get(gate) is not True for gate in REQUIRED_PROMOTION_GATES)
        or improvement < 0.05
        or priority_count < 4
    ):
        raise FinalizationError("promotion decision did not pass every required gate")
    _, checkpoint = _declared_file(
        payload,
        path_key="candidate_checkpoint_path",
        hash_key="candidate_checkpoint_sha256",
        base=snapshot.path.parent,
        label="promotion candidate checkpoint",
    )
    if checkpoint["sha256"] != validation_checkpoint_hash:
        raise FinalizationError("validation aggregate and promotion checkpoint hashes differ")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FinalizationError("promotion decision omits its evaluation artifacts")
    declared = tuple(Path(str(value)).resolve() for value in artifacts.values())
    named = _require_named_paths(
        declared,
        required_names=_REQUIRED_VALIDATION_FILES,
        root=root,
        label="paired validation artifacts",
    )
    if named[PROMOTION_DECISION_FILENAME] != snapshot.path:
        raise FinalizationError("promotion artifact map does not name its own decision")
    return snapshot, tuple(
        _file_record(named[name], root=root) for name in _REQUIRED_VALIDATION_FILES
    )


def _validate_checkpoint_manifests(
    paths: Sequence[Path | str],
    *,
    root: Path,
    promotion: _JsonSnapshot,
    locked_test: _JsonSnapshot,
) -> tuple[tuple[dict[str, Any], ...], str, Path]:
    values = tuple(Path(value).resolve() for value in paths)
    if not values or len(values) != len(set(values)):
        raise FinalizationError("checkpoint_manifest_paths must be a non-empty unique sequence")
    snapshots = tuple(_load_json(path, label="checkpoint manifest") for path in values)
    allowed = {
        CHECKPOINT_MANIFEST_SCHEMA,
        CHECKPOINT_VALIDATION_PROMOTION_SCHEMA,
        CHECKPOINT_IMPROVED_PROMOTION_SCHEMA,
        INFERENCE_EXPORT_SCHEMA,
    }
    unknown = [row.payload.get("schema") for row in snapshots if row.payload.get("schema") not in allowed]
    if unknown:
        raise FinalizationError(f"unsupported checkpoint manifest schemas: {unknown}")
    # A supplied manifest is evidence, not decoration: validate every direct
    # checkpoint/path binding it carries even when it is not the final role.
    for snapshot in snapshots:
        payload = snapshot.payload
        schema = payload.get("schema")
        if schema == CHECKPOINT_MANIFEST_SCHEMA:
            _validate_checkpoint_contract(
                payload,
                base=snapshot.path.parent,
                label=f"checkpoint manifest {snapshot.path.name}",
            )
            _declared_file(
                payload,
                path_key="checkpoint_path",
                hash_key="checkpoint_sha256",
                base=snapshot.path.parent,
                label=f"checkpoint manifest {snapshot.path.name}",
            )
        elif schema == CHECKPOINT_VALIDATION_PROMOTION_SCHEMA:
            if (
                payload.get("valid") is not True
                or payload.get("status") != "PROMOTED_VALIDATION"
                or payload.get("filename_inference_used") is not False
            ):
                raise FinalizationError("best-validation promotion manifest is invalid")
            _declared_file(
                payload,
                path_key="source_checkpoint",
                hash_key="source_checkpoint_sha256",
                base=snapshot.path.parent,
                label="best-validation source checkpoint",
            )
            _declared_file(
                payload,
                path_key="promotion_decision",
                hash_key="promotion_decision_sha256",
                base=snapshot.path.parent,
                label="best-validation promotion decision",
            )
            _declared_file(
                payload,
                path_key="source_manifest",
                hash_key="source_manifest_sha256",
                base=snapshot.path.parent,
                label="best-validation source checkpoint manifest",
            )
        elif schema == INFERENCE_EXPORT_SCHEMA:
            if (
                payload.get("valid") is not True
                or payload.get("status") != "PASS"
                or payload.get("inference_only") is not True
                or payload.get("deterministic_mean_policy") is not True
                or payload.get("contains_critic") is not False
                or payload.get("contains_optimizer") is not False
                or payload.get("contains_rollout_state") is not False
                or payload.get("contains_stochastic_sampler") is not False
            ):
                raise FinalizationError("inference actor export manifest is invalid")
            _declared_file(
                payload,
                path_key="source_checkpoint",
                hash_key="source_checkpoint_sha256",
                base=snapshot.path.parent,
                label="inference actor source checkpoint",
            )
            _declared_file(
                payload,
                path_key="source_manifest",
                hash_key="source_manifest_sha256",
                base=snapshot.path.parent,
                label="inference actor source manifest",
            )
            torchscript = payload.get("torchscript")
            if not isinstance(torchscript, Mapping) or torchscript.get("valid") is not True:
                raise FinalizationError("inference actor TorchScript evidence is invalid")
            _declared_file(
                torchscript,
                path_key="path",
                hash_key="sha256",
                bytes_key="bytes",
                base=snapshot.path.parent,
                label="inference TorchScript actor",
            )
            onnx = payload.get("onnx")
            if isinstance(onnx, Mapping) and onnx.get("supported") is True:
                if onnx.get("valid") is not True:
                    raise FinalizationError("inference actor ONNX evidence is invalid")
                _declared_file(
                    onnx,
                    path_key="path",
                    hash_key="sha256",
                    bytes_key="bytes",
                    base=snapshot.path.parent,
                    label="inference ONNX actor",
                )
    checkpoint_rows = [
        row for row in snapshots if row.payload.get("schema") == CHECKPOINT_MANIFEST_SCHEMA
    ]
    improved = [
        row for row in checkpoint_rows if row.payload.get("publication_role") == "improved"
    ]
    final_promotions = [
        row
        for row in snapshots
        if row.payload.get("schema") == CHECKPOINT_IMPROVED_PROMOTION_SCHEMA
    ]
    if len(improved) != 1 or len(final_promotions) != 1:
        raise FinalizationError(
            "checkpoint manifests require exactly one content-authorized improved manifest "
            "and one final promotion manifest"
        )
    improved_snapshot = improved[0]
    improved_payload = improved_snapshot.payload
    for key in (
        "validation_promotion_authorized",
        "locked_test_authorized",
        "promotion_authorized",
    ):
        _require_true(improved_payload, key, label="improved checkpoint manifest")
    improved_path, improved_record = _declared_file(
        improved_payload,
        path_key="checkpoint_path",
        hash_key="checkpoint_sha256",
        base=improved_snapshot.path.parent,
        label="improved checkpoint",
    )
    _within(improved_path, root, label="improved checkpoint")
    if improved_path.name != IMPROVED_CHECKPOINT_NAME:
        raise FinalizationError("authorized improved checkpoint lacks the canonical final name")
    if improved_snapshot.path not in values:
        raise FinalizationError("improved checkpoint manifest was not supplied explicitly")
    best_checkpoint, best_checkpoint_record = _declared_file(
        improved_payload,
        path_key="source_best_validation_checkpoint",
        hash_key="source_best_validation_checkpoint_sha256",
        base=improved_snapshot.path.parent,
        label="improved source best-validation checkpoint",
    )
    best_manifest, best_manifest_record = _declared_file(
        improved_payload,
        path_key="source_best_validation_manifest",
        hash_key="source_best_validation_manifest_sha256",
        base=improved_snapshot.path.parent,
        label="improved source best-validation manifest",
    )
    validation_manifest, _ = _declared_file(
        improved_payload,
        path_key="validation_promotion_manifest",
        hash_key="validation_promotion_manifest_sha256",
        base=improved_snapshot.path.parent,
        label="improved validation promotion manifest",
    )
    if best_checkpoint_record["sha256"] != improved_record["sha256"]:
        raise FinalizationError("best-validation and improved checkpoint bytes differ")
    if best_manifest not in values or validation_manifest not in values:
        raise FinalizationError(
            "best-validation and validation-promotion manifests must be explicit inputs"
        )
    best_snapshot = next(row for row in snapshots if row.path == best_manifest)
    if (
        best_snapshot.payload.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or best_snapshot.payload.get("publication_role") != "best_validation"
        or best_snapshot.payload.get("validation_promotion_authorized") is not True
        or best_snapshot.payload.get("locked_test_authorized") is not False
    ):
        raise FinalizationError("source best-validation checkpoint manifest is invalid")
    best_decision, best_decision_record = _declared_file(
        best_snapshot.payload,
        path_key="promotion_decision",
        hash_key="promotion_decision_sha256",
        base=best_snapshot.path.parent,
        label="best-validation promotion decision",
    )
    if best_decision != promotion.path or best_decision_record["sha256"] != promotion.sha256:
        raise FinalizationError("best-validation manifest names a different promotion decision")
    validation_snapshot = next(row for row in snapshots if row.path == validation_manifest)
    validation_payload = validation_snapshot.payload
    validation_promotion = validation_payload.get("promotion")
    if (
        validation_payload.get("schema") != CHECKPOINT_VALIDATION_PROMOTION_SCHEMA
        or validation_payload.get("valid") is not True
        or validation_payload.get("status") != "PROMOTED_VALIDATION"
        or validation_payload.get("promotion_scope") != "best_validation_only"
        or validation_payload.get("improved_checkpoint_authorized") is not False
        or validation_payload.get("filename_inference_used") is not False
        or validation_payload.get("validation_seeds") != list(VALIDATION_SEEDS)
        or not isinstance(validation_promotion, Mapping)
        or validation_promotion.get("promoted") is not True
        or dict(validation_promotion) != dict(promotion.payload["promotion"])
    ):
        raise FinalizationError("validation promotion manifest is incomplete or inconsistent")
    validation_decision, validation_decision_record = _declared_file(
        validation_payload,
        path_key="promotion_decision",
        hash_key="promotion_decision_sha256",
        base=validation_snapshot.path.parent,
        label="validation promotion decision",
    )
    validation_source, validation_source_record = _declared_file(
        validation_payload,
        path_key="source_checkpoint",
        hash_key="source_checkpoint_sha256",
        base=validation_snapshot.path.parent,
        label="validation source checkpoint",
    )
    del validation_source
    validation_source_manifest, _ = _declared_file(
        validation_payload,
        path_key="source_manifest",
        hash_key="source_manifest_sha256",
        base=validation_snapshot.path.parent,
        label="validation source checkpoint manifest",
    )
    if (
        validation_decision != promotion.path
        or validation_decision_record["sha256"] != promotion.sha256
        or validation_source_record["sha256"] != improved_record["sha256"]
        or validation_source_manifest not in values
    ):
        raise FinalizationError("validation source provenance is inconsistent")
    published_best = validation_payload.get("published_best_validation")
    if not isinstance(published_best, Mapping):
        raise FinalizationError("validation promotion omits the published best checkpoint")
    validation_best, validation_best_record = _declared_file(
        published_best,
        path_key="path",
        hash_key="sha256",
        base=validation_snapshot.path.parent,
        label="validation published best checkpoint",
    )
    validation_best_manifest, validation_best_manifest_record = _declared_file(
        published_best,
        path_key="manifest",
        hash_key="manifest_sha256",
        base=validation_snapshot.path.parent,
        label="validation published best manifest",
    )
    if (
        validation_best != best_checkpoint
        or validation_best_record["sha256"] != best_checkpoint_record["sha256"]
        or validation_best_manifest != best_manifest
        or validation_best_manifest_record["sha256"] != best_manifest_record["sha256"]
    ):
        raise FinalizationError("validation publication and best checkpoint differ")
    for path_key, hash_key, supplied in (
        ("locked_test_aggregate", "locked_test_aggregate_sha256", locked_test),
        ("promotion_decision", "promotion_decision_sha256", promotion),
    ):
        if path_key not in improved_payload:
            # The improved manifest inherits promotion_decision from best-validation.
            raise FinalizationError(f"improved checkpoint manifest omits {path_key}")
        declared_path, record = _declared_file(
            improved_payload,
            path_key=path_key,
            hash_key=hash_key,
            base=improved_snapshot.path.parent,
            label=f"improved {path_key}",
        )
        if declared_path != supplied.path or record["sha256"] != supplied.sha256:
            raise FinalizationError(f"improved checkpoint {path_key} provenance differs")

    final_snapshot = final_promotions[0]
    final = final_snapshot.payload
    if (
        final.get("valid") is not True
        or final.get("status") != "PROMOTED_IMPROVED"
        or final.get("two_stage_promotion") is not True
        or final.get("validation_decision_alone_cannot_authorize_improved") is not True
        or final.get("filename_inference_used") is not False
        or final.get("byte_identical_best_and_improved") is not True
        or final.get("immutable_no_overwrite") is not True
    ):
        raise FinalizationError("final checkpoint promotion manifest is not authoritative")
    locked_path, locked_record = _declared_file(
        final,
        path_key="locked_test_aggregate",
        hash_key="locked_test_aggregate_sha256",
        base=final_snapshot.path.parent,
        label="final promotion locked-test aggregate",
    )
    if locked_path != locked_test.path or locked_record["sha256"] != locked_test.sha256:
        raise FinalizationError("final promotion names a different locked-test aggregate")
    published = final.get("published_checkpoints")
    final_improved = published.get("improved") if isinstance(published, Mapping) else None
    if not isinstance(final_improved, Mapping):
        raise FinalizationError("final promotion omits the published improved checkpoint")
    declared_checkpoint, declared_record = _declared_file(
        final_improved,
        path_key="path",
        hash_key="sha256",
        base=final_snapshot.path.parent,
        label="final published improved checkpoint",
    )
    declared_manifest, declared_manifest_record = _declared_file(
        final_improved,
        path_key="manifest",
        hash_key="manifest_sha256",
        base=final_snapshot.path.parent,
        label="final published improved manifest",
    )
    if (
        declared_checkpoint != improved_path
        or declared_record["sha256"] != improved_record["sha256"]
        or declared_manifest != improved_snapshot.path
        or declared_manifest_record["sha256"] != improved_snapshot.sha256
    ):
        raise FinalizationError("final promotion and improved checkpoint manifest differ")
    final_best = published.get("best_validation") if isinstance(published, Mapping) else None
    if not isinstance(final_best, Mapping):
        raise FinalizationError("final promotion omits the best-validation checkpoint")
    declared_best, declared_best_record = _declared_file(
        final_best,
        path_key="path",
        hash_key="sha256",
        base=final_snapshot.path.parent,
        label="final published best-validation checkpoint",
    )
    declared_best_manifest, declared_best_manifest_record = _declared_file(
        final_best,
        path_key="manifest",
        hash_key="manifest_sha256",
        base=final_snapshot.path.parent,
        label="final published best-validation manifest",
    )
    if (
        declared_best != best_checkpoint
        or declared_best_record["sha256"] != best_checkpoint_record["sha256"]
        or declared_best_manifest != best_manifest
        or declared_best_manifest_record["sha256"] != best_manifest_record["sha256"]
    ):
        raise FinalizationError("final promotion and best-validation provenance differ")
    final_validation_manifest, _ = _declared_file(
        final,
        path_key="validation_promotion_manifest",
        hash_key="validation_promotion_manifest_sha256",
        base=final_snapshot.path.parent,
        label="final validation promotion manifest",
    )
    if final_validation_manifest != validation_manifest:
        raise FinalizationError("final promotion names a different validation manifest")
    if final.get("validation_promotion") != validation_payload:
        raise FinalizationError("final promotion embeds different validation evidence")
    if _require_hash(
        promotion.payload.get("candidate_checkpoint_sha256"),
        label="promotion candidate checkpoint hash",
    ) != improved_record["sha256"]:
        raise FinalizationError("promoted checkpoint bytes differ from validation candidate")

    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        record = _snapshot_record(snapshot)
        record["schema"] = snapshot.payload.get("schema")
        record["publication_role"] = snapshot.payload.get("publication_role")
        records.append(record)
    records.append({**improved_record, "role": "improved_checkpoint"})
    return tuple(records), str(improved_record["sha256"]), improved_path


def _validate_locked_test(
    path: Path | str, *, expected_checkpoint_hash: str | None = None
) -> tuple[_JsonSnapshot, tuple[dict[str, Any], ...]]:
    snapshot, checkpoint_hash, workers = _validate_batch(
        path, role="candidate", seed_set="locked-test", seeds=LOCKED_TEST_SEEDS
    )
    payload = snapshot.payload
    if payload.get("finalized") is not True or payload.get("frozen_hashes_unchanged") is not True:
        raise FinalizationError("locked-test aggregate is not finalized with frozen hashes")
    gates = payload.get("hash_gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != set(REQUIRED_LOCKED_TEST_HASH_GATES)
        or any(gates.get(gate) is not True for gate in REQUIRED_LOCKED_TEST_HASH_GATES)
    ):
        raise FinalizationError("locked-test aggregate did not pass every hash gate")
    manifest_path, _ = _declared_file(
        payload,
        path_key="checkpoint_manifest",
        hash_key="checkpoint_manifest_sha256",
        base=snapshot.path.parent,
        label="locked-test checkpoint manifest",
    )
    del manifest_path
    if expected_checkpoint_hash is not None and checkpoint_hash != expected_checkpoint_hash:
        raise FinalizationError("locked-test checkpoint differs from promoted checkpoint")
    return snapshot, workers


def _validate_video_evidence(
    validation_path: Path | str,
    checksum_path: Path | str,
    *,
    root: Path,
    improved_checkpoint_hash: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    validation = _load_json(validation_path, label="video validation")
    payload = validation.payload
    if (
        payload.get("schema") != FINAL_VIDEO_SCHEMA
        or payload.get("valid") is not True
        or payload.get("status") != "PASS"
        or payload.get("immutable_no_overwrite") is not True
        or float(payload.get("maximum_duration_s", math.inf)) > 200.0
    ):
        raise FinalizationError("video validation manifest did not pass")
    explicit_checksum = Path(checksum_path).resolve()
    declared_checksum = _path(
        payload.get("video_checksum_manifest"),
        base=validation.path.parent,
        label="video checksum manifest",
    )
    if declared_checksum != explicit_checksum or explicit_checksum.name != VIDEO_CHECKSUM_NAME:
        raise FinalizationError("video validation names a different checksum manifest")
    _within(validation.path, root, label="video validation")
    _within(explicit_checksum, root, label="video checksum")
    try:
        verification = verify_checksum_manifest(explicit_checksum, root=root)
    except (ArtifactError, OSError) as exc:
        raise FinalizationError(f"video checksum manifest is invalid: {exc}") from exc
    if verification.get("valid") is not True:
        raise FinalizationError("video checksum manifest reports a SHA-256 mismatch")
    pair = payload.get("pair_evidence")
    if not isinstance(pair, Mapping) or any(
        pair.get(key) is not True
        for key in (
            "same_seed",
            "same_live_environment_contract",
            "same_obstacle_contract",
            "same_camera",
            "same_resolution",
            "same_initial_state",
        )
    ):
        raise FinalizationError("video pair comparability evidence is incomplete")
    videos = payload.get("videos")
    if not isinstance(videos, Mapping) or set(videos) != set(_REQUIRED_VIDEO_KEYS):
        raise FinalizationError("video validation does not contain the four canonical videos")
    records: list[dict[str, Any]] = []
    for key, filename in zip(_REQUIRED_VIDEO_KEYS, _REQUIRED_VIDEO_NAMES, strict=True):
        row = videos.get(key)
        if not isinstance(row, Mapping):
            raise FinalizationError(f"video validation row {key} is invalid")
        source, record = _declared_file(
            row,
            path_key="path",
            hash_key="sha256",
            bytes_key="bytes",
            base=validation.path.parent,
            label=f"video {key}",
        )
        _within(source, root, label=f"video {key}")
        try:
            duration = float(row.get("duration_s", math.inf))
            fps = float(row.get("fps", 0.0))
            frames = int(row.get("frame_count", 0))
        except (TypeError, ValueError) as exc:
            raise FinalizationError(f"video {key} numeric validation is invalid") from exc
        if (
            source.name != filename
            or row.get("valid") is not True
            or row.get("status") != "PASS"
            or not (0.0 < duration <= 200.0)
            or fps <= 0.0
            or frames < 2
            or row.get("codec") != "h264"
            or row.get("pixel_format") != "yuv420p"
            or row.get("pix_fmt") != "yuv420p"
            or row.get("full_decode") is not True
            or row.get("timestamps_monotonic") is not True
            or row.get("monotonic") is not True
            or row.get("stitched") is not False
            or row.get("speed_modified") is not False
        ):
            raise FinalizationError(f"video {key} failed technical acceptance")
        if key in {"ppo_improved", "ppo_diagnostic"} and _require_hash(
            row.get("source_checkpoint_sha256"),
            label=f"video {key} source checkpoint hash",
        ) != improved_checkpoint_hash:
            raise FinalizationError(f"video {key} was not rendered from improved checkpoint")
        records.append(record)
    sources = payload.get("source_episodes")
    if not isinstance(sources, Mapping) or set(sources) != {"fsm", "ppo"}:
        raise FinalizationError("video source episode provenance is incomplete")
    source_records: dict[str, Any] = {}
    for role in ("fsm", "ppo"):
        source_row = sources.get(role)
        if not isinstance(source_row, Mapping):
            raise FinalizationError(f"{role} video source evidence is invalid")
        directory = _path(
            source_row.get("directory"),
            base=validation.path.parent,
            label=f"{role} video source directory",
        )
        if not directory.is_dir() or directory.is_symlink():
            raise FinalizationError(f"{role} video source directory is missing: {directory}")
        verified_source_files: list[dict[str, Any]] = []
        for key in ("source_manifest", "viewport_ledger", "policy_trace", "raw_video"):
            source_file, record = _declared_file(
                source_row,
                path_key=key,
                hash_key=f"{key}_sha256",
                base=directory,
                label=f"{role} video {key}",
            )
            try:
                source_file.relative_to(directory)
            except ValueError as exc:
                raise FinalizationError(f"{role} video {key} escapes its source directory") from exc
            verified_source_files.append(record)
        source_records[role] = {
            "directory": str(directory),
            "files": verified_source_files,
        }
    ppo_source = sources["ppo"]
    ppo_checkpoint, ppo_checkpoint_record = _declared_file(
        ppo_source,
        path_key="checkpoint",
        hash_key="checkpoint_sha256",
        base=validation.path.parent,
        label="PPO video source checkpoint",
    )
    if ppo_checkpoint_record["sha256"] != improved_checkpoint_hash:
        raise FinalizationError("video source episode is not bound to improved checkpoint")
    source_records["ppo"]["checkpoint"] = ppo_checkpoint_record
    diagnostic = payload.get("diagnostic_ass")
    if not isinstance(diagnostic, Mapping):
        raise FinalizationError("video validation omits diagnostic overlay provenance")
    diagnostic_path, diagnostic_record = _declared_file(
        diagnostic,
        path_key="path",
        hash_key="sha256",
        base=validation.path.parent,
        label="diagnostic overlay",
    )
    checksum_entries = {
        str(entry.get("path")) for entry in verification.get("entries", ())
    }
    expected_checksum_entries = {
        path.relative_to(root).as_posix()
        for path in (
            *(Path(record["path"]) for record in records),
            validation.path,
            diagnostic_path,
        )
    }
    if checksum_entries != expected_checksum_entries:
        raise FinalizationError(
            "video checksum manifest does not cover exactly the published video bundle"
        )
    return (
        {
            "validation": _snapshot_record(validation),
            "video_checksums": _file_record(explicit_checksum, root=root),
            "video_checksum_verification": verification,
            "source_episodes": source_records,
            "diagnostic_overlay": diagnostic_record,
        },
        tuple(records),
    )


def _validate_reports_and_plots(
    report_paths: Sequence[Path | str],
    plot_paths: Sequence[Path | str],
    *,
    root: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    reports = _require_named_paths(
        report_paths, required_names=REPORT_FILENAMES, root=root, label="final reports"
    )
    plots = _require_named_paths(
        plot_paths, required_names=PLOT_FILENAMES, root=root, label="final plots"
    )
    for path in reports.values():
        try:
            if not path.read_text(encoding="utf-8").strip():
                raise FinalizationError(f"final report is empty: {path}")
        except UnicodeError as exc:
            raise FinalizationError(f"final report is not UTF-8: {path}") from exc
    for path in plots.values():
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise FinalizationError(f"final plot is not a PNG: {path}")
    return (
        tuple(_file_record(reports[name], root=root) for name in REPORT_FILENAMES),
        tuple(_file_record(plots[name], root=root) for name in PLOT_FILENAMES),
    )


def _output_inventory(
    root: Path, *, excluded: Iterable[Path]
) -> dict[str, tuple[Path, str, int]]:
    excluded_set = {path.resolve() for path in excluded}
    inventory: dict[str, tuple[Path, str, int]] = {}
    if not root.exists():
        return inventory
    if not root.is_dir() or root.is_symlink():
        raise FinalizationError(f"output_root is not a safe directory: {root}")
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise FinalizationError(f"output_root contains a symbolic link: {item}")
        if not item.is_file():
            continue
        resolved = item.resolve()
        if resolved in excluded_set:
            continue
        _within(resolved, root, label="delivery inventory")
        relative = resolved.relative_to(root).as_posix()
        inventory[relative] = (resolved, sha256_file(resolved), resolved.stat().st_size)
    return inventory


def _assert_inventory_unchanged(
    root: Path,
    inventory: Mapping[str, tuple[Path, str, int]],
    *,
    excluded: Iterable[Path],
) -> None:
    current = _output_inventory(root, excluded=excluded)
    if set(current) != set(inventory):
        raise FinalizationError("delivery inventory changed during finalization")
    for relative, (_, digest, size) in inventory.items():
        current_row = current[relative]
        if current_row[1] != digest or current_row[2] != size:
            raise FinalizationError(f"delivery artifact changed during finalization: {relative}")


def _assert_evidence_records_unchanged(payloads: Sequence[Mapping[str, Any]]) -> None:
    """Re-hash every absolute ``{path, sha256}`` record before publication."""

    checked: set[tuple[Path, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            raw_path = value.get("path")
            raw_hash = value.get("sha256")
            if raw_path is not None and raw_hash is not None:
                candidate = Path(str(raw_path))
                if candidate.is_absolute():
                    path = candidate.resolve()
                    digest = _require_hash(raw_hash, label=f"evidence record {path}")
                    key = (path, digest)
                    if key not in checked:
                        if not path.is_file() or sha256_file(path) != digest:
                            raise FinalizationError(
                                f"evidence artifact changed during finalization: {path}"
                            )
                        checked.add(key)
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    for payload in payloads:
        visit(payload)


def _publish_bundle(publications: Mapping[Path, bytes]) -> tuple[Path, ...]:
    for path, content in publications.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FinalizationError(f"refusing to overwrite final artifact: {path}")
    created: list[Path] = []
    try:
        for path, content in publications.items():
            if path.exists():
                continue
            try:
                _atomic_bytes(path, content)
                created.append(path)
            except ArtifactError as exc:
                if path.is_file() and path.read_bytes() == content:
                    # Another idempotent publisher won the race.
                    continue
                else:
                    raise FinalizationError(str(exc)) from exc
    except Exception:
        for path in reversed(created):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        raise
    return tuple(created)


@_fail_closed
def finalize_ppo_phase_delivery(
    *,
    output_root: Path | str,
    training_run_dirs: Sequence[Path | str],
    baseline_aggregate_path: Path | str,
    baseline_metric_paths: Sequence[Path | str],
    validation_aggregate_path: Path | str,
    promotion_decision_path: Path | str,
    locked_test_aggregate_path: Path | str,
    checkpoint_manifest_paths: Sequence[Path | str],
    video_validation_path: Path | str,
    video_checksum_path: Path | str,
    report_paths: Sequence[Path | str],
    plot_paths: Sequence[Path | str],
) -> FinalizationPaths:
    """Validate and publish final training/evaluation/checksum provenance.

    Every argument is explicit; no success state is inferred from a filename.
    The final checksum covers every regular file below ``output_root`` (except
    itself), including the retained ``video_checksums.sha256``.
    """

    root = Path(output_root).resolve()
    manifests = root / "manifests"
    paths = FinalizationPaths(
        output_root=root,
        training_manifest=manifests / "training_manifest.json",
        evaluation_manifest=manifests / "evaluation_manifest.json",
        checksums=manifests / "checksums.sha256",
    )
    destinations = (paths.training_manifest, paths.evaluation_manifest, paths.checksums)

    training_runs = _validate_training_runs(training_run_dirs)
    baseline, _, baseline_workers = _validate_batch(
        baseline_aggregate_path,
        role="baseline",
        seed_set="validation",
        seeds=VALIDATION_SEEDS,
    )
    baseline_metrics = _validate_baseline_metrics(
        baseline_metric_paths, root=root, aggregate=baseline
    )
    validation, validation_checkpoint_hash, validation_workers = _validate_batch(
        validation_aggregate_path,
        role="candidate",
        seed_set="validation",
        seeds=VALIDATION_SEEDS,
    )
    if validation_checkpoint_hash is None:
        raise FinalizationError("validation aggregate has no candidate checkpoint hash")
    promotion, validation_artifacts = _validate_promotion(
        promotion_decision_path,
        root=root,
        validation_checkpoint_hash=validation_checkpoint_hash,
    )
    locked, locked_workers = _validate_locked_test(locked_test_aggregate_path)
    checkpoint_manifests, improved_hash, improved_path = _validate_checkpoint_manifests(
        checkpoint_manifest_paths,
        root=root,
        promotion=promotion,
        locked_test=locked,
    )
    if improved_hash not in {
        str(record["immutable_history_checkpoint"]["sha256"])
        for record in training_runs
    }:
        raise FinalizationError("improved checkpoint is not byte-bound to a supplied training run")
    # Re-run the locked binding once the independently authorized final hash is known.
    locked, locked_workers = _validate_locked_test(
        locked_test_aggregate_path, expected_checkpoint_hash=improved_hash
    )
    video_evidence, videos = _validate_video_evidence(
        video_validation_path,
        video_checksum_path,
        root=root,
        improved_checkpoint_hash=improved_hash,
    )
    reports, plots = _validate_reports_and_plots(
        report_paths, plot_paths, root=root
    )

    training_payload: dict[str, Any] = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "valid": True,
        "status": "PASS",
        "success_inferred_from_filename": False,
        "paths_and_sha256_recomputed": True,
        "required_stages": list(REQUIRED_TRAINING_STAGES),
        "stage_sequence": [row["stage"] for row in training_runs],
        "training_run_count": len(training_runs),
        "terminal_global_policy_decisions": training_runs[-1][
            "global_policy_decisions"
        ],
        "terminal_checkpoint_sha256": training_runs[-1][
            "immutable_history_checkpoint"
        ]["sha256"],
        "all_run_lifecycles_succeeded": True,
        "all_frozen_hash_audits_passed": True,
        "all_save_load_round_trips_passed": True,
        "all_reward_telemetry_complete": True,
        "training_runs": list(training_runs),
    }
    evaluation_payload: dict[str, Any] = {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "valid": True,
        "status": "PASS",
        "improvement_claim_authorized": True,
        "success_inferred_from_filename": False,
        "paths_and_sha256_recomputed": True,
        "validation_seeds": list(VALIDATION_SEEDS),
        "locked_test_seeds": list(LOCKED_TEST_SEEDS),
        "baseline": {
            "aggregate": _snapshot_record(baseline),
            "metrics": list(baseline_metrics),
            "workers": list(baseline_workers),
            "pure_fsm_zero_residual": True,
            "passed": True,
        },
        "validation": {
            "aggregate": _snapshot_record(validation),
            "artifacts": list(validation_artifacts),
            "workers": list(validation_workers),
            "promotion_decision": _snapshot_record(promotion),
            "promotion_passed": True,
        },
        "locked_test": {
            "aggregate": _snapshot_record(locked),
            "workers": list(locked_workers),
            "finalized": True,
            "passed": True,
        },
        "checkpoint": {
            "improved_path": str(improved_path),
            "improved_sha256": improved_hash,
            "manifests": list(checkpoint_manifests),
            "two_stage_promotion_passed": True,
        },
        "video": {
            **video_evidence,
            "videos": list(videos),
            "all_duration_at_most_200_s": True,
            "all_full_decode": True,
        },
        "reports": list(reports),
        "plots": list(plots),
    }
    training_content = _json_bytes(training_payload)
    evaluation_content = _json_bytes(evaluation_payload)

    inventory = _output_inventory(root, excluded=destinations)
    checksum_hashes = {relative: row[1] for relative, row in inventory.items()}
    checksum_hashes[paths.training_manifest.relative_to(root).as_posix()] = hashlib.sha256(
        training_content
    ).hexdigest()
    checksum_hashes[paths.evaluation_manifest.relative_to(root).as_posix()] = hashlib.sha256(
        evaluation_content
    ).hexdigest()
    checksum_content = (
        "\n".join(
            f"{checksum_hashes[relative]}  {relative}"
            for relative in sorted(checksum_hashes)
        )
        + "\n"
    ).encode("utf-8")
    publications = {
        paths.training_manifest: training_content,
        paths.evaluation_manifest: evaluation_content,
        paths.checksums: checksum_content,
    }
    # Preflight all destinations before the final input/inventory recheck.
    for destination, content in publications.items():
        if destination.exists() and (
            not destination.is_file() or destination.read_bytes() != content
        ):
            raise FinalizationError(f"refusing to overwrite final artifact: {destination}")
    _assert_inventory_unchanged(root, inventory, excluded=destinations)
    _assert_evidence_records_unchanged((training_payload, evaluation_payload))
    created = _publish_bundle(publications)
    try:
        verification = verify_checksum_manifest(paths.checksums, root=root)
    except (ArtifactError, OSError) as exc:
        for path in reversed(created):
            if path.is_file():
                path.unlink()
        raise FinalizationError(f"final checksum verification failed: {exc}") from exc
    if verification.get("valid") is not True:
        for path in reversed(created):
            if path.is_file():
                path.unlink()
        raise FinalizationError("final checksum verification failed")
    final_inventory = _output_inventory(root, excluded=(paths.checksums,))
    if set(final_inventory) != set(checksum_hashes) or any(
        final_inventory[relative][1] != digest
        for relative, digest in checksum_hashes.items()
    ):
        for path in reversed(created):
            if path.is_file():
                path.unlink()
        raise FinalizationError("final checksum manifest is not a complete delivery inventory")
    return paths


# Short alias for callers that already scope output_root to ppo_phase_v1.
finalize_ppo_delivery = finalize_ppo_phase_delivery


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA",
    "FINAL_STAGE_ORDER",
    "FINAL_VIDEO_SCHEMA",
    "FinalizationError",
    "FinalizationPaths",
    "REQUIRED_TRAINING_STAGES",
    "TRAINING_MANIFEST_SCHEMA",
    "finalize_ppo_delivery",
    "finalize_ppo_phase_delivery",
]
