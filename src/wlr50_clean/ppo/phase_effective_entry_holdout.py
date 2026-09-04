"""Immutable external holdout gate for calibrated phase-entry resets.

The checked-in effective-entry contract is deliberately provisional.  This
module never runs Isaac and never changes that contract.  It accepts the
contract for phase-curriculum training only after twelve separately finalized
single-phase live probes (P02--P13, seed 1003) pass in distinct Python process
instances under one committed runtime/config/frozen identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactError, atomic_write_json, config_set_record, git_head
from .vector_benchmark_matrix import (
    VectorBenchmarkMatrixError,
    _SnapshotCache,
    _absolute_lexical,
    _load_object,
    _reject_links,
    _revalidate_snapshots,
    _snapshot,
    _validate_config_set_bytes,
    _validate_frozen_audit,
    _validate_runtime_identity_pair,
    _validated_file_record,
    validate_managed_run_directory,
)


ACCEPTANCE_SCHEMA = "wlr50_clean.phase_effective_entry_holdout_acceptance.v1"
TRAINING_EVIDENCE_SCHEMA = (
    "wlr50_clean.phase_effective_entry_holdout_training_evidence.v1"
)
OUTPUT_FILENAME = "phase_effective_entry_holdout_acceptance.json"
PROBE_FILENAME = "phase_snapshot_live_probe.json"
PROBE_SCHEMA = "wlr50_clean.phase_snapshot_live_probe.v2"
RUN_MANIFEST_SCHEMA = "wlr50_clean.ppo_run_manifest.v1"
HOLDOUT_SEED = 1003
HOLDOUT_PHASES = tuple(f"P{index:02d}" for index in range(2, 14))
HOLDOUT_CONFIG_RELATIVE_PATHS = (
    "configs/ppo_training_phase_v1.yaml",
    "configs/ppo_interface_v2.yaml",
    "configs/ppo_phase_effective_entry_v1.json",
    "configs/ppo_phase_effective_entry_v1.sha256",
    "configs/ppo_observation_schema_v2.json",
    "configs/ppo_phase_action_masks_v2.yaml",
    "configs/ppo_phase_objectives_v2.yaml",
    "configs/ppo_reward_v2.yaml",
    "configs/ppo_termination_v2.yaml",
    "configs/ppo_domain_randomization_v2.yaml",
    "configs/frozen_successful_fsm.yaml",
    "configs/environment_lock.json",
    "configs/fsm_states.yaml",
    "configs/recording_motion_contract.json",
    "configs/ppo_action_projection.yaml",
    "configs/ppo_observation_schema.json",
    "configs/conformance_policy.yaml",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_MANIFEST_RELATIVE = Path(
    "artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json"
)
_BACKEND_RELATIVE = Path("src/wlr50_clean/ppo/isaac_fsm_backend.py")
_PROCESS_INSTANCE = re.compile(r"^[0-9a-f]{32}$")


class PhaseEffectiveEntryHoldoutError(RuntimeError):
    """The external holdout evidence is incomplete, stale, or mutable."""


@dataclass(frozen=True, slots=True)
class _CurrentContext:
    project_root: Path
    git_commit: str
    config_sha256: str
    config_records: tuple[Mapping[str, Any], ...]
    frozen_manifest_path: Path
    frozen_manifest: Mapping[str, Any]
    frozen_manifest_sha256: str
    backend_path: Path
    backend_sha256: str
    snapshot_bundle: Any
    effective_entry_contract: Any


def _utc_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_paths(
    project_root: Path, values: Sequence[Path | str] | None
) -> tuple[Path, ...]:
    raw = HOLDOUT_CONFIG_RELATIVE_PATHS if values is None else tuple(values)
    if not raw:
        raise PhaseEffectiveEntryHoldoutError("holdout config set cannot be empty")
    result = tuple(
        _absolute_lexical(value if Path(value).is_absolute() else project_root / value)
        for value in raw
    )
    if len(result) != len(set(result)):
        raise PhaseEffectiveEntryHoldoutError("holdout config set contains duplicates")
    for path in result:
        _reject_links(path, root=project_root, label="holdout config")
        if not path.is_file():
            raise PhaseEffectiveEntryHoldoutError(
                f"holdout config file is missing: {path}"
            )
    return result


def _current_context(
    project_root: Path | str,
    *,
    config_paths: Sequence[Path | str] | None,
    snapshot_bundle: Any | None,
    effective_entry_contract: Any | None,
    cache: _SnapshotCache,
) -> _CurrentContext:
    root = _absolute_lexical(project_root)
    if not root.is_dir():
        raise PhaseEffectiveEntryHoldoutError(f"project root is missing: {root}")
    configs = _config_paths(root, config_paths)
    config_sha, config_records = config_set_record(configs, project_root=root)
    _validate_config_set_bytes(
        config_records,
        expected_sha256=config_sha,
        project_root=root,
        cache=cache,
    )
    frozen_path = root / _FROZEN_MANIFEST_RELATIVE
    _reject_links(frozen_path, root=root, label="frozen FSM manifest")
    frozen_snapshot = _snapshot(
        frozen_path,
        label="frozen FSM manifest",
        cache=cache,
        trusted_root=root,
    )
    frozen = _load_object(frozen_path, label="frozen FSM manifest", cache=cache)
    protected = frozen.get("protected_files")
    if (
        frozen.get("algorithm") != "sha256"
        or not isinstance(protected, Mapping)
        or not protected
        or not isinstance(frozen.get("source_head"), str)
    ):
        raise PhaseEffectiveEntryHoldoutError("frozen FSM manifest is malformed")
    for relative, expected in protected.items():
        if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise PhaseEffectiveEntryHoldoutError("frozen FSM file record is malformed")
        source = root / relative
        _reject_links(source, root=root, label="frozen FSM file")
        if _snapshot(source, label="frozen FSM file", cache=cache).sha256 != expected:
            raise PhaseEffectiveEntryHoldoutError(
                f"current frozen FSM file differs from its ledger: {relative}"
            )
    backend_path = root / _BACKEND_RELATIVE
    backend = _snapshot(
        backend_path,
        label="current Isaac FSM backend",
        cache=cache,
        trusted_root=root,
    )
    if snapshot_bundle is None:
        from .phase_snapshots import capture_validated_phase_snapshot_bundle

        snapshot_root = root / "reference" / "ppo_phase_snapshots"
        snapshot_bundle = capture_validated_phase_snapshot_bundle(
            snapshot_root, canonical_root=snapshot_root
        )
    if effective_entry_contract is None:
        from .phase_effective_entry import (
            capture_validated_effective_phase_entry_contract,
        )

        effective_entry_contract = capture_validated_effective_phase_entry_contract(
            root / "configs" / "ppo_phase_effective_entry_v1.json",
            expected_snapshot_bundle=snapshot_bundle,
            environment_lock_path=root / "configs" / "environment_lock.json",
            frozen_ledger_path=frozen_path,
        )
    if (
        getattr(effective_entry_contract, "phase_snapshot_bundle_sha256", None)
        != getattr(snapshot_bundle, "bundle_sha256", None)
    ):
        raise PhaseEffectiveEntryHoldoutError(
            "current effective-entry contract is bound to another snapshot bundle"
        )
    return _CurrentContext(
        project_root=root,
        git_commit=git_head(root),
        config_sha256=config_sha,
        config_records=tuple(dict(row) for row in config_records),
        frozen_manifest_path=frozen_path,
        frozen_manifest=dict(frozen),
        frozen_manifest_sha256=frozen_snapshot.sha256,
        backend_path=backend_path,
        backend_sha256=backend.sha256,
        snapshot_bundle=snapshot_bundle,
        effective_entry_contract=effective_entry_contract,
    )


def _one_invocation_value(arguments: Any, flag: str) -> str:
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise PhaseEffectiveEntryHoldoutError("managed invocation evidence is missing")
    rows = tuple(str(value) for value in arguments)
    positions = tuple(index for index, value in enumerate(rows) if value == flag)
    if len(positions) != 1 or positions[0] + 1 >= len(rows):
        raise PhaseEffectiveEntryHoldoutError(
            f"managed invocation must contain exactly one {flag} value"
        )
    return rows[positions[0] + 1]


def _invocation_values(arguments: Any, flag: str) -> tuple[str, ...]:
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise PhaseEffectiveEntryHoldoutError("managed invocation evidence is missing")
    rows = tuple(str(value) for value in arguments)
    result: list[str] = []
    for index, value in enumerate(rows):
        if value == flag:
            if index + 1 >= len(rows):
                raise PhaseEffectiveEntryHoldoutError(
                    f"managed invocation has a value-less {flag}"
                )
            result.append(rows[index + 1])
    return tuple(result)


def _require_invocation_flag(arguments: Any, flag: str) -> None:
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise PhaseEffectiveEntryHoldoutError("managed invocation evidence is missing")
    if sum(str(value) == flag for value in arguments) != 1:
        raise PhaseEffectiveEntryHoldoutError(
            f"managed invocation must contain exactly one {flag}"
        )


def _stdout_objects(path: Path, *, cache: _SnapshotCache) -> tuple[Mapping[str, Any], ...]:
    captured = _snapshot(path, label="managed stdout", cache=cache)
    try:
        text = captured.data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PhaseEffectiveEntryHoldoutError("managed stdout is not UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        value = line.strip()
        if not value.startswith("{") or not value.endswith("}"):
            continue
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(payload)
    return tuple(rows)


def _validate_started_and_final_manifest(
    run_dir: Path,
    *,
    run_kind: str,
    entrypoint: str,
    subcommand: str,
    training_stage: str,
    context: _CurrentContext,
    cache: _SnapshotCache,
    expected_seed: int = HOLDOUT_SEED,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    final = _load_object(
        run_dir / "run_manifest.json", label="finalized run manifest", cache=cache
    )
    started = _load_object(
        run_dir / "run_manifest.started.json", label="started run manifest", cache=cache
    )
    required_started_fields = {
        "schema",
        "lifecycle",
        "immutable_run_directory",
        "run_id",
        "run_kind",
        "run_dir",
        "project_root",
        "identity",
        "configs",
        "entrypoint",
        "subcommand",
        "invocation_arguments",
    }
    identity = final.get("identity")
    if (
        final.get("schema") != RUN_MANIFEST_SCHEMA
        or final.get("lifecycle") != "SUCCEEDED"
        or final.get("exit_code") != 0
        or final.get("immutable_run_directory") is not True
        or final.get("run_kind") != run_kind
        or final.get("entrypoint") != entrypoint
        or final.get("subcommand") != subcommand
        or final.get("run_id") != run_dir.name
        or _absolute_lexical(str(final.get("run_dir", ""))) != run_dir
        or _absolute_lexical(str(final.get("project_root", "")))
        != context.project_root
        or not isinstance(identity, Mapping)
        or identity.get("git_commit") != context.git_commit
        or identity.get("config_sha256") != context.config_sha256
        or set(identity)
        != {
            "timestamp_utc",
            "git_commit",
            "config_sha256",
            "seed",
            "environment_count",
            "training_stage",
        }
        or identity.get("seed") != expected_seed
        or identity.get("environment_count") != 1
        or identity.get("training_stage") != training_stage
        or final.get("configs") != list(context.config_records)
        or set(started) != required_started_fields
        or started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
    ):
        raise PhaseEffectiveEntryHoldoutError(
            f"{run_kind} lacks the required successful immutable run identity"
        )
    for key, value in started.items():
        if key != "lifecycle" and final.get(key) != value:
            raise PhaseEffectiveEntryHoldoutError(
                f"finalized run changed started field {key!r}"
            )
    _validated_file_record(
        run_dir,
        final.get("started_manifest"),
        expected_relative_path="run_manifest.started.json",
        label="started run manifest",
        cache=cache,
    )
    if not isinstance(final.get("logs"), Mapping) or not isinstance(
        final.get("artifacts"), Mapping
    ):
        raise PhaseEffectiveEntryHoldoutError(
            "finalized run omits log or artifact digest maps"
        )
    _validate_config_set_bytes(
        final["configs"],
        expected_sha256=context.config_sha256,
        project_root=context.project_root,
        cache=cache,
    )
    return final, started


def _validate_runtime_and_frozen_pair(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    context: _CurrentContext,
    cache: _SnapshotCache,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    before_record, after_record = _validate_runtime_identity_pair(
        run_dir,
        manifest=manifest,
        project_root=context.project_root,
        expected_git_commit=context.git_commit,
        cache=cache,
    )
    runtime = _load_object(
        run_dir / "committed_runtime_identity.before.json",
        label="committed runtime identity before",
        cache=cache,
    )
    runtime_after = _load_object(
        run_dir / "committed_runtime_identity.after.json",
        label="committed runtime identity after",
        cache=cache,
    )
    if runtime != runtime_after:
        raise PhaseEffectiveEntryHoldoutError(
            "committed runtime identity changed during managed run"
        )
    backend_rows = [
        row
        for row in runtime.get("files", ())
        if isinstance(row, Mapping) and row.get("path") == _BACKEND_RELATIVE.as_posix()
    ]
    if len(backend_rows) != 1 or backend_rows[0].get("sha256") != context.backend_sha256:
        raise PhaseEffectiveEntryHoldoutError(
            "managed runtime identity does not bind the current Isaac backend"
        )
    artifacts = manifest["artifacts"]
    frozen_records: dict[str, Any] = {}
    audits: list[Mapping[str, Any]] = []
    frozen_snapshot = _snapshot(
        context.frozen_manifest_path,
        label="frozen FSM manifest",
        cache=cache,
    )
    for position in ("before", "after"):
        relative = f"frozen_hashes.{position}.json"
        captured = _validated_file_record(
            run_dir,
            artifacts.get(relative),
            expected_relative_path=relative,
            label=f"frozen hash audit {position}",
            cache=cache,
        )
        audits.append(
            _validate_frozen_audit(
                run_dir / relative,
                expected_project_root=context.project_root,
                expected_manifest_path=context.frozen_manifest_path,
                manifest=context.frozen_manifest,
                manifest_snapshot=frozen_snapshot,
                cache=cache,
            )
        )
        frozen_records[position] = captured.record()
    comparable = (
        "project_root",
        "frozen_manifest",
        "frozen_manifest_sha256",
        "source_head",
        "protected_file_count",
        "passed",
        "mismatches",
        "entries",
    )
    if any(audits[0].get(key) != audits[1].get(key) for key in comparable):
        raise PhaseEffectiveEntryHoldoutError(
            "frozen FSM evidence changed during managed run"
        )
    return before_record, after_record, str(runtime["content_sha256"])


def _validate_probe_worker(
    run_dir_value: Path | str,
    *,
    context: _CurrentContext,
    cache: _SnapshotCache,
) -> dict[str, Any]:
    run_dir = validate_managed_run_directory(
        run_dir_value,
        project_root=context.project_root,
        run_kind="phase_snapshot_live_probe",
    )
    manifest, _ = _validate_started_and_final_manifest(
        run_dir,
        run_kind="phase_snapshot_live_probe",
        entrypoint="wlr50_clean.ppo.cli",
        subcommand="phase-snapshot-live-probe",
        training_stage="phase-snapshot-live-probe",
        context=context,
        cache=cache,
    )
    invocation = manifest.get("invocation_arguments")
    phase = _one_invocation_value(invocation, "--phase")
    if (
        phase not in HOLDOUT_PHASES
        or _one_invocation_value(invocation, "--seed") != str(HOLDOUT_SEED)
        or _one_invocation_value(invocation, "--num-envs") != "1"
        or _one_invocation_value(invocation, "--phase-snapshot-prime-physics-steps")
        != "1"
        or _one_invocation_value(invocation, "--seed-set") != "train"
    ):
        raise PhaseEffectiveEntryHoldoutError(
            "phase holdout worker invocation differs from the locked one-phase gate"
        )
    _require_invocation_flag(invocation, "--deterministic")
    artifacts = manifest["artifacts"]
    probe_snapshot = _validated_file_record(
        run_dir,
        artifacts.get(PROBE_FILENAME),
        expected_relative_path=PROBE_FILENAME,
        label="phase holdout probe",
        cache=cache,
    )
    live_result_snapshot = _validated_file_record(
        run_dir,
        artifacts.get("live_command_result.json"),
        expected_relative_path="live_command_result.json",
        label="phase holdout live command result",
        cache=cache,
    )
    logs = manifest["logs"]
    stdout_snapshot = _validated_file_record(
        run_dir,
        logs.get("stdout.log"),
        expected_relative_path="stdout.log",
        label="phase holdout stdout",
        cache=cache,
    )
    stderr_snapshot = _validated_file_record(
        run_dir,
        logs.get("stderr.log"),
        expected_relative_path="stderr.log",
        label="phase holdout stderr",
        cache=cache,
    )
    runtime_before, runtime_after, runtime_content_sha = (
        _validate_runtime_and_frozen_pair(
            run_dir, manifest, context=context, cache=cache
        )
    )
    probe = _load_object(run_dir / PROBE_FILENAME, label="phase holdout probe", cache=cache)
    live_result = _load_object(
        run_dir / "live_command_result.json",
        label="phase holdout live command result",
        cache=cache,
    )
    attempts = probe.get("attempts")
    process_id = probe.get("probe_process_id")
    process_instance = probe.get("probe_process_instance_id")
    if (
        live_result.get("schema") != "wlr50_clean.live_command_result.v1"
        or live_result.get("command") != "phase-snapshot-live-probe"
        or live_result.get("exit_code") != 0
        or probe.get("schema") != PROBE_SCHEMA
        or probe.get("artifact_role") != "DIAGNOSTIC_ONLY_NOT_TRAINING_ACCEPTANCE"
        or probe.get("status") != "PASSED"
        or probe.get("passed") is not True
        or probe.get("complete") is not True
        or probe.get("seed") != HOLDOUT_SEED
        or probe.get("phases") != [phase]
        or probe.get("phase_count") != 1
        or probe.get("phase_selector_mode") != "single_phase"
        or probe.get("attempts_per_phase") != 2
        or probe.get("expected_attempt_count") != 2
        or probe.get("completed_attempt_count") != 2
        or probe.get("expected_fresh_scene_attempt_count") != 1
        or probe.get("expected_reused_scene_attempt_count") != 1
        or probe.get("fresh_scene_attempt_count") != 1
        or probe.get("reused_scene_attempt_count") != 1
        or probe.get("failure_reasons") != []
        or probe.get("failure_classification") is not None
        or isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or not isinstance(process_instance, str)
        or _PROCESS_INSTANCE.fullmatch(process_instance) is None
        or not isinstance(attempts, list)
        or len(attempts) != 2
    ):
        raise PhaseEffectiveEntryHoldoutError(
            f"phase holdout probe is incomplete or failed for {phase}"
        )
    from .phase_snapshot_live_probe import _attempt_passed

    expected_attempts = (
        ("primary", "fresh_scene", False),
        ("reused_repeat", "reused_scene", True),
    )
    for row, (kind, lifecycle, existed) in zip(attempts, expected_attempts, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("phase") != phase
            or row.get("attempt_kind") != kind
            or row.get("scene_lifecycle") != lifecycle
            or row.get("scene_existed_before") is not existed
            or row.get("reset_completed") is not True
            or row.get("exception") is not None
            or row.get("failure_classification") is not None
            or row.get("passed") is not True
            or _attempt_passed(row) is not True
        ):
            raise PhaseEffectiveEntryHoldoutError(
                f"phase holdout attempt proof is invalid: {phase}[{kind}]"
            )
    snapshot_record = context.snapshot_bundle.as_record()
    effective_record = context.effective_entry_contract.as_record()
    if (
        probe.get("phase_snapshot_bundle") != snapshot_record
        or probe.get("phase_effective_entry_contract") != effective_record
    ):
        raise PhaseEffectiveEntryHoldoutError(
            "phase holdout probe differs from the current snapshot/effective contract"
        )
    runtime_reference = probe.get("runtime_identity_before")
    frozen_reference = probe.get("frozen_hashes_before")
    post = probe.get("managed_post_checks")
    if (
        not isinstance(runtime_reference, Mapping)
        or Path(str(runtime_reference.get("path", ""))).resolve()
        != (run_dir / "committed_runtime_identity.before.json").resolve()
        or runtime_reference.get("sha256") != runtime_before["sha256"]
        or not isinstance(frozen_reference, Mapping)
        or Path(str(frozen_reference.get("path", ""))).resolve()
        != (run_dir / "frozen_hashes.before.json").resolve()
        or frozen_reference.get("sha256")
        != _snapshot(
            run_dir / "frozen_hashes.before.json",
            label="frozen hash audit before",
            cache=cache,
        ).sha256
        or not isinstance(post, Mapping)
        or Path(str(post.get("runtime_identity_after", ""))).resolve()
        != (run_dir / "committed_runtime_identity.after.json").resolve()
        or Path(str(post.get("frozen_hashes_after", ""))).resolve()
        != (run_dir / "frozen_hashes.after.json").resolve()
        or Path(str(post.get("sealed_by_run_manifest", ""))).resolve()
        != (run_dir / "run_manifest.json").resolve()
    ):
        raise PhaseEffectiveEntryHoldoutError(
            "phase holdout probe does not bind its managed before/after checks"
        )
    stdout_matches = [row for row in _stdout_objects(stdout_snapshot.path, cache=cache) if row == probe]
    if len(stdout_matches) != 1:
        raise PhaseEffectiveEntryHoldoutError(
            "phase holdout probe is not exactly bound to finalized stdout"
        )
    return {
        "schema": "wlr50_clean.phase_effective_entry_holdout_worker.v1",
        "phase": phase,
        "seed": HOLDOUT_SEED,
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "probe_process_id": process_id,
        "probe_process_instance_id": process_instance,
        "source_git_commit": context.git_commit,
        "committed_runtime_content_sha256": runtime_content_sha,
        "backend_sha256": context.backend_sha256,
        "config_sha256": context.config_sha256,
        "frozen_manifest_sha256": context.frozen_manifest_sha256,
        "phase_snapshot_bundle_sha256": context.snapshot_bundle.bundle_sha256,
        "phase_effective_entry_contract_sha256": (
            context.effective_entry_contract.contract_sha256
        ),
        "run_manifest": _snapshot(
            run_dir / "run_manifest.json", label="worker run manifest", cache=cache
        ).record(),
        "probe": probe_snapshot.record(),
        "live_command_result": live_result_snapshot.record(),
        "stdout": stdout_snapshot.record(),
        "stderr": stderr_snapshot.record(),
        "committed_runtime_identity_before": runtime_before,
        "committed_runtime_identity_after": runtime_after,
        "frozen_hashes_before": _snapshot(
            run_dir / "frozen_hashes.before.json",
            label="worker frozen before",
            cache=cache,
        ).record(),
        "frozen_hashes_after": _snapshot(
            run_dir / "frozen_hashes.after.json",
            label="worker frozen after",
            cache=cache,
        ).record(),
        "fresh_attempt_passed": True,
        "reused_attempt_passed": True,
    }


def _validate_worker_set(workers: Sequence[Mapping[str, Any]]) -> None:
    if len(workers) != len(HOLDOUT_PHASES):
        raise PhaseEffectiveEntryHoldoutError(
            "holdout requires exactly twelve one-phase workers"
        )
    if tuple(row.get("phase") for row in workers) != HOLDOUT_PHASES:
        raise PhaseEffectiveEntryHoldoutError(
            "holdout workers must cover P02 through P13 exactly once"
        )
    run_dirs = tuple(row.get("run_dir") for row in workers)
    instances = tuple(row.get("probe_process_instance_id") for row in workers)
    if len(set(run_dirs)) != len(workers) or len(set(instances)) != len(workers):
        raise PhaseEffectiveEntryHoldoutError(
            "holdout workers are not twelve distinct managed process instances"
        )
    common_fields = (
        "seed",
        "source_git_commit",
        "committed_runtime_content_sha256",
        "backend_sha256",
        "config_sha256",
        "frozen_manifest_sha256",
        "phase_snapshot_bundle_sha256",
        "phase_effective_entry_contract_sha256",
    )
    for field in common_fields:
        if len({json.dumps(row.get(field), sort_keys=True) for row in workers}) != 1:
            raise PhaseEffectiveEntryHoldoutError(
                f"holdout workers disagree on {field}"
            )


def _acceptance_payload(
    workers: Sequence[Mapping[str, Any]], context: _CurrentContext
) -> dict[str, Any]:
    rows = tuple(sorted((dict(row) for row in workers), key=lambda row: row["phase"]))
    _validate_worker_set(rows)
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "artifact_role": "PHASE_EFFECTIVE_ENTRY_EXTERNAL_HOLDOUT_ACCEPTANCE",
        "status": "PASSED",
        "passed": True,
        "created_at_utc": _utc_text(),
        "seed": HOLDOUT_SEED,
        "phases": list(HOLDOUT_PHASES),
        "phase_count": len(HOLDOUT_PHASES),
        "worker_count": len(rows),
        "independent_fresh_process_per_phase": True,
        "fresh_and_reused_attempts_passed_per_worker": True,
        "source_git_commit": context.git_commit,
        "committed_runtime_content_sha256": rows[0][
            "committed_runtime_content_sha256"
        ],
        "backend_path": str(context.backend_path),
        "backend_sha256": context.backend_sha256,
        "config_sha256": context.config_sha256,
        "configs": [dict(row) for row in context.config_records],
        "frozen_manifest": {
            "path": str(context.frozen_manifest_path),
            "sha256": context.frozen_manifest_sha256,
            "source_head": context.frozen_manifest.get("source_head"),
        },
        "phase_snapshot_bundle_sha256": context.snapshot_bundle.bundle_sha256,
        "phase_snapshot_bundle": context.snapshot_bundle.as_record(),
        "phase_effective_entry_contract_sha256": (
            context.effective_entry_contract.contract_sha256
        ),
        "phase_effective_entry_contract": (
            context.effective_entry_contract.as_record()
        ),
        "worker_set_sha256": _canonical_sha256(rows),
        "workers": list(rows),
    }


def aggregate_phase_effective_entry_holdout(
    probe_run_dirs: Sequence[Path | str],
    output_path: Path | str,
    *,
    project_root: Path | str = PROJECT_ROOT,
    config_paths: Sequence[Path | str] | None = None,
    snapshot_bundle: Any | None = None,
    effective_entry_contract: Any | None = None,
) -> Mapping[str, Any]:
    """Validate twelve finalized live workers and publish one no-clobber gate."""

    try:
        cache: _SnapshotCache = {}
        context = _current_context(
            project_root,
            config_paths=config_paths,
            snapshot_bundle=snapshot_bundle,
            effective_entry_contract=effective_entry_contract,
            cache=cache,
        )
        if len(probe_run_dirs) != len(HOLDOUT_PHASES):
            raise PhaseEffectiveEntryHoldoutError(
                "holdout aggregation requires exactly twelve probe run directories"
            )
        if len({_absolute_lexical(path) for path in probe_run_dirs}) != len(
            probe_run_dirs
        ):
            raise PhaseEffectiveEntryHoldoutError(
                "holdout aggregation contains duplicate probe run directories"
            )
        workers = [
            _validate_probe_worker(path, context=context, cache=cache)
            for path in probe_run_dirs
        ]
        if tuple(row["phase"] for row in workers) != HOLDOUT_PHASES:
            raise PhaseEffectiveEntryHoldoutError(
                "holdout probe run directories must be supplied in exact "
                "P02-through-P13 order"
            )
        payload = _acceptance_payload(workers, context)
        output = _absolute_lexical(output_path)
        run_dir = validate_managed_run_directory(
            output.parent,
            project_root=context.project_root,
            run_kind="phase_effective_entry_holdout",
        )
        if output != run_dir / OUTPUT_FILENAME:
            raise PhaseEffectiveEntryHoldoutError(
                f"holdout output must be named {OUTPUT_FILENAME} in its managed run"
            )
        _revalidate_snapshots(cache, project_root=context.project_root)
        atomic_write_json(output, payload)
        _revalidate_snapshots(cache, project_root=context.project_root)
        return payload
    except PhaseEffectiveEntryHoldoutError:
        raise
    except (ArtifactError, VectorBenchmarkMatrixError, OSError, ValueError) as exc:
        raise PhaseEffectiveEntryHoldoutError(
            f"phase effective-entry holdout aggregation rejected: {exc}"
        ) from exc


def _validate_acceptance_run(
    acceptance_path: Path,
    *,
    context: _CurrentContext,
    cache: _SnapshotCache,
) -> Mapping[str, Any]:
    run_dir = validate_managed_run_directory(
        acceptance_path.parent,
        project_root=context.project_root,
        run_kind="phase_effective_entry_holdout",
    )
    if acceptance_path != run_dir / OUTPUT_FILENAME:
        raise PhaseEffectiveEntryHoldoutError(
            f"holdout acceptance must be named {OUTPUT_FILENAME}"
        )
    manifest, _ = _validate_started_and_final_manifest(
        run_dir,
        run_kind="phase_effective_entry_holdout",
        entrypoint="wlr50_clean.ppo.phase_effective_entry_holdout",
        subcommand="aggregate",
        training_stage="effective-entry-holdout-aggregation",
        context=context,
        cache=cache,
    )
    artifacts = manifest["artifacts"]
    _validated_file_record(
        run_dir,
        artifacts.get(OUTPUT_FILENAME),
        expected_relative_path=OUTPUT_FILENAME,
        label="phase effective-entry holdout acceptance",
        cache=cache,
    )
    _validate_runtime_and_frozen_pair(
        run_dir, manifest, context=context, cache=cache
    )
    invocation_dirs = tuple(
        _absolute_lexical(value)
        for value in _invocation_values(
            manifest.get("invocation_arguments"), "--probe-run-dir"
        )
    )
    if len(invocation_dirs) != len(HOLDOUT_PHASES):
        raise PhaseEffectiveEntryHoldoutError(
            "holdout aggregation invocation does not bind twelve probe runs"
        )
    return manifest


def validate_phase_effective_entry_holdout_acceptance(
    path: Path | str,
    *,
    project_root: Path | str = PROJECT_ROOT,
    config_paths: Sequence[Path | str] | None = None,
    snapshot_bundle: Any | None = None,
    effective_entry_contract: Any | None = None,
) -> dict[str, Any]:
    """Revalidate a finalized acceptance and all twelve source workers."""

    try:
        cache: _SnapshotCache = {}
        context = _current_context(
            project_root,
            config_paths=config_paths,
            snapshot_bundle=snapshot_bundle,
            effective_entry_contract=effective_entry_contract,
            cache=cache,
        )
        acceptance_path = _absolute_lexical(path)
        manifest = _validate_acceptance_run(
            acceptance_path, context=context, cache=cache
        )
        acceptance = _load_object(
            acceptance_path,
            label="phase effective-entry holdout acceptance",
            cache=cache,
        )
        workers_raw = acceptance.get("workers")
        if not isinstance(workers_raw, list):
            raise PhaseEffectiveEntryHoldoutError(
                "holdout acceptance worker records are missing"
            )
        workers = [
            _validate_probe_worker(
                str(row.get("run_dir", "")), context=context, cache=cache
            )
            for row in workers_raw
            if isinstance(row, Mapping)
        ]
        workers.sort(key=lambda row: row["phase"])
        expected = _acceptance_payload(workers, context)
        expected.pop("created_at_utc")
        received = dict(acceptance)
        created_at = received.pop("created_at_utc", None)
        if not isinstance(created_at, str) or received != expected:
            raise PhaseEffectiveEntryHoldoutError(
                "holdout acceptance record is stale, incomplete, or inconsistent"
            )
        invocation_dirs = tuple(
            str(_absolute_lexical(value))
            for value in _invocation_values(
                manifest.get("invocation_arguments"), "--probe-run-dir"
            )
        )
        if invocation_dirs != tuple(row["run_dir"] for row in workers):
            raise PhaseEffectiveEntryHoldoutError(
                "holdout aggregation invocation differs from accepted workers"
            )
        _revalidate_snapshots(cache, project_root=context.project_root)
        return {
            "schema": TRAINING_EVIDENCE_SCHEMA,
            "path": str(acceptance_path),
            "sha256": _snapshot(
                acceptance_path,
                label="phase effective-entry holdout acceptance",
                cache=cache,
            ).sha256,
            "phase_effective_entry_contract_sha256": (
                context.effective_entry_contract.contract_sha256
            ),
            "phase_snapshot_bundle_sha256": context.snapshot_bundle.bundle_sha256,
            "source_git_commit": context.git_commit,
            "backend_sha256": context.backend_sha256,
            "config_sha256": context.config_sha256,
            "run_manifest": str(acceptance_path.parent / "run_manifest.json"),
            "run_manifest_sha256": _snapshot(
                acceptance_path.parent / "run_manifest.json",
                label="holdout aggregation run manifest",
                cache=cache,
            ).sha256,
            "acceptance": dict(acceptance),
            "passed": True,
        }
    except PhaseEffectiveEntryHoldoutError:
        raise
    except (ArtifactError, VectorBenchmarkMatrixError, OSError, ValueError) as exc:
        raise PhaseEffectiveEntryHoldoutError(
            f"phase effective-entry holdout acceptance rejected: {exc}"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--seed", type=int, required=True)
    aggregate.add_argument("--num-envs", type=int, required=True)
    aggregate.add_argument("--probe-run-dir", type=Path, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seed != HOLDOUT_SEED or args.num_envs != 1:
        raise PhaseEffectiveEntryHoldoutError(
            "holdout aggregation is locked to seed 1003 and num-envs=1"
        )
    output = args.run_dir.resolve() / OUTPUT_FILENAME
    payload = aggregate_phase_effective_entry_holdout(
        args.probe_run_dir,
        output,
        project_root=PROJECT_ROOT,
    )
    print(
        json.dumps(
            {
                "schema": ACCEPTANCE_SCHEMA,
                "path": str(output),
                "sha256": _sha256(output),
                "passed": payload.get("passed") is True,
            },
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "HOLDOUT_CONFIG_RELATIVE_PATHS",
    "HOLDOUT_PHASES",
    "HOLDOUT_SEED",
    "OUTPUT_FILENAME",
    "PhaseEffectiveEntryHoldoutError",
    "TRAINING_EVIDENCE_SCHEMA",
    "aggregate_phase_effective_entry_holdout",
    "validate_phase_effective_entry_holdout_acceptance",
]
