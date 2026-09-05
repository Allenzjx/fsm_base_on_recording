"""Offline validation of chunked PPO training and deterministic screenings.

The live launchers remain the only producers of physics evidence.  This module
captures finalized run bytes once, proves the ordered train/screen chain, and
publishes one immutable pre-finalization manifest.  A failed physical screening
is valid evidence; only a complete five-seed promotion decision can stop work.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import yaml

from .artifacts import ArtifactError, atomic_write_json
from .training_cadence import (
    TrainingCadenceError, cadence_inputs_from_payload, derive_stage_cadence,
    derive_training_cadence, validate_training_chunk_cadence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ORCHESTRATION_SCHEMA = "wlr50_clean.ppo_training_orchestration.v2"
TRAINING_ORCHESTRATION_FILENAME = "training_orchestration_manifest.json"
TRAINING_ORCHESTRATION_RUN_KIND = "training-orchestration"
VECTOR_BENCHMARK_MATRIX_RUN_KIND = "vector-benchmark-matrix"
INITIAL_CHECKPOINT_RUN_KIND = "initial-checkpoint"
INITIAL_CHECKPOINT_PUBLICATION_RUN_KIND = "initial-checkpoint-publication"
DEFAULT_TRAINING_ORCHESTRATION_RUNS = (
    PROJECT_ROOT / "runs" / "ppo_phase_v1" / TRAINING_ORCHESTRATION_RUN_KIND
)
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "ppo_training_phase_v1.yaml"
DEFAULT_VECTOR_BENCHMARK_MATRIX = (
    PROJECT_ROOT / "runs" / "ppo_phase_v1" / VECTOR_BENCHMARK_MATRIX_RUN_KIND
)
RUN_MANIFEST_SCHEMA = "wlr50_clean.ppo_run_manifest.v1"
TRAINING_RESULT_SCHEMA = "wlr50_clean.ppo_training_run.v1"
SCREENING_RESULT_SCHEMA = "wlr50_clean.ppo_checkpoint_evaluation.v1"
CHECKPOINT_MANIFEST_SCHEMA = "wlr50_clean.phase_residual_checkpoint_manifest.v1"
PROMOTION_DECISION_SCHEMA = "wlr50_clean.ppo_evaluation_artifacts.v1"
FROZEN_AUDIT_SCHEMA = "wlr50_clean.frozen_fsm_hash_audit.v1"
COMMITTED_RUNTIME_IDENTITY_SCHEMA = "wlr50_clean.committed_runtime_identity.v1"
VECTOR_MATRIX_SCHEMA = "wlr50_clean.vector_benchmark_matrix.v1"
SOFT_RESET_SCHEMA = "wlr50_clean.soft_reset_equivalence_acceptance.v1"
STAGES = ("smoke", "phase-curriculum", "full-episode")
STAGE_BUDGETS = {
    "smoke": 10_000,
    "phase-curriculum": 100_000,
    "full-episode": 100_000,
}
DETERMINISTIC_VALIDATION_INTERVAL = 10_000
VALIDATION_SEEDS = (2001, 2002, 2003, 2004, 2005)
_RUNS_RELATIVE = Path("runs/ppo_phase_v1")
_VALIDATION_HISTORY_RELATIVE = Path("outputs/ppo_phase_v1/validation_history")
_FROZEN_RELATIVE = Path("artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json")
_COMMITTED_RUNTIME_ROOTS = (
    "src/wlr50_clean",
    "src/wlr50_clean/ppo",
    "src/wlr50_clean/fsm",
    "src/wlr50_clean/sensing",
    "src/wlr50_clean/infrastructure",
    "scripts",
    "configs",
    "reference/ppo_phase_snapshots",
    "artifacts/ppo_phase_v1_start",
    "pyproject.toml",
)
_HEX = frozenset("0123456789abcdef")


class TrainingOrchestrationError(RuntimeError):
    """The ordered training/screening evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    data: bytes
    size: int
    sha256: str
    creation_time_utc_ticks: int
    last_write_time_utc_ticks: int
    trusted_root: Path | None = None

    def record(self) -> dict[str, Any]:
        return {"path": str(self.path), "bytes": self.size, "sha256": self.sha256}


_Cache = MutableMapping[Path, _Snapshot]


def _utc_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_hash(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _HEX for character in value.lower())
    )


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingOrchestrationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingOrchestrationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingOrchestrationError(f"{label} must be positive and finite")
    return result


def _reject_links(path: Path, *, root: Path, label: str) -> None:
    target = _absolute(path)
    base = _absolute(root)
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise TrainingOrchestrationError(f"{label} escapes {base}") from exc
    cursors = [base]
    cursor = base
    for part in relative.parts:
        cursor /= part
        cursors.append(cursor)
    for cursor in cursors:
        try:
            status = os.stat(cursor, follow_symlinks=False)
            link = cursor.is_symlink()
            junction_fn = getattr(cursor, "is_junction", None)
            junction = bool(junction_fn()) if callable(junction_fn) else False
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TrainingOrchestrationError(f"cannot inspect {label}: {cursor}") from exc
        if (
            link
            or junction
            or int(getattr(status, "st_file_attributes", 0)) & 0x400
        ):
            raise TrainingOrchestrationError(
                f"{label} contains a symlink/junction: {cursor}"
            )


def _managed_run_dir(
    path: Path | str, *, project_root: Path, run_kind: str, label: str
) -> Path:
    directory = _absolute(path)
    runs_root = project_root / _RUNS_RELATIVE
    try:
        relative = directory.relative_to(runs_root)
    except ValueError as exc:
        raise TrainingOrchestrationError(
            f"{label} must be inside {runs_root}"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[0] != run_kind:
        raise TrainingOrchestrationError(
            f"{label} must be {runs_root / run_kind / '<run-id>'}"
        )
    _reject_links(directory, root=project_root, label=label)
    if not directory.is_dir():
        raise TrainingOrchestrationError(f"{label} is missing: {directory}")
    return directory


def _snapshot(
    path: Path | str,
    *,
    label: str,
    cache: _Cache,
    trusted_root: Path | None = None,
) -> _Snapshot:
    selected = _absolute(path)
    trusted = None if trusted_root is None else _absolute(trusted_root)
    if trusted is not None:
        _reject_links(selected, root=trusted, label=label)
    cached = cache.get(selected)
    if cached is not None:
        if trusted is not None and cached.trusted_root is None:
            cached = _Snapshot(
                path=cached.path,
                data=cached.data,
                size=cached.size,
                sha256=cached.sha256,
                creation_time_utc_ticks=cached.creation_time_utc_ticks,
                last_write_time_utc_ticks=cached.last_write_time_utc_ticks,
                trusted_root=trusted,
            )
            cache[selected] = cached
        return cached
    try:
        selected_status = os.stat(selected, follow_symlinks=False)
    except OSError as exc:
        raise TrainingOrchestrationError(f"cannot read {label}: {selected}") from exc
    if (
        selected.is_symlink()
        or (
            callable(getattr(selected, "is_junction", None))
            and selected.is_junction()
        )
        or int(getattr(selected_status, "st_file_attributes", 0)) & 0x400
    ):
        raise TrainingOrchestrationError(f"{label} cannot be a symlink/junction")
    try:
        with selected.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise TrainingOrchestrationError(
                    f"{label} is not a regular file: {selected}"
                )
            data = stream.read()
            after = os.fstat(stream.fileno())
    except TrainingOrchestrationError:
        raise
    except OSError as exc:
        raise TrainingOrchestrationError(f"cannot read {label}: {selected}") from exc
    if (
        before.st_size != len(data)
        or after.st_size != len(data)
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or getattr(before, "st_birthtime_ns", before.st_ctime_ns)
        != getattr(after, "st_birthtime_ns", after.st_ctime_ns)
    ):
        raise TrainingOrchestrationError(f"{label} changed while being captured")
    result = _Snapshot(
        path=selected,
        data=data,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        creation_time_utc_ticks=621_355_968_000_000_000
        + getattr(before, "st_birthtime_ns", before.st_ctime_ns) // 100,
        last_write_time_utc_ticks=621_355_968_000_000_000
        + before.st_mtime_ns // 100,
        trusted_root=trusted,
    )
    cache[selected] = result
    return result


def _secure_snapshot(
    path: Path | str, *, label: str, project_root: Path, cache: _Cache
) -> _Snapshot:
    """Capture a file and bind every later revalidation to its trusted root."""

    return _snapshot(
        path,
        label=label,
        cache=cache,
        trusted_root=project_root,
    )


def _json(path: Path | str, *, label: str, cache: _Cache) -> Mapping[str, Any]:
    captured = _snapshot(path, label=label, cache=cache)
    if not captured.data:
        raise TrainingOrchestrationError(f"{label} is empty: {captured.path}")
    try:
        payload = json.loads(captured.data.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingOrchestrationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise TrainingOrchestrationError(f"{label} must be a JSON object")
    return payload


def _record(
    value: Any,
    *,
    base: Path,
    expected_path: str | None,
    label: str,
    cache: _Cache,
) -> _Snapshot:
    if not isinstance(value, Mapping):
        raise TrainingOrchestrationError(f"{label} record is missing")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise TrainingOrchestrationError(f"{label} record path is invalid")
    if expected_path is not None and raw_path.replace("\\", "/") != expected_path:
        raise TrainingOrchestrationError(f"{label} record names the wrong path")
    path = Path(raw_path)
    selected = _absolute(path if path.is_absolute() else base / path)
    if not path.is_absolute():
        try:
            selected.relative_to(base)
        except ValueError as exc:
            raise TrainingOrchestrationError(f"{label} escapes its base") from exc
    captured = _snapshot(selected, label=label, cache=cache)
    if (
        isinstance(value.get("bytes"), bool)
        or value.get("bytes") != captured.size
        or not _is_hash(value.get("sha256"))
        or str(value.get("sha256")).lower() != captured.sha256
    ):
        raise TrainingOrchestrationError(f"{label} digest/size mismatch")
    return captured


def _parse_time(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise TrainingOrchestrationError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrainingOrchestrationError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise TrainingOrchestrationError(f"{label} timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _revalidate(
    cache: Mapping[Path, _Snapshot], *, project_root: Path | None = None
) -> None:
    for path, expected in sorted(cache.items(), key=lambda item: str(item[0])):
        trusted_root = project_root or expected.trusted_root
        if trusted_root is not None:
            _reject_links(
                path,
                root=trusted_root,
                label="captured source",
            )
        try:
            path_status = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise TrainingOrchestrationError(
                f"captured source disappeared before publication: {path}"
            ) from exc
        if (
            path.is_symlink()
            or (
                callable(getattr(path, "is_junction", None))
                and path.is_junction()
            )
            or int(getattr(path_status, "st_file_attributes", 0)) & 0x400
        ):
            raise TrainingOrchestrationError(
                f"captured source became a symlink/junction: {path}"
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise TrainingOrchestrationError(
                f"captured source disappeared before publication: {path}"
            ) from exc
        if len(data) != expected.size or hashlib.sha256(data).hexdigest() != expected.sha256:
            raise TrainingOrchestrationError(
                f"captured source changed before publication: {path}"
            )


def _canonical_hash(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_frozen_audit(
    path: Path,
    *,
    project_root: Path,
    frozen_manifest_path: Path,
    frozen_manifest: Mapping[str, Any],
    frozen_snapshot: _Snapshot,
    cache: _Cache,
) -> tuple[tuple[str, str], ...]:
    audit = _json(path, label="frozen hash audit", cache=cache)
    protected = frozen_manifest.get("protected_files")
    source_head = frozen_manifest.get("source_head")
    entries = audit.get("entries")
    if (
        frozen_manifest.get("algorithm") != "sha256"
        or not isinstance(protected, Mapping)
        or len(protected) != 29
        or not _is_hash(source_head, 40)
        or audit.get("schema") != FROZEN_AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("mismatches") != []
        or audit.get("protected_file_count") != 29
        or audit.get("source_head") != source_head
        or audit.get("frozen_manifest_sha256") != frozen_snapshot.sha256
        or _absolute(str(audit.get("project_root", ""))) != project_root
        or _absolute(str(audit.get("frozen_manifest", ""))) != frozen_manifest_path
        or not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes))
        or len(entries) != 29
    ):
        raise TrainingOrchestrationError("frozen hash audit is stale or incomplete")
    observed: dict[str, str] = {}
    for row in entries:
        if not isinstance(row, Mapping):
            raise TrainingOrchestrationError("frozen hash audit entry is malformed")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or relative in observed
            or relative not in protected
            or row.get("expected_sha256") != protected[relative]
            or row.get("actual_sha256") != protected[relative]
            or row.get("exists") is not True
            or row.get("valid") is not True
        ):
            raise TrainingOrchestrationError("frozen hash audit entry failed")
        source = project_root / relative
        _reject_links(source, root=project_root, label=f"frozen file {relative}")
        captured = _snapshot(source, label=f"frozen file {relative}", cache=cache)
        if captured.sha256 != protected[relative]:
            raise TrainingOrchestrationError(
                f"frozen file changed after audit: {relative}"
            )
        observed[relative] = captured.sha256
    if set(observed) != set(protected):
        raise TrainingOrchestrationError("frozen audit entry set is not exact")
    return tuple(sorted(observed.items()))


def _validate_config_records(
    records: Any, *, project_root: Path, expected_config_sha256: Any, cache: _Cache
) -> tuple[dict[str, Any], ...]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise TrainingOrchestrationError("run config records are missing")
    normalized: list[tuple[str, _Snapshot]] = []
    names: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise TrainingOrchestrationError(f"config record {index} is malformed")
        name = str(row["path"]).replace("\\", "/")
        if name in names or Path(name).is_absolute() or ".." in Path(name).parts:
            raise TrainingOrchestrationError("config paths are duplicated or unsafe")
        names.add(name)
        captured = _record(
            row,
            base=project_root,
            expected_path=name,
            label=f"config {name}",
            cache=cache,
        )
        normalized.append((name, captured))
    normalized.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    result: list[dict[str, Any]] = []
    for name, captured in normalized:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(captured.size.to_bytes(8, "big"))
        digest.update(captured.data)
        result.append({"path": name, "bytes": captured.size, "sha256": captured.sha256})
    if not _is_hash(expected_config_sha256) or digest.hexdigest() != expected_config_sha256:
        raise TrainingOrchestrationError("run config aggregate SHA-256 mismatch")
    return tuple(result)


@lru_cache(maxsize=8)
def _committed_runtime_paths(
    project_root_text: str, git_commit: str
) -> tuple[str, ...]:
    project_root = Path(project_root_text)
    command = [
        "git",
        "-C",
        str(project_root),
        "ls-tree",
        "-r",
        "--name-only",
        git_commit,
        "--",
        *_COMMITTED_RUNTIME_ROOTS,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise TrainingOrchestrationError(
            "cannot enumerate the committed PPO runtime tree"
        ) from exc
    if result.returncode != 0:
        raise TrainingOrchestrationError(
            "cannot enumerate the committed PPO runtime tree: "
            + result.stderr.strip()
        )
    paths = tuple(
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    )
    if not paths or len(paths) != len(set(paths)):
        raise TrainingOrchestrationError(
            "committed PPO runtime tree is empty or ambiguous"
        )
    clean = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "diff",
            "--quiet",
            "--no-ext-diff",
            git_commit,
            "--",
            *_COMMITTED_RUNTIME_ROOTS,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if clean.returncode != 0:
        raise TrainingOrchestrationError(
            "current PPO runtime bytes differ from the recorded source commit"
        )
    return paths


def _validate_runtime_identity_document(
    path: Path,
    *,
    project_root: Path,
    expected_git_commit: str,
    cache: _Cache,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...], _Snapshot]:
    payload = _json(path, label="committed runtime identity", cache=cache)
    files = payload.get("files")
    if (
        payload.get("schema") != COMMITTED_RUNTIME_IDENTITY_SCHEMA
        or payload.get("git_commit") != expected_git_commit
        or not isinstance(files, Sequence)
        or isinstance(files, (str, bytes))
        or not files
        or payload.get("file_count") != len(files)
        or not _is_hash(payload.get("content_sha256"))
        or not _is_hash(payload.get("aggregate_sha256"))
    ):
        raise TrainingOrchestrationError(
            "committed runtime identity header is invalid"
        )
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "bytes",
            "sha256",
            "creation_time_utc_ticks",
            "last_write_time_utc_ticks",
        }:
            raise TrainingOrchestrationError(
                f"committed runtime identity file row {index} is malformed"
            )
        name = row.get("path")
        if (
            not isinstance(name, str)
            or not name
            or name.replace("\\", "/") != name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or name in names
        ):
            raise TrainingOrchestrationError(
                "committed runtime file paths are unsafe or duplicate"
            )
        names.add(name)
        source = project_root / name
        _reject_links(source, root=project_root, label=f"committed runtime file {name}")
        captured = _snapshot(
            source, label=f"committed runtime file {name}", cache=cache
        )
        if (
            isinstance(row.get("bytes"), bool)
            or row.get("bytes") != captured.size
            or row.get("sha256") != captured.sha256
            or isinstance(row.get("creation_time_utc_ticks"), bool)
            or not isinstance(row.get("creation_time_utc_ticks"), int)
            or row.get("creation_time_utc_ticks", 0) <= 0
            or isinstance(row.get("last_write_time_utc_ticks"), bool)
            or not isinstance(row.get("last_write_time_utc_ticks"), int)
            or row.get("last_write_time_utc_ticks", 0) <= 0
        ):
            raise TrainingOrchestrationError(
                f"committed runtime file record is stale: {name}"
            )
        normalized.append(
            {
                "path": name,
                "bytes": captured.size,
                "sha256": captured.sha256,
                "creation_time_utc_ticks": row["creation_time_utc_ticks"],
                "last_write_time_utc_ticks": row["last_write_time_utc_ticks"],
            }
        )
    committed_paths = _committed_runtime_paths(str(project_root), expected_git_commit)
    if set(names) != set(committed_paths) or len(names) != len(committed_paths):
        raise TrainingOrchestrationError(
            "committed runtime identity file inventory is incomplete"
        )
    encoded = json.dumps(
        normalized, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    content_encoded = json.dumps(
        [
            {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in normalized
        ],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(content_encoded).hexdigest() != payload.get("content_sha256"):
        raise TrainingOrchestrationError(
            "committed runtime identity content SHA-256 is invalid"
        )
    if hashlib.sha256(encoded).hexdigest() != payload.get("aggregate_sha256"):
        raise TrainingOrchestrationError(
            "committed runtime identity aggregate SHA-256 is invalid"
        )
    return payload, tuple(normalized), _snapshot(
        path, label="committed runtime identity", cache=cache
    )


def _validate_runtime_identity_pair(
    directory: Path,
    *,
    project_root: Path,
    expected_git_commit: str,
    cache: _Cache,
) -> list[dict[str, Any]]:
    before_payload, before_files, before_snapshot = (
        _validate_runtime_identity_document(
            directory / "committed_runtime_identity.before.json",
            project_root=project_root,
            expected_git_commit=expected_git_commit,
            cache=cache,
        )
    )
    after_payload, after_files, after_snapshot = _validate_runtime_identity_document(
        directory / "committed_runtime_identity.after.json",
        project_root=project_root,
        expected_git_commit=expected_git_commit,
        cache=cache,
    )
    if dict(before_payload) != dict(after_payload) or before_files != after_files:
        raise TrainingOrchestrationError(
            "committed runtime identity changed during the managed run"
        )
    return [before_snapshot.record(), after_snapshot.record()]


def _validate_finalized_run(
    run_dir: Path | str,
    *,
    project_root: Path,
    run_kind: str,
    training_stage: str | None,
    entrypoint: str,
    subcommand: str,
    cache: _Cache,
) -> dict[str, Any]:
    directory = _managed_run_dir(
        run_dir, project_root=project_root, run_kind=run_kind, label=f"{run_kind} run"
    )
    final_path = directory / "run_manifest.json"
    started_path = directory / "run_manifest.started.json"
    final = _json(final_path, label=f"{run_kind} finalized manifest", cache=cache)
    started = _json(started_path, label=f"{run_kind} started manifest", cache=cache)
    identity = final.get("identity")
    if (
        final.get("schema") != RUN_MANIFEST_SCHEMA
        or final.get("lifecycle") != "SUCCEEDED"
        or final.get("exit_code") != 0
        or final.get("immutable_run_directory") is not True
        or final.get("run_id") != directory.name
        or _absolute(str(final.get("run_dir", ""))) != directory
        or _absolute(str(final.get("project_root", ""))) != project_root
        or final.get("run_kind") != run_kind
        or final.get("entrypoint") != entrypoint
        or final.get("subcommand") != subcommand
        or started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
        or not isinstance(identity, Mapping)
        or (
            training_stage is not None
            and identity.get("training_stage") != training_stage
        )
        or not _is_hash(identity.get("git_commit"), 40)
        or isinstance(identity.get("seed"), bool)
        or not isinstance(identity.get("seed"), int)
        or isinstance(identity.get("environment_count"), bool)
        or not isinstance(identity.get("environment_count"), int)
        or identity.get("environment_count", 0) < 1
    ):
        raise TrainingOrchestrationError(f"{run_kind} finalized lifecycle is invalid")
    for key, value in started.items():
        if key != "lifecycle" and final.get(key) != value:
            raise TrainingOrchestrationError(
                f"{run_kind} finalized manifest changed started field {key!r}"
            )
    _record(
        final.get("started_manifest"),
        base=directory,
        expected_path="run_manifest.started.json",
        label=f"{run_kind} started manifest",
        cache=cache,
    )
    logs = final.get("logs")
    artifacts = final.get("artifacts")
    if not isinstance(logs, Mapping) or not isinstance(artifacts, Mapping):
        raise TrainingOrchestrationError(f"{run_kind} log/artifact map is missing")
    for name in ("stdout.log", "stderr.log"):
        row = logs.get(name)
        if not isinstance(row, Mapping):
            raise TrainingOrchestrationError(f"{run_kind} omits required {name}")
        _record(
            row,
            base=directory,
            expected_path=name,
            label=f"{run_kind} {name}",
            cache=cache,
        )
    for name, row in artifacts.items():
        if not isinstance(name, str) or not name or ".." in Path(name).parts:
            raise TrainingOrchestrationError(f"{run_kind} artifact key is unsafe")
        _record(
            row,
            base=directory,
            expected_path=name.replace("\\", "/"),
            label=f"{run_kind} artifact {name}",
            cache=cache,
        )
    runtime_names = (
        "committed_runtime_identity.before.json",
        "committed_runtime_identity.after.json",
    )
    if any(name not in artifacts for name in runtime_names):
        raise TrainingOrchestrationError(
            f"{run_kind} omits committed runtime before/after bindings"
        )
    runtime_identities = _validate_runtime_identity_pair(
        directory,
        project_root=project_root,
        expected_git_commit=str(identity.get("git_commit")),
        cache=cache,
    )
    runtime_identity_before_payload = _json(
        directory / "committed_runtime_identity.before.json",
        label=f"{run_kind} committed runtime identity before",
        cache=cache,
    )
    configs = _validate_config_records(
        final.get("configs"),
        project_root=project_root,
        expected_config_sha256=identity.get("config_sha256"),
        cache=cache,
    )
    frozen_manifest_path = project_root / _FROZEN_RELATIVE
    frozen_snapshot = _snapshot(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    frozen_manifest = _json(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    audit_rows = []
    entry_sets = []
    for name in ("frozen_hashes.before.json", "frozen_hashes.after.json"):
        if name not in artifacts:
            raise TrainingOrchestrationError(f"{run_kind} omits {name} artifact binding")
        path = directory / name
        entry_sets.append(
            _validate_frozen_audit(
                path,
                project_root=project_root,
                frozen_manifest_path=frozen_manifest_path,
                frozen_manifest=frozen_manifest,
                frozen_snapshot=frozen_snapshot,
                cache=cache,
            )
        )
        audit_rows.append(_snapshot(path, label=name, cache=cache).record())
    if entry_sets[0] != entry_sets[1]:
        raise TrainingOrchestrationError(f"{run_kind} frozen before/after audits differ")
    started_at = _parse_time(identity.get("timestamp_utc"), label=f"{run_kind} start")
    completed_at = _parse_time(
        final.get("completed_at_utc"), label=f"{run_kind} completion"
    )
    if completed_at <= started_at:
        raise TrainingOrchestrationError(
            f"{run_kind} completion is not later than its reservation"
        )
    return {
        "directory": directory,
        "payload": final,
        "identity": identity,
        "artifacts": artifacts,
        "configs": configs,
        "run_manifest": _snapshot(final_path, label="run manifest", cache=cache).record(),
        "started_at": started_at,
        "completed_at": completed_at,
        "frozen_audits": audit_rows,
        "frozen_manifest_sha256": frozen_snapshot.sha256,
        "frozen_source_head": frozen_manifest.get("source_head"),
        "committed_runtime_identities": runtime_identities,
        "committed_runtime_identity_before_payload": dict(
            runtime_identity_before_payload
        ),
    }


def _required_artifact(
    run: Mapping[str, Any], name: str, *, cache: _Cache, label: str
) -> tuple[Path, Mapping[str, Any]]:
    artifacts = run["artifacts"]
    record = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping):
        raise TrainingOrchestrationError(f"{label} is not bound by the run manifest")
    captured = _record(
        record,
        base=run["directory"],
        expected_path=name,
        label=label,
        cache=cache,
    )
    return captured.path, _json(captured.path, label=label, cache=cache)


def _path_and_hash(
    payload: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    base: Path,
    allowed_root: Path,
    label: str,
    cache: _Cache,
) -> _Snapshot:
    raw = payload.get(path_key)
    if not isinstance(raw, str) or not raw:
        raise TrainingOrchestrationError(f"{label} path is missing")
    path = Path(raw)
    selected = path if path.is_absolute() else base / path
    _reject_links(selected, root=allowed_root, label=label)
    captured = _secure_snapshot(
        selected,
        label=label,
        project_root=allowed_root,
        cache=cache,
    )
    if payload.get(hash_key) != captured.sha256:
        raise TrainingOrchestrationError(f"{label} declared SHA-256 mismatch")
    return captured


def _validate_soft_reset_binding(
    evidence: Any, *, project_root: Path, cache: _Cache
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
        raise TrainingOrchestrationError("single-env training lacks soft-reset acceptance")
    path = evidence.get("path")
    if not isinstance(path, str):
        raise TrainingOrchestrationError("soft-reset acceptance path is missing")
    try:
        from .soft_reset_equivalence import validate_soft_reset_acceptance

        validated = validate_soft_reset_acceptance(path, project_root=project_root)
    except Exception as exc:
        raise TrainingOrchestrationError(
            f"soft-reset acceptance failed strict validation: {exc}"
        ) from exc
    if dict(validated) != dict(evidence):
        raise TrainingOrchestrationError("training_result soft-reset evidence is stale")
    acceptance = _snapshot(path, label="soft-reset acceptance", cache=cache)
    run_manifest = _snapshot(
        validated["run_manifest"], label="soft-reset run manifest", cache=cache
    )
    return {
        "acceptance": acceptance.record(),
        "run_manifest": run_manifest.record(),
        "passed": True,
    }


def _validate_vector_matrix_binding(
    matrix_path: Path,
    *,
    project_root: Path,
    config_sha256: str,
    config_records: Sequence[Mapping[str, Any]],
    git_commit: str,
    run_seed: int,
    expected_num_envs: int,
    expected_seed_rows: Sequence[int],
    cache: _Cache,
) -> dict[str, Any]:
    try:
        from .vector_benchmark_matrix import validate_finalized_vector_benchmark_matrix

        validated = validate_finalized_vector_benchmark_matrix(
            matrix_path,
            expected_project_root=project_root,
            expected_config_sha256=config_sha256,
            expected_frozen_manifest_sha256=_snapshot(
                project_root / _FROZEN_RELATIVE,
                label="frozen FSM manifest",
                cache=cache,
            ).sha256,
            expected_git_commit=git_commit,
            expected_run_seed=run_seed,
            expected_num_envs=expected_num_envs,
            expected_seed_rows=expected_seed_rows,
            expected_config_records=config_records,
        )
    except Exception as exc:
        raise TrainingOrchestrationError(
            f"vector benchmark matrix failed strict validation: {exc}"
        ) from exc
    captured = _snapshot(matrix_path, label="vector benchmark matrix", cache=cache)
    if validated.get("sha256") != captured.sha256:
        raise TrainingOrchestrationError("vector matrix validator returned a stale hash")
    return dict(validated)


def _validate_training_outcome_record(
    result: Mapping[str, Any], *, stage: str, stage_decisions: int,
) -> dict[str, Any]:
    """Recompute diagnostics with the live integrity validator, without a scene.

    JSON writers sort dictionary keys. Normalize only the already-exact key
    sets into the validator's canonical ordering; never fill missing telemetry.
    """
    from .cli import CliError, _validate_training_telemetry

    raw = result.get("training_telemetry")
    if not isinstance(raw, Mapping):
        raise TrainingOrchestrationError("training outcome telemetry is missing")
    telemetry = dict(raw)
    phases = tuple(f"P{index:02d}" for index in range(1, 14))
    families = (
        "phase_task_progress", "body_stability", "contact_motion_quality",
        "control_smoothness", "residual_regularization",
    )
    counts = raw.get("phase_decision_counts")
    rewards = raw.get("reward_family_absolute_sums_by_phase")
    if (not isinstance(counts, Mapping) or set(counts) != set(phases)
            or not isinstance(rewards, Mapping) or set(rewards) != set(phases)
            or any(not isinstance(rewards[phase], Mapping)
                   or set(rewards[phase]) != set(families) for phase in phases)):
        raise TrainingOrchestrationError("training outcome phase/reward key sets are invalid")
    telemetry["phase_decision_counts"] = {phase: counts[phase] for phase in phases}
    telemetry["reward_family_absolute_sums_by_phase"] = {
        phase: {family: rewards[phase][family] for family in families} for phase in phases
    }
    try:
        expected = _validate_training_telemetry(
            telemetry, stage=stage, expected_policy_decisions=stage_decisions,
        )
    except CliError as exc:
        raise TrainingOrchestrationError(f"training outcome integrity is invalid: {exc}") from exc
    if result.get("training_outcome_diagnostics") != expected:
        raise TrainingOrchestrationError("training outcome diagnostics differ from rollout telemetry")
    return expected


def _validate_training_chunk(
    run_dir: Path | str,
    *,
    project_root: Path,
    training_config: _Snapshot,
    interval: int,
    cache: _Cache,
) -> dict[str, Any]:
    run = _validate_finalized_run(
        run_dir,
        project_root=project_root,
        run_kind="train",
        training_stage=None,  # checked against the result immediately below
        entrypoint="wlr50_clean.ppo.cli",
        subcommand="train",
        cache=cache,
    )
    result_path, result = _required_artifact(
        run, "training_result.json", cache=cache, label="training result"
    )
    stage = result.get("stage")
    if result.get("schema") != TRAINING_RESULT_SCHEMA or stage not in STAGES:
        raise TrainingOrchestrationError(f"unsupported training stage: {stage!r}")
    if run["identity"].get("training_stage") != stage:
        raise TrainingOrchestrationError("training lifecycle/result stage mismatch")
    requested = _strict_int(
        result.get("requested_policy_decisions"),
        label=f"{stage} requested decisions",
        minimum=1,
    )
    stage_decisions = _strict_int(
        result.get("stage_policy_decisions"),
        label=f"{stage} actual decisions",
        minimum=1,
    )
    global_decisions = _strict_int(
        result.get("global_policy_decisions"),
        label=f"{stage} global decisions",
        minimum=1,
    )
    iterations = _strict_int(result.get("iterations"), label="iterations", minimum=1)
    num_envs = _strict_int(result.get("num_envs"), label="num_envs", minimum=1)
    rollout = _strict_int(
        result.get("rollout_length"), label="rollout length", minimum=1
    )
    batch_decisions = num_envs * rollout
    rounding_overrun = stage_decisions - requested
    try:
        cadence = derive_stage_cadence(
            stage=stage, num_envs=num_envs,
            **_load_profile(training_config)["cadence_inputs"],
        )
        validate_training_chunk_cadence(
            cadence, requested_policy_decisions=requested, iterations=iterations,
            stage_policy_decisions=stage_decisions,
        )
    except TrainingCadenceError as exc:
        raise TrainingOrchestrationError(f"training chunk cadence is invalid: {exc}") from exc
    if (
        requested > stage_decisions
        or stage_decisions != batch_decisions * iterations
        or iterations != (requested + batch_decisions - 1) // batch_decisions
        or rounding_overrun < 0
        or rounding_overrun >= batch_decisions
        or result.get("ppo_batch_policy_decisions") != batch_decisions
        or result.get("rounding_overrun_policy_decisions") != rounding_overrun
        or result.get("budget_accounting_basis") != "requested_policy_decisions"
        or result.get("deterministic_validation_interval") != cadence["requested_policy_decisions_per_chunk"]
        or interval != cadence["base_validation_interval_policy_decisions"]
        or rollout != cadence["rollout_length"]
        or result.get("training_cadence") != cadence
        or result.get("early_stop_when_promotion_gate_passes") is not True
        or result.get("save_load_round_trip") is not True
        or run["identity"].get("environment_count") != num_envs
    ):
        raise TrainingOrchestrationError("training chunk accounting/control is invalid")
    telemetry = result.get("training_telemetry")
    round_trip = result.get("round_trip_infos")
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("reward_telemetry_complete") is not True
        or telemetry.get("policy_decision_count") != stage_decisions
        or not isinstance(round_trip, Mapping)
        or round_trip.get("global_policy_decisions") != global_decisions
        or round_trip.get("training_cadence") != cadence
    ):
        raise TrainingOrchestrationError("training chunk telemetry is incomplete")
    outcome = _validate_training_outcome_record(
        result, stage=stage, stage_decisions=stage_decisions,
    )
    if (round_trip.get("training_outcome_diagnostics") != outcome
            or round_trip.get("training_telemetry") != telemetry):
        raise TrainingOrchestrationError("training round-trip outcome telemetry is inconsistent")
    _finite_positive(result.get("wall_time_s"), label="training wall time")

    resume = _path_and_hash(
        result,
        path_key="resume_checkpoint",
        hash_key="resume_checkpoint_sha256",
        base=run["directory"],
        allowed_root=project_root,
        label="training resume checkpoint",
        cache=cache,
    )
    history_raw = result.get("immutable_history_checkpoint")
    if not isinstance(history_raw, str):
        raise TrainingOrchestrationError("immutable history checkpoint path is missing")
    history_path = Path(history_raw)
    history_selected = (
        history_path if history_path.is_absolute() else run["directory"] / history_path
    )
    _reject_links(
        history_selected, root=project_root, label="immutable history checkpoint"
    )
    history = _snapshot(
        history_selected, label="immutable history checkpoint", cache=cache
    )
    if result.get("checkpoint_sha256") != history.sha256:
        raise TrainingOrchestrationError("immutable history checkpoint hash mismatch")
    last_raw = result.get("checkpoint_last")
    if not isinstance(last_raw, str):
        raise TrainingOrchestrationError("checkpoint_last path is missing")
    checkpoint_last_path = _absolute(last_raw)
    history_manifest_path = history.path.with_name(history.path.stem + "_manifest.json")
    history_manifest = _json(
        history_manifest_path, label="immutable checkpoint manifest", cache=cache
    )
    history_manifest_snapshot = _snapshot(
        history_manifest_path, label="immutable checkpoint manifest", cache=cache
    )
    _reject_links(
        history_manifest_path,
        root=project_root,
        label="immutable checkpoint manifest",
    )
    if (
        history_manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or history_manifest.get("stage") != stage
        or history_manifest.get("global_policy_decisions") != global_decisions
        or _absolute(str(history_manifest.get("checkpoint_path", ""))) != history.path
        or history_manifest.get("checkpoint_sha256") != history.sha256
        or history_manifest.get("stage_policy_decisions") != stage_decisions
        or history_manifest.get("training_cadence") != cadence
        or history_manifest.get("training_outcome_diagnostics") != outcome
        or history_manifest.get("training_telemetry") != telemetry
        or _absolute(str(history_manifest.get("resume_checkpoint", ""))) != resume.path
        or history_manifest.get("resume_checkpoint_sha256") != resume.sha256
        or history_manifest.get("source_git_commit")
        != run["identity"].get("git_commit")
        or history_manifest.get("committed_runtime_content_sha256")
        != run["committed_runtime_identity_before_payload"].get("content_sha256")
        or _absolute(
            str(history_manifest.get("creation_runtime_identity_path", ""))
        )
        != run["directory"] / "committed_runtime_identity.before.json"
        or history_manifest.get("creation_runtime_identity_sha256")
        != run["committed_runtime_identities"][0]["sha256"]
    ):
        raise TrainingOrchestrationError("immutable checkpoint manifest is inconsistent")
    resume_global = _strict_int(
        history_manifest.get("resume_global_policy_decisions"),
        label="resume global decisions",
    )
    if global_decisions != resume_global + stage_decisions:
        raise TrainingOrchestrationError("training global step equation is invalid")
    files = history_manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TrainingOrchestrationError("checkpoint manifest input inventory is missing")
    for raw_path, raw_hash in files.items():
        if not isinstance(raw_path, str) or not _is_hash(raw_hash):
            raise TrainingOrchestrationError("checkpoint input record is malformed")
        _reject_links(raw_path, root=project_root, label="checkpoint input")
        source = _snapshot(raw_path, label="checkpoint input", cache=cache)
        if source.sha256 != raw_hash:
            raise TrainingOrchestrationError("checkpoint input hash mismatch")

    training_relative = training_config.path.relative_to(project_root).as_posix()
    config_by_name = {row["path"]: row for row in run["configs"]}
    if config_by_name.get(training_relative) != {
        "path": training_relative,
        "bytes": training_config.size,
        "sha256": training_config.sha256,
    }:
        raise TrainingOrchestrationError("training run does not bind the selected profile")
    return {
        "stage": stage,
        "training_cadence": cadence,
        "training_outcome_diagnostics": outcome,
        "requested_policy_decisions": requested,
        "stage_policy_decisions": stage_decisions,
        "global_policy_decisions": global_decisions,
        "resume_global_policy_decisions": resume_global,
        "iterations": iterations,
        "num_envs": num_envs,
        "rollout_length": rollout,
        "ppo_batch_policy_decisions": batch_decisions,
        "rounding_overrun_policy_decisions": rounding_overrun,
        "run_directory": str(run["directory"]),
        "run_manifest": run["run_manifest"],
        "training_result": _snapshot(result_path, label="training result", cache=cache).record(),
        "resume_checkpoint": resume.record(),
        "immutable_history_checkpoint": history.record(),
        "checkpoint_manifest": history_manifest_snapshot.record(),
        "checkpoint_manifest_payload": dict(history_manifest),
        # checkpoint_last is deliberately not captured here: it is mutable across
        # resume chunks.  The chain is anchored by immutable history checkpoints,
        # and the shared last checkpoint is captured once for the terminal chunk.
        "checkpoint_last_path": str(checkpoint_last_path),
        "identity": dict(run["identity"]),
        "configs": list(run["configs"]),
        "frozen_hash_audits": run["frozen_audits"],
        "committed_runtime_identities": run["committed_runtime_identities"],
        "committed_runtime_identity_before_payload": dict(
            run["committed_runtime_identity_before_payload"]
        ),
        "completed_at": run["completed_at"],
        "started_at": run["started_at"],
        "soft_reset_acceptance_raw": result.get("soft_reset_acceptance"),
        "vector_matrix_raw": result.get("vector_benchmark_matrix"),
        "vector_matrix_path": result.get("vector_benchmark_matrix_path"),
        "vector_matrix_sha256": result.get("vector_benchmark_matrix_sha256"),
    }


def _checkpoint_infos_from_snapshot(
    checkpoint: _Snapshot,
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Safely decode one captured Torch checkpoint and bind its embedded infos."""

    try:
        import torch  # type: ignore

        payload = torch.load(
            io.BytesIO(checkpoint.data), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise TrainingOrchestrationError(
            f"{label} is not a safe loadable Torch checkpoint"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("infos"), Mapping):
        raise TrainingOrchestrationError(f"{label} omits embedded checkpoint infos")
    infos = dict(payload["infos"])
    manifest_infos = {
        key: value
        for key, value in manifest.items()
        if key not in {"checkpoint_path", "checkpoint_sha256"}
    }
    if infos != manifest_infos:
        raise TrainingOrchestrationError(
            f"{label} embedded infos differ from its sidecar"
        )
    return payload, infos


def _validate_checkpoint_creation_binding(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
    expected_git_commit: str,
    expected_content_sha256: str,
    cache: _Cache,
) -> dict[str, Any]:
    raw_path = manifest.get("creation_runtime_identity_path")
    raw_sha = manifest.get("creation_runtime_identity_sha256")
    if not isinstance(raw_path, str) or not _is_hash(raw_sha):
        raise TrainingOrchestrationError(
            "checkpoint omits its creation committed-runtime identity binding"
        )
    identity_path = _absolute(raw_path)
    if identity_path.name != "committed_runtime_identity.before.json":
        raise TrainingOrchestrationError(
            "checkpoint creation runtime identity has the wrong filename"
        )
    try:
        creation_relative = identity_path.parent.relative_to(
            project_root / _RUNS_RELATIVE
        )
    except ValueError as exc:
        raise TrainingOrchestrationError(
            "checkpoint creation runtime identity escapes the managed runs root"
        ) from exc
    if (
        manifest.get("stage") == "initial_zero_residual"
        and len(creation_relative.parts) == 2
        and creation_relative.parts[0] == INITIAL_CHECKPOINT_RUN_KIND
    ):
        creation_run_kind = INITIAL_CHECKPOINT_RUN_KIND
        creation_training_stage = "initialize-zero-residual"
        creation_subcommand = "initialize-zero-residual"
    else:
        creation_run_kind = "train"
        creation_training_stage = None
        creation_subcommand = "train"
    creation_run = _validate_finalized_run(
        identity_path.parent,
        project_root=project_root,
        run_kind=creation_run_kind,
        training_stage=creation_training_stage,
        entrypoint="wlr50_clean.ppo.cli",
        subcommand=creation_subcommand,
        cache=cache,
    )
    before_record = creation_run["committed_runtime_identities"][0]
    before_payload = creation_run["committed_runtime_identity_before_payload"]
    if (
        identity_path != creation_run["directory"] / identity_path.name
        or raw_sha != before_record["sha256"]
        or manifest.get("source_git_commit") != expected_git_commit
        or before_payload.get("git_commit") != expected_git_commit
        or manifest.get("committed_runtime_content_sha256")
        != expected_content_sha256
        or before_payload.get("content_sha256") != expected_content_sha256
    ):
        raise TrainingOrchestrationError(
            "checkpoint creation runtime identity does not bind its source bytes"
        )
    return {
        "path": before_record["path"],
        "bytes": before_record["bytes"],
        "sha256": before_record["sha256"],
        "content_sha256": expected_content_sha256,
        "source_git_commit": expected_git_commit,
        "creation_run_kind": creation_run_kind,
        "creation_run_directory": str(creation_run["directory"]),
        "creation_run_started_at_utc": creation_run["identity"].get(
            "timestamp_utc"
        ),
        "creation_run_completed_at_utc": creation_run["payload"].get(
            "completed_at_utc"
        ),
        "creation_run_manifest": creation_run["run_manifest"],
    }


def _validate_initial_checkpoint(
    first_chunk: Mapping[str, Any],
    *,
    project_root: Path,
    expected_git_commit: str,
    expected_seed: int,
    cache: _Cache,
) -> dict[str, Any]:
    checkpoint_path = (
        project_root
        / "outputs"
        / "ppo_phase_v1"
        / "checkpoints"
        / "checkpoint_initial_zero_residual.pt"
    )
    resume = first_chunk["resume_checkpoint"]
    checkpoint = _secure_snapshot(
        checkpoint_path,
        label="canonical initial checkpoint",
        project_root=project_root,
        cache=cache,
    )
    if (
        _absolute(str(resume.get("path", ""))) != checkpoint_path
        or resume.get("sha256") != checkpoint.sha256
        or first_chunk["resume_global_policy_decisions"] != 0
    ):
        raise TrainingOrchestrationError(
            "first training chunk did not resume from the canonical initial checkpoint"
        )
    manifest_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_manifest.json"
    )
    manifest_snapshot = _secure_snapshot(
        manifest_path,
        label="canonical initial checkpoint manifest",
        project_root=project_root,
        cache=cache,
    )
    manifest = _json(
        manifest_path, label="canonical initial checkpoint manifest", cache=cache
    )
    expected_content = str(
        first_chunk["committed_runtime_identity_before_payload"].get(
            "content_sha256", ""
        )
    )
    from .rl_library_wrapper import CHECKPOINT_RUNTIME_CONTRACT_FIELDS

    first_history_manifest = first_chunk.get("checkpoint_manifest_payload")
    if (
        manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("stage") != "initial_zero_residual"
        or manifest.get("global_policy_decisions") != 0
        or manifest.get("training_seed") != expected_seed
        or manifest.get("zero_mean_actor_output_layer_verified") is not True
        or _absolute(str(manifest.get("checkpoint_path", ""))) != checkpoint.path
        or manifest.get("checkpoint_sha256") != checkpoint.sha256
        or manifest.get("source_git_commit") != expected_git_commit
        or manifest.get("committed_runtime_content_sha256") != expected_content
        or not isinstance(first_history_manifest, Mapping)
        or any(
            field not in manifest
            or field not in first_history_manifest
            or manifest[field] != first_history_manifest[field]
            for field in CHECKPOINT_RUNTIME_CONTRACT_FIELDS
        )
    ):
        raise TrainingOrchestrationError(
            "canonical initial checkpoint manifest is inconsistent"
        )
    payload, _ = _checkpoint_infos_from_snapshot(
        checkpoint, manifest, label="canonical initial checkpoint"
    )
    actor = payload.get("actor_state_dict")
    if not isinstance(actor, Mapping):
        raise TrainingOrchestrationError(
            "canonical initial checkpoint omits actor_state_dict"
        )
    try:
        import torch  # type: ignore

        candidates = []
        for name, weight in actor.items():
            if (
                isinstance(name, str)
                and name.endswith(".weight")
                and torch.is_tensor(weight)
                and weight.ndim == 2
                and int(weight.shape[0]) == int(manifest.get("residual_dimension", -1))
            ):
                bias_name = name[: -len("weight")] + "bias"
                bias = actor.get(bias_name)
                if torch.is_tensor(bias) and bias.ndim == 1:
                    candidates.append((name, weight, bias))
    except Exception as exc:
        raise TrainingOrchestrationError(
            "canonical initial actor state is malformed"
        ) from exc
    if len(candidates) != 1:
        raise TrainingOrchestrationError(
            "canonical initial actor output layer is ambiguous"
        )
    _, weight, bias = candidates[0]
    if (
        not bool(torch.isfinite(weight).all().item())
        or not bool(torch.isfinite(bias).all().item())
        or int(torch.count_nonzero(weight).item()) != 0
        or int(torch.count_nonzero(bias).item()) != 0
    ):
        raise TrainingOrchestrationError(
            "canonical initial actor output layer is not exact zero"
        )
    creation = _validate_checkpoint_creation_binding(
        manifest,
        project_root=project_root,
        expected_git_commit=expected_git_commit,
        expected_content_sha256=expected_content,
        cache=cache,
    )
    return {
        "path": str(checkpoint.path),
        "sha256": checkpoint.sha256,
        "manifest_path": str(manifest_snapshot.path),
        "manifest_sha256": manifest_snapshot.sha256,
        "creation_runtime_identity": creation,
        "zero_mean_actor_output_layer_verified": True,
    }


def _validate_initial_checkpoint_publication(
    publication_run_dir: Path | str,
    initial_checkpoint: Mapping[str, Any],
    *,
    project_root: Path,
    expected_seed: int,
    expected_git_commit: str,
    expected_config_sha256: str,
    expected_configs: Sequence[Mapping[str, Any]],
    first_training_started_at: datetime,
    cache: _Cache,
) -> dict[str, Any]:
    """Bind the canonical initial pair to its exact finalized publisher run."""

    run = _validate_finalized_run(
        publication_run_dir,
        project_root=project_root,
        run_kind=INITIAL_CHECKPOINT_PUBLICATION_RUN_KIND,
        training_stage="initial-checkpoint-publication",
        entrypoint="wlr50_clean.ppo.cli",
        subcommand="publish-initial-zero-residual",
        cache=cache,
    )
    identity = run["identity"]
    if (
        identity.get("seed") != expected_seed
        or identity.get("environment_count") != 1
        or identity.get("git_commit") != expected_git_commit
        or identity.get("config_sha256") != expected_config_sha256
        or list(run["configs"]) != list(expected_configs)
        or run["completed_at"] >= first_training_started_at
    ):
        raise TrainingOrchestrationError(
            "initial checkpoint publication run differs from or overlaps training"
        )
    result_path, result = _required_artifact(
        run,
        "initial_checkpoint_publication.json",
        cache=cache,
        label="initial checkpoint publication result",
    )
    canonical_path = _absolute(str(initial_checkpoint.get("path", "")))
    canonical_manifest_path = _absolute(
        str(initial_checkpoint.get("manifest_path", ""))
    )
    canonical = _secure_snapshot(
        canonical_path,
        label="canonical initial checkpoint",
        project_root=project_root,
        cache=cache,
    )
    canonical_manifest = _secure_snapshot(
        canonical_manifest_path,
        label="canonical initial checkpoint manifest",
        project_root=project_root,
        cache=cache,
    )
    creation = initial_checkpoint.get("creation_runtime_identity")
    if not isinstance(creation, Mapping):
        raise TrainingOrchestrationError(
            "initial checkpoint creation runtime evidence is missing"
        )
    creation_run_dir = _absolute(str(creation.get("creation_run_directory", "")))
    creation_kind = str(creation.get("creation_run_kind", ""))
    source_checkpoint = _path_and_hash(
        result,
        path_key="source_checkpoint",
        hash_key="source_checkpoint_sha256",
        base=run["directory"],
        allowed_root=project_root,
        label="initial publication source checkpoint",
        cache=cache,
    )
    source_manifest = _path_and_hash(
        result,
        path_key="source_checkpoint_manifest",
        hash_key="source_checkpoint_manifest_sha256",
        base=run["directory"],
        allowed_root=project_root,
        label="initial publication source manifest",
        cache=cache,
    )
    source_manifest_payload = _json(
        source_manifest.path,
        label="initial publication source manifest",
        cache=cache,
    )
    canonical_manifest_payload = _json(
        canonical_manifest.path,
        label="canonical initial checkpoint manifest",
        cache=cache,
    )
    source_core = {
        key: value
        for key, value in source_manifest_payload.items()
        if key not in {"checkpoint_path", "checkpoint_sha256"}
    }
    canonical_core = {
        key: value
        for key, value in canonical_manifest_payload.items()
        if key not in {"checkpoint_path", "checkpoint_sha256"}
    }
    reused = result.get("reused_existing")
    creator_source_checkpoint = (
        creation_run_dir / "checkpoint_initial_zero_residual.pt"
    )
    creator_source_manifest = (
        creation_run_dir / "checkpoint_initial_zero_residual_manifest.json"
    )
    source_is_canonical = (
        reused is True
        and source_checkpoint.path == canonical.path
        and source_manifest.path == canonical_manifest.path
    )
    source_is_initializer = (
        creation_kind == INITIAL_CHECKPOINT_RUN_KIND
        and source_checkpoint.path == creator_source_checkpoint
        and source_manifest.path == creator_source_manifest
    )
    if type(reused) is not bool or not (source_is_canonical or source_is_initializer):
        raise TrainingOrchestrationError(
            "initial checkpoint publication reuse/source mode is invalid"
        )
    if (
        result.get("schema")
        != "wlr50_clean.initial_zero_residual_checkpoint_publication.v1"
        or result.get("no_existing_artifact_overwritten") is not True
        or result.get("source_initializer_finalized_success") is not True
        or result.get("embedded_infos_match_manifest") is not True
        or result.get("zero_mean_actor_output_layer_verified") is not True
        or _absolute(str(result.get("checkpoint", ""))) != canonical.path
        or result.get("checkpoint_sha256") != canonical.sha256
        or result.get("checkpoint_sha256") != initial_checkpoint.get("sha256")
        or _absolute(str(result.get("checkpoint_manifest", "")))
        != canonical_manifest.path
        or result.get("checkpoint_manifest_sha256") != canonical_manifest.sha256
        or result.get("checkpoint_manifest_sha256")
        != initial_checkpoint.get("manifest_sha256")
        or result.get("creation_run_kind") != creation_kind
        or _absolute(str(result.get("creation_run_directory", "")))
        != creation_run_dir
        or source_checkpoint.sha256 != canonical.sha256
        or source_manifest_payload.get("checkpoint_sha256") != canonical.sha256
        or source_core != canonical_core
    ):
        raise TrainingOrchestrationError(
            "initial checkpoint publication does not bind creator source to canonical bytes"
        )
    creation_completed = _parse_time(
        creation.get("creation_run_completed_at_utc"),
        label="initial checkpoint creation completion",
    )
    if run["started_at"] <= creation_completed:
        raise TrainingOrchestrationError(
            "initial checkpoint publisher did not start after creator finalization"
        )
    return {
        "run_directory": str(run["directory"]),
        "run_manifest": run["run_manifest"],
        "publication_result": _snapshot(
            result_path,
            label="initial checkpoint publication result",
            cache=cache,
        ).record(),
        "reused_existing": reused,
        "source_checkpoint": source_checkpoint.record(),
        "source_checkpoint_manifest": source_manifest.record(),
        "canonical_checkpoint": canonical.record(),
        "canonical_checkpoint_manifest": canonical_manifest.record(),
        "creation_run_kind": creation_kind,
        "creation_run_directory": str(creation_run_dir),
        "frozen_hash_audits": run["frozen_audits"],
        "committed_runtime_identities": run["committed_runtime_identities"],
    }


def _validate_smoke_checkpoint(
    first_chunk: Mapping[str, Any],
    *,
    project_root: Path,
    cache: _Cache,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if first_chunk.get("stage") != "smoke":
        raise TrainingOrchestrationError("the first training chunk is not smoke")
    history = first_chunk["immutable_history_checkpoint"]
    history_manifest = first_chunk["checkpoint_manifest"]
    canonical_path = (
        project_root
        / "outputs"
        / "ppo_phase_v1"
        / "checkpoints"
        / "checkpoint_smoke.pt"
    )
    canonical = _secure_snapshot(
        canonical_path,
        label="canonical smoke checkpoint",
        project_root=project_root,
        cache=cache,
    )
    canonical_manifest_path = canonical_path.with_name(
        canonical_path.stem + "_manifest.json"
    )
    canonical_manifest_snapshot = _secure_snapshot(
        canonical_manifest_path,
        label="canonical smoke checkpoint manifest",
        project_root=project_root,
        cache=cache,
    )
    canonical_manifest = _json(
        canonical_manifest_path,
        label="canonical smoke checkpoint manifest",
        cache=cache,
    )
    history_payload = first_chunk["checkpoint_manifest_payload"]
    history_core = {
        key: value
        for key, value in history_payload.items()
        if key not in {"checkpoint_path", "checkpoint_sha256"}
    }
    canonical_core = {
        key: value
        for key, value in canonical_manifest.items()
        if key not in {"checkpoint_path", "checkpoint_sha256"}
    }
    if (
        canonical.sha256 != history.get("sha256")
        or canonical_manifest.get("checkpoint_sha256") != canonical.sha256
        or _absolute(str(canonical_manifest.get("checkpoint_path", "")))
        != canonical.path
        or canonical_core != history_core
    ):
        raise TrainingOrchestrationError(
            "canonical smoke checkpoint differs from the first smoke history"
        )
    _checkpoint_infos_from_snapshot(
        canonical, canonical_manifest, label="canonical smoke checkpoint"
    )
    return (
        {
            "path": history["path"],
            "sha256": history["sha256"],
            "manifest_path": history_manifest["path"],
            "manifest_sha256": history_manifest["sha256"],
        },
        {
            "path": str(canonical.path),
            "sha256": canonical.sha256,
            "manifest_path": str(canonical_manifest_snapshot.path),
            "manifest_sha256": canonical_manifest_snapshot.sha256,
        },
    )


def _validate_screening(
    run_dir: Path | str,
    *,
    project_root: Path,
    chunk: Mapping[str, Any],
    validation_seeds: Sequence[int],
    cache: _Cache,
) -> dict[str, Any]:
    run = _validate_finalized_run(
        run_dir,
        project_root=project_root,
        run_kind="validation-checkpoint-screening",
        training_stage="checkpoint-screening-fresh-process",
        entrypoint="wlr50_clean.ppo.cli",
        subcommand="evaluate",
        cache=cache,
    )
    identity = run["identity"]
    seed = _strict_int(identity.get("seed"), label="screening seed")
    if (
        seed not in validation_seeds
        or identity.get("environment_count") != 1
        or identity.get("git_commit") != chunk["identity"].get("git_commit")
        or identity.get("config_sha256")
        != chunk["identity"].get("config_sha256")
        or list(run["configs"]) != chunk["configs"]
    ):
        raise TrainingOrchestrationError("screening identity is not one validation seed")
    result_path, result = _required_artifact(
        run,
        "checkpoint_evaluation.json",
        cache=cache,
        label="screening checkpoint evaluation",
    )
    episodes = result.get("episodes")
    if (
        result.get("schema") != SCREENING_RESULT_SCHEMA
        or result.get("deterministic_mean_policy") is not True
        or result.get("fresh_process_single_episode") is not True
        or result.get("vec_env_step_called") is not False
        or result.get("episode_count") != 1
        or result.get("success_count") not in (0, 1)
        or not isinstance(episodes, Sequence)
        or isinstance(episodes, (str, bytes))
        or len(episodes) != 1
    ):
        raise TrainingOrchestrationError("screening is not fresh deterministic one-seed evidence")
    checkpoint = _path_and_hash(
        result,
        path_key="checkpoint",
        hash_key="checkpoint_sha256",
        base=run["directory"],
        allowed_root=project_root,
        label="screening checkpoint",
        cache=cache,
    )
    checkpoint_manifest = _path_and_hash(
        result,
        path_key="checkpoint_manifest",
        hash_key="checkpoint_manifest_sha256",
        base=run["directory"],
        allowed_root=project_root,
        label="screening checkpoint manifest",
        cache=cache,
    )
    if (
        checkpoint.sha256 != chunk["immutable_history_checkpoint"]["sha256"]
        or checkpoint_manifest.sha256 != chunk["checkpoint_manifest"]["sha256"]
        or checkpoint.path
        != _absolute(chunk["immutable_history_checkpoint"]["path"])
        or checkpoint_manifest.path != _absolute(chunk["checkpoint_manifest"]["path"])
    ):
        raise TrainingOrchestrationError("screening is bound to a different training chunk")
    provenance = result.get("checkpoint_provenance")
    infos = result.get("checkpoint_infos")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("checkpoint_sha256") != checkpoint.sha256
        or provenance.get("manifest_sha256") != checkpoint_manifest.sha256
        or provenance.get("global_policy_decisions") != chunk["global_policy_decisions"]
        or not isinstance(infos, Mapping)
        or infos.get("global_policy_decisions") != chunk["global_policy_decisions"]
    ):
        raise TrainingOrchestrationError("screening checkpoint provenance/global step is stale")
    episode = episodes[0]
    if not isinstance(episode, Mapping):
        raise TrainingOrchestrationError("screening episode summary is malformed")
    boolean_fields = (
        "task_success",
        "body_collision",
        "wheel_only_climb",
        "safety_abort",
        "under_maximum_duration",
    )
    if (
        episode.get("seed") != seed
        or any(type(episode.get(name)) is not bool for name in boolean_fields)
        or not isinstance(episode.get("termination_reason"), str)
        or not episode.get("termination_reason")
        or _strict_int(episode.get("decision_count"), label="screen decisions", minimum=1) <= 0
        or _strict_int(episode.get("physics_tick"), label="screen physics tick", minimum=1) <= 0
        or episode.get("recording_runtime_access_count") != 0
        or episode.get("in_episode_root_write_count") != 0
    ):
        raise TrainingOrchestrationError("screening episode evidence is incomplete")
    _finite_positive(episode.get("duration_s"), label="screening duration")
    computed_pass = bool(
        episode["task_success"]
        and not episode["body_collision"]
        and not episode["wheel_only_climb"]
        and not episode["safety_abort"]
        and episode["under_maximum_duration"]
    )
    if result.get("passed") is not computed_pass:
        raise TrainingOrchestrationError("screening pass bit differs from physical evidence")
    if result.get("success_count") != int(computed_pass):
        raise TrainingOrchestrationError(
            "screening success_count differs from physical evidence"
        )
    episode_dir_raw = episode.get("canonical_episode_dir")
    if not isinstance(episode_dir_raw, str):
        raise TrainingOrchestrationError("screening canonical episode path is missing")
    episode_dir = _absolute(episode_dir_raw)
    try:
        episode_dir.relative_to(run["directory"])
    except ValueError as exc:
        raise TrainingOrchestrationError("screening canonical episode escapes its run") from exc
    _reject_links(episode_dir, root=run["directory"], label="screening episode")
    if not episode_dir.is_dir():
        raise TrainingOrchestrationError("screening canonical episode is missing")
    summary_path = episode_dir / "episode_summary.json"
    if dict(_json(summary_path, label="screening episode summary", cache=cache)) != dict(episode):
        raise TrainingOrchestrationError("screening episode summary differs from result")
    trial_path = episode_dir / "trial_manifest.json"
    trial = _json(trial_path, label="screening trial manifest", cache=cache)
    if (
        trial.get("schema") != "wlr50_clean.ppo_live_trial_manifest.v1"
        or trial.get("seed") != seed
        or trial.get("decision_count") != episode.get("decision_count")
        or trial.get("result") != episode.get("termination_reason")
    ):
        raise TrainingOrchestrationError("screening trial manifest is inconsistent")
    trace_path = episode_dir / "policy_trace.jsonl"
    trace = _snapshot(trace_path, label="screening policy trace", cache=cache)
    if not trace.data:
        raise TrainingOrchestrationError("screening policy trace is empty")
    for path in (summary_path, trial_path, trace_path):
        relative = path.relative_to(run["directory"]).as_posix()
        if relative not in run["artifacts"]:
            raise TrainingOrchestrationError(
                f"screening run does not bind {relative}"
            )
    _finite_positive(result.get("wall_time_s"), label="screening wall time")
    return {
        "run_directory": str(run["directory"]),
        "run_manifest": run["run_manifest"],
        "checkpoint_evaluation": _snapshot(
            result_path, label="screening result", cache=cache
        ).record(),
        "checkpoint": checkpoint.record(),
        "checkpoint_manifest": checkpoint_manifest.record(),
        "seed": seed,
        "global_policy_decisions": chunk["global_policy_decisions"],
        "physical_passed": computed_pass,
        "complete_evidence": True,
        "episode_summary": _snapshot(
            summary_path, label="episode summary", cache=cache
        ).record(),
        "trial_manifest": _snapshot(
            trial_path, label="trial manifest", cache=cache
        ).record(),
        "policy_trace": trace.record(),
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
        "frozen_hash_audits": run["frozen_audits"],
        "committed_runtime_identities": run["committed_runtime_identities"],
    }


def _promotion_decision_history_step(
    path: Path | str, *, project_root: Path
) -> int | None:
    """Admit managed runs or one canonical, delivery-contained history slot.

    The final five-role metric export owns ``outputs/ppo_phase_v1/metrics``.
    Keeping immutable cadence inputs in its sibling ``validation_history``
    includes them in the delivery checksum without overlapping that output tree.
    The returned step is subsequently bound to the actual training checkpoint.
    """

    selected = _absolute(path)
    managed_runs_root = project_root / _RUNS_RELATIVE
    try:
        relative = selected.relative_to(managed_runs_root)
    except ValueError:
        history_root = project_root / _VALIDATION_HISTORY_RELATIVE
        try:
            relative = selected.relative_to(history_root)
        except ValueError as exc:
            raise TrainingOrchestrationError(
                f"promotion decision must be inside {managed_runs_root} or "
                f"{history_root / 'step_<global>' / 'promotion_decision.json'}"
            ) from exc
        match = (
            re.fullmatch(r"step_([1-9][0-9]*)", relative.parts[0])
            if relative.parts
            else None
        )
        if (
            len(relative.parts) != 2
            or relative.parts[1] != "promotion_decision.json"
            or match is None
        ):
            raise TrainingOrchestrationError(
                "promotion decision has a noncanonical validation_history path; "
                "expected step_<positive_global>/promotion_decision.json"
            )
        return int(match.group(1))
    if not relative.parts:
        raise TrainingOrchestrationError("promotion decision path is malformed")
    return None


def _validate_promotion_decision(
    path: Path | str,
    *,
    chunks: Sequence[Mapping[str, Any]],
    project_root: Path,
    cache: _Cache,
) -> dict[str, Any]:
    from .checkpoint_promotion import REQUIRED_PROMOTION_GATES
    from .paired_aggregate_binding import (
        PairedAggregateBindingError,
        capture_validation_aggregate,
    )

    selected = _absolute(path)
    history_step = _promotion_decision_history_step(
        selected, project_root=project_root
    )
    captured = _secure_snapshot(
        selected,
        label="promotion decision",
        project_root=project_root,
        cache=cache,
    )
    decision = _json(captured.path, label="promotion decision", cache=cache)
    promotion = decision.get("promotion")
    checks = promotion.get("checks") if isinstance(promotion, Mapping) else None
    ordered = decision.get("checks_in_evaluation_order")
    if (
        decision.get("schema") != PROMOTION_DECISION_SCHEMA
        or decision.get("baseline_checkpoint") != "pure_fsm"
        or decision.get("paired_seeds") != list(VALIDATION_SEEDS)
        or decision.get("paired_episode_count") != 5
        or decision.get("minimum_paired_seeds") != 5
        or decision.get("frozen_hashes_unchanged") is not True
        or not isinstance(promotion, Mapping)
        or type(promotion.get("promoted")) is not bool
        or not isinstance(checks, Mapping)
        or set(checks) != set(REQUIRED_PROMOTION_GATES)
        or any(type(checks[name]) is not bool for name in REQUIRED_PROMOTION_GATES)
        or not isinstance(ordered, Sequence)
        or isinstance(ordered, (str, bytes))
        or len(ordered) != len(REQUIRED_PROMOTION_GATES)
    ):
        raise TrainingOrchestrationError("promotion decision shape/gate set is invalid")
    ordered_names: list[str] = []
    ordered_values: list[bool] = []
    for row in ordered:
        if not isinstance(row, Mapping) or type(row.get("passed")) is not bool:
            raise TrainingOrchestrationError("ordered promotion gate row is invalid")
        ordered_names.append(str(row.get("gate", "")))
        ordered_values.append(row["passed"])
    if tuple(ordered_names) != REQUIRED_PROMOTION_GATES or any(
        checks[name] is not value
        for name, value in zip(ordered_names, ordered_values, strict=True)
    ):
        raise TrainingOrchestrationError("promotion gates are reordered or inconsistent")
    promoted = promotion["promoted"]
    first_failed = next(
        (name for name in REQUIRED_PROMOTION_GATES if checks[name] is False), None
    )
    if (
        promoted is not (first_failed is None)
        or promotion.get("first_failed_gate") != first_failed
        or decision.get("first_failed_gate") != first_failed
    ):
        raise TrainingOrchestrationError("promotion authorization disagrees with gate evidence")
    improvement = promotion.get("global_stability_improvement_fraction")
    improved_count = promotion.get("improved_priority_phase_count")
    if (
        isinstance(improvement, bool)
        or not isinstance(improvement, (int, float))
        or not math.isfinite(float(improvement))
        or isinstance(improved_count, bool)
        or not isinstance(improved_count, int)
    ):
        raise TrainingOrchestrationError("promotion improvement evidence is malformed")
    if promoted and (float(improvement) < 0.05 or improved_count < 4):
        raise TrainingOrchestrationError("promotion thresholds were not truly passed")
    candidate_path = _absolute(str(decision.get("candidate_checkpoint_path", "")))
    candidate_hash = decision.get("candidate_checkpoint_sha256")
    if not _is_hash(candidate_hash):
        raise TrainingOrchestrationError("promotion candidate hash is invalid")
    matches = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["immutable_history_checkpoint"]["path"] == str(candidate_path)
        and chunk["immutable_history_checkpoint"]["sha256"] == candidate_hash
    ]
    if len(matches) != 1:
        raise TrainingOrchestrationError(
            "promotion decision does not bind exactly one training checkpoint"
        )
    bound_index = matches[0]
    bound_chunk = chunks[bound_index]
    if (
        history_step is not None
        and history_step != bound_chunk["global_policy_decisions"]
    ):
        raise TrainingOrchestrationError(
            "validation_history step does not match the bound training checkpoint global step"
        )
    baseline_raw = decision.get("baseline_evaluation_aggregate")
    candidate_raw = decision.get("candidate_validation_aggregate")
    if not isinstance(baseline_raw, Mapping) or not isinstance(candidate_raw, Mapping):
        raise TrainingOrchestrationError(
            "promotion decision omits exact baseline/candidate aggregate bindings"
        )
    candidate_manifest_record = bound_chunk.get("checkpoint_manifest")
    if not isinstance(candidate_manifest_record, Mapping):
        raise TrainingOrchestrationError("bound training chunk omits its checkpoint manifest")
    try:
        baseline_capture = capture_validation_aggregate(
            str(baseline_raw.get("path", "")),
            role="baseline",
            project_root=project_root,
        )
        candidate_capture = capture_validation_aggregate(
            str(candidate_raw.get("path", "")),
            role="candidate",
            expected_checkpoint_path=candidate_path,
            expected_checkpoint_manifest_path=str(
                candidate_manifest_record.get("path", "")
            ),
            project_root=project_root,
        )
    except PairedAggregateBindingError as exc:
        raise TrainingOrchestrationError(
            f"promotion aggregate provenance is invalid: {exc}"
        ) from exc
    baseline_binding = baseline_capture.as_record()
    candidate_binding = candidate_capture.as_record()
    if dict(baseline_raw) != baseline_binding or dict(candidate_raw) != candidate_binding:
        raise TrainingOrchestrationError(
            "promotion aggregate bindings differ from re-captured source evidence"
        )
    if (
        candidate_binding.get("checkpoint_manifest_path")
        != candidate_manifest_record.get("path")
        or candidate_binding.get("checkpoint_manifest_sha256")
        != candidate_manifest_record.get("sha256")
    ):
        raise TrainingOrchestrationError(
            "promotion candidate aggregate names a different checkpoint manifest"
        )
    for role, binding in (
        ("baseline", baseline_binding),
        ("candidate", candidate_binding),
    ):
        for index, record in enumerate(binding["source_file_records"]):
            _record(
                record,
                base=project_root,
                expected_path=None,
                label=f"promotion {role} aggregate source {index}",
                cache=cache,
            )
    for worker_dir in candidate_binding["worker_run_dirs"]:
        worker_result = _json(
            _absolute(worker_dir) / "checkpoint_evaluation.json",
            label="promotion candidate worker result",
            cache=cache,
        )
        infos = worker_result.get("checkpoint_infos")
        if (
            not isinstance(infos, Mapping)
            or infos.get("global_policy_decisions")
            != bound_chunk["global_policy_decisions"]
            or _absolute(str(worker_result.get("checkpoint", ""))) != candidate_path
            or worker_result.get("checkpoint_sha256") != candidate_hash
        ):
            raise TrainingOrchestrationError(
                "promotion candidate worker is not bound to the chunk/global step"
            )
    return {
        "record": captured.record(),
        "promoted": promoted,
        "first_failed_gate": first_failed,
        "bound_chunk_index": bound_index,
        "bound_global_policy_decisions": bound_chunk["global_policy_decisions"],
        "candidate_checkpoint": dict(bound_chunk["immutable_history_checkpoint"]),
        "baseline_evaluation_aggregate": baseline_binding,
        "candidate_validation_aggregate": candidate_binding,
    }


def _load_profile(training_config: _Snapshot) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(training_config.data.decode("utf-8", errors="strict"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise TrainingOrchestrationError("training config is not valid UTF-8 YAML") from exc
    budgets = payload.get("budgets_policy_decisions") if isinstance(payload, Mapping) else None
    seeds = payload.get("seeds") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "wlr50_clean.ppo_training.phase_specific_stability.v1"
        or not isinstance(budgets, Mapping)
        or {stage.replace("-", "_"): STAGE_BUDGETS[stage] for stage in STAGES}
        != {key: budgets.get(key) for key in ("smoke", "phase_curriculum", "full_episode")}
        or budgets.get("deterministic_validation_interval")
        != DETERMINISTIC_VALIDATION_INTERVAL
        or budgets.get("early_stop_when_promotion_gate_passes") is not True
        or not isinstance(seeds, Mapping)
    ):
        raise TrainingOrchestrationError("training profile budget/orchestration contract is invalid")
    train = tuple(seeds.get("train", ()))
    validation = tuple(seeds.get("validation", ()))
    if (
        validation != VALIDATION_SEEDS
        or len(train) < 32
        or any(isinstance(value, bool) or not isinstance(value, int) for value in train)
    ):
        raise TrainingOrchestrationError("training profile seed contract is invalid")
    return {
        "stage_budgets": dict(STAGE_BUDGETS),
        "cadence_inputs": cadence_inputs_from_payload(payload),
        "deterministic_validation_interval": DETERMINISTIC_VALIDATION_INTERVAL,
        "early_stop_when_promotion_gate_passes": True,
        "train_seeds": train,
        "validation_seeds": validation,
    }


def _source_records(cache: Mapping[Path, _Snapshot]) -> list[dict[str, Any]]:
    return [cache[path].record() for path in sorted(cache, key=str)]


def _build_payload(
    *,
    training_run_dirs: Sequence[Path | str],
    screening_run_dirs: Sequence[Path | str],
    initial_checkpoint_publication_run: Path | str,
    training_config_path: Path | str,
    vector_benchmark_matrix_path: Path | str,
    promotion_decision_paths: Sequence[Path | str],
    project_root: Path,
    expected_seed: int | None,
    expected_num_envs: int | None,
    generated_at_utc: str,
    cache: _Cache,
) -> dict[str, Any]:
    if (
        not training_run_dirs
        or len(training_run_dirs) != len(screening_run_dirs)
        or len({_absolute(value) for value in training_run_dirs}) != len(training_run_dirs)
        or len({_absolute(value) for value in screening_run_dirs}) != len(screening_run_dirs)
    ):
        raise TrainingOrchestrationError(
            "training and screening runs must be non-empty, unique, one-to-one sequences"
        )
    if len({_absolute(value) for value in promotion_decision_paths}) != len(
        promotion_decision_paths
    ):
        raise TrainingOrchestrationError("promotion decision paths contain duplicates")
    training_config = _snapshot(
        training_config_path, label="training config", cache=cache
    )
    try:
        training_config.path.relative_to(project_root)
    except ValueError as exc:
        raise TrainingOrchestrationError("training config escapes the project root") from exc
    _reject_links(training_config.path, root=project_root, label="training config")
    profile = _load_profile(training_config)

    chunks = [
        _validate_training_chunk(
            value,
            project_root=project_root,
            training_config=training_config,
            interval=profile["deterministic_validation_interval"],
            cache=cache,
        )
        for value in training_run_dirs
    ]
    ranks = [STAGES.index(chunk["stage"]) for chunk in chunks]
    if ranks != sorted(ranks) or set(chunk["stage"] for chunk in chunks) != set(STAGES):
        raise TrainingOrchestrationError(
            "training chunks must include smoke -> phase-curriculum -> full-episode in order"
        )
    first = chunks[0]
    common_seed = first["identity"].get("seed")
    common_config_sha = first["identity"].get("config_sha256")
    common_git = first["identity"].get("git_commit")
    common_configs = first["configs"]
    if expected_seed is not None and common_seed != expected_seed:
        raise TrainingOrchestrationError("orchestration seed differs from training runs")
    if expected_num_envs not in (None, 1):
        raise TrainingOrchestrationError(
            "the offline orchestration job must reserve exactly one environment"
        )
    for index, chunk in enumerate(chunks):
        if (
            chunk["identity"].get("seed") != common_seed
            or chunk["identity"].get("config_sha256") != common_config_sha
            or chunk["identity"].get("git_commit") != common_git
            or chunk["configs"] != common_configs
        ):
            raise TrainingOrchestrationError("training chunk provenance differs across the chain")
        if index == 0:
            if chunk["resume_global_policy_decisions"] != 0:
                raise TrainingOrchestrationError("first training chunk must resume at global step zero")
        else:
            previous = chunks[index - 1]
            if (
                chunk["resume_global_policy_decisions"]
                != previous["global_policy_decisions"]
                or chunk["resume_checkpoint"]["path"]
                != previous["immutable_history_checkpoint"]["path"]
                or chunk["resume_checkpoint"]["sha256"]
                != previous["immutable_history_checkpoint"]["sha256"]
                or chunk["global_policy_decisions"]
                != previous["global_policy_decisions"] + chunk["stage_policy_decisions"]
            ):
                raise TrainingOrchestrationError("training resume/global-step chain is broken")

    initial_checkpoint = _validate_initial_checkpoint(
        first,
        project_root=project_root,
        expected_git_commit=str(common_git),
        expected_seed=int(common_seed),
        cache=cache,
    )
    initial_checkpoint_publication = _validate_initial_checkpoint_publication(
        initial_checkpoint_publication_run,
        initial_checkpoint,
        project_root=project_root,
        expected_seed=int(common_seed),
        expected_git_commit=str(common_git),
        expected_config_sha256=str(common_config_sha),
        expected_configs=common_configs,
        first_training_started_at=first["started_at"],
        cache=cache,
    )
    smoke_checkpoint, canonical_smoke_checkpoint = _validate_smoke_checkpoint(
        first, project_root=project_root, cache=cache
    )

    history_paths = [
        chunk["immutable_history_checkpoint"]["path"] for chunk in chunks
    ]
    if len(history_paths) != len(set(history_paths)):
        raise TrainingOrchestrationError(
            "immutable training history checkpoints are not unique"
        )
    terminal_chunk = chunks[-1]
    terminal_checkpoint_last = _snapshot(
        terminal_chunk["checkpoint_last_path"],
        label="terminal checkpoint_last",
        cache=cache,
    )
    if (
        terminal_checkpoint_last.sha256
        != terminal_chunk["immutable_history_checkpoint"]["sha256"]
    ):
        raise TrainingOrchestrationError(
            "terminal checkpoint_last differs from immutable terminal history"
        )

    screenings = []
    for index, (directory, chunk) in enumerate(
        zip(screening_run_dirs, chunks, strict=True)
    ):
        screening = _validate_screening(
            directory,
            project_root=project_root,
            chunk=chunk,
            validation_seeds=profile["validation_seeds"],
            cache=cache,
        )
        if screening["started_at"] <= chunk["completed_at"]:
            raise TrainingOrchestrationError("screening did not run after its training chunk")
        if index + 1 < len(chunks) and chunks[index + 1]["started_at"] <= screening[
            "completed_at"
        ]:
            raise TrainingOrchestrationError(
                "next training chunk began before the required screening completed"
            )
        screenings.append(screening)

    matrix_path = _absolute(vector_benchmark_matrix_path)
    selected_matrix = _json(
        matrix_path, label="vector benchmark matrix", cache=cache
    )
    selected_num_envs = _strict_int(
        selected_matrix.get("selected_num_envs"),
        label="vector matrix selected_num_envs",
        minimum=1,
    )
    train_seeds = tuple(profile["train_seeds"])
    if common_seed in train_seeds:
        offset = train_seeds.index(common_seed)
        train_seeds = train_seeds[offset:] + train_seeds[:offset]
    matrix_evidence = _validate_vector_matrix_binding(
        matrix_path,
        project_root=project_root,
        config_sha256=str(common_config_sha),
        config_records=common_configs,
        git_commit=str(common_git),
        run_seed=int(common_seed),
        expected_num_envs=selected_num_envs,
        expected_seed_rows=train_seeds[:selected_num_envs],
        cache=cache,
    )
    environment_counts_by_stage: dict[str, int] = {}
    for stage in STAGES:
        values = {chunk["num_envs"] for chunk in chunks if chunk["stage"] == stage}
        if len(values) != 1:
            raise TrainingOrchestrationError(
                f"{stage} chunks use inconsistent environment counts"
            )
        environment_counts_by_stage[stage] = values.pop()
    if environment_counts_by_stage["smoke"] != selected_num_envs:
        raise TrainingOrchestrationError(
            "smoke chunks must use the finalized matrix selected N"
        )
    if environment_counts_by_stage["phase-curriculum"] != 1:
        raise TrainingOrchestrationError(
            "phase-curriculum chunks must use one environment"
        )
    if environment_counts_by_stage["full-episode"] != selected_num_envs:
        raise TrainingOrchestrationError(
            "full-episode chunks must use the finalized matrix selected N"
        )

    try:
        cadence_plan = derive_training_cadence(
            selected_num_envs=selected_num_envs, **profile["cadence_inputs"]
        )
    except TrainingCadenceError as exc:
        raise TrainingOrchestrationError(f"training cadence profile is invalid: {exc}") from exc
    expected_chunks = cadence_plan["chunks"]
    if len(chunks) > len(expected_chunks):
        raise TrainingOrchestrationError("training history exceeds the derived cadence plan")
    for chunk, expected in zip(chunks, expected_chunks, strict=False):
        expected_cadence = expected["training_cadence"]
        if chunk.get("training_cadence") != expected_cadence:
            raise TrainingOrchestrationError("training chunk cadence differs from the matrix-bound plan")
        try:
            validate_training_chunk_cadence(
                expected_cadence,
                requested_policy_decisions=chunk["requested_policy_decisions"],
                iterations=chunk["iterations"], stage_policy_decisions=chunk["stage_policy_decisions"],
            )
        except TrainingCadenceError as exc:
            raise TrainingOrchestrationError(f"training chunk cadence is invalid: {exc}") from exc

    soft_binding: dict[str, Any] | None = None
    soft_raw: Any = None
    for chunk in chunks:
        if chunk["num_envs"] == 1:
            current_soft = chunk["soft_reset_acceptance_raw"]
            current_binding = _validate_soft_reset_binding(
                current_soft, project_root=project_root, cache=cache
            )
            if soft_binding is None:
                soft_binding = current_binding
                soft_raw = current_soft
            elif current_binding != soft_binding or current_soft != soft_raw:
                raise TrainingOrchestrationError(
                    "single-env soft-reset evidence changed across chunks"
                )
            if (
                chunk["vector_matrix_raw"] is not None
                or chunk["vector_matrix_path"] is not None
                or chunk["vector_matrix_sha256"] is not None
            ):
                raise TrainingOrchestrationError(
                    "single-env chunk unexpectedly claims vector matrix gating"
                )
        else:
            if chunk["num_envs"] != selected_num_envs:
                raise TrainingOrchestrationError(
                    "multi-env chunk did not use the finalized matrix selected N"
                )
            raw = chunk["vector_matrix_raw"]
            if (
                not isinstance(raw, Mapping)
                or dict(raw) != matrix_evidence
                or chunk["vector_matrix_path"] != str(matrix_path)
                or chunk["vector_matrix_sha256"] != matrix_evidence["sha256"]
                or chunk["soft_reset_acceptance_raw"] is not None
            ):
                raise TrainingOrchestrationError(
                    "multi-env training_result matrix binding is stale or bypassed"
                )

    promotions = [
        _validate_promotion_decision(
            path,
            chunks=chunks,
            project_root=project_root,
            cache=cache,
        )
        for path in promotion_decision_paths
    ]
    bound_indices = [row["bound_chunk_index"] for row in promotions]
    if len(bound_indices) != len(set(bound_indices)):
        raise TrainingOrchestrationError("multiple promotion decisions target one chunk")
    passing = [row for row in promotions if row["promoted"]]
    if len(passing) > 1:
        raise TrainingOrchestrationError("multiple promotion decisions claim early-stop authority")
    if passing and passing[0]["bound_chunk_index"] != len(chunks) - 1:
        raise TrainingOrchestrationError("training continued after a passing promotion decision")

    requested_by_stage = {
        stage: sum(
            chunk["requested_policy_decisions"]
            for chunk in chunks
            if chunk["stage"] == stage
        )
        for stage in STAGES
    }
    if passing:
        if (
            requested_by_stage["smoke"] != STAGE_BUDGETS["smoke"]
            or requested_by_stage["phase-curriculum"]
            != STAGE_BUDGETS["phase-curriculum"]
            or not 0
            < requested_by_stage["full-episode"]
            <= STAGE_BUDGETS["full-episode"]
        ):
            raise TrainingOrchestrationError("early-stop stage budget evidence is invalid")
        status = "PROMOTION_FOUND"
    else:
        if requested_by_stage != STAGE_BUDGETS:
            raise TrainingOrchestrationError(
                "stage requested-decision sums do not exhaust the configured budgets"
            )
        status = "BUDGET_EXHAUSTED_NO_PROMOTION"

    serial_chunks = []
    for index, (chunk, screening) in enumerate(zip(chunks, screenings, strict=True)):
        serial_chunks.append(
            {
                "index": index,
                "stage": chunk["stage"],
                "training_cadence": chunk["training_cadence"],
                "requested_policy_decisions": chunk["requested_policy_decisions"],
                "stage_policy_decisions": chunk["stage_policy_decisions"],
                "global_policy_decisions": chunk["global_policy_decisions"],
                "resume_global_policy_decisions": chunk[
                    "resume_global_policy_decisions"
                ],
                "num_envs": chunk["num_envs"],
                "rollout_length": chunk["rollout_length"],
                "iterations": chunk["iterations"],
                "ppo_batch_policy_decisions": (
                    chunk["num_envs"] * chunk["rollout_length"]
                ),
                "rounding_overrun_policy_decisions": (
                    chunk["stage_policy_decisions"]
                    - chunk["requested_policy_decisions"]
                ),
                "training": {
                    key: chunk[key]
                    for key in (
                        "run_directory",
                        "run_manifest",
                        "training_result",
                        "resume_checkpoint",
                        "immutable_history_checkpoint",
                        "checkpoint_manifest",
                        "frozen_hash_audits",
                        "committed_runtime_identities",
                    )
                },
                "screening": {
                    key: screening[key]
                    for key in (
                        "run_directory",
                        "run_manifest",
                        "checkpoint_evaluation",
                        "checkpoint",
                        "checkpoint_manifest",
                        "seed",
                        "global_policy_decisions",
                        "physical_passed",
                        "complete_evidence",
                        "episode_summary",
                        "trial_manifest",
                        "policy_trace",
                        "frozen_hash_audits",
                        "committed_runtime_identities",
                    )
                },
            }
        )
    terminal = serial_chunks[-1]
    actual_by_stage = {
        stage: sum(
            chunk["stage_policy_decisions"]
            for chunk in chunks
            if chunk["stage"] == stage
        )
        for stage in STAGES
    }
    rounding_overrun_by_stage = {
        stage: actual_by_stage[stage] - requested_by_stage[stage] for stage in STAGES
    }
    payload = {
        "schema": TRAINING_ORCHESTRATION_SCHEMA,
        "status": status,
        "valid": True,
        "generated_at_utc": generated_at_utc,
        "project_root": str(project_root),
        "training_seed": common_seed,
        "orchestration_environment_count": 1,
        "environment_counts_by_stage": environment_counts_by_stage,
        "selected_vector_num_envs": selected_num_envs,
        "git_commit": common_git,
        "config_sha256": common_config_sha,
        "config_records": list(common_configs),
        "training_config": training_config.record(),
        "initial_checkpoint": initial_checkpoint,
        "initial_checkpoint_publication": initial_checkpoint_publication,
        "smoke_checkpoint": smoke_checkpoint,
        "canonical_smoke_checkpoint": canonical_smoke_checkpoint,
        "vector_benchmark_matrix": _snapshot(
            matrix_path, label="vector benchmark matrix", cache=cache
        ).record(),
        "vector_benchmark_matrix_evidence": matrix_evidence,
        "soft_reset_acceptance": soft_binding,
        "required_stages": list(STAGES),
        "stage_budgets": dict(STAGE_BUDGETS),
        "base_validation_interval_policy_decisions": DETERMINISTIC_VALIDATION_INTERVAL,
        "base_validation_interval_scope": "smoke_and_phase_curriculum",
        "training_cadence": cadence_plan,
        "early_stop_only_on_five_seed_promotion": True,
        "stage_requested_policy_decisions": requested_by_stage,
        "stage_actual_policy_decisions": actual_by_stage,
        "stage_rounding_overrun_policy_decisions": rounding_overrun_by_stage,
        "budget_accounting_basis": "requested_policy_decisions",
        "actual_decisions_are_whole_ppo_batches": True,
        "chunk_count": len(serial_chunks),
        "chunks": serial_chunks,
        "promotion_decision_count": len(promotions),
        "promotion_decisions": promotions,
        "terminal": {
            "chunk_index": terminal["index"],
            "stage": terminal["stage"],
            "global_policy_decisions": terminal["global_policy_decisions"],
            "checkpoint": terminal["training"]["immutable_history_checkpoint"],
            "checkpoint_manifest": terminal["training"]["checkpoint_manifest"],
            "checkpoint_last": terminal_checkpoint_last.record(),
            "passing_promotion_decision": (
                None if not passing else passing[0]["record"]
            ),
            "promotion_bound_chunk_index": (
                None if not passing else passing[0]["bound_chunk_index"]
            ),
            "promotion_bound_global_policy_decisions": (
                None
                if not passing
                else serial_chunks[passing[0]["bound_chunk_index"]][
                    "global_policy_decisions"
                ]
            ),
            "promotion_candidate_checkpoint": (
                None if not passing else passing[0]["candidate_checkpoint"]
            ),
        },
    }
    records = _source_records(cache)
    payload["source_file_records"] = records
    payload["source_file_set_sha256"] = _canonical_hash(records)
    return payload


def build_training_orchestration_manifest(
    *,
    training_run_dirs: Sequence[Path | str],
    screening_run_dirs: Sequence[Path | str],
    initial_checkpoint_publication_run: Path | str,
    training_config_path: Path | str,
    vector_benchmark_matrix_path: Path | str,
    promotion_decision_paths: Sequence[Path | str] = (),
    output_path: Path | str,
    project_root: Path | str = PROJECT_ROOT,
    expected_seed: int | None = None,
    expected_num_envs: int | None = None,
    _before_publish_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Validate all evidence and publish the immutable pre-final manifest."""

    root = _absolute(project_root)
    output = _absolute(output_path)
    run_dir = _managed_run_dir(
        output.parent,
        project_root=root,
        run_kind=TRAINING_ORCHESTRATION_RUN_KIND,
        label="training orchestration run",
    )
    if output.name != TRAINING_ORCHESTRATION_FILENAME:
        raise TrainingOrchestrationError(
            f"output must be named {TRAINING_ORCHESTRATION_FILENAME}"
        )
    cache: _Cache = {}
    payload = _build_payload(
        training_run_dirs=training_run_dirs,
        screening_run_dirs=screening_run_dirs,
        initial_checkpoint_publication_run=initial_checkpoint_publication_run,
        training_config_path=training_config_path,
        vector_benchmark_matrix_path=vector_benchmark_matrix_path,
        promotion_decision_paths=promotion_decision_paths,
        project_root=root,
        expected_seed=expected_seed,
        expected_num_envs=expected_num_envs,
        generated_at_utc=_utc_text(),
        cache=cache,
    )
    started_path = run_dir / "run_manifest.started.json"
    started = _json(
        started_path, label="orchestration started manifest", cache=cache
    )
    identity = started.get("identity")
    if (
        started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
        or started.get("immutable_run_directory") is not True
        or started.get("run_kind") != TRAINING_ORCHESTRATION_RUN_KIND
        or started.get("entrypoint") != "wlr50_clean.ppo.training_orchestration"
        or started.get("subcommand") != "build-manifest"
        or _absolute(str(started.get("run_dir", ""))) != run_dir
        or _absolute(str(started.get("project_root", ""))) != root
        or not isinstance(identity, Mapping)
        or identity.get("training_stage") != "training-orchestration-prefinal"
        or identity.get("seed") != payload["training_seed"]
        or identity.get("environment_count")
        != payload["orchestration_environment_count"]
        or identity.get("git_commit") != payload["git_commit"]
        or identity.get("config_sha256") != payload["config_sha256"]
    ):
        raise TrainingOrchestrationError(
            "orchestration command is not in its matching managed reservation"
        )
    orchestration_configs = _validate_config_records(
        started.get("configs"),
        project_root=root,
        expected_config_sha256=identity.get("config_sha256"),
        cache=cache,
    )
    if list(orchestration_configs) != payload["config_records"]:
        raise TrainingOrchestrationError(
            "orchestration reservation config set differs from training"
        )
    runtime_before_path = run_dir / "committed_runtime_identity.before.json"
    _, _, runtime_before_snapshot = _validate_runtime_identity_document(
        runtime_before_path,
        project_root=root,
        expected_git_commit=str(identity.get("git_commit")),
        cache=cache,
    )
    frozen_manifest_path = root / _FROZEN_RELATIVE
    frozen_snapshot = _snapshot(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    frozen_manifest = _json(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    before_path = run_dir / "frozen_hashes.before.json"
    _validate_frozen_audit(
        before_path,
        project_root=root,
        frozen_manifest_path=frozen_manifest_path,
        frozen_manifest=frozen_manifest,
        frozen_snapshot=frozen_snapshot,
        cache=cache,
    )
    payload["orchestration_run_directory"] = str(run_dir)
    payload["orchestration_started_manifest"] = _snapshot(
        started_path, label="orchestration started manifest", cache=cache
    ).record()
    payload["orchestration_frozen_before"] = _snapshot(
        before_path, label="orchestration frozen-before audit", cache=cache
    ).record()
    payload["orchestration_runtime_identity_before"] = (
        runtime_before_snapshot.record()
    )
    records = _source_records(cache)
    payload["source_file_records"] = records
    payload["source_file_set_sha256"] = _canonical_hash(records)
    if output in cache:
        raise TrainingOrchestrationError("output overlaps a captured input")
    if _before_publish_hook is not None:
        _before_publish_hook()
    _revalidate(cache, project_root=root)
    # The hook may have replaced the validated run directory with a Windows
    # junction while leaving every captured leaf byte unchanged.
    _reject_links(
        output,
        root=root,
        label="training orchestration output",
    )
    try:
        atomic_write_json(output, payload)
    except ArtifactError as exc:
        raise TrainingOrchestrationError(str(exc)) from exc
    return payload


def validate_training_orchestration_manifest(
    path: Path | str, *, expected_project_root: Path | str = PROJECT_ROOT
) -> dict[str, Any]:
    """Reparse a finalized orchestration run and every flattened source record."""

    root = _absolute(expected_project_root)
    manifest_path = _absolute(path)
    if manifest_path.name != TRAINING_ORCHESTRATION_FILENAME:
        raise TrainingOrchestrationError("training orchestration manifest is misnamed")
    run_dir = _managed_run_dir(
        manifest_path.parent,
        project_root=root,
        run_kind=TRAINING_ORCHESTRATION_RUN_KIND,
        label="training orchestration run",
    )
    cache: _Cache = {}
    payload = _json(
        manifest_path, label="training orchestration manifest", cache=cache
    )
    if (
        payload.get("schema") != TRAINING_ORCHESTRATION_SCHEMA
        or payload.get("valid") is not True
        or payload.get("status")
        not in {"PROMOTION_FOUND", "BUDGET_EXHAUSTED_NO_PROMOTION"}
        or _absolute(str(payload.get("project_root", ""))) != root
        or payload.get("orchestration_run_directory") != str(run_dir)
    ):
        raise TrainingOrchestrationError("training orchestration header is invalid")
    source_records = payload.get("source_file_records")
    if not isinstance(source_records, Sequence) or isinstance(source_records, (str, bytes)):
        raise TrainingOrchestrationError("orchestration source file records are missing")
    declared_paths: set[Path] = set()
    source_cache: _Cache = {}
    for index, record in enumerate(source_records):
        captured = _record(
            record,
            base=root,
            expected_path=None,
            label=f"orchestration source {index}",
            cache=source_cache,
        )
        if captured.path == manifest_path or captured.path in declared_paths:
            raise TrainingOrchestrationError("orchestration sources overlap output or duplicate")
        declared_paths.add(captured.path)
    if payload.get("source_file_set_sha256") != _canonical_hash(list(source_records)):
        raise TrainingOrchestrationError("orchestration source record set hash is invalid")

    chunks = payload.get("chunks")
    decisions = payload.get("promotion_decisions")
    if (
        not isinstance(chunks, Sequence)
        or isinstance(chunks, (str, bytes))
        or not chunks
        or not isinstance(decisions, Sequence)
        or isinstance(decisions, (str, bytes))
    ):
        raise TrainingOrchestrationError("orchestration chunks/promotions are malformed")
    training_dirs: list[str] = []
    screening_dirs: list[str] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping) or chunk.get("index") != index:
            raise TrainingOrchestrationError("orchestration chunk ordering is invalid")
        training = chunk.get("training")
        screening = chunk.get("screening")
        if not isinstance(training, Mapping) or not isinstance(screening, Mapping):
            raise TrainingOrchestrationError("orchestration chunk evidence is missing")
        training_dirs.append(str(training.get("run_directory", "")))
        screening_dirs.append(str(screening.get("run_directory", "")))
    decision_paths = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("record"), Mapping):
            raise TrainingOrchestrationError("orchestration promotion record is malformed")
        decision_paths.append(str(decision["record"].get("path", "")))
    training_record = payload.get("training_config")
    matrix_record = payload.get("vector_benchmark_matrix")
    if not isinstance(training_record, Mapping) or not isinstance(matrix_record, Mapping):
        raise TrainingOrchestrationError("orchestration config/matrix records are missing")
    started_snapshot = _record(
        payload.get("orchestration_started_manifest"),
        base=run_dir,
        expected_path=None,
        label="orchestration started manifest",
        cache=source_cache,
    )
    before_snapshot = _record(
        payload.get("orchestration_frozen_before"),
        base=run_dir,
        expected_path=None,
        label="orchestration frozen-before audit",
        cache=source_cache,
    )
    runtime_before_snapshot = _record(
        payload.get("orchestration_runtime_identity_before"),
        base=run_dir,
        expected_path=None,
        label="orchestration committed-runtime identity before",
        cache=source_cache,
    )
    if started_snapshot.path != run_dir / "run_manifest.started.json":
        raise TrainingOrchestrationError(
            "orchestration started-manifest record names the wrong file"
        )
    if before_snapshot.path != run_dir / "frozen_hashes.before.json":
        raise TrainingOrchestrationError(
            "orchestration frozen-before record names the wrong file"
        )
    if (
        runtime_before_snapshot.path
        != run_dir / "committed_runtime_identity.before.json"
    ):
        raise TrainingOrchestrationError(
            "orchestration runtime-identity record names the wrong file"
        )
    _validate_runtime_identity_document(
        runtime_before_snapshot.path,
        project_root=root,
        expected_git_commit=str(payload.get("git_commit", "")),
        cache=source_cache,
    )
    frozen_manifest_path = root / _FROZEN_RELATIVE
    frozen_snapshot = _snapshot(
        frozen_manifest_path, label="frozen FSM manifest", cache=source_cache
    )
    _validate_frozen_audit(
        before_snapshot.path,
        project_root=root,
        frozen_manifest_path=frozen_manifest_path,
        frozen_manifest=_json(
            frozen_manifest_path,
            label="frozen FSM manifest",
            cache=source_cache,
        ),
        frozen_snapshot=frozen_snapshot,
        cache=source_cache,
    )
    rebuilt = _build_payload(
        training_run_dirs=training_dirs,
        screening_run_dirs=screening_dirs,
        initial_checkpoint_publication_run=str(
            payload.get("initial_checkpoint_publication", {}).get(
                "run_directory", ""
            )
            if isinstance(payload.get("initial_checkpoint_publication"), Mapping)
            else ""
        ),
        training_config_path=str(training_record.get("path", "")),
        vector_benchmark_matrix_path=str(matrix_record.get("path", "")),
        promotion_decision_paths=decision_paths,
        project_root=root,
        expected_seed=_strict_int(payload.get("training_seed"), label="training seed"),
        expected_num_envs=_strict_int(
            payload.get("orchestration_environment_count"),
            label="orchestration environment count",
            minimum=1,
        ),
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        cache=source_cache,
    )
    rebuilt["orchestration_run_directory"] = str(run_dir)
    rebuilt["orchestration_started_manifest"] = started_snapshot.record()
    rebuilt["orchestration_frozen_before"] = before_snapshot.record()
    rebuilt["orchestration_runtime_identity_before"] = (
        runtime_before_snapshot.record()
    )
    rebuilt_records = _source_records(source_cache)
    rebuilt["source_file_records"] = rebuilt_records
    rebuilt["source_file_set_sha256"] = _canonical_hash(rebuilt_records)
    if rebuilt != dict(payload):
        raise TrainingOrchestrationError(
            "orchestration manifest differs from re-parsed source semantics"
        )

    # Reuse all source snapshots in the lifecycle pass so no input is read twice.
    cache.update(source_cache)
    lifecycle = _validate_finalized_run(
        run_dir,
        project_root=root,
        run_kind=TRAINING_ORCHESTRATION_RUN_KIND,
        training_stage="training-orchestration-prefinal",
        entrypoint="wlr50_clean.ppo.training_orchestration",
        subcommand="build-manifest",
        cache=cache,
    )
    if TRAINING_ORCHESTRATION_FILENAME not in lifecycle["artifacts"]:
        raise TrainingOrchestrationError("orchestration lifecycle does not bind its manifest")
    lifecycle_identity = lifecycle["identity"]
    if (
        lifecycle_identity.get("seed") != payload["training_seed"]
        or lifecycle_identity.get("environment_count") != 1
        or lifecycle_identity.get("git_commit") != payload["git_commit"]
        or lifecycle_identity.get("config_sha256") != payload["config_sha256"]
        or list(lifecycle["configs"]) != payload["config_records"]
        or lifecycle["frozen_audits"][0] != before_snapshot.record()
        or lifecycle["committed_runtime_identities"][0]
        != runtime_before_snapshot.record()
    ):
        raise TrainingOrchestrationError(
            "orchestration lifecycle provenance differs from its training chain"
        )
    _revalidate(cache, project_root=root)
    captured = _snapshot(
        manifest_path, label="training orchestration manifest", cache=cache
    )
    return {
        "path": str(manifest_path),
        "bytes": captured.size,
        "sha256": captured.sha256,
        "payload": dict(payload),
        "source_file_records": list(source_records),
        "status": payload["status"],
        "valid": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable PPO training orchestration manifest"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--num-envs", type=int, required=True)
    build.add_argument("--training-run-dir", type=Path, action="append", required=True)
    build.add_argument("--screening-run-dir", type=Path, action="append", required=True)
    build.add_argument("--initial-checkpoint-publication-run", type=Path, required=True)
    build.add_argument("--training-config", type=Path, required=True)
    build.add_argument("--vector-benchmark-matrix", type=Path, required=True)
    build.add_argument("--promotion-decision", type=Path, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = _absolute(PROJECT_ROOT)
        payload = build_training_orchestration_manifest(
            training_run_dirs=args.training_run_dir,
            screening_run_dirs=args.screening_run_dir,
            initial_checkpoint_publication_run=(
                args.initial_checkpoint_publication_run
            ),
            training_config_path=args.training_config,
            vector_benchmark_matrix_path=args.vector_benchmark_matrix,
            promotion_decision_paths=args.promotion_decision,
            output_path=args.run_dir / TRAINING_ORCHESTRATION_FILENAME,
            project_root=root,
            expected_seed=args.seed,
            expected_num_envs=args.num_envs,
        )
        print(json.dumps(payload, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except (TrainingOrchestrationError, ArtifactError) as exc:
        print(f"TRAINING_ORCHESTRATION_ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMITTED_RUNTIME_IDENTITY_SCHEMA",
    "DEFAULT_TRAINING_ORCHESTRATION_RUNS",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_VECTOR_BENCHMARK_MATRIX",
    "DETERMINISTIC_VALIDATION_INTERVAL",
    "INITIAL_CHECKPOINT_PUBLICATION_RUN_KIND",
    "INITIAL_CHECKPOINT_RUN_KIND",
    "STAGE_BUDGETS",
    "TRAINING_ORCHESTRATION_FILENAME",
    "TRAINING_ORCHESTRATION_RUN_KIND",
    "TRAINING_ORCHESTRATION_SCHEMA",
    "TrainingOrchestrationError",
    "VECTOR_BENCHMARK_MATRIX_RUN_KIND",
    "build_training_orchestration_manifest",
    "main",
    "validate_training_orchestration_manifest",
]
