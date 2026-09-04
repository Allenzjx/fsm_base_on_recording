"""Offline, immutable selection of the largest proven vector environment count.

The live benchmark workers remain the only producers of physical evidence.
This module performs no Isaac import and no simulation: it re-hashes six
finalized workers, validates their lifecycle/stdout/frozen/config provenance,
and selects the largest 8/16/32 count for which both exact-zero and bounded
nonzero residual modes passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .artifacts import ArtifactError, atomic_write_json


MATRIX_SCHEMA = "wlr50_clean.vector_benchmark_matrix.v1"
SLOT_SCHEMA = "wlr50_clean.vector_benchmark_matrix_slot.v1"
BENCHMARK_SCHEMA = "wlr50_clean.vectorized_isaac_benchmark_run.v1"
SMOKE_SCHEMA = "wlr50_clean.vectorized_residual_smoke.v1"
RUN_MANIFEST_SCHEMA = "wlr50_clean.ppo_run_manifest.v1"
FROZEN_AUDIT_SCHEMA = "wlr50_clean.frozen_fsm_hash_audit.v1"
COMMITTED_RUNTIME_IDENTITY_SCHEMA = "wlr50_clean.committed_runtime_identity.v1"
ENVIRONMENT_COUNTS = (8, 16, 32)
MODES = ("zero", "bounded-smoke")
MINIMUM_MEASURED_TICKS = 1200
MINIMUM_POLICY_DECISIONS = 128
OUTPUT_FILENAME = "vector_benchmark_matrix.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HEX40 = frozenset("0123456789abcdef")
_FROZEN_MANIFEST_RELATIVE = Path(
    "artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json"
)
_RUNS_RELATIVE = Path("runs/ppo_phase_v1")
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


class VectorBenchmarkMatrixError(RuntimeError):
    """A worker artifact or the six-slot selection matrix is not trustworthy."""


@dataclass(frozen=True)
class _StableFileSnapshot:
    """One authoritative read used for parsing, hashing, sizing, and output records."""

    path: Path
    data: bytes
    size: int
    sha256: str
    trusted_root: Path | None = None

    def record(self) -> dict[str, Any]:
        return {"path": str(self.path), "bytes": self.size, "sha256": self.sha256}


_SnapshotCache = MutableMapping[Path, _StableFileSnapshot]


def _absolute_lexical(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_links(path: Path, *, root: Path, label: str) -> None:
    """Reject symlinks/junctions in every existing component beneath ``root``."""

    lexical = _absolute_lexical(path)
    lexical_root = _absolute_lexical(root)
    try:
        relative = lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise VectorBenchmarkMatrixError(f"{label} escapes the expected root") from exc
    cursors = [lexical_root]
    cursor = lexical_root
    for part in relative.parts:
        cursor /= part
        cursors.append(cursor)
    for cursor in cursors:
        try:
            metadata = os.stat(cursor, follow_symlinks=False)
            is_link = cursor.is_symlink()
            is_junction_fn = getattr(cursor, "is_junction", None)
            is_junction = bool(is_junction_fn()) if callable(is_junction_fn) else False
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise VectorBenchmarkMatrixError(f"cannot inspect {label}: {cursor}") from exc
        if (
            is_link
            or is_junction
            or int(getattr(metadata, "st_file_attributes", 0)) & 0x400
        ):
            raise VectorBenchmarkMatrixError(f"{label} contains a symlink/junction: {cursor}")


def _require_managed_path(
    path: Path | str,
    *,
    project_root: Path | str,
    run_kind: str,
    filename: str,
    label: str,
) -> tuple[Path, Path]:
    root = _absolute_lexical(project_root)
    runs_root = root / _RUNS_RELATIVE
    lexical = _absolute_lexical(path)
    try:
        relative = lexical.relative_to(runs_root)
    except ValueError as exc:
        raise VectorBenchmarkMatrixError(
            f"{label} must be inside {runs_root}"
        ) from exc
    if len(relative.parts) != 3 or relative.parts[0] != run_kind or relative.parts[2] != filename:
        raise VectorBenchmarkMatrixError(
            f"{label} must be {runs_root / run_kind / '<run-id>' / filename}"
        )
    _reject_links(lexical, root=root, label=label)
    if not lexical.is_file():
        raise VectorBenchmarkMatrixError(f"{label} is missing: {lexical}")
    return lexical, lexical.parent


def _require_managed_output_path(
    path: Path | str, *, project_root: Path | str
) -> Path:
    root = _absolute_lexical(project_root)
    runs_root = root / _RUNS_RELATIVE
    lexical = _absolute_lexical(path)
    try:
        relative = lexical.relative_to(runs_root)
    except ValueError as exc:
        raise VectorBenchmarkMatrixError(
            f"matrix output must be inside {runs_root}"
        ) from exc
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "vector_benchmark_matrix"
        or relative.parts[2] != OUTPUT_FILENAME
    ):
        raise VectorBenchmarkMatrixError(
            "matrix output is not in its managed vector_benchmark_matrix run"
        )
    _reject_links(lexical, root=root, label="matrix output")
    return lexical


def validate_managed_run_directory(
    path: Path | str, *, project_root: Path | str, run_kind: str
) -> Path:
    """Require an existing non-linked ``runs/ppo_phase_v1/<kind>/<id>`` directory."""

    root = _absolute_lexical(project_root)
    runs_root = root / _RUNS_RELATIVE
    run_dir = _absolute_lexical(path)
    try:
        relative = run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise VectorBenchmarkMatrixError(
            f"managed {run_kind} run must be inside {runs_root}"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[0] != run_kind or not relative.parts[1]:
        raise VectorBenchmarkMatrixError(
            f"managed run must be {runs_root / run_kind / '<run-id>'}"
        )
    _reject_links(run_dir, root=root, label=f"managed {run_kind} run")
    if not run_dir.is_dir():
        raise VectorBenchmarkMatrixError(
            f"managed {run_kind} run directory is missing: {run_dir}"
        )
    return run_dir


def _snapshot(
    path: Path | str,
    *,
    label: str,
    cache: _SnapshotCache,
    trusted_root: Path | None = None,
) -> _StableFileSnapshot:
    lexical = _absolute_lexical(path)
    trusted = None if trusted_root is None else _absolute_lexical(trusted_root)
    if trusted is not None:
        _reject_links(lexical, root=trusted, label=label)
    cached = cache.get(lexical)
    if cached is not None:
        if trusted is not None and cached.trusted_root is None:
            cached = _StableFileSnapshot(
                path=cached.path,
                data=cached.data,
                size=cached.size,
                sha256=cached.sha256,
                trusted_root=trusted,
            )
            cache[lexical] = cached
        return cached
    try:
        lexical_status = os.stat(lexical, follow_symlinks=False)
        if (
            lexical.is_symlink()
            or int(getattr(lexical_status, "st_file_attributes", 0)) & 0x400
        ):
            raise VectorBenchmarkMatrixError(
                f"{label} cannot be a symlink/junction: {lexical}"
            )
        with lexical.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise VectorBenchmarkMatrixError(f"{label} is not a regular file: {lexical}")
            data = stream.read()
            after = os.fstat(stream.fileno())
    except VectorBenchmarkMatrixError:
        raise
    except OSError as exc:
        raise VectorBenchmarkMatrixError(f"cannot read {label}: {lexical}") from exc
    if (
        before.st_size != len(data)
        or after.st_size != len(data)
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise VectorBenchmarkMatrixError(f"{label} changed while it was being read")
    result = _StableFileSnapshot(
        path=lexical,
        data=data,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        trusted_root=trusted,
    )
    cache[lexical] = result
    return result


def _revalidate_snapshots(
    cache: Mapping[Path, _StableFileSnapshot],
    *,
    project_root: Path | None = None,
) -> None:
    """Fail closed if any trusted input changed after validation."""

    for path, expected in sorted(cache.items(), key=lambda item: str(item[0])):
        trusted_root = project_root or expected.trusted_root
        if trusted_root is not None:
            _reject_links(path, root=trusted_root, label="validated input")
        try:
            metadata = os.stat(path, follow_symlinks=False)
            if (
                path.is_symlink()
                or int(getattr(metadata, "st_file_attributes", 0)) & 0x400
            ):
                raise VectorBenchmarkMatrixError(
                    f"validated input became a symlink/junction: {path}"
                )
            data = path.read_bytes()
        except OSError as exc:
            raise VectorBenchmarkMatrixError(
                f"validated input disappeared before publication: {path}"
            ) from exc
        if len(data) != expected.size or hashlib.sha256(data).hexdigest() != expected.sha256:
            raise VectorBenchmarkMatrixError(
                f"validated input changed before publication: {path}"
            )


def _utc_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _HEX40 for character in value.lower())
    )


def _load_object(
    path: Path, *, label: str, cache: _SnapshotCache
) -> Mapping[str, Any]:
    snapshot = _snapshot(path, label=label, cache=cache)
    if not snapshot.data:
        raise VectorBenchmarkMatrixError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(snapshot.data.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VectorBenchmarkMatrixError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise VectorBenchmarkMatrixError(f"{label} must be a JSON object: {path}")
    return payload


def _strict_int(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VectorBenchmarkMatrixError(f"{label} must be an integer")
    if positive and value <= 0:
        raise VectorBenchmarkMatrixError(f"{label} must be positive")
    return value


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VectorBenchmarkMatrixError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise VectorBenchmarkMatrixError(f"{label} must be positive and finite")
    return result


def _validate_config_set_bytes(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str,
    project_root: Path,
    cache: _SnapshotCache,
) -> None:
    if not records or not _is_hex(expected_sha256, 64):
        raise VectorBenchmarkMatrixError("current config set evidence is invalid")
    names: list[str] = []
    aggregate = hashlib.sha256()
    normalized: list[tuple[str, Mapping[str, Any]]] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise VectorBenchmarkMatrixError("current config record is malformed")
        name = str(record["path"])
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise VectorBenchmarkMatrixError("config record escapes the project root")
        names.append(name)
        normalized.append((name, record))
    if len(names) != len(set(names)) or names != sorted(names):
        raise VectorBenchmarkMatrixError("config records must be unique and sorted")
    for name, record in normalized:
        path = project_root / Path(name)
        _reject_links(path, root=project_root, label=f"config {name}")
        snapshot = _snapshot(path, label=f"config {name}", cache=cache)
        if (
            record.get("bytes") != snapshot.size
            or record.get("sha256") != snapshot.sha256
        ):
            raise VectorBenchmarkMatrixError(f"current config record changed: {name}")
        encoded_name = name.encode("utf-8")
        aggregate.update(len(encoded_name).to_bytes(8, "big"))
        aggregate.update(encoded_name)
        aggregate.update(snapshot.size.to_bytes(8, "big"))
        aggregate.update(snapshot.data)
    if aggregate.hexdigest() != expected_sha256:
        raise VectorBenchmarkMatrixError("current config aggregate SHA-256 mismatch")


def _validated_file_record(
    run_dir: Path,
    record: Any,
    *,
    expected_relative_path: str,
    label: str,
    cache: _SnapshotCache,
) -> _StableFileSnapshot:
    if not isinstance(record, Mapping):
        raise VectorBenchmarkMatrixError(f"{label} digest record is missing")
    if record.get("path") != expected_relative_path:
        raise VectorBenchmarkMatrixError(f"{label} digest record names the wrong path")
    selected = _absolute_lexical(run_dir / expected_relative_path)
    try:
        selected.relative_to(run_dir)
    except ValueError as exc:
        raise VectorBenchmarkMatrixError(f"{label} escapes its finalized run") from exc
    _reject_links(selected, root=run_dir, label=label)
    snapshot = _snapshot(selected, label=label, cache=cache)
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or snapshot.size != expected_bytes
        or not _is_hex(expected_sha256, 64)
        or snapshot.sha256 != str(expected_sha256).lower()
    ):
        raise VectorBenchmarkMatrixError(
            f"{label} digest mismatch or post-finalization tamper"
        )
    return snapshot


@lru_cache(maxsize=8)
def _committed_runtime_paths(
    project_root_text: str, git_commit: str
) -> tuple[str, ...]:
    project_root = Path(project_root_text)
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "ls-tree",
                "-r",
                "--name-only",
                git_commit,
                "--",
                *_COMMITTED_RUNTIME_ROOTS,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise VectorBenchmarkMatrixError(
            "cannot enumerate the committed PPO runtime tree"
        ) from exc
    paths = tuple(
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    )
    if result.returncode != 0 or not paths or len(paths) != len(set(paths)):
        raise VectorBenchmarkMatrixError(
            "cannot enumerate an exact committed PPO runtime tree"
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
        raise VectorBenchmarkMatrixError(
            "current PPO runtime bytes differ from the recorded source commit"
        )
    return paths


def _validate_runtime_identity_document(
    path: Path,
    *,
    project_root: Path,
    expected_git_commit: str,
    cache: _SnapshotCache,
) -> tuple[Mapping[str, Any], _StableFileSnapshot]:
    payload = _load_object(path, label="committed runtime identity", cache=cache)
    files = payload.get("files")
    if (
        payload.get("schema") != COMMITTED_RUNTIME_IDENTITY_SCHEMA
        or payload.get("git_commit") != expected_git_commit
        or not isinstance(files, Sequence)
        or isinstance(files, (str, bytes))
        or not files
        or payload.get("file_count") != len(files)
        or not _is_hex(payload.get("content_sha256"), 64)
        or not _is_hex(payload.get("aggregate_sha256"), 64)
    ):
        raise VectorBenchmarkMatrixError("committed runtime identity header is invalid")
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "bytes",
            "sha256",
            "creation_time_utc_ticks",
            "last_write_time_utc_ticks",
        }:
            raise VectorBenchmarkMatrixError(
                "committed runtime identity contains a malformed file row"
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
            raise VectorBenchmarkMatrixError(
                "committed runtime identity contains an unsafe/duplicate path"
            )
        names.add(name)
        source = project_root / name
        _reject_links(source, root=project_root, label=f"committed runtime file {name}")
        captured = _snapshot(
            source, label=f"committed runtime file {name}", cache=cache
        )
        creation_ticks = row.get("creation_time_utc_ticks")
        write_ticks = row.get("last_write_time_utc_ticks")
        if (
            row.get("bytes") != captured.size
            or row.get("sha256") != captured.sha256
            or isinstance(creation_ticks, bool)
            or not isinstance(creation_ticks, int)
            or creation_ticks <= 0
            or isinstance(write_ticks, bool)
            or not isinstance(write_ticks, int)
            or write_ticks <= 0
        ):
            raise VectorBenchmarkMatrixError(
                f"committed runtime identity file record is stale: {name}"
            )
        normalized.append(
            {
                "path": name,
                "bytes": captured.size,
                "sha256": captured.sha256,
                "creation_time_utc_ticks": creation_ticks,
                "last_write_time_utc_ticks": write_ticks,
            }
        )
    committed_paths = _committed_runtime_paths(str(project_root), expected_git_commit)
    if set(names) != set(committed_paths) or len(names) != len(committed_paths):
        raise VectorBenchmarkMatrixError(
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
        raise VectorBenchmarkMatrixError(
            "committed runtime identity content SHA-256 is invalid"
        )
    if hashlib.sha256(encoded).hexdigest() != payload.get("aggregate_sha256"):
        raise VectorBenchmarkMatrixError(
            "committed runtime identity aggregate SHA-256 is invalid"
        )
    return payload, _snapshot(path, label="committed runtime identity", cache=cache)


def _validate_runtime_identity_pair(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    project_root: Path,
    expected_git_commit: str,
    cache: _SnapshotCache,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise VectorBenchmarkMatrixError("managed run artifact map is missing")
    snapshots = []
    payloads = []
    for position in ("before", "after"):
        relative = f"committed_runtime_identity.{position}.json"
        snapshots.append(
            _validated_file_record(
                run_dir,
                artifacts.get(relative),
                expected_relative_path=relative,
                label=f"committed runtime identity {position}",
                cache=cache,
            )
        )
        payload, _ = _validate_runtime_identity_document(
            run_dir / relative,
            project_root=project_root,
            expected_git_commit=expected_git_commit,
            cache=cache,
        )
        payloads.append(payload)
    if dict(payloads[0]) != dict(payloads[1]):
        raise VectorBenchmarkMatrixError(
            "committed runtime identity changed during the managed run"
        )
    return snapshots[0].record(), snapshots[1].record()


def _invocation_value(arguments: Any, flag: str) -> str:
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise VectorBenchmarkMatrixError("vector benchmark invocation is missing")
    values = tuple(str(value) for value in arguments)
    positions = tuple(index for index, value in enumerate(values) if value == flag)
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        raise VectorBenchmarkMatrixError(
            f"vector benchmark invocation must contain one {flag} value"
        )
    return values[positions[0] + 1]


def _stdout_objects(
    path: Path, *, cache: _SnapshotCache
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    snapshot = _snapshot(path, label="vector benchmark stdout", cache=cache)
    try:
        text = snapshot.data.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise VectorBenchmarkMatrixError("vector benchmark stdout is not UTF-8") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            result.append(payload)
    return tuple(result)


def _validate_frozen_audit(
    path: Path,
    *,
    expected_project_root: Path,
    expected_manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_snapshot: _StableFileSnapshot,
    cache: _SnapshotCache,
) -> Mapping[str, Any]:
    audit = _load_object(path, label="vector benchmark frozen audit", cache=cache)
    protected = manifest.get("protected_files")
    source_head = manifest.get("source_head")
    if (
        manifest.get("algorithm") != "sha256"
        or not isinstance(protected, Mapping)
        or len(protected) != 29
        or len(set(protected)) != 29
        or not _is_hex(source_head, 40)
        or any(not isinstance(name, str) or not _is_hex(value, 64) for name, value in protected.items())
    ):
        raise VectorBenchmarkMatrixError("frozen FSM manifest is malformed")
    raw_entries = audit.get("entries")
    if (
        audit.get("schema") != FROZEN_AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("mismatches") != []
        or audit.get("protected_file_count") != 29
        or audit.get("source_head") != source_head
        or audit.get("frozen_manifest_sha256") != manifest_snapshot.sha256
        or _absolute_lexical(str(audit.get("project_root", ""))) != expected_project_root
        or _absolute_lexical(str(audit.get("frozen_manifest", ""))) != expected_manifest_path
        or not isinstance(raw_entries, Sequence)
        or isinstance(raw_entries, (str, bytes))
        or len(raw_entries) != 29
    ):
        raise VectorBenchmarkMatrixError(
            "vector benchmark frozen before/after audit is stale or failed"
        )
    observed: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise VectorBenchmarkMatrixError("frozen audit entry is malformed")
        name = entry.get("path")
        if (
            not isinstance(name, str)
            or name in observed
            or name not in protected
            or entry.get("expected_sha256") != protected[name]
            or entry.get("actual_sha256") != protected[name]
            or entry.get("exists") is not True
            or entry.get("valid") is not True
        ):
            raise VectorBenchmarkMatrixError("frozen audit entry set is stale or invalid")
        observed.add(name)
    if observed != set(protected):
        raise VectorBenchmarkMatrixError("frozen audit does not cover the manifest exactly")
    return audit


def _validate_passed_evidence(
    benchmark: Mapping[str, Any],
    *,
    mode: str,
    num_envs: int,
    seed_rows: tuple[int, ...],
    measured_ticks: int,
    policy_decisions: int,
) -> None:
    report = benchmark.get("report")
    smoke = benchmark.get("residual_smoke")
    resource = benchmark.get("resource_evidence")
    actual_smoke_mode = "zero" if mode == "zero" else "nonzero"
    expected_status = (
        "VECTOR_ZERO_RESIDUAL_SMOKE_PASSED"
        if mode == "zero"
        else "VECTOR_NONZERO_RESIDUAL_SMOKE_PASSED"
    )
    if benchmark.get("passed") is not True:
        raise VectorBenchmarkMatrixError("passed slot has a false benchmark pass flag")
    if (
        not isinstance(report, Mapping)
        or report.get("status") != "TRUE_BATCHED_ISAAC_VERIFIED"
        or report.get("num_envs") != num_envs
        or report.get("measured_ticks") != measured_ticks
        or report.get("true_batched_isaac_verified") is not True
        or report.get("one_simulation_context") is not True
        or report.get("articulation_tensor_instances") != num_envs
        or report.get("global_physics_steps") != measured_ticks
        or report.get("batched_articulation_writes") != measured_ticks
        or report.get("exact_pair_captures") != measured_ticks
        or report.get("exact_pair_sensor_count") != 13
        or report.get("independent_controller_count") != num_envs
        or report.get("independent_reader_count") != num_envs
        or report.get("failure_reasons") != []
        or len(tuple(report.get("final_state_ids", ()))) != num_envs
    ):
        raise VectorBenchmarkMatrixError(
            "passed vector slot lacks complete true-batched report evidence"
        )
    for name in (
        "wall_time_s",
        "physics_steps_per_second",
        "environment_steps_per_second",
    ):
        _finite_positive(report.get(name), label=f"report {name}")
    memory = resource.get("cuda_memory") if isinstance(resource, Mapping) else None
    contamination = (
        resource.get("contamination") if isinstance(resource, Mapping) else None
    )
    if (
        not isinstance(memory, Mapping)
        or memory.get("schema") != "wlr50_clean.vector_cuda_memory_evidence.v1"
        or not isinstance(memory.get("device"), str)
        or not str(memory.get("device")).startswith("cuda")
        or memory.get("peak_stats_reset_before_measured_section") is not True
        or memory.get("measurement_covers_throughput_and_residual_smoke") is not True
        or memory.get("peak_allocated_below_device_total") is not True
        or memory.get("peak_reserved_below_device_total") is not True
        or memory.get("oom_detected") is not False
        or not isinstance(contamination, Mapping)
        or contamination.get("schema")
        != "wlr50_clean.vector_contamination_evidence.v1"
        or contamination.get("evidence_complete") is not True
        or contamination.get("cross_environment_contamination_detected") is not False
        or contamination.get("fsm_state_contamination_detected") is not False
        or contamination.get("render_contamination_detected") is not False
        or contamination.get("measured_render_calls") != 0
        or contamination.get("independent_seed_count") != num_envs
        or contamination.get("independent_controller_count") != num_envs
        or contamination.get("independent_reader_count") != num_envs
        or contamination.get("independent_origin_count") != num_envs
    ):
        raise VectorBenchmarkMatrixError(
            "passed vector slot lacks CUDA capacity/contamination evidence"
        )
    memory_values = {}
    for name in (
        "allocated_bytes_at_measurement_start",
        "reserved_bytes_at_measurement_start",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "device_total_bytes",
    ):
        memory_values[name] = _strict_int(
            memory.get(name), label=f"CUDA memory {name}", positive=True
        )
    if (
        memory_values["peak_allocated_bytes"]
        < memory_values["allocated_bytes_at_measurement_start"]
        or memory_values["peak_reserved_bytes"]
        < memory_values["reserved_bytes_at_measurement_start"]
        or memory_values["peak_allocated_bytes"] >= memory_values["device_total_bytes"]
        or memory_values["peak_reserved_bytes"] >= memory_values["device_total_bytes"]
    ):
        raise VectorBenchmarkMatrixError(
            "passed vector slot CUDA peak memory is inconsistent"
        )
    if (
        not isinstance(smoke, Mapping)
        or smoke.get("schema") != SMOKE_SCHEMA
        or smoke.get("status") != expected_status
        or smoke.get("mode") != actual_smoke_mode
        or smoke.get("passed") is not True
        or smoke.get("num_envs") != num_envs
        or smoke.get("policy_decisions") != policy_decisions
        or smoke.get("physics_hz") != 120.0
        or smoke.get("decision_hz") != 15.0
        or smoke.get("physics_ticks_per_decision") != 8
        or smoke.get("measured_physics_ticks") != policy_decisions * 8
        or smoke.get("global_physics_steps") != policy_decisions * 8
        or smoke.get("batched_articulation_writes") != policy_decisions * 8
        or smoke.get("exact_pair_captures") != policy_decisions * 8
        or smoke.get("live_vectorized_isaac_backend_verified") is not True
        or smoke.get("independent_origin_count") != num_envs
        or smoke.get("independent_controller_count") != num_envs
        or smoke.get("independent_reader_count") != num_envs
        or smoke.get("independent_projection_bridge_count") != num_envs
        or smoke.get("all_masks_honored") is not True
        or smoke.get("all_zero_fast_path_expected") is not True
        or smoke.get("no_in_episode_root_writes") is not True
        or smoke.get("no_recording_runtime_access") is not True
        or smoke.get("no_termination_or_safety_events") is not True
    ):
        raise VectorBenchmarkMatrixError(
            "passed vector slot lacks complete residual-smoke evidence"
        )
    rows = smoke.get("rows")
    expected_row_count = num_envs * policy_decisions
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != expected_row_count
        or smoke.get("row_evidence_count") != expected_row_count
    ):
        raise VectorBenchmarkMatrixError("vector smoke row evidence is incomplete")
    observed_pairs: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise VectorBenchmarkMatrixError("vector smoke row is malformed")
        env_index = _strict_int(row.get("env_index"), label="smoke env_index")
        decision_index = _strict_int(
            row.get("decision_index"), label="smoke decision_index"
        )
        pair = (decision_index, env_index)
        if (
            not 0 <= env_index < num_envs
            or not 0 <= decision_index < policy_decisions
            or pair in observed_pairs
            or row.get("seed") != seed_rows[env_index]
            or row.get("mode") != actual_smoke_mode
            or row.get("physics_tick") != (decision_index + 1) * 8
            or row.get("in_episode_root_write_count") != 0
            or row.get("recording_runtime_access_count") != 0
            or row.get("terminated") is not False
            or row.get("truncated") is not False
        ):
            raise VectorBenchmarkMatrixError(
                "vector smoke row violates timing/seed/safety evidence"
            )
        observed_pairs.add(pair)
    maximum_fraction = smoke.get("maximum_observed_phase_scale_fraction")
    if isinstance(maximum_fraction, bool) or not isinstance(
        maximum_fraction, (int, float)
    ):
        raise VectorBenchmarkMatrixError("vector smoke amplitude is invalid")
    if mode == "zero":
        if (
            float(maximum_fraction) != 0.0
            or smoke.get("nonzero_active_row_count") != 0
            or smoke.get("zero_applied_equals_nominal_row_count") != len(rows)
        ):
            raise VectorBenchmarkMatrixError(
                "zero vector slot is not exact nominal identity"
            )
    elif (
        not 0.0 < float(maximum_fraction) < 0.05
        or smoke.get("nonzero_active_row_count") != len(rows)
        or smoke.get("deterministic_distinct_action_rows") is not True
    ):
        raise VectorBenchmarkMatrixError(
            "bounded vector slot lacks nonzero sub-five-percent activity"
        )


def validate_vector_benchmark_slot(
    path: Path | str,
    *,
    expected_config_sha256: str | None = None,
    expected_frozen_manifest_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_run_seed: int | None = None,
    expected_project_root: Path | str | None = None,
    expected_config_records: Sequence[Mapping[str, Any]] | None = None,
    _snapshot_cache: _SnapshotCache | None = None,
) -> dict[str, Any]:
    """Validate one successful or explicitly failed finalized live slot."""

    cache: _SnapshotCache = {} if _snapshot_cache is None else _snapshot_cache
    benchmark_path = _absolute_lexical(path)
    if benchmark_path.name != "vector_benchmark.json":
        raise VectorBenchmarkMatrixError(
            "each matrix input must be named vector_benchmark.json"
        )
    if expected_project_root is not None:
        benchmark_path, run_dir = _require_managed_path(
            benchmark_path,
            project_root=expected_project_root,
            run_kind="vector_benchmark",
            filename="vector_benchmark.json",
            label="vector benchmark",
        )
    else:
        run_dir = benchmark_path.parent
    benchmark = _load_object(benchmark_path, label="vector benchmark", cache=cache)
    manifest_path = run_dir / "run_manifest.json"
    started_path = run_dir / "run_manifest.started.json"
    manifest = _load_object(
        manifest_path, label="finalized vector run manifest", cache=cache
    )
    project_root = _absolute_lexical(str(manifest.get("project_root", "")))
    benchmark_path, managed_run_dir = _require_managed_path(
        benchmark_path,
        project_root=project_root,
        run_kind="vector_benchmark",
        filename="vector_benchmark.json",
        label="vector benchmark",
    )
    if managed_run_dir != run_dir:
        raise VectorBenchmarkMatrixError("vector benchmark run directory mismatch")
    started = _load_object(
        started_path, label="started vector run manifest", cache=cache
    )
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("immutable_run_directory") is not True
        or manifest.get("run_kind") != "vector_benchmark"
        or manifest.get("subcommand") != "vector-benchmark"
        or manifest.get("entrypoint") != "wlr50_clean.ppo.cli"
        or manifest.get("run_id") != run_dir.name
        or _absolute_lexical(str(manifest.get("run_dir", ""))) != run_dir
        or started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
    ):
        raise VectorBenchmarkMatrixError(
            "vector slot lacks a valid immutable finalized run manifest"
        )
    for key, value in started.items():
        if key != "lifecycle" and manifest.get(key) != value:
            raise VectorBenchmarkMatrixError(
                f"finalized vector manifest changed started field {key!r}"
            )
    _validated_file_record(
        run_dir,
        manifest.get("started_manifest"),
        expected_relative_path="run_manifest.started.json",
        label="started manifest",
        cache=cache,
    )
    logs = manifest.get("logs")
    artifacts = manifest.get("artifacts")
    if not isinstance(logs, Mapping) or not isinstance(artifacts, Mapping):
        raise VectorBenchmarkMatrixError(
            "finalized vector run omits log or artifact digest maps"
        )
    stdout_snapshot = _validated_file_record(
        run_dir,
        logs.get("stdout.log"),
        expected_relative_path="stdout.log",
        label="stdout",
        cache=cache,
    )
    stderr_snapshot = _validated_file_record(
        run_dir,
        logs.get("stderr.log"),
        expected_relative_path="stderr.log",
        label="stderr",
        cache=cache,
    )
    for relative in (
        "vector_benchmark.json",
        "frozen_hashes.before.json",
        "frozen_hashes.after.json",
    ):
        _validated_file_record(
            run_dir,
            artifacts.get(relative),
            expected_relative_path=relative,
            label=f"finalized {relative}",
            cache=cache,
        )

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise VectorBenchmarkMatrixError("vector run identity is missing")
    config_sha256 = identity.get("config_sha256")
    git_commit = identity.get("git_commit")
    run_seed = _strict_int(identity.get("seed"), label="vector run seed")
    num_envs = _strict_int(
        identity.get("environment_count"), label="vector environment count"
    )
    if (
        not _is_hex(config_sha256, 64)
        or not _is_hex(git_commit, 40)
        or num_envs not in ENVIRONMENT_COUNTS
        or identity.get("training_stage") != "backend-benchmark"
    ):
        raise VectorBenchmarkMatrixError("vector run identity is invalid")
    if expected_project_root is not None and project_root != _absolute_lexical(
        expected_project_root
    ):
        raise VectorBenchmarkMatrixError("vector run project root mismatch")
    if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
        raise VectorBenchmarkMatrixError("vector run config SHA-256 mismatch")
    if expected_git_commit is not None and git_commit != expected_git_commit:
        raise VectorBenchmarkMatrixError("vector run git commit mismatch")
    if expected_run_seed is not None and run_seed != expected_run_seed:
        raise VectorBenchmarkMatrixError("vector run seed mismatch")
    configs = manifest.get("configs")
    if not isinstance(configs, Sequence) or isinstance(configs, (str, bytes)):
        raise VectorBenchmarkMatrixError("vector run config records are missing")
    config_records = [dict(value) for value in configs if isinstance(value, Mapping)]
    if len(config_records) != len(configs):
        raise VectorBenchmarkMatrixError("vector run config records are malformed")
    if expected_config_records is not None and config_records != [
        dict(value) for value in expected_config_records
    ]:
        raise VectorBenchmarkMatrixError("vector run config record set mismatch")
    runtime_before, runtime_after = _validate_runtime_identity_pair(
        run_dir,
        manifest=manifest,
        project_root=project_root,
        expected_git_commit=str(git_commit),
        cache=cache,
    )

    invocation = manifest.get("invocation_arguments")
    if _invocation_value(invocation, "--seed-set") != "train":
        raise VectorBenchmarkMatrixError("vector slot did not use the train seed set")
    mode = _invocation_value(invocation, "--residual-mode")
    if mode not in MODES:
        raise VectorBenchmarkMatrixError("vector slot residual mode is invalid")
    if _invocation_value(invocation, "--seed") != str(run_seed):
        raise VectorBenchmarkMatrixError("vector slot invocation seed mismatch")
    if _invocation_value(invocation, "--num-envs") != str(num_envs):
        raise VectorBenchmarkMatrixError("vector slot invocation environment mismatch")
    try:
        measured_ticks = int(_invocation_value(invocation, "--measured-ticks"))
        policy_decisions = int(_invocation_value(invocation, "--policy-decisions"))
    except ValueError as exc:
        raise VectorBenchmarkMatrixError(
            "vector slot horizon invocation is not numeric"
        ) from exc
    if (
        measured_ticks < MINIMUM_MEASURED_TICKS
        or policy_decisions < MINIMUM_POLICY_DECISIONS
    ):
        raise VectorBenchmarkMatrixError(
            "vector slot horizons are too weak: require measured_ticks >= 1200 "
            "and policy_decisions >= 128"
        )

    if benchmark.get("schema") != BENCHMARK_SCHEMA:
        raise VectorBenchmarkMatrixError("vector benchmark schema is invalid")
    raw_seed_rows = benchmark.get("seed_rows")
    if not isinstance(raw_seed_rows, Sequence) or isinstance(
        raw_seed_rows, (str, bytes)
    ):
        raise VectorBenchmarkMatrixError("vector benchmark seed_rows are missing")
    seed_rows = tuple(
        _strict_int(value, label="vector seed row") for value in raw_seed_rows
    )
    if (
        len(seed_rows) != num_envs
        or len(set(seed_rows)) != num_envs
        or any(seed < 0 for seed in seed_rows)
        or seed_rows[0] != run_seed
    ):
        raise VectorBenchmarkMatrixError("vector benchmark seed_rows are invalid")

    stdout_objects = _stdout_objects(stdout_snapshot.path, cache=cache)
    bound_payloads = tuple(
        value for value in stdout_objects if value.get("schema") == BENCHMARK_SCHEMA
    )
    if len(bound_payloads) != 1 or dict(bound_payloads[0]) != dict(benchmark):
        raise VectorBenchmarkMatrixError(
            "vector benchmark is not exactly bound to finalized stdout"
        )
    before_path = _absolute_lexical(run_dir / "frozen_hashes.before.json")
    after_path = _absolute_lexical(run_dir / "frozen_hashes.after.json")
    stdout_audits = {
        _absolute_lexical(str(value["audit"]))
        for value in stdout_objects
        if value.get("passed") is True and isinstance(value.get("audit"), str)
    }
    if not {before_path, after_path}.issubset(stdout_audits):
        raise VectorBenchmarkMatrixError(
            "finalized stdout omits the frozen before/after audit bindings"
        )
    frozen_manifest_path = project_root / _FROZEN_MANIFEST_RELATIVE
    _reject_links(
        frozen_manifest_path, root=project_root, label="frozen FSM manifest"
    )
    frozen_manifest_snapshot = _snapshot(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    frozen_manifest = _load_object(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    frozen_manifest_sha256 = frozen_manifest_snapshot.sha256
    if (
        expected_frozen_manifest_sha256 is not None
        and frozen_manifest_sha256 != expected_frozen_manifest_sha256
    ):
        raise VectorBenchmarkMatrixError("frozen manifest SHA-256 mismatch")
    before_audit = _validate_frozen_audit(
        before_path,
        expected_project_root=project_root,
        expected_manifest_path=frozen_manifest_path,
        manifest=frozen_manifest,
        manifest_snapshot=frozen_manifest_snapshot,
        cache=cache,
    )
    after_audit = _validate_frozen_audit(
        after_path,
        expected_project_root=project_root,
        expected_manifest_path=frozen_manifest_path,
        manifest=frozen_manifest,
        manifest_snapshot=frozen_manifest_snapshot,
        cache=cache,
    )
    for field in ("protected_file_count", "entries", "mismatches", "passed"):
        if before_audit.get(field) != after_audit.get(field):
            raise VectorBenchmarkMatrixError(
                f"frozen before/after audits differ for {field}"
            )

    passed = benchmark.get("passed") is True
    if passed:
        if manifest.get("lifecycle") != "SUCCEEDED" or manifest.get("exit_code") != 0:
            raise VectorBenchmarkMatrixError(
                "passed vector slot lacks a successful finalized lifecycle"
            )
        _validate_passed_evidence(
            benchmark,
            mode=mode,
            num_envs=num_envs,
            seed_rows=seed_rows,
            measured_ticks=measured_ticks,
            policy_decisions=policy_decisions,
        )
        failure_reasons: tuple[str, ...] = ()
    else:
        report = benchmark.get("report")
        resource = benchmark.get("resource_evidence")
        raw_reasons = report.get("failure_reasons") if isinstance(report, Mapping) else None
        if (
            benchmark.get("passed") is not False
            or manifest.get("lifecycle") != "FAILED"
            or manifest.get("exit_code") != 2
            or not isinstance(report, Mapping)
            or report.get("status") != "VECTOR_BACKEND_BENCHMARK_FAILED"
            or report.get("num_envs") != num_envs
            or report.get("true_batched_isaac_verified") is not False
            or benchmark.get("residual_smoke") is not None
            or not isinstance(resource, Mapping)
            or not isinstance(resource.get("cuda_memory"), Mapping)
            or not isinstance(resource["cuda_memory"].get("oom_detected"), bool)
            or not isinstance(raw_reasons, Sequence)
            or isinstance(raw_reasons, (str, bytes))
        ):
            raise VectorBenchmarkMatrixError(
                "failed vector slot lacks a finalized capacity-failure lifecycle"
            )
        if any(not isinstance(value, str) for value in raw_reasons):
            raise VectorBenchmarkMatrixError(
                "failed vector slot reasons must be native strings"
            )
        failure_reasons = tuple(value.strip() for value in raw_reasons)
        if not failure_reasons or any(not value for value in failure_reasons):
            raise VectorBenchmarkMatrixError(
                "failed vector slot must record non-empty failure reasons"
            )

    records = {
        "vector_benchmark": _snapshot(benchmark_path, label="vector benchmark", cache=cache).record(),
        "run_manifest": _snapshot(manifest_path, label="run manifest", cache=cache).record(),
        "run_manifest_started": _snapshot(started_path, label="started manifest", cache=cache).record(),
        "stdout": stdout_snapshot.record(),
        "stderr": stderr_snapshot.record(),
        "frozen_hashes_before": _snapshot(before_path, label="frozen before audit", cache=cache).record(),
        "frozen_hashes_after": _snapshot(after_path, label="frozen after audit", cache=cache).record(),
        "frozen_manifest": frozen_manifest_snapshot.record(),
        "committed_runtime_identity_before": runtime_before,
        "committed_runtime_identity_after": runtime_after,
    }
    return {
        "schema": SLOT_SCHEMA,
        "slot": f"n{num_envs}_{mode}",
        "num_envs": num_envs,
        "mode": mode,
        "passed": passed,
        "failure_reasons": list(failure_reasons),
        "run_seed": run_seed,
        "seed_rows": list(seed_rows),
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "config_records": config_records,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "frozen_source_head": frozen_manifest.get("source_head"),
        "project_root": str(project_root),
        "measured_ticks": measured_ticks,
        "policy_decisions": policy_decisions,
        "environment_steps_per_second": (
            float(benchmark["report"]["environment_steps_per_second"])
            if passed
            else None
        ),
        "resource_evidence": dict(benchmark["resource_evidence"]),
        "records": records,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_vector_benchmark_matrix(
    benchmark_paths: Sequence[Path | str],
    *,
    output_path: Path | str,
    expected_config_sha256: str | None = None,
    expected_frozen_manifest_sha256: str | None = None,
    expected_git_commit: str | None = None,
    expected_run_seed: int | None = None,
    expected_project_root: Path | str | None = None,
    expected_config_records: Sequence[Mapping[str, Any]] | None = None,
    _before_publish_hook: Callable[[], None] | None = None,
    _managed_started_path: Path | str | None = None,
    _managed_started_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate exactly six slots, select a monotonic dual-mode capacity prefix."""

    if len(benchmark_paths) != len(ENVIRONMENT_COUNTS) * len(MODES):
        raise VectorBenchmarkMatrixError("benchmark matrix requires exactly six inputs")
    resolved = tuple(_absolute_lexical(value) for value in benchmark_paths)
    if len(set(resolved)) != len(resolved):
        raise VectorBenchmarkMatrixError("benchmark matrix contains duplicate input paths")
    snapshots: _SnapshotCache = {}
    if _managed_started_path is not None:
        started_snapshot = _snapshot(
            _managed_started_path,
            label="matrix started run manifest",
            cache=snapshots,
        )
        if started_snapshot.sha256 != _managed_started_sha256:
            raise VectorBenchmarkMatrixError(
                "matrix started manifest changed before aggregation"
            )
    if (
        expected_project_root is not None
        and expected_config_sha256 is not None
        and expected_config_records is not None
    ):
        _validate_config_set_bytes(
            expected_config_records,
            expected_sha256=expected_config_sha256,
            project_root=_absolute_lexical(expected_project_root),
            cache=snapshots,
        )
    slots = [
        validate_vector_benchmark_slot(
            path,
            expected_config_sha256=expected_config_sha256,
            expected_frozen_manifest_sha256=expected_frozen_manifest_sha256,
            expected_git_commit=expected_git_commit,
            expected_run_seed=expected_run_seed,
            expected_project_root=expected_project_root,
            expected_config_records=expected_config_records,
            _snapshot_cache=snapshots,
        )
        for path in resolved
    ]
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for slot in slots:
        key = (int(slot["num_envs"]), str(slot["mode"]))
        if key in by_key:
            raise VectorBenchmarkMatrixError(f"duplicate benchmark matrix slot: {key}")
        by_key[key] = slot
    expected_keys = {
        (num_envs, mode) for num_envs in ENVIRONMENT_COUNTS for mode in MODES
    }
    missing = sorted(expected_keys - set(by_key))
    unexpected = sorted(set(by_key) - expected_keys)
    if missing or unexpected:
        raise VectorBenchmarkMatrixError(
            f"benchmark matrix slot set mismatch; missing={missing}, unexpected={unexpected}"
        )

    common_fields = (
        "git_commit",
        "config_sha256",
        "config_records",
        "frozen_manifest_sha256",
        "frozen_source_head",
        "project_root",
        "run_seed",
        "measured_ticks",
        "policy_decisions",
    )
    reference = by_key[(8, "zero")]
    for key, slot in by_key.items():
        for field in common_fields:
            if slot[field] != reference[field]:
                raise VectorBenchmarkMatrixError(
                    f"benchmark slot {key} differs in common provenance field {field}"
                )
    for num_envs in ENVIRONMENT_COUNTS:
        zero_seeds = by_key[(num_envs, "zero")]["seed_rows"]
        bounded_seeds = by_key[(num_envs, "bounded-smoke")]["seed_rows"]
        if zero_seeds != bounded_seeds:
            raise VectorBenchmarkMatrixError(
                f"zero/bounded seed rows differ at num_envs={num_envs}"
            )
    seeds_32 = by_key[(32, "zero")]["seed_rows"]
    for num_envs in (8, 16):
        if by_key[(num_envs, "zero")]["seed_rows"] != seeds_32[:num_envs]:
            raise VectorBenchmarkMatrixError(
                "vector benchmark seed rows are not one shared 8/16/32 prefix"
            )

    dual_mode_pass = {
        num_envs: all(by_key[(num_envs, mode)]["passed"] for mode in MODES)
        for num_envs in ENVIRONMENT_COUNTS
    }
    candidates = tuple(
        num_envs for num_envs in ENVIRONMENT_COUNTS if dual_mode_pass[num_envs]
    )
    if not candidates:
        raise VectorBenchmarkMatrixError(
            "no environment count passed both zero and bounded-smoke gates"
        )
    selected_num_envs = max(candidates)
    if any(
        not dual_mode_pass[num_envs]
        for num_envs in ENVIRONMENT_COUNTS
        if num_envs < selected_num_envs
    ):
        raise VectorBenchmarkMatrixError(
            "dual-mode vector capacity is not a monotonic pass prefix"
        )
    for num_envs in ENVIRONMENT_COUNTS:
        if num_envs <= selected_num_envs:
            continue
        failed = [
            by_key[(num_envs, mode)]
            for mode in MODES
            if not by_key[(num_envs, mode)]["passed"]
        ]
        if not failed or any(not slot["failure_reasons"] for slot in failed):
            raise VectorBenchmarkMatrixError(
                f"larger failed capacity {num_envs} lacks recorded reasons"
            )

    ordered_slots = [
        by_key[(num_envs, mode)]
        for num_envs in ENVIRONMENT_COUNTS
        for mode in MODES
    ]
    source_records = [
        {
            "slot": slot["slot"],
            **slot["records"]["vector_benchmark"],
            "run_manifest_sha256": slot["records"]["run_manifest"]["sha256"],
            "stdout_sha256": slot["records"]["stdout"]["sha256"],
            "frozen_before_sha256": slot["records"]["frozen_hashes_before"][
                "sha256"
            ],
            "frozen_after_sha256": slot["records"]["frozen_hashes_after"][
                "sha256"
            ],
        }
        for slot in ordered_slots
    ]
    payload = {
        "schema": MATRIX_SCHEMA,
        "status": "VECTOR_BENCHMARK_MATRIX_ACCEPTED",
        "passed": True,
        "generated_at_utc": _utc_text(),
        "required_environment_counts": list(ENVIRONMENT_COUNTS),
        "required_modes": list(MODES),
        "required_slot_count": 6,
        "validated_slot_count": len(ordered_slots),
        "selected_num_envs": selected_num_envs,
        "selection_rule": "highest_monotonic_environment_count_where_both_modes_pass",
        "dual_mode_pass_by_environment_count": {
            str(key): value for key, value in dual_mode_pass.items()
        },
        "common_provenance": {
            field: reference[field] for field in common_fields
        },
        "selected_acceptance": {
            mode: {
                "path": by_key[(selected_num_envs, mode)]["records"][
                    "vector_benchmark"
                ]["path"],
                "sha256": by_key[(selected_num_envs, mode)]["records"][
                    "vector_benchmark"
                ]["sha256"],
            }
            for mode in MODES
        },
        "selected_capacity_checks": {
            "oom_detected": False,
            "cross_environment_contamination_detected": False,
            "fsm_state_contamination_detected": False,
            "render_contamination_detected": False,
            "resource_evidence_by_mode": {
                mode: by_key[(selected_num_envs, mode)]["resource_evidence"]
                for mode in MODES
            },
        },
        "source_artifact_set_sha256": _canonical_sha256(source_records),
        "source_artifacts": source_records,
        "slots": ordered_slots,
    }
    publication_root = _absolute_lexical(
        expected_project_root
        if expected_project_root is not None
        else reference["project_root"]
    )
    output = _require_managed_output_path(
        output_path,
        project_root=publication_root,
    )
    if (
        _managed_started_path is not None
        and output.parent != _absolute_lexical(_managed_started_path).parent
    ):
        raise VectorBenchmarkMatrixError(
            "matrix output differs from its managed started manifest run"
        )
    if _before_publish_hook is not None:
        _before_publish_hook()
    _revalidate_snapshots(snapshots, project_root=publication_root)
    _reject_links(output, root=publication_root, label="matrix output")
    try:
        atomic_write_json(output, payload)
    except ArtifactError as exc:
        raise VectorBenchmarkMatrixError(str(exc)) from exc
    return payload


def _invocation_values(arguments: Any, flag: str) -> tuple[str, ...]:
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise VectorBenchmarkMatrixError("matrix invocation is missing")
    values = tuple(str(value) for value in arguments)
    positions = tuple(index for index, value in enumerate(values) if value == flag)
    if any(index + 1 >= len(values) for index in positions):
        raise VectorBenchmarkMatrixError(f"matrix invocation has a dangling {flag}")
    return tuple(values[index + 1] for index in positions)


def validate_finalized_vector_benchmark_matrix(
    path: Path | str,
    *,
    expected_project_root: Path | str,
    expected_config_sha256: str,
    expected_frozen_manifest_sha256: str,
    expected_git_commit: str,
    expected_run_seed: int,
    expected_num_envs: int,
    expected_seed_rows: Sequence[int],
    expected_config_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Revalidate a finalized matrix and all six immutable worker runs for training."""

    project_root = _absolute_lexical(expected_project_root)
    matrix_path, run_dir = _require_managed_path(
        path,
        project_root=project_root,
        run_kind="vector_benchmark_matrix",
        filename=OUTPUT_FILENAME,
        label="vector benchmark matrix",
    )
    cache: _SnapshotCache = {}
    _validate_config_set_bytes(
        expected_config_records,
        expected_sha256=expected_config_sha256,
        project_root=project_root,
        cache=cache,
    )
    matrix = _load_object(matrix_path, label="vector benchmark matrix", cache=cache)
    manifest_path = run_dir / "run_manifest.json"
    started_path = run_dir / "run_manifest.started.json"
    manifest = _load_object(
        manifest_path, label="finalized matrix run manifest", cache=cache
    )
    started = _load_object(
        started_path, label="started matrix run manifest", cache=cache
    )
    identity = manifest.get("identity")
    configs = manifest.get("configs")
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("lifecycle") != "SUCCEEDED"
        or manifest.get("exit_code") != 0
        or manifest.get("immutable_run_directory") is not True
        or manifest.get("run_kind") != "vector_benchmark_matrix"
        or manifest.get("entrypoint") != "wlr50_clean.ppo.vector_benchmark_matrix"
        or manifest.get("subcommand") != "aggregate"
        or manifest.get("run_id") != run_dir.name
        or _absolute_lexical(str(manifest.get("run_dir", ""))) != run_dir
        or _absolute_lexical(str(manifest.get("project_root", ""))) != project_root
        or started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
        or not isinstance(identity, Mapping)
        or identity.get("training_stage") != "backend-benchmark-selection"
        or identity.get("environment_count") != 1
        or identity.get("seed") != expected_run_seed
        or identity.get("config_sha256") != expected_config_sha256
        or identity.get("git_commit") != expected_git_commit
        or not isinstance(configs, Sequence)
        or isinstance(configs, (str, bytes))
    ):
        raise VectorBenchmarkMatrixError(
            "matrix lacks a successful matching immutable run manifest"
        )
    for key, value in started.items():
        if key != "lifecycle" and manifest.get(key) != value:
            raise VectorBenchmarkMatrixError(
                f"finalized matrix manifest changed started field {key!r}"
            )
    config_records = [dict(value) for value in configs if isinstance(value, Mapping)]
    if config_records != [dict(value) for value in expected_config_records]:
        raise VectorBenchmarkMatrixError("matrix config record set mismatch")

    _validated_file_record(
        run_dir,
        manifest.get("started_manifest"),
        expected_relative_path="run_manifest.started.json",
        label="matrix started manifest",
        cache=cache,
    )
    logs = manifest.get("logs")
    artifacts = manifest.get("artifacts")
    if not isinstance(logs, Mapping) or not isinstance(artifacts, Mapping):
        raise VectorBenchmarkMatrixError("matrix manifest omits log/artifact digests")
    stdout_snapshot = _validated_file_record(
        run_dir,
        logs.get("stdout.log"),
        expected_relative_path="stdout.log",
        label="matrix stdout",
        cache=cache,
    )
    stderr_snapshot = _validated_file_record(
        run_dir,
        logs.get("stderr.log"),
        expected_relative_path="stderr.log",
        label="matrix stderr",
        cache=cache,
    )
    for relative in (
        OUTPUT_FILENAME,
        "frozen_hashes.before.json",
        "frozen_hashes.after.json",
    ):
        _validated_file_record(
            run_dir,
            artifacts.get(relative),
            expected_relative_path=relative,
            label=f"finalized matrix {relative}",
            cache=cache,
        )
    matrix_runtime_before, matrix_runtime_after = _validate_runtime_identity_pair(
        run_dir,
        manifest=manifest,
        project_root=project_root,
        expected_git_commit=expected_git_commit,
        cache=cache,
    )

    invocation = manifest.get("invocation_arguments")
    if (
        _invocation_value(invocation, "--seed") != str(expected_run_seed)
        or _invocation_value(invocation, "--num-envs") != "1"
    ):
        raise VectorBenchmarkMatrixError("matrix controlled invocation mismatch")
    benchmark_arguments = _invocation_values(invocation, "--benchmark")
    if len(benchmark_arguments) != 6:
        raise VectorBenchmarkMatrixError("matrix invocation must bind exactly six workers")

    stdout_objects = _stdout_objects(stdout_snapshot.path, cache=cache)
    bound = tuple(item for item in stdout_objects if item.get("schema") == MATRIX_SCHEMA)
    if len(bound) != 1 or dict(bound[0]) != dict(matrix):
        raise VectorBenchmarkMatrixError("matrix is not exactly bound to finalized stdout")
    before_path = run_dir / "frozen_hashes.before.json"
    after_path = run_dir / "frozen_hashes.after.json"
    stdout_audits = {
        _absolute_lexical(str(item["audit"]))
        for item in stdout_objects
        if item.get("passed") is True and isinstance(item.get("audit"), str)
    }
    if not {before_path, after_path}.issubset(stdout_audits):
        raise VectorBenchmarkMatrixError("matrix stdout omits frozen audit bindings")
    frozen_manifest_path = project_root / _FROZEN_MANIFEST_RELATIVE
    frozen_snapshot = _snapshot(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    if frozen_snapshot.sha256 != expected_frozen_manifest_sha256:
        raise VectorBenchmarkMatrixError("matrix frozen manifest SHA-256 mismatch")
    frozen_manifest = _load_object(
        frozen_manifest_path, label="frozen FSM manifest", cache=cache
    )
    before_audit = _validate_frozen_audit(
        before_path,
        expected_project_root=project_root,
        expected_manifest_path=frozen_manifest_path,
        manifest=frozen_manifest,
        manifest_snapshot=frozen_snapshot,
        cache=cache,
    )
    after_audit = _validate_frozen_audit(
        after_path,
        expected_project_root=project_root,
        expected_manifest_path=frozen_manifest_path,
        manifest=frozen_manifest,
        manifest_snapshot=frozen_snapshot,
        cache=cache,
    )
    for field in ("protected_file_count", "entries", "mismatches", "passed"):
        if before_audit.get(field) != after_audit.get(field):
            raise VectorBenchmarkMatrixError(
                f"matrix frozen audits differ for {field}"
            )

    raw_slots = matrix.get("slots")
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        raise VectorBenchmarkMatrixError("matrix slot list is missing")
    fresh_slots: list[dict[str, Any]] = []
    for stored in raw_slots:
        if not isinstance(stored, Mapping):
            raise VectorBenchmarkMatrixError("matrix contains a malformed slot")
        records = stored.get("records")
        benchmark_record = records.get("vector_benchmark") if isinstance(records, Mapping) else None
        worker_path = benchmark_record.get("path") if isinstance(benchmark_record, Mapping) else None
        if not isinstance(worker_path, str):
            raise VectorBenchmarkMatrixError("matrix slot omits its worker path")
        fresh = validate_vector_benchmark_slot(
            worker_path,
            expected_config_sha256=expected_config_sha256,
            expected_frozen_manifest_sha256=expected_frozen_manifest_sha256,
            expected_git_commit=expected_git_commit,
            expected_run_seed=expected_run_seed,
            expected_project_root=project_root,
            expected_config_records=expected_config_records,
            _snapshot_cache=cache,
        )
        if dict(stored) != fresh:
            raise VectorBenchmarkMatrixError("matrix slot differs from revalidated worker")
        fresh_slots.append(fresh)
    if len(fresh_slots) != 6:
        raise VectorBenchmarkMatrixError("matrix must contain exactly six slots")
    by_key = {(slot["num_envs"], slot["mode"]): slot for slot in fresh_slots}
    expected_keys = {
        (num_envs, mode) for num_envs in ENVIRONMENT_COUNTS for mode in MODES
    }
    if set(by_key) != expected_keys or len(by_key) != 6:
        raise VectorBenchmarkMatrixError("matrix slot key set is incomplete or duplicated")
    common_fields = (
        "git_commit",
        "config_sha256",
        "config_records",
        "frozen_manifest_sha256",
        "frozen_source_head",
        "project_root",
        "run_seed",
        "measured_ticks",
        "policy_decisions",
    )
    reference = by_key[(8, "zero")]
    for key, slot in by_key.items():
        if any(slot[field] != reference[field] for field in common_fields):
            raise VectorBenchmarkMatrixError(
                f"revalidated matrix slot {key} differs in common provenance"
            )
    for num_envs in ENVIRONMENT_COUNTS:
        if by_key[(num_envs, "zero")]["seed_rows"] != by_key[
            (num_envs, "bounded-smoke")
        ]["seed_rows"]:
            raise VectorBenchmarkMatrixError(
                f"revalidated zero/bounded seeds differ at num_envs={num_envs}"
            )
    seeds_32 = by_key[(32, "zero")]["seed_rows"]
    for num_envs in (8, 16):
        if by_key[(num_envs, "zero")]["seed_rows"] != seeds_32[:num_envs]:
            raise VectorBenchmarkMatrixError(
                "revalidated matrix seed rows are not a shared prefix"
            )
    dual_mode_pass = {
        num_envs: all(by_key[(num_envs, mode)]["passed"] for mode in MODES)
        for num_envs in ENVIRONMENT_COUNTS
    }
    candidates = [num_envs for num_envs, passed in dual_mode_pass.items() if passed]
    if not candidates:
        raise VectorBenchmarkMatrixError("matrix has no dual-mode passing capacity")
    selected_num_envs = max(candidates)
    if any(
        not dual_mode_pass[value]
        for value in ENVIRONMENT_COUNTS
        if value < selected_num_envs
    ):
        raise VectorBenchmarkMatrixError("matrix pass set is not a monotonic prefix")
    if selected_num_envs != expected_num_envs:
        raise VectorBenchmarkMatrixError(
            "training environment count differs from matrix selected_num_envs"
        )
    seed_rows = tuple(_strict_int(value, label="expected training seed") for value in expected_seed_rows)
    if tuple(by_key[(selected_num_envs, "zero")]["seed_rows"]) != seed_rows:
        raise VectorBenchmarkMatrixError("selected matrix seed rows differ from training")

    ordered = [
        by_key[(num_envs, mode)]
        for num_envs in ENVIRONMENT_COUNTS
        for mode in MODES
    ]
    source_records = [
        {
            "slot": slot["slot"],
            **slot["records"]["vector_benchmark"],
            "run_manifest_sha256": slot["records"]["run_manifest"]["sha256"],
            "stdout_sha256": slot["records"]["stdout"]["sha256"],
            "frozen_before_sha256": slot["records"]["frozen_hashes_before"]["sha256"],
            "frozen_after_sha256": slot["records"]["frozen_hashes_after"]["sha256"],
        }
        for slot in ordered
    ]
    selected_acceptance = {
        mode: {
            "path": by_key[(selected_num_envs, mode)]["records"]["vector_benchmark"]["path"],
            "sha256": by_key[(selected_num_envs, mode)]["records"]["vector_benchmark"]["sha256"],
        }
        for mode in MODES
    }
    selected_capacity_checks = {
        "oom_detected": False,
        "cross_environment_contamination_detected": False,
        "fsm_state_contamination_detected": False,
        "render_contamination_detected": False,
        "resource_evidence_by_mode": {
            mode: by_key[(selected_num_envs, mode)]["resource_evidence"]
            for mode in MODES
        },
    }
    expected_common = {field: reference[field] for field in common_fields}
    if (
        matrix.get("schema") != MATRIX_SCHEMA
        or matrix.get("status") != "VECTOR_BENCHMARK_MATRIX_ACCEPTED"
        or matrix.get("passed") is not True
        or matrix.get("required_environment_counts") != list(ENVIRONMENT_COUNTS)
        or matrix.get("required_modes") != list(MODES)
        or matrix.get("required_slot_count") != 6
        or matrix.get("validated_slot_count") != 6
        or matrix.get("selected_num_envs") != selected_num_envs
        or matrix.get("selection_rule") != "highest_monotonic_environment_count_where_both_modes_pass"
        or matrix.get("dual_mode_pass_by_environment_count")
        != {str(key): value for key, value in dual_mode_pass.items()}
        or matrix.get("common_provenance") != expected_common
        or matrix.get("selected_acceptance") != selected_acceptance
        or matrix.get("selected_capacity_checks") != selected_capacity_checks
        or matrix.get("source_artifacts") != source_records
        or matrix.get("source_artifact_set_sha256") != _canonical_sha256(source_records)
        or list(raw_slots) != ordered
    ):
        raise VectorBenchmarkMatrixError("matrix payload selection/provenance is invalid")
    invoked_workers = {
        str(_absolute_lexical(value if Path(value).is_absolute() else project_root / value))
        for value in benchmark_arguments
    }
    recorded_workers = {item["path"] for item in source_records}
    if invoked_workers != recorded_workers or len(invoked_workers) != 6:
        raise VectorBenchmarkMatrixError("matrix invocation and recorded workers differ")

    _revalidate_snapshots(cache, project_root=project_root)
    matrix_snapshot = _snapshot(
        matrix_path, label="vector benchmark matrix", cache=cache
    )
    return {
        "schema": "wlr50_clean.vector_benchmark_training_matrix_acceptance.v1",
        "path": str(matrix_path),
        "sha256": matrix_snapshot.sha256,
        "selected_num_envs": selected_num_envs,
        "run_seed": expected_run_seed,
        "config_sha256": expected_config_sha256,
        "git_commit": expected_git_commit,
        "frozen_manifest_sha256": expected_frozen_manifest_sha256,
        "frozen_source_head": frozen_manifest.get("source_head"),
        "selected_acceptance": {
            "zero": dict(selected_acceptance["zero"]),
            "bounded_nonzero": dict(selected_acceptance["bounded-smoke"]),
        },
        "selected_capacity_checks": selected_capacity_checks,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": _snapshot(
            manifest_path, label="matrix run manifest", cache=cache
        ).sha256,
        "stdout_sha256": stdout_snapshot.sha256,
        "stderr_sha256": stderr_snapshot.sha256,
        "committed_runtime_identity_before": matrix_runtime_before,
        "committed_runtime_identity_after": matrix_runtime_after,
        "passed": True,
    }


def _current_run_expectations(
    run_dir: Path, *, seed: int, num_envs: int
) -> dict[str, Any]:
    cache: _SnapshotCache = {}
    started = _load_object(
        run_dir / "run_manifest.started.json",
        label="matrix started run manifest",
        cache=cache,
    )
    identity = started.get("identity")
    configs = started.get("configs")
    project_root = _absolute_lexical(PROJECT_ROOT)
    expected_run_dir = project_root / _RUNS_RELATIVE / "vector_benchmark_matrix" / run_dir.name
    if (
        started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
        or started.get("immutable_run_directory") is not True
        or _absolute_lexical(str(started.get("project_root", ""))) != project_root
        or _absolute_lexical(str(started.get("run_dir", ""))) != run_dir
        or run_dir != expected_run_dir
        or started.get("run_kind") != "vector_benchmark_matrix"
        or started.get("entrypoint")
        != "wlr50_clean.ppo.vector_benchmark_matrix"
        or started.get("subcommand") != "aggregate"
        or not isinstance(identity, Mapping)
        or identity.get("training_stage") != "backend-benchmark-selection"
        or identity.get("seed") != seed
        or identity.get("environment_count") != num_envs
        or num_envs != 1
        or not isinstance(configs, Sequence)
        or isinstance(configs, (str, bytes))
    ):
        raise VectorBenchmarkMatrixError(
            "matrix command is not running in its managed immutable reservation"
        )
    frozen_manifest = (
        project_root
        / "artifacts"
        / "ppo_phase_v1_start"
        / "frozen_fsm_hashes.json"
    )
    _reject_links(run_dir, root=project_root, label="matrix managed run directory")
    frozen_snapshot = _snapshot(
        frozen_manifest, label="frozen FSM manifest", cache=cache
    )
    config_records = [dict(value) for value in configs if isinstance(value, Mapping)]
    if len(config_records) != len(configs):
        raise VectorBenchmarkMatrixError("matrix config records are malformed")
    return {
        "expected_config_sha256": str(identity.get("config_sha256", "")),
        "expected_frozen_manifest_sha256": frozen_snapshot.sha256,
        "expected_git_commit": str(identity.get("git_commit", "")),
        "expected_run_seed": seed,
        "expected_project_root": project_root,
        "expected_config_records": config_records,
        "_managed_started_path": run_dir / "run_manifest.started.json",
        "_managed_started_sha256": _snapshot(
            run_dir / "run_manifest.started.json",
            label="matrix started run manifest",
            cache=cache,
        ).sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate the immutable six-slot vector benchmark matrix"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--seed", type=int, required=True)
    aggregate.add_argument("--num-envs", type=int, required=True)
    aggregate.add_argument(
        "--benchmark", type=Path, action="append", required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run_dir = _absolute_lexical(args.run_dir)
        expectations = _current_run_expectations(
            run_dir, seed=args.seed, num_envs=args.num_envs
        )
        project_root = expectations["expected_project_root"]
        paths = tuple(
            _absolute_lexical(value if value.is_absolute() else project_root / value)
            for value in args.benchmark
        )
        payload = aggregate_vector_benchmark_matrix(
            paths,
            output_path=run_dir / OUTPUT_FILENAME,
            **expectations,
        )
        print(json.dumps(payload, separators=(",", ":"), allow_nan=False), flush=True)
        return 0
    except (VectorBenchmarkMatrixError, ArtifactError) as exc:
        print(f"VECTOR_BENCHMARK_MATRIX_ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENVIRONMENT_COUNTS",
    "MATRIX_SCHEMA",
    "MODES",
    "OUTPUT_FILENAME",
    "VectorBenchmarkMatrixError",
    "aggregate_vector_benchmark_matrix",
    "build_parser",
    "main",
    "validate_finalized_vector_benchmark_matrix",
    "validate_managed_run_directory",
    "validate_vector_benchmark_slot",
]
