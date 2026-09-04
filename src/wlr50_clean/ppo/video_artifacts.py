"""Publish auditable final videos from two accepted live viewport episodes.

The inputs to this module are *complete episode* captures produced by
``video_runtime.capture_live_policy_video``.  This module never selects a
phase, trims an action, changes playback speed, or joins episodes in time.
The comparison is a spatial composite whose shorter side is extended only by
cloning its final frame, as required for a real-time side-by-side comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.video_capture import (
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    find_ffmpeg,
    sha256_file,
    validate_mp4,
)
from wlr50_clean.infrastructure.scene_factory import CAMERA_EYE_M, CAMERA_TARGET_M

from .artifacts import (
    RUN_MANIFEST_SCHEMA,
    ArtifactError,
    RunIdentity,
    atomic_write_json,
    config_set_record,
    verify_checksum_manifest,
    write_checksum_manifest,
)


SOURCE_MANIFEST_NAME = "ppo_video_source_manifest.json"
SOURCE_VIDEO_NAME = "actual_viewport_video.mp4"
SOURCE_LEDGER_NAME = "viewport_frame_ledger.jsonl"
SOURCE_TRACE_NAME = "policy_trace.jsonl"

FSM_VIDEO_NAME = "fsm_baseline_clean.mp4"
PPO_VIDEO_NAME = "ppo_improved_checkpoint_clean.mp4"
COMPARISON_VIDEO_NAME = "fsm_vs_ppo_improved.mp4"
DIAGNOSTIC_VIDEO_NAME = "ppo_improved_diagnostic.mp4"
VIDEO_VALIDATION_NAME = "video_validation.json"
VIDEO_CHECKSUM_NAME = "video_checksums.sha256"
DIAGNOSTIC_ASS_NAME = "ppo_improved_diagnostic.ass"

STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
_FRAME_PERIOD_S = 1.0 / VIDEO_FPS
_SOURCE_SCHEMA = "wlr50_clean.ppo_video_source_episode.v1"
_PUBLICATION_SCHEMA = "wlr50_clean.ppo_final_videos.v1"
_LIVE_RESULT_SCHEMA = "wlr50_clean.live_command_result.v1"
_SOURCE_CAPTURE_SCHEMA = "wlr50_clean.ppo_video_source_capture_cli.v1"
_PUBLICATION_CLI_SCHEMA = "wlr50_clean.ppo_final_video_publication_cli.v1"
_RUNTIME_IDENTITY_SCHEMA = "wlr50_clean.committed_runtime_identity.v1"
_FROZEN_AUDIT_SCHEMA = "wlr50_clean.frozen_fsm_hash_audit.v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RUNS_ROOT_RELATIVE = Path("runs") / "ppo_phase_v1"
_FROZEN_MANIFEST_RELATIVE = (
    Path("artifacts") / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class PPOVideoArtifactError(RuntimeError):
    """A source or generated final video cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SourceEpisode:
    """Validated provenance and telemetry for one video source episode."""

    role: str
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    recorder_manifest_path: Path
    recorder_manifest: Mapping[str, Any]
    trial_manifest_path: Path
    trial_manifest: Mapping[str, Any]
    video_path: Path
    ledger_path: Path
    trace_path: Path
    trace: tuple[Mapping[str, Any], ...]
    source_identity: Mapping[str, Any]
    camera: Mapping[str, Any]
    video_validation: Mapping[str, Any]
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    managed_run_evidence: Mapping[str, Any]
    committed_runtime_identity: Mapping[str, Any]

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def frame_count(self) -> int:
        return int(self.video_validation["frame_count"])

    @property
    def duration_s(self) -> float:
        return float(self.video_validation["duration_s"])


@dataclass(frozen=True, slots=True)
class FinalVideoPublication:
    """Paths and verified evidence returned after immutable publication."""

    videos: Mapping[str, Path]
    validation_path: Path
    checksum_path: Path
    diagnostic_ass_path: Path
    checksum_verification: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ManagedRunValidation:
    evidence: Mapping[str, Any]
    runtime_identity: Mapping[str, Any]


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or (callable(is_junction) and is_junction())
        or (getattr(metadata, "st_file_attributes", 0) & 0x400)
    )


def _absolute(path: Path | str) -> Path:
    """Return an absolute lexical path without following links."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_no_reparse_components(path: Path | str, *, label: str) -> Path:
    selected = _absolute(path)
    for component in reversed((selected, *selected.parents)):
        try:
            redirected = _is_reparse_point(component)
        except OSError as exc:
            raise PPOVideoArtifactError(
                f"cannot inspect {label} path component: {component}"
            ) from exc
        if redirected:
            raise PPOVideoArtifactError(
                f"{label} contains a symbolic link, junction, or reparse point: {component}"
            )
    return selected


def _stable_file(
    path: Path | str,
    *,
    label: str,
    keep_bytes: bool = False,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    """Capture one regular file and derive size/hash from the same open handle."""

    selected = _require_no_reparse_components(path, label=label)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if keep_bytes else None
    size = 0
    try:
        with selected.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise PPOVideoArtifactError(f"{label} is not a regular file: {selected}")
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                if chunks is not None:
                    chunks.append(block)
            after = os.fstat(stream.fileno())
    except PPOVideoArtifactError:
        raise
    except OSError as exc:
        raise PPOVideoArtifactError(f"cannot read {label}: {selected}") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if size != before.st_size or any(
        getattr(before, field, None) != getattr(after, field, None)
        for field in stable_fields
    ):
        raise PPOVideoArtifactError(f"{label} changed while it was being captured")
    if not allow_empty and size == 0:
        raise PPOVideoArtifactError(f"{label} is empty: {selected}")
    _require_no_reparse_components(selected, label=label)
    return (
        {
            "path": str(selected),
            "bytes": size,
            "sha256": digest.hexdigest(),
        },
        None if chunks is None else b"".join(chunks),
    )


def _stable_json(
    path: Path | str, *, label: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    record, content = _stable_file(path, label=label, keep_bytes=True)
    assert content is not None
    try:
        value = json.loads(content.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PPOVideoArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise PPOVideoArtifactError(f"{label} must contain a JSON object")
    return record, value


def _strict_hash(value: Any, *, label: str, sha1: bool = False) -> str:
    text = str(value).lower() if isinstance(value, str) else ""
    pattern = _SHA1 if sha1 else _SHA256
    if pattern.fullmatch(text) is None:
        raise PPOVideoArtifactError(f"{label} is not a valid digest")
    return text


def _record_matches(
    declared: Any,
    captured: Mapping[str, Any],
    *,
    expected_relative_path: str,
    label: str,
) -> None:
    if (
        not isinstance(declared, Mapping)
        or declared.get("path") != expected_relative_path
        or isinstance(declared.get("bytes"), bool)
        or declared.get("bytes") != captured.get("bytes")
        or declared.get("sha256") != captured.get("sha256")
    ):
        raise PPOVideoArtifactError(f"{label} record is stale or malformed")


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise PPOVideoArtifactError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PPOVideoArtifactError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise PPOVideoArtifactError(f"{label} timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _validate_runtime_identity_payload(
    payload: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    files = payload.get("files")
    if (
        set(payload)
        != {
            "schema",
            "git_commit",
            "file_count",
            "content_sha256",
            "aggregate_sha256",
            "files",
        }
        or payload.get("schema") != _RUNTIME_IDENTITY_SCHEMA
        or not isinstance(files, list)
        or not files
        or isinstance(payload.get("file_count"), bool)
        or payload.get("file_count") != len(files)
    ):
        raise PPOVideoArtifactError(f"{label} has an invalid header")
    git_commit = _strict_hash(payload.get("git_commit"), label=f"{label} Git commit", sha1=True)
    aggregate_sha = _strict_hash(
        payload.get("aggregate_sha256"), label=f"{label} aggregate SHA-256"
    )
    content_sha = _strict_hash(
        payload.get("content_sha256"), label=f"{label} content SHA-256"
    )
    ordered_fields = (
        "path",
        "bytes",
        "sha256",
        "creation_time_utc_ticks",
        "last_write_time_utc_ticks",
    )
    normalized: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, row in enumerate(files):
        if (
            not isinstance(row, Mapping)
            or set(row) != set(ordered_fields)
            or not isinstance(row.get("path"), str)
            or not row.get("path")
            or Path(str(row["path"])).is_absolute()
            or ".." in Path(str(row["path"])).parts
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
            or any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), int)
                or row[field] <= 0
                for field in (
                    "creation_time_utc_ticks",
                    "last_write_time_utc_ticks",
                )
            )
        ):
            raise PPOVideoArtifactError(f"{label} file row {index} is malformed")
        path = str(row["path"]).replace("\\", "/")
        if path != row["path"]:
            raise PPOVideoArtifactError(f"{label} file row {index} path is not canonical")
        _strict_hash(row.get("sha256"), label=f"{label} file row {index} SHA-256")
        paths.append(path)
        normalized.append({field: row[field] for field in ordered_fields})
    if tuple(paths) != tuple(sorted(set(paths))):
        raise PPOVideoArtifactError(f"{label} file paths are not sorted and unique")
    encoded = json.dumps(
        normalized, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != aggregate_sha:
        raise PPOVideoArtifactError(f"{label} aggregate SHA-256 is invalid")
    content_rows = [
        {key: row[key] for key in ("path", "bytes", "sha256")}
        for row in normalized
    ]
    content = json.dumps(
        content_rows, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(content).hexdigest() != content_sha:
        raise PPOVideoArtifactError(f"{label} content SHA-256 is invalid")
    return {
        **dict(payload),
        "git_commit": git_commit,
        "aggregate_sha256": aggregate_sha,
        "content_sha256": content_sha,
    }


def _validate_current_runtime_identity(identity: Mapping[str, Any]) -> None:
    """Late import keeps video capture independent from reporting at import time."""

    from .evaluation_artifacts import (
        EvaluationArtifactError,
        _validate_current_committed_runtime_identity,
    )

    try:
        _validate_current_committed_runtime_identity(identity)
    except EvaluationArtifactError as exc:
        raise PPOVideoArtifactError(
            f"managed video run runtime no longer matches committed HEAD: {exc}"
        ) from exc


def _validate_config_records(
    records: Any, *, expected_sha256: Any, label: str
) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(row, Mapping) for row in records)
    ):
        raise PPOVideoArtifactError(f"{label} config records are missing")
    names = [str(row.get("path", "")).replace("\\", "/") for row in records]
    if (
        any(not name or Path(name).is_absolute() or ".." in Path(name).parts for name in names)
        or names != sorted(set(names))
    ):
        raise PPOVideoArtifactError(f"{label} config paths are unsafe or duplicated")
    config_paths: list[Path] = []
    for name in names:
        path = _require_no_reparse_components(_PROJECT_ROOT / Path(name), label=f"{label} config")
        try:
            path.relative_to(_PROJECT_ROOT)
        except ValueError as exc:
            raise PPOVideoArtifactError(f"{label} config escapes the project") from exc
        config_paths.append(path)
    try:
        actual_sha, actual_records = config_set_record(
            config_paths, project_root=_PROJECT_ROOT
        )
    except (ArtifactError, OSError) as exc:
        raise PPOVideoArtifactError(f"cannot validate {label} config set") from exc
    if actual_sha != expected_sha256 or actual_records != [dict(row) for row in records]:
        raise PPOVideoArtifactError(f"{label} config bytes or aggregate hash changed")
    return tuple(actual_records)


def _validate_frozen_audit(
    audit: Mapping[str, Any],
    *,
    label: str,
    frozen_manifest_path: Path,
    frozen_manifest_record: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any],
) -> tuple[datetime, Mapping[str, Any]]:
    protected = frozen_manifest.get("protected_files")
    entries = audit.get("entries")
    if (
        frozen_manifest.get("algorithm") != "sha256"
        or not isinstance(protected, Mapping)
        or not protected
        or audit.get("schema") != _FROZEN_AUDIT_SCHEMA
        or audit.get("project_root") != str(_PROJECT_ROOT)
        or _absolute(str(audit.get("frozen_manifest", ""))) != frozen_manifest_path
        or audit.get("frozen_manifest_sha256") != frozen_manifest_record["sha256"]
        or audit.get("source_head") != frozen_manifest.get("source_head")
        or audit.get("passed") is not True
        or audit.get("mismatches") != []
        or isinstance(audit.get("protected_file_count"), bool)
        or audit.get("protected_file_count") != len(protected)
        or not isinstance(entries, list)
        or len(entries) != len(protected)
    ):
        raise PPOVideoArtifactError(f"{label} is incomplete or stale")
    checked = _parse_utc(audit.get("checked_at_utc"), label=label)
    expected_rows: list[dict[str, Any]] = []
    for raw_name, raw_hash in sorted(protected.items(), key=lambda item: str(item[0])):
        name = str(raw_name).replace("\\", "/")
        expected = _strict_hash(raw_hash, label=f"{label} frozen hash {name}")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name != relative.as_posix():
            raise PPOVideoArtifactError(f"{label} contains an unsafe frozen path")
        source = _require_no_reparse_components(
            _PROJECT_ROOT / relative, label=f"{label} frozen file {name}"
        )
        record, _ = _stable_file(source, label=f"{label} frozen file {name}")
        expected_rows.append(
            {
                "path": name,
                "expected_sha256": expected,
                "actual_sha256": record["sha256"],
                "exists": True,
                "valid": record["sha256"] == expected,
            }
        )
    if entries != expected_rows or any(row["valid"] is not True for row in expected_rows):
        raise PPOVideoArtifactError(f"{label} does not match current frozen bytes")
    invariant = {
        key: value for key, value in audit.items() if key != "checked_at_utc"
    }
    return checked, invariant


def _validate_frozen_pair(
    run_dir: Path,
    *,
    artifacts: Mapping[str, Any] | None,
    label: str,
    require_after: bool,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]:
    frozen_manifest_path = _require_no_reparse_components(
        _PROJECT_ROOT / _FROZEN_MANIFEST_RELATIVE,
        label="frozen FSM manifest",
    )
    frozen_manifest_record, frozen_manifest = _stable_json(
        frozen_manifest_path, label="frozen FSM manifest"
    )
    names = ["frozen_hashes.before.json"]
    if require_after:
        names.append("frozen_hashes.after.json")
    records: list[dict[str, Any]] = []
    validations: list[tuple[datetime, Mapping[str, Any]]] = []
    for name in names:
        record, payload = _stable_json(run_dir / name, label=f"{label} {name}")
        if artifacts is not None:
            _record_matches(
                artifacts.get(name),
                record,
                expected_relative_path=name,
                label=f"{label} {name}",
            )
        records.append(record)
        validations.append(
            _validate_frozen_audit(
                payload,
                label=f"{label} {name}",
                frozen_manifest_path=frozen_manifest_path,
                frozen_manifest_record=frozen_manifest_record,
                frozen_manifest=frozen_manifest,
            )
        )
    if require_after and (
        validations[1][0] < validations[0][0]
        or dict(validations[1][1]) != dict(validations[0][1])
    ):
        raise PPOVideoArtifactError(f"{label} frozen before/after evidence differs")
    return frozen_manifest_record, tuple(records)


def _argument_value(arguments: Sequence[Any], flag: str, *, label: str) -> str:
    values: list[str] = []
    items = [str(value) for value in arguments]
    for index, item in enumerate(items):
        if item == flag:
            if index + 1 >= len(items):
                raise PPOVideoArtifactError(f"{label} invocation omits a value for {flag}")
            values.append(items[index + 1])
        elif item.startswith(flag + "="):
            values.append(item.split("=", 1)[1])
    if len(values) != 1:
        raise PPOVideoArtifactError(f"{label} invocation must contain {flag} exactly once")
    return values[0]


def _require_flag(arguments: Sequence[Any], flag: str, *, label: str) -> None:
    if sum(str(value) == flag for value in arguments) != 1:
        raise PPOVideoArtifactError(f"{label} invocation must contain {flag} exactly once")


def _validate_started_manifest(
    run_dir: Path,
    started: Mapping[str, Any],
    *,
    run_kind: str,
    training_stage: str,
    subcommand: str,
    label: str,
) -> tuple[Mapping[str, Any], tuple[str, ...], tuple[Path, ...]]:
    identity = started.get("identity")
    configs = started.get("configs")
    arguments = started.get("invocation_arguments")
    if (
        started.get("schema") != RUN_MANIFEST_SCHEMA
        or started.get("lifecycle") != "STARTED"
        or started.get("immutable_run_directory") is not True
        or started.get("run_id") != run_dir.name
        or _absolute(str(started.get("run_dir", ""))) != run_dir
        or _absolute(str(started.get("project_root", ""))) != _PROJECT_ROOT
        or started.get("run_kind") != run_kind
        or started.get("entrypoint") != "wlr50_clean.ppo.cli"
        or started.get("subcommand") != subcommand
        or not isinstance(identity, Mapping)
        or not isinstance(arguments, list)
        or any(not isinstance(value, str) for value in arguments)
    ):
        raise PPOVideoArtifactError(f"{label} started manifest is invalid")
    try:
        run_identity = RunIdentity(
            timestamp_utc=str(identity["timestamp_utc"]),
            git_commit=str(identity["git_commit"]),
            config_sha256=str(identity["config_sha256"]),
            seed=identity["seed"],
            environment_count=identity["environment_count"],
            training_stage=str(identity["training_stage"]),
        )
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{label} run identity is malformed") from exc
    if (
        set(identity)
        != {
            "timestamp_utc",
            "git_commit",
            "config_sha256",
            "seed",
            "environment_count",
            "training_stage",
        }
        or isinstance(identity.get("seed"), bool)
        or isinstance(identity.get("environment_count"), bool)
        or run_identity.seed != 4001
        or run_identity.environment_count != 1
        or run_identity.training_stage != training_stage
        or run_identity.run_id != run_dir.name
    ):
        raise PPOVideoArtifactError(f"{label} run identity differs from the locked contract")
    config_records = _validate_config_records(
        configs,
        expected_sha256=run_identity.config_sha256,
        label=label,
    )
    _argument_value(arguments, "--run-dir", label=label)
    if _argument_value(arguments, "--run-dir", label=label) != "<reserved-immutable-run-dir>":
        raise PPOVideoArtifactError(f"{label} invocation has an invalid run reservation token")
    if _argument_value(arguments, "--seed", label=label) != "4001":
        raise PPOVideoArtifactError(f"{label} invocation has the wrong seed")
    if _argument_value(arguments, "--num-envs", label=label) != "1":
        raise PPOVideoArtifactError(f"{label} invocation has the wrong environment count")
    if _argument_value(arguments, "--episode-count", label=label) != "1":
        raise PPOVideoArtifactError(f"{label} invocation has the wrong episode count")
    _require_flag(arguments, "--deterministic", label=label)
    return (
        dict(identity),
        tuple(arguments),
        tuple(_PROJECT_ROOT / Path(str(row["path"])) for row in config_records),
    )


def _validate_runtime_pair(
    run_dir: Path,
    *,
    artifacts: Mapping[str, Any] | None,
    expected_git_commit: str,
    label: str,
    require_after: bool,
) -> tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]:
    names = ["committed_runtime_identity.before.json"]
    if require_after:
        names.append("committed_runtime_identity.after.json")
    records: list[dict[str, Any]] = []
    payloads: list[Mapping[str, Any]] = []
    for name in names:
        record, payload = _stable_json(run_dir / name, label=f"{label} {name}")
        if artifacts is not None:
            _record_matches(
                artifacts.get(name),
                record,
                expected_relative_path=name,
                label=f"{label} {name}",
            )
        records.append(record)
        payloads.append(
            _validate_runtime_identity_payload(payload, label=f"{label} {name}")
        )
    if (
        payloads[0].get("git_commit") != expected_git_commit
        or (require_after and dict(payloads[0]) != dict(payloads[1]))
    ):
        raise PPOVideoArtifactError(f"{label} committed runtime before/after differs")
    _validate_current_runtime_identity(payloads[0])
    return payloads[0], tuple(records)


def _validate_final_run_inventory(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Mapping[str, dict[str, Any]], dict[str, Any], bytes]:
    artifacts = run_manifest.get("artifacts")
    logs = run_manifest.get("logs")
    if not isinstance(artifacts, Mapping) or not isinstance(logs, Mapping):
        raise PPOVideoArtifactError(f"{label} run manifest omits artifacts or logs")
    excluded = {
        "run_manifest.started.json",
        "run_manifest.json",
        "stdout.log",
        "stderr.log",
    }
    inventory: dict[str, Path] = {}
    for source in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        _require_no_reparse_components(source, label=f"{label} artifact tree")
        if source.is_file():
            relative = source.relative_to(run_dir).as_posix()
            if relative not in excluded:
                inventory[relative] = source
    if set(inventory) != set(artifacts):
        raise PPOVideoArtifactError(f"{label} run artifact inventory is not exact")
    captured_artifacts: dict[str, dict[str, Any]] = {}
    for relative, source in inventory.items():
        record, _ = _stable_file(
            source, label=f"{label} artifact {relative}", allow_empty=True
        )
        _record_matches(
            artifacts.get(relative),
            record,
            expected_relative_path=relative,
            label=f"{label} artifact {relative}",
        )
        captured_artifacts[relative] = record

    log_records: dict[str, dict[str, Any]] = {}
    stdout_content = b""
    for name in ("stdout.log", "stderr.log"):
        record, content = _stable_file(
            run_dir / name,
            label=f"{label} {name}",
            keep_bytes=name == "stdout.log",
            allow_empty=True,
        )
        _record_matches(
            logs.get(name),
            record,
            expected_relative_path=name,
            label=f"{label} {name}",
        )
        log_records[name] = record
        if content is not None:
            stdout_content = content
    return captured_artifacts, log_records, stdout_content


def _stdout_json_objects(content: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = content.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise PPOVideoArtifactError(f"{label} stdout is not valid UTF-8") from exc
    objects: list[Mapping[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            objects.append(value)
    return tuple(objects)


def _validate_source_invocation(
    invocation: Sequence[str],
    *,
    role: str,
    source_manifest: Mapping[str, Any],
    config_paths: Sequence[Path],
    label: str,
) -> None:
    common_values = {
        "--training-config",
        "--interface-config",
        "--episode-count",
        "--capture-fps",
        "--maximum-duration-s",
        "--video-source-role",
        "--run-dir",
        "--seed",
        "--num-envs",
    }
    role_values = (
        {"--residual-mode"}
        if role == "fsm"
        else {"--checkpoint", "--checkpoint-manifest"}
    )
    switches = {"--deterministic", "--no-headless"}
    allowed_values = common_values | role_values
    observed: list[str] = []
    index = 0
    while index < len(invocation):
        item = invocation[index]
        if item in switches:
            observed.append(item)
            index += 1
            continue
        if item in allowed_values:
            if index + 1 >= len(invocation):
                raise PPOVideoArtifactError(
                    f"{label} invocation omits a value for {item}"
                )
            observed.append(item)
            index += 2
            continue
        matching = next(
            (flag for flag in allowed_values if item.startswith(flag + "=")),
            None,
        )
        if matching is None:
            raise PPOVideoArtifactError(
                f"{label} invocation contains an unsupported argument: {item}"
            )
        observed.append(matching)
        index += 1
    if any(observed.count(flag) != 1 for flag in allowed_values | switches):
        raise PPOVideoArtifactError(
            f"{label} invocation does not contain the exact source-capture argument set"
        )
    declared_configs = set(config_paths)
    for flag in ("--training-config", "--interface-config"):
        config = _invocation_project_path(
            _argument_value(invocation, flag, label=label),
            label=f"{label} {flag}",
        )
        if config not in declared_configs:
            raise PPOVideoArtifactError(
                f"{label} invocation {flag} is absent from the hashed config set"
            )
    if (
        _argument_value(invocation, "--video-source-role", label=label) != role
        or not math.isclose(
            _finite(
                _argument_value(invocation, "--capture-fps", label=label),
                label=f"{label} capture FPS",
            ),
            VIDEO_FPS,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            _finite(
                _argument_value(invocation, "--maximum-duration-s", label=label),
                label=f"{label} maximum duration",
            ),
            200.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise PPOVideoArtifactError(
            f"{label} invocation differs from the locked source-capture contract"
        )
    if role == "fsm":
        if _argument_value(invocation, "--residual-mode", label=label) != "zero":
            raise PPOVideoArtifactError("FSM video source invocation is not zero residual")
    else:
        checkpoint = _invocation_project_path(
            _argument_value(invocation, "--checkpoint", label=label),
            label=f"{label} checkpoint",
        )
        manifest = _invocation_project_path(
            _argument_value(invocation, "--checkpoint-manifest", label=label),
            label=f"{label} checkpoint manifest",
        )
        if (
            checkpoint
            != _absolute(str(source_manifest.get("source_checkpoint", "")))
            or manifest
            != _absolute(str(source_manifest.get("source_checkpoint_manifest", "")))
        ):
            raise PPOVideoArtifactError(
                "PPO video source invocation differs from its loaded checkpoint evidence"
            )


def _validate_source_managed_run(
    source_directory: Path | str, *, role: str
) -> _ManagedRunValidation:
    if role not in {"fsm", "ppo"}:
        raise PPOVideoArtifactError(f"unknown video source role: {role!r}")
    source_root = _require_no_reparse_components(
        source_directory, label=f"{role} video source directory"
    )
    runs_root = _PROJECT_ROOT / _RUNS_ROOT_RELATIVE
    try:
        relative = source_root.relative_to(runs_root)
    except ValueError as exc:
        raise PPOVideoArtifactError(
            f"{role} video source is outside the canonical managed runs root"
        ) from exc
    expected_kind = f"video-source-{role}"
    if (
        len(relative.parts) != 3
        or relative.parts[0] != expected_kind
        or relative.parts[2] != "video_source"
    ):
        raise PPOVideoArtifactError(
            f"{role} video source is not a canonical {expected_kind} run artifact"
        )
    run_dir = source_root.parent
    if not source_root.is_dir() or not run_dir.is_dir():
        raise PPOVideoArtifactError(f"{role} managed video source directory is missing")
    label = f"{role} managed video source run"
    started_record, started = _stable_json(
        run_dir / "run_manifest.started.json", label=f"{label} started manifest"
    )
    identity, invocation, config_paths = _validate_started_manifest(
        run_dir,
        started,
        run_kind=expected_kind,
        training_stage=f"video-source-{role}-fresh-process",
        subcommand="capture-video-source",
        label=label,
    )
    if (
        _argument_value(invocation, "--video-source-role", label=label) != role
        or _argument_value(invocation, "--capture-fps", label=label) != "15"
        or _argument_value(invocation, "--maximum-duration-s", label=label) != "200"
    ):
        raise PPOVideoArtifactError(f"{label} invocation differs from the capture contract")
    _require_flag(invocation, "--no-headless", label=label)
    if "--headless" in invocation:
        raise PPOVideoArtifactError(f"{label} invocation requested headless capture")

    run_record, final = _stable_json(
        run_dir / "run_manifest.json", label=f"{label} final manifest"
    )
    completed = _parse_utc(final.get("completed_at_utc"), label=f"{label} completion")
    started_at = _parse_utc(identity.get("timestamp_utc"), label=f"{label} start")
    if (
        final.get("schema") != RUN_MANIFEST_SCHEMA
        or final.get("lifecycle") != "SUCCEEDED"
        or final.get("exit_code") != 0
        or completed <= started_at
        or any(
            final.get(key) != value
            for key, value in started.items()
            if key != "lifecycle"
        )
    ):
        raise PPOVideoArtifactError(f"{label} did not finalize successfully")
    _record_matches(
        final.get("started_manifest"),
        started_record,
        expected_relative_path="run_manifest.started.json",
        label=f"{label} started manifest",
    )
    captured_artifacts, log_records, stdout_content = _validate_final_run_inventory(
        run_dir, final, label=label
    )
    runtime_identity, runtime_records = _validate_runtime_pair(
        run_dir,
        artifacts=final["artifacts"],
        expected_git_commit=str(identity["git_commit"]),
        label=label,
        require_after=True,
    )
    frozen_manifest_record, frozen_records = _validate_frozen_pair(
        run_dir,
        artifacts=final["artifacts"],
        label=label,
        require_after=True,
    )

    live_record, live_result = _stable_json(
        run_dir / "live_command_result.json", label=f"{label} live command result"
    )
    _record_matches(
        final["artifacts"].get("live_command_result.json"),
        live_record,
        expected_relative_path="live_command_result.json",
        label=f"{label} live command result",
    )
    if (
        set(live_result) != {"schema", "command", "exit_code"}
        or live_result.get("schema") != _LIVE_RESULT_SCHEMA
        or live_result.get("command") != "capture-video-source"
        or live_result.get("exit_code") != 0
    ):
        raise PPOVideoArtifactError(f"{label} live command result is invalid")

    capture_record, capture_result = _stable_json(
        run_dir / "video_source_capture.json", label=f"{label} capture result"
    )
    _record_matches(
        final["artifacts"].get("video_source_capture.json"),
        capture_record,
        expected_relative_path="video_source_capture.json",
        label=f"{label} capture result",
    )
    source_manifest_record, source_manifest = _stable_json(
        source_root / SOURCE_MANIFEST_NAME, label=f"{label} source manifest"
    )
    _validate_source_invocation(
        invocation,
        role=role,
        source_manifest=source_manifest,
        config_paths=config_paths,
        label=label,
    )
    expected_source_manifest_relative = f"video_source/{SOURCE_MANIFEST_NAME}"
    _record_matches(
        final["artifacts"].get(expected_source_manifest_relative),
        source_manifest_record,
        expected_relative_path=expected_source_manifest_relative,
        label=f"{label} source manifest",
    )
    source_video = source_root / SOURCE_VIDEO_NAME
    if (
        capture_result.get("schema") != _SOURCE_CAPTURE_SCHEMA
        or capture_result.get("video_source_role") != role
        or capture_result.get("fresh_process_single_episode") is not True
        or capture_result.get("seed") != 4001
        or capture_result.get("headless") is not False
        or capture_result.get("active_viewport_configured") is not True
        or _absolute(str(capture_result.get("source_directory", ""))) != source_root
        or _absolute(str(capture_result.get("source_manifest", "")))
        != source_root / SOURCE_MANIFEST_NAME
        or _absolute(str(capture_result.get("source_video", ""))) != source_video
        or capture_result.get("capture_process_id")
        != source_manifest.get("capture_process_id")
        or capture_result.get("capture_process_instance_id")
        != source_manifest.get("capture_process_instance_id")
        or capture_result.get("checkpoint_load_provenance")
        != source_manifest.get("checkpoint_load_provenance")
        or capture_result.get("checkpoint_runtime_capture_verified") is not True
    ):
        raise PPOVideoArtifactError(f"{label} capture result and source manifest differ")
    checkpoint_capture = capture_result.get("checkpoint_runtime_capture")
    if role == "fsm":
        if checkpoint_capture is not None:
            raise PPOVideoArtifactError(
                "FSM video source unexpectedly reports a checkpoint runtime capture"
            )
    elif (
        not isinstance(checkpoint_capture, Mapping)
        or checkpoint_capture.get("schema")
        != "wlr50_clean.checkpoint_runtime_capture.v1"
        or checkpoint_capture.get("source_checkpoint_path")
        != source_manifest.get("source_checkpoint")
        or checkpoint_capture.get("source_checkpoint_sha256")
        != source_manifest.get("source_checkpoint_sha256")
        or checkpoint_capture.get("source_manifest_path")
        != source_manifest.get("source_checkpoint_manifest")
        or checkpoint_capture.get("source_manifest_sha256")
        != source_manifest.get("source_checkpoint_manifest_sha256")
        or checkpoint_capture.get("private_copy_exclusive") is not True
        or checkpoint_capture.get("runner_loads_private_copy_only") is not True
    ):
        raise PPOVideoArtifactError(
            "PPO video source checkpoint runtime capture is incomplete"
        )

    stdout_objects = _stdout_json_objects(stdout_content, label=label)
    capture_rows = [
        row for row in stdout_objects if row.get("schema") == _SOURCE_CAPTURE_SCHEMA
    ]
    if len(capture_rows) != 1 or dict(capture_rows[0]) != dict(capture_result):
        raise PPOVideoArtifactError(f"{label} stdout does not bind the exact capture result")
    audit_rows = {
        _absolute(str(row.get("audit", "")))
        for row in stdout_objects
        if row.get("passed") is True and row.get("audit")
    }
    expected_audits = {
        run_dir / "frozen_hashes.before.json",
        run_dir / "frozen_hashes.after.json",
    }
    if not expected_audits.issubset(audit_rows):
        raise PPOVideoArtifactError(f"{label} stdout omits frozen before/after results")

    evidence = {
        "run_directory": str(run_dir),
        "run_kind": expected_kind,
        "run_manifest": run_record,
        "started_manifest": started_record,
        "live_command_result": live_record,
        "capture_result": capture_record,
        "stdout": log_records["stdout.log"],
        "stderr": log_records["stderr.log"],
        "committed_runtime_identity_before": runtime_records[0],
        "committed_runtime_identity_after": runtime_records[1],
        "frozen_hashes_before": frozen_records[0],
        "frozen_hashes_after": frozen_records[1],
        "frozen_manifest": frozen_manifest_record,
        "git_commit": runtime_identity["git_commit"],
        "committed_runtime_content_sha256": runtime_identity["content_sha256"],
        "artifact_count": len(captured_artifacts),
    }
    return _ManagedRunValidation(
        evidence=evidence,
        runtime_identity=runtime_identity,
    )


def verify_video_source_managed_run(
    source_directory: Path | str, *, role: str
) -> Mapping[str, Any]:
    """Verify that a video source came from the canonical finalized live wrapper."""

    return dict(_validate_source_managed_run(source_directory, role=role).evidence)


def _invocation_project_path(value: str, *, label: str) -> Path:
    raw = Path(value)
    selected = raw if raw.is_absolute() else _PROJECT_ROOT / raw
    return _require_no_reparse_components(selected, label=label)


def _validate_publication_invocation(
    invocation: Sequence[str],
    *,
    fsm_source_dir: Path,
    ppo_source_dir: Path,
    output_root: Path,
    config_paths: Sequence[Path],
    label: str,
) -> None:
    value_flags = {
        "--training-config",
        "--interface-config",
        "--episode-count",
        "--fsm-video-source-dir",
        "--ppo-video-source-dir",
        "--output-root",
        "--run-dir",
        "--seed",
        "--num-envs",
    }
    optional_value_flags = {"--ffmpeg"}
    switch_flags = {"--deterministic"}
    index = 0
    observed: list[str] = []
    while index < len(invocation):
        item = invocation[index]
        if item in switch_flags:
            observed.append(item)
            index += 1
            continue
        if item in value_flags or item in optional_value_flags:
            if index + 1 >= len(invocation):
                raise PPOVideoArtifactError(
                    f"{label} invocation omits a value for {item}"
                )
            observed.append(item)
            index += 2
            continue
        matching = next(
            (
                flag
                for flag in (*value_flags, *optional_value_flags)
                if item.startswith(flag + "=")
            ),
            None,
        )
        if matching is None:
            raise PPOVideoArtifactError(
                f"{label} invocation contains an unsupported argument: {item}"
            )
        observed.append(matching)
        index += 1
    if (
        set(observed) - value_flags - optional_value_flags - switch_flags
        or any(observed.count(flag) != 1 for flag in value_flags | switch_flags)
        or any(observed.count(flag) > 1 for flag in optional_value_flags)
    ):
        raise PPOVideoArtifactError(
            f"{label} invocation does not contain the exact publication argument set"
        )
    declared_configs = set(config_paths)
    for flag in ("--training-config", "--interface-config"):
        config = _invocation_project_path(
            _argument_value(invocation, flag, label=label),
            label=f"{label} {flag}",
        )
        if config not in declared_configs:
            raise PPOVideoArtifactError(
                f"{label} invocation {flag} is absent from the hashed config set"
            )
    expected_paths = {
        "--fsm-video-source-dir": fsm_source_dir,
        "--ppo-video-source-dir": ppo_source_dir,
        "--output-root": output_root,
    }
    for flag, expected in expected_paths.items():
        actual = _invocation_project_path(
            _argument_value(invocation, flag, label=label),
            label=f"{label} {flag}",
        )
        if actual != expected:
            raise PPOVideoArtifactError(
                f"{label} invocation {flag} differs from the published artifact"
            )


def _publication_reservation_evidence(
    publication_run_dir: Path | str,
    *,
    fsm_source_dir: Path,
    ppo_source_dir: Path,
    output_root: Path,
    expected_runtime_identity: Mapping[str, Any],
) -> _ManagedRunValidation:
    run_dir = _require_no_reparse_components(
        publication_run_dir, label="video publication run directory"
    )
    runs_root = _PROJECT_ROOT / _RUNS_ROOT_RELATIVE
    try:
        relative = run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise PPOVideoArtifactError(
            "video publication run is outside the canonical managed runs root"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[0] != "video-publication":
        raise PPOVideoArtifactError(
            "video publication run is not a canonical video-publication reservation"
        )
    if not run_dir.is_dir():
        raise PPOVideoArtifactError("video publication run directory is missing")
    if (run_dir / "run_manifest.json").exists():
        raise PPOVideoArtifactError(
            "video publication run was already finalized before publication"
        )
    label = "video publication managed run"
    started_record, started = _stable_json(
        run_dir / "run_manifest.started.json", label=f"{label} started manifest"
    )
    identity, invocation, config_paths = _validate_started_manifest(
        run_dir,
        started,
        run_kind="video-publication",
        training_stage="video-publication-offline",
        subcommand="publish-videos",
        label=label,
    )
    _validate_publication_invocation(
        invocation,
        fsm_source_dir=fsm_source_dir,
        ppo_source_dir=ppo_source_dir,
        output_root=output_root,
        config_paths=config_paths,
        label=label,
    )
    runtime_identity, runtime_records = _validate_runtime_pair(
        run_dir,
        artifacts=None,
        expected_git_commit=str(identity["git_commit"]),
        label=label,
        require_after=False,
    )
    if dict(runtime_identity) != dict(expected_runtime_identity):
        raise PPOVideoArtifactError(
            "video publication runtime differs from its source capture runtime"
        )
    frozen_manifest_record, frozen_records = _validate_frozen_pair(
        run_dir,
        artifacts=None,
        label=label,
        require_after=False,
    )
    allowed = {
        "run_manifest.started.json",
        "committed_runtime_identity.before.json",
        "frozen_hashes.before.json",
        "stdout.log",
        "stderr.log",
    }
    present = {
        source.relative_to(run_dir).as_posix()
        for source in run_dir.rglob("*")
        if source.is_file()
    }
    if not {
        "run_manifest.started.json",
        "committed_runtime_identity.before.json",
        "frozen_hashes.before.json",
    }.issubset(present) or not present.issubset(allowed):
        raise PPOVideoArtifactError(
            "video publication reservation contains unexpected artifacts"
        )
    evidence = {
        "run_directory": str(run_dir),
        "run_kind": "video-publication",
        "started_manifest": started_record,
        "committed_runtime_identity_before": runtime_records[0],
        "frozen_hashes_before": frozen_records[0],
        "frozen_manifest": frozen_manifest_record,
        "git_commit": runtime_identity["git_commit"],
        "committed_runtime_content_sha256": runtime_identity["content_sha256"],
        "reservation_validated_before_encoding": True,
    }
    return _ManagedRunValidation(
        evidence=evidence,
        runtime_identity=runtime_identity,
    )


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    _, value = _stable_json(path, label=label)
    return value


def _inside(root: Path, value: Any, *, label: str) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = _require_no_reparse_components(candidate, label=label)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PPOVideoArtifactError(f"{label} points outside its source episode: {candidate}") from exc
    return candidate


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise PPOVideoArtifactError(f"{label} is not finite")
    return result


def _read_jsonl(path: Path, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise PPOVideoArtifactError(f"{label} is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PPOVideoArtifactError(
                f"{label} row {line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise PPOVideoArtifactError(f"{label} row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise PPOVideoArtifactError(f"{label} is empty: {path}")
    return tuple(rows)


def _validate_ledger(path: Path, recorder: Mapping[str, Any]) -> int:
    rows = _read_jsonl(path, label="viewport frame ledger")
    expected_indices = list(range(len(rows)))
    indices: list[int] = []
    steps: list[int] = []
    times: list[float] = []
    for index, row in enumerate(rows):
        try:
            indices.append(int(row["encoded_frame_index"]))
            steps.append(int(row["sim_step"]))
            times.append(_finite(row["sim_time_s"], label=f"ledger row {index} sim_time_s"))
        except (KeyError, TypeError, ValueError) as exc:
            raise PPOVideoArtifactError(f"viewport ledger row {index} is incomplete") from exc
    if indices != expected_indices:
        raise PPOVideoArtifactError("viewport ledger frame indices are not contiguous from zero")
    if not steps or steps[0] != 0 or any(
        right - left != 8 for left, right in zip(steps, steps[1:])
    ):
        raise PPOVideoArtifactError(
            "viewport ledger is not the exact global 120 Hz to 15 Hz cadence"
        )
    if any(
        abs(actual - expected_step / 120.0) > 1.0e-9
        for actual, expected_step in zip(times, steps, strict=True)
    ):
        raise PPOVideoArtifactError(
            "viewport ledger timestamps do not match their physical simulation ticks"
        )
    if int(recorder.get("frame_count", -1)) != len(rows):
        raise PPOVideoArtifactError("viewport recorder frame count differs from its ledger")
    return len(rows)


def _validate_trace(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    require_nonzero_residual: bool,
) -> tuple[Mapping[str, Any], ...]:
    rows = _read_jsonl(path, label="15 Hz policy trace")
    decision_indices: list[int] = []
    physics_ticks: list[int] = []
    times: list[float] = []
    phases: list[str] = []
    maximum_residual = 0.0
    expected_seed = int(manifest["seed"])
    for index, row in enumerate(rows):
        try:
            row_seed = int(row["seed"])
            decision_indices.append(int(row["decision_index"]))
            physics_ticks.append(int(row["physics_tick"]))
            times.append(_finite(row["sim_time_s"], label=f"trace row {index} sim_time_s"))
            phase = str(row["state_id"])
            for field in (
                "pitch_error_rad",
                "roll_error_rad",
                "pitch_rate_rad_s",
                "roll_rate_rad_s",
            ):
                _finite(row[field], label=f"trace row {index} {field}")
            residual = tuple(
                _finite(value, label=f"trace row {index} residual")
                for value in row["residual_full12"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PPOVideoArtifactError(f"policy trace row {index} is incomplete") from exc
        if phase not in STATE_IDS:
            raise PPOVideoArtifactError(f"policy trace row {index} has invalid phase {phase!r}")
        if row_seed != expected_seed:
            raise PPOVideoArtifactError(f"policy trace row {index} has the wrong seed")
        if len(residual) != 12:
            raise PPOVideoArtifactError(f"policy trace row {index} is not Full12")
        maximum_residual = max(maximum_residual, *(abs(value) for value in residual))
        phases.append(phase)
    if decision_indices != list(range(len(rows))):
        raise PPOVideoArtifactError("policy decision indices are not contiguous from zero")
    tick_deltas = [
        right - left for left, right in zip(physics_ticks, physics_ticks[1:])
    ]
    if (
        not physics_ticks
        or physics_ticks[0] != 8
        or any(delta != 8 for delta in tick_deltas[:-1])
        or (tick_deltas and not 1 <= tick_deltas[-1] <= 8)
    ):
        raise PPOVideoArtifactError(
            "policy trace does not follow the actual 15 Hz decision cadence"
        )
    if any(
        abs(actual - tick / 120.0) > 1.0e-9
        for actual, tick in zip(times, physics_ticks, strict=True)
    ):
        raise PPOVideoArtifactError(
            "policy trace timestamps differ from their physical ticks"
        )
    if set(phases) != set(STATE_IDS):
        raise PPOVideoArtifactError("policy trace does not contain every phase P01-P13")
    if int(manifest.get("decision_count", -1)) != len(rows):
        raise PPOVideoArtifactError("source manifest decision count differs from policy trace")
    if str(rows[-1].get("termination_reason")) != "SUCCESS":
        raise PPOVideoArtifactError("policy trace does not end with authoritative SUCCESS")
    if require_nonzero_residual and maximum_residual <= 1.0e-12:
        raise PPOVideoArtifactError("PPO video trace contains only zero residuals")
    if not require_nonzero_residual and maximum_residual > 1.0e-12:
        raise PPOVideoArtifactError("FSM baseline video trace is not the zero-residual policy")
    return rows


def _validate_calibration(trial: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    calibration = trial.get("ppo_calibration")
    if not isinstance(calibration, Mapping) or calibration.get("quality_passed") is not True:
        raise PPOVideoArtifactError("source trial lacks a passing reset calibration")
    home = tuple(
        _finite(value, label="home joint calibration")
        for value in calibration.get("home_joint_positions_deg8", ())
    )
    level = tuple(
        _finite(value, label="level quaternion calibration")
        for value in calibration.get("level_reference_orientation_wxyz", ())
    )
    if len(home) != 8 or len(level) != 4:
        raise PPOVideoArtifactError("source reset calibration has the wrong dimensions")
    return home, level


def _validate_source_identity(manifest: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
    identity = manifest.get("source_identity")
    if not isinstance(identity, Mapping):
        raise PPOVideoArtifactError(f"{role} source lacks reset identity evidence")
    hashes: dict[str, str] = {}
    for key in (
        "environment_hash",
        "robot_asset_hash",
        "controller_hash",
        "motion_contract_hash",
    ):
        value = str(identity.get(key, "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise PPOVideoArtifactError(f"{role} source identity has an invalid {key}")
        hashes[key] = value
    vectors: dict[str, tuple[float, ...]] = {}
    for key, dimension in (
        ("initial_root_state", 13),
        ("initial_joint_state", 24),
        ("obstacle_pose", 3),
    ):
        try:
            vector = tuple(
                _finite(value, label=f"{role} source identity {key}")
                for value in identity.get(key, ())
            )
        except TypeError as exc:
            raise PPOVideoArtifactError(
                f"{role} source identity {key} is not a vector"
            ) from exc
        if len(vector) != dimension:
            raise PPOVideoArtifactError(
                f"{role} source identity {key} must have dimension {dimension}"
            )
        vectors[key] = vector
    return {**hashes, **vectors}


def _validate_camera(
    manifest: Mapping[str, Any],
    recorder: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, Any]:
    camera = manifest.get("camera")
    if not isinstance(camera, Mapping):
        raise PPOVideoArtifactError(f"{role} source lacks locked live camera evidence")
    try:
        eye = tuple(
            _finite(value, label=f"{role} camera eye")
            for value in camera.get("eye_m", ())
        )
        target = tuple(
            _finite(value, label=f"{role} camera target")
            for value in camera.get("target_m", ())
        )
        resolution = tuple(int(value) for value in camera.get("resolution", ()))
    except (TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{role} camera evidence is invalid") from exc
    fps = _finite(camera.get("fps"), label=f"{role} camera fps")
    if (
        len(eye) != 3
        or len(target) != 3
        or any(
            abs(actual - expected) > 1.0e-12
            for actual, expected in zip(eye, CAMERA_EYE_M, strict=True)
        )
        or any(
            abs(actual - expected) > 1.0e-12
            for actual, expected in zip(target, CAMERA_TARGET_M, strict=True)
        )
        or resolution != (VIDEO_WIDTH, VIDEO_HEIGHT)
        or not math.isclose(fps, VIDEO_FPS, rel_tol=0.0, abs_tol=1.0e-12)
        or camera.get("locked_scene_snapshot") is not True
        or camera.get("active_viewport_resolution_verified") is not True
        or camera.get("render_product_path") != recorder.get("render_product_path")
        or camera.get("viewport_identity") != recorder.get("viewport_identity")
    ):
        raise PPOVideoArtifactError(f"{role} source camera differs from the locked live view")
    return {
        "eye_m": eye,
        "target_m": target,
        "resolution": resolution,
        "fps": fps,
        "render_product_path": recorder.get("render_product_path"),
    }


def _load_source_episode(
    root_value: Path | str,
    *,
    role: str,
    ffmpeg: Path,
) -> SourceEpisode:
    managed = _validate_source_managed_run(root_value, role=role)
    root = _require_no_reparse_components(root_value, label=f"{role} source directory")
    if not root.is_dir():
        raise PPOVideoArtifactError(f"{role} source directory is missing: {root}")
    manifest_path = root / SOURCE_MANIFEST_NAME
    manifest = _load_json(manifest_path, label=f"{role} source manifest")
    if manifest.get("schema") != _SOURCE_SCHEMA:
        raise PPOVideoArtifactError(f"{role} source manifest has the wrong schema")
    try:
        source_seed = int(manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{role} source manifest has an invalid seed") from exc
    if source_seed < 0:
        raise PPOVideoArtifactError(f"{role} source manifest has a negative seed")
    if (
        manifest.get("task_success") is not True
        or manifest.get("completed_p01_p13") is not True
        or manifest.get("body_collision") is not False
        or manifest.get("wheel_only_climb") is not False
        or manifest.get("stitched") is not False
        or manifest.get("speed_modified") is not False
        or manifest.get("frame_interpolation") is not False
    ):
        raise PPOVideoArtifactError(f"{role} source did not pass physical/video acceptance")
    try:
        capture_process_id = int(manifest.get("capture_process_id", -1))
        episode_count = int(manifest.get("episode_count", -1))
    except (TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{role} source process evidence is invalid") from exc
    if (
        manifest.get("fresh_process_single_episode") is not True
        or episode_count != 1
        or capture_process_id <= 0
    ):
        raise PPOVideoArtifactError(
            f"{role} source is not a fresh single-episode capture"
        )
    process_instance_id = str(manifest.get("capture_process_instance_id", ""))
    if re.fullmatch(r"[0-9a-f]{32}", process_instance_id) is None:
        raise PPOVideoArtifactError(
            f"{role} source lacks a valid fresh process instance identity"
        )
    reset_evidence = manifest.get("reset_evidence")
    if (
        not isinstance(reset_evidence, Mapping)
        or reset_evidence.get("reset_count") != 1
        or reset_evidence.get("reset_options") != {}
        or reset_evidence.get("training_phase_snapshot") is not None
        or reset_evidence.get("reset_global_simulation_resets") != 0
        or reset_evidence.get("reset_simulation_forward_syncs") != 0
    ):
        raise PPOVideoArtifactError(
            f"{role} source lacks exact first-reset freshness evidence"
        )
    if manifest.get("pre_action_observation_refreshed") is not True:
        raise PPOVideoArtifactError(
            f"{role} source used a stale pre-roll policy observation"
        )
    refresh_evidence = manifest.get("pre_action_refresh_evidence")
    if (
        not isinstance(refresh_evidence, Mapping)
        or refresh_evidence.get("schema")
        != "wlr50_clean.ppo_video_pre_action_refresh.v1"
        or refresh_evidence.get("episode_reader_reinitialized") is not True
        or refresh_evidence.get("episode_reader_first_logical_tick") != 0
        or refresh_evidence.get("controller_frame_preserved") is not True
        or refresh_evidence.get("controller_logical_tick") != 0
        or refresh_evidence.get("simulation_reset_performed") is not False
        or refresh_evidence.get("fsm_step_performed") is not False
    ):
        raise PPOVideoArtifactError(
            f"{role} source lacks an exact post-pre-roll sensing refresh"
        )
    if int(manifest.get("reset_info_recording_access_count", -1)) != 0:
        raise PPOVideoArtifactError(f"{role} source accessed Recording at runtime")
    source_identity = _validate_source_identity(manifest, role=role)
    episode_duration = _finite(manifest.get("duration_s"), label=f"{role} episode duration")
    expected_video_duration = _finite(
        manifest.get("video_duration_expected_s"), label=f"{role} expected video duration"
    )
    if episode_duration <= 0.0 or episode_duration > 200.0 or expected_video_duration > 200.0:
        raise PPOVideoArtifactError(f"{role} source exceeds the 200 second limit")
    pre_roll = _finite(
        manifest.get("pre_action_physical_hold_s"), label=f"{role} pre-action hold"
    )
    post_roll = _finite(
        manifest.get("post_success_physical_hold_s"), label=f"{role} post-success hold"
    )
    if not 0.5 <= pre_roll <= 1.0 or not 1.0 <= post_roll <= 2.0:
        raise PPOVideoArtifactError(f"{role} source has invalid pre/post action context")
    expected_pre_ticks = round(pre_roll * 120.0)
    if (
        refresh_evidence.get("physical_pre_action_ticks") != expected_pre_ticks
        or refresh_evidence.get("pre_roll_reader_last_logical_tick")
        != expected_pre_ticks
    ):
        raise PPOVideoArtifactError(
            f"{role} source refresh evidence differs from its physical pre-roll"
        )
    action_start = _finite(
        manifest.get("semantic_action_start_video_s"), label=f"{role} action start"
    )
    task_success_time = _finite(
        manifest.get("semantic_task_success_video_s"), label=f"{role} task success time"
    )
    if (
        abs(action_start - pre_roll) > 1.0e-9
        or abs(task_success_time - (pre_roll + episode_duration)) > 1.0e-9
        or abs(expected_video_duration - (pre_roll + episode_duration + post_roll)) > 1.0e-9
    ):
        raise PPOVideoArtifactError(f"{role} semantic video timeline is inconsistent")

    video_path = (root / SOURCE_VIDEO_NAME).resolve()
    declared_video = _inside(root, manifest.get("raw_video", ""), label=f"{role} raw video")
    if declared_video != video_path or not video_path.is_file():
        raise PPOVideoArtifactError(f"{role} manifest is not bound to {SOURCE_VIDEO_NAME}")
    video_sha = sha256_file(video_path)
    if str(manifest.get("raw_video_sha256", "")).lower() != video_sha:
        raise PPOVideoArtifactError(f"{role} raw video hash does not match its source manifest")

    recorder_manifest_path = _inside(
        root, manifest.get("recorder_manifest", ""), label=f"{role} recorder manifest"
    )
    recorder = _load_json(recorder_manifest_path, label=f"{role} recorder manifest")
    if recorder.get("valid") is not True or recorder.get("stitched") is not False:
        raise PPOVideoArtifactError(f"{role} viewport recorder did not pass")
    if recorder.get("speed_modified") is not False:
        raise PPOVideoArtifactError(f"{role} viewport recorder changed video speed")
    camera = _validate_camera(manifest, recorder, role=role)
    recorder_video = _inside(
        root, recorder.get("video_path", ""), label=f"{role} recorder video"
    )
    if recorder_video != video_path or str(recorder.get("video_sha256", "")).lower() != video_sha:
        raise PPOVideoArtifactError(f"{role} recorder manifest is bound to a different video")
    full_decode = recorder.get("full_decode")
    if not isinstance(full_decode, Mapping) or full_decode.get("valid") is not True:
        raise PPOVideoArtifactError(f"{role} recorder did not fully decode its source video")

    ledger_path = (root / SOURCE_LEDGER_NAME).resolve()
    declared_ledger = _inside(root, recorder.get("ledger_path", ""), label=f"{role} ledger")
    if declared_ledger != ledger_path or not ledger_path.is_file():
        raise PPOVideoArtifactError(f"{role} recorder is not bound to {SOURCE_LEDGER_NAME}")
    if str(recorder.get("ledger_sha256", "")).lower() != sha256_file(ledger_path):
        raise PPOVideoArtifactError(f"{role} viewport ledger hash mismatch")
    ledger_count = _validate_ledger(ledger_path, recorder)

    trial_manifest_path = _inside(
        root, manifest.get("trial_manifest", ""), label=f"{role} trial manifest"
    )
    trial_manifest = _load_json(trial_manifest_path, label=f"{role} trial manifest")
    _validate_calibration(trial_manifest)

    trace_path = (root / SOURCE_TRACE_NAME).resolve()
    declared_trace = _inside(
        root, manifest.get("policy_trace", ""), label=f"{role} policy trace"
    )
    if declared_trace != trace_path or not trace_path.is_file():
        raise PPOVideoArtifactError(f"{role} source is not bound to {SOURCE_TRACE_NAME}")
    if str(manifest.get("policy_trace_sha256", "")).lower() != sha256_file(trace_path):
        raise PPOVideoArtifactError(f"{role} policy trace hash mismatch")
    if not math.isclose(
        _finite(manifest.get("policy_trace_rate_hz"), label=f"{role} trace rate"),
        15.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise PPOVideoArtifactError(f"{role} policy trace is not the 15 Hz control trace")
    trace = _validate_trace(
        trace_path,
        manifest=manifest,
        require_nonzero_residual=role == "ppo",
    )
    if int(manifest.get("policy_trace_rows", -1)) != len(trace):
        raise PPOVideoArtifactError(f"{role} source trace row count mismatch")
    if abs(float(trace[-1]["sim_time_s"]) - episode_duration) > _FRAME_PERIOD_S + 1.0e-9:
        raise PPOVideoArtifactError(f"{role} policy trace does not reach task termination")
    if role == "fsm":
        if manifest.get("policy_label") != "fsm_zero_residual":
            raise PPOVideoArtifactError("FSM source is not labelled as zero residual")
        if manifest.get("deterministic_mean_policy") is not False:
            raise PPOVideoArtifactError("FSM source unexpectedly claims PPO inference")
        if manifest.get("source_checkpoint") is not None:
            raise PPOVideoArtifactError("FSM zero-residual source names a checkpoint")
        if manifest.get("used_preinitialized_fresh_episode") is not False:
            raise PPOVideoArtifactError(
                "FSM source did not perform its one reset inside capture"
            )
        checkpoint_path = None
        checkpoint_sha256 = None
    else:
        if manifest.get("policy_label") != "ppo_deterministic_mean":
            raise PPOVideoArtifactError("PPO source is not deterministic mean inference")
        if manifest.get("deterministic_mean_policy") is not True:
            raise PPOVideoArtifactError("PPO source does not assert deterministic mean inference")
        if manifest.get("used_preinitialized_fresh_episode") is not True:
            raise PPOVideoArtifactError(
                "PPO source was not captured from the runner's first fresh reset"
            )
        checkpoint_text = manifest.get("source_checkpoint")
        if not checkpoint_text:
            raise PPOVideoArtifactError("PPO source checkpoint is missing")
        checkpoint_path = Path(str(checkpoint_text)).resolve()
        if not checkpoint_path.is_file():
            raise PPOVideoArtifactError(f"PPO source checkpoint is missing: {checkpoint_path}")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if str(manifest.get("source_checkpoint_sha256", "")).lower() != checkpoint_sha256:
            raise PPOVideoArtifactError("PPO source checkpoint hash mismatch")
        checkpoint_manifest_text = manifest.get("source_checkpoint_manifest")
        if not checkpoint_manifest_text:
            raise PPOVideoArtifactError("PPO source checkpoint manifest is missing")
        checkpoint_manifest_path = Path(str(checkpoint_manifest_text)).resolve()
        checkpoint_manifest = _load_json(
            checkpoint_manifest_path,
            label="PPO source checkpoint manifest",
        )
        checkpoint_manifest_sha256 = sha256_file(checkpoint_manifest_path)
        if (
            str(manifest.get("source_checkpoint_manifest_sha256", "")).lower()
            != checkpoint_manifest_sha256
            or Path(str(checkpoint_manifest.get("checkpoint_path", ""))).resolve()
            != checkpoint_path
            or str(checkpoint_manifest.get("checkpoint_sha256", "")).lower()
            != checkpoint_sha256
            or checkpoint_manifest.get("publication_role") != "improved"
            or checkpoint_manifest.get("validation_promotion_authorized") is not True
            or checkpoint_manifest.get("locked_test_authorized") is not True
            or checkpoint_manifest.get("promotion_authorized") is not True
        ):
            raise PPOVideoArtifactError(
                "PPO source checkpoint manifest is not the promoted improved artifact"
            )
        load_provenance = manifest.get("checkpoint_load_provenance")
        if (
            not isinstance(load_provenance, Mapping)
            or Path(str(load_provenance.get("checkpoint_path", ""))).resolve()
            != checkpoint_path
            or str(load_provenance.get("checkpoint_sha256", "")).lower()
            != checkpoint_sha256
            or Path(str(load_provenance.get("manifest_path", ""))).resolve()
            != checkpoint_manifest_path
            or str(load_provenance.get("manifest_sha256", "")).lower()
            != checkpoint_manifest_sha256
            or load_provenance.get("checkpoint_infos_match_manifest") is not True
        ):
            raise PPOVideoArtifactError(
                "PPO source lacks matching strict checkpoint load evidence"
            )

    source_validation = validate_mp4(
        video_path,
        ffmpeg=ffmpeg,
        expected_width=VIDEO_WIDTH,
        expected_height=VIDEO_HEIGHT,
        expected_fps=VIDEO_FPS,
        expected_frame_count=ledger_count,
        maximum_duration_s=200.0,
        stitched=False,
        speed_modified=False,
        require_sane_container_duration=False,
    )
    if source_validation.get("valid") is not True:
        raise PPOVideoArtifactError(f"{role} source video failed independent full decode")
    if abs(float(source_validation["duration_s"]) - expected_video_duration) > 2.0 * _FRAME_PERIOD_S:
        raise PPOVideoArtifactError(f"{role} source video duration differs from its physical timeline")
    if int(source_validation["frame_count"]) != ledger_count:
        raise PPOVideoArtifactError(f"{role} source video differs from its viewport ledger")

    return SourceEpisode(
        role=role,
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        recorder_manifest_path=recorder_manifest_path,
        recorder_manifest=recorder,
        trial_manifest_path=trial_manifest_path,
        trial_manifest=trial_manifest,
        video_path=video_path,
        ledger_path=ledger_path,
        trace_path=trace_path,
        trace=trace,
        source_identity=source_identity,
        camera=camera,
        video_validation=source_validation,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        managed_run_evidence=managed.evidence,
        committed_runtime_identity=managed.runtime_identity,
    )


def _quaternion_difference(left: Sequence[float], right: Sequence[float]) -> float:
    direct = max(abs(a - b) for a, b in zip(left, right, strict=True))
    sign_flipped = max(abs(a + b) for a, b in zip(left, right, strict=True))
    return min(direct, sign_flipped)


def _validate_pair(fsm: SourceEpisode, ppo: SourceEpisode) -> Mapping[str, Any]:
    if fsm.seed != ppo.seed:
        raise PPOVideoArtifactError("FSM and PPO source videos use different seeds")
    if fsm.seed != 4001:
        raise PPOVideoArtifactError("final source videos do not use locked video seed 4001")
    if (
        fsm.root == ppo.root
        or int(fsm.manifest["capture_process_id"])
        == int(ppo.manifest["capture_process_id"])
        or fsm.manifest["capture_process_instance_id"]
        == ppo.manifest["capture_process_instance_id"]
    ):
        raise PPOVideoArtifactError(
            "FSM and PPO sources were not captured in independent fresh processes"
        )
    if dict(fsm.committed_runtime_identity) != dict(ppo.committed_runtime_identity):
        raise PPOVideoArtifactError(
            "FSM and PPO sources were captured by different committed runtimes"
        )
    recorder_keys = ("capture_backend", "render_product_path", "width", "height", "fps")
    mismatches = [
        key
        for key in recorder_keys
        if fsm.recorder_manifest.get(key) != ppo.recorder_manifest.get(key)
    ]
    if mismatches:
        raise PPOVideoArtifactError(
            f"FSM and PPO source videos use different camera/capture settings: {mismatches}"
        )
    camera_vector_differences = {
        key: max(
            abs(left - right)
            for left, right in zip(
                fsm.camera[key], ppo.camera[key], strict=True
            )
        )
        for key in ("eye_m", "target_m")
    }
    if (
        camera_vector_differences["eye_m"] > 1.0e-12
        or camera_vector_differences["target_m"] > 1.0e-12
        or fsm.camera["resolution"] != ppo.camera["resolution"]
        or fsm.camera["fps"] != ppo.camera["fps"]
    ):
        raise PPOVideoArtifactError("FSM and PPO source videos use different cameras")
    identity_hashes = (
        "environment_hash",
        "robot_asset_hash",
        "controller_hash",
        "motion_contract_hash",
    )
    identity_mismatches = [
        key
        for key in identity_hashes
        if fsm.source_identity[key] != ppo.source_identity[key]
    ]
    if identity_mismatches:
        raise PPOVideoArtifactError(
            f"FSM and PPO source identities differ: {identity_mismatches}"
        )
    vector_differences = {
        key: max(
            abs(left - right)
            for left, right in zip(
                fsm.source_identity[key], ppo.source_identity[key], strict=True
            )
        )
        for key in ("initial_root_state", "initial_joint_state", "obstacle_pose")
    }
    if (
        vector_differences["initial_root_state"] > 1.0e-9
        or vector_differences["initial_joint_state"] > 1.0e-9
        or vector_differences["obstacle_pose"] > 1.0e-12
    ):
        raise PPOVideoArtifactError(
            "FSM and PPO source identities differ in initial state or obstacle pose"
        )
    fsm_home, fsm_level = _validate_calibration(fsm.trial_manifest)
    ppo_home, ppo_level = _validate_calibration(ppo.trial_manifest)
    home_difference = max(abs(a - b) for a, b in zip(fsm_home, ppo_home, strict=True))
    level_difference = _quaternion_difference(fsm_level, ppo_level)
    if home_difference > 1.0e-6 or level_difference > 1.0e-6:
        raise PPOVideoArtifactError("FSM and PPO videos do not share the same calibrated initial state")
    return {
        "same_seed": True,
        "seed": fsm.seed,
        "independent_fresh_capture_processes": True,
        "fsm_capture_process_id": int(fsm.manifest["capture_process_id"]),
        "ppo_capture_process_id": int(ppo.manifest["capture_process_id"]),
        "fsm_capture_process_instance_id": fsm.manifest[
            "capture_process_instance_id"
        ],
        "ppo_capture_process_instance_id": ppo.manifest[
            "capture_process_instance_id"
        ],
        "same_live_environment_contract": True,
        "same_committed_runtime": True,
        "source_git_commit": fsm.committed_runtime_identity["git_commit"],
        "committed_runtime_content_sha256": fsm.committed_runtime_identity[
            "content_sha256"
        ],
        "environment_hash": fsm.source_identity["environment_hash"],
        "robot_asset_hash": fsm.source_identity["robot_asset_hash"],
        "controller_hash": fsm.source_identity["controller_hash"],
        "motion_contract_hash": fsm.source_identity["motion_contract_hash"],
        "same_obstacle_contract": True,
        "obstacle_pose": list(fsm.source_identity["obstacle_pose"]),
        "obstacle_pose_max_abs_difference_m": vector_differences["obstacle_pose"],
        "same_camera": True,
        "camera_eye_m": list(fsm.camera["eye_m"]),
        "camera_target_m": list(fsm.camera["target_m"]),
        "camera_eye_max_abs_difference_m": camera_vector_differences["eye_m"],
        "camera_target_max_abs_difference_m": camera_vector_differences[
            "target_m"
        ],
        "capture_backend": fsm.recorder_manifest.get("capture_backend"),
        "render_product_path": fsm.recorder_manifest.get("render_product_path"),
        "same_resolution": True,
        "source_resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
        "same_initial_state": True,
        "initial_root_state_max_abs_difference": vector_differences[
            "initial_root_state"
        ],
        "initial_joint_state_max_abs_difference": vector_differences[
            "initial_joint_state"
        ],
        "home_joint_max_abs_difference_deg": home_difference,
        "level_quaternion_sign_invariant_max_abs_difference": level_difference,
    }


def _run_ffmpeg(command: Sequence[str], *, label: str) -> None:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=1200,
    )
    if completed.returncode != 0:
        tail = completed.stderr[-4000:].replace("\r", " ").replace("\n", " ")
        raise PPOVideoArtifactError(f"{label} ffmpeg failed ({completed.returncode}): {tail}")


def _encode_clean_source(ffmpeg: Path, source: Path, destination: Path, *, label: str) -> None:
    _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "setpts=PTS-STARTPTS,format=yuv420p",
            "-an",
            "-sn",
            "-dn",
            "-r",
            f"{VIDEO_FPS:g}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        label=label,
    )


def _comparison_filter(fsm_frames: int, ppo_frames: int) -> tuple[str, str, int]:
    output_frames = max(fsm_frames, ppo_frames)
    output_duration = output_frames / VIDEO_FPS
    if fsm_frames < ppo_frames:
        padding_side = "fsm"
        padding_frames = ppo_frames - fsm_frames
        padding_duration = padding_frames / VIDEO_FPS
        left = (
            f"[0:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"tpad=stop_mode=clone:stop_duration={padding_duration:.9f},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=218:h=46:color=black@0.55:t=fill,"
            "drawtext=text='FSM baseline':x=28:y=26:fontcolor=white:fontsize=26[left]"
        )
        right = (
            f"[1:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=220:h=46:color=black@0.55:t=fill,"
            "drawtext=text='PPO improved':x=28:y=26:fontcolor=white:fontsize=26[right]"
        )
    elif ppo_frames < fsm_frames:
        padding_side = "ppo"
        padding_frames = fsm_frames - ppo_frames
        padding_duration = padding_frames / VIDEO_FPS
        left = (
            f"[0:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=218:h=46:color=black@0.55:t=fill,"
            "drawtext=text='FSM baseline':x=28:y=26:fontcolor=white:fontsize=26[left]"
        )
        right = (
            f"[1:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"tpad=stop_mode=clone:stop_duration={padding_duration:.9f},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=220:h=46:color=black@0.55:t=fill,"
            "drawtext=text='PPO improved':x=28:y=26:fontcolor=white:fontsize=26[right]"
        )
    else:
        padding_side = "none"
        padding_frames = 0
        left = (
            f"[0:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=218:h=46:color=black@0.55:t=fill,"
            "drawtext=text='FSM baseline':x=28:y=26:fontcolor=white:fontsize=26[left]"
        )
        right = (
            f"[1:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=220:h=46:color=black@0.55:t=fill,"
            "drawtext=text='PPO improved':x=28:y=26:fontcolor=white:fontsize=26[right]"
        )
    return (
        f"{left};{right};[left][right]hstack=inputs=2,format=yuv420p[outv]",
        padding_side,
        padding_frames,
    )


def _encode_comparison(
    ffmpeg: Path,
    fsm: Path,
    ppo: Path,
    destination: Path,
    *,
    fsm_frames: int,
    ppo_frames: int,
) -> tuple[str, int]:
    filter_graph, padding_side, padding_frames = _comparison_filter(fsm_frames, ppo_frames)
    _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(fsm),
            "-i",
            str(ppo),
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            f"{VIDEO_FPS:g}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        label="real-time FSM-versus-PPO comparison",
    )
    return padding_side, padding_frames


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100.0)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def build_diagnostic_ass(
    trace: Sequence[Mapping[str, Any]],
    *,
    action_start_video_s: float,
    video_duration_s: float,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> str:
    """Build subtitles whose changing values come directly from the 15 Hz trace."""

    if not trace:
        raise PPOVideoArtifactError("cannot build diagnostic overlay from an empty trace")
    start_offset = _finite(action_start_video_s, label="diagnostic action start")
    duration = _finite(video_duration_s, label="diagnostic video duration")
    if not 0.0 <= start_offset < duration:
        raise PPOVideoArtifactError("diagnostic action start lies outside the video")
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Telemetry,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H90000000,0,0,0,0,100,100,0,0,3,1,0,2,28,28,28,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    first_time = start_offset + _finite(trace[0]["sim_time_s"], label="first trace time")
    if first_time > start_offset:
        lines.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(start_offset)},{_ass_timestamp(first_time)},Telemetry,,0,0,0,,"
            "P01 | awaiting first physical 15 Hz telemetry sample"
        )
    for index, row in enumerate(trace):
        start = start_offset + _finite(row["sim_time_s"], label=f"trace row {index} time")
        if index + 1 < len(trace):
            end = start_offset + _finite(
                trace[index + 1]["sim_time_s"], label=f"trace row {index + 1} time"
            )
        else:
            end = duration
        start = min(max(start_offset, start), duration)
        end = min(max(start + 0.01, end), duration)
        if end <= start:
            continue
        residual = tuple(float(value) for value in row["residual_full12"])
        residual_rms = math.sqrt(sum(value * value for value in residual) / len(residual))
        pitch = math.degrees(float(row["pitch_error_rad"]))
        roll = math.degrees(float(row["roll_error_rad"]))
        pitch_rate = float(row["pitch_rate_rad_s"])
        roll_rate = float(row["roll_rate_rad_s"])
        result = row.get("termination_reason") or "RUNNING"
        first_line = _ass_text(
            f"phase {row['state_id']}   pitch {pitch:+.3f} deg   "
            f"pitch rate {pitch_rate:+.4f} rad/s"
        )
        second_line = _ass_text(
            f"roll {roll:+.3f} deg   roll rate {roll_rate:+.4f} rad/s   "
            f"residual RMS {residual_rms:.6f}   result {result}"
        )
        text = first_line + r"\N" + second_line
        lines.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(start)},{_ass_timestamp(end)},Telemetry,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _ffmpeg_filter_path(path: Path) -> str:
    text = path.resolve().as_posix()
    text = text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
    return f"'{text}'"


def _encode_diagnostic(ffmpeg: Path, source: Path, ass_path: Path, destination: Path) -> None:
    _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-vf",
            f"ass=filename={_ffmpeg_filter_path(ass_path)}",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-r",
            f"{VIDEO_FPS:g}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        label="PPO diagnostic overlay",
    )


def _require_generated_video(
    path: Path,
    *,
    ffmpeg: Path,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
) -> Mapping[str, Any]:
    validation = validate_mp4(
        path,
        ffmpeg=ffmpeg,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_fps=VIDEO_FPS,
        expected_frame_count=expected_frame_count,
        maximum_duration_s=200.0,
        stitched=False,
        speed_modified=False,
        require_sane_container_duration=True,
    )
    if validation.get("valid") is not True:
        raise PPOVideoArtifactError(f"generated video failed full validation: {path.name}")
    return validation


def _video_record(
    validation: Mapping[str, Any],
    *,
    final_path: Path,
    source_episode: str | Sequence[str],
    source_checkpoint: str | Sequence[str],
    source_checkpoint_sha256: str | Sequence[str | None],
    source_seed: int,
    processing: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(validation),
        "path": str(final_path),
        "pix_fmt": validation.get("pixel_format"),
        "source_episode": source_episode,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_seed": int(source_seed),
        "full_decode": validation.get("full_decode") is True,
        "monotonic": validation.get("timestamps_monotonic") is True,
        "stitched": False,
        "speed_modified": False,
        "processing": dict(processing),
    }


def _publish_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise PPOVideoArtifactError(f"refusing to overwrite final artifact: {destination}") from exc
    source.unlink()


def _copy_stable_source(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    """Copy one immutable input from one open handle and verify those exact bytes."""

    selected = _require_no_reparse_components(source, label=label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with selected.open("rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise PPOVideoArtifactError(f"{label} is not a regular file")
            while True:
                block = reader.read(1024 * 1024)
                if not block:
                    break
                writer.write(block)
                digest.update(block)
                size += len(block)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
    except PPOVideoArtifactError:
        raise
    except OSError as exc:
        raise PPOVideoArtifactError(f"cannot pin {label} for encoding") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    actual_sha256 = digest.hexdigest()
    if (
        size <= 0
        or size != before.st_size
        or any(
            getattr(before, field, None) != getattr(after, field, None)
            for field in stable_fields
        )
        or actual_sha256 != expected_sha256
    ):
        raise PPOVideoArtifactError(f"{label} changed before it could be pinned")
    _require_no_reparse_components(selected, label=label)
    return {
        "path": str(selected),
        "bytes": size,
        "sha256": actual_sha256,
    }


def publish_final_videos(
    *,
    fsm_source_dir: Path | str,
    ppo_source_dir: Path | str,
    output_root: Path | str,
    publication_run_dir: Path | str,
    ffmpeg: Path | str | None = None,
) -> FinalVideoPublication:
    """Create and validate the four immutable phase-residual PPO final videos."""

    try:
        executable = find_ffmpeg(ffmpeg)
    except (FileNotFoundError, OSError) as exc:
        raise PPOVideoArtifactError("ffmpeg is unavailable for final video publication") from exc
    output = _require_no_reparse_components(output_root, label="video output root")
    videos_dir = output / "videos"
    manifests_dir = output / "manifests"
    destinations = {
        "fsm_baseline": videos_dir / FSM_VIDEO_NAME,
        "ppo_improved": videos_dir / PPO_VIDEO_NAME,
        "comparison": videos_dir / COMPARISON_VIDEO_NAME,
        "ppo_diagnostic": videos_dir / DIAGNOSTIC_VIDEO_NAME,
    }
    validation_path = manifests_dir / VIDEO_VALIDATION_NAME
    checksum_path = manifests_dir / VIDEO_CHECKSUM_NAME
    ass_path = manifests_dir / DIAGNOSTIC_ASS_NAME
    all_destinations = (*destinations.values(), validation_path, checksum_path, ass_path)
    existing = [path for path in all_destinations if path.exists()]
    if existing:
        raise PPOVideoArtifactError(f"refusing to overwrite final artifact: {existing[0]}")

    fsm = _load_source_episode(fsm_source_dir, role="fsm", ffmpeg=executable)
    ppo = _load_source_episode(ppo_source_dir, role="ppo", ffmpeg=executable)
    pair_evidence = _validate_pair(fsm, ppo)
    publication_run = _publication_reservation_evidence(
        publication_run_dir,
        fsm_source_dir=fsm.root,
        ppo_source_dir=ppo.root,
        output_root=output,
        expected_runtime_identity=fsm.committed_runtime_identity,
    )

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ppo-video-publication-", dir=str(output)))
    try:
        stage_videos = staging / "videos"
        stage_manifests = staging / "manifests"
        stage_videos.mkdir(parents=True)
        stage_manifests.mkdir(parents=True)
        stage_paths = {
            "fsm_baseline": stage_videos / FSM_VIDEO_NAME,
            "ppo_improved": stage_videos / PPO_VIDEO_NAME,
            "comparison": stage_videos / COMPARISON_VIDEO_NAME,
            "ppo_diagnostic": stage_videos / DIAGNOSTIC_VIDEO_NAME,
        }
        stage_ass = stage_manifests / DIAGNOSTIC_ASS_NAME
        pinned_fsm = staging / "pinned_sources" / "fsm.mp4"
        pinned_ppo = staging / "pinned_sources" / "ppo.mp4"
        fsm_source_record = _copy_stable_source(
            fsm.video_path,
            pinned_fsm,
            expected_sha256=str(fsm.manifest["raw_video_sha256"]),
            label="FSM raw source video",
        )
        ppo_source_record = _copy_stable_source(
            ppo.video_path,
            pinned_ppo,
            expected_sha256=str(ppo.manifest["raw_video_sha256"]),
            label="PPO raw source video",
        )

        _encode_clean_source(
            executable, pinned_fsm, stage_paths["fsm_baseline"], label="FSM clean video"
        )
        _encode_clean_source(
            executable, pinned_ppo, stage_paths["ppo_improved"], label="PPO clean video"
        )
        padding_side, padding_frames = _encode_comparison(
            executable,
            stage_paths["fsm_baseline"],
            stage_paths["ppo_improved"],
            stage_paths["comparison"],
            fsm_frames=fsm.frame_count,
            ppo_frames=ppo.frame_count,
        )
        ass_document = build_diagnostic_ass(
            ppo.trace,
            action_start_video_s=float(ppo.manifest["semantic_action_start_video_s"]),
            video_duration_s=ppo.duration_s,
        )
        stage_ass.write_bytes(ass_document.encode("utf-8"))
        _encode_diagnostic(
            executable,
            stage_paths["ppo_improved"],
            stage_ass,
            stage_paths["ppo_diagnostic"],
        )

        validations = {
            "fsm_baseline": _require_generated_video(
                stage_paths["fsm_baseline"],
                ffmpeg=executable,
                expected_width=VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=fsm.frame_count,
            ),
            "ppo_improved": _require_generated_video(
                stage_paths["ppo_improved"],
                ffmpeg=executable,
                expected_width=VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=ppo.frame_count,
            ),
            "comparison": _require_generated_video(
                stage_paths["comparison"],
                ffmpeg=executable,
                expected_width=2 * VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=max(fsm.frame_count, ppo.frame_count),
            ),
            "ppo_diagnostic": _require_generated_video(
                stage_paths["ppo_diagnostic"],
                ffmpeg=executable,
                expected_width=VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=ppo.frame_count,
            ),
        }
        for name, result in validations.items():
            if float(result["duration_s"]) > 200.0:
                raise PPOVideoArtifactError(f"{name} exceeds 200 seconds")

        fsm_checkpoint_label = "frozen_successful_fsm_zero_residual"
        checkpoint_path = str(ppo.checkpoint_path)
        checkpoint_hash = str(ppo.checkpoint_sha256)
        video_records = {
            "fsm_baseline": _video_record(
                validations["fsm_baseline"],
                final_path=destinations["fsm_baseline"],
                source_episode=fsm.root.name,
                source_checkpoint=fsm_checkpoint_label,
                source_checkpoint_sha256="not_applicable",
                source_seed=fsm.seed,
                processing={
                    "kind": "single_episode_full_source_transcode",
                    "input_count": 1,
                    "cuts": False,
                    "time_scale_transform": None,
                    "timestamp_transform": "PTS-STARTPTS",
                    "source_frame_count": fsm.frame_count,
                    "output_frame_count": int(validations["fsm_baseline"]["frame_count"]),
                },
            ),
            "ppo_improved": _video_record(
                validations["ppo_improved"],
                final_path=destinations["ppo_improved"],
                source_episode=ppo.root.name,
                source_checkpoint=checkpoint_path,
                source_checkpoint_sha256=checkpoint_hash,
                source_seed=ppo.seed,
                processing={
                    "kind": "single_episode_full_source_transcode",
                    "input_count": 1,
                    "deterministic_mean_policy": True,
                    "cuts": False,
                    "time_scale_transform": None,
                    "timestamp_transform": "PTS-STARTPTS",
                    "source_frame_count": ppo.frame_count,
                    "output_frame_count": int(validations["ppo_improved"]["frame_count"]),
                },
            ),
            "comparison": _video_record(
                validations["comparison"],
                final_path=destinations["comparison"],
                source_episode=[fsm.root.name, ppo.root.name],
                source_checkpoint=[fsm_checkpoint_label, checkpoint_path],
                source_checkpoint_sha256=[None, checkpoint_hash],
                source_seed=fsm.seed,
                processing={
                    "kind": "real_time_spatial_side_by_side",
                    "temporal_stitching": False,
                    "left": "fsm_baseline",
                    "right": "ppo_improved",
                    "common_time_origin": "source_frame_zero",
                    "phase_alignment": False,
                    "time_scale_transform": None,
                    "earlier_final_frame_tpad_clone_side": padding_side,
                    "earlier_final_frame_tpad_clone_frames": padding_frames,
                    "other_frame_duplication_or_interpolation": False,
                },
            ),
            "ppo_diagnostic": _video_record(
                validations["ppo_diagnostic"],
                final_path=destinations["ppo_diagnostic"],
                source_episode=ppo.root.name,
                source_checkpoint=checkpoint_path,
                source_checkpoint_sha256=checkpoint_hash,
                source_seed=ppo.seed,
                processing={
                    "kind": "single_episode_full_source_15hz_trace_overlay",
                    "cuts": False,
                    "time_scale_transform": None,
                    "trace_path": str(ppo.trace_path),
                    "trace_sha256": sha256_file(ppo.trace_path),
                    "trace_rate_hz": 15.0,
                    "trace_sample_count": len(ppo.trace),
                    "ass_event_count": sum(
                        line.startswith("Dialogue:") for line in ass_document.splitlines()
                    ),
                    "ass_sha256": hashlib.sha256(ass_document.encode("utf-8")).hexdigest(),
                    "overlay_fields": [
                        "phase",
                        "pitch",
                        "pitch_rate",
                        "roll",
                        "roll_rate",
                        "residual_rms",
                        "task_result",
                    ],
                },
            ),
        }
        validation_payload = {
            "schema": _PUBLICATION_SCHEMA,
            "valid": True,
            "status": "PASS",
            "immutable_no_overwrite": True,
            "fps": VIDEO_FPS,
            "maximum_duration_s": 200.0,
            "pair_evidence": dict(pair_evidence),
            "publication_run": dict(publication_run.evidence),
            "source_episodes": {
                "fsm": {
                    "directory": str(fsm.root),
                    "source_manifest": str(fsm.manifest_path),
                    "source_manifest_sha256": sha256_file(fsm.manifest_path),
                    "source_manifest_bytes": fsm.manifest_path.stat().st_size,
                    "viewport_ledger": str(fsm.ledger_path),
                    "viewport_ledger_sha256": sha256_file(fsm.ledger_path),
                    "viewport_ledger_bytes": fsm.ledger_path.stat().st_size,
                    "policy_trace": str(fsm.trace_path),
                    "policy_trace_sha256": sha256_file(fsm.trace_path),
                    "policy_trace_bytes": fsm.trace_path.stat().st_size,
                    "raw_video": str(fsm.video_path),
                    "raw_video_sha256": fsm_source_record["sha256"],
                    "raw_video_bytes": fsm_source_record["bytes"],
                    "managed_run": dict(fsm.managed_run_evidence),
                },
                "ppo": {
                    "directory": str(ppo.root),
                    "source_manifest": str(ppo.manifest_path),
                    "source_manifest_sha256": sha256_file(ppo.manifest_path),
                    "source_manifest_bytes": ppo.manifest_path.stat().st_size,
                    "viewport_ledger": str(ppo.ledger_path),
                    "viewport_ledger_sha256": sha256_file(ppo.ledger_path),
                    "viewport_ledger_bytes": ppo.ledger_path.stat().st_size,
                    "policy_trace": str(ppo.trace_path),
                    "policy_trace_sha256": sha256_file(ppo.trace_path),
                    "policy_trace_bytes": ppo.trace_path.stat().st_size,
                    "raw_video": str(ppo.video_path),
                    "raw_video_sha256": ppo_source_record["sha256"],
                    "raw_video_bytes": ppo_source_record["bytes"],
                    "checkpoint": checkpoint_path,
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_bytes": ppo.checkpoint_path.stat().st_size,
                    "checkpoint_manifest": ppo.manifest[
                        "source_checkpoint_manifest"
                    ],
                    "checkpoint_manifest_sha256": ppo.manifest[
                        "source_checkpoint_manifest_sha256"
                    ],
                    "checkpoint_manifest_bytes": Path(
                        str(ppo.manifest["source_checkpoint_manifest"])
                    ).stat().st_size,
                    "checkpoint_load_provenance": dict(
                        ppo.manifest["checkpoint_load_provenance"]
                    ),
                    "deterministic_mean_policy": True,
                    "managed_run": dict(ppo.managed_run_evidence),
                },
            },
            "videos": video_records,
            "diagnostic_ass": {
                "path": str(ass_path),
                "sha256": hashlib.sha256(ass_document.encode("utf-8")).hexdigest(),
                "bytes": len(ass_document.encode("utf-8")),
                "source": "actual policy_trace.jsonl sampled at 15 Hz",
            },
            "video_checksum_manifest": str(checksum_path),
        }
        stage_validation = stage_manifests / VIDEO_VALIDATION_NAME
        atomic_write_json(stage_validation, validation_payload)
        stage_checksum = stage_manifests / VIDEO_CHECKSUM_NAME
        write_checksum_manifest(
            [*stage_paths.values(), stage_validation, stage_ass],
            stage_checksum,
            root=staging,
        )

        # Source wrapper manifests, logs, runtime identity, and every nested
        # capture artifact must still be byte-identical after all encoders
        # finish.  The encoders consumed only the stable private copies above.
        for source, role, expected in (
            (fsm.root, "fsm", fsm.managed_run_evidence),
            (ppo.root, "ppo", ppo.managed_run_evidence),
        ):
            current = _validate_source_managed_run(source, role=role)
            if dict(current.evidence) != dict(expected):
                raise PPOVideoArtifactError(
                    f"{role} managed video source changed during publication"
                )

        staged_to_final = [
            *( (stage_paths[name], destinations[name]) for name in destinations ),
            (stage_ass, ass_path),
            (stage_validation, validation_path),
            (stage_checksum, checksum_path),
        ]
        for staged, final in staged_to_final:
            _publish_no_replace(staged, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    checksum_verification = verify_checksum_manifest(checksum_path, root=output)
    if checksum_verification.get("valid") is not True:
        raise PPOVideoArtifactError("published video checksum verification failed")
    return FinalVideoPublication(
        videos=destinations,
        validation_path=validation_path,
        checksum_path=checksum_path,
        diagnostic_ass_path=ass_path,
        checksum_verification=checksum_verification,
    )


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _declared_record(
    payload: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    bytes_key: str,
    expected_path: Path,
    label: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    declared = _require_no_reparse_components(
        str(payload.get(path_key, "")), label=label
    )
    if declared != expected_path:
        raise PPOVideoArtifactError(f"{label} path differs from the canonical artifact")
    record, _ = _stable_file(declared, label=label, allow_empty=allow_empty)
    if (
        payload.get(hash_key) != record["sha256"]
        or isinstance(payload.get(bytes_key), bool)
        or payload.get(bytes_key) != record["bytes"]
    ):
        raise PPOVideoArtifactError(f"{label} byte record is stale or malformed")
    return record


def _validate_video_checksum_snapshot(
    checksum_path: Path,
    *,
    output_root: Path,
    expected_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    checksum_record, content = _stable_file(
        checksum_path,
        label="video checksum manifest",
        keep_bytes=True,
    )
    assert content is not None
    try:
        lines = content.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise PPOVideoArtifactError("video checksum manifest is not UTF-8") from exc
    entries: list[dict[str, Any]] = []
    names: list[str] = []
    for index, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise PPOVideoArtifactError(
                f"video checksum manifest line {index} is malformed"
            )
        expected_hash, name = match.groups()
        raw = Path(name)
        if raw.is_absolute() or ".." in raw.parts or raw.as_posix() != name:
            raise PPOVideoArtifactError("video checksum manifest contains an unsafe path")
        candidate = _require_no_reparse_components(
            output_root / raw, label=f"video checksum entry {name}"
        )
        try:
            candidate.relative_to(output_root)
        except ValueError as exc:
            raise PPOVideoArtifactError("video checksum entry escapes the output root") from exc
        record, _ = _stable_file(candidate, label=f"video checksum entry {name}")
        if expected_hash != record["sha256"]:
            raise PPOVideoArtifactError(f"video checksum mismatch for {name}")
        declared_record = expected_records.get(name)
        if declared_record is None or dict(declared_record) != record:
            raise PPOVideoArtifactError(
                f"video checksum entry {name} differs from independently captured bytes"
            )
        names.append(name)
        entries.append(
            {
                "path": name,
                "expected_sha256": expected_hash,
                "actual_sha256": record["sha256"],
                "valid": True,
            }
        )
    if names != sorted(set(names)) or set(names) != set(expected_records):
        raise PPOVideoArtifactError(
            "video checksum manifest does not cover exactly the final video bundle"
        )
    return checksum_record, {
        "schema": "wlr50_clean.ppo_checksums.v1",
        "manifest": str(checksum_path),
        "valid": True,
        "entries": entries,
    }


def _validate_final_publication_run(
    publication_evidence: Mapping[str, Any],
    *,
    fsm: SourceEpisode,
    ppo: SourceEpisode,
    output_root: Path,
    validation_record: Mapping[str, Any],
    checksum_record: Mapping[str, Any],
    checksum_verification: Mapping[str, Any],
    diagnostic_record: Mapping[str, Any],
    video_records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    expected_reservation_keys = {
        "run_directory",
        "run_kind",
        "started_manifest",
        "committed_runtime_identity_before",
        "frozen_hashes_before",
        "frozen_manifest",
        "git_commit",
        "committed_runtime_content_sha256",
        "reservation_validated_before_encoding",
    }
    if (
        set(publication_evidence) != expected_reservation_keys
        or publication_evidence.get("run_kind") != "video-publication"
        or publication_evidence.get("reservation_validated_before_encoding") is not True
    ):
        raise PPOVideoArtifactError("video publication reservation evidence is incomplete")
    run_dir = _require_no_reparse_components(
        str(publication_evidence.get("run_directory", "")),
        label="video publication run directory",
    )
    runs_root = _PROJECT_ROOT / _RUNS_ROOT_RELATIVE
    try:
        relative = run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise PPOVideoArtifactError(
            "video publication run is outside the canonical managed runs root"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[0] != "video-publication":
        raise PPOVideoArtifactError("video publication run has a noncanonical location")

    label = "video publication managed run"
    started_record, started = _stable_json(
        run_dir / "run_manifest.started.json", label=f"{label} started manifest"
    )
    identity, invocation, config_paths = _validate_started_manifest(
        run_dir,
        started,
        run_kind="video-publication",
        training_stage="video-publication-offline",
        subcommand="publish-videos",
        label=label,
    )
    _validate_publication_invocation(
        invocation,
        fsm_source_dir=fsm.root,
        ppo_source_dir=ppo.root,
        output_root=output_root,
        config_paths=config_paths,
        label=label,
    )
    run_record, final = _stable_json(
        run_dir / "run_manifest.json", label=f"{label} final manifest"
    )
    completed = _parse_utc(final.get("completed_at_utc"), label=f"{label} completion")
    started_at = _parse_utc(identity.get("timestamp_utc"), label=f"{label} start")
    if (
        final.get("schema") != RUN_MANIFEST_SCHEMA
        or final.get("lifecycle") != "SUCCEEDED"
        or final.get("exit_code") != 0
        or completed <= started_at
        or any(
            final.get(key) != value
            for key, value in started.items()
            if key != "lifecycle"
        )
    ):
        raise PPOVideoArtifactError("video publication run did not finalize successfully")
    _record_matches(
        final.get("started_manifest"),
        started_record,
        expected_relative_path="run_manifest.started.json",
        label=f"{label} started manifest",
    )
    captured_artifacts, log_records, stdout_content = _validate_final_run_inventory(
        run_dir, final, label=label
    )
    expected_artifacts = {
        "committed_runtime_identity.before.json",
        "committed_runtime_identity.after.json",
        "frozen_hashes.before.json",
        "frozen_hashes.after.json",
        "final_video_publication.json",
    }
    if set(captured_artifacts) != expected_artifacts:
        raise PPOVideoArtifactError(
            "video publication run artifact inventory is not the exact wrapper contract"
        )
    runtime_identity, runtime_records = _validate_runtime_pair(
        run_dir,
        artifacts=final["artifacts"],
        expected_git_commit=str(identity["git_commit"]),
        label=label,
        require_after=True,
    )
    if (
        dict(runtime_identity) != dict(fsm.committed_runtime_identity)
        or dict(runtime_identity) != dict(ppo.committed_runtime_identity)
    ):
        raise PPOVideoArtifactError(
            "video publication and source capture committed runtimes differ"
        )
    frozen_manifest_record, frozen_records = _validate_frozen_pair(
        run_dir,
        artifacts=final["artifacts"],
        label=label,
        require_after=True,
    )
    current_reservation = {
        "run_directory": str(run_dir),
        "run_kind": "video-publication",
        "started_manifest": started_record,
        "committed_runtime_identity_before": runtime_records[0],
        "frozen_hashes_before": frozen_records[0],
        "frozen_manifest": frozen_manifest_record,
        "git_commit": runtime_identity["git_commit"],
        "committed_runtime_content_sha256": runtime_identity["content_sha256"],
        "reservation_validated_before_encoding": True,
    }
    if dict(publication_evidence) != current_reservation:
        raise PPOVideoArtifactError(
            "video publication reservation changed after video validation was written"
        )

    result_record, result = _stable_json(
        run_dir / "final_video_publication.json", label=f"{label} publication result"
    )
    _record_matches(
        final["artifacts"].get("final_video_publication.json"),
        result_record,
        expected_relative_path="final_video_publication.json",
        label=f"{label} publication result",
    )
    required_result_keys = {
        "schema",
        "offline",
        "isaac_started",
        "seed",
        "publication_run_directory",
        "fsm_source_directory",
        "ppo_source_directory",
        "videos",
        "video_records",
        "video_validation",
        "video_validation_sha256",
        "video_validation_bytes",
        "video_checksums",
        "video_checksums_sha256",
        "video_checksums_bytes",
        "diagnostic_ass",
        "diagnostic_ass_sha256",
        "diagnostic_ass_bytes",
        "checksum_verification",
    }
    expected_paths = {name: record["path"] for name, record in video_records.items()}
    if (
        set(result) != required_result_keys
        or result.get("schema") != _PUBLICATION_CLI_SCHEMA
        or result.get("offline") is not True
        or result.get("isaac_started") is not False
        or result.get("seed") != 4001
        or _absolute(str(result.get("publication_run_directory", ""))) != run_dir
        or _absolute(str(result.get("fsm_source_directory", ""))) != fsm.root
        or _absolute(str(result.get("ppo_source_directory", ""))) != ppo.root
        or result.get("videos") != expected_paths
        or result.get("video_records") != _json_value(video_records)
        or result.get("video_validation") != validation_record["path"]
        or result.get("video_validation_sha256") != validation_record["sha256"]
        or result.get("video_validation_bytes") != validation_record["bytes"]
        or result.get("video_checksums") != checksum_record["path"]
        or result.get("video_checksums_sha256") != checksum_record["sha256"]
        or result.get("video_checksums_bytes") != checksum_record["bytes"]
        or result.get("diagnostic_ass") != diagnostic_record["path"]
        or result.get("diagnostic_ass_sha256") != diagnostic_record["sha256"]
        or result.get("diagnostic_ass_bytes") != diagnostic_record["bytes"]
        or result.get("checksum_verification") != _json_value(checksum_verification)
    ):
        raise PPOVideoArtifactError(
            "managed publication result does not bind the exact final video bundle"
        )
    stdout_rows = _stdout_json_objects(stdout_content, label=label)
    publication_rows = [
        row for row in stdout_rows if row.get("schema") == _PUBLICATION_CLI_SCHEMA
    ]
    if len(publication_rows) != 1 or dict(publication_rows[0]) != dict(result):
        raise PPOVideoArtifactError(
            "video publication stdout does not bind the exact publication result"
        )
    audit_rows = {
        _absolute(str(row.get("audit", "")))
        for row in stdout_rows
        if row.get("passed") is True and row.get("audit")
    }
    if not {
        run_dir / "frozen_hashes.before.json",
        run_dir / "frozen_hashes.after.json",
    }.issubset(audit_rows):
        raise PPOVideoArtifactError(
            "video publication stdout omits frozen before/after results"
        )
    return {
        "run_directory": str(run_dir),
        "run_manifest": run_record,
        "started_manifest": started_record,
        "publication_result": result_record,
        "stdout": log_records["stdout.log"],
        "stderr": log_records["stderr.log"],
        "committed_runtime_identity_before": runtime_records[0],
        "committed_runtime_identity_after": runtime_records[1],
        "frozen_hashes_before": frozen_records[0],
        "frozen_hashes_after": frozen_records[1],
        "frozen_manifest": frozen_manifest_record,
        "git_commit": runtime_identity["git_commit"],
        "committed_runtime_content_sha256": runtime_identity["content_sha256"],
        "stdout_publication_result_exact": True,
    }


def verify_final_video_publication(
    validation_path: Path | str,
    checksum_path: Path | str,
    *,
    output_root: Path | str,
    expected_improved_checkpoint_sha256: str,
    ffmpeg: Path | str | None = None,
) -> Mapping[str, Any]:
    """Re-prove source, encoding, checksum, and managed-wrapper video evidence."""

    improved_hash = _strict_hash(
        expected_improved_checkpoint_sha256,
        label="expected improved checkpoint SHA-256",
    )
    try:
        executable = find_ffmpeg(ffmpeg)
    except (FileNotFoundError, OSError) as exc:
        raise PPOVideoArtifactError("ffmpeg is unavailable for final video verification") from exc
    root = _require_no_reparse_components(output_root, label="video output root")
    expected_validation = root / "manifests" / VIDEO_VALIDATION_NAME
    expected_checksum = root / "manifests" / VIDEO_CHECKSUM_NAME
    selected_validation = _require_no_reparse_components(
        validation_path, label="video validation manifest"
    )
    selected_checksum = _require_no_reparse_components(
        checksum_path, label="video checksum manifest"
    )
    if selected_validation != expected_validation or selected_checksum != expected_checksum:
        raise PPOVideoArtifactError(
            "video validation/checksum paths are not the canonical output artifacts"
        )
    validation_record, payload = _stable_json(
        selected_validation, label="video validation manifest"
    )
    if (
        payload.get("schema") != _PUBLICATION_SCHEMA
        or payload.get("valid") is not True
        or payload.get("status") != "PASS"
        or payload.get("immutable_no_overwrite") is not True
        or payload.get("fps") != VIDEO_FPS
        or payload.get("maximum_duration_s") != 200.0
    ):
        raise PPOVideoArtifactError("video validation manifest did not pass")

    sources = payload.get("source_episodes")
    if not isinstance(sources, Mapping) or set(sources) != {"fsm", "ppo"}:
        raise PPOVideoArtifactError("video source episode evidence is incomplete")
    source_objects: dict[str, SourceEpisode] = {}
    source_file_records: dict[str, list[dict[str, Any]]] = {}
    for role in ("fsm", "ppo"):
        source_row = sources.get(role)
        if not isinstance(source_row, Mapping):
            raise PPOVideoArtifactError(f"{role} source episode record is invalid")
        source = _load_source_episode(
            str(source_row.get("directory", "")), role=role, ffmpeg=executable
        )
        source_objects[role] = source
        if source_row.get("managed_run") != _json_value(source.managed_run_evidence):
            raise PPOVideoArtifactError(
                f"{role} source managed-run evidence changed after publication"
            )
        records: list[dict[str, Any]] = []
        for key, expected_path in (
            ("source_manifest", source.manifest_path),
            ("viewport_ledger", source.ledger_path),
            ("policy_trace", source.trace_path),
            ("raw_video", source.video_path),
        ):
            records.append(
                _declared_record(
                    source_row,
                    path_key=key,
                    hash_key=f"{key}_sha256",
                    bytes_key=f"{key}_bytes",
                    expected_path=expected_path,
                    label=f"{role} video {key}",
                )
            )
        source_file_records[role] = records
    fsm = source_objects["fsm"]
    ppo = source_objects["ppo"]
    current_pair = _validate_pair(fsm, ppo)
    if payload.get("pair_evidence") != _json_value(current_pair):
        raise PPOVideoArtifactError("video pair evidence changed after publication")
    ppo_row = sources["ppo"]
    assert isinstance(ppo_row, Mapping) and ppo.checkpoint_path is not None
    checkpoint_record = _declared_record(
        ppo_row,
        path_key="checkpoint",
        hash_key="checkpoint_sha256",
        bytes_key="checkpoint_bytes",
        expected_path=ppo.checkpoint_path,
        label="PPO video improved checkpoint",
    )
    if checkpoint_record["sha256"] != improved_hash:
        raise PPOVideoArtifactError(
            "PPO source video was not rendered from the improved checkpoint"
        )
    checkpoint_manifest_path = _require_no_reparse_components(
        str(ppo.manifest["source_checkpoint_manifest"]),
        label="PPO video checkpoint manifest",
    )
    checkpoint_manifest_record = _declared_record(
        ppo_row,
        path_key="checkpoint_manifest",
        hash_key="checkpoint_manifest_sha256",
        bytes_key="checkpoint_manifest_bytes",
        expected_path=checkpoint_manifest_path,
        label="PPO video checkpoint manifest",
    )

    videos = payload.get("videos")
    video_names = {
        "fsm_baseline": FSM_VIDEO_NAME,
        "ppo_improved": PPO_VIDEO_NAME,
        "comparison": COMPARISON_VIDEO_NAME,
        "ppo_diagnostic": DIAGNOSTIC_VIDEO_NAME,
    }
    if not isinstance(videos, Mapping) or set(videos) != set(video_names):
        raise PPOVideoArtifactError("video validation does not contain exactly four videos")
    expected_frames = {
        "fsm_baseline": fsm.frame_count,
        "ppo_improved": ppo.frame_count,
        "comparison": max(fsm.frame_count, ppo.frame_count),
        "ppo_diagnostic": ppo.frame_count,
    }
    video_records: dict[str, dict[str, Any]] = {}
    for key, filename in video_names.items():
        row = videos.get(key)
        if not isinstance(row, Mapping):
            raise PPOVideoArtifactError(f"video row {key} is malformed")
        path = root / "videos" / filename
        record = _declared_record(
            row,
            path_key="path",
            hash_key="sha256",
            bytes_key="bytes",
            expected_path=path,
            label=f"final video {key}",
        )
        width = 2 * VIDEO_WIDTH if key == "comparison" else VIDEO_WIDTH
        current = validate_mp4(
            path,
            ffmpeg=executable,
            expected_width=width,
            expected_height=VIDEO_HEIGHT,
            expected_fps=VIDEO_FPS,
            expected_frame_count=expected_frames[key],
            maximum_duration_s=200.0,
            stitched=False,
            speed_modified=False,
            require_sane_container_duration=True,
        )
        checked_fields = (
            "sha256",
            "bytes",
            "duration_s",
            "container_duration_s",
            "fps",
            "frame_count",
            "resolution",
            "width",
            "height",
            "codec",
            "pixel_format",
            "full_decode",
            "timestamps_monotonic",
            "timestamps_continuous",
        )
        if (
            current.get("valid") is not True
            or current.get("status") != "PASS"
            or any(row.get(field) != current.get(field) for field in checked_fields)
            or row.get("pix_fmt") != "yuv420p"
            or row.get("monotonic") is not True
            or row.get("stitched") is not False
            or row.get("speed_modified") is not False
            or not (0.0 < float(row.get("duration_s", math.inf)) <= 200.0)
        ):
            raise PPOVideoArtifactError(
                f"final video {key} failed independent full-decode validation"
            )
        if key in {"ppo_improved", "ppo_diagnostic"} and row.get(
            "source_checkpoint_sha256"
        ) != improved_hash:
            raise PPOVideoArtifactError(
                f"final video {key} names a different improved checkpoint"
            )
        video_records[key] = record

    diagnostic = payload.get("diagnostic_ass")
    if not isinstance(diagnostic, Mapping):
        raise PPOVideoArtifactError("video validation omits diagnostic overlay evidence")
    diagnostic_record = _declared_record(
        diagnostic,
        path_key="path",
        hash_key="sha256",
        bytes_key="bytes",
        expected_path=root / "manifests" / DIAGNOSTIC_ASS_NAME,
        label="PPO diagnostic ASS overlay",
    )
    expected_checksum_records = {
        Path(record["path"]).relative_to(root).as_posix(): record
        for record in video_records.values()
    }
    expected_checksum_records[
        selected_validation.relative_to(root).as_posix()
    ] = validation_record
    expected_checksum_records[
        Path(diagnostic_record["path"]).relative_to(root).as_posix()
    ] = diagnostic_record
    checksum_record, checksum_verification = _validate_video_checksum_snapshot(
        selected_checksum,
        output_root=root,
        expected_records=expected_checksum_records,
    )
    if _absolute(str(payload.get("video_checksum_manifest", ""))) != selected_checksum:
        raise PPOVideoArtifactError("video validation names a different checksum manifest")
    publication_evidence = payload.get("publication_run")
    if not isinstance(publication_evidence, Mapping):
        raise PPOVideoArtifactError("video validation omits its managed publication run")
    publication_run = _validate_final_publication_run(
        publication_evidence,
        fsm=fsm,
        ppo=ppo,
        output_root=root,
        validation_record=validation_record,
        checksum_record=checksum_record,
        checksum_verification=checksum_verification,
        diagnostic_record=diagnostic_record,
        video_records=video_records,
    )
    for role, source in source_objects.items():
        current = _validate_source_managed_run(source.root, role=role)
        if dict(current.evidence) != dict(source.managed_run_evidence):
            raise PPOVideoArtifactError(
                f"{role} source changed during final video verification"
            )
    current_validation_record, _ = _stable_file(
        selected_validation, label="video validation manifest"
    )
    if current_validation_record != validation_record:
        raise PPOVideoArtifactError("video validation changed during verification")
    return {
        "valid": True,
        "status": "PASS",
        "validation": validation_record,
        "video_checksums": checksum_record,
        "video_checksum_verification": checksum_verification,
        "videos": video_records,
        "source_episodes": {
            role: {
                "directory": str(source_objects[role].root),
                "managed_run": dict(source_objects[role].managed_run_evidence),
                "files": source_file_records[role],
            }
            for role in ("fsm", "ppo")
        },
        "improved_checkpoint": checkpoint_record,
        "improved_checkpoint_manifest": checkpoint_manifest_record,
        "diagnostic_overlay": diagnostic_record,
        "publication_run": publication_run,
    }


__all__ = [
    "COMPARISON_VIDEO_NAME",
    "DIAGNOSTIC_VIDEO_NAME",
    "FSM_VIDEO_NAME",
    "FinalVideoPublication",
    "PPOVideoArtifactError",
    "PPO_VIDEO_NAME",
    "VIDEO_CHECKSUM_NAME",
    "VIDEO_VALIDATION_NAME",
    "build_diagnostic_ass",
    "publish_final_videos",
    "verify_final_video_publication",
    "verify_video_source_managed_run",
]
