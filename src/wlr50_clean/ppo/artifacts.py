"""Fail-closed artifact utilities for phase-residual PPO runs.

This module deliberately has no Isaac, Torch, or RL-library imports.  It owns
the small amount of infrastructure that every PPO entry point needs:

* content-addressed, non-reusable run directories;
* exclusive/atomic JSON and CSV publication;
* SHA-256 manifests and verification; and
* independent ffprobe metadata inspection plus a complete ffmpeg decode.

The run directory is an immutable identity.  A run may write new files inside
that directory, but it may never reuse an existing directory or replace either
of its lifecycle manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_MANIFEST_SCHEMA = "wlr50_clean.ppo_run_manifest.v1"
CHECKSUM_MANIFEST_SCHEMA = "wlr50_clean.ppo_checksums.v1"
VIDEO_VALIDATION_SCHEMA = "wlr50_clean.ppo_video_validation.v1"
FROZEN_HASH_AUDIT_SCHEMA = "wlr50_clean.frozen_fsm_hash_audit.v1"
DEFAULT_RUNS_RELATIVE = Path("runs") / "ppo_phase_v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TOKEN = re.compile(r"[^a-z0-9]+")


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be created or trusted."""


def _safe_token(value: str, *, label: str) -> str:
    token = _SAFE_TOKEN.sub("-", str(value).strip().lower()).strip("-")
    if not token:
        raise ArtifactError(f"{label} must contain an alphanumeric character")
    return token


def _utc_text(value: datetime | None = None) -> str:
    instant = value or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    instant = instant.astimezone(timezone.utc)
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp_token(value: str) -> str:
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactError(f"invalid UTC timestamp: {value}") from exc
    if instant.tzinfo is None:
        raise ArtifactError("run timestamp must include a timezone")
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path | str) -> str:
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"artifact file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path | str, *, relative_to: Path | str | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ArtifactError(f"artifact file is missing: {source}")
    if relative_to is None:
        display_path = str(source)
    else:
        root = Path(relative_to).resolve()
        try:
            display_path = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ArtifactError(f"artifact is outside checksum root: {source}") from exc
    return {
        "path": display_path,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def git_head(project_root: Path | str) -> str:
    root = Path(project_root).resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or _HEX40.fullmatch(commit) is None:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactError(f"cannot resolve git HEAD for {root}: {detail}")
    return commit


def _relative_project_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ArtifactError(f"config must be inside project root: {path}") from exc


def config_set_record(
    paths: Sequence[Path | str], *, project_root: Path | str
) -> tuple[str, list[dict[str, Any]]]:
    """Hash a set of configs, including their project-relative names.

    Including names prevents two differently routed configurations with equal
    bytes from receiving the same run identity.
    """

    root = Path(project_root).resolve()
    if not paths:
        raise ArtifactError("at least one explicit config is required")
    resolved: list[tuple[str, Path]] = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise ArtifactError(f"config file is missing: {path}")
        resolved.append((_relative_project_path(path, root), path))
    names = [name for name, _ in resolved]
    if len(names) != len(set(names)):
        raise ArtifactError("config list contains duplicate paths")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for name, path in sorted(resolved):
        content = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        records.append(
            {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    return digest.hexdigest(), records


def _atomic_bytes(path: Path | str, payload: bytes, *, replace: bool = False) -> Path:
    """Publish bytes atomically; exclusive publication is the default."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise ArtifactError(f"refusing to overwrite artifact: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, destination)
        elif os.name == "nt":
            # MoveFileEx without MOVEFILE_REPLACE_EXISTING is atomic and fails
            # when another process wins publication of the same destination.
            os.rename(temporary, destination)
        else:
            # POSIX rename replaces an existing file, so use an atomic hard-link
            # publication to retain no-clobber semantics.
            os.link(temporary, destination)
            temporary.unlink()
    except FileExistsError as exc:
        raise ArtifactError(f"refusing to overwrite artifact: {destination}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_json(
    path: Path | str, payload: Any, *, replace: bool = False
) -> Path:
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"JSON payload is not serializable: {exc}") from exc
    return _atomic_bytes(path, encoded, replace=replace)


def atomic_write_csv(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    replace: bool = False,
) -> Path:
    columns = tuple(str(name) for name in fieldnames)
    if not columns or len(columns) != len(set(columns)):
        raise ArtifactError("CSV fieldnames must be non-empty and unique")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    try:
        for row in rows:
            writer.writerow(dict(row))
    except (ValueError, csv.Error) as exc:
        raise ArtifactError(f"invalid CSV row: {exc}") from exc
    return _atomic_bytes(path, stream.getvalue().encode("utf-8"), replace=replace)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    timestamp_utc: str
    git_commit: str
    config_sha256: str
    seed: int
    environment_count: int
    training_stage: str

    def __post_init__(self) -> None:
        commit = self.git_commit.lower()
        config_hash = self.config_sha256.lower()
        if _HEX40.fullmatch(commit) is None:
            raise ArtifactError("git_commit must be a full 40-character SHA-1")
        if re.fullmatch(r"[0-9a-f]{64}", config_hash) is None:
            raise ArtifactError("config_sha256 must be a full SHA-256")
        if int(self.seed) < 0:
            raise ArtifactError("seed must be non-negative")
        if int(self.environment_count) <= 0:
            raise ArtifactError("environment_count must be positive")
        object.__setattr__(self, "git_commit", commit)
        object.__setattr__(self, "config_sha256", config_hash)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "environment_count", int(self.environment_count))
        object.__setattr__(self, "training_stage", _safe_token(self.training_stage, label="training_stage"))
        _timestamp_token(self.timestamp_utc)

    @property
    def run_id(self) -> str:
        return (
            f"{_timestamp_token(self.timestamp_utc)}"
            f"_g{self.git_commit[:12]}"
            f"_c{self.config_sha256[:12]}"
            f"_s{self.seed}"
            f"_n{self.environment_count}"
            f"_{self.training_stage}"
        )


@dataclass(frozen=True, slots=True)
class RunReservation:
    run_id: str
    run_dir: Path
    started_manifest: Path
    identity: RunIdentity


def reserve_run(
    *,
    project_root: Path | str,
    run_kind: str,
    config_paths: Sequence[Path | str],
    seed: int,
    environment_count: int,
    training_stage: str,
    timestamp: datetime | None = None,
    git_commit: str | None = None,
    entrypoint: str = "wlr50_clean.ppo.cli",
    subcommand: str | None = None,
    invocation_arguments: Sequence[str] = (),
) -> RunReservation:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ArtifactError(f"project root does not exist: {root}")
    config_hash, config_records = config_set_record(config_paths, project_root=root)
    identity = RunIdentity(
        timestamp_utc=_utc_text(timestamp),
        git_commit=git_commit or git_head(root),
        config_sha256=config_hash,
        seed=seed,
        environment_count=environment_count,
        training_stage=training_stage,
    )
    kind = _safe_token(run_kind, label="run_kind")
    run_parent = root / DEFAULT_RUNS_RELATIVE / kind
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = run_parent / identity.run_id
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise ArtifactError(f"immutable run directory already exists: {run_dir}") from exc
    started = run_dir / "run_manifest.started.json"
    payload = {
        "schema": RUN_MANIFEST_SCHEMA,
        "lifecycle": "STARTED",
        "immutable_run_directory": True,
        "run_id": identity.run_id,
        "run_kind": kind,
        "run_dir": str(run_dir),
        "project_root": str(root),
        "identity": asdict(identity),
        "configs": config_records,
        "entrypoint": entrypoint,
        "subcommand": subcommand,
        "invocation_arguments": [str(value) for value in invocation_arguments],
    }
    try:
        atomic_write_json(started, payload)
    except Exception:
        # The directory is intentionally retained: silently reusing an identity
        # after a partial reservation would violate provenance.
        raise
    return RunReservation(identity.run_id, run_dir, started, identity)


def finalize_run(run_dir: Path | str, *, exit_code: int) -> Path:
    directory = Path(run_dir).resolve()
    started_path = directory / "run_manifest.started.json"
    final_path = directory / "run_manifest.json"
    if not started_path.is_file():
        raise ArtifactError(f"run has no started manifest: {directory}")
    if final_path.exists():
        raise ArtifactError(f"run is already finalized: {directory}")
    started = json.loads(started_path.read_text(encoding="utf-8"))
    if started.get("schema") != RUN_MANIFEST_SCHEMA or started.get("run_id") != directory.name:
        raise ArtifactError("started manifest does not match the run directory")
    logs: dict[str, Any] = {}
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        logs[name] = file_record(path, relative_to=directory) if path.is_file() else None
    excluded = {
        "run_manifest.started.json",
        "run_manifest.json",
        "stdout.log",
        "stderr.log",
    }
    artifact_files = {
        source.relative_to(directory).as_posix(): file_record(
            source, relative_to=directory
        )
        for source in sorted(directory.rglob("*"), key=lambda value: value.as_posix())
        if source.is_file()
        and source.relative_to(directory).as_posix() not in excluded
    }
    payload = {
        **started,
        "lifecycle": "SUCCEEDED" if int(exit_code) == 0 else "FAILED",
        "completed_at_utc": _utc_text(),
        "exit_code": int(exit_code),
        "started_manifest": file_record(started_path, relative_to=directory),
        "logs": logs,
        "artifacts": artifact_files,
    }
    return atomic_write_json(final_path, payload)


def write_checksum_manifest(
    paths: Sequence[Path | str],
    output_path: Path | str,
    *,
    root: Path | str,
) -> Path:
    base = Path(root).resolve()
    destination = Path(output_path).resolve()
    unique: dict[str, Path] = {}
    for value in paths:
        source = Path(value).resolve()
        if source == destination:
            raise ArtifactError("checksum manifest cannot checksum itself")
        relative = _relative_project_path(source, base)
        if relative in unique:
            raise ArtifactError(f"duplicate checksum path: {relative}")
        if not source.is_file():
            raise ArtifactError(f"artifact file is missing: {source}")
        unique[relative] = source
    if not unique:
        raise ArtifactError("checksum manifest requires at least one file")
    lines = [f"{sha256_file(unique[name])}  {name}" for name in sorted(unique)]
    return _atomic_bytes(destination, ("\n".join(lines) + "\n").encode("utf-8"))


def verify_checksum_manifest(path: Path | str, *, root: Path | str) -> dict[str, Any]:
    manifest = Path(path)
    base = Path(root).resolve()
    if not manifest.is_file():
        raise ArtifactError(f"checksum manifest is missing: {manifest}")
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ArtifactError(f"invalid checksum line {line_number}")
        expected, relative = match.groups()
        candidate = (base / Path(relative)).resolve()
        _relative_project_path(candidate, base)
        actual = sha256_file(candidate)
        entries.append(
            {"path": relative, "expected_sha256": expected, "actual_sha256": actual, "valid": actual == expected}
        )
    if not entries:
        raise ArtifactError("checksum manifest is empty")
    return {
        "schema": CHECKSUM_MANIFEST_SCHEMA,
        "manifest": str(manifest.resolve()),
        "valid": all(entry["valid"] for entry in entries),
        "entries": entries,
    }


def verify_frozen_hashes(
    *,
    project_root: Path | str,
    frozen_manifest: Path | str,
) -> dict[str, Any]:
    """Re-hash every protected FSM file and return explicit pass/fail evidence."""

    root = Path(project_root).resolve()
    manifest_path = Path(frozen_manifest).resolve()
    if not root.is_dir():
        raise ArtifactError(f"project root does not exist: {root}")
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise ArtifactError("frozen hash manifest must be inside the project root") from exc
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"frozen hash manifest is missing: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"frozen hash manifest is invalid JSON: {exc.msg}") from exc
    protected = manifest.get("protected_files") if isinstance(manifest, Mapping) else None
    if not isinstance(protected, Mapping) or not protected:
        raise ArtifactError("frozen hash manifest has no protected_files")

    entries: list[dict[str, Any]] = []
    for raw_relative, raw_expected in sorted(protected.items()):
        relative = str(raw_relative)
        expected = str(raw_expected).lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ArtifactError(f"invalid frozen SHA-256 for {relative}")
        candidate = (root / Path(relative)).resolve()
        _relative_project_path(candidate, root)
        exists = candidate.is_file()
        actual = sha256_file(candidate) if exists else None
        entries.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": exists,
                "valid": exists and actual == expected,
            }
        )
    return {
        "schema": FROZEN_HASH_AUDIT_SCHEMA,
        "checked_at_utc": _utc_text(),
        "project_root": str(root),
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_sha256": sha256_file(manifest_path),
        "source_head": manifest.get("source_head"),
        "protected_file_count": len(entries),
        "passed": all(entry["valid"] for entry in entries),
        "mismatches": [entry["path"] for entry in entries if not entry["valid"]],
        "entries": entries,
    }


def _find_executable(explicit: Path | str | None, name: str) -> str:
    if explicit is not None:
        candidate = Path(explicit).resolve()
        if not candidate.is_file():
            raise ArtifactError(f"{name} executable is missing: {candidate}")
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise ArtifactError(f"{name} executable was not found on PATH")


def _fraction(value: Any) -> float:
    text = str(value or "0/1")
    try:
        result = float(Fraction(text))
    except (ValueError, ZeroDivisionError) as exc:
        raise ArtifactError(f"invalid frame rate reported by ffprobe: {text}") from exc
    return result if math.isfinite(result) else 0.0


def _ffprobe_json(path: Path, executable: str) -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-show_frames",
        "-show_entries",
        (
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,duration:format=duration,format_name:"
            "frame=best_effort_timestamp_time,pkt_dts_time"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace", check=False, timeout=240
    )
    if completed.returncode != 0:
        raise ArtifactError(f"ffprobe failed ({completed.returncode}): {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ArtifactError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("ffprobe payload is not an object")
    return payload


def _full_decode(path: Path, executable: str) -> tuple[bool, int, str]:
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "null",
            os.devnull,
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=240,
    )
    return completed.returncode == 0, completed.returncode, completed.stderr.strip()


def validate_video(
    path: Path | str,
    *,
    source_episode: str,
    source_checkpoint: str,
    source_seed: int,
    stitched: bool,
    speed_modified: bool,
    ffprobe: Path | str | None = None,
    ffmpeg: Path | str | None = None,
    maximum_duration_s: float = 200.0,
    expected_fps: float | None = None,
    expected_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Inspect metadata with ffprobe and independently decode every frame.

    Failures are represented in the returned record so callers can always
    publish negative evidence.  ``require_valid_video`` converts that result to
    an exception for promotion gates.
    """

    source = Path(path).resolve()
    result: dict[str, Any] = {
        "schema": VIDEO_VALIDATION_SCHEMA,
        "path": str(source),
        "valid": False,
        "status": "VIDEO_OR_ARTIFACT_ERROR",
        "source_episode": str(source_episode),
        "source_checkpoint": str(source_checkpoint),
        "source_seed": int(source_seed),
        "stitched": bool(stitched),
        "speed_modified": bool(speed_modified),
        "full_decode": False,
        "timestamps_monotonic": False,
        "maximum_duration_s": float(maximum_duration_s),
    }
    if not source.is_file() or source.stat().st_size <= 0:
        result["error"] = "video file is missing or empty"
        return result
    try:
        probe_executable = _find_executable(ffprobe, "ffprobe")
        decode_executable = _find_executable(ffmpeg, "ffmpeg")
        payload = _ffprobe_json(source, probe_executable)
        streams = payload.get("streams") or []
        if len(streams) != 1 or not isinstance(streams[0], Mapping):
            raise ArtifactError("ffprobe did not report exactly one selected video stream")
        stream = streams[0]
        frames = payload.get("frames") or []
        timestamps: list[float] = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            raw = frame.get("best_effort_timestamp_time", frame.get("pkt_dts_time"))
            if raw not in (None, "N/A"):
                timestamp = float(raw)
                if not math.isfinite(timestamp):
                    raise ArtifactError("ffprobe returned a non-finite frame timestamp")
                timestamps.append(timestamp)
        fps = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        format_row = payload.get("format") if isinstance(payload.get("format"), Mapping) else {}
        duration_raw = stream.get("duration") or format_row.get("duration") or 0.0
        duration = float(duration_raw)
        if (not math.isfinite(duration) or duration <= 0.0) and timestamps and fps > 0.0:
            duration = timestamps[-1] - timestamps[0] + 1.0 / fps
        frame_count = len(timestamps)
        reported_count = stream.get("nb_frames")
        reported_frame_count = int(reported_count) if str(reported_count).isdigit() else None
        monotonic = frame_count >= 2 and all(b > a for a, b in zip(timestamps, timestamps[1:]))
        full_decode, decode_returncode, decode_error = _full_decode(source, decode_executable)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        codec = str(stream.get("codec_name") or "").lower()
        pixel_format = str(stream.get("pix_fmt") or "").lower()
        formats = {part.strip().lower() for part in str(format_row.get("format_name") or "").split(",")}
        fps_valid = expected_fps is None or math.isclose(
            fps, float(expected_fps), rel_tol=0.0, abs_tol=0.01
        )
        resolution_valid = expected_resolution is None or (width, height) == tuple(expected_resolution)
        frame_count_valid = frame_count >= 2 and (
            reported_frame_count is None or reported_frame_count == frame_count
        )
        duration_valid = math.isfinite(duration) and 0.0 < duration <= float(maximum_duration_s)
        valid = bool(
            source.suffix.lower() == ".mp4"
            and "mp4" in formats
            and codec == "h264"
            and pixel_format == "yuv420p"
            and width > 0
            and height > 0
            and fps > 0.0
            and fps_valid
            and resolution_valid
            and frame_count_valid
            and duration_valid
            and monotonic
            and full_decode
            and not stitched
            and not speed_modified
        )
        result.update(
            {
                "valid": valid,
                "status": "PASS" if valid else "VIDEO_OR_ARTIFACT_ERROR",
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "duration_s": duration,
                "fps": fps,
                "frame_count": frame_count,
                "reported_frame_count": reported_frame_count,
                "resolution": [width, height],
                "width": width,
                "height": height,
                "codec": codec,
                "pixel_format": pixel_format,
                "container_formats": sorted(formats),
                "full_decode": full_decode,
                "decode_returncode": decode_returncode,
                "decode_error": decode_error,
                "timestamps_monotonic": monotonic,
                "first_timestamp_s": timestamps[0] if timestamps else None,
                "last_timestamp_s": timestamps[-1] if timestamps else None,
                "frame_timestamps_sha256": hashlib.sha256(
                    "\n".join(format(value, ".17g") for value in timestamps).encode("ascii")
                ).hexdigest(),
                "frame_count_valid": frame_count_valid,
                "duration_valid": duration_valid,
                "fps_valid": fps_valid,
                "resolution_valid": resolution_valid,
                "ffprobe_path": probe_executable,
                "ffmpeg_path": decode_executable,
                "error": "" if valid else "one or more video acceptance checks failed",
            }
        )
    except Exception as exc:  # preserve negative evidence for a manifest
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def require_valid_video(path: Path | str, **kwargs: Any) -> dict[str, Any]:
    result = validate_video(path, **kwargs)
    if result.get("valid") is not True:
        raise ArtifactError(str(result.get("error") or "video validation failed"))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    reserve = commands.add_parser("reserve-run", help="create a new immutable run directory")
    reserve.add_argument("--project-root", type=Path, required=True)
    reserve.add_argument("--run-kind", required=True)
    reserve.add_argument("--config", type=Path, action="append", required=True)
    reserve.add_argument("--seed", type=int, required=True)
    reserve.add_argument("--environment-count", type=int, required=True)
    reserve.add_argument("--training-stage", required=True)
    reserve.add_argument("--entrypoint", default="wlr50_clean.ppo.cli")
    reserve.add_argument("--subcommand")
    reserve.add_argument("--invocation-argument", action="append", default=[])

    finalize = commands.add_parser("finalize-run", help="seal a run lifecycle manifest")
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)

    checksums = commands.add_parser("write-checksums", help="write an exclusive SHA-256 manifest")
    checksums.add_argument("--root", type=Path, required=True)
    checksums.add_argument("--output", type=Path, required=True)
    checksums.add_argument("--file", type=Path, action="append", required=True)

    frozen = commands.add_parser(
        "verify-frozen", help="verify and save the frozen FSM byte-hash contract"
    )
    frozen.add_argument("--project-root", type=Path, required=True)
    frozen.add_argument("--manifest", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "reserve-run":
            reservation = reserve_run(
                project_root=args.project_root,
                run_kind=args.run_kind,
                config_paths=args.config,
                seed=args.seed,
                environment_count=args.environment_count,
                training_stage=args.training_stage,
                entrypoint=args.entrypoint,
                subcommand=args.subcommand,
                invocation_arguments=args.invocation_argument,
            )
            print(
                json.dumps(
                    {"run_id": reservation.run_id, "run_dir": str(reservation.run_dir)},
                    separators=(",", ":"),
                )
            )
        elif args.command == "finalize-run":
            path = finalize_run(args.run_dir, exit_code=args.exit_code)
            print(json.dumps({"manifest": str(path)}, separators=(",", ":")))
        elif args.command == "write-checksums":
            path = write_checksum_manifest(args.file, args.output, root=args.root)
            print(json.dumps({"manifest": str(path)}, separators=(",", ":")))
        elif args.command == "verify-frozen":
            audit = verify_frozen_hashes(
                project_root=args.project_root,
                frozen_manifest=args.manifest,
            )
            path = atomic_write_json(args.output, audit)
            print(
                json.dumps(
                    {"audit": str(path), "passed": audit["passed"]},
                    separators=(",", ":"),
                )
            )
            if not audit["passed"]:
                return 2
        else:  # pragma: no cover - argparse prevents this
            raise ArtifactError(f"unsupported command: {args.command}")
    except (ArtifactError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"artifact orchestration failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
