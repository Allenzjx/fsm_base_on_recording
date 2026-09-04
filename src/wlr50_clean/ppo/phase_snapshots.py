"""Extract and validate reset-only phase-entry snapshots from a frozen success trial."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    PHYSICS_DT_S,
    SERVO_COMMAND_SIGN,
    SERVO_ORDER,
    WHEEL_ORDER,
    WHEEL_VELOCITY_LIMIT_RAD_S,
    Full12Command,
    build_physical_batch,
    servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper

SNAPSHOT_SCHEMA = "wlr50_clean.ppo_phase_entry_snapshot.v2"
MANIFEST_SCHEMA = "wlr50_clean.ppo_phase_snapshot_manifest.v2"
BUNDLE_RECORD_SCHEMA = "wlr50_clean.ppo_phase_snapshot_bundle_record.v1"
BUNDLE_HASH_SCHEMA = "wlr50_clean.ppo_phase_snapshot_bundle_hash.v1"
SOURCE_COMMAND_SCHEMA = "wlr50_clean.ppo_phase_snapshot_source_command.v1"
SOURCE_MAPPER_STATE_SCHEMA = "wlr50_clean.ppo_phase_snapshot_mapper_state.v1"
PHASE_IDS = tuple(f"P{i:02d}" for i in range(1, 14))
PHYSICS_HZ = 120.0
SOURCE_SETTLE_TICKS = 180
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_SNAPSHOT_ROOT = PROJECT_ROOT / "reference" / "ppo_phase_snapshots"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_COMMAND_FILE = "full12_commands_120hz.jsonl"
SOURCE_OBSERVATION_FILE = "observation_120hz.jsonl"
SOURCE_TRANSITION_FILE = "state_transitions.jsonl"
SOURCE_LEG_CROSSING_FILE = "leg_crossing_events.jsonl"
SOURCE_TRIAL_MANIFEST_FILE = "trial_manifest.json"
SOURCE_ACK_MATCH_FIELDS = (
    "schema",
    "physics_dt_s",
    "articulation_writes_this_call",
    "canonical_order",
    "requested_full12",
    "applied_full12",
    "drive_target_full12",
    "native_drive_target_full12",
    "drive_feedback_bias_requested_full12",
    "drive_feedback_bias_realized_full12",
    "drive_feedback_final_slew_limit_deg_per_tick",
    "command_was_clamped",
    "servo_applied_drive_command_deg",
    "servo_native_drive_command_deg",
    "servo_tracking_compensation_deg",
    "servo_nominal_target_reached",
    "servo_tracking_active",
    "tracking_servo_names",
    "servo_tracking_feedback_sample_tick",
    "servo_tracking_feedback_sampled",
    "servo_joint_ids",
    "wheel_joint_ids",
    "servo_target_physical_rad",
    "wheel_target_physical_rad_s",
    "motion_start_skew_s",
)


class PhaseSnapshotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _PhaseEntryBoundary:
    """Source/target ticks for one reset-only phase-entry replay."""

    source_tick: int
    controller_anchor_tick: int | None
    target_entry_tick: int
    source_replay_steps: int
    uses_causal_predecessor: bool


@dataclass(frozen=True, slots=True)
class PhaseSnapshotFileBuffer:
    """Immutable bytes and validated digests for one phase-entry snapshot."""

    phase: str
    source_tick: int
    source_replay_steps: int
    target_entry_tick: int | None
    controller_anchor_tick: int | None
    controller_anchor_time_s: float | None
    snapshot_path: Path
    checksum_path: Path
    snapshot_bytes: bytes
    checksum_bytes: bytes
    file_sha256: str
    state_sha256: str
    checksum_file_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "source_tick": self.source_tick,
            "source_replay_steps": self.source_replay_steps,
            "target_entry_tick": self.target_entry_tick,
            "controller_anchor_tick": self.controller_anchor_tick,
            "controller_anchor_time_s": self.controller_anchor_time_s,
            "snapshot_path": str(self.snapshot_path),
            "checksum_path": str(self.checksum_path),
            "file_sha256": self.file_sha256,
            "state_sha256": self.state_sha256,
            "checksum_file_sha256": self.checksum_file_sha256,
        }


@dataclass(frozen=True, slots=True)
class ValidatedPhaseSnapshotBundle:
    """One immutable, single-read snapshot-bundle capture."""

    snapshot_root: Path
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str
    snapshots: tuple[PhaseSnapshotFileBuffer, ...]
    bundle_sha256: str
    source_trial: str | None
    filesystem_identity: tuple[tuple[Any, ...], ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_RECORD_SCHEMA,
            "snapshot_root": str(self.snapshot_root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "phase_count": len(self.snapshots),
            "snapshots": [snapshot.as_record() for snapshot in self.snapshots],
            "bundle_sha256": self.bundle_sha256,
            "source_trial": self.source_trial,
        }

    def manifest_payload(self) -> dict[str, Any]:
        return dict(
            _decode_json_object(
                self.manifest_bytes,
                label="phase snapshot manifest",
                path=self.manifest_path,
            )
        )

    def snapshot(self, phase: str) -> PhaseSnapshotFileBuffer:
        selected = next((row for row in self.snapshots if row.phase == phase), None)
        if selected is None:
            raise PhaseSnapshotError(f"validated snapshot bundle lacks {phase}")
        return selected


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _indented_json_lf_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize generated JSON with platform-independent Git-stable LF bytes."""

    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PhaseSnapshotError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _read_file_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_file():
        raise PhaseSnapshotError(f"{label} missing: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PhaseSnapshotError(f"failed to read {label}: {path}: {exc}") from exc


def _decode_json_object(payload: bytes, *, label: str, path: Path) -> Mapping[str, Any]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseSnapshotError(f"invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise PhaseSnapshotError(f"{label} must contain a JSON object: {path}")
    return decoded


def _unredirected_path(path: Path | str, *, label: str) -> Path:
    unresolved = Path(os.path.abspath(os.fspath(Path(path))))
    resolved = unresolved.resolve()
    if unresolved != resolved:
        raise PhaseSnapshotError(f"{label} must not traverse a symlink or reparse redirect")
    return resolved


def _path_identity(path: Path, *, label: str, directory: bool) -> tuple[Any, ...]:
    try:
        status = path.lstat()
    except OSError as exc:
        raise PhaseSnapshotError(f"{label} missing: {path}") from exc
    attributes = int(getattr(status, "st_file_attributes", 0))
    if path.is_symlink() or attributes & 0x400:
        raise PhaseSnapshotError(f"{label} must not be a symlink or reparse point: {path}")
    if directory and not path.is_dir():
        raise PhaseSnapshotError(f"{label} is not a directory: {path}")
    if not directory and not path.is_file():
        raise PhaseSnapshotError(f"{label} is not a regular file: {path}")
    return (
        str(path),
        "directory" if directory else "file",
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
        attributes,
    )


def _handle_identity(path: Path, stream: Any) -> tuple[Any, ...]:
    """Return the same identity shape as ``_path_identity`` for an open file."""

    try:
        status = os.fstat(stream.fileno())
    except (OSError, ValueError) as exc:
        raise PhaseSnapshotError(
            f"source file handle identity is unavailable: {path}"
        ) from exc
    return (
        str(path),
        "file",
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
        int(getattr(status, "st_file_attributes", 0)),
    )


def _same_open_path_file_identity(
    path_identity: tuple[Any, ...], handle_identity: tuple[Any, ...]
) -> bool:
    """Compare stable Windows path/handle fields (``st_ctime`` differs by API)."""

    stable_indices = (0, 1, 2, 3, 4, 5, 7)
    return all(
        path_identity[index] == handle_identity[index]
        for index in stable_indices
    )


def _same_path_identity(
    expected: tuple[Any, ...], current: tuple[Any, ...]
) -> bool:
    """Keep directory-object checks stable across unrelated child changes."""

    if expected[1] != current[1]:
        return False
    if expected[1] != "directory":
        return expected == current
    # Directory size/timestamps are namespace activity, not object identity.
    stable_indices = (0, 1, 2, 3, 7)
    return all(expected[index] == current[index] for index in stable_indices)


def _same_filesystem_identity(
    expected: tuple[tuple[Any, ...], ...],
    current: tuple[tuple[Any, ...], ...],
) -> bool:
    return len(expected) == len(current) and all(
        _same_path_identity(expected_row, current_row)
        for expected_row, current_row in zip(expected, current)
    )


def _capture_source_surface(
    trial_dir: Path | str,
) -> tuple[Path, dict[str, Path], tuple[tuple[Any, ...], ...]]:
    """Pin the unresolved trial ancestry and five non-reparse source files."""

    trial = Path(os.path.abspath(os.fspath(Path(trial_dir))))
    directory_surface = tuple(reversed(trial.parents)) + (trial,)
    identities: list[tuple[Any, ...]] = []
    for index, directory in enumerate(directory_surface):
        identities.append(
            _path_identity(
                directory,
                label=(
                    "source trial root"
                    if directory == trial
                    else f"source trial ancestor {index}"
                ),
                directory=True,
            )
        )
    if trial.resolve() != trial:
        raise PhaseSnapshotError(
            "source trial root must not traverse a symlink or reparse redirect"
        )
    paths = {
        "trial_manifest": trial / SOURCE_TRIAL_MANIFEST_FILE,
        "command": trial / SOURCE_COMMAND_FILE,
        "observation": trial / SOURCE_OBSERVATION_FILE,
        "transition": trial / SOURCE_TRANSITION_FILE,
        "leg_crossing": trial / SOURCE_LEG_CROSSING_FILE,
    }
    for role, path in paths.items():
        identities.append(
            _path_identity(path, label=f"source {role} file", directory=False)
        )
        if path.resolve() != path:
            raise PhaseSnapshotError(
                f"source {role} file must not traverse a symlink or reparse redirect"
            )
    return trial, paths, tuple(identities)


def _assert_source_surface_unchanged(
    identities: Iterable[tuple[Any, ...]],
) -> None:
    """Recheck every captured ancestor, root, and source-file identity."""

    for identity in identities:
        path = Path(identity[0])
        current = _path_identity(
            path,
            label=f"captured source path {path}",
            directory=identity[1] == "directory",
        )
        if not _same_path_identity(identity, current):
            raise PhaseSnapshotError(
                f"source trial path changed during immutable capture: {path}"
            )


def _capture_source_bytes_once(
    path: Path,
    *,
    label: str,
    expected_identity: tuple[Any, ...],
) -> bytes:
    """Read one small source file from one identity-checked binary handle."""

    try:
        with path.open("rb") as stream:
            before = _handle_identity(path, stream)
            if not _same_open_path_file_identity(expected_identity, before):
                raise PhaseSnapshotError(
                    f"{label} changed before its immutable read"
                )
            payload = stream.read()
            if _handle_identity(path, stream) != before:
                raise PhaseSnapshotError(f"{label} changed while it was read")
    except PhaseSnapshotError:
        raise
    except OSError as exc:
        raise PhaseSnapshotError(f"failed to read {label}: {path}: {exc}") from exc
    return payload


class _SingleReadJsonlCapture:
    """One binary JSONL handle whose raw bytes are parsed and hashed once."""

    def __init__(
        self,
        path: Path,
        *,
        label: str,
        expected_identity: tuple[Any, ...],
    ) -> None:
        self.path = path
        self.label = label
        self._expected_identity = expected_identity
        try:
            self._stream = path.open("rb")
        except OSError as exc:
            raise PhaseSnapshotError(f"failed to open {label}: {path}: {exc}") from exc
        self._handle_identity = _handle_identity(path, self._stream)
        if not _same_open_path_file_identity(
            expected_identity, self._handle_identity
        ):
            self._stream.close()
            raise PhaseSnapshotError(f"{label} changed before its immutable read")
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._line_number = 0
        self._finished = False

    def __enter__(self) -> "_SingleReadJsonlCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stream.close()

    def __iter__(self) -> "_SingleReadJsonlCapture":
        return self

    def __next__(self) -> dict[str, Any]:
        while True:
            raw = self._stream.readline()
            if not raw:
                self._finish()
                raise StopIteration
            self._digest.update(raw)
            self._byte_count += len(raw)
            self._line_number += 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PhaseSnapshotError(
                    f"invalid JSONL at {self.path}:{self._line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PhaseSnapshotError(
                    f"non-object JSONL row at {self.path}:{self._line_number}"
                )
            return row

    def drain(self) -> None:
        """Hash the unparsed tail without reopening or retaining the large file."""

        if self._finished:
            return
        for block in iter(lambda: self._stream.read(1024 * 1024), b""):
            self._digest.update(block)
            self._byte_count += len(block)
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        if _handle_identity(self.path, self._stream) != self._handle_identity:
            raise PhaseSnapshotError(f"{self.label} changed while it was read")
        self._finished = True

    def artifact_record(self, expected_name: str) -> dict[str, Any]:
        if not self._finished:
            raise PhaseSnapshotError(
                f"{self.label} hash is unavailable before its immutable read completes"
            )
        return {
            "name": expected_name,
            "bytes": self._byte_count,
            "sha256": self._digest.hexdigest(),
        }


def _require_within(path: Path, root: Path, *, label: str) -> None:
    if path != root and root not in path.parents:
        raise PhaseSnapshotError(f"{label} resolves outside canonical snapshot root")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PhaseSnapshotError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise PhaseSnapshotError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def _read_jsonl_bytes(payload: bytes, path: Path) -> tuple[dict[str, Any], ...]:
    """Parse JSONL solely from already captured immutable bytes."""

    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PhaseSnapshotError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise PhaseSnapshotError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_snapshot_drive_target_sha256(values: Iterable[Any]) -> str:
    """Hash one authoritative physical-drive target in canonical Full12 order."""

    target = _finite_values(values, len(FULL12_ORDER), "drive_target_full12")
    return _sha256_bytes(
        _canonical_bytes(
            {
                "canonical_order": list(FULL12_ORDER),
                "drive_target_full12": list(target),
            }
        )
    )


def phase_snapshot_actuation_contract_sha256(
    expected_atomic_ack: Mapping[str, Any],
) -> str:
    """Hash every source-ack field that the reset-only adapter must reproduce."""

    missing = [name for name in SOURCE_ACK_MATCH_FIELDS if name not in expected_atomic_ack]
    if missing:
        raise PhaseSnapshotError(
            f"source atomic ack lacks replay fields: {missing}"
        )
    return _sha256_bytes(
        _canonical_bytes(
            {name: expected_atomic_ack[name] for name in SOURCE_ACK_MATCH_FIELDS}
        )
    )


def _finite_values(values: Iterable[Any], size: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise PhaseSnapshotError(f"{label} must be a numeric sequence") from exc
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise PhaseSnapshotError(f"{label} must contain {size} finite values")
    return result


def _bool_values(values: Iterable[Any], size: int, label: str) -> tuple[bool, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise PhaseSnapshotError(f"{label} must be a boolean sequence") from exc
    if len(result) != size or any(type(value) is not bool for value in result):
        raise PhaseSnapshotError(f"{label} must contain {size} booleans")
    return tuple(bool(value) for value in result)


def _equivalent(left: Any, right: Any, *, abs_tol: float = 1.0e-12) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=abs_tol
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _equivalent(a, b, abs_tol=abs_tol) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _source_artifact_record(
    trial_manifest: Mapping[str, Any],
    *,
    artifact_key: str,
    expected_name: str,
    captured: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = trial_manifest.get("artifact_files")
    declared = artifacts.get(artifact_key) if isinstance(artifacts, Mapping) else None
    if not isinstance(declared, Mapping):
        raise PhaseSnapshotError(
            f"trial manifest lacks the {artifact_key} source artifact binding"
        )
    if declared.get("path") != expected_name:
        raise PhaseSnapshotError(
            f"trial manifest {artifact_key} path is not {expected_name}"
        )
    if captured.get("name") != expected_name:
        raise PhaseSnapshotError(
            f"captured {artifact_key} file is not {expected_name}"
        )
    byte_count = captured.get("bytes")
    digest = captured.get("sha256")
    if declared.get("bytes") != byte_count or declared.get("sha256") != digest:
        raise PhaseSnapshotError(
            f"trial manifest {artifact_key} hash/size does not match source bytes"
        )
    return {"name": expected_name, "bytes": byte_count, "sha256": digest}


def _source_standing_pose(
    trial_manifest: Mapping[str, Any],
) -> dict[str, float]:
    initialization = trial_manifest.get("environment_initialization")
    records = initialization.get("records") if isinstance(initialization, Mapping) else None
    if not isinstance(records, list) or len(records) != len(SERVO_ORDER):
        raise PhaseSnapshotError(
            "trial manifest lacks all eight authoritative standing-pose records"
        )
    by_name: dict[str, float] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise PhaseSnapshotError("standing-pose record is not an object")
        name = row.get("joint_name")
        if name not in SERVO_ORDER or name in by_name:
            raise PhaseSnapshotError("standing-pose records are duplicate or noncanonical")
        value = _finite_values(
            (row.get("standing_pose_deg"),), 1, f"standing pose {name}"
        )[0]
        by_name[str(name)] = value
    if set(by_name) != set(SERVO_ORDER):
        raise PhaseSnapshotError("standing-pose records do not cover canonical servos")
    return {name: by_name[name] for name in SERVO_ORDER}


def _mapper_configuration(
    mapper: ServoTargetMapper,
) -> dict[str, Any]:
    return {
        "physics_dt_s": mapper.physics_dt_s,
        "servo_rate_deg_s": mapper.servo_rate_deg_s,
        "maximum_delta_deg": mapper.maximum_delta_deg,
        "tracking_gain": mapper.tracking_gain,
        "tracking_limit_deg": mapper.tracking_limit_deg,
        "feedback_interval_ticks": mapper.feedback_interval_ticks,
        "standing_pose_deg": [mapper.standing_pose_deg[name] for name in SERVO_ORDER],
    }


def _mapper_state_payload(
    mapper: ServoTargetMapper,
    final_drive_servo_deg: Mapping[str, float],
    *,
    source_control_physics_tick: int | None,
) -> dict[str, Any]:
    return {
        "schema": SOURCE_MAPPER_STATE_SCHEMA,
        "source_control_physics_tick": source_control_physics_tick,
        "requested_servo_deg": [mapper._requested[name] for name in SERVO_ORDER],
        "applied_drive_command_deg": [mapper._applied[name] for name in SERVO_ORDER],
        "nominal_target_reached": [mapper._nominal_reached[name] for name in SERVO_ORDER],
        "tracking_compensation_deg": [mapper._compensation[name] for name in SERVO_ORDER],
        "tracking_active": [mapper._tracking_active[name] for name in SERVO_ORDER],
        "retiring_stale_bias": [mapper._retiring_stale_bias[name] for name in SERVO_ORDER],
        "feedback_tick": mapper._feedback_tick,
        "final_drive_servo_deg": [final_drive_servo_deg[name] for name in SERVO_ORDER],
    }


def _expected_source_ack(
    *,
    source_ack: Mapping[str, Any],
    requested: Full12Command,
    applied: Full12Command,
    native_drive: Full12Command,
    drive_target: Full12Command,
    requested_bias: tuple[float, ...],
    realized_bias: tuple[float, ...],
    mapping: Any,
    tracking_servo_names: tuple[str, ...],
    physical: Any,
) -> dict[str, Any]:
    return {
        "schema": "wlr50_clean.atomic_full12_ack.v1",
        "physics_dt_s": PHYSICS_DT_S,
        "articulation_writes_this_call": 1,
        "canonical_order": list(FULL12_ORDER),
        "requested_full12": list(requested.to_full12()),
        "applied_full12": list(applied.to_full12()),
        "drive_target_full12": list(drive_target.to_full12()),
        "native_drive_target_full12": list(native_drive.to_full12()),
        "drive_feedback_bias_requested_full12": list(requested_bias),
        "drive_feedback_bias_realized_full12": list(realized_bias),
        "drive_feedback_final_slew_limit_deg_per_tick": mapping.maximum_delta_deg
        if hasattr(mapping, "maximum_delta_deg")
        else 1.25,
        "command_was_clamped": requested != applied,
        "servo_applied_drive_command_deg": list(drive_target.servo_deg),
        "servo_native_drive_command_deg": list(native_drive.servo_deg),
        "servo_tracking_compensation_deg": list(mapping.tracking_compensation_deg),
        "servo_nominal_target_reached": list(mapping.nominal_target_reached),
        "servo_tracking_active": list(mapping.tracking_active),
        "tracking_servo_names": list(tracking_servo_names),
        "servo_tracking_feedback_sample_tick": mapping.feedback_sample_tick,
        "servo_tracking_feedback_sampled": mapping.feedback_sampled,
        "servo_joint_ids": list(source_ack.get("servo_joint_ids", ())),
        "wheel_joint_ids": list(source_ack.get("wheel_joint_ids", ())),
        "servo_target_physical_rad": list(physical.servo_target_rad),
        "wheel_target_physical_rad_s": list(physical.wheel_target_rad_s),
        "motion_start_skew_s": 0.0,
    }


def _source_replay_at_phase_ticks(
    trial_manifest: Mapping[str, Any],
    ticks: set[int],
    *,
    command_rows: Iterable[dict[str, Any]],
    observation_rows: Iterable[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Replay the Isaac-free mapper ledger and retain t-1/t state at entries."""

    if trial_manifest.get("settle_ticks") != SOURCE_SETTLE_TICKS:
        raise PhaseSnapshotError(
            f"source trial settle_ticks must be {SOURCE_SETTLE_TICKS}"
        )
    if float(trial_manifest.get("physics_hz", math.nan)) != PHYSICS_HZ:
        raise PhaseSnapshotError(f"source trial physics_hz must be {PHYSICS_HZ:g}")
    standing_pose = _source_standing_pose(trial_manifest)
    mapper = ServoTargetMapper(standing_pose, physics_dt_s=PHYSICS_DT_S)
    mapper._feedback_tick = SOURCE_SETTLE_TICKS
    final_drive = {name: 0.0 for name in SERVO_ORDER}
    maximum = max(ticks)
    result: dict[int, dict[str, Any]] = {}
    command_rows = iter(command_rows)
    observation_rows = iter(observation_rows)
    for tick in range(maximum + 1):
        try:
            command = next(command_rows)
            observation = next(observation_rows)
        except StopIteration as exc:
            raise PhaseSnapshotError(
                f"source command/observation streams end before tick {tick}"
            ) from exc
        if command.get("control_physics_tick") != tick:
            raise PhaseSnapshotError(f"source command stream is not contiguous at tick {tick}")
        if observation.get("physics_tick") != tick:
            raise PhaseSnapshotError(
                f"source observation stream is not contiguous at tick {tick}"
            )
        source_ack = command.get("atomic_ack")
        if not isinstance(source_ack, Mapping):
            raise PhaseSnapshotError(f"source command tick {tick} lacks atomic_ack")
        pre_state = _mapper_state_payload(
            mapper,
            final_drive,
            source_control_physics_tick=(None if tick == 0 else tick - 1),
        )
        requested = Full12Command.from_full12(source_ack.get("requested_full12", ()))
        applied = requested.clamped()
        command_applied = _finite_values(
            command.get("applied_full12", ()), len(FULL12_ORDER),
            f"source command applied_full12 at tick {tick}",
        )
        if not _equivalent(applied.to_full12(), command_applied):
            raise PhaseSnapshotError(
                f"source command applied_full12 is inconsistent at tick {tick}"
            )
        tracking_names = tuple(str(name) for name in command.get("tracking_servo_names", ()))
        if len(set(tracking_names)) != len(tracking_names) or any(
            name not in SERVO_ORDER for name in tracking_names
        ):
            raise PhaseSnapshotError(f"source tracking set is invalid at tick {tick}")
        joints = observation.get("joints")
        if not isinstance(joints, Mapping) or set(SERVO_ORDER) - set(joints):
            raise PhaseSnapshotError(f"source observation joints are incomplete at tick {tick}")
        measured_physical_rad = tuple(
            math.radians(
                standing_pose[name]
                + SERVO_COMMAND_SIGN[name] * float(joints[name]["position_deg"])
            )
            for name in SERVO_ORDER
        )
        mapping = mapper.advance(
            applied.servo_deg,
            measured_physical_rad,
            tracking_servo_names=tracking_names,
        )
        native_drive = Full12Command(mapping.applied_drive_command_deg, applied.wheel_rad_s)
        requested_bias = _finite_values(
            command.get("drive_feedback_bias_requested_full12", ()),
            len(FULL12_ORDER),
            f"source requested drive bias at tick {tick}",
        )
        final_servo: list[float] = []
        for index, name in enumerate(SERVO_ORDER):
            lower, upper = servo_limits_deg(name)
            value = bounded_drive_feedback_step(
                previous_deg=final_drive[name],
                native_deg=native_drive.servo_deg[index],
                bias_deg=requested_bias[index],
                maximum_delta_deg=mapper.maximum_delta_deg,
                lower_deg=lower,
                upper_deg=upper,
            )
            final_drive[name] = value
            final_servo.append(value)
        final_wheels = tuple(
            max(
                -WHEEL_VELOCITY_LIMIT_RAD_S,
                min(WHEEL_VELOCITY_LIMIT_RAD_S, native + bias),
            )
            for native, bias in zip(
                native_drive.wheel_rad_s,
                requested_bias[len(SERVO_ORDER) :],
                strict=True,
            )
        )
        drive_target = Full12Command(tuple(final_servo), final_wheels)
        realized_bias = tuple(
            final - native
            for final, native in zip(
                drive_target.to_full12(), native_drive.to_full12(), strict=True
            )
        )
        physical = build_physical_batch(drive_target, standing_pose)
        expected_ack = _expected_source_ack(
            source_ack=source_ack,
            requested=requested,
            applied=applied,
            native_drive=native_drive,
            drive_target=drive_target,
            requested_bias=requested_bias,
            realized_bias=realized_bias,
            mapping=mapping,
            tracking_servo_names=tracking_names,
            physical=physical,
        )
        expected_ack["drive_feedback_final_slew_limit_deg_per_tick"] = (
            mapper.maximum_delta_deg
        )
        for field in SOURCE_ACK_MATCH_FIELDS:
            if field not in source_ack or not _equivalent(
                source_ack[field], expected_ack[field]
            ):
                raise PhaseSnapshotError(
                    f"source mapper replay mismatch at tick {tick}: {field}"
                )
        for field in (
            "applied_full12",
            "drive_target_full12",
            "native_drive_target_full12",
            "drive_feedback_bias_requested_full12",
            "drive_feedback_bias_realized_full12",
            "tracking_servo_names",
        ):
            if field not in command or not _equivalent(command[field], source_ack[field]):
                raise PhaseSnapshotError(
                    f"source command/atomic_ack mismatch at tick {tick}: {field}"
                )
        if (
            source_ack.get("physics_tick") != SOURCE_SETTLE_TICKS + tick
            or source_ack.get("write_count") != SOURCE_SETTLE_TICKS + tick + 1
        ):
            raise PhaseSnapshotError(
                f"source atomic clock/write count is inconsistent at tick {tick}"
            )
        post_state = _mapper_state_payload(
            mapper, final_drive, source_control_physics_tick=tick
        )
        if tick in ticks:
            authoritative_ack = {
                field: source_ack[field] for field in SOURCE_ACK_MATCH_FIELDS
            }
            command_row_hash = _sha256_bytes(_canonical_bytes(command))
            observation_row_hash = _sha256_bytes(_canonical_bytes(observation))
            source_command = {
                "schema": SOURCE_COMMAND_SCHEMA,
                "control_physics_tick": tick,
                "source_atomic_physics_tick": int(source_ack["physics_tick"]),
                "source_atomic_write_count": int(source_ack["write_count"]),
                "adapter_input": {
                    "requested_full12": list(requested.to_full12()),
                    "tracking_servo_names": list(tracking_names),
                    "drive_feedback_bias_requested_full12": list(requested_bias),
                },
                "mapper_configuration": _mapper_configuration(mapper),
                "mapper_pre_state": pre_state,
                "mapper_post_state": post_state,
                "expected_atomic_ack": authoritative_ack,
                "source_command_row_canonical_sha256": command_row_hash,
                "source_observation_row_canonical_sha256": observation_row_hash,
                "drive_target_full12_sha256": phase_snapshot_drive_target_sha256(
                    authoritative_ack["drive_target_full12"]
                ),
                "actuation_contract_sha256": phase_snapshot_actuation_contract_sha256(
                    authoritative_ack
                ),
            }
            result[tick] = {
                "command": command,
                "observation": observation,
                "source_command": source_command,
            }
    if set(result) != ticks:
        raise PhaseSnapshotError("source mapper replay did not retain every phase tick")
    return result


def _contains_signed_positive_rebound_requirement(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("signed_positive_rebound_required") is True:
            return True
        return any(
            _contains_signed_positive_rebound_requirement(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_signed_positive_rebound_requirement(item) for item in value
        )
    return False


def _phase_entry_boundaries_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, _PhaseEntryBoundary]:
    result: dict[str, _PhaseEntryBoundary] = {
        "P01": _PhaseEntryBoundary(
            source_tick=0,
            controller_anchor_tick=None,
            target_entry_tick=0,
            source_replay_steps=1,
            uses_causal_predecessor=False,
        )
    }
    wait_entry_ticks: dict[str, int] = {}
    verify_result_ticks: dict[str, int] = {}
    for row in rows:
        phase = str(row.get("state_id"))
        if (
            phase in PHASE_IDS
            and row.get("from_lifecycle") == "EXECUTE_MOTION"
            and row.get("to_lifecycle") == "VERIFY_RESULT"
        ):
            time_s = float(row["sim_time_s"])
            tick = int(round(time_s * PHYSICS_HZ))
            if not math.isclose(time_s, tick / PHYSICS_HZ, abs_tol=2.0e-6):
                raise PhaseSnapshotError(
                    f"{phase} VERIFY_RESULT start is not on the 120 Hz lattice"
                )
            # A bounded recovery may legitimately issue the same phase motion
            # again and therefore create another VERIFY_RESULT transition.
            # Retain the latest transition seen before the dependent phase
            # entry; it is the causal predecessor of the eventual DONE edge.
            verify_result_ticks[phase] = tick
        if (
            phase in PHASE_IDS
            and row.get("to_lifecycle") == "WAIT_ENTRY"
            and row.get("from_lifecycle") != "WAIT_ENTRY"
        ):
            time_s = float(row["sim_time_s"])
            tick = int(round(time_s * PHYSICS_HZ))
            if not math.isclose(time_s, tick / PHYSICS_HZ, abs_tol=2.0e-6):
                raise PhaseSnapshotError(
                    f"{phase} WAIT_ENTRY start is not on the 120 Hz lattice"
                )
            if phase in wait_entry_ticks:
                raise PhaseSnapshotError(
                    f"trial contains multiple WAIT_ENTRY starts for {phase}"
                )
            wait_entry_ticks[phase] = tick
        if (
            phase in PHASE_IDS
            and row.get("from_lifecycle") == "WAIT_ENTRY"
            and row.get("to_lifecycle") == "EXECUTE_MOTION"
            and phase not in result
        ):
            time_s = float(row["sim_time_s"])
            tick = int(round(time_s * PHYSICS_HZ))
            if not math.isclose(time_s, tick / PHYSICS_HZ, abs_tol=2.0e-6):
                raise PhaseSnapshotError(f"{phase} entry is not on the 120 Hz lattice")
            details = row.get("details")
            guards = details.get("guards", ()) if isinstance(details, Mapping) else ()
            uses_causal_predecessor = phase == "P10" and (
                _contains_signed_positive_rebound_requirement(guards)
            )
            if uses_causal_predecessor:
                controller_anchor_tick = wait_entry_ticks.get(phase)
                if (
                    controller_anchor_tick is None
                    or controller_anchor_tick < 0
                    or controller_anchor_tick >= tick
                ):
                    raise PhaseSnapshotError(
                        f"{phase} signed-positive entry lacks a prior WAIT_ENTRY start"
                    )
                phase_index = PHASE_IDS.index(phase)
                predecessor = PHASE_IDS[phase_index - 1]
                source_tick = verify_result_ticks.get(predecessor)
                if (
                    source_tick is None
                    or source_tick < 0
                    or source_tick >= controller_anchor_tick
                ):
                    raise PhaseSnapshotError(
                        f"{phase} signed-positive entry lacks a prior "
                        f"{predecessor} VERIFY_RESULT start"
                    )
                source_replay_steps = tick - source_tick
            else:
                source_tick = tick
                controller_anchor_tick = None
                source_replay_steps = 1
            result[phase] = _PhaseEntryBoundary(
                source_tick=source_tick,
                controller_anchor_tick=controller_anchor_tick,
                target_entry_tick=tick,
                source_replay_steps=source_replay_steps,
                uses_causal_predecessor=uses_causal_predecessor,
            )
    if tuple(result) != PHASE_IDS:
        missing = [phase for phase in PHASE_IDS if phase not in result]
        raise PhaseSnapshotError(f"trial lacks phase-entry transitions: {missing}")
    return result


def _phase_entry_ticks_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Return target decision ticks; reset source ticks may precede them."""

    return {
        phase: boundary.target_entry_tick
        for phase, boundary in _phase_entry_boundaries_from_rows(rows).items()
    }


def phase_entry_ticks(trial_dir: Path | str) -> dict[str, int]:
    trial = Path(trial_dir).resolve()
    transitions = trial / SOURCE_TRANSITION_FILE
    if not transitions.is_file():
        raise FileNotFoundError(transitions)
    return _phase_entry_ticks_from_rows(_read_jsonl(transitions))


def _rows_at_ticks(path: Path, ticks: set[int], key: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    maximum = max(ticks)
    for row in _read_jsonl(path):
        tick = int(row[key])
        if tick in ticks and tick not in result:
            result[tick] = row
        if tick > maximum and len(result) == len(ticks):
            break
    missing = ticks - set(result)
    if missing:
        raise PhaseSnapshotError(f"{path.name} lacks ticks {sorted(missing)}")
    return result


def _event_latches(
    rows: Iterable[Mapping[str, Any]], entry_tick: int
) -> dict[str, dict[str, int | bool | None]]:
    state = {
        leg: {
            "active_lift": False,
            "active_lift_tick": None,
            "front_face_crossed": False,
            "front_face_crossed_tick": None,
            "top_loaded": False,
            "top_loaded_tick": None,
        }
        for leg in ("FL", "FR", "RL", "RR")
    }
    names = {
        "ACTIVE_LIFT": ("active_lift", "active_lift_tick"),
        "FRONT_FACE_CROSSED": ("front_face_crossed", "front_face_crossed_tick"),
        "TOP_LOADED": ("top_loaded", "top_loaded_tick"),
    }
    for row in rows:
        tick = int(row["physics_tick"])
        if tick > entry_tick:
            break
        leg = str(row.get("leg"))
        event = str(row.get("event"))
        if leg in state and event in names:
            flag, tick_name = names[event]
            state[leg][flag] = True
            state[leg][tick_name] = tick
    return state


def _expected_replay_fsm_context(
    phase: str,
    tick: int,
    controller_anchor_tick: int,
) -> tuple[str, str]:
    """Return the only legal command context for a hybrid causal replay tick."""

    if tick < controller_anchor_tick:
        predecessor = PHASE_IDS[PHASE_IDS.index(phase) - 1]
        return predecessor, "VERIFY_RESULT"
    return phase, "WAIT_ENTRY"


def _snapshot_payload(
    *,
    trial: Path,
    trial_id: str,
    phase: str,
    tick: int,
    controller_anchor_tick: int | None,
    target_entry_tick: int,
    observation: Mapping[str, Any],
    controller_command: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    source_commands: Iterable[Mapping[str, Any]],
    source_command_contexts: Iterable[Mapping[str, Any]],
    contact_event_latches: Mapping[str, Any],
    level_reference_orientation_wxyz: list[float],
) -> dict[str, Any]:
    base = observation["base"]
    joints = observation["joints"]
    wheels = observation["wheels"]
    ordered_joints = (
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    )
    ordered_wheels = (
        "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle"
    )
    completed = list(PHASE_IDS[: PHASE_IDS.index(phase)])
    if type(target_entry_tick) is not int or target_entry_tick < tick:
        raise PhaseSnapshotError(f"invalid target entry tick for {phase}")
    replay_rows = tuple(dict(item) for item in source_commands)
    if not replay_rows:
        raise PhaseSnapshotError(f"{phase} source-command replay is empty")
    replay_contexts = tuple(source_command_contexts)
    if len(replay_contexts) != len(replay_rows):
        raise PhaseSnapshotError(
            f"{phase} source-command replay lacks exact FSM context rows"
        )
    command = replay_contexts[0]
    uses_causal_predecessor = target_entry_tick > tick
    source_replay_steps = len(replay_rows)
    if uses_causal_predecessor:
        if (
            target_entry_tick - tick != source_replay_steps
            or type(controller_anchor_tick) is not int
            or not tick < controller_anchor_tick < target_entry_tick
        ):
            raise PhaseSnapshotError(
                f"{phase} hybrid replay boundary is invalid"
            )
    elif (
        target_entry_tick != tick
        or source_replay_steps != 1
        or controller_anchor_tick is not None
    ):
        raise PhaseSnapshotError(
            f"{phase} non-causal reset must retain one source command"
        )
    source_state = command.get("state_id")
    source_lifecycle = command.get("lifecycle")
    if uses_causal_predecessor:
        expected_source_state, expected_source_lifecycle = (
            _expected_replay_fsm_context(phase, tick, int(controller_anchor_tick))
        )
    else:
        expected_source_state, expected_source_lifecycle = phase, "EXECUTE_MOTION"
    if (
        source_state != expected_source_state
        or source_lifecycle != expected_source_lifecycle
    ):
        raise PhaseSnapshotError(
            f"{phase} source command state/lifecycle does not match its reset boundary"
        )
    controller_state = controller_command.get("state_id")
    controller_lifecycle = controller_command.get("lifecycle")
    expected_controller_lifecycle = (
        "WAIT_ENTRY" if uses_causal_predecessor else "EXECUTE_MOTION"
    )
    expected_controller_tick = (
        int(controller_anchor_tick) if uses_causal_predecessor else tick
    )
    if (
        controller_command.get("control_physics_tick") != expected_controller_tick
        or controller_state != phase
        or controller_lifecycle != expected_controller_lifecycle
    ):
        raise PhaseSnapshotError(
            f"{phase} controller anchor state/lifecycle is invalid"
        )
    serialized_source_commands = replay_rows
    if uses_causal_predecessor:
        serialized_source_commands = tuple(
            {
                **source,
                "source_fsm_state": context.get("state_id"),
                "source_fsm_lifecycle": context.get("lifecycle"),
                "target_entry_tick": target_entry_tick,
            }
            for source, context in zip(replay_rows, replay_contexts, strict=True)
        )
    serialized_source_command = dict(serialized_source_commands[0])
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "reset_use": "TRAINING_RESET_STATE_WRITE",
        "in_episode_root_write": "FORBIDDEN_IN_EPISODE_ROOT_WRITE",
        "source_trial": trial_id,
        "source_trial_path": str(trial),
        "source_tick": tick,
        "source_time_s": tick / PHYSICS_HZ,
        "source_artifacts": dict(source_artifacts),
        "source_command": serialized_source_command,
        "source_commands": [dict(source) for source in serialized_source_commands],
        "source_replay_steps": source_replay_steps,
        "fsm_state": controller_state,
        "fsm_lifecycle": controller_lifecycle,
        "phase_history": completed,
        "root_state": {
            "position_w_m": list(base["position_w_m"]),
            "orientation_wxyz": list(base["orientation_wxyz"]),
            "linear_velocity_w_m_s": list(base["linear_velocity_w_m_s"]),
            "angular_velocity_w_rad_s": list(base["angular_velocity_w_rad_s"]),
        },
        "joint_state": {
            "logical_position_deg": [float(joints[name]["position_deg"]) for name in ordered_joints],
            "logical_velocity_deg_s": [float(joints[name]["velocity_deg_s"]) for name in ordered_joints],
            "order": list(ordered_joints),
        },
        "wheel_state": {
            "logical_velocity_rad_s": [float(wheels[name]["velocity_rad_s"]) for name in ordered_wheels],
            "order": list(ordered_wheels),
        },
        "nominal_full12": list(command["nominal_full12"]),
        "applied_full12": list(command["applied_full12"]),
        "fsm_history": {
            "completed_phases": completed,
            "recovery_count": 0,
        },
        "contact_event_latches": dict(contact_event_latches),
        "obstacle_relative_geometry": {
            "obstacle": observation["obstacle"],
            "wheel_centers_w_m": {
                name: wheels[name]["center_w_m"] for name in ordered_wheels
            },
            "wheel_bottoms_w_m": {
                name: wheels[name]["bottom_w_m"] for name in ordered_wheels
            },
        },
        "contact_state": {
            name: {
                "class": observation["contacts"][wheels[name]["body_name"]]["contact_class"],
                "ground_active": observation["contacts"][wheels[name]["body_name"]]["ground"]["active"],
                "obstacle_active": observation["contacts"][wheels[name]["body_name"]]["obstacle"]["active"],
            }
            for name in ordered_wheels
        },
        "level_reference_orientation_wxyz": list(level_reference_orientation_wxyz),
        "snapshot_semantics": "state is written only before the first episode physics tick; live physics and frozen FSM own all subsequent state",
    }
    if uses_causal_predecessor:
        payload["controller_anchor_tick"] = controller_anchor_tick
        payload["controller_anchor_time_s"] = controller_anchor_tick / PHYSICS_HZ
        payload["target_entry_tick"] = target_entry_tick
        payload["snapshot_semantics"] = (
            "hybrid causal physical state is written from the predecessor VERIFY_RESULT "
            "anchor while controller state and latches come from the later WAIT_ENTRY "
            "anchor; every intervening real source command is replayed in order and the "
            "final post-replay live observation is the target phase-entry decision; live "
            "physics and frozen FSM own all subsequent state"
        )
    return payload


def build_phase_snapshots(
    trial_dir: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"snapshot output already exists: {output}")
    trial, source_paths, source_identities = _capture_source_surface(trial_dir)
    identity_by_path = {
        Path(identity[0]): identity
        for identity in source_identities
        if identity[1] == "file"
    }
    trial_manifest_path = source_paths["trial_manifest"]
    trial_manifest_bytes = _capture_source_bytes_once(
        trial_manifest_path,
        label="source trial manifest",
        expected_identity=identity_by_path[trial_manifest_path],
    )
    manifest_source = _decode_json_object(
        trial_manifest_bytes,
        label="source trial manifest",
        path=trial_manifest_path,
    )
    trial_id_value = manifest_source.get("trial_id")
    if not isinstance(trial_id_value, str) or not trial_id_value:
        raise PhaseSnapshotError("source trial manifest trial_id is invalid")
    trial_id = trial_id_value
    transition_bytes = _capture_source_bytes_once(
        source_paths["transition"],
        label="source transition stream",
        expected_identity=identity_by_path[source_paths["transition"]],
    )
    leg_crossing_bytes = _capture_source_bytes_once(
        source_paths["leg_crossing"],
        label="source leg-crossing stream",
        expected_identity=identity_by_path[source_paths["leg_crossing"]],
    )
    transition_rows = _read_jsonl_bytes(
        transition_bytes, source_paths["transition"]
    )
    leg_crossing_rows = _read_jsonl_bytes(
        leg_crossing_bytes, source_paths["leg_crossing"]
    )
    boundaries_by_phase = _phase_entry_boundaries_from_rows(transition_rows)
    ticks = {
        tick
        for boundary in boundaries_by_phase.values()
        for tick in range(
            boundary.source_tick,
            boundary.source_tick + boundary.source_replay_steps,
        )
    }
    with _SingleReadJsonlCapture(
        source_paths["command"],
        label="source command stream",
        expected_identity=identity_by_path[source_paths["command"]],
    ) as command_capture, _SingleReadJsonlCapture(
        source_paths["observation"],
        label="source observation stream",
        expected_identity=identity_by_path[source_paths["observation"]],
    ) as observation_capture:
        replay_rows = _source_replay_at_phase_ticks(
            manifest_source,
            ticks,
            command_rows=command_capture,
            observation_rows=observation_capture,
        )
        command_capture.drain()
        observation_capture.drain()
        command_artifact = command_capture.artifact_record(SOURCE_COMMAND_FILE)
        observation_artifact = observation_capture.artifact_record(
            SOURCE_OBSERVATION_FILE
        )
    transition_artifact = {
        "name": SOURCE_TRANSITION_FILE,
        "bytes": len(transition_bytes),
        "sha256": _sha256_bytes(transition_bytes),
    }
    leg_crossing_artifact = {
        "name": SOURCE_LEG_CROSSING_FILE,
        "bytes": len(leg_crossing_bytes),
        "sha256": _sha256_bytes(leg_crossing_bytes),
    }
    source_artifacts = {
        "trial_manifest": {
            "name": SOURCE_TRIAL_MANIFEST_FILE,
            "bytes": len(trial_manifest_bytes),
            "sha256": _sha256_bytes(trial_manifest_bytes),
        },
        "command": _source_artifact_record(
            manifest_source,
            artifact_key="command",
            expected_name=SOURCE_COMMAND_FILE,
            captured=command_artifact,
        ),
        "observation": _source_artifact_record(
            manifest_source,
            artifact_key="observation",
            expected_name=SOURCE_OBSERVATION_FILE,
            captured=observation_artifact,
        ),
        "transition": _source_artifact_record(
            manifest_source,
            artifact_key="transition",
            expected_name=SOURCE_TRANSITION_FILE,
            captured=transition_artifact,
        ),
        "leg_crossing": _source_artifact_record(
            manifest_source,
            artifact_key="leg_crossing",
            expected_name=SOURCE_LEG_CROSSING_FILE,
            captured=leg_crossing_artifact,
        ),
    }
    _assert_source_surface_unchanged(source_identities)
    observations = {tick: row["observation"] for tick, row in replay_rows.items()}
    commands = {tick: row["command"] for tick, row in replay_rows.items()}
    for phase, boundary in boundaries_by_phase.items():
        replay_ticks = range(
            boundary.source_tick,
            boundary.source_tick + boundary.source_replay_steps,
        )
        for replay_tick in replay_ticks:
            source_command_row = commands[replay_tick]
            if boundary.uses_causal_predecessor:
                assert boundary.controller_anchor_tick is not None
                expected_state, expected_lifecycle = _expected_replay_fsm_context(
                    phase, replay_tick, boundary.controller_anchor_tick
                )
            else:
                expected_state, expected_lifecycle = phase, "EXECUTE_MOTION"
            if (
                source_command_row.get("state_id") != expected_state
                or source_command_row.get("lifecycle") != expected_lifecycle
            ):
                raise PhaseSnapshotError(
                    f"{phase} source command state/lifecycle does not match its reset boundary"
                )
    level_reference_orientation = list(observations[0]["base"]["orientation_wxyz"])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    causal_predecessor_phases = [
        phase
        for phase, boundary in boundaries_by_phase.items()
        if boundary.uses_causal_predecessor
    ]
    for phase, boundary in boundaries_by_phase.items():
        tick = boundary.source_tick
        controller_tick = (
            boundary.controller_anchor_tick
            if boundary.controller_anchor_tick is not None
            else tick
        )
        payload = _snapshot_payload(
            trial=trial,
            trial_id=trial_id,
            phase=phase,
            tick=tick,
            controller_anchor_tick=boundary.controller_anchor_tick,
            target_entry_tick=boundary.target_entry_tick,
            observation=observations[tick],
            controller_command=commands[controller_tick],
            source_artifacts=source_artifacts,
            source_commands=(
                replay_rows[replay_tick]["source_command"]
                for replay_tick in range(
                    boundary.source_tick,
                    boundary.source_tick + boundary.source_replay_steps,
                )
            ),
            source_command_contexts=(
                commands[replay_tick]
                for replay_tick in range(
                    boundary.source_tick,
                    boundary.source_tick + boundary.source_replay_steps,
                )
            ),
            contact_event_latches=_event_latches(
                leg_crossing_rows, controller_tick
            ),
            level_reference_orientation_wxyz=level_reference_orientation,
        )
        state_hash = _sha256_bytes(_canonical_bytes(payload))
        complete = {**payload, "state_sha256": state_hash}
        phase_dir = output / phase
        phase_dir.mkdir()
        snapshot_path = phase_dir / "snapshot.json"
        snapshot_path.write_bytes(_indented_json_lf_bytes(complete))
        file_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        (phase_dir / "snapshot.sha256").write_bytes(
            f"{file_hash}  snapshot.json\n".encode("ascii")
        )
        row = {
            "phase": phase,
            "source_tick": tick,
            "source_replay_steps": boundary.source_replay_steps,
            "state_sha256": state_hash,
            "file_sha256": file_hash,
            "path": str(snapshot_path),
            "source_command_row_canonical_sha256": replay_rows[tick][
                "source_command"
            ]["source_command_row_canonical_sha256"],
            "source_observation_row_canonical_sha256": replay_rows[tick][
                "source_command"
            ]["source_observation_row_canonical_sha256"],
            "drive_target_full12_sha256": replay_rows[tick]["source_command"][
                "drive_target_full12_sha256"
            ],
            "actuation_contract_sha256": replay_rows[tick]["source_command"][
                "actuation_contract_sha256"
            ],
        }
        if boundary.uses_causal_predecessor:
            assert boundary.controller_anchor_tick is not None
            row["controller_anchor_tick"] = boundary.controller_anchor_tick
            row["controller_anchor_time_s"] = (
                boundary.controller_anchor_tick / PHYSICS_HZ
            )
            row["target_entry_tick"] = boundary.target_entry_tick
        rows.append(row)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_trial": trial_id,
        "source_trial_path": str(trial),
        "source_artifacts": source_artifacts,
        "physics_hz": PHYSICS_HZ,
        "phase_count": len(rows),
        "causal_predecessor_phases": causal_predecessor_phases,
        "snapshots": rows,
    }
    (output / "manifest.json").write_bytes(_indented_json_lf_bytes(manifest))
    validate_phase_snapshots(output, canonical_root=output)
    return manifest


def _validate_source_artifacts(value: Any, *, label: str) -> Mapping[str, Any]:
    expected_files = {
        "trial_manifest": SOURCE_TRIAL_MANIFEST_FILE,
        "command": SOURCE_COMMAND_FILE,
        "observation": SOURCE_OBSERVATION_FILE,
        "transition": SOURCE_TRANSITION_FILE,
        "leg_crossing": SOURCE_LEG_CROSSING_FILE,
    }
    if not isinstance(value, Mapping) or set(value) != set(expected_files):
        raise PhaseSnapshotError(
            f"{label} must bind every source file used by the builder"
        )
    for role, expected_name in expected_files.items():
        row = value[role]
        if not isinstance(row, Mapping) or set(row) != {"name", "bytes", "sha256"}:
            raise PhaseSnapshotError(f"{label}.{role} binding is incomplete")
        if row.get("name") != expected_name:
            raise PhaseSnapshotError(f"{label}.{role} has the wrong file name")
        byte_count = row.get("bytes")
        if type(byte_count) is not int or byte_count < 0:
            raise PhaseSnapshotError(f"{label}.{role}.bytes must be nonnegative")
        _require_sha256(row.get("sha256"), label=f"{label}.{role}.sha256")
    return value


def _validate_mapper_state(
    value: Any,
    *,
    label: str,
    expected_source_tick: int | None,
) -> Mapping[str, Any]:
    expected_keys = {
        "schema",
        "source_control_physics_tick",
        "requested_servo_deg",
        "applied_drive_command_deg",
        "nominal_target_reached",
        "tracking_compensation_deg",
        "tracking_active",
        "retiring_stale_bias",
        "feedback_tick",
        "final_drive_servo_deg",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PhaseSnapshotError(f"{label} fields are incomplete or unexpected")
    if value.get("schema") != SOURCE_MAPPER_STATE_SCHEMA:
        raise PhaseSnapshotError(f"{label} schema is invalid")
    if value.get("source_control_physics_tick") != expected_source_tick:
        raise PhaseSnapshotError(f"{label} source tick is invalid")
    for field in (
        "requested_servo_deg",
        "applied_drive_command_deg",
        "tracking_compensation_deg",
        "final_drive_servo_deg",
    ):
        _finite_values(value.get(field, ()), len(SERVO_ORDER), f"{label}.{field}")
    for field in (
        "nominal_target_reached",
        "tracking_active",
        "retiring_stale_bias",
    ):
        _bool_values(value.get(field, ()), len(SERVO_ORDER), f"{label}.{field}")
    feedback_tick = value.get("feedback_tick")
    if type(feedback_tick) is not int or feedback_tick < 0:
        raise PhaseSnapshotError(f"{label}.feedback_tick must be nonnegative")
    return value


def _validate_additional_source_replay_row(
    source: Mapping[str, Any],
    *,
    phase: str,
    source_tick: int,
    controller_anchor_tick: int,
    target_entry_tick: int,
    first_source: Mapping[str, Any],
    previous_source: Mapping[str, Any],
) -> None:
    """Validate one post-anchor row in a causal source-command replay."""

    expected_state, expected_lifecycle = _expected_replay_fsm_context(
        phase, source_tick, controller_anchor_tick
    )
    if set(source) != set(first_source):
        raise PhaseSnapshotError(
            f"snapshot source-command replay fields are inconsistent for {phase}"
        )
    if (
        source.get("schema") != SOURCE_COMMAND_SCHEMA
        or source.get("control_physics_tick") != source_tick
        or source.get("source_atomic_physics_tick")
        != SOURCE_SETTLE_TICKS + source_tick
        or source.get("source_atomic_write_count")
        != SOURCE_SETTLE_TICKS + source_tick + 1
        or source.get("source_fsm_state") != expected_state
        or source.get("source_fsm_lifecycle") != expected_lifecycle
        or source.get("target_entry_tick") != target_entry_tick
    ):
        raise PhaseSnapshotError(
            f"snapshot causal source-command replay context mismatch for {phase}"
        )
    for name in (
        "source_command_row_canonical_sha256",
        "source_observation_row_canonical_sha256",
        "drive_target_full12_sha256",
        "actuation_contract_sha256",
    ):
        _require_sha256(
            source.get(name), label=f"snapshot {phase} replay tick {source_tick}.{name}"
        )
    if source.get("mapper_configuration") != first_source.get(
        "mapper_configuration"
    ):
        raise PhaseSnapshotError(
            f"snapshot mapper configuration changed during replay for {phase}"
        )
    pre_state = _validate_mapper_state(
        source.get("mapper_pre_state"),
        label=f"snapshot {phase} replay tick {source_tick}.mapper_pre_state",
        expected_source_tick=source_tick - 1,
    )
    post_state = _validate_mapper_state(
        source.get("mapper_post_state"),
        label=f"snapshot {phase} replay tick {source_tick}.mapper_post_state",
        expected_source_tick=source_tick,
    )
    if pre_state != previous_source.get("mapper_post_state"):
        raise PhaseSnapshotError(
            f"snapshot mapper replay is not contiguous at tick {source_tick} for {phase}"
        )
    if pre_state.get("feedback_tick") != SOURCE_SETTLE_TICKS + source_tick:
        raise PhaseSnapshotError(
            f"snapshot mapper replay pre-state clock mismatch for {phase}"
        )
    if post_state.get("feedback_tick") != SOURCE_SETTLE_TICKS + source_tick + 1:
        raise PhaseSnapshotError(
            f"snapshot mapper replay post-state clock mismatch for {phase}"
        )

    expected_ack = source.get("expected_atomic_ack")
    if not isinstance(expected_ack, Mapping) or set(expected_ack) != set(
        SOURCE_ACK_MATCH_FIELDS
    ):
        raise PhaseSnapshotError(
            f"snapshot replay atomic ack is incomplete for {phase}"
        )
    if (
        expected_ack.get("schema") != "wlr50_clean.atomic_full12_ack.v1"
        or not _equivalent(expected_ack.get("physics_dt_s"), PHYSICS_DT_S)
        or expected_ack.get("articulation_writes_this_call") != 1
        or expected_ack.get("canonical_order") != list(FULL12_ORDER)
        or expected_ack.get("servo_tracking_feedback_sample_tick")
        != SOURCE_SETTLE_TICKS + source_tick
    ):
        raise PhaseSnapshotError(
            f"snapshot replay atomic ack metadata is invalid for {phase}"
        )
    for field in (
        "requested_full12",
        "applied_full12",
        "drive_target_full12",
        "native_drive_target_full12",
        "drive_feedback_bias_requested_full12",
        "drive_feedback_bias_realized_full12",
    ):
        _finite_values(
            expected_ack.get(field, ()),
            len(FULL12_ORDER),
            f"{phase} replay tick {source_tick}.{field}",
        )
    for field in (
        "servo_applied_drive_command_deg",
        "servo_native_drive_command_deg",
        "servo_tracking_compensation_deg",
        "servo_target_physical_rad",
    ):
        _finite_values(
            expected_ack.get(field, ()),
            len(SERVO_ORDER),
            f"{phase} replay tick {source_tick}.{field}",
        )
    _finite_values(
        expected_ack.get("wheel_target_physical_rad_s", ()),
        len(WHEEL_ORDER),
        f"{phase} replay tick {source_tick}.wheel_target_physical_rad_s",
    )
    for field in ("servo_nominal_target_reached", "servo_tracking_active"):
        _bool_values(
            expected_ack.get(field, ()),
            len(SERVO_ORDER),
            f"{phase} replay tick {source_tick}.{field}",
        )
    for field in ("command_was_clamped", "servo_tracking_feedback_sampled"):
        if type(expected_ack.get(field)) is not bool:
            raise PhaseSnapshotError(
                f"{phase} replay tick {source_tick}.{field} must be boolean"
            )
    for field in (
        "drive_feedback_final_slew_limit_deg_per_tick",
        "motion_start_skew_s",
    ):
        _finite_values(
            (expected_ack.get(field),),
            1,
            f"{phase} replay tick {source_tick}.{field}",
        )
    if not _equivalent(expected_ack.get("motion_start_skew_s"), 0.0):
        raise PhaseSnapshotError(
            f"snapshot replay motion skew is nonzero for {phase}"
        )
    tracking_names = expected_ack.get("tracking_servo_names")
    if (
        not isinstance(tracking_names, list)
        or len(set(tracking_names)) != len(tracking_names)
        or any(name not in SERVO_ORDER for name in tracking_names)
    ):
        raise PhaseSnapshotError(
            f"snapshot replay tracking names are invalid for {phase}"
        )
    servo_ids = expected_ack.get("servo_joint_ids")
    wheel_ids = expected_ack.get("wheel_joint_ids")
    if (
        not isinstance(servo_ids, list)
        or not isinstance(wheel_ids, list)
        or len(servo_ids) != len(SERVO_ORDER)
        or len(wheel_ids) != len(WHEEL_ORDER)
        or any(type(value) is not int or value < 0 for value in servo_ids + wheel_ids)
        or len(set(servo_ids + wheel_ids)) != len(FULL12_ORDER)
    ):
        raise PhaseSnapshotError(
            f"snapshot replay joint ids are invalid for {phase}"
        )

    adapter_input = source.get("adapter_input")
    if not isinstance(adapter_input, Mapping) or set(adapter_input) != {
        "requested_full12",
        "tracking_servo_names",
        "drive_feedback_bias_requested_full12",
    }:
        raise PhaseSnapshotError(
            f"snapshot replay adapter input is incomplete for {phase}"
        )
    for field in adapter_input:
        if not _equivalent(adapter_input[field], expected_ack[field]):
            raise PhaseSnapshotError(
                f"snapshot replay adapter input differs from source ack: {field}"
            )
    post_field_bindings = {
        "requested_servo_deg": expected_ack["applied_full12"][: len(SERVO_ORDER)],
        "applied_drive_command_deg": expected_ack["native_drive_target_full12"][: len(SERVO_ORDER)],
        "nominal_target_reached": expected_ack["servo_nominal_target_reached"],
        "tracking_compensation_deg": expected_ack["servo_tracking_compensation_deg"],
        "tracking_active": expected_ack["servo_tracking_active"],
        "final_drive_servo_deg": expected_ack["drive_target_full12"][: len(SERVO_ORDER)],
    }
    for field, expected in post_field_bindings.items():
        if not _equivalent(post_state[field], expected):
            raise PhaseSnapshotError(
                f"snapshot replay mapper post-state differs from source ack: {field}"
            )
    if not _equivalent(
        expected_ack.get("requested_full12"), expected_ack.get("applied_full12")
    ):
        raise PhaseSnapshotError(
            f"snapshot replay source request was clamped for {phase}"
        )
    target_hash = phase_snapshot_drive_target_sha256(
        expected_ack["drive_target_full12"]
    )
    if source.get("drive_target_full12_sha256") != target_hash:
        raise PhaseSnapshotError(
            f"snapshot replay drive-target hash mismatch for {phase}"
        )
    actuation_hash = phase_snapshot_actuation_contract_sha256(expected_ack)
    if source.get("actuation_contract_sha256") != actuation_hash:
        raise PhaseSnapshotError(
            f"snapshot replay actuation-contract hash mismatch for {phase}"
        )


def validate_phase_snapshot_payload_contract(
    payload: Mapping[str, Any],
    phase: str,
    *,
    manifest_row: Mapping[str, Any] | None = None,
    manifest_source_artifacts: Mapping[str, Any] | None = None,
    causal_predecessor_required: bool | None = None,
) -> None:
    """Fail closed on every v2 source-command replay field and digest."""

    if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("fsm_state") != phase:
        raise PhaseSnapshotError(f"invalid snapshot {phase}")
    source_tick = payload.get("source_tick")
    if type(source_tick) is not int or source_tick < 0:
        raise PhaseSnapshotError(f"snapshot source tick is invalid for {phase}")
    source_replay_steps = payload.get("source_replay_steps")
    if type(source_replay_steps) is not int or source_replay_steps <= 0:
        raise PhaseSnapshotError(
            f"snapshot source replay step count is invalid for {phase}"
        )
    has_target_entry_tick = "target_entry_tick" in payload
    target_entry_tick = payload.get("target_entry_tick")
    has_controller_anchor_tick = "controller_anchor_tick" in payload
    has_controller_anchor_time = "controller_anchor_time_s" in payload
    controller_anchor_tick = payload.get("controller_anchor_tick")
    if has_target_entry_tick:
        if (
            phase != "P10"
            or type(target_entry_tick) is not int
            or target_entry_tick != source_tick + source_replay_steps
        ):
            raise PhaseSnapshotError(
                f"snapshot causal target-entry tick is invalid for {phase}"
            )
        if (
            not has_controller_anchor_tick
            or not has_controller_anchor_time
            or type(controller_anchor_tick) is not int
            or not source_tick < controller_anchor_tick < target_entry_tick
        ):
            raise PhaseSnapshotError(
                f"snapshot controller anchor tick is invalid for {phase}"
            )
        controller_anchor_time = payload.get("controller_anchor_time_s")
        if (
            isinstance(controller_anchor_time, bool)
            or not isinstance(controller_anchor_time, (int, float))
            or not math.isfinite(float(controller_anchor_time))
            or float(controller_anchor_time)
            != controller_anchor_tick / PHYSICS_HZ
        ):
            raise PhaseSnapshotError(
                f"snapshot controller anchor time is invalid for {phase}"
            )
    elif source_replay_steps != 1:
        raise PhaseSnapshotError(
            f"snapshot non-causal source replay must contain exactly one step for {phase}"
        )
    elif has_controller_anchor_tick or has_controller_anchor_time:
        raise PhaseSnapshotError(
            f"snapshot non-causal reset declares a controller anchor for {phase}"
        )
    if causal_predecessor_required is True and not has_target_entry_tick:
        raise PhaseSnapshotError(
            f"snapshot {phase} lacks its required causal predecessor target tick"
        )
    if causal_predecessor_required is False and has_target_entry_tick:
        raise PhaseSnapshotError(
            f"snapshot {phase} unexpectedly declares causal predecessor semantics"
        )
    expected_lifecycle = "WAIT_ENTRY" if has_target_entry_tick else "EXECUTE_MOTION"
    if payload.get("fsm_lifecycle") != expected_lifecycle:
        raise PhaseSnapshotError(
            f"snapshot FSM lifecycle is inconsistent with its source tick for {phase}"
        )
    completed_phases = list(PHASE_IDS[: PHASE_IDS.index(phase)])
    if payload.get("phase_history") != completed_phases:
        raise PhaseSnapshotError(
            f"snapshot controller phase history is invalid for {phase}"
        )
    if payload.get("fsm_history") != {
        "completed_phases": completed_phases,
        "recovery_count": 0,
    }:
        raise PhaseSnapshotError(
            f"snapshot controller FSM history is invalid for {phase}"
        )
    source_time_s = payload.get("source_time_s")
    if (
        isinstance(source_time_s, bool)
        or not isinstance(source_time_s, (int, float))
        or not math.isfinite(float(source_time_s))
        or not math.isclose(
            float(source_time_s), source_tick / PHYSICS_HZ, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise PhaseSnapshotError(f"snapshot source time is invalid for {phase}")
    artifacts = _validate_source_artifacts(
        payload.get("source_artifacts"), label=f"snapshot {phase}.source_artifacts"
    )
    if manifest_source_artifacts is not None and artifacts != manifest_source_artifacts:
        raise PhaseSnapshotError(
            f"snapshot source-file binding differs from manifest for {phase}"
        )
    source = payload.get("source_command")
    source_commands = payload.get("source_commands")
    if (
        not isinstance(source_commands, list)
        or len(source_commands) != source_replay_steps
        or any(not isinstance(item, Mapping) for item in source_commands)
    ):
        raise PhaseSnapshotError(
            f"snapshot source-command replay sequence is invalid for {phase}"
        )
    if not source_commands or source != source_commands[0]:
        raise PhaseSnapshotError(
            f"snapshot source_command must equal the first replay row for {phase}"
        )
    expected_source_keys = {
        "schema",
        "control_physics_tick",
        "source_atomic_physics_tick",
        "source_atomic_write_count",
        "adapter_input",
        "mapper_configuration",
        "mapper_pre_state",
        "mapper_post_state",
        "expected_atomic_ack",
        "source_command_row_canonical_sha256",
        "source_observation_row_canonical_sha256",
        "drive_target_full12_sha256",
        "actuation_contract_sha256",
    }
    if has_target_entry_tick:
        expected_source_keys.update(
            {
                "source_fsm_state",
                "source_fsm_lifecycle",
                "target_entry_tick",
            }
        )
    if not isinstance(source, Mapping) or set(source) != expected_source_keys:
        raise PhaseSnapshotError(
            f"snapshot source-command replay fields are incomplete for {phase}"
        )
    if source.get("schema") != SOURCE_COMMAND_SCHEMA:
        raise PhaseSnapshotError(f"snapshot source-command schema is invalid for {phase}")
    if source.get("control_physics_tick") != source_tick:
        raise PhaseSnapshotError(f"snapshot source-command tick mismatch for {phase}")
    if source.get("source_atomic_physics_tick") != SOURCE_SETTLE_TICKS + source_tick:
        raise PhaseSnapshotError(f"snapshot source atomic tick mismatch for {phase}")
    if source.get("source_atomic_write_count") != SOURCE_SETTLE_TICKS + source_tick + 1:
        raise PhaseSnapshotError(f"snapshot source atomic write count mismatch for {phase}")
    if has_target_entry_tick:
        expected_source_state, expected_source_lifecycle = (
            _expected_replay_fsm_context(
                phase, source_tick, int(controller_anchor_tick)
            )
        )
        if (
            source.get("source_fsm_state") != expected_source_state
            or source.get("source_fsm_lifecycle") != expected_source_lifecycle
            or source.get("target_entry_tick") != target_entry_tick
        ):
            raise PhaseSnapshotError(
                f"snapshot causal source-command context mismatch for {phase}"
            )

    for name in (
        "source_command_row_canonical_sha256",
        "source_observation_row_canonical_sha256",
        "drive_target_full12_sha256",
        "actuation_contract_sha256",
    ):
        _require_sha256(source.get(name), label=f"snapshot {phase}.{name}")

    configuration = source.get("mapper_configuration")
    expected_configuration_keys = {
        "physics_dt_s",
        "servo_rate_deg_s",
        "maximum_delta_deg",
        "tracking_gain",
        "tracking_limit_deg",
        "feedback_interval_ticks",
        "standing_pose_deg",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != expected_configuration_keys:
        raise PhaseSnapshotError(f"snapshot mapper configuration is invalid for {phase}")
    numeric_configuration = {
        name: _finite_values((configuration.get(name),), 1, f"{phase}.{name}")[0]
        for name in (
            "physics_dt_s",
            "servo_rate_deg_s",
            "maximum_delta_deg",
            "tracking_gain",
            "tracking_limit_deg",
        )
    }
    if (
        not math.isclose(
            numeric_configuration["physics_dt_s"], PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-15
        )
        or numeric_configuration["servo_rate_deg_s"] <= 0.0
        or not math.isclose(
            numeric_configuration["maximum_delta_deg"],
            numeric_configuration["physics_dt_s"]
            * numeric_configuration["servo_rate_deg_s"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or numeric_configuration["tracking_gain"] < 0.0
        or numeric_configuration["tracking_limit_deg"] < 0.0
    ):
        raise PhaseSnapshotError(f"snapshot mapper numeric configuration is invalid for {phase}")
    interval = configuration.get("feedback_interval_ticks")
    if type(interval) is not int or interval <= 0:
        raise PhaseSnapshotError(f"snapshot mapper feedback interval is invalid for {phase}")
    _finite_values(
        configuration.get("standing_pose_deg", ()),
        len(SERVO_ORDER),
        f"snapshot {phase} standing pose",
    )

    pre_state = _validate_mapper_state(
        source.get("mapper_pre_state"),
        label=f"snapshot {phase}.mapper_pre_state",
        expected_source_tick=(None if source_tick == 0 else source_tick - 1),
    )
    post_state = _validate_mapper_state(
        source.get("mapper_post_state"),
        label=f"snapshot {phase}.mapper_post_state",
        expected_source_tick=source_tick,
    )
    if pre_state.get("feedback_tick") != SOURCE_SETTLE_TICKS + source_tick:
        raise PhaseSnapshotError(f"snapshot mapper pre-state clock mismatch for {phase}")
    if post_state.get("feedback_tick") != SOURCE_SETTLE_TICKS + source_tick + 1:
        raise PhaseSnapshotError(f"snapshot mapper post-state clock mismatch for {phase}")

    expected_ack = source.get("expected_atomic_ack")
    if not isinstance(expected_ack, Mapping) or set(expected_ack) != set(
        SOURCE_ACK_MATCH_FIELDS
    ):
        raise PhaseSnapshotError(f"snapshot expected atomic ack is incomplete for {phase}")
    if expected_ack.get("schema") != "wlr50_clean.atomic_full12_ack.v1":
        raise PhaseSnapshotError(f"snapshot expected atomic ack schema is invalid for {phase}")
    if not _equivalent(expected_ack.get("physics_dt_s"), PHYSICS_DT_S):
        raise PhaseSnapshotError(f"snapshot expected atomic dt is invalid for {phase}")
    if expected_ack.get("articulation_writes_this_call") != 1:
        raise PhaseSnapshotError(f"snapshot expected atomic write count is invalid for {phase}")
    if expected_ack.get("canonical_order") != list(FULL12_ORDER):
        raise PhaseSnapshotError(f"snapshot atomic canonical order is invalid for {phase}")
    for field in (
        "requested_full12",
        "applied_full12",
        "drive_target_full12",
        "native_drive_target_full12",
        "drive_feedback_bias_requested_full12",
        "drive_feedback_bias_realized_full12",
    ):
        _finite_values(
            expected_ack.get(field, ()), len(FULL12_ORDER), f"{phase}.{field}"
        )
    for field in (
        "servo_applied_drive_command_deg",
        "servo_native_drive_command_deg",
        "servo_tracking_compensation_deg",
        "servo_target_physical_rad",
    ):
        _finite_values(expected_ack.get(field, ()), len(SERVO_ORDER), f"{phase}.{field}")
    _finite_values(
        expected_ack.get("wheel_target_physical_rad_s", ()),
        len(WHEEL_ORDER),
        f"{phase}.wheel_target_physical_rad_s",
    )
    for field in ("servo_nominal_target_reached", "servo_tracking_active"):
        _bool_values(expected_ack.get(field, ()), len(SERVO_ORDER), f"{phase}.{field}")
    for field in ("command_was_clamped", "servo_tracking_feedback_sampled"):
        if type(expected_ack.get(field)) is not bool:
            raise PhaseSnapshotError(f"{phase}.{field} must be boolean")
    scalar_fields = (
        "drive_feedback_final_slew_limit_deg_per_tick",
        "motion_start_skew_s",
    )
    for field in scalar_fields:
        _finite_values((expected_ack.get(field),), 1, f"{phase}.{field}")
    if not _equivalent(expected_ack.get("motion_start_skew_s"), 0.0):
        raise PhaseSnapshotError(f"snapshot source motion skew is nonzero for {phase}")
    sample_tick = expected_ack.get("servo_tracking_feedback_sample_tick")
    if type(sample_tick) is not int or sample_tick != SOURCE_SETTLE_TICKS + source_tick:
        raise PhaseSnapshotError(f"snapshot source mapper sample tick is invalid for {phase}")
    tracking_names = expected_ack.get("tracking_servo_names")
    if (
        not isinstance(tracking_names, list)
        or len(set(tracking_names)) != len(tracking_names)
        or any(name not in SERVO_ORDER for name in tracking_names)
    ):
        raise PhaseSnapshotError(f"snapshot source tracking names are invalid for {phase}")
    servo_ids = expected_ack.get("servo_joint_ids")
    wheel_ids = expected_ack.get("wheel_joint_ids")
    if (
        not isinstance(servo_ids, list)
        or not isinstance(wheel_ids, list)
        or len(servo_ids) != len(SERVO_ORDER)
        or len(wheel_ids) != len(WHEEL_ORDER)
        or any(type(value) is not int or value < 0 for value in servo_ids + wheel_ids)
        or len(set(servo_ids + wheel_ids)) != len(FULL12_ORDER)
    ):
        raise PhaseSnapshotError(f"snapshot source joint ids are invalid for {phase}")

    adapter_input = source.get("adapter_input")
    if not isinstance(adapter_input, Mapping) or set(adapter_input) != {
        "requested_full12",
        "tracking_servo_names",
        "drive_feedback_bias_requested_full12",
    }:
        raise PhaseSnapshotError(f"snapshot adapter input is incomplete for {phase}")
    for field in adapter_input:
        if not _equivalent(adapter_input[field], expected_ack[field]):
            raise PhaseSnapshotError(f"snapshot adapter input differs from source ack: {field}")
    post_field_bindings = {
        "requested_servo_deg": expected_ack["applied_full12"][: len(SERVO_ORDER)],
        "applied_drive_command_deg": expected_ack["native_drive_target_full12"][: len(SERVO_ORDER)],
        "nominal_target_reached": expected_ack["servo_nominal_target_reached"],
        "tracking_compensation_deg": expected_ack["servo_tracking_compensation_deg"],
        "tracking_active": expected_ack["servo_tracking_active"],
        "final_drive_servo_deg": expected_ack["drive_target_full12"][: len(SERVO_ORDER)],
    }
    for field, expected in post_field_bindings.items():
        if not _equivalent(post_state[field], expected):
            raise PhaseSnapshotError(
                f"snapshot mapper post-state differs from source ack: {field}"
            )
    for field in ("nominal_full12", "applied_full12"):
        if not _equivalent(payload.get(field), expected_ack.get("requested_full12")):
            raise PhaseSnapshotError(
                f"snapshot {field} differs from authoritative source request for {phase}"
            )
    if not _equivalent(
        expected_ack.get("requested_full12"), expected_ack.get("applied_full12")
    ):
        raise PhaseSnapshotError(
            f"snapshot source request was clamped at phase entry for {phase}"
        )
    target_hash = phase_snapshot_drive_target_sha256(
        expected_ack["drive_target_full12"]
    )
    if source.get("drive_target_full12_sha256") != target_hash:
        raise PhaseSnapshotError(f"snapshot drive-target hash mismatch for {phase}")
    actuation_hash = phase_snapshot_actuation_contract_sha256(expected_ack)
    if source.get("actuation_contract_sha256") != actuation_hash:
        raise PhaseSnapshotError(f"snapshot actuation-contract hash mismatch for {phase}")

    previous_source = source
    for replay_offset, replay_source in enumerate(source_commands[1:], 1):
        assert isinstance(replay_source, Mapping)
        _validate_additional_source_replay_row(
            replay_source,
            phase=phase,
            source_tick=source_tick + replay_offset,
            controller_anchor_tick=int(controller_anchor_tick),
            target_entry_tick=int(target_entry_tick),
            first_source=source,
            previous_source=previous_source,
        )
        previous_source = replay_source

    if manifest_row is not None:
        if manifest_row.get("source_tick") != source_tick:
            raise PhaseSnapshotError(f"manifest source tick mismatch for {phase}")
        if manifest_row.get("source_replay_steps") != source_replay_steps:
            raise PhaseSnapshotError(
                f"manifest source replay step count mismatch for {phase}"
            )
        if has_target_entry_tick:
            if (
                manifest_row.get("target_entry_tick") != target_entry_tick
                or manifest_row.get("controller_anchor_tick")
                != controller_anchor_tick
                or manifest_row.get("controller_anchor_time_s")
                != payload.get("controller_anchor_time_s")
            ):
                raise PhaseSnapshotError(
                    f"manifest hybrid replay boundary mismatch for {phase}"
                )
        elif any(
            field in manifest_row
            for field in (
                "target_entry_tick",
                "controller_anchor_tick",
                "controller_anchor_time_s",
            )
        ):
            raise PhaseSnapshotError(
                f"manifest unexpectedly declares a hybrid replay boundary for {phase}"
            )
        bindings = {
            "source_command_row_canonical_sha256": source[
                "source_command_row_canonical_sha256"
            ],
            "source_observation_row_canonical_sha256": source[
                "source_observation_row_canonical_sha256"
            ],
            "drive_target_full12_sha256": target_hash,
            "actuation_contract_sha256": actuation_hash,
        }
        for field, expected in bindings.items():
            if manifest_row.get(field) != expected:
                raise PhaseSnapshotError(
                    f"manifest source-command binding mismatch for {phase}: {field}"
                )


def capture_validated_phase_snapshot_bundle(
    output_root: Path | str,
    *,
    canonical_root: Path | str | None = DEFAULT_PHASE_SNAPSHOT_ROOT,
) -> ValidatedPhaseSnapshotBundle:
    """Read every required file once into an immutable, fail-closed bundle."""

    root = _unredirected_path(output_root, label="phase snapshot root")
    if canonical_root is not None:
        expected_root = _unredirected_path(
            canonical_root, label="canonical phase snapshot root"
        )
        if root != expected_root:
            raise PhaseSnapshotError(
                "phase snapshot root differs from canonical project reference/ppo_phase_snapshots"
            )
    identities = [
        _path_identity(root, label="phase snapshot root", directory=True)
    ]
    manifest_path = _unredirected_path(
        root / "manifest.json", label="phase snapshot manifest"
    )
    _require_within(manifest_path, root, label="phase snapshot manifest")
    manifest_identity = _path_identity(
        manifest_path, label="phase snapshot manifest", directory=False
    )
    manifest_bytes = _read_file_bytes(manifest_path, label="phase snapshot manifest")
    if _path_identity(
        manifest_path, label="phase snapshot manifest", directory=False
    ) != manifest_identity:
        raise PhaseSnapshotError("phase snapshot manifest changed while it was read")
    identities.append(manifest_identity)
    manifest = dict(
        _decode_json_object(
            manifest_bytes,
            label="phase snapshot manifest",
            path=manifest_path,
        )
    )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PhaseSnapshotError("invalid phase snapshot manifest schema")
    if set(manifest) != {
        "schema",
        "source_trial",
        "source_trial_path",
        "source_artifacts",
        "physics_hz",
        "phase_count",
        "causal_predecessor_phases",
        "snapshots",
    }:
        raise PhaseSnapshotError("phase snapshot manifest fields are incomplete or unexpected")
    if not isinstance(manifest.get("source_trial"), str) or not manifest["source_trial"]:
        raise PhaseSnapshotError("phase snapshot manifest source_trial is invalid")
    if not isinstance(manifest.get("source_trial_path"), str) or not manifest[
        "source_trial_path"
    ]:
        raise PhaseSnapshotError("phase snapshot manifest source_trial_path is invalid")
    if type(manifest.get("phase_count")) is not int or manifest["phase_count"] != len(PHASE_IDS):
        raise PhaseSnapshotError("phase snapshot manifest must declare exactly 13 phases")
    physics_hz = manifest.get("physics_hz")
    if isinstance(physics_hz, bool) or not isinstance(physics_hz, (int, float)):
        raise PhaseSnapshotError("phase snapshot manifest physics_hz must be numeric")
    if float(physics_hz) != PHYSICS_HZ:
        raise PhaseSnapshotError(f"phase snapshot manifest physics_hz must be {PHYSICS_HZ:g}")
    manifest_source_artifacts = _validate_source_artifacts(
        manifest.get("source_artifacts"),
        label="phase snapshot manifest source_artifacts",
    )
    causal_predecessor_phases = manifest.get("causal_predecessor_phases")
    if (
        not isinstance(causal_predecessor_phases, list)
        or any(phase != "P10" for phase in causal_predecessor_phases)
        or len(set(causal_predecessor_phases)) != len(causal_predecessor_phases)
        or causal_predecessor_phases
        != [phase for phase in PHASE_IDS if phase in causal_predecessor_phases]
    ):
        raise PhaseSnapshotError(
            "phase snapshot causal-predecessor phase list is invalid"
        )
    causal_predecessor_phase_set = set(causal_predecessor_phases)
    rows = manifest.get("snapshots")
    if not isinstance(rows, list) or len(rows) != len(PHASE_IDS):
        raise PhaseSnapshotError("phase snapshot manifest must contain exactly 13 entries")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PhaseSnapshotError("phase snapshot manifest entries must be JSON objects")
    expected_row_fields = {
        "phase",
        "source_tick",
        "source_replay_steps",
        "state_sha256",
        "file_sha256",
        "path",
        "source_command_row_canonical_sha256",
        "source_observation_row_canonical_sha256",
        "drive_target_full12_sha256",
        "actuation_contract_sha256",
    }
    if any(
        set(row)
        != (
            expected_row_fields
            | {
                "controller_anchor_tick",
                "controller_anchor_time_s",
                "target_entry_tick",
            }
            if row.get("phase") in causal_predecessor_phase_set
            else expected_row_fields
        )
        for row in rows
    ):
        raise PhaseSnapshotError(
            "phase snapshot manifest entry fields are incomplete or unexpected"
        )
    if tuple(row.get("phase") for row in rows) != PHASE_IDS:
        raise PhaseSnapshotError("phase snapshot order must be P01-P13")

    buffers: list[PhaseSnapshotFileBuffer] = []
    for expected_phase, row in zip(PHASE_IDS, rows, strict=True):
        phase = str(row["phase"])
        phase_dir = _unredirected_path(root / phase, label=f"snapshot directory {phase}")
        _require_within(phase_dir, root, label=f"snapshot directory {phase}")
        identities.append(
            _path_identity(phase_dir, label=f"snapshot directory {phase}", directory=True)
        )
        snapshot_path = _unredirected_path(
            phase_dir / "snapshot.json", label=f"snapshot {phase}"
        )
        checksum_path = _unredirected_path(
            phase_dir / "snapshot.sha256", label=f"snapshot checksum sidecar {phase}"
        )
        _require_within(snapshot_path, root, label=f"snapshot {phase}")
        _require_within(checksum_path, root, label=f"snapshot checksum sidecar {phase}")
        declared_path = row.get("path")
        if (
            not isinstance(declared_path, str)
            or _unredirected_path(
                declared_path, label=f"manifest snapshot path {phase}"
            )
            != snapshot_path
        ):
            raise PhaseSnapshotError(
                f"manifest path does not resolve to the live snapshot for {phase}"
            )
        row_state_hash = _require_sha256(
            row.get("state_sha256"), label=f"manifest state hash for {phase}"
        )
        row_file_hash = _require_sha256(
            row.get("file_sha256"), label=f"manifest file hash for {phase}"
        )
        source_tick = row.get("source_tick")
        if isinstance(source_tick, bool) or not isinstance(source_tick, int) or source_tick < 0:
            raise PhaseSnapshotError(f"manifest source tick is invalid for {phase}")
        source_replay_steps = row.get("source_replay_steps")
        if type(source_replay_steps) is not int or source_replay_steps <= 0:
            raise PhaseSnapshotError(
                f"manifest source replay step count is invalid for {phase}"
            )
        target_entry_tick: int | None = None
        controller_anchor_tick: int | None = None
        controller_anchor_time: float | None = None
        if phase in causal_predecessor_phase_set:
            target_entry_tick = row.get("target_entry_tick")
            controller_anchor_tick = row.get("controller_anchor_tick")
            controller_anchor_time = row.get("controller_anchor_time_s")
            if (
                type(target_entry_tick) is not int
                or target_entry_tick != source_tick + source_replay_steps
            ):
                raise PhaseSnapshotError(
                    f"manifest causal target-entry tick is invalid for {phase}"
                )
            if (
                type(controller_anchor_tick) is not int
                or not source_tick < controller_anchor_tick < target_entry_tick
            ):
                raise PhaseSnapshotError(
                    f"manifest controller anchor tick is invalid for {phase}"
                )
            if (
                isinstance(controller_anchor_time, bool)
                or not isinstance(controller_anchor_time, (int, float))
                or not math.isfinite(float(controller_anchor_time))
                or float(controller_anchor_time)
                != controller_anchor_tick / PHYSICS_HZ
            ):
                raise PhaseSnapshotError(
                    f"manifest controller anchor time is invalid for {phase}"
                )
        elif source_replay_steps != 1:
            raise PhaseSnapshotError(
                f"manifest non-causal source replay must contain one step for {phase}"
            )

        snapshot_identity = _path_identity(
            snapshot_path, label=f"snapshot {phase}", directory=False
        )
        snapshot_bytes = _read_file_bytes(snapshot_path, label=f"snapshot {phase}")
        if _path_identity(
            snapshot_path, label=f"snapshot {phase}", directory=False
        ) != snapshot_identity:
            raise PhaseSnapshotError(f"snapshot {phase} changed while it was read")
        identities.append(snapshot_identity)
        snapshot_file_hash = _sha256_bytes(snapshot_bytes)
        if snapshot_file_hash != row_file_hash:
            raise PhaseSnapshotError(f"file hash mismatch for {phase}")
        payload = dict(
            _decode_json_object(
                snapshot_bytes,
                label=f"snapshot {phase}",
                path=snapshot_path,
            )
        )
        if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("fsm_state") != expected_phase:
            raise PhaseSnapshotError(f"invalid snapshot {phase}")
        if payload.get("source_tick") != source_tick:
            raise PhaseSnapshotError(f"source tick mismatch for {phase}")
        if (
            payload.get("source_trial") != manifest["source_trial"]
            or payload.get("source_trial_path") != manifest["source_trial_path"]
        ):
            raise PhaseSnapshotError(f"source trial binding mismatch for {phase}")
        validate_phase_snapshot_payload_contract(
            payload,
            expected_phase,
            manifest_row=row,
            manifest_source_artifacts=manifest_source_artifacts,
            causal_predecessor_required=(
                phase in causal_predecessor_phase_set
            ),
        )
        state_hash = _require_sha256(
            payload.pop("state_sha256", None), label=f"snapshot state hash for {phase}"
        )
        if _sha256_bytes(_canonical_bytes(payload)) != state_hash or state_hash != row_state_hash:
            raise PhaseSnapshotError(f"state hash mismatch for {phase}")

        checksum_identity = _path_identity(
            checksum_path, label=f"snapshot checksum sidecar {phase}", directory=False
        )
        checksum_bytes = _read_file_bytes(
            checksum_path, label=f"snapshot checksum sidecar {phase}"
        )
        if _path_identity(
            checksum_path,
            label=f"snapshot checksum sidecar {phase}",
            directory=False,
        ) != checksum_identity:
            raise PhaseSnapshotError(
                f"snapshot checksum sidecar {phase} changed while it was read"
            )
        identities.append(checksum_identity)
        checksum_line = f"{snapshot_file_hash}  snapshot.json".encode("ascii")
        if checksum_bytes not in {checksum_line + b"\n", checksum_line + b"\r\n"}:
            raise PhaseSnapshotError(f"checksum sidecar mismatch for {phase}")
        buffers.append(
            PhaseSnapshotFileBuffer(
                phase=phase,
                source_tick=source_tick,
                source_replay_steps=source_replay_steps,
                target_entry_tick=target_entry_tick,
                controller_anchor_tick=controller_anchor_tick,
                controller_anchor_time_s=controller_anchor_time,
                snapshot_path=snapshot_path,
                checksum_path=checksum_path,
                snapshot_bytes=snapshot_bytes,
                checksum_bytes=checksum_bytes,
                file_sha256=snapshot_file_hash,
                state_sha256=state_hash,
                checksum_file_sha256=_sha256_bytes(checksum_bytes),
            )
        )

    for identity in identities:
        current = _path_identity(
            Path(identity[0]),
            label=f"captured phase snapshot path {identity[0]}",
            directory=identity[1] == "directory",
        )
        if not _same_path_identity(identity, current):
            raise PhaseSnapshotError(
                f"phase snapshot path changed during bundle capture: {identity[0]}"
            )
    manifest_hash = _sha256_bytes(manifest_bytes)
    canonical_hash_payload = {
        "schema": BUNDLE_HASH_SCHEMA,
        "manifest_sha256": manifest_hash,
        "snapshots": [
            {
                "phase": entry.phase,
                "file_sha256": entry.file_sha256,
                "state_sha256": entry.state_sha256,
                "checksum_file_sha256": entry.checksum_file_sha256,
            }
            for entry in buffers
        ],
    }
    return ValidatedPhaseSnapshotBundle(
        snapshot_root=root,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_hash,
        snapshots=tuple(buffers),
        bundle_sha256=_sha256_bytes(_canonical_bytes(canonical_hash_payload)),
        source_trial=(
            None if manifest.get("source_trial") is None else str(manifest["source_trial"])
        ),
        filesystem_identity=tuple(identities),
    )


def assert_phase_snapshot_bundle_unchanged(
    expected: ValidatedPhaseSnapshotBundle,
    *,
    canonical_root: Path | str | None = None,
) -> ValidatedPhaseSnapshotBundle:
    current = capture_validated_phase_snapshot_bundle(
        expected.snapshot_root,
        canonical_root=(expected.snapshot_root if canonical_root is None else canonical_root),
    )
    if (
        current.as_record() != expected.as_record()
        or not _same_filesystem_identity(
            expected.filesystem_identity, current.filesystem_identity
        )
    ):
        raise PhaseSnapshotError(
            "phase snapshot bundle differs from the pinned immutable capture"
        )
    return current


def load_validated_phase_snapshot_payload(
    bundle: ValidatedPhaseSnapshotBundle,
    phase: str,
) -> tuple[dict[str, Any], PhaseSnapshotFileBuffer]:
    """Parse and hash one snapshot solely from its pinned immutable bytes."""

    entry = bundle.snapshot(phase)
    snapshot_bytes = entry.snapshot_bytes
    if _sha256_bytes(snapshot_bytes) != entry.file_sha256:
        raise PhaseSnapshotError(f"pinned file hash mismatch for {phase}")
    payload = dict(
        _decode_json_object(
            snapshot_bytes,
            label=f"pinned snapshot {phase}",
            path=entry.snapshot_path,
        )
    )
    if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("fsm_state") != phase:
        raise PhaseSnapshotError(f"invalid pinned snapshot {phase}")
    if payload.get("source_tick") != entry.source_tick:
        raise PhaseSnapshotError(f"pinned source tick mismatch for {phase}")
    if payload.get("source_replay_steps") != entry.source_replay_steps:
        raise PhaseSnapshotError(f"pinned source replay step mismatch for {phase}")
    if payload.get("target_entry_tick") != entry.target_entry_tick:
        raise PhaseSnapshotError(f"pinned target-entry tick mismatch for {phase}")
    if payload.get("controller_anchor_tick") != entry.controller_anchor_tick:
        raise PhaseSnapshotError(f"pinned controller-anchor tick mismatch for {phase}")
    if payload.get("controller_anchor_time_s") != entry.controller_anchor_time_s:
        raise PhaseSnapshotError(f"pinned controller-anchor time mismatch for {phase}")
    manifest = bundle.manifest_payload()
    rows = manifest.get("snapshots")
    manifest_row = next(
        (
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("phase") == phase
        ),
        None,
    ) if isinstance(rows, list) else None
    if manifest_row is None:
        raise PhaseSnapshotError(f"manifest lacks pinned snapshot row for {phase}")
    validate_phase_snapshot_payload_contract(
        payload,
        phase,
        manifest_row=manifest_row,
        manifest_source_artifacts=manifest.get("source_artifacts"),
        causal_predecessor_required=(
            phase in set(manifest.get("causal_predecessor_phases", ()))
        ),
    )
    state_hash = _require_sha256(
        payload.pop("state_sha256", None), label=f"pinned snapshot state hash for {phase}"
    )
    if _sha256_bytes(_canonical_bytes(payload)) != state_hash or state_hash != entry.state_sha256:
        raise PhaseSnapshotError(f"pinned state hash mismatch for {phase}")
    payload["state_sha256"] = state_hash
    return payload, entry


def phase_snapshot_bundle_file_hashes(
    bundle_record: Mapping[str, Any],
) -> dict[str, str]:
    """Return the exact 27-file hash subset required in checkpoint manifests."""

    if bundle_record.get("schema") != BUNDLE_RECORD_SCHEMA:
        raise PhaseSnapshotError("phase snapshot bundle record has the wrong schema")
    if bundle_record.get("phase_count") != len(PHASE_IDS):
        raise PhaseSnapshotError("phase snapshot bundle record must declare 13 phases")
    snapshots = bundle_record.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != len(PHASE_IDS):
        raise PhaseSnapshotError("phase snapshot bundle record must contain 13 entries")
    if any(not isinstance(entry, Mapping) for entry in snapshots):
        raise PhaseSnapshotError("phase snapshot bundle record entries must be objects")
    if tuple(entry.get("phase") for entry in snapshots) != PHASE_IDS:
        raise PhaseSnapshotError("phase snapshot bundle record must be ordered P01-P13")
    expected_record_fields = {
        "phase",
        "source_tick",
        "source_replay_steps",
        "target_entry_tick",
        "controller_anchor_tick",
        "controller_anchor_time_s",
        "snapshot_path",
        "checksum_path",
        "file_sha256",
        "state_sha256",
        "checksum_file_sha256",
    }
    if any(set(entry) != expected_record_fields for entry in snapshots):
        raise PhaseSnapshotError(
            "phase snapshot bundle record entry fields are incomplete or unexpected"
        )
    snapshot_root_value = bundle_record.get("snapshot_root")
    manifest_path_value = bundle_record.get("manifest_path")
    if not isinstance(snapshot_root_value, str) or not isinstance(manifest_path_value, str):
        raise PhaseSnapshotError("phase snapshot bundle paths must be strings")
    snapshot_root = Path(snapshot_root_value).resolve()
    manifest_path_object = Path(manifest_path_value).resolve()
    if manifest_path_object != snapshot_root / "manifest.json":
        raise PhaseSnapshotError("bundle manifest path differs from snapshot root")
    manifest_path = str(manifest_path_object)
    manifest_hash = _require_sha256(
        bundle_record.get("manifest_sha256"), label="bundle manifest hash"
    )
    hashes = {manifest_path: manifest_hash}
    canonical_entries = []
    for entry in snapshots:
        phase = str(entry["phase"])
        source_tick = entry.get("source_tick")
        source_replay_steps = entry.get("source_replay_steps")
        target_entry_tick = entry.get("target_entry_tick")
        controller_anchor_tick = entry.get("controller_anchor_tick")
        controller_anchor_time = entry.get("controller_anchor_time_s")
        if type(source_tick) is not int or source_tick < 0:
            raise PhaseSnapshotError(f"bundle source tick is invalid for {phase}")
        if type(source_replay_steps) is not int or source_replay_steps <= 0:
            raise PhaseSnapshotError(
                f"bundle source replay step count is invalid for {phase}"
            )
        if target_entry_tick is None:
            if (
                source_replay_steps != 1
                or controller_anchor_tick is not None
                or controller_anchor_time is not None
            ):
                raise PhaseSnapshotError(
                    f"bundle non-causal replay boundary is invalid for {phase}"
                )
        else:
            if (
                phase != "P10"
                or type(target_entry_tick) is not int
                or target_entry_tick != source_tick + source_replay_steps
            ):
                raise PhaseSnapshotError(
                    f"bundle target-entry tick is invalid for {phase}"
                )
            if (
                type(controller_anchor_tick) is not int
                or not source_tick < controller_anchor_tick < target_entry_tick
            ):
                raise PhaseSnapshotError(
                    f"bundle controller anchor tick is invalid for {phase}"
                )
            if (
                isinstance(controller_anchor_time, bool)
                or not isinstance(controller_anchor_time, (int, float))
                or not math.isfinite(float(controller_anchor_time))
                or float(controller_anchor_time)
                != controller_anchor_tick / PHYSICS_HZ
            ):
                raise PhaseSnapshotError(
                    f"bundle controller anchor time is invalid for {phase}"
                )
        expected_phase_root = snapshot_root / phase
        snapshot_path = str(Path(str(entry.get("snapshot_path", ""))).resolve())
        checksum_path = str(Path(str(entry.get("checksum_path", ""))).resolve())
        if Path(snapshot_path) != expected_phase_root / "snapshot.json" or Path(
            checksum_path
        ) != expected_phase_root / "snapshot.sha256":
            raise PhaseSnapshotError(f"bundle paths are noncanonical for {phase}")
        file_hash = _require_sha256(
            entry.get("file_sha256"), label=f"bundle file hash for {phase}"
        )
        state_hash = _require_sha256(
            entry.get("state_sha256"), label=f"bundle state hash for {phase}"
        )
        checksum_hash = _require_sha256(
            entry.get("checksum_file_sha256"),
            label=f"bundle checksum file hash for {phase}",
        )
        if snapshot_path in hashes or checksum_path in hashes or snapshot_path == checksum_path:
            raise PhaseSnapshotError("phase snapshot bundle record contains duplicate paths")
        hashes[snapshot_path] = file_hash
        hashes[checksum_path] = checksum_hash
        canonical_entries.append(
            {
                "phase": phase,
                "file_sha256": file_hash,
                "state_sha256": state_hash,
                "checksum_file_sha256": checksum_hash,
            }
        )
    canonical_payload = {
        "schema": BUNDLE_HASH_SCHEMA,
        "manifest_sha256": manifest_hash,
        "snapshots": canonical_entries,
    }
    declared_bundle_hash = _require_sha256(
        bundle_record.get("bundle_sha256"), label="phase snapshot bundle hash"
    )
    if _sha256_bytes(_canonical_bytes(canonical_payload)) != declared_bundle_hash:
        raise PhaseSnapshotError("phase snapshot canonical bundle hash mismatch")
    if len(hashes) != 27:
        raise PhaseSnapshotError("phase snapshot bundle must bind exactly 27 files")
    return hashes


def validated_phase_snapshot_bundle_record(
    output_root: Path | str,
    *,
    canonical_root: Path | str | None = DEFAULT_PHASE_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    """Return immutable evidence for one fully validated P01-P13 snapshot bundle."""

    return capture_validated_phase_snapshot_bundle(
        output_root, canonical_root=canonical_root
    ).as_record()


def validate_phase_snapshots(
    output_root: Path | str,
    *,
    canonical_root: Path | str | None = DEFAULT_PHASE_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    return capture_validated_phase_snapshot_bundle(
        output_root, canonical_root=canonical_root
    ).manifest_payload()
