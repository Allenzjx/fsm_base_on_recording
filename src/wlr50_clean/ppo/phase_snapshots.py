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


SNAPSHOT_SCHEMA = "wlr50_clean.ppo_phase_entry_snapshot.v1"
MANIFEST_SCHEMA = "wlr50_clean.ppo_phase_snapshot_manifest.v1"
BUNDLE_RECORD_SCHEMA = "wlr50_clean.ppo_phase_snapshot_bundle_record.v1"
BUNDLE_HASH_SCHEMA = "wlr50_clean.ppo_phase_snapshot_bundle_hash.v1"
PHASE_IDS = tuple(f"P{i:02d}" for i in range(1, 14))
PHYSICS_HZ = 120.0
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE_SNAPSHOT_ROOT = PROJECT_ROOT / "reference" / "ppo_phase_snapshots"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PhaseSnapshotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PhaseSnapshotFileBuffer:
    """Immutable bytes and validated digests for one phase-entry snapshot."""

    phase: str
    source_tick: int
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


def phase_entry_ticks(trial_dir: Path | str) -> dict[str, int]:
    trial = Path(trial_dir).resolve()
    transitions = trial / "state_transitions.jsonl"
    if not transitions.is_file():
        raise FileNotFoundError(transitions)
    result: dict[str, int] = {"P01": 0}
    for row in _read_jsonl(transitions):
        phase = str(row.get("state_id"))
        if phase in PHASE_IDS and row.get("to_lifecycle") == "EXECUTE_MOTION" and phase not in result:
            time_s = float(row["sim_time_s"])
            tick = int(round(time_s * PHYSICS_HZ))
            if not math.isclose(time_s, tick / PHYSICS_HZ, abs_tol=2.0e-6):
                raise PhaseSnapshotError(f"{phase} entry is not on the 120 Hz lattice")
            result[phase] = tick
    if tuple(result) != PHASE_IDS:
        missing = [phase for phase in PHASE_IDS if phase not in result]
        raise PhaseSnapshotError(f"trial lacks phase-entry transitions: {missing}")
    return result


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


def _event_latches(trial: Path, entry_tick: int) -> dict[str, dict[str, int | bool | None]]:
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
    source = trial / "leg_crossing_events.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    names = {
        "ACTIVE_LIFT": ("active_lift", "active_lift_tick"),
        "FRONT_FACE_CROSSED": ("front_face_crossed", "front_face_crossed_tick"),
        "TOP_LOADED": ("top_loaded", "top_loaded_tick"),
    }
    for row in _read_jsonl(source):
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


def _snapshot_payload(
    *,
    trial: Path,
    trial_id: str,
    phase: str,
    tick: int,
    observation: Mapping[str, Any],
    command: Mapping[str, Any],
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
    return {
        "schema": SNAPSHOT_SCHEMA,
        "reset_use": "TRAINING_RESET_STATE_WRITE",
        "in_episode_root_write": "FORBIDDEN_IN_EPISODE_ROOT_WRITE",
        "source_trial": trial_id,
        "source_trial_path": str(trial),
        "source_tick": tick,
        "source_time_s": tick / PHYSICS_HZ,
        "fsm_state": phase,
        "fsm_lifecycle": "EXECUTE_MOTION",
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
        "contact_event_latches": _event_latches(trial, tick),
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


def build_phase_snapshots(
    trial_dir: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    trial = Path(trial_dir).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"snapshot output already exists: {output}")
    manifest_source = json.loads((trial / "trial_manifest.json").read_text(encoding="utf-8"))
    trial_id = str(manifest_source.get("trial_id", trial.name))
    ticks_by_phase = phase_entry_ticks(trial)
    ticks = set(ticks_by_phase.values())
    observations = _rows_at_ticks(trial / "observation_120hz.jsonl", ticks, "physics_tick")
    commands = _rows_at_ticks(trial / "full12_commands_120hz.jsonl", ticks, "control_physics_tick")
    level_reference_orientation = list(observations[0]["base"]["orientation_wxyz"])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for phase, tick in ticks_by_phase.items():
        payload = _snapshot_payload(
            trial=trial,
            trial_id=trial_id,
            phase=phase,
            tick=tick,
            observation=observations[tick],
            command=commands[tick],
            level_reference_orientation_wxyz=level_reference_orientation,
        )
        state_hash = _sha256_bytes(_canonical_bytes(payload))
        complete = {**payload, "state_sha256": state_hash}
        phase_dir = output / phase
        phase_dir.mkdir()
        snapshot_path = phase_dir / "snapshot.json"
        snapshot_path.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
        file_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        (phase_dir / "snapshot.sha256").write_text(f"{file_hash}  snapshot.json\n", encoding="ascii")
        rows.append(
            {
                "phase": phase,
                "source_tick": tick,
                "state_sha256": state_hash,
                "file_sha256": file_hash,
                "path": str(snapshot_path),
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_trial": trial_id,
        "source_trial_path": str(trial),
        "physics_hz": PHYSICS_HZ,
        "phase_count": len(rows),
        "snapshots": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate_phase_snapshots(output, canonical_root=output)
    return manifest


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
    if type(manifest.get("phase_count")) is not int or manifest["phase_count"] != len(PHASE_IDS):
        raise PhaseSnapshotError("phase snapshot manifest must declare exactly 13 phases")
    physics_hz = manifest.get("physics_hz")
    if isinstance(physics_hz, bool) or not isinstance(physics_hz, (int, float)):
        raise PhaseSnapshotError("phase snapshot manifest physics_hz must be numeric")
    if float(physics_hz) != PHYSICS_HZ:
        raise PhaseSnapshotError(f"phase snapshot manifest physics_hz must be {PHYSICS_HZ:g}")
    rows = manifest.get("snapshots")
    if not isinstance(rows, list) or len(rows) != len(PHASE_IDS):
        raise PhaseSnapshotError("phase snapshot manifest must contain exactly 13 entries")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PhaseSnapshotError("phase snapshot manifest entries must be JSON objects")
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
        if current != identity:
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
        or current.filesystem_identity != expected.filesystem_identity
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
