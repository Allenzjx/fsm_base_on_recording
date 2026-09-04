"""Extract and validate reset-only phase-entry snapshots from a frozen success trial."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SNAPSHOT_SCHEMA = "wlr50_clean.ppo_phase_entry_snapshot.v1"
MANIFEST_SCHEMA = "wlr50_clean.ppo_phase_snapshot_manifest.v1"
PHASE_IDS = tuple(f"P{i:02d}" for i in range(1, 14))
PHYSICS_HZ = 120.0


class PhaseSnapshotError(ValueError):
    pass


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    validate_phase_snapshots(output)
    return manifest


def validate_phase_snapshots(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or int(manifest.get("phase_count", -1)) != 13:
        raise PhaseSnapshotError("invalid phase snapshot manifest")
    rows = manifest.get("snapshots", [])
    if tuple(row.get("phase") for row in rows) != PHASE_IDS:
        raise PhaseSnapshotError("phase snapshot order must be P01-P13")
    for row in rows:
        phase = str(row["phase"])
        path = root / phase / "snapshot.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("fsm_state") != phase:
            raise PhaseSnapshotError(f"invalid snapshot {phase}")
        state_hash = str(payload.pop("state_sha256"))
        if _sha256_bytes(_canonical_bytes(payload)) != state_hash or state_hash != row["state_sha256"]:
            raise PhaseSnapshotError(f"state hash mismatch for {phase}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["file_sha256"]:
            raise PhaseSnapshotError(f"file hash mismatch for {phase}")
    return manifest
