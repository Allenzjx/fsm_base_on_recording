from __future__ import annotations

import json

import pytest

from wlr50_clean.ppo.phase_snapshots import (
    PHASE_IDS,
    PhaseSnapshotError,
    build_phase_snapshots,
    validate_phase_snapshots,
)


def _observation(tick: int) -> dict:
    joints = (
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    )
    wheels = ("front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")
    bodies = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
    return {
        "physics_tick": tick,
        "base": {"position_w_m": [0, 0, .1], "orientation_wxyz": [1, 0, 0, 0], "linear_velocity_w_m_s": [0, 0, 0], "angular_velocity_w_rad_s": [0, 0, 0]},
        "joints": {name: {"position_deg": 0, "velocity_deg_s": 0} for name in joints},
        "wheels": {name: {"velocity_rad_s": 0, "body_name": body, "center_w_m": [0, 0, .05], "bottom_w_m": [0, 0, 0]} for name, body in zip(wheels, bodies)},
        "contacts": {body: {"contact_class": "GROUND", "ground": {"active": True}, "obstacle": {"active": False}} for body in bodies},
        "obstacle": {"front_x_m": .5, "back_x_m": 2.5, "left_y_m": 1, "right_y_m": -1, "bottom_z_m": 0, "top_z_m": .05},
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_and_validate_all_phase_snapshots(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trial_manifest.json").write_text(json.dumps({"trial_id": "success"}), encoding="utf-8")
    transitions = []
    for index, phase in enumerate(PHASE_IDS):
        transitions.append({"state_id": phase, "to_lifecycle": "EXECUTE_MOTION", "sim_time_s": index / 120})
    _write_jsonl(trial / "state_transitions.jsonl", transitions)
    _write_jsonl(trial / "observation_120hz.jsonl", [_observation(i) for i in range(13)])
    _write_jsonl(
        trial / "full12_commands_120hz.jsonl",
        [{"control_physics_tick": i, "nominal_full12": [0] * 12, "applied_full12": [0] * 12} for i in range(13)],
    )
    _write_jsonl(trial / "leg_crossing_events.jsonl", [])
    out = tmp_path / "snapshots"
    manifest = build_phase_snapshots(trial, out)
    assert manifest["phase_count"] == 13
    assert validate_phase_snapshots(out)["phase_count"] == 13
    p08 = json.loads((out / "P08" / "snapshot.json").read_text(encoding="utf-8"))
    assert p08["source_tick"] == 7
    assert p08["phase_history"] == list(PHASE_IDS[:7])
    assert p08["reset_use"] == "TRAINING_RESET_STATE_WRITE"


def test_refuses_to_overwrite_snapshot_set(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        build_phase_snapshots(tmp_path / "missing", target)


def test_validation_detects_tampering(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "bad", "phase_count": 0}), encoding="utf-8")
    with pytest.raises(PhaseSnapshotError):
        validate_phase_snapshots(root)
