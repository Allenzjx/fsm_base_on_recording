from __future__ import annotations

import hashlib
import json
import copy
from collections import Counter
from pathlib import Path

import pytest

from wlr50_clean.ppo import phase_snapshots as phase_snapshots_subject
from wlr50_clean.ppo.phase_snapshots import (
    MANIFEST_SCHEMA,
    PHASE_IDS,
    SNAPSHOT_SCHEMA,
    PhaseSnapshotError,
    capture_validated_phase_snapshot_bundle,
    build_phase_snapshots,
    load_validated_phase_snapshot_payload,
    phase_snapshot_bundle_file_hashes,
    validate_phase_snapshot_payload_contract,
    validate_phase_snapshots,
    validated_phase_snapshot_bundle_record,
)


def test_source_ack_replay_fields_form_an_exact_partition() -> None:
    all_fields = set(phase_snapshots_subject.SOURCE_ACK_MATCH_FIELDS)
    invariant = set(phase_snapshots_subject.SOURCE_ACK_REPLAY_INVARIANT_FIELDS)
    diagnostic = set(phase_snapshots_subject.SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS)

    assert invariant.isdisjoint(diagnostic)
    assert invariant | diagnostic == all_fields
    assert "servo_tracking_feedback_sampled" in diagnostic
    assert "servo_tracking_feedback_sampled" not in invariant


def test_directory_identity_ignores_child_churn_but_rejects_replacement(
    tmp_path: Path,
) -> None:
    identity = phase_snapshots_subject._path_identity(
        tmp_path.resolve(), label="test directory", directory=True
    )
    metadata_churn = list(identity)
    metadata_churn[4] += 1
    metadata_churn[5] += 1
    metadata_churn[6] += 1
    assert phase_snapshots_subject._same_path_identity(
        identity, tuple(metadata_churn)
    )

    replacement = list(identity)
    replacement[3] += 1
    assert not phase_snapshots_subject._same_path_identity(
        identity, tuple(replacement)
    )


def _observation(tick: int) -> dict:
    joints = (
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    )
    wheels = ("front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")
    bodies = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
    ground_active = tick % 3 == 0
    return {
        "physics_tick": tick,
        "base": {"position_w_m": [float(tick), 0, .1], "orientation_wxyz": [1, 0, 0, 0], "linear_velocity_w_m_s": [0, 0, 0], "angular_velocity_w_rad_s": [0, 0, 0]},
        "joints": {name: {"position_deg": 0, "velocity_deg_s": 0} for name in joints},
        "wheels": {name: {"velocity_rad_s": float(tick), "body_name": body, "center_w_m": [0, 0, .05], "bottom_w_m": [0, 0, 0]} for name, body in zip(wheels, bodies)},
        "contacts": {body: {"contact_class": "GROUND" if ground_active else "AIR", "ground": {"active": ground_active}, "obstacle": {"active": False}} for body in bodies},
        "obstacle": {"front_x_m": .5, "back_x_m": 2.5, "left_y_m": 1, "right_y_m": -1, "bottom_z_m": 0, "top_z_m": .05},
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(tick: int, *, state_id: str, lifecycle: str) -> dict:
    servos = (
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    )
    wheels = (
        "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle"
    )
    zero12 = [0.0] * 12
    ack = {
        "schema": "wlr50_clean.atomic_full12_ack.v1",
        "physics_tick": 180 + tick,
        "physics_dt_s": 1.0 / 120.0,
        "write_count": 181 + tick,
        "articulation_writes_this_call": 1,
        "canonical_order": list(servos + wheels),
        "requested_full12": zero12,
        "applied_full12": zero12,
        "drive_target_full12": zero12,
        "native_drive_target_full12": zero12,
        "drive_feedback_bias_requested_full12": zero12,
        "drive_feedback_bias_realized_full12": zero12,
        "drive_feedback_final_slew_limit_deg_per_tick": 1.25,
        "command_was_clamped": False,
        "servo_applied_drive_command_deg": [0.0] * 8,
        "servo_native_drive_command_deg": [0.0] * 8,
        "servo_tracking_compensation_deg": [0.0] * 8,
        "servo_nominal_target_reached": [True] * 8,
        "servo_tracking_active": [False] * 8,
        "tracking_servo_names": [],
        "servo_tracking_feedback_sample_tick": 180 + tick,
        "servo_tracking_feedback_sampled": False,
        "servo_joint_ids": list(range(8)),
        "wheel_joint_ids": list(range(8, 12)),
        "servo_target_physical_rad": [0.0] * 8,
        "wheel_target_physical_rad_s": [-0.0, 0.0, -0.0, 0.0],
        "motion_start_skew_s": 0.0,
    }
    return {
        "control_physics_tick": tick,
        "state_id": state_id,
        "lifecycle": lifecycle,
        "nominal_full12": zero12,
        "applied_full12": zero12,
        "drive_target_full12": zero12,
        "native_drive_target_full12": zero12,
        "drive_feedback_bias_requested_full12": zero12,
        "drive_feedback_bias_realized_full12": zero12,
        "tracking_servo_names": [],
        "atomic_ack": ack,
    }


def _write_source_trial(
    tmp_path,
    *,
    p09_predecessor_state: str = "P09",
    p09_predecessor_lifecycle: str = "VERIFY_RESULT",
    p10_predecessor_lifecycle: str = "WAIT_ENTRY",
    signed_positive_rebound_required: bool = True,
    include_p09_top_loaded: bool = True,
):
    trial = tmp_path / "trial"
    trial.mkdir()
    p10_physical_source_tick = PHASE_IDS.index("P10")
    p09_top_loaded_tick = p10_physical_source_tick + 2
    p09_verify_tick = p10_physical_source_tick + 6
    p10_wait_entry_tick = p10_physical_source_tick + 8
    entry_ticks = {
        phase: (
            index
            if index < PHASE_IDS.index("P10")
            else index + 18
        )
        for index, phase in enumerate(PHASE_IDS)
    }
    transitions = []
    for phase in PHASE_IDS:
        guards = []
        if phase == "P10":
            guards = [
                {
                    "name": "reference_entry_compatible",
                    "passed": True,
                    "value": {
                        "rear_right_knee_velocity": {
                            "actual_deg_s": 23.5,
                            "signed_positive_rebound_required": (
                                signed_positive_rebound_required
                            ),
                        }
                    },
                }
            ]
        if phase == "P10":
            transitions.append(
                {
                    "state_id": phase,
                    "from_lifecycle": "DONE",
                    "to_lifecycle": "WAIT_ENTRY",
                    "sim_time_s": p10_wait_entry_tick / 120,
                    "details": {},
                }
            )
        transitions.append(
            {
                "state_id": phase,
                "from_lifecycle": "WAIT_ENTRY",
                "to_lifecycle": "EXECUTE_MOTION",
                "sim_time_s": entry_ticks[phase] / 120,
                "details": {"guards": guards},
            }
        )
        if phase == "P09":
            transitions.append(
                {
                    "state_id": phase,
                    "from_lifecycle": "EXECUTE_MOTION",
                    "to_lifecycle": "VERIFY_RESULT",
                    "sim_time_s": p09_verify_tick / 120,
                    "details": {},
                }
            )
    _write_jsonl(trial / "state_transitions.jsonl", transitions)
    stream_length = max(entry_ticks.values()) + 1
    _write_jsonl(
        trial / "observation_120hz.jsonl",
        [_observation(i) for i in range(stream_length)],
    )
    phase_by_entry_tick = {tick: phase for phase, tick in entry_ticks.items()}
    commands = []
    for tick in range(stream_length):
        if p10_physical_source_tick <= tick < p09_verify_tick:
            state_id = p09_predecessor_state
            lifecycle = "EXECUTE_MOTION"
        elif p09_verify_tick <= tick < p10_wait_entry_tick:
            state_id = p09_predecessor_state
            lifecycle = p09_predecessor_lifecycle
        elif p10_wait_entry_tick <= tick < entry_ticks["P10"]:
            state_id = "P10"
            lifecycle = p10_predecessor_lifecycle
        else:
            state_id = phase_by_entry_tick[tick]
            lifecycle = "EXECUTE_MOTION"
        commands.append(_command(tick, state_id=state_id, lifecycle=lifecycle))
    _write_jsonl(
        trial / "full12_commands_120hz.jsonl",
        commands,
    )
    leg_crossing_events = [
        *(
            [
                {
                    "physics_tick": p09_top_loaded_tick,
                    "state_id": "P09",
                    "leg": "RR",
                    "event": "TOP_LOADED",
                    "evidence": {
                        "passed": True,
                        "value": {
                            "latched": True,
                            "latch_tick": p09_top_loaded_tick,
                        },
                    },
                }
            ]
            if include_p09_top_loaded
            else []
        ),
        {
            "physics_tick": p10_physical_source_tick + 3,
            "leg": "RR",
            "event": "ACTIVE_LIFT",
        },
    ]
    _write_jsonl(
        trial / "leg_crossing_events.jsonl",
        leg_crossing_events,
    )
    artifact_rows = {
        "observation": ("observation_120hz.jsonl"),
        "command": ("full12_commands_120hz.jsonl"),
        "transition": ("state_transitions.jsonl"),
        "leg_crossing": ("leg_crossing_events.jsonl"),
    }
    manifest = {
        "trial_id": "success",
        "physics_hz": 120.0,
        "settle_ticks": 180,
        "environment_initialization": {
            "records": [
                {"joint_name": name, "standing_pose_deg": 0.0}
                for name in (
                    "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
                    "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
                )
            ]
        },
        "artifact_files": {
            role: {
                "path": name,
                "bytes": (trial / name).stat().st_size,
                "sha256": _sha(trial / name),
            }
            for role, name in artifact_rows.items()
        },
    }
    (trial / "trial_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return trial


def _build_snapshot_set(tmp_path):
    trial = _write_source_trial(tmp_path)
    out = tmp_path / "snapshots"
    build_phase_snapshots(trial, out)
    return out


def test_build_and_validate_all_phase_snapshots(tmp_path):
    out = _build_snapshot_set(tmp_path)
    manifest = validate_phase_snapshots(out, canonical_root=out)
    assert manifest["phase_count"] == 13
    p08 = json.loads((out / "P08" / "snapshot.json").read_text(encoding="utf-8"))
    assert p08["source_tick"] == 7
    assert p08["phase_history"] == list(PHASE_IDS[:7])
    assert p08["reset_use"] == "TRAINING_RESET_STATE_WRITE"
    assert p08["schema"] == SNAPSHOT_SCHEMA
    assert p08["source_command"]["expected_atomic_ack"][
        "drive_target_full12"
    ] == [0.0] * 12
    assert p08["source_replay_steps"] == 1
    assert p08["source_commands"] == [p08["source_command"]]
    assert set(p08["source_artifacts"]) == {
        "trial_manifest",
        "command",
        "observation",
        "transition",
        "leg_crossing",
    }
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["causal_predecessor_phases"] == ["P10"]

    p10 = json.loads((out / "P10" / "snapshot.json").read_text(encoding="utf-8"))
    p10_manifest = next(
        row for row in manifest["snapshots"] if row["phase"] == "P10"
    )
    assert p10["source_tick"] == 9
    assert p10["predecessor_verify_tick"] == 15
    assert p10["predecessor_verify_time_s"] == 15 / 120
    assert p10["controller_anchor_tick"] == 17
    assert p10["controller_anchor_time_s"] == 17 / 120
    assert p10["target_entry_tick"] == 27
    assert p10["source_replay_steps"] == 18
    assert len(p10["source_commands"]) == 18
    assert [
        row["control_physics_tick"] for row in p10["source_commands"]
    ] == list(range(9, 27))
    assert p10["source_command"] == p10["source_commands"][0]
    assert p10["fsm_state"] == "P10"
    assert p10["fsm_lifecycle"] == "WAIT_ENTRY"
    assert p10["source_command"]["source_fsm_state"] == "P09"
    assert p10["source_command"]["source_fsm_lifecycle"] == "EXECUTE_MOTION"
    assert p10["source_command"]["target_entry_tick"] == 27
    assert all(
        row["source_fsm_state"] == "P09"
        and row["source_fsm_lifecycle"] == "EXECUTE_MOTION"
        and row["target_entry_tick"] == 27
        for row in p10["source_commands"][:6]
    )
    assert all(
        row["source_fsm_state"] == "P09"
        and row["source_fsm_lifecycle"] == "VERIFY_RESULT"
        and row["target_entry_tick"] == 27
        for row in p10["source_commands"][6:8]
    )
    assert all(
        row["source_fsm_state"] == "P10"
        and row["source_fsm_lifecycle"] == "WAIT_ENTRY"
        and row["target_entry_tick"] == 27
        for row in p10["source_commands"][8:]
    )
    assert p10["root_state"]["position_w_m"][0] == 9.0
    assert p10["wheel_state"]["logical_velocity_rad_s"] == [9.0] * 4
    assert all(row["class"] == "GROUND" for row in p10["contact_state"].values())
    assert p10["contact_event_latches"]["RR"]["active_lift"] is True
    assert p10["contact_event_latches"]["RR"]["active_lift_tick"] == 12
    assert p10["contact_event_latches"]["RR"]["top_loaded"] is True
    assert p10["contact_event_latches"]["RR"]["top_loaded_tick"] == 11
    assert p10_manifest["source_tick"] == 9
    assert p10_manifest["predecessor_verify_tick"] == 15
    assert p10_manifest["predecessor_verify_time_s"] == 15 / 120
    assert p10_manifest["controller_anchor_tick"] == 17
    assert p10_manifest["controller_anchor_time_s"] == 17 / 120
    assert p10_manifest["source_replay_steps"] == 18
    assert p10_manifest["target_entry_tick"] == 27

    p11 = json.loads((out / "P11" / "snapshot.json").read_text(encoding="utf-8"))
    p11_manifest = next(
        row for row in manifest["snapshots"] if row["phase"] == "P11"
    )
    assert p11["source_tick"] == 28
    assert p11["source_replay_steps"] == 1
    assert p11["source_commands"] == [p11["source_command"]]
    assert p11["fsm_lifecycle"] == "EXECUTE_MOTION"
    assert "target_entry_tick" not in p11
    assert "predecessor_verify_tick" not in p11
    assert "predecessor_verify_time_s" not in p11
    assert "controller_anchor_tick" not in p11
    assert "controller_anchor_time_s" not in p11
    assert "source_fsm_lifecycle" not in p11["source_command"]
    assert "target_entry_tick" not in p11_manifest
    assert "predecessor_verify_tick" not in p11_manifest
    assert "predecessor_verify_time_s" not in p11_manifest
    assert "controller_anchor_tick" not in p11_manifest
    assert "controller_anchor_time_s" not in p11_manifest


def test_signed_positive_entry_detection_is_transition_evidence_driven(tmp_path):
    trial = _write_source_trial(
        tmp_path,
        signed_positive_rebound_required=False,
        include_p09_top_loaded=False,
    )
    out = tmp_path / "snapshots"

    manifest = build_phase_snapshots(trial, out)
    p10 = json.loads((out / "P10" / "snapshot.json").read_text(encoding="utf-8"))

    assert manifest["causal_predecessor_phases"] == []
    assert p10["source_tick"] == 27
    assert p10["source_replay_steps"] == 1
    assert p10["source_commands"] == [p10["source_command"]]
    assert p10["fsm_lifecycle"] == "EXECUTE_MOTION"
    assert "target_entry_tick" not in p10
    assert "predecessor_verify_tick" not in p10
    assert "predecessor_verify_time_s" not in p10
    assert "controller_anchor_tick" not in p10
    assert "controller_anchor_time_s" not in p10


def test_trial043_hybrid_boundary_is_exactly_7577_7776_7784_7794() -> None:
    rows = []
    for index, phase in enumerate(PHASE_IDS[1:], 1):
        if phase == "P09":
            rows.append(
                {
                    "state_id": phase,
                    "from_lifecycle": "EXECUTE_MOTION",
                    "to_lifecycle": "VERIFY_RESULT",
                    "sim_time_s": 7776 / 120,
                }
            )
        if phase == "P10":
            rows.append(
                {
                    "state_id": phase,
                    "from_lifecycle": "DONE",
                    "to_lifecycle": "WAIT_ENTRY",
                    "sim_time_s": 7784 / 120,
                }
            )
        target_tick = 7794 if phase == "P10" else index
        rows.append(
            {
                "state_id": phase,
                "from_lifecycle": "WAIT_ENTRY",
                "to_lifecycle": "EXECUTE_MOTION",
                "sim_time_s": target_tick / 120,
                "details": {
                    "guards": (
                        [{"signed_positive_rebound_required": True}]
                        if phase == "P10"
                        else []
                    )
                },
            }
        )

    leg_crossing_rows = [
        {
            "physics_tick": 7579,
            "state_id": "P09",
            "leg": "RR",
            "event": "TOP_LOADED",
            "evidence": {
                "passed": True,
                "value": {"latched": True, "latch_tick": 7579},
            },
        }
    ]
    boundary = phase_snapshots_subject._phase_entry_boundaries_from_rows(
        rows, leg_crossing_rows=leg_crossing_rows
    )["P10"]

    assert boundary.source_tick == 7577
    assert boundary.predecessor_verify_tick == 7776
    assert boundary.controller_anchor_tick == 7784
    assert boundary.target_entry_tick == 7794
    assert boundary.source_replay_steps == 217


def test_later_recovery_verify_transition_does_not_ambiguate_p10_predecessor() -> None:
    rows = []
    for index, phase in enumerate(PHASE_IDS[1:], 1):
        if phase == "P09":
            rows.append(
                {
                    "state_id": "P09",
                    "from_lifecycle": "EXECUTE_MOTION",
                    "to_lifecycle": "VERIFY_RESULT",
                    "sim_time_s": 7776 / 120,
                }
            )
        if phase == "P10":
            rows.append(
                {
                    "state_id": "P10",
                    "from_lifecycle": "DONE",
                    "to_lifecycle": "WAIT_ENTRY",
                    "sim_time_s": 7784 / 120,
                }
            )
        rows.append(
            {
                "state_id": phase,
                "from_lifecycle": "WAIT_ENTRY",
                "to_lifecycle": "EXECUTE_MOTION",
                "sim_time_s": (7794 if phase == "P10" else index) / 120,
                "details": {
                    "guards": (
                        [{"signed_positive_rebound_required": True}]
                        if phase == "P10"
                        else []
                    )
                },
            }
        )
    rows.extend(
        [
            {
                "state_id": "P13",
                "from_lifecycle": "EXECUTE_MOTION",
                "to_lifecycle": "VERIFY_RESULT",
                "sim_time_s": 9000 / 120,
            },
            {
                "state_id": "P13",
                "from_lifecycle": "EXECUTE_MOTION",
                "to_lifecycle": "VERIFY_RESULT",
                "sim_time_s": 9010 / 120,
            },
        ]
    )

    boundary = phase_snapshots_subject._phase_entry_boundaries_from_rows(
        rows,
        leg_crossing_rows=[
            {
                "physics_tick": 7579,
                "state_id": "P09",
                "leg": "RR",
                "event": "TOP_LOADED",
                "evidence": {
                    "passed": True,
                    "value": {"latched": True, "latch_tick": 7579},
                },
            }
        ],
    )["P10"]

    assert boundary.source_tick == 7577
    assert boundary.predecessor_verify_tick == 7776
    assert boundary.controller_anchor_tick == 7784
    assert boundary.target_entry_tick == 7794


def test_causal_predecessor_requires_wait_entry_source_command(tmp_path):
    trial = _write_source_trial(
        tmp_path, p10_predecessor_lifecycle="EXECUTE_MOTION"
    )
    out = tmp_path / "snapshots"

    with pytest.raises(PhaseSnapshotError, match="source command state/lifecycle"):
        build_phase_snapshots(trial, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "source_kwargs",
    (
        {"p09_predecessor_state": "P10"},
        {"p09_predecessor_lifecycle": "EXECUTE_MOTION"},
    ),
)
def test_hybrid_causal_source_requires_real_p09_verify_context(
    tmp_path, source_kwargs
):
    trial = _write_source_trial(tmp_path, **source_kwargs)

    with pytest.raises(PhaseSnapshotError, match="source command state/lifecycle"):
        build_phase_snapshots(trial, tmp_path / "snapshots")


def test_signed_positive_entry_requires_semantic_wait_entry_anchor(tmp_path):
    trial = _write_source_trial(tmp_path)
    transition_path = trial / "state_transitions.jsonl"
    rows = [
        json.loads(line)
        for line in transition_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(
        transition_path,
        [
            row
            for row in rows
            if not (
                row["state_id"] == "P10"
                and row["from_lifecycle"] == "DONE"
                and row["to_lifecycle"] == "WAIT_ENTRY"
            )
        ],
    )
    manifest_path = trial / "trial_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_files"]["transition"].update(
        {
            "bytes": transition_path.stat().st_size,
            "sha256": _sha(transition_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PhaseSnapshotError, match="prior WAIT_ENTRY start"):
        build_phase_snapshots(trial, tmp_path / "snapshots")


@pytest.mark.parametrize(
    ("surface", "expected_error"),
    (
        ("payload_target", "causal target-entry tick"),
        ("manifest_target", "hybrid replay boundary"),
        ("predecessor_equals_source", "predecessor verify tick"),
        ("predecessor_verify_tick", "predecessor verify tick"),
        ("predecessor_verify_time", "predecessor verify time"),
        ("controller_anchor_tick", "controller anchor tick"),
        ("controller_anchor_time", "controller anchor time"),
        ("source_lifecycle", "causal source-command context"),
        ("snapshot_lifecycle", "FSM lifecycle"),
    ),
)
def test_causal_predecessor_audit_fields_fail_closed(
    tmp_path: Path, surface: str, expected_error: str
) -> None:
    out = _build_snapshot_set(tmp_path)
    payload = json.loads(
        (out / "P10" / "snapshot.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest_row = next(
        row for row in manifest["snapshots"] if row["phase"] == "P10"
    )

    if surface == "payload_target":
        payload["target_entry_tick"] += 1
    elif surface == "manifest_target":
        manifest_row["target_entry_tick"] += 1
    elif surface == "predecessor_equals_source":
        payload["predecessor_verify_tick"] = payload["source_tick"]
        payload["predecessor_verify_time_s"] = payload["source_time_s"]
    elif surface == "predecessor_verify_tick":
        payload["predecessor_verify_tick"] = payload["target_entry_tick"]
    elif surface == "predecessor_verify_time":
        payload["predecessor_verify_time_s"] += 1.0 / 120.0
    elif surface == "controller_anchor_tick":
        payload["controller_anchor_tick"] = payload["source_tick"]
    elif surface == "controller_anchor_time":
        payload["controller_anchor_time_s"] += 1.0 / 120.0
    elif surface == "source_lifecycle":
        payload["source_command"]["source_fsm_lifecycle"] = "VERIFY_RESULT"
        payload["source_commands"][0]["source_fsm_lifecycle"] = "VERIFY_RESULT"
    else:
        payload["fsm_lifecycle"] = "EXECUTE_MOTION"

    with pytest.raises(PhaseSnapshotError, match=expected_error):
        validate_phase_snapshot_payload_contract(
            payload,
            "P10",
            manifest_row=manifest_row,
            manifest_source_artifacts=manifest["source_artifacts"],
            causal_predecessor_required=True,
        )


@pytest.mark.parametrize(
    "surface",
    (
        "step_count",
        "missing_row",
        "discontinuous_tick",
        "tail_lifecycle",
        "controller_row_context",
        "target_command_included",
        "tail_atomic_tick",
        "tail_mapper_continuity",
        "tail_actuation_hash",
        "tail_source_row_hash",
        "first_alias",
        "manifest_step_count",
    ),
)
def test_source_command_replay_sequence_fails_closed(
    tmp_path: Path, surface: str
) -> None:
    out = _build_snapshot_set(tmp_path)
    payload = json.loads(
        (out / "P10" / "snapshot.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest_row = next(
        row for row in manifest["snapshots"] if row["phase"] == "P10"
    )
    changed = copy.deepcopy(payload)
    changed_manifest = copy.deepcopy(manifest_row)

    if surface == "step_count":
        changed["source_replay_steps"] -= 1
    elif surface == "missing_row":
        changed["source_commands"].pop()
    elif surface == "discontinuous_tick":
        changed["source_commands"][4]["control_physics_tick"] += 1
    elif surface == "tail_lifecycle":
        changed["source_commands"][4]["source_fsm_lifecycle"] = "VERIFY_RESULT"
    elif surface == "controller_row_context":
        changed["source_commands"][8]["source_fsm_state"] = "P09"
        changed["source_commands"][8]["source_fsm_lifecycle"] = "VERIFY_RESULT"
    elif surface == "target_command_included":
        changed["source_commands"][-1]["control_physics_tick"] = changed[
            "target_entry_tick"
        ]
    elif surface == "tail_atomic_tick":
        changed["source_commands"][4]["source_atomic_physics_tick"] += 1
    elif surface == "tail_mapper_continuity":
        changed["source_commands"][4]["mapper_pre_state"][
            "requested_servo_deg"
        ][0] += 1.0
    elif surface == "tail_actuation_hash":
        changed["source_commands"][4]["actuation_contract_sha256"] = "0" * 64
    elif surface == "tail_source_row_hash":
        changed["source_commands"][4][
            "source_command_row_canonical_sha256"
        ] = "invalid"
    elif surface == "first_alias":
        changed["source_command"]["control_physics_tick"] += 1
    else:
        changed_manifest["source_replay_steps"] -= 1

    with pytest.raises(PhaseSnapshotError):
        validate_phase_snapshot_payload_contract(
            changed,
            "P10",
            manifest_row=changed_manifest,
            manifest_source_artifacts=manifest["source_artifacts"],
            causal_predecessor_required=True,
        )


def test_generated_bundle_bytes_are_git_lf_normalization_stable(tmp_path):
    out = _build_snapshot_set(tmp_path)
    paths = [out / "manifest.json"]
    for phase in PHASE_IDS:
        paths.extend(
            (out / phase / "snapshot.json", out / phase / "snapshot.sha256")
        )

    assert len(paths) == 27
    for path in paths:
        payload = path.read_bytes()
        simulated_git_lf_bytes = payload.replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )
        assert b"\r" not in payload, path
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n"), path
        assert simulated_git_lf_bytes == payload, path
        assert hashlib.sha256(simulated_git_lf_bytes).digest() == hashlib.sha256(
            payload
        ).digest(), path


def test_builder_opens_each_source_artifact_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial = _write_source_trial(tmp_path)
    source_paths = {
        trial / name
        for name in (
            "trial_manifest.json",
            "full12_commands_120hz.jsonl",
            "observation_120hz.jsonl",
            "state_transitions.jsonl",
            "leg_crossing_events.jsonl",
        )
    }
    open_counts: Counter[Path] = Counter()
    original_open = Path.open

    def counted_open(path: Path, *args, **kwargs):
        if path in source_paths:
            open_counts[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    build_phase_snapshots(trial, tmp_path / "snapshots")

    assert open_counts == Counter({path: 1 for path in source_paths})


@pytest.mark.parametrize(
    "target_kind",
    (
        "ancestor",
        "root",
        "trial_manifest",
        "command",
        "observation",
        "transition",
        "leg_crossing",
    ),
)
def test_builder_rejects_every_source_reparse_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    trial = _write_source_trial(tmp_path).resolve()
    targets = {
        "ancestor": trial.parent,
        "root": trial,
        "trial_manifest": trial / "trial_manifest.json",
        "command": trial / "full12_commands_120hz.jsonl",
        "observation": trial / "observation_120hz.jsonl",
        "transition": trial / "state_transitions.jsonl",
        "leg_crossing": trial / "leg_crossing_events.jsonl",
    }
    rejected = targets[target_kind]
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: True if self == rejected else original_is_symlink(self),
    )

    with pytest.raises(PhaseSnapshotError, match="symlink or reparse"):
        build_phase_snapshots(trial, tmp_path / "snapshots")
    assert not (tmp_path / "snapshots").exists()


def test_builder_rejects_source_symlink_when_supported(tmp_path: Path) -> None:
    trial = _write_source_trial(tmp_path)
    transition = trial / "state_transitions.jsonl"
    external = tmp_path / "external_transitions.jsonl"
    external.write_bytes(transition.read_bytes())
    transition.unlink()
    try:
        transition.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(PhaseSnapshotError, match="symlink|reparse|redirect"):
        build_phase_snapshots(trial, tmp_path / "snapshots")
    assert not (tmp_path / "snapshots").exists()


def test_builder_detects_mutation_between_source_capture_and_parse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial = _write_source_trial(tmp_path)
    transition = trial / "state_transitions.jsonl"
    original_capture = phase_snapshots_subject._capture_source_bytes_once
    mutated = False

    def capture_then_mutate(path: Path, **kwargs):
        nonlocal mutated
        payload = original_capture(path, **kwargs)
        if path == transition and not mutated:
            transition.write_bytes(payload + b"\n")
            mutated = True
        return payload

    monkeypatch.setattr(
        phase_snapshots_subject,
        "_capture_source_bytes_once",
        capture_then_mutate,
    )

    with pytest.raises(PhaseSnapshotError, match="changed during immutable capture"):
        build_phase_snapshots(trial, tmp_path / "snapshots")
    assert mutated is True
    assert not (tmp_path / "snapshots").exists()


def test_validated_bundle_record_binds_manifest_and_exact_p01_p13_bytes(tmp_path):
    out = _build_snapshot_set(tmp_path)

    record = validated_phase_snapshot_bundle_record(out, canonical_root=out)

    assert record["snapshot_root"] == str(out.resolve())
    assert record["manifest_path"] == str((out / "manifest.json").resolve())
    assert record["manifest_sha256"] == hashlib.sha256(
        (out / "manifest.json").read_bytes()
    ).hexdigest()
    assert record["phase_count"] == 13
    assert len(record["snapshots"]) == 13
    assert tuple(row["phase"] for row in record["snapshots"]) == PHASE_IDS
    assert all(
        set(row)
        == {
            "phase",
            "source_tick",
            "source_replay_steps",
            "target_entry_tick",
            "predecessor_verify_tick",
            "predecessor_verify_time_s",
            "controller_anchor_tick",
            "controller_anchor_time_s",
            "snapshot_path",
            "checksum_path",
            "file_sha256",
            "state_sha256",
            "checksum_file_sha256",
        }
        for row in record["snapshots"]
    )
    p08_record = next(row for row in record["snapshots"] if row["phase"] == "P08")
    assert p08_record["source_replay_steps"] == 1
    assert p08_record["target_entry_tick"] is None
    assert p08_record["predecessor_verify_tick"] is None
    assert p08_record["predecessor_verify_time_s"] is None
    assert p08_record["controller_anchor_tick"] is None
    assert p08_record["controller_anchor_time_s"] is None
    p10_record = next(row for row in record["snapshots"] if row["phase"] == "P10")
    assert p10_record["source_tick"] == 9
    assert p10_record["source_replay_steps"] == 18
    assert p10_record["target_entry_tick"] == 27
    assert p10_record["predecessor_verify_tick"] == 15
    assert p10_record["predecessor_verify_time_s"] == 15 / 120
    assert p10_record["controller_anchor_tick"] == 17
    assert p10_record["controller_anchor_time_s"] == 17 / 120
    assert len(record["bundle_sha256"]) == 64
    assert validated_phase_snapshot_bundle_record(out, canonical_root=out) == record


def test_zero_length_physical_replay_segment_fails_closed_everywhere(tmp_path):
    out = _build_snapshot_set(tmp_path)
    bundle_record = validated_phase_snapshot_bundle_record(out, canonical_root=out)
    changed_record = copy.deepcopy(bundle_record)
    record_row = next(
        row for row in changed_record["snapshots"] if row["phase"] == "P10"
    )
    record_row["predecessor_verify_tick"] = record_row["source_tick"]
    record_row["predecessor_verify_time_s"] = record_row["source_tick"] / 120.0
    with pytest.raises(PhaseSnapshotError, match="predecessor verify tick"):
        phase_snapshot_bundle_file_hashes(changed_record)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_row = next(
        row for row in manifest["snapshots"] if row["phase"] == "P10"
    )
    manifest_row["predecessor_verify_tick"] = manifest_row["source_tick"]
    manifest_row["predecessor_verify_time_s"] = manifest_row["source_tick"] / 120.0
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(PhaseSnapshotError, match="predecessor verify tick"):
        capture_validated_phase_snapshot_bundle(out, canonical_root=out)


def test_runtime_bundle_validation_rejects_noncanonical_external_root(tmp_path):
    out = _build_snapshot_set(tmp_path)

    with pytest.raises(PhaseSnapshotError, match="differs from canonical"):
        validated_phase_snapshot_bundle_record(out)


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    (
        ("manifest.json", b"{}\n"),
        ("P04/snapshot.json", b"{}\n"),
        ("P04/snapshot.sha256", b"0" * 64 + b"  snapshot.json\n"),
    ),
)
def test_bundle_record_rejects_changed_required_bytes(
    tmp_path, relative_path, replacement
):
    out = _build_snapshot_set(tmp_path)
    (out / relative_path).write_bytes(replacement)

    with pytest.raises(PhaseSnapshotError):
        validated_phase_snapshot_bundle_record(out, canonical_root=out)


@pytest.mark.parametrize(
    "relative_path",
    ("manifest.json", "P07/snapshot.json", "P07/snapshot.sha256"),
)
def test_bundle_record_rejects_missing_required_bytes(tmp_path, relative_path):
    out = _build_snapshot_set(tmp_path)
    (out / relative_path).unlink()

    with pytest.raises(PhaseSnapshotError, match="missing"):
        validated_phase_snapshot_bundle_record(out, canonical_root=out)


def test_bundle_record_rejects_manifest_redirect_even_to_existing_snapshot(tmp_path):
    out = _build_snapshot_set(tmp_path)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshots"][0]["path"] = str((out / "P02" / "snapshot.json").resolve())
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(PhaseSnapshotError, match="does not resolve"):
        validated_phase_snapshot_bundle_record(out, canonical_root=out)


def test_pinned_loader_parses_and_hashes_only_immutable_captured_bytes(
    monkeypatch, tmp_path
):
    out = _build_snapshot_set(tmp_path)
    bundle = capture_validated_phase_snapshot_bundle(out, canonical_root=out)

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: pytest.fail(f"loader reread filesystem path {self}"),
    )
    payload, entry = load_validated_phase_snapshot_payload(bundle, "P08")

    assert payload["fsm_state"] == "P08"
    assert payload["state_sha256"] == entry.state_sha256


def test_pinned_loader_exposes_exact_hybrid_controller_anchor(tmp_path):
    out = _build_snapshot_set(tmp_path)
    bundle = capture_validated_phase_snapshot_bundle(out, canonical_root=out)

    payload, entry = load_validated_phase_snapshot_payload(bundle, "P10")

    assert payload["source_tick"] == 9
    assert entry.source_tick == 9
    assert payload["predecessor_verify_tick"] == 15
    assert entry.predecessor_verify_tick == 15
    assert payload["predecessor_verify_time_s"] == 15 / 120
    assert entry.predecessor_verify_time_s == 15 / 120
    assert payload["controller_anchor_tick"] == 17
    assert entry.controller_anchor_tick == 17
    assert payload["controller_anchor_time_s"] == 17 / 120
    assert entry.controller_anchor_time_s == 17 / 120
    assert payload["target_entry_tick"] == 27
    assert entry.target_entry_tick == 27


@pytest.mark.parametrize(
    "target_name",
    ("root", "manifest", "phase_dir", "snapshot", "sidecar"),
)
def test_bundle_capture_rejects_every_symlink_or_reparse_surface(
    monkeypatch, tmp_path, target_name
):
    out = _build_snapshot_set(tmp_path).resolve()
    targets = {
        "root": out,
        "manifest": out / "manifest.json",
        "phase_dir": out / "P03",
        "snapshot": out / "P03" / "snapshot.json",
        "sidecar": out / "P03" / "snapshot.sha256",
    }
    rejected = targets[target_name]
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: True if self == rejected else original(self),
    )

    with pytest.raises(PhaseSnapshotError, match="symlink or reparse"):
        capture_validated_phase_snapshot_bundle(out, canonical_root=out)


def test_bundle_capture_rejects_external_snapshot_symlink_when_supported(tmp_path):
    out = _build_snapshot_set(tmp_path)
    snapshot = out / "P04" / "snapshot.json"
    external = tmp_path / "external_snapshot.json"
    external.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    try:
        snapshot.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(PhaseSnapshotError, match="symlink|redirect"):
        capture_validated_phase_snapshot_bundle(out, canonical_root=out)


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


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_drive_target",
        "changed_drive_target",
        "missing_retiring_state",
        "changed_source_file_sha",
        "changed_post_tracking_flag",
    ),
)
def test_v2_source_replay_contract_fails_closed_on_every_critical_surface(
    tmp_path: Path, mutation: str
) -> None:
    out = _build_snapshot_set(tmp_path)
    payload = json.loads((out / "P03" / "snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest_row = next(row for row in manifest["snapshots"] if row["phase"] == "P03")
    changed = copy.deepcopy(payload)
    source = changed["source_command"]
    if mutation == "missing_drive_target":
        source["expected_atomic_ack"].pop("drive_target_full12")
    elif mutation == "changed_drive_target":
        source["expected_atomic_ack"]["drive_target_full12"][0] = 1.0
    elif mutation == "missing_retiring_state":
        source["mapper_pre_state"].pop("retiring_stale_bias")
    elif mutation == "changed_source_file_sha":
        changed["source_artifacts"]["command"]["sha256"] = "0" * 64
    else:
        source["mapper_post_state"]["tracking_active"][0] = True
    changed["source_commands"][0] = copy.deepcopy(source)

    with pytest.raises(PhaseSnapshotError):
        validate_phase_snapshot_payload_contract(
            changed,
            "P03",
            manifest_row=manifest_row,
            manifest_source_artifacts=manifest["source_artifacts"],
        )


def test_builder_rejects_semantically_unchanged_but_hash_changed_transition_source(
    tmp_path: Path,
) -> None:
    first = _build_snapshot_set(tmp_path)
    trial = first.parent / "trial"
    transition = trial / "state_transitions.jsonl"
    transition.write_bytes(transition.read_bytes() + b"\n")

    with pytest.raises(PhaseSnapshotError, match="transition hash/size"):
        build_phase_snapshots(trial, tmp_path / "second-snapshots")
