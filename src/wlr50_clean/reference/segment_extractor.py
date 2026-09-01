"""Build the compact P01-P13 v010 motion contract offline.

The successful 120 Hz trace is used only while generating the immutable
contract. Production code consumes the compact output and never opens the raw
recording or telemetry streams.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .recording_parser import (
FULL12_ORDER,
    SERVO_ORDER,
    WHEEL_ORDER,
    Full12Command,
    ParsedRecording,
    load_recording,
    sha256_file,
)

REFERENCE_VERSION_ID = "v010_20260806_220745_363972_manual"


LEG_TO_WHEEL = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}

WHEEL_RADIUS_M = 0.04998999834060672

NEXT_STATE_REASONS = {
    "P01": "Reference-compatible load-transfer endpoint and live FR lift-entry geometry are established.",
    "P02": "FR reference-like active joint motion and increased wheel-bottom clearance make front-plane crossing observable.",
    "P03": "Latched FR front-face clearance and top load make FL preparation safe.",
    "P04": "The FL unload workspace matches the v010 entry envelope without a force-only hard gate.",
    "P05": "FL has actively cleared the front plane and reached its placement posture; P06 continues wheel motion immediately, without a hold, to latch top load.",
    "P06": "FL top load is latched and the rear pair has reached the measured pre-edge geometry.",
    "P07": "The rear-entry pose and wheel relation match the v010 first-rear entry envelope.",
    "P08": "The v010 FR+RL support geometry and active RR-unload preparation are present.",
    "P09": "Latched RR active lift, front-face crossing, and top load prove RR-first traversal.",
    "P10": "The FL+RR-related workspace is established for the second rear leg.",
    "P11": "The FR-directed transfer action has reached the v010 RL-unload entry pose.",
    "P12": "Latched RL active lift, front-face crossing, and top load complete the rear order.",
    "P13": "All four wheels and the body establish the v010 final top-crossing geometry and final pose; zero wheel targets are applied, then the new FSM must add a measured stable-decay debounce before SUCCESS.",
}

LIVE_COMPLETION_PREDICATES = {
    "P01": ["reference-compatible joint endpoint", "FR lift-entry wheel/obstacle geometry", "no BODY contact"],
    "P02": ["FR leg-joint evidence matches the contract: knee actively changes while hip remains at its reference hold", "FR wheel-bottom clearance increase or latched AIR", "motion endpoint reached"],
    "P03": ["latched FR FRONT_FACE_CLEARED", "latched FR TOP_LOADED", "no BODY contact"],
    "P04": ["reference-compatible FL unload entry pose", "FL lift workspace/clearance trend", "no BODY contact"],
    "P05": ["FL hip/knee reference-like active delta", "latched FL FRONT_FACE_CLEARED", "placement command endpoint reached"],
    "P06": ["latched FL TOP_LOADED", "rear-pair front-plane distance matches reference pre-edge envelope", "continuous wheel command relation preserved"],
    "P07": ["joint endpoint within reference entry envelope", "rear wheel geometry aligned to v010 entry", "no BODY contact"],
    "P08": ["v010 RL-hip support/load-transfer endpoint", "RR wheel geometry and AIR/contact history compatible with subsequent active lift", "support quantities diagnostic only"],
    "P09": ["RR hip/knee reference-like active delta", "latched RR FRONT_FACE_CLEARED", "latched RR TOP_LOADED"],
    "P10": ["RR-knee command endpoint issued", "RL workspace geometry compatible", "carry-out response may remain active"],
    "P11": ["FR-directed active joint delta", "RL unload-entry geometry compatible", "carry-in/out response does not require settle"],
    "P12": ["RL hip/knee reference-like active delta", "latched RL FRONT_FACE_CLEARED", "latched RL TOP_LOADED"],
    "P13": ["all four front-face crossings latched", "all four wheels in final top geometry", "final command pose compatible", "wheel targets zero", "new-run measured wheel-speed decay stable over debounce"],
}


@dataclass(frozen=True)
class PhaseDefinition:
    state_id: str
    macro_phase: int
    name: str
    physical_purpose: str
    first_step: int
    last_step: int
    completion_event: str
    ppo_action_mask: tuple[int, ...]


PHASES = (
    PhaseDefinition("P01", 1, "INITIAL_LOAD_TRANSFER_FR", "Build the v010 diagonal/load structure and approach needed before active FR lift.", 1, 2, "fr_lift_entry_ready", (1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1)),
    PhaseDefinition("P02", 2, "FR_ACTIVE_LIFT", "Actively flex the FR leg until its wheel-bottom clearance increases.", 3, 3, "fr_active_lift_observed", (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)),
    PhaseDefinition("P03", 3, "FR_CROSS_AND_PLACE", "Carry FR through the front plane and establish FR obstacle-top contact.", 4, 4, "fr_top_loaded", (1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1)),
    PhaseDefinition("P04", 4, "FL_UNLOAD_PREP", "Change the v010 support geometry so FL can be actively lifted.", 5, 6, "fl_lift_entry_ready", (0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1)),
    PhaseDefinition("P05", 5, "FL_ACTIVE_LIFT_AND_PLACE", "Actively lift FL, clear the front plane, and commit the placement posture; top load latches during the immediately continuous P06 wheel advance.", 7, 9, "fl_front_cleared_and_placement_committed", (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)),
    PhaseDefinition("P06", 6, "FRONT_PAIR_ADVANCE", "Continue the v010 wheel relation without a hold, latch FL obstacle-top load, and advance until the rear pair nears the edge.", 10, 10, "fl_top_loaded_and_rear_pair_pre_edge", (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)),
    PhaseDefinition("P07", 7, "REAR_ENTRY_ALIGNMENT", "Align the live front/rear geometry to the v010 first-rear entry without leaving the 15 percent envelope.", 11, 13, "rear_entry_aligned", (1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0)),
    PhaseDefinition("P08", 8, "FIRST_REAR_LOAD_TRANSFER", "Load FL and prepare the FR+RL support structure that unloads RR first.", 14, 14, "rr_unload_ready", (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)),
    PhaseDefinition("P09", 9, "RR_ACTIVE_LIFT_AND_PLACE", "Actively lift RR, clear the front plane, and finish the v010 placement action that establishes RR top load.", 15, 18, "rr_top_loaded", (1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)),
    PhaseDefinition("P10", 10, "SECOND_REAR_TRANSFER_PREP", "Establish the v010 FL+RR-related support/workspace for RL unloading.", 19, 19, "rl_workspace_ready", (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)),
    PhaseDefinition("P11", 11, "TRANSFER_TOWARD_FR", "Execute the v010 FR-directed load-transfer action before RL lift.", 20, 20, "rl_unload_ready", (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    PhaseDefinition("P12", 12, "RL_ACTIVE_LIFT_AND_PLACE", "Actively lift RL, clear the front plane, and place RL on the obstacle top.", 21, 24, "rl_top_loaded", (0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1)),
    PhaseDefinition("P13", 13, "FINAL_ADVANCE_AND_RECOVERY", "Advance the whole body past the obstacle, recover the v010 final posture, and stop all wheels.", 25, 26, "final_clear_and_stopped", (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
)


def _parse_json_cell(value: str) -> dict[str, float]:
    if not value:
        return {}
    parsed = json.loads(value)
    return {str(key): float(item) for key, item in parsed.items()}


def load_dispatch_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("actual_dispatch_time_s"):
                continue
            if row.get("source_version") != REFERENCE_VERSION_ID:
                raise ValueError("dispatch trace contains non-v010 source motion")
            rows.append(
                {
                    "source_version": str(row["source_version"]),
                    "source_step": int(row["source_step_index"]),
                    "source_event": int(row["source_event_index"]),
                    "source_expanded_index": int(row.get("source_expanded_index") or 0),
                    "source_command": row.get("source_command", ""),
                    "actual_time_s": float(row["actual_dispatch_time_s"]),
                    "planned_time_s": float(row["planned_dispatch_time_s"]),
                    "servo_targets_deg": _parse_json_cell(row.get("servo_target_deg", "")),
                    "wheel_targets_rad_s": _parse_json_cell(row.get("wheel_target_rad_s", "")),
                    "atomic_batch_id": row.get("atomic_batch_id", ""),
                    "compiled_segment": int(row.get("compiled_segment_index") or -1),
                    "atomic_physics_step": int(row.get("atomic_batch_applied_sim_step") or -1),
                    "atomic_first_physics_step": int(
                        row.get("atomic_batch_first_physics_step") or -1
                    ),
                    "motion_start_skew_s": float(row.get("atomic_batch_motion_start_skew_s") or 0.0),
                    "atomic_ack_valid": str(
                        row.get("atomic_batch_ack_valid", "")
                    ).strip().lower()
                    == "true",
                    "atomic_ack_error": str(row.get("atomic_batch_ack_error", "")),
                }
            )
    if not rows:
        raise ValueError("dispatch trace contains no applied rows")
    return rows


def load_leg_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_by_leg: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("schema_version") != "wlr50.recording.compact_leg_event_ledger.v1":
                raise ValueError("unexpected leg event ledger schema")
            leg = str(row["leg"])
            event = str(row["event"])
            time_s = float(row["sim_time_s"])
            if time_s + 1e-9 < previous_by_leg.get(leg, -math.inf):
                raise ValueError(f"leg event time moved backwards for {leg}")
            previous_by_leg[leg] = time_s
            rows.append(
                {
                    "leg": leg,
                    "event": event,
                    "sim_time_s": time_s,
                    "physics_tick": int(row["physics_tick"]),
                    "contact_class": str(row["contact_class"]),
                    "normal_force_n": float(row["normal_force_n"]),
                    "front_face_center_clearance_m": float(
                        row["front_face_center_clearance_m"]
                    ),
                    "top_surface_wheel_bottom_clearance_m": float(
                        row["top_surface_wheel_bottom_clearance_m"]
                    ),
                    "wheel_center_w_m": json.loads(row["wheel_center_w_m"]),
                }
            )
    if not rows:
        raise ValueError("leg event ledger is empty")
    return rows


def _leg_event(
    rows: Sequence[Mapping[str, Any]], leg: str, event: str
) -> dict[str, Any]:
    matches = [row for row in rows if row["leg"] == leg and row["event"] == event]
    if len(matches) != 1:
        raise ValueError(f"expected one {leg} {event} event, found {len(matches)}")
    return dict(matches[0])


def _completion_evidence(
    state_id: str,
    phase_start_s: float,
    phase_end_s: float,
    leg_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required: dict[str, tuple[tuple[str, str], ...]] = {
        "P03": (("FR", "FRONT_FACE_CLEARED"), ("FR", "TOP_LOADED")),
        "P05": (("FL", "FRONT_FACE_CLEARED"),),
        "P06": (("FL", "TOP_LOADED"),),
        "P09": (("RR", "FRONT_FACE_CLEARED"), ("RR", "TOP_LOADED")),
        "P12": (("RL", "FRONT_FACE_CLEARED"), ("RL", "TOP_LOADED")),
    }
    events: list[dict[str, Any]] = []
    for leg, event_name in required.get(state_id, ()):
        row = _leg_event(leg_events, leg, event_name)
        row["relative_to_phase_start_s"] = float(row["sim_time_s"]) - phase_start_s
        row["relative_to_motion_end_s"] = float(row["sim_time_s"]) - phase_end_s
        row["latched"] = True
        events.append(row)
    positive_latencies = [
        max(0.0, float(row["relative_to_motion_end_s"])) for row in events
    ]
    return {
        "required_latched_leg_events": events,
        "sensor_completion_latency_after_motion_s": max(
            positive_latencies, default=0.0
        ),
        "uses_event_history_not_final_contact_snapshot": bool(events),
        "live_sensor_predicates": LIVE_COMPLETION_PREDICATES[state_id],
        "next_state_safe_reason": NEXT_STATE_REASONS[state_id],
    }


def _command_dict(command: Full12Command) -> dict[str, float]:
    return {**command.servos_deg, **command.wheels_rad_s}


def _vector(command: Mapping[str, float]) -> list[float]:
    return [float(command[name]) for name in FULL12_ORDER]


def _group_dispatch_rows(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: (item["actual_time_s"], item["compiled_segment"], item["source_expanded_index"])):
        key = (row["actual_time_s"], row["compiled_segment"])
        if groups and (groups[-1][0]["actual_time_s"], groups[-1][0]["compiled_segment"]) == key:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _waypoints_for_phase(
    definition: PhaseDefinition,
    recording: ParsedRecording,
    dispatch_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, float, dict[str, float], dict[str, float]]:
    selected = [
        row
        for row in dispatch_rows
        if definition.first_step <= row["source_step"] <= definition.last_step
    ]
    if not selected:
        raise ValueError(f"{definition.state_id}: no applied dispatch rows")
    phase_start = min(row["actual_time_s"] for row in selected)
    phase_end = max(row["actual_time_s"] for row in selected)
    start = _command_dict(recording.steps[definition.first_step - 1].before)
    expected_end = _command_dict(recording.steps[definition.last_step - 1].after)
    current = dict(start)
    waypoints: list[dict[str, Any]] = [
        {
            "time_s": 0.0,
            "full12": _vector(current),
            "changed_channels": [],
            "atomic_channels": [],
            "source_events": [],
            "kind": "phase_entry",
        }
    ]
    for group in _group_dispatch_rows(selected):
        changed: list[str] = []
        atomic_channels: list[str] = []
        source_events: list[dict[str, int]] = []
        commands: list[str] = []
        for row in group:
            source_events.append({"step": row["source_step"], "event": row["source_event"]})
            commands.append(row["source_command"])
            for name, value in row["servo_targets_deg"].items():
                atomic_channels.append(name)
                if not math.isclose(current[name], value, abs_tol=1e-12):
                    current[name] = value
                    changed.append(name)
            for name, value in row["wheel_targets_rad_s"].items():
                atomic_channels.append(name)
                if not math.isclose(current[name], value, abs_tol=1e-12):
                    current[name] = value
                    changed.append(name)
        applied_steps = {int(row["atomic_physics_step"]) for row in group}
        first_steps = {int(row["atomic_first_physics_step"]) for row in group}
        batch_ids = {str(row["atomic_batch_id"]) for row in group}
        ack_valid = all(bool(row["atomic_ack_valid"]) for row in group)
        same_physics_tick = (
            ack_valid
            and len(applied_steps) == 1
            and len(first_steps) == 1
            and len(batch_ids) == 1
            and min(applied_steps) >= 0
            and min(first_steps) == min(applied_steps) + 1
        )
        if len(set(atomic_channels)) >= 2 and not same_physics_tick:
            errors = sorted(
                {
                    str(row["atomic_ack_error"])
                    for row in group
                    if row["atomic_ack_error"]
                }
            )
            raise ValueError(
                f"{definition.state_id}: invalid atomic batch acknowledgement: {errors}"
            )
        waypoints.append(
            {
                "time_s": float(group[0]["actual_time_s"] - phase_start),
                "full12": _vector(current),
                "changed_channels": sorted(set(changed), key=FULL12_ORDER.index),
                "atomic_channels": sorted(set(atomic_channels), key=FULL12_ORDER.index),
                "source_events": source_events,
                "source_commands": commands,
                "atomic_batch_id": group[0]["atomic_batch_id"],
                "reference_physics_step": group[0]["atomic_physics_step"],
                "reference_first_motion_physics_step": group[0][
                    "atomic_first_physics_step"
                ],
                "atomic_batch_ack_valid": ack_valid,
                "same_physics_tick": same_physics_tick,
                "motion_start_skew_s": max(abs(float(row["motion_start_skew_s"])) for row in group),
                "kind": "reference_waypoint",
            }
        )
    for name in FULL12_ORDER:
        if not math.isclose(current[name], expected_end[name], abs_tol=1e-6):
            raise ValueError(
                f"{definition.state_id}: dispatch endpoint for {name}={current[name]} differs from accepted-step endpoint {expected_end[name]}"
            )
    return waypoints, phase_start, phase_end, start, expected_end


def _command_metrics(waypoints: Sequence[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    path_length = {name: 0.0 for name in FULL12_ORDER}
    waypoint_delta_rate = {name: 0.0 for name in SERVO_ORDER}
    wheel_integral = {name: 0.0 for name in WHEEL_ORDER}
    wheel_abs_integral = {name: 0.0 for name in WHEEL_ORDER}
    for left, right in zip(waypoints, waypoints[1:]):
        dt = max(0.0, float(right["time_s"]) - float(left["time_s"]))
        for index, name in enumerate(FULL12_ORDER):
            delta = float(right["full12"][index]) - float(left["full12"][index])
            path_length[name] += abs(delta)
            if dt > 1e-9 and name in SERVO_ORDER:
                waypoint_delta_rate[name] = max(
                    waypoint_delta_rate[name], abs(delta) / dt
                )
        for name in WHEEL_ORDER:
            index = FULL12_ORDER.index(name)
            wheel_integral[name] += float(left["full12"][index]) * dt
            wheel_abs_integral[name] += abs(float(left["full12"][index])) * dt
    return {
        "servo_target_path_length_deg": {
            name: path_length[name] for name in SERVO_ORDER
        },
        "servo_average_target_path_velocity_deg_s": {
            name: path_length[name] / max(duration_s, 1e-9)
            for name in SERVO_ORDER
        },
        "servo_velocity_limit_deg_s": {
            name: (150.0 if path_length[name] > 1e-12 else 0.0)
            for name in SERVO_ORDER
        },
        "waypoint_delta_rate_diagnostic_deg_s": waypoint_delta_rate,
        "waypoint_delta_rate_is_not_actuator_velocity": True,
        "wheel_time_weighted_average_abs_target_rad_s": {
            name: wheel_abs_integral[name] / max(duration_s, 1e-9)
            for name in WHEEL_ORDER
        },
        "wheel_peak_abs_target_rad_s": {
            name: max(
                abs(float(waypoint["full12"][FULL12_ORDER.index(name)]))
                for waypoint in waypoints
            )
            for name in WHEEL_ORDER
        },
        "wheel_integral_rad": wheel_integral,
        "estimated_wheel_surface_travel_m": {
            name: wheel_integral[name] * WHEEL_RADIUS_M for name in WHEEL_ORDER
        },
    }


def _telemetry_rows(path: Path) -> Iterable[dict[str, Any]]:
    previous_tick: int | None = None
    previous_time: float | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if row.get("source_version") != REFERENCE_VERSION_ID:
                    raise ValueError(
                        f"telemetry line {line_number}: non-v010 source version"
                    )
                tick = int(row["physics_tick"])
                time_s = float(row["sim_time_s"])
                physics_dt_s = float(row["physics_dt_s"])
                if not math.isclose(
                    physics_dt_s, 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"telemetry line {line_number}: physics dt is not 1/120"
                    )
                if previous_tick is not None:
                    if tick != previous_tick + 1:
                        raise ValueError(
                            f"telemetry line {line_number}: missing or duplicate physics tick"
                        )
                    observed_dt = time_s - float(previous_time)
                    if not math.isclose(
                        observed_dt,
                        physics_dt_s,
                        rel_tol=0.0,
                        abs_tol=2e-9,
                    ):
                        raise ValueError(
                            f"telemetry line {line_number}: non-120 Hz timestamp cadence"
                        )
                previous_tick = tick
                previous_time = time_s
                yield row


def _sample_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    full12_actual: list[float] = []
    full12_target: list[float] = []
    full12_velocity: list[float] = []
    for name in SERVO_ORDER:
        data = row["servos"][name]
        full12_actual.append(float(data["actual_command_space_deg"]))
        full12_target.append(float(data["target_command_space_deg"]))
        full12_velocity.append(float(data["velocity_command_space_deg_s"]))
    for leg in ("FL", "FR", "RL", "RR"):
        data = row["wheels"][leg]
        full12_actual.append(float(data["angular_velocity_rad_s"]))
        full12_target.append(float(data["target_velocity_rad_s"]))
        full12_velocity.append(float(data["angular_velocity_rad_s"]))
    return {
        "time_s": float(row["sim_time_s"]),
        "physics_tick": int(row["physics_tick"]),
        "control_tick": int(row["control_tick"]),
        "actual_full12": full12_actual,
        "target_full12": full12_target,
        "velocity_full12": full12_velocity,
        "root_position_w_m": list(row["root_and_imu"]["root_position_w_m"]),
        "root_quaternion_wxyz": list(row["root_and_imu"]["quaternion_wxyz"]),
        "wheel_geometry": {
            leg: {
                "center_w_m": list(row["wheels"][leg]["center_w_m"]),
                "front_face_center_clearance_m": float(row["wheels"][leg]["front_face_center_clearance_m"]),
                "top_surface_wheel_bottom_clearance_m": float(row["wheels"][leg]["top_surface_wheel_bottom_clearance_m"]),
                "contact_class": str(row["wheels"][leg]["contact"]["class"]),
                "normal_force_n": float(row["wheels"][leg]["contact"]["normal_force_n"]),
            }
            for leg in ("FL", "FR", "RL", "RR")
        },
    }


def _phase_telemetry(
    telemetry_path: Path,
    boundaries: Mapping[str, tuple[float, float]],
) -> dict[str, list[dict[str, Any]]]:
    result = {state_id: [] for state_id in boundaries}
    for row in _telemetry_rows(telemetry_path):
        time_s = float(row["sim_time_s"])
        for state_id, (start, end) in boundaries.items():
            if start - 1e-9 <= time_s <= end + 1e-9:
                result[state_id].append(_sample_payload(row))
                break
    for state_id, samples in result.items():
        if len(samples) < 2:
            raise ValueError(f"{state_id}: insufficient 120 Hz telemetry")
    return result


def _telemetry_cadence(path: Path) -> dict[str, Any]:
    count = 0
    first_tick: int | None = None
    last_tick: int | None = None
    first_time: float | None = None
    last_time: float | None = None
    min_dt = math.inf
    max_dt = -math.inf
    previous_time: float | None = None
    for row in _telemetry_rows(path):
        tick = int(row["physics_tick"])
        time_s = float(row["sim_time_s"])
        if first_tick is None:
            first_tick = tick
            first_time = time_s
        if previous_time is not None:
            dt = time_s - previous_time
            min_dt = min(min_dt, dt)
            max_dt = max(max_dt, dt)
        previous_time = time_s
        last_tick = tick
        last_time = time_s
        count += 1
    if count < 2:
        raise ValueError("telemetry stream contains fewer than two samples")
    return {
        "validated_continuous_120hz": True,
        "sample_count": count,
        "first_physics_tick": first_tick,
        "last_physics_tick": last_tick,
        "first_sim_time_s": first_time,
        "last_sim_time_s": last_time,
        "minimum_dt_s": min_dt,
        "maximum_dt_s": max_dt,
        "missing_or_duplicate_ticks": 0,
    }


def _normalized_samples(samples: Sequence[dict[str, Any]], count: int = 21) -> list[dict[str, Any]]:
    start = float(samples[0]["time_s"])
    end = float(samples[-1]["time_s"])
    result: list[dict[str, Any]] = []
    cursor = 0
    for index in range(count):
        progress = index / max(1, count - 1)
        target_time = start + progress * (end - start)
        while cursor + 1 < len(samples) and abs(float(samples[cursor + 1]["time_s"]) - target_time) <= abs(float(samples[cursor]["time_s"]) - target_time):
            cursor += 1
        row = samples[cursor]
        result.append(
            {
                "progress": progress,
                "actual_full12": row["actual_full12"],
                "target_full12": row["target_full12"],
            }
        )
    return result


def _actual_metrics(
    samples: Sequence[dict[str, Any]], active_channels: Sequence[str]
) -> dict[str, Any]:
    duration = float(samples[-1]["time_s"]) - float(samples[0]["time_s"])
    sample_dts = [
        float(right["time_s"]) - float(left["time_s"])
        for left, right in zip(samples, samples[1:])
    ]
    tick_deltas = [
        int(right["physics_tick"]) - int(left["physics_tick"])
        for left, right in zip(samples, samples[1:])
    ]
    if any(delta != 1 for delta in tick_deltas):
        raise ValueError("phase telemetry contains a missing or duplicate tick")
    average = {name: 0.0 for name in FULL12_ORDER}
    peak = {name: 0.0 for name in FULL12_ORDER}
    integrals = {name: 0.0 for name in WHEEL_ORDER}
    for name_index, name in enumerate(FULL12_ORDER):
        values = [abs(float(row["velocity_full12"][name_index])) for row in samples]
        average[name] = sum(values) / len(values)
        peak[name] = max(values)
    carry_in = [
        name
        for index, name in enumerate(FULL12_ORDER)
        if abs(float(samples[0]["velocity_full12"][index]))
        >= (1.0 if name in SERVO_ORDER else 0.05)
    ]
    carry_out = [
        name
        for index, name in enumerate(FULL12_ORDER)
        if abs(float(samples[-1]["velocity_full12"][index]))
        >= (1.0 if name in SERVO_ORDER else 0.05)
    ]
    for left, right in zip(samples, samples[1:]):
        dt = float(right["time_s"]) - float(left["time_s"])
        for name in WHEEL_ORDER:
            index = FULL12_ORDER.index(name)
            integrals[name] += 0.5 * (
                float(left["actual_full12"][index])
                + float(right["actual_full12"][index])
            ) * dt
    response_intervals: dict[str, list[dict[str, float]]] = {}
    active_window_average: dict[str, float] = {}
    for name in active_channels:
        name_index = FULL12_ORDER.index(name)
        values = [abs(float(row["velocity_full12"][name_index])) for row in samples]
        peak_value = max(values)
        floor = 1.0 if name in SERVO_ORDER else 0.05
        threshold = max(floor, 0.05 * peak_value)
        active_indices = [
            index for index, value in enumerate(values) if value >= threshold
        ]
        if active_indices:
            runs: list[list[int]] = []
            for active_index in active_indices:
                if runs and active_index == runs[-1][-1] + 1:
                    runs[-1].append(active_index)
                else:
                    runs.append([active_index])
            response_intervals[name] = []
            for run in runs:
                onset = float(samples[run[0]]["time_s"]) - float(
                    samples[0]["time_s"]
                )
                end = float(samples[run[-1]]["time_s"]) - float(
                    samples[0]["time_s"]
                )
                response_intervals[name].append(
                    {
                        "threshold": threshold,
                        "onset_s": onset,
                        "end_s": end,
                        "duration_s": max(0.0, end - onset),
                    }
                )
            active_window_average[name] = sum(
                values[index] for index in active_indices
            ) / len(active_indices)
        else:
            response_intervals[name] = []
            active_window_average[name] = 0.0
    return {
        "sample_count": len(samples),
        "sample_duration_s": duration,
        "start_physics_tick": int(samples[0]["physics_tick"]),
        "end_physics_tick": int(samples[-1]["physics_tick"]),
        "mean_sample_dt_s": sum(sample_dts) / len(sample_dts),
        "max_sample_dt_error_s": max(
            abs(value - 1.0 / 120.0) for value in sample_dts
        ),
        "actual_start_full12": samples[0]["actual_full12"],
        "actual_end_full12": samples[-1]["actual_full12"],
        "actual_delta_full12": [
            float(end) - float(start)
            for start, end in zip(samples[0]["actual_full12"], samples[-1]["actual_full12"])
        ],
        "phase_average_abs_velocity": average,
        "peak_abs_velocity": peak,
        "active_window_average_abs_velocity": active_window_average,
        "carry_in_active_response_channels": carry_in,
        "carry_out_active_response_channels": carry_out,
        "carry_response_policy": "do not wait for commanded tracking convergence; preserve the next full12 target and use semantic guards",
        "actual_wheel_integral_rad": integrals,
        "actual_estimated_wheel_surface_travel_m": {
            name: integrals[name] * WHEEL_RADIUS_M for name in WHEEL_ORDER
        },
        "measured_active_response_intervals": response_intervals,
        "response_intervals_are_diagnostic_not_nominal_overlap": True,
        "trajectory_samples_normalized": _normalized_samples(samples),
        "root_start_position_w_m": samples[0]["root_position_w_m"],
        "root_end_position_w_m": samples[-1]["root_position_w_m"],
        "wheel_geometry_start": samples[0]["wheel_geometry"],
        "wheel_geometry_end": samples[-1]["wheel_geometry"],
    }


def _command_activity_intervals(
    waypoints: Sequence[dict[str, Any]],
    active_channels: Sequence[str],
    duration_s: float,
) -> dict[str, list[dict[str, float | str]]]:
    result: dict[str, list[dict[str, float | str]]] = {
        name: [] for name in active_channels
    }
    for waypoint_index, waypoint in enumerate(waypoints):
        if waypoint_index == 0:
            continue
        previous = waypoints[waypoint_index - 1]
        for name in waypoint["changed_channels"]:
            if name not in result or name not in SERVO_ORDER:
                continue
            channel_index = FULL12_ORDER.index(name)
            delta = abs(
                float(waypoint["full12"][channel_index])
                - float(previous["full12"][channel_index])
            )
            if delta <= 1e-12:
                continue
            start = float(waypoint["time_s"])
            end = min(duration_s, start + max(1.0 / 120.0, delta / 150.0))
            result[name].append(
                {
                    "onset_s": start,
                    "end_s": end,
                    "duration_s": end - start,
                    "evidence": "servo target delta / 150 deg/s reference rate",
                }
            )
    for name in active_channels:
        if name not in WHEEL_ORDER:
            continue
        channel_index = FULL12_ORDER.index(name)
        for left, right in zip(waypoints, waypoints[1:]):
            start = float(left["time_s"])
            end = float(right["time_s"])
            if end - start <= 1e-12:
                continue
            if abs(float(left["full12"][channel_index])) < 0.05:
                continue
            result[name].append(
                {
                    "onset_s": start,
                    "end_s": end,
                    "duration_s": end - start,
                    "evidence": "nonzero wheel target zero-order hold",
                }
            )
    for name, intervals in result.items():
        merged: list[dict[str, float | str]] = []
        for interval in sorted(intervals, key=lambda row: float(row["onset_s"])):
            if merged and float(interval["onset_s"]) <= float(
                merged[-1]["end_s"]
            ) + 1e-9:
                merged[-1]["end_s"] = max(
                    float(merged[-1]["end_s"]), float(interval["end_s"])
                )
                merged[-1]["duration_s"] = float(merged[-1]["end_s"]) - float(
                    merged[-1]["onset_s"]
                )
            else:
                merged.append(dict(interval))
        result[name] = merged
    return result


def _nominal_overlap_groups(
    activity_intervals: Mapping[str, Sequence[Mapping[str, float | str]]],
) -> list[dict[str, Any]]:
    boundaries = sorted(
        {
            float(value)
            for intervals in activity_intervals.values()
            for interval in intervals
            for value in (interval["onset_s"], interval["end_s"])
        }
    )
    groups: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= 1e-9:
            continue
        active = sorted(
            (
                name
                for name, intervals in activity_intervals.items()
                if any(
                    float(interval["onset_s"]) <= start + 1e-9
                    and float(interval["end_s"]) >= end - 1e-9
                    for interval in intervals
                )
            ),
            key=FULL12_ORDER.index,
        )
        if len(active) < 2:
            continue
        row = {
            "onset_s": start,
            "end_s": end,
            "overlap_duration_s": end - start,
            "channels": active,
            "evidence": "reference command launch intervals",
        }
        if groups and groups[-1]["channels"] == active and math.isclose(
            float(groups[-1]["end_s"]), start, abs_tol=1e-9
        ):
            groups[-1]["end_s"] = end
            groups[-1]["overlap_duration_s"] = end - float(
                groups[-1]["onset_s"]
            )
        else:
            groups.append(row)
    return groups


def _result_snapshot(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    last = samples[-1]
    start_time = float(samples[0]["time_s"])
    end_time = float(last["time_s"])
    tail_start = max(start_time, end_time - 0.5)
    tail = [row for row in samples if float(row["time_s"]) >= tail_start]
    wheel_final = {
        name: float(last["actual_full12"][FULL12_ORDER.index(name)])
        for name in WHEEL_ORDER
    }
    wheel_tail_peak = {
        name: max(
            abs(float(row["actual_full12"][FULL12_ORDER.index(name)]))
            for row in tail
        )
        for name in WHEEL_ORDER
    }
    return {
        "sample_count": len(samples),
        "window_start_s": start_time,
        "window_end_s": end_time,
        "window_duration_s": end_time - start_time,
        "actual_end_full12": last["actual_full12"],
        "target_end_full12": last["target_full12"],
        "root_end_position_w_m": last["root_position_w_m"],
        "wheel_geometry_end": last["wheel_geometry"],
        "wheel_final_velocity_rad_s": wheel_final,
        "wheel_tail_peak_abs_velocity_rad_s": wheel_tail_peak,
        "historical_wheel_velocity_stable_below_0_05_rad_s": all(
            value <= 0.05 for value in wheel_tail_peak.values()
        ),
    }


def build_contract(
    recording_path: Path,
    dispatch_trace_path: Path,
    telemetry_path: Path,
    leg_events_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recording = load_recording(recording_path)
    dispatch_rows = load_dispatch_rows(dispatch_trace_path)
    leg_events = load_leg_events(leg_events_path)
    telemetry_cadence = _telemetry_cadence(telemetry_path)
    rr_clear = float(
        _leg_event(leg_events, "RR", "FRONT_FACE_CLEARED")["sim_time_s"]
    )
    rl_clear = float(
        _leg_event(leg_events, "RL", "FRONT_FACE_CLEARED")["sim_time_s"]
    )
    rr_top = float(_leg_event(leg_events, "RR", "TOP_LOADED")["sim_time_s"])
    rl_top = float(_leg_event(leg_events, "RL", "TOP_LOADED")["sim_time_s"])
    if not (rr_clear < rl_clear and rr_top < rl_top):
        raise ValueError("leg event ledger does not prove RR_FIRST")
    source_atomic_events = {
        (event.step_index, event.event_index): event
        for event in recording.events
        if event.kind == "servo_wheel_launch"
    }
    phase_rows: list[dict[str, Any]] = []
    active_boundaries: dict[str, tuple[float, float]] = {}
    for definition in PHASES:
        waypoints, start_time, end_time, start, end = _waypoints_for_phase(
            definition, recording, dispatch_rows
        )
        duration = end_time - start_time
        active_boundaries[definition.state_id] = (start_time, end_time)
        active_channels = sorted(
            {
                channel
                for waypoint in waypoints
                for channel in waypoint["changed_channels"]
            },
            key=FULL12_ORDER.index,
        )
        atomic_groups: list[dict[str, Any]] = []
        for waypoint_index, waypoint in enumerate(waypoints):
            if len(waypoint["atomic_channels"]) < 2:
                continue
            source_key = next(
                (
                    (int(item["step"]), int(item["event"]))
                    for item in waypoint["source_events"]
                    if (int(item["step"]), int(item["event"]))
                    in source_atomic_events
                ),
                None,
            )
            next_time = (
                float(waypoints[waypoint_index + 1]["time_s"])
                if waypoint_index + 1 < len(waypoints)
                else duration
            )
            group: dict[str, Any] = {
                "time_s": waypoint["time_s"],
                "channels": waypoint["atomic_channels"],
                "batch_id": waypoint.get("atomic_batch_id", ""),
                "same_physics_tick": bool(waypoint["same_physics_tick"]),
                "atomic_batch_ack_valid": bool(
                    waypoint["atomic_batch_ack_valid"]
                ),
                "reference_applied_physics_tick": int(
                    waypoint["reference_physics_step"]
                ),
                "reference_first_motion_physics_tick": int(
                    waypoint["reference_first_motion_physics_step"]
                ),
                "motion_start_skew_s": waypoint.get("motion_start_skew_s", 0.0),
                "active_until_next_waypoint_s": max(
                    0.0, next_time - float(waypoint["time_s"])
                ),
                "source_full12_atomic": source_key is not None,
            }
            if source_key is not None:
                source_event = source_atomic_events[source_key]
                group.update(
                    {
                        "source_step": source_event.step_index,
                        "source_event": source_event.event_index,
                        "source_batch_id": source_event.batch_id,
                        "required_runtime_channels": list(FULL12_ORDER),
                        "source_changed_channels": list(source_event.active_channels),
                        "source_before_full12": list(source_event.before.vector()),
                        "source_target_full12": list(source_event.after.vector()),
                        "servo_velocity_deg_s": source_event.servo_velocity_deg_s,
                    }
                )
            atomic_groups.append(group)
        command_activity_intervals = _command_activity_intervals(
            waypoints, active_channels, duration
        )
        phase_rows.append(
            {
                "state_id": definition.state_id,
                "macro_phase": definition.macro_phase,
                "state_name": definition.name,
                "physical_purpose": definition.physical_purpose,
                "reference_steps": list(range(definition.first_step, definition.last_step + 1)),
                "reference_events": [
                    [event.step_index, event.event_index]
                    for event in recording.events
                    if definition.first_step
                    <= event.step_index
                    <= definition.last_step
                ],
                "applied_dispatch_events": sorted(
                    {
                        (row["source_step"], row["source_event"])
                        for row in dispatch_rows
                        if definition.first_step
                        <= row["source_step"]
                        <= definition.last_step
                    }
                ),
                "reference_sim_start_s": start_time,
                "reference_sim_end_s": end_time,
                "active_duration_s": duration,
                "start_full12": _vector(start),
                "end_full12": _vector(end),
                "delta_full12": [end[name] - start[name] for name in FULL12_ORDER],
                "active_channels": active_channels,
                "waypoints": waypoints,
                "command_metrics": _command_metrics(waypoints, duration),
                "command_activity_intervals": command_activity_intervals,
                "atomic_groups": atomic_groups,
                "overlap_timing": _nominal_overlap_groups(
                    command_activity_intervals
                ),
                "completion_event": definition.completion_event,
                "completion_evidence": _completion_evidence(
                    definition.state_id,
                    start_time,
                    end_time,
                    leg_events,
                ),
                "ppo_action_mask_full12": list(definition.ppo_action_mask),
            }
        )
    physics_dt_s = 1.0 / 120.0
    active_readback_boundaries: dict[str, tuple[float, float]] = {}
    result_boundaries: dict[str, tuple[float, float]] = {}
    for index, phase in enumerate(phase_rows):
        start = float(phase["reference_sim_start_s"])
        motion_end = float(phase["reference_sim_end_s"])
        active_readback_boundaries[str(phase["state_id"])] = (
            start,
            motion_end + physics_dt_s + 1e-9,
        )
        if index + 1 < len(phase_rows):
            next_start = float(phase_rows[index + 1]["reference_sim_start_s"])
            result_end = next_start - 0.5 / 120.0
        else:
            result_end = float("inf")
        result_boundaries[str(phase["state_id"])] = (motion_end, result_end)
    active_telemetry = _phase_telemetry(
        telemetry_path, active_readback_boundaries
    )
    result_telemetry = _phase_telemetry(telemetry_path, result_boundaries)
    for phase in phase_rows:
        state_id = str(phase["state_id"])
        active_samples = active_telemetry[state_id]
        terminal_target = active_samples[-1]["target_full12"]
        if any(
            not math.isclose(float(actual), float(expected), abs_tol=1e-6)
            for actual, expected in zip(terminal_target, phase["end_full12"])
        ):
            raise ValueError(
                f"{state_id}: +1 tick terminal target does not equal phase endpoint"
            )
        phase["reference_sim_readback_end_s"] = float(
            active_samples[-1]["time_s"]
        )
        phase["reference_command_readback_latency_s"] = max(
            0.0,
            float(active_samples[-1]["time_s"])
            - float(phase["reference_sim_end_s"]),
        )
        phase["reference_actual"] = _actual_metrics(
            active_samples, phase["active_channels"]
        )
        result_samples = result_telemetry[state_id]
        phase["reference_sim_result_end_s"] = float(
            result_samples[-1]["time_s"]
        )
        phase["reference_observation_tail_duration_s"] = max(
            0.0,
            float(result_samples[-1]["time_s"])
            - float(phase["reference_sim_end_s"]),
        )
        phase["reference_result_observation"] = _result_snapshot(
            result_samples
        )
        phase["reference_result_observation"][
            "actual_delta_from_motion_start_full12"
        ] = [
            float(end) - float(start)
            for start, end in zip(
                phase["reference_actual"]["actual_start_full12"],
                phase["reference_result_observation"]["actual_end_full12"],
            )
        ]
        active_indexes = [
            FULL12_ORDER.index(name) for name in phase["active_channels"]
        ]
        phase["completion_evidence"]["reference_observed"] = {
            "active_channel_actual_start": {
                name: phase["reference_actual"]["actual_start_full12"][index]
                for name, index in zip(phase["active_channels"], active_indexes)
            },
            "active_channel_actual_motion_readback": {
                name: phase["reference_actual"]["actual_end_full12"][index]
                for name, index in zip(phase["active_channels"], active_indexes)
            },
            "active_channel_actual_result_readback": {
                name: phase["reference_result_observation"][
                    "actual_end_full12"
                ][index]
                for name, index in zip(phase["active_channels"], active_indexes)
            },
            "wheel_geometry_motion_start": phase["reference_actual"][
                "wheel_geometry_start"
            ],
            "wheel_geometry_motion_readback": phase["reference_actual"][
                "wheel_geometry_end"
            ],
            "wheel_geometry_result": phase["reference_result_observation"][
                "wheel_geometry_end"
            ],
            "root_start_position_w_m": phase["reference_actual"][
                "root_start_position_w_m"
            ],
            "root_result_position_w_m": phase[
                "reference_result_observation"
            ]["root_end_position_w_m"],
        }
        if state_id == "P13":
            phase["completion_evidence"]["historical_stop_evidence"] = {
                "zero_target_applied": all(
                    abs(float(value)) <= 1e-9
                    for value in phase["reference_result_observation"][
                        "target_end_full12"
                    ][8:]
                ),
                "strict_0_05_rad_s_stability_established": phase[
                    "reference_result_observation"
                ]["historical_wheel_velocity_stable_below_0_05_rad_s"],
                "new_fsm_requirement": "zero targets plus measured wheel-speed stable-decay debounce; historical false is not inherited as success evidence",
            }
    contract = {
        "schema": "wlr50_clean.recording_motion_contract.v1",
        "reference_version": "v010_20260806_220745_363972_manual",
        "rear_leg_order": "RR_FIRST",
        "cross_version_splice": False,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "servo_reference_velocity_deg_s": 150.0,
        "wheel_radius_m": WHEEL_RADIUS_M,
        "execution_semantics": {
            "full12_output_each_physics_tick": True,
            "full12_output_atomic_write": True,
            "servo_profile": "120 Hz monotonic continuous interpolation followed by a stateful hard 150 deg/s per-tick slew limit; this is the measured-response-like alternative to quintic shaping",
            "wheel_profile": "zero-order hold between compact launch/stop waypoints",
            "duplicate_time_phase_entry_then_launch": "phase_entry establishes the pre-command vector; all t=0 launch waypoints are then applied deterministically on the first physics tick",
            "state_transition_source": "live sensor completion evidence; compact timing never advances the FSM",
        },
        "full12_order": list(FULL12_ORDER),
        "servo_order8": list(SERVO_ORDER),
        "wheel_order4": list(WHEEL_ORDER),
        "source": {
            "accepted_steps_sha256": recording.sha256,
            "dispatch_trace_sha256": sha256_file(dispatch_trace_path),
            "telemetry_120hz_sha256": sha256_file(telemetry_path),
            "leg_event_ledger_sha256": sha256_file(leg_events_path),
            "telemetry_cadence": telemetry_cadence,
        },
        "source_full12_atomic_event_count": len(source_atomic_events),
        "rear_order_evidence": {
            "classification": "RR_FIRST",
            "RR_front_face_cleared_sim_time_s": rr_clear,
            "RL_front_face_cleared_sim_time_s": rl_clear,
            "RR_top_loaded_sim_time_s": rr_top,
            "RL_top_loaded_sim_time_s": rl_top,
            "validated_order": rr_clear < rl_clear and rr_top < rl_top,
            "rule": "front-face clearance after reference-like active hip/knee motion, then latched TOP_LOADED",
        },
        "tolerance": {
            "relative": 0.15,
            "joint_absolute_floor_deg": 2.0,
            "wheel_velocity_absolute_floor_rad_s": 0.05,
            "wheel_integral_absolute_floor_rad": 0.05,
            "wheel_surface_travel_absolute_floor_m": 0.05 * WHEEL_RADIUS_M,
            "feedback_correction_relative_limit": 0.15,
            "commanded_endpoint": {
                "formula": "abs(fsm_command_end-reference_end) <= max(2 deg, 0.15*abs(reference_delta))",
                "reference_baseline": "phase.end_full12 command vector",
                "required": True,
            },
            "actual_endpoint": {
                "formula": "abs(fsm_actual_readback-reference_measured_result_readback) <= max(2 deg, 0.15*abs(reference_delta))",
                "reference_baseline": "phase.reference_result_observation.actual_end_full12",
                "note": "command tracking need not converge at a phase boundary; v010 contains physical cross-state response carry-over",
                "required": True,
            },
            "commanded_delta": {
                "formula": "abs(fsm_command_delta-reference_command_delta) <= max(2 deg, 0.15*abs(reference_command_delta))",
                "reference_baseline": "phase.delta_full12",
            },
            "actual_delta": {
                "formula": "abs(fsm_actual_delta-reference_measured_actual_delta) <= max(2 deg, 0.15*abs(reference_command_delta))",
                "reference_baseline": "phase.reference_result_observation.actual_delta_from_motion_start_full12",
                "tolerance_magnitude_baseline": "phase.delta_full12",
            },
            "active_duration": {
                "ratio_range": [0.85, 1.15],
                "pre_motion_wait_excluded": True,
                "post_motion_verify_excluded": True,
                "recovery_duration_excluded": True,
            },
            "average_and_peak_velocity": {
                "ratio_range": [0.85, 1.15],
                "per_active_channel_window": True,
                "command_profile_limit_deg_s": 150.0,
                "measured_peak_baseline": "reference measured peak, which may exceed the command-profile limit through physical overshoot",
                "true_joint_hard_limit_safety_is_separate": True,
            },
            "wheel_target_velocity": {
                "ratio_range": [0.85, 1.15],
                "absolute_floor_rad_s": 0.05,
            },
            "wheel_integral_and_surface_travel": {
                "ratio_range": [0.85, 1.15],
                "integral_absolute_floor_rad": 0.05,
                "surface_travel_absolute_floor_m": 0.05 * WHEEL_RADIUS_M,
                "command_target_baseline": "phase.command_metrics.wheel_integral_rad and estimated_wheel_surface_travel_m",
                "measured_actual_baseline": "phase.reference_actual.actual_wheel_integral_rad and actual_estimated_wheel_surface_travel_m",
                "command_and_measured_gates_are_separate": True,
            },
            "trajectory": {
                "comparison": "phase-normalized reference actual versus FSM actual on active joints",
                "metric": "RMSE of phase-start-relative trajectories / max(abs(reference_delta), 2 deg), per active servo",
                "maximum_normalized_error": 0.15,
            },
            "overlap": {
                "onset_offset_relative_error_max": 0.15,
                "duration_relative_error_max": 0.15,
                "near_zero_absolute_floor_s": 0.008333333333333333,
            },
            "source_atomic_full12": {
                "required_same_120hz_tick": True,
                "maximum_onset_skew_s": 0.0,
            },
        },
        "phases": phase_rows,
    }
    semantic = {
        "schema": "wlr50_clean.semantic_segments.v1",
        "reference_version": contract["reference_version"],
        "rear_leg_order": "RR_FIRST",
        "role_mapping": {
            "FIRST_REAR": "RR",
            "SECOND_REAR": "RL",
            "FIRST_LOAD_TARGET_FRONT": "FL",
            "SECOND_LOAD_TARGET_FRONT": "FR",
            "FIRST_PREP_DIAGONAL": ["FR", "RL"],
            "SECOND_PREP_DIAGONAL": ["FL", "RR"],
        },
        "segments": [
            {
                "state_id": phase["state_id"],
                "macro_phase": phase["macro_phase"],
                "state_name": phase["state_name"],
                "physical_purpose": phase["physical_purpose"],
                "reference_steps": phase["reference_steps"],
                "reference_events": phase["reference_events"],
                "active_duration_s": phase["active_duration_s"],
                "start_full12": phase["start_full12"],
                "end_full12": phase["end_full12"],
                "delta_full12": phase["delta_full12"],
                "active_channels": phase["active_channels"],
                "atomic_groups": phase["atomic_groups"],
                "overlap_timing": phase["overlap_timing"],
                "completion_event": phase["completion_event"],
                "completion_evidence": phase["completion_evidence"],
                "completion_latency_s": phase["completion_evidence"][
                    "sensor_completion_latency_after_motion_s"
                ],
                "next_state_reason": phase["completion_evidence"][
                    "next_state_safe_reason"
                ],
            }
            for phase in phase_rows
        ],
    }
    return contract, semantic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--dispatch-trace", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--leg-events", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    args = parser.parse_args()
    contract, semantic = build_contract(
        args.recording, args.dispatch_trace, args.telemetry, args.leg_events
    )
    args.contract_output.parent.mkdir(parents=True, exist_ok=True)
    args.semantic_output.parent.mkdir(parents=True, exist_ok=True)
    args.contract_output.write_text(
        json.dumps(contract, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    args.semantic_output.write_text(
        json.dumps(semantic, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phases": len(contract["phases"]),
                "contract_output": str(args.contract_output),
                "semantic_output": str(args.semantic_output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
