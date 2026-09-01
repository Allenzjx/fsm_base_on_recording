"""Generate compact runtime StateSpecs and human-readable derivation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


COMMON_ENTRY = [
    {"guard": "previous_state_done"},
    {"guard": "no_body_obstacle_collision"},
    {"guard": "joint_hard_limits_valid"},
    {"guard": "reference_entry_compatible", "relative_limit": 0.15},
    {"guard": "critical_actuators_available"},
]

COMMON_ABORT = [
    {"guard": "body_collision_persistent_or_penetrating", "result": "TASK_FAILURE_BODY_COLLISION"},
    {"guard": "wheel_only_climb_detected", "result": "TASK_FAILURE_WHEEL_ONLY_CLIMB"},
    {"guard": "non_finite_observation_or_command", "result": "SAFETY_ABORT"},
    {"guard": "physics_explosion_or_fall", "result": "SAFETY_ABORT"},
    {"guard": "joint_hard_limit_violation", "result": "SAFETY_ABORT"},
]

COMPLETION_GUARDS: dict[str, list[dict[str, Any]]] = {
    "P01": [
        {"guard": "motion_endpoint_issued"},
        {"guard": "fr_lift_entry_geometry"},
    ],
    "P02": [
        {
            "guard": "reference_like_active_lift",
            "leg": "FR",
            "active_joints": ["front_right_knee"],
            "held_joints": ["front_right_hip"],
        },
        {"guard": "wheel_clearance_gain_or_air_history", "leg": "FR"},
    ],
    "P03": [
        {"guard": "leg_front_face_crossed_latched", "leg": "FR"},
        {"guard": "leg_top_loaded_latched", "leg": "FR"},
    ],
    "P04": [
        {"guard": "motion_endpoint_issued"},
        {"guard": "fl_lift_workspace_geometry"},
    ],
    "P05": [
        {
            "guard": "reference_like_active_lift",
            "leg": "FL",
            "active_joints": ["front_left_hip", "front_left_knee"],
        },
        {"guard": "leg_front_face_crossed_latched", "leg": "FL"},
        {"guard": "motion_endpoint_issued"},
    ],
    "P06": [
        {"guard": "leg_top_loaded_latched", "leg": "FL"},
        {"guard": "rear_pair_pre_edge_geometry"},
    ],
    "P07": [
        {"guard": "motion_endpoint_issued"},
        {"guard": "rear_entry_alignment", "relative_limit": 0.15},
    ],
    "P08": [
        {"guard": "motion_endpoint_issued", "active_joints": ["rear_left_hip"]},
        {"guard": "rr_unload_compatible_geometry"},
    ],
    "P09": [
        {
            "guard": "reference_like_active_lift",
            "leg": "RR",
            "active_joints": ["rear_right_hip", "rear_right_knee"],
        },
        {"guard": "leg_front_face_crossed_latched", "leg": "RR"},
        {"guard": "leg_top_loaded_latched", "leg": "RR"},
    ],
    "P10": [
        {"guard": "motion_endpoint_issued", "active_joints": ["rear_right_knee"]},
        {"guard": "rl_workspace_geometry"},
    ],
    "P11": [
        {
            "guard": "reference_like_active_joint_change",
            "active_joints": ["front_right_hip"],
        },
        {"guard": "rl_unload_entry_geometry"},
    ],
    "P12": [
        {
            "guard": "reference_like_active_lift",
            "leg": "RL",
            "active_joints": ["rear_left_hip", "rear_left_knee"],
        },
        {"guard": "leg_front_face_crossed_latched", "leg": "RL"},
        {"guard": "leg_top_loaded_latched", "leg": "RL"},
    ],
    "P13": [
        {"guard": "all_leg_front_face_crossings_latched"},
        {"guard": "all_wheels_final_top_geometry"},
        {"guard": "final_joint_pose_compatible", "relative_limit": 0.15},
        {"guard": "wheel_targets_zero"},
        {
            "guard": "measured_wheel_velocity_stable_decay",
            "absolute_threshold_rad_s": 0.05,
            "debounce_s": 0.5,
            "new_fsm_requirement": True,
        },
    ],
}

RELEVANT_LEGS = {
    "P01": ("FR",),
    "P02": ("FR",),
    "P03": ("FR",),
    "P04": ("FL",),
    "P05": ("FL",),
    "P06": ("FL", "RL", "RR"),
    "P07": ("RL", "RR"),
    "P08": ("RL", "RR"),
    "P09": ("RR",),
    "P10": ("RL", "RR"),
    "P11": ("FR", "RL"),
    "P12": ("RL",),
    "P13": ("FL", "FR", "RL", "RR"),
}


def _phase_velocity(phase: dict[str, Any], key: str) -> dict[str, float]:
    source = phase["reference_actual"][key]
    return {
        name: float(source[name])
        for name in phase["active_channels"]
        if name in source
    }


def _sensor_envelope(phase: dict[str, Any]) -> dict[str, Any]:
    observed = phase["completion_evidence"]["reference_observed"]
    state_id = str(phase["state_id"])
    return {
        "active_channel_actual_result_readback": observed[
            "active_channel_actual_result_readback"
        ],
        "root_result_position_w_m": observed["root_result_position_w_m"],
        "relevant_wheel_geometry_result": {
            leg: observed["wheel_geometry_result"][leg]
            for leg in RELEVANT_LEGS[state_id]
        },
        "latched_leg_events": phase["completion_evidence"][
            "required_latched_leg_events"
        ],
    }


def build_state_specs(contract: dict[str, Any]) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    phases = contract["phases"]
    for index, phase in enumerate(phases):
        state_id = str(phase["state_id"])
        sensor_latency = float(
            phase["completion_evidence"][
                "sensor_completion_latency_after_motion_s"
            ]
        )
        max_verify_wait = max(0.25, 1.15 * sensor_latency + 0.15)
        if state_id == "P13":
            max_verify_wait = 1.0
        atomic_channels = [
            {
                "onset_s": group["time_s"],
                "channels": group.get(
                    "required_runtime_channels", group["channels"]
                ),
                "same_120hz_tick": group["same_physics_tick"],
                "source_full12_atomic": group["source_full12_atomic"],
            }
            for group in phase["atomic_groups"]
        ]
        state = {
            "state_id": state_id,
            "macro_phase": int(phase["macro_phase"]),
            "state_name": str(phase["state_name"]),
            "physical_purpose": str(phase["physical_purpose"]),
            "lifecycle": [
                "WAIT_ENTRY",
                "EXECUTE_MOTION",
                "VERIFY_RESULT",
                "RECOVERY",
                "DONE",
            ],
            "reference_step": phase["reference_steps"],
            "reference_events": phase["reference_events"],
            "start_full12": phase["start_full12"],
            "end_full12": phase["end_full12"],
            "command_delta_full12": phase["delta_full12"],
            "reference_actual_endpoint_full12": phase[
                "reference_result_observation"
            ]["actual_end_full12"],
            "reference_actual_delta_full12": phase[
                "reference_result_observation"
            ]["actual_delta_from_motion_start_full12"],
            "active_channels": phase["active_channels"],
            "active_duration": phase["active_duration_s"],
            "average_velocity": _phase_velocity(
                phase, "active_window_average_abs_velocity"
            ),
            "peak_velocity": _phase_velocity(phase, "peak_abs_velocity"),
            "wheel_integral": phase["command_metrics"]["wheel_integral_rad"],
            "estimated_wheel_surface_travel_m": phase["command_metrics"][
                "estimated_wheel_surface_travel_m"
            ],
            "atomic_channels": atomic_channels,
            "overlap_timing": phase["overlap_timing"],
            "carry_in_active_response_channels": phase["reference_actual"][
                "carry_in_active_response_channels"
            ],
            "carry_out_active_response_channels": phase["reference_actual"][
                "carry_out_active_response_channels"
            ],
            "entry_conditions": COMMON_ENTRY,
            "completion_conditions": COMPLETION_GUARDS[state_id],
            "reference_sensor_envelope": _sensor_envelope(phase),
            "hard_abort_conditions": COMMON_ABORT,
            "max_verify_wait": max_verify_wait,
            "retry_budget": 1,
            "next_state": (
                str(phases[index + 1]["state_id"])
                if index + 1 < len(phases)
                else "TASK_COMPLETE"
            ),
            "recovery_state": state_id,
            "ppo_action_mask": phase["ppo_action_mask_full12"],
            "completion_event": phase["completion_event"],
            "transition_reason": phase["completion_evidence"][
                "next_state_safe_reason"
            ],
            "elapsed_time_is_not_completion_evidence": True,
        }
        states.append(state)
    return {
        "schema": "wlr50_clean.fsm_states.v1",
        "reference_version": contract["reference_version"],
        "rear_leg_order": contract["rear_leg_order"],
        "decision_hz": contract["decision_hz"],
        "motion_hz": contract["physics_hz"],
        "state_progress_watchdog_s": 0.5,
        "states": states,
    }


def _write_docs(specs: dict[str, Any], derivation: Path, transitions: Path) -> None:
    derivation_lines = [
        "# State Derivation",
        "",
        "All nominal motion is derived offline from the single frozen v010 RR_FIRST recording. Production execution loads only the compact relative motion projection; live sensor evidence, never an event cursor or absolute recording time, advances the graph.",
        "",
        "| Phase | State | Physical purpose | Reference steps | Active duration (s) | Completion event |",
        "|---|---|---|---:|---:|---|",
    ]
    transition_lines = [
        "# State Transition Table",
        "",
        "Every state follows WAIT_ENTRY → EXECUTE_MOTION → VERIFY_RESULT → RECOVERY/DONE. A 0.5 s no-progress watchdog applies only during EXECUTE_MOTION; reference response carry-over is not treated as a reason to settle or freeze.",
        "",
        "| State | Live completion guards | Next | Maximum verify wait (s) | Transition rationale |",
        "|---|---|---|---:|---|",
    ]
    for state in specs["states"]:
        steps = ",".join(str(item) for item in state["reference_step"])
        derivation_lines.append(
            f"| {state['state_id']} | {state['state_name']} | {state['physical_purpose']} | {steps} | {state['active_duration']:.3f} | {state['completion_event']} |"
        )
        guards = "; ".join(
            str(item["guard"]) for item in state["completion_conditions"]
        )
        transition_lines.append(
            f"| {state['state_id']} | {guards} | {state['next_state']} | {state['max_verify_wait']:.3f} | {state['transition_reason']} |"
        )
    derivation.parent.mkdir(parents=True, exist_ok=True)
    derivation.write_text("\n".join(derivation_lines) + "\n", encoding="utf-8")
    transitions.write_text("\n".join(transition_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--derivation", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    specs = build_state_specs(contract)
    args.states.parent.mkdir(parents=True, exist_ok=True)
    args.states.write_text(
        yaml.safe_dump(specs, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_docs(specs, args.derivation, args.transitions)
    print(json.dumps({"states": len(specs["states"]), "output": str(args.states)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
