from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import artifacts
from wlr50_clean.ppo import phase_effective_entry as effective_entry_module
from wlr50_clean.ppo.phase_effective_entry import (
    CONTACT_SOURCE,
    DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
    EffectivePhaseEntryError,
    FINGERPRINT_FIELDS,
    PHASE_IDS,
    ValidatedEffectivePhaseEntryContract,
    binary64_ulp_distance,
    build_effective_phase_entry_contract,
    capture_validated_effective_phase_entry_contract,
    validate_effective_phase_entry_comparison,
)
from wlr50_clean.ppo.phase_snapshots import (
    DEFAULT_PHASE_SNAPSHOT_ROOT,
    SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS,
    SOURCE_ACK_REPLAY_INVARIANT_FIELDS,
    capture_validated_phase_snapshot_bundle,
    load_validated_phase_snapshot_payload,
)


@pytest.fixture(scope="module")
def snapshot_bundle():
    return capture_validated_phase_snapshot_bundle(DEFAULT_PHASE_SNAPSHOT_ROOT)


@pytest.fixture(scope="module")
def contract():
    """Keep comparison tests independent of the superseded on-disk derivation."""

    contract_bytes = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_bytes()
    payload = json.loads(contract_bytes)
    sidecar = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.with_suffix(".sha256")
    environment = effective_entry_module.DEFAULT_ENVIRONMENT_LOCK_PATH
    frozen = effective_entry_module.DEFAULT_FROZEN_LEDGER_PATH
    entries = []
    for phase in PHASE_IDS:
        entry = copy.deepcopy(payload["phases"][phase])
        component_state, component_binary = _synthetic_component_state(phase)
        entry["schema"] = effective_entry_module.ENTRY_SCHEMA
        entry["effective_component_state"] = component_state
        entry["effective_component_state_binary64_hex"] = component_binary
        unhashed_entry = dict(entry)
        unhashed_entry.pop("entry_sha256", None)
        entry["entry_sha256"] = hashlib.sha256(
            effective_entry_module._canonical_bytes(unhashed_entry)
        ).hexdigest()
        entries.append((phase, entry))
    return ValidatedEffectivePhaseEntryContract(
        contract_path=DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
        sidecar_path=sidecar,
        environment_lock_path=environment,
        frozen_ledger_path=frozen,
        contract_bytes=contract_bytes,
        sidecar_bytes=sidecar.read_bytes(),
        environment_lock_bytes=environment.read_bytes(),
        frozen_ledger_bytes=frozen.read_bytes(),
        file_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        sidecar_file_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        contract_sha256=payload["contract_sha256"],
        phase_snapshot_bundle_sha256=payload["derivation"][
            "phase_snapshot_bundle"
        ]["bundle_sha256"],
        entries=tuple(entries),
        filesystem_identity=(),
    )


def _synthetic_component_state(phase: str) -> tuple[dict, dict]:
    """Return a deterministic, hash-bound complete physical state for tests."""

    phase_number = int(phase[1:])
    offset = phase_number / 100.0
    state = {
        "schema": effective_entry_module.COMPONENT_STATE_SCHEMA,
        "units": dict(effective_entry_module.COMPONENT_UNITS),
        "root_position_w_m": [offset, -offset, 0.25 + offset],
        "root_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "root_linear_velocity_w_m_s": [offset / 10.0, 0.0, -offset / 20.0],
        "root_angular_velocity_w_rad_s": [0.0, -offset / 5.0, offset / 4.0],
        "servo_logical_position_deg": [
            offset + index * 0.125 for index in range(8)
        ],
        "servo_logical_velocity_deg_s": [
            -offset - index * 0.25 for index in range(8)
        ],
        "wheel_logical_velocity_rad_s": [
            offset + index * 0.5 for index in range(4)
        ],
        "servo_order": list(effective_entry_module.SERVO_ORDER),
        "wheel_order": list(effective_entry_module.WHEEL_ORDER),
        "wheel_centers_w_m": {
            wheel: [offset + index, -offset, 0.1 + index / 100.0]
            for index, wheel in enumerate(effective_entry_module.WHEEL_ORDER)
        },
        "wheel_bottoms_w_m": {
            wheel: [offset + index, -offset, 0.05 + index / 100.0]
            for index, wheel in enumerate(effective_entry_module.WHEEL_ORDER)
        },
    }
    state["sha256"] = hashlib.sha256(
        effective_entry_module._canonical_bytes(state)
    ).hexdigest()
    normalized, binary = effective_entry_module._component_state(state)
    return normalized, binary


def _comparison(entry):
    pairs = {}
    exact = {}
    for wheel in entry["raw_contacts"]:
        if wheel == "signature_sha256":
            continue
        reference = entry["raw_contacts"][wheel]
        pairs[wheel] = {
            pair_name: {
                "pair_verified": pair["pair_verified"],
                "source": pair["source"],
                "force_w_n": list(pair["force_w_n"]),
            }
            for pair_name, pair in (
                ("ground", reference["ground"]),
                ("obstacle", reference["obstacle"]),
            )
        }
        exact[wheel] = {
            "body_name": reference["body_name"],
            "actual_class": reference["classification"],
            "actual_ground_active": reference["ground"]["active"],
            "actual_obstacle_active": reference["obstacle"]["active"],
        }
    return {
        "maximum_errors": dict(entry["post_prime_fingerprint"]),
        "effective_component_state": copy.deepcopy(
            entry["effective_component_state"]
        ),
        "raw_physx_contacts": {"pairs": pairs},
        "exact_contacts": exact,
    }


def _calibration_comparison(entry):
    comparison = _comparison(entry)
    comparison["schema"] = "wlr50_clean.phase_snapshot_live_comparison.v2"
    raw = comparison["raw_physx_contacts"]
    raw["schema"] = "wlr50_clean.phase_snapshot_raw_physx_contact.v1"
    raw["sha256"] = hashlib.sha256(
        effective_entry_module._canonical_bytes(raw)
    ).hexdigest()
    return comparison


def _entry_guard(phase: str):
    reference_value = {"checked_servo_channels": []}
    alignment = None
    if phase == "P10":
        alignment = {
            "actual_deg_s": 24.0,
            "reference_deg_s": 23.0,
            "error_deg_s": 1.0,
            "limit_deg_s": 3.45,
            "signed_positive_rebound_required": True,
        }
        reference_value = {"rear_right_knee_velocity": dict(alignment)}
    values = {
        "previous_state_done": True,
        "no_body_obstacle_collision": "no exact base_link/obstacle contact",
        "joint_hard_limits_valid": {},
        "reference_entry_compatible": reference_value,
        "critical_actuators_available": {"joint_count": 8, "wheel_count": 4},
    }
    names = list(effective_entry_module._AUTHORED_ENTRY_GUARDS)
    return {
        "schema": "wlr50_clean.phase_effective_entry_controller.v1",
        "verified": True,
        "phase": phase,
        "lifecycle": "EXECUTE_MOTION",
        "nonterminal": True,
        "unblocked": True,
        "authored_entry_guard_names": names,
        "entry_guard_evidence": [
            {
                "name": name,
                "passed": True,
                "value": values[name],
                "source": "controller",
                "reason": "",
            }
            for name in names
        ],
        "p10_signed_velocity_alignment": alignment,
    }


def _adaptive_source_replay_evidence(command, index: int, previous_post: str | None):
    """Build one self-consistent live-feedback-adaptive replay proof row."""

    tick = command["control_physics_tick"]
    feedback = {
        "schema": "wlr50_clean.phase_snapshot_live_servo_feedback.v1",
        "canonical_servo_order": list(effective_entry_module.SERVO_ORDER),
        "unit": "rad",
        "physical_position_rad": [tick / 10000.0 + i / 1000.0 for i in range(8)],
        "tensor_dtype": "torch.float64",
        "tensor_device": "cpu",
    }
    feedback["sha256"] = hashlib.sha256(
        effective_entry_module._canonical_bytes(feedback)
    ).hexdigest()
    source_pre_sha = hashlib.sha256(
        effective_entry_module._canonical_bytes(command["mapper_pre_state"])
    ).hexdigest()
    source_post_sha = hashlib.sha256(
        effective_entry_module._canonical_bytes(command["mapper_post_state"])
    ).hexdigest()
    live_pre_sha = source_pre_sha if previous_post is None else previous_post
    live_post_sha = hashlib.sha256(
        effective_entry_module._canonical_bytes(
            {
                "schema": "test.live_mapper_post.v1",
                "source_control_physics_tick": tick,
                "live_pre_state_sha256": live_pre_sha,
                "feedback_input_sha256": feedback["sha256"],
            }
        )
    ).hexdigest()
    feedback_output_sha = hashlib.sha256(
        effective_entry_module._canonical_bytes(
            {
                name: copy.deepcopy(command["expected_atomic_ack"][name])
                for name in SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS
            }
        )
    ).hexdigest()
    feedback_output = {
        name: copy.deepcopy(command["expected_atomic_ack"][name])
        for name in SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS
    }
    replayed_drive_sha = feedback_output_sha
    replayed_actuation_sha = hashlib.sha256(
        f"adaptive-actuation-{tick}".encode("ascii")
    ).hexdigest()
    output_contract = {
        "schema": "wlr50_clean.phase_snapshot_live_output_contract.v2",
        "all_values_finite": True,
        "logical_clamp_verified": True,
        "servo_aliases_verified": True,
        "realized_bias_verified": True,
        "servo_hard_limits_verified": True,
        "wheel_hard_limits_verified": True,
        "final_drive_slew_verified": True,
        "maximum_final_drive_slew_deg": 0.0,
        "maximum_allowed_final_drive_slew_deg": command["mapper_configuration"][
            "maximum_delta_deg"
        ],
        "physical_sign_order_unit_conversion_verified": True,
        "mapper_output_state_verified": True,
        "live_feedback_mapper_replay_verified": True,
        "live_feedback_input_sha256": feedback["sha256"],
        "predicted_mapper_post_state_sha256": live_post_sha,
        "feedback_conditioned_output": feedback_output,
        "feedback_conditioned_output_sha256": feedback_output_sha,
        "verified": True,
    }
    actuation = {
        "schema": "wlr50_clean.phase_snapshot_source_input_live_output.v2",
        "source_control_physics_tick": tick,
        "all_replay_invariant_fields_match": True,
        "replay_invariant_field_matches": {
            name: True for name in SOURCE_ACK_REPLAY_INVARIANT_FIELDS
        },
        "historical_feedback_field_matches": {
            name: False for name in SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS
        },
        "historical_feedback_equivalence_claimed": False,
        "live_feedback_adaptive_output_valid": True,
        "logical_target_fallback_used": False,
        "source_command_file_sha256": command["source_command_file_sha256"],
        "source_observation_file_sha256": command[
            "source_observation_file_sha256"
        ],
        "source_command_row_canonical_sha256": command[
            "source_command_row_canonical_sha256"
        ],
        "source_observation_row_canonical_sha256": command[
            "source_observation_row_canonical_sha256"
        ],
        "source_adapter_input_sha256": command["source_adapter_input_sha256"],
        "replayed_adapter_input_sha256": command["source_adapter_input_sha256"],
        "adapter_input_hash_matches": True,
        "source_drive_target_full12_sha256": command[
            "drive_target_full12_sha256"
        ],
        "replayed_drive_target_full12_sha256": replayed_drive_sha,
        "source_target_hash_matches": False,
        "source_actuation_contract_sha256": command["actuation_contract_sha256"],
        "replayed_actuation_contract_sha256": replayed_actuation_sha,
        "source_actuation_hash_matches": False,
        "live_feedback_input": feedback,
        "live_output_contract": output_contract,
        "source_atomic_physics_tick": command["source_atomic_physics_tick"],
        "reset_prime_physics_tick": 180 + index,
        "source_atomic_write_count": command["source_atomic_write_count"],
        "reset_prime_write_count": 181 + index,
        "clock_and_write_count_fields_intentionally_remapped": True,
    }
    mapper = {
        "schema": "wlr50_clean.phase_snapshot_live_mapper_transition.v2",
        "source_transition": "source_tick_t_minus_1_to_t",
        "source_control_physics_tick": tick,
        "historical_post_field_matches": {
            name: False for name in command["mapper_post_state"]
        },
        "historical_post_field_maximum_numeric_error": {
            name: 1.0 for name in command["mapper_post_state"]
        },
        "all_historical_post_fields_match": False,
        "historical_feedback_equivalence_claimed": False,
        "first_replay_tick": index == 0,
        "first_pre_state_matches_source": True if index == 0 else None,
        "source_pre_state_sha256": source_pre_sha,
        "source_post_state_sha256": source_post_sha,
        "live_pre_state_sha256": live_pre_sha,
        "live_post_state_sha256": live_post_sha,
        "feedback_tick_increment": 1,
        "feedback_schedule_verified": True,
        "ack_cross_bindings_verified": True,
        "natural_live_state_continuity_verified": True,
        "per_tick_mapper_restore_count": 0,
        "reached_naturally_by_single_atomic_apply": True,
        "restored_after_prime": False,
    }
    return actuation, mapper, live_post_sha


def _calibration_attempt(contract, snapshot_bundle, phase: str, lifecycle: str):
    snapshot_payload, snapshot = load_validated_phase_snapshot_payload(
        snapshot_bundle, phase
    )
    replay_steps = snapshot_payload["source_replay_steps"]
    target_entry_tick = snapshot.source_tick + replay_steps
    predecessor_verify_tick = snapshot_payload.get("predecessor_verify_tick")
    predecessor_verify_time_s = snapshot_payload.get("predecessor_verify_time_s")
    controller_anchor_tick = snapshot_payload.get("controller_anchor_tick")
    controller_anchor_time_s = snapshot_payload.get("controller_anchor_time_s")
    source_commands = snapshot_payload["source_commands"]
    control_ticks = [row["control_physics_tick"] for row in source_commands]
    replay_anchor_contract = effective_entry_module._expected_replay_anchor_contract(
        snapshot_payload,
        phase,
        replay_steps=replay_steps,
        target_entry_tick=target_entry_tick,
        control_ticks=tuple(control_ticks),
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
    )
    comparison = _calibration_comparison(contract.entry(phase))
    proof = {
        "schema": effective_entry_module.CALIBRATION_LIVE_PROOF_SCHEMA,
        "artifact_role": effective_entry_module.CALIBRATION_ARTIFACT_ROLE,
        "verified": True,
        "calibration_only": True,
        "phase": phase,
        "source_tick": snapshot.source_tick,
        "physical_anchor_tick": replay_anchor_contract["physical_anchor_tick"],
        "physical_anchor_time_s": replay_anchor_contract["physical_anchor_time_s"],
        "predecessor_verify_tick": predecessor_verify_tick,
        "predecessor_verify_time_s": predecessor_verify_time_s,
        "controller_anchor_tick": controller_anchor_tick,
        "controller_anchor_time_s": controller_anchor_time_s,
        "target_entry_tick": target_entry_tick,
        "target_entry_time_s": replay_anchor_contract["target_entry_time_s"],
        "source_replay_steps": replay_steps,
        "physical_to_predecessor_verify_replay_steps": replay_anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ],
        "predecessor_verify_to_controller_replay_steps": replay_anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ],
        "physical_to_controller_replay_steps": replay_anchor_contract[
            "physical_to_controller_replay_steps"
        ],
        "controller_to_target_replay_steps": replay_anchor_contract[
            "controller_to_target_replay_steps"
        ],
        "hybrid_physical_controller_anchor": replay_anchor_contract[
            "hybrid_physical_controller_anchor"
        ],
        "replay_anchor_contract": replay_anchor_contract,
        "effective_entry_offset_s": replay_steps / 120.0,
        "phase_snapshot_bundle_sha256": snapshot_bundle.bundle_sha256,
        "source_snapshot_post_prime_diagnostic": comparison,
        "failures": [],
    }
    source_actuation_matches = []
    source_mapper_post_states = []
    previous_live_post = None
    for index, command in enumerate(source_commands):
        enriched_command = {
            **command,
            "source_command_file_sha256": snapshot_payload["source_artifacts"][
                "command"
            ]["sha256"],
            "source_observation_file_sha256": snapshot_payload["source_artifacts"][
                "observation"
            ]["sha256"],
        }
        actuation, mapper, previous_live_post = _adaptive_source_replay_evidence(
            enriched_command, index, previous_live_post
        )
        source_actuation_matches.append(actuation)
        source_mapper_post_states.append(mapper)
    prime_atomic_writes = [
        {
            "physics_tick": 180 + index,
            "write_count": 181 + index,
            "source_control_physics_tick": command["control_physics_tick"],
            "observation_physics_tick": command["control_physics_tick"] + 1,
            "articulation_writes_this_call": 1,
            "source_actuation_match": source_actuation_matches[index],
            "source_mapper_post_state": source_mapper_post_states[index],
        }
        for index, command in enumerate(source_commands)
    ]
    state = {
        "state_write_count": 1,
        "root_pose_writes": 1,
        "root_velocity_writes": 1,
        "joint_state_writes": 1,
        "simulation_forward_syncs": 1,
        "source_replay_steps": replay_steps,
        "physical_anchor_tick": replay_anchor_contract["physical_anchor_tick"],
        "physical_anchor_time_s": replay_anchor_contract["physical_anchor_time_s"],
        "predecessor_verify_tick": predecessor_verify_tick,
        "predecessor_verify_time_s": predecessor_verify_time_s,
        "controller_anchor_tick": controller_anchor_tick,
        "controller_anchor_time_s": controller_anchor_time_s,
        "target_entry_tick": target_entry_tick,
        "target_entry_time_s": replay_anchor_contract["target_entry_time_s"],
        "physical_to_predecessor_verify_replay_steps": replay_anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ],
        "predecessor_verify_to_controller_replay_steps": replay_anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ],
        "physical_to_controller_replay_steps": replay_anchor_contract[
            "physical_to_controller_replay_steps"
        ],
        "controller_to_target_replay_steps": replay_anchor_contract[
            "controller_to_target_replay_steps"
        ],
        "hybrid_physical_controller_anchor": replay_anchor_contract[
            "hybrid_physical_controller_anchor"
        ],
        "replay_anchor_contract": replay_anchor_contract,
        "source_replay_fsm_contexts": replay_anchor_contract[
            "source_replay_fsm_contexts"
        ],
        "episode_sensor_tick_offset": target_entry_tick,
        "effective_entry_offset_s": replay_steps / 120.0,
        "physics_steps": replay_steps,
        "prime_physics_steps": replay_steps,
        "prime_atomic_full12_writes": replay_steps,
        "prime_atomic_writes": prime_atomic_writes,
        "contact_sensor_reads_after_prime": replay_steps,
        "fsm_clock_steps_during_priming": 0,
        "episode_clock_steps_during_priming": 0,
        "sensor_history_samples_after_reset": replay_steps,
        "source_replay_guard_updates_applied": replay_steps,
        "pre_prime_state_verified": True,
        "pre_prime_joint_state_verified": True,
        "post_prime_state_rewrite_performed": False,
        "contact_and_state_share_solver_tick": True,
        "logical_target_fallback_used": False,
        "root_state_writes_confined_before_first_episode_tick": True,
        "root_velocity_write_api": "write_root_link_velocity_to_sim",
        "pre_prime_root_link_readback": {
            "verified": True,
            "all_values_finite": True,
            "all_fields_within_production_tolerances": True,
            "physics_steps_before_readback": 0,
            "contact_sensor_reads_before_readback": 0,
        },
        "source_actuation_matches": source_actuation_matches,
        "source_actuation_match": source_actuation_matches[-1],
        "source_mapper_post_states": source_mapper_post_states,
        "source_mapper_post_state": source_mapper_post_states[-1],
        "source_adapter_input_sha256s": [
            row["source_adapter_input_sha256"] for row in source_commands
        ],
        "all_source_adapter_inputs_hash_matched": True,
        "all_live_output_contracts_verified": True,
        "all_live_mapper_transitions_verified": True,
        "live_feedback_adaptive_replay": True,
        "historical_feedback_equivalence_claimed": False,
        "initial_mapper_restore_count": 1,
        "per_tick_mapper_restore_count": 0,
        "source_replay_observation_ticks": list(
            range(snapshot.source_tick + 1, target_entry_tick + 1)
        ),
        "source_replay_safety_checks": [
            {
                "schema": "wlr50_clean.phase_effective_entry_safety.v1",
                "verified": True,
                "all_failure_flags_false": True,
                "flags": {
                    "body_collision": False,
                    "combined_physics_abort_guard": False,
                    "fall": False,
                    "hard_joint_limit": False,
                    "nan_inf": False,
                    "physics_explosion": False,
                    "wheel_only_climb": False,
                },
                "source_control_physics_tick": command["control_physics_tick"],
                "observation_physics_tick": command["control_physics_tick"] + 1,
            }
            for command in source_commands
        ],
        "all_source_replay_steps_safe": True,
        "current_contact_force_provenance": "current_final_solver_force_only",
        "classifier_cold_started_before_source_replay": True,
        "classifier_source_history_restored": False,
        "classifier_source_state_restored": False,
        "classifier_history_equivalence_claimed": False,
        "raw_sensor_history_rewarmed_from_prime": True,
        "contact_backend_reset": True,
        "contact_backend_reset_after_prime": False,
        "entry_sensor_contract": {"verified": True},
        "entry_safety_contract": {
            "schema": "wlr50_clean.phase_effective_entry_safety.v1",
            "verified": True,
            "all_failure_flags_false": True,
            "flags": {
                "body_collision": False,
                "combined_physics_abort_guard": False,
                "fall": False,
                "hard_joint_limit": False,
                "nan_inf": False,
                "physics_explosion": False,
                "wheel_only_climb": False,
            },
        },
        "entry_guard_contract": _entry_guard(phase),
        "source_snapshot_post_prime_diagnostic": comparison,
        "effective_entry_contract": proof,
    }
    return {
        "attempt_index_for_phase": 0 if lifecycle == "fresh_scene" else 1,
        "attempt_kind": "primary" if lifecycle == "fresh_scene" else "reused_repeat",
        "phase": phase,
        "scene_lifecycle": lifecycle,
        "scene_existed_before": lifecycle == "reused_scene",
        "source_tick": snapshot.source_tick,
        "predecessor_verify_tick": predecessor_verify_tick,
        "predecessor_verify_time_s": predecessor_verify_time_s,
        "controller_anchor_tick": controller_anchor_tick,
        "controller_anchor_time_s": controller_anchor_time_s,
        "target_entry_tick": target_entry_tick,
        "episode_sensor_tick_offset": target_entry_tick,
        "source_replay_steps": replay_steps,
        "effective_entry_offset_s": replay_steps / 120.0,
        "source_control_physics_ticks": control_ticks,
        "source_command_row_canonical_sha256s": [
            row["source_command_row_canonical_sha256"] for row in source_commands
        ],
        "source_observation_row_canonical_sha256s": [
            row["source_observation_row_canonical_sha256"] for row in source_commands
        ],
        "source_adapter_input_sha256s": [
            row["source_adapter_input_sha256"] for row in source_commands
        ],
        "source_drive_target_full12_sha256s": [
            row["drive_target_full12_sha256"] for row in source_commands
        ],
        "source_actuation_contract_sha256s": [
            row["actuation_contract_sha256"] for row in source_commands
        ],
        "snapshot_path": str(snapshot.snapshot_path),
        "snapshot_file_sha256": snapshot.file_sha256,
        "snapshot_state_sha256": snapshot.state_sha256,
        "physics_steps_during_reset": 180 + replay_steps,
        "post_prime_contact_sensor_read_count": replay_steps,
        "extra_physics_priming_steps": replay_steps,
        "fsm_or_episode_advanced_for_probe": False,
        "reset_completed": True,
        "passed": True,
        "failure_classification": None,
        "exception": None,
        "observation_diagnostics": {
            "observation_available": True,
            "observation_physics_tick": target_entry_tick,
            "observation_simulation_time_s": target_entry_tick / 120.0,
        },
        "clocks": {
            "authoritative_frame_committed": True,
            "backend_episode_tick": 0,
            "controller_constructed": True,
            "controller_frame_committed": True,
            "controller_frame_physics_tick": 0,
            "controller_frame_state_id": phase,
            "controller_history_length": 1,
            "controller_internal_physics_tick": 1,
            "controller_last_simulation_time_s": 0.0,
            "controller_state_id": phase,
        },
        "snapshot_state_write": state,
    }


def _nextafter(value: float, count: int) -> float:
    result = float(value)
    for _ in range(count):
        result = math.nextafter(result, math.inf)
    return result


def _rehash_component_state(state: dict) -> None:
    state.pop("sha256", None)
    state["sha256"] = hashlib.sha256(
        effective_entry_module._canonical_bytes(state)
    ).hexdigest()


def _write_contract_copy(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.with_suffix(".sha256").write_bytes(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n".encode("ascii")
    )


def test_legacy_contract_is_only_a_comparison_fixture(contract) -> None:
    assert tuple(phase for phase, _ in contract.entries) == PHASE_IDS
    assert b"\r" not in contract.contract_bytes
    assert contract.contract_bytes.endswith(b"\n")


def test_snapshot_replay_contract_binds_p10_physical_predecessor_and_controller_anchors(
    snapshot_bundle,
) -> None:
    (
        payload,
        snapshot,
        replay_steps,
        target_entry_tick,
        control_ticks,
        predecessor_verify_tick,
        predecessor_verify_time_s,
        controller_anchor_tick,
        controller_anchor_time_s,
    ) = effective_entry_module._snapshot_replay_contract(snapshot_bundle, "P10")

    assert snapshot.source_tick == 7560
    assert predecessor_verify_tick == 7776
    assert predecessor_verify_time_s == 7776 / 120.0
    assert controller_anchor_tick == 7784
    assert controller_anchor_time_s == 7784 / 120.0
    assert target_entry_tick == 7794
    assert replay_steps == 234
    assert control_ticks == tuple(range(7560, 7794))
    assert [
        (row["source_fsm_state"], row["source_fsm_lifecycle"])
        for row in payload["source_commands"]
    ] == [
        *(("P09", "EXECUTE_MOTION") for _ in range(216)),
        *(("P09", "VERIFY_RESULT") for _ in range(8)),
        *(("P10", "WAIT_ENTRY") for _ in range(10)),
    ]
    anchor_contract = effective_entry_module._expected_replay_anchor_contract(
        payload,
        "P10",
        replay_steps=replay_steps,
        target_entry_tick=target_entry_tick,
        control_ticks=control_ticks,
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
    )
    assert anchor_contract["schema"].endswith(".v2")
    assert anchor_contract["source_replay_context_transition_ticks"] == [7776, 7784]
    assert [
        (row["anchor_segment"], row["source_replay_steps"])
        for row in anchor_contract["source_replay_context_segments"]
    ] == [
            ("physical_anchor_to_predecessor_verify", 216),
        ("predecessor_verify_to_controller_anchor", 8),
        ("controller_anchor_to_target_entry", 10),
    ]

    (
        _payload,
        ordinary_snapshot,
        ordinary_replay_steps,
        ordinary_target_tick,
        _ordinary_control_ticks,
        ordinary_predecessor_tick,
        ordinary_predecessor_time,
        ordinary_controller_tick,
        ordinary_controller_time,
    ) = effective_entry_module._snapshot_replay_contract(snapshot_bundle, "P02")
    assert ordinary_replay_steps == 1
    assert ordinary_target_tick == ordinary_snapshot.source_tick + 1
    assert ordinary_predecessor_tick is None
    assert ordinary_predecessor_time is None
    assert ordinary_controller_tick is None
    assert ordinary_controller_time is None


def test_nonhybrid_p10_reuses_the_single_anchor_contract(
    monkeypatch, snapshot_bundle
) -> None:
    payload, entry = load_validated_phase_snapshot_payload(snapshot_bundle, "P02")
    payload = dict(payload)
    payload["fsm_state"] = "P10"
    single_entry = SimpleNamespace(
        source_tick=entry.source_tick,
        source_replay_steps=entry.source_replay_steps,
        target_entry_tick=None,
        predecessor_verify_tick=None,
        predecessor_verify_time_s=None,
        controller_anchor_tick=None,
        controller_anchor_time_s=None,
    )
    monkeypatch.setattr(
        effective_entry_module,
        "load_validated_phase_snapshot_payload",
        lambda _bundle, _phase: (payload, single_entry),
    )

    (
        snapshot_payload,
        _snapshot_entry,
        replay_steps,
        target_entry_tick,
        control_ticks,
        predecessor_verify_tick,
        predecessor_verify_time_s,
        controller_anchor_tick,
        controller_anchor_time_s,
    ) = effective_entry_module._snapshot_replay_contract(snapshot_bundle, "P10")
    proof = effective_entry_module._expected_replay_anchor_contract(
        snapshot_payload,
        "P10",
        replay_steps=replay_steps,
        target_entry_tick=target_entry_tick,
        control_ticks=control_ticks,
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
    )

    assert replay_steps == 1
    assert predecessor_verify_tick is None
    assert controller_anchor_tick is None
    assert proof["mode"] == "single_physical_anchor"
    assert proof["hybrid_physical_controller_anchor"] is False
    assert proof["target_entry_tick_authored"] is False


def test_calibration_attempt_uses_passed_controller_entry_and_post_prime_diagnostic(
    contract, snapshot_bundle
) -> None:
    attempt = _calibration_attempt(contract, snapshot_bundle, "P10", "fresh_scene")
    attempt["snapshot_state_write"]["priming_observation"] = {
        "maximum_errors": {field: 999.0 for field in FINGERPRINT_FIELDS}
    }

    comparison = effective_entry_module._validated_probe_attempt(
        attempt,
        phase="P10",
        lifecycle="fresh_scene",
        phase_snapshot_bundle=snapshot_bundle,
    )

    assert comparison is attempt["snapshot_state_write"][
        "source_snapshot_post_prime_diagnostic"
    ]
    assert comparison["maximum_errors"] != attempt["snapshot_state_write"][
        "priming_observation"
    ]["maximum_errors"]
    assert attempt["source_tick"] == 7560
    assert attempt["predecessor_verify_tick"] == 7776
    assert attempt["predecessor_verify_time_s"] == 7776 / 120.0
    assert attempt["controller_anchor_tick"] == 7784
    assert attempt["controller_anchor_time_s"] == 7784 / 120.0
    assert attempt["source_replay_steps"] == 234
    assert attempt["target_entry_tick"] - attempt["source_tick"] == 234
    assert len(attempt["snapshot_state_write"]["prime_atomic_writes"]) == 234
    assert attempt["clocks"]["controller_history_length"] == 1


@pytest.mark.parametrize(
    "tamper",
    (
        "failed_attempt",
        "guard",
        "controller_history",
        "calibration_proof",
        "attempt_predecessor_anchor",
        "attempt_controller_anchor",
        "state_predecessor_anchor_time",
        "state_controller_anchor_time",
        "proof_predecessor_anchor",
        "proof_controller_anchor",
        "state_physical_anchor",
        "state_target_time",
        "state_predecessor_split_steps",
        "state_split_steps",
        "state_hybrid_flag",
        "state_replay_anchor",
        "state_replay_context",
        "proof_physical_anchor",
        "proof_target_time",
        "proof_predecessor_split_steps",
        "proof_split_steps",
        "proof_hybrid_flag",
        "proof_replay_anchor",
        "replay_count",
        "replay_tick",
        "replay_match",
        "contact_reads",
        "intermediate_safety",
        "reset_local_physics_tick",
        "reset_local_write_count",
        "reset_local_remap_flag",
    ),
)
def test_calibration_attempt_rejects_failed_or_unproven_evidence(
    contract, snapshot_bundle, tamper: str
) -> None:
    attempt = _calibration_attempt(contract, snapshot_bundle, "P02", "fresh_scene")
    if tamper == "failed_attempt":
        attempt.update(
            {
                "reset_completed": False,
                "passed": False,
                "failure_classification": "ORDINARY_POST_WRITE_RESTORE_MISMATCH",
                "exception": {"type": "SensorContractFailure"},
            }
        )
    elif tamper == "guard":
        attempt["snapshot_state_write"]["entry_guard_contract"]["verified"] = False
    elif tamper == "controller_history":
        attempt["clocks"]["controller_history_length"] = 0
    elif tamper == "calibration_proof":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "source_snapshot_post_prime_diagnostic"
        ] = {"schema": "tampered"}
    elif tamper == "attempt_predecessor_anchor":
        attempt["predecessor_verify_tick"] = 1
    elif tamper == "attempt_controller_anchor":
        attempt["controller_anchor_tick"] = 1
    elif tamper == "state_predecessor_anchor_time":
        attempt["snapshot_state_write"]["predecessor_verify_time_s"] = 0.0
    elif tamper == "state_controller_anchor_time":
        attempt["snapshot_state_write"]["controller_anchor_time_s"] = 0.0
    elif tamper == "proof_predecessor_anchor":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "predecessor_verify_tick"
        ] = 1
    elif tamper == "proof_controller_anchor":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "controller_anchor_tick"
        ] = 1
    elif tamper == "state_physical_anchor":
        attempt["snapshot_state_write"]["physical_anchor_tick"] += 1
    elif tamper == "state_target_time":
        attempt["snapshot_state_write"]["target_entry_time_s"] += 1.0 / 120.0
    elif tamper == "state_predecessor_split_steps":
        attempt["snapshot_state_write"][
            "physical_to_predecessor_verify_replay_steps"
        ] = 0
    elif tamper == "state_split_steps":
        attempt["snapshot_state_write"]["physical_to_controller_replay_steps"] = 0
    elif tamper == "state_hybrid_flag":
        attempt["snapshot_state_write"]["hybrid_physical_controller_anchor"] = True
    elif tamper == "state_replay_anchor":
        attempt["snapshot_state_write"]["replay_anchor_contract"] = {
            "schema": "tampered"
        }
    elif tamper == "state_replay_context":
        attempt["snapshot_state_write"]["source_replay_fsm_contexts"][0][
            "source_control_physics_tick"
        ] += 1
    elif tamper == "proof_physical_anchor":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "physical_anchor_tick"
        ] += 1
    elif tamper == "proof_target_time":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "target_entry_time_s"
        ] += 1.0 / 120.0
    elif tamper == "proof_predecessor_split_steps":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "predecessor_verify_to_controller_replay_steps"
        ] = 0
    elif tamper == "proof_split_steps":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "controller_to_target_replay_steps"
        ] = 0
    elif tamper == "proof_hybrid_flag":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "hybrid_physical_controller_anchor"
        ] = True
    elif tamper == "proof_replay_anchor":
        attempt["snapshot_state_write"]["effective_entry_contract"][
            "replay_anchor_contract"
        ] = {"schema": "tampered"}
    elif tamper == "replay_count":
        attempt["snapshot_state_write"]["source_replay_steps"] += 1
    elif tamper == "replay_tick":
        attempt["snapshot_state_write"]["source_actuation_matches"][0][
            "source_control_physics_tick"
        ] += 1
    elif tamper == "replay_match":
        attempt["snapshot_state_write"]["source_actuation_matches"][0][
            "all_replay_invariant_fields_match"
        ] = False
    elif tamper == "contact_reads":
        attempt["snapshot_state_write"]["contact_sensor_reads_after_prime"] += 1
    elif tamper == "intermediate_safety":
        attempt["snapshot_state_write"]["source_replay_safety_checks"][0][
            "flags"
        ]["body_collision"] = True
    elif tamper == "reset_local_physics_tick":
        attempt["snapshot_state_write"]["prime_atomic_writes"][0][
            "physics_tick"
        ] += 1
    elif tamper == "reset_local_write_count":
        attempt["snapshot_state_write"]["prime_atomic_writes"][0][
            "write_count"
        ] += 1
    else:
        attempt["snapshot_state_write"]["source_actuation_matches"][0][
            "clock_and_write_count_fields_intentionally_remapped"
        ] = False

    with pytest.raises(EffectivePhaseEntryError):
        effective_entry_module._validated_probe_attempt(
            attempt,
            phase="P02",
            lifecycle="fresh_scene",
            phase_snapshot_bundle=snapshot_bundle,
        )


def test_fresh_reused_replay_proofs_must_be_identical(
    contract, snapshot_bundle
) -> None:
    fresh = _calibration_attempt(contract, snapshot_bundle, "P10", "fresh_scene")
    reused = _calibration_attempt(contract, snapshot_bundle, "P10", "reused_scene")
    effective_entry_module._assert_replay_attempts_bit_identical(fresh, reused)

    reused["predecessor_verify_tick"] += 1
    with pytest.raises(EffectivePhaseEntryError, match="attempt metadata differs"):
        effective_entry_module._assert_replay_attempts_bit_identical(fresh, reused)

    reused = _calibration_attempt(contract, snapshot_bundle, "P10", "reused_scene")
    reused["snapshot_state_write"][
        "predecessor_verify_to_controller_replay_steps"
    ] += 1
    with pytest.raises(EffectivePhaseEntryError, match="replay proof differs"):
        effective_entry_module._assert_replay_attempts_bit_identical(fresh, reused)

    reused = _calibration_attempt(contract, snapshot_bundle, "P10", "reused_scene")
    reused["snapshot_state_write"]["source_mapper_post_states"][3][
        "historical_post_field_matches"
    ]["feedback_tick"] = True
    with pytest.raises(EffectivePhaseEntryError, match="replay proof differs"):
        effective_entry_module._assert_replay_attempts_bit_identical(fresh, reused)


@pytest.mark.parametrize(
    "tamper",
    (
        "attempt_predecessor_tick",
        "attempt_predecessor_time",
        "state_predecessor_tick",
        "state_predecessor_time",
        "proof_predecessor_tick",
        "proof_predecessor_time",
        "state_predecessor_split",
        "proof_predecessor_split",
        "anchor_transition_ticks",
        "anchor_context_segment",
    ),
)
def test_p10_predecessor_verify_anchor_tamper_fails_closed(
    contract, snapshot_bundle, tamper: str
) -> None:
    attempt = _calibration_attempt(contract, snapshot_bundle, "P10", "fresh_scene")
    state = attempt["snapshot_state_write"]
    proof = state["effective_entry_contract"]
    if tamper == "attempt_predecessor_tick":
        attempt["predecessor_verify_tick"] += 1
    elif tamper == "attempt_predecessor_time":
        attempt["predecessor_verify_time_s"] += 1.0 / 120.0
    elif tamper == "state_predecessor_tick":
        state["predecessor_verify_tick"] += 1
    elif tamper == "state_predecessor_time":
        state["predecessor_verify_time_s"] += 1.0 / 120.0
    elif tamper == "proof_predecessor_tick":
        proof["predecessor_verify_tick"] += 1
    elif tamper == "proof_predecessor_time":
        proof["predecessor_verify_time_s"] += 1.0 / 120.0
    elif tamper == "state_predecessor_split":
        state["physical_to_predecessor_verify_replay_steps"] += 1
    elif tamper == "proof_predecessor_split":
        proof["predecessor_verify_to_controller_replay_steps"] += 1
    elif tamper == "anchor_transition_ticks":
        state["replay_anchor_contract"]["source_replay_context_transition_ticks"][0] += 1
    else:
        state["source_replay_fsm_contexts"][0]["anchor_segment"] = "tampered"

    with pytest.raises(EffectivePhaseEntryError):
        effective_entry_module._validated_probe_attempt(
            attempt,
            phase="P10",
            lifecycle="fresh_scene",
            phase_snapshot_bundle=snapshot_bundle,
        )


def test_builder_accepts_explicit_ordered_runs_and_binds_dynamic_commit(
    snapshot_bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    run_dirs = tuple(tmp_path / f"opaque-calibration-{index}" for index in range(12))
    phase_by_dir = {
        directory.resolve(): phase
        for directory, phase in zip(run_dirs, PHASE_IDS, strict=True)
    }

    def capture(directory, **kwargs):
        phase = phase_by_dir[Path(directory).resolve()]
        expected_commit = kwargs["expected_git_commit"]
        assert expected_commit in (None, commit)
        return (
            phase,
            {"phase": phase},
            {
                "phase": phase,
                "source_git_commit": commit,
                "runtime_content_sha256": "b" * 64,
                "identity_config_sha256": "c" * 64,
            },
        )

    monkeypatch.setattr(effective_entry_module, "_capture_calibration_run", capture)
    monkeypatch.setattr(
        effective_entry_module,
        "_validate_contract_payload",
        lambda *args, **kwargs: ("d" * 64, ()),
    )
    monkeypatch.setattr(
        effective_entry_module,
        "capture_validated_effective_phase_entry_contract",
        lambda *args, **kwargs: None,
    )

    payload = build_effective_phase_entry_contract(
        run_dirs,
        tmp_path / "derived.json",
        snapshot_bundle=snapshot_bundle,
    )

    assert payload["derivation"]["source_git_commit"] == commit
    assert tuple(payload["phases"]) == PHASE_IDS
    assert [row["phase"] for row in payload["derivation"]["calibration_artifacts"]] == list(
        PHASE_IDS
    )


def test_calibration_importer_run_kind_matches_real_managed_manifest(
    tmp_path: Path,
) -> None:
    config = tmp_path / "configs" / "calibration.yaml"
    config.parent.mkdir()
    config.write_text("calibration: true\n", encoding="utf-8")
    reservation = artifacts.reserve_run(
        project_root=tmp_path,
        run_kind="phase_effective_entry_calibration",
        config_paths=(config,),
        seed=1002,
        environment_count=1,
        training_stage=effective_entry_module.CALIBRATION_TRAINING_STAGE,
        git_commit="a" * 40,
        entrypoint="wlr50_clean.ppo.cli",
        subcommand="phase-snapshot-live-probe",
    )
    manifest = json.loads(reservation.started_manifest.read_text(encoding="utf-8"))

    assert reservation.run_dir.parent.name == "phase-effective-entry-calibration"
    assert manifest["run_kind"] == effective_entry_module.CALIBRATION_RUN_KIND
    assert manifest["run_kind"] != "phase_effective_entry_calibration"


def test_builder_rejects_out_of_order_phase_artifacts(
    snapshot_bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dirs = tuple(tmp_path / phase for phase in PHASE_IDS)
    swapped = dict(zip(run_dirs, PHASE_IDS, strict=True))
    swapped[run_dirs[0]], swapped[run_dirs[1]] = PHASE_IDS[1], PHASE_IDS[0]

    def capture(directory, **_kwargs):
        phase = swapped[Path(directory)]
        return (
            phase,
            {"phase": phase},
            {
                "phase": phase,
                "source_git_commit": "a" * 40,
                "runtime_content_sha256": "b" * 64,
                "identity_config_sha256": "c" * 64,
            },
        )

    monkeypatch.setattr(effective_entry_module, "_capture_calibration_run", capture)
    with pytest.raises(EffectivePhaseEntryError, match="ordered P02-P13"):
        build_effective_phase_entry_contract(
            run_dirs,
            tmp_path / "derived.json",
            snapshot_bundle=snapshot_bundle,
        )


def test_p01_is_explicitly_outside_the_effective_entry_contract(contract) -> None:
    with pytest.raises(EffectivePhaseEntryError, match="P01"):
        contract.entry("P01")


@pytest.mark.parametrize("tamper", ("fingerprint", "contact", "entry_sha256"))
def test_entry_copy_tamper_cannot_mutate_pinned_contract(
    contract, tamper: str
) -> None:
    phase = "P02"
    original = contract.entry(phase)
    candidate = contract.entry(phase)
    if tamper == "fingerprint":
        candidate["post_prime_fingerprint"][FINGERPRINT_FIELDS[0]] += 1.0
    elif tamper == "contact":
        candidate["raw_contacts"]["front_left_ankle"]["ground"]["active"] = not (
            candidate["raw_contacts"]["front_left_ankle"]["ground"]["active"]
        )
    else:
        candidate["entry_sha256"] = "0" * 64

    assert contract.entry(phase) == original


@pytest.mark.parametrize("tamper", ("fingerprint", "contact", "entry_sha256"))
def test_internal_entry_tree_rejects_in_place_tamper(contract, tamper: str) -> None:
    internal = contract.entries[0][1]
    with pytest.raises(TypeError):
        if tamper == "fingerprint":
            internal["post_prime_fingerprint"][FINGERPRINT_FIELDS[0]] = 1.0
        elif tamper == "contact":
            internal["raw_contacts"]["front_left_ankle"]["ground"]["active"] = False
        else:
            internal["entry_sha256"] = "0" * 64


def test_capture_records_every_file_ancestor_surface(tmp_path: Path) -> None:
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    source = nested / "contract.bin"
    source.write_bytes(b"pinned")

    payloads, identities = effective_entry_module._capture_paths_once(
        {"contract": source}
    )

    assert payloads == {"contract": b"pinned"}
    directory_paths = {
        Path(identity[0]) for identity in identities if identity[1] == "directory"
    }
    assert nested.resolve() in directory_paths
    assert tmp_path.resolve() in directory_paths


def test_capture_reads_each_file_through_one_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    original_open = effective_entry_module.os.open
    opened_flags = []

    def tracked_open(path, flags):
        opened_flags.append(flags)
        return original_open(path, flags)

    def forbidden_read_bytes(_path):
        raise AssertionError("Path.read_bytes must not reopen a pinned file")

    monkeypatch.setattr(effective_entry_module.os, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    payloads, _ = effective_entry_module._capture_paths_once({"contract": source})

    assert payloads == {"contract": b"pinned"}
    assert len(opened_flags) == 1
    no_follow = int(getattr(effective_entry_module.os, "O_NOFOLLOW", 0))
    if no_follow:
        assert opened_flags[0] & no_follow


def test_capture_rejects_symlink_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "real.bin"
    target.write_bytes(b"pinned")
    linked = tmp_path / "linked.bin"
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(EffectivePhaseEntryError, match="symlink|reparse|redirect"):
        effective_entry_module._capture_paths_once({"contract": linked})


def test_capture_rejects_open_handle_path_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    original = effective_entry_module._handle_identity

    def mismatched_handle(descriptor, path, *, label, directory):
        identity = list(
            original(descriptor, path, label=label, directory=directory)
        )
        identity[3] += 1
        return tuple(identity)

    monkeypatch.setattr(
        effective_entry_module, "_handle_identity", mismatched_handle
    )
    with pytest.raises(EffectivePhaseEntryError, match="opened contract differs"):
        effective_entry_module._capture_paths_once({"contract": source})


def test_capture_rejects_visible_ancestor_aba_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    target = tmp_path.resolve()
    original = effective_entry_module._path_identity
    calls = 0

    def changed_ancestor(path, *, label, directory):
        nonlocal calls
        identity = original(path, label=label, directory=directory)
        if path == target and directory:
            calls += 1
            if calls > 1:
                changed = list(identity)
                changed[3] += 1
                return tuple(changed)
        return identity

    monkeypatch.setattr(
        effective_entry_module, "_path_identity", changed_ancestor
    )
    with pytest.raises(EffectivePhaseEntryError, match="captured path changed"):
        effective_entry_module._capture_paths_once({"contract": source})


def test_capture_allows_unrelated_directory_metadata_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.bin"
    source.write_bytes(b"pinned")
    target = tmp_path.resolve()
    original = effective_entry_module._path_identity
    calls = 0

    def changed_directory_metadata(path, *, label, directory):
        nonlocal calls
        identity = original(path, label=label, directory=directory)
        if path == target and directory:
            calls += 1
            if calls > 1:
                changed = list(identity)
                changed[4] += 1
                changed[5] += 1
                changed[6] += 1
                changed[9] += 1
                return tuple(changed)
        return identity

    monkeypatch.setattr(
        effective_entry_module, "_path_identity", changed_directory_metadata
    )
    payloads, _ = effective_entry_module._capture_paths_once({"contract": source})

    assert payloads == {"contract": b"pinned"}


def test_fingerprint_exact_and_one_ulp_pass_but_two_ulp_fails(contract) -> None:
    phase = "P02"
    entry = contract.entry(phase)
    exact = _comparison(entry)
    proof = validate_effective_phase_entry_comparison(contract, phase, exact)
    assert proof["verified"] is True
    assert set(proof["fingerprint"]) == set(FINGERPRINT_FIELDS)

    field = FINGERPRINT_FIELDS[0]
    one_ulp = _comparison(entry)
    one_ulp["maximum_errors"][field] = _nextafter(
        entry["post_prime_fingerprint"][field], 1
    )
    assert binary64_ulp_distance(
        one_ulp["maximum_errors"][field],
        entry["post_prime_fingerprint"][field],
    ) == 1
    assert validate_effective_phase_entry_comparison(
        contract, phase, one_ulp
    )["verified"] is True

    two_ulp = _comparison(entry)
    two_ulp["maximum_errors"][field] = _nextafter(
        entry["post_prime_fingerprint"][field], 2
    )
    with pytest.raises(EffectivePhaseEntryError, match="2 ULP"):
        validate_effective_phase_entry_comparison(contract, phase, two_ulp)


def test_fingerprint_nan_is_rejected(contract) -> None:
    comparison = _comparison(contract.entry("P03"))
    comparison["maximum_errors"][FINGERPRINT_FIELDS[2]] = math.nan
    with pytest.raises(EffectivePhaseEntryError, match="finite"):
        validate_effective_phase_entry_comparison(contract, "P03", comparison)


def test_component_state_exact_and_one_ulp_pass_but_two_ulp_fails(contract) -> None:
    phase = "P02"
    entry = contract.entry(phase)
    exact = _comparison(entry)
    proof = validate_effective_phase_entry_comparison(contract, phase, exact)
    assert proof["schema"] == "wlr50_clean.ppo_phase_effective_entry_live_proof.v2"
    assert proof["component_state_max_ulp_distance"] == 0
    assert len(proof["component_state_ulp_distance"]) == 57
    assert proof["component_state_sha256"] == proof[
        "expected_component_state_sha256"
    ]

    field = "root_position_w_m"
    one_ulp = _comparison(entry)
    one_ulp["effective_component_state"][field][0] = _nextafter(
        entry["effective_component_state"][field][0], 1
    )
    _rehash_component_state(one_ulp["effective_component_state"])
    one_ulp_proof = validate_effective_phase_entry_comparison(
        contract, phase, one_ulp
    )
    assert one_ulp_proof["component_state_max_ulp_distance"] == 1
    assert one_ulp_proof["component_state_ulp_distance"][f"{field}[0]"] == 1

    two_ulp = _comparison(entry)
    two_ulp["effective_component_state"][field][0] = _nextafter(
        entry["effective_component_state"][field][0], 2
    )
    _rehash_component_state(two_ulp["effective_component_state"])
    with pytest.raises(EffectivePhaseEntryError, match="component state is 2 ULP"):
        validate_effective_phase_entry_comparison(contract, phase, two_ulp)


@pytest.mark.parametrize("tamper", ("sha256", "layout", "orientation", "nan"))
def test_component_state_tamper_fails_closed(contract, tamper: str) -> None:
    comparison = _comparison(contract.entry("P03"))
    state = comparison["effective_component_state"]
    if tamper == "sha256":
        state["sha256"] = "0" * 64
    elif tamper == "layout":
        state.pop("wheel_bottoms_w_m")
        _rehash_component_state(state)
    elif tamper == "orientation":
        state["root_orientation_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
        _rehash_component_state(state)
    else:
        state["servo_logical_velocity_deg_s"][3] = "nan"
        _rehash_component_state(state)
    with pytest.raises(EffectivePhaseEntryError):
        validate_effective_phase_entry_comparison(contract, "P03", comparison)


def test_live_force_values_are_diagnostic_but_threshold_class_is_hard(contract) -> None:
    phase = "P02"
    comparison = _comparison(contract.entry(phase))
    active_pair = next(
        (wheel, pair_name)
        for wheel, pairs in comparison["raw_physx_contacts"]["pairs"].items()
        for pair_name, pair in pairs.items()
        if math.sqrt(sum(value * value for value in pair["force_w_n"])) >= 0.25
    )
    wheel, pair_name = active_pair
    comparison["raw_physx_contacts"]["pairs"][wheel][pair_name]["force_w_n"] = [
        0.0,
        0.0,
        0.25,
    ]
    assert validate_effective_phase_entry_comparison(
        contract, phase, comparison
    )["verified"] is True

    comparison["raw_physx_contacts"]["pairs"][wheel][pair_name]["force_w_n"] = [
        0.0,
        0.0,
        0.249,
    ]
    with pytest.raises(EffectivePhaseEntryError, match="raw contact contract"):
        validate_effective_phase_entry_comparison(contract, phase, comparison)


@pytest.mark.parametrize("tamper", ["source", "pair_verified", "classifier", "double"])
def test_contact_tamper_fails_closed(contract, tamper: str) -> None:
    phase = "P02"
    comparison = _comparison(contract.entry(phase))
    wheel = next(iter(comparison["raw_physx_contacts"]["pairs"]))
    pairs = comparison["raw_physx_contacts"]["pairs"][wheel]
    if tamper == "source":
        pairs["ground"]["source"] = CONTACT_SOURCE + ".tampered"
    elif tamper == "pair_verified":
        pairs["ground"]["pair_verified"] = False
    elif tamper == "classifier":
        comparison["exact_contacts"][wheel]["actual_ground_active"] = not (
            comparison["exact_contacts"][wheel]["actual_ground_active"]
        )
    else:
        pairs["ground"]["force_w_n"] = [0.0, 0.0, 1.0]
        pairs["obstacle"]["force_w_n"] = [0.0, 0.0, 1.0]
        comparison["exact_contacts"][wheel].update(
            {
                "actual_class": "GROUND_AND_OBSTACLE",
                "actual_ground_active": True,
                "actual_obstacle_active": True,
            }
        )
    with pytest.raises(EffectivePhaseEntryError):
        validate_effective_phase_entry_comparison(contract, phase, comparison)


def test_loader_rejects_sidecar_tamper(snapshot_bundle, tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract_copy(contract_path, DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_bytes())
    contract_path.with_suffix(".sha256").write_bytes(b"0" * 64 + b"  contract.json\n")
    with pytest.raises(EffectivePhaseEntryError, match="sidecar mismatch"):
        capture_validated_effective_phase_entry_contract(
            contract_path, expected_snapshot_bundle=snapshot_bundle
        )


@pytest.mark.parametrize("kind", ["duplicate", "nan"])
def test_loader_rejects_duplicate_keys_and_nonfinite_json(
    snapshot_bundle, tmp_path: Path, kind: str
) -> None:
    payload = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_bytes()
    if kind == "duplicate":
        payload = payload.replace(
            b'{\n  "schema":',
            b'{\n  "schema": "duplicate",\n  "schema":',
            1,
        )
        expected = "duplicate JSON key"
    else:
        payload = payload.replace(
            b'  "fingerprint_max_ulp_distance": 1,',
            b'  "fingerprint_max_ulp_distance": NaN,',
            1,
        )
        expected = "non-finite JSON constant"
    contract_path = tmp_path / "contract.json"
    _write_contract_copy(contract_path, payload)
    with pytest.raises(EffectivePhaseEntryError, match=expected):
        capture_validated_effective_phase_entry_contract(
            contract_path, expected_snapshot_bundle=snapshot_bundle
        )
