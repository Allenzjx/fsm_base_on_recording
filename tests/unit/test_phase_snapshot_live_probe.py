from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo import phase_effective_entry as effective_entry_subject
from wlr50_clean.ppo import phase_snapshot_live_probe as probe_subject
from wlr50_clean.ppo.phase_snapshot_live_probe import (
    ATTEMPTS_PER_PHASE,
    PROBE_PHASES,
    PhaseSnapshotLiveProbeError,
    _attempt_passed,
    observation_diagnostics,
)
from wlr50_clean.ppo.phase_snapshots import (
    SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS,
    SOURCE_ACK_REPLAY_INVARIANT_FIELDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(phase: str = "P10") -> dict[str, object]:
    return json.loads(
        (
            PROJECT_ROOT
            / "reference"
            / "ppo_phase_snapshots"
            / phase
            / "snapshot.json"
        ).read_text(encoding="utf-8")
    )


def _pair(active: bool) -> SimpleNamespace:
    return SimpleNamespace(
        active=active,
        pair_verified=True,
        normal_force_n=1.0 if active else 0.0,
        force_w_n=(1.0, 0.0, 0.0) if active else (0.0, 0.0, 0.0),
        active_history=(active, active, active),
    )


def _matching_observation(
    snapshot: dict[str, object], *, physics_tick: int = 0
) -> SimpleNamespace:
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER

    root = snapshot["root_state"]
    joint = snapshot["joint_state"]
    wheel = snapshot["wheel_state"]
    geometry = snapshot["obstacle_relative_geometry"]
    contact_state = snapshot["contact_state"]
    joints = {
        name: SimpleNamespace(
            position_deg=joint["logical_position_deg"][index],
            velocity_deg_s=joint["logical_velocity_deg_s"][index],
        )
        for index, name in enumerate(SERVO_ORDER)
    }
    wheels = {}
    contacts = {}
    for index, name in enumerate(WHEEL_ORDER):
        body_name = f"{name}_body"
        expected = contact_state[name]
        wheels[name] = SimpleNamespace(
            body_name=body_name,
            velocity_rad_s=wheel["logical_velocity_rad_s"][index],
            center_w_m=geometry["wheel_centers_w_m"][name],
            bottom_w_m=geometry["wheel_bottoms_w_m"][name],
        )
        contacts[body_name] = SimpleNamespace(
            contact_class=expected["class"],
            ground=_pair(bool(expected["ground_active"])),
            obstacle=_pair(bool(expected["obstacle_active"])),
        )
    return SimpleNamespace(
        physics_tick=physics_tick,
        simulation_time_s=physics_tick / 120.0,
        base=SimpleNamespace(
            position_w_m=root["position_w_m"],
            orientation_wxyz=root["orientation_wxyz"],
            linear_velocity_w_m_s=root["linear_velocity_w_m_s"],
            angular_velocity_w_rad_s=root["angular_velocity_w_rad_s"],
        ),
        joints=joints,
        wheels=wheels,
        contacts=contacts,
    )


def _replay_window(snapshot: dict[str, object]):
    return probe_subject._validated_replay_window(
        snapshot,
        SimpleNamespace(
            source_tick=snapshot["source_tick"],
            source_replay_steps=snapshot["source_replay_steps"],
            target_entry_tick=snapshot.get("target_entry_tick"),
            predecessor_verify_tick=snapshot.get("predecessor_verify_tick"),
            predecessor_verify_time_s=snapshot.get("predecessor_verify_time_s"),
            controller_anchor_tick=snapshot.get("controller_anchor_tick"),
            controller_anchor_time_s=snapshot.get("controller_anchor_time_s"),
        ),
        phase=str(snapshot["fsm_state"]),
    )


def _synthetic_component_state(phase: str) -> tuple[dict, dict]:
    phase_number = int(phase[1:])
    offset = phase_number / 100.0
    state = {
        "schema": effective_entry_subject.COMPONENT_STATE_SCHEMA,
        "units": dict(effective_entry_subject.COMPONENT_UNITS),
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
        "servo_order": list(effective_entry_subject.SERVO_ORDER),
        "wheel_order": list(effective_entry_subject.WHEEL_ORDER),
        "wheel_centers_w_m": {
            wheel: [offset + index, -offset, 0.1 + index / 100.0]
            for index, wheel in enumerate(effective_entry_subject.WHEEL_ORDER)
        },
        "wheel_bottoms_w_m": {
            wheel: [offset + index, -offset, 0.05 + index / 100.0]
            for index, wheel in enumerate(effective_entry_subject.WHEEL_ORDER)
        },
    }
    state["sha256"] = hashlib.sha256(
        effective_entry_subject._canonical_bytes(state)
    ).hexdigest()
    return effective_entry_subject._component_state(state)


def _adaptive_source_replay_evidence(
    snapshot: dict[str, object],
    command: dict[str, object],
    index: int,
    previous_post: str | None,
) -> tuple[dict, dict, str]:
    tick = command["control_physics_tick"]
    feedback = {
        "schema": "wlr50_clean.phase_snapshot_live_servo_feedback.v1",
        "canonical_servo_order": list(effective_entry_subject.SERVO_ORDER),
        "unit": "rad",
        "physical_position_rad": [tick / 10000.0 + i / 1000.0 for i in range(8)],
        "tensor_dtype": "torch.float64",
        "tensor_device": "cpu",
    }
    feedback["sha256"] = hashlib.sha256(
        effective_entry_subject._canonical_bytes(feedback)
    ).hexdigest()
    source_pre_sha = hashlib.sha256(
        effective_entry_subject._canonical_bytes(command["mapper_pre_state"])
    ).hexdigest()
    source_post_sha = hashlib.sha256(
        effective_entry_subject._canonical_bytes(command["mapper_post_state"])
    ).hexdigest()
    live_pre_sha = source_pre_sha if previous_post is None else previous_post
    live_post_sha = hashlib.sha256(
        effective_entry_subject._canonical_bytes(
            {
                "schema": "test.live_mapper_post.v1",
                "source_control_physics_tick": tick,
                "live_pre_state_sha256": live_pre_sha,
                "feedback_input_sha256": feedback["sha256"],
            }
        )
    ).hexdigest()
    feedback_output = {
        name: copy.deepcopy(command["expected_atomic_ack"][name])
        for name in SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS
    }
    feedback_output_sha = hashlib.sha256(
        effective_entry_subject._canonical_bytes(feedback_output)
    ).hexdigest()
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
        "source_command_file_sha256": snapshot["source_artifacts"]["command"][
            "sha256"
        ],
        "source_observation_file_sha256": snapshot["source_artifacts"][
            "observation"
        ]["sha256"],
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
        "replayed_drive_target_full12_sha256": feedback_output_sha,
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


def _replay_proof(snapshot: dict[str, object]) -> dict[str, object]:
    commands = snapshot["source_commands"]
    steps = snapshot["source_replay_steps"]
    target = snapshot["source_tick"] + steps
    anchor_contract = effective_entry_subject._expected_replay_anchor_contract(
        snapshot,
        str(snapshot["fsm_state"]),
        replay_steps=steps,
        target_entry_tick=target,
        control_ticks=tuple(
            command["control_physics_tick"] for command in commands
        ),
        predecessor_verify_tick=snapshot.get("predecessor_verify_tick"),
        predecessor_verify_time_s=snapshot.get("predecessor_verify_time_s"),
        controller_anchor_tick=snapshot.get("controller_anchor_tick"),
        controller_anchor_time_s=snapshot.get("controller_anchor_time_s"),
    )
    matches = []
    mapper_states = []
    previous_live_post = None
    for index, command in enumerate(commands):
        actuation, mapper, previous_live_post = _adaptive_source_replay_evidence(
            snapshot, command, index, previous_live_post
        )
        matches.append(actuation)
        mapper_states.append(mapper)
    return {
        "source_replay_steps": steps,
        "physical_anchor_tick": anchor_contract["physical_anchor_tick"],
        "physical_anchor_time_s": anchor_contract["physical_anchor_time_s"],
        "predecessor_verify_tick": snapshot.get("predecessor_verify_tick"),
        "predecessor_verify_time_s": snapshot.get("predecessor_verify_time_s"),
        "controller_anchor_tick": snapshot.get("controller_anchor_tick"),
        "controller_anchor_time_s": snapshot.get("controller_anchor_time_s"),
        "target_entry_tick": target,
        "target_entry_time_s": anchor_contract["target_entry_time_s"],
        "physical_to_predecessor_verify_replay_steps": anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ],
        "predecessor_verify_to_controller_replay_steps": anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ],
        "physical_to_controller_replay_steps": anchor_contract[
            "physical_to_controller_replay_steps"
        ],
        "controller_to_target_replay_steps": anchor_contract[
            "controller_to_target_replay_steps"
        ],
        "hybrid_physical_controller_anchor": anchor_contract[
            "hybrid_physical_controller_anchor"
        ],
        "replay_anchor_contract": anchor_contract,
        "source_replay_fsm_contexts": anchor_contract[
            "source_replay_fsm_contexts"
        ],
        "episode_sensor_tick_offset": target,
        "effective_entry_offset_s": steps / 120.0,
        "prime_atomic_full12_writes": steps,
        "prime_atomic_writes": [
            {
                "physics_tick": 180 + index,
                "write_count": 181 + index,
                "source_control_physics_tick": command["control_physics_tick"],
                "observation_physics_tick": command["control_physics_tick"] + 1,
                "articulation_writes_this_call": 1,
                "source_actuation_match": matches[index],
                "source_mapper_post_state": mapper_states[index],
            }
            for index, command in enumerate(commands)
        ],
        "source_actuation_matches": matches,
        "source_actuation_match": matches[-1],
        "source_mapper_post_states": mapper_states,
        "source_mapper_post_state": mapper_states[-1],
        "source_adapter_input_sha256s": [
            command["source_adapter_input_sha256"] for command in commands
        ],
        "all_source_adapter_inputs_hash_matched": True,
        "all_live_output_contracts_verified": True,
        "all_live_mapper_transitions_verified": True,
        "live_feedback_adaptive_replay": True,
        "historical_feedback_equivalence_claimed": False,
        "initial_mapper_restore_count": 1,
        "per_tick_mapper_restore_count": 0,
        "source_replay_observation_ticks": list(
            range(snapshot["source_tick"] + 1, target + 1)
        ),
        "source_replay_guard_updates_applied": steps,
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
            for command in commands
        ],
        "all_source_replay_steps_safe": True,
    }


def test_probe_covers_every_non_p01_phase_twice() -> None:
    assert PROBE_PHASES == tuple(f"P{index:02d}" for index in range(2, 14))
    assert ATTEMPTS_PER_PHASE == 2


def test_replay_window_is_derived_from_snapshot_payload_and_manifest() -> None:
    p10 = _snapshot("P10")
    p10_window = _replay_window(p10)
    assert p10_window.source_tick == 7552
    assert p10_window.predecessor_verify_tick == 7776
    assert p10_window.predecessor_verify_time_s == 7776 / 120
    assert p10_window.controller_anchor_tick == 7784
    assert p10_window.controller_anchor_time_s == 7784 / 120
    assert p10_window.target_entry_tick == 7794
    assert p10_window.source_replay_steps == 242
    assert p10_window.predecessor_verify_tick - p10_window.source_tick == 224
    assert (
        p10_window.controller_anchor_tick
        - p10_window.predecessor_verify_tick
        == 8
    )
    assert p10_window.target_entry_tick - p10_window.controller_anchor_tick == 10
    assert p10_window.target_entry_tick - p10_window.source_tick == 242
    assert p10_window.control_ticks == tuple(
        range(p10_window.source_tick, p10_window.target_entry_tick)
    )
    assert tuple(
        (
            command["source_fsm_state"],
            command["source_fsm_lifecycle"],
        )
        for command in p10["source_commands"]
    ) == (
        (("P09", "EXECUTE_MOTION"),) * 224
        + (("P09", "VERIFY_RESULT"),) * 8
        + (("P10", "WAIT_ENTRY"),) * 10
    )
    assert _replay_proof(p10)["source_replay_observation_ticks"] == list(
        range(p10_window.source_tick + 1, p10_window.target_entry_tick + 1)
    )

    p02 = _snapshot("P02")
    p02_window = _replay_window(p02)
    assert p02_window.predecessor_verify_tick is None
    assert p02_window.predecessor_verify_time_s is None
    assert p02_window.controller_anchor_tick is None
    assert p02_window.controller_anchor_time_s is None
    assert p02_window.source_replay_steps == 1
    assert p02_window.target_entry_tick == p02_window.source_tick + 1

    nonhybrid_p10 = copy.deepcopy(p02)
    nonhybrid_p10["fsm_state"] = "P10"
    nonhybrid_p10_window = _replay_window(nonhybrid_p10)
    assert nonhybrid_p10_window.source_replay_steps == 1
    assert nonhybrid_p10_window.predecessor_verify_tick is None
    assert nonhybrid_p10_window.predecessor_verify_time_s is None
    assert nonhybrid_p10_window.controller_anchor_tick is None
    assert nonhybrid_p10_window.controller_anchor_time_s is None

    with pytest.raises(PhaseSnapshotLiveProbeError, match="manifest replay binding"):
        probe_subject._validated_replay_window(
            p10,
            SimpleNamespace(
                source_tick=p10["source_tick"],
                source_replay_steps=1,
                target_entry_tick=p10["target_entry_tick"],
                predecessor_verify_tick=p10["predecessor_verify_tick"],
                predecessor_verify_time_s=p10["predecessor_verify_time_s"],
                controller_anchor_tick=p10["controller_anchor_tick"],
                controller_anchor_time_s=p10["controller_anchor_time_s"],
            ),
            phase="P10",
        )

    with pytest.raises(PhaseSnapshotLiveProbeError, match="hybrid-anchor"):
        probe_subject._validated_replay_window(
            p10,
            SimpleNamespace(
                source_tick=p10["source_tick"],
                source_replay_steps=p10["source_replay_steps"],
                target_entry_tick=p10["target_entry_tick"],
                predecessor_verify_tick=p10["predecessor_verify_tick"],
                predecessor_verify_time_s=p10["predecessor_verify_time_s"],
                controller_anchor_tick=p10["controller_anchor_tick"] + 1,
                controller_anchor_time_s=p10["controller_anchor_time_s"],
            ),
            phase="P10",
        )

    with pytest.raises(PhaseSnapshotLiveProbeError, match="hybrid-anchor"):
        probe_subject._validated_replay_window(
            p10,
            SimpleNamespace(
                source_tick=p10["source_tick"],
                source_replay_steps=p10["source_replay_steps"],
                target_entry_tick=p10["target_entry_tick"],
                predecessor_verify_tick=p10["predecessor_verify_tick"] + 1,
                predecessor_verify_time_s=p10["predecessor_verify_time_s"],
                controller_anchor_tick=p10["controller_anchor_tick"],
                controller_anchor_time_s=p10["controller_anchor_time_s"],
            ),
            phase="P10",
        )

    malformed_p10 = copy.deepcopy(p10)
    malformed_p10["controller_anchor_time_s"] += 1.0 / 120.0
    with pytest.raises(PhaseSnapshotLiveProbeError, match="hybrid-anchor"):
        _replay_window(malformed_p10)

    malformed_predecessor = copy.deepcopy(p10)
    malformed_predecessor["predecessor_verify_time_s"] += 1.0 / 120.0
    with pytest.raises(PhaseSnapshotLiveProbeError, match="hybrid-anchor"):
        _replay_window(malformed_predecessor)

    malformed_context = copy.deepcopy(p10)
    malformed_context["source_commands"][0]["source_fsm_state"] = "P10"
    malformed_context["source_command"] = malformed_context["source_commands"][0]
    with pytest.raises(
        PhaseSnapshotLiveProbeError, match="three-segment replay contexts"
    ):
        _replay_window(malformed_context)

    malformed_p02 = copy.deepcopy(p02)
    malformed_p02["controller_anchor_tick"] = p02["source_tick"]
    malformed_p02["controller_anchor_time_s"] = p02["source_tick"] / 120.0
    malformed_p02["predecessor_verify_tick"] = p02["source_tick"]
    malformed_p02["predecessor_verify_time_s"] = p02["source_tick"] / 120.0
    with pytest.raises(
        PhaseSnapshotLiveProbeError, match="unexpectedly declares"
    ):
        _replay_window(malformed_p02)

    malformed_p02_target = copy.deepcopy(p02)
    malformed_p02_target["target_entry_tick"] = p02["source_tick"] + 1
    with pytest.raises(
        PhaseSnapshotLiveProbeError, match="only P10 may declare"
    ):
        _replay_window(malformed_p02_target)


def test_observation_diagnostics_accepts_only_exact_contacts_and_state() -> None:
    snapshot = _snapshot()
    observation = _matching_observation(snapshot)
    result = observation_diagnostics(observation, snapshot)
    assert result["physical_state_within_production_tolerances"] is True
    assert result["exact_contacts_match"] is True
    assert result["contact_mismatches"] == []

    first_wheel = next(iter(observation.wheels.values()))
    observation.contacts[first_wheel.body_name].contact_class = "AIR"
    observation.contacts[first_wheel.body_name].ground = _pair(False)
    observation.contacts[first_wheel.body_name].obstacle = _pair(False)
    failed = observation_diagnostics(observation, snapshot)
    assert failed["physical_state_within_production_tolerances"] is True
    assert failed["exact_contacts_match"] is False
    assert failed["contact_mismatches"]


def test_attempt_gate_fails_closed_on_exception_contact_or_extra_step() -> None:
    snapshot = _snapshot()
    replay_window = _replay_window(snapshot)
    replay_proof = _replay_proof(snapshot)
    anchor_contract = replay_proof["replay_anchor_contract"]
    commands = snapshot["source_commands"]
    replay_steps = snapshot["source_replay_steps"]
    diagnostic = observation_diagnostics(
        _matching_observation(
            snapshot, physics_tick=replay_window.target_entry_tick
        ),
        snapshot,
    )
    component_state, component_binary = _synthetic_component_state("P10")
    component_ulp = {
        label: 0
        for label, _ in effective_entry_subject._component_state_scalar_items(
            component_state
        )
    }
    row = {
        "phase": "P10",
        "source_tick": snapshot["source_tick"],
        "predecessor_verify_tick": replay_window.predecessor_verify_tick,
        "predecessor_verify_time_s": replay_window.predecessor_verify_time_s,
        "controller_anchor_tick": replay_window.controller_anchor_tick,
        "controller_anchor_time_s": replay_window.controller_anchor_time_s,
        "target_entry_tick": replay_window.target_entry_tick,
        "episode_sensor_tick_offset": replay_window.target_entry_tick,
        "source_replay_steps": replay_steps,
        "effective_entry_offset_s": replay_steps / 120.0,
        "source_control_physics_ticks": list(replay_window.control_ticks),
        "source_command_row_canonical_sha256s": [
            command["source_command_row_canonical_sha256"] for command in commands
        ],
        "source_observation_row_canonical_sha256s": [
            command["source_observation_row_canonical_sha256"]
            for command in commands
        ],
        "source_adapter_input_sha256s": [
            command["source_adapter_input_sha256"] for command in commands
        ],
        "source_drive_target_full12_sha256s": [
            command["drive_target_full12_sha256"] for command in commands
        ],
        "source_actuation_contract_sha256s": [
            command["actuation_contract_sha256"] for command in commands
        ],
        "reset_completed": True,
        "physics_steps_during_reset": 180 + replay_steps,
        "extra_physics_priming_steps": replay_steps,
        "post_prime_contact_sensor_read_count": replay_steps,
        "snapshot_state_write": {
            **replay_proof,
            "root_pose_writes": 1,
            "root_velocity_writes": 1,
            "joint_state_writes": 1,
            "simulation_forward_syncs": 1,
            "pre_prime_state_verified": True,
            "pre_prime_joint_state_verified": True,
            "pre_prime_root_link_readback": {
                "verified": True,
                "all_values_finite": True,
                "all_fields_within_production_tolerances": True,
                "physics_steps_before_readback": 0,
                "contact_sensor_reads_before_readback": 0,
            },
            "physics_steps": replay_steps,
            "state_write_count": 1,
            "post_prime_state_rewrite_performed": False,
            "contact_and_state_share_solver_tick": True,
            "prime_physics_steps": replay_steps,
            "logical_target_fallback_used": False,
            "current_contact_force_provenance": "current_final_solver_force_only",
            "sensor_history_samples_after_reset": replay_steps,
            "contact_sensor_reads_after_prime": replay_steps,
            "classifier_cold_started_before_source_replay": True,
            "classifier_source_history_restored": False,
            "classifier_source_state_restored": False,
            "classifier_history_equivalence_claimed": False,
            "raw_sensor_history_rewarmed_from_prime": True,
            "contact_backend_reset": True,
            "contact_backend_reset_after_prime": False,
            "fsm_clock_steps_during_priming": 0,
            "episode_clock_steps_during_priming": 0,
            "effective_entry_contract": {
                "schema": "wlr50_clean.ppo_phase_effective_entry_live_proof.v2",
                "phase": "P10",
                "effective_entry_semantics": (
                    "source_snapshot_plus_validated_replay_steps_no_rewind"
                ),
                "source_tick": replay_window.source_tick,
                "physical_anchor_tick": anchor_contract["physical_anchor_tick"],
                "physical_anchor_time_s": anchor_contract[
                    "physical_anchor_time_s"
                ],
                "predecessor_verify_tick": (
                    replay_window.predecessor_verify_tick
                ),
                "predecessor_verify_time_s": (
                    replay_window.predecessor_verify_time_s
                ),
                "controller_anchor_tick": replay_window.controller_anchor_tick,
                "controller_anchor_time_s": (
                    replay_window.controller_anchor_time_s
                ),
                "target_entry_tick": replay_window.target_entry_tick,
                "target_entry_time_s": anchor_contract["target_entry_time_s"],
                "source_replay_steps": replay_steps,
                "physical_to_predecessor_verify_replay_steps": anchor_contract[
                    "physical_to_predecessor_verify_replay_steps"
                ],
                "predecessor_verify_to_controller_replay_steps": anchor_contract[
                    "predecessor_verify_to_controller_replay_steps"
                ],
                "physical_to_controller_replay_steps": anchor_contract[
                    "physical_to_controller_replay_steps"
                ],
                "controller_to_target_replay_steps": anchor_contract[
                    "controller_to_target_replay_steps"
                ],
                "hybrid_physical_controller_anchor": anchor_contract[
                    "hybrid_physical_controller_anchor"
                ],
                "replay_anchor_contract": anchor_contract,
                "effective_entry_offset_s": replay_steps / 120.0,
                "contract_sha256": "1" * 64,
                "entry_sha256": "2" * 64,
                "fingerprint_max_ulp_distance": 1,
                "fingerprint": {},
                "component_state_allowed_max_ulp_distance": 1,
                "component_state_max_ulp_distance": 0,
                "component_state_ulp_distance": component_ulp,
                "component_state": component_state,
                "component_state_binary64_hex": component_binary,
                "component_state_sha256": component_state["sha256"],
                "expected_component_state_sha256": component_state["sha256"],
                "raw_contacts": {},
                "raw_contact_signature_sha256": "3" * 64,
                "expected_raw_contact_signature_sha256": "3" * 64,
                "verified": True,
                "failures": [],
            },
            "entry_safety_contract": {
                "schema": "wlr50_clean.phase_effective_entry_safety.v1",
                "verified": True,
                "all_failure_flags_false": True,
                "flags": {
                    "body_collision": False,
                    "wheel_only_climb": False,
                    "safety_abort": False,
                },
            },
            "entry_guard_contract": {
                "schema": "wlr50_clean.phase_effective_entry_controller.v1",
                "verified": True,
                "phase": "P10",
                "lifecycle": "EXECUTE_MOTION",
                "nonterminal": True,
                "unblocked": True,
                "p10_signed_velocity_alignment": {
                    "signed_positive_rebound_required": True,
                    "actual_deg_s": 1.0,
                },
            },
            "priming_observation": {
                "raw_physx_contact_sources_verified": True,
                "current_raw_force_hysteresis_contract_matches_snapshot": True,
            },
        },
        "observation_diagnostics": diagnostic,
        "clocks": {
            "backend_episode_tick": 0,
            "controller_frame_state_id": "P10",
            "controller_frame_physics_tick": 0,
        },
    }
    assert _attempt_passed(row, replay_window=replay_window) is True
    calibration = copy.deepcopy(row)
    comparison = {
        "schema": "wlr50_clean.phase_snapshot_live_comparison.v2",
        "maximum_errors": {"root_position_m": 0.001},
        "effective_component_state": component_state,
    }
    calibration["snapshot_state_write"]["effective_entry_contract"] = {
        "schema": (
            "wlr50_clean.ppo_phase_effective_entry_calibration_live_proof.v2"
        ),
        "artifact_role": "CALIBRATION_ONLY_NOT_TRAINING_ACCEPTANCE",
        "verified": True,
        "calibration_only": True,
        "phase": "P10",
        "source_tick": replay_window.source_tick,
        "physical_anchor_tick": anchor_contract["physical_anchor_tick"],
        "physical_anchor_time_s": anchor_contract["physical_anchor_time_s"],
        "predecessor_verify_tick": replay_window.predecessor_verify_tick,
        "predecessor_verify_time_s": replay_window.predecessor_verify_time_s,
        "controller_anchor_tick": replay_window.controller_anchor_tick,
        "controller_anchor_time_s": replay_window.controller_anchor_time_s,
        "target_entry_tick": replay_window.target_entry_tick,
        "target_entry_time_s": anchor_contract["target_entry_time_s"],
        "source_replay_steps": replay_steps,
        "physical_to_predecessor_verify_replay_steps": anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ],
        "predecessor_verify_to_controller_replay_steps": anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ],
        "physical_to_controller_replay_steps": anchor_contract[
            "physical_to_controller_replay_steps"
        ],
        "controller_to_target_replay_steps": anchor_contract[
            "controller_to_target_replay_steps"
        ],
        "hybrid_physical_controller_anchor": anchor_contract[
            "hybrid_physical_controller_anchor"
        ],
        "replay_anchor_contract": anchor_contract,
        "effective_entry_offset_s": replay_steps / 120.0,
        "phase_snapshot_bundle_sha256": "c" * 64,
        "source_snapshot_post_prime_diagnostic": comparison,
        "failures": [],
    }
    assert _attempt_passed(
        calibration, replay_window=replay_window, calibration_mode=True
    ) is True
    assert _attempt_passed(calibration, replay_window=replay_window) is False
    assert _attempt_passed(
        row, replay_window=replay_window, calibration_mode=True
    ) is False
    extra_acceptance_field = copy.deepcopy(row)
    extra_acceptance_field["snapshot_state_write"]["effective_entry_contract"][
        "unexpected"
    ] = True
    assert _attempt_passed(
        extra_acceptance_field, replay_window=replay_window
    ) is False
    extra_calibration_field = copy.deepcopy(calibration)
    extra_calibration_field["snapshot_state_write"]["effective_entry_contract"][
        "unexpected"
    ] = True
    assert _attempt_passed(
        extra_calibration_field,
        replay_window=replay_window,
        calibration_mode=True,
    ) is False
    wrong_predecessor_anchor = copy.deepcopy(row)
    wrong_predecessor_anchor["predecessor_verify_tick"] += 1
    assert _attempt_passed(
        wrong_predecessor_anchor, replay_window=replay_window
    ) is False
    wrong_row_anchor = copy.deepcopy(row)
    wrong_row_anchor["controller_anchor_tick"] += 1
    assert _attempt_passed(
        wrong_row_anchor, replay_window=replay_window
    ) is False
    missing_effective_anchor = copy.deepcopy(row)
    del missing_effective_anchor["snapshot_state_write"][
        "effective_entry_contract"
    ]["controller_anchor_tick"]
    assert _attempt_passed(
        missing_effective_anchor, replay_window=replay_window
    ) is False
    missing_effective_predecessor = copy.deepcopy(row)
    del missing_effective_predecessor["snapshot_state_write"][
        "effective_entry_contract"
    ]["predecessor_verify_tick"]
    assert _attempt_passed(
        missing_effective_predecessor, replay_window=replay_window
    ) is False
    wrong_effective_anchor_time = copy.deepcopy(row)
    wrong_effective_anchor_time["snapshot_state_write"][
        "effective_entry_contract"
    ]["controller_anchor_time_s"] += 1.0 / 120.0
    assert _attempt_passed(
        wrong_effective_anchor_time, replay_window=replay_window
    ) is False
    restored_classifier = copy.deepcopy(row)
    restored_classifier["snapshot_state_write"][
        "classifier_cold_started_before_source_replay"
    ] = False
    assert _attempt_passed(
        restored_classifier, replay_window=replay_window
    ) is False

    assert _attempt_passed(
        {**row, "reset_completed": False}, replay_window=replay_window
    ) is False
    assert _attempt_passed(
        {**row, "physics_steps_during_reset": 181},
        replay_window=replay_window,
    ) is False
    assert _attempt_passed(
        {
            **row,
            "snapshot_state_write": {
                **row["snapshot_state_write"],
                "pre_prime_state_verified": False,
            },
        },
        replay_window=replay_window,
    ) is False
    assert _attempt_passed(
        {
            **row,
            "observation_diagnostics": {
                **diagnostic,
                "exact_contacts_match": False,
            },
        },
        replay_window=replay_window,
    ) is True
    # Every source command is an acceptance surface; a mismatch at any replay
    # step fails even when the final effective-entry proof otherwise passes.
    mismatched_replay = copy.deepcopy(row)
    mismatched_replay["snapshot_state_write"]["source_actuation_matches"][3][
        "all_replay_invariant_fields_match"
    ] = False
    assert _attempt_passed(
        mismatched_replay, replay_window=replay_window
    ) is False
    wrong_reset_physics_tick = copy.deepcopy(row)
    wrong_reset_physics_tick["snapshot_state_write"]["prime_atomic_writes"][3][
        "physics_tick"
    ] += 1
    assert _attempt_passed(
        wrong_reset_physics_tick, replay_window=replay_window
    ) is False
    wrong_reset_write_count = copy.deepcopy(row)
    wrong_reset_write_count["snapshot_state_write"]["prime_atomic_writes"][3][
        "write_count"
    ] += 1
    assert _attempt_passed(
        wrong_reset_write_count, replay_window=replay_window
    ) is False
    wrong_remap_claim = copy.deepcopy(row)
    wrong_remap_claim["snapshot_state_write"]["source_actuation_matches"][3][
        "clock_and_write_count_fields_intentionally_remapped"
    ] = False
    assert _attempt_passed(
        wrong_remap_claim, replay_window=replay_window
    ) is False
    unsafe_replay = copy.deepcopy(row)
    unsafe_replay["snapshot_state_write"]["source_replay_safety_checks"][4][
        "flags"
    ]["body_collision"] = True
    assert _attempt_passed(unsafe_replay, replay_window=replay_window) is False


def test_cli_and_wrapper_bind_probe_to_one_live_environment() -> None:
    arguments = cli._parser().parse_args(
        [
            "phase-snapshot-live-probe",
            "--run-dir",
            str(PROJECT_ROOT),
            "--seed",
            "1001",
            "--num-envs",
            "1",
        ]
    )
    assert arguments.command == "phase-snapshot-live-probe"
    assert arguments.phase_snapshot_prime_physics_steps == 1
    assert arguments.phase is None
    assert "phase-snapshot-live-probe" in cli.LIVE_COMMANDS

    selected = cli._parser().parse_args(
        [
            "phase-snapshot-live-probe",
            "--run-dir",
            str(PROJECT_ROOT),
            "--seed",
            "1001",
            "--num-envs",
            "1",
            "--phase",
            "P09",
        ]
    )
    assert selected.phase == "P09"
    assert selected.calibrate_effective_entry is False

    calibration_arguments = cli._parser().parse_args(
        [
            "phase-snapshot-live-probe",
            "--run-dir",
            str(PROJECT_ROOT),
            "--seed",
            "1002",
            "--num-envs",
            "1",
            "--phase",
            "P10",
            "--calibrate-effective-entry",
        ]
    )
    assert calibration_arguments.calibrate_effective_entry is True
    cli._validate_common(calibration_arguments)

    wrapper = (
        PROJECT_ROOT / "scripts" / "run_phase_snapshot_live_probe.ps1"
    ).read_text(encoding="utf-8")
    assert '-Subcommand "phase-snapshot-live-probe"' in wrapper
    assert '-RunKind "phase_snapshot_live_probe"' in wrapper
    assert "-EnvironmentCount 1" in wrapper
    assert "-ReturnFinalizedEvidenceFailure" in wrapper
    assert '"--phase-snapshot-prime-physics-steps", $PrimePhysicsSteps' in wrapper
    assert "[ValidateSet(1)]" in wrapper
    assert "[string]$Phase = $null" in wrapper
    assert 'if ($null -ne $Phase)' in wrapper
    assert '$BaseArgs += @("--phase", $Phase)' in wrapper

    calibration_wrapper = (
        PROJECT_ROOT / "scripts" / "run_phase_effective_entry_calibration.ps1"
    ).read_text(encoding="utf-8")
    assert '-RunKind "phase_effective_entry_calibration"' in calibration_wrapper
    assert '-TrainingStage "phase-effective-entry-calibration"' in calibration_wrapper
    assert '-Subcommand "phase-snapshot-live-probe"' in calibration_wrapper
    assert '"--calibrate-effective-entry"' in calibration_wrapper
    assert "[Parameter(Mandatory = $true)]" in calibration_wrapper
    assert "-ReturnFinalizedEvidenceFailure" not in calibration_wrapper

    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "phase-snapshot-live-probe",
                "--run-dir",
                str(PROJECT_ROOT),
                "--seed",
                "1001",
                "--num-envs",
                "1",
                "--phase-snapshot-prime-physics-steps",
                "2",
            ]
        )

    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "phase-snapshot-live-probe",
                "--run-dir",
                str(PROJECT_ROOT),
                "--seed",
                "1001",
                "--num-envs",
                "1",
                "--phase",
                "P01",
            ]
        )

    common = (PROJECT_ROOT / "scripts" / "_invoke_ppo_cli.ps1").read_text(
        encoding="utf-8"
    )
    assert '$RunKindValue -ceq "phase_snapshot_live_probe"' in common
    assert '$SubcommandValue -ceq "phase-snapshot-live-probe"' in common
    assert "$null -ne $AuthoritativeLiveExitCode" in common


def test_live_probe_rejects_non_one_prime_count_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    with pytest.raises(PhaseSnapshotLiveProbeError, match="fixed legacy ABI"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=object(),
            prime_physics_steps=2,
        )


@pytest.mark.parametrize(
    "phases",
    (("P01",), ("P14",), ("p09",), (), ("P02", "P03"), "P09"),
)
def test_live_probe_rejects_p01_or_invalid_phase_selector_before_runtime_mutation(
    tmp_path: Path,
    phases: object,
) -> None:
    with pytest.raises(PhaseSnapshotLiveProbeError, match="P02 through P13"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=object(),
            phases=phases,
        )
    assert not (tmp_path / "phase_snapshot_live_probe.json").exists()


def _write_managed_prechecks(run_dir: Path) -> None:
    (run_dir / "committed_runtime_identity.before.json").write_text(
        json.dumps({"schema": "wlr50_clean.committed_runtime_identity.v1"}),
        encoding="utf-8",
    )
    (run_dir / "frozen_hashes.before.json").write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
                "passed": True,
                "mismatches": [],
            }
        ),
        encoding="utf-8",
    )


class _FakeBundle:
    def __init__(self, root: Path) -> None:
        self.snapshot_root = root.resolve()
        self.bundle_sha256 = "c" * 64

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "wlr50_clean.phase_snapshot_bundle.v1",
            "snapshot_root": str(self.snapshot_root),
            "bundle_sha256": self.bundle_sha256,
        }


class _FakeEffectiveContract:
    def __init__(self, bundle: _FakeBundle) -> None:
        self.phase_snapshot_bundle_sha256 = bundle.bundle_sha256

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "wlr50_clean.ppo_phase_effective_entry_record.v1",
            "phase_snapshot_bundle_sha256": self.phase_snapshot_bundle_sha256,
        }


def test_calibration_mode_rejects_wrong_seed_contract_or_missing_phase(
    tmp_path: Path,
) -> None:
    bundle = _FakeBundle(tmp_path)
    with pytest.raises(PhaseSnapshotLiveProbeError, match="seed 1002"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            calibration_mode=True,
            phases=("P10",),
        )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="cannot consume"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1002,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
            calibration_mode=True,
            phases=("P10",),
        )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="explicit phase"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1002,
            snapshot_bundle=bundle,
            calibration_mode=True,
        )


@pytest.fixture(autouse=True)
def _allow_fake_effective_contract_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wlr50_clean.ppo import phase_effective_entry

    monkeypatch.setattr(
        phase_effective_entry,
        "assert_effective_phase_entry_contract_unchanged",
        lambda contract, *, expected_snapshot_bundle: contract,
    )


class _ContactRejectingBackend:
    """Fake one-scene backend that yields a trustworthy ordinary mismatch."""

    def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
        self._book = dependencies
        self._scene = None
        self._episode_tick = 0
        self._episode_sensor_tick_offset = 424242
        self._phase_snapshot_integrity_failed = False

    def reset(self, **kwargs: object) -> object:
        from wlr50_clean.ppo.isaac_fsm_backend import SensorContractFailure

        self._scene = object()
        self._book.current.snapshot_write_finished = True
        self._book.current.snapshot_state_write = {
            "root_pose_writes": 1,
            "root_velocity_writes": 1,
            "joint_state_writes": 1,
            "simulation_forward_syncs": 1,
            "physics_steps": 0,
        }
        self._snapshot_restoration = {
            "physical_state": {
                "schema": "wlr50_clean.phase_snapshot_prime_without_rewind.v2",
                "reset_use": "TRAINING_RESET_STATE_WRITE",
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "root_velocity_write_api": "write_root_link_velocity_to_sim",
                "state_write_count": 1,
                "post_prime_state_rewrite_performed": False,
                "contact_and_state_share_solver_tick": True,
                "prime_physics_steps": 1,
                "prime_applied_full12": [0.0] * 12,
                "physics_steps": 1,
                "fsm_clock_steps_during_priming": 0,
                "episode_clock_steps_during_priming": 0,
                "priming_observation": {
                    "maximum_errors": {"root_position_m": 0.0003},
                    "raw_physx_contact_sources_verified": True,
                    "current_raw_force_hysteresis_contract_matches_snapshot": True,
                },
            }
        }
        self._book.current.post_snapshot_observations.append(None)
        raise SensorContractFailure(
            "phase snapshot live restoration could not be proven: contact"
        )


def _patch_probe_snapshot_loader(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    from wlr50_clean.ppo import phase_snapshots

    def load(_bundle: object, phase: str):
        payload = _snapshot(phase)
        entry = SimpleNamespace(
            source_tick=payload["source_tick"],
            source_replay_steps=payload["source_replay_steps"],
            target_entry_tick=payload.get("target_entry_tick"),
            predecessor_verify_tick=payload.get("predecessor_verify_tick"),
            predecessor_verify_time_s=payload.get("predecessor_verify_time_s"),
            controller_anchor_tick=payload.get("controller_anchor_tick"),
            controller_anchor_time_s=payload.get("controller_anchor_time_s"),
            snapshot_path=root / phase / "snapshot.json",
            file_sha256="a" * 64,
            state_sha256="b" * 64,
        )
        return payload, entry

    monkeypatch.setattr(
        phase_snapshots,
        "load_validated_phase_snapshot_payload",
        load,
    )


def test_probe_infrastructure_initialization_failure_is_fatal_but_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)

    def fail_dependencies(book: object) -> object:
        raise RuntimeError("dependency initialization failed")

    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", fail_dependencies)
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["status"] == "FAILED"
    assert report["passed"] is False
    assert report["complete"] is False
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 0
    assert not (tmp_path / "live_command_result.json").exists()


def test_backend_bundle_revalidation_failure_cannot_be_returnable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)

    class IntegrityFailingBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._phase_snapshot_integrity_failed = True
            raise isaac_fsm_backend.IsaacFSMBackendError(
                "phase snapshot bundle validation failed"
            )

    monkeypatch.setattr(
        isaac_fsm_backend, "IsaacFSMBackend", IntegrityFailingBackend
    )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 1
    assert report["attempts"][0]["failure_classification"] == (
        "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    )
    assert not (tmp_path / "live_command_result.json").exists()


@pytest.mark.parametrize(
    ("exception_factory", "message"),
    (
        (
            lambda backend: backend.SensorContractFailure(
                "critical live sensing quality failed: exact pair unavailable"
            ),
            "critical live sensing quality failed",
        ),
        (
            lambda backend: backend.IsaacFSMBackendError(
                "frozen controller clock differs from live physics"
            ),
            "frozen controller clock differs",
        ),
    ),
)
def test_post_write_infrastructure_or_frozen_failure_remains_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: object,
    message: str,
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )

    class PostWriteFatalBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            self._book.current.post_snapshot_observations.append(None)
            raise exception_factory(isaac_fsm_backend)

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", PostWriteFatalBackend)
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert message in report["attempts"][0]["exception"]["message"]
    assert not (tmp_path / "live_command_result.json").exists()


def test_final_bundle_identity_assertion_failure_is_fatal_but_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)

    class SuccessfulResetBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            return SimpleNamespace(state_id="P02", physics_tick=0)

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", SuccessfulResetBackend)

    def fail_identity_assertion(bundle: object, *, canonical_root: Path) -> object:
        raise phase_snapshots.PhaseSnapshotError("filesystem identity changed A-B-A")

    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        fail_identity_assertion,
    )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 1
    assert "filesystem identity changed A-B-A" in report["failure_reasons"][0]
    assert not (tmp_path / "live_command_result.json").exists()


def test_successful_reset_with_missing_runtime_evidence_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )

    class IncompletelyObservedSuccessfulBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            # Returning a frame without the authoritative post-write sensor
            # sample must not be downgraded to an ordinary physical mismatch.
            return SimpleNamespace(state_id="P02", physics_tick=0)

    monkeypatch.setattr(
        isaac_fsm_backend,
        "IsaacFSMBackend",
        IncompletelyObservedSuccessfulBackend,
    )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 1
    assert report["attempts"][0]["failure_classification"] == (
        "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    )
    assert "violated probe runtime invariants" in report["failure_reasons"][0]
    assert not (tmp_path / "live_command_result.json").exists()


def test_post_write_contact_rejection_remains_returnable_failed_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )

    monkeypatch.setattr(
        isaac_fsm_backend, "IsaacFSMBackend", _ContactRejectingBackend
    )
    result = probe_subject.run_phase_snapshot_live_probe(
        object(),
        run_dir=tmp_path,
        seed=1001,
        snapshot_bundle=bundle,
        effective_entry_contract=_FakeEffectiveContract(bundle),
    )

    assert result["status"] == "FAILED"
    assert result["passed"] is False
    assert result["complete"] is True
    assert result["failure_classification"] == "EFFECTIVE_ENTRY_ACCEPTANCE_MISMATCH"
    assert result["phases"] == list(PROBE_PHASES)
    assert result["phase_count"] == len(PROBE_PHASES)
    assert result["phase_selector_mode"] == "all_non_p01_phases"
    assert result["expected_attempt_count"] == len(PROBE_PHASES) * 2
    assert result["fresh_scene_attempt_count"] == 1
    assert result["reused_scene_attempt_count"] == len(PROBE_PHASES) * 2 - 1
    assert len(result["attempts"]) == len(PROBE_PHASES) * ATTEMPTS_PER_PHASE
    assert all(
        row["failure_classification"] == "EFFECTIVE_ENTRY_ACCEPTANCE_MISMATCH"
        for row in result["attempts"]
    )
    assert result["production_reset_modified"] is True
    assert result["source_replay_policy"] == (
        "derived_only_from_validated_phase_snapshot"
    )
    assert result["source_replay_steps_by_phase"]["P10"] == 242
    assert all(
        steps == 1
        for phase, steps in result["source_replay_steps_by_phase"].items()
        if phase != "P10"
    )
    assert result["controller_anchors_by_phase"]["P10"] == {
        "controller_anchor_tick": 7784,
        "controller_anchor_time_s": 7784 / 120,
    }
    assert all(
        anchor is None
        for phase, anchor in result["controller_anchors_by_phase"].items()
        if phase != "P10"
    )
    assert result["predecessor_verify_anchors_by_phase"]["P10"] == {
        "predecessor_verify_tick": 7776,
        "predecessor_verify_time_s": 7776 / 120,
    }
    assert all(
        anchor is None
        for phase, anchor in result[
            "predecessor_verify_anchors_by_phase"
        ].items()
        if phase != "P10"
    )
    p10_attempts = [
        row for row in result["attempts"] if row["phase"] == "P10"
    ]
    assert len(p10_attempts) == ATTEMPTS_PER_PHASE
    assert all(
        row["source_tick"] == 7552
        and row["predecessor_verify_tick"] == 7776
        and row["predecessor_verify_time_s"] == 7776 / 120
        and row["controller_anchor_tick"] == 7784
        and row["controller_anchor_time_s"] == 7784 / 120
        and row["target_entry_tick"] == 7794
        for row in p10_attempts
    )
    assert all(
        row["predecessor_verify_tick"] is None
        and row["predecessor_verify_time_s"] is None
        and row["controller_anchor_tick"] is None
        and row["controller_anchor_time_s"] is None
        for row in result["attempts"]
        if row["phase"] != "P10"
    )
    assert all(
        row["snapshot_state_write"]["post_prime_state_rewrite_performed"]
        is False
        for row in result["attempts"]
    )
    assert all(
        row["snapshot_state_write"]["priming_observation"]["maximum_errors"]
        == {"root_position_m": 0.0003}
        for row in result["attempts"]
    )
    assert all(
        row["snapshot_state_write"]["priming_observation"][
            "raw_physx_contact_sources_verified"
        ]
        is True
        for row in result["attempts"]
    )


def test_single_phase_selector_runs_fresh_then_reused_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )
    monkeypatch.setattr(
        isaac_fsm_backend, "IsaacFSMBackend", _ContactRejectingBackend
    )

    result = probe_subject.run_phase_snapshot_live_probe(
        object(),
        run_dir=tmp_path,
        seed=1001,
        snapshot_bundle=bundle,
        effective_entry_contract=_FakeEffectiveContract(bundle),
        phases=("P09",),
    )

    assert result["status"] == "FAILED"
    assert result["complete"] is True
    assert result["phases"] == ["P09"]
    assert result["phase_count"] == 1
    assert result["phase_selector_mode"] == "single_phase"
    assert result["expected_attempt_count"] == 2
    assert result["expected_fresh_scene_attempt_count"] == 1
    assert result["expected_reused_scene_attempt_count"] == 1
    assert result["fresh_scene_attempt_count"] == 1
    assert result["reused_scene_attempt_count"] == 1
    assert [row["phase"] for row in result["attempts"]] == ["P09", "P09"]
    assert [row["attempt_kind"] for row in result["attempts"]] == [
        "primary",
        "reused_repeat",
    ]
    assert [row["scene_lifecycle"] for row in result["attempts"]] == [
        "fresh_scene",
        "reused_scene",
    ]
    assert [row["episode_sensor_tick_offset"] for row in result["attempts"]] == [
        424242,
        424242,
    ]
    assert all(
        row["episode_sensor_tick_offset"] != row["target_entry_tick"]
        for row in result["attempts"]
    )
