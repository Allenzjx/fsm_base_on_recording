"""Read-only, same-tick actuator effect evidence for the bounded smoke gate.

The mature mapper is advanced only by the real RobotAdapter dispatch.  Its
reported native result is shared by the actual and zero-current-residual
branches below.  Frozen scalar slew/clamp and physical conversion functions
are reused; neither branch writes targets or changes mapper state.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    Full12Command,
    build_physical_batch,
    servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step


ACTUATOR_TARGET_EFFECT_SCHEMA = "wlr50_clean.actuator_target_effect_audit.v1"


class ActuatorTargetEffectError(RuntimeError):
    """The actual float32 dispatch could not be independently bound to its audit."""


def actuator_target_audit_request(
    phase_id: str,
    raw_policy_action_full12: Sequence[float],
    phase_mask_full12: Sequence[int],
) -> dict[str, Any]:
    if phase_id not in {f"P{index:02d}" for index in range(1, 14)}:
        raise ActuatorTargetEffectError("policy request phase must be P01-P13")
    action = _finite_values(raw_policy_action_full12, 12, "raw policy action")
    mask = tuple(phase_mask_full12)
    if len(mask) != 12 or any(value not in (0, 1) for value in mask):
        raise ActuatorTargetEffectError("policy request mask must contain twelve binary values")
    return {
        "policy_request_phase": phase_id,
        "raw_policy_action_full12": list(action),
        "phase_mask_full12": [int(value) for value in mask],
    }


def build_actuator_target_effect_audit(
    *,
    adapter: Any,
    actuation: Any,
    raw_ack: Mapping[str, Any],
    previous_final_drive_servo_deg: Sequence[float],
    source_phase_id: str,
    policy_request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Inspect the already completed dispatch, never simulate another write.

    The counterfactual removes only this tick's PPO residual from the same
    measured state and t-1 final drive.  It is not an independent FSM rollout.
    This function is intentionally called only by explicitly enabled audits.
    """

    import torch

    if source_phase_id not in {f"P{index:02d}" for index in range(1, 14)}:
        raise ActuatorTargetEffectError("source phase must be P01-P13")
    if raw_ack.get("articulation_writes_this_call") != 1:
        raise ActuatorTargetEffectError("audit requires one completed articulation dispatch")
    previous = _finite_values(previous_final_drive_servo_deg, 8, "previous final drive")
    native = _finite_values(raw_ack["native_drive_target_full12"], 12, "native drive")
    controller_bias = _finite_values(actuation.controller_drive_bias_full12, 12, "controller bias")
    combined_bias = _finite_values(actuation.combined_post_mapper_bias_full12, 12, "combined bias")
    if _finite_values(raw_ack["drive_feedback_bias_requested_full12"], 12, "ack bias") != combined_bias:
        raise ActuatorTargetEffectError("actual dispatch bias differs from the actuation plan")
    maximum_delta = float(raw_ack["drive_feedback_final_slew_limit_deg_per_tick"])
    if maximum_delta != float(adapter.servo_target_mapper.maximum_delta_deg):
        raise ActuatorTargetEffectError("actual dispatch slew limit differs from the frozen mapper")

    def physical_targets(bias: Sequence[float]) -> Any:
        servo = []
        for index, name in enumerate(SERVO_ORDER):
            lower, upper = servo_limits_deg(name)
            servo.append(bounded_drive_feedback_step(
                previous_deg=previous[index],
                native_deg=native[index],
                bias_deg=bias[index],
                maximum_delta_deg=maximum_delta,
                lower_deg=lower,
                upper_deg=upper,
            ))
        # Full12Command owns hard wheel limits; build_physical_batch owns all
        # standing offsets, joint signs, and degree-to-radian conversion.
        command = Full12Command(
            tuple(servo), tuple(native[index] + bias[index] for index in range(8, 12))
        ).clamped()
        return build_physical_batch(command, adapter.standing_pose_deg)

    actual_physical = physical_targets(combined_bias)
    counterfactual_physical = physical_targets(controller_bias)
    robot = adapter.robot
    servo_ids = list(adapter.joint_map.servo_ids)
    wheel_ids = list(adapter.joint_map.wheel_ids)
    if len(servo_ids) != 8 or len(wheel_ids) != 4 or len(set(servo_ids + wheel_ids)) != 12:
        raise ActuatorTargetEffectError("audit requires twelve distinct canonical joint IDs")

    def read_targets(owner: Any, attribute: str, ids: list[int]) -> Any:
        tensor = getattr(owner, attribute, None)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ActuatorTargetEffectError(f"missing single-articulation target tensor: {attribute}")
        if tensor.dtype != torch.float32:
            raise ActuatorTargetEffectError(f"target tensor is not float32: {attribute}")
        selected = tensor[:, ids].detach().clone()
        if not bool(torch.isfinite(selected).all().item()):
            raise ActuatorTargetEffectError(f"non-finite actual target tensor: {attribute}")
        return selected

    staged_servo = read_targets(robot.data, "joint_pos_target", servo_ids)
    staged_wheel = read_targets(robot.data, "joint_vel_target", wheel_ids)
    dispatched_servo = read_targets(robot, "_joint_pos_target_sim", servo_ids)
    dispatched_wheel = read_targets(robot, "_joint_vel_target_sim", wheel_ids)
    if not torch.equal(staged_servo, dispatched_servo) or not torch.equal(staged_wheel, dispatched_wheel):
        raise ActuatorTargetEffectError("setter targets differ from completed PhysX dispatch buffers")

    def cast_targets(physical: Any) -> tuple[Any, Any]:
        return (
            dispatched_servo.new_tensor([physical.servo_target_rad]),
            dispatched_wheel.new_tensor([physical.wheel_target_rad_s]),
        )

    expected_servo, expected_wheel = cast_targets(actual_physical)
    if not torch.equal(expected_servo, dispatched_servo) or not torch.equal(expected_wheel, dispatched_wheel):
        raise ActuatorTargetEffectError("frozen mapping reconstruction differs from actual dispatch")
    nominal_servo, nominal_wheel = cast_targets(counterfactual_physical)
    # Numeric inequality intentionally does not count +0 versus -0 as effect.
    changed = torch.cat((dispatched_servo != nominal_servo, dispatched_wheel != nominal_wheel), dim=1)
    changed_channels = [bool(value) for value in changed.cpu().tolist()[0]]

    def record(servo: Any, wheel: Any) -> dict[str, list[float]]:
        return {
            "servo_position_rad": servo.cpu().tolist()[0],
            "wheel_velocity_rad_s": wheel.cpu().tolist()[0],
        }

    request = {
        "policy_request_phase": None,
        "raw_policy_action_full12": None,
        "phase_mask_full12": None,
    }
    if policy_request is not None:
        request = actuator_target_audit_request(
            policy_request["policy_request_phase"],
            policy_request["raw_policy_action_full12"],
            policy_request["phase_mask_full12"],
        )
    return {
        "schema": ACTUATOR_TARGET_EFFECT_SCHEMA,
        "verified": True,
        "source_phase_id": source_phase_id,
        **request,
        "physics_tick": int(raw_ack["physics_tick"]),
        "canonical_order": list(FULL12_ORDER),
        "projected_residual_full12": list(_finite_values(actuation.projected_residual_full12, 12, "projected residual")),
        "changed_channels_full12": changed_channels,
        "changed_target_channel_count": sum(changed_channels),
        "actual_native_targets": record(dispatched_servo, dispatched_wheel),
        "counterfactual_native_targets": record(nominal_servo, nominal_wheel),
        "native_target_delta": record(dispatched_servo - nominal_servo, dispatched_wheel - nominal_wheel),
        "target_dtype": str(dispatched_servo.dtype),
        "servo_target_dtype": str(dispatched_servo.dtype),
        "wheel_target_dtype": str(dispatched_wheel.dtype),
        "setter_dispatch_targets_equal": True,
        "actual_mapping_matches_dispatch": True,
        "same_tick_counterfactual": True,
        "counterfactual_scope": "same_pre_tick_state_without_current_ppo_residual",
        "actual_target_source": "robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim",
        "previous_final_drive_servo_deg": list(previous),
        "native_drive_target_full12": list(native),
        "controller_drive_bias_full12": list(controller_bias),
        "combined_post_mapper_bias_full12": list(combined_bias),
    }


def _finite_values(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise ActuatorTargetEffectError(f"{label} must contain {size} finite values")
    return result
