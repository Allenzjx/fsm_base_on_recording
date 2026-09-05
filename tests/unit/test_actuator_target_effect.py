from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    resolve_joint_indices,
)
from wlr50_clean.infrastructure.robot_adapter import RobotAdapter
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.ppo.actuator_target_effect import (
    ACTUATOR_TARGET_EFFECT_SCHEMA,
    ActuatorTargetEffectError,
    actuator_target_audit_request,
    build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.action_projection import ActionProjector, SafetyProjection
from wlr50_clean.ppo.isaac_fsm_backend import (
    IsaacFSMBackend,
    IsaacFSMBackendError,
    _live_source_mapper_state,
    build_residual_actuation_plan,
)


ZERO = (0.0,) * 12


class TargetBufferRobot:
    """Tensor staging/dispatch seam; all command mapping is the real adapter."""

    def __init__(self):
        self.joint_names = tuple(reversed(FULL12_ORDER))
        self.data = SimpleNamespace(
            joint_pos=torch.full((1, 12), 0.75, dtype=torch.float32),
            joint_vel=torch.zeros((1, 12), dtype=torch.float32),
            joint_pos_target=torch.zeros((1, 12), dtype=torch.float32),
            joint_vel_target=torch.zeros((1, 12), dtype=torch.float32),
        )
        self._joint_pos_target_sim = torch.zeros((1, 12), dtype=torch.float32)
        self._joint_vel_target_sim = torch.zeros((1, 12), dtype=torch.float32)
        self.events = []

    def set_joint_position_target(self, target, *, joint_ids):
        self.events.append("position.setter")
        self.data.joint_pos_target[:, joint_ids] = target

    def set_joint_velocity_target(self, target, *, joint_ids):
        self.events.append("velocity.setter")
        self.data.joint_vel_target[:, joint_ids] = target

    def write_data_to_sim(self):
        self.events.append("dispatch")
        self._joint_pos_target_sim.copy_(self.data.joint_pos_target)
        self._joint_vel_target_sim.copy_(self.data.joint_vel_target)

    def update(self, dt):
        self.events.append("update")


def _adapter():
    # Skip only live USD limit installation. The tested apply_full12, mapper,
    # tensor casting, command signs, limits, and physical mapping are untouched.
    adapter = RobotAdapter.__new__(RobotAdapter)
    adapter.robot = TargetBufferRobot()
    adapter.physics_dt_s = 1.0 / 120.0
    adapter.joint_map = resolve_joint_indices(adapter.robot.joint_names)
    adapter._standing_servo_tensor = adapter.robot.data.joint_pos[:, list(adapter.joint_map.servo_ids)].clone()
    adapter.standing_pose_deg = {name: math.degrees(0.75) for name in SERVO_ORDER}
    adapter.servo_target_mapper = ServoTargetMapper(adapter.standing_pose_deg)
    adapter._final_drive_servo_deg = {name: 0.0 for name in SERVO_ORDER}
    adapter.write_count = 0
    adapter._last_physics_tick = None
    adapter.last_ack = None
    return adapter


def _dispatch(*, residual=0.15, controller_bias=0.0, phase="P01", channel=0, previous_final=0.0):
    adapter = _adapter()
    if channel < 8:
        adapter._final_drive_servo_deg[SERVO_ORDER[channel]] = previous_final
    projected = list(ZERO)
    projected[channel] = residual
    bias = list(ZERO)
    bias[channel] = controller_bias
    actuation = build_residual_actuation_plan(
        projected,
        frozen_nominal_full12=ZERO,
        drive_feedback_bias_full12=bias,
        normal_drive_bias_full12=ZERO,
    )
    previous = _live_source_mapper_state(adapter, source_control_physics_tick=0)
    raw_ack = adapter.apply_full12(
        actuation.frozen_nominal_full12,
        physics_tick=180,
        drive_feedback_bias_full12=actuation.combined_post_mapper_bias_full12,
    )
    raw_policy = list(ZERO)
    raw_policy[channel] = 0.01
    return {
        "adapter": adapter,
        "actuation": actuation,
        "raw_ack": raw_ack,
        "previous_final_drive_servo_deg": previous["final_drive_servo_deg"],
        "source_phase_id": phase,
        "policy_request": actuator_target_audit_request(phase, raw_policy, (1,) * 12),
    }


@pytest.mark.parametrize("residual", [1.0e-9, 1.0e-12])
def test_nonzero_pre_cast_residual_below_float32_quantization_has_no_effect(residual):
    inputs = _dispatch(residual=residual, phase="P13" if residual == 1.0e-12 else "P01")
    audit = build_actuator_target_effect_audit(**inputs)
    assert inputs["raw_ack"]["servo_target_physical_rad"][0] != 0.75
    assert audit["projected_residual_full12"][0] != 0.0
    assert audit["changed_target_channel_count"] == 0
    assert audit["changed_channels_full12"] == [False] * 12
    assert audit["actual_native_targets"] == audit["counterfactual_native_targets"]


@pytest.mark.parametrize("channel", range(12))
def test_one_percent_policy_visible_target_in_canonical_full12_order(channel):
    raw = list(ZERO)
    raw[channel] = 0.01
    projection = ActionProjector().project(
        raw,
        state_id="P13",
        nominal_action_full12=ZERO,
        reference_action_full12=ZERO,
        reference_delta_full12=(10.0,) * 12,
    )
    inputs = _dispatch(channel=channel, residual=projection.applied_action_full12[channel], phase="P13")
    audit = build_actuator_target_effect_audit(**inputs)
    assert audit["schema"] == ACTUATOR_TARGET_EFFECT_SCHEMA
    assert audit["verified"] is True
    assert audit["target_dtype"] == "torch.float32"
    assert audit["raw_policy_action_full12"][channel] == 0.01
    assert audit["changed_target_channel_count"] == 1
    assert audit["changed_channels_full12"] == [index == channel for index in range(12)]
    assert audit["setter_dispatch_targets_equal"] is True
    assert audit["actual_mapping_matches_dispatch"] is True
    json.dumps(audit, allow_nan=False)


def test_safety_projected_zero_cannot_count_nonzero_policy_request_as_effect():
    projection = ActionProjector().project(
        (0.01,) + ZERO[1:],
        state_id="P01",
        nominal_action_full12=ZERO,
        reference_action_full12=ZERO,
        reference_delta_full12=(10.0,) * 12,
        safety=SafetyProjection(residual_enabled=False),
    )
    assert projection.hard_safety_modified is True
    assert projection.applied_action_full12 == ZERO
    audit = build_actuator_target_effect_audit(**_dispatch(residual=projection.applied_action_full12[0]))
    assert audit["raw_policy_action_full12"][0] == 0.01
    assert audit["projected_residual_full12"] == list(ZERO)
    assert audit["changed_target_channel_count"] == 0


def test_final_slew_clamps_actual_and_counterfactual_to_same_target():
    audit = build_actuator_target_effect_audit(**_dispatch(controller_bias=1.25))
    assert audit["projected_residual_full12"][0] != 0.0
    assert audit["changed_target_channel_count"] == 0


def test_counterfactual_uses_pre_dispatch_not_updated_final_drive():
    inputs = _dispatch(previous_final=1.5)
    assert inputs["adapter"]._final_drive_servo_deg[SERVO_ORDER[0]] == 0.25
    audit = build_actuator_target_effect_audit(**inputs)
    assert audit["previous_final_drive_servo_deg"][0] == 1.5
    assert audit["changed_target_channel_count"] == 0


def test_counterfactual_retains_controller_drive_feedback():
    audit = build_actuator_target_effect_audit(**_dispatch(controller_bias=0.3))
    expected = torch.tensor(0.75 + math.radians(0.3), dtype=torch.float32).item()
    assert audit["counterfactual_native_targets"]["servo_position_rad"][0] == expected
    assert audit["changed_target_channel_count"] == 1


def test_audit_reuses_frozen_mapping_without_second_advance_apply_or_dispatch(monkeypatch):
    inputs = _dispatch()
    adapter = inputs["adapter"]
    mapper_after = _live_source_mapper_state(adapter, source_control_physics_tick=1)
    event_count = len(adapter.robot.events)

    def forbidden(*args, **kwargs):
        pytest.fail("audit must not advance mapper or mutate actuator targets")

    monkeypatch.setattr(adapter.servo_target_mapper, "advance", forbidden)
    monkeypatch.setattr(adapter, "apply_full12", forbidden)
    monkeypatch.setattr(adapter.robot, "set_joint_position_target", forbidden)
    monkeypatch.setattr(adapter.robot, "set_joint_velocity_target", forbidden)
    monkeypatch.setattr(adapter.robot, "write_data_to_sim", forbidden)
    audit = build_actuator_target_effect_audit(**inputs)
    assert audit["changed_target_channel_count"] == 1
    assert adapter.write_count == 1
    assert len(adapter.robot.events) == event_count
    assert _live_source_mapper_state(adapter, source_control_physics_tick=1) == mapper_after


@pytest.mark.parametrize("tamper", ["dispatch", "both", "dtype", "missing", "nan"])
def test_unverifiable_actual_dispatch_fails_closed(tamper):
    inputs = _dispatch()
    robot = inputs["adapter"].robot
    joint_id = inputs["adapter"].joint_map.servo_ids[0]
    if tamper == "dispatch":
        robot._joint_pos_target_sim[0, joint_id] += 0.1
    elif tamper == "both":
        robot._joint_pos_target_sim[0, joint_id] += 0.1
        robot.data.joint_pos_target[0, joint_id] += 0.1
    elif tamper == "dtype":
        robot._joint_pos_target_sim = robot._joint_pos_target_sim.to(torch.float64)
    elif tamper == "missing":
        del robot._joint_pos_target_sim
    else:
        robot._joint_pos_target_sim[0, joint_id] = float("nan")
    with pytest.raises(ActuatorTargetEffectError):
        build_actuator_target_effect_audit(**inputs)


def test_backend_audit_default_disabled_and_request_only_saves_copied_metadata():
    disabled = IsaacFSMBackend()
    assert disabled._audit_actuator_target_effect is False
    with pytest.raises(IsaacFSMBackendError, match="not enabled"):
        disabled.set_actuator_target_audit_request("P01", ZERO, (1,) * 12)
    enabled = IsaacFSMBackend(audit_actuator_target_effect=True)
    action = list(ZERO)
    mask = [1] * 12
    enabled.set_actuator_target_audit_request("P01", action, mask)
    action[0] = 1.0
    mask[0] = 0
    assert enabled._actuator_target_audit_request["raw_policy_action_full12"][0] == 0.0
    assert enabled._actuator_target_audit_request["phase_mask_full12"][0] == 1
    enabled._poison_episode_state_for_reset(clear_evidence=True)
    assert enabled._actuator_target_audit_request is None


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_only_enabled_path_reads_audit_and_emits_frame_info(enabled, monkeypatch):
    from test_isaac_fsm_backend import FakeEffectiveEntryContract, FakeRuntime
    import wlr50_clean.ppo.actuator_target_effect as audit_module

    runtime = FakeRuntime()
    backend = IsaacFSMBackend(
        dependencies=runtime.dependencies(),
        expected_effective_entry_contract=FakeEffectiveEntryContract(),
        audit_actuator_target_effect=enabled,
    )
    backend.reset(seed=7, options={"randomization_enabled": False})
    if not enabled:
        def forbidden(**kwargs):
            pytest.fail("default training path must not compute target audits")
        monkeypatch.setattr(audit_module, "build_actuator_target_effect_audit", forbidden)
    else:
        adapter = _adapter()
        backend._adapter = adapter
        backend.set_actuator_target_audit_request("P01", (0.01,) + ZERO[1:], (1,) * 12)
        original_step = runtime.sim.step

        def step_after_completed_audit(*, render):
            assert adapter.robot.events == ["position.setter", "velocity.setter", "dispatch"]
            original_step(render=render)

        monkeypatch.setattr(runtime.sim, "step", step_after_completed_audit)
    frame = backend.step_physics((0.15,) + ZERO[1:])
    assert ("actuator_target_effect_audit" in frame.info) is enabled
    if enabled:
        audit = frame.info["actuator_target_effect_audit"]
        assert audit["source_phase_id"] == "P01"
        assert audit["policy_request_phase"] == "P01"
        assert audit["changed_target_channel_count"] == 1
        assert adapter.write_count == 1
