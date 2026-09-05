"""Real projection-to-float32-dispatch-to-writer contract, without Isaac.

Only articulation buffer storage and live sensor frames are test doubles. The
v2 projector, transition bridge, residual composition, frozen mapper/adapter,
same-tick counterfactual, and stream consumer all run their production code.
"""

from __future__ import annotations

import json
import math

import pytest

from test_actuator_target_effect import _adapter
from test_live_stream_writer import _frame

from wlr50_clean.ppo.actuator_target_effect import (
    actuator_target_audit_request,
    build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.isaac_fsm_backend import (
    _live_source_mapper_state,
    build_residual_actuation_plan,
)
from wlr50_clean.ppo.live_stream_writer import LiveStreamWriter
from wlr50_clean.ppo.phase_action_masks_v2 import (
    PhaseTransitionBridge,
    build_action_projector_v2,
)


ZERO = (0.0,) * 12


def _channel(value, index=0):
    result = list(ZERO)
    result[index] = value
    return tuple(result)


def _project(projector, raw, *, phase="P01", nominal=ZERO):
    return projector.project(
        raw,
        state_id=phase,
        nominal_action_full12=nominal,
        reference_action_full12=nominal,
        reference_delta_full12=ZERO,
    )


def _write_dispatch(writer, adapter, projector, projection, raw, *, phase="P01", nominal=ZERO, tick=1):
    plan = build_residual_actuation_plan(
        projection.applied_action_full12,
        frozen_nominal_full12=nominal,
        drive_feedback_bias_full12=ZERO,
        normal_drive_bias_full12=ZERO,
    )
    previous = _live_source_mapper_state(adapter, source_control_physics_tick=tick - 1)
    ack = adapter.apply_full12(
        plan.frozen_nominal_full12,
        physics_tick=180 + tick,
        drive_feedback_bias_full12=plan.combined_post_mapper_bias_full12,
    )
    audit = build_actuator_target_effect_audit(
        adapter=adapter,
        actuation=plan,
        raw_ack=ack,
        previous_final_drive_servo_deg=previous["final_drive_servo_deg"],
        source_phase_id=phase,
        policy_request=actuator_target_audit_request(phase, raw, projector.config.mask_for(phase)),
    )
    source, current = _frame(tick - 1), _frame(tick)
    source.state_id = current.state_id = phase
    source.nominal_action_full12 = current.nominal_action_full12 = nominal
    current.info["atomic_ack"] = ack
    current.info["actuator_target_effect_audit"] = audit
    writer.write_tick(source, current, projection)
    writer.write_decision({
        "decision_index": tick - 1,
        "sim_time_s": current.sim_time_s,
        "raw_policy_action_full12": list(raw),
        "reward": {"phase_id": phase, "total": 0.0},
        "phase_transition_action_jump": [],
    })
    return audit


def _finish(writer, tick=1):
    manifest = writer.finalize(_frame(tick, terminal=True), reward_total=0.0, decision_count=tick)
    return json.loads(manifest.read_text())["action_projection_audit"]


@pytest.mark.parametrize("channel,nominal_value", [(0, 10.0), (8, 0.5)])
def test_one_percent_with_nonzero_nominal_preserves_real_dispatch_residual_binding(
    tmp_path, channel, nominal_value,
):
    projector = build_action_projector_v2()
    adapter = _adapter()
    nominal = _channel(nominal_value, channel)
    # Reach the requested native target through the actual mature mapper,
    # without inventing mapper state or measuring a slew-saturated first tick.
    for tick in range(8):
        adapter.apply_full12(nominal, physics_tick=tick)
    raw = _channel(math.atanh(0.01), channel)
    projection = _project(projector, raw, nominal=nominal)
    dispatched_residual = projection.applied_action_full12[channel] - nominal_value
    assert projection.scaled_residual_full12[channel] != dispatched_residual
    assert projection.safe_projected_residual_full12[channel] == dispatched_residual

    writer = LiveStreamWriter(tmp_path / "rounded", seed=1001, require_actuator_target_effect_audit=True)
    writer.start(_frame(0))
    proof = _write_dispatch(writer, adapter, projector, projection, raw, nominal=nominal)
    summary = _finish(writer)

    assert proof["projected_residual_full12"][channel] == dispatched_residual
    assert proof["changed_channels_full12"][channel] is True
    assert summary["actuator_target_effect_audit_complete"] is True
    assert summary["within_one_percent_smoke_amplitude"] is True
    assert summary["own_policy_actuator_target_effect_phases"] == ["P01"]


@pytest.mark.parametrize("raw_value,phase", [(1.0e-9, "P01"), (1.0e-12, "P13")])
def test_quantized_zero_cannot_survive_real_end_to_end_effect_gate(tmp_path, raw_value, phase):
    projector = build_action_projector_v2()
    raw = _channel(raw_value)
    projection = _project(projector, raw, phase=phase)
    assert projection.safe_projected_residual_full12[0] != 0.0
    writer = LiveStreamWriter(tmp_path / "quantized", seed=1001, require_actuator_target_effect_audit=True)
    writer.start(_frame(0))
    proof = _write_dispatch(writer, _adapter(), projector, projection, raw, phase=phase)
    summary = _finish(writer)

    assert proof["changed_target_channel_count"] == 0
    assert proof["actual_native_targets"] == proof["counterfactual_native_targets"]
    assert summary["nonzero_residual_phases"] == [phase]
    assert summary["actuator_target_effect_audit_complete"] is True
    assert summary["own_policy_actuator_target_effect_phases"] == []


@pytest.mark.parametrize("phase,channel,target,fraction", (
    ("P05", 8, 0.3, 5.0e-7),
    ("P09", 8, -1.07, -5.0e-7),
))
def test_final_half_pulse_at_large_wheel_target_cannot_count_quantized_zero(
    tmp_path, phase, channel, target, fraction,
):
    projector = build_action_projector_v2()
    nominal = _channel(target, channel)
    raw = _channel(math.atanh(fraction), channel)
    projection = _project(projector, raw, phase=phase, nominal=nominal)
    assert projection.safe_projected_residual_full12[channel] != 0.0
    assert projection.applied_action_full12[channel] != nominal[channel]
    writer = LiveStreamWriter(tmp_path / "large_wheel_quantized", seed=1001,
                              require_actuator_target_effect_audit=True)
    writer.start(_frame(0))
    proof = _write_dispatch(writer, _adapter(), projector, projection, raw,
                           phase=phase, nominal=nominal)
    summary = _finish(writer)

    assert proof["target_dtype"] == "torch.float32"
    assert proof["actual_native_targets"] == proof["counterfactual_native_targets"]
    assert proof["changed_target_channel_count"] == 0
    assert proof["changed_channels_full12"][channel] is False
    assert summary["nonzero_residual_phases"] == [phase]
    assert summary["actuator_target_effect_audit_complete"] is True
    assert summary["own_policy_actuator_target_effect_phases"] == []


def test_actual_bridge_hold_is_not_new_phase_own_request_even_at_decision_boundary(tmp_path):
    projector = build_action_projector_v2()
    bridge = PhaseTransitionBridge(projector)
    adapter = _adapter()
    writer = LiveStreamWriter(tmp_path / "handoff", seed=1001, require_actuator_target_effect_audit=True)
    writer.start(_frame(0))

    for tick, phase, fraction in ((1, "P12", 0.01), (2, "P13", -0.01), (3, "P13", -0.01)):
        raw = _channel(math.atanh(fraction))
        bridged = bridge.project_tick(
            raw,
            state_id=phase,
            nominal_action_full12=ZERO,
            reference_action_full12=ZERO,
            reference_delta_full12=ZERO,
        )
        proof = _write_dispatch(writer, adapter, projector, bridged.projection, raw, phase=phase, tick=tick)
        assert proof["changed_channels_full12"][0] is True
        if tick == 2:
            assert bridged.transition_metric.handoff_hold_used is True
            assert bridged.projection.raw_residual_full12[0] > 0.0 > raw[0]
            assert proof["policy_request_phase"] == proof["source_phase_id"] == "P13"
            assert "P13" not in writer.own_policy_actuator_target_effect_phases

    summary = _finish(writer, tick=3)
    assert summary["actuator_target_effect_audit_complete"] is True
    assert summary["own_policy_actuator_target_effect_by_phase"]["P13"]["changed_target_tick_count"] == 1
    rows = [json.loads(line) for line in (writer.episode_dir / "full12_commands_120hz.jsonl").read_text().splitlines()]
    assert rows[1]["own_policy_actuator_target_effect"]["incoming_handoff_tick_excluded"] is True
    assert rows[1]["own_policy_actuator_target_effect"]["counted"] is False
    assert rows[2]["own_policy_actuator_target_effect"]["counted"] is True
