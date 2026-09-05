from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wlr50_clean.fsm.controller import ControllerFrame
from wlr50_clean.fsm.state_spec import Lifecycle
from wlr50_clean.infrastructure.command_batch import WHEEL_VELOCITY_LIMIT_RAD_S
from wlr50_clean.ppo.cli import CliError, _smoke_action
from wlr50_clean.ppo.phase_action_masks_v2 import (
    PhaseTransitionBridge,
    build_action_projector_v2,
    load_phase_action_masks_v2,
)


ZERO = (0.0,) * 12


def _controller(phase="P01", *, nominal=ZERO, feedback=ZERO, normal=ZERO, tick=0):
    # The production dataclass keeps this fixture aligned with backend metadata.
    return ControllerFrame(
        physics_tick=tick, sim_time_s=tick / 120.0, state_id=phase,
        lifecycle=Lifecycle.EXECUTE_MOTION, full12=nominal,
        decision_tick=True, full12_atomic_write_required=True,
        atomic_source_event=False, tracking_servo_names=(),
        drive_feedback_bias_full12=feedback, normal_drive_bias_full12=normal,
        drive_feedback_details={}, endpoint_issued=False, termination=None,
        first_blocker=None, events=(),
    )


def _env(phase="P01", *, nominal=ZERO, feedback=ZERO, normal=ZERO):
    return SimpleNamespace(
        frame=SimpleNamespace(
            state_id=phase, phase_progress=0.0, nominal_action_full12=nominal,
            info={"raw_controller_frame": _controller(
                phase, nominal=nominal, feedback=feedback, normal=normal,
            )},
        ),
        phase_actions=load_phase_action_masks_v2(),
    )


def test_selection_includes_both_controller_biases_and_is_cached_with_stable_ties():
    nominal = ZERO[:8] + (0.0, 0.1, 0.2, 0.3)
    env = _env(nominal=nominal, feedback=ZERO[:8] + (0.5, -0.07, 0.0, 0.0),
               normal=ZERO[:8] + (0.0, -0.03, 0.0, 0.0))
    first = _smoke_action(env, 0)
    assert first[9] == math.atanh(0.00005)  # nominal alone would choose 8.
    assert first[:8] == ZERO[:8]
    env.frame.info["raw_controller_frame"] = _controller(nominal=nominal)
    assert _smoke_action(env, 1)[9] == math.atanh(0.0001)  # Do not reselect 8.
    tied = _env()
    assert _smoke_action(tied, 0)[8] == math.atanh(0.00005)


@pytest.mark.parametrize("phase,wheels,expected", (
    ("P07", (0.5, 0.0, 0.1, 0.05), 11),
    ("P08", (0.5, 0.0, 0.1, 0.05), 11),
    ("P10", (0.0, 0.05, 0.2, 0.01), 9),
    ("P11", (0.0, 0.05, 0.2, 0.01), 9),
    ("P13", (0.5, 0.4, 0.3, 0.0), 11),
))
def test_selection_uses_current_and_next_mask_not_just_nearest_wheel(phase, wheels, expected):
    env = _env(phase, nominal=ZERO[:8] + wheels)
    _smoke_action(env, 0)
    assert env._bounded_smoke_phase_channels[phase] == expected


@pytest.mark.parametrize("cap_phase", ("P01", "P02"))
def test_selection_requires_positive_cap_in_both_phases(cap_phase):
    env = _env()
    phases = dict(env.phase_actions.phases)
    scales = list(phases[cap_phase].scale_full12)
    scales[8] = 0.0
    phases[cap_phase] = replace(phases[cap_phase], scale_full12=tuple(scales))
    env.phase_actions = replace(env.phase_actions, phases=phases)
    _smoke_action(env, 0)
    assert env._bounded_smoke_phase_channels["P01"] == 9


@pytest.mark.parametrize("case", (
    "missing_frame", "missing_bias", "short_bias", "nan_bias", "infinite_bias",
    "nonnumeric_bias", "combined_wheel_overflow", "combined_servo_overflow",
    "stale_phase", "stale_nominal", "invalid_after_cache",
))
def test_missing_or_illegal_controller_metadata_is_never_silently_ignored(case):
    env = _env()
    controller = env.frame.info["raw_controller_frame"]
    if case == "missing_frame":
        del env.frame.info["raw_controller_frame"]
    elif case == "missing_bias":
        env.frame.info["raw_controller_frame"] = SimpleNamespace(
            state_id="P01", full12=ZERO, drive_feedback_bias_full12=ZERO,
        )
    elif case == "stale_phase":
        env.frame.info["raw_controller_frame"] = replace(controller, state_id="P02")
    elif case == "stale_nominal":
        env.frame.info["raw_controller_frame"] = replace(controller, full12=(1.0,) + ZERO[1:])
    else:
        if case == "invalid_after_cache":
            _smoke_action(env, 0)
        value = {
            "short_bias": (), "nan_bias": (float("nan"),) + ZERO[1:],
            "infinite_bias": ZERO[:8] + (float("inf"),) + ZERO[9:],
            "nonnumeric_bias": ("invalid",) + ZERO[1:],
            "combined_wheel_overflow": ZERO[:8] + (WHEEL_VELOCITY_LIMIT_RAD_S,) + ZERO[9:],
            "combined_servo_overflow": (31.0,) + ZERO[1:],
            "invalid_after_cache": (float("nan"),) + ZERO[1:],
        }[case]
        normal = ZERO[:8] + (0.1,) + ZERO[9:] if case == "combined_wheel_overflow" else ZERO
        env.frame.info["raw_controller_frame"] = replace(
            controller, drive_feedback_bias_full12=value, normal_drive_bias_full12=normal,
        )
    with pytest.raises(CliError, match="bounded smoke"):
        _smoke_action(env, 1)


def test_real_wheel_float32_dispatch_and_own_effect_survive_all_twelve_handoffs(tmp_path):
    from test_actuator_target_effect import _adapter
    from test_live_stream_writer import _frame
    from wlr50_clean.ppo.actuator_target_effect import (
        actuator_target_audit_request, build_actuator_target_effect_audit,
    )
    from wlr50_clean.ppo.isaac_fsm_backend import (
        _live_source_mapper_state, build_residual_actuation_plan,
    )
    from wlr50_clean.ppo.live_stream_writer import LiveStreamWriter

    feedback = ZERO[:8] + (0.2, 0.15, 0.1, 0.05)
    normal = ZERO[:8] + (-0.19, -0.13, -0.07, -0.01)
    env = _env(feedback=feedback, normal=normal)
    adapter = _adapter()
    bridge = PhaseTransitionBridge(build_action_projector_v2())
    writer = LiveStreamWriter(tmp_path / "wheel_handoffs", seed=1001,
                              require_actuator_target_effect_audit=True)
    writer.start(_frame(0))
    tick = 0
    decision = 0
    handoffs = []
    phases = [f"P{number:02d}" for number in range(1, 14)]
    for phase in phases:
        env.frame = _env(phase, feedback=feedback, normal=normal).frame
        # Six nonzero decisions through P01-P12; P13 additionally proves its
        # unchanged seventh-decision exact-zero settling behavior.
        for local_decision in range(7 if phase == "P13" else 6):
            raw = _smoke_action(env, decision)
            selected = env._bounded_smoke_phase_channels[phase]
            assert raw[:8] == ZERO[:8]
            if phase == "P13" and local_decision == 6:
                assert raw == ZERO
            transitions = []
            for _ in range(8):
                projected = bridge.project_tick(
                    raw, state_id=phase, nominal_action_full12=ZERO,
                    reference_action_full12=ZERO, reference_delta_full12=ZERO,
                )
                plan = build_residual_actuation_plan(
                    projected.projection.applied_action_full12, frozen_nominal_full12=ZERO,
                    drive_feedback_bias_full12=feedback, normal_drive_bias_full12=normal,
                )
                before = _live_source_mapper_state(adapter, source_control_physics_tick=tick)
                ack = adapter.apply_full12(plan.frozen_nominal_full12, physics_tick=180 + tick,
                                          drive_feedback_bias_full12=plan.combined_post_mapper_bias_full12)
                proof = build_actuator_target_effect_audit(
                    adapter=adapter, actuation=plan, raw_ack=ack,
                    previous_final_drive_servo_deg=before["final_drive_servo_deg"],
                    source_phase_id=phase,
                    policy_request=actuator_target_audit_request(phase, raw, env.phase_actions.mask_for(phase)),
                )
                assert proof["verified"] is True
                assert proof["target_dtype"] == "torch.float32"
                assert not any(proof["changed_channels_full12"][:8])
                transition = projected.transition_metric
                if transition is not None:
                    assert transition.handoff_hold_used is True
                    assert transition.forbidden_channel_indices == ()
                    handoffs.append((transition.from_state_id, transition.to_state_id))
                    transitions.append(transition.as_dict())
                elif raw != ZERO:
                    assert proof["changed_channels_full12"][selected] is True
                    assert (proof["actual_native_targets"]["wheel_velocity_rad_s"][selected - 8]
                            != proof["counterfactual_native_targets"]["wheel_velocity_rad_s"][selected - 8])
                source, current = _frame(tick), _frame(tick + 1)
                source.state_id = current.state_id = phase
                source.info["raw_controller_frame"] = _controller(
                    phase, feedback=feedback, normal=normal, tick=tick,
                )
                current.info.update(atomic_ack=ack, actuator_target_effect_audit=proof)
                writer.write_tick(source, current, projected.projection)
                tick += 1
            writer.write_decision({
                "decision_index": decision, "sim_time_s": tick / 120.0,
                "raw_policy_action_full12": list(raw), "reward": {"phase_id": phase, "total": 0.0},
                "phase_transition_action_jump": transitions,
            })
            decision += 1
    path = writer.finalize(_frame(tick, terminal=True), reward_total=0.0, decision_count=decision)
    audit = json.loads(path.read_text())["action_projection_audit"]
    assert handoffs == list(zip(phases[:-1], phases[1:]))
    assert ("P07", "P08") in handoffs and ("P10", "P11") in handoffs
    assert audit["phase_transition_handoff_hold_count"] == 12
    assert audit["actuator_target_effect_audit_complete"] is True
    assert audit["own_policy_actuator_target_effect_phases"] == phases
    assert audit["mask_honored_when_exercised"] is True
    assert audit["within_one_percent_smoke_amplitude"] is True
