from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo.action_projection import ActionProjector, SafetyProjection
from wlr50_clean.ppo.phase_action_masks_v2 import (
    PhaseTransitionBridge,
    build_action_projector_v2,
    load_phase_action_masks_v2,
)
from wlr50_clean.ppo.cli import _smoke_action, _smoke_rate_limit_probe


ZERO12 = (0.0,) * 12
ONE12 = (1,) * 12
PRIORITY_PHASES = ("P02", "P03", "P08", "P12", "P13")


def test_deterministic_small_smoke_pattern_is_nonzero_and_negligible() -> None:
    env = SimpleNamespace(
        frame=SimpleNamespace(
            state_id="P03",
            phase_progress=0.8,
            nominal_action_full12=(0.0, -22.9, 0.0, 45.9) + ZERO12[4:],
        ),
        phase_actions=load_phase_action_masks_v2(),
    )
    positive = _smoke_action(env, 0)
    repeated = _smoke_action(env, 1)
    assert positive[3] == 1.0e-9
    assert sum(value != 0.0 for value in positive) == 1
    assert repeated == positive
    assert max(abs(value) for value in positive + repeated) < 0.05


def test_deterministic_rate_limit_probe_uses_production_projector_off_robot() -> None:
    probe = _smoke_rate_limit_probe()

    assert probe["passed"] is True
    assert probe["applied_to_robot"] is False
    assert probe["normalized_probe_amplitude"] == 0.049
    assert "residual_rate_limit" in probe["second_clipping_stages"]


def test_smoke_pattern_arms_late_and_exercises_only_disabled_mask_channels() -> None:
    config = load_phase_action_masks_v2()
    env = SimpleNamespace(
        frame=SimpleNamespace(
            state_id="P08",
            phase_progress=0.5,
            nominal_action_full12=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0)
            + ZERO12[8:],
        ),
        phase_actions=config,
    )
    assert _smoke_action(env, 0) == ZERO12

    env.frame.phase_progress = 0.75
    armed = _smoke_action(env, 1)
    assert armed[7] == 1.0e-9
    mask = config.mask_for("P08")
    assert all(armed[index] == -1.0e-9 for index, value in enumerate(mask) if not value)
    assert all(
        armed[index] == 0.0
        for index, value in enumerate(mask)
        if value and index != 7
    )

    # Once armed, the phase remains non-zero through its transition boundary.
    env.frame.phase_progress = 0.0
    assert _smoke_action(env, 2) == armed


def _direct_project(
    projector: ActionProjector,
    raw=ZERO12,
    *,
    state_id="P03",
    nominal=ZERO12,
    previous=ZERO12,
    safety=None,
    runtime_mask=None,
    dt_s=1.0 / 120.0,
):
    return projector.project(
        raw,
        state_id=state_id,
        nominal_action_full12=nominal,
        reference_action_full12=nominal,
        reference_delta_full12=(0.1,) * 12,
        previous_projected_residual_full12=previous,
        runtime_action_mask_full12=runtime_mask,
        safety=safety,
        dt_s=dt_s,
    )


def _bridge_project(
    bridge: PhaseTransitionBridge,
    raw,
    *,
    state_id,
    nominal=ZERO12,
    safety=None,
    runtime_mask=None,
    dt_s=1.0 / 120.0,
):
    return bridge.project_tick(
        raw,
        state_id=state_id,
        nominal_action_full12=nominal,
        reference_action_full12=nominal,
        reference_delta_full12=(0.1,) * 12,
        runtime_action_mask_full12=runtime_mask,
        safety=safety,
        dt_s=dt_s,
    )


def test_v2_is_trainable_small_and_nonempty_for_every_phase() -> None:
    config = load_phase_action_masks_v2()

    assert config.training_enabled is True
    assert config.physics_ticks_per_decision == 8
    assert tuple(config.phases) == tuple(f"P{index:02d}" for index in range(1, 14))
    assert config.live_small_perturbation_smoke_required is True
    for phase in config.phases.values():
        assert any(phase.mask_full12)
        assert phase.mask_full12[:8] == (1,) * 8
        assert all(
            bool(mask) == (scale > 0.0)
            for mask, scale in zip(
                phase.mask_full12, phase.scale_full12, strict=True
            )
        )
        assert max(phase.scale_full12[:8]) <= 4.0
        assert max(phase.scale_full12[8:]) <= 0.12


def test_five_priority_phase_masks_cover_required_roles() -> None:
    config = load_phase_action_masks_v2()

    assert config.mask_for("P02") == ONE12
    p02_scale = config.physical_scale_for("P02")
    assert max(p02_scale[2:4]) < min(p02_scale[0:2] + p02_scale[4:8])

    assert config.mask_for("P03") == ONE12
    assert config.mask_for("P08")[:8] == (1,) * 8
    assert config.mask_for("P08")[8:] == (1, 0, 0, 1)
    assert config.mask_for("P12") == ONE12
    assert config.mask_for("P13") == ONE12
    assert all(any(config.mask_for(state_id)) for state_id in PRIORITY_PHASES)


def test_scale_audit_rows_explain_every_phase_channel_without_recording_cap() -> None:
    config = load_phase_action_masks_v2()
    rows = config.scale_audit_rows()

    assert len(rows) == 13 * 12
    assert {(row.state_id, row.channel_index) for row in rows} == {
        (f"P{phase:02d}", channel)
        for phase in range(1, 14)
        for channel in range(12)
    }
    assert all(row.recording_envelope_hard_cap is False for row in rows)
    assert all(row.role and row.sensitivity_tier for row in rows)
    assert all(
        row.safe_span_fraction <= config.maximum_initial_safe_span_fraction + 1.0e-12
        for row in rows
    )
    assert all(row.enabled == (row.residual_scale > 0.0) for row in rows)

    p03_fr_knee = next(
        row
        for row in rows
        if row.state_id == "P03" and row.channel_name == "front_right_knee"
    )
    assert p03_fr_knee.unit == "deg"
    assert p03_fr_knee.safety_span == pytest.approx(266.0)
    assert p03_fr_knee.residual_scale == pytest.approx(4.0)
    assert p03_fr_knee.safe_span_fraction == pytest.approx(4.0 / 266.0)
    assert p03_fr_knee.as_dict()["derivation"] == (
        "safe_span_x_phase_role_x_sensitivity_tier"
    )


def test_v2_injects_existing_projector_and_recording_is_diagnostic_only() -> None:
    config = load_phase_action_masks_v2()
    projector = build_action_projector_v2(config)

    assert type(projector) is ActionProjector
    assert projector.config.training_enabled is True
    assert projector.config.recording_envelope_hard_constraint is False
    assert projector.config.servo_residual_rate_deg_s == pytest.approx(45.0)
    assert projector.config.wheel_residual_rate_rad_s2 == pytest.approx(3.0)
    assert (
        projector.config.scale_for("P03")[3]
        * projector.config.physical_residual_scale_full12[3]
        == pytest.approx(4.0)
    )

    result = _direct_project(
        projector,
        raw=ZERO12[:3] + (100.0,) + ZERO12[4:],
        state_id="P03",
        dt_s=1.0,
    )
    # The shared conformance diagnostic suggests only the 2 deg absolute
    # floor here; it does not cap the independently derived 4 deg v2 scale.
    assert result.recording_scale_suggestion_full12[3] == pytest.approx(2.0)
    assert result.safe_projected_residual_full12[3] == pytest.approx(4.0)
    assert result.safe_projected_residual_full12[3] > result.recording_scale_suggestion_full12[3]
    assert all("recording" not in stage for stage in result.clipping_stages)


def test_v2_projector_keeps_phase_mask_slew_limits_and_safety_order() -> None:
    projector = build_action_projector_v2()
    projected = _direct_project(
        projector,
        raw=(100.0,) * 12,
        state_id="P08",
    )

    assert projected.effective_action_mask_full12[8:] == (1, 0, 0, 1)
    assert projected.safe_projected_residual_full12[0] == pytest.approx(0.375)
    assert projected.safe_projected_residual_full12[8] == pytest.approx(0.025)
    assert projected.safe_projected_residual_full12[9:11] == (0.0, 0.0)
    assert "residual_rate_limit" in projected.clipping_stages

    nominal = (1.0,) * 8 + (0.5,) * 4
    for stop_flag in ("body_collision_detected", "wheel_only_climb_detected"):
        stopped = _direct_project(
            projector,
            raw=(100.0,) * 12,
            state_id="P03",
            nominal=nominal,
            safety=SafetyProjection(**{stop_flag: True}),
        )
        assert stopped.applied_action_full12[:8] == nominal[:8]
        assert stopped.applied_action_full12[8:] == (0.0,) * 4
        assert stopped.clipping_stages[-1] == "body_collision_or_wheel_only_safety"


def test_v2_margin_projection_preserves_outside_band_frozen_nominal() -> None:
    projector = build_action_projector_v2()
    nominal = (-134.0, -59.0) + ZERO12[2:]
    projected = _direct_project(
        projector,
        raw=(-100.0, -100.0) + ZERO12[2:],
        state_id="P03",
        nominal=nominal,
        dt_s=1.0,
    )

    assert projected.limit_projected_residual_full12[:2] == (0.0, 0.0)
    assert projected.applied_action_full12 == nominal


def test_transition_bridge_holds_allowed_residual_then_slews_without_forbidden_leak() -> None:
    config = load_phase_action_masks_v2()
    projector = build_action_projector_v2(config)
    bridge = PhaseTransitionBridge(projector)

    before = _bridge_project(
        bridge,
        (100.0,) * 12,
        state_id="P07",
        dt_s=1.0,
    ).projection
    assert before.safe_projected_residual_full12[0] == pytest.approx(1.5)
    assert before.safe_projected_residual_full12[9] == pytest.approx(0.08)

    first_new_phase = _bridge_project(
        bridge,
        ZERO12,
        state_id="P08",
    )
    result = first_new_phase.projection
    metric = first_new_phase.transition_metric
    assert metric is not None
    assert metric.from_state_id == "P07"
    assert metric.to_state_id == "P08"
    assert metric.handoff_hold_used is True
    # Channel 0 remains legal and cannot jump to zero on the transition tick.
    assert result.safe_projected_residual_full12[0] == pytest.approx(
        before.safe_projected_residual_full12[0]
    )
    assert metric.residual_step_full12[0] == pytest.approx(0.0, abs=1.0e-9)
    # A retained residual that exceeds the new phase's smaller cap is reduced
    # by no more than the configured one-tick projector slew.
    assert metric.max_abs_servo_residual_step_deg <= 0.375 + 1.0e-12
    # P08 forbids FR/RL wheel residuals, so they are removed before projection.
    assert metric.forbidden_channel_indices == (9, 10)
    assert result.safe_projected_residual_full12[9:11] == (0.0, 0.0)
    assert metric.dropped_forbidden_residual_full12[9] == pytest.approx(0.08)
    assert metric.max_abs_wheel_action_jump_rad_s == pytest.approx(0.08)

    next_tick = _bridge_project(bridge, ZERO12, state_id="P08")
    assert next_tick.transition_metric is None
    assert next_tick.projection.safe_projected_residual_full12[0] == pytest.approx(1.125)
    assert (
        before.safe_projected_residual_full12[0]
        > next_tick.projection.safe_projected_residual_full12[0]
        > 0.0
    )


def test_transition_bridge_obeys_runtime_mask_and_physical_stop_immediately() -> None:
    projector = build_action_projector_v2()
    bridge = PhaseTransitionBridge(projector)
    _bridge_project(bridge, (100.0,) * 12, state_id="P02", dt_s=1.0)

    runtime_mask = (0,) + (1,) * 11
    masked = _bridge_project(
        bridge,
        ZERO12,
        state_id="P03",
        runtime_mask=runtime_mask,
    )
    assert masked.projection.safe_projected_residual_full12[0] == 0.0
    assert masked.transition_metric is not None
    assert 0 in masked.transition_metric.forbidden_channel_indices

    stopped = _bridge_project(
        bridge,
        (100.0,) * 12,
        state_id="P04",
        nominal=(1.0,) * 8 + (0.5,) * 4,
        safety=SafetyProjection(body_collision_detected=True),
    )
    assert stopped.projection.applied_action_full12[:8] == (1.0,) * 8
    assert stopped.projection.applied_action_full12[8:] == (0.0,) * 4
    assert bridge.previous_projected_residual_full12 == ZERO12
    assert stopped.transition_metric is not None
    assert stopped.transition_metric.hard_safety_modified is True


def test_transition_jump_metric_is_log_row_ready() -> None:
    bridge = PhaseTransitionBridge(build_action_projector_v2())
    _bridge_project(bridge, (1.0,) * 12, state_id="P12", dt_s=1.0)
    transitioned = _bridge_project(bridge, ZERO12, state_id="P13")
    assert transitioned.transition_metric is not None
    row = transitioned.transition_metric.as_dict()
    assert row["from_state_id"] == "P12"
    assert row["to_state_id"] == "P13"
    assert math.isfinite(row["max_abs_servo_action_jump_deg"])
    assert math.isfinite(row["max_abs_wheel_action_jump_rad_s"])
