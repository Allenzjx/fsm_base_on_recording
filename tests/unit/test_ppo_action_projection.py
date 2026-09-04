from __future__ import annotations

import math

import pytest

from wlr50_clean.ppo.action_projection import (
    ActionProjectionError,
    ActionProjector,
    SafetyProjection,
    ZeroResidualEpisodeAuditor,
    bitwise_full12_equal,
    load_action_projection_config,
)


ZERO = (0.0,) * 12


def _project(
    raw,
    *,
    state="P13",
    nominal=ZERO,
    reference=ZERO,
    delta=(10.0,) * 12,
    previous=ZERO,
    mask=None,
    safety=None,
    dt=1.0 / 15.0,
):
    return ActionProjector().project(
        raw,
        state_id=state,
        nominal_action_full12=nominal,
        reference_action_full12=reference,
        reference_delta_full12=delta,
        previous_projected_residual_full12=previous,
        runtime_action_mask_full12=mask,
        safety=safety,
        dt_s=dt,
    )


def test_phase_masks_derive_strictly_from_explicit_contract_ppo_masks() -> None:
    config = load_action_projection_config()
    assert config.training_enabled is False
    assert config.physics_ticks_per_decision == 8
    assert config.recording_envelope_hard_constraint is False
    assert config.recording_envelope_initialization_suggestion is True
    assert config.safety_limits_full12[0] == (-133.0, 133.0)
    assert config.safety_limits_full12[1] == (-58.0, 208.0)
    assert config.physical_residual_scale_full12[0] == 266.0
    assert config.physical_residual_scale_full12[8] == pytest.approx(
        4.1887902047863905
    )
    assert config.mask_for("P01") == (1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1)
    assert config.mask_for("P02") == (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert config.mask_for("P03") == (1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1)
    assert config.mask_for("P10") == (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    assert config.mask_for("P13")[2] == 1
    assert sum(config.mask_for("P13")) == 12


def test_zero_residual_uses_bitwise_nominal_fast_path() -> None:
    nominal = (-0.0, 1.25, -2.5, 0.0, 3.5, -4.5, 5.5, -6.5, -0.0, 0.2, -0.3, 0.4)
    result = _project(ZERO, nominal=nominal, reference=nominal)
    assert result.zero_residual_fast_path is True
    assert result.clipping_stages == ()
    assert bitwise_full12_equal(result.applied_action_full12, nominal)


def test_zero_raw_residual_with_nonzero_prior_decays_through_rate_limit() -> None:
    previous = (10.0,) + ZERO[1:]
    result = _project(
        ZERO,
        previous=previous,
        dt=1.0 / 120.0,
    )
    assert result.zero_residual_fast_path is False
    assert result.rate_projected_residual_full12[0] == pytest.approx(8.75)
    assert result.safe_projected_residual_full12[0] == pytest.approx(8.75)
    assert result.applied_action_full12[0] == pytest.approx(8.75)
    assert "residual_rate_limit" in result.clipping_stages


def test_zero_residual_full_episode_auditor_hashes_every_tick_bitwise() -> None:
    auditor = ZeroResidualEpisodeAuditor()
    for tick in range(257):
        nominal = (
            -0.0 if tick % 2 else 0.0,
            tick / 100.0,
            -tick / 200.0,
        ) + (0.0,) * 9
        projected = _project(ZERO, nominal=nominal, reference=nominal)
        auditor.append(nominal, projected.applied_action_full12)
    audit = auditor.finalize()
    assert audit.status == "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE"
    assert audit.tick_count == 257
    assert audit.bitwise_equal is True
    assert audit.nominal_sequence_sha256 == audit.applied_sequence_sha256


def test_recording_envelope_is_diagnostic_not_a_hard_projection() -> None:
    nominal = list(ZERO)
    reference = list(ZERO)
    delta = list((10.0,) * 12)
    nominal[7] = 12.5
    reference[7] = 10.0
    result = _project(
        (100.0,) * 12,
        state="P10",
        nominal=tuple(nominal),
        reference=tuple(reference),
        delta=tuple(delta),
        mask=(1,) * 12,  # historical/permissive masks may only be closed
    )
    assert result.effective_action_mask_full12 == (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    assert result.remaining_recording_envelope_diagnostic_full12[7] == pytest.approx(0.5)
    assert result.safe_projected_residual_full12[7] == pytest.approx(10.0)
    assert (
        result.safe_projected_residual_full12[7]
        > result.recording_scale_suggestion_full12[7]
    )
    assert result.recording_envelope_exceeded_full12[7] is True
    assert all(
        value == 0.0
        for index, value in enumerate(result.safe_projected_residual_full12)
        if index != 7
    )
    assert all("recording" not in stage for stage in result.clipping_stages)


def test_rate_limit_does_not_treat_shrinking_recording_headroom_as_safety() -> None:
    nominal = list(ZERO)
    reference = list(ZERO)
    nominal[7] = 12.8
    reference[7] = 10.0
    result = _project(
        (100.0,) * 12,
        state="P10",
        nominal=tuple(nominal),
        reference=tuple(reference),
        previous=ZERO[:7] + (2.0,) + ZERO[8:],
        dt=1.0 / 120.0,
    )
    assert result.remaining_recording_envelope_diagnostic_full12[7] == pytest.approx(0.2)
    assert result.rate_projected_residual_full12[7] == pytest.approx(3.25)
    assert result.recording_envelope_exceeded_full12[7] is True
    assert all("headroom" not in stage for stage in result.clipping_stages)


def test_zero_reference_wheel_uses_shared_absolute_floor_semantics() -> None:
    result = _project(
        ZERO[:8] + (100.0, 0.0, 0.0, 0.0),
        state="P06",
        reference=ZERO,
        delta=ZERO,
    )
    # Shared conformance `allowed_error` uses max(absolute floor, 30%*ref),
    # so a zero reference has 0.05 rad/s allowance rather than 0.015 rad/s.
    assert result.recording_scale_suggestion_full12[8] == pytest.approx(0.05)
    assert result.remaining_recording_envelope_diagnostic_full12[8] == pytest.approx(0.05)
    assert result.safe_projected_residual_full12[8] == pytest.approx(
        2.0943951023931953
    )
    assert result.recording_envelope_exceeded_full12[8] is True


def test_servo_wheel_rate_and_absolute_limits_are_separate_stages() -> None:
    rate = _project((100.0,) * 12, dt=1.0 / 120.0)
    assert rate.safe_projected_residual_full12[0] == pytest.approx(1.25)
    assert "residual_rate_limit" in rate.clipping_stages

    servo_nominal = (130.0,) + ZERO[1:]
    servo_reference = servo_nominal
    absolute_servo = _project(
        (100.0,) + ZERO[1:],
        nominal=servo_nominal,
        reference=servo_reference,
        delta=(100.0,) + ZERO[1:],
    )
    assert absolute_servo.applied_action_full12[0] == 133.0
    assert "joint_safety_margin_or_wheel_speed_limit" in absolute_servo.clipping_stages

    wheel_nominal = ZERO[:8] + (2.0, 0.0, 0.0, 0.0)
    absolute_wheel = _project(
        ZERO[:8] + (100.0, 0.0, 0.0, 0.0),
        state="P06",
        nominal=wheel_nominal,
        reference=wheel_nominal,
        delta=ZERO,
    )
    assert absolute_wheel.applied_action_full12[8] == pytest.approx(2.0943951023931953)
    assert "joint_safety_margin_or_wheel_speed_limit" in absolute_wheel.clipping_stages


@pytest.mark.parametrize(
    ("channel", "nominal_value", "raw_value"),
    (
        (0, -134.0, -100.0),
        (1, -59.0, -100.0),
        (0, 134.0, 100.0),
        (1, 209.0, 100.0),
    ),
)
def test_margin_projection_clips_only_outward_residual_and_never_moves_nominal(
    channel: int, nominal_value: float, raw_value: float
) -> None:
    nominal = list(ZERO)
    raw = list(ZERO)
    nominal[channel] = nominal_value
    raw[channel] = raw_value

    result = _project(tuple(raw), nominal=tuple(nominal), reference=tuple(nominal))

    assert result.zero_residual_fast_path is False
    assert result.rate_projected_residual_full12[channel] != 0.0
    assert result.limit_projected_residual_full12[channel] == 0.0
    assert bitwise_full12_equal(result.applied_action_full12, tuple(nominal))
    assert "joint_safety_margin_or_wheel_speed_limit" in result.clipping_stages


def test_tiny_residual_is_not_falsely_reported_as_limit_clipping() -> None:
    nominal = (9.0,) + ZERO[1:]
    result = _project(
        (1.0e-9,) + ZERO[1:],
        nominal=nominal,
        reference=nominal,
    )

    assert result.limit_projected_residual_full12 == (
        result.rate_projected_residual_full12
    )
    assert "joint_safety_margin_or_wheel_speed_limit" not in result.clipping_stages


def test_phase_mask_is_reapplied_after_rate_limit_with_nonzero_prior() -> None:
    result = _project(
        (100.0,) * 12,
        state="P10",
        previous=(2.0,) * 12,
        dt=1.0 / 120.0,
    )
    assert result.effective_action_mask_full12 == (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    assert all(
        value == 0.0
        for index, value in enumerate(result.rate_projected_residual_full12)
        if index != 7
    )
    assert "phase_active_mask_post_rate" in result.clipping_stages


def test_current_phase_scale_is_reapplied_after_rate_limit_with_untrusted_prior() -> None:
    from wlr50_clean.ppo.phase_action_masks_v2 import build_action_projector_v2

    projector = build_action_projector_v2()
    result = projector.project(
        ZERO,
        state_id="P02",
        nominal_action_full12=ZERO,
        reference_action_full12=ZERO,
        reference_delta_full12=ZERO,
        previous_projected_residual_full12=(10.0,) + ZERO[1:],
        runtime_action_mask_full12=(1,) * 12,
        dt_s=1.0 / 120.0,
    )

    assert result.rate_projected_residual_full12[0] > 1.5
    assert result.phase_scale_projected_residual_full12[0] == pytest.approx(1.5)
    assert result.safe_projected_residual_full12[0] == pytest.approx(1.5)
    assert "phase_residual_scale_cap_post_rate" in result.clipping_stages


def test_hard_safety_is_last_and_can_stop_wheels_or_override() -> None:
    nominal = ZERO[:8] + (1.0, 1.0, 1.0, 1.0)
    stopped = _project(
        ZERO,
        nominal=nominal,
        reference=nominal,
        safety=SafetyProjection(force_wheels_zero=True, reason="hard abort"),
    )
    assert stopped.applied_action_full12[8:] == (0.0,) * 4
    assert stopped.hard_safety_modified is True
    assert stopped.clipping_stages[-1] == "hard_safety"

    override = (1.0,) * 8 + (0.0,) * 4
    overridden = _project(
        (1.0,) * 12,
        safety=SafetyProjection(override_full12=override, reason="controller safe hold"),
    )
    assert overridden.applied_action_full12 == override
    assert overridden.clipping_stages[-1] == "hard_safety"


@pytest.mark.parametrize("flag", ["body_collision_detected", "wheel_only_climb_detected"])
def test_physical_stop_disables_residual_and_forces_wheels_zero(flag: str) -> None:
    nominal = (1.0,) * 8 + (0.5,) * 4
    safety = SafetyProjection(**{flag: True, "reason": flag})
    result = _project(
        (100.0,) * 12,
        nominal=nominal,
        reference=nominal,
        delta=(100.0,) * 12,
        safety=safety,
    )
    assert result.applied_action_full12[:8] == nominal[:8]
    assert result.applied_action_full12[8:] == ZERO[8:]
    assert result.clipping_stages[-1] == "body_collision_or_wheel_only_safety"

    override = (2.0,) * 8 + (1.0,) * 4
    overridden = _project(
        ZERO,
        nominal=nominal,
        reference=nominal,
        safety=SafetyProjection(
            **{flag: True, "override_full12": override, "reason": flag}
        ),
    )
    assert overridden.applied_action_full12[:8] == override[:8]
    assert overridden.applied_action_full12[8:] == ZERO[8:]


def test_hard_safety_override_must_be_finite_and_inside_actuator_limits() -> None:
    with pytest.raises(ActionProjectionError, match="NaN"):
        SafetyProjection(override_full12=(float("nan"),) + ZERO[1:])
    with pytest.raises(ActionProjectionError, match="absolute actuator"):
        SafetyProjection(override_full12=(136.0,) + ZERO[1:])
    with pytest.raises(ActionProjectionError, match="absolute actuator"):
        SafetyProjection(override_full12=ZERO[:8] + (2.2, 0.0, 0.0, 0.0))


def test_nonfinite_or_out_of_limit_nominal_is_rejected() -> None:
    with pytest.raises(ActionProjectionError, match="NaN"):
        _project((float("nan"),) + ZERO[1:])
    with pytest.raises(ActionProjectionError, match="absolute limit"):
        _project(ZERO, nominal=(999.0,) + ZERO[1:])
