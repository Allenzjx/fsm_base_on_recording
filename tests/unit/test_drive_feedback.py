from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wlr50_clean.fsm.drive_feedback import ReferenceBoundedDriveFeedback
from wlr50_clean.reference.motion_contract import load_motion_contract
from wlr50_clean.sensing.observation import (
    BodyContactObservation,
    CollisionRole,
    ContactClass,
    PairContactObservation,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BIAS_SEGMENTS = (
    (860, 860, 0.68),
    (861, 861, 0.33),
    (862, 864, 0.68),
    (865, 865, 0.15),
    (866, 867, 1.03),
    (868, 868, 0.73),
    (870, 870, 0.01),
    (871, 871, 0.40),
)

EXPECTED_NATIVE_AND_FINAL_BY_TICK = (
    (860, -1.07, -0.39),
    (861, -1.07, -0.74),
    (862, -1.07, -0.39),
    (863, -1.07, -0.39),
    (864, 0.0, 0.68),
    (865, 0.0, 0.15),
    (866, 0.0, 1.03),
    (867, 0.0, 1.03),
    (868, 0.0, 0.73),
    (869, 0.0, 0.0),
    (870, 0.0, 0.01),
    (871, 0.0, 0.40),
)


def _actual(channel_index: int, value: float) -> tuple[float, ...]:
    result = [0.0] * 12
    result[channel_index] = value
    return tuple(result)


def _rear_left_contact(
    *,
    contact_class: str = "AIR",
    ground_verified: object = True,
    ground_active: object = False,
    obstacle_verified: object = True,
    obstacle_active: object = False,
    body_name: str = "rear_left_wheel",
) -> dict[str, object]:
    return {
        "body_name": body_name,
        "contact_class": contact_class,
        "ground": {
            "pair_verified": ground_verified,
            "active": ground_active,
        },
        "obstacle": {
            "pair_verified": obstacle_verified,
            "active": obstacle_active,
        },
    }


def _immutable_rear_left_air() -> BodyContactObservation:
    def pair(other_body: str) -> PairContactObservation:
        return PairContactObservation(
            sensor_body="rear_left_wheel",
            other_body=other_body,
            active=False,
            force_w_n=(0.0, 0.0, 0.0),
            normal_force_n=0.0,
            tangential_force_n=0.0,
            contact_point_w_m=None,
            force_history_w_n=(),
            active_history=(),
            consecutive_active_ticks=0,
            source="unit_test_exact_pair",
            pair_verified=True,
        )

    return BodyContactObservation(
        body_name="rear_left_wheel",
        role=CollisionRole.WHEEL,
        contact_class=ContactClass.AIR,
        ground=pair("ground_plane"),
        obstacle=pair("obstacle"),
    )


def _trigger(
    feedback: ReferenceBoundedDriveFeedback,
    *,
    contact: object | None,
):
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    result = None
    for probe in spec.probe_samples:
        result = feedback.update(
            state_id="P09",
            motion_tick_index=probe.motion_tick,
            actual_full12=_actual(
                spec.probe_channel_index,
                probe.reference_actual_deg - spec.lag_threshold_deg,
            ),
            spec=spec,
            rear_left_contact=contact,
        )
    assert result is not None
    return spec, result


def test_p09_verified_air_adds_exact_capped_rear_left_alignment() -> None:
    feedback = ReferenceBoundedDriveFeedback()
    spec, trigger = _trigger(feedback, contact=_rear_left_contact())
    alignment = spec.contact_alignment
    assert alignment is not None
    trigger_log = trigger.as_dict()["contact_alignment"]
    assert isinstance(trigger_log, dict)
    assert trigger_log["just_triggered"] is True
    assert trigger_log["condition_passed"] is True
    assert trigger_log["contact_evidence"] == {
        "wheel_body": "rear_left_wheel",
        "contact_class": "AIR",
        "ground_pair_verified": True,
        "ground_active": False,
        "obstacle_pair_verified": True,
        "obstacle_active": False,
    }

    by_tick = {
        tick: feedback.update(
            state_id="P09",
            motion_tick_index=tick,
            actual_full12=_actual(spec.probe_channel_index, 0.0),
            spec=spec,
            rear_left_contact=None,
        )
        for tick in range(860, 873)
    }
    for tick in range(860, 871):
        assert by_tick[tick].bias_full12[4] == pytest.approx(-1.185)
        assert by_tick[tick].bias_full12[5] == pytest.approx(-1.455)
        assert by_tick[tick].as_dict()["contact_alignment"][
            "active_schedule_stage"
        ] == "full_bias"
    assert by_tick[871].bias_full12[4] == 0.0
    assert by_tick[871].bias_full12[5] == pytest.approx(-0.205)
    assert by_tick[871].as_dict()["contact_alignment"][
        "active_schedule_stage"
    ] == "release_ramp"
    assert by_tick[872].bias_full12 == pytest.approx((0.0,) * 12)
    assert by_tick[872].as_dict()["contact_alignment"][
        "active_schedule_stage"
    ] is None
    for tick, native_wheel, final_wheel in EXPECTED_NATIVE_AND_FINAL_BY_TICK:
        assert by_tick[tick].bias_full12[8] == pytest.approx(
            final_wheel - native_wheel
        )


def test_p09_contact_gate_accepts_immutable_body_contact_observation() -> None:
    feedback = ReferenceBoundedDriveFeedback()
    spec, trigger = _trigger(feedback, contact=_immutable_rear_left_air())

    assert trigger.as_dict()["contact_alignment"]["condition_passed"] is True
    active = feedback.update(
        state_id="P09",
        motion_tick_index=860,
        actual_full12=_actual(spec.probe_channel_index, 0.0),
        spec=spec,
    )
    assert active.bias_full12[4:6] == pytest.approx((-1.185, -1.455))


@pytest.mark.parametrize(
    "contact",
    (
        None,
        {},
        _rear_left_contact(contact_class="GROUND"),
        _rear_left_contact(ground_verified=False),
        _rear_left_contact(ground_verified=1),
        _rear_left_contact(ground_active=True),
        _rear_left_contact(obstacle_verified=False),
        _rear_left_contact(obstacle_active=True),
        _rear_left_contact(body_name="rear_right_wheel"),
    ),
)
def test_p09_contact_alignment_fails_closed_without_verified_air(
    contact: object | None,
) -> None:
    feedback = ReferenceBoundedDriveFeedback()
    spec, trigger = _trigger(feedback, contact=contact)
    nested = trigger.as_dict()["contact_alignment"]
    assert isinstance(nested, dict)
    assert nested["condition_evaluated"] is True
    assert nested["condition_passed"] is False
    assert nested["just_triggered"] is False
    active = feedback.update(
        state_id="P09",
        motion_tick_index=860,
        actual_full12=_actual(spec.probe_channel_index, 0.0),
        spec=spec,
        rear_left_contact=_rear_left_contact(),
    )
    assert active.bias_full12[4:6] == pytest.approx((0.0, 0.0))
    assert active.bias_full12[8] == pytest.approx(0.68)


def test_p09_two_sample_live_deficit_latches_bounded_wheel_rebound() -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    feedback = ReferenceBoundedDriveFeedback()
    first, second = spec.probe_samples

    probe_a = feedback.update(
        state_id="P09",
        motion_tick_index=first.motion_tick,
        actual_full12=_actual(
            spec.probe_channel_index,
            first.reference_actual_deg - spec.lag_threshold_deg,
        ),
        spec=spec,
    )
    probe_b = feedback.update(
        state_id="P09",
        motion_tick_index=second.motion_tick,
        actual_full12=_actual(
            spec.probe_channel_index,
            second.reference_actual_deg - spec.lag_threshold_deg,
        ),
        spec=spec,
    )
    armed_gap = [
        feedback.update(
            state_id="P09",
            motion_tick_index=tick,
            actual_full12=_actual(
                spec.probe_channel_index, second.reference_actual_deg
            ),
            spec=spec,
        )
        for tick in range(second.motion_tick + 1, spec.first_bias_tick)
    ]
    active = [
        feedback.update(
            state_id="P09",
            motion_tick_index=tick,
            actual_full12=_actual(
                spec.probe_channel_index, second.reference_actual_deg
            ),
            spec=spec,
        )
        for tick in range(spec.first_bias_tick, spec.last_bias_tick + 1)
    ]
    restored = feedback.update(
        state_id="P09",
        motion_tick_index=spec.teardown_tick,
        actual_full12=_actual(spec.probe_channel_index, second.reference_actual_deg),
        spec=spec,
    )

    assert probe_a.active is False
    assert probe_b.just_triggered is True
    assert probe_b.active is False
    assert len(armed_gap) == 0
    assert all(not item.active for item in armed_gap)
    assert all(item.trigger_tick == second.motion_tick for item in armed_gap)
    assert len(active) == 12
    assert [item.active_segment_index for item in active] == [
        next(
            (
                segment_index
                for segment_index, (first_tick, last_tick, _) in enumerate(
                    EXPECTED_BIAS_SEGMENTS
                )
                if first_tick <= tick <= last_tick
            ),
            None,
        )
        for tick in range(spec.first_bias_tick, spec.last_bias_tick + 1)
    ]
    for item in active:
        if item.active_segment_index is None:
            assert item.tick_index == 869
            assert item.active is False
            assert item.bias_full12 == pytest.approx((0.0,) * 12)
            assert item.logical_bias_rad_s == 0.0
            assert item.active_segment_first_bias_tick is None
            assert item.active_segment_last_bias_tick is None
            continue
        assert item.active is True
        assert item.active_segment_index is not None
        segment = spec.bias_segments[item.active_segment_index]
        assert item.bias_full12[
            spec.correction_channel_index
        ] == pytest.approx(segment.logical_bias_rad_s)
        assert sum(abs(value) for value in item.bias_full12) == pytest.approx(
            abs(segment.logical_bias_rad_s)
        )
        assert item.logical_bias_rad_s == pytest.approx(
            segment.logical_bias_rad_s
        )
        assert item.active_segment_first_bias_tick == segment.first_bias_tick
        assert item.active_segment_last_bias_tick == segment.last_bias_tick
    assert active[0].peak_fraction_of_reference == 0.0
    assert active[0].cumulative_fraction_of_reference == pytest.approx(
        abs(spec.additional_wheel_integral_rad)
        / abs(spec.reference_wheel_integral_rad)
    )
    assert active[0].kind == "pre_endpoint_wheel_rebound_alignment"
    assert active[0].logical_bias_rad_s == 0.68
    assert [
        active[tick - spec.first_bias_tick].logical_bias_rad_s
        for tick in range(spec.first_bias_tick, spec.last_bias_tick + 1)
    ] == pytest.approx(
        [
            next(
                (
                    logical_bias_rad_s
                    for first_tick, last_tick, logical_bias_rad_s in EXPECTED_BIAS_SEGMENTS
                    if first_tick <= tick <= last_tick
                ),
                0.0,
            )
            for tick in range(spec.first_bias_tick, spec.last_bias_tick + 1)
        ]
    )
    assert active[0].bias_segments == spec.bias_segments
    assert active[0].reference_wheel_integral_rad == pytest.approx(
        -0.9060000000012605
    )
    assert active[0].additional_wheel_integral_rad == pytest.approx(
        sum(
            logical_bias_rad_s * (last_tick - first_tick + 1)
            for first_tick, last_tick, logical_bias_rad_s in EXPECTED_BIAS_SEGMENTS
        )
        / 120.0
    )
    assert active[0].resulting_wheel_integral_rad == pytest.approx(
        -0.8526666666679271
    )
    assert active[0].cumulative_fraction_of_reference == pytest.approx(
        0.05886681383361936
    )
    assert active[0].reference_wheel_peak_abs_rad_s == pytest.approx(1.07)
    assert active[0].resulting_wheel_peak_abs_rad_s == pytest.approx(1.07)
    assert active[0].instantaneous_direction_reversal is True
    assert active[0].reason.endswith(
        "bounded first-decision alignment rebound"
    )
    endpoint_tick = round(phase.active_duration_s * 120.0)
    active_by_tick = {item.tick_index: item for item in active}
    for tick, expected_native, expected_final in EXPECTED_NATIVE_AND_FINAL_BY_TICK:
        native = (
            phase.nominal_at(tick / 120.0)[spec.correction_channel_index]
            if tick < endpoint_tick
            else phase.end_full12[spec.correction_channel_index]
        )
        assert native == pytest.approx(expected_native)
        assert native + active_by_tick[tick].bias_full12[
            spec.correction_channel_index
        ] == pytest.approx(expected_final)
    assert restored.bias_full12 == pytest.approx((0.0,) * 12)
    assert restored.active_segment_index is None
    assert restored.logical_bias_rad_s == 0.0
    assert restored.cumulative_fraction_of_reference < 0.15
    first_details = active[0].as_dict()
    restored_details = restored.as_dict()
    assert first_details["bias_segments"] == [
        {
            "first_bias_tick": first_tick,
            "last_bias_tick": last_tick,
            "logical_bias_rad_s": logical_bias_rad_s,
        }
        for first_tick, last_tick, logical_bias_rad_s in EXPECTED_BIAS_SEGMENTS
    ]
    for segment_index, segment in enumerate(spec.bias_segments):
        segment_details = active_by_tick[segment.first_bias_tick].as_dict()
        assert segment_details["active_segment_index"] == segment_index
        assert segment_details["active_segment_first_bias_tick"] == (
            segment.first_bias_tick
        )
        assert segment_details["active_segment_last_bias_tick"] == (
            segment.last_bias_tick
        )
        assert segment_details["logical_bias_rad_s"] == pytest.approx(
            segment.logical_bias_rad_s
        )
    assert restored_details["active_segment_index"] is None
    assert restored_details["logical_bias_rad_s"] == 0.0


def test_trial026_p09_probe_deficit_arms_the_p10_entry_rebound() -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    assert spec.lag_threshold_deg == pytest.approx(0.35)
    feedback = ReferenceBoundedDriveFeedback()
    trial026_observed = (-51.43816064246197, -51.61530159858626)

    results = [
        feedback.update(
            state_id="P09",
            motion_tick_index=probe.motion_tick,
            actual_full12=_actual(spec.probe_channel_index, observed),
            spec=spec,
        )
        for probe, observed in zip(
            spec.probe_samples, trial026_observed, strict=True
        )
    ]

    assert results[0].just_triggered is False
    assert results[1].just_triggered is True
    assert results[1].trigger_tick == 859


def test_p09_wheel_rebound_requires_both_live_deficit_samples() -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    feedback = ReferenceBoundedDriveFeedback()
    first, second = spec.probe_samples

    feedback.update(
        state_id="P09",
        motion_tick_index=first.motion_tick,
        actual_full12=_actual(spec.probe_channel_index, first.reference_actual_deg),
        spec=spec,
    )
    armed = feedback.update(
        state_id="P09",
        motion_tick_index=second.motion_tick,
        actual_full12=_actual(
            spec.probe_channel_index,
            second.reference_actual_deg - spec.lag_threshold_deg,
        ),
        spec=spec,
    )
    active = feedback.update(
        state_id="P09",
        motion_tick_index=spec.first_bias_tick,
        actual_full12=_actual(spec.probe_channel_index, second.reference_actual_deg),
        spec=spec,
    )

    assert armed.just_triggered is False
    assert active.active is False
    assert active.bias_full12 == pytest.approx((0.0,) * 12)


def test_p09_wheel_rebound_is_one_shot_across_a_same_state_retry() -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    feedback = ReferenceBoundedDriveFeedback()

    for probe in spec.probe_samples:
        feedback.update(
            state_id="P09",
            motion_tick_index=probe.motion_tick,
            actual_full12=_actual(
                spec.probe_channel_index,
                probe.reference_actual_deg - spec.lag_threshold_deg,
            ),
            spec=spec,
        )
    assert feedback.update(
        state_id="P09",
        motion_tick_index=spec.first_bias_tick,
        actual_full12=_actual(spec.probe_channel_index, 0.0),
        spec=spec,
    ).active

    feedback.update(
        state_id="P09",
        motion_tick_index=0,
        actual_full12=_actual(spec.probe_channel_index, 0.0),
        spec=spec,
    )
    for probe in spec.probe_samples:
        retried = feedback.update(
            state_id="P09",
            motion_tick_index=probe.motion_tick,
            actual_full12=_actual(
                spec.probe_channel_index,
                probe.reference_actual_deg - spec.lag_threshold_deg,
            ),
            spec=spec,
        )
    assert retried.just_triggered is False
    assert feedback.update(
        state_id="P09",
        motion_tick_index=spec.first_bias_tick,
        actual_full12=_actual(spec.probe_channel_index, 0.0),
        spec=spec,
    ).active is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("kind", "verify_tail_wheel_carry_alignment"),
        ("teardown_tick", 873),
        ("additional_wheel_integral_rad", 0.107),
        ("resulting_wheel_integral_rad", -0.7990000000012605),
        ("cumulative_fraction_of_reference", 0.118101545253699),
        ("reference_wheel_peak_abs_rad_s", 1.06),
        ("resulting_wheel_peak_abs_rad_s", 1.08),
        ("instantaneous_direction_reversal", False),
    ),
)
def test_p09_wheel_rebound_runtime_fails_closed(
    field: str, invalid_value: object
) -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    invalid = replace(spec, **{field: invalid_value})

    with pytest.raises(RuntimeError, match="not a valid wheel rebound"):
        ReferenceBoundedDriveFeedback().update(
            state_id="P09",
            motion_tick_index=spec.probe_samples[0].motion_tick,
            actual_full12=_actual(spec.probe_channel_index, 0.0),
            spec=invalid,
        )


def test_p09_contact_alignment_runtime_contract_fails_closed() -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None and spec.contact_alignment is not None
    invalid = replace(
        spec,
        contact_alignment=replace(spec.contact_alignment, release_tick=872),
    )

    with pytest.raises(RuntimeError, match="not a valid wheel rebound"):
        ReferenceBoundedDriveFeedback().update(
            state_id="P09",
            motion_tick_index=spec.probe_samples[0].motion_tick,
            actual_full12=_actual(spec.probe_channel_index, 0.0),
            spec=invalid,
            rear_left_contact=_rear_left_contact(),
        )


@pytest.mark.parametrize(
    ("segment_index", "field", "invalid_value"),
    (
        (0, "first_bias_tick", 861),
        (0, "last_bias_tick", 861),
        (0, "logical_bias_rad_s", 1.04),
        (1, "first_bias_tick", 860),
        (1, "logical_bias_rad_s", 0.34),
        (2, "last_bias_tick", 863),
        (2, "logical_bias_rad_s", 1.04),
        (3, "first_bias_tick", 864),
        (3, "logical_bias_rad_s", 0.34),
        (4, "last_bias_tick", 866),
        (4, "logical_bias_rad_s", 0.27),
        (5, "last_bias_tick", 869),
        (5, "logical_bias_rad_s", 0.37),
        (6, "first_bias_tick", 869),
        (6, "logical_bias_rad_s", 0.27),
        (7, "last_bias_tick", 872),
        (7, "logical_bias_rad_s", 0.37),
    ),
)
def test_p09_wheel_rebound_runtime_rejects_mutated_segments(
    segment_index: int,
    field: str,
    invalid_value: object,
) -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None
    segments = list(spec.bias_segments)
    segments[segment_index] = replace(
        segments[segment_index], **{field: invalid_value}
    )
    invalid = replace(spec, bias_segments=tuple(segments))

    with pytest.raises(RuntimeError, match="not a valid wheel rebound"):
        ReferenceBoundedDriveFeedback().update(
            state_id="P09",
            motion_tick_index=spec.probe_samples[0].motion_tick,
            actual_full12=_actual(spec.probe_channel_index, 0.0),
            spec=invalid,
        )


def test_p09_wheel_rebound_runtime_rejects_wrong_state_or_probe_reference() -> None:
    phase = load_motion_contract(
        ROOT / "configs" / "recording_motion_contract.json"
    ).phase("P09")
    spec = phase.drive_feedback
    assert spec is not None

    with pytest.raises(RuntimeError, match="not a valid wheel rebound"):
        ReferenceBoundedDriveFeedback().update(
            state_id="P08",
            motion_tick_index=spec.probe_samples[0].motion_tick,
            actual_full12=_actual(spec.probe_channel_index, 0.0),
            spec=spec,
        )

    changed_probe = replace(
        spec.probe_samples[0],
        reference_actual_deg=spec.probe_samples[0].reference_actual_deg + 0.01,
    )
    invalid = replace(
        spec,
        probe_samples=(changed_probe, spec.probe_samples[1]),
    )
    with pytest.raises(RuntimeError, match="not a valid wheel rebound"):
        ReferenceBoundedDriveFeedback().update(
            state_id="P09",
            motion_tick_index=spec.probe_samples[0].motion_tick,
            actual_full12=_actual(spec.probe_channel_index, 0.0),
            spec=invalid,
        )
