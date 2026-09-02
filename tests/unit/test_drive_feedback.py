from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wlr50_clean.fsm.drive_feedback import ReferenceBoundedDriveFeedback
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BIAS_SEGMENTS = (
    (860, 860, 0.68),
    (861, 861, 0.33),
    (862, 864, 0.68),
    (865, 865, 0.23),
    (866, 867, 1.03),
    (868, 869, 0.73),
    (870, 870, 0.01),
    (871, 871, 0.01),
)

EXPECTED_NATIVE_AND_FINAL_BY_TICK = (
    (860, -1.07, -0.39),
    (861, -1.07, -0.74),
    (862, -1.07, -0.39),
    (863, -1.07, -0.39),
    (864, 0.0, 0.68),
    (865, 0.0, 0.23),
    (866, 0.0, 1.03),
    (867, 0.0, 1.03),
    (868, 0.0, 0.73),
    (869, 0.0, 0.73),
    (870, 0.0, 0.01),
    (871, 0.0, 0.01),
)


def _actual(channel_index: int, value: float) -> tuple[float, ...]:
    result = [0.0] * 12
    result[channel_index] = value
    return tuple(result)


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
    assert all(item.active for item in active)
    assert len(active) == 12
    assert [item.active_segment_index for item in active] == [
        segment_index
        for segment_index, (first_tick, last_tick, _) in enumerate(
            EXPECTED_BIAS_SEGMENTS
        )
        for _ in range(first_tick, last_tick + 1)
    ]
    for item in active:
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
            logical_bias_rad_s
            for first_tick, last_tick, logical_bias_rad_s in EXPECTED_BIAS_SEGMENTS
            for _ in range(first_tick, last_tick + 1)
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
        -0.8491666666679272
    )
    assert active[0].cumulative_fraction_of_reference == pytest.approx(
        0.06272994849145062
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
        (5, "last_bias_tick", 868),
        (5, "logical_bias_rad_s", 0.37),
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
