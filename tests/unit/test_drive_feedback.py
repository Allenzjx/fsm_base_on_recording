from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wlr50_clean.fsm.drive_feedback import ReferenceBoundedDriveFeedback
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]


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
    assert len(active) == 20
    assert [item.active_segment_index for item in active] == [0] * 12 + [1] * 8
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
    assert active[0].logical_bias_rad_s == 0.33
    assert active[12].logical_bias_rad_s == 0.17
    assert active[0].bias_segments == spec.bias_segments
    assert active[0].reference_wheel_integral_rad == pytest.approx(
        -0.9060000000012605
    )
    assert active[0].additional_wheel_integral_rad == pytest.approx(
        (0.33 * 12.0 + 0.17 * 8.0) / 120.0
    )
    assert active[0].resulting_wheel_integral_rad == pytest.approx(
        -0.8616666666679271
    )
    assert active[0].cumulative_fraction_of_reference == pytest.approx(
        0.04893303899919609
    )
    assert active[0].reference_wheel_peak_abs_rad_s == pytest.approx(1.07)
    assert active[0].resulting_wheel_peak_abs_rad_s == pytest.approx(1.07)
    assert active[0].instantaneous_direction_reversal is True
    endpoint_tick = round(phase.active_duration_s * 120.0)
    active_by_tick = {item.tick_index: item for item in active}
    primary, secondary = spec.bias_segments
    for tick in range(primary.first_bias_tick, endpoint_tick):
        native = phase.nominal_at(tick / 120.0)[spec.correction_channel_index]
        assert native == pytest.approx(-1.07)
        assert native + active_by_tick[tick].bias_full12[
            spec.correction_channel_index
        ] == pytest.approx(-0.74)
    for tick in range(endpoint_tick, primary.last_bias_tick + 1):
        native = phase.end_full12[spec.correction_channel_index]
        assert native == pytest.approx(0.0)
        assert native + active_by_tick[tick].bias_full12[
            spec.correction_channel_index
        ] == pytest.approx(0.33)
    for tick in range(
        secondary.first_bias_tick, secondary.last_bias_tick + 1
    ):
        native = phase.end_full12[spec.correction_channel_index]
        assert native == pytest.approx(0.0)
        assert native + active_by_tick[tick].bias_full12[
            spec.correction_channel_index
        ] == pytest.approx(0.17)
    assert restored.bias_full12 == pytest.approx((0.0,) * 12)
    assert restored.active_segment_index is None
    assert restored.logical_bias_rad_s == 0.0
    assert restored.cumulative_fraction_of_reference < 0.15
    first_details = active[0].as_dict()
    second_details = active[12].as_dict()
    restored_details = restored.as_dict()
    assert first_details["bias_segments"] == [
        {
            "first_bias_tick": 860,
            "last_bias_tick": 871,
            "logical_bias_rad_s": 0.33,
        },
        {
            "first_bias_tick": 872,
            "last_bias_tick": 879,
            "logical_bias_rad_s": 0.17,
        },
    ]
    assert first_details["active_segment_index"] == 0
    assert first_details["logical_bias_rad_s"] == pytest.approx(0.33)
    assert second_details["active_segment_index"] == 1
    assert second_details["logical_bias_rad_s"] == pytest.approx(0.17)
    assert restored_details["active_segment_index"] is None
    assert restored_details["logical_bias_rad_s"] == 0.0


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
        ("teardown_tick", 879),
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
        (0, "last_bias_tick", 870),
        (0, "logical_bias_rad_s", 1.07),
        (1, "first_bias_tick", 873),
        (1, "last_bias_tick", 880),
        (1, "logical_bias_rad_s", 0.33),
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
