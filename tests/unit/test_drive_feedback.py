from __future__ import annotations

from pathlib import Path

import pytest

from wlr50_clean.fsm.drive_feedback import ReferenceBoundedDriveFeedback
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]


def _actual(channel_index: int, value: float) -> tuple[float, ...]:
    result = [0.0] * 12
    result[channel_index] = value
    return tuple(result)


def test_p09_two_sample_live_deficit_latches_bounded_carry_phase_bias() -> None:
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
    active = feedback.update(
        state_id="P09",
        motion_tick_index=spec.first_bias_tick,
        actual_full12=_actual(spec.probe_channel_index, second.reference_actual_deg),
        spec=spec,
    )
    restored = feedback.update(
        state_id="P09",
        motion_tick_index=spec.teardown_tick,
        actual_full12=_actual(spec.probe_channel_index, second.reference_actual_deg),
        spec=spec,
    )

    assert probe_a.active is False
    assert probe_b.just_triggered is True
    assert probe_b.active is False
    assert active.bias_full12[spec.correction_channel_index] == pytest.approx(1.25)
    assert active.cumulative_fraction_of_reference == pytest.approx(2.5 / 19.4)
    assert restored.bias_full12 == pytest.approx((0.0,) * 12)
    assert restored.cumulative_fraction_of_reference < 0.15


def test_p09_carry_phase_bias_requires_both_live_deficit_samples() -> None:
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


def test_p09_carry_phase_bias_is_one_shot_across_a_same_state_retry() -> None:
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
