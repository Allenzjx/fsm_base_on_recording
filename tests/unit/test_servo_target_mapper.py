from __future__ import annotations

import math

import pytest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.infrastructure.servo_target_mapper import (
    ServoTargetMapper,
    tracking_correction_step,
)


FL_KNEE = SERVO_ORDER.index("front_left_knee")


def _standing() -> dict[str, float]:
    return {name: 0.0 for name in SERVO_ORDER}


def _rad(values_deg: list[float]) -> tuple[float, ...]:
    return tuple(math.radians(value) for value in values_deg)


def test_source_full12_request_slews_servo_but_not_logical_target() -> None:
    mapper = ServoTargetMapper(_standing())
    requested = [0.0] * 8
    requested[FL_KNEE] = -22.9
    mapping = mapper.advance(
        requested,
        _rad([0.0] * 8),
        tracking_servo_names=SERVO_ORDER,
    )
    assert mapping.requested_command_deg[FL_KNEE] == pytest.approx(-22.9)
    assert mapping.applied_drive_command_deg[FL_KNEE] == pytest.approx(-1.25)
    assert mapping.nominal_target_reached[FL_KNEE] is False
    assert all(mapping.tracking_active)


def test_tracking_correction_uses_command_sign_and_sample_slew() -> None:
    assert tracking_correction_step(
        actual_error_deg=2.0,
        command_sign=1.0,
        previous_correction_deg=0.0,
        maximum_delta_deg=1.25,
    ) == pytest.approx(1.25)
    assert tracking_correction_step(
        actual_error_deg=2.0,
        command_sign=-1.0,
        previous_correction_deg=0.0,
        maximum_delta_deg=1.25,
    ) == pytest.approx(-1.25)


def test_changed_nominal_does_not_expand_explicit_tracking_segment() -> None:
    mapper = ServoTargetMapper(_standing())
    requested = [0.0] * 8
    requested[FL_KNEE] = -5.0
    mapping = mapper.advance(requested, _rad([0.0] * 8))
    assert mapping.applied_drive_command_deg[FL_KNEE] == pytest.approx(-1.25)
    assert mapping.nominal_target_reached[FL_KNEE] is False
    assert mapping.tracking_active[FL_KNEE] is False
    assert mapping.feedback_sampled is False


def test_p03_bias_freezes_through_p04_and_p05_restarts_from_drive_target() -> None:
    mapper = ServoTargetMapper(_standing())
    zero = [0.0] * 8
    for _ in range(180):
        mapper.advance(zero, _rad(zero))

    requested = [0.0] * 8
    requested[FL_KNEE] = -22.9
    measured = [0.0] * 8
    measured[FL_KNEE] = -24.6
    mapping = None
    # Feedback samples at global ticks 200..228 advance +1.25 degrees each,
    # reproducing the mature saturated +10 degree P03 drive bias.
    for _ in range(49):
        mapping = mapper.advance(
            requested,
            _rad(measured),
            tracking_servo_names=SERVO_ORDER,
        )
    assert mapping is not None
    assert mapping.nominal_target_reached[FL_KNEE] is True
    assert mapping.tracking_compensation_deg[FL_KNEE] == pytest.approx(10.0)
    assert mapping.applied_drive_command_deg[FL_KNEE] == pytest.approx(-12.9)

    for _ in range(576):
        mapping = mapper.advance(requested, _rad(measured), tracking_servo_names=())
    assert mapping.tracking_active[FL_KNEE] is False
    assert mapping.tracking_compensation_deg[FL_KNEE] == pytest.approx(10.0)
    assert mapping.applied_drive_command_deg[FL_KNEE] == pytest.approx(-12.9)

    p05 = list(requested)
    p05[FL_KNEE] = -26.1
    mapping = mapper.advance(
        p05,
        _rad(measured),
        tracking_servo_names=("front_left_knee",),
    )
    assert mapping.tracking_compensation_deg[FL_KNEE] == pytest.approx(0.0)
    assert mapping.applied_drive_command_deg[FL_KNEE] == pytest.approx(-14.15)
    assert mapping.nominal_target_reached[FL_KNEE] is False
