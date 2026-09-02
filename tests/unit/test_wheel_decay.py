from __future__ import annotations

from wlr50_clean.fsm.wheel_decay import WheelDecayDebounce


def test_zero_command_readback_is_required_before_decay_can_accumulate() -> None:
    debounce = WheelDecayDebounce()

    nonzero = debounce.update(
        sim_time_s=0.0,
        measured_velocity_rad_s=(0.0,) * 4,
        commanded_velocity_rad_s=(0.1, 0.0, 0.0, 0.0),
        threshold_rad_s=0.25,
        debounce_s=0.5,
    )
    zero = debounce.update(
        sim_time_s=0.5,
        measured_velocity_rad_s=(0.0,) * 4,
        commanded_velocity_rad_s=(0.0,) * 4,
        threshold_rad_s=0.25,
        debounce_s=0.5,
    )

    assert nonzero.eligible is False
    assert zero.eligible is True
    assert zero.passed is False
    assert zero.stable_for_s == 0.0


def test_any_physics_tick_spike_resets_the_shared_decay_state() -> None:
    debounce = WheelDecayDebounce()
    status = None
    for tick in range(98):
        velocity = (0.26, 0.0, 0.0, 0.0) if tick == 36 else (0.0,) * 4
        status = debounce.update(
            sim_time_s=tick / 120.0,
            measured_velocity_rad_s=velocity,
            commanded_velocity_rad_s=(0.0,) * 4,
            threshold_rad_s=0.25,
            debounce_s=0.5,
        )

    assert status is not None
    assert status.passed is True
    assert status.stable_since_s == 37.0 / 120.0
    assert status.stable_for_s == 0.5
