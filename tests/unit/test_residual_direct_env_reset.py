from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo.action_projection import SafetyProjection
from wlr50_clean.ppo.residual_direct_env import (
    ResidualDirectEnvError,
    ResidualEpisodeEnv,
)
from wlr50_clean.ppo.termination import TerminationSignals


ZERO12 = (0.0,) * 12
OBSERVATION125 = (0.0,) * 125


class _ResetBackend:
    def __init__(self, frame: SimpleNamespace) -> None:
        self.frame = frame

    def reset(self, *, seed: int, options: dict[str, object]) -> SimpleNamespace:
        return self.frame


class _ResetEpisode(ResidualEpisodeEnv):
    def _encode(self, frame: SimpleNamespace) -> tuple[float, ...]:
        return OBSERVATION125


def _frame(
    *,
    termination_signals: TerminationSignals | None = None,
    safety_projection: SafetyProjection | None = None,
    info: dict[str, object] | None = None,
    sim_time_s: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        physics_tick=0,
        sim_time_s=sim_time_s,
        state_id="P08",
        macro_phase=8,
        termination_signals=termination_signals or TerminationSignals(),
        safety_projection=safety_projection or SafetyProjection(),
        info={} if info is None else info,
    )


def _assert_reset_rejected(frame: SimpleNamespace, pattern: str) -> None:
    env = _ResetEpisode(_ResetBackend(frame))
    old_frame = SimpleNamespace(state_id="P01")
    old_observation = (1.0,) * 125
    env.frame = old_frame
    env.observation = old_observation
    env.done = True

    with pytest.raises(ResidualDirectEnvError, match=pattern):
        env.reset(seed=1003)

    # Validation happens before publishing the new frame or clearing the old
    # episode's terminal state.
    assert env.frame is old_frame
    assert env.observation is old_observation
    assert env.done is True


@pytest.mark.parametrize(
    ("field_name", "label"),
    (
        ("success", "SUCCESS"),
        ("body_collision", "BODY_COLLISION"),
        ("wheel_only_climb", "WHEEL_ONLY_CLIMB"),
        ("fall", "FALL"),
        ("nan_inf", "NAN_INF"),
        ("hard_joint_limit", "HARD_JOINT_LIMIT"),
        ("physics_explosion", "PHYSICS_EXPLOSION"),
        ("timeout", "TIMEOUT"),
    ),
)
def test_reset_rejects_every_terminal_signal_before_clearing_done(
    field_name: str, label: str
) -> None:
    signals = replace(TerminationSignals(), **{field_name: True})
    _assert_reset_rejected(
        _frame(termination_signals=signals),
        rf"terminal reset frame: .*{label}",
    )


def test_reset_rejects_elapsed_timeout_even_if_backend_omits_timeout_signal() -> None:
    _assert_reset_rejected(
        _frame(sim_time_s=200.0),
        r"terminal reset frame: .*TIMEOUT",
    )


@pytest.mark.parametrize(
    ("safety", "label"),
    (
        (SafetyProjection(residual_enabled=False), "residual_disabled"),
        (
            SafetyProjection(channel_mask_full12=(0,) + (1,) * 11),
            "channel_mask",
        ),
        (SafetyProjection(force_wheels_zero=True), "wheels_forced_zero"),
        (SafetyProjection(body_collision_detected=True), "body_collision"),
        (
            SafetyProjection(wheel_only_climb_detected=True),
            "wheel_only_climb",
        ),
        (SafetyProjection(override_full12=ZERO12), "override"),
        (SafetyProjection(reason="hard abort"), "reason"),
    ),
)
def test_reset_rejects_every_unsafe_safety_projection(
    safety: SafetyProjection, label: str
) -> None:
    _assert_reset_rejected(
        _frame(safety_projection=safety),
        rf"unsafe reset SafetyProjection: .*{label}",
    )


@pytest.mark.parametrize(
    "info",
    (
        {"controller_lifecycle": "WAIT_ENTRY"},
        {"controller_task_result": "SUCCESS"},
        {"controller_task_result": "INCOMPLETE_CONTROLLER_BLOCKED"},
        {"controller_termination": SimpleNamespace(result="SUCCESS")},
        {"first_blocker": {"name": "entry guard"}},
        {"controller_blocked": True},
        {"termination_mapping": {"controller_result": "SAFETY_ABORT"}},
        {"termination_mapping": {"first_blocker": {"name": "watchdog"}}},
        {
            "termination_mapping": {
                "controller_blocked_encoded_as_truncation": True
            }
        },
        {
            "raw_controller_frame": SimpleNamespace(
                lifecycle="EXECUTE_MOTION",
                termination=None,
                first_blocker={"name": "stale blocker"},
            )
        },
    ),
)
def test_reset_rejects_nonexecuting_or_terminal_controller_metadata(
    info: dict[str, object],
) -> None:
    pattern = (
        "not EXECUTE_MOTION"
        if info.get("controller_lifecycle") == "WAIT_ENTRY"
        else "controller termination or blocker state"
    )
    _assert_reset_rejected(_frame(info=info), pattern)


def test_reset_accepts_running_execute_controller_metadata() -> None:
    raw_controller_frame = SimpleNamespace(
        lifecycle="EXECUTE_MOTION",
        termination=None,
        first_blocker=None,
    )
    frame = _frame(
        info={
            "controller_lifecycle": "EXECUTE_MOTION",
            "controller_task_result": "RUNNING",
            "controller_termination": None,
            "raw_controller_frame": raw_controller_frame,
            "termination_mapping": {
                "controller_result": None,
                "controller_reason": None,
                "controller_details": {},
                "first_blocker": {},
                "active_sources": (),
                "primary_source": None,
            },
        }
    )
    env = _ResetEpisode(_ResetBackend(frame))
    env.done = True

    observation, returned_info = env.reset(seed=1003)

    assert observation == OBSERVATION125
    assert returned_info["controller_lifecycle"] == "EXECUTE_MOTION"
    assert env.frame is frame
    assert env.done is False
