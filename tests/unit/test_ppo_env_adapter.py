from __future__ import annotations

import json
from dataclasses import replace

import pytest

from wlr50_clean.ppo.action_projection import bitwise_full12_equal
from wlr50_clean.ppo.episode_logger import EpisodeLogError, EpisodeLogger
from wlr50_clean.ppo.observation_schema import PPOObservationFrame
from wlr50_clean.ppo.ppo_env_adapter import (
    AuthoritativeFrame,
    PPOEnvAdapter,
    PPOEnvError,
    load_domain_randomization_config,
)
from wlr50_clean.ppo.reward_terms import RewardSignals
from wlr50_clean.ppo.termination import TerminationSignals


ZERO = (0.0,) * 12


def _observation(tick: int, previous=ZERO) -> PPOObservationFrame:
    return PPOObservationFrame(
        state_id="P01",
        macro_phase=1,
        phase_progress=min(1.0, tick / 16.0),
        joint_position_error8=(0.0,) * 8,
        joint_velocity8=(0.0,) * 8,
        wheel_velocity4=(0.0,) * 4,
        wheel_contact_code4=(1.0,) * 4,
        leg_history12=(0.0,) * 12,
        body_orientation_wxyz4=(1.0, 0.0, 0.0, 0.0),
        body_angular_velocity3=(0.0,) * 3,
        obstacle_relative_geometry9=(0.0,) * 9,
        full_body_com3=(0.0, 0.0, 0.1),
        support_diagnostics4=(0.01, 1.0, 4.0, 1.0),
        previous_action_full12=previous,
    )


class FakeBackend:
    def __init__(self, success_tick: int = 16):
        self.success_tick = success_tick
        self.tick = 0
        self.actions = []
        self.reset_calls = []

    def _frame(self, previous=ZERO):
        nominal = (self.tick / 100.0,) + ZERO[1:]
        info = {
            "environment_hash": "environment-sha256",
            "robot_asset_hash": "robot-sha256",
            "initial_root_state": [0.0] * 13,
            "initial_joint_state": [0.0] * 24,
            "obstacle_pose": [0.0, 0.0, 0.025],
            "controller_hash": "controller-sha256",
            "motion_contract_hash": "contract-sha256",
        }
        return AuthoritativeFrame(
            physics_tick=self.tick,
            sim_time_s=self.tick / 120.0,
            state_id="P01",
            macro_phase=1,
            phase_progress=min(1.0, self.tick / 16.0),
            observation=_observation(self.tick, previous),
            nominal_action_full12=nominal,
            reference_action_full12=nominal,
            reference_delta_full12=(10.0,) * 8 + ZERO[8:],
            action_mask_full12=(1,) * 12,
            reward_signals=RewardSignals(
                forward_progress_delta_m=0.001,
                phase_progress_delta=1.0 / 16.0,
                support_margin_m=0.01,
                support_valid=True,
                task_success=self.tick >= self.success_tick,
            ),
            termination_signals=TerminationSignals(
                success=self.tick >= self.success_tick
            ),
            info=info,
        )

    def reset(self, *, seed, options):
        self.tick = 0
        self.actions = []
        self.reset_calls.append((seed, dict(options)))
        return self._frame()

    def step_physics(self, applied_action_full12):
        action = tuple(applied_action_full12)
        self.actions.append(action)
        self.tick += 1
        return self._frame(action)


def test_env_runs_authoritative_120hz_ticks_at_one_15hz_policy_step(tmp_path) -> None:
    backend = FakeBackend(success_tick=16)
    logger = EpisodeLogger()
    env = PPOEnvAdapter(backend, episode_logger=logger)
    observation, reset_info = env.reset(7)
    assert len(observation) == 85
    assert reset_info["seed"] == 7
    assert reset_info["randomization_enabled"] is False
    assert all(
        value
        == load_domain_randomization_config().hooks[name].baseline
        for name, value in reset_info["randomization_values"].items()
    )

    observation, reward, terminated, truncated, info = env.step(ZERO)
    assert len(backend.actions) == 8
    assert len(observation) == 85
    assert reward > 0.0
    assert terminated is truncated is False
    assert info["physics_ticks_executed"] == 8
    assert info["zero_residual_fast_path_all_ticks"] is True
    assert len(logger.rows) == 1

    observation, reward, terminated, truncated, info = env.step(ZERO)
    assert len(backend.actions) == 16
    assert terminated is True and truncated is False
    assert info["termination_reason"] == "SUCCESS"
    assert len(logger.rows) == 2
    logger.validate_complete()
    assert all(
        bitwise_full12_equal(action, (index / 100.0,) + ZERO[1:])
        for index, action in enumerate(backend.actions)
    )

    output = tmp_path / "baseline.jsonl"
    logger.write_jsonl(output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["terminated"] is False
    assert rows[-1]["termination_reason"] == "SUCCESS"
    assert rows[0]["observation_t_plus_1"] == rows[1]["observation_t"]


def test_env_stops_mid_decision_on_authoritative_hard_or_success_signal() -> None:
    backend = FakeBackend(success_tick=3)
    env = PPOEnvAdapter(backend)
    env.reset(11)
    _, _, terminated, truncated, info = env.step(ZERO)
    assert terminated is True and truncated is False
    assert info["physics_ticks_executed"] == 3
    assert len(backend.actions) == 3
    with pytest.raises(PPOEnvError, match="after episode"):
        env.step(ZERO)


def test_nonzero_policy_output_is_smoothed_at_120hz_and_phase_masked() -> None:
    backend = FakeBackend(success_tick=100)
    env = PPOEnvAdapter(backend)
    env.reset(5)
    env.step((10.0,) * 12)
    assert len(backend.actions) == 8
    active = {0, 1, 4, 8, 9, 10, 11}  # explicit P01 PPO action mask
    for tick, action in enumerate(backend.actions):
        nominal = (tick / 100.0,) + ZERO[1:]
        assert all(
            action[index] == nominal[index]
            for index in range(12)
            if index not in active
        )
    # The servo residual cannot jump directly to its full projected value;
    # it advances on the 120 Hz residual-rate limiter.
    assert backend.actions[0][4] == pytest.approx(1.25)
    assert backend.actions[1][4] > backend.actions[0][4]
    assert backend.actions[2][4] == pytest.approx(3.75, abs=2.0e-8)


def test_reset_is_seeded_and_randomization_remains_disabled() -> None:
    first = PPOEnvAdapter(FakeBackend())
    first_observation, first_info = first.reset(123)
    second = PPOEnvAdapter(FakeBackend())
    second_observation, second_info = second.reset(123)
    third = PPOEnvAdapter(FakeBackend())
    third_observation, third_info = third.reset(124)
    assert first_observation == second_observation
    assert first_info == second_info
    assert first_observation == third_observation
    assert first_info["randomization_values"] == third_info["randomization_values"]
    assert first_info["seed"] != third_info["seed"]
    for name in (
        "environment_hash",
        "robot_asset_hash",
        "initial_root_state",
        "initial_joint_state",
        "obstacle_pose",
        "controller_hash",
        "motion_contract_hash",
    ):
        assert first_info[name] == third_info[name]
    with pytest.raises(PPOEnvError, match="disabled"):
        PPOEnvAdapter(FakeBackend()).reset(1, {"enable_randomization": True})


@pytest.mark.parametrize("fault", ["tick", "time"])
def test_backend_must_advance_exactly_one_120hz_tick(fault) -> None:
    class BadCadenceBackend(FakeBackend):
        def step_physics(self, applied_action_full12):
            frame = super().step_physics(applied_action_full12)
            if fault == "tick":
                return replace(frame, physics_tick=frame.physics_tick + 1)
            return replace(frame, sim_time_s=frame.sim_time_s + 0.001)

    env = PPOEnvAdapter(BadCadenceBackend())
    env.reset(3)
    with pytest.raises(PPOEnvError, match="exactly"):
        env.step(ZERO)


def test_reset_requires_complete_reproducibility_metadata() -> None:
    backend = FakeBackend()
    original = backend._frame

    def incomplete(previous=ZERO):
        return replace(original(previous), info={"environment_hash": "only-one"})

    backend._frame = incomplete
    with pytest.raises(PPOEnvError, match="metadata"):
        PPOEnvAdapter(backend).reset(1)


def test_episode_logger_rejects_incomplete_final_transition() -> None:
    backend = FakeBackend(success_tick=16)
    logger = EpisodeLogger()
    env = PPOEnvAdapter(backend, episode_logger=logger)
    env.reset(2)
    env.step(ZERO)
    with pytest.raises(EpisodeLogError, match="final transition"):
        logger.validate_complete()
