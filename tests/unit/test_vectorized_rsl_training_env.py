from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Mapping

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo.action_projection import SafetyProjection
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES
from wlr50_clean.ppo.residual_direct_env import ResidualEpisodeEnv
from wlr50_clean.ppo.termination import TerminationSignals
from wlr50_clean.ppo.vectorized_isaac_backend import BatchedAuthoritativeFrame
from wlr50_clean.ppo.vectorized_residual_env import (
    VECTOR_BATCH_RESET_BARRIER_REASON,
    VectorizedResidualEnvError,
    VectorizedRslResidualEnv,
)


ZERO12 = (0.0,) * 12
NUM_ENVS = 8


@dataclass(frozen=True)
class _Reward:
    total: float
    weighted_dense: Mapping[str, float] = field(
        default_factory=lambda: {family: 0.0 for family in DENSE_FAMILIES}
    )


class _FakeVectorBackend:
    def __init__(
        self,
        *,
        num_envs: int = NUM_ENVS,
        terminal_tick: int | None = None,
        shared_controller: bool = False,
    ) -> None:
        self.num_envs = num_envs
        shared = object()
        self.controllers = tuple(
            shared if shared_controller else object() for _ in range(num_envs)
        )
        self.readers = tuple(object() for _ in range(num_envs))
        self.terminal_tick = terminal_tick
        self.reset_calls: list[tuple[int, ...]] = []
        self.action_batches: list[tuple[tuple[float, ...], ...]] = []
        self.tick = 0
        self.global_steps = 0
        self.writes = 0
        self.captures = 0
        self._seeds = tuple(range(num_envs))

    def _frames(self):
        result = []
        for row in range(self.num_envs):
            phase_id = f"P{row + 1:02d}"
            result.append(
                SimpleNamespace(
                    physics_tick=self.tick,
                    sim_time_s=self.tick / 120.0,
                    state_id=phase_id,
                    macro_phase=row + 1,
                    phase_progress=min(1.0, self.tick / 100.0),
                    nominal_action_full12=ZERO12,
                    reference_action_full12=ZERO12,
                    reference_delta_full12=ZERO12,
                    safety_projection=SafetyProjection(),
                    termination_signals=TerminationSignals(
                        success=(
                            row == 0
                            and self.terminal_tick is not None
                            and self.tick >= self.terminal_tick
                        )
                    ),
                    info={
                        "seed": self._seeds[row],
                        "env_index": row,
                        "independent_fsm_per_environment": True,
                        "in_episode_root_pose_writes": 0,
                        "in_episode_root_velocity_writes": 0,
                        "recording_accesses": 0,
                    },
                )
            )
        return tuple(result)

    def _batch(self):
        return BatchedAuthoritativeFrame(
            physics_tick=self.tick,
            sim_time_s=self.tick / 120.0,
            frames=self._frames(),
            global_physics_step_count=self.global_steps,
            batched_articulation_write_count=self.writes,
            exact_pair_capture_count=self.captures,
        )

    def reset_all(self, *, seeds):
        self._seeds = tuple(int(value) for value in seeds)
        self.reset_calls.append(self._seeds)
        self.tick = 0
        self.global_steps += 180
        self.writes = 180
        self.captures = 180
        return self._batch()

    def step_physics_batch(self, applied_actions_full12):
        actions = tuple(tuple(float(value) for value in row) for row in applied_actions_full12)
        assert len(actions) == self.num_envs
        assert all(len(row) == 12 for row in actions)
        self.action_batches.append(actions)
        self.tick += 1
        self.global_steps += 1
        self.writes += 1
        self.captures += 1
        return self._batch()


@pytest.fixture
def lightweight_row_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ResidualEpisodeEnv,
        "_encode",
        lambda self, frame: (float(frame.info["env_index"]),) * 125,
    )
    monkeypatch.setattr(
        ResidualEpisodeEnv,
        "_reward",
        lambda self, start, end, residual: _Reward(
            total=float(end.info["env_index"])
        ),
    )


def test_true_batch_step_returns_independent_observation_reward_and_done_rows(
    lightweight_row_kernel,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _FakeVectorBackend()
    env = VectorizedRslResidualEnv(
        backend,
        seeds=tuple(range(1001, 1017)),
        device="cpu",
    )

    observations, rewards, dones, extras = env.step(torch.zeros((NUM_ENVS, 12)))

    assert len(backend.action_batches) == 8
    assert all(len(batch) == NUM_ENVS for batch in backend.action_batches)
    assert tuple(observations["policy"].shape) == (NUM_ENVS, 125)
    assert observations["policy"][:, 0].tolist() == pytest.approx(list(range(NUM_ENVS)))
    assert rewards.tolist() == pytest.approx(list(range(NUM_ENVS)))
    assert dones.tolist() == [False] * NUM_ENVS
    assert extras["time_outs"].tolist() == [False] * NUM_ENVS
    assert tuple(info["physics_ticks_executed"] for info in env.last_step_infos) == (8,) * NUM_ENVS
    assert tuple(info["physics_tick"] for info in env.last_step_infos) == (8,) * NUM_ENVS


def test_terminal_row_causes_honest_synchronous_full_reset_barrier(
    lightweight_row_kernel,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _FakeVectorBackend(terminal_tick=3)
    env = VectorizedRslResidualEnv(
        backend,
        seeds=tuple(range(1001, 1017)),
        device="cpu",
    )

    _, _, dones, extras = env.step(torch.zeros((NUM_ENVS, 12)))

    assert len(backend.action_batches) == 3
    assert dones.tolist() == [True] * NUM_ENVS
    assert extras["time_outs"].tolist() == [False] + [True] * (NUM_ENVS - 1)
    assert backend.reset_calls == [
        tuple(range(1001, 1009)),
        tuple(range(1009, 1017)),
    ]
    assert len(env.completed_episodes) == NUM_ENVS
    assert env.completed_episodes[0]["termination_reason"] == "SUCCESS"
    assert env.completed_episodes[0]["vector_batch_reset_peer"] is False
    assert all(
        row["termination_reason"] == VECTOR_BATCH_RESET_BARRIER_REASON
        and row["vector_batch_reset_peer"] is True
        for row in env.completed_episodes[1:]
    )
    telemetry = env.training_telemetry()
    assert telemetry["authoritative_completed_episode_count"] == 1
    assert telemetry["authoritative_terminal_reason_counts"] == {"SUCCESS": 1}
    assert telemetry["authoritative_success_count"] == 1
    assert telemetry["vector_batch_reset_peer_count"] == NUM_ENVS - 1
    assert VECTOR_BATCH_RESET_BARRIER_REASON not in telemetry[
        "authoritative_terminal_reason_counts"
    ]


def test_vector_adapter_fails_closed_for_phase_snapshots_and_shared_fsm(
    lightweight_row_kernel,
) -> None:
    pytest.importorskip("torch")
    phase_backend = _FakeVectorBackend()
    with pytest.raises(VectorizedResidualEnvError, match="phase-snapshot"):
        VectorizedRslResidualEnv(
            phase_backend,
            seeds=tuple(range(NUM_ENVS)),
            device="cpu",
            training_phase_reset_schedule=("P01", "P02"),
        )
    assert phase_backend.reset_calls == []

    with pytest.raises(VectorizedResidualEnvError, match="controller instances"):
        VectorizedRslResidualEnv(
            _FakeVectorBackend(shared_controller=True),
            seeds=tuple(range(NUM_ENVS)),
            device="cpu",
        )


def test_cli_selects_true_batch_only_for_supported_training_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from wlr50_clean.ppo import rl_library_wrapper
    from wlr50_clean.ppo import vectorized_isaac_backend, vectorized_residual_env

    captures: dict[str, object] = {}
    profile = SimpleNamespace(seed_train=tuple(range(1001, 1033)))

    def make_backend(app, **kwargs):
        captures["backend"] = kwargs
        return SimpleNamespace(num_envs=kwargs["num_envs"])

    def make_env(backend, **kwargs):
        captures["env"] = kwargs
        return SimpleNamespace(num_envs=backend.num_envs)

    monkeypatch.setattr(
        vectorized_isaac_backend, "VectorizedIsaacFSMBackend", make_backend
    )
    monkeypatch.setattr(vectorized_residual_env, "VectorizedRslResidualEnv", make_env)
    monkeypatch.setattr(rl_library_wrapper, "load_training_profile", lambda path: profile)
    monkeypatch.setattr(
        rl_library_wrapper, "build_rsl_runner_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        rl_library_wrapper, "construct_runner", lambda *args, **kwargs: object()
    )
    base = dict(
        command="train",
        num_envs=8,
        training_config=tmp_path / "training.yaml",
        seed=1001,
        run_dir=tmp_path,
        phase_curriculum_max_decisions=None,
    )

    _, env, _, _ = cli._construct_live_runner(
        SimpleNamespace(**base, stage="full-episode"),
        object(),
        max_iterations=1,
    )

    assert env.num_envs == 8
    assert captures["backend"] == {"num_envs": 8, "device": "cuda:0"}
    assert captures["env"]["seeds"] == profile.seed_train
    assert captures["env"]["collect_trace"] is False

    with pytest.raises(cli.CliError, match="cannot independently restore phase snapshots"):
        cli._construct_live_runner(
            SimpleNamespace(**base, stage="phase-curriculum"),
            object(),
            max_iterations=1,
        )

    evaluate = {**base, "command": "evaluate"}
    with pytest.raises(cli.CliError, match="training-only"):
        cli._construct_live_runner(
            SimpleNamespace(**evaluate, stage="full-episode"),
            object(),
            max_iterations=1,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("rsl_rl") is None,
    reason="RSL-RL not installed",
)
def test_official_rsl_runner_constructs_and_learns_with_vectorized_adapter(
    lightweight_row_kernel, tmp_path
) -> None:
    from wlr50_clean.ppo.rl_library_wrapper import (
        build_rsl_runner_config,
        construct_runner,
        initialize_zero_mean_actor,
        load_training_profile,
    )

    backend = _FakeVectorBackend()
    env = VectorizedRslResidualEnv(
        backend,
        seeds=tuple(range(1001, 1017)),
        device="cpu",
    )
    profile = load_training_profile()
    config = build_rsl_runner_config(profile, seed=1001, max_iterations=1)
    config["device"] = "cpu"
    config["num_steps_per_env"] = 2

    runner = construct_runner(env, config, log_dir=tmp_path / "rsl")
    initialize_zero_mean_actor(runner)
    actions = runner.get_inference_policy(device="cpu")(
        env.get_observations(), stochastic_output=False
    )
    observations, rewards, dones, extras = env.step(actions)

    assert tuple(actions.shape) == (NUM_ENVS, 12)
    assert tuple(observations["policy"].shape) == (NUM_ENVS, 125)
    assert tuple(rewards.shape) == (NUM_ENVS,)
    assert tuple(dones.shape) == (NUM_ENVS,)
    assert tuple(extras["time_outs"].shape) == (NUM_ENVS,)

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    assert len(backend.action_batches) == 3 * 8
    telemetry = env.training_telemetry()
    assert telemetry["policy_decision_count"] == 3 * NUM_ENVS
    assert sum(telemetry["phase_decision_counts"].values()) == 3 * NUM_ENVS
    assert telemetry["reward_telemetry_complete"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("rsl_rl") is None,
    reason="RSL-RL not installed",
)
def test_official_rsl_runner_learns_across_vector_full_batch_autoresets(
    lightweight_row_kernel, tmp_path
) -> None:
    from wlr50_clean.ppo.rl_library_wrapper import (
        build_rsl_runner_config,
        construct_runner,
        load_training_profile,
    )

    backend = _FakeVectorBackend(terminal_tick=3)
    env = VectorizedRslResidualEnv(
        backend,
        seeds=tuple(range(1001, 1017)),
        device="cpu",
    )
    profile = load_training_profile()
    config = build_rsl_runner_config(profile, seed=1001, max_iterations=1)
    config["device"] = "cpu"
    config["num_steps_per_env"] = 2

    runner = construct_runner(env, config, log_dir=tmp_path / "rsl-autoreset")
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    # The initial reset plus one synchronous reset after each RSL step proves
    # that the adapter returned reset observations without an extra external
    # reset call or a step-after-done failure.
    assert len(backend.reset_calls) == 3
    assert len(backend.action_batches) == 2 * 3
    assert len(env.completed_episodes) == 2 * NUM_ENVS
    assert env.policy_decision_count == 2 * NUM_ENVS
    assert env.episode_length_buf.tolist() == [0] * NUM_ENVS
    assert sum(
        row["vector_batch_reset_peer"] is False
        for row in env.completed_episodes
    ) == 2
    assert sum(
        row["vector_batch_reset_peer"] is True
        for row in env.completed_episodes
    ) == 2 * (NUM_ENVS - 1)
    telemetry = env.training_telemetry()
    assert sum(telemetry["phase_decision_counts"].values()) == 2 * NUM_ENVS
    assert telemetry["reward_telemetry_incomplete_count"] == 0
    assert telemetry["reward_telemetry_complete"] is True
