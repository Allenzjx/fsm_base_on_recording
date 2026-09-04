from __future__ import annotations

import importlib.util
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli, residual_direct_env as residual_subject
from wlr50_clean.ppo.action_projection import SafetyProjection
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES, load_phase_objectives
from wlr50_clean.ppo.residual_direct_env import (
    PHASE_CURRICULUM_BOUNDARY_REASON,
    PHASE_CURRICULUM_BASELINE_DECISIONS,
    PHASE_CURRICULUM_HORIZON_REASON,
    PHASE_CURRICULUM_PRIORITY_STATES,
    PHASE_CURRICULUM_RESET_CYCLE,
    PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS,
    STATE_IDS,
    ResidualEpisodeEnv,
    ResidualStep,
    RslResidualVecEnv,
    build_phase_curriculum_reset_cycle,
    build_reward_dominance_telemetry,
)
from wlr50_clean.ppo.termination import (
    TerminationDecision,
    TerminationReason,
    TerminationSignals,
)
from wlr50_clean.ppo.termination_v2 import (
    TerminationEvaluatorV2,
    TerminationSignalsV2,
)


ZERO12 = (0.0,) * 12
OBSERVATION125 = (0.0,) * 125
_MISSING = object()


def test_phase_curriculum_cycle_targets_policy_decisions_not_reset_counts() -> None:
    counts = Counter(PHASE_CURRICULUM_RESET_CYCLE)
    effective_lengths = {
        phase_id: min(PHASE_CURRICULUM_BASELINE_DECISIONS[phase_id], 64)
        for phase_id in STATE_IDS
    }
    predicted_decisions = {
        phase_id: counts[phase_id] * effective_lengths[phase_id]
        for phase_id in STATE_IDS
    }
    predicted_total = sum(predicted_decisions.values())

    assert len(PHASE_CURRICULUM_RESET_CYCLE) == 128
    assert set(counts) == set(STATE_IDS)
    assert all(
        abs(
            predicted_decisions[phase_id] / predicted_total
            - PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS[phase_id]
        )
        <= 0.02
        for phase_id in STATE_IDS
    )
    # Short P08 needs many more reset samples than long P13 to yield the same
    # number of policy decisions.
    assert counts["P08"] > counts["P13"]


@dataclass(frozen=True)
class _TestReward:
    total: float = 0.0


class _BoundaryBackend:
    def __init__(
        self,
        *,
        starting_phase: str,
        next_phase: str,
        transition_tick: int,
        controller_blocked_tick: int | None = None,
    ):
        self.starting_phase = starting_phase
        self.next_phase = next_phase
        self.transition_tick = transition_tick
        self.controller_blocked_tick = controller_blocked_tick
        self.tick = 0

    def _frame(self) -> SimpleNamespace:
        phase_id = self.next_phase if self.tick >= self.transition_tick else self.starting_phase
        return SimpleNamespace(
            physics_tick=self.tick,
            sim_time_s=self.tick / 120.0,
            state_id=phase_id,
            macro_phase=int(phase_id[1:]),
            phase_progress=0.0,
            nominal_action_full12=ZERO12,
            reference_action_full12=ZERO12,
            reference_delta_full12=ZERO12,
            safety_projection=SafetyProjection(),
            termination_signals=TerminationSignals(),
            info={
                "controller_task_result": (
                    "INCOMPLETE_CONTROLLER_BLOCKED"
                    if self.controller_blocked_tick is not None
                    and self.tick >= self.controller_blocked_tick
                    else None
                )
            },
        )

    def reset(self, *, seed: int, options: dict[str, object]) -> SimpleNamespace:
        self.tick = 0
        return self._frame()

    def step_physics(self, applied_action_full12: tuple[float, ...]) -> SimpleNamespace:
        self.tick += 1
        return self._frame()


class _BoundaryEpisode(ResidualEpisodeEnv):
    def _encode(self, frame: SimpleNamespace) -> tuple[float, ...]:
        return OBSERVATION125

    def _reward(
        self,
        start: SimpleNamespace,
        end: SimpleNamespace,
        residual: tuple[float, ...],
        **_: object,
    ) -> _TestReward:
        return _TestReward()


def test_curriculum_phase_boundary_stops_on_exact_transition_tick() -> None:
    backend = _BoundaryBackend(
        starting_phase="P08", next_phase="P09", transition_tick=3
    )
    env = _BoundaryEpisode(backend)
    env.reset(seed=1001)

    step = env.step(ZERO12, stop_after_phase_id="P08")

    assert backend.tick == 3
    assert step.terminated is False
    assert step.truncated is True
    assert step.info["physics_ticks_executed"] == 3
    assert step.info["termination_reason"] == PHASE_CURRICULUM_BOUNDARY_REASON
    assert step.info["phase_curriculum_boundary"] is True


def test_residual_episode_without_curriculum_runs_full_decision() -> None:
    backend = _BoundaryBackend(
        starting_phase="P08", next_phase="P09", transition_tick=3
    )
    env = _BoundaryEpisode(backend)
    env.reset(seed=2001)

    step = env.step(ZERO12)

    assert backend.tick == 8
    assert step.terminated is False
    assert step.truncated is False
    assert step.info["phase_curriculum_boundary"] is False
    assert step.info["termination_reason"] is None


def test_controller_blocked_is_terminal_task_failure_not_timeout() -> None:
    backend = _BoundaryBackend(
        starting_phase="P05",
        next_phase="P05",
        transition_tick=999,
        controller_blocked_tick=3,
    )
    env = _BoundaryEpisode(backend)
    env.reset(seed=1001)

    step = env.step(ZERO12)

    assert backend.tick == 3
    assert step.terminated is True
    assert step.truncated is False
    assert step.info["termination_reason"] == "CONTROLLER_BLOCKED"


def test_controller_blocked_beats_same_tick_timeout_without_bootstrap() -> None:
    class _TimeoutAtBlockedTick:
        def evaluate(self, signals, *, episode_time_s: float) -> TerminationDecision:
            timed_out = episode_time_s >= 3.0 / 120.0
            return TerminationDecision(
                terminated=False,
                truncated=timed_out,
                reason=TerminationReason.TIMEOUT if timed_out else None,
                triggered_reasons=(TerminationReason.TIMEOUT,) if timed_out else (),
                diagnostics=(),
            )

    backend = _BoundaryBackend(
        starting_phase="P05",
        next_phase="P05",
        transition_tick=999,
        controller_blocked_tick=3,
    )
    env = _BoundaryEpisode(backend, termination_evaluator=_TimeoutAtBlockedTick())
    env.reset(seed=1001)

    step = env.step(ZERO12)

    assert backend.tick == 3
    assert step.terminated is True
    assert step.truncated is False
    assert step.info["termination_reason"] == "CONTROLLER_BLOCKED"


def test_controller_blocked_does_not_hide_primary_physical_failure_reason() -> None:
    class _ExplosionAtBlockedTick:
        def evaluate(self, signals, *, episode_time_s: float) -> TerminationDecision:
            exploded = episode_time_s >= 3.0 / 120.0
            return TerminationDecision(
                terminated=exploded,
                truncated=False,
                reason=TerminationReason.PHYSICS_EXPLOSION if exploded else None,
                triggered_reasons=(
                    (TerminationReason.PHYSICS_EXPLOSION,) if exploded else ()
                ),
                diagnostics=(),
            )

    backend = _BoundaryBackend(
        starting_phase="P05",
        next_phase="P05",
        transition_tick=999,
        controller_blocked_tick=3,
    )
    env = _BoundaryEpisode(
        backend,
        termination_evaluator=_ExplosionAtBlockedTick(),
    )
    env.reset(seed=1001)

    step = env.step(ZERO12)

    assert step.terminated is True
    assert step.truncated is False
    assert step.info["termination_reason"] == "PHYSICS_EXPLOSION"


def test_controller_blocked_activates_task_failure_reward_event() -> None:
    captured: dict[str, object] = {}

    class _Calculator:
        objectives = SimpleNamespace(
            phase=lambda phase_id: SimpleNamespace(
                successful_fsm_attitude_envelope_rad=None,
                attitude_envelope_excess_normalization_rad=None,
            )
        )

        def evaluate(self, signals):
            captured["signals"] = signals
            return _TestReward()

    env = ResidualEpisodeEnv.__new__(ResidualEpisodeEnv)
    env.signals = SimpleNamespace(
        progress=lambda frame: 0.0,
        contact_costs=lambda start, end: {},
    )
    env.reward_calculator = _Calculator()
    env.phase_actions = SimpleNamespace(
        physical_scale_for=lambda phase_id: (1.0,) * 12
    )
    env.previous_residual = ZERO12
    env.previous_previous_residual = ZERO12
    base = SimpleNamespace(angular_velocity_w_rad_s=(0.0, 0.0, 0.0))
    start = SimpleNamespace(
        state_id="P05",
        info={"raw_observation": SimpleNamespace(base=base)},
    )
    end = SimpleNamespace(
        state_id="P05",
        info={
            "raw_observation": SimpleNamespace(base=base),
            "level_calibration": {},
            "controller_task_result": "INCOMPLETE_CONTROLLER_BLOCKED",
        },
        termination_signals=TerminationSignals(),
    )

    env._reward(
        start,
        end,
        ZERO12,
        termination_reason=None,
        controller_blocked=True,
    )

    assert captured["signals"].task_failure is True


@pytest.mark.parametrize(
    ("signals", "expected_reason", "task_failure", "safety_abort"),
    (
        (
            TerminationSignals(body_collision=True, fall=True),
            TerminationReason.BODY_COLLISION,
            True,
            False,
        ),
        (
            TerminationSignals(body_collision=True, physics_explosion=True),
            TerminationReason.PHYSICS_EXPLOSION,
            False,
            True,
        ),
    ),
)
def test_reward_terminal_event_reuses_primary_termination_reason_when_faults_cooccur(
    signals: TerminationSignals,
    expected_reason: TerminationReason,
    task_failure: bool,
    safety_abort: bool,
) -> None:
    captured: dict[str, object] = {}
    decision = TerminationEvaluatorV2().evaluate(
        TerminationSignalsV2(
            body_collision=signals.body_collision,
            wheel_only_climb=signals.wheel_only_climb,
            fall=signals.fall,
            nan_inf=signals.nan_inf,
            hard_joint_limit=signals.hard_joint_limit,
            physics_explosion=signals.physics_explosion,
        ),
        episode_time_s=1.0,
    )
    assert decision.reason is expected_reason

    class _Calculator:
        objectives = SimpleNamespace(
            phase=lambda phase_id: SimpleNamespace(
                successful_fsm_attitude_envelope_rad=None,
                attitude_envelope_excess_normalization_rad=None,
            )
        )

        def evaluate(self, reward_signals):
            captured["signals"] = reward_signals
            return _TestReward()

    env = ResidualEpisodeEnv.__new__(ResidualEpisodeEnv)
    env.signals = SimpleNamespace(
        progress=lambda frame: 0.0,
        contact_costs=lambda start, end: {},
    )
    env.reward_calculator = _Calculator()
    env.phase_actions = SimpleNamespace(
        physical_scale_for=lambda phase_id: (1.0,) * 12
    )
    env.previous_residual = ZERO12
    env.previous_previous_residual = ZERO12
    base = SimpleNamespace(angular_velocity_w_rad_s=(0.0, 0.0, 0.0))
    start = SimpleNamespace(
        state_id="P05",
        info={"raw_observation": SimpleNamespace(base=base)},
    )
    end = SimpleNamespace(
        state_id="P05",
        info={"raw_observation": SimpleNamespace(base=base), "level_calibration": {}},
        termination_signals=signals,
    )

    env._reward(
        start,
        end,
        ZERO12,
        termination_reason=decision.reason,
        controller_blocked=False,
    )

    reward_signals = captured["signals"]
    assert reward_signals.task_failure is task_failure
    assert reward_signals.safety_abort is safety_abort
    assert sum(
        (reward_signals.final_success, reward_signals.task_failure, reward_signals.safety_abort)
    ) == 1


@pytest.mark.parametrize("phase_id", ("P08", "P11"))
def test_reward_uses_frozen_per_phase_attitude_envelope(phase_id: str) -> None:
    captured: list[object] = []

    class _Calculator:
        objectives = load_phase_objectives()

        def evaluate(self, reward_signals):
            captured.append(reward_signals)
            return _TestReward()

    env = ResidualEpisodeEnv.__new__(ResidualEpisodeEnv)
    env.signals = SimpleNamespace(
        progress=lambda frame: 0.0,
        contact_costs=lambda start, end: {},
    )
    env.reward_calculator = _Calculator()
    env.phase_actions = SimpleNamespace(
        physical_scale_for=lambda state_id: (1.0,) * 12
    )
    env.previous_residual = ZERO12
    env.previous_previous_residual = ZERO12
    base = SimpleNamespace(angular_velocity_w_rad_s=(0.0, 0.0, 0.0))
    start = SimpleNamespace(
        state_id=phase_id,
        info={"raw_observation": SimpleNamespace(base=base)},
    )
    objective = env.reward_calculator.objectives.phase(phase_id)
    envelope = objective.successful_fsm_attitude_envelope_rad
    normalization = objective.attitude_envelope_excess_normalization_rad
    assert envelope is not None
    assert normalization is not None

    for attitude, expected in (
        (envelope, 0.0),
        (envelope + normalization / 2.0, 0.5),
        (envelope + normalization * 2.0, 1.0),
    ):
        end = SimpleNamespace(
            state_id=phase_id,
            info={
                "raw_observation": SimpleNamespace(base=base),
                "level_calibration": {
                    "roll_error_to_level_rad": attitude,
                    "pitch_error_to_level_rad": 0.0,
                },
            },
        )
        env._reward(
            start,
            end,
            ZERO12,
            termination_reason=None,
            controller_blocked=False,
        )
        assert captured[-1].successful_fsm_attitude_envelope_excess == pytest.approx(
            expected
        )


def test_v2_snapshot_observation_uses_authoritative_previous_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored_action = tuple(float(index + 1) for index in range(12))
    captured: dict[str, object] = {}

    def from_live(observation, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        residual_subject.PPOObservationFrameV2,
        "from_live_observation",
        staticmethod(from_live),
    )
    env = ResidualEpisodeEnv.__new__(ResidualEpisodeEnv)
    env.observation_schema = SimpleNamespace(encode=lambda frame: OBSERVATION125)
    env.bridge = SimpleNamespace(previous_projected_residual_full12=ZERO12)
    frame = SimpleNamespace(
        state_id="P09",
        macro_phase=9,
        phase_progress=0.0,
        nominal_action_full12=ZERO12,
        observation=SimpleNamespace(previous_action_full12=restored_action),
        info={
            "raw_observation": object(),
            "controller_lifecycle": "EXECUTE_MOTION",
            # A phase snapshot has no episode atomic write yet, so this ack is
            # the deliberately stale zero-command settle evidence.
            "atomic_ack": {"applied_full12": ZERO12},
        },
    )

    encoded = env._encode(frame)

    assert encoded == OBSERVATION125
    assert captured["previous_action_full12"] == restored_action


class _VectorEpisode:
    def __init__(self, *, terminate_after: int | None = None):
        self.terminate_after = terminate_after
        self.reset_calls: list[tuple[int, object]] = []
        self.seed = 0
        self.trace: list[dict[str, object]] = []
        self.frame = SimpleNamespace(state_id="P01", sim_time_s=0.0)
        self.steps = 0

    def reset(self, *, seed: int, options: object = _MISSING):
        self.seed = seed
        self.steps = 0
        self.trace = []
        self.reset_calls.append((seed, options))
        phase_id = (
            "P01"
            if options is _MISSING
            else str(options["training_phase_snapshot"])
        )
        self.frame = SimpleNamespace(state_id=phase_id, sim_time_s=0.0)
        return OBSERVATION125, {}

    def step(self, action: list[float], *, stop_after_phase_id: str | None = None):
        self.steps += 1
        terminated = self.terminate_after is not None and self.steps >= self.terminate_after
        self.frame.sim_time_s = self.steps / 15.0
        return ResidualStep(
            OBSERVATION125,
            0.0,
            terminated,
            False,
            {
                "sim_time_s": self.frame.sim_time_s,
                "termination_reason": "SUCCESS" if terminated else None,
                "phase_curriculum_boundary": False,
                "reward": {
                    "weighted_dense": {
                        family: 0.0 for family in DENSE_FAMILIES
                    }
                },
            },
        )


def test_vec_curriculum_cycles_snapshot_resets_and_marks_horizon_timeout() -> None:
    torch = pytest.importorskip("torch")
    episode = _VectorEpisode()
    env = RslResidualVecEnv(
        [episode],
        seeds=[1001, 1002],
        device="cpu",
        training_phase_reset_schedule=("P02", "P03"),
        end_curriculum_sample_at_phase_boundary=True,
        phase_curriculum_max_decisions=1,
    )

    _, _, dones, extras = env.step(torch.zeros((1, 12)))

    assert dones.tolist() == [True]
    assert extras["time_outs"].tolist() == [True]
    assert episode.reset_calls == [
        (1001, {"training_phase_snapshot": "P02"}),
        (1002, {"training_phase_snapshot": "P03"}),
    ]
    assert env.completed_episodes[0]["termination_reason"] == PHASE_CURRICULUM_HORIZON_REASON
    assert env.completed_episodes[0]["phase_curriculum_start_state_id"] == "P02"
    assert env.completed_episodes[0]["phase_curriculum_horizon"] is True


def test_curriculum_telemetry_gates_measured_policy_decision_occupancy() -> None:
    torch = pytest.importorskip("torch")
    schedule = build_phase_curriculum_reset_cycle(
        max_decisions=1, cycle_samples=80
    )
    episode = _VectorEpisode()
    env = RslResidualVecEnv(
        [episode],
        seeds=[1001, 1002],
        device="cpu",
        training_phase_reset_schedule=schedule,
        end_curriculum_sample_at_phase_boundary=True,
        phase_curriculum_max_decisions=1,
        phase_curriculum_target_decision_fractions=(
            PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS
        ),
        phase_curriculum_occupancy_tolerance_fraction=0.0,
    )

    for _ in range(len(schedule)):
        env.step(torch.zeros((1, 12)))

    telemetry = env.training_telemetry()
    assert telemetry["phase_occupancy_fraction"] == pytest.approx(
        PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS
    )
    assert telemetry["phase_curriculum_occupancy_violations"] == []
    assert telemetry["phase_curriculum_occupancy_within_tolerance"] is True


def test_vec_non_curriculum_ignores_curriculum_horizon_and_reset_options() -> None:
    torch = pytest.importorskip("torch")
    episode = _VectorEpisode(terminate_after=2)
    env = RslResidualVecEnv(
        [episode], seeds=[2001, 2002], device="cpu", phase_curriculum_max_decisions=1
    )

    _, _, first_dones, first_extras = env.step(torch.zeros((1, 12)))
    assert first_dones.tolist() == [False]
    assert first_extras["time_outs"].tolist() == [False]
    assert episode.reset_calls == [(2001, _MISSING)]

    _, _, second_dones, second_extras = env.step(torch.zeros((1, 12)))
    assert second_dones.tolist() == [True]
    assert second_extras["time_outs"].tolist() == [False]
    assert episode.reset_calls == [(2001, _MISSING), (2002, _MISSING)]
    assert env.completed_episodes[0]["termination_reason"] == "SUCCESS"
    assert env.completed_episodes[0]["phase_curriculum_start_state_id"] is None
    telemetry = env.training_telemetry()
    assert telemetry[
        "phase_curriculum_occupancy_within_tolerance"
    ] is None
    assert telemetry["authoritative_completed_episode_count"] == 1
    assert telemetry["authoritative_terminal_reason_counts"] == {"SUCCESS": 1}
    assert telemetry["authoritative_success_count"] == 1
    assert telemetry["vector_batch_reset_peer_count"] == 0


def test_cli_wires_curriculum_only_for_phase_curriculum_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, residual_direct_env, rl_library_wrapper

    captures: list[dict[str, object]] = []
    profile = SimpleNamespace(
        seed_train=(1001, 1002),
        phase_sampling=PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS,
        phase_curriculum_baseline_decisions=PHASE_CURRICULUM_BASELINE_DECISIONS,
        phase_curriculum_reset_cycle_samples=128,
        phase_curriculum_occupancy_tolerance=0.02,
    )
    backend_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        isaac_fsm_backend,
        "IsaacFSMBackend",
        lambda app, **kwargs: backend_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        residual_direct_env,
        "ResidualEpisodeEnv",
        lambda backend, collect_trace: SimpleNamespace(backend=backend),
    )

    def make_vec(environments, **kwargs):
        captures.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(residual_direct_env, "RslResidualVecEnv", make_vec)
    monkeypatch.setattr(rl_library_wrapper, "load_training_profile", lambda path: profile)
    monkeypatch.setattr(rl_library_wrapper, "build_rsl_runner_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(rl_library_wrapper, "construct_runner", lambda *args, **kwargs: object())
    base_args = dict(
        num_envs=1,
        training_config=tmp_path / "training.yaml",
        seed=1001,
        run_dir=tmp_path,
        phase_curriculum_max_decisions=7,
    )

    cli._construct_live_runner(
        SimpleNamespace(**base_args, stage="phase-curriculum"),
        object(),
        max_iterations=1,
    )
    cli._construct_live_runner(
        SimpleNamespace(**base_args, stage="full-episode"),
        object(),
        max_iterations=1,
    )

    assert captures[0]["training_phase_reset_schedule"] == (
        build_phase_curriculum_reset_cycle(max_decisions=7)
    )
    assert captures[0]["end_curriculum_sample_at_phase_boundary"] is True
    assert captures[0]["phase_curriculum_max_decisions"] == 7
    assert captures[0]["phase_curriculum_target_decision_fractions"] == (
        PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS
    )
    assert captures[0]["phase_curriculum_occupancy_tolerance_fraction"] == 0.02
    assert backend_calls[0]["expected_phase_snapshot_bundle"] is not None
    assert backend_calls[0]["expected_effective_entry_contract"] is not None
    assert captures[1]["training_phase_reset_schedule"] is None
    assert backend_calls[1]["expected_phase_snapshot_bundle"] is not None
    assert backend_calls[1]["expected_effective_entry_contract"] is not None
    assert captures[1]["end_curriculum_sample_at_phase_boundary"] is False
    assert captures[1]["phase_curriculum_max_decisions"] == 64
    assert captures[1]["phase_curriculum_target_decision_fractions"] is None


def test_cli_reads_curriculum_horizon_from_training_environment(tmp_path) -> None:
    config = tmp_path / "training.yaml"
    config.write_text(
        "environment:\n  phase_curriculum_max_decisions: 23\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        training_config=config,
        phase_curriculum_max_decisions=None,
    )

    assert cli._phase_curriculum_horizon(args) == 23


def _phase_family_rows() -> dict[str, dict[str, float]]:
    return {
        phase_id: {family: 0.0 for family in DENSE_FAMILIES}
        for phase_id in STATE_IDS
    }


def test_reward_telemetry_reports_phase_family_matrix_and_dominance_gates() -> None:
    phase_rows = _phase_family_rows()
    phase_rows["P08"] = {family: 0.2 for family in DENSE_FAMILIES}
    passing = build_reward_dominance_telemetry(
        signed_sums={family: -0.2 for family in DENSE_FAMILIES},
        absolute_sums={family: 0.2 for family in DENSE_FAMILIES},
        absolute_sums_by_phase=phase_rows,
        incomplete_count=0,
        maximum_single_family_fraction=0.70,
        maximum_residual_regularization_fraction=0.20,
        minimum_absolute_dense_return=1.0e-12,
    )

    assert passing["reward_family_absolute_sums_by_phase"]["P08"] == {
        family: 0.2 for family in DENSE_FAMILIES
    }
    assert passing["reward_dominant_family_fraction"] == pytest.approx(0.2)
    assert passing["reward_residual_regularization_fraction"] == pytest.approx(0.2)
    assert passing["reward_dominance_within_limits"] is True

    dominated_absolute = {
        "phase_task_progress": 0.71,
        "body_stability": 0.08,
        "contact_motion_quality": 0.0,
        "control_smoothness": 0.0,
        "residual_regularization": 0.21,
    }
    dominated_phase_rows = _phase_family_rows()
    dominated_phase_rows["P08"] = dict(dominated_absolute)
    dominated = build_reward_dominance_telemetry(
        signed_sums={family: 0.0 for family in DENSE_FAMILIES},
        absolute_sums=dominated_absolute,
        absolute_sums_by_phase=dominated_phase_rows,
        incomplete_count=0,
        maximum_single_family_fraction=0.70,
        maximum_residual_regularization_fraction=0.20,
        minimum_absolute_dense_return=1.0e-12,
    )
    assert dominated["reward_single_family_within_limit"] is False
    assert dominated["reward_residual_regularization_within_limit"] is False
    assert dominated["reward_dominance_within_limits"] is False


def _passing_gate_telemetry(*, occupancy_ok: bool | None) -> dict[str, object]:
    return {
        "policy_decision_count": 128,
        "phase_decision_counts": {
            f"P{index:02d}": (116 if index == 1 else 1)
            for index in range(1, 14)
        },
        "reward_telemetry_complete": True,
        "reward_dominance_within_limits": True,
        "reward_family_absolute_sums_by_phase": _phase_family_rows(),
        "phase_curriculum_occupancy_within_tolerance": occupancy_ok,
        "phase_curriculum_occupancy_violations": [] if occupancy_ok else ["P08"],
        "authoritative_completed_episode_count": 1,
        "authoritative_terminal_reason_counts": {"SUCCESS": 1},
        "authoritative_success_count": 1,
        "vector_batch_reset_peer_count": 0,
        "completed_sample_count": 1,
    }


def test_training_gate_requires_curriculum_balance_only_for_phase_curriculum() -> None:
    telemetry = _passing_gate_telemetry(occupancy_ok=None)
    cli._validate_training_telemetry(
        telemetry, stage="full-episode", expected_policy_decisions=128
    )

    with pytest.raises(cli.CliError, match="occupancy gate failed.*P08"):
        cli._validate_training_telemetry(
            _passing_gate_telemetry(occupancy_ok=False),
            stage="phase-curriculum",
            expected_policy_decisions=128,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda telemetry: telemetry["phase_decision_counts"].__setitem__("P13", 0), "P13"),
        (lambda telemetry: telemetry.__setitem__("authoritative_success_count", 0), "SUCCESS"),
    ],
)
def test_full_episode_training_gate_requires_every_phase_and_authoritative_success(
    mutation,
    message: str,
) -> None:
    telemetry = _passing_gate_telemetry(occupancy_ok=None)
    mutation(telemetry)

    with pytest.raises(cli.CliError, match=message):
        cli._validate_training_telemetry(
            telemetry,
            stage="full-episode",
            expected_policy_decisions=128,
        )


def test_training_gate_rejects_reward_family_dominance() -> None:
    telemetry = _passing_gate_telemetry(occupancy_ok=True)
    telemetry["reward_dominance_within_limits"] = False

    with pytest.raises(cli.CliError, match="dominance gate failed"):
        cli._validate_training_telemetry(
            telemetry,
            stage="phase-curriculum",
            expected_policy_decisions=128,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("rsl_rl") is None,
    reason="RSL-RL not installed",
)
def test_official_rsl_runner_learns_across_single_env_snapshot_autoresets(
    tmp_path,
) -> None:
    from wlr50_clean.ppo.rl_library_wrapper import (
        build_rsl_runner_config,
        construct_runner,
        load_training_profile,
    )

    episode = _VectorEpisode()
    env = RslResidualVecEnv(
        [episode],
        seeds=[1001, 1002, 1003],
        device="cpu",
        training_phase_reset_schedule=("P02", "P03"),
        end_curriculum_sample_at_phase_boundary=True,
        phase_curriculum_max_decisions=1,
    )
    profile = load_training_profile()
    config = build_rsl_runner_config(profile, seed=1001, max_iterations=1)
    config["device"] = "cpu"
    config["num_steps_per_env"] = 8

    runner = construct_runner(env, config, log_dir=tmp_path / "rsl-autoreset")
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert env.policy_decision_count == 8
    assert len(env.completed_episodes) == 8
    assert len(episode.reset_calls) == 9
    assert [row[1]["training_phase_snapshot"] for row in episode.reset_calls] == [
        "P02",
        "P03",
        "P02",
        "P03",
        "P02",
        "P03",
        "P02",
        "P03",
        "P02",
    ]
    telemetry = env.training_telemetry()
    assert telemetry["policy_decision_count"] == 8
    assert telemetry["phase_decision_counts"]["P02"] == 4
    assert telemetry["phase_decision_counts"]["P03"] == 4
    assert sum(telemetry["phase_decision_counts"].values()) == 8
    assert telemetry["reward_telemetry_complete"] is True
    assert env.episode_length_buf.tolist() == [0]
    assert all(row["length"] == 1 for row in env.completed_episodes)
    assert all(
        row["termination_reason"] == PHASE_CURRICULUM_HORIZON_REASON
        for row in env.completed_episodes
    )
