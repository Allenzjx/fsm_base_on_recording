"""Backend-agnostic Residual-PPO environment around authoritative FSM ticks.

The backend owns live sensing, the fixed P01--P13 graph, and one authoritative
120 Hz physics/action tick.  This adapter owns only a 15 Hz residual decision,
projection, reward bookkeeping and Gymnasium-shaped return values.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from .action_projection import (
    ActionProjector,
    ProjectionResult,
    SafetyProjection,
)
from .episode_logger import EpisodeLogger, EpisodeTransition
from .observation_schema import ObservationSchema, PPOObservationFrame, load_observation_schema
from .reward_terms import RewardBreakdown, RewardCalculator, RewardSignals
from .termination import (
    TerminationDecision,
    TerminationEvaluator,
    TerminationReason,
    TerminationSignals,
)


DOMAIN_RANDOMIZATION_SCHEMA = "wlr50_clean.ppo_domain_randomization.v1"
DEFAULT_DOMAIN_RANDOMIZATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "ppo_domain_randomization.yaml"
)
RESET_METADATA_FIELDS = (
    "environment_hash",
    "robot_asset_hash",
    "initial_root_state",
    "initial_joint_state",
    "obstacle_pose",
    "controller_hash",
    "motion_contract_hash",
)


class PPOEnvError(ValueError):
    pass


def _vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise PPOEnvError(f"{label} must be numeric") from exc
    if len(result) != 12 or any(not math.isfinite(value) for value in result):
        raise PPOEnvError(f"{label} must contain twelve finite values")
    return result


def _mask(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != 12 or any(value not in (0, 1) for value in result):
        raise PPOEnvError("action mask must contain twelve binary values")
    return result


@dataclass(frozen=True, slots=True)
class RandomizationHook:
    name: str
    enabled: bool
    baseline: float
    minimum: float
    maximum: float
    unit: str


@dataclass(frozen=True, slots=True)
class DomainRandomizationConfig:
    schema: str
    enabled: bool
    training_enabled: bool
    nominal_evaluation_uses_frozen_environment: bool
    hooks: Mapping[str, RandomizationHook]
    path: Path

    def sample(self, seed: int) -> dict[str, float]:
        if int(seed) < 0:
            raise PPOEnvError("reset seed must be a non-negative integer")
        generator = random.Random(int(seed))
        values = {}
        for name, hook in self.hooks.items():
            values[name] = (
                generator.uniform(hook.minimum, hook.maximum)
                if self.enabled and hook.enabled
                else hook.baseline
            )
        return values


def load_domain_randomization_config(
    path: Path | str = DEFAULT_DOMAIN_RANDOMIZATION_PATH,
) -> DomainRandomizationConfig:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != DOMAIN_RANDOMIZATION_SCHEMA:
        raise PPOEnvError("unexpected domain-randomization schema")
    if bool(payload.get("training_enabled")):
        raise PPOEnvError("PPO training must remain disabled")
    if not bool(payload.get("nominal_evaluation_uses_frozen_environment")):
        raise PPOEnvError("nominal evaluation must use the frozen environment")
    hooks: dict[str, RandomizationHook] = {}
    for name, raw in payload.get("hooks", {}).items():
        limits = tuple(float(item) for item in raw["range"])
        baseline = float(raw["baseline"])
        if (
            len(limits) != 2
            or not limits[0] <= baseline <= limits[1]
            or not all(math.isfinite(item) for item in (*limits, baseline))
        ):
            raise PPOEnvError(f"domain-randomization hook {name} is invalid")
        hooks[str(name)] = RandomizationHook(
            name=str(name),
            enabled=bool(raw["enabled"]),
            baseline=baseline,
            minimum=limits[0],
            maximum=limits[1],
            unit=str(raw["unit"]),
        )
    if not hooks:
        raise PPOEnvError("domain-randomization hooks are missing")
    return DomainRandomizationConfig(
        schema=DOMAIN_RANDOMIZATION_SCHEMA,
        enabled=bool(payload["enabled"]),
        training_enabled=False,
        nominal_evaluation_uses_frozen_environment=True,
        hooks=hooks,
        path=selected,
    )


@dataclass(frozen=True, slots=True)
class AuthoritativeFrame:
    """State produced by the nominal controller before one physics tick."""

    physics_tick: int
    sim_time_s: float
    state_id: str
    macro_phase: int
    phase_progress: float
    observation: PPOObservationFrame
    nominal_action_full12: tuple[float, ...]
    reference_action_full12: tuple[float, ...]
    reference_delta_full12: tuple[float, ...]
    action_mask_full12: tuple[int, ...]
    reward_signals: RewardSignals = field(default_factory=RewardSignals)
    termination_signals: TerminationSignals = field(default_factory=TerminationSignals)
    safety_projection: SafetyProjection = field(default_factory=SafetyProjection)
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.physics_tick < 0 or not math.isfinite(float(self.sim_time_s)):
            raise PPOEnvError("authoritative frame clock is invalid")
        if self.state_id != self.observation.state_id:
            raise PPOEnvError("frame and observation state_id disagree")
        if self.macro_phase != self.observation.macro_phase:
            raise PPOEnvError("frame and observation macro_phase disagree")
        if self.macro_phase != int(self.state_id[1:]):
            raise PPOEnvError("state_id and macro_phase disagree")
        if abs(float(self.phase_progress) - self.observation.phase_progress) > 1.0e-12:
            raise PPOEnvError("frame and observation phase_progress disagree")
        for name in (
            "nominal_action_full12",
            "reference_action_full12",
            "reference_delta_full12",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        object.__setattr__(self, "action_mask_full12", _mask(self.action_mask_full12))


class AuthoritativeFSMBackend(Protocol):
    """Minimal live/replay test seam; implementations retain FSM ownership."""

    def reset(
        self, *, seed: int, options: Mapping[str, Any]
    ) -> AuthoritativeFrame: ...

    def step_physics(
        self, applied_action_full12: Sequence[float]
    ) -> AuthoritativeFrame: ...


class PPOEnvAdapter:
    """One residual decision every eight authoritative 120 Hz backend ticks."""

    metadata = {
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "actor_observation_dimension": 85,
        "nominal_action_dimension": 12,
        "residual_action_dimension": 12,
        "training_enabled": False,
    }

    def __init__(
        self,
        backend: AuthoritativeFSMBackend,
        *,
        observation_schema: ObservationSchema | None = None,
        projector: ActionProjector | None = None,
        reward_calculator: RewardCalculator | None = None,
        termination_evaluator: TerminationEvaluator | None = None,
        domain_randomization: DomainRandomizationConfig | None = None,
        episode_logger: EpisodeLogger | None = None,
    ) -> None:
        self.backend = backend
        self.observation_schema = observation_schema or load_observation_schema()
        self.projector = projector or ActionProjector()
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.termination_evaluator = termination_evaluator or TerminationEvaluator()
        self.domain_randomization = (
            domain_randomization or load_domain_randomization_config()
        )
        self.episode_logger = episode_logger
        if self.projector.config.physics_ticks_per_decision != 8:
            raise PPOEnvError("PPO decisions must contain exactly eight physics ticks")
        self._frame: AuthoritativeFrame | None = None
        self._previous_residual = (0.0,) * 12
        self._decision_tick = 0
        self._seed: int | None = None
        self._episode_id: str | None = None
        self._reset_info: dict[str, Any] = {}
        self._done = False

    def reset(
        self, seed: int, options: Mapping[str, Any] | None = None
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        reset_seed = int(seed)
        if reset_seed < 0 or reset_seed != seed:
            raise PPOEnvError("reset seed must be a non-negative integer")
        reset_options = dict(options or {})
        if reset_options.get("enable_randomization") and not self.domain_randomization.enabled:
            raise PPOEnvError("domain randomization is disabled by the versioned config")
        randomization_values = self.domain_randomization.sample(reset_seed)
        backend_options = dict(reset_options)
        backend_options["randomization_enabled"] = self.domain_randomization.enabled
        backend_options["randomization_values"] = dict(randomization_values)
        frame = self.backend.reset(seed=reset_seed, options=backend_options)
        missing = [name for name in RESET_METADATA_FIELDS if name not in frame.info]
        if missing:
            raise PPOEnvError(f"backend reset metadata is incomplete: {missing}")
        self._frame = frame
        self._previous_residual = (0.0,) * 12
        self._decision_tick = 0
        self._seed = reset_seed
        self._episode_id = str(
            reset_options.get("episode_id", f"ppo-baseline-seed-{reset_seed:010d}")
        )
        self._done = False
        self._reset_info = {
            **dict(frame.info),
            "seed": reset_seed,
            "episode_id": self._episode_id,
            "physics_hz": 120.0,
            "decision_hz": 15.0,
            "randomization_enabled": self.domain_randomization.enabled,
            "randomization_values": dict(randomization_values),
            "training_enabled": False,
        }
        observation = self.observation_schema.encode(frame.observation)
        return observation, dict(self._reset_info)

    def step(
        self, residual_action: Sequence[float]
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        if self._frame is None or self._seed is None or self._episode_id is None:
            raise PPOEnvError("reset(seed, options) must be called before step")
        if self._done:
            raise PPOEnvError("step called after episode completion")
        raw_residual = _vector(residual_action, "residual_action")
        start = self._frame
        observation_t = self.observation_schema.encode(start.observation)
        first_nominal: tuple[float, ...] | None = None
        first_applied: tuple[float, ...] | None = None
        first_mask: tuple[int, ...] | None = None
        projection_rows: list[ProjectionResult] = []
        reward_rows: list[RewardBreakdown] = []
        decision = TerminationDecision(False, False, None, (), ())
        held_raw = raw_residual

        for _ in range(self.projector.config.physics_ticks_per_decision):
            frame = self._frame
            projected = self.projector.project(
                held_raw,
                state_id=frame.state_id,
                nominal_action_full12=frame.nominal_action_full12,
                reference_action_full12=frame.reference_action_full12,
                reference_delta_full12=frame.reference_delta_full12,
                previous_projected_residual_full12=self._previous_residual,
                runtime_action_mask_full12=frame.action_mask_full12,
                safety=frame.safety_projection,
                dt_s=1.0 / 120.0,
            )
            if first_nominal is None:
                first_nominal = frame.nominal_action_full12
                first_applied = projected.applied_action_full12
                first_mask = projected.effective_action_mask_full12
            previous_residual = self._previous_residual
            next_frame = self.backend.step_physics(projected.applied_action_full12)
            if next_frame.physics_tick != frame.physics_tick + 1:
                raise PPOEnvError(
                    "backend physics_tick must advance by exactly one per tick"
                )
            expected_time = frame.sim_time_s + 1.0 / 120.0
            if not math.isclose(
                next_frame.sim_time_s,
                expected_time,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise PPOEnvError(
                    "backend sim_time_s must advance by exactly 1/120 s per tick"
                )
            fraction = tuple(
                value / scale if scale > 0.0 else 0.0
                for value, scale in zip(
                    projected.safe_projected_residual_full12,
                    projected.physical_residual_scale_full12,
                    strict=True,
                )
            )
            rate_fraction = tuple(
                (value - prior) / scale if scale > 0.0 else 0.0
                for value, prior, scale in zip(
                    projected.safe_projected_residual_full12,
                    previous_residual,
                    projected.physical_residual_scale_full12,
                    strict=True,
                )
            )
            reward_rows.append(
                self.reward_calculator.evaluate(
                    next_frame.reward_signals,
                    residual_fraction_full12=fraction,
                    residual_rate_fraction_full12=rate_fraction,
                )
            )
            signals = next_frame.termination_signals
            if next_frame.sim_time_s >= self.termination_evaluator.config.timeout_s:
                signals = replace(signals, timeout=True)
            decision = self.termination_evaluator.evaluate(signals)
            projection_rows.append(projected)
            self._frame = next_frame
            self._previous_residual = projected.safe_projected_residual_full12
            if next_frame.state_id != frame.state_id:
                # A policy has not observed the new phase yet.  Do not carry
                # its prior-phase residual across a discrete FSM transition.
                held_raw = (0.0,) * 12
                self._previous_residual = (0.0,) * 12
            if decision.terminated or decision.truncated:
                break

        assert first_nominal is not None and first_applied is not None and first_mask is not None
        next_observation = self.observation_schema.encode(self._frame.observation)
        weighted_names = tuple(self.reward_calculator.config.terms)
        aggregate_components = {
            name: sum(row.weighted_components[name] for row in reward_rows)
            for name in weighted_names
        }
        total_reward = sum(row.total for row in reward_rows)
        terminated = decision.terminated
        truncated = decision.truncated
        reason = None if decision.reason is None else decision.reason.value
        info: dict[str, Any] = {
            **dict(self._frame.info),
            "seed": self._seed,
            "episode_id": self._episode_id,
            "control_tick": self._decision_tick,
            "physics_ticks_executed": len(projection_rows),
            "reward_components": aggregate_components,
            "termination_reason": reason,
            "triggered_termination_reasons": tuple(
                item.value for item in decision.triggered_reasons
            ),
            "termination_diagnostics": tuple(item.value for item in decision.diagnostics),
            "projection_clipping_stages": tuple(
                row.clipping_stages for row in projection_rows
            ),
            "zero_residual_fast_path_all_ticks": all(
                row.zero_residual_fast_path for row in projection_rows
            ),
            "training_enabled": False,
        }
        if self.episode_logger is not None:
            self.episode_logger.append(
                EpisodeTransition(
                    episode_id=self._episode_id,
                    trial_id=self._episode_id,
                    seed=self._seed,
                    control_tick=self._decision_tick,
                    sim_time=start.sim_time_s,
                    state_id=start.state_id,
                    macro_phase=start.macro_phase,
                    phase_progress=start.phase_progress,
                    observation_t=observation_t,
                    nominal_action_t=first_nominal,
                    residual_action_t=raw_residual,
                    applied_action_t=first_applied,
                    action_mask_t=first_mask,
                    task_result=(reason or "ONGOING"),
                    reward_components_t=aggregate_components,
                    terminated=terminated,
                    truncated=truncated,
                    termination_reason=reason,
                    observation_t_plus_1=next_observation,
                    environment_hash=str(self._reset_info["environment_hash"]),
                    controller_hash=str(self._reset_info["controller_hash"]),
                    motion_contract_hash=str(self._reset_info["motion_contract_hash"]),
                    observation_schema_version=(
                        f"{self.observation_schema.schema_name}.v"
                        f"{self.observation_schema.schema_version}"
                    ),
                    action_schema_version=(
                        f"{self.projector.config.action_schema_name}.v"
                        f"{self.projector.config.action_schema_version}"
                    ),
                )
            )
        self._decision_tick += 1
        self._done = terminated or truncated
        return next_observation, total_reward, terminated, truncated, info
