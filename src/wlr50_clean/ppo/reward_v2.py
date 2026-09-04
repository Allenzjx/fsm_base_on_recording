"""Five-family, phase-specific residual PPO reward.

The reward is intentionally expressed only in normalized, Isaac-independent
signals.  Callers own sensor calibration; this layer owns phase selection,
potential shaping, dense-family aggregation, event terms, and reward audits.
There is no survival/standing-still term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import yaml

from .observation_schema_v2 import LIFECYCLE_IDS
from .phase_objectives import (
    DEFAULT_PHASE_OBJECTIVES_PATH,
    DENSE_FAMILIES,
    STATE_IDS,
    PhaseObjectiveError,
    PhaseObjectivesConfig,
    PhysicalProgressState,
    load_phase_objectives,
    phase_local_potential,
    potential_based_progress,
)


REWARD_SCHEMA_V2 = "wlr50_clean.ppo_reward.v2"
EVENT_FAMILIES = (
    "phase_completion",
    "final_success",
    "task_failure",
    "safety_abort",
)
DEFAULT_REWARD_PATH_V2 = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_reward_v2.yaml"
)


class RewardV2Error(ValueError):
    """Reward v2 configuration or input signals violate the contract."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RewardV2Error(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise RewardV2Error(f"{label} must be finite")
    return result


def _non_negative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise RewardV2Error(f"{label} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class DenseFamilyConfig:
    name: str
    sign: str
    scale: float
    clip_min: float
    clip_max: float

    def __post_init__(self) -> None:
        if self.name not in DENSE_FAMILIES:
            raise RewardV2Error(f"unknown dense family {self.name!r}")
        expected_sign = "positive" if self.name == "phase_task_progress" else "cost"
        if self.sign != expected_sign:
            raise RewardV2Error(
                f"{self.name} must use sign={expected_sign!r}, got {self.sign!r}"
            )
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise RewardV2Error(f"{self.name}.scale must be finite and positive")
        if not (
            math.isfinite(self.clip_min)
            and math.isfinite(self.clip_max)
            and self.clip_min < self.clip_max
        ):
            raise RewardV2Error(f"{self.name}.clip must be a finite increasing pair")
        if self.sign == "cost" and self.clip_min < 0.0:
            raise RewardV2Error(f"{self.name} cost clip may not admit negative values")
        if self.sign == "positive" and not (
            self.clip_min <= 0.0 <= self.clip_max
        ):
            raise RewardV2Error("progress clip must retain zero and signed differences")

    def normalize(self, value: float) -> float:
        scaled = _finite(value, self.name) / self.scale
        return max(self.clip_min, min(self.clip_max, scaled))


@dataclass(frozen=True, slots=True)
class RewardV2Config:
    progress_gamma: float
    physics_ticks_per_decision: int
    dense_families: Mapping[str, DenseFamilyConfig]
    body_stability_coefficients: Mapping[str, float]
    control_smoothness_coefficients: Mapping[str, float]
    events: Mapping[str, float]
    signal_ownership: Mapping[str, tuple[str, ...]]
    maximum_single_dense_family_fraction: float
    maximum_residual_regularization_fraction: float
    minimum_absolute_dense_return: float
    standing_still_max_reward: float

    def __post_init__(self) -> None:
        if tuple(self.dense_families) != DENSE_FAMILIES:
            raise RewardV2Error("reward must contain exactly the five dense families")
        if not math.isfinite(self.progress_gamma) or not 0.0 < self.progress_gamma <= 1.0:
            raise RewardV2Error("progress_gamma must be within (0, 1]")
        if self.physics_ticks_per_decision <= 0:
            raise RewardV2Error("physics_ticks_per_decision must be positive")
        _validate_coefficients(
            self.body_stability_coefficients,
            ("attitude", "angular_rate", "angular_acceleration"),
            "body_stability_coefficients",
        )
        _validate_coefficients(
            self.control_smoothness_coefficients,
            ("residual_first_difference", "residual_second_difference"),
            "control_smoothness_coefficients",
        )
        if tuple(self.events) != EVENT_FAMILIES:
            raise RewardV2Error("events must contain exactly the four v2 event terms")
        if self.events["phase_completion"] <= 0.0 or self.events["final_success"] <= 0.0:
            raise RewardV2Error("completion/success events must be positive")
        if self.events["task_failure"] >= 0.0 or self.events["safety_abort"] >= 0.0:
            raise RewardV2Error("failure/abort events must be negative")
        if tuple(self.signal_ownership) != DENSE_FAMILIES:
            raise RewardV2Error("signal_ownership must cover exactly the five dense families")
        if any(not signals for signals in self.signal_ownership.values()):
            raise RewardV2Error("each dense family must own at least one signal")
        for label, value in (
            (
                "maximum_single_dense_family_fraction",
                self.maximum_single_dense_family_fraction,
            ),
            (
                "maximum_residual_regularization_fraction",
                self.maximum_residual_regularization_fraction,
            ),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise RewardV2Error(f"{label} must be within (0, 1]")
        if (
            not math.isfinite(self.minimum_absolute_dense_return)
            or self.minimum_absolute_dense_return < 0.0
        ):
            raise RewardV2Error("minimum_absolute_dense_return must be non-negative")
        if not math.isfinite(self.standing_still_max_reward):
            raise RewardV2Error("standing_still_max_reward must be finite")


def _validate_coefficients(
    coefficients: Mapping[str, float], expected: tuple[str, ...], label: str
) -> None:
    if tuple(coefficients) != expected:
        raise RewardV2Error(f"{label} must contain exactly {expected}")
    values = tuple(_non_negative(coefficients[name], f"{label}.{name}") for name in expected)
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise RewardV2Error(f"{label} must sum to 1; got {sum(values)}")


@dataclass(frozen=True, slots=True)
class RewardSignalsV2:
    """Normalized live signals for one 15 Hz policy transition.

    ``previous_progress.state_id`` is the phase that owned the action.  The
    current progress state may be that same phase or its immediate successor,
    preserving FSM authority while supporting a phase-completion transition.
    All cost inputs are magnitudes: zero is best and values above the configured
    family scale are clipped.
    """

    previous_progress: PhysicalProgressState
    current_progress: PhysicalProgressState
    lifecycle: str = "EXECUTE_MOTION"
    calibrated_attitude_error: float = 0.0
    successful_fsm_attitude_envelope_excess: float = 0.0
    body_angular_rate: float = 0.0
    body_angular_acceleration: float = 0.0
    contact_motion_costs: Mapping[str, float] | None = None
    residual_first_difference: float = 0.0
    residual_second_difference: float = 0.0
    residual_magnitude: float = 0.0
    phase_completion: bool = False
    final_success: bool = False
    task_failure: bool = False
    safety_abort: bool = False

    def __post_init__(self) -> None:
        if self.lifecycle not in LIFECYCLE_IDS:
            raise RewardV2Error(f"unknown FSM lifecycle {self.lifecycle!r}")
        for name in (
            "calibrated_attitude_error",
            "successful_fsm_attitude_envelope_excess",
            "body_angular_rate",
            "body_angular_acceleration",
            "residual_first_difference",
            "residual_second_difference",
            "residual_magnitude",
        ):
            _non_negative(getattr(self, name), name)
        costs = self.contact_motion_costs or {}
        for name, value in costs.items():
            if not str(name).strip():
                raise RewardV2Error("contact/motion cost names may not be empty")
            _non_negative(value, f"contact_motion_costs.{name}")
        terminal_count = sum(
            bool(value)
            for value in (self.final_success, self.task_failure, self.safety_abort)
        )
        if terminal_count > 1:
            raise RewardV2Error(
                "final_success, task_failure, and safety_abort are mutually exclusive"
            )
        if self.final_success and self.previous_progress.state_id != "P13":
            raise RewardV2Error("final_success may only be emitted for a P13 action")


@dataclass(frozen=True, slots=True)
class RewardBreakdownV2:
    total: float
    phase_id: str
    lifecycle: str
    reward_substage: str
    physical_phase_progress: float
    phase_weights: Mapping[str, float]
    raw_dense: Mapping[str, float]
    normalized_dense: Mapping[str, float]
    weighted_dense: Mapping[str, float]
    event_components: Mapping[str, float]
    dense_total: float
    event_total: float

    def __post_init__(self) -> None:
        for label, mapping in (
            ("phase_weights", self.phase_weights),
            ("raw_dense", self.raw_dense),
            ("normalized_dense", self.normalized_dense),
            ("weighted_dense", self.weighted_dense),
        ):
            if tuple(mapping) != DENSE_FAMILIES:
                raise RewardV2Error(f"{label} must contain exactly five dense families")
        if tuple(self.event_components) != EVENT_FAMILIES:
            raise RewardV2Error("event_components must contain exactly four events")


class RewardCalculatorV2:
    """Compute one phase-specific reward per 15 Hz policy decision."""

    def __init__(
        self,
        config: RewardV2Config,
        objectives: PhaseObjectivesConfig,
    ) -> None:
        self.config = config
        self.objectives = objectives

    @classmethod
    def from_files(
        cls,
        reward_path: str | Path = DEFAULT_REWARD_PATH_V2,
        objectives_path: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
    ) -> "RewardCalculatorV2":
        return cls(load_reward_v2_config(reward_path), load_phase_objectives(objectives_path))

    def evaluate(self, signals: RewardSignalsV2) -> RewardBreakdownV2:
        action_phase = signals.previous_progress.state_id
        objective = self.objectives.phase(action_phase)
        try:
            physical_progress = phase_local_potential(
                signals.previous_progress, self.objectives
            )
            raw_progress = potential_based_progress(
                signals.previous_progress,
                signals.current_progress,
                self.objectives,
                gamma=self.config.progress_gamma,
            )
        except PhaseObjectiveError as exc:
            raise RewardV2Error(str(exc)) from exc

        weights = objective.weights_at(physical_progress)
        level_fraction = objective.level_penalty_fraction_at(physical_progress)
        attitude_cost = (
            (1.0 - level_fraction)
            * _non_negative(
                signals.successful_fsm_attitude_envelope_excess,
                "successful_fsm_attitude_envelope_excess",
            )
            + level_fraction
            * _non_negative(
                signals.calibrated_attitude_error, "calibrated_attitude_error"
            )
        )
        stability_coefficients = self.config.body_stability_coefficients
        stability_cost = (
            stability_coefficients["attitude"] * attitude_cost
            + stability_coefficients["angular_rate"]
            * _non_negative(signals.body_angular_rate, "body_angular_rate")
            + stability_coefficients["angular_acceleration"]
            * _non_negative(
                signals.body_angular_acceleration, "body_angular_acceleration"
            )
        )

        supplied_contact = signals.contact_motion_costs or {}
        missing_contact = [
            name for name in objective.contact_cost_terms if name not in supplied_contact
        ]
        if missing_contact:
            raise RewardV2Error(
                f"{action_phase} missing contact/motion costs {missing_contact}"
            )
        contact_cost = sum(
            _non_negative(supplied_contact[name], f"contact_motion_costs.{name}")
            for name in objective.contact_cost_terms
        ) / len(objective.contact_cost_terms)

        smoothness_coefficients = self.config.control_smoothness_coefficients
        smoothness_cost = (
            smoothness_coefficients["residual_first_difference"]
            * _non_negative(
                signals.residual_first_difference, "residual_first_difference"
            )
            + smoothness_coefficients["residual_second_difference"]
            * _non_negative(
                signals.residual_second_difference, "residual_second_difference"
            )
        )
        residual_cost = _non_negative(signals.residual_magnitude, "residual_magnitude")

        raw_dense = {
            "phase_task_progress": raw_progress,
            "body_stability": stability_cost,
            "contact_motion_quality": contact_cost,
            "control_smoothness": smoothness_cost,
            "residual_regularization": residual_cost,
        }
        normalized_dense = {
            name: self.config.dense_families[name].normalize(raw_dense[name])
            for name in DENSE_FAMILIES
        }
        weighted_dense = {
            name: weights[name]
            * normalized_dense[name]
            * (1.0 if self.config.dense_families[name].sign == "positive" else -1.0)
            for name in DENSE_FAMILIES
        }

        event_flags = {
            "phase_completion": signals.phase_completion,
            "final_success": signals.final_success,
            "task_failure": signals.task_failure,
            "safety_abort": signals.safety_abort,
        }
        event_components = {
            name: self.config.events[name] if event_flags[name] else 0.0
            for name in EVENT_FAMILIES
        }
        dense_total = sum(weighted_dense.values())
        event_total = sum(event_components.values())
        return RewardBreakdownV2(
            total=dense_total + event_total,
            phase_id=action_phase,
            lifecycle=signals.lifecycle,
            reward_substage=objective.substage_at(physical_progress),
            physical_phase_progress=physical_progress,
            phase_weights=weights.as_dict(),
            raw_dense=raw_dense,
            normalized_dense=normalized_dense,
            weighted_dense=weighted_dense,
            event_components=event_components,
            dense_total=dense_total,
            event_total=event_total,
        )


def _mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RewardV2Error(f"{label} must be a mapping")
    return raw


def load_reward_v2_config(
    path: str | Path = DEFAULT_REWARD_PATH_V2,
) -> RewardV2Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RewardV2Error("reward config root must be a mapping")
    if raw.get("schema") != REWARD_SCHEMA_V2:
        raise RewardV2Error(f"unsupported reward schema {raw.get('schema')!r}")
    if raw.get("training_enabled") is not True:
        raise RewardV2Error("reward v2 training_enabled must be true")
    if raw.get("aggregation") != "exactly_one_reward_per_15hz_policy_decision":
        raise RewardV2Error("reward aggregation must be exactly one value per 15 Hz decision")

    dense_raw = _mapping(raw.get("dense_families"), "dense_families")
    if tuple(dense_raw) != DENSE_FAMILIES:
        raise RewardV2Error("dense_families must be exactly the five v2 families")
    dense: dict[str, DenseFamilyConfig] = {}
    for name in DENSE_FAMILIES:
        item = _mapping(dense_raw[name], f"dense_families.{name}")
        clip = item.get("clip")
        if not isinstance(clip, list) or len(clip) != 2:
            raise RewardV2Error(f"dense_families.{name}.clip must contain two values")
        dense[name] = DenseFamilyConfig(
            name=name,
            sign=str(item.get("sign", "")),
            scale=_finite(item.get("scale"), f"dense_families.{name}.scale"),
            clip_min=_finite(clip[0], f"dense_families.{name}.clip[0]"),
            clip_max=_finite(clip[1], f"dense_families.{name}.clip[1]"),
        )

    stability_raw = _mapping(
        raw.get("body_stability_coefficients"), "body_stability_coefficients"
    )
    smoothness_raw = _mapping(
        raw.get("control_smoothness_coefficients"),
        "control_smoothness_coefficients",
    )
    events_raw = _mapping(raw.get("events"), "events")
    ownership_raw = _mapping(raw.get("signal_ownership"), "signal_ownership")
    audits_raw = _mapping(raw.get("audits"), "audits")
    return RewardV2Config(
        progress_gamma=_finite(raw.get("progress_gamma"), "progress_gamma"),
        physics_ticks_per_decision=int(raw.get("physics_ticks_per_decision", 0)),
        dense_families=dense,
        body_stability_coefficients={
            str(name): _finite(value, f"body_stability_coefficients.{name}")
            for name, value in stability_raw.items()
        },
        control_smoothness_coefficients={
            str(name): _finite(value, f"control_smoothness_coefficients.{name}")
            for name, value in smoothness_raw.items()
        },
        events={
            str(name): _finite(value, f"events.{name}")
            for name, value in events_raw.items()
        },
        signal_ownership={
            str(family): tuple(str(signal) for signal in signals)
            for family, signals in ownership_raw.items()
        },
        maximum_single_dense_family_fraction=_finite(
            audits_raw.get("maximum_single_dense_family_fraction"),
            "audits.maximum_single_dense_family_fraction",
        ),
        maximum_residual_regularization_fraction=_finite(
            audits_raw.get("maximum_residual_regularization_fraction"),
            "audits.maximum_residual_regularization_fraction",
        ),
        minimum_absolute_dense_return=_finite(
            audits_raw.get("minimum_absolute_dense_return"),
            "audits.minimum_absolute_dense_return",
        ),
        standing_still_max_reward=_finite(
            audits_raw.get("standing_still_max_reward"),
            "audits.standing_still_max_reward",
        ),
    )


@dataclass(frozen=True, slots=True)
class DuplicateSignalAudit:
    passed: bool
    duplicate_owners: Mapping[str, tuple[str, ...]]


def reward_duplicate_signal_audit(
    config: RewardV2Config | None = None,
) -> DuplicateSignalAudit:
    """Report any physical signal assigned to more than one dense family."""

    selected = config or load_reward_v2_config()
    owners: dict[str, list[str]] = {}
    for family in DENSE_FAMILIES:
        for signal in selected.signal_ownership[family]:
            owners.setdefault(signal, []).append(family)
    duplicates = {
        signal: tuple(families)
        for signal, families in sorted(owners.items())
        if len(families) > 1
    }
    return DuplicateSignalAudit(passed=not duplicates, duplicate_owners=duplicates)


@dataclass(frozen=True, slots=True)
class StandingStillExploitAudit:
    passed: bool
    maximum_reward: float
    rewards_by_phase: Mapping[str, float]
    threshold: float


def reward_standing_still_exploit_test(
    calculator: RewardCalculatorV2 | None = None,
) -> StandingStillExploitAudit:
    """Verify unchanged, zero-cost, event-free physical states earn no reward."""

    selected = calculator or RewardCalculatorV2.from_files()
    rewards: dict[str, float] = {}
    for state_id in STATE_IDS:
        objective = selected.objectives.phase(state_id)
        progress = PhysicalProgressState(
            state_id=state_id,
            normalized_terms={name: 0.5 for name in objective.potential_terms},
        )
        signals = RewardSignalsV2(
            previous_progress=progress,
            current_progress=progress,
            contact_motion_costs={name: 0.0 for name in objective.contact_cost_terms},
        )
        rewards[state_id] = selected.evaluate(signals).total
    maximum = max(rewards.values())
    threshold = selected.config.standing_still_max_reward
    return StandingStillExploitAudit(
        passed=maximum <= threshold + 1.0e-12,
        maximum_reward=maximum,
        rewards_by_phase=rewards,
        threshold=threshold,
    )


@dataclass(frozen=True, slots=True)
class SingleTermDominanceAudit:
    passed: bool
    absolute_contribution_by_family: Mapping[str, float]
    fraction_by_family: Mapping[str, float]
    dominant_family: str | None
    violations: tuple[str, ...]


def reward_single_term_dominance_audit(
    breakdowns: Iterable[RewardBreakdownV2],
    config: RewardV2Config | None = None,
) -> SingleTermDominanceAudit:
    """Audit absolute rollout contributions for family and residual dominance."""

    selected = config or load_reward_v2_config()
    absolute = {name: 0.0 for name in DENSE_FAMILIES}
    count = 0
    for breakdown in breakdowns:
        count += 1
        for name in DENSE_FAMILIES:
            absolute[name] += abs(_finite(breakdown.weighted_dense[name], name))
    total = sum(absolute.values())
    violations: list[str] = []
    if count == 0 or total < selected.minimum_absolute_dense_return:
        fractions = {name: 0.0 for name in DENSE_FAMILIES}
        dominant = None
        violations.append("insufficient_dense_signal")
    else:
        fractions = {name: absolute[name] / total for name in DENSE_FAMILIES}
        dominant = max(DENSE_FAMILIES, key=fractions.__getitem__)
        if fractions[dominant] > selected.maximum_single_dense_family_fraction:
            violations.append(f"single_family:{dominant}")
        if (
            fractions["residual_regularization"]
            > selected.maximum_residual_regularization_fraction
        ):
            violations.append("residual_regularization")
    return SingleTermDominanceAudit(
        passed=not violations,
        absolute_contribution_by_family=absolute,
        fraction_by_family=fractions,
        dominant_family=dominant,
        violations=tuple(violations),
    )


__all__ = [
    "DEFAULT_REWARD_PATH_V2",
    "EVENT_FAMILIES",
    "REWARD_SCHEMA_V2",
    "DenseFamilyConfig",
    "DuplicateSignalAudit",
    "RewardBreakdownV2",
    "RewardCalculatorV2",
    "RewardSignalsV2",
    "RewardV2Config",
    "RewardV2Error",
    "SingleTermDominanceAudit",
    "StandingStillExploitAudit",
    "load_reward_v2_config",
    "reward_duplicate_signal_audit",
    "reward_single_term_dominance_audit",
    "reward_standing_still_exploit_test",
]
