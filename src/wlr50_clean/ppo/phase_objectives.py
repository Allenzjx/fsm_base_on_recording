"""Validated phase-specific objectives for residual PPO reward v2.

This module deliberately has no Isaac Sim dependency.  It turns normalized
live physical measurements into a phase-local potential, embeds that potential
in the authoritative P01--P13 ordering, and supplies continuously interpolated
reward weights for the transfer-aware phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


PHASE_OBJECTIVES_SCHEMA = "wlr50_clean.ppo_phase_objectives.v2"
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
DENSE_FAMILIES = (
    "phase_task_progress",
    "body_stability",
    "contact_motion_quality",
    "control_smoothness",
    "residual_regularization",
)
TRANSFER_PHASES = frozenset({"P01", "P04", "P08", "P10", "P11"})
DEFAULT_PHASE_OBJECTIVES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_phase_objectives_v2.yaml"
)

# These are the table values supplied in the phase-specific PPO specification.
# P08/P11 use the active row here and have an additional prescribed capture row.
PROMPT_PHASE_WEIGHTS: Mapping[str, tuple[float, ...]] = {
    "P01": (0.50, 0.20, 0.15, 0.10, 0.05),
    "P02": (0.35, 0.35, 0.15, 0.10, 0.05),
    "P03": (0.30, 0.30, 0.25, 0.10, 0.05),
    "P04": (0.50, 0.20, 0.15, 0.10, 0.05),
    "P05": (0.35, 0.30, 0.20, 0.10, 0.05),
    "P06": (0.40, 0.30, 0.15, 0.10, 0.05),
    "P07": (0.35, 0.35, 0.10, 0.15, 0.05),
    "P08": (0.55, 0.15, 0.15, 0.10, 0.05),
    "P09": (0.35, 0.30, 0.20, 0.10, 0.05),
    "P10": (0.50, 0.20, 0.15, 0.10, 0.05),
    "P11": (0.55, 0.15, 0.15, 0.10, 0.05),
    "P12": (0.30, 0.40, 0.15, 0.10, 0.05),
    "P13": (0.30, 0.40, 0.05, 0.20, 0.05),
}
PROMPT_CAPTURE_WEIGHTS: Mapping[str, tuple[float, ...]] = {
    "P08": (0.30, 0.40, 0.15, 0.10, 0.05),
    "P11": (0.30, 0.40, 0.15, 0.10, 0.05),
}
FORBIDDEN_PROGRESS_SOURCE_TOKENS = (
    "elapsed_time",
    "recording_time",
    "time_cursor",
    "accepted_steps",
)


class PhaseObjectiveError(ValueError):
    """A phase objective or physical-progress sample violates the v2 contract."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PhaseObjectiveError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PhaseObjectiveError(f"{label} must be finite")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0 or result > 1.0:
        raise PhaseObjectiveError(f"{label} must be within [0, 1]; got {result}")
    return result


def _contains_forbidden_progress_source(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in FORBIDDEN_PROGRESS_SOURCE_TOKENS)


@dataclass(frozen=True, slots=True)
class PhaseWeights(Mapping[str, float]):
    """The five normalized dense-family weights in their canonical order."""

    phase_task_progress: float
    body_stability: float
    contact_motion_quality: float
    control_smoothness: float
    residual_regularization: float

    def __post_init__(self) -> None:
        values = self.values_tuple
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise PhaseObjectiveError("dense-family weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise PhaseObjectiveError(
                f"dense-family weights must sum to 1; got {sum(values):.17g}"
            )

    @property
    def values_tuple(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in DENSE_FAMILIES)

    def __getitem__(self, key: str) -> float:
        if key not in DENSE_FAMILIES:
            raise KeyError(key)
        return float(getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return iter(DENSE_FAMILIES)

    def __len__(self) -> int:
        return len(DENSE_FAMILIES)

    def as_dict(self) -> dict[str, float]:
        return {name: self[name] for name in DENSE_FAMILIES}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], label: str) -> "PhaseWeights":
        if set(raw) != set(DENSE_FAMILIES):
            missing = sorted(set(DENSE_FAMILIES) - set(raw))
            extra = sorted(set(raw) - set(DENSE_FAMILIES))
            raise PhaseObjectiveError(
                f"{label} must contain exactly the five dense families; "
                f"missing={missing}, extra={extra}"
            )
        return cls(*(_finite(raw[name], f"{label}.{name}") for name in DENSE_FAMILIES))

    @classmethod
    def from_values(cls, values: tuple[float, ...], label: str) -> "PhaseWeights":
        if len(values) != len(DENSE_FAMILIES):
            raise PhaseObjectiveError(f"{label} must contain five values")
        return cls(*values)

    def lerp(self, other: "PhaseWeights", fraction: float) -> "PhaseWeights":
        alpha = _unit_interval(fraction, "weight interpolation fraction")
        values = tuple(
            left + alpha * (right - left)
            for left, right in zip(self.values_tuple, other.values_tuple, strict=True)
        )
        # Linear interpolation preserves the exact unit sum mathematically; tiny
        # float drift is removed so downstream audit output remains deterministic.
        drift = 1.0 - sum(values)
        adjusted = values[:-1] + (values[-1] + drift,)
        return PhaseWeights.from_values(adjusted, "interpolated weights")


@dataclass(frozen=True, slots=True)
class TransferSchedule:
    active_end: float
    capture_end: float
    active_weights: PhaseWeights
    capture_weights: PhaseWeights
    settle_weights: PhaseWeights
    active_level_penalty_fraction: float
    capture_level_penalty_fraction: float
    settle_level_penalty_fraction: float

    def __post_init__(self) -> None:
        active_end = _unit_interval(self.active_end, "transfer active_end")
        capture_end = _unit_interval(self.capture_end, "transfer capture_end")
        if not 0.0 < active_end < capture_end < 1.0:
            raise PhaseObjectiveError(
                "transfer knots must satisfy 0 < active_end < capture_end < 1"
            )
        levels = (
            _unit_interval(
                self.active_level_penalty_fraction,
                "active level-penalty fraction",
            ),
            _unit_interval(
                self.capture_level_penalty_fraction,
                "capture level-penalty fraction",
            ),
            _unit_interval(
                self.settle_level_penalty_fraction,
                "settle level-penalty fraction",
            ),
        )
        if levels[0] > levels[1] or levels[1] > levels[2]:
            raise PhaseObjectiveError(
                "level-penalty fraction must increase monotonically through transfer"
            )

    def _interpolation(self, physical_progress: float) -> tuple[str, float]:
        progress = _unit_interval(physical_progress, "physical phase progress")
        if progress <= self.active_end:
            return "TRANSFER_ACTIVE", 0.0
        if progress <= self.capture_end:
            alpha = (progress - self.active_end) / (
                self.capture_end - self.active_end
            )
            return "CAPTURE", alpha
        alpha = (progress - self.capture_end) / (1.0 - self.capture_end)
        return "SETTLE", alpha

    def weights_at(self, physical_progress: float) -> PhaseWeights:
        stage, alpha = self._interpolation(physical_progress)
        if stage == "TRANSFER_ACTIVE":
            return self.active_weights
        if stage == "CAPTURE":
            return self.active_weights.lerp(self.capture_weights, alpha)
        return self.capture_weights.lerp(self.settle_weights, alpha)

    def level_penalty_fraction_at(self, physical_progress: float) -> float:
        stage, alpha = self._interpolation(physical_progress)
        if stage == "TRANSFER_ACTIVE":
            return self.active_level_penalty_fraction
        if stage == "CAPTURE":
            return self.active_level_penalty_fraction + alpha * (
                self.capture_level_penalty_fraction
                - self.active_level_penalty_fraction
            )
        return self.capture_level_penalty_fraction + alpha * (
            self.settle_level_penalty_fraction
            - self.capture_level_penalty_fraction
        )

    def stage_at(self, physical_progress: float) -> str:
        return self._interpolation(physical_progress)[0]


@dataclass(frozen=True, slots=True)
class PhaseObjective:
    state_id: str
    name: str
    primary_objective: str
    stability_mode: str
    active_leg: str | None
    potential_terms: Mapping[str, float]
    contact_cost_terms: tuple[str, ...]
    prompt_weights: PhaseWeights
    transfer_schedule: TransferSchedule | None = None

    def __post_init__(self) -> None:
        if self.state_id not in STATE_IDS:
            raise PhaseObjectiveError(f"unknown phase {self.state_id!r}")
        if not self.name.strip() or not self.primary_objective.strip():
            raise PhaseObjectiveError(f"{self.state_id} description may not be empty")
        if self.active_leg not in {None, "FL", "FR", "RL", "RR"}:
            raise PhaseObjectiveError(
                f"{self.state_id}.active_leg is invalid: {self.active_leg!r}"
            )
        if not self.potential_terms:
            raise PhaseObjectiveError(f"{self.state_id} has no physical potential terms")
        if any(_contains_forbidden_progress_source(name) for name in self.potential_terms):
            raise PhaseObjectiveError(
                f"{self.state_id} potential uses a forbidden time/recording source"
            )
        potential_sum = 0.0
        for name, weight in self.potential_terms.items():
            if not str(name).strip():
                raise PhaseObjectiveError(f"{self.state_id} has an empty potential name")
            numeric = _finite(weight, f"{self.state_id}.potential_terms.{name}")
            if numeric < 0.0:
                raise PhaseObjectiveError(f"{self.state_id} potential weights must be >= 0")
            potential_sum += numeric
        if not math.isclose(potential_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise PhaseObjectiveError(
                f"{self.state_id} potential weights must sum to 1; got {potential_sum}"
            )
        if not self.contact_cost_terms or len(set(self.contact_cost_terms)) != len(
            self.contact_cost_terms
        ):
            raise PhaseObjectiveError(
                f"{self.state_id} contact/motion cost terms must be non-empty and unique"
            )
        expects_transfer = self.state_id in TRANSFER_PHASES
        if expects_transfer != (self.stability_mode == "TRANSFER_AWARE"):
            raise PhaseObjectiveError(
                f"{self.state_id} transfer-aware mode does not match authoritative set"
            )
        if expects_transfer != (self.transfer_schedule is not None):
            raise PhaseObjectiveError(
                f"{self.state_id} transfer schedule does not match authoritative set"
            )

    def weights_at(self, physical_progress: float) -> PhaseWeights:
        _unit_interval(physical_progress, "physical phase progress")
        if self.transfer_schedule is None:
            return self.prompt_weights
        return self.transfer_schedule.weights_at(physical_progress)

    def level_penalty_fraction_at(self, physical_progress: float) -> float:
        _unit_interval(physical_progress, "physical phase progress")
        if self.transfer_schedule is None:
            return 1.0
        return self.transfer_schedule.level_penalty_fraction_at(physical_progress)

    def substage_at(self, physical_progress: float) -> str:
        _unit_interval(physical_progress, "physical phase progress")
        if self.transfer_schedule is None:
            return "PHASE_ACTIVE"
        return self.transfer_schedule.stage_at(physical_progress)


@dataclass(frozen=True, slots=True)
class PhaseObjectivesConfig:
    phases: Mapping[str, PhaseObjective]

    def __post_init__(self) -> None:
        if tuple(self.phases) != STATE_IDS:
            raise PhaseObjectiveError(
                "phase objectives must preserve the authoritative P01--P13 order"
            )

    def phase(self, state_id: str) -> PhaseObjective:
        try:
            return self.phases[state_id]
        except KeyError as exc:
            raise PhaseObjectiveError(f"unknown phase {state_id!r}") from exc


@dataclass(frozen=True, slots=True)
class PhysicalProgressState:
    """Normalized physical goal measurements for one authoritative FSM state."""

    state_id: str
    normalized_terms: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.state_id not in STATE_IDS:
            raise PhaseObjectiveError(f"unknown phase {self.state_id!r}")
        if not self.normalized_terms:
            raise PhaseObjectiveError("physical progress requires at least one term")
        for name, value in self.normalized_terms.items():
            if _contains_forbidden_progress_source(str(name)):
                raise PhaseObjectiveError(
                    f"physical progress may not use time/recording source {name!r}"
                )
            _unit_interval(value, f"physical progress term {name!r}")


def _load_transfer_schedule(raw: Mapping[str, Any], label: str) -> TransferSchedule:
    expected = {"active", "capture", "settle"}
    weights = raw.get("weights")
    levels = raw.get("level_penalty_fraction")
    if not isinstance(weights, Mapping) or set(weights) != expected:
        raise PhaseObjectiveError(f"{label}.weights must contain active/capture/settle")
    if not isinstance(levels, Mapping) or set(levels) != expected:
        raise PhaseObjectiveError(
            f"{label}.level_penalty_fraction must contain active/capture/settle"
        )
    return TransferSchedule(
        active_end=_finite(raw.get("active_end"), f"{label}.active_end"),
        capture_end=_finite(raw.get("capture_end"), f"{label}.capture_end"),
        active_weights=PhaseWeights.from_mapping(weights["active"], f"{label}.active"),
        capture_weights=PhaseWeights.from_mapping(
            weights["capture"], f"{label}.capture"
        ),
        settle_weights=PhaseWeights.from_mapping(weights["settle"], f"{label}.settle"),
        active_level_penalty_fraction=_finite(
            levels["active"], f"{label}.level_penalty_fraction.active"
        ),
        capture_level_penalty_fraction=_finite(
            levels["capture"], f"{label}.level_penalty_fraction.capture"
        ),
        settle_level_penalty_fraction=_finite(
            levels["settle"], f"{label}.level_penalty_fraction.settle"
        ),
    )


def load_phase_objectives(
    path: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
) -> PhaseObjectivesConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PhaseObjectiveError("phase objective config root must be a mapping")
    if raw.get("schema") != PHASE_OBJECTIVES_SCHEMA:
        raise PhaseObjectiveError(f"unsupported phase objective schema {raw.get('schema')!r}")
    if tuple(raw.get("state_ids", ())) != STATE_IDS:
        raise PhaseObjectiveError("state_ids must preserve authoritative P01--P13")
    if tuple(raw.get("dense_families", ())) != DENSE_FAMILIES:
        raise PhaseObjectiveError("dense_families must be exactly the five v2 families")
    progress_contract = raw.get("progress_contract")
    if not isinstance(progress_contract, Mapping):
        raise PhaseObjectiveError("progress_contract must be a mapping")
    forbidden = tuple(str(value) for value in progress_contract.get("forbidden_sources", ()))
    if not {"elapsed_time", "recording_time_cursor", "accepted_steps"}.issubset(
        set(forbidden)
    ):
        raise PhaseObjectiveError("progress_contract omits required forbidden sources")

    phases_raw = raw.get("phases")
    if not isinstance(phases_raw, Mapping) or tuple(phases_raw) != STATE_IDS:
        raise PhaseObjectiveError("phases must preserve authoritative P01--P13 order")
    phases: dict[str, PhaseObjective] = {}
    for state_id in STATE_IDS:
        phase_raw = phases_raw[state_id]
        if not isinstance(phase_raw, Mapping):
            raise PhaseObjectiveError(f"{state_id} must be a mapping")
        potential_raw = phase_raw.get("potential_terms")
        if not isinstance(potential_raw, Mapping):
            raise PhaseObjectiveError(f"{state_id}.potential_terms must be a mapping")
        contact_raw = phase_raw.get("contact_cost_terms")
        if not isinstance(contact_raw, list):
            raise PhaseObjectiveError(f"{state_id}.contact_cost_terms must be a list")
        prompt_weights = PhaseWeights.from_mapping(
            phase_raw.get("prompt_weights", {}), f"{state_id}.prompt_weights"
        )
        prescribed = PhaseWeights.from_values(
            PROMPT_PHASE_WEIGHTS[state_id], f"{state_id} prescribed weights"
        )
        if prompt_weights.values_tuple != prescribed.values_tuple:
            raise PhaseObjectiveError(
                f"{state_id}.prompt_weights differ from the supplied phase-weight table"
            )
        transfer_raw = phase_raw.get("transfer_schedule")
        transfer = None
        if transfer_raw is not None:
            if not isinstance(transfer_raw, Mapping):
                raise PhaseObjectiveError(f"{state_id}.transfer_schedule must be a mapping")
            transfer = _load_transfer_schedule(
                transfer_raw, f"{state_id}.transfer_schedule"
            )
        if state_id in PROMPT_CAPTURE_WEIGHTS:
            if transfer is None:
                raise PhaseObjectiveError(f"{state_id} requires capture weights")
            prescribed_capture = PhaseWeights.from_values(
                PROMPT_CAPTURE_WEIGHTS[state_id], f"{state_id} prescribed capture"
            )
            if transfer.capture_weights.values_tuple != prescribed_capture.values_tuple:
                raise PhaseObjectiveError(
                    f"{state_id} capture weights differ from the supplied table"
                )
        phases[state_id] = PhaseObjective(
            state_id=state_id,
            name=str(phase_raw.get("name", "")),
            primary_objective=str(phase_raw.get("primary_objective", "")),
            stability_mode=str(phase_raw.get("stability_mode", "")),
            active_leg=phase_raw.get("active_leg"),
            potential_terms={
                str(name): _finite(value, f"{state_id}.potential_terms.{name}")
                for name, value in potential_raw.items()
            },
            contact_cost_terms=tuple(str(value) for value in contact_raw),
            prompt_weights=prompt_weights,
            transfer_schedule=transfer,
        )
    return PhaseObjectivesConfig(phases=phases)


def phase_local_potential(
    state: PhysicalProgressState,
    objectives: PhaseObjectivesConfig,
) -> float:
    """Return ``phi_p(s)`` from only the phase's normalized physical goals."""

    objective = objectives.phase(state.state_id)
    expected = set(objective.potential_terms)
    supplied = set(state.normalized_terms)
    if supplied != expected:
        raise PhaseObjectiveError(
            f"{state.state_id} physical terms do not match objective; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    value = sum(
        objective.potential_terms[name]
        * _unit_interval(state.normalized_terms[name], f"{state.state_id}.{name}")
        for name in objective.potential_terms
    )
    return max(0.0, min(1.0, value))


def global_phase_potential(
    state: PhysicalProgressState,
    objectives: PhaseObjectivesConfig,
) -> float:
    """Return ``(phase_index - 1 + phi_p(s)) / 13``."""

    phase_index = STATE_IDS.index(state.state_id) + 1
    return (phase_index - 1 + phase_local_potential(state, objectives)) / len(
        STATE_IDS
    )


def potential_based_progress(
    previous: PhysicalProgressState,
    current: PhysicalProgressState,
    objectives: PhaseObjectivesConfig,
    *,
    gamma: float,
) -> float:
    """Compute ``gamma * Phi(s[t+1]) - Phi(s[t])`` for a legal FSM edge."""

    discount = _finite(gamma, "progress gamma")
    if discount <= 0.0 or discount > 1.0:
        raise PhaseObjectiveError("progress gamma must be within (0, 1]")
    previous_index = STATE_IDS.index(previous.state_id)
    current_index = STATE_IDS.index(current.state_id)
    if current_index not in {previous_index, min(previous_index + 1, len(STATE_IDS) - 1)}:
        raise PhaseObjectiveError(
            f"illegal FSM progress edge {previous.state_id}->{current.state_id}"
        )
    return discount * global_phase_potential(current, objectives) - global_phase_potential(
        previous, objectives
    )


__all__ = [
    "DEFAULT_PHASE_OBJECTIVES_PATH",
    "DENSE_FAMILIES",
    "FORBIDDEN_PROGRESS_SOURCE_TOKENS",
    "PHASE_OBJECTIVES_SCHEMA",
    "PROMPT_CAPTURE_WEIGHTS",
    "PROMPT_PHASE_WEIGHTS",
    "STATE_IDS",
    "TRANSFER_PHASES",
    "PhaseObjective",
    "PhaseObjectiveError",
    "PhaseObjectivesConfig",
    "PhaseWeights",
    "PhysicalProgressState",
    "TransferSchedule",
    "global_phase_potential",
    "load_phase_objectives",
    "phase_local_potential",
    "potential_based_progress",
]
