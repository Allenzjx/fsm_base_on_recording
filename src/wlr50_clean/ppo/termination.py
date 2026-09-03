"""Explicit PPO episode termination and truncation classifications."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import yaml


TERMINATION_SCHEMA = "wlr50_clean.ppo_termination.v1"
DEFAULT_TERMINATION_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_termination.yaml"
)


class TerminationReason(str, Enum):
    SUCCESS = "SUCCESS"
    BODY_COLLISION = "BODY_COLLISION"
    WHEEL_ONLY_CLIMB = "WHEEL_ONLY_CLIMB"
    FALL = "FALL"
    NAN_INF = "NAN_INF"
    HARD_JOINT_LIMIT = "HARD_JOINT_LIMIT"
    PHYSICS_EXPLOSION = "PHYSICS_EXPLOSION"
    TIMEOUT = "TIMEOUT"
    REFERENCE_CONFORMANCE = "REFERENCE_CONFORMANCE"


HARD_FAILURE_REASONS = frozenset(
    {
        TerminationReason.BODY_COLLISION,
        TerminationReason.WHEEL_ONLY_CLIMB,
        TerminationReason.FALL,
        TerminationReason.NAN_INF,
        TerminationReason.HARD_JOINT_LIMIT,
        TerminationReason.PHYSICS_EXPLOSION,
    }
)


class TerminationConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TerminationConfig:
    schema: str
    training_enabled: bool
    timeout_s: float
    conformance_mode: str
    priority: tuple[TerminationReason, ...]
    path: Path


@dataclass(frozen=True, slots=True)
class TerminationSignals:
    success: bool = False
    body_collision: bool = False
    wheel_only_climb: bool = False
    fall: bool = False
    nan_inf: bool = False
    hard_joint_limit: bool = False
    physics_explosion: bool = False
    timeout: bool = False
    reference_conformance_outside_30pct: bool = False


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    terminated: bool
    truncated: bool
    reason: TerminationReason | None
    triggered_reasons: tuple[TerminationReason, ...]
    diagnostics: tuple[TerminationReason, ...]

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise TerminationConfigurationError(
                "an episode cannot be terminated and truncated simultaneously"
            )
        if (self.terminated or self.truncated) != (self.reason is not None):
            raise TerminationConfigurationError("done status and primary reason disagree")


def load_termination_config(
    path: Path | str = DEFAULT_TERMINATION_PATH,
) -> TerminationConfig:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != TERMINATION_SCHEMA:
        raise TerminationConfigurationError("unexpected PPO termination schema")
    if bool(payload.get("training_enabled")):
        raise TerminationConfigurationError("PPO training must remain disabled")
    timeout_s = float(payload["timeout_s"])
    if not math.isfinite(timeout_s) or not 0.0 < timeout_s <= 200.0:
        raise TerminationConfigurationError("timeout must be positive and at most 200 s")
    mode = str(payload["conformance_outside_30pct"])
    if mode != "diagnostic_only":
        raise TerminationConfigurationError(
            "Recording reference divergence must remain diagnostic-only"
        )
    try:
        priority = tuple(TerminationReason(str(item)) for item in payload["priority"])
    except (TypeError, ValueError) as exc:
        raise TerminationConfigurationError("invalid termination priority") from exc
    required = HARD_FAILURE_REASONS | {
        TerminationReason.SUCCESS,
        TerminationReason.TIMEOUT,
    }
    if len(priority) != len(set(priority)) or set(priority) != required:
        raise TerminationConfigurationError(
            "priority must contain every hard reason, SUCCESS and TIMEOUT exactly once"
        )
    return TerminationConfig(
        schema=TERMINATION_SCHEMA,
        training_enabled=False,
        timeout_s=timeout_s,
        conformance_mode=mode,
        priority=priority,
        path=selected,
    )


class TerminationEvaluator:
    def __init__(self, config: TerminationConfig | None = None) -> None:
        self.config = config or load_termination_config()

    def evaluate(self, signals: TerminationSignals) -> TerminationDecision:
        active = {
            TerminationReason.SUCCESS: bool(signals.success),
            TerminationReason.BODY_COLLISION: bool(signals.body_collision),
            TerminationReason.WHEEL_ONLY_CLIMB: bool(signals.wheel_only_climb),
            TerminationReason.FALL: bool(signals.fall),
            TerminationReason.NAN_INF: bool(signals.nan_inf),
            TerminationReason.HARD_JOINT_LIMIT: bool(signals.hard_joint_limit),
            TerminationReason.PHYSICS_EXPLOSION: bool(signals.physics_explosion),
            TerminationReason.TIMEOUT: bool(signals.timeout),
        }
        triggered = tuple(reason for reason in self.config.priority if active[reason])
        diagnostics = (
            (TerminationReason.REFERENCE_CONFORMANCE,)
            if signals.reference_conformance_outside_30pct
            else ()
        )
        if triggered:
            primary = triggered[0]
            truncated = primary is TerminationReason.TIMEOUT
            return TerminationDecision(
                terminated=not truncated,
                truncated=truncated,
                reason=primary,
                triggered_reasons=triggered,
                diagnostics=diagnostics,
            )
        return TerminationDecision(False, False, None, (), diagnostics)
