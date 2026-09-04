"""Trainable PPO termination contract that delegates task truth to the frozen FSM.

The residual policy never invents a success condition.  Callers translate the
authoritative controller termination and live safety signals into
``TerminationSignalsV2``; this module only applies the versioned priority and
the 200 second truncation rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .termination import (
    HARD_FAILURE_REASONS,
    TerminationConfigurationError,
    TerminationDecision,
    TerminationReason,
)


TERMINATION_SCHEMA_V2 = "wlr50_clean.ppo_termination.v2"
DEFAULT_TERMINATION_PATH_V2 = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_termination_v2.yaml"
)


@dataclass(frozen=True, slots=True)
class TerminationConfigV2:
    schema: str
    training_enabled: bool
    timeout_s: float
    success_source: str
    conformance_mode: str
    priority: tuple[TerminationReason, ...]
    sensor_contract_failure: str
    path: Path


@dataclass(frozen=True, slots=True)
class TerminationSignalsV2:
    """Signals sampled after a live physics tick.

    ``authoritative_success`` may only be set from the frozen controller's
    task termination.  Recording divergence remains a diagnostic and cannot
    end an episode.
    """

    authoritative_success: bool = False
    body_collision: bool = False
    wheel_only_climb: bool = False
    fall: bool = False
    nan_inf: bool = False
    hard_joint_limit: bool = False
    physics_explosion: bool = False
    reference_conformance_outside_30pct: bool = False
    sensor_contract_valid: bool = True


def load_termination_config_v2(
    path: Path | str = DEFAULT_TERMINATION_PATH_V2,
) -> TerminationConfigV2:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != TERMINATION_SCHEMA_V2:
        raise TerminationConfigurationError("unexpected PPO termination v2 schema")
    if not bool(payload.get("training_enabled")):
        raise TerminationConfigurationError("PPO termination v2 must be training enabled")
    if payload.get("success_source") != "frozen_fsm_authoritative_task_success":
        raise TerminationConfigurationError("success must come from the frozen FSM")
    if payload.get("conformance_outside_30pct") != "diagnostic_only":
        raise TerminationConfigurationError("Recording divergence must be diagnostic-only")
    timeout_s = float(payload.get("timeout_s", float("nan")))
    if not math.isfinite(timeout_s) or not 0.0 < timeout_s <= 200.0:
        raise TerminationConfigurationError("timeout must be finite and at most 200 seconds")
    try:
        priority = tuple(TerminationReason(str(item)) for item in payload["priority"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TerminationConfigurationError("invalid termination v2 priority") from exc
    required = HARD_FAILURE_REASONS | {
        TerminationReason.SUCCESS,
        TerminationReason.TIMEOUT,
    }
    if len(priority) != len(set(priority)) or set(priority) != required:
        raise TerminationConfigurationError(
            "priority must contain every hard failure, SUCCESS and TIMEOUT once"
        )
    sensor_failure = str(payload.get("sensor_contract_failure", ""))
    if sensor_failure != "infrastructure_abort":
        raise TerminationConfigurationError(
            "invalid live sensing must fail closed as infrastructure_abort"
        )
    return TerminationConfigV2(
        schema=TERMINATION_SCHEMA_V2,
        training_enabled=True,
        timeout_s=timeout_s,
        success_source="frozen_fsm_authoritative_task_success",
        conformance_mode="diagnostic_only",
        priority=priority,
        sensor_contract_failure=sensor_failure,
        path=selected,
    )


class TerminationEvaluatorV2:
    def __init__(self, config: TerminationConfigV2 | None = None) -> None:
        self.config = config or load_termination_config_v2()

    def evaluate(
        self,
        signals: TerminationSignalsV2,
        *,
        episode_time_s: float,
    ) -> TerminationDecision:
        now = float(episode_time_s)
        if not math.isfinite(now) or now < 0.0:
            raise TerminationConfigurationError("episode_time_s must be finite and non-negative")
        if not signals.sensor_contract_valid:
            # Infrastructure failures are intentionally raised rather than
            # mislabeled as physical task failures or PPO truncations.
            raise RuntimeError("live sensor contract invalid: infrastructure_abort")
        active = {
            TerminationReason.SUCCESS: bool(signals.authoritative_success),
            TerminationReason.BODY_COLLISION: bool(signals.body_collision),
            TerminationReason.WHEEL_ONLY_CLIMB: bool(signals.wheel_only_climb),
            TerminationReason.FALL: bool(signals.fall),
            TerminationReason.NAN_INF: bool(signals.nan_inf),
            TerminationReason.HARD_JOINT_LIMIT: bool(signals.hard_joint_limit),
            TerminationReason.PHYSICS_EXPLOSION: bool(signals.physics_explosion),
            TerminationReason.TIMEOUT: now + 1.0e-12 >= self.config.timeout_s,
        }
        triggered = tuple(reason for reason in self.config.priority if active[reason])
        diagnostics = (
            (TerminationReason.REFERENCE_CONFORMANCE,)
            if signals.reference_conformance_outside_30pct
            else ()
        )
        if not triggered:
            return TerminationDecision(False, False, None, (), diagnostics)
        reason = triggered[0]
        return TerminationDecision(
            terminated=reason is not TerminationReason.TIMEOUT,
            truncated=reason is TerminationReason.TIMEOUT,
            reason=reason,
            triggered_reasons=triggered,
            diagnostics=diagnostics,
        )
