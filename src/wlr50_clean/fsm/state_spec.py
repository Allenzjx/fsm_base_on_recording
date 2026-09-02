"""Strict runtime projection of the authored P01--P13 state specification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml


EXPECTED_STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
ACTION_COUNT = 12
MAX_NORMAL_CORRECTION_FRACTION = 0.15
P03_RL_WHEEL_CORRECTION_FRACTION = -0.149
P03_RL_WHEEL_CHANNEL_INDEX = 10
P09_RL_WHEEL_CORRECTION_FRACTION = 0.106929411767305
P09_RL_WHEEL_CHANNEL_INDEX = 10


class Lifecycle(str, Enum):
    WAIT_ENTRY = "WAIT_ENTRY"
    EXECUTE_MOTION = "EXECUTE_MOTION"
    VERIFY_RESULT = "VERIFY_RESULT"
    RECOVERY = "RECOVERY"
    DONE = "DONE"


REQUIRED_LIFECYCLE = tuple(Lifecycle)


@dataclass(frozen=True)
class GuardSpec:
    name: str
    parameters: Mapping[str, Any]
    result: str | None = None


@dataclass(frozen=True)
class StateSpec:
    state_id: str
    macro_phase: int
    state_name: str
    physical_purpose: str
    lifecycle: tuple[Lifecycle, ...]
    entry_guards: tuple[GuardSpec, ...]
    completion_guards: tuple[GuardSpec, ...]
    hard_abort_guards: tuple[GuardSpec, ...]
    max_verify_wait_s: float
    recovery_max_verify_wait_s: float
    retry_budget: int
    next_state: str
    recovery_state: str
    completion_event: str
    transition_reason: str
    reference_actual_start_full12: tuple[float, ...]
    reference_actual_endpoint_full12: tuple[float, ...]
    normal_correction_fractions: tuple[float, ...]


@dataclass(frozen=True)
class FsmSpec:
    path: Path
    reference_version: str
    rear_leg_order: str
    decision_hz: float
    motion_hz: float
    watchdog_s: float
    states: tuple[StateSpec, ...]

    @property
    def decision_stride(self) -> int:
        return round(self.motion_hz / self.decision_hz)

    def state(self, state_id: str) -> StateSpec:
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise KeyError(state_id)


def _guard(value: Mapping[str, Any]) -> GuardSpec:
    if not isinstance(value, Mapping) or not value.get("guard"):
        raise ValueError(f"invalid guard specification: {value!r}")
    parameters = {
        str(key): item
        for key, item in value.items()
        if key not in {"guard", "result"}
    }
    return GuardSpec(
        name=str(value["guard"]),
        parameters=MappingProxyType(parameters),
        result=str(value["result"]) if value.get("result") else None,
    )


def _guards(values: Sequence[Mapping[str, Any]], label: str) -> tuple[GuardSpec, ...]:
    if not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return tuple(_guard(value) for value in values)


def _parse_state(value: Mapping[str, Any]) -> StateSpec:
    lifecycle = tuple(Lifecycle(str(item)) for item in value["lifecycle"])
    actual_endpoint = tuple(
        float(item) for item in value["reference_actual_endpoint_full12"]
    )
    actual_delta = tuple(
        float(item) for item in value["reference_actual_delta_full12"]
    )
    if len(actual_endpoint) != 12 or len(actual_delta) != 12:
        raise ValueError(f"{value.get('state_id')}: actual Full12 reference is incomplete")
    normal_correction = tuple(
        float(item)
        for item in value.get("normal_correction_fractions", (0.0,) * ACTION_COUNT)
    )
    return StateSpec(
        state_id=str(value["state_id"]),
        macro_phase=int(value["macro_phase"]),
        state_name=str(value["state_name"]),
        physical_purpose=str(value["physical_purpose"]),
        lifecycle=lifecycle,
        entry_guards=_guards(value["entry_conditions"], "entry_conditions"),
        completion_guards=_guards(
            value["completion_conditions"], "completion_conditions"
        ),
        hard_abort_guards=_guards(
            value["hard_abort_conditions"], "hard_abort_conditions"
        ),
        max_verify_wait_s=float(value["max_verify_wait"]),
        recovery_max_verify_wait_s=float(
            value.get("recovery_max_verify_wait", value["max_verify_wait"])
        ),
        retry_budget=int(value["retry_budget"]),
        next_state=str(value["next_state"]),
        recovery_state=str(value["recovery_state"]),
        completion_event=str(value["completion_event"]),
        transition_reason=str(value["transition_reason"]),
        reference_actual_start_full12=tuple(
            endpoint - delta
            for endpoint, delta in zip(actual_endpoint, actual_delta, strict=True)
        ),
        reference_actual_endpoint_full12=actual_endpoint,
        normal_correction_fractions=normal_correction,
    )


def load_fsm_spec(path: Path) -> FsmSpec:
    """Load only runtime-safe fields; provenance rows remain unavailable here."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wlr50_clean.fsm_states.v1":
        raise ValueError("unexpected FSM schema")
    if payload.get("reference_version") != "v010_20260806_220745_363972_manual":
        raise ValueError("FSM is not locked to v010")
    if payload.get("rear_leg_order") != "RR_FIRST":
        raise ValueError("FSM is not RR_FIRST")
    states = tuple(_parse_state(value) for value in payload["states"])
    spec = FsmSpec(
        path=path.resolve(),
        reference_version=str(payload["reference_version"]),
        rear_leg_order=str(payload["rear_leg_order"]),
        decision_hz=float(payload["decision_hz"]),
        motion_hz=float(payload["motion_hz"]),
        watchdog_s=float(payload["state_progress_watchdog_s"]),
        states=states,
    )
    _validate(spec, payload["states"])
    return spec


def _validate(spec: FsmSpec, raw_states: Sequence[Mapping[str, Any]]) -> None:
    ids = tuple(state.state_id for state in spec.states)
    if ids != EXPECTED_STATE_IDS:
        raise ValueError("FSM must contain fixed ordered P01-P13")
    if spec.motion_hz != 120.0 or spec.decision_hz != 15.0:
        raise ValueError("FSM rates must be 120 Hz motion and 15 Hz decisions")
    ratio = spec.motion_hz / spec.decision_hz
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("motion/decision rate ratio must be integral")
    if abs(spec.watchdog_s - 0.5) > 1e-9:
        raise ValueError("state-progress watchdog must be 0.5 seconds")
    allowed_results = {
        "TASK_FAILURE_BODY_COLLISION",
        "TASK_FAILURE_WHEEL_ONLY_CLIMB",
        "SAFETY_ABORT",
    }
    for index, (state, raw) in enumerate(zip(spec.states, raw_states, strict=True)):
        if state.macro_phase != index + 1:
            raise ValueError(f"{state.state_id}: macro phase mismatch")
        if state.lifecycle != REQUIRED_LIFECYCLE:
            raise ValueError(f"{state.state_id}: lifecycle contract changed")
        expected_next = (
            EXPECTED_STATE_IDS[index + 1]
            if index + 1 < len(EXPECTED_STATE_IDS)
            else "TASK_COMPLETE"
        )
        if state.next_state != expected_next:
            raise ValueError(f"{state.state_id}: invalid next state")
        if state.retry_budget != 1:
            raise ValueError(f"{state.state_id}: exactly one retry is required")
        if (
            not math.isfinite(state.max_verify_wait_s)
            or state.max_verify_wait_s <= 0.0
        ):
            raise ValueError(f"{state.state_id}: verify wait must be positive")
        if (
            not math.isfinite(state.recovery_max_verify_wait_s)
            or state.recovery_max_verify_wait_s < state.max_verify_wait_s
        ):
            raise ValueError(
                f"{state.state_id}: recovery verify wait must be finite and cannot be shorter"
            )
        if raw.get("elapsed_time_is_not_completion_evidence") is not True:
            raise ValueError(f"{state.state_id}: elapsed time cannot complete a state")
        if (
            len(state.reference_actual_start_full12) != 12
            or len(state.reference_actual_endpoint_full12) != 12
        ):
            raise ValueError(f"{state.state_id}: actual endpoint must be full12")
        expected_normal_correction = [0.0] * ACTION_COUNT
        if state.state_id == "P03":
            expected_normal_correction[P03_RL_WHEEL_CHANNEL_INDEX] = (
                P03_RL_WHEEL_CORRECTION_FRACTION
            )
        elif state.state_id == "P09":
            expected_normal_correction[P09_RL_WHEEL_CHANNEL_INDEX] = (
                P09_RL_WHEEL_CORRECTION_FRACTION
            )
        if (
            len(state.normal_correction_fractions) != ACTION_COUNT
            or any(
                not math.isfinite(value)
                or abs(value) > MAX_NORMAL_CORRECTION_FRACTION + 1.0e-12
                for value in state.normal_correction_fractions
            )
            or state.normal_correction_fractions
            != tuple(expected_normal_correction)
        ):
            raise ValueError(
                f"{state.state_id}: invalid bounded normal correction"
            )
        for guard in state.hard_abort_guards:
            if guard.result not in allowed_results:
                raise ValueError(f"{state.state_id}: invalid hard-abort result")
