"""Structured, sensor-driven guard evaluation with no time-as-success path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .state_spec import GuardSpec
from .task_result import TaskResult


@runtime_checkable
class GuardObservation(Protocol):
    """Optional narrow integration API implemented by the sensing layer."""

    def resolve_guard(
        self, name: str, parameters: Mapping[str, Any]
    ) -> "GuardEvidence | bool | float | None": ...


@dataclass(frozen=True)
class GuardEvidence:
    name: str
    passed: bool
    value: Any = None
    source: str = "missing"
    reason: str = ""


@dataclass(frozen=True)
class GuardReport:
    passed: bool
    evidence: tuple[GuardEvidence, ...]

    @property
    def first_blocker(self) -> GuardEvidence | None:
        return next((item for item in self.evidence if not item.passed), None)


@dataclass(frozen=True)
class HardAbort:
    result: TaskResult
    evidence: GuardEvidence


def _member(observation: Any, name: str, default: Any = None) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


def _guard_key(name: str, parameters: Mapping[str, Any]) -> tuple[str, ...]:
    leg = parameters.get("leg")
    return (f"{name}:{leg}", name) if leg else (name,)


def _coerce_evidence(name: str, value: Any, source: str) -> GuardEvidence | None:
    if isinstance(value, GuardEvidence):
        return value
    if value is None:
        return None
    if isinstance(value, Mapping) and "passed" in value:
        return GuardEvidence(
            name=name,
            passed=bool(value["passed"]),
            value=value.get("value"),
            source=str(value.get("source", source)),
            reason=str(value.get("reason", "")),
        )
    if isinstance(value, bool):
        return GuardEvidence(name, value, value, source)
    if isinstance(value, (int, float)):
        return GuardEvidence(name, bool(value), value, source)
    return None


class GuardEvaluator:
    """Resolve authored guards from live observations and controller-local facts.

    Sensing may implement ``resolve_guard`` or expose a ``guards`` mapping.  The
    latter may use either ``guard_name`` or ``guard_name:LEG`` keys.  Missing
    completion evidence is always false; elapsed time is never consulted here.
    """

    def __init__(self) -> None:
        self._debounce_start: dict[tuple[str, str], float] = {}

    def reset_state(self, state_id: str) -> None:
        for key in tuple(self._debounce_start):
            if key[0] == state_id:
                del self._debounce_start[key]

    def evaluate_all(
        self,
        guards: Sequence[GuardSpec],
        observation: Any,
        *,
        state_id: str,
        sim_time_s: float,
        local_facts: Mapping[str, Any] | None = None,
    ) -> GuardReport:
        evidence = tuple(
            self.evaluate(
                guard,
                observation,
                state_id=state_id,
                sim_time_s=sim_time_s,
                local_facts=local_facts or {},
            )
            for guard in guards
        )
        return GuardReport(all(item.passed for item in evidence), evidence)

    def evaluate(
        self,
        guard: GuardSpec,
        observation: Any,
        *,
        state_id: str,
        sim_time_s: float,
        local_facts: Mapping[str, Any],
    ) -> GuardEvidence:
        name = guard.name
        # This new-run requirement is deliberately enforced in the controller
        # from raw measured velocity.  A pre-latched boolean cannot bypass the
        # continuous 0.5 s debounce.
        if name == "measured_wheel_velocity_stable_decay":
            inferred = self._infer(
                guard,
                observation,
                state_id=state_id,
                sim_time_s=sim_time_s,
            )
            if inferred is not None:
                return inferred
            return GuardEvidence(
                name=name,
                passed=False,
                source="missing",
                reason="raw measured wheel velocities are required for debounce",
            )
        for key in _guard_key(name, guard.parameters):
            if key in local_facts:
                found = _coerce_evidence(name, local_facts[key], "controller")
                if found is not None:
                    return found

        resolver = getattr(observation, "resolve_guard", None)
        if callable(resolver):
            found = _coerce_evidence(
                name, resolver(name, guard.parameters), "sensor.resolve_guard"
            )
            if found is not None:
                return found

        guard_values = _member(observation, "guards", {})
        if isinstance(guard_values, Mapping):
            for key in _guard_key(name, guard.parameters):
                if key in guard_values:
                    found = _coerce_evidence(name, guard_values[key], "sensor.guards")
                    if found is not None:
                        return found

        for key in _guard_key(name, guard.parameters):
            found = _coerce_evidence(name, _member(observation, key), "sensor.field")
            if found is not None:
                return found

        inferred = self._infer(
            guard,
            observation,
            state_id=state_id,
            sim_time_s=sim_time_s,
        )
        if inferred is not None:
            return inferred
        return GuardEvidence(
            name=name,
            passed=False,
            source="missing",
            reason="required live evidence was not supplied",
        )

    def _infer(
        self,
        guard: GuardSpec,
        observation: Any,
        *,
        state_id: str,
        sim_time_s: float,
    ) -> GuardEvidence | None:
        inverse = {
            "no_body_obstacle_collision": "body_collision_persistent_or_penetrating",
            "joint_hard_limits_valid": "joint_hard_limit_violation",
        }
        if guard.name in inverse:
            bad = _member(observation, inverse[guard.name])
            if bad is None and guard.name == "no_body_obstacle_collision":
                status = _member(observation, "body_collision")
                bad = _member(status, "detected")
            if isinstance(bad, bool):
                return GuardEvidence(
                    guard.name, not bad, bad, "sensor.inverse", "inverse safety flag"
                )

        if guard.name == "non_finite_observation_or_command":
            finite = _member(observation, "all_finite")
            if isinstance(finite, bool):
                return GuardEvidence(guard.name, not finite, finite, "sensor.all_finite")

        if guard.name == "body_collision_persistent_or_penetrating":
            status = _member(observation, "body_collision")
            detected = _member(status, "detected")
            if isinstance(detected, bool):
                return GuardEvidence(
                    guard.name,
                    detected,
                    {
                        "detected": detected,
                        "persistent": _member(status, "persistent"),
                        "penetration_m": _member(status, "geometry_penetration_m"),
                    },
                    "sensor.body_collision",
                )

        if guard.name == "measured_wheel_velocity_stable_decay":
            velocities = None
            for field in (
                "measured_wheel_velocity_rad_s",
                "measured_wheel_velocities_rad_s",
                "wheel_velocities_rad_s",
                "wheel_velocity_rad_s",
            ):
                velocities = _member(observation, field)
                if velocities is not None:
                    break
            if velocities is None:
                wheels = _member(observation, "wheels")
                if isinstance(wheels, Mapping):
                    velocities = tuple(
                        _member(wheel, "velocity_rad_s") for wheel in wheels.values()
                    )
            flat = _numeric_vector(velocities)
            if flat:
                threshold = float(guard.parameters["absolute_threshold_rad_s"])
                debounce_s = float(guard.parameters["debounce_s"])
                maximum = max(abs(item) for item in flat)
                key = (state_id, guard.name)
                if maximum <= threshold:
                    self._debounce_start.setdefault(key, sim_time_s)
                else:
                    self._debounce_start.pop(key, None)
                held = sim_time_s - self._debounce_start.get(key, sim_time_s)
                return GuardEvidence(
                    guard.name,
                    maximum <= threshold and held + 1e-12 >= debounce_s,
                    {
                        "max_abs_rad_s": maximum,
                        "threshold_rad_s": threshold,
                        "stable_for_s": held,
                        "reference_tail_peak_rad_s": guard.parameters.get(
                            "reference_tail_peak_rad_s"
                        ),
                    },
                    "sensor.wheel_velocity_debounce",
                    "velocity must remain inside the measured v010 post-stop tail plus 15% envelope for the full debounce",
                )
        return None

    def first_hard_abort(
        self,
        guards: Sequence[GuardSpec],
        observation: Any,
        *,
        state_id: str,
        sim_time_s: float,
        local_facts: Mapping[str, Any] | None = None,
    ) -> HardAbort | None:
        """Return the first asserted abort predicate, preserving its class."""

        for guard in guards:
            evidence = self.evaluate(
                guard,
                observation,
                state_id=state_id,
                sim_time_s=sim_time_s,
                local_facts=local_facts or {},
            )
            if evidence.passed:
                if guard.result is None:
                    raise ValueError(f"hard guard {guard.name} has no result")
                return HardAbort(TaskResult(guard.result), evidence)
        return None


def _numeric_vector(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = value.values()
    if isinstance(value, (str, bytes)):
        return ()
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()
    return result if all(math.isfinite(item) for item in result) else ()


def observation_progress_vector(observation: Any) -> tuple[float, ...]:
    """Best-effort continuous progress signal used only by the stall watchdog."""

    names = (
        "progress_vector",
        "root_position_m",
        "root_position_w_m",
        "wheel_positions_rad",
        "actual_full12",
        "joint_positions_deg",
    )
    result: list[float] = []
    for name in names:
        result.extend(_numeric_vector(_member(observation, name)))
    base = _member(observation, "base")
    result.extend(_numeric_vector(_member(base, "position_w_m")))
    joints = _member(observation, "joints")
    if isinstance(joints, Mapping):
        result.extend(
            float(value)
            for joint in joints.values()
            for value in (
                _member(joint, "position_deg", 0.0),
                _member(joint, "velocity_deg_s", 0.0),
            )
        )
    wheels = _member(observation, "wheels")
    if isinstance(wheels, Mapping):
        for wheel in wheels.values():
            result.extend(_numeric_vector(_member(wheel, "center_w_m")))
            velocity = _member(wheel, "velocity_rad_s")
            if isinstance(velocity, (int, float)):
                result.append(float(velocity))
    return tuple(result)
