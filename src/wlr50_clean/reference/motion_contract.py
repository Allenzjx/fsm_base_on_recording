"""Runtime-safe loader for the compact v010-derived motion contract.

Only the compact JSON generated during offline preparation is loaded here. The
raw event stream, its cursor, and its timestamps are deliberately unavailable
to production control code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SERVO_COUNT = 8
ACTION_COUNT = 12
EXPECTED_PHASE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))


def _vector(value: Sequence[Any], *, label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != ACTION_COUNT:
        raise ValueError(f"{label}: expected {ACTION_COUNT} values")
    return result


@dataclass(frozen=True)
class Waypoint:
    time_s: float
    full12: tuple[float, ...]
    changed_channels: tuple[str, ...]
    atomic_channels: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class AtomicGroup:
    time_s: float
    channels: tuple[str, ...]
    same_physics_tick: bool
    motion_start_skew_s: float
    source_full12_atomic: bool
    required_runtime_channels: tuple[str, ...]


@dataclass(frozen=True)
class MotionPhase:
    state_id: str
    macro_phase: int
    state_name: str
    physical_purpose: str
    active_duration_s: float
    start_full12: tuple[float, ...]
    end_full12: tuple[float, ...]
    delta_full12: tuple[float, ...]
    active_channels: tuple[str, ...]
    waypoints: tuple[Waypoint, ...]
    atomic_groups: tuple[AtomicGroup, ...]
    completion_event: str
    action_mask_full12: tuple[int, ...]

    def nominal_at(self, elapsed_s: float) -> tuple[float, ...]:
        """Sample a smooth full action without advancing any FSM state."""

        time_s = min(max(float(elapsed_s), 0.0), self.active_duration_s)
        if time_s >= self.active_duration_s:
            return self.end_full12
        left_index = 0
        for index, waypoint in enumerate(self.waypoints):
            if waypoint.time_s <= time_s + 1e-12:
                left_index = index
            else:
                break
        left = self.waypoints[left_index]
        if left_index + 1 >= len(self.waypoints):
            return left.full12
        right = self.waypoints[left_index + 1]
        span = right.time_s - left.time_s
        if span <= 1e-12:
            return right.full12
        progress = min(max((time_s - left.time_s) / span, 0.0), 1.0)
        # A stateful 150 deg/s limiter in MotionExecutor follows this continuous
        # interpolation. Linear blending avoids the 1.875x peak of a quintic
        # while remaining monotonic and closer to the frozen v010 target slew.
        blend = progress
        values = list(left.full12)
        for index in range(SERVO_COUNT):
            values[index] += blend * (right.full12[index] - left.full12[index])
        # Wheel targets are velocities. Preserve the event onset and integral by
        # holding each reference value until its next compact waypoint.
        return tuple(values)

    def atomic_groups_between(
        self, previous_elapsed_s: float, elapsed_s: float
    ) -> tuple[AtomicGroup, ...]:
        lower = float(previous_elapsed_s)
        upper = float(elapsed_s)
        return tuple(
            group
            for group in self.atomic_groups
            if lower < group.time_s <= upper + 1e-12
            or (lower < 0.0 and abs(group.time_s) <= 1e-12)
        )


@dataclass(frozen=True)
class MotionContract:
    path: Path
    reference_version: str
    rear_leg_order: str
    physics_hz: float
    decision_hz: float
    full12_order: tuple[str, ...]
    relative_tolerance: float
    servo_rate_limit_deg_s: float
    phases: tuple[MotionPhase, ...]

    def phase(self, state_id: str) -> MotionPhase:
        for phase in self.phases:
            if phase.state_id == state_id:
                return phase
        raise KeyError(state_id)


def _parse_phase(value: Mapping[str, Any]) -> MotionPhase:
    waypoints = tuple(
        Waypoint(
            time_s=float(row["time_s"]),
            full12=_vector(row["full12"], label="waypoint.full12"),
            changed_channels=tuple(str(item) for item in row["changed_channels"]),
            atomic_channels=tuple(str(item) for item in row["atomic_channels"]),
            kind=str(row["kind"]),
        )
        for row in value["waypoints"]
    )
    atomic_groups = tuple(
        AtomicGroup(
            time_s=float(row["time_s"]),
            channels=tuple(str(item) for item in row["channels"]),
            same_physics_tick=bool(row["same_physics_tick"]),
            motion_start_skew_s=float(row["motion_start_skew_s"]),
            source_full12_atomic=bool(row.get("source_full12_atomic", False)),
            required_runtime_channels=tuple(
                str(item) for item in row.get("required_runtime_channels", [])
            ),
        )
        for row in value["atomic_groups"]
    )
    return MotionPhase(
        state_id=str(value["state_id"]),
        macro_phase=int(value["macro_phase"]),
        state_name=str(value["state_name"]),
        physical_purpose=str(value["physical_purpose"]),
        active_duration_s=float(value["active_duration_s"]),
        start_full12=_vector(value["start_full12"], label="start_full12"),
        end_full12=_vector(value["end_full12"], label="end_full12"),
        delta_full12=_vector(value["delta_full12"], label="delta_full12"),
        active_channels=tuple(str(item) for item in value["active_channels"]),
        waypoints=waypoints,
        atomic_groups=atomic_groups,
        completion_event=str(value["completion_event"]),
        action_mask_full12=tuple(int(item) for item in value["ppo_action_mask_full12"]),
    )


def load_motion_contract(path: Path) -> MotionContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wlr50_clean.recording_motion_contract.v1":
        raise ValueError("unexpected motion-contract schema")
    if payload.get("reference_version") != "v010_20260806_220745_363972_manual":
        raise ValueError("motion contract is not locked to v010")
    if payload.get("rear_leg_order") != "RR_FIRST":
        raise ValueError("motion contract is not RR_FIRST")
    if payload.get("cross_version_splice") is not False:
        raise ValueError("cross-version motion is forbidden")
    semantics = payload.get("execution_semantics", {})
    if semantics.get("full12_output_each_physics_tick") is not True:
        raise ValueError("contract does not require full12 physics-tick output")
    if semantics.get("full12_output_atomic_write") is not True:
        raise ValueError("contract does not require atomic full12 writes")
    phases = tuple(_parse_phase(value) for value in payload["phases"])
    _validate_phases(phases, tuple(str(item) for item in payload["full12_order"]))
    return MotionContract(
        path=path.resolve(),
        reference_version=str(payload["reference_version"]),
        rear_leg_order=str(payload["rear_leg_order"]),
        physics_hz=float(payload["physics_hz"]),
        decision_hz=float(payload["decision_hz"]),
        full12_order=tuple(str(item) for item in payload["full12_order"]),
        relative_tolerance=float(payload["tolerance"]["relative"]),
        servo_rate_limit_deg_s=float(
            payload["servo_reference_velocity_deg_s"]
        ),
        phases=phases,
    )


def _validate_phases(
    phases: tuple[MotionPhase, ...], full12_order: tuple[str, ...]
) -> None:
    if tuple(phase.state_id for phase in phases) != EXPECTED_PHASE_IDS:
        raise ValueError("the compact contract must contain ordered P01-P13")
    if len(full12_order) != ACTION_COUNT or len(set(full12_order)) != ACTION_COUNT:
        raise ValueError("full12 channel order is invalid")
    source_atomic_count = 0
    for phase in phases:
        if phase.active_duration_s <= 0.0:
            raise ValueError(f"{phase.state_id}: active duration must be positive")
        if len(phase.action_mask_full12) != ACTION_COUNT:
            raise ValueError(f"{phase.state_id}: invalid PPO action mask")
        if not phase.waypoints or abs(phase.waypoints[0].time_s) > 1e-9:
            raise ValueError(f"{phase.state_id}: missing phase-entry waypoint")
        previous = -1.0
        for waypoint in phase.waypoints:
            if waypoint.time_s + 1e-9 < previous:
                raise ValueError(f"{phase.state_id}: waypoint time moved backwards")
            if waypoint.time_s > phase.active_duration_s + 1e-6:
                raise ValueError(f"{phase.state_id}: waypoint exceeds active duration")
            previous = waypoint.time_s
        for group in phase.atomic_groups:
            if not group.same_physics_tick or group.motion_start_skew_s > 1e-9:
                raise ValueError(f"{phase.state_id}: non-atomic reference group")
            if group.source_full12_atomic:
                source_atomic_count += 1
                if group.required_runtime_channels != full12_order:
                    raise ValueError(
                        f"{phase.state_id}: source full12 event lost channel atomicity"
                    )
    if source_atomic_count != 4:
        raise ValueError("v010 contract must preserve four source full12 events")
