"""Runtime-safe loader for the compact v010-derived motion contract.

Only the compact JSON generated during offline preparation is loaded here. The
raw event stream, its cursor, and its timestamps are deliberately unavailable
to production control code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SERVO_COUNT = 8
ACTION_COUNT = 12
EXPECTED_PHASE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
PHYSICS_HZ = 120.0
DECISION_STRIDE = 8
WHEEL_HARD_LIMIT_RAD_S = 2.0943951023931953
MAX_CUMULATIVE_CORRECTION_FRACTION = 0.15
REBOUND_STATE_ID = "P09"
REBOUND_KIND = "pre_endpoint_wheel_rebound_alignment"
REBOUND_PROBE_CHANNEL = "rear_right_knee"
REBOUND_PROBE_CHANNEL_INDEX = 7
REBOUND_PROBES = (
    (858, -51.055799822535),
    (859, -51.191638624749),
)
REBOUND_LAG_THRESHOLD_DEG = 1.7
REBOUND_CORRECTION_CHANNEL = "front_left_ankle"
REBOUND_CORRECTION_CHANNEL_INDEX = 8
REBOUND_LOGICAL_BIAS_RAD_S = 0.33
REBOUND_PRE_ENDPOINT_NATIVE_RAD_S = -1.07
REBOUND_PRE_ENDPOINT_FINAL_RAD_S = -0.74
REBOUND_POST_ENDPOINT_NATIVE_RAD_S = 0.0
REBOUND_POST_ENDPOINT_FINAL_RAD_S = 0.33


def _vector(value: Sequence[Any], *, label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != ACTION_COUNT:
        raise ValueError(f"{label}: expected {ACTION_COUNT} values")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label}: expected a boolean")
    return value


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
class DriveFeedbackProbe:
    motion_tick: int
    reference_actual_deg: float


@dataclass(frozen=True)
class DriveFeedbackSpec:
    kind: str
    probe_channel: str
    probe_channel_index: int
    correction_channel: str
    correction_channel_index: int
    probe_samples: tuple[DriveFeedbackProbe, ...]
    lag_threshold_deg: float
    required_consecutive_samples: int
    first_bias_tick: int
    last_bias_tick: int
    teardown_tick: int
    logical_bias_rad_s: float
    reference_wheel_integral_rad: float
    additional_wheel_integral_rad: float
    resulting_wheel_integral_rad: float
    cumulative_fraction_of_reference: float
    reference_wheel_peak_abs_rad_s: float
    resulting_wheel_peak_abs_rad_s: float
    instantaneous_direction_reversal: bool


@dataclass(frozen=True)
class EntryVelocityAlignmentSpec:
    channel: str
    channel_index: int
    reference_velocity_deg_s: float
    relative_limit: float


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
    drive_feedback: DriveFeedbackSpec | None
    entry_velocity_alignment: EntryVelocityAlignmentSpec | None

    def nominal_at(self, elapsed_s: float) -> tuple[float, ...]:
        """Sample the causal source request without advancing FSM state."""

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
        # Waypoint timestamps are source dispatch onsets, not future target
        # arrival times.  The mature runtime mapper owns the continuous 120 Hz
        # servo-drive slew after each request; all logical channels therefore
        # hold their latest authored value until the next waypoint.
        return left.full12

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
    raw_feedback = value.get("drive_feedback")
    drive_feedback = None
    if isinstance(raw_feedback, Mapping):
        drive_feedback = DriveFeedbackSpec(
            kind=str(raw_feedback["kind"]),
            probe_channel=str(raw_feedback["probe_channel"]),
            probe_channel_index=int(raw_feedback["probe_channel_index"]),
            correction_channel=str(raw_feedback["correction_channel"]),
            correction_channel_index=int(raw_feedback["correction_channel_index"]),
            probe_samples=tuple(
                DriveFeedbackProbe(
                    motion_tick=int(row["motion_tick"]),
                    reference_actual_deg=float(row["reference_actual_deg"]),
                )
                for row in raw_feedback["probe_samples"]
            ),
            lag_threshold_deg=float(raw_feedback["lag_threshold_deg"]),
            required_consecutive_samples=int(
                raw_feedback["required_consecutive_samples"]
            ),
            first_bias_tick=int(raw_feedback["first_bias_tick"]),
            last_bias_tick=int(raw_feedback["last_bias_tick"]),
            teardown_tick=int(raw_feedback["teardown_tick"]),
            logical_bias_rad_s=float(raw_feedback["logical_bias_rad_s"]),
            reference_wheel_integral_rad=float(
                raw_feedback["reference_wheel_integral_rad"]
            ),
            additional_wheel_integral_rad=float(
                raw_feedback["additional_wheel_integral_rad"]
            ),
            resulting_wheel_integral_rad=float(
                raw_feedback["resulting_wheel_integral_rad"]
            ),
            cumulative_fraction_of_reference=float(
                raw_feedback["cumulative_fraction_of_reference"]
            ),
            reference_wheel_peak_abs_rad_s=float(
                raw_feedback["reference_wheel_peak_abs_rad_s"]
            ),
            resulting_wheel_peak_abs_rad_s=float(
                raw_feedback["resulting_wheel_peak_abs_rad_s"]
            ),
            instantaneous_direction_reversal=_boolean(
                raw_feedback["instantaneous_direction_reversal"],
                label="drive_feedback.instantaneous_direction_reversal",
            ),
        )
    raw_alignment = value.get("entry_velocity_alignment")
    entry_velocity_alignment = None
    if isinstance(raw_alignment, Mapping):
        entry_velocity_alignment = EntryVelocityAlignmentSpec(
            channel=str(raw_alignment["channel"]),
            channel_index=int(raw_alignment["channel_index"]),
            reference_velocity_deg_s=float(
                raw_alignment["reference_velocity_deg_s"]
            ),
            relative_limit=float(raw_alignment["relative_limit"]),
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
        drive_feedback=drive_feedback,
        entry_velocity_alignment=entry_velocity_alignment,
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
        feedback = phase.drive_feedback
        if feedback is not None:
            if (
                phase.state_id != REBOUND_STATE_ID
                or feedback.kind != REBOUND_KIND
            ):
                raise ValueError(f"{phase.state_id}: unknown drive-feedback kind")
            if (
                feedback.probe_channel != REBOUND_PROBE_CHANNEL
                or feedback.probe_channel_index != REBOUND_PROBE_CHANNEL_INDEX
                or full12_order[feedback.probe_channel_index]
                != feedback.probe_channel
                or feedback.correction_channel != REBOUND_CORRECTION_CHANNEL
                or feedback.correction_channel_index
                != REBOUND_CORRECTION_CHANNEL_INDEX
                or full12_order[feedback.correction_channel_index]
                != feedback.correction_channel
            ):
                raise ValueError(f"{phase.state_id}: drive-feedback channel mismatch")
            probe_ticks = tuple(item.motion_tick for item in feedback.probe_samples)
            probes = tuple(
                (item.motion_tick, item.reference_actual_deg)
                for item in feedback.probe_samples
            )
            if (
                probes != REBOUND_PROBES
                or feedback.required_consecutive_samples != len(REBOUND_PROBES)
                or abs(feedback.lag_threshold_deg - REBOUND_LAG_THRESHOLD_DEG)
                > 1.0e-12
                or feedback.first_bias_tick <= probe_ticks[-1]
                or feedback.last_bias_tick < feedback.first_bias_tick
                or feedback.teardown_tick != feedback.last_bias_tick + 1
            ):
                raise ValueError(f"{phase.state_id}: invalid drive-feedback timing")
            endpoint_tick = round(phase.active_duration_s * PHYSICS_HZ)
            if (
                probe_ticks != (endpoint_tick - 6, endpoint_tick - 5)
                or feedback.first_bias_tick != endpoint_tick - 4
                or feedback.last_bias_tick != endpoint_tick + DECISION_STRIDE - 1
                or feedback.teardown_tick != endpoint_tick + DECISION_STRIDE
            ):
                raise ValueError(
                    f"{phase.state_id}: drive-feedback does not match the "
                    "bounded pre-endpoint wheel rebound"
                )
            values = (
                feedback.lag_threshold_deg,
                feedback.logical_bias_rad_s,
                feedback.reference_wheel_integral_rad,
                feedback.additional_wheel_integral_rad,
                feedback.resulting_wheel_integral_rad,
                feedback.cumulative_fraction_of_reference,
                feedback.reference_wheel_peak_abs_rad_s,
                feedback.resulting_wheel_peak_abs_rad_s,
                *(probe.reference_actual_deg for probe in feedback.probe_samples),
            )
            active_ticks = feedback.last_bias_tick - feedback.first_bias_tick + 1
            additional = feedback.logical_bias_rad_s * active_ticks / PHYSICS_HZ
            reference = feedback.reference_wheel_integral_rad
            derived_reference = _zoh_channel_integral(
                phase, feedback.correction_channel_index
            )
            resulting = reference + additional
            cumulative = (
                math.inf if reference == 0.0 else abs(additional) / abs(reference)
            )
            reference_peak = _zoh_channel_peak_abs(
                phase, feedback.correction_channel_index
            )
            resulting_peak = _feedback_channel_peak_abs(
                phase,
                feedback.correction_channel_index,
                first_bias_tick=feedback.first_bias_tick,
                last_bias_tick=feedback.last_bias_tick,
                bias=feedback.logical_bias_rad_s,
            )
            pre_endpoint_native = tuple(
                _channel_at_motion_tick(
                    phase, feedback.correction_channel_index, tick
                )
                for tick in range(feedback.first_bias_tick, endpoint_tick)
            )
            post_endpoint_native = tuple(
                _channel_at_motion_tick(
                    phase, feedback.correction_channel_index, tick
                )
                for tick in range(endpoint_tick, feedback.last_bias_tick + 1)
            )
            if (
                any(not math.isfinite(value) for value in values)
                or feedback.lag_threshold_deg <= 0.0
                or abs(feedback.logical_bias_rad_s - REBOUND_LOGICAL_BIAS_RAD_S)
                > 1.0e-12
                or not pre_endpoint_native
                or not post_endpoint_native
                or any(
                    abs(native - REBOUND_PRE_ENDPOINT_NATIVE_RAD_S) > 1.0e-12
                    or abs(
                        native
                        + feedback.logical_bias_rad_s
                        - REBOUND_PRE_ENDPOINT_FINAL_RAD_S
                    )
                    > 1.0e-12
                    for native in pre_endpoint_native
                )
                or any(
                    abs(native - REBOUND_POST_ENDPOINT_NATIVE_RAD_S) > 1.0e-12
                    or abs(
                        native
                        + feedback.logical_bias_rad_s
                        - REBOUND_POST_ENDPOINT_FINAL_RAD_S
                    )
                    > 1.0e-12
                    for native in post_endpoint_native
                )
                or abs(phase.end_full12[feedback.correction_channel_index])
                > 1.0e-12
                or feedback.logical_bias_rad_s * reference >= 0.0
                or feedback.instantaneous_direction_reversal is not True
                or abs(reference - derived_reference) > 1.0e-12
                or abs(additional - feedback.additional_wheel_integral_rad)
                > 1.0e-12
                or abs(resulting - feedback.resulting_wheel_integral_rad)
                > 1.0e-12
                or abs(cumulative - feedback.cumulative_fraction_of_reference) > 1.0e-12
                or cumulative > MAX_CUMULATIVE_CORRECTION_FRACTION + 1.0e-12
                or abs(reference_peak - feedback.reference_wheel_peak_abs_rad_s)
                > 1.0e-12
                or abs(resulting_peak - feedback.resulting_wheel_peak_abs_rad_s)
                > 1.0e-12
                or resulting_peak > reference_peak + 1.0e-12
                or resulting_peak > WHEEL_HARD_LIMIT_RAD_S + 1.0e-12
            ):
                raise ValueError(f"{phase.state_id}: drive-feedback budget is invalid")
        alignment = phase.entry_velocity_alignment
        if alignment is not None:
            if (
                alignment.channel_index < 0
                or alignment.channel_index >= SERVO_COUNT
                or full12_order[alignment.channel_index] != alignment.channel
                or alignment.channel not in phase.active_channels
                or alignment.reference_velocity_deg_s <= 0.0
                or abs(alignment.relative_limit - 0.15) > 1.0e-12
            ):
                raise ValueError(
                    f"{phase.state_id}: invalid entry-velocity alignment"
                )
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


def _zoh_channel_integral(phase: MotionPhase, channel_index: int) -> float:
    """Integrate one compact command channel under the contract's ZOH rule."""

    result = 0.0
    for index, waypoint in enumerate(phase.waypoints):
        left = min(phase.active_duration_s, max(0.0, waypoint.time_s))
        right = (
            phase.active_duration_s
            if index + 1 == len(phase.waypoints)
            else min(
                phase.active_duration_s,
                max(0.0, phase.waypoints[index + 1].time_s),
            )
        )
        result += waypoint.full12[channel_index] * max(0.0, right - left)
    return result


def _zoh_channel_peak_abs(phase: MotionPhase, channel_index: int) -> float:
    """Return the absolute peak of one frozen ZOH command channel."""

    return max(
        abs(waypoint.full12[channel_index]) for waypoint in phase.waypoints
    )


def _channel_at_motion_tick(
    phase: MotionPhase, channel_index: int, tick: int
) -> float:
    """Match the motion sequencer's integer endpoint dispatch semantics."""

    endpoint_tick = round(phase.active_duration_s * PHYSICS_HZ)
    if tick >= endpoint_tick:
        return phase.end_full12[channel_index]
    return phase.nominal_at(tick / PHYSICS_HZ)[channel_index]


def _feedback_channel_peak_abs(
    phase: MotionPhase,
    channel_index: int,
    *,
    first_bias_tick: int,
    last_bias_tick: int,
    bias: float,
) -> float:
    """Re-derive the peak after applying a bounded tick-indexed correction."""

    endpoint_tick = round(phase.active_duration_s * PHYSICS_HZ)
    values = [
        abs(waypoint.full12[channel_index]) for waypoint in phase.waypoints
    ]
    values.extend(
        abs(
            _channel_at_motion_tick(phase, channel_index, tick) + bias
        )
        for tick in range(first_bias_tick, min(last_bias_tick, endpoint_tick - 1) + 1)
    )
    if last_bias_tick >= endpoint_tick:
        values.extend(
            abs(bias)
            for _ in range(endpoint_tick, last_bias_tick + 1)
        )
    return max(values)
