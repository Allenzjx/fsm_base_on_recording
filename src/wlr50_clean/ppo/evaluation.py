"""Read-only ingestion and paired evaluation for authoritative live-run streams.

The functions in this module consume ``observation_120hz.jsonl``,
``full12_commands_120hz.jsonl``, transition/event streams, and the run manifest.
They never write into a run directory and never infer that a candidate is
"improved" from its filename.  Promotion is returned only after the paired,
evidence-based gates have been evaluated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER

from .observation_schema_v2 import ACTIVE_LEG_BY_PHASE, LEG_TO_WHEEL, WHEEL_RADIUS_M
from .phase_objectives import DENSE_FAMILIES, STATE_IDS
from .reward_v2 import EVENT_FAMILIES
from .stability_metrics import (
    EpisodeOutcome,
    PHASE_IDS,
    PRIORITY_PHASES,
    PromotionDecision,
    StabilitySample,
    compare_phase_metrics,
    evaluate_promotion,
    residual_activity_by_phase,
    summarize_phase_samples,
)


OBSERVATION_STREAM_SCHEMA = "wlr50_clean.live_observation.v1"
OBSERVATION_FILENAME = "observation_120hz.jsonl"
COMMAND_FILENAME = "full12_commands_120hz.jsonl"
TRANSITION_FILENAME = "state_transitions.jsonl"
TASK_EVENT_FILENAME = "task_events.jsonl"
MANIFEST_FILENAME = "trial_manifest.json"


class OfflineEvaluationError(ValueError):
    """A live-run stream is missing, malformed, inconsistent, or not comparable."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OfflineEvaluationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise OfflineEvaluationError(f"{label} must be finite")
    return result


def _vector(values: Any, size: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise OfflineEvaluationError(f"{label} must contain {size} numbers")
    try:
        result = tuple(_finite(value, label) for value in values)
    except TypeError as exc:
        raise OfflineEvaluationError(f"{label} must contain {size} numbers") from exc
    if len(result) != size:
        raise OfflineEvaluationError(
            f"{label} must contain {size} numbers; received {len(result)}"
        )
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfflineEvaluationError(f"{label} must be a mapping")
    return value


def iter_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Yield JSON-object lines with source line numbers in any failure."""

    selected = Path(path)
    with selected.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise OfflineEvaluationError(
                    f"{selected.name}:{line_number} is an empty JSONL record"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OfflineEvaluationError(
                    f"{selected.name}:{line_number} is invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise OfflineEvaluationError(
                    f"{selected.name}:{line_number} must be a JSON object"
                )
            yield record


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OfflineEvaluationError(f"{path.name} is invalid JSON: {exc.msg}") from exc
    return _mapping(result, path.name)


def _normalize_quaternion(values: Any, label: str) -> tuple[float, float, float, float]:
    quaternion = _vector(values, 4, label)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1.0e-12:
        raise OfflineEvaluationError(f"{label} has zero norm")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _quat_conjugate(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    return (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])


def _quat_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quaternion_to_euler_xyz(
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def _quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotate_world_to_body(
    quaternion: Sequence[float], vector_world: Sequence[float]
) -> tuple[float, float, float]:
    result = _quaternion_matrix(quaternion).T @ np.asarray(vector_world, dtype=float)
    return tuple(float(value) for value in result)  # type: ignore[return-value]


def _wheel_normal_forces(observation: Mapping[str, Any]) -> tuple[float, ...]:
    wheels = _mapping(observation.get("wheels"), "observation.wheels")
    contacts = _mapping(observation.get("contacts"), "observation.contacts")
    values: list[float] = []
    for wheel_name in WHEEL_ORDER:
        wheel = _mapping(wheels.get(wheel_name), f"wheels.{wheel_name}")
        body_name = str(wheel.get("body_name", ""))
        contact = _mapping(contacts.get(body_name), f"contacts.{body_name}")
        normal_force = 0.0
        for pair_name in ("ground", "obstacle"):
            pair = _mapping(contact.get(pair_name), f"contacts.{body_name}.{pair_name}")
            if pair.get("pair_verified") is not True:
                raise OfflineEvaluationError(
                    f"{body_name}/{pair_name} force is not exact-pair verified"
                )
            normal_force += max(
                0.0,
                _finite(
                    pair.get("normal_force_n"),
                    f"contacts.{body_name}.{pair_name}.normal_force_n",
                ),
            )
        values.append(normal_force)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class LiveRunCalibration:
    level_reference_orientation_wxyz: tuple[float, float, float, float]
    raw_reference_roll_rad: float
    raw_reference_pitch_rad: float
    raw_reference_yaw_rad: float
    home_joint_positions_deg8: tuple[float, ...]
    wheel_normal_force_baseline_n4: tuple[float, ...]
    window_start_s: float
    window_end_s: float
    sample_count: int
    maximum_linear_speed_m_s: float
    maximum_angular_speed_rad_s: float
    quality_passed: bool
    source: str = "reset_stable_window_quaternion_mean"


def calibrate_live_run(
    observations: Iterable[Mapping[str, Any]],
    *,
    stable_window_s: float = 0.5,
    maximum_linear_speed_m_s: float = 0.10,
    maximum_angular_speed_rad_s: float = 0.20,
) -> LiveRunCalibration:
    """Calibrate level/home/load references from a short reset window."""

    duration = _finite(stable_window_s, "stable_window_s")
    if duration <= 0.0:
        raise OfflineEvaluationError("stable_window_s must be positive")
    rows: list[Mapping[str, Any]] = []
    start: float | None = None
    for observation in observations:
        if observation.get("schema") != OBSERVATION_STREAM_SCHEMA:
            raise OfflineEvaluationError("unexpected live observation schema")
        time_s = _finite(observation.get("simulation_time_s"), "simulation_time_s")
        if start is None:
            start = time_s
        if time_s > start + duration + 1.0e-12:
            break
        rows.append(observation)
    if len(rows) < 2 or start is None:
        raise OfflineEvaluationError("reset calibration window has fewer than two samples")

    quaternions = [
        _normalize_quaternion(
            _mapping(row.get("base"), "observation.base").get("orientation_wxyz"),
            "base.orientation_wxyz",
        )
        for row in rows
    ]
    anchor = np.asarray(quaternions[0], dtype=float)
    aligned = [
        np.asarray(quaternion, dtype=float)
        * (-1.0 if float(np.dot(anchor, quaternion)) < 0.0 else 1.0)
        for quaternion in quaternions
    ]
    mean = np.mean(np.stack(aligned), axis=0)
    reference = _normalize_quaternion(mean, "mean reset orientation")
    raw_roll, raw_pitch, raw_yaw = _quaternion_to_euler_xyz(reference)

    joint_rows: list[tuple[float, ...]] = []
    wheel_forces: list[tuple[float, ...]] = []
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    for row in rows:
        joints = _mapping(row.get("joints"), "observation.joints")
        joint_rows.append(
            tuple(
                _finite(
                    _mapping(joints.get(name), f"joints.{name}").get("position_deg"),
                    f"joints.{name}.position_deg",
                )
                for name in SERVO_ORDER
            )
        )
        wheel_forces.append(_wheel_normal_forces(row))
        base = _mapping(row.get("base"), "observation.base")
        linear = np.asarray(
            _vector(base.get("linear_velocity_w_m_s"), 3, "base linear velocity")
        )
        angular = np.asarray(
            _vector(base.get("angular_velocity_w_rad_s"), 3, "base angular velocity")
        )
        linear_speeds.append(float(np.linalg.norm(linear)))
        angular_speeds.append(float(np.linalg.norm(angular)))
    home = tuple(float(value) for value in np.mean(np.asarray(joint_rows), axis=0))
    force_baseline = tuple(
        float(value) for value in np.mean(np.asarray(wheel_forces), axis=0)
    )
    max_linear = max(linear_speeds)
    max_angular = max(angular_speeds)
    return LiveRunCalibration(
        level_reference_orientation_wxyz=reference,
        raw_reference_roll_rad=raw_roll,
        raw_reference_pitch_rad=raw_pitch,
        raw_reference_yaw_rad=raw_yaw,
        home_joint_positions_deg8=home,
        wheel_normal_force_baseline_n4=force_baseline,
        window_start_s=start,
        window_end_s=_finite(rows[-1].get("simulation_time_s"), "simulation_time_s"),
        sample_count=len(rows),
        maximum_linear_speed_m_s=max_linear,
        maximum_angular_speed_rad_s=max_angular,
        quality_passed=(
            max_linear <= _finite(maximum_linear_speed_m_s, "maximum_linear_speed_m_s")
            and max_angular
            <= _finite(maximum_angular_speed_rad_s, "maximum_angular_speed_rad_s")
        ),
    )


def _manifest_live_calibration(
    manifest: Mapping[str, Any],
    *,
    maximum_linear_speed_m_s: float,
    maximum_angular_speed_rad_s: float,
) -> LiveRunCalibration | None:
    """Load the pre-action reset calibration captured by the live backend.

    The live PPO writer begins its public stream at controller tick zero, after
    the backend's 1.5 s zero-command settle.  Reusing that explicitly recorded
    reset calibration avoids incorrectly treating the moving P01 prefix as a
    stationary calibration window.  Older canonical recordings do not contain
    this block and continue through :func:`calibrate_live_run` below.
    """

    payload = manifest.get("ppo_calibration")
    if payload is None:
        return None
    row = _mapping(payload, "manifest.ppo_calibration")
    reference = _normalize_quaternion(
        row.get("level_reference_orientation_wxyz"),
        "ppo_calibration.level_reference_orientation_wxyz",
    )
    raw_roll, raw_pitch, raw_yaw = _quaternion_to_euler_xyz(reference)
    home = _vector(
        row.get("home_joint_positions_deg8"),
        len(SERVO_ORDER),
        "ppo_calibration.home_joint_positions_deg8",
    )
    force_baseline = _vector(
        row.get("wheel_normal_force_baseline_n4"),
        len(WHEEL_ORDER),
        "ppo_calibration.wheel_normal_force_baseline_n4",
    )
    sample_count = int(row.get("sample_count", 0))
    if sample_count <= 0:
        raise OfflineEvaluationError("ppo_calibration.sample_count must be positive")
    max_linear = _finite(
        row.get("maximum_linear_speed_m_s"),
        "ppo_calibration.maximum_linear_speed_m_s",
    )
    max_angular = _finite(
        row.get("maximum_angular_speed_rad_s"),
        "ppo_calibration.maximum_angular_speed_rad_s",
    )
    source = str(row.get("source", "")).strip()
    if not source:
        raise OfflineEvaluationError("ppo_calibration.source must be non-empty")
    return LiveRunCalibration(
        level_reference_orientation_wxyz=reference,
        raw_reference_roll_rad=raw_roll,
        raw_reference_pitch_rad=raw_pitch,
        raw_reference_yaw_rad=raw_yaw,
        home_joint_positions_deg8=home,
        wheel_normal_force_baseline_n4=force_baseline,
        window_start_s=_finite(row.get("window_start_s"), "ppo_calibration.window_start_s"),
        window_end_s=_finite(row.get("window_end_s"), "ppo_calibration.window_end_s"),
        sample_count=sample_count,
        maximum_linear_speed_m_s=max_linear,
        maximum_angular_speed_rad_s=max_angular,
        quality_passed=(
            row.get("quality_passed") is True
            and max_linear
            <= _finite(maximum_linear_speed_m_s, "maximum_linear_speed_m_s")
            and max_angular
            <= _finite(maximum_angular_speed_rad_s, "maximum_angular_speed_rad_s")
        ),
        source=source,
    )


@dataclass(frozen=True, slots=True)
class OrientationDiagnostic:
    physics_tick: int
    time_s: float
    phase: str
    lifecycle: str
    raw_roll_rad: float
    raw_pitch_rad: float
    raw_yaw_rad: float
    calibrated_roll_error_rad: float
    calibrated_pitch_error_rad: float
    calibrated_yaw_error_rad: float
    calibrated_roll_rate_rad_s: float
    calibrated_pitch_rate_rad_s: float
    calibrated_yaw_rate_rad_s: float


@dataclass(frozen=True, slots=True)
class ResidualActivityCalibration:
    phase_scale_full12: Mapping[str, Sequence[float]]
    numeric_noise_floor_full12: Sequence[float]
    quantization_floor_full12: Sequence[float]

    def __post_init__(self) -> None:
        if tuple(self.phase_scale_full12) != STATE_IDS:
            raise OfflineEvaluationError(
                "residual phase scales must preserve complete P01-P13 order"
            )
        for phase in STATE_IDS:
            scale = _vector(self.phase_scale_full12[phase], 12, f"{phase} scale")
            if any(value < 0.0 for value in scale):
                raise OfflineEvaluationError("residual scales may not be negative")
        for label, values in (
            ("numeric noise floor", self.numeric_noise_floor_full12),
            ("quantization floor", self.quantization_floor_full12),
        ):
            if any(value < 0.0 for value in _vector(values, 12, label)):
                raise OfflineEvaluationError(f"{label} may not be negative")


@dataclass(frozen=True, slots=True)
class TerminationSummary:
    trial_id: str
    result: str
    reason: str
    final_state_id: str | None
    duration_s: float
    completed_phases: tuple[str, ...]
    completed_p01_p13: bool
    task_success: bool
    body_collision: bool
    wheel_only_climb: bool
    physics_explosion_or_fall: bool
    safety_abort: bool
    runtime_recording_access_count: int
    recovery_count: int
    failed_checks: tuple[str, ...]

    def episode_outcome(self, seed: int) -> EpisodeOutcome:
        return EpisodeOutcome(
            seed=int(seed),
            task_success=self.task_success,
            body_collision=self.body_collision,
            wheel_only_climb=self.wheel_only_climb,
            safety_abort=self.safety_abort,
            duration_s=self.duration_s,
            physics_explosion_or_fall=self.physics_explosion_or_fall,
            completed_p01_p13=self.completed_p01_p13,
        )


@dataclass(frozen=True, slots=True)
class _CommandPoint:
    tick: int
    time_s: float
    phase: str
    lifecycle: str
    nominal_full12: tuple[float, ...]
    residual_full12: tuple[float, ...]
    applied_full12: tuple[float, ...]


def _load_command_points(path: Path) -> tuple[_CommandPoint, ...]:
    points: list[_CommandPoint] = []
    for record in iter_jsonl(path):
        tick = int(record.get("control_physics_tick", -1))
        phase = str(record.get("state_id", ""))
        lifecycle = str(record.get("lifecycle", ""))
        if tick < 0 or phase not in STATE_IDS:
            raise OfflineEvaluationError("command tick/state_id is invalid")
        if lifecycle not in {
            "WAIT_ENTRY",
            "EXECUTE_MOTION",
            "VERIFY_RESULT",
            "RECOVERY",
            "DONE",
        }:
            raise OfflineEvaluationError(f"unknown command lifecycle {lifecycle!r}")
        point = _CommandPoint(
            tick=tick,
            time_s=_finite(record.get("sim_time_s"), "command sim_time_s"),
            phase=phase,
            lifecycle=lifecycle,
            nominal_full12=_vector(record.get("nominal_full12"), 12, "nominal_full12"),
            residual_full12=_vector(record.get("residual_full12"), 12, "residual_full12"),
            applied_full12=_vector(record.get("applied_full12"), 12, "applied_full12"),
        )
        if points and (point.tick <= points[-1].tick or point.time_s <= points[-1].time_s):
            raise OfflineEvaluationError("command ticks/times must be strictly increasing")
        points.append(point)
    if not points:
        raise OfflineEvaluationError("command stream is empty")
    return tuple(points)


def _orientation_diagnostic(
    observation: Mapping[str, Any],
    command: _CommandPoint,
    calibration: LiveRunCalibration,
) -> OrientationDiagnostic:
    base = _mapping(observation.get("base"), "observation.base")
    current = _normalize_quaternion(base.get("orientation_wxyz"), "base orientation")
    raw = _quaternion_to_euler_xyz(current)
    relative = _normalize_quaternion(
        _quat_multiply(
            _quat_conjugate(calibration.level_reference_orientation_wxyz), current
        ),
        "relative orientation",
    )
    calibrated = _quaternion_to_euler_xyz(relative)
    angular_world = np.asarray(
        _vector(base.get("angular_velocity_w_rad_s"), 3, "base angular velocity"),
        dtype=float,
    )
    calibrated_rates = (
        _quaternion_matrix(calibration.level_reference_orientation_wxyz).T
        @ angular_world
    )
    return OrientationDiagnostic(
        physics_tick=command.tick,
        time_s=command.time_s,
        phase=command.phase,
        lifecycle=command.lifecycle,
        raw_roll_rad=raw[0],
        raw_pitch_rad=raw[1],
        raw_yaw_rad=raw[2],
        calibrated_roll_error_rad=calibrated[0],
        calibrated_pitch_error_rad=calibrated[1],
        calibrated_yaw_error_rad=calibrated[2],
        calibrated_roll_rate_rad_s=float(calibrated_rates[0]),
        calibrated_pitch_rate_rad_s=float(calibrated_rates[1]),
        calibrated_yaw_rate_rad_s=float(calibrated_rates[2]),
    )


def _stability_sample(
    observation: Mapping[str, Any],
    command: _CommandPoint,
    calibration: LiveRunCalibration,
    diagnostic: OrientationDiagnostic,
    *,
    wheel_radius_m: float,
) -> StabilitySample:
    wheels = _mapping(observation.get("wheels"), "observation.wheels")
    bodies = _mapping(observation.get("bodies"), "observation.bodies")
    obstacle = _mapping(observation.get("obstacle"), "observation.obstacle")
    joints = _mapping(observation.get("joints"), "observation.joints")
    base = _mapping(observation.get("base"), "observation.base")
    current_orientation = _normalize_quaternion(
        base.get("orientation_wxyz"), "base orientation"
    )
    body_velocity = _rotate_world_to_body(
        current_orientation,
        _vector(base.get("linear_velocity_w_m_s"), 3, "base linear velocity"),
    )
    radius = _finite(wheel_radius_m, "wheel_radius_m")
    if radius <= 0.0:
        raise OfflineEvaluationError("wheel_radius_m must be positive")
    wheel_velocities = tuple(
        _finite(
            _mapping(wheels.get(name), f"wheels.{name}").get("velocity_rad_s"),
            f"wheels.{name}.velocity_rad_s",
        )
        for name in WHEEL_ORDER
    )
    surface_speeds = tuple(radius * value for value in wheel_velocities)
    slips = tuple(
        (surface - body_velocity[0])
        / max(abs(surface), abs(body_velocity[0]), 0.05)
        for surface in surface_speeds
    )
    normal_forces = _wheel_normal_forces(observation)

    active_leg = ACTIVE_LEG_BY_PHASE[command.phase]
    active_force = 0.0
    active_baseline = 0.0
    active_clearance = 0.0
    active_vertical_velocity = 0.0
    if active_leg is not None:
        wheel_name = LEG_TO_WHEEL[active_leg]
        wheel_index = WHEEL_ORDER.index(wheel_name)
        wheel = _mapping(wheels.get(wheel_name), f"wheels.{wheel_name}")
        active_force = normal_forces[wheel_index]
        active_baseline = calibration.wheel_normal_force_baseline_n4[wheel_index]
        if wheel.get("geometry_verified") is True:
            bottom = _vector(wheel.get("bottom_w_m"), 3, f"{wheel_name}.bottom_w_m")
            active_clearance = bottom[2] - _finite(
                obstacle.get("top_z_m"), "obstacle.top_z_m"
            )
        body_name = str(wheel.get("body_name", ""))
        wheel_body = _mapping(bodies.get(body_name), f"bodies.{body_name}")
        active_vertical_velocity = _vector(
            wheel_body.get("linear_velocity_w_m_s"),
            3,
            f"bodies.{body_name}.linear_velocity_w_m_s",
        )[2]

    actual_joints = np.asarray(
        [
            _finite(
                _mapping(joints.get(name), f"joints.{name}").get("position_deg"),
                f"joints.{name}.position_deg",
            )
            for name in SERVO_ORDER
        ],
        dtype=float,
    )
    home_error = float(
        np.sqrt(
            np.mean(
                np.square(
                    actual_joints
                    - np.asarray(calibration.home_joint_positions_deg8, dtype=float)
                )
            )
        )
    )
    return StabilitySample(
        time_s=command.time_s,
        phase=command.phase,
        lifecycle=command.lifecycle,
        roll_error_rad=diagnostic.calibrated_roll_error_rad,
        pitch_error_rad=diagnostic.calibrated_pitch_error_rad,
        roll_rate_rad_s=diagnostic.calibrated_roll_rate_rad_s,
        pitch_rate_rad_s=diagnostic.calibrated_pitch_rate_rad_s,
        yaw_rate_rad_s=diagnostic.calibrated_yaw_rate_rad_s,
        active_contact_normal_force_n=active_force,
        active_contact_baseline_n=active_baseline,
        wheel_slip4=slips,
        active_leg_clearance_m=active_clearance,
        active_leg_vertical_velocity_m_s=active_vertical_velocity,
        home_pose_error_rms_deg=home_error,
        residual_full12=command.residual_full12,
        nominal_full12=command.nominal_full12,
        applied_full12=command.applied_full12,
    )


def _transition_action_metrics(
    commands: Sequence[_CommandPoint],
) -> Mapping[str, Mapping[str, float]]:
    result: dict[str, Mapping[str, float]] = {}
    first_index: dict[str, int] = {}
    for index, command in enumerate(commands):
        first_index.setdefault(command.phase, index)
    for phase in STATE_IDS:
        index = first_index.get(phase)
        if index is None or index == 0:
            difference = np.zeros(12, dtype=float)
        else:
            difference = np.asarray(commands[index].applied_full12) - np.asarray(
                commands[index - 1].applied_full12
            )
        result[phase] = {
            "phase_entry_action_jump_rms": float(
                np.sqrt(np.mean(np.square(difference)))
            ),
            "phase_entry_servo_jump_peak_abs": float(
                np.max(np.abs(difference[:8]))
            ),
            "phase_entry_wheel_jump_peak_abs": float(
                np.max(np.abs(difference[8:]))
            ),
        }
    return result


def _wheel_stop_metric(
    commands: Sequence[_CommandPoint],
    wheel_speeds_by_tick: Mapping[int, tuple[float, ...]],
    *,
    threshold_rad_s: float,
    hold_s: float,
    physics_hz: float,
) -> tuple[float | None, bool]:
    p13 = [command for command in commands if command.phase == "P13"]
    if not p13:
        return None, False
    nonzero = [
        index
        for index, command in enumerate(p13)
        if any(abs(value) > 1.0e-12 for value in command.applied_full12[8:])
    ]
    zero_index = (nonzero[-1] + 1) if nonzero else 0
    if zero_index >= len(p13):
        return None, False
    zero_time = p13[zero_index].time_s
    hold_count = max(1, int(math.ceil(hold_s * physics_hz)))
    stable = [
        max(abs(value) for value in wheel_speeds_by_tick[command.tick])
        <= threshold_rad_s
        for command in p13[zero_index:]
        if command.tick in wheel_speeds_by_tick
    ]
    times = [
        command.time_s
        for command in p13[zero_index:]
        if command.tick in wheel_speeds_by_tick
    ]
    for index in range(0, max(0, len(stable) - hold_count + 1)):
        if all(stable[index : index + hold_count]):
            completion = times[index + hold_count - 1]
            return max(0.0, completion - zero_time), True
    return None, False


def _termination_summary(
    *,
    manifest: Mapping[str, Any],
    task_events: Sequence[Mapping[str, Any]],
    observation_hazards: Mapping[str, bool],
) -> TerminationSummary:
    termination_events = [
        event
        for event in task_events
        if str(event.get("event", "")).startswith("TRIAL_TERMINATION")
    ]
    event = termination_events[-1] if termination_events else {}
    result = str(event.get("result", manifest.get("result", "UNKNOWN")))
    manifest_result = str(manifest.get("result", result))
    if event and manifest_result != result:
        raise OfflineEvaluationError(
            f"task-event result {result!r} disagrees with manifest {manifest_result!r}"
        )
    evidence = _mapping(manifest.get("success_evidence", {}), "success_evidence")
    completed = tuple(str(value) for value in evidence.get("completed_macro_phases", ()))
    if not completed:
        completed = tuple(
            str(row.get("phase"))
            for row in manifest.get("phase_windows", ())
            if isinstance(row, Mapping)
        )
    completed_p01_p13 = completed == STATE_IDS
    analysis = _mapping(manifest.get("analysis_checks", {}), "analysis_checks")
    failed_checks = list(
        str(value)
        for value in _mapping(event.get("details", {}), "termination details").get(
            "failed_checks", ()
        )
    )
    if not failed_checks:
        failed_checks = [
            str(name) for name, passed in analysis.items() if passed is False
        ]
    body_collision = bool(evidence.get("body_collision", False)) or bool(
        observation_hazards.get("body_collision")
    )
    wheel_only = bool(evidence.get("wheel_only_climb", False)) or bool(
        observation_hazards.get("wheel_only_climb")
    )
    fall = bool(observation_hazards.get("physics_explosion_or_fall"))
    safety_abort = result == "SAFETY_ABORT" or "SAFETY_ABORT" in result
    duration = _finite(
        event.get(
            "sim_time_s",
            evidence.get("duration_s", manifest.get("duration_s", 0.0)),
        ),
        "termination duration",
    )
    runtime_access = evidence.get("recording_runtime_access_count")
    if runtime_access is None:
        runtime_access = int(bool(evidence.get("runtime_raw_recording_access", False)))
    recovery_count = int(
        _mapping(manifest.get("conformance", {}), "conformance").get(
            "recovery_count", 0
        )
    )
    task_success = (
        result == "SUCCESS"
        and completed_p01_p13
        and not body_collision
        and not wheel_only
        and not fall
        and not safety_abort
    )
    return TerminationSummary(
        trial_id=str(manifest.get("trial_id", "")),
        result=result,
        reason=str(event.get("reason", manifest.get("reason", ""))),
        final_state_id=(str(event.get("state_id")) if event.get("state_id") else None),
        duration_s=duration,
        completed_phases=completed,
        completed_p01_p13=completed_p01_p13,
        task_success=task_success,
        body_collision=body_collision,
        wheel_only_climb=wheel_only,
        physics_explosion_or_fall=fall,
        safety_abort=safety_abort,
        runtime_recording_access_count=int(runtime_access),
        recovery_count=recovery_count,
        failed_checks=tuple(failed_checks),
    )


def summarize_reward_contributions(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, float | int | str], ...]:
    """Aggregate logged v2 reward breakdowns without inventing missing values."""

    accumulators: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for record in records:
        phase = str(record.get("state_id", record.get("phase_id", "")))
        if phase not in STATE_IDS:
            raise OfflineEvaluationError(f"reward record has invalid phase {phase!r}")
        if isinstance(record.get("weighted_dense"), Mapping):
            dense = _mapping(record["weighted_dense"], "weighted_dense")
            events = _mapping(record.get("event_components", {}), "event_components")
        else:
            components = _mapping(
                record.get("reward_components_t"), "reward_components_t"
            )
            dense = components
            events = components
        missing = [name for name in DENSE_FAMILIES if name not in dense]
        if missing:
            raise OfflineEvaluationError(
                f"reward record omits v2 dense families {missing}; legacy rewards are not relabeled"
            )
        values = {
            name: _finite(dense[name], f"reward.{name}") for name in DENSE_FAMILIES
        }
        event_total = sum(
            _finite(events.get(name, 0.0), f"reward.{name}")
            for name in EVENT_FAMILIES
        )
        bucket = accumulators.setdefault(
            phase, {name: 0.0 for name in DENSE_FAMILIES}
        )
        for name, value in values.items():
            bucket[name] += value
        bucket["event_reward"] = bucket.get("event_reward", 0.0) + event_total
        counts[phase] = counts.get(phase, 0) + 1
    rows: list[dict[str, float | int | str]] = []
    for phase in STATE_IDS:
        if phase not in accumulators:
            continue
        count = counts[phase]
        bucket = accumulators[phase]
        row: dict[str, float | int | str] = {"phase": phase, "decision_count": count}
        for name in DENSE_FAMILIES:
            row[f"{name}_sum"] = bucket[name]
            row[f"{name}_mean"] = bucket[name] / count
        row["event_reward_sum"] = bucket["event_reward"]
        row["total_reward_sum"] = sum(bucket[name] for name in DENSE_FAMILIES) + bucket[
            "event_reward"
        ]
        rows.append(row)
    if not rows:
        raise OfflineEvaluationError("reward stream is empty")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class LiveRunEvaluation:
    run_directory: Path
    seed: int
    calibration: LiveRunCalibration
    stability_samples: tuple[StabilitySample, ...]
    orientation_diagnostics: tuple[OrientationDiagnostic, ...]
    phase_rows: tuple[Mapping[str, Any], ...]
    episode_row: Mapping[str, Any]
    termination: TerminationSummary
    residual_activity_rows: tuple[Mapping[str, Any], ...]
    residual_activity_evaluated: bool
    reward_contribution_rows: tuple[Mapping[str, Any], ...]
    reward_contributions_available: bool


def evaluate_live_run(
    run_directory: str | Path,
    *,
    seed: int,
    residual_calibration: ResidualActivityCalibration | None = None,
    reward_stream_path: str | Path | None = None,
    calibration_window_s: float = 0.5,
    calibration_max_linear_speed_m_s: float = 0.10,
    calibration_max_angular_speed_rad_s: float = 0.20,
    wheel_radius_m: float = WHEEL_RADIUS_M,
    wheel_stop_hold_s: float = 0.5,
) -> LiveRunEvaluation:
    """Evaluate one existing live run entirely in memory and without writes."""

    run = Path(run_directory).resolve()
    required = (OBSERVATION_FILENAME, COMMAND_FILENAME, MANIFEST_FILENAME)
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise OfflineEvaluationError(f"run directory is missing {missing}")
    manifest = _load_json_object(run / MANIFEST_FILENAME)
    physics_hz = _finite(manifest.get("physics_hz"), "manifest.physics_hz")
    if physics_hz <= 0.0:
        raise OfflineEvaluationError("manifest.physics_hz must be positive")
    commands = _load_command_points(run / COMMAND_FILENAME)
    command_by_tick = {command.tick: command for command in commands}
    if len(command_by_tick) != len(commands):
        raise OfflineEvaluationError("command stream contains duplicate ticks")

    calibration = _manifest_live_calibration(
        manifest,
        maximum_linear_speed_m_s=calibration_max_linear_speed_m_s,
        maximum_angular_speed_rad_s=calibration_max_angular_speed_rad_s,
    )
    if calibration is None:
        calibration = calibrate_live_run(
            iter_jsonl(run / OBSERVATION_FILENAME),
            stable_window_s=calibration_window_s,
            maximum_linear_speed_m_s=calibration_max_linear_speed_m_s,
            maximum_angular_speed_rad_s=calibration_max_angular_speed_rad_s,
        )
    samples: list[StabilitySample] = []
    diagnostics: list[OrientationDiagnostic] = []
    wheel_speeds_by_tick: dict[int, tuple[float, ...]] = {}
    hazards = {
        "body_collision": False,
        "wheel_only_climb": False,
        "physics_explosion_or_fall": False,
    }
    observation_count = 0
    previous_tick = -1
    for observation in iter_jsonl(run / OBSERVATION_FILENAME):
        observation_count += 1
        if observation.get("schema") != OBSERVATION_STREAM_SCHEMA:
            raise OfflineEvaluationError("unexpected live observation schema")
        tick = int(observation.get("physics_tick", -1))
        if tick <= previous_tick:
            raise OfflineEvaluationError("observation ticks must be strictly increasing")
        previous_tick = tick
        if observation.get("all_finite") is not True:
            raise OfflineEvaluationError(f"observation tick {tick} is marked non-finite")
        collision = _mapping(observation.get("body_collision"), "body_collision")
        hazards["body_collision"] |= bool(collision.get("detected", False))
        guards = _mapping(observation.get("guards", {}), "guards")
        for key, hazard_name in (
            ("wheel_only_climb_detected", "wheel_only_climb"),
            ("physics_explosion_or_fall", "physics_explosion_or_fall"),
        ):
            guard = guards.get(key)
            if isinstance(guard, Mapping):
                hazards[hazard_name] |= bool(guard.get("passed", False))
        command = command_by_tick.get(tick)
        if command is None:
            continue
        observation_time = _finite(
            observation.get("simulation_time_s"), "observation time"
        )
        if not math.isclose(
            observation_time,
            command.time_s,
            rel_tol=0.0,
            abs_tol=max(1.0e-9, 0.25 / physics_hz),
        ):
            raise OfflineEvaluationError(
                f"observation/command time mismatch at physics tick {tick}"
            )
        diagnostic = _orientation_diagnostic(observation, command, calibration)
        sample = _stability_sample(
            observation,
            command,
            calibration,
            diagnostic,
            wheel_radius_m=wheel_radius_m,
        )
        samples.append(sample)
        diagnostics.append(diagnostic)
        wheel_speeds_by_tick[tick] = _vector(
            observation.get("measured_wheel_velocity_rad_s"),
            4,
            "measured_wheel_velocity_rad_s",
        )
    missing_command_observations = sorted(set(command_by_tick) - set(wheel_speeds_by_tick))
    if missing_command_observations:
        raise OfflineEvaluationError(
            f"{len(missing_command_observations)} command ticks lack observations"
        )

    transition_path = run / TRANSITION_FILENAME
    transition_records = (
        tuple(iter_jsonl(transition_path)) if transition_path.is_file() else ()
    )
    task_event_path = run / TASK_EVENT_FILENAME
    task_events = tuple(iter_jsonl(task_event_path)) if task_event_path.is_file() else ()
    termination = _termination_summary(
        manifest=manifest,
        task_events=task_events,
        observation_hazards=hazards,
    )

    phase_rows = summarize_phase_samples(samples, physics_hz=physics_hz)
    transition_action = _transition_action_metrics(commands)
    phase_times = _mapping(manifest.get("phase_times", {}), "phase_times")
    completed_set = set(termination.completed_phases)
    transition_counts = {
        phase: sum(str(record.get("state_id")) == phase for record in transition_records)
        for phase in STATE_IDS
    }
    recovery_counts = {
        phase: sum(
            str(record.get("state_id")) == phase
            and "RECOVERY"
            in {
                str(record.get("from_lifecycle")),
                str(record.get("to_lifecycle")),
            }
            for record in transition_records
        )
        for phase in STATE_IDS
    }
    threshold = _mapping(manifest.get("conformance", {}), "conformance").get(
        "measured_wheel_velocity_decay_threshold_rad_s"
    )
    if threshold is None:
        wheel_stop_time, wheel_stop_completed = None, False
    else:
        wheel_stop_time, wheel_stop_completed = _wheel_stop_metric(
            commands,
            wheel_speeds_by_tick,
            threshold_rad_s=_finite(threshold, "wheel stop threshold"),
            hold_s=_finite(wheel_stop_hold_s, "wheel_stop_hold_s"),
            physics_hz=physics_hz,
        )
    enriched: list[Mapping[str, Any]] = []
    for row in phase_rows:
        phase = str(row["phase"])
        timing = _mapping(phase_times.get(phase, {}), f"phase_times.{phase}")
        entry = timing.get("entry_time_s")
        completion = timing.get("completion_time_s")
        motion_start = timing.get("motion_start_s")
        motion_end = timing.get("motion_end_s")
        extra = {
            "entry_time_s": (_finite(entry, f"{phase}.entry") if entry is not None else None),
            "completion_time_s": (
                _finite(completion, f"{phase}.completion")
                if completion is not None
                else None
            ),
            "phase_duration_s": (
                _finite(completion, f"{phase}.completion")
                - _finite(entry, f"{phase}.entry")
                if entry is not None and completion is not None
                else float(row["duration_s"])
            ),
            "motion_duration_s": (
                _finite(motion_end, f"{phase}.motion_end")
                - _finite(motion_start, f"{phase}.motion_start")
                if motion_start is not None and motion_end is not None
                else None
            ),
            "lifecycle_transition_count": transition_counts[phase],
            "recovery_transition_count": recovery_counts[phase],
            "phase_completion_observed": phase in completed_set,
            "wheel_stop_time_s": wheel_stop_time if phase == "P13" else None,
            "wheel_stop_completed": wheel_stop_completed if phase == "P13" else None,
            **transition_action[phase],
        }
        enriched.append({**row, **extra})

    if residual_calibration is None:
        residual_rows: tuple[Mapping[str, Any], ...] = ()
        residual_evaluated = False
    else:
        residual_rows = tuple(
            residual_activity_by_phase(
                samples,
                phase_scale_full12=residual_calibration.phase_scale_full12,
                numeric_noise_floor_full12=residual_calibration.numeric_noise_floor_full12,
                quantization_floor_full12=residual_calibration.quantization_floor_full12,
                dt_s=1.0 / physics_hz,
            )
        )
        residual_evaluated = True

    if reward_stream_path is None:
        reward_rows: tuple[Mapping[str, Any], ...] = ()
        rewards_available = False
    else:
        selected_reward_path = Path(reward_stream_path).resolve()
        reward_rows = summarize_reward_contributions(iter_jsonl(selected_reward_path))
        rewards_available = True

    phase_index = {str(row["phase"]): row for row in enriched}
    overall_pitch_rate_rms = float(
        np.sqrt(np.mean(np.square([sample.pitch_rate_rad_s for sample in samples])))
    )
    overall_roll_rate_rms = float(
        np.sqrt(np.mean(np.square([sample.roll_rate_rad_s for sample in samples])))
    )
    placement_impulse = sum(
        float(phase_index[phase]["placement_contact_impulse_n_s"])
        for phase in ("P03", "P12")
        if phase in phase_index
    )
    p13 = phase_index.get("P13", {})
    residual_nonzero_count = (
        sum(bool(row["nonzero"]) for row in residual_rows) if residual_evaluated else None
    )
    reward_total = (
        sum(float(row["total_reward_sum"]) for row in reward_rows)
        if rewards_available
        else None
    )
    episode_row: Mapping[str, Any] = {
        "trial_id": termination.trial_id,
        "seed": int(seed),
        "task_result": termination.result,
        "task_success": termination.task_success,
        "completed_p01_p13": termination.completed_p01_p13,
        "body_collision": termination.body_collision,
        "wheel_only_climb": termination.wheel_only_climb,
        "physics_explosion_or_fall": termination.physics_explosion_or_fall,
        "safety_abort": termination.safety_abort,
        "duration_s": termination.duration_s,
        "overall_pitch_rate_rms_rad_s": overall_pitch_rate_rms,
        "overall_roll_rate_rms_rad_s": overall_roll_rate_rms,
        "placement_contact_impulse_n_s": placement_impulse,
        "home_recovery_action_jerk_rms": p13.get("action_jerk_rms"),
        "final_home_pose_error_rms_deg": p13.get("home_pose_error_rms_deg"),
        "wheel_stop_time_s": wheel_stop_time,
        "wheel_stop_completed": wheel_stop_completed,
        "residual_activity_evaluated": residual_evaluated,
        "residual_nonzero_phase_count": residual_nonzero_count,
        "reward_contributions_available": rewards_available,
        "total_reward": reward_total,
        "calibration_quality_passed": calibration.quality_passed,
        "observation_sample_count": observation_count,
        "paired_stability_sample_count": len(samples),
        "runtime_recording_access_count": termination.runtime_recording_access_count,
    }
    return LiveRunEvaluation(
        run_directory=run,
        seed=int(seed),
        calibration=calibration,
        stability_samples=tuple(samples),
        orientation_diagnostics=tuple(diagnostics),
        phase_rows=tuple(enriched),
        episode_row=episode_row,
        termination=termination,
        residual_activity_rows=residual_rows,
        residual_activity_evaluated=residual_evaluated,
        reward_contribution_rows=reward_rows,
        reward_contributions_available=rewards_available,
    )


def _aggregate_phase_rows(
    runs: Sequence[LiveRunEvaluation],
) -> tuple[dict[str, float | str], ...]:
    indexed = [
        {str(row["phase"]): row for row in run.phase_rows}
        for run in runs
    ]
    if any(set(rows) != set(PHASE_IDS) for rows in indexed):
        raise OfflineEvaluationError("paired promotion requires P01-P13 metrics in every run")
    result: list[dict[str, float | str]] = []
    for phase in PHASE_IDS:
        row: dict[str, float | str] = {"phase": phase}
        common = set.intersection(*(set(rows[phase]) for rows in indexed))
        for name in sorted(common):
            if name == "phase":
                continue
            values = [rows[phase][name] for rows in indexed]
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in values
            ):
                row[name] = float(np.mean([float(value) for value in values]))
        result.append(row)
    return tuple(result)


def _improvement(baseline: float, candidate: float) -> float:
    baseline_value = _finite(baseline, "baseline metric")
    candidate_value = _finite(candidate, "candidate metric")
    if abs(baseline_value) <= 1.0e-12:
        return 0.0 if abs(candidate_value) <= 1.0e-12 else -1.0
    return (baseline_value - candidate_value) / abs(baseline_value)


@dataclass(frozen=True, slots=True)
class PairedRunComparison:
    baseline_phase_rows: tuple[Mapping[str, Any], ...]
    candidate_phase_rows: tuple[Mapping[str, Any], ...]
    phase_comparison_rows: tuple[Mapping[str, Any], ...]
    baseline_episode_rows: tuple[Mapping[str, Any], ...]
    candidate_episode_rows: tuple[Mapping[str, Any], ...]
    promotion: PromotionDecision
    overall_pitch_rate_improvement_fraction: float
    placement_impulse_improvement_fraction: float
    home_jerk_improvement_fraction: float


def paired_baseline_candidate_promotion(
    baseline_runs: Sequence[LiveRunEvaluation],
    candidate_runs: Sequence[LiveRunEvaluation],
    *,
    frozen_hashes_unchanged: bool,
    minimum_paired_seeds: int = 5,
) -> PairedRunComparison:
    """Compare matched seeds and return a decision; never rename/publish a candidate."""

    if len(baseline_runs) < minimum_paired_seeds or len(candidate_runs) != len(
        baseline_runs
    ):
        raise OfflineEvaluationError(
            f"promotion requires at least {minimum_paired_seeds} equal paired runs"
        )
    baseline_seeds = [run.seed for run in baseline_runs]
    candidate_seeds = [run.seed for run in candidate_runs]
    if baseline_seeds != candidate_seeds or len(set(baseline_seeds)) != len(
        baseline_seeds
    ):
        raise OfflineEvaluationError("baseline/candidate seeds are not unique and paired")
    baseline_phase = _aggregate_phase_rows(baseline_runs)
    candidate_phase = _aggregate_phase_rows(candidate_runs)
    phase_comparison = tuple(compare_phase_metrics(baseline_phase, candidate_phase))

    def episode_mean(runs: Sequence[LiveRunEvaluation], name: str) -> float:
        return float(np.mean([_finite(run.episode_row[name], name) for run in runs]))

    pitch_improvement = _improvement(
        episode_mean(baseline_runs, "overall_pitch_rate_rms_rad_s"),
        episode_mean(candidate_runs, "overall_pitch_rate_rms_rad_s"),
    )
    impulse_improvement = _improvement(
        episode_mean(baseline_runs, "placement_contact_impulse_n_s"),
        episode_mean(candidate_runs, "placement_contact_impulse_n_s"),
    )
    home_improvement = _improvement(
        episode_mean(baseline_runs, "home_recovery_action_jerk_rms"),
        episode_mean(candidate_runs, "home_recovery_action_jerk_rms"),
    )
    base_decision = evaluate_promotion(
        baseline_episodes=[
            run.termination.episode_outcome(run.seed) for run in baseline_runs
        ],
        candidate_episodes=[
            run.termination.episode_outcome(run.seed) for run in candidate_runs
        ],
        phase_comparison_rows=phase_comparison,
        overall_pitch_rate_improvement=pitch_improvement,
        placement_impulse_improvement=impulse_improvement,
        home_jerk_improvement=home_improvement,
        frozen_hashes_unchanged=frozen_hashes_unchanged,
        recording_runtime_access_count=sum(
            run.termination.runtime_recording_access_count
            for run in candidate_runs
        ),
    )
    checks = dict(base_decision.checks)
    checks["level_calibration_quality_passed"] = all(
        run.calibration.quality_passed
        for run in tuple(baseline_runs) + tuple(candidate_runs)
    )
    residual_available = all(
        run.residual_activity_evaluated for run in candidate_runs
    )
    checks["residual_activity_calibrated"] = residual_available
    if residual_available:
        residual_by_run = [
            {str(row["phase"]): bool(row["nonzero"]) for row in run.residual_activity_rows}
            for run in candidate_runs
        ]
        checks["priority_phases_have_real_residual"] = all(
            all(rows.get(phase, False) for phase in PRIORITY_PHASES)
            for rows in residual_by_run
        )
        checks["at_least_10_phases_have_real_residual"] = all(
            sum(rows.get(phase, False) for phase in PHASE_IDS) >= 10
            for rows in residual_by_run
        )
    else:
        checks["priority_phases_have_real_residual"] = False
        checks["at_least_10_phases_have_real_residual"] = False
    first_failed = next((name for name, passed in checks.items() if not passed), None)
    decision = replace(
        base_decision,
        promoted=first_failed is None,
        first_failed_gate=first_failed,
        checks=checks,
    )
    return PairedRunComparison(
        baseline_phase_rows=baseline_phase,
        candidate_phase_rows=candidate_phase,
        phase_comparison_rows=phase_comparison,
        baseline_episode_rows=tuple(run.episode_row for run in baseline_runs),
        candidate_episode_rows=tuple(run.episode_row for run in candidate_runs),
        promotion=decision,
        overall_pitch_rate_improvement_fraction=pitch_improvement,
        placement_impulse_improvement_fraction=impulse_improvement,
        home_jerk_improvement_fraction=home_improvement,
    )


__all__ = [
    "COMMAND_FILENAME",
    "MANIFEST_FILENAME",
    "OBSERVATION_FILENAME",
    "OBSERVATION_STREAM_SCHEMA",
    "TASK_EVENT_FILENAME",
    "TRANSITION_FILENAME",
    "LiveRunCalibration",
    "LiveRunEvaluation",
    "OfflineEvaluationError",
    "OrientationDiagnostic",
    "PairedRunComparison",
    "ResidualActivityCalibration",
    "TerminationSummary",
    "calibrate_live_run",
    "evaluate_live_run",
    "iter_jsonl",
    "paired_baseline_candidate_promotion",
    "summarize_reward_contributions",
]
