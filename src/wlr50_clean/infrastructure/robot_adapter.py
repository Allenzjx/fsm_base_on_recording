"""Narrow articulation adapter for one atomic Full12 command per physics tick.

Joint resolution and target conversion are derived from mature Recording
``sim_robot_adapter.py`` (SHA-256
``04966b8100eeb33ea55de78eaa5a74bbc3662030f82ba4d458eb429608ed64d4``).
This clean adapter has no root-state APIs, recovery/respawn paths, playback,
recording, force application, or simulation stepping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .command_batch import (
    FULL12_ORDER,
    PHYSICS_DT_S,
    SERVO_ORDER,
    WHEEL_FORWARD_SIGN,
    WHEEL_ORDER,
    CommandBatchError,
    Full12Command,
    JointIndexMap,
    build_physical_batch,
    logical_readback_from_physical,
    resolve_joint_indices,
)


class RobotAdapterError(RuntimeError):
    """Raised before an invalid or duplicate-tick command can be written."""


@dataclass(frozen=True, slots=True)
class JointStateSnapshot:
    """Canonical readback plus raw physical joint values."""

    full12: tuple[float, ...]
    servo_position_rad: tuple[float, ...]
    servo_velocity_rad_s: tuple[float, ...]
    wheel_velocity_physical_rad_s: tuple[float, ...]

    def by_name(self) -> dict[str, float]:
        return dict(zip(FULL12_ORDER, self.full12, strict=True))


class RobotAdapter:
    """Apply complete, validated commands to one Isaac Lab articulation.

    The runtime owns the 120 Hz loop and calls ``apply_full12`` once before
    each physics step. Both target tensors are fully built before either target
    setter runs; one and only one ``write_data_to_sim`` follows the two setters.
    """

    def __init__(self, robot: Any, *, physics_dt_s: float = PHYSICS_DT_S):
        self.robot = robot
        self.physics_dt_s = float(physics_dt_s)
        if not math.isclose(self.physics_dt_s, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise RobotAdapterError(
                f"adapter requires locked 120 Hz dt {PHYSICS_DT_S}; received {self.physics_dt_s}"
            )
        try:
            live_names = tuple(str(name) for name in robot.joint_names)
        except Exception as exc:
            raise RobotAdapterError("robot.joint_names is unavailable") from exc
        try:
            self.joint_map: JointIndexMap = resolve_joint_indices(live_names)
        except CommandBatchError as exc:
            raise RobotAdapterError(str(exc)) from exc

        joint_pos = _joint_matrix(robot, "joint_pos")
        if int(joint_pos.shape[0]) != 1:
            raise RobotAdapterError(
                f"locked scene requires exactly one robot instance; received {int(joint_pos.shape[0])}"
            )
        self._standing_servo_tensor = _clone_tensor(joint_pos[:, list(self.joint_map.servo_ids)])
        standing_first_row = _row_values(self._standing_servo_tensor)
        if len(standing_first_row) != len(SERVO_ORDER):
            raise RobotAdapterError("could not capture all eight standing servo positions")
        self.standing_pose_deg: dict[str, float] = {
            name: math.degrees(value)
            for name, value in zip(SERVO_ORDER, standing_first_row, strict=True)
        }
        self.write_count = 0
        self._last_physics_tick: int | None = None
        self.last_ack: dict[str, Any] | None = None

    @classmethod
    def from_scene(cls, scene_handle: Any) -> "RobotAdapter":
        return cls(scene_handle.robot, physics_dt_s=float(scene_handle.sim.get_physics_dt()))

    def apply_full12(
        self,
        command: Full12Command | Sequence[float] | Mapping[str, float],
        *,
        physics_tick: int | None = None,
    ) -> dict[str, Any]:
        """Stage all 12 targets and issue exactly one articulation write.

        ``physics_tick`` is optional for simple callers. When supplied, it must
        be strictly increasing, which catches accidental double writes in a
        runtime tick. Mapping input must contain exactly all 12 canonical keys.
        """

        requested = self._coerce_command(command)
        tick = self._validate_tick(physics_tick)
        physical = build_physical_batch(requested, self.standing_pose_deg)

        # Finish both tensors before mutating either articulation target buffer.
        position_targets = _clone_tensor(self._standing_servo_tensor)
        for local_index, value in enumerate(physical.servo_target_rad):
            position_targets[:, local_index] = float(value)
        joint_vel = _joint_matrix(self.robot, "joint_vel")
        velocity_targets = _clone_tensor(joint_vel[:, list(self.joint_map.wheel_ids)])
        for local_index, value in enumerate(physical.wheel_target_rad_s):
            velocity_targets[:, local_index] = float(value)

        self.robot.set_joint_position_target(position_targets, joint_ids=list(self.joint_map.servo_ids))
        self.robot.set_joint_velocity_target(velocity_targets, joint_ids=list(self.joint_map.wheel_ids))
        self.robot.write_data_to_sim()

        self.write_count += 1
        self._last_physics_tick = tick
        applied = physical.applied_logical
        ack: dict[str, Any] = {
            "schema": "wlr50_clean.atomic_full12_ack.v1",
            "physics_tick": tick,
            "physics_dt_s": self.physics_dt_s,
            "write_count": self.write_count,
            "articulation_writes_this_call": 1,
            "canonical_order": list(FULL12_ORDER),
            "requested_full12": list(requested.to_full12()),
            "applied_full12": list(applied.to_full12()),
            "command_was_clamped": requested != applied,
            "servo_joint_ids": list(self.joint_map.servo_ids),
            "wheel_joint_ids": list(self.joint_map.wheel_ids),
            "servo_target_physical_rad": list(physical.servo_target_rad),
            "wheel_target_physical_rad_s": list(physical.wheel_target_rad_s),
            "motion_start_skew_s": 0.0,
        }
        self.last_ack = ack
        return dict(ack)

    def get_actual_full12(self) -> tuple[float, ...]:
        """Return live q/qd in canonical recording-space units and order."""

        position = _row_values(_joint_matrix(self.robot, "joint_pos")[:, list(self.joint_map.servo_ids)])
        wheel_velocity = _row_values(_joint_matrix(self.robot, "joint_vel")[:, list(self.joint_map.wheel_ids)])
        try:
            return logical_readback_from_physical(position, wheel_velocity, self.standing_pose_deg)
        except CommandBatchError as exc:
            raise RobotAdapterError(f"invalid live joint readback: {exc}") from exc

    def get_actual_state(self) -> JointStateSnapshot:
        servo_position = _row_values(
            _joint_matrix(self.robot, "joint_pos")[:, list(self.joint_map.servo_ids)]
        )
        servo_velocity = _row_values(
            _joint_matrix(self.robot, "joint_vel")[:, list(self.joint_map.servo_ids)]
        )
        wheel_velocity = _row_values(
            _joint_matrix(self.robot, "joint_vel")[:, list(self.joint_map.wheel_ids)]
        )
        try:
            full12 = logical_readback_from_physical(
                servo_position,
                wheel_velocity,
                self.standing_pose_deg,
            )
        except CommandBatchError as exc:
            raise RobotAdapterError(f"invalid live joint readback: {exc}") from exc
        return JointStateSnapshot(
            full12=full12,
            servo_position_rad=servo_position,
            servo_velocity_rad_s=servo_velocity,
            wheel_velocity_physical_rad_s=wheel_velocity,
        )

    def update_readback(self) -> None:
        """Synchronize articulation buffers after the runtime advances physics."""

        self.robot.update(self.physics_dt_s)

    def joint_ids_by_name(self) -> dict[str, int]:
        return dict(zip(FULL12_ORDER, self.joint_map.full12_ids, strict=True))

    def wheel_physical_signs(self) -> dict[str, float]:
        return {name: float(WHEEL_FORWARD_SIGN[name]) for name in WHEEL_ORDER}

    def _coerce_command(
        self,
        command: Full12Command | Sequence[float] | Mapping[str, float],
    ) -> Full12Command:
        try:
            if isinstance(command, Full12Command):
                return command
            if isinstance(command, Mapping):
                actual = set(command)
                required = set(FULL12_ORDER)
                if actual != required:
                    missing = sorted(required - actual)
                    extra = sorted(actual - required)
                    raise CommandBatchError(
                        f"full12 mapping keys must be exact; missing={missing}, extra={extra}"
                    )
                return Full12Command.from_full12(tuple(float(command[name]) for name in FULL12_ORDER))
            return Full12Command.from_full12(command)
        except CommandBatchError as exc:
            raise RobotAdapterError(str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise RobotAdapterError("command must be Full12Command, a 12-value sequence, or exact mapping") from exc

    def _validate_tick(self, physics_tick: int | None) -> int:
        if physics_tick is None:
            return 0 if self._last_physics_tick is None else self._last_physics_tick + 1
        try:
            tick = int(physics_tick)
        except (TypeError, ValueError) as exc:
            raise RobotAdapterError("physics_tick must be an integer") from exc
        if tick < 0:
            raise RobotAdapterError("physics_tick cannot be negative")
        if self._last_physics_tick is not None and tick <= self._last_physics_tick:
            raise RobotAdapterError(
                f"physics_tick must increase strictly; last={self._last_physics_tick}, received={tick}"
            )
        return tick


def _joint_matrix(robot: Any, field: str) -> Any:
    try:
        value = getattr(robot.data, field)
        shape = tuple(value.shape)
    except Exception as exc:
        raise RobotAdapterError(f"robot.data.{field} is unavailable") from exc
    if len(shape) != 2 or shape[0] < 1:
        raise RobotAdapterError(f"robot.data.{field} must have shape (instances, joints); received {shape}")
    return value


def _clone_tensor(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    raise RobotAdapterError("joint tensor does not support clone/copy")


def _row_values(value: Any) -> tuple[float, ...]:
    try:
        row = value[0]
        detach = getattr(row, "detach", None)
        if callable(detach):
            row = detach()
        cpu = getattr(row, "cpu", None)
        if callable(cpu):
            row = cpu()
        reshape = getattr(row, "reshape", None)
        if callable(reshape):
            row = reshape(-1)
        tolist = getattr(row, "tolist", None)
        raw = tolist() if callable(tolist) else list(row)
        result = tuple(float(item) for item in raw)
    except Exception as exc:
        raise RobotAdapterError("failed to read joint tensor row") from exc
    if any(not math.isfinite(item) for item in result):
        raise RobotAdapterError("joint tensor row contains non-finite values")
    return result
