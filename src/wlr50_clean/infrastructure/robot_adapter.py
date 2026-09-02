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
    SERVO_COMMAND_SIGN,
    SERVO_ORDER,
    WHEEL_FORWARD_SIGN,
    WHEEL_ORDER,
    WHEEL_VELOCITY_LIMIT_RAD_S,
    CommandBatchError,
    Full12Command,
    JointIndexMap,
    build_physical_batch,
    logical_readback_from_physical,
    resolve_joint_indices,
    servo_limits_deg,
)
from .servo_target_mapper import (
    SERVO_TRACKING_COMPENSATION_GAIN,
    SERVO_TRACKING_COMPENSATION_MAX_DEG,
    SERVO_TRACKING_FEEDBACK_INTERVAL_TICKS,
    ServoTargetMapper,
    ServoTargetMapperError,
)


ROBOT_PRIM_PATH = "/World/WLRRobot"
PHYSX_SAFE_LIMIT_MIN_RAD = -2.0 * math.pi
PHYSX_SAFE_LIMIT_MAX_RAD = 2.0 * math.pi
PHYSX_WRITE_LIMIT_MARGIN_RAD = 1.0e-6
PHYSX_TARGET_QUANTIZATION_MARGIN_RAD = 1.0e-5


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


@dataclass(frozen=True, slots=True)
class LiveServoLimitRecord:
    """Evidence for one authoritative limit authored on the live session layer."""

    joint_name: str
    joint_id: int
    standing_pose_deg: float
    command_min_deg: float
    command_max_deg: float
    desired_actual_min_deg: float
    desired_actual_max_deg: float
    runtime_min_rad: float
    runtime_max_rad: float
    runtime_joint_prim_path: str
    runtime_authoring_layer: str
    write_source: str = "runtime_usd_session_layer_override"
    applied: bool = True
    source_asset_modified: bool = False
    stage_saved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "joint_id": self.joint_id,
            "standing_pose_deg": self.standing_pose_deg,
            "command_min_deg": self.command_min_deg,
            "command_max_deg": self.command_max_deg,
            "desired_actual_min_deg": self.desired_actual_min_deg,
            "desired_actual_max_deg": self.desired_actual_max_deg,
            "runtime_min_rad": self.runtime_min_rad,
            "runtime_max_rad": self.runtime_max_rad,
            "runtime_joint_prim_path": self.runtime_joint_prim_path,
            "runtime_authoring_layer": self.runtime_authoring_layer,
            "write_source": self.write_source,
            "applied": self.applied,
            "source_asset_modified": self.source_asset_modified,
            "stage_saved": self.stage_saved,
        }


def authoritative_servo_limits_rad(
    joint_name: str,
    standing_pose_deg: float,
) -> tuple[float, float]:
    """Map canonical command limits to strict, PhysX-safe physical limits.

    The imported asset contains stale native limits (notably a zero-radian
    lower stop on the front-left knee).  The successful mature environment
    treats command-space limits as authoritative.  This pure helper preserves
    that exact standing-pose/sign mapping and its float32 endpoint envelope.
    """

    if not math.isfinite(float(standing_pose_deg)):
        raise RobotAdapterError(f"standing pose for {joint_name} is not finite")
    try:
        command_min_deg, command_max_deg = servo_limits_deg(joint_name)
        sign = float(SERVO_COMMAND_SIGN[joint_name])
    except (CommandBatchError, KeyError) as exc:
        raise RobotAdapterError(f"unknown servo joint: {joint_name!r}") from exc
    actual_a = float(standing_pose_deg) + sign * float(command_min_deg)
    actual_b = float(standing_pose_deg) + sign * float(command_max_deg)
    desired_min_rad = max(
        math.radians(min(actual_a, actual_b)),
        PHYSX_SAFE_LIMIT_MIN_RAD,
    )
    desired_max_rad = min(
        math.radians(max(actual_a, actual_b)),
        PHYSX_SAFE_LIMIT_MAX_RAD,
    )
    runtime_min_rad = max(
        desired_min_rad - PHYSX_TARGET_QUANTIZATION_MARGIN_RAD,
        PHYSX_SAFE_LIMIT_MIN_RAD + PHYSX_WRITE_LIMIT_MARGIN_RAD,
    )
    runtime_max_rad = min(
        desired_max_rad + PHYSX_TARGET_QUANTIZATION_MARGIN_RAD,
        PHYSX_SAFE_LIMIT_MAX_RAD - PHYSX_WRITE_LIMIT_MARGIN_RAD,
    )
    if runtime_min_rad >= runtime_max_rad:
        raise RobotAdapterError(
            f"authoritative physical limits are empty for {joint_name}: "
            f"[{runtime_min_rad}, {runtime_max_rad}]"
        )
    return runtime_min_rad, runtime_max_rad


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
        self.servo_target_mapper = ServoTargetMapper(
            self.standing_pose_deg,
            physics_dt_s=self.physics_dt_s,
        )
        self._final_drive_servo_deg = {name: 0.0 for name in SERVO_ORDER}
        # This frozen environment initialization must precede the 1.5 s
        # zero-command settle. It changes only the live USD session layer and
        # Isaac Lab's limit caches; it cannot save or modify the source asset.
        self.live_servo_limit_records = self._install_authoritative_servo_limits()
        self._physx_servo_limits_verified = False
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
        tracking_servo_names: Sequence[str] = (),
        drive_feedback_bias_full12: Sequence[float] = (0.0,) * len(FULL12_ORDER),
    ) -> dict[str, Any]:
        """Stage all 12 targets and issue exactly one articulation write.

        ``physics_tick`` is optional for simple callers. When supplied, it must
        be strictly increasing, which catches accidental double writes in a
        runtime tick. Mapping input must contain exactly all 12 canonical keys.
        Drive feedback is also exact Full12 order: servo-degree offsets retain
        the bounded post-mapper path, while canonical wheel-rad/s offsets are
        applied before physical sign conversion and clamped to the hard limit.
        """

        requested = self._coerce_command(command)
        tick = self._validate_tick(physics_tick)
        logical_applied = requested.clamped()
        feedback_bias = _full12_drive_feedback_bias(drive_feedback_bias_full12)
        measured_servo_rad = _row_values(
            _joint_matrix(self.robot, "joint_pos")[:, list(self.joint_map.servo_ids)]
        )
        try:
            mapping = self.servo_target_mapper.advance(
                logical_applied.servo_deg,
                measured_servo_rad,
                tracking_servo_names=tracking_servo_names,
            )
        except ServoTargetMapperError as exc:
            raise RobotAdapterError(f"invalid servo target mapping: {exc}") from exc
        native_drive = mapping.applied_drive_command_deg
        final_drive: list[float] = []
        realized_servo_bias: list[float] = []
        for name, native, bias in zip(
            SERVO_ORDER,
            native_drive,
            feedback_bias[: len(SERVO_ORDER)],
            strict=True,
        ):
            lower, upper = servo_limits_deg(name)
            previous = self._final_drive_servo_deg[name]
            final = bounded_drive_feedback_step(
                previous_deg=previous,
                native_deg=native,
                bias_deg=bias,
                maximum_delta_deg=self.servo_target_mapper.maximum_delta_deg,
                lower_deg=lower,
                upper_deg=upper,
            )
            self._final_drive_servo_deg[name] = final
            final_drive.append(final)
            realized_servo_bias.append(final - native)
        final_wheels = tuple(
            max(
                -WHEEL_VELOCITY_LIMIT_RAD_S,
                min(WHEEL_VELOCITY_LIMIT_RAD_S, native + bias),
            )
            for native, bias in zip(
                logical_applied.wheel_rad_s,
                feedback_bias[len(SERVO_ORDER) :],
                strict=True,
            )
        )
        realized_wheel_bias = tuple(
            final - native
            for final, native in zip(
                final_wheels,
                logical_applied.wheel_rad_s,
                strict=True,
            )
        )
        drive_command = Full12Command(
            tuple(final_drive),
            final_wheels,
        )
        native_drive_command = Full12Command(
            native_drive,
            logical_applied.wheel_rad_s,
        )
        physical = build_physical_batch(drive_command, self.standing_pose_deg)

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
        ack: dict[str, Any] = {
            "schema": "wlr50_clean.atomic_full12_ack.v1",
            "physics_tick": tick,
            "physics_dt_s": self.physics_dt_s,
            "write_count": self.write_count,
            "articulation_writes_this_call": 1,
            "canonical_order": list(FULL12_ORDER),
            "requested_full12": list(requested.to_full12()),
            # Logical applied remains the canonical FSM/PPO command after hard
            # limits.  The separate drive fields expose mature target shaping.
            "applied_full12": list(logical_applied.to_full12()),
            "drive_target_full12": list(drive_command.to_full12()),
            "native_drive_target_full12": list(native_drive_command.to_full12()),
            "drive_feedback_bias_requested_full12": list(feedback_bias),
            "drive_feedback_bias_realized_full12": list(realized_servo_bias)
            + list(realized_wheel_bias),
            "drive_feedback_final_slew_limit_deg_per_tick": (
                self.servo_target_mapper.maximum_delta_deg
            ),
            "command_was_clamped": requested != logical_applied,
            "servo_applied_drive_command_deg": list(final_drive),
            "servo_native_drive_command_deg": list(native_drive),
            "servo_tracking_compensation_deg": list(
                mapping.tracking_compensation_deg
            ),
            "servo_nominal_target_reached": list(
                mapping.nominal_target_reached
            ),
            "servo_tracking_active": list(mapping.tracking_active),
            "tracking_servo_names": list(tracking_servo_names),
            "servo_tracking_feedback_sample_tick": mapping.feedback_sample_tick,
            "servo_tracking_feedback_sampled": mapping.feedback_sampled,
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

    def joint_limit_initialization_evidence(self) -> dict[str, Any]:
        records = [record.as_dict() for record in self.live_servo_limit_records]
        return {
            "schema": "wlr50_clean.authoritative_servo_limit_initialization.v1",
            "source_environment_invariant": "mature_command_space_limits",
            "runtime_authoring_layer": "session_layer",
            "session_limits_authored_joint_count": len(records),
            "physx_limits_verified": self._physx_servo_limits_verified,
            "all_eight_servo_limits_applied": (
                len(records) == len(SERVO_ORDER)
                and self._physx_servo_limits_verified
            ),
            "source_asset_modified": False,
            "stage_saved": False,
            "records": records,
            "servo_target_mapping": {
                "source_environment_invariant": "mature_ui_command_space_to_drive_target",
                "servo_rate_deg_s": self.servo_target_mapper.servo_rate_deg_s,
                "tracking_gain": SERVO_TRACKING_COMPENSATION_GAIN,
                "tracking_limit_deg": SERVO_TRACKING_COMPENSATION_MAX_DEG,
                "feedback_interval_ticks": SERVO_TRACKING_FEEDBACK_INTERVAL_TICKS,
                "changes_actuator_properties": False,
            },
        }

    def verify_authoritative_servo_limits_adopted(self) -> None:
        """Fail closed unless the live PhysX articulation adopted all limits."""

        self._physx_servo_limits_verified = False
        try:
            live_limits = self.robot.root_physx_view.get_dof_limits()
            shape = tuple(live_limits.shape)
        except Exception as exc:
            raise RobotAdapterError("live PhysX DOF limits are unavailable") from exc
        if len(shape) != 3 or shape[0] != 1 or shape[2] != 2:
            raise RobotAdapterError(
                f"live PhysX DOF limits must have shape (1, joints, 2); received {shape}"
            )
        for record in self.live_servo_limit_records:
            actual_min = _scalar_float(live_limits[0, record.joint_id, 0])
            actual_max = _scalar_float(live_limits[0, record.joint_id, 1])
            if not math.isclose(
                actual_min,
                record.runtime_min_rad,
                rel_tol=0.0,
                abs_tol=2.0e-6,
            ):
                raise RobotAdapterError(
                    f"PhysX lower limit did not adopt session value for {record.joint_name}: "
                    f"expected {record.runtime_min_rad}, received {actual_min}"
                )
            if not math.isclose(
                actual_max,
                record.runtime_max_rad,
                rel_tol=0.0,
                abs_tol=2.0e-6,
            ):
                raise RobotAdapterError(
                    f"PhysX upper limit did not adopt session value for {record.joint_name}: "
                    f"expected {record.runtime_max_rad}, received {actual_max}"
                )
        self._physx_servo_limits_verified = True

    def _install_authoritative_servo_limits(self) -> tuple[LiveServoLimitRecord, ...]:
        """Install eight command-space-derived limits on the live session only."""

        try:
            from isaaclab.sim import get_current_stage  # type: ignore
            from pxr import Usd, UsdPhysics  # type: ignore
        except Exception as exc:
            raise RobotAdapterError("live USD APIs are unavailable for servo-limit initialization") from exc

        try:
            stage = get_current_stage()
            root_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
            if not root_prim.IsValid():
                raise RobotAdapterError(f"runtime robot prim is missing: {ROBOT_PRIM_PATH}")
            session_layer = stage.GetSessionLayer()
            if session_layer is None:
                raise RobotAdapterError("live USD stage has no session layer")
            layer_identifier = str(
                getattr(session_layer, "identifier", "")
                or getattr(session_layer, "GetIdentifier", lambda: "")()
                or "anonymous_session_layer"
            )

            # Resolve every exact RevoluteJoint before authoring anything, so
            # bad stage topology cannot leave a partial initialization behind.
            resolved: list[tuple[str, int, Any, float, float, float, float, float]] = []
            for joint_name, joint_id in zip(
                SERVO_ORDER,
                self.joint_map.servo_ids,
                strict=True,
            ):
                matches = [
                    prim
                    for prim in Usd.PrimRange(root_prim)
                    if prim.GetName() == joint_name and prim.IsA(UsdPhysics.RevoluteJoint)
                ]
                if len(matches) != 1:
                    paths = [str(prim.GetPath()) for prim in matches]
                    raise RobotAdapterError(
                        f"expected one RevoluteJoint prim named {joint_name}, found {paths}"
                    )
                standing_deg = float(self.standing_pose_deg[joint_name])
                command_min_deg, command_max_deg = servo_limits_deg(joint_name)
                sign = float(SERVO_COMMAND_SIGN[joint_name])
                desired_a = standing_deg + sign * float(command_min_deg)
                desired_b = standing_deg + sign * float(command_max_deg)
                runtime_min_rad, runtime_max_rad = authoritative_servo_limits_rad(
                    joint_name,
                    standing_deg,
                )
                resolved.append(
                    (
                        joint_name,
                        int(joint_id),
                        matches[0],
                        standing_deg,
                        min(desired_a, desired_b),
                        max(desired_a, desired_b),
                        runtime_min_rad,
                        runtime_max_rad,
                    )
                )

            hard_limits = _joint_limit_matrix(self.robot, "joint_pos_limits")
            soft_limits = _joint_limit_matrix(self.robot, "soft_joint_pos_limits")
            soft_factor = float(
                getattr(getattr(self.robot, "cfg", None), "soft_joint_pos_limit_factor", 1.0)
            )
            if not math.isfinite(soft_factor) or not 0.0 < soft_factor <= 1.0:
                raise RobotAdapterError(
                    f"soft_joint_pos_limit_factor must be in (0, 1]; received {soft_factor}"
                )

            records: list[LiveServoLimitRecord] = []
            edit_target = Usd.EditTarget(session_layer)
            with Usd.EditContext(stage, edit_target):
                for (
                    joint_name,
                    joint_id,
                    prim,
                    standing_deg,
                    desired_min_deg,
                    desired_max_deg,
                    runtime_min_rad,
                    runtime_max_rad,
                ) in resolved:
                    joint = UsdPhysics.RevoluteJoint(prim)
                    lower_attr = joint.CreateLowerLimitAttr()
                    upper_attr = joint.CreateUpperLimitAttr()
                    lower_deg = math.degrees(runtime_min_rad)
                    upper_deg = math.degrees(runtime_max_rad)
                    if not lower_attr.Set(float(lower_deg)) or not upper_attr.Set(float(upper_deg)):
                        raise RobotAdapterError(f"failed to author live limits for {joint_name}")
                    authored_lower = float(lower_attr.Get())
                    authored_upper = float(upper_attr.Get())
                    # USD Physics limit attributes are float32; verification
                    # therefore allows only their expected degree-space ULPs.
                    if not math.isclose(authored_lower, lower_deg, rel_tol=0.0, abs_tol=2.0e-5):
                        raise RobotAdapterError(f"live lower-limit verification failed for {joint_name}")
                    if not math.isclose(authored_upper, upper_deg, rel_tol=0.0, abs_tol=2.0e-5):
                        raise RobotAdapterError(f"live upper-limit verification failed for {joint_name}")

                    # Keep Isaac Lab's hard/soft diagnostic cache identical to
                    # the authored limits without writing state or wheel DOFs.
                    hard_limits[:, joint_id, 0] = float(runtime_min_rad)
                    hard_limits[:, joint_id, 1] = float(runtime_max_rad)
                    midpoint = 0.5 * (runtime_min_rad + runtime_max_rad)
                    half_range = 0.5 * (runtime_max_rad - runtime_min_rad) * soft_factor
                    soft_limits[:, joint_id, 0] = midpoint - half_range
                    soft_limits[:, joint_id, 1] = midpoint + half_range
                    command_min_deg, command_max_deg = servo_limits_deg(joint_name)
                    records.append(
                        LiveServoLimitRecord(
                            joint_name=joint_name,
                            joint_id=joint_id,
                            standing_pose_deg=standing_deg,
                            command_min_deg=float(command_min_deg),
                            command_max_deg=float(command_max_deg),
                            desired_actual_min_deg=desired_min_deg,
                            desired_actual_max_deg=desired_max_deg,
                            runtime_min_rad=runtime_min_rad,
                            runtime_max_rad=runtime_max_rad,
                            runtime_joint_prim_path=str(prim.GetPath()),
                            runtime_authoring_layer=layer_identifier,
                        )
                    )
        except RobotAdapterError:
            raise
        except Exception as exc:
            raise RobotAdapterError(
                f"authoritative live servo-limit initialization failed: {type(exc).__name__}: {exc}"
            ) from exc
        if len(records) != len(SERVO_ORDER):
            raise RobotAdapterError(
                f"expected {len(SERVO_ORDER)} applied live servo limits; received {len(records)}"
            )
        return tuple(records)

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


def _joint_limit_matrix(robot: Any, field: str) -> Any:
    try:
        value = getattr(robot.data, field)
        shape = tuple(value.shape)
    except Exception as exc:
        raise RobotAdapterError(f"robot.data.{field} is unavailable") from exc
    if len(shape) != 3 or shape[0] < 1 or shape[2] != 2:
        raise RobotAdapterError(
            f"robot.data.{field} must have shape (instances, joints, 2); received {shape}"
        )
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


def _full12_drive_feedback_bias(values: Sequence[float]) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise RobotAdapterError("drive feedback bias must be numeric") from exc
    if len(result) != len(FULL12_ORDER):
        raise RobotAdapterError("drive feedback bias must contain exactly 12 values")
    if any(not math.isfinite(value) for value in result):
        raise RobotAdapterError("drive feedback bias contains a non-finite value")
    if any(
        abs(value) > SERVO_TRACKING_COMPENSATION_MAX_DEG + 1.0e-12
        for value in result[: len(SERVO_ORDER)]
    ):
        raise RobotAdapterError("drive feedback bias exceeds the bounded mapper envelope")
    if any(
        abs(value) > WHEEL_VELOCITY_LIMIT_RAD_S + 1.0e-12
        for value in result[len(SERVO_ORDER) :]
    ):
        raise RobotAdapterError("drive feedback wheel bias exceeds the wheel hard limit")
    return result


def bounded_drive_feedback_step(
    *,
    previous_deg: float,
    native_deg: float,
    bias_deg: float,
    maximum_delta_deg: float,
    lower_deg: float,
    upper_deg: float,
) -> float:
    """Apply a logical post-mapper bias without exceeding final-drive slew."""

    previous = float(previous_deg)
    native = float(native_deg)
    bias = float(bias_deg)
    maximum_delta = float(maximum_delta_deg)
    lower = float(lower_deg)
    upper = float(upper_deg)
    values = (previous, native, bias, maximum_delta, lower, upper)
    if any(not math.isfinite(value) for value in values):
        raise RobotAdapterError("bounded drive-feedback step contains non-finite data")
    if maximum_delta <= 0.0 or lower >= upper:
        raise RobotAdapterError("bounded drive-feedback step has invalid limits")
    desired = max(lower, min(upper, native + bias))
    delta = max(-maximum_delta, min(maximum_delta, desired - previous))
    return max(lower, min(upper, previous + delta))


def _scalar_float(value: Any) -> float:
    try:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        item = getattr(value, "item", None)
        result = float(item() if callable(item) else value)
    except Exception as exc:
        raise RobotAdapterError("failed to read live PhysX limit scalar") from exc
    if not math.isfinite(result):
        raise RobotAdapterError("live PhysX limit scalar is not finite")
    return result
