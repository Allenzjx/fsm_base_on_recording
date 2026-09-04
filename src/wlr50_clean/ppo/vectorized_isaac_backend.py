"""True cloned-scene Isaac backend for the 8/16/32 environment smoke gate.

This module deliberately does not implement vectorization as a list of
single-environment simulations.  Production construction creates one Isaac
Lab ``InteractiveScene`` whose articulation tensors have a leading dimension
equal to ``num_envs``.  Every backend tick stages one batched actuator write,
advances the one ``SimulationContext`` exactly once, captures the exact-pair
contact bank once, and then advances an independent frozen FSM/controller per
row.

Isaac imports remain lazy.  Ordinary Python can import this module and run the
contract/probe tests without pretending that a live benchmark took place.
"""

from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from wlr50_clean.fsm.controller import SensorFsmController
from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    WHEEL_ORDER,
    WHEEL_VELOCITY_LIMIT_RAD_S,
    Full12Command,
    build_physical_batch,
    logical_readback_from_physical,
    resolve_joint_indices,
    servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import (
    JointStateSnapshot,
    authoritative_servo_limits_rad,
    bounded_drive_feedback_step,
)
from wlr50_clean.infrastructure.scene_factory import (
    DEVICE,
    GRAVITY_M_S2,
    GROUND_DYNAMIC_FRICTION,
    GROUND_PRIM_PATH,
    GROUND_RESTITUTION,
    GROUND_SIZE_M,
    GROUND_STATIC_FRICTION,
    OBSTACLE_CENTER_M,
    OBSTACLE_CONTACT_OFFSET_M,
    OBSTACLE_DYNAMIC_FRICTION,
    OBSTACLE_HEIGHT_M,
    OBSTACLE_LENGTH_M,
    OBSTACLE_RESTITUTION,
    OBSTACLE_REST_OFFSET_M,
    OBSTACLE_STATIC_FRICTION,
    OBSTACLE_WIDTH_M,
    PHYSICS_DT_S,
    RENDER_INTERVAL_PHYSICS_STEPS,
    ROBOT_USD_SHA256,
    _build_robot_cfg,
    locked_scene_snapshot,
    validate_locked_scene,
)
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.ppo.isaac_fsm_backend import (
    DEFAULT_FSM_PATH,
    DEFAULT_MOTION_CONTRACT_PATH,
    LEVEL_CALIBRATION_TICKS,
    SETTLE_TICKS,
    IsaacFSMBackend,
    _frame_is_terminal,
    _mean_quaternion,
    _observation_quaternion,
    _validate_controller_clock,
    _validate_sensor_contract,
)
from wlr50_clean.ppo.ppo_env_adapter import AuthoritativeFrame
from wlr50_clean.sensing.contact_classifier import (
    GROUND_PAIR,
    OBSTACLE_PAIR,
    SENSED_BODIES,
    RawPairContact,
)
from wlr50_clean.sensing.geometry import ColliderGeometryCache, UsdCollisionBoundsProvider
from wlr50_clean.sensing.sensor_reader import (
    GROUND_COLLISION_PRIM_PATH,
    OBSTACLE_PRIM_PATH,
    SensorReader,
)


SUPPORTED_VECTOR_ENV_COUNTS = (8, 16, 32)
ENV_SPACING_M = 8.0
CONTACT_HISTORY_LENGTH = 3
PAIR_COUNT = 2
ZERO12 = (0.0,) * 12


def _shared_ground_size_m(
    num_envs: int,
    env_spacing_m: float,
) -> tuple[float, float]:
    """Cover every GridCloner origin with the locked 6 m local footprint.

    Isaac's ground-plane collision shape is infinite, while ``size`` controls
    the visible grid mesh.  Reproduce the locked GridCloner layout so the
    shared global plane is also visually complete for every cloned scene.
    """

    count = _validated_env_count(num_envs)
    spacing = float(env_spacing_m)
    if not math.isfinite(spacing) or spacing < 6.0:
        raise VectorizedIsaacBackendError(
            "environment spacing must be at least 6 m for the locked ground footprint"
        )
    columns_hint = int(math.sqrt(count))
    rows = int(math.ceil(count / columns_hint))
    columns = int(math.ceil(count / rows))
    return (
        float(GROUND_SIZE_M[0]) + (rows - 1) * spacing,
        float(GROUND_SIZE_M[1]) + (columns - 1) * spacing,
    )


class VectorizedIsaacBackendError(RuntimeError):
    """A true-batching, sensing, timing, or independence invariant failed."""


class VectorizedExactPairFailure(VectorizedIsaacBackendError):
    """The cloned exact-pair contact tensors cannot be trusted."""


@dataclass(frozen=True, slots=True)
class VectorBackendProbe:
    status: str
    num_envs: int
    supported_count: bool
    isaaclab_importable_in_current_python: bool
    locked_asset_verified: bool
    live_vectorization_verified: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BatchedAuthoritativeFrame:
    """One synchronized result from a single global physics advance."""

    physics_tick: int
    sim_time_s: float
    frames: tuple[AuthoritativeFrame, ...]
    global_physics_step_count: int
    batched_articulation_write_count: int
    exact_pair_capture_count: int

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(frame.state_id for frame in self.frames)

    @property
    def nominal_actions_full12(self) -> tuple[tuple[float, ...], ...]:
        return tuple(frame.nominal_action_full12 for frame in self.frames)


@dataclass(frozen=True, slots=True)
class VectorBenchmarkReport:
    status: str
    num_envs: int
    measured_ticks: int
    wall_time_s: float
    physics_steps_per_second: float
    environment_steps_per_second: float
    one_simulation_context: bool
    articulation_tensor_instances: int
    global_physics_steps: int
    batched_articulation_writes: int
    exact_pair_captures: int
    exact_pair_sensor_count: int
    independent_controller_count: int
    independent_reader_count: int
    final_state_ids: tuple[str, ...]
    true_batched_isaac_verified: bool
    failure_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def probe_vectorized_isaac_backend(
    num_envs: int,
    *,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
    verify_asset: bool = True,
) -> VectorBackendProbe:
    """Return an honest offline preflight; this function never claims a live pass."""

    count = _validated_env_count(num_envs)
    reasons: list[str] = []
    importable = module_finder("isaaclab") is not None
    if not importable:
        reasons.append(
            "isaaclab is not importable in the current Python; use the locked Isaac environment"
        )
    asset_ok = False
    try:
        validate_locked_scene(verify_asset_hash=verify_asset)
        asset_ok = True
    except Exception as exc:
        reasons.append(f"locked scene/asset validation failed: {type(exc).__name__}: {exc}")
    reasons.append(
        "live cloned tensor, exact-pair, controller-independence, and one-step checks have not run"
    )
    return VectorBackendProbe(
        status=(
            "LIVE_ISAAC_BENCHMARK_REQUIRED"
            if importable and asset_ok
            else "UNSUPPORTED_IN_CURRENT_RUNTIME"
        ),
        num_envs=count,
        supported_count=True,
        isaaclab_importable_in_current_python=importable,
        locked_asset_verified=asset_ok,
        live_vectorization_verified=False,
        reasons=tuple(reasons),
    )


class _RowRobotData:
    """One-row view of a genuinely batched articulation tensor bundle."""

    _LOCAL_POSITION_FIELDS = frozenset(
        {"body_link_pos_w", "body_com_pos_w", "root_pos_w"}
    )

    def __init__(self, source: Any, row: int, origin: Any, num_envs: int) -> None:
        self._source = source
        self._row = int(row)
        self._origin = origin
        self._num_envs = int(num_envs)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._source, name)
        shape = tuple(getattr(value, "shape", ()))
        if not shape or int(shape[0]) != self._num_envs:
            return value
        row = value[self._row : self._row + 1]
        if name in self._LOCAL_POSITION_FIELDS:
            row = row.clone()
            row[..., :3] -= self._origin
        return row


class _RowRobot:
    def __init__(self, robot: Any, row: int, origin: Any, num_envs: int) -> None:
        self.body_names = tuple(str(name) for name in robot.body_names)
        self.joint_names = tuple(str(name) for name in robot.joint_names)
        self.data = _RowRobotData(robot.data, row, origin, num_envs)


class _RowReadAdapter:
    def __init__(
        self,
        adapter: "_BatchedCommandAdapter",
        row: int,
        robot: _RowRobot,
    ) -> None:
        self._adapter = adapter
        self._row = int(row)
        self.robot = robot

    def get_actual_state(self) -> JointStateSnapshot:
        return self._adapter.actual_state(self._row)


@dataclass(frozen=True, slots=True)
class _ContactCapture:
    force: np.ndarray
    history: np.ndarray
    point: np.ndarray
    friction: np.ndarray


class _BatchedExactPairContactBank:
    """Validate and cache all exact-pair tensors once per global tick."""

    def __init__(
        self,
        sensors: Mapping[str, Any],
        origins: np.ndarray,
        num_envs: int,
        *,
        expected_filter_paths: Sequence[str] | None = None,
    ) -> None:
        if set(sensors) != set(SENSED_BODIES):
            raise VectorizedExactPairFailure(
                "contact bank must contain exactly the locked 13 body sensors"
            )
        if origins.shape != (num_envs, 3):
            raise VectorizedExactPairFailure("environment origins must have shape (N, 3)")
        self.sensors = dict(sensors)
        self.origins = origins
        self.num_envs = int(num_envs)
        self.expected_filter_paths = (
            None
            if expected_filter_paths is None
            else tuple(str(path) for path in expected_filter_paths)
        )
        if self.expected_filter_paths is not None:
            if len(self.expected_filter_paths) != PAIR_COUNT:
                raise VectorizedExactPairFailure(
                    "exact-pair bank requires obstacle then ground filter paths"
                )
            for body_name, sensor in self.sensors.items():
                configured = tuple(
                    str(path)
                    for path in getattr(
                        getattr(sensor, "cfg", None), "filter_prim_paths_expr", ()
                    )
                )
                if configured != self.expected_filter_paths:
                    raise VectorizedExactPairFailure(
                        f"{body_name} filter order/path mismatch: {configured}"
                    )
        self.capture_count = 0
        self._last_tick: int | None = None
        self._captures: dict[str, _ContactCapture] = {}

    @property
    def initialized(self) -> bool:
        return all(bool(getattr(sensor, "is_initialized", False)) for sensor in self.sensors.values())

    def reset(self) -> None:
        for sensor in self.sensors.values():
            sensor.reset()
        self._last_tick = None
        self._captures = {}

    def capture(self, physics_tick: int) -> None:
        tick = int(physics_tick)
        if self._last_tick is not None and tick <= self._last_tick:
            raise VectorizedExactPairFailure(
                "exact-pair bank may be captured only once per increasing global tick"
            )
        captured: dict[str, _ContactCapture] = {}
        expected_matrix = (self.num_envs, 1, PAIR_COUNT, 3)
        expected_history = (
            self.num_envs,
            CONTACT_HISTORY_LENGTH,
            1,
            PAIR_COUNT,
            3,
        )
        for body_name in SENSED_BODIES:
            sensor = self.sensors[body_name]
            if not bool(getattr(sensor, "is_initialized", False)):
                raise VectorizedExactPairFailure(
                    f"{body_name} ContactSensor is not initialized"
                )
            live_names = tuple(str(name) for name in getattr(sensor, "body_names", ()))
            if live_names != (body_name,):
                raise VectorizedExactPairFailure(
                    f"{body_name} sensor body resolution is ambiguous: {live_names}"
                )
            filter_count = int(getattr(getattr(sensor, "contact_physx_view", None), "filter_count", -1))
            if filter_count != PAIR_COUNT:
                raise VectorizedExactPairFailure(
                    f"{body_name} exact-pair filter_count={filter_count}, expected {PAIR_COUNT}"
                )
            data = sensor.data
            force = _numpy(getattr(data, "force_matrix_w", None))
            history = _numpy(getattr(data, "force_matrix_w_history", None))
            point = _numpy(getattr(data, "contact_pos_w", None))
            friction = _numpy(getattr(data, "friction_forces_w", None))
            if force.shape != expected_matrix:
                raise VectorizedExactPairFailure(
                    f"{body_name} force_matrix_w shape={force.shape}, expected {expected_matrix}"
                )
            if history.shape != expected_history:
                raise VectorizedExactPairFailure(
                    f"{body_name} force history shape={history.shape}, expected {expected_history}"
                )
            if point.shape != expected_matrix or friction.shape != expected_matrix:
                raise VectorizedExactPairFailure(
                    f"{body_name} point/friction tensors are not exact-pair {expected_matrix}"
                )
            if not np.isfinite(force).all() or not np.isfinite(history).all():
                raise VectorizedExactPairFailure(
                    f"{body_name} exact-pair force tensors contain non-finite values"
                )
            # Contact points are NaN by contract for inactive pairs. Friction
            # must remain finite because it contributes directly to slip cost.
            if not np.isfinite(friction).all():
                raise VectorizedExactPairFailure(
                    f"{body_name} friction tensor contains non-finite values"
                )
            captured[body_name] = _ContactCapture(force, history, point, friction)
        self._captures = captured
        self._last_tick = tick
        self.capture_count += 1

    def row_backend(self, row: int) -> "_RowContactBackend":
        if not 0 <= int(row) < self.num_envs:
            raise VectorizedExactPairFailure(f"contact row {row} is out of range")
        return _RowContactBackend(self, int(row))

    def sample_row(self, row: int) -> tuple[RawPairContact, ...]:
        if self._last_tick is None or set(self._captures) != set(SENSED_BODIES):
            raise VectorizedExactPairFailure(
                "exact-pair data must be captured after the global physics step"
            )
        result: list[RawPairContact] = []
        origin = self.origins[row]
        for body_name in SENSED_BODIES:
            capture = self._captures[body_name]
            for pair_index, (kind, canonical_path) in enumerate(
                (
                    (OBSTACLE_PAIR, OBSTACLE_PRIM_PATH),
                    (GROUND_PAIR, GROUND_COLLISION_PRIM_PATH),
                )
            ):
                raw_point = capture.point[row, 0, pair_index]
                point = None
                if np.isfinite(raw_point).all():
                    point = tuple(float(raw_point[index] - origin[index]) for index in range(3))
                result.append(
                    RawPairContact(
                        sensor_body=body_name,
                        pair_kind=kind,
                        other_body=canonical_path,
                        force_w_n=_vec3(capture.force[row, 0, pair_index]),
                        friction_force_w_n=_vec3(capture.friction[row, 0, pair_index]),
                        contact_point_w_m=point,
                        history_force_w_n=tuple(
                            _vec3(capture.history[row, history_index, 0, pair_index])
                            for history_index in range(CONTACT_HISTORY_LENGTH)
                        ),
                        source="isaaclab.ContactSensor.batched_exact_pair_force_matrix_w",
                        pair_verified=True,
                    )
                )
        return tuple(result)


class _RowContactBackend:
    last_quality: tuple[str, ...] = ()

    def __init__(self, bank: _BatchedExactPairContactBank, row: int) -> None:
        self._bank = bank
        self._row = int(row)

    def sample(self, physics_dt_s: float) -> tuple[RawPairContact, ...]:
        if not math.isclose(float(physics_dt_s), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise VectorizedExactPairFailure("row contact reader must remain at 120 Hz")
        return self._bank.sample_row(self._row)


@dataclass(frozen=True, slots=True)
class _BatchAck:
    rows: tuple[Mapping[str, Any], ...]
    articulation_writes_this_call: int
    physics_tick: int


class _BatchedCommandAdapter:
    """Per-row mature servo shaping followed by one batched articulation write."""

    def __init__(self, robot: Any, num_envs: int) -> None:
        self.robot = robot
        self.num_envs = int(num_envs)
        self.joint_map = resolve_joint_indices(tuple(str(name) for name in robot.joint_names))
        shape = tuple(robot.data.joint_pos.shape)
        if len(shape) != 2 or int(shape[0]) != self.num_envs:
            raise VectorizedIsaacBackendError(
                f"articulation joint tensor must be ({self.num_envs}, joints), received {shape}"
            )
        self._standing_servo = robot.data.joint_pos[:, list(self.joint_map.servo_ids)].clone()
        self.standing_pose_deg: list[dict[str, float]] = []
        self.mappers: list[ServoTargetMapper] = []
        self._final_drive: list[dict[str, float]] = []
        for row in range(self.num_envs):
            standing = {
                name: math.degrees(value)
                for name, value in zip(
                    SERVO_ORDER,
                    _tensor_row(self._standing_servo[row]),
                    strict=True,
                )
            }
            self.standing_pose_deg.append(standing)
            self.mappers.append(ServoTargetMapper(standing, physics_dt_s=PHYSICS_DT_S))
            self._final_drive.append({name: 0.0 for name in SERVO_ORDER})
        self._install_limits()
        self.write_count = 0
        self._last_tick: int | None = None

    def _install_limits(self) -> None:
        import torch

        limits = torch.empty(
            (self.num_envs, len(SERVO_ORDER), 2),
            dtype=self.robot.data.joint_pos.dtype,
            device=self.robot.data.joint_pos.device,
        )
        for row in range(self.num_envs):
            for local, name in enumerate(SERVO_ORDER):
                lower, upper = authoritative_servo_limits_rad(
                    name, self.standing_pose_deg[row][name]
                )
                limits[row, local, 0] = lower
                limits[row, local, 1] = upper
        self.robot.write_joint_position_limit_to_sim(
            limits,
            joint_ids=list(self.joint_map.servo_ids),
            warn_limit_violation=False,
        )
        live = self.robot.root_physx_view.get_dof_limits()
        if tuple(live.shape[:2]) != tuple(self.robot.data.joint_pos.shape):
            raise VectorizedIsaacBackendError("live PhysX limit tensor shape is incomplete")
        for row in range(self.num_envs):
            for local, joint_id in enumerate(self.joint_map.servo_ids):
                expected = _tensor_row(limits[row, local])
                actual = _tensor_row(live[row, joint_id])
                if any(
                    not math.isclose(a, b, rel_tol=0.0, abs_tol=2.0e-6)
                    for a, b in zip(actual, expected, strict=True)
                ):
                    raise VectorizedIsaacBackendError(
                        f"PhysX did not adopt authoritative servo limit in env {row}, joint {joint_id}"
                    )

    def actual_state(self, row: int) -> JointStateSnapshot:
        servo_pos = _tensor_row(
            self.robot.data.joint_pos[row, list(self.joint_map.servo_ids)]
        )
        servo_vel = _tensor_row(
            self.robot.data.joint_vel[row, list(self.joint_map.servo_ids)]
        )
        wheel_vel = _tensor_row(
            self.robot.data.joint_vel[row, list(self.joint_map.wheel_ids)]
        )
        full12 = logical_readback_from_physical(
            servo_pos, wheel_vel, self.standing_pose_deg[row]
        )
        return JointStateSnapshot(full12, servo_pos, servo_vel, wheel_vel)

    def apply_batch(
        self,
        commands: Sequence[Sequence[float]],
        *,
        physics_tick: int,
        tracking_servo_names: Sequence[Sequence[str]],
        drive_feedback_bias_full12: Sequence[Sequence[float]],
    ) -> _BatchAck:
        tick = int(physics_tick)
        if tick < 0 or (self._last_tick is not None and tick <= self._last_tick):
            raise VectorizedIsaacBackendError("batched command tick must increase strictly")
        if not (
            len(commands)
            == len(tracking_servo_names)
            == len(drive_feedback_bias_full12)
            == self.num_envs
        ):
            raise VectorizedIsaacBackendError("batched command metadata must have exactly N rows")

        position_targets = self._standing_servo.clone()
        velocity_targets = self.robot.data.joint_vel[:, list(self.joint_map.wheel_ids)].clone()
        measured = self.robot.data.joint_pos[:, list(self.joint_map.servo_ids)]
        rows: list[Mapping[str, Any]] = []
        for row in range(self.num_envs):
            requested = _full12(commands[row], f"commands[{row}]")
            logical = Full12Command.from_full12(requested)
            if logical.clamped() != logical:
                raise VectorizedIsaacBackendError(
                    f"env {row} projected action exceeds canonical actuator limits"
                )
            feedback = _full12(
                drive_feedback_bias_full12[row],
                f"drive_feedback_bias_full12[{row}]",
            )
            mapping = self.mappers[row].advance(
                logical.servo_deg,
                _tensor_row(measured[row]),
                tracking_servo_names=tuple(str(name) for name in tracking_servo_names[row]),
            )
            final_servo: list[float] = []
            for name, native, bias in zip(
                SERVO_ORDER, mapping.applied_drive_command_deg, feedback[:8], strict=True
            ):
                lower, upper = servo_limits_deg(name)
                final = bounded_drive_feedback_step(
                    previous_deg=self._final_drive[row][name],
                    native_deg=native,
                    bias_deg=bias,
                    maximum_delta_deg=self.mappers[row].maximum_delta_deg,
                    lower_deg=lower,
                    upper_deg=upper,
                )
                self._final_drive[row][name] = final
                final_servo.append(final)
            final_wheels = tuple(
                max(
                    -WHEEL_VELOCITY_LIMIT_RAD_S,
                    min(WHEEL_VELOCITY_LIMIT_RAD_S, native + bias),
                )
                for native, bias in zip(logical.wheel_rad_s, feedback[8:], strict=True)
            )
            drive = Full12Command(tuple(final_servo), final_wheels)
            physical = build_physical_batch(drive, self.standing_pose_deg[row])
            position_targets[row, :] = _like_tensor(
                physical.servo_target_rad, position_targets
            )
            velocity_targets[row, :] = _like_tensor(
                physical.wheel_target_rad_s, velocity_targets
            )
            rows.append(
                {
                    "schema": "wlr50_clean.atomic_full12_ack.v1",
                    "physics_tick": tick,
                    "physics_dt_s": PHYSICS_DT_S,
                    "write_count": self.write_count + 1,
                    "articulation_writes_this_call": 1,
                    "batched_environment_count": self.num_envs,
                    "canonical_order": list(FULL12_ORDER),
                    "requested_full12": list(requested),
                    "applied_full12": list(requested),
                    "drive_target_full12": list(drive.to_full12()),
                    "drive_feedback_bias_requested_full12": list(feedback),
                    "tracking_servo_names": list(tracking_servo_names[row]),
                }
            )

        self.robot.set_joint_position_target(
            position_targets, joint_ids=list(self.joint_map.servo_ids)
        )
        self.robot.set_joint_velocity_target(
            velocity_targets, joint_ids=list(self.joint_map.wheel_ids)
        )
        # Exactly one articulation write for all N rows.
        self.robot.write_data_to_sim()
        self.write_count += 1
        self._last_tick = tick
        return _BatchAck(tuple(rows), 1, tick)


class VectorizedIsaacFSMBackend:
    """One cloned Isaac scene with N independent frozen FSM runtimes."""

    def __init__(
        self,
        simulation_app: Any,
        *,
        num_envs: int,
        device: str = DEVICE,
        env_spacing_m: float = ENV_SPACING_M,
        fsm_path: Path | str = DEFAULT_FSM_PATH,
        motion_contract_path: Path | str = DEFAULT_MOTION_CONTRACT_PATH,
    ) -> None:
        self.num_envs = _validated_env_count(num_envs)
        self.simulation_app = simulation_app
        self.device = str(device)
        self.env_spacing_m = float(env_spacing_m)
        if not math.isfinite(self.env_spacing_m) or self.env_spacing_m < 6.0:
            raise VectorizedIsaacBackendError(
                "environment spacing must be at least 6 m for the locked ground footprint"
            )
        self.fsm_path = Path(fsm_path).resolve()
        self.motion_contract_path = Path(motion_contract_path).resolve()
        if not self.fsm_path.is_file() or not self.motion_contract_path.is_file():
            raise VectorizedIsaacBackendError("frozen FSM/config paths are missing")
        if simulation_app is None or not bool(simulation_app.is_running()):
            raise VectorizedIsaacBackendError(
                "AppLauncher must create a running SimulationApp before vector backend construction"
            )
        validate_locked_scene(verify_asset_hash=True)

        self._modules = _load_live_modules()
        self.sim, self.scene = _create_vector_scene(
            self._modules,
            num_envs=self.num_envs,
            device=self.device,
            env_spacing_m=self.env_spacing_m,
        )
        self.robot = self.scene["robot"]
        self._validate_physical_batch()
        origins = _numpy(self.scene.env_origins)
        sensors = {
            body: self.scene.sensors[_sensor_key(body)] for body in SENSED_BODIES
        }
        self.contact_bank = _BatchedExactPairContactBank(
            sensors,
            origins,
            self.num_envs,
            expected_filter_paths=(
                f"{self.scene.env_regex_ns}/Obstacle",
                "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
            ),
        )
        if not self.contact_bank.initialized:
            raise VectorizedExactPairFailure(
                "the cloned exact 13-body ContactSensor bank did not initialize"
            )

        self.command_adapter: _BatchedCommandAdapter | None = None
        self.readers: tuple[SensorReader, ...] = ()
        self.controllers: tuple[SensorFsmController, ...] = ()
        self._semantics: tuple[IsaacFSMBackend, ...] = ()
        self._controller_frames: tuple[Any, ...] = ()
        self._frames: tuple[AuthoritativeFrame, ...] = ()
        self._done: tuple[bool, ...] = ()
        self._episode_tick = 0
        self.global_physics_step_count = 0
        self._reset_count = 0

    @property
    def frames(self) -> tuple[AuthoritativeFrame, ...]:
        if not self._frames:
            raise VectorizedIsaacBackendError("reset_all must precede frame access")
        return self._frames

    def reset_all(
        self,
        *,
        seeds: Sequence[int] | None = None,
        options: Sequence[Mapping[str, Any]] | None = None,
    ) -> BatchedAuthoritativeFrame:
        """Synchronously restore and settle every clone at an episode barrier."""

        seed_rows = tuple(range(self.num_envs)) if seeds is None else tuple(int(v) for v in seeds)
        option_rows = (
            tuple({} for _ in range(self.num_envs))
            if options is None
            else tuple(dict(value) for value in options)
        )
        if len(seed_rows) != self.num_envs or len(option_rows) != self.num_envs:
            raise VectorizedIsaacBackendError("reset_all requires one seed/options row per environment")
        if any(seed < 0 for seed in seed_rows):
            raise VectorizedIsaacBackendError("reset seeds must be non-negative")
        if any(row for row in option_rows):
            raise VectorizedIsaacBackendError(
                "vector benchmark accepts no randomization or state-mutation reset options"
            )
        self._require_running("vector reset")
        self._restore_default_state()
        self.command_adapter = _BatchedCommandAdapter(self.robot, self.num_envs)
        self.contact_bank.reset()

        calibration_readers: tuple[SensorReader, ...] = ()
        calibration_samples: list[list[tuple[float, float, float, float]]] = [
            [] for _ in range(self.num_envs)
        ]
        last_ack: _BatchAck | None = None
        zeros = tuple(ZERO12 for _ in range(self.num_envs))
        empty_tracking = tuple(() for _ in range(self.num_envs))
        for settle_tick in range(SETTLE_TICKS):
            last_ack = self.command_adapter.apply_batch(
                zeros,
                physics_tick=self.global_physics_step_count,
                tracking_servo_names=empty_tracking,
                drive_feedback_bias_full12=zeros,
            )
            self._advance_global_physics()
            if settle_tick == SETTLE_TICKS - LEVEL_CALIBRATION_TICKS:
                calibration_readers = self._make_readers()
            if calibration_readers:
                local_tick = settle_tick - (SETTLE_TICKS - LEVEL_CALIBRATION_TICKS)
                for row, reader in enumerate(calibration_readers):
                    observation = reader.read(
                        physics_tick=local_tick,
                        simulation_time_s=local_tick * PHYSICS_DT_S,
                        commanded_full12=last_ack.rows[row]["drive_target_full12"],
                    )
                    _validate_sensor_contract(
                        observation, SENSED_BODIES, require_finite=True
                    )
                    calibration_samples[row].append(
                        _observation_quaternion(observation)
                    )
        if last_ack is None or any(
            len(samples) != LEVEL_CALIBRATION_TICKS
            for samples in calibration_samples
        ):
            raise VectorizedIsaacBackendError("complete vector level calibration was not observed")

        self.readers = self._make_readers()
        self.controllers = tuple(
            SensorFsmController.from_paths(self.fsm_path, self.motion_contract_path)
            for _ in range(self.num_envs)
        )
        self._assert_independent_python_state()
        self._semantics = tuple(IsaacFSMBackend() for _ in range(self.num_envs))
        observations = []
        controller_frames = []
        frames = []
        for row, (reader, controller) in enumerate(
            zip(self.readers, self.controllers, strict=True)
        ):
            observation = reader.read(
                physics_tick=0,
                simulation_time_s=0.0,
                commanded_full12=last_ack.rows[row]["drive_target_full12"],
            )
            _validate_sensor_contract(observation, SENSED_BODIES, require_finite=True)
            controller_frame = controller.step(observation, sim_time_s=0.0)
            _validate_controller_clock(controller_frame, physics_tick=0, sim_time_s=0.0)
            semantic = self._semantics[row]
            semantic._controller = controller
            semantic._level_reference_orientation = _mean_quaternion(
                calibration_samples[row]
            )
            semantic._reset_metadata = self._reset_metadata(
                row=row,
                seed=seed_rows[row],
                options=option_rows[row],
            )
            semantic._last_atomic_ack = last_ack.rows[row]
            semantic._previous_action_full12 = ZERO12
            semantic._raw_observation = observation
            semantic._controller_frame = controller_frame
            semantic._episode_tick = 0
            frame = semantic._build_authoritative_frame(
                observation, controller_frame, previous_frame=None
            )
            semantic._authoritative_frame = frame
            semantic._done = _frame_is_terminal(frame)
            observations.append(observation)
            controller_frames.append(controller_frame)
            frames.append(frame)
        self._controller_frames = tuple(controller_frames)
        self._frames = tuple(frames)
        self._done = tuple(_frame_is_terminal(frame) for frame in frames)
        self._episode_tick = 0
        self._reset_count += 1
        return self._batched_frame()

    def step_physics_batch(
        self, applied_actions_full12: Sequence[Sequence[float]]
    ) -> BatchedAuthoritativeFrame:
        """Stage N actions and perform exactly one shared physics advance."""

        if (
            self.command_adapter is None
            or not self._frames
            or not self._controller_frames
            or not self.readers
        ):
            raise VectorizedIsaacBackendError("reset_all must precede step_physics_batch")
        if any(self._done):
            done_rows = tuple(index for index, value in enumerate(self._done) if value)
            raise VectorizedIsaacBackendError(
                f"synchronous batch contains terminated rows {done_rows}; reset_all is required"
            )
        actions = tuple(
            _full12(row, f"applied_actions_full12[{index}]")
            for index, row in enumerate(applied_actions_full12)
        )
        if len(actions) != self.num_envs:
            raise VectorizedIsaacBackendError(
                f"batched actions must have shape ({self.num_envs}, 12)"
            )
        self._require_running("vector episode")
        tracking: list[tuple[str, ...]] = []
        biases: list[tuple[float, ...]] = []
        for controller_frame in self._controller_frames:
            if not bool(getattr(controller_frame, "full12_atomic_write_required", False)):
                raise VectorizedIsaacBackendError(
                    "a frozen controller did not require an atomic Full12 write"
                )
            tracking.append(
                tuple(str(name) for name in controller_frame.tracking_servo_names)
            )
            first = _full12(
                controller_frame.drive_feedback_bias_full12,
                "controller drive_feedback_bias_full12",
            )
            second = _full12(
                controller_frame.normal_drive_bias_full12,
                "controller normal_drive_bias_full12",
            )
            biases.append(tuple(a + b for a, b in zip(first, second, strict=True)))
        ack = self.command_adapter.apply_batch(
            actions,
            physics_tick=self.global_physics_step_count,
            tracking_servo_names=tuple(tracking),
            drive_feedback_bias_full12=tuple(biases),
        )
        before_steps = self.global_physics_step_count
        self._advance_global_physics()
        if self.global_physics_step_count != before_steps + 1:
            raise VectorizedIsaacBackendError(
                "one backend tick did not produce exactly one global physics step"
            )

        next_tick = self._episode_tick + 1
        next_time = next_tick * PHYSICS_DT_S
        next_controller_frames = []
        next_frames = []
        for row, (reader, controller, semantic, previous) in enumerate(
            zip(
                self.readers,
                self.controllers,
                self._semantics,
                self._frames,
                strict=True,
            )
        ):
            observation = reader.read(
                physics_tick=next_tick,
                simulation_time_s=next_time,
                commanded_full12=ack.rows[row]["drive_target_full12"],
            )
            _validate_sensor_contract(observation, SENSED_BODIES, require_finite=False)
            controller_frame = controller.step(observation, sim_time_s=next_time)
            _validate_controller_clock(
                controller_frame, physics_tick=next_tick, sim_time_s=next_time
            )
            semantic._previous_action_full12 = actions[row]
            semantic._last_atomic_ack = ack.rows[row]
            semantic._raw_observation = observation
            semantic._controller_frame = controller_frame
            semantic._episode_tick = next_tick
            frame = semantic._build_authoritative_frame(
                observation, controller_frame, previous_frame=previous
            )
            semantic._authoritative_frame = frame
            semantic._done = _frame_is_terminal(frame)
            next_controller_frames.append(controller_frame)
            next_frames.append(frame)
        self._episode_tick = next_tick
        self._controller_frames = tuple(next_controller_frames)
        self._frames = tuple(next_frames)
        self._done = tuple(_frame_is_terminal(frame) for frame in next_frames)
        return self._batched_frame()

    def benchmark(self, *, measured_ticks: int = 16) -> VectorBenchmarkReport:
        """Measure live synchronous nominal-action throughput and attest batching."""

        ticks = int(measured_ticks)
        if ticks <= 0 or ticks != measured_ticks:
            raise VectorizedIsaacBackendError("measured_ticks must be a positive integer")
        if not self._frames:
            raise VectorizedIsaacBackendError("reset_all must precede benchmark")
        start_steps = self.global_physics_step_count
        start_writes = self.command_adapter.write_count if self.command_adapter else 0
        start_captures = self.contact_bank.capture_count
        started = time.perf_counter()
        completed = 0
        reasons: list[str] = []
        try:
            for _ in range(ticks):
                actions = tuple(frame.nominal_action_full12 for frame in self.frames)
                self.step_physics_batch(actions)
                completed += 1
        except Exception as exc:
            reasons.append(f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - started
        step_delta = self.global_physics_step_count - start_steps
        write_delta = (
            0
            if self.command_adapter is None
            else self.command_adapter.write_count - start_writes
        )
        capture_delta = self.contact_bank.capture_count - start_captures
        attestation_failures = self._attestation_failures(
            backend_ticks=completed,
            global_steps=step_delta,
            batched_writes=write_delta,
            pair_captures=capture_delta,
        )
        reasons.extend(attestation_failures)
        passed = completed == ticks and not reasons
        return VectorBenchmarkReport(
            status=(
                "TRUE_BATCHED_ISAAC_VERIFIED"
                if passed
                else "VECTOR_BACKEND_BENCHMARK_FAILED"
            ),
            num_envs=self.num_envs,
            measured_ticks=completed,
            wall_time_s=elapsed,
            physics_steps_per_second=completed / elapsed if elapsed > 0.0 else 0.0,
            environment_steps_per_second=(completed * self.num_envs / elapsed if elapsed > 0.0 else 0.0),
            one_simulation_context=True,
            articulation_tensor_instances=int(self.robot.data.joint_pos.shape[0]),
            global_physics_steps=step_delta,
            batched_articulation_writes=write_delta,
            exact_pair_captures=capture_delta,
            exact_pair_sensor_count=len(self.contact_bank.sensors),
            independent_controller_count=len({id(value) for value in self.controllers}),
            independent_reader_count=len({id(value) for value in self.readers}),
            final_state_ids=tuple(frame.state_id for frame in self._frames),
            true_batched_isaac_verified=passed,
            failure_reasons=tuple(reasons),
        )

    def _advance_global_physics(self) -> None:
        self._require_running("global physics advance")
        self.sim.step(render=False)
        self.scene.update(PHYSICS_DT_S)
        self.global_physics_step_count += 1
        self.contact_bank.capture(self.global_physics_step_count)

    def _restore_default_state(self) -> None:
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        self.scene.reset()
        self.robot.update(0.0)

    def _make_readers(self) -> tuple[SensorReader, ...]:
        result = []
        for row in range(self.num_envs):
            origin = self.scene.env_origins[row]
            row_robot = _RowRobot(self.robot, row, origin, self.num_envs)
            assert self.command_adapter is not None
            read_adapter = _RowReadAdapter(self.command_adapter, row, row_robot)
            provider = UsdCollisionBoundsProvider(
                self.scene.stage,
                robot_prim_path=f"{self.scene.env_prim_paths[row]}/WLRRobot",
            )
            result.append(
                SensorReader(
                    read_adapter,
                    contact_backend=self.contact_bank.row_backend(row),
                    geometry_backend=ColliderGeometryCache(provider),
                    physics_dt_s=PHYSICS_DT_S,
                )
            )
        return tuple(result)

    def _validate_physical_batch(self) -> None:
        if self.scene.sim is not self.sim:
            raise VectorizedIsaacBackendError(
                "InteractiveScene is not attached to the one backend SimulationContext"
            )
        if (
            int(self.scene.cfg.num_envs) != self.num_envs
            or not bool(self.scene.cfg.replicate_physics)
            or not bool(self.scene.cfg.filter_collisions)
        ):
            raise VectorizedIsaacBackendError(
                "scene is not a collision-isolated replicated-physics batch"
            )
        joint_shape = tuple(self.robot.data.joint_pos.shape)
        if len(joint_shape) != 2 or int(joint_shape[0]) != self.num_envs:
            raise VectorizedIsaacBackendError(
                f"robot articulation is not truly batched: joint_pos shape={joint_shape}"
            )
        origins = _numpy(self.scene.env_origins)
        if origins.shape != (self.num_envs, 3):
            raise VectorizedIsaacBackendError(
                f"cloned environment origins shape={origins.shape}, expected {(self.num_envs, 3)}"
            )
        if len({tuple(float(v) for v in row) for row in origins}) != self.num_envs:
            raise VectorizedIsaacBackendError("cloned environments do not have distinct origins")
        if len(tuple(self.scene.env_prim_paths)) != self.num_envs:
            raise VectorizedIsaacBackendError("InteractiveScene did not create N environment prim paths")

    def _assert_independent_python_state(self) -> None:
        groups = {
            "controllers": self.controllers,
            "readers": self.readers,
            "contact_classifiers": tuple(reader.contact_classifier for reader in self.readers),
            "guard_trackers": tuple(reader.guard_tracker for reader in self.readers),
            "body_collision_detectors": tuple(
                reader.body_collision_detector for reader in self.readers
            ),
        }
        for label, values in groups.items():
            if len(values) != self.num_envs or len({id(value) for value in values}) != self.num_envs:
                raise VectorizedIsaacBackendError(
                    f"{label} are shared across cloned environments"
                )

    def _attestation_failures(
        self,
        *,
        backend_ticks: int,
        global_steps: int,
        batched_writes: int,
        pair_captures: int,
    ) -> tuple[str, ...]:
        failures = []
        if int(self.robot.data.joint_pos.shape[0]) != self.num_envs:
            failures.append("articulation tensor leading dimension is not num_envs")
        if global_steps != backend_ticks:
            failures.append("global physics step count differs from backend tick count")
        if batched_writes != backend_ticks:
            failures.append("batched articulation write count differs from backend tick count")
        if pair_captures != backend_ticks:
            failures.append("exact-pair capture count differs from backend tick count")
        if len(self.contact_bank.sensors) != len(SENSED_BODIES):
            failures.append("exact-pair sensor bank is incomplete")
        if len({id(value) for value in self.controllers}) != self.num_envs:
            failures.append("FSM controller instances are shared")
        if len({id(value) for value in self.readers}) != self.num_envs:
            failures.append("sensing histories are shared")
        return tuple(failures)

    def _batched_frame(self) -> BatchedAuthoritativeFrame:
        return BatchedAuthoritativeFrame(
            physics_tick=self._episode_tick,
            sim_time_s=self._episode_tick * PHYSICS_DT_S,
            frames=self._frames,
            global_physics_step_count=self.global_physics_step_count,
            batched_articulation_write_count=(
                0 if self.command_adapter is None else self.command_adapter.write_count
            ),
            exact_pair_capture_count=self.contact_bank.capture_count,
        )

    def _reset_metadata(
        self,
        *,
        row: int,
        seed: int,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": "wlr50_clean.vectorized_isaac_backend.reset.v1",
            "environment_hash_source": "configs/environment_lock.json",
            "robot_asset_hash": ROBOT_USD_SHA256,
            "controller_path": str(self.fsm_path),
            "motion_contract_path": str(self.motion_contract_path),
            "seed": int(seed),
            "reset_options": dict(options),
            "reset_count": self._reset_count + 1,
            "env_index": row,
            "env_origin_w_m": list(_tensor_row(self.scene.env_origins[row])),
            "num_envs": self.num_envs,
            "shared_ground_visual_size_m": list(
                _shared_ground_size_m(self.num_envs, self.env_spacing_m)
            ),
            "physics_hz": 1.0 / PHYSICS_DT_S,
            "physics_dt_s": PHYSICS_DT_S,
            "settle_ticks": SETTLE_TICKS,
            "level_calibration_sample_count": LEVEL_CALIBRATION_TICKS,
            "one_global_physics_step_per_tick": True,
            "one_batched_articulation_write_per_tick": True,
            "exact_pair_contact_fail_closed": True,
            "independent_fsm_per_environment": True,
            "in_episode_root_pose_writes": 0,
            "in_episode_root_velocity_writes": 0,
            "in_episode_force_or_impulse_writes": 0,
            "in_episode_gravity_writes": 0,
            "recording_accesses": 0,
            "locked_scene_snapshot": locked_scene_snapshot(),
        }

    def _require_running(self, context: str) -> None:
        if not bool(self.simulation_app.is_running()):
            raise VectorizedIsaacBackendError(
                f"SimulationApp stopped during {context}"
            )


@dataclass(frozen=True, slots=True)
class _LiveModules:
    sim_utils: Any
    SimulationContext: Any
    SimulationCfg: Any
    InteractiveScene: Any
    InteractiveSceneCfg: Any
    AssetBaseCfg: Any
    ArticulationCfg: Any
    ImplicitActuatorCfg: Any
    ContactSensorCfg: Any


def _load_live_modules() -> _LiveModules:
    # AppLauncher must exist before this call.
    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # type: ignore
    from isaaclab.sensors import ContactSensorCfg  # type: ignore
    from isaaclab.sim import SimulationCfg, SimulationContext  # type: ignore

    return _LiveModules(
        sim_utils=sim_utils,
        SimulationContext=SimulationContext,
        SimulationCfg=SimulationCfg,
        InteractiveScene=InteractiveScene,
        InteractiveSceneCfg=InteractiveSceneCfg,
        AssetBaseCfg=AssetBaseCfg,
        ArticulationCfg=ArticulationCfg,
        ImplicitActuatorCfg=ImplicitActuatorCfg,
        ContactSensorCfg=ContactSensorCfg,
    )


def _create_vector_scene(
    modules: _LiveModules,
    *,
    num_envs: int,
    device: str,
    env_spacing_m: float,
) -> tuple[Any, Any]:
    sim_utils = modules.sim_utils
    sim = modules.SimulationContext(
        modules.SimulationCfg(
            dt=PHYSICS_DT_S,
            render_interval=RENDER_INTERVAL_PHYSICS_STEPS,
            device=device,
            gravity=GRAVITY_M_S2,
        )
    )
    cfg = modules.InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing_m,
        replicate_physics=True,
        filter_collisions=True,
        clone_in_fabric=False,
        lazy_sensor_update=False,
    )
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=GROUND_STATIC_FRICTION,
        dynamic_friction=GROUND_DYNAMIC_FRICTION,
        restitution=GROUND_RESTITUTION,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    cfg.ground = modules.AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=_shared_ground_size_m(num_envs, env_spacing_m),
            color=(0.08, 0.09, 0.10),
            physics_material=material,
        ),
        collision_group=-1,
    )
    base_robot_cfg = _build_robot_cfg(
        {
            "sim_utils": sim_utils,
            "ArticulationCfg": modules.ArticulationCfg,
            "ImplicitActuatorCfg": modules.ImplicitActuatorCfg,
        }
    )
    cfg.robot = base_robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/WLRRobot"
    )
    cfg.obstacle = modules.AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(OBSTACLE_LENGTH_M, OBSTACLE_WIDTH_M, OBSTACLE_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=OBSTACLE_CONTACT_OFFSET_M,
                rest_offset=OBSTACLE_REST_OFFSET_M,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=OBSTACLE_STATIC_FRICTION,
                dynamic_friction=OBSTACLE_DYNAMIC_FRICTION,
                restitution=OBSTACLE_RESTITUTION,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.47, 0.33)
            ),
            semantic_tags=[("class", "height_obstacle")],
        ),
        init_state=modules.AssetBaseCfg.InitialStateCfg(pos=OBSTACLE_CENTER_M),
    )
    for body_name in SENSED_BODIES:
        setattr(
            cfg,
            _sensor_key(body_name),
            modules.ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/WLRRobot/{body_name}",
                update_period=0.0,
                history_length=CONTACT_HISTORY_LENGTH,
                debug_vis=False,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                track_air_time=True,
                force_threshold=0.25,
                max_contact_data_count_per_prim=8,
                filter_prim_paths_expr=[
                    "{ENV_REGEX_NS}/Obstacle",
                    "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
                ],
            ),
        )
    cfg.light = modules.AssetBaseCfg(
        prim_path="/World/Light/Dome",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0, color=(0.85, 0.88, 0.95)
        ),
    )
    scene = modules.InteractiveScene(cfg)
    sim.reset()
    scene.reset()
    scene.update(0.0)
    actual_dt = float(sim.get_physics_dt())
    if not math.isclose(actual_dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise VectorizedIsaacBackendError(
            f"live physics dt={actual_dt}, expected {PHYSICS_DT_S}"
        )
    return sim, scene


def _sensor_key(body_name: str) -> str:
    return f"contact_{body_name}"


def _validated_env_count(value: int) -> int:
    if isinstance(value, bool):
        raise VectorizedIsaacBackendError(
            f"num_envs must be one of {SUPPORTED_VECTOR_ENV_COUNTS}"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise VectorizedIsaacBackendError(
            f"num_envs must be one of {SUPPORTED_VECTOR_ENV_COUNTS}"
        ) from exc
    if result != value or result not in SUPPORTED_VECTOR_ENV_COUNTS:
        raise VectorizedIsaacBackendError(
            f"num_envs must be one of {SUPPORTED_VECTOR_ENV_COUNTS}"
        )
    return result


def _full12(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise VectorizedIsaacBackendError(f"{label} must be numeric Full12") from exc
    if len(result) != len(FULL12_ORDER) or any(
        not math.isfinite(value) for value in result
    ):
        raise VectorizedIsaacBackendError(f"{label} must be finite Full12")
    return result


def _numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    return np.asarray(current, dtype=float)


def _tensor_row(value: Any) -> tuple[float, ...]:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    reshape = getattr(current, "reshape", None)
    if callable(reshape):
        current = reshape(-1)
    tolist = getattr(current, "tolist", None)
    raw = tolist() if callable(tolist) else list(current)
    result = tuple(float(item) for item in raw)
    if any(not math.isfinite(item) for item in result):
        raise VectorizedIsaacBackendError("live tensor row contains non-finite values")
    return result


def _like_tensor(values: Sequence[float], reference: Any) -> Any:
    import torch

    return torch.tensor(
        tuple(float(value) for value in values),
        dtype=reference.dtype,
        device=reference.device,
    )


def _vec3(values: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or any(not math.isfinite(value) for value in result):
        raise VectorizedExactPairFailure("exact-pair vector must contain three finite values")
    return result  # type: ignore[return-value]


__all__ = [
    "BatchedAuthoritativeFrame",
    "SUPPORTED_VECTOR_ENV_COUNTS",
    "VectorBackendProbe",
    "VectorBenchmarkReport",
    "VectorizedExactPairFailure",
    "VectorizedIsaacBackendError",
    "VectorizedIsaacFSMBackend",
    "probe_vectorized_isaac_backend",
]
