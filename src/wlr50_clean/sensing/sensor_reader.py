"""Atomic 120 Hz live-observation reader with lazy Isaac sensor creation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..infrastructure.command_batch import (
    FULL12_ORDER,
    PHYSICS_DT_S,
    SERVO_COMMAND_SIGN,
    SERVO_ORDER,
    WHEEL_ORDER,
)
from .body_collision_detector import BodyCollisionDetector
from .com_diagnostics import compute_full_body_com, compute_support_diagnostics
from .contact_classifier import (
    BASE_BODY,
    GROUND_PAIR,
    OBSTACLE_PAIR,
    SENSED_BODIES,
    ContactClassifier,
    RawPairContact,
)
from .geometry import (
    ColliderGeometryCache,
    GeometrySnapshot,
    UsdCollisionBoundsProvider,
    locked_obstacle_planes,
)
from .guard_state import LiveGuardTracker
from .observation import (
    ImuObservation,
    JointObservation,
    Observation,
    RigidBodyObservation,
    Vec3,
    WheelObservation,
)


OBSTACLE_PRIM_PATH = "/World/Obstacle"
GROUND_COLLISION_PRIM_PATH = "/World/defaultGroundPlane/GroundPlane/CollisionPlane"
CONTACT_FILTERS = (
    (OBSTACLE_PAIR, OBSTACLE_PRIM_PATH),
    (GROUND_PAIR, GROUND_COLLISION_PRIM_PATH),
)


class SensingContractError(RuntimeError):
    pass


class ContactBackend(Protocol):
    last_quality: tuple[str, ...]

    def sample(self, physics_dt_s: float) -> tuple[RawPairContact, ...]: ...


class GeometryBackend(Protocol):
    def sample(self, body_positions_w_m: Mapping[str, Sequence[float]]) -> GeometrySnapshot: ...


@dataclass(frozen=True, slots=True)
class LiveSensingBackends:
    """Instrumentation bundle created after prim spawn and before sim reset."""

    contact_backend: "ExactPairContactSensorBank"
    geometry_backend: ColliderGeometryCache


class ExactPairContactSensorBank:
    """One Isaac Lab ContactSensor for each exact robot rigid body."""

    def __init__(self, sensors: Mapping[str, Any]) -> None:
        if set(sensors) != set(SENSED_BODIES):
            raise SensingContractError("contact bank must contain exactly all 13 locked bodies")
        self.sensors = dict(sensors)
        self.last_quality: tuple[str, ...] = ()

    @classmethod
    def create(cls, *, imports: Mapping[str, Any] | None = None) -> "ExactPairContactSensorBank":
        if imports is None:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore
        else:
            ContactSensor = imports["ContactSensor"]
            ContactSensorCfg = imports["ContactSensorCfg"]
        sensors: dict[str, Any] = {}
        for body_name in SENSED_BODIES:
            cfg = ContactSensorCfg(
                prim_path=f"/World/WLRRobot/{body_name}",
                update_period=0.0,
                history_length=3,
                debug_vis=False,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                track_air_time=True,
                force_threshold=0.25,
                max_contact_data_count_per_prim=8,
                filter_prim_paths_expr=[path for _, path in CONTACT_FILTERS],
            )
            sensors[body_name] = ContactSensor(cfg)
        return cls(sensors)

    @property
    def initialized(self) -> bool:
        return all(bool(getattr(sensor, "is_initialized", False)) for sensor in self.sensors.values())

    def reset(self) -> None:
        for sensor in self.sensors.values():
            sensor.reset()

    def sample(self, physics_dt_s: float) -> tuple[RawPairContact, ...]:
        samples: list[RawPairContact] = []
        quality: list[str] = []
        for body_name, sensor in self.sensors.items():
            try:
                sensor.update(float(physics_dt_s), force_recompute=True)
                data = sensor.data
                live_names = tuple(str(name) for name in getattr(sensor, "body_names", ()))
                matrix = _array(getattr(data, "force_matrix_w", None))
                history = _array(getattr(data, "force_matrix_w_history", None))
                points = _array(getattr(data, "contact_pos_w", None))
                friction = _array(getattr(data, "friction_forces_w", None))
                pair_shape_ok = (
                    live_names == (body_name,)
                    and matrix.ndim == 4
                    and matrix.shape[0] == 1
                    and matrix.shape[1] == 1
                    and matrix.shape[2] == len(CONTACT_FILTERS)
                    and matrix.shape[3] == 3
                )
                if not pair_shape_ok:
                    quality.append(
                        f"{body_name} exact-pair matrix mismatch names={live_names} shape={matrix.shape}"
                    )
                for pair_index, (pair_kind, other_path) in enumerate(CONTACT_FILTERS):
                    force = _matrix_vec(matrix, (0, 0, pair_index), pair_shape_ok)
                    force_history = _history_vecs(history, pair_index, pair_shape_ok)
                    point = _matrix_optional_vec(points, (0, 0, pair_index), pair_shape_ok)
                    friction_force = _matrix_vec(friction, (0, 0, pair_index), pair_shape_ok)
                    samples.append(
                        RawPairContact(
                            sensor_body=body_name,
                            pair_kind=pair_kind,
                            other_body=other_path,
                            force_w_n=force,
                            friction_force_w_n=friction_force,
                            contact_point_w_m=point,
                            history_force_w_n=force_history,
                            source="isaaclab.ContactSensor.force_matrix_w",
                            pair_verified=pair_shape_ok,
                        )
                    )
            except Exception as exc:
                quality.append(f"{body_name} ContactSensor unavailable: {exc}")
                for pair_kind, other_path in CONTACT_FILTERS:
                    samples.append(
                        RawPairContact(
                            sensor_body=body_name,
                            pair_kind=pair_kind,
                            other_body=other_path,
                            force_w_n=(0.0, 0.0, 0.0),
                            source=f"ContactSensor error: {exc}",
                            pair_verified=False,
                        )
                    )
        self.last_quality = tuple(quality)
        return tuple(samples)


def create_live_sensing_backends(*, sim: Any, robot: Any) -> LiveSensingBackends:
    """Scene ``before_reset(sim, robot)`` hook for live sensing resources.

    At this point the robot, ground, and obstacle prims must exist and the
    timeline must not yet have performed its initializing reset.  Constructing
    the ContactSensors here lets their normal play/reset callbacks build the
    exact PhysX pair views.  Neither argument is mutated.
    """

    dt = float(sim.get_physics_dt())
    if not math.isclose(dt, PHYSICS_DT_S, abs_tol=1.0e-12, rel_tol=0.0):
        raise SensingContractError(f"live sensing hook requires 120 Hz, received dt={dt}")
    if robot is None:
        raise SensingContractError("live sensing hook requires the spawned articulation")
    return LiveSensingBackends(
        contact_backend=ExactPairContactSensorBank.create(),
        geometry_backend=ColliderGeometryCache(UsdCollisionBoundsProvider.from_current_stage()),
    )


class SensorReader:
    """Assemble one coherent observation after each 120 Hz physics step."""

    def __init__(
        self,
        adapter: Any,
        *,
        contact_backend: ContactBackend,
        geometry_backend: GeometryBackend,
        contact_classifier: ContactClassifier | None = None,
        body_collision_detector: BodyCollisionDetector | None = None,
        guard_tracker: LiveGuardTracker | None = None,
        physics_dt_s: float = PHYSICS_DT_S,
    ) -> None:
        self.adapter = adapter
        self.robot = getattr(adapter, "robot", None)
        if self.robot is None:
            raise SensingContractError("SensorReader requires an adapter with a live robot")
        self.contact_backend = contact_backend
        self.geometry_backend = geometry_backend
        self.contact_classifier = contact_classifier or ContactClassifier()
        self.body_collision_detector = body_collision_detector or BodyCollisionDetector()
        self.physics_dt_s = float(physics_dt_s)
        if not math.isclose(self.physics_dt_s, PHYSICS_DT_S, abs_tol=1.0e-12, rel_tol=0.0):
            raise SensingContractError("SensorReader requires the locked 120 Hz physics dt")
        self.guard_tracker = guard_tracker or LiveGuardTracker(physics_hz=1.0 / self.physics_dt_s)
        self._last_tick: int | None = None
        self._last_time_s: float | None = None

    @classmethod
    def from_live_scene(
        cls,
        scene_handle: Any,
        adapter: Any,
        *,
        backends: LiveSensingBackends | None = None,
    ) -> "SensorReader":
        """Create lazy-Isaac backends after AppLauncher and scene construction.

        The caller must reset/play the SimulationContext after constructing the
        contact sensors so their PhysX views initialize before the first read.
        """

        if backends is None:
            backends = create_live_sensing_backends(sim=scene_handle.sim, robot=scene_handle.robot)
        return cls(
            adapter,
            contact_backend=backends.contact_backend,
            geometry_backend=backends.geometry_backend,
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
        )

    def read(
        self,
        *,
        physics_tick: int,
        simulation_time_s: float,
        commanded_full12: Any,
    ) -> Observation:
        tick, sim_time = self._validate_clock(physics_tick, simulation_time_s)
        command = _full12(commanded_full12)
        actual = self.adapter.get_actual_state()
        actual_full12 = tuple(float(value) for value in actual.full12)
        if len(actual_full12) != len(FULL12_ORDER):
            raise SensingContractError("adapter readback is not a complete Full12 state")

        bodies, link_positions, com_positions, com_velocities, masses = _body_state(self.robot)
        if BASE_BODY not in bodies:
            raise SensingContractError("live base_link state is unavailable")
        geometry = self.geometry_backend.sample(link_positions)
        raw_contacts = self.contact_backend.sample(self.physics_dt_s)
        contacts = self.contact_classifier.classify(raw_contacts)
        center_of_mass = compute_full_body_com(
            body_positions_w_m=com_positions,
            body_velocities_w_m_s=com_velocities,
            body_masses_kg=masses,
        )
        support = compute_support_diagnostics(
            center_of_mass=center_of_mass,
            contacts=contacts,
            wheels=geometry.wheels,
        )
        body_collision = self.body_collision_detector.evaluate(
            contacts,
            base_obstacle_penetration_m=geometry.base_obstacle_penetration_m,
        )

        joints: dict[str, JointObservation] = {}
        servo_velocity = tuple(float(value) for value in actual.servo_velocity_rad_s)
        if len(servo_velocity) != len(SERVO_ORDER):
            raise SensingContractError("adapter servo velocity readback is incomplete")
        for index, name in enumerate(SERVO_ORDER):
            position = actual_full12[index]
            velocity = math.degrees(servo_velocity[index]) * float(SERVO_COMMAND_SIGN[name])
            joints[name] = JointObservation(
                name=name,
                position_deg=position,
                velocity_deg_s=velocity,
                command_deg=command[index],
                error_deg=command[index] - position,
            )

        wheels: dict[str, WheelObservation] = {}
        for index, name in enumerate(WHEEL_ORDER):
            shape = geometry.wheels[name]
            wheels[name] = WheelObservation(
                name=name,
                body_name=shape.body_name,
                velocity_rad_s=actual_full12[len(SERVO_ORDER) + index],
                command_rad_s=command[len(SERVO_ORDER) + index],
                center_w_m=shape.center_w_m,
                bottom_w_m=shape.bottom_w_m,
                geometry_source=shape.geometry_source,
                geometry_verified=shape.verified,
            )

        quality = list(getattr(self.contact_backend, "last_quality", ()))
        quality.extend(geometry.quality)
        if not center_of_mass.valid:
            quality.append(center_of_mass.reason)
        self._last_tick, self._last_time_s = tick, sim_time
        wheel_velocity = actual_full12[len(SERVO_ORDER) :]
        progress_vector = (
            *bodies[BASE_BODY].position_w_m,
            *actual_full12,
            *(
                coordinate
                for wheel in wheels.values()
                if wheel.center_w_m is not None
                for coordinate in wheel.center_w_m
            ),
        )
        observation = Observation(
            schema="wlr50_clean.live_observation.v1",
            physics_tick=tick,
            simulation_time_s=sim_time,
            physics_dt_s=self.physics_dt_s,
            joints=joints,
            wheels=wheels,
            contacts=contacts,
            bodies=bodies,
            base=bodies[BASE_BODY],
            imu=_imu(self.robot, bodies[BASE_BODY]),
            obstacle=locked_obstacle_planes(),
            center_of_mass=center_of_mass,
            support=support,
            body_collision=body_collision,
            actual_full12=actual_full12,
            commanded_full12=command,
            measured_wheel_velocity_rad_s=wheel_velocity,
            joint_positions_deg=actual_full12[: len(SERVO_ORDER)],
            root_position_w_m=bodies[BASE_BODY].position_w_m,
            progress_vector=progress_vector,
            data_quality=tuple(quality),
        )
        guards, recent_motion = self.guard_tracker.update(observation)
        all_finite = not bool(guards["non_finite_observation_or_command"]["passed"])
        return replace(
            observation,
            recent_joint_motion_deg=recent_motion,
            all_finite=all_finite,
            guards=guards,
        )

    def _validate_clock(self, physics_tick: int, simulation_time_s: float) -> tuple[int, float]:
        tick = int(physics_tick)
        sim_time = float(simulation_time_s)
        if tick < 0 or not math.isfinite(sim_time) or sim_time < 0.0:
            raise SensingContractError("physics clock must be finite and non-negative")
        if self._last_tick is not None and tick != self._last_tick + 1:
            raise SensingContractError(
                f"observation ticks must be contiguous: last={self._last_tick}, received={tick}"
            )
        if self._last_time_s is not None:
            elapsed = sim_time - self._last_time_s
            if not math.isclose(elapsed, self.physics_dt_s, abs_tol=2.0e-6, rel_tol=0.0):
                raise SensingContractError(
                    f"observation time delta must be 1/120 s; received {elapsed}"
                )
        return tick, sim_time


def _body_state(robot: Any) -> tuple[
    dict[str, RigidBodyObservation],
    dict[str, Vec3],
    dict[str, Vec3],
    dict[str, Vec3],
    dict[str, float],
]:
    names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    if set(names) != set(SENSED_BODIES):
        raise SensingContractError(f"live rigid-body names differ from locked 13-body model: {names}")
    data = robot.data
    link_pos = _env_matrix(getattr(data, "body_link_pos_w", None), 3)
    link_quat = _env_matrix(getattr(data, "body_link_quat_w", None), 4)
    link_lin = _env_matrix(getattr(data, "body_link_lin_vel_w", None), 3)
    link_ang = _env_matrix(getattr(data, "body_link_ang_vel_w", None), 3)
    com_pos = _env_matrix(getattr(data, "body_com_pos_w", None), 3)
    com_lin = _env_matrix(getattr(data, "body_com_lin_vel_w", None), 3)
    mass_values = _env_vector(getattr(data, "default_mass", None))
    if any(array.shape[0] != len(names) for array in (link_pos, link_quat, link_lin, link_ang)):
        raise SensingContractError("live rigid-body state dimensions are incomplete")
    bodies: dict[str, RigidBodyObservation] = {}
    positions: dict[str, Vec3] = {}
    com_positions: dict[str, Vec3] = {}
    com_velocities: dict[str, Vec3] = {}
    masses: dict[str, float] = {}
    for index, name in enumerate(names):
        position = _tuple(link_pos[index], 3)
        bodies[name] = RigidBodyObservation(
            name=name,
            position_w_m=position,
            orientation_wxyz=_tuple(link_quat[index], 4),  # type: ignore[arg-type]
            linear_velocity_w_m_s=_tuple(link_lin[index], 3),
            angular_velocity_w_rad_s=_tuple(link_ang[index], 3),
        )
        positions[name] = position
        if index < com_pos.shape[0] and index < com_lin.shape[0] and index < mass_values.shape[0]:
            com_positions[name] = _tuple(com_pos[index], 3)
            com_velocities[name] = _tuple(com_lin[index], 3)
            masses[name] = float(mass_values[index])
    return bodies, positions, com_positions, com_velocities, masses


def _imu(robot: Any, base: RigidBodyObservation) -> ImuObservation:
    data = robot.data
    angular_b = _first_vec(getattr(data, "root_ang_vel_b", None), 3)
    gravity_b = _first_vec(getattr(data, "projected_gravity_b", None), 3)
    names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    base_index = names.index(BASE_BODY)
    acceleration_w = _env_matrix(getattr(data, "body_com_lin_acc_w", None), 3)[base_index]
    acceleration_b = _quat_rotate_inverse(base.orientation_wxyz, _tuple(acceleration_w, 3))
    specific_force = tuple(acceleration_b[i] - gravity_b[i] * 9.81 for i in range(3))
    return ImuObservation(
        orientation_wxyz=base.orientation_wxyz,
        angular_velocity_b_rad_s=angular_b,
        linear_acceleration_b_m_s2=acceleration_b,
        specific_force_b_m_s2=specific_force,  # type: ignore[arg-type]
        projected_gravity_b=gravity_b,
        source="live articulation base-link kinematics",
    )


def _full12(command: Any) -> tuple[float, ...]:
    if hasattr(command, "to_full12"):
        command = command.to_full12()
    elif isinstance(command, Mapping):
        if set(command) != set(FULL12_ORDER):
            raise SensingContractError("command mapping must contain exactly all Full12 channels")
        command = [command[name] for name in FULL12_ORDER]
    values = tuple(float(value) for value in command)
    if len(values) != len(FULL12_ORDER) or not all(math.isfinite(value) for value in values):
        raise SensingContractError("command must contain 12 finite values")
    return values


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    return np.asarray(current, dtype=float)


def _env_matrix(value: Any, width: int) -> np.ndarray:
    array = _array(value)
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] >= width:
        return array[0, :, :width]
    return np.empty((0, width), dtype=float)


def _env_vector(value: Any) -> np.ndarray:
    array = _array(value)
    if array.ndim == 2 and array.shape[0] == 1:
        return array[0]
    return np.empty(0, dtype=float)


def _first_vec(value: Any, width: int) -> Vec3:
    array = _array(value)
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] < width:
        raise SensingContractError("base-frame IMU signal unavailable")
    return _tuple(array[0], width)  # type: ignore[return-value]


def _tuple(value: Sequence[float], width: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value[:width])
    if len(result) != width or not all(math.isfinite(item) for item in result):
        raise SensingContractError("live tensor contains incomplete or non-finite data")
    return result


def _matrix_vec(array: np.ndarray, prefix: tuple[int, ...], valid: bool) -> Vec3:
    if not valid or array.ndim != len(prefix) + 1:
        return (0.0, 0.0, 0.0)
    try:
        return _tuple(array[prefix], 3)  # type: ignore[return-value]
    except (IndexError, SensingContractError):
        return (0.0, 0.0, 0.0)


def _matrix_optional_vec(
    array: np.ndarray, prefix: tuple[int, ...], valid: bool
) -> Vec3 | None:
    if not valid or array.ndim != len(prefix) + 1:
        return None
    try:
        values = tuple(float(item) for item in array[prefix][:3])
    except (IndexError, TypeError):
        return None
    return values if len(values) == 3 and all(math.isfinite(item) for item in values) else None  # type: ignore[return-value]


def _history_vecs(array: np.ndarray, pair_index: int, valid: bool) -> tuple[Vec3, ...]:
    if not valid or array.ndim != 5 or array.shape[0] != 1 or array.shape[2] != 1:
        return ()
    try:
        return tuple(_tuple(row, 3) for row in array[0, :, 0, pair_index, :])  # type: ignore[return-value]
    except (IndexError, SensingContractError):
        return ()


def _quat_rotate_inverse(quat_wxyz: Sequence[float], vector: Sequence[float]) -> Vec3:
    w, x, y, z = (float(value) for value in quat_wxyz)
    vx, vy, vz = (float(value) for value in vector)
    # Rotate by q conjugate using the equivalent 3x3 matrix transpose.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx + 2.0 * (x * y + w * z) * vy + 2.0 * (x * z - w * y) * vz,
        2.0 * (x * y - w * z) * vx + (1.0 - 2.0 * (x * x + z * z)) * vy + 2.0 * (y * z + w * x) * vz,
        2.0 * (x * z + w * y) * vx + 2.0 * (y * z - w * x) * vy + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )
