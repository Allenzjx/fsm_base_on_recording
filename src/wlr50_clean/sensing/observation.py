"""Immutable live-state schema consumed by the sensor-driven FSM.

The schema contains measurements only.  It deliberately has no phase index,
playback cursor, or motion-script state, so a guard can only depend on live
physics and the command that was actually issued.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


class ContactClass(str, Enum):
    """Mutually exclusive contact state for one robot rigid body."""

    AIR = "AIR"
    GROUND = "GROUND"
    OBSTACLE = "OBSTACLE"
    GROUND_AND_OBSTACLE = "GROUND_AND_OBSTACLE"
    UNVERIFIED = "UNVERIFIED"


class CollisionRole(str, Enum):
    """Failure semantics assigned to exact rigid-body names."""

    BODY = "BODY"
    LEG = "LEG"
    WHEEL = "WHEEL"


@dataclass(frozen=True, slots=True)
class JointObservation:
    name: str
    position_deg: float
    velocity_deg_s: float
    command_deg: float
    error_deg: float


@dataclass(frozen=True, slots=True)
class WheelObservation:
    name: str
    body_name: str
    velocity_rad_s: float
    command_rad_s: float
    center_w_m: Vec3 | None
    bottom_w_m: Vec3 | None
    geometry_source: str
    geometry_verified: bool


@dataclass(frozen=True, slots=True)
class PairContactObservation:
    """An exact sensor-body/other-body pair, never an aggregate net force."""

    sensor_body: str
    other_body: str
    active: bool
    force_w_n: Vec3
    normal_force_n: float
    tangential_force_n: float
    contact_point_w_m: Vec3 | None
    force_history_w_n: tuple[Vec3, ...]
    active_history: tuple[bool, ...]
    consecutive_active_ticks: int
    source: str
    pair_verified: bool


@dataclass(frozen=True, slots=True)
class BodyContactObservation:
    body_name: str
    role: CollisionRole
    contact_class: ContactClass
    ground: PairContactObservation
    obstacle: PairContactObservation


@dataclass(frozen=True, slots=True)
class RigidBodyObservation:
    name: str
    position_w_m: Vec3
    orientation_wxyz: QuatWxyz
    linear_velocity_w_m_s: Vec3
    angular_velocity_w_rad_s: Vec3


@dataclass(frozen=True, slots=True)
class ImuObservation:
    """Base-link IMU-equivalent signals derived from live rigid-body state."""

    orientation_wxyz: QuatWxyz
    angular_velocity_b_rad_s: Vec3
    linear_acceleration_b_m_s2: Vec3
    specific_force_b_m_s2: Vec3
    projected_gravity_b: Vec3
    source: str


@dataclass(frozen=True, slots=True)
class ObstaclePlanes:
    front_x_m: float
    back_x_m: float
    left_y_m: float
    right_y_m: float
    bottom_z_m: float
    top_z_m: float


@dataclass(frozen=True, slots=True)
class CenterOfMassObservation:
    position_w_m: Vec3
    velocity_w_m_s: Vec3
    total_mass_kg: float
    included_bodies: tuple[str, ...]
    source: str
    valid: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SupportDiagnostics:
    active_bodies: tuple[str, ...]
    support_points_w_m: tuple[Vec3, ...]
    convex_hull_xy_m: tuple[tuple[float, float], ...]
    com_projection_xy_m: tuple[float, float]
    signed_margin_m: float | None
    projection_inside: bool | None
    support_count: int
    source: str
    valid: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BodyCollisionStatus:
    detected: bool
    real_pair_active: bool
    persistent: bool
    geometry_penetration_m: float
    reason: str


@dataclass(frozen=True, slots=True)
class Observation:
    """One atomic 120 Hz observation sampled after the physics step."""

    schema: str
    physics_tick: int
    simulation_time_s: float
    physics_dt_s: float
    joints: Mapping[str, JointObservation]
    wheels: Mapping[str, WheelObservation]
    contacts: Mapping[str, BodyContactObservation]
    bodies: Mapping[str, RigidBodyObservation]
    base: RigidBodyObservation
    imu: ImuObservation
    obstacle: ObstaclePlanes
    center_of_mass: CenterOfMassObservation
    support: SupportDiagnostics
    body_collision: BodyCollisionStatus
    actual_full12: tuple[float, ...] = field(default_factory=tuple)
    commanded_full12: tuple[float, ...] = field(default_factory=tuple)
    measured_wheel_velocity_rad_s: tuple[float, ...] = field(default_factory=tuple)
    joint_positions_deg: tuple[float, ...] = field(default_factory=tuple)
    root_position_w_m: Vec3 = (0.0, 0.0, 0.0)
    progress_vector: tuple[float, ...] = field(default_factory=tuple)
    recent_joint_motion_deg: Mapping[str, float] = field(default_factory=dict)
    all_finite: bool = True
    guards: Mapping[str, Any] = field(default_factory=dict)
    data_quality: tuple[str, ...] = field(default_factory=tuple)

    def contact(self, body_name: str) -> BodyContactObservation:
        return self.contacts[body_name]

    def wheel(self, joint_name: str) -> WheelObservation:
        return self.wheels[joint_name]

    def resolve_guard(self, name: str, parameters: Mapping[str, Any]) -> Any:
        """Resolve authored guard names from live or latched sensor evidence."""

        leg = parameters.get("leg")
        if leg is not None:
            keyed = f"{name}:{leg}"
            if keyed in self.guards:
                return self.guards[keyed]
        if name in self.guards:
            return self.guards[name]
        if name in ("reference_entry_compatible", "final_joint_pose_compatible"):
            relative_limit = float(parameters.get("relative_limit", 0.15))
            errors = {
                joint_name: abs(self.joints[joint_name].error_deg)
                for joint_name in self.joints
            }
            tolerances = {
                joint_name: max(4.0, abs(self.joints[joint_name].command_deg) * relative_limit)
                for joint_name in self.joints
            }
            passed = all(errors[name_] <= tolerances[name_] for name_ in errors)
            return {
                "passed": passed,
                "value": {"errors_deg": errors, "tolerances_deg": tolerances},
                "source": "live joint readback vs currently held phase-boundary command",
                "reason": "servo endpoint tracking must remain within the live relative envelope",
            }
        if name == "reference_like_active_joint_change":
            active = tuple(str(item) for item in parameters.get("active_joints", ()))
            values = {joint: float(self.recent_joint_motion_deg.get(joint, 0.0)) for joint in active}
            passed = bool(active) and all(value >= 1.0 for value in values.values())
            return {
                "passed": passed,
                "value": values,
                "source": "rolling 0.5 s live joint-position variation",
                "reason": "every authored active joint must show measured motion",
            }
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready diagnostic snapshot."""

        return asdict(self)
