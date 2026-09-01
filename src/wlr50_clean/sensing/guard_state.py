"""Live-only traversal event latches and guard evidence."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from ..infrastructure.command_batch import KNEE_NAMES, SERVO_ORDER
from .observation import Observation, WheelObservation


LEG_TO_WHEEL = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}
LEG_TO_JOINTS = {
    "FL": ("front_left_hip", "front_left_knee"),
    "FR": ("front_right_hip", "front_right_knee"),
    "RL": ("rear_left_hip", "rear_left_knee"),
    "RR": ("rear_right_hip", "rear_right_knee"),
}


class LiveGuardTracker:
    """Update sensor latches once per physics observation.

    Latches record measured active lift, wheel front-plane crossing, and
    obstacle-top loading.  They contain no phase numbers, schedule times, or
    pre-authored event indices.
    """

    def __init__(self, *, physics_hz: float = 120.0) -> None:
        self.physics_hz = float(physics_hz)
        self._previous_joint_deg: dict[str, float] = {}
        window = max(3, int(round(0.5 * self.physics_hz)))
        self._recent_joint_delta = {name: deque(maxlen=window) for name in SERVO_ORDER}
        self._recent_wheel_bottom_z = {
            leg: deque(maxlen=window) for leg in LEG_TO_WHEEL
        }
        self._recent_air = {leg: deque(maxlen=window) for leg in LEG_TO_WHEEL}
        self._joint_total_motion_deg = {name: 0.0 for name in SERVO_ORDER}
        self._wheel_min_bottom_z: dict[str, float] = {}
        self._wheel_max_bottom_z: dict[str, float] = {}
        self._air_seen = {leg: False for leg in LEG_TO_WHEEL}
        self._active_lift = {leg: False for leg in LEG_TO_WHEEL}
        self._front_crossed = {leg: False for leg in LEG_TO_WHEEL}
        self._top_loaded = {leg: False for leg in LEG_TO_WHEEL}
        self._active_lift_tick: dict[str, int] = {}
        self._front_crossed_tick: dict[str, int] = {}
        self._top_loaded_tick: dict[str, int] = {}

    def update(self, observation: Observation) -> tuple[dict[str, Any], dict[str, float]]:
        recent_motion = self._update_joint_motion(observation)
        self._update_leg_latches(observation)
        guards = self._build_guards(observation)
        return guards, recent_motion

    def _update_joint_motion(self, observation: Observation) -> dict[str, float]:
        for name, joint in observation.joints.items():
            previous = self._previous_joint_deg.get(name, joint.position_deg)
            delta = abs(joint.position_deg - previous)
            self._previous_joint_deg[name] = joint.position_deg
            self._recent_joint_delta[name].append(delta)
            self._joint_total_motion_deg[name] += delta
        return {name: sum(values) for name, values in self._recent_joint_delta.items()}

    def _update_leg_latches(self, observation: Observation) -> None:
        for leg, wheel_name in LEG_TO_WHEEL.items():
            wheel = observation.wheels[wheel_name]
            contact = observation.contacts[wheel.body_name]
            pair_verified = contact.ground.pair_verified and contact.obstacle.pair_verified
            air_now = bool(pair_verified and not contact.ground.active and not contact.obstacle.active)
            self._recent_air[leg].append(air_now)
            if pair_verified:
                self._air_seen[leg] |= air_now
            if wheel.geometry_verified and wheel.bottom_w_m is not None:
                bottom_z = wheel.bottom_w_m[2]
                self._recent_wheel_bottom_z[leg].append(bottom_z)
                self._wheel_min_bottom_z[leg] = min(
                    bottom_z, self._wheel_min_bottom_z.get(leg, bottom_z)
                )
                self._wheel_max_bottom_z[leg] = max(
                    bottom_z, self._wheel_max_bottom_z.get(leg, bottom_z)
                )
            recent_bottom = self._recent_wheel_bottom_z[leg]
            recent_gain = (
                max(recent_bottom) - min(recent_bottom)
                if len(recent_bottom) >= 2
                else 0.0
            )
            recent_leg_motion = sum(
                sum(self._recent_joint_delta[name]) for name in LEG_TO_JOINTS[leg]
            )
            near_front = bool(
                wheel.geometry_verified
                and wheel.center_w_m is not None
                and -0.25
                <= wheel.center_w_m[0] - observation.obstacle.front_x_m
                <= 0.15
            )
            active_lift_now = bool(
                near_front
                and any(self._recent_air[leg])
                and recent_gain >= 0.008
                and recent_leg_motion >= 2.0
            )
            if active_lift_now and not self._active_lift[leg]:
                self._active_lift[leg] = True
                self._active_lift_tick[leg] = observation.physics_tick
            if wheel.geometry_verified and wheel.center_w_m is not None:
                crossed_now = wheel.center_w_m[0] >= observation.obstacle.front_x_m
                if crossed_now and not self._front_crossed[leg]:
                    self._front_crossed[leg] = True
                    self._front_crossed_tick[leg] = observation.physics_tick
            top_geometry = self._top_geometry(wheel, observation)
            top_loaded_now = bool(
                self._front_crossed[leg]
                and top_geometry
                and contact.obstacle.pair_verified
                and contact.obstacle.active
                and contact.obstacle.consecutive_active_ticks >= 2
            )
            if top_loaded_now and not self._top_loaded[leg]:
                self._top_loaded[leg] = True
                self._top_loaded_tick[leg] = observation.physics_tick

    def _build_guards(self, observation: Observation) -> dict[str, Any]:
        all_finite = _all_finite(observation)
        joint_violation, joint_values = _joint_limit_violation(observation)
        physics_abort, physics_values = _physics_explosion_or_fall(observation)
        wheel_only_legs = tuple(
            leg for leg in LEG_TO_WHEEL if self._front_crossed[leg] and not self._active_lift[leg]
        )
        guards: dict[str, Any] = {
            "no_body_obstacle_collision": _evidence(
                not observation.body_collision.detected,
                observation.body_collision.reason,
                "exact base-link pair and live collider corroboration",
            ),
            "body_collision_persistent_or_penetrating": _evidence(
                observation.body_collision.detected,
                observation.body_collision.reason,
                "exact base-link pair and live collider corroboration",
            ),
            "joint_hard_limits_valid": _evidence(
                not joint_violation, joint_values, "live logical servo readback"
            ),
            "joint_hard_limit_violation": _evidence(
                joint_violation, joint_values, "live logical servo readback"
            ),
            "critical_actuators_available": _evidence(
                all_finite and len(observation.joints) == 8 and len(observation.wheels) == 4,
                {"joint_count": len(observation.joints), "wheel_count": len(observation.wheels)},
                "complete finite articulation readback",
            ),
            "non_finite_observation_or_command": _evidence(
                not all_finite, {"all_finite": all_finite}, "live observation finiteness scan"
            ),
            "physics_explosion_or_fall": _evidence(
                physics_abort, physics_values, "base pose, gravity projection, and velocity"
            ),
            "wheel_only_climb_detected": _evidence(
                bool(wheel_only_legs),
                {
                    "unlifted_crossed_legs": wheel_only_legs,
                    "front_crossed_ticks": dict(self._front_crossed_tick),
                    "active_lift_ticks": dict(self._active_lift_tick),
                },
                "front-plane latch cross-checked against prior/same-sample active-lift evidence",
            ),
            "fr_lift_entry_geometry": _evidence(
                self._lift_entry_geometry("FR", observation),
                self._wheel_values("FR", observation),
                "live FR collider bottom/front-plane geometry and exact AIR pairs",
            ),
            "fl_lift_workspace_geometry": _evidence(
                self._wheel_in_front_band("FL", observation, -0.18, 0.04)
                and self._top_loaded["FR"],
                self._wheel_values("FL", observation),
                "live FL front-plane workspace with FR top-load latch",
            ),
            "rear_pair_pre_edge_geometry": _evidence(
                all(
                    self._wheel_in_front_band(leg, observation, -0.20, 0.03)
                    for leg in ("RL", "RR")
                ),
                {leg: self._wheel_values(leg, observation) for leg in ("RL", "RR")},
                "live rear-wheel collider centers relative to obstacle front plane",
            ),
            "rear_entry_alignment": _evidence(
                all(
                    self._wheel_in_front_band(leg, observation, -0.18, 0.04)
                    for leg in ("RL", "RR")
                ),
                {leg: self._wheel_values(leg, observation) for leg in ("RL", "RR")},
                "live rear-pair front-plane alignment",
            ),
            "rr_unload_compatible_geometry": _evidence(
                self._wheel_in_front_band("RR", observation, -0.13, 0.04)
                and self._top_loaded["FR"],
                self._wheel_values("RR", observation),
                "live RR workspace with retained front support latch",
            ),
            "rl_workspace_geometry": _evidence(
                self._wheel_in_front_band("RL", observation, -0.14, 0.04)
                and self._top_loaded["RR"],
                self._wheel_values("RL", observation),
                "live RL workspace with RR top-load latch",
            ),
            "rl_unload_entry_geometry": _evidence(
                self._wheel_in_front_band("RL", observation, -0.14, 0.05)
                and self._air_seen["RL"],
                self._wheel_values("RL", observation),
                "live RL workspace and exact-pair AIR history",
            ),
            "all_leg_front_face_crossings_latched": _evidence(
                all(self._front_crossed.values()),
                dict(self._front_crossed),
                "live wheel-center/front-plane crossing latches",
            ),
            "all_wheels_final_top_geometry": _evidence(
                all(self._final_top_geometry(leg, observation) for leg in LEG_TO_WHEEL),
                {leg: self._wheel_values(leg, observation) for leg in LEG_TO_WHEEL},
                "live verified collider bottoms within obstacle-top band",
            ),
        }
        for leg in LEG_TO_WHEEL:
            guards[f"reference_like_active_lift:{leg}"] = _evidence(
                self._active_lift[leg],
                {
                    "latched": self._active_lift[leg],
                    "latch_tick": self._active_lift_tick.get(leg),
                    "clearance_gain_m": self._clearance_gain(leg),
                    "recent_clearance_gain_m": self._recent_clearance_gain(leg),
                    "air_seen": self._air_seen[leg],
                    "recent_air_seen": any(self._recent_air[leg]),
                    "near_obstacle_front": self._near_front(leg, observation),
                    "joint_motion_deg": {
                        name: self._joint_total_motion_deg[name] for name in LEG_TO_JOINTS[leg]
                    },
                    "recent_joint_motion_deg": {
                        name: sum(self._recent_joint_delta[name])
                        for name in LEG_TO_JOINTS[leg]
                    },
                },
                "measured joint response plus wheel-bottom rise and exact-pair AIR history",
            )
            guards[f"wheel_clearance_gain_or_air_history:{leg}"] = _evidence(
                self._clearance_gain(leg) >= 0.008 or self._air_seen[leg],
                {"clearance_gain_m": self._clearance_gain(leg), "air_seen": self._air_seen[leg]},
                "live collider-bottom history and exact contact history",
            )
            guards[f"leg_front_face_crossed_latched:{leg}"] = _evidence(
                self._front_crossed[leg],
                {
                    "latched": self._front_crossed[leg],
                    "latch_tick": self._front_crossed_tick.get(leg),
                },
                "live wheel-center/front-plane latch",
            )
            guards[f"leg_top_loaded_latched:{leg}"] = _evidence(
                self._top_loaded[leg],
                {
                    "latched": self._top_loaded[leg],
                    "latch_tick": self._top_loaded_tick.get(leg),
                },
                "persistent exact wheel/obstacle pair plus top geometry",
            )
        return guards

    def _clearance_gain(self, leg: str) -> float:
        if leg not in self._wheel_min_bottom_z or leg not in self._wheel_max_bottom_z:
            return 0.0
        return self._wheel_max_bottom_z[leg] - self._wheel_min_bottom_z[leg]

    def _recent_clearance_gain(self, leg: str) -> float:
        values = self._recent_wheel_bottom_z[leg]
        return max(values) - min(values) if len(values) >= 2 else 0.0

    def _near_front(self, leg: str, observation: Observation) -> bool:
        wheel = observation.wheels[LEG_TO_WHEEL[leg]]
        return bool(
            wheel.geometry_verified
            and wheel.center_w_m is not None
            and -0.25
            <= wheel.center_w_m[0] - observation.obstacle.front_x_m
            <= 0.15
        )

    def _wheel_in_front_band(
        self, leg: str, observation: Observation, minimum_m: float, maximum_m: float
    ) -> bool:
        wheel = observation.wheels[LEG_TO_WHEEL[leg]]
        if not wheel.geometry_verified or wheel.center_w_m is None:
            return False
        clearance = wheel.center_w_m[0] - observation.obstacle.front_x_m
        return minimum_m <= clearance <= maximum_m

    def _lift_entry_geometry(self, leg: str, observation: Observation) -> bool:
        wheel = observation.wheels[LEG_TO_WHEEL[leg]]
        if not wheel.geometry_verified or wheel.center_w_m is None or wheel.bottom_w_m is None:
            return False
        contact = observation.contacts[wheel.body_name]
        clearance = wheel.center_w_m[0] - observation.obstacle.front_x_m
        return bool(
            -0.16 <= clearance <= 0.04
            and wheel.bottom_w_m[2] >= observation.obstacle.top_z_m + 0.02
            and contact.ground.pair_verified
            and contact.obstacle.pair_verified
            and not contact.ground.active
            and not contact.obstacle.active
        )

    def _top_geometry(self, wheel: WheelObservation, observation: Observation) -> bool:
        if not wheel.geometry_verified or wheel.bottom_w_m is None:
            return False
        gap = wheel.bottom_w_m[2] - observation.obstacle.top_z_m
        return -0.015 <= gap <= 0.025

    def _final_top_geometry(self, leg: str, observation: Observation) -> bool:
        wheel = observation.wheels[LEG_TO_WHEEL[leg]]
        return bool(
            wheel.geometry_verified
            and wheel.center_w_m is not None
            and wheel.center_w_m[0] >= observation.obstacle.front_x_m
            and self._top_geometry(wheel, observation)
        )

    def _wheel_values(self, leg: str, observation: Observation) -> dict[str, Any]:
        wheel = observation.wheels[LEG_TO_WHEEL[leg]]
        return {
            "verified": wheel.geometry_verified,
            "center_w_m": wheel.center_w_m,
            "bottom_w_m": wheel.bottom_w_m,
            "front_x_m": observation.obstacle.front_x_m,
            "top_z_m": observation.obstacle.top_z_m,
        }


def _evidence(passed: bool, value: Any, source: str, reason: str = "") -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "value": value,
        "source": source,
        "reason": reason,
    }


def _all_finite(observation: Observation) -> bool:
    values = [
        *observation.actual_full12,
        *observation.commanded_full12,
        *observation.base.position_w_m,
        *observation.base.linear_velocity_w_m_s,
        *observation.base.angular_velocity_w_rad_s,
        *observation.imu.linear_acceleration_b_m_s2,
        *observation.imu.angular_velocity_b_rad_s,
    ]
    return len(observation.actual_full12) == 12 and all(math.isfinite(value) for value in values)


def _joint_limit_violation(observation: Observation) -> tuple[bool, dict[str, Any]]:
    values: dict[str, Any] = {}
    violation = False
    for name, joint in observation.joints.items():
        limits = (-60.0, 210.0) if name in KNEE_NAMES else (-135.0, 135.0)
        outside = not (limits[0] <= joint.position_deg <= limits[1])
        values[name] = {"position_deg": joint.position_deg, "limits_deg": limits, "outside": outside}
        violation |= outside
    return violation, values


def _physics_explosion_or_fall(observation: Observation) -> tuple[bool, dict[str, float]]:
    base = observation.base
    linear_speed = math.sqrt(sum(value * value for value in base.linear_velocity_w_m_s))
    angular_speed = math.sqrt(sum(value * value for value in base.angular_velocity_w_rad_s))
    gravity_z = observation.imu.projected_gravity_b[2]
    values = {
        "base_z_m": base.position_w_m[2],
        "linear_speed_m_s": linear_speed,
        "angular_speed_rad_s": angular_speed,
        "projected_gravity_b_z": gravity_z,
    }
    abort = bool(
        base.position_w_m[2] < 0.015
        or base.position_w_m[2] > 1.0
        or linear_speed > 5.0
        or angular_speed > 20.0
        or gravity_z > -0.30
    )
    return abort, values
