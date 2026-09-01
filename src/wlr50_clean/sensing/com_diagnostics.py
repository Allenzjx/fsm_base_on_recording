"""Mass-weighted full-body center of mass and support diagnostics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .contact_classifier import SENSED_BODIES, WHEEL_BODIES
from .geometry import WheelGeometry
from .observation import (
    BodyContactObservation,
    CenterOfMassObservation,
    SupportDiagnostics,
    Vec3,
)


def compute_full_body_com(
    *,
    body_positions_w_m: Mapping[str, Sequence[float]],
    body_velocities_w_m_s: Mapping[str, Sequence[float]],
    body_masses_kg: Mapping[str, float],
    required_bodies: Sequence[str] = SENSED_BODIES,
) -> CenterOfMassObservation:
    """Compute CoM using every locked rigid body and no base-only fallback."""

    required = tuple(str(name) for name in required_bodies)
    missing = [
        name
        for name in required
        if name not in body_positions_w_m
        or name not in body_velocities_w_m_s
        or name not in body_masses_kg
    ]
    if missing:
        return _invalid_com(f"missing full-body data: {', '.join(missing)}")

    total_mass = 0.0
    weighted_position = [0.0, 0.0, 0.0]
    weighted_velocity = [0.0, 0.0, 0.0]
    for name in required:
        mass = float(body_masses_kg[name])
        position = _vec3(body_positions_w_m[name])
        velocity = _vec3(body_velocities_w_m_s[name])
        if mass <= 0.0 or not math.isfinite(mass) or position is None or velocity is None:
            return _invalid_com(f"invalid mass/CoM state for {name}")
        total_mass += mass
        for index in range(3):
            weighted_position[index] += mass * position[index]
            weighted_velocity[index] += mass * velocity[index]
    if total_mass <= 0.0 or not math.isfinite(total_mass):
        return _invalid_com("total mass is invalid")
    return CenterOfMassObservation(
        position_w_m=tuple(value / total_mass for value in weighted_position),  # type: ignore[arg-type]
        velocity_w_m_s=tuple(value / total_mass for value in weighted_velocity),  # type: ignore[arg-type]
        total_mass_kg=total_mass,
        included_bodies=required,
        source="articulation.body_com_pos_w weighted by articulation.data.default_mass",
        valid=True,
    )


def compute_support_diagnostics(
    *,
    center_of_mass: CenterOfMassObservation,
    contacts: Mapping[str, BodyContactObservation],
    wheels: Mapping[str, WheelGeometry],
) -> SupportDiagnostics:
    """Build the XY support hull from verified exact-pair contacts.

    A wheel bottom measured from its collider may locate an already verified
    contact if PhysX did not return a point.  Geometry alone never creates a
    support contact.
    """

    points: list[Vec3] = []
    active_bodies: list[str] = []
    point_sources: set[str] = set()
    wheel_by_body = {wheel.body_name: wheel for wheel in wheels.values()}
    for body_name, body_contact in contacts.items():
        active_pair = None
        if body_contact.obstacle.active and body_contact.obstacle.pair_verified:
            active_pair = body_contact.obstacle
        elif body_contact.ground.active and body_contact.ground.pair_verified:
            active_pair = body_contact.ground
        if active_pair is None:
            continue
        point = active_pair.contact_point_w_m
        if point is not None:
            point_sources.add("ContactSensor.contact_pos_w")
        elif body_name in WHEEL_BODIES:
            wheel = wheel_by_body.get(body_name)
            if wheel is not None and wheel.verified:
                point = wheel.bottom_w_m
                point_sources.add("cached_live_collider_bottom")
        if point is None or not all(math.isfinite(value) for value in point):
            continue
        active_bodies.append(body_name)
        points.append(point)

    points = _deduplicate_xy(points)
    if not center_of_mass.valid:
        return _invalid_support(points, active_bodies, "full-body CoM unavailable")
    if not points:
        return _invalid_support(points, active_bodies, "no verified support contact points")
    hull = _convex_hull([(point[0], point[1]) for point in points])
    projection = (center_of_mass.position_w_m[0], center_of_mass.position_w_m[1])
    if len(hull) < 3:
        return SupportDiagnostics(
            active_bodies=tuple(active_bodies),
            support_points_w_m=tuple(points),
            convex_hull_xy_m=tuple(hull),
            com_projection_xy_m=projection,
            signed_margin_m=None,
            projection_inside=None,
            support_count=len(points),
            source="+".join(sorted(point_sources)) or "unavailable",
            valid=False,
            reason="fewer than three non-collinear support points",
        )
    margin = _signed_convex_margin(projection, hull)
    return SupportDiagnostics(
        active_bodies=tuple(active_bodies),
        support_points_w_m=tuple(points),
        convex_hull_xy_m=tuple(hull),
        com_projection_xy_m=projection,
        signed_margin_m=margin,
        projection_inside=margin >= -1.0e-9,
        support_count=len(points),
        source="+".join(sorted(point_sources)),
        valid=True,
    )


def _invalid_com(reason: str) -> CenterOfMassObservation:
    return CenterOfMassObservation(
        position_w_m=(0.0, 0.0, 0.0),
        velocity_w_m_s=(0.0, 0.0, 0.0),
        total_mass_kg=0.0,
        included_bodies=(),
        source="unavailable",
        valid=False,
        reason=reason,
    )


def _invalid_support(
    points: Sequence[Vec3], active_bodies: Sequence[str], reason: str
) -> SupportDiagnostics:
    return SupportDiagnostics(
        active_bodies=tuple(active_bodies),
        support_points_w_m=tuple(points),
        convex_hull_xy_m=(),
        com_projection_xy_m=(0.0, 0.0),
        signed_margin_m=None,
        projection_inside=None,
        support_count=len(points),
        source="unavailable",
        valid=False,
        reason=reason,
    )


def _vec3(values: Sequence[float]) -> Vec3 | None:
    converted = tuple(float(value) for value in values)
    if len(converted) != 3 or not all(math.isfinite(value) for value in converted):
        return None
    return converted  # type: ignore[return-value]


def _deduplicate_xy(points: Sequence[Vec3], tolerance: float = 1.0e-7) -> list[Vec3]:
    result: list[Vec3] = []
    for point in points:
        if not any(math.hypot(point[0] - old[0], point[1] - old[1]) <= tolerance for old in result):
            result.append(point)
    return result


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _signed_convex_margin(
    point: tuple[float, float], hull_ccw: Sequence[tuple[float, float]]
) -> float:
    signed_distances: list[float] = []
    for index, start in enumerate(hull_ccw):
        end = hull_ccw[(index + 1) % len(hull_ccw)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        edge_length = math.hypot(dx, dy)
        signed_distances.append(
            ((dx * (point[1] - start[1])) - (dy * (point[0] - start[0])))
            / max(edge_length, 1.0e-12)
        )
    return min(signed_distances)

