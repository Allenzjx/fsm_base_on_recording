"""Live wheel/collider geometry and locked obstacle planes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .contact_classifier import BASE_BODY, SENSED_BODIES, WHEEL_BODIES
from .observation import ObstaclePlanes, Vec3


WHEEL_JOINT_TO_BODY = {
    "front_left_ankle": "front_left_wheel",
    "front_right_ankle": "front_right_wheel",
    "rear_left_ankle": "rear_left_wheel",
    "rear_right_ankle": "rear_right_wheel",
}


@dataclass(frozen=True, slots=True)
class Aabb:
    minimum_m: Vec3
    maximum_m: Vec3

    def __post_init__(self) -> None:
        if any(self.minimum_m[index] > self.maximum_m[index] for index in range(3)):
            raise ValueError("AABB minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class WheelGeometry:
    joint_name: str
    body_name: str
    center_w_m: Vec3 | None
    bottom_w_m: Vec3 | None
    collider_bounds_w_m: Aabb | None
    geometry_source: str
    verified: bool


@dataclass(frozen=True, slots=True)
class GeometrySnapshot:
    wheels: Mapping[str, WheelGeometry]
    body_bounds_w_m: Mapping[str, Aabb]
    base_obstacle_penetration_m: float
    quality: tuple[str, ...] = ()


class BoundsProvider(Protocol):
    def collision_bounds(
        self,
        body_name: str,
        *,
        body_position_w_m: Sequence[float] | None = None,
        body_orientation_wxyz: Sequence[float] | None = None,
    ) -> tuple[Aabb | None, tuple[str, ...]]: ...


@dataclass(frozen=True, slots=True)
class _MeasuredShape:
    min_offset_m: Vec3
    max_offset_m: Vec3
    collider_paths: tuple[str, ...]


def locked_obstacle_planes(
    *,
    front_x_m: float = 0.5213121737735307,
    length_m: float = 2.057375557085507,
    center_y_m: float = 0.0,
    width_m: float = 2.0,
    bottom_z_m: float = 0.0,
    height_m: float = 0.05,
) -> ObstaclePlanes:
    return ObstaclePlanes(
        front_x_m=float(front_x_m),
        back_x_m=float(front_x_m + length_m),
        left_y_m=float(center_y_m + 0.5 * width_m),
        right_y_m=float(center_y_m - 0.5 * width_m),
        bottom_z_m=float(bottom_z_m),
        top_z_m=float(bottom_z_m + height_m),
    )


def obstacle_aabb(planes: ObstaclePlanes) -> Aabb:
    return Aabb(
        minimum_m=(planes.front_x_m, planes.right_y_m, planes.bottom_z_m),
        maximum_m=(planes.back_x_m, planes.left_y_m, planes.top_z_m),
    )


class ColliderGeometryCache:
    """Cache extents measured from the live collider tree, then translate them.

    Wheels never fall back to a nominal radius.  If a live collision bound
    cannot be established, its bottom remains unavailable and guards can mark
    the geometry unverified instead of silently inventing it.
    """

    def __init__(
        self,
        bounds_provider: BoundsProvider,
        *,
        obstacle: ObstaclePlanes | None = None,
    ) -> None:
        self.bounds_provider = bounds_provider
        self.obstacle = obstacle or locked_obstacle_planes()
        self._shapes: dict[str, _MeasuredShape] = {}
        self._failures: dict[str, str] = {}

    def sample(
        self,
        body_positions_w_m: Mapping[str, Sequence[float]],
        body_orientations_wxyz: Mapping[str, Sequence[float]] | None = None,
    ) -> GeometrySnapshot:
        body_bounds: dict[str, Aabb] = {}
        quality: list[str] = []
        orientations = body_orientations_wxyz or {}
        for body_name in (BASE_BODY, *WHEEL_BODIES):
            center = _optional_vec3(body_positions_w_m.get(body_name))
            if center is None:
                quality.append(f"missing live center for {body_name}")
                continue
            shape = self._shapes.get(body_name)
            if shape is None and body_name not in self._failures:
                bounds, paths = self.bounds_provider.collision_bounds(
                    body_name,
                    body_position_w_m=center,
                    body_orientation_wxyz=orientations.get(body_name),
                )
                if bounds is None:
                    self._failures[body_name] = "no finite enabled collider bound"
                else:
                    bounds_center = tuple(
                        0.5 * (bounds.minimum_m[index] + bounds.maximum_m[index])
                        for index in range(3)
                    )
                    # X/Y offsets are measured as well; this retains an
                    # asymmetric collider's placement relative to its body.
                    min_offset = tuple(bounds.minimum_m[index] - center[index] for index in range(3))
                    max_offset = tuple(bounds.maximum_m[index] - center[index] for index in range(3))
                    if all(math.isfinite(value) for value in (*min_offset, *max_offset, *bounds_center)):
                        shape = _MeasuredShape(
                            min_offset_m=min_offset,  # type: ignore[arg-type]
                            max_offset_m=max_offset,  # type: ignore[arg-type]
                            collider_paths=tuple(paths),
                        )
                        self._shapes[body_name] = shape
                    else:
                        self._failures[body_name] = "non-finite live collider bound"
            if shape is None:
                quality.append(f"unverified collider geometry for {body_name}")
                continue
            translated = Aabb(
                minimum_m=tuple(center[i] + shape.min_offset_m[i] for i in range(3)),  # type: ignore[arg-type]
                maximum_m=tuple(center[i] + shape.max_offset_m[i] for i in range(3)),  # type: ignore[arg-type]
            )
            body_bounds[body_name] = translated

        wheels: dict[str, WheelGeometry] = {}
        for joint_name, body_name in WHEEL_JOINT_TO_BODY.items():
            center = _optional_vec3(body_positions_w_m.get(body_name))
            bounds = body_bounds.get(body_name)
            bottom = None
            source = "unavailable_no_live_collider_bound"
            verified = False
            if center is not None and bounds is not None:
                bottom = (center[0], center[1], bounds.minimum_m[2])
                shape = self._shapes[body_name]
                source = "cached_live_collider_extent:" + ",".join(shape.collider_paths)
                verified = True
            wheels[joint_name] = WheelGeometry(
                joint_name=joint_name,
                body_name=body_name,
                center_w_m=center,
                bottom_w_m=bottom,
                collider_bounds_w_m=bounds,
                geometry_source=source,
                verified=verified,
            )

        penetration = aabb_intersection_depth(body_bounds.get(BASE_BODY), obstacle_aabb(self.obstacle))
        return GeometrySnapshot(
            wheels=wheels,
            body_bounds_w_m=body_bounds,
            base_obstacle_penetration_m=penetration,
            quality=tuple(quality),
        )


class UsdCollisionBoundsProvider:
    """Read-only USD collider-bound provider; construct only after AppLauncher."""

    def __init__(self, stage: Any, *, robot_prim_path: str = "/World/WLRRobot") -> None:
        self.stage = stage
        self.robot_prim_path = robot_prim_path.rstrip("/")
        self._body_local_points: dict[str, tuple[Vec3, ...]] = {}
        self._collider_paths: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_current_stage(cls, *, robot_prim_path: str = "/World/WLRRobot") -> "UsdCollisionBoundsProvider":
        import omni.usd  # type: ignore

        return cls(omni.usd.get_context().get_stage(), robot_prim_path=robot_prim_path)

    def collision_bounds(
        self,
        body_name: str,
        *,
        body_position_w_m: Sequence[float] | None = None,
        body_orientation_wxyz: Sequence[float] | None = None,
    ) -> tuple[Aabb | None, tuple[str, ...]]:
        # These imports remain inside the runtime path so unit tests and config
        # tools do not initialize Kit or Isaac modules.
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

        if body_name not in SENSED_BODIES:
            raise ValueError(f"body is outside the locked collision map: {body_name}")
        body = self.stage.GetPrimAtPath(f"{self.robot_prim_path}/{body_name}")
        if not body or not body.IsValid():
            return None, ()
        colliders = [
            prim
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
            if (
                prim.IsValid()
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and _collision_enabled(prim, UsdPhysics)
            )
        ]
        if not colliders:
            return None, ()
        paths = tuple(collider.GetPath().pathString for collider in colliders)

        live_position = _optional_vec3(body_position_w_m)
        live_orientation = _optional_quat(body_orientation_wxyz)
        if live_position is not None and live_orientation is not None:
            from pxr import Gf  # type: ignore

            points = self._body_local_points.get(body_name)
            if points is None:
                points = _body_local_collider_points(
                    body,
                    colliders,
                    Usd=Usd,
                    UsdGeom=UsdGeom,
                    Gf=Gf,
                )
                if not points:
                    return None, ()
                self._body_local_points[body_name] = points
                self._collider_paths[body_name] = paths
            bounds = _world_bounds_from_body_local_points(
                points,
                position_w_m=live_position,
                orientation_wxyz=live_orientation,
            )
            return bounds, self._collider_paths.get(body_name, paths)

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=False,
        )
        minima = [math.inf, math.inf, math.inf]
        maxima = [-math.inf, -math.inf, -math.inf]
        valid_paths: list[str] = []
        for collider in colliders:
            aligned = cache.ComputeWorldBound(collider).ComputeAlignedRange()
            low, high = aligned.GetMin(), aligned.GetMax()
            values = tuple(float(low[i]) for i in range(3)) + tuple(float(high[i]) for i in range(3))
            if not all(math.isfinite(value) and abs(value) < 1000.0 for value in values):
                continue
            for index in range(3):
                minima[index] = min(minima[index], values[index])
                maxima[index] = max(maxima[index], values[index + 3])
            valid_paths.append(collider.GetPath().pathString)
        if not valid_paths:
            return None, ()
        return Aabb(tuple(minima), tuple(maxima)), tuple(valid_paths)  # type: ignore[arg-type]


def aabb_intersection_depth(first: Aabb | None, second: Aabb | None) -> float:
    if first is None or second is None:
        return 0.0
    overlaps = [
        min(first.maximum_m[index], second.maximum_m[index])
        - max(first.minimum_m[index], second.minimum_m[index])
        for index in range(3)
    ]
    if any(value <= 0.0 for value in overlaps):
        return 0.0
    return float(min(overlaps))


def wheel_plane_metrics(wheel: WheelGeometry, planes: ObstaclePlanes) -> dict[str, float | bool | None]:
    """Small guard-ready geometry vocabulary for one live wheel."""

    if wheel.center_w_m is None or wheel.bottom_w_m is None or not wheel.verified:
        return {
            "verified": False,
            "front_clearance_m": None,
            "bottom_to_top_m": None,
            "center_past_front": None,
        }
    return {
        "verified": True,
        "front_clearance_m": wheel.center_w_m[0] - planes.front_x_m,
        "bottom_to_top_m": wheel.bottom_w_m[2] - planes.top_z_m,
        "center_past_front": wheel.center_w_m[0] >= planes.front_x_m,
    }


def _optional_vec3(values: Sequence[float] | None) -> Vec3 | None:
    if values is None:
        return None
    converted = tuple(float(value) for value in values)
    if len(converted) != 3 or not all(math.isfinite(value) for value in converted):
        return None
    return converted  # type: ignore[return-value]


def _optional_quat(values: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    converted = tuple(float(value) for value in values)
    if len(converted) != 4 or not all(math.isfinite(value) for value in converted):
        return None
    norm = math.sqrt(sum(value * value for value in converted))
    if norm <= 1.0e-12:
        return None
    return tuple(value / norm for value in converted)  # type: ignore[return-value]


def _collision_enabled(prim: Any, UsdPhysics: Any) -> bool:
    try:
        attribute = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        value = attribute.Get() if attribute else None
    except Exception:
        value = None
    return True if value is None else bool(value)


def _body_local_collider_points(
    body: Any,
    colliders: Sequence[Any],
    *,
    Usd: Any,
    UsdGeom: Any,
    Gf: Any,
) -> tuple[Vec3, ...]:
    """Resolve exact enabled collider mesh vertices into the rigid-body frame."""

    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    try:
        world_to_body = xforms.GetLocalToWorldTransform(body).GetInverse()
    except Exception:
        return ()
    result: list[Vec3] = []
    for collider in colliders:
        meshes = [
            prim
            for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies())
            if prim.IsValid() and prim.IsA(UsdGeom.Mesh)
        ]
        if not meshes:
            return ()
        collider_point_count = 0
        for mesh_prim in meshes:
            points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            if points is None:
                try:
                    prototype = mesh_prim.GetPrimInPrototype()
                    points = UsdGeom.Mesh(prototype).GetPointsAttr().Get()
                except Exception:
                    points = None
            if points is None:
                return ()
            try:
                mesh_to_world = xforms.GetLocalToWorldTransform(mesh_prim)
            except Exception:
                return ()
            for point in points:
                try:
                    world = mesh_to_world.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                    local = world_to_body.Transform(world)
                    values = tuple(float(local[index]) for index in range(3))
                except Exception:
                    return ()
                if not all(math.isfinite(value) and abs(value) < 1000.0 for value in values):
                    return ()
                result.append(values)  # type: ignore[arg-type]
                collider_point_count += 1
        if collider_point_count == 0:
            return ()
    return tuple(result)


def _world_bounds_from_body_local_points(
    points_body_m: Sequence[Sequence[float]],
    *,
    position_w_m: Sequence[float],
    orientation_wxyz: Sequence[float],
) -> Aabb | None:
    position = _optional_vec3(position_w_m)
    quaternion = _optional_quat(orientation_wxyz)
    if position is None or quaternion is None or not points_body_m:
        return None
    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    for point in points_body_m:
        local = _optional_vec3(point)
        if local is None:
            return None
        rotated = _quat_rotate(quaternion, local)
        world = tuple(position[index] + rotated[index] for index in range(3))
        if not all(math.isfinite(value) and abs(value) < 1000.0 for value in world):
            return None
        for index in range(3):
            minima[index] = min(minima[index], world[index])
            maxima[index] = max(maxima[index], world[index])
    return Aabb(tuple(minima), tuple(maxima))  # type: ignore[arg-type]


def _quat_rotate(
    quaternion_wxyz: Sequence[float],
    vector: Sequence[float],
) -> Vec3:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
