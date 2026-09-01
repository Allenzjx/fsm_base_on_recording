from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.sensing.body_collision_detector import BodyCollisionDetector
from wlr50_clean.sensing.com_diagnostics import (
    compute_full_body_com,
    compute_support_diagnostics,
)
from wlr50_clean.sensing.contact_classifier import (
    BASE_BODY,
    GROUND_PAIR,
    OBSTACLE_PAIR,
    SENSED_BODIES,
    WHEEL_BODIES,
    ContactClassifier,
    RawPairContact,
)
from wlr50_clean.sensing.geometry import (
    Aabb,
    ColliderGeometryCache,
    GeometrySnapshot,
    UsdCollisionBoundsProvider,
    WheelGeometry,
    _world_bounds_from_body_local_points,
)
from wlr50_clean.sensing.guard_state import LiveGuardTracker
from wlr50_clean.sensing.observation import ContactClass
from wlr50_clean.sensing.sensor_reader import (
    CONTACT_FILTERS,
    ExactPairContactSensorBank,
    SensorReader,
    SensingContractError,
)


OBSTACLE_PATH = "/World/Obstacle"
GROUND_PATH = "/World/defaultGroundPlane/GroundPlane/CollisionPlane"


def _pair_samples(
    *,
    base_force: float = 0.0,
    wheel_ground: bool = False,
    with_points: bool = False,
) -> tuple[RawPairContact, ...]:
    rows: list[RawPairContact] = []
    wheel_index = {name: index for index, name in enumerate(WHEEL_BODIES)}
    for body in SENSED_BODIES:
        obstacle_force = base_force if body == BASE_BODY else 0.0
        rows.append(
            RawPairContact(
                sensor_body=body,
                pair_kind=OBSTACLE_PAIR,
                other_body=OBSTACLE_PATH,
                force_w_n=(obstacle_force, 0.0, 0.0),
                pair_verified=True,
                source="fake exact pair",
            )
        )
        ground_force = 4.0 if wheel_ground and body in wheel_index else 0.0
        point = None
        if ground_force and with_points:
            index = wheel_index[body]
            point = (0.2 * (index // 2), 0.2 * (index % 2), 0.0)
        rows.append(
            RawPairContact(
                sensor_body=body,
                pair_kind=GROUND_PAIR,
                other_body=GROUND_PATH,
                force_w_n=(0.0, 0.0, ground_force),
                contact_point_w_m=point,
                pair_verified=True,
                source="fake exact pair",
            )
        )
    return tuple(rows)


def test_body_collision_needs_exact_pair_and_second_evidence() -> None:
    classifier = ContactClassifier()
    detector = BodyCollisionDetector()

    first = classifier.classify(_pair_samples(base_force=3.0))
    status = detector.evaluate(first, base_obstacle_penetration_m=0.0)
    assert status.real_pair_active
    assert not status.persistent
    assert not status.detected

    second = classifier.classify(_pair_samples(base_force=3.0))
    status = detector.evaluate(second, base_obstacle_penetration_m=0.0)
    assert status.persistent
    assert status.detected


def test_body_collision_accepts_live_penetration_corroboration_but_not_leg_contact() -> None:
    classifier = ContactClassifier()
    samples = list(_pair_samples())
    samples = [
        RawPairContact(
            sensor_body=row.sensor_body,
            pair_kind=row.pair_kind,
            other_body=row.other_body,
            force_w_n=(2.0, 0.0, 0.0)
            if row.sensor_body == "front_left_bot" and row.pair_kind == OBSTACLE_PAIR
            else row.force_w_n,
            pair_verified=True,
        )
        for row in samples
    ]
    contacts = classifier.classify(samples)
    detector = BodyCollisionDetector()
    assert contacts["front_left_bot"].contact_class is ContactClass.OBSTACLE
    assert not detector.evaluate(contacts, base_obstacle_penetration_m=0.02).detected

    contacts = classifier.classify(_pair_samples(base_force=2.0))
    assert detector.evaluate(contacts, base_obstacle_penetration_m=0.002).detected


class _Bounds:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def collision_bounds(
        self,
        body_name: str,
        *,
        body_position_w_m=None,
        body_orientation_wxyz=None,
    ):
        self.calls[body_name] = self.calls.get(body_name, 0) + 1
        center_x = 0.1 if body_name != BASE_BODY else 0.0
        return Aabb((center_x - 0.04, -0.04, 0.001), (center_x + 0.04, 0.04, 0.099)), (
            f"/World/WLRRobot/{body_name}/collisions/mesh",
        )


def test_wheel_bottom_uses_cached_measured_collider_extent() -> None:
    provider = _Bounds()
    cache = ColliderGeometryCache(provider)
    positions = {BASE_BODY: (0.0, 0.0, 0.05)}
    positions.update({name: (0.1, 0.0, 0.05) for name in WHEEL_BODIES})
    first = cache.sample(positions)
    assert first.wheels["front_left_ankle"].bottom_w_m == pytest.approx((0.1, 0.0, 0.001))
    assert first.wheels["front_left_ankle"].verified

    moved = dict(positions)
    moved["front_left_wheel"] = (0.3, 0.0, 0.07)
    second = cache.sample(moved)
    assert second.wheels["front_left_ankle"].bottom_w_m == pytest.approx((0.3, 0.0, 0.021))
    assert provider.calls["front_left_wheel"] == 1


def test_usd_bounds_traverse_instance_proxies_and_include_guide(monkeypatch) -> None:
    traversal_token = object()
    collision_api = object()
    calls: dict[str, object] = {}

    class FakePath:
        pathString = "/World/WLRRobot/front_left_wheel/collisions/mesh"

    class FakePrim:
        def IsValid(self) -> bool:
            return True

        def HasAPI(self, api) -> bool:
            return api is collision_api

        def GetPath(self) -> FakePath:
            return FakePath()

    class FakeStage:
        def GetPrimAtPath(self, path: str) -> FakePrim:
            calls["body_path"] = path
            return FakePrim()

    class FakeRange:
        def GetMin(self):
            return (0.1, -0.1, 0.0)

        def GetMax(self):
            return (0.2, 0.1, 0.1)

    class FakeBound:
        def ComputeAlignedRange(self) -> FakeRange:
            return FakeRange()

    class FakeBBoxCache:
        def __init__(self, _time, purposes, *, useExtentsHint: bool):
            calls["purposes"] = tuple(purposes)
            calls["use_extents_hint"] = useExtentsHint

        def ComputeWorldBound(self, _prim: FakePrim) -> FakeBound:
            return FakeBound()

    fake_usd = SimpleNamespace(
        PrimRange=lambda root, predicate: (
            calls.update(root=root, predicate=predicate) or [FakePrim()]
        ),
        TimeCode=SimpleNamespace(Default=lambda: 0),
        TraverseInstanceProxies=lambda: traversal_token,
    )
    fake_usd_geom = SimpleNamespace(
        BBoxCache=FakeBBoxCache,
        Tokens=SimpleNamespace(
            default_="default", render="render", proxy="proxy", guide="guide"
        ),
    )
    pxr = ModuleType("pxr")
    pxr.Usd = fake_usd
    pxr.UsdGeom = fake_usd_geom
    pxr.UsdPhysics = SimpleNamespace(CollisionAPI=collision_api)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    bounds, paths = UsdCollisionBoundsProvider(FakeStage()).collision_bounds(
        "front_left_wheel"
    )

    assert calls["body_path"] == "/World/WLRRobot/front_left_wheel"
    assert calls["predicate"] is traversal_token
    assert calls["purposes"] == ("default", "render", "proxy", "guide")
    assert calls["use_extents_hint"] is False
    assert bounds == Aabb((0.1, -0.1, 0.0), (0.2, 0.1, 0.1))
    assert paths == ("/World/WLRRobot/front_left_wheel/collisions/mesh",)


def test_body_local_collider_points_use_live_pose_before_geometry_is_cached() -> None:
    half = 2.0**-0.5
    bounds = _world_bounds_from_body_local_points(
        ((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, -0.05), (0.0, 0.0, 0.05)),
        position_w_m=(0.25, -0.3, 0.05),
        orientation_wxyz=(half, 0.0, half, 0.0),
    )

    assert bounds is not None
    assert bounds.minimum_m == pytest.approx((0.2, -0.3, 0.0))
    assert bounds.maximum_m == pytest.approx((0.3, -0.3, 0.1))


def test_full_body_com_and_support_polygon() -> None:
    positions = {name: (float(index), 0.1, 0.2) for index, name in enumerate(SENSED_BODIES)}
    velocities = {name: (1.0, 0.0, 0.0) for name in SENSED_BODIES}
    masses = {name: 1.0 for name in SENSED_BODIES}
    com = compute_full_body_com(
        body_positions_w_m=positions,
        body_velocities_w_m_s=velocities,
        body_masses_kg=masses,
    )
    assert com.valid
    assert com.position_w_m == pytest.approx((6.0, 0.1, 0.2))
    assert len(com.included_bodies) == 13

    classifier = ContactClassifier()
    contacts = classifier.classify(_pair_samples(wheel_ground=True, with_points=True))
    wheels = {
        name: WheelGeometry(name, body, None, None, None, "unused", False)
        for name, body in zip(WHEEL_ORDER, WHEEL_BODIES, strict=True)
    }
    # Put the diagnostic CoM projection inside the 0.2 m square.
    com = compute_full_body_com(
        body_positions_w_m={name: (0.1, 0.1, 0.2) for name in SENSED_BODIES},
        body_velocities_w_m_s=velocities,
        body_masses_kg=masses,
    )
    support = compute_support_diagnostics(center_of_mass=com, contacts=contacts, wheels=wheels)
    assert support.valid
    assert support.projection_inside
    assert support.support_count == 4


def test_contact_sensor_factory_is_one_exact_sensor_per_body() -> None:
    class FakeCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSensor:
        def __init__(self, cfg):
            self.cfg = cfg
            self.is_initialized = True

    bank = ExactPairContactSensorBank.create(
        imports={"ContactSensor": FakeSensor, "ContactSensorCfg": FakeCfg}
    )
    assert len(bank.sensors) == 13
    for body, sensor in bank.sensors.items():
        assert sensor.cfg.prim_path == f"/World/WLRRobot/{body}"
        assert sensor.cfg.update_period == 0.0
        assert sensor.cfg.history_length == 3
        assert sensor.cfg.filter_prim_paths_expr == [path for _, path in CONTACT_FILTERS]


class _FakeContactBackend:
    last_quality: tuple[str, ...] = ()

    def sample(self, physics_dt_s: float):
        assert physics_dt_s == pytest.approx(1.0 / 120.0)
        return _pair_samples(wheel_ground=True, with_points=True)


class _FakeGeometryBackend:
    def sample(self, body_positions_w_m, body_orientations_wxyz=None):
        assert body_orientations_wxyz is not None
        wheels = {}
        for name, body in zip(WHEEL_ORDER, WHEEL_BODIES, strict=True):
            center = tuple(body_positions_w_m[body])
            wheels[name] = WheelGeometry(
                name,
                body,
                center,
                (center[0], center[1], center[2] - 0.049),
                None,
                "fake measured collider",
                True,
            )
        return GeometrySnapshot(wheels=wheels, body_bounds_w_m={}, base_obstacle_penetration_m=0.0)


def _fake_adapter():
    count = len(SENSED_BODIES)
    link_pos = np.zeros((1, count, 3), dtype=float)
    link_pos[0, :, 0] = np.linspace(0.0, 0.3, count)
    link_pos[0, :, 2] = 0.1
    quat = np.zeros((1, count, 4), dtype=float)
    quat[0, :, 0] = 1.0
    zeros = np.zeros((1, count, 3), dtype=float)
    data = SimpleNamespace(
        body_link_pos_w=link_pos,
        body_link_quat_w=quat,
        body_link_lin_vel_w=zeros,
        body_link_ang_vel_w=zeros,
        body_com_pos_w=link_pos.copy(),
        body_com_lin_vel_w=zeros.copy(),
        body_com_lin_acc_w=zeros.copy(),
        default_mass=np.ones((1, count), dtype=float),
        root_ang_vel_b=np.zeros((1, 3), dtype=float),
        projected_gravity_b=np.asarray([[0.0, 0.0, -1.0]]),
    )
    robot = SimpleNamespace(body_names=list(SENSED_BODIES), data=data)

    class Adapter:
        def __init__(self):
            self.robot = robot

        def get_actual_state(self):
            return SimpleNamespace(
                full12=(1.0,) * 8 + (0.2,) * 4,
                servo_velocity_rad_s=(0.1,) * 8,
            )

    return Adapter()


def test_sensor_reader_assembles_atomic_live_observation() -> None:
    reader = SensorReader(
        _fake_adapter(),
        contact_backend=_FakeContactBackend(),
        geometry_backend=_FakeGeometryBackend(),
    )
    observation = reader.read(
        physics_tick=10,
        simulation_time_s=1.0,
        commanded_full12=(2.0,) * 8 + (0.4,) * 4,
    )
    assert observation.schema == "wlr50_clean.live_observation.v1"
    assert observation.physics_tick == 10
    assert len(observation.joints) == 8
    assert len(observation.wheels) == 4
    assert len(observation.contacts) == 13
    assert observation.joints["front_left_hip"].error_deg == pytest.approx(1.0)
    assert observation.wheels["front_left_ankle"].geometry_verified
    assert observation.center_of_mass.valid
    assert observation.imu.specific_force_b_m_s2 == pytest.approx((0.0, 0.0, 9.81))

    reader.read(
        physics_tick=11,
        simulation_time_s=1.0 + 1.0 / 120.0,
        commanded_full12=(2.0,) * 8 + (0.4,) * 4,
    )
    with pytest.raises(SensingContractError, match="contiguous"):
        reader.read(
            physics_tick=13,
            simulation_time_s=1.0 + 3.0 / 120.0,
            commanded_full12=(2.0,) * 8 + (0.4,) * 4,
        )


def test_live_observation_resolves_every_sensor_guard_in_fsm_config() -> None:
    reader = SensorReader(
        _fake_adapter(),
        contact_backend=_FakeContactBackend(),
        geometry_backend=_FakeGeometryBackend(),
    )
    observation = reader.read(
        physics_tick=0,
        simulation_time_s=0.0,
        commanded_full12=(2.0,) * 8 + (0.4,) * 4,
    )
    config_path = Path(__file__).parents[2] / "configs" / "fsm_states.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    controller_local = {
        "previous_state_done",
        "motion_endpoint_issued",
        "wheel_targets_zero",
        "measured_wheel_velocity_stable_decay",
    }
    unresolved: list[str] = []
    for state in config["states"]:
        for group in ("entry_conditions", "completion_conditions", "hard_abort_conditions"):
            for authored in state[group]:
                name = authored["guard"]
                if name in controller_local:
                    continue
                parameters = {key: value for key, value in authored.items() if key not in {"guard", "result"}}
                if observation.resolve_guard(name, parameters) is None:
                    unresolved.append(f"{state['state_id']}:{name}")
    assert unresolved == []
    assert observation.measured_wheel_velocity_rad_s == pytest.approx((0.2, 0.2, 0.2, 0.2))


def test_front_crossing_requires_prior_or_same_sample_active_lift() -> None:
    reader = SensorReader(
        _fake_adapter(),
        contact_backend=_FakeContactBackend(),
        geometry_backend=_FakeGeometryBackend(),
    )
    initial = reader.read(
        physics_tick=0,
        simulation_time_s=0.0,
        commanded_full12=(2.0,) * 8 + (0.4,) * 4,
    )
    fr_wheel_name = "front_right_ankle"
    fr_body = "front_right_wheel"
    initial_wheel = initial.wheels[fr_wheel_name]
    air_contact = replace(
        initial.contacts[fr_body],
        ground=replace(initial.contacts[fr_body].ground, active=False),
        obstacle=replace(initial.contacts[fr_body].obstacle, active=False),
    )
    crossed_wheel = replace(
        initial_wheel,
        center_w_m=(initial.obstacle.front_x_m + 0.001, initial_wheel.center_w_m[1], 0.12),
        bottom_w_m=(initial.obstacle.front_x_m + 0.001, initial_wheel.center_w_m[1], 0.071),
    )

    tracker = LiveGuardTracker()
    tracker.update(initial)
    joints = dict(initial.joints)
    joints["front_right_knee"] = replace(
        joints["front_right_knee"], position_deg=joints["front_right_knee"].position_deg + 3.0
    )
    wheels = dict(initial.wheels)
    wheels[fr_wheel_name] = crossed_wheel
    contacts = dict(initial.contacts)
    contacts[fr_body] = air_contact
    active_crossing = replace(initial, physics_tick=1, joints=joints, wheels=wheels, contacts=contacts)
    guards, _ = tracker.update(active_crossing)
    assert guards["reference_like_active_lift:FR"]["passed"]
    assert not guards["wheel_only_climb_detected"]["passed"]

    no_lift_tracker = LiveGuardTracker()
    no_lift_tracker.update(initial)
    unactuated_crossing = replace(active_crossing, joints=initial.joints)
    guards, _ = no_lift_tracker.update(unactuated_crossing)
    assert guards["wheel_only_climb_detected"]["passed"]
    assert guards["wheel_only_climb_detected"]["value"]["front_crossed_ticks"]["FR"] == 1


def test_active_lift_does_not_latch_from_far_field_posture_motion() -> None:
    reader = SensorReader(
        _fake_adapter(),
        contact_backend=_FakeContactBackend(),
        geometry_backend=_FakeGeometryBackend(),
    )
    initial = reader.read(
        physics_tick=0,
        simulation_time_s=0.0,
        commanded_full12=(2.0,) * 8 + (0.4,) * 4,
    )
    tracker = LiveGuardTracker()
    tracker.update(initial)
    joints = dict(initial.joints)
    joints["rear_left_hip"] = replace(
        joints["rear_left_hip"], position_deg=joints["rear_left_hip"].position_deg + 3.0
    )
    wheel_name = "rear_left_ankle"
    wheel = initial.wheels[wheel_name]
    wheels = dict(initial.wheels)
    wheels[wheel_name] = replace(
        wheel,
        center_w_m=(initial.obstacle.front_x_m - 0.50, wheel.center_w_m[1], 0.07),
        bottom_w_m=(initial.obstacle.front_x_m - 0.50, wheel.bottom_w_m[1], 0.02),
    )
    body_name = wheels[wheel_name].body_name
    contacts = dict(initial.contacts)
    contacts[body_name] = replace(
        contacts[body_name],
        ground=replace(contacts[body_name].ground, active=False),
        obstacle=replace(contacts[body_name].obstacle, active=False),
    )
    guards, _ = tracker.update(
        replace(initial, physics_tick=1, joints=joints, wheels=wheels, contacts=contacts)
    )
    assert not guards["reference_like_active_lift:RL"]["passed"]
