from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    PHYSICS_DT_S,
    SERVO_COMMAND_SIGN,
    SERVO_ORDER,
    WHEEL_VELOCITY_LIMIT_RAD_S,
    servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import (
    PHYSX_SAFE_LIMIT_MAX_RAD,
    PHYSX_SAFE_LIMIT_MIN_RAD,
    RobotAdapter,
    RobotAdapterError,
    authoritative_servo_limits_rad,
    bounded_drive_feedback_step,
)
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper


def _adapter_for_full12_feedback() -> tuple[RobotAdapter, dict[str, object]]:
    data = SimpleNamespace(
        joint_pos=np.zeros((1, len(FULL12_ORDER)), dtype=np.float64),
        joint_vel=np.zeros((1, len(FULL12_ORDER)), dtype=np.float64),
    )
    staged: dict[str, object] = {"writes": 0}
    robot = SimpleNamespace(
        data=data,
        set_joint_position_target=lambda targets, joint_ids: staged.update(
            position=targets.copy(), servo_ids=joint_ids
        ),
        set_joint_velocity_target=lambda targets, joint_ids: staged.update(
            velocity=targets.copy(), wheel_ids=joint_ids
        ),
        write_data_to_sim=lambda: staged.update(
            writes=int(staged["writes"]) + 1
        ),
    )
    standing_pose_deg = {name: 0.0 for name in SERVO_ORDER}
    adapter = RobotAdapter.__new__(RobotAdapter)
    adapter.robot = robot
    adapter.physics_dt_s = PHYSICS_DT_S
    adapter.joint_map = SimpleNamespace(
        servo_ids=tuple(range(8)),
        wheel_ids=tuple(range(8, 12)),
    )
    adapter._standing_servo_tensor = np.zeros((1, 8), dtype=np.float64)
    adapter.standing_pose_deg = standing_pose_deg
    adapter.servo_target_mapper = ServoTargetMapper(standing_pose_deg)
    adapter._final_drive_servo_deg = {name: 0.0 for name in SERVO_ORDER}
    adapter.write_count = 0
    adapter._last_physics_tick = None
    adapter.last_ack = None
    return adapter, staged


def test_authoritative_limits_cover_every_canonical_command_endpoint() -> None:
    standing = {
        name: 0.026844 + index * 0.01
        for index, name in enumerate(SERVO_ORDER)
    }
    for name in SERVO_ORDER:
        runtime_min_rad, runtime_max_rad = authoritative_servo_limits_rad(
            name,
            standing[name],
        )
        command_min_deg, command_max_deg = servo_limits_deg(name)
        physical_endpoints = [
            math.radians(
                standing[name] + float(SERVO_COMMAND_SIGN[name]) * command_deg
            )
            for command_deg in (command_min_deg, command_max_deg)
        ]
        assert runtime_min_rad < min(physical_endpoints)
        assert runtime_max_rad > max(physical_endpoints)
        assert runtime_min_rad > PHYSX_SAFE_LIMIT_MIN_RAD
        assert runtime_max_rad < PHYSX_SAFE_LIMIT_MAX_RAD

    fl_lower, fl_upper = authoritative_servo_limits_rad(
        "front_left_knee",
        standing["front_left_knee"],
    )
    p03_target = math.radians(standing["front_left_knee"] - 22.9)
    assert fl_lower < p03_target < fl_upper
    assert math.degrees(fl_lower) == pytest.approx(
        standing["front_left_knee"] - 60.0 - math.degrees(1.0e-5)
    )


def test_unknown_servo_limit_fails_closed() -> None:
    with pytest.raises(RobotAdapterError, match="unknown servo"):
        authoritative_servo_limits_rad("not_a_joint", 0.0)


def test_p09_positive_verify_tail_bias_restores_at_880_and_preserves_final_slew() -> None:
    base = {873: 18.885057, 880: 18.885057}
    previous = base[873]
    observed: dict[int, float] = {}
    for tick in range(874, 881):
        native_tick = max(item for item in base if item <= tick)
        native = base[native_tick]
        bias = 1.25 if 874 <= tick <= 879 else 0.0
        final = bounded_drive_feedback_step(
            previous_deg=previous,
            native_deg=native,
            bias_deg=bias,
            maximum_delta_deg=1.25,
            lower_deg=-60.0,
            upper_deg=210.0,
        )
        assert abs(final - previous) <= 1.25 + 1.0e-12
        observed[tick] = final
        previous = final

    assert observed[874] == pytest.approx(20.135057)
    assert observed[879] == pytest.approx(20.135057)
    assert observed[880] == pytest.approx(18.885057)


def test_p09_rear_left_alignment_release_restores_zero_realized_bias_at_872() -> None:
    previous = {4: 15.58, 5: 18.99}
    native = {4: 16.22, 5: 20.09}
    full_bias = {4: -1.185, 5: -1.455}
    release_bias = {4: 0.0, 5: -0.205}
    realized: dict[int, dict[int, float]] = {}

    for tick in range(860, 873):
        requested = (
            full_bias
            if tick <= 870
            else release_bias
            if tick == 871
            else {4: 0.0, 5: 0.0}
        )
        realized[tick] = {}
        for index in (4, 5):
            final = bounded_drive_feedback_step(
                previous_deg=previous[index],
                native_deg=native[index],
                bias_deg=requested[index],
                maximum_delta_deg=1.25,
                lower_deg=-135.0 if index == 4 else -60.0,
                upper_deg=135.0 if index == 4 else 210.0,
            )
            assert abs(final - previous[index]) <= 1.25 + 1.0e-12
            realized[tick][index] = final - native[index]
            previous[index] = final

    assert realized[860] == pytest.approx(full_bias)
    assert realized[870] == pytest.approx(full_bias)
    assert realized[871] == pytest.approx(release_bias)
    assert realized[872] == pytest.approx({4: 0.0, 5: 0.0})


def test_full12_feedback_applies_exact_fl_minus_1_07_and_tears_down() -> None:
    adapter, staged = _adapter_for_full12_feedback()
    zero = [0.0] * len(FULL12_ORDER)
    active_bias = zero.copy()
    active_bias[8] = -1.07

    active = adapter.apply_full12(
        zero,
        physics_tick=0,
        drive_feedback_bias_full12=active_bias,
    )
    assert active["native_drive_target_full12"] == pytest.approx(zero)
    assert active["drive_target_full12"] == pytest.approx(active_bias)
    assert active["drive_feedback_bias_requested_full12"] == pytest.approx(
        active_bias
    )
    assert active["drive_feedback_bias_realized_full12"] == pytest.approx(
        active_bias
    )
    # FL canonical forward sign is negative, so -1.07 logical maps to +1.07
    # on the physical articulation target.
    assert active["wheel_target_physical_rad_s"] == pytest.approx(
        [1.07, 0.0, 0.0, 0.0]
    )
    assert staged["velocity"][0].tolist() == pytest.approx(
        [1.07, 0.0, 0.0, 0.0]
    )

    teardown = adapter.apply_full12(
        zero,
        physics_tick=1,
        drive_feedback_bias_full12=zero,
    )
    assert teardown["native_drive_target_full12"] == pytest.approx(zero)
    assert teardown["drive_target_full12"] == pytest.approx(zero)
    assert teardown["drive_feedback_bias_requested_full12"] == pytest.approx(zero)
    assert teardown["drive_feedback_bias_realized_full12"] == pytest.approx(zero)
    assert teardown["wheel_target_physical_rad_s"] == pytest.approx([0.0] * 4)
    assert staged["writes"] == 2


def test_full12_feedback_validates_all_channels_and_clamps_final_wheel() -> None:
    adapter, _staged = _adapter_for_full12_feedback()
    command = [0.0] * len(FULL12_ORDER)
    command[8] = 2.0
    feedback = [0.0] * len(FULL12_ORDER)
    feedback[8] = 0.2

    ack = adapter.apply_full12(
        command,
        physics_tick=0,
        drive_feedback_bias_full12=feedback,
    )
    assert ack["native_drive_target_full12"][8] == pytest.approx(2.0)
    assert ack["drive_target_full12"][8] == pytest.approx(
        WHEEL_VELOCITY_LIMIT_RAD_S
    )
    assert ack["drive_feedback_bias_requested_full12"][8] == pytest.approx(0.2)
    assert ack["drive_feedback_bias_realized_full12"][8] == pytest.approx(
        WHEEL_VELOCITY_LIMIT_RAD_S - 2.0
    )

    nonfinite = [0.0] * len(FULL12_ORDER)
    nonfinite[8] = math.nan
    with pytest.raises(RobotAdapterError, match="non-finite"):
        adapter.apply_full12(
            [0.0] * len(FULL12_ORDER),
            physics_tick=1,
            drive_feedback_bias_full12=nonfinite,
        )

    over_limit = [0.0] * len(FULL12_ORDER)
    over_limit[8] = WHEEL_VELOCITY_LIMIT_RAD_S + 0.01
    with pytest.raises(RobotAdapterError, match="wheel hard limit"):
        adapter.apply_full12(
            [0.0] * len(FULL12_ORDER),
            physics_tick=1,
            drive_feedback_bias_full12=over_limit,
        )


def test_adapter_authors_all_limits_on_session_layer_and_syncs_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAttr:
        def __init__(self) -> None:
            self.value: float | None = None

        def Set(self, value: float) -> bool:
            self.value = float(np.float32(value))
            return True

        def Get(self) -> float | None:
            return self.value

    class FakePrim:
        def __init__(self, name: str) -> None:
            self.name = name
            self.path = f"/World/WLRRobot/joints/{name}"
            self.lower = FakeAttr()
            self.upper = FakeAttr()

        def GetName(self) -> str:
            return self.name

        def GetPath(self) -> str:
            return self.path

        def IsA(self, api: object) -> bool:
            return api is FakeRevoluteJoint

    class FakeRoot:
        def __init__(self, prims: list[FakePrim]) -> None:
            self.prims = prims

        def IsValid(self) -> bool:
            return True

    class FakeRevoluteJoint:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def CreateLowerLimitAttr(self) -> FakeAttr:
            return self.prim.lower

        def CreateUpperLimitAttr(self) -> FakeAttr:
            return self.prim.upper

    class FakeSessionLayer:
        identifier = "anon:unit-test-session"

    class FakeStage:
        def __init__(self, root: FakeRoot) -> None:
            self.root = root
            self.session = FakeSessionLayer()
            self.edit_context_entered = False

        def GetPrimAtPath(self, path: str) -> FakeRoot:
            assert path == "/World/WLRRobot"
            return self.root

        def GetSessionLayer(self) -> FakeSessionLayer:
            return self.session

    class FakeEditContext:
        def __init__(self, stage: FakeStage, target: FakeSessionLayer) -> None:
            self.stage = stage
            self.target = target

        def __enter__(self) -> None:
            assert self.target is self.stage.session
            self.stage.edit_context_entered = True

        def __exit__(self, *_args: object) -> None:
            return None

    prims = [FakePrim(name) for name in SERVO_ORDER]
    stage = FakeStage(FakeRoot(prims))
    isaaclab = ModuleType("isaaclab")
    isaaclab_sim = ModuleType("isaaclab.sim")
    isaaclab_sim.get_current_stage = lambda: stage  # type: ignore[attr-defined]
    isaaclab.sim = isaaclab_sim  # type: ignore[attr-defined]
    pxr = ModuleType("pxr")
    pxr.Usd = SimpleNamespace(  # type: ignore[attr-defined]
        PrimRange=lambda root: root.prims,
        EditTarget=lambda layer: layer,
        EditContext=FakeEditContext,
    )
    pxr.UsdPhysics = SimpleNamespace(RevoluteJoint=FakeRevoluteJoint)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    data = SimpleNamespace(
        joint_pos=np.zeros((1, len(FULL12_ORDER)), dtype=np.float64),
        joint_vel=np.zeros((1, len(FULL12_ORDER)), dtype=np.float64),
        joint_pos_limits=np.zeros((1, len(FULL12_ORDER), 2), dtype=np.float64),
        soft_joint_pos_limits=np.zeros((1, len(FULL12_ORDER), 2), dtype=np.float64),
    )
    robot = SimpleNamespace(
        joint_names=list(FULL12_ORDER),
        data=data,
        cfg=SimpleNamespace(soft_joint_pos_limit_factor=1.0),
        root_physx_view=SimpleNamespace(
            get_dof_limits=lambda: data.joint_pos_limits.astype(np.float32)
        ),
    )

    adapter = RobotAdapter(robot)
    assert adapter.joint_limit_initialization_evidence()["physx_limits_verified"] is False
    adapter.verify_authoritative_servo_limits_adopted()

    assert stage.edit_context_entered is True
    assert len(adapter.live_servo_limit_records) == 8
    assert [record.joint_name for record in adapter.live_servo_limit_records] == list(
        SERVO_ORDER
    )
    for record, prim in zip(adapter.live_servo_limit_records, prims, strict=True):
        assert prim.lower.value == pytest.approx(math.degrees(record.runtime_min_rad))
        assert prim.upper.value == pytest.approx(math.degrees(record.runtime_max_rad))
        assert data.joint_pos_limits[0, record.joint_id, 0] == pytest.approx(
            record.runtime_min_rad
        )
        assert data.joint_pos_limits[0, record.joint_id, 1] == pytest.approx(
            record.runtime_max_rad
        )
        assert data.soft_joint_pos_limits[0, record.joint_id] == pytest.approx(
            data.joint_pos_limits[0, record.joint_id]
        )
    assert np.all(data.joint_pos_limits[0, 8:, :] == 0.0)
    evidence = adapter.joint_limit_initialization_evidence()
    assert evidence["all_eight_servo_limits_applied"] is True
    assert evidence["physx_limits_verified"] is True
    assert evidence["runtime_authoring_layer"] == "session_layer"
    assert evidence["source_asset_modified"] is False
    assert evidence["stage_saved"] is False

    staged: dict[str, object] = {"writes": 0}
    robot.set_joint_position_target = (  # type: ignore[attr-defined]
        lambda targets, joint_ids: staged.update(position=targets.copy(), servo_ids=joint_ids)
    )
    robot.set_joint_velocity_target = (  # type: ignore[attr-defined]
        lambda targets, joint_ids: staged.update(velocity=targets.copy(), wheel_ids=joint_ids)
    )
    robot.write_data_to_sim = (  # type: ignore[attr-defined]
        lambda: staged.update(writes=int(staged["writes"]) + 1)
    )
    p03 = [0.0] * 12
    p03[1] = -22.9
    p03[8:] = [-0.79, 0.0, 0.61, 0.0]
    ack = adapter.apply_full12(
        p03,
        physics_tick=0,
        tracking_servo_names=SERVO_ORDER,
    )
    assert staged["writes"] == 1
    assert ack["articulation_writes_this_call"] == 1
    assert ack["motion_start_skew_s"] == 0.0
    assert ack["requested_full12"] == p03
    assert ack["applied_full12"] == p03
    assert ack["drive_target_full12"] == pytest.approx(
        [0.0, -1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.79, 0.0, 0.61, 0.0]
    )
    assert ack["servo_applied_drive_command_deg"][1] == pytest.approx(-1.25)
    assert staged["position"][0, 1] == pytest.approx(math.radians(-1.25))
    assert staged["velocity"][0].tolist() == pytest.approx([0.79, 0.0, -0.61, 0.0])

    data.joint_pos_limits[0, 0, 0] = 0.0
    with pytest.raises(RobotAdapterError, match="PhysX lower limit"):
        adapter.verify_authoritative_servo_limits_adopted()
    assert adapter.joint_limit_initialization_evidence()["physx_limits_verified"] is False
