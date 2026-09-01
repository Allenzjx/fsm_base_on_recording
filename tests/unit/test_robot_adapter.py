from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_COMMAND_SIGN,
    SERVO_ORDER,
    servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import (
    PHYSX_SAFE_LIMIT_MAX_RAD,
    PHYSX_SAFE_LIMIT_MIN_RAD,
    RobotAdapter,
    RobotAdapterError,
    authoritative_servo_limits_rad,
)


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

    data.joint_pos_limits[0, 0, 0] = 0.0
    with pytest.raises(RobotAdapterError, match="PhysX lower limit"):
        adapter.verify_authoritative_servo_limits_adopted()
    assert adapter.joint_limit_initialization_evidence()["physx_limits_verified"] is False
