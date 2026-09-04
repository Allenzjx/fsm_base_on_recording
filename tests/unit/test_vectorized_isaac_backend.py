from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from wlr50_clean.ppo.vectorized_isaac_backend import (
    SUPPORTED_VECTOR_ENV_COUNTS,
    BatchedAuthoritativeFrame,
    VectorizedExactPairFailure,
    VectorizedIsaacBackendError,
    VectorizedIsaacFSMBackend,
    _BatchedExactPairContactBank,
    _create_vector_scene,
    probe_vectorized_isaac_backend,
)
import wlr50_clean.ppo.vectorized_isaac_backend as vector_backend_module
from wlr50_clean.sensing.contact_classifier import SENSED_BODIES
from wlr50_clean.sensing.sensor_reader import (
    GROUND_COLLISION_PRIM_PATH,
    OBSTACLE_PRIM_PATH,
)


class _FakeSensor:
    def __init__(
        self,
        body_name: str,
        num_envs: int,
        *,
        force_shape: tuple[int, ...] | None = None,
        filter_count: int = 2,
    ) -> None:
        matrix_shape = force_shape or (num_envs, 1, 2, 3)
        force = np.zeros(matrix_shape, dtype=float)
        history = np.zeros((num_envs, 3, 1, 2, 3), dtype=float)
        points = np.full((num_envs, 1, 2, 3), np.nan, dtype=float)
        friction = np.zeros((num_envs, 1, 2, 3), dtype=float)
        if force_shape is None:
            force[:, 0, 0, 2] = np.arange(num_envs, dtype=float) + 1.0
            history[:, :, 0, 0, 2] = force[:, None, 0, 0, 2]
            points[:, 0, 0, :] = np.array([0.5, 0.0, 0.05])
        self.is_initialized = True
        self.body_names = (body_name,)
        self.contact_physx_view = SimpleNamespace(filter_count=filter_count)
        self.cfg = SimpleNamespace(
            filter_prim_paths_expr=(
                "/World/envs/env_.*/Obstacle",
                "/World/envs/env_.*/defaultGroundPlane/GroundPlane/CollisionPlane",
            )
        )
        self.data = SimpleNamespace(
            force_matrix_w=force,
            force_matrix_w_history=history,
            contact_pos_w=points,
            friction_forces_w=friction,
        )
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _bank(num_envs: int = 8) -> _BatchedExactPairContactBank:
    sensors = {body: _FakeSensor(body, num_envs) for body in SENSED_BODIES}
    origins = np.zeros((num_envs, 3), dtype=float)
    origins[:, 0] = np.arange(num_envs, dtype=float) * 8.0
    return _BatchedExactPairContactBank(sensors, origins, num_envs)


@pytest.mark.parametrize("num_envs", SUPPORTED_VECTOR_ENV_COUNTS)
def test_offline_probe_never_claims_live_vectorization(num_envs: int) -> None:
    report = probe_vectorized_isaac_backend(
        num_envs,
        module_finder=lambda _name: object(),
        verify_asset=True,
    )

    assert report.status == "LIVE_ISAAC_BENCHMARK_REQUIRED"
    assert report.num_envs == num_envs
    assert report.supported_count is True
    assert report.locked_asset_verified is True
    assert report.live_vectorization_verified is False
    assert any("have not run" in reason for reason in report.reasons)
    assert report.as_dict()["live_vectorization_verified"] is False


@pytest.mark.parametrize("bad", [False, 1, 4, 7, 64, 8.5, "8"])
def test_only_required_physical_benchmark_sizes_are_accepted(bad) -> None:
    with pytest.raises(VectorizedIsaacBackendError, match="one of"):
        probe_vectorized_isaac_backend(bad, verify_asset=False)


def test_exact_pair_bank_preserves_rows_and_canonicalizes_local_paths() -> None:
    bank = _bank()
    bank.capture(1)
    row = bank.sample_row(3)

    assert len(row) == 13 * 2
    assert bank.capture_count == 1
    obstacle = row[0]
    ground = row[1]
    assert obstacle.sensor_body == SENSED_BODIES[0]
    assert obstacle.other_body == OBSTACLE_PRIM_PATH
    assert ground.other_body == GROUND_COLLISION_PRIM_PATH
    assert obstacle.pair_verified is True
    assert obstacle.force_w_n == (0.0, 0.0, 4.0)
    assert obstacle.history_force_w_n == ((0.0, 0.0, 4.0),) * 3
    # The fake world point is translated into row-local coordinates. A real
    # point this far from env 3 is intentionally odd but proves the subtraction.
    assert obstacle.contact_point_w_m == (-23.5, 0.0, 0.05)


def test_exact_pair_bank_fails_closed_on_shape_filter_and_capture_errors() -> None:
    num_envs = 8
    origins = np.zeros((num_envs, 3), dtype=float)
    bad_shape = {body: _FakeSensor(body, num_envs) for body in SENSED_BODIES}
    bad_shape[SENSED_BODIES[4]] = _FakeSensor(
        SENSED_BODIES[4], num_envs, force_shape=(1, 1, 2, 3)
    )
    with pytest.raises(VectorizedExactPairFailure, match="force_matrix_w shape"):
        _BatchedExactPairContactBank(bad_shape, origins, num_envs).capture(1)

    bad_filter = {body: _FakeSensor(body, num_envs) for body in SENSED_BODIES}
    bad_filter[SENSED_BODIES[2]] = _FakeSensor(
        SENSED_BODIES[2], num_envs, filter_count=26
    )
    with pytest.raises(VectorizedExactPairFailure, match="filter_count"):
        _BatchedExactPairContactBank(bad_filter, origins, num_envs).capture(1)

    bank = _bank()
    bank.capture(1)
    with pytest.raises(VectorizedExactPairFailure, match="once"):
        bank.capture(1)


def test_no_uncaptured_or_out_of_range_contact_row_can_be_used() -> None:
    bank = _bank()
    with pytest.raises(VectorizedExactPairFailure, match="captured"):
        bank.sample_row(0)
    with pytest.raises(VectorizedExactPairFailure, match="out of range"):
        bank.row_backend(8)


def test_exact_pair_bank_fails_closed_on_filter_order_or_path() -> None:
    num_envs = 8
    sensors = {body: _FakeSensor(body, num_envs) for body in SENSED_BODIES}
    expected = tuple(sensors[SENSED_BODIES[0]].cfg.filter_prim_paths_expr)
    sensors[SENSED_BODIES[-1]].cfg.filter_prim_paths_expr = tuple(reversed(expected))

    with pytest.raises(VectorizedExactPairFailure, match="filter order/path mismatch"):
        _BatchedExactPairContactBank(
            sensors,
            np.zeros((num_envs, 3), dtype=float),
            num_envs,
            expected_filter_paths=expected,
        )


def test_batched_frame_exposes_synchronized_rows_without_emulating_steps() -> None:
    frames = tuple(
        SimpleNamespace(state_id="P02", nominal_action_full12=(float(row),) * 12)
        for row in range(8)
    )
    batch = BatchedAuthoritativeFrame(
        physics_tick=3,
        sim_time_s=3.0 / 120.0,
        frames=frames,
        global_physics_step_count=183,
        batched_articulation_write_count=183,
        exact_pair_capture_count=183,
    )
    assert batch.state_ids == ("P02",) * 8
    assert batch.nominal_actions_full12[7] == (7.0,) * 12


def test_source_contract_has_one_global_step_and_no_single_env_physics_loop() -> None:
    advance = inspect.getsource(VectorizedIsaacFSMBackend._advance_global_physics)
    step = inspect.getsource(VectorizedIsaacFSMBackend.step_physics_batch)

    assert advance.count("self.sim.step(") == 1
    assert step.count("self._advance_global_physics()") == 1
    assert "for row" in step
    assert "self.sim.step(" not in step
    assert "IsaacFSMBackend(" not in step
    assert "build_residual_actuation_plan(" in step
    assert "plan.frozen_nominal_full12" in step
    assert "plan.combined_post_mapper_bias_full12" in step


def test_vector_reset_uses_native_hard_reset_without_indexed_state_writes() -> None:
    prepare = inspect.getsource(VectorizedIsaacFSMBackend._prepare_physical_reset)
    hard_reset = inspect.getsource(VectorizedIsaacFSMBackend._guarded_global_hard_reset)

    assert "self._guarded_global_hard_reset()" in prepare
    assert "reset(soft=False)" in hard_reset
    assert ".stop(" not in hard_reset
    assert "write_root_pose_to_sim" not in prepare + hard_reset
    assert "write_root_velocity_to_sim" not in prepare + hard_reset
    assert "write_joint_state_to_sim" not in prepare + hard_reset
    assert "restore_canonical_articulation_reset_state" not in prepare + hard_reset
    assert "default_joint_pos" not in prepare + hard_reset


def test_partial_vector_reset_request_fails_before_any_physical_mutation() -> None:
    backend = VectorizedIsaacFSMBackend.__new__(VectorizedIsaacFSMBackend)
    backend.num_envs = 8

    with pytest.raises(VectorizedIsaacBackendError, match="one seed/options row"):
        backend.reset_all(seeds=range(7))
    with pytest.raises(VectorizedIsaacBackendError, match="one seed/options row"):
        backend.reset_all(seeds=range(8), options=({},) * 7)


class _LifecycleView:
    def __init__(self, num_envs: int) -> None:
        self.limits = np.zeros((num_envs, 12, 2), dtype=float)
        self.limits[..., 0] = -1.0
        self.limits[..., 1] = 1.0

    def get_dof_limits(self) -> np.ndarray:
        return self.limits


class _LifecycleRobot:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.is_initialized = True
        self.root_physx_view = _LifecycleView(num_envs)
        self.data = SimpleNamespace(joint_pos=np.zeros((num_envs, 12), dtype=float))
        self.native_state_sha256 = "native-articulation-state"
        self.prohibited_writes = 0

    def reinitialize(self) -> None:
        self.is_initialized = True
        self.root_physx_view = _LifecycleView(self.num_envs)
        self.data = SimpleNamespace(joint_pos=np.zeros((self.num_envs, 12), dtype=float))
        self.native_state_sha256 = "native-articulation-state"

    def write_root_pose_to_sim(self, *_args, **_kwargs) -> None:
        self.prohibited_writes += 1

    def write_root_velocity_to_sim(self, *_args, **_kwargs) -> None:
        self.prohibited_writes += 1

    def write_joint_state_to_sim(self, *_args, **_kwargs) -> None:
        self.prohibited_writes += 1


class _LifecycleSensor:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self._data = SimpleNamespace()
        self.reinitialize()

    def reinitialize(self) -> None:
        self.is_initialized = True
        self.contact_physx_view = object()
        self.body_physx_view = object()
        self._data.force_matrix_w = np.zeros((self.num_envs, 1, 2, 3))
        self._data.force_matrix_w_history = np.zeros(
            (self.num_envs, 3, 1, 2, 3)
        )
        self._data.contact_pos_w = np.full((self.num_envs, 1, 2, 3), np.nan)
        self._data.friction_forces_w = np.zeros((self.num_envs, 1, 2, 3))


class _LifecycleContactBank:
    def __init__(self, num_envs: int) -> None:
        self.sensors = {
            name: _LifecycleSensor(num_envs) for name in SENSED_BODIES
        }

    @property
    def initialized(self) -> bool:
        return all(sensor.is_initialized for sensor in self.sensors.values())


class _LifecycleScene:
    def __init__(self, robot: _LifecycleRobot, bank: _LifecycleContactBank) -> None:
        self.robot = robot
        self.bank = bank
        self.reset_count = 0
        self.update_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def update(self, dt: float) -> None:
        assert dt == 0.0
        self.update_count += 1


class _LifecycleSimulation:
    def __init__(
        self,
        robot: _LifecycleRobot,
        bank: _LifecycleContactBank,
        *,
        callback_failure: BaseException | None = None,
    ) -> None:
        self.robot = robot
        self.bank = bank
        self.callback_failure = callback_failure
        self._disable_app_control_on_stop_handle = False
        self.playing = True
        self.reset_calls: list[bool] = []

    def is_playing(self) -> bool:
        return self.playing

    def reset(self, *, soft: bool) -> None:
        self.reset_calls.append(soft)
        # Model the public IsaacLab transaction: it owns the STOP guard,
        # invalidates every native view, then PLAY callbacks recreate them.
        self._disable_app_control_on_stop_handle = True
        self.playing = False
        self.robot.is_initialized = False
        self.robot.root_physx_view = None
        for sensor in self.bank.sensors.values():
            sensor.is_initialized = False
            sensor.contact_physx_view = None
            sensor.body_physx_view = None
        self.robot.reinitialize()
        for sensor in self.bank.sensors.values():
            sensor.reinitialize()
        self.playing = True
        self._disable_app_control_on_stop_handle = False
        if self.callback_failure is not None:
            import builtins

            builtins.ISAACLAB_CALLBACK_EXCEPTION = self.callback_failure


def _lifecycle_backend(monkeypatch: pytest.MonkeyPatch):
    num_envs = 8
    robot = _LifecycleRobot(num_envs)
    bank = _LifecycleContactBank(num_envs)
    scene = _LifecycleScene(robot, bank)
    simulation = _LifecycleSimulation(robot, bank)
    backend = VectorizedIsaacFSMBackend.__new__(VectorizedIsaacFSMBackend)
    backend.num_envs = num_envs
    backend.device = "cpu"
    backend.robot = robot
    backend.scene = scene
    backend.sim = simulation
    backend.contact_bank = bank
    backend._servo_joint_ids = tuple(range(8))
    backend._canonical_reset_state = SimpleNamespace(
        instance_count=num_envs,
        state_sha256="native-articulation-state",
    )
    backend._native_servo_limit_state = vector_backend_module._native_servo_limit_state(
        robot,
        servo_joint_ids=backend._servo_joint_ids,
        expected_instance_count=num_envs,
    )
    backend._physical_reset_count = 0
    backend._reset_transaction_poisoned = False
    monkeypatch.setattr(
        vector_backend_module,
        "capture_canonical_articulation_reset_state",
        lambda current: SimpleNamespace(
            instance_count=current.num_envs,
            state_sha256=current.native_state_sha256,
        ),
    )
    monkeypatch.setattr(
        __import__("builtins"), "ISAACLAB_CALLBACK_EXCEPTION", None, raising=False
    )
    return backend, robot, scene, simulation


def test_two_reused_reset_cycles_rebuild_all_native_views_and_restore_native_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, robot, scene, simulation = _lifecycle_backend(monkeypatch)

    fresh = backend._prepare_physical_reset()
    assert fresh["physical_reset_generation"] == 1
    assert fresh["physics_lifecycle_reset"] == "scene_factory_reset_before_limit_authoring"

    # Represent direct PhysX limit authoring and a progressed episode.  Each
    # later reset must discard both without any indexed state write.
    for expected_generation in (2, 3):
        robot.root_physx_view.limits[:, :8, :] += float(expected_generation)
        robot.native_state_sha256 = "progressed-episode-state"
        evidence = backend._prepare_physical_reset()
        assert evidence["physical_reset_generation"] == expected_generation
        assert evidence["global_simulation_resets_this_barrier"] == 1
        assert evidence["stop_guard_owned_by"] == "isaaclab.SimulationContext.reset"
        assert evidence["stop_callback_reinitialized_articulation_view"] is True
        assert evidence["stop_callback_reinitialized_articulation_data"] is True
        assert evidence["stop_callback_reinitialized_contact_view_count"] == 13
        assert evidence["stop_callback_reinitialized_contact_body_view_count"] == 13
        assert evidence["stop_callback_reinitialized_contact_buffer_bank_count"] == 13
        assert evidence["stop_callback_reinitialized_contact_buffer_count"] == 52
        assert evidence["root_pose_writes_this_barrier"] == 0
        assert evidence["root_velocity_writes_this_barrier"] == 0
        assert evidence["joint_state_writes_this_barrier"] == 0

    assert simulation.reset_calls == [False, False]
    assert scene.reset_count == 2
    assert scene.update_count == 2
    assert robot.prohibited_writes == 0
    assert backend._reset_transaction_poisoned is False


def test_callback_failure_poisoning_prevents_reset_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, robot, _scene, simulation = _lifecycle_backend(monkeypatch)
    backend._prepare_physical_reset()
    robot.root_physx_view.limits[:, :8, :] += 2.0
    robot.native_state_sha256 = "progressed-episode-state"
    simulation.callback_failure = RuntimeError("contact view callback failed")

    with pytest.raises(VectorizedIsaacBackendError, match="callback failed"):
        backend._prepare_physical_reset()

    assert backend._reset_transaction_poisoned is True


def test_logical_reset_requires_p01_for_controller_and_authoritative_frame() -> None:
    require_p01 = vector_backend_module._require_p01_reset_state

    require_p01(SimpleNamespace(state_id="P01"), row=0, source="controller")
    with pytest.raises(VectorizedIsaacBackendError, match="received 'P02'"):
        require_p01(SimpleNamespace(state_id="P02"), row=3, source="controller")
    with pytest.raises(VectorizedIsaacBackendError, match="received ''"):
        require_p01(SimpleNamespace(), row=7, source="authoritative frame")


def test_batched_adapter_reuses_canonical_servo_limits() -> None:
    source = inspect.getsource(
        __import__(
            "wlr50_clean.ppo.vectorized_isaac_backend", fromlist=["_BatchedCommandAdapter"]
        )._BatchedCommandAdapter.apply_batch
    )

    assert "servo_limits_deg(name)" in source
    assert "(-60.0, 210.0)" not in source
    assert "(-135.0, 135.0)" not in source


def test_ground_plane_spawner_uses_one_concrete_global_prim() -> None:
    """GroundPlaneCfg's locked spawner does not support ENV_REGEX_NS paths."""

    source = inspect.getsource(_create_vector_scene)

    assert 'prim_path="/World/defaultGroundPlane"' in source
    assert 'collision_group=-1' in source
    assert '"{ENV_REGEX_NS}/defaultGroundPlane' not in source
    assert '"{ENV_REGEX_NS}/defaultGroundPlane/GroundPlane/CollisionPlane"' not in source
    assert '"/World/defaultGroundPlane/GroundPlane/CollisionPlane"' in source


@pytest.mark.parametrize(
    ("num_envs", "expected_size"),
    ((8, (30.0, 14.0)), (16, (30.0, 30.0)), (32, (54.0, 38.0))),
)
def test_shared_ground_visual_mesh_covers_every_grid_clone(
    num_envs: int, expected_size: tuple[float, float]
) -> None:
    from wlr50_clean.ppo.vectorized_isaac_backend import _shared_ground_size_m

    assert _shared_ground_size_m(num_envs, 8.0) == expected_size
