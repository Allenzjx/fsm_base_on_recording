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
