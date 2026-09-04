from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo.action_projection import SafetyProjection
from wlr50_clean.ppo.residual_direct_env import ResidualEpisodeEnv
from wlr50_clean.ppo.termination import (
    TerminationDecision,
    TerminationReason,
    TerminationSignals,
)
from wlr50_clean.ppo.vectorized_isaac_backend import (
    BatchedAuthoritativeFrame,
    VectorizedIsaacFSMBackend,
)
from wlr50_clean.ppo.vectorized_residual_env import VectorizedRslResidualEnv
from wlr50_clean.ppo.vectorized_smoke_evidence import (
    NONZERO_SMOKE_STATUS,
    ZERO_SMOKE_STATUS,
    VectorizedSmokeEvidenceError,
    collect_vectorized_residual_smoke_evidence,
    deterministic_nonzero_action_rows,
)


ZERO12 = (0.0,) * 12
NUM_ENVS = 8


@dataclass(frozen=True)
class _Reward:
    total: float


class _FakeVectorBackend(VectorizedIsaacFSMBackend):
    def __init__(
        self,
        *,
        num_envs: int = NUM_ENVS,
        terminal_tick: int | None = None,
        controller_blocked_tick: int | None = None,
        duplicate_origins: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.controllers = tuple(object() for _ in range(num_envs))
        self.readers = tuple(object() for _ in range(num_envs))
        self.terminal_tick = terminal_tick
        self.controller_blocked_tick = controller_blocked_tick
        self.duplicate_origins = duplicate_origins
        self.tick = 0
        self.global_steps = 0
        self.writes = 0
        self.captures = 0
        self.seeds = tuple(range(num_envs))
        self.action_batches: list[tuple[tuple[float, ...], ...]] = []

    def _frames(self):
        frames = []
        for row in range(self.num_envs):
            origin_x = 0.0 if self.duplicate_origins else 8.0 * row
            frames.append(
                SimpleNamespace(
                    physics_tick=self.tick,
                    sim_time_s=self.tick / 120.0,
                    state_id=f"P{row + 1:02d}",
                    macro_phase=row + 1,
                    phase_progress=self.tick / 100.0,
                    nominal_action_full12=ZERO12,
                    reference_action_full12=ZERO12,
                    reference_delta_full12=ZERO12,
                    safety_projection=SafetyProjection(),
                    termination_signals=TerminationSignals(
                        success=bool(
                            row == 0
                            and self.terminal_tick is not None
                            and self.tick >= self.terminal_tick
                        )
                    ),
                    info={
                        "schema": "wlr50_clean.vectorized_isaac_backend.reset.v1",
                        "seed": self.seeds[row],
                        "env_index": row,
                        "env_origin_w_m": [origin_x, 0.0, 0.0],
                        "num_envs": self.num_envs,
                        "one_global_physics_step_per_tick": True,
                        "one_batched_articulation_write_per_tick": True,
                        "exact_pair_contact_fail_closed": True,
                        "independent_fsm_per_environment": True,
                        "in_episode_root_pose_writes": 0,
                        "in_episode_root_velocity_writes": 0,
                        "in_episode_force_or_impulse_writes": 0,
                        "in_episode_gravity_writes": 0,
                        "recording_accesses": 0,
                        "controller_task_result": (
                            "INCOMPLETE_CONTROLLER_BLOCKED"
                            if row == 0
                            and self.controller_blocked_tick is not None
                            and self.tick >= self.controller_blocked_tick
                            else None
                        ),
                    },
                )
            )
        return tuple(frames)

    def _batch(self):
        return BatchedAuthoritativeFrame(
            physics_tick=self.tick,
            sim_time_s=self.tick / 120.0,
            frames=self._frames(),
            global_physics_step_count=self.global_steps,
            batched_articulation_write_count=self.writes,
            exact_pair_capture_count=self.captures,
        )

    def reset_all(self, *, seeds):
        self.seeds = tuple(int(seed) for seed in seeds)
        self.tick = 0
        self.global_steps += 180
        self.writes = 180
        self.captures = 180
        return self._batch()

    def step_physics_batch(self, applied_actions_full12):
        actions = tuple(
            tuple(float(value) for value in row)
            for row in applied_actions_full12
        )
        self.action_batches.append(actions)
        self.tick += 1
        self.global_steps += 1
        self.writes += 1
        self.captures += 1
        return self._batch()


@pytest.fixture
def lightweight_row_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ResidualEpisodeEnv,
        "_encode",
        lambda self, frame: (float(frame.info["env_index"]),) * 125,
    )
    monkeypatch.setattr(
        ResidualEpisodeEnv,
        "_reward",
        lambda self, start, end, residual: _Reward(total=float(end.info["env_index"])),
    )


def _environment(backend: _FakeVectorBackend) -> VectorizedRslResidualEnv:
    pytest.importorskip("torch")
    return VectorizedRslResidualEnv(
        backend,
        seeds=tuple(range(1001, 1001 + 2 * backend.num_envs)),
        device="cpu",
    )


def test_deterministic_nonzero_rows_are_distinct_and_strictly_below_five_percent() -> None:
    first = deterministic_nonzero_action_rows(NUM_ENVS, decision_index=3)
    second = deterministic_nonzero_action_rows(NUM_ENVS, decision_index=3)

    assert first == second
    assert len(set(first)) == NUM_ENVS
    assert all(0.0 < abs(value) < 0.05 for row in first for value in row)
    assert first != deterministic_nonzero_action_rows(NUM_ENVS, decision_index=4)
    with pytest.raises(VectorizedSmokeEvidenceError, match="strictly between"):
        deterministic_nonzero_action_rows(
            NUM_ENVS, maximum_phase_scale_fraction=0.05
        )


def test_zero_smoke_proves_exact_identity_timing_and_independence(
    lightweight_row_kernel,
) -> None:
    backend = _FakeVectorBackend()
    report = collect_vectorized_residual_smoke_evidence(
        _environment(backend), mode="zero", policy_decisions=2
    )

    assert report.status == ZERO_SMOKE_STATUS
    assert report.passed is True
    assert report.row_evidence_count == NUM_ENVS * 2
    assert report.measured_physics_ticks == 16
    assert report.global_physics_steps == 16
    assert report.batched_articulation_writes == 16
    assert report.exact_pair_captures == 16
    assert report.independent_origin_count == NUM_ENVS
    assert report.independent_controller_count == NUM_ENVS
    assert report.independent_reader_count == NUM_ENVS
    assert report.independent_projection_bridge_count == NUM_ENVS
    assert report.zero_applied_equals_nominal_row_count == NUM_ENVS * 2
    assert report.nonzero_active_row_count == 0
    assert report.maximum_observed_phase_scale_fraction == 0.0
    assert all(row.zero_residual_fast_path for row in report.rows)
    assert all(
        row.applied_action_full12 == row.nominal_action_full12 for row in report.rows
    )
    assert len(backend.action_batches) == 16
    assert all(
        all(action == ZERO12 for action in batch) for batch in backend.action_batches
    )
    assert json.loads(json.dumps(report.as_dict()))["passed"] is True


def test_nonzero_smoke_proves_distinct_small_masked_activity_per_row(
    lightweight_row_kernel,
) -> None:
    backend = _FakeVectorBackend()
    report = collect_vectorized_residual_smoke_evidence(
        _environment(backend), mode="nonzero", policy_decisions=2
    )

    assert report.status == NONZERO_SMOKE_STATUS
    assert report.passed is True
    assert report.deterministic_distinct_action_rows is True
    assert report.nonzero_active_row_count == NUM_ENVS * 2
    assert 0.0 < report.maximum_observed_phase_scale_fraction < 0.05
    assert report.all_masks_honored is True
    assert report.zero_applied_equals_nominal_row_count == 0
    assert all(not row.zero_residual_fast_path for row in report.rows)
    assert all(row.active_nonzero_channel_count > 0 for row in report.rows)
    assert all(row.max_abs_phase_scale_fraction < 0.05 for row in report.rows)
    for row in report.rows:
        for residual, mask in zip(
            row.projected_residual_full12,
            row.effective_action_mask_full12,
            strict=True,
        ):
            if not mask:
                assert residual == 0.0


@pytest.mark.parametrize("failure", ["origins", "controllers"])
def test_smoke_rejects_shared_physical_or_python_state(
    lightweight_row_kernel,
    failure: str,
) -> None:
    backend = _FakeVectorBackend(duplicate_origins=failure == "origins")
    env = _environment(backend)
    if failure == "controllers":
        shared = object()
        backend.controllers = (shared,) * NUM_ENVS

    with pytest.raises(VectorizedSmokeEvidenceError, match="shared|duplicated"):
        collect_vectorized_residual_smoke_evidence(env, mode="zero")


def test_smoke_rejects_termination_and_tampered_zero_fast_path(
    lightweight_row_kernel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_env = _environment(_FakeVectorBackend(terminal_tick=3))
    with pytest.raises(VectorizedSmokeEvidenceError, match="done row"):
        collect_vectorized_residual_smoke_evidence(
            terminal_env, mode="zero", policy_decisions=1
        )

    clean_env = _environment(_FakeVectorBackend())
    original_step = clean_env.step

    def tampered_step(actions):
        result = original_step(actions)
        clean_env.last_step_infos[0]["zero_residual_fast_path"] = False
        return result

    monkeypatch.setattr(clean_env, "step", tampered_step)
    with pytest.raises(VectorizedSmokeEvidenceError, match="exact zero-residual"):
        collect_vectorized_residual_smoke_evidence(
            clean_env, mode="zero", policy_decisions=1
        )


def test_true_vector_controller_blocked_beats_same_tick_timeout_without_bootstrap(
    lightweight_row_kernel,
) -> None:
    class _TimeoutAtBlockedTick:
        def evaluate(self, signals, *, episode_time_s: float) -> TerminationDecision:
            timed_out = episode_time_s >= 3.0 / 120.0
            return TerminationDecision(
                terminated=False,
                truncated=timed_out,
                reason=TerminationReason.TIMEOUT if timed_out else None,
                triggered_reasons=(TerminationReason.TIMEOUT,) if timed_out else (),
                diagnostics=(),
            )

    torch = pytest.importorskip("torch")
    backend = _FakeVectorBackend(controller_blocked_tick=3)
    env = VectorizedRslResidualEnv(
        backend,
        seeds=tuple(range(1001, 1001 + 2 * backend.num_envs)),
        device="cpu",
        termination_evaluator=_TimeoutAtBlockedTick(),
    )

    _, _, dones, extras = env.step(torch.zeros((NUM_ENVS, 12)))

    assert dones.tolist() == [True] * NUM_ENVS
    assert extras["time_outs"].tolist()[0] is False
    assert env.last_step_infos[0]["terminated"] is True
    assert env.last_step_infos[0]["truncated"] is False
    assert env.last_step_infos[0]["termination_reason"] == "CONTROLLER_BLOCKED"
