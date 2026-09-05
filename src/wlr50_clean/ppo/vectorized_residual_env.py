"""Official RSL-RL adapter for the true cloned-scene Isaac backend.

The physical backend can advance all clones in one call, but it currently has
only a synchronous whole-batch reset.  Consequently, this adapter marks every
row done at a reset barrier when any row terminates; non-terminal peers are
reported as time-limit truncations.  It never emulates vectorization with N
single-scene calls and never claims support for per-row phase snapshots.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .observation_schema_v2 import ObservationSchemaV2, load_observation_schema_v2
from .phase_action_masks_v2 import PhaseActionMasksV2, load_phase_action_masks_v2
from .phase_objectives import DENSE_FAMILIES
from .residual_direct_env import (
    ACTION_DIMENSION,
    DECISION_HZ,
    MAX_EPISODE_DECISIONS,
    PHYSICS_HZ,
    PHYSICS_TICKS_PER_DECISION,
    STATE_IDS,
    ResidualDirectEnvError,
    ResidualEpisodeEnv,
    build_completed_episode_telemetry,
    build_reward_dominance_telemetry,
)
from .reward_v2 import RewardCalculatorV2
from .termination_v2 import TerminationEvaluatorV2, TerminationSignalsV2


VECTOR_BATCH_RESET_BARRIER_REASON = "VECTOR_BATCH_RESET_BARRIER"
ZERO12 = (0.0,) * ACTION_DIMENSION


class VectorizedResidualEnvError(ResidualDirectEnvError):
    """The true-batch training or reset contract cannot be satisfied."""


class VectorizedRslResidualEnv:
    """RSL-RL VecEnv backed by one ``VectorizedIsaacFSMBackend`` instance.

    Each row has an independent residual projection bridge, reward history and
    frozen FSM/controller in the backend.  All rows share only the one Isaac
    physics advance required by true cloned-scene batching.
    """

    def __init__(
        self,
        backend: Any,
        *,
        seeds: Sequence[int],
        device: str = "cuda:0",
        observation_schema: ObservationSchemaV2 | None = None,
        phase_actions: PhaseActionMasksV2 | None = None,
        reward_calculator: RewardCalculatorV2 | None = None,
        termination_evaluator: TerminationEvaluatorV2 | None = None,
        collect_trace: bool = False,
        training_phase_reset_schedule: Sequence[str] | None = None,
    ) -> None:
        import torch

        self.backend = backend
        self.num_envs = int(getattr(backend, "num_envs", 0))
        if self.num_envs <= 1:
            raise VectorizedResidualEnvError(
                "true-batch adapter requires a backend with more than one environment"
            )
        if training_phase_reset_schedule is not None:
            raise VectorizedResidualEnvError(
                "vector backend has no independent phase-snapshot reset; "
                "phase curriculum must use the validated single-scene adapter"
            )
        if not hasattr(backend, "reset_all") or not hasattr(
            backend, "step_physics_batch"
        ):
            raise VectorizedResidualEnvError(
                "backend must provide reset_all and step_physics_batch"
            )
        self.seed_schedule = tuple(int(seed) for seed in seeds)
        if len(self.seed_schedule) < self.num_envs:
            raise VectorizedResidualEnvError(
                "one deterministic reset seed is required per batched environment"
            )
        if any(seed < 0 for seed in self.seed_schedule):
            raise VectorizedResidualEnvError("reset seeds must be non-negative")

        self.device = torch.device(device)
        self.num_actions = ACTION_DIMENSION
        self.max_episode_length = MAX_EPISODE_DECISIONS
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.cfg = {
            "physics_hz": PHYSICS_HZ,
            "decision_hz": DECISION_HZ,
            "physics_ticks_per_decision": PHYSICS_TICKS_PER_DECISION,
            "backend_mode": "one_scene_true_batched_isaac",
            "synchronous_full_batch_reset": True,
            "independent_phase_snapshot_reset": False,
        }
        self.completed_episodes: list[dict[str, Any]] = []
        self.policy_decision_count = 0
        self.phase_decision_counts = {phase_id: 0 for phase_id in STATE_IDS}
        self.reward_family_signed_sums = {
            family: 0.0 for family in DENSE_FAMILIES
        }
        self.reward_family_absolute_sums = {
            family: 0.0 for family in DENSE_FAMILIES
        }
        self.reward_family_absolute_sums_by_phase = {
            phase_id: {family: 0.0 for family in DENSE_FAMILIES}
            for phase_id in STATE_IDS
        }
        self.reward_telemetry_incomplete_count = 0
        self.last_step_infos: tuple[Mapping[str, Any], ...] = ()
        self._seed_cursor = 0
        self._episode_seed_rows: tuple[int, ...] = ()
        self._batch: Any | None = None

        shared_observation = observation_schema or load_observation_schema_v2()
        shared_actions = phase_actions or load_phase_action_masks_v2()
        shared_reward = reward_calculator or RewardCalculatorV2.from_files()
        shared_termination = termination_evaluator or TerminationEvaluatorV2()
        self._maximum_single_reward_family_fraction = float(
            shared_reward.config.maximum_single_dense_family_fraction
        )
        self._maximum_residual_reward_fraction = float(
            shared_reward.config.maximum_residual_regularization_fraction
        )
        self._minimum_absolute_dense_return = float(
            shared_reward.config.minimum_absolute_dense_return
        )
        self.environments = tuple(
            ResidualEpisodeEnv(
                backend,
                observation_schema=shared_observation,
                phase_actions=shared_actions,
                reward_calculator=shared_reward,
                termination_evaluator=shared_termination,
                collect_trace=collect_trace,
            )
            for _ in range(self.num_envs)
        )
        if len({id(env.bridge) for env in self.environments}) != self.num_envs:
            raise VectorizedResidualEnvError("residual projection histories are shared")
        self._reset_full_batch()

    def _next_seed_rows(self) -> tuple[int, ...]:
        result = tuple(
            self.seed_schedule[(self._seed_cursor + row) % len(self.seed_schedule)]
            for row in range(self.num_envs)
        )
        self._seed_cursor += self.num_envs
        return result

    def _validate_independent_backend_state(
        self, frames: Sequence[Any], seeds: Sequence[int]
    ) -> None:
        if len(frames) != self.num_envs or len({id(frame) for frame in frames}) != self.num_envs:
            raise VectorizedResidualEnvError(
                "batched backend must return N independent authoritative frames"
            )
        controllers = tuple(getattr(self.backend, "controllers", ()))
        readers = tuple(getattr(self.backend, "readers", ()))
        if (
            len(controllers) != self.num_envs
            or len({id(value) for value in controllers}) != self.num_envs
        ):
            raise VectorizedResidualEnvError("FSM controller instances are shared or incomplete")
        if (
            len(readers) != self.num_envs
            or len({id(value) for value in readers}) != self.num_envs
        ):
            raise VectorizedResidualEnvError("sensing histories are shared or incomplete")
        for row, (frame, seed) in enumerate(zip(frames, seeds, strict=True)):
            if frame.state_id not in STATE_IDS:
                raise VectorizedResidualEnvError(
                    f"environment {row} began in unknown FSM state {frame.state_id!r}"
                )
            info = frame.info
            if int(info.get("seed", -1)) != int(seed):
                raise VectorizedResidualEnvError(
                    f"environment {row} reset seed metadata is not independent"
                )
            if info.get("independent_fsm_per_environment") is not True:
                raise VectorizedResidualEnvError(
                    f"environment {row} lacks independent FSM attestation"
                )

    def _validate_batch_step(self, previous: Any, current: Any) -> tuple[Any, ...]:
        frames = tuple(current.frames)
        if len(frames) != self.num_envs:
            raise VectorizedResidualEnvError("batched physics result does not contain N rows")
        if int(current.physics_tick) != int(previous.physics_tick) + 1:
            raise VectorizedResidualEnvError(
                "batched backend did not advance exactly one local physics tick"
            )
        for row, (before, after) in enumerate(
            zip(previous.frames, frames, strict=True)
        ):
            if after.physics_tick != before.physics_tick + 1:
                raise VectorizedResidualEnvError(
                    f"environment {row} did not advance exactly one physics tick"
                )
            if not math.isclose(
                float(after.sim_time_s) - float(before.sim_time_s),
                1.0 / PHYSICS_HZ,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise VectorizedResidualEnvError(
                    f"environment {row} did not preserve the 120 Hz clock"
                )
        counters = (
            "global_physics_step_count",
            "batched_articulation_write_count",
            "exact_pair_capture_count",
        )
        for name in counters:
            if int(getattr(current, name)) != int(getattr(previous, name)) + 1:
                raise VectorizedResidualEnvError(
                    f"true-batch attestation counter {name} did not advance once"
                )
        return frames

    def _reset_full_batch(self) -> None:
        seeds = self._next_seed_rows()
        # Deliberately pass no options. The backend supports only a synchronized
        # nominal reset; phase snapshots and per-row mutation are fail-closed.
        batch = self.backend.reset_all(seeds=seeds)
        frames = tuple(batch.frames)
        self._validate_independent_backend_state(frames, seeds)
        self._batch = batch
        self._episode_seed_rows = seeds
        observations = []
        for row_env, frame, seed in zip(
            self.environments, frames, seeds, strict=True
        ):
            row_env.seed = int(seed)
            row_env.frame = frame
            row_env.bridge.reset(state_id=frame.state_id)
            row_env.previous_residual = ZERO12
            row_env.previous_previous_residual = ZERO12
            row_env.decision_count = 0
            row_env.done = False
            row_env.trace = []
            row_env.observation = row_env._encode(frame)
            observations.append(row_env.observation)
        self.episode_length_buf.zero_()
        import torch

        self._observations = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )

    def get_observations(self) -> Any:
        from tensordict import TensorDict

        return TensorDict(
            {"policy": self._observations, "critic": self._observations},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        import torch

        if self._batch is None:
            raise VectorizedResidualEnvError("full-batch reset must precede stepping")
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise VectorizedResidualEnvError(
                f"actions must have shape {(self.num_envs, self.num_actions)}"
            )
        raw_actions = tuple(
            tuple(float(value) for value in row)
            for row in actions.detach().to("cpu").tolist()
        )
        if any(
            len(row) != ACTION_DIMENSION
            or any(not math.isfinite(value) for value in row)
            for row in raw_actions
        ):
            raise VectorizedResidualEnvError("policy actions must be finite Full12 rows")

        start_frames = tuple(self._batch.frames)
        projections: list[list[Any]] = [[] for _ in range(self.num_envs)]
        # Keep the exact nominal command paired with every projection.  The
        # final row is emitted below so the live vector smoke gate can prove
        # zero-residual bit identity without trying to reconstruct an FSM
        # command after the physics/controller clock has advanced.
        projection_nominals: list[list[tuple[float, ...]]] = [
            [] for _ in range(self.num_envs)
        ]
        transition_metrics: list[list[Mapping[str, Any]]] = [
            [] for _ in range(self.num_envs)
        ]
        decisions: list[Any | None] = [None for _ in range(self.num_envs)]
        controller_blocked = [False for _ in range(self.num_envs)]
        ticks_executed = 0
        for _ in range(PHYSICS_TICKS_PER_DECISION):
            previous_batch = self._batch
            projected_actions = []
            for row, (row_env, frame, raw) in enumerate(
                zip(
                    self.environments,
                    previous_batch.frames,
                    raw_actions,
                    strict=True,
                )
            ):
                projected = row_env.bridge.project_tick(
                    raw,
                    state_id=frame.state_id,
                    nominal_action_full12=frame.nominal_action_full12,
                    reference_action_full12=frame.reference_action_full12,
                    reference_delta_full12=frame.reference_delta_full12,
                    runtime_action_mask_full12=(1,) * ACTION_DIMENSION,
                    safety=frame.safety_projection,
                    dt_s=1.0 / PHYSICS_HZ,
                )
                projections[row].append(projected.projection)
                projection_nominals[row].append(
                    tuple(float(value) for value in frame.nominal_action_full12)
                )
                if projected.transition_metric is not None:
                    transition_metrics[row].append(
                        projected.transition_metric.as_dict()
                    )
                projected_actions.append(
                    projected.projection.applied_action_full12
                )
            current_batch = self.backend.step_physics_batch(projected_actions)
            frames = self._validate_batch_step(previous_batch, current_batch)
            self._batch = current_batch
            ticks_executed += 1
            stop_at_barrier = False
            for row, (row_env, frame) in enumerate(
                zip(self.environments, frames, strict=True)
            ):
                row_env.frame = frame
                source = frame.termination_signals
                signals = TerminationSignalsV2(
                    authoritative_success=source.success,
                    body_collision=source.body_collision,
                    wheel_only_climb=source.wheel_only_climb,
                    fall=source.fall,
                    nan_inf=source.nan_inf,
                    hard_joint_limit=source.hard_joint_limit,
                    physics_explosion=source.physics_explosion,
                    reference_conformance_outside_30pct=(
                        source.reference_conformance_outside_30pct
                    ),
                )
                decisions[row] = row_env.termination_evaluator.evaluate(
                    signals, episode_time_s=frame.sim_time_s
                )
                controller_blocked[row] = (
                    frame.info.get("controller_task_result")
                    == "INCOMPLETE_CONTROLLER_BLOCKED"
                )
                decision = decisions[row]
                stop_at_barrier = bool(
                    stop_at_barrier
                    or decision.terminated
                    or decision.truncated
                    or controller_blocked[row]
                )
            if stop_at_barrier:
                break

        if not all(decision is not None for decision in decisions):
            raise VectorizedResidualEnvError("batched termination decisions are incomplete")
        end_frames = tuple(self._batch.frames)
        physical_done = tuple(
            bool(decision.terminated or decision.truncated or blocked)
            for decision, blocked in zip(decisions, controller_blocked, strict=True)
        )
        batch_barrier = any(physical_done)
        rewards = []
        time_outs = []
        infos: list[dict[str, Any]] = []
        next_observations = []
        for row, (row_env, start, end, raw, decision, blocked) in enumerate(
            zip(
                self.environments,
                start_frames,
                end_frames,
                raw_actions,
                decisions,
                controller_blocked,
                strict=True,
            )
        ):
            projection = projections[row][-1]
            reward = row_env._reward(
                start,
                end,
                projection.safe_projected_residual_full12,
                termination_reason=decision.reason,
                controller_blocked=blocked,
            )
            rewards.append(reward.total)
            self.policy_decision_count += 1
            self.phase_decision_counts[str(start.state_id)] += 1
            weighted = getattr(reward, "weighted_dense", None)
            if isinstance(weighted, Mapping) and tuple(weighted) == DENSE_FAMILIES:
                for family in DENSE_FAMILIES:
                    value = float(weighted[family])
                    self.reward_family_signed_sums[family] += value
                    self.reward_family_absolute_sums[family] += abs(value)
                    self.reward_family_absolute_sums_by_phase[str(start.state_id)][
                        family
                    ] += abs(value)
            else:
                # Lightweight injected test kernels may intentionally return
                # only ``total``.  Production training fails closed in the CLI
                # if any rollout decision lacks the five-family breakdown.
                self.reward_telemetry_incomplete_count += 1
            # Controller-blocked is a hard task failure.  Do not expose it as
            # a timeout, otherwise RSL bootstraps across a failed trajectory.
            terminated = bool(decision.terminated or blocked)
            intrinsic_truncation = bool(not terminated and decision.truncated)
            reset_peer = bool(batch_barrier and not (terminated or intrinsic_truncation))
            truncated = bool(intrinsic_truncation or reset_peer)
            if blocked:
                reason = "CONTROLLER_BLOCKED"
            elif decision.reason is not None:
                reason = decision.reason.value
            elif reset_peer:
                reason = VECTOR_BATCH_RESET_BARRIER_REASON
            else:
                reason = None
            residual = projection.safe_projected_residual_full12
            row_env.previous_previous_residual = row_env.previous_residual
            row_env.previous_residual = residual
            row_env.decision_count += 1
            row_env.done = batch_barrier
            row_env.observation = row_env._encode(end)
            info = {
                **dict(end.info),
                "seed": self._episode_seed_rows[row],
                "physics_tick": end.physics_tick,
                "sim_time_s": end.sim_time_s,
                "decision_index": row_env.decision_count - 1,
                "physics_ticks_executed": ticks_executed,
                "raw_policy_action_full12": list(raw),
                "projection_state_id": projection.state_id,
                "projection_nominal_action_full12": list(
                    projection_nominals[row][-1]
                ),
                "projected_residual_full12": list(residual),
                "applied_action_full12": list(projection.applied_action_full12),
                "effective_action_mask_full12": list(
                    projection.effective_action_mask_full12
                ),
                "zero_residual_fast_path": bool(
                    projection.zero_residual_fast_path
                ),
                "projection_clipping_stages": list(projection.clipping_stages),
                "phase_transition_action_jump": transition_metrics[row],
                "reward": asdict(reward),
                "termination_reason": reason,
                "terminated": terminated,
                "truncated": truncated,
                "vector_batch_reset_barrier": batch_barrier,
                "vector_batch_reset_peer": reset_peer,
                "in_episode_root_write_count": int(
                    end.info.get("in_episode_root_pose_writes", 0)
                )
                + int(end.info.get("in_episode_root_velocity_writes", 0)),
                "recording_runtime_access_count": int(
                    end.info.get("recording_accesses", 0)
                ),
            }
            if info["in_episode_root_write_count"] != 0:
                raise VectorizedResidualEnvError("FORBIDDEN_IN_EPISODE_ROOT_WRITE")
            if info["recording_runtime_access_count"] != 0:
                raise VectorizedResidualEnvError(
                    "Recording runtime access is forbidden"
                )
            if row_env.collect_trace:
                row_env.trace.append(row_env._trace_row(end, reward, info))
            infos.append(info)
            next_observations.append(row_env.observation)
            time_outs.append(truncated)

        self.episode_length_buf += 1
        dones = [batch_barrier for _ in range(self.num_envs)]
        if batch_barrier:
            for row, (row_env, info) in enumerate(
                zip(self.environments, infos, strict=True)
            ):
                self.completed_episodes.append(
                    {
                        "env_index": row,
                        "seed": self._episode_seed_rows[row],
                        "length": int(self.episode_length_buf[row].item()),
                        "duration_s": float(info["sim_time_s"]),
                        "termination_reason": info["termination_reason"],
                        "trace": list(row_env.trace),
                        "vector_batch_reset_peer": info[
                            "vector_batch_reset_peer"
                        ],
                    }
                )
            self._reset_full_batch()
        else:
            self._observations = torch.tensor(
                next_observations, dtype=torch.float32, device=self.device
            )
        self.last_step_infos = tuple(infos)
        reward_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
        extras = {
            "time_outs": torch.tensor(
                time_outs, dtype=torch.bool, device=self.device
            ),
            "log": {},
        }
        return self.get_observations(), reward_tensor, done_tensor, extras

    def training_telemetry(self) -> Mapping[str, Any]:
        """Return aggregate phase occupancy and five-family reward evidence."""

        total = int(self.policy_decision_count)
        return {
            "schema": "wlr50_clean.ppo_training_telemetry.v1",
            "policy_decision_count": total,
            "phase_decision_counts": dict(self.phase_decision_counts),
            "phase_occupancy_fraction": {
                phase_id: (
                    self.phase_decision_counts[phase_id] / total if total else 0.0
                )
                for phase_id in STATE_IDS
            },
            "phase_curriculum_target_decision_fraction": None,
            "phase_curriculum_occupancy_absolute_error": None,
            "phase_curriculum_occupancy_tolerance_fraction": None,
            "phase_curriculum_occupancy_violations": [],
            "phase_curriculum_occupancy_within_tolerance": None,
            **build_reward_dominance_telemetry(
                signed_sums=self.reward_family_signed_sums,
                absolute_sums=self.reward_family_absolute_sums,
                absolute_sums_by_phase=self.reward_family_absolute_sums_by_phase,
                incomplete_count=self.reward_telemetry_incomplete_count,
                maximum_single_family_fraction=(
                    self._maximum_single_reward_family_fraction
                ),
                maximum_residual_regularization_fraction=(
                    self._maximum_residual_reward_fraction
                ),
                minimum_absolute_dense_return=self._minimum_absolute_dense_return,
            ),
            **build_completed_episode_telemetry(self.completed_episodes),
            "completed_sample_count": len(self.completed_episodes),
        }


__all__ = [
    "VECTOR_BATCH_RESET_BARRIER_REASON",
    "VectorizedResidualEnvError",
    "VectorizedRslResidualEnv",
]
