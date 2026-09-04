"""Fail-closed live-smoke evidence for the true batched RSL-RL adapter.

The functions in this module do not launch Isaac.  A caller supplies an
already constructed :class:`VectorizedRslResidualEnv`; the collector advances
it for a short, deterministic smoke and returns JSON-ready evidence only after
every timing, isolation, action-projection, and safety invariant has passed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from .action_projection import full12_bytes
from .phase_action_masks_v2 import PhaseActionMasksV2
from .residual_direct_env import (
    ACTION_DIMENSION,
    DECISION_HZ,
    PHYSICS_HZ,
    PHYSICS_TICKS_PER_DECISION,
    STATE_IDS,
)
from .vectorized_isaac_backend import VectorizedIsaacFSMBackend
from .vectorized_residual_env import VectorizedRslResidualEnv


VECTOR_SMOKE_EVIDENCE_SCHEMA = "wlr50_clean.vectorized_residual_smoke.v1"
ZERO_SMOKE_STATUS = "VECTOR_ZERO_RESIDUAL_SMOKE_PASSED"
NONZERO_SMOKE_STATUS = "VECTOR_NONZERO_RESIDUAL_SMOKE_PASSED"
MAXIMUM_NONZERO_PHASE_SCALE_FRACTION = 0.05
DEFAULT_NONZERO_PHASE_SCALE_FRACTION = 0.04
SmokeMode = Literal["zero", "nonzero"]


class VectorizedSmokeEvidenceError(RuntimeError):
    """The supplied live vector smoke did not prove every required invariant."""


@dataclass(frozen=True, slots=True)
class VectorizedSmokeRowEvidence:
    mode: str
    decision_index: int
    env_index: int
    seed: int
    phase: str
    physics_tick: int
    sim_time_s: float
    raw_policy_action_full12: tuple[float, ...]
    nominal_action_full12: tuple[float, ...]
    projected_residual_full12: tuple[float, ...]
    applied_action_full12: tuple[float, ...]
    effective_action_mask_full12: tuple[int, ...]
    physical_phase_scale_full12: tuple[float, ...]
    active_nonzero_channel_count: int
    max_abs_phase_scale_fraction: float
    zero_residual_fast_path: bool
    projection_clipping_stages: tuple[str, ...]
    in_episode_root_write_count: int
    recording_runtime_access_count: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class VectorizedSmokeEvidence:
    schema: str
    status: str
    mode: str
    passed: bool
    num_envs: int
    policy_decisions: int
    row_evidence_count: int
    physics_hz: float
    decision_hz: float
    physics_ticks_per_decision: int
    measured_physics_ticks: int
    global_physics_steps: int
    batched_articulation_writes: int
    exact_pair_captures: int
    independent_origin_count: int
    independent_controller_count: int
    independent_reader_count: int
    independent_projection_bridge_count: int
    live_vectorized_isaac_backend_verified: bool
    deterministic_distinct_action_rows: bool
    zero_applied_equals_nominal_row_count: int
    nonzero_active_row_count: int
    maximum_observed_phase_scale_fraction: float
    all_masks_honored: bool
    all_zero_fast_path_expected: bool
    no_in_episode_root_writes: bool
    no_recording_runtime_access: bool
    no_termination_or_safety_events: bool
    environment_origins_w_m: tuple[tuple[float, float, float], ...]
    rows: tuple[VectorizedSmokeRowEvidence, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise VectorizedSmokeEvidenceError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise VectorizedSmokeEvidenceError(
            f"{label} must be a positive integer"
        ) from exc
    if result <= 0 or result != value:
        raise VectorizedSmokeEvidenceError(f"{label} must be a positive integer")
    return result


def _finite_full12(values: Any, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise VectorizedSmokeEvidenceError(f"{label} must contain 12 finite values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise VectorizedSmokeEvidenceError(
            f"{label} must contain 12 finite values"
        ) from exc
    if len(result) != ACTION_DIMENSION or any(
        not math.isfinite(value) for value in result
    ):
        raise VectorizedSmokeEvidenceError(f"{label} must contain 12 finite values")
    return result


def _binary_full12(values: Any, label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise VectorizedSmokeEvidenceError(
            f"{label} must contain 12 binary values"
        ) from exc
    if len(result) != ACTION_DIMENSION or any(value not in (0, 1) for value in result):
        raise VectorizedSmokeEvidenceError(f"{label} must contain 12 binary values")
    return result


def deterministic_nonzero_action_rows(
    num_envs: int,
    *,
    decision_index: int = 0,
    maximum_phase_scale_fraction: float = DEFAULT_NONZERO_PHASE_SCALE_FRACTION,
) -> tuple[tuple[float, ...], ...]:
    """Return distinct deterministic normalized residual rows below five percent.

    The action projector applies ``tanh(raw) * physical_phase_scale``.  Every
    raw magnitude is strictly below ``maximum_phase_scale_fraction``, so its
    tanh-bounded physical target is smaller still.
    """

    count = _positive_integer(num_envs, "num_envs")
    index = int(decision_index)
    if index < 0 or index != decision_index:
        raise VectorizedSmokeEvidenceError(
            "decision_index must be a non-negative integer"
        )
    fraction = float(maximum_phase_scale_fraction)
    if not math.isfinite(fraction) or not (
        0.0 < fraction < MAXIMUM_NONZERO_PHASE_SCALE_FRACTION
    ):
        raise VectorizedSmokeEvidenceError(
            "maximum_phase_scale_fraction must be finite and strictly between 0 and 0.05"
        )
    rows = []
    for row in range(count):
        # The row-dependent magnitude proves that cloned environments did not
        # receive a broadcast scalar.  Alternating signs exercise both sides
        # of every enabled channel without introducing randomness.
        magnitude = fraction * (0.50 + 0.45 * (row + 1) / (count + 1))
        rows.append(
            tuple(
                magnitude if (row + channel + index) % 2 == 0 else -magnitude
                for channel in range(ACTION_DIMENSION)
            )
        )
    result = tuple(rows)
    if len(set(result)) != count:
        raise VectorizedSmokeEvidenceError(
            "deterministic nonzero action rows are not distinct"
        )
    return result


def _origins_and_identity_groups(
    env: VectorizedRslResidualEnv,
    num_envs: int,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]:
    backend = getattr(env, "backend", None)
    batch = getattr(env, "_batch", None)
    frames = tuple(getattr(batch, "frames", ()))
    if not isinstance(backend, VectorizedIsaacFSMBackend) or len(frames) != num_envs:
        raise VectorizedSmokeEvidenceError(
            "VectorizedRslResidualEnv is not backed by VectorizedIsaacFSMBackend "
            "or has no complete reset batch"
        )
    origins = []
    for row, frame in enumerate(frames):
        info = getattr(frame, "info", None)
        if not isinstance(info, Mapping) or int(info.get("env_index", -1)) != row:
            raise VectorizedSmokeEvidenceError(
                f"environment {row} lacks its independent reset metadata"
            )
        if (
            info.get("schema") != "wlr50_clean.vectorized_isaac_backend.reset.v1"
            or int(info.get("num_envs", -1)) != num_envs
            or info.get("one_global_physics_step_per_tick") is not True
            or info.get("one_batched_articulation_write_per_tick") is not True
            or info.get("exact_pair_contact_fail_closed") is not True
            or info.get("independent_fsm_per_environment") is not True
        ):
            raise VectorizedSmokeEvidenceError(
                f"environment {row} lacks live vector-backend attestations"
            )
        try:
            origin = tuple(float(value) for value in info["env_origin_w_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorizedSmokeEvidenceError(
                f"environment {row} lacks a finite world origin"
            ) from exc
        if len(origin) != 3 or any(not math.isfinite(value) for value in origin):
            raise VectorizedSmokeEvidenceError(
                f"environment {row} lacks a finite world origin"
            )
        origins.append(origin)
    origin_rows = tuple(origins)
    if len(set(origin_rows)) != num_envs:
        raise VectorizedSmokeEvidenceError(
            "cloned environment origins are shared or duplicated"
        )

    controllers = tuple(getattr(backend, "controllers", ()))
    readers = tuple(getattr(backend, "readers", ()))
    row_envs = tuple(getattr(env, "environments", ()))
    bridges = tuple(getattr(row_env, "bridge", None) for row_env in row_envs)
    groups = {
        "FSM controllers": controllers,
        "sensor readers": readers,
        "residual projection bridges": bridges,
    }
    identities = []
    for label, values in groups.items():
        ids = tuple(id(value) for value in values)
        if (
            len(values) != num_envs
            or any(value is None for value in values)
            or len(set(ids)) != num_envs
        ):
            raise VectorizedSmokeEvidenceError(
                f"{label} are shared, missing, or incomplete"
            )
        identities.append(ids)
    return origin_rows, tuple(identities)  # type: ignore[return-value]


def _batch_counters(batch: Any) -> tuple[int, int, int, int, float]:
    try:
        values = (
            int(batch.global_physics_step_count),
            int(batch.batched_articulation_write_count),
            int(batch.exact_pair_capture_count),
            int(batch.physics_tick),
            float(batch.sim_time_s),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise VectorizedSmokeEvidenceError(
            "batched backend counters are incomplete"
        ) from exc
    if any(value < 0 for value in values[:4]) or not math.isfinite(values[4]):
        raise VectorizedSmokeEvidenceError("batched backend counters are invalid")
    return values


def _assert_finite_tensor(tensor: Any, shape: tuple[int, ...], label: str) -> None:
    try:
        actual = tuple(int(value) for value in tensor.shape)
        finite = bool(tensor.isfinite().all().item())
    except (AttributeError, TypeError, ValueError) as exc:
        raise VectorizedSmokeEvidenceError(f"{label} is not a finite tensor") from exc
    if actual != shape or not finite:
        raise VectorizedSmokeEvidenceError(
            f"{label} must have shape {shape} and contain only finite values"
        )


def _assert_no_frame_events(env: VectorizedRslResidualEnv, num_envs: int) -> None:
    frames = tuple(getattr(getattr(env, "_batch", None), "frames", ()))
    if len(frames) != num_envs:
        raise VectorizedSmokeEvidenceError("post-step authoritative frames are incomplete")
    fields = (
        "success",
        "body_collision",
        "wheel_only_climb",
        "fall",
        "nan_inf",
        "hard_joint_limit",
        "physics_explosion",
    )
    for row, frame in enumerate(frames):
        signals = getattr(frame, "termination_signals", None)
        if signals is None or any(bool(getattr(signals, name, True)) for name in fields):
            raise VectorizedSmokeEvidenceError(
                f"environment {row} emitted a termination or safety event"
            )
        safety = getattr(frame, "safety_projection", None)
        if safety is None or not bool(getattr(safety, "neutral", False)):
            raise VectorizedSmokeEvidenceError(
                f"environment {row} activated a safety projection"
            )


def _row_evidence(
    *,
    info: Mapping[str, Any],
    expected_action: tuple[float, ...],
    mode: SmokeMode,
    decision_index: int,
    env_index: int,
    phase_actions: PhaseActionMasksV2,
) -> VectorizedSmokeRowEvidence:
    if int(info.get("env_index", -1)) != env_index:
        raise VectorizedSmokeEvidenceError(
            f"smoke row {env_index} has mismatched environment metadata"
        )
    if int(info.get("decision_index", -1)) != decision_index:
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} decision counter is not 15 Hz sequential"
        )
    if int(info.get("physics_ticks_executed", -1)) != PHYSICS_TICKS_PER_DECISION:
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} did not execute eight physics ticks"
        )
    phase = str(info.get("projection_state_id", ""))
    if phase not in STATE_IDS:
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} projected an unknown phase {phase!r}"
        )
    raw = _finite_full12(
        info.get("raw_policy_action_full12"),
        f"environment {env_index} raw action",
    )
    if full12_bytes(raw) != full12_bytes(expected_action):
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} did not preserve its deterministic action row"
        )
    nominal = _finite_full12(
        info.get("projection_nominal_action_full12"),
        f"environment {env_index} nominal action",
    )
    residual = _finite_full12(
        info.get("projected_residual_full12"),
        f"environment {env_index} projected residual",
    )
    applied = _finite_full12(
        info.get("applied_action_full12"),
        f"environment {env_index} applied action",
    )
    mask = _binary_full12(
        info.get("effective_action_mask_full12"),
        f"environment {env_index} effective mask",
    )
    expected_mask = phase_actions.mask_for(phase)
    scale = phase_actions.physical_scale_for(phase)
    if mask != expected_mask:
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} effective mask differs from phase config"
        )
    root_writes = int(info.get("in_episode_root_write_count", -1))
    recording_accesses = int(info.get("recording_runtime_access_count", -1))
    if root_writes != 0:
        raise VectorizedSmokeEvidenceError("FORBIDDEN_IN_EPISODE_ROOT_WRITE")
    if recording_accesses != 0:
        raise VectorizedSmokeEvidenceError("Recording runtime access is forbidden")
    for name in ("in_episode_force_or_impulse_writes", "in_episode_gravity_writes"):
        if int(info.get(name, 0)) != 0:
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} recorded forbidden {name}"
            )
    if (
        bool(info.get("terminated", True))
        or bool(info.get("truncated", True))
        or info.get("termination_reason") is not None
        or bool(info.get("vector_batch_reset_barrier", True))
        or bool(info.get("vector_batch_reset_peer", True))
    ):
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} terminated or hit a reset barrier"
        )
    stages = tuple(str(value) for value in info.get("projection_clipping_stages", ()))
    if any("safety" in stage for stage in stages):
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} activated a safety clipping stage"
        )

    active_nonzero = 0
    maximum_fraction = 0.0
    for channel, (value, enabled, cap, nominal_value, applied_value) in enumerate(
        zip(residual, mask, scale, nominal, applied, strict=True)
    ):
        if not math.isclose(
            applied_value - nominal_value,
            value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} channel {channel} residual/action mismatch"
            )
        if not enabled:
            if value != 0.0 or applied_value != nominal_value:
                raise VectorizedSmokeEvidenceError(
                    f"environment {env_index} leaked residual through masked channel {channel}"
                )
            continue
        if cap <= 0.0 or not math.isfinite(cap):
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} has an invalid enabled phase scale"
            )
        fraction = abs(value) / cap
        maximum_fraction = max(maximum_fraction, fraction)
        if value != 0.0:
            active_nonzero += 1

    zero_fast = bool(info.get("zero_residual_fast_path", False))
    if mode == "zero":
        if (
            any(value != 0.0 for value in raw + residual)
            or full12_bytes(applied) != full12_bytes(nominal)
            or not zero_fast
            or stages
        ):
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} failed exact zero-residual identity"
            )
    else:
        if zero_fast:
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} incorrectly used the zero fast path"
            )
        if active_nonzero <= 0:
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} has no active nonzero residual"
            )
        if not maximum_fraction < MAXIMUM_NONZERO_PHASE_SCALE_FRACTION:
            raise VectorizedSmokeEvidenceError(
                f"environment {env_index} residual reached five percent of phase scale"
            )

    try:
        seed = int(info["seed"])
        physics_tick = int(info["physics_tick"])
        sim_time_s = float(info["sim_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} lacks timing/seed evidence"
        ) from exc
    if seed < 0 or physics_tick < 0 or not math.isfinite(sim_time_s):
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} timing/seed evidence is invalid"
        )
    if not math.isclose(
        sim_time_s,
        physics_tick / PHYSICS_HZ,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise VectorizedSmokeEvidenceError(
            f"environment {env_index} time does not match its 120 Hz physics tick"
        )
    return VectorizedSmokeRowEvidence(
        mode=mode,
        decision_index=decision_index,
        env_index=env_index,
        seed=seed,
        phase=phase,
        physics_tick=physics_tick,
        sim_time_s=sim_time_s,
        raw_policy_action_full12=raw,
        nominal_action_full12=nominal,
        projected_residual_full12=residual,
        applied_action_full12=applied,
        effective_action_mask_full12=mask,
        physical_phase_scale_full12=scale,
        active_nonzero_channel_count=active_nonzero,
        max_abs_phase_scale_fraction=maximum_fraction,
        zero_residual_fast_path=zero_fast,
        projection_clipping_stages=stages,
        in_episode_root_write_count=root_writes,
        recording_runtime_access_count=recording_accesses,
        terminated=False,
        truncated=False,
    )


def collect_vectorized_residual_smoke_evidence(
    env: VectorizedRslResidualEnv,
    *,
    mode: SmokeMode,
    policy_decisions: int = 2,
    phase_actions: PhaseActionMasksV2 | None = None,
    maximum_phase_scale_fraction: float = DEFAULT_NONZERO_PHASE_SCALE_FRACTION,
) -> VectorizedSmokeEvidence:
    """Run and attest a deterministic zero- or small-nonzero vector smoke.

    This collector raises instead of returning a partial/pass-looking report
    when any invariant is missing.  It neither constructs nor closes Isaac.
    """

    if not isinstance(env, VectorizedRslResidualEnv):
        raise VectorizedSmokeEvidenceError(
            "smoke evidence requires a real VectorizedRslResidualEnv adapter"
        )
    if mode not in ("zero", "nonzero"):
        raise VectorizedSmokeEvidenceError("mode must be 'zero' or 'nonzero'")
    decisions = _positive_integer(policy_decisions, "policy_decisions")
    num_envs = _positive_integer(getattr(env, "num_envs", 0), "env.num_envs")
    if num_envs <= 1 or int(getattr(env, "num_actions", -1)) != ACTION_DIMENSION:
        raise VectorizedSmokeEvidenceError(
            "smoke requires a multi-row Full12 vector adapter"
        )
    cfg = getattr(env, "cfg", {})
    if not isinstance(cfg, Mapping) or (
        float(cfg.get("physics_hz", math.nan)) != PHYSICS_HZ
        or float(cfg.get("decision_hz", math.nan)) != DECISION_HZ
        or int(cfg.get("physics_ticks_per_decision", -1))
        != PHYSICS_TICKS_PER_DECISION
    ):
        raise VectorizedSmokeEvidenceError(
            "vector adapter does not declare the locked 120/15/8 timing contract"
        )
    row_envs = tuple(getattr(env, "environments", ()))
    row_action_configs = tuple(
        getattr(row_env, "phase_actions", None) for row_env in row_envs
    )
    if (
        len(row_action_configs) != num_envs
        or any(config is None for config in row_action_configs)
    ):
        raise VectorizedSmokeEvidenceError(
            "vector rows do not expose their phase-action configuration"
        )
    actions_config = phase_actions or row_action_configs[0]
    if not isinstance(actions_config, PhaseActionMasksV2):
        raise VectorizedSmokeEvidenceError("phase_actions must be a v2 configuration")
    if (
        actions_config.physics_hz != PHYSICS_HZ
        or actions_config.decision_hz != DECISION_HZ
        or actions_config.physics_ticks_per_decision != PHYSICS_TICKS_PER_DECISION
    ):
        raise VectorizedSmokeEvidenceError(
            "phase-action config differs from the locked 120/15/8 timing contract"
        )
    for row, config in enumerate(row_action_configs):
        if not isinstance(config, PhaseActionMasksV2) or any(
            config.mask_for(phase) != actions_config.mask_for(phase)
            or config.physical_scale_for(phase)
            != actions_config.physical_scale_for(phase)
            for phase in STATE_IDS
        ):
            raise VectorizedSmokeEvidenceError(
                f"environment {row} uses a different phase mask/scale configuration"
            )

    origins, initial_identity_groups = _origins_and_identity_groups(env, num_envs)
    initial_batch = getattr(env, "_batch", None)
    initial_counters = _batch_counters(initial_batch)
    if initial_counters[3] != 0 or initial_counters[4] != 0.0:
        raise VectorizedSmokeEvidenceError(
            "vector smoke must start from a fresh episode tick zero"
        )

    try:
        import torch
    except Exception as exc:
        raise VectorizedSmokeEvidenceError(
            f"PyTorch is required by VectorizedRslResidualEnv: {exc}"
        ) from exc

    evidence_rows: list[VectorizedSmokeRowEvidence] = []
    all_action_sets: list[tuple[tuple[float, ...], ...]] = []
    previous_counters = initial_counters
    observation_dimension: int | None = None
    for decision_index in range(decisions):
        if mode == "zero":
            source_rows = tuple((0.0,) * ACTION_DIMENSION for _ in range(num_envs))
        else:
            source_rows = deterministic_nonzero_action_rows(
                num_envs,
                decision_index=decision_index,
                maximum_phase_scale_fraction=maximum_phase_scale_fraction,
            )
        action_tensor = torch.tensor(
            source_rows,
            dtype=torch.float32,
            device=getattr(env, "device", "cpu"),
        )
        sent_rows = tuple(
            tuple(float(value) for value in row)
            for row in action_tensor.detach().to("cpu").tolist()
        )
        if mode == "nonzero" and len(set(sent_rows)) != num_envs:
            raise VectorizedSmokeEvidenceError(
                "float32 conversion collapsed distinct nonzero action rows"
            )
        all_action_sets.append(sent_rows)

        observations, rewards, dones, extras = env.step(action_tensor)
        try:
            policy = observations["policy"]
            critic = observations["critic"]
            policy_shape = tuple(int(value) for value in policy.shape)
        except (KeyError, TypeError, AttributeError) as exc:
            raise VectorizedSmokeEvidenceError(
                "vector adapter returned incomplete policy/critic observations"
            ) from exc
        if len(policy_shape) != 2 or policy_shape[0] != num_envs:
            raise VectorizedSmokeEvidenceError(
                "policy observations do not have N independent rows"
            )
        if observation_dimension is None:
            observation_dimension = policy_shape[1]
        if policy_shape[1] != observation_dimension:
            raise VectorizedSmokeEvidenceError("policy observation dimension changed")
        _assert_finite_tensor(policy, policy_shape, "policy observations")
        _assert_finite_tensor(critic, policy_shape, "critic observations")
        _assert_finite_tensor(rewards, (num_envs,), "rewards")
        if tuple(int(value) for value in dones.shape) != (num_envs,) or bool(
            dones.any().item()
        ):
            raise VectorizedSmokeEvidenceError(
                "vector smoke emitted a done row"
            )
        if not isinstance(extras, Mapping) or "time_outs" not in extras:
            raise VectorizedSmokeEvidenceError("vector smoke omitted time_outs")
        time_outs = extras["time_outs"]
        if tuple(int(value) for value in time_outs.shape) != (num_envs,) or bool(
            time_outs.any().item()
        ):
            raise VectorizedSmokeEvidenceError(
                "vector smoke emitted a truncation/time-out row"
            )

        current_batch = getattr(env, "_batch", None)
        current_counters = _batch_counters(current_batch)
        deltas = tuple(
            current - previous
            for current, previous in zip(
                current_counters[:4], previous_counters[:4], strict=True
            )
        )
        if deltas != (8, 8, 8, 8):
            raise VectorizedSmokeEvidenceError(
                "one policy decision did not advance exactly 8 physics/write/capture ticks"
            )
        if not math.isclose(
            current_counters[4] - previous_counters[4],
            1.0 / DECISION_HZ,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise VectorizedSmokeEvidenceError(
                "one policy decision did not advance exactly 1/15 second"
            )
        infos = tuple(getattr(env, "last_step_infos", ()))
        if len(infos) != num_envs or any(
            not isinstance(info, Mapping) for info in infos
        ):
            raise VectorizedSmokeEvidenceError(
                "vector adapter did not expose one info mapping per row"
            )
        expected_tick = (decision_index + 1) * PHYSICS_TICKS_PER_DECISION
        for env_index, (info, expected_action, origin) in enumerate(
            zip(infos, sent_rows, origins, strict=True)
        ):
            if int(info.get("physics_tick", -1)) != expected_tick:
                raise VectorizedSmokeEvidenceError(
                    f"environment {env_index} physics tick is not 120 Hz sequential"
                )
            if tuple(float(value) for value in info.get("env_origin_w_m", ())) != origin:
                raise VectorizedSmokeEvidenceError(
                    f"environment {env_index} world origin changed during smoke"
                )
            evidence_rows.append(
                _row_evidence(
                    info=info,
                    expected_action=expected_action,
                    mode=mode,
                    decision_index=decision_index,
                    env_index=env_index,
                    phase_actions=actions_config,
                )
            )
        _assert_no_frame_events(env, num_envs)
        previous_counters = current_counters

    _, final_identity_groups = _origins_and_identity_groups(env, num_envs)
    if final_identity_groups != initial_identity_groups:
        raise VectorizedSmokeEvidenceError(
            "controller, reader, or projection-bridge identities changed during smoke"
        )
    rows = tuple(evidence_rows)
    expected_rows = num_envs * decisions
    if len(rows) != expected_rows:
        raise VectorizedSmokeEvidenceError("vector smoke evidence rows are incomplete")
    measured_ticks = decisions * PHYSICS_TICKS_PER_DECISION
    total_deltas = tuple(
        previous_counters[index] - initial_counters[index] for index in range(3)
    )
    if total_deltas != (measured_ticks, measured_ticks, measured_ticks):
        raise VectorizedSmokeEvidenceError(
            "aggregate vector backend counters differ from measured physics ticks"
        )
    zero_exact = sum(
        full12_bytes(row.applied_action_full12)
        == full12_bytes(row.nominal_action_full12)
        and row.zero_residual_fast_path
        for row in rows
    )
    active_nonzero = sum(row.active_nonzero_channel_count > 0 for row in rows)
    maximum_observed = max(row.max_abs_phase_scale_fraction for row in rows)
    expected_zero_fast = (
        zero_exact == expected_rows
        if mode == "zero"
        else all(not row.zero_residual_fast_path for row in rows)
    )
    distinct = mode == "zero" or all(
        len(set(action_rows)) == num_envs for action_rows in all_action_sets
    )
    return VectorizedSmokeEvidence(
        schema=VECTOR_SMOKE_EVIDENCE_SCHEMA,
        status=ZERO_SMOKE_STATUS if mode == "zero" else NONZERO_SMOKE_STATUS,
        mode=mode,
        passed=True,
        num_envs=num_envs,
        policy_decisions=decisions,
        row_evidence_count=expected_rows,
        physics_hz=PHYSICS_HZ,
        decision_hz=DECISION_HZ,
        physics_ticks_per_decision=PHYSICS_TICKS_PER_DECISION,
        measured_physics_ticks=measured_ticks,
        global_physics_steps=total_deltas[0],
        batched_articulation_writes=total_deltas[1],
        exact_pair_captures=total_deltas[2],
        independent_origin_count=len(set(origins)),
        independent_controller_count=len(set(initial_identity_groups[0])),
        independent_reader_count=len(set(initial_identity_groups[1])),
        independent_projection_bridge_count=len(set(initial_identity_groups[2])),
        live_vectorized_isaac_backend_verified=True,
        deterministic_distinct_action_rows=distinct,
        zero_applied_equals_nominal_row_count=zero_exact,
        nonzero_active_row_count=active_nonzero,
        maximum_observed_phase_scale_fraction=maximum_observed,
        all_masks_honored=True,
        all_zero_fast_path_expected=expected_zero_fast,
        no_in_episode_root_writes=True,
        no_recording_runtime_access=True,
        no_termination_or_safety_events=True,
        environment_origins_w_m=origins,
        rows=rows,
    )


__all__ = [
    "DEFAULT_NONZERO_PHASE_SCALE_FRACTION",
    "MAXIMUM_NONZERO_PHASE_SCALE_FRACTION",
    "NONZERO_SMOKE_STATUS",
    "VECTOR_SMOKE_EVIDENCE_SCHEMA",
    "ZERO_SMOKE_STATUS",
    "VectorizedSmokeEvidence",
    "VectorizedSmokeEvidenceError",
    "VectorizedSmokeRowEvidence",
    "collect_vectorized_residual_smoke_evidence",
    "deterministic_nonzero_action_rows",
]
