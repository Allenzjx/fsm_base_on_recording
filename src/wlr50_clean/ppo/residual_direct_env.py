"""RSL-RL compatible phase-specific residual environment.

``ResidualEpisodeEnv`` owns one policy episode and delegates every physical
tick to an ``AuthoritativeFSMBackend``.  It deliberately has no state-write,
force, gravity, Recording, or FSM-transition API.  ``RslResidualVecEnv`` is a
thin official-RSL adapter; production currently supplies one live backend per
independent Isaac scene, while tests can exercise vector independence with
injected backends.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .observation_schema_v2 import (
    OBSERVATION_DIMENSION_V2,
    PPOObservationFrameV2,
    ObservationSchemaV2,
    load_observation_schema_v2,
)
from .phase_action_masks_v2 import (
    PhaseActionMasksV2,
    PhaseTransitionBridge,
    build_action_projector_v2,
    load_phase_action_masks_v2,
)
from .phase_objectives import (
    DENSE_FAMILIES,
    PhaseObjectivesConfig,
    PhysicalProgressState,
    load_phase_objectives,
)
from .ppo_env_adapter import AuthoritativeFSMBackend, AuthoritativeFrame
from .reward_v2 import (
    RewardBreakdownV2,
    RewardCalculatorV2,
    RewardSignalsV2,
    load_reward_v2_config,
)
from .termination import TerminationReason
from .termination_v2 import (
    TerminationEvaluatorV2,
    TerminationSignalsV2,
)


PHYSICS_HZ = 120.0
DECISION_HZ = 15.0
PHYSICS_TICKS_PER_DECISION = 8
ACTION_DIMENSION = 12
MAX_EPISODE_DECISIONS = 3000
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
PHASE_CURRICULUM_PRIORITY_STATES = ("P02", "P03", "P08", "P12", "P13")
PHASE_CURRICULUM_BOUNDARY_REASON = "PHASE_CURRICULUM_BOUNDARY"
PHASE_CURRICULUM_HORIZON_REASON = "PHASE_CURRICULUM_HORIZON"
DEFAULT_PHASE_CURRICULUM_MAX_DECISIONS = 64
DEFAULT_PHASE_CURRICULUM_RESET_CYCLE_SAMPLES = 128
DEFAULT_PHASE_CURRICULUM_OCCUPANCY_TOLERANCE = 0.02
PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS = {
    phase_id: (0.10 if phase_id in PHASE_CURRICULUM_PRIORITY_STATES else 0.0625)
    for phase_id in STATE_IDS
}
# Measured zero-residual policy-decision lengths from the accepted compact
# seed-2004 baseline.  Stage 1 samples end at the phase boundary (or at the
# 64-decision cap), so inverse-length reset weighting targets *decision*
# occupancy rather than merely balancing reset counts.
PHASE_CURRICULUM_BASELINE_DECISIONS = {
    "P01": 200,
    "P02": 8,
    "P03": 19,
    "P04": 73,
    "P05": 147,
    "P06": 384,
    "P07": 26,
    "P08": 7,
    "P09": 109,
    "P10": 4,
    "P11": 7,
    "P12": 71,
    "P13": 563,
}
WHEEL_ORDER = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
LEG_TO_WHEEL = {
    "FL": WHEEL_ORDER[0],
    "FR": WHEEL_ORDER[1],
    "RL": WHEEL_ORDER[2],
    "RR": WHEEL_ORDER[3],
}
ACTIVE_LEG = {
    "P01": "FR", "P02": "FR", "P03": "FR", "P04": "FL",
    "P05": "FL", "P06": "FL", "P07": "RR", "P08": "RR",
    "P09": "RR", "P10": "RL", "P11": "RL", "P12": "RL",
    "P13": None,
}


class ResidualDirectEnvError(RuntimeError):
    """The residual environment contract or a live transition is invalid."""


def _phase_float_mapping(
    values: Mapping[str, float | int], *, label: str, strictly_positive: bool
) -> dict[str, float]:
    if tuple(values) != STATE_IDS:
        raise ResidualDirectEnvError(f"{label} must contain ordered P01-P13")
    result = {phase_id: float(values[phase_id]) for phase_id in STATE_IDS}
    if any(
        not math.isfinite(value) or value < 0.0 or (strictly_positive and value <= 0.0)
        for value in result.values()
    ):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ResidualDirectEnvError(f"{label} values must be finite and {qualifier}")
    return result


def build_phase_curriculum_reset_cycle(
    *,
    target_decision_fractions: Mapping[str, float] = PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS,
    baseline_phase_decisions: Mapping[str, int] = PHASE_CURRICULUM_BASELINE_DECISIONS,
    max_decisions: int = DEFAULT_PHASE_CURRICULUM_MAX_DECISIONS,
    cycle_samples: int = DEFAULT_PHASE_CURRICULUM_RESET_CYCLE_SAMPLES,
) -> tuple[str, ...]:
    """Build a deterministic inverse-duration Stage-1 reset cycle.

    Reset probabilities are proportional to target decision occupancy divided
    by the measured, horizon-capped phase duration.  Largest-remainder integer
    allocation followed by smooth weighted round-robin keeps the finite cycle
    deterministic, interleaved, and fully covering P01-P13.
    """

    targets = _phase_float_mapping(
        target_decision_fractions,
        label="target decision fractions",
        strictly_positive=True,
    )
    if not math.isclose(sum(targets.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ResidualDirectEnvError("target decision fractions must sum to one")
    baseline = _phase_float_mapping(
        baseline_phase_decisions,
        label="baseline phase decisions",
        strictly_positive=True,
    )
    horizon = int(max_decisions)
    cycle_length = int(cycle_samples)
    if horizon <= 0:
        raise ResidualDirectEnvError("phase curriculum decision horizon must be positive")
    if cycle_length < len(STATE_IDS):
        raise ResidualDirectEnvError(
            "phase curriculum reset cycle must have at least one slot per phase"
        )

    inverse_duration_weights = {
        phase_id: targets[phase_id] / min(baseline[phase_id], float(horizon))
        for phase_id in STATE_IDS
    }
    weight_total = sum(inverse_duration_weights.values())
    ideal_counts = {
        phase_id: cycle_length * inverse_duration_weights[phase_id] / weight_total
        for phase_id in STATE_IDS
    }
    weights = {
        phase_id: int(math.floor(ideal_counts[phase_id])) for phase_id in STATE_IDS
    }
    remaining = cycle_length - sum(weights.values())
    remainder_order = sorted(
        STATE_IDS,
        key=lambda phase_id: (
            -(ideal_counts[phase_id] - weights[phase_id]),
            STATE_IDS.index(phase_id),
        ),
    )
    for phase_id in remainder_order[:remaining]:
        weights[phase_id] += 1
    if any(weights[phase_id] <= 0 for phase_id in STATE_IDS):
        raise ResidualDirectEnvError(
            "phase curriculum reset cycle allocation omitted a phase; increase cycle_samples"
        )

    scores = {phase_id: 0 for phase_id in STATE_IDS}
    result: list[str] = []
    for _ in range(cycle_length):
        for phase_id in STATE_IDS:
            scores[phase_id] += weights[phase_id]
        selected = max(
            STATE_IDS,
            key=lambda phase_id: (scores[phase_id], -STATE_IDS.index(phase_id)),
        )
        scores[selected] -= cycle_length
        result.append(selected)
    return tuple(result)


PHASE_CURRICULUM_RESET_CYCLE = build_phase_curriculum_reset_cycle()


def build_reward_dominance_telemetry(
    *,
    signed_sums: Mapping[str, float],
    absolute_sums: Mapping[str, float],
    absolute_sums_by_phase: Mapping[str, Mapping[str, float]],
    incomplete_count: int,
    maximum_single_family_fraction: float,
    maximum_residual_regularization_fraction: float,
    minimum_absolute_dense_return: float,
) -> dict[str, Any]:
    """Summarize five-family contribution evidence and its fail-closed gate."""

    if tuple(signed_sums) != DENSE_FAMILIES or tuple(absolute_sums) != DENSE_FAMILIES:
        raise ResidualDirectEnvError(
            "reward telemetry must contain the ordered five dense families"
        )
    if tuple(absolute_sums_by_phase) != STATE_IDS or any(
        tuple(absolute_sums_by_phase[phase_id]) != DENSE_FAMILIES
        for phase_id in STATE_IDS
    ):
        raise ResidualDirectEnvError(
            "phase reward telemetry must contain P01-P13 by five dense families"
        )
    single_limit = float(maximum_single_family_fraction)
    residual_limit = float(maximum_residual_regularization_fraction)
    minimum_total = float(minimum_absolute_dense_return)
    if (
        not math.isfinite(single_limit)
        or not 0.0 < single_limit <= 1.0
        or not math.isfinite(residual_limit)
        or not 0.0 < residual_limit <= 1.0
        or not math.isfinite(minimum_total)
        or minimum_total < 0.0
    ):
        raise ResidualDirectEnvError("reward dominance limits are invalid")

    signed = {family: float(signed_sums[family]) for family in DENSE_FAMILIES}
    if any(not math.isfinite(value) for value in signed.values()):
        raise ResidualDirectEnvError("signed reward contributions must be finite")
    absolute = {family: float(absolute_sums[family]) for family in DENSE_FAMILIES}
    if any(not math.isfinite(value) or value < 0.0 for value in absolute.values()):
        raise ResidualDirectEnvError("absolute reward contributions must be finite and non-negative")
    absolute_total = sum(absolute.values())
    signal_sufficient = absolute_total >= minimum_total and absolute_total > 0.0
    fractions = {
        family: absolute[family] / absolute_total if absolute_total > 0.0 else 0.0
        for family in DENSE_FAMILIES
    }
    dominant_family = max(DENSE_FAMILIES, key=lambda family: fractions[family])
    dominant_fraction = fractions[dominant_family]
    residual_fraction = fractions["residual_regularization"]
    telemetry_complete = int(incomplete_count) == 0
    single_within_limit = bool(signal_sufficient and dominant_fraction <= single_limit)
    residual_within_limit = bool(signal_sufficient and residual_fraction <= residual_limit)

    phase_absolute: dict[str, dict[str, float]] = {}
    phase_fractions: dict[str, dict[str, float]] = {}
    for phase_id in STATE_IDS:
        row = {
            family: float(absolute_sums_by_phase[phase_id][family])
            for family in DENSE_FAMILIES
        }
        if any(not math.isfinite(value) or value < 0.0 for value in row.values()):
            raise ResidualDirectEnvError(
                f"absolute reward contributions for {phase_id} must be finite and non-negative"
            )
        row_total = sum(row.values())
        phase_absolute[phase_id] = row
        phase_fractions[phase_id] = {
            family: row[family] / row_total if row_total > 0.0 else 0.0
            for family in DENSE_FAMILIES
        }
    for family in DENSE_FAMILIES:
        phase_total = sum(phase_absolute[phase_id][family] for phase_id in STATE_IDS)
        if not math.isclose(
            phase_total, absolute[family], rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ResidualDirectEnvError(
                f"global and phase reward contributions disagree for {family}"
            )

    return {
        "reward_family_signed_sums": signed,
        "reward_family_absolute_sums": absolute,
        "reward_family_absolute_fraction": fractions,
        "reward_family_absolute_sums_by_phase": phase_absolute,
        "reward_family_absolute_fraction_by_phase": phase_fractions,
        "reward_absolute_dense_total": absolute_total,
        "reward_dominant_family": dominant_family,
        "reward_dominant_family_fraction": dominant_fraction,
        "reward_residual_regularization_fraction": residual_fraction,
        "reward_maximum_single_family_fraction": single_limit,
        "reward_maximum_residual_regularization_fraction": residual_limit,
        "reward_minimum_absolute_dense_return": minimum_total,
        "reward_absolute_dense_signal_sufficient": signal_sufficient,
        "reward_single_family_within_limit": single_within_limit,
        "reward_residual_regularization_within_limit": residual_within_limit,
        "reward_telemetry_incomplete_count": int(incomplete_count),
        "reward_telemetry_complete": telemetry_complete,
        "reward_dominance_within_limits": bool(
            telemetry_complete and single_within_limit and residual_within_limit
        ),
    }


def build_completed_episode_telemetry(
    completed_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Separate physical episode endings from synchronous reset peers.

    A true-batch Isaac reset currently resets every clone when any clone reaches
    a terminal condition.  Those otherwise-live peer rows are valid RSL
    truncations, but they are not authoritative task outcomes and must never be
    counted as successes or failures.
    """

    terminal_reason_counts: dict[str, int] = {}
    peer_count = 0
    for index, row in enumerate(completed_episodes):
        if not isinstance(row, Mapping):
            raise ResidualDirectEnvError(
                f"completed episode row {index} is not a mapping"
            )
        if bool(row.get("vector_batch_reset_peer", False)):
            peer_count += 1
            continue
        reason_value = row.get("termination_reason")
        if isinstance(reason_value, TerminationReason):
            reason = reason_value.value
        elif isinstance(reason_value, str) and reason_value:
            reason = reason_value
        else:
            raise ResidualDirectEnvError(
                f"authoritative completed episode row {index} has no terminal reason"
            )
        terminal_reason_counts[reason] = terminal_reason_counts.get(reason, 0) + 1

    authoritative_count = sum(terminal_reason_counts.values())
    return {
        "authoritative_completed_episode_count": authoritative_count,
        "authoritative_terminal_reason_counts": terminal_reason_counts,
        "authoritative_success_count": terminal_reason_counts.get(
            TerminationReason.SUCCESS.value, 0
        ),
        "vector_batch_reset_peer_count": peer_count,
    }


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values) / max(1, len(values)))


def _guard(observation: Any, name: str) -> bool:
    guards = _member(observation, "guards", {})
    value = guards.get(name, False) if isinstance(guards, Mapping) else False
    return bool(value.get("passed", False)) if isinstance(value, Mapping) else bool(value)


def _pair_force(contact: Any) -> float:
    total = 0.0
    for pair_name in ("ground", "obstacle"):
        pair = _member(contact, pair_name)
        if pair is not None and bool(_member(pair, "pair_verified", False)):
            total += max(0.0, float(_member(pair, "normal_force_n", 0.0)))
    return total


def _wheel_measurements(observation: Any) -> dict[str, dict[str, float | bool]]:
    wheels = _member(observation, "wheels", {})
    contacts = _member(observation, "contacts", {})
    base = _member(observation, "base")
    body_speed = float(_member(base, "linear_velocity_w_m_s", (0.0, 0.0, 0.0))[0])
    rows: dict[str, dict[str, float | bool]] = {}
    forces: list[float] = []
    for name in WHEEL_ORDER:
        wheel = wheels[name]
        body_name = str(_member(wheel, "body_name"))
        contact = contacts[body_name]
        force = _pair_force(contact)
        forces.append(force)
        center = _member(wheel, "center_w_m", (0.0, 0.0, 0.0))
        bottom = _member(wheel, "bottom_w_m", (0.0, 0.0, 0.0))
        surface_speed = 0.04998999834060672 * float(_member(wheel, "velocity_rad_s", 0.0))
        slip = (surface_speed - body_speed) / max(abs(surface_speed), abs(body_speed), 0.05)
        obstacle_pair = _member(contact, "obstacle")
        rows[name] = {
            "force": force,
            "load": 0.0,
            "x": float(center[0]),
            "y": float(center[1]),
            "bottom_z": float(bottom[2]),
            "slip": slip,
            "obstacle_contact": bool(_member(obstacle_pair, "active", False)),
        }
    total = sum(forces)
    if total > 1.0e-9:
        for name, force in zip(WHEEL_ORDER, forces, strict=True):
            rows[name]["load"] = force / total
    return rows


class LivePhysicalSignalBuilder:
    """Convert deployable live sensing into bounded phase reward signals."""

    def __init__(self, objectives: PhaseObjectivesConfig | None = None) -> None:
        self.objectives = objectives or load_phase_objectives()

    def progress(self, frame: AuthoritativeFrame) -> PhysicalProgressState:
        observation = frame.info.get("raw_observation")
        if observation is None:
            raise ResidualDirectEnvError("backend frame omits raw_observation")
        wheels = _wheel_measurements(observation)
        obstacle = _member(observation, "obstacle")
        front = float(_member(obstacle, "front_x_m"))
        top = float(_member(obstacle, "top_z_m"))
        base = _member(observation, "base")
        base_x = float(_member(base, "position_w_m", (0.0, 0.0, 0.0))[0])
        speed_x = float(_member(base, "linear_velocity_w_m_s", (0.0, 0.0, 0.0))[0])
        support = _member(observation, "support")
        support_continuity = _clamp01(float(_member(support, "support_count", 0)) / 4.0)

        leg = ACTIVE_LEG[frame.state_id]
        active = wheels[LEG_TO_WHEEL[leg]] if leg is not None else None

        def clearance(row: Mapping[str, float | bool]) -> float:
            return _clamp01((float(row["bottom_z"]) - max(0.0, top - 0.05)) / 0.10)

        def front_progress(row: Mapping[str, float | bool]) -> float:
            return _clamp01((float(row["x"]) - front + 0.06) / 0.16)

        def top_placement(which: str) -> float:
            row = wheels[LEG_TO_WHEEL[which]]
            latched = _guard(observation, f"leg_top_loaded_latched:{which}")
            height = _clamp01(1.0 - abs(float(row["bottom_z"]) - top) / 0.05)
            return max(float(latched), height * front_progress(row))

        def top_contact_capture(which: str) -> float:
            row = wheels[LEG_TO_WHEEL[which]]
            latched = _guard(observation, f"leg_top_loaded_latched:{which}")
            top_gap = float(row["bottom_z"]) - top
            live_top_contact = bool(
                row["obstacle_contact"]
                and float(row["x"]) >= front
                and -0.015 <= top_gap <= 0.025
            )
            return float(latched or live_top_contact)

        loads = {leg_name: float(wheels[name]["load"]) for leg_name, name in LEG_TO_WHEEL.items()}
        speed_convergence = _clamp01(1.0 - abs(speed_x - 0.10) / 0.20)
        rear_x = 0.5 * (float(wheels[WHEEL_ORDER[2]]["x"]) + float(wheels[WHEEL_ORDER[3]]["x"]))
        terms: dict[str, float] = {
            "fr_lift_entry_geometry": 0.5 * clearance(wheels[WHEEL_ORDER[1]]) + 0.5 * (1.0 - loads["FR"]),
            "target_load_transfer": 1.0 - (loads[leg] if leg is not None else 0.0),
            "support_continuity": support_continuity,
            "fr_lift_clearance": clearance(wheels[WHEEL_ORDER[1]]),
            "approach_progress": front_progress(wheels[WHEEL_ORDER[1]]),
            "fr_front_face_progress": front_progress(wheels[WHEEL_ORDER[1]]),
            "fr_top_placement": top_placement("FR"),
            "contact_capture": top_contact_capture("FR"),
            "fl_lift_workspace": 0.5 * clearance(wheels[WHEEL_ORDER[0]]) + 0.5 * (1.0 - loads["FL"]),
            "fl_lift_clearance": clearance(wheels[WHEEL_ORDER[0]]),
            "fl_front_face_progress": front_progress(wheels[WHEEL_ORDER[0]]),
            "fl_top_placement": top_placement("FL"),
            "forward_progress": _clamp01((base_x - (front - 0.35)) / 0.70),
            "rear_pair_pre_edge_geometry": _clamp01((rear_x - front + 0.30) / 0.30),
            "speed_convergence": speed_convergence,
            "rear_entry_geometry": _clamp01((rear_x - front + 0.20) / 0.25),
            "alignment_quality": _clamp01(1.0 - abs(float(_member(base, "position_w_m", (0.0, 0.0, 0.0))[1])) / 0.20),
            "com_target_progress": (
                loads["FL"]
                if frame.state_id == "P08"
                else (loads["FR"] if frame.state_id == "P11" else 0.0)
            ),
            "fl_load_capture": loads["FL"],
            "rr_unload_progress": 1.0 - loads["RR"],
            "rr_lift_clearance": clearance(wheels[WHEEL_ORDER[3]]),
            "rr_front_face_progress": front_progress(wheels[WHEEL_ORDER[3]]),
            "rr_top_placement": top_placement("RR"),
            "rl_workspace": 0.5 * clearance(wheels[WHEEL_ORDER[2]]) + 0.5 * (1.0 - loads["RL"]),
            "support_capture": support_continuity,
            "fr_load_capture": loads["FR"],
            "rl_unload_progress": 1.0 - loads["RL"],
            "rl_lift_clearance": clearance(wheels[WHEEL_ORDER[2]]),
            "rl_front_face_progress": front_progress(wheels[WHEEL_ORDER[2]]),
            "rl_top_placement": top_placement("RL"),
            "final_forward_clearance": _clamp01((rear_x - front) / 0.30),
            "home_pose_convergence": self._home_pose_convergence(observation),
            "wheel_stop_convergence": _clamp01(1.0 - max(abs(float(_member(_member(observation, "wheels", {})[name], "velocity_rad_s", 0.0))) for name in WHEEL_ORDER) / 1.0),
        }
        objective = self.objectives.phase(frame.state_id)
        selected = {name: _clamp01(terms[name]) for name in objective.potential_terms}
        return PhysicalProgressState(frame.state_id, selected)

    @staticmethod
    def _home_pose_convergence(observation: Any) -> float:
        joints = _member(observation, "joints", {})
        errors = [abs(float(_member(row, "position_deg", 0.0))) for row in joints.values()]
        return _clamp01(1.0 - _rms(errors) / 25.0)

    def contact_costs(
        self, start: AuthoritativeFrame, end: AuthoritativeFrame
    ) -> dict[str, float]:
        before = start.info["raw_observation"]
        after = end.info["raw_observation"]
        wheels = _wheel_measurements(after)
        objective = self.objectives.phase(start.state_id)
        leg = objective.active_leg
        active = wheels[LEG_TO_WHEEL[leg]] if leg else None
        obstacle = _member(after, "obstacle")
        top = float(_member(obstacle, "top_z_m", 0.05))
        support_count = float(_member(_member(after, "support"), "support_count", 0.0))
        com_before = _member(_member(before, "center_of_mass"), "position_w_m", (0.0, 0.0, 0.0))
        com_after = _member(_member(after, "center_of_mass"), "position_w_m", (0.0, 0.0, 0.0))
        base_before = _member(_member(before, "base"), "linear_velocity_w_m_s", (0.0, 0.0, 0.0))
        base_after = _member(_member(after, "base"), "linear_velocity_w_m_s", (0.0, 0.0, 0.0))
        target_y_sign = {"P01": -1.0, "P04": 1.0, "P08": 1.0, "P10": 1.0, "P11": -1.0}.get(start.state_id, 0.0)
        delta_y = float(com_after[1]) - float(com_before[1])
        contact_history = ()
        if active is not None:
            wheel = _member(after, "wheels", {})[LEG_TO_WHEEL[leg]]
            contact = _member(after, "contacts", {})[str(_member(wheel, "body_name"))]
            obstacle_pair = _member(contact, "obstacle")
            contact_history = tuple(bool(value) for value in _member(obstacle_pair, "active_history", ()))
        chatter = sum(a != b for a, b in zip(contact_history, contact_history[1:])) / max(1, len(contact_history) - 1)
        all_costs = {
            "clearance_shortfall": 0.0 if active is None else _clamp01((top + 0.005 - float(active["bottom_z"])) / 0.05),
            "placement_impact": 0.0 if active is None else _clamp01(float(active["force"]) / 100.0),
            "contact_rebound": 0.0 if leg is None else self._active_vertical_speed(after, leg, upward_only=True) / 0.5,
            "contact_chatter": _clamp01(chatter),
            "wheel_slip": _clamp01(sum(abs(float(row["slip"])) for row in wheels.values()) / 4.0),
            "support_loss": _clamp01((3.0 - support_count) / 3.0),
            "wrong_direction_com_motion": _clamp01(max(0.0, -target_y_sign * delta_y) / 0.01),
            "orthogonal_com_oscillation": _clamp01(abs(float(com_after[0]) - float(com_before[0])) / 0.02),
            "forward_speed_variation": _clamp01(abs(float(base_after[0]) - float(base_before[0])) / 0.20),
            "overshoot": _clamp01(max(0.0, float(base_after[0]) - 0.35) / 0.50),
        }
        return {name: all_costs[name] for name in objective.contact_cost_terms}

    @staticmethod
    def _active_vertical_speed(observation: Any, leg: str, *, upward_only: bool) -> float:
        wheel = _member(observation, "wheels", {})[LEG_TO_WHEEL[leg]]
        body = _member(observation, "bodies", {})[str(_member(wheel, "body_name"))]
        value = float(_member(body, "linear_velocity_w_m_s", (0.0, 0.0, 0.0))[2])
        return max(0.0, value) if upward_only else abs(value)


@dataclass(frozen=True, slots=True)
class ResidualStep:
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class ResidualEpisodeEnv:
    """One 15 Hz residual policy wrapped around one authoritative backend."""

    def __init__(
        self,
        backend: AuthoritativeFSMBackend,
        *,
        observation_schema: ObservationSchemaV2 | None = None,
        phase_actions: PhaseActionMasksV2 | None = None,
        reward_calculator: RewardCalculatorV2 | None = None,
        termination_evaluator: TerminationEvaluatorV2 | None = None,
        collect_trace: bool = False,
        tick_callback: Callable[[AuthoritativeFrame, AuthoritativeFrame, Any], None] | None = None,
    ) -> None:
        self.backend = backend
        self.observation_schema = observation_schema or load_observation_schema_v2()
        self.phase_actions = phase_actions or load_phase_action_masks_v2()
        self.projector = build_action_projector_v2(self.phase_actions)
        self.bridge = PhaseTransitionBridge(self.projector)
        self.reward_calculator = reward_calculator or RewardCalculatorV2.from_files()
        self.termination_evaluator = termination_evaluator or TerminationEvaluatorV2()
        self.signals = LivePhysicalSignalBuilder(self.reward_calculator.objectives)
        self.collect_trace = bool(collect_trace)
        self.tick_callback = tick_callback
        self.trace: list[dict[str, Any]] = []
        self.frame: AuthoritativeFrame | None = None
        self.observation: tuple[float, ...] | None = None
        self.previous_residual = (0.0,) * ACTION_DIMENSION
        self.previous_previous_residual = (0.0,) * ACTION_DIMENSION
        self.decision_count = 0
        self.seed = 0
        self.done = False

    def reset(
        self, *, seed: int, options: Mapping[str, Any] | None = None
    ) -> tuple[tuple[float, ...], Mapping[str, Any]]:
        self.seed = int(seed)
        self.frame = self.backend.reset(seed=self.seed, options=dict(options or {}))
        self.bridge.reset(state_id=self.frame.state_id)
        self.previous_residual = (0.0,) * ACTION_DIMENSION
        self.previous_previous_residual = (0.0,) * ACTION_DIMENSION
        self.decision_count = 0
        self.done = False
        self.trace = []
        self.observation = self._encode(self.frame)
        return self.observation, dict(self.frame.info)

    def refresh_after_video_pre_action_hold(
        self,
    ) -> tuple[tuple[float, ...], Mapping[str, Any]]:
        """Refresh the tick-zero actor input after physical video pre-roll."""

        if (
            self.frame is None
            or self.observation is None
            or self.done
            or self.decision_count != 0
        ):
            raise ResidualDirectEnvError(
                "video pre-roll refresh requires a fresh unstepped episode"
            )
        refresh = getattr(self.backend, "refresh_video_pre_action_frame", None)
        if not callable(refresh):
            raise ResidualDirectEnvError(
                "backend lacks the required video pre-roll sensing refresh"
            )
        frame = refresh()
        if frame.physics_tick != 0 or frame.sim_time_s != 0.0 or frame.state_id != "P01":
            raise ResidualDirectEnvError(
                "video pre-roll sensing refresh changed the logical P01 clock"
            )
        self.frame = frame
        self.observation = self._encode(frame)
        return self.observation, dict(frame.info)

    def step(
        self,
        action: Sequence[float],
        *,
        stop_after_phase_id: str | None = None,
    ) -> ResidualStep:
        if self.frame is None or self.observation is None:
            raise ResidualDirectEnvError("reset must precede step")
        if self.done:
            raise ResidualDirectEnvError("step called after episode completion")
        if stop_after_phase_id is not None:
            if stop_after_phase_id not in STATE_IDS:
                raise ResidualDirectEnvError("phase-boundary stop must name P01-P13")
            if self.frame.state_id != stop_after_phase_id:
                raise ResidualDirectEnvError(
                    "phase-boundary stop differs from the active curriculum phase"
                )
        raw = tuple(float(value) for value in action)
        if len(raw) != ACTION_DIMENSION or any(not math.isfinite(value) for value in raw):
            raise ResidualDirectEnvError("policy action must be finite Full12")
        start = self.frame
        projections = []
        transition_metrics = []
        decision = None
        controller_blocked = False
        phase_curriculum_boundary = False
        for _ in range(PHYSICS_TICKS_PER_DECISION):
            frame = self.frame
            projected = self.bridge.project_tick(
                raw,
                state_id=frame.state_id,
                nominal_action_full12=frame.nominal_action_full12,
                reference_action_full12=frame.reference_action_full12,
                reference_delta_full12=frame.reference_delta_full12,
                # v2 masks deliberately replace the old Recording-derived
                # runtime mask.  Live hard-safety remains in SafetyProjection.
                runtime_action_mask_full12=(1,) * ACTION_DIMENSION,
                safety=frame.safety_projection,
                dt_s=1.0 / PHYSICS_HZ,
            )
            next_frame = self.backend.step_physics(projected.projection.applied_action_full12)
            if next_frame.physics_tick != frame.physics_tick + 1:
                raise ResidualDirectEnvError("backend did not advance exactly one physics tick")
            projections.append(projected.projection)
            if self.tick_callback is not None:
                self.tick_callback(frame, next_frame, projected.projection)
            if projected.transition_metric is not None:
                transition_metrics.append(projected.transition_metric.as_dict())
            self.frame = next_frame
            source = next_frame.termination_signals
            signals = TerminationSignalsV2(
                authoritative_success=source.success,
                body_collision=source.body_collision,
                wheel_only_climb=source.wheel_only_climb,
                fall=source.fall,
                nan_inf=source.nan_inf,
                hard_joint_limit=source.hard_joint_limit,
                physics_explosion=source.physics_explosion,
                reference_conformance_outside_30pct=source.reference_conformance_outside_30pct,
            )
            decision = self.termination_evaluator.evaluate(
                signals, episode_time_s=next_frame.sim_time_s
            )
            controller_blocked = (
                next_frame.info.get("controller_task_result")
                == "INCOMPLETE_CONTROLLER_BLOCKED"
            )
            phase_curriculum_boundary = bool(
                stop_after_phase_id is not None
                and next_frame.state_id != stop_after_phase_id
            )
            if (
                decision.terminated
                or decision.truncated
                or controller_blocked
                or phase_curriculum_boundary
            ):
                break
        assert decision is not None and projections
        end = self.frame
        reward_breakdown = self._reward(
            start,
            end,
            projections[-1].safe_projected_residual_full12,
            termination_reason=decision.reason,
            controller_blocked=controller_blocked,
        )
        # A frozen-controller dead end is a real task failure for PPO, not a
        # time-limit truncation.  Treating it as a timeout would let the value
        # target bootstrap across a failed episode and would omit the configured
        # task-failure event penalty.
        terminated = bool(decision.terminated or controller_blocked)
        truncated = bool(
            not terminated
            and (decision.truncated or phase_curriculum_boundary)
        )
        # Preserve the evaluator's primary physical/success reason in episode
        # evidence.  A controller dead end is a fallback terminal classification
        # and only supersedes a simultaneous time-limit truncation.
        if decision.reason not in {None, TerminationReason.TIMEOUT}:
            reason = decision.reason.value
        elif controller_blocked:
            reason = "CONTROLLER_BLOCKED"
        elif decision.reason is not None:
            reason = decision.reason.value
        elif phase_curriculum_boundary:
            reason = PHASE_CURRICULUM_BOUNDARY_REASON
        else:
            reason = None
        residual = projections[-1].safe_projected_residual_full12
        self.previous_previous_residual = self.previous_residual
        self.previous_residual = residual
        self.decision_count += 1
        self.done = terminated or truncated
        self.observation = self._encode(end)
        info = {
            **dict(end.info),
            "seed": self.seed,
            "physics_tick": end.physics_tick,
            "sim_time_s": end.sim_time_s,
            "decision_index": self.decision_count - 1,
            "physics_ticks_executed": len(projections),
            "raw_policy_action_full12": list(raw),
            "projected_residual_full12": list(residual),
            "applied_action_full12": list(projections[-1].applied_action_full12),
            "phase_transition_action_jump": transition_metrics,
            "reward": asdict(reward_breakdown),
            "termination_reason": reason,
            "terminated": terminated,
            "truncated": truncated,
            "phase_curriculum_boundary": phase_curriculum_boundary,
            "phase_curriculum_start_state_id": stop_after_phase_id,
            "in_episode_root_write_count": int(end.info.get("in_episode_root_pose_writes", 0))
            + int(end.info.get("in_episode_root_velocity_writes", 0)),
            "recording_runtime_access_count": int(end.info.get("recording_accesses", 0)),
        }
        if info["in_episode_root_write_count"] != 0:
            raise ResidualDirectEnvError("FORBIDDEN_IN_EPISODE_ROOT_WRITE")
        if info["recording_runtime_access_count"] != 0:
            raise ResidualDirectEnvError("Recording runtime access is forbidden")
        if self.collect_trace:
            self.trace.append(self._trace_row(end, reward_breakdown, info))
        return ResidualStep(self.observation, reward_breakdown.total, terminated, truncated, info)

    def _encode(self, frame: AuthoritativeFrame) -> tuple[float, ...]:
        raw = frame.info.get("raw_observation")
        if raw is None:
            raise ResidualDirectEnvError("backend frame omits raw_observation")
        v2 = PPOObservationFrameV2.from_live_observation(
            raw,
            state_id=frame.state_id,
            macro_phase=frame.macro_phase,
            lifecycle=str(frame.info.get("controller_lifecycle", "EXECUTE_MOTION")),
            phase_progress=frame.phase_progress,
            # The authoritative observation is built from the backend's
            # actual previous logical action.  This matters at a phase-entry
            # snapshot: the reset-only physical write restores a non-zero
            # action, while ``atomic_ack`` still describes the preceding
            # zero-command settle because no episode action was issued.
            previous_action_full12=frame.observation.previous_action_full12,
            previous_projected_residual_full12=self.bridge.previous_projected_residual_full12,
        )
        encoded = self.observation_schema.encode(v2)
        if len(encoded) != OBSERVATION_DIMENSION_V2:
            raise ResidualDirectEnvError("v2 observation dimension changed")
        return encoded

    def _reward(
        self,
        start: AuthoritativeFrame,
        end: AuthoritativeFrame,
        residual: Sequence[float],
        *,
        termination_reason: TerminationReason | None,
        controller_blocked: bool,
    ) -> RewardBreakdownV2:
        previous_progress = self.signals.progress(start)
        current_progress = self.signals.progress(end)
        level = end.info.get("level_calibration", {})
        attitude = math.hypot(
            float(level.get("roll_error_to_level_rad", 0.0)),
            float(level.get("pitch_error_to_level_rad", 0.0)),
        )
        phase_objective = self.reward_calculator.objectives.phase(start.state_id)
        envelope = phase_objective.successful_fsm_attitude_envelope_rad
        envelope_normalization = (
            phase_objective.attitude_envelope_excess_normalization_rad
        )
        if (envelope is None) != (envelope_normalization is None):
            raise ResidualDirectEnvError(
                f"{start.state_id} has an incomplete successful-FSM attitude envelope"
            )
        attitude_envelope_excess = (
            0.0
            if envelope is None
            else _clamp01(
                max(0.0, attitude - float(envelope))
                / float(envelope_normalization)
            )
        )
        angular = _member(end.info["raw_observation"], "base")
        angular_rate = _norm(_member(angular, "angular_velocity_w_rad_s", (0.0, 0.0, 0.0)))
        before_angular = _member(start.info["raw_observation"], "base")
        acceleration = _norm(
            tuple(
                (float(after) - float(before)) * DECISION_HZ
                for before, after in zip(
                    _member(before_angular, "angular_velocity_w_rad_s", (0.0, 0.0, 0.0)),
                    _member(angular, "angular_velocity_w_rad_s", (0.0, 0.0, 0.0)),
                    strict=True,
                )
            )
        )
        scale = self.phase_actions.physical_scale_for(start.state_id)
        normalized = tuple(
            float(value) / max(float(limit), 1.0e-9)
            for value, limit in zip(residual, scale, strict=True)
        )
        previous_normalized = tuple(
            float(value) / max(float(limit), 1.0e-9)
            for value, limit in zip(self.previous_residual, scale, strict=True)
        )
        previous_previous_normalized = tuple(
            float(value) / max(float(limit), 1.0e-9)
            for value, limit in zip(self.previous_previous_residual, scale, strict=True)
        )
        first_difference = _rms(tuple(a - b for a, b in zip(normalized, previous_normalized, strict=True)))
        second_difference = _rms(
            tuple(
                a - 2.0 * b + c
                for a, b, c in zip(
                    normalized, previous_normalized, previous_previous_normalized, strict=True
                )
            )
        )
        phase_completed = STATE_IDS.index(end.state_id) > STATE_IDS.index(start.state_id)
        # Event reward classification must be identical to the single primary
        # reason selected by TerminationEvaluatorV2.  Physical signals can
        # legitimately co-occur (for example a body collision and a fall);
        # independently OR-ing event families would either double penalize or
        # violate RewardSignalsV2's mutual-exclusion contract.
        task_failure_reasons = {
            TerminationReason.BODY_COLLISION,
            TerminationReason.WHEEL_ONLY_CLIMB,
        }
        safety_abort_reasons = {
            TerminationReason.FALL,
            TerminationReason.NAN_INF,
            TerminationReason.HARD_JOINT_LIMIT,
            TerminationReason.PHYSICS_EXPLOSION,
        }
        task_failure = bool(
            termination_reason in task_failure_reasons
            or (
                controller_blocked
                and termination_reason not in task_failure_reasons
                and termination_reason not in safety_abort_reasons
                and termination_reason is not TerminationReason.SUCCESS
            )
        )
        safety_abort = termination_reason in safety_abort_reasons
        final_success = termination_reason is TerminationReason.SUCCESS
        return self.reward_calculator.evaluate(
            RewardSignalsV2(
                previous_progress=previous_progress,
                current_progress=current_progress,
                lifecycle=str(start.info.get("controller_lifecycle", "EXECUTE_MOTION")),
                calibrated_attitude_error=_clamp01(attitude / 0.35),
                successful_fsm_attitude_envelope_excess=attitude_envelope_excess,
                body_angular_rate=_clamp01(angular_rate / 2.0),
                body_angular_acceleration=_clamp01(acceleration / 20.0),
                contact_motion_costs=self.signals.contact_costs(start, end),
                residual_first_difference=_clamp01(first_difference),
                residual_second_difference=_clamp01(second_difference),
                residual_magnitude=_clamp01(_rms(normalized)),
                phase_completion=phase_completed,
                final_success=final_success,
                task_failure=task_failure,
                safety_abort=safety_abort,
            )
        )

    @staticmethod
    def _trace_row(
        frame: AuthoritativeFrame,
        reward: RewardBreakdownV2,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        level = info.get("level_calibration", {})
        raw = info["raw_observation"]
        base = _member(raw, "base")
        return {
            "seed": info["seed"],
            "decision_index": info["decision_index"],
            "physics_tick": frame.physics_tick,
            "sim_time_s": frame.sim_time_s,
            "state_id": frame.state_id,
            "lifecycle": info.get("controller_lifecycle"),
            "roll_error_rad": level.get("roll_error_to_level_rad", 0.0),
            "pitch_error_rad": level.get("pitch_error_to_level_rad", 0.0),
            "roll_rate_rad_s": level.get("roll_change_rate_rad_s", 0.0),
            "pitch_rate_rad_s": level.get("pitch_change_rate_rad_s", 0.0),
            "angular_velocity_w_rad_s": list(_member(base, "angular_velocity_w_rad_s", (0.0, 0.0, 0.0))),
            "nominal_full12": list(frame.nominal_action_full12),
            "residual_full12": list(info["projected_residual_full12"]),
            "applied_full12": list(info["applied_action_full12"]),
            "reward": asdict(reward),
            "termination_reason": info["termination_reason"],
        }


class RslResidualVecEnv:
    """Minimal :mod:`rsl_rl` VecEnv over independent residual episodes."""

    def __init__(
        self,
        environments: Sequence[ResidualEpisodeEnv],
        *,
        seeds: Sequence[int],
        device: str = "cuda:0",
        training_phase_reset_schedule: Sequence[str] | None = None,
        end_curriculum_sample_at_phase_boundary: bool = False,
        phase_curriculum_max_decisions: int = DEFAULT_PHASE_CURRICULUM_MAX_DECISIONS,
        phase_curriculum_target_decision_fractions: Mapping[str, float] | None = None,
        phase_curriculum_occupancy_tolerance_fraction: float = DEFAULT_PHASE_CURRICULUM_OCCUPANCY_TOLERANCE,
    ) -> None:
        if not environments:
            raise ResidualDirectEnvError("at least one environment is required")
        if len(seeds) < len(environments):
            raise ResidualDirectEnvError("one reset seed is required per environment")
        import torch

        self.environments = tuple(environments)
        self.seed_schedule = tuple(int(seed) for seed in seeds)
        self._seed_cursor = len(environments)
        self.training_phase_reset_schedule = (
            None
            if training_phase_reset_schedule is None
            else tuple(str(phase_id) for phase_id in training_phase_reset_schedule)
        )
        if self.training_phase_reset_schedule is not None:
            if not self.training_phase_reset_schedule:
                raise ResidualDirectEnvError("phase reset schedule cannot be empty")
            invalid = tuple(
                phase_id
                for phase_id in self.training_phase_reset_schedule
                if phase_id not in STATE_IDS
            )
            if invalid:
                raise ResidualDirectEnvError(
                    f"phase reset schedule contains invalid states: {invalid}"
                )
        self.end_curriculum_sample_at_phase_boundary = bool(
            end_curriculum_sample_at_phase_boundary
        )
        if (
            self.end_curriculum_sample_at_phase_boundary
            and self.training_phase_reset_schedule is None
        ):
            raise ResidualDirectEnvError(
                "phase-boundary curriculum termination requires a reset schedule"
            )
        self.phase_curriculum_max_decisions = int(phase_curriculum_max_decisions)
        if (
            self.training_phase_reset_schedule is not None
            and self.phase_curriculum_max_decisions <= 0
        ):
            raise ResidualDirectEnvError(
                "phase curriculum decision horizon must be positive"
            )
        if phase_curriculum_target_decision_fractions is None:
            self.phase_curriculum_target_decision_fractions = None
        else:
            if self.training_phase_reset_schedule is None:
                raise ResidualDirectEnvError(
                    "phase decision occupancy target requires a reset schedule"
                )
            targets = _phase_float_mapping(
                phase_curriculum_target_decision_fractions,
                label="phase curriculum target decision fractions",
                strictly_positive=True,
            )
            if not math.isclose(
                sum(targets.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ResidualDirectEnvError(
                    "phase curriculum target decision fractions must sum to one"
                )
            self.phase_curriculum_target_decision_fractions = targets
        self.phase_curriculum_occupancy_tolerance_fraction = float(
            phase_curriculum_occupancy_tolerance_fraction
        )
        if (
            not math.isfinite(self.phase_curriculum_occupancy_tolerance_fraction)
            or not 0.0 <= self.phase_curriculum_occupancy_tolerance_fraction <= 1.0
        ):
            raise ResidualDirectEnvError(
                "phase curriculum occupancy tolerance must be within [0, 1]"
            )
        self._phase_reset_cursor = 0
        self._curriculum_start_states: list[str | None] = [
            None for _ in environments
        ]
        self.device = torch.device(device)
        self.num_envs = len(environments)
        self.num_actions = ACTION_DIMENSION
        self.max_episode_length = MAX_EPISODE_DECISIONS
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cfg = {
            "physics_hz": PHYSICS_HZ,
            "decision_hz": DECISION_HZ,
            "physics_ticks_per_decision": PHYSICS_TICKS_PER_DECISION,
            "training_phase_curriculum": self.training_phase_reset_schedule
            is not None,
            "phase_curriculum_max_decisions": (
                self.phase_curriculum_max_decisions
                if self.training_phase_reset_schedule is not None
                else None
            ),
            "phase_curriculum_target_decision_fractions": (
                None
                if self.phase_curriculum_target_decision_fractions is None
                else dict(self.phase_curriculum_target_decision_fractions)
            ),
            "phase_curriculum_occupancy_tolerance_fraction": (
                self.phase_curriculum_occupancy_tolerance_fraction
                if self.phase_curriculum_target_decision_fractions is not None
                else None
            ),
            "phase_curriculum_reset_cycle_samples": (
                None
                if self.training_phase_reset_schedule is None
                else len(self.training_phase_reset_schedule)
            ),
            "phase_curriculum_reset_cycle_counts": (
                None
                if self.training_phase_reset_schedule is None
                else {
                    phase_id: self.training_phase_reset_schedule.count(phase_id)
                    for phase_id in STATE_IDS
                }
            ),
        }
        self.completed_episodes: list[dict[str, Any]] = []
        self.policy_decision_count = 0
        self.phase_decision_counts = {phase_id: 0 for phase_id in STATE_IDS}
        self.phase_curriculum_reset_counts = {
            phase_id: 0 for phase_id in STATE_IDS
        }
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
        reward_config = getattr(
            getattr(self.environments[0], "reward_calculator", None),
            "config",
            None,
        )
        if reward_config is None:
            reward_config = load_reward_v2_config()
        self._maximum_single_reward_family_fraction = float(
            reward_config.maximum_single_dense_family_fraction
        )
        self._maximum_residual_reward_fraction = float(
            reward_config.maximum_residual_regularization_fraction
        )
        self._minimum_absolute_dense_return = float(
            reward_config.minimum_absolute_dense_return
        )
        observations = []
        for index, (env, seed) in enumerate(
            zip(self.environments, self.seed_schedule, strict=False)
        ):
            observation = self._reset_environment(index, env, seed)
            observations.append(observation)
        self._observations = torch.tensor(observations, dtype=torch.float32, device=self.device)

    def _next_curriculum_phase(self) -> str | None:
        schedule = self.training_phase_reset_schedule
        if schedule is None:
            return None
        phase_id = schedule[self._phase_reset_cursor % len(schedule)]
        self._phase_reset_cursor += 1
        return phase_id

    def _reset_environment(
        self, index: int, env: ResidualEpisodeEnv, seed: int
    ) -> tuple[float, ...]:
        phase_id = self._next_curriculum_phase()
        if phase_id is None:
            observation, _ = env.reset(seed=seed)
        else:
            observation, _ = env.reset(
                seed=seed,
                options={"training_phase_snapshot": phase_id},
            )
            actual_phase = None if env.frame is None else env.frame.state_id
            if actual_phase != phase_id:
                raise ResidualDirectEnvError(
                    f"phase snapshot reset requested {phase_id} but began in {actual_phase}"
                )
            self.phase_curriculum_reset_counts[phase_id] += 1
        self._curriculum_start_states[index] = phase_id
        return observation

    def get_observations(self) -> Any:
        return self.observation_tensor_dict(self._observations)

    def observation_tensor_dict(self, observations: Any) -> Any:
        """Build an RSL observation TensorDict without stepping or auto-resetting.

        Evaluation deliberately calls this with the live episode's latest
        observation and then steps :class:`ResidualEpisodeEnv` directly.  The
        ordinary vector-environment ``step`` method remains training-only and
        may therefore keep its required auto-reset behavior.
        """

        from tensordict import TensorDict
        import torch

        if isinstance(observations, torch.Tensor):
            tensor = observations.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        if tuple(tensor.shape) != (self.num_envs, OBSERVATION_DIMENSION_V2):
            raise ResidualDirectEnvError(
                "observation batch must have shape "
                f"{(self.num_envs, OBSERVATION_DIMENSION_V2)}"
            )

        return TensorDict(
            {"policy": tensor, "critic": tensor},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        import torch

        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ResidualDirectEnvError(
                f"actions must have shape {(self.num_envs, self.num_actions)}"
            )
        action_rows = actions.detach().to("cpu").tolist()
        next_observations = []
        rewards = []
        dones = []
        time_outs = []
        log: dict[str, float] = {}
        for index, (env, action) in enumerate(zip(self.environments, action_rows, strict=True)):
            action_phase = None if env.frame is None else str(env.frame.state_id)
            if action_phase not in STATE_IDS:
                raise ResidualDirectEnvError(
                    "training step has no authoritative P01-P13 action phase"
                )
            start_phase = self._curriculum_start_states[index]
            if self.end_curriculum_sample_at_phase_boundary:
                step = env.step(action, stop_after_phase_id=start_phase)
            else:
                step = env.step(action)
            rewards.append(step.reward)
            self.policy_decision_count += 1
            self.phase_decision_counts[action_phase] += 1
            weighted = step.info.get("reward", {}).get("weighted_dense", {})
            if isinstance(weighted, Mapping) and tuple(weighted) == DENSE_FAMILIES:
                for family in DENSE_FAMILIES:
                    value = float(weighted[family])
                    self.reward_family_signed_sums[family] += value
                    self.reward_family_absolute_sums[family] += abs(value)
                    self.reward_family_absolute_sums_by_phase[action_phase][
                        family
                    ] += abs(value)
            else:
                # Dependency-injected lightweight kernels used by contract
                # tests may expose only a scalar reward.  A real training run
                # is rejected by the CLI unless this counter remains zero.
                self.reward_telemetry_incomplete_count += 1
            next_length = int(self.episode_length_buf[index].item()) + 1
            curriculum_horizon = bool(
                self.training_phase_reset_schedule is not None
                and not step.terminated
                and not step.truncated
                and next_length >= self.phase_curriculum_max_decisions
            )
            done = step.terminated or step.truncated or curriculum_horizon
            dones.append(done)
            time_outs.append(step.truncated or curriculum_horizon)
            self.episode_length_buf[index] += 1
            if done:
                termination_reason = (
                    PHASE_CURRICULUM_HORIZON_REASON
                    if curriculum_horizon
                    else step.info.get("termination_reason")
                )
                trace = list(env.trace)
                if curriculum_horizon and trace:
                    trace[-1] = {
                        **trace[-1],
                        "termination_reason": PHASE_CURRICULUM_HORIZON_REASON,
                    }
                self.completed_episodes.append(
                    {
                        "seed": env.seed,
                        "length": int(self.episode_length_buf[index].item()),
                        "duration_s": float(step.info.get("sim_time_s", env.frame.sim_time_s if env.frame else 0.0)),
                        "termination_reason": termination_reason,
                        "trace": trace,
                        "phase_curriculum_start_state_id": start_phase,
                        "phase_curriculum_boundary": bool(
                            step.info.get("phase_curriculum_boundary", False)
                        ),
                        "phase_curriculum_horizon": curriculum_horizon,
                    }
                )
                seed = self.seed_schedule[self._seed_cursor % len(self.seed_schedule)]
                self._seed_cursor += 1
                observation = self._reset_environment(index, env, seed)
                self.episode_length_buf[index] = 0
                next_observations.append(observation)
            else:
                next_observations.append(step.observation)
        self._observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
        extras = {
            "time_outs": torch.tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": log,
        }
        return self.get_observations(), reward_tensor, done_tensor, extras

    def training_telemetry(self) -> Mapping[str, Any]:
        """Return deterministic aggregate rollout evidence for reports/audits."""

        total = int(self.policy_decision_count)
        occupancy = {
            phase_id: self.phase_decision_counts[phase_id] / total if total else 0.0
            for phase_id in STATE_IDS
        }
        targets = self.phase_curriculum_target_decision_fractions
        occupancy_error = (
            None
            if targets is None
            else {
                phase_id: abs(occupancy[phase_id] - targets[phase_id])
                for phase_id in STATE_IDS
            }
        )
        occupancy_violations = (
            []
            if occupancy_error is None
            else [
                phase_id
                for phase_id in STATE_IDS
                if occupancy_error[phase_id]
                > self.phase_curriculum_occupancy_tolerance_fraction
            ]
        )
        reward_telemetry = build_reward_dominance_telemetry(
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
        )
        return {
            "schema": "wlr50_clean.ppo_training_telemetry.v1",
            "policy_decision_count": total,
            "phase_decision_counts": dict(self.phase_decision_counts),
            "phase_curriculum_reset_counts": dict(
                self.phase_curriculum_reset_counts
            ),
            "phase_occupancy_fraction": occupancy,
            "phase_curriculum_target_decision_fraction": (
                None if targets is None else dict(targets)
            ),
            "phase_curriculum_occupancy_absolute_error": occupancy_error,
            "phase_curriculum_occupancy_tolerance_fraction": (
                self.phase_curriculum_occupancy_tolerance_fraction
                if targets is not None
                else None
            ),
            "phase_curriculum_occupancy_violations": occupancy_violations,
            "phase_curriculum_occupancy_within_tolerance": (
                None if targets is None else bool(total > 0 and not occupancy_violations)
            ),
            **reward_telemetry,
            **build_completed_episode_telemetry(self.completed_episodes),
            "completed_sample_count": len(self.completed_episodes),
        }


def make_live_single_env(
    simulation_app: Any,
    *,
    seed: int,
    device: str = "cuda:0",
    collect_trace: bool = False,
) -> RslResidualVecEnv:
    """Construct the one-scene live backend after ``AppLauncher`` starts."""

    from .isaac_fsm_backend import IsaacFSMBackend

    episode = ResidualEpisodeEnv(
        IsaacFSMBackend(simulation_app), collect_trace=collect_trace
    )
    return RslResidualVecEnv([episode], seeds=[seed], device=device)


__all__ = [
    "ACTION_DIMENSION",
    "DECISION_HZ",
    "DEFAULT_PHASE_CURRICULUM_MAX_DECISIONS",
    "DEFAULT_PHASE_CURRICULUM_OCCUPANCY_TOLERANCE",
    "DEFAULT_PHASE_CURRICULUM_RESET_CYCLE_SAMPLES",
    "LivePhysicalSignalBuilder",
    "MAX_EPISODE_DECISIONS",
    "PHASE_CURRICULUM_BOUNDARY_REASON",
    "PHASE_CURRICULUM_HORIZON_REASON",
    "PHASE_CURRICULUM_BASELINE_DECISIONS",
    "PHASE_CURRICULUM_PRIORITY_STATES",
    "PHASE_CURRICULUM_RESET_CYCLE",
    "PHASE_CURRICULUM_TARGET_DECISION_FRACTIONS",
    "PHYSICS_HZ",
    "PHYSICS_TICKS_PER_DECISION",
    "ResidualDirectEnvError",
    "ResidualEpisodeEnv",
    "ResidualStep",
    "RslResidualVecEnv",
    "build_phase_curriculum_reset_cycle",
    "build_completed_episode_telemetry",
    "build_reward_dominance_telemetry",
    "make_live_single_env",
]
