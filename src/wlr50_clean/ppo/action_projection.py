"""Bounded Residual-PPO action projection in canonical Full12 space."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from wlr50_clean.conformance_policy import ConformancePolicy, get_conformance_policy
from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    KNEE_NAMES,
    WHEEL_ORDER,
    servo_limits_deg,
)
from wlr50_clean.reference.motion_contract import load_motion_contract
from wlr50_clean.reference.similarity import allowed_error


ACTION_SCHEMA = "wlr50_clean.ppo_action_projection.v1"
ACTION_DIMENSION = 12
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_action_projection.yaml"
)


class ActionProjectionError(ValueError):
    """A residual action or its projection context violates the action ABI."""


def _vector(
    values: Sequence[float], label: str, *, finite: bool = True
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ActionProjectionError(f"{label} must be numeric") from exc
    if len(result) != ACTION_DIMENSION:
        raise ActionProjectionError(
            f"{label} must contain 12 values; received {len(result)}"
        )
    if finite and any(not math.isfinite(value) for value in result):
        raise ActionProjectionError(f"{label} contains a NaN or infinity")
    return result


def _mask(values: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ActionProjectionError(f"{label} must be binary") from exc
    if len(result) != ACTION_DIMENSION or any(value not in (0, 1) for value in result):
        raise ActionProjectionError(f"{label} must contain twelve binary values")
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _absolute_limits() -> tuple[tuple[float, float], ...]:
    return tuple(servo_limits_deg(name) for name in FULL12_ORDER[:8]) + tuple(
        (-2.0943951023931953, 2.0943951023931953) for _ in WHEEL_ORDER
    )


def full12_bytes(values: Sequence[float]) -> bytes:
    """IEEE-754 representation used by zero-residual equivalence audits."""

    return struct.pack(">12d", *_vector(values, "full12"))


def bitwise_full12_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    return full12_bytes(left) == full12_bytes(right)


@dataclass(frozen=True, slots=True)
class SafetyProjection:
    """Authoritative final safety constraints, evaluated after PPO bounds."""

    residual_enabled: bool = True
    channel_mask_full12: tuple[int, ...] = (1,) * ACTION_DIMENSION
    force_wheels_zero: bool = False
    body_collision_detected: bool = False
    wheel_only_climb_detected: bool = False
    override_full12: tuple[float, ...] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_mask_full12",
            _mask(self.channel_mask_full12, "safety.channel_mask_full12"),
        )
        if self.override_full12 is not None:
            override = _vector(self.override_full12, "safety.override_full12")
            for index, (value, (lower, upper)) in enumerate(
                zip(override, _absolute_limits(), strict=True)
            ):
                if not lower <= value <= upper:
                    raise ActionProjectionError(
                        "safety.override_full12 exceeds the absolute actuator "
                        f"limit at index {index}"
                    )
            object.__setattr__(
                self,
                "override_full12",
                override,
            )

    @property
    def neutral(self) -> bool:
        return bool(
            self.residual_enabled
            and self.channel_mask_full12 == (1,) * ACTION_DIMENSION
            and not self.force_wheels_zero
            and not self.body_collision_detected
            and not self.wheel_only_climb_detected
            and self.override_full12 is None
        )


@dataclass(frozen=True, slots=True)
class ActionProjectionConfig:
    schema: str
    action_schema_name: str
    action_schema_version: int
    physics_hz: float
    decision_hz: float
    bounded_transform: str
    phase_scale_full12: Mapping[str, tuple[float, ...]]
    physical_residual_scale_full12: tuple[float, ...]
    phase_mask_full12: Mapping[str, tuple[int, ...]]
    servo_residual_rate_deg_s: float
    wheel_residual_rate_rad_s2: float
    absolute_limits_full12: tuple[tuple[float, float], ...]
    safety_limits_full12: tuple[tuple[float, float], ...]
    recording_envelope_hard_constraint: bool
    recording_envelope_initialization_suggestion: bool
    training_enabled: bool
    path: Path

    @property
    def physics_ticks_per_decision(self) -> int:
        return round(self.physics_hz / self.decision_hz)

    def mask_for(self, state_id: str) -> tuple[int, ...]:
        try:
            return self.phase_mask_full12[state_id]
        except KeyError as exc:
            raise ActionProjectionError(f"unknown state_id {state_id!r}") from exc

    def scale_for(self, state_id: str) -> tuple[float, ...]:
        try:
            return self.phase_scale_full12[state_id]
        except KeyError as exc:
            raise ActionProjectionError(f"unknown state_id {state_id!r}") from exc


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    state_id: str
    raw_residual_full12: tuple[float, ...]
    bounded_residual_full12: tuple[float, ...]
    scaled_residual_full12: tuple[float, ...]
    masked_residual_full12: tuple[float, ...]
    physical_residual_scale_full12: tuple[float, ...]
    recording_scale_suggestion_full12: tuple[float, ...]
    nominal_recording_error_full12: tuple[float, ...]
    remaining_recording_envelope_diagnostic_full12: tuple[float, ...]
    recording_envelope_exceeded_full12: tuple[bool, ...]
    rate_projected_residual_full12: tuple[float, ...]
    phase_scale_projected_residual_full12: tuple[float, ...]
    limit_projected_residual_full12: tuple[float, ...]
    safe_projected_residual_full12: tuple[float, ...]
    applied_action_full12: tuple[float, ...]
    effective_action_mask_full12: tuple[int, ...]
    zero_residual_fast_path: bool
    hard_safety_modified: bool
    clipping_stages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ZeroResidualEpisodeAudit:
    status: str
    tick_count: int
    nominal_sequence_sha256: str
    applied_sequence_sha256: str
    bitwise_equal: bool


class ZeroResidualEpisodeAuditor:
    """Streaming bitwise proof for a complete frozen-baseline episode."""

    def __init__(self) -> None:
        self._nominal = hashlib.sha256()
        self._applied = hashlib.sha256()
        self._count = 0
        self._equal = True

    def append(
        self, nominal_action_full12: Sequence[float], applied_action_full12: Sequence[float]
    ) -> None:
        nominal = full12_bytes(nominal_action_full12)
        applied = full12_bytes(applied_action_full12)
        self._nominal.update(nominal)
        self._applied.update(applied)
        self._count += 1
        self._equal = self._equal and nominal == applied

    def finalize(self) -> ZeroResidualEpisodeAudit:
        if self._count <= 0:
            raise ActionProjectionError("zero-residual episode audit is empty")
        return ZeroResidualEpisodeAudit(
            status=(
                "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE"
                if self._equal
                else "ZERO_RESIDUAL_FULL_EPISODE_MISMATCH"
            ),
            tick_count=self._count,
            nominal_sequence_sha256=self._nominal.hexdigest(),
            applied_sequence_sha256=self._applied.hexdigest(),
            bitwise_equal=self._equal,
        )


def load_action_projection_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> ActionProjectionConfig:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != ACTION_SCHEMA:
        raise ActionProjectionError("unexpected action-projection schema")
    if int(payload.get("nominal_action_dimension", -1)) != ACTION_DIMENSION:
        raise ActionProjectionError("nominal action dimension must remain 12")
    if int(payload.get("residual_action_dimension", -1)) != ACTION_DIMENSION:
        raise ActionProjectionError("residual action dimension must remain 12")
    if tuple(payload.get("full12_order", ())) != FULL12_ORDER:
        raise ActionProjectionError("Full12 action order differs from the canonical ABI")
    physics_hz = float(payload["physics_hz"])
    decision_hz = float(payload["decision_hz"])
    ratio = physics_hz / decision_hz
    if physics_hz != 120.0 or decision_hz != 15.0 or abs(ratio - 8.0) > 1.0e-12:
        raise ActionProjectionError("PPO action interface must remain 120/15 Hz")
    if payload.get("bounded_transform") != "tanh":
        raise ActionProjectionError("the bounded transform must be tanh")
    if bool(payload.get("training_enabled")):
        raise ActionProjectionError("PPO training must remain disabled")
    diagnostic = payload.get("recording_envelope_diagnostic", {})
    if bool(diagnostic.get("hard_projection_constraint", True)):
        raise ActionProjectionError(
            "the Recording envelope is diagnostic and cannot be a hard projection"
        )
    if not bool(
        diagnostic.get("use_as_initial_policy_initialization_suggestion", False)
    ):
        raise ActionProjectionError(
            "the Recording envelope must remain an explicit initial-scale suggestion"
        )
    output_scale = payload.get("residual_output_scale", {})
    if (
        output_scale.get("derivation") != "configured_safe_action_range_span"
        or bool(output_scale.get("recording_envelope_used_in_projection", True))
    ):
        raise ActionProjectionError(
            "residual output scale must derive from physical ranges, not Recording"
        )

    source = payload.get("phase_action_mask", {})
    if source.get("derive_from") != "phases.ppo_action_mask_full12":
        raise ActionProjectionError(
            "phase masks must derive from phases.ppo_action_mask_full12"
        )
    contract_path = selected.parent / str(source["source"])
    contract = load_motion_contract(contract_path)
    if contract.full12_order != FULL12_ORDER:
        raise ActionProjectionError("motion contract action order differs")
    masks: dict[str, tuple[int, ...]] = {}
    for phase in contract.phases:
        masks[phase.state_id] = _mask(
            phase.action_mask_full12,
            f"motion_contract.{phase.state_id}.ppo_action_mask_full12",
        )
    if tuple(masks) != STATE_IDS:
        raise ActionProjectionError("phase-mask source must contain ordered P01-P13")

    raw_scales = payload.get("phase_scale_fraction", {})
    if tuple(raw_scales) != STATE_IDS:
        raise ActionProjectionError("phase scales must contain ordered P01-P13")
    scales: dict[str, tuple[float, ...]] = {}
    for state_id, raw in raw_scales.items():
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = _vector(raw, f"phase_scale_fraction.{state_id}")
        else:
            values = (float(raw),) * ACTION_DIMENSION
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ActionProjectionError("phase scale fractions must be within [0,1]")
        scales[str(state_id)] = values

    rate = payload.get("residual_rate_limits", {})
    servo_rate = float(rate["servo_deg_s"])
    wheel_rate = float(rate["wheel_rad_s2"])
    if servo_rate <= 0.0 or wheel_rate <= 0.0:
        raise ActionProjectionError("residual rate limits must be positive")
    configured_limits = payload.get("absolute_action_limits", {})
    expected_limits = _absolute_limits()
    declared_limits = (
        tuple(tuple(float(item) for item in configured_limits["hip_deg"]) if name not in KNEE_NAMES else tuple(float(item) for item in configured_limits["knee_deg"]) for name in FULL12_ORDER[:8])
        + tuple(tuple(float(item) for item in configured_limits["wheel_rad_s"]) for _ in WHEEL_ORDER)
    )
    if declared_limits != expected_limits:
        raise ActionProjectionError("configured absolute limits differ from command ABI")
    raw_margins = payload.get("joint_safety_margin_deg", {})
    hip_margin = float(raw_margins.get("hip", float("nan")))
    knee_margin = float(raw_margins.get("knee", float("nan")))
    if (
        not math.isfinite(hip_margin)
        or not math.isfinite(knee_margin)
        or hip_margin < 0.0
        or knee_margin < 0.0
    ):
        raise ActionProjectionError("joint safety margins must be finite and non-negative")
    margins = tuple(
        knee_margin if name in KNEE_NAMES else hip_margin
        for name in FULL12_ORDER[:8]
    ) + (0.0,) * 4
    safety_limits = tuple(
        (lower + margin, upper - margin)
        for (lower, upper), margin in zip(expected_limits, margins, strict=True)
    )
    if any(lower >= upper for lower, upper in safety_limits):
        raise ActionProjectionError("joint safety margin consumes an actuator range")
    physical_residual_scale = tuple(
        upper - lower for lower, upper in safety_limits
    )
    physical = payload.get("physical_safety_projection", {})
    required_physical = (
        "body_collision_disables_all_residuals",
        "body_collision_forces_wheels_zero",
        "wheel_only_climb_disables_all_residuals",
        "wheel_only_climb_forces_wheels_zero",
    )
    if not all(bool(physical.get(name)) for name in required_physical):
        raise ActionProjectionError(
            "body-collision and wheel-only safety projection must be explicit"
        )
    hard_safety = payload.get("hard_safety", {})
    if not bool(hard_safety.get("applied_after_all_physical_projection_stages")):
        raise ActionProjectionError(
            "hard safety must follow all physical projection stages"
        )
    zero = payload.get("zero_residual", {})
    if not zero.get("bitwise_nominal_fast_path") or not zero.get(
        "full_episode_equivalence_required"
    ):
        raise ActionProjectionError("zero-residual bitwise equivalence is mandatory")
    return ActionProjectionConfig(
        schema=ACTION_SCHEMA,
        action_schema_name=str(payload["action_schema_name"]),
        action_schema_version=int(payload["action_schema_version"]),
        physics_hz=physics_hz,
        decision_hz=decision_hz,
        bounded_transform="tanh",
        phase_scale_full12=scales,
        physical_residual_scale_full12=physical_residual_scale,
        phase_mask_full12=masks,
        servo_residual_rate_deg_s=servo_rate,
        wheel_residual_rate_rad_s2=wheel_rate,
        absolute_limits_full12=expected_limits,
        safety_limits_full12=safety_limits,
        recording_envelope_hard_constraint=False,
        recording_envelope_initialization_suggestion=True,
        training_enabled=False,
        path=selected,
    )


class ActionProjector:
    """Pure projection; the caller owns the prior residual between ticks."""

    def __init__(
        self,
        config: ActionProjectionConfig | None = None,
        policy: ConformancePolicy | None = None,
    ) -> None:
        self.config = config or load_action_projection_config()
        self.policy = policy or get_conformance_policy()
        if self.policy.active_fraction != 0.30:
            raise ActionProjectionError(
                "Residual PPO diagnostics require the central 30% policy"
            )

    def project(
        self,
        raw_residual_full12: Sequence[float],
        *,
        state_id: str,
        nominal_action_full12: Sequence[float],
        reference_action_full12: Sequence[float],
        reference_delta_full12: Sequence[float],
        previous_projected_residual_full12: Sequence[float] = (0.0,) * 12,
        runtime_action_mask_full12: Sequence[int] | None = None,
        safety: SafetyProjection | None = None,
        dt_s: float | None = None,
    ) -> ProjectionResult:
        raw = _vector(raw_residual_full12, "raw_residual_full12")
        nominal = _vector(nominal_action_full12, "nominal_action_full12")
        reference = _vector(reference_action_full12, "reference_action_full12")
        delta = _vector(reference_delta_full12, "reference_delta_full12")
        previous = _vector(
            previous_projected_residual_full12,
            "previous_projected_residual_full12",
        )
        safety_projection = safety or SafetyProjection()
        phase_mask = self.config.mask_for(state_id)
        runtime_mask = (
            (1,) * ACTION_DIMENSION
            if runtime_action_mask_full12 is None
            else _mask(runtime_action_mask_full12, "runtime_action_mask_full12")
        )
        effective_mask = tuple(
            phase * runtime * safe
            for phase, runtime, safe in zip(
                phase_mask,
                runtime_mask,
                safety_projection.channel_mask_full12,
                strict=True,
            )
        )
        step_s = 1.0 / self.config.physics_hz if dt_s is None else float(dt_s)
        if not math.isfinite(step_s) or step_s <= 0.0:
            raise ActionProjectionError("projection dt_s must be positive and finite")
        for value, (lower, upper) in zip(
            nominal, self.config.absolute_limits_full12, strict=True
        ):
            if not lower <= value <= upper:
                raise ActionProjectionError("frozen nominal action exceeds an absolute limit")

        joint_floor = self.policy.floor("joint_endpoint_delta").absolute_allowance
        wheel_floor = self.policy.floor("wheel_velocity").absolute_allowance
        # Use the same absolute-floor semantics as offline conformance.  In
        # particular, a zero-reference wheel retains the validated 0.05 rad/s
        # absolute allowance; the floor is not multiplied by 30% a second time.
        recording_scale = tuple(
            allowed_error(
                delta[index] if index < 8 else reference[index],
                absolute_floor=joint_floor if index < 8 else wheel_floor,
                fraction=self.policy.active_fraction,
            )
            for index in range(ACTION_DIMENSION)
        )
        nominal_deviation = tuple(
            abs(value - reference_value)
            for value, reference_value in zip(nominal, reference, strict=True)
        )
        diagnostic_remaining = tuple(
            max(0.0, limit - used)
            for limit, used in zip(recording_scale, nominal_deviation, strict=True)
        )

        # This branch deliberately returns the original float payload before
        # tanh/add/clamp.  It preserves signed zero and every IEEE-754 bit.
        if (
            all(value == 0.0 for value in raw)
            and all(value == 0.0 for value in previous)
            and safety_projection.neutral
        ):
            zeros = (0.0,) * ACTION_DIMENSION
            return ProjectionResult(
                state_id=state_id,
                raw_residual_full12=raw,
                bounded_residual_full12=zeros,
                scaled_residual_full12=zeros,
                masked_residual_full12=zeros,
                physical_residual_scale_full12=(
                    self.config.physical_residual_scale_full12
                ),
                recording_scale_suggestion_full12=recording_scale,
                nominal_recording_error_full12=nominal_deviation,
                remaining_recording_envelope_diagnostic_full12=diagnostic_remaining,
                recording_envelope_exceeded_full12=tuple(
                    used > limit
                    for used, limit in zip(
                        nominal_deviation, recording_scale, strict=True
                    )
                ),
                rate_projected_residual_full12=zeros,
                phase_scale_projected_residual_full12=zeros,
                limit_projected_residual_full12=zeros,
                safe_projected_residual_full12=zeros,
                applied_action_full12=nominal,
                effective_action_mask_full12=effective_mask,
                zero_residual_fast_path=True,
                hard_safety_modified=False,
                clipping_stages=(),
            )

        bounded = tuple(math.tanh(value) for value in raw)
        scales = self.config.scale_for(state_id)
        scaled = tuple(
            value * phase_scale * physical_scale
            for value, phase_scale, physical_scale in zip(
                bounded,
                scales,
                self.config.physical_residual_scale_full12,
                strict=True,
            )
        )
        masked = tuple(
            value * mask for value, mask in zip(scaled, effective_mask, strict=True)
        )
        max_steps = tuple(
            (
                self.config.servo_residual_rate_deg_s * step_s
                if index < 8
                else self.config.wheel_residual_rate_rad_s2 * step_s
            )
            for index in range(ACTION_DIMENSION)
        )
        rate_candidate = tuple(
            _clamp(value, old - maximum, old + maximum)
            for value, old, maximum in zip(
                masked, previous, max_steps, strict=True
            )
        )
        # Phase/runtime/safety masks are hard constraints.  Reapply them after
        # slew limiting so a non-zero prior residual cannot leak into a channel
        # that just became inactive.
        rate_projected = tuple(
            value if mask else 0.0
            for value, mask in zip(rate_candidate, effective_mask, strict=True)
        )
        # The rate limiter is stateful and therefore cannot, by itself, enforce
        # a newly smaller phase scale when ``previous`` came from another
        # phase (or from an untrusted caller).  Reapply the current phase's
        # physical M*S bound after slew/mask projection.  Hard safety still
        # runs last and may deliberately override this policy residual bound.
        physical_phase_caps = tuple(
            phase_scale * physical_scale
            for phase_scale, physical_scale in zip(
                scales,
                self.config.physical_residual_scale_full12,
                strict=True,
            )
        )
        phase_scale_projected = tuple(
            _clamp(value, -cap, cap)
            for value, cap in zip(
                rate_projected, physical_phase_caps, strict=True
            )
        )
        # Servo limits reserve the configured safety margin.  Work directly in
        # residual space so the projection never subtracts two nearly equal
        # absolute commands to recover a tiny residual.  Besides avoiding
        # cancellation, this makes the ownership boundary explicit: the
        # frozen FSM nominal is immutable, and only its feasible delta is
        # clipped.  A nominal already outside the reserved band may move inward
        # but is never displaced merely because the residual path is active.
        residual_intervals = tuple(
            (
                min(0.0, lower - nominal_value),
                max(0.0, upper - nominal_value),
            )
            for nominal_value, (lower, upper) in zip(
                nominal, self.config.safety_limits_full12, strict=True
            )
        )
        limit_residual = tuple(
            _clamp(residual_value, lower, upper)
            for residual_value, (lower, upper) in zip(
                phase_scale_projected, residual_intervals, strict=True
            )
        )
        limit_action = tuple(
            nominal_value
            if residual_value == 0.0
            else nominal_value + residual_value
            for nominal_value, residual_value in zip(
                nominal, limit_residual, strict=True
            )
        )

        physical_stop = bool(
            safety_projection.body_collision_detected
            or safety_projection.wheel_only_climb_detected
        )
        safe_action = limit_action
        if not safety_projection.residual_enabled or physical_stop:
            safe_action = nominal
        else:
            safe_action = tuple(
                action if mask else nominal_value
                for action, nominal_value, mask in zip(
                    safe_action, nominal, safety_projection.channel_mask_full12, strict=True
                )
            )
        if safety_projection.force_wheels_zero or physical_stop:
            safe_action = safe_action[:8] + (0.0,) * 4
        if safety_projection.override_full12 is not None:
            safe_action = safety_projection.override_full12
        # Even an authoritative safe-hold override cannot restart wheels while
        # either physical stop condition is active.
        if physical_stop:
            safe_action = safe_action[:8] + (0.0,) * 4
        safe_residual = tuple(
            value - nominal_value
            for value, nominal_value in zip(safe_action, nominal, strict=True)
        )
        stages = []
        if rate_candidate != masked:
            stages.append("residual_rate_limit")
        if rate_projected != rate_candidate:
            stages.append("phase_active_mask_post_rate")
        if phase_scale_projected != rate_projected:
            stages.append("phase_residual_scale_cap_post_rate")
        if limit_residual != phase_scale_projected:
            stages.append("joint_safety_margin_or_wheel_speed_limit")
        safety_modified = safe_action != limit_action
        if safety_modified:
            if physical_stop:
                stages.append("body_collision_or_wheel_only_safety")
            else:
                stages.append("hard_safety")
        final_recording_error = tuple(
            abs(value - reference_value)
            for value, reference_value in zip(safe_action, reference, strict=True)
        )
        return ProjectionResult(
            state_id=state_id,
            raw_residual_full12=raw,
            bounded_residual_full12=bounded,
            scaled_residual_full12=scaled,
            masked_residual_full12=masked,
            physical_residual_scale_full12=(
                self.config.physical_residual_scale_full12
            ),
            recording_scale_suggestion_full12=recording_scale,
            nominal_recording_error_full12=nominal_deviation,
            remaining_recording_envelope_diagnostic_full12=diagnostic_remaining,
            recording_envelope_exceeded_full12=tuple(
                used > limit
                for used, limit in zip(
                    final_recording_error, recording_scale, strict=True
                )
            ),
            rate_projected_residual_full12=rate_projected,
            phase_scale_projected_residual_full12=phase_scale_projected,
            limit_projected_residual_full12=limit_residual,
            safe_projected_residual_full12=safe_residual,
            applied_action_full12=tuple(safe_action),
            effective_action_mask_full12=effective_mask,
            zero_residual_fast_path=False,
            hard_safety_modified=safety_modified,
            clipping_stages=tuple(stages),
        )
