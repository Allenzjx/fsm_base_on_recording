"""Trainable phase-specific residual action policy layered on ActionProjector.

The v2 file owns phase roles, masks, and deliberately small physical-unit
caps.  It does not duplicate actuator limits, slew projection, or hard safety;
those remain authoritative in :mod:`wlr50_clean.ppo.action_projection`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from wlr50_clean.infrastructure.command_batch import FULL12_ORDER
from wlr50_clean.ppo.action_projection import (
    ACTION_DIMENSION,
    STATE_IDS,
    ActionProjectionConfig,
    ActionProjectionError,
    ActionProjector,
    ProjectionResult,
    SafetyProjection,
    load_action_projection_config,
)


PHASE_ACTION_SCHEMA_V2 = "wlr50_clean.ppo_phase_action_masks.v2"
DEFAULT_PHASE_ACTION_CONFIG_V2 = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "ppo_phase_action_masks_v2.yaml"
)
_ZERO12 = (0.0,) * ACTION_DIMENSION
_ONE12 = (1,) * ACTION_DIMENSION
_SENSITIVITY_TIERS = frozenset({"low", "medium", "high", "blocked"})


class PhaseActionV2Error(ActionProjectionError):
    """The v2 phase-action contract is malformed or used unsafely."""


def _numeric_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise PhaseActionV2Error(f"{label} must contain twelve numeric values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise PhaseActionV2Error(f"{label} must contain twelve numeric values") from exc
    if len(result) != ACTION_DIMENSION:
        raise PhaseActionV2Error(
            f"{label} must contain twelve values; received {len(result)}"
        )
    if any(not math.isfinite(value) for value in result):
        raise PhaseActionV2Error(f"{label} contains a NaN or infinity")
    return result


def _binary_mask(values: Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise PhaseActionV2Error(f"{label} must contain twelve binary values")
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise PhaseActionV2Error(f"{label} must contain twelve binary values") from exc
    if len(result) != ACTION_DIMENSION or any(value not in (0, 1) for value in result):
        raise PhaseActionV2Error(f"{label} must contain twelve binary values")
    return result


def _text_vector(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PhaseActionV2Error(f"{label} must contain twelve text values")
    result = tuple(str(value) for value in values)
    if len(result) != ACTION_DIMENSION or any(not value for value in result):
        raise PhaseActionV2Error(f"{label} must contain twelve non-empty text values")
    return result


@dataclass(frozen=True, slots=True)
class PhaseChannelPolicyV2:
    state_id: str
    objective: str
    mask_full12: tuple[int, ...]
    scale_full12: tuple[float, ...]
    role_full12: tuple[str, ...]
    sensitivity_full12: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseActionScaleAuditRow:
    """One CSV-ready explanation of a phase/channel residual cap."""

    state_id: str
    objective: str
    channel_index: int
    channel_name: str
    actuator_family: str
    unit: str
    enabled: bool
    role: str
    sensitivity_tier: str
    absolute_lower: float
    absolute_upper: float
    safety_lower: float
    safety_upper: float
    safety_span: float
    residual_scale: float
    safe_span_fraction: float
    recording_envelope_hard_cap: bool = False
    derivation: str = "safe_span_x_phase_role_x_sensitivity_tier"

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class PhaseActionMasksV2:
    schema: str
    action_schema_name: str
    action_schema_version: int
    training_enabled: bool
    physics_hz: float
    decision_hz: float
    phases: Mapping[str, PhaseChannelPolicyV2]
    base_projection: ActionProjectionConfig
    servo_residual_rate_deg_s: float
    wheel_residual_rate_rad_s2: float
    maximum_initial_safe_span_fraction: float
    maximum_initial_servo_cap_deg: float
    maximum_initial_wheel_cap_rad_s: float
    sensitivity_method: str
    live_small_perturbation_smoke_required: bool
    path: Path

    @property
    def physics_ticks_per_decision(self) -> int:
        return round(self.physics_hz / self.decision_hz)

    def policy_for(self, state_id: str) -> PhaseChannelPolicyV2:
        try:
            return self.phases[state_id]
        except KeyError as exc:
            raise PhaseActionV2Error(f"unknown state_id {state_id!r}") from exc

    def mask_for(self, state_id: str) -> tuple[int, ...]:
        return self.policy_for(state_id).mask_full12

    def physical_scale_for(self, state_id: str) -> tuple[float, ...]:
        return self.policy_for(state_id).scale_full12

    def projector_scale_fraction_for(self, state_id: str) -> tuple[float, ...]:
        return tuple(
            scale / span
            for scale, span in zip(
                self.physical_scale_for(state_id),
                self.base_projection.physical_residual_scale_full12,
                strict=True,
            )
        )

    def action_projection_config(self) -> ActionProjectionConfig:
        """Inject v2 masks/scales/rates into the mature physical projector."""

        return replace(
            self.base_projection,
            schema=PHASE_ACTION_SCHEMA_V2,
            action_schema_name=self.action_schema_name,
            action_schema_version=self.action_schema_version,
            phase_scale_full12={
                state_id: self.projector_scale_fraction_for(state_id)
                for state_id in STATE_IDS
            },
            phase_mask_full12={
                state_id: self.mask_for(state_id) for state_id in STATE_IDS
            },
            servo_residual_rate_deg_s=self.servo_residual_rate_deg_s,
            wheel_residual_rate_rad_s2=self.wheel_residual_rate_rad_s2,
            training_enabled=True,
            path=self.path,
        )

    def scale_audit_rows(self) -> tuple[PhaseActionScaleAuditRow, ...]:
        rows: list[PhaseActionScaleAuditRow] = []
        for state_id in STATE_IDS:
            phase = self.policy_for(state_id)
            for index, channel_name in enumerate(FULL12_ORDER):
                absolute_lower, absolute_upper = (
                    self.base_projection.absolute_limits_full12[index]
                )
                safety_lower, safety_upper = (
                    self.base_projection.safety_limits_full12[index]
                )
                span = self.base_projection.physical_residual_scale_full12[index]
                scale = phase.scale_full12[index]
                rows.append(
                    PhaseActionScaleAuditRow(
                        state_id=state_id,
                        objective=phase.objective,
                        channel_index=index,
                        channel_name=channel_name,
                        actuator_family="servo" if index < 8 else "wheel",
                        unit="deg" if index < 8 else "rad_s",
                        enabled=bool(phase.mask_full12[index]),
                        role=phase.role_full12[index],
                        sensitivity_tier=phase.sensitivity_full12[index],
                        absolute_lower=absolute_lower,
                        absolute_upper=absolute_upper,
                        safety_lower=safety_lower,
                        safety_upper=safety_upper,
                        safety_span=span,
                        residual_scale=scale,
                        safe_span_fraction=scale / span,
                    )
                )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class TransitionActionJumpMetric:
    from_state_id: str
    to_state_id: str
    previous_projected_residual_full12: tuple[float, ...]
    carried_projected_residual_full12: tuple[float, ...]
    dropped_forbidden_residual_full12: tuple[float, ...]
    forbidden_channel_indices: tuple[int, ...]
    clipped_phase_scale_excess_residual_full12: tuple[float, ...]
    phase_scale_clipped_channel_indices: tuple[int, ...]
    residual_step_full12: tuple[float, ...]
    applied_action_jump_full12: tuple[float, ...]
    max_abs_servo_action_jump_deg: float
    max_abs_wheel_action_jump_rad_s: float
    max_abs_servo_residual_step_deg: float
    max_abs_wheel_residual_step_rad_s: float
    handoff_hold_used: bool
    hard_safety_modified: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BridgedProjection:
    projection: ProjectionResult
    transition_metric: TransitionActionJumpMetric | None


class PhaseTransitionBridge:
    """Carry allowed residuals across a phase edge without resetting to zero.

    Call ``project_tick`` at 120 Hz with the 15 Hz policy output held between
    decisions.  On the first tick in a new phase, the old residual is first
    intersected with the new phase/runtime/safety mask.  Retained non-zero
    channels are held for that one tick, then the existing ActionProjector
    slews them toward the held policy target on subsequent ticks.  Safety mask
    closures and physical stops remain immediate and authoritative.
    """

    def __init__(self, projector: ActionProjector) -> None:
        self.projector = projector
        self._state_id: str | None = None
        self._previous_projected = _ZERO12
        self._previous_applied: tuple[float, ...] | None = None

    @property
    def previous_projected_residual_full12(self) -> tuple[float, ...]:
        return self._previous_projected

    @property
    def state_id(self) -> str | None:
        return self._state_id

    def reset(
        self,
        *,
        state_id: str | None = None,
        projected_residual_full12: Sequence[float] = _ZERO12,
        applied_action_full12: Sequence[float] | None = None,
    ) -> None:
        if state_id is not None:
            self.projector.config.mask_for(state_id)
        self._state_id = state_id
        self._previous_projected = _numeric_vector(
            projected_residual_full12, "projected_residual_full12"
        )
        self._previous_applied = (
            None
            if applied_action_full12 is None
            else _numeric_vector(applied_action_full12, "applied_action_full12")
        )

    def project_tick(
        self,
        raw_residual_full12: Sequence[float],
        *,
        state_id: str,
        nominal_action_full12: Sequence[float],
        reference_action_full12: Sequence[float],
        reference_delta_full12: Sequence[float],
        runtime_action_mask_full12: Sequence[int] | None = None,
        safety: SafetyProjection | None = None,
        dt_s: float | None = None,
    ) -> BridgedProjection:
        raw = _numeric_vector(raw_residual_full12, "raw_residual_full12")
        safety_projection = safety or SafetyProjection()
        phase_mask = self.projector.config.mask_for(state_id)
        runtime_mask = (
            _ONE12
            if runtime_action_mask_full12 is None
            else _binary_mask(
                runtime_action_mask_full12, "runtime_action_mask_full12"
            )
        )
        bridge_mask = tuple(
            phase * runtime * safe
            for phase, runtime, safe in zip(
                phase_mask,
                runtime_mask,
                safety_projection.channel_mask_full12,
                strict=True,
            )
        )
        physical_stop = bool(
            safety_projection.body_collision_detected
            or safety_projection.wheel_only_climb_detected
        )
        if (
            not safety_projection.residual_enabled
            or physical_stop
            or safety_projection.override_full12 is not None
        ):
            bridge_mask = (0,) * ACTION_DIMENSION
        elif safety_projection.force_wheels_zero:
            bridge_mask = bridge_mask[:8] + (0, 0, 0, 0)

        source_previous = self._previous_projected
        mask_retained = tuple(
            value if enabled else 0.0
            for value, enabled in zip(source_previous, bridge_mask, strict=True)
        )
        physical_phase_caps = tuple(
            fraction * physical
            for fraction, physical in zip(
                self.projector.config.scale_for(state_id),
                self.projector.config.physical_residual_scale_full12,
                strict=True,
            )
        )
        carried = tuple(
            max(-cap, min(cap, value))
            for value, cap in zip(
                mask_retained, physical_phase_caps, strict=True
            )
        )
        transition = self._state_id is not None and self._state_id != state_id
        retained_nonzero = transition and any(value != 0.0 for value in carried)
        projected_raw = (
            self._raw_that_holds(carried, state_id) if retained_nonzero else raw
        )
        result = self.projector.project(
            projected_raw,
            state_id=state_id,
            nominal_action_full12=nominal_action_full12,
            reference_action_full12=reference_action_full12,
            reference_delta_full12=reference_delta_full12,
            previous_projected_residual_full12=carried,
            runtime_action_mask_full12=runtime_mask,
            safety=safety_projection,
            dt_s=dt_s,
        )

        metric: TransitionActionJumpMetric | None = None
        if transition:
            previous_applied = self._previous_applied
            current_applied = result.applied_action_full12
            applied_jump = (
                _ZERO12
                if previous_applied is None
                else tuple(
                    current - previous
                    for current, previous in zip(
                        current_applied, previous_applied, strict=True
                    )
                )
            )
            residual_step = tuple(
                current - previous
                for current, previous in zip(
                    result.phase_scale_projected_residual_full12,
                    carried,
                    strict=True,
                )
            )
            dropped = tuple(
                previous - kept
                for previous, kept in zip(
                    source_previous, mask_retained, strict=True
                )
            )
            scale_clipped = tuple(
                retained - kept
                for retained, kept in zip(
                    mask_retained, carried, strict=True
                )
            )
            metric = TransitionActionJumpMetric(
                from_state_id=str(self._state_id),
                to_state_id=state_id,
                previous_projected_residual_full12=source_previous,
                carried_projected_residual_full12=carried,
                dropped_forbidden_residual_full12=dropped,
                forbidden_channel_indices=tuple(
                    index
                    for index, (previous, enabled) in enumerate(
                        zip(source_previous, bridge_mask, strict=True)
                    )
                    if previous != 0.0 and not enabled
                ),
                clipped_phase_scale_excess_residual_full12=scale_clipped,
                phase_scale_clipped_channel_indices=tuple(
                    index
                    for index, excess in enumerate(scale_clipped)
                    if excess != 0.0
                ),
                residual_step_full12=residual_step,
                applied_action_jump_full12=applied_jump,
                max_abs_servo_action_jump_deg=max(
                    abs(value) for value in applied_jump[:8]
                ),
                max_abs_wheel_action_jump_rad_s=max(
                    abs(value) for value in applied_jump[8:]
                ),
                max_abs_servo_residual_step_deg=max(
                    abs(value) for value in residual_step[:8]
                ),
                max_abs_wheel_residual_step_rad_s=max(
                    abs(value) for value in residual_step[8:]
                ),
                handoff_hold_used=retained_nonzero,
                hard_safety_modified=result.hard_safety_modified,
            )

        if (
            not safety_projection.residual_enabled
            or physical_stop
            or safety_projection.override_full12 is not None
        ):
            stored = _ZERO12
        else:
            stored = tuple(
                value if enabled else 0.0
                for value, enabled in zip(
                    result.safe_projected_residual_full12,
                    bridge_mask,
                    strict=True,
                )
            )
        self._state_id = state_id
        self._previous_projected = stored
        self._previous_applied = result.applied_action_full12
        return BridgedProjection(projection=result, transition_metric=metric)

    def _raw_that_holds(
        self, residual_full12: tuple[float, ...], state_id: str
    ) -> tuple[float, ...]:
        scales = tuple(
            fraction * physical
            for fraction, physical in zip(
                self.projector.config.scale_for(state_id),
                self.projector.config.physical_residual_scale_full12,
                strict=True,
            )
        )
        raw: list[float] = []
        for residual, scale in zip(residual_full12, scales, strict=True):
            if residual == 0.0 or scale == 0.0:
                raw.append(0.0)
                continue
            ratio = max(-1.0 + 1.0e-12, min(1.0 - 1.0e-12, residual / scale))
            raw.append(math.atanh(ratio))
        return tuple(raw)


def load_phase_action_masks_v2(
    path: Path | str = DEFAULT_PHASE_ACTION_CONFIG_V2,
) -> PhaseActionMasksV2:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PHASE_ACTION_SCHEMA_V2:
        raise PhaseActionV2Error("unexpected phase-action v2 schema")
    if not bool(payload.get("training_enabled")):
        raise PhaseActionV2Error("phase-action v2 must be training enabled")
    if int(payload.get("nominal_action_dimension", -1)) != ACTION_DIMENSION:
        raise PhaseActionV2Error("nominal action dimension must remain 12")
    if int(payload.get("residual_action_dimension", -1)) != ACTION_DIMENSION:
        raise PhaseActionV2Error("residual action dimension must remain 12")
    if tuple(payload.get("full12_order", ())) != FULL12_ORDER:
        raise PhaseActionV2Error("Full12 action order differs from the canonical ABI")
    if payload.get("bounded_transform") != "tanh":
        raise PhaseActionV2Error("the bounded transform must be tanh")
    physics_hz = float(payload.get("physics_hz", float("nan")))
    decision_hz = float(payload.get("decision_hz", float("nan")))
    if physics_hz != 120.0 or decision_hz != 15.0:
        raise PhaseActionV2Error("phase-action v2 must remain at 120/15 Hz")

    base_path = selected.parent / str(payload.get("base_projection_config", ""))
    base = load_action_projection_config(base_path)
    derivation = payload.get("scale_derivation", {})
    if not isinstance(derivation, Mapping):
        raise PhaseActionV2Error("scale_derivation must be a mapping")
    if derivation.get("representation") != "physical_residual_cap_in_canonical_full12_units":
        raise PhaseActionV2Error("v2 scales must be declared in physical Full12 units")
    if bool(derivation.get("recording_envelope_used_as_hard_cap", True)):
        raise PhaseActionV2Error("Recording may not be a v2 residual hard cap")
    if bool(derivation.get("recording_conformance_fraction_used_in_scale", True)):
        raise PhaseActionV2Error("Recording 30% may not determine v2 residual scales")
    if not bool(derivation.get("live_small_perturbation_smoke_required_before_training_rollout")):
        raise PhaseActionV2Error("live small-perturbation smoke must gate training rollout")
    max_fraction = float(derivation["maximum_initial_safe_span_fraction"])
    max_servo = float(derivation["maximum_initial_servo_cap_deg"])
    max_wheel = float(derivation["maximum_initial_wheel_cap_rad_s"])
    if not (0.0 < max_fraction <= 0.05 and max_servo > 0.0 and max_wheel > 0.0):
        raise PhaseActionV2Error("initial v2 scale ceilings must be small and positive")

    rates = payload.get("residual_rate_limits", {})
    if not isinstance(rates, Mapping) or rates.get("applied_by") != "ActionProjector":
        raise PhaseActionV2Error("v2 slew must be applied by ActionProjector")
    servo_rate = float(rates.get("servo_deg_s", float("nan")))
    wheel_rate = float(rates.get("wheel_rad_s2", float("nan")))
    if (
        not math.isfinite(servo_rate)
        or not math.isfinite(wheel_rate)
        or servo_rate <= 0.0
        or wheel_rate <= 0.0
        or servo_rate > base.servo_residual_rate_deg_s
        or wheel_rate > base.wheel_residual_rate_rad_s2
    ):
        raise PhaseActionV2Error(
            "v2 residual slew rates must be positive and no faster than v1 safety rates"
        )

    bridge = payload.get("transition_bridge", {})
    required_bridge_flags = (
        "project_previous_into_new_effective_mask",
        "retain_allowed_residual_for_first_new_phase_tick",
        "slew_to_zero_or_new_policy_target_after_handoff",
        "forbidden_channels_zero_before_projection",
        "safety_overrides_remain_authoritative",
    )
    if (
        not isinstance(bridge, Mapping)
        or float(bridge.get("physics_hz", 0.0)) != physics_hz
        or any(not bool(bridge.get(flag)) for flag in required_bridge_flags)
        or bool(bridge.get("reset_previous_residual_on_phase_change", True))
    ):
        raise PhaseActionV2Error("transition bridge contract is incomplete or unsafe")

    raw_phases = payload.get("phases", {})
    if not isinstance(raw_phases, Mapping) or tuple(raw_phases) != STATE_IDS:
        raise PhaseActionV2Error("phase-action v2 must contain ordered P01-P13")
    phases: dict[str, PhaseChannelPolicyV2] = {}
    for state_id in STATE_IDS:
        raw_phase = raw_phases[state_id]
        if not isinstance(raw_phase, Mapping):
            raise PhaseActionV2Error(f"phases.{state_id} must be a mapping")
        mask = _binary_mask(raw_phase.get("mask_full12", ()), f"phases.{state_id}.mask_full12")
        scales = _numeric_vector(
            raw_phase.get("scale_full12", ()), f"phases.{state_id}.scale_full12"
        )
        roles = _text_vector(
            raw_phase.get("role_full12", ()), f"phases.{state_id}.role_full12"
        )
        sensitivity = _text_vector(
            raw_phase.get("sensitivity_full12", ()),
            f"phases.{state_id}.sensitivity_full12",
        )
        if not any(mask):
            raise PhaseActionV2Error(f"{state_id} must enable at least one action channel")
        if mask[:8] != (1,) * 8:
            raise PhaseActionV2Error(f"{state_id} must expose all eight servos at small scale")
        if any(value < 0.0 for value in scales):
            raise PhaseActionV2Error(f"{state_id} contains a negative residual scale")
        if any(bool(enabled) != (scale > 0.0) for enabled, scale in zip(mask, scales, strict=True)):
            raise PhaseActionV2Error(
                f"{state_id} masks and physical scales must enable the same channels"
            )
        if any(tier not in _SENSITIVITY_TIERS for tier in sensitivity):
            raise PhaseActionV2Error(f"{state_id} contains an unknown sensitivity tier")
        if any(
            (tier == "blocked") != (not enabled)
            for tier, enabled in zip(sensitivity, mask, strict=True)
        ):
            raise PhaseActionV2Error(
                f"{state_id} blocked sensitivity tiers must match disabled channels"
            )
        for index, (scale, span) in enumerate(
            zip(scales, base.physical_residual_scale_full12, strict=True)
        ):
            family_cap = max_servo if index < 8 else max_wheel
            if scale > family_cap + 1.0e-12 or scale / span > max_fraction + 1.0e-12:
                raise PhaseActionV2Error(
                    f"{state_id} channel {index} exceeds the conservative initial scale ceiling"
                )
        phases[state_id] = PhaseChannelPolicyV2(
            state_id=state_id,
            objective=str(raw_phase.get("objective", "")),
            mask_full12=mask,
            scale_full12=scales,
            role_full12=roles,
            sensitivity_full12=sensitivity,
        )

    if phases["P02"].mask_full12 != _ONE12:
        raise PhaseActionV2Error("P02 must expose support legs, FR clearance, and all wheels")
    p02 = phases["P02"].scale_full12
    if max(p02[2:4]) >= min(p02[0:2] + p02[4:8]):
        raise PhaseActionV2Error("P02 FR clearance scales must be smaller than support scales")
    if phases["P03"].mask_full12 != _ONE12:
        raise PhaseActionV2Error("P03 must expose all twelve channels")
    if phases["P08"].mask_full12[:8] != (1,) * 8 or not (
        phases["P08"].mask_full12[8] and phases["P08"].mask_full12[11]
    ):
        raise PhaseActionV2Error("P08 must expose all servos plus FL/RR transfer wheels")
    if phases["P12"].mask_full12 != _ONE12:
        raise PhaseActionV2Error("P12 must expose swing/support servos and all needed wheels")
    if phases["P13"].mask_full12 != _ONE12:
        raise PhaseActionV2Error("P13 must expose all twelve channels")

    return PhaseActionMasksV2(
        schema=PHASE_ACTION_SCHEMA_V2,
        action_schema_name=str(payload.get("action_schema_name", "")),
        action_schema_version=int(payload.get("action_schema_version", -1)),
        training_enabled=True,
        physics_hz=physics_hz,
        decision_hz=decision_hz,
        phases=phases,
        base_projection=base,
        servo_residual_rate_deg_s=servo_rate,
        wheel_residual_rate_rad_s2=wheel_rate,
        maximum_initial_safe_span_fraction=max_fraction,
        maximum_initial_servo_cap_deg=max_servo,
        maximum_initial_wheel_cap_rad_s=max_wheel,
        sensitivity_method=str(derivation.get("sensitivity_method", "")),
        live_small_perturbation_smoke_required=True,
        path=selected,
    )


def build_action_projector_v2(
    config: PhaseActionMasksV2 | None = None,
    *,
    policy: Any | None = None,
) -> ActionProjector:
    """Return the existing ActionProjector configured with the v2 contract."""

    selected = config or load_phase_action_masks_v2()
    return ActionProjector(config=selected.action_projection_config(), policy=policy)
