"""Stable residual-PPO boundary; residuals are disabled for this delivery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
NOMINAL_ACTION_DIM = 12
RESIDUAL_ACTION_DIM = 12
OBSERVATION_DIM = 85


def _finite(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{label}: expected {size} values, received {len(result)}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label}: non-finite value")
    return result


@dataclass(frozen=True, slots=True)
class PPOObservationParts:
    joint_position_error8: tuple[float, ...]
    joint_velocity8: tuple[float, ...]
    wheel_velocity4: tuple[float, ...]
    wheel_contact_code4: tuple[float, ...]
    leg_history12: tuple[float, ...]
    body_orientation_wxyz4: tuple[float, ...]
    body_angular_velocity3: tuple[float, ...]
    obstacle_relative_geometry9: tuple[float, ...]
    full_body_com3: tuple[float, ...]
    support_diagnostics4: tuple[float, ...]

    def __post_init__(self) -> None:
        fields = (
            ("joint_position_error8", 8),
            ("joint_velocity8", 8),
            ("wheel_velocity4", 4),
            ("wheel_contact_code4", 4),
            ("leg_history12", 12),
            ("body_orientation_wxyz4", 4),
            ("body_angular_velocity3", 3),
            ("obstacle_relative_geometry9", 9),
            ("full_body_com3", 3),
            ("support_diagnostics4", 4),
        )
        for name, size in fields:
            object.__setattr__(self, name, _finite(getattr(self, name), size, name))


@dataclass(frozen=True, slots=True)
class PPOFrame:
    state_id: str
    macro_phase: int
    phase_progress: float
    nominal_action_full12: tuple[float, ...]
    action_mask_full12: tuple[int, ...]
    observation_vector: tuple[float, ...]


class ResidualInterface:
    """Compose nominal + masked residual actions with a frozen observation ABI."""

    def __init__(self, *, residual_enabled: bool = False):
        self.residual_enabled = bool(residual_enabled)

    def frame(
        self,
        *,
        state_id: str,
        macro_phase: int,
        phase_progress: float,
        nominal_action_full12: Sequence[float],
        action_mask_full12: Sequence[int],
        observation: PPOObservationParts,
        previous_action_full12: Sequence[float],
    ) -> PPOFrame:
        if state_id not in STATE_IDS:
            raise ValueError(f"unknown state_id: {state_id}")
        if int(macro_phase) != STATE_IDS.index(state_id) + 1:
            raise ValueError("macro_phase does not match state_id")
        progress = min(max(float(phase_progress), 0.0), 1.0)
        nominal = _finite(
            nominal_action_full12, NOMINAL_ACTION_DIM, "nominal_action_full12"
        )
        mask = tuple(int(value) for value in action_mask_full12)
        if len(mask) != RESIDUAL_ACTION_DIM or any(value not in (0, 1) for value in mask):
            raise ValueError("action_mask_full12 must contain twelve binary values")
        previous = _finite(
            previous_action_full12, NOMINAL_ACTION_DIM, "previous_action_full12"
        )
        one_hot = tuple(1.0 if name == state_id else 0.0 for name in STATE_IDS)
        vector = (
            one_hot
            + (progress,)
            + observation.joint_position_error8
            + observation.joint_velocity8
            + observation.wheel_velocity4
            + observation.wheel_contact_code4
            + observation.leg_history12
            + observation.body_orientation_wxyz4
            + observation.body_angular_velocity3
            + observation.obstacle_relative_geometry9
            + observation.full_body_com3
            + observation.support_diagnostics4
            + previous
        )
        if len(vector) != OBSERVATION_DIM:
            raise AssertionError("internal PPO observation layout changed")
        return PPOFrame(
            state_id=state_id,
            macro_phase=int(macro_phase),
            phase_progress=progress,
            nominal_action_full12=nominal,
            action_mask_full12=mask,
            observation_vector=vector,
        )

    def compose_action(
        self, frame: PPOFrame, residual_full12: Sequence[float] | None = None
    ) -> tuple[float, ...]:
        residual_source = (
            (0.0,) * RESIDUAL_ACTION_DIM
            if residual_full12 is None
            else residual_full12
        )
        residual = _finite(
            residual_source,
            RESIDUAL_ACTION_DIM,
            "residual_full12",
        )
        if not self.residual_enabled and any(abs(value) > 0.0 for value in residual):
            raise RuntimeError("PPO residuals are disabled; this run requires zero residual")
        return tuple(
            nominal + mask * delta
            for nominal, mask, delta in zip(
                frame.nominal_action_full12,
                frame.action_mask_full12,
                residual,
                strict=True,
            )
        )


def zero_residual_is_nominal(frame: PPOFrame) -> bool:
    return ResidualInterface(residual_enabled=False).compose_action(frame) == tuple(
        frame.nominal_action_full12
    )
