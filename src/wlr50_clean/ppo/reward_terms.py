"""Config-driven reward component interface; no optimizer or trainer lives here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml


REWARD_SCHEMA = "wlr50_clean.ppo_reward.v1"
DEFAULT_REWARD_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_reward.yaml"
)
REWARD_TERMS = (
    "task_success",
    "forward_progress",
    "phase_progress",
    "active_leg_clearance",
    "body_collision_penalty",
    "wheel_only_climb_penalty",
    "fall_penalty",
    "residual_magnitude_penalty",
    "action_rate_penalty",
    "joint_limit_penalty",
    "body_angular_velocity_penalty",
    "pitch_roll_penalty",
    "support_diagnostic",
)


class RewardConfigurationError(ValueError):
    pass


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RewardConfigurationError(f"{label} must be finite")
    return result


def _rms(values: Sequence[float], label: str) -> float:
    result = tuple(_finite(value, label) for value in values)
    if len(result) != 12:
        raise RewardConfigurationError(f"{label} must contain 12 values")
    return math.sqrt(sum(value * value for value in result) / len(result))


@dataclass(frozen=True, slots=True)
class RewardTermSpec:
    name: str
    weight: float
    scale: float
    clip_min: float
    clip_max: float

    def normalize(self, raw: float) -> float:
        value = _finite(raw, self.name) / self.scale
        return max(self.clip_min, min(self.clip_max, value))


@dataclass(frozen=True, slots=True)
class RewardConfig:
    schema: str
    training_enabled: bool
    aggregation: str
    terms: Mapping[str, RewardTermSpec]
    path: Path


@dataclass(frozen=True, slots=True)
class RewardSignals:
    task_success: bool = False
    forward_progress_delta_m: float = 0.0
    phase_progress_delta: float = 0.0
    active_leg_clearance_m: float = 0.0
    body_collision: bool = False
    wheel_only_climb: bool = False
    fall: bool = False
    joint_limit_violation: bool = False
    body_angular_speed_rad_s: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    support_margin_m: float | None = None
    support_valid: bool = False


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    raw_components: Mapping[str, float]
    normalized_components: Mapping[str, float]
    weighted_components: Mapping[str, float]


def load_reward_config(path: Path | str = DEFAULT_REWARD_PATH) -> RewardConfig:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != REWARD_SCHEMA:
        raise RewardConfigurationError("unexpected PPO reward schema")
    if bool(payload.get("training_enabled")):
        raise RewardConfigurationError("PPO training must remain disabled")
    raw_terms = payload.get("terms")
    if not isinstance(raw_terms, Mapping) or tuple(raw_terms) != REWARD_TERMS:
        raise RewardConfigurationError("reward terms are missing or reordered")
    terms: dict[str, RewardTermSpec] = {}
    for name, raw in raw_terms.items():
        clip = tuple(float(item) for item in raw["clip"])
        scale = float(raw["scale"])
        if len(clip) != 2 or not clip[0] < clip[1] or scale <= 0.0:
            raise RewardConfigurationError(f"invalid reward term {name}")
        terms[str(name)] = RewardTermSpec(
            name=str(name),
            weight=_finite(raw["weight"], f"{name}.weight"),
            scale=_finite(scale, f"{name}.scale"),
            clip_min=clip[0],
            clip_max=clip[1],
        )
    return RewardConfig(
        schema=REWARD_SCHEMA,
        training_enabled=False,
        aggregation=str(payload["aggregation"]),
        terms=terms,
        path=selected,
    )


class RewardCalculator:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or load_reward_config()

    def evaluate(
        self,
        signals: RewardSignals,
        *,
        residual_fraction_full12: Sequence[float] = (0.0,) * 12,
        residual_rate_fraction_full12: Sequence[float] = (0.0,) * 12,
    ) -> RewardBreakdown:
        """Return every component even when its current value is zero."""

        raw = {
            "task_success": float(signals.task_success),
            "forward_progress": _finite(
                signals.forward_progress_delta_m, "forward_progress_delta_m"
            ),
            "phase_progress": _finite(
                signals.phase_progress_delta, "phase_progress_delta"
            ),
            "active_leg_clearance": max(
                0.0,
                _finite(signals.active_leg_clearance_m, "active_leg_clearance_m"),
            ),
            "body_collision_penalty": float(signals.body_collision),
            "wheel_only_climb_penalty": float(signals.wheel_only_climb),
            "fall_penalty": float(signals.fall),
            "residual_magnitude_penalty": _rms(
                residual_fraction_full12, "residual_fraction_full12"
            ),
            "action_rate_penalty": _rms(
                residual_rate_fraction_full12, "residual_rate_fraction_full12"
            ),
            "joint_limit_penalty": float(signals.joint_limit_violation),
            "body_angular_velocity_penalty": abs(
                _finite(
                    signals.body_angular_speed_rad_s,
                    "body_angular_speed_rad_s",
                )
            ),
            "pitch_roll_penalty": abs(_finite(signals.pitch_rad, "pitch_rad"))
            + abs(_finite(signals.roll_rad, "roll_rad")),
            "support_diagnostic": (
                _finite(signals.support_margin_m, "support_margin_m")
                if signals.support_valid and signals.support_margin_m is not None
                else 0.0
            ),
        }
        normalized = {
            name: self.config.terms[name].normalize(raw[name])
            for name in REWARD_TERMS
        }
        weighted = {
            name: normalized[name] * self.config.terms[name].weight
            for name in REWARD_TERMS
        }
        total = sum(weighted.values())
        if not math.isfinite(total):
            raise RewardConfigurationError("reward total is non-finite")
        return RewardBreakdown(
            total=total,
            raw_components=raw,
            normalized_components=normalized,
            weighted_components=weighted,
        )
