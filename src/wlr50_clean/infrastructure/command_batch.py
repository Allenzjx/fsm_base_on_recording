"""Canonical, Isaac-free actuator command batches for the clean runtime.

The names, limits, and signs are narrowly derived from the mature Recording
``command_model.py`` (SHA-256
``70f1e4183fc711f1dcded5d44a45c661960a383634b252c944ca21705c86e905``).
This module deliberately contains no parser, replay, recording, or UI code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


SERVO_ORDER: tuple[str, ...] = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)

WHEEL_ORDER: tuple[str, ...] = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)

FULL12_ORDER: tuple[str, ...] = SERVO_ORDER + WHEEL_ORDER

KNEE_NAMES: frozenset[str] = frozenset(
    {
        "front_left_knee",
        "front_right_knee",
        "rear_left_knee",
        "rear_right_knee",
    }
)

SERVO_COMMAND_SIGN: Mapping[str, float] = MappingProxyType({
    "front_left_hip": 1.0,
    "front_left_knee": 1.0,
    "front_right_hip": 1.0,
    "front_right_knee": 1.0,
    "rear_left_hip": -1.0,
    "rear_left_knee": -1.0,
    "rear_right_hip": -1.0,
    "rear_right_knee": -1.0,
})

WHEEL_FORWARD_SIGN: Mapping[str, float] = MappingProxyType({
    "front_left_ankle": -1.0,
    "front_right_ankle": 1.0,
    "rear_left_ankle": -1.0,
    "rear_right_ankle": 1.0,
})

HIP_LIMIT_DEG: tuple[float, float] = (-135.0, 135.0)
KNEE_LIMIT_DEG: tuple[float, float] = (-60.0, 210.0)
WHEEL_VELOCITY_LIMIT_RAD_S = 2.0943951023931953
WHEEL_REFERENCE_VELOCITY_RAD_S = 0.5235987755982988
SERVO_REFERENCE_VELOCITY_DEG_S = 150.0
PHYSICS_DT_S = 1.0 / 120.0


class CommandBatchError(ValueError):
    """Raised before any actuator target is staged or written."""


@dataclass(frozen=True, slots=True)
class JointIndexMap:
    """Exact live articulation indices in canonical command order."""

    servo_ids: tuple[int, ...]
    wheel_ids: tuple[int, ...]
    live_joint_names: tuple[str, ...]

    @property
    def full12_ids(self) -> tuple[int, ...]:
        return self.servo_ids + self.wheel_ids


@dataclass(frozen=True, slots=True)
class Full12Command:
    """One complete logical command for all 8 servos and all 4 wheels.

    Servo values are recording-space degrees relative to the standing pose.
    Wheel values are canonical forward-positive angular velocities in rad/s.
    The constructor checks shape and finiteness. ``clamped`` applies the mature
    command-space limits without mutating the requested command.
    """

    servo_deg: tuple[float, ...]
    wheel_rad_s: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "servo_deg", _finite_tuple(self.servo_deg, len(SERVO_ORDER), "servo_deg"))
        object.__setattr__(self, "wheel_rad_s", _finite_tuple(self.wheel_rad_s, len(WHEEL_ORDER), "wheel_rad_s"))

    @classmethod
    def zeros(cls) -> "Full12Command":
        return cls((0.0,) * len(SERVO_ORDER), (0.0,) * len(WHEEL_ORDER))

    @classmethod
    def from_full12(cls, values: Sequence[float]) -> "Full12Command":
        ordered = _finite_tuple(values, len(FULL12_ORDER), "full12")
        return cls(ordered[: len(SERVO_ORDER)], ordered[len(SERVO_ORDER) :])

    @classmethod
    def from_mappings(
        cls,
        servo_deg: Mapping[str, float],
        wheel_rad_s: Mapping[str, float],
    ) -> "Full12Command":
        _require_exact_keys(servo_deg, SERVO_ORDER, "servo_deg")
        _require_exact_keys(wheel_rad_s, WHEEL_ORDER, "wheel_rad_s")
        return cls(
            tuple(float(servo_deg[name]) for name in SERVO_ORDER),
            tuple(float(wheel_rad_s[name]) for name in WHEEL_ORDER),
        )

    def to_full12(self) -> tuple[float, ...]:
        return self.servo_deg + self.wheel_rad_s

    def servo_by_name(self) -> dict[str, float]:
        return dict(zip(SERVO_ORDER, self.servo_deg, strict=True))

    def wheel_by_name(self) -> dict[str, float]:
        return dict(zip(WHEEL_ORDER, self.wheel_rad_s, strict=True))

    def clamped(self) -> "Full12Command":
        servos = tuple(
            _clamp(value, *servo_limits_deg(name))
            for name, value in zip(SERVO_ORDER, self.servo_deg, strict=True)
        )
        wheels = tuple(
            _clamp(value, -WHEEL_VELOCITY_LIMIT_RAD_S, WHEEL_VELOCITY_LIMIT_RAD_S)
            for value in self.wheel_rad_s
        )
        return Full12Command(servos, wheels)

    def was_clamped(self) -> bool:
        return self.clamped() != self


# ``CommandBatch`` is the concise runtime-facing name; the explicit name is
# retained for conformance reports and type annotations.
CommandBatch = Full12Command


@dataclass(frozen=True, slots=True)
class PhysicalCommandBatch:
    """Targets in the units/sign convention consumed by the articulation."""

    servo_target_rad: tuple[float, ...]
    wheel_target_rad_s: tuple[float, ...]
    applied_logical: Full12Command


def resolve_joint_indices(live_joint_names: Sequence[str]) -> JointIndexMap:
    """Resolve all required joints by exact name, rejecting ambiguity."""

    names = tuple(str(name) for name in live_joint_names)
    if not names:
        raise CommandBatchError("robot.joint_names is empty")
    positions: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        positions.setdefault(name, []).append(index)

    missing = [name for name in FULL12_ORDER if name not in positions]
    ambiguous = {name: positions[name] for name in FULL12_ORDER if len(positions.get(name, ())) > 1}
    if missing or ambiguous:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if ambiguous:
            details.append(f"ambiguous={ambiguous}")
        raise CommandBatchError("required live joint-name resolution failed: " + "; ".join(details))

    return JointIndexMap(
        servo_ids=tuple(positions[name][0] for name in SERVO_ORDER),
        wheel_ids=tuple(positions[name][0] for name in WHEEL_ORDER),
        live_joint_names=names,
    )


def servo_limits_deg(joint_name: str) -> tuple[float, float]:
    if joint_name not in SERVO_ORDER:
        raise CommandBatchError(f"unknown servo joint: {joint_name!r}")
    return KNEE_LIMIT_DEG if joint_name in KNEE_NAMES else HIP_LIMIT_DEG


def build_physical_batch(
    requested: Full12Command,
    standing_pose_deg: Mapping[str, float],
) -> PhysicalCommandBatch:
    """Clamp and convert a complete logical batch to physical targets."""

    _require_exact_keys(standing_pose_deg, SERVO_ORDER, "standing_pose_deg")
    standing = {name: _finite_float(standing_pose_deg[name], f"standing_pose_deg[{name!r}]") for name in SERVO_ORDER}
    applied = requested.clamped()
    servo_targets = tuple(
        math.radians(standing[name] + SERVO_COMMAND_SIGN[name] * command_deg)
        for name, command_deg in zip(SERVO_ORDER, applied.servo_deg, strict=True)
    )
    wheel_targets = tuple(
        WHEEL_FORWARD_SIGN[name] * canonical_rad_s
        for name, canonical_rad_s in zip(WHEEL_ORDER, applied.wheel_rad_s, strict=True)
    )
    return PhysicalCommandBatch(servo_targets, wheel_targets, applied)


def logical_readback_from_physical(
    servo_position_rad: Sequence[float],
    wheel_velocity_rad_s: Sequence[float],
    standing_pose_deg: Mapping[str, float],
) -> tuple[float, ...]:
    """Convert live physical q/qd values into canonical Full12 semantics."""

    servo_rad = _finite_tuple(servo_position_rad, len(SERVO_ORDER), "servo_position_rad")
    wheel_physical = _finite_tuple(wheel_velocity_rad_s, len(WHEEL_ORDER), "wheel_velocity_rad_s")
    _require_exact_keys(standing_pose_deg, SERVO_ORDER, "standing_pose_deg")
    servo_logical = tuple(
        SERVO_COMMAND_SIGN[name] * (math.degrees(value) - float(standing_pose_deg[name]))
        for name, value in zip(SERVO_ORDER, servo_rad, strict=True)
    )
    wheel_logical = tuple(
        value / WHEEL_FORWARD_SIGN[name]
        for name, value in zip(WHEEL_ORDER, wheel_physical, strict=True)
    )
    return servo_logical + wheel_logical


def _finite_tuple(values: Sequence[float], expected: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise CommandBatchError(f"{label} must be a numeric sequence") from exc
    if len(result) != expected:
        raise CommandBatchError(f"{label} must contain exactly {expected} values; received {len(result)}")
    invalid = [index for index, value in enumerate(result) if not math.isfinite(value)]
    if invalid:
        raise CommandBatchError(f"{label} contains non-finite values at indices {invalid}")
    return result


def _finite_float(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CommandBatchError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise CommandBatchError(f"{label} must be finite")
    return result


def _require_exact_keys(values: Mapping[str, float], expected: Sequence[str], label: str) -> None:
    actual = set(values)
    required = set(expected)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise CommandBatchError(f"{label} keys must match the locked order; missing={missing}, extra={extra}")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))
