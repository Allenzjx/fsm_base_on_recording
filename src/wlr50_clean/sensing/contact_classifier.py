"""Exact-pair contact classification with a three-sample force history."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .observation import (
    BodyContactObservation,
    CollisionRole,
    ContactClass,
    PairContactObservation,
    Vec3,
)


BASE_BODY = "base_link"
LEG_BODIES = (
    "front_left_upper",
    "front_right_upper",
    "rear_left_upper",
    "rear_right_upper",
    "front_left_bot",
    "front_right_bot",
    "rear_left_bot",
    "rear_right_bot",
)
WHEEL_BODIES = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)
SENSED_BODIES = (BASE_BODY, *LEG_BODIES, *WHEEL_BODIES)

OBSTACLE_PAIR = "obstacle"
GROUND_PAIR = "ground"

DEFAULT_ROLE_MAP: dict[str, CollisionRole] = {
    BASE_BODY: CollisionRole.BODY,
    **{name: CollisionRole.LEG for name in LEG_BODIES},
    **{name: CollisionRole.WHEEL for name in WHEEL_BODIES},
}


@dataclass(frozen=True, slots=True)
class RawPairContact:
    """Raw output for one configured ContactSensor filter pair."""

    sensor_body: str
    pair_kind: str
    other_body: str
    force_w_n: Vec3
    friction_force_w_n: Vec3 = (0.0, 0.0, 0.0)
    contact_point_w_m: Vec3 | None = None
    history_force_w_n: tuple[Vec3, ...] = ()
    source: str = "unavailable"
    pair_verified: bool = False


class ContactContractError(ValueError):
    """Raised when exact bodies or pairs are absent or ambiguous."""


class ContactClassifier:
    """Convert exact force pairs into debounced, role-aware body contacts."""

    def __init__(
        self,
        *,
        role_map: Mapping[str, CollisionRole | str] | None = None,
        force_on_n: float = 0.25,
        force_off_n: float = 0.12,
        history_length: int = 3,
    ) -> None:
        if force_on_n <= 0.0 or force_off_n < 0.0 or force_off_n >= force_on_n:
            raise ContactContractError("contact thresholds require 0 <= off < on")
        if history_length < 3:
            raise ContactContractError("contact history must contain at least three physics samples")
        raw_roles = dict(DEFAULT_ROLE_MAP if role_map is None else role_map)
        if set(raw_roles) != set(SENSED_BODIES):
            raise ContactContractError("role map must contain exactly the 13 locked rigid bodies")
        self.role_map = {name: CollisionRole(value) for name, value in raw_roles.items()}
        if self.role_map[BASE_BODY] is not CollisionRole.BODY:
            raise ContactContractError("base_link must be the sole BODY role")
        if [name for name, role in self.role_map.items() if role is CollisionRole.BODY] != [BASE_BODY]:
            raise ContactContractError("BODY collision role is reserved for base_link")
        self.force_on_n = float(force_on_n)
        self.force_off_n = float(force_off_n)
        self.history_length = int(history_length)
        self._states: dict[tuple[str, str], bool] = {}
        self._history: dict[tuple[str, str], deque[bool]] = {}

    def classify(self, samples: Iterable[RawPairContact]) -> dict[str, BodyContactObservation]:
        indexed: dict[tuple[str, str], RawPairContact] = {}
        for sample in samples:
            key = (sample.sensor_body, sample.pair_kind)
            if sample.sensor_body not in self.role_map:
                raise ContactContractError(f"unrecognized sensor body: {sample.sensor_body}")
            if sample.pair_kind not in (GROUND_PAIR, OBSTACLE_PAIR):
                raise ContactContractError(f"unrecognized pair kind: {sample.pair_kind}")
            if key in indexed:
                raise ContactContractError(f"duplicate exact contact pair: {key}")
            indexed[key] = sample

        result: dict[str, BodyContactObservation] = {}
        for body_name in SENSED_BODIES:
            ground = self._classify_pair(indexed.get((body_name, GROUND_PAIR)), body_name, GROUND_PAIR)
            obstacle = self._classify_pair(
                indexed.get((body_name, OBSTACLE_PAIR)), body_name, OBSTACLE_PAIR
            )
            if not ground.pair_verified or not obstacle.pair_verified:
                contact_class = ContactClass.UNVERIFIED
            elif ground.active and obstacle.active:
                contact_class = ContactClass.GROUND_AND_OBSTACLE
            elif obstacle.active:
                contact_class = ContactClass.OBSTACLE
            elif ground.active:
                contact_class = ContactClass.GROUND
            else:
                contact_class = ContactClass.AIR
            result[body_name] = BodyContactObservation(
                body_name=body_name,
                role=self.role_map[body_name],
                contact_class=contact_class,
                ground=ground,
                obstacle=obstacle,
            )
        return result

    def _classify_pair(
        self,
        sample: RawPairContact | None,
        body_name: str,
        pair_kind: str,
    ) -> PairContactObservation:
        key = (body_name, pair_kind)
        if sample is None:
            self._states[key] = False
            history = self._history.setdefault(key, deque(maxlen=self.history_length))
            history.appendleft(False)
            return _empty_pair(body_name, pair_kind, tuple(history), self.history_length)

        force_norm = _norm(sample.force_w_n)
        was_active = self._states.get(key, False)
        threshold = self.force_off_n if was_active else self.force_on_n
        active = bool(sample.pair_verified and force_norm >= threshold)
        self._states[key] = active

        if sample.history_force_w_n:
            raw_history = tuple(
                bool(sample.pair_verified and _norm(force) >= self.force_off_n)
                for force in sample.history_force_w_n[: self.history_length]
            )
            history_tuple = _pad_history(raw_history, self.history_length)
            local_history = self._history.setdefault(key, deque(maxlen=self.history_length))
            local_history.clear()
            local_history.extend(history_tuple)
        else:
            local_history = self._history.setdefault(key, deque(maxlen=self.history_length))
            local_history.appendleft(active)
            history_tuple = _pad_history(tuple(local_history), self.history_length)

        # A newly crossed on-threshold is authoritative even if a backend's
        # history buffer has not rolled the current sample in yet.
        if active and not history_tuple[0]:
            history_tuple = (True, *history_tuple[1:])
        consecutive = 0
        for item in history_tuple:
            if not item:
                break
            consecutive += 1
        return PairContactObservation(
            sensor_body=body_name,
            other_body=sample.other_body,
            active=active,
            force_w_n=_vec3(sample.force_w_n),
            normal_force_n=force_norm,
            tangential_force_n=_norm(sample.friction_force_w_n),
            contact_point_w_m=(
                None if sample.contact_point_w_m is None else _vec3(sample.contact_point_w_m)
            ),
            force_history_w_n=tuple(_vec3(force) for force in sample.history_force_w_n[: self.history_length]),
            active_history=history_tuple,
            consecutive_active_ticks=consecutive,
            source=sample.source,
            pair_verified=bool(sample.pair_verified),
        )


def _empty_pair(
    body_name: str, pair_kind: str, history: Sequence[bool], history_length: int
) -> PairContactObservation:
    return PairContactObservation(
        sensor_body=body_name,
        other_body=pair_kind,
        active=False,
        force_w_n=(0.0, 0.0, 0.0),
        normal_force_n=0.0,
        tangential_force_n=0.0,
        contact_point_w_m=None,
        force_history_w_n=(),
        active_history=_pad_history(tuple(history), history_length),
        consecutive_active_ticks=0,
        source="unavailable",
        pair_verified=False,
    )


def _norm(values: Sequence[float]) -> float:
    if len(values) != 3:
        return 0.0
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        return 0.0
    return math.sqrt(sum(value * value for value in converted))


def _vec3(values: Sequence[float]) -> Vec3:
    converted = tuple(float(value) for value in values)
    if len(converted) != 3 or not all(math.isfinite(value) for value in converted):
        raise ContactContractError("contact vector must contain three finite values")
    return converted  # type: ignore[return-value]


def _pad_history(values: tuple[bool, ...], size: int) -> tuple[bool, ...]:
    clipped = tuple(bool(value) for value in values[:size])
    return clipped + (False,) * (size - len(clipped))
