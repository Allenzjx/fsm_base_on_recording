"""Strict base-link obstacle collision detector.

Leg and wheel contacts are valid traversal mechanics.  A task-failing BODY
collision can therefore originate only from the exact base-link/obstacle
sensor pair, and requires either temporal persistence or corroborating live
collider penetration.
"""

from __future__ import annotations

import math
from typing import Mapping

from .contact_classifier import BASE_BODY
from .observation import BodyCollisionStatus, BodyContactObservation, CollisionRole


class BodyCollisionDetector:
    def __init__(
        self,
        *,
        obstacle_prim_path: str = "/World/Obstacle",
        persistence_ticks: int = 2,
        penetration_threshold_m: float = 0.001,
    ) -> None:
        if persistence_ticks < 2:
            raise ValueError("BODY contact persistence must be at least two physics ticks")
        if penetration_threshold_m < 0.0:
            raise ValueError("penetration threshold cannot be negative")
        self.obstacle_prim_path = str(obstacle_prim_path)
        self.persistence_ticks = int(persistence_ticks)
        self.penetration_threshold_m = float(penetration_threshold_m)

    def evaluate(
        self,
        contacts: Mapping[str, BodyContactObservation],
        *,
        base_obstacle_penetration_m: float | None,
    ) -> BodyCollisionStatus:
        base = contacts.get(BASE_BODY)
        if base is None or base.role is not CollisionRole.BODY:
            return BodyCollisionStatus(
                detected=False,
                real_pair_active=False,
                persistent=False,
                geometry_penetration_m=0.0,
                reason="base_link BODY observation unavailable",
            )
        pair = base.obstacle
        real_pair = bool(
            pair.pair_verified
            and pair.active
            and pair.sensor_body == BASE_BODY
            and pair.other_body == self.obstacle_prim_path
        )
        persistent = bool(real_pair and pair.consecutive_active_ticks >= self.persistence_ticks)
        penetration = _finite_nonnegative(base_obstacle_penetration_m)
        geometry_corroborated = bool(penetration >= self.penetration_threshold_m)
        detected = bool(real_pair and (persistent or geometry_corroborated))
        if detected and persistent and geometry_corroborated:
            reason = "exact base_link/obstacle pair persisted and live collider bounds penetrate"
        elif detected and persistent:
            reason = "exact base_link/obstacle pair persisted"
        elif detected:
            reason = "exact base_link/obstacle pair corroborated by live collider penetration"
        elif real_pair:
            reason = "single-tick exact BODY pair awaits persistence or geometry corroboration"
        elif pair.active:
            reason = "active base contact is not a verified exact obstacle pair"
        else:
            reason = "no exact base_link/obstacle contact"
        return BodyCollisionStatus(
            detected=detected,
            real_pair_active=real_pair,
            persistent=persistent,
            geometry_penetration_m=penetration,
            reason=reason,
        )


def _finite_nonnegative(value: float | None) -> float:
    if value is None:
        return 0.0
    converted = float(value)
    return converted if math.isfinite(converted) and converted > 0.0 else 0.0

