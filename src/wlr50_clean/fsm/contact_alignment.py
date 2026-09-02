"""Fail-closed live-contact gate for the bounded P09 rear-left alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from wlr50_clean.reference.contact_alignment_contract import (
    DriveFeedbackContactAlignmentSpec,
)


MAX_CUMULATIVE_CORRECTION_FRACTION = 0.15
KIND = "post_probe_rear_left_air_alignment"
WHEEL_BODY = "rear_left_wheel"
REQUIRED_CONTACT_CLASS = "AIR"
TRIGGER_TICK = 859
FIRST_BIAS_TICK = 860
LAST_FULL_BIAS_TICK = 870
RELEASE_TICK = 871
TEARDOWN_TICK = 872
FINAL_SLEW_LIMIT_DEG_PER_TICK = 1.25
CHANNELS = (
    ("rear_left_hip", 4, 15.8, -1.185, 0.0, 2.37, 0.15),
    ("rear_left_knee", 5, 19.4, -1.455, -0.205, 2.91, 0.15),
)


@dataclass(frozen=True, slots=True)
class WheelContactEvidence:
    """Strict live fields used by the alignment gate."""

    wheel_body: str | None
    contact_class: str | None
    ground_pair_verified: bool | None
    ground_active: bool | None
    obstacle_pair_verified: bool | None
    obstacle_active: bool | None

    @property
    def is_verified_air(self) -> bool:
        return bool(
            self.wheel_body == WHEEL_BODY
            and self.contact_class == REQUIRED_CONTACT_CLASS
            and self.ground_pair_verified is True
            and self.ground_active is False
            and self.obstacle_pair_verified is True
            and self.obstacle_active is False
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "wheel_body": self.wheel_body,
            "contact_class": self.contact_class,
            "ground_pair_verified": self.ground_pair_verified,
            "ground_active": self.ground_active,
            "obstacle_pair_verified": self.obstacle_pair_verified,
            "obstacle_active": self.obstacle_active,
        }


@dataclass(frozen=True, slots=True)
class ContactAlignmentFeedback:
    """Auditable result of the P09 rear-left live-contact gate."""

    spec: DriveFeedbackContactAlignmentSpec
    active: bool
    just_triggered: bool
    trigger_tick: int | None
    condition_evaluated: bool
    condition_passed: bool
    evidence: WheelContactEvidence | None
    active_schedule_stage: str | None
    requested_bias_deg_by_channel: tuple[tuple[str, float], ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.spec.kind,
            "active": self.active,
            "just_triggered": self.just_triggered,
            "configured_trigger_tick": self.spec.trigger_tick,
            "trigger_tick": self.trigger_tick,
            "condition_evaluated": self.condition_evaluated,
            "condition_passed": self.condition_passed,
            "contact_evidence": (
                None if self.evidence is None else self.evidence.as_dict()
            ),
            "wheel_body": self.spec.wheel_body,
            "required_contact_class": self.spec.required_contact_class,
            "require_ground_pair_verified": self.spec.require_ground_pair_verified,
            "require_obstacle_pair_verified": (
                self.spec.require_obstacle_pair_verified
            ),
            "first_bias_tick": self.spec.first_bias_tick,
            "last_full_bias_tick": self.spec.last_full_bias_tick,
            "release_tick": self.spec.release_tick,
            "teardown_tick": self.spec.teardown_tick,
            "final_slew_limit_deg_per_tick": (
                self.spec.final_slew_limit_deg_per_tick
            ),
            "realized_zero_required_at_teardown": (
                self.spec.realized_zero_required_at_teardown
            ),
            "nominal_endpoint_restored": self.spec.nominal_endpoint_restored,
            "raw_recording_runtime_access_required": (
                self.spec.raw_recording_runtime_access_required
            ),
            "channels": [
                {
                    "channel": channel.channel,
                    "channel_index": channel.channel_index,
                    "reference_motion_magnitude_deg": (
                        channel.reference_motion_magnitude_deg
                    ),
                    "logical_full_bias_deg": channel.logical_full_bias_deg,
                    "logical_release_bias_deg": channel.logical_release_bias_deg,
                    "outbound_plus_teardown_deg": (
                        channel.outbound_plus_teardown_deg
                    ),
                    "cumulative_fraction_of_reference": (
                        channel.cumulative_fraction_of_reference
                    ),
                }
                for channel in self.spec.channels
            ],
            "active_schedule_stage": self.active_schedule_stage,
            "requested_bias_deg_by_channel": dict(self.requested_bias_deg_by_channel),
            "reason": self.reason,
        }


def wheel_contact_evidence(value: object | None) -> WheelContactEvidence:
    """Normalize a BodyContactObservation or strict mapping without guessing."""

    def field(item: object | None, name: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(name)
        return None if item is None else getattr(item, name, None)

    def strict_bool(item: Any) -> bool | None:
        return item if isinstance(item, bool) else None

    body = field(value, "body_name")
    raw_class = field(value, "contact_class")
    contact_class = getattr(raw_class, "value", raw_class)
    ground = field(value, "ground")
    obstacle = field(value, "obstacle")
    return WheelContactEvidence(
        wheel_body=body if isinstance(body, str) else None,
        contact_class=contact_class if isinstance(contact_class, str) else None,
        ground_pair_verified=strict_bool(field(ground, "pair_verified")),
        ground_active=strict_bool(field(ground, "active")),
        obstacle_pair_verified=strict_bool(field(obstacle, "pair_verified")),
        obstacle_active=strict_bool(field(obstacle, "active")),
    )


def runtime_spec_valid(spec: DriveFeedbackContactAlignmentSpec | None) -> bool:
    """Pin the runtime pulse to the compact contract and its two budgets."""

    if spec is None:
        return False
    signature = tuple(
        (
            channel.channel,
            channel.channel_index,
            channel.reference_motion_magnitude_deg,
            channel.logical_full_bias_deg,
            channel.logical_release_bias_deg,
            channel.outbound_plus_teardown_deg,
            channel.cumulative_fraction_of_reference,
        )
        for channel in spec.channels
    )
    if (
        spec.kind != KIND
        or spec.trigger_tick != TRIGGER_TICK
        or spec.wheel_body != WHEEL_BODY
        or spec.required_contact_class != REQUIRED_CONTACT_CLASS
        or spec.require_ground_pair_verified is not True
        or spec.require_obstacle_pair_verified is not True
        or spec.first_bias_tick != FIRST_BIAS_TICK
        or spec.last_full_bias_tick != LAST_FULL_BIAS_TICK
        or spec.release_tick != RELEASE_TICK
        or spec.teardown_tick != TEARDOWN_TICK
        or not math.isfinite(spec.final_slew_limit_deg_per_tick)
        or abs(spec.final_slew_limit_deg_per_tick - FINAL_SLEW_LIMIT_DEG_PER_TICK)
        > 1.0e-12
        or spec.realized_zero_required_at_teardown is not True
        or spec.nominal_endpoint_restored is not True
        or spec.raw_recording_runtime_access_required is not False
        or signature != CHANNELS
    ):
        return False
    return all(
        all(math.isfinite(value) for value in channel[2:])
        and abs(
            abs(channel[3])
            + abs(channel[4] - channel[3])
            + abs(channel[4])
            - channel[5]
        )
        <= 1.0e-12
        and abs(channel[5] / channel[2] - channel[6]) <= 1.0e-12
        and channel[6] <= MAX_CUMULATIVE_CORRECTION_FRACTION + 1.0e-12
        and abs(channel[4] - channel[3])
        <= spec.final_slew_limit_deg_per_tick + 1.0e-12
        and abs(channel[4]) <= spec.final_slew_limit_deg_per_tick + 1.0e-12
        for channel in signature
    )
