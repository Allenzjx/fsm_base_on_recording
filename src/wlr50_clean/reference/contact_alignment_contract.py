"""Compact schema and validation for the locked P09 contact alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ACTION_COUNT = 12
SERVO_COUNT = 8
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


@dataclass(frozen=True)
class DriveFeedbackContactAlignmentChannel:
    channel: str
    channel_index: int
    reference_motion_magnitude_deg: float
    logical_full_bias_deg: float
    logical_release_bias_deg: float
    outbound_plus_teardown_deg: float
    cumulative_fraction_of_reference: float


@dataclass(frozen=True)
class DriveFeedbackContactAlignmentSpec:
    kind: str
    trigger_tick: int
    wheel_body: str
    required_contact_class: str
    require_ground_pair_verified: bool
    require_obstacle_pair_verified: bool
    first_bias_tick: int
    last_full_bias_tick: int
    release_tick: int
    teardown_tick: int
    final_slew_limit_deg_per_tick: float
    realized_zero_required_at_teardown: bool
    nominal_endpoint_restored: bool
    raw_recording_runtime_access_required: bool
    channels: tuple[DriveFeedbackContactAlignmentChannel, ...]

    def bias_at(self, motion_tick: int) -> tuple[float, ...]:
        result = [0.0] * ACTION_COUNT
        for channel in self.channels:
            if self.first_bias_tick <= motion_tick <= self.last_full_bias_tick:
                result[channel.channel_index] = channel.logical_full_bias_deg
            elif motion_tick == self.release_tick:
                result[channel.channel_index] = channel.logical_release_bias_deg
        return tuple(result)


def parse_contact_alignment(
    value: Any,
) -> DriveFeedbackContactAlignmentSpec | None:
    if not isinstance(value, Mapping):
        return None

    def boolean(raw: Any, *, label: str) -> bool:
        if not isinstance(raw, bool):
            raise ValueError(f"{label}: expected a boolean")
        return raw

    return DriveFeedbackContactAlignmentSpec(
        kind=str(value["kind"]),
        trigger_tick=int(value["trigger_tick"]),
        wheel_body=str(value["wheel_body"]),
        required_contact_class=str(value["required_contact_class"]),
        require_ground_pair_verified=boolean(
            value["require_ground_pair_verified"],
            label="drive_feedback.contact_alignment.require_ground_pair_verified",
        ),
        require_obstacle_pair_verified=boolean(
            value["require_obstacle_pair_verified"],
            label="drive_feedback.contact_alignment.require_obstacle_pair_verified",
        ),
        first_bias_tick=int(value["first_bias_tick"]),
        last_full_bias_tick=int(value["last_full_bias_tick"]),
        release_tick=int(value["release_tick"]),
        teardown_tick=int(value["teardown_tick"]),
        final_slew_limit_deg_per_tick=float(
            value["final_slew_limit_deg_per_tick"]
        ),
        realized_zero_required_at_teardown=boolean(
            value["realized_zero_required_at_teardown"],
            label=(
                "drive_feedback.contact_alignment."
                "realized_zero_required_at_teardown"
            ),
        ),
        nominal_endpoint_restored=boolean(
            value["nominal_endpoint_restored"],
            label="drive_feedback.contact_alignment.nominal_endpoint_restored",
        ),
        raw_recording_runtime_access_required=boolean(
            value["raw_recording_runtime_access_required"],
            label=(
                "drive_feedback.contact_alignment."
                "raw_recording_runtime_access_required"
            ),
        ),
        channels=tuple(
            DriveFeedbackContactAlignmentChannel(
                channel=str(row["channel"]),
                channel_index=int(row["channel_index"]),
                reference_motion_magnitude_deg=float(
                    row["reference_motion_magnitude_deg"]
                ),
                logical_full_bias_deg=float(row["logical_full_bias_deg"]),
                logical_release_bias_deg=float(row["logical_release_bias_deg"]),
                outbound_plus_teardown_deg=float(
                    row["outbound_plus_teardown_deg"]
                ),
                cumulative_fraction_of_reference=float(
                    row["cumulative_fraction_of_reference"]
                ),
            )
            for row in value["channels"]
        ),
    )


def validate_contact_alignment(
    alignment: DriveFeedbackContactAlignmentSpec | None,
    *,
    state_id: str,
    delta_full12: Sequence[float],
    active_channels: Sequence[str],
    full12_order: tuple[str, ...],
    feedback_teardown_tick: int,
    last_probe_tick: int,
    maximum_cumulative_fraction: float,
) -> None:
    if alignment is None:
        raise ValueError(f"{state_id}: contact-alignment contract is missing")
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
        for channel in alignment.channels
    )
    if (
        alignment.kind != KIND
        or alignment.wheel_body != WHEEL_BODY
        or alignment.required_contact_class != REQUIRED_CONTACT_CLASS
        or alignment.require_ground_pair_verified is not True
        or alignment.require_obstacle_pair_verified is not True
        or alignment.trigger_tick != TRIGGER_TICK
        or alignment.trigger_tick != last_probe_tick
        or alignment.first_bias_tick != FIRST_BIAS_TICK
        or alignment.last_full_bias_tick != LAST_FULL_BIAS_TICK
        or alignment.release_tick != RELEASE_TICK
        or alignment.teardown_tick != TEARDOWN_TICK
        or alignment.teardown_tick != feedback_teardown_tick
        or alignment.first_bias_tick != alignment.trigger_tick + 1
        or alignment.release_tick != alignment.last_full_bias_tick + 1
        or alignment.teardown_tick != alignment.release_tick + 1
        or not math.isfinite(alignment.final_slew_limit_deg_per_tick)
        or abs(
            alignment.final_slew_limit_deg_per_tick
            - FINAL_SLEW_LIMIT_DEG_PER_TICK
        )
        > 1.0e-12
        or alignment.realized_zero_required_at_teardown is not True
        or alignment.nominal_endpoint_restored is not True
        or alignment.raw_recording_runtime_access_required is not False
        or signature != CHANNELS
    ):
        raise ValueError(f"{state_id}: invalid contact-alignment contract")
    for channel in alignment.channels:
        path = (
            abs(channel.logical_full_bias_deg)
            + abs(channel.logical_release_bias_deg - channel.logical_full_bias_deg)
            + abs(channel.logical_release_bias_deg)
        )
        reference = abs(float(delta_full12[channel.channel_index]))
        fraction = path / reference
        if (
            channel.channel_index < 0
            or channel.channel_index >= SERVO_COUNT
            or full12_order[channel.channel_index] != channel.channel
            or channel.channel not in active_channels
            or any(
                not math.isfinite(value)
                for value in (
                    channel.reference_motion_magnitude_deg,
                    channel.logical_full_bias_deg,
                    channel.logical_release_bias_deg,
                    channel.outbound_plus_teardown_deg,
                    channel.cumulative_fraction_of_reference,
                )
            )
            or abs(channel.reference_motion_magnitude_deg - reference) > 1.0e-12
            or abs(channel.outbound_plus_teardown_deg - path) > 1.0e-12
            or abs(channel.cumulative_fraction_of_reference - fraction) > 1.0e-12
            or fraction > maximum_cumulative_fraction + 1.0e-12
            or abs(
                channel.logical_release_bias_deg - channel.logical_full_bias_deg
            )
            > alignment.final_slew_limit_deg_per_tick + 1.0e-12
            or abs(channel.logical_release_bias_deg)
            > alignment.final_slew_limit_deg_per_tick + 1.0e-12
        ):
            raise ValueError(f"{state_id}: contact-alignment budget is invalid")
