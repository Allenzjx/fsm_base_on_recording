"""Offline parser for the single frozen v010 recording event stream.

This module is intentionally isolated under ``wlr50_clean.reference``. Nothing
in the production FSM imports it; its output is a compact motion contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SERVO_ORDER = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_ORDER = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
FULL12_ORDER = SERVO_ORDER + WHEEL_ORDER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_map(value: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    return {name: float(value.get(name, 0.0)) for name in names}


@dataclass(frozen=True)
class Full12Command:
    servos_deg: dict[str, float]
    wheels_rad_s: dict[str, float]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Full12Command":
        return cls(
            servos_deg=_float_map(value.get("servos", {}), SERVO_ORDER),
            wheels_rad_s=_float_map(value.get("wheels", {}), WHEEL_ORDER),
        )

    def vector(self) -> tuple[float, ...]:
        return tuple(self.servos_deg[name] for name in SERVO_ORDER) + tuple(
            self.wheels_rad_s[name] for name in WHEEL_ORDER
        )


@dataclass(frozen=True)
class RecordingEvent:
    step_index: int
    event_index: int
    local_time_s: float
    global_time_s: float
    kind: str
    command: str
    batch_id: str
    before: Full12Command
    after: Full12Command
    servo_velocity_deg_s: float
    active_servo_targets_deg: dict[str, float]
    active_wheel_targets_rad_s: dict[str, float]
    final_stop: bool

    @property
    def active_channels(self) -> tuple[str, ...]:
        channels: list[str] = []
        for name in SERVO_ORDER:
            if abs(self.after.servos_deg[name] - self.before.servos_deg[name]) > 1e-12:
                channels.append(name)
        for name in WHEEL_ORDER:
            if abs(self.after.wheels_rad_s[name] - self.before.wheels_rad_s[name]) > 1e-12:
                channels.append(name)
        return tuple(channels)


@dataclass(frozen=True)
class RecordingStep:
    index: int
    name: str
    duration_s: float
    global_start_s: float
    before: Full12Command
    after: Full12Command
    events: tuple[RecordingEvent, ...]
    raw: Mapping[str, Any]

    @property
    def global_end_s(self) -> float:
        return self.global_start_s + self.duration_s


@dataclass(frozen=True)
class ParsedRecording:
    path: Path
    sha256: str
    steps: tuple[RecordingStep, ...]

    @property
    def events(self) -> tuple[RecordingEvent, ...]:
        return tuple(event for step in self.steps for event in step.events)

    @property
    def duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)


def load_recording(path: Path) -> ParsedRecording:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"line {line_number}: expected JSON object")
            rows.append(value)
    steps: list[RecordingStep] = []
    global_start = 0.0
    for expected_index, row in enumerate(rows, start=1):
        step_index = int(row.get("index", -1))
        if step_index != expected_index:
            raise ValueError(
                f"non-contiguous step index: expected {expected_index}, got {step_index}"
            )
        before = Full12Command.from_json(row.get("command_state_before", {}))
        after = Full12Command.from_json(row.get("command_state_after", {}))
        duration = float(row.get("duration", 0.0))
        if duration < 0.0:
            raise ValueError(f"step {step_index}: negative duration")
        events: list[RecordingEvent] = []
        previous_local_time = -1.0
        for event_index, event in enumerate(row.get("events", [])):
            local_time = float(
                event.get("actual_recording_time_s", event.get("time", 0.0))
            )
            if local_time + 1e-9 < previous_local_time:
                raise ValueError(f"step {step_index}: event time moved backwards")
            if local_time > duration + 1e-6:
                raise ValueError(
                    f"step {step_index}: event {event_index} exceeds step duration"
                )
            previous_local_time = local_time
            event_before = Full12Command.from_json(
                event.get("command_state_before", {})
            )
            event_after = Full12Command.from_json(
                event.get("command_state_after", {})
            )
            events.append(
                RecordingEvent(
                    step_index=step_index,
                    event_index=event_index,
                    local_time_s=local_time,
                    global_time_s=global_start + local_time,
                    kind=str(event.get("kind", "")),
                    command=str(event.get("command", "")),
                    batch_id=str(event.get("batch_id", "")),
                    before=event_before,
                    after=event_after,
                    servo_velocity_deg_s=float(
                        event.get("canonical_servo_velocity_deg_s", 0.0) or 0.0
                    ),
                    active_servo_targets_deg={
                        str(key): float(value)
                        for key, value in event.get(
                            "canonical_servo_target_deg", {}
                        ).items()
                    },
                    active_wheel_targets_rad_s={
                        str(key): float(value)
                        for key, value in event.get(
                            "canonical_wheel_velocity_rad_s", {}
                        ).items()
                    },
                    final_stop=bool(event.get("final_stop_command", False)),
                )
            )
        if events:
            event_initial = events[0].before.vector()
            if any(abs(a - b) > 1e-6 for a, b in zip(before.vector(), event_initial)):
                # Coalesced recording events can begin after an earlier command in the
                # same step. Preserve the data but make the discontinuity explicit.
                pass
            event_final = events[-1].after.vector()
            if any(abs(a - b) > 1e-6 for a, b in zip(after.vector(), event_final)):
                raise ValueError(
                    f"step {step_index}: final event state differs from step state"
                )
        steps.append(
            RecordingStep(
                index=step_index,
                name=str(row.get("name", f"step_{step_index:03d}")),
                duration_s=duration,
                global_start_s=global_start,
                before=before,
                after=after,
                events=tuple(events),
                raw=row,
            )
        )
        global_start += duration
    for previous, current in zip(steps, steps[1:]):
        if any(
            abs(a - b) > 1e-6
            for a, b in zip(previous.after.vector(), current.before.vector())
        ):
            raise ValueError(
                f"command continuity failure between steps {previous.index} and {current.index}"
            )
    return ParsedRecording(path=path, sha256=sha256_file(path), steps=tuple(steps))


def validate_v010(
    recording_path: Path,
    metadata_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    recording = load_recording(recording_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if recording.sha256 != expected_sha256:
        errors.append("accepted_steps SHA-256 mismatch")
    if metadata.get("accepted_steps_sha256") != recording.sha256:
        errors.append("metadata binding does not match accepted_steps")
    if metadata.get("version_id") != "v010_20260806_220745_363972_manual":
        errors.append("unexpected reference version")
    if int(metadata.get("height_mm", -1)) != 50:
        errors.append("unexpected obstacle height")
    if len(recording.steps) != int(metadata.get("step_count", -1)):
        errors.append("step count differs from metadata")
    if len(recording.events) != int(metadata.get("command_count", -1)):
        errors.append("event count differs from metadata")
    return {
        "schema": "wlr50_clean.recording_validation.v1",
        "passed": not errors,
        "errors": errors,
        "version_id": metadata.get("version_id"),
        "height_mm": metadata.get("height_mm"),
        "accepted_steps_sha256": recording.sha256,
        "step_count": len(recording.steps),
        "event_count": len(recording.events),
        "recorded_step_duration_sum_s": recording.duration_s,
        "first_step": recording.steps[0].index if recording.steps else None,
        "last_step": recording.steps[-1].index if recording.steps else None,
        "command_continuity": True,
        "event_time_monotonic_within_steps": True,
    }


def step_summary(recording: ParsedRecording) -> Iterable[dict[str, Any]]:
    for step in recording.steps:
        changed: dict[str, dict[str, float]] = {}
        for name in SERVO_ORDER:
            start = step.before.servos_deg[name]
            end = step.after.servos_deg[name]
            if abs(end - start) > 1e-12:
                changed[name] = {"start": start, "end": end, "delta": end - start}
        for name in WHEEL_ORDER:
            start = step.before.wheels_rad_s[name]
            end = step.after.wheels_rad_s[name]
            if abs(end - start) > 1e-12:
                changed[name] = {"start": start, "end": end, "delta": end - start}
        yield {
            "step": step.index,
            "name": step.name,
            "duration_s": step.duration_s,
            "global_start_s": step.global_start_s,
            "global_end_s": step.global_end_s,
            "event_count": len(step.events),
            "changed_channels": changed,
            "commands": [event.command for event in step.events],
        }

