"""Immutable trial ledgers, physical evidence audits, and final publication."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.fsm.wheel_decay import WheelDecayDebounce, WheelDecayStatus
from wlr50_clean.infrastructure.command_batch import WHEEL_VELOCITY_LIMIT_RAD_S

from .comparison import PHASE_IDS
from .conformance import (
    SIMILARITY_COLUMNS,
    TrialAnalysisError,
    _number,
    _phase,
    _phase_windows,
    _samples,
    _similarity_rows,
    _summary,
    _time,
    _vector,
)


JSONL_FILES = {
    "observation": "observation_120hz.jsonl",
    "decision": "decision_15hz.jsonl",
    "command": "full12_commands_120hz.jsonl",
    "transition": "state_transitions.jsonl",
    "task_event": "task_events.jsonl",
    "body_contact": "body_contacts.jsonl",
    "leg_crossing": "leg_crossing_events.jsonl",
}
REQUIRED_TRIAL_FILES = tuple(JSONL_FILES.values()) + (
    "reference_similarity.csv", "actual_viewport_video.mp4", "trial_manifest.json",
)
FINAL_DATA_NAMES = (
    "selected_reference.json", "recording_motion_contract.json",
    "fsm_state_table.csv", "state_derivation.md",
    "fsm_vs_recording_similarity.csv", "leg_crossing_events.csv",
    "body_collision_audit.csv", "wheel_only_climb_audit.csv",
    "successful_trial_manifest.json",
)

_SERVO_DRIVE_FEEDBACK_KIND = "verify_tail_carry_alignment"
_WHEEL_DRIVE_FEEDBACK_KIND = "verify_tail_wheel_carry_alignment"
_WHEEL_REBOUND_FEEDBACK_KIND = "pre_endpoint_wheel_rebound_alignment"
_WHEEL_TAIL_CHANNEL = "front_left_ankle"
_WHEEL_TAIL_CHANNEL_INDEX = 8
_WHEEL_PROBE_CHANNEL = "rear_right_knee"
_WHEEL_PROBE_CHANNEL_INDEX = 7
_WHEEL_TAIL_PROBE_TICKS = (858, 859)
_WHEEL_PROBE_REFERENCES_DEG = (-51.055799822535, -51.191638624749)
_WHEEL_REBOUND_LAG_THRESHOLD_DEG = 0.35
_WHEEL_TAIL_FIRST_TICK = 864
_WHEEL_ENDPOINT_TICK = 864
_WHEEL_TAIL_LAST_TICK = 871
_WHEEL_TAIL_TEARDOWN_TICK = 872
_WHEEL_TAIL_VELOCITY_RAD_S = -1.07
_WHEEL_REBOUND_FIRST_TICK = 860
_WHEEL_REBOUND_LAST_TICK = 871
_WHEEL_REBOUND_TEARDOWN_TICK = 872
_WHEEL_REBOUND_BIAS_SEGMENTS = (
    (860, 860, 0.68),
    (861, 861, 0.33),
    (862, 864, 0.68),
    (865, 865, 0.15),
    (866, 867, 1.03),
    (868, 868, 0.73),
    (870, 870, 0.01),
    (871, 871, 0.40),
)
_WHEEL_REFERENCE_INTEGRAL_RAD = -0.9060000000012605
_WHEEL_REBOUND_ADDITIONAL_INTEGRAL_RAD = 0.05333333333333334
_WHEEL_REBOUND_RESULTING_INTEGRAL_RAD = -0.8526666666679271
_WHEEL_REFERENCE_PEAK_ABS_RAD_S = 1.07
_WHEEL_REBOUND_RESULTING_PEAK_ABS_RAD_S = 1.07
_MAX_FEEDBACK_FRACTION = 0.15


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


class TrialArtifactWriter:
    """Create one never-reused directory and stream every required ledger."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._streams = {
            name: (self.run_dir / filename).open("x", encoding="utf-8", newline="\n")
            for name, filename in JSONL_FILES.items()
        }
        self._similarity_stream = (self.run_dir / "reference_similarity.csv").open(
            "x", encoding="utf-8", newline=""
        )
        self._similarity_writer = csv.DictWriter(
            self._similarity_stream, fieldnames=SIMILARITY_COLUMNS, extrasaction="raise"
        )
        self._similarity_writer.writeheader()
        self._closed = False

    @property
    def video_path(self) -> Path:
        return self.run_dir / "actual_viewport_video.mp4"

    def append(self, stream_name: str, payload: Mapping[str, Any] | Any) -> None:
        if self._closed:
            raise RuntimeError("trial writer is closed")
        if stream_name not in self._streams:
            raise KeyError(stream_name)
        self._streams[stream_name].write(
            json.dumps(_json_value(payload), separators=(",", ":"), default=_json_value) + "\n"
        )
        self._streams[stream_name].flush()

    def append_similarity(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("trial writer is closed")
        self._similarity_writer.writerow({name: row.get(name, "") for name in SIMILARITY_COLUMNS})
        self._similarity_stream.flush()

    def finalize_manifest(self, manifest: Mapping[str, Any]) -> Path:
        path = self.run_dir / "trial_manifest.json"
        if path.exists():
            raise FileExistsError("trial manifest is immutable and already exists")
        payload = dict(manifest)
        payload.setdefault("schema", "wlr50_clean.trial_manifest.v1")
        payload["artifact_files"] = {
            name: _file_record(self.run_dir / filename) for name, filename in JSONL_FILES.items()
        }
        payload["artifact_files"]["reference_similarity"] = _file_record(
            self.run_dir / "reference_similarity.csv"
        )
        if self.video_path.exists():
            payload["artifact_files"]["actual_viewport_video"] = _file_record(self.video_path)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def close(self) -> None:
        if not self._closed:
            for stream in self._streams.values():
                stream.close()
            self._similarity_stream.close()
            self._closed = True

    def __enter__(self) -> "TrialArtifactWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrialAnalysisError(f"required run ledger is missing: {path.name}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrialAnalysisError(f"{path.name}:{number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise TrialAnalysisError(f"{path.name}:{number}: row is not an object")
        rows.append(value)
    return rows


def _guard(row: Mapping[str, Any], name: str) -> tuple[bool, Mapping[str, Any]]:
    guards = row.get("guards")
    value = guards.get(name) if isinstance(guards, Mapping) else None
    return (bool(value.get("passed")), value) if isinstance(value, Mapping) else (False, {})


def _leg_audit(
    observations: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    event_names = {
        "ACTIVE_LIFT": "reference_like_active_lift:{leg}",
        "FRONT_FACE_CROSSED": "leg_front_face_crossed_latched:{leg}",
        "TOP_LOADED": "leg_top_loaded_latched:{leg}",
    }
    for leg in ("FR", "FL", "RR", "RL"):
        for event, template in event_names.items():
            canonical = next((
                row for row in raw_rows
                if str(row.get("leg", "")).upper() == leg
                and str(row.get("event") or row.get("kind") or row.get("event_type") or "").upper()
                in {event, event.replace("_FACE", "")}
            ), None)
            if canonical is not None:
                when = _number(canonical, "simulation_time_s", "sim_time_s", "time_s")
                events.append({
                    "leg": leg, "event": event, "physics_tick": canonical.get("physics_tick", ""),
                    "simulation_time_s": 0.0 if when is None else when, "state_id": _phase(canonical),
                    "source": "leg_crossing_events.jsonl",
                    "evidence": json.dumps(canonical.get("evidence", canonical), separators=(",", ":")),
                })
                continue
            guard_name = template.format(leg=leg)
            for observation in sorted(observations, key=_time):
                passed, evidence = _guard(observation, guard_name)
                if passed:
                    events.append({
                        "leg": leg, "event": event,
                        "physics_tick": observation.get("physics_tick", ""),
                        "simulation_time_s": _time(observation), "state_id": _phase(observation),
                        "source": "observation_120hz.guard_latch",
                        "evidence": json.dumps(evidence, separators=(",", ":")),
                    })
                    break
    wheel_audit: list[dict[str, Any]] = []
    for leg in ("FR", "FL", "RR", "RL"):
        by_event = {row["event"]: row for row in events if row["leg"] == leg}
        lift, crossed = by_event.get("ACTIVE_LIFT"), by_event.get("FRONT_FACE_CROSSED")
        wheel_only = lift is None or crossed is None or (
            float(lift["simulation_time_s"]) > float(crossed["simulation_time_s"]) + 1.0e-9
        )
        wheel_audit.append({
            "leg": leg, "active_lift_time_s": "" if lift is None else lift["simulation_time_s"],
            "front_face_crossed_time_s": "" if crossed is None else crossed["simulation_time_s"],
            "active_lift_preceded_or_coincided_with_crossing": not wheel_only,
            "wheel_only_climb": wheel_only,
        })
    return events, wheel_audit


def _body_audit(
    observations: Sequence[Mapping[str, Any]], body_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in body_rows:
        detected = bool(row.get("detected") or row.get("body_collision"))
        role = str(row.get("role", "BODY"))
        when = _number(row, "simulation_time_s", "sim_time_s", "time_s")
        result.append({
            "physics_tick": row.get("physics_tick", ""),
            "simulation_time_s": "" if when is None else when,
            "body_name": row.get("body_name", row.get("sensor_body", "base_link")), "role": role,
            "obstacle_pair_active": bool(row.get("obstacle_active", row.get("active", detected))),
            "persistent_or_penetrating": detected, "body_collision": detected and role == "BODY",
            "source": "body_contacts.jsonl", "reason": row.get("reason", ""),
        })
    for observation in observations:
        status = observation.get("body_collision")
        if isinstance(status, Mapping) and status.get("detected"):
            result.append({
                "physics_tick": observation.get("physics_tick", ""),
                "simulation_time_s": _time(observation), "body_name": "base_link", "role": "BODY",
                "obstacle_pair_active": status.get("real_pair_active", ""),
                "persistent_or_penetrating": True, "body_collision": True,
                "source": "observation_120hz.body_collision", "reason": status.get("reason", ""),
            })
    if not result:
        result.append({
            "physics_tick": "", "simulation_time_s": "", "body_name": "base_link", "role": "BODY",
            "obstacle_pair_active": False, "persistent_or_penetrating": False,
            "body_collision": False, "source": "complete ledgers; no positive event",
            "reason": "no BODY/obstacle collision",
        })
    return result


def _terminal_success(task_rows: Sequence[Mapping[str, Any]]) -> bool:
    results = [str(row.get("result")) for row in task_rows if row.get("result") is not None]
    failures = {"TASK_FAILURE_BODY_COLLISION", "TASK_FAILURE_WHEEL_ONLY_CLIMB"}
    return bool(results) and results[-1] == "SUCCESS" and not any(item in failures for item in results)


def _recovery_evidence(
    transitions: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[float], dict[str, int]]:
    recovery_rows = [
        row for row in transitions
        if str(row.get("from_lifecycle")) == "RECOVERY"
        and str(row.get("to_lifecycle")) == "EXECUTE_MOTION"
    ]
    # Normal sensor-derived entry corrections are as contract-sensitive as a
    # recovery correction.  Include every explicitly logged correction vector
    # in the bound audit, while retry counts remain recovery-only.
    rows = list(recovery_rows)
    rows.extend(
        row
        for row in transitions
        if row not in recovery_rows
        and isinstance(row.get("details"), Mapping)
        and "correction_fractions" in row["details"]
    )
    cumulative_by_phase_channel: dict[tuple[str, str], float] = {}
    for row in rows:
        details = row.get("details")
        raw = details.get("correction_fractions") if isinstance(details, Mapping) else None

        def correction_candidates(value: Any) -> tuple[tuple[str, Any], ...]:
            if isinstance(value, Mapping):
                return tuple((str(name), item) for name, item in value.items())
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and len(value) == 12
            ):
                return tuple((str(index), item) for index, item in enumerate(value))
            return (("invalid", math.inf),)

        candidates = correction_candidates(raw)
        if (
            isinstance(details, Mapping)
            and "recovery_correction_fractions" in details
        ):
            auxiliary = correction_candidates(
                details["recovery_correction_fractions"]
            )
            try:
                ledger_matches = len(candidates) == len(auxiliary) and all(
                    left_name == right_name
                    and math.isfinite(float(left_value))
                    and math.isfinite(float(right_value))
                    and abs(float(left_value) - float(right_value)) <= 1e-12
                    for (left_name, left_value), (right_name, right_value) in zip(
                        candidates, auxiliary, strict=True
                    )
                )
            except (TypeError, ValueError):
                ledger_matches = False
            if not ledger_matches:
                candidates = (("invalid_recovery_correction_ledger", math.inf),)
        for channel, item in candidates:
            try:
                parsed = abs(float(item))
            except (TypeError, ValueError):
                parsed = math.inf
            key = (_phase(row), channel)
            cumulative_by_phase_channel[key] = (
                cumulative_by_phase_channel.get(key, 0.0)
                + (parsed if math.isfinite(parsed) else math.inf)
            )
    nonzero_feedback_seen = False
    feedback_trigger_seen = False
    for row in commands:
        feedback = row.get("drive_feedback")
        if not isinstance(feedback, Mapping):
            continue
        raw = feedback.get("cumulative_fraction_of_reference")
        try:
            parsed = abs(float(raw))
        except (TypeError, ValueError):
            parsed = math.inf
        parsed = parsed if math.isfinite(parsed) else math.inf
        nonzero_feedback_seen = nonzero_feedback_seen or parsed > 1.0e-12
        if feedback.get("just_triggered") is not True:
            continue
        feedback_trigger_seen = True
        try:
            channel = str(int(feedback["correction_channel_index"]))
        except (KeyError, TypeError, ValueError):
            channel = "invalid_drive_feedback"
            parsed = math.inf
        key = (_phase(row), channel)
        cumulative_by_phase_channel[key] = (
            cumulative_by_phase_channel.get(key, 0.0) + parsed
        )
    if nonzero_feedback_seen and not feedback_trigger_seen:
        cumulative_by_phase_channel[("invalid", "drive_feedback")] = math.inf
    return list(cumulative_by_phase_channel.values()), {
        phase: sum(_phase(row) == phase for row in recovery_rows) for phase in PHASE_IDS
    }


def _reference_wheel_decay_threshold(contract: Mapping[str, Any]) -> float:
    """Return the v010-relative P13 measured-tail acceptance envelope.

    The successful recording does not settle below 0.05 rad/s.  Its P13
    result observation instead records the measured tail peak for every
    wheel, so the runtime and offline analyzer both use the same +15 percent
    conformance envelope around that physical evidence.
    """

    phases = contract.get("phases")
    if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        p13 = next(
            (
                phase
                for phase in phases
                if isinstance(phase, Mapping) and str(phase.get("state_id")) == "P13"
            ),
            None,
        )
        if p13 is not None:
            result = p13.get("reference_result_observation")
            peaks = (
                result.get("wheel_tail_peak_abs_velocity_rad_s")
                if isinstance(result, Mapping)
                else None
            )
            if isinstance(peaks, Mapping):
                finite = []
                for value in peaks.values():
                    try:
                        parsed = abs(float(value))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(parsed):
                        finite.append(parsed)
                if finite:
                    return 1.15 * max(finite)
    # Retain a fail-closed legacy floor for small synthetic contracts that do
    # not carry the v010 result-tail evidence.
    return 0.05


def _drive_feedback_mode(spec: Mapping[str, Any]) -> str | None:
    kind = spec.get("kind")
    if kind == _SERVO_DRIVE_FEEDBACK_KIND:
        return "servo"
    if kind == _WHEEL_DRIVE_FEEDBACK_KIND:
        return "wheel"
    if kind == _WHEEL_REBOUND_FEEDBACK_KIND:
        return "wheel_rebound"
    return None


def _is_wheel_feedback_mode(mode: str | None) -> bool:
    return mode in ("wheel", "wheel_rebound")


def _wheel_feedback_shape(
    mode: str | None,
) -> tuple[tuple[int, ...], int, int, int, float] | None:
    if mode == "wheel":
        return (
            _WHEEL_TAIL_PROBE_TICKS,
            _WHEEL_TAIL_FIRST_TICK,
            _WHEEL_TAIL_LAST_TICK,
            _WHEEL_TAIL_TEARDOWN_TICK,
            _WHEEL_TAIL_VELOCITY_RAD_S,
        )
    if mode == "wheel_rebound":
        return (
            _WHEEL_TAIL_PROBE_TICKS,
            _WHEEL_REBOUND_FIRST_TICK,
            _WHEEL_REBOUND_LAST_TICK,
            _WHEEL_REBOUND_TEARDOWN_TICK,
            _WHEEL_REBOUND_BIAS_SEGMENTS[0][2],
        )
    return None


def _wheel_feedback_segments(
    spec: Mapping[str, Any], mode: str | None
) -> tuple[tuple[int, int, float], ...] | None:
    """Parse the exact logical-bias schedule used by a wheel correction."""

    if mode == "wheel":
        try:
            segments = ((
                int(spec["first_bias_tick"]),
                int(spec["last_bias_tick"]),
                float(spec["logical_bias_rad_s"]),
            ),)
        except (KeyError, TypeError, ValueError):
            return None
    elif mode == "wheel_rebound":
        raw_segments = spec.get("bias_segments")
        if (
            not isinstance(raw_segments, Sequence)
            or isinstance(raw_segments, (str, bytes))
        ):
            return None
        parsed: list[tuple[int, int, float]] = []
        try:
            for raw in raw_segments:
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != {
                        "first_bias_tick",
                        "last_bias_tick",
                        "logical_bias_rad_s",
                    }
                    or isinstance(raw["first_bias_tick"], bool)
                    or not isinstance(raw["first_bias_tick"], int)
                    or isinstance(raw["last_bias_tick"], bool)
                    or not isinstance(raw["last_bias_tick"], int)
                    or isinstance(raw["logical_bias_rad_s"], bool)
                ):
                    return None
                parsed.append((
                    int(raw["first_bias_tick"]),
                    int(raw["last_bias_tick"]),
                    float(raw["logical_bias_rad_s"]),
                ))
        except (KeyError, TypeError, ValueError):
            return None
        segments = tuple(parsed)
        # The P09 rebound shape is deliberately immutable: accepting a shifted,
        # shortened, extended, or re-amplituded segment could hide a
        # pre-endpoint direction reversal or change the actual physical timing
        # while leaving the signed integral apparently valid.
        if segments != _WHEEL_REBOUND_BIAS_SEGMENTS:
            return None
    else:
        return None
    if not segments or any(
        first < 0
        or last < first
        or not math.isfinite(bias)
        or abs(bias) <= 1.0e-12
        for first, last, bias in segments
    ):
        return None
    if any(
        right[0] <= left[1]
        for left, right in zip(segments, segments[1:])
    ):
        return None
    return segments


def _wheel_feedback_contract_values(
    spec: Mapping[str, Any], *, physics_hz: float, mode: str | None = None
) -> tuple[float, float] | None:
    """Return the declared reference integral and derived correction fraction.

    Wheel feedback is signed-integral bounded, not a servo excursion.  The
    legacy carry remains same-direction only; the rebound kind has its own
    mandatory opposite-direction semantics.  Unitful fields prevent either
    correction from being disguised behind legacy degree-named budgets.
    """

    selected_mode = _drive_feedback_mode(spec) if mode is None else mode
    shape = _wheel_feedback_shape(selected_mode)
    segments = _wheel_feedback_segments(spec, selected_mode)
    if shape is None or segments is None:
        return None
    try:
        signed_reference_integral = float(spec["reference_wheel_integral_rad"])
        reference_integral = abs(signed_reference_integral)
        declared_integral = float(spec["additional_wheel_integral_rad"])
        declared_fraction = float(spec["cumulative_fraction_of_reference"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (
        signed_reference_integral,
        reference_integral,
        declared_integral,
        declared_fraction,
        physics_hz,
    )
    if any(not math.isfinite(value) for value in values):
        return None
    if (
        physics_hz <= 0.0
        or reference_integral <= 0.0
    ):
        return None
    expected_signed_integral = sum(
        bias * (last - first + 1) / physics_hz
        for first, last, bias in segments
    )
    expected_absolute_integral = sum(
        abs(bias) * (last - first + 1) / physics_hz
        for first, last, bias in segments
    )
    derived_fraction = expected_absolute_integral / reference_integral
    if (
        abs(declared_integral - expected_signed_integral) > 1.0e-12
        or declared_fraction < 0.0
        or derived_fraction > _MAX_FEEDBACK_FRACTION + 1.0e-12
        or abs(declared_fraction - derived_fraction) > 1.0e-12
    ):
        return None
    if selected_mode == "wheel":
        # The legacy carry is same-direction only.  A counter-carry must never
        # be smuggled through this kind merely because its absolute peak is
        # unchanged.
        if (
            segments[0][2] * signed_reference_integral <= 0.0
            or spec.get("instantaneous_direction_reversal") is True
        ):
            return None
    else:
        try:
            resulting_integral = float(spec["resulting_wheel_integral_rad"])
            reference_peak = float(spec["reference_wheel_peak_abs_rad_s"])
            resulting_peak = float(spec["resulting_wheel_peak_abs_rad_s"])
        except (KeyError, TypeError, ValueError):
            return None
        if any(
            not math.isfinite(value)
            for value in (
                resulting_integral,
                reference_peak,
                resulting_peak,
            )
        ):
            return None
        peak_magnitude_fraction = (
            math.inf
            if reference_peak <= 0.0
            else abs(resulting_peak - reference_peak) / reference_peak
        )
        if (
            any(
                bias * signed_reference_integral >= 0.0
                for _, _, bias in segments
            )
            or spec.get("instantaneous_direction_reversal") is not True
            or abs(
                signed_reference_integral - _WHEEL_REFERENCE_INTEGRAL_RAD
            )
            > 1.0e-12
            or abs(
                expected_signed_integral
                - _WHEEL_REBOUND_ADDITIONAL_INTEGRAL_RAD
            )
            > 1.0e-12
            or abs(
                resulting_integral
                - (signed_reference_integral + expected_signed_integral)
            )
            > 1.0e-12
            or abs(
                resulting_integral - _WHEEL_REBOUND_RESULTING_INTEGRAL_RAD
            )
            > 1.0e-12
            or abs(reference_peak - _WHEEL_REFERENCE_PEAK_ABS_RAD_S)
            > 1.0e-12
            or abs(
                resulting_peak - _WHEEL_REBOUND_RESULTING_PEAK_ABS_RAD_S
            )
            > 1.0e-12
            # ``peak_fraction_of_reference`` in the runtime ledger denotes
            # peak-magnitude *increase*.  Direction reversal is audited by its
            # own mandatory flag and exact native/final shapes below.
            or abs(peak_magnitude_fraction) > 1.0e-12
        ):
            return None
    return reference_integral, derived_fraction


def _phase_zoh_channel_integral(
    phase: Mapping[str, Any], channel_index: int
) -> float | None:
    """Re-derive a phase command integral from its immutable ZOH waypoints."""

    try:
        active_duration = float(phase["active_duration_s"])
        raw_waypoints = phase["waypoints"]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(active_duration)
        or active_duration <= 0.0
        or not isinstance(raw_waypoints, Sequence)
        or isinstance(raw_waypoints, (str, bytes))
        or not raw_waypoints
        or channel_index < 0
        or channel_index >= 12
    ):
        return None
    samples: list[tuple[float, float]] = []
    try:
        for waypoint in raw_waypoints:
            if not isinstance(waypoint, Mapping):
                return None
            time_s = float(waypoint["time_s"])
            full12 = waypoint["full12"]
            if (
                not math.isfinite(time_s)
                or time_s < -1.0e-9
                or time_s > active_duration + 1.0e-6
                or not isinstance(full12, Sequence)
                or isinstance(full12, (str, bytes))
                or len(full12) != 12
            ):
                return None
            target = float(full12[channel_index])
            if not math.isfinite(target):
                return None
            samples.append((time_s, target))
    except (KeyError, TypeError, ValueError):
        return None
    if abs(samples[0][0]) > 1.0e-9 or any(
        right[0] + 1.0e-9 < left[0]
        for left, right in zip(samples, samples[1:])
    ):
        return None
    result = 0.0
    for index, (time_s, target) in enumerate(samples):
        left = min(active_duration, max(0.0, time_s))
        right = (
            active_duration
            if index + 1 == len(samples)
            else min(active_duration, max(0.0, samples[index + 1][0]))
        )
        result += target * max(0.0, right - left)
    return result


def _phase_zoh_channel_peak_abs(
    phase: Mapping[str, Any], channel_index: int
) -> float | None:
    """Re-derive a channel peak from the frozen compact waypoints."""

    raw_waypoints = phase.get("waypoints")
    if (
        not isinstance(raw_waypoints, Sequence)
        or isinstance(raw_waypoints, (str, bytes))
        or not raw_waypoints
        or channel_index < 0
        or channel_index >= 12
    ):
        return None
    values: list[float] = []
    try:
        for waypoint in raw_waypoints:
            if not isinstance(waypoint, Mapping):
                return None
            full12 = waypoint["full12"]
            if (
                not isinstance(full12, Sequence)
                or isinstance(full12, (str, bytes))
                or len(full12) != 12
            ):
                return None
            value = float(full12[channel_index])
            if not math.isfinite(value):
                return None
            values.append(abs(value))
    except (KeyError, TypeError, ValueError):
        return None
    return max(values)


def _phase_channel_at_motion_tick(
    phase: Mapping[str, Any], channel_index: int, tick: int, *, physics_hz: float
) -> float | None:
    """Match the sequencer's integer endpoint dispatch for one channel."""

    try:
        active_duration = float(phase["active_duration_s"])
        endpoint = phase["end_full12"]
        waypoints = phase["waypoints"]
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(active_duration)
        or active_duration <= 0.0
        or not math.isfinite(physics_hz)
        or physics_hz <= 0.0
        or tick < 0
        or channel_index < 0
        or channel_index >= 12
        or not isinstance(endpoint, Sequence)
        or isinstance(endpoint, (str, bytes))
        or len(endpoint) != 12
        or not isinstance(waypoints, Sequence)
        or isinstance(waypoints, (str, bytes))
        or not waypoints
    ):
        return None
    try:
        if tick >= round(active_duration * physics_hz):
            value = float(endpoint[channel_index])
        else:
            elapsed = tick / physics_hz
            selected: Mapping[str, Any] | None = None
            for waypoint in waypoints:
                if not isinstance(waypoint, Mapping):
                    return None
                if float(waypoint["time_s"]) <= elapsed + 1.0e-12:
                    selected = waypoint
                else:
                    break
            if selected is None:
                return None
            full12 = selected["full12"]
            if (
                not isinstance(full12, Sequence)
                or isinstance(full12, (str, bytes))
                or len(full12) != 12
            ):
                return None
            value = float(full12[channel_index])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _finite_sequence(
    value: Any, *, expected_length: int
) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != expected_length or any(
        not math.isfinite(item) for item in result
    ):
        return None
    return result


def _wheel_rebound_atomic_ack_valid(
    row: Mapping[str, Any],
    *,
    full12_order: tuple[str, ...],
    native: tuple[float, ...],
    final: tuple[float, ...],
    requested: tuple[float, ...],
    realized: tuple[float, ...],
) -> bool:
    """Bind the logical rebound ledger to the adapter's physical-sign ack."""

    ack = row.get("atomic_ack")
    if not isinstance(ack, Mapping):
        return False
    raw_ack_order = ack.get("canonical_order")
    if (
        not isinstance(raw_ack_order, Sequence)
        or isinstance(raw_ack_order, (str, bytes))
    ):
        return False
    ack_native = _vector(ack, "native_drive_target_full12")
    ack_final = _vector(ack, "drive_target_full12")
    ack_requested = _vector(ack, "drive_feedback_bias_requested_full12")
    ack_realized = _vector(ack, "drive_feedback_bias_realized_full12")
    source_requested = _vector(ack, "requested_full12")
    source_applied = _vector(ack, "applied_full12")
    physical_wheels = _finite_sequence(
        ack.get("wheel_target_physical_rad_s"), expected_length=4
    )
    if any(
        item is None
        for item in (
            ack_native,
            ack_final,
            ack_requested,
            ack_realized,
            source_requested,
            source_applied,
            physical_wheels,
        )
    ):
        return False
    assert ack_native is not None and ack_final is not None
    assert ack_requested is not None and ack_realized is not None
    assert source_requested is not None and source_applied is not None
    assert physical_wheels is not None
    expected_physical_wheels = tuple(
        sign * value
        for sign, value in zip((-1.0, 1.0, -1.0, 1.0), final[8:], strict=True)
    )
    return bool(
        ack.get("schema") == "wlr50_clean.atomic_full12_ack.v1"
        and tuple(str(item) for item in raw_ack_order) == full12_order
        and ack.get("articulation_writes_this_call") == 1
        and _number(ack, "motion_start_skew_s") == 0.0
        and ack.get("command_was_clamped") is False
        and all(
            abs(left - right) <= 1.0e-9
            for left, right in zip(ack_native, native, strict=True)
        )
        and all(
            abs(left - right) <= 1.0e-9
            for left, right in zip(ack_final, final, strict=True)
        )
        and all(
            abs(left - right) <= 1.0e-9
            for left, right in zip(ack_requested, requested, strict=True)
        )
        and all(
            abs(left - right) <= 1.0e-9
            for left, right in zip(ack_realized, realized, strict=True)
        )
        and all(
            abs(source_requested[index] - native[index]) <= 1.0e-9
            and abs(source_applied[index] - native[index]) <= 1.0e-9
            for index in range(8, 12)
        )
        and all(
            abs(left - right) <= 1.0e-9
            for left, right in zip(
                physical_wheels, expected_physical_wheels, strict=True
            )
        )
    )


def _drive_feedback_ledger_valid(
    commands: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Audit any post-mapper correction against its explicit final-drive log."""

    feedback_phases = {
        str(phase.get("state_id")): phase
        for phase in contract.get("phases", ())
        if isinstance(phase, Mapping) and isinstance(phase.get("drive_feedback"), Mapping)
    }
    feedback_specs = {
        phase_id: phase["drive_feedback"]
        for phase_id, phase in feedback_phases.items()
    }
    rows = sorted(
        (row for row in commands if isinstance(row.get("drive_feedback"), Mapping)),
        key=_time,
    )
    if not feedback_specs:
        return not rows
    if not commands or len(rows) != len(commands):
        return False
    raw_order = contract.get("full12_order")
    if (
        not isinstance(raw_order, Sequence)
        or isinstance(raw_order, (str, bytes))
        or len(raw_order) != 12
    ):
        return False
    full12_order = tuple(str(item) for item in raw_order)
    if len(set(full12_order)) != 12:
        return False
    try:
        physics_hz = float(contract["physics_hz"])
        maximum_delta = float(contract["servo_reference_velocity_deg_s"]) / float(
            physics_hz
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    if (
        not math.isfinite(physics_hz)
        or physics_hz <= 0.0
        or not math.isfinite(maximum_delta)
        or maximum_delta <= 0.0
    ):
        return False
    for phase_id, spec in feedback_specs.items():
        mode = _drive_feedback_mode(spec)
        if mode is None:
            return False
        if _is_wheel_feedback_mode(mode):
            values = _wheel_feedback_contract_values(
                spec, physics_hz=physics_hz, mode=mode
            )
            phase = feedback_phases[phase_id]
            try:
                correction_channel = str(spec["correction_channel"])
                correction_index = int(spec["correction_channel_index"])
                probe_channel = str(spec["probe_channel"])
                probe_index = int(spec["probe_channel_index"])
                reference_integral = float(spec["reference_wheel_integral_rad"])
                source_integral = float(
                    phase["command_metrics"]["wheel_integral_rad"][
                        correction_channel
                    ]
                )
                derived_source_integral = _phase_zoh_channel_integral(
                    phase, correction_index
                )
                active_duration = float(phase["active_duration_s"])
                endpoint = tuple(float(item) for item in phase["end_full12"])
                waypoints = tuple(phase["waypoints"])
                prior_waypoints = tuple(
                    waypoint
                    for waypoint in waypoints
                    if float(waypoint["time_s"]) < active_duration - 1.0e-12
                )
                prior_target = float(
                    prior_waypoints[-1]["full12"][correction_index]
                )
                source_peak = float(
                    phase["command_metrics"]["wheel_peak_abs_target_rad_s"][
                        correction_channel
                    ]
                )
                derived_source_peak = _phase_zoh_channel_peak_abs(
                    phase, correction_index
                )
                probe_signature = tuple(
                    (
                        int(item["motion_tick"]),
                        float(item["reference_actual_deg"]),
                    )
                    for item in spec["probe_samples"]
                )
                lag_threshold = float(spec["lag_threshold_deg"])
            except (KeyError, IndexError, TypeError, ValueError):
                return False
            if (
                values is None
                or phase_id != "P09"
                or correction_channel != _WHEEL_TAIL_CHANNEL
                or correction_index != _WHEEL_TAIL_CHANNEL_INDEX
                or probe_channel != _WHEEL_PROBE_CHANNEL
                or probe_index != _WHEEL_PROBE_CHANNEL_INDEX
                or not math.isfinite(source_integral)
                or abs(reference_integral - source_integral) > 1.0e-12
                or derived_source_integral is None
                or abs(reference_integral - derived_source_integral) > 1.0e-12
                or len(endpoint) != 12
                or abs(endpoint[correction_index]) > 1.0e-12
                or abs(prior_target - _WHEEL_TAIL_VELOCITY_RAD_S) > 1.0e-12
                or not math.isfinite(source_peak)
                or derived_source_peak is None
                or abs(source_peak - derived_source_peak) > 1.0e-12
                or (
                    mode == "wheel_rebound"
                    and (
                        probe_signature
                        != tuple(
                            zip(
                                _WHEEL_TAIL_PROBE_TICKS,
                                _WHEEL_PROBE_REFERENCES_DEG,
                                strict=True,
                            )
                        )
                        or abs(
                            lag_threshold - _WHEEL_REBOUND_LAG_THRESHOLD_DEG
                        )
                        > 1.0e-12
                        or spec.get("nominal_endpoint_restored") is not True
                        or spec.get("raw_recording_runtime_access_required")
                        is not False
                        or round(active_duration * physics_hz)
                        != _WHEEL_ENDPOINT_TICK
                        or abs(source_peak - _WHEEL_REFERENCE_PEAK_ABS_RAD_S)
                        > 1.0e-12
                    )
                )
            ):
                return False
    previous_final: tuple[float, ...] | None = None
    attempt_by_phase: dict[str, int] = {}
    previous_tick_by_phase: dict[str, int] = {}
    row_attempts: list[tuple[str, int, int | None]] = []
    logged_trigger_attempts: dict[str, tuple[int, int]] = {}
    row_index_by_attempt_tick: dict[tuple[str, int, int], int] = {}
    for row_index, row in enumerate(rows):
        phase_id = _phase(row)
        feedback = row["drive_feedback"]
        raw_tick = feedback.get("tick_index", row.get("motion_tick_index"))
        try:
            tick = None if raw_tick is None else int(raw_tick)
        except (TypeError, ValueError):
            return False
        attempt = attempt_by_phase.setdefault(phase_id, 0)
        if tick is not None:
            previous_tick = previous_tick_by_phase.get(phase_id)
            if tick == 0 and previous_tick is not None:
                attempt += 1
                attempt_by_phase[phase_id] = attempt
            previous_tick_by_phase[phase_id] = tick
            key = (phase_id, attempt, tick)
            if key in row_index_by_attempt_tick:
                return False
            row_index_by_attempt_tick[key] = row_index
        row_attempts.append((phase_id, attempt, tick))
        if feedback.get("just_triggered") is True:
            if phase_id in logged_trigger_attempts or tick is None:
                return False
            logged_trigger_attempts[phase_id] = (attempt, tick)

    observation_by_time: dict[float, Mapping[str, Any]] = {}
    if observations is not None:
        try:
            for observation in observations:
                observation_by_time[round(_time(observation), 12)] = observation
        except TrialAnalysisError:
            return False

    # Derive mandatory triggers from the physical probe evidence instead of
    # trusting the controller's just_triggered flag.  This keeps a missing
    # correction auditable even when every logged trigger field is suppressed.
    trigger_attempts: dict[str, tuple[int, int]] = {}
    for phase_id, spec in feedback_specs.items():
        try:
            probe_rows = tuple(spec["probe_samples"])
            probe_ticks = tuple(int(item["motion_tick"]) for item in probe_rows)
            required_samples = int(spec["required_consecutive_samples"])
            lag_threshold = float(spec["lag_threshold_deg"])
            probe_channel = str(spec["probe_channel"])
            probe_index = int(spec["probe_channel_index"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not probe_ticks
            or required_samples != len(probe_ticks)
            or not math.isfinite(lag_threshold)
            or lag_threshold <= 0.0
            or any(
                right != left + 1
                for left, right in zip(probe_ticks, probe_ticks[1:])
            )
            or probe_index < 0
            or probe_index >= len(full12_order)
            or full12_order[probe_index] != probe_channel
        ):
            return False
        attempts = sorted(
            {
                attempt
                for row_phase, attempt, tick in row_attempts
                if row_phase == phase_id
                and tick is not None
                and tick >= probe_ticks[-1]
            }
        )
        if not attempts:
            return False
        trigger_consumed = False
        for attempt in attempts:
            consecutive = 0
            for probe, probe_tick in zip(probe_rows, probe_ticks, strict=True):
                row_index = row_index_by_attempt_tick.get(
                    (phase_id, attempt, probe_tick)
                )
                if row_index is None:
                    return False
                row = rows[row_index]
                feedback = row["drive_feedback"]
                try:
                    expected_reference = float(probe["reference_actual_deg"])
                except (KeyError, TypeError, ValueError):
                    return False
                observed = _number(feedback, "observed_deg")
                logged_reference = _number(feedback, "reference_deg")
                if (
                    observed is None
                    or logged_reference is None
                    or abs(logged_reference - expected_reference) > 1.0e-9
                ):
                    return False
                if observations is not None:
                    observation = observation_by_time.get(round(_time(row), 12))
                    actual = (
                        None
                        if observation is None
                        else _vector(observation, "actual_full12")
                    )
                    if (
                        actual is None
                        or abs(actual[probe_index] - observed) > 1.0e-9
                    ):
                        return False
                if expected_reference - observed + 1.0e-12 >= lag_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
            if not trigger_consumed and consecutive >= required_samples:
                trigger_attempts[phase_id] = (attempt, probe_ticks[-1])
                trigger_consumed = True
    if logged_trigger_attempts != trigger_attempts:
        return False

    realized_traces: dict[str, list[float]] = {
        phase_id: [] for phase_id in feedback_specs
    }
    for row, (phase_id, attempt, tick) in zip(rows, row_attempts, strict=True):
        feedback = row["drive_feedback"]
        if feedback.get("schema") != "wlr50_clean.drive_feedback.v1":
            return False
        declared = _vector(feedback, "bias_full12")
        requested = _vector(row, "drive_feedback_bias_requested_full12")
        realized = _vector(row, "drive_feedback_bias_realized_full12")
        native = _vector(row, "native_drive_target_full12")
        final = _vector(row, "drive_target_full12")
        if any(item is None for item in (declared, requested, realized, native, final)):
            return False
        assert declared is not None and requested is not None
        assert realized is not None and native is not None and final is not None
        if any(abs(left - right) > 1.0e-9 for left, right in zip(declared, requested, strict=True)):
            return False
        if any(
            abs((actual - base) - correction) > 1.0e-8
            for actual, base, correction in zip(final, native, realized, strict=True)
        ):
            return False
        if any(
            abs(value) > WHEEL_VELOCITY_LIMIT_RAD_S + 1.0e-12
            for value in final[8:]
        ):
            return False
        spec = feedback_specs.get(phase_id)
        expected_requested = [0.0] * 12
        triggered = trigger_attempts.get(phase_id)
        expected_latched = False
        expected_active = False
        if spec is not None and tick is not None:
            wheel_segments: tuple[tuple[int, int, float], ...] | None = None
            try:
                mode = _drive_feedback_mode(spec)
                probe_channel = str(spec["probe_channel"])
                probe_channel_index = int(spec["probe_channel_index"])
                correction_channel = str(spec["correction_channel"])
                correction_channel_index = int(spec["correction_channel_index"])
                probe_ticks = tuple(int(item["motion_tick"]) for item in spec["probe_samples"])
                cumulative_fraction = float(spec["cumulative_fraction_of_reference"])
                peak_fraction = (
                    0.0
                    if _is_wheel_feedback_mode(mode)
                    else float(spec["peak_fraction_of_reference"])
                )
                if mode == "wheel_rebound":
                    wheel_segments = _wheel_feedback_segments(spec, mode)
                    if wheel_segments is None:
                        return False
                    first_bias_tick = wheel_segments[0][0]
                    last_bias_tick = wheel_segments[-1][1]
                    logical_bias = 0.0
                else:
                    first_bias_tick = int(spec["first_bias_tick"])
                    last_bias_tick = int(spec["last_bias_tick"])
                    logical_bias = float(
                        spec[
                            "logical_bias_rad_s"
                            if _is_wheel_feedback_mode(mode)
                            else "logical_bias_deg"
                        ]
                    )
            except (KeyError, TypeError, ValueError):
                return False
            if mode is None or any(
                not math.isfinite(value)
                for value in (logical_bias, peak_fraction, cumulative_fraction)
            ):
                return False
            if (
                correction_channel_index < 0
                or correction_channel_index >= len(full12_order)
                or full12_order[correction_channel_index] != correction_channel
                or feedback.get("probe_channel") != probe_channel
                or feedback.get("probe_channel_index") != probe_channel_index
                or feedback.get("correction_channel") != correction_channel
                or feedback.get("correction_channel_index")
                != correction_channel_index
            ):
                return False
            if mode == "servo" and correction_channel_index >= 8:
                return False
            if _is_wheel_feedback_mode(mode):
                shape = _wheel_feedback_shape(mode)
                if shape is None:
                    return False
                (
                    expected_probe_ticks,
                    expected_first_tick,
                    expected_last_tick,
                    expected_teardown_tick,
                    expected_bias,
                ) = shape
                if (
                    (
                        mode == "wheel_rebound"
                        and feedback.get("kind") != spec.get("kind")
                    )
                    or correction_channel != _WHEEL_TAIL_CHANNEL
                    or correction_channel_index != _WHEEL_TAIL_CHANNEL_INDEX
                    or probe_channel != _WHEEL_PROBE_CHANNEL
                    or probe_channel_index != _WHEEL_PROBE_CHANNEL_INDEX
                    or probe_ticks != expected_probe_ticks
                    or first_bias_tick != expected_first_tick
                    or last_bias_tick != expected_last_tick
                    or int(spec["teardown_tick"]) != expected_teardown_tick
                    or (
                        mode != "wheel_rebound"
                        and abs(logical_bias - expected_bias) > 1.0e-12
                    )
                    or abs(peak_fraction) > 1.0e-12
                ):
                    return False
                for field in (
                    "reference_wheel_integral_rad",
                    "additional_wheel_integral_rad",
                ):
                    logged = _number(feedback, field)
                    try:
                        expected = float(spec[field])
                    except (KeyError, TypeError, ValueError):
                        return False
                    if logged is None or abs(logged - expected) > 1.0e-12:
                        return False
                if mode == "wheel_rebound":
                    if (
                        wheel_segments is None
                        or _wheel_feedback_segments(feedback, mode)
                        != wheel_segments
                    ):
                        return False
                    for field in (
                        "resulting_wheel_integral_rad",
                        "reference_wheel_peak_abs_rad_s",
                        "resulting_wheel_peak_abs_rad_s",
                    ):
                        logged = _number(feedback, field)
                        try:
                            expected = float(spec[field])
                        except (KeyError, TypeError, ValueError):
                            return False
                        if logged is None or abs(logged - expected) > 1.0e-12:
                            return False
                    if (
                        feedback.get("instantaneous_direction_reversal") is not True
                        or spec.get("instantaneous_direction_reversal") is not True
                    ):
                        return False
                else:
                    logged_bias = _number(feedback, "logical_bias_rad_s")
                    if (
                        logged_bias is None
                        or abs(logged_bias - logical_bias) > 1.0e-12
                    ):
                        return False
            expected_segment_index: int | None = None
            if triggered is not None:
                trigger_attempt, trigger_tick = triggered
                if trigger_tick != probe_ticks[-1]:
                    return False
                expected_latched = attempt == trigger_attempt and tick is not None and tick >= trigger_tick
                if mode == "wheel_rebound":
                    assert wheel_segments is not None
                    if expected_latched:
                        expected_segment_index = next(
                            (
                                index
                                for index, (first, last, _) in enumerate(
                                    wheel_segments
                                )
                                if first <= tick <= last
                            ),
                            None,
                        )
                    expected_active = bool(
                        expected_latched and expected_segment_index is not None
                    )
                    logical_bias = (
                        wheel_segments[expected_segment_index][2]
                        if expected_active
                        and expected_segment_index is not None
                        else 0.0
                    )
                else:
                    expected_active = bool(
                        expected_latched
                        and first_bias_tick <= tick <= last_bias_tick
                    )
                if expected_active:
                    expected_requested[correction_channel_index] = logical_bias
            elif mode == "wheel_rebound":
                logical_bias = 0.0
            if mode == "wheel_rebound":
                logged_segment_index = feedback.get("active_segment_index")
                if expected_segment_index is None:
                    if logged_segment_index is not None:
                        return False
                    expected_segment_first = None
                    expected_segment_last = None
                else:
                    if (
                        isinstance(logged_segment_index, bool)
                        or not isinstance(logged_segment_index, int)
                        or logged_segment_index != expected_segment_index
                    ):
                        return False
                    assert wheel_segments is not None
                    expected_segment_first = wheel_segments[
                        expected_segment_index
                    ][0]
                    expected_segment_last = wheel_segments[
                        expected_segment_index
                    ][1]
                logged_logical_bias = _number(
                    feedback, "logical_bias_rad_s"
                )
                if (
                    logged_logical_bias is None
                    or abs(logged_logical_bias - logical_bias) > 1.0e-12
                    or feedback.get("active_segment_first_bias_tick")
                    != expected_segment_first
                    or feedback.get("active_segment_last_bias_tick")
                    != expected_segment_last
                ):
                    return False
            expected_peak = peak_fraction if expected_latched else 0.0
            expected_cumulative = cumulative_fraction if expected_latched else 0.0
            try:
                logged_peak = float(feedback.get("peak_fraction_of_reference"))
                logged_cumulative = float(feedback.get("cumulative_fraction_of_reference"))
            except (TypeError, ValueError):
                return False
            if (
                abs(logged_peak - expected_peak) > 1.0e-12
                or abs(logged_cumulative - expected_cumulative) > 1.0e-12
                or bool(feedback.get("active")) != expected_active
                or (feedback.get("trigger_tick") if expected_latched else None)
                != (trigger_tick if expected_latched else None)
            ):
                return False
            if feedback.get("just_triggered") is True and not (
                expected_latched and tick == trigger_tick
            ):
                return False
            if any(
                abs(value - expected) > 1.0e-9
                for value, expected in zip(requested, expected_requested, strict=True)
            ):
                return False
            if any(
                abs(value) > 1.0e-8
                for index, value in enumerate(realized)
                if index != correction_channel_index
            ):
                return False
            realized_traces[phase_id].append(realized[correction_channel_index])
            if (
                not expected_active
                and abs(realized[correction_channel_index]) > 1.0e-8
            ):
                return False
            if expected_active and abs(
                realized[correction_channel_index] - logical_bias
            ) > 1.0e-8:
                return False
            if mode == "wheel" and expected_active:
                if abs(
                    final[correction_channel_index]
                    - _WHEEL_TAIL_VELOCITY_RAD_S
                ) > 1.0e-8:
                    return False
            if mode == "wheel_rebound":
                contract_native = _phase_channel_at_motion_tick(
                    feedback_phases[phase_id],
                    correction_channel_index,
                    tick,
                    physics_hz=physics_hz,
                )
                expected_final = (
                    None
                    if contract_native is None
                    else contract_native
                    + (logical_bias if expected_active else 0.0)
                )
                pinned_native = None
                if _WHEEL_TAIL_PROBE_TICKS[0] <= tick < _WHEEL_ENDPOINT_TICK:
                    pinned_native = _WHEEL_TAIL_VELOCITY_RAD_S
                elif _WHEEL_ENDPOINT_TICK <= tick <= _WHEEL_REBOUND_LAST_TICK:
                    pinned_native = 0.0
                if (
                    contract_native is None
                    or expected_final is None
                    or (
                        pinned_native is not None
                        and abs(contract_native - pinned_native) > 1.0e-12
                    )
                    or abs(native[correction_channel_index] - contract_native)
                    > 1.0e-8
                    or abs(final[correction_channel_index] - expected_final)
                    > 1.0e-8
                    or not _wheel_rebound_atomic_ack_valid(
                        row,
                        full12_order=full12_order,
                        native=native,
                        final=final,
                        requested=requested,
                        realized=realized,
                    )
                ):
                    return False
        else:
            try:
                logged_peak = float(feedback.get("peak_fraction_of_reference"))
                logged_cumulative = float(
                    feedback.get("cumulative_fraction_of_reference")
                )
            except (TypeError, ValueError):
                return False
            if (
                any(abs(value) > 1.0e-12 for value in requested)
                or any(abs(value) > 1.0e-8 for value in realized)
                or abs(logged_peak) > 1.0e-12
                or abs(logged_cumulative) > 1.0e-12
                or feedback.get("active") is True
                or feedback.get("just_triggered") is True
                or feedback.get("trigger_tick") is not None
            ):
                return False
        if previous_final is not None and any(
            abs(current - previous) > maximum_delta + 1.0e-8
            for current, previous in zip(final[:8], previous_final[:8], strict=True)
        ):
            return False
        previous_final = final
    teardown_row_by_phase: dict[str, Mapping[str, Any]] = {}
    for phase_id, (trigger_attempt, trigger_tick) in trigger_attempts.items():
        spec = feedback_specs.get(phase_id)
        if spec is None:
            return False
        try:
            mode = _drive_feedback_mode(spec)
            probe_rows = tuple(spec["probe_samples"])
            probe_ticks = tuple(int(item["motion_tick"]) for item in probe_rows)
            if mode == "wheel_rebound":
                wheel_segments = _wheel_feedback_segments(spec, mode)
                if wheel_segments is None:
                    return False
                first_bias_tick = wheel_segments[0][0]
                last_bias_tick = wheel_segments[-1][1]
            else:
                first_bias_tick = int(spec["first_bias_tick"])
                last_bias_tick = int(spec["last_bias_tick"])
            teardown_tick = int(spec["teardown_tick"])
            lag_threshold = float(spec["lag_threshold_deg"])
            probe_index = int(spec["probe_channel_index"])
            correction_index = int(spec["correction_channel_index"])
        except (KeyError, TypeError, ValueError):
            return False
        if mode is None or not probe_ticks or trigger_tick != probe_ticks[-1]:
            return False
        required_ticks = tuple(range(probe_ticks[0], last_bias_tick + 1))
        required_indices = []
        for tick in required_ticks:
            row_index = row_index_by_attempt_tick.get(
                (phase_id, trigger_attempt, tick)
            )
            if row_index is None:
                return False
            required_indices.append(row_index)
        if any(
            right != left + 1
            for left, right in zip(required_indices, required_indices[1:])
        ):
            return False
        try:
            physics_dt_s = 1.0 / float(contract["physics_hz"])
            if any(
                abs(_time(rows[right]) - _time(rows[left]) - physics_dt_s)
                > 1.0e-9
                for left, right in zip(
                    required_indices, required_indices[1:]
                )
            ):
                return False
        except (KeyError, TypeError, ValueError, ZeroDivisionError, TrialAnalysisError):
            return False
        for probe in probe_rows:
            try:
                probe_tick = int(probe["motion_tick"])
                expected_reference = float(probe["reference_actual_deg"])
            except (KeyError, TypeError, ValueError):
                return False
            row_index = row_index_by_attempt_tick.get(
                (phase_id, trigger_attempt, probe_tick)
            )
            if row_index is None:
                return False
            row = rows[row_index]
            feedback = row["drive_feedback"]
            observed = _number(feedback, "observed_deg")
            logged_reference = _number(feedback, "reference_deg")
            if (
                observed is None
                or logged_reference is None
                or abs(logged_reference - expected_reference) > 1.0e-9
                or expected_reference - observed + 1.0e-12 < lag_threshold
            ):
                return False
            if observations is not None:
                observation = observation_by_time.get(round(_time(row), 12))
                actual = (
                    None
                    if observation is None
                    else _vector(observation, "actual_full12")
                )
                if (
                    actual is None
                    or abs(actual[probe_index] - observed) > 1.0e-9
                ):
                    return False
        expected_first_bias_tick = trigger_tick + (
            5 if mode == "wheel" else 1 if mode == "wheel_rebound" else 3
        )
        if first_bias_tick != expected_first_bias_tick:
            return False
        if _is_wheel_feedback_mode(mode):
            shape = _wheel_feedback_shape(mode)
            if shape is None:
                return False
            (
                expected_probe_ticks,
                expected_first_tick,
                expected_last_tick,
                expected_teardown_tick,
                _,
            ) = shape
            if (
                probe_ticks != expected_probe_ticks
                or trigger_tick != expected_probe_ticks[-1]
                or first_bias_tick != expected_first_tick
                or last_bias_tick != expected_last_tick
                or teardown_tick != expected_teardown_tick
                or correction_index != _WHEEL_TAIL_CHANNEL_INDEX
            ):
                return False
        last_bias_index = row_index_by_attempt_tick[
            (phase_id, trigger_attempt, last_bias_tick)
        ]
        teardown_index = row_index_by_attempt_tick.get(
            (phase_id, trigger_attempt, teardown_tick)
        )
        if teardown_index is None:
            teardown_index = last_bias_index + 1
        if teardown_index != last_bias_index + 1 or teardown_index >= len(rows):
            return False
        teardown_row = rows[teardown_index]
        teardown_phase, teardown_attempt, logged_teardown_tick = row_attempts[
            teardown_index
        ]
        if (
            teardown_phase == phase_id
            and teardown_attempt == trigger_attempt
            and logged_teardown_tick != teardown_tick
        ):
            return False
        try:
            if abs(
                _time(teardown_row)
                - _time(rows[last_bias_index])
                - physics_dt_s
            ) > 1.0e-9:
                return False
        except (KeyError, TypeError, ValueError, ZeroDivisionError, TrialAnalysisError):
            return False
        requested = _vector(
            teardown_row, "drive_feedback_bias_requested_full12"
        )
        realized = _vector(
            teardown_row, "drive_feedback_bias_realized_full12"
        )
        declared = _vector(teardown_row["drive_feedback"], "bias_full12")
        if (
            requested is None
            or realized is None
            or declared is None
            or abs(requested[correction_index]) > 1.0e-9
            or abs(realized[correction_index]) > 1.0e-8
            or abs(declared[correction_index]) > 1.0e-9
        ):
            return False
        if _is_wheel_feedback_mode(mode):
            final = _vector(teardown_row, "drive_target_full12")
            native = _vector(teardown_row, "native_drive_target_full12")
            if (
                final is None
                or native is None
                or abs(final[correction_index] - native[correction_index])
                > 1.0e-8
                or abs(final[correction_index]) > 1.0e-8
            ):
                return False
            if mode == "wheel_rebound":
                feedback = teardown_row["drive_feedback"]
                neutral_fields = (
                    "resulting_wheel_integral_rad",
                    "reference_wheel_peak_abs_rad_s",
                    "resulting_wheel_peak_abs_rad_s",
                )
                same_phase_teardown = (
                    teardown_phase == phase_id
                    and teardown_attempt == trigger_attempt
                )
                if not _wheel_rebound_atomic_ack_valid(
                    teardown_row,
                    full12_order=full12_order,
                    native=native,
                    final=final,
                    requested=requested,
                    realized=realized,
                ):
                    return False
                if not same_phase_teardown and (
                    feedback.get("kind") is not None
                    or feedback.get("bias_segments") not in (None, [], ())
                    or feedback.get("active_segment_index") is not None
                    or feedback.get("active_segment_first_bias_tick") is not None
                    or feedback.get("active_segment_last_bias_tick") is not None
                    or (
                        (value := _number(feedback, "logical_bias_rad_s"))
                        is None
                        or abs(value) > 1.0e-12
                    )
                    or any(
                        (value := _number(feedback, field)) is None
                        or abs(value) > 1.0e-12
                        for field in neutral_fields
                    )
                    or feedback.get("instantaneous_direction_reversal") is not False
                ):
                    return False
        teardown_row_by_phase[phase_id] = teardown_row
    for phase_id, spec in feedback_specs.items():
        trace = realized_traces[phase_id]
        if not trace:
            continue
        mode = _drive_feedback_mode(spec)
        if mode is None:
            return False
        triggered = trigger_attempts.get(phase_id)
        if triggered is not None and abs(trace[-1]) > 1.0e-8:
            correction_index = int(spec["correction_channel_index"])
            teardown = _vector(
                teardown_row_by_phase[phase_id],
                "drive_feedback_bias_realized_full12",
            )
            assert teardown is not None
            trace.append(teardown[correction_index])
        if _is_wheel_feedback_mode(mode):
            values = _wheel_feedback_contract_values(
                spec, physics_hz=physics_hz, mode=mode
            )
            if values is None:
                return False
            reference_integral, integral_budget = values
            realized_signed_integral = sum(trace) / physics_hz
            realized_absolute_integral_fraction = (
                sum(abs(value) for value in trace) / physics_hz / reference_integral
            )
            try:
                declared_signed_integral = float(
                    spec["additional_wheel_integral_rad"]
                )
            except (KeyError, TypeError, ValueError):
                return False
            if (
                (
                    abs(realized_signed_integral - declared_signed_integral)
                    > 1.0e-9
                    if triggered is not None
                    else abs(realized_signed_integral) > 1.0e-9
                )
                or (
                    abs(
                        realized_absolute_integral_fraction - integral_budget
                    )
                    > 1.0e-9
                    if triggered is not None
                    else realized_absolute_integral_fraction > 1.0e-9
                )
                or realized_absolute_integral_fraction
                > _MAX_FEEDBACK_FRACTION + 1.0e-9
            ):
                return False
        else:
            try:
                excursion = abs(float(spec["reference_excursion_deg"]))
                peak_budget = float(spec["peak_fraction_of_reference"])
                cumulative_budget = float(
                    spec["cumulative_fraction_of_reference"]
                )
            except (KeyError, TypeError, ValueError):
                return False
            if excursion <= 0.0:
                return False
            realized_peak = max(abs(value) for value in trace) / excursion
            realized_cumulative = sum(
                abs(right - left) for left, right in zip(trace, trace[1:])
            ) / excursion
            if (
                realized_peak > peak_budget + 1.0e-9
                or realized_cumulative > cumulative_budget + 1.0e-9
            ):
                return False
    return True


def analyze_trial(
    run_dir: Path, contract_path: Path, *, strict_success: bool = True
) -> dict[str, Any]:
    """Analyze one run without mutation; optionally enforce final-success gates."""

    root = Path(run_dir).resolve()
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    ledgers = {name: _read_jsonl(root / filename) for name, filename in JSONL_FILES.items()}
    observations, commands = ledgers["observation"], ledgers["command"]
    transitions, task_rows = ledgers["transition"], ledgers["task_event"]
    windows = _phase_windows(transitions)
    similarity = _similarity_rows(contract, observations, commands, windows)
    leg_events, wheel_audit = _leg_audit(observations, ledgers["leg_crossing"])
    body_audit = _body_audit(observations, ledgers["body_contact"])
    summary = _summary(similarity)

    expected_atomic = int(contract.get("source_full12_atomic_event_count", 0))
    atomic_rows = [
        row for row in commands
        if bool(row.get("atomic_source_event", row.get("source_full12_atomic", False)))
    ]
    nominal_atomic = sum(
        _phase(row) in windows
        and windows[_phase(row)]["motion_start_s"] - 1.0e-9
        <= _time(row) <= windows[_phase(row)]["motion_end_s"] + 1.0e-9
        for row in atomic_rows
    )
    summary.update(
        source_full12_atomic_events_expected=expected_atomic,
        source_full12_atomic_events_observed=nominal_atomic,
        source_full12_atomic_events_total_including_recovery=len(atomic_rows),
    )
    correction_values, retry_counts = _recovery_evidence(transitions, commands)
    summary.update(
        recovery_count=sum(retry_counts.values()),
        maximum_feedback_correction_fraction=max(correction_values, default=0.0),
    )
    rear_crossings = {
        row["leg"]: float(row["simulation_time_s"])
        for row in leg_events
        if row["event"] == "FRONT_FACE_CROSSED" and row["leg"] in {"RR", "RL"}
    }
    ordered_commands, ordered_observations = sorted(commands, key=_time), sorted(observations, key=_time)
    final_command = _vector(
        ordered_commands[-1], "full12", "command_full12", "commanded_full12"
    ) if ordered_commands else None
    wheel_decay_threshold = _reference_wheel_decay_threshold(contract)
    wheel_decay = WheelDecayDebounce()
    wheel_decay_status = WheelDecayStatus(False, False, math.inf, 0.0, None)
    p13_decay_rows = _samples(
        ordered_observations,
        "P13",
        windows["P13"]["motion_end_s"],
        windows["P13"]["completion_time_s"],
    )
    for row in p13_decay_rows:
        measured = _vector(row, "actual_full12")
        commanded = _vector(row, "commanded_full12", "command_full12", "full12")
        if measured is None or commanded is None:
            wheel_decay.reset()
            wheel_decay_status = WheelDecayStatus(False, False, math.inf, 0.0, None)
            continue
        wheel_decay_status = wheel_decay.update(
            sim_time_s=_time(row),
            measured_velocity_rad_s=measured[8:],
            commanded_velocity_rad_s=commanded[8:],
            threshold_rad_s=wheel_decay_threshold,
            debounce_s=0.5,
        )
    stable_span = wheel_decay_status.stable_for_s
    summary.update(
        measured_wheel_velocity_decay_threshold_rad_s=wheel_decay_threshold,
        measured_wheel_velocity_stable_span_s=stable_span,
    )

    checks = {
        "task_result_success": _terminal_success(task_rows),
        "p01_p13_completed": tuple(windows) == PHASE_IDS,
        "body_collision_false": not any(bool(row["body_collision"]) for row in body_audit),
        "wheel_only_climb_false": not any(bool(row["wheel_only_climb"]) for row in wheel_audit),
        "all_four_active_lifts_and_crossings": len(leg_events) == 12,
        "rear_order_rr_first": set(rear_crossings) == {"RR", "RL"}
        and rear_crossings["RR"] < rear_crossings["RL"],
        "conformance_rows_populated": bool(similarity)
        and set(summary["phase_coverage"]) == set(PHASE_IDS),
        "all_normal_states_within_15_percent": summary["all_normal_states_within_15_percent"],
        "every_command_is_one_complete_full12": bool(commands) and all(
            _vector(row, "full12", "command_full12", "commanded_full12") is not None for row in commands
        ),
        "source_full12_atomic_events_exact": nominal_atomic == expected_atomic,
        "feedback_correction_reference_bounded": all(value <= 0.15 + 1.0e-12 for value in correction_values)
        and all(count <= 1 for count in retry_counts.values()),
        "drive_feedback_ledger_valid": _drive_feedback_ledger_valid(
            commands, contract, observations
        ),
        "final_wheel_targets_zero": final_command is not None
        and all(abs(value) <= 1.0e-9 for value in final_command[8:]),
        "measured_wheel_velocity_stable_decay": wheel_decay_status.passed,
        "duration_below_200_s": windows["P13"]["completion_time_s"]
        - windows["P01"]["entry_time_s"] <= 200.0,
    }
    if strict_success and (failed := [name for name, passed in checks.items() if not passed]):
        raise TrialAnalysisError(f"run is not publishable: failed checks {failed}")
    return {
        "schema": "wlr50_clean.trial_analysis.v1", "run_dir": str(root), "checks": checks,
        "phase_windows": windows, "similarity_rows": similarity,
        "conformance_summary": summary, "leg_crossing_events": leg_events,
        "body_collision_audit": body_audit, "wheel_only_climb_audit": wheel_audit,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in columns} for row in rows)
    os.replace(temporary, path)


def populate_reference_similarity(run_dir: Path, contract_path: Path) -> dict[str, Any]:
    """Atomically populate the run CSV before its immutable manifest exists."""

    root = Path(run_dir).resolve()
    if (root / "trial_manifest.json").exists():
        raise TrialAnalysisError("cannot modify a run after trial_manifest.json exists")
    analysis = analyze_trial(root, contract_path, strict_success=False)
    _write_csv(root / "reference_similarity.csv", analysis["similarity_rows"], SIMILARITY_COLUMNS)
    return analysis


def _state_rows(contract: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, phase in enumerate(contract["phases"]):
        phase_id, clock = str(phase["state_id"]), analysis["phase_windows"][phase["state_id"]]
        rows.append({
            "phase": phase_id, "macro_phase": phase.get("macro_phase", index + 1),
            "state": phase.get("state_name", phase_id), "physical_purpose": phase.get("physical_purpose", ""),
            "reference_steps": ";".join(str(item) for item in phase.get("reference_steps", ())),
            "active_channels": ";".join(str(item) for item in phase.get("active_channels", ())),
            "reference_active_duration_s": phase.get("active_duration_s", ""),
            "fsm_active_duration_s": clock["active_duration_s"], "entry_time_s": clock["entry_time_s"],
            "motion_start_s": clock["motion_start_s"], "motion_end_s": clock["motion_end_s"],
            "completion_time_s": clock["completion_time_s"], "completion_event": phase.get("completion_event", ""),
            "next_state": PHASE_IDS[index + 1] if index + 1 < len(PHASE_IDS) else "SUCCESS",
        })
    return rows


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    if _file_record(source)["sha256"] != _file_record(temporary)["sha256"]:
        temporary.unlink(missing_ok=True)
        raise TrialAnalysisError(f"copy verification failed for {source.name}")
    os.replace(temporary, destination)


def publish_successful_trial(
    *, run_dir: Path, output_dir: Path, contract_path: Path,
    selected_reference_path: Path, state_derivation_path: Path,
) -> dict[str, Any]:
    """Publish final data only after every physical and conformance gate passes."""

    root, manifest_path = Path(run_dir).resolve(), Path(run_dir).resolve() / "trial_manifest.json"
    if not manifest_path.is_file():
        raise TrialAnalysisError("trial_manifest.json is required for final publication")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = manifest.get("success_evidence", manifest)
    required = {
        "task_result_success": evidence.get("task_result", evidence.get("result")) == "SUCCESS",
        "continuous_physical_run": evidence.get("one_continuous_physical_fsm_success") is True,
        "body_collision_false": evidence.get("body_collision") is False,
        "wheel_only_climb_false": evidence.get("wheel_only_climb") is False,
        "rear_order_rr_first": evidence.get("rear_leg_order", evidence.get("rear_order")) == "RR_FIRST",
        "no_root_write": evidence.get("root_state_write_count") == 0,
        "no_teleport": evidence.get("teleport_count") == 0,
        "no_force_impulse": evidence.get("external_force_count") == 0
        and evidence.get("external_impulse_count") == 0,
        "runtime_raw_recording_access_false": evidence.get("runtime_raw_recording_access") is False,
    }
    phases = evidence.get("completed_macro_phases", evidence.get("completed_phases"))
    required["p01_p13_completed"] = tuple(phases or ()) == PHASE_IDS
    if failed := [name for name, passed in required.items() if not passed]:
        raise TrialAnalysisError(f"successful manifest failed checks {failed}")
    analysis = analyze_trial(root, contract_path, strict_success=True)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_copy(Path(selected_reference_path), destination / "selected_reference.json")
    _atomic_copy(Path(contract_path), destination / "recording_motion_contract.json")
    _atomic_copy(Path(state_derivation_path), destination / "state_derivation.md")
    _write_csv(destination / "fsm_vs_recording_similarity.csv", analysis["similarity_rows"], SIMILARITY_COLUMNS)
    state_rows = _state_rows(contract, analysis)
    _write_csv(destination / "fsm_state_table.csv", state_rows, tuple(state_rows[0]))
    for name, key in (
        ("leg_crossing_events.csv", "leg_crossing_events"),
        ("body_collision_audit.csv", "body_collision_audit"),
        ("wheel_only_climb_audit.csv", "wheel_only_climb_audit"),
    ):
        rows = analysis[key]
        _write_csv(destination / name, rows, tuple(rows[0]))
    published = dict(manifest)
    published["publication"] = {
        "schema": "wlr50_clean.final_data_publication.v1",
        "source_trial_manifest": _file_record(manifest_path),
        "checks": {**required, **analysis["checks"]},
        "conformance_summary": analysis["conformance_summary"],
        "published_files": {
            name: _file_record(destination / name)
            for name in FINAL_DATA_NAMES if name != "successful_trial_manifest.json"
        },
    }
    temporary = destination / "successful_trial_manifest.json.tmp"
    temporary.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "successful_trial_manifest.json")
    return {
        "status": "PASS", "output_dir": str(destination), "checks": analysis["checks"],
        "conformance_summary": analysis["conformance_summary"],
        "files": {name: _file_record(destination / name) for name in FINAL_DATA_NAMES},
    }


def first_blocker(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / JSONL_FILES["task_event"]
    if not path.is_file():
        return {"kind": "missing_task_events", "path": str(path)}
    for event in _read_jsonl(path):
        if event.get("event") in {"FIRST_BLOCKER", "STATE_DEADLOCK"}:
            return event
        if event.get("result") not in (None, "SUCCESS"):
            return event
    return None


def validate_trial_artifacts(run_dir: Path) -> dict[str, Any]:
    root = Path(run_dir)
    missing = [name for name in REQUIRED_TRIAL_FILES if not (root / name).is_file()]
    empty = [name for name in REQUIRED_TRIAL_FILES if (root / name).is_file() and (root / name).stat().st_size == 0]
    return {
        "schema": "wlr50_clean.trial_artifact_validation.v1", "run_dir": str(root.resolve()),
        "passed": not missing and not empty, "missing": missing, "empty": empty,
        "first_blocker": first_blocker(root),
    }
