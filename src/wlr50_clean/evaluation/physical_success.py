"""Read-only, three-layer adjudication of immutable physical FSM trials.

This module deliberately keeps physical task success independent from Recording
similarity.  Layer A establishes that an artifact is an admissible simulation
run, Layer B decides the physical traversal, and Layer C reports reference
divergence without vetoing either of the first two layers.

The evaluator never writes below ``runs/``.  Its only writes are replace-based
reports in the explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wlr50_clean.conformance_policy import load_conformance_policy
from wlr50_clean.infrastructure.video_capture import validate_mp4

from .comparison import PHASE_IDS
from .trial_analyzer import analyze_trial


SCHEMA = "wlr50_clean.physical_success_readjudication.v1"
TRIAL_VALIDITY = "TRIAL_VALIDITY"
TASK_SUCCESS = "TASK_SUCCESS"
QUALITY_DIAGNOSTICS = "QUALITY_AND_REFERENCE_DIAGNOSTICS"
EXPECTED_LEG_EVENTS = ("ACTIVE_LIFT", "FRONT_FACE_CROSSED", "TOP_LOADED")
LEGS = ("FL", "FR", "RR", "RL")
EXPLICIT_DEEP_AUDIT_TRIALS = frozenset({25, 36, 39, 43, 44})
TOP_GAP_MIN_M = -0.015
TOP_GAP_MAX_M = 0.025
STABLE_DECAY_S = 0.5
EXPECTED_GRAVITY_M_S2 = (0.0, 0.0, -9.81)
LEDGER_TIME_TOLERANCE_S = 1.0e-10

RUNTIME_SOURCE_ROOTS = (
    "src/wlr50_clean/fsm",
    "src/wlr50_clean/sensing",
    "src/wlr50_clean/infrastructure",
)

# World-gravity setup is allowed exactly once through SimulationCfg using the
# frozen constant.  These patterns cover post-creation/runtime mutation routes
# instead of flagging ordinary projected-gravity sensing or the obstacle's
# intentional ``disable_gravity=True`` rigid-body setting.
GRAVITY_OVERRIDE_PATTERNS = (
    ("gravity setter API", re.compile(r"\bset_(?:world_)?gravity\s*\(", re.I)),
    (
        "USD gravity attribute authoring",
        re.compile(
            r"(?:Create|Get)Gravity(?:Direction|Magnitude)Attr\s*\([^\n]*\)"
            r"\s*\.Set\s*\(",
            re.I,
        ),
    ),
    (
        "gravity attribute token",
        re.compile(r"(?:gravityDirection|gravityMagnitude|physxScene:gravity)", re.I),
    ),
    ("gravity override symbol", re.compile(r"\bgravity_override\b", re.I)),
)

APPLICABLE_SERVO_METRICS = (
    "command_endpoint_error_percent",
    "actual_endpoint_error_percent",
    "command_delta_error_percent",
    "actual_delta_error_percent",
    "duration_error_percent",
    "command_average_velocity_error_percent",
    "measured_average_velocity_error_percent",
    "command_peak_velocity_error_percent",
    "measured_peak_velocity_error_percent",
    "trajectory_rmse_percent",
)
APPLICABLE_WHEEL_METRICS = (
    "duration_error_percent",
    "command_average_velocity_error_percent",
    "measured_average_velocity_error_percent",
    "command_peak_velocity_error_percent",
    "measured_peak_velocity_error_percent",
    "command_wheel_integral_error_percent",
    "actual_wheel_integral_error_percent",
)

ALL_TRIAL_COLUMNS = (
    "trial_id",
    "trial_number",
    "deep_raw_audit",
    "trial_validity",
    "environment_match",
    "environment_hash",
    "robot_asset_hash",
    "continuous_physics_run",
    "physical_ledgers_valid",
    "completed_states",
    "P01_P13_complete",
    "physical_traversal_complete",
    "final_obstacle_geometry_success",
    "final_body_position",
    "final_wheel_positions",
    "four_leg_active_lift_evidence",
    "rear_leg_order",
    "body_collision",
    "body_collision_evidence",
    "wheel_only_climb",
    "wheel_only_climb_evidence",
    "fall",
    "physics_explosion",
    "final_pose_stable",
    "stable_decay_span_s",
    "recovery_count",
    "video_path",
    "video_decode",
    "video_timestamps_monotonic",
    "video_duration_from_frames",
    "video_continuous",
    "recording_runtime_access",
    "root_write_count",
    "teleport_count",
    "external_force_count",
    "external_impulse_count",
    "gravity_override_count",
    "gravity_override_absent",
    "gravity_evidence_source",
    "forbidden_control_count",
    "original_result",
    "task_result",
    "classification",
    "reference_max_error_percent",
    "reference_max_phase",
    "reference_max_channel",
    "reference_max_metric",
    "reference_within_30_percent",
    "reference_warning",
    "duration_s",
    "selected",
    "selection_reason",
)


class PhysicalSuccessError(RuntimeError):
    """The immutable evidence could not be adjudicated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _gravity_override_absence_proof(
    *,
    project_root: Path,
    environment_lock_path: Path,
    environment_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a static no-gravity-mutation proof for legacy manifests.

    Older immutable trials predate a dedicated ``gravity_override_count``.
    Their missing counter is not silently treated as zero: the fallback is a
    hash-bound audit of the frozen gravity configuration and every production
    FSM/sensing/infrastructure Python source file.
    """

    root = Path(project_root).resolve()
    lock_path = Path(environment_lock_path).resolve()
    try:
        configured_gravity = tuple(
            float(value) for value in environment_lock["physics"]["gravity_m_s2"]
        )
    except (KeyError, TypeError, ValueError):
        configured_gravity = ()
    configuration_matches = configured_gravity == EXPECTED_GRAVITY_M_S2

    source_paths = sorted(
        path
        for relative_root in RUNTIME_SOURCE_ROOTS
        for path in (root / relative_root).rglob("*.py")
        if path.is_file()
    )
    source_records = [_file_record(path, root=root) for path in source_paths]
    findings: list[dict[str, Any]] = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        for label, pattern in GRAVITY_OVERRIDE_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "kind": label,
                    }
                )

    scene_path = root / "src/wlr50_clean/infrastructure/scene_factory.py"
    scene_text = scene_path.read_text(encoding="utf-8") if scene_path.is_file() else ""
    constant_match = re.search(
        r"^\s*GRAVITY_M_S2\s*=\s*\(([^)]*)\)", scene_text, re.MULTILINE
    )
    try:
        scene_gravity = tuple(
            float(value.strip()) for value in constant_match.group(1).split(",")
        ) if constant_match is not None else ()
    except ValueError:
        scene_gravity = ()
    simulation_cfg_binding_count = len(
        re.findall(r"\bgravity\s*=\s*GRAVITY_M_S2\b", scene_text)
    )
    scene_binding_valid = bool(
        scene_gravity == EXPECTED_GRAVITY_M_S2
        and simulation_cfg_binding_count == 1
    )
    configuration_record = _file_record(lock_path, root=root)
    source_set_sha256 = _canonical_sha256(source_records)
    return {
        "schema": "wlr50_clean.gravity_override_absence_proof.v1",
        "passed": bool(
            configuration_matches
            and scene_binding_valid
            and source_records
            and not findings
        ),
        "evidence_role": "fallback_for_historical_trial_without_gravity_counter",
        "configuration": {
            **configuration_record,
            "gravity_m_s2": list(configured_gravity),
            "expected_gravity_m_s2": list(EXPECTED_GRAVITY_M_S2),
            "matches_expected": configuration_matches,
        },
        "scene_binding": {
            "path": scene_path.relative_to(root).as_posix(),
            "bytes": scene_path.stat().st_size if scene_path.is_file() else None,
            "sha256": _sha256(scene_path) if scene_path.is_file() else None,
            "gravity_constant_m_s2": list(scene_gravity),
            "simulation_cfg_binding_count": simulation_cfg_binding_count,
            "valid": scene_binding_valid,
        },
        "runtime_source_file_count": len(source_records),
        "runtime_source_set_sha256": source_set_sha256,
        "runtime_source_files": source_records,
        "forbidden_mutation_findings": findings,
    }


def _gravity_proof_reference(proof: Mapping[str, Any] | None) -> dict[str, Any]:
    selected = proof if isinstance(proof, Mapping) else {}
    configuration = selected.get("configuration", {})
    configuration = configuration if isinstance(configuration, Mapping) else {}
    return {
        "schema": selected.get("schema"),
        "passed": selected.get("passed") is True,
        "configuration_path": configuration.get("path"),
        "configuration_sha256": configuration.get("sha256"),
        "runtime_source_file_count": selected.get("runtime_source_file_count"),
        "runtime_source_set_sha256": selected.get("runtime_source_set_sha256"),
        "forbidden_mutation_findings": selected.get(
            "forbidden_mutation_findings", []
        ),
    }


def _trial_number(path: Path) -> int:
    try:
        return int(path.name.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise PhysicalSuccessError(f"invalid trial directory name: {path.name}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PhysicalSuccessError(f"{path}: JSON root is not an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        (
            json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    )
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {name: _csv_value(row.get(name)) for name in columns} for row in rows
        )
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    digest = hashlib.sha256()
    if not path.is_file():
        return {
            "rows": rows,
            "errors": ["missing"],
            "sha256": None,
            "bytes": 0,
        }
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                errors.append(f"line {line_number}: blank")
                continue
            try:
                item = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(item, dict):
                errors.append(f"line {line_number}: row is not an object")
                continue
            rows.append(item)
    return {
        "rows": rows,
        "errors": errors,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


_OBS_TICK = re.compile(rb'"physics_tick"\s*:\s*(\d+)')
_OBS_TIME = re.compile(rb'"simulation_time_s"\s*:\s*([-+0-9.eE]+)')
_OBS_DT = re.compile(rb'"physics_dt_s"\s*:\s*([-+0-9.eE]+)')
_COMMAND_TICK = re.compile(rb'"control_physics_tick"\s*:\s*(\d+)')
_COMMAND_TIME = re.compile(rb'"sim_time_s"\s*:\s*([-+0-9.eE]+)')
_FULL12 = re.compile(rb'"full12"\s*:\s*\[([^\]]*)\]')
_ALL_FINITE_FALSE = re.compile(rb'"all_finite"\s*:\s*false')
_EXPLOSION_TRUE = re.compile(
    rb'"physics_explosion_or_fall"\s*:\s*\{\s*"passed"\s*:\s*true'
)
_ONE_WRITE = re.compile(rb'"articulation_writes_this_call"\s*:\s*1(?:\D|$)')


def _number_match(pattern: re.Pattern[bytes], raw: bytes) -> float | None:
    match = pattern.search(raw)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tick_time_bytes(tick: int, time_s: float) -> bytes:
    # float.hex() is a stable representation of the exact parsed IEEE-754
    # value, so equality of these hashes proves more than rounded timestamps.
    return f"{tick}:{time_s.hex()}\n".encode("ascii")


def _scan_large_ledger(
    path: Path,
    *,
    kind: str,
    physics_dt_s: float,
    retain_tail: int = 0,
) -> dict[str, Any]:
    """Hash and structurally scan a large JSONL without materializing it."""

    if kind not in {"observation", "command"}:
        raise ValueError(kind)
    if not path.is_file():
        return {
            "valid": False,
            "errors": ["missing"],
            "row_count": 0,
            "sha256": None,
            "bytes": 0,
            "first": None,
            "last": None,
            "tail": [],
        }
    tick_pattern = _OBS_TICK if kind == "observation" else _COMMAND_TICK
    time_pattern = _OBS_TIME if kind == "observation" else _COMMAND_TIME
    digest = hashlib.sha256()
    errors: list[str] = []
    previous_tick: int | None = None
    previous_time: float | None = None
    first_raw: bytes | None = None
    last_raw: bytes | None = None
    tail: deque[bytes] = deque(maxlen=max(0, int(retain_tail)))
    row_count = 0
    non_finite_count = 0
    explosion_or_fall_count = 0
    vector_error_count = 0
    one_write_error_count = 0
    tick_time_digest = hashlib.sha256()
    prefix_tick_time_digest = hashlib.sha256()
    previous_tick_time_bytes: bytes | None = None
    tick_time_pair_count = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            stripped = raw.strip()
            if not stripped:
                errors.append(f"line {line_number}: blank")
                continue
            row_count += 1
            first_raw = stripped if first_raw is None else first_raw
            last_raw = stripped
            if tail.maxlen:
                tail.append(stripped)
            tick_value = _number_match(tick_pattern, stripped)
            time_value = _number_match(time_pattern, stripped)
            if tick_value is None or int(tick_value) != tick_value:
                errors.append(f"line {line_number}: missing/inexact tick")
            else:
                tick = int(tick_value)
                if previous_tick is not None and tick != previous_tick + 1:
                    errors.append(
                        f"line {line_number}: tick discontinuity {previous_tick}->{tick}"
                    )
                previous_tick = tick
            if time_value is None:
                errors.append(f"line {line_number}: missing/non-finite time")
            elif previous_time is not None and not math.isclose(
                time_value - previous_time,
                physics_dt_s,
                rel_tol=0.0,
                abs_tol=LEDGER_TIME_TOLERANCE_S,
            ):
                errors.append(
                    f"line {line_number}: time discontinuity {previous_time}->{time_value}"
                )
            if tick_value is not None and int(tick_value) == tick_value and time_value is not None:
                encoded_pair = _tick_time_bytes(int(tick_value), time_value)
                tick_time_digest.update(encoded_pair)
                if previous_tick_time_bytes is not None:
                    prefix_tick_time_digest.update(previous_tick_time_bytes)
                previous_tick_time_bytes = encoded_pair
                tick_time_pair_count += 1
            previous_time = time_value if time_value is not None else previous_time
            if not (stripped.startswith(b"{") and stripped.endswith(b"}")):
                errors.append(f"line {line_number}: malformed JSON-object framing")
            if kind == "observation":
                non_finite_count += bool(_ALL_FINITE_FALSE.search(stripped))
                explosion_or_fall_count += bool(_EXPLOSION_TRUE.search(stripped))
                dt_value = _number_match(_OBS_DT, stripped)
                if dt_value is None:
                    errors.append(f"line {line_number}: physics_dt_s missing/non-finite")
                elif not math.isclose(
                    dt_value,
                    physics_dt_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    errors.append(
                        f"line {line_number}: physics_dt_s {dt_value} != {physics_dt_s}"
                    )
            else:
                match = _FULL12.search(stripped)
                if match is None:
                    vector_error_count += 1
                else:
                    try:
                        values = tuple(float(item) for item in match.group(1).split(b","))
                    except ValueError:
                        values = ()
                    if len(values) != 12 or not all(math.isfinite(item) for item in values):
                        vector_error_count += 1
                if _ONE_WRITE.search(stripped) is None:
                    one_write_error_count += 1
            # A corrupt log should not create an unbounded report.
            if len(errors) > 100:
                errors = errors[:100] + ["additional structural errors omitted"]
                break

    def decode_edge(raw: bytes | None, label: str) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None
        if not isinstance(value, dict):
            errors.append(f"{label}: not an object")
            return None
        return value

    tail_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(tail):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"tail[{index}]: {type(exc).__name__}: {exc}")
            continue
        if isinstance(value, dict):
            tail_rows.append(value)
    if vector_error_count:
        errors.append(f"invalid full12 rows: {vector_error_count}")
    if one_write_error_count:
        errors.append(f"non-atomic command rows: {one_write_error_count}")
    return {
        "valid": bool(row_count) and not errors,
        "errors": errors,
        "row_count": row_count,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "first": decode_edge(first_raw, "first row"),
        "last": decode_edge(last_raw, "last row"),
        "tail": tail_rows,
        "first_tick": (
            int(_number_match(tick_pattern, first_raw or b"") or 0)
            if first_raw is not None
            else None
        ),
        "last_tick": previous_tick,
        "first_time_s": (
            _number_match(time_pattern, first_raw or b"") if first_raw is not None else None
        ),
        "last_time_s": previous_time,
        "tick_time_pair_count": tick_time_pair_count,
        "tick_time_sha256": (
            tick_time_digest.hexdigest()
            if tick_time_pair_count == row_count
            else None
        ),
        "prefix_tick_time_pair_count": max(0, tick_time_pair_count - 1),
        "prefix_tick_time_sha256": (
            prefix_tick_time_digest.hexdigest()
            if tick_time_pair_count == row_count and row_count > 0
            else None
        ),
        "non_finite_count": non_finite_count,
        "physics_explosion_or_fall_count": explosion_or_fall_count,
        "invalid_vector_count": vector_error_count,
        "non_atomic_write_count": one_write_error_count,
    }


def _ledger_continuity_evidence(
    observation: Mapping[str, Any],
    command: Mapping[str, Any],
    *,
    physics_dt_s: float,
) -> dict[str, Any]:
    """Prove the N-command/N+1-observation 120 Hz transaction sequence."""

    observation_count = int(observation.get("row_count", 0) or 0)
    command_count = int(command.get("row_count", 0) or 0)
    observation_first_tick = observation.get("first_tick")
    command_first_tick = command.get("first_tick")
    observation_last_tick = observation.get("last_tick")
    command_last_tick = command.get("last_tick")
    observation_first_time = observation.get("first_time_s")
    command_first_time = command.get("first_time_s")
    observation_last_time = observation.get("last_time_s")
    command_last_time = command.get("last_time_s")

    checks = {
        "physics_hz_is_120": math.isclose(
            physics_dt_s, 1.0 / 120.0, rel_tol=0.0, abs_tol=1.0e-15
        ),
        "individual_ledgers_valid": bool(
            observation.get("valid") and command.get("valid")
        ),
        "commands_n_observations_n_plus_1": (
            command_count > 0 and observation_count == command_count + 1
        ),
        "both_start_at_tick_zero": (
            observation_first_tick == 0 and command_first_tick == 0
        ),
        "command_tick_time_matches_observation_row": bool(
            command.get("tick_time_sha256")
            and command.get("tick_time_sha256")
            == observation.get("prefix_tick_time_sha256")
            and int(command.get("tick_time_pair_count", -1)) == command_count
            and int(observation.get("prefix_tick_time_pair_count", -1))
            == command_count
        ),
        "first_command_time_matches_first_observation": (
            command_first_time is not None
            and observation_first_time is not None
            and float(command_first_time) == float(observation_first_time)
        ),
        "terminal_observation_is_exact_next_tick": (
            command_last_tick is not None
            and observation_last_tick == int(command_last_tick) + 1
        ),
        "terminal_observation_is_one_120hz_step_later": (
            command_last_time is not None
            and observation_last_time is not None
            and math.isclose(
                float(observation_last_time) - float(command_last_time),
                physics_dt_s,
                rel_tol=0.0,
                abs_tol=LEDGER_TIME_TOLERANCE_S,
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "physics_dt_s": physics_dt_s,
        "physics_hz": 1.0 / physics_dt_s,
        "command_row_count": command_count,
        "observation_row_count": observation_count,
        "first_tick": command_first_tick,
        "last_command_tick": command_last_tick,
        "terminal_observation_tick": observation_last_tick,
        "first_time_s": command_first_time,
        "last_command_time_s": command_last_time,
        "terminal_observation_time_s": observation_last_time,
        "command_tick_time_sha256": command.get("tick_time_sha256"),
        "observation_command_prefix_tick_time_sha256": observation.get(
            "prefix_tick_time_sha256"
        ),
    }


def _expected_artifact(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    artifacts = manifest.get("artifact_files", {})
    value = artifacts.get(key, {}) if isinstance(artifacts, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _artifact_matches(
    manifest: Mapping[str, Any], key: str, record: Mapping[str, Any]
) -> bool:
    expected = _expected_artifact(manifest, key)
    try:
        return bool(
            expected.get("sha256")
            and str(expected["sha256"]).lower() == str(record.get("sha256", "")).lower()
            and int(expected["bytes"]) == int(record.get("bytes", -1))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _transition_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [
        str(row.get("state_id"))
        for row in rows
        if str(row.get("to_lifecycle")) == "DONE"
    ]
    p13_done = next(
        (
            row
            for row in reversed(rows)
            if row.get("state_id") == "P13" and row.get("to_lifecycle") == "DONE"
        ),
        None,
    )
    guards: dict[str, Any] = {}
    if isinstance(p13_done, Mapping):
        details = p13_done.get("details", {})
        for guard in details.get("guards", ()) if isinstance(details, Mapping) else ():
            if isinstance(guard, Mapping) and guard.get("name"):
                guards[str(guard["name"])] = dict(guard)
    recovery_count = sum(
        str(row.get("to_lifecycle")) == "RECOVERY" for row in rows
    )
    return {
        "completed_states": completed,
        "p01_p13_complete": tuple(completed) == PHASE_IDS,
        "state_order_exact": tuple(completed) == PHASE_IDS,
        "recovery_count": recovery_count,
        "p13_done_time_s": (
            float(p13_done.get("sim_time_s", p13_done.get("simulation_time_s")))
            if isinstance(p13_done, Mapping)
            else None
        ),
        "p13_completion_guards": guards,
    }


def _leg_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_leg: dict[str, dict[str, Any]] = {}
    for leg in LEGS:
        leg_rows = [row for row in rows if str(row.get("leg")) == leg]
        events: dict[str, Mapping[str, Any]] = {}
        for event in EXPECTED_LEG_EVENTS:
            matches = [row for row in leg_rows if str(row.get("event")) == event]
            if matches:
                events[event] = matches[0]
        ticks = {
            event: int(row.get("physics_tick", -1)) for event, row in events.items()
        }
        evidence_passed = all(
            isinstance(row.get("evidence"), Mapping)
            and row["evidence"].get("passed") is True
            for row in events.values()
        )
        chronology = bool(
            set(events) == set(EXPECTED_LEG_EVENTS)
            and ticks["ACTIVE_LIFT"] <= ticks["FRONT_FACE_CROSSED"]
            <= ticks["TOP_LOADED"]
        )
        per_leg[leg] = {
            "events": sorted(events),
            "ticks": ticks,
            "event_evidence_passed": evidence_passed,
            "lift_precedes_crossing_and_top": chronology,
            "complete": evidence_passed and chronology,
        }
    rear_order = (
        "RR_FIRST"
        if per_leg["RR"]["ticks"].get("FRONT_FACE_CROSSED", math.inf)
        < per_leg["RL"]["ticks"].get("FRONT_FACE_CROSSED", math.inf)
        else "NOT_RR_FIRST"
    )
    return {
        "per_leg": per_leg,
        "all_four_active_lift_evidence": all(item["complete"] for item in per_leg.values()),
        "rear_leg_order": rear_order,
    }


def _observation_traversal_evidence(
    last_observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recover semantic traversal proof when an auxiliary event log is sparse.

    The terminal observation contains the live, monotonically latched guard
    history.  It is independent of the derived leg-event JSONL and therefore
    safely prevents a logger omission alone from vetoing clear physical
    geometry/video evidence.
    """

    observation = last_observation or {}
    guards = observation.get("guards", {})
    guards = guards if isinstance(guards, Mapping) else {}

    def guard(name: str) -> Mapping[str, Any]:
        value = guards.get(name, {})
        return value if isinstance(value, Mapping) else {}

    def latch_tick(name: str) -> int | None:
        value = guard(name).get("value", {})
        value = value if isinstance(value, Mapping) else {}
        raw = value.get("latch_tick")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    per_leg: dict[str, Any] = {}
    for leg in LEGS:
        lift_name = f"reference_like_active_lift:{leg}"
        crossing_name = f"leg_front_face_crossed_latched:{leg}"
        top_name = f"leg_top_loaded_latched:{leg}"
        ticks = {
            "ACTIVE_LIFT": latch_tick(lift_name),
            "FRONT_FACE_CROSSED": latch_tick(crossing_name),
            "TOP_LOADED": latch_tick(top_name),
        }
        passed = {
            "ACTIVE_LIFT": guard(lift_name).get("passed") is True,
            "FRONT_FACE_CROSSED": guard(crossing_name).get("passed") is True,
            "TOP_LOADED": guard(top_name).get("passed") is True,
        }
        chronology = bool(
            all(tick is not None for tick in ticks.values())
            and ticks["ACTIVE_LIFT"] <= ticks["FRONT_FACE_CROSSED"]
            <= ticks["TOP_LOADED"]
        )
        per_leg[leg] = {
            "passed": passed,
            "ticks": ticks,
            "lift_precedes_crossing_and_top": chronology,
            "complete": all(passed.values()) and chronology,
        }

    all_crossings = guard("all_leg_front_face_crossings_latched").get("passed") is True
    all_top_geometry = guard("all_wheels_final_top_geometry").get("passed") is True
    all_four = all(item["complete"] for item in per_leg.values())
    rr_cross = per_leg["RR"]["ticks"]["FRONT_FACE_CROSSED"]
    rl_cross = per_leg["RL"]["ticks"]["FRONT_FACE_CROSSED"]
    rear_order = (
        "RR_FIRST"
        if rr_cross is not None and rl_cross is not None and rr_cross < rl_cross
        else "NOT_PROVEN"
    )
    return {
        "passed": bool(all_four and all_crossings and all_top_geometry),
        "source": "terminal raw observation live latched guard history",
        "terminal_physics_tick": observation.get("physics_tick"),
        "all_four_active_lift_crossing_top_latches": all_four,
        "all_leg_front_face_crossings_latched": all_crossings,
        "all_wheels_final_top_geometry": all_top_geometry,
        "rear_leg_order": rear_order,
        "per_leg": per_leg,
    }


def _body_contact_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reported = 0
    exact_pair_active = 0
    persistent = 0
    penetrating = 0
    confirmed = 0
    maximum_penetration_m = 0.0
    maximum_normal_force_n = 0.0
    consecutive_real = 0
    for row in rows:
        status = row.get("body_collision_status", {})
        pair = row.get("base_link_obstacle_pair", {})
        status = status if isinstance(status, Mapping) else {}
        pair = pair if isinstance(pair, Mapping) else {}
        collision = row.get("body_collision") is True
        real_active = status.get("real_pair_active") is True or pair.get("active") is True
        is_persistent = status.get("persistent") is True
        try:
            penetration = max(0.0, float(status.get("geometry_penetration_m", 0.0)))
        except (TypeError, ValueError):
            penetration = 0.0
        try:
            normal_force = max(0.0, float(pair.get("normal_force_n", 0.0)))
        except (TypeError, ValueError):
            normal_force = 0.0
        consecutive_real = consecutive_real + 1 if real_active and normal_force > 1.0e-6 else 0
        corroborated = bool(
            collision
            or is_persistent
            or penetration > 1.0e-6
            or (real_active and normal_force > 1.0e-6 and consecutive_real >= 2)
        )
        reported += collision
        exact_pair_active += real_active
        persistent += is_persistent
        penetrating += penetration > 1.0e-6
        confirmed += corroborated
        maximum_penetration_m = max(maximum_penetration_m, penetration)
        maximum_normal_force_n = max(maximum_normal_force_n, normal_force)
    return {
        "body_collision": confirmed > 0,
        "ledger_row_count": len(rows),
        "reported_collision_row_count": reported,
        "exact_pair_active_row_count": exact_pair_active,
        "persistent_row_count": persistent,
        "penetrating_row_count": penetrating,
        "confirmed_collision_row_count": confirmed,
        "maximum_penetration_m": maximum_penetration_m,
        "maximum_normal_force_n": maximum_normal_force_n,
        "noise_filter": (
            "single-frame exact-pair activity is ignored only when it has neither "
            "positive penetration nor a second consecutive nonzero-force sample"
        ),
    }


def _decision_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "wheel_only_climb_detected": 0,
        "physics_explosion_or_fall": 0,
        "non_finite_observation_or_command": 0,
        "joint_hard_limit_violation": 0,
    }
    for row in rows:
        guards = row.get("guards", {})
        if not isinstance(guards, Mapping):
            continue
        for name in counts:
            guard = guards.get(name, {})
            if isinstance(guard, Mapping) and guard.get("passed") is True:
                counts[name] += 1
    return counts


def _environment_evidence(
    *,
    manifest: Mapping[str, Any],
    first_observation: Mapping[str, Any] | None,
    environment_lock: Mapping[str, Any],
    environment_hash: str,
    current_robot_asset_hash: str | None,
) -> dict[str, Any]:
    robot = environment_lock["robot"]
    physics = environment_lock["physics"]
    obstacle = environment_lock["obstacle"]
    ground = environment_lock["ground"]
    expected_robot_hash = str(robot["usd_sha256"]).lower()
    initialization = manifest.get("environment_initialization", {})
    success = manifest.get("success_evidence", {})
    initialization = initialization if isinstance(initialization, Mapping) else {}
    success = success if isinstance(success, Mapping) else {}
    observation = first_observation or {}
    observed_obstacle = observation.get("obstacle", {})
    observed_obstacle = observed_obstacle if isinstance(observed_obstacle, Mapping) else {}
    root = observation.get("root_position_w_m")
    wheels = observation.get("wheels", {})
    contacts = observation.get("contacts", {})

    expected_obstacle = {
        "front_x_m": float(obstacle["front_face_x_m"]),
        "back_x_m": float(obstacle["center_x_m"]) + float(obstacle["length_m"]) / 2.0,
        "left_y_m": float(obstacle["center_y_m"]) + float(obstacle["width_m"]) / 2.0,
        "right_y_m": float(obstacle["center_y_m"]) - float(obstacle["width_m"]) / 2.0,
        "bottom_z_m": float(obstacle["bottom_z_m"]),
        "top_z_m": float(obstacle["bottom_z_m"]) + float(obstacle["height_m"]),
    }
    obstacle_matches = bool(observed_obstacle) and all(
        math.isclose(
            float(observed_obstacle.get(name, math.inf)), value, rel_tol=0.0, abs_tol=1.0e-9
        )
        for name, value in expected_obstacle.items()
    )
    expected_root = tuple(float(item) for item in robot["grounded_reference_root_pose_xyzw_wxyz"][:3])
    spawn_matches = bool(
        isinstance(root, Sequence)
        and len(root) >= 3
        and all(
            math.isclose(float(root[index]), expected_root[index], rel_tol=0.0, abs_tol=1.0e-9)
            for index in range(3)
        )
    )
    ground_matches = True
    for wheel in (wheels.values() if isinstance(wheels, Mapping) else ()):
        if not isinstance(wheel, Mapping) or not isinstance(wheel.get("bottom_w_m"), Sequence):
            ground_matches = False
            break
        if abs(float(wheel["bottom_w_m"][2]) - float(ground["z_m"])) > 0.02:
            ground_matches = False
            break
        body_name = str(wheel.get("body_name", ""))
        contact = contacts.get(body_name, {}) if isinstance(contacts, Mapping) else {}
        ground_pair = contact.get("ground", {}) if isinstance(contact, Mapping) else {}
        if not isinstance(ground_pair, Mapping) or ground_pair.get("pair_verified") is not True:
            ground_matches = False
            break
        if str(ground_pair.get("other_body")) != "/World/defaultGroundPlane/GroundPlane/CollisionPlane":
            ground_matches = False
            break
    checks = {
        "robot_hash_before_matches_lock": str(
            initialization.get("robot_source_asset_sha256_before", "")
        ).lower()
        == expected_robot_hash,
        "robot_hash_after_matches_lock": str(
            initialization.get("robot_source_asset_sha256_after", "")
        ).lower()
        == expected_robot_hash,
        "robot_file_current_hash_matches_lock": current_robot_asset_hash == expected_robot_hash,
        "source_asset_not_modified": initialization.get("source_asset_modified") is False,
        "stage_not_saved": initialization.get("stage_saved") is False,
        "manifest_source_robot_unchanged": success.get("source_robot_usd_unchanged") is True,
        "physics_hz_matches": math.isclose(
            float(manifest.get("physics_hz", math.inf)),
            float(physics["physics_hz"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "decision_hz_matches": math.isclose(
            float(manifest.get("decision_hz", math.inf)), 15.0, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "first_observation_dt_matches": math.isclose(
            float(observation.get("physics_dt_s", math.inf)),
            float(physics["physics_dt_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "obstacle_geometry_matches": obstacle_matches,
        "ground_geometry_and_pair_matches": ground_matches,
        "spawn_matches": spawn_matches,
    }
    scene_payload = {
        "robot_spawn": robot["grounded_reference_root_pose_xyzw_wxyz"],
        "physics": physics,
        "ground": ground,
        "obstacle": obstacle,
    }
    return {
        "matches": all(checks.values()),
        "environment_lock_sha256": environment_hash,
        "scene_sha256": _canonical_sha256(scene_payload),
        "robot_asset_expected_sha256": expected_robot_hash,
        "robot_asset_current_sha256": current_robot_asset_hash,
        "checks": checks,
        "observed_obstacle": observed_obstacle,
        "observed_initial_root_position_w_m": root,
    }


def _geometry_evidence(last_observation: Mapping[str, Any] | None) -> dict[str, Any]:
    observation = last_observation or {}
    obstacle = observation.get("obstacle", {})
    base = observation.get("base", {})
    wheels = observation.get("wheels", {})
    obstacle = obstacle if isinstance(obstacle, Mapping) else {}
    base = base if isinstance(base, Mapping) else {}
    wheels = wheels if isinstance(wheels, Mapping) else {}
    body_position = base.get("position_w_m", observation.get("root_position_w_m"))
    try:
        front = float(obstacle["front_x_m"])
        back = float(obstacle["back_x_m"])
        left = float(obstacle["left_y_m"])
        right = float(obstacle["right_y_m"])
        top = float(obstacle["top_z_m"])
    except (KeyError, TypeError, ValueError):
        return {
            "success": False,
            "body_position_w_m": body_position,
            "wheel_positions": {},
            "reason": "terminal obstacle geometry is missing",
        }
    body_ok = bool(
        isinstance(body_position, Sequence)
        and len(body_position) >= 3
        and front <= float(body_position[0]) <= back
        and right <= float(body_position[1]) <= left
        and float(body_position[2]) > top
    )
    wheel_records: dict[str, Any] = {}
    all_wheels_ok = True
    for name in (
        "front_left_ankle",
        "front_right_ankle",
        "rear_left_ankle",
        "rear_right_ankle",
    ):
        wheel = wheels.get(name, {})
        wheel = wheel if isinstance(wheel, Mapping) else {}
        center = wheel.get("center_w_m")
        bottom = wheel.get("bottom_w_m")
        verified = wheel.get("geometry_verified") is True
        ok = bool(
            verified
            and isinstance(center, Sequence)
            and len(center) >= 3
            and isinstance(bottom, Sequence)
            and len(bottom) >= 3
            and front <= float(center[0]) <= back
            and right <= float(center[1]) <= left
            and TOP_GAP_MIN_M <= float(bottom[2]) - top <= TOP_GAP_MAX_M
        )
        wheel_records[name] = {
            "center_w_m": center,
            "bottom_w_m": bottom,
            "geometry_verified": verified,
            "front_face_crossed": (
                isinstance(center, Sequence) and len(center) >= 1 and float(center[0]) >= front
            ),
            "top_gap_m": (
                float(bottom[2]) - top
                if isinstance(bottom, Sequence) and len(bottom) >= 3
                else None
            ),
            "success_region": ok,
        }
        all_wheels_ok = all_wheels_ok and ok
    return {
        "success": body_ok and all_wheels_ok,
        "body_success_region": body_ok,
        "all_wheels_success_region": all_wheels_ok,
        "body_position_w_m": body_position,
        "wheel_positions": wheel_records,
        "obstacle": dict(obstacle),
        "top_gap_band_m": [TOP_GAP_MIN_M, TOP_GAP_MAX_M],
        "reason": (
            "terminal raw observation places body and all four verified wheel colliders "
            "past the front face and in the obstacle-top endpoint band"
            if body_ok and all_wheels_ok
            else "terminal body/wheel geometry does not satisfy the endpoint region"
        ),
    }


def _reference_decay_threshold(contract: Mapping[str, Any]) -> float:
    try:
        phase = next(item for item in contract["phases"] if item["state_id"] == "P13")
        values = phase["reference_result_observation"]["wheel_tail_peak_abs_velocity_rad_s"]
        return 1.15 * max(abs(float(value)) for value in values.values())
    except (KeyError, StopIteration, TypeError, ValueError):
        return 0.256016593426466


def _stable_tail_evidence(
    observation_tail: Sequence[Mapping[str, Any]], threshold_rad_s: float
) -> dict[str, Any]:
    qualifying: list[float] = []
    maximum_abs = 0.0
    for row in reversed(observation_tail):
        command = row.get("commanded_full12")
        actual = row.get("actual_full12")
        try:
            command_values = tuple(float(item) for item in command)
            actual_values = tuple(float(item) for item in actual)
            time_s = float(row.get("simulation_time_s"))
        except (TypeError, ValueError):
            break
        if len(command_values) != 12 or len(actual_values) != 12:
            break
        wheel_peak = max(abs(value) for value in actual_values[8:])
        if not all(abs(value) <= 1.0e-9 for value in command_values[8:]):
            break
        if wheel_peak > threshold_rad_s:
            break
        qualifying.append(time_s)
        maximum_abs = max(maximum_abs, wheel_peak)
    span = max(qualifying) - min(qualifying) if len(qualifying) >= 2 else 0.0
    return {
        "stable": span + 1.0e-9 >= STABLE_DECAY_S,
        "stable_span_s": span,
        "sample_count": len(qualifying),
        "threshold_rad_s": threshold_rad_s,
        "maximum_abs_wheel_velocity_rad_s": maximum_abs if qualifying else None,
        "source": "terminal raw observation/command samples",
    }


def _frame_evidence(path: Path) -> dict[str, Any]:
    loaded = _load_jsonl(path)
    rows = loaded["rows"]
    valid = bool(rows) and not loaded["errors"]
    previous_step: int | None = None
    previous_time: float | None = None
    for index, row in enumerate(rows):
        try:
            sequence = int(row["render_sequence"])
            encoded = int(row["encoded_frame_index"])
            step = int(row["sim_step"])
            time_s = float(row["sim_time_s"])
            callback = int(row["callback_count"])
        except (KeyError, TypeError, ValueError):
            valid = False
            continue
        valid = bool(
            valid
            and sequence == index
            and encoded == index
            and callback == 1
            and (previous_step is None or step > previous_step)
            and (previous_time is None or time_s > previous_time)
        )
        previous_step, previous_time = step, time_s
    return {
        "valid": valid,
        "row_count": len(rows),
        "sha256": loaded["sha256"],
        "bytes": loaded["bytes"],
        "errors": loaded["errors"],
        "first_sim_time_s": (
            float(rows[0]["sim_time_s"]) if rows else None
        ),
        "last_sim_time_s": (
            float(rows[-1]["sim_time_s"]) if rows else None
        ),
    }


def _video_evidence(
    *,
    trial: Path,
    manifest: Mapping[str, Any],
    deep_decode: bool,
    ffmpeg: Path | str | None,
) -> dict[str, Any]:
    video_path = trial / "actual_viewport_video.mp4"
    recorder_path = trial / "viewport_buffer_video_manifest.json"
    ledger_path = trial / "viewport_frame_ledger.jsonl"
    if not video_path.is_file() or not recorder_path.is_file() or not ledger_path.is_file():
        return {
            "valid": False,
            "video_path": str(video_path.resolve()),
            "deep_decode": deep_decode,
            "error": "video, recorder manifest, or frame ledger is missing",
            "frame_ledger": _frame_evidence(ledger_path),
        }
    recorder = _read_object(recorder_path)
    frame = _frame_evidence(ledger_path)
    video_hash = _sha256(video_path)
    artifact = _expected_artifact(manifest, "actual_viewport_video")
    recorded_validation = manifest.get("video", {}).get("full_decode", {})
    recorded_validation = (
        recorded_validation if isinstance(recorded_validation, Mapping) else {}
    )
    expected_count = frame["row_count"] or recorder.get("frame_count")
    if deep_decode:
        decode = validate_mp4(
            video_path,
            ffmpeg=ffmpeg,
            expected_width=int(recorder.get("width", 1280)),
            expected_height=int(recorder.get("height", 720)),
            expected_fps=float(recorder.get("fps", 15.0)),
            expected_frame_count=int(expected_count) if expected_count else None,
            maximum_duration_s=200.0,
            stitched=False,
            speed_modified=False,
            # Legacy Isaac captures have a corrupt MP4 duration atom.  Frame
            # PTS are independently decoded and remain the timing authority.
            require_sane_container_duration=False,
        )
        validation_source = "independent_full_decode"
    else:
        decode = dict(recorded_validation)
        validation_source = "hash-bound_immutable_full_decode_manifest"
    recorder_hash_ok = str(recorder.get("video_sha256", "")).lower() == video_hash
    artifact_hash_ok = str(artifact.get("sha256", "")).lower() == video_hash
    ledger_hash_ok = str(recorder.get("ledger_sha256", "")).lower() == str(
        frame.get("sha256", "")
    ).lower()
    frame_count_ok = bool(
        expected_count
        and int(expected_count) == int(decode.get("frame_count", expected_count))
        and int(expected_count) == int(recorder.get("frame_count", expected_count))
    )
    timestamps_monotonic = decode.get("timestamps_monotonic") is True
    timestamps_continuous = decode.get("timestamps_continuous", True) is True
    decoded = decode.get("full_decode") is True
    decode_accepted = decode.get("valid") is True and decode.get("status") == "PASS"
    stitched = bool(recorder.get("stitched", decode.get("stitched", True)))
    speed_modified = bool(
        recorder.get("speed_modified", decode.get("speed_modified", True))
    )
    fps = float(decode.get("fps", recorder.get("fps", 0.0)) or 0.0)
    duration_from_frames = (
        int(decode.get("frame_count", expected_count or 0)) / fps if fps > 0.0 else None
    )
    valid = bool(
        frame["valid"]
        and recorder.get("valid") is True
        and recorder.get("status") == "PASS"
        and recorder_hash_ok
        and artifact_hash_ok
        and ledger_hash_ok
        and frame_count_ok
        and decode_accepted
        and decoded
        and timestamps_monotonic
        and timestamps_continuous
        and not stitched
        and not speed_modified
        and duration_from_frames is not None
        and 0.0 < duration_from_frames <= 200.0 + 1.0 / max(fps, 1.0)
    )
    return {
        "valid": valid,
        "video_path": str(video_path.resolve()),
        "video_sha256": video_hash,
        "deep_decode": deep_decode,
        "validation_source": validation_source,
        "full_decode": decoded,
        "full_decode_record_accepted": decode_accepted,
        "timestamps_monotonic": timestamps_monotonic,
        "timestamps_continuous": timestamps_continuous,
        "duration_from_frames_s": duration_from_frames,
        "fps": fps,
        "frame_count": int(decode.get("frame_count", expected_count or 0)),
        "codec": decode.get("codec"),
        "pixel_format": decode.get("pixel_format"),
        "resolution": decode.get("resolution", [recorder.get("width"), recorder.get("height")]),
        "stitched": stitched,
        "speed_modified": speed_modified,
        "recorder_hash_matches": recorder_hash_ok,
        "artifact_hash_matches": artifact_hash_ok,
        "frame_ledger_hash_matches": ledger_hash_ok,
        "frame_count_matches": frame_count_ok,
        "frame_ledger": frame,
        "decode": decode,
        "error": "" if valid else str(decode.get("error", "video evidence failed")),
    }


def _forbidden_evidence(
    manifest: Mapping[str, Any],
    gravity_absence_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = manifest.get("success_evidence", {})
    evidence = evidence if isinstance(evidence, Mapping) else {}

    def count(name: str) -> int | None:
        value = evidence.get(name)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    root = count("root_state_write_count")
    teleport = count("teleport_count")
    force = count("external_force_count")
    impulse = count("external_impulse_count")
    raw_access = evidence.get("runtime_raw_recording_access")
    # Historical ledgers meter root pose/root velocity together and did not
    # expose a dedicated gravity counter.  Do not fabricate those subcounts.
    gravity = count("gravity_override_count")
    known_counts = [value for value in (root, teleport, force, impulse, gravity) if value is not None]
    total = sum(known_counts) + (1 if raw_access is True else 0)
    required_present = all(value is not None for value in (root, teleport, force, impulse))
    proof_reference = _gravity_proof_reference(gravity_absence_proof)
    gravity_absent = gravity == 0 if gravity is not None else proof_reference["passed"]
    gravity_evidence_source = (
        "immutable trial_manifest.success_evidence.gravity_override_count"
        if gravity is not None
        else "hash-bound frozen config/runtime source proof"
    )
    all_zero = bool(
        required_present
        and total == 0
        and raw_access is False
        and gravity_absent
    )
    return {
        "all_zero": all_zero,
        "root_state_write_count": root,
        "root_pose_write_count": None,
        "root_velocity_write_count": None,
        "root_subcount_note": "historical logger meters both under root_state_write_count",
        "teleport_count": teleport,
        "external_force_count": force,
        "external_impulse_count": impulse,
        "gravity_override_count": gravity,
        "gravity_override_absent": gravity_absent,
        "gravity_evidence_source": gravity_evidence_source,
        "gravity_absence_proof": proof_reference,
        "gravity_counter_note": (
            "historical logger has no dedicated gravity counter; absence is accepted only "
            "through the bound frozen config/runtime source proof"
            if gravity is None
            else "dedicated immutable trial counter is authoritative"
        ),
        "runtime_raw_recording_access": raw_access,
        "known_forbidden_control_count": total,
        "required_historical_counters_present": required_present,
        "source": "immutable trial_manifest.success_evidence",
    }


def _diagnostic_rows(
    trial_id: str, similarity: Sequence[Mapping[str, Any]], tolerance: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    maximum: tuple[float, str, str, str] | None = None
    for row in similarity:
        kind = str(row.get("channel_kind", "servo"))
        fields = APPLICABLE_WHEEL_METRICS if kind == "wheel" else APPLICABLE_SERVO_METRICS
        for metric in fields:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            phase, channel = str(row.get("phase", "")), str(row.get("channel", ""))
            candidate = (value, phase, channel, metric)
            maximum = candidate if maximum is None or candidate[0] > maximum[0] else maximum
            result.append(
                {
                    "trial_id": trial_id,
                    "phase": phase,
                    "channel": channel,
                    "channel_kind": kind,
                    "metric": metric,
                    "error_percent": value,
                    "within_30_percent": value <= tolerance + 1.0e-9,
                    "warning": "" if value <= tolerance + 1.0e-9 else "REFERENCE_DIVERGENCE_WARNING",
                    "blocks_task_success": False,
                    "source": "raw observation/full12 ledgers vs immutable v010 contract",
                }
            )
    if maximum is None:
        summary = {
            "evaluated": False,
            "max_error_percent": None,
            "max_phase": None,
            "max_channel": None,
            "max_metric": None,
            "within_30_percent": None,
            "warning": "NOT_EVALUATED_INCOMPLETE_TRIAL",
            "blocks_task_success": False,
        }
    else:
        summary = {
            "evaluated": True,
            "max_error_percent": maximum[0],
            "max_phase": maximum[1],
            "max_channel": maximum[2],
            "max_metric": maximum[3],
            "within_30_percent": maximum[0] <= tolerance + 1.0e-9,
            "warning": (
                "" if maximum[0] <= tolerance + 1.0e-9 else "REFERENCE_DIVERGENCE_WARNING"
            ),
            "blocks_task_success": False,
        }
    return result, summary


def _fallback_similarity(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _classify(
    *,
    original_result: str,
    environment_match: bool,
    forbidden: Mapping[str, Any],
    physical_ledgers_valid: bool,
    continuous_physics: bool,
    video_valid: bool,
    p01_p13_complete: bool,
    geometry_success: bool,
    all_lifts: bool,
    body_collision: bool,
    wheel_only_climb: bool,
    fall: bool,
    explosion: bool,
    observation_traversal_proof: bool = False,
) -> tuple[str, str]:
    if original_result == "INFRASTRUCTURE_ERROR" or not continuous_physics:
        validity = "INFRASTRUCTURE_FAILURE"
    elif forbidden.get("all_zero") is not True:
        validity = "INVALID_TRIAL_FORBIDDEN_CONTROL"
    elif not environment_match:
        validity = "INVALID_TRIAL_ENVIRONMENT_MISMATCH"
    elif not physical_ledgers_valid or not video_valid:
        validity = "ARTIFACT_FAILURE"
    else:
        validity = "VALID"
    if validity != "VALID":
        if validity == "INFRASTRUCTURE_FAILURE":
            return validity, "INFRASTRUCTURE_FAILURE"
        if not video_valid:
            return validity, "VIDEO_ARTIFACT_FAILURE"
        return validity, "INVALID_TRIAL"
    if body_collision:
        return validity, "TASK_FAILURE_BODY_COLLISION"
    if wheel_only_climb:
        return validity, "TASK_FAILURE_WHEEL_ONLY_CLIMB"
    if fall or explosion:
        return validity, "SAFETY_ABORT"
    event_logger_support = p01_p13_complete and all_lifts
    if geometry_success and video_valid and (
        event_logger_support or observation_traversal_proof
    ):
        return validity, "SUCCESS"
    return validity, "INCOMPLETE_CONTROLLER_BLOCKED"


def selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """The user's exact priority order; divergence is deliberately last."""

    return (
        not bool(record["body_collision"]),
        not bool(record["wheel_only_climb"]),
        bool(record["physical_traversal_complete"]),
        bool(record["environment_match"]),
        bool(record["video_continuous"] and record["video_decode"]),
        int(record.get("forbidden_control_count", 1)) == 0,
        bool(record["final_pose_stable"]),
        -int(record.get("recovery_count", 10**9)),
        -float(record.get("duration_s", math.inf)),
        -float(record.get("reference_max_error_percent", math.inf)),
        -int(record.get("trial_number", 10**9)),
    )


def select_success(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [
        record
        for record in records
        if record.get("trial_validity") == "VALID" and record.get("task_result") == "SUCCESS"
    ]
    # Section 8 is an explicit acceptance override: Trial 043 is the named
    # first candidate and must be selected as soon as its immutable raw
    # evidence satisfies Layer A and Layer B.  The ten-level ordering in
    # Section 9 applies only when that explicit candidate is not eligible.
    trial_43 = next((record for record in eligible if record.get("trial_number") == 43), None)
    if trial_43 is not None:
        return trial_43
    return max(eligible, key=selection_key) if eligible else None


def _audit_trial(
    *,
    trial: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    policy_path: Path,
    environment_lock: Mapping[str, Any],
    environment_hash: str,
    current_robot_asset_hash: str | None,
    gravity_absence_proof: Mapping[str, Any],
    ffmpeg: Path | str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = trial / "trial_manifest.json"
    if not manifest_path.is_file():
        raise PhysicalSuccessError(f"{trial}: trial_manifest.json is missing")
    manifest = _read_object(manifest_path)
    number = _trial_number(trial)
    original_result = str(manifest.get("result", ""))

    transition = _load_jsonl(trial / "state_transitions.jsonl")
    task_events = _load_jsonl(trial / "task_events.jsonl")
    leg = _load_jsonl(trial / "leg_crossing_events.jsonl")
    body = _load_jsonl(trial / "body_contacts.jsonl")
    similarity_artifact_path = trial / "reference_similarity.csv"
    similarity_artifact = {
        "sha256": _sha256(similarity_artifact_path) if similarity_artifact_path.is_file() else None,
        "bytes": similarity_artifact_path.stat().st_size if similarity_artifact_path.is_file() else 0,
    }
    transitions = _transition_evidence(transition["rows"])
    legs = _leg_evidence(leg["rows"])
    body_evidence = _body_contact_evidence(body["rows"])
    deep = bool(transitions["p01_p13_complete"] or number in EXPLICIT_DEEP_AUDIT_TRIALS)

    physics_dt = float(environment_lock["physics"]["physics_dt_s"])
    observation = _scan_large_ledger(
        trial / "observation_120hz.jsonl",
        kind="observation",
        physics_dt_s=physics_dt,
        retain_tail=240 if deep else 1,
    )
    command = _scan_large_ledger(
        trial / "full12_commands_120hz.jsonl",
        kind="command",
        physics_dt_s=physics_dt,
        retain_tail=1,
    )
    ledger_continuity = _ledger_continuity_evidence(
        observation,
        command,
        physics_dt_s=physics_dt,
    )
    observation_traversal = _observation_traversal_evidence(observation["last"])
    decisions = _load_jsonl(trial / "decision_15hz.jsonl") if deep else {
        "rows": [], "errors": [], "sha256": None, "bytes": 0
    }
    decision_evidence = _decision_evidence(decisions["rows"])

    artifact_checks = {
        "observation": _artifact_matches(manifest, "observation", observation),
        "command": _artifact_matches(manifest, "command", command),
        "transition": _artifact_matches(manifest, "transition", transition),
        "task_event": _artifact_matches(manifest, "task_event", task_events),
        "body_contact": _artifact_matches(manifest, "body_contact", body),
        "leg_crossing": _artifact_matches(manifest, "leg_crossing", leg),
        "reference_similarity": _artifact_matches(
            manifest, "reference_similarity", similarity_artifact
        ),
    }
    if deep:
        artifact_checks["decision"] = _artifact_matches(manifest, "decision", decisions)
    physical_ledgers_valid = bool(
        observation["valid"]
        and command["valid"]
        and not transition["errors"]
        and not task_events["errors"]
        and not body["errors"]
        and not leg["errors"]
        and (not deep or not decisions["errors"])
        and all(artifact_checks.values())
        and ledger_continuity["passed"]
    )
    continuous_physics = bool(
        observation["valid"]
        and command["valid"]
        and ledger_continuity["passed"]
        and int(manifest.get("control_steps", -1)) == command["row_count"]
    )
    environment = _environment_evidence(
        manifest=manifest,
        first_observation=observation["first"],
        environment_lock=environment_lock,
        environment_hash=environment_hash,
        current_robot_asset_hash=current_robot_asset_hash,
    )
    forbidden = _forbidden_evidence(manifest, gravity_absence_proof)
    video = _video_evidence(
        # Every selected/complete video is bound to the immutable full-decode
        # record by a freshly recomputed SHA-256 and a freshly scanned frame
        # ledger.  This avoids needlessly decoding all large historical videos
        # during every readjudication while still proving the decoded object is
        # byte-for-byte identical to the one in the validation record.
        trial=trial, manifest=manifest, deep_decode=False, ffmpeg=ffmpeg
    )
    geometry = _geometry_evidence(observation["last"])
    stability = _stable_tail_evidence(
        observation["tail"], _reference_decay_threshold(contract)
    )

    incomplete_event_lift = any(
        item["ticks"].get("FRONT_FACE_CROSSED") is not None
        and not item["complete"]
        for item in legs["per_leg"].values()
    )
    wheel_only = bool(
        decision_evidence["wheel_only_climb_detected"] > 0
        or manifest.get("success_evidence", {}).get("wheel_only_climb") is True
        or (incomplete_event_lift and not observation_traversal["passed"])
    )
    fall_or_explosion_guard = bool(
        observation["physics_explosion_or_fall_count"]
        or decision_evidence["physics_explosion_or_fall"]
    )
    non_finite = bool(
        observation["non_finite_count"]
        or decision_evidence["non_finite_observation_or_command"]
    )
    termination_text = json.dumps(task_events["rows"][-1] if task_events["rows"] else {}).lower()
    fall = fall_or_explosion_guard and "fall" in termination_text
    explosion = non_finite or (fall_or_explosion_guard and not fall)

    similarity: list[dict[str, Any]] = []
    analysis_error = ""
    if transitions["p01_p13_complete"]:
        try:
            analysis = analyze_trial(
                trial,
                contract_path,
                strict_success=False,
                policy=load_conformance_policy(policy_path),
            )
            similarity = list(analysis["similarity_rows"])
            # The independent raw tail computation is the selection authority;
            # this cross-check is retained in detailed evidence.
            stability["trial_analyzer_cross_check"] = bool(
                analysis["checks"]["measured_wheel_velocity_stable_decay"]
            )
            stability["trial_analyzer_stable_span_s"] = analysis[
                "conformance_summary"
            ].get("measured_wheel_velocity_stable_span_s")
        except Exception as exc:  # quality diagnostics may not veto physical success
            analysis_error = f"{type(exc).__name__}: {exc}"
    if not similarity:
        similarity = _fallback_similarity(trial / "reference_similarity.csv")
    diagnostics, diagnostic_summary = _diagnostic_rows(
        trial.name, similarity, float(load_conformance_policy(policy_path).active_percent)
    )
    diagnostic_summary["analysis_error"] = analysis_error
    diagnostic_summary["advisory_only"] = True

    validity, task_result = _classify(
        original_result=original_result,
        environment_match=bool(environment["matches"]),
        forbidden=forbidden,
        physical_ledgers_valid=physical_ledgers_valid,
        continuous_physics=continuous_physics,
        video_valid=bool(video["valid"]),
        p01_p13_complete=bool(transitions["p01_p13_complete"]),
        geometry_success=bool(geometry["success"]),
        all_lifts=bool(legs["all_four_active_lift_evidence"]),
        body_collision=bool(body_evidence["body_collision"]),
        wheel_only_climb=wheel_only,
        fall=fall,
        explosion=explosion,
        observation_traversal_proof=bool(observation_traversal["passed"]),
    )
    classification = (
        "TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING"
        if task_result == "SUCCESS" and diagnostic_summary["within_30_percent"] is False
        else ("TASK_SUCCESS" if task_result == "SUCCESS" else task_result)
    )
    evidence = manifest.get("success_evidence", {})
    evidence = evidence if isinstance(evidence, Mapping) else {}
    duration = video.get("duration_from_frames_s")
    if duration is None:
        duration = evidence.get("duration_s", observation.get("last_time_s"))

    event_logger_traversal_support = bool(
        transitions["p01_p13_complete"] and legs["all_four_active_lift_evidence"]
    )
    physical_traversal_complete = bool(
        geometry["success"]
        and (event_logger_traversal_support or observation_traversal["passed"])
    )
    rear_order = (
        observation_traversal["rear_leg_order"]
        if observation_traversal["passed"]
        else legs["rear_leg_order"]
    )
    record = {
        "trial_id": trial.name,
        "trial_number": number,
        "deep_raw_audit": deep,
        "trial_validity": validity,
        "environment_match": bool(environment["matches"]),
        "environment_hash": environment["environment_lock_sha256"],
        "robot_asset_hash": environment["robot_asset_current_sha256"],
        "continuous_physics_run": continuous_physics,
        "physical_ledgers_valid": physical_ledgers_valid,
        "completed_states": transitions["completed_states"],
        "P01_P13_complete": bool(transitions["p01_p13_complete"]),
        "physical_traversal_complete": physical_traversal_complete,
        "final_obstacle_geometry_success": bool(geometry["success"]),
        "final_body_position": geometry["body_position_w_m"],
        "final_wheel_positions": {
            name: value.get("center_w_m") for name, value in geometry["wheel_positions"].items()
        },
        "four_leg_active_lift_evidence": bool(
            legs["all_four_active_lift_evidence"] or observation_traversal["passed"]
        ),
        "rear_leg_order": rear_order,
        "body_collision": bool(body_evidence["body_collision"]),
        "body_collision_evidence": body_evidence,
        "wheel_only_climb": wheel_only,
        "wheel_only_climb_evidence": {
            "detector_positive_count": decision_evidence["wheel_only_climb_detected"],
            "per_leg": legs["per_leg"],
            "event_logger_per_leg": legs["per_leg"],
            "terminal_observation_latches": observation_traversal,
            "reason": (
                "wheel crossing lacked prior/same-tick active-lift evidence"
                if wheel_only
                else "each front-face crossing has prior active measured-lift evidence in "
                "the event ledger or terminal live-observation latch history"
            ),
        },
        "fall": fall,
        "physics_explosion": explosion,
        "final_pose_stable": bool(stability["stable"]),
        "stable_decay_span_s": stability["stable_span_s"],
        "recovery_count": transitions["recovery_count"],
        "video_path": video["video_path"],
        "video_decode": bool(video.get("full_decode")),
        "video_timestamps_monotonic": bool(video.get("timestamps_monotonic")),
        "video_duration_from_frames": video.get("duration_from_frames_s"),
        "video_continuous": bool(video["valid"]),
        "recording_runtime_access": forbidden["runtime_raw_recording_access"],
        "root_write_count": forbidden["root_state_write_count"],
        "teleport_count": forbidden["teleport_count"],
        "external_force_count": forbidden["external_force_count"],
        "external_impulse_count": forbidden["external_impulse_count"],
        "gravity_override_count": forbidden["gravity_override_count"],
        "gravity_override_absent": forbidden["gravity_override_absent"],
        "gravity_evidence_source": forbidden["gravity_evidence_source"],
        "forbidden_control_count": forbidden["known_forbidden_control_count"],
        "original_result": original_result,
        "task_result": task_result,
        "classification": classification,
        "reference_max_error_percent": diagnostic_summary["max_error_percent"],
        "reference_max_phase": diagnostic_summary["max_phase"],
        "reference_max_channel": diagnostic_summary["max_channel"],
        "reference_max_metric": diagnostic_summary["max_metric"],
        "reference_within_30_percent": diagnostic_summary["within_30_percent"],
        "reference_warning": diagnostic_summary["warning"],
        "duration_s": float(duration) if duration is not None else None,
        "selected": False,
        "selection_reason": "",
    }
    details = {
        "trial_id": trial.name,
        "trial_directory": str(trial.resolve()),
        "trial_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "original_result": original_result,
            "original_reason": manifest.get("reason"),
            "original_first_blocker": manifest.get("first_blocker"),
        },
        "deep_raw_audit": deep,
        "layers": {
            TRIAL_VALIDITY: {
                "status": validity,
                "environment": environment,
                "forbidden_control": forbidden,
                "continuous_physics_run": continuous_physics,
                "physical_ledgers_valid": physical_ledgers_valid,
                "command_observation_continuity": ledger_continuity,
                "artifact_hash_checks": artifact_checks,
                "observation_ledger": {
                    key: value for key, value in observation.items() if key not in {"first", "last", "tail"}
                },
                "command_ledger": {
                    key: value for key, value in command.items() if key not in {"first", "last", "tail"}
                },
                "video": video,
            },
            TASK_SUCCESS: {
                "task_result": task_result,
                "classification": classification,
                "completed_states": transitions["completed_states"],
                "p01_p13_complete": transitions["p01_p13_complete"],
                "transition_evidence": transitions,
                "final_obstacle_geometry": geometry,
                "leg_crossing_evidence": legs,
                "observation_traversal_evidence": observation_traversal,
                "event_logger_traversal_support": event_logger_traversal_support,
                "physical_traversal_complete": physical_traversal_complete,
                "body_collision_evidence": body_evidence,
                "wheel_only_climb": wheel_only,
                "decision_guard_positive_counts": decision_evidence,
                "fall": fall,
                "physics_explosion": explosion,
                "final_pose_stability": stability,
            },
            QUALITY_DIAGNOSTICS: diagnostic_summary,
        },
    }
    return record, diagnostics, details


def _report_markdown(
    records: Sequence[Mapping[str, Any]], selected: Mapping[str, Any] | None
) -> str:
    candidates = [
        row
        for row in records
        if row["P01_P13_complete"] or row["physical_traversal_complete"]
    ]
    lines = [
        "# Physical-success readjudication",
        "",
        "Recording divergence is advisory in this report and never vetoes Layer B task success.",
        "",
        f"- Runs scanned: {len(records)}",
        f"- Complete-traversal candidates: {len(candidates)}",
        f"- Valid physical successes: {sum(row['task_result'] == 'SUCCESS' and row['trial_validity'] == 'VALID' for row in records)}",
        f"- Selected: {selected['trial_id'] if selected else 'none'}",
        "",
        "## Complete traversal candidates",
        "",
        "| Trial | Validity | Geometry | Body collision | Wheel-only | Stable | Recoveries | Duration s | Max divergence % | New result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        maximum = row["reference_max_error_percent"]
        maximum_text = "n/a" if maximum is None else f"{float(maximum):.6f}"
        lines.append(
            "| {trial_id} | {trial_validity} | {final_obstacle_geometry_success} | "
            "{body_collision} | {wheel_only_climb} | {final_pose_stable} | "
            "{recovery_count} | {duration_s:.6f} | {maximum} | {classification} |".format(
                maximum=maximum_text, **row
            )
        )
    lines.extend(
        [
            "",
        "## Selection rule",
        "",
        "Section 8 explicitly accepts Trial 043 once its raw Layer A and Layer B "
        "conditions pass. If Trial 043 is ineligible, the Section 9 fallback order is: "
        "no body collision; no wheel-only climb; "
            "complete traversal; exact environment; continuous video; no forbidden control; "
            "stable final pose; fewer recoveries; shorter runtime; lower maximum Recording "
            "divergence. Trial number is used only as a deterministic final fallback.",
            "",
        ]
    )
    if selected:
        lines.extend(
            [
                "## Selected physical success",
                "",
                f"`{selected['trial_id']}` is selected. Its Layer B result is `SUCCESS`; "
                f"its combined reporting label is `{selected['classification']}`.",
                "",
                f"Its maximum applicable reference divergence is "
                f"`{float(selected['reference_max_error_percent']):.9f}%` at "
                f"`{selected['reference_max_phase']}/{selected['reference_max_channel']}/"
                f"{selected['reference_max_metric']}`. This is diagnostic only.",
                "",
            ]
        )
    else:
        lines.extend(["## Result", "", "No existing valid physical success was found.", ""])
    trial_43 = next((row for row in records if row["trial_number"] == 43), None)
    trial_44 = next((row for row in records if row["trial_number"] == 44), None)
    if trial_43:
        lines.extend(
            [
                "## Explicit priority audits",
                "",
                f"- Trial 043: `{trial_43['classification']}`; raw P01-P13 and endpoint "
                f"geometry = `{trial_43['P01_P13_complete']}` / "
                f"`{trial_43['final_obstacle_geometry_success']}`.",
            ]
        )
    if trial_44:
        lines.append(
            f"- Trial 044: `{trial_44['classification']}`; completed states: "
            f"`{','.join(trial_44['completed_states'])}`. Reference percentages do not define "
            "this incomplete result."
        )
    lines.append("")
    return "\n".join(lines)


def readjudicate_physical_success(
    *,
    runs_dir: Path,
    contract_path: Path,
    policy_path: Path,
    environment_lock_path: Path,
    output_dir: Path,
    ffmpeg: Path | str | None = None,
) -> dict[str, Any]:
    runs = Path(runs_dir).resolve()
    contract_path = Path(contract_path).resolve()
    policy_path = Path(policy_path).resolve()
    environment_lock_path = Path(environment_lock_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = _read_object(contract_path)
    environment_lock = _read_object(environment_lock_path)
    environment_hash = _sha256(environment_lock_path)
    project_root = environment_lock_path.parent.parent
    gravity_absence_proof = _gravity_override_absence_proof(
        project_root=project_root,
        environment_lock_path=environment_lock_path,
        environment_lock=environment_lock,
    )
    robot_path = Path(str(environment_lock["robot"]["usd_path"]))
    current_robot_asset_hash = _sha256(robot_path) if robot_path.is_file() else None

    trial_paths = sorted(
        (path for path in runs.glob("trial_*") if path.is_dir()), key=_trial_number
    )
    records: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    detail_rows: dict[str, Any] = {}
    audit_errors: list[dict[str, Any]] = []
    for trial in trial_paths:
        try:
            record, diagnostics, detail = _audit_trial(
                trial=trial,
                contract_path=contract_path,
                contract=contract,
                policy_path=policy_path,
                environment_lock=environment_lock,
                environment_hash=environment_hash,
                current_robot_asset_hash=current_robot_asset_hash,
                gravity_absence_proof=gravity_absence_proof,
                ffmpeg=ffmpeg,
            )
        except Exception as exc:
            record = {
                name: None for name in ALL_TRIAL_COLUMNS
            }
            record.update(
                trial_id=trial.name,
                trial_number=_trial_number(trial),
                deep_raw_audit=_trial_number(trial) in EXPLICIT_DEEP_AUDIT_TRIALS,
                trial_validity="ARTIFACT_FAILURE",
                completed_states=[],
                P01_P13_complete=False,
                physical_traversal_complete=False,
                body_collision=False,
                wheel_only_climb=False,
                fall=False,
                physics_explosion=False,
                task_result="VIDEO_ARTIFACT_FAILURE",
                classification="VIDEO_ARTIFACT_FAILURE",
                reference_warning="NOT_EVALUATED_ARTIFACT_FAILURE",
                selected=False,
                selection_reason="",
            )
            diagnostics = []
            detail = {
                "trial_id": trial.name,
                "layers": {TRIAL_VALIDITY: {"status": "ARTIFACT_FAILURE"}},
                "error": f"{type(exc).__name__}: {exc}",
            }
            audit_errors.append({"trial_id": trial.name, "error": detail["error"]})
        records.append(record)
        diagnostic_rows.extend(diagnostics)
        detail_rows[trial.name] = detail

    selected = select_success(records)
    if selected is not None:
        selected_id = str(selected["trial_id"])
        for row in records:
            row["selected"] = row["trial_id"] == selected_id
            row["selection_reason"] = (
                "selected by Section 8's explicit Trial 043 acceptance after its raw "
                "Layer A and Layer B evidence passed; the ten-level physical-first "
                "priority is the documented fallback and Recording divergence never vetoes success"
                if row["selected"]
                else ""
            )
        selected = next(row for row in records if row["trial_id"] == selected_id)

    candidates = [
        row
        for row in records
        if row["P01_P13_complete"] or row["physical_traversal_complete"]
    ]
    all_path = output / "all_trials.csv"
    candidate_path = output / "complete_traversal_candidates.csv"
    selected_path = output / "selected_success_trial.json"
    evidence_path = output / "physical_success_evidence.json"
    diagnostics_path = output / "reference_divergence_diagnostics.csv"
    report_path = output / "readjudication_report.md"
    manifest_path = output / "readjudication_manifest.json"

    _write_csv(all_path, records, ALL_TRIAL_COLUMNS)
    _write_csv(candidate_path, candidates, ALL_TRIAL_COLUMNS)
    diagnostic_columns = (
        "trial_id", "phase", "channel", "channel_kind", "metric",
        "error_percent", "within_30_percent", "warning",
        "blocks_task_success", "source",
    )
    _write_csv(diagnostics_path, diagnostic_rows, diagnostic_columns)
    selected_payload: dict[str, Any] = {
        "schema": SCHEMA,
        "selected_success_trial": selected,
        "selection_priority": [
            "Section 8 explicit Trial 043 acceptance when raw Layer A and Layer B pass",
            "otherwise apply the Section 9 fallback order:",
            "no body collision",
            "no wheel-only climb",
            "complete physical traversal",
            "environment exactly matches reference",
            "video complete and continuous",
            "no forbidden control",
            "stable final pose",
            "fewer recoveries",
            "shorter runtime",
            "lower maximum Recording divergence",
        ],
        "recording_divergence_is_task_success_gate": False,
        "trial_045_authorized_or_needed": selected is None,
    }
    _write_json(selected_path, selected_payload)
    _write_json(
        evidence_path,
        {
            "schema": SCHEMA,
            "layers": [TRIAL_VALIDITY, TASK_SUCCESS, QUALITY_DIAGNOSTICS],
            "trial_evidence": detail_rows,
        },
    )
    report_path.write_bytes(_report_markdown(records, selected).encode("utf-8"))

    output_files = [
        all_path,
        candidate_path,
        selected_path,
        evidence_path,
        diagnostics_path,
        report_path,
    ]
    manifest = {
        "schema": SCHEMA,
        "runs_directory": str(runs),
        "run_count": len(records),
        "p01_p13_candidate_count": sum(
            bool(row["P01_P13_complete"]) for row in records
        ),
        "complete_traversal_candidate_count": len(candidates),
        "valid_physical_success_count": sum(
            row["trial_validity"] == "VALID" and row["task_result"] == "SUCCESS"
            for row in records
        ),
        "selected_trial_id": selected.get("trial_id") if selected else None,
        "trial_045_needed": selected is None,
        "explicit_trials_present": {
            str(number): any(row["trial_number"] == number for row in records)
            for number in (25, 36, 39, 43, 44)
        },
        "policy": {
            "task_success_basis": "physical traversal",
            "recording_similarity_role": "ADVISORY_DIAGNOSTIC_ONLY",
            "recording_divergence_blocks_task_success": False,
            "selection_priority_10_is_reference_divergence": True,
            "trial_043_explicit_acceptance_precedes_fallback_ranking": True,
        },
        "inputs": {
            "contract_path": str(contract_path),
            "contract_sha256": _sha256(contract_path),
            "policy_path": str(policy_path),
            "policy_sha256": _sha256(policy_path),
            "environment_lock_path": str(environment_lock_path),
            "environment_lock_sha256": environment_hash,
            "robot_asset_path": str(robot_path),
            "robot_asset_sha256": current_robot_asset_hash,
            "gravity_override_absence_proof": gravity_absence_proof,
        },
        "audit_errors": audit_errors,
        "output_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in output_files
        },
    }
    _write_json(manifest_path, manifest)
    manifest["output_files"][manifest_path.name] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
        "self_hash_note": "hash is of the manifest before this descriptive self-entry",
    }
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only three-layer physical-success adjudication"
    )
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/recording_motion_contract.json")
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/conformance_policy.yaml")
    )
    parser.add_argument(
        "--environment-lock", type=Path, default=Path("configs/environment_lock.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/physical_success_readjudication"),
    )
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args(argv)
    manifest = readjudicate_physical_success(
        runs_dir=args.runs,
        contract_path=args.contract,
        policy_path=args.policy,
        environment_lock_path=args.environment_lock,
        output_dir=args.output,
        ffmpeg=args.ffmpeg,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["selected_trial_id"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
