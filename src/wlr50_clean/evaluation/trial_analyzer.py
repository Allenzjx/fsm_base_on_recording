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

from .comparison import PHASE_IDS
from .conformance import (
    SIMILARITY_COLUMNS,
    TrialAnalysisError,
    _number,
    _phase,
    _phase_windows,
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


def _recovery_evidence(transitions: Sequence[Mapping[str, Any]]) -> tuple[list[float], dict[str, int]]:
    rows = [
        row for row in transitions
        if str(row.get("from_lifecycle")) == "RECOVERY"
        and str(row.get("to_lifecycle")) == "EXECUTE_MOTION"
    ]
    values: list[float] = []
    for row in rows:
        details = row.get("details")
        raw = details.get("correction_fractions") if isinstance(details, Mapping) else None
        if isinstance(raw, Mapping):
            candidates: Sequence[Any] = tuple(raw.values())
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 12:
            candidates = raw
        else:
            candidates = (math.inf,)
        try:
            parsed = [abs(float(item)) for item in candidates]
        except (TypeError, ValueError):
            parsed = [math.inf]
        values.extend(item if math.isfinite(item) else math.inf for item in parsed)
    return values, {phase: sum(_phase(row) == phase for row in rows) for phase in PHASE_IDS}


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
    correction_values, retry_counts = _recovery_evidence(transitions)
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
    stable_suffix: list[Mapping[str, Any]] = []
    for row in reversed(ordered_observations):
        vector = _vector(row, "actual_full12")
        if vector is None or any(abs(value) > 0.05 + 1.0e-9 for value in vector[8:]):
            break
        stable_suffix.append(row)
    stable_suffix.reverse()
    stable_span = _time(stable_suffix[-1]) - _time(stable_suffix[0]) if len(stable_suffix) >= 2 else 0.0

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
        "final_wheel_targets_zero": final_command is not None
        and all(abs(value) <= 1.0e-9 for value in final_command[8:]),
        "measured_wheel_velocity_stable_decay": stable_span >= 0.5 - 1.0 / 120.0 - 1.0e-9,
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
