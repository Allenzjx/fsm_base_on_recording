from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wlr50_clean.evaluation.comparison import PHASE_IDS
from wlr50_clean.evaluation.trial_analyzer import (
    FINAL_DATA_NAMES,
    JSONL_FILES,
    TrialAnalysisError,
    TrialArtifactWriter,
    _drive_feedback_ledger_valid,
    _recovery_evidence,
    analyze_trial,
    populate_reference_similarity,
    publish_successful_trial,
)


ORDER = (
    "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
    "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle",
)
ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_success(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    phases = []
    observations: list[dict] = []
    commands: list[dict] = []
    transitions: list[dict] = []
    for index, phase_id in enumerate(PHASE_IDS):
        start_time = index * 1.2
        end_time = start_time + 1.0
        completion_time = end_time + (0.6 if phase_id == "P13" else 0.1)
        start_value = index * 10.0
        end_value = start_value + 10.0
        start = [0.0] * 12
        end = [0.0] * 12
        start[0] = start_value
        end[0] = end_value
        phases.append({
            "state_id": phase_id,
            "macro_phase": index + 1,
            "state_name": f"STATE_{phase_id}",
            "physical_purpose": f"purpose {phase_id}",
            "reference_steps": [index + 1],
            "active_duration_s": 1.0,
            "start_full12": start,
            "end_full12": end,
            "delta_full12": [10.0] + [0.0] * 11,
            "active_channels": [ORDER[0]],
            "command_metrics": {"wheel_integral_rad": {}},
            "completion_event": f"done_{phase_id}",
            "reference_actual": {
                "active_window_average_abs_velocity": {ORDER[0]: 10.0},
                "peak_abs_velocity": {ORDER[0]: 10.0},
                "actual_wheel_integral_rad": {},
                "trajectory_samples_normalized": [
                    {"progress": 0.0, "actual_full12": start},
                    {"progress": 0.5, "actual_full12": [start_value + 5.0] + [0.0] * 11},
                    {"progress": 1.0, "actual_full12": end},
                ],
            },
            "reference_result_observation": {
                "actual_end_full12": end,
                "actual_delta_from_motion_start_full12": [10.0] + [0.0] * 11,
            },
        })
        transitions.extend([
            {"state_id": phase_id, "sim_time_s": start_time,
             "from_lifecycle": "WAIT_ENTRY", "to_lifecycle": "EXECUTE_MOTION"},
            {"state_id": phase_id, "sim_time_s": end_time,
             "from_lifecycle": "EXECUTE_MOTION", "to_lifecycle": "VERIFY_RESULT"},
            {"state_id": phase_id, "sim_time_s": completion_time,
             "from_lifecycle": "VERIFY_RESULT", "to_lifecycle": "DONE"},
        ])
        for fraction in (0.0, 0.5, 1.0):
            when = start_time + fraction
            value = start_value + 10.0 * fraction
            full12 = [value] + [0.0] * 11
            commands.append({"state_id": phase_id, "sim_time_s": when, "full12": full12})
            observations.append({
                "state_id": phase_id, "simulation_time_s": when,
                "physics_tick": int(when * 120), "actual_full12": full12,
                "commanded_full12": full12,
                "velocity_full12": [10.0] + [0.0] * 11,
                "body_collision": {"detected": False}, "guards": {},
            })
        if phase_id == "P13":
            for tick in range(1, 72):
                when = end_time + tick / 120.0
                observations.append({
                    "state_id": phase_id,
                    "simulation_time_s": when,
                    "physics_tick": int(round(when * 120)),
                    "actual_full12": end,
                    "commanded_full12": end,
                    "velocity_full12": (
                        [10.0] + [0.0] * 11 if tick == 1 else [0.0] * 12
                    ),
                    "body_collision": {"detected": False},
                    "guards": {},
                })
        observations.append({
            "state_id": phase_id, "simulation_time_s": completion_time,
            "physics_tick": int(completion_time * 120), "actual_full12": end,
            "commanded_full12": end,
            "velocity_full12": [0.0] * 12, "body_collision": {"detected": False},
            "guards": {},
        })
    contract = {"schema": "test", "full12_order": list(ORDER), "phases": phases}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    _write_jsonl(run / JSONL_FILES["observation"], observations)
    _write_jsonl(run / JSONL_FILES["command"], commands)
    _write_jsonl(run / JSONL_FILES["transition"], transitions)
    _write_jsonl(run / JSONL_FILES["task_event"], [{"event": "TERMINATION", "result": "SUCCESS"}])
    _write_jsonl(run / JSONL_FILES["body_contact"], [])
    leg_rows = []
    clocks = {"FR": (1.0, 2.0, 3.0), "FL": (3.5, 4.0, 5.0),
              "RR": (6.0, 7.0, 8.0), "RL": (8.5, 9.0, 10.0)}
    for leg, values in clocks.items():
        for event, when in zip(("ACTIVE_LIFT", "FRONT_FACE_CROSSED", "TOP_LOADED"), values):
            leg_rows.append({"leg": leg, "event": event, "simulation_time_s": when})
    _write_jsonl(run / JSONL_FILES["leg_crossing"], leg_rows)
    _write_jsonl(run / JSONL_FILES["decision"], [{"simulation_time_s": 0.0}])
    (run / "reference_similarity.csv").write_text("phase\n", encoding="utf-8")
    (run / "actual_viewport_video.mp4").write_bytes(b"one-continuous-run")
    manifest = {
        "schema": "wlr50_clean.trial_manifest.v1",
        "success_evidence": {
            "task_result": "SUCCESS", "one_continuous_physical_fsm_success": True,
            "completed_macro_phases": list(PHASE_IDS), "body_collision": False,
            "wheel_only_climb": False, "rear_leg_order": "RR_FIRST",
            "root_state_write_count": 0, "teleport_count": 0,
            "external_force_count": 0, "external_impulse_count": 0,
            "runtime_raw_recording_access": False,
        },
    }
    (run / "trial_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run, contract_path


def test_analyzer_populates_all_phases_and_maximum_summary(tmp_path: Path) -> None:
    run, contract = _make_success(tmp_path)
    analysis = analyze_trial(run, contract)
    assert analysis["checks"]["p01_p13_completed"] is True
    assert len(analysis["similarity_rows"]) == 13
    assert analysis["conformance_summary"]["maximum_endpoint_error_percent"] == 0.0
    assert analysis["conformance_summary"]["all_normal_states_within_15_percent"] is True


def test_similarity_population_must_precede_immutable_manifest(tmp_path: Path) -> None:
    run, contract = _make_success(tmp_path)
    with pytest.raises(TrialAnalysisError, match="after trial_manifest"):
        populate_reference_similarity(run, contract)
    (run / "trial_manifest.json").unlink()
    populate_reference_similarity(run, contract)
    with (run / "reference_similarity.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 13
    assert {row["phase"] for row in rows} == set(PHASE_IDS)


def test_publication_is_fail_closed_and_emits_required_final_data(tmp_path: Path) -> None:
    run, contract = _make_success(tmp_path)
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps({"reference_version": "v010"}), encoding="utf-8")
    derivation = tmp_path / "STATE_DERIVATION.md"
    derivation.write_text("# State derivation\n", encoding="utf-8")
    output = tmp_path / "final"
    result = publish_successful_trial(
        run_dir=run, output_dir=output, contract_path=contract,
        selected_reference_path=selected, state_derivation_path=derivation,
    )
    assert result["status"] == "PASS"
    assert all((output / name).is_file() for name in FINAL_DATA_NAMES)
    manifest = json.loads((output / "successful_trial_manifest.json").read_text())
    assert manifest["publication"]["conformance_summary"]["conformance_row_count"] == 13

    source_manifest = json.loads((run / "trial_manifest.json").read_text())
    source_manifest["success_evidence"]["body_collision"] = True
    (run / "trial_manifest.json").write_text(json.dumps(source_manifest))
    with pytest.raises(TrialAnalysisError, match="body_collision_false"):
        publish_successful_trial(
            run_dir=run, output_dir=output, contract_path=contract,
            selected_reference_path=selected, state_derivation_path=derivation,
        )


def test_wheel_only_evidence_is_rejected(tmp_path: Path) -> None:
    run, contract = _make_success(tmp_path)
    rows = [json.loads(line) for line in (run / JSONL_FILES["leg_crossing"]).read_text().splitlines()]
    for row in rows:
        if row["leg"] == "RR" and row["event"] == "ACTIVE_LIFT":
            row["simulation_time_s"] = 7.5
    _write_jsonl(run / JSONL_FILES["leg_crossing"], rows)
    with pytest.raises(TrialAnalysisError, match="wheel_only_climb_false"):
        analyze_trial(run, contract)


def test_sequence_recovery_corrections_are_bounded_at_15_percent(tmp_path: Path) -> None:
    run, contract = _make_success(tmp_path)
    path = run / JSONL_FILES["transition"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append({
        "state_id": "P01", "sim_time_s": 1.05,
        "from_lifecycle": "RECOVERY", "to_lifecycle": "EXECUTE_MOTION",
        "details": {"correction_fractions": [0.15] * 12},
    })
    _write_jsonl(path, rows)
    assert analyze_trial(run, contract)["checks"]["feedback_correction_reference_bounded"] is True

    rows[-1]["details"]["correction_fractions"][4] = 0.151
    _write_jsonl(path, rows)
    with pytest.raises(TrialAnalysisError, match="feedback_correction_reference_bounded"):
        analyze_trial(run, contract)


def test_post_mapper_drive_feedback_is_included_in_cumulative_budget(tmp_path: Path) -> None:
    command = {
        "state_id": "P09",
        "drive_feedback": {
            "just_triggered": True,
            "correction_channel_index": 5,
            "cumulative_fraction_of_reference": 2.5 / 19.4,
        },
    }
    values, _ = _recovery_evidence([], [command])
    assert max(values) == pytest.approx(2.5 / 19.4)

    duplicate = json.loads(json.dumps(command))
    values, _ = _recovery_evidence([], [command, duplicate])
    assert max(values) == pytest.approx(5.0 / 19.4)
    assert max(values) > 0.15


def _feedback_ledger_rows(*, residual_after_window: float = 0.0) -> list[dict]:
    peak = 1.25 / 19.4
    cumulative = 2.5 / 19.4
    rows = []
    for tick in range(862, 872):
        latched = tick >= 863
        active = 864 <= tick <= 871
        reference = {
            862: -51.50794810030658,
            863: -51.549332216487684,
        }.get(tick)
        observed = None if reference is None else reference - 1.7
        requested = [0.0] * 12
        realized = [0.0] * 12
        if active:
            requested[5] = 1.25
            realized[5] = 1.25
        native = [0.0] * 12
        final = [left + right for left, right in zip(native, realized, strict=True)]
        rows.append({
            "state_id": "P09",
            "sim_time_s": tick / 120.0,
            "motion_tick_index": tick,
            "drive_feedback": {
                "schema": "wlr50_clean.drive_feedback.v1",
                "bias_full12": requested,
                "active": active,
                "just_triggered": tick == 863,
                "tick_index": tick,
                "trigger_tick": 863 if latched else None,
                "observed_deg": observed,
                "reference_deg": reference,
                "peak_fraction_of_reference": peak if latched else 0.0,
                "cumulative_fraction_of_reference": cumulative if latched else 0.0,
                "probe_channel": "rear_right_knee",
                "probe_channel_index": 7,
                "correction_channel": "rear_left_knee",
                "correction_channel_index": 5,
            },
            "drive_feedback_bias_requested_full12": requested,
            "drive_feedback_bias_realized_full12": realized,
            "native_drive_target_full12": native,
            "drive_target_full12": final,
        })
    requested = [0.0] * 12
    realized = [0.0] * 12
    realized[5] = residual_after_window
    native = [0.0] * 12
    final = [left + right for left, right in zip(native, realized, strict=True)]
    rows.append(
        {
            "state_id": "P10",
            "sim_time_s": 872 / 120.0,
            "motion_tick_index": None,
            "drive_feedback": {
                "schema": "wlr50_clean.drive_feedback.v1",
                "bias_full12": requested,
                "active": False,
                "just_triggered": False,
                "tick_index": None,
                "trigger_tick": None,
                "peak_fraction_of_reference": 0.0,
                "cumulative_fraction_of_reference": 0.0,
                "probe_channel": None,
                "probe_channel_index": None,
                "correction_channel": None,
                "correction_channel_index": None,
            },
            "drive_feedback_bias_requested_full12": requested,
            "drive_feedback_bias_realized_full12": realized,
            "native_drive_target_full12": native,
            "drive_target_full12": final,
        }
    )
    return rows


def test_drive_feedback_ledger_enforces_window_and_endpoint_restoration() -> None:
    contract = json.loads(
        (ROOT / "configs" / "recording_motion_contract.json").read_text(
            encoding="utf-8"
        )
    )
    valid_rows = _feedback_ledger_rows()
    observations = []
    for row in valid_rows:
        actual = [0.0] * 12
        observed = row["drive_feedback"].get("observed_deg")
        if observed is not None:
            actual[7] = observed
        observations.append(
            {"simulation_time_s": row["sim_time_s"], "actual_full12": actual}
        )
    assert _drive_feedback_ledger_valid(valid_rows, contract, observations)
    mismatched_observations = json.loads(json.dumps(observations))
    mismatched_observations[0]["actual_full12"][7] += 0.01
    assert not _drive_feedback_ledger_valid(
        valid_rows, contract, mismatched_observations
    )
    assert not _drive_feedback_ledger_valid(
        _feedback_ledger_rows(residual_after_window=0.4), contract
    )
    missing_active_tick = _feedback_ledger_rows()
    del missing_active_tick[6]
    assert not _drive_feedback_ledger_valid(missing_active_tick, contract)
    unrealized_active = _feedback_ledger_rows()
    for row in unrealized_active:
        if row["drive_feedback"]["active"]:
            row["drive_feedback_bias_realized_full12"][5] = 0.0
            row["drive_target_full12"][5] = row["native_drive_target_full12"][5]
    assert not _drive_feedback_ledger_valid(unrealized_active, contract)
    wrong_cadence = _feedback_ledger_rows()
    base_time = wrong_cadence[0]["sim_time_s"]
    for index, row in enumerate(wrong_cadence[:-1]):
        row["sim_time_s"] = base_time + index / 240.0
    wrong_cadence[-1]["sim_time_s"] = (
        wrong_cadence[-2]["sim_time_s"] + 1.0 / 120.0
    )
    assert not _drive_feedback_ledger_valid(wrong_cadence, contract)
    mislabeled_teardown = _feedback_ledger_rows()
    teardown = mislabeled_teardown[-1]
    teardown["state_id"] = "P09"
    teardown["motion_tick_index"] = 873
    teardown["drive_feedback"].update(
        {
            "tick_index": 873,
            "trigger_tick": 863,
            "peak_fraction_of_reference": 1.25 / 19.4,
            "cumulative_fraction_of_reference": 2.5 / 19.4,
            "probe_channel": "rear_right_knee",
            "probe_channel_index": 7,
            "correction_channel": "rear_left_knee",
            "correction_channel_index": 5,
        }
    )
    assert not _drive_feedback_ledger_valid(mislabeled_teardown, contract)
    missing_probe_evidence = _feedback_ledger_rows()
    missing_probe_evidence[0]["drive_feedback"]["observed_deg"] = None
    assert not _drive_feedback_ledger_valid(missing_probe_evidence, contract)
    missing_restore = _feedback_ledger_rows()
    del missing_restore[-1]["drive_feedback"]
    assert not _drive_feedback_ledger_valid(missing_restore, contract)
    assert not _drive_feedback_ledger_valid(
        [{"state_id": "P09", "motion_tick_index": 760}], contract
    )


def test_writer_api_remains_streaming_and_never_reuses_directory(tmp_path: Path) -> None:
    run = tmp_path / "writer"
    with TrialArtifactWriter(run) as writer:
        writer.append("task_event", {"result": "SUCCESS"})
        writer.append_similarity({"phase": "P01"})
    with pytest.raises(FileExistsError):
        TrialArtifactWriter(run)
