from __future__ import annotations

import csv
import json
import math
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


def test_normal_shaping_and_actual_retry_vectors_share_one_cumulative_budget() -> None:
    normal = [0.0] * 12
    normal[10] = -0.149
    recovery = [0.0] * 12
    transitions = [
        {
            "state_id": "P03",
            "from_lifecycle": "WAIT_ENTRY",
            "to_lifecycle": "EXECUTE_MOTION",
            "details": {"correction_fractions": normal},
        },
        {
            "state_id": "P03",
            "from_lifecycle": "RECOVERY",
            "to_lifecycle": "EXECUTE_MOTION",
            "details": {
                "correction_fractions": recovery.copy(),
                "recovery_correction_fractions": recovery.copy(),
            },
        },
    ]

    values, retry_counts = _recovery_evidence(transitions)

    assert max(values) == pytest.approx(0.149)
    assert retry_counts["P03"] == 1
    transitions[-1]["details"]["correction_fractions"][10] = -0.001
    transitions[-1]["details"]["recovery_correction_fractions"][10] = -0.001
    values, _ = _recovery_evidence(transitions)
    assert max(values) == pytest.approx(0.15)

    transitions[-1]["details"]["correction_fractions"][10] = -0.002
    transitions[-1]["details"]["recovery_correction_fractions"][10] = -0.002
    values, _ = _recovery_evidence(transitions)
    assert max(values) == pytest.approx(0.151)

    transitions[-1]["details"]["recovery_correction_fractions"][10] = -0.001
    values, _ = _recovery_evidence(transitions)
    assert math.isinf(max(values))


def _contract_with_feedback(spec: dict) -> dict:
    contract = json.loads(
        (ROOT / "configs" / "recording_motion_contract.json").read_text(
            encoding="utf-8"
        )
    )
    p09 = next(phase for phase in contract["phases"] if phase["state_id"] == "P09")
    p09["drive_feedback"] = spec
    return contract


def _servo_feedback_contract() -> dict:
    return _contract_with_feedback(
        {
            "kind": "verify_tail_carry_alignment",
            "probe_channel": "rear_right_knee",
            "probe_channel_index": 7,
            "correction_channel": "rear_left_knee",
            "correction_channel_index": 5,
            "probe_samples": [
                {
                    "motion_tick": 870,
                    "reference_actual_deg": -50.8511468644344,
                },
                {
                    "motion_tick": 871,
                    "reference_actual_deg": -50.63441813188072,
                },
            ],
            "lag_threshold_deg": 2.4,
            "required_consecutive_samples": 2,
            "first_bias_tick": 874,
            "last_bias_tick": 879,
            "teardown_tick": 880,
            "logical_bias_deg": 1.25,
            "reference_excursion_deg": 19.4,
            "peak_fraction_of_reference": 1.25 / 19.4,
            "cumulative_fraction_of_reference": 2.5 / 19.4,
        }
    )


def _feedback_ledger_rows(*, residual_after_window: float = 0.0) -> list[dict]:
    peak = 1.25 / 19.4
    cumulative = 2.5 / 19.4
    rows = []
    for tick in range(870, 880):
        latched = tick >= 871
        active = 874 <= tick <= 879
        reference = {
            870: -50.8511468644344,
            871: -50.63441813188072,
        }.get(tick)
        observed = None if reference is None else reference - 2.4
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
                "just_triggered": tick == 871,
                "tick_index": tick,
                "trigger_tick": 871 if latched else None,
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
            "sim_time_s": 880 / 120.0,
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
    contract = _servo_feedback_contract()
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
    suppressed_trigger = json.loads(json.dumps(valid_rows))
    for row in suppressed_trigger:
        if row["state_id"] != "P09":
            continue
        zeros = [0.0] * 12
        row["drive_feedback"].update(
            {
                "bias_full12": zeros,
                "active": False,
                "just_triggered": False,
                "trigger_tick": None,
                "peak_fraction_of_reference": 0.0,
                "cumulative_fraction_of_reference": 0.0,
            }
        )
        row["drive_feedback_bias_requested_full12"] = zeros
        row["drive_feedback_bias_realized_full12"] = zeros
        row["drive_target_full12"] = row["native_drive_target_full12"]
    assert not _drive_feedback_ledger_valid(
        suppressed_trigger, contract, observations
    )
    no_deficit_rows = json.loads(json.dumps(suppressed_trigger))
    no_deficit_observations = json.loads(json.dumps(observations))
    for row, observation in zip(
        no_deficit_rows, no_deficit_observations, strict=True
    ):
        reference = row["drive_feedback"].get("reference_deg")
        if reference is not None:
            row["drive_feedback"]["observed_deg"] = reference
            observation["actual_full12"][7] = reference
    assert _drive_feedback_ledger_valid(
        no_deficit_rows, contract, no_deficit_observations
    )
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
    teardown["motion_tick_index"] = 881
    teardown["drive_feedback"].update(
        {
            "tick_index": 881,
            "trigger_tick": 871,
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


def test_drive_feedback_trigger_is_consumed_across_same_phase_retry() -> None:
    contract = _servo_feedback_contract()
    first_attempt = _feedback_ledger_rows()
    teardown = first_attempt[-1]
    teardown["state_id"] = "P09"
    teardown["motion_tick_index"] = 880
    teardown["drive_feedback"].update(
        {
            "tick_index": 880,
            "trigger_tick": 871,
            "peak_fraction_of_reference": 1.25 / 19.4,
            "cumulative_fraction_of_reference": 2.5 / 19.4,
            "probe_channel": "rear_right_knee",
            "probe_channel_index": 7,
            "correction_channel": "rear_left_knee",
            "correction_channel_index": 5,
        }
    )
    retry = json.loads(json.dumps(_feedback_ledger_rows()[:-1]))
    for row in retry:
        zeros = [0.0] * 12
        row["drive_feedback"].update(
            {
                "bias_full12": zeros,
                "active": False,
                "just_triggered": False,
                "trigger_tick": None,
                "peak_fraction_of_reference": 0.0,
                "cumulative_fraction_of_reference": 0.0,
            }
        )
        row["drive_feedback_bias_requested_full12"] = zeros
        row["drive_feedback_bias_realized_full12"] = zeros
        row["drive_target_full12"] = row["native_drive_target_full12"]
    tick_zero = json.loads(json.dumps(retry[0]))
    tick_zero["motion_tick_index"] = 0
    tick_zero["drive_feedback"].update(
        {
            "tick_index": 0,
            "observed_deg": 0.0,
            "reference_deg": None,
        }
    )
    retry.insert(0, tick_zero)
    start_time = teardown["sim_time_s"] + 1.0 / 120.0
    for index, row in enumerate(retry):
        row["sim_time_s"] = start_time + index / 120.0
    rows = first_attempt + retry
    observations = []
    for row in rows:
        actual = [0.0] * 12
        observed = row["drive_feedback"].get("observed_deg")
        if observed is not None:
            actual[7] = observed
        observations.append(
            {"simulation_time_s": row["sim_time_s"], "actual_full12": actual}
        )
    assert _drive_feedback_ledger_valid(rows, contract, observations)


_WHEEL_TAIL_VELOCITY = -1.07
_WHEEL_REFERENCE_INTEGRAL = -0.9060000000012605
_WHEEL_ADDITIONAL_INTEGRAL = _WHEEL_TAIL_VELOCITY * 8.0 / 120.0
_WHEEL_INTEGRAL_FRACTION = abs(_WHEEL_ADDITIONAL_INTEGRAL) / abs(
    _WHEEL_REFERENCE_INTEGRAL
)


def _wheel_feedback_spec() -> dict:
    return {
        "kind": "verify_tail_wheel_carry_alignment",
        "probe_channel": "rear_right_knee",
        "probe_channel_index": 7,
        "correction_channel": "front_left_ankle",
        "correction_channel_index": 8,
        "probe_samples": [
            {"motion_tick": 858, "reference_actual_deg": -51.055799822535},
            {"motion_tick": 859, "reference_actual_deg": -51.191638624749},
        ],
        "lag_threshold_deg": 1.7,
        "required_consecutive_samples": 2,
        "first_bias_tick": 864,
        "last_bias_tick": 871,
        "teardown_tick": 872,
        "logical_bias_rad_s": _WHEEL_TAIL_VELOCITY,
        "reference_wheel_integral_rad": _WHEEL_REFERENCE_INTEGRAL,
        "additional_wheel_integral_rad": _WHEEL_ADDITIONAL_INTEGRAL,
        "cumulative_fraction_of_reference": _WHEEL_INTEGRAL_FRACTION,
    }


def _wheel_feedback_contract() -> dict:
    return _contract_with_feedback(_wheel_feedback_spec())


def _wheel_feedback_ledger_rows(
    *, residual_after_window: float = 0.0
) -> list[dict]:
    rows = []
    references = {
        858: -51.055799822535,
        859: -51.191638624749,
    }
    for tick in range(858, 872):
        latched = tick >= 859
        active = 864 <= tick <= 871
        reference = references.get(tick)
        observed = None if reference is None else reference - 1.7
        requested = [0.0] * 12
        realized = [0.0] * 12
        if active:
            requested[8] = _WHEEL_TAIL_VELOCITY
            realized[8] = _WHEEL_TAIL_VELOCITY
        native = [0.0] * 12
        if tick < 864:
            native[8] = _WHEEL_TAIL_VELOCITY
        final = [
            left + right for left, right in zip(native, realized, strict=True)
        ]
        rows.append(
            {
                "state_id": "P09",
                "sim_time_s": tick / 120.0,
                "motion_tick_index": tick,
                "drive_feedback": {
                    "schema": "wlr50_clean.drive_feedback.v1",
                    "bias_full12": requested,
                    "active": active,
                    "just_triggered": tick == 859,
                    "tick_index": tick,
                    "trigger_tick": 859 if latched else None,
                    "observed_deg": observed,
                    "reference_deg": reference,
                    "peak_fraction_of_reference": 0.0,
                    "cumulative_fraction_of_reference": (
                        _WHEEL_INTEGRAL_FRACTION if latched else 0.0
                    ),
                    "logical_bias_rad_s": _WHEEL_TAIL_VELOCITY,
                    "reference_wheel_integral_rad": _WHEEL_REFERENCE_INTEGRAL,
                    "additional_wheel_integral_rad": (
                        _WHEEL_ADDITIONAL_INTEGRAL
                    ),
                    "probe_channel": "rear_right_knee",
                    "probe_channel_index": 7,
                    "correction_channel": "front_left_ankle",
                    "correction_channel_index": 8,
                },
                "drive_feedback_bias_requested_full12": requested,
                "drive_feedback_bias_realized_full12": realized,
                "native_drive_target_full12": native,
                "drive_target_full12": final,
            }
        )
    requested = [0.0] * 12
    realized = [0.0] * 12
    realized[8] = residual_after_window
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
                "observed_deg": None,
                "reference_deg": None,
                "peak_fraction_of_reference": 0.0,
                "cumulative_fraction_of_reference": 0.0,
                "logical_bias_rad_s": 0.0,
                "reference_wheel_integral_rad": 0.0,
                "additional_wheel_integral_rad": 0.0,
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


def _feedback_observations(rows: list[dict]) -> list[dict]:
    observations = []
    for row in rows:
        actual = [0.0] * 12
        observed = row["drive_feedback"].get("observed_deg")
        if observed is not None:
            actual[7] = observed
        observations.append(
            {"simulation_time_s": row["sim_time_s"], "actual_full12": actual}
        )
    return observations


def test_wheel_tail_feedback_ledger_accepts_exact_sensor_causal_carry() -> None:
    contract = _wheel_feedback_contract()
    rows = _wheel_feedback_ledger_rows()
    assert _drive_feedback_ledger_valid(
        rows, contract, _feedback_observations(rows)
    )


def test_wheel_tail_feedback_ledger_rederives_mandatory_trigger() -> None:
    contract = _wheel_feedback_contract()
    rows = _wheel_feedback_ledger_rows()
    observations = _feedback_observations(rows)
    for row in rows[:-1]:
        zeros = [0.0] * 12
        row["drive_feedback"].update(
            {
                "bias_full12": zeros,
                "active": False,
                "just_triggered": False,
                "trigger_tick": None,
                "cumulative_fraction_of_reference": 0.0,
            }
        )
        row["drive_feedback_bias_requested_full12"] = zeros
        row["drive_feedback_bias_realized_full12"] = zeros
        row["drive_target_full12"] = row["native_drive_target_full12"]
    assert not _drive_feedback_ledger_valid(rows, contract, observations)

    for row, observation in zip(rows[:-1], observations[:-1], strict=True):
        reference = row["drive_feedback"].get("reference_deg")
        if reference is not None:
            row["drive_feedback"]["observed_deg"] = reference
            observation["actual_full12"][7] = reference
    assert _drive_feedback_ledger_valid(rows, contract, observations)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_active_tick",
        "wrong_active_value",
        "other_channel_correction",
        "final_not_native_plus_realized",
        "teardown_residual",
        "wrong_cadence",
        "hard_wheel_limit",
        "logged_integral_mismatch",
    ),
)
def test_wheel_tail_feedback_ledger_fails_closed(mutation: str) -> None:
    contract = _wheel_feedback_contract()
    rows = _wheel_feedback_ledger_rows()
    if mutation == "missing_active_tick":
        del rows[9]  # tick 867
    elif mutation == "wrong_active_value":
        active = rows[6]  # tick 864
        active["drive_feedback"]["bias_full12"][8] = -1.06
        active["drive_feedback_bias_requested_full12"][8] = -1.06
        active["drive_feedback_bias_realized_full12"][8] = -1.06
        active["drive_target_full12"][8] = -1.06
    elif mutation == "other_channel_correction":
        active = rows[6]
        active["drive_feedback"]["bias_full12"][9] = 0.01
        active["drive_feedback_bias_requested_full12"][9] = 0.01
        active["drive_feedback_bias_realized_full12"][9] = 0.01
        active["drive_target_full12"][9] = 0.01
    elif mutation == "final_not_native_plus_realized":
        rows[6]["drive_target_full12"][8] += 0.01
    elif mutation == "teardown_residual":
        teardown = rows[-1]
        teardown["drive_feedback_bias_realized_full12"][8] = 0.01
        teardown["drive_target_full12"][8] = 0.01
    elif mutation == "wrong_cadence":
        rows[8]["sim_time_s"] += 1.0 / 240.0
    elif mutation == "hard_wheel_limit":
        rows[0]["native_drive_target_full12"][9] = 2.2
        rows[0]["drive_target_full12"][9] = 2.2
    elif mutation == "logged_integral_mismatch":
        rows[0]["drive_feedback"]["additional_wheel_integral_rad"] += 0.01
    assert not _drive_feedback_ledger_valid(
        rows, contract, _feedback_observations(rows)
    )


def test_wheel_tail_feedback_contract_timing_and_integral_are_pinned() -> None:
    rows = _wheel_feedback_ledger_rows()
    observations = _feedback_observations(rows)

    wrong_timing = _wheel_feedback_contract()
    wrong_timing["phases"][8]["drive_feedback"]["first_bias_tick"] = 863
    assert not _drive_feedback_ledger_valid(rows, wrong_timing, observations)

    over_budget = _wheel_feedback_contract()
    spec = over_budget["phases"][8]["drive_feedback"]
    spec["reference_wheel_integral_rad"] = -0.4
    spec["cumulative_fraction_of_reference"] = abs(
        spec["additional_wheel_integral_rad"]
    ) / abs(spec["reference_wheel_integral_rad"])
    assert spec["cumulative_fraction_of_reference"] > 0.15
    assert not _drive_feedback_ledger_valid(rows, over_budget, observations)

    inflated_reference = _wheel_feedback_contract()
    spec = inflated_reference["phases"][8]["drive_feedback"]
    spec["reference_wheel_integral_rad"] *= 2.0
    spec["cumulative_fraction_of_reference"] = abs(
        spec["additional_wheel_integral_rad"]
    ) / abs(spec["reference_wheel_integral_rad"])
    inflated_reference["phases"][8]["command_metrics"]["wheel_integral_rad"][
        "front_left_ankle"
    ] = spec["reference_wheel_integral_rad"]
    inflated_rows = _wheel_feedback_ledger_rows()
    for row in inflated_rows[:-1]:
        row["drive_feedback"]["reference_wheel_integral_rad"] = spec[
            "reference_wheel_integral_rad"
        ]
        row["drive_feedback"]["cumulative_fraction_of_reference"] = (
            spec["cumulative_fraction_of_reference"]
            if row["drive_feedback"]["trigger_tick"] is not None
            else 0.0
        )
    assert not _drive_feedback_ledger_valid(
        inflated_rows,
        inflated_reference,
        _feedback_observations(inflated_rows),
    )

    unknown_kind = _wheel_feedback_contract()
    unknown_kind["phases"][8]["drive_feedback"]["kind"] = "wheel_tail"
    assert not _drive_feedback_ledger_valid(rows, unknown_kind, observations)


_WHEEL_REBOUND_SEGMENTS = [
    {
        "first_bias_tick": 860,
        "last_bias_tick": 860,
        "logical_bias_rad_s": 0.68,
    },
    {
        "first_bias_tick": 861,
        "last_bias_tick": 861,
        "logical_bias_rad_s": 0.33,
    },
    {
        "first_bias_tick": 862,
        "last_bias_tick": 864,
        "logical_bias_rad_s": 0.68,
    },
    {
        "first_bias_tick": 865,
        "last_bias_tick": 865,
        "logical_bias_rad_s": 0.33,
    },
    {
        "first_bias_tick": 866,
        "last_bias_tick": 867,
        "logical_bias_rad_s": 1.03,
    },
    {
        "first_bias_tick": 868,
        "last_bias_tick": 869,
        "logical_bias_rad_s": 0.73,
    },
    {
        "first_bias_tick": 870,
        "last_bias_tick": 870,
        "logical_bias_rad_s": 0.01,
    },
    {
        "first_bias_tick": 871,
        "last_bias_tick": 871,
        "logical_bias_rad_s": 1.07,
    },
]
_WHEEL_REBOUND_ADDITIONAL_INTEGRAL = 0.0665
_WHEEL_REBOUND_RESULTING_INTEGRAL = -0.8395000000012605
_WHEEL_REBOUND_FRACTION = 0.07339955849879413


def _wheel_rebound_segment_index(tick: int) -> int | None:
    return next(
        (
            index
            for index, segment in enumerate(_WHEEL_REBOUND_SEGMENTS)
            if segment["first_bias_tick"] <= tick <= segment["last_bias_tick"]
        ),
        None,
    )


def _wheel_rebound_feedback_spec() -> dict:
    return {
        "kind": "pre_endpoint_wheel_rebound_alignment",
        "probe_channel": "rear_right_knee",
        "probe_channel_index": 7,
        "correction_channel": "front_left_ankle",
        "correction_channel_index": 8,
        "probe_samples": [
            {"motion_tick": 858, "reference_actual_deg": -51.055799822535},
            {"motion_tick": 859, "reference_actual_deg": -51.191638624749},
        ],
        "lag_threshold_deg": 0.35,
        "required_consecutive_samples": 2,
        "bias_segments": json.loads(json.dumps(_WHEEL_REBOUND_SEGMENTS)),
        "teardown_tick": 872,
        "reference_wheel_integral_rad": _WHEEL_REFERENCE_INTEGRAL,
        "additional_wheel_integral_rad": _WHEEL_REBOUND_ADDITIONAL_INTEGRAL,
        "resulting_wheel_integral_rad": _WHEEL_REBOUND_RESULTING_INTEGRAL,
        "reference_wheel_peak_abs_rad_s": 1.07,
        "resulting_wheel_peak_abs_rad_s": 1.07,
        "instantaneous_direction_reversal": True,
        "cumulative_fraction_of_reference": _WHEEL_REBOUND_FRACTION,
        "nominal_endpoint_restored": True,
        "raw_recording_runtime_access_required": False,
    }


def _wheel_rebound_feedback_contract() -> dict:
    return _contract_with_feedback(_wheel_rebound_feedback_spec())


def _wheel_rebound_atomic_ack(
    native: list[float],
    final: list[float],
    requested: list[float],
    realized: list[float],
) -> dict:
    return {
        "schema": "wlr50_clean.atomic_full12_ack.v1",
        "canonical_order": list(ORDER),
        "articulation_writes_this_call": 1,
        "motion_start_skew_s": 0.0,
        "command_was_clamped": False,
        "requested_full12": list(native),
        "applied_full12": list(native),
        "native_drive_target_full12": list(native),
        "drive_target_full12": list(final),
        "drive_feedback_bias_requested_full12": list(requested),
        "drive_feedback_bias_realized_full12": list(realized),
        "wheel_target_physical_rad_s": [
            -final[8],
            final[9],
            -final[10],
            final[11],
        ],
    }


def _sync_wheel_rebound_atomic_ack(row: dict) -> None:
    row["atomic_ack"] = _wheel_rebound_atomic_ack(
        row["native_drive_target_full12"],
        row["drive_target_full12"],
        row["drive_feedback_bias_requested_full12"],
        row["drive_feedback_bias_realized_full12"],
    )


def _wheel_rebound_feedback_ledger_rows(
    *, residual_after_window: float = 0.0
) -> list[dict]:
    rows = []
    references = {
        858: -51.055799822535,
        859: -51.191638624749,
    }
    for tick in range(858, 872):
        latched = tick >= 859
        segment_index = _wheel_rebound_segment_index(tick)
        active = segment_index is not None
        logical_bias = (
            _WHEEL_REBOUND_SEGMENTS[segment_index]["logical_bias_rad_s"]
            if segment_index is not None
            else 0.0
        )
        reference = references.get(tick)
        observed = None if reference is None else reference - 0.35
        requested = [0.0] * 12
        realized = [0.0] * 12
        if active:
            requested[8] = logical_bias
            realized[8] = logical_bias
        native = [0.0] * 12
        if tick < 864:
            native[8] = _WHEEL_TAIL_VELOCITY
        final = [
            left + right for left, right in zip(native, realized, strict=True)
        ]
        rows.append(
            {
                "state_id": "P09",
                "sim_time_s": tick / 120.0,
                "motion_tick_index": tick,
                "drive_feedback": {
                    "schema": "wlr50_clean.drive_feedback.v1",
                    "kind": "pre_endpoint_wheel_rebound_alignment",
                    "bias_full12": requested,
                    "active": active,
                    "just_triggered": tick == 859,
                    "tick_index": tick,
                    "trigger_tick": 859 if latched else None,
                    "observed_deg": observed,
                    "reference_deg": reference,
                    "bias_segments": json.loads(
                        json.dumps(_WHEEL_REBOUND_SEGMENTS)
                    ),
                    "active_segment_index": segment_index,
                    "active_segment_first_bias_tick": (
                        None
                        if segment_index is None
                        else _WHEEL_REBOUND_SEGMENTS[segment_index][
                            "first_bias_tick"
                        ]
                    ),
                    "active_segment_last_bias_tick": (
                        None
                        if segment_index is None
                        else _WHEEL_REBOUND_SEGMENTS[segment_index][
                            "last_bias_tick"
                        ]
                    ),
                    # No absolute peak-magnitude increase; reversal is audited
                    # independently and must never be inferred from this zero.
                    "peak_fraction_of_reference": 0.0,
                    "cumulative_fraction_of_reference": (
                        _WHEEL_REBOUND_FRACTION if latched else 0.0
                    ),
                    "logical_bias_rad_s": logical_bias,
                    "reference_wheel_integral_rad": _WHEEL_REFERENCE_INTEGRAL,
                    "additional_wheel_integral_rad": (
                        _WHEEL_REBOUND_ADDITIONAL_INTEGRAL
                    ),
                    "resulting_wheel_integral_rad": (
                        _WHEEL_REBOUND_RESULTING_INTEGRAL
                    ),
                    "reference_wheel_peak_abs_rad_s": 1.07,
                    "resulting_wheel_peak_abs_rad_s": 1.07,
                    "instantaneous_direction_reversal": True,
                    "probe_channel": "rear_right_knee",
                    "probe_channel_index": 7,
                    "correction_channel": "front_left_ankle",
                    "correction_channel_index": 8,
                },
                "drive_feedback_bias_requested_full12": requested,
                "drive_feedback_bias_realized_full12": realized,
                "native_drive_target_full12": native,
                "drive_target_full12": final,
                "atomic_ack": _wheel_rebound_atomic_ack(
                    native, final, requested, realized
                ),
            }
        )
    requested = [0.0] * 12
    realized = [0.0] * 12
    realized[8] = residual_after_window
    native = [0.0] * 12
    final = [left + right for left, right in zip(native, realized, strict=True)]
    rows.append(
        {
            "state_id": "P10",
            "sim_time_s": 872 / 120.0,
            "motion_tick_index": None,
            "drive_feedback": {
                "schema": "wlr50_clean.drive_feedback.v1",
                "kind": None,
                "bias_full12": requested,
                "active": False,
                "just_triggered": False,
                "tick_index": None,
                "trigger_tick": None,
                "observed_deg": None,
                "reference_deg": None,
                "bias_segments": [],
                "active_segment_index": None,
                "active_segment_first_bias_tick": None,
                "active_segment_last_bias_tick": None,
                "peak_fraction_of_reference": 0.0,
                "cumulative_fraction_of_reference": 0.0,
                "logical_bias_rad_s": 0.0,
                "reference_wheel_integral_rad": 0.0,
                "additional_wheel_integral_rad": 0.0,
                "resulting_wheel_integral_rad": 0.0,
                "reference_wheel_peak_abs_rad_s": 0.0,
                "resulting_wheel_peak_abs_rad_s": 0.0,
                "instantaneous_direction_reversal": False,
                "probe_channel": None,
                "probe_channel_index": None,
                "correction_channel": None,
                "correction_channel_index": None,
            },
            "drive_feedback_bias_requested_full12": requested,
            "drive_feedback_bias_realized_full12": realized,
            "native_drive_target_full12": native,
            "drive_target_full12": final,
            "atomic_ack": _wheel_rebound_atomic_ack(
                native, final, requested, realized
            ),
        }
    )
    return rows


def test_wheel_rebound_feedback_accepts_exact_partial_counteraction_and_reversal(
) -> None:
    contract = _wheel_rebound_feedback_contract()
    rows = _wheel_rebound_feedback_ledger_rows()
    assert _drive_feedback_ledger_valid(
        rows, contract, _feedback_observations(rows)
    )
    spec = contract["phases"][8]["drive_feedback"]
    assert spec["bias_segments"] == _WHEEL_REBOUND_SEGMENTS
    assert spec["additional_wheel_integral_rad"] == pytest.approx(
        0.0665
    )
    assert spec["resulting_wheel_integral_rad"] == pytest.approx(
        -0.8395000000012605
    )
    assert spec["cumulative_fraction_of_reference"] == pytest.approx(
        0.07339955849879413
    )
    assert spec["reference_wheel_peak_abs_rad_s"] == 1.07
    assert spec["resulting_wheel_peak_abs_rad_s"] == 1.07
    expected_native_and_final = {
        860: (-1.07, -0.39),
        861: (-1.07, -0.74),
        862: (-1.07, -0.39),
        863: (-1.07, -0.39),
        864: (0.0, 0.68),
        865: (0.0, 0.33),
        866: (0.0, 1.03),
        867: (0.0, 1.03),
        868: (0.0, 0.73),
        869: (0.0, 0.73),
        870: (0.0, 0.01),
        871: (0.0, 1.07),
    }
    for row in rows[2:14]:
        tick = row["motion_tick_index"]
        assert tick is not None
        segment_index = _wheel_rebound_segment_index(tick)
        assert segment_index is not None
        segment = _WHEEL_REBOUND_SEGMENTS[segment_index]
        expected_bias = segment["logical_bias_rad_s"]
        expected_native, expected_final = expected_native_and_final[tick]
        assert row["native_drive_target_full12"][8] == pytest.approx(
            expected_native
        )
        assert row["drive_target_full12"][8] == pytest.approx(expected_final)
        assert row["drive_feedback"]["active_segment_index"] == segment_index
        assert row["drive_feedback"]["active_segment_first_bias_tick"] == segment[
            "first_bias_tick"
        ]
        assert row["drive_feedback"]["active_segment_last_bias_tick"] == segment[
            "last_bias_tick"
        ]
        assert row["atomic_ack"]["wheel_target_physical_rad_s"][0] == pytest.approx(
            -expected_final
        )
    teardown = rows[14]
    assert teardown["drive_feedback_bias_realized_full12"][8] == 0.0
    assert teardown["drive_target_full12"][8] == 0.0
    assert teardown["atomic_ack"]["wheel_target_physical_rad_s"][0] == 0.0


def test_wheel_rebound_feedback_rederives_mandatory_trigger() -> None:
    contract = _wheel_rebound_feedback_contract()
    rows = _wheel_rebound_feedback_ledger_rows()
    observations = _feedback_observations(rows)
    for row in rows[:-1]:
        zeros = [0.0] * 12
        row["drive_feedback"].update(
            {
                "bias_full12": zeros,
                "active": False,
                "just_triggered": False,
                "trigger_tick": None,
                "cumulative_fraction_of_reference": 0.0,
                "active_segment_index": None,
                "active_segment_first_bias_tick": None,
                "active_segment_last_bias_tick": None,
                "logical_bias_rad_s": 0.0,
            }
        )
        row["drive_feedback_bias_requested_full12"] = zeros
        row["drive_feedback_bias_realized_full12"] = zeros
        row["drive_target_full12"] = row["native_drive_target_full12"]
        _sync_wheel_rebound_atomic_ack(row)
    assert not _drive_feedback_ledger_valid(rows, contract, observations)

    for row, observation in zip(rows[:-1], observations[:-1], strict=True):
        reference = row["drive_feedback"].get("reference_deg")
        if reference is not None:
            row["drive_feedback"]["observed_deg"] = reference
            observation["actual_full12"][7] = reference
    assert _drive_feedback_ledger_valid(rows, contract, observations)


def test_wheel_rebound_trigger_is_consumed_across_same_phase_retry() -> None:
    contract = _wheel_rebound_feedback_contract()
    first_attempt = _wheel_rebound_feedback_ledger_rows()
    teardown = first_attempt[-1]
    teardown["state_id"] = "P09"
    teardown["motion_tick_index"] = 872
    teardown["drive_feedback"].update(
        {
            "kind": "pre_endpoint_wheel_rebound_alignment",
            "tick_index": 872,
            "trigger_tick": 859,
            "cumulative_fraction_of_reference": _WHEEL_REBOUND_FRACTION,
            "bias_segments": json.loads(json.dumps(_WHEEL_REBOUND_SEGMENTS)),
            "active_segment_index": None,
            "active_segment_first_bias_tick": None,
            "active_segment_last_bias_tick": None,
            "logical_bias_rad_s": 0.0,
            "reference_wheel_integral_rad": _WHEEL_REFERENCE_INTEGRAL,
            "additional_wheel_integral_rad": (
                _WHEEL_REBOUND_ADDITIONAL_INTEGRAL
            ),
            "resulting_wheel_integral_rad": _WHEEL_REBOUND_RESULTING_INTEGRAL,
            "reference_wheel_peak_abs_rad_s": 1.07,
            "resulting_wheel_peak_abs_rad_s": 1.07,
            "instantaneous_direction_reversal": True,
            "probe_channel": "rear_right_knee",
            "probe_channel_index": 7,
            "correction_channel": "front_left_ankle",
            "correction_channel_index": 8,
        }
    )
    _sync_wheel_rebound_atomic_ack(teardown)

    retry = json.loads(json.dumps(_wheel_rebound_feedback_ledger_rows()[:-1]))
    for row in retry:
        zeros = [0.0] * 12
        row["drive_feedback"].update(
            {
                "bias_full12": zeros,
                "active": False,
                "just_triggered": False,
                "trigger_tick": None,
                "cumulative_fraction_of_reference": 0.0,
                "active_segment_index": None,
                "active_segment_first_bias_tick": None,
                "active_segment_last_bias_tick": None,
                "logical_bias_rad_s": 0.0,
            }
        )
        row["drive_feedback_bias_requested_full12"] = zeros
        row["drive_feedback_bias_realized_full12"] = zeros
        row["drive_target_full12"] = list(row["native_drive_target_full12"])
        _sync_wheel_rebound_atomic_ack(row)
    tick_zero = json.loads(json.dumps(retry[0]))
    tick_zero["motion_tick_index"] = 0
    tick_zero["drive_feedback"].update(
        {
            "tick_index": 0,
            "observed_deg": 0.0,
            "reference_deg": None,
        }
    )
    tick_zero["native_drive_target_full12"][8] = 0.0
    tick_zero["drive_target_full12"][8] = 0.0
    _sync_wheel_rebound_atomic_ack(tick_zero)
    retry.insert(0, tick_zero)
    start_time = teardown["sim_time_s"] + 1.0 / 120.0
    for index, row in enumerate(retry):
        row["sim_time_s"] = start_time + index / 120.0

    rows = first_attempt + retry
    observations = _feedback_observations(rows)
    assert _drive_feedback_ledger_valid(rows, contract, observations)

    duplicate_trigger = json.loads(json.dumps(rows))
    retry_probe = duplicate_trigger[len(first_attempt) + 2]
    assert retry_probe["motion_tick_index"] == 859
    retry_probe["drive_feedback"].update(
        {
            "just_triggered": True,
            "trigger_tick": 859,
            "cumulative_fraction_of_reference": _WHEEL_REBOUND_FRACTION,
        }
    )
    assert not _drive_feedback_ledger_valid(
        duplicate_trigger,
        contract,
        _feedback_observations(duplicate_trigger),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_active_tick",
        "missing_hold_tick",
        "duplicate_active_tick",
        "wrong_probe_native",
        "wrong_pre_endpoint_native",
        "wrong_pre_endpoint_final",
        "wrong_post_endpoint_native",
        "wrong_post_endpoint_final",
        "wrong_hold_native",
        "wrong_hold_final",
        "wrong_active_value",
        "wrong_hold_value",
        "other_channel_correction",
        "feedback_kind_mismatch",
        "logged_segments_mismatch",
        "logged_segment_index_mismatch",
        "logged_segment_bounds_mismatch",
        "logged_segment_bias_mismatch",
        "logged_peak_fraction_nonzero",
        "logged_cumulative_fraction_mismatch",
        "direction_flag_missing",
        "logged_resulting_integral_mismatch",
        "logged_reference_peak_mismatch",
        "logged_resulting_peak_mismatch",
        "teardown_residual",
        "extended_active_teardown",
        "untriggered_correction",
        "wrong_cadence",
        "hard_wheel_limit",
        "atomic_ack_missing",
        "atomic_ack_physical_sign",
        "atomic_ack_correction_mismatch",
    ),
)
def test_wheel_rebound_feedback_ledger_fails_closed(mutation: str) -> None:
    contract = _wheel_rebound_feedback_contract()
    rows = _wheel_rebound_feedback_ledger_rows()
    observations = _feedback_observations(rows)
    if mutation == "missing_active_tick":
        del rows[5]  # tick 863
    elif mutation == "missing_hold_tick":
        del rows[13]  # tick 871
    elif mutation == "duplicate_active_tick":
        rows.insert(4, json.loads(json.dumps(rows[3])))
    elif mutation == "wrong_probe_native":
        rows[0]["native_drive_target_full12"][8] = 0.0
        rows[0]["drive_target_full12"][8] = 0.0
        _sync_wheel_rebound_atomic_ack(rows[0])
    elif mutation == "wrong_pre_endpoint_native":
        rows[2]["native_drive_target_full12"][8] = -1.06
        rows[2]["drive_target_full12"][8] = -0.73
        _sync_wheel_rebound_atomic_ack(rows[2])
    elif mutation == "wrong_pre_endpoint_final":
        rows[2]["drive_target_full12"][8] = -0.73
        _sync_wheel_rebound_atomic_ack(rows[2])
    elif mutation == "wrong_post_endpoint_native":
        rows[6]["native_drive_target_full12"][8] = 0.01
        rows[6]["drive_target_full12"][8] = 1.06
        _sync_wheel_rebound_atomic_ack(rows[6])
    elif mutation == "wrong_post_endpoint_final":
        rows[6]["drive_target_full12"][8] = 1.04
        _sync_wheel_rebound_atomic_ack(rows[6])
    elif mutation == "wrong_hold_native":
        rows[10]["native_drive_target_full12"][8] = 0.01
        rows[10]["drive_target_full12"][8] = 0.39
        _sync_wheel_rebound_atomic_ack(rows[10])
    elif mutation == "wrong_hold_final":
        rows[10]["drive_target_full12"][8] = 0.37
        _sync_wheel_rebound_atomic_ack(rows[10])
    elif mutation == "wrong_active_value":
        active = rows[2]
        active["drive_feedback"]["bias_full12"][8] = 1.04
        active["drive_feedback_bias_requested_full12"][8] = 1.04
        active["drive_feedback_bias_realized_full12"][8] = 1.04
        active["drive_target_full12"][8] = -0.03
        _sync_wheel_rebound_atomic_ack(active)
    elif mutation == "wrong_hold_value":
        active = rows[10]
        active["drive_feedback"]["bias_full12"][8] = 0.37
        active["drive_feedback_bias_requested_full12"][8] = 0.37
        active["drive_feedback_bias_realized_full12"][8] = 0.37
        active["drive_target_full12"][8] = 0.37
        _sync_wheel_rebound_atomic_ack(active)
    elif mutation == "other_channel_correction":
        active = rows[2]
        active["drive_feedback"]["bias_full12"][9] = 0.01
        active["drive_feedback_bias_requested_full12"][9] = 0.01
        active["drive_feedback_bias_realized_full12"][9] = 0.01
        active["drive_target_full12"][9] = 0.01
        _sync_wheel_rebound_atomic_ack(active)
    elif mutation == "feedback_kind_mismatch":
        rows[2]["drive_feedback"]["kind"] = "verify_tail_wheel_carry_alignment"
    elif mutation == "logged_segments_mismatch":
        rows[10]["drive_feedback"]["bias_segments"][5][
            "last_bias_tick"
        ] = 868
    elif mutation == "logged_segment_index_mismatch":
        rows[10]["drive_feedback"]["active_segment_index"] = 0
    elif mutation == "logged_segment_bounds_mismatch":
        rows[10]["drive_feedback"]["active_segment_first_bias_tick"] = 867
    elif mutation == "logged_segment_bias_mismatch":
        rows[10]["drive_feedback"]["logical_bias_rad_s"] = 0.37
    elif mutation == "logged_peak_fraction_nonzero":
        rows[2]["drive_feedback"]["peak_fraction_of_reference"] = 0.01
    elif mutation == "logged_cumulative_fraction_mismatch":
        rows[10]["drive_feedback"]["cumulative_fraction_of_reference"] += 0.01
    elif mutation == "direction_flag_missing":
        rows[2]["drive_feedback"]["instantaneous_direction_reversal"] = False
    elif mutation == "logged_resulting_integral_mismatch":
        rows[2]["drive_feedback"]["resulting_wheel_integral_rad"] += 0.01
    elif mutation == "logged_reference_peak_mismatch":
        rows[2]["drive_feedback"]["reference_wheel_peak_abs_rad_s"] = 1.06
    elif mutation == "logged_resulting_peak_mismatch":
        rows[2]["drive_feedback"]["resulting_wheel_peak_abs_rad_s"] = 1.08
    elif mutation == "teardown_residual":
        teardown = rows[-1]
        teardown["drive_feedback_bias_realized_full12"][8] = 0.01
        teardown["drive_target_full12"][8] = 0.01
        _sync_wheel_rebound_atomic_ack(teardown)
    elif mutation == "extended_active_teardown":
        teardown = rows[-1]
        teardown["state_id"] = "P09"
        teardown["motion_tick_index"] = 872
        teardown["drive_feedback"].update(
            {
                "kind": "pre_endpoint_wheel_rebound_alignment",
                "bias_full12": [0.0] * 8 + [0.38, 0.0, 0.0, 0.0],
                "active": True,
                "tick_index": 872,
                "trigger_tick": 859,
                "bias_segments": json.loads(json.dumps(_WHEEL_REBOUND_SEGMENTS)),
                "active_segment_index": 7,
                "active_segment_first_bias_tick": 871,
                "active_segment_last_bias_tick": 871,
                "logical_bias_rad_s": 0.38,
                "cumulative_fraction_of_reference": _WHEEL_REBOUND_FRACTION,
                "reference_wheel_integral_rad": _WHEEL_REFERENCE_INTEGRAL,
                "additional_wheel_integral_rad": _WHEEL_REBOUND_ADDITIONAL_INTEGRAL,
                "resulting_wheel_integral_rad": _WHEEL_REBOUND_RESULTING_INTEGRAL,
                "reference_wheel_peak_abs_rad_s": 1.07,
                "resulting_wheel_peak_abs_rad_s": 1.07,
                "instantaneous_direction_reversal": True,
                "probe_channel": "rear_right_knee",
                "probe_channel_index": 7,
                "correction_channel": "front_left_ankle",
                "correction_channel_index": 8,
            }
        )
        teardown["drive_feedback_bias_requested_full12"][8] = 0.38
        teardown["drive_feedback_bias_realized_full12"][8] = 0.38
        teardown["drive_target_full12"][8] = 0.38
        _sync_wheel_rebound_atomic_ack(teardown)
    elif mutation == "untriggered_correction":
        for row, observation in zip(rows[:-1], observations[:-1], strict=True):
            feedback = row["drive_feedback"]
            reference = feedback.get("reference_deg")
            if reference is not None:
                feedback["observed_deg"] = reference
                observation["actual_full12"][7] = reference
            feedback["just_triggered"] = False
            feedback["trigger_tick"] = None
            feedback["cumulative_fraction_of_reference"] = 0.0
    elif mutation == "wrong_cadence":
        rows[8]["sim_time_s"] += 1.0 / 240.0
    elif mutation == "hard_wheel_limit":
        rows[0]["native_drive_target_full12"][9] = 2.2
        rows[0]["drive_target_full12"][9] = 2.2
        _sync_wheel_rebound_atomic_ack(rows[0])
    elif mutation == "atomic_ack_missing":
        del rows[2]["atomic_ack"]
    elif mutation == "atomic_ack_physical_sign":
        rows[6]["atomic_ack"]["wheel_target_physical_rad_s"][0] = 1.05
    elif mutation == "atomic_ack_correction_mismatch":
        rows[2]["atomic_ack"]["drive_feedback_bias_realized_full12"][8] = 0.32
    assert not _drive_feedback_ledger_valid(
        rows, contract, observations
    )


def test_wheel_rebound_contract_is_exact_and_not_a_same_sign_carry() -> None:
    rows = _wheel_rebound_feedback_ledger_rows()
    observations = _feedback_observations(rows)

    wrong_kind = _wheel_rebound_feedback_contract()
    wrong_kind["phases"][8]["drive_feedback"]["kind"] = (
        "verify_tail_wheel_carry_alignment"
    )
    assert not _drive_feedback_ledger_valid(rows, wrong_kind, observations)

    wrong_timing = _wheel_rebound_feedback_contract()
    wrong_timing["phases"][8]["drive_feedback"]["bias_segments"][0][
        "first_bias_tick"
    ] = 861
    assert not _drive_feedback_ledger_valid(rows, wrong_timing, observations)

    wrong_sign = _wheel_rebound_feedback_contract()
    wrong_sign["phases"][8]["drive_feedback"]["bias_segments"][0][
        "logical_bias_rad_s"
    ] = -0.68
    assert not _drive_feedback_ledger_valid(rows, wrong_sign, observations)

    wrong_hold = _wheel_rebound_feedback_contract()
    wrong_hold["phases"][8]["drive_feedback"]["bias_segments"][1][
        "logical_bias_rad_s"
    ] = 0.34
    assert not _drive_feedback_ledger_valid(rows, wrong_hold, observations)

    wrong_teardown = _wheel_rebound_feedback_contract()
    wrong_teardown["phases"][8]["drive_feedback"]["teardown_tick"] = 873
    assert not _drive_feedback_ledger_valid(rows, wrong_teardown, observations)

    wrong_result = _wheel_rebound_feedback_contract()
    wrong_result["phases"][8]["drive_feedback"][
        "resulting_wheel_integral_rad"
    ] += 0.01
    assert not _drive_feedback_ledger_valid(rows, wrong_result, observations)

    wrong_peak = _wheel_rebound_feedback_contract()
    wrong_peak["phases"][8]["drive_feedback"][
        "resulting_wheel_peak_abs_rad_s"
    ] = 1.08
    assert not _drive_feedback_ledger_valid(rows, wrong_peak, observations)

    wrong_direction = _wheel_rebound_feedback_contract()
    wrong_direction["phases"][8]["drive_feedback"][
        "instantaneous_direction_reversal"
    ] = False
    assert not _drive_feedback_ledger_valid(rows, wrong_direction, observations)

    wrong_probe = _wheel_rebound_feedback_contract()
    wrong_probe["phases"][8]["drive_feedback"]["probe_samples"][0][
        "reference_actual_deg"
    ] += 0.1
    coordinated_rows = json.loads(json.dumps(rows))
    coordinated_rows[0]["drive_feedback"]["reference_deg"] += 0.1
    coordinated_rows[0]["drive_feedback"]["observed_deg"] += 0.1
    assert not _drive_feedback_ledger_valid(
        coordinated_rows,
        wrong_probe,
        _feedback_observations(coordinated_rows),
    )

    wrong_lag = _wheel_rebound_feedback_contract()
    wrong_lag["phases"][8]["drive_feedback"]["lag_threshold_deg"] = 0.34
    assert not _drive_feedback_ledger_valid(rows, wrong_lag, observations)

    mutable_nominal = _wheel_rebound_feedback_contract()
    mutable_nominal["phases"][8]["drive_feedback"][
        "nominal_endpoint_restored"
    ] = False
    assert not _drive_feedback_ledger_valid(rows, mutable_nominal, observations)

    raw_runtime = _wheel_rebound_feedback_contract()
    raw_runtime["phases"][8]["drive_feedback"][
        "raw_recording_runtime_access_required"
    ] = True
    assert not _drive_feedback_ledger_valid(rows, raw_runtime, observations)

    inflated_reference = _wheel_rebound_feedback_contract()
    spec = inflated_reference["phases"][8]["drive_feedback"]
    spec["reference_wheel_integral_rad"] *= 2.0
    spec["resulting_wheel_integral_rad"] = (
        spec["reference_wheel_integral_rad"]
        + spec["additional_wheel_integral_rad"]
    )
    spec["cumulative_fraction_of_reference"] = abs(
        spec["additional_wheel_integral_rad"]
    ) / abs(spec["reference_wheel_integral_rad"])
    inflated_reference["phases"][8]["command_metrics"]["wheel_integral_rad"][
        "front_left_ankle"
    ] = spec["reference_wheel_integral_rad"]
    assert not _drive_feedback_ledger_valid(rows, inflated_reference, observations)

    forged_peak = _wheel_rebound_feedback_contract()
    p09 = forged_peak["phases"][8]
    # Equal-duration, opposite-signed early lobes preserve the frozen ZOH
    # integral exactly while raising its absolute peak above 1.07 rad/s.
    p09["waypoints"][1]["full12"][8] = 1.2
    p09["waypoints"][2]["full12"][8] = -1.2
    p09["command_metrics"]["wheel_peak_abs_target_rad_s"][
        "front_left_ankle"
    ] = 1.2
    assert not _drive_feedback_ledger_valid(rows, forged_peak, observations)


def test_writer_api_remains_streaming_and_never_reuses_directory(tmp_path: Path) -> None:
    run = tmp_path / "writer"
    with TrialArtifactWriter(run) as writer:
        writer.append("task_event", {"result": "SUCCESS"})
        writer.append_similarity({"phase": "P01"})
    with pytest.raises(FileExistsError):
        TrialArtifactWriter(run)
