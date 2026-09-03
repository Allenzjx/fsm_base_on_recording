import json
from pathlib import Path

from wlr50_clean.evaluation.physical_success import (
    _classify,
    _diagnostic_rows,
    _forbidden_evidence,
    _gravity_override_absence_proof,
    _ledger_continuity_evidence,
    _observation_traversal_evidence,
    _scan_large_ledger,
    select_success,
)


def _success(**overrides):
    row = {
        "trial_id": "trial_001",
        "trial_number": 1,
        "trial_validity": "VALID",
        "task_result": "SUCCESS",
        "body_collision": False,
        "wheel_only_climb": False,
        "physical_traversal_complete": True,
        "environment_match": True,
        "video_continuous": True,
        "video_decode": True,
        "forbidden_control_count": 0,
        "final_pose_stable": True,
        "recovery_count": 0,
        "duration_s": 100.0,
        "reference_max_error_percent": 10.0,
    }
    row.update(overrides)
    return row


def test_reference_divergence_is_diagnostic_and_cannot_veto_success() -> None:
    validity, task = _classify(
        original_result="INCOMPLETE_CONTROLLER_BLOCKED",
        environment_match=True,
        forbidden={"all_zero": True},
        physical_ledgers_valid=True,
        continuous_physics=True,
        video_valid=True,
        p01_p13_complete=True,
        geometry_success=True,
        all_lifts=True,
        body_collision=False,
        wheel_only_climb=False,
        fall=False,
        explosion=False,
    )
    rows, summary = _diagnostic_rows(
        "trial_043",
        [
            {
                "phase": "P09",
                "channel": "rear_left_ankle",
                "channel_kind": "wheel",
                "duration_error_percent": 0.0,
                "command_average_velocity_error_percent": 0.0,
                "measured_average_velocity_error_percent": 0.0,
                "command_peak_velocity_error_percent": 0.0,
                "measured_peak_velocity_error_percent": 36.458637,
                "command_wheel_integral_error_percent": 0.0,
                "actual_wheel_integral_error_percent": 0.0,
            }
        ],
        30.0,
    )
    assert (validity, task) == ("VALID", "SUCCESS")
    assert summary["within_30_percent"] is False
    assert summary["blocks_task_success"] is False
    assert any(row["warning"] == "REFERENCE_DIVERGENCE_WARNING" for row in rows)


def test_section_8_selects_trial_043_as_soon_as_it_is_eligible() -> None:
    explicit = _success(
        trial_id="trial_043", trial_number=43, final_pose_stable=True,
        recovery_count=1, duration_s=108.0, reference_max_error_percent=95.0,
    )
    recovered = _success(
        trial_id="trial_025", trial_number=25, recovery_count=1,
        duration_s=90.0, reference_max_error_percent=1.0,
    )
    no_recovery = _success(
        trial_id="trial_036", trial_number=36, recovery_count=0,
        duration_s=91.0, reference_max_error_percent=95.0,
    )
    assert select_success([explicit, recovered, no_recovery]) == explicit


def test_fallback_divergence_is_only_the_tenth_tie_breaker() -> None:
    larger = _success(trial_id="trial_036", trial_number=36, reference_max_error_percent=95.0)
    smaller = _success(trial_id="trial_039", trial_number=39, reference_max_error_percent=36.0)
    assert select_success([larger, smaller]) == smaller


def test_incomplete_p10_entry_is_not_a_task_failure_or_success() -> None:
    validity, task = _classify(
        original_result="INCOMPLETE_CONTROLLER_BLOCKED",
        environment_match=True,
        forbidden={"all_zero": True},
        physical_ledgers_valid=True,
        continuous_physics=True,
        video_valid=True,
        p01_p13_complete=False,
        geometry_success=False,
        all_lifts=False,
        body_collision=False,
        wheel_only_climb=False,
        fall=False,
        explosion=False,
    )
    assert (validity, task) == ("VALID", "INCOMPLETE_CONTROLLER_BLOCKED")


def test_exact_command_observation_tick_time_continuity(tmp_path: Path) -> None:
    dt = 1.0 / 120.0
    observation_path = tmp_path / "observation.jsonl"
    command_path = tmp_path / "command.jsonl"
    observation_path.write_text(
        "".join(
            json.dumps(
                {
                    "physics_tick": tick,
                    "simulation_time_s": tick * dt,
                    "physics_dt_s": dt,
                    "all_finite": True,
                },
                separators=(",", ":"),
            )
            + "\n"
            for tick in range(3)
        ),
        encoding="utf-8",
    )
    command_path.write_text(
        "".join(
            json.dumps(
                {
                    "control_physics_tick": tick,
                    "sim_time_s": tick * dt,
                    "full12": [0.0] * 12,
                    "articulation_writes_this_call": 1,
                },
                separators=(",", ":"),
            )
            + "\n"
            for tick in range(2)
        ),
        encoding="utf-8",
    )
    observation = _scan_large_ledger(
        observation_path, kind="observation", physics_dt_s=dt
    )
    command = _scan_large_ledger(command_path, kind="command", physics_dt_s=dt)
    evidence = _ledger_continuity_evidence(
        observation, command, physics_dt_s=dt
    )
    assert evidence["passed"] is True
    assert evidence["command_row_count"] == 2
    assert evidence["observation_row_count"] == 3
    assert evidence["terminal_observation_tick"] == 2
    assert (
        evidence["command_tick_time_sha256"]
        == evidence["observation_command_prefix_tick_time_sha256"]
    )


def test_cross_ledger_time_shift_is_not_continuous(tmp_path: Path) -> None:
    dt = 1.0 / 120.0
    observation_path = tmp_path / "observation.jsonl"
    command_path = tmp_path / "command.jsonl"
    observation_path.write_text(
        "".join(
            json.dumps(
                {
                    "physics_tick": tick,
                    "simulation_time_s": tick * dt,
                    "physics_dt_s": dt,
                    "all_finite": True,
                },
                separators=(",", ":"),
            )
            + "\n"
            for tick in range(3)
        ),
        encoding="utf-8",
    )
    command_path.write_text(
        "".join(
            json.dumps(
                {
                    "control_physics_tick": tick,
                    "sim_time_s": (tick + 1) * dt,
                    "full12": [0.0] * 12,
                    "articulation_writes_this_call": 1,
                },
                separators=(",", ":"),
            )
            + "\n"
            for tick in range(2)
        ),
        encoding="utf-8",
    )
    observation = _scan_large_ledger(
        observation_path, kind="observation", physics_dt_s=dt
    )
    command = _scan_large_ledger(command_path, kind="command", physics_dt_s=dt)
    evidence = _ledger_continuity_evidence(
        observation, command, physics_dt_s=dt
    )
    assert evidence["passed"] is False
    assert evidence["checks"]["command_tick_time_matches_observation_row"] is False
    assert evidence["checks"]["first_command_time_matches_first_observation"] is False


def test_historical_gravity_counter_uses_hash_bound_absence_proof() -> None:
    root = Path(__file__).resolve().parents[2]
    lock_path = root / "configs/environment_lock.json"
    environment_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    proof = _gravity_override_absence_proof(
        project_root=root,
        environment_lock_path=lock_path,
        environment_lock=environment_lock,
    )
    assert proof["passed"] is True
    assert proof["configuration"]["matches_expected"] is True
    assert len(proof["configuration"]["sha256"]) == 64
    assert len(proof["runtime_source_set_sha256"]) == 64
    assert proof["runtime_source_file_count"] > 0
    assert proof["forbidden_mutation_findings"] == []

    manifest = {
        "success_evidence": {
            "root_state_write_count": 0,
            "teleport_count": 0,
            "external_force_count": 0,
            "external_impulse_count": 0,
            "runtime_raw_recording_access": False,
        }
    }
    evidence = _forbidden_evidence(manifest, proof)
    assert evidence["gravity_override_count"] is None
    assert evidence["gravity_override_absent"] is True
    assert evidence["gravity_absence_proof"]["passed"] is True
    assert evidence["all_zero"] is True

    evidence_without_proof = _forbidden_evidence(manifest, {"passed": False})
    assert evidence_without_proof["gravity_override_absent"] is False
    assert evidence_without_proof["all_zero"] is False


def test_terminal_observation_latches_can_replace_missing_event_rows() -> None:
    guards = {}
    for index, leg in enumerate(("FR", "FL", "RR", "RL")):
        base_tick = 100 + index * 100
        guards[f"reference_like_active_lift:{leg}"] = {
            "passed": True,
            "value": {"latch_tick": base_tick},
        }
        guards[f"leg_front_face_crossed_latched:{leg}"] = {
            "passed": True,
            "value": {"latch_tick": base_tick + 10},
        }
        guards[f"leg_top_loaded_latched:{leg}"] = {
            "passed": True,
            "value": {"latch_tick": base_tick + 20},
        }
    guards["all_leg_front_face_crossings_latched"] = {"passed": True}
    guards["all_wheels_final_top_geometry"] = {"passed": True}
    observation = _observation_traversal_evidence(
        {"physics_tick": 999, "guards": guards}
    )
    assert observation["passed"] is True
    assert observation["rear_leg_order"] == "RR_FIRST"

    validity, task = _classify(
        original_result="INCOMPLETE_CONTROLLER_BLOCKED",
        environment_match=True,
        forbidden={"all_zero": True},
        physical_ledgers_valid=True,
        continuous_physics=True,
        video_valid=True,
        p01_p13_complete=False,
        geometry_success=True,
        all_lifts=False,
        body_collision=False,
        wheel_only_climb=False,
        fall=False,
        explosion=False,
        observation_traversal_proof=observation["passed"],
    )
    assert (validity, task) == ("VALID", "SUCCESS")


def test_checked_in_readjudication_evidence_has_stable_lf_bytes() -> None:
    output = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "analysis"
        / "physical_success_readjudication"
    )
    for path in output.iterdir():
        if path.is_file():
            assert b"\r\n" not in path.read_bytes(), path.name
