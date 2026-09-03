from __future__ import annotations

import json
from pathlib import Path

import pytest

from wlr50_clean.evaluation.comparison import (
    PHASE_IDS,
    ComparisonContractError,
    PhaseWindow,
    align_phases,
    comparison_filter,
    diagnostic_filter,
    fsm_windows_from_evidence,
    reference_windows_from_contract,
)
from wlr50_clean.evaluation.video_builder import VideoBuildError, validate_successful_trial
from wlr50_clean.infrastructure.video_capture import sha256_file


def _windows(multiplier: float = 1.0) -> tuple[PhaseWindow, ...]:
    return tuple(
        PhaseWindow(phase, index * multiplier, (index + 1) * multiplier, f"state_{phase}")
        for index, phase in enumerate(PHASE_IDS)
    )


def test_semantic_alignment_preserves_native_speed_and_holds_shorter_side() -> None:
    aligned = align_phases(_windows(1.0), _windows(1.1))
    assert len(aligned) == 13
    assert aligned[0].output_duration_s == pytest.approx(1.1)
    graph = comparison_filter(aligned, fsm_trial_id="trial_043")
    assert "setpts=PTS-STARTPTS" in graph
    assert "tpad=stop_mode=clone" in graph
    assert "setpts=PTS/" not in graph
    assert "concat=n=13" in graph
    assert "hstack=inputs=2" in graph
    assert "pad=1280:720" in graph
    assert "Recording v010" in graph
    assert "FSM  trial_043" in graph
    assert r"30\% diagnostic" in graph


def test_phase_order_is_exact_and_comparison_is_bounded() -> None:
    with pytest.raises(ComparisonContractError):
        align_phases(_windows()[:-1], _windows())
    with pytest.raises(ComparisonContractError, match="exceeding"):
        align_phases(_windows(10.0), _windows(10.0), maximum_duration_s=100.0)


def test_reference_contract_and_fsm_evidence_build_same_13_phase_clock(tmp_path: Path) -> None:
    contract = {
        "source": {"telemetry_cadence": {"first_sim_time_s": 1.5}},
        "phases": [
            {
                "state_id": phase,
                "state_name": f"reference_{phase}",
                "reference_sim_start_s": 1.5 + index,
            }
            for index, phase in enumerate(PHASE_IDS)
        ],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    reference = reference_windows_from_contract(contract_path, video_duration_s=13.0)
    assert reference == tuple(
        PhaseWindow(phase, float(index), float(index + 1), f"reference_{phase}")
        for index, phase in enumerate(PHASE_IDS)
    )

    evidence_path = tmp_path / "phases.json"
    evidence_path.write_text(
        json.dumps({"phase_windows": [row.as_dict() for row in _windows()]}),
        encoding="utf-8",
    )
    assert fsm_windows_from_evidence(evidence_path, video_duration_s=13.0) == _windows()
    overlay = diagnostic_filter(_windows())
    assert "body collision\\: false" in overlay
    assert "fontsize=18" in overlay


def _success_manifest() -> dict:
    return {
        "schema": "wlr50_clean.trial_manifest.v1",
        "success_evidence": {
            "task_result": "SUCCESS",
            "one_continuous_physical_fsm_success": True,
            "completed_macro_phases": list(PHASE_IDS),
            "body_collision": False,
            "wheel_only_climb": False,
            "rear_leg_order": "RR_FIRST",
            "root_state_write_count": 0,
            "teleport_count": 0,
            "external_force_count": 0,
            "external_impulse_count": 0,
        },
    }


def test_successful_trial_binding_is_fail_closed(tmp_path: Path) -> None:
    video = tmp_path / "actual_viewport_video.mp4"
    video.write_bytes(b"continuous-video")
    recorder = {
        "schema": "wlr50_clean.active_viewport_video.v1",
        "valid": True,
        "status": "PASS",
        "video_path": str(video.resolve()),
        "video_sha256": sha256_file(video),
    }
    (tmp_path / "viewport_buffer_video_manifest.json").write_text(json.dumps(recorder))
    manifest = _success_manifest()
    (tmp_path / "trial_manifest.json").write_text(json.dumps(manifest))
    result = validate_successful_trial(tmp_path)
    assert result["checks"]["p01_p13_completed"] is True
    assert result["video_sha256"] == sha256_file(video)

    manifest["success_evidence"]["wheel_only_climb"] = True
    (tmp_path / "trial_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(VideoBuildError, match="wheel_only_climb_false"):
        validate_successful_trial(tmp_path)


def test_external_physical_reclassification_can_accept_immutable_legacy_status(
    tmp_path: Path,
) -> None:
    video = tmp_path / "actual_viewport_video.mp4"
    video.write_bytes(b"continuous-video")
    recorder = {
        "schema": "wlr50_clean.active_viewport_video.v1",
        "valid": True,
        "status": "PASS",
        "video_path": str(video.resolve()),
        "video_sha256": sha256_file(video),
    }
    (tmp_path / "viewport_buffer_video_manifest.json").write_text(json.dumps(recorder))
    manifest = _success_manifest()
    manifest["success_evidence"]["task_result"] = "INCOMPLETE_CONTROLLER_BLOCKED"
    manifest["success_evidence"]["one_continuous_physical_fsm_success"] = False
    (tmp_path / "trial_manifest.json").write_text(json.dumps(manifest))
    physical_checks = {
        "p01_p13_completed": True,
        "body_collision_false": True,
        "wheel_only_climb_false": True,
        "rear_order_rr_first": True,
    }
    reclassification = {
        "run_dir": str(tmp_path.resolve()),
        "result_layers": {
            "trial_validity": {"result": "VALID", "checks": {}},
            "task_success": {"result": "SUCCESS", "checks": physical_checks},
        },
    }
    result = validate_successful_trial(
        tmp_path, physical_reclassification=reclassification
    )
    assert result["classification_source"] == "external_physical_reclassification"
    assert result["checks"]["reclassified_task_success"] is True

    reclassification["result_layers"]["task_success"]["result"] = "INCOMPLETE_OR_FAILED"
    with pytest.raises(VideoBuildError, match="reclassified_task_success"):
        validate_successful_trial(
            tmp_path, physical_reclassification=reclassification
        )


def test_wrapped_selected_success_trial_is_authoritative(tmp_path: Path) -> None:
    video = tmp_path / "actual_viewport_video.mp4"
    video.write_bytes(b"continuous-video")
    (tmp_path / "viewport_buffer_video_manifest.json").write_text(
        json.dumps(
            {
                "valid": True,
                "status": "PASS",
                "video_path": str(video.resolve()),
                "video_sha256": sha256_file(video),
            }
        )
    )
    manifest = _success_manifest()
    manifest["success_evidence"]["task_result"] = "INCOMPLETE_CONTROLLER_BLOCKED"
    manifest["success_evidence"]["one_continuous_physical_fsm_success"] = False
    (tmp_path / "trial_manifest.json").write_text(json.dumps(manifest))
    row = {
        "trial_id": tmp_path.name,
        "trial_validity": "VALID",
        "task_result": "SUCCESS",
        "P01_P13_complete": True,
        "body_collision": False,
        "wheel_only_climb": False,
        "rear_leg_order": "RR_FIRST",
        "video_path": str(video.resolve()),
        "video_continuous": True,
        "forbidden_control_count": 0,
    }
    result = validate_successful_trial(
        tmp_path,
        physical_reclassification={"selected_success_trial": row},
    )
    assert result["checks"]["selected_trial_matches_run"] is True

    # Direct rows are supported too, but selection can never be rebound to a
    # different immutable run directory.
    row["trial_id"] = "some_other_trial"
    with pytest.raises(VideoBuildError, match="selected_trial_matches_run"):
        validate_successful_trial(tmp_path, physical_reclassification=row)
