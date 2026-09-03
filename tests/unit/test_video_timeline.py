from __future__ import annotations

import json
from pathlib import Path

import pytest

from wlr50_clean.evaluation.comparison import (
    PHASE_IDS,
    fsm_windows_from_evidence,
)
from wlr50_clean.evaluation.video_builder import (
    FINAL_COMPARISON_NAME,
    FINAL_FSM_NAME,
    FINAL_RECORDING_NAME,
    FINAL_RECORDING_RAW_NAME,
    FSM_REQUESTED_POST_ROLL_S,
    FSM_REQUESTED_PRE_ROLL_S,
    V010_RECORDING_LEDGER_SHA256,
    VideoBuildError,
    _comparison_video_metadata,
    _frame_exact_encode_command,
    _packet_copy_window_command,
    _require_complete_video_metadata,
    _window_video_metadata,
)
from wlr50_clean.evaluation.video_timeline import (
    DecodedVideoFrame,
    LedgerFrame,
    VideoTimelineError,
    fsm_action_bounds,
    plan_action_window,
    reference_action_bounds,
    verify_native_rate_output,
)


ROOT = Path(__file__).resolve().parents[2]
FRAME_PERIOD = 1.0 / 15.0


def _frames(count: int, *, keyframes: tuple[int, ...] = (0,)):
    return tuple(
        DecodedVideoFrame(
            frame_index=index,
            pts_s=index * FRAME_PERIOD,
            checksum=f"{index:08X}",
            key_frame=index in keyframes,
        )
        for index in range(count)
    )


def _ledger(count: int, *, origin_s: float = 1.0):
    return tuple(
        LedgerFrame(
            frame_index=index,
            sim_step=100 + 8 * index,
            sim_time_s=origin_s + index * FRAME_PERIOD,
        )
        for index in range(count)
    )


def _exact_action_window(frames, ledger, *, start: float, end: float):
    return plan_action_window(
        frames,
        ledger,
        semantic_start_sim_s=start,
        semantic_end_sim_s=end,
        requested_pre_roll_s=0.0,
        requested_post_roll_s=0.0,
        maximum_preserved_roll_s=0.0,
    )


def test_action_window_uses_semantic_sim_time_and_decoded_pts() -> None:
    frames = _frames(8, keyframes=(0, 3))
    ledger = _ledger(8)
    window = _exact_action_window(
        frames,
        ledger,
        start=1.0 + FRAME_PERIOD,
        end=1.0 + 5.0 * FRAME_PERIOD,
    )
    assert (window.source_first_frame_index, window.source_last_frame_index) == (1, 5)
    assert window.expected_frame_count == 5
    assert window.leading_frames_removed == 1
    assert window.trailing_frames_removed == 2
    assert window.output_duration_s == pytest.approx(5.0 * FRAME_PERIOD)
    assert window.phase_clock_origin_sim_s == pytest.approx(1.0 + FRAME_PERIOD)
    assert window.packet_copy_eligible is False
    audit = window.as_dict()
    assert audit["selection_policy"]["container_duration_used"] is False
    assert audit["selection_policy"]["single_contiguous_source"] is True


def test_action_window_keeps_boundary_frames_when_events_fall_between_renders() -> None:
    frames = _frames(8)
    ledger = _ledger(8)
    window = _exact_action_window(
        frames,
        ledger,
        start=1.0 + 1.5 * FRAME_PERIOD,
        end=1.0 + 4.5 * FRAME_PERIOD,
    )
    # The surrounding renders are retained so neither edge of the action can
    # be clipped by a 120 Hz event falling between 15 Hz video frames.
    assert (window.source_first_frame_index, window.source_last_frame_index) == (1, 5)


def test_short_existing_context_is_preserved_without_unnecessary_transcode() -> None:
    recording_frames = _frames(1345, keyframes=(0,))
    recording_ledger = _ledger(1345, origin_s=1.5666666666666667)
    recording = plan_action_window(
        recording_frames,
        recording_ledger,
        semantic_start_sim_s=2.3666666666666667,
        semantic_end_sim_s=90.5,
    )
    assert (recording.source_first_frame_index, recording.source_last_frame_index) == (
        0,
        1344,
    )
    assert recording.is_full_source is True
    assert recording.packet_copy_eligible is True
    assert recording.available_pre_roll_s == pytest.approx(0.8)
    assert recording.available_post_roll_s == pytest.approx(2.0 / 3.0)
    assert recording.semantic_start_output_s == pytest.approx(0.8)
    assert recording.action_frame_end_output_pts_s_exclusive == pytest.approx(89.0)

    fsm_frames = _frames(1619, keyframes=(0,))
    fsm_ledger = _ledger(1619, origin_s=FRAME_PERIOD)
    fsm = plan_action_window(
        fsm_frames,
        fsm_ledger,
        semantic_start_sim_s=0.0,
        semantic_end_sim_s=107.86666666666666,
        requested_pre_roll_s=FSM_REQUESTED_PRE_ROLL_S,
        requested_post_roll_s=FSM_REQUESTED_POST_ROLL_S,
    )
    assert (fsm.source_first_frame_index, fsm.source_last_frame_index) == (0, 1618)
    assert fsm.capture_lag_after_action_start_s == pytest.approx(FRAME_PERIOD)
    assert fsm.retained_post_roll_s == pytest.approx(FRAME_PERIOD)

    fsm_metadata = _window_video_metadata(
        fsm, source_label="immutable selected FSM Trial source"
    )
    assert fsm_metadata["requested_pre_roll_s"] == 0.5
    assert fsm_metadata["requested_post_roll_s"] == 1.0
    assert fsm_metadata["retained_pre_roll_s"] == 0.0
    assert fsm_metadata["retained_post_roll_s"] == pytest.approx(FRAME_PERIOD)
    assert fsm_metadata["source_limited"] is True
    assert fsm_metadata["source_limited_pre_roll"] is True
    assert fsm_metadata["source_limited_post_roll"] is True
    assert "genuine requested pre-roll is unavailable" in fsm_metadata[
        "source_limitation_reason"
    ]
    assert "genuine post-roll" in fsm_metadata["source_limitation_reason"]
    assert fsm_metadata["edge_context_padding_applied"] is False
    assert fsm_metadata["edge_context_frames_duplicated"] is False

    comparison_metadata = _comparison_video_metadata(recording, fsm)
    required = (
        "start_sim_time",
        "end_sim_time",
        "pre_roll_s",
        "post_roll_s",
        "requested_pre_roll_s",
        "requested_post_roll_s",
    )
    assert all(comparison_metadata[name] is not None for name in required)
    assert comparison_metadata["time_domain"] == (
        "paired_source_simulation_time_s_by_side"
    )
    assert set(comparison_metadata["start_sim_time"]) == {"recording", "fsm"}
    assert comparison_metadata["source_limited"] is True
    assert comparison_metadata["edge_context_padding_applied"] is False

    complete = {
        "source_trial": ["recording", "fsm"],
        "source_video": ["recording.mp4", "fsm.mp4"],
        **comparison_metadata,
    }
    _require_complete_video_metadata({"comparison": complete})
    complete["start_sim_time"] = None
    with pytest.raises(VideoBuildError, match="start_sim_time"):
        _require_complete_video_metadata({"comparison": complete})


def test_action_window_rejects_nonconstant_sim_to_pts_mapping() -> None:
    frames = _frames(5)
    ledger = list(_ledger(5))
    ledger[3] = LedgerFrame(3, ledger[3].sim_step, ledger[3].sim_time_s + 0.01)
    with pytest.raises(VideoTimelineError, match="cadence|offset"):
        plan_action_window(
            frames,
            ledger,
            semantic_start_sim_s=1.0,
            semantic_end_sim_s=1.0 + 3.0 * FRAME_PERIOD,
            requested_pre_roll_s=0.0,
            requested_post_roll_s=0.0,
            maximum_preserved_roll_s=0.0,
        )


def test_semantic_bounds_exclude_startup_and_worker_shutdown(tmp_path: Path) -> None:
    contract = {
        "phases": [
            {
                "state_id": phase,
                "reference_sim_start_s": 2.0 + index,
                "reference_sim_end_s": 3.0 + index,
                "active_duration_s": 1.0,
            }
            for index, phase in enumerate(PHASE_IDS)
        ]
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    assert reference_action_bounds(contract_path) == (2.0, 15.0)

    phase_times = {
        phase: {"entry_time_s": float(index), "completion_time_s": float(index + 1)}
        for index, phase in enumerate(PHASE_IDS)
    }
    manifest_path = tmp_path / "trial_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "phase_times": phase_times,
                # Legacy phase window was extended to recorder shutdown.  It
                # must not override the semantic P13 completion above.
                "phase_windows": [
                    {
                        "phase": phase,
                        "start_s": float(index),
                        "end_s": 99.0 if phase == "P13" else float(index + 1),
                    }
                    for index, phase in enumerate(PHASE_IDS)
                ],
            }
        ),
        encoding="utf-8",
    )
    assert fsm_action_bounds(manifest_path) == (0.0, 13.0)
    windows = fsm_windows_from_evidence(
        manifest_path,
        video_duration_s=13.0 - FRAME_PERIOD,
        video_origin_sim_s=FRAME_PERIOD,
    )
    assert windows[1].start_s == pytest.approx(1.0 - FRAME_PERIOD)
    assert windows[-1].end_s == pytest.approx(13.0 - FRAME_PERIOD)


def test_trim_commands_are_single_source_native_rate_and_codec_constrained() -> None:
    frames = _frames(8, keyframes=(0, 3))
    ledger = _ledger(8)
    packet_window = _exact_action_window(
        frames,
        ledger,
        start=1.0,
        end=1.0 + 5.0 * FRAME_PERIOD,
    )
    packet = " ".join(
        _packet_copy_window_command(
            Path("ffmpeg"), Path("source.mp4"), Path("out.mp4"), packet_window
        )
    )
    assert packet.count("-i") == 1
    assert "-c:v copy" in packet
    assert "concat" not in packet

    encode_window = _exact_action_window(
        frames,
        ledger,
        start=1.0 + FRAME_PERIOD,
        end=1.0 + 5.0 * FRAME_PERIOD,
    )
    encode = " ".join(
        _frame_exact_encode_command(
            Path("ffmpeg"), Path("source.mp4"), Path("out.mp4"), encode_window
        )
    )
    assert encode.count("-i") == 1
    assert "trim=start_frame=1:end_frame=6,setpts=PTS-STARTPTS" in encode
    assert "setpts=PTS/" not in encode
    assert "libx264" in encode and "yuv420p" in encode
    assert "concat" not in encode


def test_packet_copy_verification_requires_selected_decoded_frames() -> None:
    frames = _frames(6)
    ledger = _ledger(6)
    window = _exact_action_window(
        frames,
        ledger,
        start=1.0,
        end=1.0 + 4.0 * FRAME_PERIOD,
    )
    output = frames[:5]
    result = verify_native_rate_output(
        output, window, require_decoded_frame_identity=True
    )
    assert result["decoded_frames_unchanged"] is True
    changed = list(output)
    changed[2] = DecodedVideoFrame(2, changed[2].pts_s, "DEADBEEF", False)
    with pytest.raises(VideoTimelineError, match="changed decoded"):
        verify_native_rate_output(
            changed, window, require_decoded_frame_identity=True
        )


def test_required_publication_filenames_are_stable() -> None:
    assert FINAL_RECORDING_RAW_NAME == "recording_v010_50mm_full_raw.mp4"
    assert FINAL_RECORDING_NAME == "recording_v010_50mm_clean.mp4"
    assert FINAL_FSM_NAME == "fsm_50mm_physical_success_clean.mp4"
    assert FINAL_COMPARISON_NAME == "recording_vs_fsm_50mm.mp4"
    assert len(V010_RECORDING_LEDGER_SHA256) == 64
    assert V010_RECORDING_LEDGER_SHA256.endswith("f3c")


def test_checked_in_video_validation_reports_complete_source_context() -> None:
    payload = json.loads(
        (ROOT / "outputs" / "final" / "video_validation.json").read_text(
            encoding="utf-8"
        )
    )
    required = (
        "source_trial",
        "source_video",
        "start_sim_time",
        "end_sim_time",
        "pre_roll_s",
        "post_roll_s",
        "time_domain",
    )
    for record in payload["videos"].values():
        assert all(record.get(field) is not None for field in required)

    fsm = payload["videos"]["fsm"]
    assert fsm["requested_pre_roll_s"] == 0.5
    assert fsm["requested_post_roll_s"] == 1.0
    assert fsm["retained_pre_roll_s"] == 0.0
    assert fsm["retained_post_roll_s"] == pytest.approx(FRAME_PERIOD)
    assert fsm["source_limited_pre_roll"] is True
    assert fsm["source_limited_post_roll"] is True
    assert fsm["edge_context_padding_applied"] is False
    assert fsm["edge_context_frames_duplicated"] is False

    comparison = payload["videos"]["comparison"]
    assert comparison["time_domain"] == "paired_source_simulation_time_s_by_side"
    assert set(comparison["start_sim_time"]) == {"recording", "fsm"}
    assert comparison["edge_context_padding_applied"] is False
