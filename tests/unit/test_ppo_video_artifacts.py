from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from wlr50_clean.ppo import video_artifacts


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_episode(
    parent: Path,
    *,
    role: str,
    seed: int,
    decision_count: int,
) -> Path:
    root = parent / role
    root.mkdir()
    video = root / "actual_viewport_video.mp4"
    video.write_bytes(f"real-{role}-viewport-video".encode())
    frame_count = decision_count + 32
    ledger = root / "viewport_frame_ledger.jsonl"
    _jsonl(
        ledger,
        [
            {
                "encoded_frame_index": index,
                "sim_step": index * 8,
                "sim_time_s": index / 15.0,
            }
            for index in range(frame_count)
        ],
    )
    trace_rows = []
    for index in range(decision_count):
        phase = f"P{min(index + 1, 13):02d}"
        trace_rows.append(
            {
                "seed": seed,
                "decision_index": index,
                "physics_tick": (index + 1) * 8,
                "sim_time_s": (index + 1) / 15.0,
                "state_id": phase,
                "lifecycle": "EXECUTE_MOTION",
                "roll_error_rad": -0.001 * index,
                "pitch_error_rad": 0.002 * index,
                "roll_rate_rad_s": -0.01 * index,
                "pitch_rate_rad_s": 0.02 * index,
                "residual_full12": (
                    [0.01 + index * 0.0001] * 12 if role == "ppo" else [0.0] * 12
                ),
                "termination_reason": "SUCCESS" if index == decision_count - 1 else None,
            }
        )
    trace_path = root / "policy_trace.jsonl"
    _jsonl(trace_path, trace_rows)
    trial = root / "trial_manifest.json"
    _json(
        trial,
        {
            "schema": "wlr50_clean.ppo_live_trial_manifest.v1",
            "ppo_calibration": {
                "quality_passed": True,
                "home_joint_positions_deg8": [0.0] * 8,
                "level_reference_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
    )
    recorder = root / "viewport_buffer_video_manifest.json"
    _json(
        recorder,
        {
            "schema": "wlr50_clean.active_viewport_video.v1",
            "valid": True,
            "capture_backend": "active_viewport_test_backend",
            "render_product_path": "/Render/Product/Viewport",
            "viewport_identity": 42 if role == "fsm" else 84,
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "frame_count": frame_count,
            "stitched": False,
            "speed_modified": False,
            "video_path": str(video),
            "video_sha256": _sha(video),
            "ledger_path": str(ledger),
            "ledger_sha256": _sha(ledger),
            "full_decode": {"valid": True},
        },
    )
    checkpoint = parent / "checkpoint_improved.pt"
    if role == "ppo" and not checkpoint.exists():
        checkpoint.write_bytes(b"real-nonzero-policy-checkpoint")
    checkpoint_manifest = parent / "checkpoint_improved_manifest.json"
    if role == "ppo" and not checkpoint_manifest.exists():
        _json(
            checkpoint_manifest,
            {
                "schema": "wlr50_clean.phase_residual_checkpoint_manifest.v1",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "publication_role": "improved",
                "validation_promotion_authorized": True,
                "locked_test_authorized": True,
                "promotion_authorized": True,
            },
        )
    episode_duration = decision_count / 15.0
    manifest = {
        "schema": "wlr50_clean.ppo_video_source_episode.v1",
        "policy_label": "ppo_deterministic_mean" if role == "ppo" else "fsm_zero_residual",
        "deterministic_mean_policy": role == "ppo",
        "fresh_process_single_episode": True,
        "episode_count": 1,
        "capture_process_id": 101 if role == "fsm" else 202,
        "capture_process_instance_id": "1" * 32 if role == "fsm" else "2" * 32,
        "used_preinitialized_fresh_episode": role == "ppo",
        "pre_action_observation_refreshed": True,
        "pre_action_refresh_evidence": {
            "schema": "wlr50_clean.ppo_video_pre_action_refresh.v1",
            "physical_pre_action_ticks": 64,
            "pre_roll_reader_last_logical_tick": 64,
            "episode_reader_reinitialized": True,
            "episode_reader_first_logical_tick": 0,
            "controller_frame_preserved": True,
            "controller_logical_tick": 0,
            "simulation_reset_performed": False,
            "fsm_step_performed": False,
        },
        "seed": seed,
        "source_checkpoint": str(checkpoint) if role == "ppo" else None,
        "source_checkpoint_sha256": _sha(checkpoint) if role == "ppo" else None,
        "source_checkpoint_manifest": (
            str(checkpoint_manifest) if role == "ppo" else None
        ),
        "source_checkpoint_manifest_sha256": (
            _sha(checkpoint_manifest) if role == "ppo" else None
        ),
        "checkpoint_load_provenance": (
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "manifest_path": str(checkpoint_manifest),
                "manifest_sha256": _sha(checkpoint_manifest),
                "checkpoint_infos_match_manifest": True,
            }
            if role == "ppo"
            else None
        ),
        "task_success": True,
        "completed_p01_p13": True,
        "body_collision": False,
        "wheel_only_climb": False,
        "duration_s": episode_duration,
        "decision_count": decision_count,
        "pre_action_physical_hold_s": 8 / 15.0,
        "post_success_physical_hold_s": 23 / 15.0,
        "semantic_action_start_video_s": 8 / 15.0,
        "semantic_task_success_video_s": 8 / 15.0 + episode_duration,
        "video_duration_expected_s": (decision_count + 31) / 15.0,
        "stitched": False,
        "speed_modified": False,
        "frame_interpolation": False,
        "source_episode_directory": str(root),
        "source_identity": {
            "environment_hash": "e" * 64,
            "robot_asset_hash": "a" * 64,
            "initial_root_state": [0.0] * 13,
            "initial_joint_state": [0.0] * 24,
            "obstacle_pose": [0.65, 0.0, 0.025],
            "controller_hash": "c" * 64,
            "motion_contract_hash": "d" * 64,
        },
        "reset_evidence": {
            "reset_count": 1,
            "reset_options": {},
            "training_phase_snapshot": None,
            "reset_global_simulation_resets": 0,
            "reset_simulation_forward_syncs": 0,
        },
        "camera": {
            "eye_m": [1.45, -1.25, 0.80],
            "target_m": [0.45, 0.0, 0.12],
            "resolution": [1280, 720],
            "fps": 15.0,
            "locked_scene_snapshot": True,
            "active_viewport_resolution_verified": True,
            "render_product_path": "/Render/Product/Viewport",
            "viewport_identity": 42 if role == "fsm" else 84,
        },
        "trial_manifest": str(trial),
        "policy_trace": str(trace_path),
        "policy_trace_sha256": _sha(trace_path),
        "policy_trace_rate_hz": 15.0,
        "policy_trace_rows": decision_count,
        "reset_info_recording_access_count": 0,
        "recorder_manifest": str(recorder),
        "raw_video": str(video),
        "raw_video_sha256": _sha(video),
    }
    _json(root / "ppo_video_source_manifest.json", manifest)
    return root


def _install_fake_video_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[list[str]]:
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"fake executable")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        Path(command[-1]).write_bytes(("encoded:" + Path(command[-1]).name).encode())
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_validate(path, **kwargs):
        source = Path(path)
        frame_count = int(kwargs["expected_frame_count"])
        width = int(kwargs["expected_width"])
        height = int(kwargs["expected_height"])
        return {
            "schema": "wlr50_clean.video_validation.v1",
            "path": str(source),
            "valid": True,
            "status": "PASS",
            "sha256": _sha(source),
            "bytes": source.stat().st_size,
            "duration_s": frame_count / 15.0,
            "container_duration_s": frame_count / 15.0,
            "fps": 15.0,
            "frame_count": frame_count,
            "resolution": [width, height],
            "width": width,
            "height": height,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "full_decode": True,
            "timestamps_monotonic": True,
            "timestamps_continuous": True,
            "stitched": False,
            "speed_modified": False,
        }

    monkeypatch.setattr(video_artifacts, "find_ffmpeg", lambda _: executable)
    monkeypatch.setattr(video_artifacts.subprocess, "run", fake_run)
    monkeypatch.setattr(video_artifacts, "validate_mp4", fake_validate)
    return calls


def test_publish_four_videos_with_real_time_padding_trace_overlay_and_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsm = _source_episode(tmp_path, role="fsm", seed=4001, decision_count=13)
    ppo = _source_episode(tmp_path, role="ppo", seed=4001, decision_count=14)
    calls = _install_fake_video_tools(monkeypatch, tmp_path)
    output = tmp_path / "outputs" / "ppo_phase_v1"

    publication = video_artifacts.publish_final_videos(
        fsm_source_dir=fsm,
        ppo_source_dir=ppo,
        output_root=output,
    )

    assert {path.name for path in publication.videos.values()} == {
        "fsm_baseline_clean.mp4",
        "ppo_improved_checkpoint_clean.mp4",
        "fsm_vs_ppo_improved.mp4",
        "ppo_improved_diagnostic.mp4",
    }
    assert all(path.is_file() for path in publication.videos.values())
    assert publication.checksum_verification["valid"] is True
    assert len(calls) == 4
    comparison = next(command for command in calls if "[outv]" in command)
    graph = comparison[comparison.index("-filter_complex") + 1]
    assert graph.count("tpad=stop_mode=clone") == 1
    assert "[0:v]setpts=PTS-STARTPTS,fps=15,tpad=stop_mode=clone" in graph
    assert "stop_duration=0.066666667" in graph
    assert "[1:v]setpts=PTS-STARTPTS,fps=15,tpad" not in graph
    assert "hstack=inputs=2" in graph
    for command in calls[:2]:
        assert command.count("-i") == 1
        assert "setpts=PTS-STARTPTS,format=yuv420p" in command
        assert command[command.index("-r") + 1] == "15"
        assert not any("trim=" in item for item in command)

    ass = publication.diagnostic_ass_path.read_text(encoding="utf-8")
    assert "phase P01" in ass
    assert "pitch rate" in ass
    assert "roll rate" in ass
    assert "residual RMS" in ass
    assert "result SUCCESS" in ass
    assert r"\N" in ass

    validation = json.loads(publication.validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["pair_evidence"]["same_seed"] is True
    assert validation["videos"]["comparison"]["resolution"] == [2560, 720]
    assert validation["videos"]["comparison"]["processing"][
        "earlier_final_frame_tpad_clone_side"
    ] == "fsm"
    for record in validation["videos"].values():
        assert record["codec"] == "h264"
        assert record["pix_fmt"] == "yuv420p"
        assert record["fps"] == 15.0
        assert record["duration_s"] <= 200.0
        assert record["full_decode"] is True
        assert record["monotonic"] is True
        assert record["stitched"] is False
        assert record["speed_modified"] is False
        assert record["source_episode"]
        assert record["source_checkpoint"]
        assert record["source_seed"] == 4001


def test_publication_rejects_different_seeds_before_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsm = _source_episode(tmp_path, role="fsm", seed=4001, decision_count=13)
    ppo = _source_episode(tmp_path, role="ppo", seed=4002, decision_count=13)
    calls = _install_fake_video_tools(monkeypatch, tmp_path)

    with pytest.raises(video_artifacts.PPOVideoArtifactError, match="different seeds"):
        video_artifacts.publish_final_videos(
            fsm_source_dir=fsm,
            ppo_source_dir=ppo,
            output_root=tmp_path / "final",
        )
    assert calls == []


def test_publication_rejects_zero_residual_ppo_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsm = _source_episode(tmp_path, role="fsm", seed=4001, decision_count=13)
    ppo = _source_episode(tmp_path, role="ppo", seed=4001, decision_count=13)
    trace_path = ppo / "policy_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["residual_full12"] = [0.0] * 12
    _jsonl(trace_path, rows)
    source_manifest_path = ppo / "ppo_video_source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["policy_trace_sha256"] = _sha(trace_path)
    _json(source_manifest_path, source_manifest)
    calls = _install_fake_video_tools(monkeypatch, tmp_path)

    with pytest.raises(video_artifacts.PPOVideoArtifactError, match="only zero residuals"):
        video_artifacts.publish_final_videos(
            fsm_source_dir=fsm,
            ppo_source_dir=ppo,
            output_root=tmp_path / "final",
        )
    assert calls == []


def test_publication_rejects_environment_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsm = _source_episode(tmp_path, role="fsm", seed=4001, decision_count=13)
    ppo = _source_episode(tmp_path, role="ppo", seed=4001, decision_count=13)
    manifest_path = ppo / "ppo_video_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_identity"]["environment_hash"] = "f" * 64
    _json(manifest_path, manifest)
    calls = _install_fake_video_tools(monkeypatch, tmp_path)

    with pytest.raises(video_artifacts.PPOVideoArtifactError, match="identities differ"):
        video_artifacts.publish_final_videos(
            fsm_source_dir=fsm,
            ppo_source_dir=ppo,
            output_root=tmp_path / "final",
        )
    assert calls == []


def test_publication_is_immutable_and_refuses_a_second_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsm = _source_episode(tmp_path, role="fsm", seed=4001, decision_count=13)
    ppo = _source_episode(tmp_path, role="ppo", seed=4001, decision_count=13)
    calls = _install_fake_video_tools(monkeypatch, tmp_path)
    output = tmp_path / "final"
    video_artifacts.publish_final_videos(
        fsm_source_dir=fsm,
        ppo_source_dir=ppo,
        output_root=output,
    )
    first_call_count = len(calls)

    with pytest.raises(video_artifacts.PPOVideoArtifactError, match="refusing to overwrite"):
        video_artifacts.publish_final_videos(
            fsm_source_dir=fsm,
            ppo_source_dir=ppo,
            output_root=output,
        )
    assert len(calls) == first_call_count
