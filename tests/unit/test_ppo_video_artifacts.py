from __future__ import annotations

import hashlib
import io
import json
import subprocess
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wlr50_clean.ppo import video_artifacts
from wlr50_clean.ppo import cli
from wlr50_clean.ppo.artifacts import (
    RUN_MANIFEST_SCHEMA,
    RunIdentity,
    config_set_record,
    finalize_run,
)


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
    directory_name: str | None = None,
) -> Path:
    root = parent / (directory_name or role)
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


def _runtime_identity(project: Path, config: Path) -> dict:
    relative = config.relative_to(project).as_posix()
    row = {
        "path": relative,
        "bytes": config.stat().st_size,
        "sha256": _sha(config),
        "creation_time_utc_ticks": 1,
        "last_write_time_utc_ticks": 2,
    }
    aggregate = hashlib.sha256(
        json.dumps([row], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    content = hashlib.sha256(
        json.dumps(
            [{key: row[key] for key in ("path", "bytes", "sha256")}],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "wlr50_clean.committed_runtime_identity.v1",
        "git_commit": "a" * 40,
        "file_count": 1,
        "content_sha256": content,
        "aggregate_sha256": aggregate,
        "files": [row],
    }


def _frozen_audit(project: Path, *, checked_at: str) -> dict:
    frozen_manifest = project / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    rows = []
    for name, expected in sorted(manifest["protected_files"].items()):
        actual = _sha(project / name)
        rows.append(
            {
                "path": name,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": True,
                "valid": actual == expected,
            }
        )
    return {
        "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
        "checked_at_utc": checked_at,
        "project_root": str(project),
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": _sha(frozen_manifest),
        "source_head": manifest["source_head"],
        "protected_file_count": len(rows),
        "passed": True,
        "mismatches": [],
        "entries": rows,
    }


def _fixture_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict]:
    project = tmp_path / "project"
    config = project / "configs" / "test.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("locked: true\n", encoding="utf-8")
    protected = project / "configs" / "frozen.txt"
    protected.write_bytes(b"frozen-bytes")
    manifest = project / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    manifest.parent.mkdir(parents=True)
    _json(
        manifest,
        {
            "algorithm": "sha256",
            "source_head": "c" * 40,
            "protected_files": {"configs/frozen.txt": _sha(protected)},
        },
    )
    runtime = _runtime_identity(project, config)
    monkeypatch.setattr(video_artifacts, "_PROJECT_ROOT", project)
    monkeypatch.setattr(
        video_artifacts, "_validate_current_runtime_identity", lambda _identity: None
    )
    return project, config, runtime


def _started_run(
    project: Path,
    config: Path,
    runtime: dict,
    *,
    run_kind: str,
    training_stage: str,
    subcommand: str,
    invocation: list[str],
) -> Path:
    config_sha, config_records = config_set_record([config], project_root=project)
    identity = RunIdentity(
        timestamp_utc="2026-09-04T01:00:00Z",
        git_commit=runtime["git_commit"],
        config_sha256=config_sha,
        seed=4001,
        environment_count=1,
        training_stage=training_stage,
    )
    run_dir = project / "runs" / "ppo_phase_v1" / run_kind / identity.run_id
    run_dir.mkdir(parents=True)
    _json(
        run_dir / "run_manifest.started.json",
        {
            "schema": RUN_MANIFEST_SCHEMA,
            "lifecycle": "STARTED",
            "immutable_run_directory": True,
            "run_id": identity.run_id,
            "run_kind": run_kind,
            "run_dir": str(run_dir),
            "project_root": str(project),
            "identity": asdict(identity),
            "configs": config_records,
            "entrypoint": "wlr50_clean.ppo.cli",
            "subcommand": subcommand,
            "invocation_arguments": invocation,
        },
    )
    _json(run_dir / "committed_runtime_identity.before.json", runtime)
    _json(
        run_dir / "frozen_hashes.before.json",
        _frozen_audit(project, checked_at="2026-09-04T01:00:01Z"),
    )
    return run_dir


def _managed_source_episode(
    project: Path,
    config: Path,
    runtime: dict,
    *,
    role: str,
    decision_count: int = 13,
    ffmpeg: Path | None = None,
) -> Path:
    invocation = [
        "--training-config",
        str(config),
        "--interface-config",
        str(config),
        "--episode-count",
        "1",
        "--deterministic",
        "--capture-fps",
        "15",
        "--maximum-duration-s",
        "200",
        "--no-headless",
        "--video-source-role",
        role,
        *( ["--residual-mode", "zero"] if role == "fsm" else [
            "--checkpoint",
            "<checkpoint>",
            "--checkpoint-manifest",
            "<checkpoint-manifest>",
        ] ),
        "--run-dir",
        "<reserved-immutable-run-dir>",
        "--seed",
        "4001",
        "--num-envs",
        "1",
    ]
    run_dir = _started_run(
        project,
        config,
        runtime,
        run_kind=f"video-source-{role}",
        training_stage=f"video-source-{role}-fresh-process",
        subcommand="capture-video-source",
        invocation=invocation,
    )
    source = _source_episode(
        run_dir,
        role=role,
        seed=4001,
        decision_count=decision_count,
        directory_name="video_source",
    )
    if ffmpeg is not None:
        frame_count = decision_count + 32
        completed = subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1280x720:rate=15",
                "-frames:v",
                str(frame_count),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(source / "actual_viewport_video.mp4"),
            ],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"bundled ffmpeg cannot make fixture video: {completed.stderr[-500:]}")
        recorder_path = source / "viewport_buffer_video_manifest.json"
        recorder = json.loads(recorder_path.read_text(encoding="utf-8"))
        recorder["video_sha256"] = _sha(source / "actual_viewport_video.mp4")
        _json(recorder_path, recorder)
        source_manifest_path = source / "ppo_video_source_manifest.json"
        refreshed = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        refreshed["raw_video_sha256"] = _sha(source / "actual_viewport_video.mp4")
        _json(source_manifest_path, refreshed)
    source_manifest = json.loads(
        (source / "ppo_video_source_manifest.json").read_text(encoding="utf-8")
    )
    if role == "ppo":
        invocation[invocation.index("--checkpoint") + 1] = source_manifest[
            "source_checkpoint"
        ]
        invocation[invocation.index("--checkpoint-manifest") + 1] = source_manifest[
            "source_checkpoint_manifest"
        ]
        started_path = run_dir / "run_manifest.started.json"
        started = json.loads(started_path.read_text(encoding="utf-8"))
        started["invocation_arguments"] = invocation
        _json(started_path, started)
    checkpoint_capture = (
        None
        if role == "fsm"
        else {
            "schema": "wlr50_clean.checkpoint_runtime_capture.v1",
            "source_checkpoint_path": source_manifest["source_checkpoint"],
            "source_checkpoint_sha256": source_manifest[
                "source_checkpoint_sha256"
            ],
            "source_manifest_path": source_manifest[
                "source_checkpoint_manifest"
            ],
            "source_manifest_sha256": source_manifest[
                "source_checkpoint_manifest_sha256"
            ],
            "private_checkpoint_path": str(run_dir / ".checkpoint-pins" / "model.pt"),
            "private_manifest_path": str(run_dir / ".checkpoint-pins" / "manifest.json"),
            "private_copy_exclusive": True,
            "runner_loads_private_copy_only": True,
        }
    )
    capture = {
        "schema": "wlr50_clean.ppo_video_source_capture_cli.v1",
        "video_source_role": role,
        "fresh_process_single_episode": True,
        "seed": 4001,
        "headless": False,
        "active_viewport_configured": True,
        "source_directory": str(source),
        "source_manifest": str(source / "ppo_video_source_manifest.json"),
        "source_video": str(source / "actual_viewport_video.mp4"),
        "capture_process_id": source_manifest["capture_process_id"],
        "capture_process_instance_id": source_manifest[
            "capture_process_instance_id"
        ],
        "checkpoint_load_provenance": source_manifest[
            "checkpoint_load_provenance"
        ],
        "checkpoint_runtime_capture_verified": True,
        "checkpoint_runtime_capture": checkpoint_capture,
    }
    _json(run_dir / "video_source_capture.json", capture)
    _json(
        run_dir / "live_command_result.json",
        {
            "schema": "wlr50_clean.live_command_result.v1",
            "command": "capture-video-source",
            "exit_code": 0,
        },
    )
    _json(run_dir / "committed_runtime_identity.after.json", runtime)
    _json(
        run_dir / "frozen_hashes.after.json",
        _frozen_audit(project, checked_at="2026-09-04T01:00:02Z"),
    )
    before_line = {
        "audit": str(run_dir / "frozen_hashes.before.json"),
        "passed": True,
    }
    after_line = {
        "audit": str(run_dir / "frozen_hashes.after.json"),
        "passed": True,
    }
    (run_dir / "stdout.log").write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"))
            for row in (before_line, capture, after_line)
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stderr.log").write_bytes(b"")
    finalize_run(run_dir, exit_code=0)
    return source


def _finish_publication_run(
    project: Path,
    config: Path,
    runtime: dict,
    *,
    fsm_source: Path,
    ppo_source: Path,
    output_root: Path,
    ffmpeg: Path,
) -> tuple[Path, dict]:
    invocation = [
        "--training-config",
        str(config),
        "--interface-config",
        str(config),
        "--episode-count",
        "1",
        "--deterministic",
        "--fsm-video-source-dir",
        str(fsm_source),
        "--ppo-video-source-dir",
        str(ppo_source),
        "--output-root",
        str(output_root),
        "--ffmpeg",
        str(ffmpeg),
        "--run-dir",
        "<reserved-immutable-run-dir>",
        "--seed",
        "4001",
        "--num-envs",
        "1",
    ]
    run_dir = _started_run(
        project,
        config,
        runtime,
        run_kind="video-publication",
        training_stage="video-publication-offline",
        subcommand="publish-videos",
        invocation=invocation,
    )
    arguments = type(
        "Args",
        (),
        {
            "seed": 4001,
            "num_envs": 1,
            "episode_count": 1,
            "deterministic": True,
            "fsm_video_source_dir": fsm_source,
            "ppo_video_source_dir": ppo_source,
            "output_root": output_root,
            "ffmpeg": ffmpeg,
            "run_dir": run_dir,
        },
    )()
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert cli._publish_videos(arguments) == 0
    result = json.loads(
        (run_dir / "final_video_publication.json").read_text(encoding="utf-8")
    )
    _json(run_dir / "committed_runtime_identity.after.json", runtime)
    _json(
        run_dir / "frozen_hashes.after.json",
        _frozen_audit(project, checked_at="2026-09-04T01:00:02Z"),
    )
    rows = [
        json.dumps(
            {
                "audit": str(run_dir / "frozen_hashes.before.json"),
                "passed": True,
            },
            separators=(",", ":"),
        ),
        stdout.getvalue().strip(),
        json.dumps(
            {
                "audit": str(run_dir / "frozen_hashes.after.json"),
                "passed": True,
            },
            separators=(",", ":"),
        ),
    ]
    (run_dir / "stdout.log").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "stderr.log").write_bytes(b"")
    finalize_run(run_dir, exit_code=0)
    return run_dir, result


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
    runtime_identity = {
        "schema": "wlr50_clean.committed_runtime_identity.v1",
        "git_commit": "a" * 40,
        "content_sha256": "b" * 64,
    }

    def fake_managed_source(_source, *, role):
        return video_artifacts._ManagedRunValidation(
            evidence={"role": role, "managed": True},
            runtime_identity=runtime_identity,
        )

    monkeypatch.setattr(
        video_artifacts, "_validate_source_managed_run", fake_managed_source
    )
    monkeypatch.setattr(
        video_artifacts,
        "_publication_reservation_evidence",
        lambda *args, **kwargs: video_artifacts._ManagedRunValidation(
            evidence={"managed": True}, runtime_identity=runtime_identity
        ),
    )
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
        publication_run_dir=tmp_path / "publication-run",
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
            publication_run_dir=tmp_path / "publication-run",
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
            publication_run_dir=tmp_path / "publication-run",
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
            publication_run_dir=tmp_path / "publication-run",
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
        publication_run_dir=tmp_path / "publication-run",
    )
    first_call_count = len(calls)

    with pytest.raises(video_artifacts.PPOVideoArtifactError, match="refusing to overwrite"):
        video_artifacts.publish_final_videos(
            fsm_source_dir=fsm,
            ppo_source_dir=ppo,
            output_root=output,
            publication_run_dir=tmp_path / "publication-run",
        )
    assert len(calls) == first_call_count


def test_managed_source_verifier_binds_wrapper_runtime_frozen_stdout_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, config, runtime = _fixture_project(tmp_path, monkeypatch)
    source = _managed_source_episode(
        project, config, runtime, role="fsm"
    )

    evidence = video_artifacts.verify_video_source_managed_run(source, role="fsm")

    assert evidence["run_kind"] == "video-source-fsm"
    assert evidence["git_commit"] == runtime["git_commit"]
    assert evidence["committed_runtime_content_sha256"] == runtime["content_sha256"]
    assert evidence["run_manifest"]["sha256"] == _sha(source.parent / "run_manifest.json")
    assert evidence["stdout"]["sha256"] == _sha(source.parent / "stdout.log")


def test_managed_source_verifier_rejects_stdout_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, config, runtime = _fixture_project(tmp_path, monkeypatch)
    source = _managed_source_episode(project, config, runtime, role="fsm")
    with (source.parent / "stdout.log").open("ab") as stream:
        stream.write(b"tamper\n")

    with pytest.raises(
        video_artifacts.PPOVideoArtifactError, match="stdout.log record is stale"
    ):
        video_artifacts.verify_video_source_managed_run(source, role="fsm")


def test_managed_source_verifier_rejects_external_and_redirected_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, config, runtime = _fixture_project(tmp_path, monkeypatch)
    (tmp_path / "external").mkdir()
    external = _source_episode(
        tmp_path / "external",
        role="fsm",
        seed=4001,
        decision_count=13,
    )
    with pytest.raises(
        video_artifacts.PPOVideoArtifactError, match="outside the canonical managed runs root"
    ):
        video_artifacts.verify_video_source_managed_run(external, role="fsm")

    source = _managed_source_episode(project, config, runtime, role="fsm")
    manifest = source / "ppo_video_source_manifest.json"
    real_manifest = source / "real-source-manifest.json"
    manifest.rename(real_manifest)
    try:
        manifest.symlink_to(real_manifest)
    except OSError:
        pytest.skip("this Windows account cannot create symbolic links")
    with pytest.raises(
        video_artifacts.PPOVideoArtifactError, match="symbolic link|reparse point"
    ):
        video_artifacts.verify_video_source_managed_run(source, role="fsm")


def test_real_ffmpeg_publication_is_fully_decoded_and_binds_final_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        ffmpeg = video_artifacts.find_ffmpeg(None)
    except FileNotFoundError:
        pytest.skip("ffmpeg is unavailable")
    project, config, runtime = _fixture_project(tmp_path, monkeypatch)
    fsm = _managed_source_episode(
        project, config, runtime, role="fsm", decision_count=13, ffmpeg=ffmpeg
    )
    ppo = _managed_source_episode(
        project, config, runtime, role="ppo", decision_count=14, ffmpeg=ffmpeg
    )
    output = project / "outputs" / "ppo_phase_v1"
    publication_run, result = _finish_publication_run(
        project,
        config,
        runtime,
        fsm_source=fsm,
        ppo_source=ppo,
        output_root=output,
        ffmpeg=ffmpeg,
    )
    improved_hash = json.loads(
        (ppo / "ppo_video_source_manifest.json").read_text(encoding="utf-8")
    )["source_checkpoint_sha256"]

    verified = video_artifacts.verify_final_video_publication(
        result["video_validation"],
        result["video_checksums"],
        output_root=output,
        expected_improved_checkpoint_sha256=improved_hash,
        ffmpeg=ffmpeg,
    )

    assert verified["valid"] is True
    assert set(verified["videos"]) == {
        "fsm_baseline",
        "ppo_improved",
        "comparison",
        "ppo_diagnostic",
    }
    validation = json.loads(Path(result["video_validation"]).read_text(encoding="utf-8"))
    assert all(row["codec"] == "h264" for row in validation["videos"].values())
    assert all(row["pixel_format"] == "yuv420p" for row in validation["videos"].values())
    assert all(row["full_decode"] is True for row in validation["videos"].values())
    assert verified["publication_run"]["run_manifest"]["sha256"] == _sha(
        publication_run / "run_manifest.json"
    )

    with (publication_run / "stdout.log").open("ab") as stream:
        stream.write(b"tamper\n")
    with pytest.raises(
        video_artifacts.PPOVideoArtifactError, match="stdout.log record is stale"
    ):
        video_artifacts.verify_final_video_publication(
            result["video_validation"],
            result["video_checksums"],
            output_root=output,
            expected_improved_checkpoint_sha256=improved_hash,
            ffmpeg=ffmpeg,
        )
