"""Publish auditable final videos from two accepted live viewport episodes.

The inputs to this module are *complete episode* captures produced by
``video_runtime.capture_live_policy_video``.  This module never selects a
phase, trims an action, changes playback speed, or joins episodes in time.
The comparison is a spatial composite whose shorter side is extended only by
cloning its final frame, as required for a real-time side-by-side comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.video_capture import (
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    find_ffmpeg,
    sha256_file,
    validate_mp4,
)
from wlr50_clean.infrastructure.scene_factory import CAMERA_EYE_M, CAMERA_TARGET_M

from .artifacts import (
    atomic_write_json,
    verify_checksum_manifest,
    write_checksum_manifest,
)


SOURCE_MANIFEST_NAME = "ppo_video_source_manifest.json"
SOURCE_VIDEO_NAME = "actual_viewport_video.mp4"
SOURCE_LEDGER_NAME = "viewport_frame_ledger.jsonl"
SOURCE_TRACE_NAME = "policy_trace.jsonl"

FSM_VIDEO_NAME = "fsm_baseline_clean.mp4"
PPO_VIDEO_NAME = "ppo_improved_checkpoint_clean.mp4"
COMPARISON_VIDEO_NAME = "fsm_vs_ppo_improved.mp4"
DIAGNOSTIC_VIDEO_NAME = "ppo_improved_diagnostic.mp4"
VIDEO_VALIDATION_NAME = "video_validation.json"
VIDEO_CHECKSUM_NAME = "video_checksums.sha256"
DIAGNOSTIC_ASS_NAME = "ppo_improved_diagnostic.ass"

STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
_FRAME_PERIOD_S = 1.0 / VIDEO_FPS
_SOURCE_SCHEMA = "wlr50_clean.ppo_video_source_episode.v1"
_PUBLICATION_SCHEMA = "wlr50_clean.ppo_final_videos.v1"


class PPOVideoArtifactError(RuntimeError):
    """A source or generated final video cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SourceEpisode:
    """Validated provenance and telemetry for one video source episode."""

    role: str
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    recorder_manifest_path: Path
    recorder_manifest: Mapping[str, Any]
    trial_manifest_path: Path
    trial_manifest: Mapping[str, Any]
    video_path: Path
    ledger_path: Path
    trace_path: Path
    trace: tuple[Mapping[str, Any], ...]
    source_identity: Mapping[str, Any]
    camera: Mapping[str, Any]
    video_validation: Mapping[str, Any]
    checkpoint_path: Path | None
    checkpoint_sha256: str | None

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def frame_count(self) -> int:
        return int(self.video_validation["frame_count"])

    @property
    def duration_s(self) -> float:
        return float(self.video_validation["duration_s"])


@dataclass(frozen=True, slots=True)
class FinalVideoPublication:
    """Paths and verified evidence returned after immutable publication."""

    videos: Mapping[str, Path]
    validation_path: Path
    checksum_path: Path
    diagnostic_ass_path: Path
    checksum_verification: Mapping[str, Any]


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise PPOVideoArtifactError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PPOVideoArtifactError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PPOVideoArtifactError(f"{label} must contain a JSON object: {path}")
    return value


def _inside(root: Path, value: Any, *, label: str) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PPOVideoArtifactError(f"{label} points outside its source episode: {candidate}") from exc
    return candidate


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise PPOVideoArtifactError(f"{label} is not finite")
    return result


def _read_jsonl(path: Path, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise PPOVideoArtifactError(f"{label} is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PPOVideoArtifactError(
                f"{label} row {line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise PPOVideoArtifactError(f"{label} row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise PPOVideoArtifactError(f"{label} is empty: {path}")
    return tuple(rows)


def _validate_ledger(path: Path, recorder: Mapping[str, Any]) -> int:
    rows = _read_jsonl(path, label="viewport frame ledger")
    expected_indices = list(range(len(rows)))
    indices: list[int] = []
    steps: list[int] = []
    times: list[float] = []
    for index, row in enumerate(rows):
        try:
            indices.append(int(row["encoded_frame_index"]))
            steps.append(int(row["sim_step"]))
            times.append(_finite(row["sim_time_s"], label=f"ledger row {index} sim_time_s"))
        except (KeyError, TypeError, ValueError) as exc:
            raise PPOVideoArtifactError(f"viewport ledger row {index} is incomplete") from exc
    if indices != expected_indices:
        raise PPOVideoArtifactError("viewport ledger frame indices are not contiguous from zero")
    if not steps or steps[0] != 0 or any(
        right - left != 8 for left, right in zip(steps, steps[1:])
    ):
        raise PPOVideoArtifactError(
            "viewport ledger is not the exact global 120 Hz to 15 Hz cadence"
        )
    if any(
        abs(actual - expected_step / 120.0) > 1.0e-9
        for actual, expected_step in zip(times, steps, strict=True)
    ):
        raise PPOVideoArtifactError(
            "viewport ledger timestamps do not match their physical simulation ticks"
        )
    if int(recorder.get("frame_count", -1)) != len(rows):
        raise PPOVideoArtifactError("viewport recorder frame count differs from its ledger")
    return len(rows)


def _validate_trace(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    require_nonzero_residual: bool,
) -> tuple[Mapping[str, Any], ...]:
    rows = _read_jsonl(path, label="15 Hz policy trace")
    decision_indices: list[int] = []
    physics_ticks: list[int] = []
    times: list[float] = []
    phases: list[str] = []
    maximum_residual = 0.0
    expected_seed = int(manifest["seed"])
    for index, row in enumerate(rows):
        try:
            row_seed = int(row["seed"])
            decision_indices.append(int(row["decision_index"]))
            physics_ticks.append(int(row["physics_tick"]))
            times.append(_finite(row["sim_time_s"], label=f"trace row {index} sim_time_s"))
            phase = str(row["state_id"])
            for field in (
                "pitch_error_rad",
                "roll_error_rad",
                "pitch_rate_rad_s",
                "roll_rate_rad_s",
            ):
                _finite(row[field], label=f"trace row {index} {field}")
            residual = tuple(
                _finite(value, label=f"trace row {index} residual")
                for value in row["residual_full12"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PPOVideoArtifactError(f"policy trace row {index} is incomplete") from exc
        if phase not in STATE_IDS:
            raise PPOVideoArtifactError(f"policy trace row {index} has invalid phase {phase!r}")
        if row_seed != expected_seed:
            raise PPOVideoArtifactError(f"policy trace row {index} has the wrong seed")
        if len(residual) != 12:
            raise PPOVideoArtifactError(f"policy trace row {index} is not Full12")
        maximum_residual = max(maximum_residual, *(abs(value) for value in residual))
        phases.append(phase)
    if decision_indices != list(range(len(rows))):
        raise PPOVideoArtifactError("policy decision indices are not contiguous from zero")
    tick_deltas = [
        right - left for left, right in zip(physics_ticks, physics_ticks[1:])
    ]
    if (
        not physics_ticks
        or physics_ticks[0] != 8
        or any(delta != 8 for delta in tick_deltas[:-1])
        or (tick_deltas and not 1 <= tick_deltas[-1] <= 8)
    ):
        raise PPOVideoArtifactError(
            "policy trace does not follow the actual 15 Hz decision cadence"
        )
    if any(
        abs(actual - tick / 120.0) > 1.0e-9
        for actual, tick in zip(times, physics_ticks, strict=True)
    ):
        raise PPOVideoArtifactError(
            "policy trace timestamps differ from their physical ticks"
        )
    if set(phases) != set(STATE_IDS):
        raise PPOVideoArtifactError("policy trace does not contain every phase P01-P13")
    if int(manifest.get("decision_count", -1)) != len(rows):
        raise PPOVideoArtifactError("source manifest decision count differs from policy trace")
    if str(rows[-1].get("termination_reason")) != "SUCCESS":
        raise PPOVideoArtifactError("policy trace does not end with authoritative SUCCESS")
    if require_nonzero_residual and maximum_residual <= 1.0e-12:
        raise PPOVideoArtifactError("PPO video trace contains only zero residuals")
    if not require_nonzero_residual and maximum_residual > 1.0e-12:
        raise PPOVideoArtifactError("FSM baseline video trace is not the zero-residual policy")
    return rows


def _validate_calibration(trial: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    calibration = trial.get("ppo_calibration")
    if not isinstance(calibration, Mapping) or calibration.get("quality_passed") is not True:
        raise PPOVideoArtifactError("source trial lacks a passing reset calibration")
    home = tuple(
        _finite(value, label="home joint calibration")
        for value in calibration.get("home_joint_positions_deg8", ())
    )
    level = tuple(
        _finite(value, label="level quaternion calibration")
        for value in calibration.get("level_reference_orientation_wxyz", ())
    )
    if len(home) != 8 or len(level) != 4:
        raise PPOVideoArtifactError("source reset calibration has the wrong dimensions")
    return home, level


def _validate_source_identity(manifest: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
    identity = manifest.get("source_identity")
    if not isinstance(identity, Mapping):
        raise PPOVideoArtifactError(f"{role} source lacks reset identity evidence")
    hashes: dict[str, str] = {}
    for key in (
        "environment_hash",
        "robot_asset_hash",
        "controller_hash",
        "motion_contract_hash",
    ):
        value = str(identity.get(key, "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise PPOVideoArtifactError(f"{role} source identity has an invalid {key}")
        hashes[key] = value
    vectors: dict[str, tuple[float, ...]] = {}
    for key, dimension in (
        ("initial_root_state", 13),
        ("initial_joint_state", 24),
        ("obstacle_pose", 3),
    ):
        try:
            vector = tuple(
                _finite(value, label=f"{role} source identity {key}")
                for value in identity.get(key, ())
            )
        except TypeError as exc:
            raise PPOVideoArtifactError(
                f"{role} source identity {key} is not a vector"
            ) from exc
        if len(vector) != dimension:
            raise PPOVideoArtifactError(
                f"{role} source identity {key} must have dimension {dimension}"
            )
        vectors[key] = vector
    return {**hashes, **vectors}


def _validate_camera(
    manifest: Mapping[str, Any],
    recorder: Mapping[str, Any],
    *,
    role: str,
) -> Mapping[str, Any]:
    camera = manifest.get("camera")
    if not isinstance(camera, Mapping):
        raise PPOVideoArtifactError(f"{role} source lacks locked live camera evidence")
    try:
        eye = tuple(
            _finite(value, label=f"{role} camera eye")
            for value in camera.get("eye_m", ())
        )
        target = tuple(
            _finite(value, label=f"{role} camera target")
            for value in camera.get("target_m", ())
        )
        resolution = tuple(int(value) for value in camera.get("resolution", ()))
    except (TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{role} camera evidence is invalid") from exc
    fps = _finite(camera.get("fps"), label=f"{role} camera fps")
    if (
        len(eye) != 3
        or len(target) != 3
        or any(
            abs(actual - expected) > 1.0e-12
            for actual, expected in zip(eye, CAMERA_EYE_M, strict=True)
        )
        or any(
            abs(actual - expected) > 1.0e-12
            for actual, expected in zip(target, CAMERA_TARGET_M, strict=True)
        )
        or resolution != (VIDEO_WIDTH, VIDEO_HEIGHT)
        or not math.isclose(fps, VIDEO_FPS, rel_tol=0.0, abs_tol=1.0e-12)
        or camera.get("locked_scene_snapshot") is not True
        or camera.get("active_viewport_resolution_verified") is not True
        or camera.get("render_product_path") != recorder.get("render_product_path")
        or camera.get("viewport_identity") != recorder.get("viewport_identity")
    ):
        raise PPOVideoArtifactError(f"{role} source camera differs from the locked live view")
    return {
        "eye_m": eye,
        "target_m": target,
        "resolution": resolution,
        "fps": fps,
        "render_product_path": recorder.get("render_product_path"),
    }


def _load_source_episode(
    root_value: Path | str,
    *,
    role: str,
    ffmpeg: Path,
) -> SourceEpisode:
    root = Path(root_value).resolve()
    if not root.is_dir():
        raise PPOVideoArtifactError(f"{role} source directory is missing: {root}")
    manifest_path = root / SOURCE_MANIFEST_NAME
    manifest = _load_json(manifest_path, label=f"{role} source manifest")
    if manifest.get("schema") != _SOURCE_SCHEMA:
        raise PPOVideoArtifactError(f"{role} source manifest has the wrong schema")
    try:
        source_seed = int(manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{role} source manifest has an invalid seed") from exc
    if source_seed < 0:
        raise PPOVideoArtifactError(f"{role} source manifest has a negative seed")
    if (
        manifest.get("task_success") is not True
        or manifest.get("completed_p01_p13") is not True
        or manifest.get("body_collision") is not False
        or manifest.get("wheel_only_climb") is not False
        or manifest.get("stitched") is not False
        or manifest.get("speed_modified") is not False
        or manifest.get("frame_interpolation") is not False
    ):
        raise PPOVideoArtifactError(f"{role} source did not pass physical/video acceptance")
    try:
        capture_process_id = int(manifest.get("capture_process_id", -1))
        episode_count = int(manifest.get("episode_count", -1))
    except (TypeError, ValueError) as exc:
        raise PPOVideoArtifactError(f"{role} source process evidence is invalid") from exc
    if (
        manifest.get("fresh_process_single_episode") is not True
        or episode_count != 1
        or capture_process_id <= 0
    ):
        raise PPOVideoArtifactError(
            f"{role} source is not a fresh single-episode capture"
        )
    process_instance_id = str(manifest.get("capture_process_instance_id", ""))
    if re.fullmatch(r"[0-9a-f]{32}", process_instance_id) is None:
        raise PPOVideoArtifactError(
            f"{role} source lacks a valid fresh process instance identity"
        )
    reset_evidence = manifest.get("reset_evidence")
    if (
        not isinstance(reset_evidence, Mapping)
        or reset_evidence.get("reset_count") != 1
        or reset_evidence.get("reset_options") != {}
        or reset_evidence.get("training_phase_snapshot") is not None
        or reset_evidence.get("reset_global_simulation_resets") != 0
        or reset_evidence.get("reset_simulation_forward_syncs") != 0
    ):
        raise PPOVideoArtifactError(
            f"{role} source lacks exact first-reset freshness evidence"
        )
    if manifest.get("pre_action_observation_refreshed") is not True:
        raise PPOVideoArtifactError(
            f"{role} source used a stale pre-roll policy observation"
        )
    refresh_evidence = manifest.get("pre_action_refresh_evidence")
    if (
        not isinstance(refresh_evidence, Mapping)
        or refresh_evidence.get("schema")
        != "wlr50_clean.ppo_video_pre_action_refresh.v1"
        or refresh_evidence.get("episode_reader_reinitialized") is not True
        or refresh_evidence.get("episode_reader_first_logical_tick") != 0
        or refresh_evidence.get("controller_frame_preserved") is not True
        or refresh_evidence.get("controller_logical_tick") != 0
        or refresh_evidence.get("simulation_reset_performed") is not False
        or refresh_evidence.get("fsm_step_performed") is not False
    ):
        raise PPOVideoArtifactError(
            f"{role} source lacks an exact post-pre-roll sensing refresh"
        )
    if int(manifest.get("reset_info_recording_access_count", -1)) != 0:
        raise PPOVideoArtifactError(f"{role} source accessed Recording at runtime")
    source_identity = _validate_source_identity(manifest, role=role)
    episode_duration = _finite(manifest.get("duration_s"), label=f"{role} episode duration")
    expected_video_duration = _finite(
        manifest.get("video_duration_expected_s"), label=f"{role} expected video duration"
    )
    if episode_duration <= 0.0 or episode_duration > 200.0 or expected_video_duration > 200.0:
        raise PPOVideoArtifactError(f"{role} source exceeds the 200 second limit")
    pre_roll = _finite(
        manifest.get("pre_action_physical_hold_s"), label=f"{role} pre-action hold"
    )
    post_roll = _finite(
        manifest.get("post_success_physical_hold_s"), label=f"{role} post-success hold"
    )
    if not 0.5 <= pre_roll <= 1.0 or not 1.0 <= post_roll <= 2.0:
        raise PPOVideoArtifactError(f"{role} source has invalid pre/post action context")
    expected_pre_ticks = round(pre_roll * 120.0)
    if (
        refresh_evidence.get("physical_pre_action_ticks") != expected_pre_ticks
        or refresh_evidence.get("pre_roll_reader_last_logical_tick")
        != expected_pre_ticks
    ):
        raise PPOVideoArtifactError(
            f"{role} source refresh evidence differs from its physical pre-roll"
        )
    action_start = _finite(
        manifest.get("semantic_action_start_video_s"), label=f"{role} action start"
    )
    task_success_time = _finite(
        manifest.get("semantic_task_success_video_s"), label=f"{role} task success time"
    )
    if (
        abs(action_start - pre_roll) > 1.0e-9
        or abs(task_success_time - (pre_roll + episode_duration)) > 1.0e-9
        or abs(expected_video_duration - (pre_roll + episode_duration + post_roll)) > 1.0e-9
    ):
        raise PPOVideoArtifactError(f"{role} semantic video timeline is inconsistent")

    video_path = (root / SOURCE_VIDEO_NAME).resolve()
    declared_video = _inside(root, manifest.get("raw_video", ""), label=f"{role} raw video")
    if declared_video != video_path or not video_path.is_file():
        raise PPOVideoArtifactError(f"{role} manifest is not bound to {SOURCE_VIDEO_NAME}")
    video_sha = sha256_file(video_path)
    if str(manifest.get("raw_video_sha256", "")).lower() != video_sha:
        raise PPOVideoArtifactError(f"{role} raw video hash does not match its source manifest")

    recorder_manifest_path = _inside(
        root, manifest.get("recorder_manifest", ""), label=f"{role} recorder manifest"
    )
    recorder = _load_json(recorder_manifest_path, label=f"{role} recorder manifest")
    if recorder.get("valid") is not True or recorder.get("stitched") is not False:
        raise PPOVideoArtifactError(f"{role} viewport recorder did not pass")
    if recorder.get("speed_modified") is not False:
        raise PPOVideoArtifactError(f"{role} viewport recorder changed video speed")
    camera = _validate_camera(manifest, recorder, role=role)
    recorder_video = _inside(
        root, recorder.get("video_path", ""), label=f"{role} recorder video"
    )
    if recorder_video != video_path or str(recorder.get("video_sha256", "")).lower() != video_sha:
        raise PPOVideoArtifactError(f"{role} recorder manifest is bound to a different video")
    full_decode = recorder.get("full_decode")
    if not isinstance(full_decode, Mapping) or full_decode.get("valid") is not True:
        raise PPOVideoArtifactError(f"{role} recorder did not fully decode its source video")

    ledger_path = (root / SOURCE_LEDGER_NAME).resolve()
    declared_ledger = _inside(root, recorder.get("ledger_path", ""), label=f"{role} ledger")
    if declared_ledger != ledger_path or not ledger_path.is_file():
        raise PPOVideoArtifactError(f"{role} recorder is not bound to {SOURCE_LEDGER_NAME}")
    if str(recorder.get("ledger_sha256", "")).lower() != sha256_file(ledger_path):
        raise PPOVideoArtifactError(f"{role} viewport ledger hash mismatch")
    ledger_count = _validate_ledger(ledger_path, recorder)

    trial_manifest_path = _inside(
        root, manifest.get("trial_manifest", ""), label=f"{role} trial manifest"
    )
    trial_manifest = _load_json(trial_manifest_path, label=f"{role} trial manifest")
    _validate_calibration(trial_manifest)

    trace_path = (root / SOURCE_TRACE_NAME).resolve()
    declared_trace = _inside(
        root, manifest.get("policy_trace", ""), label=f"{role} policy trace"
    )
    if declared_trace != trace_path or not trace_path.is_file():
        raise PPOVideoArtifactError(f"{role} source is not bound to {SOURCE_TRACE_NAME}")
    if str(manifest.get("policy_trace_sha256", "")).lower() != sha256_file(trace_path):
        raise PPOVideoArtifactError(f"{role} policy trace hash mismatch")
    if not math.isclose(
        _finite(manifest.get("policy_trace_rate_hz"), label=f"{role} trace rate"),
        15.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise PPOVideoArtifactError(f"{role} policy trace is not the 15 Hz control trace")
    trace = _validate_trace(
        trace_path,
        manifest=manifest,
        require_nonzero_residual=role == "ppo",
    )
    if int(manifest.get("policy_trace_rows", -1)) != len(trace):
        raise PPOVideoArtifactError(f"{role} source trace row count mismatch")
    if abs(float(trace[-1]["sim_time_s"]) - episode_duration) > _FRAME_PERIOD_S + 1.0e-9:
        raise PPOVideoArtifactError(f"{role} policy trace does not reach task termination")
    if role == "fsm":
        if manifest.get("policy_label") != "fsm_zero_residual":
            raise PPOVideoArtifactError("FSM source is not labelled as zero residual")
        if manifest.get("deterministic_mean_policy") is not False:
            raise PPOVideoArtifactError("FSM source unexpectedly claims PPO inference")
        if manifest.get("source_checkpoint") is not None:
            raise PPOVideoArtifactError("FSM zero-residual source names a checkpoint")
        if manifest.get("used_preinitialized_fresh_episode") is not False:
            raise PPOVideoArtifactError(
                "FSM source did not perform its one reset inside capture"
            )
        checkpoint_path = None
        checkpoint_sha256 = None
    else:
        if manifest.get("policy_label") != "ppo_deterministic_mean":
            raise PPOVideoArtifactError("PPO source is not deterministic mean inference")
        if manifest.get("deterministic_mean_policy") is not True:
            raise PPOVideoArtifactError("PPO source does not assert deterministic mean inference")
        if manifest.get("used_preinitialized_fresh_episode") is not True:
            raise PPOVideoArtifactError(
                "PPO source was not captured from the runner's first fresh reset"
            )
        checkpoint_text = manifest.get("source_checkpoint")
        if not checkpoint_text:
            raise PPOVideoArtifactError("PPO source checkpoint is missing")
        checkpoint_path = Path(str(checkpoint_text)).resolve()
        if not checkpoint_path.is_file():
            raise PPOVideoArtifactError(f"PPO source checkpoint is missing: {checkpoint_path}")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if str(manifest.get("source_checkpoint_sha256", "")).lower() != checkpoint_sha256:
            raise PPOVideoArtifactError("PPO source checkpoint hash mismatch")
        checkpoint_manifest_text = manifest.get("source_checkpoint_manifest")
        if not checkpoint_manifest_text:
            raise PPOVideoArtifactError("PPO source checkpoint manifest is missing")
        checkpoint_manifest_path = Path(str(checkpoint_manifest_text)).resolve()
        checkpoint_manifest = _load_json(
            checkpoint_manifest_path,
            label="PPO source checkpoint manifest",
        )
        checkpoint_manifest_sha256 = sha256_file(checkpoint_manifest_path)
        if (
            str(manifest.get("source_checkpoint_manifest_sha256", "")).lower()
            != checkpoint_manifest_sha256
            or Path(str(checkpoint_manifest.get("checkpoint_path", ""))).resolve()
            != checkpoint_path
            or str(checkpoint_manifest.get("checkpoint_sha256", "")).lower()
            != checkpoint_sha256
            or checkpoint_manifest.get("publication_role") != "improved"
            or checkpoint_manifest.get("validation_promotion_authorized") is not True
            or checkpoint_manifest.get("locked_test_authorized") is not True
            or checkpoint_manifest.get("promotion_authorized") is not True
        ):
            raise PPOVideoArtifactError(
                "PPO source checkpoint manifest is not the promoted improved artifact"
            )
        load_provenance = manifest.get("checkpoint_load_provenance")
        if (
            not isinstance(load_provenance, Mapping)
            or Path(str(load_provenance.get("checkpoint_path", ""))).resolve()
            != checkpoint_path
            or str(load_provenance.get("checkpoint_sha256", "")).lower()
            != checkpoint_sha256
            or Path(str(load_provenance.get("manifest_path", ""))).resolve()
            != checkpoint_manifest_path
            or str(load_provenance.get("manifest_sha256", "")).lower()
            != checkpoint_manifest_sha256
            or load_provenance.get("checkpoint_infos_match_manifest") is not True
        ):
            raise PPOVideoArtifactError(
                "PPO source lacks matching strict checkpoint load evidence"
            )

    source_validation = validate_mp4(
        video_path,
        ffmpeg=ffmpeg,
        expected_width=VIDEO_WIDTH,
        expected_height=VIDEO_HEIGHT,
        expected_fps=VIDEO_FPS,
        expected_frame_count=ledger_count,
        maximum_duration_s=200.0,
        stitched=False,
        speed_modified=False,
        require_sane_container_duration=False,
    )
    if source_validation.get("valid") is not True:
        raise PPOVideoArtifactError(f"{role} source video failed independent full decode")
    if abs(float(source_validation["duration_s"]) - expected_video_duration) > 2.0 * _FRAME_PERIOD_S:
        raise PPOVideoArtifactError(f"{role} source video duration differs from its physical timeline")
    if int(source_validation["frame_count"]) != ledger_count:
        raise PPOVideoArtifactError(f"{role} source video differs from its viewport ledger")

    return SourceEpisode(
        role=role,
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        recorder_manifest_path=recorder_manifest_path,
        recorder_manifest=recorder,
        trial_manifest_path=trial_manifest_path,
        trial_manifest=trial_manifest,
        video_path=video_path,
        ledger_path=ledger_path,
        trace_path=trace_path,
        trace=trace,
        source_identity=source_identity,
        camera=camera,
        video_validation=source_validation,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )


def _quaternion_difference(left: Sequence[float], right: Sequence[float]) -> float:
    direct = max(abs(a - b) for a, b in zip(left, right, strict=True))
    sign_flipped = max(abs(a + b) for a, b in zip(left, right, strict=True))
    return min(direct, sign_flipped)


def _validate_pair(fsm: SourceEpisode, ppo: SourceEpisode) -> Mapping[str, Any]:
    if fsm.seed != ppo.seed:
        raise PPOVideoArtifactError("FSM and PPO source videos use different seeds")
    if fsm.seed != 4001:
        raise PPOVideoArtifactError("final source videos do not use locked video seed 4001")
    if (
        fsm.root == ppo.root
        or int(fsm.manifest["capture_process_id"])
        == int(ppo.manifest["capture_process_id"])
        or fsm.manifest["capture_process_instance_id"]
        == ppo.manifest["capture_process_instance_id"]
    ):
        raise PPOVideoArtifactError(
            "FSM and PPO sources were not captured in independent fresh processes"
        )
    recorder_keys = ("capture_backend", "render_product_path", "width", "height", "fps")
    mismatches = [
        key
        for key in recorder_keys
        if fsm.recorder_manifest.get(key) != ppo.recorder_manifest.get(key)
    ]
    if mismatches:
        raise PPOVideoArtifactError(
            f"FSM and PPO source videos use different camera/capture settings: {mismatches}"
        )
    camera_vector_differences = {
        key: max(
            abs(left - right)
            for left, right in zip(
                fsm.camera[key], ppo.camera[key], strict=True
            )
        )
        for key in ("eye_m", "target_m")
    }
    if (
        camera_vector_differences["eye_m"] > 1.0e-12
        or camera_vector_differences["target_m"] > 1.0e-12
        or fsm.camera["resolution"] != ppo.camera["resolution"]
        or fsm.camera["fps"] != ppo.camera["fps"]
    ):
        raise PPOVideoArtifactError("FSM and PPO source videos use different cameras")
    identity_hashes = (
        "environment_hash",
        "robot_asset_hash",
        "controller_hash",
        "motion_contract_hash",
    )
    identity_mismatches = [
        key
        for key in identity_hashes
        if fsm.source_identity[key] != ppo.source_identity[key]
    ]
    if identity_mismatches:
        raise PPOVideoArtifactError(
            f"FSM and PPO source identities differ: {identity_mismatches}"
        )
    vector_differences = {
        key: max(
            abs(left - right)
            for left, right in zip(
                fsm.source_identity[key], ppo.source_identity[key], strict=True
            )
        )
        for key in ("initial_root_state", "initial_joint_state", "obstacle_pose")
    }
    if (
        vector_differences["initial_root_state"] > 1.0e-9
        or vector_differences["initial_joint_state"] > 1.0e-9
        or vector_differences["obstacle_pose"] > 1.0e-12
    ):
        raise PPOVideoArtifactError(
            "FSM and PPO source identities differ in initial state or obstacle pose"
        )
    fsm_home, fsm_level = _validate_calibration(fsm.trial_manifest)
    ppo_home, ppo_level = _validate_calibration(ppo.trial_manifest)
    home_difference = max(abs(a - b) for a, b in zip(fsm_home, ppo_home, strict=True))
    level_difference = _quaternion_difference(fsm_level, ppo_level)
    if home_difference > 1.0e-6 or level_difference > 1.0e-6:
        raise PPOVideoArtifactError("FSM and PPO videos do not share the same calibrated initial state")
    return {
        "same_seed": True,
        "seed": fsm.seed,
        "independent_fresh_capture_processes": True,
        "fsm_capture_process_id": int(fsm.manifest["capture_process_id"]),
        "ppo_capture_process_id": int(ppo.manifest["capture_process_id"]),
        "fsm_capture_process_instance_id": fsm.manifest[
            "capture_process_instance_id"
        ],
        "ppo_capture_process_instance_id": ppo.manifest[
            "capture_process_instance_id"
        ],
        "same_live_environment_contract": True,
        "environment_hash": fsm.source_identity["environment_hash"],
        "robot_asset_hash": fsm.source_identity["robot_asset_hash"],
        "controller_hash": fsm.source_identity["controller_hash"],
        "motion_contract_hash": fsm.source_identity["motion_contract_hash"],
        "same_obstacle_contract": True,
        "obstacle_pose": list(fsm.source_identity["obstacle_pose"]),
        "obstacle_pose_max_abs_difference_m": vector_differences["obstacle_pose"],
        "same_camera": True,
        "camera_eye_m": list(fsm.camera["eye_m"]),
        "camera_target_m": list(fsm.camera["target_m"]),
        "camera_eye_max_abs_difference_m": camera_vector_differences["eye_m"],
        "camera_target_max_abs_difference_m": camera_vector_differences[
            "target_m"
        ],
        "capture_backend": fsm.recorder_manifest.get("capture_backend"),
        "render_product_path": fsm.recorder_manifest.get("render_product_path"),
        "same_resolution": True,
        "source_resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
        "same_initial_state": True,
        "initial_root_state_max_abs_difference": vector_differences[
            "initial_root_state"
        ],
        "initial_joint_state_max_abs_difference": vector_differences[
            "initial_joint_state"
        ],
        "home_joint_max_abs_difference_deg": home_difference,
        "level_quaternion_sign_invariant_max_abs_difference": level_difference,
    }


def _run_ffmpeg(command: Sequence[str], *, label: str) -> None:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=1200,
    )
    if completed.returncode != 0:
        tail = completed.stderr[-4000:].replace("\r", " ").replace("\n", " ")
        raise PPOVideoArtifactError(f"{label} ffmpeg failed ({completed.returncode}): {tail}")


def _encode_clean_source(ffmpeg: Path, source: Path, destination: Path, *, label: str) -> None:
    _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "setpts=PTS-STARTPTS,format=yuv420p",
            "-an",
            "-sn",
            "-dn",
            "-r",
            f"{VIDEO_FPS:g}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        label=label,
    )


def _comparison_filter(fsm_frames: int, ppo_frames: int) -> tuple[str, str, int]:
    output_frames = max(fsm_frames, ppo_frames)
    output_duration = output_frames / VIDEO_FPS
    if fsm_frames < ppo_frames:
        padding_side = "fsm"
        padding_frames = ppo_frames - fsm_frames
        padding_duration = padding_frames / VIDEO_FPS
        left = (
            f"[0:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"tpad=stop_mode=clone:stop_duration={padding_duration:.9f},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=218:h=46:color=black@0.55:t=fill,"
            "drawtext=text='FSM baseline':x=28:y=26:fontcolor=white:fontsize=26[left]"
        )
        right = (
            f"[1:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=220:h=46:color=black@0.55:t=fill,"
            "drawtext=text='PPO improved':x=28:y=26:fontcolor=white:fontsize=26[right]"
        )
    elif ppo_frames < fsm_frames:
        padding_side = "ppo"
        padding_frames = fsm_frames - ppo_frames
        padding_duration = padding_frames / VIDEO_FPS
        left = (
            f"[0:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=218:h=46:color=black@0.55:t=fill,"
            "drawtext=text='FSM baseline':x=28:y=26:fontcolor=white:fontsize=26[left]"
        )
        right = (
            f"[1:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"tpad=stop_mode=clone:stop_duration={padding_duration:.9f},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=220:h=46:color=black@0.55:t=fill,"
            "drawtext=text='PPO improved':x=28:y=26:fontcolor=white:fontsize=26[right]"
        )
    else:
        padding_side = "none"
        padding_frames = 0
        left = (
            f"[0:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=218:h=46:color=black@0.55:t=fill,"
            "drawtext=text='FSM baseline':x=28:y=26:fontcolor=white:fontsize=26[left]"
        )
        right = (
            f"[1:v]setpts=PTS-STARTPTS,fps={VIDEO_FPS:g},"
            f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS,"
            "drawbox=x=16:y=16:w=220:h=46:color=black@0.55:t=fill,"
            "drawtext=text='PPO improved':x=28:y=26:fontcolor=white:fontsize=26[right]"
        )
    return (
        f"{left};{right};[left][right]hstack=inputs=2,format=yuv420p[outv]",
        padding_side,
        padding_frames,
    )


def _encode_comparison(
    ffmpeg: Path,
    fsm: Path,
    ppo: Path,
    destination: Path,
    *,
    fsm_frames: int,
    ppo_frames: int,
) -> tuple[str, int]:
    filter_graph, padding_side, padding_frames = _comparison_filter(fsm_frames, ppo_frames)
    _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(fsm),
            "-i",
            str(ppo),
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            f"{VIDEO_FPS:g}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        label="real-time FSM-versus-PPO comparison",
    )
    return padding_side, padding_frames


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100.0)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _ass_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def build_diagnostic_ass(
    trace: Sequence[Mapping[str, Any]],
    *,
    action_start_video_s: float,
    video_duration_s: float,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> str:
    """Build subtitles whose changing values come directly from the 15 Hz trace."""

    if not trace:
        raise PPOVideoArtifactError("cannot build diagnostic overlay from an empty trace")
    start_offset = _finite(action_start_video_s, label="diagnostic action start")
    duration = _finite(video_duration_s, label="diagnostic video duration")
    if not 0.0 <= start_offset < duration:
        raise PPOVideoArtifactError("diagnostic action start lies outside the video")
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Telemetry,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,"
            "&H90000000,0,0,0,0,100,100,0,0,3,1,0,2,28,28,28,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    first_time = start_offset + _finite(trace[0]["sim_time_s"], label="first trace time")
    if first_time > start_offset:
        lines.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(start_offset)},{_ass_timestamp(first_time)},Telemetry,,0,0,0,,"
            "P01 | awaiting first physical 15 Hz telemetry sample"
        )
    for index, row in enumerate(trace):
        start = start_offset + _finite(row["sim_time_s"], label=f"trace row {index} time")
        if index + 1 < len(trace):
            end = start_offset + _finite(
                trace[index + 1]["sim_time_s"], label=f"trace row {index + 1} time"
            )
        else:
            end = duration
        start = min(max(start_offset, start), duration)
        end = min(max(start + 0.01, end), duration)
        if end <= start:
            continue
        residual = tuple(float(value) for value in row["residual_full12"])
        residual_rms = math.sqrt(sum(value * value for value in residual) / len(residual))
        pitch = math.degrees(float(row["pitch_error_rad"]))
        roll = math.degrees(float(row["roll_error_rad"]))
        pitch_rate = float(row["pitch_rate_rad_s"])
        roll_rate = float(row["roll_rate_rad_s"])
        result = row.get("termination_reason") or "RUNNING"
        first_line = _ass_text(
            f"phase {row['state_id']}   pitch {pitch:+.3f} deg   "
            f"pitch rate {pitch_rate:+.4f} rad/s"
        )
        second_line = _ass_text(
            f"roll {roll:+.3f} deg   roll rate {roll_rate:+.4f} rad/s   "
            f"residual RMS {residual_rms:.6f}   result {result}"
        )
        text = first_line + r"\N" + second_line
        lines.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(start)},{_ass_timestamp(end)},Telemetry,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _ffmpeg_filter_path(path: Path) -> str:
    text = path.resolve().as_posix()
    text = text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
    return f"'{text}'"


def _encode_diagnostic(ffmpeg: Path, source: Path, ass_path: Path, destination: Path) -> None:
    _run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-vf",
            f"ass=filename={_ffmpeg_filter_path(ass_path)}",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-r",
            f"{VIDEO_FPS:g}",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        label="PPO diagnostic overlay",
    )


def _require_generated_video(
    path: Path,
    *,
    ffmpeg: Path,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
) -> Mapping[str, Any]:
    validation = validate_mp4(
        path,
        ffmpeg=ffmpeg,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_fps=VIDEO_FPS,
        expected_frame_count=expected_frame_count,
        maximum_duration_s=200.0,
        stitched=False,
        speed_modified=False,
        require_sane_container_duration=True,
    )
    if validation.get("valid") is not True:
        raise PPOVideoArtifactError(f"generated video failed full validation: {path.name}")
    return validation


def _video_record(
    validation: Mapping[str, Any],
    *,
    final_path: Path,
    source_episode: str | Sequence[str],
    source_checkpoint: str | Sequence[str],
    source_checkpoint_sha256: str | Sequence[str | None],
    source_seed: int,
    processing: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(validation),
        "path": str(final_path),
        "pix_fmt": validation.get("pixel_format"),
        "source_episode": source_episode,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_seed": int(source_seed),
        "full_decode": validation.get("full_decode") is True,
        "monotonic": validation.get("timestamps_monotonic") is True,
        "stitched": False,
        "speed_modified": False,
        "processing": dict(processing),
    }


def _publish_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise PPOVideoArtifactError(f"refusing to overwrite final artifact: {destination}") from exc
    source.unlink()


def publish_final_videos(
    *,
    fsm_source_dir: Path | str,
    ppo_source_dir: Path | str,
    output_root: Path | str,
    ffmpeg: Path | str | None = None,
) -> FinalVideoPublication:
    """Create and validate the four immutable phase-residual PPO final videos."""

    executable = find_ffmpeg(ffmpeg)
    output = Path(output_root).resolve()
    videos_dir = output / "videos"
    manifests_dir = output / "manifests"
    destinations = {
        "fsm_baseline": videos_dir / FSM_VIDEO_NAME,
        "ppo_improved": videos_dir / PPO_VIDEO_NAME,
        "comparison": videos_dir / COMPARISON_VIDEO_NAME,
        "ppo_diagnostic": videos_dir / DIAGNOSTIC_VIDEO_NAME,
    }
    validation_path = manifests_dir / VIDEO_VALIDATION_NAME
    checksum_path = manifests_dir / VIDEO_CHECKSUM_NAME
    ass_path = manifests_dir / DIAGNOSTIC_ASS_NAME
    all_destinations = (*destinations.values(), validation_path, checksum_path, ass_path)
    existing = [path for path in all_destinations if path.exists()]
    if existing:
        raise PPOVideoArtifactError(f"refusing to overwrite final artifact: {existing[0]}")

    fsm = _load_source_episode(fsm_source_dir, role="fsm", ffmpeg=executable)
    ppo = _load_source_episode(ppo_source_dir, role="ppo", ffmpeg=executable)
    pair_evidence = _validate_pair(fsm, ppo)

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ppo-video-publication-", dir=str(output)))
    try:
        stage_videos = staging / "videos"
        stage_manifests = staging / "manifests"
        stage_videos.mkdir(parents=True)
        stage_manifests.mkdir(parents=True)
        stage_paths = {
            "fsm_baseline": stage_videos / FSM_VIDEO_NAME,
            "ppo_improved": stage_videos / PPO_VIDEO_NAME,
            "comparison": stage_videos / COMPARISON_VIDEO_NAME,
            "ppo_diagnostic": stage_videos / DIAGNOSTIC_VIDEO_NAME,
        }
        stage_ass = stage_manifests / DIAGNOSTIC_ASS_NAME

        _encode_clean_source(
            executable, fsm.video_path, stage_paths["fsm_baseline"], label="FSM clean video"
        )
        _encode_clean_source(
            executable, ppo.video_path, stage_paths["ppo_improved"], label="PPO clean video"
        )
        padding_side, padding_frames = _encode_comparison(
            executable,
            stage_paths["fsm_baseline"],
            stage_paths["ppo_improved"],
            stage_paths["comparison"],
            fsm_frames=fsm.frame_count,
            ppo_frames=ppo.frame_count,
        )
        ass_document = build_diagnostic_ass(
            ppo.trace,
            action_start_video_s=float(ppo.manifest["semantic_action_start_video_s"]),
            video_duration_s=ppo.duration_s,
        )
        stage_ass.write_bytes(ass_document.encode("utf-8"))
        _encode_diagnostic(
            executable,
            stage_paths["ppo_improved"],
            stage_ass,
            stage_paths["ppo_diagnostic"],
        )

        validations = {
            "fsm_baseline": _require_generated_video(
                stage_paths["fsm_baseline"],
                ffmpeg=executable,
                expected_width=VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=fsm.frame_count,
            ),
            "ppo_improved": _require_generated_video(
                stage_paths["ppo_improved"],
                ffmpeg=executable,
                expected_width=VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=ppo.frame_count,
            ),
            "comparison": _require_generated_video(
                stage_paths["comparison"],
                ffmpeg=executable,
                expected_width=2 * VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=max(fsm.frame_count, ppo.frame_count),
            ),
            "ppo_diagnostic": _require_generated_video(
                stage_paths["ppo_diagnostic"],
                ffmpeg=executable,
                expected_width=VIDEO_WIDTH,
                expected_height=VIDEO_HEIGHT,
                expected_frame_count=ppo.frame_count,
            ),
        }
        for name, result in validations.items():
            if float(result["duration_s"]) > 200.0:
                raise PPOVideoArtifactError(f"{name} exceeds 200 seconds")

        fsm_checkpoint_label = "frozen_successful_fsm_zero_residual"
        checkpoint_path = str(ppo.checkpoint_path)
        checkpoint_hash = str(ppo.checkpoint_sha256)
        video_records = {
            "fsm_baseline": _video_record(
                validations["fsm_baseline"],
                final_path=destinations["fsm_baseline"],
                source_episode=fsm.root.name,
                source_checkpoint=fsm_checkpoint_label,
                source_checkpoint_sha256="not_applicable",
                source_seed=fsm.seed,
                processing={
                    "kind": "single_episode_full_source_transcode",
                    "input_count": 1,
                    "cuts": False,
                    "time_scale_transform": None,
                    "timestamp_transform": "PTS-STARTPTS",
                    "source_frame_count": fsm.frame_count,
                    "output_frame_count": int(validations["fsm_baseline"]["frame_count"]),
                },
            ),
            "ppo_improved": _video_record(
                validations["ppo_improved"],
                final_path=destinations["ppo_improved"],
                source_episode=ppo.root.name,
                source_checkpoint=checkpoint_path,
                source_checkpoint_sha256=checkpoint_hash,
                source_seed=ppo.seed,
                processing={
                    "kind": "single_episode_full_source_transcode",
                    "input_count": 1,
                    "deterministic_mean_policy": True,
                    "cuts": False,
                    "time_scale_transform": None,
                    "timestamp_transform": "PTS-STARTPTS",
                    "source_frame_count": ppo.frame_count,
                    "output_frame_count": int(validations["ppo_improved"]["frame_count"]),
                },
            ),
            "comparison": _video_record(
                validations["comparison"],
                final_path=destinations["comparison"],
                source_episode=[fsm.root.name, ppo.root.name],
                source_checkpoint=[fsm_checkpoint_label, checkpoint_path],
                source_checkpoint_sha256=[None, checkpoint_hash],
                source_seed=fsm.seed,
                processing={
                    "kind": "real_time_spatial_side_by_side",
                    "temporal_stitching": False,
                    "left": "fsm_baseline",
                    "right": "ppo_improved",
                    "common_time_origin": "source_frame_zero",
                    "phase_alignment": False,
                    "time_scale_transform": None,
                    "earlier_final_frame_tpad_clone_side": padding_side,
                    "earlier_final_frame_tpad_clone_frames": padding_frames,
                    "other_frame_duplication_or_interpolation": False,
                },
            ),
            "ppo_diagnostic": _video_record(
                validations["ppo_diagnostic"],
                final_path=destinations["ppo_diagnostic"],
                source_episode=ppo.root.name,
                source_checkpoint=checkpoint_path,
                source_checkpoint_sha256=checkpoint_hash,
                source_seed=ppo.seed,
                processing={
                    "kind": "single_episode_full_source_15hz_trace_overlay",
                    "cuts": False,
                    "time_scale_transform": None,
                    "trace_path": str(ppo.trace_path),
                    "trace_sha256": sha256_file(ppo.trace_path),
                    "trace_rate_hz": 15.0,
                    "trace_sample_count": len(ppo.trace),
                    "ass_event_count": sum(
                        line.startswith("Dialogue:") for line in ass_document.splitlines()
                    ),
                    "ass_sha256": hashlib.sha256(ass_document.encode("utf-8")).hexdigest(),
                    "overlay_fields": [
                        "phase",
                        "pitch",
                        "pitch_rate",
                        "roll",
                        "roll_rate",
                        "residual_rms",
                        "task_result",
                    ],
                },
            ),
        }
        validation_payload = {
            "schema": _PUBLICATION_SCHEMA,
            "valid": True,
            "status": "PASS",
            "immutable_no_overwrite": True,
            "fps": VIDEO_FPS,
            "maximum_duration_s": 200.0,
            "pair_evidence": dict(pair_evidence),
            "source_episodes": {
                "fsm": {
                    "directory": str(fsm.root),
                    "source_manifest": str(fsm.manifest_path),
                    "source_manifest_sha256": sha256_file(fsm.manifest_path),
                    "viewport_ledger": str(fsm.ledger_path),
                    "viewport_ledger_sha256": sha256_file(fsm.ledger_path),
                    "policy_trace": str(fsm.trace_path),
                    "policy_trace_sha256": sha256_file(fsm.trace_path),
                    "raw_video": str(fsm.video_path),
                    "raw_video_sha256": sha256_file(fsm.video_path),
                },
                "ppo": {
                    "directory": str(ppo.root),
                    "source_manifest": str(ppo.manifest_path),
                    "source_manifest_sha256": sha256_file(ppo.manifest_path),
                    "viewport_ledger": str(ppo.ledger_path),
                    "viewport_ledger_sha256": sha256_file(ppo.ledger_path),
                    "policy_trace": str(ppo.trace_path),
                    "policy_trace_sha256": sha256_file(ppo.trace_path),
                    "raw_video": str(ppo.video_path),
                    "raw_video_sha256": sha256_file(ppo.video_path),
                    "checkpoint": checkpoint_path,
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_manifest": ppo.manifest[
                        "source_checkpoint_manifest"
                    ],
                    "checkpoint_manifest_sha256": ppo.manifest[
                        "source_checkpoint_manifest_sha256"
                    ],
                    "checkpoint_load_provenance": dict(
                        ppo.manifest["checkpoint_load_provenance"]
                    ),
                    "deterministic_mean_policy": True,
                },
            },
            "videos": video_records,
            "diagnostic_ass": {
                "path": str(ass_path),
                "sha256": hashlib.sha256(ass_document.encode("utf-8")).hexdigest(),
                "source": "actual policy_trace.jsonl sampled at 15 Hz",
            },
            "video_checksum_manifest": str(checksum_path),
        }
        stage_validation = stage_manifests / VIDEO_VALIDATION_NAME
        atomic_write_json(stage_validation, validation_payload)
        stage_checksum = stage_manifests / VIDEO_CHECKSUM_NAME
        write_checksum_manifest(
            [*stage_paths.values(), stage_validation, stage_ass],
            stage_checksum,
            root=staging,
        )

        staged_to_final = [
            *( (stage_paths[name], destinations[name]) for name in destinations ),
            (stage_ass, ass_path),
            (stage_validation, validation_path),
            (stage_checksum, checksum_path),
        ]
        for staged, final in staged_to_final:
            _publish_no_replace(staged, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    checksum_verification = verify_checksum_manifest(checksum_path, root=output)
    if checksum_verification.get("valid") is not True:
        raise PPOVideoArtifactError("published video checksum verification failed")
    return FinalVideoPublication(
        videos=destinations,
        validation_path=validation_path,
        checksum_path=checksum_path,
        diagnostic_ass_path=ass_path,
        checksum_verification=checksum_verification,
    )


__all__ = [
    "COMPARISON_VIDEO_NAME",
    "DIAGNOSTIC_VIDEO_NAME",
    "FSM_VIDEO_NAME",
    "FinalVideoPublication",
    "PPOVideoArtifactError",
    "PPO_VIDEO_NAME",
    "VIDEO_CHECKSUM_NAME",
    "VIDEO_VALIDATION_NAME",
    "build_diagnostic_ass",
    "publish_final_videos",
]
