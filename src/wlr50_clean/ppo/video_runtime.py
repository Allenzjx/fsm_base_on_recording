"""Truthful active-viewport capture for FSM and deterministic PPO episodes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import atomic_write_json
from .live_stream_writer import LiveStreamWriter


PHYSICS_HZ = 120.0
VIDEO_FPS = 15.0
RENDER_STRIDE = 8
PRE_ACTION_TICKS = 64  # 0.533 s, inside the required 0.5--1.0 s window.
POST_SUCCESS_TICKS = 184  # 1.533 s, inside the required 1.0--2.0 s window.
PROCESS_INSTANCE_ID = uuid.uuid4().hex
SOURCE_IDENTITY_FIELDS = (
    "environment_hash",
    "robot_asset_hash",
    "initial_root_state",
    "initial_joint_state",
    "obstacle_pose",
    "controller_hash",
    "motion_contract_hash",
)


class PPOVideoRuntimeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl_no_clobber(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Atomically publish JSONL without ever replacing existing evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PPOVideoRuntimeError(f"refusing to overwrite video evidence: {destination}")
    try:
        payload = b"".join(
            (
                json.dumps(dict(row), separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
            for row in rows
        )
    except (TypeError, ValueError) as exc:
        raise PPOVideoRuntimeError("policy trace is not finite JSON") from exc
    if not payload:
        raise PPOVideoRuntimeError("policy trace is empty")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    except FileExistsError as exc:
        raise PPOVideoRuntimeError(
            f"refusing to overwrite video evidence: {destination}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _capture_frame(recorder: Any, backend: Any, *, step: int) -> None:
    recorder.before_render(sim_step=step, sim_time_s=step / PHYSICS_HZ)
    backend.render_video_frame()
    recorder.after_render()
    recorder.require_healthy()


def capture_live_policy_video(
    episode: Any,
    *,
    seed: int,
    output_directory: Path | str,
    action_factory: Callable[[Sequence[float], int], Sequence[float]],
    policy_label: str,
    checkpoint_path: Path | str | None = None,
    checkpoint_manifest_path: Path | str | None = None,
    checkpoint_load_provenance: Mapping[str, Any] | None = None,
    pre_action_ticks: int = PRE_ACTION_TICKS,
    post_success_ticks: int = POST_SUCCESS_TICKS,
    render_stride: int = RENDER_STRIDE,
    warmup_renders: int = 3,
    episode_already_reset: bool = False,
    recorder_factory: Callable[[Path], Any] | None = None,
    stream_writer_factory: Callable[..., Any] = LiveStreamWriter,
) -> Mapping[str, Any]:
    """Capture one complete P01-start episode with real physical pre/post roll.

    The caller supplies either the zero residual FSM action or deterministic
    checkpoint inference.  No policy sampling, video time scaling, temporal
    splice, or synthetic duplicate frame is performed.
    """

    if int(seed) < 0:
        raise PPOVideoRuntimeError("video seed must be non-negative")
    pre_ticks = int(pre_action_ticks)
    post_ticks = int(post_success_ticks)
    stride = int(render_stride)
    if stride <= 0 or pre_ticks <= 0 or post_ticks <= 0:
        raise PPOVideoRuntimeError("video tick counts and render stride must be positive")
    if pre_ticks % stride or post_ticks % stride:
        raise PPOVideoRuntimeError("pre/post video tick counts must align to render cadence")
    pre_seconds = pre_ticks / PHYSICS_HZ
    post_seconds = post_ticks / PHYSICS_HZ
    if not 0.5 <= pre_seconds <= 1.0:
        raise PPOVideoRuntimeError("pre-action footage must span 0.5--1.0 seconds")
    if not 1.0 <= post_seconds <= 2.0:
        raise PPOVideoRuntimeError("post-success footage must span 1.0--2.0 seconds")

    root = Path(output_directory).resolve()
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).resolve()
    checkpoint_manifest = (
        None
        if checkpoint_manifest_path is None
        else Path(checkpoint_manifest_path).resolve()
    )
    if policy_label not in {"fsm_zero_residual", "ppo_deterministic_mean"}:
        raise PPOVideoRuntimeError("video policy label is not recognized")
    if checkpoint is not None and not checkpoint.is_file():
        raise PPOVideoRuntimeError(f"checkpoint is missing: {checkpoint}")
    if policy_label == "fsm_zero_residual":
        if checkpoint is not None or checkpoint_manifest is not None:
            raise PPOVideoRuntimeError("FSM zero-residual video must not name a checkpoint")
        if checkpoint_load_provenance is not None:
            raise PPOVideoRuntimeError("FSM video must not name checkpoint load provenance")
    else:
        if checkpoint is None or checkpoint_manifest is None:
            raise PPOVideoRuntimeError(
                "PPO video requires an explicit checkpoint and checkpoint manifest"
            )
        if not checkpoint_manifest.is_file():
            raise PPOVideoRuntimeError(
                f"checkpoint manifest is missing: {checkpoint_manifest}"
            )
        if not isinstance(checkpoint_load_provenance, Mapping):
            raise PPOVideoRuntimeError("PPO video lacks strict checkpoint load provenance")
    checkpoint_sha256 = None if checkpoint is None else _sha256(checkpoint)
    checkpoint_manifest_sha256 = (
        None if checkpoint_manifest is None else _sha256(checkpoint_manifest)
    )
    if checkpoint is not None:
        assert checkpoint_manifest is not None
        assert isinstance(checkpoint_load_provenance, Mapping)
        if (
            Path(str(checkpoint_load_provenance.get("checkpoint_path", ""))).resolve()
            != checkpoint
            or str(checkpoint_load_provenance.get("checkpoint_sha256", "")).lower()
            != checkpoint_sha256
            or Path(str(checkpoint_load_provenance.get("manifest_path", ""))).resolve()
            != checkpoint_manifest
            or str(checkpoint_load_provenance.get("manifest_sha256", "")).lower()
            != checkpoint_manifest_sha256
            or checkpoint_load_provenance.get("checkpoint_infos_match_manifest") is not True
        ):
            raise PPOVideoRuntimeError(
                "PPO strict checkpoint load provenance does not match its artifacts"
            )
    if recorder_factory is None:
        from wlr50_clean.infrastructure.video_capture import ActiveViewportVideoRecorder

        recorder_factory = ActiveViewportVideoRecorder

    if not hasattr(episode, "collect_trace") or not hasattr(episode, "trace"):
        raise PPOVideoRuntimeError("video episode does not support a policy trace")
    if tuple(episode.trace):
        raise PPOVideoRuntimeError("video episode trace is not empty before capture")
    episode.collect_trace = True
    if episode_already_reset:
        if (
            int(getattr(episode, "seed", -1)) != int(seed)
            or bool(getattr(episode, "done", True))
            or int(getattr(episode, "decision_count", -1)) != 0
            or getattr(episode, "frame", None) is None
            or getattr(episode, "observation", None) is None
            or int(getattr(episode.frame, "physics_tick", -1)) != 0
        ):
            raise PPOVideoRuntimeError(
                "pre-reset video episode is not a fresh zero-decision episode"
            )
        observation = episode.observation
        reset_info = dict(episode.frame.info)
    else:
        observation, reset_info = episode.reset(seed=int(seed), options={})
    if int(reset_info.get("seed", -1)) != int(seed):
        raise PPOVideoRuntimeError("video reset evidence has the wrong seed")
    if episode.frame is None or episode.frame.state_id != "P01":
        raise PPOVideoRuntimeError("final video evaluation must start at P01")
    missing_identity = tuple(
        field for field in SOURCE_IDENTITY_FIELDS if field not in reset_info
    )
    if missing_identity:
        raise PPOVideoRuntimeError(
            f"video reset lacks source identity evidence: {missing_identity}"
        )
    source_identity = {
        field: reset_info[field] for field in SOURCE_IDENTITY_FIELDS
    }
    reset_options = reset_info.get("reset_options")
    try:
        reset_count = int(reset_info.get("reset_count", -1))
        global_resets = int(reset_info.get("reset_global_simulation_resets", -1))
        forward_syncs = int(reset_info.get("reset_simulation_forward_syncs", -1))
    except (TypeError, ValueError) as exc:
        raise PPOVideoRuntimeError("video reset freshness evidence is invalid") from exc
    if (
        reset_count != 1
        or reset_options != {}
        or reset_info.get("training_phase_snapshot") is not None
        or global_resets != 0
        or forward_syncs != 0
    ):
        raise PPOVideoRuntimeError(
            "video source is not the first full P01 reset in its fresh process"
        )
    reset_evidence = {
        "reset_count": reset_count,
        "reset_options": {},
        "training_phase_snapshot": None,
        "reset_global_simulation_resets": global_resets,
        "reset_simulation_forward_syncs": forward_syncs,
    }
    locked_scene = reset_info.get("locked_scene_snapshot")
    camera = locked_scene.get("camera") if isinstance(locked_scene, Mapping) else None
    if not isinstance(camera, Mapping):
        raise PPOVideoRuntimeError("video reset lacks locked live camera evidence")
    try:
        camera_eye = tuple(float(value) for value in camera.get("eye_m", ()))
        camera_target = tuple(float(value) for value in camera.get("target_m", ()))
    except (TypeError, ValueError) as exc:
        raise PPOVideoRuntimeError("video reset camera evidence is invalid") from exc
    if (
        len(camera_eye) != 3
        or len(camera_target) != 3
        or any(not math.isfinite(value) for value in (*camera_eye, *camera_target))
    ):
        raise PPOVideoRuntimeError("video reset camera evidence is invalid")
    backend = episode.backend
    for _ in range(int(warmup_renders)):
        backend.render_video_frame()

    writer = stream_writer_factory(root, seed=int(seed))
    recorder = recorder_factory(root)
    reward_total = 0.0
    terminal_info: Mapping[str, Any] | None = None
    recorder_manifest: Mapping[str, Any] | None = None
    pre_action_observation_refreshed = False
    try:
        if not recorder.start():
            raise PPOVideoRuntimeError(
                recorder.error or "viewport recorder failed to start"
            )
        # Frame zero is an actual active-viewport render of the settled scene.
        _capture_frame(recorder, backend, step=0)
        for physical_index in range(1, pre_ticks + 1):
            hold = backend.advance_video_pre_action_tick()
            if int(hold.get("root_state_write_count", -1)) != 0:
                raise PPOVideoRuntimeError("pre-action hold performed a root-state write")
            if any(
                bool(hold.get(field))
                for field in (
                    "body_collision",
                    "wheel_only_climb",
                    "fall",
                    "nan_inf",
                    "hard_joint_limit",
                    "physics_explosion",
                )
            ):
                raise PPOVideoRuntimeError(
                    "pre-action physical hold developed a hazard"
                )
            if physical_index % stride == 0:
                _capture_frame(recorder, backend, step=physical_index)

        refresh = getattr(episode, "refresh_after_video_pre_action_hold", None)
        if not callable(refresh):
            raise PPOVideoRuntimeError(
                "video episode lacks the required post-pre-roll sensor refresh"
            )
        observation, refreshed_info = refresh()
        if (
            episode.frame is None
            or episode.frame.state_id != "P01"
            or int(episode.frame.physics_tick) != 0
            or float(episode.frame.sim_time_s) != 0.0
            or bool(episode.done)
            or int(episode.decision_count) != 0
        ):
            raise PPOVideoRuntimeError(
                "post-pre-roll sensor refresh changed the logical P01 start"
            )
        for field in SOURCE_IDENTITY_FIELDS:
            if refreshed_info.get(field) != reset_info[field]:
                raise PPOVideoRuntimeError(
                    f"post-pre-roll sensor refresh changed source identity field {field}"
                )
        for field, expected in reset_evidence.items():
            if refreshed_info.get(field) != expected:
                raise PPOVideoRuntimeError(
                    f"post-pre-roll sensor refresh changed reset evidence field {field}"
                )
        refresh_evidence = refreshed_info.get("video_pre_action_refresh")
        if (
            not isinstance(refresh_evidence, Mapping)
            or refresh_evidence.get("schema")
            != "wlr50_clean.ppo_video_pre_action_refresh.v1"
            or int(refresh_evidence.get("physical_pre_action_ticks", -1))
            != pre_ticks
            or int(refresh_evidence.get("pre_roll_reader_last_logical_tick", -1))
            != pre_ticks
            or refresh_evidence.get("episode_reader_reinitialized") is not True
            or int(refresh_evidence.get("episode_reader_first_logical_tick", -1)) != 0
            or refresh_evidence.get("controller_frame_preserved") is not True
            or int(refresh_evidence.get("controller_logical_tick", -1)) != 0
            or refresh_evidence.get("simulation_reset_performed") is not False
            or refresh_evidence.get("fsm_step_performed") is not False
        ):
            raise PPOVideoRuntimeError(
                "post-pre-roll sensor refresh lacks exact clock/reset evidence"
            )
        refresh_evidence = dict(refresh_evidence)
        pre_action_observation_refreshed = True
        # Canonical streams begin from the live sensing snapshot taken after
        # physical pre-roll.  No stale reset-time actor observation is sealed.
        writer.start(episode.frame)
        episode.tick_callback = writer.write_tick

        while not episode.done:
            action = tuple(
                float(value)
                for value in action_factory(observation, episode.decision_count)
            )
            if len(action) != 12 or any(not math.isfinite(value) for value in action):
                raise PPOVideoRuntimeError("video policy returned a non-finite non-Full12 action")
            step_result = episode.step(action)
            observation = step_result.observation
            reward_total += float(step_result.reward)
            writer.write_decision(step_result.info)
            terminal_info = step_result.info
            assert episode.frame is not None
            global_tick = pre_ticks + int(episode.frame.physics_tick)
            if global_tick % stride == 0:
                _capture_frame(recorder, backend, step=global_tick)

        assert episode.frame is not None
        signals = episode.frame.termination_signals
        termination_reason = None if terminal_info is None else terminal_info.get(
            "termination_reason"
        )
        if (
            termination_reason != "SUCCESS"
            or not signals.success
            or signals.body_collision
            or signals.wheel_only_climb
            or signals.fall
            or signals.nan_inf
            or signals.hard_joint_limit
            or signals.physics_explosion
            or episode.frame.sim_time_s > 200.0
            or pre_seconds + episode.frame.sim_time_s + post_seconds > 200.0
        ):
            raise PPOVideoRuntimeError(
                f"source episode failed physical video acceptance: {termination_reason}"
            )
        trial_manifest = writer.finalize(
            episode.frame,
            reward_total=reward_total,
            decision_count=episode.decision_count,
        )

        for post_index in range(1, post_ticks + 1):
            hold = backend.advance_video_post_success_tick()
            if int(hold.get("root_state_write_count", -1)) != 0:
                raise PPOVideoRuntimeError("post-success hold performed a root-state write")
            if any(
                bool(hold.get(field))
                for field in (
                    "body_collision",
                    "wheel_only_climb",
                    "fall",
                    "nan_inf",
                    "hard_joint_limit",
                    "physics_explosion",
                )
            ):
                raise PPOVideoRuntimeError("post-success physical hold developed a hazard")
            global_tick = pre_ticks + int(episode.frame.physics_tick) + post_index
            if global_tick % stride == 0:
                _capture_frame(
                    recorder,
                    backend,
                    step=global_tick,
                )
        recorder_manifest = recorder.finalize()
        if recorder_manifest.get("valid") is not True:
            raise PPOVideoRuntimeError("active viewport recorder did not validate")
        if (
            int(recorder_manifest.get("width", -1)) != 1280
            or int(recorder_manifest.get("height", -1)) != 720
            or not math.isclose(
                float(recorder_manifest.get("fps", math.nan)),
                VIDEO_FPS,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise PPOVideoRuntimeError(
                "active viewport recorder did not preserve 1280x720 at 15 Hz"
            )
        try:
            encoded_frame_count = int(recorder_manifest["frame_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PPOVideoRuntimeError(
                "active viewport recorder lacks a valid frame count"
            ) from exc
        if encoded_frame_count < 2 or encoded_frame_count / VIDEO_FPS > 200.0:
            raise PPOVideoRuntimeError(
                "captured video timeline exceeds the 200 second final-video limit"
            )
    except Exception as exc:
        writer.abort()
        try:
            recorder.finalize()
        except Exception:
            pass
        partial_trace = tuple(getattr(episode, "trace", ()))
        partial_path: Path | None = None
        if partial_trace:
            try:
                partial_path = _write_jsonl_no_clobber(
                    root / "policy_trace.failed.jsonl", partial_trace
                )
            except Exception:
                partial_path = None
        try:
            atomic_write_json(
                root / "ppo_video_capture_failure.json",
                {
                    "schema": "wlr50_clean.ppo_video_capture_failure.v1",
                    "policy_label": str(policy_label),
                    "seed": int(seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_policy_trace": (
                        None if partial_path is None else str(partial_path)
                    ),
                    "partial_policy_trace_sha256": (
                        None if partial_path is None else _sha256(partial_path)
                    ),
                    "partial_policy_trace_rows": len(partial_trace),
                },
            )
        except Exception:
            pass
        raise

    assert episode.frame is not None and recorder_manifest is not None
    trace = tuple(episode.trace)
    try:
        if len(trace) != int(episode.decision_count):
            raise PPOVideoRuntimeError(
                "complete video policy trace count differs from the decision count"
            )
        if checkpoint is not None and _sha256(checkpoint) != checkpoint_sha256:
            raise PPOVideoRuntimeError("PPO checkpoint changed during video capture")
        if (
            checkpoint_manifest is not None
            and _sha256(checkpoint_manifest) != checkpoint_manifest_sha256
        ):
            raise PPOVideoRuntimeError(
                "PPO checkpoint manifest changed during video capture"
            )
        trace_path = _write_jsonl_no_clobber(root / "policy_trace.jsonl", trace)
    except Exception as exc:
        failed_trace: Path | None = None
        if trace:
            try:
                failed_trace = _write_jsonl_no_clobber(
                    root / "policy_trace.failed.jsonl", trace
                )
            except Exception:
                failed_trace = None
        try:
            atomic_write_json(
                root / "ppo_video_capture_failure.json",
                {
                    "schema": "wlr50_clean.ppo_video_capture_failure.v1",
                    "policy_label": str(policy_label),
                    "seed": int(seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_policy_trace": (
                        None if failed_trace is None else str(failed_trace)
                    ),
                    "partial_policy_trace_sha256": (
                        None if failed_trace is None else _sha256(failed_trace)
                    ),
                    "partial_policy_trace_rows": len(trace),
                },
            )
        except Exception:
            pass
        raise
    video_path = Path(str(recorder_manifest["video_path"])).resolve()
    result = {
        "schema": "wlr50_clean.ppo_video_source_episode.v1",
        "policy_label": str(policy_label),
        "deterministic_mean_policy": policy_label != "fsm_zero_residual",
        "fresh_process_single_episode": True,
        "episode_count": 1,
        "capture_process_id": os.getpid(),
        "capture_process_instance_id": PROCESS_INSTANCE_ID,
        "used_preinitialized_fresh_episode": bool(episode_already_reset),
        "pre_action_observation_refreshed": pre_action_observation_refreshed,
        "pre_action_refresh_evidence": refresh_evidence,
        "seed": int(seed),
        "source_checkpoint": None if checkpoint is None else str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_manifest": (
            None if checkpoint_manifest is None else str(checkpoint_manifest)
        ),
        "source_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_load_provenance": (
            None
            if checkpoint_load_provenance is None
            else dict(checkpoint_load_provenance)
        ),
        "task_success": True,
        "completed_p01_p13": True,
        "body_collision": False,
        "wheel_only_climb": False,
        "duration_s": float(episode.frame.sim_time_s),
        "decision_count": int(episode.decision_count),
        "reward_total": reward_total,
        "pre_action_physical_hold_s": pre_seconds,
        "post_success_physical_hold_s": post_seconds,
        "semantic_action_start_video_s": pre_seconds,
        "semantic_task_success_video_s": pre_seconds + float(episode.frame.sim_time_s),
        "video_duration_expected_s": (
            pre_seconds + float(episode.frame.sim_time_s) + post_seconds
        ),
        "stitched": False,
        "speed_modified": False,
        "frame_interpolation": False,
        "source_episode_directory": str(root),
        "source_identity": source_identity,
        "reset_evidence": reset_evidence,
        "camera": {
            "eye_m": list(camera_eye),
            "target_m": list(camera_target),
            "resolution": [1280, 720],
            "fps": VIDEO_FPS,
            "locked_scene_snapshot": True,
            "active_viewport_resolution_verified": bool(
                int(recorder_manifest.get("width", -1)) == 1280
                and int(recorder_manifest.get("height", -1)) == 720
                and math.isclose(
                    float(recorder_manifest.get("fps", math.nan)),
                    VIDEO_FPS,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            "render_product_path": recorder_manifest.get("render_product_path"),
            "viewport_identity": recorder_manifest.get("viewport_identity"),
        },
        "policy_trace": str(trace_path),
        "policy_trace_sha256": _sha256(trace_path),
        "policy_trace_rate_hz": 15.0,
        "policy_trace_rows": len(trace),
        "trial_manifest": str(trial_manifest),
        "reset_info_recording_access_count": int(reset_info.get("recording_accesses", 0)),
        "recorder_manifest": str(recorder.manifest_path),
        "raw_video": str(video_path),
        "raw_video_sha256": _sha256(video_path),
        "recorder_validation": dict(recorder_manifest),
    }
    atomic_write_json(root / "ppo_video_source_manifest.json", result)
    return result


__all__ = [
    "PHYSICS_HZ",
    "POST_SUCCESS_TICKS",
    "PROCESS_INSTANCE_ID",
    "PPOVideoRuntimeError",
    "PRE_ACTION_TICKS",
    "RENDER_STRIDE",
    "SOURCE_IDENTITY_FIELDS",
    "VIDEO_FPS",
    "capture_live_policy_video",
]
