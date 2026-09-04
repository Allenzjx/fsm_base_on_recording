from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo.video_runtime import (
    PPOVideoRuntimeError,
    capture_live_policy_video,
)


class _Writer:
    def __init__(self, root: Path, *, seed: int):
        root.mkdir()
        self.root = root
        self.seed = seed
        self.decisions = 0
        self.aborted = False

    def start(self, frame):
        self.frame = frame

    def write_tick(self, *_):
        pass

    def write_decision(self, info):
        self.decisions += 1

    def finalize(self, frame, *, reward_total, decision_count):
        path = self.root / "trial_manifest.json"
        path.write_text("{}", encoding="utf-8")
        return path

    def abort(self):
        self.aborted = True


class _Recorder:
    def __init__(self, root: Path):
        self.root = root
        self.rows = []
        self.error = None
        self.manifest_path = root / "viewport_manifest.json"

    def start(self):
        return True

    def before_render(self, *, sim_step, sim_time_s):
        self.rows.append((sim_step, sim_time_s))

    def after_render(self):
        pass

    def require_healthy(self):
        pass

    def finalize(self):
        video = self.root / "actual_viewport_video.mp4"
        video.write_bytes(b"real-test-placeholder")
        self.manifest_path.write_text("{}", encoding="utf-8")
        return {
            "valid": True,
            "video_path": str(video),
            "frame_count": len(self.rows),
            "width": 1280,
            "height": 720,
            "fps": 15.0,
            "render_product_path": "/Render/Product",
            "viewport_identity": 42,
        }


class _Backend:
    def __init__(self):
        self.pre = 0
        self.post = 0
        self.renders = 0

    def render_video_frame(self):
        self.renders += 1

    def advance_video_pre_action_tick(self):
        self.pre += 1
        return {"root_state_write_count": 0}

    def advance_video_post_success_tick(self):
        self.post += 1
        return {
            "root_state_write_count": 0,
            "body_collision": False,
            "wheel_only_climb": False,
        }


class _Episode:
    def __init__(self, *, success: bool = True):
        self.backend = _Backend()
        self.success = success
        self.tick_callback = None
        self.collect_trace = False
        self.trace = []
        self.reset_calls = 0

    def reset(self, *, seed, options):
        self.reset_calls += 1
        self.seed = seed
        self.done = False
        self.decision_count = 0
        self.trace = []
        reset_info = {
            "seed": seed,
            "recording_accesses": 0,
            "environment_hash": "e" * 64,
            "robot_asset_hash": "a" * 64,
            "initial_root_state": [0.0] * 13,
            "initial_joint_state": [0.0] * 24,
            "obstacle_pose": [0.65, 0.0, 0.025],
            "controller_hash": "c" * 64,
            "motion_contract_hash": "m" * 64,
            "reset_count": 1,
            "reset_options": {},
            "training_phase_snapshot": None,
            "reset_global_simulation_resets": 0,
            "reset_simulation_forward_syncs": 0,
            "locked_scene_snapshot": {
                "camera": {
                    "eye_m": [1.45, -1.25, 0.80],
                    "target_m": [0.45, 0.0, 0.12],
                }
            },
        }
        self.frame = SimpleNamespace(
            state_id="P01",
            physics_tick=0,
            sim_time_s=0.0,
            info=reset_info,
            termination_signals=SimpleNamespace(
                success=False,
                body_collision=False,
                wheel_only_climb=False,
                fall=False,
                nan_inf=False,
                hard_joint_limit=False,
                physics_explosion=False,
            ),
        )
        self.observation = (0.0,) * 125
        self.refresh_count = 0
        return self.observation, reset_info

    def refresh_after_video_pre_action_hold(self):
        self.refresh_count += 1
        self.observation = (64.0,) * 125
        self.frame.info = {
            **self.frame.info,
            "video_pre_action_refresh": {
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
        }
        return self.observation, dict(self.frame.info)

    def step(self, action):
        self.decision_count += 1
        self.done = self.decision_count == 2
        self.frame.physics_tick = self.decision_count * 8
        self.frame.sim_time_s = self.frame.physics_tick / 120.0
        if self.done:
            self.frame.termination_signals.success = self.success
        info = {"termination_reason": "SUCCESS" if self.success and self.done else None}
        if self.collect_trace:
            self.trace.append(
                {
                    "seed": self.seed if hasattr(self, "seed") else 0,
                    "decision_index": self.decision_count - 1,
                    "physics_tick": self.frame.physics_tick,
                    "sim_time_s": self.frame.sim_time_s,
                    "state_id": "P13" if self.done else "P01",
                    "pitch_error_rad": 0.0,
                    "roll_error_rad": 0.0,
                    "pitch_rate_rad_s": 0.0,
                    "roll_rate_rad_s": 0.0,
                    "residual_full12": [0.0] * 12,
                    **info,
                }
            )
        return SimpleNamespace(
            observation=(float(self.decision_count),) * 125,
            reward=1.0,
            info=info,
        )


def test_video_runtime_uses_real_pre_and_post_physics_holds(tmp_path: Path) -> None:
    episode = _Episode()
    policy_observations = []
    result = capture_live_policy_video(
        episode,
        seed=4001,
        output_directory=tmp_path / "capture",
        action_factory=lambda observation, decision: (
            policy_observations.append(tuple(observation)) or (0.0,) * 12
        ),
        policy_label="fsm_zero_residual",
        pre_action_ticks=64,
        post_success_ticks=184,
        warmup_renders=2,
        recorder_factory=_Recorder,
        stream_writer_factory=_Writer,
    )
    assert result["task_success"] is True
    assert result["pre_action_physical_hold_s"] == pytest.approx(64 / 120)
    assert result["post_success_physical_hold_s"] == pytest.approx(184 / 120)
    assert result["stitched"] is False
    assert result["speed_modified"] is False
    assert episode.backend.pre == 64
    assert episode.backend.post == 184
    assert episode.refresh_count == 1
    assert policy_observations[0] == (64.0,) * 125
    assert result["pre_action_observation_refreshed"] is True
    assert result["pre_action_refresh_evidence"]["episode_reader_reinitialized"] is True
    # Initial frame + 8 pre-roll frames + 2 policy frames + 23 post frames.
    assert result["recorder_validation"]["frame_count"] == 34
    assert episode.backend.renders == 36  # includes two uncaptured warmups
    source_manifest = json.loads(
        (tmp_path / "capture" / "ppo_video_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_manifest["source_identity"]["environment_hash"] == "e" * 64
    assert source_manifest["source_identity"]["obstacle_pose"] == [0.65, 0.0, 0.025]
    trace_path = tmp_path / "capture" / "policy_trace.jsonl"
    assert trace_path.is_file()
    assert source_manifest["policy_trace"] == str(trace_path.resolve())
    assert source_manifest["policy_trace_sha256"] == hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    assert source_manifest["policy_trace_rows"] == 2


def test_ppo_video_uses_the_runners_existing_fresh_reset(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_improved.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_manifest = tmp_path / "checkpoint_improved_manifest.json"
    checkpoint_manifest.write_text("{}", encoding="utf-8")
    provenance = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest_path": str(checkpoint_manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            checkpoint_manifest.read_bytes()
        ).hexdigest(),
        "checkpoint_infos_match_manifest": True,
    }
    episode = _Episode()
    episode.reset(seed=4001, options={})

    result = capture_live_policy_video(
        episode,
        seed=4001,
        output_directory=tmp_path / "ppo_capture",
        action_factory=lambda observation, decision: (0.01,) * 12,
        policy_label="ppo_deterministic_mean",
        checkpoint_path=checkpoint,
        checkpoint_manifest_path=checkpoint_manifest,
        checkpoint_load_provenance=provenance,
        episode_already_reset=True,
        pre_action_ticks=64,
        post_success_ticks=184,
        recorder_factory=_Recorder,
        stream_writer_factory=_Writer,
    )

    assert episode.reset_calls == 1
    assert result["used_preinitialized_fresh_episode"] is True
    assert result["checkpoint_load_provenance"] == provenance


def test_video_runtime_rejects_failed_physics_and_bad_timeline(tmp_path: Path) -> None:
    with pytest.raises(PPOVideoRuntimeError, match="pre-action"):
        capture_live_policy_video(
            _Episode(),
            seed=1,
            output_directory=tmp_path / "bad_window",
            action_factory=lambda observation, decision: (0.0,) * 12,
            policy_label="fsm_zero_residual",
            pre_action_ticks=8,
            post_success_ticks=184,
            recorder_factory=_Recorder,
            stream_writer_factory=_Writer,
        )
    with pytest.raises(PPOVideoRuntimeError, match="physical video acceptance"):
        capture_live_policy_video(
            _Episode(success=False),
            seed=1,
            output_directory=tmp_path / "failed",
            action_factory=lambda observation, decision: (0.0,) * 12,
            policy_label="fsm_zero_residual",
            pre_action_ticks=64,
            post_success_ticks=184,
            recorder_factory=_Recorder,
            stream_writer_factory=_Writer,
        )
    assert (tmp_path / "failed" / "policy_trace.failed.jsonl").is_file()
    failure = json.loads(
        (tmp_path / "failed" / "ppo_video_capture_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["partial_policy_trace_rows"] == 2
    assert failure["error_type"] == "PPOVideoRuntimeError"
