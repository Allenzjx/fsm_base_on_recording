from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import checkpoint_promotion, cli, rl_library_wrapper, video_artifacts
from wlr50_clean.ppo.phase_effective_entry import (
    capture_validated_effective_phase_entry_contract,
)
from wlr50_clean.ppo.phase_snapshots import (
    capture_validated_phase_snapshot_bundle,
    phase_snapshot_bundle_file_hashes,
)


def _capture_args(tmp_path: Path, **overrides):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    values = {
        "video_source_role": "fsm",
        "training_config": tmp_path / "training.yaml",
        "seed": 4001,
        "num_envs": 1,
        "episode_count": 1,
        "deterministic": True,
        "headless": False,
        "capture_fps": 15.0,
        "maximum_duration_s": 200.0,
        "run_dir": run_dir,
        "checkpoint": None,
        "checkpoint_manifest": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_video_commands_split_live_capture_from_offline_publication() -> None:
    assert "capture-video-source" in cli.LIVE_COMMANDS
    assert "publish-videos" not in cli.LIVE_COMMANDS
    assert "build-videos" not in cli.LIVE_COMMANDS


def test_capture_request_rejects_headless_and_accepts_only_locked_video_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_training_profile",
        lambda path: SimpleNamespace(video_seed=4001),
    )
    with pytest.raises(cli.CliError, match="--no-headless"):
        cli._require_video_capture_request(_capture_args(tmp_path, headless=True))
    with pytest.raises(cli.CliError, match="seed 4001"):
        cli._require_video_capture_request(_capture_args(tmp_path, seed=4002))

    args = _capture_args(tmp_path)
    cli._require_video_capture_request(args)
    assert args._video_source_root == (args.run_dir / "video_source").resolve()
    assert args._video_checkpoint_provenance is None


def test_ppo_capture_request_binds_the_promoted_checkpoint_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint_improved.pt"
    manifest = tmp_path / "checkpoint_improved_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_training_profile",
        lambda path: SimpleNamespace(video_seed=4001),
    )
    snapshot_pin = capture_validated_phase_snapshot_bundle(
        cli.DEFAULT_PHASE_SNAPSHOT_ROOT,
        canonical_root=cli.DEFAULT_PHASE_SNAPSHOT_ROOT,
    )
    snapshot_bundle = snapshot_pin.as_record()
    effective_pin = capture_validated_effective_phase_entry_contract(
        expected_snapshot_bundle=snapshot_pin,
    )
    provenance = SimpleNamespace(
        checkpoint_path=checkpoint.resolve(),
        manifest_path=manifest.resolve(),
        manifest={
            "publication_role": "improved",
            "validation_promotion_authorized": True,
            "locked_test_authorized": True,
            "promotion_authorized": True,
            "phase_snapshot_manifest": snapshot_bundle["manifest_path"],
            "phase_snapshot_manifest_sha256": snapshot_bundle["manifest_sha256"],
            "phase_snapshot_bundle_sha256": snapshot_bundle["bundle_sha256"],
            "phase_snapshot_bundle": snapshot_bundle,
            "phase_effective_entry_contract_path": str(effective_pin.contract_path),
            "phase_effective_entry_contract_file_sha256": effective_pin.file_sha256,
            "phase_effective_entry_contract_sidecar_path": str(effective_pin.sidecar_path),
            "phase_effective_entry_contract_sidecar_sha256": (
                effective_pin.sidecar_file_sha256
            ),
            "phase_effective_entry_contract_sha256": effective_pin.contract_sha256,
            "phase_effective_entry_contract": effective_pin.as_record(),
            "files": {
                **phase_snapshot_bundle_file_hashes(snapshot_bundle),
                **effective_pin.file_hashes(),
            },
        },
        as_dict=lambda: {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": "a" * 64,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        checkpoint_promotion,
        "validate_checkpoint_artifact_provenance",
        lambda checkpoint_path, manifest_path: provenance,
    )
    args = _capture_args(
        tmp_path,
        video_source_role="ppo",
        checkpoint=checkpoint,
        checkpoint_manifest=manifest,
    )

    cli._require_video_capture_request(args)

    assert args.checkpoint == checkpoint.resolve()
    assert args.checkpoint_manifest == manifest.resolve()
    assert args._video_checkpoint_provenance == provenance.as_dict()


def test_publish_videos_is_an_offline_forwarder_with_a_machine_readable_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    fsm = tmp_path / "fsm"
    ppo = tmp_path / "ppo"
    output = tmp_path / "output"
    run_dir.mkdir()
    fsm.mkdir()
    ppo.mkdir()
    calls = []
    publication = SimpleNamespace(
        videos={"fsm_baseline": output / "videos" / "fsm_baseline_clean.mp4"},
        validation_path=output / "manifests" / "video_validation.json",
        checksum_path=output / "manifests" / "video_checksums.sha256",
        diagnostic_ass_path=output / "manifests" / "ppo_improved_diagnostic.ass",
        checksum_verification={"valid": True},
    )
    for path in (
        *publication.videos.values(),
        publication.validation_path,
        publication.checksum_path,
        publication.diagnostic_ass_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("utf-8"))

    def fake_publish(**kwargs):
        calls.append(kwargs)
        return publication

    monkeypatch.setattr(video_artifacts, "publish_final_videos", fake_publish)
    args = SimpleNamespace(
        seed=4001,
        num_envs=1,
        episode_count=1,
        deterministic=True,
        fsm_video_source_dir=fsm,
        ppo_video_source_dir=ppo,
        output_root=output,
        ffmpeg=None,
        run_dir=run_dir,
    )

    assert cli._publish_videos(args) == 0
    assert calls == [
        {
            "fsm_source_dir": fsm.resolve(),
            "ppo_source_dir": ppo.resolve(),
            "output_root": output.resolve(),
            "publication_run_dir": run_dir,
            "ffmpeg": None,
        }
    ]
    result = (run_dir / "final_video_publication.json").read_text(encoding="utf-8")
    assert '"offline": true' in result
    assert '"isaac_started": false' in result
    assert '"video_records"' in result
    assert '"video_validation_sha256"' in result


def test_checkpoint_manifest_hashes_every_runtime_v2_contract(
    tmp_path: Path,
) -> None:
    runtime_rows = [
        {
            "path": "pyproject.toml",
            "bytes": 1,
            "sha256": "a" * 64,
            "creation_time_utc_ticks": 1,
            "last_write_time_utc_ticks": 2,
        }
    ]
    runtime_content_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in runtime_rows
    ]
    runtime_content = hashlib.sha256(
        json.dumps(runtime_content_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    runtime_aggregate = hashlib.sha256(
        json.dumps(runtime_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    runtime_path = tmp_path / "committed_runtime_identity.before.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.committed_runtime_identity.v1",
                "git_commit": "b" * 40,
                "file_count": 1,
                "content_sha256": runtime_content,
                "aggregate_sha256": runtime_aggregate,
                "files": runtime_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        training_config=cli.DEFAULT_TRAINING_CONFIG,
        interface_config=cli.DEFAULT_INTERFACE_CONFIG,
        seed=1001,
        run_dir=tmp_path,
    )
    payload = cli._checkpoint_manifest_payload(
        args, global_step=8, stage="smoke"
    )
    names = {Path(path).name for path in payload["files"]}
    assert payload["source_git_commit"] == "b" * 40
    assert payload["committed_runtime_content_sha256"] == runtime_content
    assert payload["creation_runtime_identity_path"] == str(runtime_path.resolve())
    assert {
        "ppo_training_phase_v1.yaml",
        "ppo_interface_v2.yaml",
        "ppo_observation_schema_v2.json",
        "ppo_phase_action_masks_v2.yaml",
        "ppo_phase_objectives_v2.yaml",
        "ppo_reward_v2.yaml",
        "ppo_termination_v2.yaml",
        "ppo_domain_randomization_v2.yaml",
        "frozen_successful_fsm.yaml",
        "environment_lock.json",
        "fsm_states.yaml",
        "recording_motion_contract.json",
    } <= names
