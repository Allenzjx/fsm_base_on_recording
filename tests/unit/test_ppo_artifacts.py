from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wlr50_clean.ppo import artifacts


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_reserve_run_identity_is_complete_and_never_reused(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "training.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")
    instant = datetime(2026, 9, 3, 12, 34, 56, 123456, tzinfo=timezone.utc)

    reservation = artifacts.reserve_run(
        project_root=tmp_path,
        run_kind="train",
        config_paths=[config],
        seed=7,
        environment_count=16,
        training_stage="phase curriculum",
        timestamp=instant,
        git_commit=COMMIT,
        subcommand="train",
        invocation_arguments=("--training-config", "configs/training.yaml"),
    )

    expected_config = artifacts.config_set_record([config], project_root=tmp_path)[0]
    assert reservation.run_id == (
        f"20260903T123456123456Z_g{COMMIT[:12]}_c{expected_config[:12]}_s7_n16_phase-curriculum"
    )
    manifest = json.loads(reservation.started_manifest.read_text(encoding="utf-8"))
    assert manifest["identity"]["git_commit"] == COMMIT
    assert manifest["identity"]["environment_count"] == 16
    assert manifest["configs"][0]["path"] == "configs/training.yaml"
    assert manifest["invocation_arguments"] == ["--training-config", "configs/training.yaml"]

    with pytest.raises(artifacts.ArtifactError, match="already exists"):
        artifacts.reserve_run(
            project_root=tmp_path,
            run_kind="train",
            config_paths=[config],
            seed=7,
            environment_count=16,
            training_stage="phase curriculum",
            timestamp=instant,
            git_commit=COMMIT,
        )


def test_atomic_json_and_csv_are_exclusive(tmp_path: Path) -> None:
    json_path = artifacts.atomic_write_json(tmp_path / "record.json", {"finite": 1.25})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"finite": 1.25}
    with pytest.raises(artifacts.ArtifactError, match="overwrite"):
        artifacts.atomic_write_json(json_path, {"other": True})
    with pytest.raises(artifacts.ArtifactError, match="serializable"):
        artifacts.atomic_write_json(tmp_path / "nan.json", {"value": float("nan")})

    csv_path = artifacts.atomic_write_csv(
        tmp_path / "metrics.csv",
        [{"phase": "P01", "score": 1.0}, {"phase": "P02", "score": 0.8}],
        fieldnames=("phase", "score"),
    )
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[1] == {"phase": "P02", "score": "0.8"}
    with pytest.raises(artifacts.ArtifactError, match="overwrite"):
        artifacts.atomic_write_csv(csv_path, [], fieldnames=("phase", "score"))


def test_checksum_manifest_detects_tampering(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "nested" / "b.bin"
    second.parent.mkdir()
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    manifest = artifacts.write_checksum_manifest(
        [second, first], tmp_path / "checksums.sha256", root=tmp_path
    )
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"{hashlib.sha256(b'alpha').hexdigest()}  a.bin",
        f"{hashlib.sha256(b'beta').hexdigest()}  nested/b.bin",
    ]
    assert artifacts.verify_checksum_manifest(manifest, root=tmp_path)["valid"] is True
    second.write_bytes(b"tampered")
    verification = artifacts.verify_checksum_manifest(manifest, root=tmp_path)
    assert verification["valid"] is False
    assert verification["entries"][1]["valid"] is False


def test_frozen_hash_audit_detects_missing_and_modified_files(tmp_path: Path) -> None:
    protected = tmp_path / "configs" / "frozen.yaml"
    protected.parent.mkdir()
    protected.write_bytes(b"frozen\n")
    manifest = tmp_path / "frozen_hashes.json"
    manifest.write_text(
        json.dumps(
            {
                "source_head": COMMIT,
                "protected_files": {
                    "configs/frozen.yaml": hashlib.sha256(b"frozen\n").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    valid = artifacts.verify_frozen_hashes(
        project_root=tmp_path, frozen_manifest=manifest
    )
    assert valid["passed"] is True
    assert valid["protected_file_count"] == 1
    assert valid["mismatches"] == []

    protected.write_bytes(b"changed\n")
    changed = artifacts.verify_frozen_hashes(
        project_root=tmp_path, frozen_manifest=manifest
    )
    assert changed["passed"] is False
    assert changed["mismatches"] == ["configs/frozen.yaml"]
    assert changed["entries"][0]["exists"] is True

    protected.unlink()
    missing = artifacts.verify_frozen_hashes(
        project_root=tmp_path, frozen_manifest=manifest
    )
    assert missing["passed"] is False
    assert missing["entries"][0]["exists"] is False


def test_finalize_run_hashes_logs_and_cannot_repeat(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "x.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    reservation = artifacts.reserve_run(
        project_root=tmp_path,
        run_kind="validation",
        config_paths=[config],
        seed=9,
        environment_count=1,
        training_stage="locked-test",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        git_commit=COMMIT,
    )
    (reservation.run_dir / "stdout.log").write_text("ok\n", encoding="utf-8")
    (reservation.run_dir / "stderr.log").write_text("", encoding="utf-8")
    final = artifacts.finalize_run(reservation.run_dir, exit_code=0)
    payload = json.loads(final.read_text(encoding="utf-8"))
    assert payload["lifecycle"] == "SUCCEEDED"
    assert payload["exit_code"] == 0
    assert payload["logs"]["stdout.log"]["sha256"] == artifacts.sha256_file(
        reservation.run_dir / "stdout.log"
    )
    assert payload["artifacts"] == {}
    with pytest.raises(artifacts.ArtifactError, match="already finalized"):
        artifacts.finalize_run(reservation.run_dir, exit_code=0)


def _probe_payload(*, codec: str = "h264", timestamps: tuple[float, ...] = (0.0, 1 / 15, 2 / 15)) -> dict:
    return {
        "streams": [
            {
                "codec_name": codec,
                "pix_fmt": "yuv420p",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "nb_frames": str(len(timestamps)),
                "duration": "0.2",
            }
        ],
        "format": {"duration": "0.2", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "frames": [{"best_effort_timestamp_time": str(value)} for value in timestamps],
    }


def test_video_validation_uses_ffprobe_and_independent_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"not decoded because subprocess is mocked")
    probe = tmp_path / "ffprobe.exe"
    ffmpeg = tmp_path / "ffmpeg.exe"
    probe.write_bytes(b"x")
    ffmpeg.write_bytes(b"x")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == str(probe.resolve()):
            return subprocess.CompletedProcess(command, 0, json.dumps(_probe_payload()), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    result = artifacts.validate_video(
        video,
        source_episode="episode-1",
        source_checkpoint="checkpoint.pt",
        source_seed=42,
        stitched=False,
        speed_modified=False,
        ffprobe=probe,
        ffmpeg=ffmpeg,
        expected_fps=15.0,
        expected_resolution=(1280, 720),
    )

    assert result["valid"] is True
    assert result["codec"] == "h264"
    assert result["pixel_format"] == "yuv420p"
    assert result["full_decode"] is True
    assert result["timestamps_monotonic"] is True
    assert result["source_episode"] == "episode-1"
    assert len(calls) == 2
    assert "-show_frames" in calls[0]
    assert calls[1][-2:] == ["null", artifacts.os.devnull]


@pytest.mark.parametrize(
    ("codec", "timestamps", "stitched", "speed_modified"),
    [
        ("hevc", (0.0, 1 / 15, 2 / 15), False, False),
        ("h264", (0.0, 2 / 15, 1 / 15), False, False),
        ("h264", (0.0, 1 / 15, 2 / 15), True, False),
        ("h264", (0.0, 1 / 15, 2 / 15), False, True),
    ],
)
def test_video_validation_rejects_noncompliant_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
    timestamps: tuple[float, ...],
    stitched: bool,
    speed_modified: bool,
) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    probe = tmp_path / "ffprobe"
    ffmpeg = tmp_path / "ffmpeg"
    probe.write_bytes(b"x")
    ffmpeg.write_bytes(b"x")

    def fake_run(command, **kwargs):
        if command[0] == str(probe.resolve()):
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_probe_payload(codec=codec, timestamps=timestamps)), ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)
    result = artifacts.validate_video(
        video,
        source_episode="e",
        source_checkpoint="c",
        source_seed=1,
        stitched=stitched,
        speed_modified=speed_modified,
        ffprobe=probe,
        ffmpeg=ffmpeg,
    )
    assert result["valid"] is False
    assert result["status"] == "VIDEO_OR_ARTIFACT_ERROR"
