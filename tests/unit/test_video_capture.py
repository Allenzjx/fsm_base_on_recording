from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.infrastructure import video_capture
from wlr50_clean.infrastructure.video_capture import (
    VIDEO_ERROR_CODE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    ActiveViewportVideoRecorder,
    validate_mp4,
)


class _HydraTexture:
    def __init__(self) -> None:
        self.resource = object()

    def get_drawable_ldr_resource(self, _: int) -> object:
        return self.resource


class _Viewport:
    render_product_path = "/Render/ActiveViewport"

    def __init__(self) -> None:
        self._hydra_texture = _HydraTexture()


class _Encoder:
    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = lifecycle
        self.path: Path | None = None
        self.frames = 0

    def start_encoding(self, path: str, fps: int, _: int, live: bool) -> bool:
        assert fps == 15 and live is True
        self.path = Path(path)
        self.lifecycle.append("start")
        return True

    def encode_next_frame_from_buffer(self, frame, width: int, height: int) -> bool:
        assert frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 4)
        assert (width, height) == (VIDEO_WIDTH, VIDEO_HEIGHT)
        self.frames += 1
        self.lifecycle.append("encode")
        return True

    def finalize_encoding(self) -> bool:
        assert self.path is not None
        self.path.write_bytes(b"synthetic-mp4")
        self.lifecycle.append("finalize")
        return True


def _fake_recorder(tmp_path: Path, *, callback_count: int = 1) -> tuple[ActiveViewportVideoRecorder, list[str]]:
    viewport = _Viewport()
    lifecycle: list[str] = []
    pending = []
    encoder = _Encoder(lifecycle)
    rgba = bytes(VIDEO_WIDTH * VIDEO_HEIGHT * 4)

    def schedule(callback, resource) -> None:
        assert resource is viewport._hydra_texture.resource
        pending.append(callback)

    def wait() -> None:
        callback = pending.pop(0)
        for _ in range(callback_count):
            callback(rgba, len(rgba), VIDEO_WIDTH, VIDEO_HEIGHT, "RGBA8")

    def validator(path: Path, **kwargs):
        assert path.read_bytes() == b"synthetic-mp4"
        assert kwargs["expected_frame_count"] == encoder.frames
        assert kwargs["require_sane_container_duration"] is False
        return {
            "valid": True,
            "frame_count": encoder.frames,
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "fps": 15.0,
            "codec": "h264",
            "pixel_format": "yuv420p",
        }

    recorder = ActiveViewportVideoRecorder(
        tmp_path,
        viewport_provider=lambda: viewport,
        capture_scheduler=schedule,
        renderer_wait=wait,
        encoder_provider=lambda: encoder,
        format_validator=lambda value: None,
        video_validator=validator,
    )
    return recorder, lifecycle


def test_active_viewport_recorder_observes_two_existing_renders(tmp_path: Path) -> None:
    recorder, lifecycle = _fake_recorder(tmp_path)
    assert recorder.start() is True
    for step in (8, 16):
        recorder.before_render(sim_step=step, sim_time_s=step / 120.0)
        lifecycle.append("runtime_render")
        recorder.after_render()
        recorder.require_healthy()
    manifest = recorder.finalize()
    assert manifest["valid"] is True
    assert manifest["one_callback_per_render"] is True
    assert manifest["extra_render_count"] == 0
    assert manifest["frame_count"] == 2
    assert lifecycle == [
        "start", "runtime_render", "encode", "runtime_render", "encode", "finalize"
    ]
    rows = [json.loads(line) for line in recorder.ledger_path.read_text().splitlines()]
    assert [row["callback_count"] for row in rows] == [1, 1]
    assert recorder.first_frame_path.read_bytes().startswith(b"\x89PNG")


def test_second_callback_fails_closed(tmp_path: Path) -> None:
    recorder, _ = _fake_recorder(tmp_path, callback_count=2)
    assert recorder.start()
    recorder.before_render(sim_step=8, sim_time_s=8 / 120.0)
    recorder.after_render()
    assert recorder.error_code == VIDEO_ERROR_CODE
    with pytest.raises(video_capture.VideoArtifactError):
        recorder.require_healthy()
    assert recorder.finalize()["valid"] is False


def test_validate_mp4_parses_full_decode_and_monotonic_pts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-used-by-fake-ffmpeg")
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"x")
    stderr = """
Duration: 00:00:00.13, start: 0.000000, bitrate: 100 kb/s
Stream #0:0: Video: h264 (High), yuv420p(progressive), 1280x720, 15 fps, 15 tbr
[showinfo] n: 0 pts: 0 pts_time:0 duration:1 duration_time:0.0667 checksum:AAAA mean:[100 128 128]
[showinfo] n: 1 pts: 1 pts_time:0.0666667 duration:1 duration_time:0.0667 checksum:BBBB mean:[101 128 128]
"""
    monkeypatch.setattr(video_capture, "find_ffmpeg", lambda _: executable)
    monkeypatch.setattr(
        video_capture.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=stderr),
    )
    result = validate_mp4(source, expected_frame_count=2)
    assert result["valid"] is True
    assert result["full_decode"] is True
    assert result["timestamps_monotonic"] is True
    assert result["timestamps_continuous"] is True
    assert result["codec"] == "h264"
    assert result["pixel_format"] == "yuv420p"
    assert result["container_duration_valid"] is True
    assert len(result["frame_pts_sha256"]) == 64
    assert len(result["decoded_frame_checksums_sha256"]) == 64


def test_validate_mp4_rejects_malformed_container_duration(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-used-by-fake-ffmpeg")
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"x")
    stderr = """
Duration: 79536:25:53.00, start: 0.000000, bitrate: 0 kb/s
Stream #0:0: Video: h264 (High), yuv420p(progressive), 1280x720, 15 fps, 15 tbr
[showinfo] n: 0 pts: 0 pts_time:0 duration:1 duration_time:0.0667 checksum:AAAA mean:[100 128 128]
[showinfo] n: 1 pts: 1 pts_time:0.0666667 duration:1 duration_time:0.0667 checksum:BBBB mean:[101 128 128]
"""
    monkeypatch.setattr(video_capture, "find_ffmpeg", lambda _: executable)
    monkeypatch.setattr(
        video_capture.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=stderr),
    )
    result = validate_mp4(source, expected_frame_count=2)
    assert result["valid"] is False
    assert result["container_duration_bounded"] is False
    assert result["container_duration_valid"] is False

    repairable = validate_mp4(
        source,
        expected_frame_count=2,
        require_sane_container_duration=False,
    )
    assert repairable["valid"] is True
    assert repairable["container_duration_required"] is False
    assert repairable["container_duration_valid"] is False


def test_validate_mp4_rejects_non_monotonic_pts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"x")
    stderr = """
Duration: 00:00:00.13, start: 0.000000
Stream #0:0: Video: h264, yuv420p, 1280x720, 15 fps
[showinfo] n: 0 pts: 0 pts_time:0 checksum:AAAA mean:[100 128 128]
[showinfo] n: 1 pts: 0 pts_time:0 checksum:BBBB mean:[101 128 128]
"""
    monkeypatch.setattr(video_capture, "find_ffmpeg", lambda _: executable)
    monkeypatch.setattr(video_capture.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr=stderr))
    result = validate_mp4(source)
    assert result["valid"] is False
    assert result["status"] == VIDEO_ERROR_CODE
    assert result["timestamps_monotonic"] is False
    assert result["timestamps_continuous"] is False
