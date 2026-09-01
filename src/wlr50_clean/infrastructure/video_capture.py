"""Fail-closed active Isaac GUI viewport capture with no extra render.

The one-pending-frame design is narrowly derived from mature Recording source
SHA-256 ``9812b3896b843eda3db20a517856dc45b3637c91894172ffe9def613b6409a2b``.
Isaac/Kit imports remain lazy until SimulationApp exists.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


VIDEO_ERROR_CODE = "VIDEO_OR_ARTIFACT_ERROR"
CAPTURE_BACKEND = "active_viewport_ldr_byte_buffer_to_omni_videoencoding"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 15.0
MAX_VIDEO_DURATION_S = 200.0

class VideoArtifactError(RuntimeError):
    """Raised when required video evidence cannot be trusted."""

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, path)

def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

def _write_rgba_png(path: Path, captured: tuple[bytes, int, int]) -> None:
    """Write a standards-compliant RGBA PNG without an image dependency."""
    payload, width, height = captured
    row_bytes = width * 4
    expected = row_bytes * height
    if len(payload) != expected:
        raise ValueError(f"RGBA checkpoint has {len(payload)} bytes, expected {expected}")
    scanlines = b"".join(
        b"\x00" + payload[offset : offset + row_bytes]
        for offset in range(0, expected, row_bytes)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(encoded)

def _capsule_rgba_bytes(buffer: Any, size: int) -> bytes:
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        payload = bytes(buffer)
        if len(payload) != int(size):
            raise ValueError(f"capture buffer has {len(payload)} bytes, expected {int(size)}")
        return payload
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    address = get_pointer(buffer, None)
    if not address:
        raise RuntimeError("PyCapsule_GetPointer returned a null address")
    return ctypes.string_at(address, int(size))

def _strict_rgba8_format(byte_format: Any) -> None:
    try:
        import omni.ui as ui  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised inside Kit
        raise RuntimeError(f"omni.ui TextureFormat unavailable: {exc}") from exc
    if byte_format != ui.TextureFormat.RGBA8_UNORM:
        raise ValueError(f"active viewport LdrColor is not RGBA8_UNORM: {byte_format!r}")

def find_ffmpeg(explicit: Path | str | None = None) -> Path:
    """Resolve ffmpeg without importing Isaac or requiring a system install."""

    candidates: list[str] = []
    if explicit is not None:
        candidates.append(str(explicit))
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if configured:
        candidates.append(configured)
    try:
        import imageio_ffmpeg  # type: ignore

        candidates.append(str(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass
    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(discovered)
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError("ffmpeg was not found; set IMAGEIO_FFMPEG_EXE")


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(
    r"Stream #0:\d+.*?Video:\s*([^\s,(]+).*?,\s*([A-Za-z0-9_]+)(?:\([^)]*\))?,\s*(\d+)x(\d+)"
)
_FPS_RE = re.compile(r",\s*([0-9]+(?:\.[0-9]+)?)\s+fps\b")
_FRAME_RE = re.compile(
    r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+0-9.eE]+).*?"
    r"checksum:([0-9A-Fa-f]+).*?mean:\[([^]]+)\]"
)


def _duration_seconds(match: re.Match[str]) -> float:
    return 3600.0 * int(match.group(1)) + 60.0 * int(match.group(2)) + float(match.group(3))

def validate_mp4(
    path: Path,
    *,
    ffmpeg: Path | str | None = None,
    expected_width: int = VIDEO_WIDTH,
    expected_height: int = VIDEO_HEIGHT,
    expected_fps: float = VIDEO_FPS,
    expected_frame_count: int | None = None,
    maximum_duration_s: float = MAX_VIDEO_DURATION_S,
    stitched: bool = False,
    speed_modified: bool = False,
) -> dict[str, Any]:
    """Fully decode; inspect every PTS/checksum rather than trusting metadata."""
    source = Path(path).resolve()
    result: dict[str, Any] = {
        "schema": "wlr50_clean.video_validation.v1",
        "path": str(source),
        "valid": False,
        "status": VIDEO_ERROR_CODE,
        "stitched": bool(stitched),
        "speed_modified": bool(speed_modified),
        "full_decode": False,
        "timestamps_monotonic": False,
        "error": "",
    }
    if not source.is_file() or source.stat().st_size <= 0:
        result["error"] = "video file is missing or empty"
        return result
    try:
        executable = find_ffmpeg(ffmpeg)
        command = [str(executable), "-hide_banner", "-nostdin", "-i", str(source),
                   "-map", "0:v:0", "-vf", "showinfo", "-an", "-fps_mode",
                   "passthrough", "-f", "null", os.devnull]
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
        text = completed.stderr
        duration_match = _DURATION_RE.search(text)
        stream_match = _VIDEO_RE.search(text)
        fps_match = _FPS_RE.search(text[: stream_match.end() + 300] if stream_match else text)
        timestamps: list[float] = []
        checksums: list[str] = []
        mean_luma: list[float] = []
        frame_numbers: list[int] = []
        for match in _FRAME_RE.finditer(text):
            frame_numbers.append(int(match.group(1)))
            timestamps.append(float(match.group(3)))
            checksums.append(match.group(4).upper())
            mean_luma.append(float(match.group(5).split()[0]))
        codec = stream_match.group(1).lower() if stream_match else ""
        pixel_format = stream_match.group(2).lower() if stream_match else ""
        width = int(stream_match.group(3)) if stream_match else 0
        height = int(stream_match.group(4)) if stream_match else 0
        fps = float(fps_match.group(1)) if fps_match else 0.0
        container_duration = _duration_seconds(duration_match) if duration_match else 0.0
        count = len(timestamps)
        sequential_frames = frame_numbers == list(range(count))
        monotonic = count > 0 and all(b > a for a, b in zip(timestamps, timestamps[1:]))
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
        decoded_duration = (
            timestamps[-1] - timestamps[0] + (statistics.median(deltas) if deltas else 1.0 / fps)
            if count and fps > 0.0 else 0.0
        )
        longest_repeat = 1 if checksums else 0
        current_repeat = 1
        for previous, current in zip(checksums, checksums[1:]):
            current_repeat = current_repeat + 1 if current == previous else 1
            longest_repeat = max(longest_repeat, current_repeat)
        black_like = sum(value <= 20.0 for value in mean_luma)
        longest_allowed_repeat = max(
            int(round(10.0 * expected_fps)), int(math.ceil(0.25 * count))
        )
        motion_evidence = len(set(checksums)) >= 2 and longest_repeat <= longest_allowed_repeat
        black_cover_absent = count > 0 and black_like <= max(1, int(math.floor(0.10 * count)))
        fps_ok = math.isclose(fps, float(expected_fps), rel_tol=0.0, abs_tol=0.01)
        duration_ok = 0.0 < decoded_duration <= float(maximum_duration_s) + 1.0 / expected_fps
        frame_count_ok = expected_frame_count is None or count == int(expected_frame_count)
        valid = bool(completed.returncode == 0 and count >= 2 and sequential_frames
                     and monotonic and codec == "h264" and pixel_format == "yuv420p"
                     and width == int(expected_width) and height == int(expected_height)
                     and fps_ok and duration_ok and frame_count_ok and not stitched
                     and not speed_modified and black_cover_absent and motion_evidence)
        result.update(
            {
                "valid": valid,
                "status": "PASS" if valid else VIDEO_ERROR_CODE,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "duration_s": decoded_duration,
                "container_duration_s": container_duration,
                "fps": fps,
                "frame_count": count,
                "resolution": [width, height],
                "width": width,
                "height": height,
                "codec": codec,
                "pixel_format": pixel_format,
                "full_decode": completed.returncode == 0 and count >= 2 and sequential_frames,
                "timestamps_monotonic": monotonic,
                "first_pts_s": timestamps[0] if timestamps else None,
                "last_pts_s": timestamps[-1] if timestamps else None,
                "unique_frame_checksums": len(set(checksums)),
                "longest_identical_frame_run": longest_repeat,
                "motion_evidence_valid": motion_evidence,
                "black_like_frame_count": black_like,
                "black_cover_absent": black_cover_absent,
                "expected_frame_count": expected_frame_count,
                "frame_count_matches": frame_count_ok,
                "ffmpeg_path": str(executable),
                "ffmpeg_returncode": completed.returncode,
                "error": "" if valid else "one or more video acceptance checks failed",
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

def require_valid_mp4(path: Path, **kwargs: Any) -> dict[str, Any]:
    validation = validate_mp4(path, **kwargs)
    if validation.get("valid") is not True:
        raise VideoArtifactError(str(validation.get("error") or VIDEO_ERROR_CODE))
    return validation

class ActiveViewportVideoRecorder:
    """Exactly one active-render-product capture callback per runtime render."""

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = True,
        fps: float = VIDEO_FPS,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        viewport_provider: Callable[[], Any] | None = None,
        capture_scheduler: Callable[..., Any] | None = None,
        renderer_wait: Callable[[], Any] | None = None,
        encoder_provider: Callable[[], Any] | None = None,
        buffer_copier: Callable[[Any, int], bytes] = _capsule_rgba_bytes,
        format_validator: Callable[[Any], None] = _strict_rgba8_format,
        video_validator: Callable[..., dict[str, Any]] = validate_mp4,
    ) -> None:
        if not math.isclose(float(fps), VIDEO_FPS, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"capture fps is locked to {VIDEO_FPS:g}")
        if (int(width), int(height)) != (VIDEO_WIDTH, VIDEO_HEIGHT):
            raise ValueError(f"capture resolution is locked to {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
        self.root = Path(root).resolve()
        self.video_path = self.root / "actual_viewport_video.mp4"
        self.ledger_path = self.root / "viewport_frame_ledger.jsonl"
        self.manifest_path = self.root / "viewport_buffer_video_manifest.json"
        self.first_frame_path = self.root / "viewport_first_frame.png"
        self.last_frame_path = self.root / "viewport_last_frame.png"
        self.enabled = bool(enabled)
        self.viewport_provider = viewport_provider
        self.capture_scheduler = capture_scheduler
        self.renderer_wait = renderer_wait
        self.encoder_provider = encoder_provider
        self.buffer_copier = buffer_copier
        self.format_validator = format_validator
        self.video_validator = video_validator
        self.viewport: Any | None = None
        self.encoder: Any | None = None
        self.render_product_path = ""
        self.viewport_identity = 0
        self.viewport_identity_check_count = 0
        self.render_product_unchanged = False
        self.started = False
        self.finalized = False
        self._encoding_started = False
        self._encoder_finalized = False
        self._pending: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._first_frame: tuple[bytes, int, int] | None = None
        self._last_frame: tuple[bytes, int, int] | None = None
        self.error = ""

    @property
    def error_code(self) -> str:
        return VIDEO_ERROR_CODE if self.error else ""

    def _fail(self, message: str) -> None:
        if not self.error:
            self.error = message

    def require_healthy(self) -> None:
        if self.error:
            raise VideoArtifactError(f"{VIDEO_ERROR_CODE}: {self.error}")
    def _load_production_dependencies(self) -> None:
        if self.viewport_provider is None:
            from omni.kit.viewport.utility import get_active_viewport  # type: ignore

            self.viewport_provider = get_active_viewport
        if self.capture_scheduler is None or self.renderer_wait is None:
            import omni.kit.renderer_capture  # type: ignore

            interface = omni.kit.renderer_capture.acquire_renderer_capture_interface()
            self.capture_scheduler = self.capture_scheduler or interface.capture_next_frame_rp_resource_callback
            self.renderer_wait = self.renderer_wait or interface.wait_async_capture
        if self.encoder_provider is None:
            from video_encoding import get_video_encoding_interface  # type: ignore

            self.encoder_provider = get_video_encoding_interface
    def start(self) -> bool:
        if self.started:
            return not self.error
        if not self.enabled:
            self._fail("active viewport video is disabled")
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_production_dependencies()
            assert self.viewport_provider is not None and self.encoder_provider is not None
            self.viewport = self.viewport_provider()
            if self.viewport is None:
                raise RuntimeError("active GUI viewport is unavailable")
            self.viewport_identity = id(self.viewport)
            self.render_product_path = str(getattr(self.viewport, "render_product_path", "") or "")
            if not self.render_product_path:
                raise RuntimeError("active viewport render_product_path is unavailable")
            self.encoder = self.encoder_provider()
            if self.encoder is None:
                raise RuntimeError("omni.videoencoding interface is unavailable")
            started = self.encoder.start_encoding(str(self.video_path), int(VIDEO_FPS), 0, True)
            if started is not True:
                raise RuntimeError("omni.videoencoding.start_encoding returned false")
            self._encoding_started = True
            self.render_product_unchanged = True
            self.started = True
        except Exception as exc:
            self._fail(f"start failed: {type(exc).__name__}: {exc}")
        return self.started and not self.error

    def _verify_viewport(self, phase: str) -> None:
        if self.viewport_provider is None or self.viewport is None:
            raise RuntimeError("active viewport identity evidence is unavailable")
        active = self.viewport_provider()
        self.viewport_identity_check_count += 1
        if active is not self.viewport or id(active) != self.viewport_identity:
            self.render_product_unchanged = False
            raise RuntimeError(f"active viewport identity changed during {phase}")
        current_path = str(getattr(active, "render_product_path", "") or "")
        if current_path != self.render_product_path:
            self.render_product_unchanged = False
            raise RuntimeError(f"active viewport render product changed during {phase}")

    def before_render(self, *, sim_step: int, sim_time_s: float) -> None:
        if not self.started or self.finalized or self.error:
            self._fail("before_render called while recorder is unavailable")
            return
        if self._pending is not None:
            self._fail("previous viewport capture is still pending")
            return
        try:
            self._verify_viewport("before_render")
            step, time_s = int(sim_step), float(sim_time_s)
            if step < 0 or not math.isfinite(time_s):
                raise ValueError("render evidence step/time is invalid")
            if self._rows:
                previous = self._rows[-1]
                if step <= int(previous["sim_step"]) or time_s <= float(previous["sim_time_s"]):
                    raise ValueError("render evidence step/time is not strictly increasing")
            pending: dict[str, Any] = {
                "render_sequence": len(self._rows), "sim_step": step, "sim_time_s": time_s,
                "callback_count": 0, "capture_bytes": None, "width": 0, "height": 0,
                "byte_format": "", "capture_resource_identity": 0,
            }
            self._pending = pending

            def callback(buffer: Any, buffer_size: int, width: int, height: int, byte_format: Any) -> None:
                current = self._pending
                if current is None:
                    self._fail("viewport callback arrived without a pending render")
                    return
                current["callback_count"] = int(current["callback_count"]) + 1
                if int(current["callback_count"]) != 1:
                    self._fail("viewport render produced more than one callback")
                    return
                try:
                    parsed_width, parsed_height, parsed_size = int(width), int(height), int(buffer_size)
                    if (parsed_width, parsed_height) != (VIDEO_WIDTH, VIDEO_HEIGHT):
                        raise ValueError(
                            f"captured {parsed_width}x{parsed_height}; expected {VIDEO_WIDTH}x{VIDEO_HEIGHT}"
                        )
                    expected_size = VIDEO_WIDTH * VIDEO_HEIGHT * 4
                    if parsed_size != expected_size:
                        raise ValueError(f"RGBA buffer has {parsed_size} bytes; expected {expected_size}")
                    self.format_validator(byte_format)
                    captured = self.buffer_copier(buffer, parsed_size)
                    if len(captured) != expected_size:
                        raise ValueError("copied RGBA byte count is invalid")
                    current.update(
                        capture_bytes=captured, width=parsed_width, height=parsed_height,
                        byte_format=str(byte_format),
                    )
                except Exception as exc:
                    self._fail(f"callback failed: {type(exc).__name__}: {exc}")

            hydra_texture = getattr(self.viewport, "_hydra_texture", None)
            getter = None if hydra_texture is None else (
                getattr(hydra_texture, "get_drawable_ldr_resource", None)
                or getattr(hydra_texture, "_get_drawable_ldr_resource", None)
            )
            if not callable(getter):
                raise RuntimeError("active viewport LdrColor RpResource getter is unavailable")
            resource = getter(0)
            if resource is None:
                raise RuntimeError("active viewport LdrColor RpResource is unavailable")
            pending["capture_resource_identity"] = id(resource)
            assert self.capture_scheduler is not None
            self.capture_scheduler(callback, resource)
        except Exception as exc:
            self._fail(f"schedule failed: {type(exc).__name__}: {exc}")

    def after_render(self) -> None:
        if not self.started or self.finalized:
            self._fail("after_render called while recorder is unavailable")
            return
        pending = self._pending
        if pending is None:
            self._fail("render completed without a scheduled viewport capture")
            return
        try:
            assert self.renderer_wait is not None
            self.renderer_wait()
            self._verify_viewport("after_render")
            if self.error:
                return
            if int(pending["callback_count"]) != 1:
                raise RuntimeError(f"viewport callback_count={pending['callback_count']}, expected 1")
            payload = pending["capture_bytes"]
            if not isinstance(payload, bytes):
                raise RuntimeError("viewport callback did not provide copied bytes")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 4)
            assert self.encoder is not None
            encoded = self.encoder.encode_next_frame_from_buffer(frame, VIDEO_WIDTH, VIDEO_HEIGHT)
            if encoded is not True:
                raise RuntimeError("omni.videoencoding.encode_next_frame_from_buffer returned false")
            stored = (payload, VIDEO_WIDTH, VIDEO_HEIGHT)
            self._first_frame = self._first_frame or stored
            self._last_frame = stored
            self._rows.append(
                {
                    "render_sequence": int(pending["render_sequence"]),
                    "sim_step": int(pending["sim_step"]),
                    "sim_time_s": float(pending["sim_time_s"]),
                    "encoded_frame_index": len(self._rows),
                    "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT,
                    "rgba_buffer_size": len(payload),
                    "byte_format": str(pending["byte_format"]),
                    "capture_backend": CAPTURE_BACKEND,
                    "render_product_path": self.render_product_path,
                    "viewport_identity": self.viewport_identity,
                    "capture_resource_identity": int(pending["capture_resource_identity"]),
                    "callback_count": 1,
                }
            )
        except Exception as exc:
            self._fail(f"encode failed: {type(exc).__name__}: {exc}")
        finally:
            self._pending = None
    def finalize(self) -> dict[str, Any]:
        """Finalize the encoder first, then validate and publish its evidence."""
        if self.finalized and self.manifest_path.is_file():
            return dict(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        self.finalized = True
        if self.started and not self.error:
            try:
                self._verify_viewport("finalize")
            except Exception as exc:
                self._fail(f"viewport identity failed: {type(exc).__name__}: {exc}")
        if self._pending is not None:
            self._fail("video finalized with a pending viewport capture")
        # This must happen while SimulationApp still exists. Runtime ownership
        # is explicit: recorder.finalize(), then simulation_app.close().
        if self.encoder is not None and self._encoding_started:
            try:
                finalized = self.encoder.finalize_encoding()
                if finalized is False:
                    self._fail("omni.videoencoding.finalize_encoding returned false")
                else:
                    self._encoder_finalized = True
            except Exception as exc:
                self._fail(f"encoder finalize failed: {type(exc).__name__}: {exc}")
        try:
            if self._first_frame is not None:
                _write_rgba_png(self.first_frame_path, self._first_frame)
            if self._last_frame is not None:
                _write_rgba_png(self.last_frame_path, self._last_frame)
            _write_jsonl(self.ledger_path, self._rows)
        except Exception as exc:
            self._fail(f"ledger/checkpoint write failed: {type(exc).__name__}: {exc}")
        decode: dict[str, Any] = {"valid": False, "error": "video unavailable"}
        if self.video_path.is_file() and self.video_path.stat().st_size > 0:
            try:
                decode = self.video_validator(
                    self.video_path,
                    expected_frame_count=len(self._rows),
                    stitched=False,
                    speed_modified=False,
                )
            except Exception as exc:
                self._fail(f"full decode failed: {type(exc).__name__}: {exc}")
        if decode.get("valid") is not True:
            self._fail(str(decode.get("error") or "encoded video failed validation"))
        count = len(self._rows)
        ledger_complete = bool(
            count >= 2
            and [int(row["render_sequence"]) for row in self._rows] == list(range(count))
            and [int(row["encoded_frame_index"]) for row in self._rows] == list(range(count))
            and all(int(row["callback_count"]) == 1 for row in self._rows)
        )
        identity_proven = bool(
            self.render_product_unchanged
            and self.viewport_identity_check_count >= 2 * count + 1
        )
        valid = bool(
            self.started and not self.error and ledger_complete and identity_proven
            and decode.get("valid") is True
            and self._encoder_finalized
            and self.first_frame_path.is_file() and self.last_frame_path.is_file()
        )
        manifest = {
            "schema": "wlr50_clean.active_viewport_video.v1",
            "valid": valid,
            "status": "PASS" if valid else VIDEO_ERROR_CODE,
            "error": self.error,
            "source": "actual_active_isaac_gui_viewport_render_product",
            "capture_backend": CAPTURE_BACKEND,
            "render_product_path": self.render_product_path,
            "viewport_identity": self.viewport_identity,
            "viewport_identity_check_count": self.viewport_identity_check_count,
            "render_product_unchanged": self.render_product_unchanged,
            "active_render_product_identity_proven": identity_proven,
            "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT, "fps": VIDEO_FPS,
            "frame_count": count,
            "frame_ledger_complete": ledger_complete,
            "one_callback_per_render": ledger_complete,
            "extra_app_update_count": 0,
            "extra_render_count": 0,
            "render_observer_only": True,
            "maximum_pending_captures": 1,
            "encoder_finalized_before_app_close": self._encoder_finalized,
            "stitched": False,
            "speed_modified": False,
            "video_path": str(self.video_path),
            "video_sha256": sha256_file(self.video_path) if self.video_path.is_file() else "",
            "ledger_path": str(self.ledger_path),
            "ledger_sha256": sha256_file(self.ledger_path) if self.ledger_path.is_file() else "",
            "first_frame_path": str(self.first_frame_path),
            "first_frame_sha256": sha256_file(self.first_frame_path) if self.first_frame_path.is_file() else "",
            "last_frame_path": str(self.last_frame_path),
            "last_frame_sha256": sha256_file(self.last_frame_path) if self.last_frame_path.is_file() else "",
            "full_decode": decode,
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest
# Short compatibility alias for integrations that used the mature class name.
ActiveViewportBufferVideoRecorder = ActiveViewportVideoRecorder
