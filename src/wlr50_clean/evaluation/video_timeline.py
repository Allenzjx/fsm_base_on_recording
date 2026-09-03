"""Auditable mapping from simulator phase time to encoded video frame PTS.

The viewport encoder is driven at 15 Hz while the controller and semantic
evidence use the 120 Hz simulator clock.  Container duration is deliberately
not used here: legacy NVIDIA MP4 files can have a corrupt duration atom even
though every H.264 frame and its PTS decode correctly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wlr50_clean.infrastructure.video_capture import VIDEO_FPS, find_ffmpeg, sha256_file


class VideoTimelineError(ValueError):
    """A source video cannot be bound to its simulator-time frame ledger."""


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    frame_index: int
    pts_s: float
    checksum: str
    key_frame: bool


@dataclass(frozen=True, slots=True)
class LedgerFrame:
    frame_index: int
    sim_step: int
    sim_time_s: float


def _sha256_text_rows(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def _sha256_float64(values: Iterable[float]) -> str:
    return hashlib.sha256(
        b"".join(struct.pack(">d", float(value)) for value in values)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ActionWindow:
    """One contiguous, frame-exact source interval containing the full action."""

    semantic_start_sim_s: float
    semantic_end_sim_s: float
    source_first_frame_index: int
    source_last_frame_index: int
    source_first_pts_s: float
    source_last_pts_s: float
    source_first_sim_time_s: float
    source_last_sim_time_s: float
    trim_start_pts_s: float
    trim_end_pts_s: float
    output_duration_s: float
    expected_frame_count: int
    leading_frames_removed: int
    trailing_frames_removed: int
    phase_clock_origin_sim_s: float
    semantic_start_output_s: float
    semantic_end_output_s: float
    action_frame_start_output_pts_s: float
    action_frame_end_output_pts_s_exclusive: float
    requested_pre_roll_s: float
    requested_post_roll_s: float
    maximum_preserved_roll_s: float
    available_pre_roll_s: float
    available_post_roll_s: float
    retained_pre_roll_s: float
    retained_post_roll_s: float
    capture_lag_after_action_start_s: float
    packet_copy_start_is_keyframe: bool
    source_selected_pts_sha256: str
    source_selected_pts_delta_sha256: str
    source_selected_checksums_sha256: str
    ledger_to_pts_offset_s: float
    maximum_ledger_to_pts_offset_deviation_s: float

    @property
    def is_full_source(self) -> bool:
        return self.leading_frames_removed == 0 and self.trailing_frames_removed == 0

    @property
    def packet_copy_eligible(self) -> bool:
        # An end cut can discard complete packets anywhere.  A decodable stream-
        # copy start must begin at an independently decodable key frame.
        return self.packet_copy_start_is_keyframe

    def map_sim_time_to_output_pts(self, sim_time_s: float) -> float:
        return max(0.0, float(sim_time_s) - self.phase_clock_origin_sim_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "wlr50_clean.video_action_window.v1",
            "selection_policy": {
                "start": "frame_at_or_immediately_before_semantic_start",
                "end": "frame_at_or_immediately_after_semantic_end_inclusive",
                "clock": "viewport_frame_ledger_sim_time_bound_to_decoded_frame_pts",
                "container_duration_used": False,
                "single_contiguous_source": True,
                "speed_modified": False,
            },
            "semantic_bounds_sim_s": {
                "start": self.semantic_start_sim_s,
                "end": self.semantic_end_sim_s,
            },
            "source_frames": {
                "first_index": self.source_first_frame_index,
                "last_index": self.source_last_frame_index,
                "count": self.expected_frame_count,
                "first_pts_s": self.source_first_pts_s,
                "last_pts_s": self.source_last_pts_s,
                "first_sim_time_s": self.source_first_sim_time_s,
                "last_sim_time_s": self.source_last_sim_time_s,
                "selected_pts_sha256": self.source_selected_pts_sha256,
                "selected_pts_delta_sha256": self.source_selected_pts_delta_sha256,
                "selected_decoded_checksums_sha256": self.source_selected_checksums_sha256,
            },
            "trim": {
                "start_pts_s": self.trim_start_pts_s,
                "end_pts_s_exclusive": self.trim_end_pts_s,
                "leading_frames_removed": self.leading_frames_removed,
                "trailing_frames_removed": self.trailing_frames_removed,
                "full_source": self.is_full_source,
            },
            "output_clock": {
                "duration_s": self.output_duration_s,
                "phase_clock_origin_sim_s": self.phase_clock_origin_sim_s,
                "detected_action_start_s": self.semantic_start_output_s,
                "detected_action_end_s": self.semantic_end_output_s,
                "action_frame_start_pts_s": self.action_frame_start_output_pts_s,
                "action_frame_end_pts_s_exclusive": (
                    self.action_frame_end_output_pts_s_exclusive
                ),
            },
            "context_roll": {
                "requested_pre_roll_s": self.requested_pre_roll_s,
                "requested_post_roll_s": self.requested_post_roll_s,
                "maximum_preserved_roll_s": self.maximum_preserved_roll_s,
                "available_pre_roll_s": self.available_pre_roll_s,
                "available_post_roll_s": self.available_post_roll_s,
                "retained_pre_roll_s": self.retained_pre_roll_s,
                "retained_post_roll_s": self.retained_post_roll_s,
                "capture_lag_after_action_start_s": self.capture_lag_after_action_start_s,
                "policy": (
                    "preserve the full source edge when its available context is "
                    "already within the allowed roll; otherwise retain the requested roll"
                ),
            },
            "ledger_pts_binding": {
                "sim_minus_pts_offset_s": self.ledger_to_pts_offset_s,
                "maximum_offset_deviation_s": self.maximum_ledger_to_pts_offset_deviation_s,
            },
            "packet_copy": {
                "eligible": self.packet_copy_eligible,
                "start_is_keyframe": self.packet_copy_start_is_keyframe,
                "fallback_if_false": "single-source frame-exact H.264 encode",
            },
        }


_SHOWINFO_FRAME_RE = re.compile(
    r"\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:\s*([-+0-9.eE]+).*?"
    r"iskey:\s*([01]).*?checksum:([0-9A-Fa-f]+)"
)


def decode_frame_timeline(
    path: Path,
    *,
    ffmpeg: Path | str | None = None,
) -> tuple[DecodedVideoFrame, ...]:
    """Fully decode a video and return each frame's PTS, checksum, and key flag."""

    source = Path(path).resolve()
    if not source.is_file():
        raise VideoTimelineError(f"video is missing: {source}")
    executable = find_ffmpeg(ffmpeg)
    completed = subprocess.run(
        [
            str(executable), "-hide_banner", "-nostats", "-nostdin", "-i", str(source),
            "-map", "0:v:0", "-vf", "showinfo", "-an", "-fps_mode", "passthrough",
            "-f", "null", os.devnull,
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    rows = tuple(
        DecodedVideoFrame(
            frame_index=int(match.group(1)),
            pts_s=float(match.group(2)),
            key_frame=match.group(3) == "1",
            checksum=match.group(4).upper(),
        )
        for match in _SHOWINFO_FRAME_RE.finditer(completed.stderr)
    )
    if completed.returncode != 0 or not rows:
        tail = completed.stderr[-3000:].replace("\r", " ").replace("\n", " ")
        raise VideoTimelineError(
            f"full frame-timeline decode failed ({completed.returncode}): {tail}"
        )
    if tuple(row.frame_index for row in rows) != tuple(range(len(rows))):
        raise VideoTimelineError("decoded frame indices are not contiguous from zero")
    if not all(right.pts_s > left.pts_s for left, right in zip(rows, rows[1:])):
        raise VideoTimelineError("decoded frame PTS are not strictly monotonic")
    return rows


def load_viewport_frame_ledger(path: Path) -> tuple[LedgerFrame, ...]:
    ledger_path = Path(path).resolve()
    if not ledger_path.is_file():
        raise VideoTimelineError(f"viewport frame ledger is missing: {ledger_path}")
    rows: list[LedgerFrame] = []
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            row = LedgerFrame(
                frame_index=int(payload["encoded_frame_index"]),
                sim_step=int(payload["sim_step"]),
                sim_time_s=float(payload["sim_time_s"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VideoTimelineError(
                f"invalid viewport frame ledger row {line_number}"
            ) from exc
        if not math.isfinite(row.sim_time_s):
            raise VideoTimelineError(f"non-finite sim time at ledger row {line_number}")
        rows.append(row)
    if not rows:
        raise VideoTimelineError("viewport frame ledger is empty")
    if tuple(row.frame_index for row in rows) != tuple(range(len(rows))):
        raise VideoTimelineError("ledger encoded_frame_index is not contiguous from zero")
    if not all(
        right.sim_step > left.sim_step and right.sim_time_s > left.sim_time_s
        for left, right in zip(rows, rows[1:])
    ):
        raise VideoTimelineError("ledger simulator time/step is not strictly monotonic")
    return tuple(rows)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoTimelineError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise VideoTimelineError(f"{label} is not finite")
    return result


def reference_action_bounds(contract_path: Path) -> tuple[float, float]:
    """Return P01 command onset through the P13 semantic action endpoint."""

    payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    phases = payload.get("phases")
    if not isinstance(phases, list) or len(phases) != 13:
        raise VideoTimelineError("reference contract must contain 13 phases")
    ids = tuple(str(row.get("state_id", "")) for row in phases if isinstance(row, Mapping))
    expected = tuple(f"P{index:02d}" for index in range(1, 14))
    if ids != expected:
        raise VideoTimelineError("reference contract phase order is not P01--P13")
    start = _finite(phases[0].get("reference_sim_start_s"), "P01 reference start")
    last = phases[-1]
    end_value = last.get("reference_sim_end_s")
    if end_value is None:
        end_value = _finite(last.get("reference_sim_start_s"), "P13 reference start") + _finite(
            last.get("active_duration_s"), "P13 active duration"
        )
    end = _finite(end_value, "P13 reference end")
    if end <= start:
        raise VideoTimelineError("reference action bounds are empty or reversed")
    return start, end


def fsm_action_bounds(trial_manifest_path: Path) -> tuple[float, float]:
    """Return P01 entry through P13 completion, excluding recorder shutdown."""

    payload = json.loads(Path(trial_manifest_path).read_text(encoding="utf-8"))
    phase_times = payload.get("phase_times")
    if not isinstance(phase_times, Mapping):
        raise VideoTimelineError("trial manifest has no phase_times object")
    expected = tuple(f"P{index:02d}" for index in range(1, 14))
    if tuple(phase_times) != expected:
        raise VideoTimelineError("trial phase_times order is not P01--P13")
    first = phase_times["P01"]
    last = phase_times["P13"]
    if not isinstance(first, Mapping) or not isinstance(last, Mapping):
        raise VideoTimelineError("trial phase timing rows are invalid")
    start = _finite(first.get("entry_time_s"), "P01 entry time")
    end = _finite(last.get("completion_time_s"), "P13 completion time")
    if end <= start:
        raise VideoTimelineError("FSM action bounds are empty or reversed")
    return start, end


def plan_action_window(
    decoded_frames: Sequence[DecodedVideoFrame],
    ledger_frames: Sequence[LedgerFrame],
    *,
    semantic_start_sim_s: float,
    semantic_end_sim_s: float,
    expected_fps: float = VIDEO_FPS,
    requested_pre_roll_s: float = 0.5,
    requested_post_roll_s: float = 0.5,
    maximum_preserved_roll_s: float = 1.0,
) -> ActionWindow:
    """Bind semantic bounds to a contiguous inclusive range of encoded frames."""

    frames = tuple(decoded_frames)
    ledger = tuple(ledger_frames)
    if len(frames) < 2 or len(frames) != len(ledger):
        raise VideoTimelineError("decoded video and frame ledger counts do not match")
    if any(frame.frame_index != row.frame_index for frame, row in zip(frames, ledger)):
        raise VideoTimelineError("decoded frame indices do not match the viewport ledger")
    start = _finite(semantic_start_sim_s, "semantic action start")
    end = _finite(semantic_end_sim_s, "semantic action end")
    if end <= start:
        raise VideoTimelineError("semantic action bounds are empty or reversed")
    requested_pre = _finite(requested_pre_roll_s, "requested pre-roll")
    requested_post = _finite(requested_post_roll_s, "requested post-roll")
    maximum_roll = _finite(maximum_preserved_roll_s, "maximum preserved roll")
    if (
        requested_pre < 0.0
        or requested_post < 0.0
        or maximum_roll < 0.0
        or requested_pre > maximum_roll
        or requested_post > maximum_roll
    ):
        raise VideoTimelineError("requested context roll is outside its allowed maximum")

    pts_deltas = [right.pts_s - left.pts_s for left, right in zip(frames, frames[1:])]
    sim_deltas = [
        right.sim_time_s - left.sim_time_s for left, right in zip(ledger, ledger[1:])
    ]
    frame_period = 1.0 / float(expected_fps)
    cadence_tolerance = max(1.0e-6, frame_period * 1.0e-4)
    if any(abs(value - frame_period) > cadence_tolerance for value in pts_deltas):
        raise VideoTimelineError("decoded frame PTS cadence is not the expected native rate")
    if any(abs(value - frame_period) > cadence_tolerance for value in sim_deltas):
        raise VideoTimelineError("viewport ledger cadence is not the expected native rate")
    offsets = [row.sim_time_s - frame.pts_s for frame, row in zip(frames, ledger)]
    offset = statistics.median(offsets)
    maximum_offset_deviation = max(abs(value - offset) for value in offsets)
    if maximum_offset_deviation > cadence_tolerance:
        raise VideoTimelineError("sim-time to decoded-PTS offset is not constant")

    boundary_tolerance = cadence_tolerance
    if start < ledger[0].sim_time_s - frame_period - boundary_tolerance:
        raise VideoTimelineError("capture starts more than one frame after semantic action onset")
    if end > ledger[-1].sim_time_s + boundary_tolerance:
        raise VideoTimelineError("capture ends before the semantic action endpoint")

    action_first_index = next(
        (index for index, row in enumerate(ledger) if row.sim_time_s >= start - boundary_tolerance),
        len(ledger) - 1,
    )
    # If onset lies between renders, keep the immediately preceding frame so
    # that the visible transition cannot be clipped.
    if (
        action_first_index > 0
        and ledger[action_first_index].sim_time_s > start + boundary_tolerance
    ):
        action_first_index -= 1
    action_last_index = next(
        (index for index, row in enumerate(ledger) if row.sim_time_s >= end - boundary_tolerance),
        len(ledger) - 1,
    )
    available_pre = max(0.0, start - ledger[0].sim_time_s)
    available_post = max(0.0, ledger[-1].sim_time_s - end)
    clip_start_sim_s = (
        ledger[0].sim_time_s
        if available_pre <= maximum_roll + boundary_tolerance
        else start - requested_pre
    )
    clip_end_sim_s = (
        ledger[-1].sim_time_s
        if available_post <= maximum_roll + boundary_tolerance
        else end + requested_post
    )
    first_index = next(
        (
            index
            for index, row in enumerate(ledger)
            if row.sim_time_s >= clip_start_sim_s - boundary_tolerance
        ),
        len(ledger) - 1,
    )
    if (
        first_index > 0
        and ledger[first_index].sim_time_s > clip_start_sim_s + boundary_tolerance
    ):
        first_index -= 1
    last_index = next(
        (
            index
            for index, row in enumerate(ledger)
            if row.sim_time_s >= clip_end_sim_s - boundary_tolerance
        ),
        len(ledger) - 1,
    )
    first_index = min(first_index, action_first_index)
    last_index = max(last_index, action_last_index)
    if last_index < first_index:
        raise VideoTimelineError("semantic action does not select a positive frame interval")

    selected_frames = frames[first_index : last_index + 1]
    selected_ledger = ledger[first_index : last_index + 1]
    trim_start_pts = selected_frames[0].pts_s
    trim_end_pts = (
        frames[last_index + 1].pts_s
        if last_index + 1 < len(frames)
        else selected_frames[-1].pts_s + statistics.median(pts_deltas)
    )
    action_end_pts = (
        frames[action_last_index + 1].pts_s
        if action_last_index + 1 < len(frames)
        else frames[action_last_index].pts_s + statistics.median(pts_deltas)
    )
    selected_pts = [row.pts_s for row in selected_frames]
    selected_pts_deltas = [
        right - left for left, right in zip(selected_pts, selected_pts[1:])
    ]
    return ActionWindow(
        semantic_start_sim_s=start,
        semantic_end_sim_s=end,
        source_first_frame_index=first_index,
        source_last_frame_index=last_index,
        source_first_pts_s=selected_frames[0].pts_s,
        source_last_pts_s=selected_frames[-1].pts_s,
        source_first_sim_time_s=selected_ledger[0].sim_time_s,
        source_last_sim_time_s=selected_ledger[-1].sim_time_s,
        trim_start_pts_s=trim_start_pts,
        trim_end_pts_s=trim_end_pts,
        output_duration_s=trim_end_pts - trim_start_pts,
        expected_frame_count=len(selected_frames),
        leading_frames_removed=first_index,
        trailing_frames_removed=len(frames) - last_index - 1,
        phase_clock_origin_sim_s=selected_ledger[0].sim_time_s,
        semantic_start_output_s=start - selected_ledger[0].sim_time_s,
        semantic_end_output_s=end - selected_ledger[0].sim_time_s,
        action_frame_start_output_pts_s=(
            frames[action_first_index].pts_s - trim_start_pts
        ),
        action_frame_end_output_pts_s_exclusive=action_end_pts - trim_start_pts,
        requested_pre_roll_s=requested_pre,
        requested_post_roll_s=requested_post,
        maximum_preserved_roll_s=maximum_roll,
        available_pre_roll_s=available_pre,
        available_post_roll_s=available_post,
        retained_pre_roll_s=max(0.0, start - selected_ledger[0].sim_time_s),
        retained_post_roll_s=max(0.0, selected_ledger[-1].sim_time_s - end),
        capture_lag_after_action_start_s=max(0.0, selected_ledger[0].sim_time_s - start),
        packet_copy_start_is_keyframe=selected_frames[0].key_frame,
        source_selected_pts_sha256=_sha256_float64(selected_pts),
        source_selected_pts_delta_sha256=_sha256_float64(selected_pts_deltas),
        source_selected_checksums_sha256=_sha256_text_rows(
            row.checksum for row in selected_frames
        ),
        ledger_to_pts_offset_s=offset,
        maximum_ledger_to_pts_offset_deviation_s=maximum_offset_deviation,
    )


def build_action_window_plan(
    *,
    video_path: Path,
    ledger_path: Path,
    semantic_start_sim_s: float,
    semantic_end_sim_s: float,
    ffmpeg: Path | str | None = None,
    requested_pre_roll_s: float = 0.5,
    requested_post_roll_s: float = 0.5,
    maximum_preserved_roll_s: float = 1.0,
) -> tuple[ActionWindow, tuple[DecodedVideoFrame, ...]]:
    """Convenience wrapper that fully decodes and hash-binds both evidence files."""

    frames = decode_frame_timeline(video_path, ffmpeg=ffmpeg)
    ledger = load_viewport_frame_ledger(ledger_path)
    window = plan_action_window(
        frames,
        ledger,
        semantic_start_sim_s=semantic_start_sim_s,
        semantic_end_sim_s=semantic_end_sim_s,
        requested_pre_roll_s=requested_pre_roll_s,
        requested_post_roll_s=requested_post_roll_s,
        maximum_preserved_roll_s=maximum_preserved_roll_s,
    )
    # Touch both hashes here so callers cannot accidentally report a plan made
    # from paths different from the evidence they later publish.
    sha256_file(Path(video_path))
    sha256_file(Path(ledger_path))
    return window, frames


def verify_native_rate_output(
    output_frames: Sequence[DecodedVideoFrame],
    window: ActionWindow,
    *,
    require_decoded_frame_identity: bool,
    require_exact_pts_delta_identity: bool | None = None,
) -> dict[str, Any]:
    """Verify frame count/cadence, plus identity when publication was packet-copy."""

    rows = tuple(output_frames)
    if len(rows) != window.expected_frame_count:
        raise VideoTimelineError(
            f"published frame count {len(rows)} != selected {window.expected_frame_count}"
        )
    pts = [row.pts_s for row in rows]
    deltas = [right - left for left, right in zip(pts, pts[1:])]
    delta_hash = _sha256_float64(deltas)
    exact_delta_identity = delta_hash == window.source_selected_pts_delta_sha256
    require_exact_delta = (
        require_decoded_frame_identity
        if require_exact_pts_delta_identity is None
        else bool(require_exact_pts_delta_identity)
    )
    if require_exact_delta and not exact_delta_identity:
        raise VideoTimelineError("published frame cadence differs from selected source frames")
    frame_period = 1.0 / VIDEO_FPS
    cadence_tolerance = max(1.0e-6, frame_period * 1.0e-4)
    native_cadence = bool(
        deltas and all(abs(value - frame_period) <= cadence_tolerance for value in deltas)
    )
    if not native_cadence:
        raise VideoTimelineError("published output does not retain native 15 Hz cadence")
    checksum_hash = _sha256_text_rows(row.checksum for row in rows)
    decoded_identity = checksum_hash == window.source_selected_checksums_sha256
    if require_decoded_frame_identity and not decoded_identity:
        raise VideoTimelineError("packet-copy publication changed decoded source frames")
    return {
        "frame_count_matches_selected_source": True,
        "source_pts_delta_sha256": window.source_selected_pts_delta_sha256,
        "output_pts_delta_sha256": delta_hash,
        "exact_pts_delta_sequence_unchanged": exact_delta_identity,
        "native_frame_cadence_unchanged": True,
        "source_decoded_checksums_sha256": window.source_selected_checksums_sha256,
        "output_decoded_checksums_sha256": checksum_hash,
        "decoded_frames_unchanged": decoded_identity,
        "timestamps_rebased_only": True,
        "speed_modified": False,
    }
