"""Publish and validate the four final v010/FSM video artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.video_capture import (
    MAX_VIDEO_DURATION_S,
    VIDEO_ERROR_CODE,
    VIDEO_FPS,
    VideoArtifactError,
    find_ffmpeg,
    sha256_file,
    validate_mp4,
)

from .comparison import (
    PHASE_IDS,
    AlignedPhase,
    PhaseWindow,
    align_phases,
    comparison_filter,
    diagnostic_filter,
    fsm_windows_from_evidence,
    reference_windows_from_contract,
)
from .video_timeline import (
    ActionWindow,
    DecodedVideoFrame,
    VideoTimelineError,
    build_action_window_plan,
    decode_frame_timeline,
    fsm_action_bounds,
    reference_action_bounds,
    verify_native_rate_output,
)


FINAL_RECORDING_RAW_NAME = "recording_v010_50mm_full_raw.mp4"
FINAL_RECORDING_NAME = "recording_v010_50mm_clean.mp4"
FINAL_FSM_NAME = "fsm_50mm_physical_success_clean.mp4"
FINAL_COMPARISON_NAME = "recording_vs_fsm_50mm.mp4"
FINAL_DIAGNOSTIC_NAME = "fsm_v010_shaped_50mm_diagnostic.mp4"
V010_RECORDING_SHA256 = "50a82b5417f413114c4b9c4e81e0b41944ca0da6af2dfdb1f6fb0bc7bb023504"
V010_RECORDING_RAW_SHA256 = "6b6d2561ef9c9c64f6eca5c009bbe0b4c0de57652b4faa2f83bf786cf2bc4fa3"
V010_RECORDING_LEDGER_SHA256 = "009de1a8e4b11b38e86ae9a88331e331ff5433299f0f93d42cbcc2be1a045f3c"
RECORDING_REQUESTED_PRE_ROLL_S = 0.5
RECORDING_REQUESTED_POST_ROLL_S = 0.5
FSM_REQUESTED_PRE_ROLL_S = 0.5
FSM_REQUESTED_POST_ROLL_S = 1.0
VIDEO_PROVENANCE_FIELDS = (
    "source_trial",
    "source_video",
    "start_sim_time",
    "end_sim_time",
    "pre_roll_s",
    "post_roll_s",
    "time_domain",
)


class VideoBuildError(VideoArtifactError):
    """A final video could not be produced from accepted continuous evidence."""


def _window_video_metadata(
    window: ActionWindow,
    *,
    source_label: str,
) -> dict[str, Any]:
    """Describe real source-edge context without inventing padding frames."""

    tolerance = 1.0e-9
    pre_limited = (
        window.retained_pre_roll_s + tolerance < window.requested_pre_roll_s
    )
    post_limited = (
        window.retained_post_roll_s + tolerance < window.requested_post_roll_s
    )
    reasons: list[str] = []
    if pre_limited:
        if window.capture_lag_after_action_start_s > tolerance:
            reasons.append(
                f"{source_label} begins {window.capture_lag_after_action_start_s:.6f} s "
                "after semantic action start, so genuine requested pre-roll is unavailable"
            )
        else:
            reasons.append(
                f"{source_label} contains only {window.retained_pre_roll_s:.6f} s of "
                "genuine pre-roll"
            )
    if post_limited:
        reasons.append(
            f"{source_label} contains only {window.retained_post_roll_s:.6f} s of "
            "genuine post-roll"
        )
    return {
        "time_domain": "source_simulation_time_s_bound_to_decoded_frame_pts",
        "start_sim_time": window.source_first_sim_time_s,
        "end_sim_time": window.source_last_sim_time_s + 1.0 / VIDEO_FPS,
        # Keep the original field names as retained-roll aliases for consumers
        # of the v1 validation record, while making requested/retained explicit.
        "pre_roll_s": window.retained_pre_roll_s,
        "post_roll_s": window.retained_post_roll_s,
        "requested_pre_roll_s": window.requested_pre_roll_s,
        "requested_post_roll_s": window.requested_post_roll_s,
        "available_pre_roll_s": window.available_pre_roll_s,
        "available_post_roll_s": window.available_post_roll_s,
        "retained_pre_roll_s": window.retained_pre_roll_s,
        "retained_post_roll_s": window.retained_post_roll_s,
        "source_limited": pre_limited or post_limited,
        "source_limited_pre_roll": pre_limited,
        "source_limited_post_roll": post_limited,
        "source_limitation_reason": (
            "; ".join(reasons) if reasons else "SOURCE_CONTEXT_SUFFICIENT"
        ),
        "edge_context_padding_applied": False,
        "edge_context_frames_duplicated": False,
    }


def _comparison_video_metadata(
    recording_window: ActionWindow,
    fsm_window: ActionWindow,
) -> dict[str, Any]:
    """Represent the two independent source clocks without nullable fields."""

    recording = _window_video_metadata(
        recording_window, source_label="immutable v010 Recording source"
    )
    fsm = _window_video_metadata(
        fsm_window, source_label="immutable selected FSM Trial source"
    )
    by_source = {"recording": recording, "fsm": fsm}

    def paired(name: str) -> dict[str, Any]:
        return {side: metadata[name] for side, metadata in by_source.items()}

    return {
        "time_domain": "paired_source_simulation_time_s_by_side",
        "start_sim_time": paired("start_sim_time"),
        "end_sim_time": paired("end_sim_time"),
        "pre_roll_s": paired("pre_roll_s"),
        "post_roll_s": paired("post_roll_s"),
        "requested_pre_roll_s": paired("requested_pre_roll_s"),
        "requested_post_roll_s": paired("requested_post_roll_s"),
        "available_pre_roll_s": paired("available_pre_roll_s"),
        "available_post_roll_s": paired("available_post_roll_s"),
        "retained_pre_roll_s": paired("retained_pre_roll_s"),
        "retained_post_roll_s": paired("retained_post_roll_s"),
        "source_limited": bool(
            recording["source_limited"] or fsm["source_limited"]
        ),
        "source_limited_pre_roll": bool(
            recording["source_limited_pre_roll"]
            or fsm["source_limited_pre_roll"]
        ),
        "source_limited_post_roll": bool(
            recording["source_limited_post_roll"]
            or fsm["source_limited_post_roll"]
        ),
        "source_limitation_reason": paired("source_limitation_reason"),
        "edge_context_padding_applied": False,
        "edge_context_frames_duplicated": False,
    }


def _require_complete_video_metadata(videos: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail publication when provenance fields are absent or contain nulls."""

    def contains_null(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, Mapping):
            return not value or any(contains_null(item) for item in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return not value or any(contains_null(item) for item in value)
        return False

    missing: dict[str, list[str]] = {}
    for video_name, record in videos.items():
        invalid = [
            field
            for field in VIDEO_PROVENANCE_FIELDS
            if field not in record or contains_null(record[field])
        ]
        if invalid:
            missing[str(video_name)] = invalid
    if missing:
        raise VideoBuildError(
            f"video validation provenance fields are incomplete: {missing}"
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.stem + ".partial" + destination.suffix)
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise VideoBuildError("copied video SHA-256 differs from its single source")
    os.replace(temporary, destination)


def _lookup(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _load_reclassification(
    value: Path | Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None]:
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        return value, {"kind": "in_memory_analysis", "sha256": None, "path": None}
    path = Path(value).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise VideoBuildError("physical reclassification is not an object")
    return payload, {"kind": "external_json", "sha256": sha256_file(path), "path": str(path)}


def validate_successful_trial(
    run_dir: Path,
    manifest_path: Path | None = None,
    *,
    physical_reclassification: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require fail-closed physical success without mutating an old manifest.

    Historical immutable manifests may contain a terminal classification made
    by an obsolete acceptance layer.  A fresh analysis can authoritatively
    reclassify that terminal result, while the manifest remains authoritative
    for immutable run identity, safety counters, phase times, and video hashes.
    """

    root = Path(run_dir).resolve()
    trial_path = Path(manifest_path).resolve() if manifest_path else root / "trial_manifest.json"
    if not trial_path.is_file():
        raise VideoBuildError(f"successful trial manifest is missing: {trial_path}")
    manifest = json.loads(trial_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise VideoBuildError("trial manifest is not an object")
    evidence = manifest.get("success_evidence", manifest)
    if not isinstance(evidence, Mapping):
        raise VideoBuildError("success_evidence is not an object")
    reclassification, reclassification_source = _load_reclassification(
        physical_reclassification
    )
    reclassified = reclassification is not None
    external_checks: Mapping[str, Any] = {}
    if reclassification is not None:
        declared_run = reclassification.get("run_dir")
        if declared_run is not None and Path(str(declared_run)).resolve() != root:
            raise VideoBuildError("physical reclassification is bound to a different run")
        declared_manifest_hash = reclassification.get("trial_manifest_sha256")
        if declared_manifest_hash is not None and declared_manifest_hash != sha256_file(trial_path):
            raise VideoBuildError("physical reclassification manifest hash does not match")
        layers = reclassification.get("result_layers")
        if isinstance(layers, Mapping):
            validity = layers.get("trial_validity")
            task = layers.get("task_success")
            if not isinstance(validity, Mapping) or not isinstance(task, Mapping):
                raise VideoBuildError("physical reclassification layers are incomplete")
            external_checks = task.get("checks", {})
            if not isinstance(external_checks, Mapping):
                raise VideoBuildError("physical reclassification task checks are invalid")
            classification_checks = {
                "reclassified_trial_valid": validity.get("result") == "VALID",
                "reclassified_task_success": task.get("result") == "SUCCESS",
            }
        else:
            # ``physical_success.py`` publishes either this wrapper or the
            # selected row directly.  Both are immutable external selection
            # evidence; neither requires rewriting the historical Trial.
            selected = reclassification.get(
                "selected_success_trial", reclassification
            )
            if not isinstance(selected, Mapping):
                raise VideoBuildError("physical reclassification selected trial is missing")
            selected_video = Path(str(selected.get("video_path", "")))
            if not selected_video.is_absolute():
                selected_video = root / selected_video
            external_checks = {
                "p01_p13_completed": selected.get("P01_P13_complete") is True,
                "body_collision_false": selected.get("body_collision") is False,
                "wheel_only_climb_false": selected.get("wheel_only_climb") is False,
                "rear_order_rr_first": selected.get("rear_leg_order") == "RR_FIRST",
            }
            classification_checks = {
                "reclassified_trial_valid": selected.get("trial_validity") == "VALID",
                "reclassified_task_success": selected.get("task_result") == "SUCCESS",
                "selected_trial_matches_run": selected.get("trial_id") == root.name,
                "selected_video_matches_run": selected_video.resolve()
                == (root / "actual_viewport_video.mp4").resolve(),
                "selected_video_continuous": selected.get("video_continuous") is True,
                "selected_no_forbidden_control": selected.get(
                    "forbidden_control_count"
                )
                == 0,
            }
    else:
        classification_checks = {
            "task_result_success": _lookup(evidence, "task_result", "result") == "SUCCESS",
            "continuous_success_declared": _lookup(
                evidence, "one_continuous_physical_fsm_success", "continuous_physical_run"
            ) is True,
        }
    checks = {
        **classification_checks,
        "body_collision_false": _lookup(
            evidence, "body_collision", "task_failure_body_collision"
        ) is False and (not reclassified or external_checks.get("body_collision_false") is True),
        "wheel_only_climb_false": _lookup(
            evidence, "wheel_only_climb", "task_failure_wheel_only_climb"
        ) is False and (not reclassified or external_checks.get("wheel_only_climb_false") is True),
        "rear_order_rr_first": _lookup(evidence, "rear_leg_order", "rear_order") == "RR_FIRST"
        and (not reclassified or external_checks.get("rear_order_rr_first") is True),
        "no_root_state_write": int(_lookup(evidence, "root_state_write_count") or 0) == 0
        and "root_state_write_count" in evidence,
        "no_teleport": int(_lookup(evidence, "teleport_count") or 0) == 0
        and "teleport_count" in evidence,
        "no_external_force_or_impulse": int(_lookup(evidence, "external_force_count") or 0) == 0
        and int(_lookup(evidence, "external_impulse_count") or 0) == 0
        and "external_force_count" in evidence and "external_impulse_count" in evidence,
    }
    phases = _lookup(evidence, "completed_macro_phases", "completed_phases")
    checks["p01_p13_completed"] = (
        tuple(phases) == PHASE_IDS if isinstance(phases, (list, tuple))
        else _lookup(evidence, "p01_p13_completed") is True
    ) and (not reclassified or external_checks.get("p01_p13_completed") is True)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise VideoBuildError(f"trial is not eligible for final video: failed checks {failed}")
    video_path = root / "actual_viewport_video.mp4"
    recorder_manifest_path = root / "viewport_buffer_video_manifest.json"
    if not recorder_manifest_path.is_file():
        raise VideoBuildError("active viewport recorder manifest is missing")
    recorder = json.loads(recorder_manifest_path.read_text(encoding="utf-8"))
    if recorder.get("valid") is not True or recorder.get("status") != "PASS":
        raise VideoBuildError("active viewport recorder manifest is invalid")
    if Path(str(recorder.get("video_path", ""))).resolve() != video_path:
        raise VideoBuildError("recorder manifest is not bound to this trial video")
    if not video_path.is_file() or recorder.get("video_sha256") != sha256_file(video_path):
        raise VideoBuildError("trial video is missing or differs from its recorder manifest")
    return {
        "manifest_path": str(trial_path),
        "manifest_sha256": sha256_file(trial_path),
        "recorder_manifest_path": str(recorder_manifest_path),
        "recorder_manifest_sha256": sha256_file(recorder_manifest_path),
        "video_path": str(video_path),
        "video_sha256": sha256_file(video_path),
        "checks": checks,
        "classification_source": (
            "external_physical_reclassification"
            if reclassified
            else "immutable_trial_manifest_legacy_success"
        ),
        "physical_reclassification": reclassification_source,
    }


def _run_ffmpeg(command: Sequence[str], *, label: str) -> None:
    completed = subprocess.run(list(command), capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        tail = completed.stderr[-3000:].replace("\r", " ").replace("\n", " ")
        raise VideoBuildError(f"{label} ffmpeg failed ({completed.returncode}): {tail}")


_FFMPEG_SHA256_RE = re.compile(r"^SHA256=([0-9a-f]{64})$", re.IGNORECASE | re.MULTILINE)


def _h264_elementary_stream_sha256(ffmpeg: Path, source: Path) -> str:
    """Hash packet-copied Annex-B H.264, independent of MP4 container atoms."""

    completed = subprocess.run(
        [
            str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error",
            "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-c:v", "copy", "-bsf:v", "h264_mp4toannexb",
            "-f", "hash", "-hash", "sha256", "-",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    match = _FFMPEG_SHA256_RE.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        tail = (completed.stderr or completed.stdout)[-3000:].replace("\r", " ").replace("\n", " ")
        raise VideoBuildError(
            f"H.264 elementary-stream hash failed ({completed.returncode}): {tail}"
        )
    return match.group(1).lower()


def _h264_action_window_sha256(
    ffmpeg: Path,
    source: Path,
    window: ActionWindow,
) -> str:
    """Hash the exact packet-selected Annex-B stream for an action window."""

    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error"]
    if window.trim_start_pts_s > 1.0e-9:
        command.extend(["-ss", f"{window.trim_start_pts_s:.9f}"])
    command.extend(
        [
            "-i", str(source), "-t", f"{window.output_duration_s:.9f}",
            "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
            "-bsf:v", "h264_mp4toannexb", "-f", "hash", "-hash", "sha256", "-",
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    match = _FFMPEG_SHA256_RE.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        tail = (completed.stderr or completed.stdout)[-3000:].replace("\r", " ").replace("\n", " ")
        raise VideoBuildError(
            f"selected H.264 action-window hash failed ({completed.returncode}): {tail}"
        )
    return match.group(1).lower()


def _atomic_packet_copy_remux(
    ffmpeg: Path,
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    """Repair MP4 atoms without re-encoding, retiming, or joining sources."""

    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    source_elementary_sha256 = _h264_elementary_stream_sha256(ffmpeg, source)
    try:
        _run_ffmpeg(
            [
                str(ffmpeg), "-hide_banner", "-nostdin", "-y",
                "-copyts", "-start_at_zero", "-i", str(source),
                "-map", "0:v:0", "-map_metadata", "0", "-an", "-sn", "-dn",
                "-c:v", "copy", "-movflags", "+faststart", str(partial),
            ],
            label="FSM packet-copy remux",
        )
        output_elementary_sha256 = _h264_elementary_stream_sha256(ffmpeg, partial)
        if output_elementary_sha256 != source_elementary_sha256:
            raise VideoBuildError("packet-copy remux changed the H.264 elementary stream")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "performed": True,
        "method": "ffmpeg_single_input_single_video_packet_copy",
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "h264_elementary_stream_sha256": source_elementary_sha256,
        "elementary_stream_unchanged": True,
        "timestamps_unchanged": False,
        "decoded_frames_unchanged": False,
        "stitched": False,
        "speed_modified": False,
    }


def _packet_copy_window_command(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    window: ActionWindow,
) -> list[str]:
    """Build a one-input packet-copy cut; exactness is verified after decode."""

    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-y"]
    if window.trim_start_pts_s > 1.0e-9:
        command.extend(["-ss", f"{window.trim_start_pts_s:.9f}"])
    command.extend(
        [
            "-i", str(source),
            "-t", f"{window.output_duration_s:.9f}",
            "-map", "0:v:0", "-map_metadata", "0", "-an", "-sn", "-dn",
            "-c:v", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", str(destination),
        ]
    )
    return command


def _frame_exact_encode_command(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    window: ActionWindow,
) -> list[str]:
    """Build a one-input frame cut with timestamp rebasing but no time scaling."""

    frame_filter = (
        f"trim=start_frame={window.source_first_frame_index}:"
        f"end_frame={window.source_last_frame_index + 1},setpts=PTS-STARTPTS"
    )
    return [
        str(ffmpeg), "-hide_banner", "-nostdin", "-y", "-i", str(source),
        "-map", "0:v:0", "-vf", frame_filter, "-an", "-sn", "-dn",
        "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(destination),
    ]


def _atomic_publish_action_window(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    window: ActionWindow,
    source_frames: Sequence[DecodedVideoFrame],
    *,
    source_container_duration_valid: bool,
) -> dict[str, Any]:
    """Publish exactly one continuous semantic action interval.

    Packet copy is used for a full source or a cut beginning at a keyframe.  A
    non-keyframe onset is necessarily decoded and re-encoded to make an exact,
    independently decodable cut.  That fallback selects frames by index and
    only subtracts the initial PTS; it contains no time scaling or stitching.
    """

    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    packet_copy_attempted = False
    packet_copy_error = ""
    publication: dict[str, Any]
    try:
        if window.is_full_source and source_container_duration_valid:
            _atomic_copy(source, destination)
            publication = {
                "performed": False,
                "method": "byte_exact_copy",
                "source_sha256": sha256_file(source),
                "output_sha256": sha256_file(destination),
                "elementary_stream_unchanged": True,
                "timestamps_rebased_only": False,
                "stitched": False,
                "speed_modified": False,
            }
        elif window.is_full_source:
            publication = _atomic_packet_copy_remux(ffmpeg, source, destination)
        else:
            if window.packet_copy_eligible:
                packet_copy_attempted = True
                try:
                    _run_ffmpeg(
                        _packet_copy_window_command(ffmpeg, source, partial, window),
                        label="action-window packet copy",
                    )
                    packet_rows = decode_frame_timeline(partial, ffmpeg=ffmpeg)
                    packet_check = verify_native_rate_output(
                        packet_rows,
                        window,
                        require_decoded_frame_identity=True,
                        require_exact_pts_delta_identity=True,
                    )
                    selected_elementary_sha256 = _h264_action_window_sha256(
                        ffmpeg, source, window
                    )
                    output_elementary_sha256 = _h264_elementary_stream_sha256(
                        ffmpeg, partial
                    )
                    if output_elementary_sha256 != selected_elementary_sha256:
                        raise VideoBuildError(
                            "packet-copy action window changed selected H.264 packets"
                        )
                    os.replace(partial, destination)
                    publication = {
                        "performed": True,
                        "method": "ffmpeg_single_input_action_window_packet_copy",
                        "source_sha256": sha256_file(source),
                        "output_sha256": sha256_file(destination),
                        "selected_source_h264_elementary_stream_sha256": (
                            selected_elementary_sha256
                        ),
                        "output_h264_elementary_stream_sha256": output_elementary_sha256,
                        "elementary_stream_unchanged": True,
                        "stitched": False,
                        "speed_modified": False,
                        **packet_check,
                    }
                except (VideoBuildError, VideoTimelineError) as exc:
                    packet_copy_error = f"{type(exc).__name__}: {exc}"
                    partial.unlink(missing_ok=True)
            if not window.packet_copy_eligible or packet_copy_error:
                _run_ffmpeg(
                    _frame_exact_encode_command(ffmpeg, source, partial, window),
                    label="frame-exact action-window encode",
                )
                encoded_rows = decode_frame_timeline(partial, ffmpeg=ffmpeg)
                encoded_check = verify_native_rate_output(
                    encoded_rows,
                    window,
                    require_decoded_frame_identity=False,
                    require_exact_pts_delta_identity=False,
                )
                os.replace(partial, destination)
                publication = {
                    "performed": True,
                    "method": "ffmpeg_single_input_frame_exact_h264_encode",
                    "source_sha256": sha256_file(source),
                    "output_sha256": sha256_file(destination),
                    "reason_reencode_required": (
                        "selected action onset is not an independently decodable keyframe"
                        if not window.packet_copy_eligible
                        else "packet-copy result failed exact retained-frame verification"
                    ),
                    "packet_copy_attempted": packet_copy_attempted,
                    "packet_copy_error": packet_copy_error,
                    "frame_selection_filter": (
                        f"trim=start_frame={window.source_first_frame_index}:"
                        f"end_frame={window.source_last_frame_index + 1}"
                    ),
                    "timestamp_transform": "PTS-STARTPTS",
                    "time_scale_transform": None,
                    "elementary_stream_unchanged": False,
                    "stitched": False,
                    "speed_modified": False,
                    **encoded_check,
                }

        published_rows = decode_frame_timeline(destination, ffmpeg=ffmpeg)
        exact_identity = publication["method"] != "ffmpeg_single_input_frame_exact_h264_encode"
        publication.update(
            verify_native_rate_output(
                published_rows,
                window,
                require_decoded_frame_identity=exact_identity,
                require_exact_pts_delta_identity=exact_identity,
            )
        )
        publication["source_frame_range"] = [
            window.source_first_frame_index,
            window.source_last_frame_index,
        ]
        publication["source_frame_count"] = window.expected_frame_count
        publication["single_contiguous_source"] = True
        return publication
    finally:
        partial.unlink(missing_ok=True)


def _encode_comparison(
    ffmpeg: Path,
    reference: Path,
    fsm: Path,
    destination: Path,
    aligned: Sequence[AlignedPhase],
    *,
    fsm_trial_id: str,
) -> None:
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    script = destination.with_name(destination.stem + ".filter.txt")
    script.write_text(
        comparison_filter(
            aligned,
            fps=VIDEO_FPS,
            fsm_trial_id=fsm_trial_id,
            contract_label="30% diagnostic",
        ),
        encoding="utf-8",
    )
    try:
        _run_ffmpeg(
            [
                str(ffmpeg), "-hide_banner", "-nostdin", "-y",
                "-i", str(reference), "-i", str(fsm),
                "-filter_complex_script", str(script), "-map", "[outv]", "-an",
                "-r", f"{VIDEO_FPS:g}", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
            ],
            label="semantic comparison",
        )
        os.replace(partial, destination)
    finally:
        script.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def _encode_diagnostic(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    windows: Sequence[PhaseWindow],
) -> None:
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    _run_ffmpeg(
        [
            str(ffmpeg), "-hide_banner", "-nostdin", "-y", "-i", str(source),
            "-vf", diagnostic_filter(windows), "-map", "0:v:0", "-an",
            "-r", f"{VIDEO_FPS:g}", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
        ],
        label="diagnostic overlay",
    )
    os.replace(partial, destination)


class FinalVideoBuilder:
    """Build only from the selected v010 and one accepted physical FSM trial."""

    def __init__(self, output_dir: Path, *, ffmpeg: Path | str | None = None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = find_ffmpeg(ffmpeg)
        self.validation_path = self.output_dir / "video_validation.json"
        self.checksum_path = self.output_dir / "checksums.sha256"

    def build(
        self,
        *,
        reference_video: Path,
        reference_raw_video: Path | None = None,
        reference_frame_ledger: Path,
        reference_contract: Path,
        successful_trial_dir: Path,
        trial_manifest: Path | None = None,
        physical_reclassification: Path | Mapping[str, Any] | None = None,
        fsm_phase_evidence: Path | None = None,
    ) -> dict[str, Any]:
        validation: dict[str, Any] = {
            "schema": "wlr50_clean.final_videos.v2",
            "status": VIDEO_ERROR_CODE,
            "valid": False,
            "selected_reference": "v010_20260806_220745_363972_manual",
            "rear_leg_order": "RR_FIRST",
            "videos": {},
            "error": "",
        }
        try:
            source_reference = Path(reference_video).resolve()
            source_reference_validation = validate_mp4(
                source_reference, ffmpeg=self.ffmpeg, stitched=False, speed_modified=False
            )
            if source_reference_validation.get("valid") is not True:
                raise VideoBuildError("selected v010 Recording video failed validation")
            if source_reference_validation.get("sha256") != V010_RECORDING_SHA256:
                raise VideoBuildError("Recording video is not the hash-locked clean v010 reference")
            source_reference_raw = (
                Path(reference_raw_video).resolve()
                if reference_raw_video is not None
                else source_reference
            )
            source_reference_raw_validation = validate_mp4(
                source_reference_raw,
                ffmpeg=self.ffmpeg,
                stitched=False,
                speed_modified=False,
                require_sane_container_duration=False,
                maximum_duration_s=4.0 * 60.0 * 60.0,
            )
            if source_reference_raw_validation.get("valid") is not True:
                raise VideoBuildError("raw v010 Recording video failed full-stream validation")
            expected_raw_sha256 = (
                V010_RECORDING_RAW_SHA256
                if reference_raw_video is not None
                else V010_RECORDING_SHA256
            )
            if source_reference_raw_validation.get("sha256") != expected_raw_sha256:
                raise VideoBuildError("raw Recording video is not the hash-locked v010 source")
            source_reference_ledger = Path(reference_frame_ledger).resolve()
            if sha256_file(source_reference_ledger) != V010_RECORDING_LEDGER_SHA256:
                raise VideoBuildError("Recording frame ledger is not the hash-locked v010 ledger")
            trial = validate_successful_trial(
                Path(successful_trial_dir),
                trial_manifest,
                physical_reclassification=physical_reclassification,
            )
            source_fsm = Path(trial["video_path"])
            source_fsm_validation = validate_mp4(
                source_fsm,
                ffmpeg=self.ffmpeg,
                stitched=False,
                speed_modified=False,
                require_sane_container_duration=False,
            )
            if source_fsm_validation.get("valid") is not True:
                raise VideoBuildError("successful FSM viewport video failed validation")

            trial_root = Path(successful_trial_dir).resolve()
            recorder_payload = json.loads(
                Path(trial["recorder_manifest_path"]).read_text(encoding="utf-8")
            )
            declared_ledger = Path(str(recorder_payload.get("ledger_path", "")))
            if not declared_ledger.is_absolute():
                declared_ledger = trial_root / declared_ledger
            source_fsm_ledger = (trial_root / "viewport_frame_ledger.jsonl").resolve()
            if declared_ledger.resolve() != source_fsm_ledger:
                raise VideoBuildError("FSM recorder manifest is bound to a different frame ledger")
            if (
                not source_fsm_ledger.is_file()
                or recorder_payload.get("ledger_sha256") != sha256_file(source_fsm_ledger)
            ):
                raise VideoBuildError("FSM viewport frame ledger is missing or hash-mismatched")

            reference_bounds = reference_action_bounds(Path(reference_contract))
            fsm_bounds = fsm_action_bounds(Path(trial["manifest_path"]))
            reference_action_window, reference_source_frames = build_action_window_plan(
                video_path=source_reference,
                ledger_path=source_reference_ledger,
                semantic_start_sim_s=reference_bounds[0],
                semantic_end_sim_s=reference_bounds[1],
                ffmpeg=self.ffmpeg,
                requested_pre_roll_s=RECORDING_REQUESTED_PRE_ROLL_S,
                requested_post_roll_s=RECORDING_REQUESTED_POST_ROLL_S,
            )
            fsm_action_window, fsm_source_frames = build_action_window_plan(
                video_path=source_fsm,
                ledger_path=source_fsm_ledger,
                semantic_start_sim_s=fsm_bounds[0],
                semantic_end_sim_s=fsm_bounds[1],
                ffmpeg=self.ffmpeg,
                requested_pre_roll_s=FSM_REQUESTED_PRE_ROLL_S,
                requested_post_roll_s=FSM_REQUESTED_POST_ROLL_S,
            )

            recording_raw_out = self.output_dir / FINAL_RECORDING_RAW_NAME
            recording_out = self.output_dir / FINAL_RECORDING_NAME
            fsm_out = self.output_dir / FINAL_FSM_NAME
            comparison_out = self.output_dir / FINAL_COMPARISON_NAME
            diagnostic_out = self.output_dir / FINAL_DIAGNOSTIC_NAME
            _atomic_copy(source_reference_raw, recording_raw_out)
            recording_publication = _atomic_publish_action_window(
                self.ffmpeg,
                source_reference,
                recording_out,
                reference_action_window,
                reference_source_frames,
                source_container_duration_valid=bool(
                    source_reference_validation.get("container_duration_valid")
                ),
            )
            fsm_publication = _atomic_publish_action_window(
                self.ffmpeg,
                source_fsm,
                fsm_out,
                fsm_action_window,
                fsm_source_frames,
                source_container_duration_valid=bool(
                    source_fsm_validation.get("container_duration_valid")
                ),
            )
            recording_clean_validation = validate_mp4(
                recording_out,
                ffmpeg=self.ffmpeg,
                expected_frame_count=reference_action_window.expected_frame_count,
                stitched=False,
                speed_modified=False,
            )
            fsm_clean_validation = validate_mp4(
                fsm_out,
                ffmpeg=self.ffmpeg,
                expected_frame_count=fsm_action_window.expected_frame_count,
                stitched=False,
                speed_modified=False,
            )
            if recording_clean_validation.get("valid") is not True:
                raise VideoBuildError("clean Recording action window failed validation")
            if fsm_clean_validation.get("valid") is not True:
                raise VideoBuildError("clean FSM action window failed validation")

            reference_windows = reference_windows_from_contract(
                Path(reference_contract),
                video_duration_s=float(recording_clean_validation["duration_s"]),
                video_origin_sim_s=reference_action_window.phase_clock_origin_sim_s,
                action_start_video_s=(
                    reference_action_window.action_frame_start_output_pts_s
                ),
                action_end_video_s=(
                    reference_action_window.action_frame_end_output_pts_s_exclusive
                ),
            )
            if fsm_phase_evidence:
                phase_source = Path(fsm_phase_evidence).resolve()
            else:
                trial_payload = json.loads(Path(trial["manifest_path"]).read_text(encoding="utf-8"))
                nested = trial_payload.get("success_evidence", {})
                has_windows = "phase_windows" in trial_payload or (
                    isinstance(nested, Mapping) and "phase_windows" in nested
                )
                phase_source = (
                    Path(trial["manifest_path"])
                    if has_windows
                    else Path(successful_trial_dir).resolve() / "state_transitions.jsonl"
                )
            fsm_windows = fsm_windows_from_evidence(
                phase_source,
                video_duration_s=float(fsm_clean_validation["duration_s"]),
                video_origin_sim_s=fsm_action_window.phase_clock_origin_sim_s,
                action_start_video_s=fsm_action_window.action_frame_start_output_pts_s,
                action_end_video_s=(
                    fsm_action_window.action_frame_end_output_pts_s_exclusive
                ),
            )
            aligned = align_phases(reference_windows, fsm_windows)
            _encode_comparison(
                self.ffmpeg,
                recording_out,
                fsm_out,
                comparison_out,
                aligned,
                fsm_trial_id=trial_root.name,
            )
            _encode_diagnostic(self.ffmpeg, fsm_out, diagnostic_out, fsm_windows)

            destinations = {
                "recording_full_raw": recording_raw_out,
                "recording": recording_out,
                "fsm": fsm_out,
                "comparison": comparison_out,
                "diagnostic": diagnostic_out,
            }
            validation["videos"]["recording_full_raw"] = {
                **source_reference_raw_validation,
                "path": str(recording_raw_out),
                "sha256": sha256_file(recording_raw_out),
                "source_trial": "v010_20260806_220745_363972_manual",
                "source_video": str(source_reference_raw),
                **_window_video_metadata(
                    reference_action_window,
                    source_label="immutable v010 Recording source",
                ),
                "artifact_role": "immutable_raw_process_video_evidence",
            }
            validation["videos"]["recording"] = {
                **recording_clean_validation,
                "source_trial": "v010_20260806_220745_363972_manual",
                "source_video": str(source_reference),
                **_window_video_metadata(
                    reference_action_window,
                    source_label="immutable v010 Recording source",
                ),
            }
            validation["videos"]["fsm"] = {
                **fsm_clean_validation,
                "source_trial": trial_root.name,
                "source_video": str(source_fsm),
                **_window_video_metadata(
                    fsm_action_window,
                    source_label="immutable selected FSM Trial source",
                ),
            }
            for name in ("comparison", "diagnostic"):
                validation["videos"][name] = validate_mp4(
                    destinations[name], ffmpeg=self.ffmpeg, stitched=False, speed_modified=False
                )
            validation["videos"]["comparison"].update(
                source_trial=[
                    "v010_20260806_220745_363972_manual",
                    trial_root.name,
                ],
                source_video=[str(recording_out), str(fsm_out)],
                **_comparison_video_metadata(
                    reference_action_window, fsm_action_window
                ),
                source_sim_time_ranges={
                    "recording": [reference_bounds[0], reference_bounds[1]],
                    "fsm": [fsm_bounds[0], fsm_bounds[1]],
                },
            )
            validation["videos"]["diagnostic"].update(
                source_trial=trial_root.name,
                source_video=str(fsm_out),
                **_window_video_metadata(
                    fsm_action_window,
                    source_label="immutable selected FSM Trial source",
                ),
            )
            _require_complete_video_metadata(validation["videos"])
            if any(row.get("valid") is not True for row in validation["videos"].values()):
                raise VideoBuildError("one or more published final videos failed validation")
            if sha256_file(recording_raw_out) != source_reference_raw_validation["sha256"]:
                raise VideoBuildError("published raw Recording is not byte-exact v010")

            validation.update(
                status="PASS",
                valid=True,
                error="",
                successful_trial=trial,
                source_validation={
                    "recording_raw": source_reference_raw_validation,
                    "recording": source_reference_validation,
                    "fsm": source_fsm_validation,
                },
                action_windows={
                    "recording": {
                        **reference_action_window.as_dict(),
                        "source_video_sha256": source_reference_validation["sha256"],
                        "frame_ledger_path": str(source_reference_ledger),
                        "frame_ledger_sha256": sha256_file(source_reference_ledger),
                        "publication": recording_publication,
                    },
                    "fsm": {
                        **fsm_action_window.as_dict(),
                        "source_video_sha256": source_fsm_validation["sha256"],
                        "frame_ledger_path": str(source_fsm_ledger),
                        "frame_ledger_sha256": sha256_file(source_fsm_ledger),
                        "publication": fsm_publication,
                    },
                },
                semantic_alignment={
                    "phases": [row.as_dict() for row in aligned],
                    "speed_modified": False,
                    "multi_trial_splice": False,
                    "shorter_phase_policy": "hold_last_frame",
                    "output_duration_s": sum(row.output_duration_s for row in aligned),
                },
                diagnostic_overlay={
                    "small_top_band_only": True,
                    "source_trial_count": 1,
                    "speed_modified": False,
                },
                recording_publication=recording_publication,
                fsm_publication=fsm_publication,
            )
            checksums = "".join(
                f"{sha256_file(path)}  {path.name}\n" for path in destinations.values()
            )
            self.checksum_path.write_text(checksums, encoding="ascii", newline="\n")
            validation["checksums_sha256"] = sha256_file(self.checksum_path)
        except Exception as exc:
            validation["status"] = VIDEO_ERROR_CODE
            validation["valid"] = False
            validation["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_json(self.validation_path, validation)
            if isinstance(exc, VideoBuildError):
                raise
            raise VideoBuildError(validation["error"]) from exc
        _atomic_json(self.validation_path, validation)
        return validation


def build_final_videos(**kwargs: Any) -> dict[str, Any]:
    """Convenience entry point; ``output_dir`` and build arguments are keyword-only."""

    output_dir = kwargs.pop("output_dir")
    ffmpeg = kwargs.pop("ffmpeg", None)
    return FinalVideoBuilder(output_dir, ffmpeg=ffmpeg).build(**kwargs)
