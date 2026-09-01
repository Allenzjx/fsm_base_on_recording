"""Validate the frozen v010 reference before building or running the FSM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg

from wlr50_clean.reference.recording_parser import sha256_file, validate_v010


EXPECTED_ACCEPTED_SHA256 = (
    "f962128da9e9551235a6f6769308eed0c947657fe804c9e0e26025f456c72e92"
)
EXPECTED_VIDEO_SHA256 = (
    "50a82b5417f413114c4b9c4e81e0b41944ca0da6af2dfdb1f6fb0bc7bb023504"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate(project_root: Path) -> dict[str, Any]:
    reference = project_root / "reference" / "v010"
    recording_result = validate_v010(
        reference / "accepted_steps.jsonl",
        reference / "metadata.json",
        expected_sha256=EXPECTED_ACCEPTED_SHA256,
    )
    source_manifest = _load_json(reference / "source_manifest.json")
    video_validation = _load_json(reference / "video_validation.json")
    keyframe_manifest = _load_json(reference / "keyframes" / "manifest.json")
    video_path = reference / "recording_clean.mp4"

    errors: list[str] = list(recording_result["errors"])
    video_sha256 = sha256_file(video_path)
    if video_sha256 != EXPECTED_VIDEO_SHA256:
        errors.append("recording_clean.mp4 SHA-256 mismatch")
    if not video_validation.get("passed"):
        errors.append("video_validation.json is not passed")
    if video_validation.get("sha256") != video_sha256:
        errors.append("video validation is not bound to recording_clean.mp4")
    if video_validation.get("rear_leg_order") != "RR_FIRST":
        errors.append("reference rear order is not RR_FIRST")
    if video_validation.get("stitched") is not False:
        errors.append("reference video is marked stitched")
    if video_validation.get("speed_modified") is not False:
        errors.append("reference video is marked speed-modified")
    if video_validation.get("body_obstacle_collision") is not False:
        errors.append("reference body collision audit failed")
    if video_validation.get("wheel_only_climb") is not False:
        errors.append("reference wheel-only climb audit failed")
    if int(video_validation.get("decoded_frame_count", -1)) != 1345:
        errors.append("reference decoded frame count differs from 1345")
    if float(video_validation.get("duration_from_decoded_frames_s", 999.0)) > 200.0:
        errors.append("reference video duration exceeds 200 seconds")

    keyframes = keyframe_manifest.get("frames", [])
    if not isinstance(keyframes, list) or len(keyframes) < 13:
        errors.append("fewer than 13 required semantic keyframes")
    else:
        for item in keyframes:
            candidate = reference / "keyframes" / str(item.get("file", ""))
            if not candidate.is_file():
                errors.append(f"missing keyframe: {candidate.name}")

    manifest_files = source_manifest.get("files", [])
    accepted_bound = any(
        row.get("destination_sha256") == EXPECTED_ACCEPTED_SHA256
        for row in manifest_files
        if isinstance(row, dict)
    )
    video_bound = any(
        row.get("destination_sha256") == EXPECTED_VIDEO_SHA256
        for row in manifest_files
        if isinstance(row, dict)
    )
    if not accepted_bound or not video_bound:
        errors.append("source manifest does not bind the accepted stream and video")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    decode = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            os.devnull,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        errors.append(f"FFmpeg full decode failed: {decode.stderr.strip()}")

    return {
        "schema": "wlr50_clean.reference_validation.v1",
        "passed": not errors,
        "errors": errors,
        "recording": recording_result,
        "video": {
            "sha256": video_sha256,
            "decode_to_eof": decode.returncode == 0,
            "frame_count": video_validation.get("decoded_frame_count"),
            "duration_s": video_validation.get("duration_from_decoded_frames_s"),
            "rear_leg_order": video_validation.get("rear_leg_order"),
            "stitched": video_validation.get("stitched"),
            "speed_modified": video_validation.get("speed_modified"),
            "body_obstacle_collision": video_validation.get(
                "body_obstacle_collision"
            ),
            "wheel_only_climb": video_validation.get("wheel_only_climb"),
        },
        "semantic_keyframe_count": len(keyframes),
        "source_manifest_bound": accepted_bound and video_bound,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reference_validation.json")
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = project_root / output
    result = validate(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
