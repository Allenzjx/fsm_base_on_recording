"""Publish and validate the four final v010/FSM video artifacts."""

from __future__ import annotations

import json
import os
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


FINAL_RECORDING_NAME = "recording_v010_50mm_clean.mp4"
FINAL_FSM_NAME = "fsm_v010_shaped_50mm_clean.mp4"
FINAL_COMPARISON_NAME = "recording_vs_fsm_v010_50mm.mp4"
FINAL_DIAGNOSTIC_NAME = "fsm_v010_shaped_50mm_diagnostic.mp4"
V010_RECORDING_SHA256 = "50a82b5417f413114c4b9c4e81e0b41944ca0da6af2dfdb1f6fb0bc7bb023504"


class VideoBuildError(VideoArtifactError):
    """A final video could not be produced from accepted continuous evidence."""


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


def validate_successful_trial(run_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Require explicit, fail-closed proof of one continuous physical success."""

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
    checks = {
        "task_result_success": _lookup(evidence, "task_result", "result") == "SUCCESS",
        "continuous_success_declared": _lookup(
            evidence, "one_continuous_physical_fsm_success", "continuous_physical_run"
        ) is True,
        "body_collision_false": _lookup(
            evidence, "body_collision", "task_failure_body_collision"
        ) is False,
        "wheel_only_climb_false": _lookup(
            evidence, "wheel_only_climb", "task_failure_wheel_only_climb"
        ) is False,
        "rear_order_rr_first": _lookup(evidence, "rear_leg_order", "rear_order") == "RR_FIRST",
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
    )
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
    }


def _run_ffmpeg(command: Sequence[str], *, label: str) -> None:
    completed = subprocess.run(list(command), capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        tail = completed.stderr[-3000:].replace("\r", " ").replace("\n", " ")
        raise VideoBuildError(f"{label} ffmpeg failed ({completed.returncode}): {tail}")


def _encode_comparison(
    ffmpeg: Path,
    reference: Path,
    fsm: Path,
    destination: Path,
    aligned: Sequence[AlignedPhase],
) -> None:
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    script = destination.with_name(destination.stem + ".filter.txt")
    script.write_text(comparison_filter(aligned, fps=VIDEO_FPS), encoding="utf-8")
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
        reference_contract: Path,
        successful_trial_dir: Path,
        trial_manifest: Path | None = None,
        fsm_phase_evidence: Path | None = None,
    ) -> dict[str, Any]:
        validation: dict[str, Any] = {
            "schema": "wlr50_clean.final_videos.v1",
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
            trial = validate_successful_trial(Path(successful_trial_dir), trial_manifest)
            source_fsm = Path(trial["video_path"])
            source_fsm_validation = validate_mp4(
                source_fsm, ffmpeg=self.ffmpeg, stitched=False, speed_modified=False
            )
            if source_fsm_validation.get("valid") is not True:
                raise VideoBuildError("successful FSM viewport video failed validation")

            recording_out = self.output_dir / FINAL_RECORDING_NAME
            fsm_out = self.output_dir / FINAL_FSM_NAME
            comparison_out = self.output_dir / FINAL_COMPARISON_NAME
            diagnostic_out = self.output_dir / FINAL_DIAGNOSTIC_NAME
            _atomic_copy(source_reference, recording_out)
            _atomic_copy(source_fsm, fsm_out)

            reference_windows = reference_windows_from_contract(
                Path(reference_contract), video_duration_s=float(source_reference_validation["duration_s"])
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
                phase_source, video_duration_s=float(source_fsm_validation["duration_s"])
            )
            aligned = align_phases(reference_windows, fsm_windows)
            _encode_comparison(self.ffmpeg, recording_out, fsm_out, comparison_out, aligned)
            _encode_diagnostic(self.ffmpeg, fsm_out, diagnostic_out, fsm_windows)

            destinations = {
                "recording": recording_out,
                "fsm": fsm_out,
                "comparison": comparison_out,
                "diagnostic": diagnostic_out,
            }
            for name, path in destinations.items():
                validation["videos"][name] = validate_mp4(
                    path, ffmpeg=self.ffmpeg, stitched=False, speed_modified=False
                )
            if any(row.get("valid") is not True for row in validation["videos"].values()):
                raise VideoBuildError("one or more published final videos failed validation")
            if sha256_file(recording_out) != source_reference_validation["sha256"]:
                raise VideoBuildError("published Recording is not a byte-exact copy of selected v010")
            if sha256_file(fsm_out) != source_fsm_validation["sha256"]:
                raise VideoBuildError("published FSM is not a byte-exact copy of the successful trial")

            validation.update(
                status="PASS",
                valid=True,
                error="",
                successful_trial=trial,
                source_validation={
                    "recording": source_reference_validation,
                    "fsm": source_fsm_validation,
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
