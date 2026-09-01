"""Semantic phase alignment for the final Recording-vs-FSM video."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))


class ComparisonContractError(ValueError):
    """Raised when a video timeline cannot prove one P01--P13 sequence."""


@dataclass(frozen=True, slots=True)
class PhaseWindow:
    phase: str
    start_s: float
    end_s: float
    state: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "state": self.state or self.phase,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True, slots=True)
class AlignedPhase:
    phase: str
    reference: PhaseWindow
    fsm: PhaseWindow
    output_duration_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reference": self.reference.as_dict(),
            "fsm": self.fsm.as_dict(),
            "output_duration_s": self.output_duration_s,
            "alignment_method": "native-speed_with_last-frame-hold",
        }


def _finite_time(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonContractError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ComparisonContractError(f"{label} is not finite")
    return parsed


def validate_phase_windows(
    windows: Iterable[PhaseWindow], *, video_duration_s: float
) -> tuple[PhaseWindow, ...]:
    rows = tuple(windows)
    duration = _finite_time(video_duration_s, "video_duration_s")
    if duration <= 0.0:
        raise ComparisonContractError("video duration must be positive")
    if tuple(row.phase for row in rows) != PHASE_IDS:
        raise ComparisonContractError("timeline must contain P01--P13 exactly once and in order")
    previous_end = 0.0
    for row in rows:
        if not (0.0 <= row.start_s < row.end_s <= duration + 1.0e-6):
            raise ComparisonContractError(f"invalid {row.phase} interval {row.start_s}..{row.end_s}")
        if not math.isclose(row.start_s, previous_end, rel_tol=0.0, abs_tol=1.0e-6):
            raise ComparisonContractError(f"{row.phase} does not continuously follow the preceding phase")
        previous_end = row.end_s
    if not math.isclose(previous_end, duration, rel_tol=0.0, abs_tol=1.0e-6):
        raise ComparisonContractError("phase timeline does not cover the complete video")
    return rows


def reference_windows_from_contract(
    contract_path: Path, *, video_duration_s: float
) -> tuple[PhaseWindow, ...]:
    """Map locked reference sim times into the continuous clean-video clock."""

    payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    phases = payload.get("phases")
    cadence = payload.get("source", {}).get("telemetry_cadence", {})
    if not isinstance(phases, list) or len(phases) != 13:
        raise ComparisonContractError("reference contract does not contain 13 phases")
    video_origin_sim_s = _finite_time(cadence.get("first_sim_time_s"), "first_sim_time_s")
    starts = [
        max(0.0, _finite_time(row.get("reference_sim_start_s"), "reference_sim_start_s") - video_origin_sim_s)
        for row in phases
    ]
    starts[0] = 0.0  # retain the short, legitimate pre-action pose in P01
    duration = _finite_time(video_duration_s, "video_duration_s")
    windows: list[PhaseWindow] = []
    for index, row in enumerate(phases):
        phase = str(row.get("state_id", ""))
        end = starts[index + 1] if index + 1 < len(starts) else duration
        windows.append(
            PhaseWindow(
                phase=phase,
                state=str(row.get("state_name") or phase),
                start_s=starts[index],
                end_s=end,
            )
        )
    return validate_phase_windows(windows, video_duration_s=duration)


def _read_json_or_jsonl(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def fsm_windows_from_evidence(
    evidence_path: Path, *, video_duration_s: float
) -> tuple[PhaseWindow, ...]:
    """Read canonical windows or group a lifecycle transition JSONL ledger."""

    payload = _read_json_or_jsonl(Path(evidence_path))
    if isinstance(payload, Mapping):
        nested = payload.get("success_evidence", {})
        rows = payload.get("phase_windows") or payload.get("phases")
        if rows is None and isinstance(nested, Mapping):
            rows = nested.get("phase_windows") or nested.get("phases")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ComparisonContractError("FSM phase evidence is not a row list")
    duration = _finite_time(video_duration_s, "video_duration_s")
    if len(rows) == 13 and all(
        isinstance(row, Mapping) and ("start_s" in row or "entry_time_s" in row)
        and ("end_s" in row or "completion_time_s" in row)
        for row in rows
    ):
        windows = [
            PhaseWindow(
                phase=str(row.get("phase") or row.get("state_id")),
                state=str(row.get("state") or row.get("state_name") or row.get("state_id") or ""),
                start_s=_finite_time(row.get("start_s", row.get("entry_time_s")), "phase start"),
                end_s=_finite_time(row.get("end_s", row.get("completion_time_s")), "phase end"),
            )
            for row in rows
        ]
        return validate_phase_windows(windows, video_duration_s=duration)

    grouped: dict[str, list[tuple[float, str]]] = {phase: [] for phase in PHASE_IDS}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        phase = str(row.get("phase") or row.get("state_id") or "")[:3]
        if phase not in grouped:
            continue
        time_value = next(
            (row[key] for key in ("sim_time_s", "time_s", "timestamp_s") if key in row),
            None,
        )
        if time_value is not None:
            grouped[phase].append(
                (_finite_time(time_value, f"{phase} transition time"), str(row.get("state_name") or row.get("state_id") or phase))
            )
    if any(not grouped[phase] for phase in PHASE_IDS):
        missing = [phase for phase in PHASE_IDS if not grouped[phase]]
        raise ComparisonContractError(f"FSM transition ledger is missing phases: {missing}")
    starts = [min(time for time, _ in grouped[phase]) for phase in PHASE_IDS]
    origin = starts[0]
    starts = [max(0.0, value - origin) for value in starts]
    starts[0] = 0.0
    windows = []
    for index, phase in enumerate(PHASE_IDS):
        end = starts[index + 1] if index + 1 < len(starts) else duration
        windows.append(
            PhaseWindow(
                phase=phase,
                state=grouped[phase][0][1],
                start_s=starts[index],
                end_s=end,
            )
        )
    return validate_phase_windows(windows, video_duration_s=duration)


def align_phases(
    reference: Sequence[PhaseWindow],
    fsm: Sequence[PhaseWindow],
    *,
    maximum_duration_s: float = 200.0,
) -> tuple[AlignedPhase, ...]:
    if tuple(row.phase for row in reference) != PHASE_IDS:
        raise ComparisonContractError("reference phase order is invalid")
    if tuple(row.phase for row in fsm) != PHASE_IDS:
        raise ComparisonContractError("FSM phase order is invalid")
    aligned = tuple(
        AlignedPhase(
            phase=phase,
            reference=left,
            fsm=right,
            output_duration_s=max(left.duration_s, right.duration_s),
        )
        for phase, left, right in zip(PHASE_IDS, reference, fsm, strict=True)
    )
    total = sum(row.output_duration_s for row in aligned)
    if total > float(maximum_duration_s) + 1.0e-6:
        raise ComparisonContractError(
            f"semantic comparison would last {total:.3f}s, exceeding {maximum_duration_s:.3f}s"
        )
    return aligned


def _drawtext_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("%", "\\%")
    )


def comparison_filter(aligned: Sequence[AlignedPhase], *, fps: float = 15.0) -> str:
    """Return a 1280x720 two-panel filter preserving every source frame rate."""

    parts: list[str] = []
    panels: list[str] = []
    for index, row in enumerate(aligned):
        output_duration = row.output_duration_s
        for input_index, window, label in (
            (0, row.reference, f"r{index}"), (1, row.fsm, f"f{index}")
        ):
            hold = max(0.0, output_duration - window.duration_s)
            parts.append(
                f"[{input_index}:v]trim=start={window.start_s:.9f}:end={window.end_s:.9f},"
                f"setpts=PTS-STARTPTS,fps={fps:g},scale=640:360:flags=lanczos,"
                f"tpad=stop_mode=clone:stop_duration={hold:.9f},"
                f"trim=duration={output_duration:.9f},setpts=PTS-STARTPTS[{label}]"
            )
        phase_label = _drawtext_text(f"{row.phase}  {row.fsm.state}")
        parts.append(
            f"[r{index}][f{index}]hstack=inputs=2,"
            "pad=1280:720:0:180:color=0x101216,"
            "drawbox=x=0:y=142:w=iw:h=38:color=black@0.65:t=fill,"
            "drawtext=expansion=none:text='Recording':x=18:y=151:fontsize=18:fontcolor=white,"
            "drawtext=expansion=none:text='FSM':x=658:y=151:fontsize=18:fontcolor=white,"
            f"drawtext=expansion=none:text='{phase_label}':x=(w-text_w)/2:y=548:fontsize=18:fontcolor=white,"
            "drawtext=expansion=none:text='v010  RR_FIRST':x=w-text_w-18:y=550:fontsize=14:fontcolor=0xB8F2C8"
            f"[p{index}]"
        )
        panels.append(f"[p{index}]")
    parts.append("".join(panels) + f"concat=n={len(panels)}:v=1:a=0,format=yuv420p[outv]")
    return ";\n".join(parts)


def diagnostic_filter(windows: Sequence[PhaseWindow]) -> str:
    filters = ["drawbox=x=0:y=0:w=iw:h=42:color=black@0.58:t=fill"]
    for row in windows:
        label = _drawtext_text(f"{row.phase}  {row.state}")
        filters.append(
            f"drawtext=expansion=none:text='{label}':x=16:y=11:fontsize=18:fontcolor=white:"
            f"enable='between(t,{row.start_s:.9f},{row.end_s:.9f})'"
        )
    filters.append(
        "drawtext=expansion=none:text='body collision\\: false   wheel-only climb\\: false   v010 RR_FIRST':"
        "x=w-text_w-16:y=13:fontsize=14:fontcolor=0xB8F2C8"
    )
    filters.append("format=yuv420p")
    return ",".join(filters)
