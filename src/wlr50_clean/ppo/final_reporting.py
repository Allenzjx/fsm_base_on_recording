"""Deterministic, fail-closed plots and final Markdown reports.

This module consumes only published evaluation CSV/JSON evidence and versioned
phase/action/reward configuration.  It does not run Isaac, evaluate a policy,
or promote a checkpoint.  In particular, a candidate is never described as
improved unless the supplied promotion decision is internally consistent and
explicitly passed every gate.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import ArtifactError, _atomic_bytes, sha256_file
from .evaluation_artifacts import (
    BASELINE_EPISODE_FILENAME,
    BASELINE_PHASE_FILENAME,
    CANDIDATE_EPISODE_FILENAME,
    CANDIDATE_PHASE_FILENAME,
    CHECKPOINT_COMPARISON_FILENAME,
    EVALUATION_ARTIFACT_SCHEMA,
    PHASE_COMPARISON_FILENAME,
    PROMOTION_DECISION_FILENAME,
    RESIDUAL_ACTIVITY_FILENAME,
    REWARD_CONTRIBUTION_FILENAME,
    TERMINATION_SUMMARY_FILENAME,
)
from .phase_action_masks_v2 import DEFAULT_PHASE_ACTION_CONFIG_V2
from .phase_objectives import DEFAULT_PHASE_OBJECTIVES_PATH, DENSE_FAMILIES
from .reward_v2 import DEFAULT_REWARD_PATH_V2
from .stability_metrics import PHASE_IDS, PRIORITY_PHASES


REPORTING_SCHEMA = "wlr50_clean.ppo_final_reporting.v1"
PLOT_FILENAMES = (
    "overall_pitch_rate_comparison.png",
    "phase_wise_pitch_rate_rms.png",
    "phase_wise_roll_pitch_rms.png",
    "fr_placement_contact_impulse.png",
    "p08_transfer_post_capture_settling.png",
    "rl_lift_body_attitude.png",
    "p13_home_pose_convergence.png",
    "residual_action_by_phase.png",
    "residual_frequency_spectrum.png",
    "fsm_vs_ppo_phase_duration.png",
)
REPORT_FILENAMES = (
    "PPO_PHASE_DESIGN.md",
    "PPO_TRAINING_REPORT.md",
    "PPO_IMPROVEMENT_REPORT.md",
)
_PHASE_METRICS = (
    "duration_s",
    "roll_rms_rad",
    "pitch_rms_rad",
    "roll_rate_rms_rad_s",
    "pitch_rate_rms_rad_s",
    "placement_contact_impulse_n_s",
    "settling_time_s",
    "home_pose_error_rms_deg",
    "action_jerk_rms",
    "residual_high_frequency_fraction",
    "applied_high_frequency_fraction",
)
_EPISODE_METRICS = (
    "duration_s",
    "overall_pitch_rate_rms_rad_s",
    "overall_roll_rate_rms_rad_s",
)
_DENSE_SUM_COLUMNS = tuple(f"{name}_sum" for name in DENSE_FAMILIES)
_PROMOTION_GATE_ORDER = (
    "p01_p13_completed",
    "task_success_rate_not_below_fsm",
    "body_collision_zero",
    "wheel_only_climb_zero",
    "fall_or_physics_explosion_zero",
    "safety_abort_zero",
    "duration_each_under_200_s",
    "duration_not_over_fsm_by_15pct",
    "frozen_hashes_unchanged",
    "recording_runtime_access_zero",
    "global_stability_improvement_at_least_5pct",
    "at_least_4_of_5_priority_phases_improve",
    "no_priority_phase_degrades_over_10pct",
    "one_visual_key_metric_gate",
    "level_calibration_quality_passed",
    "residual_activity_calibrated",
    "priority_phases_have_real_residual",
    "at_least_10_phases_have_real_residual",
)


class FinalReportingError(RuntimeError):
    """Required evidence is missing, contradictory, or would be overwritten."""


@dataclass(frozen=True, slots=True)
class FinalReportingPaths:
    output_root: Path
    plots_directory: Path
    reports_directory: Path
    overall_pitch_rate_comparison: Path
    phase_wise_pitch_rate_rms: Path
    phase_wise_roll_pitch_rms: Path
    fr_placement_contact_impulse: Path
    p08_transfer_post_capture_settling: Path
    rl_lift_body_attitude: Path
    p13_home_pose_convergence: Path
    residual_action_by_phase: Path
    residual_frequency_spectrum: Path
    fsm_vs_ppo_phase_duration: Path
    phase_design_report: Path
    training_report: Path
    improvement_report: Path

    def files(self) -> tuple[Path, ...]:
        return tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"output_root", "plots_directory", "reports_directory"}
        )


@dataclass(frozen=True, slots=True)
class _ReportingEvidence:
    metrics_directory: Path
    baseline_episode: tuple[Mapping[str, str], ...]
    baseline_phase: tuple[Mapping[str, str], ...]
    candidate_episode: tuple[Mapping[str, str], ...]
    candidate_phase: tuple[Mapping[str, str], ...]
    checkpoint_comparison: tuple[Mapping[str, str], ...]
    phase_comparison: tuple[Mapping[str, str], ...]
    residual_activity: tuple[Mapping[str, str], ...]
    reward_contribution: tuple[Mapping[str, str], ...]
    termination_summary: tuple[Mapping[str, str], ...]
    promotion: Mapping[str, Any]
    phase_config: Mapping[str, Any]
    action_config: Mapping[str, Any]
    reward_config: Mapping[str, Any]
    training_manifest: Mapping[str, Any] | None
    input_paths: tuple[Path, ...]
    seeds: tuple[int, ...]
    baseline_label: str
    candidate_label: str
    promoted: bool


def _read_csv(path: Path, required: Sequence[str]) -> tuple[Mapping[str, str], ...]:
    if not path.is_file():
        raise FinalReportingError(f"required metrics CSV is missing: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            if not fields or len(fields) != len(set(fields)):
                raise FinalReportingError(f"CSV header is empty or duplicated: {path}")
            missing = sorted(set(required).difference(fields))
            if missing:
                raise FinalReportingError(f"CSV {path.name} is missing columns {missing}")
            rows = tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FinalReportingError(f"cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise FinalReportingError(f"CSV has no evidence rows: {path}")
    if any(None in row for row in rows):
        raise FinalReportingError(f"CSV contains fields beyond its declared header: {path}")
    return rows


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalReportingError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalReportingError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FinalReportingError(f"{label} must be a JSON object: {path}")
    return value


def _read_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalReportingError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FinalReportingError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FinalReportingError(f"{label} must be a YAML mapping: {path}")
    return value


def _finite(row: Mapping[str, Any], name: str, context: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalReportingError(f"{context}.{name} must be numeric") from exc
    if not math.isfinite(value):
        raise FinalReportingError(f"{context}.{name} must be finite")
    return value


def _integer(row: Mapping[str, Any], name: str, context: str) -> int:
    value = _finite(row, name, context)
    result = int(value)
    if result != value:
        raise FinalReportingError(f"{context}.{name} must be an integer")
    return result


def _boolean(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise FinalReportingError(f"{context} must be boolean")


def _episode_index(
    rows: Sequence[Mapping[str, str]], label: str
) -> dict[int, Mapping[str, str]]:
    indexed: dict[int, Mapping[str, str]] = {}
    for offset, row in enumerate(rows):
        seed = _integer(row, "seed", f"{label}[{offset}]")
        if seed in indexed:
            raise FinalReportingError(f"{label} contains duplicate seed {seed}")
        for metric in _EPISODE_METRICS:
            if _finite(row, metric, f"{label}[seed={seed}]") < 0.0:
                raise FinalReportingError(
                    f"{label}[seed={seed}].{metric} must be non-negative"
                )
        _boolean(row["task_success"], f"{label}[seed={seed}].task_success")
        indexed[seed] = row
    return indexed


def _phase_index(
    rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int],
    label: str,
    numeric_columns: Sequence[str],
) -> dict[tuple[int, str], Mapping[str, str]]:
    indexed: dict[tuple[int, str], Mapping[str, str]] = {}
    for offset, row in enumerate(rows):
        seed = _integer(row, "seed", f"{label}[{offset}]")
        phase = str(row.get("phase", ""))
        if phase not in PHASE_IDS:
            raise FinalReportingError(f"{label} contains unknown phase {phase!r}")
        key = (seed, phase)
        if key in indexed:
            raise FinalReportingError(f"{label} duplicates seed/phase {key}")
        for metric in numeric_columns:
            _finite(row, metric, f"{label}[seed={seed},phase={phase}]")
        indexed[key] = row
    expected = {(int(seed), phase) for seed in seeds for phase in PHASE_IDS}
    if set(indexed) != expected:
        missing = sorted(expected.difference(indexed))
        extra = sorted(set(indexed).difference(expected))
        raise FinalReportingError(
            f"{label} must contain every paired seed/P01-P13 row exactly once; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return indexed


def _validate_promotion(
    payload: Mapping[str, Any], seeds: Sequence[int]
) -> tuple[bool, str, str]:
    if payload.get("schema") != EVALUATION_ARTIFACT_SCHEMA:
        raise FinalReportingError("promotion JSON has an unexpected schema")
    promotion = payload.get("promotion")
    if not isinstance(promotion, Mapping):
        raise FinalReportingError("promotion JSON omits its promotion decision")
    promoted_raw = promotion.get("promoted")
    if not isinstance(promoted_raw, bool):
        raise FinalReportingError("promotion.promoted must be an explicit boolean")
    checks = promotion.get("checks")
    ordered = payload.get("checks_in_evaluation_order")
    if not isinstance(checks, Mapping) or not checks or not isinstance(ordered, list):
        raise FinalReportingError("promotion JSON omits ordered gate evidence")
    ordered_checks: list[tuple[str, bool]] = []
    for index, row in enumerate(ordered):
        if not isinstance(row, Mapping) or not isinstance(row.get("gate"), str):
            raise FinalReportingError(f"promotion gate row {index} is invalid")
        gate = str(row["gate"])
        passed = row.get("passed")
        if not isinstance(passed, bool) or gate not in checks or checks[gate] is not passed:
            raise FinalReportingError("promotion ordered gates disagree with promotion.checks")
        ordered_checks.append((gate, passed))
    if (
        set(checks) != set(_PROMOTION_GATE_ORDER)
        or tuple(gate for gate, _ in ordered_checks) != _PROMOTION_GATE_ORDER
    ):
        raise FinalReportingError(
            "promotion JSON does not contain the complete authoritative gate order"
        )
    failed = next((gate for gate, passed in ordered_checks if not passed), None)
    nested_failed = promotion.get("first_failed_gate")
    if payload.get("first_failed_gate") != nested_failed or nested_failed != failed:
        raise FinalReportingError("promotion first-failed-gate evidence is inconsistent")
    if promoted_raw != (failed is None):
        raise FinalReportingError("promotion status disagrees with its gate results")
    try:
        paired = tuple(int(seed) for seed in payload["paired_seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalReportingError("promotion paired_seeds is invalid") from exc
    if paired != tuple(seeds) or len(set(paired)) != len(paired):
        raise FinalReportingError("promotion seeds do not match the paired CSV evidence")
    try:
        minimum = int(payload.get("minimum_paired_seeds", 5))
        episode_count = int(payload["paired_episode_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalReportingError("promotion paired episode counts are invalid") from exc
    if len(paired) < minimum or minimum < 5:
        raise FinalReportingError("promotion has fewer than five paired seeds")
    if episode_count != len(paired):
        raise FinalReportingError("promotion paired_episode_count disagrees with its seeds")
    stability = _finite(
        promotion,
        "global_stability_improvement_fraction",
        "promotion",
    )
    improved_priority = _integer(
        promotion,
        "improved_priority_phase_count",
        "promotion",
    )
    if improved_priority < 0 or improved_priority > len(PRIORITY_PHASES):
        raise FinalReportingError("promotion improved-priority count is out of range")
    if checks["global_stability_improvement_at_least_5pct"] != (stability >= 0.05):
        raise FinalReportingError("promotion global-stability gate disagrees with its value")
    if checks["at_least_4_of_5_priority_phases_improve"] != (improved_priority >= 4):
        raise FinalReportingError("promotion priority-count gate disagrees with its value")
    baseline = str(payload.get("baseline_checkpoint", "")).strip()
    candidate = str(payload.get("candidate_checkpoint", "")).strip()
    if not baseline or not candidate or baseline == candidate:
        raise FinalReportingError("promotion checkpoint labels are invalid")
    return promoted_raw, baseline, candidate


def _validate_configs(
    phase: Mapping[str, Any], action: Mapping[str, Any], reward: Mapping[str, Any]
) -> None:
    if phase.get("schema") != "wlr50_clean.ppo_phase_objectives.v2":
        raise FinalReportingError("phase objective config schema is not v2")
    if action.get("schema") != "wlr50_clean.ppo_phase_action_masks.v2":
        raise FinalReportingError("phase action config schema is not v2")
    if reward.get("schema") != "wlr50_clean.ppo_reward.v2":
        raise FinalReportingError("reward config schema is not v2")
    if tuple(phase.get("state_ids", ())) != PHASE_IDS:
        raise FinalReportingError("phase objective config does not declare ordered P01-P13")
    if tuple(phase.get("dense_families", ())) != DENSE_FAMILIES:
        raise FinalReportingError("phase objective config does not declare five dense families")
    phase_rows = phase.get("phases")
    action_rows = action.get("phases")
    if not isinstance(phase_rows, Mapping) or tuple(phase_rows) != PHASE_IDS:
        raise FinalReportingError("phase objective rows are incomplete or reordered")
    if not isinstance(action_rows, Mapping) or tuple(action_rows) != PHASE_IDS:
        raise FinalReportingError("phase action rows are incomplete or reordered")
    full12 = tuple(str(value) for value in action.get("full12_order", ()))
    if len(full12) != 12 or len(set(full12)) != 12:
        raise FinalReportingError("phase action Full12 order is invalid")
    if tuple(reward.get("dense_families", {})) != DENSE_FAMILIES:
        raise FinalReportingError("reward v2 must contain exactly five dense families")
    for phase_id in PHASE_IDS:
        objective = phase_rows[phase_id]
        action_row = action_rows[phase_id]
        if not isinstance(objective, Mapping) or not isinstance(action_row, Mapping):
            raise FinalReportingError(f"{phase_id} config row is not a mapping")
        weights = objective.get("prompt_weights")
        if not isinstance(weights, Mapping) or tuple(weights) != DENSE_FAMILIES:
            raise FinalReportingError(f"{phase_id} reward weights are incomplete")
        weight_values = tuple(_finite(weights, name, f"{phase_id}.weights") for name in DENSE_FAMILIES)
        if not math.isclose(sum(weight_values), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise FinalReportingError(f"{phase_id} reward weights do not sum to one")
        mask = tuple(action_row.get("mask_full12", ()))
        scale = tuple(action_row.get("scale_full12", ()))
        if len(mask) != 12 or any(value not in (0, 1) for value in mask):
            raise FinalReportingError(f"{phase_id} action mask is invalid")
        if len(scale) != 12:
            raise FinalReportingError(f"{phase_id} action scale is invalid")
        for index, value in enumerate(scale):
            numeric = _finite({"value": value}, "value", f"{phase_id}.scale[{index}]")
            if numeric < 0.0 or (mask[index] == 0 and numeric != 0.0):
                raise FinalReportingError(f"{phase_id} mask/scale contract is invalid")


def _load_evidence(
    metrics_directory: Path,
    *,
    phase_objectives_config: Path,
    phase_action_config: Path,
    reward_config: Path,
    training_manifest: Path | None,
) -> _ReportingEvidence:
    directory = metrics_directory.resolve()
    csv_specs = {
        "baseline_episode": (
            BASELINE_EPISODE_FILENAME,
            ("checkpoint", "seed", "task_success", *_EPISODE_METRICS),
        ),
        "baseline_phase": (
            BASELINE_PHASE_FILENAME,
            ("checkpoint", "seed", "phase", *_PHASE_METRICS),
        ),
        "candidate_episode": (
            CANDIDATE_EPISODE_FILENAME,
            ("checkpoint", "seed", "task_success", *_EPISODE_METRICS),
        ),
        "candidate_phase": (
            CANDIDATE_PHASE_FILENAME,
            ("checkpoint", "seed", "phase", *_PHASE_METRICS),
        ),
        "checkpoint_comparison": (
            CHECKPOINT_COMPARISON_FILENAME,
            (
                "role",
                "checkpoint",
                "episode_count",
                "task_success_count",
                "body_collision_count",
                "wheel_only_climb_count",
                "mean_duration_s",
                "mean_overall_pitch_rate_rms_rad_s",
                "mean_overall_roll_rate_rms_rad_s",
                "mean_placement_contact_impulse_n_s",
                "mean_home_recovery_action_jerk_rms",
            ),
        ),
        "phase_comparison": (
            PHASE_COMPARISON_FILENAME,
            ("phase", "primary_phase_score_improvement_fraction"),
        ),
        "residual_activity": (
            RESIDUAL_ACTIVITY_FILENAME,
            (
                "checkpoint",
                "seed",
                "phase",
                "normalized_residual_rms",
                "normalized_residual_peak",
                "active_channel_count",
                "nonzero",
            ),
        ),
        "reward_contribution": (
            REWARD_CONTRIBUTION_FILENAME,
            ("checkpoint", "seed", "phase", *_DENSE_SUM_COLUMNS, "total_reward_sum"),
        ),
        "termination_summary": (
            TERMINATION_SUMMARY_FILENAME,
            (
                "checkpoint",
                "seed",
                "task_success",
                "body_collision",
                "wheel_only_climb",
                "duration_s",
            ),
        ),
    }
    loaded = {
        name: _read_csv(directory / filename, required)
        for name, (filename, required) in csv_specs.items()
    }
    baseline_index = _episode_index(loaded["baseline_episode"], "baseline episode")
    candidate_index = _episode_index(loaded["candidate_episode"], "candidate episode")
    if set(baseline_index) != set(candidate_index):
        raise FinalReportingError("baseline and candidate episode seeds are not paired")
    seeds = tuple(sorted(baseline_index))
    _phase_index(
        loaded["baseline_phase"],
        seeds=seeds,
        label="baseline phase",
        numeric_columns=_PHASE_METRICS,
    )
    _phase_index(
        loaded["candidate_phase"],
        seeds=seeds,
        label="candidate phase",
        numeric_columns=_PHASE_METRICS,
    )
    _phase_index(
        loaded["residual_activity"],
        seeds=seeds,
        label="candidate residual activity",
        numeric_columns=(
            "normalized_residual_rms",
            "normalized_residual_peak",
            "active_channel_count",
        ),
    )
    for row in loaded["residual_activity"]:
        _boolean(row["nonzero"], "residual_activity.nonzero")
    _phase_index(
        loaded["reward_contribution"],
        seeds=seeds,
        label="candidate reward contribution",
        numeric_columns=(*_DENSE_SUM_COLUMNS, "total_reward_sum"),
    )
    phase_rows = loaded["phase_comparison"]
    if tuple(str(row["phase"]) for row in phase_rows) != PHASE_IDS:
        raise FinalReportingError("phase comparison must contain ordered P01-P13 exactly")
    for row in phase_rows:
        _finite(row, "primary_phase_score_improvement_fraction", "phase comparison")

    promotion_path = directory / PROMOTION_DECISION_FILENAME
    promotion = _read_json(promotion_path, "promotion decision")
    promoted, baseline_label, candidate_label = _validate_promotion(promotion, seeds)
    for label, rows in (
        (baseline_label, loaded["baseline_episode"]),
        (baseline_label, loaded["baseline_phase"]),
        (candidate_label, loaded["candidate_episode"]),
        (candidate_label, loaded["candidate_phase"]),
        (candidate_label, loaded["residual_activity"]),
        (candidate_label, loaded["reward_contribution"]),
    ):
        if any(str(row.get("checkpoint", "")) != label for row in rows):
            raise FinalReportingError(f"CSV checkpoint labels disagree with {label!r}")
    checkpoints = {str(row.get("role")): row for row in loaded["checkpoint_comparison"]}
    if set(checkpoints) != {"baseline", "candidate"} or len(loaded["checkpoint_comparison"]) != 2:
        raise FinalReportingError("checkpoint comparison must contain baseline and candidate once")
    if (
        checkpoints["baseline"].get("checkpoint") != baseline_label
        or checkpoints["candidate"].get("checkpoint") != candidate_label
    ):
        raise FinalReportingError("checkpoint comparison labels disagree with promotion JSON")
    for role, row in checkpoints.items():
        for name in (
            "episode_count",
            "task_success_count",
            "body_collision_count",
            "wheel_only_climb_count",
        ):
            count = _integer(row, name, f"checkpoint comparison {role}")
            if count < 0 or count > len(seeds):
                raise FinalReportingError(
                    f"checkpoint comparison {role}.{name} is out of range"
                )
        if _integer(row, "episode_count", f"checkpoint comparison {role}") != len(seeds):
            raise FinalReportingError(
                f"checkpoint comparison {role}.episode_count disagrees with paired seeds"
            )
        for name in (
            "mean_duration_s",
            "mean_overall_pitch_rate_rms_rad_s",
            "mean_overall_roll_rate_rms_rad_s",
            "mean_placement_contact_impulse_n_s",
            "mean_home_recovery_action_jerk_rms",
        ):
            _finite(row, name, f"checkpoint comparison {role}")
    expected_termination = {
        (label, seed)
        for label in (baseline_label, candidate_label)
        for seed in seeds
    }
    actual_termination = {
        (str(row["checkpoint"]), _integer(row, "seed", "termination summary"))
        for row in loaded["termination_summary"]
    }
    if actual_termination != expected_termination or len(loaded["termination_summary"]) != len(expected_termination):
        raise FinalReportingError("termination summary does not match both paired episode sets")
    for row in loaded["termination_summary"]:
        context = f"termination summary {row['checkpoint']}/{row['seed']}"
        for name in ("task_success", "body_collision", "wheel_only_climb"):
            _boolean(row[name], f"{context}.{name}")
        if _finite(row, "duration_s", context) < 0.0:
            raise FinalReportingError(f"{context}.duration_s must be non-negative")

    phase_path = phase_objectives_config.resolve()
    action_path = phase_action_config.resolve()
    reward_path = reward_config.resolve()
    phase_config = _read_yaml(phase_path, "phase objective config")
    action_config = _read_yaml(action_path, "phase action config")
    reward_config_value = _read_yaml(reward_path, "reward config")
    _validate_configs(phase_config, action_config, reward_config_value)
    training_value = (
        None
        if training_manifest is None
        else _read_json(training_manifest.resolve(), "training manifest")
    )
    inputs = tuple(
        directory / filename for filename, _ in csv_specs.values()
    ) + (promotion_path, phase_path, action_path, reward_path) + (
        () if training_manifest is None else (training_manifest.resolve(),)
    )
    return _ReportingEvidence(
        metrics_directory=directory,
        baseline_episode=loaded["baseline_episode"],
        baseline_phase=loaded["baseline_phase"],
        candidate_episode=loaded["candidate_episode"],
        candidate_phase=loaded["candidate_phase"],
        checkpoint_comparison=loaded["checkpoint_comparison"],
        phase_comparison=loaded["phase_comparison"],
        residual_activity=loaded["residual_activity"],
        reward_contribution=loaded["reward_contribution"],
        termination_summary=loaded["termination_summary"],
        promotion=promotion,
        phase_config=phase_config,
        action_config=action_config,
        reward_config=reward_config_value,
        training_manifest=training_value,
        input_paths=inputs,
        seeds=seeds,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        promoted=promoted,
    )


def _rows_by_seed(
    rows: Sequence[Mapping[str, str]], seeds: Sequence[int]
) -> tuple[Mapping[str, str], ...]:
    indexed = {int(row["seed"]): row for row in rows}
    return tuple(indexed[int(seed)] for seed in seeds)


def _phase_values(
    rows: Sequence[Mapping[str, str]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    spreads = []
    for phase in PHASE_IDS:
        phase_values = np.asarray(
            [_finite(row, metric, f"{phase}.{metric}") for row in rows if row["phase"] == phase],
            dtype=float,
        )
        values.append(float(np.mean(phase_values)))
        spreads.append(float(np.std(phase_values)))
    return np.asarray(values), np.asarray(spreads)


def _one_phase_values(
    rows: Sequence[Mapping[str, str]], phase: str, metric: str, seeds: Sequence[int]
) -> np.ndarray:
    indexed = {
        (int(row["seed"]), str(row["phase"])): row
        for row in rows
    }
    return np.asarray(
        [_finite(indexed[(seed, phase)], metric, f"{seed}.{phase}.{metric}") for seed in seeds],
        dtype=float,
    )


def _figure_bytes(
    title: str,
    promoted: bool,
    draw: Callable[[Any, Any], None],
    *,
    size: tuple[float, float] = (11.0, 6.2),
) -> bytes:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise FinalReportingError(f"matplotlib is required for final plots: {exc}") from exc
    settings = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbfb",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(settings):
        figure = plt.figure(figsize=size, dpi=120)
        try:
            draw(figure, plt)
            figure.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
            status = "Promotion evidence: PASSED" if promoted else "Promotion evidence: NOT PASSED"
            figure.text(0.995, 0.01, status, ha="right", va="bottom", fontsize=8)
            figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.94))
            stream = io.BytesIO()
            figure.savefig(
                stream,
                format="png",
                dpi=120,
                metadata={"Software": REPORTING_SCHEMA},
            )
            return stream.getvalue()
        except FinalReportingError:
            raise
        except Exception as exc:
            raise FinalReportingError(f"cannot render plot {title!r}: {exc}") from exc
        finally:
            plt.close(figure)


def _comparison_bars(ax: Any, baseline: np.ndarray, candidate: np.ndarray, labels: Sequence[str], ylabel: str) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    ax.bar(x - width / 2, baseline, width, label="FSM baseline", color="#6b7280")
    ax.bar(x + width / 2, candidate, width, label="PPO candidate", color="#2563eb")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.legend()


def _plot_payloads(evidence: _ReportingEvidence) -> Mapping[str, bytes]:
    seeds = evidence.seeds
    baseline_episode = _rows_by_seed(evidence.baseline_episode, seeds)
    candidate_episode = _rows_by_seed(evidence.candidate_episode, seeds)
    phase_labels = list(PHASE_IDS)
    payloads: dict[str, bytes] = {}

    baseline_overall = np.asarray(
        [_finite(row, "overall_pitch_rate_rms_rad_s", "baseline episode") for row in baseline_episode]
    )
    candidate_overall = np.asarray(
        [_finite(row, "overall_pitch_rate_rms_rad_s", "candidate episode") for row in candidate_episode]
    )

    def overall(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(seeds))
        for index in range(len(seeds)):
            ax.plot(
                (x[index] - 0.08, x[index] + 0.08),
                (baseline_overall[index], candidate_overall[index]),
                color="#9ca3af",
                linewidth=1,
            )
        ax.scatter(x - 0.08, baseline_overall, label="FSM baseline", color="#6b7280", zorder=3)
        ax.scatter(x + 0.08, candidate_overall, label="PPO candidate", color="#2563eb", zorder=3)
        ax.set_xticks(x, [str(seed) for seed in seeds])
        ax.set_xlabel("Paired validation seed")
        ax.set_ylabel("Overall pitch-rate RMS (rad/s)")
        ax.legend()

    payloads[PLOT_FILENAMES[0]] = _figure_bytes(
        "Overall Pitch-Rate Comparison", evidence.promoted, overall
    )

    base_pitch, base_pitch_std = _phase_values(evidence.baseline_phase, "pitch_rate_rms_rad_s")
    cand_pitch, cand_pitch_std = _phase_values(evidence.candidate_phase, "pitch_rate_rms_rad_s")

    def phase_pitch(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(PHASE_IDS))
        width = 0.38
        ax.bar(x - width / 2, base_pitch, width, yerr=base_pitch_std, capsize=2, label="FSM baseline", color="#6b7280")
        ax.bar(x + width / 2, cand_pitch, width, yerr=cand_pitch_std, capsize=2, label="PPO candidate", color="#2563eb")
        ax.set_xticks(x, phase_labels)
        ax.set_ylabel("Pitch-rate RMS (rad/s); mean ± population SD")
        ax.legend()

    payloads[PLOT_FILENAMES[1]] = _figure_bytes(
        "Phase-Wise Pitch-Rate RMS", evidence.promoted, phase_pitch
    )

    base_roll, _ = _phase_values(evidence.baseline_phase, "roll_rms_rad")
    cand_roll, _ = _phase_values(evidence.candidate_phase, "roll_rms_rad")
    base_att_pitch, _ = _phase_values(evidence.baseline_phase, "pitch_rms_rad")
    cand_att_pitch, _ = _phase_values(evidence.candidate_phase, "pitch_rms_rad")

    def attitude(figure: Any, _plt: Any) -> None:
        axes = figure.subplots(2, 1, sharex=True)
        _comparison_bars(axes[0], base_roll, cand_roll, phase_labels, "Roll RMS (rad)")
        _comparison_bars(axes[1], base_att_pitch, cand_att_pitch, phase_labels, "Pitch RMS (rad)")
        axes[1].set_xlabel("FSM phase")

    payloads[PLOT_FILENAMES[2]] = _figure_bytes(
        "Phase-Wise Roll/Pitch RMS", evidence.promoted, attitude, size=(11.0, 8.0)
    )

    fr_base = _one_phase_values(evidence.baseline_phase, "P03", "placement_contact_impulse_n_s", seeds)
    fr_candidate = _one_phase_values(evidence.candidate_phase, "P03", "placement_contact_impulse_n_s", seeds)

    def fr_impulse(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        _comparison_bars(ax, fr_base, fr_candidate, [str(seed) for seed in seeds], "P03 FR contact impulse (N·s)")
        ax.set_xlabel("Paired validation seed")

    payloads[PLOT_FILENAMES[3]] = _figure_bytes(
        "FR Placement Contact Impulse (P03)", evidence.promoted, fr_impulse
    )

    p08_base_rate = _one_phase_values(evidence.baseline_phase, "P08", "pitch_rate_rms_rad_s", seeds)
    p08_cand_rate = _one_phase_values(evidence.candidate_phase, "P08", "pitch_rate_rms_rad_s", seeds)
    p08_base_settle = _one_phase_values(evidence.baseline_phase, "P08", "settling_time_s", seeds)
    p08_cand_settle = _one_phase_values(evidence.candidate_phase, "P08", "settling_time_s", seeds)

    def p08(figure: Any, _plt: Any) -> None:
        axes = figure.subplots(1, 2)
        labels = [str(seed) for seed in seeds]
        _comparison_bars(axes[0], p08_base_rate, p08_cand_rate, labels, "Pitch-rate RMS (rad/s)")
        axes[0].set_title("Transfer-phase motion")
        _comparison_bars(axes[1], p08_base_settle, p08_cand_settle, labels, "Settling time (s)")
        axes[1].set_title("Post-capture settling")
        for ax in axes:
            ax.tick_params(axis="x", rotation=35)

    payloads[PLOT_FILENAMES[4]] = _figure_bytes(
        "P08 Transfer and Post-Capture Settling", evidence.promoted, p08
    )

    p12_base_roll = _one_phase_values(evidence.baseline_phase, "P12", "roll_rms_rad", seeds)
    p12_cand_roll = _one_phase_values(evidence.candidate_phase, "P12", "roll_rms_rad", seeds)
    p12_base_pitch = _one_phase_values(evidence.baseline_phase, "P12", "pitch_rms_rad", seeds)
    p12_cand_pitch = _one_phase_values(evidence.candidate_phase, "P12", "pitch_rms_rad", seeds)

    def rl_lift(figure: Any, _plt: Any) -> None:
        axes = figure.subplots(1, 2)
        labels = [str(seed) for seed in seeds]
        _comparison_bars(axes[0], p12_base_roll, p12_cand_roll, labels, "Roll RMS (rad)")
        _comparison_bars(axes[1], p12_base_pitch, p12_cand_pitch, labels, "Pitch RMS (rad)")
        for ax in axes:
            ax.set_xlabel("Paired seed")

    payloads[PLOT_FILENAMES[5]] = _figure_bytes(
        "RL Lift Body Attitude (P12)", evidence.promoted, rl_lift
    )

    p13_base_home = _one_phase_values(evidence.baseline_phase, "P13", "home_pose_error_rms_deg", seeds)
    p13_cand_home = _one_phase_values(evidence.candidate_phase, "P13", "home_pose_error_rms_deg", seeds)
    p13_base_jerk = _one_phase_values(evidence.baseline_phase, "P13", "action_jerk_rms", seeds)
    p13_cand_jerk = _one_phase_values(evidence.candidate_phase, "P13", "action_jerk_rms", seeds)

    def p13(figure: Any, _plt: Any) -> None:
        axes = figure.subplots(1, 2)
        labels = [str(seed) for seed in seeds]
        _comparison_bars(axes[0], p13_base_home, p13_cand_home, labels, "Home-pose error RMS (deg)")
        _comparison_bars(axes[1], p13_base_jerk, p13_cand_jerk, labels, "Applied-action jerk RMS")
        for ax in axes:
            ax.set_xlabel("Paired seed")

    payloads[PLOT_FILENAMES[6]] = _figure_bytes(
        "P13 Home-Pose Convergence", evidence.promoted, p13
    )

    residual_rms, residual_rms_std = _phase_values(evidence.residual_activity, "normalized_residual_rms")
    residual_peak, residual_peak_std = _phase_values(evidence.residual_activity, "normalized_residual_peak")

    def residual(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(PHASE_IDS))
        width = 0.38
        ax.bar(x - width / 2, residual_rms, width, yerr=residual_rms_std, capsize=2, label="Normalized RMS", color="#2563eb")
        ax.bar(x + width / 2, residual_peak, width, yerr=residual_peak_std, capsize=2, label="Normalized peak", color="#f59e0b")
        ax.set_xticks(x, phase_labels)
        ax.set_ylabel("Residual / configured phase scale")
        ax.legend()

    payloads[PLOT_FILENAMES[7]] = _figure_bytes(
        "PPO Residual Action by Phase", evidence.promoted, residual
    )

    residual_hf, residual_hf_std = _phase_values(evidence.candidate_phase, "residual_high_frequency_fraction")
    applied_hf, applied_hf_std = _phase_values(evidence.candidate_phase, "applied_high_frequency_fraction")

    def spectrum(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(PHASE_IDS))
        ax.errorbar(x, residual_hf, yerr=residual_hf_std, marker="o", capsize=3, label="Residual >3 Hz energy fraction", color="#2563eb")
        ax.errorbar(x, applied_hf, yerr=applied_hf_std, marker="s", capsize=3, label="Applied >3 Hz energy fraction", color="#dc2626")
        ax.set_xticks(x, phase_labels)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel("High-frequency spectral-energy fraction")
        ax.set_xlabel("FSM phase")
        ax.legend()
        ax.text(
            0.01,
            0.98,
            "Canonical CSV exports aggregate >3 Hz energy; raw PSD bins are not inferred.",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    payloads[PLOT_FILENAMES[8]] = _figure_bytes(
        "Residual Frequency-Domain Summary", evidence.promoted, spectrum
    )

    base_duration, base_duration_std = _phase_values(evidence.baseline_phase, "duration_s")
    cand_duration, cand_duration_std = _phase_values(evidence.candidate_phase, "duration_s")

    def duration(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(PHASE_IDS))
        width = 0.38
        ax.bar(x - width / 2, base_duration, width, yerr=base_duration_std, capsize=2, label="FSM baseline", color="#6b7280")
        ax.bar(x + width / 2, cand_duration, width, yerr=cand_duration_std, capsize=2, label="PPO candidate", color="#2563eb")
        ax.set_xticks(x, phase_labels)
        ax.set_ylabel("Phase duration (s); mean ± population SD")
        ax.legend()

    payloads[PLOT_FILENAMES[9]] = _figure_bytes(
        "FSM vs PPO Candidate Phase Duration", evidence.promoted, duration
    )
    if tuple(payloads) != PLOT_FILENAMES:
        raise FinalReportingError("internal plot filename order changed")
    return payloads


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "not provided"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).replace("|", "\\|").replace("\n", " ")
    if math.isfinite(numeric):
        return f"{numeric:.6g}"
    return "non-finite"


def _hash_table(paths: Sequence[Path]) -> str:
    lines = ["| Input | SHA-256 |", "|---|---|"]
    lines.extend(f"| `{path}` | `{sha256_file(path)}` |" for path in paths)
    return "\n".join(lines)


def _phase_design_report(evidence: _ReportingEvidence) -> str:
    full12 = tuple(evidence.action_config["full12_order"])
    phase_rows = evidence.phase_config["phases"]
    action_rows = evidence.action_config["phases"]
    lines = [
        "# PPO Phase Design",
        "",
        f"Schema: `{REPORTING_SCHEMA}`. This report is derived from versioned configuration, not inferred from a training curve.",
        "",
        "## Shared policy contract",
        "",
        "One shared actor and critic are conditioned on FSM phase/lifecycle. The frozen FSM owns P01–P13 ordering, transitions, recovery, and task truth; PPO contributes only phase-masked bounded Full12 residuals.",
        "",
        "## Phase-specific objectives and controls",
        "",
        "| Phase | Priority | Primary objective | Stability mode | Active residual channels | Reward weights (progress/stability/contact/smooth/residual) |",
        "|---|---:|---|---|---|---|",
    ]
    for phase_id in PHASE_IDS:
        objective = phase_rows[phase_id]
        action = action_rows[phase_id]
        channels = [
            name
            for name, mask, scale in zip(
                full12,
                action["mask_full12"],
                action["scale_full12"],
                strict=True,
            )
            if int(mask) == 1 and float(scale) > 0.0
        ]
        weights = objective["prompt_weights"]
        weight_text = "/".join(_fmt(weights[name]) for name in DENSE_FAMILIES)
        lines.append(
            "| "
            + " | ".join(
                (
                    phase_id,
                    "yes" if phase_id in PRIORITY_PHASES else "no",
                    _fmt(objective.get("primary_objective")),
                    _fmt(objective.get("stability_mode")),
                    ", ".join(channels),
                    weight_text,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Priority phases",
            "",
        ]
    )
    for phase_id in PRIORITY_PHASES:
        row = phase_rows[phase_id]
        lines.append(
            f"- **{phase_id}** — {_fmt(row.get('primary_objective'))} "
            f"Mode: `{_fmt(row.get('stability_mode'))}`."
        )
    transfer = [
        phase_id
        for phase_id in PHASE_IDS
        if phase_rows[phase_id].get("stability_mode") == "TRANSFER_AWARE"
    ]
    lines.extend(
        [
            "",
            "## Reward v2",
            "",
            "The active dense reward is limited to five normalized families:",
            "",
            *[f"- `{name}`" for name in DENSE_FAMILIES],
            "",
            f"Transfer-aware phases: {', '.join(transfer)}. Their configuration ramps stability after capture rather than treating necessary transfer attitude as a static-level error.",
            "",
            "Legacy-v1 deletion claims are intentionally not reconstructed here: only the supplied v2 configuration is authoritative for this report.",
            "",
            "## Input integrity",
            "",
            _hash_table(evidence.input_paths[-(4 if evidence.training_manifest is not None else 3):]),
            "",
        ]
    )
    return "\n".join(lines)


def _checkpoint_rows(evidence: _ReportingEvidence) -> Mapping[str, Mapping[str, str]]:
    return {str(row["role"]): row for row in evidence.checkpoint_comparison}


def _training_manifest_section(manifest: Mapping[str, Any] | None) -> str:
    if manifest is None:
        return (
            "No training manifest was supplied. Stage names, optimizer steps, "
            "environment steps, wall time, and best-checkpoint step are therefore "
            "not inferred from evaluation CSVs."
        )
    fields = (
        ("Schema", "schema"),
        ("Stage", "stage"),
        ("Requested policy decisions", "requested_policy_decisions"),
        ("Stage policy decisions", "stage_policy_decisions"),
        ("Global policy decisions", "global_policy_decisions"),
        ("Iterations", "iterations"),
        ("Parallel environments", "num_envs"),
        ("Rollout length", "rollout_length"),
        ("Wall time (s)", "wall_time_s"),
        ("Save/load round trip", "save_load_round_trip"),
        ("Checkpoint SHA-256", "checkpoint_sha256"),
    )
    lines = [
        "Only neutral, whitelisted training facts are reproduced; status or "
        "improvement labels inside the source manifest are not accepted as evidence.",
        "",
        "| Training fact | Value |",
        "|---|---|",
    ]
    lines.extend(
        f"| {label} | {_fmt(manifest.get(key))} |" for label, key in fields
    )
    return "\n".join(lines)


def _training_report(evidence: _ReportingEvidence) -> str:
    status = (
        "PASSED — promotion evidence authorizes the improved-checkpoint label."
        if evidence.promoted
        else "NOT PASSED — this report treats the PPO output only as a candidate."
    )
    checkpoints = _checkpoint_rows(evidence)
    residual_lines = [
        "| Phase | Mean normalized RMS | Mean normalized peak | Mean active channels | Nonzero in every paired seed |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in PHASE_IDS:
        rows = [row for row in evidence.residual_activity if row["phase"] == phase]
        residual_lines.append(
            f"| {phase} | {_fmt(np.mean([float(row['normalized_residual_rms']) for row in rows]))} | "
            f"{_fmt(np.mean([float(row['normalized_residual_peak']) for row in rows]))} | "
            f"{_fmt(np.mean([float(row['active_channel_count']) for row in rows]))} | "
            f"{'yes' if all(_boolean(row['nonzero'], 'nonzero') for row in rows) else 'no'} |"
        )
    reward_lines = [
        "| Phase | Mean total reward | " + " | ".join(DENSE_FAMILIES) + " |",
        "|---|---:|" + "---:|" * len(DENSE_FAMILIES),
    ]
    for phase in PHASE_IDS:
        rows = [row for row in evidence.reward_contribution if row["phase"] == phase]
        values = [
            _fmt(np.mean([float(row[f"{name}_sum"]) for row in rows]))
            for name in DENSE_FAMILIES
        ]
        reward_lines.append(
            f"| {phase} | {_fmt(np.mean([float(row['total_reward_sum']) for row in rows]))} | "
            + " | ".join(values)
            + " |"
        )
    manifest_section = _training_manifest_section(evidence.training_manifest)
    lines = [
        "# PPO Training Report",
        "",
        f"Promotion status: **{status}**",
        "",
        "## Evidence boundary",
        "",
        "This offline report reads immutable paired evaluation exports. It does not use reward curves as proof of physical improvement and does not execute Isaac or training.",
        "",
        "## Training manifest",
        "",
        manifest_section,
        "",
        "## Evaluation population",
        "",
        f"Paired seeds: {', '.join(str(seed) for seed in evidence.seeds)}.",
        "",
        "| Role | Checkpoint | Episodes | Successes | Mean duration (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for role in ("baseline", "candidate"):
        row = checkpoints[role]
        lines.append(
            f"| {role} | {_fmt(row.get('checkpoint'))} | {_fmt(row.get('episode_count'))} | "
            f"{_fmt(row.get('task_success_count'))} | {_fmt(row.get('mean_duration_s'))} |"
        )
    lines.extend(
        [
            "",
            "## PPO residual activity",
            "",
            *residual_lines,
            "",
            "## Reward contribution audit",
            "",
            *reward_lines,
            "",
            "## Reproducibility inputs",
            "",
            _hash_table(evidence.input_paths),
            "",
        ]
    )
    return "\n".join(lines)


def _improvement_report(evidence: _ReportingEvidence) -> str:
    promotion = evidence.promotion["promotion"]
    if evidence.promoted:
        verdict = (
            "**PASSED.** Every recorded promotion gate passed; the supplied evidence "
            "supports describing this checkpoint as improved over the paired FSM baseline."
        )
        fraction_heading = "Accepted improvement fraction"
    else:
        verdict = (
            "**NOT PASSED.** This document makes no PPO-improvement claim. Values below "
            "are neutral candidate-versus-baseline comparisons only."
        )
        fraction_heading = "Evaluator comparison fraction (not accepted)"
    checkpoints = _checkpoint_rows(evidence)
    base = checkpoints["baseline"]
    candidate = checkpoints["candidate"]
    metrics = (
        ("Duration (s)", "mean_duration_s"),
        ("Overall pitch-rate RMS (rad/s)", "mean_overall_pitch_rate_rms_rad_s"),
        ("Overall roll-rate RMS (rad/s)", "mean_overall_roll_rate_rms_rad_s"),
        ("Placement contact impulse (N·s)", "mean_placement_contact_impulse_n_s"),
        ("Home-recovery action jerk RMS", "mean_home_recovery_action_jerk_rms"),
    )
    lines = [
        "# PPO Improvement Decision Report",
        "",
        verdict,
        "",
        f"Candidate label: `{evidence.candidate_label}`. First failed gate: `{evidence.promotion.get('first_failed_gate')}`.",
        "",
        "## Promotion gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{row['gate']}` | {'yes' if row['passed'] else 'no'} |"
        for row in evidence.promotion["checks_in_evaluation_order"]
    )
    lines.extend(
        [
            "",
            "## Overall paired comparison",
            "",
            f"Global stability {fraction_heading.lower()}: `{_fmt(promotion.get('global_stability_improvement_fraction'))}`.",
            "",
            "| Metric | FSM baseline | PPO candidate | Candidate − FSM |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in metrics:
        baseline_value = _finite(base, key, "baseline checkpoint")
        candidate_value = _finite(candidate, key, "candidate checkpoint")
        lines.append(
            f"| {label} | {_fmt(baseline_value)} | {_fmt(candidate_value)} | {_fmt(candidate_value - baseline_value)} |"
        )
    lines.extend(
        [
            "",
            "## Phase score comparison",
            "",
            f"| Phase | Priority | {fraction_heading} |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['phase']} | {'yes' if row['phase'] in PRIORITY_PHASES else 'no'} | "
        f"{_fmt(row['primary_phase_score_improvement_fraction'])} |"
        for row in evidence.phase_comparison
    )
    lines.extend(
        [
            "",
            "## Safety and task outcomes",
            "",
            "| Role | Successes | Body collisions | Wheel-only climbs | Mean duration (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for role in ("baseline", "candidate"):
        row = checkpoints[role]
        lines.append(
            f"| {role} | {_fmt(row['task_success_count'])} | {_fmt(row['body_collision_count'])} | "
            f"{_fmt(row['wheel_only_climb_count'])} | {_fmt(row['mean_duration_s'])} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            *[f"- [{name}](../plots/{name})" for name in PLOT_FILENAMES],
            "",
            "The frequency-domain plot reports the exported >3 Hz energy fraction. Raw PSD bins are not present in the canonical phase CSV and are deliberately not reconstructed.",
            "",
        ]
    )
    return "\n".join(lines)


def _paths(output_root: Path) -> FinalReportingPaths:
    plots = output_root / "plots"
    reports = output_root / "reports"
    return FinalReportingPaths(
        output_root=output_root,
        plots_directory=plots,
        reports_directory=reports,
        overall_pitch_rate_comparison=plots / PLOT_FILENAMES[0],
        phase_wise_pitch_rate_rms=plots / PLOT_FILENAMES[1],
        phase_wise_roll_pitch_rms=plots / PLOT_FILENAMES[2],
        fr_placement_contact_impulse=plots / PLOT_FILENAMES[3],
        p08_transfer_post_capture_settling=plots / PLOT_FILENAMES[4],
        rl_lift_body_attitude=plots / PLOT_FILENAMES[5],
        p13_home_pose_convergence=plots / PLOT_FILENAMES[6],
        residual_action_by_phase=plots / PLOT_FILENAMES[7],
        residual_frequency_spectrum=plots / PLOT_FILENAMES[8],
        fsm_vs_ppo_phase_duration=plots / PLOT_FILENAMES[9],
        phase_design_report=reports / REPORT_FILENAMES[0],
        training_report=reports / REPORT_FILENAMES[1],
        improvement_report=reports / REPORT_FILENAMES[2],
    )


def _publish_idempotently(publications: Mapping[Path, bytes]) -> None:
    if len(publications) != len(set(publications)):
        raise FinalReportingError("final report output paths are duplicated")
    for path, content in publications.items():
        if path.exists() and (
            not path.is_file() or path.read_bytes() != content
        ):
            raise FinalReportingError(
                f"refusing to overwrite non-identical final report artifact: {path}"
            )
    for path, content in publications.items():
        if path.exists():
            continue
        try:
            _atomic_bytes(path, content)
        except ArtifactError as exc:
            if not path.is_file() or path.read_bytes() != content:
                raise FinalReportingError(str(exc)) from exc


def generate_final_reporting_bundle(
    metrics_directory: str | Path,
    output_root: str | Path,
    *,
    phase_objectives_config: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
    phase_action_config: str | Path = DEFAULT_PHASE_ACTION_CONFIG_V2,
    reward_config: str | Path = DEFAULT_REWARD_PATH_V2,
    training_manifest: str | Path | None = None,
) -> FinalReportingPaths:
    """Generate ten plots and three reports from verified offline evidence.

    Existing byte-identical outputs are accepted without changing timestamps.
    Any non-identical target aborts before a new output is published.
    """

    root = Path(output_root).resolve()
    if root.exists() and not root.is_dir():
        raise FinalReportingError(f"output root is not a directory: {root}")
    evidence = _load_evidence(
        Path(metrics_directory),
        phase_objectives_config=Path(phase_objectives_config),
        phase_action_config=Path(phase_action_config),
        reward_config=Path(reward_config),
        training_manifest=(
            None if training_manifest is None else Path(training_manifest)
        ),
    )
    paths = _paths(root)
    plots = _plot_payloads(evidence)
    publications: dict[Path, bytes] = {
        getattr(paths, Path(filename).stem): content
        for filename, content in plots.items()
    }
    publications.update(
        {
            paths.phase_design_report: _phase_design_report(evidence).encode("utf-8"),
            paths.training_report: _training_report(evidence).encode("utf-8"),
            paths.improvement_report: _improvement_report(evidence).encode("utf-8"),
        }
    )
    if tuple(publications) != paths.files():
        raise FinalReportingError("internal final-report output order changed")
    _publish_idempotently(publications)
    return paths


__all__ = [
    "PLOT_FILENAMES",
    "REPORT_FILENAMES",
    "REPORTING_SCHEMA",
    "FinalReportingError",
    "FinalReportingPaths",
    "generate_final_reporting_bundle",
]
