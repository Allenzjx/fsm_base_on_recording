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

from .artifacts import ArtifactError, _atomic_bytes
from .evaluation_artifacts import (
    BASELINE_VALIDATION_SEEDS,
    BASELINE_EPISODE_FILENAME,
    BASELINE_PHASE_FILENAME,
    CANDIDATE_EPISODE_FILENAME,
    CANDIDATE_PHASE_FILENAME,
    CHECKPOINT_COMPARISON_FILENAME,
    EVALUATION_ARTIFACT_SCHEMA,
    FINAL_LIFECYCLE_BUNDLE_KIND,
    FINAL_LIFECYCLE_ROLES,
    PHASE_COMPARISON_FILENAME,
    PROMOTION_DECISION_FILENAME,
    RESIDUAL_ACTIVITY_FILENAME,
    REWARD_CONTRIBUTION_FILENAME,
    TERMINATION_SUMMARY_FILENAME,
    EvaluationArtifactError,
    _capture_file_record,
    _validate_current_committed_runtime_identity,
    _paths_overlap,
    _record_unchanged,
    _require_no_reparse_components,
    validate_final_lifecycle_aggregate_evidence,
)
from .phase_action_masks_v2 import DEFAULT_PHASE_ACTION_CONFIG_V2
from .phase_objectives import (
    DEFAULT_PHASE_OBJECTIVES_PATH,
    DENSE_FAMILIES,
    PhaseObjectiveError,
    load_phase_objectives,
)
from .reward_v2 import DEFAULT_REWARD_PATH_V2
from .reward_migration import (
    DEFAULT_MIGRATION_PATH,
    RewardMigrationError,
    load_reward_migration,
)
from .stability_metrics import PHASE_IDS, PRIORITY_PHASES
from .training_orchestration import (
    TRAINING_ORCHESTRATION_SCHEMA,
    TrainingOrchestrationError,
    validate_training_orchestration_manifest,
)


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
    "residual_spectral_energy_fraction_0p0_0p5_hz",
    "residual_spectral_energy_fraction_0p5_1p0_hz",
    "residual_spectral_energy_fraction_1p0_2p0_hz",
    "residual_spectral_energy_fraction_2p0_3p0_hz",
    "residual_spectral_energy_fraction_3p0_nyquist_hz",
)
_RESIDUAL_SPECTRAL_BANDS = (
    "residual_spectral_energy_fraction_0p0_0p5_hz",
    "residual_spectral_energy_fraction_0p5_1p0_hz",
    "residual_spectral_energy_fraction_1p0_2p0_hz",
    "residual_spectral_energy_fraction_2p0_3p0_hz",
    "residual_spectral_energy_fraction_3p0_nyquist_hz",
)
_RESIDUAL_SPECTRAL_LABELS = ("0–0.5", "0.5–1", "1–2", "2–3", "3–Nyquist")
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
_FINAL_ROLE_CHECKPOINT_FILENAMES = {
    "checkpoint_initial": "checkpoint_initial_zero_residual.pt",
    "checkpoint_smoke": "checkpoint_smoke.pt",
    "checkpoint_best": "checkpoint_best_validation.pt",
    "checkpoint_improved": "checkpoint_improved.pt",
}


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
    reward_migration: Mapping[str, Any]
    training_orchestration: Mapping[str, Any]
    input_paths: tuple[Path, ...]
    input_records: tuple[Mapping[str, Any], ...]
    seeds: tuple[int, ...]
    baseline_label: str
    candidate_label: str
    promoted: bool
    duration_pair_diagnostics: Mapping[str, Any]
    checkpoint_roles: tuple[str, ...]
    final_lifecycle: bool


def _captured_file(
    path: Path, *, label: str, allow_empty: bool = False
) -> tuple[dict[str, Any], bytes]:
    try:
        return _capture_file_record(path, label=label, allow_empty=allow_empty)
    except EvaluationArtifactError as exc:
        raise FinalReportingError(str(exc)) from exc


def _read_csv(
    path: Path, required: Sequence[str]
) -> tuple[tuple[Mapping[str, str], ...], Mapping[str, Any]]:
    record, content = _captured_file(path, label="required metrics CSV")
    try:
        decoded = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        fields = tuple(reader.fieldnames or ())
        if not fields or len(fields) != len(set(fields)):
            raise FinalReportingError(f"CSV header is empty or duplicated: {path}")
        missing = sorted(set(required).difference(fields))
        if missing:
            raise FinalReportingError(f"CSV {path.name} is missing columns {missing}")
        rows = tuple(dict(row) for row in reader)
    except (UnicodeError, csv.Error) as exc:
        raise FinalReportingError(f"cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise FinalReportingError(f"CSV has no evidence rows: {path}")
    if any(None in row for row in rows):
        raise FinalReportingError(f"CSV contains fields beyond its declared header: {path}")
    return rows, record


def _read_json(
    path: Path, label: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    record, content = _captured_file(path, label=label)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FinalReportingError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FinalReportingError(f"{label} must be a JSON object: {path}")
    return value, record


def _read_yaml(
    path: Path, label: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    record, content = _captured_file(path, label=label)
    try:
        value = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise FinalReportingError(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FinalReportingError(f"{label} must be a YAML mapping: {path}")
    return value, record


def _finite(row: Mapping[str, Any], name: str, context: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalReportingError(f"{context}.{name} must be numeric") from exc
    if not math.isfinite(value):
        raise FinalReportingError(f"{context}.{name} must be finite")
    return value


def _optional_finite(row: Mapping[str, Any], name: str, context: str) -> float | None:
    """Validate a finite CSV value when present; blank means unavailable."""

    if str(row.get(name, "")).strip() == "":
        return None
    return _finite(row, name, context)


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


def _verified_file_from_row(
    row: Mapping[str, Any],
    *,
    path_name: str,
    hash_name: str,
    context: str,
) -> tuple[Path, Mapping[str, Any]]:
    raw_path = str(row.get(path_name, "")).strip()
    expected_hash = str(row.get(hash_name, "")).strip().lower()
    if not raw_path or len(expected_hash) != 64:
        raise FinalReportingError(f"{context} omits {path_name}/{hash_name}")
    record, _ = _captured_file(Path(raw_path), label=context)
    path = Path(str(record["path"]))
    if record["sha256"] != expected_hash:
        raise FinalReportingError(f"{context} {hash_name} is stale")
    return path, record


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


def _validate_residual_spectral_rows(
    rows: Sequence[Mapping[str, str]], *, label: str
) -> None:
    """Require a complete normalized five-band residual spectrum per phase row."""

    for offset, row in enumerate(rows):
        values = tuple(
            _finite(row, field, f"{label}[{offset}]")
            for field in _RESIDUAL_SPECTRAL_BANDS
        )
        if any(value < 0.0 or value > 1.0 + 1.0e-9 for value in values):
            raise FinalReportingError(
                f"{label}[{offset}] residual spectral fractions must be in [0, 1]"
            )
        total = sum(values)
        if not (
            math.isclose(total, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            or math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        ):
            raise FinalReportingError(
                f"{label}[{offset}] residual spectral fractions must sum to zero "
                "for a zero residual or approximately one"
            )


def _duration_pair_diagnostics_from_episode_rows(
    baseline: Mapping[int, Mapping[str, str]],
    candidate: Mapping[int, Mapping[str, str]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Independently recompute the matched-seed 15% duration gate."""

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_duration = _finite(
            baseline[int(seed)], "duration_s", f"baseline episode seed={seed}"
        )
        candidate_duration = _finite(
            candidate[int(seed)], "duration_s", f"candidate episode seed={seed}"
        )
        if baseline_duration <= 0.0 or candidate_duration < 0.0:
            raise FinalReportingError(
                "paired duration evidence requires a positive baseline duration "
                "and non-negative candidate duration"
            )
        allowed = baseline_duration * 1.15
        rows.append(
            {
                "seed": int(seed),
                "baseline_duration_s": baseline_duration,
                "candidate_duration_s": candidate_duration,
                "allowed_duration_s": allowed,
                "candidate_to_baseline_ratio": candidate_duration
                / baseline_duration,
                "passed": candidate_duration <= allowed,
            }
        )
    failed = [row for row in rows if row["passed"] is not True]
    worst = max(rows, key=lambda row: row["candidate_to_baseline_ratio"])
    baseline_mean = float(np.mean([row["baseline_duration_s"] for row in rows]))
    candidate_mean = float(np.mean([row["candidate_duration_s"] for row in rows]))
    return {
        "gate_semantics": "all_matched_seed_pairs_candidate_le_1p15_times_baseline",
        "paired_rows": tuple(rows),
        "baseline_mean_duration_s": baseline_mean,
        "candidate_mean_duration_s": candidate_mean,
        "mean_candidate_to_baseline_ratio": candidate_mean / baseline_mean,
        "first_violating_seed": None if not failed else int(failed[0]["seed"]),
        "worst_seed": int(worst["seed"]),
        "worst_candidate_to_baseline_ratio": float(
            worst["candidate_to_baseline_ratio"]
        ),
    }


def _validate_declared_duration_pair_diagnostics(
    declared: Any,
    recomputed: Mapping[str, Any],
) -> None:
    if not isinstance(declared, Mapping):
        raise FinalReportingError(
            "promotion omits paired per-seed duration diagnostics"
        )
    if declared.get("gate_semantics") != recomputed["gate_semantics"]:
        raise FinalReportingError("promotion duration gate semantics changed")
    declared_rows = declared.get("paired_rows")
    if not isinstance(declared_rows, (list, tuple)) or len(declared_rows) != len(
        recomputed["paired_rows"]
    ):
        raise FinalReportingError("promotion duration paired rows are incomplete")
    for expected, actual in zip(
        recomputed["paired_rows"], declared_rows, strict=True
    ):
        if not isinstance(actual, Mapping):
            raise FinalReportingError("promotion duration pair row is invalid")
        if int(actual.get("seed", -1)) != expected["seed"] or actual.get(
            "passed"
        ) is not expected["passed"]:
            raise FinalReportingError("promotion duration pair verdict is stale")
        for name in (
            "baseline_duration_s",
            "candidate_duration_s",
            "allowed_duration_s",
            "candidate_to_baseline_ratio",
        ):
            value = _finite(actual, name, "promotion duration pair")
            if not math.isclose(
                value, float(expected[name]), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise FinalReportingError(
                    f"promotion duration pair {name} differs from episode CSV"
                )
    for name in (
        "baseline_mean_duration_s",
        "candidate_mean_duration_s",
        "mean_candidate_to_baseline_ratio",
        "worst_candidate_to_baseline_ratio",
    ):
        value = _finite(declared, name, "promotion duration diagnostics")
        if not math.isclose(
            value, float(recomputed[name]), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise FinalReportingError(
                f"promotion duration diagnostic {name} differs from episode CSV"
            )
    for name in ("first_violating_seed", "worst_seed"):
        if declared.get(name) != recomputed[name]:
            raise FinalReportingError(
                f"promotion duration diagnostic {name} differs from episode CSV"
            )


def _validate_promotion(
    payload: Mapping[str, Any],
    seeds: Sequence[int],
    duration_pair_diagnostics: Mapping[str, Any],
    *,
    require_declared_duration_diagnostics: bool,
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
    duration_gate = all(
        row["passed"] is True
        for row in duration_pair_diagnostics["paired_rows"]
    )
    if checks["duration_not_over_fsm_by_15pct"] is not duration_gate:
        raise FinalReportingError(
            "promotion duration gate disagrees with raw paired episode CSV"
        )
    declared_duration = promotion.get("duration_pair_diagnostics")
    if declared_duration is not None:
        _validate_declared_duration_pair_diagnostics(
            declared_duration, duration_pair_diagnostics
        )
    elif require_declared_duration_diagnostics:
        raise FinalReportingError(
            "final promotion evidence omits paired duration diagnostics"
        )
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
    audits = reward.get("audits")
    if not isinstance(audits, Mapping):
        raise FinalReportingError("reward v2 omits dominance-audit thresholds")
    expected_audits = {
        "maximum_single_dense_family_fraction": 0.70,
        "maximum_residual_regularization_fraction": 0.20,
        "minimum_absolute_dense_return": 1.0e-12,
        "standing_still_max_reward": 0.0,
    }
    for name, expected in expected_audits.items():
        value = _finite(audits, name, "reward.audits")
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-15):
            raise FinalReportingError(
                f"reward dominance audit {name} differs from the locked v2 contract"
            )
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
    for phase_id in ("P08", "P11"):
        row = phase_rows[phase_id]
        schedule = row.get("transfer_schedule")
        if row.get("stability_mode") != "TRANSFER_AWARE" or not isinstance(
            schedule, Mapping
        ):
            raise FinalReportingError(
                f"{phase_id} must retain its transfer-aware stability schedule"
            )
        weights = schedule.get("weights")
        if not isinstance(weights, Mapping) or tuple(weights) != (
            "active",
            "capture",
            "settle",
        ):
            raise FinalReportingError(
                f"{phase_id} transfer-aware schedule is incomplete or reordered"
            )


def _load_evidence(
    metrics_directory: Path,
    *,
    phase_objectives_config: Path,
    phase_action_config: Path,
    reward_config: Path,
    reward_migration_config: Path,
    training_orchestration_manifest: Path | None,
    allow_nonfinal_two_role: bool = False,
) -> _ReportingEvidence:
    try:
        _require_no_reparse_components(metrics_directory, label="metrics directory")
    except EvaluationArtifactError as exc:
        raise FinalReportingError(str(exc)) from exc
    directory = metrics_directory.resolve()
    promotion_path = directory / PROMOTION_DECISION_FILENAME
    promotion, promotion_record = _read_json(promotion_path, "promotion decision")
    final_lifecycle = promotion.get("bundle_kind") == FINAL_LIFECYCLE_BUNDLE_KIND
    if not final_lifecycle and not allow_nonfinal_two_role:
        raise FinalReportingError(
            "final reporting requires the strict five-role lifecycle evaluation bundle"
        )
    if final_lifecycle and tuple(promotion.get("final_lifecycle_roles", ())) != FINAL_LIFECYCLE_ROLES:
        raise FinalReportingError(
            "final lifecycle promotion evidence omits the exact five-role order"
        )
    if final_lifecycle and training_orchestration_manifest is None:
        raise FinalReportingError(
            "final reporting requires an explicit prefinal training orchestration manifest"
        )
    checkpoint_required = (
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
        *(
            (
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_manifest_path",
                "checkpoint_manifest_sha256",
                "evaluation_aggregate_path",
                "evaluation_aggregate_sha256",
                "paired_seeds",
            )
            if final_lifecycle
            else ()
        ),
    )
    csv_specs = {
        "baseline_episode": (
            BASELINE_EPISODE_FILENAME,
            ("checkpoint", "seed", "task_success", *_EPISODE_METRICS),
        ),
        "baseline_phase": (
            BASELINE_PHASE_FILENAME,
            (
                "checkpoint",
                "seed",
                "phase",
                "residual_spectrum_normalization",
                *_PHASE_METRICS,
            ),
        ),
        "candidate_episode": (
            CANDIDATE_EPISODE_FILENAME,
            ("checkpoint", "seed", "task_success", *_EPISODE_METRICS),
        ),
        "candidate_phase": (
            CANDIDATE_PHASE_FILENAME,
            (
                "checkpoint",
                "seed",
                "phase",
                "residual_spectrum_normalization",
                *_PHASE_METRICS,
            ),
        ),
        "checkpoint_comparison": (
            CHECKPOINT_COMPARISON_FILENAME,
            checkpoint_required,
        ),
        "phase_comparison": (
            PHASE_COMPARISON_FILENAME,
            (
                "phase",
                "primary_phase_score_improvement_fraction",
                *(
                    ("baseline_checkpoint", "improved_checkpoint")
                    if final_lifecycle
                    else ()
                ),
            ),
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
    loaded: dict[str, tuple[Mapping[str, str], ...]] = {}
    input_records: list[Mapping[str, Any]] = []
    for name, (filename, required) in csv_specs.items():
        rows, record = _read_csv(directory / filename, required)
        loaded[name] = rows
        input_records.append(record)
    if final_lifecycle:
        declared_artifacts = promotion.get("artifact_files")
        expected_names = {filename for filename, _ in csv_specs.values()}
        if not isinstance(declared_artifacts, Mapping) or set(
            declared_artifacts
        ) != expected_names:
            raise FinalReportingError(
                "final lifecycle promotion omits exact metric artifact hashes"
            )
        records_by_name = {
            Path(str(record["path"])).name: record for record in input_records
        }
        for filename in expected_names:
            declared = declared_artifacts[filename]
            captured = records_by_name[filename]
            if not isinstance(declared, Mapping) or any(
                declared.get(key) != captured[key]
                for key in ("path", "bytes", "sha256")
            ):
                raise FinalReportingError(
                    f"final lifecycle metric artifact hash is stale: {filename}"
                )
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
    _validate_residual_spectral_rows(
        loaded["candidate_phase"], label="improved phase metrics"
    )
    if any(
        row.get("residual_spectrum_normalization") != "phase_scale_full12"
        for row in loaded["candidate_phase"]
    ):
        raise FinalReportingError(
            "improved residual spectra are not dimensionless phase-scale-normalized Full12 evidence"
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

    duration_pair_diagnostics = _duration_pair_diagnostics_from_episode_rows(
        baseline_index, candidate_index, seeds
    )
    promoted, baseline_label, candidate_label = _validate_promotion(
        promotion,
        seeds,
        duration_pair_diagnostics,
        require_declared_duration_diagnostics=final_lifecycle,
    )
    if final_lifecycle:
        if seeds != BASELINE_VALIDATION_SEEDS:
            raise FinalReportingError(
                "final lifecycle reporting requires validation seeds 2001-2005"
            )
        if promoted is not True or promotion.get("frozen_hashes_unchanged") is not True:
            raise FinalReportingError(
                "final lifecycle reporting requires an explicitly promoted five-role bundle"
            )
        if (baseline_label, candidate_label) != ("pure_fsm", "checkpoint_improved"):
            raise FinalReportingError(
                "final lifecycle promotion must compare pure_fsm with checkpoint_improved"
            )
        if any(
            row.get("baseline_checkpoint") != baseline_label
            or row.get("improved_checkpoint") != candidate_label
            for row in phase_rows
        ):
            raise FinalReportingError(
                "phase comparison labels are not pure_fsm versus checkpoint_improved"
            )
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
    checkpoint_roles = (
        FINAL_LIFECYCLE_ROLES if final_lifecycle else ("baseline", "candidate")
    )
    actual_role_order = tuple(
        str(row.get("role")) for row in loaded["checkpoint_comparison"]
    )
    if actual_role_order != checkpoint_roles:
        raise FinalReportingError(
            "checkpoint comparison does not contain the required role order"
        )
    checkpoints = {
        str(row.get("role")): row for row in loaded["checkpoint_comparison"]
    }
    if final_lifecycle:
        if any(checkpoints[role].get("checkpoint") != role for role in checkpoint_roles):
            raise FinalReportingError(
                "final checkpoint comparison role/checkpoint labels disagree"
            )
    elif (
        checkpoints["baseline"].get("checkpoint") != baseline_label
        or checkpoints["candidate"].get("checkpoint") != candidate_label
    ):
        raise FinalReportingError("checkpoint comparison labels disagree with promotion JSON")
    for role, row in checkpoints.items():
        if final_lifecycle or "paired_seeds" in row:
            try:
                row_seeds = tuple(int(seed) for seed in json.loads(row["paired_seeds"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise FinalReportingError(
                    f"checkpoint comparison {role}.paired_seeds is invalid"
                ) from exc
            if row_seeds != seeds:
                raise FinalReportingError(
                    f"checkpoint comparison {role}.paired_seeds disagree with evaluation rows"
                )
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
        ):
            _finite(row, name, f"checkpoint comparison {role}")
        for name in (
            "mean_placement_contact_impulse_n_s",
            "mean_home_recovery_action_jerk_rms",
        ):
            if final_lifecycle and role in {"checkpoint_initial", "checkpoint_smoke"}:
                _optional_finite(row, name, f"checkpoint comparison {role}")
            else:
                _finite(row, name, f"checkpoint comparison {role}")
    provenance_paths: list[Path] = []
    if final_lifecycle:
        lifecycle = promotion.get("final_lifecycle_evidence")
        if not isinstance(lifecycle, Mapping) or set(lifecycle) != set(FINAL_LIFECYCLE_ROLES):
            raise FinalReportingError(
                "promotion JSON omits exact five-role lifecycle provenance"
            )
        checkpoint_files: dict[str, Path] = {}
        checkpoint_manifest_files: dict[str, Path] = {}
        checkpoint_manifest_payloads: dict[str, Mapping[str, Any]] = {}
        aggregate_files: list[Path] = []
        canonical_directories: list[Path] = []
        runtime_identities: list[Mapping[str, Any]] = []
        lifecycle_training_seeds: list[int] = []
        lifecycle_creation_identities: dict[str, tuple[Path, str]] = {}
        for role in FINAL_LIFECYCLE_ROLES:
            row = checkpoints[role]
            record = lifecycle[role]
            if not isinstance(record, Mapping):
                raise FinalReportingError(f"final lifecycle provenance {role} is invalid")
            aggregate_path, aggregate_record = _verified_file_from_row(
                row,
                path_name="evaluation_aggregate_path",
                hash_name="evaluation_aggregate_sha256",
                context=f"checkpoint comparison {role} aggregate",
            )
            try:
                validated = validate_final_lifecycle_aggregate_evidence(
                    aggregate_path, role=role
                )
            except EvaluationArtifactError as exc:
                raise FinalReportingError(
                    f"final lifecycle aggregate {role} is invalid: {exc}"
                ) from exc
            provenance_paths.append(aggregate_path)
            input_records.append(aggregate_record)
            aggregate_files.append(aggregate_path)
            expected_groups = [dict(group) for group in validated.source_groups]
            expected_supporting = [dict(item) for item in validated.supporting_files]
            if (
                Path(str(record.get("aggregate_path", ""))).resolve()
                != validated.aggregate_path
                or record.get("aggregate_sha256") != validated.aggregate_sha256
                or row.get("evaluation_aggregate_sha256")
                != validated.aggregate_sha256
                or tuple(record.get("paired_seeds", ())) != validated.seeds
                or tuple(Path(str(path)).resolve() for path in record.get("canonical_episode_dirs", ()))
                != validated.canonical_episode_dirs
                or record.get("source_groups") != expected_groups
                or record.get("supporting_files") != expected_supporting
                or record.get("committed_runtime_identity")
                != dict(validated.committed_runtime_identity)
                or record.get("training_seed") != validated.training_seed
                or record.get("source_git_commit")
                != validated.source_git_commit
                or record.get("committed_runtime_content_sha256")
                != validated.committed_runtime_content_sha256
                or record.get("creation_runtime_identity_sha256")
                != validated.creation_runtime_identity_sha256
                or (
                    validated.creation_runtime_identity_path is None
                    and record.get("creation_runtime_identity_path") is not None
                )
                or (
                    validated.creation_runtime_identity_path is not None
                    and Path(
                        str(record.get("creation_runtime_identity_path", ""))
                    ).resolve()
                    != validated.creation_runtime_identity_path
                )
            ):
                raise FinalReportingError(
                    f"checkpoint comparison and promotion provenance disagree for {role}"
                )
            canonical_directories.extend(validated.canonical_episode_dirs)
            runtime_identities.append(validated.committed_runtime_identity)
            if validated.training_seed is not None:
                lifecycle_training_seeds.append(validated.training_seed)
            if (
                validated.creation_runtime_identity_path is not None
                and validated.creation_runtime_identity_sha256 is not None
            ):
                lifecycle_creation_identities[role] = (
                    validated.creation_runtime_identity_path,
                    validated.creation_runtime_identity_sha256,
                )
            for group in validated.source_groups:
                input_records.extend((group["run_manifest"], group["worker_result"]))
                input_records.extend(group["canonical_files"])
            input_records.extend(validated.supporting_files)
            if role == "pure_fsm":
                if any(
                    str(row.get(name, "")).strip()
                    for name in (
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_manifest_path",
                        "checkpoint_manifest_sha256",
                    )
                ):
                    raise FinalReportingError(
                        "pure_fsm checkpoint comparison must not name checkpoint bytes"
                    )
                if any(
                    record.get(name) is not None
                    for name in (
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_manifest_path",
                        "checkpoint_manifest_sha256",
                    )
                ):
                    raise FinalReportingError(
                        "pure_fsm promotion provenance must not name checkpoint bytes"
                    )
                continue
            checkpoint_path, checkpoint_record = _verified_file_from_row(
                row,
                path_name="checkpoint_path",
                hash_name="checkpoint_sha256",
                context=f"checkpoint comparison {role}",
            )
            manifest_path, manifest_record = _verified_file_from_row(
                row,
                path_name="checkpoint_manifest_path",
                hash_name="checkpoint_manifest_sha256",
                context=f"checkpoint comparison {role} manifest",
            )
            if checkpoint_path.name != _FINAL_ROLE_CHECKPOINT_FILENAMES[role]:
                raise FinalReportingError(
                    f"checkpoint comparison {role} uses the wrong checkpoint filename"
                )
            if manifest_path.name != f"{checkpoint_path.stem}_manifest.json":
                raise FinalReportingError(
                    f"checkpoint comparison {role} uses the wrong manifest filename"
                )
            if (
                Path(str(record.get("checkpoint_path", ""))).resolve()
                != checkpoint_path
                or record.get("checkpoint_sha256") != row.get("checkpoint_sha256")
                or Path(str(record.get("checkpoint_manifest_path", ""))).resolve()
                != manifest_path
                or record.get("checkpoint_manifest_sha256")
                != row.get("checkpoint_manifest_sha256")
                or validated.checkpoint_path != checkpoint_path
                or validated.checkpoint_sha256 != checkpoint_record["sha256"]
                or validated.checkpoint_manifest_path != manifest_path
                or validated.checkpoint_manifest_sha256 != manifest_record["sha256"]
            ):
                raise FinalReportingError(
                    f"checkpoint comparison and promotion checkpoint evidence disagree for {role}"
                )
            checkpoint_files[role] = checkpoint_path
            checkpoint_manifest_files[role] = manifest_path
            manifest_payload, recaptured_manifest = _read_json(
                manifest_path, f"checkpoint comparison {role} manifest"
            )
            if recaptured_manifest != manifest_record:
                raise FinalReportingError(
                    f"checkpoint comparison {role} manifest changed during capture"
                )
            checkpoint_manifest_payloads[role] = manifest_payload
            input_records.extend((checkpoint_record, manifest_record))
            provenance_paths.extend((checkpoint_path, manifest_path))
        if len(set(aggregate_files)) != 5 or len(set(canonical_directories)) != 25:
            raise FinalReportingError(
                "final lifecycle provenance must bind five aggregates and 25 distinct episodes"
            )
        if any(
            dict(identity) != dict(runtime_identities[0])
            for identity in runtime_identities[1:]
        ):
            raise FinalReportingError(
                "final lifecycle roles do not share one committed runtime identity"
            )
        if len(lifecycle_training_seeds) != 4 or len(set(lifecycle_training_seeds)) != 1:
            raise FinalReportingError(
                "final lifecycle PPO checkpoints do not share one training seed"
            )
        try:
            _validate_current_committed_runtime_identity(runtime_identities[0])
        except EvaluationArtifactError as exc:
            raise FinalReportingError(str(exc)) from exc
        if (
            checkpoints["checkpoint_best"].get("checkpoint_sha256")
            != checkpoints["checkpoint_improved"].get("checkpoint_sha256")
        ):
            raise FinalReportingError(
                "checkpoint_improved bytes differ from checkpoint_best promotion source"
            )
        best_manifest_payload = checkpoint_manifest_payloads["checkpoint_best"]
        improved_manifest_payload = checkpoint_manifest_payloads[
            "checkpoint_improved"
        ]
        cadence_decision_path = Path(
            str(improved_manifest_payload.get("promotion_decision", ""))
        )
        try:
            _require_no_reparse_components(
                cadence_decision_path,
                label="checkpoint_improved cadence promotion decision",
            )
        except EvaluationArtifactError as exc:
            raise FinalReportingError(str(exc)) from exc
        cadence_decision_path = cadence_decision_path.resolve()
        cadence_decision, cadence_decision_record = _read_json(
            cadence_decision_path, "checkpoint_improved cadence promotion decision"
        )
        if (
            improved_manifest_payload.get("promotion_decision_sha256")
            != cadence_decision_record["sha256"]
            or best_manifest_payload.get("promotion_decision")
            != improved_manifest_payload.get("promotion_decision")
            or best_manifest_payload.get("promotion_decision_sha256")
            != improved_manifest_payload.get("promotion_decision_sha256")
        ):
            raise FinalReportingError(
                "checkpoint_best/improved do not inherit one exact cadence promotion decision"
            )
        input_records.append(cadence_decision_record)
        if (
            Path(str(promotion.get("candidate_checkpoint_path", ""))).resolve()
            != checkpoint_files["checkpoint_improved"]
            or promotion.get("candidate_checkpoint_sha256")
            != checkpoints["checkpoint_improved"].get("checkpoint_sha256")
            or Path(
                str(promotion.get("candidate_checkpoint_manifest_path", ""))
            ).resolve()
            != checkpoint_manifest_files["checkpoint_improved"]
            or promotion.get("candidate_checkpoint_manifest_sha256")
            != checkpoints["checkpoint_improved"].get(
                "checkpoint_manifest_sha256"
            )
        ):
            raise FinalReportingError(
                "promotion candidate checkpoint differs from checkpoint_improved"
            )
    termination_labels = (
        checkpoint_roles if final_lifecycle else (baseline_label, candidate_label)
    )
    expected_termination = {
        (label, seed)
        for label in termination_labels
        for seed in seeds
    }
    actual_termination = {
        (str(row["checkpoint"]), _integer(row, "seed", "termination summary"))
        for row in loaded["termination_summary"]
    }
    if actual_termination != expected_termination or len(loaded["termination_summary"]) != len(expected_termination):
        raise FinalReportingError(
            "termination summary does not match every paired checkpoint role"
        )
    for row in loaded["termination_summary"]:
        context = f"termination summary {row['checkpoint']}/{row['seed']}"
        for name in ("task_success", "body_collision", "wheel_only_climb"):
            _boolean(row[name], f"{context}.{name}")
        if _finite(row, "duration_s", context) < 0.0:
            raise FinalReportingError(f"{context}.duration_s must be non-negative")

    phase_path = phase_objectives_config.resolve()
    action_path = phase_action_config.resolve()
    reward_path = reward_config.resolve()
    phase_config, phase_record = _read_yaml(phase_path, "phase objective config")
    action_config, action_record = _read_yaml(action_path, "phase action config")
    reward_config_value, reward_record = _read_yaml(reward_path, "reward config")
    try:
        load_phase_objectives(phase_path)
    except PhaseObjectiveError as exc:
        raise FinalReportingError(
            f"phase objective config failed frozen-envelope validation: {exc}"
        ) from exc
    envelope_source = phase_config["successful_fsm_attitude_envelope"]["source"]
    repository_root = DEFAULT_PHASE_OBJECTIVES_PATH.parents[1]
    for path_name, hash_name, label in (
        (
            "frozen_manifest_path",
            "frozen_manifest_sha256",
            "frozen successful-FSM manifest",
        ),
        (
            "snapshot_manifest_path",
            "snapshot_manifest_sha256",
            "phase snapshot manifest",
        ),
        (
            "level_reference_snapshot_path",
            "level_reference_snapshot_sha256",
            "P01 level-reference snapshot",
        ),
    ):
        source_record, _ = _captured_file(
            repository_root / str(envelope_source[path_name]),
            label=label,
        )
        if source_record["sha256"] != envelope_source[hash_name]:
            raise FinalReportingError(f"{label} SHA-256 changed after validation")
        input_records.append(source_record)
    _validate_configs(phase_config, action_config, reward_config_value)
    calibration_records: list[Mapping[str, Any]] = []
    if final_lifecycle:
        calibration = promotion.get("residual_activity_calibration")
        phase_scales = (
            calibration.get("phase_scale_full12")
            if isinstance(calibration, Mapping)
            else None
        )
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("schema")
            != "wlr50_clean.residual_activity_calibration.v1"
            or Path(str(calibration.get("phase_action_config", ""))).resolve()
            != action_path
            or calibration.get("phase_action_config_sha256")
            != action_record["sha256"]
            or tuple(calibration.get("full12_order", ()))
            != tuple(action_config["full12_order"])
            or tuple(calibration.get("phase_ids", ())) != PHASE_IDS
            or not isinstance(phase_scales, Mapping)
            or tuple(phase_scales) != PHASE_IDS
            or any(
                tuple(phase_scales[phase])
                != tuple(action_config["phases"][phase]["scale_full12"])
                for phase in PHASE_IDS
            )
        ):
            raise FinalReportingError(
                "residual spectrum/activity calibration is not bound to the active v2 action-scale config"
            )
        environment_path = Path(
            str(calibration.get("environment_lock", ""))
        ).resolve()
        environment_record, _ = _captured_file(
            environment_path, label="residual calibration environment lock"
        )
        if (
            calibration.get("environment_lock_sha256")
            != environment_record["sha256"]
        ):
            raise FinalReportingError(
                "residual calibration environment-lock SHA-256 is stale"
            )
        calibration_records.append(environment_record)
    try:
        migration = load_reward_migration(reward_migration_config)
    except RewardMigrationError as exc:
        raise FinalReportingError(f"reward migration evidence is invalid: {exc}") from exc
    if (
        migration.target_path != reward_path
        or migration.target_sha256 != reward_record["sha256"]
    ):
        raise FinalReportingError(
            "active reward config is not the target bound by reward migration evidence"
        )
    migration_value = migration.as_dict()
    migration_record, _ = _captured_file(
        migration.path, label="reward migration evidence"
    )
    source_reward_record, _ = _captured_file(
        migration.source_path, label="legacy reward source"
    )
    if (
        migration_record["sha256"] != migration.path_sha256
        or source_reward_record["sha256"] != migration.source_sha256
    ):
        raise FinalReportingError("reward migration source bindings changed during load")

    training_value: Mapping[str, Any]
    training_records: list[Mapping[str, Any]] = []
    if training_orchestration_manifest is None:
        if not allow_nonfinal_two_role:
            raise FinalReportingError(
                "final reporting requires training_orchestration_manifest.json"
            )
        training_value = {
            "schema": "nonfinal.testing.no_training_orchestration",
            "valid": False,
            "status": "NOT_PROVIDED",
        }
    else:
        try:
            orchestration = validate_training_orchestration_manifest(
                training_orchestration_manifest,
                expected_project_root=Path(__file__).resolve().parents[3],
            )
        except TrainingOrchestrationError as exc:
            raise FinalReportingError(
                f"training orchestration evidence is invalid: {exc}"
            ) from exc
        training_value = orchestration["payload"]
        if (
            training_value.get("schema") != TRAINING_ORCHESTRATION_SCHEMA
            or training_value.get("valid") is not True
            or orchestration.get("valid") is not True
            or training_value.get("status") != orchestration.get("status")
        ):
            raise FinalReportingError("training orchestration schema is invalid")
        training_records.append(
            {
                "path": str(orchestration["path"]),
                "bytes": int(orchestration["bytes"]),
                "sha256": str(orchestration["sha256"]),
            }
        )
        raw_sources = orchestration.get("source_file_records")
        if not isinstance(raw_sources, Sequence) or isinstance(
            raw_sources, (str, bytes, bytearray)
        ):
            raise FinalReportingError(
                "training orchestration validator omitted source file records"
            )
        training_records.extend(raw_sources)
        if final_lifecycle:
            initial = training_value.get("initial_checkpoint")
            terminal = training_value.get("terminal")
            terminal_checkpoint = (
                terminal.get("checkpoint") if isinstance(terminal, Mapping) else None
            )
            passing_promotions = tuple(
                row
                for row in training_value.get("promotion_decisions", ())
                if isinstance(row, Mapping) and row.get("promoted") is True
            )
            chunks = training_value.get("chunks")
            first_chunk = (
                chunks[0]
                if isinstance(chunks, Sequence)
                and not isinstance(chunks, (str, bytes, bytearray))
                and chunks
                and isinstance(chunks[0], Mapping)
                else None
            )
            first_training = (
                first_chunk.get("training")
                if isinstance(first_chunk, Mapping)
                else None
            )
            smoke_history = (
                first_training.get("immutable_history_checkpoint")
                if isinstance(first_training, Mapping)
                else None
            )
            smoke_checkpoint = training_value.get("smoke_checkpoint")
            canonical_smoke = training_value.get("canonical_smoke_checkpoint")
            terminal_chunk_index = (
                terminal.get("chunk_index")
                if isinstance(terminal, Mapping)
                else None
            )
            orchestrated_promotion = (
                passing_promotions[0] if len(passing_promotions) == 1 else None
            )
            orchestrated_candidate = (
                orchestrated_promotion.get("candidate_checkpoint")
                if isinstance(orchestrated_promotion, Mapping)
                else None
            )
            orchestration_creation_identities = {
                (
                    Path(str(record.get("path", ""))).resolve(),
                    str(record.get("sha256")),
                )
                for record in raw_sources
                if isinstance(record, Mapping)
                and Path(str(record.get("path", ""))).name
                == "committed_runtime_identity.before.json"
            }
            if (
                orchestration.get("status") != "PROMOTION_FOUND"
                or training_value.get("training_seed") != lifecycle_training_seeds[0]
                or training_value.get("git_commit")
                != runtime_identities[0].get("git_commit")
                or not isinstance(initial, Mapping)
                or Path(str(initial.get("path", ""))).resolve()
                != checkpoint_files["checkpoint_initial"]
                or initial.get("sha256")
                != checkpoints["checkpoint_initial"].get("checkpoint_sha256")
                or Path(str(initial.get("manifest_path", ""))).resolve()
                != checkpoint_manifest_files["checkpoint_initial"]
                or initial.get("manifest_sha256")
                != checkpoints["checkpoint_initial"].get(
                    "checkpoint_manifest_sha256"
                )
                or not isinstance(first_chunk, Mapping)
                or first_chunk.get("stage") != "smoke"
                or not isinstance(smoke_history, Mapping)
                or smoke_history.get("sha256")
                != checkpoints["checkpoint_smoke"].get("checkpoint_sha256")
                or not isinstance(smoke_checkpoint, Mapping)
                or smoke_checkpoint.get("sha256")
                != checkpoints["checkpoint_smoke"].get("checkpoint_sha256")
                or not isinstance(canonical_smoke, Mapping)
                or Path(str(canonical_smoke.get("path", ""))).resolve()
                != checkpoint_files["checkpoint_smoke"]
                or canonical_smoke.get("sha256")
                != checkpoints["checkpoint_smoke"].get("checkpoint_sha256")
                or Path(str(canonical_smoke.get("manifest_path", ""))).resolve()
                != checkpoint_manifest_files["checkpoint_smoke"]
                or canonical_smoke.get("manifest_sha256")
                != checkpoints["checkpoint_smoke"].get(
                    "checkpoint_manifest_sha256"
                )
                or not isinstance(terminal_checkpoint, Mapping)
                or terminal_chunk_index != len(chunks) - 1
                or terminal_checkpoint.get("sha256")
                != checkpoints["checkpoint_improved"].get("checkpoint_sha256")
                or len(passing_promotions) != 1
                or not isinstance(orchestrated_promotion, Mapping)
                or orchestrated_promotion.get("record")
                != cadence_decision_record
                or orchestrated_promotion.get("bound_chunk_index")
                != terminal_chunk_index
                or not isinstance(orchestrated_candidate, Mapping)
                or dict(orchestrated_candidate) != dict(terminal_checkpoint)
                or Path(
                    str(cadence_decision.get("candidate_checkpoint_path", ""))
                ).resolve()
                != Path(str(terminal_checkpoint.get("path", ""))).resolve()
                or cadence_decision.get("candidate_checkpoint_sha256")
                != terminal_checkpoint.get("sha256")
                or len(lifecycle_creation_identities) != 4
                or not set(
                    lifecycle_creation_identities.values()
                ).issubset(orchestration_creation_identities)
            ):
                raise FinalReportingError(
                    "training orchestration is not exactly bound to the initial and promoted improved lifecycle checkpoints"
                )

    input_records.extend(
        (
            promotion_record,
            phase_record,
            action_record,
            reward_record,
            migration_record,
            source_reward_record,
            *calibration_records,
            *training_records,
        )
    )
    inputs: list[Path] = []
    seen_inputs: set[Path] = set()
    for record in input_records:
        raw = record.get("path") if isinstance(record, Mapping) else None
        if not isinstance(raw, str):
            raise FinalReportingError("input provenance record omits an absolute path")
        path = Path(raw).resolve()
        if path not in seen_inputs:
            seen_inputs.add(path)
            inputs.append(path)
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
        reward_migration=migration_value,
        training_orchestration=training_value,
        input_paths=tuple(inputs),
        input_records=tuple(input_records),
        seeds=seeds,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        promoted=promoted,
        duration_pair_diagnostics=duration_pair_diagnostics,
        checkpoint_roles=tuple(checkpoint_roles),
        final_lifecycle=final_lifecycle,
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


def _comparison_bars(
    ax: Any,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: Sequence[str],
    ylabel: str,
    *,
    candidate_legend: str,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    ax.bar(x - width / 2, baseline, width, label="FSM baseline", color="#6b7280")
    ax.bar(x + width / 2, candidate, width, label=candidate_legend, color="#2563eb")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.legend()


def _plot_payloads(evidence: _ReportingEvidence) -> Mapping[str, bytes]:
    seeds = evidence.seeds
    candidate_legend = (
        "PPO improved" if evidence.final_lifecycle else "PPO candidate (non-final)"
    )
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
    lifecycle_pitch = None
    if evidence.final_lifecycle:
        checkpoint_rows = _checkpoint_rows(evidence)
        lifecycle_pitch = np.asarray(
            [
                _finite(
                    checkpoint_rows[role],
                    "mean_overall_pitch_rate_rms_rad_s",
                    f"checkpoint comparison {role}",
                )
                for role in evidence.checkpoint_roles
            ]
        )

    def overall(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        if lifecycle_pitch is not None:
            x = np.arange(len(evidence.checkpoint_roles))
            labels = [role.removeprefix("checkpoint_") for role in evidence.checkpoint_roles]
            colors = ("#6b7280", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8")
            ax.bar(x, lifecycle_pitch, color=colors)
            ax.set_xticks(x, labels, rotation=20)
            ax.set_xlabel("Matched validation lifecycle role")
            ax.set_ylabel("Mean overall pitch-rate RMS (rad/s)")
            return
        x = np.arange(len(seeds))
        for index in range(len(seeds)):
            ax.plot(
                (x[index] - 0.08, x[index] + 0.08),
                (baseline_overall[index], candidate_overall[index]),
                color="#9ca3af",
                linewidth=1,
            )
        ax.scatter(x - 0.08, baseline_overall, label="FSM baseline", color="#6b7280", zorder=3)
        ax.scatter(x + 0.08, candidate_overall, label=candidate_legend, color="#2563eb", zorder=3)
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
        ax.bar(x + width / 2, cand_pitch, width, yerr=cand_pitch_std, capsize=2, label=candidate_legend, color="#2563eb")
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
        _comparison_bars(axes[0], base_roll, cand_roll, phase_labels, "Roll RMS (rad)", candidate_legend=candidate_legend)
        _comparison_bars(axes[1], base_att_pitch, cand_att_pitch, phase_labels, "Pitch RMS (rad)", candidate_legend=candidate_legend)
        axes[1].set_xlabel("FSM phase")

    payloads[PLOT_FILENAMES[2]] = _figure_bytes(
        "Phase-Wise Roll/Pitch RMS", evidence.promoted, attitude, size=(11.0, 8.0)
    )

    fr_base = _one_phase_values(evidence.baseline_phase, "P03", "placement_contact_impulse_n_s", seeds)
    fr_candidate = _one_phase_values(evidence.candidate_phase, "P03", "placement_contact_impulse_n_s", seeds)

    def fr_impulse(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        _comparison_bars(ax, fr_base, fr_candidate, [str(seed) for seed in seeds], "P03 FR contact impulse (N·s)", candidate_legend=candidate_legend)
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
        _comparison_bars(axes[0], p08_base_rate, p08_cand_rate, labels, "Pitch-rate RMS (rad/s)", candidate_legend=candidate_legend)
        axes[0].set_title("Transfer-phase motion")
        _comparison_bars(axes[1], p08_base_settle, p08_cand_settle, labels, "Settling time (s)", candidate_legend=candidate_legend)
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
        _comparison_bars(axes[0], p12_base_roll, p12_cand_roll, labels, "Roll RMS (rad)", candidate_legend=candidate_legend)
        _comparison_bars(axes[1], p12_base_pitch, p12_cand_pitch, labels, "Pitch RMS (rad)", candidate_legend=candidate_legend)
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
        _comparison_bars(axes[0], p13_base_home, p13_cand_home, labels, "Home-pose error RMS (deg)", candidate_legend=candidate_legend)
        _comparison_bars(axes[1], p13_base_jerk, p13_cand_jerk, labels, "Applied-action jerk RMS", candidate_legend=candidate_legend)
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

    residual_bands = np.vstack(
        [
            _phase_values(evidence.candidate_phase, field)[0]
            for field in _RESIDUAL_SPECTRAL_BANDS
        ]
    )

    def spectrum(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(PHASE_IDS))
        bottom = np.zeros(len(PHASE_IDS), dtype=float)
        colors = ("#1d4ed8", "#2563eb", "#60a5fa", "#f59e0b", "#dc2626")
        for label, values, color in zip(
            _RESIDUAL_SPECTRAL_LABELS, residual_bands, colors, strict=True
        ):
            ax.bar(x, values, bottom=bottom, label=f"{label} Hz", color=color)
            bottom += values
        ax.set_xticks(x, phase_labels)
        ax.set_ylim(0.0, max(1.05, float(np.max(bottom)) * 1.05))
        ax.set_ylabel("Dimensionless residual spectral-energy fraction")
        ax.set_xlabel("FSM phase")
        ax.legend(ncol=3)

    payloads[PLOT_FILENAMES[8]] = _figure_bytes(
        "Improved-Policy Residual Frequency Spectrum by Phase",
        evidence.promoted,
        spectrum,
    )

    base_duration, base_duration_std = _phase_values(evidence.baseline_phase, "duration_s")
    cand_duration, cand_duration_std = _phase_values(evidence.candidate_phase, "duration_s")

    def duration(figure: Any, _plt: Any) -> None:
        ax = figure.add_subplot(111)
        x = np.arange(len(PHASE_IDS))
        width = 0.38
        ax.bar(x - width / 2, base_duration, width, yerr=base_duration_std, capsize=2, label="FSM baseline", color="#6b7280")
        ax.bar(x + width / 2, cand_duration, width, yerr=cand_duration_std, capsize=2, label=candidate_legend, color="#2563eb")
        ax.set_xticks(x, phase_labels)
        ax.set_ylabel("Phase duration (s); mean ± population SD")
        ax.legend()

    payloads[PLOT_FILENAMES[9]] = _figure_bytes(
        "FSM vs PPO Improved Phase Duration", evidence.promoted, duration
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


def _hash_table_records(records: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| Input | SHA-256 |", "|---|---|"]
    seen: set[tuple[str, str]] = set()
    for record in records:
        path = str(record.get("path", ""))
        digest = str(record.get("sha256", ""))
        key = (path, digest)
        if not path or len(digest) != 64 or key in seen:
            continue
        seen.add(key)
        lines.append(f"| `{path}` | `{digest}` |")
    return "\n".join(lines)


def _phase_design_report(evidence: _ReportingEvidence) -> str:
    full12 = tuple(evidence.action_config["full12_order"])
    phase_rows = evidence.phase_config["phases"]
    action_rows = evidence.action_config["phases"]
    migration_entries = evidence.reward_migration["entries"]
    audits = evidence.reward_config["audits"]
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
            f"Transfer-aware phases: {', '.join(transfer)}. In particular, P08 and P11 retain explicit active/capture/settle schedules so necessary transfer attitude is not scored as a static-level error.",
            "",
            "## Reward dominance audit",
            "",
            "| Audit | Locked value |",
            "|---|---:|",
            f"| Maximum single dense-family fraction | {_fmt(audits['maximum_single_dense_family_fraction'])} |",
            f"| Maximum residual-regularization fraction | {_fmt(audits['maximum_residual_regularization_fraction'])} |",
            f"| Minimum absolute dense return | {_fmt(audits['minimum_absolute_dense_return'])} |",
            f"| Maximum standing-still reward | {_fmt(audits['standing_still_max_reward'])} |",
            "",
            "## Reward v1 → v2 migration",
            "",
            "All 13 preserved v1 concepts have an explicit, validated disposition; the v1 source remains inactive and the v2 target remains active.",
            "",
            "| v1 concept | Disposition | v2 destination(s) | Rationale |",
            "|---|---|---|---|",
            *[
                "| "
                + " | ".join(
                    (
                        _fmt(row["v1_concept"]),
                        _fmt(row["disposition"]),
                        ", ".join(f"`{item}`" for item in row["v2_destinations"]),
                        _fmt(row["rationale"]),
                    )
                )
                + " |"
                for row in migration_entries
            ],
            "",
            "## Input integrity",
            "",
            f"Migration SHA-256: `{evidence.reward_migration['migration_sha256']}`; source v1 SHA-256: `{evidence.reward_migration['source_sha256']}`; target v2 SHA-256: `{evidence.reward_migration['target_sha256']}`.",
            "",
            _hash_table_records(evidence.input_records),
            "",
        ]
    )
    return "\n".join(lines)


def _checkpoint_rows(evidence: _ReportingEvidence) -> Mapping[str, Mapping[str, str]]:
    return {str(row["role"]): row for row in evidence.checkpoint_comparison}


def _training_orchestration_section(manifest: Mapping[str, Any]) -> str:
    if manifest.get("schema") != TRAINING_ORCHESTRATION_SCHEMA:
        return (
            "No authoritative prefinal orchestration evidence was supplied to this "
            "non-final test rendering."
        )
    terminal = manifest.get("terminal")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    checkpoint = terminal.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    fields = (
        ("Schema", manifest.get("schema")),
        ("Status", manifest.get("status")),
        ("Valid", manifest.get("valid")),
        ("Training seed", manifest.get("training_seed")),
        ("Parallel environments", manifest.get("num_envs")),
        ("Ordered chunk count", manifest.get("chunk_count")),
        ("Terminal stage", terminal.get("stage")),
        ("Terminal global policy decisions", terminal.get("global_policy_decisions")),
        ("Terminal checkpoint SHA-256", checkpoint.get("sha256")),
        (
            "Deterministic validation interval",
            manifest.get("deterministic_validation_interval"),
        ),
    )
    lines = [
        "These facts come from the independently validated prefinal training-orchestration manifest; the final training_manifest.json is not read by reporting.",
        "",
        "| Training fact | Value |",
        "|---|---|",
    ]
    lines.extend(f"| {label} | {_fmt(value)} |" for label, value in fields)
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
    manifest_section = _training_orchestration_section(
        evidence.training_orchestration
    )
    lines = [
        "# PPO Training Report",
        "",
        f"Promotion status: **{status}**",
        "",
        "## Evidence boundary",
        "",
        "This offline report reads immutable paired evaluation exports. It does not use reward curves as proof of physical improvement and does not execute Isaac or training.",
        "",
        "## Prefinal training orchestration evidence",
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
    for role in evidence.checkpoint_roles:
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
            _hash_table_records(evidence.input_records),
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
    baseline_role = "pure_fsm" if evidence.final_lifecycle else "baseline"
    candidate_role = "checkpoint_improved" if evidence.final_lifecycle else "candidate"
    base = checkpoints[baseline_role]
    candidate = checkpoints[candidate_role]
    duration_diagnostics = evidence.duration_pair_diagnostics
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
        f"Improved-checkpoint label: `{evidence.candidate_label}`. First failed gate: `{evidence.promotion.get('first_failed_gate')}`.",
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
            "## Matched-seed duration gate",
            "",
            "The 15% slowdown gate is evaluated independently for every matched seed; the group mean is diagnostic only.",
            "",
            f"Mean FSM duration: `{_fmt(duration_diagnostics['baseline_mean_duration_s'])}` s. "
            f"Mean candidate duration: `{_fmt(duration_diagnostics['candidate_mean_duration_s'])}` s. "
            f"Mean ratio: `{_fmt(duration_diagnostics['mean_candidate_to_baseline_ratio'])}`.",
            "",
            f"First violating seed: `{duration_diagnostics['first_violating_seed']}`. "
            f"Worst seed: `{duration_diagnostics['worst_seed']}`; worst ratio: "
            f"`{_fmt(duration_diagnostics['worst_candidate_to_baseline_ratio'])}`.",
            "",
            "| Seed | FSM (s) | Candidate (s) | 1.15× limit (s) | Ratio | Passed |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['seed']} | {_fmt(row['baseline_duration_s'])} | "
        f"{_fmt(row['candidate_duration_s'])} | {_fmt(row['allowed_duration_s'])} | "
        f"{_fmt(row['candidate_to_baseline_ratio'])} | "
        f"{'yes' if row['passed'] else 'no'} |"
        for row in duration_diagnostics["paired_rows"]
    )
    lines.extend(
        [
            "",
            "## Overall paired comparison",
            "",
            f"Global stability {fraction_heading.lower()}: `{_fmt(promotion.get('global_stability_improvement_fraction'))}`.",
            "",
            "| Metric | FSM baseline | PPO improved | Improved − FSM |",
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
    for role in evidence.checkpoint_roles:
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
            "The frequency-domain plot uses five validated bands from checkpoint_improved after each Full12 residual channel is divided by its bound phase-specific physical action scale; zero-scale channels must remain zero.",
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


def _assert_reporting_inputs_unchanged(
    records: Sequence[Mapping[str, Any]],
) -> None:
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FinalReportingError(
                f"reporting provenance record {index} is not a mapping"
            )
        try:
            _record_unchanged(record, label=f"reporting input {index}")
        except (EvaluationArtifactError, TypeError, ValueError) as exc:
            raise FinalReportingError(str(exc)) from exc


def _prepare_reporting_bundle(
    metrics_directory: str | Path,
    output_root: str | Path,
    *,
    phase_objectives_config: str | Path,
    phase_action_config: str | Path,
    reward_config: str | Path,
    reward_migration_config: str | Path,
    training_orchestration_manifest: str | Path | None,
    allow_nonfinal_two_role: bool,
) -> tuple[FinalReportingPaths, _ReportingEvidence, Mapping[Path, bytes]]:
    root = Path(output_root)
    try:
        _require_no_reparse_components(root, label="final reporting output root")
    except EvaluationArtifactError as exc:
        raise FinalReportingError(str(exc)) from exc
    root = root.resolve()
    if root.exists() and not root.is_dir():
        raise FinalReportingError(f"output root is not a directory: {root}")
    evidence = _load_evidence(
        Path(metrics_directory),
        phase_objectives_config=Path(phase_objectives_config),
        phase_action_config=Path(phase_action_config),
        reward_config=Path(reward_config),
        reward_migration_config=Path(reward_migration_config),
        training_orchestration_manifest=(
            None
            if training_orchestration_manifest is None
            else Path(training_orchestration_manifest)
        ),
        allow_nonfinal_two_role=allow_nonfinal_two_role,
    )
    paths = _paths(root)
    protected_trees: list[Path] = [evidence.metrics_directory]
    if evidence.final_lifecycle:
        lifecycle = evidence.promotion.get("final_lifecycle_evidence")
        if isinstance(lifecycle, Mapping):
            for role in FINAL_LIFECYCLE_ROLES:
                record = lifecycle.get(role)
                if not isinstance(record, Mapping):
                    continue
                protected_trees.extend(
                    Path(str(value)).resolve()
                    for value in record.get("canonical_episode_dirs", ())
                )
                for group in record.get("source_groups", ()):
                    if isinstance(group, Mapping):
                        protected_trees.extend(
                            Path(str(group[key])).resolve()
                            for key in (
                                "worker_run_dir",
                                "canonical_episode_dir",
                            )
                            if str(group.get(key, "")).strip()
                        )
    conflict = next(
        (
            source
            for destination in (paths.plots_directory, paths.reports_directory)
            for source in protected_trees
            if _paths_overlap(destination, source)
        ),
        None,
    )
    if conflict is not None:
        raise FinalReportingError(
            "final report output overlaps an immutable evidence tree: "
            f"output_root={root}, source={conflict}"
        )
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
    return paths, evidence, publications


def _generate_reporting_bundle(
    metrics_directory: str | Path,
    output_root: str | Path,
    *,
    phase_objectives_config: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
    phase_action_config: str | Path = DEFAULT_PHASE_ACTION_CONFIG_V2,
    reward_config: str | Path = DEFAULT_REWARD_PATH_V2,
    reward_migration_config: str | Path = DEFAULT_MIGRATION_PATH,
    training_orchestration_manifest: str | Path | None = None,
    allow_nonfinal_two_role: bool,
) -> FinalReportingPaths:
    paths, evidence, publications = _prepare_reporting_bundle(
        metrics_directory,
        output_root,
        phase_objectives_config=Path(phase_objectives_config),
        phase_action_config=Path(phase_action_config),
        reward_config=Path(reward_config),
        reward_migration_config=Path(reward_migration_config),
        training_orchestration_manifest=training_orchestration_manifest,
        allow_nonfinal_two_role=allow_nonfinal_two_role,
    )
    _assert_reporting_inputs_unchanged(evidence.input_records)
    _publish_idempotently(publications)
    return paths


def generate_final_reporting_bundle(
    metrics_directory: str | Path,
    output_root: str | Path,
    *,
    training_orchestration_manifest: str | Path,
    phase_objectives_config: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
    phase_action_config: str | Path = DEFAULT_PHASE_ACTION_CONFIG_V2,
    reward_config: str | Path = DEFAULT_REWARD_PATH_V2,
    reward_migration_config: str | Path = DEFAULT_MIGRATION_PATH,
) -> FinalReportingPaths:
    """Generate final reports only from strict five-role lifecycle evidence."""

    return _generate_reporting_bundle(
        metrics_directory,
        output_root,
        phase_objectives_config=phase_objectives_config,
        phase_action_config=phase_action_config,
        reward_config=reward_config,
        reward_migration_config=reward_migration_config,
        training_orchestration_manifest=training_orchestration_manifest,
        allow_nonfinal_two_role=False,
    )


def generate_nonfinal_two_role_reporting_bundle_for_testing(
    metrics_directory: str | Path,
    output_root: str | Path,
    *,
    phase_objectives_config: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
    phase_action_config: str | Path = DEFAULT_PHASE_ACTION_CONFIG_V2,
    reward_config: str | Path = DEFAULT_REWARD_PATH_V2,
    reward_migration_config: str | Path = DEFAULT_MIGRATION_PATH,
    training_orchestration_manifest: str | Path | None = None,
) -> FinalReportingPaths:
    """Render legacy two-role evidence without granting it final-report status."""

    return _generate_reporting_bundle(
        metrics_directory,
        output_root,
        phase_objectives_config=phase_objectives_config,
        phase_action_config=phase_action_config,
        reward_config=reward_config,
        reward_migration_config=reward_migration_config,
        training_orchestration_manifest=training_orchestration_manifest,
        allow_nonfinal_two_role=True,
    )


def verify_final_reporting_bundle(
    metrics_directory: str | Path,
    output_root: str | Path,
    *,
    training_orchestration_manifest: str | Path,
    report_paths: Sequence[str | Path],
    plot_paths: Sequence[str | Path],
    phase_objectives_config: str | Path = DEFAULT_PHASE_OBJECTIVES_PATH,
    phase_action_config: str | Path = DEFAULT_PHASE_ACTION_CONFIG_V2,
    reward_config: str | Path = DEFAULT_REWARD_PATH_V2,
    reward_migration_config: str | Path = DEFAULT_MIGRATION_PATH,
) -> Mapping[str, Any]:
    """Re-render and byte-verify the only report bundle accepted as final."""

    paths, evidence, expected = _prepare_reporting_bundle(
        metrics_directory,
        output_root,
        phase_objectives_config=phase_objectives_config,
        phase_action_config=phase_action_config,
        reward_config=reward_config,
        reward_migration_config=reward_migration_config,
        training_orchestration_manifest=training_orchestration_manifest,
        allow_nonfinal_two_role=False,
    )
    supplied_reports = {Path(path).resolve().name: Path(path).resolve() for path in report_paths}
    supplied_plots = {Path(path).resolve().name: Path(path).resolve() for path in plot_paths}
    if (
        len(tuple(report_paths)) != len(REPORT_FILENAMES)
        or len(tuple(plot_paths)) != len(PLOT_FILENAMES)
        or set(supplied_reports) != set(REPORT_FILENAMES)
        or set(supplied_plots) != set(PLOT_FILENAMES)
        or tuple(supplied_reports[name] for name in REPORT_FILENAMES)
        != paths.files()[len(PLOT_FILENAMES) :]
        or tuple(supplied_plots[name] for name in PLOT_FILENAMES)
        != paths.files()[: len(PLOT_FILENAMES)]
    ):
        raise FinalReportingError(
            "supplied report/plot paths are not the canonical strict final bundle"
        )
    output_records: list[Mapping[str, Any]] = []
    for path, content in expected.items():
        record, actual = _captured_file(path, label=f"final reporting output {path.name}")
        if actual != content:
            raise FinalReportingError(
                f"final reporting output is not byte-derived from strict evidence: {path}"
            )
        output_records.append(record)
    _assert_reporting_inputs_unchanged(evidence.input_records)
    return {
        "schema": REPORTING_SCHEMA,
        "valid": True,
        "bundle_kind": FINAL_LIFECYCLE_BUNDLE_KIND,
        "metrics_directory": str(evidence.metrics_directory),
        "training_orchestration_schema": evidence.training_orchestration.get("schema"),
        "training_orchestration_status": evidence.training_orchestration.get("status"),
        "reward_migration": dict(evidence.reward_migration),
        "input_files": [dict(record) for record in evidence.input_records],
        "outputs": [dict(record) for record in output_records],
        "five_role_artifact_provenance": evidence.promotion.get(
            "final_lifecycle_evidence"
        ),
    }


__all__ = [
    "PLOT_FILENAMES",
    "REPORT_FILENAMES",
    "REPORTING_SCHEMA",
    "FinalReportingError",
    "FinalReportingPaths",
    "generate_final_reporting_bundle",
    "generate_nonfinal_two_role_reporting_bundle_for_testing",
    "verify_final_reporting_bundle",
]
