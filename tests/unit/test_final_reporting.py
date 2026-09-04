from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wlr50_clean.ppo.evaluation_artifacts import (
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
from wlr50_clean.ppo.final_reporting import (
    PLOT_FILENAMES,
    REPORT_FILENAMES,
    FinalReportingError,
    generate_final_reporting_bundle,
)
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES
from wlr50_clean.ppo.stability_metrics import PHASE_IDS


SEEDS = (2001, 2002, 2003, 2004, 2005)
BASELINE = "pure_fsm"
CANDIDATE = "candidate_checkpoint"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode_rows(checkpoint: str, multiplier: float) -> list[dict[str, object]]:
    return [
        {
            "checkpoint": checkpoint,
            "episode_index": index,
            "seed": seed,
            "trial_id": f"{checkpoint}-{seed}",
            "run_directory": f"runs/{checkpoint}/{seed}",
            "task_result": "success",
            "task_success": True,
            "duration_s": (13.0 + index / 10.0) * multiplier,
            "overall_pitch_rate_rms_rad_s": (0.40 + index / 100.0) * multiplier,
            "overall_roll_rate_rms_rad_s": (0.30 + index / 100.0) * multiplier,
        }
        for index, seed in enumerate(SEEDS)
    ]


def _phase_rows(checkpoint: str, multiplier: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(SEEDS):
        for phase_index, phase in enumerate(PHASE_IDS, start=1):
            base = 1.0 + phase_index / 100.0 + seed_index / 1000.0
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "seed": seed,
                    "phase": phase,
                    "duration_s": base * multiplier,
                    "roll_rms_rad": 0.10 * base * multiplier,
                    "pitch_rms_rad": 0.12 * base * multiplier,
                    "roll_rate_rms_rad_s": 0.20 * base * multiplier,
                    "pitch_rate_rms_rad_s": 0.25 * base * multiplier,
                    "placement_contact_impulse_n_s": 1.20 * base * multiplier,
                    "settling_time_s": 0.80 * base * multiplier,
                    "home_pose_error_rms_deg": 2.00 * base * multiplier,
                    "action_jerk_rms": 0.30 * base * multiplier,
                    "residual_high_frequency_fraction": 0.08 * base * multiplier,
                    "applied_high_frequency_fraction": 0.04 * base * multiplier,
                }
            )
    return rows


def _write_evidence(directory: Path, *, promoted: bool) -> None:
    _write_csv(directory / BASELINE_EPISODE_FILENAME, _episode_rows(BASELINE, 1.0))
    _write_csv(directory / CANDIDATE_EPISODE_FILENAME, _episode_rows(CANDIDATE, 0.8))
    _write_csv(directory / BASELINE_PHASE_FILENAME, _phase_rows(BASELINE, 1.0))
    _write_csv(directory / CANDIDATE_PHASE_FILENAME, _phase_rows(CANDIDATE, 0.8))
    _write_csv(
        directory / CHECKPOINT_COMPARISON_FILENAME,
        [
            {
                "role": "baseline",
                "checkpoint": BASELINE,
                "episode_count": len(SEEDS),
                "task_success_count": len(SEEDS),
                "body_collision_count": 0,
                "wheel_only_climb_count": 0,
                "mean_duration_s": 13.2,
                "mean_overall_pitch_rate_rms_rad_s": 0.42,
                "mean_overall_roll_rate_rms_rad_s": 0.32,
                "mean_placement_contact_impulse_n_s": 1.2,
                "mean_home_recovery_action_jerk_rms": 0.3,
            },
            {
                "role": "candidate",
                "checkpoint": CANDIDATE,
                "episode_count": len(SEEDS),
                "task_success_count": len(SEEDS),
                "body_collision_count": 0,
                "wheel_only_climb_count": 0,
                "mean_duration_s": 10.56,
                "mean_overall_pitch_rate_rms_rad_s": 0.336,
                "mean_overall_roll_rate_rms_rad_s": 0.256,
                "mean_placement_contact_impulse_n_s": 0.96,
                "mean_home_recovery_action_jerk_rms": 0.24,
            },
        ],
    )
    _write_csv(
        directory / PHASE_COMPARISON_FILENAME,
        [
            {
                "phase": phase,
                "primary_phase_score_improvement_fraction": 0.20,
            }
            for phase in PHASE_IDS
        ],
    )
    _write_csv(
        directory / RESIDUAL_ACTIVITY_FILENAME,
        [
            {
                "checkpoint": CANDIDATE,
                "seed": seed,
                "phase": phase,
                "normalized_residual_rms": 0.10 + phase_index / 1000.0,
                "normalized_residual_peak": 0.20 + phase_index / 1000.0,
                "active_channel_count": 12,
                "nonzero": True,
            }
            for seed in SEEDS
            for phase_index, phase in enumerate(PHASE_IDS)
        ],
    )
    _write_csv(
        directory / REWARD_CONTRIBUTION_FILENAME,
        [
            {
                "checkpoint": CANDIDATE,
                "seed": seed,
                "phase": phase,
                **{f"{family}_sum": -0.1 for family in DENSE_FAMILIES},
                "total_reward_sum": 0.5,
            }
            for seed in SEEDS
            for phase in PHASE_IDS
        ],
    )
    _write_csv(
        directory / TERMINATION_SUMMARY_FILENAME,
        [
            {
                "checkpoint": checkpoint,
                "seed": seed,
                "task_success": True,
                "body_collision": False,
                "wheel_only_climb": False,
                "duration_s": 13.0 if checkpoint == BASELINE else 10.4,
            }
            for checkpoint in (BASELINE, CANDIDATE)
            for seed in SEEDS
        ],
    )
    gate_order = (
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
    failed_gate = "global_stability_improvement_at_least_5pct"
    checks = {
        gate: promoted or gate != failed_gate
        for gate in gate_order
    }
    first_failed = None if promoted else failed_gate
    decision = {
        "schema": EVALUATION_ARTIFACT_SCHEMA,
        "baseline_checkpoint": BASELINE,
        "candidate_checkpoint": CANDIDATE,
        "paired_seeds": list(SEEDS),
        "paired_episode_count": len(SEEDS),
        "minimum_paired_seeds": len(SEEDS),
        "promotion": {
            "promoted": promoted,
            "first_failed_gate": first_failed,
            "checks": checks,
            "global_stability_improvement_fraction": 0.20 if promoted else 0.0,
            "improved_priority_phase_count": 5,
        },
        "first_failed_gate": first_failed,
        "checks_in_evaluation_order": [
            {"gate": gate, "passed": passed} for gate, passed in checks.items()
        ],
    }
    (directory / PROMOTION_DECISION_FILENAME).write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_generates_exact_bundle_and_is_byte_idempotent_and_no_overwrite(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    output = tmp_path / "final"
    _write_evidence(metrics, promoted=True)

    paths = generate_final_reporting_bundle(metrics, output)

    assert tuple(path.name for path in paths.files()) == PLOT_FILENAMES + REPORT_FILENAMES
    assert len(paths.files()) == 13
    for path in paths.files()[:10]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    improvement = paths.improvement_report.read_text(encoding="utf-8")
    assert "supports describing this checkpoint as improved" in improvement
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths.files()}

    rerun = generate_final_reporting_bundle(metrics, output)

    assert rerun == paths
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths.files()
    } == before

    paths.overall_pitch_rate_comparison.write_bytes(b"conflicting plot")
    paths.training_report.unlink()
    with pytest.raises(FinalReportingError, match="refusing to overwrite"):
        generate_final_reporting_bundle(metrics, output)
    assert not paths.training_report.exists(), "preflight must finish before any publication"


def test_failed_promotion_keeps_all_reports_and_plots_candidate_neutral(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    manifest = tmp_path / "training_manifest.json"
    _write_evidence(metrics, promoted=False)
    manifest.write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.ppo_training_run.v1",
                "stage": "full-episode",
                "global_policy_decisions": 1234,
                "status": "UNTRUSTED_CHECKPOINT_IMPROVES_FSM",
            }
        ),
        encoding="utf-8",
    )

    paths = generate_final_reporting_bundle(
        metrics, tmp_path / "final", training_manifest=manifest
    )

    assert all(path.is_file() for path in paths.files())
    improvement = paths.improvement_report.read_text(encoding="utf-8")
    training = paths.training_report.read_text(encoding="utf-8")
    assert "makes no PPO-improvement claim" in improvement
    assert "supports describing this checkpoint as improved" not in improvement
    assert "Promotion status: **NOT PASSED" in training
    assert "UNTRUSTED_CHECKPOINT_IMPROVES_FSM" not in training
    assert "Global policy decisions | 1234" in training


def test_contradictory_promotion_fails_closed_before_creating_output(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    output = tmp_path / "final"
    _write_evidence(metrics, promoted=False)
    decision_path = metrics / PROMOTION_DECISION_FILENAME
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["promotion"]["promoted"] = True
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(FinalReportingError, match="promotion status disagrees"):
        generate_final_reporting_bundle(metrics, output)

    assert not output.exists()


def test_promotion_missing_an_authoritative_gate_fails_closed(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    output = tmp_path / "final"
    _write_evidence(metrics, promoted=True)
    decision_path = metrics / PROMOTION_DECISION_FILENAME
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    del decision["promotion"]["checks"]["body_collision_zero"]
    decision["checks_in_evaluation_order"] = [
        row
        for row in decision["checks_in_evaluation_order"]
        if row["gate"] != "body_collision_zero"
    ]
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(FinalReportingError, match="authoritative gate order"):
        generate_final_reporting_bundle(metrics, output)

    assert not output.exists()


def test_missing_required_csv_column_fails_closed_before_creating_output(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    output = tmp_path / "final"
    _write_evidence(metrics, promoted=True)
    rows = _episode_rows(BASELINE, 1.0)
    for row in rows:
        del row["overall_pitch_rate_rms_rad_s"]
    _write_csv(metrics / BASELINE_EPISODE_FILENAME, rows)

    with pytest.raises(FinalReportingError, match="missing columns"):
        generate_final_reporting_bundle(metrics, output)

    assert not output.exists()
