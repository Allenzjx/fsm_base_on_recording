"""Strict offline entry point for the final five-role PPO delivery.

This module intentionally does not expose the legacy paired/two-role report
path.  A successful invocation evaluates the five immutable lifecycle
aggregates, publishes the canonical metrics and reports, and only then asks
the finalizer to produce the delivery manifests and complete checksum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation_artifacts import (
    DEFAULT_REWARD_STREAM_FILENAME,
    FINAL_LIFECYCLE_ROLES,
    export_baseline_evaluation_artifacts,
    export_final_lifecycle_evaluation_artifacts,
    validate_final_lifecycle_aggregate_evidence,
)
from .final_reporting import (
    PLOT_FILENAMES,
    REPORT_FILENAMES,
    generate_final_reporting_bundle,
)
from .finalization import finalize_ppo_phase_delivery
from .phase_action_masks_v2 import DEFAULT_PHASE_ACTION_CONFIG_V2
from .phase_objectives import DEFAULT_PHASE_OBJECTIVES_PATH
from .reward_migration import DEFAULT_MIGRATION_PATH
from .reward_v2 import DEFAULT_REWARD_PATH_V2


def _path(value: str) -> Path:
    # Preserve the caller's lexical path until the strict validators inspect
    # every component; resolving here would hide a supplied symlink/junction.
    return Path(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wlr50_clean.ppo.delivery_cli",
        description="Publish and finalize the strict five-role PPO evidence bundle.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    deliver = subcommands.add_parser(
        "deliver",
        help="export five-role metrics, render reports, and finalize manifests",
    )
    deliver.add_argument("--output-root", type=_path, required=True)
    deliver.add_argument(
        "--training-orchestration-manifest", type=_path, required=True
    )
    deliver.add_argument("--pure-fsm-aggregate", type=_path, required=True)
    deliver.add_argument(
        "--checkpoint-initial-aggregate", type=_path, required=True
    )
    deliver.add_argument("--checkpoint-smoke-aggregate", type=_path, required=True)
    deliver.add_argument("--checkpoint-best-aggregate", type=_path, required=True)
    deliver.add_argument("--checkpoint-improved-aggregate", type=_path, required=True)
    deliver.add_argument("--validation-aggregate", type=_path, required=True)
    deliver.add_argument(
        "--validation-promotion-decision", type=_path, required=True
    )
    deliver.add_argument("--locked-test-aggregate", type=_path, required=True)
    deliver.add_argument(
        "--checkpoint-manifest",
        type=_path,
        action="append",
        required=True,
        dest="checkpoint_manifests",
        help=(
            "Explicit checkpoint/promotion/export manifest; repeat for every "
            "manifest in the best-to-improved authorization chain."
        ),
    )
    deliver.add_argument(
        "--inference-actor-export-run-dir", type=_path, required=True
    )
    deliver.add_argument("--video-validation", type=_path, required=True)
    deliver.add_argument("--video-checksums", type=_path, required=True)
    deliver.add_argument(
        "--phase-objectives-config",
        type=_path,
        default=DEFAULT_PHASE_OBJECTIVES_PATH,
    )
    deliver.add_argument(
        "--phase-action-config",
        type=_path,
        default=DEFAULT_PHASE_ACTION_CONFIG_V2,
    )
    deliver.add_argument(
        "--reward-config", type=_path, default=DEFAULT_REWARD_PATH_V2
    )
    deliver.add_argument(
        "--reward-migration-config", type=_path, default=DEFAULT_MIGRATION_PATH
    )
    deliver.add_argument(
        "--reward-stream-filename", default=DEFAULT_REWARD_STREAM_FILENAME
    )
    return parser


def _lifecycle_aggregates(arguments: argparse.Namespace) -> dict[str, Path]:
    aggregates = {
        "pure_fsm": arguments.pure_fsm_aggregate,
        "checkpoint_initial": arguments.checkpoint_initial_aggregate,
        "checkpoint_smoke": arguments.checkpoint_smoke_aggregate,
        "checkpoint_best": arguments.checkpoint_best_aggregate,
        "checkpoint_improved": arguments.checkpoint_improved_aggregate,
    }
    if tuple(aggregates) != FINAL_LIFECYCLE_ROLES:
        raise RuntimeError("five-role delivery aggregate order changed")
    return aggregates


def deliver(arguments: argparse.Namespace) -> Mapping[str, Any]:
    """Run the strict delivery chain for an already captured evidence set."""

    root = Path(arguments.output_root)
    metrics_directory = root / "metrics"
    aggregates = _lifecycle_aggregates(arguments)

    # Validate the pure-FSM source first, then let the five-role exporter run
    # its complete input/output-tree preflight before any output is published.
    pure_fsm = validate_final_lifecycle_aggregate_evidence(
        aggregates["pure_fsm"], role="pure_fsm"
    )
    metrics = export_final_lifecycle_evaluation_artifacts(
        metrics_directory,
        pure_fsm_aggregate=aggregates["pure_fsm"],
        checkpoint_initial_aggregate=aggregates["checkpoint_initial"],
        checkpoint_smoke_aggregate=aggregates["checkpoint_smoke"],
        checkpoint_best_aggregate=aggregates["checkpoint_best"],
        checkpoint_improved_aggregate=aggregates["checkpoint_improved"],
        frozen_hashes_unchanged=True,
        reward_stream_filename=arguments.reward_stream_filename,
    )
    # The baseline manifest is independent finalization evidence. Rebuild it
    # from the exact pure-FSM aggregate used above; its two CSVs must be byte-
    # identical to the already published five-role baseline files.
    baseline = export_baseline_evaluation_artifacts(
        metrics_directory,
        episode_directories=pure_fsm.canonical_episode_dirs,
        seeds=pure_fsm.seeds,
        baseline_name="pure_fsm",
    )
    reports = generate_final_reporting_bundle(
        metrics_directory,
        root,
        training_orchestration_manifest=arguments.training_orchestration_manifest,
        phase_objectives_config=arguments.phase_objectives_config,
        phase_action_config=arguments.phase_action_config,
        reward_config=arguments.reward_config,
        reward_migration_config=arguments.reward_migration_config,
    )

    metric_paths = tuple(Path(value) for value in metrics.as_dict().values())
    report_paths = tuple(reports.reports_directory / name for name in REPORT_FILENAMES)
    plot_paths = tuple(reports.plots_directory / name for name in PLOT_FILENAMES)
    finalized = finalize_ppo_phase_delivery(
        output_root=root,
        training_orchestration_manifest_path=(
            arguments.training_orchestration_manifest
        ),
        final_lifecycle_aggregate_paths=aggregates,
        final_lifecycle_metric_paths=metric_paths,
        baseline_aggregate_path=aggregates["pure_fsm"],
        baseline_metric_paths=(
            baseline.episode_metrics,
            baseline.phase_metrics,
            baseline.manifest,
        ),
        validation_aggregate_path=arguments.validation_aggregate,
        promotion_decision_path=arguments.validation_promotion_decision,
        locked_test_aggregate_path=arguments.locked_test_aggregate,
        checkpoint_manifest_paths=tuple(arguments.checkpoint_manifests),
        inference_actor_export_run_dir=arguments.inference_actor_export_run_dir,
        video_validation_path=arguments.video_validation,
        video_checksum_path=arguments.video_checksums,
        report_paths=report_paths,
        plot_paths=plot_paths,
    )
    return {
        "status": "PASS",
        "bundle_kind": "final_lifecycle_five_role",
        "output_root": str(finalized.output_root),
        "metrics": metrics.as_dict(),
        "baseline_manifest": str(baseline.manifest),
        "reports": [str(path) for path in report_paths],
        "plots": [str(path) for path in plot_paths],
        "training_manifest": str(finalized.training_manifest),
        "evaluation_manifest": str(finalized.evaluation_manifest),
        "checksums": str(finalized.checksums),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command != "deliver":  # pragma: no cover - argparse guards this
        raise RuntimeError(f"unsupported command: {arguments.command}")
    result = deliver(arguments)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["deliver", "main"]
