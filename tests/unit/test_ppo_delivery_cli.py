import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import delivery_cli as subject
from wlr50_clean.ppo.evaluation_artifacts import FINAL_LIFECYCLE_ROLES
from wlr50_clean.ppo.final_reporting import PLOT_FILENAMES, REPORT_FILENAMES


def _arguments(tmp_path: Path) -> argparse.Namespace:
    values = {
        "command": "deliver",
        "output_root": tmp_path / "delivery",
        "training_orchestration_manifest": tmp_path / "orchestration.json",
        "pure_fsm_aggregate": tmp_path / "pure.json",
        "checkpoint_initial_aggregate": tmp_path / "initial.json",
        "checkpoint_smoke_aggregate": tmp_path / "smoke.json",
        "checkpoint_best_aggregate": tmp_path / "best.json",
        "checkpoint_improved_aggregate": tmp_path / "improved.json",
        "validation_aggregate": tmp_path / "cadence-validation.json",
        "validation_promotion_decision": tmp_path / "cadence-promotion.json",
        "locked_test_aggregate": tmp_path / "locked.json",
        "checkpoint_manifests": [tmp_path / f"manifest-{index}.json" for index in range(5)],
        "inference_actor_export_run_dir": tmp_path / "inference-export-run",
        "video_validation": tmp_path / "video_validation.json",
        "video_checksums": tmp_path / "video_checksums.sha256",
        "phase_objectives_config": tmp_path / "phases.yaml",
        "phase_action_config": tmp_path / "actions.yaml",
        "reward_config": tmp_path / "reward.yaml",
        "reward_migration_config": tmp_path / "migration.yaml",
        "reward_stream_filename": "reward_15hz.jsonl",
    }
    return argparse.Namespace(**values)


def test_delivery_parser_has_no_legacy_two_role_shortcut() -> None:
    parser = subject._parser()
    option_strings = {
        option
        for action in parser._subparsers._group_actions[0].choices["deliver"]._actions
        for option in action.option_strings
    }
    assert "--pure-fsm-aggregate" in option_strings
    assert "--checkpoint-initial-aggregate" in option_strings
    assert "--checkpoint-smoke-aggregate" in option_strings
    assert "--checkpoint-best-aggregate" in option_strings
    assert "--checkpoint-improved-aggregate" in option_strings
    assert "--validation-aggregate" in option_strings
    assert "--validation-promotion-decision" in option_strings
    assert "--training-orchestration-manifest" in option_strings
    assert "--inference-actor-export-run-dir" in option_strings
    assert "--baseline-aggregate" not in option_strings
    assert "--candidate-aggregate" not in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args(["deliver", "--output-root", "out"])


def test_delivery_chains_exact_five_roles_into_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    calls: list[str] = []
    captured: dict = {}
    metric_names = (
        "fsm_baseline_episode_metrics.csv",
        "fsm_baseline_phase_metrics.csv",
        "candidate_episode_metrics.csv",
        "candidate_phase_metrics.csv",
        "checkpoint_comparison.csv",
        "phase_metric_comparison.csv",
        "residual_activity_by_phase.csv",
        "reward_contribution_by_phase.csv",
        "termination_summary.csv",
        "promotion_decision.json",
    )
    metrics_dir = arguments.output_root / "metrics"
    metrics_map = {f"metric_{index}": metrics_dir / name for index, name in enumerate(metric_names)}
    metrics = SimpleNamespace(
        promotion_decision=metrics_dir / "promotion_decision.json",
        as_dict=lambda: {key: str(value) for key, value in metrics_map.items()},
    )
    baseline = SimpleNamespace(
        episode_metrics=metrics_dir / "fsm_baseline_episode_metrics.csv",
        phase_metrics=metrics_dir / "fsm_baseline_phase_metrics.csv",
        manifest=metrics_dir / "fsm_baseline_evaluation_manifest.json",
    )
    reporting = SimpleNamespace(
        reports_directory=arguments.output_root / "reports",
        plots_directory=arguments.output_root / "plots",
    )
    finalized = SimpleNamespace(
        output_root=arguments.output_root.resolve(),
        training_manifest=arguments.output_root / "manifests" / "training_manifest.json",
        evaluation_manifest=arguments.output_root / "manifests" / "evaluation_manifest.json",
        checksums=arguments.output_root / "manifests" / "checksums.sha256",
    )

    def validate(path, *, role):
        calls.append("validate-pure")
        assert path == arguments.pure_fsm_aggregate
        assert role == "pure_fsm"
        return SimpleNamespace(
            canonical_episode_dirs=tuple(tmp_path / f"episode-{seed}" for seed in range(5)),
            seeds=(2001, 2002, 2003, 2004, 2005),
        )

    def export_baseline(*_args, **kwargs):
        calls.append("baseline")
        assert kwargs["baseline_name"] == "pure_fsm"
        return baseline

    def export_lifecycle(*_args, **kwargs):
        calls.append("lifecycle")
        assert tuple(key.removesuffix("_aggregate") for key in kwargs if key.endswith("_aggregate")) == FINAL_LIFECYCLE_ROLES
        assert kwargs["frozen_hashes_unchanged"] is True
        return metrics

    def render(*_args, **kwargs):
        calls.append("reports")
        assert kwargs["training_orchestration_manifest"] == arguments.training_orchestration_manifest
        return reporting

    def finalize(**kwargs):
        calls.append("finalize")
        captured.update(kwargs)
        return finalized

    monkeypatch.setattr(subject, "validate_final_lifecycle_aggregate_evidence", validate)
    monkeypatch.setattr(subject, "export_baseline_evaluation_artifacts", export_baseline)
    monkeypatch.setattr(subject, "export_final_lifecycle_evaluation_artifacts", export_lifecycle)
    monkeypatch.setattr(subject, "generate_final_reporting_bundle", render)
    monkeypatch.setattr(subject, "finalize_ppo_phase_delivery", finalize)

    result = subject.deliver(arguments)

    assert calls == ["validate-pure", "lifecycle", "baseline", "reports", "finalize"]
    assert tuple(captured["final_lifecycle_aggregate_paths"]) == FINAL_LIFECYCLE_ROLES
    assert captured["baseline_aggregate_path"] == arguments.pure_fsm_aggregate
    assert captured["validation_aggregate_path"] == arguments.validation_aggregate
    assert (
        captured["promotion_decision_path"]
        == arguments.validation_promotion_decision
    )
    assert (
        captured["inference_actor_export_run_dir"]
        == arguments.inference_actor_export_run_dir
    )
    assert tuple(Path(path).name for path in captured["final_lifecycle_metric_paths"]) == metric_names
    assert tuple(Path(path).name for path in captured["report_paths"]) == REPORT_FILENAMES
    assert tuple(Path(path).name for path in captured["plot_paths"]) == PLOT_FILENAMES
    assert result["status"] == "PASS"


def test_powershell_entry_invokes_only_the_strict_delivery_module() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "finalize_ppo_delivery.ps1").read_text(
        encoding="utf-8"
    )
    assert "wlr50_clean.ppo.delivery_cli" in script
    assert "--training-orchestration-manifest" in script
    assert "--validation-aggregate" in script
    assert "--validation-promotion-decision" in script
    assert "--inference-actor-export-run-dir" in script
    assert "ppo_phase_objectives_v2.yaml" in script
    for role in FINAL_LIFECYCLE_ROLES:
        assert "--" + role.replace("_", "-") + "-aggregate" in script.lower()
    assert "wlr50_clean.ppo.cli" not in script
