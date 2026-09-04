import re
from pathlib import Path


REQUESTED = (
    "ppo_preflight.ps1",
    "run_fsm_baseline_eval.ps1",
    "build_phase_snapshots.ps1",
    "run_zero_residual_live_validation.ps1",
    "run_nonzero_residual_smoke.ps1",
    "initialize_zero_residual_checkpoint.ps1",
    "train_phase_residual_ppo.ps1",
    "evaluate_ppo_checkpoint.ps1",
    "export_paired_ppo_evaluation.ps1",
    "build_fsm_ppo_videos.ps1",
    "publish_ppo_best_validation_checkpoint.ps1",
    "publish_ppo_improved_checkpoint.ps1",
    "export_ppo_inference_actor.ps1",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_requested_entry_scripts_are_thin_extensible_wrappers() -> None:
    scripts = _root() / "scripts"
    for name in REQUESTED:
        text = (scripts / name).read_text(encoding="utf-8")
        assert "_invoke_ppo_cli.ps1" in text, name
        assert "ValueFromRemainingArguments" in text, name
        assert "configs\\ppo_training_phase_v1.yaml" in text, name
        assert '-ConfigPath $Configs' in text, name
        assert "Remove-Item" not in text, name


def test_common_wrapper_locks_runtime_reserves_logs_and_finalizes() -> None:
    text = (_root() / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")
    for required in (
        '$ErrorActionPreference = "Stop"',
        "C:\\Users\\kskzz\\miniconda3\\envs\\env_isaaclab\\python.exe",
        '$env:PYTHONPATH = $SourceRoot',
        '"wlr50_clean.ppo.cli"',
        '"reserve-run"',
        "finalize-run",
        '"stdout.log"',
        '"stderr.log"',
        "StartsWith($RunsPrefix",
        '"live_command_result.json"',
        "authoritative exit result",
    ):
        assert required in text
    assert "Remove-Item" not in text
    assert "-Force" not in text


def test_common_wrapper_rejects_duplicate_controlled_arguments() -> None:
    text = (_root() / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")
    for option in (
        "run-dir",
        "seed",
        "num-envs",
        "training-config",
        "interface-config",
        "fsm-config",
        "snapshot-root",
        "stage",
        "checkpoint",
        "checkpoint-manifest",
        "residual-mode",
        "seed-set",
        "deterministic",
        "headless",
        "no-headless",
        "soft-reset-acceptance",
        "vector-benchmark-matrix",
        "phase-effective-entry-holdout-acceptance",
        "phase-zero-residual-rollout-evidence",
    ):
        assert f'"{option}"' in text
    assert "foreach ($Argument in @($BaseCliArgs))" in text
    assert "foreach ($Argument in @($CliArgs))" in text
    assert "$BaseOptionNames.Contains" not in text
    assert "@($BaseOptionNames)" in text
    assert "StartsWith($OptionName" in text
    assert "CliArgs cannot override managed launcher/semantic argument" in text

    guard_block = text[
        text.index("$HelperOwnedArgumentNames") : text.index("function Get-LongOptionName")
    ]
    guarded = set(re.findall(r'"([a-z0-9][a-z0-9-]*)"', guard_block))
    parser_options = set()
    for module in (
        "cli.py",
        "vector_benchmark_matrix.py",
        "phase_effective_entry_holdout.py",
        "training_orchestration.py",
    ):
        source = (
            _root() / "src" / "wlr50_clean" / "ppo" / module
        ).read_text(encoding="utf-8")
        parser_options.update(
            re.findall(r'add_argument\(\s*"--([a-z0-9][a-z0-9-]*)"', source)
        )
    assert parser_options <= guarded, sorted(parser_options - guarded)
    assert "no-headless" in guarded

    cli = (_root() / "src" / "wlr50_clean" / "ppo" / "cli.py").read_text(
        encoding="utf-8"
    )
    assert cli.count("allow_abbrev=False") == 2


def test_multi_env_train_wrapper_requires_only_finalized_matrix() -> None:
    text = (_root() / "scripts" / "train_phase_residual_ppo.ps1").read_text(
        encoding="utf-8"
    )
    assert "[string]$VectorBenchmarkMatrix" in text
    assert '"--vector-benchmark-matrix"' in text
    assert "VectorZeroBenchmarkAcceptance" not in text
    assert "VectorNonzeroBenchmarkAcceptance" not in text


def test_checkpoint_publication_scripts_are_explicit_two_stage_gates() -> None:
    scripts = _root() / "scripts"
    best = (scripts / "publish_ppo_best_validation_checkpoint.ps1").read_text(
        encoding="utf-8"
    )
    improved = (scripts / "publish_ppo_improved_checkpoint.ps1").read_text(
        encoding="utf-8"
    )
    assert '-Subcommand "promote-best-validation"' in best
    assert "$PromotionDecision" in best
    assert "$CandidateCheckpoint" in best
    assert "$CandidateManifest" in best
    assert '-Subcommand "promote-improved"' in improved
    assert "$PromotionDecision" in improved
    assert "$LockedTestAggregate" in improved
    assert "$BestValidationManifest" in improved
    assert "$ValidationPromotionManifest" in improved
    assert '-Subcommand "evaluate"' not in improved
    assert '"--episode-count"' not in improved


def test_checkpoint_evaluation_passes_manifest_to_workers_and_aggregate() -> None:
    text = (_root() / "scripts" / "evaluate_ppo_checkpoint.ps1").read_text(
        encoding="utf-8"
    )
    assert text.count('"--checkpoint-manifest"') == 2


def test_inference_actor_script_loads_only_the_final_improved_checkpoint() -> None:
    text = (_root() / "scripts" / "export_ppo_inference_actor.ps1").read_text(
        encoding="utf-8"
    )
    assert '-Subcommand "export-inference-actor"' in text
    assert "checkpoint_improved.pt" in text
    assert "checkpoint_improved_manifest.json" in text
    assert '"--deterministic"' in text


def test_paired_evaluation_script_separates_exactly_five_role_inputs() -> None:
    text = (_root() / "scripts" / "export_paired_ppo_evaluation.ps1").read_text(
        encoding="utf-8"
    )
    assert '-Subcommand "export-paired-evaluation"' in text
    assert "[ValidateCount(5, 5)]" in text
    assert '"--baseline-episode-dir"' in text
    assert '"--candidate-episode-dir"' in text
    assert '"--candidate-checkpoint"' in text
    assert '"--candidate-manifest"' in text
    assert '"--baseline-aggregate"' in text
    assert '"--candidate-validation-aggregate"' in text
    assert "[string]$BaselineAggregate" in text
    assert "[string]$CandidateValidationAggregate" in text
    assert '$SeedSet -ne "validation"' in text
    assert "$Seed -ne 2001" in text
    assert "$EpisodeCount -ne 5" in text


def test_video_script_uses_two_fresh_live_processes_then_offline_publication() -> None:
    text = (_root() / "scripts" / "build_fsm_ppo_videos.ps1").read_text(
        encoding="utf-8"
    )
    assert text.count('-Subcommand "capture-video-source"') == 1
    assert 'Invoke-SingleVideoSourceCapture -Role "fsm"' in text
    assert 'Invoke-SingleVideoSourceCapture -Role "ppo"' in text
    assert '-Subcommand "publish-videos"' in text
    assert "build-videos" not in text
    assert '"--checkpoint-manifest", $CheckpointManifest' in text
    assert '"--no-headless"' in text
    assert "$Seed -ne 4001" in text
    assert '"--episode-count", "1"' in text
    assert "capture_process_instance_id" in text


def test_managed_artifact_scripts_hash_objectives_and_environment_lock() -> None:
    scripts = _root() / "scripts"
    for name in (
        "evaluate_ppo_checkpoint.ps1",
        "export_paired_ppo_evaluation.ps1",
        "publish_ppo_best_validation_checkpoint.ps1",
        "publish_ppo_improved_checkpoint.ps1",
        "export_ppo_inference_actor.ps1",
        "build_fsm_ppo_videos.ps1",
    ):
        text = (scripts / name).read_text(encoding="utf-8")
        assert "configs\\ppo_phase_objectives_v2.yaml" in text, name
        assert "configs\\environment_lock.json" in text, name


def test_effective_entry_consumers_hash_config_and_sidecar() -> None:
    scripts = _root() / "scripts"
    consumers = []
    for path in scripts.glob("*.ps1"):
        text = path.read_text(encoding="utf-8")
        if "$Configs = @(" not in text or path.name in {
            "build_phase_snapshots.ps1",
            "run_phase_effective_entry_calibration.ps1",
        }:
            continue
        consumers.append(path.name)
        assert "configs\\ppo_phase_effective_entry_v1.json" in text, path.name
        assert "configs\\ppo_phase_effective_entry_v1.sha256" in text, path.name
    assert consumers


def test_all_managed_wrappers_and_checkpoints_hash_runtime_loaded_v1_configs() -> None:
    scripts = _root() / "scripts"
    consumers = []
    required = (
        "configs\\ppo_action_projection.yaml",
        "configs\\ppo_observation_schema.json",
        "configs\\conformance_policy.yaml",
    )
    for path in scripts.glob("*.ps1"):
        text = path.read_text(encoding="utf-8")
        if "$Configs = @(" not in text:
            continue
        consumers.append(path.name)
        for config in required:
            assert text.count(config) == 1, (path.name, config)
    assert consumers

    cli = (_root() / "src" / "wlr50_clean" / "ppo" / "cli.py").read_text(
        encoding="utf-8"
    )
    for config in (
        'PROJECT_ROOT / "configs" / "ppo_action_projection.yaml"',
        'PROJECT_ROOT / "configs" / "ppo_observation_schema.json"',
        'PROJECT_ROOT / "configs" / "conformance_policy.yaml"',
    ):
        assert config in cli


def test_final_delivery_requires_the_complete_six_manifest_chain() -> None:
    text = (_root() / "scripts" / "finalize_ppo_delivery.ps1").read_text(
        encoding="utf-8"
    )
    assert "[ValidateCount(6, 64)]" in text
    assert text.count('"--checkpoint-manifest"') == 1


def test_initial_checkpoint_entry_is_two_managed_phases_and_cadence_runs_it_first() -> None:
    scripts = _root() / "scripts"
    initializer = (scripts / "initialize_zero_residual_checkpoint.ps1").read_text(
        encoding="utf-8"
    )
    cadence = (scripts / "run_ppo_training_cadence.ps1").read_text(
        encoding="utf-8"
    )
    common = (scripts / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")

    assert initializer.count("_invoke_ppo_cli.ps1") == 2
    assert '-RunKind "initial_checkpoint"' in initializer
    assert '-Subcommand "initialize-zero-residual"' in initializer
    assert '-RunKind "initial_checkpoint_publication"' in initializer
    assert '-Subcommand "publish-initial-zero-residual"' in initializer
    assert "never start a new" in initializer
    assert "Checkpoint-only canonical partial" in initializer
    assert "creation_runtime_identity_path" in initializer
    assert "SourceCheckpoint = Join-Path $InitializeRun" in initializer
    assert '"initialize-zero-residual"' in common
    assert '$InitializeScript = Join-Path $PSScriptRoot "initialize_zero_residual_checkpoint.ps1"' in cadence
    assert cadence.index("$InitialPublicationOutput") < cadence.index("$Stages = @(")
    assert cadence.index("$InitialPublicationOutput") < cadence.index(
        "foreach ($StagePlan in $Stages)"
    )
