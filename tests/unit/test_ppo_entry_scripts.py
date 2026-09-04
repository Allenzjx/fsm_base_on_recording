from pathlib import Path


REQUESTED = (
    "ppo_preflight.ps1",
    "run_fsm_baseline_eval.ps1",
    "build_phase_snapshots.ps1",
    "run_zero_residual_live_validation.ps1",
    "run_nonzero_residual_smoke.ps1",
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
