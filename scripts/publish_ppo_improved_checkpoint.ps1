[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$PromotionDecision,

    [Parameter(Mandatory = $true)]
    [string]$LockedTestAggregate,

    [string]$BestValidationCheckpoint = "outputs\ppo_phase_v1\checkpoints\checkpoint_best_validation.pt",
    [string]$BestValidationManifest = "outputs\ppo_phase_v1\checkpoints\checkpoint_best_validation_manifest.json",
    [string]$ValidationPromotionManifest = "outputs\ppo_phase_v1\manifests\checkpoint_best_validation_promotion_manifest.json",
    [string]$OutputRoot = "outputs\ppo_phase_v1",
    [int]$Seed = 3001,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Configs = @(
    "configs\ppo_training_phase_v1.yaml",
    "configs\ppo_interface_v2.yaml",
    "configs\ppo_observation_schema_v2.json",
    "configs\ppo_phase_action_masks_v2.yaml",
    "configs\ppo_phase_objectives_v2.yaml",
    "configs\ppo_reward_v2.yaml",
    "configs\ppo_termination_v2.yaml",
    "configs\ppo_domain_randomization_v2.yaml",
    "configs\environment_lock.json",
    "configs\frozen_successful_fsm.yaml",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--promotion-decision", $PromotionDecision,
    "--locked-test-aggregate", $LockedTestAggregate,
    "--best-validation-checkpoint", $BestValidationCheckpoint,
    "--best-validation-manifest", $BestValidationManifest,
    "--validation-promotion-manifest", $ValidationPromotionManifest,
    "--output-root", $OutputRoot
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "checkpoint-improved-publication" `
    -TrainingStage "locked-test-to-improved-offline" `
    -Subcommand "promote-improved" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
