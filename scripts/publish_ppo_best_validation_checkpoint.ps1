[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$PromotionDecision,

    [Parameter(Mandatory = $true)]
    [string]$CandidateCheckpoint,

    [Parameter(Mandatory = $true)]
    [string]$CandidateManifest,

    [string]$OutputRoot = "outputs\ppo_phase_v1",
    [int]$Seed = 2001,
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
    "--candidate-checkpoint", $CandidateCheckpoint,
    "--candidate-manifest", $CandidateManifest,
    "--output-root", $OutputRoot
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "checkpoint-best-validation-publication" `
    -TrainingStage "validation-to-best-offline" `
    -Subcommand "promote-best-validation" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
