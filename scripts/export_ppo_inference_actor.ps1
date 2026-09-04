[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Checkpoint = "outputs\ppo_phase_v1\checkpoints\checkpoint_improved.pt",
    [string]$CheckpointManifest = "outputs\ppo_phase_v1\checkpoints\checkpoint_improved_manifest.json",
    [string]$OutputRoot = "outputs\ppo_phase_v1",
    [int]$Seed = 4001,
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
    "--checkpoint", $Checkpoint,
    "--checkpoint-manifest", $CheckpointManifest,
    "--output-root", $OutputRoot,
    "--deterministic"
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "inference-actor-export" `
    -TrainingStage "improved-inference-actor-export" `
    -Subcommand "export-inference-actor" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
