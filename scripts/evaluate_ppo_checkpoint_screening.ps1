[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [Parameter(Mandatory = $true)]
    [string]$CheckpointManifest,

    [ValidateRange(2001, 2005)]
    [int]$Seed = 2001,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Configs = @(
    "configs\ppo_training_phase_v1.yaml",
    "configs\ppo_interface_v2.yaml",
    "configs\ppo_phase_effective_entry_v1.json",
    "configs\ppo_phase_effective_entry_v1.sha256",
    "configs\ppo_observation_schema_v2.json",
    "configs\ppo_phase_action_masks_v2.yaml",
    "configs\ppo_phase_objectives_v2.yaml",
    "configs\ppo_reward_v2.yaml",
    "configs\ppo_termination_v2.yaml",
    "configs\ppo_domain_randomization_v2.yaml",
    "configs\environment_lock.json",
    "configs\frozen_successful_fsm.yaml",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json",
    "configs\ppo_action_projection.yaml",
    "configs\ppo_observation_schema.json",
    "configs\conformance_policy.yaml"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--checkpoint", $Checkpoint,
    "--checkpoint-manifest", $CheckpointManifest,
    "--episode-count", "1",
    "--seed-set", "validation",
    "--deterministic"
)

# The live worker records physical failure as evidence and exits successfully;
# only the later five-seed paired gate may authorize checkpoint promotion.
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "validation-checkpoint-screening" `
    -TrainingStage "checkpoint-screening-fresh-process" `
    -Subcommand "evaluate" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -CliArgs $CliArgs
