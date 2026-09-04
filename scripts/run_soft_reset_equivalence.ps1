[CmdletBinding(PositionalBinding = $false)]
param(
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
    "configs\frozen_successful_fsm.yaml",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json",
    "configs\environment_lock.json"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--episode-count", "2",
    "--seed-set", "validation",
    "--residual-mode", "zero",
    "--deterministic"
)

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "soft-reset-equivalence" `
    -TrainingStage "soft-reset-equivalence-live" `
    -Subcommand "soft-reset-equivalence" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
