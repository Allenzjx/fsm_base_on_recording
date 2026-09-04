[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1001,
    [ValidateRange(1, 4096)][int]$NumEnvs = 1,
    [ValidateRange(1, 100000)][int]$PolicyDecisions = 10000,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

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
    "configs\environment_lock.json",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--episode-count", "1",
    "--seed-set", "train",
    "--policy-decisions", [string]$PolicyDecisions,
    "--residual-mode", "bounded-smoke"
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "nonzero_residual_smoke" -TrainingStage "nonzero-residual-smoke" `
    -Subcommand "nonzero-residual-smoke" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount $NumEnvs -BaseCliArgs $BaseArgs -CliArgs $CliArgs
