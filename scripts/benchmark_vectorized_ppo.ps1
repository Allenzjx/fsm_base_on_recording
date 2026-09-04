[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1001,
    [ValidateSet(8, 16, 32)][int]$NumEnvs = 8,
    [ValidateRange(1, 10000)][int]$MeasuredTicks = 32,
    [ValidateSet("zero", "bounded-smoke")][string]$ResidualMode = "zero",
    [ValidateRange(1, 16)][int]$PolicyDecisions = 2,
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
    "--measured-ticks", [string]$MeasuredTicks,
    "--seed-set", "train",
    "--residual-mode", $ResidualMode,
    "--policy-decisions", [string]$PolicyDecisions
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "vector_benchmark" -TrainingStage "backend-benchmark" `
    -Subcommand "vector-benchmark" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount $NumEnvs -BaseCliArgs $BaseArgs -CliArgs $CliArgs
