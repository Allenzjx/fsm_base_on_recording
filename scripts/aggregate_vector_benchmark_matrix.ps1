[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateCount(6, 6)]
    [string[]]$BenchmarkPath,

    [int]$Seed = 1001,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
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
$BaseArgs = @()
foreach ($PathValue in $BenchmarkPath) {
    $BaseArgs += @("--benchmark", $PathValue)
}

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "vector_benchmark_matrix" `
    -TrainingStage "backend-benchmark-selection" `
    -Subcommand "aggregate" `
    -CliModule "wlr50_clean.ppo.vector_benchmark_matrix" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -CliArgs $CliArgs
