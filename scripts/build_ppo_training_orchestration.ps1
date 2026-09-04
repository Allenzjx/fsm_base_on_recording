[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$TrainingRunDir,

    [Parameter(Mandatory = $true)]
    [string[]]$ScreeningRunDir,

    [Parameter(Mandatory = $true)]
    [string]$InitialCheckpointPublicationRun,

    [Parameter(Mandatory = $true)]
    [string]$VectorBenchmarkMatrix,

    [string]$TrainingConfig = "configs\ppo_training_phase_v1.yaml",

    [string[]]$PromotionDecision = @(),

    [int]$Seed = 1001,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($TrainingRunDir.Count -eq 0 -or
    $TrainingRunDir.Count -ne $ScreeningRunDir.Count) {
    throw "TrainingRunDir and ScreeningRunDir must be non-empty one-to-one ordered lists"
}

$Configs = @(
    $TrainingConfig,
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
foreach ($PathValue in $TrainingRunDir) {
    $BaseArgs += @("--training-run-dir", $PathValue)
}
foreach ($PathValue in $ScreeningRunDir) {
    $BaseArgs += @("--screening-run-dir", $PathValue)
}
$BaseArgs += @(
    "--initial-checkpoint-publication-run", $InitialCheckpointPublicationRun,
    "--training-config", $TrainingConfig,
    "--vector-benchmark-matrix", $VectorBenchmarkMatrix
)
foreach ($PathValue in $PromotionDecision) {
    $BaseArgs += @("--promotion-decision", $PathValue)
}

# This command only validates immutable offline evidence.  Its reservation is
# always single-environment even though the smoke stage used matrix-selected N.
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "training_orchestration" `
    -TrainingStage "training-orchestration-prefinal" `
    -Subcommand "build-manifest" `
    -CliModule "wlr50_clean.ppo.training_orchestration" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -CliArgs $CliArgs
