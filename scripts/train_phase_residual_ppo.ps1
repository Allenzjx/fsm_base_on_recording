[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1001,
    [ValidateSet("smoke", "phase-curriculum", "full-episode", "mild-randomization")]
    [string]$Stage = "phase-curriculum",
    [ValidateRange(1, 4096)][int]$NumEnvs = 1,
    [string]$Checkpoint,
    [string]$CheckpointManifest,
    [string]$SoftResetAcceptance,
    [string]$VectorZeroBenchmarkAcceptance,
    [string]$VectorNonzeroBenchmarkAcceptance,
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
    "--stage", $Stage
)
if ($Stage -ne "smoke" -and [string]::IsNullOrWhiteSpace($Checkpoint)) {
    throw "$Stage training requires an explicit -Checkpoint; refusing to restart from the initial actor"
}
if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    $BaseArgs += @("--checkpoint", $Checkpoint)
    if (-not [string]::IsNullOrWhiteSpace($CheckpointManifest)) {
        $BaseArgs += @("--checkpoint-manifest", $CheckpointManifest)
    }
} elseif (-not [string]::IsNullOrWhiteSpace($CheckpointManifest)) {
    throw "-CheckpointManifest cannot be supplied without -Checkpoint"
}
if ($NumEnvs -eq 1) {
    if ([string]::IsNullOrWhiteSpace($SoftResetAcceptance)) {
        throw "single-env training requires -SoftResetAcceptance from run_soft_reset_equivalence.ps1"
    }
    $BaseArgs += @("--soft-reset-acceptance", $SoftResetAcceptance)
} elseif (-not [string]::IsNullOrWhiteSpace($SoftResetAcceptance)) {
    $BaseArgs += @("--soft-reset-acceptance", $SoftResetAcceptance)
}
if ($NumEnvs -gt 1) {
    if ([string]::IsNullOrWhiteSpace($VectorZeroBenchmarkAcceptance) -or
        [string]::IsNullOrWhiteSpace($VectorNonzeroBenchmarkAcceptance)) {
        throw "multi-env training requires both vector zero and bounded-nonzero benchmark acceptance artifacts"
    }
    $BaseArgs += @(
        "--vector-zero-benchmark-acceptance", $VectorZeroBenchmarkAcceptance,
        "--vector-nonzero-benchmark-acceptance", $VectorNonzeroBenchmarkAcceptance
    )
}
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "train" -TrainingStage $Stage -Subcommand "train" `
    -ConfigPath $Configs -Seed $Seed -EnvironmentCount $NumEnvs `
    -BaseCliArgs $BaseArgs -CliArgs $CliArgs
