[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1001,
    [ValidateSet("smoke", "phase-curriculum", "full-episode", "mild-randomization")]
    [string]$Stage = "phase-curriculum",
    [ValidateRange(1, 4096)][int]$NumEnvs = 1,
    [ValidateRange(1, 100000)][int]$PolicyDecisions,
    [string]$Checkpoint,
    [string]$CheckpointManifest,
    [string]$SoftResetAcceptance,
    [string]$VectorBenchmarkMatrix,
    [string]$PhaseEffectiveEntryHoldoutAcceptance,
    [string]$PhaseZeroResidualRolloutEvidence,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

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
    "configs\frozen_successful_fsm.yaml",
    "configs\environment_lock.json",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json",
    "configs\ppo_action_projection.yaml",
    "configs\ppo_observation_schema.json",
    "configs\conformance_policy.yaml"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--stage", $Stage
)
if ($PSBoundParameters.ContainsKey("PolicyDecisions")) {
    # The budget is a wrapper-owned semantic argument, never a generic override.
    $BaseArgs += @("--policy-decisions", [string]$PolicyDecisions)
}
if ($Stage -ne "smoke" -and [string]::IsNullOrWhiteSpace($Checkpoint)) {
    throw "$Stage training requires an explicit -Checkpoint; refusing to restart from the initial actor"
}
if ($Stage -eq "phase-curriculum") {
    if ([string]::IsNullOrWhiteSpace($PhaseEffectiveEntryHoldoutAcceptance)) {
        throw "phase-curriculum training requires -PhaseEffectiveEntryHoldoutAcceptance from aggregate_phase_effective_entry_holdout.ps1"
    }
    $BaseArgs += @(
        "--phase-effective-entry-holdout-acceptance",
        $PhaseEffectiveEntryHoldoutAcceptance
    )
    if ([string]::IsNullOrWhiteSpace($PhaseZeroResidualRolloutEvidence)) {
        throw "phase-curriculum training requires -PhaseZeroResidualRolloutEvidence from run_phase_zero_residual_rollout.ps1"
    }
    $BaseArgs += @(
        "--phase-zero-residual-rollout-evidence",
        $PhaseZeroResidualRolloutEvidence
    )
} elseif (-not [string]::IsNullOrWhiteSpace($PhaseEffectiveEntryHoldoutAcceptance)) {
    throw "-PhaseEffectiveEntryHoldoutAcceptance is only valid for phase-curriculum training"
} elseif (-not [string]::IsNullOrWhiteSpace($PhaseZeroResidualRolloutEvidence)) {
    throw "-PhaseZeroResidualRolloutEvidence is only valid for phase-curriculum training"
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
    if ([string]::IsNullOrWhiteSpace($VectorBenchmarkMatrix)) {
        throw "multi-env training requires -VectorBenchmarkMatrix from the six-slot offline aggregation"
    }
    $BaseArgs += @("--vector-benchmark-matrix", $VectorBenchmarkMatrix)
} elseif (-not [string]::IsNullOrWhiteSpace($VectorBenchmarkMatrix)) {
    throw "-VectorBenchmarkMatrix is only valid for multi-env training"
}
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "train" -TrainingStage $Stage -Subcommand "train" `
    -ConfigPath $Configs -Seed $Seed -EnvironmentCount $NumEnvs `
    -BaseCliArgs $BaseArgs -CliArgs $CliArgs
