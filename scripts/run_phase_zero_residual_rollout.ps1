[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1004,
    [Parameter(Mandatory = $true)]
    [string]$PhaseEffectiveEntryHoldoutAcceptance,
    [string]$SnapshotRoot = "reference\ppo_phase_snapshots",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PhaseEffectiveEntryHoldoutAcceptance)) {
    throw "-PhaseEffectiveEntryHoldoutAcceptance must name the finalized external holdout gate"
}

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
    "--snapshot-root", $SnapshotRoot,
    "--phase-snapshot-prime-physics-steps", "1",
    "--phase-effective-entry-holdout-acceptance", $PhaseEffectiveEntryHoldoutAcceptance,
    "--episode-count", "13",
    "--policy-decisions", "64",
    "--seed-set", "train",
    "--residual-mode", "zero",
    "--deterministic"
)

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "phase_zero_residual_rollout" `
    -TrainingStage "phase-zero-residual-rollout" `
    -Subcommand "phase-zero-residual-rollout" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -CliArgs $CliArgs
