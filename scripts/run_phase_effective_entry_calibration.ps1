[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateRange(1002, 1002)]
    [int]$Seed = 1002,
    [Parameter(Mandatory = $true)]
    [ValidateSet("P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13")]
    [string]$Phase,
    [ValidateSet(1)]
    [int]$PrimePhysicsSteps = 1,
    [string]$SnapshotRoot = "reference\ppo_phase_snapshots",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# The existing effective-entry files are deliberately absent from this input
# set: calibration produces their replacement, so binding their old bytes
# would create a circular contract that becomes invalid when published.
# They remain covered by the launcher's full committed-runtime identity.
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
    "configs\recording_motion_contract.json",
    "configs\ppo_action_projection.yaml",
    "configs\ppo_observation_schema.json",
    "configs\conformance_policy.yaml"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--snapshot-root", $SnapshotRoot,
    "--phase-snapshot-prime-physics-steps", $PrimePhysicsSteps,
    "--episode-count", "1",
    "--seed-set", "train",
    "--deterministic",
    "--phase", $Phase,
    "--calibrate-effective-entry"
)

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "phase_effective_entry_calibration" `
    -TrainingStage "phase-effective-entry-calibration" `
    -Subcommand "phase-snapshot-live-probe" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -CliArgs $CliArgs
