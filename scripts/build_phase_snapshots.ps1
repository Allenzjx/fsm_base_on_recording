[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 0,
    [string]$SelectedTrial = "outputs\final\selected_success_trial.json",
    [string]$SnapshotRoot = "reference\ppo_phase_snapshots",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

$ErrorActionPreference = "Stop"
$Configs = @(
    "configs\ppo_training_phase_v1.yaml",
    "configs\ppo_interface_v2.yaml",
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
    "--selected-trial", $SelectedTrial,
    "--snapshot-root", $SnapshotRoot
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "phase_snapshot_validation" -TrainingStage "phase-snapshot-build" `
    -Subcommand "build-phase-snapshots" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
