[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1001,
    [ValidateSet(1)]
    [int]$PrimePhysicsSteps = 1,
    [AllowNull()]
    [ValidateSet("P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13")]
    [string]$Phase = $null,
    [string]$SnapshotRoot = "reference\ppo_phase_snapshots",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
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
    "--snapshot-root", $SnapshotRoot,
    "--phase-snapshot-prime-physics-steps", $PrimePhysicsSteps,
    "--episode-count", "1",
    "--seed-set", "train",
    "--deterministic"
)
if ($null -ne $Phase) {
    $BaseArgs += @("--phase", $Phase)
}

# A failed reset is the expected useful outcome of this diagnostic.  The
# common launcher may return exit-code 2 only after it has sealed the failure,
# repeated the frozen/runtime checks, and finalized the immutable run.
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "phase_snapshot_live_probe" `
    -TrainingStage "phase-snapshot-live-probe" `
    -Subcommand "phase-snapshot-live-probe" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -ReturnFinalizedEvidenceFailure `
    -CliArgs $CliArgs
