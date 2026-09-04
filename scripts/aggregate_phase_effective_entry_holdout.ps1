[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(1003)]
    [int]$Seed = 1003,

    [string[]]$ProbeRunDir = @(),

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Phases = @(
    "P02", "P03", "P04", "P05", "P06", "P07",
    "P08", "P09", "P10", "P11", "P12", "P13"
)
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

$Workers = @($ProbeRunDir)
if ($Workers.Count -eq 0) {
    $Workers = @(
        foreach ($Phase in $Phases) {
            # Each script invocation starts and finalizes a separate Python
            # process.  The aggregator later verifies the per-process UUID,
            # phase, seed, runtime identity, and fresh/reused reset proofs.
            $Rows = @(
                @(
                    & (Join-Path $PSScriptRoot "run_phase_snapshot_live_probe.ps1") `
                        -Seed $Seed -PrimePhysicsSteps 1 -Phase $Phase
                ) | Where-Object {
                    -not [string]::IsNullOrWhiteSpace([string]$_)
                }
            )
            if ($Rows.Count -ne 1) {
                throw "Holdout worker $Phase did not return exactly one run directory"
            }
            [string]$Rows[0]
        }
    )
} elseif ($Workers.Count -ne $Phases.Count) {
    throw "-ProbeRunDir must contain exactly twelve finalized one-phase probe runs"
}

$BaseArgs = @()
foreach ($Worker in $Workers) {
    $BaseArgs += @("--probe-run-dir", $Worker)
}

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "phase_effective_entry_holdout" `
    -TrainingStage "effective-entry-holdout-aggregation" `
    -Subcommand "aggregate" `
    -CliModule "wlr50_clean.ppo.phase_effective_entry_holdout" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $BaseArgs `
    -CliArgs $CliArgs
