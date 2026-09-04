[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 0,
    [ValidateRange(1, 4096)][int]$NumEnvs = 1,
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
    "--interface-config", $Configs[1]
)
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "preflight" -TrainingStage "preflight" -Subcommand "preflight" `
    -ConfigPath $Configs -Seed $Seed -EnvironmentCount $NumEnvs `
    -BaseCliArgs $BaseArgs -CliArgs $CliArgs
