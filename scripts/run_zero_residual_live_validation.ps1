[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 2001,
    [ValidateRange(1, 4096)][int]$NumEnvs = 1,
    [ValidateRange(5, 1000)][int]$EpisodeCount = 5,
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
if ($NumEnvs -ne 1) {
    throw "Gate A uses one fresh Isaac process per episode; use benchmark_vectorized_ppo.ps1 for batched environments"
}

# Reusing one PhysX scene across trials was empirically shown to produce a
# different P10-entry outcome after the first reset.  Each acceptance episode
# therefore owns a fresh SimulationContext/process, matching the successful
# frozen baseline runtime and preventing hidden cross-episode simulator state.
for ($EpisodeIndex = 0; $EpisodeIndex -lt $EpisodeCount; $EpisodeIndex++) {
    $EpisodeSeed = $Seed + $EpisodeIndex
    $BaseArgs = @(
        "--training-config", $Configs[0],
        "--interface-config", $Configs[1],
        "--episode-count", "1",
        "--deterministic",
        "--residual-mode", "zero",
        "--seed-set", "validation"
    )
    & (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
        -RunKind "zero_residual_live" -TrainingStage "zero-residual-live-fresh-process" `
        -Subcommand "zero-residual-live" -ConfigPath $Configs -Seed $EpisodeSeed `
        -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
}
