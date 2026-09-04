[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineAggregate,

    [Parameter(Mandatory = $true)]
    [string]$CandidateValidationAggregate,

    [Parameter(Mandatory = $true)]
    [ValidateCount(5, 5)]
    [string[]]$BaselineEpisodeDir,

    [Parameter(Mandatory = $true)]
    [ValidateCount(5, 5)]
    [string[]]$CandidateEpisodeDir,

    [Parameter(Mandatory = $true)]
    [string]$CandidateCheckpoint,

    [Parameter(Mandatory = $true)]
    [string]$CandidateManifest,

    [ValidateSet("validation")]
    [string]$SeedSet = "validation",
    [int]$Seed = 2001,
    [int]$EpisodeCount = 5,
    [string]$MetricsOutputDir = "outputs\ppo_phase_v1\metrics",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($SeedSet -ne "validation" -or $Seed -ne 2001 -or $EpisodeCount -ne 5) {
    throw "paired evaluation export requires validation seeds 2001-2005 exactly"
}
if (($BaselineEpisodeDir | Select-Object -Unique).Count -ne 5) {
    throw "BaselineEpisodeDir must contain five distinct canonical directories"
}
if (($CandidateEpisodeDir | Select-Object -Unique).Count -ne 5) {
    throw "CandidateEpisodeDir must contain five distinct canonical directories"
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
    "configs\environment_lock.json",
    "configs\frozen_successful_fsm.yaml",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json",
    "configs\ppo_action_projection.yaml",
    "configs\ppo_observation_schema.json",
    "configs\conformance_policy.yaml"
)
$BaseArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--seed-set", $SeedSet,
    "--episode-count", [string]$EpisodeCount,
    "--candidate-checkpoint", $CandidateCheckpoint,
    "--candidate-manifest", $CandidateManifest,
    "--baseline-aggregate", $BaselineAggregate,
    "--candidate-validation-aggregate", $CandidateValidationAggregate,
    "--metrics-output-dir", $MetricsOutputDir
)
foreach ($Directory in $BaselineEpisodeDir) {
    $BaseArgs += @("--baseline-episode-dir", $Directory)
}
foreach ($Directory in $CandidateEpisodeDir) {
    $BaseArgs += @("--candidate-episode-dir", $Directory)
}

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "paired-ppo-evaluation-export" `
    -TrainingStage "validation-paired-evaluation-offline" `
    -Subcommand "export-paired-evaluation" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
