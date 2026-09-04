[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Checkpoint = "outputs\ppo_phase_v1\checkpoints\checkpoint_best_validation.pt",
    [string]$CheckpointManifest = "outputs\ppo_phase_v1\checkpoints\checkpoint_best_validation_manifest.json",
    [ValidateSet("validation", "locked-test")][string]$SeedSet = "validation",
    [int]$Seed = 2001,
    [ValidateRange(1, 4096)][int]$NumEnvs = 1,
    [ValidateRange(5, 1000)][int]$EpisodeCount = 5,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
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
    "configs\environment_lock.json",
    "configs\frozen_successful_fsm.yaml",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json"
)
if ($NumEnvs -ne 1) {
    throw "deterministic checkpoint evaluation requires one fresh Isaac process per episode"
}
if ($EpisodeCount -ne 5) {
    throw "the versioned validation and locked-test sets each contain exactly five seeds"
}
$ExpectedFirstSeed = if ($SeedSet -eq "validation") { 2001 } else { 3001 }
if ($Seed -ne $ExpectedFirstSeed) {
    throw "$SeedSet evaluation must start at configured seed $ExpectedFirstSeed"
}

# Never reuse a PhysX scene between acceptance episodes.  Each worker creates
# one SimulationContext, evaluates one seed, seals its canonical streams, and
# exits.  A policy failure is still a successfully captured worker result, so
# all requested seeds are collected before the aggregate gate is evaluated.
$EvaluationRunDirs = @()
for ($EpisodeIndex = 0; $EpisodeIndex -lt $EpisodeCount; $EpisodeIndex++) {
    $EpisodeSeed = $Seed + $EpisodeIndex
    $WorkerArgs = @(
        "--training-config", $Configs[0],
        "--interface-config", $Configs[1],
        "--checkpoint", $Checkpoint,
        "--checkpoint-manifest", $CheckpointManifest,
        "--episode-count", "1",
        "--seed-set", $SeedSet,
        "--deterministic"
    )
    $InvocationOutput = @(
        & (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
            -RunKind "$SeedSet-checkpoint-evaluation" `
            -TrainingStage "checkpoint-evaluation-$SeedSet-fresh-process" `
            -Subcommand "evaluate" -ConfigPath $Configs -Seed $EpisodeSeed `
            -EnvironmentCount 1 -BaseCliArgs $WorkerArgs -CliArgs $CliArgs
    )
    $RunCandidates = @(
        $InvocationOutput | Where-Object {
            Test-Path -LiteralPath ([string]$_) -PathType Container
        }
    )
    if ($RunCandidates.Count -ne 1) {
        throw "single-episode worker did not return exactly one run directory for seed $EpisodeSeed"
    }
    $EvaluationRunDirs += [IO.Path]::GetFullPath([string]$RunCandidates[0])
}

$AggregateArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--checkpoint", $Checkpoint,
    "--checkpoint-manifest", $CheckpointManifest,
    "--episode-count", [string]$EpisodeCount,
    "--seed-set", $SeedSet,
    "--evaluation-role", "candidate",
    "--deterministic"
)
foreach ($EvaluationRunDir in $EvaluationRunDirs) {
    $AggregateArgs += @("--evaluation-run-dir", $EvaluationRunDir)
}

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "$SeedSet-checkpoint-evaluation-batch" `
    -TrainingStage "checkpoint-evaluation-$SeedSet-aggregate" `
    -Subcommand "aggregate-evaluations" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $AggregateArgs
