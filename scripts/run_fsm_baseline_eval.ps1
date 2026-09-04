[CmdletBinding(PositionalBinding = $false)]
param(
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
    "configs\ppo_reward_v2.yaml",
    "configs\frozen_successful_fsm.yaml",
    "configs\ppo_termination_v2.yaml",
    "configs\ppo_phase_objectives_v2.yaml",
    "configs\ppo_domain_randomization_v2.yaml",
    "configs\environment_lock.json",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json"
)
if ($NumEnvs -ne 1) {
    throw "paired FSM evaluation requires one independently initialized live scene per seed"
}
if ($EpisodeCount -ne 5) {
    throw "the versioned paired baseline requires exactly five validation seeds"
}
if ($Seed -ne 2001) {
    throw "the versioned paired baseline must use configured validation seeds 2001-2005"
}

$EvaluationRunDirs = @()
$CanonicalEpisodeDirs = @()
for ($EpisodeIndex = 0; $EpisodeIndex -lt $EpisodeCount; $EpisodeIndex++) {
    $EpisodeSeed = $Seed + $EpisodeIndex
    $BaseArgs = @(
        "--training-config", $Configs[0],
        "--interface-config", $Configs[1],
        "--fsm-config", $Configs[5],
        "--episode-count", "1",
        "--seed-set", "validation",
        "--residual-mode", "zero",
        "--deterministic",
        "--evidence-only-worker"
    )
    $InvocationOutput = @(
        & (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
            -RunKind "baseline_fsm_eval" -TrainingStage "baseline-fsm-eval-fresh-process" `
            -Subcommand "baseline-eval" -ConfigPath $Configs -Seed $EpisodeSeed `
            -EnvironmentCount 1 -BaseCliArgs $BaseArgs -CliArgs $CliArgs
    )
    $RunCandidates = @(
        $InvocationOutput | Where-Object {
            Test-Path -LiteralPath ([string]$_) -PathType Container
        }
    )
    if ($RunCandidates.Count -ne 1) {
        throw "baseline worker did not return exactly one run directory for seed $EpisodeSeed"
    }
    $WorkerRunDir = [IO.Path]::GetFullPath([string]$RunCandidates[0])
    $CanonicalEpisodeDir = [IO.Path]::GetFullPath(
        (Join-Path $WorkerRunDir "episode_000_seed_$EpisodeSeed")
    )
    if (-not (Test-Path -LiteralPath $CanonicalEpisodeDir -PathType Container)) {
        throw "baseline worker omitted its canonical episode directory for seed $EpisodeSeed"
    }
    $EvaluationRunDirs += $WorkerRunDir
    $CanonicalEpisodeDirs += $CanonicalEpisodeDir
}

$AggregateArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--episode-count", [string]$EpisodeCount,
    "--seed-set", "validation",
    "--evaluation-role", "baseline",
    "--deterministic"
)
foreach ($EvaluationRunDir in $EvaluationRunDirs) {
    $AggregateArgs += @("--evaluation-run-dir", $EvaluationRunDir)
}

$AggregateOutput = @(
    & (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
        -RunKind "baseline-fsm-eval-batch" -TrainingStage "baseline-fsm-eval-aggregate" `
        -Subcommand "aggregate-evaluations" -ConfigPath $Configs -Seed $Seed `
        -EnvironmentCount 1 -BaseCliArgs $AggregateArgs
)
$AggregateRunCandidates = @(
    $AggregateOutput | Where-Object {
        Test-Path -LiteralPath ([string]$_) -PathType Container
    }
)
if ($AggregateRunCandidates.Count -ne 1) {
    throw "baseline aggregate did not return exactly one managed run directory"
}
$AggregateRunDir = [IO.Path]::GetFullPath([string]$AggregateRunCandidates[0])
$AggregatePath = Join-Path $AggregateRunDir "fsm_baseline_evaluation_aggregate.json"
if (-not (Test-Path -LiteralPath $AggregatePath -PathType Leaf)) {
    throw "baseline aggregate omitted fsm_baseline_evaluation_aggregate.json"
}
try {
    $Aggregate = Get-Content -LiteralPath $AggregatePath -Raw |
        ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "baseline aggregate is not valid JSON: $AggregatePath"
}
if ([string]$Aggregate.schema -ne "wlr50_clean.fresh_process_episode_batch.v1" -or
    [string]$Aggregate.role -ne "baseline" -or
    [string]$Aggregate.seed_set -ne "validation" -or
    [int]$Aggregate.episode_count -ne 5 -or
    $Aggregate.fresh_process_per_episode -ne $true -or
    $Aggregate.deterministic_evaluation -ne $true -or
    $Aggregate.passed -ne $true) {
    throw "baseline aggregate failed the canonical publication gate"
}

$ExportArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--episode-count", [string]$EpisodeCount,
    "--metrics-output-dir", "outputs\ppo_phase_v1\metrics"
)
foreach ($CanonicalEpisodeDir in $CanonicalEpisodeDirs) {
    $ExportArgs += @("--episode-dir", $CanonicalEpisodeDir)
}

& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "baseline-fsm-metrics" -TrainingStage "baseline-fsm-evaluation-export" `
    -Subcommand "export-baseline-evaluation" -ConfigPath $Configs -Seed $Seed `
    -EnvironmentCount 1 -BaseCliArgs $ExportArgs
