[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$TrainingOrchestrationManifest,

    [Parameter(Mandatory = $true)]
    [string]$PureFsmAggregate,

    [Parameter(Mandatory = $true)]
    [string]$CheckpointInitialAggregate,

    [Parameter(Mandatory = $true)]
    [string]$CheckpointSmokeAggregate,

    [Parameter(Mandatory = $true)]
    [string]$CheckpointBestAggregate,

    [Parameter(Mandatory = $true)]
    [string]$CheckpointImprovedAggregate,

    [Parameter(Mandatory = $true)]
    [string]$ValidationAggregate,

    [Parameter(Mandatory = $true)]
    [string]$ValidationPromotionDecision,

    [Parameter(Mandatory = $true)]
    [string]$LockedTestAggregate,

    [Parameter(Mandatory = $true)]
    [ValidateCount(6, 64)]
    [string[]]$CheckpointManifest,

    [Parameter(Mandatory = $true)]
    [string]$InferenceActorExportRunDir,

    [Parameter(Mandatory = $true)]
    [string]$VideoValidation,

    [Parameter(Mandatory = $true)]
    [string]$VideoChecksums,

    [string]$PhaseObjectivesConfig = "configs\ppo_phase_objectives_v2.yaml",
    [string]$PhaseActionConfig = "configs\ppo_phase_action_masks_v2.yaml",
    [string]$RewardConfig = "configs\ppo_reward_v2.yaml",
    [string]$RewardMigrationConfig = "configs\ppo_reward_v1_to_v2_migration.yaml",
    [string]$RewardStreamFilename = "reward_15hz.jsonl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$SourceRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "src"))
$IsaacPython = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"

if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Locked Isaac Python is missing: $IsaacPython"
}

$Arguments = @(
    "-P", "-m", "wlr50_clean.ppo.delivery_cli", "deliver",
    "--output-root", $OutputRoot,
    "--training-orchestration-manifest", $TrainingOrchestrationManifest,
    "--pure-fsm-aggregate", $PureFsmAggregate,
    "--checkpoint-initial-aggregate", $CheckpointInitialAggregate,
    "--checkpoint-smoke-aggregate", $CheckpointSmokeAggregate,
    "--checkpoint-best-aggregate", $CheckpointBestAggregate,
    "--checkpoint-improved-aggregate", $CheckpointImprovedAggregate,
    "--validation-aggregate", $ValidationAggregate,
    "--validation-promotion-decision", $ValidationPromotionDecision,
    "--locked-test-aggregate", $LockedTestAggregate,
    "--inference-actor-export-run-dir", $InferenceActorExportRunDir,
    "--video-validation", $VideoValidation,
    "--video-checksums", $VideoChecksums,
    "--phase-objectives-config", $PhaseObjectivesConfig,
    "--phase-action-config", $PhaseActionConfig,
    "--reward-config", $RewardConfig,
    "--reward-migration-config", $RewardMigrationConfig,
    "--reward-stream-filename", $RewardStreamFilename
)
foreach ($Manifest in $CheckpointManifest) {
    $Arguments += @("--checkpoint-manifest", $Manifest)
}

$env:PYTHONPATH = $SourceRoot
$env:PYTHONNOUSERSITE = "1"
Push-Location $ProjectRoot
try {
    & $IsaacPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Strict five-role PPO delivery failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
