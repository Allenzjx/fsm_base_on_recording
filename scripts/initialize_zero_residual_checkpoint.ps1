[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 1001,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$OutputRoot = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot "outputs\ppo_phase_v1")
)
$Checkpoint = Join-Path $OutputRoot "checkpoints\checkpoint_initial_zero_residual.pt"
$Manifest = Join-Path $OutputRoot "checkpoints\checkpoint_initial_zero_residual_manifest.json"
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
    "configs\recording_motion_contract.json"
)

$CheckpointExists = Test-Path -LiteralPath $Checkpoint -PathType Leaf
$ManifestExists = Test-Path -LiteralPath $Manifest -PathType Leaf
$SourceCheckpoint = $Checkpoint
$SourceManifest = $Manifest
$InitializerRoot = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot "runs\ppo_phase_v1\initial_checkpoint")
)
$InitializerPrefix = $InitializerRoot.TrimEnd(
    [IO.Path]::DirectorySeparatorChar
) + [IO.Path]::DirectorySeparatorChar

if (-not $CheckpointExists -and -not $ManifestExists) {
    $InitializeOutput = @(
        & (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
            -RunKind "initial_checkpoint" `
            -TrainingStage "initialize-zero-residual" `
            -Subcommand "initialize-zero-residual" `
            -ConfigPath $Configs `
            -Seed $Seed `
            -EnvironmentCount 1 `
            -BaseCliArgs @(
                "--training-config", $Configs[0],
                "--interface-config", $Configs[1],
                "--stage", "smoke",
                "--seed-set", "train",
                "--episode-count", "1",
                "--deterministic"
            ) `
            -CliArgs $CliArgs
    )
    $InitializeRows = @(
        $InitializeOutput |
            Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
    )
    if ($InitializeRows.Count -ne 1) {
        throw "Initializer did not return exactly one finalized managed run directory"
    }
    $InitializeRun = [IO.Path]::GetFullPath([string]$InitializeRows[0])
    if (-not $InitializeRun.StartsWith(
            $InitializerPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "Initializer returned a run outside its managed run kind: $InitializeRun"
    }
    $SourceCheckpoint = Join-Path $InitializeRun "checkpoint_initial_zero_residual.pt"
    $SourceManifest = Join-Path $InitializeRun "checkpoint_initial_zero_residual_manifest.json"
    foreach ($Required in @(
        (Join-Path $InitializeRun "run_manifest.json"),
        (Join-Path $InitializeRun "initial_checkpoint_result.json"),
        $SourceCheckpoint,
        $SourceManifest
    )) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "Finalized initializer omitted required evidence: $Required"
        }
    }
} elseif ($CheckpointExists -ne $ManifestExists) {
    # The publisher commits immutable manifest bytes first and the checkpoint
    # pathname last.  A terminated publication can therefore leave one exact
    # side.  Recover from the original finalized creator; never start a new
    # initializer whose embedded creator identity would differ.
    if ($ManifestExists) {
        try {
            $PartialManifest = Get-Content -LiteralPath $Manifest -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Partial canonical initial manifest is not valid JSON"
        }
        $IdentityValue = [string]$PartialManifest.creation_runtime_identity_path
        if ([string]::IsNullOrWhiteSpace($IdentityValue) -or
            -not [IO.Path]::IsPathRooted($IdentityValue)) {
            throw "Partial canonical manifest omits an absolute creator identity"
        }
        $IdentityPath = [IO.Path]::GetFullPath($IdentityValue)
        if ([IO.Path]::GetFileName($IdentityPath) -cne
            "committed_runtime_identity.before.json") {
            throw "Partial canonical manifest names the wrong creator identity"
        }
        $InitializeRun = [IO.Path]::GetFullPath(
            (Split-Path -Parent $IdentityPath)
        )
        if (-not $InitializeRun.StartsWith(
                $InitializerPrefix,
                [StringComparison]::OrdinalIgnoreCase
            )) {
            throw "Partial canonical manifest creator is outside the managed initializer root"
        }
        $SourceCheckpoint = Join-Path $InitializeRun `
            "checkpoint_initial_zero_residual.pt"
        $SourceManifest = Join-Path $InitializeRun `
            "checkpoint_initial_zero_residual_manifest.json"
    } else {
        if (-not (Test-Path -LiteralPath $InitializerRoot -PathType Container)) {
            throw "Checkpoint-only canonical partial has no managed initializer root"
        }
        $PartialHash = (
            Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        $Candidates = @(
            Get-ChildItem -LiteralPath $InitializerRoot -Directory |
                Where-Object {
                    ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0
                } |
                ForEach-Object {
                    $CandidateCheckpoint = Join-Path $_.FullName `
                        "checkpoint_initial_zero_residual.pt"
                    $CandidateManifest = Join-Path $_.FullName `
                        "checkpoint_initial_zero_residual_manifest.json"
                    if ((Test-Path -LiteralPath $CandidateCheckpoint -PathType Leaf) -and
                        (Test-Path -LiteralPath $CandidateManifest -PathType Leaf) -and
                        (Test-Path -LiteralPath (Join-Path $_.FullName "run_manifest.json") -PathType Leaf) -and
                        (Test-Path -LiteralPath (Join-Path $_.FullName "initial_checkpoint_result.json") -PathType Leaf) -and
                        (Get-FileHash -LiteralPath $CandidateCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ceq $PartialHash) {
                        $_.FullName
                    }
                }
        )
        if ($Candidates.Count -ne 1) {
            throw "Checkpoint-only canonical partial does not bind exactly one finalized initializer source"
        }
        $InitializeRun = [IO.Path]::GetFullPath([string]$Candidates[0])
        $SourceCheckpoint = Join-Path $InitializeRun `
            "checkpoint_initial_zero_residual.pt"
        $SourceManifest = Join-Path $InitializeRun `
            "checkpoint_initial_zero_residual_manifest.json"
    }
    foreach ($Required in @(
        (Join-Path $InitializeRun "run_manifest.json"),
        (Join-Path $InitializeRun "initial_checkpoint_result.json"),
        $SourceCheckpoint,
        $SourceManifest
    )) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "Partial recovery source omits finalized initializer evidence: $Required"
        }
    }
}

# Publication is a separate offline managed run.  Therefore no canonical file
# exists until the live initializer and its wrapper finalization have succeeded.
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "initial_checkpoint_publication" `
    -TrainingStage "initial-checkpoint-publication" `
    -Subcommand "publish-initial-zero-residual" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs @(
        "--training-config", $Configs[0],
        "--interface-config", $Configs[1],
        "--seed-set", "train",
        "--source-checkpoint", $SourceCheckpoint,
        "--source-manifest", $SourceManifest,
        "--output-root", $OutputRoot
    )
