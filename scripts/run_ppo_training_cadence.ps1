[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateCount(5, 5)]
    [string[]]$BaselineEpisodeDir,

    [Parameter(Mandatory = $true)]
    [string]$BaselineAggregate,

    [Parameter(Mandatory = $true)]
    [string]$SoftResetAcceptance,

    [Parameter(Mandatory = $true)]
    [string]$VectorBenchmarkMatrix,

    [Parameter(Mandatory = $true)]
    [string]$PhaseEffectiveEntryHoldoutAcceptance,

    [Parameter(Mandatory = $true)]
    [string]$PhaseZeroResidualRolloutEvidence,

    [string[]]$PromotionDecision = @(),

    [int]$Seed = 1001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$RunsRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "runs\ppo_phase_v1"))
$InitializeScript = Join-Path $PSScriptRoot "initialize_zero_residual_checkpoint.ps1"
$TrainScript = Join-Path $PSScriptRoot "train_phase_residual_ppo.ps1"
$ScreenScript = Join-Path $PSScriptRoot "evaluate_ppo_checkpoint_screening.ps1"
$BuildScript = Join-Path $PSScriptRoot "build_ppo_training_orchestration.ps1"
$FiveSeedScript = Join-Path $PSScriptRoot "evaluate_ppo_checkpoint.ps1"
$PairedExportScript = Join-Path $PSScriptRoot "export_paired_ppo_evaluation.ps1"
$RequestedChunkDecisions = 10000
$ValidationSeeds = @(2001, 2002, 2003, 2004, 2005)
$PromotionGates = @(
    "p01_p13_completed",
    "task_success_rate_not_below_fsm",
    "body_collision_zero",
    "wheel_only_climb_zero",
    "fall_or_physics_explosion_zero",
    "safety_abort_zero",
    "duration_each_under_200_s",
    "duration_not_over_fsm_by_15pct",
    "frozen_hashes_unchanged",
    "recording_runtime_access_zero",
    "global_stability_improvement_at_least_5pct",
    "at_least_4_of_5_priority_phases_improve",
    "no_priority_phase_degrades_over_10pct",
    "one_visual_key_metric_gate",
    "level_calibration_quality_passed",
    "residual_activity_calibrated",
    "priority_phases_have_real_residual",
    "at_least_10_phases_have_real_residual"
)

foreach ($Required in @(
    $InitializeScript,
    $TrainScript,
    $ScreenScript,
    $BuildScript,
    $FiveSeedScript,
    $PairedExportScript
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required cadence wrapper is missing: $Required"
    }
}

$ResolvedBaselineEpisodes = @()
for ($Index = 0; $Index -lt $BaselineEpisodeDir.Count; $Index++) {
    $Directory = [IO.Path]::GetFullPath(
        $(if ([IO.Path]::IsPathRooted($BaselineEpisodeDir[$Index])) {
            $BaselineEpisodeDir[$Index]
        } else {
            Join-Path $ProjectRoot $BaselineEpisodeDir[$Index]
        })
    )
    $SummaryPath = Join-Path $Directory "episode_summary.json"
    if (-not (Test-Path -LiteralPath $Directory -PathType Container) -or
        -not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Baseline canonical episode is incomplete: $Directory"
    }
    try {
        $Summary = Get-Content -LiteralPath $SummaryPath -Raw |
            ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Baseline episode summary is not valid JSON: $SummaryPath"
    }
    if ([int]$Summary.seed -ne $ValidationSeeds[$Index]) {
        throw "Baseline episodes must be ordered validation seeds 2001-2005"
    }
    $ResolvedBaselineEpisodes += $Directory
}
if (($ResolvedBaselineEpisodes | Select-Object -Unique).Count -ne 5) {
    throw "BaselineEpisodeDir must contain five distinct canonical directories"
}
$BaselineAggregatePath = [IO.Path]::GetFullPath(
    $(if ([IO.Path]::IsPathRooted($BaselineAggregate)) {
        $BaselineAggregate
    } else {
        Join-Path $ProjectRoot $BaselineAggregate
    })
)
if (-not (Test-Path -LiteralPath $BaselineAggregatePath -PathType Leaf) -or
    [IO.Path]::GetFileName($BaselineAggregatePath) -cne "fsm_baseline_evaluation_aggregate.json") {
    throw "Finalized baseline aggregate is missing or misnamed: $BaselineAggregatePath"
}

$MatrixPath = [IO.Path]::GetFullPath(
    $(if ([IO.Path]::IsPathRooted($VectorBenchmarkMatrix)) {
        $VectorBenchmarkMatrix
    } else {
        Join-Path $ProjectRoot $VectorBenchmarkMatrix
    })
)
if (-not (Test-Path -LiteralPath $MatrixPath -PathType Leaf)) {
    throw "Finalized vector benchmark matrix is missing: $MatrixPath"
}
try {
    $Matrix = Get-Content -LiteralPath $MatrixPath -Raw | ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "Vector benchmark matrix is not valid JSON: $MatrixPath"
}
$SelectedNumEnvs = [int]$Matrix.selected_num_envs
if ([string]$Matrix.schema -ne "wlr50_clean.vector_benchmark_matrix.v1" -or
    $Matrix.passed -ne $true -or
    $SelectedNumEnvs -notin @(8, 16, 32)) {
    throw "Vector benchmark matrix does not expose a valid selected N"
}

$HoldoutPath = [IO.Path]::GetFullPath(
    $(if ([IO.Path]::IsPathRooted($PhaseEffectiveEntryHoldoutAcceptance)) {
        $PhaseEffectiveEntryHoldoutAcceptance
    } else {
        Join-Path $ProjectRoot $PhaseEffectiveEntryHoldoutAcceptance
    })
)
$HoldoutRoot = [IO.Path]::GetFullPath(
    (Join-Path $RunsRoot "phase-effective-entry-holdout")
).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$HoldoutManifestPath = Join-Path (Split-Path -Parent $HoldoutPath) "run_manifest.json"
if (-not $HoldoutPath.StartsWith(
        $HoldoutRoot,
        [StringComparison]::OrdinalIgnoreCase
    ) -or
    [IO.Path]::GetFileName($HoldoutPath) -cne "phase_effective_entry_holdout_acceptance.json" -or
    -not (Test-Path -LiteralPath $HoldoutPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $HoldoutManifestPath -PathType Leaf)) {
    throw "Phase effective-entry holdout is not a finalized managed acceptance: $HoldoutPath"
}
try {
    $Holdout = Get-Content -LiteralPath $HoldoutPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
    $HoldoutManifest = Get-Content -LiteralPath $HoldoutManifestPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "Phase effective-entry holdout evidence is not valid JSON"
}
$HoldoutPhases = @($Holdout.phases)
if ([string]$Holdout.schema -cne "wlr50_clean.phase_effective_entry_holdout_acceptance.v1" -or
    [string]$Holdout.status -cne "PASSED" -or
    $Holdout.passed -ne $true -or
    [int]$Holdout.seed -ne 1003 -or
    [int]$Holdout.worker_count -ne 12 -or
    ($HoldoutPhases -join ',') -cne ((2..13 | ForEach-Object { "P{0:D2}" -f $_ }) -join ',') -or
    [string]$HoldoutManifest.schema -cne "wlr50_clean.ppo_run_manifest.v1" -or
    [string]$HoldoutManifest.lifecycle -cne "SUCCEEDED" -or
    [string]$HoldoutManifest.run_kind -cne "phase-effective-entry-holdout" -or
    [int]$HoldoutManifest.exit_code -ne 0) {
    throw "Phase effective-entry holdout evidence is incomplete or failed"
}

$PhaseRolloutPath = [IO.Path]::GetFullPath(
    $(if ([IO.Path]::IsPathRooted($PhaseZeroResidualRolloutEvidence)) {
        $PhaseZeroResidualRolloutEvidence
    } else {
        Join-Path $ProjectRoot $PhaseZeroResidualRolloutEvidence
    })
)
$PhaseRolloutRoot = [IO.Path]::GetFullPath(
    (Join-Path $RunsRoot "phase-zero-residual-rollout")
).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$PhaseRolloutManifestPath = Join-Path (
    Split-Path -Parent $PhaseRolloutPath
) "run_manifest.json"
if (-not $PhaseRolloutPath.StartsWith(
        $PhaseRolloutRoot,
        [StringComparison]::OrdinalIgnoreCase
    ) -or
    [IO.Path]::GetFileName($PhaseRolloutPath) -cne "phase_zero_residual_rollout.json" -or
    -not (Test-Path -LiteralPath $PhaseRolloutPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $PhaseRolloutManifestPath -PathType Leaf)) {
    throw "Phase zero-residual rollout is not finalized managed evidence: $PhaseRolloutPath"
}
try {
    $PhaseRollout = Get-Content -LiteralPath $PhaseRolloutPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
    $PhaseRolloutManifest = Get-Content -LiteralPath $PhaseRolloutManifestPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "Phase zero-residual rollout evidence is not valid JSON"
}
$PhaseRolloutPhases = @($PhaseRollout.phases)
$PhaseRolloutChecks = @($PhaseRollout.checks.PSObject.Properties.Value)
if ([string]$PhaseRollout.schema -cne "wlr50_clean.phase_zero_residual_rollout.v1" -or
    [string]$PhaseRollout.artifact_role -cne "PHASE_CURRICULUM_TRAINING_PREREQUISITE" -or
    [string]$PhaseRollout.status -cne "PASSED" -or
    $PhaseRollout.passed -ne $true -or
    [int]$PhaseRollout.phase_reset_count -ne 13 -or
    ($PhaseRolloutPhases -join ',') -cne ((1..13 | ForEach-Object { "P{0:D2}" -f $_ }) -join ',') -or
    [int]$PhaseRollout.max_decisions_per_phase -ne 64 -or
    [double]$PhaseRollout.physics_hz -ne 120.0 -or
    [double]$PhaseRollout.decision_hz -ne 15.0 -or
    [int]$PhaseRollout.physics_ticks_per_decision -ne 8 -or
    @($PhaseRollout.phase_rollouts).Count -ne 13 -or
    @($PhaseRollout.failure_reasons).Count -ne 0 -or
    $PhaseRolloutChecks.Count -eq 0 -or
    @($PhaseRolloutChecks | Where-Object { $_ -ne $true }).Count -ne 0 -or
    [string]$PhaseRolloutManifest.schema -cne "wlr50_clean.ppo_run_manifest.v1" -or
    [string]$PhaseRolloutManifest.lifecycle -cne "SUCCEEDED" -or
    [string]$PhaseRolloutManifest.run_kind -cne "phase-zero-residual-rollout" -or
    [int]$PhaseRolloutManifest.exit_code -ne 0) {
    throw "Phase zero-residual rollout evidence is incomplete or failed"
}

$TrainingRuns = [Collections.Generic.List[string]]::new()
$ScreeningRuns = [Collections.Generic.List[string]]::new()
$Checkpoints = [Collections.Generic.List[string]]::new()
$CheckpointHashes = [Collections.Generic.List[string]]::new()
$CollectedPromotionDecisions = [Collections.Generic.List[string]]::new()
foreach ($PathValue in $PromotionDecision) {
    $CollectedPromotionDecisions.Add([IO.Path]::GetFullPath(
        $(if ([IO.Path]::IsPathRooted($PathValue)) {
            $PathValue
        } else {
            Join-Path $ProjectRoot $PathValue
        })
    ))
}
$SeenRunDirectories = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
$PreviousCheckpoint = $null
$PreviousManifest = $null
$PreviousGlobalStep = 0L
$ChunkIndex = 0

function Confirm-SingleManagedRunDirectory {
    param(
        [object[]]$Output,
        [string]$Label,
        [string]$ExpectedRunKind = ""
    )

    $Rows = @($Output | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    if ($Rows.Count -ne 1) {
        throw "$Label did not return exactly one managed run directory"
    }
    $Directory = [IO.Path]::GetFullPath([string]$Rows[0])
    $Prefix = $RunsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $Directory.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $Directory -PathType Container)) {
        throw "$Label returned an invalid run directory: $Directory"
    }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedRunKind)) {
        $KindPrefix = [IO.Path]::GetFullPath(
            (Join-Path $RunsRoot $ExpectedRunKind)
        ).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (-not $Directory.StartsWith(
                $KindPrefix,
                [StringComparison]::OrdinalIgnoreCase
            )) {
            throw "$Label returned the wrong managed run kind: $Directory"
        }
        $RelativeRunId = $Directory.Substring($KindPrefix.Length)
        if (
            [string]::IsNullOrWhiteSpace($RelativeRunId) -or
            $RelativeRunId.Contains([IO.Path]::DirectorySeparatorChar)) {
            throw "$Label returned the wrong managed run kind: $Directory"
        }
    }
    if (-not $SeenRunDirectories.Add($Directory)) {
        throw "$Label reused a managed run directory: $Directory"
    }
    return $Directory
}

function Read-TrainingChunk {
    param(
        [string]$RunDirectory,
        [string]$ExpectedStage,
        [int]$ExpectedNumEnvs,
        [string]$ExpectedResumeCheckpoint,
        [long]$ExpectedResumeGlobalStep
    )

    $ResultPath = Join-Path $RunDirectory "training_result.json"
    if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
        throw "Training chunk omitted training_result.json: $RunDirectory"
    }
    try {
        $Result = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Training chunk result is not valid JSON: $RunDirectory"
    }
    $PpoBatchDecisions = [long]$ExpectedNumEnvs * [long]$Result.rollout_length
    $ExpectedIterations = [long][Math]::Ceiling(
        [double]$RequestedChunkDecisions / [double]$PpoBatchDecisions
    )
    if ([string]$Result.schema -ne "wlr50_clean.ppo_training_run.v1" -or
        [string]$Result.stage -cne $ExpectedStage -or
        [int]$Result.requested_policy_decisions -ne $RequestedChunkDecisions -or
        [int]$Result.num_envs -ne $ExpectedNumEnvs -or
        [long]$Result.iterations -ne $ExpectedIterations -or
        [long]$Result.ppo_batch_policy_decisions -ne $PpoBatchDecisions -or
        [long]$Result.rounding_overrun_policy_decisions -ne (
            [long]$Result.stage_policy_decisions - $RequestedChunkDecisions
        ) -or
        [string]$Result.budget_accounting_basis -cne "requested_policy_decisions" -or
        [long]$Result.global_policy_decisions -ne (
            $ExpectedResumeGlobalStep + [long]$Result.stage_policy_decisions
        ) -or
        [long]$Result.stage_policy_decisions -ne (
            [long]$Result.num_envs * [long]$Result.rollout_length * [long]$Result.iterations
        ) -or
        ([long]$Result.stage_policy_decisions - $RequestedChunkDecisions) -lt 0 -or
        ([long]$Result.stage_policy_decisions - $RequestedChunkDecisions) -ge $PpoBatchDecisions) {
        throw "Training chunk accounting differs from the requested-10k, whole-PPO-batch cadence contract: $RunDirectory"
    }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedResumeCheckpoint)) {
        $ActualResume = [IO.Path]::GetFullPath([string]$Result.resume_checkpoint)
        $ExpectedResume = [IO.Path]::GetFullPath($ExpectedResumeCheckpoint)
        if (-not $ActualResume.Equals($ExpectedResume, [StringComparison]::OrdinalIgnoreCase) -or
            [long]$ExpectedResumeGlobalStep -le 0) {
            throw "Training chunk resumed from the wrong checkpoint/global step"
        }
    } else {
        $ActualResume = [IO.Path]::GetFullPath([string]$Result.resume_checkpoint)
        $ExpectedResume = [IO.Path]::GetFullPath(
            (Join-Path $ProjectRoot "outputs\ppo_phase_v1\checkpoints\checkpoint_initial_zero_residual.pt")
        )
        if (-not $ActualResume.Equals($ExpectedResume, [StringComparison]::OrdinalIgnoreCase) -or
            $ExpectedResumeGlobalStep -ne 0) {
            throw "First training chunk did not use the canonical initial checkpoint"
        }
        $InitialManifest = Join-Path (
            Split-Path -Parent $ExpectedResume
        ) "checkpoint_initial_zero_residual_manifest.json"
        if (-not (Test-Path -LiteralPath $InitialManifest -PathType Leaf)) {
            throw "Canonical initial checkpoint manifest is missing: $InitialManifest"
        }
        try {
            $InitialPayload = Get-Content -LiteralPath $InitialManifest -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Canonical initial checkpoint manifest is not valid JSON"
        }
        $InitialSha = (Get-FileHash -LiteralPath $ExpectedResume -Algorithm SHA256).Hash.ToLowerInvariant()
        $CreationIdentity = [IO.Path]::GetFullPath(
            [string]$InitialPayload.creation_runtime_identity_path
        )
        if ([string]$InitialPayload.schema -ne "wlr50_clean.phase_residual_checkpoint_manifest.v1" -or
            [string]$InitialPayload.stage -cne "initial_zero_residual" -or
            [long]$InitialPayload.global_policy_decisions -ne 0 -or
            $InitialPayload.zero_mean_actor_output_layer_verified -ne $true -or
            [IO.Path]::GetFullPath([string]$InitialPayload.checkpoint_path) -cne $ExpectedResume -or
            [string]$InitialPayload.checkpoint_sha256 -cne $InitialSha -or
            [string]$Result.resume_checkpoint_sha256 -cne $InitialSha -or
            -not (Test-Path -LiteralPath $CreationIdentity -PathType Leaf) -or
            [string]$InitialPayload.creation_runtime_identity_sha256 -cne (
                (Get-FileHash -LiteralPath $CreationIdentity -Algorithm SHA256).Hash.ToLowerInvariant()
            )) {
            throw "Canonical initial checkpoint provenance is inconsistent"
        }
        try {
            $CreationPayload = Get-Content -LiteralPath $CreationIdentity -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Canonical initial creation runtime identity is not valid JSON"
        }
        if ([string]$CreationPayload.schema -ne "wlr50_clean.committed_runtime_identity.v1" -or
            [string]$InitialPayload.source_git_commit -cne [string]$CreationPayload.git_commit -or
            [string]$InitialPayload.committed_runtime_content_sha256 -cne [string]$CreationPayload.content_sha256) {
            throw "Canonical initial checkpoint runtime ABI is inconsistent"
        }
    }
    $Checkpoint = [IO.Path]::GetFullPath([string]$Result.immutable_history_checkpoint)
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Training chunk immutable checkpoint is missing: $Checkpoint"
    }
    $CheckpointSha = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($CheckpointSha -cne [string]$Result.checkpoint_sha256) {
        throw "Training chunk immutable checkpoint SHA-256 mismatch"
    }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedResumeCheckpoint)) {
        $ResumeSha = (Get-FileHash -LiteralPath $ActualResume -Algorithm SHA256).Hash.ToLowerInvariant()
        if ([string]$Result.resume_checkpoint_sha256 -cne $ResumeSha) {
            throw "Training chunk resume checkpoint SHA-256 mismatch"
        }
    }
    $Manifest = Join-Path (
        Split-Path -Parent $Checkpoint
    ) (([IO.Path]::GetFileNameWithoutExtension($Checkpoint)) + "_manifest.json")
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "Training chunk immutable checkpoint manifest is missing: $Manifest"
    }
    try {
        $ManifestPayload = Get-Content -LiteralPath $Manifest -Raw |
            ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Training checkpoint manifest is not valid JSON: $Manifest"
    }
    if ([IO.Path]::GetFullPath([string]$ManifestPayload.checkpoint_path) -cne $Checkpoint -or
        [string]$ManifestPayload.checkpoint_sha256 -cne $CheckpointSha -or
        [long]$ManifestPayload.global_policy_decisions -ne [long]$Result.global_policy_decisions -or
        [long]$ManifestPayload.resume_global_policy_decisions -ne $ExpectedResumeGlobalStep) {
        throw "Training checkpoint manifest is not bound to its chunk"
    }
    $RuntimeBefore = [IO.Path]::GetFullPath(
        (Join-Path $RunDirectory "committed_runtime_identity.before.json")
    )
    if (-not (Test-Path -LiteralPath $RuntimeBefore -PathType Leaf)) {
        throw "Training chunk committed-runtime identity is missing"
    }
    try {
        $RuntimePayload = Get-Content -LiteralPath $RuntimeBefore -Raw |
            ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Training chunk committed-runtime identity is not valid JSON"
    }
    $RuntimeSha = (Get-FileHash -LiteralPath $RuntimeBefore -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([IO.Path]::GetFullPath([string]$ManifestPayload.creation_runtime_identity_path) -cne $RuntimeBefore -or
        [string]$ManifestPayload.creation_runtime_identity_sha256 -cne $RuntimeSha -or
        [string]$ManifestPayload.source_git_commit -cne [string]$RuntimePayload.git_commit -or
        [string]$ManifestPayload.committed_runtime_content_sha256 -cne [string]$RuntimePayload.content_sha256) {
        throw "Training checkpoint is not bound to its creation runtime identity"
    }
    if (-not [string]::IsNullOrWhiteSpace(
        [string]$Result.immutable_history_checkpoint_manifest
    )) {
        if ([IO.Path]::GetFullPath(
                [string]$Result.immutable_history_checkpoint_manifest
            ) -cne [IO.Path]::GetFullPath($Manifest) -or
            [string]$Result.immutable_history_checkpoint_manifest_sha256 -cne (
                (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
            )) {
            throw "Training result immutable checkpoint manifest binding is stale"
        }
    }
    if ($ExpectedStage -ceq "smoke" -and $ExpectedResumeGlobalStep -eq 0) {
        $CanonicalSmoke = [IO.Path]::GetFullPath(
            (Join-Path $ProjectRoot "outputs\ppo_phase_v1\checkpoints\checkpoint_smoke.pt")
        )
        $CanonicalSmokeManifest = Join-Path (
            Split-Path -Parent $CanonicalSmoke
        ) "checkpoint_smoke_manifest.json"
        if (-not (Test-Path -LiteralPath $CanonicalSmoke -PathType Leaf) -or
            -not (Test-Path -LiteralPath $CanonicalSmokeManifest -PathType Leaf) -or
            (Get-FileHash -LiteralPath $CanonicalSmoke -Algorithm SHA256).Hash.ToLowerInvariant() -cne $CheckpointSha) {
            throw "Canonical smoke checkpoint differs from first smoke history"
        }
        try {
            $CanonicalSmokePayload = Get-Content -LiteralPath $CanonicalSmokeManifest -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Canonical smoke checkpoint manifest is not valid JSON"
        }
        $HistoryCore = [ordered]@{}
        foreach ($Property in $ManifestPayload.PSObject.Properties) {
            if ($Property.Name -notin @("checkpoint_path", "checkpoint_sha256")) {
                $HistoryCore[$Property.Name] = $Property.Value
            }
        }
        $CanonicalCore = [ordered]@{}
        foreach ($Property in $CanonicalSmokePayload.PSObject.Properties) {
            if ($Property.Name -notin @("checkpoint_path", "checkpoint_sha256")) {
                $CanonicalCore[$Property.Name] = $Property.Value
            }
        }
        if ([IO.Path]::GetFullPath([string]$CanonicalSmokePayload.checkpoint_path) -cne $CanonicalSmoke -or
            [string]$CanonicalSmokePayload.checkpoint_sha256 -cne $CheckpointSha -or
            (ConvertTo-Json $HistoryCore -Compress -Depth 100) -cne (
                ConvertTo-Json $CanonicalCore -Compress -Depth 100
            )) {
            throw "Canonical smoke checkpoint manifest differs from first smoke history"
        }
    }
    [pscustomobject]@{
        Checkpoint = $Checkpoint
        CheckpointManifest = [IO.Path]::GetFullPath($Manifest)
        CheckpointSha256 = $CheckpointSha
        GlobalStep = [long]$Result.global_policy_decisions
    }
}

function Invoke-TrainingChunk {
    param([string]$Stage, [int]$NumEnvs)

    $Arguments = @{
        Seed = $Seed
        Stage = $Stage
        NumEnvs = $NumEnvs
        # The request is exactly 10k.  The trainer truthfully records the
        # smallest whole PPO batch at or above it as stage_policy_decisions.
        CliArgs = @("--policy-decisions", [string]$RequestedChunkDecisions)
    }
    if ($null -ne $PreviousCheckpoint) {
        $Arguments.Checkpoint = $PreviousCheckpoint
        $Arguments.CheckpointManifest = $PreviousManifest
    }
    if ($NumEnvs -eq 1) {
        $Arguments.SoftResetAcceptance = $SoftResetAcceptance
    } else {
        $Arguments.VectorBenchmarkMatrix = $MatrixPath
    }
    if ($Stage -ceq "phase-curriculum") {
        $Arguments.PhaseEffectiveEntryHoldoutAcceptance = $HoldoutPath
        $Arguments.PhaseZeroResidualRolloutEvidence = $PhaseRolloutPath
    }
    $Output = @(& $TrainScript @Arguments)
    $RunDirectory = Confirm-SingleManagedRunDirectory $Output "training chunk"
    $Chunk = Read-TrainingChunk `
        -RunDirectory $RunDirectory `
        -ExpectedStage $Stage `
        -ExpectedNumEnvs $NumEnvs `
        -ExpectedResumeCheckpoint $PreviousCheckpoint `
        -ExpectedResumeGlobalStep $PreviousGlobalStep
    $TrainingRuns.Add($RunDirectory)
    $Checkpoints.Add($Chunk.Checkpoint)
    $CheckpointHashes.Add($Chunk.CheckpointSha256)
    $script:PreviousCheckpoint = $Chunk.Checkpoint
    $script:PreviousManifest = $Chunk.CheckpointManifest
    $script:PreviousGlobalStep = $Chunk.GlobalStep
    return $Chunk
}

function Invoke-FreshScreening {
    param([object]$Chunk)

    $ValidationSeed = $ValidationSeeds[$ChunkIndex % $ValidationSeeds.Count]
    $Output = @(
        & $ScreenScript `
            -Checkpoint $Chunk.Checkpoint `
            -CheckpointManifest $Chunk.CheckpointManifest `
            -Seed $ValidationSeed
    )
    $RunDirectory = Confirm-SingleManagedRunDirectory $Output "fresh screening"
    $ResultPath = Join-Path $RunDirectory "checkpoint_evaluation.json"
    if (-not (Test-Path -LiteralPath $ResultPath -PathType Leaf)) {
        throw "Fresh screening omitted checkpoint_evaluation.json"
    }
    try {
        $Result = Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Fresh screening result is not valid JSON: $RunDirectory"
    }
    if ([string]$Result.schema -ne "wlr50_clean.ppo_checkpoint_evaluation.v1" -or
        $Result.deterministic_mean_policy -ne $true -or
        $Result.fresh_process_single_episode -ne $true -or
        [int]$Result.episode_count -ne 1 -or
        [IO.Path]::GetFullPath([string]$Result.checkpoint) -cne $Chunk.Checkpoint -or
        [string]$Result.checkpoint_sha256 -cne $Chunk.CheckpointSha256 -or
        [long]$Result.checkpoint_infos.global_policy_decisions -ne $Chunk.GlobalStep) {
        throw "Fresh screening is not bound to the immediately preceding chunk"
    }
    $ScreeningRuns.Add($RunDirectory)
}

function Invoke-FiveSeedPairedPromotion {
    param([object]$Chunk)

    $EvaluationOutput = @(
        & $FiveSeedScript `
            -Checkpoint $Chunk.Checkpoint `
            -CheckpointManifest $Chunk.CheckpointManifest `
            -SeedSet "validation" `
            -Seed 2001 `
            -NumEnvs 1 `
            -EpisodeCount 5
    )
    $AggregateRun = Confirm-SingleManagedRunDirectory `
        $EvaluationOutput "five-seed checkpoint evaluation"
    $AggregatePath = Join-Path $AggregateRun "checkpoint_evaluation_aggregate.json"
    if (-not (Test-Path -LiteralPath $AggregatePath -PathType Leaf)) {
        throw "Five-seed evaluation omitted its aggregate artifact"
    }
    try {
        $Aggregate = Get-Content -LiteralPath $AggregatePath -Raw |
            ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Five-seed evaluation aggregate is not valid JSON: $AggregatePath"
    }
    $Seeds = @($Aggregate.seeds | ForEach-Object { [int]$_ })
    $CandidateEpisodes = @($Aggregate.canonical_episode_dirs)
    $Workers = @($Aggregate.workers)
    if ([string]$Aggregate.schema -ne "wlr50_clean.fresh_process_episode_batch.v1" -or
        [string]$Aggregate.role -ne "candidate" -or
        [string]$Aggregate.seed_set -ne "validation" -or
        ($Seeds -join ',') -cne ($ValidationSeeds -join ',') -or
        [int]$Aggregate.episode_count -ne 5 -or
        $Aggregate.fresh_process_per_episode -ne $true -or
        $Aggregate.deterministic_mean_policy -ne $true -or
        [IO.Path]::GetFullPath([string]$Aggregate.checkpoint) -cne $Chunk.Checkpoint -or
        [string]$Aggregate.checkpoint_sha256 -cne $Chunk.CheckpointSha256 -or
        $CandidateEpisodes.Count -ne 5 -or
        $Workers.Count -ne 5 -or
        ($CandidateEpisodes | Select-Object -Unique).Count -ne 5) {
        throw "Five-seed aggregate is not bound to the current training chunk"
    }
    for ($Index = 0; $Index -lt 5; $Index++) {
        $EpisodeDirectory = [IO.Path]::GetFullPath([string]$CandidateEpisodes[$Index])
        $WorkerResult = [IO.Path]::GetFullPath([string]$Workers[$Index].worker_result)
        if (-not (Test-Path -LiteralPath $EpisodeDirectory -PathType Container) -or
            -not (Test-Path -LiteralPath $WorkerResult -PathType Leaf) -or
            [int]$Workers[$Index].seed -ne $ValidationSeeds[$Index] -or
            [IO.Path]::GetFullPath([string]$Workers[$Index].canonical_episode_dir) -cne $EpisodeDirectory) {
            throw "Five-seed aggregate worker/episode ordering is invalid"
        }
        try {
            $Worker = Get-Content -LiteralPath $WorkerResult -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Five-seed worker result is not valid JSON: $WorkerResult"
        }
        if ([string]$Worker.schema -ne "wlr50_clean.ppo_checkpoint_evaluation.v1" -or
            [IO.Path]::GetFullPath([string]$Worker.checkpoint) -cne $Chunk.Checkpoint -or
            [string]$Worker.checkpoint_sha256 -cne $Chunk.CheckpointSha256 -or
            [long]$Worker.checkpoint_infos.global_policy_decisions -ne $Chunk.GlobalStep) {
            throw "Five-seed worker is not bound to the current checkpoint/global step"
        }
        $CandidateEpisodes[$Index] = $EpisodeDirectory
    }

    # Keep immutable promotion inputs inside final delivery, but outside its
    # metrics directory so final lifecycle export cannot overlap its sources.
    $MetricsDirectory = Join-Path $ProjectRoot (
        "outputs\ppo_phase_v1\validation_history\step_{0}" -f $Chunk.GlobalStep
    )
    if (Test-Path -LiteralPath $MetricsDirectory) {
        throw "Refusing to overwrite cadence metrics directory: $MetricsDirectory"
    }
    $PairOutput = @(
        & $PairedExportScript `
            -BaselineAggregate $BaselineAggregatePath `
            -CandidateValidationAggregate $AggregatePath `
            -BaselineEpisodeDir $ResolvedBaselineEpisodes `
            -CandidateEpisodeDir $CandidateEpisodes `
            -CandidateCheckpoint $Chunk.Checkpoint `
            -CandidateManifest $Chunk.CheckpointManifest `
            -MetricsOutputDir $MetricsDirectory `
            -SeedSet "validation" `
            -Seed 2001 `
            -EpisodeCount 5
    )
    $null = Confirm-SingleManagedRunDirectory $PairOutput "paired promotion export"
    $DecisionPath = [IO.Path]::GetFullPath(
        (Join-Path $MetricsDirectory "promotion_decision.json")
    )
    if (-not (Test-Path -LiteralPath $DecisionPath -PathType Leaf)) {
        throw "Paired export omitted promotion_decision.json for step $($Chunk.GlobalStep)"
    }
    if ($CollectedPromotionDecisions.Contains($DecisionPath)) {
        throw "Cadence promotion decision path was reused: $DecisionPath"
    }
    $CollectedPromotionDecisions.Add($DecisionPath)
}

function Test-StrictPromotionForCurrentChunk {
    param([string]$Stage, [object]$Chunk)

    $CurrentIndex = $Checkpoints.Count - 1
    foreach ($DecisionPath in $CollectedPromotionDecisions) {
        if (-not (Test-Path -LiteralPath $DecisionPath -PathType Leaf)) {
            continue
        }
        try {
            $Decision = Get-Content -LiteralPath $DecisionPath -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Promotion decision is not valid JSON: $DecisionPath"
        }
        if ($Decision.promotion.promoted -ne $true) {
            continue
        }
        $Candidate = [IO.Path]::GetFullPath([string]$Decision.candidate_checkpoint_path)
        $CandidateIndex = -1
        for ($Index = 0; $Index -lt $Checkpoints.Count; $Index++) {
            if ($Checkpoints[$Index].Equals($Candidate, [StringComparison]::OrdinalIgnoreCase) -and
                $CheckpointHashes[$Index] -ceq [string]$Decision.candidate_checkpoint_sha256) {
                $CandidateIndex = $Index
                break
            }
        }
        if ($CandidateIndex -lt 0) {
            continue
        }
        if ($CandidateIndex -lt $CurrentIndex) {
            throw "A passing five-seed promotion was discovered after later training had run"
        }
        $Seeds = @($Decision.paired_seeds | ForEach-Object { [int]$_ })
        $GateNames = @($Decision.promotion.checks.PSObject.Properties.Name)
        if ([string]$Decision.schema -ne "wlr50_clean.ppo_evaluation_artifacts.v1" -or
            ($Seeds -join ',') -cne ($ValidationSeeds -join ',') -or
            [int]$Decision.paired_episode_count -ne 5 -or
            [int]$Decision.minimum_paired_seeds -ne 5 -or
            $Decision.frozen_hashes_unchanged -ne $true -or
            $null -ne $Decision.promotion.first_failed_gate -or
            (Compare-Object $PromotionGates $GateNames).Count -ne 0) {
            throw "Promotion decision cannot authorize cadence early stop"
        }
        foreach ($Gate in $PromotionGates) {
            if ($Decision.promotion.checks.$Gate -ne $true) {
                throw "Promotion decision contains a failed gate: $Gate"
            }
        }
        if ($Stage -cne "full-episode" -or $Candidate -cne $Chunk.Checkpoint) {
            throw "Promotion cannot stop cadence before the full-episode stage"
        }
        return $true
    }
    return $false
}

$InitialPublicationOutput = @(& $InitializeScript -Seed $Seed)
$InitialPublicationRun = Confirm-SingleManagedRunDirectory `
    $InitialPublicationOutput `
    "initial checkpoint publication" `
    "initial-checkpoint-publication"
$InitialPublicationResultPath = Join-Path `
    $InitialPublicationRun "initial_checkpoint_publication.json"
if (-not (Test-Path -LiteralPath $InitialPublicationResultPath -PathType Leaf)) {
    throw "Initial checkpoint publication omitted its result artifact"
}
try {
    $InitialPublication = Get-Content -LiteralPath $InitialPublicationResultPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "Initial checkpoint publication result is not valid JSON"
}
$CanonicalInitial = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot "outputs\ppo_phase_v1\checkpoints\checkpoint_initial_zero_residual.pt")
)
$CanonicalInitialManifest = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot "outputs\ppo_phase_v1\checkpoints\checkpoint_initial_zero_residual_manifest.json")
)
if ([string]$InitialPublication.schema -cne "wlr50_clean.initial_zero_residual_checkpoint_publication.v1" -or
    $InitialPublication.no_existing_artifact_overwritten -ne $true -or
    $InitialPublication.source_initializer_finalized_success -ne $true -or
    $InitialPublication.zero_mean_actor_output_layer_verified -ne $true -or
    [IO.Path]::GetFullPath([string]$InitialPublication.checkpoint) -cne $CanonicalInitial -or
    [IO.Path]::GetFullPath([string]$InitialPublication.checkpoint_manifest) -cne $CanonicalInitialManifest -or
    -not (Test-Path -LiteralPath $CanonicalInitial -PathType Leaf) -or
    -not (Test-Path -LiteralPath $CanonicalInitialManifest -PathType Leaf) -or
    [string]$InitialPublication.checkpoint_sha256 -cne (
        (Get-FileHash -LiteralPath $CanonicalInitial -Algorithm SHA256).Hash.ToLowerInvariant()
    ) -or
    [string]$InitialPublication.checkpoint_manifest_sha256 -cne (
        (Get-FileHash -LiteralPath $CanonicalInitialManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    )) {
    throw "Initial checkpoint publication did not produce the strict canonical pair"
}

$Stages = @(
    [pscustomobject]@{ Name = "smoke"; Chunks = 1; NumEnvs = $SelectedNumEnvs },
    [pscustomobject]@{ Name = "phase-curriculum"; Chunks = 10; NumEnvs = 1 },
    [pscustomobject]@{ Name = "full-episode"; Chunks = 10; NumEnvs = $SelectedNumEnvs }
)

$StopAfterPromotion = $false
foreach ($StagePlan in $Stages) {
    for ($StageChunk = 0; $StageChunk -lt $StagePlan.Chunks; $StageChunk++) {
        $Chunk = Invoke-TrainingChunk -Stage $StagePlan.Name -NumEnvs $StagePlan.NumEnvs
        Invoke-FreshScreening -Chunk $Chunk
        if ($StagePlan.Name -ceq "full-episode") {
            Invoke-FiveSeedPairedPromotion -Chunk $Chunk
        }
        $StopAfterPromotion = Test-StrictPromotionForCurrentChunk `
            -Stage $StagePlan.Name -Chunk $Chunk
        $ChunkIndex++
        if ($StopAfterPromotion) {
            break
        }
    }
    if ($StopAfterPromotion) {
        break
    }
}

if ($TrainingRuns.Count -ne $ScreeningRuns.Count) {
    throw "Cadence did not produce exactly one fresh screening for each training chunk"
}
foreach ($DecisionPath in $CollectedPromotionDecisions) {
    if (-not (Test-Path -LiteralPath $DecisionPath -PathType Leaf)) {
        throw "Explicit promotion decision was not produced: $DecisionPath"
    }
}

$BuildOutput = @(
    & $BuildScript `
        -TrainingRunDir $TrainingRuns.ToArray() `
        -ScreeningRunDir $ScreeningRuns.ToArray() `
        -InitialCheckpointPublicationRun $InitialPublicationRun `
        -VectorBenchmarkMatrix $MatrixPath `
        -PromotionDecision $CollectedPromotionDecisions.ToArray() `
        -Seed $Seed
)
$OrchestrationRun = Confirm-SingleManagedRunDirectory $BuildOutput "orchestration build"
Write-Output $OrchestrationRun
