[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunKind,

    [Parameter(Mandatory = $true)]
    [string]$TrainingStage,

    [Parameter(Mandatory = $true)]
    [string]$Subcommand,

    [Parameter(Mandatory = $true)]
    [string[]]$ConfigPath,

    [Parameter(Mandatory = $true)]
    [int]$Seed,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4096)]
    [int]$EnvironmentCount,

    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$')]
    [string]$CliModule = "wlr50_clean.ppo.cli",

    [string[]]$BaseCliArgs = @(),

    [switch]$ReturnFinalizedEvidenceFailure,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$SourceRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "src"))
$RunsRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "runs\ppo_phase_v1"))
$IsaacPython = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
$ArtifactModule = "wlr50_clean.ppo.artifacts"

if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Locked Isaac Python is missing: $IsaacPython"
}
if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Project source directory is missing: $SourceRoot"
}

# Every live/offline evidence artifact identifies implementation bytes by the
# current Git commit.  Refuse to run while any runtime source, launcher,
# versioned config, phase snapshot, or frozen-start record differs from that
# commit; otherwise two different implementations could share one run ID and
# an older benchmark matrix could authorize newer uncommitted code.
$RuntimeIdentityPaths = @(
    "src/wlr50_clean",
    "src/wlr50_clean/ppo",
    "src/wlr50_clean/fsm",
    "src/wlr50_clean/sensing",
    "src/wlr50_clean/infrastructure",
    "scripts",
    "configs",
    "reference/ppo_phase_snapshots",
    "artifacts/ppo_phase_v1_start",
    "pyproject.toml"
)

function Get-CommittedRuntimeIdentity {
    $TrackedPaths = @(& git ls-files -- @RuntimeIdentityPaths)
    if ($LASTEXITCODE -ne 0 -or $TrackedPaths.Count -eq 0) {
        throw "Failed to enumerate committed PPO runtime files"
    }
    $GitCommit = (& git rev-parse HEAD | Out-String).Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or $GitCommit -notmatch '^[0-9a-f]{40}$') {
        throw "Failed to resolve committed PPO runtime HEAD"
    }
    # PowerShell's Sort-Object follows the current culture, whereas every
    # Python evidence validator uses ordinal Unicode ordering.  Keep the
    # cross-language JSON/hash contract deterministic on every Windows locale.
    $UniqueTrackedPaths = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::Ordinal
    )
    foreach ($TrackedPath in $TrackedPaths) {
        [void]$UniqueTrackedPaths.Add([string]$TrackedPath)
    }
    $SortedTrackedPaths = [string[]]@($UniqueTrackedPaths)
    [Array]::Sort($SortedTrackedPaths, [StringComparer]::Ordinal)
    $Rows = @(
        foreach ($RelativePath in $SortedTrackedPaths) {
            $Normalized = ([string]$RelativePath).Replace('\', '/')
            $AbsolutePath = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Normalized))
            if (-not (Test-Path -LiteralPath $AbsolutePath -PathType Leaf)) {
                throw "Committed PPO runtime file is missing: $Normalized"
            }
            $Info = Get-Item -LiteralPath $AbsolutePath
            if (($Info.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Committed PPO runtime file is redirected: $Normalized"
            }
            [ordered]@{
                path = $Normalized
                bytes = [int64]$Info.Length
                sha256 = (Get-FileHash -LiteralPath $AbsolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
                creation_time_utc_ticks = [int64]$Info.CreationTimeUtc.Ticks
                last_write_time_utc_ticks = [int64]$Info.LastWriteTimeUtc.Ticks
            }
        }
    )
    $ContentRows = @(
        foreach ($Row in $Rows) {
            [ordered]@{
                path = $Row.path
                bytes = $Row.bytes
                sha256 = $Row.sha256
            }
        }
    )
    $ContentRowsJson = ConvertTo-Json -InputObject $ContentRows -Compress -Depth 4
    $ContentSha = [Security.Cryptography.SHA256]::Create()
    try {
        $ContentDigestBytes = $ContentSha.ComputeHash(
            [Text.Encoding]::UTF8.GetBytes($ContentRowsJson)
        )
    } finally {
        $ContentSha.Dispose()
    }
    $RowsJson = ConvertTo-Json -InputObject $Rows -Compress -Depth 6
    $Sha = [Security.Cryptography.SHA256]::Create()
    try {
        $DigestBytes = $Sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($RowsJson))
    } finally {
        $Sha.Dispose()
    }
    [ordered]@{
        schema = "wlr50_clean.committed_runtime_identity.v1"
        git_commit = $GitCommit
        file_count = $Rows.Count
        content_sha256 = ([BitConverter]::ToString($ContentDigestBytes)).Replace('-', '').ToLowerInvariant()
        aggregate_sha256 = ([BitConverter]::ToString($DigestBytes)).Replace('-', '').ToLowerInvariant()
        files = $Rows
    }
}

function Test-ReturnableEvidenceFailure {
    param(
        [bool]$Enabled,
        [string]$RunKindValue,
        [string]$TrainingStageValue,
        [string]$SubcommandValue,
        [string]$CliModuleValue,
        [Nullable[int]]$AuthoritativeLiveExitCode,
        [int]$FinalExitCode,
        [bool]$RuntimeIdentityPostCheckPassed,
        [bool]$FrozenPostCheckPassed,
        [bool]$FinalManifestValidated
    )

    $AllowedEvidenceWorker = (
        ($RunKindValue -ceq "vector_benchmark" -and
            $TrainingStageValue -ceq "backend-benchmark" -and
            $SubcommandValue -ceq "vector-benchmark") -or
        ($RunKindValue -ceq "phase_snapshot_live_probe" -and
            $TrainingStageValue -ceq "phase-snapshot-live-probe" -and
            $SubcommandValue -ceq "phase-snapshot-live-probe")
    )
    return (
        $Enabled -and $AllowedEvidenceWorker -and
        $CliModuleValue -ceq "wlr50_clean.ppo.cli" -and
        $null -ne $AuthoritativeLiveExitCode -and
        $AuthoritativeLiveExitCode -eq 2 -and
        $FinalExitCode -eq 2 -and
        $RuntimeIdentityPostCheckPassed -and
        $FrozenPostCheckPassed -and
        $FinalManifestValidated
    )
}

if ($ReturnFinalizedEvidenceFailure) {
    $AllowedEvidenceWorker = (
        ($RunKind -ceq "vector_benchmark" -and
            $TrainingStage -ceq "backend-benchmark" -and
            $Subcommand -ceq "vector-benchmark") -or
        ($RunKind -ceq "phase_snapshot_live_probe" -and
            $TrainingStage -ceq "phase-snapshot-live-probe" -and
            $Subcommand -ceq "phase-snapshot-live-probe")
    )
    if (-not $AllowedEvidenceWorker -or
        $CliModule -cne "wlr50_clean.ppo.cli") {
        throw "-ReturnFinalizedEvidenceFailure is restricted to managed diagnostic evidence workers"
    }
}

Push-Location $ProjectRoot
try {
    $RuntimeStatus = @(& git status --porcelain=v1 --untracked-files=all -- @RuntimeIdentityPaths)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect the PPO runtime implementation Git state"
    }
    $PreReservationRuntimeIdentity = Get-CommittedRuntimeIdentity
} finally {
    Pop-Location
}
if ($RuntimeStatus.Count -ne 0) {
    $DirtyRuntimePaths = ($RuntimeStatus | ForEach-Object { ([string]$_).Trim() }) -join ", "
    throw "PPO runtime implementation/config must match committed HEAD before evidence capture: $DirtyRuntimePaths"
}

$ControlledArgumentPattern = '^--(?:run-dir|seed|num-envs)(?:=|$)'
foreach ($Argument in @($BaseCliArgs) + @($CliArgs)) {
    if ([string]$Argument -match $ControlledArgumentPattern) {
        throw "BaseCliArgs/CliArgs cannot override controlled argument: $Argument"
    }
}

$ResolvedConfigs = @()
foreach ($PathValue in $ConfigPath) {
    $Candidate = if ([IO.Path]::IsPathRooted($PathValue)) {
        [IO.Path]::GetFullPath($PathValue)
    } else {
        [IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
    }
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        throw "Explicit PPO config is missing: $Candidate"
    }
    $ResolvedConfigs += $Candidate
}
if (($ResolvedConfigs | Select-Object -Unique).Count -ne $ResolvedConfigs.Count) {
    throw "ConfigPath contains duplicates"
}

$Busy = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and (
        $_.Name -match '^(kit|isaac-sim)(\.exe)?$' -or
        ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(isaacsim|isaaclab|omni\.kit|wlr50_clean\.ppo\.cli)')
    )
})
if ($Busy.Count -gt 0) {
    $Details = ($Busy | ForEach-Object { "$($_.Name) pid=$($_.ProcessId)" }) -join ", "
    throw "Another Isaac/PPO process is already running: $Details"
}

$env:PYTHONPATH = $SourceRoot
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONHASHSEED = [string]$Seed

$PlannedArguments = @($BaseCliArgs) + @(
    "--run-dir", "<reserved-immutable-run-dir>",
    "--seed", [string]$Seed,
    "--num-envs", [string]$EnvironmentCount
) + @($CliArgs)

$ReserveArgs = @(
    "-P", "-m", $ArtifactModule, "reserve-run",
    "--project-root", $ProjectRoot,
    "--run-kind", $RunKind,
    "--seed", [string]$Seed,
    "--environment-count", [string]$EnvironmentCount,
    "--training-stage", $TrainingStage,
    "--entrypoint", $CliModule,
    "--subcommand", $Subcommand
)
foreach ($Config in $ResolvedConfigs) {
    $ReserveArgs += @("--config", $Config)
}
foreach ($Argument in $PlannedArguments) {
    # The equals form allows values beginning with '--' to remain values rather
    # than being parsed as options by argparse.
    $ReserveArgs += "--invocation-argument=$Argument"
}

Push-Location $ProjectRoot
try {
    $ReservationText = (& $IsaacPython @ReserveArgs | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to reserve an immutable PPO run directory"
    }
    try {
        $Reservation = $ReservationText | ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "Artifact reservation returned invalid JSON: $ReservationText"
    }
    $RunDir = [IO.Path]::GetFullPath([string]$Reservation.run_dir)
    $RunsPrefix = $RunsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $RunDir.StartsWith($RunsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact reservation escaped the PPO runs root: $RunDir"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunDir "run_manifest.started.json") -PathType Leaf)) {
        throw "Artifact reservation did not create its started manifest: $RunDir"
    }

    $StdoutPath = Join-Path $RunDir "stdout.log"
    $StderrPath = Join-Path $RunDir "stderr.log"
    $RuntimeIdentityBeforePath = Join-Path $RunDir "committed_runtime_identity.before.json"
    $RuntimeIdentityAfterPath = Join-Path $RunDir "committed_runtime_identity.after.json"
    $RuntimeIdentityBefore = Get-CommittedRuntimeIdentity
    $PreReservationJson = ConvertTo-Json -InputObject $PreReservationRuntimeIdentity -Compress -Depth 8
    $RuntimeIdentityBeforeJson = ConvertTo-Json -InputObject $RuntimeIdentityBefore -Compress -Depth 8
    if ($RuntimeIdentityBeforeJson -cne $PreReservationJson) {
        throw "PPO runtime identity changed while reserving the evidence run"
    }
    [IO.File]::WriteAllText(
        $RuntimeIdentityBeforePath,
        ($RuntimeIdentityBefore | ConvertTo-Json -Depth 8) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    $FrozenManifest = Join-Path $ProjectRoot "artifacts\ppo_phase_v1_start\frozen_fsm_hashes.json"
    if (-not (Test-Path -LiteralPath $FrozenManifest -PathType Leaf)) {
        throw "Frozen FSM hash manifest is missing: $FrozenManifest"
    }
    $Invocation = @(
        "-P", "-m", $CliModule, $Subcommand
    ) + @($BaseCliArgs) + @(
        "--run-dir", $RunDir,
        "--seed", [string]$Seed,
        "--num-envs", [string]$EnvironmentCount
    ) + @($CliArgs)

    $ExitCode = 1
    $AuthoritativeLiveExitCode = $null
    $RuntimeIdentityPostCheckPassed = $false
    $FrozenPostCheckPassed = $false
    $FinalManifestValidated = $false
    try {
        & $IsaacPython -P -m $ArtifactModule verify-frozen `
            --project-root $ProjectRoot --manifest $FrozenManifest `
            --output (Join-Path $RunDir "frozen_hashes.before.json") `
            1>> $StdoutPath 2>> $StderrPath
        if ($LASTEXITCODE -ne 0) {
            throw "Frozen FSM hash pre-check failed: $RunDir"
        }
        & $IsaacPython @Invocation 1>> $StdoutPath 2>> $StderrPath
        $ExitCode = $LASTEXITCODE
        $LiveCommands = @(
            "baseline-eval", "zero-residual-live", "nonzero-residual-smoke",
            "reset-throughput-probe", "soft-reset-equivalence", "phase-snapshot-live-probe",
            "vector-benchmark", "initialize-zero-residual", "train", "evaluate",
            "export-inference-actor", "capture-video-source"
        )
        if ($Subcommand -in $LiveCommands) {
            $LiveResultPath = Join-Path $RunDir "live_command_result.json"
            if (-not (Test-Path -LiteralPath $LiveResultPath -PathType Leaf)) {
                $ExitCode = 2
                [IO.File]::AppendAllText(
                    $StderrPath,
                    "Live command did not publish its authoritative exit result.$([Environment]::NewLine)",
                    [Text.UTF8Encoding]::new($false)
                )
            } else {
                try {
                    $LiveResult = Get-Content -LiteralPath $LiveResultPath -Raw |
                        ConvertFrom-Json -ErrorAction Stop
                    if ([string]$LiveResult.schema -ne "wlr50_clean.live_command_result.v1" -or
                        [string]$LiveResult.command -ne $Subcommand -or
                        [int]$LiveResult.exit_code -notin @(0, 1, 2)) {
                        throw "invalid live command result payload"
                    }
                    $AuthoritativeLiveExitCode = [int]$LiveResult.exit_code
                    $ExitCode = $AuthoritativeLiveExitCode
                } catch {
                    $ExitCode = 2
                    [IO.File]::AppendAllText(
                        $StderrPath,
                        "Live command result validation failed: $($_.Exception.Message)$([Environment]::NewLine)",
                        [Text.UTF8Encoding]::new($false)
                    )
                }
            }
        }
    } catch {
        $ExitCode = 1
        [IO.File]::AppendAllText(
            $StderrPath,
            "PowerShell invocation failure: $($_.Exception.Message)$([Environment]::NewLine)",
            [Text.UTF8Encoding]::new($false)
        )
    } finally {
        try {
            $RuntimeStatusAfter = @(& git status --porcelain=v1 --untracked-files=all -- @RuntimeIdentityPaths)
            if ($LASTEXITCODE -ne 0 -or $RuntimeStatusAfter.Count -ne 0) {
                throw "PPO runtime Git state changed during evidence capture"
            }
            $RuntimeIdentityAfter = Get-CommittedRuntimeIdentity
            [IO.File]::WriteAllText(
                $RuntimeIdentityAfterPath,
                ($RuntimeIdentityAfter | ConvertTo-Json -Depth 8) + [Environment]::NewLine,
                [Text.UTF8Encoding]::new($false)
            )
            $RuntimeIdentityAfterJson = ConvertTo-Json -InputObject $RuntimeIdentityAfter -Compress -Depth 8
            if ($RuntimeIdentityAfterJson -cne $RuntimeIdentityBeforeJson) {
                throw "PPO runtime bytes or filesystem identity changed during evidence capture"
            }
            $RuntimeIdentityPostCheckPassed = $true
        } catch {
            $ExitCode = 2
            [IO.File]::AppendAllText(
                $StderrPath,
                "Runtime identity post-check failed: $($_.Exception.Message)$([Environment]::NewLine)",
                [Text.UTF8Encoding]::new($false)
            )
        }
        & $IsaacPython -P -m $ArtifactModule verify-frozen `
            --project-root $ProjectRoot --manifest $FrozenManifest `
            --output (Join-Path $RunDir "frozen_hashes.after.json") `
            1>> $StdoutPath 2>> $StderrPath
        if ($LASTEXITCODE -ne 0) {
            $ExitCode = 2
        } else {
            $FrozenPostCheckPassed = $true
        }
        $FinalizeText = (& $IsaacPython -P -m $ArtifactModule finalize-run `
            --run-dir $RunDir --exit-code $ExitCode | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to finalize PPO run manifest: $RunDir"
        }
        try {
            $Finalized = $FinalizeText | ConvertFrom-Json -ErrorAction Stop
            $ExpectedFinalManifest = [IO.Path]::GetFullPath((Join-Path $RunDir "run_manifest.json"))
            if ([IO.Path]::GetFullPath([string]$Finalized.manifest) -ne $ExpectedFinalManifest) {
                throw "finalizer returned a different run manifest"
            }
            $FinalManifest = Get-Content -LiteralPath $ExpectedFinalManifest -Raw |
                ConvertFrom-Json -ErrorAction Stop
            $ExpectedLifecycle = if ($ExitCode -eq 0) { "SUCCEEDED" } else { "FAILED" }
            if ([string]$FinalManifest.schema -ne "wlr50_clean.ppo_run_manifest.v1" -or
                [string]$FinalManifest.lifecycle -cne $ExpectedLifecycle -or
                [int]$FinalManifest.exit_code -ne $ExitCode -or
                [IO.Path]::GetFullPath([string]$FinalManifest.run_dir) -ne $RunDir) {
                throw "final run manifest does not match the finalized exit state"
            }
            $FinalManifestValidated = $true
        } catch {
            throw "Artifact finalizer returned invalid JSON: $FinalizeText"
        }
    }

    $ReturnEvidenceFailure = Test-ReturnableEvidenceFailure `
        -Enabled $ReturnFinalizedEvidenceFailure.IsPresent `
        -RunKindValue $RunKind `
        -TrainingStageValue $TrainingStage `
        -SubcommandValue $Subcommand `
        -CliModuleValue $CliModule `
        -AuthoritativeLiveExitCode $AuthoritativeLiveExitCode `
        -FinalExitCode $ExitCode `
        -RuntimeIdentityPostCheckPassed $RuntimeIdentityPostCheckPassed `
        -FrozenPostCheckPassed $FrozenPostCheckPassed `
        -FinalManifestValidated $FinalManifestValidated
    if ($ExitCode -ne 0 -and -not $ReturnEvidenceFailure) {
        throw "PPO command '$Subcommand' failed with exit code $ExitCode. Logs: $RunDir"
    }
    Write-Output $RunDir
} finally {
    Pop-Location
}
