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

    [string[]]$BaseCliArgs = @(),

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$SourceRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "src"))
$RunsRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "runs\ppo_phase_v1"))
$IsaacPython = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
$CliModule = "wlr50_clean.ppo.cli"
$ArtifactModule = "wlr50_clean.ppo.artifacts"

if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Locked Isaac Python is missing: $IsaacPython"
}
if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Project source directory is missing: $SourceRoot"
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
            "soft-reset-equivalence", "vector-benchmark", "train", "evaluate",
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
                    $ExitCode = [int]$LiveResult.exit_code
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
        & $IsaacPython -P -m $ArtifactModule verify-frozen `
            --project-root $ProjectRoot --manifest $FrozenManifest `
            --output (Join-Path $RunDir "frozen_hashes.after.json") `
            1>> $StdoutPath 2>> $StderrPath
        if ($LASTEXITCODE -ne 0) {
            $ExitCode = 2
        }
        & $IsaacPython -P -m $ArtifactModule finalize-run --run-dir $RunDir --exit-code $ExitCode
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to finalize PPO run manifest: $RunDir"
        }
    }

    if ($ExitCode -ne 0) {
        throw "PPO command '$Subcommand' failed with exit code $ExitCode. Logs: $RunDir"
    }
    Write-Output $RunDir
} finally {
    Pop-Location
}
