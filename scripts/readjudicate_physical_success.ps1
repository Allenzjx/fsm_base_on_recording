param(
    [string]$Runs = "runs",
    [string]$Contract = "configs/recording_motion_contract.json",
    [string]$Policy = "configs/conformance_policy.yaml",
    [string]$EnvironmentLock = "configs/environment_lock.json",
    [string]$Output = "outputs/analysis/physical_success_readjudication",
    [string]$Ffmpeg = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "src"
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = if ($previousPythonPath) {
        "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
    } else {
        $sourceRoot
    }
    Push-Location $projectRoot
    try {
        $arguments = @(
            "-m", "wlr50_clean.evaluation.physical_success",
            "--runs", $Runs,
            "--contract", $Contract,
            "--policy", $Policy,
            "--environment-lock", $EnvironmentLock,
            "--output", $Output
        )
        if ($Ffmpeg) {
            $arguments += @("--ffmpeg", $Ffmpeg)
        }
        & python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "physical-success readjudication failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
