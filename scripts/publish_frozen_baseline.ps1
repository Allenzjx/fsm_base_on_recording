[CmdletBinding()]
param(
    [string]$ReadjudicationDir = "outputs/analysis/physical_success_readjudication",
    [string]$SelectedTrial = "43",
    [switch]$ReplaceExisting,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $ProjectRoot "src"
$PreviousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = if ($PreviousPythonPath) {
        "$SourceRoot$([IO.Path]::PathSeparator)$PreviousPythonPath"
    } else {
        $SourceRoot
    }
    Push-Location $ProjectRoot
    try {
        $Arguments = @(
            "-m", "wlr50_clean.evaluation.baseline_publication",
            "--project-root", $ProjectRoot,
            "--readjudication-dir", $ReadjudicationDir,
            "--selected-trial", $SelectedTrial
        )
        if ($ReplaceExisting) {
            $Arguments += "--replace-existing"
        }
        & $PythonExe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "frozen-baseline publication failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}
