[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$RunsRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "runs"))
$CleanSource = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "src"))
$IsaacPython = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
$ResolvedRunDir = if ([IO.Path]::IsPathRooted($RunDir)) {
    [IO.Path]::GetFullPath($RunDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $RunDir))
}
$RunsPrefix = $RunsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

if (-not $ResolvedRunDir.StartsWith($RunsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "RunDir must be a new child of $RunsRoot"
}
if (Test-Path -LiteralPath $ResolvedRunDir) {
    throw "Immutable run directory already exists: $ResolvedRunDir"
}
if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Locked Isaac Python is missing: $IsaacPython"
}

& (Join-Path $PSScriptRoot "prepare_clean_project.ps1")
if ($LASTEXITCODE -ne 0) { throw "Clean-room preparation failed" }

$Busy = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and (
        $_.Name -match '^(kit|isaac-sim)(\.exe)?$' -or
        ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(isaacsim|isaaclab|omni\.kit|wlr50_clean\.infrastructure\.app_runtime)')
    )
})
if ($Busy.Count -gt 0) {
    $Details = ($Busy | ForEach-Object { "$($_.Name) pid=$($_.ProcessId)" }) -join ", "
    throw "Another Isaac/Kit process is already running; refusing a second instance: $Details"
}

$env:PYTHONPATH = $CleanSource
$env:PYTHONNOUSERSITE = "1"
$env:HEADLESS = "0"
$env:LIVESTREAM = "0"
$env:ENABLE_CAMERAS = "1"

Push-Location $ProjectRoot
try {
    & $IsaacPython -P -m wlr50_clean.infrastructure.app_runtime --run-dir $ResolvedRunDir
    $TrialExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $TrialExitCode
