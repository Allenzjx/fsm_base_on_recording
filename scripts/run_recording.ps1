[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReferenceVideo = Join-Path $ProjectRoot "reference\v010\recording_clean.mp4"

if (-not (Test-Path -LiteralPath $ReferenceVideo -PathType Leaf)) {
    throw "REFERENCE_INCOMPLETE: the frozen v010 clean video is missing; do not substitute another version"
}

& (Join-Path $PSScriptRoot "validate_reference.ps1")
if ($LASTEXITCODE -ne 0) { throw "v010 reference validation failed" }

Write-Output $ReferenceVideo
