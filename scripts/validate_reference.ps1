[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IsaacPython = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

& $IsaacPython (Join-Path $ProjectRoot "tools\validate_reference.py") `
    --project-root $ProjectRoot `
    --output (Join-Path $ProjectRoot "artifacts\reference_validation.json")
if ($LASTEXITCODE -ne 0) {
    throw "v010 reference validation failed"
}
