[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IsaacPython = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if ((git -C $ProjectRoot branch --show-current) -ne "main") {
    throw "Clean project must remain on branch main"
}

& $IsaacPython (Join-Path $ProjectRoot "tools\verify_clean_room.py")
if ($LASTEXITCODE -ne 0) { throw "Clean-room verification failed" }

& (Join-Path $PSScriptRoot "validate_reference.ps1")
if ($LASTEXITCODE -ne 0) { throw "Reference validation failed" }

& $IsaacPython -c "from pathlib import Path; from wlr50_clean.reference.motion_contract import load_motion_contract; c=load_motion_contract(Path(r'$ProjectRoot')/'configs'/'recording_motion_contract.json'); assert len(c.phases)==13; print('compact contract: P01-P13 valid')"
if ($LASTEXITCODE -ne 0) { throw "Compact contract validation failed" }
