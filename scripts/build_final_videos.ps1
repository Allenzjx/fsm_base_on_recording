[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SuccessfulTrialDir,

    [string]$OutputDir,

    [string]$PythonExe = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $ProjectRoot "src"
$Contract = Join-Path $ProjectRoot "configs\recording_motion_contract.json"
$SelectedReference = Join-Path $ProjectRoot "configs\selected_reference.json"
$ReferenceVideo = Join-Path $ProjectRoot "reference\v010\recording_clean.mp4"
$StateDerivation = Join-Path $ProjectRoot "docs\STATE_DERIVATION.md"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot "outputs\final"
}

$RequiredFiles = @(
    $PythonExe,
    $Contract,
    $SelectedReference,
    $ReferenceVideo,
    $StateDerivation,
    (Join-Path $SuccessfulTrialDir "trial_manifest.json"),
    (Join-Path $SuccessfulTrialDir "actual_viewport_video.mp4")
)
foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "FINAL_PUBLICATION_INPUT_MISSING: $RequiredFile"
    }
}

& (Join-Path $PSScriptRoot "prepare_clean_project.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "clean-room verification failed; final publication is forbidden"
}

$PythonProgram = @'
import json
import sys
from pathlib import Path

from wlr50_clean.evaluation.trial_analyzer import analyze_trial, publish_successful_trial
from wlr50_clean.evaluation.video_builder import FinalVideoBuilder

project = Path(sys.argv[1]).resolve()
trial = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
contract = Path(sys.argv[4]).resolve()
selected = Path(sys.argv[5]).resolve()
reference_video = Path(sys.argv[6]).resolve()
derivation = Path(sys.argv[7]).resolve()

# Fail before writing outputs if the physical ledgers or +/-15 percent evidence
# do not support one complete P01--P13 success.
analysis = analyze_trial(trial, contract, strict_success=True)
videos = FinalVideoBuilder(output).build(
    reference_video=reference_video,
    reference_contract=contract,
    successful_trial_dir=trial,
)
data = publish_successful_trial(
    run_dir=trial,
    output_dir=output,
    contract_path=contract,
    selected_reference_path=selected,
    state_derivation_path=derivation,
)
print(json.dumps({
    "status": "PASS",
    "video_validation": videos,
    "conformance_summary": analysis["conformance_summary"],
    "final_data": data["files"],
}, indent=2))
'@

$PreviousPythonPath = $env:PYTHONPATH
try {
    # Production/final tooling sees only this clean package, never the old FSM.
    $env:PYTHONPATH = $SourceRoot
    & $PythonExe -c $PythonProgram `
        $ProjectRoot `
        (Resolve-Path -LiteralPath $SuccessfulTrialDir).Path `
        ([System.IO.Path]::GetFullPath($OutputDir)) `
        $Contract `
        $SelectedReference `
        $ReferenceVideo `
        $StateDerivation
    if ($LASTEXITCODE -ne 0) {
        throw "FINAL_VIDEO_OR_DATA_PUBLICATION_FAILED: Python exit code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
}
