[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SuccessfulTrialDir,

    [string]$OutputDir,

    [string]$ReferenceFrameLedger,

    [string]$ReferenceRawVideo,

    [string]$PhysicalSelectionEvidence,

    [string]$PythonExe = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $ProjectRoot "src"
$Contract = Join-Path $ProjectRoot "configs\recording_motion_contract.json"
$ReferenceVideo = Join-Path $ProjectRoot "reference\v010\recording_clean.mp4"
$ReferenceSourceManifest = Join-Path $ProjectRoot "reference\v010\source_manifest.json"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot "outputs\final"
}

if ([string]::IsNullOrWhiteSpace($PhysicalSelectionEvidence)) {
    $PhysicalSelectionEvidence = Join-Path $ProjectRoot `
        "outputs\analysis\physical_success_readjudication\selected_success_trial.json"
}

$SourceManifestPayload = Get-Content -LiteralPath $ReferenceSourceManifest -Raw | ConvertFrom-Json
$VideoSource = $SourceManifestPayload.files |
    Where-Object { $_.role -eq "complete linked v010 active-viewport recording video" } |
    Select-Object -First 1
if ($null -eq $VideoSource) {
    throw "REFERENCE_VIDEO_SOURCE_BINDING_MISSING: $ReferenceSourceManifest"
}
$PublicationManifest = Join-Path (Split-Path -Parent $VideoSource.source_path) "v010_recording_video_manifest.json"
if (-not (Test-Path -LiteralPath $PublicationManifest -PathType Leaf)) {
    throw "REFERENCE_VIDEO_PUBLICATION_MANIFEST_MISSING: $PublicationManifest"
}
$PublicationPayload = Get-Content -LiteralPath $PublicationManifest -Raw | ConvertFrom-Json

if ([string]::IsNullOrWhiteSpace($ReferenceFrameLedger)) {
    # Bind sim time through the immutable raw-capture frame ledger.  Never use
    # the legacy MP4 duration atom as the clipping clock.
    $ReferenceFrameLedger = Join-Path $PublicationPayload.source_video_run "viewport_frame_ledger.jsonl"
}
if ([string]::IsNullOrWhiteSpace($ReferenceRawVideo)) {
    $ReferenceRawVideo = $PublicationPayload.source_video_path
}

$RequiredFiles = @(
    $PythonExe,
    $Contract,
    $ReferenceVideo,
    $ReferenceRawVideo,
    $ReferenceFrameLedger,
    $PhysicalSelectionEvidence,
    (Join-Path $SuccessfulTrialDir "trial_manifest.json"),
    (Join-Path $SuccessfulTrialDir "actual_viewport_video.mp4"),
    (Join-Path $SuccessfulTrialDir "viewport_frame_ledger.jsonl")
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

from wlr50_clean.evaluation.trial_analyzer import analyze_trial
from wlr50_clean.evaluation.video_builder import FinalVideoBuilder

project = Path(sys.argv[1]).resolve()
trial = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
contract = Path(sys.argv[4]).resolve()
reference_video = Path(sys.argv[5]).resolve()
reference_raw_video = Path(sys.argv[6]).resolve()
reference_ledger = Path(sys.argv[7]).resolve()
selection_evidence = Path(sys.argv[8]).resolve()

# Reclassify the immutable raw Trial through the current physical acceptance
# layers.  The old manifest is evidence, not a mutable status flag.
# Video packaging consumes the independent physical reclassification below.
# Reference/feedback/settling diagnostics must never veto publication.
analysis = analyze_trial(trial, contract, strict_success=False)
videos = FinalVideoBuilder(output).build(
    reference_video=reference_video,
    reference_raw_video=reference_raw_video,
    reference_frame_ledger=reference_ledger,
    reference_contract=contract,
    successful_trial_dir=trial,
    physical_reclassification=selection_evidence,
)
print(json.dumps({
    "status": "PASS",
    "video_validation": videos,
    "conformance_summary": analysis["conformance_summary"],
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
        $ReferenceVideo `
        (Resolve-Path -LiteralPath $ReferenceRawVideo).Path `
        (Resolve-Path -LiteralPath $ReferenceFrameLedger).Path `
        (Resolve-Path -LiteralPath $PhysicalSelectionEvidence).Path
    if ($LASTEXITCODE -ne 0) {
        throw "FINAL_VIDEO_OR_DATA_PUBLICATION_FAILED: Python exit code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
}
