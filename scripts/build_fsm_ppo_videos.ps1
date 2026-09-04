[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Checkpoint = "outputs\ppo_phase_v1\checkpoints\checkpoint_improved.pt",
    [string]$CheckpointManifest = "outputs\ppo_phase_v1\checkpoints\checkpoint_improved_manifest.json",
    [string]$OutputRoot = "outputs\ppo_phase_v1",
    [string]$Ffmpeg = "",
    [int]$Seed = 4001,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CliArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($Seed -ne 4001) {
    throw "final FSM/PPO videos require the locked video seed 4001"
}

$Configs = @(
    "configs\ppo_training_phase_v1.yaml",
    "configs\ppo_interface_v2.yaml",
    "configs\ppo_phase_effective_entry_v1.json",
    "configs\ppo_phase_effective_entry_v1.sha256",
    "configs\ppo_observation_schema_v2.json",
    "configs\ppo_phase_action_masks_v2.yaml",
    "configs\ppo_phase_objectives_v2.yaml",
    "configs\ppo_reward_v2.yaml",
    "configs\ppo_termination_v2.yaml",
    "configs\ppo_domain_randomization_v2.yaml",
    "configs\environment_lock.json",
    "configs\frozen_successful_fsm.yaml",
    "configs\fsm_states.yaml",
    "configs\recording_motion_contract.json",
    "configs\ppo_action_projection.yaml",
    "configs\ppo_observation_schema.json",
    "configs\conformance_policy.yaml"
)
$CommonCaptureArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--episode-count", "1",
    "--deterministic",
    "--capture-fps", "15",
    "--maximum-duration-s", "200",
    "--no-headless"
)

function Invoke-SingleVideoSourceCapture {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("fsm", "ppo")][string]$Role,
        [Parameter(Mandatory = $true)][string[]]$AdditionalArgs
    )

    $BaseArgs = @($CommonCaptureArgs) + @(
        "--video-source-role", $Role
    ) + @($AdditionalArgs)
    $InvocationOutput = @(
        & (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
            -RunKind "video-source-$Role" `
            -TrainingStage "video-source-$Role-fresh-process" `
            -Subcommand "capture-video-source" `
            -ConfigPath $Configs `
            -Seed $Seed `
            -EnvironmentCount 1 `
            -BaseCliArgs $BaseArgs `
            -CliArgs $CliArgs
    )
    $RunCandidates = @(
        $InvocationOutput | Where-Object {
            Test-Path -LiteralPath ([string]$_) -PathType Container
        }
    )
    if ($RunCandidates.Count -ne 1) {
        throw "$Role capture did not return exactly one immutable run directory"
    }
    $RunDir = [IO.Path]::GetFullPath([string]$RunCandidates[0])
    $SourceDir = Join-Path $RunDir "video_source"
    foreach ($RequiredName in @(
        "actual_viewport_video.mp4",
        "viewport_buffer_video_manifest.json",
        "viewport_frame_ledger.jsonl",
        "policy_trace.jsonl",
        "ppo_video_source_manifest.json",
        "trial_manifest.json"
    )) {
        $RequiredPath = Join-Path $SourceDir $RequiredName
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "$Role capture is missing required evidence: $RequiredPath"
        }
    }
    return [PSCustomObject]@{
        RunDir = $RunDir
        SourceDir = [IO.Path]::GetFullPath($SourceDir)
        Manifest = Get-Content -LiteralPath (Join-Path $SourceDir "ppo_video_source_manifest.json") -Raw | ConvertFrom-Json
    }
}

# These invocations are intentionally sequential and single-shot.  The first
# Python/Isaac process owns only the zero-residual FSM episode; after it exits,
# a new Python/Isaac process loads and captures the promoted PPO checkpoint.
$FsmCapture = Invoke-SingleVideoSourceCapture -Role "fsm" -AdditionalArgs @(
    "--residual-mode", "zero"
)
$PpoCapture = Invoke-SingleVideoSourceCapture -Role "ppo" -AdditionalArgs @(
    "--checkpoint", $Checkpoint,
    "--checkpoint-manifest", $CheckpointManifest
)

if (
    [int]$FsmCapture.Manifest.capture_process_id -eq [int]$PpoCapture.Manifest.capture_process_id -or
    [string]$FsmCapture.Manifest.capture_process_instance_id -eq [string]$PpoCapture.Manifest.capture_process_instance_id
) {
    throw "FSM and PPO sources do not prove two independent fresh processes"
}
if (
    [string]$FsmCapture.Manifest.policy_label -ne "fsm_zero_residual" -or
    [string]$PpoCapture.Manifest.policy_label -ne "ppo_deterministic_mean"
) {
    throw "captured source manifests have the wrong policy roles"
}

$PublishArgs = @(
    "--training-config", $Configs[0],
    "--interface-config", $Configs[1],
    "--episode-count", "1",
    "--deterministic",
    "--fsm-video-source-dir", $FsmCapture.SourceDir,
    "--ppo-video-source-dir", $PpoCapture.SourceDir,
    "--output-root", $OutputRoot
)
if (-not [string]::IsNullOrWhiteSpace($Ffmpeg)) {
    $PublishArgs += @("--ffmpeg", $Ffmpeg)
}

# Publication is a third, offline process.  The command validates both source
# episodes and runs ffmpeg only; it is deliberately absent from LIVE_COMMANDS.
& (Join-Path $PSScriptRoot "_invoke_ppo_cli.ps1") `
    -RunKind "video-publication" `
    -TrainingStage "video-publication-offline" `
    -Subcommand "publish-videos" `
    -ConfigPath $Configs `
    -Seed $Seed `
    -EnvironmentCount 1 `
    -BaseCliArgs $PublishArgs
