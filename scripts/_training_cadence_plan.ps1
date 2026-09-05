# Read-only cadence descriptor and producer-side independent arithmetic checks.
# Dot-sourcing this file does not create a run, checkpoint, optimizer or simulator.
function Test-CadenceValueEqual([object]$Left, [object]$Right) {
    if ($null -eq $Left -or $null -eq $Right) { return ($null -eq $Left -and $null -eq $Right) }
    if ($Left -is [bool] -or $Right -is [bool]) { return ($Left -is [bool] -and $Right -is [bool] -and $Left -eq $Right) }
    if ($Left -is [string] -or $Right -is [string]) { return ($Left -is [string] -and $Right -is [string] -and $Left -ceq $Right) }
    if ($Left -is [array] -or $Right -is [array]) {
        if ($Left -isnot [array] -or $Right -isnot [array] -or $Left.Count -ne $Right.Count) { return $false }
        for ($i = 0; $i -lt $Left.Count; $i++) {
            if (-not (Test-CadenceValueEqual $Left[$i] $Right[$i])) { return $false }
        }
        return $true
    }
    if ($Left -is [pscustomobject] -or $Right -is [pscustomobject]) {
        if ($Left -isnot [pscustomobject] -or $Right -isnot [pscustomobject]) { return $false }
        $leftNames = @($Left.PSObject.Properties.Name | Sort-Object)
        $rightNames = @($Right.PSObject.Properties.Name | Sort-Object)
        if (($leftNames -join ',') -cne ($rightNames -join ',')) { return $false }
        foreach ($name in $leftNames) {
            if (-not (Test-CadenceValueEqual $Left.$name $Right.$name)) { return $false }
        }
        return $true
    }
    $numericTypes = @([int], [long], [double], [decimal], [single], [int16], [byte])
    if ($Left.GetType() -notin $numericTypes -or $Right.GetType() -notin $numericTypes) { return $false }
    return ([double]$Left -eq [double]$Right -and -not [double]::IsNaN([double]$Left) -and -not [double]::IsInfinity([double]$Left))
}

function Assert-CadenceRecord([object]$Actual, [object]$Expected, [string]$Label) {
    if (-not (Test-CadenceValueEqual $Actual $Expected)) { throw "$Label differs from the profile-derived cadence" }
}

function Get-TrainingCadenceDescriptor([string]$ProjectRoot, [int]$SelectedNumEnvs) {
    $config = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'configs\ppo_training_phase_v1.yaml'))
    $python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
        $rows = @(& $python -P -m wlr50_clean.ppo.training_cadence --describe-plan --training-config $config --selected-num-envs $SelectedNumEnvs 2>&1)
        if ($LASTEXITCODE -ne 0 -or $rows.Count -ne 1 -or
            @($rows | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }).Count -ne 0) {
            throw 'Pure cadence descriptor failed or polluted stdout/stderr'
        }
        $descriptor = [string]$rows[0] | ConvertFrom-Json -ErrorAction Stop
    } finally { $env:PYTHONPATH = $previousPythonPath }
    if ([string]$descriptor.profile.path -cne $config -or
        [long]$descriptor.profile.bytes -ne (Get-Item -LiteralPath $config).Length -or
        [string]$descriptor.profile.sha256 -cne (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw 'Cadence descriptor is not bound to the current profile bytes'
    }
    $plan = $descriptor.plan
    if ([string]$plan.schema -cne 'wlr50_clean.ppo_training_cadence.v1' -or
        [int]$plan.selected_num_envs -ne $SelectedNumEnvs -or
        [int]$plan.base_validation_interval_policy_decisions -ne 10000 -or
        [string]$plan.base_validation_interval_scope -cne 'smoke_and_phase_curriculum' -or
        @($plan.stage_plans).Count -ne 3) { throw 'Unexpected cadence plan schema/base scope' }
    $index = 0
    $names = @('smoke', 'phase-curriculum', 'full-episode')
    for ($s = 0; $s -lt 3; $s++) {
        $stage = $plan.stage_plans[$s]
        $n = if ($s -eq 1) { 1 } else { $SelectedNumEnvs }
        $budget = if ($s -eq 0) { 10000L } else { 100000L }
        $maxN = [int](@($stage.benchmark_env_counts) | Measure-Object -Maximum).Maximum
        if ((@($stage.benchmark_env_counts) -join ',') -cne '8,16,32' -or $SelectedNumEnvs -notin @($stage.benchmark_env_counts)) {
            throw 'Cadence benchmark capacities differ from the frozen profile contract'
        }
        if ($s -eq 2 -and ($maxN % $n -ne 0 -or ($budget * $n) % $maxN -ne 0)) {
            throw 'Full cadence capacity/budget is not exactly divisible'
        }
        $requested = if ($s -eq 2) { [long]($budget * $n / $maxN) } else { 10000L }
        $count = if ($s -eq 2) { [int]($maxN / $n) } else { [int]($budget / $requested) }
        $iterations = [long][Math]::Ceiling([double]$requested / ($n * 128))
        $expected = [pscustomobject][ordered]@{
            schema = 'wlr50_clean.ppo_training_stage_cadence.v1'; stage = $names[$s]; num_envs = $n
            cadence_basis = $(if ($s -eq 2) { 'full_stage_budget_scaled_by_selected_capacity' } else { 'configured_global_policy_decision_interval' })
            base_validation_interval_policy_decisions = 10000; base_validation_interval_scope = 'smoke_and_phase_curriculum'
            stage_requested_policy_decisions = $budget; requested_policy_decisions_per_chunk = $requested
            maximum_chunk_count = $count; ppo_iterations_per_chunk = $iterations; rollout_length = 128
            ppo_batch_policy_decisions = $n * 128; actual_policy_decisions_per_chunk = $iterations * $n * 128
            policy_decisions_per_env_per_chunk = $iterations * 128; benchmark_env_counts = @(8, 16, 32)
            maximum_benchmark_num_envs = $maxN; policy_decision_hz = 15; episode_timeout_s = 200
            minimum_full_window_policy_decisions_per_env = 3000
            full_window_covers_episode_timeout = $(if ($s -eq 2) { $true } else { $null })
        }
        Assert-CadenceRecord $stage $expected 'Stage descriptor'
        if ($requested * $count -ne $budget -or ($s -eq 2 -and ($iterations -ne 25 -or $iterations * 128 -lt 3000))) {
            throw 'Stage arithmetic does not preserve budget/full timeout window'
        }
        for ($j = 0; $j -lt $count; $j++) {
            $expectedChunk = [pscustomobject]@{ index = $index; stage_chunk_index = $j; training_cadence = $expected }
            Assert-CadenceRecord $plan.chunks[$index] $expectedChunk 'Chunk descriptor'
            $index++
        }
    }
    if (@($plan.chunks).Count -ne $index -or [int]$plan.maximum_chunk_count -ne $index) { throw 'Cadence chunk count mismatch' }
    return $descriptor
}
