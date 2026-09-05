from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from wlr50_clean.ppo import training_cadence as subject

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/ppo_training_phase_v1.yaml"


def _inputs() -> dict:
    return subject.cadence_inputs_from_payload(yaml.safe_load(PROFILE.read_bytes()))


@pytest.mark.parametrize("n,requested,chunks,total", [(8, 25_000, 4, 213_760), (16, 50_000, 2, 213_760), (32, 100_000, 1, 215_808)])
def test_real_profile_full_windows_preserve_budget_and_fixed_iterations(n, requested, chunks, total):
    plan = subject.derive_training_cadence(selected_num_envs=n, **_inputs())
    smoke, curriculum, full = plan["stage_plans"]
    assert full["requested_policy_decisions_per_chunk"] == requested
    assert full["maximum_chunk_count"] == chunks
    assert full["ppo_iterations_per_chunk"] == 25
    assert full["policy_decisions_per_env_per_chunk"] == 3200
    assert full["minimum_full_window_policy_decisions_per_env"] == 3000
    assert full["full_window_covers_episode_timeout"] is True
    assert smoke["requested_policy_decisions_per_chunk"] == 10_000
    assert curriculum["requested_policy_decisions_per_chunk"] == 10_000
    assert curriculum["num_envs"] == 1
    assert plan["maximum_chunk_count"] == 11 + chunks
    assert [row["index"] for row in plan["chunks"]] == list(range(11 + chunks))
    assert sum(row["training_cadence"]["requested_policy_decisions_per_chunk"] for row in plan["chunks"]) == 210_000
    assert sum(row["training_cadence"]["actual_policy_decisions_per_chunk"] for row in plan["chunks"]) == total
    assert full["actual_policy_decisions_per_chunk"] * chunks == 102_400


@pytest.mark.parametrize("n", [8, 16, 32])
@pytest.mark.parametrize("field", ["requested_policy_decisions", "iterations", "stage_policy_decisions"])
def test_chunk_cannot_claim_old_short_full_cadence_or_tamper_accounting(n, field):
    expected = subject.derive_stage_cadence(stage="full-episode", num_envs=n, **_inputs())
    values = dict(requested_policy_decisions=expected["requested_policy_decisions_per_chunk"],
                  iterations=25, stage_policy_decisions=expected["actual_policy_decisions_per_chunk"])
    subject.validate_training_chunk_cadence(expected, **values)
    values[field] = 10_000 if field == "requested_policy_decisions" else values[field] - 1
    with pytest.raises(subject.TrainingCadenceError, match="differs"):
        subject.validate_training_chunk_cadence(expected, **values)


@pytest.mark.parametrize("field,value", [
    ("rollout_length", True), ("rollout_length", 0), ("decision_hz", float("nan")),
    ("timeout_s", float("inf")), ("timeout_s", 0), ("timeout_s", 214),
    ("benchmark_env_counts", [8, 32, 16]), ("benchmark_env_counts", [8, 16, 16]),
    ("benchmark_env_counts", []), ("benchmark_env_counts", [8, 16, True]),
    ("benchmark_env_counts", [8, 16, 31]), ("base_validation_interval", False),
])
def test_invalid_or_insufficient_window_configuration_fails_closed(field, value):
    inputs = _inputs()
    inputs[field] = value
    with pytest.raises(subject.TrainingCadenceError):
        subject.derive_training_cadence(selected_num_envs=8, **inputs)


@pytest.mark.parametrize("stage,n", [("mild-randomization", 8), ("full-episode", 1), ("smoke", 1), ("phase-curriculum", 8), ("full-episode", True)])
def test_stage_capacity_contract(stage, n):
    with pytest.raises(subject.TrainingCadenceError):
        subject.derive_stage_cadence(stage=stage, num_envs=n, **_inputs())


def test_nondivisible_budget_fails_instead_of_silently_rounding_requests():
    inputs = _inputs()
    inputs["budgets"] = dict(inputs["budgets"], full_episode=100_001)
    with pytest.raises(subject.TrainingCadenceError, match="divisible"):
        subject.derive_training_cadence(selected_num_envs=8, **inputs)


def test_descriptor_captures_exact_profile_and_does_not_import_simulator(capsys):
    before = set(sys.modules)
    assert subject.main(["--describe-plan", "--training-config", str(PROFILE), "--selected-num-envs", "8"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert result["profile"] == {"path": str(PROFILE.resolve()), "bytes": PROFILE.stat().st_size,
                                  "sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest()}
    assert not any(name.startswith(("isaacsim", "isaaclab", "omni.", "rsl_rl")) for name in set(sys.modules) - before)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_actual_powershell_descriptor_and_exact_record_binding(n):
    powershell = shutil.which("powershell")
    if powershell is None or not Path(r"C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe").is_file():
        pytest.skip("locked Windows PowerShell/Python runtime unavailable")
    helper = str(ROOT / "scripts/_training_cadence_plan.ps1").replace("'", "''")
    project = str(ROOT).replace("'", "''")
    script = f"""
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. '{helper}'
$before = $env:PYTHONPATH
$d = Get-TrainingCadenceDescriptor '{project}' {n}
if ($env:PYTHONPATH -cne $before) {{ throw 'PYTHONPATH leaked' }}
$copy = ($d.plan | ConvertTo-Json -Depth 30 -Compress | ConvertFrom-Json)
Assert-CadenceRecord $copy $d.plan 'Roundtrip'
$copy.stage_plans[2].requested_policy_decisions_per_chunk = 10000
$rejected = $false
try {{ Assert-CadenceRecord $copy $d.plan 'Tampered' }} catch {{ $rejected = $true }}
if (-not $rejected) {{ throw 'Short full cadence was accepted' }}
$copy = ($d.plan | ConvertTo-Json -Depth 30 -Compress | ConvertFrom-Json)
$copy.stage_plans[2].full_window_covers_episode_timeout = 1
if (Test-CadenceValueEqual $copy $d.plan) {{ throw 'Boolean coerced from number' }}
$d | ConvertTo-Json -Depth 30 -Compress
"""
    # Windows environments normalize names to uppercase when copied to a dict.
    env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                            text=True, capture_output=True, timeout=30, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["plan"] == subject.derive_training_cadence(selected_num_envs=n, **_inputs())


def test_repo_and_external_producer_use_same_descriptor_and_dynamic_final_candidate():
    repo = (ROOT / "scripts/run_ppo_training_cadence.ps1").read_text()
    external_path = ROOT.parent / "ppo_phase_training_driver_20260904.ps1"
    assert "_training_cadence_plan.ps1" in repo
    assert "PolicyDecisions = [int]$ExpectedPlan.requested_policy_decisions_per_chunk" in repo
    if not external_path.is_file():
        pytest.skip("external operator driver is not repository content")
    external = external_path.read_text()
    assert "_training_cadence_plan.ps1" in external
    assert "PolicyDecisions = [int]$expectedPlan.requested_policy_decisions_per_chunk" in external
    assert "[int]$script:CadencePlan.maximum_chunk_count - 1" in external
    assert "external_training_driver.v2" in external
    assert "$script:State.training_cadence $script:CadencePlan" in external
    assert "CliArgs = @('--policy-decisions'" not in external
    assert "$history.Count -eq 21" not in external
    assert "checkpoint_initial_zero_residual.pt" in external
    assert "Unresolved " in external and "automatic retry is forbidden" in external


def test_changing_one_record_does_not_mutate_later_chunk():
    plan = subject.derive_training_cadence(selected_num_envs=8, **_inputs())
    captured = copy.deepcopy(plan["chunks"][12])
    plan["chunks"][11]["training_cadence"]["requested_policy_decisions_per_chunk"] = 10_000
    assert plan["chunks"][12] == captured
