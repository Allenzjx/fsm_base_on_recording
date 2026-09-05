from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is unavailable")


def _source_span(source: str, start: str, end: str) -> str:
    assert source.count(start) == 1, start
    assert source.count(end) == 1, end
    return source[source.index(start):source.index(end)]


@pytest.fixture
def bound_train_script(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    train = scripts / "train_phase_residual_ppo.ps1"
    train.write_bytes((ROOT / "scripts" / train.name).read_bytes())
    helper = (ROOT / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")
    # Execute the production ParamBlock, semantic guards and both argument
    # assemblers verbatim. Only lifecycle work (Git, artifact reservation and
    # Python/Isaac execution) is omitted; no permissive helper mock can swallow
    # a budget hidden in CliArgs or accept a nonexistent named parameter.
    param_block = helper[:helper.index("Set-StrictMode -Version Latest")]
    guard = _source_span(helper, "$HelperOwnedArgumentNames =", "$ResolvedConfigs = @()")
    planned = _source_span(helper, "$PlannedArguments = @(", "$ReserveArgs = @(")
    invocation = _source_span(
        helper, "$Invocation = @(", "$ExitCode = 1\n    $AuthoritativeLiveExitCode = $null"
    )
    capture = """
[ordered]@{
    run_kind = $RunKind
    training_stage = $TrainingStage
    subcommand = $Subcommand
    base_cli_args = @($BaseCliArgs)
    cli_args = @($CliArgs)
    planned_arguments = @($PlannedArguments)
    invocation = @($Invocation)
} | ConvertTo-Json -Depth 5 -Compress
"""
    (scripts / "_invoke_ppo_cli.ps1").write_text(
        param_block + '\nSet-StrictMode -Version Latest\n$ErrorActionPreference = "Stop"\n'
        + guard + planned + '\n$RunDir = "<reserved-immutable-run-dir>"\n'
        + invocation + capture,
        encoding="utf-8",
    )
    return train


def _invoke_train(script: Path, *, budget=None, cli_args=(), stage="smoke"):
    case = json.dumps({"budget": budget, "cli_args": list(cli_args), "stage": stage})
    escaped_script = str(script).replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$case = ConvertFrom-Json @'
{case}
'@
$parameters = @{{
    Seed = 1001; Stage = $case.stage; NumEnvs = 8
    VectorBenchmarkMatrix = 'matrix.json'
}}
if ($null -ne $case.budget) {{ $parameters.PolicyDecisions = $case.budget }}
if (@($case.cli_args).Count -gt 0) {{ $parameters.CliArgs = [string[]]@($case.cli_args) }}
if ($case.stage -ne 'smoke') {{
    $parameters.Checkpoint = 'checkpoint.pt'
    $parameters.CheckpointManifest = 'checkpoint_manifest.json'
}}
& '{escaped_script}' @parameters
"""
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-EncodedCommand", encoded],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )


@pytest.mark.parametrize("budget, stage", (
    (1, "smoke"), (10000, "smoke"), (25000, "full-episode"),
    (50000, "full-episode"), (100000, "full-episode"),
))
def test_named_policy_budget_reaches_real_managed_guard_and_both_argument_lists(
    bound_train_script, budget, stage
):
    result = _invoke_train(bound_train_script, budget=budget, stage=stage)
    assert result.returncode == 0, result.stderr
    capture = json.loads(result.stdout)
    assert capture["run_kind"] == capture["subcommand"] == "train"
    assert capture["training_stage"] == stage
    assert capture["cli_args"] == []
    for key in ("base_cli_args", "planned_arguments", "invocation"):
        arguments = capture[key]
        assert arguments.count("--policy-decisions") == 1
        assert arguments[arguments.index("--policy-decisions") + 1] == str(budget)
        assert arguments[arguments.index("--stage") + 1] == stage
    assert capture["invocation"][:4] == ["-P", "-m", "wlr50_clean.ppo.cli", "train"]
    assert capture["invocation"][4:] == capture["planned_arguments"]


def test_omitted_policy_budget_retains_profile_default(bound_train_script):
    result = _invoke_train(bound_train_script)
    assert result.returncode == 0, result.stderr
    capture = json.loads(result.stdout)
    for key in ("base_cli_args", "planned_arguments", "invocation"):
        assert "--policy-decisions" not in capture[key]


@pytest.mark.parametrize("budget", (0, -1, 100001, "not-a-number"))
def test_invalid_named_policy_budget_fails_real_parameter_binding(bound_train_script, budget):
    result = _invoke_train(bound_train_script, budget=budget)
    assert result.returncode != 0
    assert "PolicyDecisions" in result.stderr
    assert not result.stdout.strip()


@pytest.mark.parametrize("cli_args", (
    ("--policy-decisions", "10000"),
    ("--policy-decisions=10000",),
    ("--policy-deci", "10000"),
    ("--POLICY-DECISIONS=10000",),
))
@pytest.mark.parametrize("budget", (None, 10000))
def test_generic_policy_budget_override_remains_locked_by_real_helper(
    bound_train_script, cli_args, budget
):
    result = _invoke_train(bound_train_script, budget=budget, cli_args=cli_args)
    assert result.returncode != 0
    assert "CliArgs cannot override managed launcher/semantic argument" in result.stderr
    assert not result.stdout.strip()
