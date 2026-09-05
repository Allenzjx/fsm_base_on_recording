from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is unavailable")


@pytest.mark.parametrize("case, expected_count, rejected", (
    ("matching", 0, False),
    ("missing", 1, True),
    ("extra", 1, True),
))
def test_real_promotion_gate_count_and_branch_under_strict_mode(
    case, expected_count, rejected,
):
    source_path = str(ROOT / "scripts/run_ppo_training_cadence.ps1").replace("'", "''")
    command = f"""
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '{source_path}', [ref]$tokens, [ref]$errors)
if (@($errors).Count -ne 0) {{ throw 'Production script has parse errors' }}
$functionAst = $ast.Find({{
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Test-StrictPromotionForCurrentChunk'
}}, $true)
$branches = @($functionAst.FindAll({{
    param($node)
    $node -is [System.Management.Automation.Language.IfStatementAst] -and
        $node.Extent.Text.Contains('throw "Promotion decision cannot authorize cadence early stop"')
}}, $true))
if ($branches.Count -ne 1) {{ throw 'Expected the real promotion rejection branch' }}
$countExpressions = @($branches[0].FindAll({{
    param($node)
    $node -is [System.Management.Automation.Language.MemberExpressionAst] -and
        $node.Member.Value -eq 'Count' -and
        $node.Expression.Extent.Text.Contains('Compare-Object $PromotionGates $GateNames')
}}, $true))
if ($countExpressions.Count -ne 1) {{ throw 'Expected one gate count in the real branch' }}
$gateAssignment = $ast.Find({{
    param($node)
    $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $node.Left.Extent.Text -eq '$PromotionGates'
}}, $true)
. ([scriptblock]::Create($gateAssignment.Extent.Text))
$GateNames = @($PromotionGates)
if ('{case}' -eq 'missing') {{ $GateNames = @($PromotionGates | Select-Object -Skip 1) }}
if ('{case}' -eq 'extra') {{ $GateNames += 'unexpected_extra_gate' }}
$ValidationSeeds = @(2001, 2002, 2003, 2004, 2005)
$Seeds = @($ValidationSeeds)
$Decision = [pscustomobject]@{{
    schema = 'wlr50_clean.ppo_evaluation_artifacts.v1'
    paired_episode_count = 5
    minimum_paired_seeds = 5
    frozen_hashes_unchanged = $true
    promotion = [pscustomobject]@{{ first_failed_gate = $null }}
}}
# Both executions use AST text from production. No replacement count expression
# or permissive promotion predicate can conceal null.Count under StrictMode.
$count = & ([scriptblock]::Create($countExpressions[0].Extent.Text))
$rejected = $false
try {{
    & ([scriptblock]::Create($branches[0].Extent.Text))
}} catch {{
    if ($_.Exception.Message -cne 'Promotion decision cannot authorize cadence early stop') {{ throw }}
    $rejected = $true
}}
[pscustomobject]@{{ count = $count; rejected = $rejected }} | ConvertTo-Json -Compress
"""
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    # A PS7 parent can otherwise inject incompatible module paths into WinPS5.1.
    env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}
    result = subprocess.run(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stderr
    assert json.loads(result.stdout) == {"count": expected_count, "rejected": rejected}
