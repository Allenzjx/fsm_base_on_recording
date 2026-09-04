from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def _powershell() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is required to exercise the Windows launcher contract")
    return executable


def _powershell_literal(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _wrapper_function_script(function_name: str) -> str:
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "scripts" / "_invoke_ppo_cli.ps1"
    return rf"""
$Wrapper = {_powershell_literal(wrapper)}
$Tokens = $null
$ParseErrors = $null
$Ast = [Management.Automation.Language.Parser]::ParseFile(
    $Wrapper, [ref]$Tokens, [ref]$ParseErrors
)
if ($ParseErrors.Count -ne 0) {{ throw $ParseErrors[0] }}
$Function = $Ast.Find({{
    param($Node)
    $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and
        $Node.Name -ceq '{function_name}'
}}, $true)
if ($null -eq $Function) {{ throw 'wrapper function was not found' }}
Invoke-Expression $Function.Extent.Text
"""


def test_common_launcher_requires_committed_runtime_identity_before_reservation() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")

    assert "git status --porcelain=v1 --untracked-files=all" in text
    assert '"src/wlr50_clean/ppo"' in text
    assert '"src/wlr50_clean"' in text
    assert '"src/wlr50_clean/fsm"' in text
    assert '"src/wlr50_clean/sensing"' in text
    assert '"src/wlr50_clean/infrastructure"' in text
    assert '"scripts"' in text
    assert '"configs"' in text
    assert '"reference/ppo_phase_snapshots"' in text
    assert '"artifacts/ppo_phase_v1_start"' in text
    assert text.index("git status --porcelain=v1") < text.index("reserve-run")
    assert "must match committed HEAD before evidence capture" in text


def test_common_launcher_captures_and_rechecks_exact_runtime_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")

    for required in (
        "function Get-CommittedRuntimeIdentity",
        "creation_time_utc_ticks",
        "last_write_time_utc_ticks",
        "content_sha256",
        "committed_runtime_identity.before.json",
        "committed_runtime_identity.after.json",
        "RuntimeStatusAfter",
        "bytes or filesystem identity changed during evidence capture",
    ):
        assert required in text
    assert text.index("committed_runtime_identity.before.json") < text.index(
        "& $IsaacPython @Invocation"
    )
    assert text.index("$RuntimeIdentityAfter = Get-CommittedRuntimeIdentity") > text.index(
        "& $IsaacPython @Invocation"
    )
    assert text.rstrip().endswith("}")


def test_wrapper_runtime_identity_uses_python_ordinal_order_and_canonical_hashes() -> None:
    root = Path(__file__).resolve().parents[2]
    script = _wrapper_function_script("Get-CommittedRuntimeIdentity") + rf"""
$Assignment = $Ast.Find({{
    param($Node)
    $Node -is [Management.Automation.Language.AssignmentStatementAst] -and
        $Node.Left.Extent.Text -ceq '$RuntimeIdentityPaths'
}}, $true)
if ($null -eq $Assignment) {{ throw 'runtime path assignment was not found' }}
$ProjectRoot = {_powershell_literal(root)}
Invoke-Expression $Assignment.Extent.Text
Push-Location $ProjectRoot
try {{ Get-CommittedRuntimeIdentity | ConvertTo-Json -Compress -Depth 8 }}
finally {{ Pop-Location }}
"""
    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    identity = json.loads(completed.stdout)
    rows = identity["files"]
    paths = [row["path"] for row in rows]

    assert paths == sorted(set(paths))
    assert paths.index("configs/ppo_domain_randomization.yaml") < paths.index(
        "configs/ppo_domain_randomization_v2.yaml"
    )
    assert identity["file_count"] == len(rows)
    canonical_rows = json.dumps(
        rows, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    canonical_content = json.dumps(
        [
            {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in rows
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert identity["aggregate_sha256"] == hashlib.sha256(canonical_rows).hexdigest()
    assert identity["content_sha256"] == hashlib.sha256(canonical_content).hexdigest()


def test_expected_vector_failure_return_decision_is_strictly_scoped() -> None:
    root = Path(__file__).resolve().parents[2]
    script = _wrapper_function_script("Test-ReturnableEvidenceFailure") + r"""
$Base = @{
    Enabled = $true
    RunKindValue = 'vector_benchmark'
    TrainingStageValue = 'backend-benchmark'
    SubcommandValue = 'vector-benchmark'
    CliModuleValue = 'wlr50_clean.ppo.cli'
    AuthoritativeLiveExitCode = 2
    FinalExitCode = 2
    RuntimeIdentityPostCheckPassed = $true
    FrozenPostCheckPassed = $true
    FinalManifestValidated = $true
}
$Cases = [ordered]@{
    exact_failure = Test-ReturnableEvidenceFailure @Base
}
foreach ($Mutation in @(
    @('disabled', 'Enabled', $false),
    @('wrong_run_kind', 'RunKindValue', 'train'),
    @('wrong_stage', 'TrainingStageValue', 'smoke'),
    @('wrong_subcommand', 'SubcommandValue', 'evaluate'),
    @('wrong_module', 'CliModuleValue', 'wlr50_clean.ppo.delivery_cli'),
    @('missing_authoritative_exit', 'AuthoritativeLiveExitCode', $null),
    @('successful_live_exit', 'AuthoritativeLiveExitCode', 0),
    @('wrong_final_exit', 'FinalExitCode', 1),
    @('runtime_check_failed', 'RuntimeIdentityPostCheckPassed', $false),
    @('frozen_check_failed', 'FrozenPostCheckPassed', $false),
    @('manifest_invalid', 'FinalManifestValidated', $false)
)) {
    $Arguments = @{} + $Base
    $Arguments[$Mutation[1]] = $Mutation[2]
    $Cases[$Mutation[0]] = Test-ReturnableEvidenceFailure @Arguments
}
$Cases | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    cases = json.loads(completed.stdout)

    assert cases.pop("exact_failure") is True
    assert cases
    assert not any(cases.values())

    common = (root / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")
    benchmark = (root / "scripts" / "benchmark_vectorized_ppo.ps1").read_text(
        encoding="utf-8"
    )
    phase_probe = (
        root / "scripts" / "run_phase_snapshot_live_probe.ps1"
    ).read_text(encoding="utf-8")
    assert "-ReturnFinalizedEvidenceFailure is restricted" in common
    assert "-ReturnFinalizedEvidenceFailure" in benchmark
    assert "-ReturnFinalizedEvidenceFailure" in phase_probe
    for path in (root / "scripts").glob("*.ps1"):
        if path.name in {
            "_invoke_ppo_cli.ps1",
            "benchmark_vectorized_ppo.ps1",
            "run_phase_snapshot_live_probe.ps1",
        }:
            continue
        assert "-ReturnFinalizedEvidenceFailure" not in path.read_text(encoding="utf-8")
