from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import evaluation_artifacts, finalization
from wlr50_clean.ppo import paired_aggregate_binding
from wlr50_clean.ppo import training_orchestration as orchestration
from wlr50_clean.ppo.checkpoint_promotion import REQUIRED_PROMOTION_GATES


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _producer_directory(project_root: Path, step: int) -> Path:
    """Execute only the real producer's path assignment, never its driver."""
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required to exercise the real cadence producer")
    cadence = Path(__file__).resolve().parents[2] / "scripts/run_ppo_training_cadence.ps1"
    script = r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:WLR_PATH_TEST_CADENCE, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'Cadence syntax errors' }
$assignments = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $node.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
    $node.Left.VariablePath.UserPath -ceq 'MetricsDirectory'
}, $true))
if ($assignments.Count -ne 1) { throw 'Expected exactly one cadence metrics assignment' }
$ProjectRoot = $env:WLR_PATH_TEST_PROJECT_ROOT
$Chunk = [pscustomobject]@{ GlobalStep = [long]$env:WLR_PATH_TEST_GLOBAL_STEP }
Invoke-Expression $assignments[0].Right.Extent.Text
"""
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **os.environ,
            "WLR_PATH_TEST_CADENCE": str(cadence),
            "WLR_PATH_TEST_PROJECT_ROOT": str(project_root),
            "WLR_PATH_TEST_GLOBAL_STEP": str(step),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 1
    return Path(lines[0]).resolve()


@pytest.fixture
def decision_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path.resolve()
    step = 121600
    output_root = root / "outputs/ppo_phase_v1"
    checkpoint = output_root / "checkpoints/checkpoint_full-episode_121600.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test-only immutable checkpoint")
    checkpoint_manifest = checkpoint.with_suffix(".json")
    _write_json(checkpoint_manifest, {"global_policy_decisions": step})
    checkpoint_record = _record(checkpoint)
    manifest_record = _record(checkpoint_manifest)
    workers = []
    for seed in orchestration.VALIDATION_SEEDS:
        directory = root / f"runs/ppo_phase_v1/validation-checkpoint-evaluation/worker-{seed}"
        _write_json(
            directory / "checkpoint_evaluation.json",
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_record["sha256"],
                "checkpoint_infos": {"global_policy_decisions": step},
            },
        )
        workers.append(directory)
    captures = {}
    for role in ("baseline", "candidate"):
        aggregate = root / f"runs/ppo_phase_v1/{role}-batch/aggregate.json"
        _write_json(aggregate, {"role": role, "seeds": list(orchestration.VALIDATION_SEEDS)})
        binding = {
            **_record(aggregate),
            "checkpoint_manifest_path": str(checkpoint_manifest) if role == "candidate" else None,
            "checkpoint_manifest_sha256": manifest_record["sha256"] if role == "candidate" else None,
            "source_file_records": [
                _record(aggregate),
                *(
                    [_record(directory / "checkpoint_evaluation.json") for directory in workers]
                    if role == "candidate"
                    else []
                ),
            ],
            "worker_run_dirs": [str(directory) for directory in workers] if role == "candidate" else [],
        }
        captures[role] = binding

    def capture_aggregate(path, *, role, project_root, **kwargs):
        # Aggregate reconstruction has its own end-to-end tests.  Keep its
        # boundary deterministic here; decision/path/source hash checks below
        # and every candidate worker's checkpoint/global-step check stay real.
        assert project_root == root
        assert Path(path) == Path(captures[role]["path"])
        if role == "candidate":
            assert kwargs["expected_checkpoint_path"] == checkpoint
            assert Path(kwargs["expected_checkpoint_manifest_path"]) == checkpoint_manifest
        return SimpleNamespace(as_record=lambda: copy.deepcopy(captures[role]))

    monkeypatch.setattr(
        paired_aggregate_binding, "capture_validation_aggregate", capture_aggregate
    )
    checks = {name: True for name in REQUIRED_PROMOTION_GATES}
    payload = {
        "schema": orchestration.PROMOTION_DECISION_SCHEMA,
        "baseline_checkpoint": "pure_fsm",
        "paired_seeds": list(orchestration.VALIDATION_SEEDS),
        "paired_episode_count": 5,
        "minimum_paired_seeds": 5,
        "frozen_hashes_unchanged": True,
        "candidate_checkpoint_path": str(checkpoint),
        "candidate_checkpoint_sha256": checkpoint_record["sha256"],
        "baseline_evaluation_aggregate": copy.deepcopy(captures["baseline"]),
        "candidate_validation_aggregate": copy.deepcopy(captures["candidate"]),
        "promotion": {
            "promoted": True,
            "checks": checks,
            "first_failed_gate": None,
            "global_stability_improvement_fraction": 0.06,
            "improved_priority_phase_count": 4,
        },
        "checks_in_evaluation_order": [
            {"gate": name, "passed": True} for name in REQUIRED_PROMOTION_GATES
        ],
        "first_failed_gate": None,
    }
    chunks = [{
        "immutable_history_checkpoint": checkpoint_record,
        "checkpoint_manifest": manifest_record,
        "global_policy_decisions": step,
    }]
    return SimpleNamespace(
        root=root, output_root=output_root, step=step, payload=payload,
        chunks=chunks, checkpoint=checkpoint, checkpoint_manifest=checkpoint_manifest,
        workers=workers, captures=captures,
    )


def test_real_cadence_path_is_consumable_and_final_delivery_disjoint(decision_sources):
    data = decision_sources
    directory = _producer_directory(data.root, data.step)
    decision = directory / "promotion_decision.json"
    _write_json(decision, data.payload)
    accepted = orchestration._validate_promotion_decision(
        decision, chunks=data.chunks, project_root=data.root, cache={}
    )
    assert accepted["bound_global_policy_decisions"] == data.step
    assert accepted["record"] == _record(decision)
    assert finalization._file_record(decision, root=data.output_root) == _record(decision)
    final_metrics = data.output_root / "metrics"
    assert not evaluation_artifacts._paths_overlap(final_metrics, directory)
    sources = SimpleNamespace(
        aggregate_path=Path(data.captures["candidate"]["path"]),
        worker_run_dirs=tuple(data.workers),
        canonical_episode_dirs=(),
        checkpoint_path=data.checkpoint,
        checkpoint_manifest_path=data.checkpoint_manifest,
        supporting_files=(_record(decision),),
    )
    assert evaluation_artifacts._require_output_disjoint_from_final_sources(
        final_metrics, {"candidate": sources}
    ) == final_metrics
    with pytest.raises(evaluation_artifacts.EvaluationArtifactError, match="overlaps an input source"):
        evaluation_artifacts._require_output_disjoint_from_final_sources(
            directory, {"candidate": sources}
        )


@pytest.mark.parametrize("tamper", ("aggregate_binding", "source_bytes", "worker_global_step"))
def test_canonical_history_does_not_bypass_source_binding(decision_sources, tamper):
    data = decision_sources
    decision = data.output_root / f"validation_history/step_{data.step}/promotion_decision.json"
    if tamper == "aggregate_binding":
        data.payload["candidate_validation_aggregate"]["sha256"] = "0" * 64
    elif tamper == "source_bytes":
        Path(data.captures["candidate"]["path"]).write_bytes(b"changed after capture")
    else:
        worker = data.workers[0] / "checkpoint_evaluation.json"
        changed = json.loads(worker.read_text())
        changed["checkpoint_infos"]["global_policy_decisions"] += 1
        _write_json(worker, changed)
        data.captures["candidate"]["source_file_records"][1] = _record(worker)
        data.payload["candidate_validation_aggregate"] = copy.deepcopy(data.captures["candidate"])
    _write_json(decision, data.payload)
    with pytest.raises(orchestration.TrainingOrchestrationError):
        orchestration._validate_promotion_decision(
            decision, chunks=data.chunks, project_root=data.root, cache={}
        )


@pytest.mark.parametrize("relative", (
    "outputs/ppo_phase_v1/metrics/promotion_decision.json",
    "outputs/ppo_phase_v1/validation_history/promotion_decision.json",
    "outputs/ppo_phase_v1/validation_history/step_0121600/promotion_decision.json",
    "outputs/ppo_phase_v1/validation_history/step_121601/promotion_decision.json",
    "outputs/elsewhere/validation_history/step_121600/promotion_decision.json",
))
def test_noncanonical_output_history_and_wrong_step_are_rejected(decision_sources, relative):
    data = decision_sources
    decision = data.root / relative
    _write_json(decision, data.payload)
    with pytest.raises(orchestration.TrainingOrchestrationError):
        orchestration._validate_promotion_decision(
            decision, chunks=data.chunks, project_root=data.root, cache={}
        )


def test_canonical_history_cannot_escape_through_directory_link(decision_sources):
    data = decision_sources
    escaped = data.root / "outside_delivery"
    _write_json(escaped / "promotion_decision.json", data.payload)
    link = data.output_root / f"validation_history/step_{data.step}"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(escaped, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("Directory symlinks are unavailable")
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(escaped)],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if result.returncode != 0:
            pytest.skip("Neither directory symlinks nor junctions are available")
    decision = link / "promotion_decision.json"
    with pytest.raises(orchestration.TrainingOrchestrationError):
        orchestration._validate_promotion_decision(
            decision, chunks=data.chunks, project_root=data.root, cache={}
        )
    with pytest.raises(finalization.FinalizationError):
        finalization._file_record(decision, root=data.output_root)
