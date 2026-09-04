from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import artifacts, cli
from wlr50_clean.ppo import phase_effective_entry_holdout as holdout
from wlr50_clean.ppo import phase_snapshot_live_probe as probe
from wlr50_clean.ppo.phase_effective_entry import (
    DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
    validate_effective_phase_entry_comparison,
)
from wlr50_clean.ppo.phase_snapshots import (
    DEFAULT_PHASE_SNAPSHOT_ROOT,
    capture_validated_phase_snapshot_bundle,
    load_validated_phase_snapshot_payload,
)
from wlr50_clean.ppo.vector_benchmark_matrix import (
    VectorBenchmarkMatrixError,
    validate_managed_run_directory,
)


def test_default_holdout_config_set_includes_transitive_runtime_inputs() -> None:
    paths = set(holdout.HOLDOUT_CONFIG_RELATIVE_PATHS)
    assert {
        "configs/ppo_action_projection.yaml",
        "configs/ppo_observation_schema.json",
        "configs/conformance_policy.yaml",
    } <= paths


@pytest.fixture(scope="module")
def holdout_snapshot_bundle():
    return capture_validated_phase_snapshot_bundle(DEFAULT_PHASE_SNAPSHOT_ROOT)


@pytest.mark.parametrize("phase", ("P02", "P10"))
@pytest.mark.parametrize("tamper", (None, "replay_tick", "controller_proof"))
def test_worker_import_revalidates_real_attempt_api_with_pinned_replay_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    holdout_snapshot_bundle,
    phase: str,
    tamper: str | None,
) -> None:
    # Reuse evidence construction only.  Neither the live attempt validator,
    # the snapshot loader, nor the replay-window validator is monkeypatched.
    from test_phase_effective_entry import _calibration_attempt, _comparison

    payload = json.loads(DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_bytes())
    contract_record = {"contract_sha256": payload["contract_sha256"]}
    contract = SimpleNamespace(
        contract_sha256=payload["contract_sha256"],
        entry=lambda selected: copy.deepcopy(payload["phases"][selected]),
        as_record=lambda: copy.deepcopy(contract_record),
    )
    snapshot_payload, snapshot_entry = load_validated_phase_snapshot_payload(
        holdout_snapshot_bundle, phase
    )
    replay_window = probe._validated_replay_window(
        snapshot_payload, snapshot_entry, phase=phase
    )
    if phase == "P02":
        assert replay_window.controller_restore_contract is None
        assert replay_window.controller_restore_mode is None
    else:
        assert replay_window.controller_restore_contract is not None
        assert (
            replay_window.controller_restore_mode
            == "source_proven_execute_after_prime"
        )
    attempts = []
    for lifecycle in ("fresh_scene", "reused_scene"):
        row = _calibration_attempt(
            contract, holdout_snapshot_bundle, phase, lifecycle
        )
        calibration_proof = row["snapshot_state_write"]["effective_entry_contract"]
        acceptance_proof = dict(
            validate_effective_phase_entry_comparison(
                contract, phase, _comparison(contract.entry(phase))
            )
        )
        # Backend acceptance appends the same pinned replay-anchor fields to
        # the calibrated physical-state comparison returned above.
        row["snapshot_state_write"]["effective_entry_contract"] = {
            **{
                key: value
                for key, value in calibration_proof.items()
                if key in probe._ACCEPTANCE_EFFECTIVE_ENTRY_FIELDS
            },
            **acceptance_proof,
        }
        assert probe._attempt_passed(row, replay_window=replay_window) is True
        attempts.append(row)
    if tamper == "replay_tick":
        attempts[1]["target_entry_tick"] += 1
    elif tamper == "controller_proof":
        guard = attempts[1]["snapshot_state_write"]["entry_guard_contract"]
        if phase == "P10":
            guard["source_transition_row_canonical_sha256"] = "0" * 64
        else:
            guard["verified"] = False
    if tamper is not None:
        assert (
            probe._attempt_passed(attempts[1], replay_window=replay_window)
            is False
        )

    run_dir = tmp_path / phase
    run_dir.mkdir()
    for filename in (
        "committed_runtime_identity.before.json",
        "committed_runtime_identity.after.json",
        "frozen_hashes.before.json",
        "frozen_hashes.after.json",
        "run_manifest.json",
        "stderr.log",
    ):
        (run_dir / filename).write_text("{}\n", encoding="utf-8")

    def record(filename: str) -> dict[str, object]:
        return holdout._snapshot(
            run_dir / filename, label="test worker file", cache={}
        ).record()

    runtime_before = record("committed_runtime_identity.before.json")
    runtime_after = record("committed_runtime_identity.after.json")
    report = {
        "schema": holdout.PROBE_SCHEMA,
        "artifact_role": "DIAGNOSTIC_ONLY_NOT_TRAINING_ACCEPTANCE",
        "status": "PASSED",
        "passed": True,
        "complete": True,
        "seed": holdout.HOLDOUT_SEED,
        "phases": [phase],
        "phase_count": 1,
        "phase_selector_mode": "single_phase",
        "attempts_per_phase": 2,
        "expected_attempt_count": 2,
        "completed_attempt_count": 2,
        "expected_fresh_scene_attempt_count": 1,
        "expected_reused_scene_attempt_count": 1,
        "fresh_scene_attempt_count": 1,
        "reused_scene_attempt_count": 1,
        "failure_reasons": [],
        "failure_classification": None,
        "probe_process_id": 123,
        "probe_process_instance_id": "1" * 32,
        "attempts": attempts,
        "phase_snapshot_bundle": holdout_snapshot_bundle.as_record(),
        "phase_effective_entry_contract": contract.as_record(),
        "runtime_identity_before": runtime_before,
        "frozen_hashes_before": record("frozen_hashes.before.json"),
        "managed_post_checks": {
            "runtime_identity_after": str(
                run_dir / "committed_runtime_identity.after.json"
            ),
            "frozen_hashes_after": str(run_dir / "frozen_hashes.after.json"),
            "sealed_by_run_manifest": str(run_dir / "run_manifest.json"),
        },
    }
    (run_dir / holdout.PROBE_FILENAME).write_text(
        json.dumps(report) + "\n", encoding="utf-8"
    )
    (run_dir / "stdout.log").write_text(
        json.dumps(report) + "\n", encoding="utf-8"
    )
    (run_dir / "live_command_result.json").write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.live_command_result.v1",
                "command": "phase-snapshot-live-probe",
                "exit_code": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "invocation_arguments": [
            "--phase", phase, "--seed", "1003", "--num-envs", "1",
            "--phase-snapshot-prime-physics-steps", "1", "--seed-set", "train",
            "--deterministic",
        ],
        "artifacts": {
            filename: {**record(filename), "path": filename}
            for filename in (holdout.PROBE_FILENAME, "live_command_result.json")
        },
        "logs": {
            filename: {**record(filename), "path": filename}
            for filename in ("stdout.log", "stderr.log")
        },
    }
    context = SimpleNamespace(
        project_root=tmp_path,
        git_commit="a" * 40,
        backend_sha256="b" * 64,
        config_sha256="c" * 64,
        frozen_manifest_sha256="d" * 64,
        snapshot_bundle=holdout_snapshot_bundle,
        effective_entry_contract=contract,
    )
    # Isolate managed-launcher identity checks, which have separate coverage;
    # all file hashes, stdout binding, and physical-attempt checks stay real.
    monkeypatch.setattr(
        holdout, "validate_managed_run_directory", lambda value, **kwargs: Path(value)
    )
    monkeypatch.setattr(
        holdout, "_validate_started_and_final_manifest",
        lambda *args, **kwargs: (manifest, manifest),
    )
    monkeypatch.setattr(
        holdout, "_validate_runtime_and_frozen_pair",
        lambda *args, **kwargs: (runtime_before, runtime_after, "e" * 64),
    )
    if tamper is None:
        worker = holdout._validate_probe_worker(run_dir, context=context, cache={})
        assert worker["phase"] == phase
        assert worker["fresh_attempt_passed"] is True
        assert worker["reused_attempt_passed"] is True
    else:
        with pytest.raises(
            holdout.PhaseEffectiveEntryHoldoutError,
            match=f"attempt proof is invalid: {phase}\\[reused_repeat\\]",
        ):
            holdout._validate_probe_worker(run_dir, context=context, cache={})


def _worker(phase: str, index: int) -> dict[str, object]:
    return {
        "phase": phase,
        "run_dir": f"C:/runs/{phase}",
        "probe_process_instance_id": f"{index:032x}",
        "seed": 1003,
        "source_git_commit": "a" * 40,
        "committed_runtime_content_sha256": "b" * 64,
        "backend_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "frozen_manifest_sha256": "e" * 64,
        "phase_snapshot_bundle_sha256": "f" * 64,
        "phase_effective_entry_contract_sha256": "1" * 64,
    }


def test_holdout_worker_set_requires_exact_phase_and_process_partition() -> None:
    rows = [_worker(phase, index) for index, phase in enumerate(holdout.HOLDOUT_PHASES, 1)]
    holdout._validate_worker_set(rows)

    duplicate = [dict(row) for row in rows]
    duplicate[-1]["probe_process_instance_id"] = duplicate[0][
        "probe_process_instance_id"
    ]
    with pytest.raises(
        holdout.PhaseEffectiveEntryHoldoutError,
        match="distinct managed process instances",
    ):
        holdout._validate_worker_set(duplicate)


def test_phase_curriculum_gate_is_mandatory_but_other_stages_remain_provisional() -> None:
    with pytest.raises(cli.CliError, match="twelve independent seed-1003"):
        cli._require_training_phase_effective_entry_holdout(
            SimpleNamespace(stage="phase-curriculum")
        )
    assert (
        cli._require_training_phase_effective_entry_holdout(
            SimpleNamespace(stage="smoke")
        )
        is None
    )


def test_checkpoint_fields_bind_acceptance_and_aggregation_manifest(
    tmp_path: Path,
) -> None:
    acceptance_path = tmp_path / holdout.OUTPUT_FILENAME
    acceptance = {"status": "PASSED", "passed": True, "workers": []}
    acceptance_path.write_text(json.dumps(acceptance) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {
        "path": str(acceptance_path),
        "sha256": digest(acceptance_path),
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": digest(manifest_path),
        "phase_effective_entry_contract_sha256": "1" * 64,
        "source_git_commit": "2" * 40,
        "acceptance": acceptance,
        "passed": True,
    }

    fields, files = cli._phase_effective_entry_holdout_fields(evidence)

    assert fields["phase_effective_entry_holdout_acceptance_path"] == str(
        acceptance_path.resolve()
    )
    assert fields["phase_effective_entry_holdout_acceptance_sha256"] == digest(
        acceptance_path
    )
    assert fields["phase_effective_entry_holdout_contract_sha256"] == "1" * 64
    assert fields["phase_effective_entry_holdout_source_git_commit"] == "2" * 40
    assert fields["phase_effective_entry_holdout_acceptance"] == acceptance
    assert fields["phase_effective_entry_holdout_files"] == files


def test_training_resume_inherits_only_exact_embedded_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance_path = tmp_path / holdout.OUTPUT_FILENAME
    acceptance_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    acceptance = {"status": "PASSED", "passed": True}
    evidence = {
        "path": str(acceptance_path.resolve()),
        "sha256": digest(acceptance_path),
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": digest(manifest_path),
        "phase_effective_entry_contract_sha256": "1" * 64,
        "source_git_commit": "2" * 40,
        "acceptance": acceptance,
        "passed": True,
    }
    fields, files = cli._phase_effective_entry_holdout_fields(evidence)
    infos = {**fields, "files": dict(files)}
    monkeypatch.setattr(
        holdout,
        "validate_phase_effective_entry_holdout_acceptance",
        lambda *args, **kwargs: dict(evidence),
    )
    monkeypatch.setattr(cli, "_checkpoint_config_paths", lambda args: ())
    args = SimpleNamespace(stage="full-episode")

    inherited = cli._inherit_training_phase_effective_entry_holdout(
        args, infos, object(), object()
    )

    assert inherited == evidence
    assert args._phase_effective_entry_holdout_evidence == evidence
    assert args.phase_effective_entry_holdout_acceptance == acceptance_path.resolve()

    phase_args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_effective_entry_holdout_evidence=dict(evidence),
    )
    phase_inherited = cli._inherit_training_phase_effective_entry_holdout(
        phase_args,
        {"stage": "phase-curriculum", **infos},
        object(),
        object(),
    )
    assert phase_inherited == evidence

    tampered = dict(infos)
    tampered["phase_effective_entry_holdout_acceptance_sha256"] = "f" * 64
    with pytest.raises(cli.CliError, match="binding differs"):
        cli._inherit_training_phase_effective_entry_holdout(
            SimpleNamespace(stage="full-episode"), tampered, object(), object()
        )


def test_later_training_stage_rejects_resume_without_holdout() -> None:
    with pytest.raises(cli.CliError, match="omits phase effective-entry"):
        cli._inherit_training_phase_effective_entry_holdout(
            SimpleNamespace(stage="full-episode"), {}, object(), object()
        )


def test_phase_first_entry_keeps_explicit_holdout_when_smoke_has_none() -> None:
    explicit = {"path": "C:\\evidence\\phase_effective_entry_holdout.json"}
    args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_effective_entry_holdout_evidence=explicit,
    )
    assert (
        cli._inherit_training_phase_effective_entry_holdout(
            args, {"stage": "smoke"}, object(), object()
        )
        == explicit
    )


def test_phase_to_phase_resume_cannot_fill_missing_holdout_from_explicit() -> None:
    explicit = {"path": "C:\\evidence\\phase_effective_entry_holdout.json"}
    args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_effective_entry_holdout_evidence=explicit,
    )

    with pytest.raises(
        cli.CliError, match="omits phase effective-entry holdout ancestry"
    ):
        cli._inherit_training_phase_effective_entry_holdout(
            args,
            {"stage": "phase-curriculum"},
            object(),
            object(),
        )


def test_holdout_script_locks_seed_phases_and_separate_probe_invocations() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "aggregate_phase_effective_entry_holdout.ps1").read_text(
        encoding="utf-8"
    )
    assert "[ValidateSet(1003)]" in text
    assert '"P02", "P03"' in text
    assert '"P12", "P13"' in text
    assert "run_phase_snapshot_live_probe.ps1" in text
    assert '-RunKind "phase_effective_entry_holdout"' in text
    assert '-CliModule "wlr50_clean.ppo.phase_effective_entry_holdout"' in text
    assert '"--probe-run-dir"' in text


@pytest.mark.parametrize(
    ("requested_run_kind", "canonical_run_kind", "training_stage", "subcommand"),
    (
        (
            "phase_snapshot_live_probe",
            holdout.PROBE_RUN_KIND,
            "phase-snapshot-live-probe",
            "phase-snapshot-live-probe",
        ),
        (
            "phase_effective_entry_holdout",
            holdout.HOLDOUT_RUN_KIND,
            "effective-entry-holdout-aggregation",
            "aggregate",
        ),
    ),
)
def test_holdout_importer_accepts_real_canonical_managed_manifest_and_rejects_raw_kind(
    tmp_path: Path,
    requested_run_kind: str,
    canonical_run_kind: str,
    training_stage: str,
    subcommand: str,
) -> None:
    config = tmp_path / "configs" / "holdout.yaml"
    config.parent.mkdir()
    config.write_text("holdout: true\n", encoding="utf-8")
    config_sha256, config_records = artifacts.config_set_record(
        (config,), project_root=tmp_path
    )
    reservation = artifacts.reserve_run(
        project_root=tmp_path,
        run_kind=requested_run_kind,
        config_paths=(config,),
        seed=holdout.HOLDOUT_SEED,
        environment_count=1,
        training_stage=training_stage,
        git_commit="a" * 40,
        entrypoint=(
            "wlr50_clean.ppo.phase_effective_entry_holdout"
            if subcommand == "aggregate"
            else "wlr50_clean.ppo.cli"
        ),
        subcommand=subcommand,
    )
    artifacts.finalize_run(reservation.run_dir, exit_code=0)
    context = SimpleNamespace(
        project_root=tmp_path.resolve(),
        git_commit="a" * 40,
        config_sha256=config_sha256,
        config_records=tuple(config_records),
    )

    assert reservation.run_dir.parent.name == canonical_run_kind
    assert validate_managed_run_directory(
        reservation.run_dir,
        project_root=tmp_path,
        run_kind=canonical_run_kind,
    ) == reservation.run_dir
    final, started = holdout._validate_started_and_final_manifest(
        reservation.run_dir,
        run_kind=canonical_run_kind,
        entrypoint=(
            "wlr50_clean.ppo.phase_effective_entry_holdout"
            if subcommand == "aggregate"
            else "wlr50_clean.ppo.cli"
        ),
        subcommand=subcommand,
        training_stage=training_stage,
        context=context,
        cache={},
    )
    assert final["run_kind"] == canonical_run_kind
    assert started["run_kind"] == canonical_run_kind

    with pytest.raises(VectorBenchmarkMatrixError, match="managed run must be"):
        validate_managed_run_directory(
            reservation.run_dir,
            project_root=tmp_path,
            run_kind=requested_run_kind,
        )


def test_holdout_script_preserves_single_worker_output_as_one_full_path(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    root = Path(__file__).resolve().parents[2]
    source = root / "scripts" / "aggregate_phase_effective_entry_holdout.ps1"
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    (tmp_path / "run_phase_snapshot_live_probe.ps1").write_text(
        "param([int]$Seed,[int]$PrimePhysicsSteps,[string]$Phase)\n"
        "\"C:\\runs\\$Phase\"\n",
        encoding="utf-8",
    )
    (tmp_path / "_invoke_ppo_cli.ps1").write_text(
        "param([string]$RunKind,[string]$TrainingStage,[string]$Subcommand,"
        "[string]$CliModule,[string[]]$ConfigPath,[int]$Seed,"
        "[int]$EnvironmentCount,[string[]]$BaseCliArgs,[string[]]$CliArgs)\n"
        "$BaseCliArgs | ConvertTo-Json -Compress\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(copied)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = json.loads(completed.stdout.strip())
    assert arguments[::2] == ["--probe-run-dir"] * 12
    assert arguments[1::2] == [
        f"C:\\runs\\P{index:02d}" for index in range(2, 14)
    ]


def test_train_script_requires_external_holdout_only_for_phase_curriculum() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "train_phase_residual_ppo.ps1").read_text(
        encoding="utf-8"
    )
    assert "[string]$PhaseEffectiveEntryHoldoutAcceptance" in text
    assert '$Stage -eq "phase-curriculum"' in text
    assert '"--phase-effective-entry-holdout-acceptance"' in text
    assert "aggregate_phase_effective_entry_holdout.ps1" in text
