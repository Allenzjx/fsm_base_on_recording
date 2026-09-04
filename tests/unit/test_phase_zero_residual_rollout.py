from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import artifacts, cli
from wlr50_clean.ppo import phase_effective_entry_holdout as holdout_module
from wlr50_clean.ppo import phase_zero_residual_rollout as rollout_module
from wlr50_clean.ppo import vector_benchmark_matrix as matrix_module
from wlr50_clean.ppo.action_projection import SafetyProjection
from wlr50_clean.ppo.phase_zero_residual_rollout import (
    ARTIFACT_FILENAME,
    ARTIFACT_SCHEMA,
    MAX_DECISIONS_PER_PHASE,
    PHASE_IDS,
    PhaseZeroResidualRolloutError,
    TRAINING_EVIDENCE_SCHEMA,
    ZERO_FULL12,
    _strict_json_object,
    run_phase_zero_residual_rollout,
    validate_phase_zero_residual_rollout_evidence,
    validate_phase_zero_residual_rollout_payload,
)
from wlr50_clean.ppo.termination import TerminationSignals


def _binding() -> dict[str, str]:
    return {
        "phase_snapshot_bundle_sha256": "1" * 64,
        "phase_snapshot_manifest_sha256": "2" * 64,
        "phase_effective_entry_contract_sha256": "3" * 64,
        "phase_effective_entry_contract_file_sha256": "4" * 64,
        "phase_effective_entry_contract_sidecar_sha256": "5" * 64,
        "holdout_acceptance_path": "C:\\evidence\\holdout.json",
        "holdout_acceptance_sha256": "6" * 64,
        "holdout_run_manifest_path": "C:\\evidence\\run_manifest.json",
        "holdout_run_manifest_sha256": "7" * 64,
        "source_git_commit": "8" * 40,
    }


def _frame(
    phase_id: str,
    tick: int,
    *,
    terminal: TerminationSignals | None = None,
    safety: SafetyProjection | None = None,
    root_pose_writes: int = 0,
    lifecycle: str = "EXECUTE_MOTION",
    controller_task_result: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        state_id=phase_id,
        physics_tick=tick,
        sim_time_s=tick / 120.0,
        nominal_action_full12=ZERO_FULL12,
        termination_signals=terminal or TerminationSignals(),
        safety_projection=safety or SafetyProjection(),
        info={
            "controller_lifecycle": lifecycle,
            "controller_task_result": controller_task_result,
            "controller_termination": None,
            "termination_mapping": {
                "controller_result": None,
                "controller_reason": None,
                "controller_details": {},
                "first_blocker": {},
                "active_sources": (),
                "primary_source": None,
                "controller_blocked_encoded_as_truncation": False,
            },
            "raw_controller_frame": SimpleNamespace(
                termination=None, first_blocker={}
            ),
            "in_episode_root_pose_writes": root_pose_writes,
            "in_episode_root_velocity_writes": 0,
            "recording_accesses": 0,
            "training_phase_snapshot": phase_id,
            "phase_snapshot_restoration": {
                "requested_phase": phase_id,
                "snapshot_validated": True,
                "mode": (
                    "normal_p01_reset"
                    if phase_id == "P01"
                    else "phase_snapshot_prime_without_rewind"
                ),
            },
            "reset_count": int(phase_id[1:]),
            "reset_prime_tick_count": 0 if phase_id == "P01" else 1,
        },
    )


class _FakeEpisode:
    def __init__(self, *, fault: tuple[str, str] | None = None) -> None:
        self.frame = None
        self.tick_callback = None
        self.decision_count = 0
        self.reset_calls: list[str] = []
        self.step_calls = 0
        self.fault = fault

    def reset(self, *, seed: int, options: dict[str, object]):
        assert seed == 1004
        phase_id = str(options["training_phase_snapshot"])
        self.reset_calls.append(phase_id)
        self.decision_count = 0
        self.frame = _frame(phase_id, 0)
        return (0.0,), dict(self.frame.info)

    def _after_frame(self, before: SimpleNamespace, state_id: str) -> SimpleNamespace:
        phase_fault = self.fault if self.fault and self.fault[0] == before.state_id else None
        kwargs = {
            "lifecycle": (
                "WAIT_ENTRY"
                if state_id != before.state_id
                else "VERIFY_RESULT"
            )
        }
        if phase_fault is not None:
            kind = phase_fault[1]
            if kind == "root-write":
                kwargs["root_pose_writes"] = 1
            elif kind == "terminal":
                kwargs["terminal"] = TerminationSignals(fall=True)
            elif kind == "safety":
                kwargs["safety"] = SafetyProjection(
                    residual_enabled=False,
                    force_wheels_zero=True,
                    reason="FALL",
                )
            elif kind == "controller":
                kwargs["controller_task_result"] = "INCOMPLETE_CONTROLLER_BLOCKED"
            elif kind == "lifecycle":
                kwargs["lifecycle"] = "DONE"
        result = _frame(state_id, before.physics_tick + 1, **kwargs)
        result.info["raw_controller_frame"].events = (
            ()
            if state_id == before.state_id
            else (
                SimpleNamespace(
                    state_id=state_id,
                    from_lifecycle="DONE",
                    to_lifecycle="WAIT_ENTRY",
                    reason=f"advance fixed graph {before.state_id}->{state_id}",
                ),
            )
        )
        return result

    def step(self, action, *, stop_after_phase_id: str):
        assert tuple(action) == ZERO_FULL12
        assert self.frame is not None
        assert stop_after_phase_id == self.frame.state_id
        self.step_calls += 1
        boundary = False
        for local_tick in range(8):
            before = self.frame
            phase_index = PHASE_IDS.index(stop_after_phase_id)
            state_id = stop_after_phase_id
            if phase_index < len(PHASE_IDS) - 1 and local_tick == 2:
                state_id = PHASE_IDS[phase_index + 1]
                boundary = True
            after = self._after_frame(before, state_id)
            projection = SimpleNamespace(
                raw_residual_full12=ZERO_FULL12,
                safe_projected_residual_full12=ZERO_FULL12,
                applied_action_full12=before.nominal_action_full12,
                zero_residual_fast_path=True,
                hard_safety_modified=False,
            )
            if self.tick_callback is not None:
                self.tick_callback(before, after, projection)
            self.frame = after
            if boundary:
                break
        self.decision_count += 1
        return SimpleNamespace(
            terminated=False,
            truncated=boundary,
            info={"phase_curriculum_boundary": boundary},
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_identity(project_root: Path, run_dir: Path, git_commit: str) -> None:
    relative = "src/wlr50_clean/ppo/isaac_fsm_backend.py"
    source = project_root / relative
    data = source.read_bytes()
    rows = [
        {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "creation_time_utc_ticks": 1,
            "last_write_time_utc_ticks": 2,
        }
    ]
    encoded = json.dumps(
        rows, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    content_encoded = json.dumps(
        [
            {
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload = {
        "schema": "wlr50_clean.committed_runtime_identity.v1",
        "git_commit": git_commit,
        "file_count": len(rows),
        "content_sha256": hashlib.sha256(content_encoded).hexdigest(),
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }
    _write_json(run_dir / "committed_runtime_identity.before.json", payload)
    _write_json(run_dir / "committed_runtime_identity.after.json", payload)


def _managed_rollout_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    project_root = tmp_path / "project"
    training_config = project_root / "configs" / "ppo_training_phase_v1.yaml"
    interface_config = project_root / "configs" / "ppo_interface_v2.yaml"
    training_config.parent.mkdir(parents=True)
    training_config.write_text("training: fixture\n", encoding="utf-8")
    interface_config.write_text("interface: fixture\n", encoding="utf-8")
    config_sha256, config_records = artifacts.config_set_record(
        (training_config, interface_config), project_root=project_root
    )

    backend = project_root / "src" / "wlr50_clean" / "ppo" / "isaac_fsm_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("# committed backend fixture\n", encoding="utf-8")
    backend_sha256 = _sha256(backend)

    snapshot_root = project_root / "reference" / "ppo_phase_snapshots"
    snapshot_root.mkdir(parents=True)
    frozen_path = (
        project_root
        / "artifacts"
        / "ppo_phase_v1_start"
        / "frozen_fsm_hashes.json"
    )
    protected = {
        f"protected/frozen_{index:02d}.py": hashlib.sha256(
            f"frozen-{index}".encode("ascii")
        ).hexdigest()
        for index in range(29)
    }
    frozen_manifest = {
        "algorithm": "sha256",
        "source_head": "7" * 40,
        "protected_files": protected,
    }
    _write_json(frozen_path, frozen_manifest)

    holdout_run = (
        project_root
        / "runs"
        / "ppo_phase_v1"
        / "phase_effective_entry_holdout"
        / "holdout-fixture"
    )
    holdout_run.mkdir(parents=True)
    holdout_path = holdout_run / holdout_module.OUTPUT_FILENAME
    holdout_manifest_path = holdout_run / "run_manifest.json"
    _write_json(holdout_path, {"fixture": "validated upstream holdout"})
    _write_json(holdout_manifest_path, {"fixture": "validated upstream manifest"})

    binding = {
        **_binding(),
        "holdout_acceptance_path": str(holdout_path.resolve()),
        "holdout_acceptance_sha256": _sha256(holdout_path),
        "holdout_run_manifest_path": str(holdout_manifest_path.resolve()),
        "holdout_run_manifest_sha256": _sha256(holdout_manifest_path),
        "source_git_commit": "8" * 40,
    }
    context = SimpleNamespace(
        project_root=project_root.resolve(),
        git_commit=binding["source_git_commit"],
        config_sha256=config_sha256,
        config_records=tuple(config_records),
        frozen_manifest_path=frozen_path.resolve(),
        frozen_manifest=frozen_manifest,
        frozen_manifest_sha256=_sha256(frozen_path),
        backend_path=backend.resolve(),
        backend_sha256=backend_sha256,
        snapshot_bundle=SimpleNamespace(
            snapshot_root=snapshot_root.resolve(),
            bundle_sha256=binding["phase_snapshot_bundle_sha256"],
            manifest_sha256=binding["phase_snapshot_manifest_sha256"],
        ),
        effective_entry_contract=SimpleNamespace(
            contract_sha256=binding["phase_effective_entry_contract_sha256"],
            file_sha256=binding["phase_effective_entry_contract_file_sha256"],
            sidecar_file_sha256=binding[
                "phase_effective_entry_contract_sidecar_sha256"
            ],
        ),
    )
    monkeypatch.setattr(
        holdout_module, "_current_context", lambda *_args, **_kwargs: context
    )
    monkeypatch.setattr(
        matrix_module,
        "_committed_runtime_paths",
        lambda *_args: ("src/wlr50_clean/ppo/isaac_fsm_backend.py",),
    )

    identity = artifacts.RunIdentity(
        timestamp_utc="2026-09-04T15:00:00.000000Z",
        git_commit=str(binding["source_git_commit"]),
        config_sha256=config_sha256,
        seed=1004,
        environment_count=1,
        training_stage="phase-zero-residual-rollout",
    )
    run_dir = (
        project_root
        / "runs"
        / "ppo_phase_v1"
        / "phase_zero_residual_rollout"
        / identity.run_id
    )
    run_dir.mkdir(parents=True)
    payload = run_phase_zero_residual_rollout(
        _FakeEpisode(), seed=1004, contract_binding=binding
    )
    evidence_path = run_dir / ARTIFACT_FILENAME
    live_path = run_dir / "live_command_result.json"
    _write_json(evidence_path, payload)
    _write_json(
        live_path,
        {
            "schema": "wlr50_clean.live_command_result.v1",
            "command": "phase-zero-residual-rollout",
            "exit_code": 0,
        },
    )
    _runtime_identity(project_root, run_dir, str(binding["source_git_commit"]))

    audit_common = {
        "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
        "project_root": str(project_root.resolve()),
        "frozen_manifest": str(frozen_path.resolve()),
        "frozen_manifest_sha256": _sha256(frozen_path),
        "source_head": frozen_manifest["source_head"],
        "protected_file_count": len(protected),
        "entries": [
            {
                "path": name,
                "expected_sha256": digest,
                "actual_sha256": digest,
                "exists": True,
                "valid": True,
            }
            for name, digest in protected.items()
        ],
        "mismatches": [],
        "passed": True,
    }
    before_audit = {**audit_common, "checked_at_utc": "2026-09-04T15:00:01Z"}
    after_audit = {**audit_common, "checked_at_utc": "2026-09-04T15:01:01Z"}
    _write_json(run_dir / "frozen_hashes.before.json", before_audit)
    _write_json(run_dir / "frozen_hashes.after.json", after_audit)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text(
        "\n".join(
            json.dumps(value, separators=(",", ":"), allow_nan=False)
            for value in (before_audit, payload, after_audit)
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    invocation = [
        "--training-config",
        "configs/ppo_training_phase_v1.yaml",
        "--interface-config",
        "configs/ppo_interface_v2.yaml",
        "--snapshot-root",
        "reference/ppo_phase_snapshots",
        "--phase-snapshot-prime-physics-steps",
        "1",
        "--phase-effective-entry-holdout-acceptance",
        str(holdout_path.resolve()),
        "--episode-count",
        "13",
        "--policy-decisions",
        "64",
        "--seed-set",
        "train",
        "--residual-mode",
        "zero",
        "--deterministic",
        "--run-dir",
        "<reserved-immutable-run-dir>",
        "--seed",
        "1004",
        "--num-envs",
        "1",
    ]
    identity_record = {
        "timestamp_utc": identity.timestamp_utc,
        "git_commit": identity.git_commit,
        "config_sha256": identity.config_sha256,
        "seed": identity.seed,
        "environment_count": identity.environment_count,
        "training_stage": identity.training_stage,
    }
    started = {
        "schema": "wlr50_clean.ppo_run_manifest.v1",
        "lifecycle": "STARTED",
        "immutable_run_directory": True,
        "run_id": identity.run_id,
        "run_kind": "phase_zero_residual_rollout",
        "run_dir": str(run_dir.resolve()),
        "project_root": str(project_root.resolve()),
        "identity": identity_record,
        "configs": config_records,
        "entrypoint": "wlr50_clean.ppo.cli",
        "subcommand": "phase-zero-residual-rollout",
        "invocation_arguments": invocation,
    }
    started_path = run_dir / "run_manifest.started.json"
    _write_json(started_path, started)
    artifact_paths = (
        evidence_path,
        live_path,
        run_dir / "committed_runtime_identity.before.json",
        run_dir / "committed_runtime_identity.after.json",
        run_dir / "frozen_hashes.before.json",
        run_dir / "frozen_hashes.after.json",
    )
    final = {
        **started,
        "lifecycle": "SUCCEEDED",
        "completed_at_utc": "2026-09-04T15:01:02.000000Z",
        "exit_code": 0,
        "started_manifest": artifacts.file_record(started_path, relative_to=run_dir),
        "logs": {
            "stdout.log": artifacts.file_record(stdout_path, relative_to=run_dir),
            "stderr.log": artifacts.file_record(stderr_path, relative_to=run_dir),
        },
        "artifacts": {
            path.name: artifacts.file_record(path, relative_to=run_dir)
            for path in artifact_paths
        },
    }
    _write_json(run_dir / "run_manifest.json", final)
    return {
        "path": evidence_path,
        "run_dir": run_dir,
        "project_root": project_root,
        "binding": binding,
        "payload": payload,
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_manifest_record(
    run_dir: Path, *, section: str, filename: str
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_json(manifest_path)
    records = manifest[section]
    assert isinstance(records, dict)
    records[filename] = artifacts.file_record(run_dir / filename, relative_to=run_dir)
    _write_json(manifest_path, manifest)


def test_managed_evidence_validator_binds_complete_finalized_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _managed_rollout_fixture(tmp_path, monkeypatch)

    evidence = validate_phase_zero_residual_rollout_evidence(
        fixture["path"],
        project_root=fixture["project_root"],
        expected_contract_binding=fixture["binding"],
    )

    assert evidence["passed"] is True
    assert evidence["seed"] == 1004
    assert evidence["sha256"] == _sha256(Path(fixture["path"]))
    assert evidence["run_manifest_sha256"] == _sha256(
        Path(fixture["run_dir"]) / "run_manifest.json"
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "missing-started",
        "started-final-identity",
        "missing-started-field",
        "invocation",
        "config-sha",
        "runtime-after",
        "frozen-after",
        "stdout-duplicate",
        "live-result",
        "artifact-inventory",
        "upstream-holdout",
    ),
)
def test_managed_evidence_validator_rejects_forged_or_repaired_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    fixture = _managed_rollout_fixture(tmp_path, monkeypatch)
    run_dir = Path(fixture["run_dir"])
    started_path = run_dir / "run_manifest.started.json"
    final_path = run_dir / "run_manifest.json"

    if tamper == "missing-started":
        started_path.unlink()
    elif tamper == "started-final-identity":
        started = _load_json(started_path)
        started_identity = started["identity"]
        assert isinstance(started_identity, dict)
        started_identity["seed"] = 1005
        _write_json(started_path, started)
        final = _load_json(final_path)
        final["started_manifest"] = artifacts.file_record(
            started_path, relative_to=run_dir
        )
        _write_json(final_path, final)
    elif tamper == "missing-started-field":
        started = _load_json(started_path)
        started.pop("invocation_arguments")
        _write_json(started_path, started)
        final = _load_json(final_path)
        final.pop("invocation_arguments")
        final["started_manifest"] = artifacts.file_record(
            started_path, relative_to=run_dir
        )
        _write_json(final_path, final)
    elif tamper == "invocation":
        started = _load_json(started_path)
        invocation = started["invocation_arguments"]
        assert isinstance(invocation, list)
        invocation[invocation.index("--policy-decisions") + 1] = "63"
        _write_json(started_path, started)
        final = _load_json(final_path)
        final["invocation_arguments"] = list(invocation)
        final["started_manifest"] = artifacts.file_record(
            started_path, relative_to=run_dir
        )
        _write_json(final_path, final)
    elif tamper == "config-sha":
        started = _load_json(started_path)
        started_identity = started["identity"]
        assert isinstance(started_identity, dict)
        started_identity["config_sha256"] = "f" * 64
        _write_json(started_path, started)
        final = _load_json(final_path)
        final_identity = final["identity"]
        assert isinstance(final_identity, dict)
        final_identity["config_sha256"] = "f" * 64
        final["started_manifest"] = artifacts.file_record(
            started_path, relative_to=run_dir
        )
        _write_json(final_path, final)
    elif tamper == "runtime-after":
        runtime_after = run_dir / "committed_runtime_identity.after.json"
        runtime = _load_json(runtime_after)
        runtime["tampered_after_capture"] = True
        _write_json(runtime_after, runtime)
        _refresh_manifest_record(
            run_dir,
            section="artifacts",
            filename="committed_runtime_identity.after.json",
        )
    elif tamper == "frozen-after":
        frozen_after = run_dir / "frozen_hashes.after.json"
        frozen = _load_json(frozen_after)
        frozen["passed"] = False
        _write_json(frozen_after, frozen)
        _refresh_manifest_record(
            run_dir, section="artifacts", filename="frozen_hashes.after.json"
        )
    elif tamper == "stdout-duplicate":
        stdout = run_dir / "stdout.log"
        with stdout.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    fixture["payload"], separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        _refresh_manifest_record(run_dir, section="logs", filename="stdout.log")
    elif tamper == "live-result":
        live_path = run_dir / "live_command_result.json"
        live = _load_json(live_path)
        live["exit_code"] = 2
        _write_json(live_path, live)
        _refresh_manifest_record(
            run_dir, section="artifacts", filename="live_command_result.json"
        )
    elif tamper == "artifact-inventory":
        extra = run_dir / "forged_rollout_copy.json"
        _write_json(extra, fixture["payload"])
        final = _load_json(final_path)
        artifact_records = final["artifacts"]
        assert isinstance(artifact_records, dict)
        artifact_records[extra.name] = artifacts.file_record(extra, relative_to=run_dir)
        _write_json(final_path, final)
    else:
        holdout_path = Path(str(fixture["binding"]["holdout_acceptance_path"]))
        _write_json(holdout_path, {"fixture": "tampered after binding"})

    with pytest.raises(PhaseZeroResidualRolloutError):
        validate_phase_zero_residual_rollout_evidence(
            fixture["path"],
            project_root=fixture["project_root"],
            expected_contract_binding=fixture["binding"],
        )


def test_runs_all_phase_windows_with_full_rate_zero_residual_audit() -> None:
    episode = _FakeEpisode()
    result = run_phase_zero_residual_rollout(
        episode,
        seed=1004,
        contract_binding=_binding(),
    )

    assert result["schema"] == ARTIFACT_SCHEMA
    assert result["status"] == "PASSED"
    assert result["passed"] is True
    assert episode.reset_calls == list(PHASE_IDS)
    assert result["phase_reset_count"] == 13
    assert result["total_policy_decisions"] == 12 + MAX_DECISIONS_PER_PHASE
    assert result["total_physics_ticks"] == 12 * 3 + MAX_DECISIONS_PER_PHASE * 8
    assert all(result["checks"].values())
    assert all(
        row["status"] == "NEXT_PHASE_BOUNDARY_REACHED"
        for row in result["phase_rollouts"][:-1]
    )
    assert result["phase_rollouts"][-1]["status"] == "MAX_64_DECISIONS_REACHED"
    assert all(
        row["tick_audit"]["nominal_action_binary64_sha256"]
        == row["tick_audit"]["applied_action_binary64_sha256"]
        for row in result["phase_rollouts"]
    )
    validate_phase_zero_residual_rollout_payload(
        result, expected_contract_binding=_binding()
    )


@pytest.mark.parametrize(
    "fault",
    ("root-write", "terminal", "safety", "controller", "lifecycle"),
)
def test_first_unsafe_tick_stops_physics_and_fails_closed(fault: str) -> None:
    episode = _FakeEpisode(fault=("P04", fault))
    result = run_phase_zero_residual_rollout(
        episode,
        seed=1004,
        contract_binding=_binding(),
    )

    assert result["status"] == "FAILED"
    assert result["passed"] is False
    assert episode.reset_calls == ["P01", "P02", "P03", "P04"]
    p04 = result["phase_rollouts"][3]
    assert p04["status"] == "FAILED_FAIL_CLOSED"
    assert p04["tick_audit"]["violations"]
    assert all(
        row["status"] == "NOT_RUN_FAIL_CLOSED"
        for row in result["phase_rollouts"][4:]
    )
    with pytest.raises(PhaseZeroResidualRolloutError):
        validate_phase_zero_residual_rollout_payload(
            result, expected_contract_binding=_binding()
        )


def test_recording_conformance_diagnostic_does_not_veto_rollout() -> None:
    episode = _FakeEpisode()
    original = episode._after_frame

    def with_diagnostic(before, state_id):
        frame = original(before, state_id)
        frame.termination_signals = TerminationSignals(
            reference_conformance_outside_30pct=True
        )
        return frame

    episode._after_frame = with_diagnostic
    result = run_phase_zero_residual_rollout(
        episode,
        seed=1004,
        contract_binding=_binding(),
    )

    assert result["passed"] is True
    assert all(
        row["tick_audit"][
            "reference_conformance_outside_30pct_diagnostic_tick_count"
        ]
        == row["physics_tick_count"]
        for row in result["phase_rollouts"]
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update(status="FAILED"),
        lambda value: value["checks"].update(no_post_tick0_root_state_writes=False),
        lambda value: value["phase_rollouts"][0]["tick_audit"].update(
            zero_root_write_tick_count=0
        ),
        lambda value: value["phase_rollouts"][0].update(final_state_id="P13"),
        lambda value: value["contract_binding"].update(
            phase_snapshot_bundle_sha256="f" * 64
        ),
    ),
)
def test_payload_validator_rejects_tamper(mutate) -> None:
    result = run_phase_zero_residual_rollout(
        _FakeEpisode(), seed=1004, contract_binding=_binding()
    )
    tampered = deepcopy(result)
    mutate(tampered)
    with pytest.raises(PhaseZeroResidualRolloutError):
        validate_phase_zero_residual_rollout_payload(
            tampered, expected_contract_binding=_binding()
        )


def test_strict_decoder_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(PhaseZeroResidualRolloutError, match="duplicate key"):
        _strict_json_object(b'{"passed":true,"passed":true}', label="test")
    with pytest.raises(PhaseZeroResidualRolloutError, match="non-finite"):
        _strict_json_object(b'{"value":NaN}', label="test")


def test_cli_and_managed_wrappers_expose_fail_closed_rollout_gate() -> None:
    parser = cli._parser()
    parsed = parser.parse_args(
        [
            "phase-zero-residual-rollout",
            "--run-dir",
            ".",
            "--seed",
            "1004",
            "--num-envs",
            "1",
            "--episode-count",
            "13",
            "--policy-decisions",
            "64",
            "--phase-effective-entry-holdout-acceptance",
            "holdout.json",
            "--deterministic",
        ]
    )
    assert parsed.command == "phase-zero-residual-rollout"
    assert "phase-zero-residual-rollout" in cli.LIVE_COMMANDS
    assert parsed.phase_effective_entry_holdout_acceptance == Path("holdout.json")

    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts" / "run_phase_zero_residual_rollout.ps1").read_text(
        encoding="utf-8"
    )
    common = (root / "scripts" / "_invoke_ppo_cli.ps1").read_text(
        encoding="utf-8"
    )
    training = (root / "scripts" / "train_phase_residual_ppo.ps1").read_text(
        encoding="utf-8"
    )
    assert '-RunKind "phase_zero_residual_rollout"' in wrapper
    assert '-Subcommand "phase-zero-residual-rollout"' in wrapper
    assert '"--episode-count", "13"' in wrapper
    assert '"--policy-decisions", "64"' in wrapper
    assert '"phase-zero-residual-rollout"' in common
    assert "PhaseZeroResidualRolloutEvidence" in training
    assert '"--phase-zero-residual-rollout-evidence"' in training


def test_phase_curriculum_training_rejects_missing_rollout_before_imports() -> None:
    args = SimpleNamespace(
        stage="phase-curriculum",
        phase_zero_residual_rollout_evidence=None,
    )
    with pytest.raises(cli.CliError, match="phase-zero-residual-rollout-evidence"):
        cli._require_training_phase_zero_residual_rollout(args)


def test_checkpoint_fields_bind_rollout_and_managed_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / ARTIFACT_FILENAME
    manifest = tmp_path / "run_manifest.json"
    artifact.write_bytes(b"rollout\n")
    manifest.write_bytes(b"manifest\n")
    evidence = {
        "schema": TRAINING_EVIDENCE_SCHEMA,
        "path": str(artifact),
        "sha256": cli._sha256(artifact),
        "run_manifest": str(manifest),
        "run_manifest_sha256": cli._sha256(manifest),
        "contract_binding": _binding(),
        "seed": 1004,
        "passed": True,
    }

    fields, files = cli._phase_zero_residual_rollout_fields(evidence)

    assert fields["phase_zero_residual_rollout_evidence"] == evidence
    assert fields["phase_zero_residual_rollout_evidence_path"] == str(artifact)
    assert fields["phase_zero_residual_rollout_evidence_sha256"] == evidence["sha256"]
    assert fields["phase_zero_residual_rollout_files"] == files
    artifact.write_bytes(b"tampered\n")
    with pytest.raises(cli.CliError, match="checkpoint evidence is invalid"):
        cli._phase_zero_residual_rollout_fields(evidence)


def _install_inherited_rollout_stubs(
    monkeypatch: pytest.MonkeyPatch,
    evidence: dict[str, object],
    fields: dict[str, object],
) -> None:
    monkeypatch.setattr(
        rollout_module,
        "build_contract_binding",
        lambda snapshot, effective, holdout: _binding(),
    )
    monkeypatch.setattr(
        rollout_module,
        "validate_phase_zero_residual_rollout_evidence",
        lambda path, **kwargs: dict(evidence),
    )
    monkeypatch.setattr(
        cli,
        "_phase_zero_residual_rollout_fields",
        lambda current: (dict(fields), dict(fields["phase_zero_residual_rollout_files"])),
    )


@pytest.mark.parametrize("stage", ("full-episode", "mild-randomization"))
def test_later_stage_inherits_and_revalidates_all_rollout_fields(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    evidence: dict[str, object] = {
        "schema": TRAINING_EVIDENCE_SCHEMA,
        "path": "C:\\evidence\\phase_zero_residual_rollout.json",
        "sha256": "9" * 64,
        "run_manifest": "C:\\evidence\\run_manifest.json",
        "run_manifest_sha256": "a" * 64,
        "contract_binding": _binding(),
        "seed": 1004,
        "passed": True,
    }
    files = {
        str(evidence["path"]): str(evidence["sha256"]),
        str(evidence["run_manifest"]): str(evidence["run_manifest_sha256"]),
    }
    fields: dict[str, object] = {
        "phase_zero_residual_rollout_evidence_path": evidence["path"],
        "phase_zero_residual_rollout_evidence_sha256": evidence["sha256"],
        "phase_zero_residual_rollout_run_manifest_path": evidence["run_manifest"],
        "phase_zero_residual_rollout_run_manifest_sha256": evidence[
            "run_manifest_sha256"
        ],
        "phase_zero_residual_rollout_evidence": evidence,
        "phase_zero_residual_rollout_files": files,
    }
    _install_inherited_rollout_stubs(monkeypatch, evidence, fields)
    args = SimpleNamespace(
        stage=stage,
        _phase_effective_entry_holdout_evidence={"passed": True},
    )
    infos = {**fields}

    inherited = cli._inherit_training_phase_zero_residual_rollout(
        args, infos, object(), object()
    )

    assert inherited == evidence
    assert args._phase_zero_residual_rollout_evidence == evidence
    assert args.phase_zero_residual_rollout_evidence == Path(str(evidence["path"]))


@pytest.mark.parametrize("stage", ("full-episode", "mild-randomization"))
def test_later_stage_rejects_missing_rollout_ancestry(stage: str) -> None:
    args = SimpleNamespace(stage=stage)
    with pytest.raises(
        cli.CliError, match="omits phase zero-residual rollout ancestry"
    ):
        cli._inherit_training_phase_zero_residual_rollout(
            args, {}, object(), object()
        )


def test_phase_to_phase_resume_requires_same_explicit_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded = {
        "path": "C:\\evidence\\phase_zero_residual_rollout.json",
        "passed": True,
    }
    fields = {
        "phase_zero_residual_rollout_evidence": embedded,
        "phase_zero_residual_rollout_files": {},
    }
    _install_inherited_rollout_stubs(monkeypatch, embedded, fields)
    args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_effective_entry_holdout_evidence={"passed": True},
        _phase_zero_residual_rollout_evidence={
            "path": "C:\\evidence\\different.json",
            "passed": True,
        },
    )
    with pytest.raises(cli.CliError, match="differs from the explicit"):
        cli._inherit_training_phase_zero_residual_rollout(
            args,
            {"stage": "phase-curriculum", **fields},
            object(),
            object(),
        )


def test_phase_to_phase_resume_inherits_same_explicit_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = {
        "path": "C:\\evidence\\phase_zero_residual_rollout.json",
        "passed": True,
    }
    fields = {
        "phase_zero_residual_rollout_evidence": evidence,
        "phase_zero_residual_rollout_files": {},
    }
    _install_inherited_rollout_stubs(monkeypatch, evidence, fields)
    args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_effective_entry_holdout_evidence={"passed": True},
        _phase_zero_residual_rollout_evidence=dict(evidence),
    )

    inherited = cli._inherit_training_phase_zero_residual_rollout(
        args,
        {"stage": "phase-curriculum", **fields},
        object(),
        object(),
    )

    assert inherited == evidence
    assert args._phase_zero_residual_rollout_evidence == evidence
    assert args.phase_zero_residual_rollout_evidence == Path(evidence["path"])


def test_inherited_rollout_rejects_individual_manifest_field_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = {
        "path": "C:\\evidence\\phase_zero_residual_rollout.json",
        "passed": True,
    }
    fields = {
        "phase_zero_residual_rollout_evidence_sha256": "9" * 64,
        "phase_zero_residual_rollout_evidence": evidence,
        "phase_zero_residual_rollout_files": {
            str(evidence["path"]): "9" * 64,
        },
    }
    _install_inherited_rollout_stubs(monkeypatch, evidence, fields)
    infos = {**fields, "phase_zero_residual_rollout_evidence_sha256": "a" * 64}
    args = SimpleNamespace(
        stage="full-episode",
        _phase_effective_entry_holdout_evidence={"passed": True},
    )

    with pytest.raises(cli.CliError, match="binding differs for"):
        cli._inherit_training_phase_zero_residual_rollout(
            args,
            infos,
            object(),
            object(),
        )


def test_phase_first_entry_keeps_explicit_rollout_when_resume_has_none() -> None:
    explicit = {"path": "C:\\evidence\\phase_zero_residual_rollout.json"}
    args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_zero_residual_rollout_evidence=explicit,
    )
    assert (
        cli._inherit_training_phase_zero_residual_rollout(
            args, {"stage": "smoke"}, object(), object()
        )
        == explicit
    )


def test_phase_to_phase_resume_cannot_fill_missing_rollout_from_explicit() -> None:
    explicit = {"path": "C:\\evidence\\phase_zero_residual_rollout.json"}
    args = SimpleNamespace(
        stage="phase-curriculum",
        _phase_zero_residual_rollout_evidence=explicit,
    )

    with pytest.raises(cli.CliError, match="omits phase zero-residual rollout ancestry"):
        cli._inherit_training_phase_zero_residual_rollout(
            args,
            {"stage": "phase-curriculum"},
            object(),
            object(),
        )


def test_rollout_horizon_cannot_be_weakened() -> None:
    with pytest.raises(PhaseZeroResidualRolloutError, match="exactly 64"):
        run_phase_zero_residual_rollout(
            _FakeEpisode(),
            seed=1004,
            contract_binding=_binding(),
            max_decisions_per_phase=63,
        )


def test_evidence_filename_is_stable() -> None:
    assert ARTIFACT_FILENAME == "phase_zero_residual_rollout.json"
