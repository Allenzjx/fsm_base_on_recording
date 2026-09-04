from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wlr50_clean.ppo import artifacts
from wlr50_clean.ppo import training_orchestration as subject


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _create_windows_junction(link: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {result.stderr or result.stdout}")


def _install_chain_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    smoke_num_envs: int = 8,
    phase_num_envs: int = 1,
    full_num_envs: int = 1,
    full_requested: int = 100_000,
    promotion_bound_index: int | None = None,
) -> dict[str, object]:
    root = tmp_path.resolve()
    profile = root / "configs" / "ppo_training_phase_v1.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_bytes(subject.DEFAULT_TRAINING_CONFIG.read_bytes())
    matrix = root / "matrix.json"
    _write_json(matrix, {"selected_num_envs": 8})
    matrix_record = _record(matrix)
    matrix_evidence = {
        "path": str(matrix.resolve()),
        "sha256": matrix_record["sha256"],
        "selected_num_envs": 8,
        "passed": True,
    }
    soft_raw = {"path": str(root / "soft.json"), "passed": True}
    soft_binding = {"acceptance": {"path": soft_raw["path"]}, "passed": True}
    config_record = _record(profile)
    configs = [
        {
            "path": "configs/ppo_training_phase_v1.yaml",
            "bytes": config_record["bytes"],
            "sha256": config_record["sha256"],
        }
    ]
    identity = {
        "seed": 1001,
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
    }
    specs = [("smoke", 10_000, smoke_num_envs)]
    specs += [("phase-curriculum", 10_000, phase_num_envs)] * 10
    if full_requested:
        assert full_requested % 10_000 == 0
        specs += [("full-episode", 10_000, full_num_envs)] * (
            full_requested // 10_000
        )

    initial = root / "initial.pt"
    initial.write_bytes(b"initial")
    previous = _record(initial)
    initial_manifest = root / "initial_manifest.json"
    _write_json(initial_manifest, {"fixture": "initial"})
    chunks: list[dict[str, object]] = []
    training_dirs: list[Path] = []
    screening_dirs: list[Path] = []
    base_time = datetime(2026, 9, 4, tzinfo=timezone.utc)
    global_step = 0
    terminal_bytes = b""
    terminal_last = root / "checkpoint_last.pt"
    for index, (stage, requested, num_envs) in enumerate(specs):
        global_step += requested
        history = root / "history" / f"checkpoint_{index:02d}.pt"
        history.parent.mkdir(exist_ok=True)
        terminal_bytes = f"history-{index}".encode("ascii")
        history.write_bytes(terminal_bytes)
        history_record = _record(history)
        checkpoint_manifest = history.with_suffix(".manifest.json")
        _write_json(checkpoint_manifest, {"chunk": index})
        started = base_time + timedelta(seconds=index * 4)
        chunk = {
            "stage": stage,
            "requested_policy_decisions": requested,
            "stage_policy_decisions": requested,
            "global_policy_decisions": global_step,
            "resume_global_policy_decisions": global_step - requested,
            "iterations": 1,
            "num_envs": num_envs,
            "rollout_length": requested // num_envs,
            "run_directory": str(root / f"training-{index}"),
            "run_manifest": history_record,
            "training_result": history_record,
            "resume_checkpoint": previous,
            "immutable_history_checkpoint": history_record,
            "checkpoint_manifest": _record(checkpoint_manifest),
            "checkpoint_manifest_payload": {"chunk": index},
            "checkpoint_last_path": str(terminal_last.resolve()),
            "identity": dict(identity),
            "configs": list(configs),
            "frozen_hash_audits": [],
            "committed_runtime_identities": [],
            "committed_runtime_identity_before_payload": {
                "content_sha256": "d" * 64
            },
            "completed_at": started + timedelta(seconds=1),
            "started_at": started,
            "soft_reset_acceptance_raw": soft_raw if num_envs == 1 else None,
            "vector_matrix_raw": matrix_evidence if num_envs > 1 else None,
            "vector_matrix_path": str(matrix.resolve()) if num_envs > 1 else None,
            "vector_matrix_sha256": (
                matrix_record["sha256"] if num_envs > 1 else None
            ),
        }
        chunks.append(chunk)
        previous = history_record
        training_dirs.append(root / f"training-{index}")
        screening_dirs.append(root / f"screening-{index}")
    terminal_last.write_bytes(terminal_bytes)

    by_training = {str(path.resolve()): row for path, row in zip(training_dirs, chunks)}
    by_screening = {
        str(path.resolve()): index for index, path in enumerate(screening_dirs)
    }

    def fake_training(run_dir: Path | str, **_: object) -> dict[str, object]:
        return dict(by_training[str(Path(run_dir).resolve())])

    def fake_screening(
        run_dir: Path | str, *, chunk: dict[str, object], **_: object
    ) -> dict[str, object]:
        index = by_screening[str(Path(run_dir).resolve())]
        record = chunk["immutable_history_checkpoint"]
        return {
            "run_directory": str(Path(run_dir).resolve()),
            "run_manifest": record,
            "checkpoint_evaluation": record,
            "checkpoint": record,
            "checkpoint_manifest": chunk["checkpoint_manifest"],
            "seed": 2001 + (index % 5),
            "global_policy_decisions": chunk["global_policy_decisions"],
            "physical_passed": False,
            "complete_evidence": True,
            "episode_summary": record,
            "trial_manifest": record,
            "policy_trace": record,
            "started_at": chunk["completed_at"] + timedelta(seconds=1),
            "completed_at": chunk["completed_at"] + timedelta(seconds=2),
            "frozen_hash_audits": [],
            "committed_runtime_identities": [],
        }

    monkeypatch.setattr(subject, "_validate_training_chunk", fake_training)
    monkeypatch.setattr(subject, "_validate_screening", fake_screening)
    monkeypatch.setattr(
        subject, "_validate_vector_matrix_binding", lambda *args, **kwargs: matrix_evidence
    )
    monkeypatch.setattr(
        subject, "_validate_soft_reset_binding", lambda *args, **kwargs: soft_binding
    )
    monkeypatch.setattr(
        subject,
        "_validate_initial_checkpoint",
        lambda *args, **kwargs: {
            "path": previous["path"] if not chunks else str(initial.resolve()),
            "sha256": _record(initial)["sha256"],
            "manifest_path": str(initial_manifest.resolve()),
            "manifest_sha256": _record(initial_manifest)["sha256"],
            "zero_mean_actor_output_layer_verified": True,
        },
    )
    monkeypatch.setattr(
        subject,
        "_validate_smoke_checkpoint",
        lambda *args, **kwargs: (
            {
                "path": chunks[0]["immutable_history_checkpoint"]["path"],
                "sha256": chunks[0]["immutable_history_checkpoint"]["sha256"],
                "manifest_path": chunks[0]["checkpoint_manifest"]["path"],
                "manifest_sha256": chunks[0]["checkpoint_manifest"]["sha256"],
            },
            {
                "path": chunks[0]["immutable_history_checkpoint"]["path"],
                "sha256": chunks[0]["immutable_history_checkpoint"]["sha256"],
                "manifest_path": chunks[0]["checkpoint_manifest"]["path"],
                "manifest_sha256": chunks[0]["checkpoint_manifest"]["sha256"],
            },
        ),
    )
    initial_publication_run = root / "initial-checkpoint-publication"
    monkeypatch.setattr(
        subject,
        "_validate_initial_checkpoint_publication",
        lambda *args, **kwargs: {
            "run_directory": str(initial_publication_run.resolve()),
            "reused_existing": False,
        },
    )

    decisions: list[Path] = []
    if promotion_bound_index is not None:
        decision = root / "promotion.json"
        _write_json(decision, {"fixture": True})
        decisions.append(decision)

        def fake_promotion(path: Path | str, *, cache: dict, **_: object) -> dict:
            captured = subject._snapshot(path, label="promotion", cache=cache)
            bound = chunks[promotion_bound_index]
            return {
                "record": captured.record(),
                "promoted": True,
                "first_failed_gate": None,
                "bound_chunk_index": promotion_bound_index,
                "candidate_checkpoint": bound["immutable_history_checkpoint"],
            }

        monkeypatch.setattr(subject, "_validate_promotion_decision", fake_promotion)

    return {
        "root": root,
        "training_config": profile,
        "matrix": matrix,
        "training_dirs": training_dirs,
        "screening_dirs": screening_dirs,
        "initial_publication_run": initial_publication_run,
        "decisions": decisions,
        "chunks": chunks,
    }


def _build_from_stubs(inputs: dict[str, object]) -> dict[str, object]:
    return subject._build_payload(
        training_run_dirs=inputs["training_dirs"],
        screening_run_dirs=inputs["screening_dirs"],
        initial_checkpoint_publication_run=inputs["initial_publication_run"],
        training_config_path=inputs["training_config"],
        vector_benchmark_matrix_path=inputs["matrix"],
        promotion_decision_paths=inputs["decisions"],
        project_root=inputs["root"],
        expected_seed=1001,
        expected_num_envs=1,
        generated_at_utc="2026-09-04T00:00:00Z",
        cache={},
    )


def test_budget_exhaustion_accepts_mixed_env_chain_and_failed_screenings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _install_chain_stubs(tmp_path, monkeypatch)
    payload = _build_from_stubs(inputs)
    assert payload["status"] == "BUDGET_EXHAUSTED_NO_PROMOTION"
    assert payload["valid"] is True
    assert payload["environment_counts_by_stage"] == {
        "smoke": 8,
        "phase-curriculum": 1,
        "full-episode": 1,
    }
    assert payload["selected_vector_num_envs"] == 8
    assert payload["chunk_count"] == 21
    assert all(not row["screening"]["physical_passed"] for row in payload["chunks"])


@pytest.mark.parametrize(
    ("smoke_num_envs", "phase_num_envs", "full_num_envs", "message"),
    (
        (1, 1, 1, "smoke"),
        (8, 8, 1, "phase-curriculum"),
        (8, 1, 8, "full-episode"),
    ),
)
def test_stage_environment_rules_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_num_envs: int,
    phase_num_envs: int,
    full_num_envs: int,
    message: str,
) -> None:
    inputs = _install_chain_stubs(
        tmp_path,
        monkeypatch,
        smoke_num_envs=smoke_num_envs,
        phase_num_envs=phase_num_envs,
        full_num_envs=full_num_envs,
    )
    with pytest.raises(subject.TrainingOrchestrationError, match=message):
        _build_from_stubs(inputs)


def test_true_terminal_promotion_allows_early_full_episode_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _install_chain_stubs(
        tmp_path,
        monkeypatch,
        full_requested=10_000,
        promotion_bound_index=11,
    )
    payload = _build_from_stubs(inputs)
    assert payload["status"] == "PROMOTION_FOUND"
    assert payload["stage_requested_policy_decisions"]["full-episode"] == 10_000
    assert payload["budget_accounting_basis"] == "requested_policy_decisions"
    assert payload["actual_decisions_are_whole_ppo_batches"] is True
    assert payload["stage_actual_policy_decisions"]["full-episode"] == 10_000
    assert payload["stage_rounding_overrun_policy_decisions"]["full-episode"] == 0
    assert payload["promotion_decisions"][0]["bound_chunk_index"] == 11
    assert payload["terminal"]["promotion_bound_chunk_index"] == 11
    assert payload["terminal"]["promotion_bound_global_policy_decisions"] == (
        payload["chunks"][11]["global_policy_decisions"]
    )
    assert payload["terminal"]["passing_promotion_decision"] == (
        payload["promotion_decisions"][0]["record"]
    )
    assert payload["terminal"]["promotion_candidate_checkpoint"] == (
        payload["promotion_decisions"][0]["candidate_checkpoint"]
    )


def test_promotion_decision_must_be_inside_managed_runs_root(tmp_path: Path) -> None:
    escaped = tmp_path / "promotion_decision.json"
    _write_json(escaped, {})

    with pytest.raises(subject.TrainingOrchestrationError, match="must be inside"):
        subject._validate_promotion_decision(
            escaped,
            chunks=(),
            project_root=tmp_path.resolve(),
            cache={},
        )


def test_initial_checkpoint_safe_load_proves_exact_zero_and_creation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path.resolve()
    checkpoint = (
        root
        / "outputs"
        / "ppo_phase_v1"
        / "checkpoints"
        / "checkpoint_initial_zero_residual.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    creation_run = root / "runs" / "ppo_phase_v1" / "train" / "creation"
    creation_run.mkdir(parents=True)
    identity_path = creation_run / "committed_runtime_identity.before.json"
    _write_json(identity_path, {"fixture": "runtime"})
    identity_record = _record(identity_path)
    content_sha = "d" * 64
    infos = {
        "schema": subject.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "initial_zero_residual",
        "training_seed": 1001,
        "global_policy_decisions": 0,
        "residual_dimension": 12,
        "source_git_commit": "b" * 40,
        "committed_runtime_content_sha256": content_sha,
        "creation_runtime_identity_path": str(identity_path.resolve()),
        "creation_runtime_identity_sha256": identity_record["sha256"],
        "zero_mean_actor_output_layer_verified": True,
    }
    from wlr50_clean.ppo.rl_library_wrapper import CHECKPOINT_RUNTIME_CONTRACT_FIELDS

    for field in CHECKPOINT_RUNTIME_CONTRACT_FIELDS:
        infos.setdefault(field, f"fixture-{field}")

    def write_checkpoint(weight_value: float) -> dict[str, object]:
        torch.save(
            {
                "actor_state_dict": {
                    "mlp.0.weight": torch.full((12, 3), weight_value),
                    "mlp.0.bias": torch.zeros(12),
                },
                "infos": infos,
            },
            checkpoint,
        )
        payload = {
            **infos,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        _write_json(
            checkpoint.with_name(checkpoint.stem + "_manifest.json"), payload
        )
        return _record(checkpoint)

    resume = write_checkpoint(0.0)
    first = {
        "resume_checkpoint": resume,
        "resume_global_policy_decisions": 0,
        "committed_runtime_identity_before_payload": {
            "content_sha256": content_sha
        },
        "checkpoint_manifest_payload": dict(infos),
    }
    monkeypatch.setattr(
        subject,
        "_validate_finalized_run",
        lambda *args, **kwargs: {
            "directory": creation_run,
            "committed_runtime_identities": [identity_record, identity_record],
            "committed_runtime_identity_before_payload": {
                "git_commit": "b" * 40,
                "content_sha256": content_sha,
            },
            "identity": {"timestamp_utc": "2026-09-04T00:00:00Z"},
            "payload": {"completed_at_utc": "2026-09-04T00:00:01Z"},
            "run_manifest": {"path": str(creation_run / "run_manifest.json")},
        },
    )

    evidence = subject._validate_initial_checkpoint(
        first,
        project_root=root,
        expected_git_commit="b" * 40,
        expected_seed=1001,
        cache={},
    )
    assert evidence["path"] == str(checkpoint.resolve())
    assert evidence["zero_mean_actor_output_layer_verified"] is True
    assert evidence["creation_runtime_identity"]["sha256"] == identity_record["sha256"]

    first["resume_checkpoint"] = write_checkpoint(0.25)
    with pytest.raises(subject.TrainingOrchestrationError, match="not exact zero"):
        subject._validate_initial_checkpoint(
            first,
            project_root=root,
            expected_git_commit="b" * 40,
            expected_seed=1001,
            cache={},
        )


def test_initial_checkpoint_creation_binding_accepts_dedicated_finalized_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    creation_run = (
        root / "runs" / "ppo_phase_v1" / "initial_checkpoint" / "creation"
    )
    creation_run.mkdir(parents=True)
    identity = creation_run / "committed_runtime_identity.before.json"
    _write_json(identity, {"runtime": "identity"})
    identity_record = _record(identity)
    seen: dict[str, object] = {}

    def validate(*args, **kwargs):
        seen.update(kwargs)
        return {
            "directory": creation_run,
            "committed_runtime_identities": [identity_record, identity_record],
            "committed_runtime_identity_before_payload": {
                "git_commit": "b" * 40,
                "content_sha256": "d" * 64,
            },
            "identity": {"timestamp_utc": "2026-09-04T00:00:00Z"},
            "payload": {"completed_at_utc": "2026-09-04T00:00:01Z"},
            "run_manifest": {"path": str(creation_run / "run_manifest.json")},
        }

    monkeypatch.setattr(subject, "_validate_finalized_run", validate)
    result = subject._validate_checkpoint_creation_binding(
        {
            "stage": "initial_zero_residual",
            "source_git_commit": "b" * 40,
            "committed_runtime_content_sha256": "d" * 64,
            "creation_runtime_identity_path": str(identity.resolve()),
            "creation_runtime_identity_sha256": identity_record["sha256"],
        },
        project_root=root,
        expected_git_commit="b" * 40,
        expected_content_sha256="d" * 64,
        cache={},
    )

    assert seen["run_kind"] == "initial_checkpoint"
    assert seen["training_stage"] == "initialize-zero-residual"
    assert seen["subcommand"] == "initialize-zero-residual"
    assert result["creation_run_kind"] == "initial_checkpoint"


def test_initial_publication_rejects_manual_same_byte_source_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    creator = root / "runs" / "ppo_phase_v1" / "initial_checkpoint" / "creator"
    publisher = (
        root
        / "runs"
        / "ppo_phase_v1"
        / "initial_checkpoint_publication"
        / "publisher"
    )
    canonical_dir = root / "outputs" / "ppo_phase_v1" / "checkpoints"
    manual_dir = root / "manual-copy"
    for directory in (creator, publisher, canonical_dir, manual_dir):
        directory.mkdir(parents=True)
    checkpoint_bytes = b"verified-initial"
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    creator_checkpoint = creator / "checkpoint_initial_zero_residual.pt"
    creator_checkpoint.write_bytes(checkpoint_bytes)
    creator_manifest = creator / "checkpoint_initial_zero_residual_manifest.json"
    _write_json(
        creator_manifest,
        {
            "stage": "initial_zero_residual",
            "checkpoint_path": str(creator_checkpoint),
            "checkpoint_sha256": digest,
        },
    )
    canonical = canonical_dir / "checkpoint_initial_zero_residual.pt"
    canonical.write_bytes(checkpoint_bytes)
    canonical_manifest = canonical_dir / "checkpoint_initial_zero_residual_manifest.json"
    _write_json(
        canonical_manifest,
        {
            "stage": "initial_zero_residual",
            "checkpoint_path": str(canonical),
            "checkpoint_sha256": digest,
        },
    )
    manual_checkpoint = manual_dir / creator_checkpoint.name
    manual_manifest = manual_dir / creator_manifest.name
    manual_checkpoint.write_bytes(checkpoint_bytes)
    manual_manifest.write_bytes(creator_manifest.read_bytes())
    result_path = publisher / "initial_checkpoint_publication.json"

    def result(source_checkpoint: Path, source_manifest: Path) -> dict[str, object]:
        payload = {
            "schema": "wlr50_clean.initial_zero_residual_checkpoint_publication.v1",
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": digest,
            "source_checkpoint_manifest": str(source_manifest),
            "source_checkpoint_manifest_sha256": _record(source_manifest)["sha256"],
            "checkpoint": str(canonical),
            "checkpoint_sha256": digest,
            "checkpoint_manifest": str(canonical_manifest),
            "checkpoint_manifest_sha256": _record(canonical_manifest)["sha256"],
            "reused_existing": False,
            "no_existing_artifact_overwritten": True,
            "source_initializer_finalized_success": True,
            "embedded_infos_match_manifest": True,
            "zero_mean_actor_output_layer_verified": True,
            "creation_run_kind": "initial_checkpoint",
            "creation_run_directory": str(creator),
        }
        _write_json(result_path, payload)
        return payload

    run = {
        "directory": publisher,
        "identity": {
            "seed": 1001,
            "environment_count": 1,
            "git_commit": "b" * 40,
            "config_sha256": "a" * 64,
        },
        "configs": [{"path": "config"}],
        "started_at": datetime(2026, 9, 4, 0, 0, 2, tzinfo=timezone.utc),
        "completed_at": datetime(2026, 9, 4, 0, 0, 3, tzinfo=timezone.utc),
        "run_manifest": {"path": str(publisher / "run_manifest.json")},
        "frozen_audits": [],
        "committed_runtime_identities": [],
    }
    monkeypatch.setattr(subject, "_validate_finalized_run", lambda *args, **kwargs: run)
    current_result = result(creator_checkpoint, creator_manifest)
    monkeypatch.setattr(
        subject,
        "_required_artifact",
        lambda *args, **kwargs: (result_path, current_result),
    )
    initial = {
        "path": str(canonical),
        "sha256": digest,
        "manifest_path": str(canonical_manifest),
        "manifest_sha256": _record(canonical_manifest)["sha256"],
        "creation_runtime_identity": {
            "creation_run_kind": "initial_checkpoint",
            "creation_run_directory": str(creator),
            "creation_run_completed_at_utc": "2026-09-04T00:00:01Z",
        },
    }
    valid = subject._validate_initial_checkpoint_publication(
        publisher,
        initial,
        project_root=root,
        expected_seed=1001,
        expected_git_commit="b" * 40,
        expected_config_sha256="a" * 64,
        expected_configs=[{"path": "config"}],
        first_training_started_at=datetime(
            2026, 9, 4, 0, 0, 4, tzinfo=timezone.utc
        ),
        cache={},
    )
    assert valid["source_checkpoint"]["path"] == str(creator_checkpoint)

    current_result = result(manual_checkpoint, manual_manifest)
    with pytest.raises(subject.TrainingOrchestrationError, match="reuse/source mode"):
        subject._validate_initial_checkpoint_publication(
            publisher,
            initial,
            project_root=root,
            expected_seed=1001,
            expected_git_commit="b" * 40,
            expected_config_sha256="a" * 64,
            expected_configs=[{"path": "config"}],
            first_training_started_at=datetime(
                2026, 9, 4, 0, 0, 4, tzinfo=timezone.utc
            ),
            cache={},
        )


@pytest.mark.parametrize("wrong_kind", ["train", "initial_checkpoint"])
def test_initial_publication_rejects_wrong_kind_publisher(
    tmp_path: Path, wrong_kind: str
) -> None:
    root = tmp_path.resolve()
    run = root / "runs" / "ppo_phase_v1" / wrong_kind / "publisher"
    run.mkdir(parents=True)
    if wrong_kind == "initial_checkpoint_publication":
        pytest.fail("parameter must exercise a wrong run kind")

    with pytest.raises(subject.TrainingOrchestrationError, match="must be"):
        subject._validate_initial_checkpoint_publication(
            run,
            {},
            project_root=root,
            expected_seed=1001,
            expected_git_commit="b" * 40,
            expected_config_sha256="a" * 64,
            expected_configs=(),
            first_training_started_at=datetime.now(timezone.utc),
            cache={},
        )


def test_initial_publication_rejects_failed_publisher(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    run = (
        root
        / "runs"
        / "ppo_phase_v1"
        / "initial_checkpoint_publication"
        / "failed-publisher"
    )
    run.mkdir(parents=True)
    _write_json(
        run / "run_manifest.json",
        {"schema": subject.RUN_MANIFEST_SCHEMA, "lifecycle": "FAILED"},
    )
    _write_json(run / "run_manifest.started.json", {})

    with pytest.raises(subject.TrainingOrchestrationError, match="finalized lifecycle"):
        subject._validate_initial_checkpoint_publication(
            run,
            {},
            project_root=root,
            expected_seed=1001,
            expected_git_commit="b" * 40,
            expected_config_sha256="a" * 64,
            expected_configs=(),
            first_training_started_at=datetime.now(timezone.utc),
            cache={},
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_canonical_initial_and_smoke_inputs_reject_junction_ancestry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    redirected = tmp_path / "redirected-outputs"
    checkpoints = redirected / "ppo_phase_v1" / "checkpoints"
    checkpoints.mkdir(parents=True)
    initial = checkpoints / "checkpoint_initial_zero_residual.pt"
    smoke = checkpoints / "checkpoint_smoke.pt"
    initial.write_bytes(b"outside-initial")
    smoke.write_bytes(b"outside-smoke")
    _write_json(
        checkpoints / "checkpoint_initial_zero_residual_manifest.json", {}
    )
    _write_json(checkpoints / "checkpoint_smoke_manifest.json", {})
    _create_windows_junction(root / "outputs", redirected)
    try:
        with pytest.raises(subject.TrainingOrchestrationError, match="symlink|junction"):
            subject._validate_initial_checkpoint(
                {
                    "resume_checkpoint": {
                        "path": str(root / "outputs" / "ppo_phase_v1" / "checkpoints" / initial.name),
                        "sha256": "0" * 64,
                    },
                    "resume_global_policy_decisions": 0,
                },
                project_root=root,
                expected_git_commit="b" * 40,
                expected_seed=1001,
                cache={},
            )
        with pytest.raises(subject.TrainingOrchestrationError, match="symlink|junction"):
            subject._validate_smoke_checkpoint(
                {
                    "stage": "smoke",
                    "immutable_history_checkpoint": {},
                    "checkpoint_manifest": {},
                },
                project_root=root,
                cache={},
            )
    finally:
        if (root / "outputs").exists():
            os.rmdir(root / "outputs")


def test_smoke_checkpoint_is_byte_and_embedded_abi_copy_of_first_history(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path.resolve()
    history = root / "history" / "checkpoint_smoke_10000.pt"
    canonical = (
        root / "outputs" / "ppo_phase_v1" / "checkpoints" / "checkpoint_smoke.pt"
    )
    history.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    infos = {
        "schema": subject.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "smoke",
        "global_policy_decisions": 10_000,
    }
    torch.save({"actor_state_dict": {}, "infos": infos}, history)
    canonical.write_bytes(history.read_bytes())
    digest = hashlib.sha256(history.read_bytes()).hexdigest()
    history_manifest = history.with_name(history.stem + "_manifest.json")
    canonical_manifest = canonical.with_name(canonical.stem + "_manifest.json")
    _write_json(
        history_manifest,
        {**infos, "checkpoint_path": str(history.resolve()), "checkpoint_sha256": digest},
    )
    _write_json(
        canonical_manifest,
        {**infos, "checkpoint_path": str(canonical.resolve()), "checkpoint_sha256": digest},
    )
    first = {
        "stage": "smoke",
        "immutable_history_checkpoint": _record(history),
        "checkpoint_manifest": _record(history_manifest),
        "checkpoint_manifest_payload": json.loads(
            history_manifest.read_text(encoding="utf-8")
        ),
    }

    smoke, canonical_record = subject._validate_smoke_checkpoint(
        first, project_root=root, cache={}
    )
    assert smoke["path"] == str(history.resolve())
    assert canonical_record["path"] == str(canonical.resolve())
    assert smoke["sha256"] == canonical_record["sha256"]

    canonical.write_bytes(b"different")
    with pytest.raises(subject.TrainingOrchestrationError, match="differs"):
        subject._validate_smoke_checkpoint(first, project_root=root, cache={})


def test_promotion_cannot_authorize_later_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _install_chain_stubs(
        tmp_path,
        monkeypatch,
        promotion_bound_index=11,
    )
    with pytest.raises(subject.TrainingOrchestrationError, match="continued after"):
        _build_from_stubs(inputs)


def test_publication_hook_detects_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    config = root / "config.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")
    config_sha, config_records = artifacts.config_set_record((config,), project_root=root)
    run_dir = (
        root
        / "runs"
        / "ppo_phase_v1"
        / "training_orchestration"
        / "fixture-run"
    )
    run_dir.mkdir(parents=True)
    source = root / "source.json"
    _write_json(source, {"value": 1})
    frozen = root / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    _write_json(frozen, {})
    _write_json(run_dir / "frozen_hashes.before.json", {})
    runtime_before = run_dir / "committed_runtime_identity.before.json"
    _write_json(runtime_before, {})
    _write_json(
        run_dir / "run_manifest.started.json",
        {
            "schema": subject.RUN_MANIFEST_SCHEMA,
            "lifecycle": "STARTED",
            "immutable_run_directory": True,
            "run_kind": "training_orchestration",
            "entrypoint": "wlr50_clean.ppo.training_orchestration",
            "subcommand": "build-manifest",
            "run_dir": str(run_dir.resolve()),
            "project_root": str(root),
            "configs": config_records,
            "identity": {
                "training_stage": "training-orchestration-prefinal",
                "seed": 1001,
                "environment_count": 1,
                "git_commit": "b" * 40,
                "config_sha256": config_sha,
            },
        },
    )

    def fake_payload(*, cache: dict, **_: object) -> dict[str, object]:
        subject._snapshot(source, label="mutable fixture", cache=cache)
        return {
            "training_seed": 1001,
            "orchestration_environment_count": 1,
            "git_commit": "b" * 40,
            "config_sha256": config_sha,
            "config_records": config_records,
        }

    monkeypatch.setattr(subject, "_build_payload", fake_payload)
    monkeypatch.setattr(subject, "_validate_frozen_audit", lambda *args, **kwargs: ())

    def fake_runtime(path: Path, *, cache: dict, **_: object):
        captured = subject._snapshot(path, label="runtime fixture", cache=cache)
        return {}, (), captured

    monkeypatch.setattr(subject, "_validate_runtime_identity_document", fake_runtime)

    def mutate() -> None:
        _write_json(source, {"value": 2})

    with pytest.raises(subject.TrainingOrchestrationError, match="changed before publication"):
        subject.build_training_orchestration_manifest(
            training_run_dirs=[root / "unused-train"],
            screening_run_dirs=[root / "unused-screen"],
            initial_checkpoint_publication_run=root / "unused-publication",
            training_config_path=config,
            vector_benchmark_matrix_path=root / "unused-matrix",
            output_path=run_dir / subject.TRAINING_ORCHESTRATION_FILENAME,
            project_root=root,
            expected_seed=1001,
            expected_num_envs=1,
            _before_publish_hook=mutate,
        )
    assert not (run_dir / subject.TRAINING_ORCHESTRATION_FILENAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_publication_hook_rejects_run_directory_replaced_by_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "project").resolve()
    config = root / "configs" / "ppo_training_phase_v1.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture: true\n", encoding="utf-8")
    config_sha, config_records = artifacts.config_set_record(
        (config,), project_root=root
    )
    run_dir = (
        root
        / "runs"
        / "ppo_phase_v1"
        / "training_orchestration"
        / "fixture-run"
    )
    run_dir.mkdir(parents=True)
    source = root / "source.json"
    _write_json(source, {"value": 1})
    frozen = root / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    _write_json(frozen, {})
    _write_json(run_dir / "frozen_hashes.before.json", {})
    runtime_before = run_dir / "committed_runtime_identity.before.json"
    _write_json(runtime_before, {})
    _write_json(
        run_dir / "run_manifest.started.json",
        {
            "schema": subject.RUN_MANIFEST_SCHEMA,
            "lifecycle": "STARTED",
            "immutable_run_directory": True,
            "run_kind": "training_orchestration",
            "entrypoint": "wlr50_clean.ppo.training_orchestration",
            "subcommand": "build-manifest",
            "run_dir": str(run_dir),
            "project_root": str(root),
            "configs": config_records,
            "identity": {
                "training_stage": "training-orchestration-prefinal",
                "seed": 1001,
                "environment_count": 1,
                "git_commit": "b" * 40,
                "config_sha256": config_sha,
            },
        },
    )

    def fake_payload(*, cache: dict, **_: object) -> dict[str, object]:
        subject._snapshot(source, label="stable fixture", cache=cache)
        return {
            "training_seed": 1001,
            "orchestration_environment_count": 1,
            "git_commit": "b" * 40,
            "config_sha256": config_sha,
            "config_records": config_records,
        }

    monkeypatch.setattr(subject, "_build_payload", fake_payload)
    monkeypatch.setattr(subject, "_validate_frozen_audit", lambda *args, **kwargs: ())

    def fake_runtime(path: Path, *, cache: dict, **_: object):
        captured = subject._snapshot(path, label="runtime fixture", cache=cache)
        return {}, (), captured

    monkeypatch.setattr(subject, "_validate_runtime_identity_document", fake_runtime)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = outside / "moved-run"

    def redirect() -> None:
        run_dir.rename(moved)
        _create_windows_junction(run_dir, moved)

    try:
        with pytest.raises(subject.TrainingOrchestrationError, match="symlink|junction"):
            subject.build_training_orchestration_manifest(
                training_run_dirs=[root / "unused-train"],
                screening_run_dirs=[root / "unused-screen"],
                initial_checkpoint_publication_run=root / "unused-publication",
                training_config_path=config,
                vector_benchmark_matrix_path=root / "unused-matrix",
                output_path=run_dir / subject.TRAINING_ORCHESTRATION_FILENAME,
                project_root=root,
                expected_seed=1001,
                expected_num_envs=1,
                _before_publish_hook=redirect,
            )
        assert not (moved / subject.TRAINING_ORCHESTRATION_FILENAME).exists()
    finally:
        if run_dir.exists():
            os.rmdir(run_dir)


def test_committed_runtime_identity_pair_binds_head_files_and_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    sources = [root / "configs" / "a.yaml", root / "src" / "runtime.py"]
    sources[0].parent.mkdir(parents=True)
    sources[1].parent.mkdir(parents=True)
    sources[0].write_text("a: 1\n", encoding="utf-8")
    sources[1].write_text("VALUE = 1\n", encoding="utf-8")
    cache: dict = {}
    rows = []
    for path in sources:
        captured = subject._snapshot(path, label="runtime source", cache=cache)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": captured.size,
                "sha256": captured.sha256,
                "creation_time_utc_ticks": captured.creation_time_utc_ticks,
                "last_write_time_utc_ticks": captured.last_write_time_utc_ticks,
            }
        )
    aggregate = hashlib.sha256(
        json.dumps(
            rows, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    content = hashlib.sha256(
        json.dumps(
            [
                {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
                for row in rows
            ],
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": subject.COMMITTED_RUNTIME_IDENTITY_SCHEMA,
        "git_commit": "b" * 40,
        "file_count": 2,
        "content_sha256": content,
        "aggregate_sha256": aggregate,
        "files": rows,
    }
    monkeypatch.setattr(
        subject,
        "_committed_runtime_paths",
        lambda *args: tuple(row["path"] for row in rows),
    )
    run_dir = root / "run"
    _write_json(run_dir / "committed_runtime_identity.before.json", payload)
    _write_json(run_dir / "committed_runtime_identity.after.json", payload)
    current = sources[0].stat()
    os.utime(
        sources[0],
        ns=(current.st_atime_ns, current.st_mtime_ns + 2_000_000_000),
    )
    records = subject._validate_runtime_identity_pair(
        run_dir,
        project_root=root,
        expected_git_commit="b" * 40,
        cache={},
    )
    assert len(records) == 2

    tampered = json.loads(json.dumps(payload))
    tampered["files"][0]["last_write_time_utc_ticks"] += 1
    tampered["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            tampered["files"],
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(run_dir / "committed_runtime_identity.after.json", tampered)
    with pytest.raises(subject.TrainingOrchestrationError, match="changed during"):
        subject._validate_runtime_identity_pair(
            run_dir,
            project_root=root,
            expected_git_commit="b" * 40,
            cache={},
        )

    bad_content = json.loads(json.dumps(payload))
    bad_content["content_sha256"] = "0" * 64
    _write_json(run_dir / "committed_runtime_identity.before.json", bad_content)
    _write_json(run_dir / "committed_runtime_identity.after.json", bad_content)
    with pytest.raises(subject.TrainingOrchestrationError, match="content SHA-256"):
        subject._validate_runtime_identity_pair(
            run_dir,
            project_root=root,
            expected_git_commit="b" * 40,
            cache={},
        )


def test_wrapper_uses_managed_offline_cli_module() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "build_ppo_training_orchestration.ps1").read_text(
        encoding="utf-8"
    )
    for required in (
        "_invoke_ppo_cli.ps1",
        'RunKind "training_orchestration"',
        'TrainingStage "training-orchestration-prefinal"',
        'CliModule "wlr50_clean.ppo.training_orchestration"',
        'EnvironmentCount 1',
        '"--training-run-dir"',
        '"--screening-run-dir"',
        '"--initial-checkpoint-publication-run"',
        '"--vector-benchmark-matrix"',
        '"--promotion-decision"',
    ):
        assert required in text
    assert "Remove-Item" not in text


def test_cadence_driver_runs_21_guarded_chunks_and_fresh_screenings() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "run_ppo_training_cadence.ps1").read_text(
        encoding="utf-8"
    )
    assert "$RequestedChunkDecisions = 10000" in text
    assert 'Name = "smoke"; Chunks = 1; NumEnvs = $SelectedNumEnvs' in text
    assert 'Name = "phase-curriculum"; Chunks = 10; NumEnvs = 1' in text
    assert 'Name = "full-episode"; Chunks = 10; NumEnvs = 1' in text
    assert 'CliArgs = @("--policy-decisions", [string]$RequestedChunkDecisions)' in text
    assert "whole-PPO-batch cadence contract" in text
    assert "rounding_overrun_policy_decisions" in text
    assert "Invoke-TrainingChunk" in text
    assert "Invoke-FreshScreening -Chunk $Chunk" in text
    assert "[ValidateCount(5, 5)]" in text
    assert "$BaselineEpisodeDir" in text
    assert "$BaselineAggregate" in text
    assert "-CandidateValidationAggregate $AggregatePath" in text
    assert '$FiveSeedScript = Join-Path $PSScriptRoot "evaluate_ppo_checkpoint.ps1"' in text
    assert "Invoke-FiveSeedPairedPromotion -Chunk $Chunk" in text
    assert '"runs\\ppo_phase_v1\\cadence_validation\\step_{0}"' in text
    assert "checkpoint_evaluation_aggregate.json" in text
    assert '$PairedExportScript = Join-Path $PSScriptRoot "export_paired_ppo_evaluation.ps1"' in text
    assert "immutable_history_checkpoint" in text
    assert "checkpoint_initial_zero_residual.pt" in text
    assert "-InitialCheckpointPublicationRun $InitialPublicationRun" in text
    assert '"initial_checkpoint_publication"' in text
    assert "checkpoint_smoke.pt" in text
    assert "creation_runtime_identity_sha256" in text
    assert "CheckpointManifest" in text
    assert "fresh_process_single_episode" in text
    assert "paired_episode_count -ne 5" in text
    assert "minimum_paired_seeds -ne 5" in text
    assert "PromotionGates" in text
    assert 'if ($Stage -cne "full-episode"' in text
    assert "build_ppo_training_orchestration.ps1" in text
    assert "Remove-Item" not in text


def test_public_contract_exports_validator_error_and_constants() -> None:
    assert subject.TRAINING_ORCHESTRATION_SCHEMA == (
        "wlr50_clean.ppo_training_orchestration.v1"
    )
    assert subject.TRAINING_ORCHESTRATION_FILENAME == (
        "training_orchestration_manifest.json"
    )
    assert issubclass(subject.TrainingOrchestrationError, RuntimeError)
    assert callable(subject.validate_training_orchestration_manifest)
