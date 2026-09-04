from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import (
    cli,
    evaluation_artifacts,
    initial_checkpoint,
    rl_library_wrapper,
)
from wlr50_clean.ppo.phase_snapshots import capture_validated_phase_snapshot_bundle


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_windows_junction(link: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {result.stderr or result.stdout}")


def test_initialize_command_is_live_but_publication_is_offline() -> None:
    parser = cli._parser()
    initialize = parser.parse_args(
        [
            "initialize-zero-residual",
            "--run-dir",
            ".",
            "--seed",
            "1001",
            "--num-envs",
            "1",
        ]
    )
    publish = parser.parse_args(
        [
            "publish-initial-zero-residual",
            "--run-dir",
            ".",
            "--seed",
            "1001",
            "--num-envs",
            "1",
            "--source-checkpoint",
            "source.pt",
            "--source-manifest",
            "source_manifest.json",
        ]
    )

    assert initialize.command in cli.LIVE_COMMANDS
    assert publish.command not in cli.LIVE_COMMANDS


def test_live_initializer_saves_loads_and_verifies_only_inside_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "managed-initializer"
    run_dir.mkdir()
    calls: list[str] = []
    runner = object()
    bundle = capture_validated_phase_snapshot_bundle(
        cli.DEFAULT_PHASE_SNAPSHOT_ROOT,
        canonical_root=cli.DEFAULT_PHASE_SNAPSHOT_ROOT,
    )
    profile = SimpleNamespace(seed_train=(1001, 1002))
    env = SimpleNamespace(num_envs=1)
    manifest_payload = {
        "schema": rl_library_wrapper.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "initial_zero_residual",
        "training_seed": 1001,
        "global_policy_decisions": 0,
        "zero_mean_actor_output_layer_verified": True,
    }
    monkeypatch.setattr(cli, "_capture_runtime_snapshot_bundle", lambda args: bundle)
    monkeypatch.setattr(cli, "_revalidate_pinned_snapshot_bundle", lambda value: None)
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: (profile, env, runner, {"runner": "official"}),
    )
    monkeypatch.setattr(
        cli,
        "_checkpoint_manifest_payload",
        lambda *args, **kwargs: dict(manifest_payload),
    )
    monkeypatch.setattr(
        cli,
        "_current_checkpoint_runtime_contract",
        lambda *args, **kwargs: {"runtime": "current"},
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "seed_training_rngs",
        lambda seed: {"seed": seed},
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "capture_training_rng_state",
        lambda *, seed: {"schema": "rng", "seed": seed},
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "optimizer_learning_rate",
        lambda value: 3.0e-4,
    )
    zero_checks = iter((True, True))
    monkeypatch.setattr(
        rl_library_wrapper,
        "initialize_zero_mean_actor",
        lambda value: calls.append("initialize"),
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "zero_mean_actor_output_layer_verified",
        lambda value: next(zero_checks),
    )

    def save(value, path, *, manifest):
        calls.append("save")
        checkpoint = Path(path)
        checkpoint.write_bytes(b"run-local-checkpoint")
        sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
        _write_json(
            sidecar,
            {
                **manifest,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
        )
        return checkpoint.resolve(), sidecar.resolve()

    monkeypatch.setattr(rl_library_wrapper, "save_checkpoint_with_manifest", save)
    capture = SimpleNamespace(
        checkpoint_sha256=hashlib.sha256(b"run-local-checkpoint").hexdigest(),
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(cli, "_pin_live_checkpoint", lambda *args, **kwargs: capture)
    round_trip = dict(manifest_payload)
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_checkpoint_round_trip",
        lambda *args, **kwargs: calls.append("load") or round_trip,
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "validate_resume_checkpoint_provenance",
        lambda *args, **kwargs: SimpleNamespace(global_policy_decisions=0),
    )
    args = SimpleNamespace(
        num_envs=1,
        seed_set="train",
        stage="smoke",
        checkpoint=None,
        checkpoint_manifest=None,
        seed=1001,
        run_dir=run_dir,
    )

    assert cli._initialize_zero_residual(args, object()) == 0
    assert calls == ["initialize", "save", "load"]
    result = json.loads((run_dir / "initial_checkpoint_result.json").read_text())
    assert result["save_load_round_trip"] is True
    assert result["zero_mean_actor_output_layer_verified_after_load"] is True
    assert Path(result["checkpoint"]).parent == run_dir.resolve()
    assert not (tmp_path / "outputs").exists()


def _fake_evidence(checkpoint: Path, manifest: Path) -> initial_checkpoint.InitialCheckpointEvidence:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return initial_checkpoint.InitialCheckpointEvidence(
        checkpoint_path=checkpoint.resolve(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        manifest_path=manifest.resolve(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        manifest=payload,
        creation_run_kind="initial_checkpoint",
        creation_run_directory=checkpoint.parent,
        creation_run_manifest={"path": "run_manifest.json"},
        checkpoint_bytes=checkpoint.read_bytes(),
        manifest_bytes=manifest.read_bytes(),
    )


def test_publication_is_no_clobber_and_reuses_only_an_identical_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "runs" / "source" / initial_checkpoint.INITIAL_CHECKPOINT_NAME
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified-zero-checkpoint")
    source_manifest = source.with_name(initial_checkpoint.INITIAL_MANIFEST_NAME)
    source_payload = {
        "stage": "initial_zero_residual",
        "checkpoint_path": str(source.resolve()),
        "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "creation_runtime_identity_path": "creator",
    }
    _write_json(source_manifest, source_payload)

    def fake_validate(checkpoint, manifest, **kwargs):
        return _fake_evidence(Path(checkpoint), Path(manifest))

    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        fake_validate,
    )
    output = root / "outputs" / "ppo_phase_v1"
    first = initial_checkpoint.publish_initial_zero_residual_checkpoint(
        source_checkpoint=source,
        source_manifest=source_manifest,
        output_root=output,
        project_root=root,
        expected_seed=1001,
    )
    assert first.reused_existing is False
    original_checkpoint = first.checkpoint_path.read_bytes()
    original_manifest = first.manifest_path.read_bytes()
    published_payload = json.loads(original_manifest)
    assert published_payload["checkpoint_path"] == str(first.checkpoint_path)

    second = initial_checkpoint.publish_initial_zero_residual_checkpoint(
        source_checkpoint=first.checkpoint_path,
        source_manifest=first.manifest_path,
        output_root=output,
        project_root=root,
        expected_seed=1001,
    )
    assert second.reused_existing is True
    assert first.checkpoint_path.read_bytes() == original_checkpoint
    assert first.manifest_path.read_bytes() == original_manifest

    source.write_bytes(b"different-but-otherwise-validated")
    source_payload["checkpoint_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_json(source_manifest, source_payload)
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="differs"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=source_manifest,
            output_root=output,
            project_root=root,
            expected_seed=1001,
        )
    assert first.checkpoint_path.read_bytes() == original_checkpoint
    assert first.manifest_path.read_bytes() == original_manifest


def test_publication_rejects_incomplete_canonical_pair_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )
    output = root / "outputs" / "ppo_phase_v1"
    canonical = output / "checkpoints" / initial_checkpoint.INITIAL_CHECKPOINT_NAME
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"partial-canonical")

    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="incomplete"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=manifest,
            output_root=output,
            project_root=root,
        )
    assert canonical.read_bytes() == b"partial-canonical"
    assert not canonical.with_name(initial_checkpoint.INITIAL_MANIFEST_NAME).exists()


def test_publication_recovers_exact_checkpoint_only_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )
    output = root / "outputs" / "ppo_phase_v1"
    canonical = output / "checkpoints" / initial_checkpoint.INITIAL_CHECKPOINT_NAME
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(source.read_bytes())

    recovered = initial_checkpoint.publish_initial_zero_residual_checkpoint(
        source_checkpoint=source,
        source_manifest=manifest,
        output_root=output,
        project_root=root,
    )

    assert recovered.reused_existing is False
    assert canonical.read_bytes() == source.read_bytes()
    assert recovered.manifest_path.is_file()


def test_exclusive_publication_uses_stable_object_identity_and_cleans_staging(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published.bin"
    payload = b"complete-payload"

    owned = initial_checkpoint._publish_bytes_exclusive(destination, payload)

    assert owned.path == destination
    assert destination.read_bytes() == payload
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


def test_exclusive_publication_cleans_private_staging_after_early_fsync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "published.bin"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(initial_checkpoint.os, "fsync", fail_fsync)
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="fsync"):
        initial_checkpoint._publish_bytes_exclusive(destination, b"payload")

    assert list(tmp_path.iterdir()) == []


def test_publication_rejects_redirected_output_ancestry_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "outputs").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )

    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="symlink|junction"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=manifest,
            output_root=root / "outputs" / "ppo_phase_v1",
            project_root=root,
        )
    assert not (outside / "ppo_phase_v1").exists()


def test_publication_rejects_windows_reparse_attribute_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    output = root / "outputs" / "ppo_phase_v1"
    (root / "outputs").mkdir()
    real_stat = os.stat

    class MarkedStat:
        st_file_attributes = 0x400

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def marked_stat(path, *args, **kwargs):
        status = real_stat(path, *args, **kwargs)
        if Path(path) == root / "outputs":
            return MarkedStat(status)
        return status

    monkeypatch.setattr(initial_checkpoint.os, "stat", marked_stat)
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="symlink|junction"):
        initial_checkpoint._validate_publication_paths(
            root=root,
            output=output,
            checkpoint=output / "checkpoints" / initial_checkpoint.INITIAL_CHECKPOINT_NAME,
            manifest=output / "checkpoints" / initial_checkpoint.INITIAL_MANIFEST_NAME,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_publication_hook_rejects_output_parent_replaced_by_junction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "outputs").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = outside / "redirected"
    redirected.mkdir()
    moved = outside / "original-outputs"
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )

    def redirect() -> None:
        (root / "outputs").rename(moved)
        _create_windows_junction(root / "outputs", redirected)

    try:
        with pytest.raises(
            initial_checkpoint.InitialCheckpointError, match="symlink|junction"
        ):
            initial_checkpoint.publish_initial_zero_residual_checkpoint(
                source_checkpoint=source,
                source_manifest=manifest,
                output_root=root / "outputs" / "ppo_phase_v1",
                project_root=root,
                _before_publish_hook=redirect,
            )
        assert not (redirected / "ppo_phase_v1").exists()
    finally:
        link = root / "outputs"
        if link.exists():
            os.rmdir(link)


def test_publication_detects_source_mutation_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "source.pt"
    source.write_bytes(b"source-before")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )

    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="identity changed"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=manifest,
            output_root=root / "outputs" / "ppo_phase_v1",
            project_root=root,
            _before_publish_hook=lambda: source.write_bytes(b"source-after-longer"),
        )
    assert not (root / "outputs").exists()


def test_second_atomic_publication_failure_leaves_exact_manifest_and_retry_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )
    real_publish = initial_checkpoint._publish_bytes_exclusive
    calls = 0

    def fail_second(path: Path, payload: bytes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise initial_checkpoint.InitialCheckpointError("second atomic write failed")
        return real_publish(path, payload)

    monkeypatch.setattr(initial_checkpoint, "_publish_bytes_exclusive", fail_second)
    output = root / "outputs" / "ppo_phase_v1"
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="second atomic"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=manifest,
            output_root=output,
            project_root=root,
        )
    canonical = output / "checkpoints" / initial_checkpoint.INITIAL_CHECKPOINT_NAME
    canonical_manifest = output / "checkpoints" / initial_checkpoint.INITIAL_MANIFEST_NAME
    assert not canonical.exists()
    assert canonical_manifest.is_file()

    monkeypatch.setattr(initial_checkpoint, "_publish_bytes_exclusive", real_publish)
    recovered = initial_checkpoint.publish_initial_zero_residual_checkpoint(
        source_checkpoint=source,
        source_manifest=manifest,
        output_root=output,
        project_root=root,
    )
    assert recovered.reused_existing is False
    assert canonical.read_bytes() == source.read_bytes()
    assert canonical_manifest.is_file()


def test_final_validation_failure_keeps_exact_pair_for_safe_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})

    def validate(checkpoint, sidecar, **kwargs):
        if Path(checkpoint).resolve() != source.resolve():
            raise initial_checkpoint.InitialCheckpointError("final validation failed")
        return _fake_evidence(source, manifest)

    monkeypatch.setattr(
        initial_checkpoint, "validate_initial_zero_residual_checkpoint", validate
    )
    output = root / "outputs" / "ppo_phase_v1"
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="final validation"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=manifest,
            output_root=output,
            project_root=root,
        )
    canonical = output / "checkpoints" / initial_checkpoint.INITIAL_CHECKPOINT_NAME
    canonical_manifest = output / "checkpoints" / initial_checkpoint.INITIAL_MANIFEST_NAME
    assert canonical.read_bytes() == source.read_bytes()
    assert canonical_manifest.is_file()

    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda checkpoint, sidecar, **kwargs: _fake_evidence(
            Path(checkpoint), Path(sidecar)
        ),
    )
    recovered = initial_checkpoint.publish_initial_zero_residual_checkpoint(
        source_checkpoint=source,
        source_manifest=manifest,
        output_root=output,
        project_root=root,
    )
    assert recovered.reused_existing is True


def test_validation_failure_never_deletes_a_foreign_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    source = root / "source.pt"
    source.write_bytes(b"source")
    manifest = root / "source_manifest.json"
    _write_json(manifest, {"checkpoint_path": str(source.resolve())})
    output = root / "outputs" / "ppo_phase_v1"
    canonical = output / "checkpoints" / initial_checkpoint.INITIAL_CHECKPOINT_NAME
    foreign = b"foreign-replacement-must-survive"

    def validate(checkpoint, sidecar, **kwargs):
        selected = Path(checkpoint).resolve()
        if selected == canonical.resolve():
            replacement = canonical.with_suffix(".foreign")
            replacement.write_bytes(foreign)
            replacement.replace(canonical)
            raise initial_checkpoint.InitialCheckpointError("injected final failure")
        return _fake_evidence(source, manifest)

    monkeypatch.setattr(
        initial_checkpoint, "validate_initial_zero_residual_checkpoint", validate
    )
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="injected"):
        initial_checkpoint.publish_initial_zero_residual_checkpoint(
            source_checkpoint=source,
            source_manifest=manifest,
            output_root=output,
            project_root=root,
        )
    assert canonical.read_bytes() == foreign
    assert (output / "checkpoints" / initial_checkpoint.INITIAL_MANIFEST_NAME).is_file()


def test_offline_validator_requires_embedded_infos_and_exact_zero_actor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path.resolve()
    checkpoint = root / "checkpoint.pt"
    manifest_path = root / "checkpoint_manifest.json"
    infos = {
        "schema": rl_library_wrapper.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "initial_zero_residual",
        "training_seed": 1001,
        "global_policy_decisions": 0,
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "zero_mean_actor_output_layer_verified": True,
        "source_git_commit": "b" * 40,
        "committed_runtime_content_sha256": "d" * 64,
        "creation_runtime_identity_path": str(root / "runtime.before.json"),
        "creation_runtime_identity_sha256": "e" * 64,
        "optimizer_learning_rate": 3.0e-4,
        "training_rng_seed_evidence": {"seed": 1001},
        "training_rng_state": {
            "schema": "wlr50_clean.training_rng_state.v1",
            "seed": 1001,
        },
    }

    def write(weight: float) -> None:
        torch.save(
            {
                "actor_state_dict": {
                    "actor.mlp.0.weight": torch.ones((8, 3)),
                    "actor.mlp.0.bias": torch.zeros(8),
                    "actor.mlp.2.weight": torch.full((12, 8), weight),
                    "actor.mlp.2.bias": torch.zeros(12),
                },
                "infos": infos,
            },
            checkpoint,
        )
        _write_json(
            manifest_path,
            {
                **infos,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
        )

    monkeypatch.setattr(initial_checkpoint, "git_head", lambda root: "b" * 40)
    monkeypatch.setattr(
        initial_checkpoint, "_validate_runtime_contract", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        initial_checkpoint,
        "_validate_creation_run",
        lambda *args, **kwargs: (
            "initial_checkpoint",
            root / "runs" / "creator",
            {"path": "run_manifest.json"},
        ),
    )

    write(0.0)
    evidence = initial_checkpoint.validate_initial_zero_residual_checkpoint(
        checkpoint, manifest_path, project_root=root, expected_seed=1001
    )
    assert evidence.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    write(0.25)
    with pytest.raises(initial_checkpoint.InitialCheckpointError, match="not exact zero"):
        initial_checkpoint.validate_initial_zero_residual_checkpoint(
            checkpoint, manifest_path, project_root=root, expected_seed=1001
        )


def test_train_no_longer_creates_initial_checkpoint_and_preflight_blocks_missing_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import inspect

    source = inspect.getsource(cli._train)
    assert "initialize_zero_mean_actor" not in source
    assert "if not initial_path.exists()" not in source

    monkeypatch.setattr(
        initial_checkpoint,
        "validate_initial_zero_residual_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            initial_checkpoint.InitialCheckpointError("missing canonical pair")
        ),
    )
    with pytest.raises(cli.CliError, match="initialize_zero_residual_checkpoint.ps1"):
        cli._require_canonical_initial_checkpoint(
            SimpleNamespace(checkpoint=None, seed=1001)
        )
    dispatch_source = inspect.getsource(cli._dispatch_live)
    assert dispatch_source.index("_require_canonical_initial_checkpoint") < (
        dispatch_source.index("from isaaclab.app import AppLauncher")
    )


def test_final_evaluation_accepts_dedicated_creator_only_for_initial_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "initial-run"
    run_dir.mkdir()
    identity_path = run_dir / "committed_runtime_identity.before.json"
    identity_path.write_bytes(b"runtime")
    identity_record = {
        "path": str(identity_path.resolve()),
        "bytes": len(b"runtime"),
        "sha256": hashlib.sha256(b"runtime").hexdigest(),
    }
    run_payload = {
        "schema": evaluation_artifacts.RUN_MANIFEST_SCHEMA,
        "lifecycle": "SUCCEEDED",
        "exit_code": 0,
        "run_kind": "initial_checkpoint",
        "entrypoint": "wlr50_clean.ppo.cli",
        "subcommand": "initialize-zero-residual",
        "immutable_run_directory": True,
        "run_dir": str(run_dir.resolve()),
        "identity": {
            "training_stage": "initialize-zero-residual",
            "environment_count": 1,
        },
    }
    _write_json(run_dir / "run_manifest.json", run_payload)
    runtime = {"git_commit": "b" * 40, "content_sha256": "d" * 64}
    monkeypatch.setattr(
        evaluation_artifacts,
        "_validate_worker_runtime_identity",
        lambda *args, **kwargs: (identity_record, identity_record, runtime),
    )
    manifest = {
        "stage": "initial_zero_residual",
        "source_git_commit": "b" * 40,
        "committed_runtime_content_sha256": "d" * 64,
        "creation_runtime_identity_path": str(identity_path.resolve()),
        "creation_runtime_identity_sha256": identity_record["sha256"],
    }

    records, returned_runtime = evaluation_artifacts._validate_checkpoint_creation_runtime(
        manifest, role="checkpoint_initial"
    )
    assert records[1] == identity_record
    assert returned_runtime == runtime
    with pytest.raises(
        evaluation_artifacts.EvaluationArtifactError, match="allowed finalized"
    ):
        evaluation_artifacts._validate_checkpoint_creation_runtime(
            manifest, role="checkpoint_smoke"
        )
