from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
import torch

from wlr50_clean.ppo.checkpoint_runtime_capture import (
    CheckpointRuntimeCaptureError,
    capture_checkpoint_bundle,
)


def test_every_live_cli_checkpoint_load_and_provenance_check_uses_capture() -> None:
    """Prevent a new live command from silently reopening caller-owned bytes."""

    from wlr50_clean.ppo import cli

    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    protected = {
        "load_checkpoint_round_trip",
        "validate_resume_checkpoint_provenance",
        "export_inference_actor",
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in protected
    ]
    assert calls
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "captured_bundle" in keywords, (
            f"live call to {call.func.id} at line {call.lineno} omits "
            "captured_bundle"
        )
from wlr50_clean.ppo.rl_library_wrapper import (
    RlLibraryConfigurationError,
    load_checkpoint_round_trip,
    validate_resume_checkpoint_provenance,
)


def _checkpoint_pair(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    checkpoint = tmp_path / "source.pt"
    infos: dict[str, object] = {
        "schema": "wlr50_clean.phase_residual_checkpoint_manifest.v1",
        "stage": "full-episode",
        "training_seed": 1001,
        "global_policy_decisions": 210_000,
    }
    torch.save(
        {
            "model_state_dict": {"actor.0.weight": torch.arange(6).reshape(2, 3)},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(7)}},
                "param_groups": [{"lr": 3.0e-4, "params": [0]}],
            },
            "infos": infos,
        },
        checkpoint,
    )
    digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    manifest = checkpoint.with_name("source_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                **infos,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, manifest, infos


def test_capture_loads_only_private_exact_copy_and_preserves_optimizer(tmp_path):
    checkpoint, manifest, infos = _checkpoint_pair(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="evaluation"
    ) as capture:
        assert capture.private_checkpoint_path.parent.is_relative_to(run_dir)
        assert capture.private_checkpoint_path != checkpoint
        assert capture.private_checkpoint_path.read_bytes() == checkpoint.read_bytes()
        decoded = torch.load(
            capture.private_checkpoint_path, map_location="cpu", weights_only=True
        )
        assert decoded["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(
            3.0e-4
        )
        assert capture.assert_loaded_infos(decoded["infos"]) == infos
        private_directory = capture.private_directory
    assert not private_directory.exists()


def test_rsl_wrapper_loads_private_copy_and_reports_source_provenance(tmp_path):
    checkpoint, manifest, infos = _checkpoint_pair(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class _Runner:
        device = "cpu"

        def __init__(self) -> None:
            self.loaded_path: Path | None = None

        def load(self, path, **_kwargs):
            self.loaded_path = Path(path).resolve()
            return torch.load(path, map_location="cpu", weights_only=True)["infos"]

    runner = _Runner()
    with capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="evaluation"
    ) as capture:
        loaded = load_checkpoint_round_trip(
            runner, checkpoint, captured_bundle=capture
        )
        provenance = validate_resume_checkpoint_provenance(
            checkpoint,
            loaded,
            manifest_path=manifest,
            captured_bundle=capture,
        )
        assert runner.loaded_path == capture.private_checkpoint_path
        assert loaded == infos
        assert provenance.checkpoint_path == checkpoint.resolve()
        assert provenance.manifest_path == manifest.resolve()
        assert provenance.checkpoint_sha256 == capture.checkpoint_sha256
        assert provenance.manifest_sha256 == capture.manifest_sha256


def test_rsl_wrapper_rejects_capture_for_another_source(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    other = tmp_path / "other.pt"
    other.write_bytes(checkpoint.read_bytes())
    with capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="evaluation"
    ) as capture:
        runner = type("Runner", (), {"device": "cpu", "load": lambda *_a, **_k: {}})()
        with pytest.raises(RlLibraryConfigurationError, match="different source"):
            load_checkpoint_round_trip(runner, other, captured_bundle=capture)


def test_capture_detects_source_change(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    capture = capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="train-resume"
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(CheckpointRuntimeCaptureError, match="source checkpoint"):
        capture.assert_sources_unchanged()
    capture.cleanup()


def test_capture_detects_a_to_b_to_a_replacement(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    original = checkpoint.read_bytes()
    original_times = checkpoint.stat()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    capture = capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="video"
    )
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"B")
    os.replace(replacement, checkpoint)
    restored = tmp_path / "restored.pt"
    restored.write_bytes(original)
    os.utime(restored, ns=(original_times.st_atime_ns, original_times.st_mtime_ns))
    os.replace(restored, checkpoint)
    with pytest.raises(CheckpointRuntimeCaptureError, match="source checkpoint"):
        capture.assert_sources_unchanged()
    capture.cleanup()


def test_capture_detects_private_copy_tamper(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    capture = capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="export"
    )
    capture.private_checkpoint_path.write_bytes(b"tamper")
    with pytest.raises(CheckpointRuntimeCaptureError, match="private checkpoint"):
        capture.assert_private_copy_unchanged()
    capture.cleanup()


def test_capture_rejects_manifest_rewriting_embedded_infos(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["training_seed"] = 9999
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(CheckpointRuntimeCaptureError, match="rewrites embedded infos"):
        capture_checkpoint_bundle(
            checkpoint, manifest, run_directory=run_dir, purpose="evaluation"
        )


def test_cleanup_never_recursively_deletes_unexpected_files(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    capture = capture_checkpoint_bundle(
        checkpoint, manifest, run_directory=run_dir, purpose="evaluation"
    )
    unexpected = capture.private_directory / "unexpected.txt"
    unexpected.write_text("leave me", encoding="utf-8")
    capture.cleanup()
    assert unexpected.read_text(encoding="utf-8") == "leave me"


def test_capture_rejects_symlink_component_when_supported(tmp_path):
    checkpoint, manifest, _ = _checkpoint_pair(tmp_path)
    linked = tmp_path / "linked.pt"
    try:
        linked.symlink_to(checkpoint)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(CheckpointRuntimeCaptureError, match="symlink or reparse"):
        capture_checkpoint_bundle(
            linked, manifest, run_directory=run_dir, purpose="evaluation"
        )
