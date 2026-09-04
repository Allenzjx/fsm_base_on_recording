from __future__ import annotations

import json
import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo import isaac_fsm_backend
from wlr50_clean.ppo.phase_effective_entry import (
    DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
)
from wlr50_clean.ppo.phase_snapshots import capture_validated_phase_snapshot_bundle


def _copied_snapshot_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "ppo_phase_snapshots"
    shutil.copytree(cli.DEFAULT_PHASE_SNAPSHOT_ROOT, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["snapshots"]:
        row["path"] = str((root / row["phase"] / "snapshot.json").resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root.resolve()


def _checkpoint_args(root: Path) -> SimpleNamespace:
    run_dir = root.parent / "managed-run"
    run_dir.mkdir(exist_ok=True)
    files = [
        {
            "path": "pyproject.toml",
            "bytes": 1,
            "sha256": "a" * 64,
            "creation_time_utc_ticks": 1,
            "last_write_time_utc_ticks": 2,
        }
    ]
    content_rows = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in files
    ]
    content_sha256 = hashlib.sha256(
        json.dumps(content_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    aggregate_sha256 = hashlib.sha256(
        json.dumps(files, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (run_dir / "committed_runtime_identity.before.json").write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.committed_runtime_identity.v1",
                "git_commit": "b" * 40,
                "file_count": 1,
                "content_sha256": content_sha256,
                "aggregate_sha256": aggregate_sha256,
                "files": files,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        snapshot_root=root,
        training_config=cli.DEFAULT_TRAINING_CONFIG,
        interface_config=cli.DEFAULT_INTERFACE_CONFIG,
        seed=1001,
        run_dir=run_dir,
    )


def _install_effective_contract_for_snapshot_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    root: Path,
) -> None:
    """Rebind the checked-in calibration payload to this test-only path copy."""

    snapshot = capture_validated_phase_snapshot_bundle(root, canonical_root=root)
    payload = json.loads(DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["derivation"]["phase_snapshot_bundle"] = snapshot.as_record()
    for row in payload["derivation"]["calibration_artifacts"]:
        row["phase_snapshot_bundle_sha256"] = snapshot.bundle_sha256
    unhashed = dict(payload)
    unhashed.pop("contract_sha256")
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    contract_path = (tmp_path / "ppo_phase_effective_entry_v1.json").resolve()
    contract_bytes = (
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    contract_path.write_bytes(contract_bytes)
    contract_path.with_suffix(".sha256").write_bytes(
        (
            f"{hashlib.sha256(contract_bytes).hexdigest()}  "
            f"{contract_path.name}\n"
        ).encode("ascii")
    )
    monkeypatch.setattr(cli, "DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH", contract_path)


def test_checkpoint_contract_binds_manifest_bundle_and_every_snapshot_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: root)
    _install_effective_contract_for_snapshot_copy(monkeypatch, tmp_path, root)

    payload = cli._checkpoint_manifest_payload(
        _checkpoint_args(root), global_step=4096, stage="phase-curriculum"
    )
    bundle = payload["phase_snapshot_bundle"]

    assert payload["phase_snapshot_manifest"] == str(root / "manifest.json")
    assert payload["phase_snapshot_manifest_sha256"] == bundle["manifest_sha256"]
    assert payload["phase_snapshot_bundle_sha256"] == bundle["bundle_sha256"]
    assert bundle["phase_count"] == 13
    assert len(bundle["snapshots"]) == 13
    bound_paths = set(payload["files"])
    assert str(root / "manifest.json") in bound_paths
    assert all(row["snapshot_path"] in bound_paths for row in bundle["snapshots"])
    assert all(row["checksum_path"] in bound_paths for row in bundle["snapshots"])


def test_checkpoint_payload_uses_injected_pin_without_recapturing_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: root)
    _install_effective_contract_for_snapshot_copy(monkeypatch, tmp_path, root)
    args = _checkpoint_args(root)
    pinned = cli._capture_runtime_snapshot_bundle(args)
    monkeypatch.setattr(
        cli,
        "_capture_runtime_snapshot_bundle",
        lambda args: pytest.fail("pinned checkpoint payload must not recapture snapshots"),
    )

    payload = cli._checkpoint_manifest_payload(
        args,
        global_step=4096,
        stage="phase-curriculum",
        pinned_snapshot_bundle=pinned,
    )

    assert payload["phase_snapshot_bundle_sha256"] == pinned.bundle_sha256


def test_training_pin_rejects_a_to_b_to_a_byte_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: root)
    _install_effective_contract_for_snapshot_copy(monkeypatch, tmp_path, root)
    pinned = cli._capture_runtime_snapshot_bundle(_checkpoint_args(root))
    target = root / "P05" / "snapshot.json"
    original = target.read_bytes()
    original_mtime = target.stat().st_mtime_ns
    target.write_bytes(original + b"B")
    target.write_bytes(original)
    os.utime(
        target,
        ns=(original_mtime + 2_000_000_000, original_mtime + 2_000_000_000),
    )

    with pytest.raises(cli.CliError, match="pinned phase snapshot bundle changed"):
        cli._revalidate_pinned_snapshot_bundle(pinned)


def test_backend_loader_reads_each_bundle_file_once_and_never_rereads_selected_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(isaac_fsm_backend, "DEFAULT_PHASE_SNAPSHOT_ROOT", root)
    original_read_bytes = Path.read_bytes
    counts: dict[Path, int] = {}

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        counts[resolved] = counts.get(resolved, 0) + 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    loaded = isaac_fsm_backend._load_validated_phase_snapshot("P09")

    assert loaded.phase_id == "P09"
    assert len(counts) == 27
    assert set(counts.values()) == {1}


def test_backend_loader_revalidates_pinned_bundle_on_every_phase_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(isaac_fsm_backend, "DEFAULT_PHASE_SNAPSHOT_ROOT", root)
    pinned = capture_validated_phase_snapshot_bundle(root, canonical_root=root)
    assert isaac_fsm_backend._load_validated_phase_snapshot(
        "P03", expected_bundle=pinned
    ).phase_id == "P03"
    (root / "P03" / "snapshot.sha256").write_bytes(b"changed")

    with pytest.raises(
        isaac_fsm_backend.IsaacFSMBackendError,
        match="bundle validation failed",
    ):
        isaac_fsm_backend._load_validated_phase_snapshot(
            "P03", expected_bundle=pinned
        )


@pytest.mark.parametrize(
    ("relative_path", "mutation"),
    (
        ("manifest.json", "change"),
        ("P06/snapshot.json", "change"),
        ("P06/snapshot.sha256", "change"),
        ("manifest.json", "remove"),
        ("P06/snapshot.json", "remove"),
        ("P06/snapshot.sha256", "remove"),
    ),
)
def test_current_runtime_contract_rejects_changed_or_missing_snapshot_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    mutation: str,
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: root)
    _install_effective_contract_for_snapshot_copy(monkeypatch, tmp_path, root)
    args = _checkpoint_args(root)
    cli._current_checkpoint_runtime_contract(args)
    target = root / relative_path
    if mutation == "remove":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(cli.CliError, match="pinned phase snapshot bundle changed"):
        cli._current_checkpoint_runtime_contract(args)


def test_runtime_rejects_snapshot_root_other_than_live_loader_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested = _copied_snapshot_bundle(tmp_path)
    live = tmp_path / "different-live-root"
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: live.resolve())

    with pytest.raises(cli.CliError, match="exact bundle used by the live backend"):
        cli._validated_runtime_snapshot_bundle(_checkpoint_args(requested))


def test_checkpoint_consumer_rejects_stale_snapshot_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: root)
    _install_effective_contract_for_snapshot_copy(monkeypatch, tmp_path, root)
    bundle = cli._validated_runtime_snapshot_bundle(_checkpoint_args(root))
    manifest = {
        "phase_snapshot_manifest": bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": "0" * 64,
        "phase_snapshot_bundle": bundle,
    }

    with pytest.raises(cli.CliError, match="phase_snapshot_bundle_sha256"):
        cli._require_manifest_snapshot_contract(
            manifest, bundle, label="test checkpoint manifest"
        )


@pytest.mark.parametrize("mutation", ("missing", "mismatched"))
def test_checkpoint_consumer_requires_all_27_bound_file_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    root = _copied_snapshot_bundle(tmp_path)
    monkeypatch.setattr(cli, "_live_phase_snapshot_root", lambda: root)
    _install_effective_contract_for_snapshot_copy(monkeypatch, tmp_path, root)
    args = _checkpoint_args(root)
    manifest = cli._checkpoint_manifest_payload(
        args, global_step=1, stage="smoke"
    )
    bundle = manifest["phase_snapshot_bundle"]
    selected_path = bundle["snapshots"][0]["snapshot_path"]
    if mutation == "missing":
        manifest["files"].pop(selected_path)
    else:
        manifest["files"][selected_path] = "0" * 64

    with pytest.raises(cli.CliError, match="all 27 phase snapshot files"):
        cli._require_manifest_snapshot_contract(
            manifest, bundle, label="test checkpoint manifest"
        )


def test_phase_curriculum_training_checks_snapshot_bundle_before_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def reject(_args):
        calls.append("snapshot")
        raise cli.CliError("snapshot rejected")

    monkeypatch.setattr(cli, "_capture_runtime_snapshot_bundle", reject)
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: pytest.fail("runner must not be constructed"),
    )
    args = SimpleNamespace(stage="phase-curriculum", snapshot_root=tmp_path)

    with pytest.raises(cli.CliError, match="snapshot rejected"):
        cli._train(args, object())
    assert calls == ["snapshot"]
