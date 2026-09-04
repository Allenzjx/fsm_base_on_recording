from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wlr50_clean.ppo.phase_snapshots import (
    PHASE_IDS,
    PhaseSnapshotError,
    capture_validated_phase_snapshot_bundle,
    build_phase_snapshots,
    load_validated_phase_snapshot_payload,
    validate_phase_snapshots,
    validated_phase_snapshot_bundle_record,
)


def _observation(tick: int) -> dict:
    joints = (
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    )
    wheels = ("front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")
    bodies = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
    return {
        "physics_tick": tick,
        "base": {"position_w_m": [0, 0, .1], "orientation_wxyz": [1, 0, 0, 0], "linear_velocity_w_m_s": [0, 0, 0], "angular_velocity_w_rad_s": [0, 0, 0]},
        "joints": {name: {"position_deg": 0, "velocity_deg_s": 0} for name in joints},
        "wheels": {name: {"velocity_rad_s": 0, "body_name": body, "center_w_m": [0, 0, .05], "bottom_w_m": [0, 0, 0]} for name, body in zip(wheels, bodies)},
        "contacts": {body: {"contact_class": "GROUND", "ground": {"active": True}, "obstacle": {"active": False}} for body in bodies},
        "obstacle": {"front_x_m": .5, "back_x_m": 2.5, "left_y_m": 1, "right_y_m": -1, "bottom_z_m": 0, "top_z_m": .05},
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _build_snapshot_set(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trial_manifest.json").write_text(json.dumps({"trial_id": "success"}), encoding="utf-8")
    transitions = []
    for index, phase in enumerate(PHASE_IDS):
        transitions.append({"state_id": phase, "to_lifecycle": "EXECUTE_MOTION", "sim_time_s": index / 120})
    _write_jsonl(trial / "state_transitions.jsonl", transitions)
    _write_jsonl(trial / "observation_120hz.jsonl", [_observation(i) for i in range(13)])
    _write_jsonl(
        trial / "full12_commands_120hz.jsonl",
        [{"control_physics_tick": i, "nominal_full12": [0] * 12, "applied_full12": [0] * 12} for i in range(13)],
    )
    _write_jsonl(trial / "leg_crossing_events.jsonl", [])
    out = tmp_path / "snapshots"
    build_phase_snapshots(trial, out)
    return out


def test_build_and_validate_all_phase_snapshots(tmp_path):
    out = _build_snapshot_set(tmp_path)
    manifest = validate_phase_snapshots(out, canonical_root=out)
    assert manifest["phase_count"] == 13
    p08 = json.loads((out / "P08" / "snapshot.json").read_text(encoding="utf-8"))
    assert p08["source_tick"] == 7
    assert p08["phase_history"] == list(PHASE_IDS[:7])
    assert p08["reset_use"] == "TRAINING_RESET_STATE_WRITE"


def test_validated_bundle_record_binds_manifest_and_exact_p01_p13_bytes(tmp_path):
    out = _build_snapshot_set(tmp_path)

    record = validated_phase_snapshot_bundle_record(out, canonical_root=out)

    assert record["snapshot_root"] == str(out.resolve())
    assert record["manifest_path"] == str((out / "manifest.json").resolve())
    assert record["manifest_sha256"] == hashlib.sha256(
        (out / "manifest.json").read_bytes()
    ).hexdigest()
    assert record["phase_count"] == 13
    assert len(record["snapshots"]) == 13
    assert tuple(row["phase"] for row in record["snapshots"]) == PHASE_IDS
    assert all(
        set(row)
        == {
            "phase",
            "snapshot_path",
            "checksum_path",
            "file_sha256",
            "state_sha256",
            "checksum_file_sha256",
        }
        for row in record["snapshots"]
    )
    assert len(record["bundle_sha256"]) == 64
    assert validated_phase_snapshot_bundle_record(out, canonical_root=out) == record


def test_runtime_bundle_validation_rejects_noncanonical_external_root(tmp_path):
    out = _build_snapshot_set(tmp_path)

    with pytest.raises(PhaseSnapshotError, match="differs from canonical"):
        validated_phase_snapshot_bundle_record(out)


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    (
        ("manifest.json", b"{}\n"),
        ("P04/snapshot.json", b"{}\n"),
        ("P04/snapshot.sha256", b"0" * 64 + b"  snapshot.json\n"),
    ),
)
def test_bundle_record_rejects_changed_required_bytes(
    tmp_path, relative_path, replacement
):
    out = _build_snapshot_set(tmp_path)
    (out / relative_path).write_bytes(replacement)

    with pytest.raises(PhaseSnapshotError):
        validated_phase_snapshot_bundle_record(out, canonical_root=out)


@pytest.mark.parametrize(
    "relative_path",
    ("manifest.json", "P07/snapshot.json", "P07/snapshot.sha256"),
)
def test_bundle_record_rejects_missing_required_bytes(tmp_path, relative_path):
    out = _build_snapshot_set(tmp_path)
    (out / relative_path).unlink()

    with pytest.raises(PhaseSnapshotError, match="missing"):
        validated_phase_snapshot_bundle_record(out, canonical_root=out)


def test_bundle_record_rejects_manifest_redirect_even_to_existing_snapshot(tmp_path):
    out = _build_snapshot_set(tmp_path)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshots"][0]["path"] = str((out / "P02" / "snapshot.json").resolve())
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(PhaseSnapshotError, match="does not resolve"):
        validated_phase_snapshot_bundle_record(out, canonical_root=out)


def test_pinned_loader_parses_and_hashes_only_immutable_captured_bytes(
    monkeypatch, tmp_path
):
    out = _build_snapshot_set(tmp_path)
    bundle = capture_validated_phase_snapshot_bundle(out, canonical_root=out)

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: pytest.fail(f"loader reread filesystem path {self}"),
    )
    payload, entry = load_validated_phase_snapshot_payload(bundle, "P08")

    assert payload["fsm_state"] == "P08"
    assert payload["state_sha256"] == entry.state_sha256


@pytest.mark.parametrize(
    "target_name",
    ("root", "manifest", "phase_dir", "snapshot", "sidecar"),
)
def test_bundle_capture_rejects_every_symlink_or_reparse_surface(
    monkeypatch, tmp_path, target_name
):
    out = _build_snapshot_set(tmp_path).resolve()
    targets = {
        "root": out,
        "manifest": out / "manifest.json",
        "phase_dir": out / "P03",
        "snapshot": out / "P03" / "snapshot.json",
        "sidecar": out / "P03" / "snapshot.sha256",
    }
    rejected = targets[target_name]
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: True if self == rejected else original(self),
    )

    with pytest.raises(PhaseSnapshotError, match="symlink or reparse"):
        capture_validated_phase_snapshot_bundle(out, canonical_root=out)


def test_bundle_capture_rejects_external_snapshot_symlink_when_supported(tmp_path):
    out = _build_snapshot_set(tmp_path)
    snapshot = out / "P04" / "snapshot.json"
    external = tmp_path / "external_snapshot.json"
    external.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    try:
        snapshot.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(PhaseSnapshotError, match="symlink|redirect"):
        capture_validated_phase_snapshot_bundle(out, canonical_root=out)


def test_refuses_to_overwrite_snapshot_set(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        build_phase_snapshots(tmp_path / "missing", target)


def test_validation_detects_tampering(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "bad", "phase_count": 0}), encoding="utf-8")
    with pytest.raises(PhaseSnapshotError):
        validate_phase_snapshots(root)
