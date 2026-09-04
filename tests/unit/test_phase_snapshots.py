from __future__ import annotations

import hashlib
import json
import copy
from collections import Counter
from pathlib import Path

import pytest

from wlr50_clean.ppo import phase_snapshots as phase_snapshots_subject
from wlr50_clean.ppo.phase_snapshots import (
    MANIFEST_SCHEMA,
    PHASE_IDS,
    SNAPSHOT_SCHEMA,
    PhaseSnapshotError,
    capture_validated_phase_snapshot_bundle,
    build_phase_snapshots,
    load_validated_phase_snapshot_payload,
    validate_phase_snapshot_payload_contract,
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(tick: int) -> dict:
    servos = (
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
    )
    wheels = (
        "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle"
    )
    zero12 = [0.0] * 12
    ack = {
        "schema": "wlr50_clean.atomic_full12_ack.v1",
        "physics_tick": 180 + tick,
        "physics_dt_s": 1.0 / 120.0,
        "write_count": 181 + tick,
        "articulation_writes_this_call": 1,
        "canonical_order": list(servos + wheels),
        "requested_full12": zero12,
        "applied_full12": zero12,
        "drive_target_full12": zero12,
        "native_drive_target_full12": zero12,
        "drive_feedback_bias_requested_full12": zero12,
        "drive_feedback_bias_realized_full12": zero12,
        "drive_feedback_final_slew_limit_deg_per_tick": 1.25,
        "command_was_clamped": False,
        "servo_applied_drive_command_deg": [0.0] * 8,
        "servo_native_drive_command_deg": [0.0] * 8,
        "servo_tracking_compensation_deg": [0.0] * 8,
        "servo_nominal_target_reached": [True] * 8,
        "servo_tracking_active": [False] * 8,
        "tracking_servo_names": [],
        "servo_tracking_feedback_sample_tick": 180 + tick,
        "servo_tracking_feedback_sampled": False,
        "servo_joint_ids": list(range(8)),
        "wheel_joint_ids": list(range(8, 12)),
        "servo_target_physical_rad": [0.0] * 8,
        "wheel_target_physical_rad_s": [-0.0, 0.0, -0.0, 0.0],
        "motion_start_skew_s": 0.0,
    }
    return {
        "control_physics_tick": tick,
        "state_id": PHASE_IDS[tick],
        "nominal_full12": zero12,
        "applied_full12": zero12,
        "drive_target_full12": zero12,
        "native_drive_target_full12": zero12,
        "drive_feedback_bias_requested_full12": zero12,
        "drive_feedback_bias_realized_full12": zero12,
        "tracking_servo_names": [],
        "atomic_ack": ack,
    }


def _write_source_trial(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    transitions = []
    for index, phase in enumerate(PHASE_IDS):
        transitions.append({"state_id": phase, "to_lifecycle": "EXECUTE_MOTION", "sim_time_s": index / 120})
    _write_jsonl(trial / "state_transitions.jsonl", transitions)
    _write_jsonl(trial / "observation_120hz.jsonl", [_observation(i) for i in range(13)])
    _write_jsonl(
        trial / "full12_commands_120hz.jsonl",
        [_command(i) for i in range(13)],
    )
    _write_jsonl(trial / "leg_crossing_events.jsonl", [])
    artifact_rows = {
        "observation": ("observation_120hz.jsonl"),
        "command": ("full12_commands_120hz.jsonl"),
        "transition": ("state_transitions.jsonl"),
        "leg_crossing": ("leg_crossing_events.jsonl"),
    }
    manifest = {
        "trial_id": "success",
        "physics_hz": 120.0,
        "settle_ticks": 180,
        "environment_initialization": {
            "records": [
                {"joint_name": name, "standing_pose_deg": 0.0}
                for name in (
                    "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
                    "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
                )
            ]
        },
        "artifact_files": {
            role: {
                "path": name,
                "bytes": (trial / name).stat().st_size,
                "sha256": _sha(trial / name),
            }
            for role, name in artifact_rows.items()
        },
    }
    (trial / "trial_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return trial


def _build_snapshot_set(tmp_path):
    trial = _write_source_trial(tmp_path)
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
    assert p08["schema"] == SNAPSHOT_SCHEMA
    assert p08["source_command"]["expected_atomic_ack"][
        "drive_target_full12"
    ] == [0.0] * 12
    assert set(p08["source_artifacts"]) == {
        "trial_manifest",
        "command",
        "observation",
        "transition",
        "leg_crossing",
    }
    assert manifest["schema"] == MANIFEST_SCHEMA


def test_generated_bundle_bytes_are_git_lf_normalization_stable(tmp_path):
    out = _build_snapshot_set(tmp_path)
    paths = [out / "manifest.json"]
    for phase in PHASE_IDS:
        paths.extend(
            (out / phase / "snapshot.json", out / phase / "snapshot.sha256")
        )

    assert len(paths) == 27
    for path in paths:
        payload = path.read_bytes()
        simulated_git_lf_bytes = payload.replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )
        assert b"\r" not in payload, path
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n"), path
        assert simulated_git_lf_bytes == payload, path
        assert hashlib.sha256(simulated_git_lf_bytes).digest() == hashlib.sha256(
            payload
        ).digest(), path


def test_builder_opens_each_source_artifact_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial = _write_source_trial(tmp_path)
    source_paths = {
        trial / name
        for name in (
            "trial_manifest.json",
            "full12_commands_120hz.jsonl",
            "observation_120hz.jsonl",
            "state_transitions.jsonl",
            "leg_crossing_events.jsonl",
        )
    }
    open_counts: Counter[Path] = Counter()
    original_open = Path.open

    def counted_open(path: Path, *args, **kwargs):
        if path in source_paths:
            open_counts[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    build_phase_snapshots(trial, tmp_path / "snapshots")

    assert open_counts == Counter({path: 1 for path in source_paths})


@pytest.mark.parametrize(
    "target_kind",
    (
        "ancestor",
        "root",
        "trial_manifest",
        "command",
        "observation",
        "transition",
        "leg_crossing",
    ),
)
def test_builder_rejects_every_source_reparse_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    trial = _write_source_trial(tmp_path).resolve()
    targets = {
        "ancestor": trial.parent,
        "root": trial,
        "trial_manifest": trial / "trial_manifest.json",
        "command": trial / "full12_commands_120hz.jsonl",
        "observation": trial / "observation_120hz.jsonl",
        "transition": trial / "state_transitions.jsonl",
        "leg_crossing": trial / "leg_crossing_events.jsonl",
    }
    rejected = targets[target_kind]
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: True if self == rejected else original_is_symlink(self),
    )

    with pytest.raises(PhaseSnapshotError, match="symlink or reparse"):
        build_phase_snapshots(trial, tmp_path / "snapshots")
    assert not (tmp_path / "snapshots").exists()


def test_builder_rejects_source_symlink_when_supported(tmp_path: Path) -> None:
    trial = _write_source_trial(tmp_path)
    transition = trial / "state_transitions.jsonl"
    external = tmp_path / "external_transitions.jsonl"
    external.write_bytes(transition.read_bytes())
    transition.unlink()
    try:
        transition.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(PhaseSnapshotError, match="symlink|reparse|redirect"):
        build_phase_snapshots(trial, tmp_path / "snapshots")
    assert not (tmp_path / "snapshots").exists()


def test_builder_detects_mutation_between_source_capture_and_parse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial = _write_source_trial(tmp_path)
    transition = trial / "state_transitions.jsonl"
    original_capture = phase_snapshots_subject._capture_source_bytes_once
    mutated = False

    def capture_then_mutate(path: Path, **kwargs):
        nonlocal mutated
        payload = original_capture(path, **kwargs)
        if path == transition and not mutated:
            transition.write_bytes(payload + b"\n")
            mutated = True
        return payload

    monkeypatch.setattr(
        phase_snapshots_subject,
        "_capture_source_bytes_once",
        capture_then_mutate,
    )

    with pytest.raises(PhaseSnapshotError, match="changed during immutable capture"):
        build_phase_snapshots(trial, tmp_path / "snapshots")
    assert mutated is True
    assert not (tmp_path / "snapshots").exists()


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


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_drive_target",
        "changed_drive_target",
        "missing_retiring_state",
        "changed_source_file_sha",
        "changed_post_tracking_flag",
    ),
)
def test_v2_source_replay_contract_fails_closed_on_every_critical_surface(
    tmp_path: Path, mutation: str
) -> None:
    out = _build_snapshot_set(tmp_path)
    payload = json.loads((out / "P03" / "snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest_row = next(row for row in manifest["snapshots"] if row["phase"] == "P03")
    changed = copy.deepcopy(payload)
    source = changed["source_command"]
    if mutation == "missing_drive_target":
        source["expected_atomic_ack"].pop("drive_target_full12")
    elif mutation == "changed_drive_target":
        source["expected_atomic_ack"]["drive_target_full12"][0] = 1.0
    elif mutation == "missing_retiring_state":
        source["mapper_pre_state"].pop("retiring_stale_bias")
    elif mutation == "changed_source_file_sha":
        changed["source_artifacts"]["command"]["sha256"] = "0" * 64
    else:
        source["mapper_post_state"]["tracking_active"][0] = True

    with pytest.raises(PhaseSnapshotError):
        validate_phase_snapshot_payload_contract(
            changed,
            "P03",
            manifest_row=manifest_row,
            manifest_source_artifacts=manifest["source_artifacts"],
        )


def test_builder_rejects_semantically_unchanged_but_hash_changed_transition_source(
    tmp_path: Path,
) -> None:
    first = _build_snapshot_set(tmp_path)
    trial = first.parent / "trial"
    transition = trial / "state_transitions.jsonl"
    transition.write_bytes(transition.read_bytes() + b"\n")

    with pytest.raises(PhaseSnapshotError, match="transition hash/size"):
        build_phase_snapshots(trial, tmp_path / "second-snapshots")
