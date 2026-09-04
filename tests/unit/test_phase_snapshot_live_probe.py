from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo import phase_snapshot_live_probe as probe_subject
from wlr50_clean.ppo.phase_snapshot_live_probe import (
    ATTEMPTS_PER_PHASE,
    PROBE_PHASES,
    PhaseSnapshotLiveProbeError,
    _attempt_passed,
    observation_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(phase: str = "P10") -> dict[str, object]:
    return json.loads(
        (
            PROJECT_ROOT
            / "reference"
            / "ppo_phase_snapshots"
            / phase
            / "snapshot.json"
        ).read_text(encoding="utf-8")
    )


def _pair(active: bool) -> SimpleNamespace:
    return SimpleNamespace(
        active=active,
        pair_verified=True,
        normal_force_n=1.0 if active else 0.0,
        force_w_n=(1.0, 0.0, 0.0) if active else (0.0, 0.0, 0.0),
        active_history=(active, active, active),
    )


def _matching_observation(snapshot: dict[str, object]) -> SimpleNamespace:
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER

    root = snapshot["root_state"]
    joint = snapshot["joint_state"]
    wheel = snapshot["wheel_state"]
    geometry = snapshot["obstacle_relative_geometry"]
    contact_state = snapshot["contact_state"]
    joints = {
        name: SimpleNamespace(
            position_deg=joint["logical_position_deg"][index],
            velocity_deg_s=joint["logical_velocity_deg_s"][index],
        )
        for index, name in enumerate(SERVO_ORDER)
    }
    wheels = {}
    contacts = {}
    for index, name in enumerate(WHEEL_ORDER):
        body_name = f"{name}_body"
        expected = contact_state[name]
        wheels[name] = SimpleNamespace(
            body_name=body_name,
            velocity_rad_s=wheel["logical_velocity_rad_s"][index],
            center_w_m=geometry["wheel_centers_w_m"][name],
            bottom_w_m=geometry["wheel_bottoms_w_m"][name],
        )
        contacts[body_name] = SimpleNamespace(
            contact_class=expected["class"],
            ground=_pair(bool(expected["ground_active"])),
            obstacle=_pair(bool(expected["obstacle_active"])),
        )
    return SimpleNamespace(
        physics_tick=0,
        simulation_time_s=0.0,
        base=SimpleNamespace(
            position_w_m=root["position_w_m"],
            orientation_wxyz=root["orientation_wxyz"],
            linear_velocity_w_m_s=root["linear_velocity_w_m_s"],
            angular_velocity_w_rad_s=root["angular_velocity_w_rad_s"],
        ),
        joints=joints,
        wheels=wheels,
        contacts=contacts,
    )


def test_probe_covers_every_non_p01_phase_twice() -> None:
    assert PROBE_PHASES == tuple(f"P{index:02d}" for index in range(2, 14))
    assert ATTEMPTS_PER_PHASE == 2


def test_observation_diagnostics_accepts_only_exact_contacts_and_state() -> None:
    snapshot = _snapshot()
    observation = _matching_observation(snapshot)
    result = observation_diagnostics(observation, snapshot)
    assert result["physical_state_within_production_tolerances"] is True
    assert result["exact_contacts_match"] is True
    assert result["contact_mismatches"] == []

    first_wheel = next(iter(observation.wheels.values()))
    observation.contacts[first_wheel.body_name].contact_class = "AIR"
    observation.contacts[first_wheel.body_name].ground = _pair(False)
    observation.contacts[first_wheel.body_name].obstacle = _pair(False)
    failed = observation_diagnostics(observation, snapshot)
    assert failed["physical_state_within_production_tolerances"] is True
    assert failed["exact_contacts_match"] is False
    assert failed["contact_mismatches"]


def test_attempt_gate_fails_closed_on_exception_contact_or_extra_step() -> None:
    diagnostic = observation_diagnostics(_matching_observation(_snapshot()), _snapshot())
    row = {
        "phase": "P10",
        "reset_completed": True,
        "physics_steps_during_reset": 180,
        "snapshot_state_write": {
            "root_pose_writes": 1,
            "root_velocity_writes": 1,
            "joint_state_writes": 1,
            "simulation_forward_syncs": 1,
            "physics_steps": 0,
        },
        "observation_diagnostics": diagnostic,
        "clocks": {
            "backend_episode_tick": 0,
            "controller_frame_state_id": "P10",
            "controller_frame_physics_tick": 0,
        },
    }
    assert _attempt_passed(row) is True

    assert _attempt_passed({**row, "reset_completed": False}) is False
    assert _attempt_passed({**row, "physics_steps_during_reset": 181}) is False
    assert _attempt_passed(
        {
            **row,
            "observation_diagnostics": {
                **diagnostic,
                "exact_contacts_match": False,
            },
        }
    ) is False


def test_cli_and_wrapper_bind_probe_to_one_live_environment() -> None:
    arguments = cli._parser().parse_args(
        [
            "phase-snapshot-live-probe",
            "--run-dir",
            str(PROJECT_ROOT),
            "--seed",
            "1001",
            "--num-envs",
            "1",
        ]
    )
    assert arguments.command == "phase-snapshot-live-probe"
    assert "phase-snapshot-live-probe" in cli.LIVE_COMMANDS

    wrapper = (
        PROJECT_ROOT / "scripts" / "run_phase_snapshot_live_probe.ps1"
    ).read_text(encoding="utf-8")
    assert '-Subcommand "phase-snapshot-live-probe"' in wrapper
    assert '-RunKind "phase_snapshot_live_probe"' in wrapper
    assert "-EnvironmentCount 1" in wrapper
    assert "-ReturnFinalizedEvidenceFailure" in wrapper

    common = (PROJECT_ROOT / "scripts" / "_invoke_ppo_cli.ps1").read_text(
        encoding="utf-8"
    )
    assert '$RunKindValue -ceq "phase_snapshot_live_probe"' in common
    assert '$SubcommandValue -ceq "phase-snapshot-live-probe"' in common
    assert "$null -ne $AuthoritativeLiveExitCode" in common


def _write_managed_prechecks(run_dir: Path) -> None:
    (run_dir / "committed_runtime_identity.before.json").write_text(
        json.dumps({"schema": "wlr50_clean.committed_runtime_identity.v1"}),
        encoding="utf-8",
    )
    (run_dir / "frozen_hashes.before.json").write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
                "passed": True,
                "mismatches": [],
            }
        ),
        encoding="utf-8",
    )


class _FakeBundle:
    def __init__(self, root: Path) -> None:
        self.snapshot_root = root.resolve()

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "wlr50_clean.phase_snapshot_bundle.v1",
            "snapshot_root": str(self.snapshot_root),
        }


def _patch_probe_snapshot_loader(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    from wlr50_clean.ppo import phase_snapshots

    entry = SimpleNamespace(
        source_tick=1,
        snapshot_path=root / "snapshot.json",
        file_sha256="a" * 64,
        state_sha256="b" * 64,
    )
    monkeypatch.setattr(
        phase_snapshots,
        "load_validated_phase_snapshot_payload",
        lambda bundle, phase: ({}, entry),
    )


def test_probe_infrastructure_initialization_failure_is_fatal_but_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_managed_prechecks(tmp_path)
    bundle = _FakeBundle(tmp_path)

    def fail_dependencies(book: object) -> object:
        raise RuntimeError("dependency initialization failed")

    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", fail_dependencies)
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(), run_dir=tmp_path, seed=1001, snapshot_bundle=bundle
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["status"] == "FAILED"
    assert report["passed"] is False
    assert report["complete"] is False
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 0
    assert not (tmp_path / "live_command_result.json").exists()


def test_backend_bundle_revalidation_failure_cannot_be_returnable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)

    class IntegrityFailingBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._phase_snapshot_integrity_failed = True
            raise isaac_fsm_backend.IsaacFSMBackendError(
                "phase snapshot bundle validation failed"
            )

    monkeypatch.setattr(
        isaac_fsm_backend, "IsaacFSMBackend", IntegrityFailingBackend
    )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(), run_dir=tmp_path, seed=1001, snapshot_bundle=bundle
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 1
    assert report["attempts"][0]["failure_classification"] == (
        "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    )
    assert not (tmp_path / "live_command_result.json").exists()


@pytest.mark.parametrize(
    ("exception_factory", "message"),
    (
        (
            lambda backend: backend.SensorContractFailure(
                "critical live sensing quality failed: exact pair unavailable"
            ),
            "critical live sensing quality failed",
        ),
        (
            lambda backend: backend.IsaacFSMBackendError(
                "frozen controller clock differs from live physics"
            ),
            "frozen controller clock differs",
        ),
    ),
)
def test_post_write_infrastructure_or_frozen_failure_remains_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: object,
    message: str,
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )

    class PostWriteFatalBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            self._book.current.post_snapshot_observations.append(None)
            raise exception_factory(isaac_fsm_backend)

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", PostWriteFatalBackend)
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(), run_dir=tmp_path, seed=1001, snapshot_bundle=bundle
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert message in report["attempts"][0]["exception"]["message"]
    assert not (tmp_path / "live_command_result.json").exists()


def test_final_bundle_identity_assertion_failure_is_fatal_but_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)

    class SuccessfulResetBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            return SimpleNamespace(state_id="P02", physics_tick=0)

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", SuccessfulResetBackend)

    def fail_identity_assertion(bundle: object, *, canonical_root: Path) -> object:
        raise phase_snapshots.PhaseSnapshotError("filesystem identity changed A-B-A")

    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        fail_identity_assertion,
    )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(), run_dir=tmp_path, seed=1001, snapshot_bundle=bundle
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 1
    assert "filesystem identity changed A-B-A" in report["failure_reasons"][0]
    assert not (tmp_path / "live_command_result.json").exists()


def test_successful_reset_with_missing_runtime_evidence_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )

    class IncompletelyObservedSuccessfulBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            # Returning a frame without the authoritative post-write sensor
            # sample must not be downgraded to an ordinary physical mismatch.
            return SimpleNamespace(state_id="P02", physics_tick=0)

    monkeypatch.setattr(
        isaac_fsm_backend,
        "IsaacFSMBackend",
        IncompletelyObservedSuccessfulBackend,
    )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="integrity or infrastructure"):
        probe_subject.run_phase_snapshot_live_probe(
            object(), run_dir=tmp_path, seed=1001, snapshot_bundle=bundle
        )

    report = json.loads((tmp_path / "phase_snapshot_live_probe.json").read_text())
    assert report["failure_classification"] == "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    assert report["completed_attempt_count"] == 1
    assert report["attempts"][0]["failure_classification"] == (
        "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
    )
    assert "violated probe runtime invariants" in report["failure_reasons"][0]
    assert not (tmp_path / "live_command_result.json").exists()


def test_post_write_contact_rejection_remains_returnable_failed_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, phase_snapshots

    _write_managed_prechecks(tmp_path)
    _patch_probe_snapshot_loader(monkeypatch, tmp_path)
    bundle = _FakeBundle(tmp_path)
    monkeypatch.setattr(probe_subject, "_instrumented_dependencies", lambda book: book)
    monkeypatch.setattr(
        phase_snapshots,
        "assert_phase_snapshot_bundle_unchanged",
        lambda bundle, *, canonical_root: bundle,
    )

    class ContactRejectingBackend:
        def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
            self._book = dependencies
            self._scene = None
            self._episode_tick = 0
            self._phase_snapshot_integrity_failed = False

        def reset(self, **kwargs: object) -> object:
            self._scene = object()
            self._book.current.snapshot_write_finished = True
            self._book.current.snapshot_state_write = {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
            }
            # A captured post-write sample is the fail-closed boundary between
            # a useful restoration diagnostic and an infrastructure failure.
            self._book.current.post_snapshot_observations.append(None)
            raise isaac_fsm_backend.SensorContractFailure(
                "phase snapshot live restoration could not be proven: contact"
            )

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", ContactRejectingBackend)
    result = probe_subject.run_phase_snapshot_live_probe(
        object(), run_dir=tmp_path, seed=1001, snapshot_bundle=bundle
    )

    assert result["status"] == "FAILED"
    assert result["passed"] is False
    assert result["complete"] is True
    assert result["failure_classification"] == "ORDINARY_POST_WRITE_RESTORE_MISMATCH"
    assert len(result["attempts"]) == len(PROBE_PHASES) * ATTEMPTS_PER_PHASE
    assert all(
        row["failure_classification"] == "ORDINARY_POST_WRITE_RESTORE_MISMATCH"
        for row in result["attempts"]
    )
