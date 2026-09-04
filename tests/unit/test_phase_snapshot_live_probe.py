from __future__ import annotations

import copy
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
    snapshot = _snapshot()
    diagnostic = observation_diagnostics(_matching_observation(snapshot), snapshot)
    source_target_sha256 = snapshot["source_command"][
        "drive_target_full12_sha256"
    ]
    row = {
        "phase": "P10",
        "source_drive_target_full12_sha256": source_target_sha256,
        "reset_completed": True,
        "physics_steps_during_reset": 181,
        "extra_physics_priming_steps": 1,
        "post_prime_contact_sensor_read_count": 1,
        "snapshot_state_write": {
            "root_pose_writes": 1,
            "root_velocity_writes": 1,
            "joint_state_writes": 1,
            "simulation_forward_syncs": 1,
            "pre_prime_state_verified": True,
            "pre_prime_joint_state_verified": True,
            "pre_prime_root_link_readback": {
                "verified": True,
                "all_values_finite": True,
                "all_fields_within_production_tolerances": True,
                "physics_steps_before_readback": 0,
                "contact_sensor_reads_before_readback": 0,
            },
            "physics_steps": 1,
            "state_write_count": 1,
            "post_prime_state_rewrite_performed": False,
            "contact_and_state_share_solver_tick": True,
            "prime_physics_steps": 1,
            "prime_atomic_full12_writes": 1,
            "logical_target_fallback_used": False,
            "current_contact_force_provenance": "current_final_solver_force_only",
            "sensor_history_samples_after_reset": 1,
            "source_actuation_match": {
                "all_fields_match": True,
                "source_target_hash_matches": True,
                "logical_target_fallback_used": False,
                "source_drive_target_full12_sha256": source_target_sha256,
            },
            "contact_sensor_reads_after_prime": 1,
            "classifier_cold_started_before_only_episode_read": True,
            "classifier_restored_before_only_episode_read": False,
            "classifier_source_history_restored": False,
            "classifier_source_state_restored": False,
            "classifier_history_equivalence_claimed": False,
            "raw_sensor_history_rewarmed_from_prime": True,
            "contact_backend_reset": True,
            "contact_backend_reset_after_prime": False,
            "fsm_clock_steps_during_priming": 0,
            "episode_clock_steps_during_priming": 0,
            "effective_entry_contract": {
                "schema": "wlr50_clean.ppo_phase_effective_entry_live_proof.v1",
                "verified": True,
                "failures": [],
            },
            "entry_safety_contract": {
                "schema": "wlr50_clean.phase_effective_entry_safety.v1",
                "verified": True,
                "all_failure_flags_false": True,
                "flags": {
                    "body_collision": False,
                    "wheel_only_climb": False,
                    "safety_abort": False,
                },
            },
            "entry_guard_contract": {
                "schema": "wlr50_clean.phase_effective_entry_controller.v1",
                "verified": True,
                "phase": "P10",
                "lifecycle": "EXECUTE_MOTION",
                "nonterminal": True,
                "unblocked": True,
                "p10_signed_velocity_alignment": {
                    "signed_positive_rebound_required": True,
                    "actual_deg_s": 1.0,
                },
            },
            "priming_observation": {
                "raw_physx_contact_sources_verified": True,
                "current_raw_force_hysteresis_contract_matches_snapshot": True,
            },
        },
        "observation_diagnostics": diagnostic,
        "clocks": {
            "backend_episode_tick": 0,
            "controller_frame_state_id": "P10",
            "controller_frame_physics_tick": 0,
        },
    }
    assert _attempt_passed(row) is True
    calibration = copy.deepcopy(row)
    comparison = {
        "schema": "wlr50_clean.phase_snapshot_live_comparison.v1",
        "maximum_errors": {"root_position_m": 0.001},
    }
    calibration["snapshot_state_write"]["effective_entry_contract"] = {
        "schema": (
            "wlr50_clean.ppo_phase_effective_entry_calibration_live_proof.v1"
        ),
        "artifact_role": "CALIBRATION_ONLY_NOT_TRAINING_ACCEPTANCE",
        "verified": True,
        "calibration_only": True,
        "phase": "P10",
        "source_snapshot_post_prime_diagnostic": comparison,
        "failures": [],
    }
    assert _attempt_passed(calibration, calibration_mode=True) is True
    assert _attempt_passed(calibration) is False
    assert _attempt_passed(row, calibration_mode=True) is False
    restored_classifier = copy.deepcopy(row)
    restored_classifier["snapshot_state_write"][
        "classifier_restored_before_only_episode_read"
    ] = True
    assert _attempt_passed(restored_classifier) is False

    assert _attempt_passed({**row, "reset_completed": False}) is False
    assert _attempt_passed({**row, "physics_steps_during_reset": 182}) is False
    assert _attempt_passed(
        {
            **row,
            "snapshot_state_write": {
                **row["snapshot_state_write"],
                "pre_prime_state_verified": False,
            },
        }
    ) is False
    assert _attempt_passed(
        {
            **row,
            "observation_diagnostics": {
                **diagnostic,
                "exact_contacts_match": False,
            },
        }
    ) is True
    # Source-t replay equality remains recorded for diagnosis, but the
    # calibrated production entry/safety/guard proofs are authoritative.
    assert _attempt_passed(
        {
            **row,
            "snapshot_state_write": {
                **row["snapshot_state_write"],
                "source_actuation_match": {
                    "all_fields_match": False,
                    "source_target_hash_matches": False,
                    "logical_target_fallback_used": True,
                    "source_drive_target_full12_sha256": "0" * 64,
                },
            },
        }
    ) is True


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
    assert arguments.phase_snapshot_prime_physics_steps == 1
    assert arguments.phase is None
    assert "phase-snapshot-live-probe" in cli.LIVE_COMMANDS

    selected = cli._parser().parse_args(
        [
            "phase-snapshot-live-probe",
            "--run-dir",
            str(PROJECT_ROOT),
            "--seed",
            "1001",
            "--num-envs",
            "1",
            "--phase",
            "P09",
        ]
    )
    assert selected.phase == "P09"
    assert selected.calibrate_effective_entry is False

    calibration_arguments = cli._parser().parse_args(
        [
            "phase-snapshot-live-probe",
            "--run-dir",
            str(PROJECT_ROOT),
            "--seed",
            "1002",
            "--num-envs",
            "1",
            "--phase",
            "P10",
            "--calibrate-effective-entry",
        ]
    )
    assert calibration_arguments.calibrate_effective_entry is True
    cli._validate_common(calibration_arguments)

    wrapper = (
        PROJECT_ROOT / "scripts" / "run_phase_snapshot_live_probe.ps1"
    ).read_text(encoding="utf-8")
    assert '-Subcommand "phase-snapshot-live-probe"' in wrapper
    assert '-RunKind "phase_snapshot_live_probe"' in wrapper
    assert "-EnvironmentCount 1" in wrapper
    assert "-ReturnFinalizedEvidenceFailure" in wrapper
    assert '"--phase-snapshot-prime-physics-steps", $PrimePhysicsSteps' in wrapper
    assert "[ValidateSet(1)]" in wrapper
    assert "[string]$Phase = $null" in wrapper
    assert 'if ($null -ne $Phase)' in wrapper
    assert '$BaseArgs += @("--phase", $Phase)' in wrapper

    calibration_wrapper = (
        PROJECT_ROOT / "scripts" / "run_phase_effective_entry_calibration.ps1"
    ).read_text(encoding="utf-8")
    assert '-RunKind "phase_effective_entry_calibration"' in calibration_wrapper
    assert '-TrainingStage "phase-effective-entry-calibration"' in calibration_wrapper
    assert '-Subcommand "phase-snapshot-live-probe"' in calibration_wrapper
    assert '"--calibrate-effective-entry"' in calibration_wrapper
    assert "[Parameter(Mandatory = $true)]" in calibration_wrapper
    assert "-ReturnFinalizedEvidenceFailure" not in calibration_wrapper

    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "phase-snapshot-live-probe",
                "--run-dir",
                str(PROJECT_ROOT),
                "--seed",
                "1001",
                "--num-envs",
                "1",
                "--phase-snapshot-prime-physics-steps",
                "2",
            ]
        )

    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "phase-snapshot-live-probe",
                "--run-dir",
                str(PROJECT_ROOT),
                "--seed",
                "1001",
                "--num-envs",
                "1",
                "--phase",
                "P01",
            ]
        )

    common = (PROJECT_ROOT / "scripts" / "_invoke_ppo_cli.ps1").read_text(
        encoding="utf-8"
    )
    assert '$RunKindValue -ceq "phase_snapshot_live_probe"' in common
    assert '$SubcommandValue -ceq "phase-snapshot-live-probe"' in common
    assert "$null -ne $AuthoritativeLiveExitCode" in common


def test_live_probe_rejects_non_one_prime_count_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    with pytest.raises(PhaseSnapshotLiveProbeError, match="exactly one"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=object(),
            prime_physics_steps=2,
        )


@pytest.mark.parametrize(
    "phases",
    (("P01",), ("P14",), ("p09",), (), ("P02", "P03"), "P09"),
)
def test_live_probe_rejects_p01_or_invalid_phase_selector_before_runtime_mutation(
    tmp_path: Path,
    phases: object,
) -> None:
    with pytest.raises(PhaseSnapshotLiveProbeError, match="P02 through P13"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=object(),
            phases=phases,
        )
    assert not (tmp_path / "phase_snapshot_live_probe.json").exists()


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
        self.bundle_sha256 = "c" * 64

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "wlr50_clean.phase_snapshot_bundle.v1",
            "snapshot_root": str(self.snapshot_root),
            "bundle_sha256": self.bundle_sha256,
        }


class _FakeEffectiveContract:
    def __init__(self, bundle: _FakeBundle) -> None:
        self.phase_snapshot_bundle_sha256 = bundle.bundle_sha256

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "wlr50_clean.ppo_phase_effective_entry_record.v1",
            "phase_snapshot_bundle_sha256": self.phase_snapshot_bundle_sha256,
        }


def test_calibration_mode_rejects_wrong_seed_contract_or_missing_phase(
    tmp_path: Path,
) -> None:
    bundle = _FakeBundle(tmp_path)
    with pytest.raises(PhaseSnapshotLiveProbeError, match="seed 1002"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            calibration_mode=True,
            phases=("P10",),
        )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="cannot consume"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1002,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
            calibration_mode=True,
            phases=("P10",),
        )
    with pytest.raises(PhaseSnapshotLiveProbeError, match="explicit phase"):
        probe_subject.run_phase_snapshot_live_probe(
            object(),
            run_dir=tmp_path,
            seed=1002,
            snapshot_bundle=bundle,
            calibration_mode=True,
        )


@pytest.fixture(autouse=True)
def _allow_fake_effective_contract_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wlr50_clean.ppo import phase_effective_entry

    monkeypatch.setattr(
        phase_effective_entry,
        "assert_effective_phase_entry_contract_unchanged",
        lambda contract, *, expected_snapshot_bundle: contract,
    )


class _ContactRejectingBackend:
    """Fake one-scene backend that yields a trustworthy ordinary mismatch."""

    def __init__(self, *args: object, dependencies: object, **kwargs: object) -> None:
        self._book = dependencies
        self._scene = None
        self._episode_tick = 0
        self._phase_snapshot_integrity_failed = False

    def reset(self, **kwargs: object) -> object:
        from wlr50_clean.ppo.isaac_fsm_backend import SensorContractFailure

        self._scene = object()
        self._book.current.snapshot_write_finished = True
        self._book.current.snapshot_state_write = {
            "root_pose_writes": 1,
            "root_velocity_writes": 1,
            "joint_state_writes": 1,
            "simulation_forward_syncs": 1,
            "physics_steps": 0,
        }
        self._snapshot_restoration = {
            "physical_state": {
                "schema": "wlr50_clean.phase_snapshot_prime_without_rewind.v1",
                "reset_use": "TRAINING_RESET_STATE_WRITE",
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "simulation_forward_syncs": 1,
                "root_velocity_write_api": "write_root_link_velocity_to_sim",
                "state_write_count": 1,
                "post_prime_state_rewrite_performed": False,
                "contact_and_state_share_solver_tick": True,
                "prime_physics_steps": 1,
                "prime_applied_full12": [0.0] * 12,
                "physics_steps": 1,
                "fsm_clock_steps_during_priming": 0,
                "episode_clock_steps_during_priming": 0,
                "priming_observation": {
                    "maximum_errors": {"root_position_m": 0.0003},
                    "raw_physx_contact_sources_verified": True,
                    "current_raw_force_hysteresis_contract_matches_snapshot": True,
                },
            }
        }
        self._book.current.post_snapshot_observations.append(None)
        raise SensorContractFailure(
            "phase snapshot live restoration could not be proven: contact"
        )


def _patch_probe_snapshot_loader(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    from wlr50_clean.ppo import phase_snapshots

    def load(_bundle: object, phase: str):
        payload = _snapshot(phase)
        entry = SimpleNamespace(
            source_tick=payload["source_tick"],
            snapshot_path=root / phase / "snapshot.json",
            file_sha256="a" * 64,
            state_sha256="b" * 64,
        )
        return payload, entry

    monkeypatch.setattr(
        phase_snapshots,
        "load_validated_phase_snapshot_payload",
        load,
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
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
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
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
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
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
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
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
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
            object(),
            run_dir=tmp_path,
            seed=1001,
            snapshot_bundle=bundle,
            effective_entry_contract=_FakeEffectiveContract(bundle),
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

    monkeypatch.setattr(
        isaac_fsm_backend, "IsaacFSMBackend", _ContactRejectingBackend
    )
    result = probe_subject.run_phase_snapshot_live_probe(
        object(),
        run_dir=tmp_path,
        seed=1001,
        snapshot_bundle=bundle,
        effective_entry_contract=_FakeEffectiveContract(bundle),
    )

    assert result["status"] == "FAILED"
    assert result["passed"] is False
    assert result["complete"] is True
    assert result["failure_classification"] == "EFFECTIVE_ENTRY_ACCEPTANCE_MISMATCH"
    assert result["phases"] == list(PROBE_PHASES)
    assert result["phase_count"] == len(PROBE_PHASES)
    assert result["phase_selector_mode"] == "all_non_p01_phases"
    assert result["expected_attempt_count"] == len(PROBE_PHASES) * 2
    assert result["fresh_scene_attempt_count"] == 1
    assert result["reused_scene_attempt_count"] == len(PROBE_PHASES) * 2 - 1
    assert len(result["attempts"]) == len(PROBE_PHASES) * ATTEMPTS_PER_PHASE
    assert all(
        row["failure_classification"] == "EFFECTIVE_ENTRY_ACCEPTANCE_MISMATCH"
        for row in result["attempts"]
    )
    assert result["production_reset_modified"] is True
    assert result["extra_physics_priming_steps"] == 1
    assert all(
        row["snapshot_state_write"]["post_prime_state_rewrite_performed"]
        is False
        for row in result["attempts"]
    )
    assert all(
        row["snapshot_state_write"]["priming_observation"]["maximum_errors"]
        == {"root_position_m": 0.0003}
        for row in result["attempts"]
    )
    assert all(
        row["snapshot_state_write"]["priming_observation"][
            "raw_physx_contact_sources_verified"
        ]
        is True
        for row in result["attempts"]
    )


def test_single_phase_selector_runs_fresh_then_reused_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        isaac_fsm_backend, "IsaacFSMBackend", _ContactRejectingBackend
    )

    result = probe_subject.run_phase_snapshot_live_probe(
        object(),
        run_dir=tmp_path,
        seed=1001,
        snapshot_bundle=bundle,
        effective_entry_contract=_FakeEffectiveContract(bundle),
        phases=("P09",),
    )

    assert result["status"] == "FAILED"
    assert result["complete"] is True
    assert result["phases"] == ["P09"]
    assert result["phase_count"] == 1
    assert result["phase_selector_mode"] == "single_phase"
    assert result["expected_attempt_count"] == 2
    assert result["expected_fresh_scene_attempt_count"] == 1
    assert result["expected_reused_scene_attempt_count"] == 1
    assert result["fresh_scene_attempt_count"] == 1
    assert result["reused_scene_attempt_count"] == 1
    assert [row["phase"] for row in result["attempts"]] == ["P09", "P09"]
    assert [row["attempt_kind"] for row in result["attempts"]] == [
        "primary",
        "reused_repeat",
    ]
    assert [row["scene_lifecycle"] for row in result["attempts"]] == [
        "fresh_scene",
        "reused_scene",
    ]
