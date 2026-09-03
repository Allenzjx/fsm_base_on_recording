import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from wlr50_clean.evaluation import baseline_publication as publication


FULL12 = [
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    robot = tmp_path / "robot.usd"
    robot.write_bytes(b"immutable robot asset\n")

    # Create every byte that belongs to the fixed freeze allowlist. Specific
    # schemas below replace the placeholders that publication validates.
    for relative in (*publication.FROZEN_MOTION_PATHS, *publication.PPO_INTERFACE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n", encoding="utf-8")
    for relative_root in publication.RUNTIME_ROOTS:
        path = root / relative_root / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# runtime fixture\n", encoding="utf-8")

    environment = {
        "schema": "environment.fixture.v1",
        "robot": {"usd_path": str(robot), "usd_sha256": _sha(robot)},
        "physics": {"physics_hz": 120.0, "render_fps": 15.0},
        "obstacle": {"height_m": 0.05},
    }
    _write_json(root / "configs/environment_lock.json", environment)
    _write_json(
        root / "configs/selected_reference.json",
        {
            "reference_version": "v010",
            "version_id": "v010_fixture",
            "rear_leg_order": "RR_FIRST",
            "runtime_recording_access_authorized": False,
        },
    )
    _write_json(root / "configs/collision_role_map.json", {"base_link": "BODY"})
    _write_yaml(root / "configs/sensors.yaml", {"schema": "sensors.fixture.v1"})
    _write_yaml(root / "configs/fsm_states.yaml", {"states": list(publication.STATE_IDS)})
    _write_json(
        root / "configs/recording_motion_contract.json",
        {
            "schema": "contract.fixture.v1",
            "reference_version": "v010_fixture",
            "full12_order": FULL12,
            "phases": [
                {
                    "state_id": state,
                    "active_channels": [FULL12[(index - 1) % len(FULL12)]],
                    "ppo_action_mask_full12": [
                        int(channel == FULL12[(index - 1) % len(FULL12)])
                        for channel in FULL12
                    ],
                }
                for index, state in enumerate(publication.STATE_IDS, 1)
            ],
        },
    )
    _write_yaml(
        root / "configs/conformance_policy.yaml",
        {
            "schema": "policy.fixture.v1",
            "active_tolerance": {
                "percent": 30.0,
                "role": "advisory_reference_divergence_diagnostic",
                "blocks_task_success": False,
                "blocks_baseline_freeze": False,
                "blocks_video_publication": False,
                "blocks_ppo_readiness": False,
            },
            "runtime": {
                "conformance_can_block_entry": False,
                "conformance_can_block_completion": False,
                "conformance_can_block_task_success": False,
            },
            "ppo": {
                "reference_divergence_is_hard_action_bound": False,
                "hard_bound_sources": [
                    "actuator_hard_limits",
                    "joint_safety_margins",
                    "wheel_velocity_limits",
                    "phase_action_masks",
                    "action_rate_limits",
                    "body_collision_safety_projection",
                    "wheel_only_climb_safety_projection",
                ],
            },
        },
    )
    _write_yaml(
        root / "configs/ppo_interface.yaml",
        {
            "state_ids": list(publication.STATE_IDS),
            "macro_phase_ids": publication.MACRO_PHASE_IDS,
            "observation_dimension": 85,
            "nominal_action_dimension": 12,
            "residual_action_dimension": 12,
            "training_enabled": False,
        },
    )
    _write_json(
        root / "configs/ppo_observation_schema.json",
        {
            "schema": "wlr50_clean.ppo_observation_schema.v1",
            "dimension": 85,
            "state_ids": list(publication.STATE_IDS),
            "macro_phase_ids": publication.MACRO_PHASE_IDS,
            "features": [
                {
                    "group": "fixture",
                    "offset": 0,
                    "size": 85,
                    "names": [f"x{index}" for index in range(85)],
                }
            ],
        },
    )
    _write_yaml(
        root / "configs/ppo_action_projection.yaml",
        {
            "schema": "wlr50_clean.ppo_action_projection.v1",
            "action_schema_name": "wlr50_clean.residual_full12",
            "action_schema_version": 1,
            "physics_hz": 120.0,
            "decision_hz": 15.0,
            "nominal_action_dimension": 12,
            "residual_action_dimension": 12,
            "training_enabled": False,
            "full12_order": FULL12,
            "bounded_transform": "tanh",
            "recording_envelope_diagnostic": {
                "hard_projection_constraint": False,
            },
            "residual_output_scale": {
                "recording_envelope_used_in_projection": False,
            },
            "phase_action_mask": {
                "source": "recording_motion_contract.json",
                "derive_from": "phases.ppo_action_mask_full12",
            },
            "residual_rate_limits": {"servo_deg_s": 150.0, "wheel_rad_s2": 31.4},
            "absolute_action_limits": {
                "hip_deg": [-135.0, 135.0],
                "knee_deg": [-60.0, 210.0],
                "wheel_rad_s": [-2.094, 2.094],
            },
            "joint_safety_margin_deg": {"hip": 2.0, "knee": 2.0},
            "physical_safety_projection": {
                "body_collision_disables_all_residuals": True,
                "body_collision_forces_wheels_zero": True,
                "wheel_only_climb_disables_all_residuals": True,
                "wheel_only_climb_forces_wheels_zero": True,
            },
            "hard_safety": {"applied_after_all_physical_projection_stages": True},
            "zero_residual": {
                "bitwise_nominal_fast_path": True,
                "full_episode_equivalence_required": True,
            },
        },
    )
    _write_yaml(
        root / "configs/ppo_domain_randomization.yaml",
        {
            "schema": "wlr50_clean.ppo_domain_randomization.v1",
            "enabled": False,
            "training_enabled": False,
            "nominal_evaluation_uses_frozen_environment": True,
            "hooks": {
                "fixture": {
                    "enabled": False,
                    "baseline": 0.0,
                    "range": [-1.0, 1.0],
                    "unit": "fixture",
                }
            },
        },
    )
    _write_yaml(
        root / "configs/ppo_reward.yaml",
        {
            "schema": "wlr50_clean.ppo_reward.v1",
            "training_enabled": False,
            "aggregation": "sum_over_120hz_ticks_in_one_15hz_decision",
            "terms": {"task_success": {"weight": 1.0}},
        },
    )
    _write_yaml(
        root / "configs/ppo_termination.yaml",
        {
            "schema": "wlr50_clean.ppo_termination.v1",
            "training_enabled": False,
            "timeout_s": 200.0,
            "conformance_outside_30pct": "diagnostic_only",
            "priority": [
                "NAN_INF",
                "PHYSICS_EXPLOSION",
                "BODY_COLLISION",
                "WHEEL_ONLY_CLIMB",
                "FALL",
                "HARD_JOINT_LIMIT",
                "SUCCESS",
                "TIMEOUT",
            ],
        },
    )

    trial = root / "runs/trial_043_fixture"
    trial.mkdir(parents=True)
    role_names = {
        "observation": "observation_120hz.jsonl",
        "decision": "decision_15hz.jsonl",
        "command": "full12_commands_120hz.jsonl",
        "transition": "state_transitions.jsonl",
        "task_event": "task_events.jsonl",
        "body_contact": "body_contacts.jsonl",
        "leg_crossing": "leg_crossing_events.jsonl",
        "reference_similarity": "reference_similarity.csv",
        "actual_viewport_video": "actual_viewport_video.mp4",
    }
    artifacts = {}
    for role, name in role_names.items():
        path = trial / name
        path.write_bytes(f"fixture {role}\n".encode())
        artifacts[role] = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    _write_json(
        trial / "trial_manifest.json",
        {
            "trial_id": trial.name,
            "result": "INCOMPLETE_CONTROLLER_BLOCKED",
            "reason": "legacy conformance veto",
            "first_blocker": {"name": "measured velocity diagnostic"},
            "reference_version": "v010_fixture",
            "rear_leg_order": "RR_FIRST",
            "success_evidence": {
                "p01_p13_completed": True,
                "completed_macro_phases": list(publication.STATE_IDS),
                "body_collision": False,
                "wheel_only_climb": False,
                "runtime_raw_recording_access": False,
                "root_state_write_count": 0,
                "teleport_count": 0,
                "external_force_count": 0,
                "external_impulse_count": 0,
                "source_robot_usd_unchanged": True,
            },
            "video": {
                "valid": True,
                "stitched": False,
                "speed_modified": False,
                "full_decode": {
                    "full_decode": True,
                    "timestamps_monotonic": True,
                },
            },
            "artifact_files": artifacts,
        },
    )

    readjudication = root / "outputs/analysis/physical_success_readjudication"
    readjudication.mkdir(parents=True)
    selected = {
        "trial_id": trial.name,
        "trial_number": 43,
        "trial_validity": "VALID",
        "task_result": "SUCCESS",
        "classification": "TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING",
        "P01_P13_complete": True,
        "physical_traversal_complete": True,
        "environment_match": True,
        "environment_hash": _sha(root / "configs/environment_lock.json"),
        "robot_asset_hash": _sha(robot),
        "video_continuous": True,
        "recording_runtime_access": False,
        "forbidden_control_count": 0,
        "body_collision": False,
        "wheel_only_climb": False,
        "fall": False,
        "physics_explosion": False,
        "final_pose_stable": True,
        "rear_leg_order": "RR_FIRST",
        "duration_s": 107.933333,
    }
    selected_path = readjudication / "selected_success_trial.json"
    evidence_path = readjudication / "physical_success_evidence.json"
    diagnostics_path = readjudication / "reference_divergence_diagnostics.csv"
    _write_json(selected_path, {"selected_success_trial": selected})
    _write_json(
        evidence_path,
        {
            "trial_evidence": {
                trial.name: {
                    "trial_id": trial.name,
                    "layers": {
                        "TRIAL_VALIDITY": {"status": "VALID"},
                        "TASK_SUCCESS": {"task_result": "SUCCESS"},
                    },
                }
            }
        },
    )
    with diagnostics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "trial_id",
                "phase",
                "channel",
                "metric",
                "error_percent",
                "within_30_percent",
                "warning",
                "blocks_task_success",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": trial.name,
                "phase": "P09",
                "channel": "rear_left_ankle",
                "metric": "measured_peak_velocity_error_percent",
                "error_percent": "36.458636795326846",
                "within_30_percent": "false",
                "warning": "REFERENCE_DIVERGENCE_WARNING",
                "blocks_task_success": "false",
            }
        )
    _write_json(
        readjudication / "readjudication_manifest.json",
        {
            "selected_trial_id": trial.name,
            "output_files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in (selected_path, evidence_path, diagnostics_path)
            },
        },
    )
    return root, readjudication, trial


def _fake_git(root: Path, *arguments: str) -> str:
    del root
    if arguments == ("rev-parse", "HEAD"):
        return "b9d44bacbe61999bb7cb9564e3dde21deaf336ad\n"
    if arguments[:2] == ("status", "--porcelain=v1"):
        return " M fixture\0"
    raise AssertionError(arguments)


def test_publication_freezes_trial_043_without_conformance_veto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, readjudication, trial = _fixture(tmp_path)
    raw_hashes = {path.name: _sha(path) for path in trial.iterdir() if path.is_file()}
    monkeypatch.setattr(publication, "_git_value", _fake_git)

    result = publication.publish_frozen_baseline(
        project_root=root,
        readjudication_dir=readjudication,
        expected_trial="43",
    )

    assert result.selected_trial_id == "trial_043_fixture"
    assert result.frozen_config_path.is_file()
    assert {path.name for path in result.published_files} == {
        "frozen_successful_fsm.yaml",
        *publication.FINAL_FILENAMES,
    }
    selected = json.loads(
        (root / "outputs/final/selected_success_trial.json").read_text(encoding="utf-8")
    )["selected_success_trial"]
    assert selected["task_result"] == "SUCCESS"
    assert selected["classification"] == "TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING"
    assert selected["reference_divergence_blocks_task_success"] is False
    frozen = yaml.safe_load(
        (root / "configs/frozen_successful_fsm.yaml").read_text(encoding="utf-8")
    )
    assert frozen["status"] == publication.FREEZE_STATUS
    assert frozen["physics_hz"] == 120.0
    assert frozen["decision_hz"] == 15.0
    assert frozen["observation_dimension"] == 85
    assert frozen["nominal_action_dimension"] == 12
    assert frozen["residual_action_dimension"] == 12
    assert frozen["training_enabled"] is False
    action = json.loads(
        (root / "outputs/final/ppo_action_schema.json").read_text(encoding="utf-8")
    )
    assert action["recording_reference"]["hard_action_bound"] is False
    assert tuple(action["phase_action_masks"]) == publication.STATE_IDS
    assert action["macro_phase_ids"] == publication.MACRO_PHASE_IDS
    assert action["environment_reset"]["frozen_environment_hash"] == selected["environment_hash"]
    assert action["episode_end_contract"]["truncated_reasons"] == ["TIMEOUT"]
    assert selected["reference_quality"]["blocks_task_success"] is False
    assert raw_hashes == {
        path.name: _sha(path) for path in trial.iterdir() if path.is_file()
    }


def test_corrupt_raw_artifact_fails_before_any_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, readjudication, trial = _fixture(tmp_path)
    (trial / "task_events.jsonl").write_text("mutated\n", encoding="utf-8")
    monkeypatch.setattr(publication, "_git_value", _fake_git)

    with pytest.raises(publication.BaselinePublicationError, match="hash mismatch"):
        publication.publish_frozen_baseline(
            project_root=root,
            readjudication_dir=readjudication,
            expected_trial="43",
        )

    assert not (root / "configs/frozen_successful_fsm.yaml").exists()
    assert not (root / "outputs/final").exists()


def test_runtime_raw_recording_access_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, readjudication, _ = _fixture(tmp_path)
    (root / "src/wlr50_clean/fsm/controller.py").write_text(
        "open('accepted_steps.jsonl')\n", encoding="utf-8"
    )
    monkeypatch.setattr(publication, "_git_value", _fake_git)

    with pytest.raises(publication.BaselinePublicationError, match="raw Recording access"):
        publication.publish_frozen_baseline(
            project_root=root,
            readjudication_dir=readjudication,
            expected_trial="43",
        )

    assert not (root / "configs/frozen_successful_fsm.yaml").exists()
    assert not (root / "outputs/final").exists()


def test_ppo_recording_headroom_cannot_be_reintroduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, readjudication, _ = _fixture(tmp_path)
    action_path = root / "configs/ppo_action_projection.yaml"
    action = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    action["recording_envelope_diagnostic"]["hard_projection_constraint"] = True
    _write_yaml(action_path, action)
    monkeypatch.setattr(publication, "_git_value", _fake_git)

    with pytest.raises(publication.BaselinePublicationError, match="hard_projection_constraint"):
        publication.publish_frozen_baseline(
            project_root=root,
            readjudication_dir=readjudication,
            expected_trial="43",
        )

    assert not (root / "outputs/final").exists()


def test_readjudication_input_helpers_accept_direct_and_list_shapes() -> None:
    direct = {
        "trial_id": "trial_043_fixture",
        "trial_validity": "VALID",
        "task_result": "SUCCESS",
    }
    assert publication._selected_record(direct) == direct
    evidence = publication._evidence_for_trial(
        {"trials": [{"trial_id": "trial_043_fixture", "proof": "raw"}]},
        "43",
    )
    assert evidence["proof"] == "raw"


def test_missing_legacy_phase_ledger_does_not_veto_geometry_proven_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, readjudication, trial = _fixture(tmp_path)
    manifest_path = trial / "trial_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["success_evidence"].pop("p01_p13_completed")
    manifest["success_evidence"].pop("completed_macro_phases")
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(publication, "_git_value", _fake_git)

    result = publication.publish_frozen_baseline(
        project_root=root,
        readjudication_dir=readjudication,
        expected_trial="43",
    )

    assert result.selected_trial_id == "trial_043_fixture"
