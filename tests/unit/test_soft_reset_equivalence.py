from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo.soft_reset_equivalence import (
    ACCEPTANCE_CHECK_NAMES,
    CompactZeroResidualTickAudit,
    PHASE_IDS,
    RESET_METADATA_FIELDS,
    SOFT_RESET_ACCEPTANCE_FILENAME,
    SOFT_RESET_ACCEPTANCE_SCHEMA,
    SoftResetEquivalenceError,
    actor_observation_v2_fingerprint,
    compact_trace_row,
    compare_compact_traces,
    compare_full_rate_tick_audits,
    compare_initial_actor_observations,
    compare_reset_metadata,
    compare_reward_totals,
    soft_reset_contract_hashes,
    validate_soft_reset_acceptance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZERO12 = (0.0,) * 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_rows() -> list[dict[str, object]]:
    return [
        {
            "decision_index": index,
            "physics_tick": (index + 1) * 8,
            "sim_time_s": (index + 1) / 15.0,
            "state_id": phase,
            "lifecycle": "EXECUTE_MOTION",
            "phase_progress": 1.0,
            "physics_ticks_executed": 8,
            "actor_observation_v2_dimension": 125,
            "actor_observation_v2_sha256": f"{index:064x}",
            "reward_total": 1.0,
            "reward_breakdown_sha256": f"{index + 1:064x}",
            "nominal_full12": [index / 100.0] * 12,
            "residual_full12": [0.0] * 12,
            "applied_full12": [index / 100.0] * 12,
            "controller_task_result": "SUCCESS" if phase == "P13" else "RUNNING",
            "termination_reason": "SUCCESS" if phase == "P13" else None,
        }
        for index, phase in enumerate(PHASE_IDS)
    ]


def _reset_metadata(*, reused: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "environment_hash": "environment",
        "robot_asset_hash": "robot",
        "canonical_reset_state_source": "baseline_order_post_limit_authoring_pre_settle",
        "canonical_reset_state_sha256": "canonical-reset-sha256",
        "canonical_reset_state_instance_count": 1,
        "canonical_reset_restore_applied": False,
        "canonical_reset_applied_sha256": None,
        "pre_limit_native_state_observed_sha256": "pre-limit-native-sha256",
        "pre_limit_native_state_instance_count": 1,
        "pre_limit_native_state_matches_canonical": True,
        "pre_settle_native_state_observed_sha256": "canonical-reset-sha256",
        "pre_settle_native_state_matches_canonical": True,
        "adapter_standing_pose_deg": [0.0] * 8,
        "canonical_settled_state_source": "natural_post_baseline_order_settle",
        "canonical_settled_state_sha256": "canonical-settled-sha256",
        "canonical_settled_restore_applied": False,
        "canonical_settled_applied_sha256": None,
        "observed_settled_state_sha256": "canonical-settled-sha256",
        "physics_lifecycle_reset": (
            "session_limits_removed_then_hard_reset"
            if reused
            else "scene_factory_reset_before_limit_authoring"
        ),
        "reset_contact_sensor_count": 13,
        "reset_initialization_order": (
            "physics_reset_without_session_limits_then_author_limits_then_settle"
        ),
        "pre_physics_session_limit_state_sha256": "empty-session-limits-sha256",
        "pre_physics_composed_limit_state_sha256": "source-limits-sha256",
        "pre_physics_composed_limit_state_matches_canonical": True,
        "session_limit_specs_present_during_physics_reset": 0,
        "session_limit_specs_removed_before_reset": 16 if reused else 0,
        "removed_session_limit_state_sha256": "a" * 64 if reused else None,
        "post_author_session_limit_state_sha256": "a" * 64,
        "post_author_session_limit_state_matches_canonical": True,
        "session_limit_specs_after_authoring": 16,
        "environment_initialization": {
            "all_eight_servo_limits_applied": True,
            "physx_limits_verified": True,
        },
        "controller_hash": "controller",
        "motion_contract_hash": "motion",
        "seed": 2001,
        "reset_count": 2 if reused else 1,
        "reset_options": {},
        "initial_root_state": [0.0] * 13,
        "initial_joint_state": [0.0] * 24,
        "obstacle_pose": [1.5, 0.0, 0.025],
        "level_reference_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "reset_root_pose_writes": 0,
        "reset_root_velocity_writes": 0,
        "reset_joint_state_writes": 0,
        "reset_global_simulation_resets": 1,
        "reset_simulation_forward_syncs": 0,
        "settle_ticks": 180,
        "randomization_enabled": False,
        "raw_recording_access": False,
        "locked_scene_snapshot": {"gravity": [0.0, 0.0, -9.81]},
    }
    assert set(payload) == set(RESET_METADATA_FIELDS)
    return payload


def _write_accepted_gate(root: Path) -> Path:
    root.mkdir(parents=True)
    traces = (_trace_rows(), _trace_rows())
    trace_names = (
        "episode_0_fresh_scene_compact_trace.jsonl",
        "episode_1_soft_reset_reuse_compact_trace.jsonl",
    )
    trace_paths = []
    for name, rows in zip(trace_names, traces, strict=True):
        trace_path = root / name
        trace_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        trace_paths.append(trace_path)
    phase_counts = {phase: 8 for phase in PHASE_IDS}
    summaries = []
    summary_paths = []
    for index, role in enumerate(("fresh_scene", "soft_reset_reuse")):
        summary = {
            "episode_index": index,
            "reset_role": role,
            "seed": 2001,
            "authoritative_success": True,
            "task_success": True,
            "termination_reason": "SUCCESS",
            "duration_s": 13.0 / 15.0,
            "decision_count": 13,
            "physics_tick": 104,
            "phase_ids_observed": list(PHASE_IDS),
            "decision_phase_ids_observed": list(PHASE_IDS),
            "completed_p01_p13": True,
            "body_collision": False,
            "wheel_only_climb": False,
            "safety_abort": False,
            "under_maximum_duration": True,
            "reward_total": 13.0,
            "initial_actor_observation_v2_dimension": 125,
            "initial_actor_observation_v2_sha256": "b" * 64,
            "zero_residual_tick_audit": {
                "status": "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE",
                "tick_count": 104,
                "nominal_sequence_sha256": "a" * 64,
                "applied_sequence_sha256": "a" * 64,
                "bitwise_equal": True,
                "raw_zero_tick_count": 104,
                "projected_zero_tick_count": 104,
                "zero_fast_path_tick_count": 104,
                "phase_ids_observed": list(PHASE_IDS),
                "physics_tick_count_by_phase": phase_counts,
                "passed": True,
            },
            "reset_metadata": _reset_metadata(reused=index == 1),
            "recording_runtime_access_count": 0,
            "in_episode_root_write_count": 0,
            "trace_path": str(trace_paths[index]),
            "trace_sha256": _sha256(trace_paths[index]),
        }
        summary_path = root / f"episode_{index}_{role}_summary.json"
        summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        summaries.append(summary)
        summary_paths.append(summary_path)
    artifacts = (
        trace_paths[0],
        summary_paths[0],
        trace_paths[1],
        summary_paths[1],
    )
    records = [
        {"path": item.name, "bytes": item.stat().st_size, "sha256": _sha256(item)}
        for item in artifacts
    ]
    trace_comparison = compare_compact_traces(*traces)
    reset_comparison = compare_reset_metadata(
        summaries[0]["reset_metadata"], summaries[1]["reset_metadata"]
    )
    full_rate_tick_audit_comparison = compare_full_rate_tick_audits(
        summaries[0]["zero_residual_tick_audit"],
        summaries[1]["zero_residual_tick_audit"],
    )
    initial_actor_observation_comparison = compare_initial_actor_observations(
        summaries[0], summaries[1]
    )
    reward_total_comparison = compare_reward_totals(
        summaries[0], summaries[1], traces[0], traces[1]
    )
    acceptance = root / SOFT_RESET_ACCEPTANCE_FILENAME
    acceptance.write_text(
        json.dumps(
            {
                "schema": SOFT_RESET_ACCEPTANCE_SCHEMA,
                "passed": True,
                "seed": 2001,
                "episode_count": 2,
                "backend_instance_count": 1,
                "full_rate_raw_streams_written": False,
                "compact_trace_fields": list(trace_comparison["fields"]),
                "checks": {name: True for name in ACCEPTANCE_CHECK_NAMES},
                "episodes": summaries,
                "reset_metadata_comparison": reset_comparison,
                "full_rate_tick_audit_comparison": full_rate_tick_audit_comparison,
                "initial_actor_observation_comparison": (
                    initial_actor_observation_comparison
                ),
                "reward_total_comparison": reward_total_comparison,
                "trace_comparison": trace_comparison,
                "contract_file_sha256": soft_reset_contract_hashes(PROJECT_ROOT),
                "contract_file_sha256_at_end": soft_reset_contract_hashes(
                    PROJECT_ROOT
                ),
                "artifacts": records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "wlr50_clean.ppo_run_manifest.v1",
                "lifecycle": "SUCCEEDED",
                "exit_code": 0,
                "immutable_run_directory": True,
                "run_kind": "soft-reset-equivalence",
                "run_dir": str(root.resolve()),
                "project_root": str(PROJECT_ROOT.resolve()),
                "identity": {
                    "seed": 2001,
                    "environment_count": 1,
                    "training_stage": "soft-reset-equivalence-live",
                },
                "entrypoint": "wlr50_clean.ppo.cli",
                "subcommand": "soft-reset-equivalence",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return acceptance


def test_tick_audit_is_streaming_and_bitwise_strict() -> None:
    audit = CompactZeroResidualTickAudit()
    source = SimpleNamespace(state_id="P01", nominal_action_full12=(0.25,) * 12)
    current = SimpleNamespace(state_id="P01")
    projection = SimpleNamespace(
        raw_residual_full12=ZERO12,
        safe_projected_residual_full12=ZERO12,
        applied_action_full12=(0.25,) * 12,
        zero_residual_fast_path=True,
    )
    for _ in range(8):
        audit.append(source, current, projection)
    result = audit.finalize()
    assert result["passed"] is True
    assert result["tick_count"] == 8
    assert result["raw_zero_tick_count"] == 8
    assert "raw" not in audit.__dict__

    mismatch = CompactZeroResidualTickAudit()
    mismatch.append(
        source,
        current,
        SimpleNamespace(
            raw_residual_full12=ZERO12,
            safe_projected_residual_full12=ZERO12,
            applied_action_full12=(0.25,) * 11 + (0.25000000000000006,),
            zero_residual_fast_path=True,
        ),
    )
    assert mismatch.finalize()["passed"] is False


def test_compact_trace_hashes_the_exact_actor_observation() -> None:
    frame = SimpleNamespace(
        physics_tick=8,
        sim_time_s=1.0 / 15.0,
        state_id="P01",
        phase_progress=0.1,
        nominal_action_full12=ZERO12,
    )
    info = {
        "decision_index": 0,
        "controller_lifecycle": "EXECUTE_MOTION",
        "physics_ticks_executed": 8,
        "reward": {"total": 1.0},
        "projected_residual_full12": ZERO12,
        "applied_action_full12": ZERO12,
        "controller_task_result": "RUNNING",
        "termination_reason": None,
    }
    first = compact_trace_row(
        frame,
        info,
        actor_observation_v2=(0.0,) * 125,
    )
    second = compact_trace_row(
        frame,
        info,
        actor_observation_v2=(0.0,) * 124 + (1.0,),
    )
    assert first["actor_observation_v2_dimension"] == 125
    assert first["actor_observation_v2_sha256"] != second[
        "actor_observation_v2_sha256"
    ]
    with pytest.raises(SoftResetEquivalenceError, match="finite v2 vector"):
        compact_trace_row(
            frame,
            info,
            actor_observation_v2=(float("nan"),) * 125,
        )
    with pytest.raises(SoftResetEquivalenceError, match="finite v2 vector"):
        actor_observation_v2_fingerprint((0.0,) * 124)


def test_cross_episode_full_rate_and_initial_actor_comparisons_are_exact() -> None:
    phase_counts = {phase: 8 for phase in PHASE_IDS}
    audit = {
        "nominal_sequence_sha256": "a" * 64,
        "applied_sequence_sha256": "a" * 64,
        "tick_count": 104,
        "phase_ids_observed": list(PHASE_IDS),
        "physics_tick_count_by_phase": phase_counts,
    }
    same = json.loads(json.dumps(audit))
    comparison = compare_full_rate_tick_audits(audit, same)
    assert comparison["exactly_equal"] is True
    assert comparison["mismatched_fields"] == []

    same["physics_tick_count_by_phase"]["P10"] = 9
    mismatch = compare_full_rate_tick_audits(audit, same)
    assert mismatch["exactly_equal"] is False
    assert mismatch["mismatched_fields"] == ["physics_tick_count_by_phase"]

    initial = {
        "initial_actor_observation_v2_dimension": 125,
        "initial_actor_observation_v2_sha256": "b" * 64,
    }
    assert compare_initial_actor_observations(initial, dict(initial))[
        "exactly_equal"
    ] is True
    changed = dict(initial)
    changed["initial_actor_observation_v2_sha256"] = "c" * 64
    assert compare_initial_actor_observations(initial, changed)[
        "exactly_equal"
    ] is False

    traces = (_trace_rows(), _trace_rows())
    summaries = ({"reward_total": 13.0}, {"reward_total": 13.0})
    assert compare_reward_totals(*summaries, *traces)["passed"] is True
    summaries[1]["reward_total"] = 12.0
    assert compare_reward_totals(*summaries, *traces)["passed"] is False


def test_compact_trace_comparison_is_exact_through_p10_and_whole_episode() -> None:
    fresh = _trace_rows()
    reused = json.loads(json.dumps(fresh))
    comparison = compare_compact_traces(fresh, reused)
    assert comparison["through_p10"]["exactly_equal"] is True
    assert comparison["whole_episode"]["exactly_equal"] is True

    reused[9]["lifecycle"] = "WAIT_FOR_SETTLE"
    mismatch = compare_compact_traces(fresh, reused)
    assert mismatch["through_p10"]["exactly_equal"] is False
    assert mismatch["whole_episode"]["exactly_equal"] is False
    assert mismatch["through_p10"]["first_mismatch"]["fields"] == ["lifecycle"]


def test_reset_metadata_proves_baseline_order_lifecycle_equivalence() -> None:
    fresh = _reset_metadata(reused=False)
    reused = _reset_metadata(reused=True)
    comparison = compare_reset_metadata(fresh, reused)
    assert comparison["backend_instance_count"] == 1
    assert comparison["checks"]["fresh_used_one_scene_factory_physics_reset"] is True
    assert comparison["checks"]["reused_used_one_pre_limit_hard_physics_reset"] is True
    assert comparison["checks"]["fresh_pre_settle_state_reached_canonical"] is True
    assert comparison["checks"]["reused_pre_settle_state_reached_canonical"] is True
    assert comparison["passed"] is True

    reused["reset_global_simulation_resets"] = 0
    assert compare_reset_metadata(fresh, reused)["passed"] is False


def test_acceptance_validator_rejects_stale_contract_or_tampered_evidence(
    tmp_path: Path,
) -> None:
    acceptance = _write_accepted_gate(tmp_path / "gate")
    evidence = validate_soft_reset_acceptance(acceptance, project_root=PROJECT_ROOT)
    assert evidence["passed"] is True
    assert evidence["path"] == str(acceptance.resolve())

    original_acceptance = acceptance.read_text(encoding="utf-8")
    stale = json.loads(original_acceptance)
    first_contract_path = next(iter(stale["contract_file_sha256_at_end"]))
    stale["contract_file_sha256_at_end"][first_contract_path] = "0" * 64
    acceptance.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(SoftResetEquivalenceError, match="contract hashes are stale"):
        validate_soft_reset_acceptance(acceptance, project_root=PROJECT_ROOT)
    acceptance.write_text(original_acceptance, encoding="utf-8")

    artifact = acceptance.parent / "episode_1_soft_reset_reuse_summary.json"
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SoftResetEquivalenceError, match="hash mismatch"):
        validate_soft_reset_acceptance(acceptance, project_root=PROJECT_ROOT)


def test_acceptance_validator_recomputes_cross_episode_full_rate_audit(
    tmp_path: Path,
) -> None:
    acceptance_path = _write_accepted_gate(tmp_path / "gate")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    summary_path = acceptance_path.parent / "episode_1_soft_reset_reuse_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["zero_residual_tick_audit"]["nominal_sequence_sha256"] = "c" * 64
    summary["zero_residual_tick_audit"]["applied_sequence_sha256"] = "c" * 64
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    acceptance["episodes"][1] = summary
    for record in acceptance["artifacts"]:
        if record["path"] == summary_path.name:
            record["bytes"] = summary_path.stat().st_size
            record["sha256"] = _sha256(summary_path)
    acceptance_path.write_text(json.dumps(acceptance) + "\n", encoding="utf-8")

    with pytest.raises(
        SoftResetEquivalenceError,
        match="full-rate tick audit comparison is inconsistent",
    ):
        validate_soft_reset_acceptance(acceptance_path, project_root=PROJECT_ROOT)


def test_single_env_training_gate_fails_before_live_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "train"
    run_dir.mkdir()
    monkeypatch.setattr(
        cli,
        "_dispatch_live",
        lambda args: pytest.fail("missing reset proof must fail before AppLauncher"),
    )
    code = cli.main(
        [
            "train",
            "--run-dir",
            str(run_dir),
            "--seed",
            "1001",
            "--num-envs",
            "1",
        ]
    )
    assert code == 2
    assert "--soft-reset-acceptance" in capsys.readouterr().err


def test_single_env_training_gate_accepts_explicit_finalized_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "train"
    run_dir.mkdir()
    acceptance = _write_accepted_gate(tmp_path / "gate")
    captured = {}

    def fake_dispatch(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_dispatch_live", fake_dispatch)
    assert (
        cli.main(
            [
                "train",
                "--run-dir",
                str(run_dir),
                "--seed",
                "1001",
                "--num-envs",
                "1",
                "--soft-reset-acceptance",
                str(acceptance),
            ]
        )
        == 0
    )
    assert captured["args"]._soft_reset_acceptance_evidence["passed"] is True


def test_soft_reset_command_uses_one_backend_and_writes_compact_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wlr50_clean.ppo import isaac_fsm_backend, residual_direct_env

    backend_instances = []

    class FakeBackend:
        def __init__(self, simulation_app):
            self.simulation_app = simulation_app
            backend_instances.append(self)

    class FakeEpisode:
        def __init__(self, backend, *, collect_trace):
            assert backend is backend_instances[0]
            assert collect_trace is False
            self.backend = backend
            self.reset_count = 0
            self.tick_callback = None
            self.decision_count = 0
            self.done = False
            self.frame = None

        def reset(self, *, seed):
            assert seed == 2001
            self.reset_count += 1
            self.decision_count = 0
            self.done = False
            self.frame = SimpleNamespace(
                physics_tick=0,
                sim_time_s=0.0,
                state_id="P01",
                phase_progress=0.0,
                nominal_action_full12=ZERO12,
                termination_signals=_signals(success=False),
                info=_reset_metadata(reused=self.reset_count == 2),
            )
            return (0.0,) * 125, dict(self.frame.info)

        def step(self, action):
            assert tuple(action) == ZERO12
            index = self.decision_count
            phase = PHASE_IDS[index]
            nominal = (index / 100.0,) * 12
            projection = SimpleNamespace(
                raw_residual_full12=ZERO12,
                safe_projected_residual_full12=ZERO12,
                applied_action_full12=nominal,
                zero_residual_fast_path=True,
            )
            source = SimpleNamespace(
                state_id=phase, nominal_action_full12=nominal
            )
            current = SimpleNamespace(state_id=phase)
            for _ in range(8):
                assert self.tick_callback is not None
                self.tick_callback(source, current, projection)
            terminal = phase == "P13"
            self.decision_count += 1
            self.done = terminal
            self.frame = SimpleNamespace(
                physics_tick=self.decision_count * 8,
                sim_time_s=self.decision_count / 15.0,
                state_id=phase,
                phase_progress=1.0,
                nominal_action_full12=nominal,
                termination_signals=_signals(success=terminal),
                info={},
            )
            return SimpleNamespace(
                observation=(float(index),) + (0.0,) * 124,
                reward=1.0,
                info={
                    "decision_index": index,
                    "controller_lifecycle": "EXECUTE_MOTION",
                    "physics_ticks_executed": 8,
                    "projected_residual_full12": ZERO12,
                    "applied_action_full12": nominal,
                    "reward": {"total": 1.0},
                    "controller_task_result": "SUCCESS" if terminal else "RUNNING",
                    "termination_reason": "SUCCESS" if terminal else None,
                    "recording_runtime_access_count": 0,
                    "in_episode_root_write_count": 0,
                },
            )

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", FakeBackend)
    monkeypatch.setattr(residual_direct_env, "ResidualEpisodeEnv", FakeEpisode)
    args = SimpleNamespace(
        num_envs=1,
        episode_count=2,
        residual_mode="zero",
        deterministic=True,
        seed=2001,
        run_dir=tmp_path,
        maximum_duration_s=200.0,
    )
    assert cli._soft_reset_equivalence(args, object()) == 0
    assert len(backend_instances) == 1
    acceptance = json.loads(
        (tmp_path / SOFT_RESET_ACCEPTANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert acceptance["passed"] is True
    assert acceptance["backend_instance_count"] == 1
    assert acceptance["full_rate_raw_streams_written"] is False
    assert acceptance["trace_comparison"]["through_p10"]["exactly_equal"] is True
    assert acceptance["trace_comparison"]["whole_episode"]["exactly_equal"] is True
    assert acceptance["full_rate_tick_audit_comparison"]["exactly_equal"] is True
    assert acceptance["initial_actor_observation_comparison"]["exactly_equal"] is True
    assert acceptance["reward_total_comparison"]["passed"] is True
    assert acceptance["checks"]["contract_files_unchanged_during_run"] is True
    assert (
        acceptance["contract_file_sha256"]
        == acceptance["contract_file_sha256_at_end"]
    )
    assert acceptance["reset_metadata_comparison"]["passed"] is True
    assert len(acceptance["artifacts"]) == 4
    assert not list(tmp_path.glob("*raw*"))


def _signals(*, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        body_collision=False,
        wheel_only_climb=False,
        fall=False,
        nan_inf=False,
        hard_joint_limit=False,
        physics_explosion=False,
    )


def _install_reset_probe_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    short_first_step: bool = False,
    change_reused_nominal: bool = False,
) -> list[object]:
    from wlr50_clean.ppo import isaac_fsm_backend, residual_direct_env

    backend_instances: list[object] = []

    class FakeBackend:
        def __init__(self, simulation_app):
            self.simulation_app = simulation_app
            backend_instances.append(self)

    class FakeEpisode:
        def __init__(self, backend, *, collect_trace):
            assert backend is backend_instances[0]
            assert collect_trace is False
            self.reset_count = 0
            self.tick_callback = None
            self.decision_count = 0
            self.done = False
            self.frame = None

        def reset(self, *, seed):
            assert seed == 2001
            self.reset_count += 1
            self.decision_count = 0
            self.done = False
            self.frame = SimpleNamespace(
                physics_tick=0,
                info=_reset_metadata(reused=self.reset_count == 2),
            )
            return (0.0,) * 125, dict(self.frame.info)

        def step(self, action):
            assert tuple(action) == ZERO12
            assert self.frame is not None
            tick_count = 4 if short_first_step and self.decision_count == 0 else 8
            nominal_value = self.decision_count / 100.0
            if change_reused_nominal and self.reset_count == 2:
                nominal_value += 0.01
            nominal = (nominal_value,) * 12
            projection = SimpleNamespace(
                raw_residual_full12=ZERO12,
                safe_projected_residual_full12=ZERO12,
                applied_action_full12=nominal,
                zero_residual_fast_path=True,
            )
            source = SimpleNamespace(
                state_id="P01", nominal_action_full12=nominal
            )
            current = SimpleNamespace(state_id="P01")
            for _ in range(tick_count):
                assert self.tick_callback is not None
                self.tick_callback(source, current, projection)
            self.decision_count += 1
            self.frame = SimpleNamespace(
                physics_tick=(self.decision_count - 1) * 8 + tick_count,
                info={},
            )
            return SimpleNamespace(
                info={"physics_ticks_executed": tick_count},
            )

    monkeypatch.setattr(isaac_fsm_backend, "IsaacFSMBackend", FakeBackend)
    monkeypatch.setattr(residual_direct_env, "ResidualEpisodeEnv", FakeEpisode)
    return backend_instances


def _reset_probe_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=1,
        episode_count=2,
        policy_decisions=8,
        residual_mode="zero",
        deterministic=True,
        seed=2001,
        run_dir=tmp_path,
    )


def test_reset_throughput_probe_records_exact_short_two_reset_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend_instances = _install_reset_probe_fakes(monkeypatch)
    timer = iter((0.0, 2.0, 3.0, 5.0, 10.0, 13.0, 15.0, 19.0))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(timer))

    assert cli._reset_throughput_probe(_reset_probe_args(tmp_path), object()) == 0
    assert len(backend_instances) == 1
    probe_path = tmp_path / cli.RESET_THROUGHPUT_PROBE_FILENAME
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    assert payload["schema"] == cli.RESET_THROUGHPUT_PROBE_SCHEMA
    assert payload["status"] == "PASSED"
    assert payload["artifact_role"] == "RESET_THROUGHPUT_DIAGNOSTIC_ONLY"
    assert payload["soft_reset_equivalence_gate_eligible"] is False
    assert payload["policy_decisions_per_episode"] == 8
    assert payload["physics_ticks_per_episode"] == 64
    assert [row["reset_role"] for row in payload["episodes"]] == [
        "fresh_scene",
        "soft_reset_reuse",
    ]
    assert payload["episodes"][0]["reset_wall_s"] == 2.0
    assert payload["episodes"][0]["step_wall_s"] == 2.0
    assert payload["episodes"][0]["ticks_per_wall_s"] == 32.0
    assert payload["episodes"][1]["reset_wall_s"] == 3.0
    assert payload["episodes"][1]["step_wall_s"] == 4.0
    assert payload["episodes"][1]["ticks_per_wall_s"] == 16.0
    assert all(
        len(row["tick0_actor_observation_v2_sha256"]) == 64
        and len(row["nominal_action_120hz_sha256"]) == 64
        and row["nominal_action_120hz_sha256"]
        == row["applied_action_120hz_sha256"]
        and row["physics_tick_count"] == 64
        for row in payload["episodes"]
    )
    assert payload["reset_metadata_comparison"]["passed"] is True
    assert not (tmp_path / SOFT_RESET_ACCEPTANCE_FILENAME).exists()


def test_reset_throughput_probe_rejects_short_step_before_publishing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_reset_probe_fakes(monkeypatch, short_first_step=True)
    timer = iter((0.0, 1.0, 2.0))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(timer))

    with pytest.raises(cli.CliError, match="did not advance exactly 8 physics ticks"):
        cli._reset_throughput_probe(_reset_probe_args(tmp_path), object())
    assert not (tmp_path / cli.RESET_THROUGHPUT_PROBE_FILENAME).exists()
    assert not (tmp_path / SOFT_RESET_ACCEPTANCE_FILENAME).exists()


def test_reset_throughput_probe_fails_closed_on_cross_reset_stream_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_reset_probe_fakes(monkeypatch, change_reused_nominal=True)
    timer = iter((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(timer))

    assert cli._reset_throughput_probe(_reset_probe_args(tmp_path), object()) == 2
    payload = json.loads(
        (tmp_path / cli.RESET_THROUGHPUT_PROBE_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["status"] == "FAILED"
    assert payload["checks"]["nominal_120hz_sequences_equal"] is False
    assert payload["checks"]["applied_120hz_sequences_equal"] is False
    assert payload["soft_reset_equivalence_gate_eligible"] is False
    assert not (tmp_path / SOFT_RESET_ACCEPTANCE_FILENAME).exists()


def test_parser_dispatch_and_powershell_entrypoint_are_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parsed = cli._parser().parse_args(
        [
            "soft-reset-equivalence",
            "--run-dir",
            str(tmp_path),
            "--seed",
            "2001",
            "--num-envs",
            "1",
            "--episode-count",
            "2",
            "--deterministic",
        ]
    )
    assert parsed.command == "soft-reset-equivalence"
    assert parsed.episode_count == 2
    assert "soft-reset-equivalence" in cli.LIVE_COMMANDS

    app = SimpleNamespace(update=lambda: None, close=lambda **kwargs: None)
    isaaclab = ModuleType("isaaclab")
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda **kwargs: SimpleNamespace(app=app)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    calls = []
    monkeypatch.setattr(
        cli,
        "_soft_reset_equivalence",
        lambda args, simulation_app: calls.append((args, simulation_app)) or 17,
    )
    assert cli._dispatch_live(parsed) == 17
    assert calls == [(parsed, app)]

    script = (PROJECT_ROOT / "scripts" / "run_soft_reset_equivalence.ps1").read_text(
        encoding="utf-8"
    )
    assert "_invoke_ppo_cli.ps1" in script
    assert '-Subcommand "soft-reset-equivalence"' in script
    assert '"--episode-count", "2"' in script
    assert '"--residual-mode", "zero"' in script
    assert '"--deterministic"' in script
    assert "EnvironmentCount 1" in script

    train_script = (
        PROJECT_ROOT / "scripts" / "train_phase_residual_ppo.ps1"
    ).read_text(encoding="utf-8")
    assert "[string]$SoftResetAcceptance" in train_script
    assert '"--soft-reset-acceptance"' in train_script
    assert "single-env training requires" in train_script


def test_reset_throughput_probe_parser_dispatch_and_wrapper_are_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parsed = cli._parser().parse_args(
        [
            "reset-throughput-probe",
            "--run-dir",
            str(tmp_path),
            "--seed",
            "2001",
            "--num-envs",
            "1",
            "--episode-count",
            "2",
            "--policy-decisions",
            "8",
            "--deterministic",
        ]
    )
    assert parsed.command == "reset-throughput-probe"
    assert parsed.policy_decisions == 8
    assert "reset-throughput-probe" in cli.LIVE_COMMANDS

    app = SimpleNamespace(update=lambda: None, close=lambda **kwargs: None)
    isaaclab = ModuleType("isaaclab")
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda **kwargs: SimpleNamespace(app=app)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    calls = []
    monkeypatch.setattr(
        cli,
        "_reset_throughput_probe",
        lambda args, simulation_app: calls.append((args, simulation_app)) or 0,
    )
    assert cli._dispatch_live(parsed) == 0
    assert calls == [(parsed, app)]

    script = (
        PROJECT_ROOT / "scripts" / "run_reset_throughput_probe.ps1"
    ).read_text(encoding="utf-8")
    assert "_invoke_ppo_cli.ps1" in script
    assert '-RunKind "reset-throughput-probe"' in script
    assert '-TrainingStage "reset-throughput-probe-live"' in script
    assert '-Subcommand "reset-throughput-probe"' in script
    assert '"--episode-count", "2"' in script
    assert '"--policy-decisions", "8"' in script
    assert '"--residual-mode", "zero"' in script
    assert '"--deterministic"' in script
    assert "EnvironmentCount 1" in script
    assert SOFT_RESET_ACCEPTANCE_FILENAME not in script

    common_wrapper = (
        PROJECT_ROOT / "scripts" / "_invoke_ppo_cli.ps1"
    ).read_text(encoding="utf-8")
    assert '"reset-throughput-probe"' in common_wrapper
