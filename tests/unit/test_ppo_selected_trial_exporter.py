from __future__ import annotations

import json

import pytest
import pyarrow.parquet as pq

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.observation_schema import PPOObservationFrame, load_observation_schema
from wlr50_clean.ppo.selected_trial_exporter import (
    SelectedTrialExportError,
    SelectedTrialMetadata,
    SelectedTrialStreamingExporter,
)


ZERO = (0.0,) * 12
P01_ACTION_MASK = (1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1)


def _raw_observation(tick: int) -> dict:
    return {
        "schema": "wlr50_clean.live_observation.v1",
        "physics_tick": tick,
        "simulation_time_s": tick / 120.0,
        "physics_dt_s": 1.0 / 120.0,
        "joints": {
            name: {"error_deg": 0.0, "velocity_deg_s": 0.0}
            for name in SERVO_ORDER
        },
        "wheels": {
            name: {"body_name": f"body_{index}", "velocity_rad_s": 0.0}
            for index, name in enumerate(WHEEL_ORDER)
        },
        "contacts": {
            f"body_{index}": {"contact_class": "GROUND"}
            for index in range(4)
        },
        "guards": {
            "wheel_only_climb_detected": {"passed": False},
            "physics_explosion_or_fall": {"passed": False},
            "joint_hard_limit_violation": {"passed": False},
        },
        "base": {
            "position_w_m": [tick / 1200.0, 0.0, 0.1],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "angular_velocity_w_rad_s": [0.0, 0.0, 0.0],
        },
        "obstacle": {
            "front_x_m": 0.5,
            "back_x_m": 2.5,
            "left_y_m": 1.0,
            "right_y_m": -1.0,
            "bottom_z_m": 0.0,
            "top_z_m": 0.05,
        },
        "center_of_mass": {"position_w_m": [tick / 1200.0, 0.0, 0.1]},
        "support": {
            "signed_margin_m": 0.02,
            "projection_inside": True,
            "support_count": 4,
            "valid": True,
        },
        "body_collision": {"detected": False},
    }


def _actor_raw(tick: int) -> tuple[float, ...]:
    frame = PPOObservationFrame(
        state_id="P01",
        macro_phase=1,
        phase_progress=tick / 16.0,
        joint_position_error8=ZERO[:8],
        joint_velocity8=ZERO[:8],
        wheel_velocity4=ZERO[:4],
        wheel_contact_code4=(1.0,) * 4,
        leg_history12=ZERO,
        body_orientation_wxyz4=(1.0, 0.0, 0.0, 0.0),
        body_angular_velocity3=(0.0,) * 3,
        obstacle_relative_geometry9=(0.0,) * 9,
        full_body_com3=(tick / 1200.0, 0.0, 0.1),
        support_diagnostics4=(0.02, 1.0, 4.0, 1.0),
        previous_action_full12=ZERO,
    )
    return load_observation_schema().encode(frame, normalized=False)


def _command(tick: int, *, residual=ZERO) -> dict:
    nominal = (float(tick),) + ZERO[1:]
    applied = tuple(a + b for a, b in zip(nominal, residual, strict=True))
    return {
        "control_physics_tick": tick,
        "sim_time_s": tick / 120.0,
        "state_id": "P01",
        "nominal_full12": nominal,
        "residual_full12": residual,
        "full12": applied,
        "commanded_full12": applied,
        "applied_full12": applied,
        "ppo": {
            "state_id": "P01",
            "macro_phase": 1,
            "phase_progress": tick / 16.0,
            "action_mask_full12": P01_ACTION_MASK,
            "observation_vector": _actor_raw(tick),
        },
    }


def _metadata(**overrides) -> SelectedTrialMetadata:
    values = {
        "trial_id": "trial_test",
        "episode_id": "trial-test-baseline",
        "seed": 7,
        "environment_hash": "env-sha256",
        "controller_hash": "controller-sha256",
        "motion_contract_hash": "contract-sha256",
        "selected_trial_confirmed": True,
        "physical_task_success": True,
        "reference_conformance_diagnostic_only": True,
        "terminated": True,
        "truncated": False,
        "termination_reason": "SUCCESS",
    }
    values.update(overrides)
    return SelectedTrialMetadata(**values)


def _write_ledgers(tmp_path, *, bad_time_tick=None, nonzero_tick=None):
    command_path = tmp_path / "full12_commands_120hz.jsonl"
    observation_path = tmp_path / "observation_120hz.jsonl"
    commands = []
    for tick in range(16):
        residual = (0.1,) + ZERO[1:] if tick == nonzero_tick else ZERO
        row = _command(tick, residual=residual)
        if tick == bad_time_tick:
            row["sim_time_s"] += 0.001
        commands.append(row)
    observations = [_raw_observation(tick) for tick in range(17)]
    command_path.write_text(
        "".join(json.dumps(row) + "\n" for row in commands), encoding="utf-8"
    )
    observation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )
    return command_path, observation_path


def test_streams_120hz_ledgers_into_15hz_baseline_transitions(tmp_path) -> None:
    command_path, observation_path = _write_ledgers(tmp_path)
    result = SelectedTrialStreamingExporter().export_to_logger(
        command_path, observation_path, metadata=_metadata()
    )
    assert result.transition_count == 2
    assert result.command_row_count == 16
    assert result.observation_row_count == 17
    assert result.first_physics_tick == 0
    assert result.terminal_physics_tick == 16
    assert len(result.command_ledger_sha256) == 64
    assert len(result.observation_ledger_sha256) == 64
    assert result.zero_residual_equivalence.residual_action_all_zero is True
    assert result.zero_residual_equivalence.applied_action_bitwise_equal_nominal is True
    assert result.full_physics_tick_zero_residual_equivalence.tick_count == 16
    assert result.full_physics_tick_zero_residual_equivalence.bitwise_equal is True
    rows = result.logger.rows
    assert rows[0].control_tick == 0 and rows[1].control_tick == 1
    assert rows[0].sim_time == 0.0 and rows[1].sim_time == pytest.approx(8 / 120)
    assert rows[0].observation_t_plus_1 == rows[1].observation_t
    assert rows[0].done is False
    assert rows[1].terminated is True
    assert rows[1].termination_reason == "SUCCESS"
    assert len(rows[0].observation_t) == 85
    assert len(rows[0].nominal_action_t) == len(rows[0].residual_action_t) == 12


def test_unconfirmed_trial_is_rejected_before_any_export() -> None:
    with pytest.raises(SelectedTrialExportError, match="not confirmed"):
        _metadata(selected_trial_confirmed=False)


def test_source_cadence_and_zero_equivalence_are_derived_not_trusted(tmp_path) -> None:
    command_path, observation_path = _write_ledgers(tmp_path, bad_time_tick=5)
    with pytest.raises(SelectedTrialExportError, match="exactly 1/120"):
        SelectedTrialStreamingExporter().export_to_logger(
            command_path, observation_path, metadata=_metadata()
        )

    command_path, observation_path = _write_ledgers(tmp_path, nonzero_tick=5)
    with pytest.raises(SelectedTrialExportError, match="non-zero residual"):
        SelectedTrialStreamingExporter().export_to_logger(
            command_path, observation_path, metadata=_metadata()
        )


def test_logged_ppo_action_mask_must_match_frozen_phase_mask(tmp_path) -> None:
    command_path, observation_path = _write_ledgers(tmp_path)
    rows = [
        json.loads(line)
        for line in command_path.read_text(encoding="utf-8").splitlines()
    ]
    # Tick 1 is not a 15 Hz transition boundary; every 120 Hz logged mask must
    # nevertheless agree with the frozen contract.
    rows[1]["ppo"]["action_mask_full12"] = [0] * 12
    command_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(SelectedTrialExportError, match="frozen P01 phase mask"):
        SelectedTrialStreamingExporter().export_to_logger(
            command_path, observation_path, metadata=_metadata()
        )


def test_confirmed_export_writes_real_jsonl_parquet_and_full_tick_proof(tmp_path) -> None:
    command_path, observation_path = _write_ledgers(tmp_path)
    destination = tmp_path / "final"
    result = SelectedTrialStreamingExporter().export_artifacts(
        command_path,
        observation_path,
        metadata=_metadata(),
        output_directory=destination,
    )
    jsonl_path = destination / "ppo_baseline_transitions.jsonl"
    parquet_path = destination / "ppo_baseline_transitions.parquet"
    manifest_path = destination / "ppo_baseline_dataset_manifest.json"
    zero_path = destination / "zero_residual_equivalence.json"
    assert all(
        path.is_file()
        for path in (jsonl_path, parquet_path, manifest_path, zero_path)
    )
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2
    table = pq.read_table(parquet_path)
    assert table.num_rows == 2
    assert set(table.column("trial_id").to_pylist()) == {"trial_test"}
    assert set(table.column("task_result").to_pylist()) == {"SUCCESS"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = manifest["source_120hz_zero_residual_equivalence"]
    assert proof["tick_count"] == 16
    assert proof["bitwise_equal"] is True
    assert proof["residual_action_all_zero"] is True
    assert manifest["source_ledgers"]["command_120hz"]["row_count"] == 16
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    assert zero["full_episode_checked"] is True
    assert zero["source_120hz"]["tick_count"] == 16
    assert zero["source_120hz"]["bitwise_equal"] is True
    assert zero["exported_15hz_transitions"]["transition_count"] == 2
    assert zero["ppo_training_started"] is False
    assert manifest["files"]["zero_residual_equivalence"]["path"] == zero_path.name
    assert result.transition_count == 2

    with pytest.raises(SelectedTrialExportError, match="refusing to overwrite"):
        SelectedTrialStreamingExporter().export_artifacts(
            command_path,
            observation_path,
            metadata=_metadata(),
            output_directory=destination,
        )
