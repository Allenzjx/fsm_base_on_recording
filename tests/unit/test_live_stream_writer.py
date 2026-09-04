from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.evaluation import ResidualActivityCalibration, evaluate_live_run
from wlr50_clean.ppo.live_stream_writer import LiveStreamWriter
from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES, STATE_IDS


class _Observation(SimpleNamespace):
    def as_dict(self):
        return {
            "schema": "wlr50_clean.live_observation.v1",
            "physics_tick": self.physics_tick,
            "simulation_time_s": self.physics_tick / 120.0,
        }


def _observation(tick: int) -> _Observation:
    contacts = {}
    wheels = {}
    for index, name in enumerate(WHEEL_ORDER):
        body = f"wheel_body_{index}"
        wheels[name] = SimpleNamespace(body_name=body)
        contacts[body] = SimpleNamespace(
            ground=SimpleNamespace(pair_verified=True, normal_force_n=10.0),
            obstacle=SimpleNamespace(pair_verified=True, normal_force_n=0.0),
        )
    for index in range(9):
        contacts[f"body_{index}"] = SimpleNamespace(
            ground=SimpleNamespace(pair_verified=True, normal_force_n=0.0),
            obstacle=SimpleNamespace(pair_verified=True, normal_force_n=0.0),
        )
    return _Observation(
        physics_tick=tick,
        joints={name: SimpleNamespace(position_deg=0.0) for name in SERVO_ORDER},
        wheels=wheels,
        contacts=contacts,
        base=SimpleNamespace(
            linear_velocity_w_m_s=(0.0, 0.0, 0.0),
            angular_velocity_w_rad_s=(0.0, 0.0, 0.0),
        ),
    )


def _frame(tick: int, *, terminal: bool = False):
    return SimpleNamespace(
        physics_tick=tick,
        sim_time_s=tick / 120.0,
        state_id="P13" if terminal else "P01",
        nominal_action_full12=(0.0,) * 12,
        termination_signals=SimpleNamespace(
            success=terminal,
            body_collision=False,
            wheel_only_climb=False,
        ),
        info={
            "raw_observation": _observation(tick),
            "raw_controller_frame": SimpleNamespace(events=()),
            "controller_lifecycle": "DONE" if terminal else "EXECUTE_MOTION",
            "level_calibration": {
                "valid": True,
                "sample_count": 30,
                "level_reference_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "termination_mapping": {
                "controller_result": "SUCCESS" if terminal else None,
                "controller_reason": "done" if terminal else None,
            },
        },
    )


def _projection(*, nonzero: bool = False):
    raw = ((0.04,) * 8 + (0.04,) * 4) if nonzero else (0.0,) * 12
    mask = (1,) * 8 + (0,) * 4 if nonzero else (1,) * 12
    masked = raw[:8] + (0.0,) * 4 if nonzero else raw
    return SimpleNamespace(
        raw_residual_full12=raw,
        masked_residual_full12=masked,
        safe_projected_residual_full12=masked,
        applied_action_full12=masked,
        effective_action_mask_full12=mask,
        zero_residual_fast_path=not nonzero,
        hard_safety_modified=False,
        clipping_stages=("residual_rate_limit",) if nonzero else (),
    )


def test_live_writer_proves_zero_bitwise_equivalence(tmp_path: Path) -> None:
    writer = LiveStreamWriter(tmp_path / "zero", seed=2001)
    source = _frame(0)
    writer.start(source)
    writer.write_tick(source, _frame(1), _projection())
    writer.write_decision(
        {
            "decision_index": 0,
            "sim_time_s": 1 / 120,
            "raw_policy_action_full12": [0.0] * 12,
            "reward": {"phase_id": "P01", "total": 0.0},
            "phase_transition_action_jump": [],
        }
    )
    terminal = _frame(1, terminal=True)
    writer.finalize(terminal, reward_total=0.0, decision_count=1)
    manifest = json.loads((tmp_path / "zero" / "trial_manifest.json").read_text())
    audit = manifest["action_projection_audit"]
    assert audit["zero_input_all_ticks_bitwise_equivalent"] is True
    assert audit["zero_residual_fast_path_tick_count"] == 1
    assert audit["exact_pair_contact_contract_valid"] is True


def test_live_writer_audits_small_nonzero_mask_and_rate_projection(tmp_path: Path) -> None:
    writer = LiveStreamWriter(tmp_path / "nonzero", seed=1001)
    source = _frame(0)
    writer.start(source)
    writer.write_tick(source, _frame(1), _projection(nonzero=True))
    writer.write_decision(
        {
            "decision_index": 0,
            "sim_time_s": 1 / 120,
            "raw_policy_action_full12": [0.04] * 12,
            "reward": {"phase_id": "P01", "total": 0.0},
            "phase_transition_action_jump": [
                {"from_state_id": "P01", "handoff_hold_used": True}
            ],
        }
    )
    writer.finalize(_frame(1, terminal=True), reward_total=0.0, decision_count=1)
    manifest = json.loads((tmp_path / "nonzero" / "trial_manifest.json").read_text())
    audit = manifest["action_projection_audit"]
    assert audit["within_five_percent_smoke_amplitude"] is True
    assert audit["mask_honored_when_exercised"] is True
    assert audit["rate_limit_tick_count"] == 1
    assert audit["phase_transition_bridge_count"] == 1
    assert audit["phase_transition_handoff_hold_count"] == 1


def test_live_writer_does_not_misclassify_internal_bridge_hold_as_policy_amplitude(
    tmp_path: Path,
) -> None:
    writer = LiveStreamWriter(tmp_path / "bridge", seed=1001)
    source = _frame(0)
    projection = _projection(nonzero=True)
    projection.raw_residual_full12 = (0.13,) * 12
    writer.start(source)
    writer.write_tick(source, _frame(1), projection)
    writer.write_decision(
        {
            "decision_index": 0,
            "sim_time_s": 1 / 120,
            "raw_policy_action_full12": [0.049] * 12,
            "reward": {"phase_id": "P01", "total": 0.0},
            "phase_transition_action_jump": [{"from_state_id": "P01"}],
        }
    )
    writer.finalize(_frame(1, terminal=True), reward_total=0.0, decision_count=1)
    audit = json.loads(
        (tmp_path / "bridge" / "trial_manifest.json").read_text()
    )["action_projection_audit"]
    assert audit["maximum_absolute_normalized_policy_action"] == 0.049
    assert audit["maximum_absolute_internal_bridge_raw_action"] == 0.13
    assert audit["within_five_percent_smoke_amplitude"] is True


def _complete_observation(tick: int, *, wheel_velocity: float = 0.0):
    wheel_bodies = {
        wheel_name: f"{wheel_name}_body" for wheel_name in WHEEL_ORDER
    }
    wheels = {
        wheel_name: SimpleNamespace(
            body_name=body_name,
            velocity_rad_s=wheel_velocity,
            bottom_w_m=(0.5, 0.0, 0.06),
            geometry_verified=True,
        )
        for wheel_name, body_name in wheel_bodies.items()
    }
    bodies = {
        body_name: SimpleNamespace(linear_velocity_w_m_s=(0.0, 0.0, 0.0))
        for body_name in wheel_bodies.values()
    }
    contacts = {
        body_name: SimpleNamespace(
            ground=SimpleNamespace(pair_verified=True, normal_force_n=10.0),
            obstacle=SimpleNamespace(pair_verified=True, normal_force_n=0.0),
        )
        for body_name in wheel_bodies.values()
    }
    for index in range(9):
        contacts[f"other_body_{index}"] = SimpleNamespace(
            ground=SimpleNamespace(pair_verified=True, normal_force_n=0.0),
            obstacle=SimpleNamespace(pair_verified=True, normal_force_n=0.0),
        )
    return SimpleNamespace(
        schema="wlr50_clean.live_observation.v1",
        physics_tick=tick,
        simulation_time_s=tick / 120.0,
        all_finite=True,
        base=SimpleNamespace(
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            linear_velocity_w_m_s=(0.0, 0.0, 0.0),
            angular_velocity_w_rad_s=(0.0, 0.0, 0.0),
        ),
        joints={
            name: SimpleNamespace(position_deg=0.0) for name in SERVO_ORDER
        },
        wheels=wheels,
        bodies=bodies,
        contacts=contacts,
        obstacle=SimpleNamespace(top_z_m=0.05),
        body_collision=SimpleNamespace(detected=False),
        guards={
            "wheel_only_climb_detected": {"passed": False},
            "physics_explosion_or_fall": {"passed": False},
        },
    )


def _complete_frame(
    tick: int,
    phase: str,
    *,
    nominal: tuple[float, ...] = (0.0,) * 12,
    events=(),
    terminal: bool = False,
):
    return SimpleNamespace(
        physics_tick=tick,
        sim_time_s=tick / 120.0,
        state_id=phase,
        nominal_action_full12=nominal,
        termination_signals=SimpleNamespace(
            success=terminal,
            body_collision=False,
            wheel_only_climb=False,
        ),
        info={
            "raw_observation": _complete_observation(
                tick, wheel_velocity=nominal[-1]
            ),
            "raw_controller_frame": SimpleNamespace(events=tuple(events)),
            "controller_lifecycle": "DONE" if terminal else "EXECUTE_MOTION",
            "level_calibration": {
                "valid": True,
                "sample_count": 30,
                "level_reference_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "termination_mapping": {
                "controller_result": "SUCCESS" if terminal else None,
                "controller_reason": "done" if terminal else None,
            },
        },
    )


def _zero_projection(nominal: tuple[float, ...]):
    return SimpleNamespace(
        raw_residual_full12=(0.0,) * 12,
        masked_residual_full12=(0.0,) * 12,
        safe_projected_residual_full12=(0.0,) * 12,
        applied_action_full12=nominal,
        effective_action_mask_full12=(1,) * 12,
        zero_residual_fast_path=True,
        hard_safety_modified=False,
        clipping_stages=(),
    )


def test_compact_live_projection_is_complete_for_offline_evaluation(
    tmp_path: Path,
) -> None:
    run = tmp_path / "compact_evaluation"
    writer = LiveStreamWriter(run, seed=2001)
    writer.start(_complete_frame(0, "P01"))
    tick = 0
    for phase in STATE_IDS:
        for local_tick in range(4):
            tick += 1
            wheel_command = (
                0.1 if phase == "P13" and local_tick < 2 else 0.0
            )
            nominal = (0.0,) * 8 + (wheel_command,) * 4
            events = []
            if local_tick == 0:
                events.extend(
                    (
                        {
                            "state_id": phase,
                            "from_lifecycle": "DONE",
                            "to_lifecycle": "WAIT_ENTRY",
                            "sim_time_s": tick / 120.0,
                        },
                        {
                            "state_id": phase,
                            "from_lifecycle": "WAIT_ENTRY",
                            "to_lifecycle": "EXECUTE_MOTION",
                            "sim_time_s": tick / 120.0,
                        },
                    )
                )
            elif local_tick == 2:
                events.append(
                    {
                        "state_id": phase,
                        "from_lifecycle": "EXECUTE_MOTION",
                        "to_lifecycle": "VERIFY_RESULT",
                        "sim_time_s": tick / 120.0,
                    }
                )
            elif local_tick == 3:
                events.append(
                    {
                        "state_id": phase,
                        "from_lifecycle": "VERIFY_RESULT",
                        "to_lifecycle": "DONE",
                        "sim_time_s": tick / 120.0,
                    }
                )
            source = _complete_frame(tick - 1, phase, nominal=nominal)
            current = _complete_frame(tick, phase, nominal=nominal, events=events)
            writer.write_tick(source, current, _zero_projection(nominal))
            writer.write_decision(
                {
                    "decision_index": tick - 1,
                    "sim_time_s": tick / 120.0,
                    "raw_policy_action_full12": [0.0] * 12,
                    "reward": {
                        "phase_id": phase,
                        "total": 0.0,
                        "weighted_dense": {
                            family: 0.0 for family in DENSE_FAMILIES
                        },
                        "event_components": {
                            "phase_completion": 0.0,
                            "final_success": 0.0,
                            "task_failure": 0.0,
                            "safety_abort": 0.0,
                        },
                    },
                    "phase_transition_action_jump": [],
                }
            )
    terminal = _complete_frame(tick, "P13", terminal=True)
    writer.finalize(terminal, reward_total=0.0, decision_count=tick)

    first_observation = json.loads(
        (run / "observation_120hz.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_observation["stream_projection"] == (
        "stability_and_safety_evaluation_v1"
    )
    assert set(first_observation) >= {
        "base",
        "joints",
        "wheels",
        "bodies",
        "contacts",
        "obstacle",
        "body_collision",
        "guards",
        "measured_wheel_velocity_rad_s",
    }
    assert "center_of_mass" not in first_observation

    evaluated = evaluate_live_run(
        run,
        seed=2001,
        residual_calibration=ResidualActivityCalibration(
            phase_scale_full12={phase: (1.0,) * 12 for phase in STATE_IDS},
            numeric_noise_floor_full12=(1.0e-6,) * 12,
            quantization_floor_full12=(1.0e-6,) * 12,
        ),
        reward_stream_path=run / "reward_15hz.jsonl",
        wheel_stop_hold_s=1.0 / 120.0,
    )
    assert evaluated.termination.task_success is True
    assert evaluated.termination.completed_phases == STATE_IDS
    assert len(evaluated.phase_rows) == 13
    assert evaluated.episode_row["paired_stability_sample_count"] == 52
    assert evaluated.reward_contributions_available is True
