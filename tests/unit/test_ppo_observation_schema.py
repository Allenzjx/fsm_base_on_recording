from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.observation_schema import (
    OBSERVATION_DIMENSION,
    NonFiniteObservationError,
    ObservationSchemaError,
    PPOObservationFrame,
    lifecycle_phase_progress,
    load_observation_schema,
)
from wlr50_clean.ppo.residual_interface import PPOObservationParts, ResidualInterface


def _frame(**overrides) -> PPOObservationFrame:
    values = {
        "state_id": "P01",
        "macro_phase": 1,
        "phase_progress": 0.25,
        "joint_position_error8": tuple(float(i + 1) for i in range(8)),
        "joint_velocity8": (0.0,) * 8,
        "wheel_velocity4": (0.0,) * 4,
        "wheel_contact_code4": (0.0, 1.0, 2.0, 3.0),
        "leg_history12": (0.0,) * 12,
        "body_orientation_wxyz4": (1.0, 0.0, 0.0, 0.0),
        "body_angular_velocity3": (0.0,) * 3,
        "obstacle_relative_geometry9": (0.0,) * 9,
        "full_body_com3": (0.0,) * 3,
        "support_diagnostics4": (0.0, 1.0, 4.0, 1.0),
        "previous_action_full12": (0.0,) * 12,
    }
    values.update(overrides)
    return PPOObservationFrame(**values)


def test_observation_schema_freezes_all_85_offsets_and_names(tmp_path) -> None:
    schema = load_observation_schema()
    assert schema.dimension == OBSERVATION_DIMENSION == 85
    rows = schema.feature_rows()
    assert tuple(row["offset"] for row in rows) == tuple(range(85))
    assert len({row["feature_name"] for row in rows}) == 85
    assert rows[0]["feature_name"] == "state_is_P01"
    assert rows[12]["feature_name"] == "state_is_P13"
    assert rows[13]["feature_name"] == "phase_progress"
    assert rows[14]["feature_name"] == "front_left_hip_position_error"
    assert rows[73]["feature_name"] == "previous_front_left_hip"
    assert rows[84]["feature_name"] == "previous_rear_right_ankle"
    assert schema.online_normalization_updates is False
    assert schema.normalization_status == "INITIAL_NORMALIZATION_STATISTICS_NOT_TRAINING_FINAL"

    output = tmp_path / "schema.csv"
    schema.write_csv(output)
    with output.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 85


def test_raw_order_is_unchanged_and_normalized_vector_is_finite() -> None:
    schema = load_observation_schema()
    frame = _frame()
    raw = schema.encode(frame, normalized=False)
    assert len(raw) == 85
    assert raw[:13] == (1.0,) + (0.0,) * 12
    assert raw[13] == 0.25
    assert raw[14:22] == tuple(float(i + 1) for i in range(8))
    assert raw[34:38] == (0.0, 1.0, 2.0, 3.0)
    normalized = schema.encode(frame)
    assert len(normalized) == 85
    assert all(math.isfinite(value) for value in normalized)
    assert normalized[14] == pytest.approx(1.0 / 135.0)
    assert normalized[15] == pytest.approx(2.0 / 210.0)
    assert normalized[37] == 1.0


def test_raw_v1_vector_is_exactly_legacy_85d_abi() -> None:
    frame = _frame()
    legacy = ResidualInterface().frame(
        state_id=frame.state_id,
        macro_phase=frame.macro_phase,
        phase_progress=frame.phase_progress,
        nominal_action_full12=(0.0,) * 12,
        action_mask_full12=(1,) * 12,
        observation=PPOObservationParts(
            joint_position_error8=frame.joint_position_error8,
            joint_velocity8=frame.joint_velocity8,
            wheel_velocity4=frame.wheel_velocity4,
            wheel_contact_code4=frame.wheel_contact_code4,
            leg_history12=frame.leg_history12,
            body_orientation_wxyz4=frame.body_orientation_wxyz4,
            body_angular_velocity3=frame.body_angular_velocity3,
            obstacle_relative_geometry9=frame.obstacle_relative_geometry9,
            full_body_com3=frame.full_body_com3,
            support_diagnostics4=frame.support_diagnostics4,
        ),
        previous_action_full12=frame.previous_action_full12,
    )
    assert frame.raw_vector() == legacy.observation_vector


def test_live_encoder_uses_names_instead_of_mapping_insertion_order() -> None:
    joints = {
        name: SimpleNamespace(error_deg=100.0 + index, velocity_deg_s=200.0 + index)
        for index, name in reversed(tuple(enumerate(SERVO_ORDER)))
    }
    wheels = {
        name: SimpleNamespace(
            body_name=f"body_{index}", velocity_rad_s=300.0 + index
        )
        for index, name in reversed(tuple(enumerate(WHEEL_ORDER)))
    }
    contacts = {
        f"body_{index}": SimpleNamespace(contact_class="GROUND")
        for index in reversed(range(4))
    }
    observation = SimpleNamespace(
        joints=joints,
        wheels=wheels,
        contacts=contacts,
        guards={},
        base=SimpleNamespace(
            position_w_m=(0.1, 0.2, 0.3),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            angular_velocity_w_rad_s=(0.0, 0.0, 0.0),
        ),
        obstacle=SimpleNamespace(
            front_x_m=0.5,
            back_x_m=2.5,
            left_y_m=1.0,
            right_y_m=-1.0,
            bottom_z_m=0.0,
            top_z_m=0.05,
        ),
        center_of_mass=SimpleNamespace(position_w_m=(0.0, 0.0, 0.1)),
        support=SimpleNamespace(
            signed_margin_m=None,
            projection_inside=None,
            support_count=4,
            valid=True,
        ),
    )
    frame = PPOObservationFrame.from_live_observation(
        observation,
        state_id="P03",
        macro_phase=3,
        phase_progress=0.5,
        previous_action_full12=(0.0,) * 12,
    )
    assert frame.joint_position_error8 == tuple(100.0 + i for i in range(8))
    assert frame.joint_velocity8 == tuple(200.0 + i for i in range(8))
    assert frame.wheel_velocity4 == tuple(300.0 + i for i in range(4))
    assert frame.wheel_contact_code4 == (1.0,) * 4
    assert frame.support_diagnostics4 == (0.0, -1.0, 4.0, 1.0)


def test_state_macro_finite_and_lifecycle_policies_are_explicit() -> None:
    with pytest.raises(ObservationSchemaError, match="macro_phase"):
        _frame(state_id="P02", macro_phase=1)
    with pytest.raises(NonFiniteObservationError):
        _frame(joint_velocity8=(float("nan"),) + (0.0,) * 7)
    assert lifecycle_phase_progress("WAIT_ENTRY", elapsed_s=99, active_duration_s=1) == 0.0
    assert lifecycle_phase_progress("EXECUTE_MOTION", elapsed_s=0.5, active_duration_s=2) == 0.25
    assert lifecycle_phase_progress("VERIFY_RESULT") == 1.0
    assert lifecycle_phase_progress("DONE") == 1.0


def test_v1_loader_rejects_any_canonical_feature_reorder(tmp_path) -> None:
    source = Path(__file__).resolve().parents[2] / "configs" / "ppo_observation_schema.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    names = payload["features"][2]["names"]
    names[0], names[1] = names[1], names[0]
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ObservationSchemaError, match="immutable canonical"):
        load_observation_schema(reordered)
