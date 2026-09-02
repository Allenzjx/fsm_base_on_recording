import json
from pathlib import Path

import pytest
import yaml

from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]


def test_compact_contract_covers_p01_p13_and_all_source_events() -> None:
    path = ROOT / "configs" / "recording_motion_contract.json"
    contract = load_motion_contract(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert [phase.state_id for phase in contract.phases] == [
        f"P{index:02d}" for index in range(1, 14)
    ]
    assert sum(len(phase["reference_events"]) for phase in raw["phases"]) == 168
    assert [step for phase in raw["phases"] for step in phase["reference_steps"]] == list(
        range(1, 27)
    )
    assert raw["source"]["telemetry_cadence"]["validated_continuous_120hz"] is True
    assert raw["source"]["telemetry_cadence"]["sample_count"] == 10760
    assert raw["rear_order_evidence"]["validated_order"] is True


def test_four_source_atomic_events_require_all_twelve_channels() -> None:
    contract = load_motion_contract(ROOT / "configs" / "recording_motion_contract.json")
    groups = [
        group
        for phase in contract.phases
        for group in phase.atomic_groups
        if group.source_full12_atomic
    ]
    assert len(groups) == 4
    assert all(group.same_physics_tick for group in groups)
    assert all(group.motion_start_skew_s == 0.0 for group in groups)
    assert all(group.required_runtime_channels == contract.full12_order for group in groups)


def test_state_specs_have_required_contract_fields() -> None:
    payload = yaml.safe_load((ROOT / "configs" / "fsm_states.yaml").read_text(encoding="utf-8"))
    required = {
        "state_id",
        "macro_phase",
        "state_name",
        "physical_purpose",
        "reference_step",
        "reference_events",
        "start_full12",
        "end_full12",
        "active_channels",
        "active_duration",
        "average_velocity",
        "peak_velocity",
        "wheel_integral",
        "atomic_channels",
        "overlap_timing",
        "entry_conditions",
        "completion_conditions",
        "hard_abort_conditions",
        "max_verify_wait",
        "retry_budget",
        "next_state",
        "recovery_state",
        "ppo_action_mask",
        "normal_correction_fractions",
    }
    assert len(payload["states"]) == 13
    assert all(required <= set(state) for state in payload["states"])
    p13 = next(state for state in payload["states"] if state["state_id"] == "P13")
    assert p13["max_verify_wait"] == pytest.approx(1.0)
    assert p13["recovery_max_verify_wait"] == pytest.approx(1.5)
    p03 = next(state for state in payload["states"] if state["state_id"] == "P03")
    assert p03["normal_correction_fractions"] == pytest.approx(
        (0.0,) * 10 + (-0.149, 0.0)
    )
    assert all(
        state["normal_correction_fractions"] == [0.0] * 12
        for state in payload["states"]
        if state["state_id"] != "P03"
    )


def test_execution_projection_hides_absolute_reference_provenance() -> None:
    contract = load_motion_contract(ROOT / "configs" / "recording_motion_contract.json")
    forbidden_attributes = {
        "reference_sim_start_s",
        "reference_sim_end_s",
        "source_steps",
        "source_events",
        "source_batch_id",
        "cursor",
    }
    for phase in contract.phases:
        assert not (forbidden_attributes & set(vars(phase)))
        for group in phase.atomic_groups:
            assert "source_batch_id" not in vars(group)


def test_p04_p05_boundary_and_atomic_ticks_remain_frozen() -> None:
    contract = load_motion_contract(ROOT / "configs" / "recording_motion_contract.json")
    p04 = next(phase for phase in contract.phases if phase.state_id == "P04")
    p05 = next(phase for phase in contract.phases if phase.state_id == "P05")
    assert p04.end_full12 == p05.start_full12
    assert [round(group.time_s * 120) for group in p04.atomic_groups] == [0, 528]
    assert [round(group.time_s * 120) for group in p05.atomic_groups] == [104, 1088]


def test_p09_wheel_rebound_is_compact_and_strictly_reference_bounded() -> None:
    path = ROOT / "configs" / "recording_motion_contract.json"
    contract = load_motion_contract(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    feedback = contract.phase("P09").drive_feedback
    assert feedback is not None
    assert feedback.kind == "pre_endpoint_wheel_rebound_alignment"
    assert [
        (item.motion_tick, item.reference_actual_deg)
        for item in feedback.probe_samples
    ] == [
        (858, -51.055799822535),
        (859, -51.191638624749),
    ]
    assert feedback.lag_threshold_deg == 0.35
    assert feedback.required_consecutive_samples == 2
    assert feedback.probe_channel == "rear_right_knee"
    assert feedback.probe_channel_index == 7
    assert feedback.correction_channel == "front_left_ankle"
    assert feedback.correction_channel_index == 8
    assert [
        (
            segment.first_bias_tick,
            segment.last_bias_tick,
            segment.logical_bias_rad_s,
        )
        for segment in feedback.bias_segments
    ] == [
        (860, 860, 0.68),
        (861, 861, 0.33),
        (862, 864, 0.68),
        (865, 865, 0.15),
        (866, 867, 1.03),
        (868, 868, 0.73),
        (870, 870, 0.01),
        (871, 871, 0.40),
    ]
    assert feedback.first_bias_tick == 860
    assert feedback.last_bias_tick == 871
    assert feedback.teardown_tick == 872
    assert feedback.reference_wheel_integral_rad == pytest.approx(
        -0.9060000000012605
    )
    assert feedback.additional_wheel_integral_rad == pytest.approx(
        (
            0.68
            + 0.15
            + 0.68 * 3.0
            + 0.33
            + 1.03 * 2.0
            + 0.73
            + 0.01
            + 0.40
        )
        / 120.0
    )
    assert feedback.resulting_wheel_integral_rad == pytest.approx(
        -0.8526666666679271
    )
    assert feedback.resulting_wheel_integral_rad == pytest.approx(
        feedback.reference_wheel_integral_rad
        + feedback.additional_wheel_integral_rad
    )
    assert feedback.cumulative_fraction_of_reference == pytest.approx(
        0.05886681383361936
    )
    assert feedback.cumulative_fraction_of_reference == pytest.approx(
        abs(feedback.additional_wheel_integral_rad)
        / abs(feedback.reference_wheel_integral_rad)
    )
    assert feedback.cumulative_fraction_of_reference < 0.15
    assert feedback.reference_wheel_peak_abs_rad_s == pytest.approx(1.07)
    assert feedback.resulting_wheel_peak_abs_rad_s == pytest.approx(1.07)
    assert feedback.instantaneous_direction_reversal is True
    contact_alignment = feedback.contact_alignment
    assert contact_alignment is not None
    assert contact_alignment.kind == "post_probe_rear_left_air_alignment"
    assert contact_alignment.trigger_tick == 859
    assert contact_alignment.wheel_body == "rear_left_wheel"
    assert contact_alignment.required_contact_class == "AIR"
    assert contact_alignment.require_ground_pair_verified is True
    assert contact_alignment.require_obstacle_pair_verified is True
    assert (
        contact_alignment.first_bias_tick,
        contact_alignment.last_full_bias_tick,
        contact_alignment.release_tick,
        contact_alignment.teardown_tick,
    ) == (860, 870, 871, 872)
    assert contact_alignment.final_slew_limit_deg_per_tick == pytest.approx(1.25)
    alignment_rows = [
        (
            channel.channel,
            channel.channel_index,
            channel.reference_motion_magnitude_deg,
            channel.logical_full_bias_deg,
            channel.logical_release_bias_deg,
            channel.outbound_plus_teardown_deg,
            channel.cumulative_fraction_of_reference,
        )
        for channel in contact_alignment.channels
    ]
    assert [(row[0], row[1]) for row in alignment_rows] == [
        ("rear_left_hip", 4),
        ("rear_left_knee", 5),
    ]
    assert [row[2:] for row in alignment_rows] == pytest.approx(
        [
            (15.8, -1.185, 0.0, 2.37, 0.15),
            (19.4, -1.455, -0.205, 2.91, 0.15),
        ]
    )
    raw_p09 = next(phase for phase in raw["phases"] if phase["state_id"] == "P09")
    assert raw_p09["active_duration_s"] == pytest.approx(7.2)
    assert raw_p09["command_metrics"]["wheel_integral_rad"][
        "front_left_ankle"
    ] == pytest.approx(-0.9060000000012605)
    endpoint_tick = round(contract.phase("P09").active_duration_s * 120.0)
    expected_native_and_final = (
        (860, -1.07, -0.39),
        (861, -1.07, -0.74),
        (862, -1.07, -0.39),
        (863, -1.07, -0.39),
        (864, 0.0, 0.68),
        (865, 0.0, 0.15),
        (866, 0.0, 1.03),
        (867, 0.0, 1.03),
        (868, 0.0, 0.73),
        (869, 0.0, 0.0),
        (870, 0.0, 0.01),
        (871, 0.0, 0.40),
    )
    segments_by_tick = {
        tick: segment
        for segment in feedback.bias_segments
        for tick in range(segment.first_bias_tick, segment.last_bias_tick + 1)
    }
    for tick, expected_native, expected_final in expected_native_and_final:
        native = (
            contract.phase("P09").nominal_at(tick / 120.0)[8]
            if tick < endpoint_tick
            else contract.phase("P09").end_full12[8]
        )
        assert native == pytest.approx(expected_native)
        segment = segments_by_tick.get(tick)
        bias = 0.0 if segment is None else segment.logical_bias_rad_s
        assert native + bias == pytest.approx(expected_final)
    assert all(
        phase.drive_feedback is None
        for phase in contract.phases
        if phase.state_id != "P09"
    )

    alignment = contract.phase("P10").entry_velocity_alignment
    assert alignment is not None
    assert alignment.channel == "rear_right_knee"
    assert alignment.channel_index == 7
    assert alignment.reference_velocity_deg_s == pytest.approx(
        23.585333053160202
    )
    assert alignment.relative_limit == 0.15
    assert all(
        phase.entry_velocity_alignment is None
        for phase in contract.phases
        if phase.state_id != "P10"
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("kind", "verify_tail_carry_alignment"),
        ("probe_channel", "rear_left_knee"),
        ("lag_threshold_deg", 0.34),
        ("correction_channel_index", 5),
        ("teardown_tick", 873),
        ("additional_wheel_integral_rad", 0.107),
        ("resulting_wheel_integral_rad", -0.7990000000012605),
        ("cumulative_fraction_of_reference", 0.118101545253699),
        ("reference_wheel_peak_abs_rad_s", 1.06),
        ("resulting_wheel_peak_abs_rad_s", 1.08),
        ("instantaneous_direction_reversal", False),
    ),
)
def test_p09_wheel_rebound_contract_fails_closed(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    source = ROOT / "configs" / "recording_motion_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    p09 = next(phase for phase in payload["phases"] if phase["state_id"] == "P09")
    p09["drive_feedback"][field] = invalid_value
    candidate = tmp_path / f"invalid_{field}.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="P09:.*drive-feedback"):
        load_motion_contract(candidate)


@pytest.mark.parametrize(
    ("segment_index", "field", "invalid_value"),
    (
        (0, "first_bias_tick", 861),
        (0, "last_bias_tick", 861),
        (0, "logical_bias_rad_s", 0.67),
        (1, "first_bias_tick", 860),
        (1, "logical_bias_rad_s", 0.34),
        (2, "last_bias_tick", 863),
        (2, "logical_bias_rad_s", 0.67),
        (3, "first_bias_tick", 864),
        (3, "logical_bias_rad_s", 0.34),
        (4, "last_bias_tick", 866),
        (4, "logical_bias_rad_s", 1.02),
        (5, "last_bias_tick", 869),
        (5, "logical_bias_rad_s", 0.72),
        (6, "first_bias_tick", 869),
        (6, "logical_bias_rad_s", 0.27),
        (7, "last_bias_tick", 872),
        (7, "logical_bias_rad_s", 0.37),
    ),
)
def test_p09_wheel_rebound_segments_fail_closed(
    tmp_path: Path,
    segment_index: int,
    field: str,
    invalid_value: object,
) -> None:
    source = ROOT / "configs" / "recording_motion_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    p09 = next(phase for phase in payload["phases"] if phase["state_id"] == "P09")
    p09["drive_feedback"]["bias_segments"][segment_index][field] = invalid_value
    candidate = tmp_path / f"invalid_segment_{segment_index}_{field}.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="P09:.*drive-feedback"):
        load_motion_contract(candidate)


@pytest.mark.parametrize("invalid_value", ("true", "false", 1, 0))
def test_p09_wheel_rebound_requires_a_json_boolean(
    tmp_path: Path, invalid_value: object
) -> None:
    source = ROOT / "configs" / "recording_motion_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    p09 = next(phase for phase in payload["phases"] if phase["state_id"] == "P09")
    p09["drive_feedback"]["instantaneous_direction_reversal"] = invalid_value
    candidate = tmp_path / f"invalid_direction_type_{invalid_value}.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a boolean"):
        load_motion_contract(candidate)


def test_p09_wheel_rebound_pins_locked_probe_references(tmp_path: Path) -> None:
    source = ROOT / "configs" / "recording_motion_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    p09 = next(phase for phase in payload["phases"] if phase["state_id"] == "P09")
    p09["drive_feedback"]["probe_samples"][0]["reference_actual_deg"] += 0.01
    candidate = tmp_path / "invalid_probe_reference.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="P09: invalid drive-feedback timing"):
        load_motion_contract(candidate)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    (
        (("required_contact_class",), "GROUND"),
        (("require_ground_pair_verified",), False),
        (("release_tick",), 872),
        (("channels", 0, "logical_full_bias_deg"), -1.186),
        (("channels", 1, "logical_release_bias_deg"), 0.0),
        (("channels", 1, "cumulative_fraction_of_reference"), 0.149),
    ),
)
def test_p09_contact_alignment_contract_fails_closed(
    tmp_path: Path, path: tuple[object, ...], invalid_value: object
) -> None:
    source = ROOT / "configs" / "recording_motion_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    p09 = next(phase for phase in payload["phases"] if phase["state_id"] == "P09")
    target = p09["drive_feedback"]["contact_alignment"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    candidate = tmp_path / "invalid_contact_alignment.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="P09:.*contact-alignment"):
        load_motion_contract(candidate)


def test_p09_wheel_rebound_rederives_the_reference_integral(
    tmp_path: Path,
) -> None:
    source = ROOT / "configs" / "recording_motion_contract.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    p09 = next(phase for phase in payload["phases"] if phase["state_id"] == "P09")
    feedback = p09["drive_feedback"]
    feedback["reference_wheel_integral_rad"] *= 10.0
    feedback["resulting_wheel_integral_rad"] = (
        feedback["reference_wheel_integral_rad"]
        + feedback["additional_wheel_integral_rad"]
    )
    feedback["cumulative_fraction_of_reference"] = abs(
        feedback["additional_wheel_integral_rad"]
    ) / abs(feedback["reference_wheel_integral_rad"])
    candidate = tmp_path / "inflated_reference_integral.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="P09: drive-feedback budget"):
        load_motion_contract(candidate)
