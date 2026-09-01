import json
from pathlib import Path

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
    }
    assert len(payload["states"]) == 13
    assert all(required <= set(state) for state in payload["states"])


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
