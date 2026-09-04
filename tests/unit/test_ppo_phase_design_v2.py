from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from wlr50_clean.ppo.observation_schema import PPOObservationFrame
from wlr50_clean.ppo.observation_schema_v2 import (
    ADDITIONAL_FEATURE_GROUPS,
    LIFECYCLE_IDS,
    OBSERVATION_DIMENSION_V2,
    ObservationSchemaV2Error,
    PPOObservationFrameV2,
    load_observation_schema_v2,
)
from wlr50_clean.ppo.phase_objectives import (
    DENSE_FAMILIES,
    PROMPT_CAPTURE_WEIGHTS,
    PROMPT_PHASE_WEIGHTS,
    STATE_IDS,
    SUCCESSFUL_FSM_ATTITUDE_ENVELOPE_DERIVATION_SHA256,
    TRANSFER_PHASES,
    PhaseObjectiveError,
    PhysicalProgressState,
    global_phase_potential,
    load_phase_objectives,
    phase_local_potential,
    potential_based_progress,
)
from wlr50_clean.ppo.reward_v2 import (
    EVENT_FAMILIES,
    RewardCalculatorV2,
    RewardSignalsV2,
    reward_duplicate_signal_audit,
    reward_single_term_dominance_audit,
    reward_standing_still_exploit_test,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _attitude_envelope_digest(payload: dict[str, object]) -> str:
    envelope = payload["successful_fsm_attitude_envelope"]
    assert isinstance(envelope, dict)
    canonical = {
        name: envelope[name]
        for name in (
            "schema",
            "source",
            "derivation",
            "excess_normalization_rad",
            "phases",
        )
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _v1_frame() -> PPOObservationFrame:
    return PPOObservationFrame(
        state_id="P01",
        macro_phase=1,
        phase_progress=0.25,
        joint_position_error8=tuple(float(index + 1) for index in range(8)),
        joint_velocity8=(0.0,) * 8,
        wheel_velocity4=(0.0,) * 4,
        wheel_contact_code4=(0.0, 1.0, 2.0, 3.0),
        leg_history12=(0.0,) * 12,
        body_orientation_wxyz4=(1.0, 0.0, 0.0, 0.0),
        body_angular_velocity3=(0.0,) * 3,
        obstacle_relative_geometry9=(0.0,) * 9,
        full_body_com3=(0.0,) * 3,
        support_diagnostics4=(0.0, 1.0, 4.0, 1.0),
        previous_action_full12=(0.0,) * 12,
    )


def _v2_frame() -> PPOObservationFrameV2:
    return PPOObservationFrameV2(
        v1=_v1_frame(),
        lifecycle="VERIFY_RESULT",
        body_linear_velocity_body3=(0.1, 0.2, 0.3),
        imu_linear_acceleration_body3=(1.0, 2.0, 3.0),
        full_body_com_velocity3=(0.4, 0.5, 0.6),
        wheel_normal_forces4=(10.0, 20.0, 30.0, 40.0),
        wheel_load_fractions4=(0.1, 0.2, 0.3, 0.4),
        wheel_slip4=(-0.1, 0.0, 0.1, 0.2),
        active_leg_wheel_bottom_clearance_m=0.025,
        active_leg_vertical_velocity_m_s=-0.03,
        previous_projected_residual_full12=tuple(
            0.01 * index for index in range(12)
        ),
    )


def _progress(calculator: RewardCalculatorV2, state_id: str, value: float) -> PhysicalProgressState:
    objective = calculator.objectives.phase(state_id)
    return PhysicalProgressState(
        state_id=state_id,
        normalized_terms={name: value for name in objective.potential_terms},
    )


def _zero_contact(calculator: RewardCalculatorV2, state_id: str) -> dict[str, float]:
    return {
        name: 0.0
        for name in calculator.objectives.phase(state_id).contact_cost_terms
    }


def test_observation_v2_is_auto_derived_125d_with_exact_v1_prefix() -> None:
    schema = load_observation_schema_v2()
    payload = json.loads(
        (REPOSITORY / "configs" / "ppo_observation_schema_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert "dimension" not in payload
    assert schema.dimension == OBSERVATION_DIMENSION_V2 == 125
    assert sum(feature.size for feature in schema.features) == 125

    frame = _v2_frame()
    raw = schema.encode(frame, normalized=False)
    assert raw[:85] == frame.v1.raw_vector()
    assert raw[85:90] == (0.0, 0.0, 1.0, 0.0, 0.0)
    assert raw[-12:] == frame.previous_projected_residual_full12
    assert len(raw) == 85 + sum(len(names) for _, names in ADDITIONAL_FEATURE_GROUPS)


def test_observation_v2_metadata_covers_every_feature_and_deployment_frame() -> None:
    rows = load_observation_schema_v2().feature_rows()
    assert tuple(row["offset"] for row in rows) == tuple(range(125))
    assert len({row["name"] for row in rows}) == 125
    assert all(row["source"] for row in rows)
    assert all(row["frame"] for row in rows)
    assert {row["deployability"] for row in rows} == {"runtime_observable"}
    assert {row["name"] for row in rows[85:90]} == {
        f"lifecycle_is_{name}" for name in LIFECYCLE_IDS
    }


def test_observation_v2_loader_rejects_any_v1_prefix_change(tmp_path: Path) -> None:
    source = REPOSITORY / "configs" / "ppo_observation_schema_v2.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["features"][0]["names"][0] = "state_is_not_P01"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ObservationSchemaV2Error, match="first 85"):
        load_observation_schema_v2(changed)


def test_all_prompt_phase_weights_are_exact_and_sum_to_one() -> None:
    objectives = load_phase_objectives()
    assert tuple(objectives.phases) == STATE_IDS
    for state_id, expected in PROMPT_PHASE_WEIGHTS.items():
        actual = objectives.phase(state_id).prompt_weights.values_tuple
        assert actual == expected
        assert sum(actual) == pytest.approx(1.0, abs=1.0e-12)
    for state_id, expected in PROMPT_CAPTURE_WEIGHTS.items():
        schedule = objectives.phase(state_id).transfer_schedule
        assert schedule is not None
        assert schedule.capture_weights.values_tuple == expected
        assert sum(expected) == pytest.approx(1.0, abs=1.0e-12)


def test_transfer_set_is_exact_and_weight_interpolation_is_continuous() -> None:
    objectives = load_phase_objectives()
    actual_transfer = {
        state_id
        for state_id, objective in objectives.phases.items()
        if objective.transfer_schedule is not None
    }
    assert actual_transfer == TRANSFER_PHASES
    for state_id in sorted(TRANSFER_PHASES):
        objective = objectives.phase(state_id)
        schedule = objective.transfer_schedule
        assert schedule is not None
        for knot in (schedule.active_end, schedule.capture_end):
            before = objective.weights_at(knot - 1.0e-8).values_tuple
            at = objective.weights_at(knot).values_tuple
            after = objective.weights_at(knot + 1.0e-8).values_tuple
            assert max(abs(left - right) for left, right in zip(before, at)) < 1.0e-6
            assert max(abs(left - right) for left, right in zip(at, after)) < 1.0e-6
        assert objective.level_penalty_fraction_at(0.0) == 0.0
        assert objective.level_penalty_fraction_at(1.0) == 1.0
        assert objective.weights_at(1.0).body_stability >= objective.weights_at(0.0).body_stability


def test_transfer_attitude_envelopes_are_bound_to_frozen_trial043_evidence() -> None:
    objectives = load_phase_objectives()
    evidence = objectives.successful_fsm_attitude_envelope

    assert evidence.derivation_sha256 == (
        SUCCESSFUL_FSM_ATTITUDE_ENVELOPE_DERIVATION_SHA256
    )
    assert evidence.selected_trial_id == "trial_043_20260902_clean_v010"
    assert tuple(evidence.phase_max_attitude_error_rad) == (
        "P01",
        "P04",
        "P08",
        "P10",
        "P11",
    )
    assert evidence.phase_sample_counts == {
        "P01": 1600,
        "P04": 584,
        "P08": 56,
        "P10": 26,
        "P11": 62,
    }
    assert evidence.phase_max_attitude_error_rad["P08"] == pytest.approx(
        0.22242370433007352
    )
    assert evidence.phase_max_attitude_error_rad["P11"] == pytest.approx(
        0.17444540287005458
    )
    assert evidence.source["frozen_manifest_sha256"] == (
        "bd6f1c43322fcd428475d4377c7c7737a21b01699bb105b6ec4b38a8bd3f60aa"
    )
    assert evidence.source["level_reference_snapshot_sha256"] == (
        "4e5d0ef67984de1f4ee3cef533fb6a19f3f11723789f582f9e8273c8f53c6395"
    )
    for state_id in STATE_IDS:
        objective = objectives.phase(state_id)
        if state_id in TRANSFER_PHASES:
            assert objective.successful_fsm_attitude_envelope_rad == pytest.approx(
                evidence.phase_max_attitude_error_rad[state_id]
            )
            assert (
                objective.attitude_envelope_excess_normalization_rad
                == evidence.excess_normalization_rad
            )
        else:
            assert objective.successful_fsm_attitude_envelope_rad is None
            assert objective.attitude_envelope_excess_normalization_rad is None


def test_attitude_envelope_value_tamper_fails_closed(tmp_path: Path) -> None:
    source = REPOSITORY / "configs" / "ppo_phase_objectives_v2.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["successful_fsm_attitude_envelope"]["phases"]["P08"][
        "max_attitude_error_rad"
    ] += 0.01
    changed = tmp_path / "tampered_phase_objectives.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PhaseObjectiveError, match="derivation digest mismatch"):
        load_phase_objectives(changed)


def test_attitude_envelope_source_hash_rederivation_still_fails_locked_v1(
    tmp_path: Path,
) -> None:
    source = REPOSITORY / "configs" / "ppo_phase_objectives_v2.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    envelope = payload["successful_fsm_attitude_envelope"]
    envelope["source"]["observation_stream_sha256"] = "0" * 64
    envelope["derivation_sha256"] = _attitude_envelope_digest(payload)
    changed = tmp_path / "rederived_tampered_phase_objectives.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PhaseObjectiveError, match="not the locked v1 evidence"):
        load_phase_objectives(changed)


def test_attitude_envelope_loader_never_opens_raw_recording_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: object, **kwargs: object):
        opened.append(path.resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    load_phase_objectives()

    assert opened
    assert all(path.suffix != ".jsonl" for path in opened)
    assert all(path == REPOSITORY or REPOSITORY in path.parents for path in opened)
    assert {
        "ppo_phase_objectives_v2.yaml",
        "frozen_successful_fsm_manifest.json",
        "manifest.json",
        "snapshot.json",
    }.issubset({path.name for path in opened})


def test_phase_potential_uses_only_physical_terms_and_global_formula() -> None:
    objectives = load_phase_objectives()
    p08 = objectives.phase("P08")
    state = PhysicalProgressState(
        state_id="P08",
        normalized_terms={name: 0.6 for name in p08.potential_terms},
    )
    assert phase_local_potential(state, objectives) == pytest.approx(0.6)
    assert global_phase_potential(state, objectives) == pytest.approx((7.0 + 0.6) / 13.0)
    assert potential_based_progress(state, state, objectives, gamma=0.99) < 0.0

    with pytest.raises(PhaseObjectiveError, match="time/recording"):
        PhysicalProgressState("P08", {"elapsed_time": 0.5})
    p10 = objectives.phase("P10")
    skipped = PhysicalProgressState(
        "P10", {name: 0.0 for name in p10.potential_terms}
    )
    with pytest.raises(PhaseObjectiveError, match="illegal FSM"):
        potential_based_progress(state, skipped, objectives, gamma=0.99)


def test_reward_has_exactly_five_dense_families_plus_four_events() -> None:
    calculator = RewardCalculatorV2.from_files()
    previous = _progress(calculator, "P08", 0.60)
    current = _progress(calculator, "P08", 0.70)
    breakdown = calculator.evaluate(
        RewardSignalsV2(
            previous_progress=previous,
            current_progress=current,
            lifecycle="EXECUTE_MOTION",
            calibrated_attitude_error=0.25,
            successful_fsm_attitude_envelope_excess=0.05,
            body_angular_rate=0.20,
            body_angular_acceleration=0.10,
            contact_motion_costs={
                name: 0.1
                for name in calculator.objectives.phase("P08").contact_cost_terms
            },
            residual_first_difference=0.10,
            residual_second_difference=0.20,
            residual_magnitude=0.10,
            phase_completion=True,
        )
    )
    assert tuple(breakdown.raw_dense) == DENSE_FAMILIES
    assert tuple(breakdown.weighted_dense) == DENSE_FAMILIES
    assert tuple(breakdown.event_components) == EVENT_FAMILIES
    assert breakdown.event_components["phase_completion"] == 1.0
    assert breakdown.total == pytest.approx(breakdown.dense_total + breakdown.event_total)
    assert math.isfinite(breakdown.total)


def test_transfer_attitude_blends_envelope_to_level_without_a_switch() -> None:
    calculator = RewardCalculatorV2.from_files()

    def stability_cost(progress_value: float) -> float:
        progress = _progress(calculator, "P08", progress_value)
        return calculator.evaluate(
            RewardSignalsV2(
                previous_progress=progress,
                current_progress=progress,
                calibrated_attitude_error=1.0,
                successful_fsm_attitude_envelope_excess=0.0,
                contact_motion_costs=_zero_contact(calculator, "P08"),
            )
        ).raw_dense["body_stability"]

    assert stability_cost(0.50) == 0.0
    assert stability_cost(0.70) > 0.0
    assert stability_cost(1.00) == pytest.approx(0.35)
    assert abs(stability_cost(0.85 - 1.0e-8) - stability_cost(0.85 + 1.0e-8)) < 1.0e-6


def test_reward_audits_reject_idle_positive_reward_and_residual_dominance() -> None:
    calculator = RewardCalculatorV2.from_files()
    duplicate = reward_duplicate_signal_audit(calculator.config)
    standing = reward_standing_still_exploit_test(calculator)
    assert duplicate.passed
    assert duplicate.duplicate_owners == {}
    assert standing.passed
    assert standing.maximum_reward <= 0.0

    progress = _progress(calculator, "P01", 0.5)
    base = calculator.evaluate(
        RewardSignalsV2(
            previous_progress=progress,
            current_progress=progress,
            contact_motion_costs=_zero_contact(calculator, "P01"),
        )
    )
    residual_dominated = replace(
        base,
        weighted_dense={
            "phase_task_progress": 0.01,
            "body_stability": 0.01,
            "contact_motion_quality": 0.01,
            "control_smoothness": 0.01,
            "residual_regularization": -1.0,
        },
    )
    audit = reward_single_term_dominance_audit(
        [residual_dominated], calculator.config
    )
    assert not audit.passed
    assert audit.dominant_family == "residual_regularization"
    assert "residual_regularization" in audit.violations
