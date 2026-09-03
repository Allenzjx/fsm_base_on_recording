from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wlr50_clean.conformance_policy import (
    POLICY_SCHEMA,
    load_conformance_policy,
)
from wlr50_clean.reference.similarity import allowed_error, within_contract


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "conformance_policy.yaml"


def test_central_policy_declares_active_and_legacy_tolerances() -> None:
    policy = load_conformance_policy(POLICY_PATH)

    assert policy.schema == POLICY_SCHEMA
    assert policy.active_fraction == pytest.approx(0.30)
    assert policy.active_percent == pytest.approx(30.0)
    assert policy.legacy_fraction == pytest.approx(0.15)
    assert policy.legacy_percent == pytest.approx(15.0)
    assert policy.preserve_legacy_result is True
    assert policy.reference_bounded_correction_fraction == pytest.approx(0.15)
    assert policy.minimum_time_scale == pytest.approx(0.70)
    assert policy.maximum_time_scale == pytest.approx(1.30)
    assert policy.hard_safety_unchanged is True
    assert policy.same_source_event_same_tick is True
    assert policy.conformance_can_block_entry is False
    assert policy.conformance_can_block_completion is False
    assert policy.conformance_can_block_task_success is False


def test_active_and_legacy_envelopes_use_the_same_declared_floor() -> None:
    policy = load_conformance_policy(POLICY_PATH)
    floor = policy.floor("joint_endpoint_delta").absolute_allowance

    assert allowed_error(
        10.0, absolute_floor=floor, fraction=policy.active_fraction
    ) == pytest.approx(3.0)
    assert allowed_error(
        10.0, absolute_floor=floor, fraction=policy.legacy_fraction
    ) == pytest.approx(2.0)
    assert within_contract(
        12.5, 10.0, absolute_floor=floor, fraction=policy.active_fraction
    ) is True
    assert within_contract(
        12.5, 10.0, absolute_floor=floor, fraction=policy.legacy_fraction
    ) is False


def test_policy_rejects_recording_conformance_as_a_task_success_gate(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["conformance_can_block_task_success"] = True
    path = tmp_path / "divergent_policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must not block"):
        load_conformance_policy(path)
