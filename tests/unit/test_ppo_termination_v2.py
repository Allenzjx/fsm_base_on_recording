from __future__ import annotations

import pytest

from wlr50_clean.ppo.termination import TerminationReason
from wlr50_clean.ppo.termination_v2 import (
    TerminationEvaluatorV2,
    TerminationSignalsV2,
    load_termination_config_v2,
)


def test_v2_is_trainable_fsm_owned_and_bounded_to_200_seconds() -> None:
    config = load_termination_config_v2()
    assert config.training_enabled is True
    assert config.success_source == "frozen_fsm_authoritative_task_success"
    assert config.timeout_s == 200.0


def test_recording_divergence_is_diagnostic_only() -> None:
    decision = TerminationEvaluatorV2().evaluate(
        TerminationSignalsV2(reference_conformance_outside_30pct=True),
        episode_time_s=1.0,
    )
    assert not decision.terminated and not decision.truncated
    assert decision.diagnostics == (TerminationReason.REFERENCE_CONFORMANCE,)


def test_hard_safety_beats_success_and_timeout_is_truncation() -> None:
    evaluator = TerminationEvaluatorV2()
    collision = evaluator.evaluate(
        TerminationSignalsV2(authoritative_success=True, body_collision=True),
        episode_time_s=12.0,
    )
    assert collision.terminated and collision.reason is TerminationReason.BODY_COLLISION
    timeout = evaluator.evaluate(TerminationSignalsV2(), episode_time_s=200.0)
    assert timeout.truncated and timeout.reason is TerminationReason.TIMEOUT


def test_sensor_contract_fails_closed_as_infrastructure_error() -> None:
    with pytest.raises(RuntimeError, match="infrastructure_abort"):
        TerminationEvaluatorV2().evaluate(
            TerminationSignalsV2(sensor_contract_valid=False), episode_time_s=0.0
        )
