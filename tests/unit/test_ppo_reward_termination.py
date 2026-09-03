from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wlr50_clean.ppo.reward_terms import (
    REWARD_TERMS,
    RewardCalculator,
    RewardSignals,
    load_reward_config,
)
from wlr50_clean.ppo.termination import (
    TerminationConfigurationError,
    TerminationEvaluator,
    TerminationReason,
    TerminationSignals,
    load_termination_config,
)


def test_reward_is_config_driven_complete_and_finite() -> None:
    config = load_reward_config()
    assert config.training_enabled is False
    assert tuple(config.terms) == REWARD_TERMS
    result = RewardCalculator(config).evaluate(
        RewardSignals(
            forward_progress_delta_m=0.01,
            phase_progress_delta=0.05,
            active_leg_clearance_m=0.025,
            body_collision=True,
            body_angular_speed_rad_s=2.0,
            pitch_rad=0.1,
            roll_rad=-0.2,
            support_margin_m=0.01,
            support_valid=True,
        ),
        residual_fraction_full12=(0.5,) * 12,
        residual_rate_fraction_full12=(0.25,) * 12,
    )
    assert tuple(result.raw_components) == REWARD_TERMS
    assert tuple(result.weighted_components) == REWARD_TERMS
    assert result.weighted_components["body_collision_penalty"] < 0.0
    assert result.weighted_components["forward_progress"] > 0.0
    assert result.raw_components["residual_magnitude_penalty"] == 0.5


def test_zero_residual_baseline_has_no_residual_or_rate_penalty() -> None:
    result = RewardCalculator().evaluate(RewardSignals())
    assert result.raw_components["residual_magnitude_penalty"] == 0.0
    assert result.raw_components["action_rate_penalty"] == 0.0


def test_termination_priority_is_explicit_and_mutually_classified() -> None:
    evaluator = TerminationEvaluator()
    decision = evaluator.evaluate(
        TerminationSignals(success=True, body_collision=True, nan_inf=True)
    )
    assert decision.terminated is True
    assert decision.truncated is False
    assert decision.reason is TerminationReason.NAN_INF
    assert decision.triggered_reasons == (
        TerminationReason.NAN_INF,
        TerminationReason.BODY_COLLISION,
        TerminationReason.SUCCESS,
    )


def test_timeout_is_truncation_and_conformance_is_diagnostic_only() -> None:
    config = load_termination_config()
    assert config.training_enabled is False
    timeout = TerminationEvaluator(config).evaluate(TerminationSignals(timeout=True))
    assert timeout.terminated is False
    assert timeout.truncated is True
    assert timeout.reason is TerminationReason.TIMEOUT

    diagnostic = TerminationEvaluator(config).evaluate(
        TerminationSignals(reference_conformance_outside_30pct=True)
    )
    assert diagnostic.terminated is diagnostic.truncated is False
    assert diagnostic.reason is None
    assert diagnostic.diagnostics == (TerminationReason.REFERENCE_CONFORMANCE,)


def test_loader_rejects_conformance_truncation_mode(tmp_path) -> None:
    source = Path(__file__).resolve().parents[2] / "configs" / "ppo_termination.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["conformance_outside_30pct"] = "truncation"
    invalid = tmp_path / "ppo_termination.yaml"
    invalid.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(TerminationConfigurationError, match="diagnostic-only"):
        load_termination_config(invalid)
