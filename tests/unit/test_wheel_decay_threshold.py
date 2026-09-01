from __future__ import annotations

import pytest

from wlr50_clean.evaluation.trial_analyzer import (
    _recovery_evidence,
    _reference_wheel_decay_threshold,
)


def test_p13_decay_threshold_uses_fifteen_percent_reference_tail_envelope() -> None:
    contract = {
        "phases": [
            {
                "state_id": "P13",
                "reference_result_observation": {
                    "wheel_tail_peak_abs_velocity_rad_s": {
                        "front_left_ankle": 0.21535305678844452,
                        "front_right_ankle": 0.000988999498076737,
                        "rear_left_ankle": 0.007579915691167116,
                        "rear_right_ankle": 0.22262312471866608,
                    }
                },
            }
        ]
    }

    threshold = _reference_wheel_decay_threshold(contract)

    assert threshold == pytest.approx(0.25601659342646597)
    assert 0.20 < threshold
    assert 0.27 > threshold


def test_missing_reference_tail_keeps_legacy_fail_closed_floor() -> None:
    assert _reference_wheel_decay_threshold({"phases": []}) == pytest.approx(0.05)


def test_normal_entry_feedback_is_included_in_correction_bound_audit() -> None:
    fractions = [0.0] * 12
    fractions[7] = 0.15
    values, retry_counts = _recovery_evidence(
        [
            {
                "state_id": "P10",
                "from_lifecycle": "WAIT_ENTRY",
                "to_lifecycle": "EXECUTE_MOTION",
                "details": {"correction_fractions": fractions},
            }
        ]
    )

    assert max(values) == pytest.approx(0.15)
    assert sum(retry_counts.values()) == 0
