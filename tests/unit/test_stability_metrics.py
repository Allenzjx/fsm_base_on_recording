from __future__ import annotations

from wlr50_clean.ppo.stability_metrics import (
    EpisodeOutcome,
    LOWER_IS_BETTER,
    PHASE_IDS,
    PHASE_SCORE_METRICS,
    PRIORITY_PHASES,
    StabilitySample,
    compare_phase_metrics,
    evaluate_promotion,
    residual_activity_by_phase,
    residual_spectrum_band_fractions,
    summarize_phase_samples,
)

import numpy as np
import pytest


def _sample(phase: str, tick: int, residual: float = 0.0) -> StabilitySample:
    return StabilitySample(
        time_s=tick / 120.0,
        phase=phase,
        lifecycle="EXECUTE_MOTION",
        roll_error_rad=0.01,
        pitch_error_rad=0.02,
        roll_rate_rad_s=0.03,
        pitch_rate_rad_s=0.04,
        residual_full12=(residual,) * 12,
    )


def test_phase_metrics_include_required_rate_and_smoothness_fields():
    rows = summarize_phase_samples(
        [_sample("P02", i, 0.01 * i) for i in range(8)],
        phase_scale_full12={phase: (1.0,) * 12 for phase in PHASE_IDS},
    )
    assert rows[0]["phase"] == "P02"
    assert rows[0]["pitch_rate_rms_rad_s"] > 0.0
    assert rows[0]["pitch_rate_p95_abs_rad_s"] > 0.0
    assert rows[0]["angular_acceleration_rms_rad_s2"] == 0.0
    assert rows[0]["residual_action_jerk_rms"] < 1e-12
    spectrum = {
        name: value
        for name, value in rows[0].items()
        if name.startswith("residual_spectral_energy_fraction_")
    }
    assert len(spectrum) == 5
    assert sum(spectrum.values()) == pytest.approx(1.0)
    assert rows[0]["residual_spectrum_normalization"] == "phase_scale_full12"


def test_phase_residual_spectrum_is_dimensionless_across_servo_and_wheel_channels():
    sample_hz = 120.0
    time = np.arange(0.0, 4.0, 1.0 / sample_hz)
    rows = []
    for index, time_s in enumerate(time):
        residual = [0.0] * 12
        residual[0] = 2.0 * np.sin(2.0 * np.pi * 1.5 * time_s)  # degrees
        residual[8] = 0.1 * np.sin(2.0 * np.pi * 4.0 * time_s)  # rad/s
        rows.append(
            StabilitySample(
                time_s=float(time_s),
                phase="P02",
                lifecycle="EXECUTE_MOTION",
                roll_error_rad=0.0,
                pitch_error_rad=0.0,
                roll_rate_rad_s=0.0,
                pitch_rate_rad_s=0.0,
                residual_full12=tuple(residual),
            )
        )
    scales = {phase: (1.0,) * 12 for phase in PHASE_IDS}
    scales["P02"] = (2.0,) + (1.0,) * 7 + (0.1,) + (1.0,) * 3
    summary = summarize_phase_samples(
        rows,
        phase_scale_full12=scales,
        physics_hz=sample_hz,
    )[0]

    assert summary["residual_spectral_energy_fraction_1p0_2p0_hz"] == pytest.approx(
        0.5, abs=1.0e-12
    )
    assert summary[
        "residual_spectral_energy_fraction_3p0_nyquist_hz"
    ] == pytest.approx(0.5, abs=1.0e-12)


def test_residual_spectrum_uses_real_fft_bins_and_excludes_dc():
    sample_hz = 120.0
    time = np.arange(0.0, 4.0, 1.0 / sample_hz)
    signal = np.zeros((time.size, 12), dtype=float)
    signal[:, 0] = 4.0 + np.sin(2.0 * np.pi * 1.5 * time)
    signal[:, 1] = 0.5 * np.sin(2.0 * np.pi * 4.0 * time)

    bands = residual_spectrum_band_fractions(signal, 1.0 / sample_hz)

    assert sum(bands.values()) == pytest.approx(1.0, abs=1.0e-12)
    assert bands["residual_spectral_energy_fraction_1p0_2p0_hz"] == pytest.approx(
        0.8, abs=1.0e-12
    )
    assert bands[
        "residual_spectral_energy_fraction_3p0_nyquist_hz"
    ] == pytest.approx(0.2, abs=1.0e-12)
    assert bands["residual_spectral_energy_fraction_0p0_0p5_hz"] < 1.0e-20


def test_residual_spectrum_zero_and_short_signals_are_honestly_zero():
    zero = residual_spectrum_band_fractions(np.zeros((120, 12)), 1.0 / 120.0)
    short = residual_spectrum_band_fractions(np.ones((3, 12)), 1.0 / 120.0)
    assert set(zero) == set(short)
    assert all(value == 0.0 for value in zero.values())
    assert all(value == 0.0 for value in short.values())


def test_residual_activity_threshold_uses_one_percent_of_phase_scale():
    samples = [_sample(phase, 0, 0.0) for phase in PHASE_IDS]
    samples += [_sample("P02", 1, 0.02)]
    scales = {phase: (1.0,) * 12 for phase in PHASE_IDS}
    rows = residual_activity_by_phase(
        samples,
        phase_scale_full12=scales,
        numeric_noise_floor_full12=(1e-5,) * 12,
        quantization_floor_full12=(1e-4,) * 12,
    )
    indexed = {row["phase"]: row for row in rows}
    assert indexed["P02"]["nonzero"] is True
    assert indexed["P01"]["nonzero"] is False


def test_primary_phase_score_uses_only_physically_relevant_metrics():
    base = {
        "pitch_rate_rms_rad_s",
        "roll_rate_rms_rad_s",
        "angular_acceleration_rms_rad_s2",
    }
    assert set(PHASE_SCORE_METRICS) == set(PHASE_IDS)
    assert all(base <= set(metrics) for metrics in PHASE_SCORE_METRICS.values())
    assert {
        phase
        for phase, metrics in PHASE_SCORE_METRICS.items()
        if "placement_contact_impulse_n_s" in metrics
    } == {"P03", "P05", "P09", "P12"}
    assert {
        phase
        for phase, metrics in PHASE_SCORE_METRICS.items()
        if "settling_time_s" in metrics
    } == {"P01", "P04", "P08", "P10", "P11"}
    assert {
        phase
        for phase, metrics in PHASE_SCORE_METRICS.items()
        if "action_jerk_rms" in metrics
    } == {"P13"}
    assert set().union(*map(set, PHASE_SCORE_METRICS.values())) == set(
        LOWER_IS_BETTER
    )


def test_inapplicable_zero_baseline_metric_does_not_poison_primary_score():
    baseline = []
    candidate = []
    for phase in PHASE_IDS:
        baseline_row = {"phase": phase, **{name: 1.0 for name in LOWER_IS_BETTER}}
        candidate_row = dict(baseline_row)
        if phase in {"P02", "P08", "P13"}:
            baseline_row["placement_contact_impulse_n_s"] = 0.0
            candidate_row["placement_contact_impulse_n_s"] = 0.01
        baseline.append(baseline_row)
        candidate.append(candidate_row)

    compared = {row["phase"]: row for row in compare_phase_metrics(baseline, candidate)}
    for phase in ("P02", "P08", "P13"):
        assert compared[phase]["placement_contact_impulse_n_s_improvement_fraction"] == -1.0
        assert compared[phase]["primary_phase_score_improvement_fraction"] == 0.0


def test_relevant_phase_specific_metric_contributes_to_primary_score():
    baseline = [
        {"phase": phase, **{name: 1.0 for name in LOWER_IS_BETTER}}
        for phase in PHASE_IDS
    ]
    candidate = [dict(row) for row in baseline]
    candidate[PHASE_IDS.index("P03")]["placement_contact_impulse_n_s"] = 2.0
    candidate[PHASE_IDS.index("P13")]["action_jerk_rms"] = 2.0

    compared = {row["phase"]: row for row in compare_phase_metrics(baseline, candidate)}
    assert compared["P03"]["primary_phase_score_improvement_fraction"] == -0.25
    assert compared["P13"]["primary_phase_score_improvement_fraction"] == -0.25


def test_promotion_enforces_global_and_priority_phase_gates():
    baseline = [EpisodeOutcome(i, True, False, False, False, 100.0) for i in range(5)]
    candidate = [EpisodeOutcome(i, True, False, False, False, 102.0) for i in range(5)]
    phase_rows = [
        {"phase": phase, "primary_phase_score_improvement_fraction": 0.08}
        for phase in PHASE_IDS
    ]
    decision = evaluate_promotion(
        baseline_episodes=baseline,
        candidate_episodes=candidate,
        phase_comparison_rows=phase_rows,
        overall_pitch_rate_improvement=0.06,
        placement_impulse_improvement=0.0,
        home_jerk_improvement=0.0,
        frozen_hashes_unchanged=True,
        recording_runtime_access_count=0,
    )
    assert decision.promoted is True
    assert decision.improved_priority_phase_count == len(PRIORITY_PHASES)


def test_promotion_duration_gate_is_per_seed_not_offset_by_group_mean():
    baseline = [EpisodeOutcome(i, True, False, False, False, 100.0) for i in range(5)]
    candidate_durations = (116.0, 90.0, 90.0, 90.0, 90.0)
    candidate = [
        EpisodeOutcome(i, True, False, False, False, duration)
        for i, duration in enumerate(candidate_durations)
    ]
    phase_rows = [
        {"phase": phase, "primary_phase_score_improvement_fraction": 0.08}
        for phase in PHASE_IDS
    ]

    decision = evaluate_promotion(
        baseline_episodes=baseline,
        candidate_episodes=candidate,
        phase_comparison_rows=phase_rows,
        overall_pitch_rate_improvement=0.06,
        placement_impulse_improvement=0.0,
        home_jerk_improvement=0.0,
        frozen_hashes_unchanged=True,
        recording_runtime_access_count=0,
    )

    assert sum(candidate_durations) / len(candidate_durations) < 100.0
    assert decision.promoted is False
    assert decision.first_failed_gate == "duration_not_over_fsm_by_15pct"
    assert decision.duration_pair_diagnostics["first_violating_seed"] == 0
    assert decision.duration_pair_diagnostics["worst_seed"] == 0
    assert decision.duration_pair_diagnostics[
        "worst_candidate_to_baseline_ratio"
    ] == pytest.approx(1.16)


def test_promotion_rejects_one_priority_phase_over_ten_percent_worse():
    episodes = [EpisodeOutcome(i, True, False, False, False, 100.0) for i in range(5)]
    phase_rows = [
        {
            "phase": phase,
            "primary_phase_score_improvement_fraction": (-0.11 if phase == "P03" else 0.12),
        }
        for phase in PHASE_IDS
    ]
    decision = evaluate_promotion(
        baseline_episodes=episodes,
        candidate_episodes=episodes,
        phase_comparison_rows=phase_rows,
        overall_pitch_rate_improvement=0.10,
        placement_impulse_improvement=0.10,
        home_jerk_improvement=0.10,
        frozen_hashes_unchanged=True,
        recording_runtime_access_count=0,
    )
    assert decision.promoted is False
    assert decision.first_failed_gate == "no_priority_phase_degrades_over_10pct"
