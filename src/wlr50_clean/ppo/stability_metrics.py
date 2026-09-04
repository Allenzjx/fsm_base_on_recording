"""Physical stability, smoothness, activity, and checkpoint-promotion metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


PHASE_IDS = tuple(f"P{i:02d}" for i in range(1, 14))
PRIORITY_PHASES = ("P02", "P03", "P08", "P12", "P13")
PHASE_SCORE_WEIGHTS = {
    phase: (0.12 if phase in PRIORITY_PHASES else 0.05) for phase in PHASE_IDS
}
LOWER_IS_BETTER = (
    "pitch_rate_rms_rad_s",
    "roll_rate_rms_rad_s",
    "angular_acceleration_rms_rad_s2",
    "placement_contact_impulse_n_s",
    "settling_time_s",
    "action_jerk_rms",
)

# Fixed non-overlapping bands for the public residual spectrum.  Energy at
# DC is deliberately excluded: the plot is intended to show temporal action
# activity, while a constant offset is already represented by residual RMS.
# The final band is clipped to the actual Nyquist frequency by the FFT.
RESIDUAL_SPECTRUM_BANDS_HZ = (
    (0.0, 0.5, "0p0_0p5"),
    (0.5, 1.0, "0p5_1p0"),
    (1.0, 2.0, "1p0_2p0"),
    (2.0, 3.0, "2p0_3p0"),
    (3.0, math.inf, "3p0_nyquist"),
)

# The paired artifact retains every metric in ``LOWER_IS_BETTER``, but a
# phase's primary score must only average quantities that are physically
# meaningful in that phase.  In particular, a zero placement impulse outside
# a placement phase is an inapplicable metric, not a perfect denominator.
_BASE_STABILITY_SCORE_METRICS = (
    "pitch_rate_rms_rad_s",
    "roll_rate_rms_rad_s",
    "angular_acceleration_rms_rad_s2",
)
PHASE_SCORE_METRICS = {
    "P01": _BASE_STABILITY_SCORE_METRICS + ("settling_time_s",),
    "P02": _BASE_STABILITY_SCORE_METRICS,
    "P03": _BASE_STABILITY_SCORE_METRICS + ("placement_contact_impulse_n_s",),
    "P04": _BASE_STABILITY_SCORE_METRICS + ("settling_time_s",),
    "P05": _BASE_STABILITY_SCORE_METRICS + ("placement_contact_impulse_n_s",),
    "P06": _BASE_STABILITY_SCORE_METRICS,
    "P07": _BASE_STABILITY_SCORE_METRICS,
    "P08": _BASE_STABILITY_SCORE_METRICS + ("settling_time_s",),
    "P09": _BASE_STABILITY_SCORE_METRICS + ("placement_contact_impulse_n_s",),
    "P10": _BASE_STABILITY_SCORE_METRICS + ("settling_time_s",),
    "P11": _BASE_STABILITY_SCORE_METRICS + ("settling_time_s",),
    "P12": _BASE_STABILITY_SCORE_METRICS + ("placement_contact_impulse_n_s",),
    "P13": _BASE_STABILITY_SCORE_METRICS + ("action_jerk_rms",),
}


class MetricsError(ValueError):
    pass


def _finite_vector(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != size or any(not math.isfinite(v) for v in result):
        raise MetricsError(f"{label} must contain {size} finite values")
    return result


@dataclass(frozen=True, slots=True)
class StabilitySample:
    time_s: float
    phase: str
    lifecycle: str
    roll_error_rad: float
    pitch_error_rad: float
    roll_rate_rad_s: float
    pitch_rate_rad_s: float
    yaw_rate_rad_s: float = 0.0
    active_contact_normal_force_n: float = 0.0
    active_contact_baseline_n: float = 0.0
    wheel_slip4: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    active_leg_clearance_m: float = 0.0
    active_leg_vertical_velocity_m_s: float = 0.0
    home_pose_error_rms_deg: float = 0.0
    residual_full12: tuple[float, ...] = (0.0,) * 12
    nominal_full12: tuple[float, ...] = (0.0,) * 12
    applied_full12: tuple[float, ...] = (0.0,) * 12

    def __post_init__(self) -> None:
        if self.phase not in PHASE_IDS:
            raise MetricsError(f"unknown phase {self.phase!r}")
        scalar_names = (
            "time_s", "roll_error_rad", "pitch_error_rad", "roll_rate_rad_s",
            "pitch_rate_rad_s", "yaw_rate_rad_s", "active_contact_normal_force_n",
            "active_contact_baseline_n", "active_leg_clearance_m",
            "active_leg_vertical_velocity_m_s", "home_pose_error_rms_deg",
        )
        for name in scalar_names:
            if not math.isfinite(float(getattr(self, name))):
                raise MetricsError(f"{name} is non-finite")
        object.__setattr__(self, "wheel_slip4", _finite_vector(self.wheel_slip4, 4, "wheel_slip4"))
        for name in ("residual_full12", "nominal_full12", "applied_full12"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), 12, name))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _p95_abs(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95.0)) if values.size else 0.0


def _vector_rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _high_frequency_fraction(signal: np.ndarray, dt_s: float, cutoff_hz: float = 3.0) -> float:
    if signal.shape[0] < 4:
        return 0.0
    centered = signal - np.mean(signal, axis=0, keepdims=True)
    spectrum = np.square(np.abs(np.fft.rfft(centered, axis=0)))
    frequencies = np.fft.rfftfreq(signal.shape[0], d=dt_s)
    total = float(np.sum(spectrum[1:]))
    if total <= 1.0e-20:
        return 0.0
    return float(np.sum(spectrum[frequencies >= cutoff_hz]) / total)


def residual_spectrum_band_fractions(
    signal: np.ndarray,
    dt_s: float,
) -> dict[str, float]:
    """Return real FFT energy fractions in the fixed public frequency bands.

    ``signal`` is a time-by-channel array sampled at ``dt_s``.  The transform
    is performed on every channel after removing its mean, then channel power
    is summed.  Fractions therefore describe the actual multichannel residual
    time series without inventing PSD samples from aggregate metrics.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 2:
        raise MetricsError("residual spectrum signal must be a 2D time-by-channel array")
    if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
        raise MetricsError("residual spectrum dt_s must be finite and positive")
    if not np.all(np.isfinite(values)):
        raise MetricsError("residual spectrum signal contains non-finite values")
    keys = tuple(
        f"residual_spectral_energy_fraction_{label}_hz"
        for _lower, _upper, label in RESIDUAL_SPECTRUM_BANDS_HZ
    )
    if values.shape[0] < 4:
        return {key: 0.0 for key in keys}

    centered = values - np.mean(values, axis=0, keepdims=True)
    power = np.square(np.abs(np.fft.rfft(centered, axis=0)))
    frequencies = np.fft.rfftfreq(values.shape[0], d=float(dt_s))
    # Remove DC from both numerator and denominator.  Half-open finite bands
    # assign every positive-frequency FFT bin exactly once; the last band
    # includes Nyquist.
    positive = frequencies > 0.0
    total = float(np.sum(power[positive]))
    if total <= 1.0e-20:
        return {key: 0.0 for key in keys}

    result: dict[str, float] = {}
    for (lower, upper, _label), key in zip(
        RESIDUAL_SPECTRUM_BANDS_HZ, keys, strict=True
    ):
        if math.isinf(upper):
            selected = (frequencies >= lower) & positive
        else:
            selected = (frequencies >= lower) & (frequencies < upper) & positive
        result[key] = float(np.sum(power[selected]) / total)
    return result


def _settling_time(
    *,
    time: np.ndarray,
    roll: np.ndarray,
    pitch: np.ndarray,
    roll_rate: np.ndarray,
    pitch_rate: np.ndarray,
    contact_excess: np.ndarray,
    dt_s: float,
    attitude_tolerance_rad: float,
    rate_tolerance_rad_s: float,
    hold_s: float,
) -> float:
    """Time after the phase's largest disturbance until a sustained quiet window."""

    if time.size == 0:
        return 0.0
    if np.any(contact_excess > 0.0):
        anchor = int(np.argmax(contact_excess))
    else:
        disturbance = np.sqrt(
            np.square(roll)
            + np.square(pitch)
            + np.square(roll_rate)
            + np.square(pitch_rate)
        )
        anchor = int(np.argmax(disturbance))
    stable = (
        (np.abs(roll) <= attitude_tolerance_rad)
        & (np.abs(pitch) <= attitude_tolerance_rad)
        & (np.abs(roll_rate) <= rate_tolerance_rad_s)
        & (np.abs(pitch_rate) <= rate_tolerance_rad_s)
    )
    hold_count = max(1, int(math.ceil(hold_s / dt_s)))
    for index in range(anchor, len(stable) - hold_count + 1):
        if bool(np.all(stable[index : index + hold_count])):
            return max(0.0, float(time[index] - time[anchor]))
    return max(dt_s, float(time[-1] - time[anchor] + dt_s))


def summarize_phase_samples(
    samples: Iterable[StabilitySample], *,
    phase_scale_full12: Mapping[str, Sequence[float]],
    physics_hz: float = 120.0,
    settling_attitude_tolerance_rad: float = math.radians(2.0),
    settling_rate_tolerance_rad_s: float = math.radians(5.0),
    settling_hold_s: float = 0.25,
) -> list[dict[str, float | str | int]]:
    rows = list(samples)
    if not rows:
        raise MetricsError("stability sample stream is empty")
    dt = 1.0 / float(physics_hz)
    result: list[dict[str, float | str | int]] = []
    for phase in PHASE_IDS:
        group = [row for row in rows if row.phase == phase]
        if not group:
            continue
        time = np.asarray([row.time_s for row in group], dtype=float)
        roll = np.asarray([row.roll_error_rad for row in group], dtype=float)
        pitch = np.asarray([row.pitch_error_rad for row in group], dtype=float)
        roll_rate = np.asarray([row.roll_rate_rad_s for row in group], dtype=float)
        pitch_rate = np.asarray([row.pitch_rate_rad_s for row in group], dtype=float)
        angular_rate = np.stack((roll_rate, pitch_rate, np.asarray([r.yaw_rate_rad_s for r in group])), axis=1)
        angular_acceleration = np.diff(angular_rate, axis=0) / dt
        force = np.asarray(
            [max(0.0, r.active_contact_normal_force_n - r.active_contact_baseline_n) for r in group],
            dtype=float,
        )
        slip = np.asarray([row.wheel_slip4 for row in group], dtype=float)
        residual = np.asarray([row.residual_full12 for row in group], dtype=float)
        scale = np.asarray(
            _finite_vector(phase_scale_full12[phase], 12, f"{phase} residual scale"),
            dtype=float,
        )
        if np.any(scale < 0.0) or not np.any(scale > 0.0):
            raise MetricsError(f"{phase} residual scale must be nonnegative and nonzero")
        disabled = scale <= 0.0
        if np.any(np.abs(residual[:, disabled]) > 1.0e-12):
            raise MetricsError(f"{phase} has residual activity on a zero-scale channel")
        normalized_residual = np.divide(
            residual,
            scale[None, :],
            out=np.zeros_like(residual),
            where=scale[None, :] > 0.0,
        )
        nominal = np.asarray([row.nominal_full12 for row in group], dtype=float)
        applied = np.asarray([row.applied_full12 for row in group], dtype=float)
        residual_rate = np.diff(residual, axis=0)
        residual_jerk = np.diff(residual, n=2, axis=0)
        nominal_rate = np.diff(nominal, axis=0)
        applied_rate = np.diff(applied, axis=0)
        applied_jerk = np.diff(applied, n=2, axis=0)
        result.append(
            {
                "phase": phase,
                "sample_count": len(group),
                "duration_s": max(dt, float(time[-1] - time[0] + dt)),
                "roll_rms_rad": _rms(roll),
                "pitch_rms_rad": _rms(pitch),
                "roll_rate_rms_rad_s": _rms(roll_rate),
                "pitch_rate_rms_rad_s": _rms(pitch_rate),
                "pitch_rate_p95_abs_rad_s": _p95_abs(pitch_rate),
                "pitch_rate_peak_abs_rad_s": float(np.max(np.abs(pitch_rate))),
                "roll_rate_peak_abs_rad_s": float(np.max(np.abs(roll_rate))),
                "angular_acceleration_rms_rad_s2": _vector_rms(angular_acceleration),
                "placement_contact_impulse_n_s": float(np.sum(force) * dt),
                "settling_time_s": _settling_time(
                    time=time,
                    roll=roll,
                    pitch=pitch,
                    roll_rate=roll_rate,
                    pitch_rate=pitch_rate,
                    contact_excess=force,
                    dt_s=dt,
                    attitude_tolerance_rad=float(settling_attitude_tolerance_rad),
                    rate_tolerance_rad_s=float(settling_rate_tolerance_rad_s),
                    hold_s=float(settling_hold_s),
                ),
                "wheel_slip_integral": float(np.sum(np.abs(slip)) * dt),
                "active_leg_min_clearance_m": float(min(row.active_leg_clearance_m for row in group)),
                "active_leg_contact_vertical_speed_abs_m_s": float(
                    max(abs(row.active_leg_vertical_velocity_m_s) for row in group)
                ),
                "home_pose_error_rms_deg": _rms(
                    np.asarray([row.home_pose_error_rms_deg for row in group], dtype=float)
                ),
                "residual_rms": _vector_rms(residual),
                "residual_peak": float(np.max(np.abs(residual))),
                "residual_action_rate_rms": _vector_rms(residual_rate),
                "residual_action_jerk_rms": _vector_rms(residual_jerk),
                "nominal_action_rate_rms": _vector_rms(nominal_rate),
                "applied_action_rate_rms": _vector_rms(applied_rate),
                "action_jerk_rms": _vector_rms(applied_jerk),
                "residual_high_frequency_fraction": _high_frequency_fraction(residual, dt),
                "applied_high_frequency_fraction": _high_frequency_fraction(applied, dt),
                "residual_spectrum_normalization": "phase_scale_full12",
                **residual_spectrum_band_fractions(normalized_residual, dt),
            }
        )
    return result


def _safe_improvement(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise MetricsError("comparison metric is non-finite")
    if abs(baseline) <= 1.0e-12:
        return 0.0 if abs(candidate) <= 1.0e-12 else -1.0
    return (baseline - candidate) / abs(baseline)


def phase_stability_cost(row: Mapping[str, float | str | int]) -> float:
    """Dimensionless aggregate; callers must normalize rows to paired FSM first."""

    phase = str(row.get("phase", ""))
    if phase not in PHASE_SCORE_METRICS:
        raise MetricsError("phase stability cost requires a P01-P13 phase")
    return float(np.mean([float(row[name]) for name in PHASE_SCORE_METRICS[phase]]))


def compare_phase_metrics(
    baseline_rows: Iterable[Mapping[str, float | str | int]],
    candidate_rows: Iterable[Mapping[str, float | str | int]],
) -> list[dict[str, float | str]]:
    baseline = {str(row["phase"]): row for row in baseline_rows}
    candidate = {str(row["phase"]): row for row in candidate_rows}
    if set(baseline) != set(PHASE_IDS) or set(candidate) != set(PHASE_IDS):
        raise MetricsError("paired phase comparison requires all P01-P13 rows")
    result = []
    for phase in PHASE_IDS:
        improvements = {
            name: _safe_improvement(float(baseline[phase][name]), float(candidate[phase][name]))
            for name in LOWER_IS_BETTER
        }
        result.append(
            {
                "phase": phase,
                **{f"{name}_improvement_fraction": value for name, value in improvements.items()},
                "primary_phase_score_improvement_fraction": float(
                    np.mean(
                        [
                            improvements[name]
                            for name in PHASE_SCORE_METRICS[phase]
                        ]
                    )
                ),
            }
        )
    return result


def global_stability_improvement(
    phase_comparison_rows: Iterable[Mapping[str, float | str]],
) -> float:
    indexed = {str(row["phase"]): row for row in phase_comparison_rows}
    if set(indexed) != set(PHASE_IDS):
        raise MetricsError("global stability score requires P01-P13")
    return sum(
        PHASE_SCORE_WEIGHTS[phase]
        * float(indexed[phase]["primary_phase_score_improvement_fraction"])
        for phase in PHASE_IDS
    )


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    seed: int
    task_success: bool
    body_collision: bool
    wheel_only_climb: bool
    safety_abort: bool
    duration_s: float
    physics_explosion_or_fall: bool = False
    completed_p01_p13: bool = True


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    first_failed_gate: str | None
    checks: Mapping[str, bool]
    global_stability_improvement_fraction: float
    improved_priority_phase_count: int
    duration_pair_diagnostics: Mapping[str, object]


def _paired_duration_diagnostics(
    baseline_episodes: Sequence[EpisodeOutcome],
    candidate_episodes: Sequence[EpisodeOutcome],
) -> tuple[bool, dict[str, object]]:
    """Evaluate the 15% duration cap on each explicitly paired seed.

    A group mean is retained as a reporting diagnostic, but it has no gate
    authority: a fast candidate on one seed must not conceal a slowdown over
    15% on another matched seed.
    """

    rows: list[dict[str, object]] = []
    for baseline, candidate in zip(
        baseline_episodes, candidate_episodes, strict=True
    ):
        baseline_duration = float(baseline.duration_s)
        candidate_duration = float(candidate.duration_s)
        if (
            not math.isfinite(baseline_duration)
            or not math.isfinite(candidate_duration)
            or baseline_duration <= 0.0
            or candidate_duration < 0.0
        ):
            raise MetricsError(
                "paired promotion episode durations must be finite with a "
                "positive baseline and non-negative candidate"
            )
        allowed = baseline_duration * 1.15
        ratio = candidate_duration / baseline_duration
        rows.append(
            {
                "seed": int(baseline.seed),
                "baseline_duration_s": baseline_duration,
                "candidate_duration_s": candidate_duration,
                "allowed_duration_s": allowed,
                "candidate_to_baseline_ratio": ratio,
                "passed": candidate_duration <= allowed,
            }
        )
    failed_rows = [row for row in rows if row["passed"] is not True]
    worst = max(rows, key=lambda row: float(row["candidate_to_baseline_ratio"]))
    baseline_mean = float(np.mean([float(row["baseline_duration_s"]) for row in rows]))
    candidate_mean = float(np.mean([float(row["candidate_duration_s"]) for row in rows]))
    return not failed_rows, {
        "gate_semantics": "all_matched_seed_pairs_candidate_le_1p15_times_baseline",
        "paired_rows": tuple(rows),
        "baseline_mean_duration_s": baseline_mean,
        "candidate_mean_duration_s": candidate_mean,
        "mean_candidate_to_baseline_ratio": candidate_mean / baseline_mean,
        "first_violating_seed": (
            None if not failed_rows else int(failed_rows[0]["seed"])
        ),
        "worst_seed": int(worst["seed"]),
        "worst_candidate_to_baseline_ratio": float(
            worst["candidate_to_baseline_ratio"]
        ),
    }


def evaluate_promotion(
    *,
    baseline_episodes: Sequence[EpisodeOutcome],
    candidate_episodes: Sequence[EpisodeOutcome],
    phase_comparison_rows: Sequence[Mapping[str, float | str]],
    overall_pitch_rate_improvement: float,
    placement_impulse_improvement: float,
    home_jerk_improvement: float,
    frozen_hashes_unchanged: bool,
    recording_runtime_access_count: int,
) -> PromotionDecision:
    if not baseline_episodes or len(baseline_episodes) != len(candidate_episodes):
        raise MetricsError("promotion requires equal non-empty paired episode sets")
    if [e.seed for e in baseline_episodes] != [e.seed for e in candidate_episodes]:
        raise MetricsError("baseline and candidate seeds are not paired")
    baseline_success = sum(e.task_success for e in baseline_episodes) / len(baseline_episodes)
    candidate_success = sum(e.task_success for e in candidate_episodes) / len(candidate_episodes)
    duration_gate, duration_diagnostics = _paired_duration_diagnostics(
        baseline_episodes, candidate_episodes
    )
    phase_rows = {str(row["phase"]): row for row in phase_comparison_rows}
    stability = global_stability_improvement(phase_comparison_rows)
    priority_scores = [float(phase_rows[p]["primary_phase_score_improvement_fraction"]) for p in PRIORITY_PHASES]
    improved_priority = sum(score > 0.0 for score in priority_scores)
    checks = {
        "p01_p13_completed": all(e.completed_p01_p13 for e in candidate_episodes),
        "task_success_rate_not_below_fsm": candidate_success >= baseline_success,
        "body_collision_zero": not any(e.body_collision for e in candidate_episodes),
        "wheel_only_climb_zero": not any(e.wheel_only_climb for e in candidate_episodes),
        "fall_or_physics_explosion_zero": not any(
            e.physics_explosion_or_fall for e in candidate_episodes
        ),
        "safety_abort_zero": not any(e.safety_abort for e in candidate_episodes),
        "duration_each_under_200_s": all(e.duration_s <= 200.0 for e in candidate_episodes),
        "duration_not_over_fsm_by_15pct": duration_gate,
        "frozen_hashes_unchanged": bool(frozen_hashes_unchanged),
        "recording_runtime_access_zero": int(recording_runtime_access_count) == 0,
        "global_stability_improvement_at_least_5pct": stability >= 0.05,
        "at_least_4_of_5_priority_phases_improve": improved_priority >= 4,
        "no_priority_phase_degrades_over_10pct": all(score >= -0.10 for score in priority_scores),
        "one_visual_key_metric_gate": (
            overall_pitch_rate_improvement >= 0.05
            or placement_impulse_improvement >= 0.10
            or home_jerk_improvement >= 0.10
        ),
    }
    failed = next((name for name, passed in checks.items() if not passed), None)
    return PromotionDecision(
        promoted=failed is None,
        first_failed_gate=failed,
        checks=checks,
        global_stability_improvement_fraction=stability,
        improved_priority_phase_count=improved_priority,
        duration_pair_diagnostics=duration_diagnostics,
    )


def residual_activity_by_phase(
    samples: Iterable[StabilitySample],
    *,
    phase_scale_full12: Mapping[str, Sequence[float]],
    numeric_noise_floor_full12: Sequence[float],
    quantization_floor_full12: Sequence[float],
    dt_s: float = 1.0 / 120.0,
) -> list[dict[str, float | str | int | bool]]:
    rows = list(samples)
    noise = _finite_vector(numeric_noise_floor_full12, 12, "numeric noise floor")
    quantization = _finite_vector(quantization_floor_full12, 12, "quantization floor")
    result = []
    for phase in PHASE_IDS:
        scale = _finite_vector(phase_scale_full12[phase], 12, f"{phase} scale")
        thresholds = tuple(max(q, 3.0 * n, 0.01 * s) for q, n, s in zip(quantization, noise, scale, strict=True))
        values = np.asarray([row.residual_full12 for row in rows if row.phase == phase], dtype=float)
        if values.size == 0:
            active = np.zeros((0, 12), dtype=bool)
            rms = peak = 0.0
        else:
            active = np.abs(values) > np.asarray(thresholds)[None, :]
            rms = _vector_rms(values / np.maximum(np.asarray(scale), 1.0e-12)[None, :])
            peak = float(np.max(np.abs(values) / np.maximum(np.asarray(scale), 1.0e-12)[None, :]))
        result.append(
            {
                "phase": phase,
                "normalized_residual_rms": rms,
                "normalized_residual_peak": peak,
                "active_channel_count": int(np.count_nonzero(np.any(active, axis=0))) if active.size else 0,
                "residual_duration_s": float(np.count_nonzero(np.any(active, axis=1)) * dt_s) if active.size else 0.0,
                "nonzero": bool(active.size and np.any(active)),
            }
        )
    return result


def promotion_to_dict(decision: PromotionDecision) -> dict[str, object]:
    return asdict(decision)
