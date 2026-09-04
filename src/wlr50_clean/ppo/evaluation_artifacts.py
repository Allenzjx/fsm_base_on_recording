"""Fail-closed orchestration and publication for paired PPO evaluation.

This module deliberately sits above :mod:`wlr50_clean.ppo.evaluation`.  The
live-run evaluator remains the single source of physical metrics and promotion
semantics; this layer only:

* derives residual-activity thresholds from versioned action/environment data;
* evaluates immutable canonical episode directories as a checked sequence; and
* publishes deterministic CSV/JSON evidence without replacing prior results.

No filename or caller-provided label can turn a failed candidate into an
``improved`` checkpoint.  The machine-readable decision always contains the
first failed gate returned by the authoritative paired evaluator.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    WHEEL_ORDER,
)

from .artifacts import ArtifactError, atomic_write_csv, atomic_write_json, sha256_file
from .evaluation import (
    LiveRunEvaluation,
    ResidualActivityCalibration,
    evaluate_live_run,
    paired_baseline_candidate_promotion,
)
from .phase_action_masks_v2 import (
    DEFAULT_PHASE_ACTION_CONFIG_V2,
    load_phase_action_masks_v2,
)
from .stability_metrics import PHASE_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENVIRONMENT_LOCK = PROJECT_ROOT / "configs" / "environment_lock.json"
EVALUATION_ARTIFACT_SCHEMA = "wlr50_clean.ppo_evaluation_artifacts.v1"
RESIDUAL_CALIBRATION_SCHEMA = "wlr50_clean.residual_activity_calibration.v1"
DEFAULT_REWARD_STREAM_FILENAME = "reward_15hz.jsonl"
DEFAULT_METRICS_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "ppo_phase_v1" / "metrics"
BASELINE_VALIDATION_SEEDS = (2001, 2002, 2003, 2004, 2005)

CANONICAL_EPISODE_FILES = (
    "observation_120hz.jsonl",
    "full12_commands_120hz.jsonl",
    "state_transitions.jsonl",
    "task_events.jsonl",
    "reward_15hz.jsonl",
    "policy_trace.jsonl",
    "trial_manifest.json",
    "episode_summary.json",
)
NONEMPTY_CANONICAL_EPISODE_FILES = frozenset(
    name for name in CANONICAL_EPISODE_FILES if name != "state_transitions.jsonl"
)

BASELINE_EPISODE_FILENAME = "fsm_baseline_episode_metrics.csv"
BASELINE_PHASE_FILENAME = "fsm_baseline_phase_metrics.csv"
BASELINE_EVALUATION_MANIFEST_FILENAME = "fsm_baseline_evaluation_manifest.json"
CANDIDATE_EPISODE_FILENAME = "candidate_episode_metrics.csv"
CANDIDATE_PHASE_FILENAME = "candidate_phase_metrics.csv"
CHECKPOINT_COMPARISON_FILENAME = "checkpoint_comparison.csv"
PHASE_COMPARISON_FILENAME = "phase_metric_comparison.csv"
RESIDUAL_ACTIVITY_FILENAME = "residual_activity_by_phase.csv"
REWARD_CONTRIBUTION_FILENAME = "reward_contribution_by_phase.csv"
TERMINATION_SUMMARY_FILENAME = "termination_summary.csv"
PROMOTION_DECISION_FILENAME = "promotion_decision.json"


class EvaluationArtifactError(ArtifactError):
    """Evaluation inputs are incomparable or output evidence cannot be trusted."""


@dataclass(frozen=True, slots=True)
class FreshProcessEpisodeBatch:
    """Validated canonical episodes captured by independent Isaac workers."""

    role: str
    seeds: tuple[int, ...]
    canonical_episode_dirs: tuple[Path, ...]
    episode_rows: tuple[Mapping[str, Any], ...]
    worker_rows: tuple[Mapping[str, Any], ...]


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def collect_fresh_process_episode_workers(
    worker_run_directories: Sequence[str | Path],
    *,
    seeds: Sequence[int],
    role: str,
    checkpoint_path: str | Path | None = None,
) -> FreshProcessEpisodeBatch:
    """Collect a complete baseline or candidate batch without mutating it.

    Every worker must be a finalized, successful orchestration process that
    captured exactly one episode.  ``role='candidate'`` additionally binds all
    workers to one checkpoint hash.  The returned canonical directories are
    ready for :func:`evaluate_canonical_episode_dirs`, so baseline CSV export
    and paired promotion share one provenance path.
    """

    selected_role = str(role).strip().lower()
    if selected_role not in {"baseline", "candidate"}:
        raise EvaluationArtifactError("fresh-process worker role must be baseline or candidate")
    run_dirs = tuple(Path(value).resolve() for value in worker_run_directories)
    seed_values: list[int] = []
    for value in seeds:
        if isinstance(value, bool):
            raise EvaluationArtifactError(
                "fresh-process worker seeds must be non-negative integers"
            )
        try:
            selected = int(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationArtifactError(
                "fresh-process worker seeds must be non-negative integers"
            ) from exc
        if selected < 0 or selected != value:
            raise EvaluationArtifactError(
                "fresh-process worker seeds must be non-negative integers"
            )
        seed_values.append(selected)
    selected_seeds = tuple(seed_values)
    if not run_dirs or len(run_dirs) != len(selected_seeds):
        raise EvaluationArtifactError(
            "worker directories and expected seeds must be equal, non-empty sequences"
        )
    if len(set(run_dirs)) != len(run_dirs):
        raise EvaluationArtifactError("fresh-process worker directories must be unique")
    if (
        any(seed < 0 for seed in selected_seeds)
        or len(set(selected_seeds)) != len(selected_seeds)
    ):
        raise EvaluationArtifactError("fresh-process worker seeds must be unique and non-negative")

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).resolve()
    checkpoint_hash = None
    if selected_role == "candidate":
        if checkpoint is None or not checkpoint.is_file():
            raise EvaluationArtifactError("candidate workers require one existing checkpoint")
        checkpoint_hash = sha256_file(checkpoint)
    elif checkpoint is not None:
        raise EvaluationArtifactError("baseline workers must not name a PPO checkpoint")

    canonical_dirs: list[Path] = []
    episodes: list[Mapping[str, Any]] = []
    workers: list[Mapping[str, Any]] = []
    for run_dir, expected_seed in zip(run_dirs, selected_seeds, strict=True):
        lifecycle_path = run_dir / "run_manifest.json"
        result_name = (
            "checkpoint_evaluation.json"
            if selected_role == "candidate"
            else "acceptance.json"
        )
        result_path = run_dir / result_name
        lifecycle = _load_json_object(lifecycle_path, "worker lifecycle manifest")
        result = _load_json_object(result_path, f"{selected_role} worker result")
        if lifecycle.get("lifecycle") != "SUCCEEDED" or lifecycle.get("exit_code") != 0:
            raise EvaluationArtifactError(
                f"fresh-process worker did not finalize successfully: {run_dir}"
            )
        if selected_role == "candidate":
            if (
                result.get("schema") != "wlr50_clean.ppo_checkpoint_evaluation.v1"
                or result.get("fresh_process_single_episode") is not True
                or result.get("vec_env_step_called") is not False
                or result.get("deterministic_mean_policy") is not True
            ):
                raise EvaluationArtifactError(
                    f"candidate worker contract is invalid: {run_dir}"
                )
            if Path(str(result.get("checkpoint", ""))).resolve() != checkpoint:
                raise EvaluationArtifactError(
                    f"candidate worker used a different checkpoint: {run_dir}"
                )
            if result.get("checkpoint_sha256") != checkpoint_hash:
                raise EvaluationArtifactError(
                    f"candidate worker checkpoint hash differs: {run_dir}"
                )
        elif (
            result.get("schema") != "wlr50_clean.live_residual_gate.v1"
            or result.get("mode") != "zero"
        ):
            raise EvaluationArtifactError(f"baseline worker contract is invalid: {run_dir}")
        if result.get("episode_count") != 1 or len(result.get("episodes", ())) != 1:
            raise EvaluationArtifactError(
                f"fresh-process worker did not capture exactly one episode: {run_dir}"
            )

        episode = dict(result["episodes"][0])
        if int(episode.get("seed", -1)) != expected_seed:
            raise EvaluationArtifactError(
                f"worker seed differs from expected seed {expected_seed}: {run_dir}"
            )
        raw_episode_dir = episode.get("canonical_episode_dir")
        if raw_episode_dir:
            episode_dir = Path(str(raw_episode_dir)).resolve()
        else:
            trial_path = Path(str(episode.get("trial_manifest_path", ""))).resolve()
            episode_dir = trial_path.parent
        if not _path_within(episode_dir, run_dir) or episode_dir.parent != run_dir:
            raise EvaluationArtifactError(
                f"canonical episode escaped its worker directory: {run_dir}"
            )
        missing = [
            name
            for name in CANONICAL_EPISODE_FILES
            if not (episode_dir / name).is_file()
            or (
                name in NONEMPTY_CANONICAL_EPISODE_FILES
                and (episode_dir / name).stat().st_size <= 0
            )
        ]
        if missing:
            raise EvaluationArtifactError(
                f"canonical episode is missing {missing}: {episode_dir}"
            )
        trial_path = episode_dir / "trial_manifest.json"
        trial = _load_json_object(trial_path, "canonical trial manifest")
        if int(trial.get("seed", -1)) != expected_seed:
            raise EvaluationArtifactError(
                f"canonical trial seed differs from expected seed {expected_seed}: {run_dir}"
            )
        if trial.get("action_projection_audit", {}).get(
            "exact_pair_contact_contract_valid"
        ) is not True:
            raise EvaluationArtifactError(
                f"exact-pair contact contract was not verified: {run_dir}"
            )
        episode["canonical_episode_dir"] = str(episode_dir)
        canonical_dirs.append(episode_dir)
        episodes.append(episode)
        workers.append(
            {
                "role": selected_role,
                "seed": expected_seed,
                "run_dir": str(run_dir),
                "run_manifest_sha256": sha256_file(lifecycle_path),
                "worker_result": str(result_path),
                "worker_result_sha256": sha256_file(result_path),
                "worker_gate_passed": result.get("passed") is True,
                "mode_specific_checks": dict(result.get("mode_specific_checks", {})),
                "canonical_episode_dir": str(episode_dir),
                "trial_manifest_sha256": sha256_file(trial_path),
            }
        )
    return FreshProcessEpisodeBatch(
        role=selected_role,
        seeds=selected_seeds,
        canonical_episode_dirs=tuple(canonical_dirs),
        episode_rows=tuple(episodes),
        worker_rows=tuple(workers),
    )


def _finite_positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationArtifactError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise EvaluationArtifactError(f"{label} must be finite and positive")
    return result


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationArtifactError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationArtifactError(f"{label} is invalid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationArtifactError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class VersionedResidualActivityCalibration:
    """Residual activity calibration plus its reproducible derivation evidence."""

    calibration: ResidualActivityCalibration
    phase_action_config: Path
    phase_action_config_sha256: str
    environment_lock: Path
    environment_lock_sha256: str
    servo_target_quantization_margin_rad: float
    servo_command_quantization_floor_deg: float
    wheel_velocity_limit_rad_s: float
    wheel_command_quantization_floor_rad_s: float
    numeric_representation: str
    numeric_noise_derivation: str
    activity_threshold_formula: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_CALIBRATION_SCHEMA,
            "phase_action_config": str(self.phase_action_config),
            "phase_action_config_sha256": self.phase_action_config_sha256,
            "environment_lock": str(self.environment_lock),
            "environment_lock_sha256": self.environment_lock_sha256,
            "full12_order": list(FULL12_ORDER),
            "phase_ids": list(PHASE_IDS),
            "phase_scale_full12": {
                phase: list(self.calibration.phase_scale_full12[phase])
                for phase in PHASE_IDS
            },
            "numeric_noise_floor_full12": list(
                self.calibration.numeric_noise_floor_full12
            ),
            "quantization_floor_full12": list(
                self.calibration.quantization_floor_full12
            ),
            "servo_target_quantization_margin_rad": (
                self.servo_target_quantization_margin_rad
            ),
            "servo_command_quantization_floor_deg": (
                self.servo_command_quantization_floor_deg
            ),
            "wheel_velocity_limit_rad_s": self.wheel_velocity_limit_rad_s,
            "wheel_command_quantization_floor_rad_s": (
                self.wheel_command_quantization_floor_rad_s
            ),
            "numeric_representation": self.numeric_representation,
            "numeric_noise_derivation": self.numeric_noise_derivation,
            "activity_threshold_formula": self.activity_threshold_formula,
        }


def build_versioned_residual_activity_calibration(
    *,
    phase_action_config: str | Path = DEFAULT_PHASE_ACTION_CONFIG_V2,
    environment_lock: str | Path = DEFAULT_ENVIRONMENT_LOCK,
) -> VersionedResidualActivityCalibration:
    """Derive activity floors without a hand-selected nonzero epsilon.

    Servo command quantization is the explicit USD target-quantization margin
    in the frozen environment, converted from radians to command-space degrees.
    Wheel target quantization is one IEEE-754 binary32 ULP at the configured
    wheel velocity limit, matching the float32 actuator target boundary.  The
    much smaller logging-noise floor is one binary64 ULP at the largest
    configured phase residual scale for each channel.  The final threshold is
    still computed by :func:`residual_activity_by_phase` using the mandated
    ``max(quantization, 3 * noise, 1% * phase scale)`` formula.
    """

    action_path = Path(phase_action_config).resolve()
    environment_path = Path(environment_lock).resolve()
    action = load_phase_action_masks_v2(action_path)
    environment = _load_json_object(environment_path, "environment lock")
    if environment.get("schema") != "wlr50_clean.environment_lock.v1":
        raise EvaluationArtifactError("unexpected environment-lock schema")
    if tuple(environment.get("canonical_action_order_full12", ())) != FULL12_ORDER:
        raise EvaluationArtifactError(
            "environment lock Full12 order differs from the canonical action ABI"
        )
    if tuple(environment.get("servo_order8", ())) != SERVO_ORDER:
        raise EvaluationArtifactError("environment lock servo order is not canonical")
    if tuple(environment.get("wheel_order4", ())) != WHEEL_ORDER:
        raise EvaluationArtifactError("environment lock wheel order is not canonical")

    joint_limits = environment.get("authoritative_command_space_joint_limits")
    actuators = environment.get("actuators")
    if not isinstance(joint_limits, Mapping) or not isinstance(actuators, Mapping):
        raise EvaluationArtifactError(
            "environment lock omits actuator or command-quantization evidence"
        )
    quantization_margin_rad = _finite_positive(
        joint_limits.get("target_quantization_margin_rad"),
        "target_quantization_margin_rad",
    )
    wheel_limit = _finite_positive(
        actuators.get("wheel_velocity_limit_rad_s"),
        "wheel_velocity_limit_rad_s",
    )
    servo_quantization_deg = math.degrees(quantization_margin_rad)
    wheel_quantization = float(np.spacing(np.float32(wheel_limit)))
    if not math.isfinite(wheel_quantization) or wheel_quantization <= 0.0:
        raise EvaluationArtifactError("cannot derive wheel float32 quantization floor")

    phase_scales = {
        phase: action.physical_scale_for(phase) for phase in PHASE_IDS
    }
    maximum_scales = tuple(
        max(float(phase_scales[phase][index]) for phase in PHASE_IDS)
        for index in range(len(FULL12_ORDER))
    )
    # JSON numbers are decoded into Python binary64 floats.  One ULP at the
    # largest configured phase scale is therefore an evidence-based lower
    # bound for serialization/numeric noise, not a tuned activity threshold.
    numeric_noise = tuple(
        math.ulp(value) if value > 0.0 else math.ulp(1.0)
        for value in maximum_scales
    )
    quantization = (
        (servo_quantization_deg,) * len(SERVO_ORDER)
        + (wheel_quantization,) * len(WHEEL_ORDER)
    )
    calibration = ResidualActivityCalibration(
        phase_scale_full12=phase_scales,
        numeric_noise_floor_full12=numeric_noise,
        quantization_floor_full12=quantization,
    )
    return VersionedResidualActivityCalibration(
        calibration=calibration,
        phase_action_config=action_path,
        phase_action_config_sha256=sha256_file(action_path),
        environment_lock=environment_path,
        environment_lock_sha256=sha256_file(environment_path),
        servo_target_quantization_margin_rad=quantization_margin_rad,
        servo_command_quantization_floor_deg=servo_quantization_deg,
        wheel_velocity_limit_rad_s=wheel_limit,
        wheel_command_quantization_floor_rad_s=wheel_quantization,
        numeric_representation="JSON/Python IEEE-754 binary64; actuator targets binary32",
        numeric_noise_derivation=(
            "one binary64 ULP at each channel's maximum configured v2 phase scale"
        ),
        activity_threshold_formula=(
            "max(actuator_command_quantization_floor, "
            "3*numeric_logging_noise_floor, 0.01*configured_phase_residual_scale)"
        ),
    )


def _directory_inventory(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Cheap immutable-input guard: relative path, size, and mtime for every file."""

    return tuple(
        (
            item.relative_to(path).as_posix(),
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in sorted(path.rglob("*"), key=lambda value: value.as_posix())
        if item.is_file()
    )


def _require_phase_order(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> None:
    phases = tuple(str(row.get("phase", "")) for row in rows)
    if phases != PHASE_IDS:
        raise EvaluationArtifactError(
            f"{label} must contain ordered P01-P13 exactly; received {phases}"
        )


def evaluate_canonical_episode_dirs(
    episode_directories: Sequence[str | Path],
    *,
    seeds: Sequence[int],
    residual_calibration: (
        ResidualActivityCalibration | VersionedResidualActivityCalibration
    ),
    reward_stream_filename: str = DEFAULT_REWARD_STREAM_FILENAME,
    require_reward_stream: bool = True,
    evaluation_options: Mapping[str, Any] | None = None,
) -> tuple[LiveRunEvaluation, ...]:
    """Evaluate a sequence of canonical episode directories without mutation.

    Inputs and seeds are positional.  Seeds must be unique; callers comparing
    baseline and candidate sets should pass the same ordered seed sequence to
    each call.  The function verifies that every evaluator result contains all
    phases and that no file appeared, disappeared, or changed while it was
    being read.
    """

    directories = tuple(Path(value).resolve() for value in episode_directories)
    seed_values: list[int] = []
    for value in seeds:
        if isinstance(value, bool):
            raise EvaluationArtifactError("episode seeds must be non-negative integers")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationArtifactError(
                "episode seeds must be non-negative integers"
            ) from exc
        if seed < 0 or seed != value:
            raise EvaluationArtifactError("episode seeds must be non-negative integers")
        seed_values.append(seed)
    selected_seeds = tuple(seed_values)
    if not directories or len(directories) != len(selected_seeds):
        raise EvaluationArtifactError(
            "episode directories and seeds must be equal, non-empty sequences"
        )
    if len(set(directories)) != len(directories):
        raise EvaluationArtifactError("canonical episode directories must be unique")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise EvaluationArtifactError("canonical episode seeds must be unique")
    reward_name = Path(reward_stream_filename)
    if reward_name.name != str(reward_name) or not reward_name.name:
        raise EvaluationArtifactError("reward stream filename must be a plain filename")
    options = dict(evaluation_options or {})
    reserved = {"seed", "residual_calibration", "reward_stream_path"}
    overlap = sorted(reserved.intersection(options))
    if overlap:
        raise EvaluationArtifactError(
            f"evaluation_options may not override reserved arguments: {overlap}"
        )
    calibration = (
        residual_calibration.calibration
        if isinstance(residual_calibration, VersionedResidualActivityCalibration)
        else residual_calibration
    )

    evaluations: list[LiveRunEvaluation] = []
    for directory, seed in zip(directories, selected_seeds, strict=True):
        if not directory.is_dir():
            raise EvaluationArtifactError(
                f"canonical episode directory is missing: {directory}"
            )
        reward_path = directory / reward_name
        if require_reward_stream and not reward_path.is_file():
            raise EvaluationArtifactError(
                f"canonical episode reward stream is missing: {reward_path}"
            )
        inventory_before = _directory_inventory(directory)
        evaluated = evaluate_live_run(
            directory,
            seed=seed,
            residual_calibration=calibration,
            reward_stream_path=reward_path if reward_path.is_file() else None,
            **options,
        )
        inventory_after = _directory_inventory(directory)
        if inventory_after != inventory_before:
            raise EvaluationArtifactError(
                f"canonical episode changed during read-only evaluation: {directory}"
            )
        if int(evaluated.seed) != seed:
            raise EvaluationArtifactError("live evaluator returned a mismatched seed")
        _require_phase_order(evaluated.phase_rows, label=f"{directory.name} phase metrics")
        if not evaluated.residual_activity_evaluated:
            raise EvaluationArtifactError(
                f"{directory.name} lacks calibrated residual activity"
            )
        _require_phase_order(
            evaluated.residual_activity_rows,
            label=f"{directory.name} residual activity",
        )
        if require_reward_stream:
            if not evaluated.reward_contributions_available:
                raise EvaluationArtifactError(
                    f"{directory.name} lacks reward contribution evidence"
                )
            _require_phase_order(
                evaluated.reward_contribution_rows,
                label=f"{directory.name} reward contributions",
            )
        evaluations.append(evaluated)
    return tuple(evaluations)


@dataclass(frozen=True, slots=True)
class EvaluationArtifactPaths:
    output_directory: Path
    baseline_episode_metrics: Path
    baseline_phase_metrics: Path
    candidate_episode_metrics: Path
    candidate_phase_metrics: Path
    checkpoint_comparison: Path
    phase_metric_comparison: Path
    residual_activity_by_phase: Path
    reward_contribution_by_phase: Path
    termination_summary: Path
    promotion_decision: Path

    def as_dict(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "output_directory"
        }


def _matched_runs(
    baseline_runs: Sequence[LiveRunEvaluation],
    candidate_runs: Sequence[LiveRunEvaluation],
    *,
    minimum_paired_seeds: int,
) -> tuple[tuple[LiveRunEvaluation, ...], tuple[LiveRunEvaluation, ...]]:
    minimum = int(minimum_paired_seeds)
    if minimum <= 0:
        raise EvaluationArtifactError("minimum_paired_seeds must be positive")
    if len(baseline_runs) < minimum or len(candidate_runs) != len(baseline_runs):
        raise EvaluationArtifactError(
            f"paired export requires at least {minimum} equal baseline/candidate runs"
        )

    def index(runs: Sequence[LiveRunEvaluation], label: str) -> dict[int, LiveRunEvaluation]:
        indexed: dict[int, LiveRunEvaluation] = {}
        for run in runs:
            seed = int(run.seed)
            if seed in indexed:
                raise EvaluationArtifactError(f"{label} seeds must be unique")
            _require_phase_order(run.phase_rows, label=f"{label} seed {seed} phase metrics")
            indexed[seed] = run
        return indexed

    baseline = index(baseline_runs, "baseline")
    candidate = index(candidate_runs, "candidate")
    if set(baseline) != set(candidate):
        raise EvaluationArtifactError("baseline and candidate seeds are not matched")
    seeds = sorted(baseline)
    return (
        tuple(baseline[seed] for seed in seeds),
        tuple(candidate[seed] for seed in seeds),
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return value


def _normalized_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {str(name): _csv_value(value) for name, value in row.items()}
        for row in rows
    )


def _fieldnames(
    rows: Sequence[Mapping[str, Any]], *, preferred: Sequence[str]
) -> tuple[str, ...]:
    available = {str(name) for row in rows for name in row}
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available.difference(ordered)))
    if not ordered:
        raise EvaluationArtifactError("cannot publish a CSV with no columns")
    return tuple(ordered)


def _json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationArtifactError(f"JSON evidence is not serializable: {exc}") from exc


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=tuple(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    try:
        writer.writerows(rows)
    except (ValueError, csv.Error) as exc:
        raise EvaluationArtifactError(f"CSV evidence is invalid: {exc}") from exc
    return stream.getvalue().encode("utf-8")


@dataclass(frozen=True, slots=True)
class _CsvPublication:
    path: Path
    rows: tuple[Mapping[str, Any], ...]
    fieldnames: tuple[str, ...]

    @property
    def content(self) -> bytes:
        return _csv_bytes(self.rows, self.fieldnames)


@dataclass(frozen=True, slots=True)
class _JsonPublication:
    path: Path
    payload: Any

    @property
    def content(self) -> bytes:
        return _json_bytes(self.payload)


def _publish_idempotently(
    csv_publications: Sequence[_CsvPublication],
    json_publications: Sequence[_JsonPublication],
) -> None:
    publications = tuple(csv_publications) + tuple(json_publications)
    paths = [publication.path for publication in publications]
    if len(paths) != len(set(paths)):
        raise EvaluationArtifactError("evaluation artifact paths must be unique")
    # Preflight every existing target before creating any new artifact.  This
    # avoids a predictable partial bundle when one old file disagrees.
    for publication in publications:
        if publication.path.exists():
            if not publication.path.is_file() or publication.path.read_bytes() != publication.content:
                raise EvaluationArtifactError(
                    f"refusing to overwrite non-identical evaluation artifact: {publication.path}"
                )
    for publication in csv_publications:
        if publication.path.exists():
            continue
        try:
            atomic_write_csv(
                publication.path,
                publication.rows,
                fieldnames=publication.fieldnames,
            )
        except ArtifactError as exc:
            if not publication.path.is_file() or publication.path.read_bytes() != publication.content:
                raise EvaluationArtifactError(str(exc)) from exc
    for publication in json_publications:
        if publication.path.exists():
            continue
        try:
            atomic_write_json(publication.path, publication.payload)
        except ArtifactError as exc:
            if not publication.path.is_file() or publication.path.read_bytes() != publication.content:
                raise EvaluationArtifactError(str(exc)) from exc


def _episode_rows(
    runs: Sequence[LiveRunEvaluation], checkpoint: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "checkpoint": checkpoint,
            "episode_index": index,
            "run_directory": str(run.run_directory),
            **dict(run.episode_row),
        }
        for index, run in enumerate(runs)
    )


def _phase_rows(
    runs: Sequence[LiveRunEvaluation], checkpoint: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "checkpoint": checkpoint,
            "seed": int(run.seed),
            "trial_id": run.termination.trial_id,
            "run_directory": str(run.run_directory),
            **dict(row),
        }
        for run in runs
        for row in run.phase_rows
    )


@dataclass(frozen=True, slots=True)
class BaselineEvaluationArtifactPaths:
    """The candidate-independent baseline evidence bundle."""

    output_directory: Path
    episode_metrics: Path
    phase_metrics: Path
    manifest: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "episode_metrics": str(self.episode_metrics),
            "phase_metrics": str(self.phase_metrics),
            "manifest": str(self.manifest),
        }


def _require_complete_baseline_runs(
    runs: Sequence[LiveRunEvaluation], *, seeds: Sequence[int]
) -> tuple[LiveRunEvaluation, ...]:
    selected = tuple(runs)
    expected_seeds = tuple(int(value) for value in seeds)
    if len(selected) != len(BASELINE_VALIDATION_SEEDS):
        raise EvaluationArtifactError(
            "baseline export requires exactly five canonical validation episodes"
        )
    if expected_seeds != BASELINE_VALIDATION_SEEDS:
        raise EvaluationArtifactError(
            "baseline export requires validation seeds 2001-2005 in canonical order"
        )
    if tuple(int(run.seed) for run in selected) != expected_seeds:
        raise EvaluationArtifactError("baseline evaluator returned seeds out of order")
    for run in selected:
        _require_phase_order(
            run.phase_rows, label=f"baseline seed {run.seed} phase metrics"
        )
        incomplete = tuple(
            str(row["phase"])
            for row in run.phase_rows
            if row.get("phase_completion_observed") is not True
        )
        if incomplete or not run.termination.completed_p01_p13:
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} has incomplete phases: {incomplete or PHASE_IDS}"
            )
        if not run.termination.task_success:
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} is not an authoritative task success"
            )
        if (
            run.termination.body_collision
            or run.termination.wheel_only_climb
            or run.termination.safety_abort
            or run.termination.physics_explosion_or_fall
        ):
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} violates physical acceptance"
            )
        if run.termination.duration_s > 200.0:
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} exceeds the 200 second limit"
            )
        if run.termination.runtime_recording_access_count != 0:
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} accessed Recording data at runtime"
            )
        if not run.calibration.quality_passed:
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} has invalid reset calibration"
            )
        _require_phase_order(
            run.residual_activity_rows,
            label=f"baseline seed {run.seed} residual activity",
        )
        if any(row.get("nonzero") is not False for row in run.residual_activity_rows):
            raise EvaluationArtifactError(
                f"baseline seed {run.seed} is not a zero-residual FSM episode"
            )
    return selected


def _canonical_source_records(
    directories: Sequence[Path], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory, seed in zip(directories, seeds, strict=True):
        before = _directory_inventory(directory)
        files = []
        for name in CANONICAL_EPISODE_FILES:
            path = directory / name
            if not path.is_file():
                raise EvaluationArtifactError(
                    f"baseline canonical episode is missing {name}: {directory}"
                )
            files.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if _directory_inventory(directory) != before:
            raise EvaluationArtifactError(
                f"baseline canonical episode changed while hashing: {directory}"
            )
        records.append(
            {
                "seed": int(seed),
                "canonical_episode_dir": str(directory),
                "files": files,
            }
        )
    return records


def export_baseline_evaluation_artifacts(
    output_directory: str | Path = DEFAULT_METRICS_OUTPUT_DIRECTORY,
    *,
    episode_directories: Sequence[str | Path],
    seeds: Sequence[int],
    residual_calibration_evidence: VersionedResidualActivityCalibration | None = None,
    evaluation_options: Mapping[str, Any] | None = None,
    baseline_name: str = "pure_fsm",
) -> BaselineEvaluationArtifactPaths:
    """Evaluate and publish the five candidate-independent FSM baseline runs.

    Publication is deterministic and idempotent.  Existing byte-identical
    files are reused; a conflicting file causes the entire three-file bundle
    to fail before any new output is created.
    """

    label = str(baseline_name).strip()
    if not label:
        raise EvaluationArtifactError("baseline name cannot be empty")
    directories = tuple(Path(value).resolve() for value in episode_directories)
    selected_seeds = tuple(int(value) for value in seeds)
    if len(directories) != len(BASELINE_VALIDATION_SEEDS):
        raise EvaluationArtifactError(
            "baseline export requires exactly five canonical validation episodes"
        )
    if selected_seeds != BASELINE_VALIDATION_SEEDS:
        raise EvaluationArtifactError(
            "baseline export requires validation seeds 2001-2005 in canonical order"
        )
    calibration = residual_calibration_evidence or (
        build_versioned_residual_activity_calibration()
    )
    runs = _require_complete_baseline_runs(
        evaluate_canonical_episode_dirs(
            directories,
            seeds=selected_seeds,
            residual_calibration=calibration,
            require_reward_stream=True,
            evaluation_options=evaluation_options,
        ),
        seeds=selected_seeds,
    )
    source_records = _canonical_source_records(directories, selected_seeds)

    output = Path(output_directory).resolve()
    paths = BaselineEvaluationArtifactPaths(
        output_directory=output,
        episode_metrics=output / BASELINE_EPISODE_FILENAME,
        phase_metrics=output / BASELINE_PHASE_FILENAME,
        manifest=output / BASELINE_EVALUATION_MANIFEST_FILENAME,
    )
    episode_rows = _normalized_rows(_episode_rows(runs, label))
    phase_rows = _normalized_rows(_phase_rows(runs, label))
    episode_publication = _CsvPublication(
        paths.episode_metrics,
        episode_rows,
        _fieldnames(
            episode_rows,
            preferred=(
                "checkpoint",
                "episode_index",
                "seed",
                "trial_id",
                "run_directory",
                "task_result",
                "task_success",
            ),
        ),
    )
    phase_publication = _CsvPublication(
        paths.phase_metrics,
        phase_rows,
        _fieldnames(
            phase_rows,
            preferred=("checkpoint", "seed", "trial_id", "run_directory", "phase"),
        ),
    )
    manifest = {
        "schema": "wlr50_clean.fsm_baseline_evaluation.v1",
        "baseline": label,
        "candidate_required": False,
        "episode_count": len(runs),
        "phase_count_per_episode": len(PHASE_IDS),
        "phase_metric_row_count": len(phase_rows),
        "validation_seeds": list(selected_seeds),
        "all_p01_p13_complete": True,
        "all_authoritative_success": True,
        "all_zero_residual": True,
        "source_episodes": source_records,
        "residual_activity_calibration": calibration.as_dict(),
        "artifacts": {
            "episode_metrics": {
                "path": str(paths.episode_metrics),
                "bytes": len(episode_publication.content),
                "sha256": hashlib.sha256(episode_publication.content).hexdigest(),
            },
            "phase_metrics": {
                "path": str(paths.phase_metrics),
                "bytes": len(phase_publication.content),
                "sha256": hashlib.sha256(phase_publication.content).hexdigest(),
            },
            "manifest": str(paths.manifest),
        },
    }
    _publish_idempotently(
        (episode_publication, phase_publication),
        (_JsonPublication(paths.manifest, manifest),),
    )
    return paths


def _candidate_residual_rows(
    runs: Sequence[LiveRunEvaluation], checkpoint: str
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.residual_activity_evaluated:
            raise EvaluationArtifactError(
                f"candidate seed {run.seed} lacks calibrated residual activity"
            )
        _require_phase_order(
            run.residual_activity_rows,
            label=f"candidate seed {run.seed} residual activity",
        )
        rows.extend(
            {
                "checkpoint": checkpoint,
                "seed": int(run.seed),
                "trial_id": run.termination.trial_id,
                "run_directory": str(run.run_directory),
                **dict(row),
            }
            for row in run.residual_activity_rows
        )
    return tuple(rows)


def _candidate_reward_rows(
    runs: Sequence[LiveRunEvaluation], checkpoint: str
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.reward_contributions_available:
            raise EvaluationArtifactError(
                f"candidate seed {run.seed} lacks reward contribution evidence"
            )
        _require_phase_order(
            run.reward_contribution_rows,
            label=f"candidate seed {run.seed} reward contributions",
        )
        rows.extend(
            {
                "checkpoint": checkpoint,
                "seed": int(run.seed),
                "trial_id": run.termination.trial_id,
                "run_directory": str(run.run_directory),
                **dict(row),
            }
            for row in run.reward_contribution_rows
        )
    return tuple(rows)


def _termination_rows(
    baseline_runs: Sequence[LiveRunEvaluation],
    candidate_runs: Sequence[LiveRunEvaluation],
    baseline_label: str,
    candidate_label: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for checkpoint, runs in (
        (baseline_label, baseline_runs),
        (candidate_label, candidate_runs),
    ):
        rows.extend(
            {
                "checkpoint": checkpoint,
                "seed": int(run.seed),
                "run_directory": str(run.run_directory),
                **asdict(run.termination),
            }
            for run in runs
        )
    return tuple(rows)


def _paired_phase_rows(comparison: Any) -> tuple[dict[str, Any], ...]:
    baseline = {str(row["phase"]): row for row in comparison.baseline_phase_rows}
    candidate = {str(row["phase"]): row for row in comparison.candidate_phase_rows}
    differences = {
        str(row["phase"]): row for row in comparison.phase_comparison_rows
    }
    rows: list[dict[str, Any]] = []
    for phase in PHASE_IDS:
        base = baseline[phase]
        proposed = candidate[phase]
        common = sorted(set(base).intersection(proposed).difference({"phase"}))
        row: dict[str, Any] = {"phase": phase}
        for name in common:
            row[f"fsm_baseline_{name}"] = base[name]
            row[f"candidate_{name}"] = proposed[name]
        row.update(
            {
                name: value
                for name, value in differences[phase].items()
                if name != "phase"
            }
        )
        rows.append(row)
    return tuple(rows)


def _mean_numeric(runs: Sequence[LiveRunEvaluation], name: str) -> float | None:
    values = [run.episode_row.get(name) for run in runs]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        return None
    return float(np.mean([float(value) for value in values]))


def _checkpoint_row(
    *,
    role: str,
    checkpoint: str,
    runs: Sequence[LiveRunEvaluation],
    checkpoint_path: Path | None,
    comparison: Any,
) -> dict[str, Any]:
    candidate = role == "candidate"
    return {
        "role": role,
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "episode_count": len(runs),
        "paired_seeds": [int(run.seed) for run in runs],
        "task_success_count": sum(run.termination.task_success for run in runs),
        "task_success_rate": sum(run.termination.task_success for run in runs) / len(runs),
        "body_collision_count": sum(run.termination.body_collision for run in runs),
        "wheel_only_climb_count": sum(run.termination.wheel_only_climb for run in runs),
        "physics_explosion_or_fall_count": sum(
            run.termination.physics_explosion_or_fall for run in runs
        ),
        "safety_abort_count": sum(run.termination.safety_abort for run in runs),
        "p01_p13_completed_count": sum(
            run.termination.completed_p01_p13 for run in runs
        ),
        "runtime_recording_access_count": sum(
            run.termination.runtime_recording_access_count for run in runs
        ),
        "mean_duration_s": _mean_numeric(runs, "duration_s"),
        "mean_overall_pitch_rate_rms_rad_s": _mean_numeric(
            runs, "overall_pitch_rate_rms_rad_s"
        ),
        "mean_overall_roll_rate_rms_rad_s": _mean_numeric(
            runs, "overall_roll_rate_rms_rad_s"
        ),
        "mean_placement_contact_impulse_n_s": _mean_numeric(
            runs, "placement_contact_impulse_n_s"
        ),
        "mean_home_recovery_action_jerk_rms": _mean_numeric(
            runs, "home_recovery_action_jerk_rms"
        ),
        "mean_total_reward": _mean_numeric(runs, "total_reward"),
        "global_stability_improvement_fraction": (
            comparison.promotion.global_stability_improvement_fraction
            if candidate
            else 0.0
        ),
        "overall_pitch_rate_improvement_fraction": (
            comparison.overall_pitch_rate_improvement_fraction if candidate else 0.0
        ),
        "placement_impulse_improvement_fraction": (
            comparison.placement_impulse_improvement_fraction if candidate else 0.0
        ),
        "home_jerk_improvement_fraction": (
            comparison.home_jerk_improvement_fraction if candidate else 0.0
        ),
        "improved_priority_phase_count": (
            comparison.promotion.improved_priority_phase_count if candidate else 0
        ),
        "promotion_passed": comparison.promotion.promoted if candidate else None,
        "first_failed_gate": (
            comparison.promotion.first_failed_gate if candidate else None
        ),
    }


def export_paired_evaluation_artifacts(
    output_directory: str | Path,
    *,
    baseline_runs: Sequence[LiveRunEvaluation],
    candidate_runs: Sequence[LiveRunEvaluation],
    frozen_hashes_unchanged: bool,
    candidate_checkpoint_name: str,
    candidate_checkpoint_path: str | Path | None = None,
    baseline_name: str = "pure_fsm",
    minimum_paired_seeds: int = 5,
    residual_calibration_evidence: (
        VersionedResidualActivityCalibration | None
    ) = None,
) -> EvaluationArtifactPaths:
    """Publish a complete paired evaluation bundle atomically per file.

    Existing byte-identical files make the operation idempotent.  Any existing
    non-identical target aborts the entire preflight before a new target is
    created.  The output is evidence only; checkpoint promotion/copying remains
    a separate operation guarded by ``promotion_decision.json``.
    """

    baseline_label = str(baseline_name).strip()
    candidate_label = str(candidate_checkpoint_name).strip()
    if not baseline_label or not candidate_label or baseline_label == candidate_label:
        raise EvaluationArtifactError(
            "baseline and candidate checkpoint names must be non-empty and distinct"
        )
    baseline, candidate = _matched_runs(
        baseline_runs,
        candidate_runs,
        minimum_paired_seeds=minimum_paired_seeds,
    )
    checkpoint_path = (
        None if candidate_checkpoint_path is None else Path(candidate_checkpoint_path).resolve()
    )
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise EvaluationArtifactError(
            f"candidate checkpoint is missing: {checkpoint_path}"
        )

    comparison = paired_baseline_candidate_promotion(
        baseline,
        candidate,
        frozen_hashes_unchanged=bool(frozen_hashes_unchanged),
        minimum_paired_seeds=int(minimum_paired_seeds),
    )
    directory = Path(output_directory).resolve()
    paths = EvaluationArtifactPaths(
        output_directory=directory,
        baseline_episode_metrics=directory / BASELINE_EPISODE_FILENAME,
        baseline_phase_metrics=directory / BASELINE_PHASE_FILENAME,
        candidate_episode_metrics=directory / CANDIDATE_EPISODE_FILENAME,
        candidate_phase_metrics=directory / CANDIDATE_PHASE_FILENAME,
        checkpoint_comparison=directory / CHECKPOINT_COMPARISON_FILENAME,
        phase_metric_comparison=directory / PHASE_COMPARISON_FILENAME,
        residual_activity_by_phase=directory / RESIDUAL_ACTIVITY_FILENAME,
        reward_contribution_by_phase=directory / REWARD_CONTRIBUTION_FILENAME,
        termination_summary=directory / TERMINATION_SUMMARY_FILENAME,
        promotion_decision=directory / PROMOTION_DECISION_FILENAME,
    )

    baseline_episode = _normalized_rows(_episode_rows(baseline, baseline_label))
    baseline_phase = _normalized_rows(_phase_rows(baseline, baseline_label))
    candidate_episode = _normalized_rows(_episode_rows(candidate, candidate_label))
    candidate_phase = _normalized_rows(_phase_rows(candidate, candidate_label))
    residual = _normalized_rows(_candidate_residual_rows(candidate, candidate_label))
    reward = _normalized_rows(_candidate_reward_rows(candidate, candidate_label))
    termination = _normalized_rows(
        _termination_rows(baseline, candidate, baseline_label, candidate_label)
    )
    phase_comparison = _normalized_rows(_paired_phase_rows(comparison))
    checkpoint_comparison = _normalized_rows(
        (
            _checkpoint_row(
                role="baseline",
                checkpoint=baseline_label,
                runs=baseline,
                checkpoint_path=None,
                comparison=comparison,
            ),
            _checkpoint_row(
                role="candidate",
                checkpoint=candidate_label,
                runs=candidate,
                checkpoint_path=checkpoint_path,
                comparison=comparison,
            ),
        )
    )

    decision = comparison.promotion
    decision_payload: dict[str, Any] = {
        "schema": EVALUATION_ARTIFACT_SCHEMA,
        "baseline_checkpoint": baseline_label,
        "candidate_checkpoint": candidate_label,
        "candidate_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "candidate_checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path else None
        ),
        "paired_seeds": [int(run.seed) for run in baseline],
        "paired_episode_count": len(baseline),
        "minimum_paired_seeds": int(minimum_paired_seeds),
        "frozen_hashes_unchanged": bool(frozen_hashes_unchanged),
        "promotion": asdict(decision),
        "first_failed_gate": decision.first_failed_gate,
        "checks_in_evaluation_order": [
            {"gate": gate, "passed": bool(passed)}
            for gate, passed in decision.checks.items()
        ],
        "overall_pitch_rate_improvement_fraction": (
            comparison.overall_pitch_rate_improvement_fraction
        ),
        "placement_impulse_improvement_fraction": (
            comparison.placement_impulse_improvement_fraction
        ),
        "home_jerk_improvement_fraction": comparison.home_jerk_improvement_fraction,
        "artifacts": paths.as_dict(),
        "residual_activity_calibration": (
            residual_calibration_evidence.as_dict()
            if residual_calibration_evidence is not None
            else None
        ),
    }

    csv_specs = (
        _CsvPublication(
            paths.baseline_episode_metrics,
            baseline_episode,
            _fieldnames(
                baseline_episode,
                preferred=(
                    "checkpoint",
                    "episode_index",
                    "seed",
                    "trial_id",
                    "run_directory",
                    "task_result",
                    "task_success",
                ),
            ),
        ),
        _CsvPublication(
            paths.baseline_phase_metrics,
            baseline_phase,
            _fieldnames(
                baseline_phase,
                preferred=("checkpoint", "seed", "trial_id", "run_directory", "phase"),
            ),
        ),
        _CsvPublication(
            paths.candidate_episode_metrics,
            candidate_episode,
            _fieldnames(
                candidate_episode,
                preferred=(
                    "checkpoint",
                    "episode_index",
                    "seed",
                    "trial_id",
                    "run_directory",
                    "task_result",
                    "task_success",
                ),
            ),
        ),
        _CsvPublication(
            paths.candidate_phase_metrics,
            candidate_phase,
            _fieldnames(
                candidate_phase,
                preferred=("checkpoint", "seed", "trial_id", "run_directory", "phase"),
            ),
        ),
        _CsvPublication(
            paths.checkpoint_comparison,
            checkpoint_comparison,
            _fieldnames(
                checkpoint_comparison,
                preferred=("role", "checkpoint", "checkpoint_path", "checkpoint_sha256"),
            ),
        ),
        _CsvPublication(
            paths.phase_metric_comparison,
            phase_comparison,
            _fieldnames(phase_comparison, preferred=("phase",)),
        ),
        _CsvPublication(
            paths.residual_activity_by_phase,
            residual,
            _fieldnames(
                residual,
                preferred=("checkpoint", "seed", "trial_id", "run_directory", "phase"),
            ),
        ),
        _CsvPublication(
            paths.reward_contribution_by_phase,
            reward,
            _fieldnames(
                reward,
                preferred=("checkpoint", "seed", "trial_id", "run_directory", "phase"),
            ),
        ),
        _CsvPublication(
            paths.termination_summary,
            termination,
            _fieldnames(
                termination,
                preferred=(
                    "checkpoint",
                    "seed",
                    "trial_id",
                    "run_directory",
                    "result",
                    "task_success",
                ),
            ),
        ),
    )
    _publish_idempotently(csv_specs, (_JsonPublication(paths.promotion_decision, decision_payload),))
    return paths


__all__ = [
    "BASELINE_EVALUATION_MANIFEST_FILENAME",
    "BASELINE_EPISODE_FILENAME",
    "BASELINE_PHASE_FILENAME",
    "BASELINE_VALIDATION_SEEDS",
    "CANONICAL_EPISODE_FILES",
    "CANDIDATE_EPISODE_FILENAME",
    "CANDIDATE_PHASE_FILENAME",
    "CHECKPOINT_COMPARISON_FILENAME",
    "DEFAULT_ENVIRONMENT_LOCK",
    "DEFAULT_METRICS_OUTPUT_DIRECTORY",
    "DEFAULT_REWARD_STREAM_FILENAME",
    "EVALUATION_ARTIFACT_SCHEMA",
    "EvaluationArtifactError",
    "EvaluationArtifactPaths",
    "BaselineEvaluationArtifactPaths",
    "FreshProcessEpisodeBatch",
    "PHASE_COMPARISON_FILENAME",
    "PROMOTION_DECISION_FILENAME",
    "RESIDUAL_ACTIVITY_FILENAME",
    "RESIDUAL_CALIBRATION_SCHEMA",
    "REWARD_CONTRIBUTION_FILENAME",
    "TERMINATION_SUMMARY_FILENAME",
    "VersionedResidualActivityCalibration",
    "build_versioned_residual_activity_calibration",
    "collect_fresh_process_episode_workers",
    "evaluate_canonical_episode_dirs",
    "export_baseline_evaluation_artifacts",
    "export_paired_evaluation_artifacts",
]
