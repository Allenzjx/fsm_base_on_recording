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
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    WHEEL_ORDER,
)

from .artifacts import (
    RUN_MANIFEST_SCHEMA,
    ArtifactError,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from .checkpoint_promotion import (
    CHECKPOINT_IMPROVED_PROMOTION_SCHEMA,
    CHECKPOINT_MANIFEST_SCHEMA,
    CheckpointPromotionError,
    FROZEN_HASH_FIELDS,
    LOCKED_TEST_SEEDS,
    PROMOTION_MANIFEST_NAME,
    VALIDATION_PROMOTION_MANIFEST_NAME,
    _load_embedded_checkpoint_infos,
    _validate_best_validation_source,
    _validate_checkpoint_snapshot_contract,
    _validate_locked_test_aggregate,
    validate_checkpoint_artifact_provenance,
)
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
FINAL_LIFECYCLE_BUNDLE_KIND = "final_lifecycle_five_role"
FINAL_LIFECYCLE_ROLES = (
    "pure_fsm",
    "checkpoint_initial",
    "checkpoint_smoke",
    "checkpoint_best",
    "checkpoint_improved",
)
_FINAL_LIFECYCLE_REQUIRED_PASS_ROLES = frozenset(
    {"pure_fsm", "checkpoint_best", "checkpoint_improved"}
)
_FINAL_LIFECYCLE_CHECKPOINT_FILENAMES = {
    "checkpoint_initial": "checkpoint_initial_zero_residual.pt",
    "checkpoint_smoke": "checkpoint_smoke.pt",
    "checkpoint_best": "checkpoint_best_validation.pt",
    "checkpoint_improved": "checkpoint_improved.pt",
}
_FINAL_LIFECYCLE_RUNTIME_HASH_PATHS = {
    "controller_hash": PROJECT_ROOT / "configs" / "fsm_states.yaml",
    "environment_hash": PROJECT_ROOT / "configs" / "environment_lock.json",
    "observation_schema_hash": PROJECT_ROOT / "configs" / "ppo_observation_schema_v2.json",
    "action_schema_hash": PROJECT_ROOT / "configs" / "ppo_phase_action_masks_v2.yaml",
    "reward_config_hash": PROJECT_ROOT / "configs" / "ppo_reward_v2.yaml",
}
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


@dataclass(frozen=True, slots=True)
class FinalLifecycleAggregateEvidence:
    """Revalidated provenance for one fixed final-lifecycle evaluation role."""

    role: str
    aggregate_path: Path
    aggregate_sha256: str
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    checkpoint_manifest_path: Path | None
    checkpoint_manifest_sha256: str | None
    training_seed: int | None
    source_git_commit: str | None
    committed_runtime_content_sha256: str | None
    creation_runtime_identity_path: Path | None
    creation_runtime_identity_sha256: str | None
    seeds: tuple[int, ...]
    worker_run_dirs: tuple[Path, ...]
    canonical_episode_dirs: tuple[Path, ...]
    source_groups: tuple[Mapping[str, Any], ...]
    supporting_files: tuple[Mapping[str, Any], ...]
    committed_runtime_identity: Mapping[str, Any]


def _is_reparse_point(path: Path) -> bool:
    """Return true for a symlink, junction, or other Windows reparse point."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or (callable(is_junction) and is_junction())
        or (getattr(metadata, "st_file_attributes", 0) & 0x400)
    )


def _require_no_reparse_components(path: Path, *, label: str) -> None:
    """Reject path routing that can change targets without changing path text."""

    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        if _is_reparse_point(component):
            raise EvaluationArtifactError(
                f"{label} contains a symbolic link or reparse point: {component}"
            )


def _resolved_path(path: str | Path, *, label: str) -> Path:
    raw = Path(path)
    _require_no_reparse_components(raw, label=label)
    return raw.resolve()


def _capture_file_record(
    path: str | Path, *, label: str, allow_empty: bool = False
) -> tuple[dict[str, Any], bytes]:
    """Read a source once and derive size/SHA from those exact bytes."""

    resolved = _resolved_path(path, label=label)
    try:
        with resolved.open("rb") as stream:
            content = stream.read()
    except OSError as exc:
        raise EvaluationArtifactError(f"cannot read {label}: {resolved}: {exc}") from exc
    if not allow_empty and not content:
        raise EvaluationArtifactError(f"{label} is empty: {resolved}")
    return (
        {
            "path": str(resolved),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        content,
    )


def _capture_json_object(
    path: str | Path, *, label: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    record, content = _capture_file_record(path, label=label)
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationArtifactError(f"{label} must contain a JSON object")
    return record, payload


def _record_unchanged(record: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "path": record.get("path"),
        "bytes": record.get("bytes"),
        "sha256": record.get("sha256"),
    }
    current, _ = _capture_file_record(
        str(expected["path"] or ""),
        label=label,
        allow_empty=expected["bytes"] == 0,
    )
    if current != expected:
        raise EvaluationArtifactError(f"{label} changed after provenance capture")


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_within(left, right) or _path_within(right, left)


_RUNTIME_IDENTITY_GIT_PATHS = (
    "src/wlr50_clean",
    "src/wlr50_clean/ppo",
    "src/wlr50_clean/fsm",
    "src/wlr50_clean/sensing",
    "src/wlr50_clean/infrastructure",
    "scripts",
    "configs",
    "reference/ppo_phase_snapshots",
    "artifacts/ppo_phase_v1_start",
    "pyproject.toml",
)


def _validate_worker_runtime_identity(
    run_dir: Path, run_manifest: Mapping[str, Any], *, label: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    before_record, before = _capture_json_object(
        run_dir / "committed_runtime_identity.before.json",
        label=f"{label} committed runtime identity before",
    )
    after_record, after = _capture_json_object(
        run_dir / "committed_runtime_identity.after.json",
        label=f"{label} committed runtime identity after",
    )
    files = before.get("files")
    if (
        before.get("schema") != "wlr50_clean.committed_runtime_identity.v1"
        or dict(before) != dict(after)
        or not isinstance(before.get("git_commit"), str)
        or len(str(before["git_commit"])) != 40
        or any(character not in "0123456789abcdef" for character in before["git_commit"])
        or not isinstance(files, list)
        or not files
        or before.get("file_count") != len(files)
        or not isinstance(before.get("aggregate_sha256"), str)
        or len(str(before["aggregate_sha256"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(before["aggregate_sha256"])
        )
        or not isinstance(before.get("content_sha256"), str)
        or len(str(before["content_sha256"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(before["content_sha256"])
        )
    ):
        raise EvaluationArtifactError(
            f"{label} committed runtime identities are incomplete or differ"
        )
    paths = tuple(str(row.get("path", "")) for row in files if isinstance(row, Mapping))
    if len(paths) != len(files) or paths != tuple(sorted(set(paths))):
        raise EvaluationArtifactError(
            f"{label} committed runtime identity file inventory is invalid"
        )
    ordered_row_fields = (
        "path",
        "bytes",
        "sha256",
        "creation_time_utc_ticks",
        "last_write_time_utc_ticks",
    )
    required_row_fields = set(ordered_row_fields)
    for index, row in enumerate(files):
        if (
            not isinstance(row, Mapping)
            or set(row) != required_row_fields
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])
            or any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), int)
                or row[field] <= 0
                for field in (
                    "creation_time_utc_ticks",
                    "last_write_time_utc_ticks",
                )
            )
        ):
            raise EvaluationArtifactError(
                f"{label} committed runtime identity file row {index} is invalid"
            )
    encoded_rows = json.dumps(
        [
            {field: row[field] for field in ordered_row_fields}
            for row in files
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded_rows).hexdigest() != before["aggregate_sha256"]:
        raise EvaluationArtifactError(
            f"{label} committed runtime identity aggregate SHA-256 is invalid"
        )
    content_rows = [
        {
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in files
    ]
    encoded_content = json.dumps(
        content_rows, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(encoded_content).hexdigest() != before["content_sha256"]:
        raise EvaluationArtifactError(
            f"{label} committed runtime identity content SHA-256 is invalid"
        )
    artifacts = run_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationArtifactError(
            f"{label} run manifest omits committed runtime identity artifacts"
        )
    for filename, captured in (
        ("committed_runtime_identity.before.json", before_record),
        ("committed_runtime_identity.after.json", after_record),
    ):
        declared = artifacts.get(filename)
        if (
            not isinstance(declared, Mapping)
            or declared.get("bytes") != captured["bytes"]
            or declared.get("sha256") != captured["sha256"]
        ):
            raise EvaluationArtifactError(
                f"{label} run manifest does not bind {filename}"
            )
    return before_record, after_record, before


def _validate_current_committed_runtime_identity(identity: Mapping[str, Any]) -> None:
    """Require the recorded implementation inventory to still equal committed HEAD."""

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(PROJECT_ROOT), *arguments),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvaluationArtifactError(
                "cannot verify committed runtime identity against Git HEAD"
            ) from exc
        return completed.stdout

    head = git("rev-parse", "HEAD").strip().lower()
    if identity.get("git_commit") != head:
        raise EvaluationArtifactError(
            "committed runtime identity does not match the current Git HEAD"
        )
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_RUNTIME_IDENTITY_GIT_PATHS,
    )
    if status.strip():
        raise EvaluationArtifactError(
            "current PPO runtime implementation/config differs from Git HEAD"
        )
    tracked = tuple(
        sorted(
            line.replace("\\", "/")
            for line in git("ls-files", "--", *_RUNTIME_IDENTITY_GIT_PATHS).splitlines()
            if line
        )
    )
    rows = identity.get("files")
    if (
        not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or tuple(str(row.get("path", "")) for row in rows) != tracked
    ):
        raise EvaluationArtifactError(
            "committed runtime identity does not cover the exact tracked runtime inventory"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EvaluationArtifactError("committed runtime identity row is malformed")
        raw = Path(str(row.get("path", "")))
        if raw.is_absolute() or ".." in raw.parts:
            raise EvaluationArtifactError("committed runtime identity path is unsafe")
        record, _ = _capture_file_record(
            PROJECT_ROOT.joinpath(*raw.parts),
            label=f"committed runtime file {index}",
        )
        if record["bytes"] != row.get("bytes") or record["sha256"] != row.get("sha256"):
            raise EvaluationArtifactError(
                f"committed runtime file differs from recorded HEAD identity: {raw}"
            )


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
    run_dirs = tuple(
        _resolved_path(value, label="fresh-process worker directory")
        for value in worker_run_directories
    )
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

    checkpoint = (
        None
        if checkpoint_path is None
        else _resolved_path(checkpoint_path, label="candidate checkpoint")
    )
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
        if (
            isinstance(episode.get("seed"), bool)
            or not isinstance(episode.get("seed"), int)
            or episode.get("seed") != expected_seed
        ):
            raise EvaluationArtifactError(
                f"worker seed differs from expected seed {expected_seed}: {run_dir}"
            )
        raw_episode_dir = episode.get("canonical_episode_dir")
        if raw_episode_dir:
            episode_dir = _resolved_path(
                str(raw_episode_dir), label="canonical episode directory"
            )
        else:
            trial_path = _resolved_path(
                str(episode.get("trial_manifest_path", "")),
                label="canonical trial manifest",
            )
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
        if (
            isinstance(trial.get("seed"), bool)
            or not isinstance(trial.get("seed"), int)
            or trial.get("seed") != expected_seed
        ):
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


def _require_canonical_phase_prefix(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[str, ...]:
    """Require the non-empty P01-starting prefix produced by an early failure."""

    phases = tuple(str(row.get("phase", "")) for row in rows)
    if not phases:
        raise EvaluationArtifactError(f"{label} must contain at least one phase")
    if len(set(phases)) != len(phases) or any(phase not in PHASE_IDS for phase in phases):
        raise EvaluationArtifactError(
            f"{label} contains duplicate or non-canonical phases: {phases}"
        )
    expected = PHASE_IDS[: len(phases)]
    if phases != expected:
        raise EvaluationArtifactError(
            f"{label} must be a contiguous canonical phase prefix from P01; "
            f"received {phases}"
        )
    return phases


def evaluate_canonical_episode_dirs(
    episode_directories: Sequence[str | Path],
    *,
    seeds: Sequence[int],
    residual_calibration: (
        ResidualActivityCalibration | VersionedResidualActivityCalibration
    ),
    reward_stream_filename: str = DEFAULT_REWARD_STREAM_FILENAME,
    require_reward_stream: bool = True,
    require_complete_phase_sequence: bool = True,
    evaluation_options: Mapping[str, Any] | None = None,
) -> tuple[LiveRunEvaluation, ...]:
    """Evaluate a sequence of canonical episode directories without mutation.

    Inputs and seeds are positional.  Seeds must be unique; callers comparing
    baseline and candidate sets should pass the same ordered seed sequence to
    each call.  By default every evaluator result must contain all phases.
    ``require_complete_phase_sequence=False`` admits truthful early physical
    failures, while still requiring each metric stream to be the non-empty,
    contiguous canonical prefix P01..Pn.  No file may appear, disappear, or
    change while it is being read.
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
        if require_complete_phase_sequence:
            _require_phase_order(
                evaluated.phase_rows, label=f"{directory.name} phase metrics"
            )
            phase_sequence = PHASE_IDS
        else:
            phase_sequence = _require_canonical_phase_prefix(
                evaluated.phase_rows, label=f"{directory.name} phase metrics"
            )
        if not evaluated.residual_activity_evaluated:
            raise EvaluationArtifactError(
                f"{directory.name} lacks calibrated residual activity"
            )
        if require_complete_phase_sequence:
            _require_phase_order(
                evaluated.residual_activity_rows,
                label=f"{directory.name} residual activity",
            )
            residual_sequence = PHASE_IDS
        else:
            residual_sequence = _require_canonical_phase_prefix(
                evaluated.residual_activity_rows,
                label=f"{directory.name} residual activity",
            )
        if require_reward_stream:
            if not evaluated.reward_contributions_available:
                raise EvaluationArtifactError(
                    f"{directory.name} lacks reward contribution evidence"
                )
            if require_complete_phase_sequence:
                _require_phase_order(
                    evaluated.reward_contribution_rows,
                    label=f"{directory.name} reward contributions",
                )
                reward_sequence = PHASE_IDS
            else:
                reward_sequence = _require_canonical_phase_prefix(
                    evaluated.reward_contribution_rows,
                    label=f"{directory.name} reward contributions",
                )
            if (
                residual_sequence != phase_sequence
                or reward_sequence != phase_sequence
            ):
                raise EvaluationArtifactError(
                    f"{directory.name} metric streams do not share one phase prefix"
                )
        elif residual_sequence != phase_sequence:
            raise EvaluationArtifactError(
                f"{directory.name} metric streams do not share one phase prefix"
            )
        evaluations.append(evaluated)
    return tuple(evaluations)


def _require_checkpoint_abi(
    checkpoint: Path,
    checkpoint_hash: str,
    checkpoint_bytes: bytes,
    manifest: Mapping[str, Any],
    *,
    role: str,
) -> int:
    """Validate the common checkpoint sidecar and embedded policy ABI."""

    if manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise EvaluationArtifactError(f"{role} checkpoint manifest has the wrong schema")
    if (
        _resolved_path(
            str(manifest.get("checkpoint_path", "")),
            label=f"{role} manifest checkpoint path",
        )
        != checkpoint
        or manifest.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise EvaluationArtifactError(
            f"{role} checkpoint manifest does not bind its checkpoint bytes"
        )
    expected_integers = {
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
    }
    try:
        training_seed = int(manifest.get("training_seed"))
        dimensions_valid = all(
            not isinstance(manifest.get(name), bool)
            and int(manifest.get(name)) == expected
            and manifest.get(name) == expected
            for name, expected in expected_integers.items()
        )
        timing_valid = math.isclose(
            float(manifest.get("physics_hz")), 120.0, rel_tol=0.0, abs_tol=1.0e-12
        ) and math.isclose(
            float(manifest.get("decision_hz")), 15.0, rel_tol=0.0, abs_tol=1.0e-12
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationArtifactError(f"{role} checkpoint ABI is incomplete") from exc
    if (
        not dimensions_valid
        or not timing_valid
        or isinstance(manifest.get("training_seed"), bool)
        or manifest.get("training_seed") != training_seed
        or training_seed not in range(1001, 1033)
        or isinstance(manifest.get("global_policy_decisions"), bool)
        or not isinstance(manifest.get("global_policy_decisions"), int)
        or manifest["global_policy_decisions"] < 0
    ):
        raise EvaluationArtifactError(f"{role} checkpoint ABI is not 125/125/12 at 120/15 Hz")
    for field in FROZEN_HASH_FIELDS:
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise EvaluationArtifactError(f"{role} checkpoint omits valid {field}")
        current_path = _FINAL_LIFECYCLE_RUNTIME_HASH_PATHS[field]
        if not current_path.is_file() or sha256_file(current_path) != value:
            raise EvaluationArtifactError(
                f"{role} checkpoint {field} differs from the current runtime config"
            )
    source_git_commit = manifest.get("source_git_commit")
    if (
        not isinstance(source_git_commit, str)
        or len(source_git_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_git_commit
        )
    ):
        raise EvaluationArtifactError(
            f"{role} checkpoint omits a valid source_git_commit"
        )
    for field in (
        "committed_runtime_content_sha256",
        "creation_runtime_identity_sha256",
    ):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise EvaluationArtifactError(f"{role} checkpoint omits valid {field}")
    creation_path = _resolved_path(
        str(manifest.get("creation_runtime_identity_path", "")),
        label=f"{role} creation runtime identity",
    )
    if creation_path.name != "committed_runtime_identity.before.json":
        raise EvaluationArtifactError(
            f"{role} checkpoint creation runtime identity path is invalid"
        )
    try:
        embedded_infos = _load_embedded_checkpoint_infos(checkpoint_bytes)
        _validate_checkpoint_snapshot_contract(manifest, embedded_infos)
    except CheckpointPromotionError as exc:
        raise EvaluationArtifactError(
            f"{role} checkpoint snapshot/embedded ABI is invalid: {exc}"
        ) from exc
    return training_seed


def _require_initial_zero_actor_output(
    checkpoint_bytes: bytes, manifest: Mapping[str, Any]
) -> None:
    """Prove the canonical initial actor's only Full12 output layer is exact zero."""

    try:
        import torch  # type: ignore

        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
        actor = payload.get("actor_state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(actor, Mapping):
            raise ValueError("actor_state_dict is missing")
        residual_dimension = int(manifest["residual_dimension"])
        candidates = []
        for name, weight in actor.items():
            if (
                isinstance(name, str)
                and name.endswith(".weight")
                and torch.is_tensor(weight)
                and weight.ndim == 2
                and int(weight.shape[0]) == residual_dimension
            ):
                bias = actor.get(name[: -len("weight")] + "bias")
                if (
                    torch.is_tensor(bias)
                    and bias.ndim == 1
                    and int(bias.shape[0]) == residual_dimension
                ):
                    candidates.append((weight, bias))
        if len(candidates) != 1:
            raise ValueError("actor Full12 output layer is ambiguous")
        weight, bias = candidates[0]
        if (
            not bool(torch.isfinite(weight).all().item())
            or not bool(torch.isfinite(bias).all().item())
            or int(torch.count_nonzero(weight).item()) != 0
            or int(torch.count_nonzero(bias).item()) != 0
        ):
            raise ValueError("actor Full12 output layer is not exact zero")
    except Exception as exc:
        raise EvaluationArtifactError(
            "checkpoint_initial actor output layer is not provably exact zero"
        ) from exc


def _validate_checkpoint_creation_runtime(
    manifest: Mapping[str, Any], *, role: str
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    """Bind a checkpoint to the finalized managed train run that created it."""

    identity_path = _resolved_path(
        str(manifest.get("creation_runtime_identity_path", "")),
        label=f"{role} creation runtime identity",
    )
    if identity_path.name != "committed_runtime_identity.before.json":
        raise EvaluationArtifactError(
            f"{role} creation runtime identity path is not the before artifact"
        )
    run_dir = identity_path.parent
    run_record, run_payload = _capture_json_object(
        run_dir / "run_manifest.json",
        label=f"{role} creation training run manifest",
    )
    creation_run_kind = str(run_payload.get("run_kind", ""))
    allowed_creation_run = creation_run_kind == "train" or (
        role == "checkpoint_initial"
        and manifest.get("stage") == "initial_zero_residual"
        and creation_run_kind == "initial_checkpoint"
        and run_payload.get("entrypoint") == "wlr50_clean.ppo.cli"
        and run_payload.get("subcommand") == "initialize-zero-residual"
        and isinstance(run_payload.get("identity"), Mapping)
        and run_payload["identity"].get("training_stage")
        == "initialize-zero-residual"
        and run_payload["identity"].get("environment_count") == 1
    )
    if (
        run_payload.get("schema") != RUN_MANIFEST_SCHEMA
        or run_payload.get("lifecycle") != "SUCCEEDED"
        or run_payload.get("exit_code") != 0
        or not allowed_creation_run
        or run_payload.get("immutable_run_directory") is not True
        or _resolved_path(
            str(run_payload.get("run_dir", "")),
            label=f"{role} creation run directory",
        )
        != run_dir
    ):
        raise EvaluationArtifactError(
            f"{role} checkpoint creation run is not an allowed finalized managed run"
        )
    before_record, after_record, identity = _validate_worker_runtime_identity(
        run_dir,
        run_payload,
        label=f"{role} checkpoint creation run",
    )
    if (
        Path(str(before_record["path"])) != identity_path
        or before_record["sha256"]
        != manifest.get("creation_runtime_identity_sha256")
        or identity.get("git_commit") != manifest.get("source_git_commit")
        or identity.get("content_sha256")
        != manifest.get("committed_runtime_content_sha256")
    ):
        raise EvaluationArtifactError(
            f"{role} checkpoint creation runtime provenance is inconsistent"
        )
    return (run_record, before_record, after_record), identity


def _capture_bound_file(
    payload: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    raw_path = payload.get(path_key)
    expected_hash = payload.get(hash_key)
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise EvaluationArtifactError(f"{label} omits {path_key}/{hash_key}")
    record, _ = _capture_file_record(raw_path, label=label)
    if record["sha256"] != expected_hash:
        raise EvaluationArtifactError(f"{label} SHA-256 is stale")
    return Path(record["path"]), record


def _validate_checkpoint_role_contract(
    *,
    role: str,
    checkpoint: Path,
    checkpoint_record: Mapping[str, Any],
    checkpoint_bytes: bytes,
    manifest_path: Path,
    manifest_record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Use production promotion validators to prove the checkpoint's role."""

    checkpoint_hash = str(checkpoint_record["sha256"])
    manifest_hash = str(manifest_record["sha256"])
    _require_checkpoint_abi(
        checkpoint, checkpoint_hash, checkpoint_bytes, manifest, role=role
    )
    creation_records, _ = _validate_checkpoint_creation_runtime(
        manifest, role=role
    )
    supporting: list[Mapping[str, Any]] = list(creation_records)
    if role == "checkpoint_initial":
        global_decisions = manifest.get("global_policy_decisions")
        if (
            manifest.get("stage") != "initial_zero_residual"
            or isinstance(global_decisions, bool)
            or not isinstance(global_decisions, int)
            or global_decisions != 0
            or manifest.get("zero_mean_actor_output_layer_verified") is not True
            or manifest.get("publication_role") not in (None, "")
            or manifest.get("validation_promotion_authorized") is True
            or manifest.get("locked_test_authorized") is True
            or manifest.get("promotion_authorized") is True
        ):
            raise EvaluationArtifactError(
                "checkpoint_initial is not the untrained zero-residual policy"
            )
        _require_initial_zero_actor_output(checkpoint_bytes, manifest)
        return tuple(supporting)

    try:
        provenance = validate_checkpoint_artifact_provenance(checkpoint, manifest_path)
    except CheckpointPromotionError as exc:
        raise EvaluationArtifactError(f"{role} checkpoint provenance is invalid: {exc}") from exc
    if (
        provenance.checkpoint_sha256 != checkpoint_hash
        or provenance.manifest_sha256 != manifest_hash
        or provenance.checkpoint_path != checkpoint
        or provenance.manifest_path != manifest_path
    ):
        raise EvaluationArtifactError(f"{role} checkpoint provenance changed during validation")

    if role == "checkpoint_smoke":
        global_decisions = manifest.get("global_policy_decisions")
        if (
            manifest.get("stage") != "smoke"
            or isinstance(global_decisions, bool)
            or not isinstance(global_decisions, int)
            or global_decisions <= 0
            or manifest.get("publication_role") not in (None, "")
            or manifest.get("validation_promotion_authorized") is True
            or manifest.get("locked_test_authorized") is True
            or manifest.get("promotion_authorized") is True
        ):
            raise EvaluationArtifactError(
                "checkpoint_smoke is not a non-promoted smoke-stage checkpoint"
            )
        return tuple(supporting)

    if role == "checkpoint_best":
        if manifest.get("stage") != "full-episode":
            raise EvaluationArtifactError(
                "checkpoint_best is not a full-episode-stage checkpoint"
            )
        validation_path = _resolved_path(
            checkpoint.parent.parent
            / "manifests"
            / VALIDATION_PROMOTION_MANIFEST_NAME,
            label="checkpoint_best validation promotion manifest",
        )
        promotion_path = _resolved_path(
            str(manifest.get("promotion_decision", "")),
            label="checkpoint_best promotion decision",
        )
        try:
            validated, resolved_validation, _, validation_hash = (
                _validate_best_validation_source(
                    checkpoint,
                    manifest_path,
                    validation_path,
                    promotion_path,
                )
            )
        except CheckpointPromotionError as exc:
            raise EvaluationArtifactError(
                f"checkpoint_best validation-promotion chain is invalid: {exc}"
            ) from exc
        if validated.checkpoint_sha256 != checkpoint_hash:
            raise EvaluationArtifactError("checkpoint_best bytes changed during validation")
        validation_record, _ = _capture_file_record(
            resolved_validation, label="checkpoint_best validation promotion manifest"
        )
        promotion_record, _ = _capture_file_record(
            promotion_path, label="checkpoint_best promotion decision"
        )
        if validation_record["sha256"] != validation_hash:
            raise EvaluationArtifactError(
                "checkpoint_best validation promotion changed during validation"
            )
        supporting.extend((validation_record, promotion_record))
        return tuple(supporting)

    if role != "checkpoint_improved":
        raise EvaluationArtifactError(f"unsupported checkpoint lifecycle role: {role}")
    required_true = (
        "validation_promotion_authorized",
        "locked_test_authorized",
        "promotion_authorized",
    )
    if (
        manifest.get("stage") != "full-episode"
        or manifest.get("publication_role") != "improved"
        or any(
        manifest.get(name) is not True for name in required_true
        )
    ):
        raise EvaluationArtifactError(
            "checkpoint_improved lacks two-stage promotion authorization"
        )
    best_path, best_record = _capture_bound_file(
        manifest,
        path_key="source_best_validation_checkpoint",
        hash_key="source_best_validation_checkpoint_sha256",
        label="checkpoint_improved source best checkpoint",
    )
    best_manifest_path, best_manifest_record = _capture_bound_file(
        manifest,
        path_key="source_best_validation_manifest",
        hash_key="source_best_validation_manifest_sha256",
        label="checkpoint_improved source best manifest",
    )
    validation_path, validation_record = _capture_bound_file(
        manifest,
        path_key="validation_promotion_manifest",
        hash_key="validation_promotion_manifest_sha256",
        label="checkpoint_improved validation promotion manifest",
    )
    decision_path, decision_record = _capture_bound_file(
        manifest,
        path_key="promotion_decision",
        hash_key="promotion_decision_sha256",
        label="checkpoint_improved promotion decision",
    )
    locked_path, locked_record = _capture_bound_file(
        manifest,
        path_key="locked_test_aggregate",
        hash_key="locked_test_aggregate_sha256",
        label="checkpoint_improved locked-test aggregate",
    )
    try:
        best, resolved_validation, validation_payload, validation_hash = _validate_best_validation_source(
            best_path,
            best_manifest_path,
            validation_path,
            decision_path,
        )
        locked = _validate_locked_test_aggregate(locked_path, best)
    except CheckpointPromotionError as exc:
        raise EvaluationArtifactError(
            f"checkpoint_improved promotion chain is invalid: {exc}"
        ) from exc
    expected_locked_summary = {
        "aggregate": str(locked.path),
        "aggregate_sha256": locked.sha256,
        "seeds": list(LOCKED_TEST_SEEDS),
        "success_count": len(LOCKED_TEST_SEEDS),
        "body_collision_count": 0,
        "wheel_only_climb_count": 0,
        "safety_abort_count": 0,
        "all_under_maximum_duration": True,
        "hash_gates": dict(locked.payload["hash_gates"]),
        "worker_artifact_sha256": [
            dict(row) for row in locked.worker_artifact_sha256
        ],
    }
    if (
        best.checkpoint_sha256 != checkpoint_hash
        or best.manifest.get("stage") != "full-episode"
        or best.manifest.get("training_seed") != manifest.get("training_seed")
        or resolved_validation != validation_path
        or validation_hash != validation_record["sha256"]
        or locked.sha256 != locked_record["sha256"]
        or manifest.get("locked_test") != expected_locked_summary
    ):
        raise EvaluationArtifactError(
            "checkpoint_improved is not byte-bound to its validated best/locked-test chain"
        )
    promotion_manifest_path = _resolved_path(
        checkpoint.parent.parent / "manifests" / PROMOTION_MANIFEST_NAME,
        label="checkpoint_improved promotion manifest",
    )
    promotion_record, promotion_payload = _capture_json_object(
        promotion_manifest_path, label="checkpoint_improved promotion manifest"
    )
    published = promotion_payload.get("published_checkpoints")
    published_best = published.get("best_validation") if isinstance(published, Mapping) else None
    published_improved = published.get("improved") if isinstance(published, Mapping) else None
    if (
        promotion_payload.get("schema") != CHECKPOINT_IMPROVED_PROMOTION_SCHEMA
        or promotion_payload.get("valid") is not True
        or promotion_payload.get("status") != "PROMOTED_IMPROVED"
        or promotion_payload.get("two_stage_promotion") is not True
        or promotion_payload.get("validation_decision_alone_cannot_authorize_improved")
        is not True
        or promotion_payload.get("filename_inference_used") is not False
        or promotion_payload.get("byte_identical_best_and_improved") is not True
        or promotion_payload.get("immutable_no_overwrite") is not True
        or _resolved_path(
            str(promotion_payload.get("validation_promotion_manifest", "")),
            label="final promotion validation manifest",
        )
        != validation_path
        or promotion_payload.get("validation_promotion_manifest_sha256")
        != validation_record["sha256"]
        or promotion_payload.get("validation_promotion") != validation_payload
        or _resolved_path(
            str(promotion_payload.get("promotion_decision", "")),
            label="final promotion decision",
        )
        != decision_path
        or promotion_payload.get("promotion_decision_sha256")
        != decision_record["sha256"]
        or _resolved_path(
            str(promotion_payload.get("locked_test_aggregate", "")),
            label="final promotion locked-test aggregate",
        )
        != locked_path
        or promotion_payload.get("locked_test_aggregate_sha256")
        != locked_record["sha256"]
        or promotion_payload.get("locked_test") != expected_locked_summary
        or not isinstance(published_best, Mapping)
        or not isinstance(published_improved, Mapping)
        or _resolved_path(
            str(published_best.get("path", "")), label="published best checkpoint"
        )
        != best_path
        or published_best.get("sha256") != checkpoint_hash
        or _resolved_path(
            str(published_best.get("manifest", "")), label="published best manifest"
        )
        != best_manifest_path
        or published_best.get("manifest_sha256") != best_manifest_record["sha256"]
        or _resolved_path(
            str(published_improved.get("path", "")),
            label="published improved checkpoint",
        )
        != checkpoint
        or published_improved.get("sha256") != checkpoint_hash
        or _resolved_path(
            str(published_improved.get("manifest", "")),
            label="published improved manifest",
        )
        != manifest_path
        or published_improved.get("manifest_sha256") != manifest_hash
    ):
        raise EvaluationArtifactError(
            "checkpoint_improved final promotion manifest is inconsistent"
        )
    supporting.extend(
        (
            best_record,
            best_manifest_record,
            validation_record,
            decision_record,
            locked_record,
            promotion_record,
        )
    )
    return tuple(supporting)


def _final_lifecycle_aggregate_evidence(
    aggregate_path: str | Path,
    *,
    role: str,
) -> FinalLifecycleAggregateEvidence:
    """Revalidate one aggregate and every worker/checkpoint it references."""

    selected_role = str(role).strip()
    if selected_role not in FINAL_LIFECYCLE_ROLES:
        raise EvaluationArtifactError(f"unknown final lifecycle role: {selected_role!r}")
    path = _resolved_path(
        aggregate_path, label=f"{selected_role} evaluation aggregate"
    )
    aggregate_record, payload = _capture_json_object(
        path, label=f"{selected_role} evaluation aggregate"
    )
    aggregate_hash = str(aggregate_record["sha256"])
    if payload.get("schema") != "wlr50_clean.fresh_process_episode_batch.v1":
        raise EvaluationArtifactError(
            f"{selected_role} aggregate has an unexpected schema"
        )
    expected_worker_role = "baseline" if selected_role == "pure_fsm" else "candidate"
    if payload.get("role") != expected_worker_role:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate has role {payload.get('role')!r}, "
            f"expected {expected_worker_role!r}"
        )
    if payload.get("seed_set") != "validation":
        raise EvaluationArtifactError(
            f"{selected_role} aggregate is not a validation-seed aggregate"
        )
    seeds_raw = payload.get("seeds")
    if not isinstance(seeds_raw, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate seeds are invalid"
        )
    seeds = tuple(seeds_raw)
    if seeds != BASELINE_VALIDATION_SEEDS:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate must use paired validation seeds 2001-2005"
        )
    for name in ("fresh_process_per_episode", "deterministic_evaluation"):
        if payload.get(name) is not True:
            raise EvaluationArtifactError(
                f"{selected_role} aggregate requires {name}=true"
            )
    if not isinstance(payload.get("passed"), bool):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate passed field must be boolean"
        )
    checkpoint_record: Mapping[str, Any] | None = None
    checkpoint_bytes: bytes | None = None
    if selected_role == "pure_fsm":
        if payload.get("pure_fsm_zero_residual") is not True:
            raise EvaluationArtifactError(
                "pure_fsm aggregate is not authoritative zero-residual FSM evidence"
            )
        if payload.get("checkpoint") is not None or payload.get("checkpoint_sha256") is not None:
            raise EvaluationArtifactError(
                "pure_fsm aggregate must not name a PPO checkpoint"
            )
        checkpoint = None
        checkpoint_hash = None
    else:
        if payload.get("deterministic_mean_policy") is not True:
            raise EvaluationArtifactError(
                f"{selected_role} aggregate is not deterministic mean-policy evidence"
            )
        checkpoint = _resolved_path(
            str(payload.get("checkpoint", "")),
            label=f"{selected_role} checkpoint",
        )
        expected_filename = _FINAL_LIFECYCLE_CHECKPOINT_FILENAMES[selected_role]
        if checkpoint.name != expected_filename or not checkpoint.is_file():
            raise EvaluationArtifactError(
                f"{selected_role} must bind existing {expected_filename}"
            )
        checkpoint_record, checkpoint_bytes = _capture_file_record(
            checkpoint, label=f"{selected_role} checkpoint"
        )
        checkpoint_hash = str(checkpoint_record["sha256"])
        if payload.get("checkpoint_sha256") != checkpoint_hash:
            raise EvaluationArtifactError(
                f"{selected_role} aggregate checkpoint SHA-256 is stale"
            )

    workers_raw = payload.get("workers")
    episodes_raw = payload.get("episodes")
    directories_raw = payload.get("canonical_episode_dirs")
    if (
        not isinstance(workers_raw, list)
        or not isinstance(episodes_raw, list)
        or not isinstance(directories_raw, list)
        or any(not isinstance(row, Mapping) for row in workers_raw)
        or any(not isinstance(row, Mapping) for row in episodes_raw)
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate worker/episode evidence is malformed"
        )
    if (
        payload.get("episode_count") != len(seeds)
        or len(workers_raw) != len(seeds)
        or len(episodes_raw) != len(seeds)
        or len(directories_raw) != len(seeds)
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate is not a complete five-worker evaluation"
        )
    if any(
        not isinstance(row.get("worker_gate_passed"), bool) for row in workers_raw
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate worker gate fields must be boolean"
        )
    boolean_episode_fields = (
        "task_success",
        "body_collision",
        "wheel_only_climb",
        "safety_abort",
        "under_maximum_duration",
    )
    for index, (seed, row) in enumerate(zip(seeds, episodes_raw, strict=True)):
        if (
            isinstance(row.get("seed"), bool)
            or not isinstance(row.get("seed"), int)
            or row.get("seed") != seed
            or any(not isinstance(row.get(name), bool) for name in boolean_episode_fields)
        ):
            raise EvaluationArtifactError(
                f"{selected_role} aggregate episode {index} has malformed seed/booleans"
            )
    worker_gate_pass_count = sum(
        row.get("worker_gate_passed") is True for row in workers_raw
    )
    if payload.get("worker_gate_pass_count") != worker_gate_pass_count:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate worker gate count is inconsistent"
        )
    expected_counts = {
        "success_count": sum(row.get("task_success") is True for row in episodes_raw),
        "body_collision_count": sum(
            row.get("body_collision") is True for row in episodes_raw
        ),
        "wheel_only_climb_count": sum(
            row.get("wheel_only_climb") is True for row in episodes_raw
        ),
        "safety_abort_count": sum(
            row.get("safety_abort") is True for row in episodes_raw
        ),
    }
    if any(payload.get(name) != value for name, value in expected_counts.items()):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate outcome counts disagree with its episodes"
        )
    if payload.get("all_under_maximum_duration") is not all(
        row.get("under_maximum_duration") is True for row in episodes_raw
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate duration gate disagrees with its episodes"
        )
    try:
        infrastructure_clean = all(
            int(row.get("recording_runtime_access_count", -1)) == 0
            and int(row.get("in_episode_root_write_count", -1)) == 0
            for row in episodes_raw
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate infrastructure counters are invalid"
        ) from exc
    if not infrastructure_clean:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate contains runtime recording access or root writes"
        )
    expected_passed = bool(
        all(row.get("task_success") is True for row in episodes_raw)
        and all(row.get("body_collision") is False for row in episodes_raw)
        and all(row.get("wheel_only_climb") is False for row in episodes_raw)
        and all(row.get("safety_abort") is False for row in episodes_raw)
        and all(row.get("under_maximum_duration") is True for row in episodes_raw)
        and infrastructure_clean
        and worker_gate_pass_count == len(seeds)
    )
    if payload.get("passed") is not expected_passed:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate passed field disagrees with worker/episode evidence"
        )
    if selected_role in _FINAL_LIFECYCLE_REQUIRED_PASS_ROLES and not expected_passed:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate must pass every physical and worker gate"
        )
    try:
        worker_run_dirs = tuple(
            _resolved_path(
                str(row["run_dir"]), label=f"{selected_role} worker directory"
            )
            for row in workers_raw
        )
        aggregate_dirs = tuple(
            _resolved_path(
                str(value), label=f"{selected_role} canonical episode directory"
            )
            for value in directories_raw
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate paths are invalid"
        ) from exc
    batch = collect_fresh_process_episode_workers(
        worker_run_dirs,
        seeds=seeds,
        role=expected_worker_role,
        checkpoint_path=checkpoint,
    )
    if aggregate_dirs != batch.canonical_episode_dirs:
        raise EvaluationArtifactError(
            f"{selected_role} aggregate canonical directories disagree with its workers"
        )
    if tuple(dict(row) for row in workers_raw) != tuple(
        dict(row) for row in batch.worker_rows
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate worker hashes or paths are stale"
        )
    if tuple(dict(row) for row in episodes_raw) != tuple(
        dict(row) for row in batch.episode_rows
    ):
        raise EvaluationArtifactError(
            f"{selected_role} aggregate episode rows disagree with its workers"
        )

    manifest_path: Path | None = None
    manifest_hash: str | None = None
    manifest_record: Mapping[str, Any] | None = None
    manifest_payload: Mapping[str, Any] | None = None
    training_seed: int | None = None
    supporting_files: tuple[Mapping[str, Any], ...] = ()
    if checkpoint is not None:
        expected_manifest_name = f"{checkpoint.stem}_manifest.json"
        for seed, worker in zip(seeds, batch.worker_rows, strict=True):
            result_path = _resolved_path(
                str(worker["worker_result"]),
                label=f"{selected_role} seed {seed} checkpoint evaluation",
            )
            result = _load_json_object(
                result_path, f"{selected_role} seed {seed} checkpoint evaluation"
            )
            raw_manifest = str(result.get("checkpoint_manifest", "")).strip()
            if not raw_manifest:
                raise EvaluationArtifactError(
                    f"{selected_role} seed {seed} omits its checkpoint manifest"
                )
            current_manifest = _resolved_path(
                raw_manifest, label=f"{selected_role} seed {seed} checkpoint manifest"
            )
            if current_manifest.name != expected_manifest_name or not current_manifest.is_file():
                raise EvaluationArtifactError(
                    f"{selected_role} seed {seed} names an invalid checkpoint manifest"
                )
            current_manifest_record, current_manifest_payload = _capture_json_object(
                current_manifest,
                label=f"{selected_role} seed {seed} checkpoint manifest",
            )
            current_manifest_hash = str(current_manifest_record["sha256"])
            if result.get("checkpoint_manifest_sha256") != current_manifest_hash:
                raise EvaluationArtifactError(
                    f"{selected_role} seed {seed} checkpoint manifest SHA-256 is stale"
                )
            if manifest_path is None:
                manifest_path = current_manifest
                manifest_hash = current_manifest_hash
                manifest_record = current_manifest_record
                manifest_payload = current_manifest_payload
            elif current_manifest != manifest_path or current_manifest_hash != manifest_hash:
                raise EvaluationArtifactError(
                    f"{selected_role} workers do not share one checkpoint manifest"
                )
        assert manifest_path is not None and manifest_hash is not None
        assert manifest_record is not None and manifest_payload is not None
        assert checkpoint_record is not None
        assert checkpoint_bytes is not None
        supporting_files = _validate_checkpoint_role_contract(
            role=selected_role,
            checkpoint=checkpoint,
            checkpoint_record=checkpoint_record,
            checkpoint_bytes=checkpoint_bytes,
            manifest_path=manifest_path,
            manifest_record=manifest_record,
            manifest=manifest_payload,
        )
        training_seed = int(manifest_payload["training_seed"])

    source_groups: list[Mapping[str, Any]] = []
    committed_runtime_identity: Mapping[str, Any] | None = None
    for seed, worker, episode_dir in zip(
        seeds, batch.worker_rows, batch.canonical_episode_dirs, strict=True
    ):
        run_dir = _resolved_path(
            str(worker["run_dir"]), label=f"{selected_role} seed {seed} worker directory"
        )
        run_record, run_payload = _capture_json_object(
            run_dir / "run_manifest.json",
            label=f"{selected_role} seed {seed} run manifest",
        )
        identity_before, identity_after, identity = _validate_worker_runtime_identity(
            run_dir,
            run_payload,
            label=f"{selected_role} seed {seed}",
        )
        if committed_runtime_identity is None:
            committed_runtime_identity = identity
        elif dict(identity) != dict(committed_runtime_identity):
            raise EvaluationArtifactError(
                f"{selected_role} workers do not share one committed runtime identity"
            )
        result_record, _ = _capture_file_record(
            str(worker["worker_result"]),
            label=f"{selected_role} seed {seed} worker result",
        )
        canonical_records: list[Mapping[str, Any]] = []
        for name in CANONICAL_EPISODE_FILES:
            record, _ = _capture_file_record(
                episode_dir / name,
                label=f"{selected_role} seed {seed} canonical {name}",
                allow_empty=name not in NONEMPTY_CANONICAL_EPISODE_FILES,
            )
            canonical_records.append({"name": name, **record})
        trial_record = next(
            record for record in canonical_records if record["name"] == "trial_manifest.json"
        )
        if (
            run_record["sha256"] != worker["run_manifest_sha256"]
            or result_record["sha256"] != worker["worker_result_sha256"]
            or trial_record["sha256"] != worker["trial_manifest_sha256"]
        ):
            raise EvaluationArtifactError(
                f"{selected_role} seed {seed} worker provenance changed during capture"
            )
        source_groups.append(
            {
                "role": selected_role,
                "seed": seed,
                "worker_run_dir": str(run_dir),
                "canonical_episode_dir": str(episode_dir),
                "run_manifest": run_record,
                "worker_result": result_record,
                "committed_runtime_identity_before": identity_before,
                "committed_runtime_identity_after": identity_after,
                "trial_manifest": dict(trial_record),
                "canonical_files": canonical_records,
            }
        )

    _record_unchanged(
        aggregate_record, label=f"{selected_role} evaluation aggregate"
    )
    assert committed_runtime_identity is not None
    if manifest_payload is not None and (
        manifest_payload.get("source_git_commit")
        != committed_runtime_identity.get("git_commit")
        or manifest_payload.get("committed_runtime_content_sha256")
        != committed_runtime_identity.get("content_sha256")
    ):
        raise EvaluationArtifactError(
            f"{selected_role} checkpoint runtime contract differs from its evaluation workers"
        )
    return FinalLifecycleAggregateEvidence(
        role=selected_role,
        aggregate_path=path,
        aggregate_sha256=aggregate_hash,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_manifest_path=manifest_path,
        checkpoint_manifest_sha256=manifest_hash,
        training_seed=training_seed,
        source_git_commit=(
            None
            if manifest_payload is None
            else str(manifest_payload["source_git_commit"])
        ),
        committed_runtime_content_sha256=(
            None
            if manifest_payload is None
            else str(manifest_payload["committed_runtime_content_sha256"])
        ),
        creation_runtime_identity_sha256=(
            None
            if manifest_payload is None
            else str(manifest_payload["creation_runtime_identity_sha256"])
        ),
        creation_runtime_identity_path=(
            None
            if manifest_payload is None
            else _resolved_path(
                str(manifest_payload["creation_runtime_identity_path"]),
                label=f"{selected_role} creation runtime identity",
            )
        ),
        seeds=seeds,
        worker_run_dirs=worker_run_dirs,
        canonical_episode_dirs=batch.canonical_episode_dirs,
        source_groups=tuple(source_groups),
        supporting_files=supporting_files,
        committed_runtime_identity=committed_runtime_identity,
    )


def validate_final_lifecycle_aggregate_evidence(
    aggregate_path: str | Path, *, role: str
) -> FinalLifecycleAggregateEvidence:
    """Public strict validator shared by reporting and delivery finalization."""

    return _final_lifecycle_aggregate_evidence(aggregate_path, role=role)


def _source_group_file_records(
    evidence: FinalLifecycleAggregateEvidence,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for group in evidence.source_groups:
        records.extend(
            (
                group["run_manifest"],
                group["worker_result"],
                group["committed_runtime_identity_before"],
                group["committed_runtime_identity_after"],
            )
        )
        records.extend(group["canonical_files"])
    records.extend(evidence.supporting_files)
    return tuple(records)


def _revalidate_final_lifecycle_sources(
    evidence_by_role: Mapping[str, FinalLifecycleAggregateEvidence],
) -> None:
    """Re-hash every captured source immediately before publication."""

    for role, evidence in evidence_by_role.items():
        if sha256_file(evidence.aggregate_path) != evidence.aggregate_sha256:
            raise EvaluationArtifactError(f"{role} aggregate changed during final evaluation")
        if evidence.checkpoint_path is not None and (
            sha256_file(evidence.checkpoint_path) != evidence.checkpoint_sha256
        ):
            raise EvaluationArtifactError(f"{role} checkpoint changed during final evaluation")
        if evidence.checkpoint_manifest_path is not None and (
            sha256_file(evidence.checkpoint_manifest_path)
            != evidence.checkpoint_manifest_sha256
        ):
            raise EvaluationArtifactError(
                f"{role} checkpoint manifest changed during final evaluation"
            )
        for index, record in enumerate(_source_group_file_records(evidence)):
            _record_unchanged(record, label=f"{role} provenance source {index}")


def _require_output_disjoint_from_final_sources(
    output_directory: str | Path,
    evidence_by_role: Mapping[str, FinalLifecycleAggregateEvidence],
) -> Path:
    output = _resolved_path(output_directory, label="final lifecycle output directory")
    input_paths: list[Path] = []
    input_trees: list[Path] = []
    for evidence in evidence_by_role.values():
        input_paths.append(evidence.aggregate_path)
        input_trees.extend(evidence.worker_run_dirs)
        input_trees.extend(evidence.canonical_episode_dirs)
        if evidence.checkpoint_path is not None:
            input_paths.append(evidence.checkpoint_path)
        if evidence.checkpoint_manifest_path is not None:
            input_paths.append(evidence.checkpoint_manifest_path)
        input_paths.extend(Path(str(row["path"])) for row in evidence.supporting_files)
    conflict = next(
        (
            source
            for source in (*input_trees, *input_paths)
            if _paths_overlap(output, source)
        ),
        None,
    )
    if conflict is not None:
        raise EvaluationArtifactError(
            "final lifecycle output overlaps an input source: "
            f"output={output}, source={conflict}"
        )
    return output


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

    def index(
        runs: Sequence[LiveRunEvaluation],
        label: str,
        *,
        require_complete: bool,
    ) -> dict[int, LiveRunEvaluation]:
        indexed: dict[int, LiveRunEvaluation] = {}
        for run in runs:
            seed = int(run.seed)
            if seed in indexed:
                raise EvaluationArtifactError(f"{label} seeds must be unique")
            if require_complete:
                _require_phase_order(
                    run.phase_rows, label=f"{label} seed {seed} phase metrics"
                )
            else:
                phase_sequence = _require_canonical_phase_prefix(
                    run.phase_rows, label=f"{label} seed {seed} phase metrics"
                )
                completed = tuple(run.termination.completed_phases)
                if run.termination.task_success:
                    if (
                        phase_sequence != PHASE_IDS
                        or completed != PHASE_IDS
                        or run.termination.completed_p01_p13 is not True
                    ):
                        raise EvaluationArtifactError(
                            f"{label} seed {seed} reports success without P01-P13"
                        )
                elif completed not in {phase_sequence, phase_sequence[:-1]} or (
                    run.termination.final_state_id is not None
                    and run.termination.final_state_id != phase_sequence[-1]
                ):
                    raise EvaluationArtifactError(
                        f"{label} seed {seed} failure disagrees with its phase prefix"
                    )
            indexed[seed] = run
        return indexed

    baseline = index(baseline_runs, "baseline", require_complete=True)
    # A physically unsuccessful fresh-process worker may stop at any truthful
    # P01..Pn prefix.  Aggregate provenance later proves whether this is a
    # failed capacity/evaluation result; only a complete candidate can pass.
    candidate = index(candidate_runs, "candidate", require_complete=False)
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
        residual_phases = _require_canonical_phase_prefix(
            run.residual_activity_rows,
            label=f"candidate seed {run.seed} residual activity",
        )
        phase_phases = _require_canonical_phase_prefix(
            run.phase_rows, label=f"candidate seed {run.seed} phase metrics"
        )
        if residual_phases != phase_phases:
            raise EvaluationArtifactError(
                f"candidate seed {run.seed} residual activity differs from its phase prefix"
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
        reward_phases = _require_canonical_phase_prefix(
            run.reward_contribution_rows,
            label=f"candidate seed {run.seed} reward contributions",
        )
        phase_phases = _require_canonical_phase_prefix(
            run.phase_rows, label=f"candidate seed {run.seed} phase metrics"
        )
        if reward_phases != phase_phases:
            raise EvaluationArtifactError(
                f"candidate seed {run.seed} reward contributions differ from its phase prefix"
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


def _final_lifecycle_termination_rows(
    runs_by_role: Mapping[str, Sequence[LiveRunEvaluation]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if tuple(runs_by_role) != FINAL_LIFECYCLE_ROLES:
        raise EvaluationArtifactError("final lifecycle role order changed")
    for role in FINAL_LIFECYCLE_ROLES:
        rows.extend(
            {
                "checkpoint": role,
                "seed": int(run.seed),
                "run_directory": str(run.run_directory),
                **asdict(run.termination),
            }
            for run in runs_by_role[role]
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
        proposed = candidate.get(phase)
        difference = differences.get(phase, {})
        if proposed is None:
            rows.append(
                {
                    "phase": phase,
                    **{
                        f"fsm_baseline_{name}": value
                        for name, value in base.items()
                        if name != "phase"
                    },
                    **{
                        name: value
                        for name, value in difference.items()
                        if name != "phase"
                    },
                    "candidate_phase_observed_in_all_runs": False,
                }
            )
            continue
        common = sorted(set(base).intersection(proposed).difference({"phase"}))
        row: dict[str, Any] = {"phase": phase}
        for name in common:
            row[f"fsm_baseline_{name}"] = base[name]
            row[f"candidate_{name}"] = proposed[name]
        row.update(
            {
                name: value
                for name, value in difference.items()
                if name != "phase"
            }
        )
        if difference.get("comparison_available") is False:
            row["candidate_phase_observed_in_all_runs"] = True
        rows.append(row)
    return tuple(rows)


def _final_lifecycle_phase_rows(comparison: Any) -> tuple[dict[str, Any], ...]:
    """Return only the authoritative pure-FSM versus improved phase comparison."""

    baseline = {str(row["phase"]): row for row in comparison.baseline_phase_rows}
    improved = {str(row["phase"]): row for row in comparison.candidate_phase_rows}
    differences = {
        str(row["phase"]): row for row in comparison.phase_comparison_rows
    }
    rows: list[dict[str, Any]] = []
    for phase in PHASE_IDS:
        base = baseline[phase]
        proposed = improved[phase]
        common = sorted(set(base).intersection(proposed).difference({"phase"}))
        row: dict[str, Any] = {
            "phase": phase,
            "baseline_checkpoint": "pure_fsm",
            "improved_checkpoint": "checkpoint_improved",
        }
        for name in common:
            row[f"pure_fsm_{name}"] = base[name]
            row[f"checkpoint_improved_{name}"] = proposed[name]
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


def _final_lifecycle_checkpoint_row(
    *,
    evidence: FinalLifecycleAggregateEvidence,
    runs: Sequence[LiveRunEvaluation],
    comparison: Any,
) -> dict[str, Any]:
    row = _checkpoint_row(
        role=("candidate" if evidence.role == "checkpoint_improved" else "baseline"),
        checkpoint=evidence.role,
        runs=runs,
        checkpoint_path=evidence.checkpoint_path,
        comparison=comparison,
    )
    row.update(
        {
            "role": evidence.role,
            "checkpoint": evidence.role,
            "checkpoint_sha256": evidence.checkpoint_sha256,
            "checkpoint_manifest_path": (
                str(evidence.checkpoint_manifest_path)
                if evidence.checkpoint_manifest_path is not None
                else None
            ),
            "checkpoint_manifest_sha256": evidence.checkpoint_manifest_sha256,
            "evaluation_aggregate_path": str(evidence.aggregate_path),
            "evaluation_aggregate_sha256": evidence.aggregate_sha256,
        }
    )
    return row


def _validated_paired_aggregate_binding(
    value: Mapping[str, Any],
    *,
    role: str,
    runs: Sequence[LiveRunEvaluation],
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    """Validate the immutable aggregate record supplied by the strict capturer."""

    if not isinstance(value, Mapping):
        raise EvaluationArtifactError(f"{role} evaluation aggregate binding is missing")
    records = value.get("source_file_records")
    canonical_dirs = [str(run.run_directory.resolve()) for run in runs]
    worker_dirs = [str(Path(path).resolve().parent) for path in canonical_dirs]
    evaluated_physical_pass = all(
        run.termination.task_success
        and run.termination.completed_p01_p13
        and not run.termination.body_collision
        and not run.termination.wheel_only_climb
        and not run.termination.physics_explosion_or_fall
        and not run.termination.safety_abort
        and run.termination.duration_s <= 200.0
        and run.termination.runtime_recording_access_count == 0
        for run in runs
    )
    if (
        value.get("schema") != "wlr50_clean.validation_aggregate_binding.v1"
        or value.get("role") != role
        or value.get("seeds") != [int(run.seed) for run in runs]
        or value.get("canonical_episode_dirs") != canonical_dirs
        or value.get("worker_run_dirs") != worker_dirs
        or not isinstance(value.get("physical_passed"), bool)
        or (role == "baseline" and value.get("physical_passed") is not True)
        or (value.get("physical_passed") is True and not evaluated_physical_pass)
        or not isinstance(records, list)
        or not records
        or any(not isinstance(row, Mapping) for row in records)
    ):
        raise EvaluationArtifactError(
            f"{role} evaluation aggregate binding is malformed or spliced"
        )
    paths = [str(row.get("path", "")) for row in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)) or any(
        not Path(path).is_absolute() for path in paths
    ):
        raise EvaluationArtifactError(
            f"{role} evaluation aggregate source inventory is invalid"
        )
    for index, row in enumerate(records):
        digest = row.get("sha256")
        if (
            set(row) != {"path", "bytes", "sha256"}
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvaluationArtifactError(
                f"{role} evaluation aggregate source record {index} is invalid"
            )
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != value.get("source_file_set_sha256"):
        raise EvaluationArtifactError(
            f"{role} evaluation aggregate source inventory hash is invalid"
        )
    aggregate_path = str(value.get("path", ""))
    aggregate_record = next(
        (row for row in records if row.get("path") == aggregate_path), None
    )
    if (
        aggregate_record is None
        or aggregate_record.get("bytes") != value.get("bytes")
        or aggregate_record.get("sha256") != value.get("sha256")
    ):
        raise EvaluationArtifactError(
            f"{role} evaluation aggregate record is absent from its source inventory"
        )
    if role == "candidate":
        if checkpoint_path is None:
            if any(
                value.get(name) is not None
                for name in (
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "checkpoint_manifest_path",
                    "checkpoint_manifest_sha256",
                )
            ):
                raise EvaluationArtifactError(
                    "checkpoint-free candidate aggregate binding names a checkpoint"
                )
        else:
            manifest_digest = value.get("checkpoint_manifest_sha256")
            if (
                Path(str(value.get("checkpoint_path", ""))).resolve()
                != checkpoint_path
                or value.get("checkpoint_sha256") != sha256_file(checkpoint_path)
                or not str(value.get("checkpoint_manifest_path", "")).strip()
                or not isinstance(manifest_digest, str)
                or len(manifest_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in manifest_digest
                )
            ):
                raise EvaluationArtifactError(
                    "candidate validation aggregate binds a different checkpoint or manifest"
                )
    return dict(value)


def export_paired_evaluation_artifacts(
    output_directory: str | Path,
    *,
    baseline_runs: Sequence[LiveRunEvaluation],
    candidate_runs: Sequence[LiveRunEvaluation],
    frozen_hashes_unchanged: bool,
    candidate_checkpoint_name: str,
    candidate_checkpoint_path: str | Path | None = None,
    baseline_evaluation_aggregate: Mapping[str, Any],
    candidate_validation_aggregate: Mapping[str, Any],
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
    baseline_aggregate = _validated_paired_aggregate_binding(
        baseline_evaluation_aggregate,
        role="baseline",
        runs=baseline,
        checkpoint_path=None,
    )
    candidate_aggregate = _validated_paired_aggregate_binding(
        candidate_validation_aggregate,
        role="candidate",
        runs=candidate,
        checkpoint_path=checkpoint_path,
    )

    candidate_has_incomplete_phase_evidence = any(
        tuple(str(row.get("phase", "")) for row in run.phase_rows) != PHASE_IDS
        for run in candidate
    )
    if (
        candidate_has_incomplete_phase_evidence
        and candidate_aggregate["physical_passed"] is not False
    ):
        raise EvaluationArtifactError(
            "candidate aggregate claims a pass for incomplete P01-P13 evidence"
        )

    comparison = paired_baseline_candidate_promotion(
        baseline,
        candidate,
        frozen_hashes_unchanged=bool(frozen_hashes_unchanged),
        minimum_paired_seeds=int(minimum_paired_seeds),
    )
    if candidate_aggregate["physical_passed"] is False and comparison.promotion.promoted:
        raise EvaluationArtifactError(
            "candidate aggregate failure is not represented by the promotion gates"
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
        "baseline_evaluation_aggregate": baseline_aggregate,
        "candidate_validation_aggregate": candidate_aggregate,
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
    for role, aggregate in (
        ("baseline", baseline_aggregate),
        ("candidate", candidate_aggregate),
    ):
        for index, record in enumerate(aggregate["source_file_records"]):
            _record_unchanged(
                record, label=f"{role} paired aggregate source {index}"
            )
    _publish_idempotently(
        csv_specs, (_JsonPublication(paths.promotion_decision, decision_payload),)
    )
    return paths


def _require_final_lifecycle_runs(
    role: str,
    runs: Sequence[LiveRunEvaluation],
) -> tuple[LiveRunEvaluation, ...]:
    selected = tuple(runs)
    if tuple(int(run.seed) for run in selected) != BASELINE_VALIDATION_SEEDS:
        raise EvaluationArtifactError(
            f"{role} evaluator did not return validation seeds 2001-2005 in order"
        )
    require_accepted = role in _FINAL_LIFECYCLE_REQUIRED_PASS_ROLES
    for run in selected:
        if require_accepted:
            _require_phase_order(
                run.phase_rows, label=f"{role} seed {run.seed} phase metrics"
            )
            phase_sequence = PHASE_IDS
        else:
            phase_sequence = _require_canonical_phase_prefix(
                run.phase_rows, label=f"{role} seed {run.seed} phase metrics"
            )
        residual_sequence = _require_canonical_phase_prefix(
            run.residual_activity_rows,
            label=f"{role} seed {run.seed} residual activity",
        )
        reward_sequence = _require_canonical_phase_prefix(
            run.reward_contribution_rows,
            label=f"{role} seed {run.seed} reward contributions",
        )
        if residual_sequence != phase_sequence or reward_sequence != phase_sequence:
            raise EvaluationArtifactError(
                f"{role} seed {run.seed} metric streams do not share one phase prefix"
            )
        completed = tuple(run.termination.completed_phases)
        if run.termination.task_success:
            if (
                phase_sequence != PHASE_IDS
                or completed != PHASE_IDS
                or run.termination.completed_p01_p13 is not True
            ):
                raise EvaluationArtifactError(
                    f"{role} seed {run.seed} reports success without complete P01-P13 evidence"
                )
        else:
            allowed_completed = {phase_sequence, phase_sequence[:-1]}
            if completed not in allowed_completed or (
                run.termination.final_state_id is not None
                and run.termination.final_state_id != phase_sequence[-1]
            ):
                raise EvaluationArtifactError(
                    f"{role} seed {run.seed} failure termination disagrees with its P01 phase prefix"
                )
        if run.termination.runtime_recording_access_count != 0:
            raise EvaluationArtifactError(
                f"{role} seed {run.seed} accessed frozen recordings at runtime"
            )
        if not run.calibration.quality_passed:
            raise EvaluationArtifactError(
                f"{role} seed {run.seed} lacks accepted level calibration"
            )
        if require_accepted and (
            not run.termination.completed_p01_p13
            or not run.termination.task_success
            or run.termination.body_collision
            or run.termination.wheel_only_climb
            or run.termination.physics_explosion_or_fall
            or run.termination.safety_abort
            or run.termination.duration_s > 200.0
        ):
            raise EvaluationArtifactError(
                f"{role} seed {run.seed} is not a complete accepted lifecycle evaluation"
            )
    if role in {"pure_fsm", "checkpoint_initial"}:
        if any(
            row.get("nonzero") is not False
            or float(row.get("normalized_residual_rms", float("nan"))) != 0.0
            or float(row.get("normalized_residual_peak", float("nan"))) != 0.0
            or int(row.get("active_channel_count", -1)) != 0
            for run in selected
            for row in run.residual_activity_rows
        ):
            raise EvaluationArtifactError(
                f"{role} evaluation contains nonzero residual activity"
            )
        if any(
            any(value != 0.0 for value in sample.residual_full12)
            for run in selected
            for sample in run.stability_samples
        ):
            raise EvaluationArtifactError(
                f"{role} evaluation contains a nonzero projected residual sample"
            )
    return selected


def export_final_lifecycle_evaluation_artifacts(
    output_directory: str | Path,
    *,
    pure_fsm_aggregate: str | Path,
    checkpoint_initial_aggregate: str | Path,
    checkpoint_smoke_aggregate: str | Path,
    checkpoint_best_aggregate: str | Path,
    checkpoint_improved_aggregate: str | Path,
    frozen_hashes_unchanged: bool,
    residual_calibration_evidence: (
        VersionedResidualActivityCalibration | None
    ) = None,
    reward_stream_filename: str = DEFAULT_REWARD_STREAM_FILENAME,
    evaluation_options: Mapping[str, Any] | None = None,
) -> EvaluationArtifactPaths:
    """Publish the strict five-role final checkpoint lifecycle comparison.

    This is deliberately separate from :func:`export_paired_evaluation_artifacts`,
    whose two-role output remains the candidate-selection/promotion artifact.
    Every aggregate, worker result, checkpoint, and checkpoint manifest is
    revalidated before the canonical episodes are parsed.  Only pure FSM and
    ``checkpoint_improved`` feed the paired phase comparison; residual and
    reward exports are likewise tied exclusively to ``checkpoint_improved``.
    """

    aggregate_arguments = {
        "pure_fsm": pure_fsm_aggregate,
        "checkpoint_initial": checkpoint_initial_aggregate,
        "checkpoint_smoke": checkpoint_smoke_aggregate,
        "checkpoint_best": checkpoint_best_aggregate,
        "checkpoint_improved": checkpoint_improved_aggregate,
    }
    if tuple(aggregate_arguments) != FINAL_LIFECYCLE_ROLES:
        raise EvaluationArtifactError("final lifecycle aggregate order changed")
    if frozen_hashes_unchanged is not True:
        raise EvaluationArtifactError(
            "final lifecycle export requires an explicit passing frozen-hash gate"
        )
    aggregate_paths = tuple(
        _resolved_path(value, label=f"{role} aggregate")
        for role, value in aggregate_arguments.items()
    )
    if len(set(aggregate_paths)) != len(FINAL_LIFECYCLE_ROLES):
        raise EvaluationArtifactError("final lifecycle aggregate paths must be distinct")
    evidence_by_role = {
        role: _final_lifecycle_aggregate_evidence(path, role=role)
        for role, path in zip(FINAL_LIFECYCLE_ROLES, aggregate_paths, strict=True)
    }
    runtime_identities = tuple(
        evidence.committed_runtime_identity for evidence in evidence_by_role.values()
    )
    if any(
        dict(identity) != dict(runtime_identities[0])
        for identity in runtime_identities[1:]
    ):
        raise EvaluationArtifactError(
            "final lifecycle roles were produced by different committed runtimes"
        )
    _validate_current_committed_runtime_identity(runtime_identities[0])
    directory = _require_output_disjoint_from_final_sources(
        output_directory, evidence_by_role
    )
    checkpoint_paths = tuple(
        evidence.checkpoint_path
        for evidence in evidence_by_role.values()
        if evidence.checkpoint_path is not None
    )
    manifest_paths = tuple(
        evidence.checkpoint_manifest_path
        for evidence in evidence_by_role.values()
        if evidence.checkpoint_manifest_path is not None
    )
    if len(checkpoint_paths) != 4 or len(set(checkpoint_paths)) != 4:
        raise EvaluationArtifactError("final lifecycle checkpoints must be four distinct files")
    if len(manifest_paths) != 4 or len(set(manifest_paths)) != 4:
        raise EvaluationArtifactError(
            "final lifecycle checkpoint manifests must be four distinct files"
        )
    training_seeds = tuple(
        evidence.training_seed
        for role, evidence in evidence_by_role.items()
        if role != "pure_fsm"
    )
    if len(set(training_seeds)) != 1 or training_seeds[0] is None:
        raise EvaluationArtifactError(
            "all PPO lifecycle checkpoints must share one valid training seed"
        )
    canonical_dirs = tuple(
        directory
        for evidence in evidence_by_role.values()
        for directory in evidence.canonical_episode_dirs
    )
    if len(canonical_dirs) != 25 or len(set(canonical_dirs)) != 25:
        raise EvaluationArtifactError(
            "final lifecycle evidence requires 25 distinct canonical episode directories"
        )
    if (
        evidence_by_role["checkpoint_best"].checkpoint_sha256
        != evidence_by_role["checkpoint_improved"].checkpoint_sha256
    ):
        raise EvaluationArtifactError(
            "checkpoint_improved bytes must equal the promoted checkpoint_best bytes"
        )

    calibration = residual_calibration_evidence or (
        build_versioned_residual_activity_calibration()
    )
    runs_by_role: dict[str, tuple[LiveRunEvaluation, ...]] = {}
    for role in FINAL_LIFECYCLE_ROLES:
        evidence = evidence_by_role[role]
        runs_by_role[role] = _require_final_lifecycle_runs(
            role,
            evaluate_canonical_episode_dirs(
                evidence.canonical_episode_dirs,
                seeds=evidence.seeds,
                residual_calibration=calibration,
                reward_stream_filename=reward_stream_filename,
                require_reward_stream=True,
                require_complete_phase_sequence=(
                    role in _FINAL_LIFECYCLE_REQUIRED_PASS_ROLES
                ),
                evaluation_options=evaluation_options,
            ),
        )

    baseline = runs_by_role["pure_fsm"]
    improved = runs_by_role["checkpoint_improved"]
    comparison = paired_baseline_candidate_promotion(
        baseline,
        improved,
        frozen_hashes_unchanged=bool(frozen_hashes_unchanged),
        minimum_paired_seeds=len(BASELINE_VALIDATION_SEEDS),
    )
    if not comparison.promotion.promoted:
        raise EvaluationArtifactError(
            "checkpoint_improved does not pass the authoritative paired promotion gates: "
            f"{comparison.promotion.first_failed_gate}"
        )
    improved_evidence = evidence_by_role["checkpoint_improved"]
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

    baseline_episode = _normalized_rows(_episode_rows(baseline, "pure_fsm"))
    baseline_phase = _normalized_rows(_phase_rows(baseline, "pure_fsm"))
    improved_episode = _normalized_rows(
        _episode_rows(improved, "checkpoint_improved")
    )
    improved_phase = _normalized_rows(_phase_rows(improved, "checkpoint_improved"))
    residual = _normalized_rows(
        _candidate_residual_rows(improved, "checkpoint_improved")
    )
    reward = _normalized_rows(
        _candidate_reward_rows(improved, "checkpoint_improved")
    )
    termination = _normalized_rows(_final_lifecycle_termination_rows(runs_by_role))
    phase_comparison = _normalized_rows(_final_lifecycle_phase_rows(comparison))
    checkpoint_comparison = _normalized_rows(
        _final_lifecycle_checkpoint_row(
            evidence=evidence_by_role[role],
            runs=runs_by_role[role],
            comparison=comparison,
        )
        for role in FINAL_LIFECYCLE_ROLES
    )

    decision = comparison.promotion
    lifecycle_records = {
        role: {
            "aggregate_path": str(evidence.aggregate_path),
            "aggregate_sha256": evidence.aggregate_sha256,
            "checkpoint_path": (
                str(evidence.checkpoint_path)
                if evidence.checkpoint_path is not None
                else None
            ),
            "checkpoint_sha256": evidence.checkpoint_sha256,
            "checkpoint_manifest_path": (
                str(evidence.checkpoint_manifest_path)
                if evidence.checkpoint_manifest_path is not None
                else None
            ),
            "checkpoint_manifest_sha256": evidence.checkpoint_manifest_sha256,
            "training_seed": evidence.training_seed,
            "source_git_commit": evidence.source_git_commit,
            "committed_runtime_content_sha256": (
                evidence.committed_runtime_content_sha256
            ),
            "creation_runtime_identity_sha256": (
                evidence.creation_runtime_identity_sha256
            ),
            "creation_runtime_identity_path": (
                None
                if evidence.creation_runtime_identity_path is None
                else str(evidence.creation_runtime_identity_path)
            ),
            "paired_seeds": list(evidence.seeds),
            "canonical_episode_dirs": [
                str(path) for path in evidence.canonical_episode_dirs
            ],
            "source_groups": [dict(group) for group in evidence.source_groups],
            "supporting_files": [
                dict(record) for record in evidence.supporting_files
            ],
            "committed_runtime_identity": dict(
                evidence.committed_runtime_identity
            ),
        }
        for role, evidence in evidence_by_role.items()
    }
    decision_payload: dict[str, Any] = {
        "schema": EVALUATION_ARTIFACT_SCHEMA,
        "bundle_kind": FINAL_LIFECYCLE_BUNDLE_KIND,
        "final_lifecycle_roles": list(FINAL_LIFECYCLE_ROLES),
        "baseline_checkpoint": "pure_fsm",
        "candidate_checkpoint": "checkpoint_improved",
        "candidate_checkpoint_path": str(improved_evidence.checkpoint_path),
        "candidate_checkpoint_sha256": improved_evidence.checkpoint_sha256,
        "candidate_checkpoint_manifest_path": str(
            improved_evidence.checkpoint_manifest_path
        ),
        "candidate_checkpoint_manifest_sha256": (
            improved_evidence.checkpoint_manifest_sha256
        ),
        "paired_seeds": list(BASELINE_VALIDATION_SEEDS),
        "paired_episode_count": len(BASELINE_VALIDATION_SEEDS),
        "minimum_paired_seeds": len(BASELINE_VALIDATION_SEEDS),
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
        "final_lifecycle_evidence": lifecycle_records,
        "artifacts": paths.as_dict(),
        "residual_activity_calibration": calibration.as_dict(),
    }

    _require_no_reparse_components(directory, label="final lifecycle output directory")
    _revalidate_final_lifecycle_sources(evidence_by_role)

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
            improved_episode,
            _fieldnames(
                improved_episode,
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
            improved_phase,
            _fieldnames(
                improved_phase,
                preferred=("checkpoint", "seed", "trial_id", "run_directory", "phase"),
            ),
        ),
        _CsvPublication(
            paths.checkpoint_comparison,
            checkpoint_comparison,
            _fieldnames(
                checkpoint_comparison,
                preferred=(
                    "role",
                    "checkpoint",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "checkpoint_manifest_path",
                    "checkpoint_manifest_sha256",
                    "evaluation_aggregate_path",
                    "evaluation_aggregate_sha256",
                ),
            ),
        ),
        _CsvPublication(
            paths.phase_metric_comparison,
            phase_comparison,
            _fieldnames(
                phase_comparison,
                preferred=("phase", "baseline_checkpoint", "improved_checkpoint"),
            ),
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
    decision_payload["artifact_files"] = {
        publication.path.name: {
            "path": str(publication.path),
            "bytes": len(publication.content),
            "sha256": hashlib.sha256(publication.content).hexdigest(),
        }
        for publication in csv_specs
    }
    _publish_idempotently(
        csv_specs,
        (_JsonPublication(paths.promotion_decision, decision_payload),),
    )
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
    "FINAL_LIFECYCLE_BUNDLE_KIND",
    "FINAL_LIFECYCLE_ROLES",
    "FinalLifecycleAggregateEvidence",
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
    "export_final_lifecycle_evaluation_artifacts",
    "export_paired_evaluation_artifacts",
    "validate_final_lifecycle_aggregate_evidence",
]
