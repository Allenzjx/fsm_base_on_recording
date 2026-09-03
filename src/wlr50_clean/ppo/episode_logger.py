"""Validated PPO transition records with JSONL and Parquet serializers."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TRANSITION_SCHEMA = "wlr50_clean.ppo_transition.v1"
DATASET_MANIFEST_SCHEMA = "wlr50_clean.ppo_baseline_dataset_manifest.v1"


class EpisodeLogError(ValueError):
    pass


def _float_tuple(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise EpisodeLogError(f"{label} must be numeric") from exc
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise EpisodeLogError(f"{label} must contain {size} finite values")
    return result


def _mask(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != 12 or any(value not in (0, 1) for value in result):
        raise EpisodeLogError("action_mask_t must contain twelve binary values")
    return result


def _bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f">{len(values)}d", *values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    episode_id: str
    trial_id: str
    seed: int
    control_tick: int
    sim_time: float
    state_id: str
    macro_phase: int
    phase_progress: float
    observation_t: tuple[float, ...]
    nominal_action_t: tuple[float, ...]
    residual_action_t: tuple[float, ...]
    applied_action_t: tuple[float, ...]
    action_mask_t: tuple[int, ...]
    task_result: str
    reward_components_t: Mapping[str, float]
    terminated: bool
    truncated: bool
    termination_reason: str | None
    observation_t_plus_1: tuple[float, ...]
    environment_hash: str
    controller_hash: str
    motion_contract_hash: str
    observation_schema_version: str
    action_schema_version: str
    schema: str = TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRANSITION_SCHEMA:
            raise EpisodeLogError("unexpected transition schema")
        if not self.episode_id or not self.trial_id or self.seed < 0 or self.control_tick < 0:
            raise EpisodeLogError("episode identity is invalid")
        if not self.task_result:
            raise EpisodeLogError("task_result is required")
        if self.state_id not in tuple(f"P{index:02d}" for index in range(1, 14)):
            raise EpisodeLogError("transition state_id is invalid")
        if self.macro_phase != int(self.state_id[1:]):
            raise EpisodeLogError("transition state_id and macro_phase disagree")
        sim_time = float(self.sim_time)
        progress = float(self.phase_progress)
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise EpisodeLogError("sim_time must be finite and non-negative")
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise EpisodeLogError("phase_progress must be within [0,1]")
        object.__setattr__(self, "sim_time", sim_time)
        object.__setattr__(self, "phase_progress", progress)
        for name, size in (
            ("observation_t", 85),
            ("nominal_action_t", 12),
            ("residual_action_t", 12),
            ("applied_action_t", 12),
            ("observation_t_plus_1", 85),
        ):
            object.__setattr__(
                self, name, _float_tuple(getattr(self, name), size, name)
            )
        object.__setattr__(self, "action_mask_t", _mask(self.action_mask_t))
        components = {
            str(name): float(value) for name, value in self.reward_components_t.items()
        }
        if not components or any(not math.isfinite(value) for value in components.values()):
            raise EpisodeLogError("reward_components_t must be finite and non-empty")
        object.__setattr__(self, "reward_components_t", components)
        if self.terminated and self.truncated:
            raise EpisodeLogError("transition cannot terminate and truncate together")
        if (self.terminated or self.truncated) != (self.termination_reason is not None):
            raise EpisodeLogError("done flags and termination_reason disagree")
        for name in (
            "environment_hash",
            "controller_hash",
            "motion_contract_hash",
            "observation_schema_version",
            "action_schema_version",
        ):
            if not getattr(self, name):
                raise EpisodeLogError(f"{name} is required")

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def as_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "observation_t",
            "nominal_action_t",
            "residual_action_t",
            "applied_action_t",
            "action_mask_t",
            "observation_t_plus_1",
        ):
            payload[name] = list(payload[name])
        payload["reward_components_t"] = dict(self.reward_components_t)
        return payload


@dataclass(frozen=True, slots=True)
class BaselineEquivalenceEvidence:
    status: str
    transition_count: int
    residual_action_all_zero: bool
    applied_action_bitwise_equal_nominal: bool
    nominal_sequence_sha256: str
    applied_sequence_sha256: str


class EpisodeLogger:
    """In-memory validation first; serialization never mutates source Trials."""

    def __init__(self) -> None:
        self._rows: list[EpisodeTransition] = []

    @property
    def rows(self) -> tuple[EpisodeTransition, ...]:
        return tuple(self._rows)

    def append(self, transition: EpisodeTransition) -> None:
        if self._rows:
            previous = self._rows[-1]
            if previous.done:
                raise EpisodeLogError("cannot append after the final transition")
            if transition.episode_id != previous.episode_id:
                raise EpisodeLogError("one logger may contain only one episode")
            if transition.trial_id != previous.trial_id:
                raise EpisodeLogError("one logger may contain only one Trial")
            if transition.control_tick != previous.control_tick + 1:
                raise EpisodeLogError("control_tick is not contiguous")
            if transition.sim_time <= previous.sim_time:
                raise EpisodeLogError("sim_time is not strictly increasing")
            if _bytes(previous.observation_t_plus_1) != _bytes(
                transition.observation_t
            ):
                raise EpisodeLogError("dataset observation continuity failed")
            for name in (
                "seed",
                "environment_hash",
                "controller_hash",
                "motion_contract_hash",
                "observation_schema_version",
                "action_schema_version",
            ):
                if getattr(transition, name) != getattr(previous, name):
                    raise EpisodeLogError(f"episode field {name} changed")
        elif transition.control_tick != 0:
            raise EpisodeLogError("first control_tick must be zero")
        self._rows.append(transition)

    def validate_complete(self) -> None:
        if not self._rows:
            raise EpisodeLogError("episode dataset is empty")
        if not self._rows[-1].done:
            raise EpisodeLogError("final transition does not have a done flag")
        if any(row.done for row in self._rows[:-1]):
            raise EpisodeLogError("a non-final transition has a done flag")

    def validate_baseline_equivalence(self) -> BaselineEquivalenceEvidence:
        """Derive baseline claims from rows; never trust a caller assertion."""

        self.validate_complete()
        nominal_digest = hashlib.sha256()
        applied_digest = hashlib.sha256()
        residual_zero = True
        action_equal = True
        for row in self._rows:
            residual_zero = residual_zero and all(
                value == 0.0 for value in row.residual_action_t
            )
            nominal = _bytes(row.nominal_action_t)
            applied = _bytes(row.applied_action_t)
            nominal_digest.update(nominal)
            applied_digest.update(applied)
            action_equal = action_equal and nominal == applied
        evidence = BaselineEquivalenceEvidence(
            status=(
                "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE"
                if residual_zero and action_equal
                else "BASELINE_EQUIVALENCE_REJECTED"
            ),
            transition_count=len(self._rows),
            residual_action_all_zero=residual_zero,
            applied_action_bitwise_equal_nominal=action_equal,
            nominal_sequence_sha256=nominal_digest.hexdigest(),
            applied_sequence_sha256=applied_digest.hexdigest(),
        )
        if not residual_zero:
            raise EpisodeLogError(
                "baseline dataset contains a non-zero residual_action_t"
            )
        if not action_equal:
            raise EpisodeLogError(
                "baseline applied_action_t is not bitwise equal to nominal_action_t"
            )
        return evidence

    def write_jsonl(self, path: Path, *, require_complete: bool = True) -> Path:
        if require_complete:
            self.validate_complete()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in self._rows:
                stream.write(json.dumps(row.as_json_dict(), separators=(",", ":")) + "\n")
        return path

    def write_parquet(self, path: Path, *, require_complete: bool = True) -> Path:
        if require_complete:
            self.validate_complete()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # dependency is intentionally lazy for Isaac runtime
            raise RuntimeError(
                "Parquet export requires the optional pyarrow PPO artifact dependency"
            ) from exc
        reward_names = tuple(sorted(self._rows[0].reward_components_t))
        schema = pa.schema(
            [
                pa.field("schema", pa.string(), nullable=False),
                pa.field("episode_id", pa.string(), nullable=False),
                pa.field("trial_id", pa.string(), nullable=False),
                pa.field("seed", pa.int64(), nullable=False),
                pa.field("control_tick", pa.int64(), nullable=False),
                pa.field("sim_time", pa.float64(), nullable=False),
                pa.field("state_id", pa.string(), nullable=False),
                pa.field("macro_phase", pa.int16(), nullable=False),
                pa.field("phase_progress", pa.float64(), nullable=False),
                pa.field("observation_t", pa.list_(pa.float64(), 85), nullable=False),
                pa.field("nominal_action_t", pa.list_(pa.float64(), 12), nullable=False),
                pa.field("residual_action_t", pa.list_(pa.float64(), 12), nullable=False),
                pa.field("applied_action_t", pa.list_(pa.float64(), 12), nullable=False),
                pa.field("action_mask_t", pa.list_(pa.int8(), 12), nullable=False),
                pa.field("task_result", pa.string(), nullable=False),
                pa.field(
                    "reward_components_t",
                    pa.struct(
                        [pa.field(name, pa.float64(), nullable=False) for name in reward_names]
                    ),
                    nullable=False,
                ),
                pa.field("terminated", pa.bool_(), nullable=False),
                pa.field("truncated", pa.bool_(), nullable=False),
                pa.field("termination_reason", pa.string()),
                pa.field(
                    "observation_t_plus_1", pa.list_(pa.float64(), 85), nullable=False
                ),
                pa.field("environment_hash", pa.string(), nullable=False),
                pa.field("controller_hash", pa.string(), nullable=False),
                pa.field("motion_contract_hash", pa.string(), nullable=False),
                pa.field("observation_schema_version", pa.string(), nullable=False),
                pa.field("action_schema_version", pa.string(), nullable=False),
            ]
        )
        records = []
        for row in self._rows:
            record = row.as_json_dict()
            if tuple(sorted(record["reward_components_t"])) != reward_names:
                raise EpisodeLogError("reward component schema changed within episode")
            record["reward_components_t"] = {
                name: record["reward_components_t"][name] for name in reward_names
            }
            records.append(record)
        table = pa.Table.from_pylist(records, schema=schema)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd", version="2.6")
        return path

    def write_manifest(
        self,
        path: Path,
        *,
        source_trial: str,
        jsonl_path: Path,
        parquet_path: Path,
        claimed_zero_residual_equivalence_sha256: str | None = None,
    ) -> Path:
        evidence = self.validate_baseline_equivalence()
        if (
            claimed_zero_residual_equivalence_sha256 is not None
            and claimed_zero_residual_equivalence_sha256
            != evidence.nominal_sequence_sha256
        ):
            raise EpisodeLogError(
                "caller-asserted zero-residual equivalence hash disagrees with rows"
            )
        payload = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "source_trial": source_trial,
            "transition_count": len(self._rows),
            "residual_action": {
                "classification": "all_zero_baseline_demonstration",
                "all_zero_verified": evidence.residual_action_all_zero,
            },
            "ppo_training_started": False,
            "zero_residual_full_episode_equivalence": {
                "status": "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE",
                "transition_count": evidence.transition_count,
                "applied_action_bitwise_equal_nominal": (
                    evidence.applied_action_bitwise_equal_nominal
                ),
                "nominal_sequence_sha256": evidence.nominal_sequence_sha256,
                "applied_sequence_sha256": evidence.applied_sequence_sha256,
            },
            "files": {
                "jsonl": {
                    "path": jsonl_path.name,
                    "bytes": jsonl_path.stat().st_size,
                    "sha256": _sha256(jsonl_path),
                },
                "parquet": {
                    "path": parquet_path.name,
                    "bytes": parquet_path.stat().st_size,
                    "sha256": _sha256(parquet_path),
                },
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
        return path
