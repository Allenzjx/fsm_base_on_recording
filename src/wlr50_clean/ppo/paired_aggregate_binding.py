"""Strict immutable binding for paired validation aggregates.

This module validates the aggregate lifecycle, reconstructs its five-worker
batch, snapshots every aggregate/worker/canonical source, and offers an
explicit revalidation hook around metric export.  Physical candidate failure
remains valid evidence; the official baseline aggregate must pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .evaluation_artifacts import (
    CANONICAL_EPISODE_FILES,
    NONEMPTY_CANONICAL_EPISODE_FILES,
    EvaluationArtifactError,
    FreshProcessEpisodeBatch,
    collect_fresh_process_episode_workers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "wlr50_clean.validation_aggregate_binding.v1"
RUN_SCHEMA = "wlr50_clean.ppo_run_manifest.v1"
AGGREGATE_SCHEMA = "wlr50_clean.fresh_process_episode_batch.v1"
VALIDATION_SEEDS = (2001, 2002, 2003, 2004, 2005)
_REPARSE = 0x400


class PairedAggregateBindingError(RuntimeError):
    """Raised when a baseline/candidate aggregate can be spliced or mutated."""


@dataclass(frozen=True, slots=True)
class _Identity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    attributes: int
    links: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_Identity":
        return cls(
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(getattr(value, "st_file_attributes", 0)),
            int(value.st_nlink),
        )


@dataclass(frozen=True, slots=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    identity: _Identity

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": len(self.data),
            "sha256": self.sha256,
        }


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_redirect(path: Path, *, root: Path, label: str) -> Path:
    selected = _absolute(path)
    try:
        relative = selected.relative_to(_absolute(root))
    except ValueError as exc:
        raise PairedAggregateBindingError(f"{label} escapes {root}") from exc
    cursor = _absolute(root)
    for part in relative.parts:
        cursor /= part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise PairedAggregateBindingError(f"cannot inspect {label}: {cursor}") from exc
        junction = getattr(cursor, "is_junction", None)
        if (
            cursor.is_symlink()
            or (callable(junction) and junction())
            or int(getattr(metadata, "st_file_attributes", 0)) & _REPARSE
        ):
            raise PairedAggregateBindingError(
                f"{label} contains a symlink/reparse point: {cursor}"
            )
    if selected.resolve() != selected:
        raise PairedAggregateBindingError(f"{label} is redirected")
    return selected


def _capture(path: Path, *, root: Path, label: str) -> _Snapshot:
    selected = _reject_redirect(path, root=root, label=label)
    try:
        lexical_before = selected.lstat()
        with selected.open("rb") as stream:
            before = os.fstat(stream.fileno())
            data = stream.read()
            after = os.fstat(stream.fileno())
        lexical_after = selected.lstat()
    except OSError as exc:
        raise PairedAggregateBindingError(f"cannot capture {label}: {selected}") from exc
    identities = tuple(
        _Identity.from_stat(value)
        for value in (lexical_before, before, after, lexical_after)
    )
    if any(value != identities[0] for value in identities[1:]):
        raise PairedAggregateBindingError(f"{label} changed while captured")
    identity = identities[0]
    if (
        not stat.S_ISREG(identity.mode)
        or identity.attributes & _REPARSE
        or identity.size != len(data)
    ):
        raise PairedAggregateBindingError(f"{label} is not one regular file")
    return _Snapshot(selected, data, hashlib.sha256(data).hexdigest(), identity)


def _json(snapshot: _Snapshot, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PairedAggregateBindingError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise PairedAggregateBindingError(f"{label} must be a JSON object")
    return payload


def _record_matches(
    record: Any, snapshot: _Snapshot, *, relative: str, label: str
) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("path") != relative
        or record.get("bytes") != len(snapshot.data)
        or record.get("sha256") != snapshot.sha256
    ):
        raise PairedAggregateBindingError(f"{label} digest record is stale")


def _canonical_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CapturedValidationAggregate:
    role: str
    aggregate_path: Path
    aggregate_payload: Mapping[str, Any]
    batch: FreshProcessEpisodeBatch
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    checkpoint_manifest_path: Path | None
    checkpoint_manifest_sha256: str | None
    source_snapshots: tuple[_Snapshot, ...]
    project_root: Path

    @property
    def source_file_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            snapshot.record()
            for snapshot in sorted(self.source_snapshots, key=lambda value: str(value.path))
        )

    def as_record(self) -> dict[str, Any]:
        records = self.source_file_records
        result = {
            "schema": SCHEMA,
            "path": str(self.aggregate_path),
            "bytes": len(next(
                value.data for value in self.source_snapshots if value.path == self.aggregate_path
            )),
            "sha256": next(
                value.sha256 for value in self.source_snapshots if value.path == self.aggregate_path
            ),
            "role": self.role,
            "physical_passed": self.aggregate_payload.get("passed") is True,
            "seeds": list(self.batch.seeds),
            "worker_run_dirs": [str(Path(row["run_dir"])) for row in self.batch.worker_rows],
            "canonical_episode_dirs": [str(path) for path in self.batch.canonical_episode_dirs],
            "source_file_records": list(records),
            "source_file_set_sha256": _canonical_hash(records),
        }
        if self.role == "candidate":
            result["checkpoint_path"] = str(self.checkpoint_path)
            result["checkpoint_sha256"] = self.checkpoint_sha256
            result["checkpoint_manifest_path"] = str(self.checkpoint_manifest_path)
            result["checkpoint_manifest_sha256"] = self.checkpoint_manifest_sha256
        return result

    def assert_unchanged(self) -> None:
        for expected in self.source_snapshots:
            current = _capture(
                expected.path,
                root=self.project_root,
                label="paired aggregate source",
            )
            if current.identity != expected.identity or current.sha256 != expected.sha256:
                raise PairedAggregateBindingError(
                    f"paired aggregate source changed: {expected.path}"
                )


def capture_validation_aggregate(
    aggregate_path: Path | str,
    *,
    role: str,
    expected_checkpoint_path: Path | str | None = None,
    expected_checkpoint_manifest_path: Path | str | None = None,
    project_root: Path | str = PROJECT_ROOT,
) -> CapturedValidationAggregate:
    """Capture one exact baseline/candidate validation aggregate and all sources."""

    selected_role = str(role).strip().lower()
    if selected_role not in {"baseline", "candidate"}:
        raise PairedAggregateBindingError("aggregate role must be baseline or candidate")
    root = _absolute(project_root)
    path = _reject_redirect(_absolute(aggregate_path), root=root, label="aggregate")
    expected_name = (
        "fsm_baseline_evaluation_aggregate.json"
        if selected_role == "baseline"
        else "checkpoint_evaluation_aggregate.json"
    )
    expected_kind = (
        "baseline-fsm-eval-batch"
        if selected_role == "baseline"
        else "validation-checkpoint-evaluation-batch"
    )
    runs_root = root / "runs" / "ppo_phase_v1" / expected_kind
    try:
        relative = path.relative_to(runs_root)
    except ValueError as exc:
        raise PairedAggregateBindingError(
            f"{selected_role} aggregate is outside its managed runs root"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[1] != expected_name:
        raise PairedAggregateBindingError(
            f"{selected_role} aggregate has the wrong managed path"
        )
    run_dir = path.parent
    snapshots: dict[Path, _Snapshot] = {}

    def capture(value: Path, label: str) -> _Snapshot:
        selected = _absolute(value)
        if selected not in snapshots:
            snapshots[selected] = _capture(selected, root=root, label=label)
        return snapshots[selected]

    aggregate_snapshot = capture(path, "validation aggregate")
    aggregate = _json(aggregate_snapshot, label="validation aggregate")
    manifest_snapshot = capture(run_dir / "run_manifest.json", "aggregate run manifest")
    manifest = _json(manifest_snapshot, label="aggregate run manifest")
    started_snapshot = capture(
        run_dir / "run_manifest.started.json", "aggregate started manifest"
    )
    started = _json(started_snapshot, label="aggregate started manifest")
    identity = manifest.get("identity")
    if (
        manifest.get("schema") != RUN_SCHEMA
        or manifest.get("lifecycle") != "SUCCEEDED"
        or manifest.get("exit_code") != 0
        or manifest.get("immutable_run_directory") is not True
        or manifest.get("run_kind") != expected_kind
        or manifest.get("entrypoint") != "wlr50_clean.ppo.cli"
        or manifest.get("subcommand") != "aggregate-evaluations"
        or _absolute(str(manifest.get("project_root", ""))) != root
        or _absolute(str(manifest.get("run_dir", ""))) != run_dir
        or started.get("schema") != RUN_SCHEMA
        or started.get("lifecycle") != "STARTED"
        or not isinstance(identity, Mapping)
        or identity.get("seed") != 2001
        or identity.get("environment_count") != 1
    ):
        raise PairedAggregateBindingError("aggregate finalized lifecycle is invalid")
    for key, value in started.items():
        if key != "lifecycle" and manifest.get(key) != value:
            raise PairedAggregateBindingError(
                f"aggregate final manifest changed started field {key!r}"
            )
    _record_matches(
        manifest.get("started_manifest"),
        started_snapshot,
        relative="run_manifest.started.json",
        label="aggregate started manifest",
    )
    logs = manifest.get("logs")
    artifacts = manifest.get("artifacts")
    if not isinstance(logs, Mapping) or not isinstance(artifacts, Mapping):
        raise PairedAggregateBindingError("aggregate log/artifact maps are missing")
    for relative_name, record in (*logs.items(), *artifacts.items()):
        if not isinstance(relative_name, str) or ".." in Path(relative_name).parts:
            raise PairedAggregateBindingError("aggregate artifact path is unsafe")
        source = capture(run_dir / relative_name, f"aggregate source {relative_name}")
        _record_matches(record, source, relative=relative_name, label=relative_name)
    if expected_name not in artifacts:
        raise PairedAggregateBindingError("aggregate run does not bind its aggregate JSON")
    for name in (
        "frozen_hashes.before.json",
        "frozen_hashes.after.json",
        "committed_runtime_identity.before.json",
        "committed_runtime_identity.after.json",
    ):
        if name not in artifacts:
            raise PairedAggregateBindingError(f"aggregate run omits {name}")
    runtime_before = _json(
        snapshots[run_dir / "committed_runtime_identity.before.json"],
        label="runtime before",
    )
    runtime_after = _json(
        snapshots[run_dir / "committed_runtime_identity.after.json"],
        label="runtime after",
    )
    if runtime_before != runtime_after or runtime_before.get("git_commit") != identity.get(
        "git_commit"
    ):
        raise PairedAggregateBindingError("aggregate runtime identity changed")
    for name in ("frozen_hashes.before.json", "frozen_hashes.after.json"):
        audit = _json(snapshots[run_dir / name], label=name)
        if audit.get("passed") is not True or audit.get("mismatches") != []:
            raise PairedAggregateBindingError(f"aggregate {name} failed")

    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("role") != selected_role
        or aggregate.get("seed_set") != "validation"
        or aggregate.get("seeds") != list(VALIDATION_SEEDS)
        or aggregate.get("episode_count") != 5
        or aggregate.get("fresh_process_per_episode") is not True
        or aggregate.get("deterministic_evaluation") is not True
        or not isinstance(aggregate.get("passed"), bool)
    ):
        raise PairedAggregateBindingError("validation aggregate header is invalid")
    if selected_role == "baseline" and aggregate.get("passed") is not True:
        raise PairedAggregateBindingError("official baseline aggregate did not pass")
    workers = aggregate.get("workers")
    episodes = aggregate.get("episodes")
    raw_episode_dirs = aggregate.get("canonical_episode_dirs")
    if (
        not isinstance(workers, list)
        or len(workers) != 5
        or any(not isinstance(row, Mapping) for row in workers)
        or not isinstance(episodes, list)
        or len(episodes) != 5
        or any(not isinstance(row, Mapping) for row in episodes)
        or not isinstance(raw_episode_dirs, list)
        or len(raw_episode_dirs) != 5
    ):
        raise PairedAggregateBindingError("validation aggregate worker list is invalid")
    worker_kind = (
        "baseline_fsm_eval"
        if selected_role == "baseline"
        else "validation-checkpoint-evaluation"
    )
    worker_stage = (
        "baseline-fsm-eval-fresh-process"
        if selected_role == "baseline"
        else "checkpoint-evaluation-validation-fresh-process"
    )
    worker_subcommand = "baseline-eval" if selected_role == "baseline" else "evaluate"
    worker_root = root / "runs" / "ppo_phase_v1" / worker_kind
    worker_dirs: list[Path] = []
    declared_episode_dirs: list[Path] = []
    worker_manifests: list[Mapping[str, Any]] = []
    for index, (row, episode_row, raw_episode_dir) in enumerate(
        zip(workers, episodes, raw_episode_dirs, strict=True)
    ):
        worker_dir = _reject_redirect(
            _absolute(str(row.get("run_dir", ""))),
            root=root,
            label=f"{selected_role} worker {index}",
        )
        try:
            worker_relative = worker_dir.relative_to(worker_root)
        except ValueError as exc:
            raise PairedAggregateBindingError(
                f"{selected_role} worker {index} is outside its managed runs root"
            ) from exc
        if len(worker_relative.parts) != 1:
            raise PairedAggregateBindingError(
                f"{selected_role} worker {index} has the wrong managed path"
            )
        episode_dir = _reject_redirect(
            _absolute(str(raw_episode_dir)),
            root=root,
            label=f"{selected_role} canonical episode {index}",
        )
        if (
            episode_dir.parent != worker_dir
            or _absolute(str(row.get("canonical_episode_dir", ""))) != episode_dir
            or row.get("seed") != VALIDATION_SEEDS[index]
            or episode_row.get("seed") != VALIDATION_SEEDS[index]
        ):
            raise PairedAggregateBindingError(
                f"{selected_role} worker/episode {index} path or seed is inconsistent"
            )

        # Snapshot every worker and canonical source *before* any helper is
        # allowed to parse it.  The same native file identity is then checked
        # after reconstruction and again around metric publication.
        worker_manifest_snapshot = capture(
            worker_dir / "run_manifest.json", "worker run manifest"
        )
        worker_manifest = _json(worker_manifest_snapshot, label="worker run manifest")
        worker_started = capture(
            worker_dir / "run_manifest.started.json", "worker started manifest"
        )
        _record_matches(
            worker_manifest.get("started_manifest"),
            worker_started,
            relative="run_manifest.started.json",
            label="worker started manifest",
        )
        for group_name in ("logs", "artifacts"):
            group = worker_manifest.get(group_name)
            if not isinstance(group, Mapping):
                raise PairedAggregateBindingError("worker source map is missing")
            for relative_name, record in group.items():
                if (
                    not isinstance(relative_name, str)
                    or not relative_name
                    or Path(relative_name).is_absolute()
                    or ".." in Path(relative_name).parts
                ):
                    raise PairedAggregateBindingError("worker source path is unsafe")
                source = capture(worker_dir / relative_name, "worker source")
                _record_matches(
                    record,
                    source,
                    relative=relative_name,
                    label="worker source",
                )
        for filename in CANONICAL_EPISODE_FILES:
            source = capture(episode_dir / filename, "canonical episode source")
            if filename in NONEMPTY_CANONICAL_EPISODE_FILES and not source.data:
                raise PairedAggregateBindingError(
                    f"canonical episode source is empty: {episode_dir / filename}"
                )
        worker_result_path = _reject_redirect(
            _absolute(str(row.get("worker_result", ""))),
            root=root,
            label=f"{selected_role} worker {index} result",
        )
        if worker_result_path.parent != worker_dir:
            raise PairedAggregateBindingError("worker result escaped its managed run")
        worker_result_snapshot = capture(worker_result_path, "worker result")
        trial_snapshot = snapshots[episode_dir / "trial_manifest.json"]
        if (
            row.get("run_manifest_sha256") != worker_manifest_snapshot.sha256
            or row.get("worker_result_sha256") != worker_result_snapshot.sha256
            or row.get("trial_manifest_sha256") != trial_snapshot.sha256
        ):
            raise PairedAggregateBindingError(
                f"{selected_role} worker {index} digest binding is stale"
            )
        worker_dirs.append(worker_dir)
        declared_episode_dirs.append(episode_dir)
        worker_manifests.append(worker_manifest)
    if len(set(worker_dirs)) != 5 or len(set(declared_episode_dirs)) != 5:
        raise PairedAggregateBindingError("validation worker/episode paths are duplicated")

    checkpoint = (
        None
        if selected_role == "baseline"
        else _absolute(expected_checkpoint_path or str(aggregate.get("checkpoint", "")))
    )
    checkpoint_manifest: Path | None = None
    checkpoint_manifest_hash: str | None = None
    if selected_role == "baseline":
        if (
            aggregate.get("checkpoint") is not None
            or aggregate.get("checkpoint_sha256") is not None
            or aggregate.get("pure_fsm_zero_residual") is not True
            or expected_checkpoint_manifest_path is not None
        ):
            raise PairedAggregateBindingError("baseline aggregate names a checkpoint")
        checkpoint_hash = None
    else:
        if expected_checkpoint_path is None or expected_checkpoint_manifest_path is None:
            raise PairedAggregateBindingError(
                "candidate aggregate requires an explicit checkpoint and manifest"
            )
        checkpoint = _reject_redirect(
            checkpoint, root=root, label="candidate checkpoint"
        )
        checkpoint_manifest = _reject_redirect(
            _absolute(expected_checkpoint_manifest_path),
            root=root,
            label="candidate checkpoint manifest",
        )
        if not checkpoint.is_file() or not checkpoint_manifest.is_file():
            raise PairedAggregateBindingError("candidate aggregate checkpoint is missing")
        checkpoint_snapshot = capture(checkpoint, "candidate checkpoint")
        checkpoint_manifest_snapshot = capture(
            checkpoint_manifest, "candidate checkpoint manifest"
        )
        checkpoint_hash = checkpoint_snapshot.sha256
        checkpoint_manifest_hash = checkpoint_manifest_snapshot.sha256
        if (
            _absolute(str(aggregate.get("checkpoint", ""))) != checkpoint
            or aggregate.get("checkpoint_sha256") != checkpoint_hash
            or aggregate.get("deterministic_mean_policy") is not True
        ):
            raise PairedAggregateBindingError(
                "candidate aggregate names a different checkpoint"
            )

    # Reuse the central managed-run/frozen/runtime/config validator instead of
    # maintaining a weaker parallel interpretation here.  Its cache is
    # intentionally separate; our native snapshots prove that none of those
    # sources changed between the two reads.
    try:
        from .training_orchestration import (
            TrainingOrchestrationError,
            _validate_finalized_run,
        )

        strict_cache: dict[Path, Any] = {}
        _validate_finalized_run(
            run_dir,
            project_root=root,
            run_kind=expected_kind,
            training_stage=(
                "baseline-fsm-eval-aggregate"
                if selected_role == "baseline"
                else "checkpoint-evaluation-validation-aggregate"
            ),
            entrypoint="wlr50_clean.ppo.cli",
            subcommand="aggregate-evaluations",
            cache=strict_cache,
        )
        for worker_dir in worker_dirs:
            _validate_finalized_run(
                worker_dir,
                project_root=root,
                run_kind=worker_kind,
                training_stage=worker_stage,
                entrypoint="wlr50_clean.ppo.cli",
                subcommand=worker_subcommand,
                cache=strict_cache,
            )
    except (TrainingOrchestrationError, OSError, ValueError) as exc:
        raise PairedAggregateBindingError(
            f"{selected_role} aggregate managed provenance is invalid: {exc}"
        ) from exc

    for expected in tuple(snapshots.values()):
        current = _capture(expected.path, root=root, label="pre-reconstruction source")
        if current.identity != expected.identity or current.sha256 != expected.sha256:
            raise PairedAggregateBindingError(
                f"paired aggregate source changed before reconstruction: {expected.path}"
            )
    try:
        batch = collect_fresh_process_episode_workers(
            worker_dirs,
            seeds=VALIDATION_SEEDS,
            role=selected_role,
            checkpoint_path=checkpoint,
        )
    except EvaluationArtifactError as exc:
        raise PairedAggregateBindingError(str(exc)) from exc
    for expected in tuple(snapshots.values()):
        current = _capture(expected.path, root=root, label="post-reconstruction source")
        if current.identity != expected.identity or current.sha256 != expected.sha256:
            raise PairedAggregateBindingError(
                f"paired aggregate source changed during reconstruction: {expected.path}"
            )
    if (
        [dict(value) for value in batch.worker_rows] != aggregate.get("workers")
        or [dict(value) for value in batch.episode_rows] != aggregate.get("episodes")
        or [str(value) for value in batch.canonical_episode_dirs]
        != aggregate.get("canonical_episode_dirs")
    ):
        raise PairedAggregateBindingError(
            "aggregate payload differs from its reconstructed worker batch"
        )
    expected_passed = bool(
        all(row.get("task_success") is True for row in batch.episode_rows)
        and all(row.get("body_collision") is False for row in batch.episode_rows)
        and all(row.get("wheel_only_climb") is False for row in batch.episode_rows)
        and all(row.get("safety_abort") is False for row in batch.episode_rows)
        and all(row.get("under_maximum_duration") is True for row in batch.episode_rows)
        and all(int(row.get("recording_runtime_access_count", -1)) == 0 for row in batch.episode_rows)
        and all(int(row.get("in_episode_root_write_count", -1)) == 0 for row in batch.episode_rows)
        and all(row.get("worker_gate_passed") is True for row in batch.worker_rows)
    )
    if aggregate.get("passed") is not expected_passed:
        raise PairedAggregateBindingError("aggregate passed flag is inconsistent")
    expected_counts = {
        "success_count": sum(row.get("task_success") is True for row in batch.episode_rows),
        "body_collision_count": sum(
            row.get("body_collision") is True for row in batch.episode_rows
        ),
        "wheel_only_climb_count": sum(
            row.get("wheel_only_climb") is True for row in batch.episode_rows
        ),
        "safety_abort_count": sum(
            row.get("safety_abort") is True for row in batch.episode_rows
        ),
        "worker_gate_pass_count": sum(
            row.get("worker_gate_passed") is True for row in batch.worker_rows
        ),
    }
    if any(aggregate.get(name) != value for name, value in expected_counts.items()) or (
        aggregate.get("all_under_maximum_duration")
        is not all(
            row.get("under_maximum_duration") is True for row in batch.episode_rows
        )
    ):
        raise PairedAggregateBindingError("aggregate outcome counts are inconsistent")
    if selected_role == "candidate":
        assert checkpoint_manifest is not None and checkpoint_manifest_hash is not None
        for index, row in enumerate(batch.worker_rows):
            result = _json(
                snapshots[_absolute(str(row["worker_result"]))],
                label=f"candidate worker {index} result",
            )
            if (
                _absolute(str(result.get("checkpoint_manifest", "")))
                != checkpoint_manifest
                or result.get("checkpoint_manifest_sha256")
                != checkpoint_manifest_hash
            ):
                raise PairedAggregateBindingError(
                    "candidate workers do not share the explicit checkpoint manifest"
                )

    result = CapturedValidationAggregate(
        role=selected_role,
        aggregate_path=path,
        aggregate_payload=dict(aggregate),
        batch=batch,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_manifest_path=checkpoint_manifest,
        checkpoint_manifest_sha256=checkpoint_manifest_hash,
        source_snapshots=tuple(snapshots.values()),
        project_root=root,
    )
    result.assert_unchanged()
    return result


__all__ = (
    "CapturedValidationAggregate",
    "PairedAggregateBindingError",
    "SCHEMA",
    "capture_validation_aggregate",
)
