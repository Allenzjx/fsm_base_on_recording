"""Fail-closed initialization and publication of the canonical zero PPO actor.

The live initializer writes only inside its immutable managed run.  This
offline module accepts that checkpoint only after the run has finalized
successfully, revalidates its embedded RSL infos and exact-zero output layer,
and then publishes the canonical pair without overwriting existing evidence.
"""

from __future__ import annotations

import io
import hashlib
import json
import math
import os
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Mapping

from .artifacts import ArtifactError, git_head


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ppo_phase_v1"
INITIAL_CHECKPOINT_NAME = "checkpoint_initial_zero_residual.pt"
INITIAL_MANIFEST_NAME = "checkpoint_initial_zero_residual_manifest.json"
INITIAL_RESULT_NAME = "initial_checkpoint_result.json"
INITIAL_RESULT_SCHEMA = "wlr50_clean.initial_zero_residual_checkpoint_run.v1"
PUBLICATION_SCHEMA = "wlr50_clean.initial_zero_residual_checkpoint_publication.v1"
INITIAL_RUN_KIND = "initial-checkpoint"
INITIAL_RUN_STAGE = "initialize-zero-residual"


class InitialCheckpointError(RuntimeError):
    """The initial checkpoint is missing, stale, redirected, or non-zero."""


@dataclass(frozen=True, slots=True)
class InitialCheckpointEvidence:
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    creation_run_kind: str
    creation_run_directory: Path
    creation_run_manifest: Mapping[str, Any]
    checkpoint_bytes: bytes
    manifest_bytes: bytes


@dataclass(frozen=True, slots=True)
class InitialCheckpointPublication:
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    reused_existing: bool
    creation_run_kind: str
    creation_run_directory: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PUBLICATION_SCHEMA,
            "source_checkpoint": str(self.source_checkpoint_path),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_checkpoint_manifest": str(self.source_manifest_path),
            "source_checkpoint_manifest_sha256": self.source_manifest_sha256,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_manifest": str(self.manifest_path),
            "checkpoint_manifest_sha256": self.manifest_sha256,
            "reused_existing": self.reused_existing,
            "no_existing_artifact_overwritten": True,
            "source_initializer_finalized_success": True,
            "embedded_infos_match_manifest": True,
            "zero_mean_actor_output_layer_verified": True,
            "creation_run_kind": self.creation_run_kind,
            "creation_run_directory": str(self.creation_run_directory),
        }


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    born_ns: int


def _identity(status: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=int(status.st_dev),
        inode=int(status.st_ino),
        size=int(status.st_size),
        modified_ns=int(status.st_mtime_ns),
        changed_ns=int(status.st_ctime_ns),
        born_ns=int(getattr(status, "st_birthtime_ns", status.st_ctime_ns)),
    )


def _same_file_object(left: _FileIdentity, right: _FileIdentity) -> bool:
    """Compare stable file identity fields; Windows changes ctime on rename/open."""

    return left.device == right.device and left.inode == right.inode


@dataclass(slots=True)
class _OpenSource:
    path: Path
    stream: BinaryIO
    data: bytes
    identity: _FileIdentity
    project_root: Path
    label: str

    def assert_unchanged(self) -> None:
        from .training_orchestration import TrainingOrchestrationError, _reject_links

        try:
            _reject_links(self.path, root=self.project_root, label=self.label)
            handle_identity = _identity(os.fstat(self.stream.fileno()))
            path_status = os.stat(self.path, follow_symlinks=False)
            path_identity = _identity(path_status)
            position = self.stream.tell()
            self.stream.seek(0)
            current_data = self.stream.read()
            self.stream.seek(position)
            is_junction = getattr(self.path, "is_junction", None)
            attributes = int(getattr(path_status, "st_file_attributes", 0))
        except (OSError, TrainingOrchestrationError) as exc:
            raise InitialCheckpointError(
                f"captured source disappeared or became unreadable: {self.path}"
            ) from exc
        if (
            not _same_file_object(handle_identity, self.identity)
            or not _same_file_object(path_identity, self.identity)
            or handle_identity.size != self.identity.size
            or path_identity.size != self.identity.size
            or handle_identity.modified_ns != self.identity.modified_ns
            or path_identity.modified_ns != self.identity.modified_ns
            or current_data != self.data
            or not stat.S_ISREG(path_status.st_mode)
            or self.path.is_symlink()
            or (callable(is_junction) and is_junction())
            or attributes & 0x400
        ):
            raise InitialCheckpointError(
                f"captured source identity changed before publication: {self.path}"
            )


@dataclass(slots=True)
class _SourcePairCapture:
    checkpoint: _OpenSource
    manifest: _OpenSource

    def assert_unchanged(self) -> None:
        self.checkpoint.assert_unchanged()
        self.manifest.assert_unchanged()

    def seeded_cache(self) -> dict[Path, Any]:
        from .training_orchestration import _Snapshot

        return {
            source.path: _Snapshot(
                path=source.path,
                data=source.data,
                size=len(source.data),
                sha256=hashlib.sha256(source.data).hexdigest(),
                creation_time_utc_ticks=(
                    621_355_968_000_000_000 + source.identity.born_ns // 100
                ),
                last_write_time_utc_ticks=(
                    621_355_968_000_000_000 + source.identity.modified_ns // 100
                ),
            )
            for source in (self.checkpoint, self.manifest)
        }


def _open_source(
    path: Path, *, label: str, project_root: Path
) -> _OpenSource:
    try:
        stream = path.open("rb")
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise InitialCheckpointError(f"{label} is not a regular file: {path}")
        data = stream.read()
        after = os.fstat(stream.fileno())
    except Exception:
        if "stream" in locals():
            stream.close()
        raise
    if not data or _identity(before) != _identity(after) or len(data) != before.st_size:
        stream.close()
        raise InitialCheckpointError(f"{label} changed while being captured: {path}")
    return _OpenSource(
        path=path,
        stream=stream,
        data=data,
        identity=_identity(before),
        project_root=project_root,
        label=label,
    )


@contextmanager
def _capture_source_pair(
    checkpoint: Path, manifest: Path, *, project_root: Path
) -> Iterator[_SourcePairCapture]:
    from .training_orchestration import TrainingOrchestrationError, _reject_links

    try:
        _reject_links(checkpoint, root=project_root, label="source initial checkpoint")
        _reject_links(
            manifest, root=project_root, label="source initial checkpoint manifest"
        )
    except TrainingOrchestrationError as exc:
        raise InitialCheckpointError(str(exc)) from exc
    checkpoint_source = _open_source(
        checkpoint,
        label="source initial checkpoint",
        project_root=project_root,
    )
    try:
        manifest_source = _open_source(
            manifest,
            label="source initial checkpoint manifest",
            project_root=project_root,
        )
    except Exception:
        checkpoint_source.stream.close()
        raise
    captured = _SourcePairCapture(checkpoint_source, manifest_source)
    try:
        captured.assert_unchanged()
        yield captured
        captured.assert_unchanged()
    finally:
        manifest_source.stream.close()
        checkpoint_source.stream.close()


def _manifest_core(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"checkpoint_path", "checkpoint_sha256"}
    }


def _exact_finite_number(value: Any, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and math.isclose(
        number, expected, rel_tol=0.0, abs_tol=0.0
    )


def _validate_runtime_contract(
    manifest: Mapping[str, Any], *, project_root: Path, cache: Any
) -> None:
    from .phase_snapshots import (
        PhaseSnapshotError,
        capture_validated_phase_snapshot_bundle,
        phase_snapshot_bundle_file_hashes,
    )
    from .phase_effective_entry import (
        EffectivePhaseEntryError,
        capture_validated_effective_phase_entry_contract,
    )
    from .training_orchestration import _reject_links, _snapshot

    snapshot_root = project_root / "reference" / "ppo_phase_snapshots"
    try:
        snapshot_pin = capture_validated_phase_snapshot_bundle(
            snapshot_root, canonical_root=snapshot_root
        )
        bundle = snapshot_pin.as_record()
        snapshot_hashes = phase_snapshot_bundle_file_hashes(bundle)
        effective_pin = capture_validated_effective_phase_entry_contract(
            project_root / "configs" / "ppo_phase_effective_entry_v1.json",
            expected_snapshot_bundle=snapshot_pin,
            environment_lock_path=project_root / "configs" / "environment_lock.json",
            frozen_ledger_path=(
                project_root
                / "artifacts"
                / "ppo_phase_v1_start"
                / "frozen_fsm_hashes.json"
            ),
        )
    except (OSError, PhaseSnapshotError, EffectivePhaseEntryError) as exc:
        raise InitialCheckpointError(
            f"current phase reset contract is invalid: {exc}"
        ) from exc
    expected_snapshot_fields = {
        "phase_snapshot_manifest": bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": bundle["bundle_sha256"],
        "phase_snapshot_bundle": bundle,
        "phase_effective_entry_contract_path": str(effective_pin.contract_path),
        "phase_effective_entry_contract_file_sha256": effective_pin.file_sha256,
        "phase_effective_entry_contract_sidecar_path": str(effective_pin.sidecar_path),
        "phase_effective_entry_contract_sidecar_sha256": (
            effective_pin.sidecar_file_sha256
        ),
        "phase_effective_entry_contract_sha256": effective_pin.contract_sha256,
        "phase_effective_entry_contract": effective_pin.as_record(),
    }
    differing = [
        field
        for field, value in expected_snapshot_fields.items()
        if manifest.get(field) != value
    ]
    if differing:
        raise InitialCheckpointError(
            "initial checkpoint phase snapshot contract differs: "
            + ", ".join(differing)
        )

    config_paths = (
        project_root / "configs" / "ppo_training_phase_v1.yaml",
        project_root / "configs" / "ppo_interface_v2.yaml",
        project_root / "configs" / "ppo_phase_effective_entry_v1.json",
        project_root / "configs" / "ppo_phase_effective_entry_v1.sha256",
        project_root / "configs" / "ppo_observation_schema_v2.json",
        project_root / "configs" / "ppo_phase_action_masks_v2.yaml",
        project_root / "configs" / "ppo_phase_objectives_v2.yaml",
        project_root / "configs" / "ppo_reward_v2.yaml",
        project_root / "configs" / "ppo_termination_v2.yaml",
        project_root / "configs" / "ppo_domain_randomization_v2.yaml",
        project_root / "configs" / "frozen_successful_fsm.yaml",
        project_root / "configs" / "environment_lock.json",
        project_root / "configs" / "fsm_states.yaml",
        project_root / "configs" / "recording_motion_contract.json",
    )
    expected_files = {str(path): None for path in config_paths}
    expected_files.update(snapshot_hashes)
    expected_files.update(effective_pin.file_hashes())
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(expected_files):
        raise InitialCheckpointError(
            "initial checkpoint input inventory is incomplete or contains extras"
        )
    for raw_path, declared_hash in files.items():
        if not isinstance(raw_path, str) or not isinstance(declared_hash, str):
            raise InitialCheckpointError("initial checkpoint input record is malformed")
        path = Path(raw_path)
        _reject_links(path, root=project_root, label="initial checkpoint input")
        captured = _snapshot(path, label="initial checkpoint input", cache=cache)
        if captured.sha256 != declared_hash:
            raise InitialCheckpointError(
                f"initial checkpoint input SHA-256 is stale: {path}"
            )
    expected_named_hashes = {
        "controller_hash": project_root / "configs" / "fsm_states.yaml",
        "environment_hash": project_root / "configs" / "environment_lock.json",
        "observation_schema_hash": (
            project_root / "configs" / "ppo_observation_schema_v2.json"
        ),
        "action_schema_hash": (
            project_root / "configs" / "ppo_phase_action_masks_v2.yaml"
        ),
        "reward_config_hash": project_root / "configs" / "ppo_reward_v2.yaml",
    }
    for field, path in expected_named_hashes.items():
        if manifest.get(field) != _snapshot(
            path, label=f"initial checkpoint {field}", cache=cache
        ).sha256:
            raise InitialCheckpointError(f"initial checkpoint {field} is stale")


def _load_and_validate_embedded_checkpoint(
    checkpoint_bytes: bytes, manifest: Mapping[str, Any]
) -> None:
    try:
        import torch  # type: ignore

        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise InitialCheckpointError(
            "initial checkpoint cannot be safely decoded with weights_only=True"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("infos"), Mapping):
        raise InitialCheckpointError("initial checkpoint omits embedded RSL infos")
    infos = dict(payload["infos"])
    if infos != _manifest_core(manifest):
        raise InitialCheckpointError(
            "initial checkpoint embedded infos differ from its sidecar"
        )
    actor = payload.get("actor_state_dict")
    if not isinstance(actor, Mapping):
        raise InitialCheckpointError("initial checkpoint omits actor_state_dict")
    residual_dimension = int(manifest.get("residual_dimension", -1))
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
        raise InitialCheckpointError(
            "initial checkpoint actor Full12 output layer is ambiguous"
        )
    weight, bias = candidates[0]
    if (
        not bool(torch.isfinite(weight).all().item())
        or not bool(torch.isfinite(bias).all().item())
        or int(torch.count_nonzero(weight).item()) != 0
        or int(torch.count_nonzero(bias).item()) != 0
    ):
        raise InitialCheckpointError(
            "initial checkpoint actor output layer is not exact zero"
        )


def _validate_creation_run(
    manifest: Mapping[str, Any], *, project_root: Path, cache: Any
) -> tuple[str, Path, Mapping[str, Any]]:
    from .training_orchestration import (
        TrainingOrchestrationError,
        _absolute,
        _json,
        _record,
        _required_artifact,
        _validate_finalized_run,
    )

    identity_path = _absolute(str(manifest.get("creation_runtime_identity_path", "")))
    if identity_path.name != "committed_runtime_identity.before.json":
        raise InitialCheckpointError(
            "initial checkpoint creation runtime identity has the wrong filename"
        )
    runs_root = project_root / "runs" / "ppo_phase_v1"
    try:
        relative = identity_path.parent.relative_to(runs_root)
    except ValueError as exc:
        raise InitialCheckpointError(
            "initial checkpoint creation runtime escapes the managed runs root"
        ) from exc
    if len(relative.parts) != 2:
        raise InitialCheckpointError("initial checkpoint creation run path is malformed")
    run_kind = relative.parts[0]
    if run_kind == INITIAL_RUN_KIND:
        training_stage = INITIAL_RUN_STAGE
        subcommand = "initialize-zero-residual"
    elif run_kind == "train":
        # A previously published checkpoint from a successful legacy train run
        # remains reusable.  Failed train runs are rejected by lifecycle checks.
        training_stage = None
        subcommand = "train"
    else:
        raise InitialCheckpointError(
            "initial checkpoint was not created by an initializer or legacy train run"
        )
    try:
        run = _validate_finalized_run(
            identity_path.parent,
            project_root=project_root,
            run_kind=run_kind,
            training_stage=training_stage,
            entrypoint="wlr50_clean.ppo.cli",
            subcommand=subcommand,
            cache=cache,
        )
    except TrainingOrchestrationError as exc:
        raise InitialCheckpointError(
            f"initial checkpoint creation run is not finalized success: {exc}"
        ) from exc
    before = run["committed_runtime_identities"][0]
    before_payload = run["committed_runtime_identity_before_payload"]
    if (
        identity_path != run["directory"] / "committed_runtime_identity.before.json"
        or manifest.get("creation_runtime_identity_sha256") != before["sha256"]
        or manifest.get("source_git_commit") != before_payload.get("git_commit")
        or manifest.get("committed_runtime_content_sha256")
        != before_payload.get("content_sha256")
        or run["identity"].get("seed") != manifest.get("training_seed")
    ):
        raise InitialCheckpointError(
            "initial checkpoint creation runtime binding is inconsistent"
        )
    if run_kind == INITIAL_RUN_KIND:
        if run["identity"].get("environment_count") != 1:
            raise InitialCheckpointError("initializer run was not single-environment")
        result_path, result = _required_artifact(
            run,
            INITIAL_RESULT_NAME,
            cache=cache,
            label="initial checkpoint result",
        )
        checkpoint_record = _record(
            run["artifacts"].get(INITIAL_CHECKPOINT_NAME),
            base=run["directory"],
            expected_path=INITIAL_CHECKPOINT_NAME,
            label="initializer checkpoint",
            cache=cache,
        )
        manifest_record = _record(
            run["artifacts"].get(INITIAL_MANIFEST_NAME),
            base=run["directory"],
            expected_path=INITIAL_MANIFEST_NAME,
            label="initializer checkpoint manifest",
            cache=cache,
        )
        staged_manifest = _json(
            manifest_record.path,
            label="initializer checkpoint manifest",
            cache=cache,
        )
        if (
            result.get("schema") != INITIAL_RESULT_SCHEMA
            or result.get("stage") != "initial_zero_residual"
            or result.get("seed") != manifest.get("training_seed")
            or result.get("num_envs") != 1
            or result.get("global_policy_decisions") != 0
            or result.get("save_load_round_trip") is not True
            or result.get("checkpoint_private_capture_verified") is not True
            or result.get("zero_mean_actor_output_layer_verified_before_save")
            is not True
            or result.get("zero_mean_actor_output_layer_verified_after_load")
            is not True
            or result.get("phase_snapshot_bundle")
            != manifest.get("phase_snapshot_bundle")
            or result.get("phase_effective_entry_contract")
            != manifest.get("phase_effective_entry_contract")
            or _absolute(str(result.get("checkpoint", "")))
            != checkpoint_record.path
            or result.get("checkpoint_sha256") != checkpoint_record.sha256
            or _absolute(str(result.get("checkpoint_manifest", "")))
            != manifest_record.path
            or result.get("checkpoint_manifest_sha256") != manifest_record.sha256
            or _manifest_core(staged_manifest) != _manifest_core(manifest)
            or staged_manifest.get("checkpoint_sha256")
            != manifest.get("checkpoint_sha256")
            or staged_manifest.get("checkpoint_sha256")
            != checkpoint_record.sha256
            or result_path != run["directory"] / INITIAL_RESULT_NAME
        ):
            raise InitialCheckpointError(
                "initializer result does not bind its verified checkpoint pair"
            )
    return run_kind, run["directory"], run["run_manifest"]


def validate_initial_zero_residual_checkpoint(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    *,
    project_root: Path | str = PROJECT_ROOT,
    expected_seed: int | None = None,
    _seeded_cache: Mapping[Path, Any] | None = None,
) -> InitialCheckpointEvidence:
    """Validate an initial checkpoint, its current ABI, and finalized creator."""

    from .training_orchestration import (
        CHECKPOINT_MANIFEST_SCHEMA,
        TrainingOrchestrationError,
        _absolute,
        _json,
        _reject_links,
        _revalidate,
        _snapshot,
    )

    root = _absolute(project_root)
    checkpoint = _absolute(checkpoint_path)
    sidecar = _absolute(manifest_path)
    cache: Any = {} if _seeded_cache is None else dict(_seeded_cache)
    seeded_paths = frozenset(cache)
    try:
        _reject_links(checkpoint, root=root, label="initial checkpoint")
        _reject_links(sidecar, root=root, label="initial checkpoint manifest")
        checkpoint_snapshot = _snapshot(
            checkpoint, label="initial checkpoint", cache=cache
        )
        manifest = _json(sidecar, label="initial checkpoint manifest", cache=cache)
        manifest_snapshot = _snapshot(
            sidecar, label="initial checkpoint manifest", cache=cache
        )
    except TrainingOrchestrationError as exc:
        raise InitialCheckpointError(str(exc)) from exc
    seed = manifest.get("training_seed")
    try:
        current_git_head = git_head(root)
    except ArtifactError as exc:
        raise InitialCheckpointError(str(exc)) from exc
    if (
        manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or manifest.get("stage") != "initial_zero_residual"
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or (expected_seed is not None and seed != expected_seed)
        or manifest.get("global_policy_decisions") != 0
        or manifest.get("actor_observation_dimension") != 125
        or manifest.get("critic_observation_dimension") != 125
        or manifest.get("residual_dimension") != 12
        or not _exact_finite_number(manifest.get("physics_hz"), 120.0)
        or not _exact_finite_number(manifest.get("decision_hz"), 15.0)
        or manifest.get("zero_mean_actor_output_layer_verified") is not True
        or _absolute(str(manifest.get("checkpoint_path", ""))) != checkpoint
        or manifest.get("checkpoint_sha256") != checkpoint_snapshot.sha256
        or manifest.get("source_git_commit") != current_git_head
        or not isinstance(manifest.get("training_rng_seed_evidence"), Mapping)
        or manifest["training_rng_seed_evidence"].get("seed") != seed
        or not isinstance(manifest.get("training_rng_state"), Mapping)
        or manifest["training_rng_state"].get("seed") != seed
        or manifest["training_rng_state"].get("schema")
        != "wlr50_clean.training_rng_state.v1"
    ):
        raise InitialCheckpointError("initial checkpoint manifest header is invalid")
    learning_rate = manifest.get("optimizer_learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise InitialCheckpointError("initial checkpoint optimizer state is invalid")
    try:
        _validate_runtime_contract(manifest, project_root=root, cache=cache)
        _load_and_validate_embedded_checkpoint(checkpoint_snapshot.data, manifest)
        run_kind, run_directory, run_manifest = _validate_creation_run(
            manifest, project_root=root, cache=cache
        )
        _revalidate(
            {path: snapshot for path, snapshot in cache.items() if path not in seeded_paths},
            project_root=root,
        )
    except TrainingOrchestrationError as exc:
        raise InitialCheckpointError(str(exc)) from exc
    return InitialCheckpointEvidence(
        checkpoint_path=checkpoint_snapshot.path,
        checkpoint_sha256=checkpoint_snapshot.sha256,
        manifest_path=manifest_snapshot.path,
        manifest_sha256=manifest_snapshot.sha256,
        manifest=dict(manifest),
        creation_run_kind=run_kind,
        creation_run_directory=run_directory,
        creation_run_manifest=dict(run_manifest),
        checkpoint_bytes=checkpoint_snapshot.data,
        manifest_bytes=manifest_snapshot.data,
    )


@dataclass(frozen=True, slots=True)
class _OwnedArtifact:
    path: Path
    identity: _FileIdentity
    data: bytes


class _ExclusiveCollision(InitialCheckpointError):
    """Another publisher created the immutable destination first."""


def _publish_bytes_exclusive(destination: Path, payload: bytes) -> _OwnedArtifact:
    """Atomically publish complete bytes without ever replacing a destination.

    Windows rename is no-clobber but requires the Python CRT handle to be
    closed first.  POSIX uses a hard link because rename would overwrite; its
    staging alias lives in a call-exclusive directory and is removed through a
    held directory descriptor only after its inode identity is rechecked.
    """

    if not destination.parent.is_dir():
        raise InitialCheckpointError(
            f"canonical publication parent is missing: {destination.parent}"
        )
    temporary_identity: _FileIdentity
    temporary: Path | None = None
    staging_fd: int | None = None
    parent_fd: int | None = None
    staging_name: str | None = None
    staging_identity: _FileIdentity | None = None
    staging_payload_name = "payload"
    linked = False
    cleanup_failure: str | None = None
    try:
        if os.name == "nt":
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            with temporary.open("x+b") as stream:
                temporary_identity = _identity(os.fstat(stream.fileno()))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                temporary_identity = _identity(os.fstat(stream.fileno()))
                stream.seek(0)
                if stream.read() != payload:
                    raise InitialCheckpointError(
                        f"canonical publication staging bytes are invalid: {temporary}"
                    )
            os.rename(temporary, destination)
        else:
            staging_name = f".{destination.name}.{uuid.uuid4().hex}.stage"
            directory_flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
            parent_fd = os.open(destination.parent, directory_flags)
            os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
            staging_identity = _identity(
                os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
            )
            staging_fd = os.open(staging_name, directory_flags, dir_fd=parent_fd)
            descriptor = os.open(
                staging_payload_name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
                dir_fd=staging_fd,
            )
            with os.fdopen(descriptor, "w+b") as stream:
                temporary_identity = _identity(os.fstat(stream.fileno()))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                temporary_identity = _identity(os.fstat(stream.fileno()))
                stream.seek(0)
                if stream.read() != payload:
                    raise InitialCheckpointError(
                        "canonical publication staging bytes are invalid"
                    )
            os.link(
                staging_payload_name,
                destination.name,
                src_dir_fd=staging_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            linked = True
        with destination.open("rb") as published:
            handle_status = os.fstat(published.fileno())
            destination_status = os.stat(destination, follow_symlinks=False)
            destination_identity = _identity(destination_status)
            published_data = published.read()
            after_status = os.fstat(published.fileno())
        if (
            not _same_file_object(temporary_identity, destination_identity)
            or not _same_file_object(temporary_identity, _identity(handle_status))
            or not _same_file_object(temporary_identity, _identity(after_status))
            or not stat.S_ISREG(destination_status.st_mode)
            or int(getattr(destination_status, "st_file_attributes", 0)) & 0x400
            or published_data != payload
        ):
            raise InitialCheckpointError(
                "canonical publication destination does not reference the complete "
                f"staged bytes: {destination}"
            )
    except FileExistsError as exc:
        raise _ExclusiveCollision(
            f"canonical initial checkpoint appeared concurrently: {destination}"
        ) from exc
    except InitialCheckpointError:
        raise
    except OSError as exc:
        raise InitialCheckpointError(
            f"canonical publication failed for {destination}: {exc}"
        ) from exc
    finally:
        if os.name == "nt" and temporary is not None and temporary.exists():
            try:
                status = os.stat(temporary, follow_symlinks=False)
                if _same_file_object(_identity(status), temporary_identity):
                    temporary.unlink()
                else:
                    cleanup_failure = "Windows staging identity changed"
            except (FileNotFoundError, OSError, UnboundLocalError) as exc:
                cleanup_failure = f"Windows staging cleanup failed: {exc}"
        payload_removed = staging_fd is None
        if os.name != "nt" and staging_fd is not None:
            try:
                staged_status = os.stat(
                    staging_payload_name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
                if _same_file_object(_identity(staged_status), temporary_identity):
                    os.unlink(staging_payload_name, dir_fd=staging_fd)
                    payload_removed = True
                else:
                    cleanup_failure = "POSIX staging payload identity changed"
            except FileNotFoundError:
                payload_removed = True
            except (OSError, UnboundLocalError) as exc:
                cleanup_failure = f"POSIX staging payload cleanup failed: {exc}"
            finally:
                os.close(staging_fd)
        if (
            os.name != "nt"
            and payload_removed
            and parent_fd is not None
            and staging_name is not None
            and staging_identity is not None
        ):
            try:
                current_stage = os.stat(
                    staging_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if _same_file_object(_identity(current_stage), staging_identity):
                    os.rmdir(staging_name, dir_fd=parent_fd)
                else:
                    cleanup_failure = "POSIX staging directory identity changed"
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_failure = f"POSIX staging directory cleanup failed: {exc}"
        if parent_fd is not None:
            os.close(parent_fd)
    if cleanup_failure is not None:
        raise InitialCheckpointError(cleanup_failure)
    return _OwnedArtifact(destination, temporary_identity, payload)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InitialCheckpointError(
            f"canonical initial manifest is not JSON serializable: {exc}"
        ) from exc


def _validate_publication_paths(
    *, root: Path, output: Path, checkpoint: Path, manifest: Path
) -> None:
    from .training_orchestration import TrainingOrchestrationError, _reject_links

    expected_output = root / "outputs" / "ppo_phase_v1"
    if output != expected_output:
        raise InitialCheckpointError(
            f"initial checkpoint output root must be canonical: {expected_output}"
        )
    for path, label in (
        (output, "initial checkpoint output root"),
        (checkpoint.parent, "initial checkpoint canonical directory"),
        (checkpoint, "canonical initial checkpoint"),
        (manifest, "canonical initial checkpoint manifest"),
    ):
        try:
            path.relative_to(root)
            _reject_links(path, root=root, label=label)
        except (ValueError, TrainingOrchestrationError) as exc:
            raise InitialCheckpointError(str(exc)) from exc
    for directory, label in (
        (output, "initial checkpoint output root"),
        (checkpoint.parent, "initial checkpoint canonical directory"),
    ):
        if directory.exists() and not directory.is_dir():
            raise InitialCheckpointError(f"{label} is not a directory: {directory}")


@contextmanager
def _capture_exact_existing(
    path: Path,
    expected: bytes,
    *,
    label: str,
    project_root: Path,
) -> Iterator[_OpenSource]:
    """Hold an exact partial publication stable while its peer is completed."""

    from .training_orchestration import TrainingOrchestrationError, _reject_links

    try:
        _reject_links(path, root=project_root, label=label)
    except TrainingOrchestrationError as exc:
        raise InitialCheckpointError(str(exc)) from exc
    opened = _open_source(path, label=label, project_root=project_root)
    try:
        opened.assert_unchanged()
        if opened.data != expected:
            raise InitialCheckpointError(
                f"incomplete canonical pair has different {label} bytes; refusing to overwrite it"
            )
        yield opened
        opened.assert_unchanged()
    finally:
        opened.stream.close()


def publish_initial_zero_residual_checkpoint(
    *,
    source_checkpoint: Path | str,
    source_manifest: Path | str,
    output_root: Path | str = OUTPUT_ROOT,
    project_root: Path | str = PROJECT_ROOT,
    expected_seed: int | None = None,
    _before_publish_hook: Callable[[], None] | None = None,
) -> InitialCheckpointPublication:
    """Publish once, or strictly validate and reuse an existing canonical pair."""

    root = Path(os.path.abspath(os.fspath(project_root)))
    output = Path(os.path.abspath(os.fspath(output_root)))
    canonical = output / "checkpoints" / INITIAL_CHECKPOINT_NAME
    canonical_manifest = output / "checkpoints" / INITIAL_MANIFEST_NAME
    _validate_publication_paths(
        root=root,
        output=output,
        checkpoint=canonical,
        manifest=canonical_manifest,
    )
    source_checkpoint_path = Path(os.path.abspath(os.fspath(source_checkpoint)))
    source_manifest_path = Path(os.path.abspath(os.fspath(source_manifest)))
    try:
        with _capture_source_pair(
            source_checkpoint_path, source_manifest_path, project_root=root
        ) as captured:
            seeded_cache = captured.seeded_cache()
            source = validate_initial_zero_residual_checkpoint(
                source_checkpoint_path,
                source_manifest_path,
                project_root=root,
                expected_seed=expected_seed,
                _seeded_cache=seeded_cache,
            )
            captured.assert_unchanged()
            payload = {
                **dict(source.manifest),
                "checkpoint_path": str(canonical),
                "checkpoint_sha256": source.checkpoint_sha256,
            }
            manifest_bytes = _json_bytes(payload)
            wrote_artifact = False
            hook_called = False

            # A pair is committed by preparing the manifest first and linking
            # the checkpoint last.  A crash can therefore leave one complete,
            # immutable side.  A retry may finish that pair only when the held
            # bytes match this exact validated source; foreign or truncated
            # partials are never overwritten or removed.
            for _attempt in range(6):
                captured.assert_unchanged()
                _validate_publication_paths(
                    root=root,
                    output=output,
                    checkpoint=canonical,
                    manifest=canonical_manifest,
                )
                checkpoint_exists = canonical.exists()
                manifest_exists = canonical_manifest.exists()

                if checkpoint_exists and manifest_exists:
                    existing = validate_initial_zero_residual_checkpoint(
                        canonical,
                        canonical_manifest,
                        project_root=root,
                        expected_seed=expected_seed,
                        _seeded_cache=seeded_cache,
                    )
                    captured.assert_unchanged()
                    if (
                        existing.checkpoint_sha256 != source.checkpoint_sha256
                        or _manifest_core(existing.manifest)
                        != _manifest_core(source.manifest)
                    ):
                        raise InitialCheckpointError(
                            "existing canonical initial checkpoint differs; refusing to overwrite it"
                        )
                    return InitialCheckpointPublication(
                        source_checkpoint_path=source.checkpoint_path,
                        source_checkpoint_sha256=source.checkpoint_sha256,
                        source_manifest_path=source.manifest_path,
                        source_manifest_sha256=source.manifest_sha256,
                        checkpoint_path=existing.checkpoint_path,
                        checkpoint_sha256=existing.checkpoint_sha256,
                        manifest_path=existing.manifest_path,
                        manifest_sha256=existing.manifest_sha256,
                        reused_existing=not wrote_artifact,
                        creation_run_kind=existing.creation_run_kind,
                        creation_run_directory=existing.creation_run_directory,
                    )

                if not hook_called and _before_publish_hook is not None:
                    _before_publish_hook()
                    hook_called = True
                    captured.assert_unchanged()
                    _validate_publication_paths(
                        root=root,
                        output=output,
                        checkpoint=canonical,
                        manifest=canonical_manifest,
                    )
                    continue
                hook_called = True

                if not canonical.parent.exists():
                    canonical.parent.mkdir(parents=True, exist_ok=True)
                    _validate_publication_paths(
                        root=root,
                        output=output,
                        checkpoint=canonical,
                        manifest=canonical_manifest,
                    )
                    continue

                try:
                    if manifest_exists and not checkpoint_exists:
                        with _capture_exact_existing(
                            canonical_manifest,
                            manifest_bytes,
                            label="canonical initial checkpoint manifest",
                            project_root=root,
                        ):
                            _publish_bytes_exclusive(
                                canonical, source.checkpoint_bytes
                            )
                        wrote_artifact = True
                    elif checkpoint_exists and not manifest_exists:
                        with _capture_exact_existing(
                            canonical,
                            source.checkpoint_bytes,
                            label="canonical initial checkpoint",
                            project_root=root,
                        ):
                            _publish_bytes_exclusive(
                                canonical_manifest, manifest_bytes
                            )
                        wrote_artifact = True
                    else:
                        # The sidecar is immutable prepare evidence.  The
                        # checkpoint pathname is the final commit marker.
                        _publish_bytes_exclusive(canonical_manifest, manifest_bytes)
                        wrote_artifact = True
                except _ExclusiveCollision:
                    # Re-read the complete state.  A same-source concurrent
                    # winner is reusable; different bytes fail in the next
                    # reconciliation branch.
                    continue
            raise InitialCheckpointError(
                "canonical initial checkpoint publication did not reach a stable state"
            )
    except Exception as exc:
        # Canonical path deletion is intentionally forbidden here.  A
        # check-then-unlink rollback can delete a foreign replacement after an
        # A->B race.  Complete exact partials remain fail-closed to consumers
        # and are safely recoverable on the next same-source invocation.
        if isinstance(exc, InitialCheckpointError):
            raise
        raise InitialCheckpointError(str(exc)) from exc
