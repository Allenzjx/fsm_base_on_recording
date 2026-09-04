"""Immutable runtime capture for checkpoint/sidecar pairs.

Live RSL operations must never load a checkpoint through a caller-controlled
path after separately hashing that path.  This module captures both source
files once, writes byte-for-byte private copies beneath the managed run
directory, and keeps enough filesystem identity evidence to detect source or
copy replacement for the full lifetime of the operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import uuid


CHECKPOINT_CAPTURE_SCHEMA = "wlr50_clean.checkpoint_runtime_capture.v1"
_REPARSE_POINT_ATTRIBUTE = 0x400
_PURPOSE = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")


class CheckpointRuntimeCaptureError(RuntimeError):
    """Raised when checkpoint bytes cannot be pinned without ambiguity."""


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int
    attributes: int
    link_count: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            changed_ns=int(value.st_ctime_ns),
            attributes=int(getattr(value, "st_file_attributes", 0)),
            link_count=int(value.st_nlink),
        )


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    identity: _FileIdentity


def _unredirected_absolute_path(path: Path | str, *, label: str) -> Path:
    """Return an absolute path after rejecting every existing redirector."""

    absolute = Path(os.path.abspath(os.fspath(Path(path))))
    for component in reversed((absolute, *absolute.parents)):
        try:
            status = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise CheckpointRuntimeCaptureError(
                f"cannot inspect {label} path component: {component}"
            ) from exc
        is_junction = getattr(component, "is_junction", None)
        attributes = int(getattr(status, "st_file_attributes", 0))
        if (
            component.is_symlink()
            or (callable(is_junction) and is_junction())
            or attributes & _REPARSE_POINT_ATTRIBUTE
        ):
            raise CheckpointRuntimeCaptureError(
                f"{label} contains a symlink or reparse point: {component}"
            )
    resolved = absolute.resolve()
    if resolved != absolute:
        raise CheckpointRuntimeCaptureError(f"{label} path is redirected")
    return resolved


def _read_snapshot(path: Path, *, label: str) -> _FileSnapshot:
    """Read exactly once through one descriptor and bind its filesystem identity."""

    try:
        lexical_before = path.lstat()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
        lexical_after = path.lstat()
    except OSError as exc:
        raise CheckpointRuntimeCaptureError(f"cannot capture {label}: {path}") from exc
    identities = tuple(
        _FileIdentity.from_stat(value)
        for value in (lexical_before, before, after, lexical_after)
    )
    if any(identity != identities[0] for identity in identities[1:]):
        raise CheckpointRuntimeCaptureError(f"{label} changed while being captured")
    identity = identities[0]
    if not stat.S_ISREG(identity.mode):
        raise CheckpointRuntimeCaptureError(f"{label} is not a regular file")
    if identity.attributes & _REPARSE_POINT_ATTRIBUTE:
        raise CheckpointRuntimeCaptureError(f"{label} is a reparse point")
    if not payload or identity.size != len(payload):
        raise CheckpointRuntimeCaptureError(f"{label} is empty or incompletely read")
    return _FileSnapshot(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        identity=identity,
    )


def _load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointRuntimeCaptureError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise CheckpointRuntimeCaptureError(f"{label} must contain a JSON object")
    return dict(value)


def _load_embedded_infos(payload: bytes) -> dict[str, Any]:
    try:
        import torch  # type: ignore

        decoded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointRuntimeCaptureError(
            "checkpoint cannot be safely decoded with the restricted PyTorch loader"
        ) from exc
    if not isinstance(decoded, Mapping) or not isinstance(decoded.get("infos"), Mapping):
        raise CheckpointRuntimeCaptureError("checkpoint omits embedded RSL infos")
    return dict(decoded["infos"])


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> _FileSnapshot:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise CheckpointRuntimeCaptureError(
            f"cannot create exclusive private {label}: {path}"
        ) from exc
    snapshot = _read_snapshot(path, label=f"private {label}")
    if snapshot.payload != payload:
        raise CheckpointRuntimeCaptureError(f"private {label} differs after write")
    if snapshot.identity.link_count != 1:
        raise CheckpointRuntimeCaptureError(f"private {label} is unexpectedly hard-linked")
    return snapshot


@dataclass
class CapturedCheckpointBundle:
    """One source pair pinned to private, exclusive files for a live operation."""

    source_checkpoint_path: Path
    source_manifest_path: Path
    checkpoint_sha256: str
    manifest_sha256: str
    manifest_payload: Mapping[str, Any]
    embedded_infos: Mapping[str, Any]
    private_directory: Path
    private_checkpoint_path: Path
    private_manifest_path: Path
    _source_checkpoint: _FileSnapshot = field(repr=False)
    _source_manifest: _FileSnapshot = field(repr=False)
    _private_checkpoint: _FileSnapshot = field(repr=False)
    _private_manifest: _FileSnapshot = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def schema(self) -> str:
        return CHECKPOINT_CAPTURE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_checkpoint_path": str(self.source_checkpoint_path),
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "source_manifest_path": str(self.source_manifest_path),
            "source_manifest_sha256": self.manifest_sha256,
            "private_checkpoint_path": str(self.private_checkpoint_path),
            "private_manifest_path": str(self.private_manifest_path),
            "private_copy_exclusive": True,
            "runner_loads_private_copy_only": True,
        }

    def _require_open(self) -> None:
        if self._closed:
            raise CheckpointRuntimeCaptureError("checkpoint capture is already closed")

    def assert_loaded_infos(self, infos: Mapping[str, Any] | None) -> dict[str, Any]:
        self._require_open()
        if not isinstance(infos, Mapping):
            raise CheckpointRuntimeCaptureError("runner returned invalid checkpoint infos")
        loaded = dict(infos)
        if loaded != dict(self.embedded_infos):
            raise CheckpointRuntimeCaptureError(
                "runner-loaded infos differ from the captured checkpoint infos"
            )
        self.assert_private_copy_unchanged()
        return loaded

    def assert_private_copy_unchanged(self) -> None:
        self._require_open()
        checkpoint = _read_snapshot(
            _unredirected_absolute_path(
                self.private_checkpoint_path, label="private checkpoint"
            ),
            label="private checkpoint",
        )
        manifest = _read_snapshot(
            _unredirected_absolute_path(
                self.private_manifest_path, label="private checkpoint manifest"
            ),
            label="private checkpoint manifest",
        )
        if (
            checkpoint.identity != self._private_checkpoint.identity
            or checkpoint.sha256 != self.checkpoint_sha256
            or manifest.identity != self._private_manifest.identity
            or manifest.sha256 != self.manifest_sha256
        ):
            raise CheckpointRuntimeCaptureError(
                "private checkpoint capture changed during the live operation"
            )

    def assert_sources_unchanged(self) -> None:
        self._require_open()
        checkpoint = _read_snapshot(
            _unredirected_absolute_path(
                self.source_checkpoint_path, label="source checkpoint"
            ),
            label="source checkpoint",
        )
        manifest = _read_snapshot(
            _unredirected_absolute_path(
                self.source_manifest_path, label="source checkpoint manifest"
            ),
            label="source checkpoint manifest",
        )
        if (
            checkpoint.identity != self._source_checkpoint.identity
            or checkpoint.sha256 != self.checkpoint_sha256
            or manifest.identity != self._source_manifest.identity
            or manifest.sha256 != self.manifest_sha256
        ):
            raise CheckpointRuntimeCaptureError(
                "source checkpoint or manifest changed during the live operation"
            )

    def assert_unchanged(self) -> None:
        self.assert_private_copy_unchanged()
        self.assert_sources_unchanged()

    def cleanup(self) -> None:
        if self._closed:
            return
        # Delete only the two exact files created by this capture.  Never use a
        # recursive operation; an unexpected file leaves the directory behind.
        for target in (self.private_checkpoint_path, self.private_manifest_path):
            if target.parent != self.private_directory:
                raise CheckpointRuntimeCaptureError(
                    "refusing to clean a private checkpoint outside its capture directory"
                )
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                raise CheckpointRuntimeCaptureError(
                    f"cannot remove private checkpoint capture file: {target}"
                ) from exc
        try:
            self.private_directory.rmdir()
        except OSError:
            # Safe cleanup deliberately leaves a nonempty directory untouched.
            pass
        self._closed = True

    def __enter__(self) -> "CapturedCheckpointBundle":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.assert_unchanged()
        finally:
            self.cleanup()
        return False


def capture_checkpoint_bundle(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    *,
    run_directory: Path | str,
    purpose: str,
) -> CapturedCheckpointBundle:
    """Capture and privately pin a checkpoint pair beneath ``run_directory``."""

    if _PURPOSE.fullmatch(purpose) is None:
        raise CheckpointRuntimeCaptureError(
            "checkpoint capture purpose must be a short filesystem-safe label"
        )
    checkpoint = _unredirected_absolute_path(
        checkpoint_path, label="source checkpoint"
    )
    manifest = _unredirected_absolute_path(
        manifest_path, label="source checkpoint manifest"
    )
    if checkpoint == manifest:
        raise CheckpointRuntimeCaptureError(
            "checkpoint and checkpoint manifest must be distinct files"
        )
    checkpoint_snapshot = _read_snapshot(checkpoint, label="source checkpoint")
    manifest_snapshot = _read_snapshot(
        manifest, label="source checkpoint manifest"
    )
    manifest_payload = _load_json(
        manifest_snapshot.payload, label="source checkpoint manifest"
    )
    embedded_infos = _load_embedded_infos(checkpoint_snapshot.payload)
    declared_path = manifest_payload.get("checkpoint_path")
    if not isinstance(declared_path, str):
        raise CheckpointRuntimeCaptureError(
            "checkpoint manifest is bound to a different source path"
        )
    declared_checkpoint = _unredirected_absolute_path(
        declared_path, label="manifest-declared checkpoint"
    )
    if declared_checkpoint != checkpoint:
        raise CheckpointRuntimeCaptureError(
            "checkpoint manifest is bound to a different source path"
        )
    if manifest_payload.get("checkpoint_sha256") != checkpoint_snapshot.sha256:
        raise CheckpointRuntimeCaptureError(
            "checkpoint manifest hash differs from the captured source bytes"
        )
    differing = [
        key
        for key, value in embedded_infos.items()
        if key not in manifest_payload or manifest_payload[key] != value
    ]
    if differing:
        raise CheckpointRuntimeCaptureError(
            "checkpoint manifest rewrites embedded infos: " + ", ".join(differing)
        )

    run_root = _unredirected_absolute_path(run_directory, label="managed run directory")
    if not run_root.is_dir():
        raise CheckpointRuntimeCaptureError(
            f"managed run directory is missing: {run_root}"
        )
    pins_root = run_root / ".checkpoint-pins"
    try:
        pins_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise CheckpointRuntimeCaptureError(
            f"cannot create checkpoint pin root: {pins_root}"
        ) from exc
    pins_root = _unredirected_absolute_path(
        pins_root, label="checkpoint pin root"
    )
    private_directory = pins_root / f"{purpose}-{uuid.uuid4().hex}"
    try:
        private_directory.mkdir(mode=0o700)
    except OSError as exc:
        raise CheckpointRuntimeCaptureError(
            f"cannot create exclusive checkpoint capture directory: {private_directory}"
        ) from exc
    private_directory = _unredirected_absolute_path(
        private_directory, label="checkpoint capture directory"
    )
    private_checkpoint_path = private_directory / "checkpoint.pt"
    private_manifest_path = private_directory / "checkpoint_manifest.json"
    try:
        private_checkpoint = _write_exclusive(
            private_checkpoint_path,
            checkpoint_snapshot.payload,
            label="checkpoint",
        )
        private_manifest = _write_exclusive(
            private_manifest_path,
            manifest_snapshot.payload,
            label="checkpoint manifest",
        )
    except Exception:
        for target in (private_checkpoint_path, private_manifest_path):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            private_directory.rmdir()
        except OSError:
            pass
        raise
    return CapturedCheckpointBundle(
        source_checkpoint_path=checkpoint,
        source_manifest_path=manifest,
        checkpoint_sha256=checkpoint_snapshot.sha256,
        manifest_sha256=manifest_snapshot.sha256,
        manifest_payload=manifest_payload,
        embedded_infos=embedded_infos,
        private_directory=private_directory,
        private_checkpoint_path=private_checkpoint_path,
        private_manifest_path=private_manifest_path,
        _source_checkpoint=checkpoint_snapshot,
        _source_manifest=manifest_snapshot,
        _private_checkpoint=private_checkpoint,
        _private_manifest=private_manifest,
    )


__all__ = [
    "CHECKPOINT_CAPTURE_SCHEMA",
    "CapturedCheckpointBundle",
    "CheckpointRuntimeCaptureError",
    "capture_checkpoint_bundle",
]
