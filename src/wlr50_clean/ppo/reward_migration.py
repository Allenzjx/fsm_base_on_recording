"""Auditable migration map from the inactive reward v1 to reward v2.

The map is reporting evidence only.  It cannot enable the v1 reward and it
does not execute either calculator.  Loading fails closed if either referenced
configuration or the complete ordered v1 concept inventory changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .phase_objectives import DENSE_FAMILIES
from .reward_terms import REWARD_SCHEMA, REWARD_TERMS
from .reward_v2 import REWARD_SCHEMA_V2


MIGRATION_SCHEMA = "wlr50_clean.ppo_reward_migration.v1"
DEFAULT_MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "ppo_reward_v1_to_v2_migration.yaml"
)
ALLOWED_DISPOSITIONS = frozenset(
    {
        "MERGED_INTO_DENSE_FAMILY",
        "MOVED_TO_EVENT",
        "MOVED_TO_SAFETY_EVENT",
        "MERGED_WITH_POSITIVE_SURVIVAL_REMOVED",
    }
)
EVENT_DESTINATIONS = frozenset(
    {
        "event.phase_completion",
        "event.final_success",
        "event.task_failure",
        "event.safety_abort",
    }
)
SAFETY_DESTINATIONS = frozenset(
    {
        "safety.body_collision_override",
        "safety.wheel_only_climb_override",
        "safety.joint_margin_projection",
    }
)


class RewardMigrationError(ValueError):
    """The declared migration no longer matches its source or target."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _captured_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RewardMigrationError(f"cannot read {label}: {path}") from exc


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RewardMigrationError(f"{label} must be a mapping")
    return value


def _resolve_bound_path(raw: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise RewardMigrationError(f"{label} path must be a non-empty string")
    requested = Path(raw)
    if requested.is_absolute() or ".." in requested.parts:
        raise RewardMigrationError(f"{label} path must be project-relative without '..'")
    lexical = project_root.joinpath(*requested.parts)
    # Do not let a symlink or Windows junction turn an apparently in-project
    # audit binding into an external mutable dependency.
    current = project_root
    for component in requested.parts:
        current = current / component
        try:
            stat = current.lstat()
        except OSError as exc:
            raise RewardMigrationError(f"{label} file is missing: {current}") from exc
        attributes = int(getattr(stat, "st_file_attributes", 0))
        if current.is_symlink() or attributes & 0x400:
            raise RewardMigrationError(f"{label} path contains a symlink or reparse point")
    path = lexical.resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise RewardMigrationError(f"{label} path escaped the project root") from exc
    if not path.is_file():
        raise RewardMigrationError(f"{label} file is missing: {path}")
    return path


@dataclass(frozen=True, slots=True)
class RewardMigrationEntry:
    v1_concept: str
    disposition: str
    v2_destinations: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "v1_concept": self.v1_concept,
            "disposition": self.disposition,
            "v2_destinations": list(self.v2_destinations),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class RewardMigrationEvidence:
    path: Path
    path_sha256: str
    source_path: Path
    source_sha256: str
    target_path: Path
    target_sha256: str
    entries: tuple[RewardMigrationEntry, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MIGRATION_SCHEMA,
            "migration_path": str(self.path),
            "migration_sha256": self.path_sha256,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "target_path": str(self.target_path),
            "target_sha256": self.target_sha256,
            "legacy_source_preserved_and_inactive": True,
            "active_v2_dense_families": list(DENSE_FAMILIES),
            "v1_concept_count": len(self.entries),
            "all_v1_concepts_explicitly_disposed": True,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def load_reward_migration(
    path: str | Path = DEFAULT_MIGRATION_PATH,
) -> RewardMigrationEvidence:
    selected = Path(path).resolve()
    if not selected.is_file():
        raise RewardMigrationError(f"reward migration file is missing: {selected}")
    migration_bytes = _captured_bytes(selected, label="reward migration")
    try:
        payload = yaml.safe_load(migration_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RewardMigrationError("reward migration is not valid UTF-8") from exc
    root = _mapping(payload, label="reward migration")
    if root.get("schema") != MIGRATION_SCHEMA:
        raise RewardMigrationError("unexpected reward migration schema")

    project_root = Path(__file__).resolve().parents[3]
    source = _mapping(root.get("source"), label="source binding")
    target = _mapping(root.get("target"), label="target binding")
    source_path = _resolve_bound_path(
        source.get("path"), project_root=project_root, label="source reward"
    )
    target_path = _resolve_bound_path(
        target.get("path"), project_root=project_root, label="target reward"
    )
    source_bytes = _captured_bytes(source_path, label="source reward")
    target_bytes = _captured_bytes(target_path, label="target reward")
    try:
        source_decoded = yaml.safe_load(source_bytes.decode("utf-8"))
        target_decoded = yaml.safe_load(target_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RewardMigrationError("bound reward config is not valid UTF-8") from exc
    source_payload = _mapping(source_decoded, label="source reward")
    target_payload = _mapping(target_decoded, label="target reward")
    if (
        source.get("schema") != REWARD_SCHEMA
        or source_payload.get("schema") != REWARD_SCHEMA
        or source.get("required_training_enabled") is not False
        or source_payload.get("training_enabled") is not False
    ):
        raise RewardMigrationError("source binding is not the preserved inactive reward v1")
    if tuple(_mapping(source_payload.get("terms"), label="source reward terms")) != REWARD_TERMS:
        raise RewardMigrationError("source reward concept inventory changed")
    if (
        target.get("schema") != REWARD_SCHEMA_V2
        or target_payload.get("schema") != REWARD_SCHEMA_V2
        or target.get("required_training_enabled") is not True
        or target_payload.get("training_enabled") is not True
    ):
        raise RewardMigrationError("target binding is not the active reward v2")
    if tuple(
        _mapping(target_payload.get("dense_families"), label="target dense families")
    ) != DENSE_FAMILIES:
        raise RewardMigrationError("target reward does not expose exactly five dense families")

    raw_entries = root.get("migrations")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(REWARD_TERMS):
        raise RewardMigrationError("migration must contain one row for every v1 concept")
    entries: list[RewardMigrationEntry] = []
    for index, raw in enumerate(raw_entries):
        row = _mapping(raw, label=f"migration row {index}")
        concept = row.get("v1_concept")
        disposition = row.get("disposition")
        destinations = row.get("v2_destinations")
        rationale = row.get("rationale")
        if concept != REWARD_TERMS[index]:
            raise RewardMigrationError("migration concept order differs from reward v1")
        if disposition not in ALLOWED_DISPOSITIONS:
            raise RewardMigrationError(f"invalid disposition for {concept}")
        if (
            not isinstance(destinations, list)
            or not destinations
            or any(not isinstance(item, str) or not item for item in destinations)
            or len(destinations) != len(set(destinations))
        ):
            raise RewardMigrationError(f"invalid v2 destinations for {concept}")
        allowed_destinations = (
            {f"dense.{family}" for family in DENSE_FAMILIES}
            | EVENT_DESTINATIONS
            | SAFETY_DESTINATIONS
        )
        if any(item not in allowed_destinations for item in destinations):
            raise RewardMigrationError(f"unknown v2 destination for {concept}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise RewardMigrationError(f"migration rationale is missing for {concept}")
        if disposition == "MOVED_TO_EVENT" and any(
            item not in EVENT_DESTINATIONS for item in destinations
        ):
            raise RewardMigrationError(f"event-only disposition is inconsistent for {concept}")
        if disposition == "MOVED_TO_SAFETY_EVENT" and not any(
            item in EVENT_DESTINATIONS for item in destinations
        ):
            raise RewardMigrationError(f"safety disposition lacks an event for {concept}")
        if disposition.startswith("MERGED") and not any(
            item.startswith("dense.") for item in destinations
        ):
            raise RewardMigrationError(f"dense merge lacks a dense destination for {concept}")
        if concept in {"body_collision_penalty", "wheel_only_climb_penalty"} and (
            "event.task_failure" not in destinations
            or "event.safety_abort" in destinations
        ):
            raise RewardMigrationError(
                f"{concept} must remain an authoritative task-failure event"
            )
        if concept in {"fall_penalty", "joint_limit_penalty"} and (
            "event.safety_abort" not in destinations
        ):
            raise RewardMigrationError(
                f"{concept} must remain an explicit safety-abort event"
            )
        entries.append(
            RewardMigrationEntry(
                v1_concept=str(concept),
                disposition=str(disposition),
                v2_destinations=tuple(destinations),
                rationale=rationale.strip(),
            )
        )

    if (
        _captured_bytes(selected, label="reward migration") != migration_bytes
        or _captured_bytes(source_path, label="source reward") != source_bytes
        or _captured_bytes(target_path, label="target reward") != target_bytes
    ):
        raise RewardMigrationError("reward migration inputs changed while validating")

    return RewardMigrationEvidence(
        path=selected,
        path_sha256=_sha256_bytes(migration_bytes),
        source_path=source_path,
        source_sha256=_sha256_bytes(source_bytes),
        target_path=target_path,
        target_sha256=_sha256_bytes(target_bytes),
        entries=tuple(entries),
    )


__all__ = [
    "ALLOWED_DISPOSITIONS",
    "DEFAULT_MIGRATION_PATH",
    "MIGRATION_SCHEMA",
    "RewardMigrationEntry",
    "RewardMigrationError",
    "RewardMigrationEvidence",
    "load_reward_migration",
]
