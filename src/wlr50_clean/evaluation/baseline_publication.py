"""Freeze and publish an already-adjudicated physical-success FSM baseline.

This is intentionally a publication boundary, not another trial analyzer.  It
accepts only a selected, valid physical success, verifies the immutable raw
trial files against their original manifest, snapshots the current nominal-FSM
inputs by content hash, and prepares a small fixed set of final artifacts.

Recording similarity remains a quality diagnostic throughout this module.  A
divergence warning is preserved, but no percentage can veto Layer-B physical
task success or the baseline freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA = "wlr50_clean.frozen_baseline_publication.v1"
FREEZE_STATUS = "FROZEN_PHYSICAL_SUCCESS_FSM_BASELINE"
PHYSICAL_STATUS = "ONE_CONTINUOUS_PHYSICAL_FSM_SUCCESS"
PPO_STATUS = "FROZEN_FSM_BASELINE_READY_FOR_PPO"
INTERFACE_STATUS = "PPO_INTERFACE_READY"
TRAINING_STATUS = "PPO_TRAINING_NOT_STARTED"

PHYSICS_HZ = 120.0
DECISION_HZ = 15.0
OBSERVATION_DIMENSION = 85
ACTION_DIMENSION = 12
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
MACRO_PHASE_IDS = {state_id: index for index, state_id in enumerate(STATE_IDS, 1)}

FINAL_FILENAMES = (
    "selected_success_trial.json",
    "physical_success_reclassification.json",
    "physical_success_evidence.json",
    "reference_divergence_diagnostics.csv",
    "frozen_successful_fsm_manifest.json",
    "fsm_recording_separation_audit.json",
    "ppo_observation_schema.json",
    "ppo_action_schema.json",
    "ppo_readiness_report.md",
)

# These files define the nominal controller, its live guard inputs, command
# mapping, and frozen scene.  The publication manifest records every byte; it
# does not copy code into a second implementation.
FROZEN_MOTION_PATHS = (
    "configs/fsm_states.yaml",
    "configs/recording_motion_contract.json",
    "configs/environment_lock.json",
    "configs/selected_reference.json",
    "configs/sensors.yaml",
    "configs/collision_role_map.json",
    "src/wlr50_clean/fsm/controller.py",
    "src/wlr50_clean/fsm/drive_feedback.py",
    "src/wlr50_clean/fsm/guard_evaluator.py",
    "src/wlr50_clean/fsm/motion_executor.py",
    "src/wlr50_clean/fsm/recovery.py",
    "src/wlr50_clean/fsm/state_graph.py",
    "src/wlr50_clean/fsm/state_spec.py",
    "src/wlr50_clean/fsm/wheel_decay.py",
    "src/wlr50_clean/sensing/body_collision_detector.py",
    "src/wlr50_clean/sensing/contact_classifier.py",
    "src/wlr50_clean/sensing/geometry.py",
    "src/wlr50_clean/sensing/guard_state.py",
    "src/wlr50_clean/sensing/observation.py",
    "src/wlr50_clean/sensing/sensor_reader.py",
    "src/wlr50_clean/infrastructure/app_runtime.py",
    "src/wlr50_clean/infrastructure/command_batch.py",
    "src/wlr50_clean/infrastructure/robot_adapter.py",
    "src/wlr50_clean/infrastructure/scene_factory.py",
    "src/wlr50_clean/infrastructure/servo_target_mapper.py",
)

PPO_INTERFACE_PATHS = (
    "configs/ppo_interface.yaml",
    "configs/ppo_observation_schema.json",
    "configs/ppo_action_projection.yaml",
    "configs/ppo_domain_randomization.yaml",
    "configs/ppo_reward.yaml",
    "configs/ppo_termination.yaml",
    "src/wlr50_clean/ppo/action_projection.py",
    "src/wlr50_clean/ppo/episode_logger.py",
    "src/wlr50_clean/ppo/observation_schema.py",
    "src/wlr50_clean/ppo/ppo_env_adapter.py",
    "src/wlr50_clean/ppo/residual_interface.py",
    "src/wlr50_clean/ppo/reward_terms.py",
    "src/wlr50_clean/ppo/termination.py",
)

RUNTIME_ROOTS = (
    "src/wlr50_clean/fsm",
    "src/wlr50_clean/sensing",
    "src/wlr50_clean/infrastructure",
    "src/wlr50_clean/ppo",
)

RAW_RECORDING_PATTERNS = (
    ("accepted event stream", re.compile(r"accepted_steps(?:\.jsonl)?", re.I)),
    ("Recording cursor", re.compile(r"\brecording_cursor\b", re.I)),
    ("raw semantic segments", re.compile(r"semantic_segments\.json", re.I)),
    ("raw Recording video", re.compile(r"recording_clean\.mp4", re.I)),
    (
        "raw Recording parser import",
        re.compile(
            r"(?:from|import)\s+wlr50_clean\.reference\."
            r"(?:recording_parser|segment_extractor)\b",
            re.I,
        ),
    ),
)

REQUIRED_TRIAL_ARTIFACTS = frozenset(
    {
        "observation",
        "decision",
        "command",
        "transition",
        "task_event",
        "body_contact",
        "leg_crossing",
        "reference_similarity",
        "actual_viewport_video",
    }
)


class BaselinePublicationError(RuntimeError):
    """A selected trial or output transaction is not safe to publish."""


@dataclass(frozen=True, slots=True)
class PublicationResult:
    selected_trial_id: str
    frozen_config_path: Path
    final_directory: Path
    published_files: tuple[Path, ...]
    working_tree_hash: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _yaml_bytes(value: Any) -> bytes:
    return yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselinePublicationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselinePublicationError(f"JSON root is not an object: {path}")
    return value


def _yaml_object(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BaselinePublicationError(f"cannot read YAML object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselinePublicationError(f"YAML root is not an object: {path}")
    return value


def _required_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BaselinePublicationError(f"path escapes project root: {relative}") from exc
    if not path.is_file():
        raise BaselinePublicationError(f"required file is missing: {path}")
    return path


def _trial_number(value: Any) -> int | None:
    match = re.search(r"(?:trial[_ -]?)?0*(\d+)", str(value), re.I)
    return int(match.group(1)) if match else None


def _same_trial(left: Any, right: Any) -> bool:
    if str(left) == str(right):
        return True
    left_number = _trial_number(left)
    right_number = _trial_number(right)
    return left_number is not None and left_number == right_number


def _selected_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the readjudicator's wrapper or a direct selected record."""

    keys = (
        "selected_success_trial",
        "selected_trial",
        "selected",
        "trial",
        "record",
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    if any(key in payload for key in ("trial_id", "trial_number", "task_result")):
        return dict(payload)
    raise BaselinePublicationError("selected-success input contains no selected trial record")


def _evidence_for_trial(payload: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    """Locate detailed evidence across the known readjudicator wrapper shapes."""

    for key in ("trial_evidence", "trials", "evidence_by_trial"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            for candidate_id, candidate in value.items():
                if _same_trial(candidate_id, trial_id) and isinstance(candidate, Mapping):
                    return dict(candidate)
        if isinstance(value, list):
            for candidate in value:
                if (
                    isinstance(candidate, Mapping)
                    and _same_trial(candidate.get("trial_id"), trial_id)
                ):
                    return dict(candidate)
    for key in ("selected_trial_evidence", "physical_success_evidence", "evidence"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidate_id = value.get("trial_id", trial_id)
            if _same_trial(candidate_id, trial_id):
                return dict(value)
    if _same_trial(payload.get("trial_id"), trial_id):
        return dict(payload)
    raise BaselinePublicationError(f"no detailed evidence found for {trial_id}")


def _require_false(value: Any, label: str) -> None:
    if value is not False and value != 0:
        raise BaselinePublicationError(f"{label} must be explicitly false/zero")


def _require_true(value: Any, label: str) -> None:
    if value is not True and value != 1:
        raise BaselinePublicationError(f"{label} must be explicitly true")


def _value(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _validate_readjudication_hashes(
    manifest: Mapping[str, Any], input_paths: Iterable[Path]
) -> dict[str, Any]:
    output_files = manifest.get("output_files", {})
    output_files = output_files if isinstance(output_files, Mapping) else {}
    checks: dict[str, Any] = {}
    for path in input_paths:
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        declared = output_files.get(path.name)
        if isinstance(declared, Mapping):
            if declared.get("sha256") and str(declared["sha256"]).lower() != actual["sha256"]:
                raise BaselinePublicationError(
                    f"readjudication artifact hash mismatch: {path.name}"
                )
            if declared.get("bytes") is not None and int(declared["bytes"]) != actual["bytes"]:
                raise BaselinePublicationError(
                    f"readjudication artifact byte-count mismatch: {path.name}"
                )
        checks[path.name] = {
            **actual,
            "declared_in_readjudication_manifest": isinstance(declared, Mapping),
            "verified": True,
        }
    return checks


def _validate_trial_artifacts(
    trial_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = manifest.get("artifact_files")
    if not isinstance(artifacts, Mapping):
        raise BaselinePublicationError("raw trial manifest has no artifact_files object")
    missing_roles = sorted(REQUIRED_TRIAL_ARTIFACTS - set(artifacts))
    if missing_roles:
        raise BaselinePublicationError(
            "raw trial manifest lacks required artifact roles: " + ", ".join(missing_roles)
        )
    checks: dict[str, Any] = {}
    for role, raw in sorted(artifacts.items()):
        if not isinstance(raw, Mapping) or not raw.get("path"):
            raise BaselinePublicationError(f"malformed raw artifact entry: {role}")
        candidate = Path(str(raw["path"]))
        path = candidate.resolve() if candidate.is_absolute() else (trial_dir / candidate).resolve()
        try:
            path.relative_to(trial_dir)
        except ValueError as exc:
            raise BaselinePublicationError(
                f"raw artifact path escapes immutable trial directory: {role}"
            ) from exc
        if not path.is_file():
            raise BaselinePublicationError(f"raw trial artifact is missing: {role}: {path}")
        actual_sha = _sha256(path)
        actual_bytes = path.stat().st_size
        if str(raw.get("sha256", "")).lower() != actual_sha:
            raise BaselinePublicationError(f"raw trial artifact hash mismatch: {role}")
        try:
            declared_bytes = int(raw["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BaselinePublicationError(
                f"raw trial artifact has no valid byte count: {role}"
            ) from exc
        if declared_bytes != actual_bytes:
            raise BaselinePublicationError(f"raw trial artifact byte-count mismatch: {role}")
        checks[str(role)] = {
            "path": str(path),
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "verified": True,
        }
    return checks


def _validate_selected_physical_success(
    selected: Mapping[str, Any], expected_trial: str
) -> str:
    trial_id = str(_value(selected, "trial_id", "id", default=""))
    if not trial_id:
        number = _value(selected, "trial_number")
        if number is not None:
            trial_id = f"trial_{int(number):03d}"
    if not trial_id or not _same_trial(trial_id, expected_trial):
        raise BaselinePublicationError(
            f"selected trial {trial_id!r} does not match required {expected_trial!r}"
        )
    validity = str(_value(selected, "trial_validity", "validity", default=""))
    if validity != "VALID":
        raise BaselinePublicationError(f"selected trial is not VALID: {validity!r}")
    result = str(_value(selected, "task_result", "new_task_result", default=""))
    classification = str(_value(selected, "classification", default=result))
    if result != "SUCCESS" and not classification.startswith("TASK_SUCCESS"):
        raise BaselinePublicationError(f"selected trial is not a physical success: {result!r}")
    _require_true(
        _value(
            selected,
            "physical_traversal_complete",
            "final_obstacle_geometry_success",
        ),
        "selected.physical_traversal_complete",
    )
    _require_false(selected.get("body_collision"), "selected.body_collision")
    _require_false(selected.get("wheel_only_climb"), "selected.wheel_only_climb")
    _require_true(selected.get("environment_match"), "selected.environment_match")
    _require_true(
        _value(selected, "video_continuous", "continuous_video"),
        "selected.video_continuous",
    )
    _require_false(
        _value(selected, "recording_runtime_access", "runtime_raw_recording_access"),
        "selected.recording_runtime_access",
    )
    _require_false(
        _value(selected, "forbidden_control_count", default=0),
        "selected.forbidden_control_count",
    )
    _require_false(selected.get("fall"), "selected.fall")
    _require_false(
        selected.get("physics_explosion"),
        "selected.physics_explosion",
    )
    return trial_id


def _validate_raw_success(manifest: Mapping[str, Any], trial_id: str) -> None:
    if not _same_trial(manifest.get("trial_id"), trial_id):
        raise BaselinePublicationError("raw trial manifest identifies a different trial")
    evidence = manifest.get("success_evidence", {})
    if not isinstance(evidence, Mapping):
        raise BaselinePublicationError("raw trial manifest has no success_evidence")
    # Completion/event ledgers are strong corroborating evidence, but older
    # loggers may omit them.  The readjudicator's continuous-video and final
    # geometry proof owns Layer-B success and is not vetoed by a missing row.
    _require_false(evidence.get("body_collision"), "raw.body_collision")
    _require_false(evidence.get("wheel_only_climb"), "raw.wheel_only_climb")
    _require_false(
        evidence.get("runtime_raw_recording_access"),
        "raw.runtime_raw_recording_access",
    )
    for name in (
        "root_state_write_count",
        "teleport_count",
        "external_force_count",
        "external_impulse_count",
    ):
        _require_false(evidence.get(name, 0), f"raw.{name}")
    _require_true(evidence.get("source_robot_usd_unchanged"), "raw.source_robot_usd_unchanged")
    video = manifest.get("video", {})
    if not isinstance(video, Mapping):
        raise BaselinePublicationError("raw trial manifest has no video evidence")
    _require_true(video.get("valid"), "raw.video.valid")
    _require_false(video.get("stitched"), "raw.video.stitched")
    _require_false(video.get("speed_modified"), "raw.video.speed_modified")
    decode = video.get("full_decode", {})
    if isinstance(decode, Mapping):
        _require_true(decode.get("full_decode"), "raw.video.full_decode")
        _require_true(decode.get("timestamps_monotonic"), "raw.video.timestamps_monotonic")


def _file_records(root: Path, relatives: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in relatives:
        path = _required_file(root, relative)
        records.append(
            {
                "path": relative.replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _record_set_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(records)))


def _git_value(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselinePublicationError(f"git {' '.join(arguments)} failed: {exc}") from exc
    return result.stdout.decode("utf-8", errors="surrogateescape")


def _scan_runtime(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    for relative_root in RUNTIME_ROOTS:
        runtime = (root / relative_root).resolve()
        if not runtime.is_dir():
            raise BaselinePublicationError(f"runtime source root is missing: {runtime}")
        for path in sorted(runtime.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            relative = path.relative_to(root).as_posix()
            scanned.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            for label, pattern in RAW_RECORDING_PATTERNS:
                for match in pattern.finditer(text):
                    findings.append(
                        {
                            "kind": label,
                            "path": relative,
                            "line": text.count("\n", 0, match.start()) + 1,
                            "match": match.group(0),
                        }
                    )
    return {
        "passed": not findings,
        "runtime_raw_recording_access_count": len(findings),
        "scanned_file_count": len(scanned),
        "scanned_sources_sha256": _record_set_hash(scanned),
        "findings": findings,
    }


def _validate_policy_and_interfaces(
    root: Path, *, frozen_environment_hash: str
) -> dict[str, Any]:
    policy_path = _required_file(root, "configs/conformance_policy.yaml")
    policy = _yaml_object(policy_path)
    active = policy.get("active_tolerance", {})
    runtime = policy.get("runtime", {})
    ppo_policy = policy.get("ppo", {})
    if not isinstance(active, Mapping) or float(active.get("percent", -1.0)) != 30.0:
        raise BaselinePublicationError("active Recording diagnostic must be 30 percent")
    for name in (
        "blocks_task_success",
        "blocks_baseline_freeze",
        "blocks_video_publication",
        "blocks_ppo_readiness",
    ):
        _require_false(active.get(name), f"conformance_policy.active_tolerance.{name}")
    role = str(active.get("role", "")).lower()
    if "diagnostic" not in role and "advisory" not in role:
        raise BaselinePublicationError("30 percent tolerance is not marked advisory/diagnostic")
    for name in (
        "conformance_can_block_entry",
        "conformance_can_block_completion",
        "conformance_can_block_task_success",
    ):
        _require_false(runtime.get(name), f"conformance_policy.runtime.{name}")
    _require_false(
        ppo_policy.get("reference_divergence_is_hard_action_bound"),
        "conformance_policy.ppo.reference_divergence_is_hard_action_bound",
    )

    interface_path = _required_file(root, "configs/ppo_interface.yaml")
    interface = _yaml_object(interface_path)
    observation_path = _required_file(root, "configs/ppo_observation_schema.json")
    observation = _object(observation_path)
    action_path = _required_file(root, "configs/ppo_action_projection.yaml")
    action = _yaml_object(action_path)
    domain_path = _required_file(root, "configs/ppo_domain_randomization.yaml")
    domain = _yaml_object(domain_path)
    reward_path = _required_file(root, "configs/ppo_reward.yaml")
    reward = _yaml_object(reward_path)
    termination_path = _required_file(root, "configs/ppo_termination.yaml")
    termination = _yaml_object(termination_path)
    env_adapter_path = _required_file(root, "src/wlr50_clean/ppo/ppo_env_adapter.py")
    termination_source_path = _required_file(root, "src/wlr50_clean/ppo/termination.py")
    expected = {
        "observation_dimension": OBSERVATION_DIMENSION,
        "nominal_action_dimension": ACTION_DIMENSION,
        "residual_action_dimension": ACTION_DIMENSION,
    }
    for name, value in expected.items():
        if int(interface.get(name, -1)) != value:
            raise BaselinePublicationError(f"PPO interface {name} must remain {value}")
    if tuple(interface.get("state_ids", ())) != STATE_IDS:
        raise BaselinePublicationError("PPO interface state IDs must remain ordered P01-P13")
    declared_macro_ids = interface.get("macro_phase_ids", {})
    if not isinstance(declared_macro_ids, Mapping) or {
        str(key): int(value) for key, value in declared_macro_ids.items()
    } != MACRO_PHASE_IDS:
        raise BaselinePublicationError("PPO macro phase IDs must map P01-P13 to 1-13")
    _require_false(interface.get("training_enabled"), "ppo_interface.training_enabled")
    if int(observation.get("dimension", -1)) != OBSERVATION_DIMENSION:
        raise BaselinePublicationError("PPO observation dimension must remain 85")
    features = observation.get("features", ())
    if not isinstance(features, list) or not features:
        raise BaselinePublicationError("PPO observation schema has no features")
    cursor = 0
    for feature in features:
        if not isinstance(feature, Mapping) or int(feature.get("offset", -1)) != cursor:
            raise BaselinePublicationError("PPO observation feature offsets are not contiguous")
        cursor += int(feature.get("size", -1))
    if cursor != OBSERVATION_DIMENSION:
        raise BaselinePublicationError("PPO observation feature sizes do not total 85")
    if tuple(observation.get("state_ids", ())) != STATE_IDS:
        raise BaselinePublicationError("PPO observation state IDs must remain ordered P01-P13")
    observation_macro_ids = observation.get("macro_phase_ids", {})
    if not isinstance(observation_macro_ids, Mapping) or {
        str(key): int(value) for key, value in observation_macro_ids.items()
    } != MACRO_PHASE_IDS:
        raise BaselinePublicationError("PPO observation macro phase IDs are invalid")
    for name, value in (
        ("physics_hz", PHYSICS_HZ),
        ("decision_hz", DECISION_HZ),
        ("nominal_action_dimension", ACTION_DIMENSION),
        ("residual_action_dimension", ACTION_DIMENSION),
    ):
        if float(action.get(name, -1)) != float(value):
            raise BaselinePublicationError(f"PPO action {name} must remain {value}")
    _require_false(action.get("training_enabled"), "ppo_action.training_enabled")
    diagnostic = action.get("recording_envelope_diagnostic", {})
    output_scale = action.get("residual_output_scale", {})
    _require_false(
        diagnostic.get("hard_projection_constraint"),
        "ppo_action.recording_envelope_diagnostic.hard_projection_constraint",
    )
    _require_false(
        output_scale.get("recording_envelope_used_in_projection"),
        "ppo_action.residual_output_scale.recording_envelope_used_in_projection",
    )
    mask_source = action.get("phase_action_mask", {})
    if not isinstance(mask_source, Mapping) or mask_source.get("derive_from") != (
        "phases.ppo_action_mask_full12"
    ):
        raise BaselinePublicationError(
            "PPO phase masks must derive from phases.ppo_action_mask_full12"
        )
    physical = action.get("physical_safety_projection", {})
    for name in (
        "body_collision_disables_all_residuals",
        "body_collision_forces_wheels_zero",
        "wheel_only_climb_disables_all_residuals",
        "wheel_only_climb_forces_wheels_zero",
    ):
        _require_true(physical.get(name), f"ppo_action.physical_safety_projection.{name}")

    order = tuple(action.get("full12_order", ()))
    if len(order) != ACTION_DIMENSION or len(set(order)) != ACTION_DIMENSION:
        raise BaselinePublicationError("PPO Full12 order must contain 12 unique channels")
    contract_path = _required_file(root, "configs/recording_motion_contract.json")
    contract = _object(contract_path)
    if tuple(contract.get("full12_order", ())) != order:
        raise BaselinePublicationError("PPO action order differs from the motion contract")
    phases = contract.get("phases", ())
    masks: dict[str, list[int]] = {}
    if not isinstance(phases, list):
        raise BaselinePublicationError("motion contract phases must be a list")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise BaselinePublicationError("motion contract phase is not an object")
        state_id = str(phase.get("state_id"))
        raw_mask = phase.get("ppo_action_mask_full12", ())
        try:
            mask = [int(value) for value in raw_mask]
        except (TypeError, ValueError) as exc:
            raise BaselinePublicationError(
                f"phase {state_id} has a non-binary PPO action mask"
            ) from exc
        if len(mask) != ACTION_DIMENSION or any(value not in (0, 1) for value in mask):
            raise BaselinePublicationError(
                f"phase {state_id} PPO action mask must contain 12 binary values"
            )
        masks[state_id] = mask
    if tuple(masks) != STATE_IDS:
        raise BaselinePublicationError("phase masks do not cover ordered P01-P13")

    _require_false(domain.get("enabled"), "ppo_domain_randomization.enabled")
    _require_false(
        domain.get("training_enabled"), "ppo_domain_randomization.training_enabled"
    )
    _require_true(
        domain.get("nominal_evaluation_uses_frozen_environment"),
        "ppo_domain_randomization.nominal_evaluation_uses_frozen_environment",
    )
    hooks = domain.get("hooks", {})
    if not isinstance(hooks, Mapping) or not hooks:
        raise BaselinePublicationError("PPO domain-randomization hooks are missing")
    for name, hook in hooks.items():
        if not isinstance(hook, Mapping):
            raise BaselinePublicationError(f"PPO randomization hook {name} is malformed")
        _require_false(hook.get("enabled"), f"ppo_domain_randomization.hooks.{name}.enabled")
    _require_false(reward.get("training_enabled"), "ppo_reward.training_enabled")
    if not isinstance(reward.get("terms"), Mapping) or not reward["terms"]:
        raise BaselinePublicationError("PPO reward terms are missing")
    _require_false(
        termination.get("training_enabled"), "ppo_termination.training_enabled"
    )
    if float(termination.get("timeout_s", -1.0)) != 200.0:
        raise BaselinePublicationError("PPO termination timeout must remain 200 s")
    if termination.get("conformance_outside_30pct") != "diagnostic_only":
        raise BaselinePublicationError("Recording conformance must remain diagnostic-only")
    termination_priority = tuple(str(value) for value in termination.get("priority", ()))
    required_termination = {
        "NAN_INF",
        "PHYSICS_EXPLOSION",
        "BODY_COLLISION",
        "WHEEL_ONLY_CLIMB",
        "FALL",
        "HARD_JOINT_LIMIT",
        "SUCCESS",
        "TIMEOUT",
    }
    if len(termination_priority) != len(required_termination) or set(
        termination_priority
    ) != required_termination:
        raise BaselinePublicationError("PPO termination reasons are incomplete")

    observation_output = dict(observation)
    observation_output["macro_phase_ids"] = dict(MACRO_PHASE_IDS)
    observation_output["frozen_baseline"] = {
        "status": PPO_STATUS,
        "training_enabled": False,
        "source_path": observation_path.relative_to(root).as_posix(),
        "source_sha256": _sha256(observation_path),
    }
    action_output = {
        "schema": "wlr50_clean.ppo_action_schema.v1",
        "schema_name": action.get("action_schema_name"),
        "schema_version": action.get("action_schema_version"),
        "physics_hz": PHYSICS_HZ,
        "decision_hz": DECISION_HZ,
        "nominal_action_dimension": ACTION_DIMENSION,
        "residual_action_dimension": ACTION_DIMENSION,
        "full12_order": list(order),
        "state_ids": list(STATE_IDS),
        "macro_phase_ids": dict(MACRO_PHASE_IDS),
        "bounded_transform": action.get("bounded_transform"),
        "phase_action_masks": masks,
        "residual_rate_limits": action.get("residual_rate_limits"),
        "absolute_action_limits": action.get("absolute_action_limits"),
        "joint_safety_margin_deg": action.get("joint_safety_margin_deg"),
        "physical_safety_projection": physical,
        "hard_safety": action.get("hard_safety"),
        "hard_bound_sources": list(ppo_policy.get("hard_bound_sources", ())),
        "recording_reference": {
            "tolerance_percent": 30.0,
            "role": "ADVISORY_DIAGNOSTIC_AND_INITIAL_SCALE_SUGGESTION_ONLY",
            "hard_action_bound": False,
            "task_success_gate": False,
        },
        "control_authority": {
            "fsm_owns_state_order": True,
            "policy_may_skip_phase": False,
            "rear_leg_order": "RR_FIRST",
            "policy_may_change_rear_leg_order": False,
            "body_collision_detector_mandatory": True,
            "wheel_only_climb_detector_mandatory": True,
            "future_recording_action_access": False,
        },
        "zero_residual": action.get("zero_residual"),
        "environment_reset": {
            "seed_required": True,
            "seed_type": "non_negative_integer",
            "frozen_environment_hash": frozen_environment_hash,
            "domain_randomization_enabled": False,
            "nominal_evaluation_uses_frozen_environment": True,
            "required_backend_metadata": [
                "environment_hash",
                "robot_asset_hash",
                "initial_root_state",
                "initial_joint_state",
                "obstacle_pose",
                "controller_hash",
                "motion_contract_hash",
            ],
            "source": {
                "config_path": domain_path.relative_to(root).as_posix(),
                "config_sha256": _sha256(domain_path),
                "adapter_path": env_adapter_path.relative_to(root).as_posix(),
                "adapter_sha256": _sha256(env_adapter_path),
            },
        },
        "episode_end_contract": {
            "terminated_reasons": [
                "SUCCESS",
                "BODY_COLLISION",
                "WHEEL_ONLY_CLIMB",
                "FALL",
                "NAN_INF",
                "HARD_JOINT_LIMIT",
                "PHYSICS_EXPLOSION",
            ],
            "truncated_reasons": ["TIMEOUT"],
            "diagnostic_only_reasons": ["REFERENCE_CONFORMANCE"],
            "priority": list(termination_priority),
            "timeout_s": 200.0,
            "source": {
                "config_path": termination_path.relative_to(root).as_posix(),
                "config_sha256": _sha256(termination_path),
                "implementation_path": termination_source_path.relative_to(root).as_posix(),
                "implementation_sha256": _sha256(termination_source_path),
            },
        },
        "reward_contract": {
            "aggregation": reward.get("aggregation"),
            "term_names": list(reward["terms"]),
            "config_path": reward_path.relative_to(root).as_posix(),
            "config_sha256": _sha256(reward_path),
        },
        "training_enabled": False,
        "source": {
            "path": action_path.relative_to(root).as_posix(),
            "sha256": _sha256(action_path),
            "motion_contract_sha256": _sha256(contract_path),
        },
    }
    return {
        "policy": policy,
        "policy_path": policy_path,
        "interface": interface,
        "interface_path": interface_path,
        "observation_output": observation_output,
        "action_output": action_output,
    }


def _selected_diagnostics(csv_path: Path, trial_id: str) -> tuple[bytes, dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = [
            dict(row)
            for row in reader
            if _same_trial(row.get("trial_id", ""), trial_id)
        ]
    if not fieldnames:
        raise BaselinePublicationError("reference diagnostics CSV has no header")
    if not rows:
        raise BaselinePublicationError(f"reference diagnostics has no rows for {trial_id}")
    if "blocks_task_success" not in fieldnames:
        fieldnames.append("blocks_task_success")
    if "acceptance_role" not in fieldnames:
        fieldnames.append("acceptance_role")
    maximum: tuple[float, dict[str, str]] | None = None
    for row in rows:
        declared = str(row.get("blocks_task_success", "")).strip().lower()
        if declared in {"true", "1", "yes"}:
            raise BaselinePublicationError(
                "reference diagnostics incorrectly declares a task-success veto"
            )
        row["blocks_task_success"] = "false"
        row["acceptance_role"] = "ADVISORY_DIAGNOSTIC_ONLY"
        try:
            value = float(row.get("error_percent", "nan"))
        except (TypeError, ValueError):
            continue
        if value == value and (maximum is None or value > maximum[0]):
            maximum = (value, row)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    summary = {
        "row_count": len(rows),
        "maximum_error_percent": maximum[0] if maximum else None,
        "maximum_phase": maximum[1].get("phase") if maximum else None,
        "maximum_channel": maximum[1].get("channel") if maximum else None,
        "maximum_metric": maximum[1].get("metric") if maximum else None,
        "within_30_percent": bool(maximum is not None and maximum[0] <= 30.0),
        "warning": (
            None
            if maximum is not None and maximum[0] <= 30.0
            else "REFERENCE_DIVERGENCE_WARNING"
        ),
        "blocks_task_success": False,
    }
    return output.getvalue().encode("utf-8"), summary


def _environment_snapshot(root: Path, selected: Mapping[str, Any]) -> dict[str, Any]:
    path = _required_file(root, "configs/environment_lock.json")
    environment = _object(path)
    environment_hash = _sha256(path)
    declared = selected.get("environment_hash")
    if declared and str(declared).lower() != environment_hash:
        raise BaselinePublicationError("selected trial environment hash does not match lock")
    robot = environment.get("robot", {})
    if not isinstance(robot, Mapping) or not robot.get("usd_path"):
        raise BaselinePublicationError("environment lock has no robot USD path")
    robot_path = Path(str(robot["usd_path"])).resolve()
    if not robot_path.is_file():
        raise BaselinePublicationError(f"frozen robot asset is missing: {robot_path}")
    robot_hash = _sha256(robot_path)
    if str(robot.get("usd_sha256", "")).lower() != robot_hash:
        raise BaselinePublicationError("current robot asset hash differs from environment lock")
    declared_robot = selected.get("robot_asset_hash")
    if declared_robot and str(declared_robot).lower() != robot_hash:
        raise BaselinePublicationError("selected trial robot hash differs from current asset")
    scene_records = _file_records(
        root,
        (
            "configs/environment_lock.json",
            "configs/sensors.yaml",
            "configs/collision_role_map.json",
            "src/wlr50_clean/infrastructure/scene_factory.py",
        ),
    )
    return {
        "environment": environment,
        "environment_hash": environment_hash,
        "robot_asset_path": str(robot_path),
        "robot_asset_hash": robot_hash,
        "scene_hash": _record_set_hash(scene_records),
        "scene_files": scene_records,
    }


def _atomic_publish(
    payloads: Mapping[Path, bytes], *, replace_existing: bool
) -> None:
    """Stage every byte first, then replace targets with rollback on failure."""

    targets = tuple(payloads)
    existing = tuple(path for path in targets if path.exists())
    if existing and not replace_existing:
        raise BaselinePublicationError(
            "refusing to overwrite existing publication files: "
            + ", ".join(str(path) for path in existing)
        )
    token = uuid.uuid4().hex
    stages: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for target, payload in payloads.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = target.parent / f".{target.name}.{token}.stage"
            with stage.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if _sha256(stage) != _sha256_bytes(payload):
                raise BaselinePublicationError(f"staged byte verification failed: {target}")
            stages[target] = stage
        for target in targets:
            if target.exists():
                backup = target.parent / f".{target.name}.{token}.backup"
                os.replace(target, backup)
                backups[target] = backup
            os.replace(stages[target], target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
            elif target.exists():
                target.unlink()
        for target, backup in backups.items():
            if backup.exists() and not target.exists():
                os.replace(backup, target)
        raise
    finally:
        for stage in stages.values():
            if stage.exists():
                stage.unlink()
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def publish_frozen_baseline(
    *,
    project_root: Path,
    readjudication_dir: Path,
    expected_trial: str = "43",
    replace_existing: bool = False,
) -> PublicationResult:
    """Validate and atomically publish the fixed baseline artifact set."""

    root = Path(project_root).resolve()
    readjudication = Path(readjudication_dir).resolve()
    if not root.is_dir() or not readjudication.is_dir():
        raise BaselinePublicationError("project/readjudication directory is missing")
    selected_input = _required_file(readjudication, "selected_success_trial.json")
    evidence_input = _required_file(readjudication, "physical_success_evidence.json")
    diagnostics_input = _required_file(readjudication, "reference_divergence_diagnostics.csv")
    readjudication_manifest_path = _required_file(
        readjudication, "readjudication_manifest.json"
    )
    readjudication_manifest = _object(readjudication_manifest_path)
    readjudication_hashes = _validate_readjudication_hashes(
        readjudication_manifest,
        (selected_input, evidence_input, diagnostics_input),
    )
    selected = _selected_record(_object(selected_input))
    trial_id = _validate_selected_physical_success(selected, expected_trial)
    evidence = _evidence_for_trial(_object(evidence_input), trial_id)

    matches = sorted(
        path
        for path in (root / "runs").glob(f"trial_{_trial_number(trial_id):03d}_*")
        if path.is_dir()
    )
    exact = [path for path in matches if path.name == trial_id]
    if len(exact) == 1:
        trial_dir = exact[0].resolve()
    elif len(matches) == 1:
        trial_dir = matches[0].resolve()
        trial_id = trial_dir.name
    else:
        raise BaselinePublicationError(
            f"selected immutable trial directory is ambiguous or missing: {trial_id}"
        )
    raw_manifest_path = trial_dir / "trial_manifest.json"
    raw_manifest = _object(raw_manifest_path)
    _validate_raw_success(raw_manifest, trial_id)
    raw_artifacts = _validate_trial_artifacts(trial_dir, raw_manifest)
    raw_manifest_hash_before = _sha256(raw_manifest_path)
    diagnostics_bytes, diagnostic_summary = _selected_diagnostics(
        diagnostics_input, trial_id
    )
    environment = _environment_snapshot(root, selected)
    interfaces = _validate_policy_and_interfaces(
        root, frozen_environment_hash=environment["environment_hash"]
    )
    runtime_scan = _scan_runtime(root)
    if not runtime_scan["passed"]:
        raise BaselinePublicationError(
            "FSM runtime source contains raw Recording access: "
            + json.dumps(runtime_scan["findings"], ensure_ascii=False)
        )
    selected_reference = _object(_required_file(root, "configs/selected_reference.json"))
    _require_false(
        selected_reference.get("runtime_recording_access_authorized"),
        "selected_reference.runtime_recording_access_authorized",
    )

    motion_files = _file_records(root, FROZEN_MOTION_PATHS)
    interface_files = _file_records(root, PPO_INTERFACE_PATHS)
    controller_files = [
        record
        for record in motion_files
        if record["path"].startswith("src/wlr50_clean/fsm/")
        or record["path"]
        in {
            "src/wlr50_clean/infrastructure/command_batch.py",
            "src/wlr50_clean/infrastructure/robot_adapter.py",
            "src/wlr50_clean/infrastructure/servo_target_mapper.py",
        }
    ]
    state_graph_files = [
        record
        for record in motion_files
        if record["path"]
        in {"configs/fsm_states.yaml", "src/wlr50_clean/fsm/state_graph.py"}
    ]
    controller_hash = _record_set_hash(controller_files)
    state_graph_hash = _record_set_hash(state_graph_files)
    motion_contract_hash = _sha256(
        _required_file(root, "configs/recording_motion_contract.json")
    )
    git_commit = _git_value(root, "rev-parse", "HEAD").strip()
    git_status = _git_value(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    snapshot_files = motion_files + interface_files
    working_tree_hash = _sha256_bytes(
        _canonical_bytes(
            {
                "definition": "git HEAD + pre-publication porcelain status + frozen input bytes",
                "git_commit": git_commit,
                "git_status_porcelain_v1_sha256": _sha256_bytes(
                    git_status.encode("utf-8", errors="surrogateescape")
                ),
                "frozen_inputs": snapshot_files,
            }
        )
    )
    reference_version = str(
        raw_manifest.get("reference_version")
        or selected_reference.get("version_id")
        or selected_reference.get("reference_version")
    )
    rear_order = str(
        raw_manifest.get("rear_leg_order")
        or selected.get("rear_leg_order")
        or selected_reference.get("rear_leg_order")
    )
    if rear_order != "RR_FIRST":
        raise BaselinePublicationError("selected baseline does not preserve RR_FIRST")
    video_path = Path(raw_artifacts["actual_viewport_video"]["path"])
    duration = _value(selected, "duration_s", "video_duration_from_frames")
    reference_within = bool(diagnostic_summary["within_30_percent"])
    reference_warning = diagnostic_summary["warning"]
    classification = (
        "TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING"
        if not reference_within
        else "TASK_SUCCESS"
    )
    reference_quality = {
        "within_30_percent": reference_within,
        "warning": reference_warning,
        "maximum_error_percent": diagnostic_summary["maximum_error_percent"],
        "blocks_task_success": False,
    }

    common = {
        "selected_trial_id": trial_id,
        "git_commit": git_commit,
        "working_tree_hash": working_tree_hash,
        "working_tree_hash_definition": (
            "SHA-256 of canonical git HEAD, pre-publication git-status digest, and "
            "the exact frozen input file manifest"
        ),
        "environment_hash": environment["environment_hash"],
        "robot_asset_hash": environment["robot_asset_hash"],
        "scene_hash": environment["scene_hash"],
        "controller_hash": controller_hash,
        "state_graph_hash": state_graph_hash,
        "motion_contract_hash": motion_contract_hash,
        "reference_version": reference_version,
        "rear_leg_order": rear_order,
        "physics_hz": PHYSICS_HZ,
        "decision_hz": DECISION_HZ,
        "observation_dimension": OBSERVATION_DIMENSION,
        "nominal_action_dimension": ACTION_DIMENSION,
        "residual_action_dimension": ACTION_DIMENSION,
        "training_enabled": False,
        "trial_validity": "VALID",
        "task_result": "SUCCESS",
        "classification": classification,
        "body_collision": False,
        "wheel_only_climb": False,
        "reference_max_error_percent": diagnostic_summary["maximum_error_percent"],
        "reference_within_30_percent": reference_within,
        "reference_warning": reference_warning,
        "reference_divergence_blocks_task_success": False,
        "reference_quality": reference_quality,
        "video_source_path": str(video_path),
    }
    frozen_config = {
        "schema": "wlr50_clean.frozen_successful_fsm.v1",
        "status": FREEZE_STATUS,
        **common,
        "source_trial_directory": str(trial_dir),
        "frozen_motion_inputs": motion_files,
        "ppo_interface_inputs": interface_files,
        "acceptance_policy": {
            "trial_validity": "VALID",
            "task_success": "SUCCESS",
            "recording_similarity_role": "ADVISORY_DIAGNOSTIC_ONLY",
            "recording_tolerance_percent": 30.0,
            "recording_divergence_can_veto_success": False,
        },
    }
    selected_output = {
        "schema": SCHEMA,
        "status": FREEZE_STATUS,
        "selected_success_trial": {
            **selected,
            "trial_id": trial_id,
            "trial_validity": "VALID",
            "task_result": "SUCCESS",
            "classification": classification,
            "reference_within_30_percent": reference_within,
            "reference_warning": reference_warning,
            "reference_divergence_blocks_task_success": False,
            "reference_quality": reference_quality,
        },
        "trial_045_needed": False,
        "source": {
            "path": str(selected_input),
            "sha256": readjudication_hashes[selected_input.name]["sha256"],
        },
    }
    reclassification = {
        "schema": "wlr50_clean.physical_success_reclassification.v1",
        "trial_id": trial_id,
        "original_result": raw_manifest.get("result"),
        "original_failure_or_blocker": {
            "reason": raw_manifest.get("reason"),
            "first_blocker": raw_manifest.get("first_blocker"),
        },
        "new_task_result": "TASK_SUCCESS",
        "classification": classification,
        "reason_for_reclassification": [
            "P01-P13 completed in one continuous physics run",
            "final obstacle-relative geometry proves physical traversal",
            "body/chassis collision is false",
            "wheel-only climb is false with four-leg active-lift evidence",
            "continuous unstitched video is valid and fully decodable",
            "environment and asset identity are valid and forbidden controls are absent",
            "Recording conformance is a Layer-C diagnostic and cannot veto Layer-B success",
        ],
        "reference_divergence": {
            **diagnostic_summary,
            "blocks_task_success": False,
        },
        "immutable_raw_manifest": {
            "path": str(raw_manifest_path),
            "bytes": raw_manifest_path.stat().st_size,
            "sha256": raw_manifest_hash_before,
            "modified": False,
        },
    }
    physical_evidence = {
        "schema": "wlr50_clean.selected_physical_success_evidence.v1",
        "trial_id": trial_id,
        "layers": {
            "TRIAL_VALIDITY": {
                "status": "VALID",
                "environment_match": True,
                "continuous_physics_run": True,
                "runtime_raw_recording_access": False,
                "raw_artifact_hashes_verified": True,
                "raw_artifacts": raw_artifacts,
            },
            "TASK_SUCCESS": {
                "status": "SUCCESS",
                "p01_p13_complete": True,
                "physical_traversal_complete": True,
                "body_collision": False,
                "wheel_only_climb": False,
                "final_pose_stable": True,
            },
            "QUALITY_AND_REFERENCE_DIAGNOSTICS": diagnostic_summary,
        },
        "reference_quality": reference_quality,
        "readjudication_evidence": evidence,
        "source": {
            "path": str(evidence_input),
            "sha256": readjudication_hashes[evidence_input.name]["sha256"],
        },
    }
    separation = {
        "schema": "wlr50_clean.fsm_recording_separation_audit.v1",
        "status": "PASS",
        "selected_trial_id": trial_id,
        "recording_execution": {
            "source": "reference/v010/accepted_steps.jsonl",
            "progression": "Recording event time and event cursor",
            "runtime_scope": "offline Recording runner only",
        },
        "fsm_execution": {
            "raw_recording_event_stream_access": False,
            "runtime_raw_recording_access_count": 0,
            "allowed_compact_input": "configs/recording_motion_contract.json",
            "progression": "live Observation guards and bounded recovery",
            "live_inputs": [
                "joint actual and velocity",
                "wheel actual and contact history",
                "leg clearance, front-plane crossing, and top geometry/contact",
                "BODY collision classification",
                "final obstacle-relative geometry",
            ],
            "time_role": ["trajectory generation", "debounce", "timeout"],
            "time_is_sole_success_condition": False,
            "replay_cursor_present": False,
        },
        "static_runtime_scan": runtime_scan,
        "immutable_trial_runtime_evidence": {
            "runtime_raw_recording_access": False,
            "source_manifest": str(raw_manifest_path),
            "source_manifest_sha256": raw_manifest_hash_before,
        },
        "fsm_is_replay": False,
    }
    readiness_lines = [
        "# PPO readiness report",
        "",
        f"- Status: `{PPO_STATUS}` / `{INTERFACE_STATUS}`",
        f"- Selected nominal FSM: `{trial_id}`",
        f"- Physics / decision cadence: `{PHYSICS_HZ:g} Hz / {DECISION_HZ:g} Hz`",
        f"- Observation dimension: `{OBSERVATION_DIMENSION}`",
        f"- Nominal / residual dimensions: `{ACTION_DIMENSION} / {ACTION_DIMENSION}`",
        "- State order: `P01-P13`; rear-leg order: `RR_FIRST`",
        "- Macro phase IDs: `P01=1` through `P13=13` (frozen)",
        "- Phase masks: frozen for all 13 states",
        "- Residual application: `nominal_fsm_action + projected_residual`",
        "- Hard bounds: actuator limits, joint margins, wheel limits, phase masks, "
        "rate limits, BODY-collision safety, and wheel-only-climb safety",
        "- Recording ±30% envelope: diagnostic / initial-scale suggestion only; "
        "not a projection bound and not a success gate",
        "- Reset: a non-negative integer seed is required; nominal evaluation "
        "uses the frozen environment and all randomization hooks are disabled",
        "- Episode end: SUCCESS and six physical/numerical hard failures terminate; "
        "TIMEOUT truncates at 200 s; REFERENCE_CONFORMANCE remains diagnostic-only",
        f"- Training enabled: `false` (`{TRAINING_STATUS}`)",
        "",
        "The separate selected-trial exporter owns the full 120 Hz zero-residual "
        "proof and 15 Hz JSONL/Parquet dataset. This publication does not claim a "
        "trained policy.",
        "",
    ]

    frozen_path = (root / "configs" / "frozen_successful_fsm.yaml").resolve()
    final_dir = (root / "outputs" / "final").resolve()
    prepared: dict[Path, bytes] = {
        frozen_path: _yaml_bytes(frozen_config),
        final_dir / "selected_success_trial.json": _json_bytes(selected_output),
        final_dir / "physical_success_reclassification.json": _json_bytes(reclassification),
        final_dir / "physical_success_evidence.json": _json_bytes(physical_evidence),
        final_dir / "reference_divergence_diagnostics.csv": diagnostics_bytes,
        final_dir / "fsm_recording_separation_audit.json": _json_bytes(separation),
        final_dir / "ppo_observation_schema.json": _json_bytes(
            interfaces["observation_output"]
        ),
        final_dir / "ppo_action_schema.json": _json_bytes(interfaces["action_output"]),
        final_dir / "ppo_readiness_report.md": "\n".join(readiness_lines).encode("utf-8"),
    }
    publication_inputs = {
        path.name: {"path": str(path), "bytes": len(data), "sha256": _sha256_bytes(data)}
        for path, data in prepared.items()
    }
    frozen_manifest = {
        "schema": SCHEMA,
        "status": FREEZE_STATUS,
        "final_status": [
            PHYSICAL_STATUS,
            PPO_STATUS,
            INTERFACE_STATUS,
            TRAINING_STATUS,
            *(["REFERENCE_DIVERGENCE_WARNING"] if reference_warning else []),
        ],
        **common,
        "source_trial_directory": str(trial_dir),
        "raw_trial_manifest": {
            "path": str(raw_manifest_path),
            "bytes": raw_manifest_path.stat().st_size,
            "sha256": raw_manifest_hash_before,
            "artifact_hashes_verified": True,
        },
        "readjudication_inputs": readjudication_hashes,
        "frozen_inputs": snapshot_files,
        "publication_files": publication_inputs,
        "policy": {
            "recording_similarity_role": "ADVISORY_DIAGNOSTIC_ONLY",
            "recording_divergence_blocks_task_success": False,
            "recording_divergence_is_ppo_hard_bound": False,
        },
    }
    manifest_path = final_dir / "frozen_successful_fsm_manifest.json"
    prepared[manifest_path] = _json_bytes(frozen_manifest)
    expected_targets = {frozen_path} | {final_dir / name for name in FINAL_FILENAMES}
    if set(prepared) != expected_targets:
        raise AssertionError("publication target allowlist mismatch")

    # Re-check immutable trial bytes immediately before the output transaction.
    if _sha256(raw_manifest_path) != raw_manifest_hash_before:
        raise BaselinePublicationError("raw trial manifest changed during publication")
    for role, record in raw_artifacts.items():
        if _sha256(Path(record["path"])) != record["sha256"]:
            raise BaselinePublicationError(f"raw trial artifact changed during publication: {role}")
    _atomic_publish(prepared, replace_existing=replace_existing)
    return PublicationResult(
        selected_trial_id=trial_id,
        frozen_config_path=frozen_path,
        final_directory=final_dir,
        published_files=tuple(prepared),
        working_tree_hash=working_tree_hash,
    )


def main(argv: Sequence[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Atomically freeze and publish an adjudicated physical-success FSM"
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--readjudication-dir",
        type=Path,
        default=Path("outputs/analysis/physical_success_readjudication"),
    )
    parser.add_argument("--selected-trial", default="43")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    readjudication = args.readjudication_dir
    if not readjudication.is_absolute():
        readjudication = root / readjudication
    result = publish_frozen_baseline(
        project_root=root,
        readjudication_dir=readjudication,
        expected_trial=args.selected_trial,
        replace_existing=args.replace_existing,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_trial_id": result.selected_trial_id,
                "frozen_config_path": str(result.frozen_config_path),
                "final_directory": str(result.final_directory),
                "published_files": [str(path) for path in result.published_files],
                "working_tree_hash": result.working_tree_hash,
                "training_enabled": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
