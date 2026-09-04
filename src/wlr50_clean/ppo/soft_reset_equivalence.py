"""Evidence and validation for single-scene soft-reset equivalence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_projection import ZeroResidualEpisodeAuditor, full12_bytes


SOFT_RESET_ACCEPTANCE_SCHEMA = "wlr50_clean.soft_reset_equivalence.v3"
SOFT_RESET_ACCEPTANCE_FILENAME = "soft_reset_equivalence_acceptance.json"
PHASE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
ZERO_FULL12 = (0.0,) * 12
TRACE_FIELDS = (
    "decision_index",
    "physics_tick",
    "sim_time_s",
    "state_id",
    "lifecycle",
    "phase_progress",
    "physics_ticks_executed",
    "actor_observation_v2_dimension",
    "actor_observation_v2_sha256",
    "nominal_full12",
    "residual_full12",
    "applied_full12",
    "controller_task_result",
    "termination_reason",
)
ACCEPTANCE_CHECK_NAMES = (
    "one_backend_instance_two_episodes",
    "both_authoritative_success",
    "both_complete_p01_p13",
    "both_no_body_collision",
    "both_no_wheel_only_climb",
    "both_no_safety_abort",
    "both_under_maximum_duration",
    "both_zero_residual_bitwise_all_ticks",
    "decision_counts_match_compact_traces",
    "physics_tick_counts_match_audits",
    "no_runtime_recording_access",
    "no_in_episode_root_writes",
    "reset_metadata_equivalent",
    "deterministic_trace_equal_through_p10",
    "deterministic_trace_equal_whole_episode",
)
RESET_METADATA_FIELDS = (
    "environment_hash",
    "robot_asset_hash",
    "canonical_reset_state_source",
    "canonical_reset_state_sha256",
    "canonical_reset_state_instance_count",
    "canonical_reset_restore_applied",
    "canonical_reset_applied_sha256",
    "pre_limit_native_state_observed_sha256",
    "pre_limit_native_state_instance_count",
    "pre_limit_native_state_matches_canonical",
    "pre_settle_native_state_observed_sha256",
    "pre_settle_native_state_matches_canonical",
    "adapter_standing_pose_deg",
    "canonical_settled_state_source",
    "canonical_settled_state_sha256",
    "canonical_settled_restore_applied",
    "canonical_settled_applied_sha256",
    "observed_settled_state_sha256",
    "physics_lifecycle_reset",
    "reset_contact_sensor_count",
    "reset_initialization_order",
    "pre_physics_session_limit_state_sha256",
    "pre_physics_composed_limit_state_sha256",
    "pre_physics_composed_limit_state_matches_canonical",
    "session_limit_specs_present_during_physics_reset",
    "session_limit_specs_removed_before_reset",
    "removed_session_limit_state_sha256",
    "post_author_session_limit_state_sha256",
    "post_author_session_limit_state_matches_canonical",
    "session_limit_specs_after_authoring",
    "environment_initialization",
    "controller_hash",
    "motion_contract_hash",
    "seed",
    "reset_count",
    "reset_options",
    "initial_root_state",
    "initial_joint_state",
    "obstacle_pose",
    "level_reference_orientation_wxyz",
    "reset_root_pose_writes",
    "reset_root_velocity_writes",
    "reset_joint_state_writes",
    "reset_global_simulation_resets",
    "reset_simulation_forward_syncs",
    "settle_ticks",
    "randomization_enabled",
    "raw_recording_access",
    "locked_scene_snapshot",
)
SOFT_RESET_CONTRACT_RELATIVE_PATHS = (
    "artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json",
    "configs/environment_lock.json",
    "configs/fsm_states.yaml",
    "configs/recording_motion_contract.json",
    "configs/ppo_interface_v2.yaml",
    "configs/ppo_phase_action_masks_v2.yaml",
    "configs/ppo_observation_schema_v2.json",
    "configs/ppo_termination_v2.yaml",
    "src/wlr50_clean/ppo/action_projection.py",
    "src/wlr50_clean/ppo/isaac_fsm_backend.py",
    "src/wlr50_clean/ppo/phase_action_masks_v2.py",
    "src/wlr50_clean/ppo/observation_schema_v2.py",
    "src/wlr50_clean/ppo/residual_direct_env.py",
    "src/wlr50_clean/ppo/soft_reset_equivalence.py",
)


class SoftResetEquivalenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_row_bytes(row: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SoftResetEquivalenceError(f"compact trace is not canonical JSON: {exc}") from exc


class CompactZeroResidualTickAudit:
    """Streaming 120 Hz action proof without retaining raw sensor frames."""

    def __init__(self) -> None:
        self._actions = ZeroResidualEpisodeAuditor()
        self.tick_count = 0
        self.raw_zero_tick_count = 0
        self.projected_zero_tick_count = 0
        self.zero_fast_path_tick_count = 0
        self._phase_ids: list[str] = []
        self._tick_count_by_phase = {phase_id: 0 for phase_id in PHASE_IDS}

    def _observe_phase(self, frame: Any, *, count_tick: bool) -> None:
        state_id = str(getattr(frame, "state_id", ""))
        if state_id not in PHASE_IDS:
            raise SoftResetEquivalenceError(
                f"tick audit observed invalid controller phase {state_id!r}"
            )
        if not self._phase_ids or self._phase_ids[-1] != state_id:
            self._phase_ids.append(state_id)
        if count_tick:
            self._tick_count_by_phase[state_id] += 1

    def append(self, source: Any, current: Any, projection: Any) -> None:
        self._observe_phase(source, count_tick=True)
        self._observe_phase(current, count_tick=False)
        self._actions.append(
            source.nominal_action_full12, projection.applied_action_full12
        )
        zero_bytes = full12_bytes(ZERO_FULL12)
        if full12_bytes(projection.raw_residual_full12) == zero_bytes:
            self.raw_zero_tick_count += 1
        if full12_bytes(projection.safe_projected_residual_full12) == zero_bytes:
            self.projected_zero_tick_count += 1
        if bool(projection.zero_residual_fast_path):
            self.zero_fast_path_tick_count += 1
        self.tick_count += 1

    def finalize(self) -> dict[str, Any]:
        action = self._actions.finalize()
        passed = bool(
            self.tick_count > 0
            and action.bitwise_equal
            and self.raw_zero_tick_count == self.tick_count
            and self.projected_zero_tick_count == self.tick_count
            and self.zero_fast_path_tick_count == self.tick_count
        )
        return {
            **asdict(action),
            "raw_zero_tick_count": self.raw_zero_tick_count,
            "projected_zero_tick_count": self.projected_zero_tick_count,
            "zero_fast_path_tick_count": self.zero_fast_path_tick_count,
            "phase_ids_observed": list(self._phase_ids),
            "physics_tick_count_by_phase": {
                phase_id: count
                for phase_id, count in self._tick_count_by_phase.items()
                if count > 0
            },
            "passed": passed,
        }


def compact_trace_row(
    frame: Any,
    info: Mapping[str, Any],
    *,
    actor_observation_v2: Sequence[float],
) -> dict[str, Any]:
    """Select only deterministic controller/action fields from one decision."""

    try:
        actor_vector = tuple(float(value) for value in actor_observation_v2)
    except (TypeError, ValueError) as exc:
        raise SoftResetEquivalenceError(
            "actor observation cannot be serialized as a numeric vector"
        ) from exc
    if not actor_vector or any(not math.isfinite(value) for value in actor_vector):
        raise SoftResetEquivalenceError(
            "actor observation must be a non-empty finite vector"
        )
    actor_bytes = json.dumps(
        actor_vector,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    row = {
        "decision_index": int(info["decision_index"]),
        "physics_tick": int(frame.physics_tick),
        "sim_time_s": float(frame.sim_time_s),
        "state_id": str(frame.state_id),
        "lifecycle": str(info.get("controller_lifecycle")),
        "phase_progress": float(frame.phase_progress),
        "physics_ticks_executed": int(info["physics_ticks_executed"]),
        "actor_observation_v2_dimension": len(actor_vector),
        "actor_observation_v2_sha256": hashlib.sha256(actor_bytes).hexdigest(),
        "nominal_full12": [float(value) for value in frame.nominal_action_full12],
        "residual_full12": [float(value) for value in info["projected_residual_full12"]],
        "applied_full12": [float(value) for value in info["applied_action_full12"]],
        "controller_task_result": str(info.get("controller_task_result")),
        "termination_reason": info.get("termination_reason"),
    }
    _canonical_row_bytes(row)
    return row


def _trace_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_row_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _through_p10(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    result = []
    saw_p10 = False
    for row in rows:
        state = str(row.get("state_id", ""))
        if state not in PHASE_IDS:
            raise SoftResetEquivalenceError(f"compact trace has invalid state {state!r}")
        if state == "P10":
            saw_p10 = True
        if int(state[1:]) <= 10:
            result.append(row)
    if not saw_p10:
        raise SoftResetEquivalenceError("compact trace never reached P10")
    return tuple(result)


def _first_trace_mismatch(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for index in range(min(len(left), len(right))):
        if _canonical_row_bytes(left[index]) == _canonical_row_bytes(right[index]):
            continue
        fields = [name for name in TRACE_FIELDS if left[index].get(name) != right[index].get(name)]
        return {
            "index": index,
            "fields": fields,
            "fresh": dict(left[index]),
            "reused": dict(right[index]),
        }
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "fields": ["trace_length"],
            "fresh_length": len(left),
            "reused_length": len(right),
        }
    return None


def compare_compact_traces(
    fresh: Sequence[Mapping[str, Any]], reused: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    first = tuple(fresh)
    second = tuple(reused)
    if not first or not second:
        raise SoftResetEquivalenceError("both compact traces must be non-empty")
    first_p10 = _through_p10(first)
    second_p10 = _through_p10(second)
    p10_mismatch = _first_trace_mismatch(first_p10, second_p10)
    whole_mismatch = _first_trace_mismatch(first, second)
    return {
        "schema": "wlr50_clean.soft_reset_trace_comparison.v1",
        "fields": list(TRACE_FIELDS),
        "through_p10": {
            "fresh_row_count": len(first_p10),
            "reused_row_count": len(second_p10),
            "fresh_sha256": _trace_digest(first_p10),
            "reused_sha256": _trace_digest(second_p10),
            "exactly_equal": p10_mismatch is None,
            "first_mismatch": p10_mismatch,
        },
        "whole_episode": {
            "fresh_row_count": len(first),
            "reused_row_count": len(second),
            "fresh_sha256": _trace_digest(first),
            "reused_sha256": _trace_digest(second),
            "exactly_equal": whole_mismatch is None,
            "first_mismatch": whole_mismatch,
        },
    }


def select_reset_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in RESET_METADATA_FIELDS if name not in info]
    if missing:
        raise SoftResetEquivalenceError(f"backend reset metadata is incomplete: {missing}")
    return {name: info[name] for name in RESET_METADATA_FIELDS}


def compare_reset_metadata(
    fresh: Mapping[str, Any], reused: Mapping[str, Any]
) -> dict[str, Any]:
    same_fields = (
        "environment_hash",
        "robot_asset_hash",
        "canonical_reset_state_source",
        "canonical_reset_state_sha256",
        "canonical_reset_state_instance_count",
        "pre_limit_native_state_observed_sha256",
        "pre_limit_native_state_instance_count",
        "pre_limit_native_state_matches_canonical",
        "pre_settle_native_state_observed_sha256",
        "pre_settle_native_state_matches_canonical",
        "adapter_standing_pose_deg",
        "canonical_settled_state_source",
        "canonical_settled_state_sha256",
        "observed_settled_state_sha256",
        "reset_contact_sensor_count",
        "reset_initialization_order",
        "pre_physics_session_limit_state_sha256",
        "pre_physics_composed_limit_state_sha256",
        "pre_physics_composed_limit_state_matches_canonical",
        "session_limit_specs_present_during_physics_reset",
        "post_author_session_limit_state_sha256",
        "post_author_session_limit_state_matches_canonical",
        "session_limit_specs_after_authoring",
        "environment_initialization",
        "controller_hash",
        "motion_contract_hash",
        "seed",
        "reset_options",
        "initial_root_state",
        "initial_joint_state",
        "obstacle_pose",
        "level_reference_orientation_wxyz",
        "settle_ticks",
        "randomization_enabled",
        "raw_recording_access",
        "locked_scene_snapshot",
    )
    checks = {
        "fresh_reset_count_is_one": int(fresh.get("reset_count", -1)) == 1,
        "reused_reset_count_is_two": int(reused.get("reset_count", -1)) == 2,
        "fresh_used_one_scene_factory_physics_reset": int(
            fresh.get("reset_global_simulation_resets", -1)
        )
        == 1
        and fresh.get("physics_lifecycle_reset")
        == "scene_factory_reset_before_limit_authoring",
        "fresh_used_no_step_free_forward_sync": int(
            fresh.get("reset_simulation_forward_syncs", -1)
        )
        == 0,
        "fresh_used_no_reset_root_pose_write": int(
            fresh.get("reset_root_pose_writes", -1)
        )
        == 0,
        "fresh_used_no_reset_root_velocity_write": int(
            fresh.get("reset_root_velocity_writes", -1)
        )
        == 0,
        "fresh_used_no_reset_joint_state_write": int(
            fresh.get("reset_joint_state_writes", -1)
        )
        == 0,
        "fresh_did_not_use_indexed_canonical_restore": not bool(
            fresh.get("canonical_reset_restore_applied", True)
        )
        and fresh.get("canonical_reset_applied_sha256") is None,
        "fresh_physx_started_without_session_limit_specs": int(
            fresh.get("session_limit_specs_present_during_physics_reset", -1)
        )
        == 0
        and int(fresh.get("session_limit_specs_removed_before_reset", -1)) == 0
        and fresh.get("removed_session_limit_state_sha256") is None
        and bool(
            fresh.get(
                "pre_physics_composed_limit_state_matches_canonical", False
            )
        ),
        "fresh_pre_limit_native_state_is_canonical": bool(
            fresh.get("pre_limit_native_state_matches_canonical", False)
        )
        and int(fresh.get("pre_limit_native_state_instance_count", -1)) == 1,
        "fresh_pre_settle_state_reached_canonical": bool(
            fresh.get("pre_settle_native_state_matches_canonical", False)
        )
        and fresh.get("pre_settle_native_state_observed_sha256")
        == fresh.get("canonical_reset_state_sha256"),
        "fresh_authored_exactly_sixteen_session_limit_specs": int(
            fresh.get("session_limit_specs_after_authoring", -1)
        )
        == 16
        and bool(
            fresh.get(
                "post_author_session_limit_state_matches_canonical", False
            )
        ),
        "fresh_natural_settle_reached_canonical_state": fresh.get(
            "observed_settled_state_sha256"
        )
        == fresh.get("canonical_settled_state_sha256"),
        "fresh_reinitialized_all_contact_sensors": int(
            fresh.get("reset_contact_sensor_count", -1)
        )
        == 13,
        "reused_used_one_pre_limit_hard_physics_reset": int(
            reused.get("reset_global_simulation_resets", -1)
        )
        == 1
        and reused.get("physics_lifecycle_reset")
        == "session_limits_removed_then_hard_reset",
        "reused_used_no_step_free_forward_sync": int(
            reused.get("reset_simulation_forward_syncs", -1)
        )
        == 0,
        "reused_used_no_reset_root_pose_write": int(
            reused.get("reset_root_pose_writes", -1)
        )
        == 0,
        "reused_used_no_reset_root_velocity_write": int(
            reused.get("reset_root_velocity_writes", -1)
        )
        == 0,
        "reused_used_no_reset_joint_state_write": int(
            reused.get("reset_joint_state_writes", -1)
        )
        == 0,
        "reused_did_not_use_indexed_canonical_restore": not bool(
            reused.get("canonical_reset_restore_applied", True)
        )
        and reused.get("canonical_reset_applied_sha256") is None,
        "reused_removed_exactly_sixteen_session_limit_specs": int(
            reused.get("session_limit_specs_removed_before_reset", -1)
        )
        == 16
        and isinstance(reused.get("removed_session_limit_state_sha256"), str)
        and len(str(reused.get("removed_session_limit_state_sha256"))) == 64,
        "reused_physx_started_without_session_limit_specs": int(
            reused.get("session_limit_specs_present_during_physics_reset", -1)
        )
        == 0
        and bool(
            reused.get(
                "pre_physics_composed_limit_state_matches_canonical", False
            )
        ),
        "reused_pre_limit_native_state_is_canonical": bool(
            reused.get("pre_limit_native_state_matches_canonical", False)
        )
        and int(reused.get("pre_limit_native_state_instance_count", -1)) == 1,
        "reused_pre_settle_state_reached_canonical": bool(
            reused.get("pre_settle_native_state_matches_canonical", False)
        )
        and reused.get("pre_settle_native_state_observed_sha256")
        == reused.get("canonical_reset_state_sha256"),
        "reused_reauthored_removed_session_limit_state_exactly": int(
            reused.get("session_limit_specs_after_authoring", -1)
        )
        == 16
        and bool(
            reused.get(
                "post_author_session_limit_state_matches_canonical", False
            )
        )
        and reused.get("post_author_session_limit_state_sha256")
        == reused.get("removed_session_limit_state_sha256"),
        "reused_natural_settle_reached_canonical_state": reused.get(
            "observed_settled_state_sha256"
        )
        == reused.get("canonical_settled_state_sha256"),
        "reused_reinitialized_all_contact_sensors": int(
            reused.get("reset_contact_sensor_count", -1)
        )
        == 13,
        "fresh_did_not_restore_canonical_settled_state": not bool(
            fresh.get("canonical_settled_restore_applied", True)
        )
        and fresh.get("canonical_settled_applied_sha256") is None,
        "reused_did_not_overwrite_natural_settled_state": not bool(
            reused.get("canonical_settled_restore_applied", True)
        )
        and reused.get("canonical_settled_applied_sha256") is None,
        **{
            f"same_{name}": fresh.get(name) == reused.get(name)
            for name in same_fields
        },
    }
    return {
        "schema": "wlr50_clean.soft_reset_metadata_comparison.v3",
        "backend_instance_count": 1,
        "same_backend_instance_reused": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def soft_reset_contract_hashes(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    result = {}
    for relative in SOFT_RESET_CONTRACT_RELATIVE_PATHS:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SoftResetEquivalenceError(f"contract path escaped project: {relative}") from exc
        if not path.is_file():
            raise SoftResetEquivalenceError(f"soft-reset contract file is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SoftResetEquivalenceError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise SoftResetEquivalenceError(f"{label} must contain a JSON object")
    return dict(value)


def _load_compact_trace(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise SoftResetEquivalenceError(
                        f"compact trace has a blank row at line {line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise SoftResetEquivalenceError(
                        f"compact trace row {line_number} is not an object"
                    )
                selected = dict(row)
                if tuple(selected) != TRACE_FIELDS:
                    raise SoftResetEquivalenceError(
                        f"compact trace row {line_number} has non-canonical fields"
                    )
                _canonical_row_bytes(selected)
                rows.append(selected)
    except (OSError, json.JSONDecodeError) as exc:
        raise SoftResetEquivalenceError("compact trace is unreadable or invalid JSONL") from exc
    if not rows:
        raise SoftResetEquivalenceError("compact trace is empty")
    return tuple(rows)


def validate_soft_reset_acceptance(
    acceptance_path: str | Path, *, project_root: str | Path
) -> dict[str, Any]:
    """Validate a finalized live gate before enabling training auto-reset."""

    path = Path(acceptance_path).resolve()
    root = Path(project_root).resolve()
    if not path.is_file() or path.name != SOFT_RESET_ACCEPTANCE_FILENAME:
        raise SoftResetEquivalenceError(
            f"soft-reset acceptance artifact is missing or misnamed: {path}"
        )
    payload = _load_json_object(path, label="soft-reset acceptance")
    if payload.get("schema") != SOFT_RESET_ACCEPTANCE_SCHEMA:
        raise SoftResetEquivalenceError("unexpected soft-reset acceptance schema")
    if payload.get("passed") is not True:
        raise SoftResetEquivalenceError("soft-reset equivalence gate did not pass")
    required_checks = payload.get("checks")
    if not isinstance(required_checks, Mapping) or set(required_checks) != set(
        ACCEPTANCE_CHECK_NAMES
    ):
        raise SoftResetEquivalenceError("soft-reset acceptance checks are incomplete")
    if not all(
        required_checks.get(name) is True for name in ACCEPTANCE_CHECK_NAMES
    ):
        raise SoftResetEquivalenceError("soft-reset acceptance includes a failed check")
    if (
        payload.get("episode_count") != 2
        or payload.get("backend_instance_count") != 1
        or payload.get("full_rate_raw_streams_written") is not False
        or payload.get("compact_trace_fields") != list(TRACE_FIELDS)
    ):
        raise SoftResetEquivalenceError("soft-reset acceptance run shape is invalid")
    expected_hashes = soft_reset_contract_hashes(root)
    if payload.get("contract_file_sha256") != expected_hashes:
        raise SoftResetEquivalenceError("soft-reset acceptance contract hashes are stale")
    lifecycle_path = path.parent / "run_manifest.json"
    lifecycle = _load_json_object(lifecycle_path, label="soft-reset run manifest")
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("schema") != "wlr50_clean.ppo_run_manifest.v1"
        or lifecycle.get("lifecycle") != "SUCCEEDED"
        or lifecycle.get("exit_code") != 0
        or lifecycle.get("immutable_run_directory") is not True
        or lifecycle.get("run_kind") != "soft-reset-equivalence"
        or Path(str(lifecycle.get("run_dir", ""))).resolve() != path.parent
        or Path(str(lifecycle.get("project_root", ""))).resolve() != root
        or lifecycle.get("entrypoint") != "wlr50_clean.ppo.cli"
        or lifecycle.get("subcommand") != "soft-reset-equivalence"
    ):
        raise SoftResetEquivalenceError("soft-reset gate lifecycle is not successful")
    identity = lifecycle.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("seed") != payload.get("seed")
        or identity.get("environment_count") != 1
        or identity.get("training_stage") != "soft-reset-equivalence-live"
    ):
        raise SoftResetEquivalenceError("soft-reset gate lifecycle identity is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise SoftResetEquivalenceError("soft-reset artifact inventory is missing")
    if len(artifacts) != 4:
        raise SoftResetEquivalenceError(
            "soft-reset artifact inventory must contain two traces and two summaries"
        )
    expected_names = {
        "episode_0_fresh_scene_compact_trace.jsonl",
        "episode_0_fresh_scene_summary.json",
        "episode_1_soft_reset_reuse_compact_trace.jsonl",
        "episode_1_soft_reset_reuse_summary.json",
    }
    artifact_by_name: dict[str, Path] = {}
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise SoftResetEquivalenceError("soft-reset artifact record is invalid")
        artifact = (path.parent / str(record.get("path", ""))).resolve()
        try:
            artifact.relative_to(path.parent)
        except ValueError as exc:
            raise SoftResetEquivalenceError("soft-reset artifact escaped its run") from exc
        if (
            not artifact.is_file()
            or artifact.stat().st_size != record.get("bytes")
            or _sha256(artifact) != record.get("sha256")
        ):
            raise SoftResetEquivalenceError(f"soft-reset artifact hash mismatch: {artifact}")
        if artifact.name in artifact_by_name:
            raise SoftResetEquivalenceError("soft-reset artifact inventory contains duplicates")
        artifact_by_name[artifact.name] = artifact
    if set(artifact_by_name) != expected_names:
        raise SoftResetEquivalenceError("soft-reset artifact inventory has unexpected names")

    fresh_trace = _load_compact_trace(
        artifact_by_name["episode_0_fresh_scene_compact_trace.jsonl"]
    )
    reused_trace = _load_compact_trace(
        artifact_by_name["episode_1_soft_reset_reuse_compact_trace.jsonl"]
    )
    trace_comparison = compare_compact_traces(fresh_trace, reused_trace)
    if payload.get("trace_comparison") != trace_comparison:
        raise SoftResetEquivalenceError("soft-reset trace comparison is inconsistent")
    fresh_summary = _load_json_object(
        artifact_by_name["episode_0_fresh_scene_summary.json"],
        label="fresh-scene summary",
    )
    reused_summary = _load_json_object(
        artifact_by_name["episode_1_soft_reset_reuse_summary.json"],
        label="soft-reset reuse summary",
    )
    summaries = (fresh_summary, reused_summary)
    if payload.get("episodes") != list(summaries):
        raise SoftResetEquivalenceError("soft-reset summaries differ from acceptance payload")
    if tuple(row.get("reset_role") for row in summaries) != (
        "fresh_scene",
        "soft_reset_reuse",
    ):
        raise SoftResetEquivalenceError("soft-reset episode roles are invalid")
    expected_trace_paths = (
        artifact_by_name["episode_0_fresh_scene_compact_trace.jsonl"],
        artifact_by_name["episode_1_soft_reset_reuse_compact_trace.jsonl"],
    )
    if (
        tuple(row.get("episode_index") for row in summaries) != (0, 1)
        or any(row.get("seed") != payload.get("seed") for row in summaries)
        or any(
            Path(str(row.get("trace_path", ""))).name != trace_path.name
            or row.get("trace_sha256") != _sha256(trace_path)
            for row, trace_path in zip(summaries, expected_trace_paths, strict=True)
        )
        or any(
            isinstance(row.get("duration_s"), bool)
            or not isinstance(row.get("duration_s"), (int, float))
            or not math.isfinite(float(row["duration_s"]))
            or not 0.0 < float(row["duration_s"]) <= 200.0
            for row in summaries
        )
    ):
        raise SoftResetEquivalenceError("soft-reset episode summary identity is invalid")
    try:
        reset_comparison = compare_reset_metadata(
            select_reset_metadata(fresh_summary["reset_metadata"]),
            select_reset_metadata(reused_summary["reset_metadata"]),
        )
    except (KeyError, TypeError) as exc:
        raise SoftResetEquivalenceError("soft-reset summaries omit reset metadata") from exc
    if payload.get("reset_metadata_comparison") != reset_comparison:
        raise SoftResetEquivalenceError("soft-reset metadata comparison is inconsistent")

    def _audit_passes(summary: Mapping[str, Any]) -> bool:
        audit = summary.get("zero_residual_tick_audit")
        if not isinstance(audit, Mapping):
            return False
        tick_count = audit.get("tick_count")
        phase_counts = audit.get("physics_tick_count_by_phase")
        return bool(
            isinstance(tick_count, int)
            and not isinstance(tick_count, bool)
            and tick_count > 0
            and isinstance(phase_counts, Mapping)
            and set(phase_counts).issubset(PHASE_IDS)
            and bool(phase_counts)
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in phase_counts.values()
            )
            and audit.get("passed") is True
            and audit.get("status") == "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE"
            and audit.get("bitwise_equal") is True
            and isinstance(audit.get("nominal_sequence_sha256"), str)
            and len(audit["nominal_sequence_sha256"]) == 64
            and audit.get("applied_sequence_sha256")
            == audit.get("nominal_sequence_sha256")
            and audit.get("raw_zero_tick_count") == tick_count
            and audit.get("projected_zero_tick_count") == tick_count
            and audit.get("zero_fast_path_tick_count") == tick_count
            and audit.get("phase_ids_observed") == list(PHASE_IDS)
            and sum(phase_counts.values()) == tick_count
        )

    computed_checks = {
        "one_backend_instance_two_episodes": payload.get("backend_instance_count") == 1,
        "both_authoritative_success": all(
            row.get("authoritative_success") is True
            and row.get("task_success") is True
            and row.get("termination_reason") == "SUCCESS"
            for row in summaries
        ),
        "both_complete_p01_p13": all(
            row.get("completed_p01_p13") is True
            and row.get("phase_ids_observed") == list(PHASE_IDS)
            for row in summaries
        ),
        "both_no_body_collision": all(
            row.get("body_collision") is False for row in summaries
        ),
        "both_no_wheel_only_climb": all(
            row.get("wheel_only_climb") is False for row in summaries
        ),
        "both_no_safety_abort": all(
            row.get("safety_abort") is False for row in summaries
        ),
        "both_under_maximum_duration": all(
            row.get("under_maximum_duration") is True for row in summaries
        ),
        "both_zero_residual_bitwise_all_ticks": all(
            _audit_passes(row) for row in summaries
        ),
        "decision_counts_match_compact_traces": all(
            row.get("decision_count") == len(trace)
            for row, trace in zip(summaries, (fresh_trace, reused_trace), strict=True)
        ),
        "physics_tick_counts_match_audits": all(
            row.get("physics_tick")
            == row.get("zero_residual_tick_audit", {}).get("tick_count")
            for row in summaries
        ),
        "no_runtime_recording_access": all(
            row.get("recording_runtime_access_count") == 0 for row in summaries
        ),
        "no_in_episode_root_writes": all(
            row.get("in_episode_root_write_count") == 0 for row in summaries
        ),
        "reset_metadata_equivalent": reset_comparison["passed"] is True,
        "deterministic_trace_equal_through_p10": trace_comparison["through_p10"][
            "exactly_equal"
        ]
        is True,
        "deterministic_trace_equal_whole_episode": trace_comparison["whole_episode"][
            "exactly_equal"
        ]
        is True,
    }
    if computed_checks != dict(required_checks) or not all(computed_checks.values()):
        raise SoftResetEquivalenceError("soft-reset acceptance checks do not match evidence")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "run_manifest": str(lifecycle_path),
        "run_manifest_sha256": _sha256(lifecycle_path),
        "contract_file_sha256": expected_hashes,
        "passed": True,
    }


__all__ = [
    "ACCEPTANCE_CHECK_NAMES",
    "CompactZeroResidualTickAudit",
    "PHASE_IDS",
    "RESET_METADATA_FIELDS",
    "SOFT_RESET_ACCEPTANCE_FILENAME",
    "SOFT_RESET_ACCEPTANCE_SCHEMA",
    "SOFT_RESET_CONTRACT_RELATIVE_PATHS",
    "SoftResetEquivalenceError",
    "TRACE_FIELDS",
    "compact_trace_row",
    "compare_compact_traces",
    "compare_reset_metadata",
    "select_reset_metadata",
    "soft_reset_contract_hashes",
    "validate_soft_reset_acceptance",
]
