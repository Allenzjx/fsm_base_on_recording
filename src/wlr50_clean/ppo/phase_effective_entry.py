"""Pinned effective phase-entry contract derived from independent live probes.

Each phase snapshot declares an immutable source-command replay window.
Production reset writes the source state once, replays every declared command
with one real PhysX step apiece, and never rewinds.  This module owns the
separate calibrated contract for the resulting target-entry state.  P01 is
intentionally outside this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .phase_snapshots import (
    SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS,
    SOURCE_ACK_REPLAY_INVARIANT_FIELDS,
    ValidatedPhaseSnapshotBundle,
    load_validated_phase_snapshot_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "ppo_phase_effective_entry_v1.json"
)
DEFAULT_EFFECTIVE_ENTRY_SIDECAR_PATH = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.with_suffix(
    ".sha256"
)
DEFAULT_ENVIRONMENT_LOCK_PATH = PROJECT_ROOT / "configs" / "environment_lock.json"
DEFAULT_FROZEN_LEDGER_PATH = (
    PROJECT_ROOT / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
)

CONTRACT_SCHEMA = "wlr50_clean.ppo_phase_effective_entry_contract.v2"
ENTRY_SCHEMA = "wlr50_clean.ppo_phase_effective_entry.v2"
DERIVATION_SCHEMA = "wlr50_clean.ppo_phase_effective_entry_derivation.v2"
CALIBRATION_SCHEMA = "wlr50_clean.ppo_phase_effective_entry_calibration.v2"
RECORD_SCHEMA = "wlr50_clean.ppo_phase_effective_entry_contract_record.v2"
CONTRACT_HASH_SCHEMA = "wlr50_clean.ppo_phase_effective_entry_contract_hash.v2"
PROBE_SCHEMA = "wlr50_clean.phase_snapshot_live_probe.v3"
# ``artifacts.reserve_run`` canonicalizes run-kind tokens with ``_safe_token``.
# Keep this consumer bound to the value actually written into the immutable
# run directory and both lifecycle manifests, rather than the wrapper's raw
# underscore-separated input.
CALIBRATION_RUN_KIND = "phase-effective-entry-calibration"
CALIBRATION_TRAINING_STAGE = "phase-effective-entry-calibration"
CALIBRATION_ARTIFACT_ROLE = "CALIBRATION_ONLY_NOT_TRAINING_ACCEPTANCE"
CALIBRATION_LIVE_PROOF_SCHEMA = (
    "wlr50_clean.ppo_phase_effective_entry_calibration_live_proof.v2"
)
PHASE_IDS = tuple(f"P{index:02d}" for index in range(2, 14))
SERVO_ORDER = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_ORDER = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
PAIR_ORDER = ("ground", "obstacle")
FINGERPRINT_FIELDS = (
    "root_position_m",
    "root_orientation",
    "root_linear_velocity_m_s",
    "root_angular_velocity_rad_s",
    "servo_position_deg",
    "servo_velocity_deg_s",
    "wheel_velocity_rad_s",
    "wheel_center_m",
    "wheel_bottom_m",
)
FINGERPRINT_MAX_ULP_DISTANCE = 1
COMPONENT_STATE_SCHEMA = "wlr50_clean.phase_effective_entry_component_state.v1"
COMPONENT_STATE_MAX_ULP_DISTANCE = 1
COMPONENT_VECTOR_SIZES = {
    "root_position_w_m": 3,
    "root_orientation_wxyz": 4,
    "root_linear_velocity_w_m_s": 3,
    "root_angular_velocity_w_rad_s": 3,
    "servo_logical_position_deg": 8,
    "servo_logical_velocity_deg_s": 8,
    "wheel_logical_velocity_rad_s": 4,
}
COMPONENT_GEOMETRY_FIELDS = ("wheel_centers_w_m", "wheel_bottoms_w_m")
COMPONENT_UNITS = {
    "root_position_w_m": "m",
    "root_orientation_wxyz": "unit_quaternion",
    "root_linear_velocity_w_m_s": "m/s",
    "root_angular_velocity_w_rad_s": "rad/s",
    "servo_logical_position_deg": "deg",
    "servo_logical_velocity_deg_s": "deg/s",
    "wheel_logical_velocity_rad_s": "rad/s",
    "wheel_centers_w_m": "m",
    "wheel_bottoms_w_m": "m",
}
CONTACT_FORCE_ON_N = 0.25
CONTACT_FORCE_OFF_N = 0.12
CONTACT_SOURCE = "isaaclab.ContactSensor.force_matrix_w"
PHYSICS_HZ = 120.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_AUTHORED_ENTRY_GUARDS = (
    "previous_state_done",
    "no_body_obstacle_collision",
    "joint_hard_limits_valid",
    "reference_entry_compatible",
    "critical_actuators_available",
)
_P10_COMPLETION_GUARDS = (
    "motion_endpoint_issued",
    "rl_workspace_geometry",
)
_RESET_SETTLE_TICKS = 180
SOURCE_PROVEN_EXECUTE_RESTORE_MODE = "source_proven_execute_after_prime"
SOURCE_PROVEN_EXECUTE_RESTORE_SCHEMA = (
    "wlr50_clean.phase_snapshot_source_proven_execute_restore.v1"
)


class EffectivePhaseEntryError(ValueError):
    """The effective-entry calibration or live evidence is not trustworthy."""


def phase_entry_time_s(physics_ticks: int) -> float:
    """Map an integer 120 Hz phase-entry tick to its canonical binary64 time.

    Division is intentional.  Multiplying a large tick by the rounded binary64
    reciprocal can differ by one ULP (for example, tick 1817), so every producer
    and verifier of phase-entry evidence must call this function.
    """

    if type(physics_ticks) is not int or physics_ticks < 0:
        raise EffectivePhaseEntryError(
            "phase-entry physics tick must be a non-negative integer"
        )
    return physics_ticks / PHYSICS_HZ


def _freeze_json_value(value: Any) -> Any:
    """Recursively freeze one already-validated JSON value."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """Return an isolated mutable JSON-shaped copy of a frozen value."""

    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ValidatedEffectivePhaseEntryContract:
    """One immutable capture of the contract and every local trust anchor."""

    contract_path: Path
    sidecar_path: Path
    environment_lock_path: Path
    frozen_ledger_path: Path
    contract_bytes: bytes
    sidecar_bytes: bytes
    environment_lock_bytes: bytes
    frozen_ledger_bytes: bytes
    file_sha256: str
    sidecar_file_sha256: str
    contract_sha256: str
    phase_snapshot_bundle_sha256: str
    entries: tuple[tuple[str, Mapping[str, Any]], ...]
    filesystem_identity: tuple[tuple[Any, ...], ...]

    def __post_init__(self) -> None:
        # ``frozen=True`` protects only dataclass attributes.  Freeze every
        # nested JSON container as well so no caller can mutate the pinned
        # contract through the public ``entries`` attribute.
        object.__setattr__(
            self,
            "entries",
            tuple(
                (str(phase), _freeze_json_value(value))
                for phase, value in self.entries
            ),
        )

    def entry(self, phase: str) -> Mapping[str, Any]:
        if phase == "P01":
            raise EffectivePhaseEntryError("P01 must not use the effective-entry contract")
        selected = next((value for name, value in self.entries if name == phase), None)
        if selected is None:
            raise EffectivePhaseEntryError(f"effective-entry contract lacks {phase}")
        # Never expose an internal mapping proxy or one of its nested values.
        # Backend consumers receive a disposable deep copy.
        return _thaw_json_value(selected)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": RECORD_SCHEMA,
            "contract_path": str(self.contract_path),
            "sidecar_path": str(self.sidecar_path),
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "contract_sha256": self.contract_sha256,
            "phase_snapshot_bundle_sha256": self.phase_snapshot_bundle_sha256,
            "portability_scope": "same_locked_host_runtime_only",
            "calibration_status": (
                "provisional_pending_independent_fresh_holdout"
            ),
            "environment_lock_path": str(self.environment_lock_path),
            "environment_lock_sha256": _sha256_bytes(self.environment_lock_bytes),
            "frozen_ledger_path": str(self.frozen_ledger_path),
            "frozen_ledger_sha256": _sha256_bytes(self.frozen_ledger_bytes),
            "phase_count": len(self.entries),
            "phases": [phase for phase, _ in self.entries],
        }

    def file_hashes(self) -> dict[str, str]:
        return {
            str(self.contract_path): self.file_sha256,
            str(self.sidecar_path): self.sidecar_file_sha256,
            str(self.environment_lock_path): _sha256_bytes(self.environment_lock_bytes),
            str(self.frozen_ledger_path): _sha256_bytes(self.frozen_ledger_bytes),
        }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_lf_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EffectivePhaseEntryError(f"{label} must be lowercase SHA-256")
    return value


def _require_git_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise EffectivePhaseEntryError(f"{label} must be a lowercase 40-hex commit")
    return value


def _validated_source_proven_execute_restore(
    snapshot_payload: Mapping[str, Any], phase: str
) -> Mapping[str, Any] | None:
    """Validate P10's source-authenticated post-entry controller restore.

    The special P10 reset does not ask a freshly constructed controller to
    reproduce the very short WAIT_ENTRY velocity window after a PhysX state
    write.  Instead, the immutable successful-trial WAIT_ENTRY->EXECUTE row
    proves that the authored guards passed, one source P10 EXECUTE command is
    applied to obtain a current solver/contact sample, and the episode starts
    at motion tick one.  This helper deliberately validates the complete
    nested contract at every consumer boundary.
    """

    mode_present = "controller_restore_mode" in snapshot_payload
    contract_present = "controller_restore_contract" in snapshot_payload
    mode = snapshot_payload.get("controller_restore_mode")
    contract = snapshot_payload.get("controller_restore_contract")
    if not mode_present and not contract_present:
        if phase == "P10":
            raise EffectivePhaseEntryError(
                "P10 requires the source-proven EXECUTE controller restore"
            )
        return None
    if mode != SOURCE_PROVEN_EXECUTE_RESTORE_MODE or not contract_present:
        raise EffectivePhaseEntryError(
            f"validated controller-restore mode/contract is incomplete for {phase}"
        )
    if phase != "P10" or snapshot_payload.get("fsm_state") != "P10":
        raise EffectivePhaseEntryError(
            "source-proven EXECUTE restore is permitted only for P10"
        )
    expected_fields = {
        "schema",
        "source_transition_tick",
        "source_transition_time_s",
        "source_transition_row",
        "source_transition_row_canonical_sha256",
        "authored_entry_guard_order",
        "authored_entry_guard_evidence",
        "prime_command_tick",
        "effective_observation_tick",
        "consumed_motion_tick",
        "first_episode_motion_tick",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_fields:
        raise EffectivePhaseEntryError(
            "validated P10 source-proven EXECUTE restore contract is malformed"
        )
    source_tick = snapshot_payload.get("source_tick")
    replay_steps = snapshot_payload.get("source_replay_steps")
    transition_tick = contract.get("source_transition_tick")
    transition_time = contract.get("source_transition_time_s")
    effective_tick = contract.get("effective_observation_tick")
    if (
        contract.get("schema") != SOURCE_PROVEN_EXECUTE_RESTORE_SCHEMA
        or type(source_tick) is not int
        or type(replay_steps) is not int
        or replay_steps != 1
        or type(transition_tick) is not int
        or transition_tick != source_tick
        or isinstance(transition_time, bool)
        or not isinstance(transition_time, (int, float))
        or not math.isfinite(float(transition_time))
        or float(transition_time) != phase_entry_time_s(transition_tick)
        or contract.get("prime_command_tick") != source_tick
        or type(effective_tick) is not int
        or effective_tick != source_tick + replay_steps
        or contract.get("consumed_motion_tick") != 0
        or contract.get("first_episode_motion_tick") != 1
    ):
        raise EffectivePhaseEntryError(
            "validated P10 source-proven EXECUTE restore timing is invalid"
        )
    if any(
        name in snapshot_payload
        for name in (
            "target_entry_tick",
            "predecessor_verify_tick",
            "predecessor_verify_time_s",
            "controller_anchor_tick",
            "controller_anchor_time_s",
        )
    ):
        raise EffectivePhaseEntryError(
            "source-proven P10 restore must not declare legacy hybrid anchors"
        )
    commands = snapshot_payload.get("source_commands")
    if (
        not isinstance(commands, list)
        or len(commands) != 1
        or not isinstance(commands[0], Mapping)
        or commands[0].get("control_physics_tick") != source_tick
        or commands[0].get("source_fsm_state") != "P10"
        or commands[0].get("source_fsm_lifecycle") != "EXECUTE_MOTION"
        or snapshot_payload.get("fsm_lifecycle") != "EXECUTE_MOTION"
    ):
        raise EffectivePhaseEntryError(
            "source-proven P10 restore must prime one P10 EXECUTE command"
        )
    transition = contract.get("source_transition_row")
    if (
        not isinstance(transition, Mapping)
        or set(transition)
        != {
            "sim_time_s",
            "state_id",
            "from_lifecycle",
            "to_lifecycle",
            "reason",
            "details",
        }
        or transition.get("state_id") != "P10"
        or transition.get("from_lifecycle") != "WAIT_ENTRY"
        or transition.get("to_lifecycle") != "EXECUTE_MOTION"
        or transition.get("sim_time_s") != float(transition_time)
        or transition.get("reason") != "all live entry guards passed"
    ):
        raise EffectivePhaseEntryError(
            "validated P10 source transition is not the exact WAIT_ENTRY-to-EXECUTE row"
        )
    transition_hash = _require_sha256(
        contract.get("source_transition_row_canonical_sha256"),
        "P10 source transition row SHA",
    )
    if transition_hash != _sha256_bytes(_canonical_bytes(transition)):
        raise EffectivePhaseEntryError(
            "validated P10 source transition row hash is invalid"
        )
    order = contract.get("authored_entry_guard_order")
    evidence = contract.get("authored_entry_guard_evidence")
    details = transition.get("details")
    if (
        order != list(_AUTHORED_ENTRY_GUARDS)
        or not isinstance(evidence, list)
        or len(evidence) != len(_AUTHORED_ENTRY_GUARDS)
        or not isinstance(details, Mapping)
        or details.get("guards") != evidence
        or tuple(
            row.get("name") for row in evidence if isinstance(row, Mapping)
        )
        != _AUTHORED_ENTRY_GUARDS
    ):
        raise EffectivePhaseEntryError(
            "validated P10 source transition does not bind the authored guard order"
        )
    for row in evidence:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "passed", "value", "source", "reason"}
            or row.get("passed") is not True
            or not isinstance(row.get("source"), str)
            or not row["source"]
            or not isinstance(row.get("reason"), str)
        ):
            raise EffectivePhaseEntryError(
                "validated P10 source transition guard evidence is invalid"
            )
    return contract


def _source_proven_execute_bindings(
    snapshot_payload: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    """Return the canonical outer-proof fields for the special P10 restore."""

    contract = _validated_source_proven_execute_restore(snapshot_payload, phase)
    if contract is None:
        return {}
    return {
        "controller_restore_mode": SOURCE_PROVEN_EXECUTE_RESTORE_MODE,
        "source_transition_verified": True,
        "source_transition_tick": contract["source_transition_tick"],
        "source_transition_time_s": contract["source_transition_time_s"],
        "source_transition_row_canonical_sha256": contract[
            "source_transition_row_canonical_sha256"
        ],
        "consumed_motion_tick": contract["consumed_motion_tick"],
        "first_episode_motion_tick": contract["first_episode_motion_tick"],
        "entry_guards_from_source_not_replayed": True,
    }


def _snapshot_replay_contract(
    phase_snapshot_bundle: ValidatedPhaseSnapshotBundle,
    phase: str,
) -> tuple[
    Mapping[str, Any],
    Any,
    int,
    int,
    tuple[int, ...],
    int | None,
    float | None,
    int | None,
    float | None,
]:
    """Return the replay window proven by the pinned payload and manifest row.

    ``load_validated_phase_snapshot_payload`` revalidates the payload against
    the manifest captured in ``phase_snapshot_bundle``.  The checks below are
    deliberately repeated at this consumer boundary: calibration evidence may
    never choose its own replay count or target tick.
    """

    payload, snapshot = load_validated_phase_snapshot_payload(
        phase_snapshot_bundle, phase
    )
    source_tick = payload.get("source_tick")
    replay_steps = payload.get("source_replay_steps")
    commands = payload.get("source_commands")
    if (
        type(source_tick) is not int
        or source_tick < 0
        or type(replay_steps) is not int
        or replay_steps <= 0
        or not isinstance(commands, list)
        or len(commands) != replay_steps
        or any(not isinstance(command, Mapping) for command in commands)
    ):
        raise EffectivePhaseEntryError(
            f"validated phase snapshot replay window is invalid for {phase}"
        )
    restore_contract = _validated_source_proven_execute_restore(payload, phase)
    source_proven_execute = restore_contract is not None
    snapshot_restore_mode = getattr(snapshot, "controller_restore_mode", None)
    snapshot_transition_hash = getattr(
        snapshot, "source_transition_row_canonical_sha256", None
    )
    if source_proven_execute:
        assert restore_contract is not None
        if (
            snapshot_restore_mode != SOURCE_PROVEN_EXECUTE_RESTORE_MODE
            or snapshot_transition_hash
            != restore_contract["source_transition_row_canonical_sha256"]
        ):
            raise EffectivePhaseEntryError(
                "validated P10 manifest does not bind its source-proven restore"
            )
    elif snapshot_restore_mode is not None or snapshot_transition_hash is not None:
        raise EffectivePhaseEntryError(
            f"validated manifest declares an unpaired controller restore for {phase}"
        )
    target_entry_tick = source_tick + replay_steps
    hybrid = "target_entry_tick" in payload
    payload_target = payload.get("target_entry_tick")
    snapshot_target = getattr(snapshot, "target_entry_tick", None)
    has_predecessor_verify_tick = "predecessor_verify_tick" in payload
    has_predecessor_verify_time = "predecessor_verify_time_s" in payload
    predecessor_verify_tick = payload.get("predecessor_verify_tick")
    predecessor_verify_time = payload.get("predecessor_verify_time_s")
    snapshot_predecessor_verify_tick = getattr(
        snapshot, "predecessor_verify_tick", None
    )
    snapshot_predecessor_verify_time = getattr(
        snapshot, "predecessor_verify_time_s", None
    )
    has_controller_anchor_tick = "controller_anchor_tick" in payload
    has_controller_anchor_time = "controller_anchor_time_s" in payload
    controller_anchor_tick = payload.get("controller_anchor_tick")
    controller_anchor_time = payload.get("controller_anchor_time_s")
    snapshot_controller_anchor_tick = getattr(
        snapshot, "controller_anchor_tick", None
    )
    snapshot_controller_anchor_time = getattr(
        snapshot, "controller_anchor_time_s", None
    )
    if hybrid:
        if phase != "P10":
            raise EffectivePhaseEntryError(
                f"only P10 may declare a hybrid replay window, not {phase}"
            )
        if (
            not has_predecessor_verify_tick
            or not has_predecessor_verify_time
            or type(predecessor_verify_tick) is not int
            or not source_tick < predecessor_verify_tick < target_entry_tick
            or isinstance(predecessor_verify_time, bool)
            or not isinstance(predecessor_verify_time, (int, float))
            or not math.isfinite(float(predecessor_verify_time))
            or float(predecessor_verify_time)
            != phase_entry_time_s(predecessor_verify_tick)
            or snapshot_predecessor_verify_tick != predecessor_verify_tick
            or snapshot_predecessor_verify_time != predecessor_verify_time
            or not has_controller_anchor_tick
            or not has_controller_anchor_time
            or type(controller_anchor_tick) is not int
            or not predecessor_verify_tick
            < controller_anchor_tick
            < target_entry_tick
            or isinstance(controller_anchor_time, bool)
            or not isinstance(controller_anchor_time, (int, float))
            or not math.isfinite(float(controller_anchor_time))
            or float(controller_anchor_time)
            != phase_entry_time_s(controller_anchor_tick)
            or payload_target != target_entry_tick
            or snapshot_target != target_entry_tick
            or snapshot_controller_anchor_tick != controller_anchor_tick
            or snapshot_controller_anchor_time != controller_anchor_time
        ):
            raise EffectivePhaseEntryError(
                "validated P10 physical/predecessor/controller replay anchors are invalid"
            )
        predecessor_verify_time = float(predecessor_verify_time)
        controller_anchor_time = float(controller_anchor_time)
    else:
        if (
            has_predecessor_verify_tick
            or has_predecessor_verify_time
            or predecessor_verify_tick is not None
            or predecessor_verify_time is not None
            or snapshot_predecessor_verify_tick is not None
            or snapshot_predecessor_verify_time is not None
            or has_controller_anchor_tick
            or has_controller_anchor_time
            or controller_anchor_tick is not None
            or controller_anchor_time is not None
            or snapshot_controller_anchor_tick is not None
            or snapshot_controller_anchor_time is not None
            or payload_target is not None
            or snapshot_target is not None
            or replay_steps != 1
        ):
            raise EffectivePhaseEntryError(
                f"validated non-hybrid phase snapshot declares replay anchors for {phase}"
            )
        predecessor_verify_tick = None
        predecessor_verify_time = None
        controller_anchor_tick = None
        controller_anchor_time = None
    if (
        getattr(snapshot, "source_tick", None) != source_tick
        or getattr(snapshot, "source_replay_steps", None) != replay_steps
        or (
            payload_target is not None
            and (
                type(payload_target) is not int
                or payload_target != target_entry_tick
                or snapshot_target != target_entry_tick
            )
        )
        or (payload_target is None and snapshot_target is not None)
    ):
        raise EffectivePhaseEntryError(
            f"validated phase snapshot manifest replay binding differs for {phase}"
        )
    control_ticks = tuple(source_tick + index for index in range(replay_steps))
    if (
        tuple(command.get("control_physics_tick") for command in commands)
        != control_ticks
        or payload.get("source_command") != commands[0]
    ):
        raise EffectivePhaseEntryError(
            f"validated phase snapshot source-command sequence is invalid for {phase}"
        )
    if hybrid:
        assert isinstance(predecessor_verify_tick, int)
        assert isinstance(controller_anchor_tick, int)
        expected_contexts = tuple(
            ("P09", "EXECUTE_MOTION")
            if tick < predecessor_verify_tick
            else ("P09", "VERIFY_RESULT")
            if tick < controller_anchor_tick
            else ("P10", "WAIT_ENTRY")
            for tick in control_ticks
        )
        actual_contexts = tuple(
            (
                command.get("source_fsm_state"),
                command.get("source_fsm_lifecycle"),
            )
            for command in commands
        )
        if (
            actual_contexts != expected_contexts
            or payload.get("fsm_state") != "P10"
            or payload.get("fsm_lifecycle") != "WAIT_ENTRY"
        ):
            raise EffectivePhaseEntryError(
                "validated P10 replay does not bridge P09 EXECUTE_MOTION through "
                "P09 VERIFY_RESULT to the P10 WAIT_ENTRY controller anchor from "
                "its physical history anchor"
            )
    elif source_proven_execute:
        command = commands[0]
        if (
            command.get("source_fsm_state") != "P10"
            or command.get("source_fsm_lifecycle") != "EXECUTE_MOTION"
        ):
            raise EffectivePhaseEntryError(
                "validated source-proven P10 replay context is not EXECUTE_MOTION"
            )
    return (
        payload,
        snapshot,
        replay_steps,
        target_entry_tick,
        control_ticks,
        predecessor_verify_tick,
        predecessor_verify_time,
        controller_anchor_tick,
        controller_anchor_time,
    )


def _expected_replay_anchor_contract(
    snapshot_payload: Mapping[str, Any],
    phase: str,
    *,
    replay_steps: int,
    target_entry_tick: int,
    control_ticks: Sequence[int],
    predecessor_verify_tick: int | None,
    predecessor_verify_time_s: float | None,
    controller_anchor_tick: int | None,
    controller_anchor_time_s: float | None,
) -> dict[str, Any]:
    """Recompute the backend replay-anchor proof from pinned snapshot bytes."""

    source_tick = int(snapshot_payload["source_tick"])
    restore_contract = _validated_source_proven_execute_restore(
        snapshot_payload, phase
    )
    source_proven_execute = restore_contract is not None
    hybrid = "target_entry_tick" in snapshot_payload
    if hybrid:
        if phase != "P10":
            raise EffectivePhaseEntryError(
                f"only P10 may declare a hybrid replay-anchor proof, not {phase}"
            )
        assert predecessor_verify_tick is not None
        assert controller_anchor_tick is not None
        predecessor = PHASE_IDS[PHASE_IDS.index(phase) - 1]
        contexts = [
            {
                "source_control_physics_tick": tick,
                "source_fsm_state": predecessor if tick < controller_anchor_tick else phase,
                "source_fsm_lifecycle": (
                    "EXECUTE_MOTION"
                    if tick < predecessor_verify_tick
                    else "VERIFY_RESULT"
                    if tick < controller_anchor_tick
                    else "WAIT_ENTRY"
                ),
                "anchor_segment": (
                    "physical_anchor_to_predecessor_verify"
                    if tick < predecessor_verify_tick
                    else "predecessor_verify_to_controller_anchor"
                    if tick < controller_anchor_tick
                    else "controller_anchor_to_target_entry"
                ),
            }
            for tick in control_ticks
        ]
        segments = [
            {
                "first_source_tick": source_tick,
                "last_source_tick": predecessor_verify_tick - 1,
                "source_replay_steps": predecessor_verify_tick - source_tick,
                "source_fsm_state": predecessor,
                "source_fsm_lifecycle": "EXECUTE_MOTION",
                "anchor_segment": "physical_anchor_to_predecessor_verify",
            },
            {
                "first_source_tick": predecessor_verify_tick,
                "last_source_tick": controller_anchor_tick - 1,
                "source_replay_steps": controller_anchor_tick
                - predecessor_verify_tick,
                "source_fsm_state": predecessor,
                "source_fsm_lifecycle": "VERIFY_RESULT",
                "anchor_segment": "predecessor_verify_to_controller_anchor",
            },
            {
                "first_source_tick": controller_anchor_tick,
                "last_source_tick": target_entry_tick - 1,
                "source_replay_steps": target_entry_tick - controller_anchor_tick,
                "source_fsm_state": phase,
                "source_fsm_lifecycle": "WAIT_ENTRY",
                "anchor_segment": "controller_anchor_to_target_entry",
            },
        ]
    elif source_proven_execute:
        contexts = [
            {
                "source_control_physics_tick": source_tick,
                "source_fsm_state": phase,
                "source_fsm_lifecycle": "EXECUTE_MOTION",
                "anchor_segment": "source_proven_execute_prime",
            }
        ]
        segments = [
            {
                "first_source_tick": source_tick,
                "last_source_tick": source_tick,
                "source_replay_steps": 1,
                "source_fsm_state": phase,
                "source_fsm_lifecycle": "EXECUTE_MOTION",
                "anchor_segment": "source_proven_execute_prime",
            }
        ]
    else:
        contexts = [
            {
                "source_control_physics_tick": source_tick,
                "source_fsm_state": None,
                "source_fsm_lifecycle": None,
                "anchor_segment": "single_source_command",
            }
        ]
        segments = [
            {
                "first_source_tick": source_tick,
                "last_source_tick": source_tick,
                "source_replay_steps": 1,
                "source_fsm_state": None,
                "source_fsm_lifecycle": None,
                "anchor_segment": "single_source_command",
            }
        ]
    physical_to_controller_steps = (
        None
        if controller_anchor_tick is None
        else controller_anchor_tick - source_tick
    )
    physical_to_predecessor_verify_steps = (
        None
        if predecessor_verify_tick is None
        else predecessor_verify_tick - source_tick
    )
    predecessor_verify_to_controller_steps = (
        None
        if predecessor_verify_tick is None or controller_anchor_tick is None
        else controller_anchor_tick - predecessor_verify_tick
    )
    controller_to_target_steps = (
        None
        if controller_anchor_tick is None
        else target_entry_tick - controller_anchor_tick
    )
    result = {
        "schema": (
            "wlr50_clean.phase_snapshot_replay_anchor_contract.v3"
            if source_proven_execute
            else "wlr50_clean.phase_snapshot_replay_anchor_contract.v2"
        ),
        "verified": True,
        "phase": phase,
        "mode": (
            "hybrid_physical_predecessor_verify_and_controller_anchors"
            if hybrid
            else SOURCE_PROVEN_EXECUTE_RESTORE_MODE
            if source_proven_execute
            else "single_physical_anchor"
        ),
        "physical_anchor_tick": source_tick,
        "physical_anchor_time_s": phase_entry_time_s(source_tick),
        "physical_state_anchor_tick": source_tick,
        "physical_state_anchor_role": "physical_anchor",
        "predecessor_verify_tick": predecessor_verify_tick,
        "predecessor_verify_time_s": predecessor_verify_time_s,
        "controller_anchor_tick": controller_anchor_tick,
        "controller_anchor_time_s": controller_anchor_time_s,
        "controller_state_anchor_tick": (
            source_tick if source_proven_execute else controller_anchor_tick
        ),
        "controller_state_anchor_role": (
            "source_proven_transition"
            if source_proven_execute
            else "controller_anchor"
            if hybrid
            else None
        ),
        "latch_snapshot_anchor_tick": (
            source_tick
            if source_proven_execute
            else controller_anchor_tick
            if controller_anchor_tick is not None
            else source_tick
        ),
        "latch_snapshot_anchor_role": (
            "source_proven_transition"
            if source_proven_execute
            else "controller_anchor"
            if hybrid
            else "physical_anchor"
        ),
        "target_entry_tick": target_entry_tick,
        "target_entry_time_s": phase_entry_time_s(target_entry_tick),
        "target_entry_tick_authored": hybrid,
        "source_replay_steps": replay_steps,
        "physical_to_predecessor_verify_replay_steps": (
            physical_to_predecessor_verify_steps
        ),
        "predecessor_verify_to_controller_replay_steps": (
            predecessor_verify_to_controller_steps
        ),
        "physical_to_controller_replay_steps": physical_to_controller_steps,
        "controller_to_target_replay_steps": controller_to_target_steps,
        "hybrid_physical_controller_anchor": hybrid,
        "source_replay_first_tick": source_tick,
        "source_replay_last_tick": target_entry_tick - 1,
        "source_replay_context_transition_tick": controller_anchor_tick,
        "source_replay_context_transition_ticks": (
            []
            if predecessor_verify_tick is None or controller_anchor_tick is None
            else [predecessor_verify_tick, controller_anchor_tick]
        ),
        "source_replay_fsm_contexts": contexts,
        "source_replay_context_segments": segments,
        "all_source_replay_contexts_verified": True,
    }
    if source_proven_execute:
        assert restore_contract is not None
        result.update(
            {
                "controller_restore_mode": SOURCE_PROVEN_EXECUTE_RESTORE_MODE,
                "controller_restore_contract": dict(restore_contract),
                "source_transition_verified": True,
                "source_transition_tick": restore_contract[
                    "source_transition_tick"
                ],
                "source_transition_time_s": restore_contract[
                    "source_transition_time_s"
                ],
                "source_transition_row_canonical_sha256": restore_contract[
                    "source_transition_row_canonical_sha256"
                ],
                "consumed_motion_tick": restore_contract["consumed_motion_tick"],
                "first_episode_motion_tick": restore_contract[
                    "first_episode_motion_tick"
                ],
                "entry_guards_from_source_not_replayed": True,
            }
        )
    return result


def _binary64_hex(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise EffectivePhaseEntryError("binary64 value must be finite")
    return struct.pack(">d", number).hex()


def _binary64_from_hex(value: Any, label: str) -> float:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise EffectivePhaseEntryError(f"{label} must be 16 lowercase binary64 hex digits")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(number):
        raise EffectivePhaseEntryError(f"{label} must encode a finite binary64")
    return number


def binary64_ulp_distance(left: float, right: float) -> int:
    """Return exact IEEE-754 distance for finite non-negative fingerprints."""

    first = float(left)
    second = float(right)
    if (
        not math.isfinite(first)
        or not math.isfinite(second)
        or first < 0.0
        or second < 0.0
    ):
        raise EffectivePhaseEntryError(
            "fingerprint ULP comparison requires finite non-negative values"
        )
    first_bits = struct.unpack(">Q", struct.pack(">d", first))[0]
    second_bits = struct.unpack(">Q", struct.pack(">d", second))[0]
    return abs(first_bits - second_bits)


def _decode_object(value: bytes, *, label: str, path: Path) -> dict[str, Any]:
    def object_without_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise EffectivePhaseEntryError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = item
        return result

    def reject_constant(constant: str) -> Any:
        raise EffectivePhaseEntryError(
            f"non-finite JSON constant in {label}: {constant}"
        )

    try:
        result = json.loads(
            value,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except EffectivePhaseEntryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EffectivePhaseEntryError(f"invalid {label} JSON: {path}: {exc}") from exc
    if not isinstance(result, Mapping):
        raise EffectivePhaseEntryError(f"{label} must be a JSON object: {path}")
    return dict(result)


def _absolute_unredirected(path: Path | str, *, label: str) -> Path:
    unresolved = Path(os.path.abspath(os.fspath(Path(path))))
    resolved = unresolved.resolve()
    if unresolved != resolved:
        raise EffectivePhaseEntryError(f"{label} must not traverse a symlink/reparse redirect")
    return resolved


def _identity_from_status(
    path: Path,
    status: os.stat_result,
    *,
    label: str,
    directory: bool,
) -> tuple[Any, ...]:
    attributes = int(getattr(status, "st_file_attributes", 0))
    reparse_tag = int(getattr(status, "st_reparse_tag", 0))
    if stat.S_ISLNK(status.st_mode) or attributes & 0x400 or reparse_tag:
        raise EffectivePhaseEntryError(f"{label} must not be a symlink/reparse point")
    if directory and not stat.S_ISDIR(status.st_mode):
        raise EffectivePhaseEntryError(f"{label} is not a directory: {path}")
    if not directory and not stat.S_ISREG(status.st_mode):
        raise EffectivePhaseEntryError(f"{label} is not a regular file: {path}")
    return (
        str(path),
        "directory" if directory else "file",
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
        attributes,
        int(status.st_mode),
        int(status.st_nlink),
        reparse_tag,
    )


def _path_identity(path: Path, *, label: str, directory: bool) -> tuple[Any, ...]:
    try:
        status = path.lstat()
    except OSError as exc:
        raise EffectivePhaseEntryError(f"{label} is missing: {path}") from exc
    return _identity_from_status(
        path, status, label=label, directory=directory
    )


def _handle_identity(
    descriptor: int, path: Path, *, label: str, directory: bool
) -> tuple[Any, ...]:
    try:
        status = os.fstat(descriptor)
    except OSError as exc:
        raise EffectivePhaseEntryError(
            f"could not inspect opened {label}: {path}"
        ) from exc
    return _identity_from_status(
        path,
        status,
        label=f"opened {label}",
        directory=directory,
    )


def _same_opened_object(
    path_identity: tuple[Any, ...], handle_identity: tuple[Any, ...]
) -> bool:
    # Python 3.13 on Windows exposes creation time as path ``st_ctime`` while
    # ``fstat`` still reports a change-time-like value.  Compare every field
    # that has identical handle/path semantics; each side is still compared
    # against itself before/after with its own ctime included.
    comparable_indexes = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)
    return all(
        path_identity[index] == handle_identity[index]
        for index in comparable_indexes
    )


def _same_path_identity(
    expected: tuple[Any, ...], current: tuple[Any, ...]
) -> bool:
    """Compare files exactly while ignoring ordinary directory churn.

    Directory size, timestamps, and link counts may change when an unrelated
    sibling is created (notably under the shared Windows temporary directory).
    Those fields do not identify the directory object and made immutable
    captures spuriously fail while other processes were active.  Device/inode,
    type, mode, attributes, and reparse tag still detect replacement or path
    redirection; regular files retain the full byte-affecting comparison.
    """

    if expected[1] != current[1]:
        return False
    if expected[1] != "directory":
        return expected == current
    stable_indexes = (0, 1, 2, 3, 7, 8, 10)
    return all(expected[index] == current[index] for index in stable_indexes)


def _same_filesystem_identity(
    expected: tuple[tuple[Any, ...], ...],
    current: tuple[tuple[Any, ...], ...],
) -> bool:
    return len(expected) == len(current) and all(
        _same_path_identity(expected_row, current_row)
        for expected_row, current_row in zip(expected, current)
    )


def _directory_surface(directory: Path) -> tuple[Path, ...]:
    return tuple(reversed(directory.parents)) + (directory,)


def _read_regular_file_once(
    path: Path,
    *,
    label: str,
    expected_path_identity: tuple[Any, ...],
) -> bytes:
    flags = os.O_RDONLY
    for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, flag_name, 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(path), flags)
        handle_before = _handle_identity(
            descriptor, path, label=label, directory=False
        )
        if not _same_opened_object(expected_path_identity, handle_before):
            raise EffectivePhaseEntryError(
                f"opened {label} differs from its checked path identity"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        handle_after = _handle_identity(
            descriptor, path, label=label, directory=False
        )
        if handle_after != handle_before or len(payload) != int(handle_after[4]):
            raise EffectivePhaseEntryError(f"{label} changed while its handle was read")
        path_after = _path_identity(path, label=label, directory=False)
        if path_after != expected_path_identity or not _same_opened_object(
            path_after, handle_after
        ):
            raise EffectivePhaseEntryError(f"{label} path changed while its handle was read")
        return payload
    except EffectivePhaseEntryError:
        raise
    except OSError as exc:
        raise EffectivePhaseEntryError(f"could not read {label}: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _capture_paths_once(
    files: Mapping[str, Path], *, directory: Path | None = None
) -> tuple[dict[str, bytes], tuple[tuple[Any, ...], ...]]:
    resolved_files = {
        label: _absolute_unredirected(raw_path, label=label)
        for label, raw_path in files.items()
    }
    resolved_directory = (
        None
        if directory is None
        else _absolute_unredirected(directory, label="capture directory")
    )
    surface_paths: list[Path] = []
    surface_seen: set[str] = set()

    def add_surface(item: Path) -> None:
        for ancestor in _directory_surface(item):
            key = str(ancestor).casefold() if os.name == "nt" else str(ancestor)
            if key not in surface_seen:
                surface_seen.add(key)
                surface_paths.append(ancestor)

    if resolved_directory is not None:
        add_surface(resolved_directory)
    for path in resolved_files.values():
        add_surface(path.parent)
    surface_identities = [
        _path_identity(item, label="capture directory surface", directory=True)
        for item in surface_paths
    ]
    file_identities: list[tuple[Any, ...]] = []
    payloads: dict[str, bytes] = {}
    for label, path in resolved_files.items():
        identity = _path_identity(path, label=label, directory=False)
        payloads[label] = _read_regular_file_once(
            path, label=label, expected_path_identity=identity
        )
        file_identities.append(identity)
    identities = surface_identities + file_identities
    for identity in identities:
        current = _path_identity(
            Path(identity[0]),
            label=f"captured path {identity[0]}",
            directory=identity[1] == "directory",
        )
        if not _same_path_identity(identity, current):
            raise EffectivePhaseEntryError(f"captured path changed: {identity[0]}")
    return payloads, tuple(identities)


def _file_record(path: Path, payload: bytes, *, root: Path | None = None) -> dict[str, Any]:
    shown_path = path
    if root is not None:
        try:
            shown_path = path.relative_to(root)
        except ValueError:
            # Explicit calibration inputs may live outside the checkout.  Such
            # evidence remains host-bound by its canonical absolute path.
            shown_path = path
    shown = str(shown_path).replace("\\", "/")
    return {"path": shown, "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _record_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EffectivePhaseEntryError(f"{label} path is invalid")
    raw = Path(value)
    if not raw.is_absolute():
        if ".." in raw.parts:
            raise EffectivePhaseEntryError(f"{label} path escapes the project root")
        raw = PROJECT_ROOT.joinpath(*raw.parts)
    return _absolute_unredirected(raw, label=label)


def _validated_runtime_identity(
    value: Any, *, expected_git_commit: str | None
) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise EffectivePhaseEntryError("calibration runtime identity must be an object")
    commit = _require_git_commit(value.get("git_commit"), "runtime git commit")
    if expected_git_commit is not None and commit != expected_git_commit:
        raise EffectivePhaseEntryError("calibration runtime commit differs")
    files = value.get("files")
    if (
        value.get("schema") != "wlr50_clean.committed_runtime_identity.v1"
        or not isinstance(files, list)
        or not files
        or value.get("file_count") != len(files)
    ):
        raise EffectivePhaseEntryError("calibration runtime identity header is invalid")
    ordered_fields = (
        "path",
        "bytes",
        "sha256",
        "creation_time_utc_ticks",
        "last_write_time_utc_ticks",
    )
    paths: list[str] = []
    normalized: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        if (
            not isinstance(row, Mapping)
            or set(row) != set(ordered_fields)
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or any(
                type(row.get(field)) is not int or row[field] <= 0
                for field in (
                    "creation_time_utc_ticks",
                    "last_write_time_utc_ticks",
                )
            )
        ):
            raise EffectivePhaseEntryError(
                f"calibration runtime identity row {index} is invalid"
            )
        digest = _require_sha256(row.get("sha256"), f"runtime file {index} SHA")
        paths.append(row["path"])
        normalized.append({field: row[field] for field in ordered_fields})
        content.append(
            {"path": row["path"], "bytes": row["bytes"], "sha256": digest}
        )
    if tuple(paths) != tuple(sorted(set(paths))):
        raise EffectivePhaseEntryError("calibration runtime inventory is not unique/ordered")
    content_sha = _require_sha256(value.get("content_sha256"), "runtime content SHA")
    aggregate_sha = _require_sha256(
        value.get("aggregate_sha256"), "runtime aggregate SHA"
    )
    encoded_content = json.dumps(
        content, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    encoded_aggregate = json.dumps(
        normalized, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if _sha256_bytes(encoded_content) != content_sha:
        raise EffectivePhaseEntryError("calibration runtime content SHA is invalid")
    if _sha256_bytes(encoded_aggregate) != aggregate_sha:
        raise EffectivePhaseEntryError("calibration runtime aggregate SHA is invalid")
    return commit, content_sha


def _validate_manifest_config_set(
    manifest: Mapping[str, Any], *, expected_config_sha256: str
) -> None:
    rows = manifest.get("configs")
    if not isinstance(rows, list) or not rows:
        raise EffectivePhaseEntryError("calibration config inventory is missing")
    names: list[str] = []
    paths: dict[str, Path] = {}
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
        ):
            raise EffectivePhaseEntryError(
                f"calibration config record {index} is invalid"
            )
        _require_sha256(row.get("sha256"), f"calibration config {index} SHA")
        names.append(row["path"])
        paths[row["path"]] = PROJECT_ROOT.joinpath(*Path(row["path"]).parts)
    if tuple(names) != tuple(sorted(set(names))):
        raise EffectivePhaseEntryError("calibration config inventory is not unique/ordered")
    captured, _ = _capture_paths_once(paths)
    digest = hashlib.sha256()
    for row in rows:
        name = str(row["path"])
        data = captured[name]
        if row["bytes"] != len(data) or row["sha256"] != _sha256_bytes(data):
            raise EffectivePhaseEntryError(
                f"calibration config differs from recorded identity: {name}"
            )
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    if digest.hexdigest() != expected_config_sha256:
        raise EffectivePhaseEntryError("calibration config-set SHA is invalid")


def _raw_class(ground: bool, obstacle: bool) -> str:
    if ground and obstacle:
        raise EffectivePhaseEntryError("ground and obstacle cannot both be active")
    if obstacle:
        return "OBSTACLE"
    if ground:
        return "GROUND"
    return "AIR"


def _calibrated_contacts(comparison: Mapping[str, Any]) -> dict[str, Any]:
    raw = comparison.get("raw_physx_contacts")
    pairs = raw.get("pairs") if isinstance(raw, Mapping) else None
    classified = comparison.get("exact_contacts")
    if not isinstance(pairs, Mapping) or not isinstance(classified, Mapping):
        raise EffectivePhaseEntryError("calibration omits raw/classified contacts")
    result: dict[str, Any] = {}
    signature: dict[str, Any] = {}
    for wheel in WHEEL_ORDER:
        wheel_pairs = pairs.get(wheel)
        class_row = classified.get(wheel)
        if not isinstance(wheel_pairs, Mapping) or not isinstance(class_row, Mapping):
            raise EffectivePhaseEntryError(f"calibration contact is missing {wheel}")
        pair_result: dict[str, Any] = {}
        active: dict[str, bool] = {}
        for pair_name in PAIR_ORDER:
            pair = wheel_pairs.get(pair_name)
            if not isinstance(pair, Mapping):
                raise EffectivePhaseEntryError(f"calibration contact lacks {wheel}.{pair_name}")
            try:
                force = tuple(float(value) for value in pair.get("force_w_n", ()))
            except (TypeError, ValueError) as exc:
                raise EffectivePhaseEntryError("calibration contact force is invalid") from exc
            if len(force) != 3 or any(not math.isfinite(value) for value in force):
                raise EffectivePhaseEntryError("calibration contact force must be finite vec3")
            force_norm = math.sqrt(sum(value * value for value in force))
            if force_norm >= CONTACT_FORCE_ON_N:
                is_active = True
            elif force_norm < CONTACT_FORCE_OFF_N:
                is_active = False
            else:
                raise EffectivePhaseEntryError(
                    f"calibration force lies in hysteresis gap: {wheel}.{pair_name}"
                )
            if pair.get("pair_verified") is not True or pair.get("source") != CONTACT_SOURCE:
                raise EffectivePhaseEntryError(
                    f"calibration raw contact source is unverified: {wheel}.{pair_name}"
                )
            active[pair_name] = is_active
            pair_result[pair_name] = {
                "active": is_active,
                "pair_verified": True,
                "source": CONTACT_SOURCE,
                "force_w_n": list(force),
                "force_w_n_binary64_hex": [_binary64_hex(value) for value in force],
                "force_norm_n": force_norm,
                "force_norm_n_binary64_hex": _binary64_hex(force_norm),
            }
        raw_class = _raw_class(active["ground"], active["obstacle"])
        actual_class = str(class_row.get("actual_class"))
        actual_ground = class_row.get("actual_ground_active")
        actual_obstacle = class_row.get("actual_obstacle_active")
        body_name = class_row.get("body_name")
        if (
            actual_class != raw_class
            or actual_ground is not active["ground"]
            or actual_obstacle is not active["obstacle"]
            or body_name != wheel.replace("_ankle", "_wheel")
        ):
            raise EffectivePhaseEntryError(
                f"calibration classifier differs from raw contact for {wheel}"
            )
        result[wheel] = {
            "body_name": body_name,
            "classification": raw_class,
            **pair_result,
        }
        signature[wheel] = {
            "body_name": result[wheel]["body_name"],
            "classification": raw_class,
            "ground_active": active["ground"],
            "obstacle_active": active["obstacle"],
            "pair_verified": {name: True for name in PAIR_ORDER},
            "source": {name: CONTACT_SOURCE for name in PAIR_ORDER},
        }
    result["signature_sha256"] = _sha256_bytes(_canonical_bytes(signature))
    return result


def _fingerprint(value: Any) -> tuple[dict[str, float], dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(FINGERPRINT_FIELDS):
        raise EffectivePhaseEntryError("post-prime fingerprint must contain exactly nine fields")
    values: dict[str, float] = {}
    binary: dict[str, str] = {}
    for field in FINGERPRINT_FIELDS:
        try:
            number = float(value[field])
        except (TypeError, ValueError) as exc:
            raise EffectivePhaseEntryError(f"invalid fingerprint field {field}") from exc
        if not math.isfinite(number) or number < 0.0:
            raise EffectivePhaseEntryError(f"fingerprint field {field} must be finite/nonnegative")
        values[field] = number
        binary[field] = _binary64_hex(number)
    return values, binary


def _component_vector(value: Any, *, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise EffectivePhaseEntryError(f"{label} must be an exact Full{size} vector")
    if any(type(item) is bool for item in value):
        raise EffectivePhaseEntryError(f"{label} must not contain booleans")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise EffectivePhaseEntryError(f"{label} contains an invalid binary64 value") from exc
    if any(not math.isfinite(item) for item in result):
        raise EffectivePhaseEntryError(f"{label} must contain only finite values")
    return result


def _component_state(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and canonicalize the complete post-replay physical state."""

    expected_fields = {
        "schema",
        "units",
        *COMPONENT_VECTOR_SIZES,
        "servo_order",
        "wheel_order",
        *COMPONENT_GEOMETRY_FIELDS,
        "sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise EffectivePhaseEntryError(
            "effective component state fields are incomplete or unexpected"
        )
    if (
        value.get("schema") != COMPONENT_STATE_SCHEMA
        or value.get("units") != COMPONENT_UNITS
        or tuple(value.get("servo_order", ())) != SERVO_ORDER
        or tuple(value.get("wheel_order", ())) != WHEEL_ORDER
    ):
        raise EffectivePhaseEntryError(
            "effective component state schema/order/units are invalid"
        )
    declared_sha = _require_sha256(
        value.get("sha256"), "effective component state SHA"
    )
    unhashed = _thaw_json_value(value)
    unhashed.pop("sha256", None)
    if _sha256_bytes(_canonical_bytes(unhashed)) != declared_sha:
        raise EffectivePhaseEntryError("effective component state SHA mismatch")

    normalized: dict[str, Any] = {
        "schema": COMPONENT_STATE_SCHEMA,
        "units": dict(COMPONENT_UNITS),
    }
    binary: dict[str, Any] = {}
    for field, size in COMPONENT_VECTOR_SIZES.items():
        vector = _component_vector(value.get(field), size=size, label=field)
        normalized[field] = vector
        binary[field] = [_binary64_hex(item) for item in vector]
    orientation = normalized["root_orientation_wxyz"]
    norm = math.sqrt(sum(item * item for item in orientation))
    first_nonzero = next((item for item in orientation if item != 0.0), None)
    if abs(norm - 1.0) > 2.0e-15 or (
        first_nonzero is not None and first_nonzero < 0.0
    ):
        raise EffectivePhaseEntryError(
            "effective component orientation is not normalized/canonical-sign"
        )
    normalized["servo_order"] = list(value["servo_order"])
    normalized["wheel_order"] = list(value["wheel_order"])
    for field in COMPONENT_GEOMETRY_FIELDS:
        rows = value.get(field)
        if not isinstance(rows, Mapping) or set(rows) != set(WHEEL_ORDER):
            raise EffectivePhaseEntryError(
                f"effective component geometry is invalid: {field}"
            )
        normalized[field] = {}
        binary[field] = {}
        for wheel in WHEEL_ORDER:
            vector = _component_vector(
                rows[wheel], size=3, label=f"{field}.{wheel}"
            )
            normalized[field][wheel] = vector
            binary[field][wheel] = [_binary64_hex(item) for item in vector]
    if _sha256_bytes(_canonical_bytes(normalized)) != declared_sha:
        raise EffectivePhaseEntryError(
            "effective component state is not canonical binary64 JSON"
        )
    normalized["sha256"] = declared_sha
    return normalized, binary


def _component_state_scalar_items(
    state: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    rows: list[tuple[str, float]] = []
    for field in COMPONENT_VECTOR_SIZES:
        rows.extend(
            (f"{field}[{index}]", float(item))
            for index, item in enumerate(state[field])
        )
    for field in COMPONENT_GEOMETRY_FIELDS:
        for wheel in WHEEL_ORDER:
            rows.extend(
                (f"{field}.{wheel}[{index}]", float(item))
                for index, item in enumerate(state[field][wheel])
            )
    return tuple(rows)


def _component_binary64_ulp_distance(left: float, right: float) -> int:
    """Return an ordered IEEE-754 distance that also supports negative values."""

    first = float(left)
    second = float(right)
    if not math.isfinite(first) or not math.isfinite(second):
        raise EffectivePhaseEntryError(
            "component-state ULP comparison requires finite values"
        )

    def ordered(value: float) -> int:
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        if bits & (1 << 63):
            return (~bits) & ((1 << 64) - 1)
        return bits | (1 << 63)

    return abs(ordered(first) - ordered(second))


def _validate_bound_artifact(
    record: Any, payload: bytes, *, expected_path: str, label: str
) -> None:
    if not isinstance(record, Mapping):
        raise EffectivePhaseEntryError(f"run manifest omits {label}")
    if (
        record.get("path") != expected_path
        or record.get("bytes") != len(payload)
        or record.get("sha256") != _sha256_bytes(payload)
    ):
        raise EffectivePhaseEntryError(f"run manifest binding differs for {label}")


def _validate_controller_entry_guard(
    value: Any, *, phase: str, snapshot_payload: Mapping[str, Any]
) -> None:
    restore_contract = _validated_source_proven_execute_restore(
        snapshot_payload, phase
    )
    expected_fields = {
        "schema",
        "verified",
        "phase",
        "lifecycle",
        "nonterminal",
        "unblocked",
        "authored_entry_guard_names",
        "entry_guard_evidence",
        "p10_signed_velocity_alignment",
    }
    if restore_contract is not None:
        expected_fields = {
            "schema",
            "verified",
            "phase",
            "lifecycle",
            "nonterminal",
            "unblocked",
            "mode",
            "source_transition_verified",
            "source_transition_tick",
            "source_transition_time_s",
            "source_transition_row_canonical_sha256",
            "authored_entry_guard_names",
            "entry_guard_evidence",
            "p10_signed_velocity_alignment",
            "entry_guards_from_source_not_replayed",
            "new_entry_event_count",
            "consumed_motion_tick",
            "first_episode_motion_tick",
            "motion_tick_after_first_episode_frame",
            "completion_guards_remain_pending",
            "completion_guard_names",
        }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise EffectivePhaseEntryError("calibration controller entry proof is malformed")
    evidence = value.get("entry_guard_evidence")
    if (
        value.get("schema")
        != (
            "wlr50_clean.phase_effective_entry_controller.v2"
            if restore_contract is not None
            else "wlr50_clean.phase_effective_entry_controller.v1"
        )
        or value.get("verified") is not True
        or value.get("phase") != phase
        or value.get("lifecycle") != "EXECUTE_MOTION"
        or value.get("nonterminal") is not True
        or value.get("unblocked") is not True
        or value.get("authored_entry_guard_names") != list(_AUTHORED_ENTRY_GUARDS)
        or not isinstance(evidence, list)
        or len(evidence) != len(_AUTHORED_ENTRY_GUARDS)
    ):
        raise EffectivePhaseEntryError("calibration controller entry guard did not pass")
    if restore_contract is not None and (
        value.get("mode")
        != SOURCE_PROVEN_EXECUTE_RESTORE_MODE
        or value.get("source_transition_verified") is not True
        or value.get("source_transition_tick")
        != restore_contract["source_transition_tick"]
        or value.get("source_transition_time_s")
        != restore_contract["source_transition_time_s"]
        or value.get("source_transition_row_canonical_sha256")
        != restore_contract["source_transition_row_canonical_sha256"]
        or value.get("consumed_motion_tick")
        != restore_contract["consumed_motion_tick"]
        or value.get("first_episode_motion_tick")
        != restore_contract["first_episode_motion_tick"]
        or value.get("motion_tick_after_first_episode_frame")
        != restore_contract["first_episode_motion_tick"] + 1
        or value.get("entry_guards_from_source_not_replayed") is not True
        or value.get("new_entry_event_count") != 0
        or value.get("completion_guards_remain_pending") is not True
        or value.get("completion_guard_names")
        != list(_P10_COMPLETION_GUARDS)
        or value.get("authored_entry_guard_names")
        != restore_contract["authored_entry_guard_order"]
        or evidence != restore_contract["authored_entry_guard_evidence"]
    ):
        raise EffectivePhaseEntryError(
            "P10 controller proof is not bound to the source-proven EXECUTE transition"
        )
    if tuple(
        row.get("name") for row in evidence if isinstance(row, Mapping)
    ) != _AUTHORED_ENTRY_GUARDS:
        raise EffectivePhaseEntryError("calibration authored entry guards are not exact/ordered")
    for row in evidence:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "passed", "value", "source", "reason"}
            or row.get("passed") is not True
            or not isinstance(row.get("source"), str)
            or not row["source"]
            or not isinstance(row.get("reason"), str)
        ):
            raise EffectivePhaseEntryError("calibration controller guard evidence is invalid")
    by_name = {str(row["name"]): row for row in evidence}
    if by_name["previous_state_done"].get("value") is not True:
        raise EffectivePhaseEntryError("calibration previous-state guard is not proven")
    critical = by_name["critical_actuators_available"].get("value")
    if critical != {"joint_count": 8, "wheel_count": 4}:
        raise EffectivePhaseEntryError("calibration critical-actuator guard is invalid")
    reference = by_name["reference_entry_compatible"].get("value")
    if not isinstance(reference, Mapping):
        raise EffectivePhaseEntryError("calibration reference-entry guard lacks evidence")
    velocity_rows = {
        str(name): row
        for name, row in reference.items()
        if str(name).endswith("_velocity") and isinstance(row, Mapping)
    }
    alignment = value.get("p10_signed_velocity_alignment")
    if phase != "P10":
        if alignment is not None or velocity_rows:
            raise EffectivePhaseEntryError("non-P10 calibration has signed-velocity evidence")
        return
    if set(velocity_rows) != {"rear_right_knee_velocity"}:
        raise EffectivePhaseEntryError("P10 calibration lacks its unique velocity guard")
    row = velocity_rows["rear_right_knee_velocity"]
    if alignment != row or set(row) != {
        "actual_deg_s",
        "reference_deg_s",
        "error_deg_s",
        "limit_deg_s",
        "signed_positive_rebound_required",
    }:
        raise EffectivePhaseEntryError("P10 signed-velocity alignment is inconsistent")
    try:
        actual = float(row["actual_deg_s"])
        reference_velocity = float(row["reference_deg_s"])
        error = float(row["error_deg_s"])
        limit = float(row["limit_deg_s"])
    except (TypeError, ValueError, KeyError) as exc:
        raise EffectivePhaseEntryError("P10 signed-velocity values are invalid") from exc
    if (
        any(not math.isfinite(item) for item in (actual, reference_velocity, error, limit))
        or actual <= 0.0
        or reference_velocity <= 0.0
        or limit < 0.0
        or row.get("signed_positive_rebound_required") is not True
        or not math.isclose(error, actual - reference_velocity, rel_tol=0.0, abs_tol=1e-12)
        or abs(error) > limit + 1e-12
    ):
        raise EffectivePhaseEntryError("P10 signed positive velocity guard did not pass")


def _replay_evidence_failures(
    state: Mapping[str, Any],
    snapshot_payload: Mapping[str, Any],
    *,
    replay_steps: int,
    target_entry_tick: int,
    control_ticks: Sequence[int],
    predecessor_verify_tick: int | None,
    predecessor_verify_time_s: float | None,
    controller_anchor_tick: int | None,
    controller_anchor_time_s: float | None,
) -> tuple[str, ...]:
    """Validate every replay write/read against the pinned command sequence."""

    failures: list[str] = []
    commands = snapshot_payload.get("source_commands")
    if not isinstance(commands, list) or len(commands) != replay_steps:
        return ("pinned source-command sequence length",)
    actuation_matches = state.get("source_actuation_matches")
    mapper_states = state.get("source_mapper_post_states")
    atomic_writes = state.get("prime_atomic_writes")
    safety_checks = state.get("source_replay_safety_checks")
    if not isinstance(actuation_matches, list) or len(actuation_matches) != replay_steps:
        failures.append("source_actuation_matches length")
    if not isinstance(mapper_states, list) or len(mapper_states) != replay_steps:
        failures.append("source_mapper_post_states length")
    if not isinstance(atomic_writes, list) or len(atomic_writes) != replay_steps:
        failures.append("prime_atomic_writes length")
    if not isinstance(safety_checks, list) or len(safety_checks) != replay_steps:
        failures.append("source_replay_safety_checks length")
    if failures:
        return tuple(failures)

    assert isinstance(actuation_matches, list)
    assert isinstance(mapper_states, list)
    assert isinstance(atomic_writes, list)
    assert isinstance(safety_checks, list)
    previous_live_mapper_post_sha256: str | None = None
    expected_anchor_contract = _expected_replay_anchor_contract(
        snapshot_payload,
        str(snapshot_payload["fsm_state"]),
        replay_steps=replay_steps,
        target_entry_tick=target_entry_tick,
        control_ticks=control_ticks,
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
    )
    for index, (command, expected_tick) in enumerate(
        zip(commands, control_ticks, strict=True)
    ):
        reset_physics_tick = _RESET_SETTLE_TICKS + index
        reset_write_count = reset_physics_tick + 1
        actuation = actuation_matches[index]
        mapper = mapper_states[index]
        atomic = atomic_writes[index]
        safety = safety_checks[index]
        if not isinstance(actuation, Mapping):
            failures.append(f"source_actuation_matches[{index}]")
            continue
        invariant_matches = actuation.get("replay_invariant_field_matches")
        diagnostic_matches = actuation.get("historical_feedback_field_matches")
        output_contract = actuation.get("live_output_contract")
        feedback_input = actuation.get("live_feedback_input")
        feedback_unhashed = (
            dict(feedback_input) if isinstance(feedback_input, Mapping) else {}
        )
        feedback_sha256 = feedback_unhashed.pop("sha256", None)
        source_target_matches = (
            actuation.get("source_drive_target_full12_sha256")
            == actuation.get("replayed_drive_target_full12_sha256")
        )
        source_actuation_matches = (
            actuation.get("source_actuation_contract_sha256")
            == actuation.get("replayed_actuation_contract_sha256")
        )
        if (
            actuation.get("schema")
            != "wlr50_clean.phase_snapshot_source_input_live_output.v2"
            or actuation.get("source_control_physics_tick") != expected_tick
            or actuation.get("all_replay_invariant_fields_match") is not True
            or not isinstance(invariant_matches, Mapping)
            or set(invariant_matches) != set(SOURCE_ACK_REPLAY_INVARIANT_FIELDS)
            or any(value is not True for value in invariant_matches.values())
            or not isinstance(diagnostic_matches, Mapping)
            or set(diagnostic_matches)
            != set(SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS)
            or any(type(value) is not bool for value in diagnostic_matches.values())
            or actuation.get("historical_feedback_equivalence_claimed") is not False
            or actuation.get("live_feedback_adaptive_output_valid") is not True
            or actuation.get("logical_target_fallback_used") is not False
            or actuation.get("source_command_file_sha256")
            != snapshot_payload.get("source_artifacts", {}).get("command", {}).get(
                "sha256"
            )
            or actuation.get("source_observation_file_sha256")
            != snapshot_payload.get("source_artifacts", {}).get(
                "observation", {}
            ).get("sha256")
            or actuation.get("source_command_row_canonical_sha256")
            != command.get("source_command_row_canonical_sha256")
            or actuation.get("source_observation_row_canonical_sha256")
            != command.get("source_observation_row_canonical_sha256")
            or actuation.get("source_adapter_input_sha256")
            != command.get("source_adapter_input_sha256")
            or actuation.get("replayed_adapter_input_sha256")
            != command.get("source_adapter_input_sha256")
            or actuation.get("adapter_input_hash_matches") is not True
            or actuation.get("source_drive_target_full12_sha256")
            != command.get("drive_target_full12_sha256")
            or actuation.get("source_actuation_contract_sha256")
            != command.get("actuation_contract_sha256")
            or actuation.get("source_target_hash_matches")
            is not source_target_matches
            or actuation.get("source_actuation_hash_matches")
            is not source_actuation_matches
            or not isinstance(output_contract, Mapping)
            or set(output_contract)
            != {
                "schema",
                "all_values_finite",
                "logical_clamp_verified",
                "servo_aliases_verified",
                "realized_bias_verified",
                "servo_hard_limits_verified",
                "wheel_hard_limits_verified",
                "final_drive_slew_verified",
                "maximum_final_drive_slew_deg",
                "maximum_allowed_final_drive_slew_deg",
                "physical_sign_order_unit_conversion_verified",
                "mapper_output_state_verified",
                "live_feedback_mapper_replay_verified",
                "live_feedback_input_sha256",
                "predicted_mapper_post_state_sha256",
                "feedback_conditioned_output",
                "feedback_conditioned_output_sha256",
                "verified",
            }
            or output_contract.get("schema")
            != "wlr50_clean.phase_snapshot_live_output_contract.v2"
            or output_contract.get("verified") is not True
            or any(
                output_contract.get(name) is not True
                for name in (
                    "all_values_finite",
                    "logical_clamp_verified",
                    "servo_aliases_verified",
                    "realized_bias_verified",
                    "servo_hard_limits_verified",
                    "wheel_hard_limits_verified",
                    "final_drive_slew_verified",
                    "physical_sign_order_unit_conversion_verified",
                    "mapper_output_state_verified",
                    "live_feedback_mapper_replay_verified",
                )
            )
            or output_contract.get("live_feedback_input_sha256")
            != feedback_sha256
            or _SHA256.fullmatch(
                str(output_contract.get("predicted_mapper_post_state_sha256", ""))
            )
            is None
            or _SHA256.fullmatch(
                str(output_contract.get("feedback_conditioned_output_sha256", ""))
            )
            is None
            or not isinstance(
                output_contract.get("feedback_conditioned_output"), Mapping
            )
            or set(output_contract.get("feedback_conditioned_output", {}))
            != set(SOURCE_ACK_FEEDBACK_DIAGNOSTIC_FIELDS)
            or output_contract.get("feedback_conditioned_output_sha256")
            != _sha256_bytes(
                _canonical_bytes(output_contract.get("feedback_conditioned_output", {}))
            )
            or type(output_contract.get("maximum_final_drive_slew_deg")) is bool
            or not isinstance(
                output_contract.get("maximum_final_drive_slew_deg"), (int, float)
            )
            or not math.isfinite(
                float(output_contract.get("maximum_final_drive_slew_deg", math.nan))
            )
            or float(output_contract.get("maximum_final_drive_slew_deg", -1.0)) < 0.0
            or output_contract.get("maximum_allowed_final_drive_slew_deg")
            != command.get("mapper_configuration", {}).get("maximum_delta_deg")
            or float(output_contract.get("maximum_final_drive_slew_deg", math.inf))
            > float(output_contract.get("maximum_allowed_final_drive_slew_deg", -1.0))
            + 1.0e-12
            or not isinstance(feedback_input, Mapping)
            or set(feedback_input)
            != {
                "schema",
                "canonical_servo_order",
                "unit",
                "physical_position_rad",
                "tensor_dtype",
                "tensor_device",
                "sha256",
            }
            or feedback_input.get("schema")
            != "wlr50_clean.phase_snapshot_live_servo_feedback.v1"
            or feedback_input.get("canonical_servo_order")
            != [
                "front_left_hip",
                "front_left_knee",
                "front_right_hip",
                "front_right_knee",
                "rear_left_hip",
                "rear_left_knee",
                "rear_right_hip",
                "rear_right_knee",
            ]
            or feedback_input.get("unit") != "rad"
            or not isinstance(feedback_input.get("tensor_dtype"), str)
            or not feedback_input.get("tensor_dtype")
            or not isinstance(feedback_input.get("tensor_device"), str)
            or not feedback_input.get("tensor_device")
            or not isinstance(feedback_input.get("physical_position_rad"), list)
            or len(feedback_input.get("physical_position_rad", ())) != len(SERVO_ORDER)
            or any(
                not isinstance(value, (int, float))
                or type(value) is bool
                or not math.isfinite(float(value))
                for value in feedback_input.get("physical_position_rad", ())
            )
            or feedback_sha256 != _sha256_bytes(_canonical_bytes(feedback_unhashed))
            or actuation.get("source_atomic_physics_tick")
            != command.get("source_atomic_physics_tick")
            or actuation.get("source_atomic_write_count")
            != command.get("source_atomic_write_count")
            or actuation.get("reset_prime_physics_tick") != reset_physics_tick
            or actuation.get("reset_prime_write_count") != reset_write_count
            or actuation.get(
                "clock_and_write_count_fields_intentionally_remapped"
            )
            is not True
        ):
            failures.append(f"source_actuation_matches[{index}] contract")
        if not isinstance(mapper, Mapping):
            failures.append(f"source_mapper_post_states[{index}]")
        else:
            mapper_matches = mapper.get("historical_post_field_matches")
            mapper_errors = mapper.get(
                "historical_post_field_maximum_numeric_error"
            )
            if (
                set(mapper)
                != {
                    "schema",
                    "source_transition",
                    "source_control_physics_tick",
                    "first_replay_tick",
                    "first_pre_state_matches_source",
                    "historical_post_field_matches",
                    "historical_post_field_maximum_numeric_error",
                    "all_historical_post_fields_match",
                    "historical_feedback_equivalence_claimed",
                    "source_pre_state_sha256",
                    "source_post_state_sha256",
                    "live_pre_state_sha256",
                    "live_post_state_sha256",
                    "feedback_tick_increment",
                    "feedback_schedule_verified",
                    "ack_cross_bindings_verified",
                    "natural_live_state_continuity_verified",
                    "per_tick_mapper_restore_count",
                    "restored_after_prime",
                    "reached_naturally_by_single_atomic_apply",
                }
                or mapper.get("source_transition")
                != "source_tick_t_minus_1_to_t"
                or mapper.get("schema")
                != "wlr50_clean.phase_snapshot_live_mapper_transition.v2"
                or mapper.get("source_control_physics_tick") != expected_tick
                or not isinstance(mapper_matches, Mapping)
                or set(mapper_matches) != set(command.get("mapper_post_state", {}))
                or any(type(value) is not bool for value in mapper_matches.values())
                or mapper.get("all_historical_post_fields_match")
                is not all(mapper_matches.values())
                or not isinstance(mapper_errors, Mapping)
                or set(mapper_errors) != set(command.get("mapper_post_state", {}))
                or any(
                    type(value) is bool
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in mapper_errors.values()
                )
                or mapper.get("historical_feedback_equivalence_claimed") is not False
                or mapper.get("first_replay_tick") is not (index == 0)
                or mapper.get("first_pre_state_matches_source")
                is not (True if index == 0 else None)
                or mapper.get("source_pre_state_sha256")
                != _sha256_bytes(_canonical_bytes(command.get("mapper_pre_state", {})))
                or mapper.get("source_post_state_sha256")
                != _sha256_bytes(_canonical_bytes(command.get("mapper_post_state", {})))
                or (
                    index == 0
                    and mapper.get("live_pre_state_sha256")
                    != mapper.get("source_pre_state_sha256")
                )
                or _SHA256.fullmatch(str(mapper.get("live_pre_state_sha256", "")))
                is None
                or _SHA256.fullmatch(str(mapper.get("live_post_state_sha256", "")))
                is None
                or (
                    previous_live_mapper_post_sha256 is not None
                    and mapper.get("live_pre_state_sha256")
                    != previous_live_mapper_post_sha256
                )
                or mapper.get("feedback_tick_increment") != 1
                or mapper.get("feedback_schedule_verified") is not True
                or mapper.get("ack_cross_bindings_verified") is not True
                or not isinstance(output_contract, Mapping)
                or output_contract.get("predicted_mapper_post_state_sha256")
                != mapper.get("live_post_state_sha256")
                or mapper.get("natural_live_state_continuity_verified") is not True
                or mapper.get("per_tick_mapper_restore_count") != 0
                or mapper.get("reached_naturally_by_single_atomic_apply") is not True
                or mapper.get("restored_after_prime") is not False
            ):
                failures.append(f"source_mapper_post_states[{index}] contract")
            previous_live_mapper_post_sha256 = (
                str(mapper.get("live_post_state_sha256"))
                if _SHA256.fullmatch(str(mapper.get("live_post_state_sha256", "")))
                else None
            )
        if not isinstance(atomic, Mapping):
            failures.append(f"prime_atomic_writes[{index}]")
        elif (
            atomic.get("source_control_physics_tick") != expected_tick
            or atomic.get("observation_physics_tick") != expected_tick + 1
            or atomic.get("physics_tick") != reset_physics_tick
            or atomic.get("write_count") != reset_write_count
            or atomic.get("articulation_writes_this_call") != 1
            or atomic.get("source_actuation_match") != actuation
            or atomic.get("source_mapper_post_state") != mapper
        ):
            failures.append(f"prime_atomic_writes[{index}] contract")
        safety_flags = safety.get("flags") if isinstance(safety, Mapping) else None
        if (
            not isinstance(safety, Mapping)
            or safety.get("schema") != "wlr50_clean.phase_effective_entry_safety.v1"
            or safety.get("verified") is not True
            or safety.get("all_failure_flags_false") is not True
            or safety.get("source_control_physics_tick") != expected_tick
            or safety.get("observation_physics_tick") != expected_tick + 1
            or not isinstance(safety_flags, Mapping)
            or set(safety_flags)
            != {
                "body_collision",
                "combined_physics_abort_guard",
                "fall",
                "hard_joint_limit",
                "nan_inf",
                "physics_explosion",
                "wheel_only_climb",
            }
            or any(
                type(value) is not bool or value is not False
                for value in safety_flags.values()
            )
        ):
            failures.append(f"source_replay_safety_checks[{index}] contract")

    if state.get("source_actuation_match") != actuation_matches[-1]:
        failures.append("final source_actuation_match alias")
    if state.get("source_mapper_post_state") != mapper_states[-1]:
        failures.append("final source_mapper_post_state alias")
    if state.get("source_adapter_input_sha256s") != [
        command.get("source_adapter_input_sha256") for command in commands
    ]:
        failures.append("source_adapter_input_sha256s")
    for field, expected in (
        ("all_source_adapter_inputs_hash_matched", True),
        ("all_live_output_contracts_verified", True),
        ("all_live_mapper_transitions_verified", True),
        ("live_feedback_adaptive_replay", True),
        ("historical_feedback_equivalence_claimed", False),
        ("initial_mapper_restore_count", 1),
        ("per_tick_mapper_restore_count", 0),
    ):
        if state.get(field) != expected:
            failures.append(field)
    if state.get("source_replay_steps") != replay_steps:
        failures.append("source_replay_steps")
    anchor_bindings = {
        "physical_anchor_tick": expected_anchor_contract["physical_anchor_tick"],
        "physical_anchor_time_s": expected_anchor_contract["physical_anchor_time_s"],
        "predecessor_verify_tick": predecessor_verify_tick,
        "predecessor_verify_time_s": predecessor_verify_time_s,
        "controller_anchor_tick": controller_anchor_tick,
        "controller_anchor_time_s": controller_anchor_time_s,
        "target_entry_time_s": expected_anchor_contract["target_entry_time_s"],
        "physical_to_predecessor_verify_replay_steps": expected_anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ],
        "predecessor_verify_to_controller_replay_steps": expected_anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ],
        "physical_to_controller_replay_steps": expected_anchor_contract[
            "physical_to_controller_replay_steps"
        ],
        "controller_to_target_replay_steps": expected_anchor_contract[
            "controller_to_target_replay_steps"
        ],
        "hybrid_physical_controller_anchor": expected_anchor_contract[
            "hybrid_physical_controller_anchor"
        ],
        "replay_anchor_contract": expected_anchor_contract,
        "source_replay_fsm_contexts": expected_anchor_contract[
            "source_replay_fsm_contexts"
        ],
    }
    for name, expected in anchor_bindings.items():
        if name not in state or state.get(name) != expected:
            failures.append(name)
    if state.get("target_entry_tick") != target_entry_tick:
        failures.append("target_entry_tick")
    if state.get("episode_sensor_tick_offset") != target_entry_tick:
        failures.append("episode_sensor_tick_offset")
    if state.get("all_source_replay_steps_safe") is not True:
        failures.append("all_source_replay_steps_safe")
    if state.get("source_replay_observation_ticks") != list(
        range(
            int(snapshot_payload["source_tick"]) + 1,
            target_entry_tick + 1,
        )
    ):
        failures.append("source_replay_observation_ticks")
    return tuple(failures)


def _validated_probe_attempt(
    attempt: Any,
    *,
    phase: str,
    lifecycle: str,
    phase_snapshot_bundle: ValidatedPhaseSnapshotBundle,
) -> Mapping[str, Any]:
    if not isinstance(attempt, Mapping):
        raise EffectivePhaseEntryError("calibration attempt must be an object")
    state = attempt.get("snapshot_state_write")
    if not isinstance(state, Mapping):
        raise EffectivePhaseEntryError("calibration attempt omits snapshot-state proof")
    root = state.get("pre_prime_root_link_readback")
    actuation = state.get("source_actuation_match")
    mapper = state.get("source_mapper_post_state")
    clocks = attempt.get("clocks")
    observation = attempt.get("observation_diagnostics")
    safety = state.get("entry_safety_contract")
    safety_flags = safety.get("flags") if isinstance(safety, Mapping) else None
    sensor = state.get("entry_sensor_contract")
    effective_proof = state.get("effective_entry_contract")
    comparison = state.get("source_snapshot_post_prime_diagnostic")
    (
        snapshot_payload,
        snapshot,
        replay_steps,
        target_entry_tick,
        control_ticks,
        predecessor_verify_tick,
        predecessor_verify_time_s,
        controller_anchor_tick,
        controller_anchor_time_s,
    ) = _snapshot_replay_contract(phase_snapshot_bundle, phase)
    restore_bindings = _source_proven_execute_bindings(snapshot_payload, phase)
    calibration_proof_fields = {
        "schema",
        "artifact_role",
        "verified",
        "calibration_only",
        "phase",
        "source_tick",
        "physical_anchor_tick",
        "physical_anchor_time_s",
        "predecessor_verify_tick",
        "predecessor_verify_time_s",
        "controller_anchor_tick",
        "controller_anchor_time_s",
        "target_entry_tick",
        "target_entry_time_s",
        "source_replay_steps",
        "physical_to_predecessor_verify_replay_steps",
        "predecessor_verify_to_controller_replay_steps",
        "physical_to_controller_replay_steps",
        "controller_to_target_replay_steps",
        "hybrid_physical_controller_anchor",
        "replay_anchor_contract",
        "effective_entry_offset_s",
        "phase_snapshot_bundle_sha256",
        "source_snapshot_post_prime_diagnostic",
        "failures",
        *restore_bindings,
    }
    try:
        _validate_controller_entry_guard(
            state.get("entry_guard_contract"),
            phase=phase,
            snapshot_payload=snapshot_payload,
        )
    except EffectivePhaseEntryError:
        controller_entry_proof_valid = False
    else:
        controller_entry_proof_valid = True
    replay_failures = _replay_evidence_failures(
        state,
        snapshot_payload,
        replay_steps=replay_steps,
        target_entry_tick=target_entry_tick,
        control_ticks=control_ticks,
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
    )
    source_commands = snapshot_payload["source_commands"]
    expected_anchor_contract = _expected_replay_anchor_contract(
        snapshot_payload,
        phase,
        replay_steps=replay_steps,
        target_entry_tick=target_entry_tick,
        control_ticks=control_ticks,
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
    )
    expected_scalars = {
        "state_write_count": 1,
        "root_pose_writes": 1,
        "root_velocity_writes": 1,
        "joint_state_writes": 1,
        "simulation_forward_syncs": 1,
        "source_replay_steps": replay_steps,
        "predecessor_verify_tick": predecessor_verify_tick,
        "predecessor_verify_time_s": predecessor_verify_time_s,
        "controller_anchor_tick": controller_anchor_tick,
        "controller_anchor_time_s": controller_anchor_time_s,
        "target_entry_tick": target_entry_tick,
        "episode_sensor_tick_offset": target_entry_tick,
        "effective_entry_offset_s": phase_entry_time_s(replay_steps),
        "physics_steps": replay_steps,
        "prime_physics_steps": replay_steps,
        "prime_atomic_full12_writes": replay_steps,
        "contact_sensor_reads_after_prime": replay_steps,
        "fsm_clock_steps_during_priming": 0,
        "episode_clock_steps_during_priming": 0,
        "sensor_history_samples_after_reset": replay_steps,
        "source_replay_guard_updates_applied": replay_steps,
    }
    scalar_failures = [
        name for name, expected in expected_scalars.items() if state.get(name) != expected
    ]
    boolean_contract = bool(
        state.get("pre_prime_state_verified") is True
        and state.get("pre_prime_joint_state_verified") is True
        and state.get("post_prime_state_rewrite_performed") is False
        and state.get("contact_and_state_share_solver_tick") is True
        and state.get("logical_target_fallback_used") is False
        and state.get("root_state_writes_confined_before_first_episode_tick") is True
        and state.get("root_velocity_write_api")
        == "write_root_link_velocity_to_sim"
        and isinstance(root, Mapping)
        and root.get("verified") is True
        and root.get("all_values_finite") is True
        and root.get("all_fields_within_production_tolerances") is True
        and root.get("physics_steps_before_readback") == 0
        and root.get("contact_sensor_reads_before_readback") == 0
        and isinstance(actuation, Mapping)
        and actuation.get("schema")
        == "wlr50_clean.phase_snapshot_source_input_live_output.v2"
        and actuation.get("all_replay_invariant_fields_match") is True
        and isinstance(actuation.get("replay_invariant_field_matches"), Mapping)
        and set(actuation["replay_invariant_field_matches"])
        == set(SOURCE_ACK_REPLAY_INVARIANT_FIELDS)
        and all(
            value is True
            for value in actuation["replay_invariant_field_matches"].values()
        )
        and actuation.get("adapter_input_hash_matches") is True
        and actuation.get("live_feedback_adaptive_output_valid") is True
        and actuation.get("historical_feedback_equivalence_claimed") is False
        and actuation.get("logical_target_fallback_used") is False
        and isinstance(mapper, Mapping)
        and mapper.get("schema")
        == "wlr50_clean.phase_snapshot_live_mapper_transition.v2"
        and mapper.get("historical_feedback_equivalence_claimed") is False
        and mapper.get("natural_live_state_continuity_verified") is True
        and mapper.get("ack_cross_bindings_verified") is True
        and mapper.get("feedback_schedule_verified") is True
        and mapper.get("per_tick_mapper_restore_count") == 0
        and mapper.get("reached_naturally_by_single_atomic_apply") is True
        and mapper.get("restored_after_prime") is False
        and attempt.get("phase") == phase
        and attempt.get("scene_lifecycle") == lifecycle
        and type(attempt.get("source_tick")) is int
        and attempt.get("source_tick") == snapshot.source_tick
        and attempt.get("snapshot_file_sha256") == snapshot.file_sha256
        and attempt.get("snapshot_state_sha256") == snapshot.state_sha256
        and Path(str(attempt.get("snapshot_path", ""))).resolve()
        == snapshot.snapshot_path
        and attempt.get("source_replay_steps") == replay_steps
        and "predecessor_verify_tick" in attempt
        and attempt.get("predecessor_verify_tick") == predecessor_verify_tick
        and "predecessor_verify_time_s" in attempt
        and attempt.get("predecessor_verify_time_s") == predecessor_verify_time_s
        and "controller_anchor_tick" in attempt
        and attempt.get("controller_anchor_tick") == controller_anchor_tick
        and "controller_anchor_time_s" in attempt
        and attempt.get("controller_anchor_time_s") == controller_anchor_time_s
        and attempt.get("target_entry_tick") == target_entry_tick
        and attempt.get("episode_sensor_tick_offset") == target_entry_tick
        and attempt.get("effective_entry_offset_s")
        == phase_entry_time_s(replay_steps)
        and all(attempt.get(name) == expected for name, expected in restore_bindings.items())
        and attempt.get("source_control_physics_ticks") == list(control_ticks)
        and attempt.get("source_command_row_canonical_sha256s")
        == [row["source_command_row_canonical_sha256"] for row in source_commands]
        and attempt.get("source_observation_row_canonical_sha256s")
        == [row["source_observation_row_canonical_sha256"] for row in source_commands]
        and attempt.get("source_adapter_input_sha256s")
        == [row["source_adapter_input_sha256"] for row in source_commands]
        and attempt.get("source_drive_target_full12_sha256s")
        == [row["drive_target_full12_sha256"] for row in source_commands]
        and attempt.get("source_actuation_contract_sha256s")
        == [row["actuation_contract_sha256"] for row in source_commands]
        and attempt.get("physics_steps_during_reset") == 180 + replay_steps
        and attempt.get("post_prime_contact_sensor_read_count") == replay_steps
        and attempt.get("extra_physics_priming_steps") == replay_steps
        and attempt.get("fsm_or_episode_advanced_for_probe") is False
        and attempt.get("reset_completed") is True
        and attempt.get("passed") is True
        and attempt.get("failure_classification") is None
        and attempt.get("exception") is None
        and isinstance(observation, Mapping)
        and observation.get("observation_available") is True
        and observation.get("observation_physics_tick") == target_entry_tick
        and observation.get("observation_simulation_time_s")
        == phase_entry_time_s(target_entry_tick)
        and isinstance(clocks, Mapping)
        and clocks.get("authoritative_frame_committed") is True
        and clocks.get("backend_episode_tick") == 0
        and clocks.get("controller_constructed") is True
        and clocks.get("controller_frame_committed") is True
        and clocks.get("controller_frame_physics_tick") == 0
        and clocks.get("controller_frame_state_id") == phase
        and clocks.get("controller_history_length")
        == (0 if restore_bindings else 1)
        and clocks.get("controller_internal_physics_tick") == 1
        and clocks.get("controller_last_simulation_time_s") == 0.0
        and clocks.get("controller_state_id") == phase
        and state.get("current_contact_force_provenance")
        == "current_final_solver_force_only"
        and state.get("classifier_cold_started_before_source_replay") is True
        and state.get("classifier_source_history_restored") is False
        and state.get("classifier_source_state_restored") is False
        and state.get("classifier_history_equivalence_claimed") is False
        and state.get("all_source_adapter_inputs_hash_matched") is True
        and state.get("all_live_output_contracts_verified") is True
        and state.get("all_live_mapper_transitions_verified") is True
        and state.get("live_feedback_adaptive_replay") is True
        and state.get("historical_feedback_equivalence_claimed") is False
        and state.get("initial_mapper_restore_count") == 1
        and state.get("per_tick_mapper_restore_count") == 0
        and state.get("raw_sensor_history_rewarmed_from_prime") is True
        and state.get("contact_backend_reset") is True
        and state.get("contact_backend_reset_after_prime") is False
        and isinstance(sensor, Mapping)
        and sensor.get("verified") is True
        and isinstance(safety, Mapping)
        and safety.get("schema") == "wlr50_clean.phase_effective_entry_safety.v1"
        and safety.get("verified") is True
        and safety.get("all_failure_flags_false") is True
        and isinstance(safety_flags, Mapping)
        and set(safety_flags)
        == {
            "body_collision",
            "combined_physics_abort_guard",
            "fall",
            "hard_joint_limit",
            "nan_inf",
            "physics_explosion",
            "wheel_only_climb",
        }
        and all(type(value) is bool and value is False for value in safety_flags.values())
        and isinstance(effective_proof, Mapping)
        and set(effective_proof) == calibration_proof_fields
        and effective_proof.get("schema") == CALIBRATION_LIVE_PROOF_SCHEMA
        and effective_proof.get("artifact_role") == CALIBRATION_ARTIFACT_ROLE
        and effective_proof.get("verified") is True
        and effective_proof.get("calibration_only") is True
        and effective_proof.get("phase") == phase
        and effective_proof.get("source_tick") == snapshot.source_tick
        and effective_proof.get("physical_anchor_tick")
        == expected_anchor_contract["physical_anchor_tick"]
        and effective_proof.get("physical_anchor_time_s")
        == expected_anchor_contract["physical_anchor_time_s"]
        and effective_proof.get("predecessor_verify_tick")
        == predecessor_verify_tick
        and effective_proof.get("predecessor_verify_time_s")
        == predecessor_verify_time_s
        and effective_proof.get("controller_anchor_tick")
        == controller_anchor_tick
        and effective_proof.get("controller_anchor_time_s")
        == controller_anchor_time_s
        and effective_proof.get("target_entry_tick") == target_entry_tick
        and effective_proof.get("target_entry_time_s")
        == expected_anchor_contract["target_entry_time_s"]
        and effective_proof.get("source_replay_steps") == replay_steps
        and effective_proof.get("physical_to_predecessor_verify_replay_steps")
        == expected_anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ]
        and effective_proof.get("predecessor_verify_to_controller_replay_steps")
        == expected_anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ]
        and effective_proof.get("physical_to_controller_replay_steps")
        == expected_anchor_contract["physical_to_controller_replay_steps"]
        and effective_proof.get("controller_to_target_replay_steps")
        == expected_anchor_contract["controller_to_target_replay_steps"]
        and effective_proof.get("hybrid_physical_controller_anchor")
        == expected_anchor_contract["hybrid_physical_controller_anchor"]
        and effective_proof.get("replay_anchor_contract")
        == expected_anchor_contract
        and effective_proof.get("effective_entry_offset_s")
        == phase_entry_time_s(replay_steps)
        and effective_proof.get("phase_snapshot_bundle_sha256")
        == phase_snapshot_bundle.bundle_sha256
        and all(
            effective_proof.get(name) == expected
            for name, expected in restore_bindings.items()
        )
        and effective_proof.get("source_snapshot_post_prime_diagnostic") == comparison
        and effective_proof.get("failures") == []
        and controller_entry_proof_valid
    )
    if scalar_failures or replay_failures or not boolean_contract:
        raise EffectivePhaseEntryError(
            "calibration reset recipe is invalid: "
            + ", ".join((*scalar_failures, *replay_failures))
        )
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("schema")
        != "wlr50_clean.phase_snapshot_live_comparison.v2"
    ):
        raise EffectivePhaseEntryError("calibration lacks post-prime comparison")
    _validate_controller_entry_guard(
        state.get("entry_guard_contract"),
        phase=phase,
        snapshot_payload=snapshot_payload,
    )
    raw = comparison.get("raw_physx_contacts")
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema", "pairs", "sha256"}
        or raw.get("schema") != "wlr50_clean.phase_snapshot_raw_physx_contact.v1"
    ):
        raise EffectivePhaseEntryError("calibration lacks annotated raw contacts")
    declared = _require_sha256(raw.get("sha256"), "annotated raw-contact SHA")
    unhashed = dict(raw)
    unhashed.pop("sha256", None)
    if _sha256_bytes(_canonical_bytes(unhashed)) != declared:
        raise EffectivePhaseEntryError("annotated raw-contact SHA is invalid")
    _fingerprint(comparison.get("maximum_errors"))
    _component_state(comparison.get("effective_component_state"))
    _calibrated_contacts(comparison)
    return comparison


def _assert_attempts_bit_identical(
    fresh: Mapping[str, Any], reused: Mapping[str, Any]
) -> None:
    fresh_values, fresh_binary = _fingerprint(fresh.get("maximum_errors"))
    reused_values, reused_binary = _fingerprint(reused.get("maximum_errors"))
    if fresh_values != reused_values or fresh_binary != reused_binary:
        raise EffectivePhaseEntryError(
            "fresh/reused post-prime fingerprints are not binary64-identical"
        )
    fresh_state, fresh_state_binary = _component_state(
        fresh.get("effective_component_state")
    )
    reused_state, reused_state_binary = _component_state(
        reused.get("effective_component_state")
    )
    if (
        fresh_state != reused_state
        or fresh_state_binary != reused_state_binary
    ):
        raise EffectivePhaseEntryError(
            "fresh/reused complete component states are not binary64-identical"
        )
    fresh_raw = fresh["raw_physx_contacts"]
    reused_raw = reused["raw_physx_contacts"]
    if fresh_raw.get("sha256") != reused_raw.get("sha256"):
        raise EffectivePhaseEntryError(
            "fresh/reused annotated raw-contact SHA differs"
        )
    def classifications(value: Mapping[str, Any]) -> dict[str, Any]:
        rows = value.get("exact_contacts")
        if not isinstance(rows, Mapping):
            raise EffectivePhaseEntryError("calibration classifier rows are missing")
        return {
            wheel: {
                "body_name": rows[wheel].get("body_name"),
                "class": rows[wheel].get("actual_class"),
                "ground": rows[wheel].get("actual_ground_active"),
                "obstacle": rows[wheel].get("actual_obstacle_active"),
            }
            for wheel in WHEEL_ORDER
        }
    if classifications(fresh) != classifications(reused):
        raise EffectivePhaseEntryError("fresh/reused classifier result differs")


def _assert_replay_attempts_bit_identical(
    fresh: Mapping[str, Any], reused: Mapping[str, Any]
) -> None:
    """Require the fresh/reused reset to execute the same replay recipe."""

    special_restore = bool(
        fresh.get("controller_restore_mode") is not None
        or reused.get("controller_restore_mode") is not None
    )
    source_restore_fields = (
        "controller_restore_mode",
        "source_transition_verified",
        "source_transition_tick",
        "source_transition_time_s",
        "source_transition_row_canonical_sha256",
        "consumed_motion_tick",
        "first_episode_motion_tick",
        "entry_guards_from_source_not_replayed",
    )
    attempt_fields = (
        "source_tick",
        "predecessor_verify_tick",
        "predecessor_verify_time_s",
        "controller_anchor_tick",
        "controller_anchor_time_s",
        "source_replay_steps",
        "target_entry_tick",
        "effective_entry_offset_s",
        "episode_sensor_tick_offset",
        "source_control_physics_ticks",
        "source_command_row_canonical_sha256s",
        "source_observation_row_canonical_sha256s",
        "source_adapter_input_sha256s",
        "source_drive_target_full12_sha256s",
        "source_actuation_contract_sha256s",
        "physics_steps_during_reset",
        "post_prime_contact_sensor_read_count",
        "extra_physics_priming_steps",
        *(source_restore_fields if special_restore else ()),
    )
    state_fields = (
        "source_replay_steps",
        "physical_anchor_tick",
        "physical_anchor_time_s",
        "predecessor_verify_tick",
        "predecessor_verify_time_s",
        "controller_anchor_tick",
        "controller_anchor_time_s",
        "target_entry_tick",
        "target_entry_time_s",
        "physical_to_predecessor_verify_replay_steps",
        "predecessor_verify_to_controller_replay_steps",
        "physical_to_controller_replay_steps",
        "controller_to_target_replay_steps",
        "hybrid_physical_controller_anchor",
        "replay_anchor_contract",
        "source_replay_fsm_contexts",
        "episode_sensor_tick_offset",
        "effective_entry_offset_s",
        "physics_steps",
        "prime_physics_steps",
        "prime_atomic_full12_writes",
        "prime_atomic_writes",
        "source_actuation_matches",
        "source_mapper_post_states",
        "source_adapter_input_sha256s",
        "all_source_adapter_inputs_hash_matched",
        "all_live_output_contracts_verified",
        "all_live_mapper_transitions_verified",
        "live_feedback_adaptive_replay",
        "historical_feedback_equivalence_claimed",
        "initial_mapper_restore_count",
        "per_tick_mapper_restore_count",
        "source_actuation_match",
        "source_mapper_post_state",
        "source_replay_observation_ticks",
        "contact_sensor_reads_after_prime",
        "sensor_history_samples_after_reset",
        "source_replay_guard_updates_applied",
        "source_replay_safety_checks",
        "all_source_replay_steps_safe",
        *(("entry_guard_contract",) if special_restore else ()),
    )
    if any(fresh.get(name) != reused.get(name) for name in attempt_fields):
        raise EffectivePhaseEntryError(
            "fresh/reused source-replay attempt metadata differs"
        )
    fresh_state = fresh.get("snapshot_state_write")
    reused_state = reused.get("snapshot_state_write")
    if not isinstance(fresh_state, Mapping) or not isinstance(reused_state, Mapping):
        raise EffectivePhaseEntryError("fresh/reused replay proof is missing")
    if any(fresh_state.get(name) != reused_state.get(name) for name in state_fields):
        raise EffectivePhaseEntryError("fresh/reused source-replay proof differs")


def _capture_calibration_run(
    run_dir: Path,
    *,
    phase_snapshot_bundle: ValidatedPhaseSnapshotBundle,
    environment_lock_bytes: bytes,
    frozen_ledger_bytes: bytes,
    expected_git_commit: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    directory = _absolute_unredirected(run_dir, label="calibration run directory")
    names = {
        "run_manifest": "run_manifest.json",
        "probe": "phase_snapshot_live_probe.json",
        "runtime_before": "committed_runtime_identity.before.json",
        "runtime_after": "committed_runtime_identity.after.json",
        "frozen_before": "frozen_hashes.before.json",
        "frozen_after": "frozen_hashes.after.json",
    }
    paths = {role: directory / name for role, name in names.items()}
    captured, identities = _capture_paths_once(paths, directory=directory)
    decoded = {
        role: _decode_object(captured[role], label=role, path=paths[role])
        for role in names
    }
    manifest = decoded["run_manifest"]
    identity = manifest.get("identity")
    manifest_commit = (
        _require_git_commit(identity.get("git_commit"), "calibration git commit")
        if isinstance(identity, Mapping)
        else None
    )
    if (
        manifest.get("schema") != "wlr50_clean.ppo_run_manifest.v1"
        or manifest.get("run_kind") != CALIBRATION_RUN_KIND
        or manifest.get("subcommand") != "phase-snapshot-live-probe"
        or manifest.get("immutable_run_directory") is not True
        or not isinstance(manifest.get("run_dir"), str)
        or not manifest["run_dir"]
        or manifest.get("run_id") != directory.name
        or Path(str(manifest.get("run_dir", ""))).resolve() != directory
        or not isinstance(manifest.get("project_root"), str)
        or not manifest["project_root"]
        or Path(str(manifest.get("project_root", ""))).resolve() != PROJECT_ROOT
        or not isinstance(identity, Mapping)
        or manifest_commit is None
        or (expected_git_commit is not None and manifest_commit != expected_git_commit)
        or identity.get("training_stage") != CALIBRATION_TRAINING_STAGE
        or type(identity.get("seed")) is not int
        or identity.get("seed") != 1002
        or type(identity.get("environment_count")) is not int
        or identity.get("environment_count") != 1
        or _SHA256.fullmatch(str(identity.get("config_sha256", ""))) is None
        or manifest.get("exit_code") != 0
        or manifest.get("lifecycle") != "SUCCEEDED"
    ):
        raise EffectivePhaseEntryError("calibration run manifest identity is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EffectivePhaseEntryError("calibration run manifest omits artifacts")
    for role in ("probe", "runtime_before", "runtime_after", "frozen_before", "frozen_after"):
        _validate_bound_artifact(
            artifacts.get(names[role]),
            captured[role],
            expected_path=names[role],
            label=role,
        )
    environment_record = next(
        (
            row
            for row in manifest.get("configs", ())
            if isinstance(row, Mapping) and row.get("path") == "configs/environment_lock.json"
        ),
        None,
    )
    if (
        not isinstance(environment_record, Mapping)
        or environment_record.get("bytes") != len(environment_lock_bytes)
        or environment_record.get("sha256") != _sha256_bytes(environment_lock_bytes)
    ):
        raise EffectivePhaseEntryError("calibration environment-lock binding is invalid")
    config_sha = _require_sha256(identity.get("config_sha256"), "identity config SHA")
    _validate_manifest_config_set(manifest, expected_config_sha256=config_sha)
    runtime_before = decoded["runtime_before"]
    runtime_after = decoded["runtime_after"]
    if runtime_before != runtime_after:
        raise EffectivePhaseEntryError("calibration runtime identity is invalid")
    runtime_commit, runtime_content_sha = _validated_runtime_identity(
        runtime_before, expected_git_commit=manifest_commit
    )
    if runtime_commit != manifest_commit:
        raise EffectivePhaseEntryError("calibration manifest/runtime commit differs")
    frozen_sha = _sha256_bytes(frozen_ledger_bytes)
    for role in ("frozen_before", "frozen_after"):
        audit = decoded[role]
        if (
            audit.get("schema") != "wlr50_clean.frozen_fsm_hash_audit.v1"
            or audit.get("passed") is not True
            or audit.get("mismatches") != []
            or audit.get("frozen_manifest_sha256") != frozen_sha
        ):
            raise EffectivePhaseEntryError(f"calibration {role} audit is invalid")

    probe = decoded["probe"]
    phases = probe.get("phases")
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else None
    attempts = probe.get("attempts")
    if phase in PHASE_IDS:
        (
            expected_snapshot_payload,
            _snapshot_file,
            expected_replay_steps,
            expected_target_entry_tick,
            _expected_control_ticks,
            expected_predecessor_verify_tick,
            expected_predecessor_verify_time_s,
            expected_controller_anchor_tick,
            expected_controller_anchor_time_s,
        ) = _snapshot_replay_contract(phase_snapshot_bundle, str(phase))
        expected_restore_bindings = _source_proven_execute_bindings(
            expected_snapshot_payload, str(phase)
        )
    else:
        expected_replay_steps = None
        expected_target_entry_tick = None
        _expected_control_ticks = ()
        expected_predecessor_verify_tick = None
        expected_predecessor_verify_time_s = None
        expected_controller_anchor_tick = None
        expected_controller_anchor_time_s = None
        expected_restore_bindings = {}
    if (
        probe.get("schema") != PROBE_SCHEMA
        or probe.get("artifact_role") != CALIBRATION_ARTIFACT_ROLE
        or probe.get("calibration_mode") is not True
        or probe.get("status") != "PASSED"
        or probe.get("passed") is not True
        or probe.get("seed") != 1002
        or phase not in PHASE_IDS
        or probe.get("phase_selector_mode") != "single_phase"
        or probe.get("phase_count") != 1
        or probe.get("attempts_per_phase") != 2
        or probe.get("complete") is not True
        or probe.get("expected_attempt_count") != 2
        or probe.get("completed_attempt_count") != 2
        or probe.get("expected_fresh_scene_attempt_count") != 1
        or probe.get("expected_reused_scene_attempt_count") != 1
        or probe.get("fresh_scene_attempt_count") != 1
        or probe.get("reused_scene_attempt_count") != 1
        or probe.get("failure_classification") is not None
        or probe.get("failure_reasons") != []
        or probe.get("phase_effective_entry_contract") is not None
        or probe.get("production_reset_modified") is not True
        or probe.get("production_reset_mode")
        != "validated_source_command_sequence_replay_without_rewind"
        or probe.get("source_replay_policy")
        != "derived_only_from_validated_phase_snapshot"
        or probe.get("source_replay_steps_by_phase")
        != {phase: expected_replay_steps}
        or probe.get("controller_restore_modes_by_phase")
        != {phase: expected_restore_bindings.get("controller_restore_mode")}
        or probe.get("source_transition_hashes_by_phase")
        != {
            phase: expected_restore_bindings.get(
                "source_transition_row_canonical_sha256"
            )
        }
        or probe.get("predecessor_verify_anchors_by_phase")
        != {
            phase: (
                {
                    "predecessor_verify_tick": expected_predecessor_verify_tick,
                    "predecessor_verify_time_s": expected_predecessor_verify_time_s,
                }
                if expected_predecessor_verify_tick is not None
                else None
            )
        }
        or probe.get("controller_anchors_by_phase")
        != {
            phase: (
                {
                    "controller_anchor_tick": expected_controller_anchor_tick,
                    "controller_anchor_time_s": expected_controller_anchor_time_s,
                }
                if expected_controller_anchor_tick is not None
                else None
            )
        }
        or not isinstance(attempts, list)
        or len(attempts) != 2
    ):
        raise EffectivePhaseEntryError("calibration probe completeness is invalid")
    fresh, reused = attempts
    if (
        not isinstance(fresh, Mapping)
        or not isinstance(reused, Mapping)
        or fresh.get("phase") != phase
        or reused.get("phase") != phase
        or fresh.get("attempt_kind") != "primary"
        or fresh.get("attempt_index_for_phase") != 0
        or fresh.get("scene_lifecycle") != "fresh_scene"
        or fresh.get("scene_existed_before") is not False
        or reused.get("attempt_kind") != "reused_repeat"
        or reused.get("attempt_index_for_phase") != 1
        or reused.get("scene_lifecycle") != "reused_scene"
        or reused.get("scene_existed_before") is not True
    ):
        raise EffectivePhaseEntryError("calibration fresh/reused attempt ordering is invalid")
    if probe.get("phase_snapshot_bundle") != phase_snapshot_bundle.as_record():
        raise EffectivePhaseEntryError("calibration snapshot-bundle binding is invalid")
    runtime_reference = probe.get("runtime_identity_before")
    frozen_reference = probe.get("frozen_hashes_before")
    if (
        not isinstance(runtime_reference, Mapping)
        or set(runtime_reference) != {"path", "sha256", "schema"}
        or Path(str(runtime_reference.get("path", ""))).resolve()
        != paths["runtime_before"]
        or runtime_reference.get("schema")
        != "wlr50_clean.committed_runtime_identity.v1"
        or runtime_reference.get("sha256") != _sha256_bytes(captured["runtime_before"])
        or not isinstance(frozen_reference, Mapping)
        or set(frozen_reference) != {"path", "sha256", "schema", "passed"}
        or Path(str(frozen_reference.get("path", ""))).resolve()
        != paths["frozen_before"]
        or frozen_reference.get("schema") != "wlr50_clean.frozen_fsm_hash_audit.v1"
        or frozen_reference.get("passed") is not True
        or frozen_reference.get("sha256") != _sha256_bytes(captured["frozen_before"])
    ):
        raise EffectivePhaseEntryError("calibration probe precheck binding is invalid")
    comparison = _validated_probe_attempt(
        fresh,
        phase=str(phase),
        lifecycle="fresh_scene",
        phase_snapshot_bundle=phase_snapshot_bundle,
    )
    reused_comparison = _validated_probe_attempt(
        reused,
        phase=str(phase),
        lifecycle="reused_scene",
        phase_snapshot_bundle=phase_snapshot_bundle,
    )
    _assert_attempts_bit_identical(comparison, reused_comparison)
    _assert_replay_attempts_bit_identical(fresh, reused)
    fingerprint, fingerprint_binary = _fingerprint(comparison.get("maximum_errors"))
    component_state, component_state_binary = _component_state(
        comparison.get("effective_component_state")
    )
    contacts = _calibrated_contacts(comparison)
    entry_payload: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        "phase": phase,
        "source_tick": int(fresh.get("source_tick")),
        "predecessor_verify_tick": expected_predecessor_verify_tick,
        "predecessor_verify_time_s": expected_predecessor_verify_time_s,
        "controller_anchor_tick": expected_controller_anchor_tick,
        "controller_anchor_time_s": expected_controller_anchor_time_s,
        "target_entry_tick": expected_target_entry_tick,
        "source_replay_steps": expected_replay_steps,
        "effective_entry_offset_s": phase_entry_time_s(expected_replay_steps),
        "calibration_probe_file_sha256": _sha256_bytes(captured["probe"]),
        "post_prime_fingerprint": fingerprint,
        "post_prime_fingerprint_binary64_hex": fingerprint_binary,
        "effective_component_state": component_state,
        "effective_component_state_binary64_hex": component_state_binary,
        "raw_contacts": contacts,
        **expected_restore_bindings,
    }
    entry_payload["entry_sha256"] = _sha256_bytes(_canonical_bytes(entry_payload))
    calibration_record = {
        "schema": CALIBRATION_SCHEMA,
        "phase": phase,
        "run_id": str(manifest.get("run_id")),
        "seed": int(identity.get("seed")),
        "source_git_commit": manifest_commit,
        "files": {
            role: _file_record(paths[role], captured[role], root=PROJECT_ROOT)
            for role in names
        },
        "runtime_content_sha256": runtime_content_sha,
        "identity_config_sha256": config_sha,
        "phase_snapshot_bundle_sha256": phase_snapshot_bundle.bundle_sha256,
        "environment_lock_sha256": _sha256_bytes(environment_lock_bytes),
        "frozen_ledger_sha256": frozen_sha,
        "fresh_attempt_index": 0,
        "reused_attempt_index": 1,
        "captured_filesystem_identity_count": len(identities),
    }
    return str(phase), entry_payload, calibration_record


def build_effective_phase_entry_contract(
    calibration_run_dirs: Sequence[Path | str],
    output_path: Path | str = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
    *,
    snapshot_bundle: ValidatedPhaseSnapshotBundle,
    environment_lock_path: Path | str = DEFAULT_ENVIRONMENT_LOCK_PATH,
    frozen_ledger_path: Path | str = DEFAULT_FROZEN_LEDGER_PATH,
) -> Mapping[str, Any]:
    """Mechanically derive all P02-P13 entries and publish the sidecar last."""

    output = Path(output_path).resolve()
    sidecar = output.with_suffix(".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"effective-entry output already exists: {output}")
    anchors, anchor_identities = _capture_paths_once(
        {
            "environment_lock": Path(environment_lock_path),
            "frozen_ledger": Path(frozen_ledger_path),
        }
    )
    if len(calibration_run_dirs) != len(PHASE_IDS):
        raise EffectivePhaseEntryError("builder requires exactly 12 calibration runs")
    entries: dict[str, Any] = {}
    calibrations: list[dict[str, Any]] = []
    source_git_commit: str | None = None
    for expected_phase, directory in zip(
        PHASE_IDS, calibration_run_dirs, strict=True
    ):
        phase, entry, calibration = _capture_calibration_run(
            Path(directory),
            phase_snapshot_bundle=snapshot_bundle,
            environment_lock_bytes=anchors["environment_lock"],
            frozen_ledger_bytes=anchors["frozen_ledger"],
            expected_git_commit=source_git_commit,
        )
        if phase != expected_phase:
            raise EffectivePhaseEntryError(
                "builder calibration inputs must be explicitly ordered P02-P13"
            )
        calibration_commit = _require_git_commit(
            calibration.get("source_git_commit"), f"{phase} calibration commit"
        )
        if source_git_commit is None:
            source_git_commit = calibration_commit
        elif calibration_commit != source_git_commit:
            raise EffectivePhaseEntryError("all 12 calibrations must share one commit")
        entries[phase] = entry
        calibrations.append(calibration)
    if tuple(entries) != PHASE_IDS:
        raise EffectivePhaseEntryError("calibration runs must cover P02 through P13 exactly")
    if source_git_commit is None:
        raise EffectivePhaseEntryError("calibration commit is unavailable")
    runtime_hashes = {
        _require_sha256(row.get("runtime_content_sha256"), "runtime content SHA")
        for row in calibrations
    }
    config_hashes = {
        _require_sha256(row.get("identity_config_sha256"), "identity config SHA")
        for row in calibrations
    }
    if len(runtime_hashes) != 1 or len(config_hashes) != 1:
        raise EffectivePhaseEntryError(
            "all 12 calibrations must share one runtime and config identity"
        )
    derivation = {
        "schema": DERIVATION_SCHEMA,
        "source_git_commit": source_git_commit,
        "phase_snapshot_bundle": snapshot_bundle.as_record(),
        "environment_lock": _file_record(
            Path(environment_lock_path).resolve(), anchors["environment_lock"]
        ),
        "frozen_ledger": _file_record(
            Path(frozen_ledger_path).resolve(), anchors["frozen_ledger"]
        ),
        "runtime_content_sha256": next(iter(runtime_hashes)),
        "identity_config_sha256": next(iter(config_hashes)),
        "calibration_artifacts": calibrations,
    }
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "derivation": derivation,
        "portability_scope": "same_locked_host_runtime_only",
        "calibration_status": "provisional_pending_independent_fresh_holdout",
        "effective_entry_semantics": (
            "source_snapshot_plus_validated_replay_steps_no_rewind"
        ),
        "physics_dt_s": phase_entry_time_s(1),
        "fingerprint_fields": list(FINGERPRINT_FIELDS),
        "fingerprint_max_ulp_distance": FINGERPRINT_MAX_ULP_DISTANCE,
        "component_state_schema": COMPONENT_STATE_SCHEMA,
        "component_state_max_ulp_distance": COMPONENT_STATE_MAX_ULP_DISTANCE,
        "contact_contract": {
            "force_on_n": CONTACT_FORCE_ON_N,
            "force_off_n": CONTACT_FORCE_OFF_N,
            "source": CONTACT_SOURCE,
            "active_rule": "force_norm_n >= force_on_n",
            "inactive_rule": "force_norm_n < force_off_n",
            "hysteresis_gap_forbidden": True,
            "classifier_must_equal_raw": True,
            "double_active_forbidden": True,
        },
        "phase_count": len(PHASE_IDS),
        "phases": {phase: entries[phase] for phase in PHASE_IDS},
    }
    contract["contract_sha256"] = _sha256_bytes(_canonical_bytes(contract))
    contract_bytes = _json_lf_bytes(contract)
    for identity in anchor_identities:
        if not _same_path_identity(
            identity,
            _path_identity(
            Path(identity[0]),
            label=f"captured trust anchor {identity[0]}",
            directory=identity[1] == "directory",
            ),
        ):
            raise EffectivePhaseEntryError(
                f"captured trust anchor changed: {identity[0]}"
            )
    _validate_contract_payload(
        contract,
        expected_snapshot_bundle=snapshot_bundle,
        environment_lock_path=_absolute_unredirected(
            environment_lock_path, label="environment lock"
        ),
        environment_lock_bytes=anchors["environment_lock"],
        frozen_ledger_path=_absolute_unredirected(
            frozen_ledger_path, label="frozen ledger"
        ),
        frozen_ledger_bytes=anchors["frozen_ledger"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(contract_bytes)
    with sidecar.open("xb") as handle:
        handle.write(
            f"{_sha256_bytes(contract_bytes)}  {output.name}\n".encode("ascii")
        )
    capture_validated_effective_phase_entry_contract(
        output,
        expected_snapshot_bundle=snapshot_bundle,
        environment_lock_path=environment_lock_path,
        frozen_ledger_path=frozen_ledger_path,
    )
    return contract


def _validate_contract_payload(
    payload: Mapping[str, Any],
    *,
    expected_snapshot_bundle: ValidatedPhaseSnapshotBundle,
    environment_lock_path: Path,
    environment_lock_bytes: bytes,
    frozen_ledger_path: Path,
    frozen_ledger_bytes: bytes,
) -> tuple[str, tuple[tuple[str, Mapping[str, Any]], ...]]:
    expected_top_fields = {
        "schema",
        "derivation",
        "portability_scope",
        "calibration_status",
        "effective_entry_semantics",
        "physics_dt_s",
        "fingerprint_fields",
        "fingerprint_max_ulp_distance",
        "component_state_schema",
        "component_state_max_ulp_distance",
        "contact_contract",
        "phase_count",
        "phases",
        "contract_sha256",
    }
    if set(payload) != expected_top_fields:
        raise EffectivePhaseEntryError(
            "effective-entry top-level fields are incomplete or unexpected"
        )
    contract = dict(payload)
    contract_hash = _require_sha256(contract.pop("contract_sha256", None), "contract_sha256")
    if _sha256_bytes(_canonical_bytes(contract)) != contract_hash:
        raise EffectivePhaseEntryError("effective-entry semantic hash mismatch")
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise EffectivePhaseEntryError("invalid effective-entry contract schema")
    if (
        payload.get("portability_scope") != "same_locked_host_runtime_only"
        or payload.get("calibration_status")
        != "provisional_pending_independent_fresh_holdout"
        or payload.get("effective_entry_semantics")
        != "source_snapshot_plus_validated_replay_steps_no_rewind"
        or payload.get("physics_dt_s") != phase_entry_time_s(1)
        or payload.get("fingerprint_fields") != list(FINGERPRINT_FIELDS)
        or payload.get("fingerprint_max_ulp_distance") != 1
        or payload.get("component_state_schema") != COMPONENT_STATE_SCHEMA
        or payload.get("component_state_max_ulp_distance")
        != COMPONENT_STATE_MAX_ULP_DISTANCE
        or payload.get("phase_count") != len(PHASE_IDS)
    ):
        raise EffectivePhaseEntryError("effective-entry top-level contract is invalid")
    contact = payload.get("contact_contract")
    if not isinstance(contact, Mapping) or contact != {
        "force_on_n": CONTACT_FORCE_ON_N,
        "force_off_n": CONTACT_FORCE_OFF_N,
        "source": CONTACT_SOURCE,
        "active_rule": "force_norm_n >= force_on_n",
        "inactive_rule": "force_norm_n < force_off_n",
        "hysteresis_gap_forbidden": True,
        "classifier_must_equal_raw": True,
        "double_active_forbidden": True,
    }:
        raise EffectivePhaseEntryError("effective-entry contact contract is invalid")
    derivation = payload.get("derivation")
    derivation_commit = (
        _require_git_commit(
            derivation.get("source_git_commit"), "derivation source git commit"
        )
        if isinstance(derivation, Mapping)
        else None
    )
    if (
        not isinstance(derivation, Mapping)
        or set(derivation)
        != {
            "schema",
            "source_git_commit",
            "phase_snapshot_bundle",
            "environment_lock",
            "frozen_ledger",
            "runtime_content_sha256",
            "identity_config_sha256",
            "calibration_artifacts",
        }
        or derivation.get("schema") != DERIVATION_SCHEMA
        or derivation_commit is None
        or derivation.get("phase_snapshot_bundle")
        != expected_snapshot_bundle.as_record()
    ):
        raise EffectivePhaseEntryError("effective-entry derivation binding is invalid")
    _require_sha256(derivation.get("runtime_content_sha256"), "runtime content SHA")
    _require_sha256(derivation.get("identity_config_sha256"), "identity config SHA")
    for role, path, data in (
        ("environment_lock", environment_lock_path, environment_lock_bytes),
        ("frozen_ledger", frozen_ledger_path, frozen_ledger_bytes),
    ):
        row = derivation.get(role)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or Path(str(row.get("path"))).resolve() != path
            or row.get("bytes") != len(data)
            or row.get("sha256") != _sha256_bytes(data)
        ):
            raise EffectivePhaseEntryError(f"effective-entry {role} binding is invalid")
    calibrations = derivation.get("calibration_artifacts")
    if (
        not isinstance(calibrations, list)
        or len(calibrations) != len(PHASE_IDS)
        or tuple(row.get("phase") for row in calibrations if isinstance(row, Mapping))
        != PHASE_IDS
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "schema",
                "phase",
                "run_id",
                "seed",
                "source_git_commit",
                "files",
                "runtime_content_sha256",
                "identity_config_sha256",
                "phase_snapshot_bundle_sha256",
                "environment_lock_sha256",
                "frozen_ledger_sha256",
                "fresh_attempt_index",
                "reused_attempt_index",
                "captured_filesystem_identity_count",
            }
            or row.get("schema") != CALIBRATION_SCHEMA
            or row.get("source_git_commit") != derivation_commit
            or row.get("seed") != 1002
            or row.get("phase_snapshot_bundle_sha256")
            != expected_snapshot_bundle.bundle_sha256
            or row.get("environment_lock_sha256") != _sha256_bytes(environment_lock_bytes)
            or row.get("frozen_ledger_sha256") != _sha256_bytes(frozen_ledger_bytes)
            or row.get("runtime_content_sha256")
            != derivation.get("runtime_content_sha256")
            or row.get("identity_config_sha256")
            != derivation.get("identity_config_sha256")
            or row.get("fresh_attempt_index") != 0
            or row.get("reused_attempt_index") != 1
            for row in calibrations
        )
    ):
        raise EffectivePhaseEntryError("effective-entry calibration provenance is invalid")
    calibration_probe_hashes: dict[str, str] = {}
    recomputed_entries: dict[str, Mapping[str, Any]] = {}
    for phase, row in zip(PHASE_IDS, calibrations, strict=True):
        if row.get("phase") != phase or not isinstance(row.get("run_id"), str):
            raise EffectivePhaseEntryError(
                f"effective-entry calibration identity mismatch for {phase}"
            )
        files = row.get("files")
        if not isinstance(files, Mapping) or set(files) != {
            "run_manifest",
            "probe",
            "runtime_before",
            "runtime_after",
            "frozen_before",
            "frozen_after",
        }:
            raise EffectivePhaseEntryError(
                f"effective-entry calibration files are incomplete for {phase}"
            )
        for role, file_row in files.items():
            if (
                not isinstance(file_row, Mapping)
                or set(file_row) != {"path", "bytes", "sha256"}
                or type(file_row.get("bytes")) is not int
                or file_row["bytes"] < 0
                or not isinstance(file_row.get("path"), str)
            ):
                raise EffectivePhaseEntryError(
                    f"effective-entry calibration file binding is invalid: {phase}.{role}"
                )
            _require_sha256(file_row.get("sha256"), f"{phase}.{role}.sha256")
        calibration_probe_hashes[phase] = str(files["probe"]["sha256"])
        manifest_path = _record_path(
            files["run_manifest"].get("path"),
            label=f"{phase} calibration run manifest",
        )
        if manifest_path.name != "run_manifest.json":
            raise EffectivePhaseEntryError(
                f"effective-entry calibration manifest path is invalid for {phase}"
            )
        run_dir = manifest_path.parent
        for role, filename in {
            "run_manifest": "run_manifest.json",
            "probe": "phase_snapshot_live_probe.json",
            "runtime_before": "committed_runtime_identity.before.json",
            "runtime_after": "committed_runtime_identity.after.json",
            "frozen_before": "frozen_hashes.before.json",
            "frozen_after": "frozen_hashes.after.json",
        }.items():
            if _record_path(
                files[role].get("path"), label=f"{phase}.{role} calibration file"
            ) != run_dir / filename:
                raise EffectivePhaseEntryError(
                    f"effective-entry calibration file escaped its run: {phase}.{role}"
                )
        captured_phase, captured_entry, captured_record = _capture_calibration_run(
            run_dir,
            phase_snapshot_bundle=expected_snapshot_bundle,
            environment_lock_bytes=environment_lock_bytes,
            frozen_ledger_bytes=frozen_ledger_bytes,
            expected_git_commit=derivation_commit,
        )
        if captured_phase != phase or captured_record != dict(row):
            raise EffectivePhaseEntryError(
                f"effective-entry calibration evidence changed for {phase}"
            )
        recomputed_entries[phase] = captured_entry
    phases = payload.get("phases")
    if not isinstance(phases, Mapping) or tuple(phases) != PHASE_IDS:
        raise EffectivePhaseEntryError("effective-entry phases must be ordered P02-P13")
    validated: list[tuple[str, Mapping[str, Any]]] = []
    for phase in PHASE_IDS:
        row = phases[phase]
        if not isinstance(row, Mapping):
            raise EffectivePhaseEntryError(f"effective entry {phase} is invalid")
        (
            expected_snapshot_payload,
            snapshot_file,
            expected_replay_steps,
            expected_target_entry_tick,
            _control_ticks,
            expected_predecessor_verify_tick,
            expected_predecessor_verify_time_s,
            expected_controller_anchor_tick,
            expected_controller_anchor_time_s,
        ) = _snapshot_replay_contract(expected_snapshot_bundle, phase)
        expected_restore_bindings = _source_proven_execute_bindings(
            expected_snapshot_payload, phase
        )
        expected_entry_fields = {
            "schema",
            "phase",
            "source_tick",
            "predecessor_verify_tick",
            "predecessor_verify_time_s",
            "controller_anchor_tick",
            "controller_anchor_time_s",
            "target_entry_tick",
            "source_replay_steps",
            "effective_entry_offset_s",
            "calibration_probe_file_sha256",
            "post_prime_fingerprint",
            "post_prime_fingerprint_binary64_hex",
            "effective_component_state",
            "effective_component_state_binary64_hex",
            "raw_contacts",
            "entry_sha256",
            *expected_restore_bindings,
        }
        if set(row) != expected_entry_fields:
            raise EffectivePhaseEntryError(
                f"effective entry fields are incomplete or unexpected for {phase}"
            )
        unhashed = dict(row)
        entry_hash = _require_sha256(unhashed.pop("entry_sha256", None), f"{phase}.entry_sha256")
        if _sha256_bytes(_canonical_bytes(unhashed)) != entry_hash:
            raise EffectivePhaseEntryError(f"effective entry hash mismatch for {phase}")
        if (
            row.get("schema") != ENTRY_SCHEMA
            or row.get("phase") != phase
            or type(row.get("source_tick")) is not int
            or row.get("source_tick") != snapshot_file.source_tick
            or "predecessor_verify_tick" not in row
            or row.get("predecessor_verify_tick")
            != expected_predecessor_verify_tick
            or "predecessor_verify_time_s" not in row
            or row.get("predecessor_verify_time_s")
            != expected_predecessor_verify_time_s
            or row.get("controller_anchor_tick")
            != expected_controller_anchor_tick
            or row.get("controller_anchor_time_s")
            != expected_controller_anchor_time_s
            or row.get("target_entry_tick") != expected_target_entry_tick
            or row.get("source_replay_steps") != expected_replay_steps
            or row.get("effective_entry_offset_s")
            != phase_entry_time_s(expected_replay_steps)
            or row.get("calibration_probe_file_sha256")
            != calibration_probe_hashes[phase]
            or any(
                row.get(name) != expected
                for name, expected in expected_restore_bindings.items()
            )
        ):
            raise EffectivePhaseEntryError(f"effective entry identity is invalid for {phase}")
        values, binary = _fingerprint(row.get("post_prime_fingerprint"))
        if row.get("post_prime_fingerprint_binary64_hex") != binary:
            raise EffectivePhaseEntryError(f"effective fingerprint binary64 mismatch for {phase}")
        for field in FINGERPRINT_FIELDS:
            if _binary64_from_hex(binary[field], f"{phase}.{field}") != values[field]:
                raise EffectivePhaseEntryError(f"effective fingerprint value mismatch for {phase}")
        component_state, component_binary = _component_state(
            row.get("effective_component_state")
        )
        if row.get("effective_component_state_binary64_hex") != component_binary:
            raise EffectivePhaseEntryError(
                f"effective component-state binary64 mismatch for {phase}"
            )
        for label, number in _component_state_scalar_items(component_state):
            parts = label.replace("]", "").replace("[", ".").split(".")
            encoded: Any = component_binary
            for part in parts:
                encoded = encoded[int(part)] if part.isdigit() else encoded[part]
            if _binary64_from_hex(encoded, f"{phase}.{label}") != number:
                raise EffectivePhaseEntryError(
                    f"effective component-state value mismatch for {phase}.{label}"
                )
        contacts = row.get("raw_contacts")
        if not isinstance(contacts, Mapping) or set(contacts) != {*WHEEL_ORDER, "signature_sha256"}:
            raise EffectivePhaseEntryError(f"effective contacts are incomplete for {phase}")
        signature: dict[str, Any] = {}
        for wheel in WHEEL_ORDER:
            wheel_row = contacts[wheel]
            if (
                not isinstance(wheel_row, Mapping)
                or set(wheel_row)
                != {"body_name", "classification", "ground", "obstacle"}
                or wheel_row.get("body_name") != wheel.replace("_ankle", "_wheel")
            ):
                raise EffectivePhaseEntryError(f"effective contact {phase}.{wheel} is invalid")
            ground = wheel_row.get("ground")
            obstacle = wheel_row.get("obstacle")
            if not isinstance(ground, Mapping) or not isinstance(obstacle, Mapping):
                raise EffectivePhaseEntryError(f"effective contact pairs missing for {phase}.{wheel}")
            active = {}
            for pair_name, pair in (("ground", ground), ("obstacle", obstacle)):
                if set(pair) != {
                    "active",
                    "pair_verified",
                    "source",
                    "force_w_n",
                    "force_w_n_binary64_hex",
                    "force_norm_n",
                    "force_norm_n_binary64_hex",
                }:
                    raise EffectivePhaseEntryError(
                        f"effective contact fields are invalid for {phase}.{wheel}.{pair_name}"
                    )
                force = pair.get("force_w_n")
                binary_force = pair.get("force_w_n_binary64_hex")
                if (
                    not isinstance(force, list)
                    or len(force) != 3
                    or not isinstance(binary_force, list)
                    or len(binary_force) != 3
                    or type(pair.get("active")) is not bool
                ):
                    raise EffectivePhaseEntryError("effective contact force is incomplete")
                decoded = [
                    _binary64_from_hex(value, f"{phase}.{wheel}.{pair_name}")
                    for value in binary_force
                ]
                if decoded != [float(value) for value in force]:
                    raise EffectivePhaseEntryError("effective contact binary64 mismatch")
                norm = math.sqrt(sum(value * value for value in decoded))
                declared_norm = float(pair.get("force_norm_n"))
                if (
                    not math.isfinite(declared_norm)
                    or _binary64_from_hex(
                        pair.get("force_norm_n_binary64_hex"),
                        f"{phase}.{wheel}.{pair_name}.norm",
                    )
                    != declared_norm
                    or binary64_ulp_distance(norm, declared_norm) > 1
                    or pair.get("pair_verified") is not True
                    or pair.get("source") != CONTACT_SOURCE
                ):
                    raise EffectivePhaseEntryError("effective contact provenance is invalid")
                expected_active = bool(pair.get("active"))
                threshold_active = norm >= CONTACT_FORCE_ON_N
                threshold_inactive = norm < CONTACT_FORCE_OFF_N
                if not (threshold_active or threshold_inactive) or expected_active != threshold_active:
                    raise EffectivePhaseEntryError("effective contact threshold is invalid")
                active[pair_name] = expected_active
            classification = _raw_class(active["ground"], active["obstacle"])
            if wheel_row.get("classification") != classification:
                raise EffectivePhaseEntryError("effective raw classification is invalid")
            signature[wheel] = {
                "body_name": wheel_row.get("body_name"),
                "classification": classification,
                "ground_active": active["ground"],
                "obstacle_active": active["obstacle"],
                "pair_verified": {name: True for name in PAIR_ORDER},
                "source": {name: CONTACT_SOURCE for name in PAIR_ORDER},
            }
        expected_signature = _require_sha256(
            contacts.get("signature_sha256"), f"{phase}.raw_contact_signature"
        )
        if _sha256_bytes(_canonical_bytes(signature)) != expected_signature:
            raise EffectivePhaseEntryError(f"effective contact signature mismatch for {phase}")
        if recomputed_entries.get(phase) != dict(row):
            raise EffectivePhaseEntryError(
                f"effective entry was not derived from calibration evidence for {phase}"
            )
        validated.append((phase, dict(row)))
    return contract_hash, tuple(validated)


def capture_validated_effective_phase_entry_contract(
    contract_path: Path | str = DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
    *,
    expected_snapshot_bundle: ValidatedPhaseSnapshotBundle,
    environment_lock_path: Path | str = DEFAULT_ENVIRONMENT_LOCK_PATH,
    frozen_ledger_path: Path | str = DEFAULT_FROZEN_LEDGER_PATH,
) -> ValidatedEffectivePhaseEntryContract:
    """Single-read and pin config, sidecar, environment lock, and frozen ledger."""

    contract = _absolute_unredirected(contract_path, label="effective-entry contract")
    sidecar = _absolute_unredirected(
        contract.with_suffix(".sha256"), label="effective-entry sidecar"
    )
    environment = _absolute_unredirected(environment_lock_path, label="environment lock")
    frozen = _absolute_unredirected(frozen_ledger_path, label="frozen ledger")
    captured, identities = _capture_paths_once(
        {
            "contract": contract,
            "sidecar": sidecar,
            "environment_lock": environment,
            "frozen_ledger": frozen,
        }
    )
    file_hash = _sha256_bytes(captured["contract"])
    expected_sidecar = f"{file_hash}  {contract.name}\n".encode("ascii")
    if captured["sidecar"] != expected_sidecar:
        raise EffectivePhaseEntryError("effective-entry checksum sidecar mismatch")
    payload = _decode_object(captured["contract"], label="effective-entry contract", path=contract)
    contract_hash, entries = _validate_contract_payload(
        payload,
        expected_snapshot_bundle=expected_snapshot_bundle,
        environment_lock_path=environment,
        environment_lock_bytes=captured["environment_lock"],
        frozen_ledger_path=frozen,
        frozen_ledger_bytes=captured["frozen_ledger"],
    )
    return ValidatedEffectivePhaseEntryContract(
        contract_path=contract,
        sidecar_path=sidecar,
        environment_lock_path=environment,
        frozen_ledger_path=frozen,
        contract_bytes=captured["contract"],
        sidecar_bytes=captured["sidecar"],
        environment_lock_bytes=captured["environment_lock"],
        frozen_ledger_bytes=captured["frozen_ledger"],
        file_sha256=file_hash,
        sidecar_file_sha256=_sha256_bytes(captured["sidecar"]),
        contract_sha256=contract_hash,
        phase_snapshot_bundle_sha256=expected_snapshot_bundle.bundle_sha256,
        entries=entries,
        filesystem_identity=identities,
    )


def assert_effective_phase_entry_contract_unchanged(
    expected: ValidatedEffectivePhaseEntryContract,
    *,
    expected_snapshot_bundle: ValidatedPhaseSnapshotBundle,
) -> None:
    current = capture_validated_effective_phase_entry_contract(
        expected.contract_path,
        expected_snapshot_bundle=expected_snapshot_bundle,
        environment_lock_path=expected.environment_lock_path,
        frozen_ledger_path=expected.frozen_ledger_path,
    )
    if (
        current.as_record() != expected.as_record()
        or not _same_filesystem_identity(
            expected.filesystem_identity, current.filesystem_identity
        )
    ):
        raise EffectivePhaseEntryError("effective-entry pinned contract changed")


def validate_effective_phase_entry_comparison(
    contract: ValidatedEffectivePhaseEntryContract,
    phase: str,
    comparison: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Hard-gate one post-prime sample against its calibrated effective entry."""

    entry = contract.entry(phase)
    actual, _ = _fingerprint(comparison.get("maximum_errors"))
    reference = entry["post_prime_fingerprint"]
    fingerprint_rows: dict[str, Any] = {}
    for field in FINGERPRINT_FIELDS:
        distance = binary64_ulp_distance(actual[field], float(reference[field]))
        fingerprint_rows[field] = {
            "reference": float(reference[field]),
            "reference_binary64_hex": entry["post_prime_fingerprint_binary64_hex"][field],
            "actual": actual[field],
            "actual_binary64_hex": _binary64_hex(actual[field]),
            "ulp_distance": distance,
            "passed": distance <= FINGERPRINT_MAX_ULP_DISTANCE,
        }
    failures = [
        f"{field} fingerprint is {row['ulp_distance']} ULP from calibration"
        for field, row in fingerprint_rows.items()
        if row["passed"] is not True
    ]
    actual_component, actual_component_binary = _component_state(
        comparison.get("effective_component_state")
    )
    reference_component, _reference_component_binary = _component_state(
        entry.get("effective_component_state")
    )
    actual_component_items = dict(_component_state_scalar_items(actual_component))
    reference_component_items = dict(
        _component_state_scalar_items(reference_component)
    )
    if set(actual_component_items) != set(reference_component_items):
        raise EffectivePhaseEntryError(
            "live/calibrated component-state layouts differ"
        )
    component_ulp_distance: dict[str, int] = {}
    for label, actual_number in actual_component_items.items():
        distance = _component_binary64_ulp_distance(
            actual_number, reference_component_items[label]
        )
        component_ulp_distance[label] = distance
        if distance > COMPONENT_STATE_MAX_ULP_DISTANCE:
            failures.append(
                f"{label} component state is {distance} ULP from calibration"
            )
    component_state_max_ulp_distance = max(component_ulp_distance.values())
    raw = comparison.get("raw_physx_contacts")
    pairs = raw.get("pairs") if isinstance(raw, Mapping) else None
    classified = comparison.get("exact_contacts")
    contacts_proof: dict[str, Any] = {}
    signature: dict[str, Any] = {}
    if not isinstance(pairs, Mapping) or not isinstance(classified, Mapping):
        failures.append("live comparison omits raw/classified contacts")
    else:
        for wheel in WHEEL_ORDER:
            reference_wheel = entry["raw_contacts"][wheel]
            live_pairs = pairs.get(wheel)
            class_row = classified.get(wheel)
            if not isinstance(live_pairs, Mapping) or not isinstance(class_row, Mapping):
                failures.append(f"{wheel} live contact is missing")
                continue
            live_active: dict[str, bool] = {}
            pair_proof: dict[str, Any] = {}
            for pair_name in PAIR_ORDER:
                pair = live_pairs.get(pair_name)
                if not isinstance(pair, Mapping):
                    failures.append(f"{wheel}.{pair_name} live pair is missing")
                    continue
                try:
                    force = tuple(float(value) for value in pair.get("force_w_n", ()))
                except (TypeError, ValueError):
                    force = ()
                finite = len(force) == 3 and all(math.isfinite(value) for value in force)
                norm = math.sqrt(sum(value * value for value in force)) if finite else math.nan
                active = bool(finite and norm >= CONTACT_FORCE_ON_N)
                inactive = bool(finite and norm < CONTACT_FORCE_OFF_N)
                verified = pair.get("pair_verified") is True
                source_ok = pair.get("source") == CONTACT_SOURCE
                threshold_ok = active or inactive
                expected_active = bool(reference_wheel[pair_name]["active"])
                matches_reference = threshold_ok and active == expected_active
                passed = finite and verified and source_ok and threshold_ok and matches_reference
                pair_proof[pair_name] = {
                    "force_w_n": list(force) if finite else None,
                    "force_w_n_binary64_hex": (
                        [_binary64_hex(value) for value in force] if finite else None
                    ),
                    "force_norm_n": norm if finite else None,
                    "raw_active": active if threshold_ok else None,
                    "reference_active": expected_active,
                    "pair_verified": verified,
                    "source": pair.get("source"),
                    "passed": passed,
                }
                if not passed:
                    failures.append(f"{wheel}.{pair_name} raw contact contract failed")
                live_active[pair_name] = active
            if set(live_active) != set(PAIR_ORDER):
                continue
            try:
                classification = _raw_class(
                    live_active["ground"], live_active["obstacle"]
                )
            except EffectivePhaseEntryError:
                classification = "GROUND_AND_OBSTACLE"
                failures.append(f"{wheel} has double-active raw contact")
            classifier_class = str(class_row.get("actual_class"))
            classifier_ground = class_row.get("actual_ground_active")
            classifier_obstacle = class_row.get("actual_obstacle_active")
            classifier_matches_raw = bool(
                classifier_class == classification
                and classifier_ground is live_active["ground"]
                and classifier_obstacle is live_active["obstacle"]
            )
            reference_matches = bool(
                classification == reference_wheel["classification"]
                and live_active["ground"] is reference_wheel["ground"]["active"]
                and live_active["obstacle"] is reference_wheel["obstacle"]["active"]
                and class_row.get("body_name") == reference_wheel["body_name"]
            )
            if not classifier_matches_raw:
                failures.append(f"{wheel} classifier differs from current raw force")
            if not reference_matches:
                failures.append(f"{wheel} raw contact signature differs from calibration")
            contacts_proof[wheel] = {
                "body_name": class_row.get("body_name"),
                "raw_classification": classification,
                "classifier_classification": classifier_class,
                "classifier_matches_raw": classifier_matches_raw,
                "reference_matches": reference_matches,
                **pair_proof,
            }
            signature[wheel] = {
                "body_name": class_row.get("body_name"),
                "classification": classification,
                "ground_active": live_active["ground"],
                "obstacle_active": live_active["obstacle"],
                "pair_verified": {
                    name: pair_proof[name]["pair_verified"] for name in PAIR_ORDER
                },
                "source": {name: pair_proof[name]["source"] for name in PAIR_ORDER},
            }
    signature_sha = _sha256_bytes(_canonical_bytes(signature)) if len(signature) == 4 else None
    expected_signature_sha = entry["raw_contacts"]["signature_sha256"]
    if signature_sha != expected_signature_sha:
        failures.append("raw contact signature SHA differs from calibration")
    proof = {
        "schema": "wlr50_clean.ppo_phase_effective_entry_live_proof.v2",
        "phase": phase,
        "effective_entry_semantics": (
            "source_snapshot_plus_validated_replay_steps_no_rewind"
        ),
        "source_tick": entry.get("source_tick"),
        "predecessor_verify_tick": entry.get("predecessor_verify_tick"),
        "predecessor_verify_time_s": entry.get("predecessor_verify_time_s"),
        "controller_anchor_tick": entry.get("controller_anchor_tick"),
        "controller_anchor_time_s": entry.get("controller_anchor_time_s"),
        "target_entry_tick": entry.get("target_entry_tick"),
        "source_replay_steps": entry.get("source_replay_steps"),
        "effective_entry_offset_s": entry.get("effective_entry_offset_s"),
        "contract_sha256": contract.contract_sha256,
        "entry_sha256": entry["entry_sha256"],
        "fingerprint_max_ulp_distance": FINGERPRINT_MAX_ULP_DISTANCE,
        "fingerprint": fingerprint_rows,
        "component_state_allowed_max_ulp_distance": (
            COMPONENT_STATE_MAX_ULP_DISTANCE
        ),
        "component_state_max_ulp_distance": component_state_max_ulp_distance,
        "component_state_ulp_distance": component_ulp_distance,
        "component_state": actual_component,
        "component_state_binary64_hex": actual_component_binary,
        "component_state_sha256": actual_component["sha256"],
        "expected_component_state_sha256": reference_component["sha256"],
        "raw_contacts": contacts_proof,
        "raw_contact_signature_sha256": signature_sha,
        "expected_raw_contact_signature_sha256": expected_signature_sha,
        "failures": list(dict.fromkeys(failures)),
        "verified": not failures,
    }
    if entry.get("controller_restore_mode") is not None:
        for name in (
            "controller_restore_mode",
            "source_transition_verified",
            "source_transition_tick",
            "source_transition_time_s",
            "source_transition_row_canonical_sha256",
            "consumed_motion_tick",
            "first_episode_motion_tick",
            "entry_guards_from_source_not_replayed",
        ):
            proof[name] = entry.get(name)
    if failures:
        raise EffectivePhaseEntryError(
            "effective phase entry does not match calibrated contract: "
            + ", ".join(proof["failures"])
        )
    return proof


__all__ = [
    "COMPONENT_STATE_MAX_ULP_DISTANCE",
    "COMPONENT_STATE_SCHEMA",
    "CONTACT_FORCE_OFF_N",
    "CONTACT_FORCE_ON_N",
    "CONTACT_SOURCE",
    "CONTRACT_SCHEMA",
    "DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH",
    "EffectivePhaseEntryError",
    "FINGERPRINT_FIELDS",
    "PHASE_IDS",
    "SOURCE_PROVEN_EXECUTE_RESTORE_MODE",
    "SOURCE_PROVEN_EXECUTE_RESTORE_SCHEMA",
    "ValidatedEffectivePhaseEntryContract",
    "assert_effective_phase_entry_contract_unchanged",
    "binary64_ulp_distance",
    "build_effective_phase_entry_contract",
    "capture_validated_effective_phase_entry_contract",
    "phase_entry_time_s",
    "validate_effective_phase_entry_comparison",
]
