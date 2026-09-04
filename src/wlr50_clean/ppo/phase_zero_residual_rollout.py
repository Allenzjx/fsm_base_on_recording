"""Bounded live zero-residual rollouts from every phase-entry reset.

This module is deliberately independent from checkpoint promotion and the
effective-entry calibration builder.  It exercises the production
``ResidualEpisodeEnv`` path after the external effective-entry holdout has
passed, and emits one compact managed-run artifact that can be required before
phase-curriculum training starts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_projection import full12_bytes


ARTIFACT_FILENAME = "phase_zero_residual_rollout.json"
ARTIFACT_SCHEMA = "wlr50_clean.phase_zero_residual_rollout.v1"
TRAINING_EVIDENCE_SCHEMA = (
    "wlr50_clean.phase_zero_residual_rollout_training_evidence.v1"
)
# ``artifacts.reserve_run`` canonicalizes the wrapper's underscore-separated
# request before it creates the managed directory and lifecycle manifests.
RUN_KIND = "phase-zero-residual-rollout"
TRAINING_STAGE = "phase-zero-residual-rollout"
SUBCOMMAND = "phase-zero-residual-rollout"
PHASE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
MAX_DECISIONS_PER_PHASE = 64
PHYSICS_TICKS_PER_DECISION = 8
PHYSICS_HZ = 120.0
DECISION_HZ = 15.0
ZERO_FULL12 = (0.0,) * 12
_ZERO_BYTES = full12_bytes(ZERO_FULL12)
_EMPTY_VALUES = (None, "", (), [], {})
_HARD_TERMINATION_FIELDS = (
    "success",
    "body_collision",
    "wheel_only_climb",
    "fall",
    "nan_inf",
    "hard_joint_limit",
    "physics_explosion",
    "timeout",
)
_CHECK_NAMES = (
    "holdout_acceptance_prevalidated",
    "snapshot_and_effective_entry_contracts_bound",
    "one_backend_instance",
    "all_p01_p13_resets_executed",
    "all_windows_ended_at_next_boundary_or_max_64_decisions",
    "exact_120hz_15hz_cadence",
    "zero_residual_bitwise_nominal_on_every_physics_tick",
    "no_post_tick0_root_state_writes",
    "no_terminal_safety_or_controller_blocker",
    "no_runtime_recording_access",
    "phase_boundary_order_preserved",
)


class PhaseZeroResidualRolloutError(RuntimeError):
    """The bounded physical rollout or its evidence is invalid."""


class _StopUnsafeRollout(PhaseZeroResidualRolloutError):
    """Stop taking physical actions after the first audited violation."""


def _utc_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _is_empty(value: Any) -> bool:
    return value in _EMPTY_VALUES


def _strict_zero_counter(info: Mapping[str, Any], name: str) -> bool:
    value = info.get(name)
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _hard_termination_flags(frame: Any) -> dict[str, bool]:
    signals = frame.termination_signals
    return {
        name: bool(_member(signals, name, False))
        for name in _HARD_TERMINATION_FIELDS
    }


def _safety_is_neutral(frame: Any) -> bool:
    safety = frame.safety_projection
    return bool(
        _member(safety, "residual_enabled", False) is True
        and tuple(_member(safety, "channel_mask_full12", ())) == (1,) * 12
        and _member(safety, "force_wheels_zero", True) is False
        and _member(safety, "body_collision_detected", True) is False
        and _member(safety, "wheel_only_climb_detected", True) is False
        and _member(safety, "override_full12") is None
        and _member(safety, "reason") is None
    )


def _controller_is_unblocked(
    frame: Any, *, allowed_lifecycles: tuple[str, ...]
) -> bool:
    info = frame.info
    if info.get("controller_lifecycle") not in allowed_lifecycles:
        return False
    if not _is_empty(info.get("controller_termination")):
        return False
    if _enum_text(info.get("controller_task_result")) not in (None, "", "RUNNING"):
        return False
    for key in (
        "controller_reason",
        "controller_details",
        "first_blocker",
        "pending_blocker",
        "controller_blocker",
    ):
        if not _is_empty(info.get(key)):
            return False
    for key in ("controller_blocked", "controller_blocked_encoded_as_truncation"):
        if bool(info.get(key, False)):
            return False
    mapping = info.get("termination_mapping")
    if not isinstance(mapping, Mapping):
        return False
    if _enum_text(mapping.get("controller_result")) not in (None, "", "RUNNING"):
        return False
    for key in (
        "controller_reason",
        "controller_details",
        "first_blocker",
        "active_sources",
        "primary_source",
    ):
        if not _is_empty(mapping.get(key)):
            return False
    if bool(mapping.get("controller_blocked_encoded_as_truncation", False)):
        return False
    raw = info.get("raw_controller_frame")
    if raw is not None:
        if not _is_empty(_member(raw, "termination")):
            return False
        if not _is_empty(_member(raw, "first_blocker")):
            return False
    return True


def _frame_failures(
    frame: Any,
    *,
    phase_id: str,
    tick0: bool,
    allowed_lifecycles: tuple[str, ...] = ("EXECUTE_MOTION",),
) -> list[str]:
    info = frame.info
    failures: list[str] = []
    if frame.state_id not in PHASE_IDS:
        failures.append("unknown_fsm_phase")
    if tick0 and frame.state_id != phase_id:
        failures.append("reset_phase_mismatch")
    if any(_hard_termination_flags(frame).values()):
        failures.append("authoritative_terminal_signal")
    if not _safety_is_neutral(frame):
        failures.append("non_neutral_safety_projection")
    if not _controller_is_unblocked(
        frame, allowed_lifecycles=allowed_lifecycles
    ):
        failures.append("controller_lifecycle_or_blocker_invalid")
    for name in (
        "in_episode_root_pose_writes",
        "in_episode_root_velocity_writes",
    ):
        if not _strict_zero_counter(info, name):
            failures.append(f"{name}_not_zero")
    if not _strict_zero_counter(info, "recording_accesses"):
        failures.append("recording_runtime_access_not_zero")
    if tick0:
        if int(frame.physics_tick) != 0 or float(frame.sim_time_s) != 0.0:
            failures.append("reset_logical_clock_not_zero")
        if info.get("training_phase_snapshot") != phase_id:
            failures.append("training_phase_snapshot_metadata_mismatch")
        restoration = info.get("phase_snapshot_restoration")
        if (
            not isinstance(restoration, Mapping)
            or restoration.get("requested_phase") != phase_id
            or restoration.get("snapshot_validated") is not True
        ):
            failures.append("phase_snapshot_restoration_not_verified")
    return failures


def _expected_next_phase(phase_id: str) -> str | None:
    index = PHASE_IDS.index(phase_id)
    return None if index + 1 == len(PHASE_IDS) else PHASE_IDS[index + 1]


def _boundary_transition_event_present(
    frame: Any, *, from_phase: str, to_phase: str
) -> bool:
    raw = frame.info.get("raw_controller_frame")
    events = _member(raw, "events", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return False
    expected_reason = f"advance fixed graph {from_phase}->{to_phase}"
    return any(
        _member(event, "state_id") == to_phase
        and _enum_text(_member(event, "from_lifecycle")) == "DONE"
        and _enum_text(_member(event, "to_lifecycle")) == "WAIT_ENTRY"
        and _member(event, "reason") == expected_reason
        for event in events
    )


@dataclass(slots=True)
class PhaseTickAudit:
    """Fail-fast per-physics-tick audit with compact streaming evidence."""

    phase_id: str
    tick_count: int = 0
    zero_raw_tick_count: int = 0
    zero_projected_tick_count: int = 0
    zero_fast_path_tick_count: int = 0
    nominal_applied_equal_tick_count: int = 0
    neutral_safety_tick_count: int = 0
    nonterminal_tick_count: int = 0
    lifecycle_valid_unblocked_tick_count: int = 0
    boundary_transition_event_count: int = 0
    zero_root_write_tick_count: int = 0
    zero_recording_access_tick_count: int = 0
    reference_conformance_diagnostic_tick_count: int = 0
    state_ids_observed: list[str] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    _nominal_sha: Any = field(default_factory=hashlib.sha256, repr=False)
    _applied_sha: Any = field(default_factory=hashlib.sha256, repr=False)
    _raw_residual_sha: Any = field(default_factory=hashlib.sha256, repr=False)
    _projected_residual_sha: Any = field(default_factory=hashlib.sha256, repr=False)

    def validate_tick0(self, frame: Any) -> dict[str, Any]:
        failures = _frame_failures(frame, phase_id=self.phase_id, tick0=True)
        result = {
            "physics_tick": int(frame.physics_tick),
            "sim_time_s": float(frame.sim_time_s),
            "state_id": str(frame.state_id),
            "hard_termination_flags": _hard_termination_flags(frame),
            "recording_reference_conformance_outside_30pct_diagnostic": bool(
                _member(
                    frame.termination_signals,
                    "reference_conformance_outside_30pct",
                    False,
                )
            ),
            "safety_neutral": _safety_is_neutral(frame),
            "controller_running_unblocked": _controller_is_unblocked(
                frame, allowed_lifecycles=("EXECUTE_MOTION",)
            ),
            "in_episode_root_pose_writes": frame.info.get(
                "in_episode_root_pose_writes"
            ),
            "in_episode_root_velocity_writes": frame.info.get(
                "in_episode_root_velocity_writes"
            ),
            "recording_accesses": frame.info.get("recording_accesses"),
            "passed": not failures,
            "failure_reasons": failures,
        }
        if failures:
            self.violations.append(
                {"physics_tick": int(frame.physics_tick), "reasons": failures}
            )
            raise _StopUnsafeRollout(
                f"{self.phase_id} tick-zero authoritative gate failed: "
                + ", ".join(failures)
            )
        return result

    def append(self, before: Any, after: Any, projection: Any) -> None:
        expected_next = _expected_next_phase(self.phase_id)
        boundary = bool(
            expected_next is not None and after.state_id == expected_next
        )
        allowed_lifecycles = (
            ("WAIT_ENTRY", "EXECUTE_MOTION")
            if boundary
            else ("EXECUTE_MOTION", "VERIFY_RESULT")
        )
        failures = _frame_failures(
            after,
            phase_id=self.phase_id,
            tick0=False,
            allowed_lifecycles=allowed_lifecycles,
        )
        if before.state_id != self.phase_id:
            failures.append("physics_tick_started_outside_requested_phase")
        if after.state_id not in (self.phase_id, expected_next):
            failures.append("phase_boundary_skipped_or_reversed")
        if boundary and not _boundary_transition_event_present(
            after, from_phase=self.phase_id, to_phase=expected_next
        ):
            failures.append("phase_boundary_transition_event_missing")
        if int(after.physics_tick) != int(before.physics_tick) + 1:
            failures.append("physics_tick_not_strictly_incremented")
        expected_time = float(before.sim_time_s) + 1.0 / PHYSICS_HZ
        if not math.isclose(
            float(after.sim_time_s), expected_time, rel_tol=0.0, abs_tol=1.0e-12
        ):
            failures.append("sim_time_not_exact_120hz_increment")

        raw = full12_bytes(projection.raw_residual_full12)
        projected = full12_bytes(projection.safe_projected_residual_full12)
        nominal = full12_bytes(before.nominal_action_full12)
        applied = full12_bytes(projection.applied_action_full12)
        if raw != _ZERO_BYTES:
            failures.append("raw_policy_residual_not_bitwise_zero")
        if projected != _ZERO_BYTES:
            failures.append("projected_residual_not_bitwise_zero")
        if not bool(projection.zero_residual_fast_path):
            failures.append("zero_residual_fast_path_not_used")
        if nominal != applied:
            failures.append("zero_residual_applied_action_differs_from_nominal")
        if bool(projection.hard_safety_modified):
            failures.append("hard_safety_modified_zero_residual_action")

        self.tick_count += 1
        self.zero_raw_tick_count += int(raw == _ZERO_BYTES)
        self.zero_projected_tick_count += int(projected == _ZERO_BYTES)
        self.zero_fast_path_tick_count += int(bool(projection.zero_residual_fast_path))
        self.nominal_applied_equal_tick_count += int(nominal == applied)
        self.neutral_safety_tick_count += int(_safety_is_neutral(after))
        self.nonterminal_tick_count += int(
            not any(_hard_termination_flags(after).values())
        )
        self.lifecycle_valid_unblocked_tick_count += int(
            _controller_is_unblocked(
                after, allowed_lifecycles=allowed_lifecycles
            )
        )
        self.boundary_transition_event_count += int(
            boundary
            and _boundary_transition_event_present(
                after, from_phase=self.phase_id, to_phase=expected_next
            )
        )
        self.zero_root_write_tick_count += int(
            _strict_zero_counter(after.info, "in_episode_root_pose_writes")
            and _strict_zero_counter(after.info, "in_episode_root_velocity_writes")
        )
        self.zero_recording_access_tick_count += int(
            _strict_zero_counter(after.info, "recording_accesses")
        )
        self.reference_conformance_diagnostic_tick_count += int(
            bool(
                _member(
                    after.termination_signals,
                    "reference_conformance_outside_30pct",
                    False,
                )
            )
        )
        if not self.state_ids_observed or self.state_ids_observed[-1] != after.state_id:
            self.state_ids_observed.append(str(after.state_id))
        self._nominal_sha.update(nominal)
        self._applied_sha.update(applied)
        self._raw_residual_sha.update(raw)
        self._projected_residual_sha.update(projected)
        if failures:
            self.violations.append(
                {"physics_tick": int(after.physics_tick), "reasons": failures}
            )
            raise _StopUnsafeRollout(
                f"{self.phase_id} unsafe physics tick {after.physics_tick}: "
                + ", ".join(failures)
            )

    def as_record(self) -> dict[str, Any]:
        counts = {
            "tick_count": self.tick_count,
            "zero_raw_tick_count": self.zero_raw_tick_count,
            "zero_projected_tick_count": self.zero_projected_tick_count,
            "zero_fast_path_tick_count": self.zero_fast_path_tick_count,
            "nominal_applied_equal_tick_count": (
                self.nominal_applied_equal_tick_count
            ),
            "neutral_safety_tick_count": self.neutral_safety_tick_count,
            "nonterminal_tick_count": self.nonterminal_tick_count,
            "lifecycle_valid_unblocked_tick_count": (
                self.lifecycle_valid_unblocked_tick_count
            ),
            "zero_root_write_tick_count": self.zero_root_write_tick_count,
            "zero_recording_access_tick_count": (
                self.zero_recording_access_tick_count
            ),
        }
        passed = bool(
            self.tick_count > 0
            and all(value == self.tick_count for value in counts.values())
            and not self.violations
            and self._nominal_sha.digest() == self._applied_sha.digest()
            and self._raw_residual_sha.digest()
            == self._projected_residual_sha.digest()
        )
        return {
            "schema": "wlr50_clean.phase_zero_residual_tick_audit.v1",
            **counts,
            "state_ids_observed": list(self.state_ids_observed),
            "boundary_transition_event_count": self.boundary_transition_event_count,
            "nominal_action_binary64_sha256": self._nominal_sha.hexdigest(),
            "applied_action_binary64_sha256": self._applied_sha.hexdigest(),
            "raw_residual_binary64_sha256": self._raw_residual_sha.hexdigest(),
            "projected_residual_binary64_sha256": (
                self._projected_residual_sha.hexdigest()
            ),
            "reference_conformance_outside_30pct_diagnostic_tick_count": (
                self.reference_conformance_diagnostic_tick_count
            ),
            "reference_conformance_is_quality_diagnostic_only": True,
            "violations": list(self.violations),
            "passed": passed,
        }


def _binding_value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def build_contract_binding(
    snapshot_bundle: Any,
    effective_entry_contract: Any,
    holdout_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact upstream proof binding stored in the rollout artifact."""

    snapshot_sha = _binding_value(snapshot_bundle, "bundle_sha256")
    effective_sha = _binding_value(effective_entry_contract, "contract_sha256")
    effective_snapshot_sha = _binding_value(
        effective_entry_contract, "phase_snapshot_bundle_sha256"
    )
    if (
        not isinstance(snapshot_sha, str)
        or len(snapshot_sha) != 64
        or not isinstance(effective_sha, str)
        or len(effective_sha) != 64
        or effective_snapshot_sha != snapshot_sha
        or holdout_evidence.get("passed") is not True
        or holdout_evidence.get("phase_snapshot_bundle_sha256") != snapshot_sha
        or holdout_evidence.get("phase_effective_entry_contract_sha256")
        != effective_sha
    ):
        raise PhaseZeroResidualRolloutError(
            "rollout upstream snapshot/effective-entry/holdout binding is invalid"
        )
    result = {
        "phase_snapshot_bundle_sha256": snapshot_sha,
        "phase_snapshot_manifest_sha256": _binding_value(
            snapshot_bundle, "manifest_sha256"
        ),
        "phase_effective_entry_contract_sha256": effective_sha,
        "phase_effective_entry_contract_file_sha256": _binding_value(
            effective_entry_contract, "file_sha256"
        ),
        "phase_effective_entry_contract_sidecar_sha256": _binding_value(
            effective_entry_contract, "sidecar_file_sha256"
        ),
        "holdout_acceptance_path": str(
            Path(str(holdout_evidence.get("path", ""))).resolve()
        ),
        "holdout_acceptance_sha256": holdout_evidence.get("sha256"),
        "holdout_run_manifest_path": str(
            Path(str(holdout_evidence.get("run_manifest", ""))).resolve()
        ),
        "holdout_run_manifest_sha256": holdout_evidence.get(
            "run_manifest_sha256"
        ),
        "source_git_commit": holdout_evidence.get("source_git_commit"),
    }
    for name, value in result.items():
        if name.endswith("_path"):
            if not value:
                raise PhaseZeroResidualRolloutError(
                    f"rollout upstream binding omits {name}"
                )
        elif not isinstance(value, str) or len(value) not in (40, 64):
            raise PhaseZeroResidualRolloutError(
                f"rollout upstream binding has invalid {name}"
            )
    return result


def _not_run_phase(phase_id: str, reason: str) -> dict[str, Any]:
    return {
        "phase_id": phase_id,
        "status": "NOT_RUN_FAIL_CLOSED",
        "passed": False,
        "failure_reasons": [reason],
    }


def run_phase_zero_residual_rollout(
    episode: Any,
    *,
    seed: int,
    contract_binding: Mapping[str, Any],
    max_decisions_per_phase: int = MAX_DECISIONS_PER_PHASE,
) -> dict[str, Any]:
    """Run P01--P13 reset windows, stopping physical work on first violation."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PhaseZeroResidualRolloutError("rollout seed must be a non-negative integer")
    if max_decisions_per_phase != MAX_DECISIONS_PER_PHASE:
        raise PhaseZeroResidualRolloutError(
            "phase zero-residual rollout horizon must be exactly 64 decisions"
        )
    binding = dict(contract_binding)
    required_binding = {
        "phase_snapshot_bundle_sha256",
        "phase_snapshot_manifest_sha256",
        "phase_effective_entry_contract_sha256",
        "phase_effective_entry_contract_file_sha256",
        "phase_effective_entry_contract_sidecar_sha256",
        "holdout_acceptance_path",
        "holdout_acceptance_sha256",
        "holdout_run_manifest_path",
        "holdout_run_manifest_sha256",
        "source_git_commit",
    }
    if set(binding) != required_binding:
        raise PhaseZeroResidualRolloutError("rollout contract binding is incomplete")

    phase_records: list[dict[str, Any]] = []
    fatal_reason: str | None = None
    for phase_id in PHASE_IDS:
        if fatal_reason is not None:
            phase_records.append(_not_run_phase(phase_id, fatal_reason))
            continue
        audit = PhaseTickAudit(phase_id)
        record: dict[str, Any] = {
            "phase_id": phase_id,
            "status": "RUNNING",
            "passed": False,
            "seed": seed,
            "max_decisions": MAX_DECISIONS_PER_PHASE,
        }
        try:
            episode.tick_callback = None
            _, reset_info = episode.reset(
                seed=seed,
                options={"training_phase_snapshot": phase_id},
            )
            frame = episode.frame
            if frame is None:
                raise PhaseZeroResidualRolloutError(
                    f"{phase_id} reset returned no authoritative frame"
                )
            tick0 = audit.validate_tick0(frame)
            record.update(
                {
                    "reset_count": reset_info.get("reset_count"),
                    "reset_mode": _member(
                        reset_info.get("phase_snapshot_restoration"), "mode"
                    ),
                    "reset_prime_tick_count": reset_info.get(
                        "reset_prime_tick_count"
                    ),
                    "tick0": tick0,
                }
            )
            episode.tick_callback = audit.append
            decision_count = 0
            boundary_reached = False
            last_step = None
            while decision_count < MAX_DECISIONS_PER_PHASE:
                if episode.frame is None or episode.frame.state_id != phase_id:
                    boundary_reached = True
                    break
                last_step = episode.step(
                    ZERO_FULL12, stop_after_phase_id=phase_id
                )
                decision_count += 1
                if last_step.terminated:
                    raise _StopUnsafeRollout(
                        f"{phase_id} became terminal during zero-residual rollout"
                    )
                if bool(last_step.info.get("phase_curriculum_boundary", False)):
                    if not last_step.truncated:
                        raise _StopUnsafeRollout(
                            f"{phase_id} boundary did not use the bounded curriculum stop"
                        )
                    boundary_reached = True
                    break
                if last_step.truncated:
                    raise _StopUnsafeRollout(
                        f"{phase_id} truncated before its next phase boundary"
                    )
            if episode.frame is None:
                raise PhaseZeroResidualRolloutError(
                    f"{phase_id} lost its authoritative final frame"
                )
            final_state = str(episode.frame.state_id)
            expected_next = _expected_next_phase(phase_id)
            if boundary_reached and final_state != expected_next:
                raise _StopUnsafeRollout(
                    f"{phase_id} boundary reached {final_state}, expected {expected_next}"
                )
            if not boundary_reached and (
                decision_count != MAX_DECISIONS_PER_PHASE or final_state != phase_id
            ):
                raise _StopUnsafeRollout(
                    f"{phase_id} rollout ended without a valid boundary or max horizon"
                )
            tick_audit = audit.as_record()
            if not tick_audit["passed"]:
                raise _StopUnsafeRollout(
                    f"{phase_id} full-rate zero-residual audit failed"
                )
            record.update(
                {
                    "status": (
                        "NEXT_PHASE_BOUNDARY_REACHED"
                        if boundary_reached
                        else "MAX_64_DECISIONS_REACHED"
                    ),
                    "passed": True,
                    "decision_count": decision_count,
                    "physics_tick_count": int(episode.frame.physics_tick),
                    "final_state_id": final_state,
                    "expected_next_phase_id": expected_next,
                    "boundary_reached": boundary_reached,
                    "last_step_terminated": bool(
                        False if last_step is None else last_step.terminated
                    ),
                    "last_step_truncated_for_boundary": bool(
                        last_step is not None
                        and last_step.truncated
                        and last_step.info.get("phase_curriculum_boundary", False)
                    ),
                    "tick_audit": tick_audit,
                    "failure_reasons": [],
                }
            )
        except Exception as exc:
            fatal_reason = f"{type(exc).__name__}: {exc}"
            record.update(
                {
                    "status": "FAILED_FAIL_CLOSED",
                    "passed": False,
                    "decision_count": int(getattr(episode, "decision_count", 0)),
                    "physics_tick_count": (
                        None
                        if getattr(episode, "frame", None) is None
                        else int(episode.frame.physics_tick)
                    ),
                    "final_state_id": (
                        None
                        if getattr(episode, "frame", None) is None
                        else str(episode.frame.state_id)
                    ),
                    "boundary_reached": False,
                    "tick_audit": audit.as_record(),
                    "failure_reasons": [fatal_reason],
                }
            )
        finally:
            episode.tick_callback = None
        phase_records.append(record)

    executed = [row for row in phase_records if row["status"] != "NOT_RUN_FAIL_CLOSED"]
    checks = {
        "holdout_acceptance_prevalidated": True,
        "snapshot_and_effective_entry_contracts_bound": True,
        "one_backend_instance": True,
        "all_p01_p13_resets_executed": len(executed) == len(PHASE_IDS),
        "all_windows_ended_at_next_boundary_or_max_64_decisions": all(
            row.get("status")
            in {"NEXT_PHASE_BOUNDARY_REACHED", "MAX_64_DECISIONS_REACHED"}
            for row in phase_records
        ),
        "exact_120hz_15hz_cadence": all(
            isinstance(row.get("physics_tick_count"), int)
            and 0 < int(row["physics_tick_count"])
            <= MAX_DECISIONS_PER_PHASE * PHYSICS_TICKS_PER_DECISION
            and row.get("tick_audit", {}).get("tick_count")
            == row.get("physics_tick_count")
            for row in phase_records
        ),
        "zero_residual_bitwise_nominal_on_every_physics_tick": all(
            row.get("tick_audit", {}).get("passed") is True
            for row in phase_records
        ),
        "no_post_tick0_root_state_writes": all(
            row.get("tick_audit", {}).get("zero_root_write_tick_count")
            == row.get("tick_audit", {}).get("tick_count")
            for row in phase_records
        ),
        "no_terminal_safety_or_controller_blocker": all(
            row.get("tick_audit", {}).get("neutral_safety_tick_count")
            == row.get("tick_audit", {}).get("tick_count")
            and row.get("tick_audit", {}).get("nonterminal_tick_count")
            == row.get("tick_audit", {}).get("tick_count")
            and row.get("tick_audit", {}).get(
                "lifecycle_valid_unblocked_tick_count"
            )
            == row.get("tick_audit", {}).get("tick_count")
            and row.get("tick0", {}).get("passed") is True
            for row in phase_records
        ),
        "no_runtime_recording_access": all(
            row.get("tick_audit", {}).get("zero_recording_access_tick_count")
            == row.get("tick_audit", {}).get("tick_count")
            for row in phase_records
        ),
        "phase_boundary_order_preserved": all(
            row.get("boundary_reached") is False
            or row.get("final_state_id") == row.get("expected_next_phase_id")
            for row in phase_records
        ),
    }
    passed = all(checks.values()) and all(
        row.get("passed") is True for row in phase_records
    )
    total_ticks = sum(
        int(row.get("tick_audit", {}).get("tick_count", 0))
        for row in phase_records
    )
    total_decisions = sum(
        int(row.get("decision_count", 0)) for row in phase_records
    )
    return {
        "schema": ARTIFACT_SCHEMA,
        "artifact_role": "PHASE_CURRICULUM_TRAINING_PREREQUISITE",
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "created_at_utc": _utc_text(),
        "seed": seed,
        "backend_instance_count": 1,
        "phase_reset_count": len(executed),
        "phases": list(PHASE_IDS),
        "max_decisions_per_phase": MAX_DECISIONS_PER_PHASE,
        "physics_hz": PHYSICS_HZ,
        "decision_hz": DECISION_HZ,
        "physics_ticks_per_decision": PHYSICS_TICKS_PER_DECISION,
        "zero_residual_full12": list(ZERO_FULL12),
        "contract_binding": binding,
        "checks": checks,
        "phase_rollouts": phase_records,
        "total_policy_decisions": total_decisions,
        "total_physics_ticks": total_ticks,
        "failure_reasons": [] if fatal_reason is None else [fatal_reason],
    }


def _strict_json_object(data: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise PhaseZeroResidualRolloutError(
                    f"{label} contains duplicate key {name!r}"
                )
            result[name] = value
        return result

    def reject_constant(value: str) -> None:
        raise PhaseZeroResidualRolloutError(
            f"{label} contains non-finite JSON constant {value}"
        )

    try:
        payload = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseZeroResidualRolloutError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise PhaseZeroResidualRolloutError(f"{label} must be a JSON object")
    return payload


def validate_phase_zero_residual_rollout_payload(
    payload: Mapping[str, Any],
    *,
    expected_contract_binding: Mapping[str, Any],
) -> None:
    """Strict semantic validation shared by tests and the managed-run loader."""

    checks = payload.get("checks")
    rows = payload.get("phase_rollouts")
    if (
        payload.get("schema") != ARTIFACT_SCHEMA
        or payload.get("artifact_role")
        != "PHASE_CURRICULUM_TRAINING_PREREQUISITE"
        or payload.get("status") != "PASSED"
        or payload.get("passed") is not True
        or isinstance(payload.get("seed"), bool)
        or not isinstance(payload.get("seed"), int)
        or payload.get("seed") < 0
        or payload.get("backend_instance_count") != 1
        or payload.get("phase_reset_count") != len(PHASE_IDS)
        or payload.get("phases") != list(PHASE_IDS)
        or payload.get("max_decisions_per_phase") != MAX_DECISIONS_PER_PHASE
        or payload.get("physics_hz") != PHYSICS_HZ
        or payload.get("decision_hz") != DECISION_HZ
        or payload.get("physics_ticks_per_decision")
        != PHYSICS_TICKS_PER_DECISION
        or payload.get("zero_residual_full12") != list(ZERO_FULL12)
        or payload.get("contract_binding") != dict(expected_contract_binding)
        or payload.get("failure_reasons") != []
        or not isinstance(checks, Mapping)
        or len(checks) != len(_CHECK_NAMES)
        or set(checks) != set(_CHECK_NAMES)
        or not all(checks.get(name) is True for name in _CHECK_NAMES)
        or not isinstance(rows, list)
        or len(rows) != len(PHASE_IDS)
    ):
        raise PhaseZeroResidualRolloutError(
            "phase zero-residual rollout header/checks are incomplete or failed"
        )
    if tuple(row.get("phase_id") for row in rows if isinstance(row, Mapping)) != PHASE_IDS:
        raise PhaseZeroResidualRolloutError("phase rollout order is not P01-P13")
    total_decisions = 0
    total_ticks = 0
    for phase_id, row in zip(PHASE_IDS, rows, strict=True):
        if not isinstance(row, Mapping):
            raise PhaseZeroResidualRolloutError(f"{phase_id} rollout row is invalid")
        audit = row.get("tick_audit")
        tick0 = row.get("tick0")
        decisions = row.get("decision_count")
        ticks = row.get("physics_tick_count")
        boundary = row.get("boundary_reached")
        if (
            row.get("passed") is not True
            or row.get("status")
            not in {"NEXT_PHASE_BOUNDARY_REACHED", "MAX_64_DECISIONS_REACHED"}
            or row.get("failure_reasons") != []
            or row.get("seed") != payload.get("seed")
            or not isinstance(decisions, int)
            or isinstance(decisions, bool)
            or not 1 <= decisions <= MAX_DECISIONS_PER_PHASE
            or not isinstance(ticks, int)
            or isinstance(ticks, bool)
            or not 1 <= ticks <= decisions * PHYSICS_TICKS_PER_DECISION
            or not isinstance(tick0, Mapping)
            or tick0.get("passed") is not True
            or not isinstance(audit, Mapping)
            or audit.get("passed") is not True
            or audit.get("tick_count") != ticks
            or audit.get("violations") != []
        ):
            raise PhaseZeroResidualRolloutError(
                f"{phase_id} rollout row failed its bounded full-rate audit"
            )
        for count_name in (
            "zero_raw_tick_count",
            "zero_projected_tick_count",
            "zero_fast_path_tick_count",
            "nominal_applied_equal_tick_count",
            "neutral_safety_tick_count",
            "nonterminal_tick_count",
            "lifecycle_valid_unblocked_tick_count",
            "zero_root_write_tick_count",
            "zero_recording_access_tick_count",
        ):
            if audit.get(count_name) != ticks:
                raise PhaseZeroResidualRolloutError(
                    f"{phase_id} rollout audit count {count_name} is incomplete"
                )
        if (
            audit.get("nominal_action_binary64_sha256")
            != audit.get("applied_action_binary64_sha256")
            or audit.get("raw_residual_binary64_sha256")
            != audit.get("projected_residual_binary64_sha256")
        ):
            raise PhaseZeroResidualRolloutError(
                f"{phase_id} zero-residual binary64 proof is inconsistent"
            )
        if boundary is True:
            if (
                row.get("status") != "NEXT_PHASE_BOUNDARY_REACHED"
                or row.get("final_state_id") != _expected_next_phase(phase_id)
                or row.get("last_step_terminated") is not False
                or row.get("last_step_truncated_for_boundary") is not True
                or audit.get("boundary_transition_event_count") != 1
            ):
                raise PhaseZeroResidualRolloutError(
                    f"{phase_id} phase boundary evidence is invalid"
                )
        elif (
            boundary is not False
            or row.get("status") != "MAX_64_DECISIONS_REACHED"
            or decisions != MAX_DECISIONS_PER_PHASE
            or ticks != MAX_DECISIONS_PER_PHASE * PHYSICS_TICKS_PER_DECISION
            or row.get("final_state_id") != phase_id
            or row.get("last_step_terminated") is not False
            or row.get("last_step_truncated_for_boundary") is not False
            or audit.get("boundary_transition_event_count") != 0
        ):
            raise PhaseZeroResidualRolloutError(
                f"{phase_id} max-horizon evidence is invalid"
            )
        total_decisions += decisions
        total_ticks += ticks
    if (
        payload.get("total_policy_decisions") != total_decisions
        or payload.get("total_physics_ticks") != total_ticks
    ):
        raise PhaseZeroResidualRolloutError("phase rollout aggregate counts are invalid")


def _validate_rollout_context_binding(
    context: Any, expected_contract_binding: Mapping[str, Any]
) -> None:
    expected = {
        "phase_snapshot_bundle_sha256": getattr(
            context.snapshot_bundle, "bundle_sha256", None
        ),
        "phase_snapshot_manifest_sha256": getattr(
            context.snapshot_bundle, "manifest_sha256", None
        ),
        "phase_effective_entry_contract_sha256": getattr(
            context.effective_entry_contract, "contract_sha256", None
        ),
        "phase_effective_entry_contract_file_sha256": getattr(
            context.effective_entry_contract, "file_sha256", None
        ),
        "phase_effective_entry_contract_sidecar_sha256": getattr(
            context.effective_entry_contract, "sidecar_file_sha256", None
        ),
        "source_git_commit": context.git_commit,
    }
    if any(expected_contract_binding.get(name) != value for name, value in expected.items()):
        raise PhaseZeroResidualRolloutError(
            "phase rollout binding differs from the current committed contracts"
        )


def _validate_rollout_invocation(
    arguments: Any,
    *,
    project_root: Path,
    context: Any,
    expected_contract_binding: Mapping[str, Any],
    seed: int,
) -> None:
    from .phase_effective_entry_holdout import (
        _one_invocation_value,
        _require_invocation_flag,
    )
    from .vector_benchmark_matrix import _absolute_lexical

    value_flags = (
        "--training-config",
        "--interface-config",
        "--snapshot-root",
        "--phase-snapshot-prime-physics-steps",
        "--phase-effective-entry-holdout-acceptance",
        "--episode-count",
        "--policy-decisions",
        "--seed-set",
        "--residual-mode",
        "--run-dir",
        "--seed",
        "--num-envs",
    )
    if (
        not isinstance(arguments, Sequence)
        or isinstance(arguments, (str, bytes))
        or any(not isinstance(value, str) for value in arguments)
        or len(arguments) != len(value_flags) * 2 + 1
    ):
        raise PhaseZeroResidualRolloutError(
            "phase rollout invocation is not the locked managed command"
        )
    values = {flag: _one_invocation_value(arguments, flag) for flag in value_flags}
    _require_invocation_flag(arguments, "--deterministic")
    expected_scalars = {
        "--phase-snapshot-prime-physics-steps": "1",
        "--episode-count": str(len(PHASE_IDS)),
        "--policy-decisions": str(MAX_DECISIONS_PER_PHASE),
        "--seed-set": "train",
        "--residual-mode": "zero",
        "--run-dir": "<reserved-immutable-run-dir>",
        "--seed": str(seed),
        "--num-envs": "1",
    }
    if any(values[flag] != value for flag, value in expected_scalars.items()):
        raise PhaseZeroResidualRolloutError(
            "phase rollout invocation differs from the locked zero-residual gate"
        )

    def project_path(value: str) -> Path:
        candidate = Path(value)
        return _absolute_lexical(
            candidate if candidate.is_absolute() else project_root / candidate
        )

    expected_paths = {
        "--training-config": project_root / "configs" / "ppo_training_phase_v1.yaml",
        "--interface-config": project_root / "configs" / "ppo_interface_v2.yaml",
        "--snapshot-root": Path(context.snapshot_bundle.snapshot_root),
        "--phase-effective-entry-holdout-acceptance": _absolute_lexical(
            str(expected_contract_binding.get("holdout_acceptance_path", ""))
        ),
    }
    if any(
        project_path(values[flag]) != _absolute_lexical(expected)
        for flag, expected in expected_paths.items()
    ):
        raise PhaseZeroResidualRolloutError(
            "phase rollout invocation uses an unbound config, snapshot, or holdout path"
        )


def validate_phase_zero_residual_rollout_evidence(
    path: Path | str,
    *,
    project_root: Path | str,
    expected_contract_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one finalized managed rollout without accepting replacement bytes."""

    from .artifacts import ArtifactError
    from .phase_effective_entry_holdout import (
        HOLDOUT_RUN_KIND,
        OUTPUT_FILENAME as HOLDOUT_OUTPUT_FILENAME,
        PhaseEffectiveEntryHoldoutError,
        _current_context,
        _stdout_objects,
        _validate_runtime_and_frozen_pair,
        _validate_started_and_final_manifest,
    )
    from .vector_benchmark_matrix import (
        VectorBenchmarkMatrixError,
        _SnapshotCache,
        _absolute_lexical,
        _revalidate_snapshots,
        _snapshot,
        _validated_file_record,
        validate_managed_run_directory,
    )

    try:
        root = _absolute_lexical(project_root)
        selected = _absolute_lexical(path)
        if selected.name != ARTIFACT_FILENAME:
            raise PhaseZeroResidualRolloutError(
                f"rollout evidence must be named {ARTIFACT_FILENAME}"
            )
        run_dir = validate_managed_run_directory(
            selected.parent, project_root=root, run_kind=RUN_KIND
        )
        cache: _SnapshotCache = {}
        context = _current_context(
            root,
            config_paths=None,
            snapshot_bundle=None,
            effective_entry_contract=None,
            cache=cache,
        )
        _validate_rollout_context_binding(context, expected_contract_binding)
        evidence_snapshot = _snapshot(
            selected,
            label="phase zero-residual rollout evidence",
            cache=cache,
            trusted_root=run_dir,
        )
        payload = _strict_json_object(
            evidence_snapshot.data, label="phase zero-residual rollout evidence"
        )
        validate_phase_zero_residual_rollout_payload(
            payload, expected_contract_binding=expected_contract_binding
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest, _ = _validate_started_and_final_manifest(
            run_dir,
            run_kind=RUN_KIND,
            entrypoint="wlr50_clean.ppo.cli",
            subcommand=SUBCOMMAND,
            training_stage=TRAINING_STAGE,
            context=context,
            cache=cache,
            expected_seed=payload["seed"],
        )
        _validate_rollout_invocation(
            manifest.get("invocation_arguments"),
            project_root=root,
            context=context,
            expected_contract_binding=expected_contract_binding,
            seed=payload["seed"],
        )
        artifacts = manifest["artifacts"]
        logs = manifest["logs"]
        expected_artifacts = {
            ARTIFACT_FILENAME,
            "live_command_result.json",
            "committed_runtime_identity.before.json",
            "committed_runtime_identity.after.json",
            "frozen_hashes.before.json",
            "frozen_hashes.after.json",
        }
        if set(artifacts) != expected_artifacts or set(logs) != {
            "stdout.log",
            "stderr.log",
        }:
            raise PhaseZeroResidualRolloutError(
                "phase rollout finalized inventory is incomplete or ambiguous"
            )
        _validated_file_record(
            run_dir,
            artifacts.get(ARTIFACT_FILENAME),
            expected_relative_path=ARTIFACT_FILENAME,
            label="phase rollout evidence",
            cache=cache,
        )
        live_snapshot = _validated_file_record(
            run_dir,
            artifacts.get("live_command_result.json"),
            expected_relative_path="live_command_result.json",
            label="phase rollout live command result",
            cache=cache,
        )
        live = _strict_json_object(
            live_snapshot.data, label="phase rollout live command result"
        )
        if live != {
            "schema": "wlr50_clean.live_command_result.v1",
            "command": SUBCOMMAND,
            "exit_code": 0,
        }:
            raise PhaseZeroResidualRolloutError(
                "phase rollout live command result is invalid"
            )
        stdout_snapshot = _validated_file_record(
            run_dir,
            logs.get("stdout.log"),
            expected_relative_path="stdout.log",
            label="phase rollout stdout",
            cache=cache,
        )
        _validated_file_record(
            run_dir,
            logs.get("stderr.log"),
            expected_relative_path="stderr.log",
            label="phase rollout stderr",
            cache=cache,
        )
        _validate_runtime_and_frozen_pair(
            run_dir, manifest, context=context, cache=cache
        )
        stdout_rollouts = [
            row
            for row in _stdout_objects(stdout_snapshot.path, cache=cache)
            if row.get("schema") == ARTIFACT_SCHEMA
        ]
        if len(stdout_rollouts) != 1 or stdout_rollouts[0] != payload:
            raise PhaseZeroResidualRolloutError(
                "phase rollout artifact is not uniquely bound to finalized stdout"
            )

        holdout_path = _absolute_lexical(
            str(expected_contract_binding.get("holdout_acceptance_path", ""))
        )
        holdout_manifest_path = _absolute_lexical(
            str(expected_contract_binding.get("holdout_run_manifest_path", ""))
        )
        holdout_run_dir = validate_managed_run_directory(
            holdout_path.parent,
            project_root=root,
            run_kind=HOLDOUT_RUN_KIND,
        )
        if (
            holdout_path != holdout_run_dir / HOLDOUT_OUTPUT_FILENAME
            or holdout_manifest_path != holdout_run_dir / "run_manifest.json"
        ):
            raise PhaseZeroResidualRolloutError(
                "phase rollout binding names an invalid holdout managed run"
            )
        for upstream_path, hash_name, label in (
            (
                holdout_path,
                "holdout_acceptance_sha256",
                "phase rollout holdout acceptance",
            ),
            (
                holdout_manifest_path,
                "holdout_run_manifest_sha256",
                "phase rollout holdout run manifest",
            ),
        ):
            upstream = _snapshot(
                upstream_path,
                label=label,
                cache=cache,
                trusted_root=root,
            )
            if upstream.sha256 != expected_contract_binding.get(hash_name):
                raise PhaseZeroResidualRolloutError(
                    f"{label} changed after the rollout binding was created"
                )
        _revalidate_snapshots(cache, project_root=root)
        manifest_snapshot = _snapshot(
            manifest_path, label="phase rollout run manifest", cache=cache
        )
        return {
            "schema": TRAINING_EVIDENCE_SCHEMA,
            "path": str(selected),
            "sha256": evidence_snapshot.sha256,
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": manifest_snapshot.sha256,
            "contract_binding": dict(expected_contract_binding),
            "seed": payload["seed"],
            "passed": True,
        }
    except PhaseZeroResidualRolloutError:
        raise
    except (
        ArtifactError,
        OSError,
        ValueError,
        PhaseEffectiveEntryHoldoutError,
        VectorBenchmarkMatrixError,
    ) as exc:
        raise PhaseZeroResidualRolloutError(
            f"phase zero-residual rollout evidence rejected: {exc}"
        ) from exc


__all__ = [
    "ARTIFACT_FILENAME",
    "ARTIFACT_SCHEMA",
    "DECISION_HZ",
    "MAX_DECISIONS_PER_PHASE",
    "PHASE_IDS",
    "PHYSICS_HZ",
    "PHYSICS_TICKS_PER_DECISION",
    "PhaseTickAudit",
    "PhaseZeroResidualRolloutError",
    "RUN_KIND",
    "SUBCOMMAND",
    "TRAINING_EVIDENCE_SCHEMA",
    "TRAINING_STAGE",
    "ZERO_FULL12",
    "build_contract_binding",
    "run_phase_zero_residual_rollout",
    "validate_phase_zero_residual_rollout_evidence",
    "validate_phase_zero_residual_rollout_payload",
]
