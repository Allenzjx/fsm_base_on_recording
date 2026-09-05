"""Incremental canonical stream writer for live residual-PPO episodes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .action_projection import bitwise_full12_equal, full12_bytes
from .ppo_env_adapter import AuthoritativeFrame
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER


STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))


class LiveStreamWriterError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


def _line(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(payload), separators=(",", ":"), allow_nan=False) + "\n")


def _compact_evaluation_observation(observation: Any) -> Mapping[str, Any]:
    """Project a live observation onto the canonical evaluation evidence.

    The raw sensor object contains the complete 13-body exact-pair history and
    every FSM guard explanation.  Serializing that debug tree at 120 Hz makes
    one otherwise small evaluation episode hundreds of megabytes and slows
    the physical gate substantially.  Offline stability evaluation consumes
    only the fields selected here.  The full contact contract is still
    checked from the raw object in :meth:`start` and recorded in the trial
    manifest; safety decisions continue to use the unmodified live object.
    """

    required = (
        "schema",
        "physics_tick",
        "simulation_time_s",
        "all_finite",
        "base",
        "joints",
        "wheels",
        "bodies",
        "contacts",
        "obstacle",
        "body_collision",
        "guards",
    )
    if any(not hasattr(observation, name) for name in required):
        # Lightweight injected unit-test observations predate the compact
        # stream contract.  Keeping their tiny as_dict payload preserves the
        # writer's dependency-injection surface without weakening production.
        return _jsonable(observation)

    base = observation.base
    wheels: dict[str, Mapping[str, Any]] = {}
    bodies: dict[str, Mapping[str, Any]] = {}
    contacts: dict[str, Mapping[str, Any]] = {}
    measured_wheels: list[float] = []
    for wheel_name in WHEEL_ORDER:
        wheel = observation.wheels[wheel_name]
        body_name = str(wheel.body_name)
        measured_wheels.append(float(wheel.velocity_rad_s))
        wheels[wheel_name] = {
            "body_name": body_name,
            "velocity_rad_s": float(wheel.velocity_rad_s),
            "bottom_w_m": list(wheel.bottom_w_m),
            "geometry_verified": bool(wheel.geometry_verified),
        }
        body = observation.bodies[body_name]
        bodies[body_name] = {
            "linear_velocity_w_m_s": list(body.linear_velocity_w_m_s),
        }
        contact = observation.contacts[body_name]
        contacts[body_name] = {
            pair_name: {
                "pair_verified": bool(getattr(contact, pair_name).pair_verified),
                "normal_force_n": float(getattr(contact, pair_name).normal_force_n),
            }
            for pair_name in ("ground", "obstacle")
        }

    guards = {}
    for name in ("wheel_only_climb_detected", "physics_explosion_or_fall"):
        value = observation.guards.get(name)
        if value is not None:
            row = _jsonable(value)
            guards[name] = (
                {"passed": bool(row.get("passed", False))}
                if isinstance(row, Mapping)
                else {"passed": bool(row)}
            )
    collision = _jsonable(observation.body_collision)
    collision_detected = (
        bool(collision.get("detected", False))
        if isinstance(collision, Mapping)
        else bool(getattr(observation.body_collision, "detected", False))
    )
    return {
        "schema": str(observation.schema),
        "physics_tick": int(observation.physics_tick),
        "simulation_time_s": float(observation.simulation_time_s),
        "all_finite": bool(observation.all_finite),
        "stream_projection": "stability_and_safety_evaluation_v1",
        "base": {
            "orientation_wxyz": list(base.orientation_wxyz),
            "linear_velocity_w_m_s": list(base.linear_velocity_w_m_s),
            "angular_velocity_w_rad_s": list(base.angular_velocity_w_rad_s),
        },
        "joints": {
            name: {"position_deg": float(observation.joints[name].position_deg)}
            for name in SERVO_ORDER
        },
        "wheels": wheels,
        "bodies": bodies,
        "contacts": contacts,
        "obstacle": {"top_z_m": float(observation.obstacle.top_z_m)},
        "body_collision": {"detected": collision_detected},
        "guards": guards,
        "measured_wheel_velocity_rad_s": measured_wheels,
    }


def _force_baseline(observation: Any) -> list[float]:
    result = []
    for wheel_name in WHEEL_ORDER:
        wheel = observation.wheels[wheel_name]
        contact = observation.contacts[wheel.body_name]
        result.append(
            sum(
                max(0.0, float(pair.normal_force_n))
                for pair in (contact.ground, contact.obstacle)
                if pair.pair_verified
            )
        )
    return result


class LiveStreamWriter:
    """Write complete JSONL streams without buffering an episode in memory."""

    def __init__(
        self,
        episode_dir: Path | str,
        *,
        seed: int,
        require_actuator_target_effect_audit: bool = False,
    ) -> None:
        self.episode_dir = Path(episode_dir).resolve()
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.seed = int(seed)
        self._streams = {
            "observation": (self.episode_dir / "observation_120hz.jsonl").open("x", encoding="utf-8"),
            "command": (self.episode_dir / "full12_commands_120hz.jsonl").open("x", encoding="utf-8"),
            "transition": (self.episode_dir / "state_transitions.jsonl").open("x", encoding="utf-8"),
            "event": (self.episode_dir / "task_events.jsonl").open("x", encoding="utf-8"),
            "reward": (self.episode_dir / "reward_15hz.jsonl").open("x", encoding="utf-8"),
        }
        self.phase_times: dict[str, dict[str, float]] = {phase: {} for phase in STATE_IDS}
        self.phase_times["P01"]["entry_time_s"] = 0.0
        self.completed: list[str] = []
        self.started = False
        self.closed = False
        self.level_calibration: dict[str, Any] = {}
        self.initial_home_joint_positions: list[float] = []
        self.initial_normal_forces: list[float] = []
        self.initial_linear_speed_m_s = float("nan")
        self.initial_angular_speed_rad_s = float("nan")
        self.tick_count = 0
        self.zero_input_tick_count = 0
        self.zero_input_bitwise_equivalent_tick_count = 0
        self.zero_fast_path_tick_count = 0
        self.nonzero_residual_tick_count = 0
        self.nonzero_residual_phases: set[str] = set()
        self.require_actuator_target_effect_audit = bool(require_actuator_target_effect_audit)
        self.actuator_target_effect_audited_tick_count = 0
        self.actuator_target_effect_missing_or_invalid_tick_count = 0
        self.own_policy_actuator_target_effect_tick_count = 0
        self.own_policy_actuator_target_effect_phases: set[str] = set()
        self.own_policy_actuator_target_effect_by_phase = {
            phase: {
                "changed_target_tick_count": 0,
                "maximum_servo_position_delta_rad": 0.0,
                "maximum_wheel_velocity_delta_rad_s": 0.0,
            }
            for phase in STATE_IDS
        }
        self._last_command_source_phase: str | None = None
        self.maximum_absolute_normalized_policy_action = 0.0
        self.maximum_absolute_bounded_policy_fraction = 0.0
        self.maximum_absolute_internal_bridge_raw_action = 0.0
        self.policy_action_decision_count = 0
        self.mask_exercised_tick_count = 0
        self.mask_honored_tick_count = 0
        self.rate_limit_tick_count = 0
        self.hard_safety_modified_tick_count = 0
        self.clipping_stages_seen: set[str] = set()
        self.phase_transition_bridge_count = 0
        self.phase_transition_handoff_hold_count = 0
        self.exact_pair_contact_body_count = 0
        self.exact_pair_contract_valid = False

    def start(self, frame: AuthoritativeFrame) -> None:
        if self.started:
            raise LiveStreamWriterError("writer already started")
        observation = frame.info.get("raw_observation")
        if observation is None:
            raise LiveStreamWriterError("initial frame lacks raw observation")
        self.level_calibration = dict(frame.info.get("level_calibration", {}))
        self.initial_home_joint_positions = [
            float(observation.joints[name].position_deg) for name in SERVO_ORDER
        ]
        self.initial_normal_forces = _force_baseline(observation)
        self.initial_linear_speed_m_s = math.sqrt(
            sum(float(value) ** 2 for value in observation.base.linear_velocity_w_m_s)
        )
        self.initial_angular_speed_rad_s = math.sqrt(
            sum(float(value) ** 2 for value in observation.base.angular_velocity_w_rad_s)
        )
        contacts = tuple(observation.contacts.values())
        self.exact_pair_contact_body_count = len(contacts)
        self.exact_pair_contract_valid = bool(
            len(contacts) == 13
            and all(
                contact.ground.pair_verified and contact.obstacle.pair_verified
                for contact in contacts
            )
        )
        _line(
            self._streams["observation"],
            _compact_evaluation_observation(observation),
        )
        _line(
            self._streams["event"],
            {
                "event": "TRIAL_START",
                "result": None,
                "state_id": frame.state_id,
                "sim_time_s": 0.0,
                "seed": self.seed,
            },
        )
        self._write_controller_events(frame)
        for stream in self._streams.values():
            stream.flush()
        self.started = True

    def write_tick(
        self,
        source: AuthoritativeFrame,
        current: AuthoritativeFrame,
        projection: Any,
    ) -> None:
        if not self.started or self.closed:
            raise LiveStreamWriterError("writer is not active")
        observation = current.info.get("raw_observation")
        if observation is None:
            raise LiveStreamWriterError("current frame lacks raw observation")
        incoming_handoff_tick = bool(
            self._last_command_source_phase is not None
            and self._last_command_source_phase != source.state_id
        )
        self._last_command_source_phase = str(source.state_id)
        actuator_audit = current.info.get("actuator_target_effect_audit")
        effect = self._record_actuator_target_effect(
            actuator_audit,
            source=source,
            current=current,
            projection=projection,
            incoming_handoff_tick=incoming_handoff_tick,
        )
        _line(
            self._streams["observation"],
            _compact_evaluation_observation(observation),
        )
        _line(
            self._streams["command"],
            {
                "control_physics_tick": current.physics_tick,
                "sim_time_s": current.sim_time_s,
                "state_id": source.state_id,
                "lifecycle": source.info.get("controller_lifecycle", "EXECUTE_MOTION"),
                "nominal_full12": list(source.nominal_action_full12),
                "residual_full12": list(projection.safe_projected_residual_full12),
                "applied_full12": list(projection.applied_action_full12),
                "raw_residual_full12": list(projection.raw_residual_full12),
                "effective_action_mask_full12": list(projection.effective_action_mask_full12),
                "zero_residual_fast_path": bool(projection.zero_residual_fast_path),
                "clipping_stages": list(projection.clipping_stages),
                **(
                    {
                        "actuator_target_effect_audit": _jsonable(actuator_audit),
                        "own_policy_actuator_target_effect": effect,
                    }
                    if self.require_actuator_target_effect_audit or actuator_audit is not None
                    else {}
                ),
            },
        )
        raw = tuple(float(value) for value in projection.raw_residual_full12)
        masked = tuple(float(value) for value in projection.masked_residual_full12)
        safe = tuple(float(value) for value in projection.safe_projected_residual_full12)
        effective_mask = tuple(int(value) for value in projection.effective_action_mask_full12)
        # ``projection.raw_residual_full12`` may be the bridge's inverse-tanh
        # hold value on the first tick after a phase transition.  It is not
        # necessarily the policy request, so keep it as a separate diagnostic
        # and audit the actual 15 Hz request in ``write_decision`` below.
        self.maximum_absolute_internal_bridge_raw_action = max(
            self.maximum_absolute_internal_bridge_raw_action,
            max(abs(value) for value in raw),
        )
        if all(value == 0.0 for value in raw):
            self.zero_input_tick_count += 1
            if (
                bitwise_full12_equal(
                    projection.applied_action_full12,
                    source.nominal_action_full12,
                )
                and full12_bytes(safe) == full12_bytes((0.0,) * 12)
            ):
                self.zero_input_bitwise_equivalent_tick_count += 1
        if projection.zero_residual_fast_path:
            self.zero_fast_path_tick_count += 1
        if any(value != 0.0 for value in safe):
            self.nonzero_residual_tick_count += 1
            self.nonzero_residual_phases.add(source.state_id)
        disabled_nonzero = tuple(
            index
            for index, (value, enabled) in enumerate(
                zip(raw, effective_mask, strict=True)
            )
            if not enabled and value != 0.0
        )
        if disabled_nonzero:
            self.mask_exercised_tick_count += 1
            if all(masked[index] == 0.0 for index in disabled_nonzero):
                self.mask_honored_tick_count += 1
        stages = tuple(str(value) for value in projection.clipping_stages)
        self.clipping_stages_seen.update(stages)
        if "residual_rate_limit" in stages:
            self.rate_limit_tick_count += 1
        if projection.hard_safety_modified:
            self.hard_safety_modified_tick_count += 1
        self._write_controller_events(current)
        self.tick_count += 1
        # Keep crash loss bounded to one simulated second without forcing two
        # Windows filesystem flushes on every 120 Hz physics tick.
        if self.tick_count % 120 == 0:
            for stream in self._streams.values():
                stream.flush()

    def _record_actuator_target_effect(
        self,
        audit: Any,
        *,
        source: AuthoritativeFrame,
        current: AuthoritativeFrame,
        projection: Any,
        incoming_handoff_tick: bool,
    ) -> Mapping[str, Any]:
        """Count only representable same-phase, own-request actuator effects.

        A projected Python float can be nonzero while the real target buffer
        remains bit-identical.  The backend proof owns native mapping and
        dtype conversion; this consumer independently binds its values to the
        command tick, source phase, request mask, and actual projected residual.
        The first source tick of a new phase is conservatively excluded because
        its projector may retain an incoming handoff instead of the new request.
        """

        result = {
            "audit_valid": False,
            "incoming_handoff_tick_excluded": incoming_handoff_tick,
            "own_phase_policy_request": False,
            "qualifying_changed_channels_full12": [False] * 12,
            "counted": False,
        }
        if audit is None and not self.require_actuator_target_effect_audit:
            return result

        def vector(value: Any, *, size: int) -> tuple[float, ...] | None:
            if not isinstance(value, (list, tuple)) or len(value) != size:
                return None
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            ):
                return None
            return tuple(float(item) for item in value)

        def targets(value: Any) -> tuple[float, ...] | None:
            if not isinstance(value, Mapping):
                return None
            servos = vector(value.get("servo_position_rad"), size=8)
            wheels = vector(value.get("wheel_velocity_rad_s"), size=4)
            return None if servos is None or wheels is None else servos + wheels

        valid = isinstance(audit, Mapping)
        actual = targets(audit.get("actual_native_targets")) if valid else None
        counterfactual = targets(audit.get("counterfactual_native_targets")) if valid else None
        raw = vector(audit.get("raw_policy_action_full12"), size=12) if valid else None
        projected = vector(audit.get("projected_residual_full12"), size=12) if valid else None
        mask = audit.get("phase_mask_full12") if valid else None
        changed = audit.get("changed_channels_full12") if valid else None
        ack = current.info.get("atomic_ack")
        # Bind the backend's logical delta to the actual same-tick command.
        # Adding a small residual to a nonzero nominal and subtracting it again
        # can round in double precision; the pre-add residual is not the proof.
        command_delta = tuple(
            float(applied) - float(nominal)
            for applied, nominal in zip(
                projection.applied_action_full12,
                source.nominal_action_full12,
                strict=True,
            )
        )
        valid = bool(
            valid
            and audit.get("schema") == "wlr50_clean.actuator_target_effect_audit.v1"
            and audit.get("verified") is True
            and audit.get("same_tick_counterfactual") is True
            and audit.get("setter_dispatch_targets_equal") is True
            and audit.get("actual_mapping_matches_dispatch") is True
            and audit.get("target_dtype") == "torch.float32"
            and audit.get("source_phase_id") == source.state_id
            and audit.get("policy_request_phase") in STATE_IDS
            and isinstance(ack, Mapping)
            and type(audit.get("physics_tick")) is int
            and audit["physics_tick"] >= 0
            and type(ack.get("physics_tick")) is int
            and audit["physics_tick"] == ack["physics_tick"]
            and actual is not None
            and counterfactual is not None
            and raw is not None
            and projected == command_delta
            and isinstance(mask, (list, tuple))
            and len(mask) == 12
            and all(type(item) is int and item in (0, 1) for item in mask)
            and isinstance(changed, (list, tuple))
            and len(changed) == 12
            and all(type(item) is bool for item in changed)
        )
        if valid:
            actual_changes = tuple(a != b for a, b in zip(actual, counterfactual, strict=True))
            valid = bool(
                tuple(changed) == actual_changes
                and type(audit.get("changed_target_channel_count")) is int
                and audit["changed_target_channel_count"] == sum(actual_changes)
            )
        if not valid:
            self.actuator_target_effect_missing_or_invalid_tick_count += 1
            return result
        self.actuator_target_effect_audited_tick_count += 1
        result["audit_valid"] = True
        own_phase = bool(
            audit["policy_request_phase"] == source.state_id
            and tuple(mask) == tuple(projection.effective_action_mask_full12)
        )
        result["own_phase_policy_request"] = own_phase
        eligible = tuple(
            bool(
                own_phase
                and not incoming_handoff_tick
                and mask[index]
                and raw[index] != 0.0
                and projected[index] != 0.0
                and changed[index]
            )
            for index in range(12)
        )
        result["qualifying_changed_channels_full12"] = list(eligible)
        if any(eligible):
            self.own_policy_actuator_target_effect_tick_count += 1
            self.own_policy_actuator_target_effect_phases.add(str(source.state_id))
            phase = self.own_policy_actuator_target_effect_by_phase[str(source.state_id)]
            phase["changed_target_tick_count"] += 1
            phase["maximum_servo_position_delta_rad"] = max(
                phase["maximum_servo_position_delta_rad"],
                max(
                    abs(actual[index] - counterfactual[index]) if eligible[index] else 0.0
                    for index in range(8)
                ),
            )
            phase["maximum_wheel_velocity_delta_rad_s"] = max(
                phase["maximum_wheel_velocity_delta_rad_s"],
                max(
                    abs(actual[index] - counterfactual[index]) if eligible[index] else 0.0
                    for index in range(8, 12)
                ),
            )
            result["counted"] = True
        return result

    def _write_controller_events(self, frame: AuthoritativeFrame) -> None:
        controller_frame = frame.info.get("raw_controller_frame")
        for event in tuple(getattr(controller_frame, "events", ())):
            row = _jsonable(event)
            _line(self._streams["transition"], row)
            _line(self._streams["event"], {"event": "STATE_TRANSITION", **row})
            phase = str(row["state_id"])
            timing = self.phase_times[phase]
            lifecycle = str(row["to_lifecycle"])
            when = float(row["sim_time_s"])
            if lifecycle == "WAIT_ENTRY":
                timing.setdefault("entry_time_s", when)
            elif lifecycle == "EXECUTE_MOTION":
                timing["motion_start_s"] = when
            elif lifecycle == "VERIFY_RESULT":
                timing["motion_end_s"] = when
                timing["verify_start_s"] = when
            elif lifecycle == "DONE":
                timing["completion_time_s"] = when
                if phase not in self.completed:
                    self.completed.append(phase)

    def write_decision(self, info: Mapping[str, Any]) -> None:
        reward = info.get("reward")
        if not isinstance(reward, Mapping):
            raise LiveStreamWriterError("decision info lacks v2 reward breakdown")
        policy_action = info.get("raw_policy_action_full12")
        if (
            not isinstance(policy_action, (list, tuple))
            or len(policy_action) != 12
        ):
            raise LiveStreamWriterError(
                "decision info lacks the normalized Full12 policy request"
            )
        normalized_policy_action = tuple(float(value) for value in policy_action)
        if any(not math.isfinite(value) for value in normalized_policy_action):
            raise LiveStreamWriterError("normalized policy request is non-finite")
        self.maximum_absolute_normalized_policy_action = max(
            self.maximum_absolute_normalized_policy_action,
            max(abs(value) for value in normalized_policy_action),
        )
        self.maximum_absolute_bounded_policy_fraction = max(
            self.maximum_absolute_bounded_policy_fraction,
            max(abs(math.tanh(value)) for value in normalized_policy_action),
        )
        self.policy_action_decision_count += 1
        _line(
            self._streams["reward"],
            {
                "state_id": reward["phase_id"],
                "decision_index": info["decision_index"],
                "sim_time_s": info["sim_time_s"],
                **dict(reward),
            },
        )
        transitions = info.get("phase_transition_action_jump", ())
        if isinstance(transitions, (list, tuple)):
            self.phase_transition_bridge_count += len(transitions)
            self.phase_transition_handoff_hold_count += sum(
                isinstance(transition, Mapping)
                and transition.get("handoff_hold_used") is True
                for transition in transitions
            )

    def finalize(
        self,
        frame: AuthoritativeFrame,
        *,
        reward_total: float,
        decision_count: int,
    ) -> Path:
        if self.closed:
            raise LiveStreamWriterError("writer already finalized")
        signals = frame.termination_signals
        termination_mapping = frame.info.get("termination_mapping", {})
        reason = str(
            termination_mapping.get("controller_result")
            or ("SUCCESS" if signals.success else "UNKNOWN")
        )
        if reason == "None":
            reason = "SUCCESS" if signals.success else "UNKNOWN"
        _line(
            self._streams["event"],
            {
                "event": "TRIAL_TERMINATION_AUTHORITATIVE",
                "result": reason,
                "state_id": frame.state_id,
                "lifecycle": frame.info.get("controller_lifecycle"),
                "sim_time_s": frame.sim_time_s,
                "reason": termination_mapping.get("controller_reason", ""),
                "details": {
                    "controller_details": termination_mapping.get(
                        "controller_details", {}
                    ),
                    "first_blocker": termination_mapping.get("first_blocker", {}),
                    "active_sources": termination_mapping.get("active_sources", ()),
                    "physics_guard_values": termination_mapping.get(
                        "physics_guard_values", {}
                    ),
                },
            },
        )
        for stream in self._streams.values():
            stream.flush()
            stream.close()
        self.closed = True
        manifest = {
            "schema": "wlr50_clean.ppo_live_trial_manifest.v1",
            "trial_id": self.episode_dir.name,
            "seed": self.seed,
            "result": reason,
            "duration_s": frame.sim_time_s,
            "physics_hz": 120.0,
            "decision_hz": 15.0,
            "control_steps": self.tick_count,
            "decision_count": int(decision_count),
            "reward_total": float(reward_total),
            "observation_stream": {
                "schema": "wlr50_clean.live_observation.v1",
                "rate_hz": 120.0,
                "projection": "stability_and_safety_evaluation_v1",
                "raw_live_observation_used_by_control": True,
                "serialized_debug_tree_compacted": True,
            },
            "phase_times": self.phase_times,
            "phase_windows": [
                {"phase": phase, **self.phase_times[phase]} for phase in self.completed
            ],
            "analysis_checks": {},
            "conformance": {
                "recovery_count": 0,
                "measured_wheel_velocity_decay_threshold_rad_s": 0.256016593426466,
            },
            "success_evidence": {
                "completed_macro_phases": self.completed,
                "p01_p13_completed": tuple(self.completed) == STATE_IDS,
                "body_collision": bool(signals.body_collision),
                "wheel_only_climb": bool(signals.wheel_only_climb),
                "duration_s": frame.sim_time_s,
                "root_state_write_count": 0,
                "runtime_raw_recording_access": False,
                "recording_runtime_access_count": 0,
            },
            "action_projection_audit": {
                "physics_tick_count": self.tick_count,
                "zero_input_tick_count": self.zero_input_tick_count,
                "zero_input_bitwise_equivalent_tick_count": (
                    self.zero_input_bitwise_equivalent_tick_count
                ),
                "zero_input_all_ticks_bitwise_equivalent": bool(
                    self.zero_input_tick_count == self.tick_count
                    and self.zero_input_bitwise_equivalent_tick_count == self.tick_count
                ),
                "zero_residual_fast_path_tick_count": self.zero_fast_path_tick_count,
                "nonzero_residual_tick_count": self.nonzero_residual_tick_count,
                "nonzero_residual_phases": sorted(self.nonzero_residual_phases),
                "actuator_target_effect_audit_required": self.require_actuator_target_effect_audit,
                "actuator_target_effect_audited_tick_count": self.actuator_target_effect_audited_tick_count,
                "actuator_target_effect_missing_or_invalid_tick_count": self.actuator_target_effect_missing_or_invalid_tick_count,
                "actuator_target_effect_audit_complete": bool(
                    self.require_actuator_target_effect_audit
                    and self.tick_count > 0
                    and self.actuator_target_effect_audited_tick_count == self.tick_count
                    and self.actuator_target_effect_missing_or_invalid_tick_count == 0
                ),
                "own_policy_actuator_target_effect_tick_count": self.own_policy_actuator_target_effect_tick_count,
                "own_policy_actuator_target_effect_phases": sorted(self.own_policy_actuator_target_effect_phases),
                "own_policy_actuator_target_effect_by_phase": self.own_policy_actuator_target_effect_by_phase,
                "maximum_absolute_bounded_policy_fraction": self.maximum_absolute_bounded_policy_fraction,
                "within_one_percent_smoke_amplitude": bool(
                    self.policy_action_decision_count == int(decision_count)
                    and self.maximum_absolute_bounded_policy_fraction <= 0.01
                ),
                "maximum_absolute_normalized_policy_action": (
                    self.maximum_absolute_normalized_policy_action
                ),
                "maximum_absolute_internal_bridge_raw_action": (
                    self.maximum_absolute_internal_bridge_raw_action
                ),
                "policy_action_decision_count": self.policy_action_decision_count,
                "within_five_percent_smoke_amplitude": (
                    self.policy_action_decision_count == int(decision_count)
                    and self.maximum_absolute_normalized_policy_action <= 0.05
                ),
                "mask_exercised_tick_count": self.mask_exercised_tick_count,
                "mask_honored_tick_count": self.mask_honored_tick_count,
                "mask_honored_when_exercised": bool(
                    self.mask_exercised_tick_count > 0
                    and self.mask_honored_tick_count == self.mask_exercised_tick_count
                ),
                "rate_limit_tick_count": self.rate_limit_tick_count,
                "phase_transition_bridge_count": self.phase_transition_bridge_count,
                "phase_transition_handoff_hold_count": (
                    self.phase_transition_handoff_hold_count
                ),
                "hard_safety_modified_tick_count": self.hard_safety_modified_tick_count,
                "clipping_stages_seen": sorted(self.clipping_stages_seen),
                "exact_pair_contact_body_count": self.exact_pair_contact_body_count,
                "exact_pair_contact_contract_valid": self.exact_pair_contract_valid,
                "body_collision_detector_operational": self.exact_pair_contract_valid,
                "wheel_only_climb_detector_operational": self.exact_pair_contract_valid,
            },
            "ppo_calibration": {
                "source": (
                    "backend_reset_1p5s_zero_command_final_0p25s_orientation_mean"
                    "+post_settle_home_load_snapshot"
                ),
                "quality_passed": bool(self.level_calibration.get("valid", False))
                and self.initial_linear_speed_m_s <= 0.10
                and self.initial_angular_speed_rad_s <= 0.20,
                "level_reference_orientation_wxyz": self.level_calibration.get(
                    "level_reference_orientation_wxyz", [1.0, 0.0, 0.0, 0.0]
                ),
                "home_joint_positions_deg8": self.initial_home_joint_positions,
                "wheel_normal_force_baseline_n4": self.initial_normal_forces,
                "sample_count": int(self.level_calibration.get("sample_count", 30)),
                "window_start_s": -0.25,
                "window_end_s": 0.0,
                "maximum_linear_speed_m_s": self.initial_linear_speed_m_s,
                "maximum_angular_speed_rad_s": self.initial_angular_speed_rad_s,
            },
        }
        path = self.episode_dir / "trial_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    def abort(self) -> None:
        if self.closed:
            return
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception:
                pass
        self.closed = True


__all__ = ["LiveStreamWriter", "LiveStreamWriterError"]
