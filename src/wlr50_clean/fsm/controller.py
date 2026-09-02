"""Sensor-driven P01--P13 controller with 120/15 Hz clock separation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.reference.motion_contract import MotionContract, load_motion_contract

from .drive_feedback import ReferenceBoundedDriveFeedback
from .guard_evaluator import (
    GuardEvidence,
    GuardEvaluator,
    GuardReport,
    observation_progress_vector,
)
from .motion_executor import (
    FeedbackCorrection,
    MAX_CORRECTION_FRACTION,
    MotionExecutor,
    MotionTick,
    ProgressWatchdog,
    WatchdogBlocker,
)
from .recovery import RecoveryPlanner
from .state_graph import StateGraph
from .state_spec import FsmSpec, Lifecycle, StateSpec, load_fsm_spec
from .task_result import TaskResult, TaskTermination


@dataclass(frozen=True)
class ControllerEvent:
    sim_time_s: float
    state_id: str
    from_lifecycle: str
    to_lifecycle: str
    reason: str
    details: Mapping[str, Any]


@dataclass(frozen=True)
class ControllerFrame:
    physics_tick: int
    sim_time_s: float
    state_id: str
    lifecycle: Lifecycle
    full12: tuple[float, ...]
    decision_tick: bool
    full12_atomic_write_required: bool
    atomic_source_event: bool
    tracking_servo_names: tuple[str, ...]
    drive_feedback_bias_full12: tuple[float, ...]
    drive_feedback_details: Mapping[str, Any]
    endpoint_issued: bool
    termination: TaskTermination | None
    first_blocker: Mapping[str, Any] | None
    events: tuple[ControllerEvent, ...]


class SensorFsmController:
    """One call per physics tick; always returns one atomic full12 action."""

    def __init__(self, spec: FsmSpec, contract: MotionContract) -> None:
        _validate_pair(spec, contract)
        self.spec = spec
        self.contract = contract
        self.graph = StateGraph(spec)
        self.guard_evaluator = GuardEvaluator()
        self.motion = MotionExecutor(
            physics_hz=contract.physics_hz,
            servo_rate_limit_deg_s=contract.servo_rate_limit_deg_s,
            initial_full12=contract.phases[0].start_full12,
        )
        self.recovery = RecoveryPlanner(contract.full12_order)
        self.drive_feedback = ReferenceBoundedDriveFeedback()
        self.watchdog = ProgressWatchdog(spec.watchdog_s)
        self.state = self.graph.first
        self.lifecycle = Lifecycle.WAIT_ENTRY
        self.retries_used = 0
        self.physics_tick = 0
        self._last_sim_time_s: float | None = None
        self._verify_started_s: float | None = None
        self._wait_entry_started_s: float | None = None
        self._endpoint_issued = False
        self._previous_state_done = True
        self._pending_blocker: GuardEvidence | WatchdogBlocker | None = None
        self._first_blocker: dict[str, Any] | None = None
        self._tracking_servo_names: tuple[str, ...] = ()
        self._drive_feedback_tick_index: int | None = None
        self._decision_lattice_origin_tick = 0
        self.termination: TaskTermination | None = None
        self.history: list[ControllerEvent] = []

    @classmethod
    def from_paths(cls, fsm_path: Path, motion_contract_path: Path) -> "SensorFsmController":
        return cls(load_fsm_spec(fsm_path), load_motion_contract(motion_contract_path))

    @property
    def phase(self):
        return self.contract.phase(self.state.state_id)

    @property
    def first_blocker(self) -> Mapping[str, Any] | None:
        return self._first_blocker

    def step(self, observation: Any, *, sim_time_s: float | None = None) -> ControllerFrame:
        now = (
            self.physics_tick / self.spec.motion_hz
            if sim_time_s is None
            else float(sim_time_s)
        )
        events: list[ControllerEvent] = []
        decision_tick = self._decision_due()
        if self._last_sim_time_s is not None and now + 1e-12 < self._last_sim_time_s:
            self._terminate(
                TaskResult.INFRASTRUCTURE_ERROR,
                now,
                "simulation time moved backwards",
                {"previous_sim_time_s": self._last_sim_time_s},
            )
        self._last_sim_time_s = now

        if self.termination is None and self.lifecycle is Lifecycle.VERIFY_RESULT:
            self.guard_evaluator.observe_continuous_completion_guards(
                self.state.completion_guards,
                observation,
                state_id=self.state.state_id,
                sim_time_s=now,
            )

        if self.termination is None and decision_tick:
            self._run_decision_transitions(observation, now, events)

        motion_tick: MotionTick | None = None
        drive_feedback = self.drive_feedback.update(
            state_id=self.state.state_id,
            motion_tick_index=None,
            actual_full12=_actual_full12(observation),
            spec=None,
        )
        if self.termination is None and self.lifecycle is Lifecycle.EXECUTE_MOTION:
            motion_tick = self.motion.tick()
            command = motion_tick.full12
            self._drive_feedback_tick_index = motion_tick.tick_index
            drive_feedback = self.drive_feedback.update(
                state_id=self.state.state_id,
                motion_tick_index=motion_tick.tick_index,
                actual_full12=_actual_full12(observation),
                spec=self.phase.drive_feedback,
            )
            self._tracking_servo_names = motion_tick.tracking_servo_names
            if motion_tick.endpoint_issued:
                self._endpoint_issued = True
                self._verify_started_s = now
                self._transition(
                    Lifecycle.VERIFY_RESULT,
                    now,
                    "compact motion endpoint issued; awaiting live result guards",
                    events,
                    {"motion_elapsed_s": motion_tick.elapsed_s},
                )
            else:
                blocker = self.watchdog.update(
                    sim_time_s=now,
                    state_id=self.state.state_id,
                    lifecycle=self.lifecycle.value,
                    target_full12=command,
                    actual_progress=observation_progress_vector(observation),
                )
                if blocker is not None:
                    self._remember_blocker(blocker)
                    self._recover_or_terminate(blocker, now, events)
        else:
            if (
                self.termination is None
                and self.lifecycle is Lifecycle.VERIFY_RESULT
                and self.phase.drive_feedback is not None
                and self._drive_feedback_tick_index is not None
            ):
                self._drive_feedback_tick_index += 1
                drive_feedback = self.drive_feedback.update(
                    state_id=self.state.state_id,
                    motion_tick_index=self._drive_feedback_tick_index,
                    actual_full12=_actual_full12(observation),
                    spec=self.phase.drive_feedback,
                )
            command = self._held_or_safe_command()
            if (
                self.termination is None
                and self.lifecycle is Lifecycle.WAIT_ENTRY
            ):
                if self._wait_entry_started_s is None:
                    self._wait_entry_started_s = now
                blocker = self.watchdog.update(
                    sim_time_s=now,
                    state_id=self.state.state_id,
                    lifecycle=self.lifecycle.value,
                    target_full12=command,
                    actual_progress=observation_progress_vector(observation),
                )
                if blocker is not None:
                    self._remember_blocker(self._pending_blocker or blocker)
                    self._terminate_blocked(blocker, now)
                elif (
                    self._pending_blocker is not None
                    and now - self._wait_entry_started_s + 1e-12
                    >= self.spec.watchdog_s
                ):
                    # Allow only the explicit short carry-over budget.  Live
                    # jitter must not keep an incompatible entry alive until
                    # the trial-wide timeout.
                    self._remember_blocker(self._pending_blocker)
                    self._terminate_blocked(self._pending_blocker, now)

        tracking_servo_names = (
            () if self.termination is not None else self._tracking_servo_names
        )

        frame = ControllerFrame(
            physics_tick=self.physics_tick,
            sim_time_s=now,
            state_id=self.state.state_id,
            lifecycle=self.lifecycle,
            full12=command,
            decision_tick=decision_tick,
            full12_atomic_write_required=True,
            atomic_source_event=bool(
                motion_tick is not None and motion_tick.source_full12_atomic
            ),
            tracking_servo_names=tracking_servo_names,
            drive_feedback_bias_full12=drive_feedback.bias_full12,
            drive_feedback_details=drive_feedback.as_dict(),
            endpoint_issued=self._endpoint_issued,
            termination=self.termination,
            first_blocker=self._first_blocker,
            events=tuple(events),
        )
        self.physics_tick += 1
        return frame

    def _run_decision_transitions(
        self, observation: Any, now: float, events: list[ControllerEvent]
    ) -> None:
        # DONE and a ready next-state entry may collapse into the same decision
        # tick.  Both lifecycle transitions remain explicit in the event log,
        # while no artificial 15 Hz pause is injected into the 120 Hz motion.
        for _ in range(4):
            # A velocity-aligned entry may move the 15 Hz decision lattice
            # after the preceding phase completes.  Re-check here because
            # DONE -> WAIT_ENTRY can otherwise collapse into the old lattice's
            # decision and consume the live carry-over sample too early.
            if self.lifecycle is Lifecycle.WAIT_ENTRY and not self._decision_due():
                return
            # Task failures and safety aborts are lifecycle-independent.  A
            # collision that first becomes persistent on the endpoint tick,
            # for example, must not be downgraded to a VERIFY/next-entry
            # blocker merely because EXECUTE_MOTION has just ended.
            abort = self.guard_evaluator.first_hard_abort(
                self.state.hard_abort_guards,
                observation,
                state_id=self.state.state_id,
                sim_time_s=now,
                local_facts=self._local_facts(observation),
            )
            if abort is not None:
                self._remember_blocker(abort.evidence)
                self._terminate(
                    abort.result,
                    now,
                    f"hard guard asserted: {abort.evidence.name}",
                    {"evidence": asdict(abort.evidence)},
                )
                return

            if self.lifecycle is Lifecycle.EXECUTE_MOTION:
                return

            if self.lifecycle is Lifecycle.WAIT_ENTRY:
                report = self.guard_evaluator.evaluate_all(
                    self.state.entry_guards,
                    observation,
                    state_id=self.state.state_id,
                    sim_time_s=now,
                    local_facts=self._local_facts(observation),
                )
                if not report.passed:
                    # Sensor startup/settling latency is not a trial blocker.
                    # Latch this only if the no-progress watchdog actually fires.
                    self._pending_blocker = report.first_blocker
                    return
                self._pending_blocker = None
                self._wait_entry_started_s = None
                correction = FeedbackCorrection(
                    self.state.normal_correction_fractions
                )
                self.motion.start_phase(self.phase, correction)
                self.watchdog.reset()
                self._endpoint_issued = False
                details = dict(_report_details(report))
                details["correction_fractions"] = correction.fractions
                self._transition(
                    Lifecycle.EXECUTE_MOTION,
                    now,
                    "all live entry guards passed",
                    events,
                    details,
                )
                # Safety predicates are checked now that EXECUTE is active.
                continue

            if self.lifecycle is Lifecycle.VERIFY_RESULT:
                report = self.guard_evaluator.evaluate_all(
                    self.state.completion_guards,
                    observation,
                    state_id=self.state.state_id,
                    sim_time_s=now,
                    local_facts=self._local_facts(observation),
                )
                if report.passed:
                    self._transition(
                        Lifecycle.DONE,
                        now,
                        self.state.transition_reason,
                        events,
                        _report_details(report),
                    )
                    next_state = self.graph.next(self.state.state_id)
                    if next_state is None:
                        self._terminate(
                            TaskResult.SUCCESS,
                            now,
                            "P13 live completion guards passed",
                            {"completion_event": self.state.completion_event},
                        )
                        return
                    self._enter_next_state(next_state, now, events)
                    continue
                self._pending_blocker = report.first_blocker
                verify_start = (
                    now if self._verify_started_s is None else self._verify_started_s
                )
                verify_elapsed = now - verify_start
                verify_wait_s = (
                    self.state.recovery_max_verify_wait_s
                    if self.retries_used > 0
                    else self.state.max_verify_wait_s
                )
                if verify_elapsed + 1e-12 >= verify_wait_s:
                    self._recover_or_terminate(report.first_blocker, now, events)
                return

            if self.lifecycle is Lifecycle.RECOVERY:
                if self.retries_used >= self.state.retry_budget:
                    self._terminate_blocked(self._pending_blocker, now)
                    return
                blocker_name = _blocker_name(self._pending_blocker)
                try:
                    plan = self.recovery.plan(
                        phase=self.phase,
                        observation=observation,
                        blocked_guard=blocker_name,
                        blocker_evidence=self._pending_blocker,
                    )
                    cumulative = tuple(
                        abs(normal) + abs(recovery)
                        for normal, recovery in zip(
                            self.state.normal_correction_fractions,
                            plan.correction.fractions,
                            strict=True,
                        )
                    )
                    if any(
                        value > MAX_CORRECTION_FRACTION + 1e-12
                        for value in cumulative
                    ):
                        raise ValueError(
                            "normal plus retry correction exceeds the cumulative "
                            "15% reference bound"
                        )
                    retry_correction = plan.correction
                except (TypeError, ValueError) as exc:
                    self._terminate(
                        TaskResult.INFRASTRUCTURE_ERROR,
                        now,
                        "invalid recovery feedback interface",
                        {"error": str(exc), "blocked_guard": blocker_name},
                    )
                    return
                self.retries_used += 1
                # A retry is a distinct, bounded correction attempt.  Reapplying
                # the state's nominal shaping here would silently spend that
                # fraction twice while the transition ledger records one retry.
                terminal_pose_retry = (
                    self.state.state_id == "P13"
                    and blocker_name == "final_joint_pose_compatible"
                )
                if terminal_pose_retry:
                    # P13 has already established final top geometry and zero
                    # wheel targets.  Correct only its held endpoint; replaying
                    # the 17.8 s advance would duplicate wheel travel and change
                    # the support state that produced the live pose evidence.
                    self.motion.start_phase_at_endpoint(self.phase, retry_correction)
                else:
                    self.motion.start_phase(self.phase, retry_correction)
                self.guard_evaluator.reset_state(self.state.state_id)
                self.watchdog.reset()
                self._endpoint_issued = False
                self._verify_started_s = None
                self._transition(
                    Lifecycle.EXECUTE_MOTION,
                    now,
                    plan.reason,
                    events,
                    {
                        "retry": self.retries_used,
                        "blocked_guard": blocker_name,
                        "correction_fractions": retry_correction.fractions,
                        "recovery_correction_fractions": retry_correction.fractions,
                        "recovery_motion": (
                            "terminal_endpoint_hold"
                            if terminal_pose_retry
                            else "full_phase_retry"
                        ),
                    },
                )
                continue

            return

    def _enter_next_state(
        self, next_state: StateSpec, now: float, events: list[ControllerEvent]
    ) -> None:
        previous = self.state.state_id
        self.graph.validate_transition(previous, next_state.state_id)
        self.state = next_state
        self.lifecycle = Lifecycle.WAIT_ENTRY
        alignment = self.phase.entry_velocity_alignment
        if alignment is not None:
            self._decision_lattice_origin_tick = (
                self.physics_tick + alignment.first_decision_delay_ticks
            )
        self.retries_used = 0
        self._endpoint_issued = False
        self._verify_started_s = None
        self._previous_state_done = True
        self._pending_blocker = None
        self._wait_entry_started_s = now
        # The completed phase's final servo segment remains active through its
        # bounded VERIFY tail, then freezes exactly at the next phase boundary.
        # A blocked next entry must not keep accumulating the prior phase's
        # tracking compensation.
        self._tracking_servo_names = ()
        self._drive_feedback_tick_index = None
        self.guard_evaluator.reset_state(next_state.state_id)
        self.watchdog.reset()
        event = ControllerEvent(
            sim_time_s=now,
            state_id=next_state.state_id,
            from_lifecycle=Lifecycle.DONE.value,
            to_lifecycle=Lifecycle.WAIT_ENTRY.value,
            reason=f"advance fixed graph {previous}->{next_state.state_id}",
            details={},
        )
        events.append(event)
        self.history.append(event)

    def _decision_due(self) -> bool:
        origin = self._decision_lattice_origin_tick
        return (
            self.physics_tick >= origin
            and (self.physics_tick - origin) % self.spec.decision_stride == 0
        )

    def _recover_or_terminate(
        self,
        blocker: GuardEvidence | WatchdogBlocker | None,
        now: float,
        events: list[ControllerEvent],
    ) -> None:
        self._pending_blocker = blocker
        self._remember_blocker(blocker)
        if self.retries_used >= self.state.retry_budget:
            self._terminate_blocked(blocker, now)
            return
        self._transition(
            Lifecycle.RECOVERY,
            now,
            "live result/progress blocked; enter bounded recovery",
            events,
            _blocker_details(blocker),
        )

    def _terminate_blocked(
        self, blocker: GuardEvidence | WatchdogBlocker | None, now: float
    ) -> None:
        if self.lifecycle is Lifecycle.WAIT_ENTRY and self.retries_used == 0:
            reason = (
                "live entry guards remained incompatible after the bounded wait "
                "window; no recovery retry was attempted"
            )
        else:
            reason = "controller exhausted its one reference-bounded retry"
        self._terminate(
            TaskResult.INCOMPLETE_CONTROLLER_BLOCKED,
            now,
            reason,
            {
                "current_blocker": _blocker_details(blocker),
                "first_blocker": self._first_blocker,
                "retries_used": self.retries_used,
            },
        )

    def _local_facts(self, observation: Any) -> dict[str, Any]:
        held = self.motion.held_full12(self.phase.start_full12)
        return {
            "previous_state_done": self._previous_state_done,
            "motion_endpoint_issued": self._endpoint_issued,
            "wheel_targets_zero": all(abs(value) <= 1e-9 for value in held[8:]),
            "reference_entry_compatible": self._entry_compatibility(observation),
            "final_joint_pose_compatible": self._final_pose_compatibility(observation),
            "drive_feedback_cycle_complete": self._drive_feedback_cycle_complete(),
        }

    def _drive_feedback_cycle_complete(self) -> GuardEvidence:
        feedback = self.phase.drive_feedback
        triggered = self.drive_feedback.trigger_latched
        tick = self._drive_feedback_tick_index
        passed = bool(
            feedback is None
            or not triggered
            or (tick is not None and tick >= feedback.last_bias_tick)
        )
        return GuardEvidence(
            "drive_feedback_cycle_complete",
            passed,
            {
                "triggered": triggered,
                "current_tick": tick,
                "required_last_bias_tick": (
                    None if feedback is None else feedback.last_bias_tick
                ),
            },
            "controller.live_drive_feedback",
            "a live-triggered bounded drive-feedback correction must finish before P09 can advance",
        )

    def _entry_compatibility(self, observation: Any) -> GuardEvidence:
        actual = _actual_servo_positions(observation, self.contract.full12_order[:8])
        velocity = _actual_servo_velocities(
            observation, self.contract.full12_order[:8]
        )
        active = set(self.phase.active_channels)
        checked = [
            index
            for index, channel in enumerate(self.contract.full12_order[:8])
            if channel in active
        ]
        if not checked:
            return GuardEvidence(
                "reference_entry_compatible",
                True,
                {"checked_servo_channels": ()},
                "controller.phase_context",
                "phase has no active servo entry constraint",
            )
        if actual is None:
            return GuardEvidence(
                "reference_entry_compatible",
                False,
                source="controller.phase_context",
                reason="live servo readback unavailable",
            )
        errors = {}
        passed = True
        for index in checked:
            channel = self.contract.full12_order[index]
            reference_actual_start = self.state.reference_actual_start_full12[index]
            error = actual[index] - reference_actual_start
            limit = max(2.0, 0.15 * abs(self.phase.delta_full12[index]))
            errors[channel] = {
                "actual_deg": actual[index],
                "reference_actual_start_deg": reference_actual_start,
                "error_deg": error,
                "limit_deg": limit,
            }
            passed = passed and abs(error) <= limit + 1e-12
        alignment = self.phase.entry_velocity_alignment
        if alignment is not None:
            reference_velocity = alignment.reference_velocity_deg_s
            velocity_limit = alignment.relative_limit * abs(reference_velocity)
            actual_velocity = (
                None if velocity is None else velocity[alignment.channel_index]
            )
            velocity_error = (
                None
                if actual_velocity is None
                else actual_velocity - reference_velocity
            )
            velocity_passed = bool(
                velocity_error is not None
                and actual_velocity is not None
                and actual_velocity > 0.0
                and abs(velocity_error) <= velocity_limit + 1.0e-12
            )
            errors[f"{alignment.channel}_velocity"] = {
                "actual_deg_s": actual_velocity,
                "reference_deg_s": reference_velocity,
                "error_deg_s": velocity_error,
                "limit_deg_s": velocity_limit,
                "signed_positive_rebound_required": True,
            }
            passed = passed and velocity_passed
        return GuardEvidence(
            "reference_entry_compatible",
            passed,
            errors,
            "controller.live_servo_vs_v010_actual_start",
            "active servos compare with measured v010 entry position; any authored signed entry velocity must also remain within 15%",
        )

    def _final_pose_compatibility(self, observation: Any) -> GuardEvidence:
        if self.state.state_id != "P13":
            return GuardEvidence(
                "final_joint_pose_compatible",
                False,
                source="controller.phase_context",
                reason="final pose guard is only valid in P13",
            )
        actual = _actual_servo_positions(observation, self.contract.full12_order[:8])
        if actual is None:
            return GuardEvidence(
                "final_joint_pose_compatible",
                False,
                source="controller.phase_context",
                reason="live servo readback unavailable",
            )
        reference = self.state.reference_actual_endpoint_full12
        errors = {}
        passed = True
        for index, channel in enumerate(self.contract.full12_order[:8]):
            error = actual[index] - reference[index]
            limit = max(2.0, 0.15 * abs(self.phase.delta_full12[index]))
            errors[channel] = {"error_deg": error, "limit_deg": limit}
            passed = passed and abs(error) <= limit + 1e-12
        return GuardEvidence(
            "final_joint_pose_compatible",
            passed,
            errors,
            "controller.live_servo_vs_v010_actual_endpoint",
            "all final servos use max(2 deg, 15% of phase delta)",
        )

    def _held_or_safe_command(self) -> tuple[float, ...]:
        held = self.motion.held_full12(self.phase.start_full12)
        if self.termination is None:
            return held
        return held[:8] + (0.0, 0.0, 0.0, 0.0)

    def _transition(
        self,
        destination: Lifecycle,
        now: float,
        reason: str,
        events: list[ControllerEvent],
        details: Mapping[str, Any],
    ) -> None:
        source = self.lifecycle
        self.lifecycle = destination
        event = ControllerEvent(
            sim_time_s=now,
            state_id=self.state.state_id,
            from_lifecycle=source.value,
            to_lifecycle=destination.value,
            reason=reason,
            details=details,
        )
        events.append(event)
        self.history.append(event)

    def _remember_blocker(
        self, blocker: GuardEvidence | WatchdogBlocker | None
    ) -> None:
        if blocker is not None and self._first_blocker is None:
            self._first_blocker = _blocker_details(blocker)

    def _terminate(
        self,
        result: TaskResult,
        now: float,
        reason: str,
        details: Mapping[str, Any],
    ) -> None:
        if self.termination is None:
            self.termination = TaskTermination(
                result=result,
                state_id=self.state.state_id,
                lifecycle=self.lifecycle.value,
                sim_time_s=now,
                reason=reason,
                details=details,
            )

    def abort_infrastructure(self, reason: str, *, sim_time_s: float) -> None:
        self._terminate(TaskResult.INFRASTRUCTURE_ERROR, sim_time_s, reason, {})

    def abort_video_or_artifact(self, reason: str, *, sim_time_s: float) -> None:
        self._terminate(TaskResult.VIDEO_OR_ARTIFACT_ERROR, sim_time_s, reason, {})


def _validate_pair(spec: FsmSpec, contract: MotionContract) -> None:
    if spec.reference_version != contract.reference_version:
        raise ValueError("FSM and motion contract reference versions differ")
    if spec.rear_leg_order != contract.rear_leg_order:
        raise ValueError("FSM and motion contract rear-leg order differs")
    if spec.motion_hz != contract.physics_hz or spec.decision_hz != contract.decision_hz:
        raise ValueError("FSM and motion-contract rates differ")
    spec_ids = tuple(state.state_id for state in spec.states)
    phase_ids = tuple(phase.state_id for phase in contract.phases)
    if spec_ids != phase_ids:
        raise ValueError("FSM states and compact motion phases differ")
    for state, phase in zip(spec.states, contract.phases, strict=True):
        if state.completion_event != phase.completion_event:
            raise ValueError(f"{state.state_id}: completion-event contract mismatch")


def _actual_servo_positions(
    observation: Any, servo_order: Sequence[str]
) -> tuple[float, ...] | None:
    if isinstance(observation, Mapping):
        direct = observation.get("actual_full12")
        joints = observation.get("joints")
    else:
        direct = getattr(observation, "actual_full12", None)
        joints = getattr(observation, "joints", None)
    if direct is not None:
        try:
            values = tuple(float(item) for item in direct)
        except (TypeError, ValueError):
            return None
        return values[:8] if len(values) >= 8 else None
    if not isinstance(joints, Mapping):
        return None
    values = []
    for channel in servo_order:
        if channel not in joints:
            return None
        joint = joints[channel]
        value = (
            joint.get("position_deg")
            if isinstance(joint, Mapping)
            else getattr(joint, "position_deg", None)
        )
        if value is None:
            return None
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return tuple(values)


def _actual_full12(observation: Any) -> tuple[float, ...] | None:
    direct = (
        observation.get("actual_full12")
        if isinstance(observation, Mapping)
        else getattr(observation, "actual_full12", None)
    )
    if direct is None:
        return None
    try:
        values = tuple(float(item) for item in direct)
    except (TypeError, ValueError):
        return None
    return values if len(values) == 12 else None


def _actual_servo_velocities(
    observation: Any, servo_order: Sequence[str]
) -> tuple[float, ...] | None:
    if isinstance(observation, Mapping):
        direct = observation.get(
            "velocity_full12", observation.get("actual_velocity_full12")
        )
        joints = observation.get("joints")
    else:
        direct = getattr(
            observation,
            "velocity_full12",
            getattr(observation, "actual_velocity_full12", None),
        )
        joints = getattr(observation, "joints", None)
    if direct is not None:
        try:
            values = tuple(float(item) for item in direct)
        except (TypeError, ValueError):
            return None
        return values[:8] if len(values) >= 8 else None
    if not isinstance(joints, Mapping):
        return None
    values = []
    for channel in servo_order:
        if channel not in joints:
            return None
        joint = joints[channel]
        value = (
            joint.get("velocity_deg_s")
            if isinstance(joint, Mapping)
            else getattr(joint, "velocity_deg_s", None)
        )
        if value is None:
            return None
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return tuple(values)


def _report_details(report: GuardReport) -> Mapping[str, Any]:
    return {"guards": tuple(asdict(item) for item in report.evidence)}


def _blocker_name(blocker: GuardEvidence | WatchdogBlocker | None) -> str:
    if isinstance(blocker, GuardEvidence):
        return blocker.name
    if isinstance(blocker, WatchdogBlocker):
        return "state_progress_watchdog"
    return "unknown_live_guard"


def _blocker_details(
    blocker: GuardEvidence | WatchdogBlocker | None,
) -> dict[str, Any]:
    if blocker is None:
        return {"name": "unknown_live_guard"}
    details = asdict(blocker)
    details.setdefault("name", _blocker_name(blocker))
    return details
