"""120 Hz logical full-action execution for the compact v010 contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.reference.motion_contract import AtomicGroup, MotionPhase


ACTION_COUNT = 12
SERVO_COUNT = 8
MAX_CORRECTION_FRACTION = 0.15
MIN_TIME_SCALE = 0.85
MAX_TIME_SCALE = 1.15


def _full12(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in values)
    if len(result) != ACTION_COUNT:
        raise ValueError(f"{label}: expected {ACTION_COUNT} values")
    return result


@dataclass(frozen=True)
class FeedbackCorrection:
    """Per-channel fractional change to the reference excursion."""

    fractions: tuple[float, ...] = (0.0,) * ACTION_COUNT

    def __post_init__(self) -> None:
        if len(self.fractions) != ACTION_COUNT:
            raise ValueError("feedback correction must contain 12 fractions")
        if any(abs(value) > MAX_CORRECTION_FRACTION + 1e-12 for value in self.fractions):
            raise ValueError("feedback correction exceeds the +/-15% bound")


@dataclass(frozen=True)
class MotionTick:
    state_id: str
    tick_index: int
    elapsed_s: float
    full12: tuple[float, ...]
    nominal_full12: tuple[float, ...]
    atomic_groups: tuple[AtomicGroup, ...]
    tracking_servo_names: tuple[str, ...]
    endpoint_issued: bool
    target_progressed: bool

    @property
    def source_full12_atomic(self) -> bool:
        return any(group.source_full12_atomic for group in self.atomic_groups)


@dataclass(frozen=True)
class WatchdogBlocker:
    state_id: str
    lifecycle: str
    first_stalled_at_s: float
    detected_at_s: float
    no_progress_for_s: float
    last_target_full12: tuple[float, ...]
    last_actual_progress: tuple[float, ...]
    reason: str = "no target or actual progress for watchdog interval"


class ProgressWatchdog:
    def __init__(
        self,
        timeout_s: float = 0.5,
        *,
        target_epsilon: float = 1e-9,
        actual_epsilon: float = 1e-5,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("watchdog timeout must be positive")
        self.timeout_s = float(timeout_s)
        self.target_epsilon = float(target_epsilon)
        self.actual_epsilon = float(actual_epsilon)
        self.reset()

    def reset(self) -> None:
        self._last_progress_s: float | None = None
        self._last_target: tuple[float, ...] = ()
        self._last_actual: tuple[float, ...] = ()
        self._first_blocker: WatchdogBlocker | None = None

    def update(
        self,
        *,
        sim_time_s: float,
        state_id: str,
        lifecycle: str,
        target_full12: Sequence[float],
        actual_progress: Sequence[float],
    ) -> WatchdogBlocker | None:
        target = tuple(float(item) for item in target_full12)
        actual = tuple(float(item) for item in actual_progress)
        if self._last_progress_s is None:
            self._last_progress_s = float(sim_time_s)
            self._last_target = target
            self._last_actual = actual
            return None

        progressed = _changed(target, self._last_target, self.target_epsilon)
        progressed = progressed or _changed(actual, self._last_actual, self.actual_epsilon)
        if progressed:
            self._last_progress_s = float(sim_time_s)
            self._last_target = target
            self._last_actual = actual
            return None

        stalled_for = float(sim_time_s) - self._last_progress_s
        if stalled_for + 1e-12 < self.timeout_s:
            return None
        if self._first_blocker is None:
            self._first_blocker = WatchdogBlocker(
                state_id=state_id,
                lifecycle=lifecycle,
                first_stalled_at_s=self._last_progress_s,
                detected_at_s=float(sim_time_s),
                no_progress_for_s=stalled_for,
                last_target_full12=target,
                last_actual_progress=actual,
            )
        return self._first_blocker


def _changed(left: tuple[float, ...], right: tuple[float, ...], epsilon: float) -> bool:
    if len(left) != len(right):
        return bool(left or right)
    return any(abs(a - b) > epsilon for a, b in zip(left, right, strict=True))


class MotionExecutor:
    """Emit canonical nominal requests; response state survives phase changes.

    The Isaac adapter owns the mature 150 deg/s requested-to-drive mapping and
    tracking compensation.  Keeping that physical shaping below this logical
    executor preserves source Full12 dispatch provenance and one atomic write.
    """

    def __init__(
        self,
        *,
        physics_hz: float = 120.0,
        servo_rate_limit_deg_s: float = 150.0,
        initial_full12: Sequence[float] | None = None,
    ) -> None:
        if physics_hz <= 0.0 or servo_rate_limit_deg_s <= 0.0:
            raise ValueError("executor rates must be positive")
        self.physics_hz = float(physics_hz)
        self.dt_s = 1.0 / self.physics_hz
        self.servo_rate_limit_deg_s = float(servo_rate_limit_deg_s)
        self.max_servo_step_deg = self.servo_rate_limit_deg_s * self.dt_s
        self._last_full12 = (
            _full12(initial_full12, "initial_full12")
            if initial_full12 is not None
            else None
        )
        self._phase: MotionPhase | None = None
        self._correction = FeedbackCorrection()
        self._time_scale = 1.0
        self._tick_index = 0
        self._source_atomic_emitted = 0

    @property
    def last_full12(self) -> tuple[float, ...] | None:
        return self._last_full12

    @property
    def phase(self) -> MotionPhase | None:
        return self._phase

    @property
    def source_atomic_emitted(self) -> int:
        return self._source_atomic_emitted

    @property
    def effective_active_duration_s(self) -> float:
        if self._phase is None:
            return 0.0
        return self._endpoint_tick(self._phase) * self.dt_s

    def start_phase(
        self,
        phase: MotionPhase,
        correction: FeedbackCorrection | None = None,
        *,
        time_scale: float = 1.0,
    ) -> None:
        scale = float(time_scale)
        if (
            not math.isfinite(scale)
            or scale < MIN_TIME_SCALE - 1.0e-12
            or scale > MAX_TIME_SCALE + 1.0e-12
        ):
            raise ValueError("motion time scale exceeds the +/-15% bound")
        self._phase = phase
        self._correction = correction or FeedbackCorrection()
        self._time_scale = scale
        self._tick_index = 0
        if self._last_full12 is None:
            self._last_full12 = phase.start_full12

    def start_phase_at_endpoint(
        self,
        phase: MotionPhase,
        correction: FeedbackCorrection | None = None,
        *,
        time_scale: float = 1.0,
    ) -> None:
        """Issue one corrected endpoint without replaying the phase body.

        This is the bounded terminal-pose recovery primitive.  P13's endpoint
        waypoint keeps stopped wheels stopped and, because its authored atomic
        channels are wheel-only, does not restart servo tracking segments.
        Physical target slew remains owned by the adapter during VERIFY_RESULT.
        """

        self.start_phase(phase, correction, time_scale=time_scale)
        self._tick_index = self._endpoint_tick(phase)

    def tick(self) -> MotionTick:
        if self._phase is None or self._last_full12 is None:
            raise RuntimeError("start_phase must be called before tick")
        phase = self._phase
        elapsed_s = self._tick_index * self.dt_s
        waypoint_index = self._scaled_waypoint_index_at_tick(
            phase, self._tick_index
        )
        reference = phase.waypoints[waypoint_index].full12
        nominal = self._apply_correction(phase, reference)
        # Offline dispatch times contain harmless floating-point drift around
        # exact 120 Hz boundaries.  Quantize to the authored physics tick so an
        # event is neither delayed nor emitted twice.
        atomic_groups = tuple(
            group
            for group in phase.atomic_groups
            if self._scaled_source_tick(group.time_s) == self._tick_index
        )
        # The logical request remains a complete same-tick Full12.  Authored
        # waypoint times are causal source-dispatch onsets, so requests hold
        # until the next event.  The adapter alone creates the mature smooth
        # 120 Hz servo-drive path, then stages all twelve targets in one write.
        full12 = nominal
        waypoint = phase.waypoints[waypoint_index]
        tracking_servo_names = tuple(
            name for name in waypoint.atomic_channels if name in SERVO_ORDER
        )
        self._source_atomic_emitted += sum(
            1 for group in atomic_groups if group.source_full12_atomic
        )
        target_progressed = _changed(full12, self._last_full12, 1e-12)
        endpoint_issued = (
            self._tick_index >= self._endpoint_tick(phase)
            and not _changed(full12[:SERVO_COUNT], nominal[:SERVO_COUNT], 1e-9)
            and not _changed(full12[SERVO_COUNT:], phase.end_full12[SERVO_COUNT:], 1e-9)
        )
        result = MotionTick(
            state_id=phase.state_id,
            tick_index=self._tick_index,
            elapsed_s=elapsed_s,
            full12=full12,
            nominal_full12=nominal,
            atomic_groups=atomic_groups,
            tracking_servo_names=tracking_servo_names,
            endpoint_issued=endpoint_issued,
            target_progressed=target_progressed,
        )
        self._last_full12 = full12
        self._tick_index += 1
        return result

    def _nominal_at_tick(
        self, phase: MotionPhase, tick_index: int
    ) -> tuple[float, ...]:
        left_index = self._waypoint_index_at_tick(phase, tick_index)
        return phase.waypoints[left_index].full12

    def _waypoint_index_at_tick(self, phase: MotionPhase, tick_index: int) -> int:
        left_index = 0
        for index, waypoint in enumerate(phase.waypoints):
            if round(waypoint.time_s * self.physics_hz) <= tick_index:
                left_index = index
            else:
                break
        return left_index

    def _scaled_waypoint_index_at_tick(
        self, phase: MotionPhase, tick_index: int
    ) -> int:
        left_index = 0
        for index, waypoint in enumerate(phase.waypoints):
            if self._scaled_source_tick(waypoint.time_s) <= tick_index:
                left_index = index
            else:
                break
        return left_index

    def _scaled_source_tick(self, time_s: float) -> int:
        source_tick = round(float(time_s) * self.physics_hz)
        return round(source_tick * self._time_scale)

    def _endpoint_tick(self, phase: MotionPhase) -> int:
        source_tick = round(phase.active_duration_s * self.physics_hz)
        return round(source_tick * self._time_scale)

    def held_full12(self, fallback: Sequence[float]) -> tuple[float, ...]:
        return self._last_full12 or _full12(fallback, "fallback")

    def _apply_correction(
        self, phase: MotionPhase, reference: Sequence[float]
    ) -> tuple[float, ...]:
        result = []
        for start, value, fraction in zip(
            phase.start_full12,
            reference,
            self._correction.fractions,
            strict=True,
        ):
            result.append(start + (value - start) * (1.0 + fraction))
        return tuple(result)
