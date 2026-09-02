from __future__ import annotations

import math
from pathlib import Path

import pytest

from wlr50_clean.fsm.controller import SensorFsmController
from wlr50_clean.fsm.motion_executor import (
    FeedbackCorrection,
    MotionExecutor,
    ProgressWatchdog,
)
from wlr50_clean.fsm.state_graph import StateGraph
from wlr50_clean.fsm.state_spec import EXPECTED_STATE_IDS, Lifecycle, load_fsm_spec
from wlr50_clean.fsm.task_result import TASK_FAILURE_RESULTS, TaskResult
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def spec():
    return load_fsm_spec(ROOT / "configs" / "fsm_states.yaml")


@pytest.fixture(scope="module")
def contract():
    return load_motion_contract(ROOT / "configs" / "recording_motion_contract.json")


def _live_guards(*, completion: bool) -> dict[str, bool]:
    return {
        "no_body_obstacle_collision": True,
        "joint_hard_limits_valid": True,
        "reference_entry_compatible": True,
        "critical_actuators_available": True,
        "body_collision_persistent_or_penetrating": False,
        "wheel_only_climb_detected": False,
        "non_finite_observation_or_command": False,
        "physics_explosion_or_fall": False,
        "joint_hard_limit_violation": False,
        "fr_lift_entry_geometry": completion,
    }


def test_fixed_state_graph_and_lifecycle(spec) -> None:
    assert tuple(state.state_id for state in spec.states) == EXPECTED_STATE_IDS
    assert all(state.lifecycle == tuple(Lifecycle) for state in spec.states)
    graph = StateGraph(spec)
    assert graph.next("P01").state_id == "P02"
    assert graph.next("P13") is None
    with pytest.raises(ValueError):
        graph.validate_transition("P01", "P03")


def test_decision_and_motion_rates_are_locked(spec) -> None:
    assert spec.motion_hz == 120.0
    assert spec.decision_hz == 15.0
    assert spec.decision_stride == 8
    assert spec.watchdog_s == 0.5


def test_all_four_source_atomic_events_emit_as_full12(contract) -> None:
    executor = MotionExecutor(
        physics_hz=contract.physics_hz,
        servo_rate_limit_deg_s=contract.servo_rate_limit_deg_s,
        initial_full12=contract.phases[0].start_full12,
    )
    emitted = []
    for phase in contract.phases:
        executor.start_phase(phase)
        tick_count = math.ceil(phase.active_duration_s * contract.physics_hz) + 1
        for _ in range(tick_count):
            tick = executor.tick()
            assert len(tick.full12) == 12
            if tick.source_full12_atomic:
                assert tick.full12 == tick.nominal_full12
                emitted.append((phase.state_id, tick.tick_index, tick.full12))
    assert [item[0] for item in emitted] == ["P03", "P07", "P09", "P13"]
    assert emitted[0][1] == 0  # the P03 t=0 launch is not skipped
    assert executor.source_atomic_emitted == 4


def test_source_atomic_request_is_immediate_but_physical_slew_is_adapter_owned(
    contract,
) -> None:
    p02 = contract.phase("P02")
    p03 = contract.phase("P03")
    executor = MotionExecutor(
        physics_hz=120.0,
        servo_rate_limit_deg_s=150.0,
        initial_full12=p02.start_full12,
    )
    executor.start_phase(p02)
    first = executor.tick()
    assert first.full12 == first.nominal_full12
    carried = executor.last_full12
    executor.start_phase(p03)
    assert executor.last_full12 == carried
    boundary = executor.tick()
    assert boundary.source_full12_atomic
    assert boundary.nominal_full12 == p03.waypoints[1].full12
    assert boundary.full12 == boundary.nominal_full12
    assert tuple(boundary.tracking_servo_names) == tuple(contract.full12_order[:8])
    assert boundary.full12[8:] == boundary.nominal_full12[8:]  # wheel ZOH


def test_tracking_names_follow_waypoint_segments_not_phase_union(contract) -> None:
    executor = MotionExecutor(initial_full12=contract.phase("P03").start_full12)
    executor.start_phase(contract.phase("P03"))
    launch = executor.tick()
    assert launch.tracking_servo_names == tuple(contract.full12_order[:8])
    for _ in range(143):
        assert executor.tick().tracking_servo_names == tuple(
            contract.full12_order[:8]
        )
    wheel_stop = executor.tick()
    assert wheel_stop.tick_index == 144
    assert wheel_stop.tracking_servo_names == ()

    executor.start_phase(contract.phase("P04"))
    assert executor.tick().tracking_servo_names == ()
    for _ in range(535):
        tick = executor.tick()
    assert tick.tick_index == 535
    assert tick.tracking_servo_names == ()
    assert executor.tick().tracking_servo_names == ("rear_left_hip",)

    executor.start_phase(contract.phase("P05"))
    assert executor.tick().tracking_servo_names == ("front_left_knee",)
    for _ in range(31):
        executor.tick()
    assert executor.tick().tracking_servo_names == ("front_left_hip",)
    for _ in range(71):
        executor.tick()
    assert executor.tick().tracking_servo_names == ()


def test_p12_servo_requests_switch_only_at_authored_causal_ticks(contract) -> None:
    p12 = contract.phase("P12")
    executor = MotionExecutor(initial_full12=p12.start_full12)
    executor.start_phase(p12)
    samples = {}
    for tick_index in range(561):
        tick = executor.tick()
        if tick_index in (0, 31, 32, 39, 40, 552, 559, 560):
            samples[tick_index] = tick.full12[4]

    assert samples[0] == pytest.approx(15.4)
    assert samples[31] == pytest.approx(15.4)
    assert samples[32] == pytest.approx(13.2)
    assert samples[39] == pytest.approx(13.2)
    assert samples[40] == pytest.approx(0.5)
    assert samples[552] == pytest.approx(-7.9)
    assert samples[559] == pytest.approx(-7.9)
    assert samples[560] == pytest.approx(-10.1)
    assert p12.nominal_at(31.0 / 120.0)[4] == pytest.approx(15.4)
    assert p12.nominal_at(32.0 / 120.0)[4] == pytest.approx(13.2)


def test_next_phase_boundary_freezes_completed_tracking_owner(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    controller._tracking_servo_names = ("rear_left_hip",)
    events = []
    controller._enter_next_state(spec.state("P02"), 1.0, events)
    assert controller.lifecycle is Lifecycle.WAIT_ENTRY
    assert controller._tracking_servo_names == ()


def test_feedback_correction_cannot_exceed_fifteen_percent(contract) -> None:
    FeedbackCorrection((0.15, -0.15) + (0.0,) * 10)
    with pytest.raises(ValueError):
        FeedbackCorrection((0.150001,) + (0.0,) * 11)

    p03 = contract.phase("P03")
    fractions = (0.0, 0.15) + (0.0,) * 10
    executor = MotionExecutor(initial_full12=p03.start_full12)
    executor.start_phase(p03, FeedbackCorrection(fractions))
    atomic = executor.tick()
    expected_knee = p03.start_full12[1] + 1.15 * (
        p03.waypoints[1].full12[1] - p03.start_full12[1]
    )
    assert atomic.source_full12_atomic
    assert atomic.full12 == atomic.nominal_full12
    assert atomic.full12[1] == pytest.approx(expected_knee)


def test_p10_entry_keeps_the_unmodified_reference_motion(
    spec, contract
) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P10")
    p10 = contract.phase("P10")
    alignment = p10.entry_velocity_alignment
    assert alignment is not None
    guards = _live_guards(completion=False)
    actual = list(controller.state.reference_actual_start_full12)
    # Trial011 proved that a full +15% P10 excursion correction was physically
    # too late to repair the carry-in phase.  A guard-compatible entry must
    # therefore retain v010's authored P10 request unchanged.
    actual[7] = -51.465317474601946
    velocity = [0.0] * 12
    velocity[7] = alignment.reference_velocity_deg_s
    observation = {
        "guards": guards,
        "actual_full12": actual,
        "velocity_full12": velocity,
    }

    frames = [controller.step(observation, sim_time_s=0.0)]
    for tick_index in range(1, 17):
        frames.append(
            controller.step(observation, sim_time_s=tick_index / 120.0)
        )

    transition = next(
        event
        for event in frames[0].events
        if event.from_lifecycle == Lifecycle.WAIT_ENTRY.value
        and event.to_lifecycle == Lifecycle.EXECUTE_MOTION.value
    )
    fractions = transition.details["correction_fractions"]
    assert fractions == pytest.approx((0.0,) * 12)
    assert frames[0].full12[7] == pytest.approx(-34.6)
    assert frames[8].full12[7] == pytest.approx(-29.3)
    assert frames[16].full12[7] == pytest.approx(-27.2)
    assert frames[0].full12[:7] + frames[0].full12[8:] == pytest.approx(
        p10.waypoints[1].full12[:7] + p10.waypoints[1].full12[8:]
    )


def test_controller_exposes_live_latched_p09_drive_feedback(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P09")
    controller.lifecycle = Lifecycle.EXECUTE_MOTION
    phase = contract.phase("P09")
    feedback = phase.drive_feedback
    assert feedback is not None
    controller.motion.start_phase(phase)
    guards = _live_guards(completion=False)
    frames = {}
    for tick in range(feedback.teardown_tick + 1):
        actual = list(controller.state.reference_actual_start_full12)
        probe = next(
            (item for item in feedback.probe_samples if item.motion_tick == tick),
            None,
        )
        if probe is not None:
            actual[feedback.probe_channel_index] = (
                probe.reference_actual_deg - feedback.lag_threshold_deg
            )
        frames[tick] = controller.step(
            {
                "guards": guards,
                "actual_full12": actual,
                "progress_vector": (float(tick),),
            },
            sim_time_s=tick / 120.0,
        )

    first = frames[feedback.first_bias_tick]
    secondary_first = frames[feedback.bias_segments[1].first_bias_tick]
    tail = frames[feedback.last_bias_tick]
    restored = frames[feedback.teardown_tick]
    assert first.lifecycle is Lifecycle.EXECUTE_MOTION
    assert first.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(feedback.bias_segments[0].logical_bias_rad_s)
    assert secondary_first.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(feedback.bias_segments[1].logical_bias_rad_s)
    assert tail.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(feedback.bias_segments[1].logical_bias_rad_s)
    assert first.full12[feedback.correction_channel_index] == pytest.approx(-1.07)
    assert first.full12[feedback.correction_channel_index] + first.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(-0.74)
    assert tail.full12[feedback.correction_channel_index] == 0.0
    assert tail.full12[feedback.correction_channel_index] + tail.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(0.17)
    assert restored.drive_feedback_bias_full12 == pytest.approx((0.0,) * 12)
    assert first.drive_feedback_details[
        "cumulative_fraction_of_reference"
    ] == pytest.approx(
        feedback.cumulative_fraction_of_reference
    )
    assert first.drive_feedback_details["kind"] == feedback.kind
    assert first.drive_feedback_details["active_segment_index"] == 0
    assert first.drive_feedback_details["logical_bias_rad_s"] == pytest.approx(0.33)
    assert secondary_first.drive_feedback_details["active_segment_index"] == 1
    assert secondary_first.drive_feedback_details[
        "logical_bias_rad_s"
    ] == pytest.approx(0.17)
    assert first.drive_feedback_details[
        "resulting_wheel_integral_rad"
    ] == pytest.approx(feedback.resulting_wheel_integral_rad)
    assert first.drive_feedback_details[
        "reference_wheel_peak_abs_rad_s"
    ] == pytest.approx(feedback.reference_wheel_peak_abs_rad_s)
    assert first.drive_feedback_details[
        "resulting_wheel_peak_abs_rad_s"
    ] == pytest.approx(feedback.resulting_wheel_peak_abs_rad_s)
    assert first.drive_feedback_details["instantaneous_direction_reversal"] is True
    assert (
        first.drive_feedback_details["probe_channel"] == feedback.probe_channel
    )
    assert (
        first.drive_feedback_details["probe_channel_index"]
        == feedback.probe_channel_index
    )
    assert (
        first.drive_feedback_details["correction_channel"]
        == feedback.correction_channel
    )
    assert (
        first.drive_feedback_details["correction_channel_index"]
        == feedback.correction_channel_index
    )
    assert restored.drive_feedback_details["tick_index"] == feedback.teardown_tick


@pytest.mark.parametrize(
    ("velocity_scale", "expected_lifecycle", "expected_motion_tick"),
    (
        (1.0, Lifecycle.EXECUTE_MOTION, 0),
        (0.5, Lifecycle.WAIT_ENTRY, None),
    ),
)
def test_p09_wheel_rebound_is_zero_at_the_delayed_p10_entry_decision(
    spec,
    contract,
    velocity_scale: float,
    expected_lifecycle: Lifecycle,
    expected_motion_tick: int | None,
) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P09")
    controller.lifecycle = Lifecycle.EXECUTE_MOTION
    p09 = contract.phase("P09")
    feedback = p09.drive_feedback
    alignment = contract.phase("P10").entry_velocity_alignment
    assert feedback is not None and alignment is not None
    controller.motion.start_phase(p09)

    frames = {}
    for tick in range(feedback.teardown_tick + 1):
        completing = tick == feedback.teardown_tick
        physical_completion_ready = (
            tick
            >= round(p09.active_duration_s * 120.0) + spec.decision_stride
        )
        guards = _live_guards(completion=False)
        for name in (
            "reference_like_active_lift:RR",
            "leg_front_face_crossed_latched:RR",
            "leg_top_loaded_latched:RR",
        ):
            guards[name] = physical_completion_ready
        actual = list(
            controller.state.reference_actual_start_full12
            if not completing
            else spec.state("P10").reference_actual_start_full12
        )
        probe = next(
            (item for item in feedback.probe_samples if item.motion_tick == tick),
            None,
        )
        if probe is not None:
            actual[feedback.probe_channel_index] = (
                probe.reference_actual_deg - feedback.lag_threshold_deg
            )
        velocity = [0.0] * 12
        if completing:
            velocity[alignment.channel_index] = (
                alignment.reference_velocity_deg_s * velocity_scale
            )
        frames[tick] = controller.step(
            {
                "guards": guards,
                "actual_full12": actual,
                "velocity_full12": velocity,
                "progress_vector": (float(tick),),
            },
            sim_time_s=tick / 120.0,
        )

    for tick in range(
        feedback.probe_samples[-1].motion_tick + 1,
        feedback.first_bias_tick,
    ):
        armed = frames[tick]
        assert armed.state_id == "P09"
        assert armed.lifecycle is Lifecycle.EXECUTE_MOTION
        assert armed.drive_feedback_bias_full12 == pytest.approx((0.0,) * 12)
        assert not any(
            event.to_lifecycle == Lifecycle.DONE.value for event in armed.events
        )
    held = frames[feedback.first_bias_tick]
    assert held.state_id == "P09"
    assert held.lifecycle is Lifecycle.EXECUTE_MOTION
    assert held.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(feedback.bias_segments[0].logical_bias_rad_s)
    assert held.full12[feedback.correction_channel_index] == pytest.approx(-1.07)
    assert held.full12[feedback.correction_channel_index] + held.drive_feedback_bias_full12[
        feedback.correction_channel_index
    ] == pytest.approx(-0.74)
    assert not any(
        event.to_lifecycle == Lifecycle.DONE.value for event in held.events
    )
    endpoint_tick = round(p09.active_duration_s * 120.0)
    primary, secondary = feedback.bias_segments
    for tick in range(primary.first_bias_tick, endpoint_tick):
        active = frames[tick]
        assert active.lifecycle is Lifecycle.EXECUTE_MOTION
        assert active.full12[feedback.correction_channel_index] == pytest.approx(
            -1.07
        )
        assert active.full12[
            feedback.correction_channel_index
        ] + active.drive_feedback_bias_full12[
            feedback.correction_channel_index
        ] == pytest.approx(-0.74)
    for tick in range(endpoint_tick, primary.last_bias_tick + 1):
        active = frames[tick]
        assert active.lifecycle is Lifecycle.VERIFY_RESULT
        assert active.full12[feedback.correction_channel_index] == pytest.approx(
            0.0
        )
        assert active.full12[
            feedback.correction_channel_index
        ] + active.drive_feedback_bias_full12[
            feedback.correction_channel_index
        ] == pytest.approx(0.33)
    for tick in range(
        secondary.first_bias_tick, secondary.last_bias_tick + 1
    ):
        active = frames[tick]
        assert active.lifecycle is Lifecycle.VERIFY_RESULT
        assert active.full12[feedback.correction_channel_index] == pytest.approx(
            0.0
        )
        assert active.full12[
            feedback.correction_channel_index
        ] + active.drive_feedback_bias_full12[
            feedback.correction_channel_index
        ] == pytest.approx(0.17)
    frame = frames[feedback.teardown_tick]
    assert frame.state_id == "P10"
    assert frame.lifecycle is expected_lifecycle
    assert frame.drive_feedback_bias_full12 == pytest.approx((0.0,) * 12)
    assert frame.drive_feedback_details["tick_index"] == expected_motion_tick
    assert any(
        event.state_id == "P09" and event.to_lifecycle == Lifecycle.DONE.value
        for event in frame.events
    )
    if expected_lifecycle is Lifecycle.EXECUTE_MOTION:
        assert any(
            event.state_id == "P10"
            and event.to_lifecycle == Lifecycle.EXECUTE_MOTION.value
            for event in frame.events
        )
    else:
        assert not any(
            event.state_id == "P10"
            and event.to_lifecycle == Lifecycle.EXECUTE_MOTION.value
            for event in frame.events
        )


def test_p09_without_live_deficit_does_not_add_a_verify_quantum(
    spec, contract
) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P09")
    controller.lifecycle = Lifecycle.EXECUTE_MOTION
    p09 = contract.phase("P09")
    feedback = p09.drive_feedback
    alignment = contract.phase("P10").entry_velocity_alignment
    assert feedback is not None and alignment is not None
    controller.motion.start_phase(p09)

    decision_tick = round(p09.active_duration_s * 120.0) + spec.decision_stride
    for tick in range(decision_tick + 1):
        completing = tick == decision_tick
        guards = _live_guards(completion=False)
        for name in (
            "reference_like_active_lift:RR",
            "leg_front_face_crossed_latched:RR",
            "leg_top_loaded_latched:RR",
        ):
            guards[name] = completing
        actual = list(
            spec.state("P10").reference_actual_start_full12
            if completing
            else controller.state.reference_actual_start_full12
        )
        probe = next(
            (item for item in feedback.probe_samples if item.motion_tick == tick),
            None,
        )
        if probe is not None:
            actual[feedback.probe_channel_index] = probe.reference_actual_deg
        velocity = [0.0] * 12
        if completing:
            velocity[alignment.channel_index] = alignment.reference_velocity_deg_s
        frame = controller.step(
            {
                "guards": guards,
                "actual_full12": actual,
                "velocity_full12": velocity,
                "progress_vector": (float(tick),),
            },
            sim_time_s=tick / 120.0,
        )

    assert frame.state_id == "P10"
    assert frame.lifecycle is Lifecycle.EXECUTE_MOTION
    assert frame.drive_feedback_bias_full12 == pytest.approx((0.0,) * 12)


@pytest.mark.parametrize(
    ("velocity_scale", "passed"),
    (
        (0.85, True),
        (1.15, True),
        (0.849, False),
        (1.151, False),
        (-1.0, False),
    ),
)
def test_p10_entry_requires_signed_reference_velocity_corridor(
    spec, contract, velocity_scale: float, passed: bool
) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P10")
    alignment = controller.phase.entry_velocity_alignment
    assert alignment is not None
    velocity = [0.0] * 12
    velocity[alignment.channel_index] = (
        alignment.reference_velocity_deg_s * velocity_scale
    )

    evidence = controller._entry_compatibility(
        {
            "actual_full12": controller.state.reference_actual_start_full12,
            "velocity_full12": velocity,
        }
    )

    assert evidence.passed is passed
    details = evidence.value[f"{alignment.channel}_velocity"]
    assert details["reference_deg_s"] == pytest.approx(
        alignment.reference_velocity_deg_s
    )
    assert details["limit_deg_s"] == pytest.approx(
        0.15 * alignment.reference_velocity_deg_s
    )


def test_p10_entry_velocity_guard_fails_closed_without_live_velocity(
    spec, contract
) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P10")
    evidence = controller._entry_compatibility(
        {"actual_full12": controller.state.reference_actual_start_full12}
    )
    assert evidence.passed is False


def test_watchdog_reports_detailed_first_stall() -> None:
    watchdog = ProgressWatchdog(0.5)
    command = (0.0,) * 12
    assert watchdog.update(
        sim_time_s=2.0,
        state_id="P04",
        lifecycle="EXECUTE_MOTION",
        target_full12=command,
        actual_progress=(1.0,),
    ) is None
    assert watchdog.update(
        sim_time_s=2.49,
        state_id="P04",
        lifecycle="EXECUTE_MOTION",
        target_full12=command,
        actual_progress=(1.0,),
    ) is None
    blocker = watchdog.update(
        sim_time_s=2.5,
        state_id="P04",
        lifecycle="EXECUTE_MOTION",
        target_full12=command,
        actual_progress=(1.0,),
    )
    assert blocker is not None
    assert blocker.state_id == "P04"
    assert blocker.no_progress_for_s == pytest.approx(0.5)
    assert blocker.first_stalled_at_s == 2.0


def test_elapsed_time_never_completes_a_state(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    guards = _live_guards(completion=False)
    # Actual progress changes every tick, so only the missing live completion
    # guard can block this trial; elapsed duration itself cannot pass it.
    for index in range(4000):
        frame = controller.step(
            {
                "guards": guards,
                "progress_vector": (index * 0.001,),
                "actual_full12": contract.phase("P01").start_full12,
            },
            sim_time_s=index / 120.0,
        )
        if frame.termination is not None:
            break
    assert frame.termination is not None
    assert frame.termination.result is TaskResult.INCOMPLETE_CONTROLLER_BLOCKED
    assert controller.state.state_id == "P01"
    assert controller.retries_used == 1
    assert (
        frame.termination.reason
        == "controller exhausted its one reference-bounded retry"
    )
    assert frame.first_blocker["name"] == "fr_lift_entry_geometry"


def test_measured_wheel_decay_is_debounced_inside_controller(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P13")
    controller.lifecycle = Lifecycle.VERIFY_RESULT
    controller._endpoint_issued = True
    controller._verify_started_s = 0.0
    guards = {
        guard.name: True for guard in controller.state.completion_guards
    }
    # Even a supplied True guard cannot bypass measured velocity debounce.
    first = controller.step(
        {
            "guards": guards,
            "wheel_velocities_rad_s": (0.0,) * 4,
            "commanded_full12": (0.0,) * 12,
            "actual_full12": controller.state.reference_actual_endpoint_full12,
        },
        sim_time_s=0.0,
    )
    assert first.termination is None
    assert first.lifecycle is Lifecycle.VERIFY_RESULT
    for index in range(1, 65):
        frame = controller.step(
            {
                "guards": guards,
                "wheel_velocities_rad_s": (0.0,) * 4,
                "commanded_full12": (0.0,) * 12,
                "actual_full12": controller.state.reference_actual_endpoint_full12,
            },
            sim_time_s=index / 120.0,
        )
        if frame.termination:
            break
    assert frame.termination is not None
    assert frame.termination.result is TaskResult.SUCCESS
    assert frame.sim_time_s >= 0.5


def test_nondecision_wheel_velocity_spike_resets_continuous_debounce(
    spec, contract
) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P13")
    controller.lifecycle = Lifecycle.VERIFY_RESULT
    controller._endpoint_issued = True
    controller._verify_started_s = 0.0
    guards = {guard.name: True for guard in controller.state.completion_guards}
    endpoint = controller.state.reference_actual_endpoint_full12
    threshold = next(
        float(guard.parameters["absolute_threshold_rad_s"])
        for guard in controller.state.completion_guards
        if guard.name == "measured_wheel_velocity_stable_decay"
    )

    frame = None
    for index in range(105):
        # Tick 36 is not a 15 Hz decision tick.  It must still reset the
        # continuous evidence, exactly like Trial011's missed tick 10636.
        velocity = (threshold + 0.01, 0.0, 0.0, 0.0) if index == 36 else (0.0,) * 4
        frame = controller.step(
            {
                "guards": guards,
                "wheel_velocities_rad_s": velocity,
                "commanded_full12": (0.0,) * 12,
                "actual_full12": endpoint,
            },
            sim_time_s=index / 120.0,
        )
        if frame.termination is not None:
            break

    assert frame is not None and frame.termination is not None
    assert frame.termination.result is TaskResult.SUCCESS
    assert frame.sim_time_s >= (36.0 / 120.0 + 0.5)


def test_p13_decay_threshold_is_derived_from_v010_tail(spec, contract) -> None:
    state = spec.state("P13")
    guard = next(
        item
        for item in state.completion_guards
        if item.name == "measured_wheel_velocity_stable_decay"
    )
    reference_tail_peak = float(guard.parameters["reference_tail_peak_rad_s"])
    assert reference_tail_peak == pytest.approx(0.22262312471866608)
    assert guard.parameters["absolute_threshold_rad_s"] == pytest.approx(
        1.15 * reference_tail_peak
    )
    assert guard.parameters["reference_relative_allowance"] == pytest.approx(0.15)


def test_hard_abort_is_lifecycle_independent(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    blocked_entry = _live_guards(completion=False)
    blocked_entry["no_body_obstacle_collision"] = False
    blocked_entry["body_collision_persistent_or_penetrating"] = True
    observation = {
        "guards": blocked_entry,
        "progress_vector": (0.0,),
        "actual_full12": contract.phase("P01").start_full12,
    }
    first = controller.step(observation)
    assert first.lifecycle is Lifecycle.WAIT_ENTRY
    assert first.termination is not None
    assert first.termination.result is TaskResult.TASK_FAILURE_BODY_COLLISION

    verify = SensorFsmController(spec, contract)
    verify.lifecycle = Lifecycle.VERIFY_RESULT
    verify._endpoint_issued = True
    verify._verify_started_s = 0.0
    frame = verify.step(observation, sim_time_s=0.0)
    assert frame.termination is not None
    assert frame.termination.result is TaskResult.TASK_FAILURE_BODY_COLLISION


def test_entry_compatibility_uses_active_servos_and_ignores_wheels(spec, contract) -> None:
    phase = contract.phase("P01")
    guards = _live_guards(completion=False)
    probe = SensorFsmController(spec, contract)
    actual = list(probe.state.reference_actual_start_full12)
    # P01's active rear-left hip has a 5.64 degree limit.  Wheel readback is
    # intentionally extreme and must not affect entry compatibility.
    actual[4] = probe.state.reference_actual_start_full12[4] + 5.6
    actual[8:] = [100.0, -100.0, 50.0, -50.0]
    controller = SensorFsmController(spec, contract)
    frame = controller.step({"guards": guards, "actual_full12": actual})
    assert frame.lifecycle is Lifecycle.EXECUTE_MOTION
    assert frame.first_blocker is None

    actual[4] = probe.state.reference_actual_start_full12[4] + 5.7
    blocked = SensorFsmController(spec, contract)
    frame = blocked.step({"guards": guards, "actual_full12": actual})
    assert frame.lifecycle is Lifecycle.WAIT_ENTRY
    assert frame.first_blocker is None  # transient sensor/entry mismatch is not latched


def test_wait_entry_guard_jitter_cannot_extend_past_watchdog(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    guards = _live_guards(completion=False)
    guards["critical_actuators_available"] = False
    frame = None
    for index in range(61):
        frame = controller.step(
            {
                "guards": guards,
                # Deliberately changing progress used to keep the generic
                # no-motion watchdog alive indefinitely.
                "progress_vector": (index * 1.0e-3,),
                "actual_full12": controller.state.reference_actual_start_full12,
            },
            sim_time_s=index / 120.0,
        )
    assert frame is not None and frame.termination is not None
    assert frame.termination.result is TaskResult.INCOMPLETE_CONTROLLER_BLOCKED
    assert controller.lifecycle is Lifecycle.WAIT_ENTRY
    assert controller.retries_used == 0
    assert (
        frame.termination.reason
        == "live entry guards remained incompatible after the bounded wait window; "
        "no recovery retry was attempted"
    )
    assert frame.termination.details["retries_used"] == 0
    assert frame.first_blocker["name"] == "critical_actuators_available"


def test_entry_compatibility_uses_measured_reference_start(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    controller.state = spec.state("P05")
    actual = controller.state.reference_actual_start_full12
    # The measured v010 P05 knee entry differs substantially from the command
    # start due to the continuous physical response carried across P04/P05.
    assert abs(actual[1] - controller.phase.start_full12[1]) > 2.0
    evidence = controller._entry_compatibility({"actual_full12": actual})
    assert evidence.passed is True
    assert (
        evidence.value["front_left_knee"]["reference_actual_start_deg"]
        == actual[1]
    )


def test_initial_sensor_latency_does_not_pollute_first_blocker(spec, contract) -> None:
    controller = SensorFsmController(spec, contract)
    guards = _live_guards(completion=False)
    frame = controller.step({"guards": guards, "progress_vector": (0.0,)})
    assert frame.lifecycle is Lifecycle.WAIT_ENTRY
    assert frame.first_blocker is None
    for index in range(1, 9):
        frame = controller.step(
            {
                "guards": guards,
                "progress_vector": (float(index),),
                "actual_full12": contract.phase("P01").start_full12,
            },
            sim_time_s=index / 120.0,
        )
    assert frame.lifecycle is Lifecycle.EXECUTE_MOTION
    assert frame.first_blocker is None


def test_result_classes_are_disjoint_and_exhaustive() -> None:
    expected = {
        "SUCCESS",
        "TASK_FAILURE_BODY_COLLISION",
        "TASK_FAILURE_WHEEL_ONLY_CLIMB",
        "INCOMPLETE_CONTROLLER_BLOCKED",
        "SAFETY_ABORT",
        "INFRASTRUCTURE_ERROR",
        "VIDEO_OR_ARTIFACT_ERROR",
    }
    assert {item.value for item in TaskResult} == expected
    assert TASK_FAILURE_RESULTS == {
        TaskResult.TASK_FAILURE_BODY_COLLISION,
        TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB,
    }
