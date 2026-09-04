from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo.isaac_fsm_backend import (
    DEFAULT_FSM_PATH,
    DEFAULT_MOTION_CONTRACT_PATH,
    LEVEL_CALIBRATION_TICKS,
    SETTLE_TICKS,
    BackendDependencies,
    IsaacFSMBackend,
    IsaacFSMBackendError,
    LoadedPhaseSnapshot,
    SensorContractFailure,
    build_residual_actuation_plan,
    capture_canonical_articulation_reset_state,
    restore_canonical_articulation_reset_state,
    _load_validated_phase_snapshot,
    _restore_controller_from_snapshot,
    _restore_guard_tracker_from_snapshot,
)


SERVO_NAMES = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_NAMES = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
BODY_NAMES = tuple(f"body_{index:02d}" for index in range(13))
ZERO = (0.0,) * 12


def _value(name: str):
    return SimpleNamespace(value=name)


def _guard(passed: bool):
    return {"passed": passed, "value": passed, "source": "fake.live"}


def _observation(tick: int, fault: str | None = None, *, invalid_pair=False):
    pitch = 0.2
    orientation = (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0)
    base_z = 0.0 if fault == "fall" else 0.10
    linear_velocity = (6.0, 0.0, 0.0) if fault == "explosion" else (0.12, 0.0, 0.0)
    angular_velocity = (
        (float("nan"), 0.0, 0.0)
        if fault == "nan"
        else (0.0, 0.4, 0.0)
    )
    base = SimpleNamespace(
        name=BODY_NAMES[0],
        position_w_m=(tick * 0.001, 0.0, base_z),
        orientation_wxyz=orientation,
        linear_velocity_w_m_s=linear_velocity,
        angular_velocity_w_rad_s=angular_velocity,
    )
    contacts = {}
    for index, body_name in enumerate(BODY_NAMES):
        contacts[body_name] = SimpleNamespace(
            body_name=body_name,
            contact_class=_value("GROUND"),
            ground=SimpleNamespace(
                pair_verified=not (invalid_pair and index == 0), active=True
            ),
            obstacle=SimpleNamespace(pair_verified=True, active=False),
        )
    joints = {
        name: SimpleNamespace(
            name=name,
            position_deg=0.0,
            velocity_deg_s=0.0,
            command_deg=0.0,
            error_deg=0.0,
        )
        for name in SERVO_NAMES
    }
    wheels = {
        name: SimpleNamespace(
            name=name,
            body_name=BODY_NAMES[index],
            velocity_rad_s=0.0,
            command_rad_s=0.0,
            center_w_m=(0.2 + index * 0.1, 0.0, 0.05),
            bottom_w_m=(0.2 + index * 0.1, 0.0, 0.01),
            geometry_verified=True,
        )
        for index, name in enumerate(WHEEL_NAMES)
    }
    guards = {
        "wheel_only_climb_detected": _guard(fault == "wheel"),
        "non_finite_observation_or_command": _guard(fault == "nan"),
        "joint_hard_limit_violation": _guard(fault == "joint"),
        "physics_explosion_or_fall": _guard(fault in {"fall", "explosion"}),
    }
    for leg in ("FL", "FR", "RL", "RR"):
        for name in (
            "reference_like_active_lift",
            "leg_front_face_crossed_latched",
            "leg_top_loaded_latched",
        ):
            guards[f"{name}:{leg}"] = _guard(False)
    return SimpleNamespace(
        schema="fake.live_observation.v1",
        physics_tick=tick,
        simulation_time_s=tick / 120.0,
        physics_dt_s=1.0 / 120.0,
        joints=joints,
        wheels=wheels,
        contacts=contacts,
        bodies={name: base for name in BODY_NAMES},
        base=base,
        imu=SimpleNamespace(
            projected_gravity_b=(0.0, 0.0, -1.0),
            angular_velocity_b_rad_s=(0.0, 0.4, 0.0),
            linear_acceleration_b_m_s2=(0.0, 0.0, 0.0),
        ),
        obstacle=SimpleNamespace(
            front_x_m=0.5,
            back_x_m=2.5,
            left_y_m=-1.0,
            right_y_m=1.0,
            bottom_z_m=0.0,
            top_z_m=0.05,
        ),
        center_of_mass=SimpleNamespace(
            position_w_m=(tick * 0.001, 0.0, 0.09),
            velocity_w_m_s=(0.12, 0.0, 0.0),
            included_bodies=BODY_NAMES,
            valid=True,
        ),
        support=SimpleNamespace(
            signed_margin_m=0.01,
            projection_inside=True,
            support_count=4,
            valid=True,
        ),
        body_collision=SimpleNamespace(detected=fault == "body"),
        actual_full12=ZERO,
        all_finite=fault != "nan",
        guards=guards,
        data_quality=(),
    )


def _snapshot_payload(phase="P09"):
    phase_index = int(phase[1:]) - 1
    latches = {}
    for leg in ("FL", "FR", "RL", "RR"):
        completed = leg in {"FL", "FR"} if phase_index >= 8 else False
        latches[leg] = {
            "active_lift": completed,
            "active_lift_tick": 10 if completed else None,
            "front_face_crossed": completed,
            "front_face_crossed_tick": 20 if completed else None,
            "top_loaded": completed,
            "top_loaded_tick": 30 if completed else None,
        }
    history = list(f"P{index:02d}" for index in range(1, phase_index + 1))
    pitch = 0.2
    return {
        "schema": "wlr50_clean.ppo_phase_entry_snapshot.v1",
        "reset_use": "TRAINING_RESET_STATE_WRITE",
        "in_episode_root_write": "FORBIDDEN_IN_EPISODE_ROOT_WRITE",
        "source_tick": 100,
        "source_time_s": 100 / 120.0,
        "fsm_state": phase,
        "fsm_lifecycle": "EXECUTE_MOTION",
        "phase_history": history,
        "root_state": {
            "position_w_m": [0.0, 0.0, 0.1],
            "orientation_wxyz": [
                math.cos(pitch / 2.0),
                0.0,
                math.sin(pitch / 2.0),
                0.0,
            ],
            "linear_velocity_w_m_s": [0.12, 0.0, 0.0],
            "angular_velocity_w_rad_s": [0.0, 0.4, 0.0],
        },
        "joint_state": {
            "logical_position_deg": [0.0] * 8,
            "logical_velocity_deg_s": [0.0] * 8,
            "order": list(SERVO_NAMES),
        },
        "wheel_state": {
            "logical_velocity_rad_s": [0.0] * 4,
            "order": list(WHEEL_NAMES),
        },
        "nominal_full12": [0.0] * 12,
        "applied_full12": [0.0] * 12,
        "fsm_history": {"completed_phases": history, "recovery_count": 0},
        "contact_event_latches": latches,
        "obstacle_relative_geometry": {
            "obstacle": {},
            "wheel_centers_w_m": {
                name: [0.2 + index * 0.1, 0.0, 0.05]
                for index, name in enumerate(WHEEL_NAMES)
            },
            "wheel_bottoms_w_m": {
                name: [0.2 + index * 0.1, 0.0, 0.01]
                for index, name in enumerate(WHEEL_NAMES)
            },
        },
        "contact_state": {
            name: {
                "class": "GROUND",
                "ground_active": True,
                "obstacle_active": False,
            }
            for name in WHEEL_NAMES
        },
        "level_reference_orientation_wxyz": [
            math.cos(pitch / 2.0),
            0.0,
            math.sin(pitch / 2.0),
            0.0,
        ],
        "snapshot_semantics": "reset-only test snapshot",
        "state_sha256": "fake-state-sha256",
    }


class FakeMotion:
    def __init__(self, phase):
        self.phase = phase
        self._tick_index = 0
        self.effective_active_duration_s = 1.0
        self.servo_rate_limit_deg_s = 150.0

    def _scaled_waypoint_index_at_tick(self, phase, tick):
        return 0


class FakeController:
    def __init__(self, runtime):
        self.runtime = runtime
        self.phase = SimpleNamespace(
            state_id="P01",
            macro_phase=1,
            start_full12=ZERO,
            delta_full12=(10.0,) * 8 + ZERO[8:],
            action_mask_full12=(1,) * 12,
            waypoints=(SimpleNamespace(full12=ZERO),),
        )
        self.motion = FakeMotion(self.phase)
        self.physics_tick = 0
        self.abort_calls = []

    def step(self, observation, *, sim_time_s):
        tick = self.physics_tick
        self.runtime.events.append(("controller.step", tick, sim_time_s))
        self.motion._tick_index = tick + 1
        termination = None
        if tick == 1 and self.runtime.fault == "success":
            termination = SimpleNamespace(result=_value("SUCCESS"), reason="P13 guards")
        elif tick == 1 and self.runtime.fault == "blocked":
            termination = SimpleNamespace(
                result=_value("INCOMPLETE_CONTROLLER_BLOCKED"), reason="watchdog"
            )
        frame = SimpleNamespace(
            physics_tick=tick,
            sim_time_s=sim_time_s,
            state_id=self.phase.state_id,
            lifecycle=_value("EXECUTE_MOTION"),
            full12=(tick / 100.0,) + ZERO[1:],
            decision_tick=tick % 8 == 0,
            full12_atomic_write_required=True,
            atomic_source_event=False,
            tracking_servo_names=("front_left_hip",),
            drive_feedback_bias_full12=(0.1,) + ZERO[1:],
            normal_drive_bias_full12=(0.2,) + ZERO[1:],
            drive_feedback_details={"fake": True},
            endpoint_issued=False,
            termination=termination,
            first_blocker=None,
            events=(),
        )
        self.physics_tick += 1
        return frame

    def abort_infrastructure(self, reason, *, sim_time_s):
        self.abort_calls.append((reason, sim_time_s))


class FakeReader:
    def __init__(self, runtime, role):
        self.runtime = runtime
        self.role = role
        self.last_command = None
        self.ticks = []

    def read(self, *, physics_tick, simulation_time_s, commanded_full12):
        if self.runtime.strict_reader_clock and physics_tick != len(self.ticks):
            raise RuntimeError(
                f"non-contiguous fake reader clock: {physics_tick} after {self.ticks}"
            )
        self.ticks.append(physics_tick)
        self.last_command = tuple(commanded_full12)
        self.runtime.events.append(
            (f"reader.{self.role}", physics_tick, simulation_time_s)
        )
        fault = (
            self.runtime.fault
            if self.role == "live" and physics_tick == 1
            else None
        )
        invalid = bool(
            self.runtime.invalid_pair and self.role == "live" and physics_tick == 1
        )
        return _observation(physics_tick, fault, invalid_pair=invalid)


class FakeAdapter:
    physics_dt_s = 1.0 / 120.0

    def __init__(self, runtime):
        self.runtime = runtime
        self.write_count = 0
        self.standing_pose_deg = {name: 0.0 for name in SERVO_NAMES}
        self.servo_target_mapper = SimpleNamespace(servo_rate_deg_s=150.0)

    def apply_full12(
        self,
        command,
        *,
        physics_tick,
        tracking_servo_names,
        drive_feedback_bias_full12,
    ):
        action = tuple(float(value) for value in command)
        bias = tuple(float(value) for value in drive_feedback_bias_full12)
        self.write_count += 1
        drive_target = tuple(
            value + offset for value, offset in zip(action, bias, strict=True)
        )
        self.runtime.events.append(
            (
                "adapter.apply",
                physics_tick,
                action,
                tuple(tracking_servo_names),
                bias,
            )
        )
        return {
            "physics_tick": physics_tick,
            "write_count": self.write_count,
            "articulation_writes_this_call": 1,
            "motion_start_skew_s": 0.0,
            "applied_full12": action,
            "drive_target_full12": drive_target,
            "drive_feedback_bias_requested_full12": bias,
        }

    def update_readback(self):
        self.runtime.events.append(("adapter.update",))

    def verify_authoritative_servo_limits_adopted(self):
        self.runtime.events.append(("adapter.verify_limits",))

    def joint_limit_initialization_evidence(self):
        return {
            "schema": "fake.limit.initialization.v1",
            "all_eight_servo_limits_applied": True,
            "source_asset_modified": False,
            "stage_saved": False,
        }


class FakeSimulation:
    def __init__(self, runtime):
        self.runtime = runtime
        self.step_count = 0

    def step(self, *, render):
        self.step_count += 1
        self.runtime.events.append(("sim.step", render))

    def render(self):
        self.runtime.events.append(("sim.render",))


class FakeRuntime:
    def __init__(self, *, fault=None, invalid_pair=False, strict_reader_clock=False):
        self.fault = fault
        self.invalid_pair = invalid_pair
        self.strict_reader_clock = strict_reader_clock
        self.events = []
        self.reader_count = 0
        self.reset_scene_count = 0
        self.adapter_create_count = 0
        self.sim = FakeSimulation(self)
        self.scene = SimpleNamespace(
            sim=self.sim,
            robot=SimpleNamespace(),
            instrumentation=SimpleNamespace(
                contact_backend=SimpleNamespace(initialized=True)
            ),
            app_is_running=lambda: True,
        )
        self.adapter = None
        self.controller = None
        self.live_readers = []
        self.all_readers = []

    def dependencies(self):
        def create_scene(**kwargs):
            self.events.append(("create_scene", tuple(sorted(kwargs))))
            hook = kwargs["before_reset"]
            self.scene.instrumentation = hook(self.scene.sim, self.scene.robot)
            return self.scene

        def create_sensing_backends(**kwargs):
            self.events.append(("create_sensing_backends", tuple(sorted(kwargs))))
            return SimpleNamespace(contact_backend=SimpleNamespace(initialized=True))

        def adapter_from_scene(scene):
            self.adapter_create_count += 1
            self.events.append(("adapter_from_scene", self.adapter_create_count))
            self.adapter = FakeAdapter(self)
            return self.adapter

        def reader_from_scene(scene, adapter, backends):
            role = "calibration" if self.reader_count % 2 == 0 else "live"
            self.reader_count += 1
            reader = FakeReader(self, role)
            self.all_readers.append(reader)
            if role == "live":
                self.live_readers.append(reader)
            return reader

        def controller_from_paths(fsm_path, contract_path):
            self.controller = FakeController(self)
            return self.controller

        canonical_reset_state = SimpleNamespace(
            instance_count=1, state_sha256="fake-canonical-reset-sha256"
        )

        def capture_reset_state(scene):
            assert scene is self.scene
            self.events.append(("capture_reset_state", canonical_reset_state))
            return canonical_reset_state

        def reset_scene(scene, reset_state):
            assert reset_state is None or reset_state is canonical_reset_state
            self.reset_scene_count += 1
            self.events.append(("reset_scene", reset_state))
            return {
                "root_pose_writes": 0,
                "root_velocity_writes": 0,
                "joint_state_writes": 0,
                "global_simulation_resets": 1,
                "simulation_forward_syncs": 0,
                "physics_lifecycle_reset": "hard_stop_play",
                "reset_contact_sensor_count": 13,
            }

        def restore_settled_state(scene, reset_state):
            assert scene is self.scene
            assert reset_state is canonical_reset_state
            self.events.append(("restore_settled_state", reset_state))
            return {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "global_simulation_resets": 0,
                "simulation_forward_syncs": 1,
                "canonical_settled_restore_applied": True,
                "canonical_settled_applied_sha256": reset_state.state_sha256,
            }

        def load_phase_snapshot(phase):
            payload = _snapshot_payload(phase)
            self.events.append(("load_phase_snapshot", phase))
            return LoadedPhaseSnapshot(
                phase_id=phase,
                payload=payload,
                state_sha256="fake-state-sha256",
                file_sha256="fake-file-sha256",
                snapshot_path=Path("fake/snapshot.json"),
            )

        def write_phase_snapshot(scene, adapter, snapshot):
            self.events.append(("write_phase_snapshot", snapshot["fsm_state"]))
            return {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "global_simulation_resets": 0,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
                "verified": True,
            }

        def restore_guard_snapshot(reader, snapshot):
            self.events.append(("restore_guard_snapshot", snapshot["fsm_state"]))
            reader.restored_latches = snapshot["contact_event_latches"]
            return {"history_is_independent": True, "restored": True}

        def restore_controller_snapshot(controller, snapshot):
            phase = snapshot["fsm_state"]
            self.events.append(("restore_controller_snapshot", phase))
            controller.phase.state_id = phase
            controller.phase.macro_phase = int(phase[1:])
            controller.motion.phase = controller.phase
            controller.restored_history = tuple(snapshot["phase_history"])
            return {
                "state_id": phase,
                "lifecycle": "EXECUTE_MOTION",
                "history_is_independent": True,
            }

        return BackendDependencies(
            create_scene=create_scene,
            create_sensing_backends=create_sensing_backends,
            adapter_from_scene=adapter_from_scene,
            reader_from_scene=reader_from_scene,
            controller_from_paths=controller_from_paths,
            capture_reset_state=capture_reset_state,
            reset_scene=reset_scene,
            restore_settled_state=restore_settled_state,
            locked_scene_snapshot=lambda: {
                "schema": "fake.locked_scene.v1",
                "physics": {"dt_s": 1.0 / 120.0},
            },
            expected_contact_bodies=BODY_NAMES,
            robot_asset_hash="fake-robot-sha256",
            load_phase_snapshot=load_phase_snapshot,
            write_phase_snapshot=write_phase_snapshot,
            restore_controller_snapshot=restore_controller_snapshot,
            restore_guard_snapshot=restore_guard_snapshot,
        )


def _backend(runtime):
    return IsaacFSMBackend(dependencies=runtime.dependencies())


def test_reset_settles_and_step_preserves_atomic_order_and_drive_feedback() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)

    initial = backend.reset(seed=7, options={"randomization_enabled": False})

    assert runtime.sim.step_count == SETTLE_TICKS
    assert runtime.adapter.write_count == SETTLE_TICKS
    assert sum(row[0] == "reader.calibration" for row in runtime.events) == LEVEL_CALIBRATION_TICKS
    assert initial.physics_tick == 0
    assert initial.sim_time_s == 0.0
    assert initial.info["settle_atomic_full12_writes"] == SETTLE_TICKS
    assert initial.info["level_calibration_sample_count"] == LEVEL_CALIBRATION_TICKS
    assert initial.info["raw_observation"] is backend.raw_observation
    assert initial.info["raw_controller_frame"] is backend.controller_frame
    assert initial.info["level_calibration"]["raw_pitch_rad"] == pytest.approx(0.2)
    assert initial.info["level_calibration"]["pitch_error_to_level_rad"] == pytest.approx(0.0)

    runtime.events.clear()
    action = (0.5,) + ZERO[1:]
    following = backend.step_physics(action)

    assert [row[0] for row in runtime.events] == [
        "adapter.apply",
        "sim.step",
        "adapter.update",
        "reader.live",
        "controller.step",
    ]
    apply_event = runtime.events[0]
    assert apply_event[1] == SETTLE_TICKS
    # PPO residuals are composed after the mature nominal mapper.  Presenting
    # them as new nominal targets would clear frozen tracking compensation on
    # every 15 Hz policy update.
    assert apply_event[2] == ZERO
    assert apply_event[3] == ("front_left_hip",)
    assert apply_event[4][0] == pytest.approx(0.8)
    assert runtime.live_readers[-1].last_command[0] == pytest.approx(0.8)
    assert following.physics_tick == 1
    assert following.sim_time_s == pytest.approx(1.0 / 120.0)
    assert following.observation.previous_action_full12 == action
    assert following.reward_signals.forward_progress_delta_m == pytest.approx(0.001)
    assert following.info["in_episode_root_pose_writes"] == 0
    assert following.info["in_episode_root_velocity_writes"] == 0
    assert following.info["in_episode_force_or_impulse_writes"] == 0
    assert following.info["in_episode_gravity_writes"] == 0
    assert following.info["recording_accesses"] == 0
    ack = following.info["atomic_ack"]
    assert ack["ppo_actuation_contract"] == (
        "frozen_nominal_plus_post_mapper_residual.v1"
    )
    assert ack["fsm_nominal_mapper_input_full12"] == list(ZERO)
    assert ack["ppo_projected_applied_full12"] == list(action)
    assert ack["ppo_projected_residual_full12"] == list(action)
    assert ack["controller_drive_bias_full12"][0] == pytest.approx(0.3)
    assert ack["combined_post_mapper_bias_full12"][0] == pytest.approx(0.8)


def test_residual_actuation_plan_preserves_nominal_mapper_input() -> None:
    nominal = (9.0, -22.9) + ZERO[2:]
    projected = (9.25, -22.8) + ZERO[2:]
    feedback = (0.5, -0.25) + ZERO[2:]
    normal = (0.1, 0.2) + ZERO[2:]

    plan = build_residual_actuation_plan(
        projected,
        frozen_nominal_full12=nominal,
        drive_feedback_bias_full12=feedback,
        normal_drive_bias_full12=normal,
    )

    assert plan.frozen_nominal_full12 == nominal
    assert plan.projected_residual_full12[:2] == pytest.approx((0.25, 0.1))
    assert plan.controller_drive_bias_full12[:2] == pytest.approx((0.6, -0.05))
    assert plan.combined_post_mapper_bias_full12[:2] == pytest.approx((0.85, 0.05))


def test_invalid_exact_pair_fails_closed_after_one_physics_step() -> None:
    runtime = FakeRuntime(invalid_pair=True)
    backend = _backend(runtime)
    backend.reset(seed=3, options={})

    with pytest.raises(SensorContractFailure, match="exact ground pair is unverified"):
        backend.step_physics(ZERO)

    assert runtime.adapter.write_count == SETTLE_TICKS + 1
    assert runtime.sim.step_count == SETTLE_TICKS + 1
    assert len(runtime.controller.abort_calls) == 1
    # The invalid observation is never offered to the nominal controller as a
    # normal tick; its explicit infrastructure-abort API receives the failure.
    assert runtime.controller.physics_tick == 1


@pytest.mark.parametrize(
    ("fault", "field"),
    [
        ("body", "body_collision"),
        ("wheel", "wheel_only_climb"),
        ("nan", "nan_inf"),
        ("joint", "hard_joint_limit"),
        ("fall", "fall"),
        ("explosion", "physics_explosion"),
        ("success", "success"),
        ("blocked", "timeout"),
    ],
)
def test_authoritative_termination_sources_are_mapped(fault, field) -> None:
    runtime = FakeRuntime(fault=fault)
    backend = _backend(runtime)
    backend.reset(seed=11, options={})

    frame = backend.step_physics(ZERO)

    assert getattr(frame.termination_signals, field) is True
    if fault == "nan":
        assert frame.info["actor_observation_fallback_due_to_nonfinite"] is True
    if fault == "blocked":
        assert frame.info["termination_mapping"]["controller_result"] == (
            "INCOMPLETE_CONTROLLER_BLOCKED"
        )
        assert frame.info["termination_mapping"][
            "controller_blocked_encoded_as_truncation"
        ] is True
    with pytest.raises(IsaacFSMBackendError, match="after authoritative termination"):
        backend.step_physics(ZERO)


def test_each_episode_uses_the_same_hard_physics_lifecycle_reset() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    first = backend.reset(seed=1, options={})
    second = backend.reset(seed=2, options={})

    assert runtime.reset_scene_count == 2
    assert runtime.adapter_create_count == 3
    assert runtime.sim.step_count == 2 * SETTLE_TICKS
    assert first.info["reset_global_simulation_resets"] == 1
    assert second.info["reset_global_simulation_resets"] == 1
    assert first.info["reset_root_pose_writes"] == 0
    assert second.info["reset_root_pose_writes"] == 0
    assert second.info["reset_root_velocity_writes"] == 0
    assert second.info["reset_joint_state_writes"] == 0
    assert second.info["reset_simulation_forward_syncs"] == 0
    assert first.info["canonical_reset_restore_applied"] is False
    assert second.info["canonical_reset_restore_applied"] is False
    assert first.info["hard_reset_native_state_matches_canonical"] is True
    assert second.info["hard_reset_native_state_matches_canonical"] is True
    assert second.info["canonical_settled_restore_applied"] is False
    assert second.info["in_episode_root_pose_writes"] == 0
    assert sum(event[0] == "capture_reset_state" for event in runtime.events) == 4
    assert sum(event[0] == "restore_settled_state" for event in runtime.events) == 0


@pytest.mark.parametrize("mismatch", ["sha256", "instance_count"])
def test_reused_hard_reset_native_mismatch_fails_before_settle_step(
    mismatch: str,
) -> None:
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    capture_reset_state = dependencies.capture_reset_state
    capture_count = 0

    def capture_with_reuse_pre_settle_mismatch(scene):
        nonlocal capture_count
        capture_count += 1
        state = capture_reset_state(scene)
        if capture_count != 3:
            return state
        return SimpleNamespace(
            instance_count=(
                state.instance_count + 1
                if mismatch == "instance_count"
                else state.instance_count
            ),
            state_sha256=(
                "mismatched-pre-settle-sha256"
                if mismatch == "sha256"
                else state.state_sha256
            ),
        )

    backend = IsaacFSMBackend(
        dependencies=replace(
            dependencies,
            capture_reset_state=capture_with_reuse_pre_settle_mismatch,
        )
    )
    backend.reset(seed=1, options={})
    steps_before_reuse = runtime.sim.step_count
    runtime.events.clear()

    with pytest.raises(
        IsaacFSMBackendError,
        match="did not reproduce the canonical pre-settle state",
    ):
        backend.reset(seed=2, options={})

    assert runtime.sim.step_count == steps_before_reuse
    assert not any(event[0] == "sim.step" for event in runtime.events)
    assert not any(event[0] == "controller.step" for event in runtime.events)


@pytest.mark.parametrize("mismatch", ["sha256", "level_reference"])
def test_reused_natural_settle_mismatch_fails_before_episode_tick_zero(
    mismatch: str,
) -> None:
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    capture_reset_state = dependencies.capture_reset_state
    capture_count = 0

    def capture_with_reuse_natural_settle_mismatch(scene):
        nonlocal capture_count
        capture_count += 1
        state = capture_reset_state(scene)
        if mismatch != "sha256" or capture_count != 4:
            return state
        return SimpleNamespace(
            instance_count=state.instance_count,
            state_sha256="mismatched-natural-settle-sha256",
        )

    backend = IsaacFSMBackend(
        dependencies=replace(
            dependencies,
            capture_reset_state=capture_with_reuse_natural_settle_mismatch,
        )
    )
    backend.reset(seed=1, options={})
    first_controller = runtime.controller
    first_controller_tick = first_controller.physics_tick
    if mismatch == "level_reference":
        backend._canonical_level_reference_orientation = (1.0, 0.0, 0.0, 0.0)
    steps_before_reuse = runtime.sim.step_count
    runtime.events.clear()

    expected = (
        "did not reproduce the canonical natural-settle state"
        if mismatch == "sha256"
        else "did not reproduce the level calibration"
    )
    with pytest.raises(IsaacFSMBackendError, match=expected):
        backend.reset(seed=2, options={})

    assert runtime.sim.step_count == steps_before_reuse + SETTLE_TICKS
    assert runtime.controller is first_controller
    assert first_controller.physics_tick == first_controller_tick
    assert not any(event[0] == "reader.live" for event in runtime.events)
    assert not any(event[0] == "controller.step" for event in runtime.events)


def test_canonical_reset_uses_usd_authored_live_pose_not_zero_default_cache() -> None:
    import torch

    authored_root = torch.tensor(
        [[0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    authored_joint = torch.tensor(
        [[0.08, 0.03, -0.01, 0.13, -0.18, 0.16, -0.32, 0.22, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    authored_velocity = torch.zeros_like(authored_joint)

    class FakeRobot:
        def __init__(self) -> None:
            self.data = SimpleNamespace(
                root_state_w=authored_root.clone(),
                joint_pos=authored_joint.clone(),
                joint_vel=authored_velocity.clone(),
                # This deliberately reproduces the Isaac Lab cache mismatch:
                # joint_pos={} creates zeros even though the USD pose is not zero.
                default_joint_pos=torch.zeros_like(authored_joint),
            )
            self.writes = {}
            self.reset_count = 0

        def write_root_pose_to_sim(self, value):
            self.writes["root_pose"] = value.clone()

        def write_root_velocity_to_sim(self, value):
            self.writes["root_velocity"] = value.clone()

        def write_joint_state_to_sim(self, position, velocity):
            self.writes["joint_position"] = position.clone()
            self.writes["joint_velocity"] = velocity.clone()

        def reset(self):
            self.reset_count += 1

    robot = FakeRobot()
    canonical = capture_canonical_articulation_reset_state(robot)
    restore_canonical_articulation_reset_state(
        robot, canonical, expected_instance_count=1
    )

    assert canonical.instance_count == 1
    assert canonical.state_sha256
    assert torch.equal(robot.writes["joint_position"], authored_joint)
    assert not torch.equal(
        robot.writes["joint_position"], robot.data.default_joint_pos
    )
    assert torch.equal(robot.writes["root_pose"], authored_root[:, :7])
    assert torch.equal(robot.writes["root_velocity"], authored_root[:, 7:])
    assert robot.reset_count == 1

    canonical.joint_position[0, 0] += 1.0
    with pytest.raises(IsaacFSMBackendError, match="changed after capture"):
        restore_canonical_articulation_reset_state(
            robot, canonical, expected_instance_count=1
        )


def test_phase_snapshot_reset_restores_independent_phase_state_and_proves_live_state() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)

    frame = backend.reset(seed=91, options={"training_phase_snapshot": "P09"})

    assert frame.state_id == "P09"
    assert frame.macro_phase == 9
    assert frame.info["training_phase_snapshot"] == "P09"
    restoration = frame.info["phase_snapshot_restoration"]
    assert restoration["snapshot_validated"] is True
    assert restoration["mode"] == "phase_entry_snapshot"
    assert restoration["live_observation"]["verified"] is True
    assert restoration["controller_state"]["history_is_independent"] is True
    assert restoration["guard_state"]["history_is_independent"] is True
    assert frame.info["reset_root_pose_writes"] == 1
    assert frame.info["reset_root_velocity_writes"] == 1
    assert frame.info["reset_joint_state_writes"] == 1
    assert frame.info["reset_simulation_forward_syncs"] == 1
    event_names = [row[0] for row in runtime.events]
    assert event_names.index("write_phase_snapshot") < event_names.index(
        "restore_guard_snapshot"
    )
    assert event_names.index("restore_guard_snapshot") < event_names.index(
        "reader.live"
    )
    assert event_names.index("restore_controller_snapshot") < event_names.index(
        "controller.step"
    )

    runtime.events.clear()
    following = backend.step_physics(ZERO)
    assert following.state_id == "P09"
    assert following.info["in_episode_root_pose_writes"] == 0
    assert all("snapshot" not in row[0] for row in runtime.events)


def test_reused_phase_snapshot_reports_hard_reset_plus_snapshot_write() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)

    backend.reset(seed=91, options={"training_phase_snapshot": "P09"})
    frame = backend.reset(seed=92, options={"training_phase_snapshot": "P09"})

    assert frame.info["reset_count"] == 2
    assert frame.info["reset_global_simulation_resets"] == 1
    assert frame.info["reset_simulation_forward_syncs"] == 1
    assert frame.info["reset_root_pose_writes"] == 1
    assert frame.info["reset_root_velocity_writes"] == 1
    assert frame.info["reset_joint_state_writes"] == 1


def test_explicit_p01_snapshot_keeps_the_normal_p01_reset_path() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)

    frame = backend.reset(seed=4, options={"training_phase_snapshot": "P01"})

    assert frame.state_id == "P01"
    assert frame.info["phase_snapshot_restoration"]["snapshot_validated"] is True
    assert frame.info["phase_snapshot_restoration"]["mode"] == "normal_p01_reset"
    assert frame.info["reset_root_pose_writes"] == 0
    assert not any(row[0] == "write_phase_snapshot" for row in runtime.events)
    assert not any(row[0] == "restore_controller_snapshot" for row in runtime.events)


def test_phase_snapshot_fails_before_scene_mutation_without_complete_restore_seams() -> None:
    runtime = FakeRuntime()
    dependencies = replace(runtime.dependencies(), restore_guard_snapshot=None)
    backend = IsaacFSMBackend(dependencies=dependencies)

    with pytest.raises(IsaacFSMBackendError, match="restoration seams"):
        backend.reset(seed=8, options={"training_phase_snapshot": "P08"})

    assert runtime.sim.step_count == 0
    assert not any(row[0] == "create_scene" for row in runtime.events)


@pytest.mark.parametrize("phase", ["P00", "P14", "p09", 9])
def test_phase_snapshot_option_requires_an_exact_p01_to_p13_id(phase) -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    with pytest.raises(IsaacFSMBackendError, match="P01 through P13"):
        backend.reset(seed=1, options={"training_phase_snapshot": phase})
    assert runtime.sim.step_count == 0


def test_all_checked_in_phase_snapshots_pass_backend_proof_validation() -> None:
    loaded = tuple(_load_validated_phase_snapshot(phase) for phase in (
        f"P{index:02d}" for index in range(1, 14)
    ))
    assert tuple(row.phase_id for row in loaded) == tuple(
        f"P{index:02d}" for index in range(1, 14)
    )
    assert all("reference/ppo_phase_snapshots" in row.snapshot_path.as_posix() for row in loaded)


def test_real_frozen_controller_and_guard_tracker_restore_from_checked_snapshot() -> None:
    from wlr50_clean.fsm.controller import SensorFsmController
    from wlr50_clean.sensing.contact_classifier import ContactClassifier
    from wlr50_clean.sensing.guard_state import LiveGuardTracker

    loaded = _load_validated_phase_snapshot("P09")
    controller = SensorFsmController.from_paths(
        DEFAULT_FSM_PATH, DEFAULT_MOTION_CONTRACT_PATH
    )
    controller_proof = _restore_controller_from_snapshot(controller, loaded.payload)
    assert controller_proof["state_id"] == "P09"
    assert controller_proof["lifecycle"] == "EXECUTE_MOTION"
    assert controller.motion._tick_index == 0
    assert controller._ppo_restored_phase_history == tuple(
        f"P{index:02d}" for index in range(1, 9)
    )
    assert controller.history == []

    reader = SimpleNamespace(
        guard_tracker=LiveGuardTracker(),
        contact_classifier=ContactClassifier(),
    )
    guard_proof = _restore_guard_tracker_from_snapshot(reader, loaded.payload)
    assert guard_proof["active_lift"]["FR"] is True
    assert guard_proof["active_lift"]["FL"] is True
    assert guard_proof["active_lift"]["RR"] is False
    assert guard_proof["classifier_wheel_pairs_restored"] == 8


def test_snapshot_live_state_mismatch_fails_closed_before_first_controller_tick() -> None:
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    payload = _snapshot_payload("P08")
    payload["root_state"]["position_w_m"][0] = 0.25
    mismatched = LoadedPhaseSnapshot(
        phase_id="P08",
        payload=payload,
        state_sha256="fake-state-sha256",
        file_sha256="fake-file-sha256",
        snapshot_path=Path("fake/P08/snapshot.json"),
    )
    dependencies = replace(
        dependencies,
        load_phase_snapshot=lambda phase: mismatched,
    )
    backend = IsaacFSMBackend(dependencies=dependencies)

    with pytest.raises(SensorContractFailure, match="could not be proven"):
        backend.reset(seed=81, options={"training_phase_snapshot": "P08"})

    assert runtime.controller.physics_tick == 0
    assert not any(row[0] == "controller.step" for row in runtime.events)


def test_invalid_action_and_prohibited_options_fail_before_episode_mutation() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    with pytest.raises(IsaacFSMBackendError, match="prohibited"):
        backend.reset(seed=1, options={"recording_path": "forbidden.jsonl"})
    assert runtime.sim.step_count == 0

    backend.reset(seed=1, options={})
    writes = runtime.adapter.write_count
    steps = runtime.sim.step_count
    with pytest.raises(IsaacFSMBackendError, match="twelve finite"):
        backend.step_physics((0.0,) * 11)
    assert runtime.adapter.write_count == writes
    assert runtime.sim.step_count == steps


def test_video_hold_ticks_are_real_physics_and_outside_the_fsm_clock() -> None:
    runtime = FakeRuntime(fault="success")
    backend = _backend(runtime)
    initial = backend.reset(seed=17, options={})

    first = backend.advance_video_pre_action_tick()
    second = backend.advance_video_pre_action_tick()
    backend.render_video_frame()
    assert first["kind"] == "pre_action"
    assert second["physical_tick"] == first["physical_tick"] + 1
    assert initial.physics_tick == 0
    assert backend.controller_frame.physics_tick == 0

    terminal = backend.step_physics(ZERO)
    assert terminal.physics_tick == 1
    assert terminal.termination_signals.success is True
    post = backend.advance_video_post_success_tick()
    assert post["kind"] == "post_success"
    assert post["body_collision"] is False
    assert post["wheel_only_climb"] is False
    assert runtime.sim.step_count == SETTLE_TICKS + 2 + 1 + 1
    assert ("sim.render",) in runtime.events

    apply_ticks = [
        event[1] for event in runtime.events if event[0] == "adapter.apply"
    ]
    assert apply_ticks[-4:] == [
        SETTLE_TICKS,
        SETTLE_TICKS + 1,
        SETTLE_TICKS + 2,
        SETTLE_TICKS + 3,
    ]


def test_video_preroll_restarts_sensor_clock_without_a_second_physics_reset() -> None:
    runtime = FakeRuntime(strict_reader_clock=True)
    backend = _backend(runtime)
    backend.reset(seed=170, options={})
    original_episode_reader = runtime.all_readers[-1]

    backend.advance_video_pre_action_tick()
    backend.advance_video_pre_action_tick()
    refreshed = backend.refresh_video_pre_action_frame()
    refreshed_episode_reader = runtime.all_readers[-1]

    assert original_episode_reader.ticks == [0, 1, 2]
    assert refreshed_episode_reader is not original_episode_reader
    assert refreshed_episode_reader.ticks == [0]
    assert refreshed.info["video_pre_action_refresh"] == {
        "schema": "wlr50_clean.ppo_video_pre_action_refresh.v1",
        "physical_pre_action_ticks": 2,
        "pre_roll_reader_last_logical_tick": 2,
        "episode_reader_reinitialized": True,
        "episode_reader_first_logical_tick": 0,
        "controller_frame_preserved": True,
        "controller_logical_tick": 0,
        "simulation_reset_performed": False,
        "fsm_step_performed": False,
    }
    # Exactly the reset that began the episode; the video refresh adds none.
    assert runtime.reset_scene_count == 1

    stepped = backend.step_physics(ZERO)
    assert stepped.physics_tick == 1
    assert refreshed_episode_reader.ticks == [0, 1]


def test_video_hold_hooks_fail_closed_in_the_wrong_lifecycle() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    with pytest.raises(IsaacFSMBackendError, match="reset must precede"):
        backend.advance_video_pre_action_tick()
    backend.reset(seed=18, options={})
    with pytest.raises(IsaacFSMBackendError, match="authoritative task success"):
        backend.advance_video_post_success_tick()
    backend.step_physics(ZERO)
    with pytest.raises(IsaacFSMBackendError, match="before the first episode tick"):
        backend.advance_video_pre_action_tick()
