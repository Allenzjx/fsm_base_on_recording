from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import wlr50_clean.ppo.isaac_fsm_backend as backend_module
from wlr50_clean.ppo.isaac_fsm_backend import (
    DEFAULT_FSM_PATH,
    DEFAULT_MOTION_CONTRACT_PATH,
    LEVEL_CALIBRATION_TICKS,
    PHASE_SNAPSHOT_PRIME_PHYSICS_STEPS,
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
    _reset_physics_lifecycle,
    _restore_controller_from_snapshot,
    _restore_guard_tracker_from_snapshot,
    _write_phase_snapshot_state,
)
from wlr50_clean.ppo.phase_snapshots import (
    SNAPSHOT_SCHEMA,
    SOURCE_COMMAND_SCHEMA,
    SOURCE_MAPPER_STATE_SCHEMA,
    phase_snapshot_actuation_contract_sha256,
    phase_snapshot_drive_target_sha256,
)
from wlr50_clean.ppo.phase_effective_entry import EffectivePhaseEntryError


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


class FakeEffectiveEntryContract:
    contract_sha256 = "2" * 64
    phase_snapshot_bundle_sha256 = "3" * 64

    def entry(self, phase: str):
        if phase == "P01":
            raise EffectivePhaseEntryError("P01 is not calibrated")
        if phase not in {f"P{index:02d}" for index in range(2, 14)}:
            raise EffectivePhaseEntryError(f"missing phase {phase}")
        return {
            "schema": "fake.phase_effective_entry.v1",
            "phase": phase,
            "entry_sha256": "4" * 64,
        }

    def as_record(self):
        return {
            "schema": "fake.phase_effective_entry_contract_record.v1",
            "contract_sha256": self.contract_sha256,
            "phase_snapshot_bundle_sha256": self.phase_snapshot_bundle_sha256,
            "phase_count": 12,
        }


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
                pair_verified=not (invalid_pair and index == 0),
                active=True,
                source="isaaclab.ContactSensor.force_matrix_w",
                force_w_n=(0.0, 0.0, 1.0),
            ),
            obstacle=SimpleNamespace(
                pair_verified=True,
                active=False,
                source="isaaclab.ContactSensor.force_matrix_w",
                force_w_n=(0.0, 0.0, 0.0),
            ),
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
    expected_ack = {
        "schema": "wlr50_clean.atomic_full12_ack.v1",
        "physics_dt_s": 1.0 / 120.0,
        "articulation_writes_this_call": 1,
        "canonical_order": list(SERVO_NAMES + WHEEL_NAMES),
        "requested_full12": [0.0] * 12,
        "applied_full12": [0.0] * 12,
        "drive_target_full12": [0.0] * 12,
        "native_drive_target_full12": [0.0] * 12,
        "drive_feedback_bias_requested_full12": [0.0] * 12,
        "drive_feedback_bias_realized_full12": [0.0] * 12,
        "drive_feedback_final_slew_limit_deg_per_tick": 1.25,
        "command_was_clamped": False,
        "servo_applied_drive_command_deg": [0.0] * 8,
        "servo_native_drive_command_deg": [0.0] * 8,
        "servo_tracking_compensation_deg": [0.0] * 8,
        "servo_nominal_target_reached": [True] * 8,
        "servo_tracking_active": [False] * 8,
        "tracking_servo_names": [],
        "servo_tracking_feedback_sample_tick": 280,
        "servo_tracking_feedback_sampled": False,
        "servo_joint_ids": list(range(8)),
        "wheel_joint_ids": list(range(8, 12)),
        "servo_target_physical_rad": [0.0] * 8,
        "wheel_target_physical_rad_s": [-0.0, 0.0, -0.0, 0.0],
        "motion_start_skew_s": 0.0,
    }
    pre_state = {
        "schema": SOURCE_MAPPER_STATE_SCHEMA,
        "source_control_physics_tick": 99,
        "requested_servo_deg": [0.0] * 8,
        "applied_drive_command_deg": [0.0] * 8,
        "nominal_target_reached": [True] * 8,
        "tracking_compensation_deg": [0.0] * 8,
        "tracking_active": [False] * 8,
        "retiring_stale_bias": [False] * 8,
        "feedback_tick": 280,
        "final_drive_servo_deg": [0.0] * 8,
    }
    post_state = {
        **pre_state,
        "source_control_physics_tick": 100,
        "feedback_tick": 281,
    }
    source_command = {
        "schema": SOURCE_COMMAND_SCHEMA,
        "control_physics_tick": 100,
        "source_atomic_physics_tick": 280,
        "source_atomic_write_count": 281,
        "adapter_input": {
            "requested_full12": [0.0] * 12,
            "tracking_servo_names": [],
            "drive_feedback_bias_requested_full12": [0.0] * 12,
        },
        "mapper_configuration": {
            "physics_dt_s": 1.0 / 120.0,
            "servo_rate_deg_s": 150.0,
            "maximum_delta_deg": 1.25,
            "tracking_gain": 8.0,
            "tracking_limit_deg": 10.0,
            "feedback_interval_ticks": 4,
            "standing_pose_deg": [0.0] * 8,
        },
        "mapper_pre_state": pre_state,
        "mapper_post_state": post_state,
        "expected_atomic_ack": expected_ack,
        "source_command_row_canonical_sha256": "c" * 64,
        "source_observation_row_canonical_sha256": "d" * 64,
        "drive_target_full12_sha256": phase_snapshot_drive_target_sha256(
            expected_ack["drive_target_full12"]
        ),
        "actuation_contract_sha256": phase_snapshot_actuation_contract_sha256(
            expected_ack
        ),
    }
    return {
        "schema": SNAPSHOT_SCHEMA,
        "reset_use": "TRAINING_RESET_STATE_WRITE",
        "in_episode_root_write": "FORBIDDEN_IN_EPISODE_ROOT_WRITE",
        "source_tick": 100,
        "source_time_s": 100 / 120.0,
        "source_artifacts": {
            "trial_manifest": {
                "name": "trial_manifest.json",
                "bytes": 1,
                "sha256": "e" * 64,
            },
            "command": {
                "name": "full12_commands_120hz.jsonl",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "observation": {
                "name": "observation_120hz.jsonl",
                "bytes": 1,
                "sha256": "b" * 64,
            },
            "transition": {
                "name": "state_transitions.jsonl",
                "bytes": 1,
                "sha256": "f" * 64,
            },
            "leg_crossing": {
                "name": "leg_crossing_events.jsonl",
                "bytes": 1,
                "sha256": "1" * 64,
            },
        },
        "source_command": source_command,
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
        self.state = SimpleNamespace(
            entry_guards=tuple(
                SimpleNamespace(name=name)
                for name in (
                    "previous_state_done",
                    "no_body_obstacle_collision",
                    "joint_hard_limits_valid",
                    "reference_entry_compatible",
                    "critical_actuators_available",
                )
            )
        )
        self.motion = FakeMotion(self.phase)
        self.physics_tick = 0
        self.lifecycle = _value("EXECUTE_MOTION")
        self.termination = None
        self.abort_calls = []
        self._pending_blocker = None

    def step(self, observation, *, sim_time_s):
        tick = self.physics_tick
        self.runtime.events.append(("controller.step", tick, sim_time_s))
        self.motion._tick_index = tick + 1
        events = ()
        if self.lifecycle.value == "WAIT_ENTRY":
            guard_rows = []
            for guard in self.state.entry_guards:
                value = True
                passed = not (
                    self.runtime.entry_guard_failure
                    and guard.name == "previous_state_done"
                )
                if (
                    self.phase.state_id == "P10"
                    and guard.name == "reference_entry_compatible"
                ):
                    value = {
                        "rear_right_knee_velocity": {
                            "actual_deg_s": self.runtime.p10_entry_velocity_deg_s,
                            "signed_positive_rebound_required": True,
                        }
                    }
                guard_rows.append(
                    {
                        "name": guard.name,
                        "passed": passed,
                        "value": value,
                        "source": "fake.live.entry_guard",
                        "reason": "passed",
                    }
                )
                if not passed:
                    self._pending_blocker = SimpleNamespace(
                        name=guard.name,
                        passed=False,
                        value=value,
                        source="fake.live.entry_guard",
                        reason="failed",
                    )
            events = (
                SimpleNamespace(
                    state_id=self.phase.state_id,
                    from_lifecycle="WAIT_ENTRY",
                    to_lifecycle="EXECUTE_MOTION",
                    reason="all live entry guards passed",
                    details={"guards": tuple(guard_rows)},
                ),
            )
            self.lifecycle = _value("EXECUTE_MOTION")
        termination = None
        if tick == 0 and self.runtime.entry_controller_result is not None:
            termination = SimpleNamespace(
                result=_value(self.runtime.entry_controller_result),
                reason="invalid tick-zero terminal result",
            )
            self.termination = termination
        elif tick == 1 and self.runtime.fault == "success":
            termination = SimpleNamespace(result=_value("SUCCESS"), reason="P13 guards")
        elif tick == 1 and self.runtime.fault == "blocked":
            termination = SimpleNamespace(
                result=_value("INCOMPLETE_CONTROLLER_BLOCKED"), reason="watchdog"
            )
        frame = SimpleNamespace(
            physics_tick=tick,
            sim_time_s=sim_time_s,
            state_id=self.phase.state_id,
            lifecycle=(
                _value(self.runtime.entry_frame_lifecycle)
                if tick == 0 and self.runtime.entry_frame_lifecycle is not None
                else self.lifecycle
            ),
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
            first_blocker=(
                {"name": "fake_entry_blocker"}
                if tick == 0 and self.runtime.entry_first_blocker
                else None
            ),
            events=events,
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
        self.contact_classifier = SimpleNamespace(
            force_on_n=0.25,
            force_off_n=0.12,
            history_length=3,
            _states={},
            _history={},
        )

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
            self.runtime.entry_fault
            if self.role == "live" and physics_tick == 0
            else (
                self.runtime.fault
                if self.role == "live" and physics_tick == 1
                else None
            )
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
        self.last_physics_tick = None
        self.standing_pose_deg = {name: 0.0 for name in SERVO_NAMES}
        self.servo_target_mapper = SimpleNamespace(
            physics_dt_s=1.0 / 120.0,
            servo_rate_deg_s=150.0,
            maximum_delta_deg=1.25,
            tracking_gain=8.0,
            tracking_limit_deg=10.0,
            feedback_interval_ticks=4,
            standing_pose_deg=dict(self.standing_pose_deg),
            _requested={name: 0.0 for name in SERVO_NAMES},
            _applied={name: 0.0 for name in SERVO_NAMES},
            _nominal_reached={name: True for name in SERVO_NAMES},
            _compensation={name: 0.0 for name in SERVO_NAMES},
            _tracking_active={name: False for name in SERVO_NAMES},
            _retiring_stale_bias={name: False for name in SERVO_NAMES},
            _feedback_tick=0,
        )
        self._final_drive_servo_deg = {name: 0.0 for name in SERVO_NAMES}

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
        if self.last_physics_tick is not None:
            assert physics_tick > self.last_physics_tick
        self.last_physics_tick = physics_tick
        sample_tick = self.servo_target_mapper._feedback_tick
        tracking = tuple(tracking_servo_names)
        for name, value in zip(SERVO_NAMES, action[:8], strict=True):
            self.servo_target_mapper._requested[name] = value
            self.servo_target_mapper._applied[name] = value
            self.servo_target_mapper._nominal_reached[name] = True
            self.servo_target_mapper._compensation[name] = 0.0
            self.servo_target_mapper._tracking_active[name] = name in tracking
            self.servo_target_mapper._retiring_stale_bias[name] = False
        self.servo_target_mapper._feedback_tick += 1
        self.write_count += 1
        drive_target = tuple(
            value + offset for value, offset in zip(action, bias, strict=True)
        )
        for name, value in zip(SERVO_NAMES, drive_target[:8], strict=True):
            self._final_drive_servo_deg[name] = value
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
            "schema": "wlr50_clean.atomic_full12_ack.v1",
            "physics_tick": physics_tick,
            "physics_dt_s": 1.0 / 120.0,
            "write_count": self.write_count,
            "articulation_writes_this_call": 1,
            "motion_start_skew_s": 0.0,
            "canonical_order": list(SERVO_NAMES + WHEEL_NAMES),
            "requested_full12": action,
            "applied_full12": action,
            "drive_target_full12": drive_target,
            "native_drive_target_full12": action,
            "drive_feedback_bias_requested_full12": bias,
            "drive_feedback_bias_realized_full12": bias,
            "drive_feedback_final_slew_limit_deg_per_tick": 1.25,
            "command_was_clamped": False,
            "servo_applied_drive_command_deg": drive_target[:8],
            "servo_native_drive_command_deg": action[:8],
            "servo_tracking_compensation_deg": (0.0,) * 8,
            "servo_nominal_target_reached": (True,) * 8,
            "servo_tracking_active": tuple(name in tracking for name in SERVO_NAMES),
            "tracking_servo_names": tracking,
            "servo_tracking_feedback_sample_tick": sample_tick,
            "servo_tracking_feedback_sampled": bool(tracking and sample_tick % 4 == 0),
            "servo_joint_ids": tuple(range(8)),
            "wheel_joint_ids": tuple(range(8, 12)),
            "servo_target_physical_rad": tuple(math.radians(value) for value in drive_target[:8]),
            "wheel_target_physical_rad_s": (
                -drive_target[8],
                drive_target[9],
                -drive_target[10],
                drive_target[11],
            ),
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
    def __init__(
        self,
        *,
        fault=None,
        invalid_pair=False,
        strict_reader_clock=False,
        effective_entry_error: str | None = None,
        entry_fault: str | None = None,
        entry_controller_result: str | None = None,
        entry_guard_failure: bool = False,
        p10_entry_velocity_deg_s: float = 1.0,
        entry_frame_lifecycle: str | None = None,
        entry_first_blocker: bool = False,
    ):
        self.fault = fault
        self.invalid_pair = invalid_pair
        self.strict_reader_clock = strict_reader_clock
        self.effective_entry_error = effective_entry_error
        self.entry_fault = entry_fault
        self.entry_controller_result = entry_controller_result
        self.entry_guard_failure = entry_guard_failure
        self.p10_entry_velocity_deg_s = p10_entry_velocity_deg_s
        self.entry_frame_lifecycle = entry_frame_lifecycle
        self.entry_first_blocker = entry_first_blocker
        self.events = []
        self.reader_count = 0
        self.reader_count_this_reset = 0
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
            role = "calibration" if self.reader_count_this_reset == 0 else "live"
            self.reader_count += 1
            self.reader_count_this_reset += 1
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
            self.reader_count_this_reset = 0
            self.events.append(("reset_scene", reset_state))
            fresh = reset_state is None
            return {
                "root_pose_writes": 0,
                "root_velocity_writes": 0,
                "joint_state_writes": 0,
                "global_simulation_resets": 1,
                "simulation_forward_syncs": 0,
                "physics_lifecycle_reset": (
                    "scene_factory_reset_before_limit_authoring"
                    if fresh
                    else "session_limits_removed_then_hard_reset"
                ),
                "reset_contact_sensor_count": 13,
                "reset_initialization_order": (
                    "physics_reset_without_session_limits_then_author_limits_then_settle"
                ),
                "pre_physics_session_limit_state_sha256": "e" * 64,
                "pre_physics_composed_limit_state_sha256": "c" * 64,
                "session_limit_specs_present_during_physics_reset": 0,
                "session_limit_specs_removed_before_reset": 0 if fresh else 16,
                "removed_session_limit_state_sha256": None if fresh else "a" * 64,
                "pre_limit_native_state_observed_sha256": "b" * 64,
                "pre_limit_native_state_instance_count": 1,
            }

        def capture_session_limit_state(scene):
            assert scene is self.scene
            self.events.append(("capture_session_limit_state",))
            return {
                "property_count": 16,
                "state_sha256": "a" * 64,
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

        def write_phase_snapshot(
            scene,
            adapter,
            snapshot,
            *,
            reset_contact_backend,
            state_write_index,
        ):
            self.events.append(
                (
                    "write_phase_snapshot",
                    snapshot["fsm_state"],
                    reset_contact_backend,
                    state_write_index,
                )
            )
            pre_state = backend_module._install_source_mapper_pre_state(
                adapter, snapshot
            )
            root = snapshot["root_state"]
            return {
                "root_pose_writes": 1,
                "root_velocity_writes": 1,
                "joint_state_writes": 1,
                "global_simulation_resets": 0,
                "simulation_forward_syncs": 1,
                "physics_steps": 0,
                "state_write_index": state_write_index,
                "contact_backend_reset": reset_contact_backend,
                "root_velocity_write_api": "write_root_link_velocity_to_sim",
                "source_mapper_pre_state": pre_state,
                "pre_prime_root_link_readback": {
                    "schema": (
                        "wlr50_clean.phase_snapshot_pre_prime_root_link_readback.v1"
                    ),
                    "body_name": "base_link",
                    "expected": dict(root),
                    "observed": dict(root),
                    "maximum_errors": {
                        "root_position_m": 0.0,
                        "root_orientation_quaternion_distance": 0.0,
                        "root_linear_velocity_m_s": 0.0,
                        "root_angular_velocity_rad_s": 0.0,
                    },
                    "physics_steps_before_readback": 0,
                    "contact_sensor_reads_before_readback": 0,
                    "all_values_finite": True,
                    "all_fields_within_production_tolerances": True,
                    "verified": True,
                },
                "pre_prime_joint_state_verified": True,
                "pre_prime_state_verified": True,
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
            controller.lifecycle = _value("WAIT_ENTRY")
            controller.termination = None
            return {
                "state_id": phase,
                "lifecycle": "WAIT_ENTRY",
                "history_is_independent": True,
                "entry_guards_pending_effective_tick_zero": True,
            }

        def verify_effective_entry(contract, phase, comparison):
            self.events.append(("verify_effective_entry", phase))
            if self.effective_entry_error is not None:
                raise EffectivePhaseEntryError(self.effective_entry_error)
            entry = contract.entry(phase)
            return {
                "schema": "fake.phase_effective_entry_live_proof.v1",
                "verified": True,
                "phase": phase,
                "contract_sha256": contract.contract_sha256,
                "entry_sha256": entry["entry_sha256"],
                "fingerprint_maximum_ulp_distance": 0,
                "contact_contract_verified": True,
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
            capture_session_limit_state=capture_session_limit_state,
            verify_effective_entry=verify_effective_entry,
        )


def _backend(runtime):
    return IsaacFSMBackend(
        dependencies=runtime.dependencies(),
        expected_effective_entry_contract=FakeEffectiveEntryContract(),
    )


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
    assert initial.info["effective_phase_entry_semantics"] == "natural_p01_post_settle"
    assert initial.info["reset_generation"] == 1
    assert initial.info["reset_generation_commit"] == (
        "committed_after_authoritative_entry_gate"
    )
    assert initial.info["phase_snapshot_restoration"]["authoritative_entry"][
        "verified"
    ] is True
    assert initial.info["next_post_reset_command_tick"] == SETTLE_TICKS
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
    assert following.info["phase_snapshot_restoration"]["authoritative_entry"][
        "verified"
    ] is True
    assert following.info["reset_generation"] == 1
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


def test_each_episode_recreates_the_baseline_reset_before_limit_order() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    first = backend.reset(seed=1, options={})
    second = backend.reset(seed=2, options={})

    assert runtime.reset_scene_count == 2
    assert runtime.adapter_create_count == 2
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
    assert first.info["physics_lifecycle_reset"] == (
        "scene_factory_reset_before_limit_authoring"
    )
    assert second.info["physics_lifecycle_reset"] == (
        "session_limits_removed_then_hard_reset"
    )
    assert first.info["session_limit_specs_removed_before_reset"] == 0
    assert second.info["session_limit_specs_removed_before_reset"] == 16
    assert first.info["pre_limit_native_state_matches_canonical"] is True
    assert second.info["pre_limit_native_state_matches_canonical"] is True
    assert first.info["pre_settle_native_state_matches_canonical"] is True
    assert second.info["pre_settle_native_state_matches_canonical"] is True
    assert second.info["canonical_settled_restore_applied"] is False
    assert second.info["in_episode_root_pose_writes"] == 0
    assert sum(event[0] == "capture_reset_state" for event in runtime.events) == 4
    assert sum(event[0] == "restore_settled_state" for event in runtime.events) == 0


@pytest.mark.parametrize("prior_guard", [False, True])
def test_reused_lifecycle_orders_stop_remove_reset_and_native_capture(
    monkeypatch: pytest.MonkeyPatch,
    prior_guard: bool,
) -> None:
    events: list[str] = []
    authored = {
        "property_count": 16,
        "state_sha256": "a" * 64,
        "composed_state_sha256": "runtime-composed",
    }
    cleared = {
        "property_count": 0,
        "state_sha256": "e" * 64,
        "composed_property_count": 16,
        "composed_state_sha256": "source-composed",
    }
    inspections = iter((authored, cleared))

    def inspect(scene):
        events.append("inspect")
        return next(inspections)

    def remove(scene, before):
        assert before is authored
        assert scene.sim._disable_app_control_on_stop_handle is True
        events.append("remove")
        return cleared

    monkeypatch.setattr(backend_module, "_session_servo_limit_state", inspect)
    monkeypatch.setattr(
        backend_module,
        "_remove_session_servo_limit_specs",
        remove,
    )
    monkeypatch.setattr(
        backend_module,
        "capture_canonical_articulation_reset_state",
        lambda robot: events.append("capture_native")
        or SimpleNamespace(instance_count=1, state_sha256="b" * 64),
    )
    sim = SimpleNamespace(_disable_app_control_on_stop_handle=prior_guard)

    def stop() -> None:
        assert sim._disable_app_control_on_stop_handle is True
        events.append("stop")

    def reset(*, soft: bool) -> None:
        assert sim._disable_app_control_on_stop_handle is True
        events.append(f"reset:{soft}")
        # IsaacLab's reset currently writes False before returning.  The outer
        # transaction must still restore the value observed before STOP.
        sim._disable_app_control_on_stop_handle = False

    sim.stop = stop
    sim.is_stopped = lambda: events.append("is_stopped") or True
    sim.reset = reset
    sim.is_playing = lambda: events.append("is_playing") or True
    robot = SimpleNamespace(update=lambda dt: events.append(f"robot.update:{dt}"))
    contacts = SimpleNamespace(
        initialized=True,
        sensors={str(index): object() for index in range(13)},
        reset=lambda: events.append("contact.reset"),
    )
    scene = SimpleNamespace(
        sim=sim,
        robot=robot,
        instrumentation=SimpleNamespace(contact_backend=contacts),
    )

    evidence = _reset_physics_lifecycle(
        scene,
        SimpleNamespace(instance_count=1, state_sha256="canonical"),
    )

    assert events == [
        "stop",
        "is_stopped",
        "inspect",
        "remove",
        "reset:False",
        "is_playing",
        "robot.update:0.0",
        "contact.reset",
        "inspect",
        "capture_native",
    ]
    assert evidence["session_limit_specs_removed_before_reset"] == 16
    assert evidence["session_limit_specs_present_during_physics_reset"] == 0
    assert evidence["pre_physics_composed_limit_state_sha256"] == (
        "source-composed"
    )
    assert sim._disable_app_control_on_stop_handle is prior_guard


def test_reused_lifecycle_requires_stop_guard_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sim = SimpleNamespace(
        stop=lambda: events.append("stop"),
        is_stopped=lambda: events.append("is_stopped") or True,
        reset=lambda *, soft: events.append(f"reset:{soft}"),
        is_playing=lambda: events.append("is_playing") or True,
    )
    scene = SimpleNamespace(sim=sim)
    monkeypatch.setattr(
        backend_module,
        "_session_servo_limit_state",
        lambda scene: events.append("inspect") or {},
    )
    monkeypatch.setattr(
        backend_module,
        "_remove_session_servo_limit_specs",
        lambda scene, before: events.append("remove") or {},
    )

    with pytest.raises(
        IsaacFSMBackendError,
        match="app-control STOP guard is unavailable",
    ):
        _reset_physics_lifecycle(
            scene,
            SimpleNamespace(instance_count=1, state_sha256="canonical"),
        )

    assert events == []


@pytest.mark.parametrize("invalid_guard", [None, 0, "false"])
def test_reused_lifecycle_rejects_non_boolean_stop_guard_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
    invalid_guard: object,
) -> None:
    events: list[str] = []
    sim = SimpleNamespace(
        _disable_app_control_on_stop_handle=invalid_guard,
        stop=lambda: events.append("stop"),
        is_stopped=lambda: events.append("is_stopped") or True,
        reset=lambda *, soft: events.append(f"reset:{soft}"),
        is_playing=lambda: events.append("is_playing") or True,
    )
    scene = SimpleNamespace(sim=sim)
    monkeypatch.setattr(
        backend_module,
        "_session_servo_limit_state",
        lambda scene: events.append("inspect") or {},
    )
    monkeypatch.setattr(
        backend_module,
        "_remove_session_servo_limit_specs",
        lambda scene, before: events.append("remove") or {},
    )

    with pytest.raises(
        IsaacFSMBackendError,
        match="app-control STOP guard is not boolean",
    ):
        _reset_physics_lifecycle(
            scene,
            SimpleNamespace(instance_count=1, state_sha256="canonical"),
        )

    assert events == []
    assert sim._disable_app_control_on_stop_handle is invalid_guard


@pytest.mark.parametrize("failing_operation", ["stop", "remove", "reset"])
@pytest.mark.parametrize("prior_guard", [False, True])
def test_reused_lifecycle_restores_stop_guard_on_transaction_exception(
    monkeypatch: pytest.MonkeyPatch,
    failing_operation: str,
    prior_guard: bool,
) -> None:
    events: list[str] = []
    authored = {
        "property_count": 16,
        "state_sha256": "a" * 64,
        "composed_state_sha256": "runtime-composed",
    }
    cleared = {
        "property_count": 0,
        "state_sha256": "e" * 64,
        "composed_property_count": 16,
        "composed_state_sha256": "source-composed",
    }
    sim = SimpleNamespace(_disable_app_control_on_stop_handle=prior_guard)

    def fail_if_selected(operation: str) -> None:
        assert sim._disable_app_control_on_stop_handle is True
        events.append(operation)
        if operation == failing_operation:
            raise RuntimeError(f"injected {operation} failure")

    def stop() -> None:
        fail_if_selected("stop")

    def remove(scene, before):
        assert before is authored
        fail_if_selected("remove")
        return cleared

    def reset(*, soft: bool) -> None:
        assert soft is False
        fail_if_selected("reset")

    sim.stop = stop
    sim.is_stopped = lambda: True
    sim.reset = reset
    sim.is_playing = lambda: True
    scene = SimpleNamespace(sim=sim)
    monkeypatch.setattr(
        backend_module,
        "_session_servo_limit_state",
        lambda scene: authored,
    )
    monkeypatch.setattr(
        backend_module,
        "_remove_session_servo_limit_specs",
        remove,
    )

    with pytest.raises(
        IsaacFSMBackendError,
        match=rf"physics lifecycle reset failed: RuntimeError: injected {failing_operation} failure",
    ):
        _reset_physics_lifecycle(
            scene,
            SimpleNamespace(instance_count=1, state_sha256="canonical"),
        )

    assert sim._disable_app_control_on_stop_handle is prior_guard
    assert events.count(failing_operation) == 1
    expected_prefix = ["stop", "remove", "reset"]
    assert events == expected_prefix[: expected_prefix.index(failing_operation) + 1]


def test_reused_pre_limit_native_mismatch_fails_before_adapter_or_settle() -> None:
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    reset_scene = dependencies.reset_scene
    reset_count = 0

    def reset_with_native_mismatch(scene, reset_state):
        nonlocal reset_count
        reset_count += 1
        evidence = dict(reset_scene(scene, reset_state))
        if reset_count == 2:
            evidence["pre_limit_native_state_observed_sha256"] = "d" * 64
        return evidence

    backend = IsaacFSMBackend(
        dependencies=replace(dependencies, reset_scene=reset_with_native_mismatch)
    )
    backend.reset(seed=1, options={})
    steps_before_reuse = runtime.sim.step_count
    adapters_before_reuse = runtime.adapter_create_count

    with pytest.raises(
        IsaacFSMBackendError,
        match="did not reproduce the canonical native pre-limit state",
    ):
        backend.reset(seed=2, options={})

    assert runtime.sim.step_count == steps_before_reuse
    assert runtime.adapter_create_count == adapters_before_reuse


def test_reused_source_limit_mismatch_fails_before_adapter_or_settle() -> None:
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    reset_scene = dependencies.reset_scene
    reset_count = 0

    def reset_with_source_limit_mismatch(scene, reset_state):
        nonlocal reset_count
        reset_count += 1
        evidence = dict(reset_scene(scene, reset_state))
        if reset_count == 2:
            evidence["pre_physics_composed_limit_state_sha256"] = "d" * 64
        return evidence

    backend = IsaacFSMBackend(
        dependencies=replace(
            dependencies,
            reset_scene=reset_with_source_limit_mismatch,
        )
    )
    backend.reset(seed=1, options={})
    steps_before_reuse = runtime.sim.step_count
    adapters_before_reuse = runtime.adapter_create_count

    with pytest.raises(
        IsaacFSMBackendError,
        match="source-composed servo limits changed",
    ):
        backend.reset(seed=2, options={})

    assert runtime.sim.step_count == steps_before_reuse
    assert runtime.adapter_create_count == adapters_before_reuse


def test_reused_limit_reauthor_mismatch_fails_before_settle() -> None:
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    capture_count = 0

    def capture_mismatched_reauthor(scene):
        nonlocal capture_count
        capture_count += 1
        return {
            "property_count": 16,
            "state_sha256": ("a" if capture_count == 1 else "d") * 64,
        }

    backend = IsaacFSMBackend(
        dependencies=replace(
            dependencies,
            capture_session_limit_state=capture_mismatched_reauthor,
        )
    )
    backend.reset(seed=1, options={})
    steps_before_reuse = runtime.sim.step_count

    with pytest.raises(
        IsaacFSMBackendError,
        match="did not reproduce the removed session limit state",
    ):
        backend.reset(seed=2, options={})

    assert runtime.sim.step_count == steps_before_reuse


@pytest.mark.parametrize("mismatch", ["sha256", "instance_count"])
def test_reused_post_author_native_mismatch_fails_before_settle_step(
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
    physical = restoration["physical_state"]
    assert physical["schema"] == "wlr50_clean.phase_snapshot_prime_without_rewind.v1"
    assert physical["reset_use"] == "TRAINING_RESET_STATE_WRITE"
    assert physical["state_write_count"] == 1
    assert physical["post_prime_state_rewrite_performed"] is False
    assert physical["contact_and_state_share_solver_tick"] is True
    assert physical["prime_physics_steps"] == PHASE_SNAPSHOT_PRIME_PHYSICS_STEPS
    assert physical["prime_applied_full12"] == list(
        _snapshot_payload("P09")["applied_full12"]
    )
    assert physical["prime_atomic_full12_writes"] == 1
    assert physical["logical_target_fallback_used"] is False
    assert physical["source_actuation_match"]["all_fields_match"] is True
    assert physical["source_actuation_match"]["source_target_hash_matches"] is True
    assert physical["source_mapper_post_state"]["all_fields_match"] is True
    assert physical["source_mapper_post_state"][
        "reached_naturally_by_single_atomic_apply"
    ] is True
    assert physical["fsm_clock_steps_during_priming"] == 0
    assert physical["episode_clock_steps_during_priming"] == 0
    assert physical["contact_sensor_reads_after_prime"] == 1
    assert physical["classifier_restored_before_only_episode_read"] is False
    assert physical["classifier_source_state_restored"] is False
    assert physical["classifier_source_history_restored"] is False
    assert physical["classifier_cold_started_before_only_episode_read"] is True
    assert physical["classifier_history_equivalence_claimed"] is False
    assert physical["raw_sensor_history_rewarmed_from_prime"] is True
    assert physical["current_contact_force_provenance"] == (
        "current_final_solver_force_only"
    )
    assert physical["sensor_history_samples_after_reset"] == 1
    assert physical["source_snapshot_post_prime_diagnostic"]["verified"] is True
    assert physical["effective_entry_contract"]["verified"] is True
    assert physical["entry_safety_contract"]["verified"] is True
    assert physical["entry_guard_contract"]["verified"] is True
    assert physical["authoritative_entry_contract"]["verified"] is True
    assert restoration["effective_entry"]["verified"] is True
    assert restoration["entry_guards"]["lifecycle"] == "EXECUTE_MOTION"
    assert frame.info["reset_root_pose_writes"] == 1
    assert frame.info["reset_root_velocity_writes"] == 1
    assert frame.info["reset_joint_state_writes"] == 1
    assert frame.info["reset_simulation_forward_syncs"] == 1
    assert frame.info["reset_prime_tick_count"] == 1
    assert frame.info["next_post_reset_command_tick"] == SETTLE_TICKS + 1
    assert frame.info["first_episode_physical_command_tick_actual"] is None
    assert frame.info["effective_phase_entry_semantics"] == "snapshot_plus_one_physics_tick"
    applied_ticks = [row[1] for row in runtime.events if row[0] == "adapter.apply"]
    assert applied_ticks == list(range(SETTLE_TICKS + 1))
    assert runtime.sim.step_count == SETTLE_TICKS + 1
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
    assert next(row[1] for row in runtime.events if row[0] == "adapter.apply") == (
        SETTLE_TICKS + 1
    )
    assert following.info["first_episode_physical_command_tick_actual"] == (
        SETTLE_TICKS + 1
    )
    assert all("snapshot" not in row[0] for row in runtime.events)


def test_effective_entry_calibration_skips_only_the_prior_contract() -> None:
    runtime = FakeRuntime()
    backend = IsaacFSMBackend(
        dependencies=runtime.dependencies(),
        allow_effective_entry_calibration=True,
    )

    frame = backend.reset(seed=1002, options={"training_phase_snapshot": "P10"})

    restoration = frame.info["phase_snapshot_restoration"]
    proof = restoration["physical_state"]["effective_entry_contract"]
    assert proof["schema"] == (
        "wlr50_clean.ppo_phase_effective_entry_calibration_live_proof.v1"
    )
    assert proof["artifact_role"] == "CALIBRATION_ONLY_NOT_TRAINING_ACCEPTANCE"
    assert proof["calibration_only"] is True
    assert proof["phase"] == "P10"
    assert proof["verified"] is True
    assert proof["source_snapshot_post_prime_diagnostic"] == restoration[
        "physical_state"
    ]["source_snapshot_post_prime_diagnostic"]
    assert restoration["physical_state"]["entry_safety_contract"]["verified"] is True
    assert restoration["physical_state"]["entry_guard_contract"]["verified"] is True
    assert not any(row[0] == "verify_effective_entry" for row in runtime.events)
    assert sum(row[0] == "controller.step" for row in runtime.events) == 1


def test_effective_entry_calibration_is_mutually_exclusive_with_contract() -> None:
    with pytest.raises(IsaacFSMBackendError, match="cannot consume"):
        IsaacFSMBackend(
            dependencies=FakeRuntime().dependencies(),
            expected_effective_entry_contract=FakeEffectiveEntryContract(),
            allow_effective_entry_calibration=True,
        )


def test_phase_snapshot_state_write_uses_root_link_velocity_api_only() -> None:
    torch = pytest.importorskip("torch")
    payload = _snapshot_payload("P09")

    class Robot:
        def __init__(self) -> None:
            self.body_names = ("base_link",)
            self.data = SimpleNamespace(
                default_root_state=torch.zeros((1, 13), dtype=torch.float64),
                default_joint_pos=torch.zeros((1, 12), dtype=torch.float64),
                default_joint_vel=torch.zeros((1, 12), dtype=torch.float64),
                body_link_pos_w=torch.zeros((1, 1, 3), dtype=torch.float64),
                body_link_quat_w=torch.zeros((1, 1, 4), dtype=torch.float64),
                body_link_lin_vel_w=torch.zeros((1, 1, 3), dtype=torch.float64),
                body_link_ang_vel_w=torch.zeros((1, 1, 3), dtype=torch.float64),
            )
            self.link_velocity = None
            self.com_velocity_alias_calls = 0

        def write_root_pose_to_sim(self, value):
            self.root_pose = value.clone()
            self.data.body_link_pos_w[:, 0, :] = value[:, :3]
            self.data.body_link_quat_w[:, 0, :] = value[:, 3:]

        def write_root_link_velocity_to_sim(self, value):
            self.link_velocity = value.clone()
            self.data.body_link_lin_vel_w[:, 0, :] = value[:, :3]
            self.data.body_link_ang_vel_w[:, 0, :] = value[:, 3:]

        def write_root_velocity_to_sim(self, value):
            self.com_velocity_alias_calls += 1

        def write_joint_state_to_sim(self, position, velocity):
            self.joint_position = position.clone()
            self.joint_velocity = velocity.clone()

        def reset(self):
            return None

        def update(self, dt):
            self.update_dt = dt

    robot = Robot()
    contact_backend = SimpleNamespace(reset_calls=0)

    def reset_contacts():
        contact_backend.reset_calls += 1

    contact_backend.reset = reset_contacts
    sim = SimpleNamespace(forward_calls=0)

    def forward():
        sim.forward_calls += 1

    sim.forward = forward
    mapper = SimpleNamespace(
        physics_dt_s=1.0 / 120.0,
        servo_rate_deg_s=150.0,
        maximum_delta_deg=1.25,
        tracking_gain=8.0,
        tracking_limit_deg=10.0,
        feedback_interval_ticks=4,
        standing_pose_deg={name: 0.0 for name in SERVO_NAMES},
        _requested={name: 0.0 for name in SERVO_NAMES},
        _applied={name: 0.0 for name in SERVO_NAMES},
        _nominal_reached={name: True for name in SERVO_NAMES},
        _compensation={name: 0.0 for name in SERVO_NAMES},
        _tracking_active={name: False for name in SERVO_NAMES},
        _retiring_stale_bias={name: False for name in SERVO_NAMES},
        _feedback_tick=0,
    )
    adapter = SimpleNamespace(
        joint_map=SimpleNamespace(
            servo_ids=tuple(range(8)), wheel_ids=tuple(range(8, 12))
        ),
        standing_pose_deg={name: 0.0 for name in SERVO_NAMES},
        servo_target_mapper=mapper,
        _final_drive_servo_deg={name: 0.0 for name in SERVO_NAMES},
        get_actual_state=lambda: SimpleNamespace(
            full12=ZERO, servo_velocity_rad_s=(0.0,) * 8
        ),
    )
    scene = SimpleNamespace(
        robot=robot,
        sim=sim,
        instrumentation=SimpleNamespace(contact_backend=contact_backend),
    )

    proof = _write_phase_snapshot_state(scene, adapter, payload)

    assert robot.link_velocity is not None
    assert robot.link_velocity[0].tolist() == pytest.approx(
        payload["root_state"]["linear_velocity_w_m_s"]
        + payload["root_state"]["angular_velocity_w_rad_s"]
    )
    assert robot.com_velocity_alias_calls == 0
    assert proof["root_velocity_write_api"] == "write_root_link_velocity_to_sim"
    assert proof["pre_prime_state_verified"] is True
    root_proof = proof["pre_prime_root_link_readback"]
    assert root_proof["verified"] is True
    assert root_proof["observed"]["position_w_m"] == pytest.approx(
        payload["root_state"]["position_w_m"]
    )
    assert root_proof["observed"]["linear_velocity_w_m_s"] == pytest.approx(
        payload["root_state"]["linear_velocity_w_m_s"]
    )
    assert contact_backend.reset_calls == 1
    assert sim.forward_calls == 1


def test_pre_prime_root_link_readback_fails_closed_on_link_velocity_alias_drift() -> None:
    numpy = pytest.importorskip("numpy")
    payload = _snapshot_payload("P09")
    root = payload["root_state"]
    robot = SimpleNamespace(
        body_names=("base_link",),
        data=SimpleNamespace(
            body_link_pos_w=numpy.asarray([[root["position_w_m"]]], dtype=float),
            body_link_quat_w=numpy.asarray(
                [[root["orientation_wxyz"]]], dtype=float
            ),
            body_link_lin_vel_w=numpy.asarray(
                [[[root["linear_velocity_w_m_s"][0] + 0.001, 0.0, 0.0]]],
                dtype=float,
            ),
            body_link_ang_vel_w=numpy.asarray(
                [[root["angular_velocity_w_rad_s"]]], dtype=float
            ),
        ),
    )

    with pytest.raises(IsaacFSMBackendError, match="root_linear_velocity_m_s"):
        backend_module._verify_pre_prime_root_link_write(robot, root)


@pytest.mark.parametrize("prime_steps", [0, 2, 8, True, 1.0])
def test_phase_snapshot_backend_rejects_any_prime_count_other_than_one(
    prime_steps,
) -> None:
    with pytest.raises(IsaacFSMBackendError, match="exactly one"):
        IsaacFSMBackend(
            dependencies=FakeRuntime().dependencies(),
            phase_snapshot_prime_physics_steps=prime_steps,
        )


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
    assert frame.info["reset_prime_tick_count"] == 1


def test_explicit_p01_snapshot_keeps_the_normal_p01_reset_path() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)

    frame = backend.reset(seed=4, options={"training_phase_snapshot": "P01"})

    assert frame.state_id == "P01"
    assert frame.info["phase_snapshot_restoration"]["snapshot_validated"] is True
    assert frame.info["phase_snapshot_restoration"]["mode"] == "normal_p01_reset"
    assert frame.info["effective_phase_entry_semantics"] == (
        "validated_p01_natural_post_settle"
    )
    assert frame.info["reset_root_pose_writes"] == 0
    assert not any(row[0] == "write_phase_snapshot" for row in runtime.events)
    assert not any(row[0] == "restore_controller_snapshot" for row in runtime.events)


def test_failed_second_reset_poisoned_the_successful_prior_generation() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    first = backend.reset(seed=4, options={})
    first_adapter = backend._adapter

    runtime.entry_fault = "body"
    with pytest.raises(
        SensorContractFailure, match="authoritative-frame gate failed"
    ):
        backend.reset(seed=5, options={})

    assert first.info["reset_generation"] == 1
    assert backend._reset_generation == 2
    assert backend._committed_reset_generation is None
    assert backend._reset_count == 1
    assert backend._adapter is None
    assert backend._reader is None
    assert backend._controller is None
    assert backend._raw_observation is None
    assert backend._controller_frame is None
    assert backend._authoritative_frame is None
    assert runtime.adapter is not first_adapter
    assert backend._snapshot_restoration["authoritative_entry"]["verified"] is False
    with pytest.raises(IsaacFSMBackendError, match="committed reset generation"):
        backend.step_physics(ZERO)

    runtime.entry_fault = None
    recovered = backend.reset(seed=6, options={})
    assert recovered.info["reset_generation"] == 3
    assert backend._committed_reset_generation == 3
    assert backend._reset_count == 2


def test_invalid_second_reset_poisoned_prior_generation_before_physics() -> None:
    runtime = FakeRuntime()
    backend = _backend(runtime)
    backend.reset(seed=4, options={})
    physics_steps_before = runtime.sim.step_count

    with pytest.raises(IsaacFSMBackendError, match="prohibited"):
        backend.reset(seed=5, options={"recording_path": "forbidden.jsonl"})

    assert runtime.sim.step_count == physics_steps_before
    assert backend._reset_generation == 2
    assert backend._committed_reset_generation is None
    assert backend._authoritative_frame is None
    assert backend._adapter is None
    with pytest.raises(IsaacFSMBackendError, match="committed reset generation"):
        backend.step_physics(ZERO)


@pytest.mark.parametrize(
    "entry_fault", ["body", "wheel", "fall", "nan", "joint", "explosion"]
)
def test_p01_physical_entry_failures_never_commit(entry_fault: str) -> None:
    backend = _backend(FakeRuntime(entry_fault=entry_fault))

    with pytest.raises(SensorContractFailure):
        backend.reset(seed=4, options={})

    assert backend._committed_reset_generation is None
    assert backend._reset_count == 0
    with pytest.raises(IsaacFSMBackendError, match="committed reset generation"):
        backend.step_physics(ZERO)


@pytest.mark.parametrize(
    "entry_controller_result", ["SUCCESS", "INCOMPLETE_CONTROLLER_BLOCKED"]
)
def test_p01_terminal_or_timeout_entry_never_commits(
    entry_controller_result: str,
) -> None:
    backend = _backend(
        FakeRuntime(entry_controller_result=entry_controller_result)
    )

    with pytest.raises(
        SensorContractFailure, match="authoritative-frame gate failed"
    ):
        backend.reset(seed=4, options={})

    assert backend._committed_reset_generation is None
    assert backend._reset_count == 0


def test_p01_non_neutral_safety_projection_never_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(FakeRuntime())
    build = backend._build_authoritative_frame

    def unsafe_frame(*args, **kwargs):
        frame = build(*args, **kwargs)
        return replace(
            frame,
            safety_projection=replace(
                frame.safety_projection,
                residual_enabled=False,
                reason="test-only non-neutral projection",
            ),
        )

    monkeypatch.setattr(backend, "_build_authoritative_frame", unsafe_frame)

    with pytest.raises(SensorContractFailure, match="non-neutral safety"):
        backend.reset(seed=4, options={})

    assert backend._committed_reset_generation is None
    assert backend._reset_count == 0


@pytest.mark.parametrize(
    "runtime",
    [
        FakeRuntime(entry_frame_lifecycle="WAIT_ENTRY"),
        FakeRuntime(entry_first_blocker=True),
    ],
    ids=("non-execute", "blocked-without-termination"),
)
def test_p01_nonrunning_or_blocked_controller_never_commits(runtime) -> None:
    backend = _backend(runtime)

    with pytest.raises(
        SensorContractFailure,
        match="authoritative controller frame is not running/nonterminal",
    ):
        backend.reset(seed=4, options={})

    assert backend._committed_reset_generation is None
    assert backend._reset_count == 0


def test_phase_snapshot_fails_before_scene_mutation_without_complete_restore_seams() -> None:
    runtime = FakeRuntime()
    dependencies = replace(runtime.dependencies(), restore_guard_snapshot=None)
    backend = IsaacFSMBackend(
        dependencies=dependencies,
        expected_effective_entry_contract=FakeEffectiveEntryContract(),
    )

    with pytest.raises(IsaacFSMBackendError, match="effective-entry seams"):
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


def test_all_checked_snapshots_replay_source_drive_with_one_real_adapter_write() -> None:
    numpy = pytest.importorskip("numpy")
    from wlr50_clean.infrastructure.command_batch import (
        SERVO_COMMAND_SIGN,
        JointIndexMap,
    )
    from wlr50_clean.infrastructure.robot_adapter import RobotAdapter
    from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper

    class TargetRobot:
        def __init__(self, joint_position):
            self.data = SimpleNamespace(
                joint_pos=joint_position.copy(),
                joint_vel=numpy.zeros_like(joint_position),
            )
            self.articulation_write_count = 0

        def set_joint_position_target(self, value, *, joint_ids):
            self.position_target = value.copy()
            self.position_target_ids = tuple(joint_ids)

        def set_joint_velocity_target(self, value, *, joint_ids):
            self.velocity_target = value.copy()
            self.velocity_target_ids = tuple(joint_ids)

        def write_data_to_sim(self):
            self.articulation_write_count += 1

    for phase in (f"P{index:02d}" for index in range(1, 14)):
        snapshot = _load_validated_phase_snapshot(phase).payload
        source = snapshot["source_command"]
        expected = source["expected_atomic_ack"]
        standing = dict(
            zip(
                SERVO_NAMES,
                source["mapper_configuration"]["standing_pose_deg"],
                strict=True,
            )
        )
        servo_ids = tuple(expected["servo_joint_ids"])
        wheel_ids = tuple(expected["wheel_joint_ids"])
        joint_position = numpy.zeros((1, 12), dtype=numpy.float64)
        for index, name in enumerate(SERVO_NAMES):
            joint_position[:, servo_ids[index]] = math.radians(
                standing[name]
                + SERVO_COMMAND_SIGN[name]
                * snapshot["joint_state"]["logical_position_deg"][index]
            )
        robot = TargetRobot(joint_position)
        adapter = object.__new__(RobotAdapter)
        adapter.robot = robot
        adapter.physics_dt_s = 1.0 / 120.0
        adapter.joint_map = JointIndexMap(
            servo_ids=servo_ids,
            wheel_ids=wheel_ids,
            live_joint_names=SERVO_NAMES + WHEEL_NAMES,
        )
        adapter.standing_pose_deg = standing
        adapter._standing_servo_tensor = numpy.asarray(
            [[math.radians(standing[name]) for name in SERVO_NAMES]],
            dtype=numpy.float64,
        )
        adapter.servo_target_mapper = ServoTargetMapper(standing)
        adapter._final_drive_servo_deg = {name: 0.0 for name in SERVO_NAMES}
        adapter.write_count = SETTLE_TICKS
        adapter._last_physics_tick = SETTLE_TICKS - 1
        adapter.last_ack = None
        backend_module._install_source_mapper_pre_state(adapter, snapshot)

        ack = adapter.apply_full12(
            source["adapter_input"]["requested_full12"],
            physics_tick=SETTLE_TICKS,
            tracking_servo_names=source["adapter_input"]["tracking_servo_names"],
            drive_feedback_bias_full12=source["adapter_input"][
                "drive_feedback_bias_requested_full12"
            ],
        )
        ack_match = backend_module._verify_source_prime_ack(snapshot, ack)
        post_match = backend_module._verify_source_mapper_post_state(adapter, snapshot)

        assert robot.articulation_write_count == 1
        assert ack_match["all_fields_match"] is True
        assert ack_match["source_target_hash_matches"] is True
        assert post_match["all_fields_match"] is True

    p03 = _load_validated_phase_snapshot("P03").payload["source_command"]
    assert p03["expected_atomic_ack"]["servo_tracking_feedback_sampled"] is True
    p10_payload = _load_validated_phase_snapshot("P10").payload
    p10 = p10_payload["source_command"]
    assert p10_payload["source_tick"] == 7793
    assert p10_payload["target_entry_tick"] == 7794
    assert p10["source_fsm_lifecycle"] == "WAIT_ENTRY"
    assert p10["expected_atomic_ack"]["drive_feedback_bias_requested_full12"][
        7
    ] == pytest.approx(0.0)
    assert p10["expected_atomic_ack"]["drive_feedback_bias_realized_full12"][
        7
    ] == pytest.approx(0.0)


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
    assert controller_proof["lifecycle"] == "WAIT_ENTRY"
    assert controller_proof["entry_guards_pending_effective_tick_zero"] is True
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
    assert guard_proof["classifier_wheel_pairs_restored"] == 0
    assert guard_proof["classifier_source_state_restored"] is False
    assert guard_proof["classifier_source_history_restored"] is False
    assert reader.contact_classifier._states == {}
    assert reader.contact_classifier._history == {}


def test_source_snapshot_drift_is_diagnostic_after_effective_contract_passes() -> None:
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
    backend = IsaacFSMBackend(
        dependencies=dependencies,
        expected_effective_entry_contract=FakeEffectiveEntryContract(),
    )

    frame = backend.reset(seed=81, options={"training_phase_snapshot": "P08"})

    diagnostic = frame.info["phase_snapshot_restoration"][
        "source_snapshot_diagnostic"
    ]
    assert diagnostic["verified"] is False
    assert frame.info["phase_snapshot_restoration"]["effective_entry"][
        "verified"
    ] is True
    assert sum(row[0] == "controller.step" for row in runtime.events) == 1


def test_effective_entry_mismatch_fails_before_first_controller_tick() -> None:
    runtime = FakeRuntime(effective_entry_error="fingerprint exceeds one ULP")
    backend = _backend(runtime)

    with pytest.raises(SensorContractFailure, match="effective-entry contract failed"):
        backend.reset(seed=81, options={"training_phase_snapshot": "P08"})

    assert runtime.controller.physics_tick == 0
    assert not any(row[0] == "controller.step" for row in runtime.events)
    assert backend._reset_count == 0


@pytest.mark.parametrize("entry_fault", ["body", "wheel", "fall", "joint", "explosion"])
def test_effective_entry_safety_failures_do_not_commit_reset(entry_fault: str) -> None:
    runtime = FakeRuntime(entry_fault=entry_fault)
    backend = _backend(runtime)

    with pytest.raises(SensorContractFailure, match="safety gate failed"):
        backend.reset(seed=81, options={"training_phase_snapshot": "P08"})

    assert backend._reset_count == 0
    assert backend._authoritative_frame is None
    assert backend._snapshot_restoration["entry_safety"]["verified"] is False


def test_effective_entry_nan_fails_sensor_contract_without_committing_reset() -> None:
    runtime = FakeRuntime(entry_fault="nan")
    backend = _backend(runtime)

    with pytest.raises(SensorContractFailure, match="non-finite"):
        backend.reset(seed=81, options={"training_phase_snapshot": "P08"})

    assert backend._reset_count == 0
    assert backend._snapshot_restoration["entry_sensor"]["verified"] is False


@pytest.mark.parametrize(
    "runtime",
    [
        FakeRuntime(entry_controller_result="SUCCESS"),
        FakeRuntime(entry_controller_result="INCOMPLETE_CONTROLLER_BLOCKED"),
        FakeRuntime(entry_guard_failure=True),
        FakeRuntime(p10_entry_velocity_deg_s=-1.0),
    ],
    ids=("terminal", "blocked", "guard-failed", "p10-signed-velocity"),
)
def test_effective_controller_entry_failure_does_not_commit_reset(runtime) -> None:
    backend = _backend(runtime)
    phase = "P10" if runtime.p10_entry_velocity_deg_s < 0.0 else "P08"

    with pytest.raises(SensorContractFailure, match="controller/guard gate failed"):
        backend.reset(seed=81, options={"training_phase_snapshot": phase})

    assert backend._reset_count == 0
    assert backend._authoritative_frame is None
    assert backend._snapshot_restoration["entry_guards"]["verified"] is False
    if runtime.entry_guard_failure:
        assert backend._snapshot_restoration["entry_guards"][
            "pending_entry_blocker"
        ]["name"] == "previous_state_done"


def test_p10_effective_entry_proves_signed_positive_velocity_guard() -> None:
    runtime = FakeRuntime(p10_entry_velocity_deg_s=2.5)
    frame = _backend(runtime).reset(
        seed=81, options={"training_phase_snapshot": "P10"}
    )

    proof = frame.info["phase_snapshot_restoration"]["entry_guards"]
    assert proof["verified"] is True
    assert proof["p10_signed_velocity_alignment"] == {
        "actual_deg_s": 2.5,
        "signed_positive_rebound_required": True,
    }


def test_reference_conformance_remains_diagnostic_at_effective_entry() -> None:
    frame = _backend(FakeRuntime()).reset(
        seed=81, options={"training_phase_snapshot": "P08"}
    )
    candidate = replace(
        frame,
        termination_signals=replace(
            frame.termination_signals,
            reference_conformance_outside_30pct=True,
        ),
    )

    proof = backend_module._verify_effective_authoritative_entry(candidate)

    assert "reference_conformance_outside_30pct" not in proof["termination_flags"]
    assert proof["diagnostic_ignored_for_termination"] == {
        "reference_conformance_outside_30pct": True
    }


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
