"""Own one GUI Isaac process and one continuous 120 Hz sensor-FSM trial.

This module is deliberately safe to import in ordinary Python.  Before the
``SimulationApp`` exists it imports only the standard library and
``isaaclab.app.AppLauncher``.  Every Isaac, controller, sensor, evaluation,
and video dependency is loaded only after the one initial ``app.update()``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHYSICS_HZ = 120.0
PHYSICS_DT_S = 1.0 / PHYSICS_HZ
DECISION_HZ = 15.0
RENDER_STRIDE = 8
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
MAX_CONTROL_SECONDS = 200.0
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
LEG_ORDER = ("FL", "FR", "RL", "RR")


class ArtifactWriteError(RuntimeError):
    """A required immutable trial ledger could not be extended."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    run_dir: Path
    fsm_path: Path
    motion_contract_path: Path
    max_control_seconds: float = MAX_CONTROL_SECONDS
    settle_seconds: float = 1.5
    warmup_renders: int = 3

    @property
    def settle_ticks(self) -> int:
        return round(self.settle_seconds * PHYSICS_HZ)

    @property
    def maximum_control_ticks(self) -> int:
        return round(self.max_control_seconds * PHYSICS_HZ)


def render_due(completed_control_steps: int) -> bool:
    """True only for the sole render after each eight completed physics steps."""

    steps = int(completed_control_steps)
    return steps > 0 and steps % RENDER_STRIDE == 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one clean continuous WLR 50 mm FSM trial")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fsm", type=Path, default=PROJECT_ROOT / "configs" / "fsm_states.yaml")
    parser.add_argument(
        "--motion-contract",
        type=Path,
        default=PROJECT_ROOT / "configs" / "recording_motion_contract.json",
    )
    parser.add_argument("--max-control-seconds", type=float, default=MAX_CONTROL_SECONDS)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--warmup-renders", type=int, default=3)
    return parser


def _config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"immutable run directory already exists: {run_dir}")
    maximum = float(args.max_control_seconds)
    if not math.isfinite(maximum) or maximum <= 0.0 or maximum > MAX_CONTROL_SECONDS:
        raise ValueError("max control duration must be in (0, 200] seconds")
    settle = float(args.settle_seconds)
    if not math.isclose(settle, 1.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("the locked mature-v010 zero-command settle is exactly 1.5 seconds")
    warmup = int(args.warmup_renders)
    if warmup != 3:
        raise ValueError("the locked active-viewport warmup is exactly three renders")
    fsm_path = Path(args.fsm).resolve()
    contract_path = Path(args.motion_contract).resolve()
    locked_fsm = (PROJECT_ROOT / "configs" / "fsm_states.yaml").resolve()
    locked_contract = (PROJECT_ROOT / "configs" / "recording_motion_contract.json").resolve()
    if fsm_path != locked_fsm or contract_path != locked_contract:
        raise ValueError("physical trials require the project's locked v010 FSM and motion contract")
    for label, path in (("FSM", fsm_path), ("motion contract", contract_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    return RuntimeConfig(run_dir, fsm_path, contract_path, maximum, settle, warmup)


def _write_bootstrap_failure(
    config: RuntimeConfig,
    exc: BaseException,
    *,
    simulation_app_created: bool,
) -> None:
    """Leave an immutable infrastructure classification when Kit cannot start."""

    try:
        config.run_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema": "wlr50_clean.bootstrap_failure.v1",
            "result": "INFRASTRUCTURE_ERROR",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "simulation_app_created": bool(simulation_app_created),
        }
        (config.run_dir / "bootstrap_failure.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        (config.run_dir / "trial_manifest.json").write_text(
            json.dumps({**payload, "schema": "wlr50_clean.trial_manifest.v1"}, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _anti_throttle_settings() -> dict[str, Any]:
    import carb  # type: ignore

    settings = carb.settings.get_settings()
    values = {
        "/app/runLoops/main/rateLimitEnabled": False,
        "/app/renderer/sleepMsOnFocus": 0,
        "/app/renderer/sleepMsOutOfFocus": 0,
        "/app/renderer/skipWhileMinimized": False,
    }
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
    settings.set_int("/app/renderer/sleepMsOnFocus", 0)
    settings.set_int("/app/renderer/sleepMsOutOfFocus", 0)
    settings.set_bool("/app/renderer/skipWhileMinimized", False)
    return values


def _configure_active_viewport() -> Any:
    from omni.kit.viewport.utility import get_active_viewport  # type: ignore

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("GUI mode requires an active Isaac viewport")
    setter = getattr(viewport, "set_texture_resolution", None)
    getter = getattr(viewport, "get_texture_resolution", None)
    if not callable(setter) or not callable(getter):
        raise RuntimeError("active viewport texture-resolution API is unavailable")
    setter((VIDEO_WIDTH, VIDEO_HEIGHT))
    if tuple(int(value) for value in getter()) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        raise RuntimeError("active viewport rejected the locked 1280x720 texture resolution")
    return viewport


def _guard_passed(observation: Any, key: str) -> bool:
    value = observation.guards.get(key)
    if isinstance(value, Mapping):
        return bool(value.get("passed"))
    return bool(value)


def _append_artifact(writer: Any, stream_name: str, payload: Any) -> None:
    try:
        writer.append(stream_name, payload)
    except Exception as exc:
        raise ArtifactWriteError(
            f"required {stream_name} artifact append failed: {type(exc).__name__}: {exc}"
        ) from exc


def _observation_log(writer: Any, observation: Any, state_id: str) -> None:
    _append_artifact(writer, "observation", observation.as_dict())
    base = observation.contacts["base_link"].obstacle
    _append_artifact(
        writer,
        "body_contact",
        {
            "state_id": state_id,
            "physics_tick": observation.physics_tick,
            "sim_time_s": observation.simulation_time_s,
            "simulation_time_s": observation.simulation_time_s,
            "role": "BODY",
            "body_name": "base_link",
            "detected": observation.body_collision.detected,
            "body_collision": observation.body_collision.detected,
            "obstacle_active": base.active,
            "body_collision_status": asdict(observation.body_collision),
            "base_link_obstacle_pair": asdict(base),
        },
    )


def _validate_initial_observation(observation: Any) -> None:
    failures: list[str] = []
    if len(observation.contacts) != 13:
        failures.append(f"contact body count={len(observation.contacts)}, expected 13")
    for body_name, contact in observation.contacts.items():
        if not contact.ground.pair_verified or not contact.obstacle.pair_verified:
            failures.append(f"{body_name} exact ground/obstacle contact pair is unverified")
    if len(observation.wheels) != 4:
        failures.append(f"wheel geometry count={len(observation.wheels)}, expected 4")
    for wheel in observation.wheels.values():
        if not wheel.geometry_verified or wheel.center_w_m is None or wheel.bottom_w_m is None:
            failures.append(f"{wheel.name} live collider geometry is unverified")
    if len(observation.bodies) != 13:
        failures.append(f"rigid-body state count={len(observation.bodies)}, expected 13")
    if not observation.center_of_mass.valid or len(observation.center_of_mass.included_bodies) != 13:
        failures.append("full 13-body center-of-mass observation is invalid")
    if not observation.all_finite:
        failures.append("initial live observation contains non-finite values")
    failures.extend(str(item) for item in observation.data_quality)
    if failures:
        raise RuntimeError("critical initial sensing quality failed: " + "; ".join(dict.fromkeys(failures)))


def _append_leg_latches(
    writer: Any,
    observation: Any,
    previous: dict[str, bool],
    rear_lifts: list[tuple[str, float]],
    state_id: str,
) -> None:
    for leg in LEG_ORDER:
        for event, guard in (
            ("ACTIVE_LIFT", f"reference_like_active_lift:{leg}"),
            ("FRONT_FACE_CROSSED", f"leg_front_face_crossed_latched:{leg}"),
            ("TOP_LOADED", f"leg_top_loaded_latched:{leg}"),
        ):
            active = _guard_passed(observation, guard)
            if active and not previous.get(guard, False):
                row = {
                    "event": event,
                    "leg": leg,
                    "state_id": state_id,
                    "physics_tick": observation.physics_tick,
                    "sim_time_s": observation.simulation_time_s,
                    "simulation_time_s": observation.simulation_time_s,
                    "evidence": observation.guards.get(guard),
                }
                _append_artifact(writer, "leg_crossing", row)
                if event == "ACTIVE_LIFT" and leg in ("RR", "RL"):
                    rear_lifts.append((leg, float(observation.simulation_time_s)))
            previous[guard] = active


def _ppo_payload(interface: Any, controller: Any, frame: Any, observation: Any, previous: Sequence[float]) -> tuple[dict[str, Any], tuple[float, ...]]:
    from wlr50_clean.ppo.residual_interface import PPOObservationParts

    contact_code = {"AIR": 0.0, "GROUND": 1.0, "OBSTACLE": 2.0, "GROUND_AND_OBSTACLE": 3.0, "UNVERIFIED": -1.0}
    wheel_items = tuple(observation.wheels.values())
    legs = ("FL", "FR", "RL", "RR")
    history = tuple(
        float(_guard_passed(observation, f"{guard}:{leg}"))
        for leg in legs
        for guard in ("reference_like_active_lift", "leg_front_face_crossed_latched", "leg_top_loaded_latched")
    )
    obstacle = observation.obstacle
    base = observation.base.position_w_m
    geometry = (
        obstacle.front_x_m - base[0], obstacle.back_x_m - base[0],
        obstacle.left_y_m - base[1], obstacle.right_y_m - base[1],
        obstacle.bottom_z_m - base[2], obstacle.top_z_m - base[2],
        obstacle.back_x_m - obstacle.front_x_m,
        obstacle.right_y_m - obstacle.left_y_m,
        obstacle.top_z_m - obstacle.bottom_z_m,
    )
    support = observation.support
    parts = PPOObservationParts(
        joint_position_error8=tuple(joint.error_deg for joint in observation.joints.values()),
        joint_velocity8=tuple(joint.velocity_deg_s for joint in observation.joints.values()),
        wheel_velocity4=tuple(wheel.velocity_rad_s for wheel in wheel_items),
        wheel_contact_code4=tuple(contact_code[observation.contacts[wheel.body_name].contact_class.value] for wheel in wheel_items),
        leg_history12=history,
        body_orientation_wxyz4=observation.base.orientation_wxyz,
        body_angular_velocity3=observation.base.angular_velocity_w_rad_s,
        obstacle_relative_geometry9=geometry,
        full_body_com3=observation.center_of_mass.position_w_m,
        support_diagnostics4=(
            float(support.signed_margin_m or 0.0),
            -1.0 if support.projection_inside is None else float(support.projection_inside),
            float(support.support_count), float(support.valid),
        ),
    )
    phase = controller.phase
    progress = 0.0
    if controller.motion.phase is not None:
        duration_s = controller.motion.effective_active_duration_s
        if duration_s > 0.0:
            progress = min(
                1.0,
                max(
                    0.0,
                    (controller.motion._tick_index - 1)
                    * PHYSICS_DT_S
                    / duration_s,
                ),
            )
    ppo_frame = interface.frame(
        state_id=frame.state_id,
        macro_phase=phase.macro_phase,
        phase_progress=progress,
        nominal_action_full12=frame.full12,
        action_mask_full12=phase.action_mask_full12,
        observation=parts,
        previous_action_full12=previous,
    )
    action = interface.compose_action(ppo_frame, (0.0,) * 12)
    if action != tuple(frame.full12):
        raise RuntimeError("zero residual changed the nominal FSM action")
    return asdict(ppo_frame), action


def _run_live(config: RuntimeConfig, simulation_app: Any) -> int:
    # Every non-stdlib dependency below is intentionally post-AppLauncher and
    # post-initial-update.
    from wlr50_clean.evaluation.trial_analyzer import (
        TrialArtifactWriter,
        analyze_trial,
        populate_reference_similarity,
    )
    from wlr50_clean.fsm.controller import SensorFsmController
    from wlr50_clean.fsm.task_result import TaskResult, TaskTermination
    from wlr50_clean.infrastructure.command_batch import Full12Command
    from wlr50_clean.infrastructure.robot_adapter import RobotAdapter
    from wlr50_clean.infrastructure.scene_factory import (
        ROBOT_USD_SHA256,
        create_scene,
        verify_robot_asset,
    )
    from wlr50_clean.infrastructure.video_capture import ActiveViewportVideoRecorder, VideoArtifactError
    from wlr50_clean.ppo.residual_interface import ResidualInterface
    from wlr50_clean.sensing.sensor_reader import SensorReader, create_live_sensing_backends

    anti_throttle = _anti_throttle_settings()
    writer = TrialArtifactWriter(config.run_dir)
    recorder: Any | None = None
    controller: Any | None = None
    adapter: Any | None = None
    current_observation: Any | None = None
    terminal: TaskTermination | None = None
    video_manifest: dict[str, Any] = {"valid": False, "status": "VIDEO_OR_ARTIFACT_ERROR"}
    completed: list[str] = []
    phase_times: dict[str, dict[str, Any]] = {state: {} for state in STATE_IDS}
    phase_times["P01"]["entry_time_s"] = 0.0
    previous_latches: dict[str, bool] = {}
    rear_lifts: list[tuple[str, float]] = []
    body_collision_seen = False
    wheel_only_seen = False
    control_steps = 0
    first_blocker_written = False
    previous_action: tuple[float, ...] = (0.0,) * 12
    environment_initialization: dict[str, Any] = {
        "schema": "wlr50_clean.authoritative_servo_limit_initialization.v1",
        "all_eight_servo_limits_applied": False,
        "source_asset_modified": None,
        "stage_saved": None,
        "robot_source_asset_sha256_before": None,
        "robot_source_asset_sha256_after": None,
        "robot_source_asset_hash_unchanged": False,
    }

    def terminate(result: TaskResult, reason: str, details: Mapping[str, Any] | None = None) -> TaskTermination:
        state_id = controller.state.state_id if controller is not None else "P01"
        lifecycle = controller.lifecycle.value if controller is not None else "WAIT_ENTRY"
        return TaskTermination(result, state_id, lifecycle, control_steps * PHYSICS_DT_S, reason, dict(details or {}))

    try:
        _append_artifact(writer, "task_event", {"event": "TRIAL_START", "result": None, "reference_version": "v010_20260806_220745_363972_manual", "rear_leg_order": "RR_FIRST"})
        source_asset_sha256_before = verify_robot_asset()
        environment_initialization["robot_source_asset_sha256_before"] = source_asset_sha256_before
        scene = create_scene(
            simulation_app=simulation_app,
            before_reset=lambda sim, robot: create_live_sensing_backends(
                sim=sim, robot=robot
            ),
        )
        adapter = RobotAdapter.from_scene(scene)
        source_asset_sha256_after = verify_robot_asset()
        source_asset_hash_unchanged = bool(
            source_asset_sha256_before
            == source_asset_sha256_after
            == ROBOT_USD_SHA256
        )
        environment_initialization.update(
            {
                "robot_source_asset_sha256_after": source_asset_sha256_after,
                "robot_source_asset_hash_unchanged": source_asset_hash_unchanged,
            }
        )
        if not source_asset_hash_unchanged:
            raise RuntimeError("source robot USD hash changed during live servo-limit initialization")
        backends = scene.instrumentation
        if backends is None or not backends.contact_backend.initialized:
            raise RuntimeError("the exact 13-body ContactSensor bank did not initialize at scene reset")

        zero = Full12Command.zeros()
        for settle_tick in range(config.settle_ticks):
            if not scene.app_is_running():
                raise RuntimeError("SimulationApp stopped during zero-command physical settle")
            adapter.apply_full12(zero, physics_tick=settle_tick)
            scene.sim.step(render=False)
            adapter.update_readback()
        adapter.verify_authoritative_servo_limits_adopted()
        environment_initialization = {
            **adapter.joint_limit_initialization_evidence(),
            "robot_source_asset_sha256_before": source_asset_sha256_before,
            "robot_source_asset_sha256_after": source_asset_sha256_after,
            "robot_source_asset_hash_unchanged": True,
        }

        viewport = _configure_active_viewport()
        for _ in range(config.warmup_renders):
            scene.sim.render()
        reader = SensorReader.from_live_scene(scene, adapter, backends=backends)
        controller = SensorFsmController.from_paths(config.fsm_path, config.motion_contract_path)
        if not math.isclose(
            adapter.servo_target_mapper.servo_rate_deg_s,
            controller.motion.servo_rate_limit_deg_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("motion contract and mature servo target mapper rates differ")
        residual = ResidualInterface(residual_enabled=False)
        current_observation = reader.read(physics_tick=0, simulation_time_s=0.0, commanded_full12=zero)
        _validate_initial_observation(current_observation)
        _observation_log(writer, current_observation, controller.state.state_id)
        _append_leg_latches(
            writer,
            current_observation,
            previous_latches,
            rear_lifts,
            controller.state.state_id,
        )
        body_collision_seen = bool(current_observation.body_collision.detected)
        wheel_only_seen = _guard_passed(current_observation, "wheel_only_climb_detected")
        recorder = ActiveViewportVideoRecorder(config.run_dir)
        if not recorder.start():
            raise VideoArtifactError(recorder.error or "active viewport recorder failed to start")

        for control_tick in range(config.maximum_control_ticks):
            if terminal is not None and render_due(control_steps):
                break
            if not scene.app_is_running():
                terminal = terminate(TaskResult.INFRASTRUCTURE_ERROR, "SimulationApp stopped during the continuous trial")
                break
            now = control_tick * PHYSICS_DT_S
            lifecycle_before = controller.lifecycle.value
            frame = controller.step(current_observation, sim_time_s=now)
            entering_execute = any(event.to_lifecycle == "EXECUTE_MOTION" for event in frame.events)
            executing_command = lifecycle_before == "EXECUTE_MOTION" or entering_execute

            ppo_frame, action = _ppo_payload(residual, controller, frame, current_observation, previous_action)
            total_drive_bias = tuple(
                feedback + normal
                for feedback, normal in zip(
                    frame.drive_feedback_bias_full12,
                    frame.normal_drive_bias_full12,
                    strict=True,
                )
            )
            ack = adapter.apply_full12(
                action,
                physics_tick=config.settle_ticks + control_tick,
                tracking_servo_names=frame.tracking_servo_names,
                drive_feedback_bias_full12=total_drive_bias,
            )
            if ack["articulation_writes_this_call"] != 1 or ack["motion_start_skew_s"] != 0.0:
                raise RuntimeError("atomic Full12 articulation write contract failed")
            _append_artifact(
                writer,
                "command",
                {
                    "control_physics_tick": control_tick, "sim_time_s": now,
                    "state_id": frame.state_id, "lifecycle": frame.lifecycle.value,
                    "nominal_full12": list(frame.full12), "residual_full12": [0.0] * 12,
                    "full12": list(action), "commanded_full12": list(action),
                    "applied_full12": list(ack["applied_full12"]),
                    "drive_target_full12": list(ack["drive_target_full12"]),
                    "native_drive_target_full12": list(
                        ack["native_drive_target_full12"]
                    ),
                    "drive_feedback_bias_requested_full12": list(
                        ack["drive_feedback_bias_requested_full12"]
                    ),
                    "drive_feedback_bias_realized_full12": list(
                        ack["drive_feedback_bias_realized_full12"]
                    ),
                    "atomic_ack": ack, "ppo": ppo_frame,
                    "tracking_servo_names": list(frame.tracking_servo_names),
                    "drive_feedback": dict(frame.drive_feedback_details),
                    "normal_drive_bias_full12": list(
                        frame.normal_drive_bias_full12
                    ),
                    "source_full12_atomic": frame.atomic_source_event,
                    "atomic_source_event": frame.atomic_source_event,
                    "motion_tick_index": controller.motion._tick_index - 1 if executing_command else None,
                },
            )
            if frame.decision_tick:
                _append_artifact(
                    writer,
                    "decision",
                    {
                        "physics_tick": control_tick, "sim_time_s": now, "state_id": frame.state_id,
                        "lifecycle": frame.lifecycle.value, "guards": current_observation.guards,
                        "first_blocker": frame.first_blocker,
                    },
                )
            for event in frame.events:
                _append_artifact(writer, "transition", event)
                _append_artifact(writer, "task_event", {"event": "STATE_TRANSITION", **asdict(event)})
                timing = phase_times[event.state_id]
                if event.to_lifecycle == "WAIT_ENTRY":
                    timing.setdefault("entry_time_s", event.sim_time_s)
                elif event.to_lifecycle == "EXECUTE_MOTION":
                    timing["motion_start_s"] = event.sim_time_s
                elif event.to_lifecycle == "VERIFY_RESULT":
                    timing["motion_end_s"] = event.sim_time_s
                    timing["verify_start_s"] = event.sim_time_s
                elif event.to_lifecycle == "DONE":
                    timing["completion_time_s"] = event.sim_time_s
                    if event.state_id not in completed:
                        completed.append(event.state_id)

            scene.sim.step(render=False)
            adapter.update_readback()
            control_steps += 1
            if render_due(control_steps):
                recorder.before_render(sim_step=control_steps, sim_time_s=control_steps * PHYSICS_DT_S)
                scene.sim.render()
                recorder.after_render()
                recorder.require_healthy()

            next_observation = reader.read(
                physics_tick=control_steps,
                simulation_time_s=control_steps * PHYSICS_DT_S,
                commanded_full12=ack["drive_target_full12"],
            )
            _observation_log(writer, next_observation, frame.state_id)
            _append_leg_latches(
                writer,
                next_observation,
                previous_latches,
                rear_lifts,
                frame.state_id,
            )
            body_collision_seen |= bool(next_observation.body_collision.detected)
            wheel_only_seen |= _guard_passed(next_observation, "wheel_only_climb_detected")
            current_observation = next_observation
            previous_action = action

            if frame.first_blocker is not None and not first_blocker_written:
                _append_artifact(
                    writer,
                    "task_event",
                    {"event": "FIRST_BLOCKER", "state_id": frame.state_id, "sim_time_s": now, "details": frame.first_blocker},
                )
                first_blocker_written = True
            if frame.termination is not None and terminal is None:
                terminal = frame.termination

        if terminal is None:
            terminal = terminate(
                TaskResult.INCOMPLETE_CONTROLLER_BLOCKED,
                "maximum 200 second control duration reached before P13 success",
                {"first_blocker": controller.first_blocker},
            )
    except (VideoArtifactError, ArtifactWriteError) as exc:
        if terminal is None or terminal.result is TaskResult.SUCCESS:
            terminal = terminate(TaskResult.VIDEO_OR_ARTIFACT_ERROR, str(exc))
    except Exception as exc:
        terminal = terminate(
            TaskResult.INFRASTRUCTURE_ERROR,
            f"{type(exc).__name__}: {exc}",
            {"traceback": traceback.format_exc()},
        )

    try:
        if recorder is not None:
            video_manifest = recorder.finalize()
            expected_frames = control_steps // RENDER_STRIDE
            if int(video_manifest.get("frame_count", -1)) != expected_frames:
                raise VideoArtifactError(
                    f"capture cadence mismatch: expected {expected_frames} frames for "
                    f"{control_steps} physics ticks, received {video_manifest.get('frame_count')}"
                )
            if video_manifest.get("valid") is not True and (terminal is None or terminal.result is TaskResult.SUCCESS):
                terminal = terminate(TaskResult.VIDEO_OR_ARTIFACT_ERROR, str(video_manifest.get("error") or "video validation failed"))
    except Exception as exc:
        if terminal is None or terminal.result is TaskResult.SUCCESS:
            terminal = terminate(TaskResult.VIDEO_OR_ARTIFACT_ERROR, f"video finalize failed: {exc}")
    if terminal is None:
        terminal = terminate(TaskResult.INFRASTRUCTURE_ERROR, "runtime ended without a terminal classification")

    rear_only = [(leg, when) for leg, when in rear_lifts if leg in ("RR", "RL")]
    rear_order = "UNKNOWN"
    if len(rear_only) >= 2:
        if rear_only[0][1] < rear_only[1][1]:
            rear_order = "RR_FIRST" if rear_only[0][0] == "RR" else "RL_FIRST"
    duration_s = control_steps * PHYSICS_DT_S
    if adapter is not None and adapter.write_count != config.settle_ticks + control_steps:
        terminal = terminate(
            TaskResult.INFRASTRUCTURE_ERROR,
            "Full12 write count differs from exactly one write per settle/control physics tick",
            {
                "expected": config.settle_ticks + control_steps,
                "actual": adapter.write_count,
            },
        )
    if terminal.result is TaskResult.SUCCESS and body_collision_seen:
        terminal = terminate(
            TaskResult.TASK_FAILURE_BODY_COLLISION,
            "a persistent or penetrating BODY/obstacle contact was measured",
        )
    if terminal.result is TaskResult.SUCCESS and wheel_only_seen:
        terminal = terminate(
            TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB,
            "a wheel crossed and top-loaded without measured active-lift evidence",
        )
    if terminal.result is TaskResult.SUCCESS and rear_order != "RR_FIRST":
        terminal = terminate(
            TaskResult.INCOMPLETE_CONTROLLER_BLOCKED,
            "measured rear active-lift order was not an unambiguous RR_FIRST sequence",
            {"rear_active_lifts": rear_only, "observed_rear_leg_order": rear_order},
        )
    analysis: dict[str, Any] = {
        "checks": {},
        "conformance_summary": {
            "conformance_row_count": 0,
            "all_normal_states_within_15_percent": False,
        },
    }
    if completed == list(STATE_IDS):
        try:
            provisional = analyze_trial(
                config.run_dir,
                config.motion_contract_path,
                strict_success=False,
            )
            analysis = provisional
            nonterminal_checks = {
                name: passed
                for name, passed in provisional["checks"].items()
                if name != "task_result_success"
            }
            failed_checks = [name for name, passed in nonterminal_checks.items() if not passed]
            if terminal.result is TaskResult.SUCCESS and failed_checks:
                terminal = terminate(
                    TaskResult.INCOMPLETE_CONTROLLER_BLOCKED,
                    "completed P01-P13 evidence failed physical or v010 conformance checks",
                    {"failed_checks": failed_checks, "conformance": provisional["conformance_summary"]},
                )
        except Exception as exc:
            if terminal.result is TaskResult.SUCCESS:
                terminal = terminate(
                    TaskResult.VIDEO_OR_ARTIFACT_ERROR,
                    f"pre-manifest trial analysis failed: {type(exc).__name__}: {exc}",
                )
    phase_windows = []
    for index, state_id in enumerate(STATE_IDS):
        start = float(phase_times[state_id].get("entry_time_s", 0.0))
        if index + 1 < len(STATE_IDS):
            end = float(phase_times[STATE_IDS[index + 1]].get("entry_time_s", start))
        else:
            end = duration_s
        phase_windows.append({"phase": state_id, "state": state_id, "start_s": start, "end_s": end})
    try:
        _append_artifact(
            writer,
            "task_event",
            {"event": "TRIAL_TERMINATION", "result": terminal.result.value, "state_id": terminal.state_id, "sim_time_s": terminal.sim_time_s, "reason": terminal.reason, "details": terminal.details},
        )
    except Exception:
        pass
    writer.close()
    if completed == list(STATE_IDS):
        try:
            analysis = populate_reference_similarity(config.run_dir, config.motion_contract_path)
        except Exception as exc:
            if terminal.result is TaskResult.SUCCESS:
                terminal = terminate(
                    TaskResult.VIDEO_OR_ARTIFACT_ERROR,
                    f"similarity artifact population failed: {type(exc).__name__}: {exc}",
                )
                with (config.run_dir / "task_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "event": "TRIAL_TERMINATION_SUPERSEDED",
                                "result": terminal.result.value,
                                "state_id": terminal.state_id,
                                "sim_time_s": terminal.sim_time_s,
                                "reason": terminal.reason,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    physical_success = bool(
        terminal.result is TaskResult.SUCCESS
        and completed == list(STATE_IDS)
        and not body_collision_seen
        and not wheel_only_seen
        and rear_order == "RR_FIRST"
        and duration_s <= MAX_CONTROL_SECONDS
        and video_manifest.get("valid") is True
        and environment_initialization.get("all_eight_servo_limits_applied") is True
        and environment_initialization.get("robot_source_asset_hash_unchanged") is True
        and environment_initialization.get("servo_target_mapping", {}).get(
            "source_environment_invariant"
        ) == "mature_ui_command_space_to_drive_target"
        and analysis.get("checks")
        and all(bool(value) for value in analysis["checks"].values())
    )
    success_evidence = {
        "task_result": terminal.result.value,
        "one_continuous_physical_fsm_success": physical_success,
        "completed_macro_phases": completed,
        "p01_p13_completed": completed == list(STATE_IDS),
        "body_collision": body_collision_seen,
        "wheel_only_climb": wheel_only_seen,
        "rear_leg_order": "RR_FIRST",
        "observed_rear_leg_order": rear_order,
        "duration_s": duration_s,
        "root_state_write_count": 0,
        "teleport_count": 0,
        "external_force_count": 0,
        "external_impulse_count": 0,
        "runtime_raw_recording_access": False,
        "authoritative_servo_limits_installed": bool(
            environment_initialization.get("all_eight_servo_limits_applied")
        ),
        "source_robot_usd_unchanged": bool(
            environment_initialization.get("robot_source_asset_hash_unchanged")
        ),
        "mature_servo_target_mapping": bool(
            environment_initialization.get("servo_target_mapping", {}).get(
                "source_environment_invariant"
            )
            == "mature_ui_command_space_to_drive_target"
        ),
    }
    first_blocker = None if controller is None else controller.first_blocker
    if first_blocker is None and terminal.result is not TaskResult.SUCCESS:
        first_blocker = {
            "state_id": terminal.state_id,
            "result": terminal.result.value,
            "reason": terminal.reason,
            "details": terminal.details,
        }
    manifest = {
        "trial_id": config.run_dir.name,
        "result": terminal.result.value,
        "reason": terminal.reason,
        "first_blocker": first_blocker,
        "reference_version": "v010_20260806_220745_363972_manual",
        "rear_leg_order": "RR_FIRST",
        "physics_hz": PHYSICS_HZ,
        "decision_hz": DECISION_HZ,
        "control_steps": control_steps,
        "settle_ticks": config.settle_ticks,
        "full12_control_write_count": control_steps,
        "full12_total_write_count": 0 if adapter is None else adapter.write_count,
        "render_count": control_steps // RENDER_STRIDE,
        "initial_app_update_count": 1,
        "app_update_count_during_recording": 0,
        "phase_times": phase_times,
        "phase_windows": phase_windows,
        "analysis_checks": analysis.get("checks", {}),
        "conformance": analysis.get("conformance_summary", {}),
        "anti_throttle_settings": anti_throttle,
        "environment_initialization": environment_initialization,
        "video": video_manifest,
        "success_evidence": success_evidence,
    }
    try:
        writer.finalize_manifest(manifest)
    except Exception:
        return 3
    print(json.dumps({"run_dir": str(config.run_dir), "result": terminal.result.value, "success": physical_success}, separators=(",", ":")))
    return 0 if physical_success else 2


def main(argv: Sequence[str] | None = None) -> int:
    # AppLauncher is the only non-stdlib import allowed before SimulationApp.
    from isaaclab.app import AppLauncher  # type: ignore

    parser = _parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except Exception as exc:
        parser.error(str(exc))
    if bool(getattr(args, "headless", False)):
        parser.error("the physical evidence trial requires the visible GUI viewport")

    simulation_app: Any | None = None
    try:
        launcher = AppLauncher(args)
        simulation_app = launcher.app
        simulation_app.update()
        return _run_live(config, simulation_app)
    except Exception as exc:
        _write_bootstrap_failure(
            config,
            exc,
            simulation_app_created=simulation_app is not None,
        )
        print(f"INFRASTRUCTURE_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
