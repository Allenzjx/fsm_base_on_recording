"""Live, single-environment bridge from projected PPO actions to the frozen FSM.

The module is intentionally safe to import before :class:`AppLauncher` has
created ``SimulationApp``.  All Isaac, scene, sensor, adapter, and controller
imports live in :func:`_load_live_dependencies` and therefore occur only when
``reset`` first needs the production runtime.  Tests may inject the small
``BackendDependencies`` seam and run without Isaac installed or imported.

The backend owns exactly one controller call and one atomic Full12 write for
each episode-relative 120 Hz physics tick.  The controller remains the source
of the nominal command, phase, lifecycle, task result, tracking set, and both
drive-feedback paths.  The frozen nominal is supplied unchanged to its mature
servo target mapper; the already-projected PPO delta is injected through the
mapper's bounded post-mapper drive-bias seam.  This prevents policy decisions
from being mistaken for new FSM motion segments and clearing the frozen
controller's tracking compensation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .action_projection import SafetyProjection
from .observation_schema import NonFiniteObservationError, PPOObservationFrame
from .ppo_env_adapter import AuthoritativeFrame
from .reward_terms import RewardSignals
from .termination import TerminationSignals


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FSM_PATH = PROJECT_ROOT / "configs" / "fsm_states.yaml"
DEFAULT_MOTION_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "recording_motion_contract.json"
)
DEFAULT_ENVIRONMENT_LOCK_PATH = PROJECT_ROOT / "configs" / "environment_lock.json"
DEFAULT_PHASE_SNAPSHOT_ROOT = PROJECT_ROOT / "reference" / "ppo_phase_snapshots"

PHYSICS_HZ = 120.0
PHYSICS_DT_S = 1.0 / PHYSICS_HZ
SETTLE_SECONDS = 1.5
SETTLE_TICKS = round(SETTLE_SECONDS * PHYSICS_HZ)
LEVEL_CALIBRATION_SECONDS = 0.25
LEVEL_CALIBRATION_TICKS = round(LEVEL_CALIBRATION_SECONDS * PHYSICS_HZ)
FULL12_SIZE = 12
ZERO_FULL12 = (0.0,) * FULL12_SIZE
PHASE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
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

_ACTIVE_LEG_BY_STATE = {
    "P02": "FR",
    "P03": "FR",
    "P05": "FL",
    "P09": "RR",
    "P12": "RL",
}
_LEG_TO_WHEEL = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}


class IsaacFSMBackendError(RuntimeError):
    """The live backend could not preserve its authoritative runtime contract."""


class SensorContractFailure(IsaacFSMBackendError):
    """Critical exact-pair or geometry sensing became untrustworthy."""


@dataclass(frozen=True, slots=True)
class LoadedPhaseSnapshot:
    """One hash-validated local phase-entry reset artifact."""

    phase_id: str
    payload: Mapping[str, Any]
    state_sha256: str
    file_sha256: str
    snapshot_path: Path


@dataclass(frozen=True, slots=True)
class CanonicalArticulationResetState:
    """Native articulation state captured before the first physical settle.

    The locked USD authors a non-zero standing joint pose while
    ``ArticulationCfg.InitialStateCfg(joint_pos={})`` leaves Isaac Lab's
    ``default_joint_pos`` cache at zero.  Consequently that cache is not a
    faithful reset source for this asset.  These tensors are cloned directly
    after the one scene-construction ``SimulationContext.reset()`` and are
    reused only at later episode reset boundaries.
    """

    root_state: Any
    joint_position: Any
    joint_velocity: Any
    instance_count: int
    state_sha256: str


@dataclass(frozen=True, slots=True)
class ResidualActuationPlan:
    """Auditable split between frozen nominal shaping and PPO actuation."""

    frozen_nominal_full12: tuple[float, ...]
    projected_applied_full12: tuple[float, ...]
    projected_residual_full12: tuple[float, ...]
    controller_drive_bias_full12: tuple[float, ...]
    combined_post_mapper_bias_full12: tuple[float, ...]

    def annotate_ack(self, ack: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(ack)
        result.update(
            {
                "ppo_actuation_contract": "frozen_nominal_plus_post_mapper_residual.v1",
                "fsm_nominal_mapper_input_full12": list(self.frozen_nominal_full12),
                "ppo_projected_applied_full12": list(self.projected_applied_full12),
                "ppo_projected_residual_full12": list(self.projected_residual_full12),
                "controller_drive_bias_full12": list(
                    self.controller_drive_bias_full12
                ),
                "combined_post_mapper_bias_full12": list(
                    self.combined_post_mapper_bias_full12
                ),
            }
        )
        return result


def build_residual_actuation_plan(
    projected_applied_full12: Sequence[float],
    *,
    frozen_nominal_full12: Sequence[float],
    drive_feedback_bias_full12: Sequence[float],
    normal_drive_bias_full12: Sequence[float],
) -> ResidualActuationPlan:
    """Keep the mature mapper on FSM nominal and add PPO after that mapper.

    ``ServoTargetMapper`` deliberately interprets a changed logical request as
    a new motion target and clears its sampled tracking compensation.  A PPO
    residual changes at 15 Hz and is not a new FSM segment, so feeding the sum
    into that mapper breaks the nominal controller even for tiny residuals.
    The existing bounded drive-bias seam is expressed in the same canonical
    Full12 units and is therefore the correct actuator-composition boundary.
    """

    projected = _full12(projected_applied_full12, "projected_applied_full12")
    nominal = _full12(frozen_nominal_full12, "frozen_nominal_full12")
    feedback = _full12(drive_feedback_bias_full12, "drive_feedback_bias_full12")
    normal = _full12(normal_drive_bias_full12, "normal_drive_bias_full12")
    residual = tuple(
        applied - baseline
        for applied, baseline in zip(projected, nominal, strict=True)
    )
    controller_bias = tuple(
        first + second for first, second in zip(feedback, normal, strict=True)
    )
    combined = tuple(
        baseline + delta
        for baseline, delta in zip(controller_bias, residual, strict=True)
    )
    if any(not math.isfinite(value) for value in controller_bias + combined):
        raise IsaacFSMBackendError("residual actuation plan contains non-finite bias")
    return ResidualActuationPlan(
        frozen_nominal_full12=nominal,
        projected_applied_full12=projected,
        projected_residual_full12=residual,
        controller_drive_bias_full12=controller_bias,
        combined_post_mapper_bias_full12=combined,
    )


@dataclass(frozen=True, slots=True)
class BackendDependencies:
    """Late-bound production dependencies and the Isaac-free unit-test seam."""

    create_scene: Callable[..., Any]
    create_sensing_backends: Callable[..., Any]
    adapter_from_scene: Callable[[Any], Any]
    reader_from_scene: Callable[..., Any]
    controller_from_paths: Callable[[Path, Path], Any]
    capture_reset_state: Callable[[Any], CanonicalArticulationResetState]
    reset_scene: Callable[
        [Any, CanonicalArticulationResetState], Mapping[str, Any]
    ]
    restore_settled_state: Callable[
        [Any, CanonicalArticulationResetState], Mapping[str, Any]
    ]
    locked_scene_snapshot: Callable[[], Mapping[str, Any]]
    expected_contact_bodies: tuple[str, ...]
    robot_asset_hash: str
    load_phase_snapshot: Callable[[str], LoadedPhaseSnapshot] | None = None
    write_phase_snapshot: Callable[..., Mapping[str, Any]] | None = None
    restore_controller_snapshot: Callable[..., Mapping[str, Any]] | None = None
    restore_guard_snapshot: Callable[..., Mapping[str, Any]] | None = None


def _load_live_dependencies() -> BackendDependencies:
    """Import Isaac-facing code only after the caller has launched Isaac."""

    from wlr50_clean.fsm.controller import SensorFsmController
    from wlr50_clean.infrastructure.robot_adapter import RobotAdapter
    from wlr50_clean.infrastructure.scene_factory import (
        ROBOT_USD_SHA256,
        create_scene,
        locked_scene_snapshot,
    )
    from wlr50_clean.sensing.contact_classifier import SENSED_BODIES
    from wlr50_clean.sensing.sensor_reader import (
        SensorReader,
        create_live_sensing_backends,
    )

    return BackendDependencies(
        create_scene=create_scene,
        create_sensing_backends=create_live_sensing_backends,
        adapter_from_scene=RobotAdapter.from_scene,
        reader_from_scene=lambda scene, adapter, backends: SensorReader.from_live_scene(
            scene, adapter, backends=backends
        ),
        controller_from_paths=SensorFsmController.from_paths,
        capture_reset_state=lambda scene: capture_canonical_articulation_reset_state(
            scene.robot
        ),
        reset_scene=_restore_default_articulation_state,
        restore_settled_state=_restore_canonical_settled_articulation_state,
        locked_scene_snapshot=locked_scene_snapshot,
        expected_contact_bodies=tuple(SENSED_BODIES),
        robot_asset_hash=str(ROBOT_USD_SHA256),
        load_phase_snapshot=_load_validated_phase_snapshot,
        write_phase_snapshot=_write_phase_snapshot_state,
        restore_controller_snapshot=_restore_controller_from_snapshot,
        restore_guard_snapshot=_restore_guard_tracker_from_snapshot,
    )


class IsaacFSMBackend:
    """One live Isaac scene driven by one authoritative frozen FSM instance.

    ``simulation_app`` must already have been created by ``AppLauncher`` for a
    production backend.  The backend never launches or closes the application,
    records video, opens a Recording, changes gravity, or applies forces.  Root
    and joint state restoration is confined to a subsequent ``reset`` boundary;
    :meth:`step_physics` has no state-write path other than ``apply_full12``.
    """

    def __init__(
        self,
        simulation_app: Any | None = None,
        *,
        fsm_path: Path | str = DEFAULT_FSM_PATH,
        motion_contract_path: Path | str = DEFAULT_MOTION_CONTRACT_PATH,
        dependencies: BackendDependencies | None = None,
    ) -> None:
        self.simulation_app = simulation_app
        self.fsm_path = Path(fsm_path).resolve()
        self.motion_contract_path = Path(motion_contract_path).resolve()
        self._dependencies = dependencies
        self._scene: Any | None = None
        self._adapter: Any | None = None
        self._reader: Any | None = None
        self._controller: Any | None = None
        self._raw_observation: Any | None = None
        self._controller_frame: Any | None = None
        self._authoritative_frame: AuthoritativeFrame | None = None
        self._last_valid_actor_observation: PPOObservationFrame | None = None
        self._last_atomic_ack: Mapping[str, Any] | None = None
        self._level_reference_orientation: tuple[float, float, float, float] | None = None
        self._level_calibration: dict[str, Any] = {}
        self._reset_metadata: dict[str, Any] = {}
        self._snapshot_restoration: dict[str, Any] = {}
        self._previous_action_full12 = ZERO_FULL12
        self._episode_tick = 0
        self._video_pre_action_tick_count = 0
        self._video_post_terminal_tick_count = 0
        self._reset_count = 0
        self._done = False
        self._body_collision_seen = False
        self._wheel_only_seen = False
        self._nan_inf_seen = False
        self._joint_limit_seen = False
        self._fall_seen = False
        self._physics_explosion_seen = False
        self._canonical_reset_state: CanonicalArticulationResetState | None = None
        self._canonical_settled_state: CanonicalArticulationResetState | None = None
        self._canonical_level_reference_orientation: (
            tuple[float, float, float, float] | None
        ) = None

    @property
    def raw_observation(self) -> Any | None:
        """Latest unnormalized frozen sensing frame."""

        return self._raw_observation

    @property
    def controller_frame(self) -> Any | None:
        """Latest raw :class:`SensorFsmController` output frame."""

        return self._controller_frame

    @property
    def level_calibration(self) -> Mapping[str, Any]:
        """Latest raw and home-relative body-level measurements."""

        return dict(self._level_calibration)

    @property
    def last_atomic_ack(self) -> Mapping[str, Any] | None:
        return None if self._last_atomic_ack is None else dict(self._last_atomic_ack)

    def reset(
        self, *, seed: int, options: Mapping[str, Any]
    ) -> AuthoritativeFrame:
        """Restore/create the locked scene, settle for 1.5 s, and start P01.

        The production environment is deterministic at this stage.  A true
        randomization request is rejected rather than silently mutating the
        frozen baseline.  The seed is still recorded for paired rollouts.
        """

        reset_seed = _non_negative_seed(seed)
        reset_options = dict(options)
        _validate_reset_options(reset_options)
        dependencies = self._dependencies or _load_live_dependencies()
        self._dependencies = dependencies
        snapshot_phase = _snapshot_phase_option(reset_options)
        loaded_snapshot: LoadedPhaseSnapshot | None = None
        if snapshot_phase is not None:
            if dependencies.load_phase_snapshot is None:
                raise IsaacFSMBackendError(
                    "phase curriculum requested but no validated snapshot loader exists"
                )
            loaded_snapshot = dependencies.load_phase_snapshot(snapshot_phase)
            if loaded_snapshot.phase_id != snapshot_phase:
                raise IsaacFSMBackendError(
                    "phase snapshot loader returned a different FSM state"
                )
            _validate_phase_snapshot_payload(loaded_snapshot.payload, snapshot_phase)
            if snapshot_phase != "P01" and any(
                callback is None
                for callback in (
                    dependencies.write_phase_snapshot,
                    dependencies.restore_guard_snapshot,
                    dependencies.restore_controller_snapshot,
                )
            ):
                raise IsaacFSMBackendError(
                    "phase curriculum lacks physical/controller/guard restoration seams"
                )

        reset_writes = {
            "root_pose_writes": 0,
            "root_velocity_writes": 0,
            "joint_state_writes": 0,
            "global_simulation_resets": 0,
            "simulation_forward_syncs": 0,
        }
        if self._scene is None:
            self._scene = dependencies.create_scene(
                simulation_app=self.simulation_app,
                before_reset=lambda sim, robot: dependencies.create_sensing_backends(
                    sim=sim, robot=robot
                ),
            )
            self._canonical_reset_state = dependencies.capture_reset_state(
                self._scene
            )
        else:
            if self._canonical_reset_state is None:
                raise IsaacFSMBackendError(
                    "canonical pre-settle articulation state is unavailable"
                )
            reset_writes = dict(
                dependencies.reset_scene(
                    self._scene, self._canonical_reset_state
                )
            )

        scene = self._scene
        backends = getattr(scene, "instrumentation", None)
        contact_backend = getattr(backends, "contact_backend", None)
        if backends is None or contact_backend is None or not bool(
            getattr(contact_backend, "initialized", False)
        ):
            raise SensorContractFailure(
                "the exact 13-body ContactSensor bank did not initialize"
            )

        adapter = dependencies.adapter_from_scene(scene)
        calibration_reader: Any | None = None
        calibration_samples: list[tuple[float, float, float, float]] = []
        calibration_start = SETTLE_TICKS - LEVEL_CALIBRATION_TICKS
        latest_settle_ack: Mapping[str, Any] | None = None
        for settle_tick in range(SETTLE_TICKS):
            _require_running(scene, "zero-command physical settle")
            latest_settle_ack = self._atomic_apply(
                adapter,
                ZERO_FULL12,
                physics_tick=settle_tick,
                tracking_servo_names=(),
                drive_feedback_bias_full12=ZERO_FULL12,
            )
            scene.sim.step(render=False)
            adapter.update_readback()
            if settle_tick >= calibration_start:
                if calibration_reader is None:
                    calibration_reader = dependencies.reader_from_scene(
                        scene, adapter, backends
                    )
                local_tick = settle_tick - calibration_start
                sample = calibration_reader.read(
                    physics_tick=local_tick,
                    simulation_time_s=local_tick * PHYSICS_DT_S,
                    commanded_full12=latest_settle_ack["drive_target_full12"],
                )
                _validate_sensor_contract(
                    sample,
                    dependencies.expected_contact_bodies,
                    require_finite=True,
                )
                calibration_samples.append(_observation_quaternion(sample))

        if len(calibration_samples) != LEVEL_CALIBRATION_TICKS:
            raise IsaacFSMBackendError(
                "level calibration did not observe the complete stable window"
            )
        verify_limits = getattr(adapter, "verify_authoritative_servo_limits_adopted", None)
        if not callable(verify_limits):
            raise IsaacFSMBackendError(
                "RobotAdapter lacks authoritative servo-limit verification"
            )
        verify_limits()
        self._level_reference_orientation = _mean_quaternion(calibration_samples)

        normal_p01_reset = loaded_snapshot is None or snapshot_phase == "P01"
        if self._canonical_settled_state is None:
            self._canonical_settled_state = dependencies.capture_reset_state(scene)
            self._canonical_level_reference_orientation = (
                self._level_reference_orientation
            )
        elif normal_p01_reset:
            settled_proof = dict(
                dependencies.restore_settled_state(
                    scene, self._canonical_settled_state
                )
            )
            reset_writes = _merge_reset_writes(reset_writes, settled_proof)
            reset_writes.update(
                {
                    "canonical_settled_restore_applied": True,
                    "canonical_settled_applied_sha256": (
                        self._canonical_settled_state.state_sha256
                    ),
                }
            )
            if self._canonical_level_reference_orientation is None:
                raise IsaacFSMBackendError(
                    "canonical post-settle level reference is unavailable"
                )
            self._level_reference_orientation = (
                self._canonical_level_reference_orientation
            )

        self._snapshot_restoration = {
            "requested_phase": snapshot_phase,
            "mode": "normal_p01_reset",
            "snapshot_validated": loaded_snapshot is not None,
        }
        initial_command = ZERO_FULL12
        if loaded_snapshot is not None:
            self._snapshot_restoration.update(
                {
                    "state_sha256": loaded_snapshot.state_sha256,
                    "file_sha256": loaded_snapshot.file_sha256,
                    "snapshot_path": str(loaded_snapshot.snapshot_path),
                    "source_tick": int(loaded_snapshot.payload["source_tick"]),
                }
            )
        if loaded_snapshot is not None and snapshot_phase != "P01":
            assert dependencies.write_phase_snapshot is not None
            physical_proof = dict(
                dependencies.write_phase_snapshot(
                    scene, adapter, loaded_snapshot.payload
                )
            )
            reset_writes = _merge_reset_writes(reset_writes, physical_proof)
            self._snapshot_restoration.update(
                {
                    "mode": "phase_entry_snapshot",
                    "physical_state": physical_proof,
                }
            )
            self._level_reference_orientation = _normalized_quaternion(
                loaded_snapshot.payload["level_reference_orientation_wxyz"]
            )
            initial_command = _full12(
                loaded_snapshot.payload["applied_full12"],
                "phase snapshot applied_full12",
            )

        reader = dependencies.reader_from_scene(scene, adapter, backends)
        controller = dependencies.controller_from_paths(
            self.fsm_path, self.motion_contract_path
        )
        _validate_rate_contract(adapter, controller)
        if loaded_snapshot is not None and snapshot_phase != "P01":
            assert dependencies.restore_guard_snapshot is not None
            assert dependencies.restore_controller_snapshot is not None
            guard_proof = dict(
                dependencies.restore_guard_snapshot(
                    reader, loaded_snapshot.payload
                )
            )
            controller_proof = dict(
                dependencies.restore_controller_snapshot(
                    controller, loaded_snapshot.payload
                )
            )
            self._snapshot_restoration.update(
                {
                    "guard_state": guard_proof,
                    "controller_state": controller_proof,
                }
            )
        observation = reader.read(
            physics_tick=0,
            simulation_time_s=0.0,
            commanded_full12=initial_command,
        )
        _validate_sensor_contract(
            observation,
            dependencies.expected_contact_bodies,
            require_finite=True,
        )
        if loaded_snapshot is not None and snapshot_phase != "P01":
            observation_proof = _verify_phase_snapshot_observation(
                observation, loaded_snapshot.payload
            )
            self._snapshot_restoration["live_observation"] = observation_proof
        controller_frame = controller.step(observation, sim_time_s=0.0)
        _validate_controller_clock(controller_frame, physics_tick=0, sim_time_s=0.0)
        expected_state = snapshot_phase or "P01"
        if str(getattr(controller_frame, "state_id", "")) != expected_state:
            raise IsaacFSMBackendError(
                "restored frozen controller did not emit the requested phase"
            )

        self._adapter = adapter
        self._reader = reader
        self._controller = controller
        self._raw_observation = observation
        self._controller_frame = controller_frame
        self._previous_action_full12 = initial_command
        self._episode_tick = 0
        self._video_pre_action_tick_count = 0
        self._video_post_terminal_tick_count = 0
        self._reset_count += 1
        self._done = False
        self._body_collision_seen = False
        self._wheel_only_seen = False
        self._nan_inf_seen = False
        self._joint_limit_seen = False
        self._fall_seen = False
        self._physics_explosion_seen = False
        self._last_valid_actor_observation = None
        self._last_atomic_ack = latest_settle_ack
        self._reset_metadata = self._make_reset_metadata(
            observation,
            seed=reset_seed,
            options=reset_options,
            reset_writes=reset_writes,
        )
        result = self._build_authoritative_frame(
            observation,
            controller_frame,
            previous_frame=None,
        )
        self._authoritative_frame = result
        self._done = _frame_is_terminal(result)
        return result

    def step_physics(
        self, applied_action_full12: Sequence[float]
    ) -> AuthoritativeFrame:
        """Atomically write one projected Full12, then advance physics once."""

        action = _full12(applied_action_full12, "applied_action_full12")
        if (
            self._scene is None
            or self._adapter is None
            or self._reader is None
            or self._controller is None
            or self._controller_frame is None
            or self._authoritative_frame is None
        ):
            raise IsaacFSMBackendError("reset(seed, options) must precede step_physics")
        if self._done:
            raise IsaacFSMBackendError("step_physics called after authoritative termination")

        scene = self._scene
        _require_running(scene, "continuous PPO episode")
        source_frame = self._controller_frame
        if not bool(getattr(source_frame, "full12_atomic_write_required", False)):
            raise IsaacFSMBackendError(
                "frozen controller did not require one complete Full12 write"
            )
        actuation = build_residual_actuation_plan(
            action,
            frozen_nominal_full12=getattr(source_frame, "full12"),
            drive_feedback_bias_full12=getattr(
                source_frame, "drive_feedback_bias_full12"
            ),
            normal_drive_bias_full12=getattr(
                source_frame, "normal_drive_bias_full12"
            ),
        )
        physical_tick = (
            SETTLE_TICKS + self._video_pre_action_tick_count + self._episode_tick
        )
        raw_ack = self._atomic_apply(
            self._adapter,
            actuation.frozen_nominal_full12,
            physics_tick=physical_tick,
            tracking_servo_names=tuple(
                str(name) for name in getattr(source_frame, "tracking_servo_names", ())
            ),
            drive_feedback_bias_full12=(
                actuation.combined_post_mapper_bias_full12
            ),
        )
        if (
            _full12(raw_ack["applied_full12"], "atomic ack applied_full12")
            != actuation.frozen_nominal_full12
        ):
            raise IsaacFSMBackendError(
                "RobotAdapter silently changed the frozen nominal mapper input"
            )
        ack = actuation.annotate_ack(raw_ack)

        # This is the only physics advance in the episode tick.  In particular,
        # no render, root-state write, force, impulse, or gravity mutation is
        # hidden in this method.
        scene.sim.step(render=False)
        self._adapter.update_readback()
        next_tick = self._episode_tick + 1
        next_time = next_tick * PHYSICS_DT_S
        try:
            observation = self._reader.read(
                physics_tick=next_tick,
                simulation_time_s=next_time,
                # Preserve both frozen drive-feedback paths in sensor error
                # calculations; this is deliberately not the logical action.
                commanded_full12=ack["drive_target_full12"],
            )
            assert self._dependencies is not None
            _validate_sensor_contract(
                observation,
                self._dependencies.expected_contact_bodies,
                require_finite=False,
            )
        except SensorContractFailure as exc:
            abort = getattr(self._controller, "abort_infrastructure", None)
            if callable(abort):
                abort(str(exc), sim_time_s=next_time)
            raise

        controller_frame = self._controller.step(
            observation, sim_time_s=next_time
        )
        _validate_controller_clock(
            controller_frame, physics_tick=next_tick, sim_time_s=next_time
        )
        previous = self._authoritative_frame
        self._episode_tick = next_tick
        self._previous_action_full12 = action
        self._raw_observation = observation
        self._controller_frame = controller_frame
        self._last_atomic_ack = ack
        result = self._build_authoritative_frame(
            observation,
            controller_frame,
            previous_frame=previous,
        )
        self._authoritative_frame = result
        self._done = _frame_is_terminal(result)
        return result

    def advance_video_pre_action_tick(self) -> Mapping[str, Any]:
        """Advance one real zero-command tick before the first filmed action.

        This hook exists only for honest viewport pre-roll.  It advances the
        live physics scene and issues one atomic standing command, but does not
        advance the frozen FSM clock or count as an episode action.  It is
        unavailable after the first controller tick.
        """

        if (
            self._scene is None
            or self._adapter is None
            or self._reader is None
            or self._dependencies is None
            or self._controller_frame is None
            or self._authoritative_frame is None
        ):
            raise IsaacFSMBackendError("reset must precede video pre-roll")
        if self._episode_tick != 0 or self._done:
            raise IsaacFSMBackendError(
                "video pre-roll is allowed only before the first episode tick"
            )
        _require_running(self._scene, "viewport pre-action hold")
        physical_tick = SETTLE_TICKS + self._video_pre_action_tick_count
        ack = self._atomic_apply(
            self._adapter,
            ZERO_FULL12,
            physics_tick=physical_tick,
            tracking_servo_names=(),
            drive_feedback_bias_full12=ZERO_FULL12,
        )
        self._scene.sim.step(render=False)
        self._adapter.update_readback()
        self._video_pre_action_tick_count += 1
        self._last_atomic_ack = ack
        evidence_tick = self._video_pre_action_tick_count
        observation = self._reader.read(
            physics_tick=evidence_tick,
            simulation_time_s=evidence_tick * PHYSICS_DT_S,
            commanded_full12=ack["drive_target_full12"],
        )
        _validate_sensor_contract(
            observation,
            self._dependencies.expected_contact_bodies,
            require_finite=True,
        )
        self._raw_observation = observation
        signals, termination_evidence = self._termination_signals(
            observation, self._controller_frame
        )
        if any(
            (
                signals.body_collision,
                signals.wheel_only_climb,
                signals.fall,
                signals.nan_inf,
                signals.hard_joint_limit,
                signals.physics_explosion,
            )
        ):
            raise IsaacFSMBackendError(
                "a physical hazard appeared during the pre-action video hold"
            )
        return {
            "schema": "wlr50_clean.ppo_video_physical_hold_tick.v1",
            "kind": "pre_action",
            "physical_tick": physical_tick,
            "video_hold_tick": self._video_pre_action_tick_count,
            "applied_full12": tuple(ack["applied_full12"]),
            "body_collision": signals.body_collision,
            "wheel_only_climb": signals.wheel_only_climb,
            "fall": signals.fall,
            "nan_inf": signals.nan_inf,
            "hard_joint_limit": signals.hard_joint_limit,
            "physics_explosion": signals.physics_explosion,
            "termination_evidence": termination_evidence,
            "root_state_write_count": 0,
        }

    def refresh_video_pre_action_frame(self) -> AuthoritativeFrame:
        """Refresh live sensing after filmed pre-roll without advancing the FSM.

        The physical hold intentionally leaves the episode-relative controller
        clock at tick zero.  Before the first policy action, re-read that live
        physical state and rebuild the actor observation around the unchanged
        P01 controller frame.  This is neither a reset nor an FSM step.
        """

        if (
            self._scene is None
            or self._adapter is None
            or self._reader is None
            or self._dependencies is None
            or self._controller_frame is None
            or self._authoritative_frame is None
            or self._last_atomic_ack is None
        ):
            raise IsaacFSMBackendError("reset must precede video pre-roll refresh")
        if self._video_pre_action_tick_count <= 0 or self._episode_tick != 0 or self._done:
            raise IsaacFSMBackendError(
                "video pre-roll refresh requires a live unstepped P01 episode"
            )
        backends = getattr(self._scene, "instrumentation", None)
        contact_backend = getattr(backends, "contact_backend", None)
        if backends is None or contact_backend is None or not bool(
            getattr(contact_backend, "initialized", False)
        ):
            raise SensorContractFailure(
                "the exact 13-body ContactSensor bank became unavailable during video pre-roll"
            )
        # The original SensorReader has consumed logical ticks 0..N while
        # auditing every physical pre-roll hold.  A new reader is required to
        # begin the actual episode at logical tick zero; re-reading tick zero
        # from the original reader would violate its contiguous-clock guard.
        refreshed_reader = self._dependencies.reader_from_scene(
            self._scene, self._adapter, backends
        )
        observation = refreshed_reader.read(
            physics_tick=0,
            simulation_time_s=0.0,
            commanded_full12=self._last_atomic_ack["drive_target_full12"],
        )
        _validate_sensor_contract(
            observation,
            self._dependencies.expected_contact_bodies,
            require_finite=True,
        )
        self._reader = refreshed_reader
        self._raw_observation = observation
        refreshed = self._build_authoritative_frame(
            observation,
            self._controller_frame,
            previous_frame=None,
        )
        if (
            refreshed.physics_tick != 0
            or refreshed.sim_time_s != 0.0
            or refreshed.state_id != "P01"
            or _frame_is_terminal(refreshed)
        ):
            raise IsaacFSMBackendError(
                "video pre-roll refresh changed the initial P01 controller state"
            )
        refresh_evidence = {
            "schema": "wlr50_clean.ppo_video_pre_action_refresh.v1",
            "physical_pre_action_ticks": self._video_pre_action_tick_count,
            "pre_roll_reader_last_logical_tick": self._video_pre_action_tick_count,
            "episode_reader_reinitialized": True,
            "episode_reader_first_logical_tick": 0,
            "controller_frame_preserved": True,
            "controller_logical_tick": 0,
            "simulation_reset_performed": False,
            "fsm_step_performed": False,
        }
        refreshed = replace(
            refreshed,
            info={
                **dict(refreshed.info),
                "video_pre_action_refresh": refresh_evidence,
            },
        )
        self._authoritative_frame = refreshed
        return refreshed

    def advance_video_post_success_tick(self) -> Mapping[str, Any]:
        """Advance one real post-success hold tick and recheck live hazards."""

        if (
            self._scene is None
            or self._adapter is None
            or self._reader is None
            or self._controller_frame is None
            or self._authoritative_frame is None
        ):
            raise IsaacFSMBackendError("reset must precede video post-roll")
        if not self._done or not self._authoritative_frame.termination_signals.success:
            raise IsaacFSMBackendError(
                "video post-roll requires authoritative task success"
            )
        _require_running(self._scene, "viewport post-success hold")
        physical_tick = (
            SETTLE_TICKS
            + self._video_pre_action_tick_count
            + self._episode_tick
            + self._video_post_terminal_tick_count
        )
        # Preserve terminal servo posture while making the completed wheel-stop
        # condition explicit during the physical hold.
        hold_action = self._previous_action_full12[:8] + (0.0,) * 4
        source_frame = self._controller_frame
        feedback = _full12(
            getattr(source_frame, "drive_feedback_bias_full12"),
            "controller drive_feedback_bias_full12",
        )
        normal = _full12(
            getattr(source_frame, "normal_drive_bias_full12"),
            "controller normal_drive_bias_full12",
        )
        total_drive_bias = tuple(
            first + second for first, second in zip(feedback, normal, strict=True)
        )
        ack = self._atomic_apply(
            self._adapter,
            hold_action,
            physics_tick=physical_tick,
            tracking_servo_names=tuple(
                str(name) for name in getattr(source_frame, "tracking_servo_names", ())
            ),
            drive_feedback_bias_full12=total_drive_bias,
        )
        self._scene.sim.step(render=False)
        self._adapter.update_readback()
        self._video_post_terminal_tick_count += 1
        evidence_tick = self._episode_tick + self._video_post_terminal_tick_count
        observation = self._reader.read(
            physics_tick=evidence_tick,
            simulation_time_s=evidence_tick * PHYSICS_DT_S,
            commanded_full12=ack["drive_target_full12"],
        )
        assert self._dependencies is not None
        _validate_sensor_contract(
            observation,
            self._dependencies.expected_contact_bodies,
            require_finite=True,
        )
        self._raw_observation = observation
        self._last_atomic_ack = ack
        signals, termination_evidence = self._termination_signals(
            observation, source_frame
        )
        if any(
            (
                signals.body_collision,
                signals.wheel_only_climb,
                signals.fall,
                signals.nan_inf,
                signals.hard_joint_limit,
                signals.physics_explosion,
            )
        ):
            raise IsaacFSMBackendError(
                "a physical hazard appeared during the post-success video hold"
            )
        return {
            "schema": "wlr50_clean.ppo_video_physical_hold_tick.v1",
            "kind": "post_success",
            "physical_tick": physical_tick,
            "video_hold_tick": self._video_post_terminal_tick_count,
            "applied_full12": tuple(ack["applied_full12"]),
            "body_collision": signals.body_collision,
            "wheel_only_climb": signals.wheel_only_climb,
            "fall": signals.fall,
            "nan_inf": signals.nan_inf,
            "hard_joint_limit": signals.hard_joint_limit,
            "physics_explosion": signals.physics_explosion,
            "termination_evidence": termination_evidence,
            "root_state_write_count": 0,
        }

    def render_video_frame(self) -> None:
        """Render the current live scene once for the active viewport recorder."""

        if self._scene is None:
            raise IsaacFSMBackendError("reset must precede viewport rendering")
        _require_running(self._scene, "active viewport render")
        self._scene.sim.render()

    def _atomic_apply(
        self,
        adapter: Any,
        command: Sequence[float],
        *,
        physics_tick: int,
        tracking_servo_names: Sequence[str],
        drive_feedback_bias_full12: Sequence[float],
    ) -> Mapping[str, Any]:
        before = int(getattr(adapter, "write_count", -1))
        if before < 0:
            raise IsaacFSMBackendError("RobotAdapter.write_count is unavailable")
        ack = adapter.apply_full12(
            command,
            physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names,
            drive_feedback_bias_full12=drive_feedback_bias_full12,
        )
        if not isinstance(ack, Mapping):
            raise IsaacFSMBackendError("RobotAdapter returned no atomic Full12 ack")
        after = int(getattr(adapter, "write_count", -1))
        if after != before + 1:
            raise IsaacFSMBackendError(
                "one backend tick must increment RobotAdapter.write_count exactly once"
            )
        required = {
            "physics_tick",
            "write_count",
            "articulation_writes_this_call",
            "motion_start_skew_s",
            "applied_full12",
            "drive_target_full12",
            "drive_feedback_bias_requested_full12",
        }
        missing = sorted(required - set(ack))
        if missing:
            raise IsaacFSMBackendError(f"atomic Full12 ack is incomplete: {missing}")
        if (
            int(ack["physics_tick"]) != int(physics_tick)
            or int(ack["write_count"]) != after
            or int(ack["articulation_writes_this_call"]) != 1
            or float(ack["motion_start_skew_s"]) != 0.0
        ):
            raise IsaacFSMBackendError("atomic Full12 articulation write contract failed")
        _full12(ack["applied_full12"], "atomic ack applied_full12")
        _full12(ack["drive_target_full12"], "atomic ack drive_target_full12")
        requested_bias = _full12(
            ack["drive_feedback_bias_requested_full12"],
            "atomic ack drive_feedback_bias_requested_full12",
        )
        if requested_bias != tuple(float(value) for value in drive_feedback_bias_full12):
            raise IsaacFSMBackendError("RobotAdapter did not preserve drive-feedback input")
        return dict(ack)

    def _build_authoritative_frame(
        self,
        observation: Any,
        controller_frame: Any,
        *,
        previous_frame: AuthoritativeFrame | None,
    ) -> AuthoritativeFrame:
        controller = self._controller
        if controller is None:
            raise IsaacFSMBackendError("controller is unavailable")
        phase = getattr(controller, "phase", None)
        if phase is None:
            raise IsaacFSMBackendError("frozen controller phase is unavailable")
        state_id = str(getattr(controller_frame, "state_id"))
        macro_phase = int(getattr(phase, "macro_phase"))
        phase_progress = _phase_progress(controller, state_id)
        nominal = _full12(getattr(controller_frame, "full12"), "nominal Full12")
        reference = _reference_action(controller, controller_frame, phase)
        reference_delta = _full12(
            getattr(phase, "delta_full12"), "phase reference delta"
        )
        action_mask = tuple(int(value) for value in getattr(phase, "action_mask_full12"))
        if len(action_mask) != FULL12_SIZE or any(value not in (0, 1) for value in action_mask):
            raise IsaacFSMBackendError("frozen phase action mask is not binary Full12")

        termination, termination_details = self._termination_signals(
            observation, controller_frame
        )
        level = _level_measurement(
            observation, self._level_reference_orientation
        )
        self._level_calibration = level
        actor_fallback = False
        try:
            actor_observation = PPOObservationFrame.from_live_observation(
                observation,
                state_id=state_id,
                macro_phase=macro_phase,
                phase_progress=phase_progress,
                previous_action_full12=self._previous_action_full12,
            )
        except NonFiniteObservationError:
            if not termination.nan_inf or self._last_valid_actor_observation is None:
                raise
            # Preserve a finite terminal transition for the trainer.  Only the
            # actor payload falls back; the raw invalid observation remains
            # exposed and NAN_INF remains the authoritative termination signal.
            actor_fallback = True
            actor_observation = replace(
                self._last_valid_actor_observation,
                state_id=state_id,
                macro_phase=macro_phase,
                phase_progress=phase_progress,
                previous_action_full12=self._previous_action_full12,
            )
        if not actor_fallback:
            self._last_valid_actor_observation = actor_observation

        prior_raw = (
            None
            if previous_frame is None
            else previous_frame.info.get("raw_observation")
        )
        prior_x = _base_x(prior_raw)
        current_x = _base_x(observation)
        forward_delta = (
            0.0
            if previous_frame is None
            or prior_x is None
            or current_x is None
            else current_x - prior_x
        )
        progress_delta = (
            0.0
            if previous_frame is None or previous_frame.state_id != state_id
            else phase_progress - previous_frame.phase_progress
        )
        support = _member(observation, "support")
        support_margin = _member(support, "signed_margin_m")
        support_valid = bool(_member(support, "valid", False))
        angular_speed = _norm3(
            _member(_member(observation, "base"), "angular_velocity_w_rad_s", ())
        )
        rewards = RewardSignals(
            task_success=termination.success,
            forward_progress_delta_m=_finite_or_zero(forward_delta),
            phase_progress_delta=_finite_or_zero(progress_delta),
            active_leg_clearance_m=_active_leg_clearance(observation, state_id),
            body_collision=termination.body_collision,
            wheel_only_climb=termination.wheel_only_climb,
            fall=termination.fall,
            joint_limit_violation=termination.hard_joint_limit,
            body_angular_speed_rad_s=_finite_or_zero(angular_speed),
            pitch_rad=_finite_or_zero(level["pitch_error_to_level_rad"]),
            roll_rad=_finite_or_zero(level["roll_error_to_level_rad"]),
            support_margin_m=(
                float(support_margin)
                if support_margin is not None and math.isfinite(float(support_margin))
                else None
            ),
            support_valid=support_valid,
        )
        safety = SafetyProjection(
            residual_enabled=not any(
                (
                    termination.body_collision,
                    termination.wheel_only_climb,
                    termination.fall,
                    termination.nan_inf,
                    termination.hard_joint_limit,
                    termination.physics_explosion,
                )
            ),
            channel_mask_full12=(1,) * FULL12_SIZE,
            force_wheels_zero=bool(
                termination.body_collision
                or termination.wheel_only_climb
                or termination.fall
                or termination.nan_inf
                or termination.hard_joint_limit
                or termination.physics_explosion
            ),
            body_collision_detected=termination.body_collision,
            wheel_only_climb_detected=termination.wheel_only_climb,
            reason=termination_details.get("primary_source"),
        )
        lifecycle = _enum_value(getattr(controller_frame, "lifecycle", "UNKNOWN"))
        task_termination = getattr(controller_frame, "termination", None)
        task_result = _enum_value(getattr(task_termination, "result", None))
        ack = None if self._last_atomic_ack is None else dict(self._last_atomic_ack)
        info = {
            **self._reset_metadata,
            "raw_observation": observation,
            "raw_controller_frame": controller_frame,
            "level_calibration": dict(level),
            "controller_lifecycle": lifecycle,
            "controller_task_result": task_result,
            "controller_termination": task_termination,
            "termination_mapping": termination_details,
            "atomic_ack": ack,
            "drive_target_full12": (
                list(ZERO_FULL12)
                if ack is None
                else list(ack["drive_target_full12"])
            ),
            "drive_feedback_bias_requested_full12": (
                list(ZERO_FULL12)
                if ack is None
                else list(ack["drive_feedback_bias_requested_full12"])
            ),
            "actor_observation_fallback_due_to_nonfinite": actor_fallback,
            "in_episode_root_pose_writes": 0,
            "in_episode_root_velocity_writes": 0,
            "in_episode_force_or_impulse_writes": 0,
            "in_episode_gravity_writes": 0,
            "recording_accesses": 0,
        }
        return AuthoritativeFrame(
            physics_tick=int(getattr(controller_frame, "physics_tick")),
            sim_time_s=float(getattr(controller_frame, "sim_time_s")),
            state_id=state_id,
            macro_phase=macro_phase,
            phase_progress=phase_progress,
            observation=actor_observation,
            nominal_action_full12=nominal,
            reference_action_full12=reference,
            reference_delta_full12=reference_delta,
            action_mask_full12=action_mask,
            reward_signals=rewards,
            termination_signals=termination,
            safety_projection=safety,
            info=info,
        )

    def _termination_signals(
        self, observation: Any, controller_frame: Any
    ) -> tuple[TerminationSignals, dict[str, Any]]:
        body_now = bool(
            _member(_member(observation, "body_collision"), "detected", False)
        )
        wheel_now = _guard_asserted(observation, "wheel_only_climb_detected")
        nan_now = not bool(_member(observation, "all_finite", False)) or _guard_asserted(
            observation, "non_finite_observation_or_command"
        )
        joint_now = _guard_asserted(observation, "joint_hard_limit_violation")
        fall_now, explosion_now, physics_values = _fall_and_explosion(observation)

        controller_termination = getattr(controller_frame, "termination", None)
        controller_result = _enum_value(
            getattr(controller_termination, "result", None)
        )
        body_now |= controller_result == "TASK_FAILURE_BODY_COLLISION"
        wheel_now |= controller_result == "TASK_FAILURE_WHEEL_ONLY_CLIMB"
        if controller_result == "SAFETY_ABORT" and not any(
            (nan_now, joint_now, fall_now, explosion_now)
        ):
            # A new frozen safety guard cannot be silently treated as success.
            # PHYSICS_EXPLOSION is the conservative existing safety bucket;
            # the exact controller result/reason remains visible in info.
            explosion_now = True

        self._body_collision_seen |= body_now
        self._wheel_only_seen |= wheel_now
        self._nan_inf_seen |= nan_now
        self._joint_limit_seen |= joint_now
        self._fall_seen |= fall_now
        self._physics_explosion_seen |= explosion_now
        blocked = controller_result == "INCOMPLETE_CONTROLLER_BLOCKED"
        success = bool(
            controller_result == "SUCCESS"
            and not any(
                (
                    self._body_collision_seen,
                    self._wheel_only_seen,
                    self._nan_inf_seen,
                    self._joint_limit_seen,
                    self._fall_seen,
                    self._physics_explosion_seen,
                )
            )
        )
        signals = TerminationSignals(
            success=success,
            body_collision=self._body_collision_seen,
            wheel_only_climb=self._wheel_only_seen,
            fall=self._fall_seen,
            nan_inf=self._nan_inf_seen,
            hard_joint_limit=self._joint_limit_seen,
            physics_explosion=self._physics_explosion_seen,
            # The current public termination ABI has no controller-blocked
            # member.  Treat it as a non-task truncation and preserve its exact
            # classification below rather than inventing a physical failure.
            timeout=blocked,
            reference_conformance_outside_30pct=False,
        )
        active = [
            name
            for name, value in (
                ("NAN_INF", signals.nan_inf),
                ("PHYSICS_EXPLOSION", signals.physics_explosion),
                ("BODY_COLLISION", signals.body_collision),
                ("WHEEL_ONLY_CLIMB", signals.wheel_only_climb),
                ("FALL", signals.fall),
                ("HARD_JOINT_LIMIT", signals.hard_joint_limit),
                ("SUCCESS", signals.success),
                ("CONTROLLER_BLOCKED", blocked),
            )
            if value
        ]
        return signals, {
            "schema": "wlr50_clean.isaac_fsm_backend.termination_mapping.v1",
            "controller_result": controller_result,
            "controller_reason": getattr(controller_termination, "reason", None),
            "controller_details": dict(
                getattr(controller_termination, "details", {}) or {}
            ),
            "first_blocker": dict(
                getattr(controller_frame, "first_blocker", {}) or {}
            ),
            "active_sources": tuple(active),
            "primary_source": active[0] if active else None,
            "physics_guard_values": physics_values,
            "controller_blocked_encoded_as_truncation": blocked,
        }

    def _make_reset_metadata(
        self,
        observation: Any,
        *,
        seed: int,
        options: Mapping[str, Any],
        reset_writes: Mapping[str, int],
    ) -> dict[str, Any]:
        assert self._dependencies is not None
        scene_snapshot = dict(self._dependencies.locked_scene_snapshot())
        base = _member(observation, "base")
        root_state = (
            tuple(float(value) for value in _member(base, "position_w_m", ()))
            + tuple(float(value) for value in _member(base, "orientation_wxyz", ()))
            + tuple(float(value) for value in _member(base, "linear_velocity_w_m_s", ()))
            + tuple(float(value) for value in _member(base, "angular_velocity_w_rad_s", ()))
        )
        actual = _full12(_member(observation, "actual_full12"), "initial actual Full12")
        joints = _member(observation, "joints", {})
        wheels = _member(observation, "wheels", {})
        joint_velocity = tuple(
            float(_member(row, "velocity_deg_s", 0.0)) for row in joints.values()
        ) + tuple(float(_member(row, "velocity_rad_s", 0.0)) for row in wheels.values())
        initial_joint_state = actual + joint_velocity
        obstacle = _member(observation, "obstacle")
        obstacle_pose = (
            0.5
            * (
                float(_member(obstacle, "front_x_m"))
                + float(_member(obstacle, "back_x_m"))
            ),
            0.5
            * (
                float(_member(obstacle, "left_y_m"))
                + float(_member(obstacle, "right_y_m"))
            ),
            0.5
            * (
                float(_member(obstacle, "bottom_z_m"))
                + float(_member(obstacle, "top_z_m"))
            ),
        )
        environment_hash = (
            _sha256_file(DEFAULT_ENVIRONMENT_LOCK_PATH)
            if DEFAULT_ENVIRONMENT_LOCK_PATH.is_file()
            else _canonical_hash(scene_snapshot)
        )
        standing_pose = getattr(self._adapter, "standing_pose_deg", None)
        if not isinstance(standing_pose, Mapping) or set(standing_pose) != set(
            SERVO_ORDER
        ):
            raise IsaacFSMBackendError(
                "RobotAdapter standing-pose reset evidence is unavailable"
            )
        limit_evidence = self._adapter.joint_limit_initialization_evidence()
        return {
            "environment_hash": environment_hash,
            "robot_asset_hash": self._dependencies.robot_asset_hash,
            "canonical_reset_state_source": "fresh_scene_post_sim_reset_pre_settle",
            "canonical_reset_state_sha256": (
                None
                if self._canonical_reset_state is None
                else self._canonical_reset_state.state_sha256
            ),
            "canonical_reset_state_instance_count": (
                0
                if self._canonical_reset_state is None
                else self._canonical_reset_state.instance_count
            ),
            "canonical_reset_restore_applied": bool(
                reset_writes.get("canonical_reset_restore_applied", False)
            ),
            "canonical_reset_applied_sha256": reset_writes.get(
                "canonical_reset_applied_sha256"
            ),
            "canonical_settled_state_sha256": (
                None
                if self._canonical_settled_state is None
                else self._canonical_settled_state.state_sha256
            ),
            "canonical_settled_state_source": "fresh_scene_post_settle",
            "canonical_settled_restore_applied": bool(
                reset_writes.get("canonical_settled_restore_applied", False)
            ),
            "canonical_settled_applied_sha256": reset_writes.get(
                "canonical_settled_applied_sha256"
            ),
            "adapter_standing_pose_deg": [
                float(standing_pose[name]) for name in SERVO_ORDER
            ],
            "initial_root_state": list(root_state),
            "initial_joint_state": list(initial_joint_state),
            "obstacle_pose": list(obstacle_pose),
            "controller_hash": _sha256_file(self.fsm_path),
            "motion_contract_hash": _sha256_file(self.motion_contract_path),
            "seed": seed,
            "reset_count": self._reset_count,
            "reset_options": dict(options),
            "physics_hz": PHYSICS_HZ,
            "physics_dt_s": PHYSICS_DT_S,
            "settle_seconds": SETTLE_SECONDS,
            "settle_ticks": SETTLE_TICKS,
            "settle_atomic_full12_writes": SETTLE_TICKS,
            "level_calibration_window_s": LEVEL_CALIBRATION_SECONDS,
            "level_calibration_sample_count": LEVEL_CALIBRATION_TICKS,
            "level_reference_orientation_wxyz": list(
                self._level_reference_orientation or (1.0, 0.0, 0.0, 0.0)
            ),
            "reset_root_pose_writes": int(reset_writes.get("root_pose_writes", 0)),
            "reset_root_velocity_writes": int(
                reset_writes.get("root_velocity_writes", 0)
            ),
            "reset_joint_state_writes": int(
                reset_writes.get("joint_state_writes", 0)
            ),
            "reset_global_simulation_resets": int(
                reset_writes.get("global_simulation_resets", 0)
            ),
            "reset_simulation_forward_syncs": int(
                reset_writes.get("simulation_forward_syncs", 0)
            ),
            "environment_initialization": limit_evidence,
            "locked_scene_snapshot": scene_snapshot,
            "randomization_enabled": False,
            "raw_recording_access": False,
            "training_phase_snapshot": self._snapshot_restoration.get(
                "requested_phase"
            ),
            "phase_snapshot_restoration": dict(self._snapshot_restoration),
        }


def _snapshot_phase_option(options: Mapping[str, Any]) -> str | None:
    value = options.get("training_phase_snapshot")
    if value is None:
        return None
    if not isinstance(value, str) or value not in PHASE_IDS:
        raise IsaacFSMBackendError(
            "training_phase_snapshot must be one exact state id from P01 through P13"
        )
    return value


def _load_validated_phase_snapshot(phase_id: str) -> LoadedPhaseSnapshot:
    """Load only the local derived snapshot set; never follow source-trial paths."""

    from .phase_snapshots import (
        SNAPSHOT_SCHEMA,
        validate_phase_snapshots,
    )

    if phase_id not in PHASE_IDS:
        raise IsaacFSMBackendError(f"unknown phase snapshot {phase_id!r}")
    root = DEFAULT_PHASE_SNAPSHOT_ROOT.resolve()
    try:
        manifest = validate_phase_snapshots(root)
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"phase snapshot manifest validation failed: {type(exc).__name__}: {exc}"
        ) from exc
    rows = tuple(manifest.get("snapshots", ()))
    selected = next((row for row in rows if row.get("phase") == phase_id), None)
    if selected is None:
        raise IsaacFSMBackendError(f"validated snapshot manifest lacks {phase_id}")
    snapshot_path = (root / phase_id / "snapshot.json").resolve()
    if root not in snapshot_path.parents:
        raise IsaacFSMBackendError("phase snapshot resolved outside the locked local root")
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacFSMBackendError(f"cannot read local phase snapshot {phase_id}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != SNAPSHOT_SCHEMA:
        raise IsaacFSMBackendError(f"phase snapshot {phase_id} has an invalid schema")
    _validate_phase_snapshot_payload(payload, phase_id)
    file_hash = _sha256_file(snapshot_path)
    state_hash = str(payload["state_sha256"])
    if (
        state_hash != str(selected.get("state_sha256"))
        or file_hash != str(selected.get("file_sha256"))
    ):
        raise IsaacFSMBackendError(f"phase snapshot {phase_id} hash proof disagrees")
    return LoadedPhaseSnapshot(
        phase_id=phase_id,
        payload=dict(payload),
        state_sha256=state_hash,
        file_sha256=file_hash,
        snapshot_path=snapshot_path,
    )


def _validate_phase_snapshot_payload(
    payload: Mapping[str, Any], phase_id: str
) -> None:
    failures: list[str] = []
    expected_history = list(PHASE_IDS[: PHASE_IDS.index(phase_id)])
    if payload.get("schema") != "wlr50_clean.ppo_phase_entry_snapshot.v1":
        failures.append("snapshot schema is invalid")
    if payload.get("reset_use") != "TRAINING_RESET_STATE_WRITE":
        failures.append("reset_use is not TRAINING_RESET_STATE_WRITE")
    if payload.get("in_episode_root_write") != "FORBIDDEN_IN_EPISODE_ROOT_WRITE":
        failures.append("in-episode root-write prohibition is absent")
    if payload.get("fsm_state") != phase_id:
        failures.append("fsm_state differs from the requested phase")
    if payload.get("fsm_lifecycle") != "EXECUTE_MOTION":
        failures.append("phase entry lifecycle is not EXECUTE_MOTION")
    if payload.get("phase_history") != expected_history:
        failures.append("phase history is not the exact fixed-graph prefix")
    fsm_history = payload.get("fsm_history", {})
    if not isinstance(fsm_history, Mapping) or fsm_history.get(
        "completed_phases"
    ) != expected_history:
        failures.append("FSM completed-phase history is inconsistent")
    if int(fsm_history.get("recovery_count", -1)) != 0:
        failures.append("snapshot contains recovery state")

    root = payload.get("root_state", {})
    for name, size in (
        ("position_w_m", 3),
        ("orientation_wxyz", 4),
        ("linear_velocity_w_m_s", 3),
        ("angular_velocity_w_rad_s", 3),
    ):
        if not isinstance(root, Mapping) or _finite_vector(root.get(name), size) is None:
            failures.append(f"root_state.{name} is not finite Full{size}")
    joint = payload.get("joint_state", {})
    wheel = payload.get("wheel_state", {})
    if not isinstance(joint, Mapping) or tuple(joint.get("order", ())) != SERVO_ORDER:
        failures.append("servo snapshot order differs from the canonical order")
    if not isinstance(wheel, Mapping) or tuple(wheel.get("order", ())) != WHEEL_ORDER:
        failures.append("wheel snapshot order differs from the canonical order")
    for values, size, label in (
        (joint.get("logical_position_deg") if isinstance(joint, Mapping) else None, 8, "servo position"),
        (joint.get("logical_velocity_deg_s") if isinstance(joint, Mapping) else None, 8, "servo velocity"),
        (wheel.get("logical_velocity_rad_s") if isinstance(wheel, Mapping) else None, 4, "wheel velocity"),
        (payload.get("nominal_full12"), 12, "nominal Full12"),
        (payload.get("applied_full12"), 12, "applied Full12"),
        (payload.get("level_reference_orientation_wxyz"), 4, "level reference"),
    ):
        if _finite_vector(values, size) is None:
            failures.append(f"{label} is incomplete or non-finite")
    if payload.get("nominal_full12") != payload.get("applied_full12"):
        failures.append("selected successful snapshot is not zero-residual")

    latches = payload.get("contact_event_latches", {})
    if not isinstance(latches, Mapping) or set(latches) != set(_LEG_TO_WHEEL):
        failures.append("contact-event latches do not cover exactly four legs")
    else:
        for leg, row in latches.items():
            if not isinstance(row, Mapping):
                failures.append(f"{leg} latch row is invalid")
                continue
            for flag, tick_name in (
                ("active_lift", "active_lift_tick"),
                ("front_face_crossed", "front_face_crossed_tick"),
                ("top_loaded", "top_loaded_tick"),
            ):
                active = row.get(flag)
                tick = row.get(tick_name)
                if not isinstance(active, bool) or (
                    active and (not isinstance(tick, int) or tick > int(payload["source_tick"]))
                ) or (not active and tick is not None):
                    failures.append(f"{leg}.{flag} latch/tick proof is inconsistent")
    contact_state = payload.get("contact_state", {})
    if not isinstance(contact_state, Mapping) or set(contact_state) != set(WHEEL_ORDER):
        failures.append("wheel contact snapshot is incomplete")
    geometry = payload.get("obstacle_relative_geometry", {})
    if not isinstance(geometry, Mapping):
        failures.append("obstacle-relative geometry is missing")
    else:
        for field in ("wheel_centers_w_m", "wheel_bottoms_w_m"):
            rows = geometry.get(field, {})
            if not isinstance(rows, Mapping) or set(rows) != set(WHEEL_ORDER):
                failures.append(f"{field} is incomplete")
            elif any(_finite_vector(rows[name], 3) is None for name in WHEEL_ORDER):
                failures.append(f"{field} contains invalid coordinates")
    if failures:
        raise IsaacFSMBackendError(
            f"phase snapshot {phase_id} restoration proof is incomplete: "
            + "; ".join(dict.fromkeys(failures))
        )


def _write_phase_snapshot_state(
    scene: Any, adapter: Any, snapshot: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Write one validated phase state at reset, then synchronize without a step."""

    from wlr50_clean.infrastructure.command_batch import (
        SERVO_COMMAND_SIGN,
        WHEEL_FORWARD_SIGN,
    )

    robot = scene.robot
    root = snapshot["root_state"]
    joint = snapshot["joint_state"]
    wheel = snapshot["wheel_state"]
    position = tuple(float(value) for value in joint["logical_position_deg"])
    servo_velocity = tuple(float(value) for value in joint["logical_velocity_deg_s"])
    wheel_velocity = tuple(float(value) for value in wheel["logical_velocity_rad_s"])
    nominal = _full12(snapshot["nominal_full12"], "snapshot nominal_full12")
    try:
        root_state = robot.data.default_root_state.clone()
        joint_position = robot.data.default_joint_pos.clone()
        joint_velocity = robot.data.default_joint_vel.clone()
        root_pose = (
            tuple(float(value) for value in root["position_w_m"])
            + tuple(float(value) for value in root["orientation_wxyz"])
        )
        root_speed = (
            tuple(float(value) for value in root["linear_velocity_w_m_s"])
            + tuple(float(value) for value in root["angular_velocity_w_rad_s"])
        )
        for index, value in enumerate(root_pose):
            root_state[:, index] = value
        for index, value in enumerate(root_speed, start=7):
            root_state[:, index] = value
        for local, name in enumerate(SERVO_ORDER):
            joint_id = adapter.joint_map.servo_ids[local]
            joint_position[:, joint_id] = math.radians(
                adapter.standing_pose_deg[name]
                + SERVO_COMMAND_SIGN[name] * position[local]
            )
            joint_velocity[:, joint_id] = math.radians(
                SERVO_COMMAND_SIGN[name] * servo_velocity[local]
            )
        for local, name in enumerate(WHEEL_ORDER):
            joint_id = adapter.joint_map.wheel_ids[local]
            joint_velocity[:, joint_id] = WHEEL_FORWARD_SIGN[name] * wheel_velocity[local]

        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(joint_position, joint_velocity)
        robot.reset()
        contact_backend = getattr(
            getattr(scene, "instrumentation", None), "contact_backend", None
        )
        reset_contacts = getattr(contact_backend, "reset", None)
        if not callable(reset_contacts):
            raise IsaacFSMBackendError("exact-pair sensor bank cannot reset")
        reset_contacts()
        forward = getattr(scene.sim, "forward", None)
        if not callable(forward):
            raise IsaacFSMBackendError(
                "SimulationContext.forward is required to prove reset state without stepping"
            )
        forward()
        robot.update(0.0)
    except IsaacFSMBackendError:
        raise
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"phase snapshot reset-only state write failed: {type(exc).__name__}: {exc}"
        ) from exc

    mapper = getattr(adapter, "servo_target_mapper", None)
    required_mapper_fields = (
        "_requested",
        "_applied",
        "_nominal_reached",
        "_compensation",
        "_tracking_active",
        "_retiring_stale_bias",
        "_feedback_tick",
    )
    if mapper is None or any(not hasattr(mapper, name) for name in required_mapper_fields):
        raise IsaacFSMBackendError(
            "cannot prove phase-entry restoration of the frozen drive mapper"
        )
    for index, name in enumerate(SERVO_ORDER):
        mapper._requested[name] = nominal[index]
        mapper._applied[name] = nominal[index]
        mapper._nominal_reached[name] = True
        mapper._compensation[name] = 0.0
        mapper._tracking_active[name] = False
        mapper._retiring_stale_bias[name] = False
        adapter._final_drive_servo_deg[name] = nominal[index]
    mapper._feedback_tick = SETTLE_TICKS + int(snapshot["source_tick"])

    actual = adapter.get_actual_state()
    actual_full12 = tuple(float(value) for value in actual.full12)
    expected_full12 = position + wheel_velocity
    position_error = _maximum_absolute_error(actual_full12, expected_full12)
    physical_servo_velocity = tuple(float(value) for value in actual.servo_velocity_rad_s)
    expected_physical_velocity = tuple(
        math.radians(SERVO_COMMAND_SIGN[name] * value)
        for name, value in zip(SERVO_ORDER, servo_velocity, strict=True)
    )
    velocity_error = _maximum_absolute_error(
        physical_servo_velocity, expected_physical_velocity
    )
    if position_error > 2.0e-4 or velocity_error > 2.0e-5:
        raise IsaacFSMBackendError(
            "phase snapshot joint readback differs from the reset-only write"
        )
    return {
        "schema": "wlr50_clean.phase_snapshot_physical_restore.v1",
        "root_pose_writes": 1,
        "root_velocity_writes": 1,
        "joint_state_writes": 1,
        "simulation_forward_syncs": 1,
        "physics_steps": 0,
        "adapter_mapper_restored": True,
        "adapter_feedback_tick": mapper._feedback_tick,
        "maximum_logical_joint_or_wheel_error": position_error,
        "maximum_physical_servo_velocity_error_rad_s": velocity_error,
    }


def _restore_controller_from_snapshot(
    controller: Any, snapshot: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Rebuild a fresh frozen controller at a proven phase-entry boundary."""

    from wlr50_clean.fsm.motion_executor import FeedbackCorrection
    from wlr50_clean.fsm.state_spec import Lifecycle

    phase_id = str(snapshot["fsm_state"])
    if (
        int(getattr(controller, "physics_tick", -1)) != 0
        or getattr(controller, "termination", None) is not None
        or tuple(getattr(controller, "history", ()))
    ):
        raise IsaacFSMBackendError(
            "phase curriculum requires a fresh independent frozen controller"
        )
    state = controller.graph.state(phase_id)
    controller.state = state
    if str(controller.phase.state_id) != phase_id:
        raise IsaacFSMBackendError("controller graph and motion phase disagree")
    controller.lifecycle = Lifecycle.EXECUTE_MOTION
    controller.retries_used = 0
    controller.physics_tick = 0
    controller._last_sim_time_s = None
    controller._verify_started_s = None
    controller._wait_entry_started_s = None
    controller._endpoint_issued = False
    controller._previous_state_done = True
    controller._pending_blocker = None
    controller._first_blocker = None
    controller._tracking_servo_names = ()
    controller._drive_feedback_tick_index = None
    controller._decision_lattice_origin_tick = 0
    controller.termination = None
    controller.history = []
    controller.watchdog.reset()
    controller.guard_evaluator.reset_state(phase_id)

    phase = controller.phase
    controller.motion._last_full12 = tuple(phase.start_full12)
    correction = FeedbackCorrection(
        state.normal_correction_fractions
        if state.normal_correction_domain == "logical_command"
        else ZERO_FULL12
    )
    controller.motion.start_phase(
        phase,
        correction,
        time_scale=state.normal_time_scale,
    )
    completed = tuple(str(item) for item in snapshot["phase_history"])
    controller._ppo_restored_phase_history = completed
    return {
        "schema": "wlr50_clean.phase_snapshot_controller_restore.v1",
        "state_id": controller.state.state_id,
        "lifecycle": controller.lifecycle.value,
        "physics_tick": controller.physics_tick,
        "motion_tick_index": controller.motion._tick_index,
        "completed_phases": completed,
        "previous_state_done": controller._previous_state_done,
        "retries_used": controller.retries_used,
        "termination": None,
        "history_is_independent": controller.history == [],
    }


def _restore_guard_tracker_from_snapshot(
    reader: Any, snapshot: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Restore measured cumulative latches into one fresh reader instance."""

    from collections import deque

    tracker = getattr(reader, "guard_tracker", None)
    classifier = getattr(reader, "contact_classifier", None)
    required_tracker = (
        "_previous_joint_deg",
        "_recent_joint_delta",
        "_recent_wheel_bottom_z",
        "_recent_air",
        "_joint_total_motion_deg",
        "_wheel_min_bottom_z",
        "_wheel_max_bottom_z",
        "_air_seen",
        "_active_lift",
        "_front_crossed",
        "_top_loaded",
        "_active_lift_tick",
        "_front_crossed_tick",
        "_top_loaded_tick",
    )
    if tracker is None or any(not hasattr(tracker, name) for name in required_tracker):
        raise IsaacFSMBackendError("fresh live guard tracker cannot restore snapshot latches")
    if classifier is None or not hasattr(classifier, "_states") or not hasattr(
        classifier, "_history"
    ):
        raise IsaacFSMBackendError("fresh exact-pair classifier cannot restore contact state")

    source_tick = int(snapshot["source_tick"])
    latch_rows = snapshot["contact_event_latches"]
    tracker._previous_joint_deg = dict(
        zip(
            SERVO_ORDER,
            (float(value) for value in snapshot["joint_state"]["logical_position_deg"]),
            strict=True,
        )
    )
    for values in tracker._recent_joint_delta.values():
        values.clear()
    tracker._joint_total_motion_deg = {name: 0.0 for name in SERVO_ORDER}
    tracker._wheel_min_bottom_z.clear()
    tracker._wheel_max_bottom_z.clear()
    for values in tracker._recent_wheel_bottom_z.values():
        values.clear()
    for values in tracker._recent_air.values():
        values.clear()
    tracker._active_lift = {
        leg: bool(latch_rows[leg]["active_lift"]) for leg in _LEG_TO_WHEEL
    }
    tracker._front_crossed = {
        leg: bool(latch_rows[leg]["front_face_crossed"]) for leg in _LEG_TO_WHEEL
    }
    tracker._top_loaded = {
        leg: bool(latch_rows[leg]["top_loaded"]) for leg in _LEG_TO_WHEEL
    }

    def relative_ticks(field: str) -> dict[str, int]:
        return {
            leg: int(latch_rows[leg][field]) - source_tick
            for leg in _LEG_TO_WHEEL
            if latch_rows[leg][field] is not None
        }

    tracker._active_lift_tick = relative_ticks("active_lift_tick")
    tracker._front_crossed_tick = relative_ticks("front_face_crossed_tick")
    tracker._top_loaded_tick = relative_ticks("top_loaded_tick")

    contact_state = snapshot["contact_state"]
    geometry = snapshot["obstacle_relative_geometry"]
    for leg, wheel_name in _LEG_TO_WHEEL.items():
        contact_class = str(contact_state[wheel_name]["class"])
        air = contact_class == "AIR"
        tracker._air_seen[leg] = bool(air or tracker._active_lift[leg])
        tracker._recent_air[leg].append(air)
        bottom_z = float(geometry["wheel_bottoms_w_m"][wheel_name][2])
        tracker._recent_wheel_bottom_z[leg].append(bottom_z)
        tracker._wheel_min_bottom_z[leg] = bottom_z
        tracker._wheel_max_bottom_z[leg] = bottom_z

    history_length = int(getattr(classifier, "history_length", 3))
    body_by_wheel = {
        "front_left_ankle": "front_left_wheel",
        "front_right_ankle": "front_right_wheel",
        "rear_left_ankle": "rear_left_wheel",
        "rear_right_ankle": "rear_right_wheel",
    }
    classifier._states.clear()
    classifier._history.clear()
    for wheel_name, body_name in body_by_wheel.items():
        row = contact_state[wheel_name]
        for pair_name, field in (
            ("ground", "ground_active"),
            ("obstacle", "obstacle_active"),
        ):
            active = bool(row[field])
            key = (body_name, pair_name)
            classifier._states[key] = active
            classifier._history[key] = deque(
                (active,) * history_length, maxlen=history_length
            )
    return {
        "schema": "wlr50_clean.phase_snapshot_guard_restore.v1",
        "active_lift": dict(tracker._active_lift),
        "front_face_crossed": dict(tracker._front_crossed),
        "top_loaded": dict(tracker._top_loaded),
        "air_seen": dict(tracker._air_seen),
        "source_ticks_shifted_to_episode_relative": True,
        "classifier_wheel_pairs_restored": 8,
        "history_is_independent": True,
    }


def _verify_phase_snapshot_observation(
    observation: Any, snapshot: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Fail closed unless the post-write live sample reproduces the artifact."""

    failures: list[str] = []
    root = snapshot["root_state"]
    base = _member(observation, "base")
    errors = {
        "root_position_m": _maximum_absolute_error(
            _member(base, "position_w_m", ()), root["position_w_m"]
        ),
        "root_orientation": _quaternion_distance(
            _member(base, "orientation_wxyz", ()), root["orientation_wxyz"]
        ),
        "root_linear_velocity_m_s": _maximum_absolute_error(
            _member(base, "linear_velocity_w_m_s", ()),
            root["linear_velocity_w_m_s"],
        ),
        "root_angular_velocity_rad_s": _maximum_absolute_error(
            _member(base, "angular_velocity_w_rad_s", ()),
            root["angular_velocity_w_rad_s"],
        ),
    }
    if errors["root_position_m"] > 2.0e-4 or errors["root_orientation"] > 2.0e-5:
        failures.append("root pose")
    if errors["root_linear_velocity_m_s"] > 2.0e-4:
        failures.append("root linear velocity")
    if errors["root_angular_velocity_rad_s"] > 2.0e-4:
        failures.append("root angular velocity")

    joints = _member(observation, "joints", {})
    joint_snapshot = snapshot["joint_state"]
    position_error = _maximum_absolute_error(
        tuple(float(_member(joints[name], "position_deg")) for name in SERVO_ORDER),
        joint_snapshot["logical_position_deg"],
    )
    velocity_error = _maximum_absolute_error(
        tuple(float(_member(joints[name], "velocity_deg_s")) for name in SERVO_ORDER),
        joint_snapshot["logical_velocity_deg_s"],
    )
    errors["servo_position_deg"] = position_error
    errors["servo_velocity_deg_s"] = velocity_error
    if position_error > 0.02:
        failures.append("servo position")
    if velocity_error > 0.02:
        failures.append("servo velocity")

    wheels = _member(observation, "wheels", {})
    wheel_error = _maximum_absolute_error(
        tuple(float(_member(wheels[name], "velocity_rad_s")) for name in WHEEL_ORDER),
        snapshot["wheel_state"]["logical_velocity_rad_s"],
    )
    errors["wheel_velocity_rad_s"] = wheel_error
    if wheel_error > 2.0e-4:
        failures.append("wheel velocity")

    geometry = snapshot["obstacle_relative_geometry"]
    center_error = max(
        _maximum_absolute_error(
            _member(wheels[name], "center_w_m", ()),
            geometry["wheel_centers_w_m"][name],
        )
        for name in WHEEL_ORDER
    )
    bottom_error = max(
        _maximum_absolute_error(
            _member(wheels[name], "bottom_w_m", ()),
            geometry["wheel_bottoms_w_m"][name],
        )
        for name in WHEEL_ORDER
    )
    errors["wheel_center_m"] = center_error
    errors["wheel_bottom_m"] = bottom_error
    if center_error > 0.002 or bottom_error > 0.002:
        failures.append("wheel geometry")

    contacts = _member(observation, "contacts", {})
    for wheel_name in WHEEL_ORDER:
        wheel = wheels[wheel_name]
        contact = contacts[_member(wheel, "body_name")]
        expected = snapshot["contact_state"][wheel_name]
        actual_class = _enum_value(_member(contact, "contact_class"))
        if (
            actual_class != str(expected["class"])
            or bool(_member(_member(contact, "ground"), "active", False))
            != bool(expected["ground_active"])
            or bool(_member(_member(contact, "obstacle"), "active", False))
            != bool(expected["obstacle_active"])
        ):
            failures.append(f"{wheel_name} exact contact state")
    if failures:
        raise SensorContractFailure(
            "phase snapshot live restoration could not be proven: "
            + ", ".join(dict.fromkeys(failures))
        )
    return {
        "schema": "wlr50_clean.phase_snapshot_live_proof.v1",
        "verified": True,
        "tolerances": {
            "root_position_m": 2.0e-4,
            "root_orientation_quaternion_distance": 2.0e-5,
            "root_velocity": 2.0e-4,
            "servo_position_deg": 0.02,
            "servo_velocity_deg_s": 0.02,
            "wheel_velocity_rad_s": 2.0e-4,
            "wheel_geometry_m": 0.002,
            "wheel_contact_state": "exact",
        },
        "maximum_errors": errors,
    }


def _merge_reset_writes(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        name: value
        for source in (first, second)
        for name, value in source.items()
        if name
        not in {
            "root_pose_writes",
            "root_velocity_writes",
            "joint_state_writes",
            "global_simulation_resets",
            "simulation_forward_syncs",
        }
    }
    for name in (
        "root_pose_writes",
        "root_velocity_writes",
        "joint_state_writes",
        "global_simulation_resets",
        "simulation_forward_syncs",
    ):
        left = int(first.get(name, 0))
        right = int(second.get(name, 0))
        if left < 0 or right < 0:
            raise IsaacFSMBackendError("reset-only state-write counters cannot be negative")
        result[name] = left + right
    return result


def _maximum_absolute_error(left: Any, right: Any) -> float:
    try:
        first = tuple(float(value) for value in left)
        second = tuple(float(value) for value in right)
    except (TypeError, ValueError):
        return math.inf
    if len(first) != len(second) or not first:
        return math.inf
    if any(not math.isfinite(value) for value in first + second):
        return math.inf
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


def _quaternion_distance(left: Any, right: Any) -> float:
    try:
        first = _normalized_quaternion(left)
        second = _normalized_quaternion(right)
    except IsaacFSMBackendError:
        return math.inf
    direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))
    antipodal = math.sqrt(sum((a + b) ** 2 for a, b in zip(first, second, strict=True)))
    return min(direct, antipodal)


def _flat_finite_tensor_values(value: Any, label: str) -> tuple[float, ...]:
    """Copy a live tensor to a finite, device-independent hash payload."""

    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    reshape = getattr(current, "reshape", None)
    if callable(reshape):
        current = reshape(-1)
    tolist = getattr(current, "tolist", None)
    try:
        raw = tolist() if callable(tolist) else list(current)
        result = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise IsaacFSMBackendError(
            f"{label} cannot be serialized as a numeric tensor"
        ) from exc
    if not result or any(not math.isfinite(item) for item in result):
        raise IsaacFSMBackendError(f"{label} must contain finite values")
    return result


def _canonical_articulation_state_identity(
    root_state: Any, joint_position: Any, joint_velocity: Any
) -> tuple[int, str]:
    try:
        root_shape = tuple(int(value) for value in root_state.shape)
        position_shape = tuple(int(value) for value in joint_position.shape)
        velocity_shape = tuple(int(value) for value in joint_velocity.shape)
    except Exception as exc:
        raise IsaacFSMBackendError(
            "canonical articulation reset tensor shapes are unavailable"
        ) from exc
    if (
        len(root_shape) != 2
        or root_shape[1] != 13
        or len(position_shape) != 2
        or position_shape != velocity_shape
        or position_shape[0] != root_shape[0]
        or position_shape[1] <= 0
    ):
        raise IsaacFSMBackendError(
            "canonical articulation reset tensors have inconsistent shapes"
        )
    payload = {
        "root_state": _flat_finite_tensor_values(root_state, "canonical root state"),
        "joint_position": _flat_finite_tensor_values(
            joint_position, "canonical joint position"
        ),
        "joint_velocity": _flat_finite_tensor_values(
            joint_velocity, "canonical joint velocity"
        ),
        "root_shape": root_shape,
        "joint_shape": position_shape,
    }
    return root_shape[0], _canonical_hash(payload)


def capture_canonical_articulation_reset_state(
    robot: Any,
) -> CanonicalArticulationResetState:
    """Clone the USD-authored live state before any settle or command tick."""

    try:
        root_state = robot.data.root_state_w.clone()
        joint_position = robot.data.joint_pos.clone()
        joint_velocity = robot.data.joint_vel.clone()
    except Exception as exc:
        raise IsaacFSMBackendError(
            "cannot clone the live USD-authored articulation reset state"
        ) from exc
    instance_count, state_sha256 = _canonical_articulation_state_identity(
        root_state, joint_position, joint_velocity
    )
    return CanonicalArticulationResetState(
        root_state=root_state,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        instance_count=instance_count,
        state_sha256=state_sha256,
    )


def restore_canonical_articulation_reset_state(
    robot: Any,
    canonical_state: CanonicalArticulationResetState,
    *,
    expected_instance_count: int,
) -> None:
    """Write one previously captured native state at a reset boundary."""

    if not isinstance(canonical_state, CanonicalArticulationResetState):
        raise IsaacFSMBackendError("canonical articulation reset state is invalid")
    if canonical_state.instance_count != int(expected_instance_count):
        raise IsaacFSMBackendError(
            "canonical articulation reset state has the wrong instance count"
        )
    live_count, live_sha256 = _canonical_articulation_state_identity(
        canonical_state.root_state,
        canonical_state.joint_position,
        canonical_state.joint_velocity,
    )
    if (
        live_count != canonical_state.instance_count
        or live_sha256 != canonical_state.state_sha256
    ):
        raise IsaacFSMBackendError(
            "canonical articulation reset tensors changed after capture"
        )
    try:
        root_state = canonical_state.root_state.clone()
        joint_position = canonical_state.joint_position.clone()
        joint_velocity = canonical_state.joint_velocity.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(joint_position, joint_velocity)
        robot.reset()
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"canonical articulation restore failed: {type(exc).__name__}: {exc}"
        ) from exc


def _restore_canonical_settled_articulation_state(
    scene: Any, canonical_state: CanonicalArticulationResetState
) -> Mapping[str, Any]:
    """Restore the fresh post-settle visible state without a physics step.

    The ordinary pre-settle reset and all 180 settle ticks have already run,
    so contact/solver history was rebuilt naturally.  This final reset-boundary
    write removes the remaining visible GPU-PhysX drift while retaining that
    rebuilt history.  It deliberately does not reset contact sensors again.
    """

    robot = scene.robot
    try:
        restore_canonical_articulation_reset_state(
            robot, canonical_state, expected_instance_count=1
        )
        forward = getattr(scene.sim, "forward", None)
        if not callable(forward):
            raise IsaacFSMBackendError(
                "SimulationContext.forward is required for post-settle restore"
            )
        forward()
        robot.update(0.0)
    except IsaacFSMBackendError:
        raise
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"post-settle articulation restore failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "root_pose_writes": 1,
        "root_velocity_writes": 1,
        "joint_state_writes": 1,
        "global_simulation_resets": 0,
        "simulation_forward_syncs": 1,
        "canonical_settled_restore_applied": True,
        "canonical_settled_applied_sha256": canonical_state.state_sha256,
    }


def _restore_default_articulation_state(
    scene: Any, canonical_state: CanonicalArticulationResetState
) -> Mapping[str, Any]:
    """Soft-reset to the captured USD-authored state at an episode boundary.

    A process-level/fresh-scene rollout and a reused scene must have the same
    physical history.  Calling ``SimulationContext.reset()`` here proved not
    to be equivalent on the locked Windows/PhysX runtime: after the first
    episode it deterministically changed the P09 rebound enough to miss P10's
    signed-velocity entry gate.  Isaac Lab's indexed state-write reset is the
    appropriate episode reset path.  The one global reset needed to create the
    stage remains owned by ``SceneFactory.create_scene``.

    No physics step is taken here.  State writes, sensor-history reset and a
    ``forward()`` synchronization are confined to the reset boundary.
    """

    robot = scene.robot
    try:
        restore_canonical_articulation_reset_state(
            robot, canonical_state, expected_instance_count=1
        )
        instrumentation = getattr(scene, "instrumentation", None)
        contact_backend = getattr(instrumentation, "contact_backend", None)
        reset_contacts = getattr(contact_backend, "reset", None)
        if not callable(reset_contacts):
            raise IsaacFSMBackendError("exact-pair sensor bank cannot soft-reset")
        reset_contacts()
        forward = getattr(scene.sim, "forward", None)
        if not callable(forward):
            raise IsaacFSMBackendError(
                "SimulationContext.forward is required for a step-free soft reset"
            )
        forward()
        robot.update(0.0)
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"reset-only articulation restore failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "root_pose_writes": 1,
        "root_velocity_writes": 1,
        "joint_state_writes": 1,
        "global_simulation_resets": 0,
        "simulation_forward_syncs": 1,
        "canonical_reset_state_sha256": canonical_state.state_sha256,
        "canonical_reset_state_source": "fresh_scene_post_sim_reset_pre_settle",
        "canonical_reset_restore_applied": True,
        "canonical_reset_applied_sha256": canonical_state.state_sha256,
    }


def _validate_reset_options(options: Mapping[str, Any]) -> None:
    if bool(options.get("randomization_enabled", False)) or bool(
        options.get("enable_randomization", False)
    ):
        raise IsaacFSMBackendError(
            "single-environment baseline backend does not permit randomization"
        )
    forbidden_fragments = (
        "recording_path",
        "replay",
        "root_state",
        "root_velocity",
        "external_force",
        "impulse",
        "gravity",
    )
    for key in _nested_keys(options):
        lowered = key.lower()
        if any(fragment in lowered for fragment in forbidden_fragments):
            raise IsaacFSMBackendError(
                f"reset option {key!r} requests a prohibited runtime mutation/source"
            )


def _nested_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    result: list[str] = []
    for key, child in value.items():
        result.append(str(key))
        result.extend(_nested_keys(child))
    return tuple(result)


def _non_negative_seed(value: int) -> int:
    if isinstance(value, bool):
        raise IsaacFSMBackendError("seed must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise IsaacFSMBackendError("seed must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise IsaacFSMBackendError("seed must be a non-negative integer")
    return result


def _full12(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise IsaacFSMBackendError(f"{label} must be numeric") from exc
    if len(result) != FULL12_SIZE or any(not math.isfinite(value) for value in result):
        raise IsaacFSMBackendError(f"{label} must contain twelve finite values")
    return result


def _require_running(scene: Any, context: str) -> None:
    running = getattr(scene, "app_is_running", None)
    if not callable(running) or not bool(running()):
        raise IsaacFSMBackendError(f"SimulationApp stopped during {context}")


def _validate_rate_contract(adapter: Any, controller: Any) -> None:
    adapter_rate = float(adapter.servo_target_mapper.servo_rate_deg_s)
    controller_rate = float(controller.motion.servo_rate_limit_deg_s)
    if not math.isclose(adapter_rate, controller_rate, rel_tol=0.0, abs_tol=1.0e-12):
        raise IsaacFSMBackendError(
            "motion contract and mature servo target mapper rates differ"
        )
    dt = float(getattr(adapter, "physics_dt_s", PHYSICS_DT_S))
    if not math.isclose(dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise IsaacFSMBackendError("RobotAdapter is not running at exactly 120 Hz")


def _validate_controller_clock(
    frame: Any, *, physics_tick: int, sim_time_s: float
) -> None:
    if int(getattr(frame, "physics_tick", -1)) != physics_tick:
        raise IsaacFSMBackendError(
            "frozen controller did not consume exactly one call per physics tick"
        )
    if not math.isclose(
        float(getattr(frame, "sim_time_s", float("nan"))),
        sim_time_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IsaacFSMBackendError("frozen controller clock differs from live physics")


def _validate_sensor_contract(
    observation: Any,
    expected_contact_bodies: Sequence[str],
    *,
    require_finite: bool,
) -> None:
    failures: list[str] = []
    contacts = _member(observation, "contacts", {})
    expected = set(str(name) for name in expected_contact_bodies)
    actual = set(str(name) for name in contacts)
    if len(expected) != 13 or actual != expected:
        failures.append(
            "contact bodies differ from the exact locked 13-body sensor bank"
        )
    for body_name in sorted(actual):
        contact = contacts[body_name]
        for pair_name in ("ground", "obstacle"):
            pair = _member(contact, pair_name)
            if pair is None or not bool(_member(pair, "pair_verified", False)):
                failures.append(f"{body_name} exact {pair_name} pair is unverified")
        if _enum_value(_member(contact, "contact_class")) == "UNVERIFIED":
            failures.append(f"{body_name} contact classification is UNVERIFIED")
    wheels = _member(observation, "wheels", {})
    if len(wheels) != 4:
        failures.append(f"wheel geometry count={len(wheels)}, expected 4")
    for wheel in wheels.values():
        if (
            not bool(_member(wheel, "geometry_verified", False))
            or _member(wheel, "center_w_m") is None
            or _member(wheel, "bottom_w_m") is None
        ):
            failures.append(
                f"{_member(wheel, 'name', 'unknown wheel')} live collider geometry is unverified"
            )
    bodies = _member(observation, "bodies", {})
    if len(bodies) != 13:
        failures.append(f"rigid-body state count={len(bodies)}, expected 13")
    center_of_mass = _member(observation, "center_of_mass")
    if not bool(_member(center_of_mass, "valid", False)) or len(
        _member(center_of_mass, "included_bodies", ())
    ) != 13:
        failures.append("full 13-body center-of-mass observation is invalid")
    if require_finite and not bool(_member(observation, "all_finite", False)):
        failures.append("live observation contains non-finite values")
    failures.extend(str(item) for item in _member(observation, "data_quality", ()))
    if failures:
        unique = "; ".join(dict.fromkeys(failures))
        raise SensorContractFailure(f"critical live sensing quality failed: {unique}")


def _phase_progress(controller: Any, state_id: str) -> float:
    motion = getattr(controller, "motion", None)
    motion_phase = getattr(motion, "phase", None)
    if motion is None or motion_phase is None or str(motion_phase.state_id) != state_id:
        return 0.0
    duration = float(getattr(motion, "effective_active_duration_s", 0.0))
    if duration <= 0.0:
        return 0.0
    tick_index = int(getattr(motion, "_tick_index", 0)) - 1
    return max(0.0, min(1.0, tick_index * PHYSICS_DT_S / duration))


def _reference_action(controller: Any, frame: Any, phase: Any) -> tuple[float, ...]:
    motion = getattr(controller, "motion", None)
    motion_phase = getattr(motion, "phase", None)
    if motion_phase is None or str(getattr(motion_phase, "state_id", "")) != str(
        getattr(frame, "state_id", "")
    ):
        return _full12(getattr(phase, "start_full12"), "phase reference start")
    tick_index = max(0, int(getattr(motion, "_tick_index", 0)) - 1)
    waypoints = tuple(getattr(motion_phase, "waypoints", ()))
    resolver = getattr(motion, "_scaled_waypoint_index_at_tick", None)
    if waypoints and callable(resolver):
        index = int(resolver(motion_phase, tick_index))
        return _full12(waypoints[index].full12, "phase reference waypoint")
    nominal_at = getattr(motion_phase, "nominal_at", None)
    if callable(nominal_at):
        return _full12(
            nominal_at(tick_index * PHYSICS_DT_S), "phase reference action"
        )
    return _full12(getattr(frame, "full12"), "phase reference fallback")


def _guard_asserted(observation: Any, name: str) -> bool:
    guards = _member(observation, "guards", {})
    value = guards.get(name, False) if isinstance(guards, Mapping) else False
    if isinstance(value, Mapping):
        return bool(value.get("passed", False))
    return bool(value)


def _fall_and_explosion(observation: Any) -> tuple[bool, bool, dict[str, float]]:
    base = _member(observation, "base")
    imu = _member(observation, "imu")
    position = _member(base, "position_w_m", ())
    base_z = _finite_index(position, 2)
    linear_speed = _norm3(_member(base, "linear_velocity_w_m_s", ()))
    angular_speed = _norm3(_member(base, "angular_velocity_w_rad_s", ()))
    gravity_z = _finite_index(_member(imu, "projected_gravity_b", ()), 2)
    values = {
        "base_z_m": base_z,
        "linear_speed_m_s": linear_speed,
        "angular_speed_rad_s": angular_speed,
        "projected_gravity_b_z": gravity_z,
    }
    finite = all(math.isfinite(value) for value in values.values())
    if not finite:
        return False, False, values
    fall = bool(base_z < 0.015 or gravity_z > -0.30)
    explosion = bool(base_z > 1.0 or linear_speed > 5.0 or angular_speed > 20.0)
    combined = _guard_asserted(observation, "physics_explosion_or_fall")
    if combined and not (fall or explosion):
        explosion = True
    return fall, explosion, values


def _active_leg_clearance(observation: Any, state_id: str) -> float:
    leg = _ACTIVE_LEG_BY_STATE.get(state_id)
    if leg is None:
        return 0.0
    wheels = _member(observation, "wheels", {})
    wheel = wheels.get(_LEG_TO_WHEEL[leg]) if isinstance(wheels, Mapping) else None
    bottom = _member(wheel, "bottom_w_m")
    obstacle = _member(observation, "obstacle")
    bottom_z = _finite_index(bottom or (), 2)
    obstacle_bottom = float(_member(obstacle, "bottom_z_m", 0.0))
    if not math.isfinite(bottom_z) or not math.isfinite(obstacle_bottom):
        return 0.0
    return max(0.0, bottom_z - obstacle_bottom)


def _observation_quaternion(observation: Any) -> tuple[float, float, float, float]:
    return _normalized_quaternion(
        _member(_member(observation, "base"), "orientation_wxyz", ())
    )


def _mean_quaternion(
    samples: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    normalized = tuple(_normalized_quaternion(sample) for sample in samples)
    if not normalized:
        raise IsaacFSMBackendError("level calibration received no orientation samples")
    anchor = normalized[0]
    aligned = tuple(
        tuple(-value for value in sample)
        if sum(a * b for a, b in zip(anchor, sample, strict=True)) < 0.0
        else sample
        for sample in normalized
    )
    mean = tuple(
        sum(sample[index] for sample in aligned) / len(aligned) for index in range(4)
    )
    return _normalized_quaternion(mean)


def _level_measurement(
    observation: Any,
    reference: Sequence[float] | None,
) -> dict[str, Any]:
    current_values = _member(
        _member(observation, "base"), "orientation_wxyz", ()
    )
    current_finite = _finite_vector(current_values, 4)
    if current_finite is None:
        current = (1.0, 0.0, 0.0, 0.0)
        raw_valid = False
    else:
        current = _normalized_quaternion(current_finite)
        raw_valid = True
    level_reference = _normalized_quaternion(
        reference or (1.0, 0.0, 0.0, 0.0)
    )
    relative = _quat_multiply(_quat_conjugate(level_reference), current)
    raw_roll, raw_pitch, raw_yaw = _quat_to_euler(current)
    error_roll, error_pitch, error_yaw = _quat_to_euler(relative)
    angular_values = _member(
        _member(observation, "base"), "angular_velocity_w_rad_s", ()
    )
    angular = _finite_vector(angular_values, 3) or (0.0, 0.0, 0.0)
    roll_axis = _quat_rotate(level_reference, (1.0, 0.0, 0.0))
    pitch_axis = _quat_rotate(level_reference, (0.0, 1.0, 0.0))
    return {
        "schema": "wlr50_clean.level_calibration.v1",
        "sample_count": LEVEL_CALIBRATION_TICKS,
        "window_s": LEVEL_CALIBRATION_SECONDS,
        "valid": raw_valid,
        "level_reference_orientation_wxyz": tuple(level_reference),
        "raw_orientation_wxyz": tuple(current_values),
        "raw_roll_rad": raw_roll,
        "raw_pitch_rad": raw_pitch,
        "raw_yaw_rad": raw_yaw,
        "relative_orientation_wxyz": tuple(relative),
        "roll_error_to_level_rad": error_roll,
        "pitch_error_to_level_rad": error_pitch,
        "yaw_error_to_level_rad": error_yaw,
        "raw_angular_velocity_world_rad_s": tuple(angular_values),
        "calibrated_roll_axis_world": tuple(roll_axis),
        "calibrated_pitch_axis_world": tuple(pitch_axis),
        "roll_change_rate_rad_s": sum(
            a * b for a, b in zip(angular, roll_axis, strict=True)
        ),
        "pitch_change_rate_rad_s": sum(
            a * b for a, b in zip(angular, pitch_axis, strict=True)
        ),
    }


def _normalized_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    finite = _finite_vector(values, 4)
    if finite is None:
        raise IsaacFSMBackendError("body orientation quaternion is not finite Full4")
    norm = math.sqrt(sum(value * value for value in finite))
    if norm <= 1.0e-12:
        raise IsaacFSMBackendError("body orientation quaternion has zero norm")
    return tuple(value / norm for value in finite)  # type: ignore[return-value]


def _quat_conjugate(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    w, x, y, z = quaternion
    return (w, -x, -y, -z)


def _quat_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return _normalized_quaternion(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _quat_rotate(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    # Rotation-matrix form avoids normalizing a pure-vector quaternion.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def _quat_to_euler(quaternion: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _base_x(observation: Any | None) -> float | None:
    if observation is None:
        return None
    value = _finite_index(
        _member(_member(observation, "base"), "position_w_m", ()), 0
    )
    return value if math.isfinite(value) else None


def _frame_is_terminal(frame: AuthoritativeFrame) -> bool:
    signals = frame.termination_signals
    return any(
        (
            signals.success,
            signals.body_collision,
            signals.wheel_only_climb,
            signals.fall,
            signals.nan_inf,
            signals.hard_joint_limit,
            signals.physics_explosion,
            signals.timeout,
        )
    )


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _finite_vector(values: Any, size: int) -> tuple[float, ...] | None:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(result) != size or any(not math.isfinite(value) for value in result):
        return None
    return result


def _finite_index(values: Any, index: int) -> float:
    try:
        return float(values[index])
    except (IndexError, KeyError, TypeError, ValueError):
        return float("nan")


def _norm3(values: Any) -> float:
    finite = _finite_vector(values, 3)
    if finite is None:
        return float("nan")
    return math.sqrt(sum(value * value for value in finite))


def _finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
