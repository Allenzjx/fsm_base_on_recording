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
from .phase_snapshots import (
    SNAPSHOT_SCHEMA,
    SOURCE_ACK_MATCH_FIELDS,
    SOURCE_MAPPER_STATE_SCHEMA,
    PhaseSnapshotError,
    ValidatedPhaseSnapshotBundle,
    phase_snapshot_actuation_contract_sha256,
    phase_snapshot_drive_target_sha256,
    validate_phase_snapshot_payload_contract,
)
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
PHASE_SNAPSHOT_PRIME_PHYSICS_STEPS = 1
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
    faithful reset source for this asset.  These tensors are cloned after the
    baseline-order reset and live limit authoring, before the physical settle,
    and are evidence only: later resets must reproduce them naturally.
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
        [Any, CanonicalArticulationResetState | None], Mapping[str, Any]
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
    capture_session_limit_state: Callable[[Any], Mapping[str, Any]] | None = None


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
        reset_scene=_reset_physics_lifecycle,
        restore_settled_state=_restore_canonical_settled_articulation_state,
        locked_scene_snapshot=locked_scene_snapshot,
        expected_contact_bodies=tuple(SENSED_BODIES),
        robot_asset_hash=str(ROBOT_USD_SHA256),
        load_phase_snapshot=_load_validated_phase_snapshot,
        write_phase_snapshot=_write_phase_snapshot_state,
        restore_controller_snapshot=_restore_controller_from_snapshot,
        restore_guard_snapshot=_restore_guard_tracker_from_snapshot,
        capture_session_limit_state=_session_servo_limit_state,
    )


class IsaacFSMBackend:
    """One live Isaac scene driven by one authoritative frozen FSM instance.

    ``simulation_app`` must already have been created by ``AppLauncher`` for a
    production backend.  The backend never launches or closes the application,
    records video, opens a Recording, changes gravity, or applies forces.  Root
    and the stop/play physics lifecycle reset is confined to ``reset``;
    :meth:`step_physics` has no state-write path other than ``apply_full12``.
    """

    def __init__(
        self,
        simulation_app: Any | None = None,
        *,
        fsm_path: Path | str = DEFAULT_FSM_PATH,
        motion_contract_path: Path | str = DEFAULT_MOTION_CONTRACT_PATH,
        dependencies: BackendDependencies | None = None,
        expected_phase_snapshot_bundle: ValidatedPhaseSnapshotBundle | None = None,
        phase_snapshot_prime_physics_steps: int = PHASE_SNAPSHOT_PRIME_PHYSICS_STEPS,
    ) -> None:
        self.simulation_app = simulation_app
        self.fsm_path = Path(fsm_path).resolve()
        self.motion_contract_path = Path(motion_contract_path).resolve()
        self._dependencies = dependencies
        if (
            expected_phase_snapshot_bundle is not None
            and expected_phase_snapshot_bundle.snapshot_root
            != DEFAULT_PHASE_SNAPSHOT_ROOT.resolve()
        ):
            raise IsaacFSMBackendError(
                "injected phase snapshot bundle does not use the production loader root"
            )
        self._expected_phase_snapshot_bundle = expected_phase_snapshot_bundle
        if (
            isinstance(phase_snapshot_prime_physics_steps, bool)
            or not isinstance(phase_snapshot_prime_physics_steps, int)
            or phase_snapshot_prime_physics_steps
            != PHASE_SNAPSHOT_PRIME_PHYSICS_STEPS
        ):
            raise IsaacFSMBackendError(
                "phase snapshot priming must use exactly one real physics step"
            )
        self._phase_snapshot_prime_physics_steps = (
            phase_snapshot_prime_physics_steps
        )
        self._phase_snapshot_integrity_failed = False
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
        self._reset_prime_tick_count = 0
        self._episode_tick = 0
        self._first_episode_physical_command_tick_actual: int | None = None
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
        self._canonical_pre_limit_native_state_sha256: str | None = None
        self._canonical_pre_limit_native_state_instance_count: int | None = None
        self._canonical_pre_physics_composed_limit_state_sha256: str | None = None
        self._canonical_authored_session_limit_state_sha256: str | None = None
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
        self._reset_prime_tick_count = 0
        self._first_episode_physical_command_tick_actual = None
        dependencies = self._dependencies or _load_live_dependencies()
        self._dependencies = dependencies
        snapshot_phase = _snapshot_phase_option(reset_options)
        loaded_snapshot: LoadedPhaseSnapshot | None = None
        if snapshot_phase is not None:
            if self._phase_snapshot_integrity_failed:
                raise IsaacFSMBackendError(
                    "phase snapshot integrity previously failed in this backend"
                )
            if (
                self._expected_phase_snapshot_bundle is None
                and dependencies.load_phase_snapshot is None
            ):
                raise IsaacFSMBackendError(
                    "phase curriculum requested but no validated snapshot loader exists"
                )
            try:
                loaded_snapshot = (
                    _load_validated_phase_snapshot(
                        snapshot_phase,
                        expected_bundle=self._expected_phase_snapshot_bundle,
                    )
                    if self._expected_phase_snapshot_bundle is not None
                    else dependencies.load_phase_snapshot(snapshot_phase)
                )
            except Exception:
                self._phase_snapshot_integrity_failed = True
                raise
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

        reset_writes = dict(
            dependencies.reset_scene(self._scene, self._canonical_reset_state)
        )
        pre_limit_sha256 = str(
            reset_writes.get("pre_limit_native_state_observed_sha256", "")
        )
        pre_limit_instance_count = int(
            reset_writes.get("pre_limit_native_state_instance_count", 0)
        )
        if not pre_limit_sha256 or pre_limit_instance_count != 1:
            raise IsaacFSMBackendError(
                "baseline-order reset did not expose one native pre-limit state"
            )
        if self._canonical_pre_limit_native_state_sha256 is None:
            self._canonical_pre_limit_native_state_sha256 = pre_limit_sha256
            self._canonical_pre_limit_native_state_instance_count = (
                pre_limit_instance_count
            )
        elif (
            pre_limit_sha256
            != self._canonical_pre_limit_native_state_sha256
            or pre_limit_instance_count
            != self._canonical_pre_limit_native_state_instance_count
        ):
            raise IsaacFSMBackendError(
                "reset did not reproduce the canonical native pre-limit state"
            )
        reset_writes["pre_limit_native_state_matches_canonical"] = True
        composed_limit_sha256 = str(
            reset_writes.get("pre_physics_composed_limit_state_sha256", "")
        )
        if len(composed_limit_sha256) != 64:
            raise IsaacFSMBackendError(
                "pre-limit composed state has no canonical SHA-256"
            )
        if self._canonical_pre_physics_composed_limit_state_sha256 is None:
            self._canonical_pre_physics_composed_limit_state_sha256 = (
                composed_limit_sha256
            )
        elif (
            composed_limit_sha256
            != self._canonical_pre_physics_composed_limit_state_sha256
        ):
            raise IsaacFSMBackendError(
                "source-composed servo limits changed between episode resets"
            )
        reset_writes["pre_physics_composed_limit_state_matches_canonical"] = True

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
        capture_session_limits = dependencies.capture_session_limit_state
        if not callable(capture_session_limits):
            raise IsaacFSMBackendError(
                "session-layer limit evidence callback is unavailable"
            )
        authored_limit_state = dict(capture_session_limits(scene))
        if int(authored_limit_state.get("property_count", -1)) != 16:
            raise IsaacFSMBackendError(
                "RobotAdapter did not author exactly 16 session limit specs"
            )
        authored_limit_sha256 = str(
            authored_limit_state.get("state_sha256", "")
        )
        if len(authored_limit_sha256) != 64:
            raise IsaacFSMBackendError(
                "authored session limit state has no canonical SHA-256"
            )
        removed_limit_sha256 = reset_writes.get(
            "removed_session_limit_state_sha256"
        )
        if (
            removed_limit_sha256 is not None
            and authored_limit_sha256 != removed_limit_sha256
        ):
            raise IsaacFSMBackendError(
                "RobotAdapter did not reproduce the removed session limit state"
            )
        if self._canonical_authored_session_limit_state_sha256 is None:
            self._canonical_authored_session_limit_state_sha256 = (
                authored_limit_sha256
            )
        elif (
            authored_limit_sha256
            != self._canonical_authored_session_limit_state_sha256
        ):
            raise IsaacFSMBackendError(
                "authored session limit state differs between episodes"
            )
        reset_writes.update(
            {
                "post_author_session_limit_state_sha256": (
                    authored_limit_sha256
                ),
                "post_author_session_limit_state_matches_canonical": True,
                "session_limit_specs_after_authoring": 16,
            }
        )
        observed_reset_state = dependencies.capture_reset_state(scene)
        if self._canonical_reset_state is None:
            self._canonical_reset_state = observed_reset_state
        elif (
            observed_reset_state.instance_count
            != self._canonical_reset_state.instance_count
            or observed_reset_state.state_sha256
            != self._canonical_reset_state.state_sha256
        ):
            raise IsaacFSMBackendError(
                "baseline-order reset did not reproduce the canonical pre-settle state"
            )
        reset_writes.update(
            {
                "pre_settle_native_state_observed_sha256": (
                    observed_reset_state.state_sha256
                ),
                "pre_settle_native_state_matches_canonical": True,
            }
        )
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
        observed_settled_state = dependencies.capture_reset_state(scene)
        reset_writes["observed_settled_state_sha256"] = (
            observed_settled_state.state_sha256
        )
        if self._canonical_settled_state is None:
            self._canonical_settled_state = observed_settled_state
            self._canonical_level_reference_orientation = (
                self._level_reference_orientation
            )
        elif normal_p01_reset:
            if (
                observed_settled_state.instance_count
                != self._canonical_settled_state.instance_count
                or observed_settled_state.state_sha256
                != self._canonical_settled_state.state_sha256
            ):
                raise IsaacFSMBackendError(
                    "baseline-order reset did not reproduce the canonical natural-settle state"
                )
            if self._canonical_level_reference_orientation is None:
                raise IsaacFSMBackendError(
                    "canonical post-settle level reference is unavailable"
                )
            if (
                self._level_reference_orientation
                != self._canonical_level_reference_orientation
            ):
                raise IsaacFSMBackendError(
                    "baseline-order reset did not reproduce the level calibration"
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
            initial_command = _full12(
                loaded_snapshot.payload["applied_full12"],
                "phase snapshot applied_full12",
            )
            initial_state_write = dict(
                dependencies.write_phase_snapshot(
                    scene,
                    adapter,
                    loaded_snapshot.payload,
                    reset_contact_backend=True,
                    state_write_index=1,
                )
            )
            source_command = loaded_snapshot.payload["source_command"]
            adapter_input = source_command["adapter_input"]
            _require_running(scene, "phase snapshot reset contact priming")
            physical_tick = SETTLE_TICKS + self._reset_prime_tick_count
            prime_ack = dict(
                self._atomic_apply(
                    adapter,
                    adapter_input["requested_full12"],
                    physics_tick=physical_tick,
                    tracking_servo_names=adapter_input["tracking_servo_names"],
                    drive_feedback_bias_full12=adapter_input[
                        "drive_feedback_bias_requested_full12"
                    ],
                )
            )
            source_actuation_match = dict(
                _verify_source_prime_ack(loaded_snapshot.payload, prime_ack)
            )
            source_mapper_post_state = dict(
                _verify_source_mapper_post_state(adapter, loaded_snapshot.payload)
            )
            scene.sim.step(render=False)
            adapter.update_readback()
            self._reset_prime_tick_count += 1
            prime_drive_target = _full12(
                prime_ack["drive_target_full12"],
                "phase snapshot prime drive_target_full12",
            )
            prime_acks: list[dict[str, Any]] = [
                {
                    "physics_tick": int(prime_ack["physics_tick"]),
                    "write_count": int(prime_ack["write_count"]),
                    "articulation_writes_this_call": int(
                        prime_ack["articulation_writes_this_call"]
                    ),
                    "applied_full12": list(
                        _full12(
                            prime_ack["applied_full12"],
                            "phase snapshot prime applied_full12",
                        )
                    ),
                    "native_drive_target_full12": list(
                        _full12(
                            prime_ack["native_drive_target_full12"],
                            "phase snapshot prime native_drive_target_full12",
                        )
                    ),
                    "drive_target_full12": list(prime_drive_target),
                    "source_actuation_match": source_actuation_match,
                }
            ]
            latest_settle_ack = prime_ack

            mapper = getattr(adapter, "servo_target_mapper", None)
            mapper_requested = tuple(
                float(mapper._requested[name]) for name in SERVO_ORDER
            )
            mapper_applied = tuple(
                float(mapper._applied[name]) for name in SERVO_ORDER
            )

            physical_proof = dict(initial_state_write)
            physical_proof.update(
                {
                    "schema": "wlr50_clean.phase_snapshot_prime_without_rewind.v1",
                    "reset_use": "TRAINING_RESET_STATE_WRITE",
                    "state_write_count": 1,
                    "initial_state_write": initial_state_write,
                    "post_prime_state_rewrite_performed": False,
                    "contact_and_state_share_solver_tick": True,
                    "prime_physics_steps": self._reset_prime_tick_count,
                    "prime_atomic_full12_writes": len(prime_acks),
                    "prime_atomic_writes": prime_acks,
                    "prime_applied_full12": list(initial_command),
                    "prime_native_drive_target_full12": list(
                        prime_ack["native_drive_target_full12"]
                    ),
                    "prime_drive_target_full12": list(prime_drive_target),
                    "source_actuation_match": source_actuation_match,
                    "source_mapper_post_state": source_mapper_post_state,
                    "source_target_sha256": source_command[
                        "drive_target_full12_sha256"
                    ],
                    "logical_target_fallback_used": False,
                    "mapper_state_after_prime": {
                        "requested_servo_deg": list(mapper_requested),
                        "applied_servo_deg": list(mapper_applied),
                        "feedback_tick": int(mapper._feedback_tick),
                        "matches_source_post_command_state": True,
                    },
                    "physics_steps": self._reset_prime_tick_count,
                    "articulation_update_dt_s_per_prime_step": PHYSICS_DT_S,
                    "effective_entry_state": "snapshot_plus_one_physics_tick",
                    "effective_entry_offset_s": PHYSICS_DT_S,
                    "fsm_clock_steps_during_priming": 0,
                    "episode_clock_steps_during_priming": 0,
                    "current_contact_force_provenance": (
                        "current_final_solver_force_only"
                    ),
                    "sensor_history_samples_after_reset": 1,
                    "root_state_writes_confined_before_first_episode_tick": True,
                }
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

        controller = dependencies.controller_from_paths(
            self.fsm_path, self.motion_contract_path
        )
        _validate_rate_contract(adapter, controller)
        if loaded_snapshot is not None and snapshot_phase != "P01":
            # This is the sole ContactSensor read after the real prime.  The
            # reader first receives the artifact's historical guard state so
            # the same observation can safely become the controller's episode
            # frame.  Verification independently checks the *current* raw
            # PhysX force against the real hysteresis threshold: off for an
            # artifact-active pair, on for an artifact-inactive pair.  Sensor
            # force history is intentionally re-warmed, never claimed restored.
            reader = dependencies.reader_from_scene(scene, adapter, backends)
            assert dependencies.restore_guard_snapshot is not None
            assert dependencies.restore_controller_snapshot is not None
            guard_proof = dict(
                dependencies.restore_guard_snapshot(reader, loaded_snapshot.payload)
            )
            controller_proof = dict(
                dependencies.restore_controller_snapshot(
                    controller, loaded_snapshot.payload
                )
            )
            classifier = getattr(reader, "contact_classifier", None)
            contact_force_on_n = float(getattr(classifier, "force_on_n", math.nan))
            contact_force_off_n = float(getattr(classifier, "force_off_n", math.nan))
            if (
                not math.isfinite(contact_force_on_n)
                or not math.isfinite(contact_force_off_n)
                or contact_force_on_n <= 0.0
                or contact_force_off_n < 0.0
                or contact_force_off_n >= contact_force_on_n
            ):
                raise SensorContractFailure(
                    "phase snapshot live restoration could not be proven: "
                    "classifier raw-force hysteresis thresholds are unavailable"
                )
            observation = reader.read(
                physics_tick=0,
                simulation_time_s=0.0,
                commanded_full12=prime_acks[-1]["drive_target_full12"],
            )
            _validate_sensor_contract(
                observation,
                dependencies.expected_contact_bodies,
                require_finite=True,
            )
            priming_comparison = _compare_phase_snapshot_observation(
                observation,
                loaded_snapshot.payload,
                contact_force_on_n=contact_force_on_n,
                contact_force_off_n=contact_force_off_n,
            )
            physical_proof.update(
                {
                    "priming_observation": priming_comparison,
                    "contact_sensor_reads_after_prime": 1,
                    "classifier_restored_before_only_episode_read": True,
                    "classifier_current_force_hysteresis_contract_verified": True,
                    "classifier_history_equivalence_claimed": False,
                    "raw_sensor_history_rewarmed_from_prime": True,
                    "restored_classifier_state_used": "hysteresis_active_state_only",
                    "restored_guard_state_used": "cumulative_event_latches",
                    "contact_backend_reset_after_prime": False,
                }
            )
            observation_proof = _verify_phase_snapshot_observation(
                observation,
                loaded_snapshot.payload,
                contact_force_on_n=contact_force_on_n,
                contact_force_off_n=contact_force_off_n,
            )
            physical_proof.update(
                {
                    "episode_live_observation": observation_proof,
                    "episode_verification_followed_classifier_restore": True,
                }
            )
            self._snapshot_restoration.update(
                {
                    "guard_state": guard_proof,
                    "controller_state": controller_proof,
                    "live_observation": observation_proof,
                }
            )
        else:
            reader = dependencies.reader_from_scene(scene, adapter, backends)
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
            SETTLE_TICKS
            + self._reset_prime_tick_count
            + self._video_pre_action_tick_count
            + self._episode_tick
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
        if self._episode_tick == 0:
            if self._first_episode_physical_command_tick_actual is not None:
                raise IsaacFSMBackendError(
                    "first episode physical command tick was already recorded"
                )
            self._first_episode_physical_command_tick_actual = physical_tick
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
        physical_tick = (
            SETTLE_TICKS
            + self._reset_prime_tick_count
            + self._video_pre_action_tick_count
        )
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
            + self._reset_prime_tick_count
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
            "first_episode_physical_command_tick_actual": (
                self._first_episode_physical_command_tick_actual
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
        restoration_mode = self._snapshot_restoration.get("mode")
        if self._reset_prime_tick_count > 0:
            if self._reset_prime_tick_count != PHASE_SNAPSHOT_PRIME_PHYSICS_STEPS:
                raise IsaacFSMBackendError(
                    "phase snapshot reset prime tick count is not exactly one"
                )
            effective_entry_semantics = "snapshot_plus_one_physics_tick"
        elif (
            restoration_mode == "normal_p01_reset"
            and self._snapshot_restoration.get("snapshot_validated") is True
        ):
            effective_entry_semantics = "validated_p01_natural_post_settle"
        else:
            effective_entry_semantics = "natural_p01_post_settle"
        return {
            "environment_hash": environment_hash,
            "robot_asset_hash": self._dependencies.robot_asset_hash,
            "canonical_reset_state_source": "baseline_order_post_limit_authoring_pre_settle",
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
            "pre_limit_native_state_observed_sha256": reset_writes.get(
                "pre_limit_native_state_observed_sha256"
            ),
            "pre_limit_native_state_instance_count": int(
                reset_writes.get("pre_limit_native_state_instance_count", 0)
            ),
            "pre_limit_native_state_matches_canonical": bool(
                reset_writes.get(
                    "pre_limit_native_state_matches_canonical", False
                )
            ),
            "pre_settle_native_state_observed_sha256": reset_writes.get(
                "pre_settle_native_state_observed_sha256"
            ),
            "pre_settle_native_state_matches_canonical": bool(
                reset_writes.get(
                    "pre_settle_native_state_matches_canonical", False
                )
            ),
            "canonical_settled_state_sha256": (
                None
                if self._canonical_settled_state is None
                else self._canonical_settled_state.state_sha256
            ),
            "canonical_settled_state_source": "natural_post_baseline_order_settle",
            "canonical_settled_restore_applied": bool(
                reset_writes.get("canonical_settled_restore_applied", False)
            ),
            "canonical_settled_applied_sha256": reset_writes.get(
                "canonical_settled_applied_sha256"
            ),
            "observed_settled_state_sha256": reset_writes.get(
                "observed_settled_state_sha256"
            ),
            "physics_lifecycle_reset": reset_writes.get(
                "physics_lifecycle_reset"
            ),
            "reset_contact_sensor_count": int(
                reset_writes.get("reset_contact_sensor_count", 0)
            ),
            "reset_initialization_order": reset_writes.get(
                "reset_initialization_order"
            ),
            "pre_physics_session_limit_state_sha256": reset_writes.get(
                "pre_physics_session_limit_state_sha256"
            ),
            "pre_physics_composed_limit_state_sha256": reset_writes.get(
                "pre_physics_composed_limit_state_sha256"
            ),
            "pre_physics_composed_limit_state_matches_canonical": bool(
                reset_writes.get(
                    "pre_physics_composed_limit_state_matches_canonical",
                    False,
                )
            ),
            "session_limit_specs_present_during_physics_reset": int(
                reset_writes.get(
                    "session_limit_specs_present_during_physics_reset", -1
                )
            ),
            "session_limit_specs_removed_before_reset": int(
                reset_writes.get("session_limit_specs_removed_before_reset", 0)
            ),
            "removed_session_limit_state_sha256": reset_writes.get(
                "removed_session_limit_state_sha256"
            ),
            "post_author_session_limit_state_sha256": reset_writes.get(
                "post_author_session_limit_state_sha256"
            ),
            "post_author_session_limit_state_matches_canonical": bool(
                reset_writes.get(
                    "post_author_session_limit_state_matches_canonical", False
                )
            ),
            "session_limit_specs_after_authoring": int(
                reset_writes.get("session_limit_specs_after_authoring", 0)
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
            "reset_prime_tick_count": self._reset_prime_tick_count,
            "reset_prime_duration_s": self._reset_prime_tick_count * PHYSICS_DT_S,
            "next_post_reset_command_tick": (
                SETTLE_TICKS + self._reset_prime_tick_count
            ),
            "effective_phase_entry_semantics": effective_entry_semantics,
            "fsm_and_episode_clock_at_effective_entry": 0,
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


def _load_validated_phase_snapshot(
    phase_id: str,
    *,
    expected_bundle: ValidatedPhaseSnapshotBundle | None = None,
) -> LoadedPhaseSnapshot:
    """Load one snapshot from one immutable buffer pinned to the local bundle."""

    from .phase_snapshots import (
        PhaseSnapshotError,
        assert_phase_snapshot_bundle_unchanged,
        capture_validated_phase_snapshot_bundle,
        load_validated_phase_snapshot_payload,
    )

    if phase_id not in PHASE_IDS:
        raise IsaacFSMBackendError(f"unknown phase snapshot {phase_id!r}")
    root = DEFAULT_PHASE_SNAPSHOT_ROOT.resolve()
    try:
        if expected_bundle is None:
            bundle = capture_validated_phase_snapshot_bundle(
                root, canonical_root=root
            )
        else:
            if expected_bundle.snapshot_root != root:
                raise PhaseSnapshotError(
                    "pinned phase snapshot bundle uses a different loader root"
                )
            bundle = assert_phase_snapshot_bundle_unchanged(
                expected_bundle, canonical_root=root
            )
        payload, entry = load_validated_phase_snapshot_payload(bundle, phase_id)
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"phase snapshot bundle validation failed: {type(exc).__name__}: {exc}"
        ) from exc
    _validate_phase_snapshot_payload(payload, phase_id)
    return LoadedPhaseSnapshot(
        phase_id=phase_id,
        payload=payload,
        state_sha256=entry.state_sha256,
        file_sha256=entry.file_sha256,
        snapshot_path=entry.snapshot_path,
    )


def _validate_phase_snapshot_payload(
    payload: Mapping[str, Any], phase_id: str
) -> None:
    failures: list[str] = []
    try:
        validate_phase_snapshot_payload_contract(payload, phase_id)
    except PhaseSnapshotError as exc:
        failures.append(str(exc))
    expected_history = list(PHASE_IDS[: PHASE_IDS.index(phase_id)])
    if payload.get("schema") != SNAPSHOT_SCHEMA:
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


def _source_replay_values_match(left: Any, right: Any) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left is right
    if type(left) is int and type(right) is int:
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _source_replay_values_match(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _source_replay_numeric_error(left: Any, right: Any) -> float:
    if type(left) is bool or type(right) is bool:
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max(
            (_source_replay_numeric_error(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    return 0.0


def _live_source_mapper_state(
    adapter: Any,
    *,
    source_control_physics_tick: int | None,
) -> dict[str, Any]:
    mapper = getattr(adapter, "servo_target_mapper", None)
    required = (
        "_requested",
        "_applied",
        "_nominal_reached",
        "_compensation",
        "_tracking_active",
        "_retiring_stale_bias",
        "_feedback_tick",
    )
    if mapper is None or any(not hasattr(mapper, name) for name in required):
        raise IsaacFSMBackendError(
            "cannot prove source mapper replay state on RobotAdapter"
        )
    final_drive = getattr(adapter, "_final_drive_servo_deg", None)
    if not isinstance(final_drive, Mapping) or set(final_drive) != set(SERVO_ORDER):
        raise IsaacFSMBackendError(
            "cannot prove source final-drive replay state on RobotAdapter"
        )
    return {
        "schema": SOURCE_MAPPER_STATE_SCHEMA,
        "source_control_physics_tick": source_control_physics_tick,
        "requested_servo_deg": [mapper._requested[name] for name in SERVO_ORDER],
        "applied_drive_command_deg": [mapper._applied[name] for name in SERVO_ORDER],
        "nominal_target_reached": [mapper._nominal_reached[name] for name in SERVO_ORDER],
        "tracking_compensation_deg": [mapper._compensation[name] for name in SERVO_ORDER],
        "tracking_active": [mapper._tracking_active[name] for name in SERVO_ORDER],
        "retiring_stale_bias": [mapper._retiring_stale_bias[name] for name in SERVO_ORDER],
        "feedback_tick": int(mapper._feedback_tick),
        "final_drive_servo_deg": [final_drive[name] for name in SERVO_ORDER],
    }


def _install_source_mapper_pre_state(
    adapter: Any,
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load source t-1 state only; articulation targets remain untouched."""

    source = snapshot["source_command"]
    configuration = source["mapper_configuration"]
    expected = source["mapper_pre_state"]
    mapper = getattr(adapter, "servo_target_mapper", None)
    config_fields = {
        "physics_dt_s": getattr(mapper, "physics_dt_s", None),
        "servo_rate_deg_s": getattr(mapper, "servo_rate_deg_s", None),
        "maximum_delta_deg": getattr(mapper, "maximum_delta_deg", None),
        "tracking_gain": getattr(mapper, "tracking_gain", None),
        "tracking_limit_deg": getattr(mapper, "tracking_limit_deg", None),
        "feedback_interval_ticks": getattr(mapper, "feedback_interval_ticks", None),
        "standing_pose_deg": [
            getattr(mapper, "standing_pose_deg", {}).get(name) for name in SERVO_ORDER
        ],
    }
    config_matches = {
        name: _source_replay_values_match(configuration[name], config_fields[name])
        for name in configuration
    }
    if not all(config_matches.values()):
        mismatched = [name for name, match in config_matches.items() if not match]
        raise IsaacFSMBackendError(
            "live RobotAdapter mapper configuration differs from snapshot source: "
            + ", ".join(mismatched)
        )
    for index, name in enumerate(SERVO_ORDER):
        mapper._requested[name] = float(expected["requested_servo_deg"][index])
        mapper._applied[name] = float(expected["applied_drive_command_deg"][index])
        mapper._nominal_reached[name] = bool(expected["nominal_target_reached"][index])
        mapper._compensation[name] = float(expected["tracking_compensation_deg"][index])
        mapper._tracking_active[name] = bool(expected["tracking_active"][index])
        mapper._retiring_stale_bias[name] = bool(expected["retiring_stale_bias"][index])
        adapter._final_drive_servo_deg[name] = float(
            expected["final_drive_servo_deg"][index]
        )
    mapper._feedback_tick = int(expected["feedback_tick"])
    observed = _live_source_mapper_state(
        adapter,
        source_control_physics_tick=expected["source_control_physics_tick"],
    )
    state_matches = {
        field: _source_replay_values_match(expected[field], observed[field])
        for field in expected
    }
    if not all(state_matches.values()):
        raise IsaacFSMBackendError(
            "RobotAdapter did not retain the source mapper pre-state"
        )
    return {
        "schema": "wlr50_clean.phase_snapshot_source_mapper_pre_state.v1",
        "source_transition": "source_tick_t_minus_1_to_t",
        "configuration_matches": config_matches,
        "all_configuration_fields_match": True,
        "pre_state_sha256": _canonical_hash(expected),
        "all_pre_state_fields_match": True,
        "articulation_writes": 0,
    }


def _verify_source_mapper_post_state(
    adapter: Any,
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = snapshot["source_command"]
    expected = source["mapper_post_state"]
    observed = _live_source_mapper_state(
        adapter,
        source_control_physics_tick=int(snapshot["source_tick"]),
    )
    matches = {
        field: _source_replay_values_match(expected[field], observed[field])
        for field in expected
    }
    errors = {
        field: _source_replay_numeric_error(expected[field], observed[field])
        for field in expected
    }
    if not all(matches.values()):
        mismatched = [name for name, match in matches.items() if not match]
        raise IsaacFSMBackendError(
            "source mapper post-state replay mismatch: " + ", ".join(mismatched)
        )
    return {
        "schema": "wlr50_clean.phase_snapshot_source_mapper_post_state.v1",
        "source_transition": "source_tick_t_minus_1_to_t",
        "field_matches": matches,
        "field_maximum_numeric_error": errors,
        "all_fields_match": True,
        "post_state_sha256": _canonical_hash(expected),
        "restored_after_prime": False,
        "reached_naturally_by_single_atomic_apply": True,
    }


def _verify_source_prime_ack(
    snapshot: Mapping[str, Any],
    ack: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = snapshot["source_command"]
    expected = source["expected_atomic_ack"]
    missing = [field for field in SOURCE_ACK_MATCH_FIELDS if field not in ack]
    if missing:
        raise IsaacFSMBackendError(
            f"phase snapshot prime ack lacks source fields: {missing}"
        )
    matches = {
        field: _source_replay_values_match(expected[field], ack[field])
        for field in SOURCE_ACK_MATCH_FIELDS
    }
    errors = {
        field: _source_replay_numeric_error(expected[field], ack[field])
        for field in SOURCE_ACK_MATCH_FIELDS
    }
    if not all(matches.values()):
        mismatched = [name for name, match in matches.items() if not match]
        raise IsaacFSMBackendError(
            "phase snapshot prime differs from authoritative source ack: "
            + ", ".join(mismatched)
        )
    source_target_sha256 = str(source["drive_target_full12_sha256"])
    replayed_target_sha256 = phase_snapshot_drive_target_sha256(
        ack["drive_target_full12"]
    )
    source_actuation_sha256 = str(source["actuation_contract_sha256"])
    replayed_actuation_sha256 = phase_snapshot_actuation_contract_sha256(ack)
    if (
        replayed_target_sha256 != source_target_sha256
        or replayed_actuation_sha256 != source_actuation_sha256
    ):
        raise IsaacFSMBackendError(
            "phase snapshot prime hashes differ from authoritative source actuation"
        )
    artifacts = snapshot["source_artifacts"]
    return {
        "schema": "wlr50_clean.phase_snapshot_source_actuation_match.v1",
        "source_transition": "source_tick_t_minus_1_to_t",
        "source_command_file_sha256": artifacts["command"]["sha256"],
        "source_observation_file_sha256": artifacts["observation"]["sha256"],
        "source_command_row_canonical_sha256": source[
            "source_command_row_canonical_sha256"
        ],
        "source_observation_row_canonical_sha256": source[
            "source_observation_row_canonical_sha256"
        ],
        "source_drive_target_full12_sha256": source_target_sha256,
        "replayed_drive_target_full12_sha256": replayed_target_sha256,
        "source_actuation_contract_sha256": source_actuation_sha256,
        "replayed_actuation_contract_sha256": replayed_actuation_sha256,
        "field_matches": matches,
        "field_maximum_numeric_error": errors,
        "all_fields_match": True,
        "source_target_hash_matches": True,
        "logical_target_fallback_used": False,
        "source_atomic_physics_tick": source["source_atomic_physics_tick"],
        "reset_prime_physics_tick": int(ack["physics_tick"]),
        "source_atomic_write_count": source["source_atomic_write_count"],
        "reset_prime_write_count": int(ack["write_count"]),
        "clock_and_write_count_fields_intentionally_remapped": True,
    }


def _verify_pre_prime_root_link_write(
    robot: Any,
    root: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read back the written ``base_link`` state without sampling sensors.

    The snapshot records the source observation's base-link state, so the
    root velocity must be checked through Isaac Lab's link-state tensors.  A
    root-write call count alone cannot prove that the requested state reached
    PhysX, especially because the legacy root-velocity alias addresses the
    center-of-mass velocity instead.
    """

    body_names = tuple(str(name) for name in getattr(robot, "body_names", ()))
    if body_names.count("base_link") != 1:
        raise IsaacFSMBackendError(
            "phase snapshot base_link root-state readback is unavailable"
        )
    base_index = body_names.index("base_link")
    data = getattr(robot, "data", None)
    if data is None:
        raise IsaacFSMBackendError(
            "phase snapshot base_link root-state readback is unavailable"
        )

    def read_vector(field: str, size: int) -> tuple[float, ...]:
        tensor = getattr(data, field, None)
        try:
            current = tensor[0, base_index]
        except (IndexError, KeyError, TypeError) as exc:
            raise IsaacFSMBackendError(
                f"phase snapshot base_link {field} readback is unavailable"
            ) from exc
        values = _flat_finite_tensor_values(
            current, f"phase snapshot base_link {field} readback"
        )
        if len(values) != size:
            raise IsaacFSMBackendError(
                f"phase snapshot base_link {field} readback has the wrong shape"
            )
        return values

    observed = {
        "position_w_m": read_vector("body_link_pos_w", 3),
        "orientation_wxyz": read_vector("body_link_quat_w", 4),
        "linear_velocity_w_m_s": read_vector("body_link_lin_vel_w", 3),
        "angular_velocity_w_rad_s": read_vector("body_link_ang_vel_w", 3),
    }
    expected = {
        "position_w_m": tuple(float(value) for value in root["position_w_m"]),
        "orientation_wxyz": tuple(
            float(value) for value in root["orientation_wxyz"]
        ),
        "linear_velocity_w_m_s": tuple(
            float(value) for value in root["linear_velocity_w_m_s"]
        ),
        "angular_velocity_w_rad_s": tuple(
            float(value) for value in root["angular_velocity_w_rad_s"]
        ),
    }
    tolerances = {
        "root_position_m": 2.0e-4,
        "root_orientation_quaternion_distance": 2.0e-5,
        "root_linear_velocity_m_s": 2.0e-4,
        "root_angular_velocity_rad_s": 2.0e-4,
    }
    errors = {
        "root_position_m": _maximum_absolute_error(
            observed["position_w_m"], expected["position_w_m"]
        ),
        "root_orientation_quaternion_distance": _quaternion_distance(
            observed["orientation_wxyz"], expected["orientation_wxyz"]
        ),
        "root_linear_velocity_m_s": _maximum_absolute_error(
            observed["linear_velocity_w_m_s"],
            expected["linear_velocity_w_m_s"],
        ),
        "root_angular_velocity_rad_s": _maximum_absolute_error(
            observed["angular_velocity_w_rad_s"],
            expected["angular_velocity_w_rad_s"],
        ),
    }
    within_tolerances = all(
        errors[name] <= tolerance for name, tolerance in tolerances.items()
    )
    if not within_tolerances:
        mismatched = [
            name
            for name, tolerance in tolerances.items()
            if errors[name] > tolerance
        ]
        raise IsaacFSMBackendError(
            "phase snapshot base_link readback differs from the reset-only write: "
            + ", ".join(mismatched)
        )
    return {
        "schema": "wlr50_clean.phase_snapshot_pre_prime_root_link_readback.v1",
        "body_name": "base_link",
        "snapshot_state_semantics": "base_link_link_frame_state",
        "source_fields": {
            "position_w_m": "robot.data.body_link_pos_w",
            "orientation_wxyz": "robot.data.body_link_quat_w",
            "linear_velocity_w_m_s": "robot.data.body_link_lin_vel_w",
            "angular_velocity_w_rad_s": "robot.data.body_link_ang_vel_w",
        },
        "read_after_simulation_forward": True,
        "read_after_robot_update_zero_dt": True,
        "physics_steps_before_readback": 0,
        "contact_sensor_reads_before_readback": 0,
        "expected": {name: list(values) for name, values in expected.items()},
        "observed": {name: list(values) for name, values in observed.items()},
        "maximum_errors": errors,
        "production_tolerances": tolerances,
        "all_values_finite": True,
        "all_fields_within_production_tolerances": True,
        "verified": True,
    }


def _write_phase_snapshot_state(
    scene: Any,
    adapter: Any,
    snapshot: Mapping[str, Any],
    *,
    reset_contact_backend: bool = True,
    state_write_index: int = 1,
) -> Mapping[str, Any]:
    """Write one validated phase state before its reset-only contact prime."""

    from wlr50_clean.infrastructure.command_batch import (
        SERVO_COMMAND_SIGN,
        WHEEL_FORWARD_SIGN,
    )

    if reset_contact_backend is not True or state_write_index != 1:
        raise IsaacFSMBackendError(
            "production phase reset permits exactly one pre-prime state write"
        )
    robot = scene.robot
    root = snapshot["root_state"]
    joint = snapshot["joint_state"]
    wheel = snapshot["wheel_state"]
    position = tuple(float(value) for value in joint["logical_position_deg"])
    servo_velocity = tuple(float(value) for value in joint["logical_velocity_deg_s"])
    wheel_velocity = tuple(float(value) for value in wheel["logical_velocity_rad_s"])
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
        write_root_link_velocity = getattr(
            robot, "write_root_link_velocity_to_sim", None
        )
        if not callable(write_root_link_velocity):
            raise IsaacFSMBackendError(
                "phase snapshot requires Isaac Lab root-link velocity writes"
            )
        write_root_link_velocity(root_state[:, 7:])
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

    pre_prime_root_link_readback = dict(
        _verify_pre_prime_root_link_write(robot, root)
    )
    source_mapper_pre_state = dict(
        _install_source_mapper_pre_state(adapter, snapshot)
    )
    mapper = adapter.servo_target_mapper

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
        "state_write_index": 1,
        "contact_backend_reset": True,
        "root_velocity_write_api": "write_root_link_velocity_to_sim",
        "adapter_mapper_restored": True,
        "adapter_mapper_restored_to": "source tick t-1 post-command state",
        "source_mapper_pre_state": source_mapper_pre_state,
        "source_transition": "source_tick_t_minus_1_to_t",
        "adapter_feedback_tick": mapper._feedback_tick,
        "maximum_logical_joint_or_wheel_error": position_error,
        "maximum_physical_servo_velocity_error_rad_s": velocity_error,
        "pre_prime_root_link_readback": pre_prime_root_link_readback,
        "pre_prime_joint_state_verified": True,
        "pre_prime_state_verified": True,
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


def _compare_phase_snapshot_observation(
    observation: Any,
    snapshot: Mapping[str, Any],
    *,
    contact_force_on_n: float,
    contact_force_off_n: float,
) -> dict[str, Any]:
    """Return exact production errors plus independently sourced raw contacts."""

    tolerances = {
        "root_position_m": 2.0e-4,
        "root_orientation_quaternion_distance": 2.0e-5,
        "root_velocity": 2.0e-4,
        "servo_position_deg": 0.02,
        "servo_velocity_deg_s": 0.02,
        "wheel_velocity_rad_s": 2.0e-4,
        "wheel_geometry_m": 0.002,
        "wheel_contact_state": "exact",
        "raw_contact_source": "isaaclab.ContactSensor.force_matrix_w",
        "raw_contact_force_on_n": contact_force_on_n,
        "raw_contact_force_off_n": contact_force_off_n,
    }
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
    exact_contacts: dict[str, Any] = {}
    raw_contacts: dict[str, Any] = {}
    for wheel_name in WHEEL_ORDER:
        wheel = wheels[wheel_name]
        body_name = str(_member(wheel, "body_name"))
        contact = contacts[body_name]
        expected = snapshot["contact_state"][wheel_name]
        actual_class = _enum_value(_member(contact, "contact_class"))
        actual_ground = bool(
            _member(_member(contact, "ground"), "active", False)
        )
        actual_obstacle = bool(
            _member(_member(contact, "obstacle"), "active", False)
        )
        matches = bool(
            actual_class != str(expected["class"])
            or actual_ground != bool(expected["ground_active"])
            or actual_obstacle != bool(expected["obstacle_active"])
        ) is False
        exact_contacts[wheel_name] = {
            "body_name": body_name,
            "expected_class": str(expected["class"]),
            "actual_class": actual_class,
            "expected_ground_active": bool(expected["ground_active"]),
            "actual_ground_active": actual_ground,
            "expected_obstacle_active": bool(expected["obstacle_active"]),
            "actual_obstacle_active": actual_obstacle,
            "matches": matches,
        }
        if not matches:
            failures.append(f"{wheel_name} exact contact state")

        raw_contacts[wheel_name] = {}
        for pair_name in ("ground", "obstacle"):
            pair = _member(contact, pair_name)
            pair_verified = bool(_member(pair, "pair_verified", False))
            source = str(_member(pair, "source", ""))
            force = _finite_vector(_member(pair, "force_w_n", ()), 3)
            force_norm = (
                None
                if force is None
                else math.sqrt(sum(value * value for value in force))
            )
            expected_active = bool(expected[f"{pair_name}_active"])
            required_threshold = (
                contact_force_off_n if expected_active else contact_force_on_n
            )
            hysteresis_contract_matches = bool(
                pair_verified
                and force_norm is not None
                and (
                    force_norm >= required_threshold
                    if expected_active
                    else force_norm < required_threshold
                )
            )
            active_from_fresh_on_threshold = bool(
                pair_verified
                and force_norm is not None
                and force_norm >= contact_force_on_n
            )
            raw_contacts[wheel_name][pair_name] = {
                "pair_verified": pair_verified,
                "source": source,
                "force_w_n": None if force is None else list(force),
                "force_norm_n": force_norm,
                "force_on_n": contact_force_on_n,
                "force_off_n": contact_force_off_n,
                "expected_active": expected_active,
                "required_current_force_threshold_n": required_threshold,
                "required_threshold_kind": (
                    "force_off_for_restored_active_state"
                    if expected_active
                    else "force_on_for_restored_inactive_state"
                ),
                "current_force_hysteresis_contract_matches_snapshot": (
                    hysteresis_contract_matches
                ),
                "active_from_fresh_on_threshold": active_from_fresh_on_threshold,
                "fresh_on_threshold_matches_snapshot": (
                    active_from_fresh_on_threshold == expected_active
                ),
            }
            if (
                not pair_verified
                or source != "isaaclab.ContactSensor.force_matrix_w"
                or force is None
            ):
                failures.append(f"{wheel_name} {pair_name} raw PhysX contact source")
            if not hysteresis_contract_matches:
                failures.append(
                    f"{wheel_name} {pair_name} current raw PhysX force hysteresis"
                )

    physical_ok = not any(
        (
            errors["root_position_m"] > 2.0e-4,
            errors["root_orientation"] > 2.0e-5,
            errors["root_linear_velocity_m_s"] > 2.0e-4,
            errors["root_angular_velocity_rad_s"] > 2.0e-4,
            errors["servo_position_deg"] > 0.02,
            errors["servo_velocity_deg_s"] > 0.02,
            errors["wheel_velocity_rad_s"] > 2.0e-4,
            errors["wheel_center_m"] > 0.002,
            errors["wheel_bottom_m"] > 0.002,
        )
    )
    raw_contact_record = {
        "schema": "wlr50_clean.phase_snapshot_raw_physx_contact.v1",
        "pairs": raw_contacts,
    }
    raw_contact_record["sha256"] = _canonical_hash(raw_contact_record)
    return {
        "schema": "wlr50_clean.phase_snapshot_live_comparison.v1",
        "verified": not failures,
        "failures": list(dict.fromkeys(failures)),
        "tolerances": tolerances,
        "maximum_errors": errors,
        "physical_state_within_production_tolerances": physical_ok,
        "exact_contacts": exact_contacts,
        "exact_contacts_match": all(
            row["matches"] for row in exact_contacts.values()
        ),
        "raw_physx_contacts": raw_contact_record,
        "raw_physx_contact_sources_verified": all(
            row[pair]["pair_verified"]
            and row[pair]["source"]
            == "isaaclab.ContactSensor.force_matrix_w"
            and row[pair]["force_w_n"] is not None
            for row in raw_contacts.values()
            for pair in ("ground", "obstacle")
        ),
        "current_raw_force_hysteresis_contract_matches_snapshot": all(
            row[pair]["current_force_hysteresis_contract_matches_snapshot"]
            for row in raw_contacts.values()
            for pair in ("ground", "obstacle")
        ),
        "strong_fresh_on_threshold_diagnostic_matches_snapshot": all(
            row[pair]["fresh_on_threshold_matches_snapshot"]
            for row in raw_contacts.values()
            for pair in ("ground", "obstacle")
        ),
    }


def _verify_phase_snapshot_observation(
    observation: Any,
    snapshot: Mapping[str, Any],
    *,
    contact_force_on_n: float,
    contact_force_off_n: float,
) -> Mapping[str, Any]:
    """Fail closed unless the post-prime live sample reproduces the artifact."""

    comparison = _compare_phase_snapshot_observation(
        observation,
        snapshot,
        contact_force_on_n=contact_force_on_n,
        contact_force_off_n=contact_force_off_n,
    )
    failures = tuple(str(value) for value in comparison["failures"])
    if failures:
        raise SensorContractFailure(
            "phase snapshot live restoration could not be proven: "
            + ", ".join(failures)
        )
    return {
        **comparison,
        "schema": "wlr50_clean.phase_snapshot_live_proof.v1",
        "verified": True,
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


_SERVO_LIMIT_ATTRIBUTE_NAMES = (
    "physics:lowerLimit",
    "physics:upperLimit",
)


def _session_servo_limit_state(scene: Any) -> dict[str, Any]:
    """Describe only the 16 live limit opinions in the USD session layer."""

    try:
        from isaaclab.sim import get_current_stage  # type: ignore
        from pxr import Sdf, Usd, UsdPhysics  # type: ignore

        from wlr50_clean.infrastructure.scene_factory import ROBOT_PRIM_PATH

        stage = get_current_stage()
        scene_stage = getattr(getattr(scene, "sim", None), "stage", None)
        if (
            scene_stage is None
            or stage != scene_stage
            or stage.GetRootLayer() != scene_stage.GetRootLayer()
            or stage.GetSessionLayer() != scene_stage.GetSessionLayer()
        ):
            raise IsaacFSMBackendError(
                "current USD stage differs from the backend scene stage"
            )
        token_names = (
            str(UsdPhysics.Tokens.physicsLowerLimit),
            str(UsdPhysics.Tokens.physicsUpperLimit),
        )
        if token_names != _SERVO_LIMIT_ATTRIBUTE_NAMES:
            raise IsaacFSMBackendError(
                "USD Physics servo-limit tokens differ from the locked contract"
            )
        root_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not root_prim.IsValid():
            raise IsaacFSMBackendError(
                f"runtime robot prim is missing: {ROBOT_PRIM_PATH}"
            )
        session_layer = stage.GetSessionLayer()
        if session_layer is None:
            raise IsaacFSMBackendError("live USD stage has no session layer")

        resolved: list[tuple[str, Any]] = []
        for joint_name in SERVO_ORDER:
            matches = [
                prim
                for prim in Usd.PrimRange(root_prim)
                if prim.GetName() == joint_name
                and prim.IsA(UsdPhysics.RevoluteJoint)
            ]
            if len(matches) != 1:
                raise IsaacFSMBackendError(
                    f"expected one RevoluteJoint prim named {joint_name}; "
                    f"found {[str(prim.GetPath()) for prim in matches]}"
                )
            resolved.append((joint_name, matches[0]))

        properties: list[dict[str, Any]] = []
        composed_properties: list[dict[str, Any]] = []
        for joint_name, prim in resolved:
            for attribute_name in _SERVO_LIMIT_ATTRIBUTE_NAMES:
                property_path = prim.GetPath().AppendProperty(attribute_name)
                composed_attribute = stage.GetAttributeAtPath(property_path)
                composed_value = (
                    composed_attribute.Get()
                    if composed_attribute.IsValid()
                    else None
                )
                if composed_value is None:
                    canonical_composed_value: float | str = "UNAUTHORED"
                else:
                    try:
                        numeric_composed_value = float(composed_value)
                    except (TypeError, ValueError) as exc:
                        raise IsaacFSMBackendError(
                            f"composed limit is not numeric: {property_path}"
                        ) from exc
                    if math.isnan(numeric_composed_value):
                        raise IsaacFSMBackendError(
                            f"composed limit is NaN: {property_path}"
                        )
                    canonical_composed_value = (
                        numeric_composed_value
                        if math.isfinite(numeric_composed_value)
                        else repr(numeric_composed_value)
                    )
                composed_properties.append(
                    {
                        "property_path": str(property_path),
                        "composed_default_deg": canonical_composed_value,
                    }
                )

                spec = session_layer.GetPropertyAtPath(Sdf.Path(property_path))
                if spec is None:
                    continue
                try:
                    value = float(spec.default)
                except (TypeError, ValueError) as exc:
                    raise IsaacFSMBackendError(
                        f"session limit opinion has no finite numeric default: {property_path}"
                    ) from exc
                if not math.isfinite(value):
                    raise IsaacFSMBackendError(
                        f"session limit opinion is non-finite: {property_path}"
                    )
                properties.append(
                    {
                        "joint_name": joint_name,
                        "prim_path": str(prim.GetPath()),
                        "attribute_name": attribute_name,
                        "property_path": str(property_path),
                        "default_deg": value,
                    }
                )
        properties.sort(key=lambda row: str(row["property_path"]))
        composed_properties.sort(key=lambda row: str(row["property_path"]))
        identifier = str(
            getattr(session_layer, "identifier", "")
            or getattr(session_layer, "GetIdentifier", lambda: "")()
            or "anonymous_session_layer"
        )
        return {
            "schema": "wlr50_clean.session_servo_limit_state.v1",
            "session_layer_identifier": identifier,
            "property_count": len(properties),
            "properties": properties,
            "composed_property_count": len(composed_properties),
            "composed_properties": composed_properties,
            # The anonymous layer identifier is deliberately excluded: it is
            # provenance, not part of the authored physical values.
            "state_sha256": _canonical_hash(properties),
            "composed_state_sha256": _canonical_hash(composed_properties),
        }
    except IsaacFSMBackendError:
        raise
    except Exception as exc:
        raise IsaacFSMBackendError(
            "could not inspect live session-layer servo limits: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _remove_session_servo_limit_specs(
    scene: Any, before: Mapping[str, Any]
) -> dict[str, Any]:
    """Remove, rather than block, the 16 session opinions before PhysX reload."""

    if int(before.get("property_count", -1)) != (
        len(SERVO_ORDER) * len(_SERVO_LIMIT_ATTRIBUTE_NAMES)
    ):
        raise IsaacFSMBackendError(
            "reused reset requires exactly 16 authored session limit specs"
        )
    try:
        from isaaclab.sim import get_current_stage  # type: ignore
        from pxr import Sdf, Usd  # type: ignore

        stage = get_current_stage()
        session_layer = stage.GetSessionLayer()
        if session_layer is None:
            raise IsaacFSMBackendError("live USD stage has no session layer")
        targets: list[tuple[Any, str]] = []
        for row in before.get("properties", ()):
            if not isinstance(row, Mapping):
                raise IsaacFSMBackendError(
                    "session limit state contains a malformed property"
                )
            prim = stage.GetPrimAtPath(str(row.get("prim_path", "")))
            attribute_name = str(row.get("attribute_name", ""))
            if not prim.IsValid() or attribute_name not in (
                _SERVO_LIMIT_ATTRIBUTE_NAMES
            ):
                raise IsaacFSMBackendError(
                    "session limit state contains an invalid target property"
                )
            property_path = prim.GetPath().AppendProperty(attribute_name)
            if session_layer.GetPropertyAtPath(property_path) is None:
                raise IsaacFSMBackendError(
                    f"session limit property disappeared before removal: {property_path}"
                )
            targets.append((prim, attribute_name))
        edit_target = Usd.EditTarget(session_layer)
        with Sdf.ChangeBlock():
            with Usd.EditContext(stage, edit_target):
                for prim, attribute_name in targets:
                    if not prim.RemoveProperty(attribute_name):
                        raise IsaacFSMBackendError(
                            "could not remove session limit property: "
                        f"{prim.GetPath()}.{attribute_name}"
                    )
        after = _session_servo_limit_state(scene)
        if int(after.get("property_count", -1)) != 0:
            raise IsaacFSMBackendError(
                "session limit property specs remained after exact removal"
            )
        return after
    except IsaacFSMBackendError:
        raise
    except Exception as exc:
        raise IsaacFSMBackendError(
            "could not remove live session-layer servo limits: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _contact_reset_evidence(scene: Any, *, reset_history: bool) -> int:
    instrumentation = getattr(scene, "instrumentation", None)
    contact_backend = getattr(instrumentation, "contact_backend", None)
    if contact_backend is None or not bool(
        getattr(contact_backend, "initialized", False)
    ):
        raise IsaacFSMBackendError(
            "exact-pair sensor bank did not initialize after physics reset"
        )
    if reset_history:
        reset_contacts = getattr(contact_backend, "reset", None)
        if not callable(reset_contacts):
            raise IsaacFSMBackendError("exact-pair sensor bank cannot reset")
        reset_contacts()
    sensor_count = len(getattr(contact_backend, "sensors", {}))
    if sensor_count != 13:
        raise IsaacFSMBackendError(
            f"physics reset expected 13 contact sensors; received {sensor_count}"
        )
    return sensor_count


def _reset_physics_lifecycle(
    scene: Any, canonical_state: CanonicalArticulationResetState | None
) -> Mapping[str, Any]:
    """Reproduce the successful baseline's reset-before-limit order.

    The frozen successful runtime creates PhysX from the source USD first and
    only then authors the 16 live session-layer servo limits.  Reloading PhysX
    while those opinions are already present changes the standing pose by
    several degrees and deterministically blocks P10.  A reused episode must
    therefore remove those exact session specs, perform a hard reload, and let
    the next ``RobotAdapter`` re-author them before the natural settle.

    The fresh scene has already received its one authoritative reset inside
    ``SceneFactory.create_scene``.  No indexed state write is used here.
    """

    try:
        if canonical_state is not None:
            simulation = scene.sim
            stop = getattr(simulation, "stop", None)
            if not callable(stop):
                raise IsaacFSMBackendError(
                    "SimulationContext.stop is required before session limit removal"
                )
            is_stopped = getattr(simulation, "is_stopped", None)
            if not callable(is_stopped):
                raise IsaacFSMBackendError(
                    "SimulationContext.is_stopped is required before session limit removal"
                )
            reset = getattr(simulation, "reset", None)
            if not callable(reset):
                raise IsaacFSMBackendError(
                    "SimulationContext.reset is required for an episode lifecycle reset"
                )

            # IsaacLab's standalone STOP callback renders until the timeline
            # resumes unless this private guard is armed.  A naked stop can
            # therefore turn the reused reset into an unbounded app-update
            # loop.  Treat the known guard as a required lifecycle contract:
            # its absence/type mismatch fails before STOP or USD mutation, and
            # its exact prior value is restored on every transaction exit.
            guard_name = "_disable_app_control_on_stop_handle"
            guard_missing = object()
            prior_guard = getattr(simulation, guard_name, guard_missing)
            if prior_guard is guard_missing:
                raise IsaacFSMBackendError(
                    "SimulationContext app-control STOP guard is unavailable"
                )
            if type(prior_guard) is not bool:
                raise IsaacFSMBackendError(
                    "SimulationContext app-control STOP guard is not boolean"
                )

            try:
                setattr(simulation, guard_name, True)
                if getattr(simulation, guard_name, guard_missing) is not True:
                    raise IsaacFSMBackendError(
                        "could not arm SimulationContext app-control STOP guard"
                    )
                stop()
                if not bool(is_stopped()):
                    raise IsaacFSMBackendError(
                        "physics timeline did not stop before session limit removal"
                    )
                before = _session_servo_limit_state(scene)
                removed_count = int(before.get("property_count", -1))
                removed_sha256 = str(before.get("state_sha256", ""))
                cleared = _remove_session_servo_limit_specs(scene, before)
                reset(soft=False)
            finally:
                setattr(simulation, guard_name, prior_guard)
                if getattr(simulation, guard_name, guard_missing) is not prior_guard:
                    raise IsaacFSMBackendError(
                        "could not restore SimulationContext app-control STOP guard"
                    )

            is_playing = getattr(simulation, "is_playing", None)
            if not callable(is_playing) or not bool(is_playing()):
                raise IsaacFSMBackendError(
                    "physics timeline did not play after the no-limit reset"
                )
            # Match SceneFactory.create_scene exactly: its authoritative reset
            # is followed by an update, not an indexed articulation restore.
            scene.robot.update(0.0)
            lifecycle = "session_limits_removed_then_hard_reset"
            sensor_count = _contact_reset_evidence(scene, reset_history=True)
        else:
            before = _session_servo_limit_state(scene)
            removed_count = 0
            removed_sha256: str | None = None
            if int(before.get("property_count", -1)) != 0:
                raise IsaacFSMBackendError(
                    "fresh SceneFactory reset unexpectedly contains session limit specs"
                )
            cleared = before
            lifecycle = "scene_factory_reset_before_limit_authoring"
            sensor_count = _contact_reset_evidence(scene, reset_history=False)

        pre_physics_state = _session_servo_limit_state(scene)
        if (
            int(pre_physics_state.get("property_count", -1)) != 0
            or pre_physics_state.get("state_sha256")
            != cleared.get("state_sha256")
            or pre_physics_state.get("composed_state_sha256")
            != cleared.get("composed_state_sha256")
            or int(pre_physics_state.get("composed_property_count", -1))
            != len(SERVO_ORDER) * len(_SERVO_LIMIT_ATTRIBUTE_NAMES)
        ):
            raise IsaacFSMBackendError(
                "session limit state changed during the no-limit physics reset"
            )
        native_without_limits = capture_canonical_articulation_reset_state(
            scene.robot
        )
    except IsaacFSMBackendError:
        raise
    except Exception as exc:
        raise IsaacFSMBackendError(
            f"physics lifecycle reset failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "root_pose_writes": 0,
        "root_velocity_writes": 0,
        "joint_state_writes": 0,
        "global_simulation_resets": 1,
        "simulation_forward_syncs": 0,
        "physics_lifecycle_reset": lifecycle,
        "reset_contact_sensor_count": sensor_count,
        "reset_initialization_order": (
            "physics_reset_without_session_limits_then_author_limits_then_settle"
        ),
        "pre_physics_session_limit_state_sha256": pre_physics_state[
            "state_sha256"
        ],
        "pre_physics_composed_limit_state_sha256": pre_physics_state[
            "composed_state_sha256"
        ],
        "session_limit_specs_present_during_physics_reset": int(
            pre_physics_state["property_count"]
        ),
        "session_limit_specs_removed_before_reset": removed_count,
        "removed_session_limit_state_sha256": removed_sha256,
        "pre_limit_native_state_observed_sha256": (
            native_without_limits.state_sha256
        ),
        "pre_limit_native_state_instance_count": (
            native_without_limits.instance_count
        ),
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
