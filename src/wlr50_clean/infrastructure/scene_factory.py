"""Build the hash-locked 50 mm Isaac scene after AppLauncher is running.

The physical configuration is narrowly derived from mature Recording
``sim_obstacle_scene.py`` (source SHA-256
``ed42de331eedc36919af1775b6d754457dda283386a3e3cb9984c8e2de0fe471``)
and the clean environment lock. Isaac imports are intentionally lazy so this
module remains importable in ordinary Python. This module never saves a stage.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .command_batch import SERVO_ORDER, WHEEL_ORDER, WHEEL_VELOCITY_LIMIT_RAD_S


ROBOT_USD_PATH = Path("C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd")
ROBOT_USD_SHA256 = "e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892"

ROBOT_PRIM_PATH = "/World/WLRRobot"
GROUND_PRIM_PATH = "/World/defaultGroundPlane"
OBSTACLE_PRIM_PATH = "/World/Obstacle"

PHYSICS_DT_S = 1.0 / 120.0
RENDER_INTERVAL_PHYSICS_STEPS = 8
DEVICE = "cuda:0"
GRAVITY_M_S2 = (0.0, 0.0, -9.81)

ROBOT_SPAWN_POSITION_M = (0.0, 0.0, 0.04)
ROBOT_SPAWN_ROTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
MAX_DEPENETRATION_VELOCITY_M_S = 1.0
SOLVER_POSITION_ITERATIONS = 8
SOLVER_VELOCITY_ITERATIONS = 2

SERVO_EFFORT_LIMIT_NM = 2.7
SERVO_STIFFNESS = 600.0
SERVO_DAMPING = 60.0
SERVO_ARMATURE = 0.005
WHEEL_STIFFNESS = 0.0
WHEEL_DAMPING = 20.0
WHEEL_ARMATURE = 0.002

GROUND_SIZE_M = (6.0, 6.0)
GROUND_TRANSLATION_M = (0.0, 0.0, 0.0)
GROUND_STATIC_FRICTION = 1.25
GROUND_DYNAMIC_FRICTION = 1.05
GROUND_RESTITUTION = 0.0

OBSTACLE_HEIGHT_M = 0.05
OBSTACLE_FRONT_FACE_X_M = 0.5213121737735307
OBSTACLE_LENGTH_M = 2.057375557085507
OBSTACLE_WIDTH_M = 2.0
OBSTACLE_CENTER_M = (1.5499999523162842, 0.0, 0.025)
OBSTACLE_CONTACT_OFFSET_M = 0.005
OBSTACLE_REST_OFFSET_M = 0.0
OBSTACLE_STATIC_FRICTION = 1.2
OBSTACLE_DYNAMIC_FRICTION = 1.0
OBSTACLE_RESTITUTION = 0.0

CAMERA_EYE_M = (1.45, -1.25, 0.80)
CAMERA_TARGET_M = (0.45, 0.0, 0.12)


class SceneContractError(RuntimeError):
    """Raised before scene construction when a locked dependency is wrong."""


@dataclass(frozen=True, slots=True)
class SceneHandle:
    """Objects owned by the runtime; SimulationApp ownership remains external."""

    sim: Any
    robot: Any
    simulation_app: Any | None = None
    instrumentation: Any | None = None

    @property
    def physics_dt_s(self) -> float:
        return float(self.sim.get_physics_dt())

    def app_is_running(self) -> bool:
        return self.simulation_app is None or bool(self.simulation_app.is_running())


def locked_scene_snapshot() -> dict[str, Any]:
    """Return an Isaac-free representation suitable for tests and manifests."""

    return {
        "robot": {
            "usd_path": ROBOT_USD_PATH.as_posix(),
            "usd_sha256": ROBOT_USD_SHA256,
            "prim_path": ROBOT_PRIM_PATH,
            "spawn_position_m": ROBOT_SPAWN_POSITION_M,
            "spawn_rotation_wxyz": ROBOT_SPAWN_ROTATION_WXYZ,
        },
        "physics": {
            "dt_s": PHYSICS_DT_S,
            "render_interval_physics_steps": RENDER_INTERVAL_PHYSICS_STEPS,
            "device": DEVICE,
            "gravity_m_s2": GRAVITY_M_S2,
            "solver_position_iterations": SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": SOLVER_VELOCITY_ITERATIONS,
        },
        "actuators": {
            "servo": {
                "effort_limit_nm": SERVO_EFFORT_LIMIT_NM,
                "velocity_limit": None,
                "stiffness": SERVO_STIFFNESS,
                "damping": SERVO_DAMPING,
                "armature": SERVO_ARMATURE,
            },
            "wheel": {
                "effort_limit": None,
                "velocity_limit_rad_s": WHEEL_VELOCITY_LIMIT_RAD_S,
                "stiffness": WHEEL_STIFFNESS,
                "damping": WHEEL_DAMPING,
                "armature": WHEEL_ARMATURE,
            },
        },
        "ground": {
            "prim_path": GROUND_PRIM_PATH,
            "size_m": GROUND_SIZE_M,
            "translation_m": GROUND_TRANSLATION_M,
            "static_friction": GROUND_STATIC_FRICTION,
            "dynamic_friction": GROUND_DYNAMIC_FRICTION,
            "restitution": GROUND_RESTITUTION,
            "friction_combine_mode": "max",
            "restitution_combine_mode": "min",
        },
        "obstacle": {
            "prim_path": OBSTACLE_PRIM_PATH,
            "front_face_x_m": OBSTACLE_FRONT_FACE_X_M,
            "center_m": OBSTACLE_CENTER_M,
            "size_m": (OBSTACLE_LENGTH_M, OBSTACLE_WIDTH_M, OBSTACLE_HEIGHT_M),
            "kinematic": True,
            "gravity_disabled": True,
            "contact_offset_m": OBSTACLE_CONTACT_OFFSET_M,
            "rest_offset_m": OBSTACLE_REST_OFFSET_M,
            "static_friction": OBSTACLE_STATIC_FRICTION,
            "dynamic_friction": OBSTACLE_DYNAMIC_FRICTION,
            "restitution": OBSTACLE_RESTITUTION,
        },
        "camera": {"eye_m": CAMERA_EYE_M, "target_m": CAMERA_TARGET_M},
        "contact_reporting": {"activate_contact_sensors": True, "observation_only": True},
    }


def validate_locked_scene(*, verify_asset_hash: bool = True) -> dict[str, Any]:
    """Validate invariants without importing or starting Isaac."""

    expected_center_x = OBSTACLE_FRONT_FACE_X_M + 0.5 * OBSTACLE_LENGTH_M
    if not math.isclose(expected_center_x, OBSTACLE_CENTER_M[0], rel_tol=0.0, abs_tol=1.0e-12):
        raise SceneContractError(
            f"obstacle front/length imply center x={expected_center_x!r}, not {OBSTACLE_CENTER_M[0]!r}"
        )
    if not math.isclose(OBSTACLE_CENTER_M[2], 0.5 * OBSTACLE_HEIGHT_M, rel_tol=0.0, abs_tol=1.0e-12):
        raise SceneContractError("obstacle center z does not place its bottom at ground z=0")
    if not math.isclose(PHYSICS_DT_S, 1.0 / 120.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise SceneContractError("physics time step is not exactly the locked 120 Hz value")
    result = locked_scene_snapshot()
    if verify_asset_hash:
        result["robot"]["verified_sha256"] = verify_robot_asset()
    return result


def verify_robot_asset(path: Path = ROBOT_USD_PATH) -> str:
    """Read and hash the external asset; never modifies it."""

    resolved = path.resolve()
    if resolved != ROBOT_USD_PATH.resolve():
        raise SceneContractError(f"robot USD path is not the locked asset: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"locked robot USD was not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != ROBOT_USD_SHA256:
        raise SceneContractError(
            f"robot USD SHA-256 mismatch: expected {ROBOT_USD_SHA256}, received {actual}"
        )
    return actual


def create_scene(
    *,
    simulation_app: Any | None = None,
    imports: Mapping[str, Any] | None = None,
    before_reset: Callable[[Any, Any], Any] | None = None,
) -> SceneHandle:
    """Create and reset the locked scene after the caller starts AppLauncher.

    ``imports`` is an injection seam for tests. In production it is omitted and
    Isaac modules are imported lazily. No save/export operation exists here.
    """

    validate_locked_scene(verify_asset_hash=True)
    modules = dict(imports) if imports is not None else _isaac_imports()
    sim_utils = modules["sim_utils"]
    simulation_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT_S,
        render_interval=RENDER_INTERVAL_PHYSICS_STEPS,
        device=DEVICE,
        gravity=GRAVITY_M_S2,
    )
    sim = modules["SimulationContext"](simulation_cfg)
    sim.set_camera_view(eye=list(CAMERA_EYE_M), target=list(CAMERA_TARGET_M))

    _spawn_ground(sim_utils)
    _spawn_lighting(sim_utils)
    robot = modules["Articulation"](_build_robot_cfg(modules))
    _spawn_obstacle(sim_utils)
    # Observation sensors must be instantiated after all prims exist but before
    # the one authoritative reset initializes their PhysX views.
    instrumentation = before_reset(sim, robot) if before_reset is not None else None

    sim.reset()
    robot.update(0.0)
    sim.set_camera_view(eye=list(CAMERA_EYE_M), target=list(CAMERA_TARGET_M))
    actual_dt = float(sim.get_physics_dt())
    if not math.isclose(actual_dt, PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise SceneContractError(f"live simulation dt mismatch: expected {PHYSICS_DT_S}, received {actual_dt}")
    return SceneHandle(
        sim=sim,
        robot=robot,
        simulation_app=simulation_app,
        instrumentation=instrumentation,
    )


def _spawn_ground(sim_utils: Any) -> None:
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=GROUND_STATIC_FRICTION,
        dynamic_friction=GROUND_DYNAMIC_FRICTION,
        restitution=GROUND_RESTITUTION,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    cfg = sim_utils.GroundPlaneCfg(
        size=GROUND_SIZE_M,
        color=(0.08, 0.09, 0.10),
        physics_material=material,
    )
    # The locked ground is at the spawner's zero-Z default. Omitting a
    # translation avoids any fallback USD transform authoring.
    cfg.func(GROUND_PRIM_PATH, cfg)


def _spawn_lighting(sim_utils: Any) -> None:
    dome = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.88, 0.95))
    dome.func("/World/Light/Dome", dome)
    key = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.96, 0.90), angle=0.35)
    key.func("/World/Light/Key", key, translation=(1.5, -2.0, 4.0))


def _spawn_obstacle(sim_utils: Any) -> None:
    cfg = sim_utils.CuboidCfg(
        size=(OBSTACLE_LENGTH_M, OBSTACLE_WIDTH_M, OBSTACLE_HEIGHT_M),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=OBSTACLE_CONTACT_OFFSET_M,
            rest_offset=OBSTACLE_REST_OFFSET_M,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=OBSTACLE_STATIC_FRICTION,
            dynamic_friction=OBSTACLE_DYNAMIC_FRICTION,
            restitution=OBSTACLE_RESTITUTION,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.47, 0.33)),
        semantic_tags=[("class", "height_obstacle")],
    )
    cfg.func(OBSTACLE_PRIM_PATH, cfg, translation=OBSTACLE_CENTER_M)


def _build_robot_cfg(modules: Mapping[str, Any]) -> Any:
    sim_utils = modules["sim_utils"]
    articulation_cfg = modules["ArticulationCfg"]
    actuator_cfg = modules["ImplicitActuatorCfg"]
    # Contact reporting is enabled solely for clean observation sensors. It
    # does not alter materials, rigid-body properties, gravity, or actuators.
    return articulation_cfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH.as_posix(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY_M_S,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
        ),
        init_state=articulation_cfg.InitialStateCfg(
            pos=ROBOT_SPAWN_POSITION_M,
            rot=ROBOT_SPAWN_ROTATION_WXYZ,
            joint_pos={},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip_knee_position_servos": actuator_cfg(
                joint_names_expr=["|".join(SERVO_ORDER)],
                effort_limit_sim=SERVO_EFFORT_LIMIT_NM,
                velocity_limit_sim=None,
                stiffness=SERVO_STIFFNESS,
                damping=SERVO_DAMPING,
                armature=SERVO_ARMATURE,
            ),
            "wheel_velocity_motors": actuator_cfg(
                joint_names_expr=["|".join(WHEEL_ORDER)],
                effort_limit_sim=None,
                velocity_limit_sim=WHEEL_VELOCITY_LIMIT_RAD_S,
                stiffness=WHEEL_STIFFNESS,
                damping=WHEEL_DAMPING,
                armature=WHEEL_ARMATURE,
            ),
        },
    )


def _isaac_imports() -> dict[str, Any]:
    # AppLauncher must already exist before this function is called.
    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore
    from isaaclab.sim import SimulationContext  # type: ignore

    return {
        "sim_utils": sim_utils,
        "SimulationContext": SimulationContext,
        "Articulation": Articulation,
        "ArticulationCfg": ArticulationCfg,
        "ImplicitActuatorCfg": ImplicitActuatorCfg,
    }
