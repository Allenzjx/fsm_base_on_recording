"""Diagnostic live proof for no-rewind phase-snapshot source replay.

The probe wraps production Isaac-facing dependencies so that a strict reset
rejection still records real PhysX contacts, source-replay drift, and clock
non-advancement.  Priming belongs to ``TRAINING_RESET_STATE_WRITE``.  Ordinary
phases do not advance the controller; source-proven P10 consumes authenticated
motion tick zero after the real prime while the episode clock remains at zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

from .phase_effective_entry import (
    EffectivePhaseEntryError,
    SOURCE_PROVEN_EXECUTE_RESTORE_MODE,
    _expected_replay_anchor_contract,
    _replay_evidence_failures,
    _source_proven_execute_bindings,
    _validate_controller_entry_guard,
    _validated_source_proven_execute_restore,
    phase_entry_time_s,
)


PROBE_SCHEMA = "wlr50_clean.phase_snapshot_live_probe.v3"
PROBE_FILENAME = "phase_snapshot_live_probe.json"
PROBE_PHASES = tuple(f"P{index:02d}" for index in range(2, 14))
ATTEMPTS_PER_PHASE = 2
_MISSING = object()
_PROBE_PROCESS_INSTANCE_ID = uuid.uuid4().hex

_CALIBRATION_EFFECTIVE_ENTRY_FIELDS = frozenset(
    {
        "schema",
        "artifact_role",
        "verified",
        "calibration_only",
        "phase",
        "source_tick",
        "physical_anchor_tick",
        "physical_anchor_time_s",
        "predecessor_verify_tick",
        "predecessor_verify_time_s",
        "controller_anchor_tick",
        "controller_anchor_time_s",
        "target_entry_tick",
        "target_entry_time_s",
        "source_replay_steps",
        "physical_to_predecessor_verify_replay_steps",
        "predecessor_verify_to_controller_replay_steps",
        "physical_to_controller_replay_steps",
        "controller_to_target_replay_steps",
        "hybrid_physical_controller_anchor",
        "replay_anchor_contract",
        "effective_entry_offset_s",
        "phase_snapshot_bundle_sha256",
        "source_snapshot_post_prime_diagnostic",
        "failures",
    }
)

_ACCEPTANCE_EFFECTIVE_ENTRY_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "effective_entry_semantics",
        "source_tick",
        "physical_anchor_tick",
        "physical_anchor_time_s",
        "predecessor_verify_tick",
        "predecessor_verify_time_s",
        "controller_anchor_tick",
        "controller_anchor_time_s",
        "target_entry_tick",
        "target_entry_time_s",
        "source_replay_steps",
        "physical_to_predecessor_verify_replay_steps",
        "predecessor_verify_to_controller_replay_steps",
        "physical_to_controller_replay_steps",
        "controller_to_target_replay_steps",
        "hybrid_physical_controller_anchor",
        "replay_anchor_contract",
        "effective_entry_offset_s",
        "contract_sha256",
        "entry_sha256",
        "fingerprint_max_ulp_distance",
        "fingerprint",
        "component_state_allowed_max_ulp_distance",
        "component_state_max_ulp_distance",
        "component_state_ulp_distance",
        "component_state",
        "component_state_binary64_hex",
        "component_state_sha256",
        "expected_component_state_sha256",
        "raw_contacts",
        "raw_contact_signature_sha256",
        "expected_raw_contact_signature_sha256",
        "failures",
        "verified",
    }
)


class PhaseSnapshotLiveProbeError(RuntimeError):
    """The diagnostic probe itself could not produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class _ReplayWindow:
    payload: Mapping[str, Any]
    source_tick: int
    predecessor_verify_tick: int | None
    predecessor_verify_time_s: float | None
    controller_anchor_tick: int | None
    controller_anchor_time_s: float | None
    target_entry_tick: int
    source_replay_steps: int
    control_ticks: tuple[int, ...]
    controller_restore_mode: str | None
    controller_restore_contract: Mapping[str, Any] | None
    restore_bindings: Mapping[str, Any]


def _validated_replay_window(
    snapshot: Mapping[str, Any], entry: Any, *, phase: str
) -> _ReplayWindow:
    """Cross-check payload replay metadata against its validated manifest row."""

    source_tick = snapshot.get("source_tick")
    replay_steps = snapshot.get("source_replay_steps")
    commands = snapshot.get("source_commands")
    if (
        type(source_tick) is not int
        or source_tick < 0
        or type(replay_steps) is not int
        or replay_steps <= 0
        or not isinstance(commands, list)
        or len(commands) != replay_steps
        or any(not isinstance(command, Mapping) for command in commands)
    ):
        raise PhaseSnapshotLiveProbeError(
            f"validated phase snapshot replay window is invalid for {phase}"
        )
    try:
        restore_contract = _validated_source_proven_execute_restore(snapshot, phase)
        restore_bindings = _source_proven_execute_bindings(snapshot, phase)
    except EffectivePhaseEntryError as exc:
        raise PhaseSnapshotLiveProbeError(
            f"validated controller-restore contract is invalid for {phase}"
        ) from exc
    restore_mode = (
        None
        if restore_contract is None
        else SOURCE_PROVEN_EXECUTE_RESTORE_MODE
    )
    entry_restore_mode = getattr(entry, "controller_restore_mode", None)
    entry_transition_hash = getattr(
        entry, "source_transition_row_canonical_sha256", None
    )
    if restore_contract is not None:
        if (
            entry_restore_mode != restore_mode
            or entry_transition_hash
            != restore_contract["source_transition_row_canonical_sha256"]
        ):
            raise PhaseSnapshotLiveProbeError(
                "validated P10 manifest restore binding differs from its payload"
            )
    elif entry_restore_mode is not None or entry_transition_hash is not None:
        raise PhaseSnapshotLiveProbeError(
            f"validated manifest has an unpaired controller restore for {phase}"
        )
    target_entry_tick = source_tick + replay_steps
    hybrid = "target_entry_tick" in snapshot
    payload_target = snapshot.get("target_entry_tick")
    entry_target = getattr(entry, "target_entry_tick", None)
    if (
        getattr(entry, "source_tick", None) != source_tick
        or getattr(entry, "source_replay_steps", None) != replay_steps
        or (
            payload_target is not None
            and (
                type(payload_target) is not int
                or payload_target != target_entry_tick
                or entry_target != target_entry_tick
            )
        )
        or (payload_target is None and entry_target is not None)
    ):
        raise PhaseSnapshotLiveProbeError(
            f"validated phase snapshot manifest replay binding differs for {phase}"
        )
    has_controller_anchor_tick = "controller_anchor_tick" in snapshot
    has_controller_anchor_time = "controller_anchor_time_s" in snapshot
    has_predecessor_verify_tick = "predecessor_verify_tick" in snapshot
    has_predecessor_verify_time = "predecessor_verify_time_s" in snapshot
    payload_controller_anchor_tick = snapshot.get("controller_anchor_tick")
    payload_controller_anchor_time = snapshot.get("controller_anchor_time_s")
    payload_predecessor_verify_tick = snapshot.get("predecessor_verify_tick")
    payload_predecessor_verify_time = snapshot.get("predecessor_verify_time_s")
    entry_controller_anchor_tick = getattr(entry, "controller_anchor_tick", None)
    entry_controller_anchor_time = getattr(entry, "controller_anchor_time_s", None)
    entry_predecessor_verify_tick = getattr(
        entry, "predecessor_verify_tick", None
    )
    entry_predecessor_verify_time = getattr(
        entry, "predecessor_verify_time_s", None
    )
    if hybrid:
        if phase != "P10":
            raise PhaseSnapshotLiveProbeError(
                f"only P10 may declare a hybrid replay window, not {phase}"
            )
        if (
            payload_target is None
            or not has_controller_anchor_tick
            or not has_controller_anchor_time
            or not has_predecessor_verify_tick
            or not has_predecessor_verify_time
            or type(payload_controller_anchor_tick) is not int
            or type(payload_predecessor_verify_tick) is not int
            or not (
                source_tick
                < payload_predecessor_verify_tick
                < payload_controller_anchor_tick
                < target_entry_tick
            )
            or isinstance(payload_controller_anchor_time, bool)
            or not isinstance(payload_controller_anchor_time, (int, float))
            or not math.isfinite(float(payload_controller_anchor_time))
            or float(payload_controller_anchor_time)
            != phase_entry_time_s(payload_controller_anchor_tick)
            or isinstance(payload_predecessor_verify_time, bool)
            or not isinstance(payload_predecessor_verify_time, (int, float))
            or not math.isfinite(float(payload_predecessor_verify_time))
            or float(payload_predecessor_verify_time)
            != phase_entry_time_s(payload_predecessor_verify_tick)
            or entry_controller_anchor_tick != payload_controller_anchor_tick
            or entry_controller_anchor_time != payload_controller_anchor_time
            or entry_predecessor_verify_tick
            != payload_predecessor_verify_tick
            or entry_predecessor_verify_time
            != payload_predecessor_verify_time
            or snapshot.get("fsm_state") != "P10"
            or snapshot.get("fsm_lifecycle") != "WAIT_ENTRY"
        ):
            raise PhaseSnapshotLiveProbeError(
                "validated phase snapshot hybrid-anchor binding differs "
                f"for {phase}"
            )
        predecessor_verify_tick: int | None = payload_predecessor_verify_tick
        predecessor_verify_time_s: float | None = float(
            payload_predecessor_verify_time
        )
        controller_anchor_tick: int | None = payload_controller_anchor_tick
        controller_anchor_time_s: float | None = float(
            payload_controller_anchor_time
        )
    else:
        if (
            payload_target is not None
            or entry_target is not None
            or replay_steps != 1
            or has_predecessor_verify_tick
            or has_predecessor_verify_time
            or has_controller_anchor_tick
            or has_controller_anchor_time
            or entry_predecessor_verify_tick is not None
            or entry_predecessor_verify_time is not None
            or entry_controller_anchor_tick is not None
            or entry_controller_anchor_time is not None
        ):
            raise PhaseSnapshotLiveProbeError(
                "validated non-hybrid phase snapshot unexpectedly declares "
                f"replay anchors for {phase}"
            )
        predecessor_verify_tick = None
        predecessor_verify_time_s = None
        controller_anchor_tick = None
        controller_anchor_time_s = None
    control_ticks = tuple(source_tick + index for index in range(replay_steps))
    if (
        tuple(command.get("control_physics_tick") for command in commands)
        != control_ticks
        or snapshot.get("source_command") != commands[0]
    ):
        raise PhaseSnapshotLiveProbeError(
            f"validated phase snapshot source-command sequence is invalid for {phase}"
        )
    if hybrid:
        assert predecessor_verify_tick is not None
        assert controller_anchor_tick is not None
        expected_contexts = tuple(
            ("P09", "EXECUTE_MOTION")
            if tick < predecessor_verify_tick
            else (
                ("P09", "VERIFY_RESULT")
                if tick < controller_anchor_tick
                else ("P10", "WAIT_ENTRY")
            )
            for tick in control_ticks
        )
        actual_contexts = tuple(
            (
                command.get("source_fsm_state"),
                command.get("source_fsm_lifecycle"),
            )
            for command in commands
        )
        if actual_contexts != expected_contexts:
            raise PhaseSnapshotLiveProbeError(
                "validated P10 three-segment replay contexts are invalid"
            )
    elif restore_contract is not None:
        command = commands[0]
        if (
            command.get("source_fsm_state") != "P10"
            or command.get("source_fsm_lifecycle") != "EXECUTE_MOTION"
        ):
            raise PhaseSnapshotLiveProbeError(
                "validated source-proven P10 replay context is not EXECUTE_MOTION"
            )
    return _ReplayWindow(
        payload=snapshot,
        source_tick=source_tick,
        predecessor_verify_tick=predecessor_verify_tick,
        predecessor_verify_time_s=predecessor_verify_time_s,
        controller_anchor_tick=controller_anchor_tick,
        controller_anchor_time_s=controller_anchor_time_s,
        target_entry_tick=target_entry_tick,
        source_replay_steps=replay_steps,
        control_ticks=control_ticks,
        controller_restore_mode=restore_mode,
        controller_restore_contract=restore_contract,
        restore_bindings=restore_bindings,
    )


def _selected_probe_phases(
    phases: Sequence[str] | None,
) -> tuple[str, ...]:
    if phases is None:
        return PROBE_PHASES
    if isinstance(phases, (str, bytes)) or not isinstance(phases, Sequence):
        raise PhaseSnapshotLiveProbeError(
            "phases must select exactly one phase from P02 through P13"
        )
    selected = tuple(phases)
    if (
        len(selected) != 1
        or type(selected[0]) is not str
        or selected[0] not in PROBE_PHASES
    ):
        raise PhaseSnapshotLiveProbeError(
            "phases must select exactly one phase from P02 through P13"
        )
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise PhaseSnapshotLiveProbeError(f"live observation omits {name!r}")


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _float_vector(value: Any, length: int) -> tuple[float, ...] | None:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != length or any(not math.isfinite(item) for item in result):
        return None
    return result


def _maximum_error(left: Any, right: Any, length: int) -> float | None:
    lhs = _float_vector(left, length)
    rhs = _float_vector(right, length)
    if lhs is None or rhs is None:
        return None
    return max(abs(a - b) for a, b in zip(lhs, rhs, strict=True))


def _maximum_present(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return max(float(value) for value in values if value is not None)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quaternion_error(left: Any, right: Any) -> float | None:
    lhs = _float_vector(left, 4)
    rhs = _float_vector(right, 4)
    if lhs is None or rhs is None:
        return None
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return None
    lhs = tuple(value / lhs_norm for value in lhs)
    rhs = tuple(value / rhs_norm for value in rhs)
    direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(lhs, rhs, strict=True)))
    negated = math.sqrt(sum((a + b) ** 2 for a, b in zip(lhs, rhs, strict=True)))
    return min(direct, negated)


def _pair_record(value: Any) -> dict[str, Any]:
    force = _float_vector(_member(value, "force_w_n", ()), 3)
    history = _member(value, "active_history", ())
    return {
        "active": bool(_member(value, "active", False)),
        "pair_verified": bool(_member(value, "pair_verified", False)),
        "normal_force_n": _finite_float(
            _member(value, "normal_force_n", 0.0)
        ),
        "force_w_n": None if force is None else list(force),
        "active_history": [bool(item) for item in history],
    }


def observation_diagnostics(
    observation: Any | None, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare one captured reset observation with production tolerances."""

    tolerances = {
        "root_position_m": 2.0e-4,
        "root_orientation": 2.0e-5,
        "root_linear_velocity_m_s": 2.0e-4,
        "root_angular_velocity_rad_s": 2.0e-4,
        "servo_position_deg": 0.02,
        "servo_velocity_deg_s": 0.02,
        "wheel_velocity_rad_s": 2.0e-4,
        "wheel_center_m": 0.002,
        "wheel_bottom_m": 0.002,
    }
    if observation is None:
        return {
            "observation_available": False,
            "physical_errors": {name: None for name in tolerances},
            "production_tolerances": tolerances,
            "physical_state_within_production_tolerances": False,
            "exact_contacts": {},
            "contact_mismatches": ["post-snapshot observation unavailable"],
            "exact_contacts_match": False,
        }

    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER

    base = _member(observation, "base")
    root = snapshot["root_state"]
    joints = _member(observation, "joints", {})
    wheels = _member(observation, "wheels", {})
    geometry = snapshot["obstacle_relative_geometry"]
    physical_errors: dict[str, float | None] = {
        "root_position_m": _maximum_error(
            _member(base, "position_w_m", ()), root["position_w_m"], 3
        ),
        "root_orientation": _quaternion_error(
            _member(base, "orientation_wxyz", ()), root["orientation_wxyz"]
        ),
        "root_linear_velocity_m_s": _maximum_error(
            _member(base, "linear_velocity_w_m_s", ()),
            root["linear_velocity_w_m_s"],
            3,
        ),
        "root_angular_velocity_rad_s": _maximum_error(
            _member(base, "angular_velocity_w_rad_s", ()),
            root["angular_velocity_w_rad_s"],
            3,
        ),
        "servo_position_deg": _maximum_error(
            tuple(_member(joints[name], "position_deg") for name in SERVO_ORDER),
            snapshot["joint_state"]["logical_position_deg"],
            len(SERVO_ORDER),
        ),
        "servo_velocity_deg_s": _maximum_error(
            tuple(_member(joints[name], "velocity_deg_s") for name in SERVO_ORDER),
            snapshot["joint_state"]["logical_velocity_deg_s"],
            len(SERVO_ORDER),
        ),
        "wheel_velocity_rad_s": _maximum_error(
            tuple(_member(wheels[name], "velocity_rad_s") for name in WHEEL_ORDER),
            snapshot["wheel_state"]["logical_velocity_rad_s"],
            len(WHEEL_ORDER),
        ),
        "wheel_center_m": _maximum_present(tuple(
            _maximum_error(
                _member(wheels[name], "center_w_m", ()),
                geometry["wheel_centers_w_m"][name],
                3,
            )
            for name in WHEEL_ORDER
        )),
        "wheel_bottom_m": _maximum_present(tuple(
            _maximum_error(
                _member(wheels[name], "bottom_w_m", ()),
                geometry["wheel_bottoms_w_m"][name],
                3,
            )
            for name in WHEEL_ORDER
        )),
    }
    physical_ok = all(
        error is not None and error <= tolerances[name]
        for name, error in physical_errors.items()
    )

    contacts = _member(observation, "contacts", {})
    exact_contacts: dict[str, Any] = {}
    mismatches: list[str] = []
    for wheel_name in WHEEL_ORDER:
        wheel = wheels[wheel_name]
        body_name = str(_member(wheel, "body_name"))
        actual = contacts.get(body_name) if isinstance(contacts, Mapping) else None
        expected = snapshot["contact_state"][wheel_name]
        if actual is None:
            mismatches.append(f"{wheel_name}: contact body {body_name!r} unavailable")
            exact_contacts[wheel_name] = {
                "body_name": body_name,
                "expected": dict(expected),
                "actual": None,
                "matches": False,
            }
            continue
        ground = _member(actual, "ground")
        obstacle = _member(actual, "obstacle")
        actual_class = _enum_value(_member(actual, "contact_class"))
        actual_ground = bool(_member(ground, "active", False))
        actual_obstacle = bool(_member(obstacle, "active", False))
        matches = bool(
            actual_class == str(expected["class"])
            and actual_ground == bool(expected["ground_active"])
            and actual_obstacle == bool(expected["obstacle_active"])
        )
        if not matches:
            mismatches.append(
                f"{wheel_name}: expected {expected['class']} "
                f"g={bool(expected['ground_active'])} "
                f"o={bool(expected['obstacle_active'])}, received "
                f"{actual_class} g={actual_ground} o={actual_obstacle}"
            )
        exact_contacts[wheel_name] = {
            "body_name": body_name,
            "expected": {
                "class": str(expected["class"]),
                "ground_active": bool(expected["ground_active"]),
                "obstacle_active": bool(expected["obstacle_active"]),
            },
            "actual": {
                "class": actual_class,
                "ground": _pair_record(ground),
                "obstacle": _pair_record(obstacle),
            },
            "matches": matches,
        }
    return {
        "observation_available": True,
        "observation_physics_tick": int(_member(observation, "physics_tick", -1)),
        "observation_simulation_time_s": _finite_float(
            _member(observation, "simulation_time_s", math.nan)
        ),
        "physical_errors": physical_errors,
        "production_tolerances": tolerances,
        "physical_state_within_production_tolerances": physical_ok,
        "exact_contacts": exact_contacts,
        "contact_mismatches": mismatches,
        "exact_contacts_match": not mismatches,
    }


@dataclass
class _AttemptCapture:
    phase: str
    attempt_kind: str
    scene_existed_before: bool
    physics_steps: int = 0
    simulation_forwards: int = 0
    simulation_resets: int = 0
    simulation_stops: int = 0
    snapshot_write_finished: bool = False
    reset_writes: Mapping[str, Any] | None = None
    snapshot_state_write: Mapping[str, Any] | None = None
    post_snapshot_observations: list[Any] = field(default_factory=list)
    controllers: list[Any] = field(default_factory=list)


class _CaptureBook:
    def __init__(self) -> None:
        self.current: _AttemptCapture | None = None
        self.scene_creation_count = 0

    def increment(self, name: str) -> None:
        if self.current is not None:
            setattr(self.current, name, int(getattr(self.current, name)) + 1)


class _CountingSimulation:
    def __init__(self, target: Any, book: _CaptureBook) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_book", book)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        object.__getattribute__(self, "_book").increment("physics_steps")
        return object.__getattribute__(self, "_target").step(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        object.__getattribute__(self, "_book").increment("simulation_forwards")
        return object.__getattribute__(self, "_target").forward(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        object.__getattribute__(self, "_book").increment("simulation_resets")
        return object.__getattribute__(self, "_target").reset(*args, **kwargs)

    def stop(self, *args: Any, **kwargs: Any) -> Any:
        object.__getattribute__(self, "_book").increment("simulation_stops")
        return object.__getattribute__(self, "_target").stop(*args, **kwargs)


class _CapturingReader:
    def __init__(self, target: Any, book: _CaptureBook, post_snapshot: bool) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_book", book)
        object.__setattr__(self, "_post_snapshot", post_snapshot)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        observation = object.__getattribute__(self, "_target").read(
            *args, **kwargs
        )
        book = object.__getattribute__(self, "_book")
        if object.__getattribute__(self, "_post_snapshot") and book.current is not None:
            book.current.post_snapshot_observations.append(observation)
        return observation


def _instrumented_dependencies(book: _CaptureBook) -> Any:
    """Wrap only diagnostic seams; all physical behavior remains production."""

    from .isaac_fsm_backend import _load_live_dependencies

    base = _load_live_dependencies()

    def create_scene(*args: Any, **kwargs: Any) -> Any:
        scene = base.create_scene(*args, **kwargs)
        book.scene_creation_count += 1
        return replace(scene, sim=_CountingSimulation(scene.sim, book))

    def reset_scene(scene: Any, canonical: Any) -> Mapping[str, Any]:
        result = dict(base.reset_scene(scene, canonical))
        if book.current is not None:
            book.current.reset_writes = result
        return result

    def write_phase_snapshot(
        scene: Any,
        adapter: Any,
        snapshot: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        result = dict(
            base.write_phase_snapshot(scene, adapter, snapshot, **kwargs)
        )
        if book.current is not None:
            book.current.snapshot_state_write = result
            book.current.snapshot_write_finished = True
        return result

    def reader_from_scene(scene: Any, adapter: Any, backends: Any) -> Any:
        reader = base.reader_from_scene(scene, adapter, backends)
        post_snapshot = bool(
            book.current is not None and book.current.snapshot_write_finished
        )
        return _CapturingReader(reader, book, post_snapshot)

    def controller_from_paths(*args: Any, **kwargs: Any) -> Any:
        controller = base.controller_from_paths(*args, **kwargs)
        if book.current is not None:
            book.current.controllers.append(controller)
        return controller

    return replace(
        base,
        create_scene=create_scene,
        reset_scene=reset_scene,
        write_phase_snapshot=write_phase_snapshot,
        reader_from_scene=reader_from_scene,
        controller_from_paths=controller_from_paths,
    )


def _controller_clocks(controller: Any | None, frame: Any | None) -> dict[str, Any]:
    state = None if controller is None else getattr(controller, "state", None)
    return {
        "controller_constructed": controller is not None,
        "controller_state_id": None
        if state is None
        else str(getattr(state, "state_id", "")),
        "controller_internal_physics_tick": None
        if controller is None
        else int(getattr(controller, "physics_tick", -1)),
        "controller_last_simulation_time_s": None
        if controller is None
        else _finite_float(getattr(controller, "_last_sim_time_s", None)),
        "controller_history_length": None
        if controller is None
        else len(tuple(getattr(controller, "history", ()))),
        "controller_frame_committed": frame is not None,
        "controller_frame_state_id": None
        if frame is None
        else str(getattr(frame, "state_id", "")),
        "controller_frame_physics_tick": None
        if frame is None
        else int(getattr(frame, "physics_tick", -1)),
    }


def _attempt_passed(
    row: Mapping[str, Any],
    *,
    replay_window: _ReplayWindow,
    calibration_mode: bool = False,
) -> bool:
    diagnostic = row.get("observation_diagnostics")
    clocks = row.get("clocks")
    snapshot_write = row.get("snapshot_state_write")
    if not all(isinstance(value, Mapping) for value in (diagnostic, clocks, snapshot_write)):
        return False
    prime_steps = replay_window.source_replay_steps
    if (
        row.get("source_replay_steps") != prime_steps
        or "predecessor_verify_tick" not in row
        or row.get("predecessor_verify_tick")
        != replay_window.predecessor_verify_tick
        or "predecessor_verify_time_s" not in row
        or row.get("predecessor_verify_time_s")
        != replay_window.predecessor_verify_time_s
        or "controller_anchor_tick" not in row
        or row.get("controller_anchor_tick")
        != replay_window.controller_anchor_tick
        or "controller_anchor_time_s" not in row
        or row.get("controller_anchor_time_s")
        != replay_window.controller_anchor_time_s
        or row.get("target_entry_tick") != replay_window.target_entry_tick
        or row.get("episode_sensor_tick_offset")
        != replay_window.target_entry_tick
        or row.get("effective_entry_offset_s") != phase_entry_time_s(prime_steps)
        or row.get("source_control_physics_ticks")
        != list(replay_window.control_ticks)
        or row.get("extra_physics_priming_steps") != prime_steps
        or any(
            row.get(name) != expected
            for name, expected in replay_window.restore_bindings.items()
        )
    ):
        return False
    source_commands = replay_window.payload["source_commands"]
    expected_anchor_contract = _expected_replay_anchor_contract(
        replay_window.payload,
        str(row.get("phase")),
        replay_steps=prime_steps,
        target_entry_tick=replay_window.target_entry_tick,
        control_ticks=replay_window.control_ticks,
        predecessor_verify_tick=replay_window.predecessor_verify_tick,
        predecessor_verify_time_s=replay_window.predecessor_verify_time_s,
        controller_anchor_tick=replay_window.controller_anchor_tick,
        controller_anchor_time_s=replay_window.controller_anchor_time_s,
    )
    if (
        row.get("source_command_row_canonical_sha256s")
        != [command["source_command_row_canonical_sha256"] for command in source_commands]
        or row.get("source_observation_row_canonical_sha256s")
        != [
            command["source_observation_row_canonical_sha256"]
            for command in source_commands
        ]
        or row.get("source_adapter_input_sha256s")
        != [command["source_adapter_input_sha256"] for command in source_commands]
        or row.get("source_drive_target_full12_sha256s")
        != [command["drive_target_full12_sha256"] for command in source_commands]
        or row.get("source_actuation_contract_sha256s")
        != [command["actuation_contract_sha256"] for command in source_commands]
    ):
        return False
    source_match = snapshot_write.get("source_actuation_match")
    if not isinstance(source_match, Mapping):
        return False
    pre_prime_root = snapshot_write.get("pre_prime_root_link_readback")
    if not isinstance(pre_prime_root, Mapping):
        return False
    effective_entry = snapshot_write.get("effective_entry_contract")
    entry_safety = snapshot_write.get("entry_safety_contract")
    entry_guards = snapshot_write.get("entry_guard_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (effective_entry, entry_safety, entry_guards)
    ):
        return False
    expected_effective_entry_fields = (
        _CALIBRATION_EFFECTIVE_ENTRY_FIELDS
        if calibration_mode
        else _ACCEPTANCE_EFFECTIVE_ENTRY_FIELDS
    ) | frozenset(replay_window.restore_bindings)
    if set(effective_entry) != expected_effective_entry_fields:
        return False
    if _replay_evidence_failures(
        snapshot_write,
        replay_window.payload,
        replay_steps=prime_steps,
        target_entry_tick=replay_window.target_entry_tick,
        control_ticks=replay_window.control_ticks,
        predecessor_verify_tick=replay_window.predecessor_verify_tick,
        predecessor_verify_time_s=replay_window.predecessor_verify_time_s,
        controller_anchor_tick=replay_window.controller_anchor_tick,
        controller_anchor_time_s=replay_window.controller_anchor_time_s,
    ):
        return False
    safety_flags = entry_safety.get("flags")
    if not isinstance(safety_flags, Mapping):
        return False
    if replay_window.controller_restore_contract is None:
        # Preserve the long-standing diagnostic gate for ordinary phases.  Its
        # full authored-guard validation is owned by calibration import.
        controller_entry_proof_ok = True
    else:
        try:
            _validate_controller_entry_guard(
                entry_guards,
                phase=str(row.get("phase")),
                snapshot_payload=replay_window.payload,
            )
        except EffectivePhaseEntryError:
            controller_entry_proof_ok = False
        else:
            controller_entry_proof_ok = True
    p10_alignment_ok = True
    if row.get("phase") == "P10":
        alignment = entry_guards.get("p10_signed_velocity_alignment")
        actual_velocity = (
            alignment.get("actual_deg_s")
            if isinstance(alignment, Mapping)
            else None
        )
        p10_alignment_ok = bool(
            isinstance(alignment, Mapping)
            and alignment.get("signed_positive_rebound_required") is True
            and isinstance(actual_velocity, (int, float))
            and not isinstance(actual_velocity, bool)
            and math.isfinite(float(actual_velocity))
            and float(actual_velocity) > 0.0
        )
    anchor_proof_ok = bool(
        effective_entry.get("physical_anchor_tick")
        == expected_anchor_contract["physical_anchor_tick"]
        and effective_entry.get("physical_anchor_time_s")
        == expected_anchor_contract["physical_anchor_time_s"]
        and "predecessor_verify_tick" in effective_entry
        and effective_entry.get("predecessor_verify_tick")
        == replay_window.predecessor_verify_tick
        and "predecessor_verify_time_s" in effective_entry
        and effective_entry.get("predecessor_verify_time_s")
        == replay_window.predecessor_verify_time_s
        and "controller_anchor_tick" in effective_entry
        and effective_entry.get("controller_anchor_tick")
        == replay_window.controller_anchor_tick
        and "controller_anchor_time_s" in effective_entry
        and effective_entry.get("controller_anchor_time_s")
        == replay_window.controller_anchor_time_s
        and effective_entry.get("target_entry_tick")
        == replay_window.target_entry_tick
        and effective_entry.get("target_entry_time_s")
        == expected_anchor_contract["target_entry_time_s"]
        and effective_entry.get("source_replay_steps") == prime_steps
        and effective_entry.get(
            "physical_to_predecessor_verify_replay_steps"
        )
        == expected_anchor_contract[
            "physical_to_predecessor_verify_replay_steps"
        ]
        and effective_entry.get(
            "predecessor_verify_to_controller_replay_steps"
        )
        == expected_anchor_contract[
            "predecessor_verify_to_controller_replay_steps"
        ]
        and effective_entry.get("physical_to_controller_replay_steps")
        == expected_anchor_contract["physical_to_controller_replay_steps"]
        and effective_entry.get("controller_to_target_replay_steps")
        == expected_anchor_contract["controller_to_target_replay_steps"]
        and effective_entry.get("hybrid_physical_controller_anchor")
        == expected_anchor_contract["hybrid_physical_controller_anchor"]
        and effective_entry.get("replay_anchor_contract")
        == expected_anchor_contract
    )
    if calibration_mode:
        diagnostic_proof = effective_entry.get(
            "source_snapshot_post_prime_diagnostic"
        )
        effective_entry_ok = bool(
            effective_entry.get("schema")
            == "wlr50_clean.ppo_phase_effective_entry_calibration_live_proof.v2"
            and effective_entry.get("artifact_role")
            == "CALIBRATION_ONLY_NOT_TRAINING_ACCEPTANCE"
            and effective_entry.get("verified") is True
            and effective_entry.get("calibration_only") is True
            and effective_entry.get("phase") == row.get("phase")
            and effective_entry.get("source_tick") == replay_window.source_tick
            and anchor_proof_ok
            and effective_entry.get("effective_entry_offset_s")
            == phase_entry_time_s(prime_steps)
            and all(
                effective_entry.get(name) == expected
                for name, expected in replay_window.restore_bindings.items()
            )
            and isinstance(diagnostic_proof, Mapping)
            and diagnostic_proof.get("schema")
            == "wlr50_clean.phase_snapshot_live_comparison.v2"
            and not effective_entry.get("failures")
        )
    else:
        effective_entry_ok = bool(
            effective_entry.get("schema")
            == "wlr50_clean.ppo_phase_effective_entry_live_proof.v2"
            and effective_entry.get("phase") == row.get("phase")
            and effective_entry.get("effective_entry_semantics")
            == "source_snapshot_plus_validated_replay_steps_no_rewind"
            and effective_entry.get("source_tick") == replay_window.source_tick
            and anchor_proof_ok
            and effective_entry.get("effective_entry_offset_s")
            == phase_entry_time_s(prime_steps)
            and all(
                effective_entry.get(name) == expected
                for name, expected in replay_window.restore_bindings.items()
            )
            and effective_entry.get("verified") is True
            and not effective_entry.get("failures")
        )
    return bool(
        row.get("reset_completed") is True
        # The source-t comparison is diagnostic only.  Production acceptance
        # is the separately calibrated snapshot-plus-one-PhysX-tick contract.
        and diagnostic.get("observation_available") is True
        and diagnostic.get("observation_physics_tick")
        == replay_window.target_entry_tick
        and diagnostic.get("observation_simulation_time_s")
        == phase_entry_time_s(replay_window.target_entry_tick)
        and clocks.get("backend_episode_tick") == 0
        and clocks.get("controller_frame_state_id") == row.get("phase")
        and clocks.get("controller_frame_physics_tick") == 0
        and snapshot_write.get("root_pose_writes") == 1
        and snapshot_write.get("root_velocity_writes") == 1
        and snapshot_write.get("joint_state_writes") == 1
        and snapshot_write.get("simulation_forward_syncs") == 1
        and snapshot_write.get("pre_prime_state_verified") is True
        and snapshot_write.get("pre_prime_joint_state_verified") is True
        and pre_prime_root.get("verified") is True
        and pre_prime_root.get("all_values_finite") is True
        and pre_prime_root.get("all_fields_within_production_tolerances") is True
        and pre_prime_root.get("physics_steps_before_readback") == 0
        and pre_prime_root.get("contact_sensor_reads_before_readback") == 0
        and snapshot_write.get("physics_steps") == prime_steps
        and snapshot_write.get("state_write_count") == 1
        and snapshot_write.get("post_prime_state_rewrite_performed") is False
        and snapshot_write.get("contact_and_state_share_solver_tick") is True
        and snapshot_write.get("prime_physics_steps") == prime_steps
        and snapshot_write.get("effective_entry_offset_s")
        == phase_entry_time_s(prime_steps)
        and snapshot_write.get("prime_atomic_full12_writes") == prime_steps
        and snapshot_write.get("logical_target_fallback_used") is False
        and snapshot_write.get("current_contact_force_provenance")
        == "current_final_solver_force_only"
        and snapshot_write.get("sensor_history_samples_after_reset") == prime_steps
        and snapshot_write.get("contact_sensor_reads_after_prime") == prime_steps
        and snapshot_write.get("source_replay_guard_updates_applied") == prime_steps
        # Phase snapshots intentionally restore only physical/controller
        # latches.  The contact classifier is cold-started before the real
        # post-write replay sequence and is then checked by the calibrated
        # effective-entry proof below.  Requiring the old source-snapshot
        # hysteresis result here both referenced a field no longer emitted by
        # the backend and contradicted that cold-start contract.
        and snapshot_write.get("classifier_cold_started_before_source_replay")
        is True
        and snapshot_write.get("classifier_source_history_restored") is False
        and snapshot_write.get("classifier_source_state_restored") is False
        and snapshot_write.get("classifier_history_equivalence_claimed") is False
        and snapshot_write.get("raw_sensor_history_rewarmed_from_prime") is True
        and snapshot_write.get("contact_backend_reset") is True
        and snapshot_write.get("contact_backend_reset_after_prime") is False
        and effective_entry_ok
        and entry_safety.get("schema")
        == "wlr50_clean.phase_effective_entry_safety.v1"
        and entry_safety.get("verified") is True
        and entry_safety.get("all_failure_flags_false") is True
        and not any(bool(value) for value in safety_flags.values())
        and entry_guards.get("schema")
        == (
            "wlr50_clean.phase_effective_entry_controller.v2"
            if replay_window.controller_restore_contract is not None
            else "wlr50_clean.phase_effective_entry_controller.v1"
        )
        and entry_guards.get("verified") is True
        and entry_guards.get("phase") == row.get("phase")
        and entry_guards.get("lifecycle") == "EXECUTE_MOTION"
        and entry_guards.get("nonterminal") is True
        and entry_guards.get("unblocked") is True
        and controller_entry_proof_ok
        and p10_alignment_ok
        and snapshot_write.get("fsm_clock_steps_during_priming") == 0
        and snapshot_write.get("episode_clock_steps_during_priming") == 0
        and row.get("physics_steps_during_reset") == 180 + prime_steps
        and row.get("post_prime_contact_sensor_read_count") == prime_steps
    )


def _is_ordinary_reset_rejection(
    exc: BaseException,
    *,
    backend: Any,
    capture: _AttemptCapture,
) -> bool:
    """Return true only for an observable post-write restoration mismatch.

    The production backend deliberately shares one public error hierarchy for
    restoration mismatches and infrastructure failures.  A finalized FAILED
    diagnostic is allowed only for its exact phase-restoration sensor failure,
    after the snapshot write and a post-write observation.  Critical sensing,
    controller/runtime, and immutable-bundle failures must escape instead.
    """

    from .isaac_fsm_backend import SensorContractFailure

    ordinary_prefixes = (
        "phase snapshot live restoration could not be proven: ",
        "phase effective-entry contract failed: ",
        "effective phase entry safety gate failed: ",
        "effective phase entry controller/guard gate failed: ",
    )
    return bool(
        type(exc) is SensorContractFailure
        and str(exc).startswith(ordinary_prefixes)
        and not bool(getattr(backend, "_phase_snapshot_integrity_failed", False))
        and capture.snapshot_write_finished
        and bool(capture.post_snapshot_observations)
    )


def _seal_fatal_probe_failure(
    output: Path,
    payload: dict[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    exc: BaseException,
) -> None:
    """Best-effort seal of diagnostic context before a fatal failure escapes."""

    payload.update(
        {
            "status": "FAILED",
            "passed": False,
            "complete": False,
            "completed_attempt_count": len(attempts),
            "fresh_scene_attempt_count": sum(
                row.get("scene_lifecycle") == "fresh_scene" for row in attempts
            ),
            "reused_scene_attempt_count": sum(
                row.get("scene_lifecycle") == "reused_scene" for row in attempts
            ),
            "failure_classification": "FATAL_INTEGRITY_OR_INFRASTRUCTURE",
            "failure_reasons": [
                f"probe infrastructure/integrity failed: {type(exc).__name__}: {exc}"
            ],
        }
    )
    try:
        _write_probe(output, payload)
    except Exception as write_exc:
        raise PhaseSnapshotLiveProbeError(
            "fatal phase-snapshot probe failure could not be sealed: "
            f"{type(write_exc).__name__}: {write_exc}"
        ) from exc


def _evidence_reference(path: Path, *, expected_schema: str) -> dict[str, Any]:
    if not path.is_file():
        raise PhaseSnapshotLiveProbeError(f"managed evidence is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseSnapshotLiveProbeError(
            f"managed evidence is invalid JSON: {path}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != expected_schema:
        raise PhaseSnapshotLiveProbeError(
            f"managed evidence has the wrong schema: {path}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema": expected_schema,
    }


def _write_probe(path: Path, payload: Mapping[str, Any]) -> None:
    from .artifacts import atomic_write_json

    atomic_write_json(path, payload, replace=path.exists())


def run_phase_snapshot_live_probe(
    simulation_app: Any,
    *,
    run_dir: Path,
    seed: int,
    snapshot_bundle: Any,
    effective_entry_contract: Any | None = None,
    calibration_mode: bool = False,
    prime_physics_steps: int = 1,
    phases: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Run two resets for all P02-P13 or one explicitly selected phase."""

    from .isaac_fsm_backend import IsaacFSMBackend
    from .phase_snapshots import (
        assert_phase_snapshot_bundle_unchanged,
        load_validated_phase_snapshot_payload,
    )
    from .phase_effective_entry import (
        assert_effective_phase_entry_contract_unchanged,
    )

    if (
        isinstance(prime_physics_steps, bool)
        or not isinstance(prime_physics_steps, int)
        or prime_physics_steps != 1
    ):
        raise PhaseSnapshotLiveProbeError(
            "prime_physics_steps is a fixed legacy ABI sentinel; replay length "
            "is derived from each validated snapshot"
        )
    selected_phases = _selected_probe_phases(phases)
    if type(calibration_mode) is not bool:
        raise PhaseSnapshotLiveProbeError(
            "calibration_mode must be an explicit boolean"
        )
    if not isinstance(getattr(snapshot_bundle, "bundle_sha256", None), str):
        raise PhaseSnapshotLiveProbeError(
            "phase-snapshot probe requires a pinned snapshot bundle"
        )
    if calibration_mode:
        if effective_entry_contract is not None:
            raise PhaseSnapshotLiveProbeError(
                "calibration mode cannot consume an effective-entry contract"
            )
        if phases is None or len(selected_phases) != 1:
            raise PhaseSnapshotLiveProbeError(
                "calibration mode requires exactly one explicit phase"
            )
        if int(seed) != 1002:
            raise PhaseSnapshotLiveProbeError(
                "effective-entry calibration requires locked seed 1002"
            )
    elif (
        effective_entry_contract is None
        or not callable(getattr(effective_entry_contract, "as_record", None))
        or not isinstance(
            getattr(effective_entry_contract, "phase_snapshot_bundle_sha256", None),
            str,
        )
    ):
        raise PhaseSnapshotLiveProbeError(
            "phase-snapshot probe requires pinned snapshot/effective-entry contracts"
        )
    if (
        not calibration_mode
        and effective_entry_contract is not None
        and effective_entry_contract.phase_snapshot_bundle_sha256
        != snapshot_bundle.bundle_sha256
    ):
        raise PhaseSnapshotLiveProbeError(
            "effective-entry contract belongs to a different snapshot bundle"
        )
    loaded_phases: dict[str, tuple[Mapping[str, Any], Any, _ReplayWindow]] = {}
    for phase in selected_phases:
        snapshot, entry = load_validated_phase_snapshot_payload(
            snapshot_bundle, phase
        )
        loaded_phases[phase] = (
            snapshot,
            entry,
            _validated_replay_window(snapshot, entry, phase=phase),
        )
    expected_attempt_count = len(selected_phases) * ATTEMPTS_PER_PHASE
    expected_fresh_count = 1
    expected_reused_count = expected_attempt_count - expected_fresh_count
    output = Path(run_dir).resolve() / PROBE_FILENAME
    runtime_before = _evidence_reference(
        Path(run_dir) / "committed_runtime_identity.before.json",
        expected_schema="wlr50_clean.committed_runtime_identity.v1",
    )
    frozen_before_path = Path(run_dir) / "frozen_hashes.before.json"
    if not frozen_before_path.is_file():
        raise PhaseSnapshotLiveProbeError(
            f"managed frozen pre-check is missing: {frozen_before_path}"
        )
    try:
        frozen_payload = json.loads(frozen_before_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseSnapshotLiveProbeError(
            "managed frozen pre-check is not valid JSON"
        ) from exc
    if (
        not isinstance(frozen_payload, Mapping)
        or frozen_payload.get("schema")
        != "wlr50_clean.frozen_fsm_hash_audit.v1"
        or frozen_payload.get("passed") is not True
        or frozen_payload.get("mismatches") != []
    ):
        raise PhaseSnapshotLiveProbeError(
            "managed frozen pre-check is incomplete or failed"
        )
    frozen_before = {
        "path": str(frozen_before_path.resolve()),
        "sha256": _sha256(frozen_before_path),
        "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
        "passed": True,
    }
    payload: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "artifact_role": (
            "CALIBRATION_ONLY_NOT_TRAINING_ACCEPTANCE"
            if calibration_mode
            else "DIAGNOSTIC_ONLY_NOT_TRAINING_ACCEPTANCE"
        ),
        "calibration_mode": calibration_mode,
        "probe_process_id": os.getpid(),
        "probe_process_instance_id": _PROBE_PROCESS_INSTANCE_ID,
        "status": "RUNNING",
        "passed": False,
        "seed": int(seed),
        "phases": list(selected_phases),
        "phase_count": len(selected_phases),
        "phase_selector_mode": (
            "all_non_p01_phases" if phases is None else "single_phase"
        ),
        "attempts_per_phase": ATTEMPTS_PER_PHASE,
        "expected_attempt_count": expected_attempt_count,
        "expected_fresh_scene_attempt_count": expected_fresh_count,
        "expected_reused_scene_attempt_count": expected_reused_count,
        "completed_attempt_count": 0,
        "production_reset_modified": True,
        "production_reset_mode": (
            "validated_source_command_sequence_replay_without_rewind"
        ),
        "source_replay_policy": "derived_only_from_validated_phase_snapshot",
        "source_replay_steps_by_phase": {
            phase: loaded_phases[phase][2].source_replay_steps
            for phase in selected_phases
        },
        "controller_restore_modes_by_phase": {
            phase: loaded_phases[phase][2].controller_restore_mode
            for phase in selected_phases
        },
        "source_transition_hashes_by_phase": {
            phase: loaded_phases[phase][2].restore_bindings.get(
                "source_transition_row_canonical_sha256"
            )
            for phase in selected_phases
        },
        "controller_anchors_by_phase": {
            phase: (
                None
                if loaded_phases[phase][2].controller_anchor_tick is None
                else {
                    "controller_anchor_tick": loaded_phases[phase][
                        2
                    ].controller_anchor_tick,
                    "controller_anchor_time_s": loaded_phases[phase][
                        2
                    ].controller_anchor_time_s,
                }
            )
            for phase in selected_phases
        },
        "predecessor_verify_anchors_by_phase": {
            phase: (
                None
                if loaded_phases[phase][2].predecessor_verify_tick is None
                else {
                    "predecessor_verify_tick": loaded_phases[phase][
                        2
                    ].predecessor_verify_tick,
                    "predecessor_verify_time_s": loaded_phases[phase][
                        2
                    ].predecessor_verify_time_s,
                }
            )
            for phase in selected_phases
        },
        "runtime_identity_before": runtime_before,
        "frozen_hashes_before": frozen_before,
        "managed_post_checks": {
            "runtime_identity_after": str(
                (Path(run_dir) / "committed_runtime_identity.after.json").resolve()
            ),
            "frozen_hashes_after": str(
                (Path(run_dir) / "frozen_hashes.after.json").resolve()
            ),
            "sealed_by_run_manifest": str(
                (Path(run_dir) / "run_manifest.json").resolve()
            ),
        },
        "phase_snapshot_bundle": snapshot_bundle.as_record(),
        "phase_effective_entry_contract": (
            None
            if effective_entry_contract is None
            else effective_entry_contract.as_record()
        ),
        "attempts": [],
        "failure_reasons": [],
    }
    _write_probe(output, payload)

    book = _CaptureBook()
    attempts: list[dict[str, Any]] = []
    try:
        dependencies = _instrumented_dependencies(book)
        backend = IsaacFSMBackend(
            simulation_app,
            dependencies=dependencies,
            expected_phase_snapshot_bundle=snapshot_bundle,
            expected_effective_entry_contract=effective_entry_contract,
            allow_effective_entry_calibration=calibration_mode,
        )
        for phase in selected_phases:
            snapshot, entry, replay_window = loaded_phases[phase]
            source_commands = snapshot["source_commands"]
            for repeat in range(ATTEMPTS_PER_PHASE):
                scene_existed = backend._scene is not None
                capture = _AttemptCapture(
                    phase=phase,
                    attempt_kind="primary" if repeat == 0 else "reused_repeat",
                    scene_existed_before=scene_existed,
                )
                book.current = capture
                exception: dict[str, str] | None = None
                caught_exception: BaseException | None = None
                frame = None
                try:
                    frame = backend.reset(
                        seed=int(seed),
                        options={"training_phase_snapshot": phase},
                    )
                except Exception as exc:  # evidence must survive a rejected phase
                    caught_exception = exc
                    exception = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                finally:
                    book.current = None

                observation = (
                    capture.post_snapshot_observations[-1]
                    if capture.post_snapshot_observations
                    else None
                )
                controller = capture.controllers[-1] if capture.controllers else None
                diagnostic = observation_diagnostics(observation, snapshot)
                row: dict[str, Any] = {
                    "phase": phase,
                    "source_tick": int(entry.source_tick),
                    "predecessor_verify_tick": (
                        replay_window.predecessor_verify_tick
                    ),
                    "predecessor_verify_time_s": (
                        replay_window.predecessor_verify_time_s
                    ),
                    "controller_anchor_tick": (
                        replay_window.controller_anchor_tick
                    ),
                    "controller_anchor_time_s": (
                        replay_window.controller_anchor_time_s
                    ),
                    "target_entry_tick": replay_window.target_entry_tick,
                    "episode_sensor_tick_offset": int(
                        getattr(backend, "_episode_sensor_tick_offset", -1)
                    ),
                    "source_replay_steps": replay_window.source_replay_steps,
                    "effective_entry_offset_s": (
                        phase_entry_time_s(replay_window.source_replay_steps)
                    ),
                    **replay_window.restore_bindings,
                    "source_control_physics_ticks": list(replay_window.control_ticks),
                    "snapshot_path": str(entry.snapshot_path),
                    "snapshot_file_sha256": entry.file_sha256,
                    "snapshot_state_sha256": entry.state_sha256,
                    "source_command_file_sha256": snapshot["source_artifacts"][
                        "command"
                    ]["sha256"],
                    "source_observation_file_sha256": snapshot[
                        "source_artifacts"
                    ]["observation"]["sha256"],
                    "source_command_row_canonical_sha256s": [
                        command["source_command_row_canonical_sha256"]
                        for command in source_commands
                    ],
                    "source_observation_row_canonical_sha256s": [
                        command["source_observation_row_canonical_sha256"]
                        for command in source_commands
                    ],
                    "source_adapter_input_sha256s": [
                        command["source_adapter_input_sha256"]
                        for command in source_commands
                    ],
                    "source_drive_target_full12_sha256s": [
                        command["drive_target_full12_sha256"]
                        for command in source_commands
                    ],
                    "source_actuation_contract_sha256s": [
                        command["actuation_contract_sha256"]
                        for command in source_commands
                    ],
                    "attempt_index_for_phase": repeat,
                    "attempt_kind": capture.attempt_kind,
                    "scene_lifecycle": (
                        "reused_scene" if scene_existed else "fresh_scene"
                    ),
                    "scene_existed_before": scene_existed,
                    "scene_creation_count_total": book.scene_creation_count,
                    "reset_completed": frame is not None and exception is None,
                    "exception": exception,
                    "failure_classification": (
                        None
                        if caught_exception is None
                        else (
                            "EFFECTIVE_ENTRY_ACCEPTANCE_MISMATCH"
                            if _is_ordinary_reset_rejection(
                                caught_exception,
                                backend=backend,
                                capture=capture,
                            )
                            else "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
                        )
                    ),
                    "physics_steps_during_reset": capture.physics_steps,
                    "simulation_forwards_during_reset": capture.simulation_forwards,
                    "simulation_resets_during_reset": capture.simulation_resets,
                    "simulation_stops_during_reset": capture.simulation_stops,
                    "extra_physics_priming_steps": (
                        replay_window.source_replay_steps
                    ),
                    "post_prime_contact_sensor_read_count": len(
                        capture.post_snapshot_observations
                    ),
                    "fsm_or_episode_advanced_for_probe": False,
                    "reset_writes": None
                    if capture.reset_writes is None
                    else dict(capture.reset_writes),
                    "snapshot_state_write": (
                        dict(
                            getattr(backend, "_snapshot_restoration", {}).get(
                                "physical_state", {}
                            )
                        )
                        if isinstance(
                            getattr(backend, "_snapshot_restoration", {}).get(
                                "physical_state"
                            ),
                            Mapping,
                        )
                        else (
                            None
                            if capture.snapshot_state_write is None
                            else dict(capture.snapshot_state_write)
                        )
                    ),
                    "observation_diagnostics": diagnostic,
                    "clocks": {
                        **_controller_clocks(controller, frame),
                        "backend_episode_tick": int(
                            getattr(backend, "_episode_tick", -1)
                        ),
                        "authoritative_frame_committed": frame is not None,
                    },
                }
                row["passed"] = _attempt_passed(
                    row,
                    replay_window=replay_window,
                    calibration_mode=calibration_mode,
                )
                attempts.append(row)
                payload["attempts"] = attempts
                payload["completed_attempt_count"] = len(attempts)
                _write_probe(output, payload)
                if caught_exception is not None and not _is_ordinary_reset_rejection(
                    caught_exception,
                    backend=backend,
                    capture=capture,
                ):
                    raise PhaseSnapshotLiveProbeError(
                        "phase-snapshot reset failed before a trustworthy post-write "
                        "diagnostic was available"
                    ) from caught_exception
                assert_phase_snapshot_bundle_unchanged(
                    snapshot_bundle,
                    canonical_root=snapshot_bundle.snapshot_root,
                )
                if effective_entry_contract is not None:
                    assert_effective_phase_entry_contract_unchanged(
                        effective_entry_contract,
                        expected_snapshot_bundle=snapshot_bundle,
                    )
                if caught_exception is None and row["passed"] is not True:
                    # A production reset which returns successfully must agree
                    # with every probe runtime invariant.  Otherwise the live
                    # verifier, instrumentation, or controller/clock contract
                    # is inconsistent; that is not an ordinary restoration
                    # rejection which a finalized exit-code-2 diagnostic may
                    # represent.
                    row["failure_classification"] = (
                        "FATAL_INTEGRITY_OR_INFRASTRUCTURE"
                    )
                    raise PhaseSnapshotLiveProbeError(
                        "successful phase-snapshot reset violated probe runtime "
                        "invariants"
                    )
    except Exception as exc:
        _seal_fatal_probe_failure(output, payload, attempts, exc)
        raise PhaseSnapshotLiveProbeError(
            "phase-snapshot live probe terminated on an integrity or "
            "infrastructure failure"
        ) from exc
    else:
        failures = [
            f"{row['phase']}[{row['attempt_kind']}]"
            for row in attempts
            if row.get("passed") is not True
        ]
        payload["failure_reasons"] = failures
        payload["failure_classification"] = (
            None if not failures else "EFFECTIVE_ENTRY_ACCEPTANCE_MISMATCH"
        )

    fresh_count = sum(
        row.get("scene_lifecycle") == "fresh_scene" for row in attempts
    )
    reused_count = sum(
        row.get("scene_lifecycle") == "reused_scene" for row in attempts
    )
    complete = bool(
        len(attempts) == payload["expected_attempt_count"]
        and fresh_count == payload["expected_fresh_scene_attempt_count"]
        and reused_count == payload["expected_reused_scene_attempt_count"]
        and attempts
        and attempts[0].get("scene_lifecycle") == "fresh_scene"
    )
    passed = bool(complete and all(row.get("passed") is True for row in attempts))
    payload.update(
        {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "complete": complete,
            "completed_attempt_count": len(attempts),
            "fresh_scene_attempt_count": fresh_count,
            "reused_scene_attempt_count": reused_count,
        }
    )
    try:
        assert_phase_snapshot_bundle_unchanged(
            snapshot_bundle,
            canonical_root=snapshot_bundle.snapshot_root,
        )
        if effective_entry_contract is not None:
            assert_effective_phase_entry_contract_unchanged(
                effective_entry_contract,
                expected_snapshot_bundle=snapshot_bundle,
            )
    except Exception as exc:
        _seal_fatal_probe_failure(output, payload, attempts, exc)
        raise PhaseSnapshotLiveProbeError(
            "phase-snapshot live probe terminated on an integrity or "
            "infrastructure failure"
        ) from exc
    _write_probe(output, payload)
    return payload


__all__ = [
    "ATTEMPTS_PER_PHASE",
    "PROBE_FILENAME",
    "PROBE_PHASES",
    "PROBE_SCHEMA",
    "PhaseSnapshotLiveProbeError",
    "observation_diagnostics",
    "run_phase_snapshot_live_probe",
]
