"""Versioned, deterministic 85-dimensional actor-observation schema.

The existing actor ABI is preserved: this module makes its implicit ordering,
units, normalization, clipping and non-finite policy explicit.  It has no
Isaac dependency and never reads raw Recording events.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    WHEEL_ORDER,
)


OBSERVATION_SCHEMA = "wlr50_clean.ppo_observation_schema.v1"
OBSERVATION_DIMENSION = 85
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_observation_schema.json"
)
CONTACT_CODE = {
    "AIR": 0.0,
    "GROUND": 1.0,
    "OBSTACLE": 2.0,
    "GROUND_AND_OBSTACLE": 3.0,
    "UNVERIFIED": -1.0,
}
CANONICAL_FEATURE_GROUPS = (
    ("state_one_hot", tuple(f"state_is_{state}" for state in STATE_IDS)),
    ("phase_progress", ("phase_progress",)),
    (
        "joint_position_error",
        tuple(f"{name}_position_error" for name in SERVO_ORDER),
    ),
    ("joint_velocity", tuple(f"{name}_velocity" for name in SERVO_ORDER)),
    ("wheel_velocity", tuple(f"{name}_velocity" for name in WHEEL_ORDER)),
    (
        "wheel_contact_code",
        (
            "front_left_wheel_contact_code",
            "front_right_wheel_contact_code",
            "rear_left_wheel_contact_code",
            "rear_right_wheel_contact_code",
        ),
    ),
    (
        "leg_air_top_crossing_history",
        tuple(
            f"{leg}_{suffix}"
            for leg in ("FL", "FR", "RL", "RR")
            for suffix in (
                "active_lift_latched",
                "front_face_crossed_latched",
                "top_loaded_latched",
            )
        ),
    ),
    ("body_orientation_wxyz", ("base_qw", "base_qx", "base_qy", "base_qz")),
    (
        "body_angular_velocity",
        (
            "base_angular_velocity_x",
            "base_angular_velocity_y",
            "base_angular_velocity_z",
        ),
    ),
    (
        "obstacle_relative_geometry",
        (
            "obstacle_front_dx",
            "obstacle_back_dx",
            "obstacle_left_dy",
            "obstacle_right_dy",
            "obstacle_bottom_dz",
            "obstacle_top_dz",
            "obstacle_length",
            "obstacle_signed_width",
            "obstacle_height",
        ),
    ),
    ("raw_full_body_com", ("full_body_com_x", "full_body_com_y", "full_body_com_z")),
    (
        "support_diagnostics",
        (
            "support_signed_margin",
            "support_projection_inside",
            "support_count",
            "support_valid",
        ),
    ),
    ("previous_action", tuple(f"previous_{name}" for name in FULL12_ORDER)),
)


class ObservationSchemaError(ValueError):
    """The actor observation cannot be represented by the frozen schema."""


class NonFiniteObservationError(ObservationSchemaError):
    """A NaN or infinity was rejected by the configured episode policy."""


def _finite_vector(
    values: Sequence[float], expected: int, label: str
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ObservationSchemaError(f"{label} must be numeric") from exc
    if len(result) != expected:
        raise ObservationSchemaError(
            f"{label} must contain {expected} values; received {len(result)}"
        )
    bad = [index for index, value in enumerate(result) if not math.isfinite(value)]
    if bad:
        raise NonFiniteObservationError(
            f"{label} contains non-finite values at indices {bad}"
        )
    return result


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _optional_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _guard_passed(observation: Any, name: str) -> bool:
    guards = _field(observation, "guards")
    value = guards.get(name, False)
    if isinstance(value, Mapping):
        return bool(value.get("passed", False))
    return bool(value)


def _contact_class_name(value: Any) -> str:
    contact_class = _field(value, "contact_class")
    enum_value = getattr(contact_class, "value", contact_class)
    return str(enum_value)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    group: str
    offset: int
    size: int
    names: tuple[str, ...]
    units: tuple[str, ...]
    normalization_method: str
    normalization_scale: tuple[float, ...]
    clip_min: float
    clip_max: float
    source: str

    @property
    def stop(self) -> int:
        return self.offset + self.size

    def normalize(self, values: Sequence[float]) -> tuple[float, ...]:
        vector = _finite_vector(values, self.size, self.group)
        if self.normalization_method == "identity":
            normalized = vector
        elif self.normalization_method == "fixed_scale":
            normalized = tuple(
                value / scale
                for value, scale in zip(
                    vector, self.normalization_scale, strict=True
                )
            )
        else:  # defensive: loader rejects this before construction
            raise ObservationSchemaError(
                f"unsupported normalization method {self.normalization_method!r}"
            )
        return tuple(
            max(self.clip_min, min(self.clip_max, value)) for value in normalized
        )


@dataclass(frozen=True, slots=True)
class PPOObservationFrame:
    """Raw groups in the frozen actor order, prior to normalization."""

    state_id: str
    macro_phase: int
    phase_progress: float
    joint_position_error8: tuple[float, ...]
    joint_velocity8: tuple[float, ...]
    wheel_velocity4: tuple[float, ...]
    wheel_contact_code4: tuple[float, ...]
    leg_history12: tuple[float, ...]
    body_orientation_wxyz4: tuple[float, ...]
    body_angular_velocity3: tuple[float, ...]
    obstacle_relative_geometry9: tuple[float, ...]
    full_body_com3: tuple[float, ...]
    support_diagnostics4: tuple[float, ...]
    previous_action_full12: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.state_id not in STATE_IDS:
            raise ObservationSchemaError(f"unknown state_id {self.state_id!r}")
        if self.macro_phase != STATE_IDS.index(self.state_id) + 1:
            raise ObservationSchemaError("state_id and macro_phase disagree")
        progress = float(self.phase_progress)
        if not math.isfinite(progress):
            raise NonFiniteObservationError("phase_progress is non-finite")
        object.__setattr__(self, "phase_progress", max(0.0, min(1.0, progress)))
        for name, size in (
            ("joint_position_error8", 8),
            ("joint_velocity8", 8),
            ("wheel_velocity4", 4),
            ("wheel_contact_code4", 4),
            ("leg_history12", 12),
            ("body_orientation_wxyz4", 4),
            ("body_angular_velocity3", 3),
            ("obstacle_relative_geometry9", 9),
            ("full_body_com3", 3),
            ("support_diagnostics4", 4),
            ("previous_action_full12", 12),
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(getattr(self, name), size, name),
            )

    def raw_vector(self) -> tuple[float, ...]:
        one_hot = tuple(1.0 if state == self.state_id else 0.0 for state in STATE_IDS)
        result = (
            one_hot
            + (self.phase_progress,)
            + self.joint_position_error8
            + self.joint_velocity8
            + self.wheel_velocity4
            + self.wheel_contact_code4
            + self.leg_history12
            + self.body_orientation_wxyz4
            + self.body_angular_velocity3
            + self.obstacle_relative_geometry9
            + self.full_body_com3
            + self.support_diagnostics4
            + self.previous_action_full12
        )
        if len(result) != OBSERVATION_DIMENSION:
            raise AssertionError("internal PPO observation ordering changed")
        return result

    @classmethod
    def from_live_observation(
        cls,
        observation: Any,
        *,
        state_id: str,
        macro_phase: int,
        phase_progress: float,
        previous_action_full12: Sequence[float],
    ) -> "PPOObservationFrame":
        """Build a frame using explicit canonical names, never dict order."""

        joints = _field(observation, "joints")
        wheels = _field(observation, "wheels")
        contacts = _field(observation, "contacts")
        base = _field(observation, "base")
        obstacle = _field(observation, "obstacle")
        center_of_mass = _field(observation, "center_of_mass")
        support = _field(observation, "support")

        wheel_rows = tuple(wheels[name] for name in WHEEL_ORDER)
        wheel_codes = []
        for wheel in wheel_rows:
            body_name = str(_field(wheel, "body_name"))
            class_name = _contact_class_name(contacts[body_name])
            if class_name not in CONTACT_CODE:
                raise ObservationSchemaError(
                    f"unknown wheel contact class {class_name!r}"
                )
            wheel_codes.append(CONTACT_CODE[class_name])

        history = tuple(
            float(_guard_passed(observation, f"{guard}:{leg}"))
            for leg in ("FL", "FR", "RL", "RR")
            for guard in (
                "reference_like_active_lift",
                "leg_front_face_crossed_latched",
                "leg_top_loaded_latched",
            )
        )
        base_position = _field(base, "position_w_m")
        front_x = float(_field(obstacle, "front_x_m"))
        back_x = float(_field(obstacle, "back_x_m"))
        left_y = float(_field(obstacle, "left_y_m"))
        right_y = float(_field(obstacle, "right_y_m"))
        bottom_z = float(_field(obstacle, "bottom_z_m"))
        top_z = float(_field(obstacle, "top_z_m"))
        geometry = (
            front_x - float(base_position[0]),
            back_x - float(base_position[0]),
            left_y - float(base_position[1]),
            right_y - float(base_position[1]),
            bottom_z - float(base_position[2]),
            top_z - float(base_position[2]),
            back_x - front_x,
            right_y - left_y,
            top_z - bottom_z,
        )
        signed_margin = _optional_field(support, "signed_margin_m")
        projection_inside = _optional_field(support, "projection_inside")
        support_values = (
            0.0 if signed_margin is None else float(signed_margin),
            -1.0 if projection_inside is None else float(bool(projection_inside)),
            float(_field(support, "support_count")),
            float(bool(_field(support, "valid"))),
        )
        return cls(
            state_id=state_id,
            macro_phase=macro_phase,
            phase_progress=phase_progress,
            joint_position_error8=tuple(
                float(_field(joints[name], "error_deg")) for name in SERVO_ORDER
            ),
            joint_velocity8=tuple(
                float(_field(joints[name], "velocity_deg_s")) for name in SERVO_ORDER
            ),
            wheel_velocity4=tuple(
                float(_field(wheel, "velocity_rad_s")) for wheel in wheel_rows
            ),
            wheel_contact_code4=tuple(wheel_codes),
            leg_history12=history,
            body_orientation_wxyz4=tuple(_field(base, "orientation_wxyz")),
            body_angular_velocity3=tuple(_field(base, "angular_velocity_w_rad_s")),
            obstacle_relative_geometry9=geometry,
            full_body_com3=tuple(_field(center_of_mass, "position_w_m")),
            support_diagnostics4=support_values,
            previous_action_full12=tuple(previous_action_full12),
        )


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    schema: str
    schema_name: str
    schema_version: int
    dimension: int
    nan_policy: str
    online_normalization_updates: bool
    normalization_status: str
    normalization_sources: tuple[str, ...]
    state_ids: tuple[str, ...]
    full12_order: tuple[str, ...]
    features: tuple[FeatureSpec, ...]
    path: Path

    def encode(
        self, frame: PPOObservationFrame, *, normalized: bool = True
    ) -> tuple[float, ...]:
        if frame.state_id not in self.state_ids:
            raise ObservationSchemaError(f"state {frame.state_id} is outside schema")
        raw = frame.raw_vector()
        if not normalized:
            return raw
        output: list[float] = []
        for feature in self.features:
            output.extend(feature.normalize(raw[feature.offset : feature.stop]))
        result = tuple(output)
        if len(result) != self.dimension or any(
            not math.isfinite(value) for value in result
        ):
            raise NonFiniteObservationError("encoded actor observation is invalid")
        return result

    def normalize_raw_vector(
        self, values: Sequence[float]
    ) -> tuple[float, ...]:
        """Normalize a canonical 85D ledger row without changing its order."""

        raw = _finite_vector(values, self.dimension, "raw actor observation")
        output: list[float] = []
        for feature in self.features:
            output.extend(feature.normalize(raw[feature.offset : feature.stop]))
        result = tuple(output)
        if len(result) != self.dimension:
            raise AssertionError("normalization changed the 85D observation ABI")
        return result

    def feature_rows(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for group in self.features:
            for local_index, (name, unit, scale) in enumerate(
                zip(group.names, group.units, group.normalization_scale, strict=True)
            ):
                rows.append(
                    {
                        "offset": group.offset + local_index,
                        "feature_name": name,
                        "group": group.group,
                        "unit": unit,
                        "normalization_method": group.normalization_method,
                        "normalization_scale": scale,
                        "clip_min": group.clip_min,
                        "clip_max": group.clip_max,
                        "nan_policy": self.nan_policy,
                        "source": group.source,
                    }
                )
        return tuple(rows)

    def write_csv(self, path: Path) -> None:
        rows = self.feature_rows()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _string_tuple(value: Any, size: int, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = (value,) * size
    elif isinstance(value, Sequence):
        result = tuple(str(item) for item in value)
    else:
        raise ObservationSchemaError(f"{label} must be a string or sequence")
    if len(result) != size:
        raise ObservationSchemaError(f"{label} must contain {size} values")
    return result


def _scale_tuple(value: Any, size: int, method: str) -> tuple[float, ...]:
    if method == "identity":
        return (1.0,) * size
    raw = value.get("scale")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        result = tuple(float(item) for item in raw)
    else:
        result = (float(raw),) * size
    if len(result) != size or any(
        not math.isfinite(item) or item <= 0.0 for item in result
    ):
        raise ObservationSchemaError("normalization scale is invalid")
    return result


def load_observation_schema(
    path: Path | str = DEFAULT_SCHEMA_PATH,
) -> ObservationSchema:
    selected = Path(path).resolve()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if payload.get("schema") != OBSERVATION_SCHEMA:
        raise ObservationSchemaError("unexpected observation schema")
    if int(payload.get("dimension", -1)) != OBSERVATION_DIMENSION:
        raise ObservationSchemaError("actor observation dimension must remain 85")
    if tuple(payload.get("state_ids", ())) != STATE_IDS:
        raise ObservationSchemaError("state ordering must remain P01-P13")
    if tuple(payload.get("full12_order", ())) != FULL12_ORDER:
        raise ObservationSchemaError("Full12 order differs from the canonical ABI")
    if payload.get("nan_policy") != "reject_episode":
        raise ObservationSchemaError("NaN policy must reject the episode")
    if bool(payload.get("online_normalization_updates")):
        raise ObservationSchemaError("evaluation normalization cannot update online")

    raw_features = tuple(payload.get("features", ()))
    if len(raw_features) != len(CANONICAL_FEATURE_GROUPS):
        raise ObservationSchemaError(
            "feature groups differ from the immutable canonical 85D layout"
        )
    features = []
    expected_offset = 0
    names_seen: set[str] = set()
    for raw, (expected_group, expected_names) in zip(
        raw_features, CANONICAL_FEATURE_GROUPS, strict=True
    ):
        size = int(raw["size"])
        offset = int(raw["offset"])
        if (
            str(raw.get("group")) != expected_group
            or tuple(str(item) for item in raw.get("names", ())) != expected_names
            or size != len(expected_names)
            or offset != expected_offset
        ):
            raise ObservationSchemaError(
                "feature group/name/order differs from the immutable canonical 85D layout"
            )
        names = tuple(str(item) for item in raw["names"])
        if len(names) != size or names_seen.intersection(names):
            raise ObservationSchemaError("feature names must be complete and unique")
        names_seen.update(names)
        normalization = raw.get("normalization", {})
        method = str(normalization.get("method"))
        if method not in {"identity", "fixed_scale"}:
            raise ObservationSchemaError(f"unsupported normalization {method!r}")
        clip = tuple(float(item) for item in raw["clip"])
        if len(clip) != 2 or not clip[0] < clip[1]:
            raise ObservationSchemaError("feature clip must be [minimum, maximum]")
        feature = FeatureSpec(
            group=str(raw["group"]),
            offset=offset,
            size=size,
            names=names,
            units=_string_tuple(raw["units"], size, "units"),
            normalization_method=method,
            normalization_scale=_scale_tuple(normalization, size, method),
            clip_min=clip[0],
            clip_max=clip[1],
            source=str(raw["source"]),
        )
        features.append(feature)
        expected_offset = feature.stop
    if expected_offset != OBSERVATION_DIMENSION or len(names_seen) != OBSERVATION_DIMENSION:
        raise ObservationSchemaError("feature layout does not cover exactly 85 values")
    stats = payload.get("normalization_statistics", {})
    if stats.get("status") != "INITIAL_NORMALIZATION_STATISTICS_NOT_TRAINING_FINAL":
        raise ObservationSchemaError("normalization status is not explicit")
    return ObservationSchema(
        schema=OBSERVATION_SCHEMA,
        schema_name=str(payload["schema_name"]),
        schema_version=int(payload["schema_version"]),
        dimension=OBSERVATION_DIMENSION,
        nan_policy="reject_episode",
        online_normalization_updates=False,
        normalization_status=str(stats["status"]),
        normalization_sources=tuple(str(item) for item in stats.get("sources", ())),
        state_ids=STATE_IDS,
        full12_order=FULL12_ORDER,
        features=tuple(features),
        path=selected,
    )


def lifecycle_phase_progress(
    lifecycle: str, *, elapsed_s: float = 0.0, active_duration_s: float = 0.0
) -> float:
    """Canonical progress that does not leak the previous phase into WAIT_ENTRY."""

    name = str(getattr(lifecycle, "value", lifecycle))
    if name == "WAIT_ENTRY":
        return 0.0
    if name in {"VERIFY_RESULT", "RECOVERY", "DONE"}:
        return 1.0
    if name != "EXECUTE_MOTION":
        raise ObservationSchemaError(f"unknown FSM lifecycle {name!r}")
    duration = float(active_duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(elapsed_s) / duration))
