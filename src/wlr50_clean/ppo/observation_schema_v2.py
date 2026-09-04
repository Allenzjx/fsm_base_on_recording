"""Versioned phase/stability actor observation layered on the frozen 85D v1 ABI.

The first 85 values are delegated to :mod:`observation_schema` without any
reinterpretation.  New runtime-observable groups are appended in a stable
order.  This module has no Isaac dependency; a live backend only needs to pass
the existing observation value objects.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import FULL12_ORDER, WHEEL_ORDER

from .observation_schema import (
    OBSERVATION_DIMENSION as V1_OBSERVATION_DIMENSION,
    PPOObservationFrame,
    load_observation_schema,
)


OBSERVATION_SCHEMA_V2 = "wlr50_clean.ppo_observation_schema.v2"
STATE_IDS = tuple(f"P{index:02d}" for index in range(1, 14))
LIFECYCLE_IDS = (
    "WAIT_ENTRY",
    "EXECUTE_MOTION",
    "VERIFY_RESULT",
    "RECOVERY",
    "DONE",
)
DEFAULT_SCHEMA_PATH_V2 = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_observation_schema_v2.json"
)
WHEEL_RADIUS_M = 0.04998999834060672
ACTIVE_LEG_BY_PHASE: Mapping[str, str | None] = {
    "P01": "FR",
    "P02": "FR",
    "P03": "FR",
    "P04": "FL",
    "P05": "FL",
    "P06": "FL",
    "P07": "RR",
    "P08": "RR",
    "P09": "RR",
    "P10": "RL",
    "P11": "RL",
    "P12": "RL",
    "P13": None,
}
LEG_TO_WHEEL = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}

ADDITIONAL_FEATURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "fsm_lifecycle_one_hot",
        tuple(f"lifecycle_is_{name}" for name in LIFECYCLE_IDS),
    ),
    (
        "body_linear_velocity_body",
        (
            "base_linear_velocity_body_x",
            "base_linear_velocity_body_y",
            "base_linear_velocity_body_z",
        ),
    ),
    (
        "imu_linear_acceleration_body",
        (
            "imu_linear_acceleration_body_x",
            "imu_linear_acceleration_body_y",
            "imu_linear_acceleration_body_z",
        ),
    ),
    (
        "full_body_com_velocity",
        (
            "full_body_com_velocity_x",
            "full_body_com_velocity_y",
            "full_body_com_velocity_z",
        ),
    ),
    (
        "wheel_normal_force",
        tuple(name.replace("ankle", "wheel_normal_force") for name in WHEEL_ORDER),
    ),
    (
        "wheel_load_fraction",
        tuple(name.replace("ankle", "wheel_load_fraction") for name in WHEEL_ORDER),
    ),
    (
        "wheel_longitudinal_slip",
        tuple(name.replace("ankle", "wheel_slip") for name in WHEEL_ORDER),
    ),
    (
        "active_leg_wheel_bottom_clearance",
        ("active_leg_wheel_bottom_clearance",),
    ),
    ("active_leg_vertical_velocity", ("active_leg_vertical_velocity",)),
    (
        "previous_projected_residual",
        tuple(f"previous_projected_residual_{name}" for name in FULL12_ORDER),
    ),
)
OBSERVATION_DIMENSION_V2 = V1_OBSERVATION_DIMENSION + sum(
    len(names) for _, names in ADDITIONAL_FEATURE_GROUPS
)


class ObservationSchemaV2Error(ValueError):
    """The v2 actor observation or its metadata violates the versioned ABI."""


class NonFiniteObservationV2Error(ObservationSchemaV2Error):
    """A non-finite runtime observation was rejected."""


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _optional_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_vector(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ObservationSchemaV2Error(f"{label} must be numeric") from exc
    if len(result) != size:
        raise ObservationSchemaV2Error(
            f"{label} must contain {size} values; received {len(result)}"
        )
    bad = [index for index, value in enumerate(result) if not math.isfinite(value)]
    if bad:
        raise NonFiniteObservationV2Error(
            f"{label} contains non-finite values at indices {bad}"
        )
    return result


def _string_tuple(value: Any, size: int, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = (value,) * size
    elif isinstance(value, Sequence):
        result = tuple(str(item) for item in value)
    else:
        raise ObservationSchemaV2Error(f"{label} must be a string or sequence")
    if len(result) != size:
        raise ObservationSchemaV2Error(f"{label} must contain {size} values")
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
        raise ObservationSchemaV2Error("normalization scale is invalid")
    return result


def _quat_rotate_inverse(
    quaternion_wxyz: Sequence[float], vector_world: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a world vector into the body frame using a normalized wxyz quaternion."""

    q = _finite_vector(quaternion_wxyz, 4, "body quaternion")
    vector = _finite_vector(vector_world, 3, "world vector")
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 1.0e-12:
        raise ObservationSchemaV2Error("body quaternion has zero norm")
    w, x, y, z = (value / norm for value in q)
    # Transpose of the body-to-world rotation matrix.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vector[0]
        + 2.0 * (x * y + w * z) * vector[1]
        + 2.0 * (x * z - w * y) * vector[2],
        2.0 * (x * y - w * z) * vector[0]
        + (1.0 - 2.0 * (x * x + z * z)) * vector[1]
        + 2.0 * (y * z + w * x) * vector[2],
        2.0 * (x * z + w * y) * vector[0]
        + 2.0 * (y * z - w * x) * vector[1]
        + (1.0 - 2.0 * (x * x + y * y)) * vector[2],
    )


@dataclass(frozen=True, slots=True)
class FeatureSpecV2:
    group: str
    offset: int
    names: tuple[str, ...]
    units: tuple[str, ...]
    normalization_method: str
    normalization_scale: tuple[float, ...]
    clip_min: float
    clip_max: float
    source: str
    frame: str
    deployability: str

    @property
    def size(self) -> int:
        return len(self.names)

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
        else:
            raise ObservationSchemaV2Error(
                f"unsupported normalization {self.normalization_method!r}"
            )
        return tuple(
            max(self.clip_min, min(self.clip_max, value)) for value in normalized
        )


@dataclass(frozen=True, slots=True)
class PPOObservationFrameV2:
    """The immutable 85D v1 frame plus appended stability measurements."""

    v1: PPOObservationFrame
    lifecycle: str
    body_linear_velocity_body3: tuple[float, ...]
    imu_linear_acceleration_body3: tuple[float, ...]
    full_body_com_velocity3: tuple[float, ...]
    wheel_normal_forces4: tuple[float, ...]
    wheel_load_fractions4: tuple[float, ...]
    wheel_slip4: tuple[float, ...]
    active_leg_wheel_bottom_clearance_m: float
    active_leg_vertical_velocity_m_s: float
    previous_projected_residual_full12: tuple[float, ...]

    def __post_init__(self) -> None:
        lifecycle = str(getattr(self.lifecycle, "value", self.lifecycle))
        if lifecycle not in LIFECYCLE_IDS:
            raise ObservationSchemaV2Error(f"unknown FSM lifecycle {lifecycle!r}")
        object.__setattr__(self, "lifecycle", lifecycle)
        for name, size in (
            ("body_linear_velocity_body3", 3),
            ("imu_linear_acceleration_body3", 3),
            ("full_body_com_velocity3", 3),
            ("wheel_normal_forces4", 4),
            ("wheel_load_fractions4", 4),
            ("wheel_slip4", 4),
            ("previous_projected_residual_full12", 12),
        ):
            object.__setattr__(
                self, name, _finite_vector(getattr(self, name), size, name)
            )
        forces = self.wheel_normal_forces4
        fractions = self.wheel_load_fractions4
        if any(value < 0.0 for value in forces):
            raise ObservationSchemaV2Error("wheel normal forces cannot be negative")
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ObservationSchemaV2Error("wheel load fractions must lie in [0,1]")
        if sum(forces) > 1.0e-9 and not math.isclose(
            sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise ObservationSchemaV2Error(
                "loaded wheel fractions must sum to one"
            )
        for name in (
            "active_leg_wheel_bottom_clearance_m",
            "active_leg_vertical_velocity_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise NonFiniteObservationV2Error(f"{name} is non-finite")
            object.__setattr__(self, name, value)

    @property
    def state_id(self) -> str:
        return self.v1.state_id

    def raw_vector(self) -> tuple[float, ...]:
        lifecycle_one_hot = tuple(
            1.0 if name == self.lifecycle else 0.0 for name in LIFECYCLE_IDS
        )
        result = (
            self.v1.raw_vector()
            + lifecycle_one_hot
            + self.body_linear_velocity_body3
            + self.imu_linear_acceleration_body3
            + self.full_body_com_velocity3
            + self.wheel_normal_forces4
            + self.wheel_load_fractions4
            + self.wheel_slip4
            + (self.active_leg_wheel_bottom_clearance_m,)
            + (self.active_leg_vertical_velocity_m_s,)
            + self.previous_projected_residual_full12
        )
        if len(result) != OBSERVATION_DIMENSION_V2:
            raise AssertionError("internal v2 observation ordering changed")
        return result

    @classmethod
    def from_live_observation(
        cls,
        observation: Any,
        *,
        state_id: str,
        macro_phase: int,
        lifecycle: str,
        phase_progress: float,
        previous_action_full12: Sequence[float],
        previous_projected_residual_full12: Sequence[float],
        wheel_radius_m: float = WHEEL_RADIUS_M,
    ) -> "PPOObservationFrameV2":
        """Build v2 exclusively from deployable live observation fields."""

        v1 = PPOObservationFrame.from_live_observation(
            observation,
            state_id=state_id,
            macro_phase=macro_phase,
            phase_progress=phase_progress,
            previous_action_full12=previous_action_full12,
        )
        base = _field(observation, "base")
        imu = _field(observation, "imu")
        center_of_mass = _field(observation, "center_of_mass")
        wheels = _field(observation, "wheels")
        contacts = _field(observation, "contacts")
        bodies = _field(observation, "bodies")
        obstacle = _field(observation, "obstacle")

        body_velocity = _quat_rotate_inverse(
            _field(base, "orientation_wxyz"),
            _field(base, "linear_velocity_w_m_s"),
        )
        wheel_rows = tuple(wheels[name] for name in WHEEL_ORDER)
        normal_forces = []
        for wheel in wheel_rows:
            contact = contacts[str(_field(wheel, "body_name"))]
            force = 0.0
            for pair_name in ("ground", "obstacle"):
                pair = _field(contact, pair_name)
                if bool(_optional_field(pair, "pair_verified", False)):
                    force += max(0.0, float(_field(pair, "normal_force_n")))
            normal_forces.append(force)
        total_force = sum(normal_forces)
        load_fractions = (
            tuple(force / total_force for force in normal_forces)
            if total_force > 1.0e-9
            else (0.0, 0.0, 0.0, 0.0)
        )

        radius = float(wheel_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ObservationSchemaV2Error("wheel_radius_m must be positive and finite")
        forward_speed = body_velocity[0]
        wheel_surface_speeds = tuple(
            radius * float(_field(wheel, "velocity_rad_s")) for wheel in wheel_rows
        )
        slips = tuple(
            (surface - forward_speed)
            / max(abs(surface), abs(forward_speed), 0.05)
            for surface in wheel_surface_speeds
        )

        active_leg = ACTIVE_LEG_BY_PHASE[state_id]
        clearance = 0.0
        vertical_velocity = 0.0
        if active_leg is not None:
            wheel = wheels[LEG_TO_WHEEL[active_leg]]
            bottom = _optional_field(wheel, "bottom_w_m")
            if bool(_optional_field(wheel, "geometry_verified", False)) and bottom is not None:
                clearance = float(bottom[2]) - float(_field(obstacle, "top_z_m"))
            body_name = str(_field(wheel, "body_name"))
            wheel_body = bodies[body_name]
            vertical_velocity = float(_field(wheel_body, "linear_velocity_w_m_s")[2])

        return cls(
            v1=v1,
            lifecycle=lifecycle,
            body_linear_velocity_body3=body_velocity,
            imu_linear_acceleration_body3=tuple(
                _field(imu, "linear_acceleration_b_m_s2")
            ),
            full_body_com_velocity3=tuple(
                _field(center_of_mass, "velocity_w_m_s")
            ),
            wheel_normal_forces4=tuple(normal_forces),
            wheel_load_fractions4=load_fractions,
            wheel_slip4=slips,
            active_leg_wheel_bottom_clearance_m=clearance,
            active_leg_vertical_velocity_m_s=vertical_velocity,
            previous_projected_residual_full12=tuple(
                previous_projected_residual_full12
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationSchemaV2:
    schema: str
    schema_name: str
    schema_version: int
    dimension: int
    v1_prefix_dimension: int
    nan_policy: str
    online_normalization_updates: bool
    normalization_status: str
    state_ids: tuple[str, ...]
    lifecycle_ids: tuple[str, ...]
    full12_order: tuple[str, ...]
    features: tuple[FeatureSpecV2, ...]
    path: Path

    def encode(
        self, frame: PPOObservationFrameV2, *, normalized: bool = True
    ) -> tuple[float, ...]:
        raw = frame.raw_vector()
        if len(raw) != self.dimension:
            raise ObservationSchemaV2Error("frame and schema dimensions disagree")
        if not normalized:
            return raw
        output: list[float] = []
        for feature in self.features:
            output.extend(feature.normalize(raw[feature.offset : feature.stop]))
        result = tuple(output)
        if len(result) != self.dimension or any(
            not math.isfinite(value) for value in result
        ):
            raise NonFiniteObservationV2Error("encoded v2 observation is invalid")
        return result

    def feature_rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for group in self.features:
            for index, (name, unit, scale) in enumerate(
                zip(group.names, group.units, group.normalization_scale, strict=True)
            ):
                rows.append(
                    {
                        "offset": group.offset + index,
                        "name": name,
                        "group": group.group,
                        "unit": unit,
                        "source": group.source,
                        "frame": group.frame,
                        "normalization": json.dumps(
                            {
                                "method": group.normalization_method,
                                "scale": scale,
                            },
                            separators=(",", ":"),
                        ),
                        "clip": json.dumps(
                            [group.clip_min, group.clip_max], separators=(",", ":")
                        ),
                        "nan_policy": self.nan_policy,
                        "deployability": group.deployability,
                    }
                )
        if len(rows) != self.dimension:
            raise AssertionError("feature metadata does not cover the v2 dimension")
        return tuple(rows)

    def write_csv(self, path: Path | str) -> None:
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        rows = self.feature_rows()
        with selected.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, path: Path | str) -> None:
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.schema,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "dimension": self.dimension,
            "dimension_policy": "sum_feature_sizes",
            "v1_prefix_dimension": self.v1_prefix_dimension,
            "features": self.feature_rows(),
        }
        selected.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_observation_schema_v2(
    path: Path | str = DEFAULT_SCHEMA_PATH_V2,
) -> ObservationSchemaV2:
    selected = Path(path).resolve()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if payload.get("schema") != OBSERVATION_SCHEMA_V2:
        raise ObservationSchemaV2Error("unexpected v2 observation schema")
    if payload.get("dimension_policy") != "sum_feature_sizes":
        raise ObservationSchemaV2Error("v2 dimension must be derived from feature sizes")
    if tuple(payload.get("state_ids", ())) != STATE_IDS:
        raise ObservationSchemaV2Error("state ordering must remain P01-P13")
    if tuple(payload.get("lifecycle_ids", ())) != LIFECYCLE_IDS:
        raise ObservationSchemaV2Error("lifecycle ordering is invalid")
    if tuple(payload.get("full12_order", ())) != FULL12_ORDER:
        raise ObservationSchemaV2Error("Full12 order differs from the canonical ABI")
    if payload.get("nan_policy") != "reject_episode":
        raise ObservationSchemaV2Error("NaN policy must reject the episode")
    if bool(payload.get("online_normalization_updates")):
        raise ObservationSchemaV2Error(
            "validation/test normalization cannot update online"
        )

    v1_schema = load_observation_schema()
    raw_features = tuple(payload.get("features", ()))
    expected_group_count = len(v1_schema.features) + len(ADDITIONAL_FEATURE_GROUPS)
    if len(raw_features) != expected_group_count:
        raise ObservationSchemaV2Error("v2 feature group count is invalid")

    features: list[FeatureSpecV2] = []
    names_seen: set[str] = set()
    expected_offset = 0
    for raw in raw_features:
        names = tuple(str(item) for item in raw.get("names", ()))
        if not names or names_seen.intersection(names):
            raise ObservationSchemaV2Error("feature names must be nonempty and unique")
        names_seen.update(names)
        method = str(raw.get("normalization", {}).get("method"))
        if method not in {"identity", "fixed_scale"}:
            raise ObservationSchemaV2Error(
                f"unsupported normalization method {method!r}"
            )
        clip = tuple(float(item) for item in raw.get("clip", ()))
        if len(clip) != 2 or not all(math.isfinite(item) for item in clip) or clip[0] >= clip[1]:
            raise ObservationSchemaV2Error("feature clip must be a finite [min,max]")
        feature = FeatureSpecV2(
            group=str(raw.get("group")),
            offset=expected_offset,
            names=names,
            units=_string_tuple(raw.get("units"), len(names), "units"),
            normalization_method=method,
            normalization_scale=_scale_tuple(
                raw.get("normalization", {}), len(names), method
            ),
            clip_min=clip[0],
            clip_max=clip[1],
            source=str(raw.get("source", "")),
            frame=str(raw.get("frame", "")),
            deployability=str(raw.get("deployability", "")),
        )
        if not feature.source or not feature.frame:
            raise ObservationSchemaV2Error("every feature group needs source and frame")
        if feature.deployability != "runtime_observable":
            raise ObservationSchemaV2Error(
                "actor features must all be runtime observable"
            )
        features.append(feature)
        expected_offset = feature.stop

    # Prove that the entire legacy vector is a byte-for-byte semantic prefix:
    # same groups, names, units, normalization and clips at the same offsets.
    for legacy, current in zip(
        v1_schema.features, features[: len(v1_schema.features)], strict=True
    ):
        if (
            current.group != legacy.group
            or current.offset != legacy.offset
            or current.names != legacy.names
            or current.units != legacy.units
            or current.normalization_method != legacy.normalization_method
            or current.normalization_scale != legacy.normalization_scale
            or current.clip_min != legacy.clip_min
            or current.clip_max != legacy.clip_max
        ):
            raise ObservationSchemaV2Error(
                "the first 85 v2 features must preserve the complete v1 ABI"
            )

    additional = features[len(v1_schema.features) :]
    for feature, (expected_group, expected_names) in zip(
        additional, ADDITIONAL_FEATURE_GROUPS, strict=True
    ):
        if feature.group != expected_group or feature.names != expected_names:
            raise ObservationSchemaV2Error(
                "appended v2 feature group/name/order is not canonical"
            )
    if expected_offset != OBSERVATION_DIMENSION_V2:
        raise ObservationSchemaV2Error(
            "auto-derived v2 dimension differs from the canonical feature layout"
        )
    statistics = payload.get("normalization_statistics", {})
    if statistics.get("update_scope") != "train_only" or not bool(
        statistics.get("frozen_for_validation_and_test")
    ):
        raise ObservationSchemaV2Error(
            "normalization statistics must update only on training data"
        )
    return ObservationSchemaV2(
        schema=OBSERVATION_SCHEMA_V2,
        schema_name=str(payload.get("schema_name")),
        schema_version=int(payload.get("schema_version", -1)),
        dimension=expected_offset,
        v1_prefix_dimension=v1_schema.dimension,
        nan_policy="reject_episode",
        online_normalization_updates=False,
        normalization_status=str(statistics.get("status")),
        state_ids=STATE_IDS,
        lifecycle_ids=LIFECYCLE_IDS,
        full12_order=FULL12_ORDER,
        features=tuple(features),
        path=selected,
    )
