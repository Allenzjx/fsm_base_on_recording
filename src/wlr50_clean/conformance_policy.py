"""Typed, single-source Recording/FSM conformance policy.

Hard physical safety is intentionally outside the numeric tolerance.  The
policy only changes reference-conformance envelopes and preserves an explicit
legacy 15 percent reporting view.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


POLICY_SCHEMA = "wlr50_clean.conformance_policy.v3"
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "conformance_policy.yaml"


@dataclass(frozen=True, slots=True)
class AbsoluteFloor:
    absolute_allowance: float
    unit: str
    source: str


@dataclass(frozen=True, slots=True)
class ConformancePolicy:
    schema: str
    reference_version: str
    rear_leg_order: str
    active_fraction: float
    active_percent: float
    legacy_fraction: float
    legacy_percent: float
    preserve_legacy_result: bool
    hard_safety_unchanged: bool
    same_source_event_same_tick: bool
    conformance_can_block_entry: bool
    conformance_can_block_completion: bool
    conformance_can_block_task_success: bool
    conformance_is_not_body_safety: bool
    reference_bounded_correction_fraction: float
    absolute_floors: Mapping[str, AbsoluteFloor]
    gate_classes: Mapping[str, tuple[str, ...]]
    path: Path

    @property
    def minimum_time_scale(self) -> float:
        return 1.0 - self.active_fraction

    @property
    def maximum_time_scale(self) -> float:
        return 1.0 + self.active_fraction

    def floor(self, name: str) -> AbsoluteFloor:
        try:
            return self.absolute_floors[name]
        except KeyError as exc:
            raise KeyError(f"unknown conformance absolute floor: {name}") from exc

    def fraction(self, *, legacy: bool = False) -> float:
        return self.legacy_fraction if legacy else self.active_fraction

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, path: Path = DEFAULT_POLICY_PATH
    ) -> "ConformancePolicy":
        if payload.get("schema") != POLICY_SCHEMA:
            raise ValueError(f"conformance policy schema must be {POLICY_SCHEMA}")
        active = payload.get("active_tolerance")
        legacy = payload.get("legacy_reporting")
        hard = payload.get("hard_safety")
        atomic = payload.get("atomic_event_timing")
        runtime = payload.get("runtime")
        raw_floors = payload.get("absolute_floors")
        raw_gates = payload.get("gate_classes")
        if not all(isinstance(item, Mapping) for item in (
            active, legacy, hard, atomic, runtime, raw_floors, raw_gates
        )):
            raise ValueError("conformance policy sections must be mappings")
        assert isinstance(active, Mapping)
        assert isinstance(legacy, Mapping)
        assert isinstance(hard, Mapping)
        assert isinstance(atomic, Mapping)
        assert isinstance(runtime, Mapping)
        assert isinstance(raw_floors, Mapping)
        assert isinstance(raw_gates, Mapping)
        active_fraction = float(active["fraction"])
        active_percent = float(active["percent"])
        legacy_fraction = float(legacy["legacy_fraction"])
        legacy_percent = float(legacy.get("legacy_percent", 100.0 * legacy_fraction))
        if not (0.0 < legacy_fraction < active_fraction < 1.0):
            raise ValueError("tolerance fractions must satisfy 0 < legacy < active < 1")
        if abs(active_percent - 100.0 * active_fraction) > 1.0e-12:
            raise ValueError("active fraction and percent disagree")
        if abs(legacy_percent - 100.0 * legacy_fraction) > 1.0e-12:
            raise ValueError("legacy fraction and percent disagree")
        floors: dict[str, AbsoluteFloor] = {}
        for name, raw in raw_floors.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"absolute floor {name} must be a mapping")
            floor = AbsoluteFloor(
                absolute_allowance=float(raw["absolute_allowance"]),
                unit=str(raw["unit"]),
                source=str(raw["source"]),
            )
            if floor.absolute_allowance <= 0.0 or not floor.source:
                raise ValueError(f"absolute floor {name} is invalid")
            floors[str(name)] = floor
        gates: dict[str, tuple[str, ...]] = {}
        for name in ("hard_physical_safety", "reference_conformance", "diagnostic_only"):
            values = raw_gates.get(name)
            if not isinstance(values, list) or not values:
                raise ValueError(f"gate class {name} must be a non-empty list")
            gates[name] = tuple(str(item) for item in values)
        policy = cls(
            schema=POLICY_SCHEMA,
            reference_version=str(payload["reference_version"]),
            rear_leg_order=str(payload["rear_leg_order"]),
            active_fraction=active_fraction,
            active_percent=active_percent,
            legacy_fraction=legacy_fraction,
            legacy_percent=legacy_percent,
            preserve_legacy_result=bool(legacy["preserve_15_percent_result"]),
            hard_safety_unchanged=bool(hard["unchanged"]),
            same_source_event_same_tick=bool(
                atomic["same_source_event_requires_same_physics_tick"]
            ),
            conformance_can_block_entry=bool(runtime["conformance_can_block_entry"]),
            conformance_can_block_completion=bool(
                runtime["conformance_can_block_completion"]
            ),
            conformance_can_block_task_success=bool(
                runtime["conformance_can_block_task_success"]
            ),
            conformance_is_not_body_safety=bool(
                runtime["conformance_is_not_body_safety"]
            ),
            reference_bounded_correction_fraction=float(
                runtime["reference_bounded_correction_fraction"]
            ),
            absolute_floors=floors,
            gate_classes=gates,
            path=Path(path).resolve(),
        )
        if not policy.hard_safety_unchanged:
            raise ValueError("hard physical safety must remain unchanged")
        if policy.conformance_can_block_task_success:
            raise ValueError("Recording conformance must not block physical task success")
        if policy.conformance_can_block_entry or policy.conformance_can_block_completion:
            raise ValueError("the advisory Recording tolerance cannot be a runtime gate")
        if not 0.0 < policy.reference_bounded_correction_fraction <= policy.active_fraction:
            raise ValueError("nominal FSM correction fraction is invalid")
        return policy


def load_conformance_policy(path: Path | str = DEFAULT_POLICY_PATH) -> ConformancePolicy:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("conformance policy root must be a mapping")
    return ConformancePolicy.from_mapping(payload, path=selected)


@lru_cache(maxsize=1)
def get_conformance_policy() -> ConformancePolicy:
    return load_conformance_policy(DEFAULT_POLICY_PATH)
