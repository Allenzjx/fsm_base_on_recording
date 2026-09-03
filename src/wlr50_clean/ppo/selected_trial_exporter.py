"""Streaming, read-only conversion of a confirmed Trial into 15 Hz transitions.

The exporter consumes the immutable 120 Hz command and observation ledgers one
row at a time.  It does not choose a Trial, write final artifacts, or infer
that an unconfirmed Trial is successful.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .action_projection import (
    ActionProjectionConfig,
    ZeroResidualEpisodeAudit,
    ZeroResidualEpisodeAuditor,
    bitwise_full12_equal,
    load_action_projection_config,
)
from .episode_logger import (
    BaselineEquivalenceEvidence,
    EpisodeLogger,
    EpisodeTransition,
)
from .observation_schema import (
    ObservationSchema,
    PPOObservationFrame,
    load_observation_schema,
)
from .reward_terms import RewardCalculator, RewardSignals


PHYSICS_HZ = 120.0
DECISION_HZ = 15.0
PHYSICS_TICKS_PER_DECISION = 8
ZERO_RESIDUAL_EQUIVALENCE_SCHEMA = (
    "wlr50_clean.zero_residual_full_episode_equivalence.v1"
)


class SelectedTrialExportError(ValueError):
    """A source ledger or explicit selection claim is not exportable."""


def _full12(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SelectedTrialExportError(f"{label} must be numeric") from exc
    if len(result) != 12 or any(not math.isfinite(value) for value in result):
        raise SelectedTrialExportError(f"{label} must contain twelve finite values")
    return result


def _action_mask(values: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SelectedTrialExportError(f"{label} must be binary") from exc
    if len(result) != 12 or any(value not in (0, 1) for value in result):
        raise SelectedTrialExportError(
            f"{label} must contain twelve binary values"
        )
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectedTrialExportError(f"{label} must be an object")
    return value


def _clock(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SelectedTrialExportError(f"{label} must be finite and non-negative")
    return result


def _guard_triggered(observation: Mapping[str, Any], name: str) -> bool:
    raw = _mapping(observation.get("guards", {}), "observation.guards").get(name, False)
    if isinstance(raw, Mapping):
        return bool(raw.get("passed", False))
    return bool(raw)


class _JsonlReader:
    """Binary line iterator that proves the exact immutable input bytes read."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = Path(path).resolve()
        self.label = label
        self.sha256 = hashlib.sha256()
        self.row_count = 0

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        with self.path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                self.sha256.update(raw)
                if not raw.strip():
                    raise SelectedTrialExportError(
                        f"{self.label} contains a blank row at line {line_number}"
                    )
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SelectedTrialExportError(
                        f"{self.label} contains invalid JSON at line {line_number}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise SelectedTrialExportError(
                        f"{self.label} line {line_number} is not an object"
                    )
                self.row_count += 1
                yield value


@dataclass(frozen=True, slots=True)
class SelectedTrialMetadata:
    """Facts supplied only after the external Trial-selection gate passes."""

    trial_id: str
    episode_id: str
    seed: int
    environment_hash: str
    controller_hash: str
    motion_contract_hash: str
    selected_trial_confirmed: bool
    physical_task_success: bool
    reference_conformance_diagnostic_only: bool
    terminated: bool
    truncated: bool
    termination_reason: str

    def __post_init__(self) -> None:
        if not self.selected_trial_confirmed:
            raise SelectedTrialExportError(
                "selected Trial is not confirmed; final dataset export is blocked"
            )
        if not self.physical_task_success:
            raise SelectedTrialExportError(
                "selected Trial is not classified as a physical task success"
            )
        if not self.reference_conformance_diagnostic_only:
            raise SelectedTrialExportError(
                "reference divergence must remain diagnostic-only for selection"
            )
        if not self.trial_id or not self.episode_id or int(self.seed) < 0:
            raise SelectedTrialExportError("selected Trial identity is invalid")
        if not all(
            (self.environment_hash, self.controller_hash, self.motion_contract_hash)
        ):
            raise SelectedTrialExportError("selected Trial hashes are required")
        if self.terminated == self.truncated:
            raise SelectedTrialExportError(
                "exactly one final terminated/truncated flag is required"
            )
        if not self.termination_reason:
            raise SelectedTrialExportError("final termination_reason is required")


@dataclass(frozen=True, slots=True)
class SelectedTrialExportResult:
    source_trial: str
    transition_count: int
    command_row_count: int
    observation_row_count: int
    first_physics_tick: int
    terminal_physics_tick: int
    command_ledger_sha256: str
    observation_ledger_sha256: str
    full_physics_tick_zero_residual_equivalence: ZeroResidualEpisodeAudit
    zero_residual_equivalence: BaselineEquivalenceEvidence
    logger: EpisodeLogger


@dataclass(frozen=True, slots=True)
class _Sample:
    physics_tick: int
    sim_time_s: float
    state_id: str
    macro_phase: int
    phase_progress: float
    observation: tuple[float, ...]
    nominal: tuple[float, ...]
    residual: tuple[float, ...]
    applied: tuple[float, ...]
    action_mask: tuple[int, ...]
    raw_observation: Mapping[str, Any]


class SelectedTrialStreamingExporter:
    """Build baseline transitions without loading either 120 Hz ledger in RAM."""

    def __init__(
        self,
        *,
        observation_schema: ObservationSchema | None = None,
        action_config: ActionProjectionConfig | None = None,
        reward_calculator: RewardCalculator | None = None,
    ) -> None:
        self.observation_schema = observation_schema or load_observation_schema()
        self.action_config = action_config or load_action_projection_config()
        self.reward_calculator = reward_calculator or RewardCalculator()
        if (
            self.action_config.physics_hz != PHYSICS_HZ
            or self.action_config.decision_hz != DECISION_HZ
            or self.action_config.physics_ticks_per_decision
            != PHYSICS_TICKS_PER_DECISION
        ):
            raise SelectedTrialExportError("export cadence must remain 120/15 Hz")

    @staticmethod
    def _validate_step(
        *,
        tick: int,
        time_s: float,
        previous_tick: int | None,
        previous_time_s: float | None,
        label: str,
    ) -> None:
        if previous_tick is None:
            return
        if tick != previous_tick + 1:
            raise SelectedTrialExportError(
                f"{label} physics tick must advance by exactly one"
            )
        if not math.isclose(
            time_s,
            float(previous_time_s) + 1.0 / PHYSICS_HZ,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SelectedTrialExportError(
                f"{label} time must advance by exactly 1/120 s"
            )

    def _sample(
        self,
        command: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> _Sample:
        ppo = _mapping(command.get("ppo"), "command.ppo")
        state_id = str(command.get("state_id"))
        if str(ppo.get("state_id")) != state_id:
            raise SelectedTrialExportError("command and PPO state_id disagree")
        macro_phase = int(ppo.get("macro_phase", -1))
        if macro_phase != int(state_id[1:]):
            raise SelectedTrialExportError("state_id and macro_phase disagree")
        progress = float(ppo.get("phase_progress", float("nan")))
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise SelectedTrialExportError("phase_progress is invalid")
        raw_actor = tuple(ppo.get("observation_vector", ()))
        actor = self.observation_schema.normalize_raw_vector(raw_actor)
        state_bits = tuple(float(value) for value in raw_actor[:13])
        expected_bits = tuple(
            1.0 if index == macro_phase - 1 else 0.0 for index in range(13)
        )
        if state_bits != expected_bits or float(raw_actor[13]) != progress:
            raise SelectedTrialExportError(
                "ledger actor vector state/progress prefix is inconsistent"
            )
        nominal = _full12(command.get("nominal_full12", ()), "nominal_full12")
        residual = _full12(command.get("residual_full12", ()), "residual_full12")
        logical = _full12(command.get("full12", ()), "full12")
        commanded = _full12(
            command.get("commanded_full12", ()), "commanded_full12"
        )
        applied = _full12(command.get("applied_full12", ()), "applied_full12")
        logged_action_mask = _action_mask(
            ppo.get("action_mask_full12", ()),
            "command.ppo.action_mask_full12",
        )
        frozen_action_mask = self.action_config.mask_for(state_id)
        if logged_action_mask != frozen_action_mask:
            raise SelectedTrialExportError(
                "logged PPO action_mask_full12 disagrees with the frozen "
                f"{state_id} phase mask"
            )
        # The selected baseline must be proven here, not asserted by metadata.
        if any(value != 0.0 for value in residual):
            raise SelectedTrialExportError("source Trial contains a non-zero residual")
        if not all(
            bitwise_full12_equal(nominal, value)
            for value in (logical, commanded, applied)
        ):
            raise SelectedTrialExportError(
                "source Trial logical/commanded/applied full12 is not bitwise equal "
                "to nominal_full12"
            )
        return _Sample(
            physics_tick=int(command["control_physics_tick"]),
            sim_time_s=_clock(command["sim_time_s"], "command.sim_time_s"),
            state_id=state_id,
            macro_phase=macro_phase,
            phase_progress=progress,
            observation=actor,
            nominal=nominal,
            residual=residual,
            applied=applied,
            action_mask=logged_action_mask,
            raw_observation=observation,
        )

    def _terminal_actor(self, pending: _Sample, observation: Mapping[str, Any]) -> tuple[float, ...]:
        frame = PPOObservationFrame.from_live_observation(
            observation,
            state_id=pending.state_id,
            macro_phase=pending.macro_phase,
            phase_progress=pending.phase_progress,
            previous_action_full12=pending.applied,
        )
        return self.observation_schema.encode(frame)

    @staticmethod
    def _position_x(observation: Mapping[str, Any]) -> float:
        base = _mapping(observation.get("base"), "observation.base")
        position = tuple(float(value) for value in base.get("position_w_m", ()))
        if len(position) != 3 or any(not math.isfinite(value) for value in position):
            raise SelectedTrialExportError("observation base position is invalid")
        return position[0]

    def _reward(
        self,
        current: _Sample,
        next_observation: Mapping[str, Any],
        *,
        next_phase_progress: float,
        final_success: bool,
    ) -> Mapping[str, float]:
        base = _mapping(next_observation.get("base"), "observation.base")
        angular = tuple(float(value) for value in base.get("angular_velocity_w_rad_s", ()))
        quaternion = tuple(float(value) for value in base.get("orientation_wxyz", ()))
        if len(angular) != 3 or len(quaternion) != 4:
            raise SelectedTrialExportError("observation base orientation/rate is invalid")
        qw, qx, qy, qz = quaternion
        roll = math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        support = _mapping(next_observation.get("support", {}), "observation.support")
        support_valid = bool(support.get("valid", False))
        margin = support.get("signed_margin_m")
        body_collision = bool(
            _mapping(
                next_observation.get("body_collision", {}),
                "observation.body_collision",
            ).get("detected", False)
        )
        signals = RewardSignals(
            task_success=final_success,
            forward_progress_delta_m=(
                self._position_x(next_observation)
                - self._position_x(current.raw_observation)
            ),
            phase_progress_delta=next_phase_progress - current.phase_progress,
            body_collision=body_collision,
            wheel_only_climb=_guard_triggered(
                next_observation, "wheel_only_climb_detected"
            ),
            fall=_guard_triggered(next_observation, "physics_explosion_or_fall"),
            joint_limit_violation=_guard_triggered(
                next_observation, "joint_hard_limit_violation"
            ),
            body_angular_speed_rad_s=math.sqrt(sum(value * value for value in angular)),
            pitch_rad=pitch,
            roll_rad=roll,
            support_margin_m=(None if margin is None else float(margin)),
            support_valid=support_valid,
        )
        return self.reward_calculator.evaluate(signals).weighted_components

    def _append(
        self,
        logger: EpisodeLogger,
        metadata: SelectedTrialMetadata,
        current: _Sample,
        next_actor: tuple[float, ...],
        next_observation: Mapping[str, Any],
        next_phase_progress: float,
        control_tick: int,
        *,
        final: bool,
    ) -> None:
        logger.append(
            EpisodeTransition(
                episode_id=metadata.episode_id,
                trial_id=metadata.trial_id,
                seed=int(metadata.seed),
                control_tick=control_tick,
                sim_time=current.sim_time_s,
                state_id=current.state_id,
                macro_phase=current.macro_phase,
                phase_progress=current.phase_progress,
                observation_t=current.observation,
                nominal_action_t=current.nominal,
                residual_action_t=current.residual,
                applied_action_t=current.applied,
                action_mask_t=current.action_mask,
                task_result="SUCCESS",
                reward_components_t=self._reward(
                    current,
                    next_observation,
                    next_phase_progress=next_phase_progress,
                    final_success=(final and metadata.termination_reason == "SUCCESS"),
                ),
                terminated=(metadata.terminated if final else False),
                truncated=(metadata.truncated if final else False),
                termination_reason=(metadata.termination_reason if final else None),
                observation_t_plus_1=next_actor,
                environment_hash=metadata.environment_hash,
                controller_hash=metadata.controller_hash,
                motion_contract_hash=metadata.motion_contract_hash,
                observation_schema_version=(
                    f"{self.observation_schema.schema_name}.v"
                    f"{self.observation_schema.schema_version}"
                ),
                action_schema_version=(
                    f"{self.action_config.action_schema_name}.v"
                    f"{self.action_config.action_schema_version}"
                ),
            )
        )

    def export_to_logger(
        self,
        command_ledger: Path,
        observation_ledger: Path,
        *,
        metadata: SelectedTrialMetadata,
        logger: EpisodeLogger | None = None,
    ) -> SelectedTrialExportResult:
        """Read both ledgers once and return validated rows; no files are written."""

        output = logger or EpisodeLogger()
        if output.rows:
            raise SelectedTrialExportError("export logger must initially be empty")
        command_reader = _JsonlReader(Path(command_ledger), "command ledger")
        observation_reader = _JsonlReader(Path(observation_ledger), "observation ledger")
        observation_rows = iter(observation_reader)
        pending: _Sample | None = None
        first_tick: int | None = None
        previous_command_tick: int | None = None
        previous_command_time: float | None = None
        previous_observation_tick: int | None = None
        previous_observation_time: float | None = None
        control_tick = 0
        full_tick_auditor = ZeroResidualEpisodeAuditor()

        for command in command_reader:
            try:
                observation = next(observation_rows)
            except StopIteration as exc:
                raise SelectedTrialExportError(
                    "observation ledger ended before command ledger"
                ) from exc
            command_tick = int(command["control_physics_tick"])
            command_time = _clock(command["sim_time_s"], "command.sim_time_s")
            observation_tick = int(observation["physics_tick"])
            observation_time = _clock(
                observation["simulation_time_s"], "observation.simulation_time_s"
            )
            self._validate_step(
                tick=command_tick,
                time_s=command_time,
                previous_tick=previous_command_tick,
                previous_time_s=previous_command_time,
                label="command ledger",
            )
            self._validate_step(
                tick=observation_tick,
                time_s=observation_time,
                previous_tick=previous_observation_tick,
                previous_time_s=previous_observation_time,
                label="observation ledger",
            )
            if command_tick != observation_tick or not math.isclose(
                command_time, observation_time, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise SelectedTrialExportError(
                    "command and observation ledger clocks are not aligned"
                )
            if first_tick is None:
                first_tick = command_tick
            ppo_every_tick = _mapping(command.get("ppo"), "command.ppo")
            state_every_tick = str(command.get("state_id"))
            if str(ppo_every_tick.get("state_id")) != state_every_tick:
                raise SelectedTrialExportError(
                    "command and PPO state_id disagree"
                )
            logged_mask_every_tick = _action_mask(
                ppo_every_tick.get("action_mask_full12", ()),
                "command.ppo.action_mask_full12",
            )
            if logged_mask_every_tick != self.action_config.mask_for(
                state_every_tick
            ):
                raise SelectedTrialExportError(
                    "logged PPO action_mask_full12 disagrees with the frozen "
                    f"{state_every_tick} phase mask"
                )
            nominal_every_tick = _full12(
                command.get("nominal_full12", ()), "nominal_full12"
            )
            residual_every_tick = _full12(
                command.get("residual_full12", ()), "residual_full12"
            )
            logical_every_tick = _full12(command.get("full12", ()), "full12")
            commanded_every_tick = _full12(
                command.get("commanded_full12", ()), "commanded_full12"
            )
            applied_every_tick = _full12(
                command.get("applied_full12", ()), "applied_full12"
            )
            if any(value != 0.0 for value in residual_every_tick):
                raise SelectedTrialExportError(
                    "source Trial contains a non-zero residual"
                )
            if not all(
                bitwise_full12_equal(nominal_every_tick, value)
                for value in (
                    logical_every_tick,
                    commanded_every_tick,
                    applied_every_tick,
                )
            ):
                raise SelectedTrialExportError(
                    "source Trial logical/commanded/applied full12 is not bitwise "
                    "equal to nominal_full12"
                )
            full_tick_auditor.append(nominal_every_tick, applied_every_tick)
            if (command_tick - first_tick) % PHYSICS_TICKS_PER_DECISION == 0:
                sample = self._sample(command, observation)
                if pending is not None:
                    if sample.physics_tick != pending.physics_tick + PHYSICS_TICKS_PER_DECISION:
                        raise SelectedTrialExportError(
                            "15 Hz sample spacing is not exactly eight physics ticks"
                        )
                    self._append(
                        output,
                        metadata,
                        pending,
                        sample.observation,
                        sample.raw_observation,
                        sample.phase_progress,
                        control_tick,
                        final=False,
                    )
                    control_tick += 1
                pending = sample
            previous_command_tick = command_tick
            previous_command_time = command_time
            previous_observation_tick = observation_tick
            previous_observation_time = observation_time

        if pending is None or first_tick is None or previous_command_tick is None:
            raise SelectedTrialExportError("source ledgers contain no exportable command")
        try:
            terminal_observation = next(observation_rows)
        except StopIteration as exc:
            raise SelectedTrialExportError(
                "observation ledger lacks the terminal next observation"
            ) from exc
        terminal_tick = int(terminal_observation["physics_tick"])
        terminal_time = _clock(
            terminal_observation["simulation_time_s"],
            "terminal observation.simulation_time_s",
        )
        self._validate_step(
            tick=terminal_tick,
            time_s=terminal_time,
            previous_tick=previous_observation_tick,
            previous_time_s=previous_observation_time,
            label="observation ledger",
        )
        if terminal_tick != previous_command_tick + 1:
            raise SelectedTrialExportError(
                "terminal observation must immediately follow the final command"
            )
        if terminal_tick != pending.physics_tick + PHYSICS_TICKS_PER_DECISION:
            raise SelectedTrialExportError(
                "final 15 Hz interval is not exactly eight physics ticks"
            )
        try:
            extra = next(observation_rows)
        except StopIteration:
            extra = None
        if extra is not None:
            raise SelectedTrialExportError(
                "observation ledger has rows after the terminal next observation"
            )
        terminal_actor = self._terminal_actor(pending, terminal_observation)
        self._append(
            output,
            metadata,
            pending,
            terminal_actor,
            terminal_observation,
            pending.phase_progress,
            control_tick,
            final=True,
        )
        evidence = output.validate_baseline_equivalence()
        full_tick_evidence = full_tick_auditor.finalize()
        return SelectedTrialExportResult(
            source_trial=metadata.trial_id,
            transition_count=len(output.rows),
            command_row_count=command_reader.row_count,
            observation_row_count=observation_reader.row_count,
            first_physics_tick=first_tick,
            terminal_physics_tick=terminal_tick,
            command_ledger_sha256=command_reader.sha256.hexdigest(),
            observation_ledger_sha256=observation_reader.sha256.hexdigest(),
            full_physics_tick_zero_residual_equivalence=full_tick_evidence,
            zero_residual_equivalence=evidence,
            logger=output,
        )

    def export_artifacts(
        self,
        command_ledger: Path,
        observation_ledger: Path,
        *,
        metadata: SelectedTrialMetadata,
        output_directory: Path,
    ) -> SelectedTrialExportResult:
        """Write the baseline dataset and its standalone full-episode proof."""

        destination = Path(output_directory).resolve()
        jsonl_path = destination / "ppo_baseline_transitions.jsonl"
        parquet_path = destination / "ppo_baseline_transitions.parquet"
        manifest_path = destination / "ppo_baseline_dataset_manifest.json"
        zero_path = destination / "zero_residual_equivalence.json"
        existing = tuple(
            path
            for path in (jsonl_path, parquet_path, manifest_path, zero_path)
            if path.exists()
        )
        if existing:
            raise SelectedTrialExportError(
                "refusing to overwrite existing PPO baseline artifacts: "
                + ", ".join(path.name for path in existing)
            )
        result = self.export_to_logger(
            command_ledger,
            observation_ledger,
            metadata=metadata,
        )
        created: list[Path] = []
        try:
            # Parquet is attempted first so a missing optional dependency cannot
            # leave a JSONL file that looks like a complete artifact set.
            result.logger.write_parquet(parquet_path)
            created.append(parquet_path)
            result.logger.write_jsonl(jsonl_path)
            created.append(jsonl_path)
            zero_payload = {
                "schema": ZERO_RESIDUAL_EQUIVALENCE_SCHEMA,
                "status": (
                    result.full_physics_tick_zero_residual_equivalence.status
                ),
                "selected_trial_id": metadata.trial_id,
                "full_episode_checked": True,
                "source_120hz": {
                    **asdict(result.full_physics_tick_zero_residual_equivalence),
                    "residual_action_all_zero": True,
                    "command_row_count": result.command_row_count,
                    "first_physics_tick": result.first_physics_tick,
                    "terminal_observation_physics_tick": result.terminal_physics_tick,
                    "source": "every full12_commands_120hz.jsonl row",
                },
                "exported_15hz_transitions": asdict(
                    result.zero_residual_equivalence
                ),
                "zero_residual_action_full12": [0.0] * 12,
                "physics_hz": PHYSICS_HZ,
                "decision_hz": DECISION_HZ,
                "physics_ticks_per_transition": PHYSICS_TICKS_PER_DECISION,
                "ppo_training_started": False,
            }
            destination.mkdir(parents=True, exist_ok=True)
            zero_temporary = zero_path.with_name(zero_path.name + ".tmp")
            zero_temporary.write_text(
                json.dumps(zero_payload, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(zero_temporary, zero_path)
            created.append(zero_path)
            result.logger.write_manifest(
                manifest_path,
                source_trial=metadata.trial_id,
                jsonl_path=jsonl_path,
                parquet_path=parquet_path,
                claimed_zero_residual_equivalence_sha256=(
                    result.zero_residual_equivalence.nominal_sequence_sha256
                ),
            )
            created.append(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_120hz_zero_residual_equivalence"] = {
                **asdict(result.full_physics_tick_zero_residual_equivalence),
                "residual_action_all_zero": True,
                "source": "every full12_commands_120hz.jsonl row",
            }
            manifest["source_ledgers"] = {
                "command_120hz": {
                    "row_count": result.command_row_count,
                    "sha256": result.command_ledger_sha256,
                },
                "observation_120hz": {
                    "row_count": result.observation_row_count,
                    "sha256": result.observation_ledger_sha256,
                },
                "sampling": {
                    "physics_hz": PHYSICS_HZ,
                    "decision_hz": DECISION_HZ,
                    "physics_ticks_per_transition": PHYSICS_TICKS_PER_DECISION,
                },
            }
            manifest["files"]["zero_residual_equivalence"] = {
                "path": zero_path.name,
                "bytes": zero_path.stat().st_size,
                "sha256": hashlib.sha256(zero_path.read_bytes()).hexdigest(),
            }
            manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
            manifest_temporary.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(manifest_temporary, manifest_path)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            zero_path.with_name(zero_path.name + ".tmp").unlink(missing_ok=True)
            manifest_path.with_name(manifest_path.name + ".tmp").unlink(missing_ok=True)
            raise
        return result
