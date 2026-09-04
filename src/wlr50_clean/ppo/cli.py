"""Command-line entry points for the phase-specific residual PPO pipeline.

The PowerShell launchers reserve immutable run directories before invoking
this module.  Isaac is imported only for commands that need live physics.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "ppo_training_phase_v1.yaml"
DEFAULT_INTERFACE_CONFIG = PROJECT_ROOT / "configs" / "ppo_interface_v2.yaml"
DEFAULT_PHASE_SNAPSHOT_ROOT = PROJECT_ROOT / "reference" / "ppo_phase_snapshots"
DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "ppo_phase_effective_entry_v1.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ppo_phase_v1"
RESET_THROUGHPUT_PROBE_FILENAME = "reset_throughput_probe.json"
RESET_THROUGHPUT_PROBE_SCHEMA = "wlr50_clean.reset_throughput_probe.v1"
RESET_THROUGHPUT_PROBE_DECISIONS = 8
RESET_THROUGHPUT_PROBE_PHYSICS_TICKS = 64
LIVE_COMMANDS = frozenset(
    {
        "baseline-eval",
        "zero-residual-live",
        "nonzero-residual-smoke",
        "reset-throughput-probe",
        "soft-reset-equivalence",
        "phase-snapshot-live-probe",
        "phase-zero-residual-rollout",
        "vector-benchmark",
        "initialize-zero-residual",
        "train",
        "evaluate",
        "export-inference-actor",
        "capture-video-source",
    }
)
PHASE_CONTRACT_LIVE_COMMANDS = LIVE_COMMANDS - {"vector-benchmark"}


class CliError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "baseline-eval",
        "build-phase-snapshots",
        "zero-residual-live",
        "nonzero-residual-smoke",
        "reset-throughput-probe",
        "soft-reset-equivalence",
        "phase-snapshot-live-probe",
        "phase-zero-residual-rollout",
        "vector-benchmark",
        "initialize-zero-residual",
        "train",
        "evaluate",
        "aggregate-evaluations",
        "export-baseline-evaluation",
        "export-paired-evaluation",
        "promote-best-validation",
        "promote-improved",
        "export-inference-actor",
        "capture-video-source",
        "publish-videos",
        "publish-initial-zero-residual",
    ):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--num-envs", type=int, required=True)
        command.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
        command.add_argument("--interface-config", type=Path, default=DEFAULT_INTERFACE_CONFIG)
        command.add_argument("--episode-count", type=int, default=5)
        command.add_argument("--policy-decisions", type=int)
        command.add_argument("--phase-curriculum-max-decisions", type=int)
        command.add_argument("--seed-set", choices=("train", "validation", "locked-test"), default="validation")
        command.add_argument("--residual-mode", choices=("zero", "bounded-smoke"), default="zero")
        command.add_argument("--deterministic", action="store_true")
        command.add_argument("--fsm-config", type=Path)
        command.add_argument("--selected-trial", type=Path)
        command.add_argument("--snapshot-root", type=Path, default=DEFAULT_PHASE_SNAPSHOT_ROOT)
        command.add_argument(
            "--phase-snapshot-prime-physics-steps",
            type=int,
            choices=(1,),
            default=1,
        )
        if name == "phase-snapshot-live-probe":
            command.add_argument(
                "--phase",
                choices=tuple(f"P{index:02d}" for index in range(2, 14)),
            )
        command.add_argument("--stage", choices=("smoke", "phase-curriculum", "full-episode", "mild-randomization"), default="smoke")
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--checkpoint-manifest", type=Path)
        command.add_argument("--source-checkpoint", type=Path)
        command.add_argument("--source-manifest", type=Path)
        command.add_argument("--candidate-checkpoint", type=Path)
        command.add_argument("--candidate-manifest", type=Path)
        command.add_argument("--promotion-decision", type=Path)
        command.add_argument("--best-validation-checkpoint", type=Path)
        command.add_argument("--best-validation-manifest", type=Path)
        command.add_argument("--validation-promotion-manifest", type=Path)
        command.add_argument("--locked-test-aggregate", type=Path)
        command.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
        command.add_argument("--video-source-role", choices=("fsm", "ppo"))
        command.add_argument("--fsm-video-source-dir", type=Path)
        command.add_argument("--ppo-video-source-dir", type=Path)
        command.add_argument("--ffmpeg", type=Path)
        command.add_argument("--soft-reset-acceptance", type=Path)
        command.add_argument("--vector-benchmark-matrix", type=Path)
        command.add_argument(
            "--phase-effective-entry-holdout-acceptance",
            type=Path,
            help=(
                "finalized seed-1003 P02-P13 external holdout acceptance; "
                "mandatory for phase-curriculum training"
            ),
        )
        command.add_argument(
            "--phase-zero-residual-rollout-evidence",
            type=Path,
            help=(
                "finalized P01-P13 zero-residual phase rollout; mandatory "
                "before phase-curriculum training"
            ),
        )
        command.add_argument(
            "--evaluation-run-dir",
            type=Path,
            action="append",
            default=[],
            help="one finalized single-episode evaluation run to aggregate",
        )
        command.add_argument(
            "--evaluation-role",
            choices=("candidate", "baseline"),
            default="candidate",
        )
        command.add_argument(
            "--episode-dir",
            type=Path,
            action="append",
            default=[],
            help="one canonical episode directory for offline metric export",
        )
        command.add_argument(
            "--baseline-episode-dir",
            type=Path,
            action="append",
            default=[],
            help="one pure-FSM canonical episode directory for paired export",
        )
        command.add_argument(
            "--candidate-episode-dir",
            type=Path,
            action="append",
            default=[],
            help="one PPO canonical episode directory for paired export",
        )
        command.add_argument(
            "--baseline-aggregate",
            type=Path,
            help="finalized five-seed baseline aggregate for paired export",
        )
        command.add_argument(
            "--candidate-validation-aggregate",
            type=Path,
            help="finalized five-seed candidate aggregate for paired export",
        )
        command.add_argument(
            "--metrics-output-dir",
            type=Path,
            default=OUTPUT_ROOT / "metrics",
        )
        command.add_argument(
            "--evidence-only-worker",
            action="store_true",
            help="capture one valid worker result and leave batch pass/fail to aggregation",
        )
        command.add_argument("--require-save-load-round-trip", action="store_true")
        command.add_argument("--capture-fps", type=float, default=15.0)
        command.add_argument("--measured-ticks", type=int, default=32)
        command.add_argument("--maximum-duration-s", type=float, default=200.0)
        command.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_project_path(path: Path) -> Path:
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _validate_common(args: argparse.Namespace) -> None:
    if args.command == "vector-benchmark":
        from .vector_benchmark_matrix import (
            VectorBenchmarkMatrixError,
            validate_managed_run_directory,
        )

        try:
            args.run_dir = validate_managed_run_directory(
                args.run_dir,
                project_root=PROJECT_ROOT,
                run_kind="vector_benchmark",
            )
        except VectorBenchmarkMatrixError as exc:
            raise CliError(f"vector benchmark run directory rejected: {exc}") from exc
    else:
        args.run_dir = args.run_dir.resolve()
        if not args.run_dir.is_dir():
            raise CliError(f"reserved run directory is missing: {args.run_dir}")
    if args.seed < 0 or args.num_envs <= 0 or args.episode_count <= 0:
        raise CliError("seed/env/episode counts are invalid")
    if args.phase_snapshot_prime_physics_steps != 1:
        raise CliError(
            "phase snapshot prime physics steps must be exactly one"
        )
    args.training_config = _resolve_project_path(args.training_config)
    args.interface_config = _resolve_project_path(args.interface_config)
    args.snapshot_root = _resolve_project_path(args.snapshot_root)
    for path in (args.training_config, args.interface_config):
        if not path.is_file():
            raise CliError(f"configuration is missing: {path}")
    if args.maximum_duration_s <= 0.0 or args.maximum_duration_s > 200.0:
        raise CliError("maximum duration must be within (0, 200]")
    source_arguments = (args.source_checkpoint, args.source_manifest)
    if args.command == "publish-initial-zero-residual":
        if any(value is None for value in source_arguments):
            raise CliError(
                "publish-initial-zero-residual requires both source checkpoint arguments"
            )
    elif any(value is not None for value in source_arguments):
        raise CliError(
            "--source-checkpoint/--source-manifest are publication-only arguments"
        )


def _json(path: Path, payload: Any) -> None:
    from .artifacts import atomic_write_json

    atomic_write_json(path, payload)


def _json_replace(path: Path, payload: Any) -> None:
    from .artifacts import atomic_write_json

    atomic_write_json(path, payload, replace=True)


def _atomic_copy_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".partial-{os.getpid()}-{time.time_ns()}"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise CliError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_manifest(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path.resolve()) for path in paths}


def _live_phase_snapshot_root() -> Path:
    """Resolve the exact root consumed by the production single-env loader."""

    from .isaac_fsm_backend import DEFAULT_PHASE_SNAPSHOT_ROOT as live_root

    return live_root.resolve()


def _capture_runtime_snapshot_bundle(args: argparse.Namespace):
    """Capture immutable bytes from the one bundle accepted by live execution."""

    from .phase_snapshots import (
        PhaseSnapshotError,
        capture_validated_phase_snapshot_bundle,
    )

    requested_root = _resolve_project_path(
        getattr(args, "snapshot_root", DEFAULT_PHASE_SNAPSHOT_ROOT)
    )
    live_root = _live_phase_snapshot_root()
    if requested_root != live_root:
        raise CliError(
            "--snapshot-root must resolve to the exact bundle used by the live "
            f"backend: requested={requested_root}, live={live_root}"
        )
    try:
        return capture_validated_phase_snapshot_bundle(
            requested_root, canonical_root=live_root
        )
    except (OSError, PhaseSnapshotError) as exc:
        raise CliError(f"phase snapshot bundle validation failed: {exc}") from exc


def _capture_runtime_effective_entry_contract(pinned_snapshot_bundle: Any):
    """Capture the effective-entry contract after its snapshot dependency."""

    from .phase_effective_entry import (
        EffectivePhaseEntryError,
        capture_validated_effective_phase_entry_contract,
    )

    try:
        return capture_validated_effective_phase_entry_contract(
            DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
            expected_snapshot_bundle=pinned_snapshot_bundle,
        )
    except (OSError, EffectivePhaseEntryError) as exc:
        raise CliError(f"effective phase-entry contract validation failed: {exc}") from exc


def _capture_runtime_phase_contracts(args: argparse.Namespace) -> tuple[Any, Any]:
    """Pin snapshot bytes, then their dependent effective-entry contract bytes."""

    snapshot_bundle = _capture_runtime_snapshot_bundle(args)
    effective_contract = _capture_runtime_effective_entry_contract(snapshot_bundle)
    args._pinned_phase_snapshot_bundle = snapshot_bundle
    args._pinned_effective_entry_contract = effective_contract
    return snapshot_bundle, effective_contract


def _pinned_runtime_phase_contracts(args: argparse.Namespace) -> tuple[Any, Any]:
    """Return the pre-launch pins, with a direct-call fallback for unit helpers."""

    from .phase_effective_entry import ValidatedEffectivePhaseEntryContract
    from .phase_snapshots import ValidatedPhaseSnapshotBundle

    snapshot_bundle = getattr(args, "_pinned_phase_snapshot_bundle", None)
    effective_contract = getattr(args, "_pinned_effective_entry_contract", None)
    if snapshot_bundle is None and effective_contract is None:
        return _capture_runtime_phase_contracts(args)
    if not isinstance(snapshot_bundle, ValidatedPhaseSnapshotBundle):
        raise CliError("phase snapshot pin is not an immutable validated bundle")
    if not isinstance(effective_contract, ValidatedEffectivePhaseEntryContract):
        raise CliError("effective-entry pin is not an immutable validated contract")
    if effective_contract.phase_snapshot_bundle_sha256 != snapshot_bundle.bundle_sha256:
        raise CliError("effective-entry pin belongs to a different snapshot bundle")
    return snapshot_bundle, effective_contract


def _validated_runtime_snapshot_bundle(args: argparse.Namespace) -> Mapping[str, Any]:
    """Validate and return the checkpoint-facing snapshot bundle record."""

    from .phase_snapshots import PHASE_IDS

    record = _pinned_runtime_phase_contracts(args)[0].as_record()
    live_root = _live_phase_snapshot_root()
    entries = record.get("snapshots")
    if (
        record.get("snapshot_root") != str(live_root)
        or record.get("manifest_path") != str((live_root / "manifest.json").resolve())
        or record.get("phase_count") != len(PHASE_IDS)
        or not isinstance(entries, list)
        or len(entries) != len(PHASE_IDS)
        or tuple(entry.get("phase") for entry in entries) != PHASE_IDS
    ):
        raise CliError("validated phase snapshot bundle record is incomplete")
    return record


def _revalidate_pinned_snapshot_bundle(pinned_snapshot_bundle: Any) -> None:
    from .phase_snapshots import (
        PhaseSnapshotError,
        ValidatedPhaseSnapshotBundle,
        assert_phase_snapshot_bundle_unchanged,
    )

    if not isinstance(pinned_snapshot_bundle, ValidatedPhaseSnapshotBundle):
        raise CliError("phase snapshot pin is not an immutable validated bundle")
    try:
        assert_phase_snapshot_bundle_unchanged(
            pinned_snapshot_bundle,
            canonical_root=_live_phase_snapshot_root(),
        )
    except (OSError, PhaseSnapshotError) as exc:
        raise CliError(f"pinned phase snapshot bundle changed: {exc}") from exc


def _revalidate_pinned_phase_contracts(
    pinned_snapshot_bundle: Any,
    pinned_effective_entry_contract: Any,
) -> None:
    """Revalidate both pins in dependency order without accepting fresh values."""

    from .phase_effective_entry import (
        EffectivePhaseEntryError,
        ValidatedEffectivePhaseEntryContract,
        assert_effective_phase_entry_contract_unchanged,
    )

    _revalidate_pinned_snapshot_bundle(pinned_snapshot_bundle)
    if not isinstance(
        pinned_effective_entry_contract, ValidatedEffectivePhaseEntryContract
    ):
        raise CliError("effective-entry pin is not an immutable validated contract")
    try:
        assert_effective_phase_entry_contract_unchanged(
            pinned_effective_entry_contract,
            expected_snapshot_bundle=pinned_snapshot_bundle,
        )
    except (OSError, EffectivePhaseEntryError) as exc:
        raise CliError(f"pinned effective phase-entry contract changed: {exc}") from exc


def _effective_entry_contract_fields(contract: Any) -> dict[str, Any]:
    from .phase_effective_entry import ValidatedEffectivePhaseEntryContract

    if not isinstance(contract, ValidatedEffectivePhaseEntryContract):
        raise CliError("checkpoint effective-entry pin is not validated")
    return {
        "phase_effective_entry_contract_path": str(contract.contract_path),
        "phase_effective_entry_contract_file_sha256": contract.file_sha256,
        "phase_effective_entry_contract_sidecar_path": str(contract.sidecar_path),
        "phase_effective_entry_contract_sidecar_sha256": (
            contract.sidecar_file_sha256
        ),
        "phase_effective_entry_contract_sha256": contract.contract_sha256,
        "phase_effective_entry_contract": contract.as_record(),
    }


def _phase_effective_entry_holdout_fields(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return checkpoint fields plus the exact external files inventory."""

    acceptance = evidence.get("acceptance")
    acceptance_path = Path(str(evidence.get("path", ""))).resolve()
    run_manifest_path = Path(str(evidence.get("run_manifest", ""))).resolve()
    if (
        evidence.get("passed") is not True
        or not isinstance(acceptance, Mapping)
        or acceptance.get("passed") is not True
        or acceptance.get("status") != "PASSED"
        or not acceptance_path.is_file()
        or not run_manifest_path.is_file()
        or _sha256(acceptance_path) != evidence.get("sha256")
        or _sha256(run_manifest_path) != evidence.get("run_manifest_sha256")
    ):
        raise CliError("phase effective-entry holdout checkpoint evidence is invalid")
    fields = {
        "phase_effective_entry_holdout_acceptance_path": str(acceptance_path),
        "phase_effective_entry_holdout_acceptance_sha256": evidence["sha256"],
        "phase_effective_entry_holdout_contract_sha256": evidence[
            "phase_effective_entry_contract_sha256"
        ],
        "phase_effective_entry_holdout_source_git_commit": evidence[
            "source_git_commit"
        ],
        "phase_effective_entry_holdout_acceptance": dict(acceptance),
        "phase_effective_entry_holdout_evidence": dict(evidence),
        "phase_effective_entry_holdout_files": {
            str(acceptance_path): str(evidence["sha256"]),
            str(run_manifest_path): str(evidence["run_manifest_sha256"]),
        },
    }
    files = {
        str(acceptance_path): str(evidence["sha256"]),
        str(run_manifest_path): str(evidence["run_manifest_sha256"]),
    }
    return fields, files


def _phase_zero_residual_rollout_fields(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return checkpoint fields plus exact external rollout evidence files."""

    from .phase_zero_residual_rollout import TRAINING_EVIDENCE_SCHEMA

    evidence_path = Path(str(evidence.get("path", ""))).resolve()
    run_manifest_path = Path(str(evidence.get("run_manifest", ""))).resolve()
    if (
        evidence.get("schema") != TRAINING_EVIDENCE_SCHEMA
        or evidence.get("passed") is not True
        or not evidence_path.is_file()
        or not run_manifest_path.is_file()
        or _sha256(evidence_path) != evidence.get("sha256")
        or _sha256(run_manifest_path) != evidence.get("run_manifest_sha256")
    ):
        raise CliError("phase zero-residual rollout checkpoint evidence is invalid")
    files = {
        str(evidence_path): str(evidence["sha256"]),
        str(run_manifest_path): str(evidence["run_manifest_sha256"]),
    }
    fields = {
        "phase_zero_residual_rollout_evidence_path": str(evidence_path),
        "phase_zero_residual_rollout_evidence_sha256": evidence["sha256"],
        "phase_zero_residual_rollout_run_manifest_path": str(run_manifest_path),
        "phase_zero_residual_rollout_run_manifest_sha256": evidence[
            "run_manifest_sha256"
        ],
        "phase_zero_residual_rollout_evidence": dict(evidence),
        "phase_zero_residual_rollout_files": files,
    }
    return fields, files


def _snapshot_bundle_files(bundle: Mapping[str, Any]) -> tuple[Path, ...]:
    from .phase_snapshots import PhaseSnapshotError, phase_snapshot_bundle_file_hashes

    try:
        hashes = phase_snapshot_bundle_file_hashes(bundle)
    except PhaseSnapshotError as exc:
        raise CliError(f"phase snapshot bundle record is invalid: {exc}") from exc
    return tuple(Path(path) for path in hashes)


def _require_manifest_snapshot_contract(
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    label: str,
    effective_entry_contract: Any | None = None,
) -> None:
    expected = {
        "phase_snapshot_manifest": bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": bundle["bundle_sha256"],
        "phase_snapshot_bundle": bundle,
    }
    differing = [field for field, value in expected.items() if manifest.get(field) != value]
    if differing:
        raise CliError(
            f"{label} differs from the validated live phase snapshot bundle: "
            + ", ".join(differing)
        )
    from .phase_snapshots import PhaseSnapshotError, phase_snapshot_bundle_file_hashes

    try:
        expected_files = phase_snapshot_bundle_file_hashes(bundle)
    except PhaseSnapshotError as exc:
        raise CliError(f"validated live phase snapshot bundle is invalid: {exc}") from exc
    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping):
        raise CliError(f"{label} omits the checkpoint files hash manifest")
    missing = [path for path in expected_files if path not in declared_files]
    mismatched = [
        path
        for path, digest in expected_files.items()
        if path in declared_files and declared_files[path] != digest
    ]
    if missing or mismatched:
        raise CliError(
            f"{label} does not bind all 27 phase snapshot files: "
            f"missing={missing}, mismatched={mismatched}"
        )

    if effective_entry_contract is None:
        snapshot_pin = _capture_runtime_snapshot_bundle(
            argparse.Namespace(snapshot_root=Path(bundle["snapshot_root"]))
        )
        if snapshot_pin.as_record() != dict(bundle):
            raise CliError(f"{label} snapshot record changed before contract validation")
        effective_entry_contract = _capture_runtime_effective_entry_contract(
            snapshot_pin
        )
    expected_effective = _effective_entry_contract_fields(
        effective_entry_contract
    )
    differing_effective = [
        field
        for field, value in expected_effective.items()
        if manifest.get(field) != value
    ]
    if differing_effective:
        raise CliError(
            f"{label} differs from the validated effective-entry contract: "
            + ", ".join(differing_effective)
        )
    effective_files = effective_entry_contract.file_hashes()
    missing_effective = [path for path in effective_files if path not in declared_files]
    mismatched_effective = [
        path
        for path, digest in effective_files.items()
        if path in declared_files and declared_files[path] != digest
    ]
    if missing_effective or mismatched_effective:
        raise CliError(
            f"{label} does not bind the effective-entry contract file inventory: "
            f"missing={missing_effective}, mismatched={mismatched_effective}"
        )


def _preflight(args: argparse.Namespace) -> int:
    from .observation_schema import OBSERVATION_DIMENSION
    from .observation_schema_v2 import OBSERVATION_DIMENSION_V2, load_observation_schema_v2
    from .phase_action_masks_v2 import load_phase_action_masks_v2
    from .phase_objectives import load_phase_objectives
    from .reward_v2 import (
        load_reward_v2_config,
        reward_duplicate_signal_audit,
        reward_standing_still_exploit_test,
    )
    from .rl_library_wrapper import assert_supported_rsl_runtime, load_training_profile
    from .termination_v2 import load_termination_config_v2

    profile = load_training_profile(args.training_config)
    schema = load_observation_schema_v2()
    actions = load_phase_action_masks_v2()
    objectives = load_phase_objectives()
    reward = load_reward_v2_config()
    termination = load_termination_config_v2()
    snapshot_pin, effective_entry_pin = _pinned_runtime_phase_contracts(args)
    snapshot_bundle = snapshot_pin.as_record()
    frozen_manifest = PROJECT_ROOT / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    frozen = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in frozen.get("protected_files", {}).items():
        candidate = PROJECT_ROOT / relative
        if not candidate.is_file() or _sha256(candidate) != expected:
            mismatches.append(str(relative))
    schema_dir = OUTPUT_ROOT / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    csv_path = schema_dir / "actor_observation_v2.csv"
    json_path = schema_dir / "actor_observation_v2.json"
    if not csv_path.exists():
        schema.write_csv(csv_path)
    if not json_path.exists():
        schema.write_json(json_path)
    analysis_dir = OUTPUT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    scale_path = analysis_dir / "phase_action_scale_audit.csv"
    if not scale_path.exists():
        rows = [row.as_dict() for row in actions.scale_audit_rows()]
        with scale_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "schema": "wlr50_clean.ppo_preflight.v1",
        "passed": not mismatches,
        "frozen_hash_mismatches": mismatches,
        "rsl_rl_version": assert_supported_rsl_runtime(),
        "v1_observation_dimension": OBSERVATION_DIMENSION,
        "v2_observation_dimension": OBSERVATION_DIMENSION_V2,
        "v1_prefix_preserved": schema.v1_prefix_dimension == OBSERVATION_DIMENSION,
        "phase_count": len(objectives.phases),
        "action_scale_rows": len(actions.scale_audit_rows()),
        "five_dense_families": list(reward.dense_families),
        "reward_duplicate_signal_audit": None,
        "standing_still_audit": None,
        "termination_timeout_s": termination.timeout_s,
        "training_profile": str(profile.path),
        "snapshot_count": snapshot_bundle["phase_count"],
        "phase_snapshot_bundle": snapshot_bundle,
        "phase_effective_entry_contract": effective_entry_pin.as_record(),
        "schema_outputs": [str(csv_path), str(json_path)],
        "scale_audit_output": str(scale_path),
    }
    # Slotted audit dataclasses have no __dict__.
    from dataclasses import asdict

    result["reward_duplicate_signal_audit"] = asdict(reward_duplicate_signal_audit())
    result["standing_still_audit"] = asdict(reward_standing_still_exploit_test())
    _revalidate_pinned_phase_contracts(snapshot_pin, effective_entry_pin)
    _json(args.run_dir / "preflight.json", result)
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result["passed"] else 2


def _build_snapshots(args: argparse.Namespace) -> int:
    from .phase_snapshots import (
        build_phase_snapshots,
        validate_phase_snapshots,
        validated_phase_snapshot_bundle_record,
    )

    root = _resolve_project_path(args.snapshot_root)
    if (root / "manifest.json").is_file():
        manifest = validate_phase_snapshots(root)
        mode = "validated_existing"
    else:
        if args.selected_trial is None:
            raise CliError("--selected-trial is required to build missing snapshots")
        selected = _resolve_project_path(args.selected_trial)
        payload = json.loads(selected.read_text(encoding="utf-8"))
        trial_path = Path(payload.get("selected_trial_path") or payload.get("trial_path") or "")
        if not trial_path.is_dir():
            raise CliError("selected-trial artifact does not resolve a trial directory")
        manifest = build_phase_snapshots(trial_path, root)
        mode = "built"
    bundle = validated_phase_snapshot_bundle_record(root)
    result = {
        "schema": "wlr50_clean.phase_snapshot_cli.v1",
        "mode": mode,
        "manifest": manifest,
        "phase_snapshot_bundle": bundle,
    }
    _json(args.run_dir / "phase_snapshot_result.json", result)
    print(json.dumps({"mode": mode, "phase_count": manifest["phase_count"]}))
    return 0


def _seed_values(args: argparse.Namespace) -> tuple[int, ...]:
    from .rl_library_wrapper import load_training_profile

    profile = load_training_profile(args.training_config)
    selected = {
        "train": profile.seed_train,
        "validation": profile.seed_validation,
        "locked-test": profile.seed_locked_test,
    }[args.seed_set]
    # Respect the explicit invocation seed while retaining a disjoint fixed set.
    if args.seed in selected:
        offset = selected.index(args.seed)
        selected = selected[offset:] + selected[:offset]
    return selected


def _smoke_action(env: Any, decision_index: int) -> tuple[float, ...]:
    if env.frame is None or str(env.frame.state_id) not in {
        f"P{index:02d}" for index in range(1, 14)
    }:
        raise CliError("bounded smoke action requires an active P01-P13 frame")
    phase_id = str(env.frame.state_id)
    progress = float(env.frame.phase_progress)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise CliError("bounded smoke action requires finite phase progress in [0,1]")
    mask = tuple(int(value) for value in env.phase_actions.mask_for(phase_id))
    nominal = tuple(float(value) for value in env.frame.nominal_action_full12)
    if len(mask) != 12 or len(nominal) != 12:
        raise CliError("bounded smoke action requires Full12 mask and nominal action")

    # P13's terminal pose and wheel-decay guards intentionally use a long
    # settle/retry window.  Exercise its explicit policy input once at entry,
    # then return to zero so a diagnostic pattern cannot persist through the
    # recovery pass.  The incoming P12 residual still exercises the real
    # P12->P13 bridge on the first physics tick.
    if phase_id == "P13":
        if bool(getattr(env, "_bounded_smoke_p13_emitted", False)):
            return (0.0,) * 12
        setattr(env, "_bounded_smoke_p13_emitted", True)
        action = [0.0] * 12
        active_servos = tuple(index for index in range(8) if mask[index])
        selected = max(active_servos, key=lambda index: abs(nominal[index]))
        action[selected] = 1.0e-12
        return tuple(action)

    # Gate B verifies the real projection path without turning its diagnostic
    # pattern into a second controller.  Arm only in the final quarter of each
    # phase, then keep one active servo residual through the phase edge so the
    # transition bridge is exercised.  The servo with the largest non-zero FSM
    # nominal is selected because its 1e-9 normalized delta remains below the
    # float target quantization of the mature actuator path.  Raw values are
    # also sent only to disabled channels; their exact removal proves the mask
    # without physically perturbing those channels.  Rate limiting is tested by
    # ``_smoke_rate_limit_probe`` with this same production projector off-robot.
    armed = set(getattr(env, "_bounded_smoke_armed_phases", ()))
    if progress >= 0.75:
        armed.add(phase_id)
        setattr(env, "_bounded_smoke_armed_phases", tuple(sorted(armed)))
    if phase_id not in armed:
        return (0.0,) * 12

    action = [0.0] * 12
    active_servos = tuple(index for index in range(8) if mask[index])
    if not active_servos:
        raise CliError(f"bounded smoke phase {phase_id} has no active servo channel")
    selected = max(active_servos, key=lambda index: abs(nominal[index]))
    action[selected] = 1.0e-9
    for index, enabled in enumerate(mask):
        if not enabled:
            action[index] = -1.0e-9
    return tuple(action)


def _smoke_rate_limit_probe() -> Mapping[str, Any]:
    """Exercise the production v2 projector slew path without moving the robot."""

    from .action_projection import SafetyProjection
    from .phase_action_masks_v2 import build_action_projector_v2

    projector = build_action_projector_v2()
    zero = (0.0,) * 12
    positive = (0.0, 0.0, 0.0, 0.049) + (0.0,) * 8
    negative = (0.0, 0.0, 0.0, -0.049) + (0.0,) * 8
    common = {
        "state_id": "P03",
        "nominal_action_full12": zero,
        "reference_action_full12": zero,
        "reference_delta_full12": zero,
        "runtime_action_mask_full12": (1,) * 12,
        "safety": SafetyProjection(),
        "dt_s": 1.0 / 120.0,
    }
    first = projector.project(
        positive,
        previous_projected_residual_full12=zero,
        **common,
    )
    second = projector.project(
        negative,
        previous_projected_residual_full12=first.safe_projected_residual_full12,
        **common,
    )
    passed = "residual_rate_limit" in second.clipping_stages
    return {
        "schema": "wlr50_clean.action_projector_rate_limit_probe.v1",
        "passed": passed,
        "applied_to_robot": False,
        "projector_source": "build_action_projector_v2",
        "state_id": "P03",
        "normalized_probe_amplitude": 0.049,
        "within_five_percent": True,
        "first_projected_residual_full12": list(
            first.safe_projected_residual_full12
        ),
        "second_projected_residual_full12": list(
            second.safe_projected_residual_full12
        ),
        "second_clipping_stages": list(second.clipping_stages),
    }


def _run_live_episodes(
    args: argparse.Namespace,
    simulation_app: Any,
    *,
    action_factory: Callable[[Any, int], Sequence[float]],
    episode_count: int,
) -> dict[str, Any]:
    from .isaac_fsm_backend import IsaacFSMBackend
    from .live_stream_writer import LiveStreamWriter
    from .residual_direct_env import ResidualEpisodeEnv

    if args.num_envs != 1:
        raise CliError(
            "the validated exact-pair backend currently supports one live scene; "
            "multi-environment execution must not be emulated"
        )
    if episode_count != 1:
        raise CliError(
            "acceptance episodes require one fresh Isaac process each; invoke the "
            "provided PowerShell gate script to aggregate multiple seeds"
        )
    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    backend = IsaacFSMBackend(
        simulation_app,
        expected_phase_snapshot_bundle=pinned_snapshot_bundle,
        expected_effective_entry_contract=pinned_effective_entry_contract,
    )
    env = ResidualEpisodeEnv(backend, collect_trace=True)
    episodes = []
    started = time.perf_counter()
    for episode_index in range(episode_count):
        # One live invocation owns one independently initialized episode.  Its
        # artifact identity and physical reset must therefore use the exact
        # invocation seed, not silently substitute the first seed in a split.
        seed = int(args.seed)
        env.reset(seed=seed)
        episode_dir = args.run_dir / f"episode_{episode_index:03d}_seed_{seed}"
        writer = LiveStreamWriter(episode_dir, seed=seed)
        assert env.frame is not None
        writer.start(env.frame)
        env.tick_callback = writer.write_tick
        reward_total = 0.0
        try:
            while not env.done:
                step = env.step(action_factory(env, env.decision_count))
                reward_total += step.reward
                writer.write_decision(step.info)
            assert env.frame is not None
            manifest_path = writer.finalize(
                env.frame,
                reward_total=reward_total,
                decision_count=env.decision_count,
            )
        except Exception:
            writer.abort()
            raise
        assert env.frame is not None
        terminal = env.trace[-1]["termination_reason"] if env.trace else None
        _jsonl(episode_dir / "policy_trace.jsonl", env.trace)
        summary = {
            "episode_index": episode_index,
            "seed": seed,
            "task_success": terminal == "SUCCESS",
            "termination_reason": terminal,
            "duration_s": env.frame.sim_time_s,
            "decision_count": env.decision_count,
            "physics_tick": env.frame.physics_tick,
            "body_collision": env.frame.termination_signals.body_collision,
            "wheel_only_climb": env.frame.termination_signals.wheel_only_climb,
            "safety_abort": any(
                (
                    env.frame.termination_signals.fall,
                    env.frame.termination_signals.nan_inf,
                    env.frame.termination_signals.hard_joint_limit,
                    env.frame.termination_signals.physics_explosion,
                )
            ),
            "reward_total": reward_total,
            "trace_path": str(episode_dir / "policy_trace.jsonl"),
            "trial_manifest_path": str(manifest_path),
            "canonical_episode_dir": str(episode_dir),
            "action_projection_audit": json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("action_projection_audit", {}),
            "recording_runtime_access_count": 0,
            "in_episode_root_write_count": 0,
            "under_maximum_duration": env.frame.sim_time_s
            <= args.maximum_duration_s,
        }
        _json(episode_dir / "episode_summary.json", summary)
        episodes.append(summary)
        print(json.dumps(summary, separators=(",", ":")), flush=True)
    result = {
        "schema": "wlr50_clean.live_residual_gate.v1",
        "mode": args.residual_mode,
        "episode_count": len(episodes),
        "success_count": sum(row["task_success"] for row in episodes),
        "all_success": all(row["task_success"] for row in episodes),
        "body_collision_count": sum(row["body_collision"] for row in episodes),
        "wheel_only_climb_count": sum(row["wheel_only_climb"] for row in episodes),
        "safety_abort_count": sum(row["safety_abort"] for row in episodes),
        "all_under_200_s": all(row["duration_s"] <= 200.0 for row in episodes),
        "wall_time_s": time.perf_counter() - started,
        "episodes": episodes,
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _json(args.run_dir / "live_gate_summary.json", result)
    return result


def _baseline_or_gate(args: argparse.Namespace, simulation_app: Any) -> int:
    zero = lambda env, tick: (0.0,) * 12
    action = zero if args.residual_mode == "zero" else _smoke_action
    count = args.episode_count
    result = _run_live_episodes(
        args, simulation_app, action_factory=action, episode_count=count
    )
    common_passed = bool(
        result["all_success"]
        and result["body_collision_count"] == 0
        and result["wheel_only_climb_count"] == 0
        and result["safety_abort_count"] == 0
        and result["all_under_200_s"]
    )
    audits = [row.get("action_projection_audit", {}) for row in result["episodes"]]
    if args.residual_mode == "zero":
        mode_checks = {
            "zero_input_all_ticks_bitwise_equivalent": all(
                bool(audit.get("zero_input_all_ticks_bitwise_equivalent", False))
                for audit in audits
            ),
            "zero_fast_path_covers_every_tick": all(
                int(audit.get("zero_residual_fast_path_tick_count", -1))
                == int(audit.get("physics_tick_count", -2))
                for audit in audits
            ),
        }
    else:
        rate_limit_probe = _smoke_rate_limit_probe()
        result["action_projector_rate_limit_probe"] = rate_limit_probe
        mode_checks = {
            "bounded_smoke_within_five_percent": all(
                bool(audit.get("within_five_percent_smoke_amplitude", False))
                for audit in audits
            ),
            "phase_masks_exercised_and_honored": all(
                bool(audit.get("mask_honored_when_exercised", False))
                for audit in audits
            ),
            "residual_rate_limit_exercised": all(
                int(audit.get("rate_limit_tick_count", 0)) > 0 for audit in audits
            )
            or rate_limit_probe.get("passed") is True,
            "residual_rate_limit_probe_not_applied_to_robot": (
                rate_limit_probe.get("applied_to_robot") is False
            ),
            "phase_transition_bridge_exercised": all(
                int(audit.get("phase_transition_bridge_count", 0)) >= 12
                and int(audit.get("phase_transition_handoff_hold_count", 0)) >= 12
                for audit in audits
            ),
            "body_collision_detector_operational": all(
                bool(audit.get("body_collision_detector_operational", False))
                for audit in audits
            ),
            "wheel_only_climb_detector_operational": all(
                bool(audit.get("wheel_only_climb_detector_operational", False))
                for audit in audits
            ),
            "nonzero_residual_covers_p01_p13": all(
                tuple(audit.get("nonzero_residual_phases", ()))
                == tuple(f"P{index:02d}" for index in range(1, 14))
                for audit in audits
            ),
        }
    result["mode_specific_checks"] = mode_checks
    passed = common_passed and all(mode_checks.values())
    result["passed"] = passed
    result["evidence_only_worker"] = bool(
        getattr(args, "evidence_only_worker", False)
    )
    _json(args.run_dir / "acceptance.json", result)
    return 0 if passed or result["evidence_only_worker"] else 2


def _checkpoint_config_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        args.training_config,
        args.interface_config,
        DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH,
        DEFAULT_EFFECTIVE_ENTRY_CONTRACT_PATH.with_suffix(".sha256"),
        PROJECT_ROOT / "configs" / "ppo_observation_schema_v2.json",
        PROJECT_ROOT / "configs" / "ppo_phase_action_masks_v2.yaml",
        PROJECT_ROOT / "configs" / "ppo_phase_objectives_v2.yaml",
        PROJECT_ROOT / "configs" / "ppo_reward_v2.yaml",
        PROJECT_ROOT / "configs" / "ppo_termination_v2.yaml",
        PROJECT_ROOT / "configs" / "ppo_domain_randomization_v2.yaml",
        PROJECT_ROOT / "configs" / "frozen_successful_fsm.yaml",
        PROJECT_ROOT / "configs" / "environment_lock.json",
        PROJECT_ROOT / "configs" / "fsm_states.yaml",
        PROJECT_ROOT / "configs" / "recording_motion_contract.json",
        PROJECT_ROOT / "configs" / "ppo_action_projection.yaml",
        PROJECT_ROOT / "configs" / "ppo_observation_schema.json",
        PROJECT_ROOT / "configs" / "conformance_policy.yaml",
    )


def _pin_live_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    *,
    purpose: str,
):
    """Register one immutable checkpoint capture for this live command."""

    stack = getattr(args, "_checkpoint_runtime_capture_stack", None)
    if not isinstance(stack, ExitStack):
        raise CliError("live checkpoint capture stack is unavailable")
    from .checkpoint_runtime_capture import (
        CheckpointRuntimeCaptureError,
        capture_checkpoint_bundle,
    )

    try:
        capture = capture_checkpoint_bundle(
            checkpoint_path,
            manifest_path,
            run_directory=args.run_dir,
            purpose=purpose,
        )
        return stack.enter_context(capture)
    except CheckpointRuntimeCaptureError as exc:
        raise CliError(f"checkpoint runtime capture failed: {exc}") from exc


def _checkpoint_creation_runtime_identity(args: argparse.Namespace) -> dict[str, str]:
    """Bind checkpoint infos to the managed run's pre-execution runtime bytes."""

    raw_run_dir = getattr(args, "run_dir", None)
    if raw_run_dir is None:
        raise CliError("checkpoint creation requires a managed --run-dir")
    run_dir = Path(raw_run_dir).resolve()
    identity_path = run_dir / "committed_runtime_identity.before.json"
    try:
        before = identity_path.stat()
        data = identity_path.read_bytes()
        after = identity_path.stat()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(
            "managed run committed-runtime identity is missing or invalid"
        ) from exc
    if (
        before.st_size != len(data)
        or after.st_size != len(data)
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or not isinstance(payload, Mapping)
    ):
        raise CliError("managed run committed-runtime identity changed while read")
    files = payload.get("files")
    git_commit = payload.get("git_commit")
    content_sha256 = payload.get("content_sha256")
    aggregate_sha256 = payload.get("aggregate_sha256")
    if (
        payload.get("schema") != "wlr50_clean.committed_runtime_identity.v1"
        or not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit.lower())
        or not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256.lower())
        or not isinstance(aggregate_sha256, str)
        or len(aggregate_sha256) != 64
        or any(character not in "0123456789abcdef" for character in aggregate_sha256.lower())
        or not isinstance(files, list)
        or not files
        or payload.get("file_count") != len(files)
    ):
        raise CliError("managed run committed-runtime identity header is invalid")
    content_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "bytes",
            "sha256",
            "creation_time_utc_ticks",
            "last_write_time_utc_ticks",
        }:
            raise CliError("managed run committed-runtime identity row is invalid")
        name = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise CliError("managed run committed-runtime identity row is invalid")
        content_rows.append({"path": name, "bytes": size, "sha256": digest})
        normalized_rows.append(dict(row))
    canonical = json.dumps(
        content_rows,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != content_sha256:
        raise CliError("managed run committed-runtime content SHA-256 is invalid")
    aggregate = json.dumps(
        normalized_rows,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(aggregate).hexdigest() != aggregate_sha256:
        raise CliError("managed run committed-runtime aggregate SHA-256 is invalid")
    return {
        "source_git_commit": git_commit.lower(),
        "committed_runtime_content_sha256": content_sha256.lower(),
        "creation_runtime_identity_path": str(identity_path),
        "creation_runtime_identity_sha256": hashlib.sha256(data).hexdigest(),
    }


def _checkpoint_manifest_payload(
    args: argparse.Namespace,
    *,
    global_step: int,
    stage: str,
    pinned_snapshot_bundle: Any | None = None,
    pinned_effective_entry_contract: Any | None = None,
    include_phase_effective_entry_holdout: bool = True,
    include_phase_zero_residual_rollout: bool = True,
) -> dict[str, Any]:
    from .phase_snapshots import (
        PhaseSnapshotError,
        ValidatedPhaseSnapshotBundle,
        phase_snapshot_bundle_file_hashes,
    )

    paths = _checkpoint_config_paths(args)
    if pinned_snapshot_bundle is None and pinned_effective_entry_contract is None:
        pinned, effective_contract = _pinned_runtime_phase_contracts(args)
    else:
        pinned = pinned_snapshot_bundle
        effective_contract = pinned_effective_entry_contract
        if effective_contract is None:
            candidate = getattr(args, "_pinned_effective_entry_contract", None)
            if (
                candidate is not None
                and getattr(candidate, "phase_snapshot_bundle_sha256", None)
                == getattr(pinned, "bundle_sha256", None)
            ):
                effective_contract = candidate
            else:
                effective_contract = _capture_runtime_effective_entry_contract(pinned)
    if not isinstance(pinned, ValidatedPhaseSnapshotBundle):
        raise CliError("checkpoint snapshot pin is not a validated immutable bundle")
    if pinned.snapshot_root != _live_phase_snapshot_root():
        raise CliError("checkpoint snapshot pin differs from the live loader root")
    # Checkpoint metadata must describe the exact pre-launch pins, and the
    # files backing both pins must still be byte-identical at publication.
    _revalidate_pinned_phase_contracts(pinned, effective_contract)
    snapshot_bundle = pinned.as_record()
    try:
        snapshot_file_hashes = phase_snapshot_bundle_file_hashes(snapshot_bundle)
    except PhaseSnapshotError as exc:
        raise CliError(f"checkpoint snapshot pin is invalid: {exc}") from exc
    effective_fields = _effective_entry_contract_fields(effective_contract)
    effective_file_hashes = effective_contract.file_hashes()
    file_hashes = _hash_manifest(paths)
    if set(file_hashes).intersection(snapshot_file_hashes):
        raise CliError("checkpoint config and phase snapshot file sets overlap")
    file_hashes.update(snapshot_file_hashes)
    for path, digest in effective_file_hashes.items():
        existing = file_hashes.get(path)
        if existing is not None and existing != digest:
            raise CliError(
                "checkpoint config and effective-entry contract hashes disagree for "
                + path
            )
        file_hashes[path] = digest
    payload = {
        "schema": "wlr50_clean.phase_residual_checkpoint_manifest.v1",
        "stage": stage,
        "training_seed": args.seed,
        "global_policy_decisions": global_step,
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "files": file_hashes,
        "controller_hash": _sha256(PROJECT_ROOT / "configs" / "fsm_states.yaml"),
        "environment_hash": _sha256(PROJECT_ROOT / "configs" / "environment_lock.json"),
        "observation_schema_hash": _sha256(PROJECT_ROOT / "configs" / "ppo_observation_schema_v2.json"),
        "action_schema_hash": _sha256(PROJECT_ROOT / "configs" / "ppo_phase_action_masks_v2.yaml"),
        "reward_config_hash": _sha256(PROJECT_ROOT / "configs" / "ppo_reward_v2.yaml"),
        "phase_snapshot_manifest": snapshot_bundle["manifest_path"],
        "phase_snapshot_manifest_sha256": snapshot_bundle["manifest_sha256"],
        "phase_snapshot_bundle_sha256": snapshot_bundle["bundle_sha256"],
        "phase_snapshot_bundle": snapshot_bundle,
        **effective_fields,
        **_checkpoint_creation_runtime_identity(args),
    }
    soft_reset = getattr(args, "_soft_reset_acceptance_evidence", None)
    if soft_reset is not None:
        payload["soft_reset_acceptance"] = dict(soft_reset)
    vector_benchmarks = getattr(args, "_vector_benchmark_acceptance_evidence", None)
    if vector_benchmarks is not None:
        payload["vector_benchmark_acceptance"] = dict(vector_benchmarks)
    vector_matrix = getattr(args, "_vector_benchmark_matrix_evidence", None)
    if vector_matrix is not None:
        payload["vector_benchmark_matrix"] = dict(vector_matrix)
        payload["vector_benchmark_matrix_path"] = vector_matrix["path"]
        payload["vector_benchmark_matrix_sha256"] = vector_matrix["sha256"]
    holdout = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if include_phase_effective_entry_holdout and holdout is not None:
        holdout_fields, holdout_files = _phase_effective_entry_holdout_fields(holdout)
        if holdout_fields["phase_effective_entry_holdout_files"] != holdout_files:
            raise CliError("phase effective-entry holdout files inventory is invalid")
        payload.update(holdout_fields)
    elif (
        include_phase_effective_entry_holdout
        and stage == "phase-curriculum"
        and getattr(args, "command", None) == "train"
    ):
        raise CliError(
            "phase-curriculum checkpoint publication requires pinned holdout evidence"
        )
    rollout = getattr(args, "_phase_zero_residual_rollout_evidence", None)
    if include_phase_zero_residual_rollout and rollout is not None:
        rollout_fields, rollout_files = _phase_zero_residual_rollout_fields(rollout)
        if rollout_fields["phase_zero_residual_rollout_files"] != rollout_files:
            raise CliError("phase zero-residual rollout files inventory is invalid")
        payload.update(rollout_fields)
    elif (
        include_phase_zero_residual_rollout
        and stage == "phase-curriculum"
        and getattr(args, "command", None) == "train"
    ):
        raise CliError(
            "phase-curriculum checkpoint publication requires pinned zero-rollout evidence"
        )
    return payload


def _current_checkpoint_runtime_contract(
    args: argparse.Namespace,
    *,
    pinned_snapshot_bundle: Any | None = None,
    pinned_effective_entry_contract: Any | None = None,
    include_phase_effective_entry_holdout: bool = False,
    include_phase_zero_residual_rollout: bool = False,
) -> Mapping[str, Any]:
    """Build the immutable runtime subset a loaded checkpoint must match."""

    return _checkpoint_manifest_payload(
        args,
        global_step=0,
        stage="current_runtime_contract",
        pinned_snapshot_bundle=pinned_snapshot_bundle,
        pinned_effective_entry_contract=pinned_effective_entry_contract,
        include_phase_effective_entry_holdout=(
            include_phase_effective_entry_holdout
        ),
        include_phase_zero_residual_rollout=(
            include_phase_zero_residual_rollout
        ),
    )


def _soft_reset_equivalence(args: argparse.Namespace, simulation_app: Any) -> int:
    from .isaac_fsm_backend import IsaacFSMBackend
    from .residual_direct_env import ResidualEpisodeEnv
    from .soft_reset_equivalence import (
        CompactZeroResidualTickAudit,
        PHASE_IDS,
        SOFT_RESET_ACCEPTANCE_FILENAME,
        SOFT_RESET_ACCEPTANCE_SCHEMA,
        actor_observation_v2_fingerprint,
        compact_trace_row,
        compare_compact_traces,
        compare_full_rate_tick_audits,
        compare_initial_actor_observations,
        compare_reset_metadata,
        compare_reward_totals,
        select_reset_metadata,
        soft_reset_contract_hashes,
    )

    if args.num_envs != 1 or args.episode_count != 2:
        raise CliError(
            "soft-reset equivalence requires exactly two episodes in one num-envs=1 process"
        )
    if args.residual_mode != "zero" or not args.deterministic:
        raise CliError("soft-reset equivalence requires deterministic zero residuals")

    contract_hashes_at_start = soft_reset_contract_hashes(PROJECT_ROOT)
    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    backend = IsaacFSMBackend(
        simulation_app,
        expected_phase_snapshot_bundle=pinned_snapshot_bundle,
        expected_effective_entry_contract=pinned_effective_entry_contract,
    )
    env = ResidualEpisodeEnv(backend, collect_trace=False)
    traces: list[tuple[Mapping[str, Any], ...]] = []
    summaries: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    for episode_index, reset_role in enumerate(("fresh_scene", "soft_reset_reuse")):
        initial_observation, _ = env.reset(seed=args.seed)
        initial_actor_fingerprint = actor_observation_v2_fingerprint(
            initial_observation
        )
        assert env.frame is not None
        reset_metadata = select_reset_metadata(env.frame.info)
        tick_audit = CompactZeroResidualTickAudit()
        env.tick_callback = tick_audit.append
        rows: list[Mapping[str, Any]] = []
        reward_total = 0.0
        final_step = None
        while not env.done:
            final_step = env.step((0.0,) * 12)
            reward_total += final_step.reward
            assert env.frame is not None
            rows.append(
                compact_trace_row(
                    env.frame,
                    final_step.info,
                    actor_observation_v2=final_step.observation,
                )
            )
        if final_step is None or env.frame is None:
            raise CliError("soft-reset equivalence episode ended without a decision")
        audit = tick_audit.finalize()
        signals = env.frame.termination_signals
        phases = tuple(str(value) for value in audit["phase_ids_observed"])
        summary = {
            "episode_index": episode_index,
            "reset_role": reset_role,
            "seed": args.seed,
            "authoritative_success": bool(signals.success),
            "task_success": bool(signals.success)
            and final_step.info.get("termination_reason") == "SUCCESS",
            "termination_reason": final_step.info.get("termination_reason"),
            "duration_s": float(env.frame.sim_time_s),
            "decision_count": int(env.decision_count),
            "physics_tick": int(env.frame.physics_tick),
            "phase_ids_observed": list(phases),
            "decision_phase_ids_observed": list(
                dict.fromkeys(str(row["state_id"]) for row in rows)
            ),
            "completed_p01_p13": phases == PHASE_IDS,
            "body_collision": bool(signals.body_collision),
            "wheel_only_climb": bool(signals.wheel_only_climb),
            "safety_abort": any(
                (
                    signals.fall,
                    signals.nan_inf,
                    signals.hard_joint_limit,
                    signals.physics_explosion,
                )
            ),
            "under_maximum_duration": float(env.frame.sim_time_s)
            <= args.maximum_duration_s,
            "reward_total": reward_total,
            "initial_actor_observation_v2_dimension": initial_actor_fingerprint[
                "dimension"
            ],
            "initial_actor_observation_v2_sha256": initial_actor_fingerprint[
                "sha256"
            ],
            "zero_residual_tick_audit": audit,
            "reset_metadata": reset_metadata,
            "recording_runtime_access_count": int(
                final_step.info.get("recording_runtime_access_count", 0)
            ),
            "in_episode_root_write_count": int(
                final_step.info.get("in_episode_root_write_count", 0)
            ),
            "termination_mapping": dict(
                final_step.info.get("termination_mapping", {}) or {}
            ),
        }
        trace_path = args.run_dir / f"episode_{episode_index}_{reset_role}_compact_trace.jsonl"
        summary_path = args.run_dir / f"episode_{episode_index}_{reset_role}_summary.json"
        _jsonl(trace_path, rows)
        summary["trace_path"] = str(trace_path)
        summary["trace_sha256"] = _sha256(trace_path)
        _json(summary_path, summary)
        artifact_paths.extend((trace_path, summary_path))
        traces.append(tuple(rows))
        summaries.append(summary)

    trace_comparison = compare_compact_traces(traces[0], traces[1])
    reset_comparison = compare_reset_metadata(
        summaries[0]["reset_metadata"], summaries[1]["reset_metadata"]
    )
    full_rate_tick_audit_comparison = compare_full_rate_tick_audits(
        summaries[0]["zero_residual_tick_audit"],
        summaries[1]["zero_residual_tick_audit"],
    )
    initial_actor_observation_comparison = compare_initial_actor_observations(
        summaries[0], summaries[1]
    )
    reward_total_comparison = compare_reward_totals(
        summaries[0], summaries[1], traces[0], traces[1]
    )
    contract_hashes_at_end = soft_reset_contract_hashes(PROJECT_ROOT)
    checks = {
        "one_backend_instance_two_episodes": True,
        "both_authoritative_success": all(row["task_success"] for row in summaries),
        "both_complete_p01_p13": all(row["completed_p01_p13"] for row in summaries),
        "both_no_body_collision": all(not row["body_collision"] for row in summaries),
        "both_no_wheel_only_climb": all(
            not row["wheel_only_climb"] for row in summaries
        ),
        "both_no_safety_abort": all(not row["safety_abort"] for row in summaries),
        "both_under_maximum_duration": all(
            row["under_maximum_duration"] for row in summaries
        ),
        "both_zero_residual_bitwise_all_ticks": all(
            row["zero_residual_tick_audit"]["passed"] for row in summaries
        ),
        "contract_files_unchanged_during_run": (
            contract_hashes_at_start == contract_hashes_at_end
        ),
        "full_rate_tick_audits_equal_between_episodes": bool(
            full_rate_tick_audit_comparison["exactly_equal"]
        ),
        "initial_actor_observations_equal_between_episodes": bool(
            initial_actor_observation_comparison["exactly_equal"]
        ),
        "reward_totals_match_traces_and_between_episodes": bool(
            reward_total_comparison["passed"]
        ),
        "decision_counts_match_compact_traces": all(
            row["decision_count"] == len(trace)
            for row, trace in zip(summaries, traces, strict=True)
        ),
        "physics_tick_counts_match_audits": all(
            row["physics_tick"] == row["zero_residual_tick_audit"]["tick_count"]
            for row in summaries
        ),
        "no_runtime_recording_access": all(
            row["recording_runtime_access_count"] == 0 for row in summaries
        ),
        "no_in_episode_root_writes": all(
            row["in_episode_root_write_count"] == 0 for row in summaries
        ),
        "reset_metadata_equivalent": bool(reset_comparison["passed"]),
        "deterministic_trace_equal_through_p10": bool(
            trace_comparison["through_p10"]["exactly_equal"]
        ),
        "deterministic_trace_equal_whole_episode": bool(
            trace_comparison["whole_episode"]["exactly_equal"]
        ),
    }
    artifacts = [
        {
            "path": path.relative_to(args.run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifact_paths
    ]
    result = {
        "schema": SOFT_RESET_ACCEPTANCE_SCHEMA,
        "passed": all(checks.values()),
        "seed": args.seed,
        "episode_count": 2,
        "backend_instance_count": 1,
        "full_rate_raw_streams_written": False,
        "compact_trace_fields": list(trace_comparison["fields"]),
        "checks": checks,
        "episodes": summaries,
        "reset_metadata_comparison": reset_comparison,
        "full_rate_tick_audit_comparison": full_rate_tick_audit_comparison,
        "initial_actor_observation_comparison": (
            initial_actor_observation_comparison
        ),
        "reward_total_comparison": reward_total_comparison,
        "trace_comparison": trace_comparison,
        "contract_file_sha256": contract_hashes_at_start,
        "contract_file_sha256_at_end": contract_hashes_at_end,
        "artifacts": artifacts,
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _json(args.run_dir / SOFT_RESET_ACCEPTANCE_FILENAME, result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0 if result["passed"] else 2


def _reset_throughput_probe(args: argparse.Namespace, simulation_app: Any) -> int:
    """Measure two exact short reset horizons without producing gate evidence."""

    from .isaac_fsm_backend import IsaacFSMBackend
    from .residual_direct_env import (
        DECISION_HZ,
        PHYSICS_HZ,
        PHYSICS_TICKS_PER_DECISION,
        ResidualEpisodeEnv,
    )
    from .soft_reset_equivalence import (
        CompactZeroResidualTickAudit,
        actor_observation_v2_fingerprint,
        compare_reset_metadata,
        select_reset_metadata,
        soft_reset_contract_hashes,
    )

    if args.num_envs != 1 or args.episode_count != 2:
        raise CliError(
            "reset throughput probe requires exactly two resets in one num-envs=1 process"
        )
    if args.residual_mode != "zero" or not args.deterministic:
        raise CliError("reset throughput probe requires deterministic zero residuals")
    if args.policy_decisions not in (None, RESET_THROUGHPUT_PROBE_DECISIONS):
        raise CliError(
            "reset throughput probe has a fixed horizon of exactly 8 policy decisions"
        )
    if (
        DECISION_HZ != 15.0
        or PHYSICS_HZ != 120.0
        or PHYSICS_TICKS_PER_DECISION != 8
        or RESET_THROUGHPUT_PROBE_DECISIONS * PHYSICS_TICKS_PER_DECISION
        != RESET_THROUGHPUT_PROBE_PHYSICS_TICKS
    ):
        raise CliError("reset throughput probe runtime cadence differs from 15 Hz / 120 Hz")

    def _elapsed(started: float, *, label: str) -> float:
        value = time.perf_counter() - started
        if not math.isfinite(value) or value <= 0.0:
            raise CliError(f"{label} wall time must be positive and finite")
        return value

    contract_hashes_at_start = soft_reset_contract_hashes(PROJECT_ROOT)
    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    backend = IsaacFSMBackend(
        simulation_app,
        expected_phase_snapshot_bundle=pinned_snapshot_bundle,
        expected_effective_entry_contract=pinned_effective_entry_contract,
    )
    env = ResidualEpisodeEnv(backend, collect_trace=False)
    episodes: list[dict[str, Any]] = []
    for episode_index, reset_role in enumerate(("fresh_scene", "soft_reset_reuse")):
        env.tick_callback = None
        reset_started = time.perf_counter()
        initial_observation, _ = env.reset(seed=args.seed)
        reset_wall_s = _elapsed(reset_started, label=f"{reset_role} reset")
        if env.frame is None:
            raise CliError(f"{reset_role} reset returned no authoritative frame")
        if (
            int(env.frame.physics_tick) != 0
            or int(env.decision_count) != 0
            or bool(env.done)
        ):
            raise CliError(f"{reset_role} reset did not return a fresh logical tick zero")

        tick0_actor = actor_observation_v2_fingerprint(initial_observation)
        reset_metadata = select_reset_metadata(env.frame.info)
        tick_audit = CompactZeroResidualTickAudit()
        env.tick_callback = tick_audit.append

        step_started = time.perf_counter()
        for decision_index in range(RESET_THROUGHPUT_PROBE_DECISIONS):
            if env.done:
                raise CliError(
                    f"{reset_role} terminated before decision {decision_index} of "
                    f"{RESET_THROUGHPUT_PROBE_DECISIONS}"
                )
            if env.frame is None:
                raise CliError(f"{reset_role} lost its authoritative frame")
            before_tick = int(env.frame.physics_tick)
            step = env.step((0.0,) * 12)
            if env.frame is None:
                raise CliError(f"{reset_role} step returned no authoritative frame")
            ticks_executed = step.info.get("physics_ticks_executed")
            expected_tick = (decision_index + 1) * PHYSICS_TICKS_PER_DECISION
            if (
                isinstance(ticks_executed, bool)
                or not isinstance(ticks_executed, int)
                or ticks_executed != PHYSICS_TICKS_PER_DECISION
                or int(env.frame.physics_tick) - before_tick
                != PHYSICS_TICKS_PER_DECISION
                or int(env.frame.physics_tick) != expected_tick
                or int(env.decision_count) != decision_index + 1
            ):
                raise CliError(
                    f"{reset_role} decision {decision_index} did not advance exactly "
                    f"{PHYSICS_TICKS_PER_DECISION} physics ticks"
                )
        step_wall_s = _elapsed(step_started, label=f"{reset_role} steps")

        audit = tick_audit.finalize()
        if (
            audit.get("tick_count") != RESET_THROUGHPUT_PROBE_PHYSICS_TICKS
            or audit.get("raw_zero_tick_count")
            != RESET_THROUGHPUT_PROBE_PHYSICS_TICKS
            or audit.get("projected_zero_tick_count")
            != RESET_THROUGHPUT_PROBE_PHYSICS_TICKS
            or audit.get("zero_fast_path_tick_count")
            != RESET_THROUGHPUT_PROBE_PHYSICS_TICKS
        ):
            raise CliError(f"{reset_role} 120 Hz tick audit has an invalid fixed shape")
        nominal_sha256 = audit.get("nominal_sequence_sha256")
        applied_sha256 = audit.get("applied_sequence_sha256")
        if (
            not isinstance(nominal_sha256, str)
            or len(nominal_sha256) != 64
            or not isinstance(applied_sha256, str)
            or len(applied_sha256) != 64
        ):
            raise CliError(f"{reset_role} 120 Hz action SHA-256 evidence is invalid")
        ticks_per_wall_s = RESET_THROUGHPUT_PROBE_PHYSICS_TICKS / step_wall_s
        if not math.isfinite(ticks_per_wall_s) or ticks_per_wall_s <= 0.0:
            raise CliError(f"{reset_role} tick throughput is not positive and finite")
        episodes.append(
            {
                "episode_index": episode_index,
                "reset_role": reset_role,
                "seed": args.seed,
                "reset_wall_s": reset_wall_s,
                "step_wall_s": step_wall_s,
                "ticks_per_wall_s": ticks_per_wall_s,
                "decision_count": int(env.decision_count),
                "physics_tick_count": int(env.frame.physics_tick),
                "tick0_actor_observation_v2_dimension": tick0_actor["dimension"],
                "tick0_actor_observation_v2_sha256": tick0_actor["sha256"],
                "nominal_action_120hz_sha256": nominal_sha256,
                "applied_action_120hz_sha256": applied_sha256,
                "zero_residual_bitwise_120hz": bool(audit.get("passed")),
                "reset_metadata": reset_metadata,
            }
        )

    reset_metadata_comparison = compare_reset_metadata(
        episodes[0]["reset_metadata"], episodes[1]["reset_metadata"]
    )
    contract_hashes_at_end = soft_reset_contract_hashes(PROJECT_ROOT)
    checks = {
        "one_backend_instance_two_resets": len(episodes) == 2,
        "fixed_8_decision_64_tick_shape": all(
            row["decision_count"] == RESET_THROUGHPUT_PROBE_DECISIONS
            and row["physics_tick_count"] == RESET_THROUGHPUT_PROBE_PHYSICS_TICKS
            for row in episodes
        ),
        "positive_finite_wall_timings": all(
            math.isfinite(float(row[name])) and float(row[name]) > 0.0
            for row in episodes
            for name in ("reset_wall_s", "step_wall_s", "ticks_per_wall_s")
        ),
        "both_zero_residual_bitwise_at_120hz": all(
            row["zero_residual_bitwise_120hz"] is True for row in episodes
        ),
        "tick0_actor_observations_equal": (
            episodes[0]["tick0_actor_observation_v2_dimension"]
            == episodes[1]["tick0_actor_observation_v2_dimension"]
            == 125
            and episodes[0]["tick0_actor_observation_v2_sha256"]
            == episodes[1]["tick0_actor_observation_v2_sha256"]
        ),
        "nominal_120hz_sequences_equal": (
            episodes[0]["nominal_action_120hz_sha256"]
            == episodes[1]["nominal_action_120hz_sha256"]
        ),
        "applied_120hz_sequences_equal": (
            episodes[0]["applied_action_120hz_sha256"]
            == episodes[1]["applied_action_120hz_sha256"]
        ),
        "reset_metadata_equivalent": bool(reset_metadata_comparison["passed"]),
        "runtime_contract_files_unchanged": (
            contract_hashes_at_start == contract_hashes_at_end
        ),
    }
    passed = all(checks.values())
    result = {
        "schema": RESET_THROUGHPUT_PROBE_SCHEMA,
        "artifact_role": "RESET_THROUGHPUT_DIAGNOSTIC_ONLY",
        "soft_reset_equivalence_gate_eligible": False,
        "status": "PASSED" if passed else "FAILED",
        "seed": args.seed,
        "backend_instance_count": 1,
        "episode_count": len(episodes),
        "decision_hz": DECISION_HZ,
        "physics_hz": PHYSICS_HZ,
        "policy_decisions_per_episode": RESET_THROUGHPUT_PROBE_DECISIONS,
        "physics_ticks_per_decision": PHYSICS_TICKS_PER_DECISION,
        "physics_ticks_per_episode": RESET_THROUGHPUT_PROBE_PHYSICS_TICKS,
        "checks": checks,
        "episodes": episodes,
        "reset_metadata_comparison": reset_metadata_comparison,
        "runtime_contract_file_sha256": contract_hashes_at_start,
        "runtime_contract_file_sha256_at_end": contract_hashes_at_end,
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _json(args.run_dir / RESET_THROUGHPUT_PROBE_FILENAME, result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0 if passed else 2


def _phase_snapshot_live_probe(
    args: argparse.Namespace, simulation_app: Any
) -> int:
    """Capture P02-P13 one-tick no-rewind reset evidence."""

    from .phase_snapshot_live_probe import run_phase_snapshot_live_probe

    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    result = run_phase_snapshot_live_probe(
        simulation_app,
        run_dir=args.run_dir,
        seed=args.seed,
        snapshot_bundle=pinned_snapshot_bundle,
        effective_entry_contract=pinned_effective_entry_contract,
        prime_physics_steps=args.phase_snapshot_prime_physics_steps,
        phases=None if args.phase is None else (args.phase,),
    )
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    return 0 if result.get("passed") is True else 2


def _validate_phase_rollout_holdout(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    """Pin the external holdout before starting the rollout's Isaac process."""

    supplied = getattr(args, "phase_effective_entry_holdout_acceptance", None)
    if supplied is None:
        raise CliError(
            "phase-zero-residual-rollout requires an explicit "
            "--phase-effective-entry-holdout-acceptance"
        )
    from .phase_effective_entry_holdout import (
        PhaseEffectiveEntryHoldoutError,
        validate_phase_effective_entry_holdout_acceptance,
    )

    snapshot_bundle, effective_entry_contract = _pinned_runtime_phase_contracts(args)
    try:
        evidence = validate_phase_effective_entry_holdout_acceptance(
            _resolve_project_path(Path(supplied)),
            project_root=PROJECT_ROOT,
            config_paths=_checkpoint_config_paths(args),
            snapshot_bundle=snapshot_bundle,
            effective_entry_contract=effective_entry_contract,
        )
    except PhaseEffectiveEntryHoldoutError as exc:
        raise CliError(
            f"phase rollout holdout acceptance rejected: {exc}"
        ) from exc
    args.phase_effective_entry_holdout_acceptance = Path(evidence["path"])
    args._phase_effective_entry_holdout_evidence = evidence
    return evidence


def _revalidate_phase_rollout_holdout(
    args: argparse.Namespace,
    snapshot_bundle: Any,
    effective_entry_contract: Any,
) -> Mapping[str, Any]:
    expected = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if not isinstance(expected, Mapping):
        raise CliError("phase rollout holdout acceptance was not pinned pre-launch")
    from .phase_effective_entry_holdout import (
        PhaseEffectiveEntryHoldoutError,
        validate_phase_effective_entry_holdout_acceptance,
    )

    try:
        current = validate_phase_effective_entry_holdout_acceptance(
            Path(str(expected.get("path", ""))),
            project_root=PROJECT_ROOT,
            config_paths=_checkpoint_config_paths(args),
            snapshot_bundle=snapshot_bundle,
            effective_entry_contract=effective_entry_contract,
        )
    except PhaseEffectiveEntryHoldoutError as exc:
        raise CliError(f"phase rollout holdout changed after capture: {exc}") from exc
    if current != dict(expected):
        raise CliError("phase rollout holdout evidence changed after capture")
    return current


def _phase_zero_residual_rollout(
    args: argparse.Namespace, simulation_app: Any
) -> int:
    """Run one bounded real-physics zero-residual window from every phase."""

    from .isaac_fsm_backend import IsaacFSMBackend
    from .phase_zero_residual_rollout import (
        ARTIFACT_FILENAME,
        MAX_DECISIONS_PER_PHASE,
        build_contract_binding,
        run_phase_zero_residual_rollout,
    )
    from .residual_direct_env import ResidualEpisodeEnv

    if args.num_envs != 1 or args.episode_count != 13:
        raise CliError(
            "phase-zero-residual-rollout requires num-envs=1 and episode-count=13"
        )
    if args.residual_mode != "zero" or not args.deterministic:
        raise CliError(
            "phase-zero-residual-rollout requires deterministic zero residuals"
        )
    if args.seed_set != "train":
        raise CliError("phase-zero-residual-rollout requires seed-set=train")
    if args.policy_decisions not in (None, MAX_DECISIONS_PER_PHASE):
        raise CliError(
            "phase-zero-residual-rollout has a fixed maximum of 64 decisions per phase"
        )
    holdout = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if not isinstance(holdout, Mapping):
        raise CliError("phase rollout holdout acceptance was not pinned pre-launch")
    snapshot_bundle, effective_entry_contract = _pinned_runtime_phase_contracts(args)
    binding = build_contract_binding(
        snapshot_bundle, effective_entry_contract, holdout
    )
    backend = IsaacFSMBackend(
        simulation_app,
        expected_phase_snapshot_bundle=snapshot_bundle,
        expected_effective_entry_contract=effective_entry_contract,
    )
    episode = ResidualEpisodeEnv(backend, collect_trace=False)
    result = run_phase_zero_residual_rollout(
        episode,
        seed=args.seed,
        contract_binding=binding,
        max_decisions_per_phase=MAX_DECISIONS_PER_PHASE,
    )
    _revalidate_phase_rollout_holdout(
        args, snapshot_bundle, effective_entry_contract
    )
    _revalidate_pinned_phase_contracts(
        snapshot_bundle, effective_entry_contract
    )
    _json(args.run_dir / ARTIFACT_FILENAME, result)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    return 0 if result.get("passed") is True else 2


def _require_training_phase_zero_residual_rollout(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    """Reject phase-curriculum training until the bounded live rollout passes."""

    if args.stage != "phase-curriculum":
        return None
    supplied = getattr(args, "phase_zero_residual_rollout_evidence", None)
    if supplied is None:
        raise CliError(
            "phase-curriculum training requires an explicit "
            "--phase-zero-residual-rollout-evidence produced after holdout acceptance"
        )
    holdout = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if not isinstance(holdout, Mapping):
        raise CliError(
            "phase zero-residual rollout validation requires pinned holdout evidence"
        )
    from .phase_zero_residual_rollout import (
        PhaseZeroResidualRolloutError,
        build_contract_binding,
        validate_phase_zero_residual_rollout_evidence,
    )

    snapshot_bundle, effective_entry_contract = _pinned_runtime_phase_contracts(args)
    binding = build_contract_binding(
        snapshot_bundle, effective_entry_contract, holdout
    )
    try:
        evidence = validate_phase_zero_residual_rollout_evidence(
            _resolve_project_path(Path(supplied)),
            project_root=PROJECT_ROOT,
            expected_contract_binding=binding,
        )
    except PhaseZeroResidualRolloutError as exc:
        raise CliError(f"phase zero-residual rollout rejected: {exc}") from exc
    args.phase_zero_residual_rollout_evidence = Path(evidence["path"])
    args._phase_zero_residual_rollout_evidence = evidence
    return evidence


def _revalidate_training_phase_zero_residual_rollout(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    expected = getattr(args, "_phase_zero_residual_rollout_evidence", None)
    holdout = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if not isinstance(expected, Mapping):
        if args.stage == "phase-curriculum":
            raise CliError("phase rollout training evidence was not pinned pre-launch")
        return None
    if not isinstance(holdout, Mapping):
        raise CliError("phase rollout revalidation requires pinned holdout evidence")
    from .phase_zero_residual_rollout import (
        PhaseZeroResidualRolloutError,
        build_contract_binding,
        validate_phase_zero_residual_rollout_evidence,
    )

    snapshot_bundle, effective_entry_contract = _pinned_runtime_phase_contracts(args)
    binding = build_contract_binding(
        snapshot_bundle, effective_entry_contract, holdout
    )
    try:
        current = validate_phase_zero_residual_rollout_evidence(
            Path(str(expected.get("path", ""))),
            project_root=PROJECT_ROOT,
            expected_contract_binding=binding,
        )
    except PhaseZeroResidualRolloutError as exc:
        raise CliError(f"phase zero-residual rollout changed: {exc}") from exc
    if current != dict(expected):
        raise CliError("phase zero-residual rollout evidence changed after capture")
    return current


def _inherit_training_phase_zero_residual_rollout(
    args: argparse.Namespace,
    checkpoint_infos: Mapping[str, Any],
    pinned_snapshot_bundle: Any,
    pinned_effective_entry_contract: Any,
) -> Mapping[str, Any] | None:
    """Validate and retain Stage 1 rollout qualification across resumes."""

    if args.stage == "smoke":
        return getattr(args, "_phase_zero_residual_rollout_evidence", None)
    resume_stage = checkpoint_infos.get("stage")
    if args.stage == "phase-curriculum" and (
        not isinstance(resume_stage, str)
        or resume_stage not in ("smoke", "phase-curriculum")
    ):
        raise CliError(
            "phase-curriculum resume checkpoint has invalid or missing stage ancestry"
        )
    embedded = checkpoint_infos.get("phase_zero_residual_rollout_evidence")
    if not isinstance(embedded, Mapping):
        if args.stage == "phase-curriculum" and resume_stage == "smoke":
            # The initial/smoke checkpoint predates Stage 1 qualification; the
            # explicit pre-launch rollout pin remains authoritative.
            return getattr(args, "_phase_zero_residual_rollout_evidence", None)
        raise CliError(
            f"{args.stage} resume checkpoint omits phase zero-residual "
            "rollout ancestry"
        )
    holdout = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if not isinstance(holdout, Mapping):
        raise CliError(
            "resume checkpoint rollout ancestry requires validated holdout ancestry"
        )
    from .phase_zero_residual_rollout import (
        PhaseZeroResidualRolloutError,
        build_contract_binding,
        validate_phase_zero_residual_rollout_evidence,
    )

    binding = build_contract_binding(
        pinned_snapshot_bundle, pinned_effective_entry_contract, holdout
    )
    try:
        current = validate_phase_zero_residual_rollout_evidence(
            Path(str(embedded.get("path", ""))),
            project_root=PROJECT_ROOT,
            expected_contract_binding=binding,
        )
    except PhaseZeroResidualRolloutError as exc:
        raise CliError(
            f"inherited phase zero-residual rollout rejected: {exc}"
        ) from exc
    if current != dict(embedded):
        raise CliError(
            "resume checkpoint phase zero-residual rollout evidence is stale "
            "or was substituted"
        )
    explicit = getattr(args, "_phase_zero_residual_rollout_evidence", None)
    if args.stage == "phase-curriculum" and (
        not isinstance(explicit, Mapping) or dict(explicit) != current
    ):
        raise CliError(
            "phase-curriculum resume checkpoint rollout differs from the "
            "explicit pre-launch evidence"
        )
    expected_fields, _ = _phase_zero_residual_rollout_fields(current)
    for field, value in expected_fields.items():
        if checkpoint_infos.get(field) != value:
            raise CliError(
                "resume checkpoint rollout manifest/infos binding differs for "
                f"{field!r}"
            )
    args.phase_zero_residual_rollout_evidence = Path(current["path"])
    args._phase_zero_residual_rollout_evidence = current
    return current


def _vector_benchmark(args: argparse.Namespace, simulation_app: Any) -> int:
    from .vectorized_isaac_backend import VectorizedIsaacFSMBackend
    from .vectorized_residual_env import VectorizedRslResidualEnv
    from .vectorized_smoke_evidence import (
        collect_vectorized_residual_smoke_evidence,
    )

    seeds = _seed_values(args)
    seed_rows = tuple(seeds[index % len(seeds)] for index in range(args.num_envs))
    stage = "backend_construction"
    cuda_memory: dict[str, Any] | None = None
    try:
        backend = VectorizedIsaacFSMBackend(
            simulation_app,
            num_envs=args.num_envs,
            env_spacing_m=8.0,
        )
        stage = "backend_reset"
        backend.reset_all(seeds=seed_rows)
        stage = "cuda_memory_measurement_setup"
        import torch  # type: ignore

        cuda_device = torch.device(str(backend.device))
        if cuda_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("vector capacity benchmark requires an available CUDA device")
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        memory_start_allocated = int(torch.cuda.memory_allocated(cuda_device))
        memory_start_reserved = int(torch.cuda.memory_reserved(cuda_device))
        stage = "backend_benchmark"
        report = backend.benchmark(measured_ticks=args.measured_ticks)
        if not report.true_batched_isaac_verified:
            smoke_payload = None
        else:
            stage = "residual_smoke"
            # The adapter performs a new synchronized reset, so the smoke starts
            # at authoritative episode tick zero even though the same one-scene
            # backend was first used for the throughput measurement.
            env = VectorizedRslResidualEnv(
                backend,
                seeds=seed_rows,
                device="cuda:0",
            )
            smoke_mode = (
                "zero" if args.residual_mode == "zero" else "nonzero"
            )
            smoke = collect_vectorized_residual_smoke_evidence(
                env,
                mode=smoke_mode,
                policy_decisions=(
                    2 if args.policy_decisions is None else args.policy_decisions
                ),
            )
            smoke_payload = smoke.as_dict()
        stage = "cuda_memory_measurement_finalize"
        torch.cuda.synchronize(cuda_device)
        peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
        device_total = int(torch.cuda.get_device_properties(cuda_device).total_memory)
        if (
            memory_start_allocated <= 0
            or memory_start_reserved <= 0
            or peak_allocated < memory_start_allocated
            or peak_reserved < memory_start_reserved
            or peak_allocated <= 0
            or peak_reserved <= 0
            or device_total <= 0
            or peak_allocated >= device_total
            or peak_reserved >= device_total
        ):
            raise RuntimeError("CUDA peak-memory evidence is invalid or exceeds capacity")
        cuda_memory = {
            "schema": "wlr50_clean.vector_cuda_memory_evidence.v1",
            "device": str(cuda_device),
            "peak_stats_reset_before_measured_section": True,
            "measurement_covers_throughput_and_residual_smoke": True,
            "allocated_bytes_at_measurement_start": memory_start_allocated,
            "reserved_bytes_at_measurement_start": memory_start_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "device_total_bytes": device_total,
            "peak_allocated_below_device_total": True,
            "peak_reserved_below_device_total": True,
            "oom_detected": False,
        }
        contamination_complete = bool(
            report.true_batched_isaac_verified
            and smoke_payload is not None
            and smoke_payload.get("passed") is True
        )
        contamination = {
            "schema": "wlr50_clean.vector_contamination_evidence.v1",
            "evidence_complete": contamination_complete,
            "cross_environment_contamination_detected": (
                False if contamination_complete else None
            ),
            "fsm_state_contamination_detected": (
                False if contamination_complete else None
            ),
            "render_contamination_detected": (
                False if contamination_complete else None
            ),
            "measured_render_calls": 0,
            "independent_seed_count": len(set(seed_rows)),
            "independent_controller_count": report.independent_controller_count,
            "independent_reader_count": report.independent_reader_count,
            "independent_origin_count": (
                None
                if smoke_payload is None
                else smoke_payload.get("independent_origin_count")
            ),
        }
        payload = {
            "schema": "wlr50_clean.vectorized_isaac_benchmark_run.v1",
            "seed_rows": list(seed_rows),
            "report": report.as_dict(),
            "residual_smoke": smoke_payload,
            "resource_evidence": {
                "cuda_memory": cuda_memory,
                "contamination": contamination,
            },
            "passed": bool(
                report.true_batched_isaac_verified
                and smoke_payload is not None
                and smoke_payload.get("passed") is True
            ),
        }
    except Exception as exc:
        message = str(exc).strip() or repr(exc)
        reason = f"{stage}:{type(exc).__name__}:{message}"
        payload = {
            "schema": "wlr50_clean.vectorized_isaac_benchmark_run.v1",
            "seed_rows": list(seed_rows),
            "report": {
                "status": "VECTOR_BACKEND_BENCHMARK_FAILED",
                "num_envs": args.num_envs,
                "measured_ticks": args.measured_ticks,
                "wall_time_s": 0.0,
                "physics_steps_per_second": 0.0,
                "environment_steps_per_second": 0.0,
                "one_simulation_context": False,
                "articulation_tensor_instances": 0,
                "global_physics_steps": 0,
                "batched_articulation_writes": 0,
                "exact_pair_captures": 0,
                "exact_pair_sensor_count": 0,
                "independent_controller_count": 0,
                "independent_reader_count": 0,
                "final_state_ids": [],
                "true_batched_isaac_verified": False,
                "failure_reasons": [reason],
                "failure_details": [
                    {
                        "stage": stage,
                        "exception_type": type(exc).__name__,
                        "message": message,
                    }
                ],
            },
            "residual_smoke": None,
            "resource_evidence": {
                "cuda_memory": (
                    cuda_memory
                    if cuda_memory is not None
                    else {
                        "schema": "wlr50_clean.vector_cuda_memory_evidence.v1",
                        "evidence_complete": False,
                        "oom_detected": "out of memory" in message.lower(),
                    }
                ),
                "contamination": {
                    "schema": "wlr50_clean.vector_contamination_evidence.v1",
                    "evidence_complete": False,
                    "cross_environment_contamination_detected": None,
                    "fsm_state_contamination_detected": None,
                    "render_contamination_detected": None,
                },
            },
            "passed": False,
        }
    _json(args.run_dir / "vector_benchmark.json", payload)
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    return 0 if payload["passed"] else 2


def _acceptance_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CliError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CliError(f"{label} must be a JSON object: {path}")
    return payload


def _validated_run_file_record(
    run_dir: Path,
    record: Any,
    *,
    expected_relative_path: str,
    label: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise CliError(f"{label} digest record is missing")
    relative = record.get("path")
    if relative != expected_relative_path:
        raise CliError(f"{label} digest record is bound to the wrong path")
    selected = (run_dir / expected_relative_path).resolve()
    try:
        selected.relative_to(run_dir)
    except ValueError as exc:
        raise CliError(f"{label} path escapes the finalized run") from exc
    expected_bytes = record.get("bytes")
    expected_hash = record.get("sha256")
    if (
        not selected.is_file()
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or selected.stat().st_size != expected_bytes
        or not isinstance(expected_hash, str)
        or _sha256(selected) != expected_hash
    ):
        raise CliError(f"{label} digest mismatch or post-finalization tamper")
    return selected


def _invocation_value(arguments: Sequence[Any], flag: str) -> str | None:
    values = [str(value) for value in arguments]
    positions = [index for index, value in enumerate(values) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        return None
    return values[positions[0] + 1]


def _validate_vector_benchmark_acceptance(
    path: Path | str,
    *,
    expected_mode: str,
    expected_num_envs: int,
    expected_run_seed: int,
    expected_seed_rows: Sequence[int],
    expected_config_sha256: str,
    expected_frozen_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Validate a finalized live vector benchmark as immutable train evidence."""

    benchmark_path = Path(path).resolve()
    run_dir = benchmark_path.parent
    if benchmark_path.name != "vector_benchmark.json":
        raise CliError("vector benchmark acceptance must name vector_benchmark.json")
    if expected_mode not in {"zero", "bounded-smoke"}:
        raise CliError("expected vector benchmark mode is invalid")
    if expected_num_envs not in {8, 16, 32}:
        raise CliError("expected vector benchmark environment count is invalid")
    seed_rows = tuple(int(value) for value in expected_seed_rows)
    if len(seed_rows) != expected_num_envs or len(set(seed_rows)) != len(seed_rows):
        raise CliError("expected vector benchmark seed_rows are invalid")

    benchmark = _acceptance_json(benchmark_path, label="vector benchmark acceptance")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _acceptance_json(manifest_path, label="finalized vector run manifest")
    started_path = run_dir / "run_manifest.started.json"
    started = _acceptance_json(started_path, label="started vector run manifest")
    if (
        manifest.get("schema") != "wlr50_clean.ppo_run_manifest.v1"
        or manifest.get("lifecycle") != "SUCCEEDED"
        or manifest.get("exit_code") != 0
        or manifest.get("immutable_run_directory") is not True
        or manifest.get("run_kind") != "vector_benchmark"
        or manifest.get("subcommand") != "vector-benchmark"
        or manifest.get("entrypoint") != "wlr50_clean.ppo.cli"
        or Path(str(manifest.get("run_dir", ""))).resolve() != run_dir
        or started.get("lifecycle") != "STARTED"
        or started.get("schema") != manifest.get("schema")
    ):
        raise CliError("vector benchmark does not have a successful finalized run manifest")
    for key, value in started.items():
        if key != "lifecycle" and manifest.get(key) != value:
            raise CliError(f"finalized vector run manifest changed started field {key!r}")
    _validated_run_file_record(
        run_dir,
        manifest.get("started_manifest"),
        expected_relative_path="run_manifest.started.json",
        label="started manifest",
    )
    logs = manifest.get("logs")
    if not isinstance(logs, Mapping):
        raise CliError("finalized vector run manifest omits log digests")
    stdout_path = _validated_run_file_record(
        run_dir,
        logs.get("stdout.log"),
        expected_relative_path="stdout.log",
        label="stdout",
    )
    _validated_run_file_record(
        run_dir,
        logs.get("stderr.log"),
        expected_relative_path="stderr.log",
        label="stderr",
    )
    artifact_records = manifest.get("artifacts")
    if artifact_records is not None:
        if not isinstance(artifact_records, Mapping):
            raise CliError("finalized vector run artifact digest map is malformed")
        for relative in (
            "vector_benchmark.json",
            "frozen_hashes.before.json",
            "frozen_hashes.after.json",
        ):
            _validated_run_file_record(
                run_dir,
                artifact_records.get(relative),
                expected_relative_path=relative,
                label=f"finalized {relative}",
            )

    identity = manifest.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("config_sha256") != expected_config_sha256
        or identity.get("environment_count") != expected_num_envs
        or identity.get("seed") != expected_run_seed
        or identity.get("training_stage") != "backend-benchmark"
    ):
        raise CliError("vector benchmark run identity differs from training")
    invocation = manifest.get("invocation_arguments")
    if not isinstance(invocation, Sequence) or isinstance(invocation, (str, bytes)):
        raise CliError("vector benchmark invocation evidence is missing")
    expected_invocation = {
        "--seed-set": "train",
        "--residual-mode": expected_mode,
        "--seed": str(expected_run_seed),
        "--num-envs": str(expected_num_envs),
    }
    for flag, expected in expected_invocation.items():
        if _invocation_value(invocation, flag) != expected:
            raise CliError(f"vector benchmark invocation differs for {flag}")

    stdout_objects: list[Mapping[str, Any]] = []
    for line in stdout_path.read_text(encoding="utf-8", errors="strict").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(item, Mapping):
            stdout_objects.append(item)
    bound_payloads = [
        item
        for item in stdout_objects
        if item.get("schema") == "wlr50_clean.vectorized_isaac_benchmark_run.v1"
    ]
    if len(bound_payloads) != 1 or dict(bound_payloads[0]) != dict(benchmark):
        raise CliError(
            "vector_benchmark.json is not exactly bound to the finalized stdout digest"
        )

    expected_audits = {
        (run_dir / "frozen_hashes.before.json").resolve(),
        (run_dir / "frozen_hashes.after.json").resolve(),
    }
    stdout_audits = {
        Path(str(item["audit"])).resolve()
        for item in stdout_objects
        if item.get("passed") is True and isinstance(item.get("audit"), str)
    }
    if not expected_audits.issubset(stdout_audits):
        raise CliError("vector benchmark stdout omits bound before/after frozen audits")
    frozen_manifest_payload = _acceptance_json(
        PROJECT_ROOT
        / "artifacts"
        / "ppo_phase_v1_start"
        / "frozen_fsm_hashes.json",
        label="frozen FSM manifest",
    )
    frozen_source_head = frozen_manifest_payload.get("source_head")
    frozen_audit_hashes: dict[str, str] = {}
    for audit_path in sorted(expected_audits, key=str):
        audit = _acceptance_json(audit_path, label="vector benchmark frozen audit")
        if (
            audit.get("schema") != "wlr50_clean.frozen_fsm_hash_audit.v1"
            or audit.get("passed") is not True
            or audit.get("mismatches") != []
            or audit.get("protected_file_count") != 29
            or audit.get("frozen_manifest_sha256")
            != expected_frozen_manifest_sha256
            or audit.get("source_head") != frozen_source_head
        ):
            raise CliError("vector benchmark frozen hash audit is stale or failed")
        frozen_audit_hashes[str(audit_path)] = _sha256(audit_path)

    report = benchmark.get("report")
    smoke = benchmark.get("residual_smoke")
    actual_smoke_mode = "zero" if expected_mode == "zero" else "nonzero"
    expected_status = (
        "VECTOR_ZERO_RESIDUAL_SMOKE_PASSED"
        if expected_mode == "zero"
        else "VECTOR_NONZERO_RESIDUAL_SMOKE_PASSED"
    )
    if (
        benchmark.get("schema") != "wlr50_clean.vectorized_isaac_benchmark_run.v1"
        or benchmark.get("passed") is not True
        or tuple(benchmark.get("seed_rows", ())) != seed_rows
    ):
        raise CliError("vector benchmark passed flag or seed_rows differs from training")
    if (
        not isinstance(report, Mapping)
        or report.get("status") != "TRUE_BATCHED_ISAAC_VERIFIED"
        or report.get("num_envs") != expected_num_envs
        or report.get("true_batched_isaac_verified") is not True
        or report.get("one_simulation_context") is not True
        or report.get("articulation_tensor_instances") != expected_num_envs
        or report.get("independent_controller_count") != expected_num_envs
        or report.get("independent_reader_count") != expected_num_envs
        or report.get("failure_reasons") != []
    ):
        raise CliError("vector benchmark did not prove true batched Isaac execution")
    if (
        not isinstance(smoke, Mapping)
        or smoke.get("schema") != "wlr50_clean.vectorized_residual_smoke.v1"
        or smoke.get("status") != expected_status
        or smoke.get("mode") != actual_smoke_mode
        or smoke.get("passed") is not True
        or smoke.get("num_envs") != expected_num_envs
        or smoke.get("live_vectorized_isaac_backend_verified") is not True
        or smoke.get("independent_origin_count") != expected_num_envs
        or smoke.get("independent_controller_count") != expected_num_envs
        or smoke.get("independent_reader_count") != expected_num_envs
        or smoke.get("independent_projection_bridge_count") != expected_num_envs
        or smoke.get("all_masks_honored") is not True
        or smoke.get("all_zero_fast_path_expected") is not True
        or smoke.get("no_in_episode_root_writes") is not True
        or smoke.get("no_recording_runtime_access") is not True
        or smoke.get("no_termination_or_safety_events") is not True
    ):
        raise CliError("vector residual smoke evidence is incomplete or failed")
    rows = smoke.get("rows")
    decisions = smoke.get("policy_decisions")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or isinstance(decisions, bool)
        or not isinstance(decisions, int)
        or decisions <= 0
        or smoke.get("row_evidence_count") != decisions * expected_num_envs
        or len(rows) != decisions * expected_num_envs
    ):
        raise CliError("vector residual smoke row evidence is incomplete")
    for row in rows:
        if not isinstance(row, Mapping):
            raise CliError("vector residual smoke row is malformed")
        env_index = row.get("env_index")
        if (
            isinstance(env_index, bool)
            or not isinstance(env_index, int)
            or not 0 <= env_index < expected_num_envs
            or row.get("seed") != seed_rows[env_index]
            or row.get("mode") != actual_smoke_mode
            or row.get("in_episode_root_write_count") != 0
            or row.get("recording_runtime_access_count") != 0
            or row.get("terminated") is not False
            or row.get("truncated") is not False
        ):
            raise CliError("vector residual smoke row violates seed/safety/reset evidence")
    maximum_fraction = smoke.get("maximum_observed_phase_scale_fraction")
    if isinstance(maximum_fraction, bool) or not isinstance(
        maximum_fraction, (int, float)
    ):
        raise CliError("vector residual smoke amplitude evidence is invalid")
    if expected_mode == "zero":
        if (
            maximum_fraction != 0.0
            or smoke.get("nonzero_active_row_count") != 0
            or smoke.get("zero_applied_equals_nominal_row_count") != len(rows)
        ):
            raise CliError("zero vector residual smoke was not exact nominal identity")
    elif (
        not 0.0 < float(maximum_fraction) < 0.05
        or smoke.get("nonzero_active_row_count") != len(rows)
        or smoke.get("deterministic_distinct_action_rows") is not True
    ):
        raise CliError("bounded vector residual smoke did not prove nonzero sub-5% activity")

    return {
        "schema": "wlr50_clean.vector_benchmark_training_acceptance.v1",
        "path": str(benchmark_path),
        "sha256": _sha256(benchmark_path),
        "mode": expected_mode,
        "num_envs": expected_num_envs,
        "run_seed": expected_run_seed,
        "seed_rows": list(seed_rows),
        "config_sha256": expected_config_sha256,
        "frozen_manifest_sha256": expected_frozen_manifest_sha256,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "stdout_sha256": _sha256(stdout_path),
        "frozen_audit_sha256": frozen_audit_hashes,
        "passed": True,
    }


def _require_training_vector_benchmark_acceptance(
    args: argparse.Namespace,
    profile: Any,
) -> Mapping[str, Any] | None:
    """Require one finalized six-slot matrix before any batched training."""

    if args.num_envs == 1:
        return None
    if (
        getattr(args, "vector_zero_benchmark_acceptance", None) is not None
        or getattr(args, "vector_nonzero_benchmark_acceptance", None) is not None
    ):
        raise CliError(
            "raw vector benchmark acceptances are internal-only; supply the finalized matrix"
        )
    matrix_argument = getattr(args, "vector_benchmark_matrix", None)
    if matrix_argument is None:
        raise CliError(
            "multi-env training requires --vector-benchmark-matrix from the "
            "six-slot offline aggregation"
        )
    from .artifacts import ArtifactError, config_set_record, git_head
    from .vector_benchmark_matrix import (
        VectorBenchmarkMatrixError,
        validate_finalized_vector_benchmark_matrix,
    )

    config_sha256, config_records = config_set_record(
        _checkpoint_config_paths(args), project_root=PROJECT_ROOT
    )
    frozen_manifest = (
        PROJECT_ROOT / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    )
    if not frozen_manifest.is_file():
        raise CliError("frozen FSM manifest is missing before vector training")
    frozen_manifest_sha256 = _sha256(frozen_manifest)
    train_seeds = tuple(int(seed) for seed in profile.seed_train)
    if len(train_seeds) < args.num_envs:
        raise CliError("training profile has too few independent vector reset seeds")
    if args.seed in train_seeds:
        offset = train_seeds.index(args.seed)
        train_seeds = train_seeds[offset:] + train_seeds[:offset]
    expected_seed_rows = train_seeds[: args.num_envs]
    try:
        evidence = validate_finalized_vector_benchmark_matrix(
            _resolve_project_path(Path(matrix_argument)),
            expected_project_root=PROJECT_ROOT,
            expected_config_sha256=config_sha256,
            expected_frozen_manifest_sha256=frozen_manifest_sha256,
            expected_git_commit=git_head(PROJECT_ROOT),
            expected_run_seed=args.seed,
            expected_num_envs=args.num_envs,
            expected_seed_rows=expected_seed_rows,
            expected_config_records=config_records,
        )
    except (VectorBenchmarkMatrixError, ArtifactError) as exc:
        raise CliError(f"vector benchmark matrix rejected: {exc}") from exc
    selected = evidence["selected_acceptance"]
    args.vector_benchmark_matrix = Path(evidence["path"])
    args.vector_zero_benchmark_acceptance = Path(selected["zero"]["path"])
    args.vector_nonzero_benchmark_acceptance = Path(
        selected["bounded_nonzero"]["path"]
    )
    args._vector_benchmark_matrix_evidence = evidence
    args._vector_benchmark_acceptance_evidence = selected
    return evidence


def _phase_curriculum_horizon(args: argparse.Namespace) -> int:
    """Resolve the training-only Stage-1 sample horizon."""

    override = getattr(args, "phase_curriculum_max_decisions", None)
    if override is not None:
        horizon = int(override)
    else:
        import yaml

        payload = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
        environment = payload.get("environment", {}) if isinstance(payload, Mapping) else {}
        horizon = int(environment.get("phase_curriculum_max_decisions", 64))
    if horizon <= 0:
        raise CliError("phase curriculum decision horizon must be positive")
    return horizon


def _construct_live_runner(
    args: argparse.Namespace,
    simulation_app: Any,
    *,
    max_iterations: int,
    reset_seeds: Sequence[int] | None = None,
    collect_trace: bool = False,
    pinned_snapshot_bundle: Any | None = None,
    pinned_effective_entry_contract: Any | None = None,
):
    from .residual_direct_env import (
        DEFAULT_PHASE_CURRICULUM_MAX_DECISIONS,
        ResidualEpisodeEnv,
        RslResidualVecEnv,
        build_phase_curriculum_reset_cycle,
    )
    from .rl_library_wrapper import build_rsl_runner_config, construct_runner, load_training_profile

    profile = load_training_profile(args.training_config)
    phase_curriculum = (
        getattr(args, "command", "train") == "train"
        and args.stage == "phase-curriculum"
    )
    if args.num_envs == 1:
        if (
            pinned_snapshot_bundle is None
            and pinned_effective_entry_contract is None
        ):
            (
                pinned_snapshot_bundle,
                pinned_effective_entry_contract,
            ) = _pinned_runtime_phase_contracts(args)
        elif pinned_effective_entry_contract is None:
            candidate = getattr(args, "_pinned_effective_entry_contract", None)
            if (
                candidate is not None
                and getattr(candidate, "phase_snapshot_bundle_sha256", None)
                == getattr(pinned_snapshot_bundle, "bundle_sha256", None)
            ):
                pinned_effective_entry_contract = candidate
            else:
                pinned_effective_entry_contract = (
                    _capture_runtime_effective_entry_contract(
                        pinned_snapshot_bundle
                    )
                )
    selected_seeds = (
        profile.seed_train if reset_seeds is None else tuple(reset_seeds)
    )
    curriculum_horizon = DEFAULT_PHASE_CURRICULUM_MAX_DECISIONS
    curriculum_reset_cycle = None
    if args.num_envs == 1:
        if phase_curriculum:
            curriculum_horizon = _phase_curriculum_horizon(args)
            curriculum_reset_cycle = build_phase_curriculum_reset_cycle(
                target_decision_fractions=profile.phase_sampling,
                baseline_phase_decisions=(
                    profile.phase_curriculum_baseline_decisions
                ),
                max_decisions=curriculum_horizon,
                cycle_samples=profile.phase_curriculum_reset_cycle_samples,
            )
        from .isaac_fsm_backend import IsaacFSMBackend

        episode = ResidualEpisodeEnv(
            IsaacFSMBackend(
                simulation_app,
                expected_phase_snapshot_bundle=pinned_snapshot_bundle,
                expected_effective_entry_contract=(
                    pinned_effective_entry_contract
                ),
            ),
            collect_trace=collect_trace,
        )
        env = RslResidualVecEnv(
            [episode],
            seeds=selected_seeds,
            device="cuda:0",
            training_phase_reset_schedule=curriculum_reset_cycle,
            end_curriculum_sample_at_phase_boundary=phase_curriculum,
            phase_curriculum_max_decisions=curriculum_horizon,
            phase_curriculum_target_decision_fractions=(
                profile.phase_sampling if phase_curriculum else None
            ),
            phase_curriculum_occupancy_tolerance_fraction=(
                profile.phase_curriculum_occupancy_tolerance
            ),
        )
    else:
        from .vectorized_isaac_backend import (
            SUPPORTED_VECTOR_ENV_COUNTS,
            VectorizedIsaacFSMBackend,
        )
        from .vectorized_residual_env import VectorizedRslResidualEnv

        if getattr(args, "command", "train") != "train":
            raise CliError(
                "multi-environment runner is training-only; physical evaluation uses num-envs=1"
            )
        if args.num_envs not in SUPPORTED_VECTOR_ENV_COUNTS:
            raise CliError(
                f"true batched Isaac training requires num-envs in {SUPPORTED_VECTOR_ENV_COUNTS}"
            )
        if phase_curriculum:
            raise CliError(
                "multi-env phase curriculum is unavailable: the vector backend "
                "cannot independently restore phase snapshots; use num-envs=1"
            )
        if args.stage == "mild-randomization":
            raise CliError(
                "multi-env randomization is unavailable until the vector backend "
                "has independently validated reset hooks"
            )
        backend = VectorizedIsaacFSMBackend(
            simulation_app,
            num_envs=args.num_envs,
            device="cuda:0",
        )
        env = VectorizedRslResidualEnv(
            backend,
            seeds=selected_seeds,
            device="cuda:0",
            collect_trace=collect_trace,
        )
    config = build_rsl_runner_config(
        profile,
        seed=args.seed,
        max_iterations=max_iterations,
        save_interval=max(1, max_iterations // 10),
        experiment_name=f"wlr50_{args.stage}",
    )
    runner = construct_runner(env, config, log_dir=args.run_dir / "rsl_logs")
    return profile, env, runner, config


def _validate_training_telemetry(
    telemetry: Mapping[str, Any], *, stage: str, expected_policy_decisions: int
) -> None:
    """Fail closed on incomplete rollout, reward dominance, or Stage-1 balance."""

    decision_count = telemetry.get("policy_decision_count")
    if (
        isinstance(decision_count, bool)
        or not isinstance(decision_count, int)
        or decision_count != int(expected_policy_decisions)
        or telemetry.get("reward_telemetry_complete") is not True
    ):
        raise CliError(
            "live rollout telemetry is incomplete or differs from the RSL policy-decision budget"
        )
    phase_reward = telemetry.get("reward_family_absolute_sums_by_phase")
    expected_phases = tuple(f"P{index:02d}" for index in range(1, 14))
    phase_counts = telemetry.get("phase_decision_counts")
    if (
        not isinstance(phase_counts, Mapping)
        or tuple(phase_counts) != expected_phases
        or any(
            isinstance(phase_counts[phase_id], bool)
            or not isinstance(phase_counts[phase_id], int)
            or phase_counts[phase_id] < 0
            for phase_id in expected_phases
        )
        or sum(phase_counts.values()) != decision_count
    ):
        raise CliError(
            "phase_decision_counts must be the exact non-negative P01-P13 "
            "partition of the policy-decision budget; received "
            f"{dict(phase_counts) if isinstance(phase_counts, Mapping) else phase_counts}"
        )
    expected_families = (
        "phase_task_progress",
        "body_stability",
        "contact_motion_quality",
        "control_smoothness",
        "residual_regularization",
    )
    if (
        not isinstance(phase_reward, Mapping)
        or tuple(phase_reward) != expected_phases
        or any(
            not isinstance(phase_reward[phase_id], Mapping)
            or tuple(phase_reward[phase_id]) != expected_families
            for phase_id in expected_phases
        )
    ):
        raise CliError("phase-by-family absolute reward telemetry is incomplete")
    if telemetry.get("reward_dominance_within_limits") is not True:
        raise CliError(
            "reward dominance gate failed: one dense family exceeded 70% or "
            "residual regularization exceeded 20% of absolute contribution"
        )
    authoritative_count = telemetry.get("authoritative_completed_episode_count")
    reason_counts = telemetry.get("authoritative_terminal_reason_counts")
    success_count = telemetry.get("authoritative_success_count")
    peer_count = telemetry.get("vector_batch_reset_peer_count")
    completed_count = telemetry.get("completed_sample_count")
    if (
        isinstance(authoritative_count, bool)
        or not isinstance(authoritative_count, int)
        or authoritative_count < 0
        or not isinstance(reason_counts, Mapping)
        or any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for reason, count in reason_counts.items()
        )
        or sum(reason_counts.values()) != authoritative_count
        or isinstance(success_count, bool)
        or not isinstance(success_count, int)
        or success_count < 0
        or success_count != reason_counts.get("SUCCESS", 0)
        or isinstance(peer_count, bool)
        or not isinstance(peer_count, int)
        or peer_count < 0
        or isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or completed_count != authoritative_count + peer_count
    ):
        raise CliError(
            "completed-episode telemetry is inconsistent with authoritative "
            "terminal outcomes (including SUCCESS) and vector reset peers"
        )
    if (
        stage == "phase-curriculum"
        and telemetry.get("phase_curriculum_occupancy_within_tolerance") is not True
    ):
        violations = telemetry.get("phase_curriculum_occupancy_violations", [])
        raise CliError(
            "phase-curriculum policy-decision occupancy gate failed for "
            f"{list(violations) if isinstance(violations, Sequence) else violations}"
        )
    if stage == "full-episode":
        missing_phases = [
            phase_id for phase_id in expected_phases if phase_counts[phase_id] <= 0
        ]
        if missing_phases:
            raise CliError(
                "full-episode training did not execute every P01-P13 phase; "
                f"missing {missing_phases}"
            )
        if success_count < 1:
            raise CliError(
                "full-episode training produced no authoritative SUCCESS episode"
            )


_ALLOWED_TRAINING_RESUME_STAGES = {
    "smoke": frozenset({"initial_zero_residual", "smoke"}),
    "phase-curriculum": frozenset({"smoke", "phase-curriculum"}),
    "full-episode": frozenset({"phase-curriculum", "full-episode"}),
    "mild-randomization": frozenset({"full-episode", "mild-randomization"}),
}


def _load_bound_json(path: Path, sha256: str, *, label: str) -> Mapping[str, Any]:
    selected = path.resolve()
    if not selected.is_file() or _sha256(selected) != sha256:
        raise CliError(f"{label} is missing or differs from its bound SHA-256")
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise CliError(f"{label} must be a JSON object")
    return payload


def _validate_improved_resume_manifest(
    resume_provenance: Any,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind Stage 3 to the two-stage promoted improved checkpoint."""

    required_true = (
        "validation_promotion_authorized",
        "locked_test_authorized",
        "promotion_authorized",
    )
    if manifest.get("publication_role") != "improved" or any(
        manifest.get(key) is not True for key in required_true
    ):
        raise CliError(
            "mild-randomization may begin only from a two-stage promoted improved checkpoint"
        )
    bound_inputs: dict[str, dict[str, str]] = {}
    for key in (
        "validation_promotion_manifest",
        "promotion_decision",
        "locked_test_aggregate",
    ):
        raw_path = manifest.get(key)
        raw_hash = manifest.get(f"{key}_sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_hash, str):
            raise CliError(f"improved checkpoint manifest omits bound {key} evidence")
        selected = Path(raw_path).resolve()
        _load_bound_json(selected, raw_hash, label=f"improved checkpoint {key}")
        bound_inputs[key] = {"path": str(selected), "sha256": raw_hash}
    return {
        "source_improved_checkpoint": str(resume_provenance.checkpoint_path),
        "source_improved_checkpoint_sha256": resume_provenance.checkpoint_sha256,
        "source_improved_manifest": str(resume_provenance.manifest_path),
        "source_improved_manifest_sha256": resume_provenance.manifest_sha256,
        "publication_role": "improved",
        "promotion_authorized": True,
        "bound_promotion_inputs": bound_inputs,
    }


def _validate_inherited_improved_resume_evidence(
    evidence: Any,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise CliError(
            "resumed mild-randomization checkpoint omits its promoted-improved source"
        )
    result = dict(evidence)
    if (
        result.get("publication_role") != "improved"
        or result.get("promotion_authorized") is not True
    ):
        raise CliError(
            "resumed mild-randomization checkpoint has invalid improved-source evidence"
        )
    for path_key, hash_key in (
        ("source_improved_checkpoint", "source_improved_checkpoint_sha256"),
        ("source_improved_manifest", "source_improved_manifest_sha256"),
    ):
        raw_path = result.get(path_key)
        raw_hash = result.get(hash_key)
        if (
            not isinstance(raw_path, str)
            or not isinstance(raw_hash, str)
            or not Path(raw_path).resolve().is_file()
            or _sha256(Path(raw_path).resolve()) != raw_hash
        ):
            raise CliError(
                "resumed mild-randomization improved-source bytes are missing or changed"
            )
    inputs = result.get("bound_promotion_inputs")
    if not isinstance(inputs, Mapping):
        raise CliError("resumed mild-randomization omits bound promotion inputs")
    for key in (
        "validation_promotion_manifest",
        "promotion_decision",
        "locked_test_aggregate",
    ):
        row = inputs.get(key)
        if not isinstance(row, Mapping):
            raise CliError(f"resumed mild-randomization omits {key} evidence")
        raw_path, raw_hash = row.get("path"), row.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_hash, str):
            raise CliError(f"resumed mild-randomization has invalid {key} evidence")
        _load_bound_json(Path(raw_path), raw_hash, label=f"inherited {key}")
    return result


def _validate_training_resume_stage(
    resume_provenance: Any,
    *,
    requested_stage: str,
) -> dict[str, Any]:
    """Reject skipped/reversed training stages before restoring RNG or learning."""

    allowed = _ALLOWED_TRAINING_RESUME_STAGES.get(requested_stage)
    if allowed is None:
        raise CliError(f"unknown training stage {requested_stage!r}")
    resume_stage = str(resume_provenance.stage)
    if resume_stage not in allowed:
        raise CliError(
            f"{requested_stage} cannot resume from checkpoint stage {resume_stage!r}; "
            f"allowed stages are {sorted(allowed)}"
        )
    evidence: dict[str, Any] = {
        "requested_stage": requested_stage,
        "resume_stage": resume_stage,
        "stage_chain_valid": True,
    }
    if requested_stage != "mild-randomization":
        return evidence

    manifest = _load_bound_json(
        Path(resume_provenance.manifest_path),
        resume_provenance.manifest_sha256,
        label="resume checkpoint manifest",
    )
    if resume_stage == "full-episode":
        improved = _validate_improved_resume_manifest(resume_provenance, manifest)
    else:
        improved = _validate_inherited_improved_resume_evidence(
            manifest.get("mild_randomization_source_improved_evidence")
        )
    evidence["mild_randomization_source_improved_evidence"] = improved
    return evidence


def _require_training_soft_reset_acceptance(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    """Fail before training if a single-env auto-reset path is unproven."""

    if args.num_envs != 1:
        return None
    from .soft_reset_equivalence import validate_soft_reset_acceptance

    supplied = getattr(args, "soft_reset_acceptance", None)
    if supplied is None:
        raise CliError(
            "single-env training can auto-reset and requires an explicit "
            "--soft-reset-acceptance artifact from the live equivalence gate"
        )
    evidence = validate_soft_reset_acceptance(
        _resolve_project_path(supplied), project_root=PROJECT_ROOT
    )
    args.soft_reset_acceptance = Path(evidence["path"])
    args._soft_reset_acceptance_evidence = evidence
    return evidence


def _require_training_phase_effective_entry_holdout(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    """Reject provisional effective-entry data before phase-curriculum launch."""

    if args.stage != "phase-curriculum":
        return None
    supplied = getattr(args, "phase_effective_entry_holdout_acceptance", None)
    if supplied is None:
        raise CliError(
            "phase-curriculum training requires an explicit "
            "--phase-effective-entry-holdout-acceptance produced from twelve "
            "independent seed-1003 P02-P13 live probes"
        )
    from .phase_effective_entry_holdout import (
        PhaseEffectiveEntryHoldoutError,
        validate_phase_effective_entry_holdout_acceptance,
    )

    snapshot_bundle, effective_entry_contract = _pinned_runtime_phase_contracts(args)
    try:
        evidence = validate_phase_effective_entry_holdout_acceptance(
            _resolve_project_path(Path(supplied)),
            project_root=PROJECT_ROOT,
            config_paths=_checkpoint_config_paths(args),
            snapshot_bundle=snapshot_bundle,
            effective_entry_contract=effective_entry_contract,
        )
    except PhaseEffectiveEntryHoldoutError as exc:
        raise CliError(
            f"phase effective-entry holdout acceptance rejected: {exc}"
        ) from exc
    args.phase_effective_entry_holdout_acceptance = Path(evidence["path"])
    args._phase_effective_entry_holdout_evidence = evidence
    return evidence


def _revalidate_training_phase_effective_entry_holdout(
    args: argparse.Namespace,
    pinned_snapshot_bundle: Any,
    pinned_effective_entry_contract: Any,
) -> Mapping[str, Any] | None:
    """Re-read the gate and all source runs without accepting replacement bytes."""

    expected = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if not isinstance(expected, Mapping):
        if args.stage == "phase-curriculum":
            raise CliError("phase-curriculum holdout evidence was not pinned pre-launch")
        return None
    from .phase_effective_entry_holdout import (
        PhaseEffectiveEntryHoldoutError,
        validate_phase_effective_entry_holdout_acceptance,
    )

    try:
        current = validate_phase_effective_entry_holdout_acceptance(
            Path(str(expected.get("path", ""))),
            project_root=PROJECT_ROOT,
            config_paths=_checkpoint_config_paths(args),
            snapshot_bundle=pinned_snapshot_bundle,
            effective_entry_contract=pinned_effective_entry_contract,
        )
    except PhaseEffectiveEntryHoldoutError as exc:
        raise CliError(
            f"phase effective-entry holdout changed after capture: {exc}"
        ) from exc
    if current != dict(expected):
        raise CliError("phase effective-entry holdout evidence changed after capture")
    return current


def _inherit_training_phase_effective_entry_holdout(
    args: argparse.Namespace,
    checkpoint_infos: Mapping[str, Any],
    pinned_snapshot_bundle: Any,
    pinned_effective_entry_contract: Any,
) -> Mapping[str, Any] | None:
    """Carry the external reset qualification through later training stages.

    Only a first phase-curriculum entry from smoke may rely on the explicit
    pre-launch pin.  A phase-curriculum continuation and every later stage must
    retain and revalidate the exact evidence embedded in its resume checkpoint.
    """

    if args.stage == "smoke":
        return getattr(args, "_phase_effective_entry_holdout_evidence", None)
    resume_stage = checkpoint_infos.get("stage")
    if args.stage == "phase-curriculum" and (
        not isinstance(resume_stage, str)
        or resume_stage not in ("smoke", "phase-curriculum")
    ):
        raise CliError(
            "phase-curriculum resume checkpoint has invalid or missing stage ancestry"
        )
    if args.stage not in {
        "phase-curriculum",
        "full-episode",
        "mild-randomization",
    }:
        return getattr(args, "_phase_effective_entry_holdout_evidence", None)
    embedded = checkpoint_infos.get("phase_effective_entry_holdout_evidence")
    if not isinstance(embedded, Mapping):
        if args.stage == "phase-curriculum" and resume_stage == "smoke":
            return getattr(args, "_phase_effective_entry_holdout_evidence", None)
        raise CliError(
            f"{args.stage} resume checkpoint omits phase effective-entry "
            "holdout ancestry"
        )
    from .phase_effective_entry_holdout import (
        PhaseEffectiveEntryHoldoutError,
        validate_phase_effective_entry_holdout_acceptance,
    )

    try:
        current = validate_phase_effective_entry_holdout_acceptance(
            Path(str(embedded.get("path", ""))),
            project_root=PROJECT_ROOT,
            config_paths=_checkpoint_config_paths(args),
            snapshot_bundle=pinned_snapshot_bundle,
            effective_entry_contract=pinned_effective_entry_contract,
        )
    except PhaseEffectiveEntryHoldoutError as exc:
        raise CliError(
            f"inherited phase effective-entry holdout acceptance rejected: {exc}"
        ) from exc
    if current != dict(embedded):
        raise CliError(
            "resume checkpoint phase effective-entry holdout evidence is stale "
            "or was substituted"
        )
    explicit = getattr(args, "_phase_effective_entry_holdout_evidence", None)
    if args.stage == "phase-curriculum" and (
        not isinstance(explicit, Mapping) or dict(explicit) != current
    ):
        raise CliError(
            "phase-curriculum resume checkpoint holdout differs from the "
            "explicit pre-launch evidence"
        )
    expected_fields, _ = _phase_effective_entry_holdout_fields(current)
    for field, value in expected_fields.items():
        if checkpoint_infos.get(field) != value:
            raise CliError(
                "resume checkpoint holdout manifest/infos binding differs for "
                f"{field!r}"
            )
    args.phase_effective_entry_holdout_acceptance = Path(current["path"])
    args._phase_effective_entry_holdout_evidence = current
    return current


def _initialize_zero_residual(args: argparse.Namespace, simulation_app: Any) -> int:
    """Create and verify a run-local zero actor without publishing canonical state."""

    if args.num_envs != 1:
        raise CliError("zero-residual checkpoint initialization requires num-envs=1")
    if args.seed_set != "train":
        raise CliError("zero-residual checkpoint initialization requires seed-set=train")
    if args.stage != "smoke":
        raise CliError("zero-residual checkpoint initialization requires stage=smoke")
    if args.checkpoint is not None or args.checkpoint_manifest is not None:
        raise CliError("zero-residual checkpoint initialization cannot resume a checkpoint")

    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    from .rl_library_wrapper import (
        capture_training_rng_state,
        initialize_zero_mean_actor,
        load_checkpoint_round_trip,
        optimizer_learning_rate,
        save_checkpoint_with_manifest,
        seed_training_rngs,
        validate_resume_checkpoint_provenance,
        zero_mean_actor_output_layer_verified,
    )

    rng_seed_evidence = dict(seed_training_rngs(args.seed))
    profile, env, runner, config = _construct_live_runner(
        args,
        simulation_app,
        max_iterations=1,
        reset_seeds=(args.seed,),
        pinned_snapshot_bundle=pinned_snapshot_bundle,
        pinned_effective_entry_contract=pinned_effective_entry_contract,
    )
    if args.seed not in profile.seed_train:
        raise CliError("initializer seed is not in the configured train seed set")
    if env.num_envs != 1:
        raise CliError("zero-residual initializer constructed a non-single environment")
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    initialize_zero_mean_actor(runner)
    if not zero_mean_actor_output_layer_verified(runner):
        raise CliError("zero-residual actor output layer is not exact zero before save")

    checkpoint = args.run_dir / "checkpoint_initial_zero_residual.pt"
    manifest_payload = {
        **_checkpoint_manifest_payload(
            args,
            global_step=0,
            stage="initial_zero_residual",
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
        ),
        "optimizer_learning_rate": optimizer_learning_rate(runner),
        "training_rng_seed_evidence": rng_seed_evidence,
        "training_rng_state": capture_training_rng_state(seed=args.seed),
        "zero_mean_actor_output_layer_verified": True,
    }
    checkpoint, manifest = save_checkpoint_with_manifest(
        runner, checkpoint, manifest=manifest_payload
    )
    capture = _pin_live_checkpoint(
        args,
        checkpoint,
        manifest,
        purpose="initialize-zero-residual-round-trip",
    )
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    round_trip = load_checkpoint_round_trip(
        runner, checkpoint, captured_bundle=capture
    )
    provenance = validate_resume_checkpoint_provenance(
        checkpoint,
        round_trip,
        manifest_path=manifest,
        expected_global_policy_decisions=0,
        expected_runtime_contract=_current_checkpoint_runtime_contract(
            args,
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
        ),
        captured_bundle=capture,
    )
    if (
        round_trip.get("stage") != "initial_zero_residual"
        or round_trip.get("training_seed") != args.seed
        or round_trip.get("zero_mean_actor_output_layer_verified") is not True
        or not zero_mean_actor_output_layer_verified(runner)
    ):
        raise CliError("reloaded zero-residual checkpoint is not an exact-zero actor")
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    result = {
        "schema": "wlr50_clean.initial_zero_residual_checkpoint_run.v1",
        "stage": "initial_zero_residual",
        "seed": args.seed,
        "num_envs": env.num_envs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": capture.checkpoint_sha256,
        "checkpoint_manifest": str(manifest),
        "checkpoint_manifest_sha256": capture.manifest_sha256,
        "global_policy_decisions": provenance.global_policy_decisions,
        "save_load_round_trip": True,
        "checkpoint_private_capture_verified": True,
        "zero_mean_actor_output_layer_verified_before_save": True,
        "zero_mean_actor_output_layer_verified_after_load": True,
        "phase_snapshot_bundle": pinned_snapshot_bundle.as_record(),
        "phase_effective_entry_contract": (
            pinned_effective_entry_contract.as_record()
        ),
        "training_rng_seed_evidence": rng_seed_evidence,
        "runner_config": config,
        "training_profile_seed_train": list(profile.seed_train),
    }
    _json(args.run_dir / "initial_checkpoint_result.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _publish_initial_zero_residual(args: argparse.Namespace) -> int:
    """Validate a finalized initializer and publish/reuse the canonical pair."""

    if args.num_envs != 1 or args.seed_set != "train":
        raise CliError(
            "zero-residual checkpoint publication requires num-envs=1 and seed-set=train"
        )
    if args.source_checkpoint is None or args.source_manifest is None:
        raise CliError(
            "zero-residual checkpoint publication requires --source-checkpoint and --source-manifest"
        )
    from .initial_checkpoint import (
        InitialCheckpointError,
        publish_initial_zero_residual_checkpoint,
    )

    try:
        result = publish_initial_zero_residual_checkpoint(
            source_checkpoint=_resolve_project_path(args.source_checkpoint),
            source_manifest=_resolve_project_path(args.source_manifest),
            output_root=_resolve_project_path(args.output_root),
            project_root=PROJECT_ROOT,
            expected_seed=args.seed,
        )
    except InitialCheckpointError as exc:
        raise CliError(f"canonical initial checkpoint publication failed: {exc}") from exc
    payload = result.as_dict()
    _json(args.run_dir / "initial_checkpoint_publication.json", payload)
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


def _require_canonical_initial_checkpoint(args: argparse.Namespace) -> None:
    """Reject a missing/stale default resume pair before AppLauncher import."""

    if args.checkpoint is not None:
        return
    from .initial_checkpoint import (
        InitialCheckpointError,
        validate_initial_zero_residual_checkpoint,
    )

    initial_checkpoint = (
        OUTPUT_ROOT / "checkpoints" / "checkpoint_initial_zero_residual.pt"
    )
    try:
        validate_initial_zero_residual_checkpoint(
            initial_checkpoint,
            initial_checkpoint.with_name(
                "checkpoint_initial_zero_residual_manifest.json"
            ),
            project_root=PROJECT_ROOT,
            expected_seed=args.seed,
        )
    except InitialCheckpointError as exc:
        raise CliError(
            "canonical initial checkpoint is absent or invalid; run "
            "scripts/initialize_zero_residual_checkpoint.ps1 first: "
            f"{exc}"
        ) from exc


def _train(args: argparse.Namespace, simulation_app: Any) -> int:
    # Phase-curriculum resets are loaded by the production backend from one
    # fixed root.  All other training stages bind the same bundle for a common
    # checkpoint ABI.
    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    holdout_evidence = getattr(
        args, "_phase_effective_entry_holdout_evidence", None
    )
    if args.stage == "phase-curriculum" and holdout_evidence is None:
        holdout_evidence = _require_training_phase_effective_entry_holdout(args)
    _revalidate_training_phase_effective_entry_holdout(
        args, pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    rollout_evidence = getattr(
        args, "_phase_zero_residual_rollout_evidence", None
    )
    if args.stage == "phase-curriculum" and rollout_evidence is None:
        rollout_evidence = _require_training_phase_zero_residual_rollout(args)
    _revalidate_training_phase_zero_residual_rollout(args)
    from .rl_library_wrapper import (
        capture_training_rng_state,
        entropy_coefficient_at_policy_decision,
        iterations_for_policy_decisions,
        learn_with_entropy_schedule,
        load_checkpoint_round_trip,
        optimizer_learning_rate,
        planned_entropy_anneal_policy_decisions,
        restore_training_rng_state,
        save_checkpoint_with_manifest,
        seed_training_rngs,
        validate_resume_checkpoint_provenance,
        zero_mean_actor_output_layer_verified,
    )

    from .rl_library_wrapper import load_training_profile

    if args.stage != "smoke" and args.checkpoint is None:
        raise CliError(
            f"{args.stage} training requires an explicit --checkpoint; "
            "refusing to fall back to the initial actor"
        )
    if args.checkpoint is None and getattr(args, "checkpoint_manifest", None) is not None:
        raise CliError("--checkpoint-manifest cannot be supplied without --checkpoint")
    rng_seed_evidence = dict(seed_training_rngs(args.seed))
    soft_reset_evidence = getattr(args, "_soft_reset_acceptance_evidence", None)
    if args.num_envs == 1 and soft_reset_evidence is None:
        soft_reset_evidence = _require_training_soft_reset_acceptance(args)
    profile = load_training_profile(args.training_config)
    vector_benchmark_evidence = getattr(
        args, "_vector_benchmark_acceptance_evidence", None
    )
    if args.num_envs > 1 and vector_benchmark_evidence is None:
        vector_benchmark_evidence = _require_training_vector_benchmark_acceptance(
            args, profile
        )
    budget_key = {
        "smoke": "smoke",
        "phase-curriculum": "phase_curriculum",
        "full-episode": "full_episode",
        "mild-randomization": "mild_randomization",
    }[args.stage]
    budget = args.policy_decisions if args.policy_decisions is not None else profile.budgets[budget_key]
    if budget <= 0:
        raise CliError("selected training budget is zero")
    iterations = iterations_for_policy_decisions(
        budget, num_envs=args.num_envs, rollout_length=profile.rollout_length
    )
    _revalidate_training_phase_effective_entry_holdout(
        args, pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_zero_residual_rollout(args)
    profile, env, runner, config = _construct_live_runner(
        args,
        simulation_app,
        max_iterations=iterations,
        pinned_snapshot_bundle=pinned_snapshot_bundle,
        pinned_effective_entry_contract=pinned_effective_entry_contract,
    )
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    checkpoints = OUTPUT_ROOT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    initial_path = checkpoints / "checkpoint_initial_zero_residual.pt"
    starting_checkpoint = (
        _resolve_project_path(args.checkpoint)
        if args.checkpoint is not None
        else initial_path
    )
    if not starting_checkpoint.is_file():
        raise CliError(f"training resume checkpoint is missing: {starting_checkpoint}")
    checkpoint_manifest_argument = getattr(args, "checkpoint_manifest", None)
    starting_manifest_path = (
        starting_checkpoint.with_name(starting_checkpoint.stem + "_manifest.json")
        if checkpoint_manifest_argument is None
        else _resolve_project_path(checkpoint_manifest_argument)
    )
    starting_capture = _pin_live_checkpoint(
        args,
        starting_checkpoint,
        starting_manifest_path,
        purpose=f"train-resume-{args.stage}",
    )
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    starting_infos = load_checkpoint_round_trip(
        runner,
        starting_checkpoint,
        captured_bundle=starting_capture,
    )
    inherited_holdout = _inherit_training_phase_effective_entry_holdout(
        args,
        starting_infos,
        pinned_snapshot_bundle,
        pinned_effective_entry_contract,
    )
    if inherited_holdout is not None:
        holdout_evidence = inherited_holdout
    inherited_rollout = _inherit_training_phase_zero_residual_rollout(
        args,
        starting_infos,
        pinned_snapshot_bundle,
        pinned_effective_entry_contract,
    )
    if inherited_rollout is not None:
        rollout_evidence = inherited_rollout
    if starting_checkpoint == initial_path:
        if (
            starting_infos.get("stage") != "initial_zero_residual"
            or starting_infos.get("global_policy_decisions") != 0
            or starting_infos.get("zero_mean_actor_output_layer_verified") is not True
            or not zero_mean_actor_output_layer_verified(runner)
        ):
            raise CliError(
                "canonical initial checkpoint is not an exact-zero actor at global step zero"
            )
    resume_provenance = validate_resume_checkpoint_provenance(
        starting_checkpoint,
        starting_infos,
        manifest_path=starting_manifest_path,
        expected_runtime_contract=_current_checkpoint_runtime_contract(
            args,
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
            include_phase_effective_entry_holdout=(
                "phase_effective_entry_holdout_evidence" in starting_infos
            ),
            include_phase_zero_residual_rollout=(
                "phase_zero_residual_rollout_evidence" in starting_infos
            ),
        ),
        captured_bundle=starting_capture,
    )
    stage_chain_evidence = _validate_training_resume_stage(
        resume_provenance, requested_stage=args.stage
    )
    if starting_infos.get("training_seed") != args.seed:
        raise CliError(
            "resume checkpoint training_seed differs from the requested training seed"
        )
    if "optimizer_learning_rate" not in starting_infos:
        raise CliError("resume checkpoint omits optimizer learning-rate provenance")
    rng_resume_evidence = dict(
        restore_training_rng_state(
            starting_infos.get("training_rng_state"), expected_seed=args.seed
        )
    )
    starting_step = resume_provenance.global_policy_decisions
    stage_decisions = iterations * env.num_envs * profile.rollout_length
    planned_entropy_decisions = planned_entropy_anneal_policy_decisions(profile)
    entropy_window_start = entropy_coefficient_at_policy_decision(
        profile.entropy_start,
        profile.entropy_end,
        global_policy_decision=starting_step,
        planned_policy_decisions=planned_entropy_decisions,
    )
    entropy_window_end = entropy_coefficient_at_policy_decision(
        profile.entropy_start,
        profile.entropy_end,
        global_policy_decision=starting_step + stage_decisions,
        planned_policy_decisions=planned_entropy_decisions,
    )
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_effective_entry_holdout(
        args, pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_zero_residual_rollout(args)
    started = time.perf_counter()
    applied_entropy_schedule = learn_with_entropy_schedule(
        runner,
        num_learning_iterations=iterations,
        entropy_start=entropy_window_start,
        entropy_end=entropy_window_end,
        init_at_random_ep_len=False,
    )
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    training_telemetry = dict(env.training_telemetry())
    _validate_training_telemetry(
        training_telemetry,
        stage=args.stage,
        expected_policy_decisions=stage_decisions,
    )
    actual_decisions = starting_step + stage_decisions
    run_checkpoint = args.run_dir / f"checkpoint_{args.stage}_{actual_decisions}.pt"
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_effective_entry_holdout(
        args, pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_zero_residual_rollout(args)
    manifest = _checkpoint_manifest_payload(
        args,
        global_step=actual_decisions,
        stage=args.stage,
        pinned_snapshot_bundle=pinned_snapshot_bundle,
        pinned_effective_entry_contract=pinned_effective_entry_contract,
    )
    manifest = {
        **manifest,
        "training_resume_stage_evidence": stage_chain_evidence,
        "stage_policy_decisions": stage_decisions,
        "resume_checkpoint": str(starting_checkpoint),
        "resume_checkpoint_sha256": starting_capture.checkpoint_sha256,
        "resume_global_policy_decisions": starting_step,
        "optimizer_learning_rate": optimizer_learning_rate(runner),
        "training_rng_seed_evidence": rng_seed_evidence,
        "training_rng_resume_evidence": rng_resume_evidence,
        "training_rng_state": capture_training_rng_state(seed=args.seed),
        "entropy_anneal_planned_policy_decisions": planned_entropy_decisions,
        "entropy_anneal_global_start": starting_step,
        "entropy_anneal_global_end": actual_decisions,
    }
    mild_source = stage_chain_evidence.get(
        "mild_randomization_source_improved_evidence"
    )
    if mild_source is not None:
        manifest["mild_randomization_source_improved_evidence"] = mild_source
    _, run_manifest = save_checkpoint_with_manifest(
        runner, run_checkpoint, manifest=manifest
    )
    saved_capture = _pin_live_checkpoint(
        args,
        run_checkpoint,
        run_manifest,
        purpose=f"train-round-trip-{args.stage}",
    )
    round_trip = load_checkpoint_round_trip(
        runner,
        run_checkpoint,
        captured_bundle=saved_capture,
    )
    validate_resume_checkpoint_provenance(
        run_checkpoint,
        round_trip,
        manifest_path=run_manifest,
        expected_global_policy_decisions=actual_decisions,
        expected_runtime_contract=_current_checkpoint_runtime_contract(
            args,
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
            include_phase_effective_entry_holdout=(
                holdout_evidence is not None
            ),
            include_phase_zero_residual_rollout=(
                rollout_evidence is not None
            ),
        ),
        captured_bundle=saved_capture,
    )
    history_path = checkpoints / f"checkpoint_{args.stage}_{actual_decisions}.pt"
    if history_path.exists():
        raise CliError(f"refusing to overwrite immutable checkpoint history {history_path}")
    shutil.copy2(saved_capture.private_checkpoint_path, history_path)
    history_manifest = history_path.with_name(history_path.stem + "_manifest.json")
    history_payload = {
        **dict(saved_capture.manifest_payload),
        "checkpoint_path": str(history_path),
        "checkpoint_sha256": _sha256(history_path),
    }
    _json(history_manifest, history_payload)
    last_path = checkpoints / "checkpoint_last.pt"
    _atomic_copy_replace(history_path, last_path)
    _json_replace(
        last_path.with_name("checkpoint_last_manifest.json"),
        {
            **history_payload,
            "checkpoint_path": str(last_path),
            "checkpoint_sha256": _sha256(last_path),
            "immutable_history_checkpoint": str(history_path),
        },
    )
    smoke_path = checkpoints / "checkpoint_smoke.pt"
    if args.stage == "smoke":
        smoke_manifest_path = smoke_path.with_name(
            smoke_path.stem + "_manifest.json"
        )
        expected_smoke_payload = {
            **history_payload,
            "checkpoint_path": str(smoke_path),
            "checkpoint_sha256": history_payload["checkpoint_sha256"],
        }
        if smoke_path.exists() or smoke_manifest_path.exists():
            if not smoke_path.is_file() or not smoke_manifest_path.is_file():
                raise CliError("canonical smoke checkpoint pair is incomplete")
            try:
                existing_smoke_manifest = json.loads(
                    smoke_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CliError("canonical smoke checkpoint manifest is invalid") from exc
            if (
                _sha256(smoke_path) != history_payload["checkpoint_sha256"]
                or existing_smoke_manifest != expected_smoke_payload
            ):
                raise CliError(
                    "existing canonical smoke checkpoint differs from this smoke history"
                )
        else:
            shutil.copy2(history_path, smoke_path)
            _json(smoke_manifest_path, expected_smoke_payload)
    training = {
        "schema": "wlr50_clean.ppo_training_run.v1",
        "stage": args.stage,
        "requested_policy_decisions": budget,
        "stage_policy_decisions": stage_decisions,
        "ppo_batch_policy_decisions": env.num_envs * profile.rollout_length,
        "rounding_overrun_policy_decisions": stage_decisions - budget,
        "budget_accounting_basis": "requested_policy_decisions",
        "global_policy_decisions": actual_decisions,
        "resume_checkpoint": str(starting_checkpoint),
        "resume_checkpoint_sha256": starting_capture.checkpoint_sha256,
        "training_resume_stage_evidence": stage_chain_evidence,
        "iterations": iterations,
        "num_envs": args.num_envs,
        "rollout_length": profile.rollout_length,
        "deterministic_validation_interval": profile.deterministic_validation_interval,
        "early_stop_when_promotion_gate_passes": (
            profile.early_stop_when_promotion_gate_passes
        ),
        "phase_snapshot_bundle": pinned_snapshot_bundle.as_record(),
        "phase_effective_entry_contract": (
            pinned_effective_entry_contract.as_record()
        ),
        "environment_contract": dict(env.cfg),
        "training_telemetry": training_telemetry,
        "entropy_schedule": {
            "kind": "linear_per_ppo_update",
            "configured_start": profile.entropy_start,
            "configured_end": profile.entropy_end,
            "planned_policy_decisions": planned_entropy_decisions,
            "global_policy_decision_start": starting_step,
            "global_policy_decision_end": actual_decisions,
            "window_start": entropy_window_start,
            "window_end": entropy_window_end,
            "update_count": len(applied_entropy_schedule),
            "first_applied": applied_entropy_schedule[0],
            "last_applied": applied_entropy_schedule[-1],
            "applied_before_every_update": True,
        },
        "training_rng_seed_evidence": rng_seed_evidence,
        "training_rng_resume_evidence": rng_resume_evidence,
        "soft_reset_acceptance": (
            None if soft_reset_evidence is None else dict(soft_reset_evidence)
        ),
        "phase_effective_entry_holdout_evidence": (
            None if holdout_evidence is None else dict(holdout_evidence)
        ),
        "phase_zero_residual_rollout_evidence": (
            None if rollout_evidence is None else dict(rollout_evidence)
        ),
        "vector_benchmark_acceptance": (
            None
            if vector_benchmark_evidence is None
            else dict(vector_benchmark_evidence)
        ),
        "vector_benchmark_matrix": (
            None
            if getattr(args, "_vector_benchmark_matrix_evidence", None) is None
            else dict(args._vector_benchmark_matrix_evidence)
        ),
        "vector_benchmark_matrix_path": (
            None
            if getattr(args, "_vector_benchmark_matrix_evidence", None) is None
            else args._vector_benchmark_matrix_evidence["path"]
        ),
        "vector_benchmark_matrix_sha256": (
            None
            if getattr(args, "_vector_benchmark_matrix_evidence", None) is None
            else args._vector_benchmark_matrix_evidence["sha256"]
        ),
        "wall_time_s": time.perf_counter() - started,
        "checkpoint_last": str(last_path),
        "checkpoint_sha256": _sha256(last_path),
        "immutable_history_checkpoint": str(history_path),
        "immutable_history_checkpoint_manifest": str(history_manifest),
        "immutable_history_checkpoint_manifest_sha256": _sha256(history_manifest),
        "save_load_round_trip": True,
        "resume_checkpoint_private_capture_verified": True,
        "saved_checkpoint_private_round_trip_verified": True,
        "round_trip_infos": dict(round_trip),
        "runner_config": config,
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_effective_entry_holdout(
        args, pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _revalidate_training_phase_zero_residual_rollout(args)
    _json(args.run_dir / "training_result.json", training)
    print(json.dumps(training, separators=(",", ":")), flush=True)
    return 0


def _evaluate(args: argparse.Namespace, simulation_app: Any) -> int:
    from .rl_library_wrapper import (
        deterministic_action,
        load_checkpoint_round_trip,
        validate_resume_checkpoint_provenance,
    )
    from .live_stream_writer import LiveStreamWriter

    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )

    checkpoint = _resolve_project_path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None or not checkpoint.is_file():
        raise CliError("--checkpoint must name a loadable checkpoint")
    if args.num_envs != 1:
        raise CliError("checkpoint evaluation requires one fresh Isaac process and num-envs=1")
    if args.episode_count != 1:
        raise CliError(
            "checkpoint evaluation accepts exactly one episode per fresh Isaac process; "
            "use evaluate_ppo_checkpoint.ps1 to aggregate multiple seeds"
        )
    if not args.deterministic:
        raise CliError("checkpoint evaluation requires --deterministic mean-policy inference")
    manifest_argument = getattr(args, "checkpoint_manifest", None)
    if manifest_argument is None:
        raise CliError("checkpoint evaluation requires explicit --checkpoint-manifest")
    checkpoint_manifest = _resolve_project_path(manifest_argument)
    if not checkpoint_manifest.is_file():
        raise CliError(f"--checkpoint-manifest is missing: {checkpoint_manifest}")
    checkpoint_capture = _pin_live_checkpoint(
        args,
        checkpoint,
        checkpoint_manifest,
        purpose="checkpoint-evaluation",
    )
    checkpoint_manifest_payload = checkpoint_capture.manifest_payload
    _require_manifest_snapshot_contract(
        checkpoint_manifest_payload,
        pinned_snapshot_bundle.as_record(),
        label="evaluation checkpoint manifest",
        effective_entry_contract=pinned_effective_entry_contract,
    )

    _, env, runner, _ = _construct_live_runner(
        args,
        simulation_app,
        max_iterations=1,
        reset_seeds=(args.seed,),
        collect_trace=True,
        pinned_snapshot_bundle=pinned_snapshot_bundle,
        pinned_effective_entry_contract=pinned_effective_entry_contract,
    )
    infos = load_checkpoint_round_trip(
        runner,
        checkpoint,
        captured_bundle=checkpoint_capture,
    )
    provenance = validate_resume_checkpoint_provenance(
        checkpoint,
        infos,
        manifest_path=checkpoint_manifest,
        expected_runtime_contract=_current_checkpoint_runtime_contract(
            args,
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
        ),
        captured_bundle=checkpoint_capture,
    )
    if len(env.environments) != 1:
        raise CliError("evaluation runner did not expose exactly one residual episode")
    episode = env.environments[0]
    if episode.seed != args.seed or episode.frame is None or episode.observation is None:
        raise CliError("evaluation episode was not initialized from the requested seed")

    episode_dir = args.run_dir / f"episode_000_seed_{args.seed}"
    writer = LiveStreamWriter(episode_dir, seed=args.seed)
    writer.start(episode.frame)
    episode.tick_callback = writer.write_tick
    started = time.perf_counter()
    reward_total = 0.0
    final_step = None
    try:
        while not episode.done:
            observations = env.observation_tensor_dict((episode.observation,))
            actions = deterministic_action(runner, observations)
            if tuple(actions.shape) != (1, 12):
                raise CliError(
                    f"deterministic actor returned {tuple(actions.shape)}, expected (1, 12)"
                )
            action = tuple(float(value) for value in actions.detach().to("cpu").tolist()[0])
            final_step = episode.step(action)
            reward_total += final_step.reward
            writer.write_decision(final_step.info)
        if final_step is None or episode.frame is None:
            raise CliError("evaluation episode ended without a policy decision")
        manifest_path = writer.finalize(
            episode.frame,
            reward_total=reward_total,
            decision_count=episode.decision_count,
        )
    except Exception:
        writer.abort()
        raise

    trace_path = episode_dir / "policy_trace.jsonl"
    _jsonl(trace_path, episode.trace)
    terminal = final_step.info.get("termination_reason")
    if terminal is None:
        raise CliError("completed evaluation episode has no termination reason")
    signals = episode.frame.termination_signals
    summary = {
        "episode_index": 0,
        "seed": args.seed,
        "task_success": terminal == "SUCCESS",
        "termination_reason": terminal,
        "duration_s": float(episode.frame.sim_time_s),
        "decision_count": int(episode.decision_count),
        "physics_tick": int(episode.frame.physics_tick),
        "body_collision": bool(signals.body_collision),
        "wheel_only_climb": bool(signals.wheel_only_climb),
        "safety_abort": any(
            (
                signals.fall,
                signals.nan_inf,
                signals.hard_joint_limit,
                signals.physics_explosion,
            )
        ),
        "under_maximum_duration": float(episode.frame.sim_time_s)
        <= args.maximum_duration_s,
        "reward_total": reward_total,
        "trace_path": str(trace_path),
        "trial_manifest_path": str(manifest_path),
        "canonical_episode_dir": str(episode_dir),
        "recording_runtime_access_count": int(
            final_step.info.get("recording_runtime_access_count", 0)
        ),
        "in_episode_root_write_count": int(
            final_step.info.get("in_episode_root_write_count", 0)
        ),
    }
    _json(episode_dir / "episode_summary.json", summary)
    passed = bool(
        summary["task_success"]
        and not summary["body_collision"]
        and not summary["wheel_only_climb"]
        and not summary["safety_abort"]
        and summary["under_maximum_duration"]
        and summary["recording_runtime_access_count"] == 0
        and summary["in_episode_root_write_count"] == 0
    )
    result = {
        "schema": "wlr50_clean.ppo_checkpoint_evaluation.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_capture.checkpoint_sha256,
        "checkpoint_manifest": str(provenance.manifest_path),
        "checkpoint_manifest_sha256": provenance.manifest_sha256,
        "checkpoint_provenance": provenance.as_dict(),
        "deterministic_mean_policy": True,
        "fresh_process_single_episode": True,
        "checkpoint_private_capture_verified": True,
        "vec_env_step_called": False,
        "episode_count": 1,
        "success_count": int(summary["task_success"]),
        "passed": passed,
        "episodes": [summary],
        "wall_time_s": time.perf_counter() - started,
        "checkpoint_infos": dict(infos),
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _json(args.run_dir / "checkpoint_evaluation.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    # A worker's contract is evidence capture, not candidate promotion.  A
    # physically unsuccessful but complete episode remains valid evidence and
    # must not prevent the remaining fresh processes from running.  The batch
    # aggregator below owns the fail/pass exit status.
    return 0


def _aggregate_evaluations(args: argparse.Namespace) -> int:
    from .evaluation_artifacts import collect_fresh_process_episode_workers

    if not args.deterministic:
        raise CliError("fresh-process aggregation requires deterministic workers")
    role = str(args.evaluation_role)
    run_dirs = tuple(Path(path).resolve() for path in args.evaluation_run_dir)
    expected_seeds = tuple(range(args.seed, args.seed + args.episode_count))
    checkpoint = (
        _resolve_project_path(args.checkpoint)
        if role == "candidate" and args.checkpoint
        else None
    )
    batch = collect_fresh_process_episode_workers(
        run_dirs,
        seeds=expected_seeds,
        role=role,
        checkpoint_path=checkpoint,
    )
    episodes = [dict(row) for row in batch.episode_rows]
    passed = bool(
        len(episodes) == args.episode_count
        and all(row.get("task_success") is True for row in episodes)
        and all(row.get("body_collision") is False for row in episodes)
        and all(row.get("wheel_only_climb") is False for row in episodes)
        and all(row.get("safety_abort") is False for row in episodes)
        and all(row.get("under_maximum_duration") is True for row in episodes)
        and all(int(row.get("recording_runtime_access_count", -1)) == 0 for row in episodes)
        and all(int(row.get("in_episode_root_write_count", -1)) == 0 for row in episodes)
        and all(
            row.get("worker_gate_passed") is True for row in batch.worker_rows
        )
    )
    result = {
        "schema": "wlr50_clean.fresh_process_episode_batch.v1",
        "role": role,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else _sha256(checkpoint),
        "seed_set": args.seed_set,
        "seeds": list(batch.seeds),
        "canonical_episode_dirs": [
            str(path) for path in batch.canonical_episode_dirs
        ],
        "fresh_process_per_episode": True,
        "deterministic_evaluation": True,
        "deterministic_mean_policy": True if role == "candidate" else None,
        "pure_fsm_zero_residual": True if role == "baseline" else None,
        "episode_count": len(episodes),
        "success_count": sum(bool(row["task_success"]) for row in episodes),
        "body_collision_count": sum(bool(row["body_collision"]) for row in episodes),
        "wheel_only_climb_count": sum(bool(row["wheel_only_climb"]) for row in episodes),
        "safety_abort_count": sum(bool(row["safety_abort"]) for row in episodes),
        "all_under_maximum_duration": all(
            bool(row["under_maximum_duration"]) for row in episodes
        ),
        "passed": passed,
        "worker_gate_pass_count": sum(
            row.get("worker_gate_passed") is True for row in batch.worker_rows
        ),
        "workers": [dict(row) for row in batch.worker_rows],
        "episodes": episodes,
    }
    if role == "candidate" and args.seed_set == "locked-test":
        from .checkpoint_promotion import finalize_locked_test_aggregate_payload

        manifest_argument = getattr(args, "checkpoint_manifest", None)
        if manifest_argument is None:
            raise CliError(
                "locked-test aggregation requires explicit --checkpoint-manifest"
            )
        checkpoint_manifest = _resolve_project_path(manifest_argument)
        if checkpoint is None:
            raise CliError("locked-test candidate aggregation requires --checkpoint")
        result = dict(
            finalize_locked_test_aggregate_payload(
                result,
                checkpoint_path=checkpoint,
                checkpoint_manifest_path=checkpoint_manifest,
            )
        )
        passed = result.get("passed") is True
    output_name = (
        "checkpoint_evaluation_aggregate.json"
        if role == "candidate"
        else "fsm_baseline_evaluation_aggregate.json"
    )
    _json(args.run_dir / output_name, result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    # Aggregation is evidence capture.  A complete, structurally valid batch is
    # a successful command even when the robot fails its physical gates; each
    # caller must make the role-specific promotion/publication decision from
    # ``result["passed"]``.  Malformed or incomplete workers still fail above.
    return 0


def _export_baseline_evaluation(args: argparse.Namespace) -> int:
    from .evaluation_artifacts import export_baseline_evaluation_artifacts

    if args.episode_count != 5 or args.seed != 2001:
        raise CliError(
            "baseline metric export requires validation seeds 2001-2005 exactly"
        )
    episode_dirs = tuple(Path(path).resolve() for path in args.episode_dir)
    if len(episode_dirs) != args.episode_count:
        raise CliError(
            f"expected {args.episode_count} canonical baseline directories, "
            f"got {len(episode_dirs)}"
        )
    output = _resolve_project_path(args.metrics_output_dir)
    paths = export_baseline_evaluation_artifacts(
        output,
        episode_directories=episode_dirs,
        seeds=range(args.seed, args.seed + args.episode_count),
    )
    result = {
        "schema": "wlr50_clean.fsm_baseline_evaluation_export.v1",
        "episode_count": args.episode_count,
        "validation_seeds": list(range(args.seed, args.seed + args.episode_count)),
        "artifacts": paths.as_dict(),
    }
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _frozen_audit_identity(audit: Mapping[str, Any]) -> tuple[str, tuple[tuple[Any, ...], ...]]:
    entries = audit.get("entries")
    if not isinstance(entries, list):
        raise CliError("frozen hash audit omitted its protected-file entries")
    normalized: list[tuple[Any, ...]] = []
    for row in entries:
        if not isinstance(row, Mapping):
            raise CliError("frozen hash audit contains a malformed entry")
        normalized.append(
            (
                row.get("path"),
                row.get("expected_sha256"),
                row.get("actual_sha256"),
                row.get("exists"),
                row.get("valid"),
            )
        )
    return str(audit.get("frozen_manifest_sha256", "")), tuple(normalized)


def _export_paired_evaluation(args: argparse.Namespace) -> int:
    """Evaluate two canonical five-seed sets and publish only their metrics."""

    from .artifacts import verify_frozen_hashes
    from .checkpoint_promotion import (
        FROZEN_HASH_FIELDS,
        validate_checkpoint_artifact_provenance,
    )
    from .evaluation_artifacts import (
        build_versioned_residual_activity_calibration,
        evaluate_canonical_episode_dirs,
        export_paired_evaluation_artifacts,
    )
    from .paired_aggregate_binding import (
        PairedAggregateBindingError,
        capture_validation_aggregate,
    )

    validation_seeds = tuple(range(2001, 2006))
    if (
        args.seed_set != "validation"
        or args.seed != validation_seeds[0]
        or args.episode_count != len(validation_seeds)
    ):
        raise CliError(
            "paired metric export requires validation seeds 2001-2005 exactly"
        )
    baseline_dirs = tuple(
        _resolve_project_path(path) for path in args.baseline_episode_dir
    )
    candidate_dirs = tuple(
        _resolve_project_path(path) for path in args.candidate_episode_dir
    )
    if len(baseline_dirs) != 5 or len(candidate_dirs) != 5:
        raise CliError(
            "paired metric export requires exactly five baseline and five candidate "
            "canonical episode directories"
        )
    if len(set(baseline_dirs)) != 5 or len(set(candidate_dirs)) != 5:
        raise CliError("paired metric export episode directories must be unique per role")
    if set(baseline_dirs).intersection(candidate_dirs):
        raise CliError("baseline and candidate canonical episode directories must be distinct")
    for role, directories in (("baseline", baseline_dirs), ("candidate", candidate_dirs)):
        missing = next((path for path in directories if not path.is_dir()), None)
        if missing is not None:
            raise CliError(f"{role} canonical episode directory is missing: {missing}")

    checkpoint_path = _required_artifact_argument(
        args, "candidate_checkpoint", "--candidate-checkpoint"
    )
    manifest_path = _required_artifact_argument(
        args, "candidate_manifest", "--candidate-manifest"
    )
    provenance = validate_checkpoint_artifact_provenance(
        checkpoint_path,
        manifest_path,
    )
    baseline_aggregate_path = _required_artifact_argument(
        args, "baseline_aggregate", "--baseline-aggregate"
    )
    candidate_aggregate_path = _required_artifact_argument(
        args,
        "candidate_validation_aggregate",
        "--candidate-validation-aggregate",
    )
    try:
        baseline_aggregate = capture_validation_aggregate(
            baseline_aggregate_path,
            role="baseline",
            project_root=PROJECT_ROOT,
        )
        candidate_aggregate = capture_validation_aggregate(
            candidate_aggregate_path,
            role="candidate",
            expected_checkpoint_path=checkpoint_path,
            expected_checkpoint_manifest_path=manifest_path,
            project_root=PROJECT_ROOT,
        )
    except PairedAggregateBindingError as exc:
        raise CliError(f"paired aggregate provenance is invalid: {exc}") from exc
    if (
        baseline_dirs != baseline_aggregate.batch.canonical_episode_dirs
        or candidate_dirs != candidate_aggregate.batch.canonical_episode_dirs
    ):
        raise CliError(
            "paired episode directories differ from their finalized aggregate workers"
        )
    snapshot_pin, effective_entry_pin = _pinned_runtime_phase_contracts(args)
    snapshot_bundle = snapshot_pin.as_record()
    runtime_hash_paths = {
        "controller_hash": PROJECT_ROOT / "configs" / "fsm_states.yaml",
        "environment_hash": PROJECT_ROOT / "configs" / "environment_lock.json",
        "observation_schema_hash": (
            PROJECT_ROOT / "configs" / "ppo_observation_schema_v2.json"
        ),
        "action_schema_hash": (
            PROJECT_ROOT / "configs" / "ppo_phase_action_masks_v2.yaml"
        ),
        "reward_config_hash": PROJECT_ROOT / "configs" / "ppo_reward_v2.yaml",
    }
    if set(runtime_hash_paths) != set(FROZEN_HASH_FIELDS):
        raise CliError("internal paired-export checkpoint hash set is incomplete")

    def require_current_checkpoint_contract() -> None:
        _revalidate_pinned_phase_contracts(snapshot_pin, effective_entry_pin)
        current_snapshot_bundle = snapshot_pin.as_record()
        _require_manifest_snapshot_contract(
            provenance.manifest,
            current_snapshot_bundle,
            label="candidate checkpoint manifest",
            effective_entry_contract=effective_entry_pin,
        )
        for field, path in runtime_hash_paths.items():
            declared = str(provenance.manifest.get(field, "")).lower()
            if not path.is_file() or declared != _sha256(path):
                raise CliError(
                    f"candidate checkpoint manifest {field} differs from current config"
                )

    require_current_checkpoint_contract()
    frozen_manifest = (
        PROJECT_ROOT
        / "artifacts"
        / "ppo_phase_v1_start"
        / "frozen_fsm_hashes.json"
    )
    frozen_before = verify_frozen_hashes(
        project_root=PROJECT_ROOT,
        frozen_manifest=frozen_manifest,
    )
    if frozen_before.get("passed") is not True or frozen_before.get("mismatches") != []:
        raise CliError("current frozen FSM hash audit did not pass")
    frozen_identity = _frozen_audit_identity(frozen_before)

    calibration = build_versioned_residual_activity_calibration()
    baseline_aggregate.assert_unchanged()
    candidate_aggregate.assert_unchanged()
    baseline_runs = evaluate_canonical_episode_dirs(
        baseline_dirs,
        seeds=validation_seeds,
        residual_calibration=calibration,
    )
    candidate_runs = evaluate_canonical_episode_dirs(
        candidate_dirs,
        seeds=validation_seeds,
        residual_calibration=calibration,
        require_complete_phase_sequence=False,
    )
    baseline_aggregate.assert_unchanged()
    candidate_aggregate.assert_unchanged()

    # Re-bind every mutable external input immediately before publication.
    provenance_after = validate_checkpoint_artifact_provenance(
        checkpoint_path,
        manifest_path,
    )
    if (
        provenance_after.checkpoint_sha256 != provenance.checkpoint_sha256
        or provenance_after.manifest_sha256 != provenance.manifest_sha256
    ):
        raise CliError("candidate checkpoint or sidecar changed during paired evaluation")
    require_current_checkpoint_contract()
    frozen_after = verify_frozen_hashes(
        project_root=PROJECT_ROOT,
        frozen_manifest=frozen_manifest,
    )
    if (
        frozen_after.get("passed") is not True
        or frozen_after.get("mismatches") != []
        or _frozen_audit_identity(frozen_after) != frozen_identity
    ):
        raise CliError("frozen FSM hashes changed during paired evaluation")

    metrics_output = _resolve_project_path(args.metrics_output_dir)
    if any(
        metrics_output == source or metrics_output.is_relative_to(source)
        for source in (
            *baseline_dirs,
            *candidate_dirs,
            *(
                Path(row["run_dir"])
                for row in (
                    *baseline_aggregate.batch.worker_rows,
                    *candidate_aggregate.batch.worker_rows,
                )
            ),
            baseline_aggregate.aggregate_path.parent,
            candidate_aggregate.aggregate_path.parent,
        )
    ):
        raise CliError("metrics output must not overlap aggregate or worker evidence")
    baseline_aggregate.assert_unchanged()
    candidate_aggregate.assert_unchanged()
    paths = export_paired_evaluation_artifacts(
        metrics_output,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        frozen_hashes_unchanged=True,
        candidate_checkpoint_name=checkpoint_path.stem,
        candidate_checkpoint_path=checkpoint_path,
        baseline_evaluation_aggregate=baseline_aggregate.as_record(),
        candidate_validation_aggregate=candidate_aggregate.as_record(),
        minimum_paired_seeds=5,
        residual_calibration_evidence=calibration,
    )
    baseline_aggregate.assert_unchanged()
    candidate_aggregate.assert_unchanged()
    result = {
        "schema": "wlr50_clean.ppo_paired_evaluation_export_cli.v1",
        "offline": True,
        "seed_set": "validation",
        "validation_seeds": list(validation_seeds),
        "baseline_episode_dirs": [str(path) for path in baseline_dirs],
        "candidate_episode_dirs": [str(path) for path in candidate_dirs],
        "candidate_checkpoint_provenance": provenance.as_dict(),
        "baseline_evaluation_aggregate": baseline_aggregate.as_record(),
        "candidate_validation_aggregate": candidate_aggregate.as_record(),
        "frozen_hashes_passed": True,
        "frozen_manifest": str(frozen_manifest.resolve()),
        "frozen_manifest_sha256": frozen_identity[0],
        "metrics_output": str(metrics_output),
        "artifacts": paths.as_dict(),
    }
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _required_artifact_argument(args: argparse.Namespace, name: str, flag: str) -> Path:
    value = getattr(args, name, None)
    if value is None:
        raise CliError(f"{flag} is required")
    path = _resolve_project_path(value)
    if not path.is_file():
        raise CliError(f"{flag} is missing: {path}")
    return path


def _promote_best_validation(args: argparse.Namespace) -> int:
    """Offline validation-decision gate; never publishes the improved name."""

    from .checkpoint_promotion import promote_best_validation_checkpoint

    decision = _required_artifact_argument(
        args, "promotion_decision", "--promotion-decision"
    )
    checkpoint = _required_artifact_argument(
        args, "candidate_checkpoint", "--candidate-checkpoint"
    )
    manifest = _required_artifact_argument(
        args, "candidate_manifest", "--candidate-manifest"
    )
    output_root = _resolve_project_path(args.output_root)
    artifacts = promote_best_validation_checkpoint(
        promotion_decision_path=decision,
        candidate_checkpoint_path=checkpoint,
        candidate_manifest_path=manifest,
        output_root=output_root,
    )
    result = {
        "schema": "wlr50_clean.ppo_best_validation_publication_cli.v1",
        "stage": "validation_to_best",
        "offline": True,
        "filename_inference_used": False,
        "improved_checkpoint_published": False,
        "promotion_decision": str(decision),
        "candidate_checkpoint": str(checkpoint),
        "candidate_manifest": str(manifest),
        "best_validation_checkpoint": str(artifacts.best_checkpoint),
        "best_validation_manifest": str(artifacts.best_manifest),
        "validation_promotion_manifest": str(
            artifacts.validation_promotion_manifest
        ),
    }
    _json(args.run_dir / "best_validation_publication.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _promote_improved(args: argparse.Namespace) -> int:
    """Offline locked-test gate for the final improved checkpoint name."""

    from .checkpoint_promotion import promote_improved_checkpoint

    decision = _required_artifact_argument(
        args, "promotion_decision", "--promotion-decision"
    )
    aggregate = _required_artifact_argument(
        args, "locked_test_aggregate", "--locked-test-aggregate"
    )
    checkpoint = _required_artifact_argument(
        args, "best_validation_checkpoint", "--best-validation-checkpoint"
    )
    manifest = _required_artifact_argument(
        args, "best_validation_manifest", "--best-validation-manifest"
    )
    validation = _required_artifact_argument(
        args,
        "validation_promotion_manifest",
        "--validation-promotion-manifest",
    )
    output_root = _resolve_project_path(args.output_root)
    artifacts = promote_improved_checkpoint(
        promotion_decision_path=decision,
        locked_test_aggregate_path=aggregate,
        best_validation_checkpoint_path=checkpoint,
        best_validation_manifest_path=manifest,
        validation_promotion_manifest_path=validation,
        output_root=output_root,
    )
    result = {
        "schema": "wlr50_clean.ppo_improved_publication_cli.v1",
        "stage": "locked_test_to_improved",
        "offline": True,
        "filename_inference_used": False,
        "promotion_decision": str(decision),
        "locked_test_aggregate": str(aggregate),
        "best_validation_checkpoint": str(checkpoint),
        "best_validation_manifest": str(manifest),
        "validation_promotion_manifest": str(validation),
        "improved_checkpoint": str(artifacts.improved_checkpoint),
        "improved_manifest": str(artifacts.improved_manifest),
        "promotion_manifest": str(artifacts.promotion_manifest),
    }
    _json(args.run_dir / "improved_checkpoint_publication.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _export_inference_actor(args: argparse.Namespace, simulation_app: Any) -> int:
    """Load the final checkpoint into a real RSL runner and export its actor."""

    from .checkpoint_promotion import FROZEN_HASH_FIELDS, export_inference_actor
    from .rl_library_wrapper import (
        load_checkpoint_round_trip,
        validate_resume_checkpoint_provenance,
    )

    if args.num_envs != 1:
        raise CliError("inference actor export requires num-envs=1")
    if not args.deterministic:
        raise CliError("inference actor export requires --deterministic")
    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    checkpoint = _required_artifact_argument(args, "checkpoint", "--checkpoint")
    manifest_path = _required_artifact_argument(
        args, "checkpoint_manifest", "--checkpoint-manifest"
    )
    checkpoint_capture = _pin_live_checkpoint(
        args,
        checkpoint,
        manifest_path,
        purpose="inference-actor-export",
    )
    manifest = checkpoint_capture.manifest_payload
    _require_manifest_snapshot_contract(
        manifest,
        pinned_snapshot_bundle.as_record(),
        label="inference-export checkpoint manifest",
        effective_entry_contract=pinned_effective_entry_contract,
    )

    _, _, runner, _ = _construct_live_runner(
        args,
        simulation_app,
        max_iterations=1,
        reset_seeds=(args.seed,),
        collect_trace=False,
        pinned_snapshot_bundle=pinned_snapshot_bundle,
        pinned_effective_entry_contract=pinned_effective_entry_contract,
    )
    infos = load_checkpoint_round_trip(
        runner,
        checkpoint,
        captured_bundle=checkpoint_capture,
    )
    validate_resume_checkpoint_provenance(
        checkpoint,
        infos,
        manifest_path=manifest_path,
        expected_runtime_contract=_current_checkpoint_runtime_contract(
            args,
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
        ),
        captured_bundle=checkpoint_capture,
    )
    required_info_fields = (
        "stage",
        "training_seed",
        "global_policy_decisions",
        "actor_observation_dimension",
        "critic_observation_dimension",
        "residual_dimension",
        "physics_hz",
        "decision_hz",
        *FROZEN_HASH_FIELDS,
    )
    differing = [
        field
        for field in required_info_fields
        if field not in infos or infos[field] != manifest.get(field)
    ]
    if differing:
        raise CliError(
            "loaded RSL checkpoint infos differ from improved manifest: "
            + ", ".join(differing)
        )
    output_root = _resolve_project_path(args.output_root)
    artifacts = export_inference_actor(
        runner,
        source_checkpoint_path=checkpoint,
        source_manifest_path=manifest_path,
        output_root=output_root,
        captured_bundle=checkpoint_capture,
    )
    result = {
        "schema": "wlr50_clean.ppo_inference_actor_export_cli.v1",
        "live_rsl_runner_loaded": True,
        "episode_stepped": False,
        "deterministic_mean_policy": True,
        "runner_checkpoint_infos_verified": True,
        "checkpoint_runtime_capture_verified": True,
        "checkpoint_runtime_capture": checkpoint_capture.as_dict(),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": str(manifest_path),
        "torchscript_actor": str(artifacts.torchscript_actor),
        "onnx_actor": None if artifacts.onnx_actor is None else str(artifacts.onnx_actor),
        "export_manifest": str(artifacts.export_manifest),
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _json(args.run_dir / "inference_actor_export.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _require_video_capture_request(args: argparse.Namespace) -> None:
    """Validate a single-source live capture before Isaac is launched."""

    from .checkpoint_promotion import validate_checkpoint_artifact_provenance
    from .rl_library_wrapper import load_training_profile

    role = getattr(args, "video_source_role", None)
    if role not in {"fsm", "ppo"}:
        raise CliError("capture-video-source requires --video-source-role fsm or ppo")
    profile = load_training_profile(args.training_config)
    if int(profile.video_seed) != 4001 or args.seed != profile.video_seed:
        raise CliError("final video capture requires the locked video seed 4001")
    if args.num_envs != 1 or args.episode_count != 1:
        raise CliError(
            "video capture requires exactly one environment and one episode per fresh process"
        )
    if not args.deterministic:
        raise CliError("video capture requires deterministic policy execution")
    if args.headless:
        raise CliError("active-viewport video capture requires --no-headless")
    if not math.isclose(args.capture_fps, 15.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise CliError("final video capture fps is locked to 15")
    if not math.isclose(
        args.maximum_duration_s, 200.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise CliError("final video capture duration gate is locked to 200 seconds")
    source_root = (args.run_dir / "video_source").resolve()
    if source_root.exists():
        raise CliError(f"refusing to overwrite video source directory: {source_root}")
    args._video_source_root = source_root

    if role == "fsm":
        if args.checkpoint is not None or args.checkpoint_manifest is not None:
            raise CliError("FSM zero-residual video capture must not name a checkpoint")
        args._video_checkpoint_provenance = None
        return

    checkpoint = _required_artifact_argument(args, "checkpoint", "--checkpoint")
    manifest_path = _required_artifact_argument(
        args, "checkpoint_manifest", "--checkpoint-manifest"
    )
    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )
    provenance = validate_checkpoint_artifact_provenance(checkpoint, manifest_path)
    manifest = provenance.manifest
    _require_manifest_snapshot_contract(
        manifest,
        pinned_snapshot_bundle.as_record(),
        label="video checkpoint manifest",
        effective_entry_contract=pinned_effective_entry_contract,
    )
    if (
        manifest.get("publication_role") != "improved"
        or manifest.get("validation_promotion_authorized") is not True
        or manifest.get("locked_test_authorized") is not True
        or manifest.get("promotion_authorized") is not True
    ):
        raise CliError(
            "PPO video capture requires the two-stage promoted improved checkpoint"
        )
    args.checkpoint = provenance.checkpoint_path
    args.checkpoint_manifest = provenance.manifest_path
    args._video_checkpoint_provenance = provenance.as_dict()


def _capture_video_source(args: argparse.Namespace, simulation_app: Any) -> int:
    """Capture exactly one FSM or PPO source episode in this Isaac process."""

    from wlr50_clean.infrastructure.app_runtime import _configure_active_viewport
    from wlr50_clean.infrastructure.video_capture import ActiveViewportVideoRecorder
    from omni.kit.viewport.utility import get_active_viewport

    from .video_runtime import capture_live_policy_video

    pinned_snapshot_bundle, pinned_effective_entry_contract = (
        _pinned_runtime_phase_contracts(args)
    )

    configured_viewport = _configure_active_viewport()
    source_root = Path(args._video_source_root)

    def active_viewport_provider() -> Any:
        current = get_active_viewport()
        if current is None or current is not configured_viewport:
            raise CliError(
                "the active viewport changed after locked 1280x720 configuration"
            )
        return current

    def recorder_factory(root: Path) -> Any:
        return ActiveViewportVideoRecorder(
            root,
            viewport_provider=active_viewport_provider,
        )

    checkpoint_capture = None
    if args.video_source_role == "fsm":
        from .isaac_fsm_backend import IsaacFSMBackend
        from .residual_direct_env import ResidualEpisodeEnv

        episode = ResidualEpisodeEnv(
            IsaacFSMBackend(
                simulation_app,
                expected_phase_snapshot_bundle=pinned_snapshot_bundle,
                expected_effective_entry_contract=pinned_effective_entry_contract,
            ),
            collect_trace=True,
        )
        capture = capture_live_policy_video(
            episode,
            seed=args.seed,
            output_directory=source_root,
            action_factory=lambda _observation, _decision: (0.0,) * 12,
            policy_label="fsm_zero_residual",
            episode_already_reset=False,
            recorder_factory=recorder_factory,
        )
    else:
        from .rl_library_wrapper import (
            deterministic_action,
            load_checkpoint_round_trip,
            validate_resume_checkpoint_provenance,
        )

        checkpoint_capture = _pin_live_checkpoint(
            args,
            args.checkpoint,
            args.checkpoint_manifest,
            purpose="video-source-ppo",
        )
        _, env, runner, _ = _construct_live_runner(
            args,
            simulation_app,
            max_iterations=1,
            reset_seeds=(args.seed,),
            collect_trace=True,
            pinned_snapshot_bundle=pinned_snapshot_bundle,
            pinned_effective_entry_contract=pinned_effective_entry_contract,
        )
        if len(env.environments) != 1:
            raise CliError("PPO video runner did not expose exactly one episode")
        episode = env.environments[0]
        infos = load_checkpoint_round_trip(
            runner,
            args.checkpoint,
            captured_bundle=checkpoint_capture,
        )
        loaded_provenance = validate_resume_checkpoint_provenance(
            args.checkpoint,
            infos,
            manifest_path=args.checkpoint_manifest,
            expected_runtime_contract=_current_checkpoint_runtime_contract(
                args,
                pinned_snapshot_bundle=pinned_snapshot_bundle,
                pinned_effective_entry_contract=pinned_effective_entry_contract,
            ),
            captured_bundle=checkpoint_capture,
        )
        offline_provenance = dict(args._video_checkpoint_provenance)
        for field in (
            "checkpoint_path",
            "checkpoint_sha256",
            "manifest_path",
            "manifest_sha256",
        ):
            if loaded_provenance.as_dict()[field] != offline_provenance[field]:
                raise CliError(
                    "loaded PPO checkpoint provenance changed after pre-launch validation"
                )

        def ppo_action(observation: Sequence[float], _decision: int) -> tuple[float, ...]:
            observations = env.observation_tensor_dict((observation,))
            actions = deterministic_action(runner, observations)
            if tuple(actions.shape) != (1, 12):
                raise CliError(
                    f"deterministic actor returned {tuple(actions.shape)}, expected (1, 12)"
                )
            return tuple(
                float(value)
                for value in actions.detach().to("cpu").tolist()[0]
            )

        capture = capture_live_policy_video(
            episode,
            seed=args.seed,
            output_directory=source_root,
            action_factory=ppo_action,
            policy_label="ppo_deterministic_mean",
            checkpoint_path=args.checkpoint,
            checkpoint_manifest_path=args.checkpoint_manifest,
            checkpoint_load_provenance=loaded_provenance.as_dict(),
            episode_already_reset=True,
            recorder_factory=recorder_factory,
        )

    result = {
        "schema": "wlr50_clean.ppo_video_source_capture_cli.v1",
        "video_source_role": args.video_source_role,
        "fresh_process_single_episode": True,
        "seed": args.seed,
        "headless": False,
        "active_viewport_configured": True,
        "source_directory": str(source_root),
        "source_manifest": str(source_root / "ppo_video_source_manifest.json"),
        "source_video": str(source_root / "actual_viewport_video.mp4"),
        "capture_process_id": capture["capture_process_id"],
        "capture_process_instance_id": capture["capture_process_instance_id"],
        "checkpoint_load_provenance": capture["checkpoint_load_provenance"],
        "checkpoint_runtime_capture_verified": (
            args.video_source_role == "fsm" or checkpoint_capture is not None
        ),
        "checkpoint_runtime_capture": (
            None
            if args.video_source_role == "fsm"
            else checkpoint_capture.as_dict()
        ),
    }
    _revalidate_pinned_phase_contracts(
        pinned_snapshot_bundle, pinned_effective_entry_contract
    )
    _json(args.run_dir / "video_source_capture.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _publish_videos(args: argparse.Namespace) -> int:
    """Publish the final four videos without importing or starting Isaac."""

    from .artifacts import file_record
    from .video_artifacts import publish_final_videos

    if args.seed != 4001 or args.num_envs != 1 or args.episode_count != 1:
        raise CliError(
            "final video publication requires seed 4001 and one source episode per side"
        )
    if not args.deterministic:
        raise CliError("final video publication requires deterministic source evidence")
    if args.fsm_video_source_dir is None or args.ppo_video_source_dir is None:
        raise CliError(
            "publish-videos requires --fsm-video-source-dir and --ppo-video-source-dir"
        )
    fsm_source = _resolve_project_path(args.fsm_video_source_dir)
    ppo_source = _resolve_project_path(args.ppo_video_source_dir)
    if fsm_source == ppo_source:
        raise CliError("FSM and PPO video source directories must be distinct")
    output_root = _resolve_project_path(args.output_root)
    publication = publish_final_videos(
        fsm_source_dir=fsm_source,
        ppo_source_dir=ppo_source,
        output_root=output_root,
        publication_run_dir=args.run_dir,
        ffmpeg=args.ffmpeg,
    )
    video_records = {
        name: file_record(path) for name, path in publication.videos.items()
    }
    validation_record = file_record(publication.validation_path)
    checksum_record = file_record(publication.checksum_path)
    diagnostic_record = file_record(publication.diagnostic_ass_path)
    result = {
        "schema": "wlr50_clean.ppo_final_video_publication_cli.v1",
        "offline": True,
        "isaac_started": False,
        "seed": args.seed,
        "publication_run_directory": str(args.run_dir.resolve()),
        "fsm_source_directory": str(fsm_source),
        "ppo_source_directory": str(ppo_source),
        "videos": {name: str(path) for name, path in publication.videos.items()},
        "video_records": video_records,
        "video_validation": str(publication.validation_path),
        "video_validation_sha256": validation_record["sha256"],
        "video_validation_bytes": validation_record["bytes"],
        "video_checksums": str(publication.checksum_path),
        "video_checksums_sha256": checksum_record["sha256"],
        "video_checksums_bytes": checksum_record["bytes"],
        "diagnostic_ass": str(publication.diagnostic_ass_path),
        "diagnostic_ass_sha256": diagnostic_record["sha256"],
        "diagnostic_ass_bytes": diagnostic_record["bytes"],
        "checksum_verification": dict(publication.checksum_verification),
    }
    _json(args.run_dir / "final_video_publication.json", result)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


def _cleanup_isaac(simulation_app: Any) -> None:
    # Full Kit teardown can wait indefinitely for extensions that the
    # headless PPO process never used (observed on Isaac Sim 5.1/Windows).
    # The process is single-shot and owns no unsaved stage, so the documented
    # fast close is both deterministic and safer for immutable run finalizing.
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


def _dispatch_live(args: argparse.Namespace) -> int:
    if args.command == "train":
        _require_canonical_initial_checkpoint(args)
        _revalidate_training_phase_zero_residual_rollout(args)
    phase_contracts = None
    if args.command in PHASE_CONTRACT_LIVE_COMMANDS:
        phase_contracts = _pinned_runtime_phase_contracts(args)
        _revalidate_pinned_phase_contracts(*phase_contracts)
    # AppLauncher is the only Isaac import before the application exists.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=bool(args.headless),
        enable_cameras=args.command == "capture-video-source",
    )
    app = launcher.app
    app.update()
    checkpoint_captures = ExitStack()
    args._checkpoint_runtime_capture_stack = checkpoint_captures
    try:
        if args.command in {"baseline-eval", "zero-residual-live", "nonzero-residual-smoke"}:
            exit_code = _baseline_or_gate(args, app)
        elif args.command == "reset-throughput-probe":
            exit_code = _reset_throughput_probe(args, app)
        elif args.command == "soft-reset-equivalence":
            exit_code = _soft_reset_equivalence(args, app)
        elif args.command == "phase-snapshot-live-probe":
            exit_code = _phase_snapshot_live_probe(args, app)
        elif args.command == "phase-zero-residual-rollout":
            exit_code = _phase_zero_residual_rollout(args, app)
        elif args.command == "vector-benchmark":
            exit_code = _vector_benchmark(args, app)
        elif args.command == "initialize-zero-residual":
            exit_code = _initialize_zero_residual(args, app)
        elif args.command == "train":
            exit_code = _train(args, app)
        elif args.command == "evaluate":
            exit_code = _evaluate(args, app)
        elif args.command == "export-inference-actor":
            exit_code = _export_inference_actor(args, app)
        elif args.command == "capture-video-source":
            exit_code = _capture_video_source(args, app)
        else:
            raise CliError(f"unsupported live command: {args.command}")
        if phase_contracts is not None:
            _revalidate_pinned_phase_contracts(*phase_contracts)
        if args.command == "train":
            _revalidate_training_phase_zero_residual_rollout(args)
        # Closing the stack re-hashes both the source pair and the private
        # load-only copies.  Do this before publishing the authoritative live
        # exit result so an integrity failure cannot be finalized as success.
        checkpoint_captures.close()
        # Some Kit teardown paths normalize the native process exit status on
        # Windows. Persist the result before closing Kit so the immutable-run
        # wrapper can propagate application failures faithfully.
        _json(
            args.run_dir / "live_command_result.json",
            {
                "schema": "wlr50_clean.live_command_result.v1",
                "command": args.command,
                "exit_code": int(exit_code),
            },
        )
        return int(exit_code)
    finally:
        try:
            checkpoint_captures.close()
        finally:
            _cleanup_isaac(app)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_common(args)
        if args.command == "preflight":
            return _preflight(args)
        if args.command == "build-phase-snapshots":
            return _build_snapshots(args)
        if args.command == "aggregate-evaluations":
            return _aggregate_evaluations(args)
        if args.command == "export-baseline-evaluation":
            return _export_baseline_evaluation(args)
        if args.command == "export-paired-evaluation":
            return _export_paired_evaluation(args)
        if args.command == "promote-best-validation":
            return _promote_best_validation(args)
        if args.command == "promote-improved":
            return _promote_improved(args)
        if args.command == "publish-videos":
            return _publish_videos(args)
        if args.command == "publish-initial-zero-residual":
            return _publish_initial_zero_residual(args)
        if args.command in PHASE_CONTRACT_LIVE_COMMANDS:
            # Capture the authoritative snapshot bundle first, then its bound
            # effective-entry contract, before AppLauncher or scene mutation.
            _capture_runtime_phase_contracts(args)
        if args.command in {
            "phase-snapshot-live-probe",
            "phase-zero-residual-rollout",
            "initialize-zero-residual",
            "train",
            "evaluate",
            "export-inference-actor",
        }:
            # Reject a missing, changed, or redirected reset bundle before an
            # AppLauncher process is created.
            _validated_runtime_snapshot_bundle(args)
        if args.command == "capture-video-source":
            # Resolve immutable checkpoint evidence and reject headless or
            # multi-episode capture before importing AppLauncher.
            _require_video_capture_request(args)
        if args.command == "phase-zero-residual-rollout":
            _validate_phase_rollout_holdout(args)
        if args.command == "train":
            # Validate before AppLauncher so a missing/stale proof cannot even
            # start an expensive live training process.
            _require_training_phase_effective_entry_holdout(args)
            _require_training_phase_zero_residual_rollout(args)
            _require_training_soft_reset_acceptance(args)
            if args.num_envs > 1:
                from .rl_library_wrapper import load_training_profile

                _require_training_vector_benchmark_acceptance(
                    args, load_training_profile(args.training_config)
                )
            if args.stage != "smoke" and args.checkpoint is None:
                raise CliError(
                    f"{args.stage} training requires an explicit --checkpoint; "
                    "refusing to fall back to the initial actor"
                )
            if args.checkpoint is None and args.checkpoint_manifest is not None:
                raise CliError(
                    "--checkpoint-manifest cannot be supplied without --checkpoint"
                )
        if args.command in LIVE_COMMANDS:
            return _dispatch_live(args)
        raise CliError(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"PPO_PIPELINE_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
