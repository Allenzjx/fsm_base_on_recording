from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from wlr50_clean.ppo import artifacts, cli
from wlr50_clean.ppo import vector_benchmark_matrix as matrix_subject
from wlr50_clean.ppo.artifacts import config_set_record, file_record
from wlr50_clean.ppo.vector_benchmark_matrix import (
    MATRIX_SCHEMA,
    VectorBenchmarkMatrixError,
    aggregate_vector_benchmark_matrix,
    validate_finalized_vector_benchmark_matrix,
)


GIT_COMMIT = "c" * 40
FROZEN_SOURCE_HEAD = "7d6bfda0da593e2cace2accd8bc81d300bdd9288"
RUN_SEED = 1001


def _create_windows_junction(link: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {result.stderr or result.stdout}")


@pytest.fixture(autouse=True)
def _committed_runtime_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        matrix_subject,
        "_committed_runtime_paths",
        lambda *_args: (
            "configs/ppo_interface_v2.yaml",
            "configs/ppo_training_phase_v1.yaml",
        ),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _runtime_identity_files(
    project_root: Path, directory: Path, *, git_commit: str
) -> tuple[Path, Path]:
    rows = []
    for index, name in enumerate(
        ("configs/ppo_interface_v2.yaml", "configs/ppo_training_phase_v1.yaml"),
        1,
    ):
        source = project_root / name
        data = source.read_bytes()
        rows.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "creation_time_utc_ticks": index,
                "last_write_time_utc_ticks": index + 10,
            }
        )
    encoded = json.dumps(
        rows, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    content_encoded = json.dumps(
        [
            {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in rows
        ],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload = {
        "schema": "wlr50_clean.committed_runtime_identity.v1",
        "git_commit": git_commit,
        "file_count": len(rows),
        "content_sha256": hashlib.sha256(content_encoded).hexdigest(),
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }
    before = directory / "committed_runtime_identity.before.json"
    after = directory / "committed_runtime_identity.after.json"
    _write_json(before, payload)
    _write_json(after, payload)
    return before, after


def _frozen_manifest(project_root: Path) -> tuple[Path, dict[str, str], str]:
    path = project_root / "artifacts" / "ppo_phase_v1_start" / "frozen_fsm_hashes.json"
    protected = {
        f"protected/frozen_{index:02d}.py": hashlib.sha256(
            f"frozen-{index}".encode("ascii")
        ).hexdigest()
        for index in range(29)
    }
    if not path.exists():
        _write_json(
            path,
            {
                "algorithm": "sha256",
                "source_head": FROZEN_SOURCE_HEAD,
                "protected_files": protected,
            },
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, protected, digest


def _matrix_output(project_root: Path, name: str = "matrix") -> Path:
    directory = (
        project_root
        / "runs"
        / "ppo_phase_v1"
        / "vector-benchmark-matrix"
        / name
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "vector_benchmark_matrix.json"


def _config_evidence(project_root: Path) -> tuple[str, list[dict[str, Any]]]:
    first = project_root / "configs" / "ppo_training_phase_v1.yaml"
    second = project_root / "configs" / "ppo_interface_v2.yaml"
    if not first.exists():
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("training: fixture\n", encoding="utf-8")
        second.write_text("interface: fixture\n", encoding="utf-8")
    return config_set_record((first, second), project_root=project_root)


def _smoke(mode: str, num_envs: int, seed_rows: tuple[int, ...]) -> dict[str, Any]:
    actual_mode = "zero" if mode == "zero" else "nonzero"
    rows = []
    for decision in range(128):
        for env_index, seed in enumerate(seed_rows):
            rows.append(
                {
                    "mode": actual_mode,
                    "decision_index": decision,
                    "env_index": env_index,
                    "seed": seed,
                    "physics_tick": (decision + 1) * 8,
                    "in_episode_root_write_count": 0,
                    "recording_runtime_access_count": 0,
                    "terminated": False,
                    "truncated": False,
                }
            )
    return {
        "schema": "wlr50_clean.vectorized_residual_smoke.v1",
        "status": (
            "VECTOR_ZERO_RESIDUAL_SMOKE_PASSED"
            if mode == "zero"
            else "VECTOR_NONZERO_RESIDUAL_SMOKE_PASSED"
        ),
        "mode": actual_mode,
        "passed": True,
        "num_envs": num_envs,
        "policy_decisions": 128,
        "row_evidence_count": len(rows),
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "physics_ticks_per_decision": 8,
        "measured_physics_ticks": 1024,
        "global_physics_steps": 1024,
        "batched_articulation_writes": 1024,
        "exact_pair_captures": 1024,
        "independent_origin_count": num_envs,
        "independent_controller_count": num_envs,
        "independent_reader_count": num_envs,
        "independent_projection_bridge_count": num_envs,
        "live_vectorized_isaac_backend_verified": True,
        "deterministic_distinct_action_rows": mode != "zero",
        "zero_applied_equals_nominal_row_count": len(rows) if mode == "zero" else 0,
        "nonzero_active_row_count": 0 if mode == "zero" else len(rows),
        "maximum_observed_phase_scale_fraction": 0.0 if mode == "zero" else 0.04,
        "all_masks_honored": True,
        "all_zero_fast_path_expected": True,
        "no_in_episode_root_writes": True,
        "no_recording_runtime_access": True,
        "no_termination_or_safety_events": True,
        "rows": rows,
    }


def _benchmark(
    mode: str,
    num_envs: int,
    seed_rows: tuple[int, ...],
    *,
    passed: bool,
    failure_reasons: tuple[str, ...] = ("GPU capacity exhausted",),
) -> dict[str, Any]:
    resource_evidence = {
        "cuda_memory": (
            {
                "schema": "wlr50_clean.vector_cuda_memory_evidence.v1",
                "device": "cuda:0",
                "peak_stats_reset_before_measured_section": True,
                "measurement_covers_throughput_and_residual_smoke": True,
                "allocated_bytes_at_measurement_start": 1_000_000,
                "reserved_bytes_at_measurement_start": 2_000_000,
                "peak_allocated_bytes": 3_000_000,
                "peak_reserved_bytes": 4_000_000,
                "device_total_bytes": 24_000_000_000,
                "peak_allocated_below_device_total": True,
                "peak_reserved_below_device_total": True,
                "oom_detected": False,
            }
            if passed
            else {
                "schema": "wlr50_clean.vector_cuda_memory_evidence.v1",
                "evidence_complete": False,
                    "oom_detected": "out of memory"
                    in " ".join(str(value) for value in failure_reasons).lower(),
            }
        ),
        "contamination": (
            {
                "schema": "wlr50_clean.vector_contamination_evidence.v1",
                "evidence_complete": True,
                "cross_environment_contamination_detected": False,
                "fsm_state_contamination_detected": False,
                "render_contamination_detected": False,
                "measured_render_calls": 0,
                "independent_seed_count": num_envs,
                "independent_controller_count": num_envs,
                "independent_reader_count": num_envs,
                "independent_origin_count": num_envs,
            }
            if passed
            else {
                "schema": "wlr50_clean.vector_contamination_evidence.v1",
                "evidence_complete": False,
                "cross_environment_contamination_detected": None,
                "fsm_state_contamination_detected": None,
                "render_contamination_detected": None,
            }
        ),
    }
    return {
        "schema": "wlr50_clean.vectorized_isaac_benchmark_run.v1",
        "seed_rows": list(seed_rows),
        "report": {
            "status": (
                "TRUE_BATCHED_ISAAC_VERIFIED"
                if passed
                else "VECTOR_BACKEND_BENCHMARK_FAILED"
            ),
            "num_envs": num_envs,
            "measured_ticks": 1200 if passed else 0,
            "wall_time_s": 1.0 if passed else 0.0,
            "physics_steps_per_second": 1200.0 if passed else 0.0,
            "environment_steps_per_second": 1200.0 * num_envs if passed else 0.0,
            "one_simulation_context": True,
            "articulation_tensor_instances": num_envs,
            "global_physics_steps": 1200 if passed else 0,
            "batched_articulation_writes": 1200 if passed else 0,
            "exact_pair_captures": 1200 if passed else 0,
            "exact_pair_sensor_count": 13,
            "independent_controller_count": num_envs,
            "independent_reader_count": num_envs,
            "final_state_ids": ["P01"] * num_envs,
            "true_batched_isaac_verified": passed,
            "failure_reasons": list(failure_reasons) if not passed else [],
        },
        "residual_smoke": _smoke(mode, num_envs, seed_rows) if passed else None,
        "resource_evidence": resource_evidence,
        "passed": passed,
    }


def _write_slot(
    project_root: Path,
    *,
    num_envs: int,
    mode: str,
    passed: bool,
    config_sha256: str | None = None,
    git_commit: str = GIT_COMMIT,
    seed_rows: tuple[int, ...] | None = None,
    failure_reasons: tuple[str, ...] = ("GPU capacity exhausted",),
) -> Path:
    project_root.mkdir(parents=True, exist_ok=True)
    frozen_path, protected, frozen_sha256 = _frozen_manifest(project_root)
    current_config_sha256, configs = _config_evidence(project_root)
    if config_sha256 is None:
        config_sha256 = current_config_sha256
    worker_root = project_root / "runs" / "ppo_phase_v1" / "vector-benchmark"
    worker_root.mkdir(parents=True, exist_ok=True)
    directory = worker_root / f"slot_n{num_envs}_{mode}_{len(tuple(worker_root.iterdir()))}"
    directory.mkdir()
    seeds = seed_rows or tuple(range(RUN_SEED, RUN_SEED + num_envs))
    payload = _benchmark(
        mode,
        num_envs,
        seeds,
        passed=passed,
        failure_reasons=failure_reasons,
    )
    benchmark_path = directory / "vector_benchmark.json"
    _write_json(benchmark_path, payload)
    audit_common = {
        "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
        "project_root": str(project_root.resolve()),
        "frozen_manifest": str(frozen_path.resolve()),
        "frozen_manifest_sha256": frozen_sha256,
        "source_head": FROZEN_SOURCE_HEAD,
        "protected_file_count": 29,
        "entries": [
            {
                "path": name,
                "expected_sha256": digest,
                "actual_sha256": digest,
                "exists": True,
                "valid": True,
            }
            for name, digest in protected.items()
        ],
        "mismatches": [],
        "passed": True,
    }
    before = directory / "frozen_hashes.before.json"
    after = directory / "frozen_hashes.after.json"
    _write_json(before, {**audit_common, "checked_at_utc": "2026-09-04T01:00:00Z"})
    _write_json(after, {**audit_common, "checked_at_utc": "2026-09-04T01:01:00Z"})
    identity = {
        "config_sha256": config_sha256,
        "environment_count": num_envs,
        "git_commit": git_commit,
        "seed": RUN_SEED,
        "timestamp_utc": "2026-09-04T01:00:00Z",
        "training_stage": "backend-benchmark",
    }
    started_payload = {
        "schema": "wlr50_clean.ppo_run_manifest.v1",
        "lifecycle": "STARTED",
        "run_id": directory.name,
        "run_dir": str(directory.resolve()),
        "run_kind": "vector-benchmark",
        "project_root": str(project_root.resolve()),
        "immutable_run_directory": True,
        "identity": identity,
        "configs": configs,
        "entrypoint": "wlr50_clean.ppo.cli",
        "subcommand": "vector-benchmark",
        "invocation_arguments": [
            "--measured-ticks",
            "1200",
            "--seed-set",
            "train",
            "--residual-mode",
            mode,
            "--policy-decisions",
            "128",
            "--run-dir",
            "<reserved-immutable-run-dir>",
            "--seed",
            str(RUN_SEED),
            "--num-envs",
            str(num_envs),
        ],
    }
    started = directory / "run_manifest.started.json"
    _write_json(started, started_payload)
    stdout = directory / "stdout.log"
    stderr = directory / "stderr.log"
    stdout.write_text(
        json.dumps({"audit": str(before.resolve()), "passed": True}, separators=(",", ":"))
        + "\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n"
        + json.dumps({"audit": str(after.resolve()), "passed": True}, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")
    runtime_before, runtime_after = _runtime_identity_files(
        project_root, directory, git_commit=git_commit
    )
    final_payload = {
        **started_payload,
        "lifecycle": "SUCCEEDED" if passed else "FAILED",
        "completed_at_utc": "2026-09-04T01:01:00Z",
        "exit_code": 0 if passed else 2,
        "started_manifest": file_record(started, relative_to=directory),
        "logs": {
            "stdout.log": file_record(stdout, relative_to=directory),
            "stderr.log": file_record(stderr, relative_to=directory),
        },
        "artifacts": {
            path.name: file_record(path, relative_to=directory)
            for path in (
                benchmark_path,
                before,
                after,
                runtime_before,
                runtime_after,
            )
        },
    }
    _write_json(directory / "run_manifest.json", final_payload)
    return benchmark_path


def _matrix(
    tmp_path: Path,
    *,
    passed_counts: tuple[int, ...] = (8, 16),
) -> list[Path]:
    paths = []
    for num_envs in (8, 16, 32):
        for mode in ("zero", "bounded-smoke"):
            paths.append(
                _write_slot(
                    tmp_path,
                    num_envs=num_envs,
                    mode=mode,
                    passed=num_envs in passed_counts,
                )
            )
    return paths


def _change_runtime_after_timestamp(run_dir: Path) -> None:
    after = run_dir / "committed_runtime_identity.after.json"
    payload = json.loads(after.read_text(encoding="utf-8"))
    payload["files"][0]["last_write_time_utc_ticks"] += 1
    normalized = [
        {
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "creation_time_utc_ticks": row["creation_time_utc_ticks"],
            "last_write_time_utc_ticks": row["last_write_time_utc_ticks"],
        }
        for row in payload["files"]
    ]
    encoded = json.dumps(
        normalized,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["aggregate_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json(after, payload)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][after.name] = file_record(after, relative_to=run_dir)
    _write_json(manifest_path, manifest)


def _finalized_matrix_run(
    project_root: Path, paths: list[Path], *, selected_num_envs: int = 16
) -> Path:
    output = _matrix_output(project_root, "finalized")
    payload = aggregate_vector_benchmark_matrix(paths, output_path=output)
    assert payload["selected_num_envs"] == selected_num_envs
    run_dir = output.parent
    worker_manifest = json.loads(
        (paths[0].parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (paths[0].parent / "frozen_hashes.before.json").read_text(encoding="utf-8")
    )
    before = run_dir / "frozen_hashes.before.json"
    after = run_dir / "frozen_hashes.after.json"
    _write_json(before, {**audit, "checked_at_utc": "2026-09-04T02:00:00Z"})
    _write_json(after, {**audit, "checked_at_utc": "2026-09-04T02:01:00Z"})
    identity = {
        "config_sha256": worker_manifest["identity"]["config_sha256"],
        "environment_count": 1,
        "git_commit": GIT_COMMIT,
        "seed": RUN_SEED,
        "timestamp_utc": "2026-09-04T02:00:00Z",
        "training_stage": "backend-benchmark-selection",
    }
    invocation: list[str] = []
    for worker in paths:
        invocation.extend(("--benchmark", str(worker.resolve())))
    invocation.extend(
        (
            "--run-dir",
            "<reserved-immutable-run-dir>",
            "--seed",
            str(RUN_SEED),
            "--num-envs",
            "1",
        )
    )
    started_payload = {
        "schema": "wlr50_clean.ppo_run_manifest.v1",
        "lifecycle": "STARTED",
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_kind": "vector-benchmark-matrix",
        "project_root": str(project_root.resolve()),
        "immutable_run_directory": True,
        "identity": identity,
        "configs": worker_manifest["configs"],
        "entrypoint": "wlr50_clean.ppo.vector_benchmark_matrix",
        "subcommand": "aggregate",
        "invocation_arguments": invocation,
    }
    started = run_dir / "run_manifest.started.json"
    _write_json(started, started_payload)
    stdout = run_dir / "stdout.log"
    stderr = run_dir / "stderr.log"
    stdout.write_text(
        json.dumps({"audit": str(before.resolve()), "passed": True}, separators=(",", ":"))
        + "\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n"
        + json.dumps({"audit": str(after.resolve()), "passed": True}, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")
    runtime_before, runtime_after = _runtime_identity_files(
        project_root, run_dir, git_commit=GIT_COMMIT
    )
    final_payload = {
        **started_payload,
        "lifecycle": "SUCCEEDED",
        "completed_at_utc": "2026-09-04T02:01:00Z",
        "exit_code": 0,
        "started_manifest": file_record(started, relative_to=run_dir),
        "logs": {
            "stdout.log": file_record(stdout, relative_to=run_dir),
            "stderr.log": file_record(stderr, relative_to=run_dir),
        },
        "artifacts": {
            item.name: file_record(item, relative_to=run_dir)
            for item in (output, before, after, runtime_before, runtime_after)
        },
    }
    _write_json(run_dir / "run_manifest.json", final_payload)
    return output


def test_matrix_selects_highest_dual_mode_pass_and_records_larger_failures(
    tmp_path: Path,
) -> None:
    paths = list(reversed(_matrix(tmp_path)))
    output = _matrix_output(tmp_path)

    payload = aggregate_vector_benchmark_matrix(paths, output_path=output)

    assert payload["schema"] == MATRIX_SCHEMA
    assert payload["passed"] is True
    assert payload["validated_slot_count"] == 6
    assert payload["selected_num_envs"] == 16
    assert payload["common_provenance"]["frozen_source_head"] == FROZEN_SOURCE_HEAD
    assert payload["dual_mode_pass_by_environment_count"] == {
        "8": True,
        "16": True,
        "32": False,
    }
    assert payload["selected_capacity_checks"]["oom_detected"] is False
    assert payload["selected_capacity_checks"][
        "cross_environment_contamination_detected"
    ] is False
    assert payload["selected_capacity_checks"][
        "fsm_state_contamination_detected"
    ] is False
    assert payload["selected_capacity_checks"]["render_contamination_detected"] is False
    assert set(payload["selected_capacity_checks"]["resource_evidence_by_mode"]) == {
        "zero",
        "bounded-smoke",
    }
    assert len(payload["source_artifacts"]) == 6
    assert all(len(row["sha256"]) == 64 for row in payload["source_artifacts"])
    assert all(
        slot["failure_reasons"]
        for slot in payload["slots"]
        if slot["num_envs"] == 32
    )
    assert json.loads(output.read_text(encoding="utf-8"))["selected_num_envs"] == 16


def test_worker_runtime_identity_ignores_current_metadata_but_rejects_pair_change(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    config = tmp_path / "configs" / "ppo_training_phase_v1.yaml"
    current = config.stat()
    os.utime(config, ns=(current.st_atime_ns, current.st_mtime_ns + 2_000_000_000))
    aggregate_vector_benchmark_matrix(
        paths, output_path=_matrix_output(tmp_path, "metadata-only")
    )

    paths = _matrix(tmp_path / "pair-change")
    _change_runtime_after_timestamp(paths[0].parent)
    with pytest.raises(VectorBenchmarkMatrixError, match="changed during"):
        aggregate_vector_benchmark_matrix(
            paths,
            output_path=_matrix_output(tmp_path / "pair-change", "rejected"),
        )


def test_matrix_rejects_short_formal_benchmark_horizon(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    run_dir = paths[0].parent
    started_path = run_dir / "run_manifest.started.json"
    started = json.loads(started_path.read_text(encoding="utf-8"))
    position = started["invocation_arguments"].index("--measured-ticks") + 1
    started["invocation_arguments"][position] = "1199"
    _write_json(started_path, started)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["invocation_arguments"] = list(started["invocation_arguments"])
    manifest["started_manifest"] = file_record(started_path, relative_to=run_dir)
    _write_json(manifest_path, manifest)
    with pytest.raises(VectorBenchmarkMatrixError, match="too weak"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=_matrix_output(tmp_path, "short-horizon")
        )


def test_passed_slot_rejects_invalid_cuda_peak_capacity_evidence() -> None:
    seed_rows = tuple(range(RUN_SEED, RUN_SEED + 8))
    payload = _benchmark("zero", 8, seed_rows, passed=True)
    memory = payload["resource_evidence"]["cuda_memory"]
    memory["peak_reserved_bytes"] = memory["device_total_bytes"]
    with pytest.raises(VectorBenchmarkMatrixError, match="CUDA peak memory"):
        matrix_subject._validate_passed_evidence(
            payload,
            mode="zero",
            num_envs=8,
            seed_rows=seed_rows,
            measured_ticks=1200,
            policy_decisions=128,
        )


def test_vector_benchmark_wrapper_defaults_to_formal_horizon() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "benchmark_vectorized_ppo.ps1").read_text(
        encoding="utf-8"
    )
    assert "[ValidateRange(1200, 100000)][int]$MeasuredTicks = 1200" in text
    assert "[ValidateRange(128, 4096)][int]$PolicyDecisions = 128" in text


def test_matrix_rejects_missing_duplicate_and_nonmonotonic_slots(tmp_path: Path) -> None:
    paths = _matrix(tmp_path / "complete")
    with pytest.raises(VectorBenchmarkMatrixError, match="exactly six"):
        aggregate_vector_benchmark_matrix(paths[:-1], output_path=_matrix_output(tmp_path / "complete", "missing"))
    with pytest.raises(VectorBenchmarkMatrixError, match="duplicate input paths"):
        aggregate_vector_benchmark_matrix(
            paths[:-1] + [paths[0]], output_path=_matrix_output(tmp_path / "complete", "duplicate")
        )

    nonmonotonic = _matrix(tmp_path / "nonmonotonic", passed_counts=(8, 32))
    with pytest.raises(VectorBenchmarkMatrixError, match="monotonic"):
        aggregate_vector_benchmark_matrix(
            nonmonotonic, output_path=_matrix_output(tmp_path / "nonmonotonic")
        )


@pytest.mark.parametrize("mismatch", ["config", "commit", "seed"])
def test_matrix_rejects_cross_slot_provenance_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    paths = _matrix(tmp_path)
    replacement = _write_slot(
        tmp_path,
        num_envs=32,
        mode="bounded-smoke",
        passed=False,
        config_sha256="f" * 64 if mismatch == "config" else None,
        git_commit="1" * 40 if mismatch == "commit" else GIT_COMMIT,
        seed_rows=(
            tuple(range(1002, 1034))
            if mismatch == "seed"
            else tuple(range(1001, 1033))
        ),
    )
    paths[-1] = replacement

    with pytest.raises(VectorBenchmarkMatrixError, match="mismatch|differs|seed_rows"):
        aggregate_vector_benchmark_matrix(paths, output_path=_matrix_output(tmp_path, "bad"))


def test_failed_slot_requires_recorded_reason_and_finalized_digest_binding(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    paths[-1] = _write_slot(
        tmp_path,
        num_envs=32,
        mode="bounded-smoke",
        passed=False,
        failure_reasons=(),
    )
    with pytest.raises(VectorBenchmarkMatrixError, match="failure reasons"):
        aggregate_vector_benchmark_matrix(paths, output_path=_matrix_output(tmp_path, "reasonless"))

    paths = _matrix(tmp_path / "tampered")
    paths[0].write_text(paths[0].read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(VectorBenchmarkMatrixError, match="digest mismatch|tamper"):
        aggregate_vector_benchmark_matrix(paths, output_path=_matrix_output(tmp_path / "tampered"))


def test_failed_slot_rejects_non_string_reason(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    paths[-1] = _write_slot(
        tmp_path,
        num_envs=32,
        mode="bounded-smoke",
        passed=False,
        failure_reasons=(123,),  # type: ignore[arg-type]
    )
    with pytest.raises(VectorBenchmarkMatrixError, match="native strings"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=_matrix_output(tmp_path, "malformed-reason")
        )


def test_matrix_detects_input_mutation_before_publication(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    output = _matrix_output(tmp_path, "toctou")

    def mutate() -> None:
        paths[0].write_bytes(paths[0].read_bytes() + b" ")

    with pytest.raises(VectorBenchmarkMatrixError, match="changed before publication"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=output, _before_publish_hook=mutate
        )
    assert not output.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_matrix_rejects_output_run_replaced_by_junction_before_write(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    output = _matrix_output(tmp_path, "redirected-output")
    run_dir = output.parent
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = outside / "moved-matrix-run"

    def redirect() -> None:
        run_dir.rename(moved)
        _create_windows_junction(run_dir, moved)

    try:
        with pytest.raises(VectorBenchmarkMatrixError, match="symlink|junction"):
            aggregate_vector_benchmark_matrix(
                paths,
                output_path=output,
                _before_publish_hook=redirect,
            )
        assert not (moved / output.name).exists()
    finally:
        if run_dir.exists():
            os.rmdir(run_dir)


def test_matrix_rejects_worker_outside_managed_runs_root(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    escaped_dir = tmp_path / "escaped-worker"
    shutil.copytree(paths[0].parent, escaped_dir)
    paths[0] = escaped_dir / "vector_benchmark.json"

    with pytest.raises(VectorBenchmarkMatrixError, match="must be inside"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=_matrix_output(tmp_path, "escaped")
        )


def test_matrix_rejects_legacy_underscore_worker_manifest_kind(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    run_dir = paths[0].parent
    started_path = run_dir / "run_manifest.started.json"
    final_path = run_dir / "run_manifest.json"
    started = json.loads(started_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    started["run_kind"] = "vector_benchmark"
    final["run_kind"] = "vector_benchmark"
    _write_json(started_path, started)
    final["started_manifest"] = file_record(started_path, relative_to=run_dir)
    _write_json(final_path, final)

    with pytest.raises(VectorBenchmarkMatrixError, match="valid immutable"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=_matrix_output(tmp_path, "legacy-worker-kind")
        )


def test_matrix_rejects_symlinked_worker_path(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    link = (
        tmp_path
        / "runs"
        / "ppo_phase_v1"
        / "vector-benchmark"
        / "linked-worker"
    )
    try:
        os.symlink(paths[0].parent, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    paths[0] = link / "vector_benchmark.json"

    with pytest.raises(VectorBenchmarkMatrixError, match="symlink|junction"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=_matrix_output(tmp_path, "linked")
        )


@pytest.mark.parametrize("tamper", ["ppo-commit-source-head", "duplicate-entry"])
def test_frozen_audit_binds_manifest_source_and_exact_29_entry_set(
    tmp_path: Path, tamper: str
) -> None:
    paths = _matrix(tmp_path)
    audit_path = paths[0].parent / "frozen_hashes.before.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if tamper == "ppo-commit-source-head":
        audit["source_head"] = GIT_COMMIT
    else:
        audit["entries"][-1] = dict(audit["entries"][0])
    _write_json(audit_path, audit)
    manifest_path = paths[0].parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][audit_path.name] = file_record(
        audit_path, relative_to=paths[0].parent
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(VectorBenchmarkMatrixError, match="frozen audit|stale|entry"):
        aggregate_vector_benchmark_matrix(
            paths, output_path=_matrix_output(tmp_path, f"frozen-{tamper}")
        )


def test_matrix_publication_is_no_overwrite(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    output = _matrix_output(tmp_path)
    aggregate_vector_benchmark_matrix(paths, output_path=output)

    with pytest.raises(VectorBenchmarkMatrixError, match="overwrite"):
        aggregate_vector_benchmark_matrix(paths, output_path=output)


def test_finalized_matrix_revalidates_all_six_workers_for_training(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    matrix = _finalized_matrix_run(tmp_path, paths)
    frozen_path, _, frozen_sha256 = _frozen_manifest(tmp_path)
    del frozen_path
    worker_manifest = json.loads(
        (paths[0].parent / "run_manifest.json").read_text(encoding="utf-8")
    )

    evidence = validate_finalized_vector_benchmark_matrix(
        matrix,
        expected_project_root=tmp_path,
        expected_config_sha256=worker_manifest["identity"]["config_sha256"],
        expected_frozen_manifest_sha256=frozen_sha256,
        expected_git_commit=GIT_COMMIT,
        expected_run_seed=RUN_SEED,
        expected_num_envs=16,
        expected_seed_rows=tuple(range(RUN_SEED, RUN_SEED + 16)),
        expected_config_records=worker_manifest["configs"],
    )

    assert evidence["passed"] is True
    assert evidence["selected_num_envs"] == 16
    assert evidence["path"] == str(matrix.resolve())
    assert len(evidence["sha256"]) == 64
    assert set(evidence["selected_acceptance"]) == {"zero", "bounded_nonzero"}


def test_finalized_matrix_rejects_changed_matrix_runtime_identity(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    matrix = _finalized_matrix_run(tmp_path, paths)
    _change_runtime_after_timestamp(matrix.parent)
    frozen_path, _, frozen_sha256 = _frozen_manifest(tmp_path)
    del frozen_path
    worker_manifest = json.loads(
        (paths[0].parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(VectorBenchmarkMatrixError, match="changed during"):
        validate_finalized_vector_benchmark_matrix(
            matrix,
            expected_project_root=tmp_path,
            expected_config_sha256=worker_manifest["identity"]["config_sha256"],
            expected_frozen_manifest_sha256=frozen_sha256,
            expected_git_commit=GIT_COMMIT,
            expected_run_seed=RUN_SEED,
            expected_num_envs=16,
            expected_seed_rows=tuple(range(RUN_SEED, RUN_SEED + 16)),
            expected_config_records=worker_manifest["configs"],
        )


def test_finalized_matrix_rejects_legacy_underscore_manifest_kind(
    tmp_path: Path,
) -> None:
    paths = _matrix(tmp_path)
    matrix = _finalized_matrix_run(tmp_path, paths)
    run_dir = matrix.parent
    started_path = run_dir / "run_manifest.started.json"
    final_path = run_dir / "run_manifest.json"
    started = json.loads(started_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    started["run_kind"] = "vector_benchmark_matrix"
    final["run_kind"] = "vector_benchmark_matrix"
    _write_json(started_path, started)
    final["started_manifest"] = file_record(started_path, relative_to=run_dir)
    _write_json(final_path, final)
    _, _, frozen_sha256 = _frozen_manifest(tmp_path)
    worker_manifest = json.loads(
        (paths[0].parent / "run_manifest.json").read_text(encoding="utf-8")
    )

    with pytest.raises(VectorBenchmarkMatrixError, match="matching immutable"):
        validate_finalized_vector_benchmark_matrix(
            matrix,
            expected_project_root=tmp_path,
            expected_config_sha256=worker_manifest["identity"]["config_sha256"],
            expected_frozen_manifest_sha256=frozen_sha256,
            expected_git_commit=GIT_COMMIT,
            expected_run_seed=RUN_SEED,
            expected_num_envs=16,
            expected_seed_rows=tuple(range(RUN_SEED, RUN_SEED + 16)),
            expected_config_records=worker_manifest["configs"],
        )


def test_multi_env_training_preflight_consumes_only_finalized_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _matrix(tmp_path)
    matrix = _finalized_matrix_run(tmp_path, paths)
    worker_manifest = json.loads(
        (paths[0].parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    frozen_path, _, frozen_sha256 = _frozen_manifest(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_checkpoint_config_paths", lambda _args: (tmp_path / "config",))
    monkeypatch.setattr(
        artifacts,
        "config_set_record",
        lambda *_args, **_kwargs: (
            worker_manifest["identity"]["config_sha256"],
            worker_manifest["configs"],
        ),
    )
    monkeypatch.setattr(artifacts, "git_head", lambda _root: GIT_COMMIT)
    monkeypatch.setattr(cli, "_sha256", lambda path: frozen_sha256 if Path(path) == frozen_path else "0" * 64)
    args = type(
        "Args",
        (),
        {
            "num_envs": 16,
            "seed": RUN_SEED,
            "vector_benchmark_matrix": matrix,
            "vector_zero_benchmark_acceptance": None,
            "vector_nonzero_benchmark_acceptance": None,
        },
    )()
    profile = type("Profile", (), {"seed_train": tuple(range(1001, 1033))})()

    evidence = cli._require_training_vector_benchmark_acceptance(args, profile)

    assert evidence is not None
    assert args._vector_benchmark_matrix_evidence["sha256"] == evidence["sha256"]
    assert args.vector_zero_benchmark_acceptance.name == "vector_benchmark.json"
    assert args.vector_nonzero_benchmark_acceptance.name == "vector_benchmark.json"


def test_powershell_wrapper_routes_to_offline_module() -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts" / "aggregate_vector_benchmark_matrix.ps1").read_text(
        encoding="utf-8"
    )
    common = (root / "scripts" / "_invoke_ppo_cli.ps1").read_text(encoding="utf-8")

    assert "[ValidateCount(6, 6)]" in wrapper
    assert "_invoke_ppo_cli.ps1" in wrapper
    assert '-CliModule "wlr50_clean.ppo.vector_benchmark_matrix"' in wrapper
    assert '-RunKind "vector_benchmark_matrix"' in wrapper
    assert '-EnvironmentCount 1' in wrapper
    assert '[string]$CliModule = "wlr50_clean.ppo.cli"' in common
