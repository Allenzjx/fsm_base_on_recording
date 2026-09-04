from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from wlr50_clean.ppo import cli


CONFIG_SHA256 = "a" * 64
FROZEN_MANIFEST_SHA256 = "b" * 64
GIT_COMMIT = "c" * 40
RUN_SEED = 1001
NUM_ENVS = 8
EXPECTED_SEED_ROWS = tuple(range(1001, 1001 + NUM_ENVS))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _smoke_rows(mode: str, seed_rows: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for decision in range(2):
        for env_index, seed in enumerate(seed_rows):
            amplitude = 0.0 if mode == "zero" else (env_index + 1) * 5.0e-5
            raw = [amplitude] * 12
            rows.append(
                {
                    "mode": mode,
                    "decision_index": decision,
                    "env_index": env_index,
                    "seed": seed,
                    "phase": "P01",
                    "physics_tick": 8 * (decision + 1),
                    "sim_time_s": (decision + 1) / 15.0,
                    "raw_policy_action_full12": raw,
                    "nominal_action_full12": [0.0] * 12,
                    "projected_residual_full12": raw,
                    "applied_action_full12": raw,
                    "effective_action_mask_full12": [1] * 12,
                    "physical_phase_scale_full12": [0.01] * 12,
                    "active_nonzero_channel_count": 0 if mode == "zero" else 12,
                    "max_abs_phase_scale_fraction": (
                        0.0 if mode == "zero" else amplitude / 0.01
                    ),
                    "zero_residual_fast_path": mode == "zero",
                    "projection_clipping_stages": [],
                    "in_episode_root_write_count": 0,
                    "recording_runtime_access_count": 0,
                    "terminated": False,
                    "truncated": False,
                }
            )
    return rows


def _benchmark_payload(mode: str, seed_rows: tuple[int, ...]) -> dict[str, Any]:
    rows = _smoke_rows(mode, seed_rows)
    expected_rows = len(rows)
    return {
        "schema": "wlr50_clean.vectorized_isaac_benchmark_run.v1",
        "seed_rows": list(seed_rows),
        "report": {
            "status": "TRUE_BATCHED_ISAAC_VERIFIED",
            "num_envs": NUM_ENVS,
            "measured_ticks": 32,
            "wall_time_s": 1.0,
            "physics_steps_per_second": 32.0,
            "environment_steps_per_second": 256.0,
            "one_simulation_context": True,
            "articulation_tensor_instances": NUM_ENVS,
            "global_physics_steps": 32,
            "batched_articulation_writes": 32,
            "exact_pair_captures": 32,
            "exact_pair_sensor_count": 13,
            "independent_controller_count": NUM_ENVS,
            "independent_reader_count": NUM_ENVS,
            "final_state_ids": ["P01"] * NUM_ENVS,
            "true_batched_isaac_verified": True,
            "failure_reasons": [],
        },
        "residual_smoke": {
            "schema": "wlr50_clean.vectorized_residual_smoke.v1",
            "status": (
                "VECTOR_ZERO_RESIDUAL_SMOKE_PASSED"
                if mode == "zero"
                else "VECTOR_NONZERO_RESIDUAL_SMOKE_PASSED"
            ),
            "mode": mode,
            "passed": True,
            "num_envs": NUM_ENVS,
            "policy_decisions": 2,
            "row_evidence_count": expected_rows,
            "physics_hz": 120.0,
            "decision_hz": 15.0,
            "physics_ticks_per_decision": 8,
            "measured_physics_ticks": 16,
            "global_physics_steps": 16,
            "batched_articulation_writes": 16,
            "exact_pair_captures": 16,
            "independent_origin_count": NUM_ENVS,
            "independent_controller_count": NUM_ENVS,
            "independent_reader_count": NUM_ENVS,
            "independent_projection_bridge_count": NUM_ENVS,
            "live_vectorized_isaac_backend_verified": True,
            "deterministic_distinct_action_rows": True,
            "zero_applied_equals_nominal_row_count": (
                expected_rows if mode == "zero" else 0
            ),
            "nonzero_active_row_count": 0 if mode == "zero" else expected_rows,
            "maximum_observed_phase_scale_fraction": (
                0.0 if mode == "zero" else 0.04
            ),
            "all_masks_honored": True,
            "all_zero_fast_path_expected": True,
            "no_in_episode_root_writes": True,
            "no_recording_runtime_access": True,
            "no_termination_or_safety_events": True,
            "environment_origins_w_m": [
                [x, y, 0.0]
                for x in (12.0, 4.0, -4.0, -12.0)
                for y in (-4.0, 4.0)
            ],
            "rows": rows,
        },
        "passed": True,
    }


def _write_finalized_benchmark(
    root: Path,
    *,
    mode: str = "zero",
    seed_rows: tuple[int, ...] = EXPECTED_SEED_ROWS,
) -> Path:
    root.mkdir(parents=True)
    benchmark = root / "vector_benchmark.json"
    payload = _benchmark_payload(mode, seed_rows)
    _write_json(benchmark, payload)

    frozen_common = {
        "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
        "project_root": str(root.parent.resolve()),
        "frozen_manifest": str((root.parent / "frozen_fsm_hashes.json").resolve()),
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_head": GIT_COMMIT,
        "protected_file_count": 29,
        "entries": [],
        "mismatches": [],
        "passed": True,
    }
    before = root / "frozen_hashes.before.json"
    after = root / "frozen_hashes.after.json"
    _write_json(before, {**frozen_common, "checked_at_utc": "2026-09-04T01:00:00Z"})
    _write_json(after, {**frozen_common, "checked_at_utc": "2026-09-04T01:01:00Z"})

    config_paths = (
        "configs/environment_lock.json",
        "configs/frozen_successful_fsm.yaml",
        "configs/fsm_states.yaml",
        "configs/ppo_domain_randomization_v2.yaml",
        "configs/ppo_interface_v2.yaml",
        "configs/ppo_observation_schema_v2.json",
        "configs/ppo_phase_action_masks_v2.yaml",
        "configs/ppo_phase_objectives_v2.yaml",
        "configs/ppo_reward_v2.yaml",
        "configs/ppo_termination_v2.yaml",
        "configs/ppo_training_phase_v1.yaml",
        "configs/recording_motion_contract.json",
    )
    configs = [
        {"path": path, "bytes": index + 1, "sha256": f"{index + 1:064x}"}
        for index, path in enumerate(config_paths)
    ]
    run_id = "20260904T010000000000Z_gcccccccccccc_caaaaaaaaaaaa_s1001_n8_backend-benchmark"
    identity = {
        "config_sha256": CONFIG_SHA256,
        "environment_count": NUM_ENVS,
        "git_commit": GIT_COMMIT,
        "seed": RUN_SEED,
        "timestamp_utc": "2026-09-04T01:00:00Z",
        "training_stage": "backend-benchmark",
    }
    invocation = [
        "--training-config",
        "configs/ppo_training_phase_v1.yaml",
        "--interface-config",
        "configs/ppo_interface_v2.yaml",
        "--measured-ticks",
        "32",
        "--seed-set",
        "train",
        "--residual-mode",
        "zero" if mode == "zero" else "bounded-smoke",
        "--policy-decisions",
        "2",
        "--run-dir",
        "<reserved-immutable-run-dir>",
        "--seed",
        str(RUN_SEED),
        "--num-envs",
        str(NUM_ENVS),
    ]
    started_payload = {
        "schema": "wlr50_clean.ppo_run_manifest.v1",
        "lifecycle": "STARTED",
        "run_id": run_id,
        "run_dir": str(root.resolve()),
        "run_kind": "vector_benchmark",
        "project_root": str(root.parent.resolve()),
        "immutable_run_directory": True,
        "identity": identity,
        "configs": configs,
        "entrypoint": "wlr50_clean.ppo.cli",
        "subcommand": "vector-benchmark",
        "invocation_arguments": invocation,
    }
    started = root / "run_manifest.started.json"
    _write_json(started, started_payload)

    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    stdout.write_text(
        json.dumps({"audit": str(before), "passed": True}, separators=(",", ":"))
        + "\n[Info] Isaac vector benchmark\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n"
        + json.dumps({"audit": str(after), "passed": True}, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")
    final_payload = {
        **started_payload,
        "lifecycle": "SUCCEEDED",
        "exit_code": 0,
        "completed_at_utc": "2026-09-04T01:01:00Z",
        "logs": {
            "stdout.log": _record(stdout, relative_to=root),
            "stderr.log": _record(stderr, relative_to=root),
        },
        "started_manifest": _record(started, relative_to=root),
    }
    _write_json(root / "run_manifest.json", final_payload)
    return benchmark


def _validate(path: Path, *, expected_seed_rows: tuple[int, ...]) -> dict[str, Any]:
    return dict(
        cli._validate_vector_benchmark_acceptance(
            path,
            expected_mode="zero",
            expected_num_envs=NUM_ENVS,
            expected_run_seed=RUN_SEED,
            expected_seed_rows=expected_seed_rows,
            expected_config_sha256=CONFIG_SHA256,
            expected_frozen_manifest_sha256=FROZEN_MANIFEST_SHA256,
        )
    )


@pytest.mark.parametrize("tamper_target", ["artifact", "stdout"])
def test_vector_acceptance_is_bound_to_finalized_stdout_digest(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    benchmark = _write_finalized_benchmark(tmp_path / "gate")
    evidence = _validate(benchmark, expected_seed_rows=EXPECTED_SEED_ROWS)
    assert evidence["path"] == str(benchmark.resolve())

    if tamper_target == "artifact":
        payload = json.loads(benchmark.read_text(encoding="utf-8"))
        # Keep every pass bit true: only the measured evidence differs from the
        # copy cryptographically bound into finalized stdout.
        payload["report"]["wall_time_s"] = 2.0
        _write_json(benchmark, payload)
    else:
        with (benchmark.parent / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write("post-finalization tamper\n")

    with pytest.raises(cli.CliError, match="stdout|digest|bound|tamper"):
        _validate(benchmark, expected_seed_rows=EXPECTED_SEED_ROWS)


def test_vector_acceptance_seed_rows_must_match_training_reset_rows(
    tmp_path: Path,
) -> None:
    benchmark = _write_finalized_benchmark(
        tmp_path / "gate",
        seed_rows=tuple(range(2001, 2001 + NUM_ENVS)),
    )

    with pytest.raises(cli.CliError, match="seed_rows"):
        _validate(benchmark, expected_seed_rows=EXPECTED_SEED_ROWS)
