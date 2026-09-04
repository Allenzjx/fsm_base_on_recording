from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo import vectorized_isaac_backend


def _args(run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=run_dir,
        num_envs=16,
        measured_ticks=32,
        policy_decisions=2,
        residual_mode="zero",
    )


def test_vector_benchmark_construction_exception_publishes_strict_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        vectorized_isaac_backend, "VectorizedIsaacFSMBackend", FailingBackend
    )
    monkeypatch.setattr(cli, "_seed_values", lambda _args: tuple(range(1001, 1033)))

    exit_code = cli._vector_benchmark(_args(tmp_path), object())

    assert exit_code == 2
    payload = json.loads((tmp_path / "vector_benchmark.json").read_text("utf-8"))
    assert payload["passed"] is False
    assert payload["residual_smoke"] is None
    assert payload["report"]["status"] == "VECTOR_BACKEND_BENCHMARK_FAILED"
    assert payload["report"]["num_envs"] == 16
    assert payload["report"]["true_batched_isaac_verified"] is False
    assert payload["report"]["failure_reasons"]
    assert payload["resource_evidence"]["cuda_memory"]["oom_detected"] is True
    assert payload["resource_evidence"]["contamination"]["evidence_complete"] is False
    assert payload["report"]["failure_details"] == [
        {
            "stage": "backend_construction",
            "exception_type": "RuntimeError",
            "message": "CUDA out of memory",
        }
    ]


def test_vector_benchmark_does_not_swallow_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InterruptedBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        vectorized_isaac_backend, "VectorizedIsaacFSMBackend", InterruptedBackend
    )
    monkeypatch.setattr(cli, "_seed_values", lambda _args: tuple(range(1001, 1033)))

    with pytest.raises(KeyboardInterrupt):
        cli._vector_benchmark(_args(tmp_path), object())
    assert not (tmp_path / "vector_benchmark.json").exists()


def test_core_parser_exposes_matrix_but_not_raw_acceptance_bypass(
    tmp_path: Path,
) -> None:
    parsed = cli._parser().parse_args(
        [
            "train",
            "--run-dir",
            str(tmp_path),
            "--seed",
            "1001",
            "--num-envs",
            "16",
            "--vector-benchmark-matrix",
            str(tmp_path / "vector_benchmark_matrix.json"),
        ]
    )
    assert parsed.vector_benchmark_matrix == tmp_path / "vector_benchmark_matrix.json"

    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "train",
                "--run-dir",
                str(tmp_path),
                "--seed",
                "1001",
                "--num-envs",
                "16",
                "--vector-zero-benchmark-acceptance",
                str(tmp_path / "vector_benchmark.json"),
            ]
        )


def test_vector_worker_rejects_run_directory_outside_managed_root(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(command="vector-benchmark", run_dir=tmp_path)
    with pytest.raises(cli.CliError, match="run directory rejected"):
        cli._validate_common(args)
