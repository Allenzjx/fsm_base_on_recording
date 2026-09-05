"""Pure, profile-derived train/screen cadence; no simulator or optimizer imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE_CADENCE_SCHEMA = "wlr50_clean.ppo_training_stage_cadence.v1"
TRAINING_CADENCE_SCHEMA = "wlr50_clean.ppo_training_cadence.v1"
STAGES = ("smoke", "phase-curriculum", "full-episode")


class TrainingCadenceError(ValueError):
    """The configured budget cannot produce the required honest training window."""


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingCadenceError(f"{label} must be a positive non-boolean integer")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingCadenceError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise TrainingCadenceError(f"{label} must be finite and positive")
    return result


def derive_stage_cadence(
    *, stage: str, num_envs: int, budgets: Mapping[str, int],
    benchmark_env_counts: Sequence[int], rollout_length: int,
    decision_hz: float, timeout_s: float, base_validation_interval: int,
) -> dict[str, Any]:
    """Derive an exact chunk request; full chunks span a complete timeout window.

    A window is available rollout capacity, not a claim that physical resets or
    successes occurred. Those outcomes must remain independently measured.
    """
    if stage not in STAGES:
        raise TrainingCadenceError(f"unsupported cadence stage: {stage!r}")
    n = _integer(num_envs, "num_envs")
    rollout = _integer(rollout_length, "rollout_length")
    base = _integer(base_validation_interval, "base_validation_interval")
    hz = _positive(decision_hz, "decision_hz")
    timeout = _positive(timeout_s, "timeout_s")
    if not math.isfinite(hz * timeout):
        raise TrainingCadenceError("timeout decision window is non-finite")
    if not isinstance(budgets, Mapping):
        raise TrainingCadenceError("stage budgets must be a mapping")
    stage_budgets = {
        name: _integer(budgets.get(name.replace("-", "_")), f"{name} budget")
        for name in STAGES
    }
    if isinstance(benchmark_env_counts, (str, bytes)):
        raise TrainingCadenceError("benchmark_env_counts must be a sequence")
    counts = tuple(_integer(value, "benchmark capacity") for value in benchmark_env_counts)
    if not counts or len(set(counts)) != len(counts) or tuple(sorted(counts)) != counts:
        raise TrainingCadenceError("benchmark capacities must be unique, ascending and non-empty")
    max_n = max(counts)
    if stage == "phase-curriculum":
        if n != 1:
            raise TrainingCadenceError("phase-curriculum cadence requires one environment")
    elif n not in counts:
        raise TrainingCadenceError(f"{stage} num_envs must be a configured benchmark capacity")
    budget = stage_budgets[stage]
    if stage == "full-episode":
        if max_n % n or (budget * n) % max_n:
            raise TrainingCadenceError("full-episode cadence requires exactly divisible capacity/budget")
        requested = budget * n // max_n
        chunks = max_n // n
        basis = "full_stage_budget_scaled_by_selected_capacity"
    else:
        if budget % base:
            raise TrainingCadenceError(f"{stage} budget must divide into exact base intervals")
        requested, chunks = base, budget // base
        basis = "configured_global_policy_decision_interval"
    _integer(requested, "requested chunk")
    _integer(chunks, "chunk count")
    batch = n * rollout
    iterations = (requested + batch - 1) // batch
    per_env = iterations * rollout
    minimum_window = math.ceil(timeout * hz)
    if stage == "full-episode" and per_env < minimum_window:
        raise TrainingCadenceError("full-episode cadence cannot span the configured episode timeout")
    return {
        "schema": STAGE_CADENCE_SCHEMA,
        "stage": stage, "num_envs": n, "cadence_basis": basis,
        "base_validation_interval_policy_decisions": base,
        "base_validation_interval_scope": "smoke_and_phase_curriculum",
        "stage_requested_policy_decisions": budget,
        "requested_policy_decisions_per_chunk": requested,
        "maximum_chunk_count": chunks,
        "ppo_iterations_per_chunk": iterations,
        "rollout_length": rollout, "ppo_batch_policy_decisions": batch,
        "actual_policy_decisions_per_chunk": iterations * batch,
        "policy_decisions_per_env_per_chunk": per_env,
        "benchmark_env_counts": list(counts), "maximum_benchmark_num_envs": max_n,
        "policy_decision_hz": hz, "episode_timeout_s": timeout,
        "minimum_full_window_policy_decisions_per_env": minimum_window,
        "full_window_covers_episode_timeout": True if stage == "full-episode" else None,
    }


def validate_training_chunk_cadence(
    expected: Mapping[str, Any], *, requested_policy_decisions: int,
    iterations: int, stage_policy_decisions: int,
) -> None:
    """Validate against a freshly derived record, never an untrusted declaration."""
    for value, label, key in (
        (requested_policy_decisions, "requested decisions", "requested_policy_decisions_per_chunk"),
        (iterations, "iterations", "ppo_iterations_per_chunk"),
        (stage_policy_decisions, "actual decisions", "actual_policy_decisions_per_chunk"),
    ):
        if _integer(value, label) != expected[key]:
            raise TrainingCadenceError(f"training {label} differs from the derived stage cadence")


def cadence_inputs_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only configured inputs from caller-owned, already captured bytes."""
    try:
        return {
            "budgets": payload["budgets_policy_decisions"],
            "benchmark_env_counts": payload["environment"]["benchmark_env_counts"],
            "rollout_length": payload["ppo"]["rollout_length_per_env"],
            "decision_hz": payload["timing"]["policy_decision_hz"],
            "timeout_s": payload["timing"]["episode_timeout_s"],
            "base_validation_interval": payload["budgets_policy_decisions"]["deterministic_validation_interval"],
        }
    except (KeyError, TypeError) as exc:
        raise TrainingCadenceError("profile omits required cadence inputs") from exc


def derive_training_cadence(*, selected_num_envs: int, **inputs: Any) -> dict[str, Any]:
    plans = [derive_stage_cadence(stage=stage,
             num_envs=1 if stage == "phase-curriculum" else selected_num_envs,
             **inputs) for stage in STAGES]
    chunks = []
    for plan in plans:
        for stage_index in range(plan["maximum_chunk_count"]):
            chunks.append({"index": len(chunks), "stage_chunk_index": stage_index,
                           "training_cadence": dict(plan)})
    return {
        "schema": TRAINING_CADENCE_SCHEMA,
        "selected_num_envs": selected_num_envs,
        "base_validation_interval_policy_decisions": inputs["base_validation_interval"],
        "base_validation_interval_scope": "smoke_and_phase_curriculum",
        "stage_plans": plans, "chunks": chunks,
        "maximum_chunk_count": len(chunks),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Thin read-only JSON descriptor for PowerShell; never launches Isaac."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--describe-plan", action="store_true", required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--selected-num-envs", type=int, required=True)
    args = parser.parse_args(argv)
    import yaml

    path = args.training_config.resolve(strict=True)
    data = path.read_bytes()
    payload = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != "wlr50_clean.ppo_training.phase_specific_stability.v1":
        raise TrainingCadenceError("unexpected training profile schema")
    plan = derive_training_cadence(selected_num_envs=args.selected_num_envs,
                                   **cadence_inputs_from_payload(payload))
    print(json.dumps({"profile": {"path": str(path), "bytes": len(data),
          "sha256": hashlib.sha256(data).hexdigest()}, "plan": plan}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
