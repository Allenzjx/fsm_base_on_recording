"""RSL-RL 5.0 integration for the phase-specific residual environment.

The module intentionally contains no PPO optimizer implementation.  It turns
the versioned project profile into the configuration consumed by the official
``rsl_rl.runners.OnPolicyRunner`` and provides auditable checkpoint helpers.
RSL-RL and Torch imports are lazy so the configuration can be inspected by the
ordinary unit-test interpreter without starting Isaac Sim.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .checkpoint_runtime_capture import CapturedCheckpointBundle


TRAINING_SCHEMA = "wlr50_clean.ppo_training.phase_specific_stability.v1"
CHECKPOINT_MANIFEST_SCHEMA = "wlr50_clean.phase_residual_checkpoint_manifest.v1"
RSL_DISTRIBUTION = "rsl-rl-lib"
RSL_VERSION = "5.0.1"
TRAINING_RNG_STATE_SCHEMA = "wlr50_clean.training_rng_state.v1"
CHECKPOINT_RUNTIME_CONTRACT_FIELDS = (
    "source_git_commit",
    "committed_runtime_content_sha256",
    "actor_observation_dimension",
    "critic_observation_dimension",
    "residual_dimension",
    "physics_hz",
    "decision_hz",
    "files",
    "controller_hash",
    "environment_hash",
    "observation_schema_hash",
    "action_schema_hash",
    "reward_config_hash",
    "phase_snapshot_manifest",
    "phase_snapshot_manifest_sha256",
    "phase_snapshot_bundle_sha256",
    "phase_snapshot_bundle",
    "phase_effective_entry_contract_path",
    "phase_effective_entry_contract_file_sha256",
    "phase_effective_entry_contract_sidecar_path",
    "phase_effective_entry_contract_sidecar_sha256",
    "phase_effective_entry_contract_sha256",
    "phase_effective_entry_contract",
)
DEFAULT_TRAINING_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ppo_training_phase_v1.yaml"
)


class RlLibraryConfigurationError(ValueError):
    """The training profile cannot be represented by the selected library."""


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    path: Path
    seed_train: tuple[int, ...]
    seed_validation: tuple[int, ...]
    seed_locked_test: tuple[int, ...]
    video_seed: int
    physics_hz: float
    decision_hz: float
    ticks_per_decision: int
    timeout_s: float
    initial_num_envs: int
    benchmark_env_counts: tuple[int, ...]
    phase_curriculum_max_decisions: int
    phase_curriculum_reset_cycle_samples: int
    phase_curriculum_occupancy_tolerance: float
    phase_curriculum_baseline_decisions: Mapping[str, int]
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    initial_action_std: float
    gamma: float
    lam: float
    clip_ratio: float
    learning_rate: float
    schedule: str
    target_kl: float
    rollout_length: int
    update_epochs: int
    num_minibatches: int
    value_loss_coefficient: float
    entropy_start: float
    entropy_end: float
    max_grad_norm: float
    deterministic_validation_interval: int
    early_stop_when_promotion_gate_passes: bool
    budgets: Mapping[str, int]
    phase_sampling: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ResumeCheckpointProvenance:
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    global_policy_decisions: int
    stage: str
    checkpoint_infos_match_manifest: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "global_policy_decisions": self.global_policy_decisions,
            "stage": self.stage,
            "checkpoint_infos_match_manifest": self.checkpoint_infos_match_manifest,
        }


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RlLibraryConfigurationError(f"{label} must be positive and finite")
    return result


def _seeds(value: Sequence[Any], label: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise RlLibraryConfigurationError(f"{label} must contain unique non-negative seeds")
    return result


def load_training_profile(
    path: Path | str = DEFAULT_TRAINING_PATH,
) -> TrainingProfile:
    selected = Path(path).resolve()
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != TRAINING_SCHEMA:
        raise RlLibraryConfigurationError("unexpected PPO training schema")
    if payload.get("library") != RSL_DISTRIBUTION or str(payload.get("library_version")) != RSL_VERSION:
        raise RlLibraryConfigurationError("training must use the pinned RSL-RL 5.0.1 profile")

    timing = payload["timing"]
    physics_hz = _positive_float(timing["physics_hz"], "physics_hz")
    decision_hz = _positive_float(timing["policy_decision_hz"], "policy_decision_hz")
    ticks = int(timing["physics_ticks_per_decision"])
    if physics_hz != 120.0 or decision_hz != 15.0 or ticks != 8:
        raise RlLibraryConfigurationError("the training interface must remain 120/15 Hz with 8 ticks")
    timeout_s = _positive_float(timing["episode_timeout_s"], "episode_timeout_s")
    if timeout_s > 200.0:
        raise RlLibraryConfigurationError("episode timeout exceeds 200 seconds")

    seeds = payload["seeds"]
    train = _seeds(seeds["train"], "train seeds")
    validation = _seeds(seeds["validation"], "validation seeds")
    locked = _seeds(seeds["locked_test"], "locked-test seeds")
    video = int(seeds["video"])
    if set(train) & set(validation) or set(train) & set(locked) or set(validation) & set(locked):
        raise RlLibraryConfigurationError("train, validation, and locked-test seeds overlap")
    if video in set(train) | set(validation) | set(locked):
        raise RlLibraryConfigurationError("video seed must be disjoint from data split seeds")

    network = payload["network"]
    actor_dims = tuple(int(item) for item in network["actor_hidden_dims"])
    critic_dims = tuple(int(item) for item in network["critic_hidden_dims"])
    if not actor_dims or not critic_dims or any(item <= 0 for item in (*actor_dims, *critic_dims)):
        raise RlLibraryConfigurationError("actor/critic hidden dimensions are invalid")
    if network.get("initial_actor_mean") != "zero":
        raise RlLibraryConfigurationError("the initial actor mean must be zero")

    ppo = payload["ppo"]
    if not bool(ppo.get("normalize_advantage")) or not bool(ppo.get("clip_value_loss")):
        raise RlLibraryConfigurationError("advantage normalization and clipped value loss are required")
    entropy_start = float(ppo["entropy_coefficient_start"])
    entropy_end = float(ppo["entropy_coefficient_end"])
    if (
        not math.isfinite(entropy_start)
        or not math.isfinite(entropy_end)
        or entropy_start < 0.0
        or entropy_end < 0.0
        or entropy_end > entropy_start
    ):
        raise RlLibraryConfigurationError(
            "entropy coefficients must define a finite non-negative annealing schedule"
        )
    phase_sampling = {str(k): float(v) for k, v in payload["phase_curriculum_sampling"].items()}
    expected_phases = tuple(f"P{i:02d}" for i in range(1, 14))
    if (
        tuple(phase_sampling) != expected_phases
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in phase_sampling.values()
        )
        or not math.isclose(sum(phase_sampling.values()), 1.0, abs_tol=1e-12)
    ):
        raise RlLibraryConfigurationError("phase sampling must cover P01-P13 and sum to one")
    raw_budgets = payload.get("budgets_policy_decisions")
    stage_budget_keys = (
        "smoke",
        "phase_curriculum",
        "full_episode",
        "mild_randomization",
    )
    budget_control_keys = {
        "deterministic_validation_interval",
        "early_stop_when_promotion_gate_passes",
    }
    if not isinstance(raw_budgets, Mapping) or set(raw_budgets) != (
        set(stage_budget_keys) | budget_control_keys
    ):
        raise RlLibraryConfigurationError(
            "budgets_policy_decisions must contain exactly four stages plus "
            "deterministic validation and early-stop controls"
        )
    if any(
        isinstance(raw_budgets[key], bool) or not isinstance(raw_budgets[key], int)
        for key in stage_budget_keys
    ):
        raise RlLibraryConfigurationError("training stage budgets must be strict integers")
    budgets = {key: raw_budgets[key] for key in stage_budget_keys}
    if any(value < 0 for value in budgets.values()):
        raise RlLibraryConfigurationError("training budgets cannot be negative")
    deterministic_validation_interval = raw_budgets[
        "deterministic_validation_interval"
    ]
    if (
        isinstance(deterministic_validation_interval, bool)
        or not isinstance(deterministic_validation_interval, int)
        or deterministic_validation_interval <= 0
    ):
        raise RlLibraryConfigurationError(
            "deterministic_validation_interval must be a positive integer"
        )
    early_stop_when_promotion_gate_passes = raw_budgets[
        "early_stop_when_promotion_gate_passes"
    ]
    if type(early_stop_when_promotion_gate_passes) is not bool:
        raise RlLibraryConfigurationError(
            "early_stop_when_promotion_gate_passes must be a strict boolean"
        )

    env = payload["environment"]
    counts = tuple(int(item) for item in env["benchmark_env_counts"])
    if counts != (8, 16, 32):
        raise RlLibraryConfigurationError("throughput benchmark must cover 8, 16, and 32 envs")
    phase_curriculum_max_decisions = int(env["phase_curriculum_max_decisions"])
    phase_curriculum_reset_cycle_samples = int(
        env["phase_curriculum_reset_cycle_samples"]
    )
    phase_curriculum_occupancy_tolerance = float(
        env["phase_curriculum_occupancy_tolerance_fraction"]
    )
    baseline_phase_decisions = {
        str(key): int(value)
        for key, value in env["phase_curriculum_baseline_decisions"].items()
    }
    if phase_curriculum_max_decisions <= 0:
        raise RlLibraryConfigurationError(
            "phase curriculum decision horizon must be positive"
        )
    if phase_curriculum_reset_cycle_samples < len(expected_phases):
        raise RlLibraryConfigurationError(
            "phase curriculum reset cycle must cover every phase"
        )
    if (
        not math.isfinite(phase_curriculum_occupancy_tolerance)
        or not 0.0 <= phase_curriculum_occupancy_tolerance <= 1.0
    ):
        raise RlLibraryConfigurationError(
            "phase curriculum occupancy tolerance must be within [0, 1]"
        )
    if tuple(baseline_phase_decisions) != expected_phases or any(
        value <= 0 for value in baseline_phase_decisions.values()
    ):
        raise RlLibraryConfigurationError(
            "phase curriculum baseline decisions must contain positive P01-P13 values"
        )
    return TrainingProfile(
        path=selected,
        seed_train=train,
        seed_validation=validation,
        seed_locked_test=locked,
        video_seed=video,
        physics_hz=physics_hz,
        decision_hz=decision_hz,
        ticks_per_decision=ticks,
        timeout_s=timeout_s,
        initial_num_envs=int(env["initial_num_envs"]),
        benchmark_env_counts=counts,
        phase_curriculum_max_decisions=phase_curriculum_max_decisions,
        phase_curriculum_reset_cycle_samples=phase_curriculum_reset_cycle_samples,
        phase_curriculum_occupancy_tolerance=(
            phase_curriculum_occupancy_tolerance
        ),
        phase_curriculum_baseline_decisions=baseline_phase_decisions,
        actor_hidden_dims=actor_dims,
        critic_hidden_dims=critic_dims,
        activation=str(network["activation"]),
        initial_action_std=_positive_float(network["initial_normalized_action_std"], "initial action std"),
        gamma=_positive_float(ppo["discount_gamma"], "gamma"),
        lam=_positive_float(ppo["gae_lambda"], "GAE lambda"),
        clip_ratio=_positive_float(ppo["clip_ratio"], "clip ratio"),
        learning_rate=_positive_float(ppo["learning_rate"], "learning rate"),
        schedule=str(ppo["learning_rate_schedule"]),
        target_kl=_positive_float(ppo["target_kl"], "target KL"),
        rollout_length=int(ppo["rollout_length_per_env"]),
        update_epochs=int(ppo["update_epochs"]),
        num_minibatches=int(ppo["num_minibatches"]),
        value_loss_coefficient=_positive_float(ppo["value_loss_coefficient"], "value loss coefficient"),
        entropy_start=entropy_start,
        entropy_end=entropy_end,
        max_grad_norm=_positive_float(ppo["max_grad_norm"], "max grad norm"),
        deterministic_validation_interval=deterministic_validation_interval,
        early_stop_when_promotion_gate_passes=(
            early_stop_when_promotion_gate_passes
        ),
        budgets=budgets,
        phase_sampling=phase_sampling,
    )


def installed_rsl_version() -> str:
    return importlib.metadata.version(RSL_DISTRIBUTION)


def assert_supported_rsl_runtime() -> str:
    actual = installed_rsl_version()
    if actual != RSL_VERSION:
        raise RlLibraryConfigurationError(
            f"Isaac Lab requires {RSL_DISTRIBUTION}=={RSL_VERSION}; found {actual}"
        )
    return actual


def seed_training_rngs(seed: int) -> Mapping[str, Any]:
    """Seed every RNG used by network construction and PPO action sampling.

    RSL-RL 5.0.1 carries a ``seed`` key in common runner configurations but its
    :class:`OnPolicyRunner` does not consume that key.  Seeding therefore has
    to happen explicitly before the live environment and runner are built.
    ``PYTHONHASHSEED`` is also checked here, while the launcher remains
    responsible for setting it before Python starts.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise RlLibraryConfigurationError(
            "training RNG seed must be a non-boolean uint32 integer"
        )
    import numpy as np  # type: ignore
    import torch  # type: ignore

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    declared_hash_seed = os.environ.get("PYTHONHASHSEED")
    if declared_hash_seed not in (None, str(seed)):
        raise RlLibraryConfigurationError(
            "PYTHONHASHSEED differs from the requested training seed"
        )
    os.environ["PYTHONHASHSEED"] = str(seed)
    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": cuda_available,
        "torch_cuda_device_count": int(torch.cuda.device_count())
        if cuda_available
        else 0,
        "python_hash_seed": str(seed),
        "torch_deterministic_algorithms_forced": False,
    }


def _encode_torch_rng_state(state: Any) -> str:
    payload = bytes(int(value) for value in state.detach().to("cpu").tolist())
    return base64.b64encode(payload).decode("ascii")


def _decode_torch_rng_state(encoded: Any, *, label: str) -> Any:
    if not isinstance(encoded, str) or not encoded:
        raise RlLibraryConfigurationError(f"{label} is missing or invalid")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise RlLibraryConfigurationError(f"{label} is not valid base64") from exc
    if not payload:
        raise RlLibraryConfigurationError(f"{label} decoded to an empty state")
    import torch  # type: ignore

    return torch.tensor(list(payload), dtype=torch.uint8, device="cpu")


def capture_training_rng_state(*, seed: int) -> dict[str, Any]:
    """Return a JSON-safe snapshot sufficient to continue PPO exploration."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise RlLibraryConfigurationError(
            "training RNG state seed must be a non-boolean uint32 integer"
        )
    import numpy as np  # type: ignore
    import torch  # type: ignore

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_available = bool(torch.cuda.is_available())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else []
    return {
        "schema": TRAINING_RNG_STATE_SCHEMA,
        "seed": seed,
        "python_random": {
            "version": int(python_state[0]),
            "state": [int(value) for value in python_state[1]],
            "gauss_next": None
            if python_state[2] is None
            else float(python_state[2]),
        },
        "numpy_random": {
            "bit_generator": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1].tolist()],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": _encode_torch_rng_state(torch.get_rng_state()),
        "torch_cuda": [_encode_torch_rng_state(state) for state in cuda_states],
        "torch_cuda_device_count": len(cuda_states),
    }


def restore_training_rng_state(
    state: Mapping[str, Any] | None, *, expected_seed: int
) -> Mapping[str, Any]:
    """Restore a checkpointed training RNG snapshot before the next rollout."""

    if not isinstance(state, Mapping) or state.get("schema") != TRAINING_RNG_STATE_SCHEMA:
        raise RlLibraryConfigurationError(
            "resume checkpoint omits a supported training RNG state"
        )
    if state.get("seed") != expected_seed:
        raise RlLibraryConfigurationError(
            "resume checkpoint training RNG seed differs from the requested seed"
        )
    python_state = state.get("python_random")
    numpy_state = state.get("numpy_random")
    if not isinstance(python_state, Mapping) or not isinstance(numpy_state, Mapping):
        raise RlLibraryConfigurationError("resume checkpoint RNG payload is malformed")
    raw_python_values = python_state.get("state")
    raw_numpy_values = numpy_state.get("state")
    if (
        not isinstance(raw_python_values, list)
        or not raw_python_values
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_python_values)
        or not isinstance(raw_numpy_values, list)
        or not raw_numpy_values
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_numpy_values)
    ):
        raise RlLibraryConfigurationError("resume checkpoint RNG arrays are malformed")

    import numpy as np  # type: ignore
    import torch  # type: ignore

    random.setstate(
        (
            int(python_state.get("version")),
            tuple(int(value) for value in raw_python_values),
            python_state.get("gauss_next"),
        )
    )
    np.random.set_state(
        (
            str(numpy_state.get("bit_generator")),
            np.asarray(raw_numpy_values, dtype=np.uint32),
            int(numpy_state.get("position")),
            int(numpy_state.get("has_gauss")),
            float(numpy_state.get("cached_gaussian")),
        )
    )
    torch.set_rng_state(
        _decode_torch_rng_state(state.get("torch_cpu"), label="torch CPU RNG state")
    )
    raw_cuda_states = state.get("torch_cuda")
    declared_cuda_count = state.get("torch_cuda_device_count")
    if (
        not isinstance(raw_cuda_states, list)
        or isinstance(declared_cuda_count, bool)
        or not isinstance(declared_cuda_count, int)
        or declared_cuda_count < 0
        or len(raw_cuda_states) != declared_cuda_count
    ):
        raise RlLibraryConfigurationError("resume checkpoint CUDA RNG state is malformed")
    actual_cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if actual_cuda_count != declared_cuda_count:
        raise RlLibraryConfigurationError(
            "resume checkpoint CUDA RNG device count differs from the current runtime"
        )
    if declared_cuda_count:
        torch.cuda.set_rng_state_all(
            [
                _decode_torch_rng_state(value, label=f"torch CUDA RNG state {index}")
                for index, value in enumerate(raw_cuda_states)
            ]
        )
    return {
        "schema": TRAINING_RNG_STATE_SCHEMA,
        "seed": expected_seed,
        "python_random_restored": True,
        "numpy_random_restored": True,
        "torch_cpu_restored": True,
        "torch_cuda_states_restored": declared_cuda_count,
    }


def iterations_for_policy_decisions(
    decisions: int, *, num_envs: int, rollout_length: int
) -> int:
    if decisions <= 0 or num_envs <= 0 or rollout_length <= 0:
        raise RlLibraryConfigurationError("decision budget, num_envs and rollout length must be positive")
    return int(math.ceil(decisions / (num_envs * rollout_length)))


def planned_entropy_anneal_policy_decisions(profile: TrainingProfile) -> int:
    """Return the versioned nominal training plan, excluding validation cadence."""

    stage_names = ("smoke", "phase_curriculum", "full_episode", "mild_randomization")
    missing = [name for name in stage_names if name not in profile.budgets]
    if missing:
        raise RlLibraryConfigurationError(
            f"training profile omits entropy-plan budgets {missing}"
        )
    total = sum(int(profile.budgets[name]) for name in stage_names)
    if total <= 0:
        raise RlLibraryConfigurationError(
            "planned entropy anneal policy-decision budget must be positive"
        )
    return total


def entropy_coefficient_at_policy_decision(
    start: float,
    end: float,
    *,
    global_policy_decision: int,
    planned_policy_decisions: int,
) -> float:
    """Interpolate entropy from the checkpoint global step, clamped at plan end."""

    start_value = float(start)
    end_value = float(end)
    step = _strict_nonnegative_global_step(
        global_policy_decision, label="entropy global_policy_decision"
    )
    if isinstance(planned_policy_decisions, bool) or not isinstance(
        planned_policy_decisions, int
    ) or planned_policy_decisions <= 0:
        raise RlLibraryConfigurationError(
            "entropy planned_policy_decisions must be a positive integer"
        )
    if (
        not math.isfinite(start_value)
        or start_value < 0.0
        or not math.isfinite(end_value)
        or end_value < 0.0
    ):
        raise RlLibraryConfigurationError(
            "entropy schedule endpoints must be finite and non-negative"
        )
    fraction = min(float(step) / float(planned_policy_decisions), 1.0)
    return start_value + (end_value - start_value) * fraction


def build_rsl_runner_config(
    profile: TrainingProfile,
    *,
    seed: int,
    max_iterations: int,
    save_interval: int = 1,
    experiment_name: str = "wlr50_phase_residual_ppo",
    entropy_coefficient: float | None = None,
) -> dict[str, Any]:
    """Build the native RSL-RL 5.0 dictionary without importing the library."""

    if max_iterations <= 0 or save_interval <= 0:
        raise RlLibraryConfigurationError("runner iterations and save interval must be positive")
    entropy = profile.entropy_start if entropy_coefficient is None else float(entropy_coefficient)
    if not math.isfinite(entropy) or entropy < 0.0:
        raise RlLibraryConfigurationError("entropy coefficient must be finite and non-negative")
    model_common = {
        "class_name": "MLPModel",
        "activation": profile.activation,
        "obs_normalization": False,
    }
    return {
        "class_name": "OnPolicyRunner",
        "seed": int(seed),
        "device": "cuda:0",
        "num_steps_per_env": profile.rollout_length,
        "max_iterations": int(max_iterations),
        "save_interval": int(save_interval),
        "experiment_name": str(experiment_name),
        "run_name": "",
        "logger": "tensorboard",
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            **model_common,
            "hidden_dims": list(profile.actor_hidden_dims),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": profile.initial_action_std,
                "std_type": "scalar",
            },
        },
        "critic": {
            **model_common,
            "hidden_dims": list(profile.critic_hidden_dims),
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": profile.update_epochs,
            "num_mini_batches": profile.num_minibatches,
            "clip_param": profile.clip_ratio,
            "gamma": profile.gamma,
            "lam": profile.lam,
            "value_loss_coef": profile.value_loss_coefficient,
            "entropy_coef": entropy,
            "learning_rate": profile.learning_rate,
            "max_grad_norm": profile.max_grad_norm,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": profile.schedule,
            "desired_kl": profile.target_kl,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "share_cnn_encoders": False,
        },
        "multi_gpu": None,
    }


def construct_runner(env: Any, config: Mapping[str, Any], *, log_dir: Path | str | None) -> Any:
    """Construct the official runner; call only after AppLauncher exists."""

    assert_supported_rsl_runtime()
    from rsl_rl.runners import OnPolicyRunner  # type: ignore

    return OnPolicyRunner(
        env,
        copy.deepcopy(dict(config)),
        log_dir=None if log_dir is None else str(Path(log_dir).resolve()),
        device=str(config.get("device", "cuda:0")),
    )


def initialize_zero_mean_actor(runner: Any) -> None:
    """Make the initial deterministic residual exactly zero for every input."""

    import torch  # type: ignore

    linears = [module for module in runner.alg.actor.mlp.modules() if isinstance(module, torch.nn.Linear)]
    if not linears:
        raise RlLibraryConfigurationError("RSL actor has no linear output layer")
    with torch.no_grad():
        linears[-1].weight.zero_()
        if linears[-1].bias is not None:
            linears[-1].bias.zero_()


def zero_mean_actor_output_layer_verified(runner: Any) -> bool:
    """Return true only when the loaded actor's final affine map is exact zero."""

    import torch  # type: ignore

    linears = [
        module
        for module in runner.alg.actor.mlp.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linears:
        raise RlLibraryConfigurationError("RSL actor has no linear output layer")
    output = linears[-1]
    if not torch.isfinite(output.weight).all() or torch.count_nonzero(output.weight):
        return False
    if output.bias is not None and (
        not torch.isfinite(output.bias).all() or torch.count_nonzero(output.bias)
    ):
        return False
    return True


def set_entropy_coefficient(runner: Any, value: float) -> None:
    coefficient = float(value)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise RlLibraryConfigurationError("entropy coefficient must be finite and non-negative")
    runner.alg.entropy_coef = coefficient


def linear_entropy_schedule(
    start: float, end: float, *, num_updates: int
) -> tuple[float, ...]:
    """Return the coefficient applied immediately before each PPO update."""

    start_value = float(start)
    end_value = float(end)
    if (
        not math.isfinite(start_value)
        or start_value < 0.0
        or not math.isfinite(end_value)
        or end_value < 0.0
    ):
        raise RlLibraryConfigurationError(
            "entropy schedule endpoints must be finite and non-negative"
        )
    updates = int(num_updates)
    if updates <= 0:
        raise RlLibraryConfigurationError("entropy schedule needs at least one update")
    if updates == 1:
        return (start_value,)
    return tuple(
        start_value + (end_value - start_value) * index / (updates - 1)
        for index in range(updates)
    )


def learn_with_entropy_schedule(
    runner: Any,
    *,
    num_learning_iterations: int,
    entropy_start: float,
    entropy_end: float,
    init_at_random_ep_len: bool = False,
) -> tuple[float, ...]:
    """Run native RSL learning while setting entropy before every update.

    RSL-RL owns rollout collection and PPO optimization.  This adapter only
    wraps the algorithm's update boundary so the configured endpoint is not a
    dead field.  It restores the original method even when learning fails and
    fails closed if RSL performs a different number of updates than requested.
    """

    schedule = linear_entropy_schedule(
        entropy_start, entropy_end, num_updates=num_learning_iterations
    )
    algorithm = getattr(runner, "alg", None)
    if algorithm is None or not callable(getattr(algorithm, "update", None)):
        raise RlLibraryConfigurationError("runner.alg.update is unavailable")
    original_update = algorithm.update
    instance_attributes = getattr(algorithm, "__dict__", None)
    had_instance_update = bool(
        isinstance(instance_attributes, dict) and "update" in instance_attributes
    )
    previous_instance_update = (
        instance_attributes.get("update") if had_instance_update else None
    )
    applied: list[float] = []

    def scheduled_update(*args: Any, **kwargs: Any) -> Any:
        if len(applied) >= len(schedule):
            raise RlLibraryConfigurationError(
                "RSL performed more PPO updates than the entropy schedule"
            )
        coefficient = schedule[len(applied)]
        set_entropy_coefficient(runner, coefficient)
        applied.append(coefficient)
        return original_update(*args, **kwargs)

    try:
        algorithm.update = scheduled_update
    except (AttributeError, TypeError) as exc:
        raise RlLibraryConfigurationError(
            "runner.alg.update cannot be instrumented for entropy annealing"
        ) from exc
    try:
        runner.learn(
            num_learning_iterations=len(schedule),
            init_at_random_ep_len=bool(init_at_random_ep_len),
        )
    finally:
        if had_instance_update:
            algorithm.update = previous_instance_update
        else:
            try:
                del algorithm.update
            except AttributeError:
                algorithm.update = original_update
    if tuple(applied) != schedule:
        raise RlLibraryConfigurationError(
            "RSL PPO update count did not match the entropy schedule"
        )
    return tuple(applied)


def deterministic_action(runner: Any, observations: Any) -> Any:
    """Return the actor mean, never a stochastic video/evaluation sample."""

    import torch  # type: ignore

    policy = runner.get_inference_policy(device=str(runner.device))
    policy.eval()
    with torch.inference_mode():
        return policy(observations.to(runner.device), stochastic_output=False)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_nonnegative_global_step(value: Any, *, label: str) -> int:
    """Return a JSON/RSL manifest step without accepting bool coercion."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RlLibraryConfigurationError(
            f"{label} must be a non-boolean, non-negative integer"
        )
    return value


def _checkpoint_digest(value: Any, *, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RlLibraryConfigurationError(f"{label} must be a SHA-256 digest")
    return digest


def validate_resume_checkpoint_provenance(
    checkpoint_path: Path | str,
    checkpoint_infos: Mapping[str, Any] | None,
    *,
    manifest_path: Path | str | None = None,
    expected_global_policy_decisions: int | None = None,
    expected_runtime_contract: Mapping[str, Any] | None = None,
    captured_bundle: CapturedCheckpointBundle | None = None,
) -> ResumeCheckpointProvenance:
    """Fail closed unless an RSL checkpoint and its sidecar describe one resume state.

    RSL stores ``infos`` *inside* the checkpoint, so the final whole-file digest
    cannot also be embedded in those infos without a self-reference.  The
    sidecar therefore binds the actual checkpoint bytes to a SHA-256 digest,
    while exact per-key comparison binds the loaded infos to that same sidecar.
    """

    checkpoint = Path(checkpoint_path).resolve()
    if captured_bundle is None:
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RlLibraryConfigurationError(
                f"resume checkpoint is missing or empty: {checkpoint}"
            )
    else:
        if checkpoint != captured_bundle.source_checkpoint_path:
            raise RlLibraryConfigurationError(
                "captured checkpoint bundle belongs to a different source path"
            )
        try:
            captured_bundle.assert_sources_unchanged()
            captured_bundle.assert_private_copy_unchanged()
        except RuntimeError as exc:
            raise RlLibraryConfigurationError(
                f"captured checkpoint bundle is no longer immutable: {exc}"
            ) from exc
    if not isinstance(checkpoint_infos, Mapping):
        raise RlLibraryConfigurationError("resume checkpoint infos are missing or invalid")
    infos = dict(checkpoint_infos)
    if infos.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise RlLibraryConfigurationError("resume checkpoint infos have the wrong schema")
    infos_step = _strict_nonnegative_global_step(
        infos.get("global_policy_decisions"),
        label="resume checkpoint infos global_policy_decisions",
    )
    stage = infos.get("stage")
    if not isinstance(stage, str) or not stage.strip():
        raise RlLibraryConfigurationError("resume checkpoint infos stage is missing or invalid")
    if expected_global_policy_decisions is not None:
        expected_step = _strict_nonnegative_global_step(
            expected_global_policy_decisions,
            label="expected global_policy_decisions",
        )
        if infos_step != expected_step:
            raise RlLibraryConfigurationError(
                "resume checkpoint global_policy_decisions does not match the expected step"
            )

    sidecar = (
        checkpoint.with_name(checkpoint.stem + "_manifest.json")
        if manifest_path is None
        else Path(manifest_path).resolve()
    )
    if captured_bundle is None:
        if not sidecar.is_file() or sidecar.stat().st_size <= 0:
            raise RlLibraryConfigurationError(
                f"resume checkpoint manifest is missing or empty: {sidecar}"
            )
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RlLibraryConfigurationError(
                f"resume checkpoint manifest is not valid JSON: {sidecar}"
            ) from exc
    else:
        if sidecar != captured_bundle.source_manifest_path:
            raise RlLibraryConfigurationError(
                "captured checkpoint bundle belongs to a different manifest path"
            )
        payload = dict(captured_bundle.manifest_payload)
    if not isinstance(payload, Mapping):
        raise RlLibraryConfigurationError("resume checkpoint manifest must be a JSON object")
    manifest = dict(payload)
    if manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA:
        raise RlLibraryConfigurationError("resume checkpoint manifest has the wrong schema")
    manifest_step = _strict_nonnegative_global_step(
        manifest.get("global_policy_decisions"),
        label="resume checkpoint manifest global_policy_decisions",
    )
    if manifest_step != infos_step:
        raise RlLibraryConfigurationError(
            "resume checkpoint infos and manifest global_policy_decisions disagree"
        )
    if manifest.get("stage") != stage:
        raise RlLibraryConfigurationError("resume checkpoint infos and manifest stage disagree")

    declared_path = manifest.get("checkpoint_path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != checkpoint:
        raise RlLibraryConfigurationError(
            "resume checkpoint manifest is bound to a different checkpoint path"
        )
    actual_checkpoint_sha256 = (
        sha256_file(checkpoint)
        if captured_bundle is None
        else captured_bundle.checkpoint_sha256
    )
    declared_checkpoint_sha256 = _checkpoint_digest(
        manifest.get("checkpoint_sha256"),
        label="resume checkpoint manifest checkpoint_sha256",
    )
    if declared_checkpoint_sha256 != actual_checkpoint_sha256:
        raise RlLibraryConfigurationError(
            "resume checkpoint bytes do not match the checkpoint manifest"
        )

    for key, value in infos.items():
        if key not in manifest or manifest[key] != value:
            raise RlLibraryConfigurationError(
                f"resume checkpoint infos and manifest disagree for {key!r}"
            )

    if expected_runtime_contract is not None:
        if not isinstance(expected_runtime_contract, Mapping):
            raise RlLibraryConfigurationError(
                "expected checkpoint runtime contract must be a mapping"
            )
        for field in CHECKPOINT_RUNTIME_CONTRACT_FIELDS:
            if field not in expected_runtime_contract:
                raise RlLibraryConfigurationError(
                    f"current runtime contract omits {field!r}"
                )
            if field not in infos:
                raise RlLibraryConfigurationError(
                    f"resume checkpoint infos omit runtime field {field!r}"
                )
            if infos[field] != expected_runtime_contract[field]:
                raise RlLibraryConfigurationError(
                    f"resume checkpoint runtime contract differs for {field!r}"
                )

    if captured_bundle is not None:
        try:
            captured_bundle.assert_loaded_infos(infos)
            captured_bundle.assert_sources_unchanged()
        except RuntimeError as exc:
            raise RlLibraryConfigurationError(
                f"captured checkpoint provenance changed: {exc}"
            ) from exc

    return ResumeCheckpointProvenance(
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_checkpoint_sha256,
        manifest_path=sidecar,
        manifest_sha256=(
            sha256_file(sidecar)
            if captured_bundle is None
            else captured_bundle.manifest_sha256
        ),
        global_policy_decisions=infos_step,
        stage=stage,
    )


def save_checkpoint_with_manifest(
    runner: Any,
    checkpoint_path: Path | str,
    *,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    target = Path(checkpoint_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    # RSL-RL 5.0.1 creates ``Logger.writer`` only when ``learn()`` starts,
    # but ``OnPolicyRunner.save()`` reads it unconditionally.  The immutable
    # zero-residual checkpoint is intentionally saved before the first learn
    # call, so make that upstream lazy attribute explicit without opening a
    # SummaryWriter early.  ``learn()`` will initialize/replace it normally.
    logger = getattr(runner, "logger", None)
    if logger is not None and not hasattr(logger, "writer"):
        logger.writer = None
    runner.save(str(target), infos=dict(manifest))
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("RSL-RL did not produce a checkpoint")
    manifest_path = target.with_name(target.stem + "_manifest.json")
    payload = {**dict(manifest), "checkpoint_path": str(target), "checkpoint_sha256": sha256_file(target)}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, manifest_path


def optimizer_learning_rate(runner: Any) -> float:
    """Read the single effective PPO optimizer learning rate fail-closed."""

    algorithm = getattr(runner, "alg", None)
    optimizer = getattr(algorithm, "optimizer", None)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise RlLibraryConfigurationError("runner PPO optimizer has no parameter groups")
    rates = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise RlLibraryConfigurationError("runner PPO optimizer group is malformed")
        rate = float(group.get("lr"))
        if not math.isfinite(rate) or rate <= 0.0:
            raise RlLibraryConfigurationError(
                "runner PPO optimizer learning rate is invalid"
            )
        rates.append(rate)
    if any(not math.isclose(rate, rates[0], rel_tol=0.0, abs_tol=0.0) for rate in rates[1:]):
        raise RlLibraryConfigurationError(
            "runner PPO optimizer parameter groups use different learning rates"
        )
    return rates[0]


def synchronize_loaded_optimizer_learning_rate(
    runner: Any, *, expected: float | None = None
) -> float:
    """Synchronize RSL's adaptive-LR scalar with its restored optimizer state.

    RSL-RL 5.0.1 restores the optimizer parameter groups but not the separate
    ``PPO.learning_rate`` attribute used by its adaptive KL schedule.  Without
    this synchronization, the first adaptation after a resume jumps back to
    the profile's initial learning rate.
    """

    rate = optimizer_learning_rate(runner)
    if expected is not None:
        expected_rate = float(expected)
        if not math.isfinite(expected_rate) or expected_rate <= 0.0:
            raise RlLibraryConfigurationError(
                "checkpoint optimizer_learning_rate is invalid"
            )
        if not math.isclose(rate, expected_rate, rel_tol=1.0e-12, abs_tol=0.0):
            raise RlLibraryConfigurationError(
                "checkpoint optimizer state and manifest learning rate disagree"
            )
    runner.alg.learning_rate = rate
    return rate


def load_checkpoint_round_trip(
    runner: Any,
    checkpoint_path: Path | str,
    *,
    captured_bundle: CapturedCheckpointBundle | None = None,
) -> Mapping[str, Any]:
    source = Path(checkpoint_path).resolve()
    if captured_bundle is None:
        target = source
        if not target.is_file():
            raise FileNotFoundError(target)
    else:
        if source != captured_bundle.source_checkpoint_path:
            raise RlLibraryConfigurationError(
                "checkpoint loader received a capture for a different source path"
            )
        try:
            captured_bundle.assert_unchanged()
        except RuntimeError as exc:
            raise RlLibraryConfigurationError(
                f"checkpoint capture failed before runner load: {exc}"
            ) from exc
        target = captured_bundle.private_checkpoint_path
    infos = runner.load(str(target), strict=True, map_location=str(runner.device))
    if infos is None:
        return {}
    if not isinstance(infos, Mapping):
        raise RlLibraryConfigurationError(
            "RSL checkpoint infos must be a mapping or null"
        )
    result = dict(infos)
    if captured_bundle is not None:
        try:
            result = captured_bundle.assert_loaded_infos(result)
            captured_bundle.assert_sources_unchanged()
        except RuntimeError as exc:
            raise RlLibraryConfigurationError(
                f"checkpoint capture failed after runner load: {exc}"
            ) from exc
    expected_rate = result.get("optimizer_learning_rate")
    if expected_rate is not None:
        synchronize_loaded_optimizer_learning_rate(runner, expected=expected_rate)
    return result
