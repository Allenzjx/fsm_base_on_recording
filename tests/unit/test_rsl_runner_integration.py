from __future__ import annotations

import importlib.util

import pytest

from wlr50_clean.ppo.rl_library_wrapper import (
    CHECKPOINT_MANIFEST_SCHEMA,
    build_rsl_runner_config,
    construct_runner,
    initialize_zero_mean_actor,
    load_checkpoint_round_trip,
    load_training_profile,
    save_checkpoint_with_manifest,
    validate_resume_checkpoint_provenance,
)


@pytest.mark.skipif(importlib.util.find_spec("rsl_rl") is None, reason="RSL-RL not installed")
def test_official_rsl_runner_constructs_zero_actor_and_round_trips_provenance(
    tmp_path,
) -> None:
    import torch
    from tensordict import TensorDict

    class DummyVecEnv:
        num_envs = 2
        num_actions = 12
        max_episode_length = 10
        device = "cpu"
        cfg = {}
        episode_length_buf = torch.zeros(2, dtype=torch.long)

        def get_observations(self):
            return TensorDict(
                {
                    "policy": torch.zeros(2, 125),
                    "critic": torch.zeros(2, 125),
                },
                batch_size=[2],
            )

        def step(self, actions):
            return (
                self.get_observations(),
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                {
                    "time_outs": torch.zeros(2, dtype=torch.bool),
                    "log": {},
                },
            )

    profile = load_training_profile()
    config = build_rsl_runner_config(profile, seed=1, max_iterations=1)
    config["device"] = "cpu"
    runner = construct_runner(DummyVecEnv(), config, log_dir=None)
    initialize_zero_mean_actor(runner)
    action = runner.get_inference_policy(device="cpu")(
        DummyVecEnv().get_observations(), stochastic_output=False
    )
    assert tuple(action.shape) == (2, 12)
    assert torch.count_nonzero(action).item() == 0

    manifest = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "injected-rsl-abi",
        "global_policy_decisions": 256,
        "training_seed": 1,
    }
    checkpoint, sidecar = save_checkpoint_with_manifest(
        runner,
        tmp_path / "official_rsl.pt",
        manifest=manifest,
    )
    loaded_infos = load_checkpoint_round_trip(runner, checkpoint)
    provenance = validate_resume_checkpoint_provenance(
        checkpoint,
        loaded_infos,
        manifest_path=sidecar,
        expected_global_policy_decisions=256,
    )

    assert loaded_infos == manifest
    assert provenance.global_policy_decisions == 256
