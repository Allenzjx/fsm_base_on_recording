from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import cli
from wlr50_clean.ppo import checkpoint_promotion
from wlr50_clean.ppo.rl_library_wrapper import CHECKPOINT_RUNTIME_CONTRACT_FIELDS
from wlr50_clean.ppo.evaluation_artifacts import (
    CANONICAL_EPISODE_FILES,
    collect_fresh_process_episode_workers,
)


_TEST_FROZEN_HASHES = {
    "controller_hash": "a" * 64,
    "environment_hash": "b" * 64,
    "observation_schema_hash": "c" * 64,
    "action_schema_hash": "d" * 64,
    "reward_config_hash": "e" * 64,
}


def _best_validation_manifest(checkpoint: Path) -> Path:
    path = checkpoint.with_name("checkpoint_best_validation_manifest.json")
    payload = {
        "schema": checkpoint_promotion.CHECKPOINT_MANIFEST_SCHEMA,
        "stage": "full-episode",
        "training_seed": 1001,
        "global_policy_decisions": 100_000,
        "actor_observation_dimension": 125,
        "critic_observation_dimension": 125,
        "residual_dimension": 12,
        "physics_hz": 120.0,
        "decision_hz": 15.0,
        "files": {"training.yaml": "f" * 64},
        **_TEST_FROZEN_HASHES,
        "publication_role": "best_validation",
        "validation_promotion_authorized": True,
        "locked_test_authorized": False,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": cli._sha256(checkpoint),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class _FakeActions:
    shape = (1, 12)

    def detach(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def tolist(self):
        return [[0.125] * 12]


def _signals(*, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        body_collision=False,
        wheel_only_climb=False,
        fall=False,
        nan_inf=False,
        hard_joint_limit=False,
        physics_explosion=False,
    )


def test_evaluate_steps_episode_directly_and_writes_canonical_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wlr50_clean.ppo import live_stream_writer, rl_library_wrapper

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"real checkpoint bytes")
    checkpoint_manifest = _best_validation_manifest(checkpoint)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial_frame = SimpleNamespace(
        sim_time_s=0.0,
        physics_tick=0,
        termination_signals=_signals(success=False),
    )
    final_frame = SimpleNamespace(
        sim_time_s=107.0,
        physics_tick=12840,
        termination_signals=_signals(success=True),
    )

    class FakeEpisode:
        def __init__(self) -> None:
            self.seed = 2001
            self.frame = initial_frame
            self.observation = (0.0,) * 125
            self.done = False
            self.decision_count = 0
            self.trace: list[dict[str, object]] = []
            self.tick_callback = None
            self.actions: list[tuple[float, ...]] = []

        def step(self, action):
            self.actions.append(tuple(action))
            assert self.tick_callback is not None
            self.tick_callback(self.frame, final_frame, object())
            self.frame = final_frame
            self.observation = (1.0,) * 125
            self.done = True
            self.decision_count = 1
            info = {
                "termination_reason": "SUCCESS",
                "recording_runtime_access_count": 0,
                "in_episode_root_write_count": 0,
            }
            self.trace.append({"termination_reason": "SUCCESS"})
            return SimpleNamespace(reward=3.5, info=info)

    episode = FakeEpisode()

    class FakeVecEnv:
        environments = (episode,)

        def __init__(self) -> None:
            self.observation_batches = []
            self.vector_step_calls = 0

        def observation_tensor_dict(self, rows):
            self.observation_batches.append(tuple(rows))
            return "tensor-dict"

        def step(self, actions):
            self.vector_step_calls += 1
            raise AssertionError("evaluation must not call the auto-reset vector step")

    vec_env = FakeVecEnv()
    runner = object()
    construction = {}

    def fake_construct(args, simulation_app, **kwargs):
        construction.update(kwargs)
        return object(), vec_env, runner, {}

    monkeypatch.setattr(cli, "_construct_live_runner", fake_construct)
    manifest_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        cli,
        "_current_checkpoint_runtime_contract",
        lambda args: {
            key: manifest_payload[key] for key in CHECKPOINT_RUNTIME_CONTRACT_FIELDS
        },
    )
    checkpoint_infos = {
        key: manifest_payload[key]
        for key in (
            "schema",
            "stage",
            "training_seed",
            "global_policy_decisions",
            *CHECKPOINT_RUNTIME_CONTRACT_FIELDS,
        )
    }
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_checkpoint_round_trip",
        lambda selected_runner, path: checkpoint_infos,
    )
    action_calls = []

    def fake_action(selected_runner, observations):
        action_calls.append((selected_runner, observations))
        return _FakeActions()

    monkeypatch.setattr(rl_library_wrapper, "deterministic_action", fake_action)

    writers = []

    class FakeWriter:
        def __init__(self, episode_dir, *, seed):
            self.episode_dir = Path(episode_dir)
            self.episode_dir.mkdir()
            self.seed = seed
            self.started = []
            self.ticks = []
            self.decisions = []
            self.finalized = []
            self.aborted = False
            writers.append(self)

        def start(self, frame):
            self.started.append(frame)

        def write_tick(self, *values):
            self.ticks.append(values)

        def write_decision(self, info):
            self.decisions.append(info)

        def finalize(self, frame, *, reward_total, decision_count):
            self.finalized.append((frame, reward_total, decision_count))
            path = self.episode_dir / "trial_manifest.json"
            path.write_text("{}\n", encoding="utf-8")
            return path

        def abort(self):
            self.aborted = True

    monkeypatch.setattr(live_stream_writer, "LiveStreamWriter", FakeWriter)
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        num_envs=1,
        episode_count=1,
        deterministic=True,
        seed=2001,
        training_config=cli.DEFAULT_TRAINING_CONFIG,
        interface_config=cli.DEFAULT_INTERFACE_CONFIG,
        run_dir=run_dir,
        maximum_duration_s=200.0,
    )

    assert cli._evaluate(args, object()) == 0
    assert construction == {
        "max_iterations": 1,
        "reset_seeds": (2001,),
        "collect_trace": True,
    }
    assert vec_env.vector_step_calls == 0
    assert len(vec_env.observation_batches) == 1
    assert action_calls == [(runner, "tensor-dict")]
    assert episode.actions == [(0.125,) * 12]
    assert writers[0].started == [initial_frame]
    assert len(writers[0].ticks) == 1
    assert writers[0].finalized == [(final_frame, 3.5, 1)]
    result = json.loads((run_dir / "checkpoint_evaluation.json").read_text())
    assert result["fresh_process_single_episode"] is True
    assert result["vec_env_step_called"] is False
    assert result["passed"] is True
    assert result["episodes"][0]["canonical_episode_dir"].endswith(
        "episode_000_seed_2001"
    )
    assert (run_dir / "episode_000_seed_2001" / "policy_trace.jsonl").is_file()


@pytest.mark.parametrize(
    ("episode_count", "num_envs", "deterministic", "message"),
    (
        (2, 1, True, "exactly one episode"),
        (1, 2, True, "num-envs=1"),
        (1, 1, False, "--deterministic"),
    ),
)
def test_evaluate_rejects_non_fresh_or_nondeterministic_contract_before_scene_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    episode_count: int,
    num_envs: int,
    deterministic: bool,
    message: str,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: pytest.fail("scene must not be constructed"),
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        num_envs=num_envs,
        episode_count=episode_count,
        deterministic=deterministic,
    )
    with pytest.raises(cli.CliError, match=message):
        cli._evaluate(args, object())


def test_evaluate_requires_explicit_checkpoint_manifest_before_scene_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: pytest.fail("scene must not be constructed"),
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_manifest=None,
        num_envs=1,
        episode_count=1,
        deterministic=True,
    )

    with pytest.raises(cli.CliError, match="explicit --checkpoint-manifest"):
        cli._evaluate(args, object())


def test_evaluate_rejects_empty_checkpoint_infos_before_episode_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wlr50_clean.ppo import live_stream_writer, rl_library_wrapper

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_manifest = _best_validation_manifest(checkpoint)
    episode = SimpleNamespace(step=lambda action: pytest.fail("episode must not step"))
    env = SimpleNamespace(environments=(episode,))
    monkeypatch.setattr(
        cli,
        "_construct_live_runner",
        lambda *args, **kwargs: (object(), env, object(), {}),
    )
    monkeypatch.setattr(
        rl_library_wrapper,
        "load_checkpoint_round_trip",
        lambda runner, path: {},
    )
    monkeypatch.setattr(
        live_stream_writer,
        "LiveStreamWriter",
        lambda *args, **kwargs: pytest.fail("writer must not be created"),
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        num_envs=1,
        episode_count=1,
        deterministic=True,
        seed=2001,
        training_config=cli.DEFAULT_TRAINING_CONFIG,
        interface_config=cli.DEFAULT_INTERFACE_CONFIG,
    )

    with pytest.raises(
        rl_library_wrapper.RlLibraryConfigurationError,
        match="infos have the wrong schema",
    ):
        cli._evaluate(args, object())


def _write_worker(
    root: Path,
    *,
    index: int,
    seed: int,
    checkpoint: Path,
    task_success: bool = True,
) -> Path:
    run_dir = root / f"worker_{index}"
    episode_dir = run_dir / f"episode_000_seed_{seed}"
    episode_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "lifecycle": "SUCCEEDED",
                "exit_code": 0,
                "configs": [
                    {"path": path, "sha256": _TEST_FROZEN_HASHES[field]}
                    for field, path in (
                        (
                            "observation_schema_hash",
                            "configs/ppo_observation_schema_v2.json",
                        ),
                        (
                            "action_schema_hash",
                            "configs/ppo_phase_action_masks_v2.yaml",
                        ),
                        ("reward_config_hash", "configs/ppo_reward_v2.yaml"),
                    )
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_audit = {
        "schema": "wlr50_clean.frozen_fsm_hash_audit.v1",
        "frozen_manifest_sha256": "f" * 64,
        "passed": True,
        "mismatches": [],
        "entries": [
            {
                "path": path,
                "expected_sha256": _TEST_FROZEN_HASHES[field],
                "actual_sha256": _TEST_FROZEN_HASHES[field],
                "exists": True,
                "valid": True,
            }
            for field, path in (
                ("controller_hash", "configs/fsm_states.yaml"),
                ("environment_hash", "configs/environment_lock.json"),
            )
        ],
    }
    for audit_name in ("frozen_hashes.before.json", "frozen_hashes.after.json"):
        (run_dir / audit_name).write_text(
            json.dumps(frozen_audit) + "\n", encoding="utf-8"
        )
    for name in CANONICAL_EPISODE_FILES:
        if name == "trial_manifest.json":
            payload = {
                "schema": "wlr50_clean.ppo_live_trial_manifest.v1",
                "seed": seed,
                "result": "SUCCESS" if task_success else "TASK_FAILURE",
                "success_evidence": {
                    "p01_p13_completed": task_success,
                    "body_collision": False,
                    "wheel_only_climb": False,
                    "duration_s": 12.0,
                },
                "action_projection_audit": {
                    "exact_pair_contact_contract_valid": True,
                },
            }
            (episode_dir / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
        else:
            (episode_dir / name).write_text("{}\n", encoding="utf-8")
    episode = {
        "seed": seed,
        "task_success": task_success,
        "duration_s": 12.0,
        "body_collision": False,
        "wheel_only_climb": False,
        "safety_abort": False,
        "under_maximum_duration": True,
        "recording_runtime_access_count": 0,
        "in_episode_root_write_count": 0,
        "canonical_episode_dir": str(episode_dir.resolve()),
    }
    evaluation = {
        "schema": "wlr50_clean.ppo_checkpoint_evaluation.v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": cli._sha256(checkpoint),
        "fresh_process_single_episode": True,
        "vec_env_step_called": False,
        "deterministic_mean_policy": True,
        "episode_count": 1,
        "passed": task_success,
        "episodes": [episode],
        "checkpoint_infos": dict(_TEST_FROZEN_HASHES),
    }
    (run_dir / "checkpoint_evaluation.json").write_text(
        json.dumps(evaluation) + "\n", encoding="utf-8"
    )
    return run_dir


def test_aggregate_evaluations_requires_and_summarizes_five_fresh_workers(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate checkpoint")
    workers = [
        _write_worker(tmp_path, index=index, seed=2001 + index, checkpoint=checkpoint)
        for index in range(5)
    ]
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    args = SimpleNamespace(
        checkpoint=checkpoint,
        evaluation_run_dir=workers,
        episode_count=5,
        run_dir=aggregate_dir,
        seed=2001,
        seed_set="validation",
        evaluation_role="candidate",
        deterministic=True,
    )

    assert cli._aggregate_evaluations(args) == 0
    payload = json.loads(
        (aggregate_dir / "checkpoint_evaluation_aggregate.json").read_text()
    )
    assert payload["passed"] is True
    assert payload["fresh_process_per_episode"] is True
    assert payload["seeds"] == [2001, 2002, 2003, 2004, 2005]
    assert len(payload["workers"]) == 5
    assert all(len(row["trial_manifest_sha256"]) == 64 for row in payload["workers"])


def test_aggregate_evaluations_keeps_valid_failure_evidence_and_fails_gate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint_best_validation.pt"
    checkpoint.write_bytes(b"candidate checkpoint")
    checkpoint_manifest = _best_validation_manifest(checkpoint)
    workers = [
        _write_worker(
            tmp_path,
            index=index,
            seed=3001 + index,
            checkpoint=checkpoint,
            task_success=index != 3,
        )
        for index in range(5)
    ]
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        evaluation_run_dir=workers,
        episode_count=5,
        run_dir=aggregate_dir,
        seed=3001,
        seed_set="locked-test",
        evaluation_role="candidate",
        deterministic=True,
    )

    assert cli._aggregate_evaluations(args) == 2
    payload = json.loads(
        (aggregate_dir / "checkpoint_evaluation_aggregate.json").read_text()
    )
    assert payload["passed"] is False
    assert payload["success_count"] == 4
    assert payload["finalized"] is True
    assert payload["worker_artifact_hashes_recomputed"] is True
    assert payload["checkpoint_manifest"] == str(checkpoint_manifest.resolve())
    assert all(payload["hash_gates"].values())
    assert all(
        row["all_artifact_hashes_recomputed"] is True for row in payload["workers"]
    )


def test_locked_test_aggregate_finalizes_all_promotion_hash_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint_best_validation.pt"
    checkpoint.write_bytes(b"validation-selected checkpoint")
    checkpoint_manifest = _best_validation_manifest(checkpoint)
    workers = [
        _write_worker(
            tmp_path,
            index=index,
            seed=3001 + index,
            checkpoint=checkpoint,
        )
        for index in range(5)
    ]
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    args = SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        evaluation_run_dir=workers,
        episode_count=5,
        run_dir=aggregate_dir,
        seed=3001,
        seed_set="locked-test",
        evaluation_role="candidate",
        deterministic=True,
    )

    assert cli._aggregate_evaluations(args) == 0
    payload = json.loads(
        (aggregate_dir / "checkpoint_evaluation_aggregate.json").read_text()
    )
    assert payload["finalized"] is True
    assert payload["passed"] is True
    assert payload["frozen_hashes_unchanged"] is True
    assert set(payload["hash_gates"]) == set(
        checkpoint_promotion.REQUIRED_LOCKED_TEST_HASH_GATES
    )
    assert all(payload["hash_gates"].values())
    assert all(
        len(row["frozen_hashes_before_sha256"]) == 64
        and len(row["frozen_hashes_after_sha256"]) == 64
        for row in payload["workers"]
    )


def test_evaluation_script_runs_one_process_per_episode_then_aggregates() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "evaluate_ppo_checkpoint.ps1"
    ).read_text(encoding="utf-8")
    assert 'for ($EpisodeIndex = 0; $EpisodeIndex -lt $EpisodeCount;' in script
    assert '"--episode-count", "1"' in script
    assert '-Subcommand "aggregate-evaluations"' in script
    assert '"--evaluation-run-dir"' in script
    assert "EnvironmentCount 1" in script
    assert "aggregate-evaluations" not in cli.LIVE_COMMANDS


def test_fresh_process_collector_supports_baseline_workers(tmp_path: Path) -> None:
    workers = []
    for index in range(5):
        seed = 2001 + index
        run_dir = tmp_path / f"baseline_{index}"
        episode_dir = run_dir / f"episode_000_seed_{seed}"
        episode_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"lifecycle": "SUCCEEDED", "exit_code": 0}) + "\n",
            encoding="utf-8",
        )
        for name in CANONICAL_EPISODE_FILES:
            if name == "trial_manifest.json":
                value = {
                    "seed": seed,
                    "action_projection_audit": {
                        "exact_pair_contact_contract_valid": True,
                    },
                }
                text = json.dumps(value) + "\n"
            elif name == "state_transitions.jsonl":
                text = ""
            else:
                text = "{}\n"
            (episode_dir / name).write_text(text, encoding="utf-8")
        episode = {
            "seed": seed,
            "task_success": True,
            "body_collision": False,
            "wheel_only_climb": False,
            "safety_abort": False,
            "under_maximum_duration": True,
            "recording_runtime_access_count": 0,
            "in_episode_root_write_count": 0,
            # Exercise compatibility with workers captured before the explicit
            # canonical_episode_dir field was added.
            "trial_manifest_path": str(episode_dir / "trial_manifest.json"),
        }
        (run_dir / "acceptance.json").write_text(
            json.dumps(
                {
                    "schema": "wlr50_clean.live_residual_gate.v1",
                    "mode": "zero",
                    "episode_count": 1,
                    "passed": True,
                    "mode_specific_checks": {
                        "zero_input_all_ticks_bitwise_equivalent": True,
                        "zero_fast_path_covers_every_tick": True,
                    },
                    "episodes": [episode],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        workers.append(run_dir)

    batch = collect_fresh_process_episode_workers(
        workers,
        seeds=range(2001, 2006),
        role="baseline",
    )
    assert batch.role == "baseline"
    assert batch.seeds == (2001, 2002, 2003, 2004, 2005)
    assert batch.canonical_episode_dirs == tuple(
        worker / f"episode_000_seed_{2001 + index}"
        for index, worker in enumerate(workers)
    )


def test_baseline_script_collects_five_workers_for_shared_aggregator() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_fsm_baseline_eval.ps1"
    ).read_text(encoding="utf-8")
    assert '"--evidence-only-worker"' in script
    assert '"--evaluation-role", "baseline"' in script
    assert '-Subcommand "aggregate-evaluations"' in script
    assert '"--evaluation-run-dir"' in script
    assert '-Subcommand "export-baseline-evaluation"' in script
    assert '"--episode-dir"' in script
    assert "export-baseline-evaluation" not in cli.LIVE_COMMANDS


def test_baseline_export_cli_is_offline_and_forwards_canonical_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wlr50_clean.ppo import evaluation_artifacts

    directories = [tmp_path / f"episode_{seed}" for seed in range(2001, 2006)]
    captured = {}

    def fake_export(output, **kwargs):
        captured["output"] = output
        captured.update(kwargs)
        return SimpleNamespace(
            as_dict=lambda: {
                "episode_metrics": str(Path(output) / "fsm_baseline_episode_metrics.csv"),
                "phase_metrics": str(Path(output) / "fsm_baseline_phase_metrics.csv"),
                "manifest": str(Path(output) / "fsm_baseline_evaluation_manifest.json"),
            }
        )

    monkeypatch.setattr(
        evaluation_artifacts, "export_baseline_evaluation_artifacts", fake_export
    )
    args = SimpleNamespace(
        episode_count=5,
        seed=2001,
        episode_dir=directories,
        metrics_output_dir=tmp_path / "metrics",
    )

    assert cli._export_baseline_evaluation(args) == 0
    assert captured["output"] == (tmp_path / "metrics").resolve()
    assert captured["episode_directories"] == tuple(
        path.resolve() for path in directories
    )
    assert tuple(captured["seeds"]) == (2001, 2002, 2003, 2004, 2005)
