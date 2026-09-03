from __future__ import annotations

import json

import pytest
import pyarrow.parquet as pq

from wlr50_clean.ppo.episode_logger import (
    EpisodeLogError,
    EpisodeLogger,
    EpisodeTransition,
)


ZERO12 = (0.0,) * 12
ZERO85 = (0.0,) * 85


def _transition(
    *,
    nominal=ZERO12,
    residual=ZERO12,
    applied=ZERO12,
) -> EpisodeTransition:
    return EpisodeTransition(
        episode_id="baseline-episode",
        trial_id="trial_test",
        seed=7,
        control_tick=0,
        sim_time=0.0,
        state_id="P01",
        macro_phase=1,
        phase_progress=0.0,
        observation_t=ZERO85,
        nominal_action_t=nominal,
        residual_action_t=residual,
        applied_action_t=applied,
        action_mask_t=(0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1),
        task_result="SUCCESS",
        reward_components_t={"task_success": 1.0, "forward_progress": 0.0},
        terminated=True,
        truncated=False,
        termination_reason="SUCCESS",
        observation_t_plus_1=ZERO85,
        environment_hash="environment-sha256",
        controller_hash="controller-sha256",
        motion_contract_hash="contract-sha256",
        observation_schema_version="wlr50_clean.actor_observation.v1",
        action_schema_version="wlr50_clean.residual_full12.v1",
    )


def test_logger_derives_zero_residual_bitwise_equivalence_itself() -> None:
    nominal = (-0.0, 1.0) + ZERO12[2:]
    logger = EpisodeLogger()
    logger.append(_transition(nominal=nominal, applied=nominal))
    evidence = logger.validate_baseline_equivalence()
    assert evidence.status == "ZERO_RESIDUAL_FULL_EPISODE_EQUIVALENCE"
    assert evidence.residual_action_all_zero is True
    assert evidence.applied_action_bitwise_equal_nominal is True
    assert evidence.nominal_sequence_sha256 == evidence.applied_sequence_sha256


def test_logger_rejects_nonzero_residual_before_claiming_baseline() -> None:
    logger = EpisodeLogger()
    logger.append(_transition(residual=(0.01,) + ZERO12[1:]))
    with pytest.raises(EpisodeLogError, match="non-zero residual"):
        logger.validate_baseline_equivalence()


def test_logger_rejects_numerically_equal_but_not_bitwise_equal_action() -> None:
    logger = EpisodeLogger()
    logger.append(
        _transition(
            nominal=(-0.0,) + ZERO12[1:],
            applied=(0.0,) + ZERO12[1:],
        )
    )
    with pytest.raises(EpisodeLogError, match="bitwise equal"):
        logger.validate_baseline_equivalence()


def test_logger_rejects_false_caller_asserted_equivalence_hash(tmp_path) -> None:
    logger = EpisodeLogger()
    logger.append(_transition())
    with pytest.raises(EpisodeLogError, match="caller-asserted"):
        logger.write_manifest(
            tmp_path / "manifest.json",
            source_trial="trial_test",
            jsonl_path=tmp_path / "missing.jsonl",
            parquet_path=tmp_path / "missing.parquet",
            claimed_zero_residual_equivalence_sha256="false-claim",
        )


def test_parquet_round_trip_has_typed_fixed_size_vectors(tmp_path) -> None:
    logger = EpisodeLogger()
    logger.append(_transition())
    jsonl_path = logger.write_jsonl(tmp_path / "baseline.jsonl")
    parquet_path = logger.write_parquet(tmp_path / "baseline.parquet")
    table = pq.read_table(parquet_path)
    assert table.num_rows == 1
    assert table.schema.field("observation_t").type.list_size == 85
    assert table.schema.field("nominal_action_t").type.list_size == 12
    assert table.schema.field("residual_action_t").type.list_size == 12
    assert table.schema.field("applied_action_t").type.list_size == 12
    assert table.schema.field("action_mask_t").type.list_size == 12
    row = table.to_pylist()[0]
    assert row["trial_id"] == "trial_test"
    assert row["task_result"] == "SUCCESS"
    assert row["residual_action_t"] == list(ZERO12)

    evidence = logger.validate_baseline_equivalence()
    manifest_path = logger.write_manifest(
        tmp_path / "manifest.json",
        source_trial="trial_test",
        jsonl_path=jsonl_path,
        parquet_path=parquet_path,
        claimed_zero_residual_equivalence_sha256=(
            evidence.nominal_sequence_sha256
        ),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proof = manifest["zero_residual_full_episode_equivalence"]
    assert proof["applied_action_bitwise_equal_nominal"] is True
    assert proof["nominal_sequence_sha256"] == proof["applied_sequence_sha256"]
