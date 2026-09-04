from __future__ import annotations

import copy

import pytest
import yaml

from wlr50_clean.ppo.phase_objectives import DENSE_FAMILIES
from wlr50_clean.ppo.reward_migration import (
    RewardMigrationError,
    load_reward_migration,
)
from wlr50_clean.ppo.reward_terms import REWARD_TERMS


def test_reward_migration_binds_every_v1_concept_to_compact_v2(tmp_path):
    evidence = load_reward_migration()
    payload = evidence.as_dict()

    assert tuple(entry.v1_concept for entry in evidence.entries) == REWARD_TERMS
    assert tuple(payload["active_v2_dense_families"]) == DENSE_FAMILIES
    assert payload["legacy_source_preserved_and_inactive"] is True
    assert payload["all_v1_concepts_explicitly_disposed"] is True
    assert payload["v1_concept_count"] == 13
    assert len(payload["source_sha256"]) == 64
    assert len(payload["target_sha256"]) == 64
    by_concept = {entry.v1_concept: entry for entry in evidence.entries}
    for concept in ("body_collision_penalty", "wheel_only_climb_penalty"):
        assert "event.task_failure" in by_concept[concept].v2_destinations
        assert "event.safety_abort" not in by_concept[concept].v2_destinations
    assert "event.safety_abort" in by_concept["fall_penalty"].v2_destinations


def test_reward_migration_rejects_missing_or_reordered_concept(tmp_path):
    original = load_reward_migration().path
    payload = yaml.safe_load(original.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["migrations"][0], changed["migrations"][1] = (
        changed["migrations"][1],
        changed["migrations"][0],
    )
    path = tmp_path / "migration.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")

    with pytest.raises(RewardMigrationError, match="concept order"):
        load_reward_migration(path)


def test_reward_migration_rejects_unknown_destination(tmp_path):
    original = load_reward_migration().path
    payload = yaml.safe_load(original.read_text(encoding="utf-8"))
    payload["migrations"][0]["v2_destinations"] = ["dense.survival"]
    path = tmp_path / "migration.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RewardMigrationError, match="unknown v2 destination"):
        load_reward_migration(path)
