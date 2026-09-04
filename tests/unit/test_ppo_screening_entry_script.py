from pathlib import Path


def test_screening_wrapper_is_single_episode_deterministic_and_nonpromoting() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "evaluate_ppo_checkpoint_screening.ps1").read_text(
        encoding="utf-8"
    )

    assert '"--episode-count", "1"' in text
    assert '"--seed-set", "validation"' in text
    assert '"--deterministic"' in text
    assert '-Subcommand "evaluate"' in text
    assert "promote-best-validation" not in text
    assert "promote-improved" not in text
