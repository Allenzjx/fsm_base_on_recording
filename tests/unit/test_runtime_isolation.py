import json
from pathlib import Path

from tools.verify_clean_room import load_manifest, scan_text, verify_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_clean_room_scan_and_hash_manifest() -> None:
    assert scan_text() == []
    assert verify_manifest(load_manifest()) == []


def test_production_sources_do_not_open_raw_reference_data() -> None:
    runtime_roots = [
        ROOT / "src" / "wlr50_clean" / name
        for name in ("fsm", "sensing", "infrastructure", "ppo")
    ]
    forbidden = (
        "accepted_steps.jsonl",
        "semantic_segments.json",
        "recording_clean.mp4",
        "recording_cursor",
    )
    for runtime_root in runtime_roots:
        for path in runtime_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert all(value not in text for value in forbidden), path


def test_selected_reference_forbids_runtime_raw_access() -> None:
    selected = json.loads(
        (ROOT / "configs" / "selected_reference.json").read_text(encoding="utf-8")
    )
    assert selected["reference_version"] == "v010"
    assert selected["rear_leg_order"] == "RR_FIRST"
    assert selected["cross_version_splice"] is False
    assert selected["runtime_recording_access_authorized"] is False
