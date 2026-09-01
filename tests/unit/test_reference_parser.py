from pathlib import Path

from wlr50_clean.reference.recording_parser import load_recording, validate_v010


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_v010_parser_and_binding() -> None:
    reference = ROOT / "reference" / "v010"
    result = validate_v010(
        reference / "accepted_steps.jsonl",
        reference / "metadata.json",
        expected_sha256="f962128da9e9551235a6f6769308eed0c947657fe804c9e0e26025f456c72e92",
    )
    assert result["passed"] is True
    assert result["step_count"] == 26
    assert result["event_count"] == 168
    parsed = load_recording(reference / "accepted_steps.jsonl")
    assert [step.index for step in parsed.steps] == list(range(1, 27))


def test_v010_contains_exactly_four_full12_launch_events() -> None:
    parsed = load_recording(ROOT / "reference" / "v010" / "accepted_steps.jsonl")
    events = [event for event in parsed.events if event.kind == "servo_wheel_launch"]
    assert [(event.step_index, event.event_index) for event in events] == [
        (4, 0),
        (12, 0),
        (18, 0),
        (26, 0),
    ]
