"""The fixed, linear P01--P13 graph and its per-state lifecycle cursor."""

from __future__ import annotations

from dataclasses import dataclass

from .state_spec import EXPECTED_STATE_IDS, FsmSpec, Lifecycle, StateSpec


@dataclass
class StateCursor:
    state_id: str = EXPECTED_STATE_IDS[0]
    lifecycle: Lifecycle = Lifecycle.WAIT_ENTRY
    retries_used: int = 0


class StateGraph:
    def __init__(self, spec: FsmSpec) -> None:
        self._spec = spec
        self._by_id = {state.state_id: state for state in spec.states}

    @property
    def first(self) -> StateSpec:
        return self._spec.states[0]

    def state(self, state_id: str) -> StateSpec:
        return self._by_id[state_id]

    def next(self, state_id: str) -> StateSpec | None:
        next_id = self.state(state_id).next_state
        return None if next_id == "TASK_COMPLETE" else self.state(next_id)

    def validate_transition(self, source: str, destination: str | None) -> None:
        expected = self.next(source)
        expected_id = expected.state_id if expected else None
        if destination != expected_id:
            raise ValueError(
                f"illegal state transition {source}->{destination}; "
                f"expected {expected_id}"
            )

