"""Terminal classifications for the clean traversal controller.

The two task failures are intentionally explicit.  Controller, safety,
infrastructure, and evidence-production failures must never be folded into a
generic traversal failure because they answer different validation questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class TaskResult(str, Enum):
    SUCCESS = "SUCCESS"
    TASK_FAILURE_BODY_COLLISION = "TASK_FAILURE_BODY_COLLISION"
    TASK_FAILURE_WHEEL_ONLY_CLIMB = "TASK_FAILURE_WHEEL_ONLY_CLIMB"
    INCOMPLETE_CONTROLLER_BLOCKED = "INCOMPLETE_CONTROLLER_BLOCKED"
    SAFETY_ABORT = "SAFETY_ABORT"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    VIDEO_OR_ARTIFACT_ERROR = "VIDEO_OR_ARTIFACT_ERROR"


TASK_FAILURE_RESULTS = frozenset(
    {
        TaskResult.TASK_FAILURE_BODY_COLLISION,
        TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB,
    }
)


@dataclass(frozen=True)
class TaskTermination:
    result: TaskResult
    state_id: str
    lifecycle: str
    sim_time_s: float
    reason: str
    details: Mapping[str, Any]

    @property
    def is_success(self) -> bool:
        return self.result is TaskResult.SUCCESS

    @property
    def is_task_failure(self) -> bool:
        return self.result in TASK_FAILURE_RESULTS

