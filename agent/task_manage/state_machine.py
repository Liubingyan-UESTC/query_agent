"""任务状态机：合法转移表与进出钩子。

转移规则只在这里裁定。`Task.record_status()` 只记账，不 prescreen 方向。
`VALIDATING → PLANNING` 用于校验不通过的重规划；次数上限由阶段处理器按
`TASK_MAX_REPLAN` 拦截，状态机本身只保证转移合法。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent.common.enums import TERMINAL_TASK_STATUSES, TaskStatus
from agent.common.errors import TaskStateError

if TYPE_CHECKING:
    from agent.models.task import Task

__all__ = ["LEGAL_TRANSITIONS", "TaskStateMachine"]

TransitionHook = Callable[["Task", TaskStatus], None]

LEGAL_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.INTENDING, TaskStatus.CANCELED, TaskStatus.FAILED}),
    TaskStatus.INTENDING: frozenset({TaskStatus.PLANNING, TaskStatus.FAILED, TaskStatus.CANCELED}),
    TaskStatus.PLANNING: frozenset(
        {TaskStatus.EXECUTING, TaskStatus.WAITING_USER, TaskStatus.FAILED, TaskStatus.CANCELED}
    ),
    TaskStatus.EXECUTING: frozenset(
        {
            TaskStatus.EXECUTING,
            TaskStatus.VALIDATING,
            TaskStatus.RETRYING,
            TaskStatus.WAITING_USER,
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        }
    ),
    TaskStatus.RETRYING: frozenset({TaskStatus.EXECUTING, TaskStatus.FAILED, TaskStatus.CANCELED}),
    TaskStatus.WAITING_USER: frozenset(
        {TaskStatus.EXECUTING, TaskStatus.PLANNING, TaskStatus.CANCELED, TaskStatus.FAILED}
    ),
    TaskStatus.VALIDATING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.PLANNING, TaskStatus.FAILED, TaskStatus.CANCELED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELED: frozenset(),
}


class TaskStateMachine:
    """可穷举、可测试的转移表。钩子供步骤 31 事件流复用。"""

    def __init__(self) -> None:
        self._on_enter: dict[TaskStatus, list[TransitionHook]] = defaultdict(list)
        self._on_exit: dict[TaskStatus, list[TransitionHook]] = defaultdict(list)

    def can(self, source: TaskStatus, target: TaskStatus) -> bool:
        return target in LEGAL_TRANSITIONS.get(source, frozenset())

    def is_terminal(self, status: TaskStatus) -> bool:
        return status in TERMINAL_TASK_STATUSES

    def assert_transition(self, source: TaskStatus, target: TaskStatus) -> None:
        if self.is_terminal(source):
            raise TaskStateError(f"终态 {source.value} 不可再转移")
        if not self.can(source, target):
            allowed = sorted(item.value for item in LEGAL_TRANSITIONS.get(source, frozenset()))
            raise TaskStateError(f"非法状态转移：{source.value} → {target.value}；允许：{allowed}")

    def on_enter(self, status: TaskStatus, hook: TransitionHook) -> None:
        self._on_enter[status].append(hook)

    def on_exit(self, status: TaskStatus, hook: TransitionHook) -> None:
        self._on_exit[status].append(hook)

    def transition(self, task: Task, target: TaskStatus, *, note: str | None = None) -> None:
        source = task.status
        if source is target and source is not TaskStatus.EXECUTING:
            raise TaskStateError(f"不能原地停留在 {source.value}")
        self.assert_transition(source, target)
        for hook in self._on_exit[source]:
            hook(task, source)
        task.record_status(target, note=note)
        for hook in self._on_enter[target]:
            hook(task, target)
