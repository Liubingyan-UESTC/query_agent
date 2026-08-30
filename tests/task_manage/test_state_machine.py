"""步骤 19：穷举转移表、非法转移报错、终态不可再转移。"""

from __future__ import annotations

import pytest

from agent.common.enums import TaskStatus
from agent.common.errors import TaskStateError
from agent.models.task import Task
from agent.task_manage.state_machine import LEGAL_TRANSITIONS, TaskStateMachine


def test_legal_pairs_are_exactly_the_table() -> None:
    machine = TaskStateMachine()
    for source in TaskStatus:
        for target in TaskStatus:
            expected = target in LEGAL_TRANSITIONS[source]
            assert machine.can(source, target) is expected, f"{source.value} → {target.value}"


def test_illegal_transition_raises() -> None:
    machine = TaskStateMachine()
    with pytest.raises(TaskStateError, match="非法状态转移"):
        machine.assert_transition(TaskStatus.CREATED, TaskStatus.COMPLETED)


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED])
def test_terminal_cannot_leave(status: TaskStatus) -> None:
    machine = TaskStateMachine()
    assert machine.is_terminal(status)
    with pytest.raises(TaskStateError, match="终态"):
        machine.assert_transition(status, TaskStatus.PLANNING)


def test_transition_updates_task_and_fires_hooks() -> None:
    machine = TaskStateMachine()
    seen: list[str] = []
    machine.on_exit(TaskStatus.CREATED, lambda task, status: seen.append(f"exit:{status.value}"))
    machine.on_enter(
        TaskStatus.INTENDING, lambda task, status: seen.append(f"enter:{status.value}")
    )
    task = Task.create("q", session_id="sess_1")
    machine.transition(task, TaskStatus.INTENDING, note="start")

    assert task.status is TaskStatus.INTENDING
    assert task.summary.status is TaskStatus.INTENDING
    assert seen == ["exit:created", "enter:intending"]
    assert task.status_history[-1].note == "start"


def test_same_status_is_illegal_except_executing() -> None:
    machine = TaskStateMachine()
    task = Task.create("q", session_id="sess_1")
    machine.transition(task, TaskStatus.INTENDING)
    with pytest.raises(TaskStateError, match="原地停留"):
        machine.transition(task, TaskStatus.INTENDING)

    for status in (
        TaskStatus.PLANNING,
        TaskStatus.EXECUTING,
    ):
        machine.transition(task, status)
    machine.transition(task, TaskStatus.EXECUTING)
    assert task.status is TaskStatus.EXECUTING
