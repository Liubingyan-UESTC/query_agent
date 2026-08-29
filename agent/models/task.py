"""Task 实体：一次用户请求的生命周期载体。

本模块只提供数据访问：字段、序列化、`touch()`、`record_status()`。
**状态转移是否合法不在此处裁定**——那是步骤 19 `TaskStateMachine` 的职责。
若把转移规则写进模型，状态机与模型会各持一份规则，新增状态时必然漏改。

`record_status()` 仍有存在必要：`Task.status`、`summary.status`、`status_history`
三处必须同步，调用方若各改各的，归档后会出现「实体已完成、摘要仍显示 intending」
的撕裂。它只记账，不 prescreen 转移方向，`CREATED → COMPLETED` 在本层完全合法。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from agent.common.enums import TaskStatus
from agent.common.ids import new_session_id, new_task_id, new_trace_id
from agent.models.base import AgentModel
from agent.models.task_summary import TaskSummary

__all__ = ["StatusRecord", "Task"]


class StatusRecord(AgentModel):
    """`Task.status_history` 的一条记录。只记「变成了什么」，不记转移是否合法。"""

    status: TaskStatus
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str | None = None


class Task(AgentModel):
    """一次用户请求。"""

    task_id: str = Field(default_factory=new_task_id)
    session_id: str
    status: TaskStatus = TaskStatus.CREATED
    summary: TaskSummary
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str = Field(default_factory=new_trace_id)
    error: dict[str, Any] | None = None
    retry_count: int = 0
    status_history: list[StatusRecord] = Field(default_factory=list)

    @field_validator("retry_count")
    @classmethod
    def _retry_count_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"retry_count 必须 >= 0，实际为 {value}")
        return value

    @model_validator(mode="after")
    def _task_id_matches_summary(self) -> "Task":
        if self.summary.task_id != self.task_id:
            raise ValueError(
                f"Task.task_id={self.task_id!r} 与 summary.task_id={self.summary.task_id!r} 不一致"
            )
        return self

    @classmethod
    def create(
        cls,
        content: str,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        trace_id: str | None = None,
    ) -> "Task":
        """创建任务及其摘要，二者共享 `task_id`，状态均为 `CREATED`。

        这是任务出生的唯一推荐入口：手工拼 `Task` + `TaskSummary` 很容易把
        两个 id 写岔，而错误要到归档或按 id 反查时才暴露。
        """
        resolved_id = task_id or new_task_id()
        now = datetime.now(UTC)
        return cls(
            task_id=resolved_id,
            session_id=session_id or new_session_id(),
            status=TaskStatus.CREATED,
            summary=TaskSummary(task_id=resolved_id, content=content, status=TaskStatus.CREATED),
            created_at=now,
            updated_at=now,
            trace_id=trace_id or new_trace_id(),
            status_history=[StatusRecord(status=TaskStatus.CREATED, at=now)],
        )

    def touch(self) -> None:
        """刷新 `updated_at`。任何就地修改摘要/产物后都应调用，供轮询接口判断新鲜度。"""
        self.updated_at = datetime.now(UTC)

    def record_status(self, status: TaskStatus, *, note: str | None = None) -> None:
        """同步实体状态、摘要状态与历史记录。不做转移合法性校验。"""
        now = datetime.now(UTC)
        self.status = status
        self.summary.status = status
        self.status_history = [
            *self.status_history,
            StatusRecord(status=status, at=now, note=note),
        ]
        self.updated_at = now
