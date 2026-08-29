"""Task 实体：一次用户请求的生命周期载体。

本模块只提供数据访问：字段、序列化、`touch()`、`record_status()`。
**状态转移是否合法不在此处裁定**——那是步骤 19 `TaskStateMachine` 的职责。
若把转移规则写进模型，状态机与模型会各持一份规则，新增状态时必然漏改。

`Task.status`、`summary.status`、`status_history` 三处必须同步。调用方若各改各的，
归档后会出现「实体已完成、摘要仍显示 intending」的撕裂，而前端读的是 summary。
同步只允许走 `record_status()`：

- 构造期与 `Task.status = ...` 由 `model_validator` 拦住不一致；
- `summary.status` 字段冻结，直接赋值会失败；
- `record_status()` 用已校验副本整体覆盖自身，避免分步赋值经过中间撕裂态。

它只记账，不 prescreen 转移方向，`CREATED → COMPLETED` 在本层完全合法。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from agent.common.enums import TaskStatus
from agent.common.errors import AgentError
from agent.common.ids import new_task_id, new_trace_id
from agent.models.base import AgentModel
from agent.models.error import ErrorInfo
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
    error: ErrorInfo | None = None
    retry_count: int = 0
    status_history: list[StatusRecord] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        # 构造完成后再禁止直接赋 status。Pydantic 的 after 校验器是「先写入再校验」，
        # 失败时新值已经落在字段上，拦不住撕裂。
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "status" and self.__dict__.get("_sealed"):
            raise ValueError("Task.status 不能直接赋值；请改用 record_status() 同步三处状态")
        super().__setattr__(name, value)

    @field_validator("retry_count")
    @classmethod
    def _retry_count_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"retry_count 必须 >= 0，实际为 {value}")
        return value

    @field_validator("error", mode="before")
    @classmethod
    def _coerce_error(cls, value: Any) -> Any:
        if value is None or isinstance(value, ErrorInfo):
            return value
        if isinstance(value, (AgentError, dict)):
            return ErrorInfo.coerce(value)
        return value

    @model_validator(mode="after")
    def _summary_is_consistent(self) -> "Task":
        if self.summary.task_id != self.task_id:
            raise ValueError(
                f"Task.task_id={self.task_id!r} 与 summary.task_id={self.summary.task_id!r} 不一致"
            )
        if self.summary.status is not self.status:
            raise ValueError(
                f"Task.status={self.status.value!r} 与 summary.status="
                f"{self.summary.status.value!r} 不一致；请改用 record_status() 同步三处状态"
            )
        return self

    @classmethod
    def create(
        cls,
        content: str,
        *,
        session_id: str,
        task_id: str | None = None,
        trace_id: str | None = None,
    ) -> "Task":
        """创建任务及其摘要，二者共享 `task_id`，状态均为 `CREATED`。

        `session_id` 必填：漏传时若静默 `new_session_id()`，每个请求都会变成独立会话，
        WorkingMemory 完全失效且全程无报错。需要临时会话的调用方显式传入
        `session_id=new_session_id()`。
        """
        resolved_id = task_id or new_task_id()
        now = datetime.now(UTC)
        return cls(
            task_id=resolved_id,
            session_id=session_id,
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
        updated = self.model_copy(
            update={
                "status": status,
                "summary": self.summary.model_copy(update={"status": status}),
                "status_history": [
                    *self.status_history,
                    StatusRecord(status=status, at=now, note=note),
                ],
                "updated_at": now,
            }
        )
        self._adopt(updated)

    def _adopt(self, other: "Task") -> None:
        """用已校验副本覆盖自身，避免分步赋值经过「只改了一处」的中间态。"""
        object.__setattr__(self, "__dict__", dict(other.__dict__))
        object.__setattr__(self, "__pydantic_fields_set__", set(other.__pydantic_fields_set__))
        object.__setattr__(self, "__pydantic_extra__", getattr(other, "__pydantic_extra__", None))
        object.__setattr__(
            self, "__pydantic_private__", getattr(other, "__pydantic_private__", None)
        )
