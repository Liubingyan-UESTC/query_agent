"""阶段处理器基类：统一输入输出、耗时埋点、异常兜底与取消检查。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.common.enums import TaskStatus
from agent.common.errors import AgentError, TaskCanceledError
from agent.common.logging import get_logger
from agent.config.settings import AppSettings
from agent.context_manage.context_manager import ContextManager
from agent.llm.base import BaseLLMClient
from agent.memory_manage.memory_manager import MemoryManager
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.error import ErrorInfo
from agent.models.message import Message
from agent.models.task import Task
from agent.prompt.assembler import PromptAssembler
from agent.tool_manage.manager import ToolManager

__all__ = ["BaseStage", "StageDeps", "StageResult", "StageRuntime"]

logger = get_logger(__name__)


@dataclass
class StageRuntime:
    """单次 `TaskManager.run` 的可变护栏。"""

    started_at: float
    tool_calls: int = 0
    canceled: bool = False
    cancel_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    next_status: TaskStatus
    updated_fields: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    error: ErrorInfo | None = None
    should_continue: bool = True
    elapsed_ms: float = 0.0
    note: str | None = None

    @classmethod
    def failed(
        cls,
        error: AgentError | ErrorInfo,
        *,
        note: str | None = None,
        updated_fields: dict[str, Any] | None = None,
    ) -> StageResult:
        info = error if isinstance(error, ErrorInfo) else ErrorInfo.from_error(error)
        return cls(
            next_status=TaskStatus.FAILED,
            error=info,
            should_continue=False,
            note=note or info.message,
            updated_fields=updated_fields or {},
        )

    @classmethod
    def canceled(cls, reason: str | None = None) -> StageResult:
        return cls(
            next_status=TaskStatus.CANCELED,
            error=ErrorInfo(code="task_canceled", message=reason or "任务已取消"),
            should_continue=False,
            note=reason or "任务已取消",
        )


@dataclass
class StageDeps:
    llm: BaseLLMClient
    context_manager: ContextManager
    memory_manager: MemoryManager
    tool_manager: ToolManager
    prompt_assembler: PromptAssembler
    settings: AppSettings
    runtime: StageRuntime
    injector: Callable[[str, list[str]], Any] | None = None


class BaseStage(ABC):
    """TaskManager 只调度本接口。子类实现 `_run`。"""

    stage_name: str

    def run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
        started = time.perf_counter()
        if deps.runtime.canceled:
            return StageResult.canceled(deps.runtime.cancel_reason)
        try:
            _check_duration(deps)
            result = self._run(task, window, deps)
        except TaskCanceledError as exc:
            result = StageResult.canceled(exc.message)
        except AgentError as exc:
            logger.exception(
                "stage_failed", extra={"stage": self.stage_name, "task_id": task.task_id}
            )
            result = StageResult.failed(exc)
        except Exception as exc:
            logger.exception(
                "stage_crashed", extra={"stage": self.stage_name, "task_id": task.task_id}
            )
            result = StageResult.failed(
                AgentError(f"阶段 {self.stage_name} 异常：{exc}", detail=repr(exc))
            )
        object.__setattr__(result, "elapsed_ms", (time.perf_counter() - started) * 1000)
        logger.info(
            "stage_finished",
            extra={
                "stage": self.stage_name,
                "next_status": result.next_status.value,
                "elapsed_ms": result.elapsed_ms,
                "task_id": task.task_id,
            },
        )
        return result

    @abstractmethod
    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
        raise NotImplementedError


def _check_duration(deps: StageDeps) -> None:
    elapsed = time.monotonic() - deps.runtime.started_at
    limit = deps.settings.task.max_duration_seconds
    if elapsed > limit:
        raise AgentError(f"任务超过总时长上限 {limit}s（已用 {elapsed:.1f}s）")
