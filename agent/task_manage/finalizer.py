"""终态收口：归档 WorkingMemory、写错误原因、按配置裁剪失败内容。"""

from __future__ import annotations

from agent.common.enums import ContextScope, TaskStatus
from agent.config.settings import AppSettings
from agent.context_manage.context_manager import ContextManager
from agent.memory_manage.memory_manager import MemoryManager
from agent.models.context_window import ContextWindow
from agent.models.task import Task
from agent.models.task_summary import TaskSummary

__all__ = ["TaskFinalizer"]


class TaskFinalizer:
    def __init__(
        self,
        memory_manager: MemoryManager,
        context_manager: ContextManager,
        settings: AppSettings,
    ) -> None:
        self._memory = memory_manager
        self._context = context_manager
        self._settings = settings

    def finalize(self, task: Task, window: ContextWindow) -> None:
        if task.status is TaskStatus.FAILED or task.status is TaskStatus.CANCELED:
            self._write_error_output(task, window)
            window = self._context.load_window(task.task_id)
            archive_content = bool(getattr(self._settings.task, "archive_failed_content", False))
            self._archive(task, window, include_body=archive_content)
            return
        if task.status is TaskStatus.COMPLETED:
            self._archive(task, window, include_body=True)

    def _write_error_output(self, task: Task, window: ContextWindow) -> None:
        reason = ""
        if task.error is not None:
            reason = task.error.message
        output = window.summary.output or reason or task.status.value
        if output != window.summary.output:
            self._context.update_summary(task.task_id, output=output)
            payload = task.summary.to_dict()
            payload["output"] = output
            payload["status"] = task.status.value
            task._adopt(task.model_copy(update={"summary": TaskSummary.from_dict(payload)}))

    def _archive(self, task: Task, window: ContextWindow, *, include_body: bool) -> None:
        current = window.split_by_scope()[ContextScope.CURRENT]
        slim = ContextWindow(
            task_id=window.task_id,
            session_id=window.session_id,
            summary=task.summary,
            content=list(current.messages) if include_body else [],
            artifacts=dict(current.artifacts) if include_body else {},
        )
        working = self._memory.working(task.session_id)
        working.archive(slim)
        working.trim(self._settings.memory.working_memory_max_tasks)
