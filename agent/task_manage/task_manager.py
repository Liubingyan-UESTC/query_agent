"""TaskManager：建任务、按状态机调度阶段、终态收口。"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

from pydantic import ValidationError

from agent.common.enums import PromptStage, TaskStatus
from agent.common.errors import ContextError, TaskStateError
from agent.common.logging import get_logger, log_context
from agent.config.settings import AppSettings
from agent.context_manage.context_manager import ContextManager
from agent.context_manage.related_injector import RelatedContextInjector
from agent.memory_manage.memory_manager import MemoryManager
from agent.models.error import ErrorInfo
from agent.models.task import Task
from agent.models.task_summary import TaskSummary
from agent.store.base import Store
from agent.store.keys import make_key
from agent.task_manage.finalizer import TaskFinalizer
from agent.task_manage.stages.base import BaseStage, StageDeps, StageResult, StageRuntime
from agent.task_manage.stages.execute import ExecuteStage
from agent.task_manage.stages.intent import IntentStage
from agent.task_manage.stages.plan import PlanStage
from agent.task_manage.stages.validate import ValidateStage
from agent.task_manage.state_machine import TaskStateMachine

__all__ = ["NS_TASK", "NS_TASK_INDEX", "TaskManager"]

logger = get_logger(__name__)

NS_TASK = "tasks"
NS_TASK_INDEX = "task_index"
INDEX_SESSION = "global"


class TaskManager:
    def __init__(
        self,
        *,
        store: Store,
        settings: AppSettings,
        context_manager: ContextManager,
        memory_manager: MemoryManager,
        tool_manager: Any,
        prompt_assembler: Any,
        llm: Any,
        state_machine: TaskStateMachine | None = None,
        stages: dict[TaskStatus, BaseStage] | None = None,
        finalizer: TaskFinalizer | None = None,
    ) -> None:
        self.settings = settings
        self.context = context_manager
        self.memory = memory_manager
        self.tools = tool_manager
        self.assembler = prompt_assembler
        self.llm = llm
        self._store = store
        self.machine = state_machine or TaskStateMachine()
        self.stages = stages or {
            TaskStatus.INTENDING: IntentStage(),
            TaskStatus.PLANNING: PlanStage(),
            TaskStatus.EXECUTING: ExecuteStage(),
            TaskStatus.RETRYING: ExecuteStage(),
            TaskStatus.VALIDATING: ValidateStage(),
        }
        self.finalizer = finalizer or TaskFinalizer(self.memory, self.context, self.settings)
        self._runtimes: dict[str, StageRuntime] = {}

    def create_task(self, session_id: str, query: str, *, task_id: str | None = None) -> Task:
        task = Task.create(query, session_id=session_id, task_id=task_id)
        prompt = self.memory.knowledge.get_system_prompt(PromptStage.INTENT_RECOGNITION)
        self.context.create_window(task.task_id, session_id, query, prompt)
        self._save_task(task)
        self._runtimes[task.task_id] = StageRuntime(started_at=time.monotonic())
        logger.info("task_created", extra={"task_id": task.task_id, "session_id": session_id})
        return task

    def run(self, task_id: str) -> Task:
        task = self.get_task(task_id)
        with log_context(task_id=task.task_id, session_id=task.session_id, trace_id=task.trace_id):
            while not task.status.is_terminal:
                if task.status is TaskStatus.WAITING_USER:
                    break
                task = self.step(task_id)
        return task

    def step(self, task_id: str) -> Task:
        task = self.get_task(task_id)
        runtime = self._runtime(task)
        with log_context(task_id=task.task_id, session_id=task.session_id, trace_id=task.trace_id):
            if runtime.canceled:
                return self._fail_or_cancel(task, TaskStatus.CANCELED, runtime.cancel_reason)
            if time.monotonic() - runtime.started_at > self.settings.task.max_duration_seconds:
                return self._fail_or_cancel(task, TaskStatus.FAILED, "任务超过总时长上限")
            if task.status.is_terminal:
                raise TaskStateError(f"终态 {task.status.value} 不可再推进")
            if task.status is TaskStatus.WAITING_USER:
                return task
            if task.status is TaskStatus.CREATED:
                self._transition(task, TaskStatus.INTENDING)
                task = self.get_task(task_id)
            if task.status is TaskStatus.RETRYING:
                self._transition(task, TaskStatus.EXECUTING, note="同步骤重试")
                task = self.get_task(task_id)
            stage = self.stages.get(task.status)
            if stage is None:
                raise TaskStateError(f"状态 {task.status.value} 没有对应阶段处理器")
            window = self.context.load_window(task_id)
            result = stage.run(task, window, self._deps_for(task, runtime))
            return self._apply(task, result)

    def cancel(self, task_id: str, reason: str = "用户取消") -> Task:
        runtime = self._runtime(self.get_task(task_id))
        runtime.canceled = True
        runtime.cancel_reason = reason
        task = self.get_task(task_id)
        if not task.status.is_terminal:
            return self._fail_or_cancel(task, TaskStatus.CANCELED, reason)
        return task

    def get_task(self, task_id: str) -> Task:
        session_id = self._index_get(task_id)
        if session_id is None:
            raise TaskStateError(f"没有任务 {task_id}")
        payload = self._require_store().get(self._task_key(session_id, task_id))
        if not isinstance(payload, dict):
            raise TaskStateError(f"任务 {task_id} 载荷缺失或损坏")
        try:
            return Task.from_dict(payload)
        except ValidationError as exc:
            raise TaskStateError(f"任务 {task_id} 无法还原", detail=str(exc)) from exc

    def get_result(self, task_id: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        return {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "status": task.status.value,
            "intent": task.summary.intent.value if task.summary.intent else None,
            "related_task_ids": list(task.summary.related_task_ids),
            "output": task.summary.output,
            "result": task.summary.result,
            "operations": [item.to_dict() for item in task.summary.operations],
            "error": task.error.to_dict() if task.error else None,
        }

    def _apply(self, task: Task, result: StageResult) -> Task:
        self._sync_from_window(task)
        self._merge_fields(task, result.updated_fields)
        if result.error is not None and result.next_status is TaskStatus.FAILED:
            task.error = result.error
        if result.messages:
            for message in result.messages:
                try:
                    self.context.append_message(task.task_id, message)
                except ContextError:
                    continue
        if result.artifacts:
            for artifact in result.artifacts:
                try:
                    self.context.add_artifact(task.task_id, artifact)
                except ContextError:
                    continue
        if task.status is not result.next_status:
            self._transition(task, result.next_status, note=result.note)
        else:
            self._save_task(task)
            self._bind_window(task)
        task = self.get_task(task.task_id)
        if task.status.is_terminal:
            window = self.context.load_window(task.task_id)
            self.finalizer.finalize(task, window)
            self._save_task(task)
            self._bind_window(task)
            task = self.get_task(task.task_id)
        return task

    def _fail_or_cancel(self, task: Task, status: TaskStatus, reason: str | None) -> Task:
        info = ErrorInfo(
            code="task_canceled" if status is TaskStatus.CANCELED else "agent_error",
            message=reason or status.value,
        )
        task.error = info
        if task.status is not status:
            try:
                self._transition(task, status, note=reason)
            except TaskStateError:
                task.record_status(status, note=reason)
                self._save_task(task)
        task = self.get_task(task.task_id)
        window = self.context.load_window(task.task_id)
        self.finalizer.finalize(task, window)
        self._save_task(task)
        self._bind_window(task)
        return self.get_task(task.task_id)

    def _transition(self, task: Task, target: TaskStatus, *, note: str | None = None) -> None:
        self.machine.transition(task, target, note=note)
        self._save_task(task)
        self._bind_window(task)

    def _sync_from_window(self, task: Task) -> None:
        """阶段可能已写窗口；转状态前把除 status 外的 summary 拉回 Task。"""
        window = self.context.load_window(task.task_id)
        payload = window.summary.to_dict()
        payload["status"] = task.status.value
        task._adopt(task.model_copy(update={"summary": TaskSummary.from_dict(payload)}))

    def _merge_fields(self, task: Task, fields: dict[str, Any]) -> None:
        if not fields:
            return
        payload = task.summary.to_dict()
        payload.update(fields)
        payload["status"] = task.status.value
        task._adopt(task.model_copy(update={"summary": TaskSummary.from_dict(payload)}))
        writable = {key: value for key, value in fields.items() if key != "status"}
        if writable:
            with suppress(ContextError):
                self.context.update_summary(task.task_id, **writable)

    def _bind_window(self, task: Task) -> None:
        window = self.context.load_window(task.task_id)
        window.bind_summary(task.summary)
        self.context.save_window(window)

    def _deps_for(self, task: Task, runtime: StageRuntime) -> StageDeps:
        injector = RelatedContextInjector(
            self.memory.working(task.session_id),
            self.context,
            self.settings,
        )
        return StageDeps(
            llm=self.llm,
            context_manager=self.context,
            memory_manager=self.memory,
            tool_manager=self.tools,
            prompt_assembler=self.assembler,
            settings=self.settings,
            runtime=runtime,
            injector=injector.inject,
        )

    def _runtime(self, task: Task) -> StageRuntime:
        current = self._runtimes.get(task.task_id)
        if current is None:
            current = StageRuntime(started_at=time.monotonic())
            self._runtimes[task.task_id] = current
        return current

    def _save_task(self, task: Task) -> None:
        store = self._require_store()
        store.set(self._index_key(task.task_id), task.session_id)
        store.set(self._task_key(task.session_id, task.task_id), task.to_dict())

    def _index_get(self, task_id: str) -> str | None:
        value = self._require_store().get(self._index_key(task_id))
        return str(value) if value is not None else None

    def _require_store(self) -> Store:
        if self._store is None:
            raise TaskStateError("TaskManager 未配置 Store，无法持久化任务")
        return self._store

    def _task_key(self, session_id: str, task_id: str) -> str:
        return make_key(session_id, NS_TASK, task_id)

    def _index_key(self, task_id: str) -> str:
        return make_key(INDEX_SESSION, NS_TASK_INDEX, task_id)
