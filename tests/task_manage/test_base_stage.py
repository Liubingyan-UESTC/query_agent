"""步骤 20：异常封装为 StageResult，取消在入口即时生效。"""

from __future__ import annotations

import time

from agent.common.enums import TaskStatus
from agent.common.errors import AgentError
from agent.config.settings import LLMSettings, TaskSettings, load_settings
from agent.context_manage.context_manager import ContextManager
from agent.llm.mock_client import MockLLMClient
from agent.memory_manage.memory_manager import MemoryManager
from agent.models.context_window import ContextWindow
from agent.models.task import Task
from agent.prompt.assembler import PromptAssembler
from agent.store.memory_store import MemoryStore
from agent.task_manage.stages.base import BaseStage, StageDeps, StageRuntime
from agent.tool_manage.manager import ToolManager


class _Boom(BaseStage):
    stage_name = "boom"

    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> object:
        raise RuntimeError("exploded")


class _Ok(BaseStage):
    stage_name = "ok"

    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> object:
        from agent.task_manage.stages.base import StageResult

        return StageResult(next_status=TaskStatus.PLANNING)


def _deps(
    *, canceled: bool = False, duration: float = 300.0
) -> tuple[Task, ContextWindow, StageDeps]:
    settings = load_settings(
        env_file=None,
        llm=LLMSettings(use_mock=True),
        task=TaskSettings(max_duration_seconds=duration),
    )
    store = MemoryStore()
    memory = MemoryManager(store)
    context = ContextManager(store, settings)
    task = Task.create("q", session_id="sess_1")
    window = ContextWindow.from_task(task)
    deps = StageDeps(
        llm=MockLLMClient(),
        context_manager=context,
        memory_manager=memory,
        tool_manager=ToolManager(use_mock=True, discover=False),
        prompt_assembler=PromptAssembler(memory.knowledge),
        settings=settings,
        runtime=StageRuntime(started_at=time.monotonic(), canceled=canceled, cancel_reason="stop"),
    )
    return task, window, deps


def test_exception_becomes_failed_result() -> None:
    task, window, deps = _deps()
    result = _Boom().run(task, window, deps)

    assert result.next_status is TaskStatus.FAILED
    assert result.error is not None
    assert result.should_continue is False
    assert "exploded" in result.error.message
    assert result.elapsed_ms >= 0


def test_agent_error_is_wrapped() -> None:
    class _Raise(BaseStage):
        stage_name = "raise"

        def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> object:
            raise AgentError("业务失败")

    result = _Raise().run(*_deps())
    assert result.error is not None
    assert result.error.message == "业务失败"


def test_cancel_at_entry_skips_stage_body() -> None:
    task, window, deps = _deps(canceled=True)
    result = _Boom().run(task, window, deps)

    assert result.next_status is TaskStatus.CANCELED
    assert result.error is not None
    assert result.error.code == "task_canceled"


def test_duration_guard_fails_task() -> None:
    task, window, deps = _deps(duration=0.01)
    deps.runtime.started_at = time.monotonic() - 1
    result = _Ok().run(task, window, deps)

    assert result.next_status is TaskStatus.FAILED
    assert result.error is not None
    assert "总时长" in result.error.message
