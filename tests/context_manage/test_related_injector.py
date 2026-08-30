"""步骤 22：注入可被规划使用、split_by_scope 剥离、顺序稳定、缺失 id 不炸。"""

from __future__ import annotations

from agent.common.enums import ArtifactType, ContextScope, IntentType, PromptStage, TaskStatus
from agent.common.ids import new_session_id, new_task_id
from agent.config.settings import LLMSettings, load_settings
from agent.context_manage.context_manager import ContextManager
from agent.context_manage.related_injector import RelatedContextInjector
from agent.memory_manage.knowledge_memory import KnowledgeMemory
from agent.memory_manage.working_memory import WorkingMemory
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task_summary import TaskSummary
from agent.store.memory_store import MemoryStore


def _archive(session: str, store: object, *, task_id: str, title: str) -> None:
    summary = TaskSummary(
        task_id=task_id,
        content=title,
        intent=IntentType.NEW_QUERY,
        status=TaskStatus.COMPLETED,
        output=f"{title} 完成",
    )
    window = ContextWindow(task_id=task_id, session_id=session, summary=summary)
    window.append_message(Message.user(title))
    window.append_message(Message.assistant("查完了"))
    window.put_artifact(
        Artifact(
            artifact_id=f"art_{task_id[-6:]}",
            task_id=task_id,
            producer="search",
            artifact_type=ArtifactType.TABLE,
            title=title,
            data=[{"k": title}],
        )
    )
    WorkingMemory(session, store).archive(window)  # type: ignore[arg-type]


def test_inject_marks_related_and_split_strips_them() -> None:
    settings = load_settings(env_file=None, llm=LLMSettings(use_mock=True))
    store = MemoryStore()
    session = new_session_id()
    first = new_task_id()
    second = new_task_id()
    _archive(session, store, task_id=first, title="第一次查询")
    _archive(session, store, task_id=second, title="第二次查询")

    ctx = ContextManager(store, settings)
    current_id = new_task_id()
    ctx.create_window(
        current_id,
        session,
        "接着分析",
        KnowledgeMemory().get_system_prompt(PromptStage.INTENT_RECOGNITION),
    )
    injector = RelatedContextInjector(WorkingMemory(session, store), ctx, settings)
    report = injector.inject(current_id, [first, second, "task_missing"])

    assert report.injected == [first, second]
    assert "task_missing" in report.skipped

    window = ctx.load_window(current_id)
    related_msgs = [item for item in window.content if item.scope is ContextScope.RELATED]
    assert related_msgs
    assert [item.source_task_id for item in related_msgs if item.role.value == "system"][:2] == [
        first,
        second,
    ]
    assert f"art_{first[-6:]}" in window.artifacts
    assert window.artifacts[f"art_{first[-6:]}"].scope is ContextScope.RELATED

    sliced = window.split_by_scope()
    assert f"art_{first[-6:]}" in sliced[ContextScope.RELATED].artifacts
    assert f"art_{first[-6:]}" not in sliced[ContextScope.CURRENT].artifacts
    assert all(item.scope is ContextScope.RELATED for item in sliced[ContextScope.RELATED].messages)

    again = injector.inject(current_id, [first])
    assert first in again.skipped
