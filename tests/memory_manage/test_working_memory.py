"""步骤 11 验收：归档-读取往返、按 task_id 检索、顺序稳定、trim、会话隔离。"""

import pytest

from agent.common.enums import ArtifactType, ContextScope
from agent.common.errors import AgentMemoryError
from agent.memory_manage.working_memory import (
    NS_ARTIFACT,
    NS_CONTENT,
    NS_INDEX,
    NS_SUMMARY,
    WorkingMemory,
)
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task import Task
from agent.store.keys import make_key
from agent.store.memory_store import MemoryStore


def _window(
    session_id: str,
    query: str,
    *,
    extra_messages: list[Message] | None = None,
    extra_artifacts: list[Artifact] | None = None,
) -> ContextWindow:
    task = Task.create(query, session_id=session_id)
    window = ContextWindow.from_task(task)
    window.append_message(Message.user(query))
    for message in extra_messages or []:
        window.append_message(message)
    for artifact in extra_artifacts or []:
        window.put_artifact(artifact)
    return window


def _artifact(task_id: str, title: str, *, scope: ContextScope = ContextScope.CURRENT) -> Artifact:
    return Artifact(
        task_id=task_id if scope is ContextScope.CURRENT else "task_other",
        producer="kibana_query",
        artifact_type=ArtifactType.TABLE,
        title=title,
        data=[{"n": 1}],
        scope=scope,
        source_task_id=None if scope is ContextScope.CURRENT else "task_other",
    )


def test_archive_round_trip() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    window = _window("sess_1", "查昨天的错误")
    window.put_artifact(_artifact(window.task_id, "错误表"))

    memory.archive(window)
    summary, content, artifacts = memory.get_task_bundle(window.task_id)

    assert summary.task_id == window.task_id
    assert summary.content == "查昨天的错误"
    assert [item.content for item in content] == ["查昨天的错误"]
    assert artifacts[0].title == "错误表"
    assert memory.list_task_ids() == [window.task_id]


def test_related_and_history_are_not_archived() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    window = _window(
        "sess_1",
        "再分析一下",
        extra_messages=[
            Message.assistant(
                "来自关联任务",
                scope=ContextScope.RELATED,
                source_task_id="task_up",
            ),
            Message.user("历史片段", scope=ContextScope.HISTORY, source_task_id="task_old"),
        ],
        extra_artifacts=[
            _artifact("ignored", "关联表", scope=ContextScope.RELATED),
        ],
    )
    window.put_artifact(_artifact(window.task_id, "本任务表"))

    memory.archive(window)
    _summary, content, artifacts = memory.get_task_bundle(window.task_id)

    assert [item.content for item in content] == ["再分析一下"]
    assert [item.title for item in artifacts] == ["本任务表"]
    assert all(item.scope is ContextScope.CURRENT for item in content)
    assert all(item.scope is ContextScope.CURRENT for item in artifacts)


def test_write_order_is_stable_and_limit_takes_recent() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    first = _window("sess_1", "第一问")
    second = _window("sess_1", "第二问")
    third = _window("sess_1", "第三问")
    for window in (first, second, third):
        memory.archive(window)

    assert memory.list_task_ids() == [first.task_id, second.task_id, third.task_id]
    history = memory.get_summary_history(limit=2)
    assert [item["content"] for item in history] == ["第二问", "第三问"]
    assert [item["content"] for item in memory.get_summary_history()] == [
        "第一问",
        "第二问",
        "第三问",
    ]


def test_rearchive_overwrites_without_duplicating_order() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    window = _window("sess_1", "原问")
    memory.archive(window)

    window.summary.result = "已完成"
    memory.archive(window)

    assert memory.list_task_ids() == [window.task_id]
    bundled, *_ = memory.get_task_bundle(window.task_id)
    assert bundled.result == "已完成"


def test_trim_evicts_oldest() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    windows = [_window("sess_1", f"问{i}") for i in range(3)]
    for window in windows:
        memory.archive(window)

    memory.trim(1)

    assert memory.list_task_ids() == [windows[2].task_id]
    with pytest.raises(AgentMemoryError, match=windows[0].task_id):
        memory.get_task_bundle(windows[0].task_id)
    assert memory.get_content_history(windows[0].task_id) == []
    assert memory.get_artifact_history(windows[0].task_id) == []
    assert store.get(make_key("sess_1", NS_SUMMARY, windows[0].task_id)) is None
    assert store.get(make_key("sess_1", NS_CONTENT, windows[0].task_id)) is None
    assert store.get(make_key("sess_1", NS_ARTIFACT, windows[0].task_id)) is None


def test_trim_zero_clears_the_session() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    memory.archive(_window("sess_1", "将被清空"))
    memory.trim(0)

    assert memory.list_task_ids() == []


def test_trim_noop_when_under_limit() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    window = _window("sess_1", "只此一问")
    memory.archive(window)
    memory.trim(5)

    assert memory.list_task_ids() == [window.task_id]


def test_sessions_are_isolated() -> None:
    store = MemoryStore()
    one = WorkingMemory("sess_1", store)
    two = WorkingMemory("sess_2", store)
    first = _window("sess_1", "会话一")
    second = _window("sess_2", "会话二")
    one.archive(first)
    two.archive(second)

    assert one.list_task_ids() == [first.task_id]
    assert two.list_task_ids() == [second.task_id]
    assert first.task_id not in two.list_task_ids()
    with pytest.raises(AgentMemoryError):
        two.get_task_bundle(first.task_id)


def test_archive_rejects_foreign_session() -> None:
    memory = WorkingMemory("sess_1", MemoryStore())
    window = _window("sess_2", "别人的窗口")

    with pytest.raises(AgentMemoryError, match="sess_2"):
        memory.archive(window)


def test_missing_bundle_and_empty_histories() -> None:
    memory = WorkingMemory("sess_1", MemoryStore())

    with pytest.raises(AgentMemoryError, match="没有归档"):
        memory.get_task_bundle("task_absent")
    assert memory.get_content_history("task_absent") == []
    assert memory.get_artifact_history("task_absent") == []
    assert memory.get_summary_history() == []
    assert memory.get_summary_history(limit=0) == []


def test_returned_payloads_are_copies() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    window = _window("sess_1", "查日志")
    memory.archive(window)

    history = memory.get_summary_history()
    history[0]["content"] = "被改了"
    content = memory.get_content_history(window.task_id)
    content[0].content = "也被改了"

    assert memory.get_summary_history()[0]["content"] == "查日志"
    assert memory.get_content_history(window.task_id)[0].content == "查日志"


def test_index_without_summary_is_corrupt() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    store.push(make_key("sess_1", NS_INDEX, "task_ids"), "task_ghost")

    with pytest.raises(AgentMemoryError, match="缺少 summary"):
        memory.get_summary_history()


def test_rejects_bad_session_id_and_negative_bounds() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError, match="session_id"):
        WorkingMemory("", store)
    with pytest.raises(ValueError, match="冒号"):
        WorkingMemory("sess:1", store)

    memory = WorkingMemory("sess_1", store)
    with pytest.raises(ValueError, match="limit"):
        memory.get_summary_history(limit=-1)
    with pytest.raises(ValueError, match="max_tasks"):
        memory.trim(-1)


def test_namespaces_match_the_plan() -> None:
    assert NS_SUMMARY == "task_summary_history"
    assert NS_CONTENT == "task_content_history"
    assert NS_ARTIFACT == "task_artifact_history"
    assert NS_INDEX == "wm_index"


def test_rearchive_keeps_first_write_position() -> None:
    store = MemoryStore()
    memory = WorkingMemory("sess_1", store)
    first = _window("sess_1", "先")
    second = _window("sess_1", "后")
    memory.archive(first)
    memory.archive(second)
    first.append_message(Message.assistant("补充"))
    memory.archive(first)

    assert memory.list_task_ids() == [first.task_id, second.task_id]
    assert [item.role.value for item in memory.get_content_history(first.task_id)] == [
        "user",
        "assistant",
    ]
