"""步骤 13 验收：创建后 CREATED + system 就位；非法字段拒绝；产物只入引用与预览。"""

from __future__ import annotations

import pytest

from agent.common.enums import ArtifactType, ContextScope, IntentType, MessageRole, TaskStatus
from agent.common.errors import ContextError
from agent.config.settings import LLMSettings, load_settings
from agent.context_manage.context_manager import (
    INDEX_SESSION,
    NS_INDEX,
    NS_WINDOW,
    ContextManager,
)
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.store.keys import make_key
from agent.store.memory_store import MemoryStore

SYSTEM = "你是查询 Agent。"
QUERY = "查昨天的错误日志"
_SECRET = "ROW6_SECRET_TOKEN"


def _mgr() -> tuple[ContextManager, MemoryStore]:
    store = MemoryStore()
    return ContextManager(store), store


def _create(
    mgr: ContextManager, task_id: str = "task_1", session_id: str = "sess_1"
) -> ContextWindow:
    return mgr.create_window(task_id, session_id, QUERY, SYSTEM)


def _table(task_id: str) -> Artifact:
    rows = [{"n": i, "msg": f"row-{i}"} for i in range(5)]
    rows.append({"n": 5, "msg": _SECRET})
    return Artifact(
        task_id=task_id,
        producer="search",
        artifact_type=ArtifactType.TABLE,
        title="错误表",
        data=rows,
    )


def test_create_window_is_created_with_system_message() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)

    assert window.summary.status is TaskStatus.CREATED
    assert window.summary.content == QUERY
    assert window.content[0].role is MessageRole.SYSTEM
    assert window.content[0].content == SYSTEM
    loaded = mgr.load_window(window.task_id)
    assert loaded.summary.status is TaskStatus.CREATED
    assert loaded.content[0].role is MessageRole.SYSTEM


def test_create_window_rejects_blank_query_and_prompt() -> None:
    mgr, _store = _mgr()
    with pytest.raises(ContextError, match="query"):
        mgr.create_window("task_1", "sess_1", "  ", SYSTEM)
    with pytest.raises(ContextError, match="system_prompt"):
        mgr.create_window("task_1", "sess_1", QUERY, "")


def test_create_window_rejects_duplicate_task() -> None:
    mgr, _store = _mgr()
    _create(mgr)
    with pytest.raises(ContextError, match="已存在"):
        _create(mgr)


def test_load_missing_window() -> None:
    mgr, _store = _mgr()
    with pytest.raises(ContextError, match="没有任务"):
        mgr.load_window("task_missing")


def test_index_without_payload_is_corrupt() -> None:
    mgr, store = _mgr()
    store.set(make_key(INDEX_SESSION, NS_INDEX, "task_1"), "sess_1")
    with pytest.raises(ContextError, match="有索引但缺少窗口"):
        mgr.load_window("task_1")


def test_payload_must_be_object() -> None:
    mgr, store = _mgr()
    store.set(make_key(INDEX_SESSION, NS_INDEX, "task_1"), "sess_1")
    store.set(make_key("sess_1", NS_WINDOW, "task_1"), ["not", "a", "dict"])
    with pytest.raises(ContextError, match="不是对象"):
        mgr.load_window("task_1")


def test_payload_must_be_valid_window() -> None:
    mgr, store = _mgr()
    store.set(make_key(INDEX_SESSION, NS_INDEX, "task_1"), "sess_1")
    store.set(make_key("sess_1", NS_WINDOW, "task_1"), {"task_id": "task_1"})
    with pytest.raises(ContextError, match="无法还原"):
        mgr.load_window("task_1")


def test_save_rejects_session_rebind() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    window.session_id = "sess_other"
    with pytest.raises(ContextError, match="已绑定 session"):
        mgr.save_window(window)


def test_illegal_key_is_context_error() -> None:
    mgr, _store = _mgr()
    with pytest.raises(ContextError, match="冒号"):
        mgr.create_window("task:1", "sess_1", QUERY, SYSTEM)
    with pytest.raises(ContextError, match="冒号"):
        mgr.create_window("task_1", "sess:1", QUERY, SYSTEM)


def test_update_summary_writes_whitelisted_fields() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    mgr.update_summary(
        window.task_id,
        intent="new_query",
        related_task_ids=["task_0"],
        result="ok",
        output="done",
    )
    loaded = mgr.load_window(window.task_id)
    assert loaded.summary.intent is IntentType.NEW_QUERY
    assert loaded.summary.related_task_ids == ["task_0"]
    assert loaded.summary.result == "ok"
    assert loaded.summary.output == "done"
    assert loaded.summary.status is TaskStatus.CREATED
    assert loaded.summary.content == QUERY


def test_update_summary_rejects_unknown_and_frozen_fields() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    with pytest.raises(ContextError, match="必须指定字段"):
        mgr.update_summary(window.task_id)
    with pytest.raises(ContextError, match="不在白名单"):
        mgr.update_summary(window.task_id, status="completed")
    with pytest.raises(ContextError, match="不在白名单"):
        mgr.update_summary(window.task_id, content="被模型改写")
    with pytest.raises(ContextError, match="不在白名单"):
        mgr.update_summary(window.task_id, extra="dirty")


def test_update_summary_rejects_bad_types() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    with pytest.raises(ContextError, match="类型不合法"):
        mgr.update_summary(window.task_id, intent="NEW-QUERY")


def test_append_message_persists() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    mgr.append_message(window.task_id, Message.user(QUERY))
    loaded = mgr.load_window(window.task_id)
    assert [item.role for item in loaded.content] == [MessageRole.SYSTEM, MessageRole.USER]


def test_append_duplicate_message_id_is_rejected() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    existing = window.content[0]
    with pytest.raises(ContextError, match="已有 message_id"):
        mgr.append_message(window.task_id, existing)


def test_add_artifact_puts_preview_not_full_data() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    artifact = _table(window.task_id)
    artifact_id = mgr.add_artifact(window.task_id, artifact)

    loaded = mgr.load_window(window.task_id)
    stored = loaded.get_artifact(artifact_id)
    assert stored.data[-1]["msg"] == _SECRET
    preview_msg = loaded.content[-1]
    assert preview_msg.role is MessageRole.ASSISTANT
    assert preview_msg.artifact_refs == [artifact_id]
    assert artifact_id in preview_msg.content
    assert "[产物引用]" in preview_msg.content
    assert _SECRET not in preview_msg.content
    assert "row-0" in preview_msg.content


def test_add_artifact_rejects_non_current_and_duplicates() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    related = Artifact(
        task_id="task_other",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        title="关联",
        data=[{"a": 1}],
        scope=ContextScope.RELATED,
        source_task_id="task_other",
    )
    with pytest.raises(ContextError, match="inject_segment"):
        mgr.add_artifact(window.task_id, related)

    artifact = _table(window.task_id)
    mgr.add_artifact(window.task_id, artifact)
    with pytest.raises(ContextError, match="已有 artifact_id"):
        mgr.add_artifact(window.task_id, artifact)


def test_add_artifact_rejects_foreign_current_task() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    foreign = Artifact(
        task_id="task_other",
        producer="search",
        artifact_type=ArtifactType.SCALAR,
        title="错绑",
        data=1,
    )
    with pytest.raises(ContextError, match="必须属于本任务"):
        mgr.add_artifact(window.task_id, foreign)


def test_inject_segment_stamps_scope_and_skips_preview() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    source = "task_up"
    artifact = Artifact(
        task_id=source,
        producer="search",
        artifact_type=ArtifactType.TABLE,
        title="上游表",
        data=[{"x": 1}],
        scope=ContextScope.CURRENT,
    )
    mgr.inject_segment(
        window.task_id,
        [Message.user("上游问")],
        [artifact],
        source_task_id=source,
        scope=ContextScope.RELATED,
    )
    loaded = mgr.load_window(window.task_id)
    injected = loaded.content[-1]
    assert injected.scope is ContextScope.RELATED
    assert injected.source_task_id == source
    assert injected.artifact_refs == []
    stored = loaded.get_artifact(artifact.artifact_id)
    assert stored.scope is ContextScope.RELATED
    assert stored.source_task_id == source
    assert all(
        "[产物引用]" not in item.content
        for item in loaded.content
        if item.role is MessageRole.ASSISTANT
    )


def test_inject_segment_rejects_current_and_empty() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    with pytest.raises(ContextError, match="不能写入 scope=current"):
        mgr.inject_segment(
            window.task_id,
            [Message.user("x")],
            [],
            source_task_id="task_up",
            scope=ContextScope.CURRENT,
        )
    with pytest.raises(ContextError, match="source_task_id"):
        mgr.inject_segment(
            window.task_id,
            [Message.user("x")],
            [],
            source_task_id="",
            scope=ContextScope.HISTORY,
        )
    with pytest.raises(ContextError, match="不能同时为空"):
        mgr.inject_segment(
            window.task_id, [], [], source_task_id="task_up", scope=ContextScope.HISTORY
        )


def test_inject_duplicate_artifact_is_rejected() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    artifact = Artifact(
        task_id="task_up",
        producer="search",
        artifact_type=ArtifactType.TEXT,
        title="t",
        data="v",
        scope=ContextScope.RELATED,
        source_task_id="task_up",
    )
    mgr.inject_segment(
        window.task_id,
        [],
        [artifact],
        source_task_id="task_up",
        scope=ContextScope.RELATED,
    )
    with pytest.raises(ContextError, match="已有 artifact_id"):
        mgr.inject_segment(
            window.task_id,
            [],
            [artifact],
            source_task_id="task_up",
            scope=ContextScope.RELATED,
        )


def test_snapshot_omits_artifact_data() -> None:
    mgr, _store = _mgr()
    window = _create(mgr)
    artifact = _table(window.task_id)
    mgr.add_artifact(window.task_id, artifact)
    snap = mgr.snapshot(window.task_id)

    assert snap["task_id"] == window.task_id
    assert snap["summary"]["content"] == QUERY
    assert snap["artifacts"][0]["artifact_id"] == artifact.artifact_id
    assert "data" not in snap["artifacts"][0]
    assert _SECRET not in str(snap["artifacts"])
    assert (
        _SECRET
        in mgr.load_window(window.task_id).get_artifact(artifact.artifact_id).data[-1]["msg"]
    )


def test_settings_from_app_settings() -> None:
    settings = load_settings(env_file=None, llm=LLMSettings(use_mock=True))
    mgr = ContextManager(MemoryStore(), settings)
    assert mgr.settings.max_total_tokens == settings.context.max_total_tokens
    window = mgr.create_window("task_1", "sess_1", QUERY, SYSTEM)
    assert window.task_id == "task_1"


def test_settings_from_context_settings() -> None:
    from agent.config.settings import ContextSettings

    ctx = ContextSettings.from_env(None)
    mgr = ContextManager(MemoryStore(), ctx)
    assert mgr.settings is ctx
