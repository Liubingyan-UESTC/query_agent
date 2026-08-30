"""步骤 14：四阶段结构、预算裁剪优先级、裁剪后仍是合法对话。"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from agent.common.enums import (
    ArtifactType,
    ContextScope,
    IntentType,
    MessageRole,
    PromptStage,
    TaskStatus,
)
from agent.config.settings import ContextSettings
from agent.context_manage.token_counter import CharEstimateCounter
from agent.context_manage.window_policy import (
    FEWSHOT_SOURCE_TASK_ID,
    HISTORY_SUMMARY_FIELDS,
    StageAssembleOptions,
    TokenBudget,
    _truncate_bodies,
    assemble_stage_messages,
    clip_history_summaries,
    clip_to_budget,
    is_tool_error,
    last_k_turns,
)
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message, ToolCall
from agent.models.task_summary import Operation, TaskSummary

COUNTER = CharEstimateCounter()


class FakeKnowledge:
    def __init__(self) -> None:
        self._shots = [
            Message.user("查昨天的错误", message_id="msg_shot_u"),
            Message.assistant("用 search 查 logs-app", message_id="msg_shot_a"),
        ]

    def get_system_prompt(
        self, stage: PromptStage | str, intent: IntentType | str | None = None
    ) -> str:
        label = stage.value if isinstance(stage, PromptStage) else stage
        extra = f":{intent}" if intent is not None else ""
        return f"SYS {label}{extra}"

    def get_skill(self, intent: IntentType | str) -> Any:
        return SimpleNamespace(
            name="kibana 新查询",
            description="拉取数据",
            allowed_tools=["search"],
            field_dict_refs=["logs-app"],
            few_shots=self._shots,
            output_schema={"type": "object"},
        )

    def get_field_dict(self, index: str | None = None) -> str:
        return f"# 字段字典 {index or 'all'}\n| @timestamp | date |"

    def get_few_shots(self, intent: IntentType | str) -> list[Message]:
        return list(self._shots)

    def get_allowed_tools(self, intent: IntentType | str) -> list[str]:
        return list(self.get_skill(intent).allowed_tools)


def _ts(second: int) -> datetime:
    return datetime(2026, 8, 1, 12, 0, second, tzinfo=UTC)


def _window(**overrides: Any) -> ContextWindow:
    summary = TaskSummary(
        task_id="task_1",
        content="查昨天的错误日志",
        intent=IntentType.NEW_QUERY,
        status=TaskStatus.PLANNING,
        operations=[Operation(index=0, name="查错误", tool="search", args={"index": "logs-app"})],
    )
    window = ContextWindow(task_id="task_1", session_id="sess_1", summary=summary)
    window.append_message(
        Message.system("旧系统提示词", message_id="msg_old_sys", created_at=_ts(1))
    )
    window.append_message(
        Message.user("查昨天的错误日志", message_id="msg_user", created_at=_ts(2))
    )
    for key, value in overrides.items():
        setattr(window, key, value)
    return window


def _legal(messages: list[Message]) -> None:
    assert messages, "裁剪后不能空"
    assert messages[0].role is MessageRole.SYSTEM
    assert sum(1 for item in messages if item.role is MessageRole.SYSTEM) == 1
    open_calls: set[str] = set()
    for message in messages:
        if message.role is MessageRole.ASSISTANT:
            for call in message.tool_calls:
                open_calls.add(call.id)
        elif message.role is MessageRole.TOOL:
            assert message.tool_call_id in open_calls, "tool 消息必须对应 tool_call"
            open_calls.discard(message.tool_call_id)
    assert not open_calls, f"未闭合的 tool_call: {open_calls}"


# ============================================================ 历史投影


def test_clip_history_takes_recent_and_projects_fields() -> None:
    history = [
        {
            "task_id": "task_0",
            "content": "旧",
            "status": "completed",
            "intent": "chat",
            "output": "x",
        },
        {"task_id": "task_1", "content": "新", "status": "completed", "extra": 1},
    ]

    clipped = clip_history_summaries(history, limit=1)

    assert clipped == [{"task_id": "task_1", "content": "新", "status": "completed"}]
    assert set(clipped[0]) <= set(HISTORY_SUMMARY_FIELDS)


def test_clip_history_zero_limit() -> None:
    assert clip_history_summaries([{"task_id": "t"}], limit=0) == []


# ============================================================ 四阶段结构


def test_intent_stage_feeds_content_status_and_history() -> None:
    window = _window()
    history = [
        {"task_id": "task_x", "content": "更早", "status": "failed", "intent": "unknown"},
        {"task_id": "task_0", "content": "上一问", "status": "completed", "intent": "chat"},
    ]
    messages = assemble_stage_messages(
        window,
        PromptStage.INTENT_RECOGNITION,
        FakeKnowledge(),
        options=StageAssembleOptions(summary_history=history, history_summary_limit=1),
        counter=COUNTER,
    )

    _legal(messages)
    assert messages[0].content.startswith("SYS intent_recognition")
    user = messages[1]
    assert user.role is MessageRole.USER
    assert "查昨天的错误日志" in user.content
    assert '"status": "planning"' in user.content
    assert "task_0" in user.content
    assert "task_x" not in user.content
    assert "operations" not in user.content
    assert "旧系统提示词" not in "".join(item.content for item in messages)


def test_plan_stage_includes_skill_field_dict_and_full_summary() -> None:
    window = _window()
    messages = assemble_stage_messages(
        window,
        PromptStage.PLAN,
        FakeKnowledge(),
        intent=IntentType.NEW_QUERY,
        counter=COUNTER,
    )

    _legal(messages)
    assert messages[0].content.startswith("SYS plan:new_query")
    assert "kibana 新查询" in messages[0].content
    assert "允许的工具：search" in messages[0].content
    assert "# 字段字典 logs-app" in messages[0].content
    assert messages[1].role is MessageRole.USER
    assert messages[1].content == "查昨天的错误"
    assert messages[2].role is MessageRole.ASSISTANT
    payload = messages[3].content
    assert "查昨天的错误日志" in payload
    assert "search" in payload
    assert "产物索引" in payload
    assert "对话摘要" in payload
    assert "[user/current]" in payload


def test_plan_stage_requires_intent() -> None:
    window = _window()
    window.summary = window.summary.model_copy(update={"intent": None})
    with pytest.raises(ValueError, match="必须提供 intent"):
        assemble_stage_messages(window, PromptStage.PLAN, FakeKnowledge(), counter=COUNTER)


def test_execute_stage_keeps_recent_turns_and_tool_schemas() -> None:
    window = _window()
    for index in range(3):
        window.append_message(
            Message.user(f"旧轮 {index}", message_id=f"msg_old_{index}", created_at=_ts(10 + index))
        )
    window.append_message(Message.user("最近一问", message_id="msg_recent", created_at=_ts(20)))
    operation = window.summary.operations[0]
    schemas = [{"type": "function", "function": {"name": "search"}}]
    messages = assemble_stage_messages(
        window,
        PromptStage.EXECUTE,
        FakeKnowledge(),
        options=StageAssembleOptions(
            operation=operation,
            tool_schemas=schemas,
            recent_content_limit=2,
        ),
        counter=COUNTER,
    )

    _legal(messages)
    text = "\n".join(item.content for item in messages)
    assert "SYS execute" in messages[0].content
    assert "当前步骤" in messages[0].content
    assert "search" in messages[0].content
    assert "最近一问" in text
    assert "旧轮 0" not in text


def test_validate_stage_includes_query_summary_digest_and_index() -> None:
    window = _window()
    artifact = Artifact(
        artifact_id="art_1",
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        title="错误表",
        data=[{"n": 1}],
    )
    window.put_artifact(artifact)
    messages = assemble_stage_messages(
        window, PromptStage.VALIDATE, FakeKnowledge(), counter=COUNTER
    )

    _legal(messages)
    assert messages[0].content.startswith("SYS validate")
    body = messages[1].content
    assert "原始请求" in body
    assert "查昨天的错误日志" in body
    assert "art_1" in body
    assert "错误表" in body
    assert "对话摘要" in body


# ============================================================ 预算裁剪


def _budget(max_total: int) -> TokenBudget:
    return TokenBudget(
        max_total=max_total, summary=max_total, content=max_total, artifacts=max_total
    )


def test_clip_drops_history_then_related_then_old_current() -> None:
    system = Message.system("S", message_id="s", created_at=_ts(0))
    history = Message.user(
        "H" * 40,
        message_id="h",
        scope=ContextScope.HISTORY,
        source_task_id="task_old",
        created_at=_ts(1),
    )
    related = Message.user(
        "R" * 40,
        message_id="r",
        scope=ContextScope.RELATED,
        source_task_id="task_up",
        created_at=_ts(2),
    )
    old_current = Message.user("C_OLD" * 8, message_id="c0", created_at=_ts(3))
    new_current = Message.user("C_NEW" * 8, message_id="c1", created_at=_ts(4))
    messages = [system, history, related, old_current, new_current]
    full = COUNTER.count_messages(messages)
    keep_core = COUNTER.count_messages([system, new_current])
    budget = _budget(keep_core + (full - keep_core) // 4)

    clipped = clip_to_budget(messages, budget, COUNTER)

    _legal(clipped)
    contents = [item.content for item in clipped]
    assert "S" in contents[0]
    assert "C_NEW" * 8 in contents
    assert "H" * 40 not in contents
    # 预算只够留下当前任务的新消息时，HISTORY / RELATED 必须先走
    assert all(item.scope is not ContextScope.HISTORY for item in clipped[1:])


def test_clip_keeps_tool_error_over_older_same_scope() -> None:
    system = Message.system("S", message_id="s", created_at=_ts(0))
    old_user = Message.user("old-user-payload-xxxx", message_id="u0", created_at=_ts(1))
    assistant = Message.assistant(
        "",
        message_id="a0",
        created_at=_ts(2),
        tool_calls=[ToolCall(id="call_err", name="search", arguments="{}")],
    )
    tool_err = Message.tool_result(
        "search 失败：timeout",
        tool_call_id="call_err",
        message_id="t0",
        created_at=_ts(3),
        meta={"error": True},
    )
    newer = Message.user("newer-user-payload-xxxx", message_id="u1", created_at=_ts(4))
    messages = [system, old_user, assistant, tool_err, newer]
    # 只留 system + 错误组 + 最新 user 会超，必须丢掉最旧非错误
    keep_error = COUNTER.count_messages([system, assistant, tool_err, newer])
    clipped = clip_to_budget(messages, _budget(keep_error), COUNTER)

    _legal(clipped)
    ids = [item.message_id for item in clipped]
    assert "t0" in ids
    assert "a0" in ids
    assert "u0" not in ids


def test_clip_drops_assistant_and_tools_as_a_group() -> None:
    system = Message.system("S", message_id="s", created_at=_ts(0))
    user = Message.user("keep-me-please", message_id="u", created_at=_ts(1))
    assistant = Message.assistant(
        "call",
        message_id="a",
        created_at=_ts(2),
        tool_calls=[
            ToolCall(id="c1", name="search", arguments="{}"),
            ToolCall(id="c2", name="search", arguments="{}"),
        ],
    )
    tool_a = Message.tool_result("ok-a", tool_call_id="c1", message_id="t1", created_at=_ts(3))
    tool_b = Message.tool_result("ok-b", tool_call_id="c2", message_id="t2", created_at=_ts(4))
    messages = [system, user, assistant, tool_a, tool_b]
    keep_user = COUNTER.count_messages([system, user])
    clipped = clip_to_budget(messages, _budget(keep_user), COUNTER)

    _legal(clipped)
    roles = [item.role for item in clipped]
    assert MessageRole.TOOL not in roles
    assert all(not item.tool_calls for item in clipped)


def test_clip_converts_extra_system_so_system_stays_unique() -> None:
    messages = [
        Message.system("主系统", message_id="s0", created_at=_ts(0)),
        Message.system(
            "关联说明",
            message_id="s1",
            scope=ContextScope.RELATED,
            source_task_id="task_up",
            created_at=_ts(1),
        ),
        Message.user("问", message_id="u", created_at=_ts(2)),
    ]
    clipped = clip_to_budget(messages, _budget(10_000), COUNTER)

    _legal(clipped)
    assert clipped[1].role is MessageRole.USER
    assert clipped[1].content.startswith("【系统摘录】")


def test_is_tool_error_reads_meta_and_text() -> None:
    assert is_tool_error(Message.tool_result("ok", tool_call_id="c", meta={"error": True}))
    assert is_tool_error(Message.tool_result("工具失败：字段非法", tool_call_id="c"))
    assert not is_tool_error(Message.tool_result("共 3 行", tool_call_id="c"))
    assert not is_tool_error(Message.user("失败了"))


def test_last_k_turns_keeps_tool_groups() -> None:
    items = [
        Message.user("u0", message_id="u0"),
        Message.assistant(
            "",
            message_id="a0",
            tool_calls=[ToolCall(id="c", name="search", arguments="{}")],
        ),
        Message.tool_result("ok", tool_call_id="c", message_id="t0"),
        Message.user("u1", message_id="u1"),
    ]
    kept = last_k_turns(items, 2)
    assert [item.message_id for item in kept] == ["a0", "t0", "u1"]


def test_token_budget_from_settings() -> None:
    settings = ContextSettings.from_env(None)
    budget = TokenBudget.from_settings(settings)
    assert budget.max_total == settings.max_total_tokens
    assert budget.summary + budget.content + budget.artifacts == settings.max_total_tokens


def test_plan_few_shots_are_copies_and_marked_history() -> None:
    knowledge = FakeKnowledge()
    original = knowledge.get_few_shots(IntentType.NEW_QUERY)[0].content
    messages = assemble_stage_messages(
        _window(), PromptStage.PLAN, knowledge, intent=IntentType.NEW_QUERY, counter=COUNTER
    )
    shot = messages[1]
    assert shot.scope is ContextScope.HISTORY
    assert shot.source_task_id == FEWSHOT_SOURCE_TASK_ID
    shot.content = "MUTATED"
    assert knowledge.get_few_shots(IntentType.NEW_QUERY)[0].content == original


def test_execute_preview_only_for_cited_artifacts() -> None:
    window = _window()
    window.put_artifact(
        Artifact(
            artifact_id="art_noise",
            task_id="task_1",
            producer="search",
            artifact_type=ArtifactType.TABLE,
            title="无关表",
            data=[{"n": 1}],
        )
    )
    window.put_artifact(
        Artifact(
            artifact_id="art_need",
            task_id="task_1",
            producer="search",
            artifact_type=ArtifactType.TABLE,
            title="需要的表",
            data=[{"n": 2}],
        )
    )
    bare = assemble_stage_messages(
        window,
        PromptStage.EXECUTE,
        FakeKnowledge(),
        options=StageAssembleOptions(operation=window.summary.operations[0]),
        counter=COUNTER,
    )
    assert all("相关产物预览" not in item.content for item in bare)

    cited = window.summary.operations[0].model_copy(update={"args": {"artifact_id": "art_need"}})
    messages = assemble_stage_messages(
        window,
        PromptStage.EXECUTE,
        FakeKnowledge(),
        options=StageAssembleOptions(operation=cited),
        counter=COUNTER,
    )
    text = "\n".join(item.content for item in messages)
    assert "art_need" in text
    assert "art_noise" not in text


def test_truncate_bodies_shortens_non_system() -> None:
    system = Message.system("S", message_id="s")
    user = Message.user("KEEPME" * 80, message_id="u")
    budget = COUNTER.count_messages([system]) + 8
    clipped = _truncate_bodies([system, user], budget, COUNTER)
    assert clipped[0].role is MessageRole.SYSTEM
    assert len(clipped[1].content) < len(user.content)
    assert COUNTER.count_messages(clipped) <= budget
