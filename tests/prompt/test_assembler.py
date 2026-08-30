"""步骤 15：四类 prompt 的结构断言 + golden 快照。"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.common.enums import ArtifactType, IntentType, MessageRole, TaskStatus
from agent.context_manage.token_counter import CharEstimateCounter
from agent.memory_manage.knowledge_memory import KnowledgeMemory
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task_summary import Operation, TaskSummary
from agent.prompt.assembler import (
    EMPTY_PLAN_OUTPUT_SCHEMA,
    EXECUTE_OUTPUT_SCHEMA,
    INTENT_OUTPUT_SCHEMA,
    PLAN_OUTPUT_SCHEMA,
    VALIDATE_OUTPUT_SCHEMA,
    AssembledPrompt,
    EmptyPlanOutput,
    PlanOutput,
    PromptAssembler,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN") == "1"

HISTORY = [
    {
        "task_id": "task_0",
        "content": "昨天 query-agent 有多少 ERROR",
        "status": "completed",
        "intent": "new_query",
        "output": "共 12 条",
    }
]


def _ts() -> datetime:
    return datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)


def _window() -> ContextWindow:
    summary = TaskSummary(
        task_id="task_1",
        content="查昨天 query-agent 的 ERROR 日志",
        intent=IntentType.NEW_QUERY,
        related_task_ids=["task_0"],
        status=TaskStatus.PLANNING,
        operations=[
            Operation(
                index=0,
                name="查错误日志",
                tool="search",
                args={"index": "logs-app", "limit": 20},
            )
        ],
    )
    window = ContextWindow(task_id="task_1", session_id="sess_1", summary=summary)
    window.append_message(
        Message.system("窗口内旧系统提示词", message_id="msg_sys", created_at=_ts())
    )
    window.append_message(
        Message.user("查昨天 query-agent 的 ERROR 日志", message_id="msg_user", created_at=_ts())
    )
    window.put_artifact(
        Artifact(
            artifact_id="art_fixed",
            task_id="task_1",
            producer="search",
            artifact_type=ArtifactType.TABLE,
            title="错误表",
            data=[{"service": "query-agent", "level": "ERROR"}],
        )
    )
    return window


def _assembler() -> PromptAssembler:
    return PromptAssembler(KnowledgeMemory(), counter=CharEstimateCounter())


def render_prompt(assembled: AssembledPrompt) -> str:
    """把请求渲染成可 diff 的文本。只用 to_llm_dict，避免 ID / 时间戳抖动。"""
    lines: list[str] = []
    if assembled.request.purpose is not None:
        lines.extend([f"# purpose: {assembled.request.purpose}", ""])
    for message in assembled.request.messages:
        payload = message.to_llm_dict()
        lines.append(f"## {payload['role']}")
        lines.append(str(payload.get("content") or ""))
        if payload.get("tool_calls"):
            lines.append(json.dumps(payload["tool_calls"], ensure_ascii=False, indent=2))
        if payload.get("name"):
            lines.append(f"name: {payload['name']}")
        if payload.get("tool_call_id"):
            lines.append(f"tool_call_id: {payload['tool_call_id']}")
        lines.append("")
    if assembled.request.tools:
        lines.append("## tools")
        lines.append(json.dumps(assembled.request.tools, ensure_ascii=False, indent=2))
        lines.append("")
    lines.append("## output_schema")
    lines.append(json.dumps(assembled.output_schema, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("")
    return "\n".join(lines)


def _assert_golden(name: str, text: str) -> None:
    path = GOLDEN_DIR / name
    if UPDATE_GOLDEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.is_file(), f"缺少 golden 文件 {path}，请设置 UPDATE_GOLDEN=1 生成"
    assert text == path.read_text(encoding="utf-8"), f"prompt 快照与 {name} 不一致"


def test_intent_prompt_schema_and_fields() -> None:
    assembled = _assembler().build_intent_prompt(_window(), HISTORY)
    request = assembled.request

    assert request.purpose == "intent"
    assert request.messages[0].role is MessageRole.SYSTEM
    assert "你是面向 Kibana" in request.messages[0].content
    assert "当前阶段是意图识别" in request.messages[0].content
    user = request.messages[1].content
    assert "查昨天 query-agent 的 ERROR 日志" in user
    assert "task_0" in user
    assert "operations" not in user
    assert assembled.output_schema == INTENT_OUTPUT_SCHEMA
    assert "new_query" in assembled.output_schema["properties"]["intent"]["enum"]


def test_plan_prompt_uses_skill_schema_and_does_not_repeat_base() -> None:
    assembled = _assembler().build_plan_prompt(_window())
    system = assembled.request.messages[0].content

    assert assembled.request.purpose == "plan"
    assert system.count("你是面向 Kibana") == 1
    assert "意图为 new_query" in system
    assert "允许的工具：search" in system
    assert "## logs-app" in system
    assert assembled.output_schema == PLAN_OUTPUT_SCHEMA
    assert assembled.output_model is PlanOutput
    assert "maxItems" not in assembled.output_schema["properties"]["operations"]


def test_chat_plan_schema_forbids_operations() -> None:
    window = _window()
    window.summary = window.summary.model_copy(update={"intent": IntentType.CHAT, "operations": []})
    assembled = _assembler().build_plan_prompt(window)

    assert assembled.output_schema == EMPTY_PLAN_OUTPUT_SCHEMA
    assert assembled.output_model is EmptyPlanOutput
    assert assembled.output_schema["properties"]["operations"]["maxItems"] == 0


def test_plan_prompt_requires_intent() -> None:
    window = _window()
    window.summary = window.summary.model_copy(update={"intent": None})
    with pytest.raises(ValueError, match="必须已填写 intent"):
        _assembler().build_plan_prompt(window)


def test_execute_prompt_attaches_tools_and_operation() -> None:
    window = _window()
    schemas = [
        {
            "type": "function",
            "function": {"name": "search", "description": "查询", "parameters": {"type": "object"}},
        }
    ]
    assembled = _assembler().build_execute_prompt(window, window.summary.operations[0], schemas)

    assert assembled.request.purpose == "execute"
    assert assembled.request.tools == schemas
    assert "当前步骤" in assembled.request.messages[0].content
    assert assembled.output_schema == EXECUTE_OUTPUT_SCHEMA


def test_validate_prompt_uses_primary_purpose() -> None:
    assembled = _assembler().build_validate_prompt(_window())

    assert assembled.request.purpose is None
    body = assembled.request.messages[1].content
    assert "原始请求" in body
    assert "art_fixed" in body
    assert assembled.output_schema == VALIDATE_OUTPUT_SCHEMA


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        (
            "intent.txt",
            lambda: _assembler().build_intent_prompt(_window(), HISTORY),
        ),
        ("plan.txt", lambda: _assembler().build_plan_prompt(_window())),
        (
            "execute.txt",
            lambda: _assembler().build_execute_prompt(
                _window(),
                _window().summary.operations[0],
                [{"type": "function", "function": {"name": "search"}}],
            ),
        ),
        ("validate.txt", lambda: _assembler().build_validate_prompt(_window())),
    ],
)
def test_prompt_goldens(name: str, builder: object) -> None:
    assembled = builder()  # type: ignore[operator]
    _assert_golden(name, render_prompt(assembled))
