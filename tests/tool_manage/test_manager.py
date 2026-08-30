"""步骤 17：参数校验失败不中断、超时被中断、意图白名单过滤。"""

from __future__ import annotations

import time

import pytest

from agent.common.enums import ArtifactType, IntentType
from agent.common.errors import ToolError
from agent.models.base import AgentModel
from agent.tool_manage.base import BaseTool, ToolContext, ToolResult
from agent.tool_manage.manager import ToolManager
from agent.tool_manage.registry import ToolRegistry, register_tool


class _SleepArgs(AgentModel):
    seconds: float = 0.01


class _Slow(BaseTool):
    name = "slow"
    description = "睡眠"
    args_schema = _SleepArgs
    produces = ArtifactType.TEXT
    timeout = 0.05

    def run(self, args: _SleepArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        time.sleep(args.seconds)
        return ToolResult.success("woke")


def _ctx() -> ToolContext:
    return ToolContext(task_id="task_1", session_id="sess_1", trace_id="trace_1")


def test_invoke_validation_failure_returns_ok_false() -> None:
    manager = ToolManager(use_mock=True)
    result = manager.invoke("search", {"limit": "not-int"}, _ctx())

    assert result.ok is False
    assert result.error is not None
    assert result.error.retryable is False
    assert "参数校验失败" in result.text


def test_invoke_unknown_tool_returns_ok_false() -> None:
    manager = ToolManager(use_mock=True)
    result = manager.invoke("nope", {}, _ctx())

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "tool_not_found"


def test_timeout_is_interrupted() -> None:
    registry = ToolRegistry()
    register_tool(_Slow, registry=registry)
    manager = ToolManager(registry, use_mock=False, discover=False)
    result = manager.invoke("slow", {"seconds": 2.0}, _ctx())

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "tool_timeout"
    assert result.error.retryable is True


def test_intent_whitelist_filters_tools() -> None:
    manager = ToolManager(
        use_mock=True,
        allowed_tools_lookup=lambda intent: (
            ["search"] if intent in {IntentType.NEW_QUERY, "new_query"} else []
        ),
    )

    names = [tool.name for tool in manager.list_tools(IntentType.NEW_QUERY)]
    assert names == ["search"]
    assert manager.list_tools(IntentType.CHAT) == []
    schemas = manager.get_schemas(allowed_tools=["search"])
    assert schemas[0]["function"]["name"] == "search"


def test_intent_filter_without_lookup_raises() -> None:
    manager = ToolManager(use_mock=True)
    with pytest.raises(ToolError, match="allowed_tools"):
        manager.list_tools(IntentType.NEW_QUERY)


def test_list_all_when_intent_omitted() -> None:
    manager = ToolManager(use_mock=True)
    assert {tool.name for tool in manager.list_tools()} == {"search", "analysis", "export"}


def test_invoke_enforces_intent_whitelist() -> None:
    manager = ToolManager(
        use_mock=True,
        allowed_tools_lookup=lambda intent: [] if intent == IntentType.CHAT else ["search"],
    )
    missing = manager.invoke("search", {"index": "logs-app"}, _ctx())
    assert missing.ok is False
    assert missing.error is not None
    assert "必须传入 intent" in missing.text

    blocked = manager.invoke("search", {"index": "logs-app"}, _ctx(), intent=IntentType.CHAT)
    assert blocked.ok is False
    assert blocked.error is not None
    assert blocked.error.code == "tool_not_found"

    allowed = manager.invoke(
        "search", {"index": "logs-app", "limit": 1}, _ctx(), intent="new_query"
    )
    assert allowed.ok is True


def test_real_search_requires_es_client() -> None:
    with pytest.raises(ToolError, match="es_client"):
        ToolManager(use_mock=False)
