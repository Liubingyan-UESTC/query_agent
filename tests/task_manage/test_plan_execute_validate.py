"""步骤 23–24：规划校验、执行循环、重试、护栏、CHAT 直答、校验重规划。"""

from __future__ import annotations

import json
from itertools import pairwise

from agent.common.enums import IntentType, TaskStatus
from agent.common.ids import new_session_id
from agent.llm.base import LLMResponse, TokenUsage
from agent.llm.mock_client import MockLLMClient
from agent.models.message import ToolCall
from tests.task_manage.conftest import build_harness, make_settings


def _tool_reply(name: str, arguments: dict[str, object], call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        usage=TokenUsage(),
        model="mock",
        latency_ms=0.0,
    )


def _intent(intent: str = "new_query") -> object:
    return MockLLMClient.reply(
        json.dumps(
            {
                "intent": intent,
                "related_task_ids": [],
                "reason": "q",
                "confidence": 0.9,
            }
        )
    )


def _plan_search(*, limit: int = 2) -> object:
    body = {
        "operations": [
            {
                "index": 0,
                "name": "查日志",
                "tool": "search",
                "args": {"index": "logs-app", "limit": limit},
                "expect": "表",
            }
        ]
    }
    return MockLLMClient.reply(json.dumps(body, ensure_ascii=False))


def test_multi_step_and_window_parts() -> None:
    llm = MockLLMClient(
        [
            _intent(),
            MockLLMClient.reply(
                '{"operations":['
                '{"index":0,"name":"查一页","tool":"search",'
                '"args":{"index":"logs-app","limit":1},"expect":"t"},'
                '{"index":1,"name":"再查","tool":"search",'
                '"args":{"index":"logs-app","limit":2},"expect":"t"}'
                "]}"
            ),
            _tool_reply("search", {"index": "logs-app", "limit": 1}, "c1"),
            MockLLMClient.reply('{"status":"done","conclusion":"第一页就绪"}'),
            _tool_reply("search", {"index": "logs-app", "limit": 2}, "c2"),
            MockLLMClient.reply('{"status":"done","conclusion":"第二页就绪"}'),
            MockLLMClient.reply(
                '{"satisfied":true,"missing":[],"suggestion":"","final_output":"两步都完成"}'
            ),
        ]
    )
    harness = build_harness(llm)
    task = harness.manager.run(
        harness.manager.create_task(new_session_id(), "查昨天 query-agent 的 ERROR").task_id
    )

    assert task.status is TaskStatus.COMPLETED
    assert [item.status.value for item in task.summary.operations] == ["succeeded", "succeeded"]
    window = harness.context.load_window(task.task_id)
    assert window.summary.output == "两步都完成"
    assert sum(1 for item in window.artifacts.values() if item.producer == "search") >= 2
    assert any(item.role.value == "tool" for item in window.content)


def test_tool_retry_then_success() -> None:
    llm = MockLLMClient(
        [
            _intent(),
            _plan_search(limit=1),
            _tool_reply("search", {"index": "no-such-index", "limit": 1}, "bad"),
            _tool_reply("search", {"index": "logs-app", "limit": 1}, "ok"),
            MockLLMClient.reply('{"status":"done","conclusion":"重试成功"}'),
            MockLLMClient.reply(
                '{"satisfied":true,"missing":[],"suggestion":"","final_output":"ok"}'
            ),
        ]
    )
    harness = build_harness(llm)
    task = harness.manager.run(harness.manager.create_task(new_session_id(), "查").task_id)

    assert task.status is TaskStatus.COMPLETED
    assert any(record.status is TaskStatus.RETRYING for record in task.status_history)


def test_retry_exhausted_goes_failed() -> None:
    replies: list[object] = [_intent(), _plan_search(limit=1)]
    for index in range(4):
        replies.append(_tool_reply("search", {"index": "no-such-index", "limit": 1}, f"b{index}"))
    llm = MockLLMClient(replies)
    harness = build_harness(llm)
    task = harness.manager.run(harness.manager.create_task(new_session_id(), "查").task_id)

    assert task.status is TaskStatus.FAILED


def test_max_steps_guard() -> None:
    llm = MockLLMClient(
        [
            _intent(),
            MockLLMClient.reply(
                '{"operations":['
                '{"index":0,"name":"a","tool":"search","args":{"index":"logs-app"},"expect":"x"},'
                '{"index":1,"name":"b","tool":"search","args":{"index":"logs-app"},"expect":"y"}'
                "]}"
            ),
        ]
    )
    harness = build_harness(llm, settings=make_settings(max_steps=1))
    task = harness.manager.run(harness.manager.create_task(new_session_id(), "查").task_id)
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert "步骤数" in task.error.message


def test_chat_zero_tool_direct_answer() -> None:
    llm = MockLLMClient(
        [
            _intent("chat"),
            MockLLMClient.reply('{"operations":[]}'),
            MockLLMClient.reply('{"status":"done","conclusion":"我是查询助手"}'),
            MockLLMClient.reply(
                '{"satisfied":true,"missing":[],"suggestion":"","final_output":"我是查询助手"}'
            ),
        ]
    )
    harness = build_harness(llm)
    task = harness.manager.run(harness.manager.create_task(new_session_id(), "你能做什么").task_id)

    assert task.status is TaskStatus.COMPLETED
    assert task.summary.intent is IntentType.CHAT
    assert task.summary.operations == []
    assert task.summary.output == "我是查询助手"


def test_validate_replan_then_complete() -> None:
    llm = MockLLMClient(
        [
            _intent(),
            _plan_search(limit=1),
            _tool_reply("search", {"index": "logs-app", "limit": 1}),
            MockLLMClient.reply('{"status":"done","conclusion":"先交一版"}'),
            MockLLMClient.reply(
                '{"satisfied":false,"missing":["需要更多行"],"suggestion":"再查一次","final_output":""}'
            ),
            _plan_search(limit=2),
            _tool_reply("search", {"index": "logs-app", "limit": 2}, "c2"),
            MockLLMClient.reply('{"status":"done","conclusion":"补齐了"}'),
            MockLLMClient.reply(
                '{"satisfied":true,"missing":[],"suggestion":"","final_output":"完整结果"}'
            ),
        ]
    )
    harness = build_harness(llm)
    task = harness.manager.run(harness.manager.create_task(new_session_id(), "查错误").task_id)

    assert task.status is TaskStatus.COMPLETED
    assert task.summary.output == "完整结果"
    statuses = [record.status for record in task.status_history]
    assert any(
        earlier is TaskStatus.VALIDATING and later is TaskStatus.PLANNING
        for earlier, later in pairwise(statuses)
    )
