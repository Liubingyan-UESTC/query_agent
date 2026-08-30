"""步骤 24：MockLLM + MockTool 端到端闭环。"""

from __future__ import annotations

import json

import pytest

from agent.common.enums import ContextScope, TaskStatus
from agent.common.errors import TaskStateError
from agent.common.ids import new_session_id
from agent.llm.base import LLMResponse, TokenUsage
from agent.llm.mock_client import MockLLMClient
from agent.models.message import ToolCall
from agent.task_manage.state_machine import TaskStateMachine
from tests.task_manage.conftest import build_harness


def _tool_reply(arguments: dict[str, object], call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name="search", arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        usage=TokenUsage(),
        model="mock",
        latency_ms=0.0,
    )


def _happy_path(related: list[str] | None = None) -> list[object]:
    return [
        MockLLMClient.reply(
            json.dumps(
                {
                    "intent": "new_query",
                    "related_task_ids": related or [],
                    "reason": "q",
                    "confidence": 0.9,
                }
            )
        ),
        MockLLMClient.reply(
            '{"operations":[{"index":0,"name":"查日志","tool":"search",'
            '"args":{"index":"logs-app","limit":2},"expect":"表"}]}'
        ),
        _tool_reply({"index": "logs-app", "limit": 2}),
        MockLLMClient.reply('{"status":"done","conclusion":"已取回日志"}'),
        MockLLMClient.reply(
            '{"satisfied":true,"missing":[],"suggestion":"","final_output":"查询完成"}'
        ),
    ]


def test_created_to_completed() -> None:
    harness = build_harness(MockLLMClient(_happy_path()))
    task = harness.manager.create_task(new_session_id(), "查昨天的 ERROR")
    assert task.status is TaskStatus.CREATED
    task = harness.manager.run(task.task_id)

    assert task.status is TaskStatus.COMPLETED
    assert harness.manager.get_result(task.task_id)["output"] == "查询完成"
    archived = harness.memory.working(task.session_id).list_task_ids()
    assert task.task_id in archived
    statuses = [record.status for record in task.status_history]
    assert statuses[0] is TaskStatus.CREATED
    assert TaskStatus.INTENDING in statuses
    assert TaskStatus.COMPLETED in statuses


def test_second_task_injects_first_and_does_not_rearchive_related() -> None:
    session = new_session_id()
    first = build_harness(MockLLMClient(_happy_path()))
    one = first.manager.create_task(session, "查昨天的 ERROR")
    one = first.manager.run(one.task_id)
    assert one.status is TaskStatus.COMPLETED

    llm = MockLLMClient(_happy_path(related=[one.task_id]))
    from agent.task_manage.task_manager import TaskManager

    manager = TaskManager(
        store=first.store,
        settings=first.settings,
        context_manager=first.context,
        memory_manager=first.memory,
        tool_manager=first.manager.tools,
        prompt_assembler=first.manager.assembler,
        llm=llm,
    )
    two = manager.create_task(session, "继续看刚才的结果")
    two = manager.run(two.task_id)

    assert two.status is TaskStatus.COMPLETED
    assert two.summary.related_task_ids == [one.task_id]
    live = first.context.load_window(two.task_id)
    related_msgs = [item for item in live.content if item.scope is ContextScope.RELATED]
    assert related_msgs
    assert any(item.source_task_id == one.task_id for item in related_msgs)

    archived = first.memory.working(session).get_content_history(two.task_id)
    assert all(item.scope is ContextScope.CURRENT for item in archived)
    first_arts = {
        item.artifact_id for item in first.memory.working(session).get_artifact_history(one.task_id)
    }
    second_arts = {
        item.artifact_id for item in first.memory.working(session).get_artifact_history(two.task_id)
    }
    assert first_arts.isdisjoint(second_arts)


def test_failed_and_canceled_branches() -> None:
    bad = build_harness(
        MockLLMClient(
            [
                MockLLMClient.reply(
                    '{"intent":"new_query","related_task_ids":[],"reason":"q","confidence":0.9}'
                ),
                MockLLMClient.reply(
                    '{"operations":[{"index":0,"name":"非法","tool":"not_a_tool",'
                    '"args":{},"expect":"x"}]}'
                ),
                MockLLMClient.reply(
                    '{"operations":[{"index":0,"name":"非法","tool":"not_a_tool",'
                    '"args":{},"expect":"x"}]}'
                ),
            ]
        )
    )
    failed = bad.manager.run(bad.manager.create_task(new_session_id(), "坏规划").task_id)
    assert failed.status is TaskStatus.FAILED
    assert failed.summary.output
    assert failed.task_id in bad.memory.working(failed.session_id).list_task_ids()

    harness = build_harness(MockLLMClient())
    task = harness.manager.create_task(new_session_id(), "要取消")
    task = harness.manager.cancel(task.task_id, "用户取消")
    assert task.status is TaskStatus.CANCELED
    assert task.summary.output


def test_illegal_transition_is_blocked() -> None:
    machine = TaskStateMachine()
    with pytest.raises(TaskStateError, match="非法状态转移"):
        machine.assert_transition(TaskStatus.CREATED, TaskStatus.COMPLETED)
