"""步骤 21：各意图、关联任务过滤、解析失败/低置信度降级为 CHAT。"""

from __future__ import annotations

import json

import pytest

from agent.common.enums import IntentType, TaskStatus
from agent.common.errors import LLMTimeoutError
from agent.common.ids import new_session_id
from agent.config.settings import LLMSettings
from agent.llm.mock_client import MockLLMClient
from agent.llm.resilient import ResilientLLMClient
from tests.task_manage.conftest import build_harness, make_settings


def _intent_json(
    intent: str,
    *,
    related: list[str] | None = None,
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "related_task_ids": related or [],
            "reason": "ok",
            "confidence": confidence,
        }
    )


@pytest.mark.parametrize(
    "intent",
    [
        IntentType.NEW_QUERY,
        IntentType.ANALYSIS,
        IntentType.EXPORT,
        IntentType.CHAT,
        IntentType.UNKNOWN,
    ],
)
def test_recognizes_each_intent(intent: IntentType) -> None:
    llm = MockLLMClient([MockLLMClient.reply(_intent_json(intent.value))])
    harness = build_harness(llm)
    task = harness.manager.create_task(new_session_id(), "查昨天的错误")
    task = harness.manager.step(task.task_id)

    assert task.status is TaskStatus.PLANNING
    assert task.summary.intent is intent
    first = llm.calls[0]
    dumped = "\n".join(message.content for message in first.messages)
    assert "operations" not in dumped
    assert '"content"' in dumped or "查昨天的错误" in dumped
    assert "intending" in dumped or "created" in dumped


def test_related_ids_kept_only_when_in_history() -> None:
    session = new_session_id()
    first_llm = MockLLMClient(
        [
            MockLLMClient.reply(_intent_json("chat")),
            MockLLMClient.reply('{"operations":[]}'),
            MockLLMClient.reply('{"status":"done","conclusion":"你好"}'),
            MockLLMClient.reply(
                '{"satisfied":true,"missing":[],"suggestion":"","final_output":"好"}'
            ),
        ]
    )
    first = build_harness(first_llm)
    prior = first.manager.create_task(session, "随便聊聊")
    first.manager.run(prior.task_id)

    llm = MockLLMClient(
        [MockLLMClient.reply(_intent_json("new_query", related=[prior.task_id, "task_ghost"]))]
    )
    harness = build_harness(llm, settings=first.settings)
    harness.store = first.store  # type: ignore[misc]
    # 复用同一 store / memory，才能看见历史任务
    from agent.task_manage.task_manager import TaskManager

    harness = type(harness)(
        manager=TaskManager(
            store=first.store,
            settings=first.settings,
            context_manager=first.context,
            memory_manager=first.memory,
            tool_manager=first.manager.tools,
            prompt_assembler=first.manager.assembler,
            llm=llm,
        ),
        store=first.store,
        memory=first.memory,
        context=first.context,
        settings=first.settings,
    )
    task = harness.manager.create_task(session, "按刚才的结果再查")
    task = harness.manager.step(task.task_id)

    assert task.summary.related_task_ids == [prior.task_id]


def test_parse_failure_degrades_to_chat() -> None:
    llm = MockLLMClient([MockLLMClient.reply("这不是 JSON"), MockLLMClient.reply("还不是 JSON")])
    harness = build_harness(llm)
    task = harness.manager.create_task(new_session_id(), "嗯？")
    task = harness.manager.step(task.task_id)

    assert task.summary.intent is IntentType.CHAT
    assert task.status is TaskStatus.PLANNING


def test_low_confidence_degrades_to_chat() -> None:
    llm = MockLLMClient([MockLLMClient.reply(_intent_json("new_query", confidence=0.1))])
    harness = build_harness(llm, settings=make_settings(intent_min_confidence=0.4))
    task = harness.manager.create_task(new_session_id(), "也许是查询")
    task = harness.manager.step(task.task_id)

    assert task.summary.intent is IntentType.CHAT


def test_timeout_instance_is_consumed_once() -> None:
    """脚本化超时只能消费一次；测健壮性须手工包 Resilient。"""
    attempts = {"n": 0}

    def flaky(_request: object) -> object:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LLMTimeoutError("timeout")
        return MockLLMClient.reply(_intent_json("new_query"))

    inner = MockLLMClient([flaky])
    client = ResilientLLMClient([inner], LLMSettings(use_mock=True), sleeper=lambda _: None)
    harness = build_harness(client)
    task = harness.manager.create_task(new_session_id(), "查日志")
    task = harness.manager.step(task.task_id)

    assert task.summary.intent is IntentType.NEW_QUERY
    assert attempts["n"] == 2
