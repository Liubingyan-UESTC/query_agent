"""步骤 14：token 计数可插拔，tiktoken 缺失时走字符估算。"""

from __future__ import annotations

from agent.context_manage.token_counter import (
    CharEstimateCounter,
    default_counter,
)
from agent.models.message import Message, ToolCall


def test_empty_text_is_zero() -> None:
    assert CharEstimateCounter().count_text("") == 0


def test_ascii_uses_divisor_of_four() -> None:
    counter = CharEstimateCounter()

    assert counter.count_text("abcd") == 1
    assert counter.count_text("abcde") == 2


def test_cjk_counts_one_per_char() -> None:
    counter = CharEstimateCounter()

    assert counter.count_text("查询") == 2
    assert counter.count_text("查询abcd") == 3


def test_count_messages_includes_tool_calls() -> None:
    counter = CharEstimateCounter()
    messages = [
        Message.system("你是查询 Agent。"),
        Message.assistant(
            "",
            tool_calls=[ToolCall(id="call_1", name="search", arguments='{"q":"x"}')],
        ),
        Message.tool_result("ok", tool_call_id="call_1"),
    ]

    total = counter.count_messages(messages)
    only_system = counter.count_messages(messages[:1])

    assert total > only_system


def test_default_counter_never_raises() -> None:
    counter = default_counter()
    assert counter.count_text("hello 查询") >= 1
