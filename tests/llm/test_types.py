"""步骤 9：LLMRequest / LLMResponse / LLMChunk 的字段与校验。"""

import pytest
from pydantic import ValidationError

from agent.common.enums import ContextScope
from agent.llm.base import LLMChunk, LLMRequest, LLMResponse, TokenUsage
from agent.models.message import Message, ToolCall


def test_request_accepts_message_objects() -> None:
    request = LLMRequest(messages=[Message.user("hi")])

    assert request.messages[0].content == "hi"
    assert request.stream is False


def test_request_coerces_openai_shaped_dicts() -> None:
    request = LLMRequest(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(request.messages[0], Message)
    assert request.messages[0].role.value == "user"


def test_request_rejects_string_as_messages() -> None:
    """字符串也是 Sequence，必须排除，否则会按字符拆成一堆非法 message。"""
    with pytest.raises(ValidationError):
        LLMRequest(messages="hi")  # type: ignore[arg-type]


def test_request_rejects_non_mapping_message() -> None:
    with pytest.raises(ValidationError, match="messages\\[0\\]"):
        LLMRequest(messages=[123])  # type: ignore[list-item]


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[Message.user("hi")], unknown=1)  # type: ignore[call-arg]


def test_response_round_trip() -> None:
    response = LLMResponse(
        content="ok",
        tool_calls=[ToolCall(id="call_1", name="search", arguments="{}")],
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model="gpt-4o-mini",
        latency_ms=12.5,
        raw={"id": "chatcmpl-1"},
    )

    restored = LLMResponse.from_dict(response.to_dict())

    assert restored.content == "ok"
    assert restored.tool_calls[0].name == "search"
    assert restored.usage.total_tokens == 3


def test_chunk_defaults() -> None:
    chunk = LLMChunk()

    assert chunk.content == ""
    assert chunk.tool_call_deltas == []
    assert chunk.finish_reason is None


def test_orchestration_fields_stay_on_message_not_request() -> None:
    """请求体拼装用 to_llm_dict()，编排字段不能因为进了 LLMRequest 就泄漏。"""
    message = Message.user(
        "查昨天的错误",
        artifact_refs=["art_1"],
        scope=ContextScope.RELATED,
        source_task_id="task_up",
    )
    request = LLMRequest(messages=[message])

    payload = request.messages[0].to_llm_dict()

    assert payload == {"role": "user", "content": "查昨天的错误"}
    assert "artifact_refs" not in payload
    assert "source_task_id" not in payload
