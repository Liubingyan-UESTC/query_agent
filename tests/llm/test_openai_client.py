"""步骤 9 验收：打桩 HTTP 层，验证请求拼装、响应解析、异常映射与流式顺序。"""

import json

import httpx2
import openai
import pytest

from agent.common.enums import ContextScope
from agent.common.errors import LLMError, LLMRateLimitError, LLMTimeoutError
from agent.llm.base import LLMChunk, LLMRequest
from agent.llm.openai_client import OpenAICompatClient, _dump, translate_openai_error
from agent.models.message import Message
from tests.llm.conftest import (
    completion_payload,
    default_profile,
    json_handler,
    make_client,
    sse_handler,
    text_chunk,
)


class StepClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _chat(handler, request=None, **client_kwargs):
    client, recorder = make_client(handler, **client_kwargs)
    try:
        response = client.chat(request or LLMRequest(messages=[Message.user("hi")]))
    finally:
        client.close()
    return response, recorder


# ============================================================ 请求体拼装


def test_chat_sends_openai_compatible_body() -> None:
    response, recorder = _chat(json_handler(completion_payload()))

    assert response.content == "hello"
    body = recorder.last_body
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128
    assert body["stream"] is False
    assert "tools" not in body
    assert "response_format" not in body
    assert recorder.requests[0].headers["authorization"] == "Bearer sk-test"


def test_orchestration_fields_never_appear_in_http_body() -> None:
    message = Message.user(
        "查昨天的错误",
        artifact_refs=["art_1"],
        scope=ContextScope.RELATED,
        source_task_id="task_up",
        meta={"channel": "web"},
    )
    _, recorder = _chat(
        json_handler(completion_payload()),
        request=LLMRequest(messages=[message]),
    )

    dumped = json.dumps(recorder.last_body)
    assert "artifact_refs" not in dumped
    assert "source_task_id" not in dumped
    assert "message_id" not in dumped
    assert "channel" not in dumped


def test_request_overrides_profile_defaults() -> None:
    _, recorder = _chat(
        json_handler(completion_payload()),
        request=LLMRequest(
            messages=[Message.user("hi")],
            model="intent-model",
            temperature=0.7,
            max_tokens=32,
            tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
            response_format={"type": "json_object"},
        ),
    )

    body = recorder.last_body
    assert body["model"] == "intent-model"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 32
    assert body["tools"][0]["function"]["name"] == "search"
    assert body["response_format"] == {"type": "json_object"}


def test_empty_tools_are_omitted() -> None:
    _, recorder = _chat(
        json_handler(completion_payload()),
        request=LLMRequest(messages=[Message.user("hi")], tools=[]),
    )

    assert "tools" not in recorder.last_body


# ============================================================ 响应解析


def test_chat_parses_tool_calls_and_null_content() -> None:
    payload = completion_payload(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"err"}'},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    )
    response, _ = _chat(json_handler(payload))

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].parsed_arguments() == {"q": "err"}
    assert response.usage.total_tokens == 18
    assert response.raw is not None
    assert response.raw["id"] == "chatcmpl-1"


def test_missing_usage_becomes_zeros() -> None:
    response, _ = _chat(json_handler(completion_payload(include_usage=False)))

    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0


def test_latency_ms_comes_from_injected_clock() -> None:
    response, _ = _chat(
        json_handler(completion_payload()),
        clock=StepClock([1.0, 1.25]),
    )

    assert response.latency_ms == pytest.approx(250.0)


def test_empty_choices_raises_retryable_llm_error() -> None:
    with pytest.raises(LLMError, match="choices") as caught:
        _chat(json_handler(completion_payload(choices=[])))

    assert caught.value.retryable is True


def test_missing_message_raises_retryable_llm_error() -> None:
    payload = completion_payload(choices=[{"index": 0, "finish_reason": "stop"}])
    with pytest.raises(LLMError, match="message") as caught:
        _chat(json_handler(payload))

    assert caught.value.retryable is True


def test_malformed_tool_call_raises_llm_error() -> None:
    payload = completion_payload(
        content=None,
        tool_calls=[{"id": "call_1", "type": "function", "function": {}}],
    )
    with pytest.raises(LLMError, match="tool_calls") as caught:
        _chat(json_handler(payload))

    assert caught.value.retryable is False


def test_response_model_falls_back_to_request_when_missing() -> None:
    payload = completion_payload(model="")
    # 兼容网关可能把 model 留空；SDK 会收成空串，客户端应回落
    payload["model"] = None
    response, _ = _chat(
        json_handler(payload),
        request=LLMRequest(messages=[Message.user("hi")], model="intent-model"),
    )

    assert response.model == "intent-model"


# ============================================================ 调用护栏


def test_chat_rejects_stream_flag() -> None:
    client, _ = make_client(json_handler(completion_payload()))
    with pytest.raises(ValueError, match="stream_chat"):
        client.chat(LLMRequest(messages=[Message.user("hi")], stream=True))
    client.close()


def test_empty_messages_are_rejected() -> None:
    client, _ = make_client(json_handler(completion_payload()))
    with pytest.raises(ValueError, match="messages"):
        client.chat(LLMRequest(messages=[]))
    client.close()


def test_sdk_retries_are_disabled() -> None:
    """步骤 10 独占重试。429 在本层只应打一枪。"""
    calls = {"n": 0}

    def once(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(LLMRateLimitError):
        _chat(once)

    assert calls["n"] == 1


# ============================================================ 异常映射（HTTP）


@pytest.mark.parametrize(
    ("status", "body", "error_cls", "retryable"),
    [
        (429, {"error": {"message": "slow down"}}, LLMRateLimitError, True),
        (500, {"error": {"message": "boom"}}, LLMError, True),
        (401, {"error": {"message": "bad key"}}, LLMError, False),
        (400, {"error": {"message": "bad req"}}, LLMError, False),
        (408, {"error": {"message": "timed out"}}, LLMTimeoutError, True),
    ],
)
def test_http_errors_are_translated(status, body, error_cls, retryable) -> None:
    with pytest.raises(error_cls) as caught:
        _chat(json_handler(body, status=status))

    assert caught.value.retryable is retryable
    assert caught.value.detail is not None
    assert str(status) in caught.value.detail or caught.value.detail


def test_timeout_exception_becomes_llm_timeout() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.TimeoutException("timed out")

    with pytest.raises(LLMTimeoutError) as caught:
        _chat(boom)

    assert caught.value.retryable is True


def test_connection_error_is_retryable() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(LLMError) as caught:
        _chat(boom)

    assert caught.value.retryable is True
    assert caught.value.code == "llm_error"


def test_error_detail_is_clipped() -> None:
    huge = "x" * 2000
    with pytest.raises(LLMError) as caught:
        _chat(json_handler({"error": {"message": huge}}, status=400))

    assert caught.value.detail is not None
    assert caught.value.detail.endswith("…")
    assert len(caught.value.detail) == 501


# ============================================================ 流式


def test_stream_chat_preserves_chunk_order() -> None:
    chunks = [
        text_chunk(""),
        text_chunk("Hel"),
        text_chunk("lo"),
        text_chunk(None, finish_reason="stop"),
    ]
    client, recorder = make_client(sse_handler(chunks))
    try:
        received = list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    finally:
        client.close()

    assert [item.content for item in received] == ["", "Hel", "lo", ""]
    assert received[-1].finish_reason == "stop"
    assert recorder.last_body["stream"] is True


def test_stream_chat_passes_through_tool_call_deltas() -> None:
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    client, _ = make_client(sse_handler(chunks))
    try:
        received = list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    finally:
        client.close()

    assert received[0].tool_call_deltas[0]["id"] == "call_1"
    assert received[0].tool_call_deltas[0]["function"]["name"] == "search"
    assert received[1].tool_call_deltas[0]["function"]["arguments"] == '{"q"'
    assert received[-1].finish_reason == "tool_calls"


def test_stream_empty_choices_and_null_delta() -> None:
    chunks = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
        },
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": None, "finish_reason": "stop"}],
        },
    ]
    client, _ = make_client(sse_handler(chunks))
    try:
        received = list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    finally:
        client.close()

    assert received[0] == LLMChunk(content="", model="gpt-4o-mini")
    assert received[1].content == ""
    assert received[1].finish_reason == "stop"


def test_stream_http_error_is_translated() -> None:
    client, _ = make_client(json_handler({"error": {"message": "slow"}}, status=429))
    with pytest.raises(LLMRateLimitError):
        list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    client.close()


def test_stream_malformed_sse_is_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        first = text_chunk("A")
        text = "data: " + json.dumps(first) + "\n\ndata: NOTJSON\n\ndata: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=text)

    client, _ = make_client(handler)
    with pytest.raises(LLMError, match="流式") as caught:
        list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    client.close()

    assert caught.value.retryable is True


def test_stream_parse_error_is_not_rewrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = make_client(sse_handler([text_chunk("A")]))

    def boom(_self: OpenAICompatClient, _chunk: object) -> LLMChunk:
        raise LLMError("解析失败", retryable=False)

    monkeypatch.setattr(OpenAICompatClient, "_parse_chunk", boom)
    with pytest.raises(LLMError, match="解析失败") as caught:
        list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    client.close()

    assert caught.value.message == "解析失败"


def test_stream_openai_error_during_iteration() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")

    class ExplodingStream:
        def __iter__(self) -> object:
            raise openai.APITimeoutError(request)

        def close(self) -> None:
            return None

    client, _ = make_client(json_handler(completion_payload()))
    client._sdk.chat.completions.create = lambda **_kwargs: ExplodingStream()  # type: ignore[method-assign]
    with pytest.raises(LLMTimeoutError):
        list(client.stream_chat(LLMRequest(messages=[Message.user("hi")])))
    client.close()


def test_stream_rejects_empty_messages() -> None:
    client, _ = make_client(sse_handler([text_chunk("A")]))
    with pytest.raises(ValueError, match="messages"):
        list(client.stream_chat(LLMRequest(messages=[])))
    client.close()


# ============================================================ 翻译函数与工具


def test_translate_passes_through_llm_error() -> None:
    original = LLMTimeoutError("already")

    assert translate_openai_error(original) is original


def test_translate_generic_openai_error() -> None:
    translated = translate_openai_error(openai.OpenAIError("weird"))

    assert isinstance(translated, LLMError)
    assert translated.retryable is False


def test_translate_api_error() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    exc = openai.APIError("weird", request, body=None)
    translated = translate_openai_error(exc)

    assert translated.retryable is True


def test_translate_status_with_string_body() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(400, text="plain", request=request)
    exc = openai.APIStatusError("bad", response=response, body="plain")
    translated = translate_openai_error(exc)

    assert translated.retryable is False
    assert translated.detail is not None
    assert "plain" in translated.detail


def test_translate_status_429_without_rate_limit_subclass() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(429, json={"error": {"message": "x"}}, request=request)
    exc = openai.APIStatusError("limited", response=response, body=None)
    translated = translate_openai_error(exc)

    assert isinstance(translated, LLMRateLimitError)


def test_translate_unknown_exception() -> None:
    translated = translate_openai_error(RuntimeError("boom"))

    assert isinstance(translated, LLMError)
    assert translated.retryable is False


def test_dump_accepts_plain_dict_and_rejects_other() -> None:
    assert _dump({"a": 1}) == {"a": 1}
    with pytest.raises(TypeError, match="dict"):
        _dump(object())

    class NotADict:
        def model_dump(self, *, mode: str = "json") -> list[str]:
            return ["not", "a", "dict"]

    with pytest.raises(TypeError, match="dict"):
        _dump(NotADict())


def test_construct_without_injected_http_client() -> None:
    """覆盖 http_client 缺省分支；不发起真实请求。"""
    client = OpenAICompatClient(default_profile())
    client.close()


def test_context_manager_closes() -> None:
    client, _ = make_client(json_handler(completion_payload()))
    with client as entered:
        assert entered is client
        entered.chat(LLMRequest(messages=[Message.user("hi")]))
