"""步骤 9 测试辅助：用 httpx2.MockTransport 打桩 HTTP 层。"""

import json
from collections.abc import Callable

import httpx2
import pytest
from pydantic import SecretStr

from agent.config.settings import LLMProfile
from agent.llm.openai_client import OpenAICompatClient

CompletionHandler = Callable[[httpx2.Request], httpx2.Response]


def default_profile(**overrides: object) -> LLMProfile:
    kwargs: dict[str, object] = {
        "name": "primary",
        "base_url": "https://example.test/v1",
        "api_key": SecretStr("sk-test"),
        "model": "gpt-4o-mini",
        "timeout": 15.0,
        "temperature": 0.2,
        "max_tokens": 128,
    }
    kwargs.update(overrides)
    return LLMProfile(**kwargs)  # type: ignore[arg-type]


def completion_payload(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, object]] | None = None,
    finish_reason: str | None = "stop",
    model: str = "gpt-4o-mini",
    usage: dict[str, int] | None = None,
    include_usage: bool = True,
    choices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if choices is None:
        message: dict[str, object] = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        choices = [
            {"index": 0, "message": message, "finish_reason": finish_reason},
        ]
    payload: dict[str, object] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": choices,
    }
    if include_usage:
        payload["usage"] = usage or {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
    return payload


def json_handler(
    payload: dict[str, object],
    *,
    status: int = 200,
) -> CompletionHandler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=payload)

    return handler


def sse_body(chunks: list[dict[str, object]]) -> str:
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def sse_handler(chunks: list[dict[str, object]]) -> CompletionHandler:
    text = sse_body(chunks)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=text)

    return handler


def text_chunk(content: str | None, *, finish_reason: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }


class RecordingTransport:
    """记下每一发请求，并交给内部 handler 出响应。"""

    def __init__(self, handler: CompletionHandler) -> None:
        self.handler = handler
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last_body(self) -> dict[str, object]:
        return json.loads(self.requests[-1].content)


def make_client(
    handler: CompletionHandler,
    *,
    profile: LLMProfile | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[OpenAICompatClient, RecordingTransport]:
    recorder = RecordingTransport(handler)
    http_client = httpx2.Client(transport=httpx2.MockTransport(recorder))
    client = OpenAICompatClient(
        profile or default_profile(),
        http_client=http_client,
        clock=clock,
    )
    return client, recorder


@pytest.fixture
def profile() -> LLMProfile:
    return default_profile()
