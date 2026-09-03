"""OpenAI 兼容客户端，以及"重试 + 降级"的包装器。

两条设计决定：

1. **SDK 的内置重试关掉**（``max_retries=0``）。重试策略集中在 :class:`ResilientLLMClient`
   一处；两层各自退避的话，实际重试次数会变成乘法，超时表现无从预测。
2. **错误先翻译再抛**：``translate_openai_error`` 按状态码判定 ``retryable``，
   上层只认这个布尔值，不必再去认识 SDK 的异常类型。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from agent.config import LLMProfile
from agent.errors import LLMError
from agent.llm.base import BaseLLMClient, LLMRequest, LLMResponse, TokenUsage
from agent.models import ToolCall

__all__ = ["OpenAICompatClient", "ResilientLLMClient", "translate_openai_error"]

# 异常 detail 的截断长度：SDK 有时会把整个响应体塞进 message，日志会被撑爆
_DETAIL_LIMIT = 500

# 指数退避：0.5s、1s、2s…… 封顶 8s
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 8.0

# 这些状态码是瞬时故障，值得重试
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def translate_openai_error(exc: BaseException) -> LLMError:
    """把 openai SDK 的异常翻译成带 ``retryable`` 语义的 :class:`LLMError`。"""
    status = getattr(exc, "status_code", None)
    detail = str(exc)[:_DETAIL_LIMIT]
    if isinstance(status, int):
        retryable = status in _RETRYABLE_STATUS
        return LLMError(f"模型返回 HTTP {status}：{detail}", retryable=retryable, detail=detail)
    # 连接错误/超时没有状态码，一律按可重试处理
    return LLMError(f"模型调用失败：{detail}", retryable=True, detail=detail)


class OpenAICompatClient(BaseLLMClient):
    """调用任意 OpenAI 兼容端点。

    ``client_factory`` 是给测试用的注入点：传入一个假的 client 就能在不触网的情况下
    验证请求装配与响应解析。
    """

    def __init__(
        self,
        profile: LLMProfile,
        *,
        client_factory: Callable[[LLMProfile], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.profile = profile
        self._clock = clock or time.perf_counter
        self._client = (client_factory or _default_client)(profile)

    def chat(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": request.model or self.profile.model,
            "messages": request.to_llm_messages(),
            "temperature": (
                request.temperature if request.temperature is not None else self.profile.temperature
            ),
            "max_tokens": request.max_tokens or self.profile.max_tokens,
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.response_format:
            payload["response_format"] = request.response_format

        started = self._clock()
        try:
            completion = self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise translate_openai_error(exc) from exc
        return _parse_completion(completion, latency_ms=(self._clock() - started) * 1000)

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()


def _default_client(profile: LLMProfile) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=profile.api_key.get_secret_value(),
        base_url=profile.base_url,
        timeout=profile.timeout,
        # 重试交给 ResilientLLMClient，见模块 docstring
        max_retries=0,
    )


def _parse_completion(completion: Any, *, latency_ms: float) -> LLMResponse:
    """把 SDK 对象收成自有的 :class:`LLMResponse`。"""
    payload = (
        completion.model_dump(mode="json")
        if hasattr(completion, "model_dump")
        else dict(completion)
    )
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("模型返回中没有 choices", retryable=True, detail=payload)
    choice = choices[0]
    message = choice.get("message") or {}
    raw_calls = message.get("tool_calls") or []
    try:
        tool_calls = [ToolCall.from_llm_dict(item) for item in raw_calls]
    except ValueError as exc:
        raise LLMError(f"tool_calls 结构不合法：{exc}", retryable=True, detail=raw_calls) from exc

    usage_payload = payload.get("usage") or {}
    return LLMResponse(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason"),
        model=payload.get("model") or "",
        usage=TokenUsage(
            prompt_tokens=usage_payload.get("prompt_tokens") or 0,
            completion_tokens=usage_payload.get("completion_tokens") or 0,
            total_tokens=usage_payload.get("total_tokens") or 0,
        ),
        latency_ms=latency_ms,
    )


class ResilientLLMClient(BaseLLMClient):
    """按"单端点重试 → 换下一个端点"的顺序保证可用性。

    - 同一端点内最多尝试 ``max_retries + 1`` 次，仅在 ``retryable=True`` 时退避重试；
    - 端点耗尽后切到下一个降级模型；
    - 不可重试的错误（如 401、参数非法）**立即抛出**，不浪费剩余端点。
    """

    def __init__(
        self,
        clients: Sequence[BaseLLMClient],
        *,
        max_retries: int = 2,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("ResilientLLMClient 至少需要一个底层客户端")
        self._clients = list(clients)
        self._max_retries = max_retries
        self._sleep = sleeper or time.sleep

    def chat(self, request: LLMRequest) -> LLMResponse:
        last_error: LLMError | None = None
        for client in self._clients:
            for attempt in range(self._max_retries + 1):
                try:
                    return client.chat(request)
                except LLMError as exc:
                    if not exc.retryable:
                        raise
                    last_error = exc
                    if attempt < self._max_retries:
                        self._sleep(min(_BACKOFF_CAP, _BACKOFF_BASE * (2**attempt)))
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        for client in self._clients:
            client.close()
