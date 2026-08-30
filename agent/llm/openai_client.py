"""基于 openai SDK 的兼容模式客户端。

只负责：把 `LLMRequest` 拼成 chat.completions 载荷、把响应收成 `LLMResponse` /
`LLMChunk`、把 SDK 异常翻译成步骤 3 的 `LLMError` 子类。

重试、降级、按用途路由属于步骤 10 的 `ResilientLLMClient`。因此这里把 SDK
自带的 `max_retries` 钉死为 0——两层各自退避会让实际重试次数变成乘法。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any, Self

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from agent.common.errors import LLMError, LLMRateLimitError, LLMTimeoutError
from agent.config.settings import LLMProfile
from agent.llm.base import BaseLLMClient, LLMChunk, LLMRequest, LLMResponse, TokenUsage
from agent.models.message import ToolCall

__all__ = ["OpenAICompatClient", "translate_openai_error"]

_DETAIL_LIMIT = 500


def translate_openai_error(exc: BaseException) -> LLMError:
    """把 openai SDK 异常翻译为带 `retryable` 语义的 `LLMError`。

    已经是 `LLMError` 的原样返回，避免解析阶段抛出的错误被再包一层。
    未识别的异常视为不可重试：那是程序缺陷，退避没有意义。
    """
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError("模型请求超时", detail=str(exc))
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError("模型服务触发限流", detail=_status_detail(exc))
    if isinstance(exc, openai.APIStatusError):
        return _from_status(exc)
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("模型服务连接失败", retryable=True, detail=str(exc))
    if isinstance(exc, openai.APIError):
        return LLMError("模型服务返回无法处理的响应", retryable=True, detail=str(exc))
    if isinstance(exc, openai.OpenAIError):
        return LLMError(f"模型调用失败：{exc}", retryable=False, detail=str(exc))
    return LLMError(f"模型调用失败：{exc}", retryable=False, detail=str(exc))


def _from_status(exc: openai.APIStatusError) -> LLMError:
    status = exc.status_code
    detail = _status_detail(exc)
    if status == 408:
        return LLMTimeoutError("模型请求超时", detail=detail)
    if status == 429:
        return LLMRateLimitError("模型服务触发限流", detail=detail)
    if status >= 500:
        return LLMError("模型服务内部错误", retryable=True, detail=detail)
    return LLMError("模型请求被拒绝", retryable=False, detail=detail)


def _status_detail(exc: openai.APIStatusError) -> str:
    parts = [f"status={exc.status_code}", str(exc)]
    if exc.body is not None:
        parts.append(exc.body if isinstance(exc.body, str) else repr(exc.body))
    return _clip("; ".join(parts))


def _clip(value: object) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= _DETAIL_LIMIT:
        return text
    return text[:_DETAIL_LIMIT] + "…"


class OpenAICompatClient(BaseLLMClient):
    """OpenAI 兼容 chat.completions 客户端。

    `http_client` 可注入，供单测用 `httpx2.MockTransport` 打桩 HTTP 层。
    `clock` 可注入，供单测断言 `latency_ms` 而不真实 sleep。
    """

    def __init__(
        self,
        profile: LLMProfile,
        *,
        http_client: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.profile = profile
        self._clock = clock or time.perf_counter
        kwargs: dict[str, Any] = {
            "api_key": profile.api_key.get_secret_value(),
            "base_url": profile.base_url,
            "timeout": profile.timeout,
            "max_retries": 0,
        }
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._sdk = OpenAI(**kwargs)

    def chat(self, request: LLMRequest) -> LLMResponse:
        if request.stream:
            raise ValueError("chat() 不接受 stream=True，请调用 stream_chat()")
        self._require_messages(request)
        started = self._clock()
        try:
            completion = self._sdk.chat.completions.create(
                **self._build_kwargs(request, stream=False)
            )
        except openai.OpenAIError as exc:
            raise translate_openai_error(exc) from exc
        latency_ms = (self._clock() - started) * 1000.0
        return self._parse_completion(completion, request, latency_ms)

    def stream_chat(self, request: LLMRequest) -> Iterator[LLMChunk]:
        self._require_messages(request)
        try:
            stream = self._sdk.chat.completions.create(**self._build_kwargs(request, stream=True))
        except openai.OpenAIError as exc:
            raise translate_openai_error(exc) from exc
        try:
            for chunk in stream:
                yield self._parse_chunk(chunk)
        except LLMError:
            raise
        except openai.OpenAIError as exc:
            raise translate_openai_error(exc) from exc
        except Exception as exc:
            raise LLMError("读取流式响应失败", retryable=True, detail=str(exc)) from exc
        finally:
            stream.close()

    def close(self) -> None:
        super().close()
        self._sdk.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object,
    ) -> None:
        self.close()

    def _resolved_model(self, request: LLMRequest) -> str:
        return request.model or self.profile.model

    def _build_kwargs(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._resolved_model(request),
            "messages": [message.to_llm_dict() for message in request.messages],
            "temperature": (
                self.profile.temperature if request.temperature is None else request.temperature
            ),
            "max_tokens": (
                self.profile.max_tokens if request.max_tokens is None else request.max_tokens
            ),
            "stream": stream,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        return kwargs

    def _require_messages(self, request: LLMRequest) -> None:
        if not request.messages:
            raise ValueError("LLMRequest.messages 不能为空")

    def _parse_completion(
        self,
        completion: ChatCompletion,
        request: LLMRequest,
        latency_ms: float,
    ) -> LLMResponse:
        raw = _dump(completion)
        choices = completion.choices or []
        if not choices:
            raise LLMError(
                "模型响应缺少 choices",
                retryable=True,
                detail=_clip(raw),
            )
        message = choices[0].message
        if message is None:
            raise LLMError(
                "模型响应缺少 message",
                retryable=True,
                detail=_clip(raw),
            )
        try:
            tool_calls = [
                ToolCall.from_llm_dict(_dump(item)) for item in (message.tool_calls or [])
            ]
        except ValueError as exc:
            raise LLMError(
                "模型响应中的 tool_calls 无法解析",
                retryable=False,
                detail=str(exc),
            ) from exc
        usage = completion.usage
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choices[0].finish_reason,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            ),
            model=completion.model or self._resolved_model(request),
            latency_ms=latency_ms,
            raw=raw,
        )

    def _parse_chunk(self, chunk: ChatCompletionChunk) -> LLMChunk:
        choices = chunk.choices or []
        if not choices:
            return LLMChunk(content="", model=chunk.model)
        choice = choices[0]
        delta = choice.delta
        content = ""
        tool_call_deltas: list[dict[str, Any]] = []
        if delta is not None:
            content = delta.content or ""
            if delta.tool_calls:
                tool_call_deltas = [_dump(item) for item in delta.tool_calls]
        return LLMChunk(
            content=content,
            tool_call_deltas=tool_call_deltas,
            finish_reason=choice.finish_reason,
            model=chunk.model,
        )


def _dump(value: object) -> dict[str, Any]:
    """把 SDK 模型打成 JSON 友好 dict；非模型对象原样要求已是 mapping。"""
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        dumped = dumper(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return value
    raise TypeError(f"无法把 {type(value).__name__} 转为 dict")
