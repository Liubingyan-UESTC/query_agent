"""带重试、降级与按用途路由的 LLM 装饰器。

同一底层客户端只对 `retryable=True` 的异常做指数退避；重试耗尽后再按
`fallbacks` 换下一个。不可重试的错误立即抛出——换端点也治不好 400。

SDK 层重试已关闭，本模块是唯一的退避入口，`LLMProfile.max_retries`
表示「首次失败之后」的重试次数。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from typing import TypeVar

from agent.common.errors import LLMError
from agent.common.logging import get_logger
from agent.config.settings import LLMProfile, LLMSettings
from agent.llm.base import BaseLLMClient, LLMChunk, LLMRequest, LLMResponse
from agent.models.base import AgentModel

__all__ = ["LLMCallStats", "ResilientLLMClient"]

_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 8.0

T = TypeVar("T")

logger = get_logger(__name__)


class LLMCallStats(AgentModel):
    """一次逻辑调用（含重试与降级）的埋点。"""

    model: str
    profile_name: str
    purpose: str | None = None
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retry_count: int = 0
    fallback_index: int = 0
    ok: bool


class ResilientLLMClient(BaseLLMClient):
    """装饰一组底层客户端：主模型优先，失败后按链降级。"""

    def __init__(
        self,
        clients: Sequence[BaseLLMClient],
        settings: LLMSettings,
        *,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("ResilientLLMClient 至少需要一个底层客户端")
        self._clients = list(clients)
        self.settings = settings
        self._sleeper = sleeper or time.sleep
        self._clock = clock or time.perf_counter
        self.last_stats: LLMCallStats | None = None

    def chat(self, request: LLMRequest) -> LLMResponse:
        if request.stream:
            raise ValueError("chat() 不接受 stream=True，请调用 stream_chat()")
        return self._run(request, lambda client, req: client.chat(req))

    def stream_chat(self, request: LLMRequest) -> Iterator[LLMChunk]:
        # 生成器要先 next 一次，创建阶段的超时才能被 _run 捕获并降级
        iterator = self._run(request, _open_stream)
        yield from iterator

    def close(self) -> None:
        super().close()
        for client in self._clients:
            client.close()

    def _prepare(self, request: LLMRequest) -> LLMRequest:
        if request.model is not None or request.purpose is None:
            return request
        return request.model_copy(update={"model": self.settings.model_for(request.purpose)})

    def _profile_at(self, index: int) -> LLMProfile:
        chain = self.settings.profile_chain()
        if index < len(chain):
            return chain[index]
        return self.settings.primary

    def _run(self, request: LLMRequest, invoke: Callable[[BaseLLMClient, LLMRequest], T]) -> T:
        prepared = self._prepare(request)
        started = self._clock()
        last_error: LLMError | None = None
        retries = 0

        for fallback_index, client in enumerate(self._clients):
            profile = self._profile_at(fallback_index)
            attempts = profile.max_retries + 1
            for attempt in range(attempts):
                try:
                    result = invoke(client, prepared)
                except LLMError as exc:
                    last_error = exc
                    if not exc.retryable:
                        self._publish(
                            prepared,
                            profile,
                            fallback_index,
                            retries,
                            started,
                            ok=False,
                        )
                        raise
                    if attempt + 1 < attempts:
                        self._sleeper(min(_BACKOFF_CAP, _BACKOFF_BASE * (2**attempt)))
                        retries += 1
                    continue
                response = result if isinstance(result, LLMResponse) else None
                self._publish(
                    prepared,
                    profile,
                    fallback_index,
                    retries,
                    started,
                    ok=True,
                    response=response,
                )
                return result

        assert last_error is not None
        self._publish(
            prepared,
            self._profile_at(len(self._clients) - 1),
            len(self._clients) - 1,
            retries,
            started,
            ok=False,
        )
        raise last_error

    def _publish(
        self,
        request: LLMRequest,
        profile: LLMProfile,
        fallback_index: int,
        retry_count: int,
        started: float,
        *,
        ok: bool,
        response: LLMResponse | None = None,
    ) -> None:
        latency_ms = (self._clock() - started) * 1000.0
        model = (response.model if response is not None else None) or request.model or profile.model
        stats = LLMCallStats(
            model=model,
            profile_name=profile.name,
            purpose=request.purpose,
            latency_ms=latency_ms,
            prompt_tokens=response.usage.prompt_tokens if response is not None else 0,
            completion_tokens=response.usage.completion_tokens if response is not None else 0,
            retry_count=retry_count,
            fallback_index=fallback_index,
            ok=ok,
        )
        self.last_stats = stats
        logger.info(
            "llm_call",
            extra={
                "llm_model": stats.model,
                "llm_profile": stats.profile_name,
                "llm_purpose": stats.purpose,
                "llm_latency_ms": stats.latency_ms,
                "llm_prompt_tokens": stats.prompt_tokens,
                "llm_completion_tokens": stats.completion_tokens,
                "llm_retry_count": stats.retry_count,
                "llm_fallback_index": stats.fallback_index,
                "llm_ok": stats.ok,
            },
        )


def _open_stream(client: BaseLLMClient, request: LLMRequest) -> Iterator[LLMChunk]:
    iterator = client.stream_chat(request)
    try:
        first = next(iterator)
    except StopIteration:
        return iter(())
    return chain((first,), iterator)
