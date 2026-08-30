"""按脚本回放的 LLM 客户端。后续所有阶段测试的基石。

支持两种匹配：按入队顺序消费，或用 `when` 谓词抢先命中。收到的完整
`LLMRequest`（含 messages）全部记入 `calls`，供断言 prompt 内容。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from agent.common.errors import LLMError
from agent.llm.base import BaseLLMClient, LLMChunk, LLMRequest, LLMResponse, TokenUsage

__all__ = ["MockLLMClient", "MockTurn"]

MockReply = LLMResponse | BaseException | Callable[[LLMRequest], LLMResponse]


@dataclass
class MockTurn:
    """一条预设回放。`when` 为 `None` 时按入队顺序消费。"""

    reply: MockReply
    when: Callable[[LLMRequest], bool] | None = None
    used: bool = False


class MockLLMClient(BaseLLMClient):
    """不访问网络的脚本客户端。"""

    def __init__(
        self,
        responses: Sequence[MockReply] | None = None,
        *,
        default_model: str = "mock",
    ) -> None:
        self.default_model = default_model
        self.turns: list[MockTurn] = [MockTurn(reply=item) for item in (responses or [])]
        self.calls: list[LLMRequest] = []

    @classmethod
    def reply(
        cls,
        content: str,
        *,
        model: str = "mock",
        finish_reason: str = "stop",
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        **kwargs: Any,
    ) -> LLMResponse:
        """构造一条成功响应，少写重复字段。"""
        return LLMResponse(
            content=content,
            tool_calls=[],
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model,
            latency_ms=latency_ms,
            **kwargs,
        )

    def enqueue(
        self,
        reply: MockReply,
        *,
        when: Callable[[LLMRequest], bool] | None = None,
    ) -> None:
        self.turns.append(MockTurn(reply=reply, when=when))

    def recorded_messages(self) -> list[list[dict[str, Any]]]:
        """每次调用投喂给模型的协议字段，便于断言 prompt。"""
        return [[message.to_llm_dict() for message in request.messages] for request in self.calls]

    def chat(self, request: LLMRequest) -> LLMResponse:
        if request.stream:
            raise ValueError("chat() 不接受 stream=True，请调用 stream_chat()")
        return self._resolve(request)

    def stream_chat(self, request: LLMRequest) -> Iterator[LLMChunk]:
        response = self._resolve(request)
        if response.content:
            yield LLMChunk(content=response.content, model=response.model)
        yield LLMChunk(
            content="",
            finish_reason=response.finish_reason,
            model=response.model,
        )

    def _resolve(self, request: LLMRequest) -> LLMResponse:
        if not request.messages:
            raise ValueError("LLMRequest.messages 不能为空")
        self.calls.append(request)
        turn = self._take(request)
        reply = turn.reply
        try:
            if isinstance(reply, BaseException):
                raise reply
            if callable(reply):
                return reply(request)
            return reply
        except Exception:
            # 可调用脚本可能是「先失败再成功」的 flaky，留给下一次重试再命中
            if callable(reply):
                turn.used = False
            raise

    def _take(self, request: LLMRequest) -> MockTurn:
        for turn in self.turns:
            if not turn.used and turn.when is not None and turn.when(request):
                turn.used = True
                return turn
        for turn in self.turns:
            if not turn.used and turn.when is None:
                turn.used = True
                return turn
        raise LLMError("没有匹配的 Mock 响应", retryable=False)
