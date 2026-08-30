"""LLM 客户端抽象：请求 / 响应 / 分块与厂商无关的调用面。

上层（阶段处理器、PromptAssembler）只依赖本模块，不感知 OpenAI / 其它供应商。
`messages` 使用 `Message` 而非裸 dict：发出去之前统一走 `to_llm_dict()`，
编排字段不会漏进载荷。
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator

from agent.models.base import AgentModel
from agent.models.message import Message, ToolCall

__all__ = [
    "BaseLLMClient",
    "LLMChunk",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
]


class TokenUsage(AgentModel):
    """一次补全消耗的 token。键名对齐 OpenAI `usage` 对象。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(AgentModel):
    """一次模型调用的入参。

    `model` / `temperature` / `max_tokens` 为 `None` 时由客户端回落到
    `LLMProfile` 的对应字段，避免每个调用方都复制一份默认值。
    `stream` 只表达意图：`chat()` 拒绝 `True`，`stream_chat()` 始终流式，
    避免把 iterator 当成 `LLMResponse` 使用。
    """

    messages: list[Message]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(cls, value: object) -> object:
        """允许传入已是 OpenAI 形状的 mapping，省掉调用方先包一层 Message。"""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return value
        coerced: list[object] = []
        for index, item in enumerate(value):
            if isinstance(item, Message):
                coerced.append(item)
            elif isinstance(item, Mapping):
                coerced.append(Message.model_validate(item))
            else:
                raise ValueError(
                    f"messages[{index}] 必须是 Message 或 mapping，实际是 {type(item).__name__}"
                )
        return coerced


class LLMChunk(AgentModel):
    """`stream_chat` 的一个增量片。

    流式 `tool_calls` 是按 token 切开的碎片（`id` / `name` / `arguments` 可能为
    `None`），不能收成完整 `ToolCall`。调用方按 `index` 自行拼接，或走非流式 `chat()`。
    """

    content: str = ""
    tool_call_deltas: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None


class LLMResponse(AgentModel):
    """一次非流式补全的结果。"""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: TokenUsage
    model: str
    latency_ms: float
    raw: dict[str, Any] | None = None


class BaseLLMClient(ABC):
    """模型调用的唯一抽象。步骤 10 的装饰器与 Mock 都实现这一面。"""

    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        """同步补全。`request.stream` 必须为 `False`。"""

    @abstractmethod
    def stream_chat(self, request: LLMRequest) -> Iterator[LLMChunk]:
        """流式补全。分块顺序与供应商推送顺序一致。"""

    def close(self) -> None:
        """释放底层连接。默认无资源，具体实现按需覆盖。"""
        return
