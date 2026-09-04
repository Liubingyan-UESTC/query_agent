"""LLM 客户端的抽象接口与请求/响应结构。

只定义同步的 :meth:`BaseLLMClient.chat`——本系统的出口是控制台，一次任务只在最后打印
最终结果，流式输出没有消费者，因此不引入 stream 接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from agent.models import AgentModel, Message, ToolCall

__all__ = ["BaseLLMClient", "LLMRequest", "LLMResponse", "TokenUsage"]


class TokenUsage(AgentModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(AgentModel):
    """一次模型调用的全部输入。

    ``model`` / ``temperature`` / ``max_tokens`` 留空表示"用 profile 里的默认值"，
    这样调用方不必在每个阶段重复写一遍默认参数。
    """

    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None

    stage: str | None = None
    """当前所处的编排阶段（intent/plan/execute/validate）。

    只用于日志与 MockLLM 的启发式应答，**不参与真实模型路由**——按用途选模型会让
    配置与状态机强耦合，收益不抵成本。
    """

    def to_llm_messages(self) -> list[dict[str, Any]]:
        return [message.to_llm_dict() for message in self.messages]


class LLMResponse(AgentModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class BaseLLMClient(ABC):
    """所有模型客户端的接口。"""

    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        """发起一次对话补全。失败时抛 :class:`~agent.errors.LLMError`。"""

    def close(self) -> None:  # pragma: no cover - 默认无资源可释放
        return None
