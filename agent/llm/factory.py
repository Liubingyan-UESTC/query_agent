"""模型客户端的装配入口。

装配规则只有一条分支：**没配 key 就用 Mock**。这样控制台零配置即可跑通全链路，
而配了 key 就自动切到真实端点 + 降级链。
"""

from __future__ import annotations

from collections.abc import Callable

from agent.config import AppSettings, LLMProfile, LLMSettings
from agent.llm.base import BaseLLMClient
from agent.llm.limiter import ConcurrencyGate, GatedLLMClient
from agent.llm.mock_client import MockLLMClient
from agent.llm.openai_client import OpenAICompatClient, ResilientLLMClient

__all__ = ["build_llm_client"]

ClientFactory = Callable[[LLMProfile], BaseLLMClient]


def build_llm_client(
    settings: AppSettings | LLMSettings,
    *,
    mock: MockLLMClient | None = None,
    client_factory: ClientFactory | None = None,
) -> BaseLLMClient:
    """按配置构建客户端。

    ``client_factory`` 是测试注入点：可以在不触网的情况下验证降级链的行为。

    装配顺序是 ``Resilient(Gated(端点))``：闸门在重试**里面**，所以退避 sleep 不占
    并发许可。全链路共享一个 gate，见 :class:`~agent.llm.limiter.ConcurrencyGate`。
    """
    llm = settings.llm if isinstance(settings, AppSettings) else settings
    if llm.use_mock_client():
        return mock if mock is not None else MockLLMClient()

    factory: ClientFactory = client_factory or OpenAICompatClient
    gate = ConcurrencyGate(llm.max_concurrency)
    clients = [GatedLLMClient(factory(profile), gate) for profile in llm.profile_chain()]
    return ResilientLLMClient(clients, max_retries=llm.primary.max_retries)
