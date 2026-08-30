"""按配置装配 LLM 客户端。业务代码只走这一处，不手拼 OpenAI / Mock / 装饰器。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.config.settings import AppSettings, LLMProfile, LLMSettings
from agent.llm.base import BaseLLMClient
from agent.llm.mock_client import MockLLMClient
from agent.llm.openai_client import OpenAICompatClient
from agent.llm.resilient import ResilientLLMClient

__all__ = ["build_llm_client"]

ClientFactory = Callable[[LLMProfile], BaseLLMClient]


def build_llm_client(
    settings: AppSettings | LLMSettings,
    *,
    mock: MockLLMClient | None = None,
    client_factory: ClientFactory | None = None,
    sleeper: Callable[[float], None] | None = None,
    http_client: Any | None = None,
) -> BaseLLMClient:
    """`LLM_USE_MOCK=true` 时返回 Mock；否则按 profile 链包一层 `ResilientLLMClient`。"""
    llm = settings.llm if isinstance(settings, AppSettings) else settings
    if llm.use_mock:
        return mock if mock is not None else MockLLMClient()

    def default_factory(profile: LLMProfile) -> BaseLLMClient:
        return OpenAICompatClient(profile, http_client=http_client)

    factory = client_factory or default_factory
    clients = [factory(profile) for profile in llm.profile_chain()]
    return ResilientLLMClient(clients, llm, sleeper=sleeper)
