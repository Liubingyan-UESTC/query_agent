"""LLM 层：模型客户端的统一抽象与装配入口。"""

from agent.llm.base import BaseLLMClient, LLMRequest, LLMResponse, TokenUsage
from agent.llm.factory import build_llm_client
from agent.llm.limiter import ConcurrencyGate, GatedLLMClient
from agent.llm.mock_client import MockLLMClient
from agent.llm.openai_client import (
    OpenAICompatClient,
    ResilientLLMClient,
    strip_reasoning,
    translate_openai_error,
)
from agent.llm.structured import call_structured, extract_json

__all__ = [
    "BaseLLMClient",
    "ConcurrencyGate",
    "GatedLLMClient",
    "LLMRequest",
    "LLMResponse",
    "MockLLMClient",
    "OpenAICompatClient",
    "ResilientLLMClient",
    "TokenUsage",
    "build_llm_client",
    "call_structured",
    "extract_json",
    "strip_reasoning",
    "translate_openai_error",
]
