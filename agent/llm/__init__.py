"""LLM Layer：OpenAI 兼容客户端、健壮性装饰、结构化输出与 Mock 实现。

本层依赖 common / config / models，不依赖 store 与各 manager。
"""

from agent.llm.base import BaseLLMClient, LLMChunk, LLMRequest, LLMResponse, TokenUsage
from agent.llm.factory import build_llm_client
from agent.llm.mock_client import MockLLMClient, MockTurn
from agent.llm.openai_client import OpenAICompatClient, translate_openai_error
from agent.llm.resilient import LLMCallStats, ResilientLLMClient
from agent.llm.structured import call_structured, extract_json

__all__ = [
    "BaseLLMClient",
    "LLMCallStats",
    "LLMChunk",
    "LLMRequest",
    "LLMResponse",
    "MockLLMClient",
    "MockTurn",
    "OpenAICompatClient",
    "ResilientLLMClient",
    "TokenUsage",
    "build_llm_client",
    "call_structured",
    "extract_json",
    "translate_openai_error",
]
