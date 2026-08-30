"""LLM Layer：OpenAI 兼容客户端、健壮性装饰、结构化输出与 Mock 实现。

本层依赖 common / config / models，不依赖 store 与各 manager。
步骤 9 交付抽象与兼容客户端；步骤 10 再补降级重试、结构化输出与 Mock。
"""

from agent.llm.base import BaseLLMClient, LLMChunk, LLMRequest, LLMResponse, TokenUsage
from agent.llm.openai_client import OpenAICompatClient, translate_openai_error

__all__ = [
    "BaseLLMClient",
    "LLMChunk",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatClient",
    "TokenUsage",
    "translate_openai_error",
]
