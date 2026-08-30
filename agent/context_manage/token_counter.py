"""可插拔 token 计数：tiktoken 可用则精确，否则按字符估算。

不把 tiktoken 列为硬依赖。本地/CI 没装时走估算，避免为了数 token
把整个 LLM 供应商的编码表拉进运行时。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from agent.models.message import Message

__all__ = [
    "CHAR_ESTIMATE_ASCII_DIVISOR",
    "PER_MESSAGE_OVERHEAD",
    "CharEstimateCounter",
    "TiktokenCounter",
    "TokenCounter",
    "default_counter",
]

# 英文/符号大约 4 字符 1 token；CJK 按 1 字符 1 token（偏保守，宁可少投也不超窗）
CHAR_ESTIMATE_ASCII_DIVISOR = 4
PER_MESSAGE_OVERHEAD = 4

_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
)


class TokenCounter(Protocol):
    """窗口装配只依赖这一面，不关心底层是 tiktoken 还是估算。"""

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: Sequence[Message | Mapping[str, Any]]) -> int: ...


class CharEstimateCounter:
    """按字符估算。CJK 按 1:1，其余按 4 字符 1 token。"""

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        cjk = 0
        for char in text:
            code = ord(char)
            if any(start <= code <= end for start, end in _CJK_RANGES):
                cjk += 1
        other = len(text) - cjk
        return cjk + (other + CHAR_ESTIMATE_ASCII_DIVISOR - 1) // CHAR_ESTIMATE_ASCII_DIVISOR

    def count_messages(self, messages: Sequence[Message | Mapping[str, Any]]) -> int:
        return sum(_count_one(self, item) for item in messages)


class TiktokenCounter:
    """用 OpenAI `cl100k_base` 编码精确计数。"""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken 未安装，无法使用 TiktokenCounter") from exc
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_messages(self, messages: Sequence[Message | Mapping[str, Any]]) -> int:
        return sum(_count_one(self, item) for item in messages)


def default_counter() -> TokenCounter:
    """优先 tiktoken，否则回退字符估算。"""
    try:
        return TiktokenCounter()
    except RuntimeError:
        return CharEstimateCounter()


def _count_one(counter: TokenCounter, item: Message | Mapping[str, Any]) -> int:
    payload = item.to_llm_dict() if isinstance(item, Message) else dict(item)
    total = PER_MESSAGE_OVERHEAD + counter.count_text(str(payload.get("role", "")))
    total += counter.count_text(str(payload.get("content") or ""))
    if payload.get("name"):
        total += counter.count_text(str(payload["name"]))
    if payload.get("tool_call_id"):
        total += counter.count_text(str(payload["tool_call_id"]))
    for call in payload.get("tool_calls") or []:
        if isinstance(call, Mapping):
            function = call.get("function")
            if isinstance(function, Mapping):
                total += counter.count_text(str(function.get("name") or ""))
                total += counter.count_text(str(function.get("arguments") or ""))
            else:
                total += counter.count_text(str(call))
        else:
            total += counter.count_text(str(call))
    return total
