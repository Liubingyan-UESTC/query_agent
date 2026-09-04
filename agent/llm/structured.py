"""结构化输出：让模型稳定吐出可解析的 JSON。

三层递进，越靠后越"脏"但越管用：

1. 先请求 ``response_format={"type": "json_object"}``；端点不支持就去掉重来一次；
2. 从文本里提取 JSON——整段解析 → ```json 围栏 → 首个 ``{`` 到末个 ``}``；
3. 解析出来但过不了 schema 校验时，把错误回喂给模型让它改一次。

改一次仍失败就抛 :class:`~agent.errors.LLMResponseFormatError`（不可重试）——同样的
提示词再来一遍只会复现同一个错误。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agent.errors import LLMError, LLMResponseFormatError
from agent.llm.base import BaseLLMClient, LLMRequest
from agent.models import Message

__all__ = ["call_structured", "extract_json"]

T = TypeVar("T", bound=BaseModel)

_JSON_OBJECT_FORMAT = {"type": "json_object"}

# 端点不支持 response_format 时的报错特征
_FORMAT_UNSUPPORTED_HINTS = ("response_format", "json_object", "json_schema")


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON。三种形态依次尝试，都失败则抛 ``ValueError``。"""
    stripped = text.strip()
    if not stripped:
        raise ValueError("模型返回了空内容")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = _extract_fenced(stripped)
    if fenced is not None:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法从输出中解析 JSON：{exc}") from exc

    raise ValueError("输出中找不到 JSON 对象")


def _extract_fenced(text: str) -> str | None:
    """取出 ```json ... ``` 围栏里的内容；有多段围栏时取第一段像 JSON 对象的。"""
    fence = "```"
    if fence not in text:
        return None
    for part in text.split(fence)[1:]:
        candidate = part.removeprefix("json").strip()
        if candidate.startswith("{"):
            return candidate
    return None


def call_structured(
    client: BaseLLMClient,
    messages: Sequence[Message],
    schema: type[T],
    *,
    stage: str | None = None,
    max_repair: int = 1,
    model: str | None = None,
    temperature: float | None = 0.0,
) -> T:
    """调用模型并把输出解析成 ``schema`` 实例。"""
    conversation = list(messages)
    use_json_format = True
    last_error = ""

    for attempt in range(max_repair + 1):
        request = LLMRequest(
            messages=conversation,
            model=model,
            temperature=temperature,
            response_format=_JSON_OBJECT_FORMAT if use_json_format else None,
            stage=stage,
        )
        try:
            response = client.chat(request)
        except LLMError as exc:
            if use_json_format and _is_format_unsupported(exc):
                # 端点不认 response_format，去掉重来（不计入 repair 次数）
                use_json_format = False
                continue
            raise

        try:
            payload = extract_json(response.content)
            return schema.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            if attempt >= max_repair:
                break
            conversation = [
                *conversation,
                Message.assistant(response.content),
                Message.user(
                    f"上次输出无法通过校验：{last_error}\n"
                    f"请只输出修正后的 JSON 对象，不要任何解释文字或代码围栏。"
                ),
            ]

    raise LLMResponseFormatError(
        f"模型在 {max_repair + 1} 次尝试后仍未给出合法的 {schema.__name__}",
        detail=last_error,
    )


def _is_format_unsupported(exc: LLMError) -> bool:
    message = f"{exc.message} {exc.detail}".lower()
    return any(hint in message for hint in _FORMAT_UNSUPPORTED_HINTS)
