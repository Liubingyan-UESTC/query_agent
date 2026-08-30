"""结构化调用：优先 json_schema，失败则提示词约束 + 提取 + 一次修复重问。

`call_structured` 是阶段处理器拿到可靠 dict 的唯一入口。最终仍无法通过
校验时抛 `LLMResponseFormatError`（不可重试——修复已经用过了）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.common.errors import LLMError, LLMResponseFormatError
from agent.config.settings import LLMPurpose
from agent.llm.base import BaseLLMClient, LLMRequest
from agent.models.message import Message

__all__ = ["call_structured", "extract_json"]

SchemaArg = dict[str, Any] | type[BaseModel]

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_UNSUPPORTED_MARKERS = ("response_format", "json_schema", "json_object")
_SCHEMA_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def call_structured(
    client: BaseLLMClient,
    messages: Sequence[Message],
    schema: SchemaArg,
    *,
    max_repair: int = 1,
    prefer_json_schema: bool = True,
    model: str | None = None,
    purpose: LLMPurpose | None = None,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """调用模型并返回通过 schema 校验的 JSON 对象。

    `max_repair` 是校验失败后的重问次数，不含首次尝试，也不含
    「json_schema 不被端点支持」那一次回退。
    """
    if max_repair < 0:
        raise ValueError("max_repair 不能为负")
    if not messages:
        raise ValueError("call_structured 的 messages 不能为空")

    schema_dict, model_cls = _normalize_schema(schema)
    use_format = prefer_json_schema
    conversation = _seed_conversation(messages, schema_dict, use_format=use_format)
    repairs_left = max_repair
    last_error = "模型未返回可解析的 JSON"

    while True:
        request = LLMRequest(
            messages=conversation,
            response_format=_json_schema_format(schema_dict) if use_format else None,
            model=model,
            purpose=purpose,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            response = client.chat(request)
        except LLMError as exc:
            if use_format and _is_format_unsupported(exc):
                use_format = False
                conversation = _seed_conversation(messages, schema_dict, use_format=False)
                continue
            raise

        try:
            return _validate(extract_json(response.content), schema_dict, model_cls)
        except ValueError as exc:
            last_error = str(exc)
            if repairs_left <= 0:
                raise LLMResponseFormatError(
                    "模型输出无法通过 schema 校验",
                    detail=last_error,
                ) from exc
            repairs_left -= 1
            conversation = [
                *conversation,
                Message.assistant(response.content),
                Message.user(f"上次输出无法通过校验：{exc}\n请只输出修正后的 JSON 对象。"),
            ]


def extract_json(text: str) -> object:
    """从模型文本中抽出 JSON。先整段解析，再代码块，再取首尾花括号。"""
    stripped = text.strip()
    if not stripped:
        raise ValueError("模型返回了空内容")
    for candidate in _candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("模型输出中没有合法 JSON")


def _candidates(text: str) -> list[str]:
    found = [text]
    fenced = _FENCE.search(text)
    if fenced is not None:
        found.append(fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        found.append(text[start : end + 1])
    return found


def _normalize_schema(schema: SchemaArg) -> tuple[dict[str, Any], type[BaseModel] | None]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema(), schema
    if isinstance(schema, dict):
        return schema, None
    raise TypeError(
        f"schema 必须是 JSON Schema dict 或 BaseModel 子类，实际是 {type(schema).__name__}"
    )


def _json_schema_format(schema: dict[str, Any]) -> dict[str, Any]:
    raw_name = schema.get("title") or "result"
    name = _SCHEMA_NAME_SAFE.sub("_", str(raw_name)).strip("_") or "result"
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": False, "schema": schema},
    }


def _schema_hint(schema: dict[str, Any]) -> str:
    dumped = json.dumps(schema, ensure_ascii=False, indent=2)
    return f"你必须只输出一个符合下列 JSON Schema 的 JSON 对象，不要输出解释性文字。\n{dumped}"


def _is_format_unsupported(exc: LLMError) -> bool:
    if exc.retryable:
        return False
    text = f"{exc.message} {exc.detail or ''}".lower()
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


def _seed_conversation(
    messages: Sequence[Message],
    schema: dict[str, Any],
    *,
    use_format: bool,
) -> list[Message]:
    if use_format:
        return list(messages)
    return [*messages, Message.user(_schema_hint(schema))]


def _validate(
    data: object,
    schema: dict[str, Any],
    model_cls: type[BaseModel] | None,
) -> dict[str, Any]:
    if model_cls is not None:
        try:
            return model_cls.model_validate(data).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
    return _validate_object(data, schema)


def _validate_object(data: object, schema: dict[str, Any]) -> dict[str, Any]:
    expected = schema.get("type")
    if expected and expected != "object":
        raise ValueError(f"当前只支持 type=object 的 schema，实际是 {expected!r}")
    if not isinstance(data, dict):
        raise ValueError("输出必须是 JSON 对象")

    required = schema.get("required") or []
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"缺少字段：{missing}")

    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        extra = set(data) - set(properties)
        if extra:
            raise ValueError(f"多余字段：{sorted(extra)}")

    for key, spec in properties.items():
        if key not in data:
            continue
        if not isinstance(spec, dict):
            continue
        _check_property(key, data[key], spec)
    return data


def _check_property(key: str, value: object, spec: dict[str, Any]) -> None:
    allowed = spec.get("enum")
    if allowed is not None and value not in allowed:
        raise ValueError(f"{key} 不在 {list(allowed)} 内")
    type_spec = spec.get("type")
    if type_spec is None:
        return
    names = type_spec if isinstance(type_spec, list) else [type_spec]
    if not any(_is_json_type(value, name) for name in names):
        raise ValueError(f"{key} 期望类型 {type_spec}，实际是 {type(value).__name__}")


def _is_json_type(value: object, spec: object) -> bool:
    if spec == "string":
        return isinstance(value, str)
    if spec == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec == "boolean":
        return isinstance(value, bool)
    if spec == "object":
        return isinstance(value, dict)
    if spec == "array":
        return isinstance(value, list)
    if spec == "null":
        return value is None
    return True
