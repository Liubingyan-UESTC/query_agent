"""对话消息：Context Window 三部分中「task_content」的原子结构。

`Message` 同时承载两类信息，二者的读者不同，故用两套出口分开：

- **协议字段**（role / content / name / tool_calls / tool_call_id）——投喂 LLM，
  由 `to_llm_dict()` 输出，格式严格遵循 OpenAI messages 规范。
- **编排字段**（message_id / artifact_refs / scope / source_task_id / created_at / meta）
  ——供 ContextManager 裁剪窗口、按 scope 归档、按 artifact_id 取全量数据，
  绝不进入发给模型的载荷，否则既浪费 token 又会诱导模型模仿这些字段作答。

大体积数据一律不放在 `content` 里：工具产出经 `Artifact` 存放，消息中只留
`artifact_refs` 与预览文本。
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import Field, model_validator

from agent.common.enums import ContextScope, MessageRole
from agent.common.ids import new_message_id
from agent.models.base import AgentModel

__all__ = ["Message", "ToolCall"]

# OpenAI 目前只定义了 function 一种 tool_call 类型
_FUNCTION_CALL_TYPE = "function"


class ToolCall(AgentModel):
    """一次工具调用请求。字段与 OpenAI `tool_calls` 元素一一对应。

    `arguments` 保持 OpenAI 原样的 JSON **字符串**而非 dict：模型返回的就是字符串，
    提前解析会让「模型吐了非法 JSON」这个错误丢失原始载荷，无从复现。
    需要结构化访问时调用 `parsed_arguments()`。
    """

    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        """解析 `arguments`。非法 JSON 或非对象结构都抛 `ValueError`。"""
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具调用 {self.name} 的 arguments 不是合法 JSON：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"工具调用 {self.name} 的 arguments 必须是 JSON 对象")
        return parsed

    def to_llm_dict(self) -> dict[str, Any]:
        """还原为 OpenAI 的嵌套结构。"""
        return {
            "id": self.id,
            "type": _FUNCTION_CALL_TYPE,
            "function": {"name": self.name, "arguments": self.arguments},
        }

    @classmethod
    def from_llm_dict(cls, payload: Mapping[str, Any]) -> "ToolCall":
        """由 OpenAI 的嵌套结构还原，与 `to_llm_dict()` 互为逆运算。

        嵌套拆解只在这里写一次：步骤 9 解析模型响应时若各自手工取
        `payload["function"]["name"]`，缺字段的报错会退化成裸 `KeyError`，
        看不出是模型返回不合规还是解析代码写错。
        """
        call_type = payload.get("type", _FUNCTION_CALL_TYPE)
        if call_type != _FUNCTION_CALL_TYPE:
            raise ValueError(f"不支持的 tool_call 类型：{call_type!r}")
        function = payload.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool_call 缺少 function 对象")
        try:
            return cls(
                id=payload["id"],
                name=function["name"],
                arguments=function.get("arguments", "{}"),
            )
        except KeyError as exc:
            raise ValueError(f"tool_call 缺少必需字段 {exc.args[0]!r}") from exc


class Message(AgentModel):
    """一条对话消息。"""

    message_id: str = Field(default_factory=new_message_id)
    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    scope: ContextScope = ContextScope.CURRENT
    source_task_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_protocol_invariants(self) -> "Message":
        """校验 OpenAI 的硬性约束，避免非法组合一路传到 API 才 400。"""
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError(f"只有 assistant 消息可携带 tool_calls，当前 role={self.role.value}")
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("role=tool 的消息必须提供 tool_call_id，否则模型无法对应请求")
        if self.tool_call_id and self.role is not MessageRole.TOOL:
            raise ValueError(f"tool_call_id 仅对 role=tool 有意义，当前 role={self.role.value}")
        if self.scope is not ContextScope.CURRENT and not self.source_task_id:
            raise ValueError(
                f"scope={self.scope.value} 的消息必须提供 source_task_id，否则注入来源无法追溯"
            )
        return self

    def to_llm_dict(self) -> dict[str, Any]:
        """输出 OpenAI messages 规范的载荷，只含协议字段。

        `content` 恒定存在（哪怕为空串），因为 OpenAI 要求该键必须出现；
        其余可选键按需省略，不输出 `null` 占位以免额外消耗 token。
        """
        payload: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [call.to_llm_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def system(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs: Any) -> "Message":
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str = "", **kwargs: Any) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)

    @classmethod
    def tool_result(cls, content: str, *, tool_call_id: str, **kwargs: Any) -> "Message":
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            **kwargs,
        )
