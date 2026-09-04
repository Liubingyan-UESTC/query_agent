"""数据模型：黑板上流动的全部结构。

约定两条：

1. 所有模型继承 :class:`AgentModel`，即 ``extra="forbid"`` + ``validate_assignment=True``。
   字段名写错、类型写错都在赋值当场报错，而不是在几层之后变成一个 ``None``。
2. :class:`Message` 区分"协议字段"与"编排字段"：只有 ``to_llm_dict()`` 输出的那几个
   键会进入发给模型的载荷，``message_id`` / ``created_at`` 这类内部字段永不外泄，
   既省 token，也避免模型模仿这些字段作答。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.enums import IntentType, MessageRole, OperationStatus, TaskStatus

__all__ = [
    "AgentModel",
    "Message",
    "Operation",
    "Task",
    "TaskSummary",
    "ToolCall",
    "new_message_id",
    "new_session_id",
    "new_task_id",
    "new_tool_call_id",
]

# OpenAI 目前只定义了 function 一种 tool_call 类型
_FUNCTION_CALL_TYPE = "function"


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_task_id() -> str:
    return _short_id("task")


def new_session_id() -> str:
    return _short_id("sess")


def new_message_id() -> str:
    return _short_id("msg")


def new_tool_call_id() -> str:
    return _short_id("call")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentModel(BaseModel):
    """全部数据模型的基类。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dict(self) -> dict[str, Any]:
        """转为可直接 ``json.dumps`` 的字典（datetime / 枚举都已是字符串）。"""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls.model_validate(payload)


# ============================================================ 消息与工具调用


class ToolCall(AgentModel):
    """一次工具调用请求，字段与 OpenAI ``tool_calls`` 元素一一对应。

    ``arguments`` 保持模型返回的 JSON **字符串**而非 dict：提前解析会让"模型吐了非法
    JSON"这个错误丢失原始载荷，无从复现。需要结构化访问时调用 :meth:`parsed_arguments`。
    """

    id: str = Field(default_factory=new_tool_call_id)
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        """解析 ``arguments``；非法 JSON 或非对象结构都抛 ``ValueError``。"""
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具调用 {self.name} 的 arguments 不是合法 JSON：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"工具调用 {self.name} 的 arguments 必须是 JSON 对象")
        return parsed

    def to_llm_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": _FUNCTION_CALL_TYPE,
            "function": {"name": self.name, "arguments": self.arguments},
        }

    @classmethod
    def from_llm_dict(cls, payload: Mapping[str, Any]) -> ToolCall:
        """由 OpenAI 的嵌套结构还原，与 :meth:`to_llm_dict` 互为逆运算。

        嵌套拆解只写这一处：调用方各自去取 ``payload["function"]["name"]`` 的话，
        缺字段时只会得到一个裸 ``KeyError``，分不清是模型返回不合规还是解析代码写错。
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
                arguments=function.get("arguments") or "{}",
            )
        except KeyError as exc:
            raise ValueError(f"tool_call 缺少必需字段 {exc.args[0]!r}") from exc


class Message(AgentModel):
    """一条对话消息，即 context window 中 task_content 的原子。"""

    message_id: str = Field(default_factory=new_message_id)
    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_protocol_invariants(self) -> Message:
        """校验 OpenAI 的硬性约束，避免非法组合一路传到 API 才 400。"""
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError(f"只有 assistant 消息可携带 tool_calls，当前 role={self.role.value}")
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("role=tool 的消息必须提供 tool_call_id，否则模型无法对应请求")
        if self.tool_call_id and self.role is not MessageRole.TOOL:
            raise ValueError(f"tool_call_id 仅对 role=tool 有意义，当前 role={self.role.value}")
        return self

    def to_llm_dict(self) -> dict[str, Any]:
        """输出 OpenAI messages 规范的载荷，只含协议字段。

        ``content`` 恒定存在（哪怕是空串），因为 OpenAI 要求该键必须出现；
        其余可选键按需省略，不输出 ``null`` 占位以免白费 token。
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
    def system(cls, content: str, **kwargs: Any) -> Message:
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs: Any) -> Message:
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str = "", **kwargs: Any) -> Message:
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str | None = None) -> Message:
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


# ============================================================ 任务摘要与任务


class Operation(AgentModel):
    """规划阶段产出的单个步骤。

    ``suggested_tool`` 只是给模型的提示而非硬绑定：执行阶段由模型用 function calling
    自行决定实际调用哪个工具，规划的建议错了也能纠正。
    """

    index: int = Field(ge=0)
    description: str
    suggested_tool: str | None = None
    status: OperationStatus = OperationStatus.PENDING
    result: str | None = None


class TaskSummary(AgentModel):
    """context window 记录之一：一次任务的总结。

    字段与 docs/require.md 的 task summary 一一对应（``related task`` 落地为
    ``related_task_ids``）。意图识别阶段只投喂 ``content`` 与 ``status`` 两项，
    见 :attr:`INTENT_PROMPT_FIELDS`。
    """

    INTENT_PROMPT_FIELDS: ClassVar[tuple[str, ...]] = ("task_id", "content", "status")

    task_id: str
    content: str
    intent: IntentType | None = None
    related_task_ids: list[str] = Field(default_factory=list)
    operations: list[Operation] = Field(default_factory=list)
    result: str | None = None
    status: TaskStatus = TaskStatus.CREATED
    output: str | None = None

    def to_prompt_dict(self, fields: tuple[str, ...] | None = None) -> dict[str, Any]:
        """按白名单投影出投喂模型的字段，默认给全量。

        显式点名而不是"排除几个字段"：将来新增内部字段时，默认不会泄漏给模型。
        """
        payload = self.to_dict()
        if fields is None:
            return payload
        unknown = set(fields) - set(payload)
        if unknown:
            raise ValueError(f"未知的 summary 字段：{sorted(unknown)}")
        return {key: payload[key] for key in fields}

    def pending_operations(self) -> list[Operation]:
        return [op for op in self.operations if op.status is not OperationStatus.SUCCEEDED]


class Task(AgentModel):
    """一次用户请求。状态与 summary.status 始终同步。"""

    task_id: str = Field(default_factory=new_task_id)
    session_id: str
    status: TaskStatus = TaskStatus.CREATED
    summary: TaskSummary
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _status_must_match_summary(self) -> Task:
        if self.summary.task_id != self.task_id:
            raise ValueError(
                f"summary.task_id({self.summary.task_id}) 与 task_id({self.task_id}) 不一致"
            )
        if self.summary.status is not self.status:
            raise ValueError("summary.status 必须与 task.status 同步，请改用 transition_to()")
        return self

    @classmethod
    def create(cls, content: str, *, session_id: str, task_id: str | None = None) -> Task:
        tid = task_id or new_task_id()
        return cls(
            task_id=tid,
            session_id=session_id,
            summary=TaskSummary(task_id=tid, content=content),
        )

    def transition_to(self, target: TaskStatus) -> None:
        """按状态机推进任务；非法转移抛 :class:`~agent.errors.TaskStateError`。

        两处状态（``task.status`` 与 ``summary.status``）只在这里同时改写，
        避免出现"前端读 summary、编排读 task"时两边不一致的无声 bug。

        赋值顺序有讲究：``validate_assignment=True`` 会在写 ``self.status`` 时立刻校验
        "两处状态一致"，所以必须先写 summary 再写 task，否则中间态自己就把自己拦下了。
        合法性检查在最前，失败时两处都还没动过。
        """
        new_status = self.status.transition_to(target, self.task_id)
        self.summary.status = new_status
        self.status = new_status
        self.updated_at = _utcnow()
