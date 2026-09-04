"""MockLLMClient：不触网的模型替身，两种模式。

1. **脚本回放**（测试用）——按顺序吐出预置的 :class:`LLMResponse`，可穿插异常，
   并记录收到的每个请求，便于断言"到底喂了什么给模型"。
2. **启发式应答**（零配置演示用）——脚本用尽后按 ``request.stage`` 生成一个结构合法、
   语义合理的回复，让没有 API key 的用户也能在控制台把五个阶段完整跑一遍。

启发式模式依赖两条与 ContextManager 约定好的提示词格式（都做了兜底，格式变了不会崩）：

- 意图阶段的 user 消息是一个 JSON，含 ``current_task`` 与 ``history``；
- 执行阶段的 user 指令里有一行 ``建议工具：<name>``。

它只是演示脚手架，不追求"聪明"——真实推理请配 ``LLM_API_KEY``。
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from agent.enums import MessageRole
from agent.llm.base import BaseLLMClient, LLMRequest, LLMResponse
from agent.models import Message, ToolCall, new_tool_call_id

__all__ = ["MockLLMClient", "MockReply"]

MockReply = LLMResponse | BaseException | Callable[[LLMRequest], LLMResponse]

# 演示用的中文关键词 → 日志里实际存在的英文片段
_KEYWORD_HINTS: dict[str, str] = {
    "错误": "ERROR",
    "报错": "ERROR",
    "异常": "ERROR",
    "警告": "WARN",
    "超时": "timeout",
    "订单": "order-service",
    "支付": "payment-service",
    "网关": "gateway",
    "用户": "user-service",
    "库存": "inventory-service",
}

# 演示用的分组字段猜测
_GROUP_BY_HINTS: dict[str, str] = {
    "服务": "service",
    "service": "service",
    "级别": "level",
    "等级": "level",
    "level": "level",
    "主机": "host",
    "host": "host",
    "状态码": "status_code",
    "status": "status_code",
}

_TOOL_HINT_RE = re.compile(r"建议工具[:：]\s*(\S+)")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}")
_CALL_ID_PREFIX_RE = re.compile(r"^\[tool_call_id=[^\]]+\]\s*")


class MockLLMClient(BaseLLMClient):
    """脚本化 + 启发式的模型替身。"""

    def __init__(
        self, responses: Sequence[MockReply] | None = None, *, model: str = "mock"
    ) -> None:
        self._queue: deque[MockReply] = deque(responses or ())
        self._model = model
        self.calls: list[LLMRequest] = []

    # ---------------------------------------------------------------- 构造便利

    @classmethod
    def reply(cls, content: str = "", *, finish_reason: str = "stop") -> LLMResponse:
        return LLMResponse(content=content, finish_reason=finish_reason, model="mock")

    @classmethod
    def json_reply(cls, payload: dict[str, Any]) -> LLMResponse:
        return cls.reply(json.dumps(payload, ensure_ascii=False))

    @classmethod
    def tool_reply(
        cls,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> LLMResponse:
        call = ToolCall(
            id=call_id or new_tool_call_id(),
            name=name,
            arguments=json.dumps(arguments or {}, ensure_ascii=False),
        )
        return LLMResponse(tool_calls=[call], finish_reason="tool_calls", model="mock")

    # ---------------------------------------------------------------- 队列管理

    def enqueue(self, *replies: MockReply) -> None:
        self._queue.extend(replies)

    def recorded_messages(self) -> list[list[dict[str, Any]]]:
        """每次调用实际喂给模型的 messages，用于断言 prompt 装配。"""
        return [request.to_llm_messages() for request in self.calls]

    # ---------------------------------------------------------------- 主接口

    def chat(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._queue:
            item = self._queue.popleft()
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return item(request)
            return item
        return self._heuristic(request)

    # ---------------------------------------------------------------- 启发式

    def _heuristic(self, request: LLMRequest) -> LLMResponse:
        handlers: dict[str, Callable[[LLMRequest], LLMResponse]] = {
            "intent": self._intent_reply,
            "plan": self._plan_reply,
            "execute": self._execute_reply,
            "validate": self._validate_reply,
        }
        handler = handlers.get(request.stage or "")
        if handler is None:
            return self.reply("(mock) 收到，但没有对应的阶段处理器。")
        return handler(request)

    def _intent_reply(self, request: LLMRequest) -> LLMResponse:
        query, history_ids = _intent_inputs(request)
        text = query.lower()
        if any(word in text for word in ("你好", "你是谁", "能做什么", "帮助", "hello", "help")):
            intent = "chat"
        elif any(word in text for word in ("统计", "分析", "多少", "分布", "平均", "top", "count")):
            intent = "analysis"
        elif any(word in text for word in ("查", "看", "search", "日志", "log", "error", "找")):
            intent = "query"
        elif len(query.strip()) < 4:
            intent = "unknown"
        else:
            intent = "query"

        related: list[str] = []
        if history_ids and any(
            word in query for word in ("刚才", "上面", "前面", "上述", "继续", "这些", "那些")
        ):
            related = [history_ids[-1]]

        return self.json_reply(
            {
                "intent": intent,
                "related_task_ids": related,
                "reason": f"(mock) 依据关键词判定为 {intent}",
                "clarification": (
                    "请补充你想查询的服务名或时间范围。" if intent == "unknown" else ""
                ),
            }
        )

    def _plan_reply(self, request: LLMRequest) -> LLMResponse:
        query = _last_user_text(request)
        operations: list[dict[str, str | None]]
        if "analysis" in (query or "") or any(
            word in query for word in ("统计", "分析", "分布", "多少")
        ):
            operations = [
                {"description": "检索相关日志记录", "suggested_tool": "search_tool"},
                {"description": "对检索结果做分组统计", "suggested_tool": "analysis_tool"},
            ]
        elif any(word in query for word in ("你好", "你是谁", "能做什么", "帮助")):
            operations = [{"description": "直接回答用户的问题", "suggested_tool": None}]
        else:
            operations = [{"description": "按关键字检索日志", "suggested_tool": "search_tool"}]
        return self.json_reply({"operations": operations})

    def _execute_reply(self, request: LLMRequest) -> LLMResponse:
        """执行阶段：本步骤建议的工具还没调过就调它，调过了就给结论。

        判据是"建议工具的结果是否已在对话里"，而不是"最后一条消息是不是工具结果"：
        执行阶段的指令永远拼在消息列表末尾，后者永远为假，会让 mock 无限调下去。
        """
        instruction = _last_user_text(request)
        available = _available_tool_names(request)
        wanted = _suggested_tool(instruction)

        if not available or wanted is None or wanted not in available:
            return self.reply("(mock) 我是日志查询助手，可以按关键字检索日志并做分组统计。")

        served = _last_tool_message(request, tool_name=wanted)
        if served is not None:
            # 预览的首行就是工具给的一句话概括，正好当结论，比截断 160 字符可读；
            # 去掉 [tool_call_id=...] 前缀——那是给模型对账用的，不该出现在给用户的答复里
            headline = next((line for line in served.content.splitlines() if line.strip()), "")
            return self.reply(f"(mock) {_CALL_ID_PREFIX_RE.sub('', headline)}")

        if wanted == "analysis_tool":
            source_id = _last_tool_call_id(request, tool_name="search_tool")
            if source_id is not None:
                return self.tool_reply(
                    "analysis_tool",
                    {
                        "tool_call_id": source_id,
                        "group_by": _guess_group_by(instruction),
                        "metric": "count",
                    },
                )
            wanted = "search_tool"

        return self.tool_reply("search_tool", {"keyword": _guess_keyword(request), "limit": 20})

    def _validate_reply(self, request: LLMRequest) -> LLMResponse:
        """校验阶段拿到的是 summary 快照，没有 assistant 消息可抄，只能从步骤结论里拼。"""
        outputs = [
            str(step["result"])
            for step in _validate_operations(request)
            if isinstance(step, dict) and step.get("result")
        ]
        fallback = next(
            (
                message.content
                for message in reversed(request.messages)
                if message.role is MessageRole.ASSISTANT and message.content
            ),
            "(mock) 任务已执行完成。",
        )
        return self.json_reply(
            {
                "satisfied": True,
                "output": "\n".join(outputs) if outputs else fallback,
                "reason": "(mock) 步骤均已执行，判定为满足需求。",
            }
        )


# ================================================================ 解析辅助


def _last_user_text(request: LLMRequest) -> str:
    for message in reversed(request.messages):
        if message.role is MessageRole.USER:
            return message.content
    return ""


def _intent_inputs(request: LLMRequest) -> tuple[str, list[str]]:
    """从意图阶段的 user 消息里取出"当前 query"与"历史任务 id"。

    优先按约定的 JSON 结构解析；解析不了就退回纯文本，保证提示词改版也不会崩。
    """
    raw = _last_user_text(request)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, []
    if not isinstance(payload, dict):
        return raw, []
    current = payload.get("current_task") or {}
    query = current.get("content", "") if isinstance(current, dict) else ""
    history = payload.get("history") or []
    history_ids = [
        item["task_id"]
        for item in history
        if isinstance(item, dict) and isinstance(item.get("task_id"), str)
    ]
    return query or raw, history_ids


def _available_tool_names(request: LLMRequest) -> set[str]:
    names: set[str] = set()
    for schema in request.tools or []:
        function = schema.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _suggested_tool(instruction: str) -> str | None:
    match = _TOOL_HINT_RE.search(instruction)
    if match is None:
        return None
    name = match.group(1).strip("。.，,")
    return None if name in {"无", "none", "None"} else name


def _last_tool_call_id(request: LLMRequest, *, tool_name: str) -> str | None:
    message = _last_tool_message(request, tool_name=tool_name)
    return None if message is None else message.tool_call_id


def _last_tool_message(request: LLMRequest, *, tool_name: str) -> Message | None:
    for message in reversed(request.messages):
        if message.role is MessageRole.TOOL and message.name == tool_name:
            return message
    return None


def _validate_operations(request: LLMRequest) -> list[Any]:
    """从校验阶段的 user 消息里取出步骤列表；格式不符就返回空表。"""
    try:
        payload = json.loads(_last_user_text(request))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    operations = payload.get("operations")
    return operations if isinstance(operations, list) else []


def _guess_keyword(request: LLMRequest) -> str:
    """从用户原始问题里猜一个能在日志里命中的关键字。"""
    texts = [m.content for m in request.messages if m.role is MessageRole.USER]
    blob = " ".join(texts)
    for chinese, english in _KEYWORD_HINTS.items():
        if chinese in blob:
            return english
    for token in _ASCII_TOKEN_RE.findall(blob):
        if str(token).lower() not in {"mock", "tool", "json"}:
            return str(token)
    return ""


def _guess_group_by(instruction: str) -> str:
    lowered = instruction.lower()
    for hint, field in _GROUP_BY_HINTS.items():
        if hint in instruction or hint in lowered:
            return field
    return "service"
