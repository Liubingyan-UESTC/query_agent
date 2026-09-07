"""上下文管理：黑板结构与阶段消息装配。

**黑板**（:class:`ContextWindow`）就是 require.md 说的 context window，恰好三项记录：

============  ============================================================
task_summary  一次任务的总结（8 个字段）
tool_result   ``{tool_call_id: {tool_name, tool_result}}``，工具结果的全量存放处
task_content  System / User / Assistant / Tool 四种角色的完整聊天记录
============  ============================================================

**唯一写入通道**是 :class:`ContextManager`。上层不直接改 window 的字段，这样"谁改了黑板"
只有一个答案；``summary`` 还有字段白名单，防止模型顺手把 ``status`` 或 ``content`` 改了
——那两个字段只能由 TaskManager 经状态机改写。

工具结果的处理是这里的关键：全量进 ``tool_result``，**回给模型的只有预览 + tool_call_id**，
它想看全量就自己调 ``fetch_tool_result``。这样几十条日志不会一次次挤占上下文。
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import Field

from agent.config import AppSettings
from agent.enums import IntentType, MessageRole
from agent.knowledge import KnowledgeMemory
from agent.models import AgentModel, Message, Operation, Task, TaskSummary
from agent.tools.base import ToolResult

__all__ = ["ContextManager", "ContextWindow", "ToolResultRecord"]


class ToolResultRecord(AgentModel):
    """黑板上的一条工具结果。

    ``tool_name`` 与 ``tool_result`` 来自 require.md 的定义；``ok`` 是必要的最小扩展，
    校验阶段要靠它区分"查到了 0 条"和"这一步压根失败了"。
    """

    tool_name: str
    tool_result: Any
    ok: bool = True


class ContextWindow(AgentModel):
    """一次任务的全部工作空间——黑板本体。"""

    task_id: str
    session_id: str
    summary: TaskSummary
    tool_results: dict[str, ToolResultRecord] = Field(default_factory=dict)
    content: list[Message] = Field(default_factory=list)

    @classmethod
    def for_task(cls, task: Task) -> ContextWindow:
        """与 Task 共用同一个 summary 对象，避免两处摘要各自漂移。"""
        return cls(task_id=task.task_id, session_id=task.session_id, summary=task.summary)

    def recent_content(self, limit: int) -> list[Message]:
        """取最近 limit 条非 system 消息，且保证 tool_calls 与 tool 应答**双向**成对。

        OpenAI 的两条硬性约束都要满足，否则下一次请求直接 400：

        - ``role=tool`` 的消息必须能在前文找到同 id 的 tool_call——直接切片会把开头的
          tool 消息切成孤儿；
        - 带 ``tool_calls`` 的 assistant 消息必须有对应的 tool 应答——正常流程里应答紧跟
          在后面（切片是后缀，不会切散），但工具预算在半途耗尽时会留下没应答的调用，
          这条消息也不能进下一次请求。
        """
        body = [message for message in self.content if message.role is not MessageRole.SYSTEM]
        window = body[-limit:] if limit > 0 else []

        answered = {
            message.tool_call_id
            for message in window
            # role=tool 一定带 tool_call_id（Message 的校验器保证），这里显式过滤只为收窄类型
            if message.role is MessageRole.TOOL and message.tool_call_id is not None
        }
        requested = {
            call.id
            for message in window
            if message.role is MessageRole.ASSISTANT
            for call in message.tool_calls
        }
        return [
            message
            for message in window
            if _is_protocol_closed(message, requested=requested, answered=answered)
        ]


class ContextManager:
    """黑板的唯一写入通道，兼阶段消息装配。"""

    SUMMARY_WRITABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"intent", "related_task_ids", "operations", "result", "output"}
    )
    """``status`` / ``task_id`` / ``content`` 不在其中：状态只能经状态机改，
    ``content`` 是用户原文，改了就无从追溯。"""

    def __init__(self, settings: AppSettings, knowledge: KnowledgeMemory) -> None:
        self._settings = settings
        self._knowledge = knowledge

    # ---------------------------------------------------------------- 写入

    def create_window(self, task: Task) -> ContextWindow:
        """建窗：写入系统提示词与用户原始请求。"""
        window = ContextWindow.for_task(task)
        window.content.append(Message.system(self._knowledge.task_system_prompt()))
        window.content.append(Message.user(task.summary.content))
        return window

    def append_message(self, window: ContextWindow, message: Message) -> None:
        window.content.append(message)

    def update_summary(self, window: ContextWindow, **fields: Any) -> None:
        """按白名单更新 task summary。"""
        unknown = set(fields) - self.SUMMARY_WRITABLE_FIELDS
        if unknown:
            raise ValueError(
                f"summary 字段 {sorted(unknown)} 不允许在此写入；"
                f"可写字段：{sorted(self.SUMMARY_WRITABLE_FIELDS)}"
            )
        for key, value in fields.items():
            setattr(window.summary, key, value)

    def amend_user_request(self, window: ContextWindow, supplement: str) -> None:
        """澄清环节：把用户补充的信息并入原始请求。

        这是**唯一**允许改写 ``summary.content`` 的入口，而且是有名字的一个操作，
        不是从白名单上开个口子——澄清之后重新做意图识别，模型必须看到补充信息，
        否则它只会再问一遍同样的问题。
        """
        window.summary.content = f"{window.summary.content}\n补充说明：{supplement}"
        window.content.append(Message.user(supplement))

    def record_tool_result(
        self,
        window: ContextWindow,
        *,
        tool_call_id: str,
        tool_name: str,
        result: ToolResult,
    ) -> str:
        """把全量结果写入黑板，返回**回给模型的预览文本**。"""
        window.tool_results[tool_call_id] = ToolResultRecord(
            tool_name=tool_name,
            tool_result=result.data if result.ok else result.error,
            ok=result.ok,
        )
        return self.build_preview(tool_call_id, result)

    def read_tool_result(self, window: ContextWindow, tool_call_id: str) -> Any:
        record = window.tool_results.get(tool_call_id)
        return None if record is None else record.tool_result

    # ---------------------------------------------------------------- 预览

    def build_preview(self, tool_call_id: str, result: ToolResult) -> str:
        """结果预览：一句概括 + 前若干条样本，并告诉模型怎么取全量。"""
        if not result.ok:
            return f"[tool_call_id={tool_call_id}] 调用失败：{result.error}"

        lines = [f"[tool_call_id={tool_call_id}] {result.summary}"]
        samples = _sample_rows(result.data, self._settings.tool.preview_max_rows)
        if samples:
            lines.append(f"样本（前 {len(samples)} 条）：")
            lines.extend(json.dumps(row, ensure_ascii=False) for row in samples)
            lines.append(
                f"完整结果已存入黑板，需要全部数据时调用 "
                f'fetch_tool_result(tool_call_id="{tool_call_id}")。'
            )
        text = "\n".join(lines)
        limit = self._settings.tool.preview_max_chars
        if len(text) > limit:
            text = text[:limit] + "…（预览已截断，完整结果用 fetch_tool_result 取回）"
        return text

    # ---------------------------------------------------------------- 阶段消息装配

    def build_intent_messages(
        self,
        window: ContextWindow,
        history_summaries: list[dict[str, Any]],
    ) -> list[Message]:
        """第一步：只投喂当前 summary 的 content/status 与历史任务摘要。

        不投喂 task_content——意图识别只需要"要做什么"，喂完整对话既费 token
        又容易让模型跑偏去回答上一轮的问题。
        """
        limit = self._settings.context.history_summary_limit
        payload = {
            "current_task": window.summary.to_prompt_dict(TaskSummary.INTENT_PROMPT_FIELDS),
            "history": history_summaries[-limit:] if limit else [],
        }
        return [
            Message.system(self._knowledge.intent_prompt()),
            Message.user(json.dumps(payload, ensure_ascii=False, indent=2)),
        ]

    def build_plan_messages(
        self,
        window: ContextWindow,
        related_contents: dict[str, list[Message]] | None = None,
    ) -> list[Message]:
        """第二步：系统提示词按意图组装，并注入关联任务的历史对话。"""
        intent = window.summary.intent or IntentType.QUERY
        messages = [Message.system(self._knowledge.plan_prompt(intent))]

        related_block = self._render_related(related_contents or {})
        if related_block:
            messages.append(Message.user(related_block))

        messages.append(Message.user(f"用户请求：{window.summary.content}\n请规划执行步骤。"))
        return messages

    def build_execute_messages(
        self,
        window: ContextWindow,
        operation: Operation,
        *,
        final_round: bool = False,
    ) -> list[Message]:
        """第三步：系统提示词 + 近 K 条本任务对话 + 当前步骤指令。

        指令里显式写出「建议工具」，模型可以照做也可以自己换——规划只是建议。
        ``final_round=True`` 时改成收敛指令：这一轮不发工具清单，要求它用手上的数据
        直接作答，避免"一路换关键字再试"把轮次耗光。
        """
        intent = window.summary.intent or IntentType.QUERY
        total = len(window.summary.operations)
        suggested = operation.suggested_tool or "无"
        instruction = (
            f"当前步骤 {operation.index + 1}/{total}：{operation.description}\n"
            f"建议工具：{suggested}\n"
            f"用户的原始请求是：{window.summary.content}"
        )
        if final_round:
            instruction = (
                f"当前步骤 {operation.index + 1}/{total}：{operation.description}\n"
                f"用户的原始请求是：{window.summary.content}\n"
                f"**这是本步骤的最后一轮，不要再调用工具。**"
                f"请基于上面已经拿到的工具结果，直接给出这一步的结论；"
                f"如果数据不足，就说明已知的部分和还缺什么。"
            )
        return [
            Message.system(self._knowledge.execute_prompt(intent)),
            *window.recent_content(self._settings.context.recent_content_limit),
            Message.user(instruction),
        ]

    def build_validate_messages(self, window: ContextWindow) -> list[Message]:
        """第四步：投喂 summary 全貌 + 工具结果概览，判断是否满足用户请求。"""
        payload = {
            "user_request": window.summary.content,
            "intent": window.summary.intent.value if window.summary.intent else None,
            "operations": [
                {
                    "step": op.index + 1,
                    "description": op.description,
                    "status": op.status.value,
                    "result": op.result,
                }
                for op in window.summary.operations
            ],
            "tool_results": [
                {"tool_call_id": call_id, "tool_name": record.tool_name, "ok": record.ok}
                for call_id, record in window.tool_results.items()
            ],
        }
        return [
            Message.system(self._knowledge.validate_prompt()),
            Message.user(json.dumps(payload, ensure_ascii=False, indent=2)),
        ]

    # ---------------------------------------------------------------- 内部

    def _render_related(self, related_contents: dict[str, list[Message]]) -> str:
        """把关联任务的历史对话压成一段文本。

        用 user 角色注入而不是原样塞回 assistant/tool 消息：那些消息属于**别的任务**，
        混进当前对话会让模型以为工具刚刚被调用过，进而跳过本该做的取数。
        """
        limit = self._settings.context.related_content_limit
        blocks: list[str] = []
        for task_id, messages in related_contents.items():
            usable = [
                message
                for message in messages
                if message.role in (MessageRole.USER, MessageRole.ASSISTANT) and message.content
            ]
            if not usable:
                continue
            lines = [f"[关联任务 {task_id}]"]
            lines.extend(
                f"{message.role.value}: {message.content}" for message in usable[-limit:] if limit
            )
            blocks.append("\n".join(lines))
        if not blocks:
            return ""
        header = "以下是相关历史任务的上下文，供你参考（它们已经执行完毕）：\n\n"
        return header + "\n\n".join(blocks)


def _is_protocol_closed(
    message: Message,
    *,
    requested: set[str],
    answered: set[str],
) -> bool:
    """这条消息在当前窗口里是否"配得上对"。"""
    if message.role is MessageRole.TOOL:
        return message.tool_call_id in requested
    if message.tool_calls:
        return all(call.id in answered for call in message.tool_calls)
    return True


def _sample_rows(data: Any, max_rows: int) -> list[Any]:
    """从工具结果里取几行有代表性的样本给模型看。"""
    if isinstance(data, dict):
        for key in ("records", "groups"):
            value = data.get(key)
            if isinstance(value, list):
                return value[:max_rows]
        return []
    if isinstance(data, list):
        return data[:max_rows]
    return []
