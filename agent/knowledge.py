"""KnowledgeMemory：模型为完成任务必须知道的外部知识。

require.md 把知识分为两类载体：**system prompt** 与 **skill**。这里统一以系统提示词的
形式提供，按"阶段 × 意图"两个维度装配：

- 阶段决定**要模型做什么**（识别意图 / 规划步骤 / 执行步骤 / 校验结果）；
- 意图决定**给它哪些背景**（查询要知道日志字段，闲聊只要知道自己是谁）。

提示词集中在这一个模块里，改口径不必翻遍编排代码。
"""

from __future__ import annotations

from agent.enums import IntentType

__all__ = ["KIBANA_FIELDS", "KnowledgeMemory"]

KIBANA_FIELDS: dict[str, str] = {
    "timestamp": "日志时间，ISO8601 字符串，如 2026-08-31T01:05:00Z",
    "level": "日志级别：ERROR / WARN / INFO / DEBUG",
    "service": (
        "服务名：order-service / payment-service / user-service / inventory-service / gateway"
    ),
    "host": "主机名：node-1 ~ node-4",
    "message": "日志正文，英文短句，如 Read timed out after 3000ms",
    "status_code": "HTTP 状态码，整数，如 200 / 500 / 502 / 504",
    "latency_ms": "耗时毫秒数，整数",
    "user_id": "用户标识，如 u_1001",
    "trace_id": "调用链标识",
}

_IDENTITY = (
    "你是一个日志查询与分析助手，服务于运维和研发同学。"
    "你能按关键字检索日志库，并对检索结果做分组统计。"
    "你只依据工具返回的数据回答，不编造日志内容。"
)

_JSON_ONLY = "只输出一个 JSON 对象，不要任何解释文字，也不要代码围栏。"


def _field_reference() -> str:
    lines = [f"- {name}：{desc}" for name, desc in KIBANA_FIELDS.items()]
    return "日志库的字段如下：\n" + "\n".join(lines)


class KnowledgeMemory:
    """按阶段与意图装配系统提示词。

    ``tool_catalog`` 由 ToolManager 现算现传，避免提示词里的工具清单与真实注册表脱节。
    """

    def __init__(self, tool_catalog: str = "") -> None:
        self.tool_catalog = tool_catalog

    # ---------------------------------------------------------------- 任务级

    def task_system_prompt(self, intent: IntentType | None = None) -> str:
        """任务创建时写进 task_content 首条 system 消息的内容。"""
        parts = [_IDENTITY]
        if intent in (IntentType.QUERY, IntentType.ANALYSIS):
            parts.append(_field_reference())
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- 阶段级

    def intent_prompt(self) -> str:
        return "\n\n".join(
            [
                _IDENTITY,
                "现在做意图识别。你会看到当前请求，以及本会话中历史任务的摘要列表。",
                "意图取值：\n"
                "- query：检索日志原文\n"
                "- analysis：对日志做统计、分组、排序\n"
                "- chat：闲聊，或询问你的身份与能力\n"
                "- unknown：信息不足以判断，需要用户澄清",
                "如果当前请求依赖某个历史任务（例如「刚才那批结果」「继续统计上面的数据」），"
                "把那些任务的 task_id 放进 related_task_ids；没有就给空数组。",
                f'输出格式：{{"intent": "...", "related_task_ids": [], "reason": "...", '
                f'"clarification": "意图为 unknown 时，写一句要问用户的话，否则留空"}}\n'
                f"{_JSON_ONLY}",
            ]
        )

    def plan_prompt(self, intent: IntentType) -> str:
        guidance = {
            IntentType.QUERY: (
                "查询类任务通常一步就够：用 search_tool 按关键字检索。"
                "关键字要选日志里真实出现的英文片段，例如 ERROR、timeout、order-service。"
            ),
            IntentType.ANALYSIS: (
                "统计类任务通常两步：先用 search_tool 取数，再用 analysis_tool 对上一步的"
                "结果分组统计。第二步必须引用第一步返回的 tool_call_id。"
            ),
            IntentType.CHAT: (
                "闲聊类任务不需要工具，规划一个「直接回答」的步骤即可，suggested_tool 填 null。"
            ),
            IntentType.UNKNOWN: (
                "信息不足时，规划一个「向用户说明还缺什么」的步骤，suggested_tool 填 null。"
            ),
        }[intent]

        parts = [_IDENTITY, "现在做任务规划：把用户请求拆成可执行的步骤。", guidance]
        if intent in (IntentType.QUERY, IntentType.ANALYSIS):
            parts.append(_field_reference())
        if self.tool_catalog:
            parts.append("可用工具：\n" + self.tool_catalog)
        parts.append(
            '输出格式：{"operations": [{"description": "这一步要做什么", '
            '"suggested_tool": "工具名或 null"}]}\n'
            f"步骤要少而精，能一步完成就不要拆成两步。{_JSON_ONLY}"
        )
        return "\n\n".join(parts)

    def execute_prompt(self, intent: IntentType) -> str:
        parts = [
            _IDENTITY,
            "现在执行当前步骤。你可以调用工具，也可以直接给出这一步的结论。",
            "规则：\n"
            "- 需要数据就调用工具，不要凭空作答；\n"
            "- 工具结果只会回给你一段预览，全量数据存在黑板上；"
            "需要完整数据时用 fetch_tool_result(tool_call_id) 取回；\n"
            "- 拿到足够的信息后，用一两句中文说明这一步的结论，不要再调用工具。",
        ]
        if intent in (IntentType.QUERY, IntentType.ANALYSIS):
            parts.append(_field_reference())
        return "\n\n".join(parts)

    def validate_prompt(self) -> str:
        return "\n\n".join(
            [
                _IDENTITY,
                "现在校验：已执行的步骤与工具结果，是否已经回答了用户最初的请求？",
                "满足就把最终答复写进 output——直接面向用户，要具体（给出数字、服务名等），"
                "不要出现「步骤」「工具」这类内部说法。\n"
                "不满足就说明还缺什么，系统会补充执行。",
                f'输出格式：{{"satisfied": true/false, "output": "给用户的最终答复", '
                f'"reason": "判定理由"}}\n{_JSON_ONLY}',
            ]
        )
