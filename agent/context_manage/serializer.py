"""把阶段视图序列化为 OpenAI messages 列表。

`build_messages` 是步骤 14 对外的主入口：内部走窗口策略装配 + 预算裁剪，
再 `to_llm_dict()`。编排字段（scope / artifact_refs / message_id）不会出舱。
"""

from __future__ import annotations

from typing import Any

from agent.common.enums import IntentType, PromptStage
from agent.config.settings import ContextSettings
from agent.context_manage.token_counter import TokenCounter
from agent.context_manage.window_policy import (
    KnowledgeView,
    StageAssembleOptions,
    TokenBudget,
    assemble_stage_messages,
)
from agent.models.context_window import ContextWindow
from agent.models.message import Message

__all__ = ["build_message_objects", "build_messages"]


def build_message_objects(
    window: ContextWindow,
    stage: PromptStage | str,
    knowledge: KnowledgeView,
    *,
    intent: IntentType | str | None = None,
    settings: ContextSettings | None = None,
    budget: TokenBudget | None = None,
    counter: TokenCounter | None = None,
    options: StageAssembleOptions | None = None,
) -> list[Message]:
    """返回裁剪后的 Message 列表，供单测断言对话合法性。"""
    return assemble_stage_messages(
        window,
        stage,
        knowledge,
        intent=intent,
        settings=settings,
        budget=budget,
        counter=counter,
        options=options,
    )


def build_messages(
    window: ContextWindow,
    stage: PromptStage | str,
    knowledge: KnowledgeView,
    *,
    intent: IntentType | str | None = None,
    settings: ContextSettings | None = None,
    budget: TokenBudget | None = None,
    counter: TokenCounter | None = None,
    options: StageAssembleOptions | None = None,
) -> list[dict[str, Any]]:
    """计划原文签名的落地：窗口 + 阶段 + 知识 → OpenAI messages。"""
    return [
        message.to_llm_dict()
        for message in build_message_objects(
            window,
            stage,
            knowledge,
            intent=intent,
            settings=settings,
            budget=budget,
            counter=counter,
            options=options,
        )
    ]
