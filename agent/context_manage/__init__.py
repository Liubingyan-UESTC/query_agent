"""ContextManager：Context Window 生命周期、装配策略与关联任务注入。

本层依赖 common / config / models / store，不引用 llm / memory / tool / prompt。
知识面以 `KnowledgeView` Protocol 注入，实现类留在 memory 层。
"""

from agent.context_manage.context_manager import (
    INDEX_SESSION,
    NS_INDEX,
    NS_WINDOW,
    SUMMARY_WRITABLE_FIELDS,
    ContextManager,
)
from agent.context_manage.serializer import build_message_objects, build_messages
from agent.context_manage.token_counter import (
    CharEstimateCounter,
    TiktokenCounter,
    TokenCounter,
    default_counter,
)
from agent.context_manage.window_policy import (
    FEWSHOT_SOURCE_TASK_ID,
    HISTORY_SUMMARY_FIELDS,
    KnowledgeView,
    SkillView,
    StageAssembleOptions,
    TokenBudget,
    assemble_stage_messages,
    clip_history_summaries,
    clip_to_budget,
    group_turns,
    is_tool_error,
    last_k_turns,
)

__all__ = [
    "FEWSHOT_SOURCE_TASK_ID",
    "HISTORY_SUMMARY_FIELDS",
    "INDEX_SESSION",
    "NS_INDEX",
    "NS_WINDOW",
    "SUMMARY_WRITABLE_FIELDS",
    "CharEstimateCounter",
    "ContextManager",
    "KnowledgeView",
    "SkillView",
    "StageAssembleOptions",
    "TiktokenCounter",
    "TokenBudget",
    "TokenCounter",
    "assemble_stage_messages",
    "build_message_objects",
    "build_messages",
    "clip_history_summaries",
    "clip_to_budget",
    "default_counter",
    "group_turns",
    "is_tool_error",
    "last_k_turns",
]
