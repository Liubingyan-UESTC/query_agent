"""MemoryManager：WorkingMemory（会话级）与 KnowledgeMemory（外部知识）。

本层依赖 common / config / models / store，与 llm / tool_manage 互不引用。
步骤 11 交付 WorkingMemory；步骤 12 再补 KnowledgeMemory 与门面。
"""

from agent.memory_manage.working_memory import (
    NS_ARTIFACT,
    NS_CONTENT,
    NS_INDEX,
    NS_SUMMARY,
    WorkingMemory,
)

__all__ = [
    "NS_ARTIFACT",
    "NS_CONTENT",
    "NS_INDEX",
    "NS_SUMMARY",
    "WorkingMemory",
]
