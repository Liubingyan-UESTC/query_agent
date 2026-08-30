"""MemoryManager：WorkingMemory（会话级）与 KnowledgeMemory（外部知识）。

本层依赖 common / config / models / store，与 llm / tool_manage 互不引用。
知识资源文件在 `agent.knowledge`，本包只负责加载与校验，不在 import 时拉入该包。
"""

from agent.memory_manage.knowledge_memory import KnowledgeMemory, Skill
from agent.memory_manage.memory_manager import MemoryManager
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
    "KnowledgeMemory",
    "MemoryManager",
    "Skill",
    "WorkingMemory",
]
