"""MemoryManager：会话级 WorkingMemory 与外置 KnowledgeMemory 的单一入口。"""

from __future__ import annotations

from pathlib import Path

from agent.memory_manage.knowledge_memory import KnowledgeMemory
from agent.memory_manage.working_memory import WorkingMemory
from agent.store.base import Store

__all__ = ["MemoryManager"]


class MemoryManager:
    """上层只依赖本门面，不分别构造 WorkingMemory / KnowledgeMemory。"""

    def __init__(
        self,
        store: Store,
        *,
        knowledge: KnowledgeMemory | None = None,
        knowledge_root: Path | str | None = None,
    ) -> None:
        self._store = store
        self._knowledge = knowledge if knowledge is not None else KnowledgeMemory(knowledge_root)

    def working(self, session_id: str) -> WorkingMemory:
        return WorkingMemory(session_id, self._store)

    @property
    def knowledge(self) -> KnowledgeMemory:
        return self._knowledge
