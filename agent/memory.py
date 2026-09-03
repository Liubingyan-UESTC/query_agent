"""MemoryManager：会话级记忆。

require.md 把记忆分成两类：

- :class:`WorkingMemory`——当前会话里**历史任务**的上下文窗口集合，三段历史与黑板的
  三项记录一一对应。任务进入终态时整窗归档；意图识别阶段回投 ``task_summary_history``，
  执行阶段按 related_task_ids 回投 ``task_context_history``。
- ``KnowledgeMemory``——外部知识（见 :mod:`agent.knowledge`），由 :attr:`MemoryManager.knowledge`
  暴露，两者合起来就是模型"记得"的全部内容。

存储用进程内字典：没有真实数据库，会话结束即失效，这与"控制台单机跑通"的目标一致。
"""

from __future__ import annotations

from typing import Any

from agent.context import ContextWindow
from agent.knowledge import KnowledgeMemory
from agent.models import Message

__all__ = ["MemoryManager", "WorkingMemory"]


class WorkingMemory:
    """一个会话内所有已完成任务的记忆，三段历史严格对应黑板三项记录。"""

    def __init__(self, session_id: str, *, max_tasks: int = 20) -> None:
        self.session_id = session_id
        self._max_tasks = max_tasks
        self._order: list[str] = []
        self.task_summary_history: dict[str, dict[str, Any]] = {}
        self.tool_result_history: dict[str, dict[str, Any]] = {}
        self.task_context_history: dict[str, list[Message]] = {}

    # ---------------------------------------------------------------- 写入

    def archive(self, window: ContextWindow) -> None:
        """把一次任务的整个窗口写入三段历史。重复归档同一任务视为覆盖更新。"""
        task_id = window.task_id
        if task_id not in self.task_summary_history:
            self._order.append(task_id)
        self.task_summary_history[task_id] = window.summary.to_dict()
        self.tool_result_history[task_id] = {
            call_id: record.to_dict() for call_id, record in window.tool_results.items()
        }
        self.task_context_history[task_id] = list(window.content)
        self._trim()

    def _trim(self) -> None:
        while len(self._order) > self._max_tasks:
            oldest = self._order.pop(0)
            self.task_summary_history.pop(oldest, None)
            self.tool_result_history.pop(oldest, None)
            self.task_context_history.pop(oldest, None)

    # ---------------------------------------------------------------- 读取

    def task_ids(self) -> list[str]:
        """按归档顺序返回任务 id（旧 → 新）。"""
        return list(self._order)

    def summaries(self, limit: int | None = None) -> list[dict[str, Any]]:
        """意图识别阶段投喂的历史摘要列表，按时间顺序。"""
        ordered = [self.task_summary_history[task_id] for task_id in self._order]
        return ordered if limit is None else ordered[-limit:]

    def context_of(self, task_id: str) -> list[Message]:
        return list(self.task_context_history.get(task_id, []))

    def tool_results_of(self, task_id: str) -> dict[str, Any]:
        return dict(self.tool_result_history.get(task_id, {}))

    def related_contents(self, task_ids: list[str]) -> dict[str, list[Message]]:
        """按 related_task_ids 取出关联任务的对话，跳过本会话里不存在的 id。"""
        found = {}
        for task_id in task_ids:
            messages = self.task_context_history.get(task_id)
            if messages:
                found[task_id] = list(messages)
        return found

    def known_task_ids(self) -> set[str]:
        """用于过滤模型编造的 related_task_ids。"""
        return set(self._order)


class MemoryManager:
    """会话记忆的门面：``working(session_id)`` 取工作记忆，``knowledge`` 取外部知识。"""

    def __init__(self, knowledge: KnowledgeMemory, *, max_tasks: int = 20) -> None:
        self.knowledge = knowledge
        self._max_tasks = max_tasks
        self._sessions: dict[str, WorkingMemory] = {}

    def working(self, session_id: str) -> WorkingMemory:
        memory = self._sessions.get(session_id)
        if memory is None:
            memory = WorkingMemory(session_id, max_tasks=self._max_tasks)
            self._sessions[session_id] = memory
        return memory

    def archive(self, window: ContextWindow) -> None:
        self.working(window.session_id).archive(window)

    def sessions(self) -> list[str]:
        return sorted(self._sessions)
