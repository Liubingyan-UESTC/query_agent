"""会话级 WorkingMemory：三部分历史，按 task_id 索引、按写入顺序排列。

`archive` 只落 `scope=CURRENT` 的 content / artifacts，summary 始终是本任务
名片。RELATED / HISTORY 是注入片段，再归档会让记忆随轮次指数膨胀。

写入顺序不依赖 `keys()`：Redis KEYS 无序。独立 List 键 `wm_index/task_ids`
才是 `list_task_ids` / `trim` / `get_summary_history` 的顺序来源。
`get_task_bundle` 是步骤 22 关联任务注入的唯一读口。
"""

from __future__ import annotations

from typing import Any

from agent.common.enums import ContextScope
from agent.common.errors import AgentMemoryError
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task_summary import TaskSummary
from agent.store.base import Store
from agent.store.keys import make_key

__all__ = [
    "NS_ARTIFACT",
    "NS_CONTENT",
    "NS_INDEX",
    "NS_SUMMARY",
    "WorkingMemory",
]

NS_SUMMARY = "task_summary_history"
NS_CONTENT = "task_content_history"
NS_ARTIFACT = "task_artifact_history"
NS_INDEX = "wm_index"
_INDEX_ID = "task_ids"


class WorkingMemory:
    """单个会话的三部分历史。不同 session 的键互相隔离。"""

    def __init__(self, session_id: str, store: Store) -> None:
        if not session_id or ":" in session_id:
            raise ValueError(f"session_id 不能为空或含冒号，实际为 {session_id!r}")
        self.session_id = session_id
        self._store = store

    def archive(self, window: ContextWindow) -> None:
        """写入当前任务的三部分。重复归档覆盖数据、不改变写入顺序。"""
        if window.session_id != self.session_id:
            raise AgentMemoryError(
                f"不能把 session={window.session_id!r} 的窗口归档到 session={self.session_id!r}"
            )
        current = window.split_by_scope()[ContextScope.CURRENT]
        self._store.set(self._summary_key(window.task_id), window.summary.to_dict())
        self._store.set(
            self._content_key(window.task_id),
            [message.to_dict() for message in current.messages],
        )
        self._store.set(
            self._artifact_key(window.task_id),
            [artifact.to_dict() for artifact in current.artifacts.values()],
        )
        if window.task_id not in self.list_task_ids():
            self._store.push(self._order_key(), window.task_id)

    def get_summary_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """按写入顺序返回摘要 dict。`limit` 取最近 N 条；`None` 为全量。"""
        if limit is not None and limit < 0:
            raise ValueError("limit 不能为负")
        task_ids = self.list_task_ids()
        if limit is not None:
            task_ids = task_ids[-limit:]
        history: list[dict[str, Any]] = []
        for task_id in task_ids:
            payload = self._store.get(self._summary_key(task_id))
            if payload is None:
                raise AgentMemoryError(
                    f"会话 {self.session_id} 的任务 {task_id} 有索引但缺少 summary"
                )
            history.append(payload)
        return history

    def get_content_history(self, task_id: str) -> list[Message]:
        payload = self._store.get(self._content_key(task_id))
        if not payload:
            return []
        return [Message.from_dict(item) for item in payload]

    def get_artifact_history(self, task_id: str) -> list[Artifact]:
        payload = self._store.get(self._artifact_key(task_id))
        if not payload:
            return []
        return [Artifact.from_dict(item) for item in payload]

    def get_task_bundle(self, task_id: str) -> tuple[TaskSummary, list[Message], list[Artifact]]:
        payload = self._store.get(self._summary_key(task_id))
        if payload is None:
            raise AgentMemoryError(f"会话 {self.session_id} 没有归档任务 {task_id}")
        return (
            TaskSummary.from_dict(payload),
            self.get_content_history(task_id),
            self.get_artifact_history(task_id),
        )

    def list_task_ids(self) -> list[str]:
        return [str(item) for item in self._store.range(self._order_key())]

    def trim(self, max_tasks: int) -> None:
        """淘汰最旧的任务，直到数量不超过 `max_tasks`。"""
        if max_tasks < 0:
            raise ValueError("max_tasks 不能为负")
        ids = self.list_task_ids()
        overflow = len(ids) - max_tasks
        if overflow <= 0:
            return
        for task_id in ids[:overflow]:
            self._forget(task_id)
        self._store.trim(self._order_key(), overflow, -1)

    def _forget(self, task_id: str) -> None:
        self._store.delete(self._summary_key(task_id))
        self._store.delete(self._content_key(task_id))
        self._store.delete(self._artifact_key(task_id))

    def _summary_key(self, task_id: str) -> str:
        return make_key(self.session_id, NS_SUMMARY, task_id)

    def _content_key(self, task_id: str) -> str:
        return make_key(self.session_id, NS_CONTENT, task_id)

    def _artifact_key(self, task_id: str) -> str:
        return make_key(self.session_id, NS_ARTIFACT, task_id)

    def _order_key(self) -> str:
        return make_key(self.session_id, NS_INDEX, _INDEX_ID)
