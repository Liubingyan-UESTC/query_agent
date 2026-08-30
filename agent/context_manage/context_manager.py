"""ContextManager：Context Window 的唯一写入通道。

窗口活在 Store 里，不走 WorkingMemory——那是终态归档。`load_window` 只有
`task_id`，而键规范带 session，所以用 `global/ctx_index/{task_id}` 反查。

`update_summary` 只接受白名单字段：LLM 脏字段和 `status` / `task_id` /
`content` 一律拒绝。改状态是 Task.record_status 的事，原查询不能被模型改写。

`add_artifact` 只接受 CURRENT：全量进 artifacts，content 只追加引用 + preview。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from agent.common.enums import ContextScope, TaskStatus
from agent.common.errors import ContextError
from agent.config.settings import AppSettings, ContextSettings
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task_summary import TaskSummary
from agent.store.base import Store
from agent.store.keys import make_key

__all__ = [
    "INDEX_SESSION",
    "NS_INDEX",
    "NS_WINDOW",
    "SUMMARY_WRITABLE_FIELDS",
    "ContextManager",
]

NS_WINDOW = "context_window"
NS_INDEX = "ctx_index"
INDEX_SESSION = "global"

# 阶段处理器 / 模型可写的 summary 字段。status 冻结走 record_status；
# content 是用户原文；task_id 是身份。三者都不允许从这里改。
SUMMARY_WRITABLE_FIELDS: frozenset[str] = frozenset(
    {"intent", "related_task_ids", "operations", "result", "output"}
)


class ContextManager:
    """黑板的唯一写入入口。上层改窗口必须经这里，不要直接 `window.append_message`。"""

    def __init__(
        self,
        store: Store,
        settings: ContextSettings | AppSettings | None = None,
    ) -> None:
        self._store = store
        if isinstance(settings, AppSettings):
            self.settings = settings.context
        elif settings is None:
            # from_env(None) 避免构造期去读开发者本地 .env
            self.settings = ContextSettings.from_env(None)
        else:
            self.settings = settings

    def create_window(
        self,
        task_id: str,
        session_id: str,
        query: str,
        system_prompt: str,
    ) -> ContextWindow:
        """创建窗口：status=CREATED，第一条消息是 system。不创建 Task 实体。"""
        if not query or not query.strip():
            raise ContextError("创建窗口时 query 不能为空")
        if not system_prompt or not system_prompt.strip():
            raise ContextError("创建窗口时 system_prompt 不能为空")
        if self._store.get(self._index_key(task_id)) is not None:
            raise ContextError(f"任务 {task_id} 的窗口已存在")
        window = ContextWindow(
            task_id=task_id,
            session_id=session_id,
            summary=TaskSummary(task_id=task_id, content=query, status=TaskStatus.CREATED),
        )
        window.append_message(Message.system(system_prompt))
        self.save_window(window)
        return window

    def load_window(self, task_id: str) -> ContextWindow:
        session_id = self._store.get(self._index_key(task_id))
        if session_id is None:
            raise ContextError(f"没有任务 {task_id} 的窗口")
        payload = self._store.get(self._window_key(str(session_id), task_id))
        if payload is None:
            raise ContextError(f"任务 {task_id} 有索引但缺少窗口")
        if not isinstance(payload, dict):
            raise ContextError(f"任务 {task_id} 的窗口载荷不是对象")
        try:
            return ContextWindow.from_dict(payload)
        except ValidationError as exc:
            raise ContextError(f"任务 {task_id} 的窗口无法还原", detail=str(exc)) from exc

    def save_window(self, window: ContextWindow) -> None:
        index_key = self._index_key(window.task_id)
        window_key = self._window_key(window.session_id, window.task_id)
        indexed = self._store.get(index_key)
        if indexed is not None and str(indexed) != window.session_id:
            raise ContextError(
                f"任务 {window.task_id} 已绑定 session={indexed!r}，"
                f"不能改存到 session={window.session_id!r}"
            )
        self._store.set(index_key, window.session_id)
        self._store.set(window_key, window.to_dict())

    def update_summary(self, task_id: str, **fields: Any) -> None:
        if not fields:
            raise ContextError("update_summary 必须指定字段")
        unknown = sorted(set(fields) - SUMMARY_WRITABLE_FIELDS)
        if unknown:
            raise ContextError(
                f"summary 字段不在白名单内：{unknown}；可写字段为 {sorted(SUMMARY_WRITABLE_FIELDS)}"
            )
        window = self.load_window(task_id)
        payload = window.summary.to_dict()
        payload.update(fields)
        try:
            window.summary = TaskSummary.from_dict(payload)
        except ValidationError as exc:
            raise ContextError("summary 字段类型不合法", detail=str(exc)) from exc
        self.save_window(window)

    def append_message(self, task_id: str, message: Message) -> None:
        self._mutate(task_id, lambda window: window.append_message(message))

    def add_artifact(self, task_id: str, artifact: Artifact) -> str:
        """全量入 artifacts，content 只追加引用与 preview。只接受 CURRENT。"""
        if artifact.scope is not ContextScope.CURRENT:
            raise ContextError(
                f"add_artifact 只接受 scope=current 的产物，实际为 {artifact.scope.value}；"
                "注入片段请走 inject_segment"
            )

        def _put(window: ContextWindow) -> None:
            if artifact.artifact_id in window.artifacts:
                raise ContextError(f"窗口 {task_id} 已有 artifact_id={artifact.artifact_id}")
            window.put_artifact(artifact)
            window.append_message(_preview_message(artifact))

        self._mutate(task_id, _put)
        return artifact.artifact_id

    def inject_segment(
        self,
        task_id: str,
        messages: Sequence[Message],
        artifacts: Sequence[Artifact],
        source_task_id: str,
        scope: ContextScope,
    ) -> None:
        """写入关联/历史片段。不自动追加预览——调用方（步骤 22）自己组织消息。"""
        if scope is ContextScope.CURRENT:
            raise ContextError("inject_segment 不能写入 scope=current，那会与本任务归档混在一起")
        if not source_task_id:
            raise ContextError("inject_segment 必须提供 source_task_id")
        if not messages and not artifacts:
            raise ContextError("inject_segment 的 messages 与 artifacts 不能同时为空")

        def _inject(window: ContextWindow) -> None:
            for message in messages:
                window.append_message(
                    message.model_copy(update={"scope": scope, "source_task_id": source_task_id})
                )
            for artifact in artifacts:
                stamped = artifact.model_copy(
                    update={"scope": scope, "source_task_id": source_task_id}
                )
                if stamped.artifact_id in window.artifacts:
                    raise ContextError(f"窗口 {task_id} 已有 artifact_id={stamped.artifact_id}")
                window.put_artifact(stamped)

        self._mutate(task_id, _inject)

    def snapshot(self, task_id: str) -> dict[str, Any]:
        """只读快照：summary + 消息 + 产物目录，不含 artifact.data。"""
        window = self.load_window(task_id)
        return {
            "task_id": window.task_id,
            "session_id": window.session_id,
            "summary": window.summary.to_dict(),
            "messages": [message.to_dict() for message in window.content],
            "artifacts": [item.to_dict() for item in window.list_artifact_index(scope=None)],
        }

    def _mutate(self, task_id: str, mutator: Callable[[ContextWindow], None]) -> None:
        window = self.load_window(task_id)
        try:
            mutator(window)
        except ContextError:
            raise
        except ValueError as exc:
            raise ContextError(str(exc)) from exc
        self.save_window(window)

    def _index_key(self, task_id: str) -> str:
        try:
            return make_key(INDEX_SESSION, NS_INDEX, task_id)
        except ValueError as exc:
            raise ContextError(str(exc)) from exc

    def _window_key(self, session_id: str, task_id: str) -> str:
        try:
            return make_key(session_id, NS_WINDOW, task_id)
        except ValueError as exc:
            raise ContextError(str(exc)) from exc


def _preview_message(artifact: Artifact) -> Message:
    return Message.assistant(
        f"[产物引用] artifact_id={artifact.artifact_id}\n{artifact.preview()}",
        artifact_refs=[artifact.artifact_id],
    )
