"""Context Window：一次任务的黑板。

三部分聚合在同一对象里，便于整体读写与整体归档：

- `summary`：任务名片（意图、步骤、结论）
- `content`：对话记录
- `artifacts`：工具产物全量，按 `artifact_id` 索引

本模块只做容器操作，不含裁剪、预算、写回策略。步骤 13 的 ContextManager 是
唯一写入通道；步骤 14 才决定哪些片段投喂模型。这里提供的 `split_by_scope()`
只是把已标好 scope 的内容分开，供写回 WorkingMemory 时只取 `CURRENT`。
"""

from collections.abc import Collection

from pydantic import Field, model_validator

from agent.common.enums import ArtifactType, ContextScope, MessageRole
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.models.message import Message
from agent.models.task import Task
from agent.models.task_summary import TaskSummary

__all__ = ["ArtifactIndex", "ContextWindow", "ScopeSlice"]


class ArtifactIndex(AgentModel):
    """产物目录项：给前端和提示词看的元数据，不含 `data`。"""

    artifact_id: str
    title: str
    artifact_type: ArtifactType
    producer: str
    row_count: int | None
    size_bytes: int | None
    scope: ContextScope


class ScopeSlice(AgentModel):
    """`split_by_scope()` 切出的一片：该 scope 下的消息与产物。"""

    messages: list[Message] = Field(default_factory=list)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)


class ContextWindow(AgentModel):
    """一次任务的工作窗口。"""

    task_id: str
    session_id: str
    summary: TaskSummary
    content: list[Message] = Field(default_factory=list)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _invariants(self) -> "ContextWindow":
        if self.summary.task_id != self.task_id:
            raise ValueError(
                f"ContextWindow.task_id={self.task_id!r} 与 summary.task_id="
                f"{self.summary.task_id!r} 不一致"
            )
        mismatched = [key for key, item in self.artifacts.items() if key != item.artifact_id]
        if mismatched:
            raise ValueError(f"artifacts 的键必须等于 artifact_id，下列键对不上：{mismatched}")
        foreign = [
            item.artifact_id
            for item in self.artifacts.values()
            if item.scope is ContextScope.CURRENT and item.task_id != self.task_id
        ]
        if foreign:
            raise ValueError(
                f"scope=current 的产物必须属于本任务，下列 artifact 的 task_id 对不上：{foreign}"
            )
        return self

    @classmethod
    def from_task(cls, task: Task) -> "ContextWindow":
        """从任务创建空窗口，复用同一份 `summary` 对象。

        `Task.record_status()` 会用 `model_copy` 换掉 `Task.summary`，出生时的
        共享引用随即断开。换完必须调用 `bind_summary(task.summary)`，否则
        Store 会持久化一份过期摘要。步骤 13 的状态回写必须走这个口。
        """
        return cls(task_id=task.task_id, session_id=task.session_id, summary=task.summary)

    def bind_summary(self, summary: TaskSummary) -> None:
        """把任务侧换新的摘要绑回窗口。`record_status` 之后必须调用。"""
        self.summary = summary

    def append_message(self, message: Message) -> None:
        if any(existing.message_id == message.message_id for existing in self.content):
            raise ValueError(f"窗口已有 message_id={message.message_id}")
        self.content = [*self.content, message]

    def get_messages(
        self,
        scope: ContextScope | None = None,
        roles: Collection[MessageRole] | None = None,
    ) -> list[Message]:
        """按 scope / 角色过滤，保持追加顺序。`None` 表示不按该维过滤。

        返回值始终是新列表：直接交出 `self.content` 会让调用方 `append`
        绕过 `append_message` 的去重。
        """
        if roles is not None and len(roles) == 0:
            raise ValueError("roles 为空几乎总是漏传，若要全部角色请传 None")
        selected = self.content
        if scope is not None:
            selected = [item for item in selected if item.scope is scope]
        if roles is not None:
            allowed = set(roles)
            selected = [item for item in selected if item.role in allowed]
        return list(selected)

    def put_artifact(self, artifact: Artifact) -> None:
        self.artifacts = {**self.artifacts, artifact.artifact_id: artifact}

    def get_artifact(self, artifact_id: str) -> Artifact:
        try:
            return self.artifacts[artifact_id]
        except KeyError:
            raise KeyError(f"窗口 {self.task_id} 没有 artifact_id={artifact_id}") from None

    def list_artifact_index(
        self, scope: ContextScope | None = ContextScope.CURRENT
    ) -> list[ArtifactIndex]:
        """产物目录，按写入顺序。默认只列 `CURRENT`，避免把关联任务的表暴露给前端。

        需要全量时显式传 `scope=None`。不含 `data`。
        """
        items = [item for item in self.artifacts.values() if scope is None or item.scope is scope]
        return [
            ArtifactIndex(
                artifact_id=item.artifact_id,
                title=item.title,
                artifact_type=item.artifact_type,
                producer=item.producer,
                row_count=item.row_count,
                size_bytes=item.size_bytes,
                scope=item.scope,
            )
            for item in items
        ]

    def split_by_scope(self) -> dict[ContextScope, ScopeSlice]:
        """按 scope 切开消息与产物。每个 scope 都有条目，避免调用方漏判空切片。

        写回 WorkingMemory 只应取 `ContextScope.CURRENT`：RELATED / HISTORY
        是注入片段，再归档会让会话记忆随轮次指数膨胀。
        """
        slices = {scope: ScopeSlice() for scope in ContextScope}
        for message in self.content:
            slices[message.scope].messages.append(message)
        for artifact in self.artifacts.values():
            slices[artifact.scope].artifacts[artifact.artifact_id] = artifact
        return slices
