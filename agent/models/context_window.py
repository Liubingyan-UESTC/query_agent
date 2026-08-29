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
    def _summary_task_id_matches(self) -> "ContextWindow":
        if self.summary.task_id != self.task_id:
            raise ValueError(
                f"ContextWindow.task_id={self.task_id!r} 与 summary.task_id="
                f"{self.summary.task_id!r} 不一致"
            )
        return self

    @classmethod
    def from_task(cls, task: Task) -> "ContextWindow":
        """从任务创建空窗口，复用同一份 `summary` 对象。

        黑板语义要求窗口与任务看到同一份摘要。步骤 6 的 `record_status()`
        会用 `model_copy` 换掉 `Task.summary`，换完之后调用方必须把新摘要
        写回窗口（步骤 13 的 `update_summary` / 状态机钩子负责），本方法
        只保证出生时二者是同一对象。
        """
        return cls(task_id=task.task_id, session_id=task.session_id, summary=task.summary)

    def append_message(self, message: Message) -> None:
        if any(existing.message_id == message.message_id for existing in self.content):
            raise ValueError(f"窗口已有 message_id={message.message_id}")
        self.content = [*self.content, message]

    def get_messages(
        self,
        scope: ContextScope | None = None,
        roles: Collection[MessageRole] | None = None,
    ) -> list[Message]:
        """按 scope / 角色过滤，保持追加顺序。`None` 表示不按该维过滤。"""
        if roles is not None and len(roles) == 0:
            raise ValueError("roles 为空几乎总是漏传，若要全部角色请传 None")
        selected = self.content
        if scope is not None:
            selected = [item for item in selected if item.scope is scope]
        if roles is not None:
            allowed = set(roles)
            selected = [item for item in selected if item.role in allowed]
        return selected

    def put_artifact(self, artifact: Artifact) -> None:
        self.artifacts = {**self.artifacts, artifact.artifact_id: artifact}

    def get_artifact(self, artifact_id: str) -> Artifact:
        try:
            return self.artifacts[artifact_id]
        except KeyError:
            raise KeyError(f"窗口 {self.task_id} 没有 artifact_id={artifact_id}") from None

    def list_artifact_index(self) -> list[ArtifactIndex]:
        """产物目录，按写入顺序。不含 `data`，避免目录接口把结果集打出去。"""
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
            for item in self.artifacts.values()
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
