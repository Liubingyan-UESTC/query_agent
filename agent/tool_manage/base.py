"""工具抽象：声明、OpenAI schema、调用结果与只读上下文。

工具不得持有 ContextWindow 写权限，只返回 `ToolResult`，由执行阶段
经 ContextManager 落盘，保证黑板单一写入通道。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Protocol, Self

from pydantic import BaseModel

from agent.common.enums import ArtifactType
from agent.common.errors import AgentError, ToolInvocationError
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.models.error import ErrorInfo

__all__ = [
    "ArtifactsReader",
    "BaseTool",
    "FieldCatalog",
    "ToolContext",
    "ToolDeps",
    "ToolResult",
]


class FieldCatalog(Protocol):
    """字段字典的只读面。实现在 KnowledgeMemory，这里不引用 memory 层。"""

    def has_index(self, name: str) -> bool: ...

    def field_names(self, index: str) -> Collection[str]: ...


class ArtifactsReader(Protocol):
    def __call__(self, artifact_id: str) -> Artifact: ...


@dataclass(slots=True)
class ToolDeps:
    """实例化真实工具所需的外部依赖。Mock 工具忽略这些字段。"""

    settings: Any
    es_client: Any = None
    field_catalog: FieldCatalog | None = None
    export_dir: Any = None


@dataclass(slots=True)
class ToolContext:
    """工具调用的只读上下文。产物只能读，不能写回窗口。"""

    task_id: str
    session_id: str
    trace_id: str
    artifacts_reader: Callable[[str], Artifact] | None = None

    def get_artifact(self, artifact_id: str) -> Artifact:
        if self.artifacts_reader is None:
            raise ToolInvocationError(f"没有产物读取器，无法读取 artifact_id={artifact_id}")
        try:
            return self.artifacts_reader(artifact_id)
        except KeyError as exc:
            raise ToolInvocationError(f"没有 artifact_id={artifact_id}") from exc


class ToolResult(AgentModel):
    """一次工具调用的结果。`error` 用 ErrorInfo，异常对象不能进 Store。"""

    ok: bool
    artifact: Artifact | None = None
    text: str = ""
    error: ErrorInfo | None = None
    elapsed_ms: float = 0.0

    @classmethod
    def success(
        cls,
        text: str,
        artifact: Artifact | None = None,
        *,
        elapsed_ms: float = 0.0,
    ) -> Self:
        return cls(ok=True, text=text, artifact=artifact, elapsed_ms=elapsed_ms)

    @classmethod
    def fail(cls, error: AgentError | ErrorInfo, *, elapsed_ms: float = 0.0) -> Self:
        info = error if isinstance(error, ErrorInfo) else ErrorInfo.from_error(error)
        return cls(ok=False, text=info.message, error=info, elapsed_ms=elapsed_ms)


class BaseTool(ABC):
    """全部工具的共同契约。"""

    name: str
    description: str
    args_schema: type[BaseModel]
    produces: ArtifactType | None = None
    timeout: float | None = None
    is_mock: bool = False

    @abstractmethod
    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """执行工具。业务错误返回 `ToolResult(ok=False)`，不要直接写窗口。"""

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI `tools` 字段接受的 function 声明。"""
        parameters = self.args_schema.model_json_schema(by_alias=True)
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @classmethod
    def from_deps(cls, _deps: ToolDeps) -> Self:
        """默认无参构造。需要 ES / 字段目录的子类覆盖本方法。"""
        return cls()
