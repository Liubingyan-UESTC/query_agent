"""全局枚举与常量：固化状态、意图、角色等取值域。

两个贯穿性决策：

- **一律继承 `StrEnum`**：成员值即小写字符串，`json.dumps` 可直接序列化，
  且 `TaskStatus.CREATED == "created"` 成立，跨进程/跨语言传递时无需额外转换层。
- **一律支持容错解析**：取值域的输入来源包含 LLM 输出与 HTTP 请求体，二者都不可信。
  `_missing_` 钩子统一做大小写、分隔符、驼峰归一，再查别名表，
  因此 `IntentType("NEW-QUERY")`、`IntentType("newQuery")`、`IntentType("newquery")`
  都能落到 `IntentType.NEW_QUERY`。

别名表只收录三类来源，不做无根据的扩展：v0 代码中的既有笔误、
OpenAI 协议的历史名称、以及模型输出中高频出现的同义词。
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

__all__ = [
    "ACTIVE_TASK_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "AgentEnum",
    "ArtifactType",
    "ContextScope",
    "IntentType",
    "MessageRole",
    "OperationStatus",
    "TaskStatus",
]

# 驼峰边界：小写字母或数字后紧跟大写字母处插入下划线
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# 空白、连字符、点号一律视作下划线
_SEPARATORS = re.compile(r"[\s\-.]+")


class AgentEnum(StrEnum):
    """全部枚举的基类：小写字符串取值 + 容错解析。"""

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        """非规范写法 → 规范成员值。子类按需覆盖。"""
        return {}

    @classmethod
    def normalize(cls, raw: str) -> str:
        """把任意书写风格归一为 snake_case 小写。"""
        return _SEPARATORS.sub("_", _CAMEL_BOUNDARY.sub("_", raw.strip())).lower()

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        """枚举按值查找失败时的兜底，使 `Cls(raw)` 本身即具备容错能力。"""
        if not isinstance(value, str):
            return None
        normalized = cls.normalize(value)
        canonical = cls._aliases().get(normalized, normalized)
        for member in cls:
            if member.value == canonical:
                return member
        return None

    @classmethod
    def from_str(cls, value: object, *, default: Self | None = None) -> Self:
        """容错解析。无法识别时返回 `default`，未给 `default` 则抛 `ValueError`。

        故意抛标准 `ValueError` 而非 `AgentError`：取值域解析属于通用数据校验，
        该失败该归入 `LLMResponseFormatError` 还是 HTTP 400，只有调用方知道。
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            member = cls._missing_(value)
            if member is not None:
                return member
        if default is not None:
            return default
        raise ValueError(f"无法解析为 {cls.__name__}：{value!r}；合法取值为 {cls.values()}")

    @classmethod
    def values(cls) -> list[str]:
        """全部规范取值。供构造 JSON Schema 的 enum 约束与接口文档使用。"""
        return [member.value for member in cls]


class TaskStatus(AgentEnum):
    """任务状态。转移规则由步骤 19 的 `TaskStateMachine` 裁定，本处只固化取值域。"""

    CREATED = "created"
    INTENDING = "intending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    WAITING_USER = "waiting_user"
    RETRYING = "retrying"

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        return {
            # 双写 L 是英美拼写差异，模型输出中两种都常见
            "cancelled": "canceled",
            "cancel": "canceled",
            # require.md 中的既有拼写
            "intenting": "intending",
            "intent": "intending",
            "done": "completed",
            "complete": "completed",
            "success": "completed",
            "error": "failed",
            "fail": "failed",
            "retry": "retrying",
            "waiting_for_user": "waiting_user",
            "wait_user": "waiting_user",
        }

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_TASK_STATUSES


class IntentType(AgentEnum):
    """用户请求意图。决定装配哪套系统提示词、放开哪些工具。

    原名 `TaskType`（见 `agent/task_manage/type.py` 的兼容重导出）。
    """

    NEW_QUERY = "new_query"
    ANALYSIS = "analysis"
    EXPORT = "export"
    CHAT = "chat"
    UNKNOWN = "unknown"

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        return {
            # v0 的 TaskType 成员拼写
            "newquery": "new_query",
            "analisis": "analysis",
            # 模型输出中的高频同义词
            "query": "new_query",
            "search": "new_query",
            "new": "new_query",
            "analyse": "analysis",
            "analyze": "analysis",
            "statistics": "analysis",
            "download": "export",
            "conversation": "chat",
            "other": "unknown",
            "none": "unknown",
        }


class MessageRole(AgentEnum):
    """对话角色。取值与 OpenAI messages 协议一致，可直接投喂模型。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        return {
            # require.md 中把该角色写作 Assist
            "assist": "assistant",
            "ai": "assistant",
            "model": "assistant",
            "human": "user",
            # OpenAI 早期协议中工具消息的角色名
            "function": "tool",
        }


class ArtifactType(AgentEnum):
    """工具产物类型。决定 `Artifact.preview()` 的摘要方式与前端渲染形态。"""

    TABLE = "table"
    SCALAR = "scalar"
    CHART = "chart"
    FILE = "file"
    TEXT = "text"

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        return {
            "dataframe": "table",
            "rows": "table",
            "records": "table",
            "number": "scalar",
            "metric": "scalar",
            "figure": "chart",
            "plot": "chart",
            "document": "file",
            "string": "text",
        }


class OperationStatus(AgentEnum):
    """`task_summary.operations` 中单个步骤的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        return {
            "success": "succeeded",
            "succeed": "succeeded",
            "ok": "succeeded",
            "done": "succeeded",
            "error": "failed",
            "failure": "failed",
            "in_progress": "running",
            "executing": "running",
            "todo": "pending",
            "skip": "skipped",
        }

    @property
    def is_finished(self) -> bool:
        return self in {OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.SKIPPED}


class ContextScope(AgentEnum):
    """上下文片段的归属，用于窗口裁剪优先级与归档过滤。

    只有 `CURRENT` 会被写回 WorkingMemory：`RELATED`（关联任务注入）与
    `HISTORY`（历史片段）参与推理但不参与归档，否则会话记忆将随轮次指数膨胀。
    """

    CURRENT = "current"
    RELATED = "related"
    HISTORY = "history"

    @classmethod
    def _aliases(cls) -> Mapping[str, str]:
        return {
            "injected": "related",
            "historical": "history",
            "past": "history",
        }


TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED}
)
"""终态：不允许再发生任何转移。步骤 19 的状态机以此为唯一依据。"""

ACTIVE_TASK_STATUSES: frozenset[TaskStatus] = frozenset(TaskStatus) - TERMINAL_TASK_STATUSES
"""非终态：任务仍在推进中，可被取消。"""
