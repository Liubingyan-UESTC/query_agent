"""全局枚举与常量：固化状态、意图、角色等取值域。

一律继承 `StrEnum`：成员值即小写字符串，`json.dumps` 可直接序列化，且
`TaskStatus.CREATED == "created"` 成立，跨进程/跨语言传递时无需额外转换层。

**取值域是严格的**：只接受下列成员的规范拼写。大小写变体、连字符与驼峰写法、
同义词、笔误一律视为非法输入并抛错。不做归一化、不设别名表是有意为之——
把错误的拼写猜成正确成员会掩盖真正的缺陷（提示词写错、前端传错字段、模型不遵守
schema），而猜测规则本身会长期膨胀并需要维护。约束模型输出的正确位置是用
`values()` 生成 JSON Schema 的 enum 约束，而不是在解析端兜底。
"""

from enum import StrEnum
from typing import Self

__all__ = [
    "ACTIVE_TASK_STATUSES",
    "FINISHED_OPERATION_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "AgentEnum",
    "ArtifactType",
    "ContextScope",
    "IntentType",
    "MessageRole",
    "OperationStatus",
    "PromptStage",
    "TaskStatus",
]


class AgentEnum(StrEnum):
    """全部枚举的基类：小写字符串取值 + 严格解析。"""

    @classmethod
    def from_str(cls, value: object, *, default: Self | None = None) -> Self:
        """严格解析：只认规范取值，不做任何归一化或别名映射。

        无法识别时返回 `default`；未给 `default` 则抛 `ValueError`。
        `default` 是调用方显式选择的兜底，与「猜测错误拼写」是两回事。

        直接委托给 `cls(value)`，走枚举自带的哈希查表而不自行线性扫描成员。
        本方法相对 `cls(value)` 的唯一增量，是错误信息里带上合法取值列表
        （可直接回灌重试提示词）以及可选的 `default`。

        故意抛标准 `ValueError` 而非 `AgentError`：取值域解析属于通用数据校验，
        该失败应归入 `LLMResponseFormatError` 还是 HTTP 400，只有调用方知道。
        """
        if isinstance(value, str):
            # StrEnum 成员本身也是 str，故此分支同时覆盖「传入成员」的情形
            try:
                return cls(value)
            except ValueError:
                return cls._fallback_or_raise(value, default)
        return cls._fallback_or_raise(value, default)

    @classmethod
    def _fallback_or_raise(cls, value: object, default: Self | None) -> Self:
        """解析失败的唯一出口，避免错误信息在多处重复。"""
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


class PromptStage(AgentEnum):
    """装配系统提示词的阶段。取值与 `agent/knowledge/system_prompts/` 文件名对齐。

    不能复用 `TaskStatus`：任务状态是 `intending`，提示词文件是 `intent_recognition.md`。
    """

    INTENT_RECOGNITION = "intent_recognition"
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"


class MessageRole(AgentEnum):
    """对话角色。取值与 OpenAI messages 协议一致，可直接投喂模型。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ArtifactType(AgentEnum):
    """工具产物类型。决定 `Artifact.preview()` 的摘要方式与前端渲染形态。"""

    TABLE = "table"
    SCALAR = "scalar"
    CHART = "chart"
    FILE = "file"
    TEXT = "text"


class OperationStatus(AgentEnum):
    """`task_summary.operations` 中单个步骤的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def is_finished(self) -> bool:
        return self in FINISHED_OPERATION_STATUSES


class ContextScope(AgentEnum):
    """上下文片段的归属，用于窗口裁剪优先级与归档过滤。

    只有 `CURRENT` 会被写回 WorkingMemory：`RELATED`（关联任务注入）与
    `HISTORY`（历史片段）参与推理但不参与归档，否则会话记忆将随轮次指数膨胀。
    """

    CURRENT = "current"
    RELATED = "related"
    HISTORY = "history"


TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED}
)
"""终态：不允许再发生任何转移。步骤 19 的状态机以此为唯一依据。"""

ACTIVE_TASK_STATUSES: frozenset[TaskStatus] = frozenset(TaskStatus) - TERMINAL_TASK_STATUSES
"""非终态：任务仍在推进中，可被取消。"""

FINISHED_OPERATION_STATUSES: frozenset[OperationStatus] = frozenset(
    {OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.SKIPPED}
)
"""已结束的步骤状态：不会再被调度执行。"""
