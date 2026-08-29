"""通用基础设施：全局枚举与状态常量、异常体系、结构化日志、ID 生成。

本层不依赖 agent 内的任何其他子包（含 config），是依赖图的根，
因此可被任意模块安全导入而不会形成环（由 tests/test_project_skeleton.py 守护）。
"""

from agent.common.enums import (
    ACTIVE_TASK_STATUSES,
    FINISHED_OPERATION_STATUSES,
    TERMINAL_TASK_STATUSES,
    AgentEnum,
    ArtifactType,
    ContextScope,
    IntentType,
    MessageRole,
    OperationStatus,
    TaskStatus,
)
from agent.common.errors import (
    AgentError,
    AgentMemoryError,
    ConfigError,
    ContextError,
    LLMError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTimeoutError,
    TaskCanceledError,
    TaskStateError,
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from agent.common.ids import (
    new_artifact_id,
    new_id,
    new_message_id,
    new_session_id,
    new_task_id,
    new_trace_id,
)
from agent.common.logging import (
    clear_log_context,
    get_log_context,
    get_logger,
    log_context,
    set_log_context,
    setup_logging,
)

__all__ = [
    "ACTIVE_TASK_STATUSES",
    "FINISHED_OPERATION_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "AgentEnum",
    "AgentError",
    "AgentMemoryError",
    "ArtifactType",
    "ConfigError",
    "ContextError",
    "ContextScope",
    "IntentType",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseFormatError",
    "LLMTimeoutError",
    "MessageRole",
    "OperationStatus",
    "TaskCanceledError",
    "TaskStateError",
    "TaskStatus",
    "ToolError",
    "ToolInvocationError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "clear_log_context",
    "get_log_context",
    "get_logger",
    "log_context",
    "new_artifact_id",
    "new_id",
    "new_message_id",
    "new_session_id",
    "new_task_id",
    "new_trace_id",
    "set_log_context",
    "setup_logging",
]
