"""纯数据模型：Message、Artifact、TaskSummary、Task、ContextWindow。

本层不依赖除 common / config 之外的任何内部模块，也不含业务策略。
"""

from agent.models.artifact import (
    DEFAULT_PREVIEW_CELL_CHARS,
    DEFAULT_PREVIEW_MAX_CHARS,
    DEFAULT_PREVIEW_ROWS,
    Artifact,
)
from agent.models.base import AgentModel
from agent.models.message import Message, ToolCall
from agent.models.task import StatusRecord, Task
from agent.models.task_summary import INTENT_PROMPT_FIELDS, Operation, TaskSummary

__all__ = [
    "DEFAULT_PREVIEW_CELL_CHARS",
    "DEFAULT_PREVIEW_MAX_CHARS",
    "DEFAULT_PREVIEW_ROWS",
    "INTENT_PROMPT_FIELDS",
    "AgentModel",
    "Artifact",
    "Message",
    "Operation",
    "StatusRecord",
    "Task",
    "TaskSummary",
    "ToolCall",
]
