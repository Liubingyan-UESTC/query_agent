"""任务事件：让"记日志"与"落库"都变成订阅者。

``TaskManager`` 在四个节点发事件，谁想知道就订阅，核心不必认识日志格式，更不必认识
数据库::

    TaskManager
       └─ emit(TaskEvent)
            ├─ LoggingListener  →  控制台 + logs/log.txt
            └─ DbListener       →  sessions / tasks / messages / tool_calls

事件上挂着 ``window``（黑板的**引用**，不是拷贝）。落库方每次事件按 ``message_id``
差量补写消息即可，不用去数"哪些地方会往窗口里追加消息"——将来加了新的追加点也不会漏。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from agent.enums import TaskStatus

if TYPE_CHECKING:
    from agent.context import ContextWindow

__all__ = ["EventKind", "TaskEvent", "TaskListener", "emit_all"]


class EventKind(StrEnum):
    TASK_CREATED = "task_created"
    """任务已创建，黑板已初始化。"""

    STATUS_CHANGED = "status_changed"
    """状态机完成一次转移。"""

    TOOL_CALLED = "tool_called"
    """一次工具调用执行完毕（成功或失败都发）。"""

    TASK_FINISHED = "task_finished"
    """任务进入终态，结果已写进 summary.output。"""


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """一个任务事件。``summary`` 是给人看的一句话，``payload`` 是给程序看的结构化字段。"""

    kind: EventKind
    task_id: str
    session_id: str
    status: TaskStatus
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    window: ContextWindow | None = None
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def level(self) -> int:
        """这条事件该用什么日志级别——排查时先看的行必须自己浮上来。"""
        if self.status in (TaskStatus.FAILED, TaskStatus.ABORTED):
            return logging.ERROR
        if self.status in (TaskStatus.CANCELED, TaskStatus.RETRYING):
            return logging.WARNING
        if self.kind is EventKind.TOOL_CALLED and not self.payload.get("ok", True):
            return logging.WARNING
        return logging.INFO


TaskListener = Callable[[TaskEvent], None]

_logger = logging.getLogger("agent.events")


def emit_all(listeners: tuple[TaskListener, ...], event: TaskEvent) -> None:
    """逐个通知监听器，**任何一个抛异常都不许影响任务本身**。

    落库失败、磁盘满了、监听器有 bug——这些都不该让用户的查询请求失败。异常记成
    ERROR 供排查，任务继续走。
    """
    for listener in listeners:
        try:
            listener(event)
        except Exception:
            name = getattr(listener, "__name__", type(listener).__name__)
            _logger.exception("事件监听器 %s 处理 %s 失败", name, event.kind.value)
