"""日志系统：一处配置，控制台与文件双写，按大小轮转。

格式固定为四列，正好对应"什么时候、多严重、任务在哪个状态、发生了什么"：

.. code-block:: text

    2026-09-04 10:12:33 | INFO    | agent.task | executing  | 状态转移 planning→executing
    2026-09-04 10:12:34 | INFO    | agent.tool | executing  | search_tool → ok，命中 19 条
    2026-09-04 10:12:36 | WARNING | agent.task | retrying   | 临时错误，退避 1.0s 后重试

第三列是**任务状态**：它随事件变化，不属于 logger 的固有属性，所以走 ``extra`` 传入，
再用一个 Filter 给没带这个字段的记录（Django 请求日志等）补上 ``-``。否则格式化会直接
抛 KeyError——第三方库往我们的 handler 里写日志时必然不带这个字段。

``setup_logging`` 幂等：重复调用不会把 handler 叠加成重复输出（Django 的 autoreload
会把 settings 模块导入两次，这条保证是必需的）。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

from agent.config import AppSettings, LogSettings

if TYPE_CHECKING:
    from agent.events import TaskEvent

__all__ = ["LoggingListener", "get_logger", "setup_logging"]

ROOT_LOGGER_NAME = "agent"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-12s | %(task_status)-10s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# handler 上打这个标记，重复 setup 时才认得出"这是我装的"
_MARK = "_agent_handler"


class _TaskStatusFilter(logging.Filter):
    """给没有 task_status 的记录补一个占位，避免格式化时 KeyError。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "task_status"):
            record.task_status = "-"
        return True


def setup_logging(settings: AppSettings | LogSettings, *, force: bool = False) -> logging.Logger:
    """配置 ``agent`` 这棵 logger 树，返回根 logger。

    - 目录不存在就建（含父目录）——这是 Django 启动时"确保 ./logs 存在"的落点；
    - 控制台走 **stderr**：CLI 的问答输出在 stdout，两者分开，重定向时互不干扰。
    """
    log = settings.log if isinstance(settings, AppSettings) else settings
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(log.level)
    # 不向 root 冒泡：否则 Django/pytest 自带的 root handler 会让每条日志重复一遍
    logger.propagate = False

    installed = [handler for handler in logger.handlers if getattr(handler, _MARK, False)]
    if installed and not force:
        return logger
    for handler in installed:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    status_filter = _TaskStatusFilter()

    log.resolved_dir().mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log.resolved_file(),
        maxBytes=log.max_bytes,
        backupCount=log.backup_count,
        encoding="utf-8",
    )
    for handler in _handlers(log, file_handler):
        handler.setLevel(log.level)
        handler.setFormatter(formatter)
        handler.addFilter(status_filter)
        setattr(handler, _MARK, True)
        logger.addHandler(handler)

    return logger


def _handlers(log: LogSettings, file_handler: logging.Handler) -> list[logging.Handler]:
    handlers = [file_handler]
    if log.to_console:
        handlers.append(logging.StreamHandler(sys.stderr))
    return handlers


def get_logger(name: str) -> logging.Logger:
    """取 ``agent.<name>`` 下的 logger，保证挂在同一棵树上。"""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


class LoggingListener:
    """把 :class:`~agent.events.TaskEvent` 写成日志的监听器。

    级别按事件性质分：终态里 FAILED/ABORTED 记 ERROR、CANCELED 记 WARNING、
    重试记 WARNING，其余记 INFO；工具失败也抬成 WARNING——这些正是排查时要先看的行。

    摘要一律压成**一行**并截断：模型的最终答复可能是一张几十行的 markdown 表，原样写进
    日志会把「一个事件一行」这个前提破坏掉，`grep` 与 `tail` 就都不好用了。全文在
    `tasks.summary_json` 里，日志只负责让人看到发生了什么。
    """

    def __init__(self, *, summary_max_chars: int = 300) -> None:
        self._task_logger = get_logger("task")
        self._tool_logger = get_logger("tool")
        self._summary_max_chars = summary_max_chars

    def __call__(self, event: TaskEvent) -> None:
        from agent.events import EventKind

        logger = self._tool_logger if event.kind is EventKind.TOOL_CALLED else self._task_logger
        logger.log(
            event.level,
            "%s task=%s session=%s",
            one_line(event.summary, self._summary_max_chars),
            event.task_id,
            event.session_id,
            extra={"task_status": event.status.value},
        )


def one_line(text: str, limit: int) -> str:
    """把任意文本压成一行并截断，供日志摘要使用。"""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "…"
