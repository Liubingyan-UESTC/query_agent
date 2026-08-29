"""JSON 结构化日志与请求级上下文。

设计要点：

- **上下文自动注入**：`trace_id / session_id / task_id` 存放在 `contextvars` 中，由
  `ContextFilter` 注入每条日志记录。业务代码只管 `logger.info("...")`，无需层层传递
  这三个 ID，也就不会出现"某条日志忘记带 trace_id 导致链路断裂"。
- **contextvars 而非 threading.local**：既能隔离线程，也能隔离 asyncio 任务，
  适配步骤 30 的 SSE 流式接口（后台线程执行任务 + 主协程推送事件）。
- **日志不得反噬业务**：JSON 序列化对未知类型回退为 `str()`，
  一条日志写不出去也不应中断任务。

模块名与标准库 `logging` 同名不构成问题：Python 3 默认绝对导入，
本模块内的 `import logging` 取到的是标准库。
"""

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, TextIO

from agent.common.errors import AgentError, ConfigError

__all__ = [
    "ContextFilter",
    "JsonFormatter",
    "clear_log_context",
    "get_log_context",
    "get_logger",
    "log_context",
    "set_log_context",
    "setup_logging",
]

_TRACE_ID: ContextVar[str | None] = ContextVar("agent_trace_id", default=None)
_SESSION_ID: ContextVar[str | None] = ContextVar("agent_session_id", default=None)
_TASK_ID: ContextVar[str | None] = ContextVar("agent_task_id", default=None)

_CONTEXT_VARS: dict[str, ContextVar[str | None]] = {
    "trace_id": _TRACE_ID,
    "session_id": _SESSION_ID,
    "task_id": _TASK_ID,
}

# 标记由本模块安装的 handler，使 setup_logging 可重复调用而不叠加输出
_HANDLER_TAG = "_agent_json_handler"

# LogRecord 的固有属性；其余属性即调用方通过 extra= 传入的业务字段
_RESERVED_RECORD_KEYS = frozenset(
    vars(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))
) | {"message", "asctime", "taskName"}


# ============================================================ 上下文


def get_log_context() -> dict[str, str]:
    """返回当前上下文中已设置的 ID。未设置的键不出现，避免日志里出现 null 噪声。"""
    context = {}
    for name, var in _CONTEXT_VARS.items():
        value = var.get()
        if value is not None:
            context[name] = value
    return context


def _validate_fields(fields: dict[str, str | None]) -> None:
    unknown = set(fields) - set(_CONTEXT_VARS)
    if unknown:
        raise ConfigError(
            f"未知的日志上下文字段：{sorted(unknown)}；可用字段为 {sorted(_CONTEXT_VARS)}"
        )


def set_log_context(**fields: str | None) -> None:
    """设置上下文 ID，作用范围为当前线程/协程。

    适用于请求入口这类"设置后不需要还原"的场景（如步骤 26 的 TraceIdMiddleware）。
    需要精确还原时请用 `log_context` 上下文管理器。
    """
    _validate_fields(fields)
    for name, value in fields.items():
        _CONTEXT_VARS[name].set(value)


def clear_log_context() -> None:
    """清空全部上下文 ID。"""
    for var in _CONTEXT_VARS.values():
        var.set(None)


@contextmanager
def log_context(**fields: str | None) -> Iterator[None]:
    """在代码块内绑定上下文 ID，退出时精确还原为进入前的值。

    还原依赖 contextvars 的 token 而非"记下旧值再写回"，因此嵌套使用也不会互相干扰。
    """
    _validate_fields(fields)
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    try:
        for name, value in fields.items():
            var = _CONTEXT_VARS[name]
            tokens.append((var, var.set(value)))
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


# ============================================================ 格式化


class ContextFilter(logging.Filter):
    """把 contextvars 中的 ID 注入日志记录。

    已存在的同名属性不被覆盖，因此调用方通过 `extra=` 显式指定的值优先。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for name, value in get_log_context().items():
            if not hasattr(record, name):
                setattr(record, name, value)
        return True


class JsonFormatter(logging.Formatter):
    """把日志记录序列化为单行 JSON。"""

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        super().__init__()
        self.ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        created = datetime.fromtimestamp(record.created, tz=UTC)
        payload: dict[str, Any] = {
            "timestamp": created.isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for name in _CONTEXT_VARS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value

        payload["source"] = f"{record.module}.{record.funcName}:{record.lineno}"

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_KEYS and key not in _CONTEXT_VARS
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info and record.exc_info[1] is not None:
            payload["error"] = self._describe_exception(record.exc_info[1])
            payload["traceback"] = self.formatException(record.exc_info)

        # default=str 兜底不可序列化的值：日志失败不应中断业务
        return json.dumps(payload, ensure_ascii=self.ensure_ascii, default=str)

    @staticmethod
    def _describe_exception(exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, AgentError):
            described = exc.to_dict()
            described["type"] = type(exc).__name__
            return described
        return {"type": type(exc).__name__, "message": str(exc)}


# ============================================================ 初始化


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.strip().upper())
    if resolved is None:
        raise ConfigError(
            f"未知的日志级别：{level!r}；可用级别为 {sorted(logging.getLevelNamesMapping())}"
        )
    return resolved


def _force_utf8(stream: TextIO) -> None:
    """Windows 控制台默认非 UTF-8，中文日志会因编码失败而丢失。

    `errors="backslashreplace"` 确保即便重配失败也不会抛异常打断日志输出。
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    with suppress(ValueError, OSError):
        reconfigure(encoding="utf-8", errors="backslashreplace")


def setup_logging(
    level: str | int = "INFO",
    *,
    stream: TextIO | None = None,
    json_format: bool = True,
) -> logging.Handler:
    """配置根日志器，返回本次安装的 handler 以便测试断言与手工卸载。

    可重复调用：再次调用会替换上一次安装的 handler，而不是叠加输出。
    配置根日志器而非 `agent` 专属日志器，是为了让第三方库的日志也进入同一 JSON 管道。
    """
    root = logging.getLogger()

    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_TAG, False):
            root.removeHandler(existing)
            existing.close()

    target = stream if stream is not None else sys.stderr
    if stream is None:
        _force_utf8(target)

    handler: logging.Handler = logging.StreamHandler(target)
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler.addFilter(ContextFilter())
    setattr(handler, _HANDLER_TAG, True)

    root.addHandler(handler)
    root.setLevel(_normalize_level(level))
    return handler


def get_logger(name: str) -> logging.Logger:
    """统一的日志器获取入口，调用方一律传 `__name__`。"""
    return logging.getLogger(name)
