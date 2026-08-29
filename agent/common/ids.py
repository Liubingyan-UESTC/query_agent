"""标识符生成。

格式为 `<前缀>_<UTC 时间戳>_<短随机>`，例如 `task_20260829T230501123_9f3a1c`。
三段式的目的是排查友好：前缀让日志里一眼看出实体类型，毫秒级 UTC 时间戳使 ID
按字典序即按生成时序排列，短随机段消除同毫秒内的碰撞。

时间戳统一用 UTC，避免多时区部署下 ID 的排序语义被破坏。
"""

import re
import secrets
from datetime import UTC, datetime

__all__ = [
    "ARTIFACT_ID_PREFIX",
    "ID_PATTERN",
    "SESSION_ID_PREFIX",
    "TASK_ID_PREFIX",
    "TRACE_ID_PREFIX",
    "new_artifact_id",
    "new_id",
    "new_session_id",
    "new_task_id",
    "new_trace_id",
]

TASK_ID_PREFIX = "task"
SESSION_ID_PREFIX = "sess"
ARTIFACT_ID_PREFIX = "art"
TRACE_ID_PREFIX = "trace"

# 随机段字节数；3 字节 = 6 个十六进制字符，配合毫秒时间戳足以避免碰撞
_RANDOM_BYTES = 3

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"

ID_PATTERN = re.compile(r"^(?P<prefix>[a-z]+)_(?P<timestamp>\d{8}T\d{9})_(?P<random>[0-9a-f]{6})$")


def new_id(prefix: str) -> str:
    """生成带指定前缀的标识符。"""
    now = datetime.now(UTC)
    # strftime 无毫秒占位符，故由 microsecond 手工截断到毫秒
    timestamp = f"{now.strftime(_TIMESTAMP_FORMAT)}{now.microsecond // 1000:03d}"
    return f"{prefix}_{timestamp}_{secrets.token_hex(_RANDOM_BYTES)}"


def new_task_id() -> str:
    return new_id(TASK_ID_PREFIX)


def new_session_id() -> str:
    return new_id(SESSION_ID_PREFIX)


def new_artifact_id() -> str:
    return new_id(ARTIFACT_ID_PREFIX)


def new_trace_id() -> str:
    return new_id(TRACE_ID_PREFIX)
