"""线程安全的内存 Store，TTL 惰性过期。

每个公开方法持 `RLock`：保证单次操作原子，并发 `push` 不会丢元素。
`get+set` 仍是两次操作，不提供事务——那是 Redis 实现的事。

值在写入与读出时都 `deepcopy`，避免调用方就地修改污染存储，也避免
`get_messages` 那种「交出内部列表」的泄漏再次发生。
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from agent.common.errors import StoreError
from agent.store.base import Store

__all__ = ["MemoryStore"]

_Kind = Literal["kv", "list"]


@dataclass
class _Entry:
    kind: _Kind
    value: Any
    expires_at: float | None = None


class MemoryStore(Store):
    """进程内存储。`clock` 可注入，供 TTL 单测推进时间而不真实 sleep。"""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = RLock()
        self._entries: dict[str, _Entry] = {}

    # ============================================================ KV

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._live(key, "kv")
            if entry is None:
                return None
            return copy.deepcopy(entry.value)

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        with self._lock:
            self._reject_wrong_kind(key, "kv")
            if ttl is not None and ttl <= 0:
                self._entries.pop(key, None)
                return
            self._entries[key] = _Entry(
                kind="kv",
                value=copy.deepcopy(value),
                expires_at=None if ttl is None else self._clock() + ttl,
            )

    def delete(self, key: str) -> bool:
        with self._lock:
            if self._live(key, expected=None) is None:
                self._entries.pop(key, None)
                return False
            del self._entries[key]
            return True

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._live(key, expected=None) is not None

    def keys(self, prefix: str) -> list[str]:
        with self._lock:
            found: list[str] = []
            expired: list[str] = []
            now = self._clock()
            for key, entry in self._entries.items():
                if not key.startswith(prefix):
                    continue
                if entry.expires_at is not None and entry.expires_at <= now:
                    expired.append(key)
                    continue
                found.append(key)
            for key in expired:
                del self._entries[key]
            return found

    def expire(self, key: str, ttl: float) -> bool:
        with self._lock:
            entry = self._live(key, expected=None)
            if entry is None:
                return False
            if ttl <= 0:
                del self._entries[key]
                return True
            entry.expires_at = self._clock() + ttl
            return True

    # ============================================================ List

    def push(self, key: str, *values: Any) -> int:
        if not values:
            raise ValueError("push 至少需要一个值")
        with self._lock:
            self._reject_wrong_kind(key, "list")
            entry = self._live(key, "list")
            if entry is None:
                entry = _Entry(kind="list", value=[])
                self._entries[key] = entry
            entry.value.extend(copy.deepcopy(v) for v in values)
            return len(entry.value)

    def range(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        with self._lock:
            entry = self._live(key, "list")
            if entry is None:
                return []
            return copy.deepcopy(_closed_slice(entry.value, start, end))

    def trim(self, key: str, start: int, end: int) -> None:
        with self._lock:
            entry = self._live(key, "list")
            if entry is None:
                return
            entry.value = _closed_slice(entry.value, start, end)

    def length(self, key: str) -> int:
        with self._lock:
            entry = self._live(key, "list")
            return 0 if entry is None else len(entry.value)

    # ============================================================ 内部

    def _live(self, key: str, expected: _Kind | None) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= self._clock():
            del self._entries[key]
            return None
        if expected is not None and entry.kind != expected:
            raise StoreError(f"键 {key!r} 的类型是 {entry.kind}，不能按 {expected} 访问")
        return entry

    def _reject_wrong_kind(self, key: str, expected: _Kind) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.expires_at is not None and entry.expires_at <= self._clock():
            del self._entries[key]
            return
        if entry.kind != expected:
            raise StoreError(f"键 {key!r} 的类型是 {entry.kind}，不能按 {expected} 访问")


def _closed_slice(values: list[Any], start: int, end: int) -> list[Any]:
    """对齐 Redis LRANGE / LTRIM：闭区间，支持负下标。"""
    length = len(values)
    if length == 0:
        return []
    start_i = start if start >= 0 else length + start
    end_i = end if end >= 0 else length + end
    start_i = max(0, start_i)
    end_i = min(length - 1, end_i)
    if start_i > end_i:
        return []
    return values[start_i : end_i + 1]
