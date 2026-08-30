"""存储抽象：KV 与 List 两套接口。

业务代码只依赖本模块，不感知内存还是 Redis。键名规范见 `make_key`。
方法语义对齐 Redis：`range` / `trim` 的下标含负数、`end` 闭区间，
以便步骤 27 的 Redis 实现直接复用契约测试。
"""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["KVStore", "ListStore", "Store"]


class KVStore(ABC):
    """字符串键到任意 JSON 友好值的映射。"""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """键不存在或已过期时返回 `None`。返回值是拷贝，调用方就地修改不影响存储。"""

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """写入。`ttl` 为秒；`None` 表示永不过期；`<= 0` 视为立即过期（不保留）。"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除。键存在且未过期返回 `True`。"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """过期键视为不存在。"""

    @abstractmethod
    def keys(self, prefix: str) -> list[str]:
        """返回未过期且以 `prefix` 开头的键。调用方应带上末尾冒号以免 `s1` 命中 `s10`。"""

    @abstractmethod
    def expire(self, key: str, ttl: float) -> bool:
        """给已有键设 TTL。键不存在或已过期返回 `False`。`ttl <= 0` 立即删除。"""


class ListStore(ABC):
    """列表键，语义对齐 Redis List（右推、闭区间下标）。"""

    @abstractmethod
    def push(self, key: str, *values: Any) -> int:
        """追加到右侧，返回新长度。至少要有一个值。"""

    @abstractmethod
    def range(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        """闭区间切片，支持负下标。键不存在返回空列表。返回值是拷贝。"""

    @abstractmethod
    def trim(self, key: str, start: int, end: int) -> None:
        """就地保留闭区间。规范化后 start > end 则清空。"""

    @abstractmethod
    def length(self, key: str) -> int:
        """键不存在或已过期返回 0。"""


class Store(KVStore, ListStore, ABC):
    """同时提供 KV 与 List 的后端。内存与 Redis 都实现这一面。"""
