"""ES 客户端协议与本地 Fake。真实连接在步骤 35 再接。"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Protocol

from agent.common.errors import ToolInvocationError

__all__ = ["ESClient", "FakeESClient", "StaticFieldCatalog"]


class ESClient(Protocol):
    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]: ...


class StaticFieldCatalog:
    """测试与无 KnowledgeMemory 场景用的静态字段目录。"""

    def __init__(self, indexes: Mapping[str, Collection[str]]) -> None:
        self._indexes = {name: frozenset(fields) for name, fields in indexes.items()}

    def has_index(self, name: str) -> bool:
        return name in self._indexes

    def field_names(self, index: str) -> frozenset[str]:
        try:
            return self._indexes[index]
        except KeyError:
            known = sorted(self._indexes)
            raise ToolInvocationError(f"未知索引 {index!r}；已有：{known}") from None


class FakeESClient:
    """按 index 回放预先放入的行。记录每次 DSL，供单测断言拼装。"""

    def __init__(self, fixtures: Mapping[str, list[dict[str, Any]]] | None = None) -> None:
        self.fixtures: dict[str, list[dict[str, Any]]] = {
            name: list(rows) for name, rows in (fixtures or {}).items()
        }
        self.calls: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"index": index, "body": body})
        rows = list(self.fixtures.get(index, []))
        size = body.get("size")
        limit = int(size) if isinstance(size, int) else len(rows)
        hits = rows[: max(limit, 0)]
        return {
            "hits": {
                "total": {"value": len(rows)},
                "hits": [{"_id": str(i), "_source": row} for i, row in enumerate(hits)],
            },
            "aggregations": {},
        }
