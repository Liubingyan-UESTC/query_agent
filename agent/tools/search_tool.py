"""SearchTool：按关键字检索日志。

没有真实数据库，用一个 JSON 文件（``TOOL_DATA_FILE``，默认 ``data/logs.json``）代替
Kibana 索引。关键字在记录的**所有字段值**上做大小写不敏感的子串匹配——模型不必先知道
字段名就能查到东西，这正是"关键字检索"的语义。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agent.config import ToolSettings
from agent.errors import ToolError
from agent.tools.base import BaseTool, ToolContext, ToolResult

__all__ = ["SearchArgs", "SearchTool", "load_log_index"]


class SearchArgs(BaseModel):
    keyword: str = Field(
        description=(
            "检索关键字，在日志的所有字段上做大小写不敏感子串匹配。"
            "用空格分隔多个词表示**同时满足**，例如 'order-service ERROR' 表示"
            "既属于 order-service 又是 ERROR 级别。空字符串表示不过滤。"
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        description="最多返回多少条记录。",
    )


def load_log_index(path: Path) -> dict[str, Any]:
    """读取假数据库文件。文件缺失或格式不对都是配置问题，直接抛不可重试的错误。"""
    if not path.exists():
        raise ToolError(f"找不到数据文件：{path}（检查 TOOL_DATA_FILE 配置）")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"数据文件不是合法 JSON：{path}", detail=str(exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ToolError(f"数据文件缺少 records 数组：{path}")
    return payload


def _matches(record: dict[str, Any], terms: list[str]) -> bool:
    """每个词都要在**某个**字段里出现（词之间是 AND，字段之间是 OR）。

    整串当一个子串匹配是行不通的：用户最常问的就是「order-service 的 ERROR 日志」，
    而 "order-service ERROR" 这个字符串不会完整出现在任何字段里，只会命中 0 条，
    把模型逼进「换个词再试」的死循环。
    """
    if not terms:
        return True
    haystack = [str(value).lower() for value in record.values()]
    return all(any(term in value for value in haystack) for term in terms)


class SearchTool(BaseTool):
    name: ClassVar[str] = "search_tool"
    description: ClassVar[str] = (
        "按关键字检索日志库。关键字会在 timestamp/level/service/host/message/"
        "status_code/latency_ms/user_id/trace_id 等所有字段上做大小写不敏感子串匹配。"
        "多个词用空格分隔表示同时满足：keyword='order-service ERROR' 查该服务的错误日志，"
        "keyword='ERROR' 查全部错误，keyword='timeout' 查超时。"
        "keyword 传空字符串则返回全部记录。"
    )
    args_schema: ClassVar[type[BaseModel]] = SearchArgs

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings
        # 索引是懒加载的，而工具实例被所有会话共享，所以首次加载必须加锁：不加锁时
        # N 个并发请求会各自读一遍盘、各自解析一遍 JSON，最后互相覆盖 _index。
        # 结果虽然等价（加载是幂等的），但把一次启动开销放大成了 N 次。
        self._lock = threading.Lock()
        self._index: dict[str, Any] | None = None

    def _records(self) -> tuple[str, list[dict[str, Any]]]:
        index = self._index
        if index is None:
            with self._lock:
                # 双重检查：等锁期间别的线程可能已经加载好了
                if self._index is None:
                    self._index = load_log_index(self._settings.resolved_data_file())
                index = self._index
        # 返回拷贝，调用方的过滤/截断不会碰到共享的那一份
        return str(index.get("index", "logs")), list(index["records"])

    def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:  # noqa: ARG002 - 检索无需上下文
        index, records = self._records()
        terms = [term.lower() for term in args.keyword.split()]
        hits = [record for record in records if _matches(record, terms)]

        limit = min(args.limit, self._settings.max_rows)
        returned = hits[:limit]
        truncated = len(hits) > len(returned)

        keyword_text = args.keyword or "(全部)"
        summary = f"关键字 {keyword_text} 命中 {len(hits)} 条，返回 {len(returned)} 条"
        if truncated:
            summary += "（已截断）"
        return ToolResult.success(
            {
                "index": index,
                "keyword": args.keyword,
                "matched_terms": terms,
                "total": len(hits),
                "returned": len(returned),
                "truncated": truncated,
                "records": returned,
            },
            summary,
        )
