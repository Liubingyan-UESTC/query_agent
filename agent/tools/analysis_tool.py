"""AnalysisTool：对黑板上已有的检索结果做分组统计。

它不重新查库，而是按 ``tool_call_id`` 取回上一步 SearchTool 的全量结果——这正是
tool_result 黑板存在的意义：大块数据只存一份，后续步骤靠 id 引用。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolContext, ToolResult

__all__ = ["AnalysisArgs", "AnalysisTool"]

Metric = Literal["count", "sum", "avg", "max", "min"]


class AnalysisArgs(BaseModel):
    tool_call_id: str = Field(
        description="要分析的数据来自哪次工具调用，填该次调用返回的 tool_call_id。",
    )
    group_by: str = Field(
        description="分组字段名，例如 service / level / host / status_code。",
    )
    metric: Metric = Field(
        default="count",
        description="统计方式：count 计数；sum/avg/max/min 需要同时指定数值字段 field。",
    )
    field: str | None = Field(
        default=None,
        description="数值字段名，例如 latency_ms；metric=count 时忽略。",
    )
    top: int = Field(default=10, ge=1, description="最多返回前几组（按统计值降序）。")


def _extract_records(payload: Any) -> list[dict[str, Any]] | None:
    """从黑板取回的结果里找出记录数组，兼容"直接是数组"与"包在 records 里"两种形态。"""
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return None


def _numbers(values: Iterable[Any]) -> list[float] | None:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numbers.append(float(value))
    return numbers


def _aggregate(values: list[float], metric: Metric) -> float:
    if metric == "sum":
        return sum(values)
    if metric == "avg":
        return round(sum(values) / len(values), 2)
    if metric == "max":
        return max(values)
    return min(values)


class AnalysisTool(BaseTool):
    name: ClassVar[str] = "analysis_tool"
    description: ClassVar[str] = (
        "对之前某次检索的结果做分组统计。必须先用 search_tool 取数，再把它返回的 "
        "tool_call_id 传进来。支持 count 计数与 sum/avg/max/min 数值聚合。"
    )
    args_schema: ClassVar[type[BaseModel]] = AnalysisArgs

    def run(self, args: AnalysisArgs, ctx: ToolContext) -> ToolResult:
        payload = ctx.result_of(args.tool_call_id)
        if payload is None:
            return ToolResult.failure(
                f"黑板上没有 tool_call_id={args.tool_call_id} 的结果，"
                f"请先调用 search_tool 并使用它返回的 tool_call_id。"
            )

        records = _extract_records(payload)
        if not records:
            return ToolResult.failure(
                f"tool_call_id={args.tool_call_id} 的结果里没有可分析的记录。"
            )

        available = sorted({key for record in records for key in record})
        if args.group_by not in available:
            return ToolResult.failure(f"字段 {args.group_by!r} 不存在，可用字段：{available}")
        if args.metric != "count":
            if args.field is None:
                return ToolResult.failure(f"metric={args.metric} 需要同时指定数值字段 field。")
            if args.field not in available:
                return ToolResult.failure(f"字段 {args.field!r} 不存在，可用字段：{available}")

        buckets: dict[str, list[Any]] = {}
        for record in records:
            key = str(record.get(args.group_by))
            buckets.setdefault(key, []).append(record.get(args.field) if args.field else None)

        groups: list[dict[str, Any]] = []
        for key, values in buckets.items():
            if args.metric == "count":
                groups.append({"key": key, "value": len(values)})
                continue
            numbers = _numbers(values)
            if numbers is None:
                return ToolResult.failure(
                    f"字段 {args.field!r} 在分组 {key!r} 中包含非数值，无法做 {args.metric} 聚合。"
                )
            groups.append({"key": key, "value": _aggregate(numbers, args.metric)})

        groups.sort(key=lambda item: (-float(item["value"]), item["key"]))
        groups = groups[: args.top]

        metric_text = args.metric if args.metric == "count" else f"{args.metric}({args.field})"
        detail = "，".join(f"{item['key']}={item['value']}" for item in groups)
        return ToolResult.success(
            {
                "source_tool_call_id": args.tool_call_id,
                "group_by": args.group_by,
                "metric": args.metric,
                "field": args.field,
                "analyzed_records": len(records),
                "groups": groups,
            },
            f"按 {args.group_by} 统计 {metric_text}（共 {len(records)} 条）：{detail}",
        )
