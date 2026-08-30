"""AnalysisTool：对已有 Artifact 做聚合 / 排序 / 过滤 / 统计，不回查 ES。"""

from __future__ import annotations

import json
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from agent.common.enums import ArtifactType
from agent.common.errors import ToolError, ToolInvocationError
from agent.common.ids import new_artifact_id
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.tool_manage.args import AnalysisAgg, QueryFilter
from agent.tool_manage.base import BaseTool, ToolContext, ToolResult
from agent.tool_manage.registry import register_tool

__all__ = ["AnalysisArgs", "AnalysisTool"]


class AnalysisArgs(AgentModel):
    artifact_id: str = Field(min_length=1)
    group_by: list[str] = Field(default_factory=list)
    aggregations: list[AnalysisAgg] = Field(default_factory=list)
    sort_by: str | None = None
    sort_order: Literal["asc", "desc"] = "asc"
    filters: list[QueryFilter] = Field(default_factory=list)
    limit: int | None = Field(default=None, gt=0)


@register_tool
class AnalysisTool(BaseTool):
    name = "analysis"
    description = "对已有表格产物做过滤、分组聚合与排序，不重复查询数据源。"
    args_schema = AnalysisArgs
    produces = ArtifactType.TABLE
    is_mock = False

    def run(self, args: AnalysisArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            source = ctx.get_artifact(args.artifact_id)
            frame = _to_frame(source)
        except ToolError as exc:
            return ToolResult.fail(exc)
        if frame.empty:
            return ToolResult.fail(
                ToolInvocationError("分析结果为空：输入产物没有行", retryable=False)
            )
        try:
            frame = _apply_filters(frame, args.filters)
            if frame.empty:
                return ToolResult.fail(ToolInvocationError("过滤后结果为空", retryable=False))
            frame = _aggregate(frame, args.group_by, args.aggregations)
            frame = _sort(frame, args.sort_by, args.sort_order)
            if args.limit is not None:
                frame = frame.head(args.limit)
        except (TypeError, ValueError, KeyError) as exc:
            return ToolResult.fail(
                ToolInvocationError(f"分析类型或字段错误：{exc}", retryable=False, detail=repr(exc))
            )

        records = json.loads(frame.to_json(orient="records", date_format="iso"))
        schema = {str(col): str(dtype) for col, dtype in frame.dtypes.items()}
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            task_id=ctx.task_id,
            producer=self.name,
            artifact_type=ArtifactType.TABLE,
            title=f"分析 {args.artifact_id}",
            data_schema=schema,
            data=records,
            row_count=len(records),
            meta={"source_artifact_id": args.artifact_id},
        )
        return ToolResult.success(f"分析完成，得到 {len(records)} 行", artifact)


def _to_frame(artifact: Artifact) -> pd.DataFrame:
    if not isinstance(artifact.data, list):
        raise ToolInvocationError("分析只接受表格类 list 数据")
    return pd.DataFrame(artifact.data)


def _apply_filters(frame: pd.DataFrame, filters: list[QueryFilter]) -> pd.DataFrame:
    result = frame
    for item in filters:
        if item.field not in result.columns:
            raise KeyError(f"过滤字段不存在：{item.field}")
        series = result[item.field]
        if item.op == "eq":
            result = result[series == item.value]
        elif item.op == "neq":
            result = result[series != item.value]
        elif item.op == "gt":
            result = result[series > item.value]
        elif item.op == "gte":
            result = result[series >= item.value]
        elif item.op == "lt":
            result = result[series < item.value]
        elif item.op == "lte":
            result = result[series <= item.value]
        elif item.op == "contains":
            result = result[series.astype(str).str.contains(str(item.value), na=False)]
        elif item.op == "in":
            result = result[series.isin(item.value)]
    return result


def _aggregate(
    frame: pd.DataFrame,
    group_by: list[str],
    aggregations: list[AnalysisAgg],
) -> pd.DataFrame:
    if not aggregations:
        return frame
    missing = [name for name in group_by if name not in frame.columns]
    if missing:
        raise KeyError(f"分组字段不存在：{missing}")
    named: dict[str, tuple[str, str]] = {}
    for item in aggregations:
        _pandas_func(item.func, item.field, frame)
        named[f"{item.field}_{item.func}"] = (item.field, item.func)
    if group_by:
        return frame.groupby(group_by, dropna=False).agg(**named).reset_index()
    payload: dict[str, Any] = {}
    for alias, (field, func) in named.items():
        series = frame[field]
        if func == "count":
            payload[alias] = int(series.count())
        elif func == "sum":
            payload[alias] = series.sum()
        elif func == "mean":
            payload[alias] = series.mean()
        elif func == "min":
            payload[alias] = series.min()
        else:
            payload[alias] = series.max()
    return pd.DataFrame([payload])


def _pandas_func(func: str, field: str, frame: pd.DataFrame) -> Any:
    if field not in frame.columns and func != "count":
        raise KeyError(f"聚合字段不存在：{field}")
    if func == "count":
        return "count"
    if func == "mean":
        return "mean"
    if func in {"sum", "min", "max"}:
        if field in frame.columns and not _is_numeric(frame[field]):
            raise TypeError(f"字段 {field} 不是数值，不能做 {func}")
        return func
    raise ValueError(f"不支持的聚合：{func}")


def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series))


def _sort(frame: pd.DataFrame, sort_by: str | None, order: str) -> pd.DataFrame:
    if not sort_by:
        return frame
    if sort_by not in frame.columns:
        raise KeyError(f"排序字段不存在：{sort_by}")
    if order not in {"asc", "desc"}:
        raise ValueError(f"sort_order 只能是 asc/desc，实际为 {order!r}")
    return frame.sort_values(sort_by, ascending=order == "asc")
