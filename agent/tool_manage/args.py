"""工具入参的共享形状。Search / Analysis 都要过滤条件，避免各写一套 op。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agent.models.base import AgentModel

__all__ = ["AnalysisAgg", "QueryFilter", "SearchAggregation", "TimeRange"]


class TimeRange(AgentModel):
    start: str | None = None
    end: str | None = None


class QueryFilter(AgentModel):
    field: str = Field(min_length=1)
    op: Literal["eq", "neq", "gt", "gte", "lt", "lte", "contains", "in"] = "eq"
    value: Any = None


class SearchAggregation(AgentModel):
    name: str = Field(min_length=1)
    kind: Literal["terms", "date_histogram", "avg", "sum", "min", "max", "count"]
    field: str | None = None
    interval: str | None = None


class AnalysisAgg(AgentModel):
    field: str = Field(min_length=1)
    func: Literal["count", "sum", "mean", "min", "max"]
