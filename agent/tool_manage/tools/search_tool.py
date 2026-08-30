"""SearchTool：结构化条件 → ES DSL → TABLE Artifact。"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from agent.common.enums import ArtifactType
from agent.common.errors import ToolError, ToolInvocationError
from agent.common.ids import new_artifact_id
from agent.config.settings import ToolSettings
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.tool_manage.args import QueryFilter, SearchAggregation, TimeRange
from agent.tool_manage.base import BaseTool, FieldCatalog, ToolContext, ToolDeps, ToolResult
from agent.tool_manage.es_client import ESClient
from agent.tool_manage.registry import register_tool

__all__ = ["SearchArgs", "SearchTool"]

_TIME_FIELD = "@timestamp"


class SearchArgs(AgentModel):
    index: str = Field(min_length=1)
    time_range: TimeRange | None = None
    filters: list[QueryFilter] = Field(default_factory=list)
    aggregations: list[SearchAggregation] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    limit: int = Field(default=100, ge=0)


@register_tool
class SearchTool(BaseTool):
    name = "search"
    description = "按索引、时间范围、过滤与聚合从 Kibana/ES 取数，返回表格产物。"
    args_schema = SearchArgs
    produces = ArtifactType.TABLE
    is_mock = False

    def __init__(
        self,
        es_client: ESClient,
        field_catalog: FieldCatalog,
        settings: ToolSettings | None = None,
    ) -> None:
        self._es = es_client
        self._catalog = field_catalog
        self._settings = settings if settings is not None else ToolSettings.from_env(None)

    @classmethod
    def from_deps(cls, deps: ToolDeps) -> SearchTool:
        if deps.es_client is None:
            raise ToolError("SearchTool 需要显式注入 es_client，非 mock 不得回落 FakeESClient")
        if deps.field_catalog is None:
            raise ToolError("SearchTool 需要显式注入 field_catalog")
        return cls(deps.es_client, deps.field_catalog, deps.settings)

    def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            _validate_search_fields(args, self._catalog)
            body = build_search_dsl(args)
            raw = self._es.search(index=args.index, body=body)
        except ToolError as exc:
            return ToolResult.fail(exc)
        rows, total = _hits_to_rows(raw)
        cap = self._settings.max_result_rows
        truncated = rows[: min(args.limit, cap)]
        schema = dict.fromkeys(truncated[0] if truncated else {}, "keyword")
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            task_id=ctx.task_id,
            producer=self.name,
            artifact_type=ArtifactType.TABLE,
            title=f"{args.index} 查询结果",
            data_schema=schema,
            data=truncated,
            row_count=total,
            meta={"truncated": total > len(truncated), "dsl": body},
        )
        return ToolResult.success(
            f"查询 {args.index} 得到 {total} 行，返回 {len(truncated)} 行",
            artifact,
        )


def build_search_dsl(args: SearchArgs) -> dict[str, Any]:
    must: list[dict[str, Any]] = []
    if args.time_range and (args.time_range.start or args.time_range.end):
        rng: dict[str, str] = {}
        if args.time_range.start:
            rng["gte"] = args.time_range.start
        if args.time_range.end:
            rng["lte"] = args.time_range.end
        must.append({"range": {_TIME_FIELD: rng}})
    for item in args.filters:
        must.append(_filter_clause(item))
    query: dict[str, Any] = {"bool": {"must": must}} if must else {"match_all": {}}
    size = 0 if args.aggregations and args.limit == 0 else args.limit
    body: dict[str, Any] = {"size": size, "query": query}
    if args.fields:
        body["_source"] = args.fields
    if args.aggregations:
        body["aggs"] = {item.name: _agg_clause(item) for item in args.aggregations}
    return body


def _filter_clause(item: QueryFilter) -> dict[str, Any]:
    if item.op == "eq":
        return {"term": {item.field: item.value}}
    if item.op == "neq":
        return {"bool": {"must_not": [{"term": {item.field: item.value}}]}}
    if item.op == "contains":
        return {"match": {item.field: item.value}}
    if item.op == "in":
        return {"terms": {item.field: item.value}}
    mapping = {"gt": "gt", "gte": "gte", "lt": "lt", "lte": "lte"}
    return {"range": {item.field: {mapping[item.op]: item.value}}}


def _agg_clause(item: SearchAggregation) -> dict[str, Any]:
    if item.kind == "count":
        return {"value_count": {"field": item.field or "_index"}}
    if item.kind == "terms":
        if not item.field:
            raise ToolInvocationError("terms 聚合必须指定 field")
        return {"terms": {"field": item.field}}
    if item.kind == "date_histogram":
        if not item.field:
            raise ToolInvocationError("date_histogram 必须指定 field")
        return {"date_histogram": {"field": item.field, "calendar_interval": item.interval or "1d"}}
    if not item.field:
        raise ToolInvocationError(f"{item.kind} 聚合必须指定 field")
    return {item.kind: {"field": item.field}}


def _validate_search_fields(args: SearchArgs, catalog: FieldCatalog) -> None:
    if not catalog.has_index(args.index):
        raise ToolInvocationError(f"字段字典没有索引 {args.index!r}")
    known = set(catalog.field_names(args.index))
    used: list[str] = []
    if args.time_range and (args.time_range.start or args.time_range.end):
        used.append(_TIME_FIELD)
    used.extend(item.field for item in args.filters)
    used.extend(item.field for item in args.aggregations if item.field)
    used.extend(args.fields)
    illegal = sorted({name for name in used if name not in known})
    if illegal:
        raise ToolInvocationError(
            f"索引 {args.index} 不允许字段 {illegal}；合法字段为 {sorted(known)}"
        )


def _hits_to_rows(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    hits_block = raw.get("hits")
    if not isinstance(hits_block, dict):
        return [], 0
    total_raw = hits_block.get("total", 0)
    total = int(total_raw.get("value", 0)) if isinstance(total_raw, dict) else int(total_raw or 0)
    rows: list[dict[str, Any]] = []
    for hit in hits_block.get("hits") or []:
        if isinstance(hit, dict) and isinstance(hit.get("_source"), dict):
            rows.append(hit["_source"])
    return rows, total
