"""步骤 18：FakeESClient 验证 DSL、非法字段拦截、max_result_rows 截断。"""

from __future__ import annotations

from agent.common.enums import ArtifactType
from agent.config.settings import ToolSettings
from agent.tool_manage.args import QueryFilter, SearchAggregation, TimeRange
from agent.tool_manage.base import ToolContext
from agent.tool_manage.es_client import FakeESClient, StaticFieldCatalog
from agent.tool_manage.tools.search_tool import SearchArgs, SearchTool, build_search_dsl

CATALOG = StaticFieldCatalog(
    {
        "logs-app": {"@timestamp", "service", "level", "message", "status"},
        "metrics-host": {"@timestamp", "host", "cpu_pct"},
    }
)

ROWS = [
    {"@timestamp": "2026-08-30T10:00:00Z", "service": "query-agent", "level": "ERROR"},
    {"@timestamp": "2026-08-30T10:01:00Z", "service": "billing", "level": "INFO"},
    {"@timestamp": "2026-08-30T10:02:00Z", "service": "query-agent", "level": "ERROR"},
]


def _tool(settings: ToolSettings | None = None) -> tuple[SearchTool, FakeESClient]:
    client = FakeESClient({"logs-app": ROWS})
    return SearchTool(client, CATALOG, settings), client


def _ctx() -> ToolContext:
    return ToolContext(task_id="task_1", session_id="sess_1", trace_id="trace_1")


def test_dsl_includes_time_range_filters_and_aggs() -> None:
    args = SearchArgs(
        index="logs-app",
        time_range=TimeRange(start="now-1d", end="now"),
        filters=[QueryFilter(field="level", op="eq", value="ERROR")],
        aggregations=[SearchAggregation(name="by_svc", kind="terms", field="service")],
        fields=["service", "level"],
        limit=20,
    )
    dsl = build_search_dsl(args)

    assert dsl["size"] == 20
    assert dsl["_source"] == ["service", "level"]
    assert {"range": {"@timestamp": {"gte": "now-1d", "lte": "now"}}} in dsl["query"]["bool"][
        "must"
    ]
    assert {"term": {"level": "ERROR"}} in dsl["query"]["bool"]["must"]
    assert dsl["aggs"]["by_svc"] == {"terms": {"field": "service"}}
    assert "query_string" not in str(dsl)


def test_search_returns_table_and_records_dsl() -> None:
    tool, client = _tool()
    result = tool.run(SearchArgs(index="logs-app", limit=2), _ctx())

    assert result.ok is True
    assert result.artifact is not None
    assert result.artifact.artifact_type is ArtifactType.TABLE
    assert result.artifact.row_count == 3
    assert len(result.artifact.data) == 2
    assert client.calls[0]["index"] == "logs-app"
    assert client.calls[0]["body"]["size"] == 2


def test_illegal_field_is_rejected() -> None:
    tool, _client = _tool()
    result = tool.run(
        SearchArgs(index="logs-app", filters=[QueryFilter(field="unknown_col", value="x")]),
        _ctx(),
    )

    assert result.ok is False
    assert result.error is not None
    assert "unknown_col" in result.text


def test_unknown_index_is_rejected() -> None:
    tool, _client = _tool()
    result = tool.run(SearchArgs(index="no-such"), _ctx())

    assert result.ok is False
    assert "没有索引" in result.text


def test_max_result_rows_truncates() -> None:
    settings = ToolSettings(max_result_rows=1)
    tool, _client = _tool(settings)
    result = tool.run(SearchArgs(index="logs-app", limit=100), _ctx())

    assert result.ok is True
    assert result.artifact is not None
    assert len(result.artifact.data) == 1
    assert result.artifact.row_count == 3
    assert result.artifact.meta["truncated"] is True


def test_filter_and_agg_clause_variants() -> None:
    args = SearchArgs(
        index="logs-app",
        filters=[
            QueryFilter(field="level", op="neq", value="INFO"),
            QueryFilter(field="message", op="contains", value="timeout"),
            QueryFilter(field="service", op="in", value=["query-agent"]),
            QueryFilter(field="status", op="gte", value=500),
        ],
        aggregations=[
            SearchAggregation(name="cnt", kind="count", field="service"),
            SearchAggregation(name="avg_status", kind="avg", field="status"),
            SearchAggregation(
                name="by_day", kind="date_histogram", field="@timestamp", interval="1h"
            ),
        ],
        limit=0,
    )
    dsl = build_search_dsl(args)
    assert dsl["size"] == 0
    assert {"bool": {"must_not": [{"term": {"level": "INFO"}}]}} in dsl["query"]["bool"]["must"]
    assert {"match": {"message": "timeout"}} in dsl["query"]["bool"]["must"]
    assert {"terms": {"service": ["query-agent"]}} in dsl["query"]["bool"]["must"]
    assert {"range": {"status": {"gte": 500}}} in dsl["query"]["bool"]["must"]
    assert dsl["aggs"]["cnt"] == {"value_count": {"field": "service"}}
    assert dsl["aggs"]["avg_status"] == {"avg": {"field": "status"}}
    assert dsl["aggs"]["by_day"]["date_histogram"]["calendar_interval"] == "1h"


def test_terms_without_field_fails() -> None:
    tool, _client = _tool()
    result = tool.run(
        SearchArgs(
            index="logs-app",
            aggregations=[SearchAggregation(name="x", kind="terms")],
        ),
        _ctx(),
    )
    assert result.ok is False
    assert "必须指定 field" in result.text


def test_hits_payload_without_hits_block() -> None:
    client = FakeESClient({"logs-app": ROWS})
    client.search = lambda **kwargs: {"took": 1}  # type: ignore[method-assign]
    tool = SearchTool(client, CATALOG)
    result = tool.run(SearchArgs(index="logs-app"), _ctx())
    assert result.ok is True
    assert result.artifact is not None
    assert result.artifact.data == []
    assert result.artifact.row_count == 0
