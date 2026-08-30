"""步骤 18：AnalysisTool 对空结果集与类型异常给出明确错误。"""

from __future__ import annotations

from agent.common.enums import ArtifactType
from agent.models.artifact import Artifact
from agent.tool_manage.args import AnalysisAgg, QueryFilter
from agent.tool_manage.base import ToolContext
from agent.tool_manage.tools.analysis_tool import AnalysisArgs, AnalysisTool


def _ctx(artifact: Artifact) -> ToolContext:
    return ToolContext(
        task_id="task_1",
        session_id="sess_1",
        trace_id="trace_1",
        artifacts_reader=lambda aid: (
            artifact if aid == artifact.artifact_id else (_ for _ in ()).throw(KeyError(aid))
        ),
    )


def _table(rows: list[dict[str, object]], artifact_id: str = "art_1") -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        data=rows,
    )


def test_groupby_count() -> None:
    source = _table(
        [
            {"service": "a", "n": 1},
            {"service": "a", "n": 3},
            {"service": "b", "n": 2},
        ]
    )
    result = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            group_by=["service"],
            aggregations=[AnalysisAgg(field="n", func="sum")],
            sort_by="n_sum",
            sort_order="desc",
        ),
        _ctx(source),
    )

    assert result.ok is True
    assert result.artifact is not None
    assert result.artifact.data[0]["service"] == "a"
    assert result.artifact.data[0]["n_sum"] == 4
    assert result.artifact.meta["source_artifact_id"] == "art_1"


def test_empty_input_is_explicit_error() -> None:
    result = AnalysisTool().run(AnalysisArgs(artifact_id="art_1"), _ctx(_table([])))

    assert result.ok is False
    assert "没有行" in result.text


def test_type_error_on_non_numeric_sum() -> None:
    source = _table([{"service": "a"}, {"service": "b"}])
    result = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            aggregations=[AnalysisAgg(field="service", func="sum")],
        ),
        _ctx(source),
    )

    assert result.ok is False
    assert "不是数值" in result.text


def test_missing_group_field() -> None:
    result = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            group_by=["nope"],
            aggregations=[AnalysisAgg(field="n", func="count")],
        ),
        _ctx(_table([{"n": 1}])),
    )

    assert result.ok is False
    assert "分组字段" in result.text


def test_filter_then_empty() -> None:
    result = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            filters=[QueryFilter(field="service", op="eq", value="missing")],
        ),
        _ctx(_table([{"service": "a"}])),
    )

    assert result.ok is False
    assert "过滤后结果为空" in result.text


def test_global_aggregation_and_filters() -> None:
    source = _table([{"service": "a", "n": 2}, {"service": "b", "n": 4}, {"service": "a", "n": 1}])
    result = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            filters=[QueryFilter(field="service", op="neq", value="b")],
            aggregations=[
                AnalysisAgg(field="n", func="mean"),
                AnalysisAgg(field="n", func="min"),
                AnalysisAgg(field="n", func="max"),
                AnalysisAgg(field="n", func="count"),
            ],
        ),
        _ctx(source),
    )
    assert result.ok is True
    assert result.artifact is not None
    row = result.artifact.data[0]
    assert row["n_count"] == 2
    assert row["n_min"] == 1
    assert row["n_max"] == 2


def test_contains_and_in_filters() -> None:
    source = _table(
        [
            {"service": "query-agent", "n": 1},
            {"service": "billing", "n": 2},
        ]
    )
    result = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            filters=[QueryFilter(field="service", op="contains", value="query")],
        ),
        _ctx(source),
    )
    assert result.ok is True
    assert result.artifact is not None
    assert len(result.artifact.data) == 1

    result_in = AnalysisTool().run(
        AnalysisArgs(
            artifact_id="art_1",
            filters=[QueryFilter(field="service", op="in", value=["billing"])],
        ),
        _ctx(source),
    )
    assert result_in.ok is True
    assert result_in.artifact is not None
    assert result_in.artifact.data[0]["service"] == "billing"


def test_non_list_data_and_missing_artifact() -> None:
    bad = Artifact(
        artifact_id="art_1",
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.SCALAR,
        data=1,
    )
    result = AnalysisTool().run(AnalysisArgs(artifact_id="art_1"), _ctx(bad))
    assert result.ok is False
    assert "表格" in result.text

    missing = AnalysisTool().run(
        AnalysisArgs(artifact_id="art_missing"),
        ToolContext(
            task_id="task_1",
            session_id="sess_1",
            trace_id="trace_1",
            artifacts_reader=lambda aid: (_ for _ in ()).throw(KeyError(aid)),
        ),
    )
    assert missing.ok is False


def test_missing_sort_field() -> None:
    result = AnalysisTool().run(
        AnalysisArgs(artifact_id="art_1", sort_by="nope"),
        _ctx(_table([{"n": 1}])),
    )
    assert result.ok is False
    assert "排序字段" in result.text
