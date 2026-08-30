"""步骤 17：Mock 工具读取 fixture 并产出真实 Artifact。"""

from __future__ import annotations

from pathlib import Path

from agent.common.enums import ArtifactType
from agent.models.artifact import Artifact
from agent.tool_manage.base import ToolContext
from agent.tool_manage.manager import ToolManager
from agent.tool_manage.tools.export_tool import ExportArgs
from agent.tool_manage.tools.mock_export_tool import MockExportTool


def _ctx(artifacts: dict[str, Artifact] | None = None) -> ToolContext:
    store = artifacts or {}
    return ToolContext(
        task_id="task_1",
        session_id="sess_1",
        trace_id="trace_1",
        artifacts_reader=store.__getitem__,
    )


def test_mock_search_reads_fixture() -> None:
    manager = ToolManager(use_mock=True)
    result = manager.invoke("search", {"index": "logs-app", "limit": 2}, _ctx())

    assert result.ok is True
    assert result.artifact is not None
    assert result.artifact.artifact_type is ArtifactType.TABLE
    assert result.artifact.producer == "search"
    assert len(result.artifact.data) == 2
    assert result.artifact.row_count == 3
    assert result.artifact.data[0]["service"] == "query-agent"


def test_mock_analysis_groups_existing_artifact() -> None:
    manager = ToolManager(use_mock=True)
    source = Artifact(
        artifact_id="art_src",
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        data=[
            {"service": "query-agent"},
            {"service": "query-agent"},
            {"service": "billing"},
        ],
    )
    result = manager.invoke(
        "analysis",
        {"artifact_id": "art_src", "group_by": "service"},
        _ctx({"art_src": source}),
    )

    assert result.ok is True
    assert result.artifact is not None
    rows = {row["service"]: row["count"] for row in result.artifact.data}
    assert rows == {"query-agent": 2, "billing": 1}


def test_mock_export_writes_csv(tmp_path) -> None:
    source = Artifact(
        artifact_id="art_src",
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        data=[{"service": "query-agent", "n": 1}],
    )
    tool = MockExportTool(tmp_path)
    result = tool.run(
        ExportArgs(artifact_id="art_src", filename="out.csv"), _ctx({"art_src": source})
    )
    assert result.ok is True
    assert result.artifact is not None
    assert result.artifact.artifact_type is ArtifactType.FILE
    assert Path(result.artifact.storage_ref or "").is_file()


def test_mock_analysis_empty_input() -> None:
    manager = ToolManager(use_mock=True)
    empty = Artifact(
        artifact_id="art_empty",
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        data=[],
        row_count=0,
    )
    result = manager.invoke("analysis", {"artifact_id": "art_empty"}, _ctx({"art_empty": empty}))

    assert result.ok is False
    assert "为空" in result.text
