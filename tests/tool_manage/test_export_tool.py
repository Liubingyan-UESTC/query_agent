"""步骤 18：导出 CSV / XLSX 可被正常打开。"""

from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import load_workbook

from agent.common.enums import ArtifactType
from agent.models.artifact import Artifact
from agent.tool_manage.base import ToolContext
from agent.tool_manage.tools.export_tool import ExportArgs, ExportTool


def _artifact() -> Artifact:
    return Artifact(
        artifact_id="art_1",
        task_id="task_1",
        producer="search",
        artifact_type=ArtifactType.TABLE,
        data=[{"service": "query-agent", "n": 2}, {"service": "billing", "n": 1}],
    )


def _ctx(artifact: Artifact) -> ToolContext:
    return ToolContext(
        task_id="task_1",
        session_id="sess_1",
        trace_id="trace_1",
        artifacts_reader=lambda aid: artifact,
    )


def test_export_csv_round_trip(tmp_path: Path) -> None:
    tool = ExportTool(tmp_path)
    result = tool.run(
        ExportArgs(artifact_id="art_1", format="csv", filename="out.csv"), _ctx(_artifact())
    )

    assert result.ok is True
    assert result.artifact is not None
    assert result.artifact.artifact_type is ArtifactType.FILE
    path = Path(result.artifact.storage_ref or "")
    assert path.is_file()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["service"] == "query-agent"
    assert rows[1]["n"] == "1"


def test_export_xlsx_can_be_opened(tmp_path: Path) -> None:
    tool = ExportTool(tmp_path)
    result = tool.run(
        ExportArgs(artifact_id="art_1", format="xlsx", filename="out.xlsx"), _ctx(_artifact())
    )

    assert result.ok is True
    path = Path(result.artifact.storage_ref if result.artifact else "")
    book = load_workbook(path)
    sheet = book.active
    assert sheet is not None
    assert [cell.value for cell in sheet[1]] == ["service", "n"]
    assert sheet["A2"].value == "query-agent"
    assert sheet["B3"].value == 1


def test_export_rejects_path_escape(tmp_path: Path) -> None:
    tool = ExportTool(tmp_path)
    result = tool.run(
        ExportArgs(artifact_id="art_1", format="csv", filename="../escape.csv"),
        _ctx(_artifact()),
    )

    assert result.ok is False
    assert "越出" in result.text
