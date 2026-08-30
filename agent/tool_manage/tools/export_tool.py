"""ExportTool：把 Artifact 导出为 CSV / XLSX，返回 FILE 产物。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from pydantic import Field

from agent.common.enums import ArtifactType
from agent.common.errors import ToolError, ToolInvocationError
from agent.common.ids import new_artifact_id
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.tool_manage.base import BaseTool, ToolContext, ToolDeps, ToolResult
from agent.tool_manage.registry import register_tool

__all__ = ["ExportArgs", "ExportTool"]


class ExportArgs(AgentModel):
    artifact_id: str = Field(min_length=1)
    format: Literal["csv", "xlsx"] = "csv"
    filename: str | None = None


@register_tool
class ExportTool(BaseTool):
    name = "export"
    description = "把已有表格产物导出为 CSV 或 XLSX 文件。"
    args_schema = ExportArgs
    produces = ArtifactType.FILE
    is_mock = False

    def __init__(self, export_dir: Path | None = None) -> None:
        self.export_dir = Path(export_dir) if export_dir is not None else Path("data/exports")

    @classmethod
    def from_deps(cls, deps: ToolDeps) -> ExportTool:
        directory = deps.export_dir if deps.export_dir is not None else deps.settings.export_dir
        return cls(Path(directory))

    def run(self, args: ExportArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            source = ctx.get_artifact(args.artifact_id)
            rows = _rows(source)
        except ToolError as exc:
            return ToolResult.fail(exc)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        filename = args.filename or f"{args.artifact_id}.{args.format}"
        path = (self.export_dir / filename).resolve()
        if not path.is_relative_to(self.export_dir.resolve()):
            return ToolResult.fail(ToolInvocationError("导出路径越出 export_dir", retryable=False))
        if args.format == "csv":
            _write_csv(path, rows)
        else:
            _write_xlsx(path, rows)
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            task_id=ctx.task_id,
            producer=self.name,
            artifact_type=ArtifactType.FILE,
            title=path.name,
            storage_ref=str(path),
            size_bytes=path.stat().st_size,
            row_count=len(rows),
            meta={"source_artifact_id": args.artifact_id, "format": args.format},
        )
        return ToolResult.success(f"已导出 {path.name}（{len(rows)} 行）", artifact)


def _rows(artifact: Artifact) -> list[dict[str, object]]:
    if not isinstance(artifact.data, list) or not artifact.data:
        raise ToolInvocationError("导出的产物没有表格数据", retryable=False)
    rows: list[dict[str, object]] = []
    for item in artifact.data:
        if not isinstance(item, dict):
            raise ToolInvocationError("导出只接受对象行组成的表格", retryable=False)
        rows.append(item)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise ToolInvocationError("导出 xlsx 需要安装 openpyxl") from exc
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    columns = list(rows[0])
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(col) for col in columns])
    book.save(path)
