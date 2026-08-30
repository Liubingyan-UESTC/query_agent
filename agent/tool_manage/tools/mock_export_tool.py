"""Mock 导出：把已有 Artifact 写成 CSV 桩文件，不依赖 openpyxl。"""

from __future__ import annotations

import csv
from pathlib import Path

from agent.common.enums import ArtifactType
from agent.common.errors import ToolError, ToolInvocationError
from agent.common.ids import new_artifact_id
from agent.models.artifact import Artifact
from agent.tool_manage.base import BaseTool, ToolContext, ToolDeps, ToolResult
from agent.tool_manage.registry import register_tool
from agent.tool_manage.tools.export_tool import ExportArgs

__all__ = ["MockExportTool"]


@register_tool
class MockExportTool(BaseTool):
    name = "export"
    description = "Mock 导出：把已有表格写成 CSV，供 EXPORT 意图在无真实落盘引擎时打通。"
    args_schema = ExportArgs
    produces = ArtifactType.FILE
    is_mock = True

    def __init__(self, export_dir: Path | None = None) -> None:
        self.export_dir = Path(export_dir) if export_dir is not None else Path("data/exports")

    @classmethod
    def from_deps(cls, deps: ToolDeps) -> MockExportTool:
        directory = deps.export_dir if deps.export_dir is not None else deps.settings.export_dir
        return cls(Path(directory))

    def run(self, args: ExportArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            source = ctx.get_artifact(args.artifact_id)
            rows = _rows(source)
        except ToolError as exc:
            return ToolResult.fail(exc)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        filename = args.filename or f"{args.artifact_id}.csv"
        path = (self.export_dir / filename).resolve()
        if not path.is_relative_to(self.export_dir.resolve()):
            return ToolResult.fail(ToolInvocationError("导出路径越出 export_dir", retryable=False))
        _write_csv(path, rows)
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            task_id=ctx.task_id,
            producer=self.name,
            artifact_type=ArtifactType.FILE,
            title=path.name,
            storage_ref=str(path),
            size_bytes=path.stat().st_size,
            row_count=len(rows),
            meta={"source_artifact_id": args.artifact_id, "format": "csv", "mock": True},
        )
        return ToolResult.success(f"mock 已导出 {path.name}（{len(rows)} 行）", artifact)


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
