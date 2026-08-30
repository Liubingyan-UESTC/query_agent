"""Mock 分析工具：对已有 Artifact 做简单计数 / 分组，也可回放 fixture。"""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from agent.common.enums import ArtifactType
from agent.common.errors import ToolInvocationError
from agent.common.ids import new_artifact_id
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.tool_manage.base import BaseTool, ToolContext, ToolResult
from agent.tool_manage.registry import register_tool

__all__ = ["MockAnalysisArgs", "MockAnalysisTool"]


class MockAnalysisArgs(AgentModel):
    artifact_id: str = Field(min_length=1)
    group_by: str | None = None


@register_tool
class MockAnalysisTool(BaseTool):
    name = "analysis"
    description = "Mock 分析：对已有表格按字段计数。"
    args_schema = MockAnalysisArgs
    produces = ArtifactType.TABLE
    is_mock = True

    def run(self, args: MockAnalysisArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        source = ctx.get_artifact(args.artifact_id)
        if not isinstance(source.data, list) or not source.data:
            return ToolResult.fail(ToolInvocationError("输入产物为空，无法分析", retryable=False))
        if args.group_by:
            values = []
            for row in source.data:
                if not isinstance(row, dict):
                    return ToolResult.fail(ToolInvocationError("分析只接受对象行", retryable=False))
                values.append(row.get(args.group_by))
            counts = Counter(values)
            records = [{args.group_by: key, "count": count} for key, count in counts.items()]
        else:
            records = [{"count": len(source.data)}]
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            task_id=ctx.task_id,
            producer=self.name,
            artifact_type=ArtifactType.TABLE,
            title=f"mock 分析 {args.artifact_id}",
            data=records,
            row_count=len(records),
            meta={"source_artifact_id": args.artifact_id},
        )
        return ToolResult.success(f"mock 分析得到 {len(records)} 行", artifact)
