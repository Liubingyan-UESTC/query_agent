"""Mock 查询工具：读取 tests/fixtures/mock_search_logs.json，产出真实结构的 TABLE。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from agent.common.enums import ArtifactType
from agent.common.errors import ToolInvocationError
from agent.common.ids import new_artifact_id
from agent.config.settings import ToolSettings
from agent.models.artifact import Artifact
from agent.models.base import AgentModel
from agent.tool_manage.base import BaseTool, ToolContext, ToolDeps, ToolResult
from agent.tool_manage.registry import register_tool

__all__ = ["MockSearchArgs", "MockSearchTool"]


class MockSearchArgs(AgentModel):
    index: str = "logs-app"
    limit: int = Field(default=100, gt=0)


@register_tool
class MockSearchTool(BaseTool):
    name = "search"
    description = "Mock 查询：从本地 fixture 返回表格，不访问 Elasticsearch。"
    args_schema = MockSearchArgs
    produces = ArtifactType.TABLE
    is_mock = True

    def __init__(
        self,
        fixture_path: Path | None = None,
        settings: ToolSettings | None = None,
    ) -> None:
        self.fixture_path = fixture_path or _default_fixture()
        self._settings = settings if settings is not None else ToolSettings.from_env(None)

    @classmethod
    def from_deps(cls, deps: ToolDeps) -> MockSearchTool:
        return cls(settings=deps.settings)

    def run(self, args: MockSearchArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        payload = _load_fixture(self.fixture_path)
        raw_hits = payload.get("hits")
        rows = list(raw_hits) if isinstance(raw_hits, list) else []
        if payload.get("index") and payload["index"] != args.index:
            rows = [row for row in rows if row.get("_index", args.index) == args.index]
        cap = min(args.limit, self._settings.max_result_rows)
        truncated = rows[:cap]
        schema = {str(key): "keyword" for key in (truncated[0] if truncated else {})}
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            task_id=ctx.task_id,
            producer=self.name,
            artifact_type=ArtifactType.TABLE,
            title=f"{args.index} mock 结果",
            data_schema=schema,
            data=truncated,
            row_count=len(rows),
            meta={"fixture": str(self.fixture_path), "truncated": len(rows) > len(truncated)},
        )
        return ToolResult.success(f"mock 查询 {args.index} 得到 {len(rows)} 行", artifact)


def _default_fixture() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "mock_search_logs.json"


def _load_fixture(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ToolInvocationError(f"缺少 mock fixture：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ToolInvocationError(f"mock fixture 必须是对象：{path}")
    return payload
