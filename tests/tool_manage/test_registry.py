"""步骤 16：注册 / 发现 / 重名 / to_openai_schema。"""

from __future__ import annotations

import pytest

from agent.common.enums import ArtifactType
from agent.common.errors import ToolError
from agent.config.settings import ToolSettings
from agent.models.base import AgentModel
from agent.tool_manage.base import BaseTool, ToolContext, ToolDeps, ToolResult
from agent.tool_manage.registry import ToolRegistry, register_tool


class _EmptyArgs(AgentModel):
    q: str = "x"


class _Echo(BaseTool):
    name = "echo"
    description = "回声"
    args_schema = _EmptyArgs
    produces = ArtifactType.TEXT

    def run(self, args: _EmptyArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        return ToolResult.success(args.q)


def test_register_and_instantiate() -> None:
    registry = ToolRegistry()
    register_tool(_Echo, registry=registry)
    tools = registry.instantiate(ToolDeps(settings=ToolSettings.from_env(None)), use_mock=False)

    assert list(tools) == ["echo"]
    schema = tools["echo"].to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"] == "回声"
    assert schema["function"]["parameters"]["properties"]["q"]["default"] == "x"


def test_duplicate_name_is_rejected() -> None:
    registry = ToolRegistry()

    class Other(_Echo):
        name = "echo"
        description = "另一个"

        def run(self, args: _EmptyArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
            return ToolResult.success("other")

    register_tool(_Echo, registry=registry)
    register_tool(Other, registry=registry)
    with pytest.raises(ToolError, match="重名"):
        registry.instantiate(ToolDeps(settings=ToolSettings.from_env(None)), use_mock=False)


def test_same_class_registered_twice_is_idempotent() -> None:
    registry = ToolRegistry()
    register_tool(_Echo, registry=registry)
    register_tool(_Echo, registry=registry)
    tools = registry.instantiate(ToolDeps(settings=ToolSettings.from_env(None)))
    assert list(tools) == ["echo"]


def test_register_rejects_non_tool() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolError, match="BaseTool"):
        registry.add_class(object)  # type: ignore[arg-type]


def test_discover_loads_packaged_tools() -> None:
    registry = ToolRegistry()
    registry.discover()
    mock_names = {cls.name for cls in registry.classes if cls.is_mock}
    real_names = {cls.name for cls in registry.classes if not cls.is_mock}
    assert mock_names == {"search", "analysis", "export"}
    assert real_names == {"search", "analysis", "export"}


def test_decorator_without_parens() -> None:
    registry = ToolRegistry()

    @register_tool(registry=registry)
    class Ping(BaseTool):
        name = "ping"
        description = "ping"
        args_schema = _EmptyArgs

        def run(self, args: _EmptyArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
            return ToolResult.success("pong")

    assert "ping" in {cls.name for cls in registry.classes}
