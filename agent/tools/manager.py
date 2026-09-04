"""ToolManager：工具的注册、发现与调用。

调用路径上有两类错误，处理方式不同：

- **未注册的工具**——模型凭空捏造了一个名字，而工具清单就在提示词里。这属于不可恢复，
  抛 :class:`~agent.errors.ToolError`，由编排层转终态；
- **参数不合法**——模型选对了工具但填错了参数。这是**可恢复**的：返回一个 ``ok=False``
  的 :class:`ToolResult`，错误原文会作为 tool 消息回到模型面前，它可以在本步骤剩余的
  轮次里改参数重试。真改不过来，最终由步骤轮次护栏兜底。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from agent.config import AppSettings, ToolSettings
from agent.errors import ToolError
from agent.models import ToolCall
from agent.tools.analysis_tool import AnalysisTool
from agent.tools.base import BaseTool, ToolContext, ToolResult
from agent.tools.fetch_tool import FetchToolResultTool
from agent.tools.search_tool import SearchTool

__all__ = ["ToolManager", "build_tool_manager"]


class ToolManager:
    """工具注册表 + 调用入口。"""

    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    # ---------------------------------------------------------------- 注册与发现

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"工具重名：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(
                f"未注册的工具：{name}。可用工具：{self.names()}",
                detail={"available": self.names()},
            )
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def is_internal(self, name: str) -> bool:
        """内部工具的结果不写入黑板，见 :attr:`BaseTool.internal`。"""
        return self.has(name) and self._tools[name].internal

    def openai_schemas(self) -> list[dict[str, Any]]:
        """按名称排序输出，保证同一份配置每次生成的提示词一致（便于比对与缓存）。"""
        return [self._tools[name].to_openai_schema() for name in self.names()]

    def catalog(self) -> str:
        """工具清单的自然语言描述，供规划阶段的系统提示词使用。"""
        lines = [f"- {name}：{self._tools[name].description}" for name in self.names()]
        return "\n".join(lines)

    # ---------------------------------------------------------------- 调用

    def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """执行一次工具调用。未注册的工具抛错，参数问题收成失败结果。"""
        tool = self.get(call.name)
        try:
            raw_args = call.parsed_arguments()
        except ValueError as exc:
            return ToolResult.failure(f"{exc}")

        try:
            args = tool.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult.failure(f"参数不符合 {tool.name} 的定义：{_describe(exc)}")

        try:
            return tool.run(args, ctx)
        except ToolError:
            raise
        except Exception as exc:
            return ToolResult.failure(f"{tool.name} 执行异常：{exc}")


def _describe(exc: ValidationError) -> str:
    """把 pydantic 的错误压成一行，模型才好照着改。"""
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "；".join(parts)


def build_tool_manager(settings: AppSettings | ToolSettings) -> ToolManager:
    """装配默认工具集。"""
    tool_settings = settings.tool if isinstance(settings, AppSettings) else settings
    tools: Sequence[BaseTool] = (
        SearchTool(tool_settings),
        AnalysisTool(),
        FetchToolResultTool(),
    )
    return ToolManager(tools)
