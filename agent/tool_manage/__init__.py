"""ToolManager：工具抽象、注册发现、调用管理与具体工具实例。

本层依赖 common / config / models，与 llm / memory / context 互不引用。
字段目录与 ES 客户端以 Protocol 注入。
"""

from agent.tool_manage.args import AnalysisAgg, QueryFilter, SearchAggregation, TimeRange
from agent.tool_manage.base import (
    BaseTool,
    FieldCatalog,
    ToolContext,
    ToolDeps,
    ToolResult,
)
from agent.tool_manage.es_client import ESClient, FakeESClient, StaticFieldCatalog
from agent.tool_manage.manager import ToolManager
from agent.tool_manage.registry import ToolRegistry, default_registry, register_tool

__all__ = [
    "AnalysisAgg",
    "BaseTool",
    "ESClient",
    "FakeESClient",
    "FieldCatalog",
    "QueryFilter",
    "SearchAggregation",
    "StaticFieldCatalog",
    "TimeRange",
    "ToolContext",
    "ToolDeps",
    "ToolManager",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "register_tool",
]
