"""工具契约。

一个工具 = 四个类属性（``name`` / ``description`` / ``args_schema`` / ``internal``）
加一个 :meth:`BaseTool.run`。参数用 pydantic 模型声明，OpenAI 的 function schema 由
``model_json_schema()`` 自动生成——参数定义只写一遍，不会出现"schema 和实现对不上"。

工具**只读**上下文：它拿到的是一个按 tool_call_id 取回历史结果的读函数，没有任何写
黑板的能力。写入统一由 ContextManager 完成，这样"谁改了黑板"永远只有一个答案。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from pydantic import BaseModel

from agent.models import AgentModel

__all__ = ["BaseTool", "ToolContext", "ToolResult", "ToolResultReader"]

ToolResultReader = Callable[[str], Any]
"""按 tool_call_id 取回黑板上的全量结果；取不到返回 ``None``。"""


@dataclass(slots=True)
class ToolContext:
    """一次工具调用的只读环境。"""

    task_id: str
    session_id: str
    read_tool_result: ToolResultReader

    def result_of(self, tool_call_id: str) -> Any:
        return self.read_tool_result(tool_call_id)


class ToolResult(AgentModel):
    """工具执行结果。

    ``data`` 是全量结果（写入黑板 tool_result），``summary`` 是给模型看的一句话概括
    （进预览首行）。失败不抛异常而是返回 ``ok=False``：模型能看到失败原因，还有机会在
    本步骤的剩余轮次里改参数重试。
    """

    ok: bool = True
    data: Any = None
    summary: str = ""
    error: str | None = None

    @classmethod
    def success(cls, data: Any, summary: str) -> Self:
        return cls(ok=True, data=data, summary=summary)

    @classmethod
    def failure(cls, error: str) -> Self:
        return cls(ok=False, error=error, summary=f"工具执行失败：{error}")


class BaseTool(ABC):
    """所有工具的基类。"""

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    internal: ClassVar[bool] = False
    """内部工具的结果**不写入**黑板 tool_result，直接回给模型。

    目前只有 ``fetch_tool_result`` 是内部工具：它取回的就是黑板里已有的数据，
    再存一份等于把同一份内容在窗口里翻倍。
    """

    @abstractmethod
    def run(self, args: Any, ctx: ToolContext) -> ToolResult:
        """执行工具。``args`` 已经过 :attr:`args_schema` 校验。"""

    @classmethod
    def to_openai_schema(cls) -> dict[str, Any]:
        """生成 OpenAI function calling 的工具声明。"""
        parameters = cls.args_schema.model_json_schema()
        # 内联 $defs，部分兼容端点不认 $ref
        parameters.pop("$defs", None)
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": parameters,
            },
        }
