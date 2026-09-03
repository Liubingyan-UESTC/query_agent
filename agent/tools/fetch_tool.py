"""FetchToolResultTool：按 tool_call_id 取回工具结果的全量内容。

require.md 里的那个"特殊工具"。普通工具的结果只把**预览**回给模型，全量留在黑板上；
模型觉得预览不够用时，就调这个工具把全量取回来。

它是 ``internal=True``：返回值直接进对话，**不再写入** tool_result 黑板。否则同一份
数据会在窗口里存两遍，正好抵消掉"大结果不进上下文"的设计目的。
"""

from __future__ import annotations

import json
from typing import ClassVar

from pydantic import BaseModel, Field

from agent.tools.base import BaseTool, ToolContext, ToolResult

__all__ = ["FetchArgs", "FetchToolResultTool"]


class FetchArgs(BaseModel):
    tool_call_id: str = Field(
        description="要取回全量结果的那次工具调用的 tool_call_id。",
    )


class FetchToolResultTool(BaseTool):
    name: ClassVar[str] = "fetch_tool_result"
    description: ClassVar[str] = (
        "按 tool_call_id 取回某次工具调用的完整结果。当结果预览被截断、你需要看到全部"
        "记录时使用。注意：只能取回普通工具（如 search_tool / analysis_tool）的结果。"
    )
    args_schema: ClassVar[type[BaseModel]] = FetchArgs
    internal: ClassVar[bool] = True

    def run(self, args: FetchArgs, ctx: ToolContext) -> ToolResult:
        payload = ctx.result_of(args.tool_call_id)
        if payload is None:
            return ToolResult.failure(
                f"黑板上没有 tool_call_id={args.tool_call_id} 的结果。"
                f"注意 fetch_tool_result 自身的结果不会存入黑板，无法被再次取回。"
            )
        text = json.dumps(payload, ensure_ascii=False)
        return ToolResult.success(
            payload, f"tool_call_id={args.tool_call_id} 的全量结果（{len(text)} 字符）"
        )
