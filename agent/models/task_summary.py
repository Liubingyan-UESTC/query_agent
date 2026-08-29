"""任务摘要：Context Window 三部分中「task_summary」的字段集。

这是模型与前端共享的「任务名片」。意图识别阶段只看 `content` / `status`，规划阶段
才看 `operations`，归档时整份写入 WorkingMemory。因此除了全量 `to_dict()`，还提供
`to_prompt_dict(fields)`：调用方显式点名要投喂的字段，避免把尚未填写的 operations
或内部错误泄漏进提示词。

字段名以开发计划 1.2 节的修正为准（`related_task_ids`，不是需求文档里的
`related task`）。不做旧名别名——笔误就是错的。
"""

from collections.abc import Sequence
from typing import Any

from pydantic import Field, field_validator

from agent.common.enums import IntentType, OperationStatus, TaskStatus
from agent.models.base import AgentModel

__all__ = ["INTENT_PROMPT_FIELDS", "Operation", "TaskSummary"]

# 需求文档第 1 步：意图识别只投喂 content 与 status
INTENT_PROMPT_FIELDS: tuple[str, ...] = ("content", "status")


class Operation(AgentModel):
    """`TaskSummary.operations` 中的一个步骤。

    `result_ref` 指向产物的 `artifact_id`，不内嵌数据：全量结果走 Artifact，
    摘要里只留引用。`args` 是已解析的对象——与 `ToolCall.arguments`（原样 JSON
    字符串）不同：operations 是规划阶段校验过的内部结构，不是模型原始输出。
    """

    index: int
    name: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: OperationStatus = OperationStatus.PENDING
    result_ref: str | None = None
    error: str | None = None

    @field_validator("index")
    @classmethod
    def _index_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"operation.index 必须 >= 0，实际为 {value}")
        return value


class TaskSummary(AgentModel):
    """一次任务的状态摘要。字段集与需求文档一致（命名按 1.2 节修正）。"""

    task_id: str
    content: str
    intent: IntentType | None = None
    related_task_ids: list[str] = Field(default_factory=list)
    operations: list[Operation] = Field(default_factory=list)
    result: str | None = None
    status: TaskStatus = TaskStatus.CREATED
    output: str | None = None

    @field_validator("operations")
    @classmethod
    def _operation_indices_are_unique(cls, value: list[Operation]) -> list[Operation]:
        """重复 index 会让执行循环无法判断下一步该跑哪条。"""
        indices = [item.index for item in value]
        if len(indices) != len(set(indices)):
            raise ValueError(f"operations 的 index 必须唯一，实际为 {indices}")
        return value

    def to_prompt_dict(self, fields: Sequence[str]) -> dict[str, Any]:
        """按白名单投影，供提示词装配使用。

        `fields` 必须显式给出：默认全量会把尚未发生的 operations、失败现场的
        `output` 一并投喂，诱导模型抄作业或泄漏内部错误。空列表同样拒绝——
        那几乎总是调用方漏传，而不是真想发一份空摘要。
        """
        if not fields:
            raise ValueError("to_prompt_dict 必须指定字段，空投影几乎总是漏传")
        payload = self.to_dict()
        unknown = [name for name in fields if name not in payload]
        if unknown:
            raise ValueError(f"未知字段 {unknown}；合法字段为 {sorted(payload)}")
        return {name: payload[name] for name in fields}

    def operation(self, index: int) -> Operation:
        """按 index 取步骤。找不到时抛 `KeyError`，与 dict 访问语义一致。"""
        for item in self.operations:
            if item.index == index:
                return item
        raise KeyError(f"summary {self.task_id} 没有 index={index} 的 operation")
