"""向后兼容的枚举重导出层。

枚举的唯一定义处已迁移到 `agent/common/enums.py`（见开发计划步骤 4）；本模块保留
仅为不破坏既有 `from agent.task_manage.type import ...` 的引用。新代码请直接从
`agent.common.enums` 导入。

`TaskType` 是 `IntentType` 的类别名。v0 的成员拼写（`NEWQUERY`、`ANALISIS`）是笔误，
已按开发计划 1.2 节更正为 `NEW_QUERY`、`ANALYSIS`，且不保留任何形式的兼容：
`TaskType.NEWQUERY` 会抛 `AttributeError`，mypy 亦会静态报错。
"""

from agent.common.enums import IntentType, TaskStatus

# v0 时期的名称
TaskType = IntentType

__all__ = ["IntentType", "TaskStatus", "TaskType"]
