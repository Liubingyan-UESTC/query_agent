"""向后兼容的枚举重导出层。

枚举的唯一定义处已迁移到 `agent/common/enums.py`（见开发计划步骤 4）；本模块保留
仅为不破坏既有 `from agent.task_manage.type import ...` 的引用。新代码请直接从
`agent.common.enums` 导入。

与 v0 的差异：`TaskType` 现为 `IntentType` 的别名，成员由 `NEWQUERY / ANALISIS`
更正为 `NEW_QUERY / ANALYSIS` 并新增 `UNKNOWN`。旧成员名不再作为属性提供，但旧的
字符串写法仍能被 `IntentType.from_str()` 正确解析（别名表已收录）。
"""

from agent.common.enums import IntentType, TaskStatus

# v0 时期的名称
TaskType = IntentType

__all__ = ["IntentType", "TaskStatus", "TaskType"]
