"""向后兼容的 Task 重导出层。

Task 的唯一定义处已迁移到 `agent/models/task.py`（开发计划步骤 6）。
本模块保留仅为不破坏既有 `from agent.task_manage.task import Task` 的引用。
新代码请直接从 `agent.models.task` 导入。

状态转移逻辑不属于本模块，也不属于模型层，由步骤 19 的 TaskStateMachine 负责。
"""

from agent.models.task import StatusRecord, Task

__all__ = ["StatusRecord", "Task"]
