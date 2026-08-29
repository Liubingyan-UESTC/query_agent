"""Task 实体骨架。

当前仅保证语法完整与可导入；完整字段集（session_id / summary / trace_id /
retry_count / status_history 等）与序列化方法在步骤 6 落地。
状态转移逻辑不属于本模块，由步骤 19 的 TaskStateMachine 负责。
"""

from agent.task_manage.type import TaskStatus


class Task:
    def __init__(
        self,
        task_id: str,
        status: TaskStatus = TaskStatus.CREATED,
    ) -> None:
        self.task_id = task_id
        self.status = status
