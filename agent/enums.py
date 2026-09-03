"""枚举：任务状态机、意图类型、消息角色、步骤状态。

``TaskStatus`` 及其转移表按 docs/require.md 给出的定义落地——状态机是这个系统的
主干，转移表就写在枚举自身上，不再另立 StateMachine 类，避免"合法转移"出现两份定义。
"""

from __future__ import annotations

from enum import Enum

from agent.errors import TaskStateError

__all__ = ["IntentType", "MessageRole", "OperationStatus", "TaskStatus"]


class TaskStatus(str, Enum):
    """
    任务状态枚举。

    所有状态按职能分为五类：生命周期主线、异步等待、人工介入、异常重试、终态。
    """

    # ==================== 一、核心生命周期（主线流程） ====================
    CREATED = "created"
    """任务已创建，上下文已初始化，等待进入意图识别。"""

    INTENTING = "intenting"
    """正在进行意图识别（LLM 解析用户输入）。"""

    PLANNING = "planning"
    """正在进行任务规划（生成 operations 步骤列表）。"""

    EXECUTING = "executing"
    """正在同步执行操作步骤（含工具调用、循环等）。"""

    VALIDATING = "validating"
    """正在验证执行结果是否满足用户需求。"""

    # ==================== 二、异步与长时操作（解耦阻塞） ====================
    WAITING = "waiting"
    """任务正在等待外部异步操作完成（如后台导出、Kibana 慢查询回调）。此时系统已释放线程，不占用 Agent 资源。"""

    PAUSED = "paused"
    """任务被人为暂停（如管理员审核、用户临时中断），恢复后可继续流转。"""

    # ==================== 三、人工介入（不确定场景兜底） ====================
    AWAITING_APPROVAL = "awaiting_approval"
    """任务在执行敏感操作前（如导出大量数据、删除索引），需等待用户二次确认。"""

    CLARIFYING = "clarifying"
    """任务因意图不明确（Unknown）进入人工澄清环节，等待用户补充查询条件。"""

    # ==================== 四、异常与重试（防死锁） ====================
    RETRYING = "retrying"
    """任务因临时错误（如 LLM API 429 限流、网络抖动）进入重试队列，等待指数退避后再次执行。"""

    # ==================== 五、终态（不可转移） ====================
    COMPLETED = "completed"
    """任务成功完成，结果已返回用户且已归档。"""

    CANCELED = "canceled"
    """用户主动取消任务。"""

    FAILED = "failed"
    """任务因不可恢复的错误（如模型持续输出格式错误、工具权限不足）而终止，不再自动重试。"""

    ABORTED = "aborted"
    """任务因系统级熔断（如全局超时、资源耗尽）被强制终止，区别于用户主动取消。"""

    # ==================== 辅助属性（便于状态机逻辑判断） ====================

    @property
    def is_terminal(self) -> bool:
        """是否为终态（任务生命周期已结束）。"""
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELED,
            TaskStatus.FAILED,
            TaskStatus.ABORTED,
        }

    @property
    def is_async_waiting(self) -> bool:
        """是否需要释放线程等待外部回调。"""
        return self in {
            TaskStatus.WAITING,
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.CLARIFYING,
        }

    @property
    def is_retryable(self) -> bool:
        """是否为可重试的临时异常状态（TaskManager 需配合退避策略）。"""
        return self == TaskStatus.RETRYING

    @property
    def requires_user_input(self) -> bool:
        """是否需要用户提供额外输入才能继续。"""
        return self in {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.CLARIFYING,
        }

    # ==================== 状态转移合法性校验 ====================

    def can_transition_to(self, target: TaskStatus) -> bool:
        """
        判断当前状态是否允许转移到目标状态。

        严格遵循状态机流转规则，非法转移将抛出 ValueError。
        """
        # 定义合法的转移映射表（当前状态 -> 允许的下一个状态集合）
        transitions: dict[TaskStatus, set[TaskStatus]] = {
            TaskStatus.CREATED: {TaskStatus.INTENTING, TaskStatus.CANCELED},
            TaskStatus.INTENTING: {
                TaskStatus.PLANNING,
                TaskStatus.CLARIFYING,  # 意图不明，转人工澄清
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
            },
            TaskStatus.PLANNING: {
                TaskStatus.EXECUTING,
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
            },
            TaskStatus.EXECUTING: {
                TaskStatus.WAITING,  # 触发异步操作
                TaskStatus.VALIDATING,  # 所有步骤同步执行完成
                TaskStatus.RETRYING,  # 遇到临时错误
                TaskStatus.AWAITING_APPROVAL,  # 遇到敏感操作
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,  # 系统熔断
            },
            TaskStatus.WAITING: {
                TaskStatus.EXECUTING,  # 异步回调完成，继续执行
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            },
            TaskStatus.RETRYING: {
                TaskStatus.EXECUTING,  # 重试成功
                TaskStatus.FAILED,  # 重试次数耗尽
                TaskStatus.CANCELED,
                TaskStatus.ABORTED,
            },
            TaskStatus.AWAITING_APPROVAL: {
                TaskStatus.EXECUTING,  # 用户确认
                TaskStatus.CANCELED,  # 用户拒绝
                TaskStatus.FAILED,
            },
            TaskStatus.CLARIFYING: {
                TaskStatus.INTENTING,  # 用户补充信息后重新意图识别
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
            },
            TaskStatus.PAUSED: {
                TaskStatus.EXECUTING,  # 恢复执行
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            },
            TaskStatus.VALIDATING: {
                TaskStatus.COMPLETED,  # 验证通过
                TaskStatus.EXECUTING,  # 验证不通过，补充执行
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            },
            # 终态不可再转移
            TaskStatus.COMPLETED: set(),
            TaskStatus.CANCELED: set(),
            TaskStatus.FAILED: set(),
            TaskStatus.ABORTED: set(),
        }
        allowed = transitions.get(self, set())
        return target in allowed

    def transition_to(self, target: TaskStatus, task_id: str) -> TaskStatus:
        """
        执行状态转移，若非法则抛出异常（配合乐观锁使用）。
        """
        if not self.can_transition_to(target):
            raise TaskStateError(f"Task {task_id} 非法状态转移: {self.value} -> {target.value}")
        return target


class IntentType(str, Enum):
    """用户请求的意图。决定规划/执行阶段装配哪套系统提示词。"""

    QUERY = "query"
    """查询数据：按条件从日志库检索原始记录。"""

    ANALYSIS = "analysis"
    """分析统计：对检索到的数据做分组、聚合、排序。"""

    CHAT = "chat"
    """闲聊或询问系统能力，不需要工具。"""

    UNKNOWN = "unknown"
    """意图不明，需要用户澄清（对应 TaskStatus.CLARIFYING）。"""


class MessageRole(str, Enum):
    """对话消息角色，取值与 OpenAI messages 规范一致。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class OperationStatus(str, Enum):
    """规划出的单个步骤的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
