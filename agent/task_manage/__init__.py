"""TaskManager：任务状态机、阶段处理器、终态收口与任务调度。"""

from agent.task_manage.finalizer import TaskFinalizer
from agent.task_manage.state_machine import LEGAL_TRANSITIONS, TaskStateMachine
from agent.task_manage.task_manager import TaskManager

__all__ = ["LEGAL_TRANSITIONS", "TaskFinalizer", "TaskManager", "TaskStateMachine"]
