"""阶段处理器：意图识别、规划、执行、校验。"""

from agent.task_manage.stages.base import BaseStage, StageDeps, StageResult, StageRuntime
from agent.task_manage.stages.execute import ExecuteStage
from agent.task_manage.stages.intent import IntentStage
from agent.task_manage.stages.plan import PlanStage
from agent.task_manage.stages.validate import ValidateStage

__all__ = [
    "BaseStage",
    "ExecuteStage",
    "IntentStage",
    "PlanStage",
    "StageDeps",
    "StageResult",
    "StageRuntime",
    "ValidateStage",
]
