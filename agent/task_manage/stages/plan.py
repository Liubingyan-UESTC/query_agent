"""规划阶段：产出 operations，校验工具白名单与参数 schema。"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from agent.common.enums import IntentType, TaskStatus
from agent.common.errors import AgentError, LLMResponseFormatError
from agent.llm.structured import call_structured
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task import Task
from agent.models.task_summary import Operation
from agent.prompt.assembler import CHAT_INTENTS, EmptyPlanOutput, PlanOutput, PlanStepOut
from agent.task_manage.stages.base import BaseStage, StageDeps, StageResult

__all__ = ["PlanStage"]


class PlanStage(BaseStage):
    stage_name = "plan"

    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
        window = deps.context_manager.load_window(task.task_id)
        intent = window.summary.intent
        if intent is None:
            raise AgentError("规划阶段窗口必须已填写 intent")

        if window.summary.related_task_ids and deps.injector is not None:
            deps.injector(task.task_id, list(window.summary.related_task_ids))
            window = deps.context_manager.load_window(task.task_id)

        operations, error = _plan_operations(window, deps, intent)
        if error is not None:
            return StageResult.failed(error)

        if len(operations) > deps.settings.task.max_steps:
            return StageResult.failed(
                AgentError(f"规划步骤数 {len(operations)} 超过上限 {deps.settings.task.max_steps}")
            )

        payload = [item.to_dict() for item in operations]
        deps.context_manager.update_summary(task.task_id, operations=payload)
        return StageResult(
            next_status=TaskStatus.EXECUTING,
            updated_fields={"operations": payload},
        )


def _plan_operations(
    window: ContextWindow,
    deps: StageDeps,
    intent: IntentType,
) -> tuple[list[Operation], AgentError | None]:
    assembled = deps.prompt_assembler.build_plan_prompt(window)
    model = assembled.output_model or (EmptyPlanOutput if intent in CHAT_INTENTS else PlanOutput)
    try:
        data = call_structured(
            deps.llm,
            assembled.request.messages,
            model,
            purpose=assembled.request.purpose,
        )
        steps = PlanOutput.model_validate(data).operations
        operations = _to_operations(steps)
        problem = _validate_plan(operations, intent, deps)
        if problem is None:
            return operations, None
        return _repair_plan(assembled.request.messages, model, problem, deps, intent)
    except LLMResponseFormatError as exc:
        return [], AgentError("规划输出无法通过校验", detail=exc.message)


def _repair_plan(
    messages: list[Message],
    model: type[BaseModel],
    problem: str,
    deps: StageDeps,
    intent: IntentType,
) -> tuple[list[Operation], AgentError | None]:
    conversation = [
        *messages,
        Message.user(f"上次规划无法通过校验：{problem}\n请只输出修正后的 JSON。"),
    ]
    try:
        data = call_structured(deps.llm, conversation, model, purpose="plan", max_repair=0)
        operations = _to_operations(PlanOutput.model_validate(data).operations)
    except (LLMResponseFormatError, ValidationError) as exc:
        return [], AgentError("规划修复后仍无法通过校验", detail=str(exc))
    leftover = _validate_plan(operations, intent, deps)
    if leftover is not None:
        return [], AgentError(leftover)
    return operations, None


def _to_operations(steps: list[PlanStepOut]) -> list[Operation]:
    return [
        Operation(index=step.index, name=step.name, tool=step.tool, args=dict(step.args))
        for step in steps
    ]


def _validate_plan(operations: list[Operation], intent: IntentType, deps: StageDeps) -> str | None:
    allowed = list(deps.memory_manager.knowledge.get_allowed_tools(intent))
    if intent in CHAT_INTENTS and operations:
        return "CHAT/UNKNOWN 不能规划工具步骤"
    if intent not in CHAT_INTENTS and not operations:
        return "该意图必须规划至少一个步骤"
    for item in operations:
        if item.tool not in allowed:
            return f"工具 {item.tool} 不在意图 {intent.value} 的白名单内：{allowed}"
        try:
            tool = deps.tool_manager.get(item.tool)
        except AgentError as exc:
            return exc.message
        try:
            tool.args_schema.model_validate(item.args or {})
        except ValidationError as exc:
            return f"步骤 {item.index} 参数不合法：{exc}"
    return None
