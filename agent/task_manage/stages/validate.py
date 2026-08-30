"""校验阶段：对照原始请求判断是否完成，必要时重规划。"""

from __future__ import annotations

from typing import Any

from agent.common.enums import TaskStatus
from agent.common.errors import AgentError, LLMResponseFormatError
from agent.llm.structured import call_structured
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task import Task
from agent.prompt.assembler import ValidateOutput
from agent.task_manage.stages.base import BaseStage, StageDeps, StageResult

__all__ = ["ValidateStage"]


class ValidateStage(BaseStage):
    stage_name = "validate"

    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
        window = deps.context_manager.load_window(task.task_id)
        assembled = deps.prompt_assembler.build_validate_prompt(window)
        try:
            data = call_structured(
                deps.llm,
                assembled.request.messages,
                assembled.output_model or ValidateOutput,
                purpose=assembled.request.purpose,
            )
            parsed = ValidateOutput.model_validate(data)
        except LLMResponseFormatError as exc:
            return StageResult.failed(AgentError("校验输出无法解析", detail=exc.message))

        if parsed.satisfied:
            fields = {"output": parsed.final_output or window.summary.output or parsed.suggestion}
            deps.context_manager.update_summary(task.task_id, **fields)
            return StageResult(next_status=TaskStatus.COMPLETED, updated_fields=fields)

        replans = _replan_count(task)
        if replans >= deps.settings.task.max_replan:
            return StageResult.failed(
                AgentError(
                    f"校验未通过且重规划次数已达上限 {deps.settings.task.max_replan}",
                    detail="; ".join(parsed.missing) or parsed.suggestion,
                ),
                updated_fields={"output": parsed.suggestion or "校验未通过"},
            )

        missing = "；".join(parsed.missing) or parsed.suggestion or "校验认为结果不完整"
        deps.context_manager.append_message(
            task.task_id,
            Message.user(f"补充要求：{missing}"),
        )
        extras: dict[str, Any] = {}
        if parsed.suggestion:
            extras["output"] = parsed.suggestion
            deps.context_manager.update_summary(task.task_id, output=parsed.suggestion)
        return StageResult(
            next_status=TaskStatus.PLANNING,
            updated_fields=extras,
            note=f"重规划：{missing}",
        )


def _replan_count(task: Task) -> int:
    """已经发生的 VALIDATING → PLANNING 次数。"""
    statuses = [record.status for record in task.status_history]
    count = 0
    for index in range(1, len(statuses)):
        if statuses[index - 1] is TaskStatus.VALIDATING and statuses[index] is TaskStatus.PLANNING:
            count += 1
    return count
