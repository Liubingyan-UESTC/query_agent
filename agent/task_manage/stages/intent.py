"""意图识别阶段：填写 intent 与 related_task_ids。"""

from __future__ import annotations

from agent.common.enums import IntentType, TaskStatus
from agent.common.errors import LLMResponseFormatError
from agent.common.logging import get_logger
from agent.llm.structured import call_structured
from agent.models.context_window import ContextWindow
from agent.models.task import Task
from agent.prompt.assembler import IntentOutput
from agent.task_manage.stages.base import BaseStage, StageDeps, StageResult

__all__ = ["IntentStage"]

logger = get_logger(__name__)


class IntentStage(BaseStage):
    stage_name = "intent"

    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
        working = deps.memory_manager.working(task.session_id)
        history = working.get_summary_history(limit=deps.settings.context.history_summary_limit)
        assembled = deps.prompt_assembler.build_intent_prompt(window, history)
        try:
            data = call_structured(
                deps.llm,
                assembled.request.messages,
                assembled.output_model or IntentOutput,
                purpose=assembled.request.purpose,
            )
            parsed = IntentOutput.model_validate(data)
        except LLMResponseFormatError as exc:
            logger.warning(
                "intent_degraded",
                extra={"task_id": task.task_id, "reason": "parse_failed", "detail": exc.message},
            )
            parsed = IntentOutput(
                intent=IntentType.CHAT,
                reason=f"解析失败降级为 chat：{exc.message}",
                confidence=0.0,
            )

        threshold = deps.settings.task.intent_min_confidence
        if parsed.confidence < threshold:
            logger.warning(
                "intent_degraded",
                extra={
                    "task_id": task.task_id,
                    "reason": "low_confidence",
                    "confidence": parsed.confidence,
                },
            )
            parsed = IntentOutput(
                intent=IntentType.CHAT,
                related_task_ids=parsed.related_task_ids,
                reason=f"置信度 {parsed.confidence} 低于 {threshold}，降级为 chat。{parsed.reason}",
                confidence=parsed.confidence,
            )

        known = set(working.list_task_ids())
        kept: list[str] = []
        for related_id in parsed.related_task_ids:
            if related_id in known and related_id != task.task_id:
                kept.append(related_id)
            else:
                logger.warning(
                    "related_task_dropped",
                    extra={"task_id": task.task_id, "related_task_id": related_id},
                )

        fields = {
            "intent": parsed.intent.value,
            "related_task_ids": kept,
        }
        deps.context_manager.update_summary(task.task_id, **fields)
        return StageResult(
            next_status=TaskStatus.PLANNING, updated_fields=fields, note=parsed.reason
        )
