"""执行循环：按 operations 逐步调工具或收步骤结论。"""

from __future__ import annotations

from agent.common.enums import ContextScope, OperationStatus, TaskStatus
from agent.common.errors import AgentError, LLMResponseFormatError
from agent.common.ids import new_id
from agent.llm.structured import call_structured, extract_json
from agent.models.context_window import ContextWindow
from agent.models.message import Message, ToolCall
from agent.models.task import Task
from agent.models.task_summary import Operation
from agent.prompt.assembler import CHAT_INTENTS, ExecuteOutput
from agent.task_manage.stages.base import BaseStage, StageDeps, StageResult, _check_duration
from agent.tool_manage.base import ToolContext, ToolResult

__all__ = ["ExecuteStage"]


class ExecuteStage(BaseStage):
    stage_name = "execute"

    def _run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
        window = deps.context_manager.load_window(task.task_id)
        operations = list(window.summary.operations)
        if not operations:
            return _direct_answer(task, window, deps)

        current = _next_operation(operations, resume=task.retry_count > 0)
        if current is None:
            return _finish(task, window, deps, operations)

        if deps.runtime.tool_calls >= deps.settings.task.max_tool_calls:
            return StageResult.failed(AgentError("达到单任务工具调用次数上限"))

        current = current.model_copy(update={"status": OperationStatus.RUNNING})
        operations[current.index] = current
        _save_operations(deps, task.task_id, operations)
        window = deps.context_manager.load_window(task.task_id)

        intent = window.summary.intent
        allowed = (
            list(deps.memory_manager.knowledge.get_allowed_tools(intent))
            if intent is not None
            else []
        )
        schemas = deps.tool_manager.get_schemas(allowed_tools=allowed) if allowed else []
        assembled = deps.prompt_assembler.build_execute_prompt(window, current, schemas)
        response = deps.llm.chat(assembled.request)

        if response.tool_calls:
            outcome = _run_tools(task, current, response.tool_calls, deps, allowed)
            if outcome is not None:
                return outcome
            operations = list(deps.context_manager.load_window(task.task_id).summary.operations)
            operations[current.index] = operations[current.index].model_copy(
                update={"status": OperationStatus.RUNNING}
            )
            _save_operations(deps, task.task_id, operations)
            return StageResult(next_status=TaskStatus.EXECUTING, note="等待模型给出步骤结论")

        conclusion = _parse_conclusion(response.content)
        if conclusion is None:
            return StageResult.failed(AgentError("执行阶段未能解析步骤结论"))
        if conclusion.status == "continue":
            deps.context_manager.append_message(
                task.task_id, Message.assistant(conclusion.conclusion)
            )
            return StageResult(next_status=TaskStatus.EXECUTING, note=conclusion.conclusion)

        last_ref = _latest_result_ref(deps, task.task_id)
        operations[current.index] = current.model_copy(
            update={
                "status": OperationStatus.SUCCEEDED,
                "result_ref": last_ref,
                "error": None,
            }
        )
        _save_operations(deps, task.task_id, operations)
        deps.context_manager.append_message(task.task_id, Message.assistant(conclusion.conclusion))
        task.retry_count = 0
        if _next_operation(operations) is None:
            return _finish(task, deps.context_manager.load_window(task.task_id), deps, operations)
        return StageResult(next_status=TaskStatus.EXECUTING, note=conclusion.conclusion)


def _direct_answer(task: Task, window: ContextWindow, deps: StageDeps) -> StageResult:
    dummy = Operation(index=0, name="直答", tool="")
    assembled = deps.prompt_assembler.build_execute_prompt(window, dummy, [])
    try:
        data = call_structured(
            deps.llm,
            assembled.request.messages,
            assembled.output_model or ExecuteOutput,
            purpose=assembled.request.purpose,
        )
        parsed = ExecuteOutput.model_validate(data)
    except LLMResponseFormatError as exc:
        if window.summary.intent in CHAT_INTENTS:
            parsed = ExecuteOutput(status="done", conclusion=window.summary.content)
        else:
            return StageResult.failed(AgentError("直答输出无法解析", detail=exc.message))
    fields = {"result": parsed.conclusion, "output": parsed.conclusion}
    deps.context_manager.update_summary(task.task_id, **fields)
    deps.context_manager.append_message(task.task_id, Message.assistant(parsed.conclusion))
    return StageResult(next_status=TaskStatus.VALIDATING, updated_fields=fields)


def _finish(
    task: Task,
    window: ContextWindow,
    deps: StageDeps,
    operations: list[Operation],
) -> StageResult:
    lines = []
    for item in operations:
        if item.status is OperationStatus.SUCCEEDED:
            lines.append(item.name + (f" → {item.result_ref}" if item.result_ref else ""))
    result_text = "；".join(lines) or "已完成"
    output = window.summary.output or result_text
    fields = {"result": window.summary.result or result_text, "output": output}
    deps.context_manager.update_summary(task.task_id, **fields)
    return StageResult(next_status=TaskStatus.VALIDATING, updated_fields=fields)


def _run_tools(
    task: Task,
    operation: Operation,
    calls: list[ToolCall],
    deps: StageDeps,
    allowed: list[str],
) -> StageResult | None:
    deps.context_manager.append_message(task.task_id, Message.assistant("", tool_calls=list(calls)))
    window = deps.context_manager.load_window(task.task_id)
    ctx = ToolContext(
        task_id=task.task_id,
        session_id=task.session_id,
        trace_id=task.trace_id,
        artifacts_reader=window.artifacts.__getitem__,
    )
    for call in calls:
        _check_duration(deps)
        if deps.runtime.canceled:
            return StageResult.canceled(deps.runtime.cancel_reason)
        if deps.runtime.tool_calls >= deps.settings.task.max_tool_calls:
            return StageResult.failed(AgentError("达到单任务工具调用次数上限"))
        deps.runtime.tool_calls += 1
        try:
            raw_args: object = call.parsed_arguments()
        except ValueError as exc:
            raw_args = {}
            result = ToolResult.fail(AgentError(str(exc), retryable=True))
        else:
            result = deps.tool_manager.invoke(
                call.name,
                raw_args,
                ctx,
                intent=window.summary.intent,
                allowed_tools=allowed,
            )
        if result.artifact is not None:
            deps.context_manager.add_artifact(task.task_id, result.artifact)
            window = deps.context_manager.load_window(task.task_id)
            ctx.artifacts_reader = window.artifacts.__getitem__
        deps.context_manager.append_message(
            task.task_id,
            Message.tool_result(
                result.text,
                tool_call_id=call.id or new_id("call"),
                meta={"error": (not result.ok)},
            ),
        )
        if result.ok:
            continue
        return _handle_tool_failure(task, operation, result, deps)
    return None


def _handle_tool_failure(
    task: Task,
    operation: Operation,
    result: ToolResult,
    deps: StageDeps,
) -> StageResult:
    retryable = bool(result.error and result.error.retryable)
    message = result.text or (result.error.message if result.error else "工具失败")
    window = deps.context_manager.load_window(task.task_id)
    operations = list(window.summary.operations)
    operations[operation.index] = operation.model_copy(
        update={"status": OperationStatus.FAILED, "error": message}
    )
    _save_operations(deps, task.task_id, operations)
    if retryable and task.retry_count < deps.settings.tool.max_retry:
        task.retry_count += 1
        return StageResult(next_status=TaskStatus.RETRYING, note=message)
    return StageResult.failed(
        AgentError(message, retryable=False, detail=result.error.detail if result.error else None)
    )


def _parse_conclusion(content: str) -> ExecuteOutput | None:
    try:
        return ExecuteOutput.model_validate(extract_json(content))
    except (ValueError, Exception):
        return None


def _next_operation(operations: list[Operation], *, resume: bool = False) -> Operation | None:
    for item in operations:
        if item.status is OperationStatus.PENDING:
            return item
        if item.status is OperationStatus.RUNNING:
            return item
        if resume and item.status is OperationStatus.FAILED:
            return item
    return None


def _latest_result_ref(deps: StageDeps, task_id: str) -> str | None:
    window = deps.context_manager.load_window(task_id)
    for artifact in reversed(list(window.artifacts.values())):
        if artifact.scope is ContextScope.CURRENT:
            return artifact.artifact_id
    return None


def _save_operations(deps: StageDeps, task_id: str, operations: list[Operation]) -> None:
    deps.context_manager.update_summary(task_id, operations=[item.to_dict() for item in operations])
