"""TaskManager：用状态机驱动一次任务走完 require.md 的五个步骤。

::

    created → intenting → planning → executing → validating → completed
                  ↓                   ↑    ↓         ↓
              clarifying ─────────────┘ retrying ────┘（不满足则补充执行）

每个阶段是一个方法，直接改黑板并**返回下一个状态**——不引入阶段结果对象、依赖容器、
运行时容器那一整套间接层。整个推进循环就是 :meth:`TaskManager.run` 里的一个 while。

出错时的三条去向：

- ``RETRYING``——执行阶段遇到可重试的临时错误（限流、网络抖动），退避后重跑当前步骤；
- ``FAILED``——不可恢复：模型反复给不出合法 JSON、调用了不存在的工具、重试次数用尽；
- ``ABORTED``——系统熔断：步骤数、工具调用数、单步轮数、重校验次数任一护栏耗尽。

护栏全部读 ``settings.task``，判定集中在 :meth:`_spend_tool_call` 与各阶段入口，
不散落在各处。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.config import AppSettings
from agent.context import ContextManager, ContextWindow
from agent.enums import IntentType, OperationStatus, TaskStatus
from agent.errors import AgentError, TaskStateError
from agent.knowledge import KnowledgeMemory
from agent.llm import BaseLLMClient, LLMRequest, build_llm_client, call_structured
from agent.memory import MemoryManager
from agent.models import Message, Operation, Task, ToolCall, new_session_id
from agent.tools import ToolContext, ToolManager, build_tool_manager

__all__ = [
    "IntentDecision",
    "PlanDecision",
    "TaskManager",
    "ValidateDecision",
    "build_task_manager",
]


class _CircuitBreakError(AgentError):
    """护栏耗尽。与普通错误分开，好让编排层转 ABORTED 而不是 FAILED。"""

    code = "circuit_break"


class _StageSchema(BaseModel):
    """阶段结构化输出的基类：多余字段一律忽略。

    模型爱附赠 ``confidence`` / ``thoughts`` 之类的键，为此让整个任务失败不划算。
    """

    model_config = ConfigDict(extra="ignore")


class IntentDecision(_StageSchema):
    intent: IntentType
    related_task_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    clarification: str = ""


class PlanStep(_StageSchema):
    description: str
    suggested_tool: str | None = None


class PlanDecision(_StageSchema):
    operations: list[PlanStep] = Field(default_factory=list)


class ValidateDecision(_StageSchema):
    satisfied: bool
    output: str = ""
    reason: str = ""


class TaskManager:
    """任务的创建、推进与归档。"""

    def __init__(
        self,
        *,
        settings: AppSettings,
        llm: BaseLLMClient,
        context: ContextManager,
        memory: MemoryManager,
        tools: ToolManager,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.context = context
        self.memory = memory
        self.tools = tools
        self._sleep = sleeper or time.sleep

        self._tasks: dict[str, Task] = {}
        self._windows: dict[str, ContextWindow] = {}
        self._tool_calls: dict[str, int] = {}
        self._revalidations: dict[str, int] = {}
        self._retries: dict[str, int] = {}

    # ================================================================ 对外接口

    def create_task(self, content: str, *, session_id: str | None = None) -> Task:
        """第零步：建任务、建窗、写入系统提示词与用户原文。"""
        task = Task.create(content, session_id=session_id or new_session_id())
        self._tasks[task.task_id] = task
        self._windows[task.task_id] = self.context.create_window(task)
        self._tool_calls[task.task_id] = 0
        self._revalidations[task.task_id] = 0
        self._retries[task.task_id] = 0
        return task

    def task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskStateError(f"未知任务：{task_id}")
        return task

    def window(self, task_id: str) -> ContextWindow:
        self.task(task_id)
        return self._windows[task_id]

    def run(self, task_id: str) -> Task:
        """推进任务，直到进入终态或需要用户输入（CLARIFYING）。

        每轮都要求状态真的变了。转移表里没有任何自环，所以"状态没动"只可能是某个阶段
        忘了转移——那是代码 bug。让它当场报错，好过让控制台静默挂死。
        """
        task = self.task(task_id)
        while not task.status.is_terminal and not task.status.requires_user_input:
            before = task.status
            self._step(task)
            if task.status is before:
                raise TaskStateError(f"阶段 {before.value} 未推进状态，中止以避免死循环")
        return task

    def provide_clarification(self, task_id: str, supplement: str) -> Task:
        """用户补充信息后回到意图识别（require.md 的 clarifying → intenting）。"""
        task = self.task(task_id)
        if task.status is not TaskStatus.CLARIFYING:
            raise TaskStateError(f"任务 {task_id} 当前状态是 {task.status.value}，不在等待澄清")
        window = self._windows[task_id]
        self.context.amend_user_request(window, supplement)
        self.context.update_summary(window, output=None)
        task.transition_to(TaskStatus.INTENTING)
        return self.run(task_id)

    def cancel(self, task_id: str, reason: str = "用户取消") -> Task:
        task = self.task(task_id)
        if task.status.is_terminal:
            return task
        self._terminate(task, TaskStatus.CANCELED, reason)
        return task

    # ================================================================ 推进

    def _step(self, task: Task) -> None:
        window = self._windows[task.task_id]
        try:
            if task.status is TaskStatus.CREATED:
                task.transition_to(TaskStatus.INTENTING)
                return
            handlers = {
                TaskStatus.INTENTING: self._run_intent,
                TaskStatus.PLANNING: self._run_plan,
                TaskStatus.EXECUTING: self._run_execute,
                TaskStatus.VALIDATING: self._run_validate,
                TaskStatus.RETRYING: self._run_retry,
            }
            handler = handlers.get(task.status)
            if handler is None:
                raise TaskStateError(f"状态 {task.status.value} 没有对应的处理阶段")
            handler(task, window)
        except _CircuitBreakError as exc:
            self._terminate(task, TaskStatus.ABORTED, exc.message)
        except AgentError as exc:
            self._handle_error(task, exc)

        if task.status.is_terminal:
            self.memory.archive(window)

    def _handle_error(self, task: Task, exc: AgentError) -> None:
        """可重试的临时错误进 RETRYING，其余直接进终态。

        ``retryable`` 由错误自己声明（限流、网络抖动为真；工具未注册、格式错误为假）。
        转移表只给了 ``executing → retrying`` 这一条边，所以其他阶段的临时错误仍走 FAILED
        ——这是 require.md 的规定，不在这里绕开。
        """
        if not exc.retryable or not task.status.can_transition_to(TaskStatus.RETRYING):
            self._terminate(task, TaskStatus.FAILED, exc.message)
            return

        used = self._retries[task.task_id] + 1
        limit = self.settings.task.max_retry
        if used > limit:
            self._terminate(task, TaskStatus.FAILED, f"重试 {limit} 次后仍失败：{exc.message}")
            return
        self._retries[task.task_id] = used
        task.transition_to(TaskStatus.RETRYING)

    def _run_retry(self, task: Task, window: ContextWindow) -> None:  # noqa: ARG002 - 阶段签名一致
        """指数退避后回到执行阶段，从未完成的步骤继续。"""
        attempt = self._retries[task.task_id]
        self._sleep(self.settings.task.retry_backoff_seconds * (2 ** (attempt - 1)))
        task.transition_to(TaskStatus.EXECUTING)

    # ---------------------------------------------------------------- 第一步：意图

    def _run_intent(self, task: Task, window: ContextWindow) -> None:
        working = self.memory.working(task.session_id)
        messages = self.context.build_intent_messages(
            window, working.summaries(self.settings.context.history_summary_limit)
        )
        decision = call_structured(self.llm, messages, IntentDecision, stage="intent")

        # 模型可能编造 task_id，只保留本会话里真实存在的
        known = working.known_task_ids()
        related = [task_id for task_id in decision.related_task_ids if task_id in known]
        self.context.update_summary(window, intent=decision.intent, related_task_ids=related)

        if decision.intent is IntentType.UNKNOWN:
            question = (
                decision.clarification or "请补充更具体的查询条件（服务名、级别、关键字等）。"
            )
            self.context.append_message(window, Message.assistant(question))
            # 澄清问题借 output 字段回传给控制台——它就是这一轮要展示给用户的内容
            self.context.update_summary(window, output=question)
            task.transition_to(TaskStatus.CLARIFYING)
            return

        task.transition_to(TaskStatus.PLANNING)

    # ---------------------------------------------------------------- 第二步：规划

    def _run_plan(self, task: Task, window: ContextWindow) -> None:
        working = self.memory.working(task.session_id)
        related = working.related_contents(window.summary.related_task_ids)
        messages = self.context.build_plan_messages(window, related)
        decision = call_structured(self.llm, messages, PlanDecision, stage="plan")

        max_steps = self.settings.task.max_steps
        if len(decision.operations) > max_steps:
            raise _CircuitBreakError(
                f"规划出 {len(decision.operations)} 个步骤，超过上限 {max_steps}"
            )

        steps = decision.operations or [PlanStep(description="直接回答用户的请求")]
        operations = [
            Operation(
                index=index,
                description=step.description,
                # 规划里写了个不存在的工具时清空：执行阶段模型仍可自己选对的那个
                suggested_tool=step.suggested_tool
                if step.suggested_tool and self.tools.has(step.suggested_tool)
                else None,
            )
            for index, step in enumerate(steps)
        ]
        self.context.update_summary(window, operations=operations)
        task.transition_to(TaskStatus.EXECUTING)

    # ---------------------------------------------------------------- 第三步：执行

    def _run_execute(self, task: Task, window: ContextWindow) -> None:
        """遍历尚未完成的步骤；每步内部是"模型↔工具"的小循环。"""
        for operation in window.summary.operations:
            if operation.status is OperationStatus.SUCCEEDED:
                continue
            operation.status = OperationStatus.RUNNING
            self._run_operation(task, window, operation)

        results = [
            f"步骤{op.index + 1}：{op.result}" for op in window.summary.operations if op.result
        ]
        self.context.update_summary(window, result="\n".join(results))
        task.transition_to(TaskStatus.VALIDATING)

    def _run_operation(self, task: Task, window: ContextWindow, operation: Operation) -> None:
        max_rounds = self.settings.task.max_rounds_per_step
        for _round in range(max_rounds):
            messages = self.context.build_execute_messages(window, operation)
            response = self.llm.chat(
                LLMRequest(
                    messages=messages,
                    tools=self.tools.openai_schemas(),
                    stage="execute",
                )
            )
            if not response.wants_tool:
                operation.result = response.content
                operation.status = OperationStatus.SUCCEEDED
                self.context.append_message(window, Message.assistant(response.content))
                return

            self.context.append_message(
                window,
                Message.assistant(response.content, tool_calls=response.tool_calls),
            )
            for call in response.tool_calls:
                self._invoke_tool(task, window, call)

        operation.status = OperationStatus.FAILED
        raise _CircuitBreakError(
            f"步骤 {operation.index + 1} 在 {max_rounds} 轮内没有得出结论：{operation.description}"
        )

    def _invoke_tool(self, task: Task, window: ContextWindow, call: ToolCall) -> None:
        """执行一次工具调用，并把"该回给模型什么"写回对话。"""
        self._spend_tool_call(task)
        result = self.tools.invoke(call, self._tool_context(task, window))

        if self.tools.is_internal(call.name):
            # fetch_tool_result：全量直接回给模型，不再写入黑板（否则同一份数据翻倍）
            text = _dump(result.data) if result.ok else f"取回失败：{result.error}"
        else:
            text = self.context.record_tool_result(
                window,
                tool_call_id=call.id,
                tool_name=call.name,
                result=result,
            )

        # 无论是否内部工具，都必须回一条同 id 的 tool 消息：
        # OpenAI 要求 assistant 的每个 tool_call 都有应答，缺一条下轮请求就会 400。
        self.context.append_message(
            window, Message.tool(text, tool_call_id=call.id, name=call.name)
        )

    def _spend_tool_call(self, task: Task) -> None:
        used = self._tool_calls[task.task_id] + 1
        limit = self.settings.task.max_tool_calls
        if used > limit:
            raise _CircuitBreakError(f"工具调用次数超过上限 {limit}")
        self._tool_calls[task.task_id] = used

    def _tool_context(self, task: Task, window: ContextWindow) -> ToolContext:
        return ToolContext(
            task_id=task.task_id,
            session_id=task.session_id,
            read_tool_result=lambda call_id: self.context.read_tool_result(window, call_id),
        )

    # ---------------------------------------------------------------- 第四步：校验

    def _run_validate(self, task: Task, window: ContextWindow) -> None:
        messages = self.context.build_validate_messages(window)
        decision = call_structured(self.llm, messages, ValidateDecision, stage="validate")

        if decision.satisfied:
            output = decision.output or window.summary.result or "任务已完成。"
            self.context.update_summary(window, output=output)
            self.context.append_message(window, Message.assistant(output))
            task.transition_to(TaskStatus.COMPLETED)
            return

        used = self._revalidations[task.task_id] + 1
        limit = self.settings.task.max_revalidate
        if used > limit:
            raise _CircuitBreakError(f"校验连续 {limit} 次未通过：{decision.reason}")
        self._revalidations[task.task_id] = used

        # 补充执行：追加一个新步骤，执行循环才有事可做
        operations = [
            *window.summary.operations,
            Operation(
                index=len(window.summary.operations),
                description=f"根据校验反馈补充执行：{decision.reason}",
            ),
        ]
        self.context.update_summary(window, operations=operations)
        task.transition_to(TaskStatus.EXECUTING)

    # ---------------------------------------------------------------- 终态

    def _terminate(self, task: Task, target: TaskStatus, reason: str) -> None:
        """转终态并把原因写进 output——控制台就是靠它告诉用户发生了什么。

        终态是按转移表挑的，不是想落哪个就落哪个：

        - ``created`` 只能去 ``intenting`` 或 ``canceled``，所以失败前先推进一格；
        - ``planning`` 没有到 ``aborted`` 的边（require.md 的表里只有 executing /
          waiting / retrying / paused / validating 能熔断），此时退回 ``failed``。

        宁可换一个语义相近的终态，也不绕过状态机——转移表是这个系统的主干约束。
        """
        window = self._windows[task.task_id]
        if task.status is TaskStatus.CREATED and target is not TaskStatus.CANCELED:
            task.transition_to(TaskStatus.INTENTING)
        if not task.status.can_transition_to(target):
            target = TaskStatus.FAILED
        task.transition_to(target)
        task.error = {"status": target.value, "reason": reason}
        self.context.update_summary(window, output=f"[{target.value}] {reason}")
        self.memory.archive(window)


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def build_task_manager(
    settings: AppSettings,
    *,
    llm: BaseLLMClient | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> TaskManager:
    """按配置装配一整套依赖。控制台与测试都从这里拿 TaskManager。"""
    tools = build_tool_manager(settings)
    knowledge = KnowledgeMemory(tools.catalog())
    return TaskManager(
        settings=settings,
        llm=llm or build_llm_client(settings),
        context=ContextManager(settings, knowledge),
        memory=MemoryManager(knowledge, max_tasks=settings.memory.max_tasks),
        tools=tools,
        sleeper=sleeper,
    )
