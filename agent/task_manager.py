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

**并发模型**：一个 TaskManager 服务所有会话，不同会话可以同时推进（HTTP 层按 session
加锁，见 :mod:`server.api.runtime`），同一会话内串行。因此这里的约定是：

- 任务注册表（``_slots``）是**跨会话共享**的，一切读写都在 ``_registry_lock`` 内，
  且锁内只做字典操作，绝不调用模型或工具——那会把跨会话的并发又退化成串行；
- 单个任务的状态（黑板、护栏计数器）收在 :class:`_TaskSlot` 里。它只被持有该会话锁的
  那一个线程访问，所以槽内字段不需要再加锁。

把计数器从五个平行字典改成一个槽，不只是整理：平行字典下"任务存在"不是原子的
（``_tasks`` 已插入而 ``_retries`` 还没有），并发读会撞上 KeyError。
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.config import AppSettings
from agent.context import ContextManager, ContextWindow
from agent.enums import IntentType, OperationStatus, TaskStatus
from agent.errors import AgentError, TaskStateError
from agent.events import EventKind, TaskEvent, TaskListener, emit_all
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


_logger = logging.getLogger(__name__)


class _CircuitBreakError(AgentError):
    """护栏耗尽。与普通错误分开，好让编排层转 ABORTED 而不是 FAILED。"""

    code = "circuit_break"


@dataclass
class _TaskSlot:
    """一个任务的全部进程内状态：黑板 + 三个护栏计数器。

    原先是五个按 task_id 并列的字典。合成一个槽之后，"任务已注册"变成一次原子的字典
    插入——不会再出现 ``_tasks`` 有了、``_retries`` 还没有的中间态。
    """

    task: Task
    window: ContextWindow
    tool_calls: int = 0
    revalidations: int = 0
    retries: int = 0


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
        listeners: Sequence[TaskListener] = (),
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.context = context
        self.memory = memory
        self.tools = tools
        self._listeners = tuple(listeners)
        self._sleep = sleeper or time.sleep

        # 跨会话共享，一切访问都在锁内；锁内只做字典操作，不碰模型/工具/磁盘
        self._registry_lock = threading.Lock()
        self._slots: dict[str, _TaskSlot] = {}
        # session_id → 该会话的 task_id（按创建顺序）。有了它，pending_clarification
        # 只需扫本会话，既避免全表遍历，也不会再迭代一个别的会话正在插入的字典。
        self._session_tasks: dict[str, list[str]] = {}

        # 临时监听器：HTTP 层在跑一轮任务时临时订阅，SSE 推给客户端。
        # 与 _listeners 的差别：这里只对**本进程正在跑的这一轮**感兴趣，块退出就清。
        # 锁内只做列表拷贝，不调用户代码——避免订阅方慢阻塞状态机。
        self._ephemeral_lock = threading.Lock()
        self._ephemeral: list[TaskListener] = []

    # ================================================================ 任务注册表

    def _slot(self, task_id: str) -> _TaskSlot:
        with self._registry_lock:
            slot = self._slots.get(task_id)
        if slot is None:
            raise TaskStateError(f"未知任务：{task_id}")
        return slot

    # ================================================================ 对外接口

    def create_task(self, content: str, *, session_id: str | None = None) -> Task:
        """第零步：建任务、建窗、写入系统提示词与用户原文。"""
        task = Task.create(content, session_id=session_id or new_session_id())
        # 建窗要装配系统提示词，放在锁外——锁内只允许字典操作
        slot = _TaskSlot(task=task, window=self.context.create_window(task))
        with self._registry_lock:
            self._slots[task.task_id] = slot
            self._session_tasks.setdefault(task.session_id, []).append(task.task_id)
        self._emit(
            EventKind.TASK_CREATED,
            task,
            summary=f"任务已创建：{content}",
            payload={"content": content},
        )
        return task

    def task(self, task_id: str) -> Task:
        return self._slot(task_id).task

    def window(self, task_id: str) -> ContextWindow:
        return self._slot(task_id).window

    def pending_clarification(self, session_id: str) -> Task | None:
        """本会话里正等待澄清的任务（正常最多一个）。

        前端靠它判断"这一轮输入是新问题，还是上一轮追问的答案"——控制台与 HTTP 层都需要
        这个判断，所以放在内核而不是各自实现一遍。

        先在锁内把本会话的槽拷成一个列表，再到锁外读状态：锁的持有时间与本会话任务数
        成正比，且不会与别的会话的 ``create_task`` 撞上"字典在迭代中改变了大小"。
        """
        with self._registry_lock:
            slots = [
                self._slots[task_id]
                for task_id in self._session_tasks.get(session_id, ())
                if task_id in self._slots
            ]
        for slot in reversed(slots):
            if slot.task.status is TaskStatus.CLARIFYING:
                return slot.task
        return None

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
        slot = self._slot(task_id)
        task = slot.task
        if task.status is not TaskStatus.CLARIFYING:
            raise TaskStateError(f"任务 {task_id} 当前状态是 {task.status.value}，不在等待澄清")
        window = slot.window
        self.context.amend_user_request(window, supplement)
        self.context.update_summary(window, output=None)
        self._transition(task, TaskStatus.INTENTING, note="用户补充信息后重新识别意图")
        return self.run(task_id)

    def cancel(self, task_id: str, reason: str = "用户取消") -> Task:
        task = self.task(task_id)
        if task.status.is_terminal:
            return task
        self._terminate(task, TaskStatus.CANCELED, reason)
        return task

    # ================================================================ 事件

    def _emit(
        self,
        kind: EventKind,
        task: Task,
        *,
        summary: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """状态机的发事件出口：永久 listener + 临时 listener 一起通知。"""
        with self._registry_lock:
            slot = self._slots.get(task.task_id)
        event = TaskEvent(
            kind=kind,
            task_id=task.task_id,
            session_id=task.session_id,
            status=task.status,
            summary=summary,
            payload=payload or {},
            window=slot.window if slot is not None else None,
        )
        emit_all(self._listeners, event)
        # 临时 listener 在锁外调用——订阅方是 SSE 队列推送，慢的话会丢事件，
        # 但不能让订阅方阻塞状态机的推进
        with self._ephemeral_lock:
            ephemerals = list(self._ephemeral)
        for listener in ephemerals:
            try:
                listener(event)
            except Exception:
                _logger.exception("临时监听器处理 %s 失败", kind.value)

    # ================================================================ 临时订阅

    def add_ephemeral(self, listener: TaskListener) -> None:
        with self._ephemeral_lock:
            self._ephemeral.append(listener)

    def remove_ephemeral(self, listener: TaskListener) -> None:
        with self._ephemeral_lock, contextlib.suppress(ValueError):
            self._ephemeral.remove(listener)

    def _transition(self, task: Task, target: TaskStatus, *, note: str = "") -> None:
        """状态机推进的**唯一**入口：转移 + 发事件。

        全部转移点都走这里，"每次状态转移都留下一条日志与一条库记录"就由结构保证，
        而不是靠每处都记得补一行。
        """
        before = task.status
        task.transition_to(target)
        summary = f"状态转移 {before.value}→{target.value}"
        self._emit(
            EventKind.STATUS_CHANGED,
            task,
            summary=f"{summary}：{note}" if note else summary,
            payload={"from": before.value, "to": target.value, "note": note},
        )
        if target.is_terminal:
            self._emit(
                EventKind.TASK_FINISHED,
                task,
                summary=f"任务结束（{target.value}）：{task.summary.output or '无输出'}",
                payload={"output": task.summary.output, "intent": _intent_value(task)},
            )

    # ================================================================ 推进

    def _step(self, task: Task) -> None:
        window = self._slot(task.task_id).window
        try:
            if task.status is TaskStatus.CREATED:
                self._transition(task, TaskStatus.INTENTING)
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

        used = self._slot(task.task_id).retries + 1
        limit = self.settings.task.max_retry
        if used > limit:
            self._terminate(task, TaskStatus.FAILED, f"重试 {limit} 次后仍失败：{exc.message}")
            return
        self._slot(task.task_id).retries = used
        self._transition(
            task,
            TaskStatus.RETRYING,
            note=f"第 {used}/{limit} 次重试，临时错误：{exc.message}",
        )

    def _run_retry(self, task: Task, window: ContextWindow) -> None:  # noqa: ARG002 - 阶段签名一致
        """指数退避后回到执行阶段，从未完成的步骤继续。"""
        attempt = self._slot(task.task_id).retries
        delay = self.settings.task.retry_backoff_seconds * (2 ** (attempt - 1))
        # 单次退避封顶，避免 attempt 很大时 sleep 几小时把 worker 线程吊死
        cap = self.settings.task.retry_backoff_cap_seconds
        delay = min(delay, cap)
        self._sleep(delay)
        self._transition(task, TaskStatus.EXECUTING, note=f"退避 {delay}s 后重试")

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
            self._transition(task, TaskStatus.CLARIFYING, note=f"意图不明，向用户追问：{question}")
            return

        self._transition(task, TaskStatus.PLANNING, note=f"意图识别为 {decision.intent.value}")

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
        self._transition(task, TaskStatus.EXECUTING, note=f"规划出 {len(operations)} 个步骤")

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
        self._transition(task, TaskStatus.VALIDATING, note="所有步骤执行完毕")

    def _run_operation(self, task: Task, window: ContextWindow, operation: Operation) -> None:
        """单步内的「模型 ↔ 工具」循环。

        **最后一轮强制收敛**：不再把工具清单发给模型，并在指令里明说"这是最后一轮，
        基于已有结果直接给结论"。否则模型很容易一路换着关键字试到轮次耗尽，明明手上
        已经有数据，任务却熔断成 ABORTED、用户什么都拿不到。
        """
        max_rounds = self.settings.task.max_rounds_per_step
        for round_index in range(max_rounds):
            final_round = round_index == max_rounds - 1
            messages = self.context.build_execute_messages(
                window, operation, final_round=final_round
            )
            response = self.llm.chat(
                LLMRequest(
                    messages=messages,
                    tools=None if final_round else self.tools.openai_schemas(),
                    stage="execute",
                )
            )
            if final_round and response.wants_tool:
                # 没给工具还硬要调，说明模型不配合；有正文就当结论，否则只能熔断
                if not response.content:
                    break
                response = response.model_copy(update={"tool_calls": []})

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

        self._emit(
            EventKind.TOOL_CALLED,
            task,
            summary=f"调用 {call.name} → {'ok' if result.ok else '失败'}：{result.summary}",
            payload={
                "call_id": call.id,
                "tool_name": call.name,
                "arguments": call.arguments,
                "ok": result.ok,
                "internal": self.tools.is_internal(call.name),
                # 内部工具不落黑板，落库方要拿全量只能从这里取
                "result": result.data if result.ok else None,
                "error": result.error,
            },
        )

    def _spend_tool_call(self, task: Task) -> None:
        slot = self._slot(task.task_id)
        used = slot.tool_calls + 1
        limit = self.settings.task.max_tool_calls
        if used > limit:
            raise _CircuitBreakError(f"工具调用次数超过上限 {limit}")
        slot.tool_calls = used

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
            self._transition(task, TaskStatus.COMPLETED, note="校验通过")
            return

        used = self._slot(task.task_id).revalidations + 1
        limit = self.settings.task.max_revalidate
        if used > limit:
            raise _CircuitBreakError(f"校验连续 {limit} 次未通过：{decision.reason}")
        self._slot(task.task_id).revalidations = used

        # 补充执行：追加一个新步骤，执行循环才有事可做
        operations = [
            *window.summary.operations,
            Operation(
                index=len(window.summary.operations),
                description=f"根据校验反馈补充执行：{decision.reason}",
            ),
        ]
        self.context.update_summary(window, operations=operations)
        self._transition(
            task, TaskStatus.EXECUTING, note=f"校验未通过，补充执行：{decision.reason}"
        )

    # ---------------------------------------------------------------- 终态

    def _terminate(self, task: Task, target: TaskStatus, reason: str) -> None:
        """转终态并把原因写进 output——控制台就是靠它告诉用户发生了什么。

        终态是按转移表挑的，不是想落哪个就落哪个：

        - ``created`` 只能去 ``intenting`` 或 ``canceled``，所以失败前先推进一格；
        - ``planning`` 没有到 ``aborted`` 的边（require.md 的表里只有 executing /
          waiting / retrying / paused / validating 能熔断），此时退回 ``failed``。

        宁可换一个语义相近的终态，也不绕过状态机——转移表是这个系统的主干约束。

        写入顺序：**先把 output/error 准备好，再落状态**。终态转移会同时发出
        TASK_FINISHED 事件，日志与落库都从事件上读 output；若先转移，它们看到的就是
        一个还没填的 None。
        """
        window = self._slot(task.task_id).window
        if task.status is TaskStatus.CREATED and target is not TaskStatus.CANCELED:
            self._transition(task, TaskStatus.INTENTING)
        if not task.status.can_transition_to(target):
            target = TaskStatus.FAILED

        task.error = {"status": target.value, "reason": reason}
        self.context.update_summary(window, output=f"[{target.value}] {reason}")
        self._transition(task, target, note=reason)
        self.memory.archive(window)


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _intent_value(task: Task) -> str | None:
    return task.summary.intent.value if task.summary.intent else None


def build_task_manager(
    settings: AppSettings,
    *,
    llm: BaseLLMClient | None = None,
    listeners: Sequence[TaskListener] = (),
    sleeper: Callable[[float], None] | None = None,
) -> TaskManager:
    """按配置装配一整套依赖。控制台与 Django 层都从这里拿 TaskManager。"""
    tools = build_tool_manager(settings)
    knowledge = KnowledgeMemory(tools.catalog())
    return TaskManager(
        settings=settings,
        llm=llm or build_llm_client(settings),
        context=ContextManager(settings, knowledge),
        memory=MemoryManager(knowledge, max_tasks=settings.memory.max_tasks),
        tools=tools,
        listeners=listeners,
        sleeper=sleeper,
    )
