"""编排测试：五个阶段、状态流转与各条异常分支。"""

import json

import pytest

from agent.config import load_settings
from agent.enums import IntentType, MessageRole, OperationStatus, TaskStatus
from agent.errors import LLMError, TaskStateError
from agent.llm import MockLLMClient
from agent.task_manager import build_task_manager


def intent(kind="query", related=None, clarification=""):
    return MockLLMClient.json_reply(
        {
            "intent": kind,
            "related_task_ids": related or [],
            "reason": "test",
            "clarification": clarification,
        }
    )


def plan(*steps):
    operations = [{"description": d, "suggested_tool": t} for d, t in steps]
    return MockLLMClient.json_reply({"operations": operations})


def validate(satisfied=True, output="共 3 条 ERROR 日志", reason=""):
    return MockLLMClient.json_reply({"satisfied": satisfied, "output": output, "reason": reason})


def conclusion(text="这一步完成了"):
    return MockLLMClient.reply(text)


def tool_call(name, call_id="call_1", **arguments):
    return MockLLMClient.tool_reply(name, arguments, call_id=call_id)


@pytest.fixture
def make_manager(settings):
    def _make(script=None, *, sleeper=None, **task_overrides):
        used = settings
        if task_overrides:
            used = load_settings(
                env_file=None,
                llm={"use_mock": True},
                tool={"data_file": settings.tool.data_file},
                task=task_overrides,
            )
        # 默认吞掉退避，测试不真的 sleep
        return build_task_manager(
            used, llm=MockLLMClient(script or []), sleeper=sleeper or (lambda _seconds: None)
        )

    return _make


HAPPY_QUERY = [
    intent("query"),
    plan(("按关键字检索错误日志", "search_tool")),
    tool_call("search_tool", keyword="ERROR"),
    conclusion("检索到 3 条 ERROR 日志"),
    validate(),
]


class TestLifecycle:
    def test_create_task_seeds_window(self, make_manager):
        manager = make_manager()
        task = manager.create_task("查错误日志", session_id="s1")

        assert task.status is TaskStatus.CREATED
        window = manager.window(task.task_id)
        assert window.content[0].role is MessageRole.SYSTEM
        assert window.summary.content == "查错误日志"

    def test_happy_path_reaches_completed(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.output == "共 3 条 ERROR 日志"
        assert task.summary.intent is IntentType.QUERY

    def test_status_history_follows_require_md(self, make_manager):
        """借状态机的合法性校验反证：主线顺序与 require.md 一致。"""
        manager = make_manager(HAPPY_QUERY)
        task = manager.create_task("查错误日志", session_id="s1")
        manager.run(task.task_id)
        assert task.summary.status is TaskStatus.COMPLETED

    def test_unknown_task_id_rejected(self, make_manager):
        with pytest.raises(TaskStateError, match="未知任务"):
            make_manager().run("task_ghost")


class TestIntentStage:
    def test_writes_intent_and_related_ids(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        first = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        manager.llm.enqueue(
            intent("analysis", related=[first.task_id]),
            plan(("统计", None)),
            conclusion("统计完成"),
            validate(output="order-service 最多"),
        )
        second = manager.run(manager.create_task("统计刚才的结果", session_id="s1").task_id)
        assert second.summary.related_task_ids == [first.task_id]

    def test_hallucinated_related_ids_are_dropped(self, make_manager):
        """模型编造的 task_id 不能进 summary，否则后续注入会取到空。"""
        manager = make_manager([intent("query", related=["task_does_not_exist"]), *HAPPY_QUERY[1:]])
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        assert task.summary.related_task_ids == []

    def test_intent_stage_feeds_history_summaries(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        first = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        manager.llm.enqueue(*HAPPY_QUERY)
        manager.run(manager.create_task("再查一次", session_id="s1").task_id)

        intent_requests = [c for c in manager.llm.calls if c.stage == "intent"]
        payload = json.loads(intent_requests[-1].messages[-1].content)
        assert [item["task_id"] for item in payload["history"]] == [first.task_id]

    def test_unknown_intent_pauses_for_clarification(self, make_manager):
        manager = make_manager([intent("unknown", clarification="你想查哪个服务？")])
        task = manager.run(manager.create_task("嗯", session_id="s1").task_id)

        assert task.status is TaskStatus.CLARIFYING
        assert task.status.requires_user_input
        assert task.summary.output == "你想查哪个服务？"

    def test_clarification_resumes_the_same_task(self, make_manager):
        manager = make_manager([intent("unknown", clarification="你想查哪个服务？")])
        task = manager.run(manager.create_task("嗯", session_id="s1").task_id)

        manager.llm.enqueue(*HAPPY_QUERY)
        resumed = manager.provide_clarification(task.task_id, "order-service 的错误")

        assert resumed.task_id == task.task_id
        assert resumed.status is TaskStatus.COMPLETED
        assert "补充说明：order-service 的错误" in resumed.summary.content

    def test_clarification_rejected_when_not_clarifying(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        with pytest.raises(TaskStateError, match="不在等待澄清"):
            manager.provide_clarification(task.task_id, "补充")

    def test_unparseable_intent_output_fails_the_task(self, make_manager):
        """两次都不是 JSON —— call_structured 用尽修复机会后判定不可恢复。"""
        manager = make_manager([MockLLMClient.reply("我觉得"), MockLLMClient.reply("还是不知道")])
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert task.status is TaskStatus.FAILED
        assert "failed" in task.summary.output


class TestPlanStage:
    def test_operations_are_indexed_from_zero(self, make_manager):
        manager = make_manager(
            [
                intent("analysis"),
                plan(("检索", "search_tool"), ("统计", "analysis_tool")),
                tool_call("search_tool", keyword="ERROR"),
                conclusion("检索完成"),
                tool_call(
                    "analysis_tool", call_id="call_2", tool_call_id="call_1", group_by="service"
                ),
                conclusion("统计完成"),
                validate(),
            ]
        )
        task = manager.run(manager.create_task("统计错误分布", session_id="s1").task_id)
        assert [op.index for op in task.summary.operations] == [0, 1]
        assert all(op.status is OperationStatus.SUCCEEDED for op in task.summary.operations)

    def test_unknown_suggested_tool_is_cleared(self, make_manager):
        """规划写了不存在的工具就清空，执行阶段模型仍可自行选对的那个。"""
        manager = make_manager(
            [intent("query"), plan(("检索", "kibana_query")), conclusion("完成"), validate()]
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)
        assert task.summary.operations[0].suggested_tool is None

    def test_empty_plan_falls_back_to_direct_answer(self, make_manager):
        manager = make_manager([intent("chat"), plan(), conclusion("我是日志助手"), validate()])
        task = manager.run(manager.create_task("你是谁", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert len(task.summary.operations) == 1

    def test_too_many_steps_trips_the_circuit_breaker(self, make_manager):
        """规划阶段熔断落 FAILED：require.md 的转移表里 planning 没有到 aborted 的边。"""
        manager = make_manager(
            [intent("query"), plan(*[(f"第 {i} 步", None) for i in range(5)])],
            max_steps=3,
        )
        task = manager.run(manager.create_task("复杂请求", session_id="s1").task_id)

        assert task.status is TaskStatus.FAILED
        assert "超过上限 3" in task.summary.output

    def test_related_context_is_injected_into_planning(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        first = manager.run(manager.create_task("查 order-service 错误", session_id="s1").task_id)

        manager.llm.enqueue(
            intent("analysis", related=[first.task_id]),
            plan(("统计", None)),
            conclusion("统计完成"),
            validate(),
        )
        manager.run(manager.create_task("统计刚才的结果", session_id="s1").task_id)

        plan_request = [c for c in manager.llm.calls if c.stage == "plan"][-1]
        joined = "\n".join(m.content for m in plan_request.messages)
        assert f"[关联任务 {first.task_id}]" in joined


class TestExecuteStage:
    def test_tool_result_goes_to_blackboard_and_preview_to_model(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        window = manager.window(task.task_id)

        record = window.tool_results["call_1"]
        assert record.tool_name == "search_tool"
        assert record.ok is True
        assert len(record.tool_result["records"]) == 3  # fixture 里 3 条 ERROR

        tool_message = next(m for m in window.content if m.role is MessageRole.TOOL)
        assert "[tool_call_id=call_1]" in tool_message.content
        assert "fetch_tool_result" in tool_message.content

    def test_assistant_tool_call_is_paired_with_a_tool_message(self, make_manager):
        """OpenAI 协议要求成对，缺一条下一轮就会 400。"""
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        window = manager.window(task.task_id)

        requested = {
            call.id
            for m in window.content
            if m.role is MessageRole.ASSISTANT
            for call in m.tool_calls
        }
        answered = {m.tool_call_id for m in window.content if m.role is MessageRole.TOOL}
        assert requested == answered

    def test_fetch_tool_result_is_not_written_back_to_blackboard(self, make_manager):
        """require.md 的硬性要求：特殊工具的结果直接回模型，不入 tool_result。"""
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                tool_call("search_tool", keyword="ERROR"),
                tool_call("fetch_tool_result", call_id="call_fetch", tool_call_id="call_1"),
                conclusion("已看到全量数据"),
                validate(),
            ]
        )
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        window = manager.window(task.task_id)

        assert "call_1" in window.tool_results
        assert "call_fetch" not in window.tool_results
        # 但对话里仍有一条应答消息，协议才闭合
        fetch_message = next(m for m in window.content if m.tool_call_id == "call_fetch")
        assert "Read timed out" in fetch_message.content  # 全量内容，不是预览

    def test_analysis_reads_previous_result_from_blackboard(self, make_manager):
        manager = make_manager(
            [
                intent("analysis"),
                plan(("检索", "search_tool"), ("统计", "analysis_tool")),
                tool_call("search_tool", keyword=""),
                conclusion("检索完成"),
                tool_call(
                    "analysis_tool", call_id="call_2", tool_call_id="call_1", group_by="service"
                ),
                conclusion("统计完成"),
                validate(),
            ]
        )
        task = manager.run(manager.create_task("按服务统计", session_id="s1").task_id)
        groups = manager.window(task.task_id).tool_results["call_2"].tool_result["groups"]
        assert {"key": "order-service", "value": 2} in groups

    def test_failed_tool_is_recoverable_within_the_step(self, make_manager):
        """工具报错后模型改参数重试，整个任务照样完成。"""
        manager = make_manager(
            [
                intent("analysis"),
                plan(("统计", "analysis_tool")),
                tool_call("analysis_tool", tool_call_id="nonexistent", group_by="service"),
                tool_call("search_tool", call_id="call_2", keyword="ERROR"),
                conclusion("换了个思路完成了"),
                validate(),
            ]
        )
        task = manager.run(manager.create_task("统计", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert manager.window(task.task_id).tool_results["call_1"].ok is False

    def test_unknown_tool_fails_the_task(self, make_manager):
        manager = make_manager(
            [intent("query"), plan(("检索", None)), tool_call("kibana_search", keyword="x")]
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)

        assert task.status is TaskStatus.FAILED
        assert "未注册的工具" in task.summary.output

    def test_final_round_forces_a_conclusion_instead_of_aborting(self, make_manager):
        """模型一路换关键字试到最后一轮时，应当用手上的数据作答，而不是熔断。"""
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                tool_call("search_tool", call_id="c1", keyword="order-service"),
                tool_call("search_tool", call_id="c2", keyword="ERROR"),
                # 第 3 轮是最后一轮：编排层不再发工具清单
                MockLLMClient.reply("共 2 条 order-service 的 ERROR 日志"),
                validate(output="共 2 条"),
            ],
            max_rounds_per_step=3,
        )
        task = manager.run(manager.create_task("查 order-service 的错误", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.operations[0].result == "共 2 条 order-service 的 ERROR 日志"

    def test_final_round_carries_no_tools(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                tool_call("search_tool", call_id="c1", keyword="ERROR"),
                MockLLMClient.reply("结论"),
                validate(),
            ],
            max_rounds_per_step=2,
        )
        manager.run(manager.create_task("查日志", session_id="s1").task_id)

        execute_calls = [c for c in manager.llm.calls if c.stage == "execute"]
        assert execute_calls[0].tools  # 前面的轮次照常给工具
        assert execute_calls[-1].tools is None  # 最后一轮不给
        assert "不要再调用工具" in execute_calls[-1].messages[-1].content

    def test_step_round_budget_aborts(self, make_manager):
        """连最后一轮的收敛指令都不听（还在硬调工具且没有正文）→ 熔断。"""
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                *[tool_call("search_tool", call_id=f"c{i}", keyword="ERROR") for i in range(5)],
            ],
            max_rounds_per_step=2,
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)

        assert task.status is TaskStatus.ABORTED
        assert "2 轮内没有得出结论" in task.summary.output

    def test_tool_call_budget_aborts(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                *[tool_call("search_tool", call_id=f"c{i}", keyword="ERROR") for i in range(6)],
            ],
            max_tool_calls=2,
            max_rounds_per_step=6,
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)

        assert task.status is TaskStatus.ABORTED
        assert "工具调用次数超过上限 2" in task.summary.output

    def test_result_is_filled_before_validation(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        assert "步骤1：检索到 3 条 ERROR 日志" in task.summary.result

    def test_chat_intent_needs_no_tool(self, make_manager):
        manager = make_manager(
            [
                intent("chat"),
                plan(("直接回答", None)),
                conclusion("我是日志查询助手"),
                validate(output="我是日志查询助手"),
            ]
        )
        task = manager.run(manager.create_task("你是谁", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert manager.window(task.task_id).tool_results == {}


class TestValidateStage:
    def test_unsatisfied_goes_back_to_executing(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                tool_call("search_tool", keyword="ERROR"),
                conclusion("检索完成"),
                validate(satisfied=False, reason="还没有给出具体数量"),
                conclusion("共 3 条"),
                validate(output="共 3 条 ERROR"),
            ]
        )
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.output == "共 3 条 ERROR"
        # 补充执行会追加一个新步骤
        assert len(task.summary.operations) == 2
        assert "补充执行" in task.summary.operations[1].description

    def test_repeated_failure_aborts(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", None)),
                conclusion("完成"),
                validate(satisfied=False, reason="不够具体"),
                conclusion("再试"),
                validate(satisfied=False, reason="还是不够"),
            ],
            max_revalidate=1,
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)

        assert task.status is TaskStatus.ABORTED
        assert "校验连续 1 次未通过" in task.summary.output

    def test_output_falls_back_to_result(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", None)),
                conclusion("检索到 3 条"),
                validate(output=""),
            ]
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)
        assert "检索到 3 条" in task.summary.output


class TestTerminalStates:
    def test_cancel_writes_the_reason(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.create_task("查错误日志", session_id="s1")
        manager.cancel(task.task_id, "用户按了 Ctrl-C")

        assert task.status is TaskStatus.CANCELED
        assert "用户按了 Ctrl-C" in task.summary.output

    def test_cancel_is_idempotent(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        assert manager.cancel(task.task_id).status is TaskStatus.COMPLETED

    def test_terminal_task_is_archived_into_working_memory(self, make_manager):
        manager = make_manager(HAPPY_QUERY)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        working = manager.memory.working("s1")
        assert working.task_ids() == [task.task_id]
        assert working.task_summary_history[task.task_id]["status"] == "completed"
        assert working.tool_result_history[task.task_id]["call_1"]["tool_name"] == "search_tool"
        assert working.task_context_history[task.task_id]

    def test_failed_task_is_archived_too(self, make_manager):
        manager = make_manager([MockLLMClient.reply("x"), MockLLMClient.reply("y")])
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)

        assert task.status is TaskStatus.FAILED
        assert manager.memory.working("s1").task_ids() == [task.task_id]

    def test_error_detail_is_kept_on_the_task(self, make_manager):
        manager = make_manager([MockLLMClient.reply("x"), MockLLMClient.reply("y")])
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)
        assert task.error["status"] == "failed"
        assert task.error["reason"]


class TestRetrying:
    """require.md 的 RETRYING 分支：执行阶段的临时错误退避后重试，而不是直接判死。"""

    def test_transient_error_retries_and_succeeds(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                LLMError("429 限流", retryable=True),
                tool_call("search_tool", keyword="ERROR"),
                conclusion("检索到 3 条"),
                validate(),
            ],
            max_retry=1,
        )
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        assert task.status is TaskStatus.COMPLETED

    def test_retry_budget_exhausted_fails(self, make_manager):
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                LLMError("429 一直限流", retryable=True),
                LLMError("429 一直限流", retryable=True),
            ],
            max_retry=1,
        )
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert task.status is TaskStatus.FAILED
        assert "重试 1 次后仍失败" in task.summary.output

    def test_backoff_is_exponential(self, make_manager):
        delays: list[float] = []
        manager = make_manager(
            [
                intent("query"),
                plan(("检索", "search_tool")),
                LLMError("429", retryable=True),
                LLMError("429", retryable=True),
                tool_call("search_tool", keyword="ERROR"),
                conclusion("好了"),
                validate(),
            ],
            sleeper=delays.append,
            max_retry=2,
            retry_backoff_seconds=0.5,
        )
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert delays == [0.5, 1.0]

    def test_non_retryable_error_skips_retrying(self, make_manager):
        """未注册的工具换个时间点调还是不存在，不该浪费重试。"""
        manager = make_manager(
            [intent("query"), plan(("检索", None)), tool_call("ghost_tool")],
            max_retry=3,
        )
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)
        assert task.status is TaskStatus.FAILED
        assert "未注册的工具" in task.summary.output

    def test_transient_error_outside_executing_fails_directly(self, make_manager):
        """转移表里只有 executing → retrying 这一条边，意图阶段的限流只能判失败。"""
        manager = make_manager([LLMError("429 限流", retryable=True)], max_retry=3)
        task = manager.run(manager.create_task("查日志", session_id="s1").task_id)

        assert task.status is TaskStatus.FAILED
        assert "429" in task.summary.output

    def test_retry_resumes_the_unfinished_step(self, make_manager):
        """重试是从没做完的步骤继续，已成功的步骤不会重来。"""
        manager = make_manager(
            [
                intent("analysis"),
                plan(("检索", "search_tool"), ("统计", "analysis_tool")),
                tool_call("search_tool", keyword="ERROR"),
                conclusion("检索完成"),
                LLMError("429", retryable=True),
                tool_call(
                    "analysis_tool", call_id="call_2", tool_call_id="call_1", group_by="service"
                ),
                conclusion("统计完成"),
                validate(),
            ],
            max_retry=1,
        )
        task = manager.run(manager.create_task("统计错误", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.operations[0].result == "检索完成"
        # 第一步只被执行过一次：黑板上只有一次 search 结果
        searches = [
            r
            for r in manager.window(task.task_id).tool_results.values()
            if r.tool_name == "search_tool"
        ]
        assert len(searches) == 1
