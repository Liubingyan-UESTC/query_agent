"""黑板与阶段消息装配测试。"""

import json

import pytest

from agent.context import ContextManager, ContextWindow, ToolResultRecord
from agent.enums import IntentType, MessageRole, OperationStatus, TaskStatus
from agent.knowledge import KnowledgeMemory
from agent.models import Message, Operation, Task, TaskSummary, ToolCall
from agent.tools.base import ToolResult


@pytest.fixture
def knowledge():
    return KnowledgeMemory("- search_tool：按关键字检索\n- analysis_tool：分组统计")


@pytest.fixture
def manager(settings, knowledge):
    return ContextManager(settings, knowledge)


@pytest.fixture
def task():
    return Task.create("查一下 order-service 的错误日志", session_id="sess_1")


@pytest.fixture
def window(manager, task):
    return manager.create_window(task)


def search_result(count: int = 2) -> ToolResult:
    records = [{"level": "ERROR", "service": "order-service", "seq": i} for i in range(count)]
    return ToolResult.success(
        {"index": "logs", "total": count, "returned": count, "records": records},
        f"关键字 ERROR 命中 {count} 条",
    )


class TestBlackboardShape:
    def test_window_holds_exactly_three_records(self):
        """require.md 的黑板只有三项：summary / tool_result / task_content。"""
        assert set(ContextWindow.model_fields) == {
            "task_id",
            "session_id",
            "summary",
            "tool_results",
            "content",
        }

    def test_tool_result_record_shape(self):
        record = ToolResultRecord(tool_name="search_tool", tool_result={"total": 1})
        assert record.to_dict() == {
            "tool_name": "search_tool",
            "tool_result": {"total": 1},
            "ok": True,
        }

    def test_window_shares_summary_with_task(self, manager, task):
        """共用同一个 summary 对象，状态推进后窗口里看到的也是新值。"""
        window = manager.create_window(task)
        task.transition_to(TaskStatus.INTENTING)
        assert window.summary.status is TaskStatus.INTENTING

    def test_create_window_seeds_system_and_user(self, window):
        assert window.content[0].role is MessageRole.SYSTEM
        assert window.content[1].role is MessageRole.USER
        assert window.content[1].content == "查一下 order-service 的错误日志"

    def test_window_serializes_roundtrip(self, manager, window):
        manager.record_tool_result(
            window, tool_call_id="c1", tool_name="search_tool", result=search_result()
        )
        restored = ContextWindow.from_dict(json.loads(json.dumps(window.to_dict())))
        assert restored.tool_results["c1"].tool_name == "search_tool"
        assert len(restored.content) == len(window.content)


class TestSummaryWrites:
    def test_whitelisted_fields_are_writable(self, manager, window):
        manager.update_summary(window, intent=IntentType.QUERY, related_task_ids=["t0"])
        assert window.summary.intent is IntentType.QUERY
        assert window.summary.related_task_ids == ["t0"]

    @pytest.mark.parametrize("field", ["status", "task_id", "content"])
    def test_protected_fields_are_rejected(self, manager, window, field):
        """状态只能走状态机；task_id/content 是任务身份，改了无从追溯。"""
        with pytest.raises(ValueError, match="不允许在此写入"):
            manager.update_summary(window, **{field: "x"})

    def test_unknown_field_is_rejected(self, manager, window):
        with pytest.raises(ValueError, match="不允许在此写入"):
            manager.update_summary(window, nonexistent=1)


class TestToolResultBlackboard:
    def test_full_result_goes_to_the_blackboard(self, manager, window):
        manager.record_tool_result(
            window, tool_call_id="c1", tool_name="search_tool", result=search_result(5)
        )
        record = window.tool_results["c1"]
        assert record.tool_name == "search_tool"
        assert len(record.tool_result["records"]) == 5

    def test_preview_carries_id_summary_and_samples(self, manager, window, settings):
        preview = manager.record_tool_result(
            window, tool_call_id="c1", tool_name="search_tool", result=search_result(10)
        )
        assert "[tool_call_id=c1]" in preview
        assert "命中 10 条" in preview
        # 只给前 N 条样本，不是全量
        assert preview.count('"seq"') == settings.tool.preview_max_rows
        assert "fetch_tool_result" in preview

    def test_preview_is_much_smaller_than_full_result(self, manager, window):
        result = search_result(200)
        preview = manager.record_tool_result(
            window, tool_call_id="c1", tool_name="search_tool", result=result
        )
        assert len(preview) < len(json.dumps(result.data, ensure_ascii=False)) / 5

    def test_preview_respects_char_cap(self, log_file):
        from agent.config import load_settings

        settings = load_settings(
            env_file=None,
            tool={"data_file": log_file, "preview_max_chars": 120, "preview_max_rows": 50},
        )
        manager = ContextManager(settings, KnowledgeMemory())
        preview = manager.build_preview("c1", search_result(50))
        assert len(preview) <= 120 + 40  # 上限 + 截断提示
        assert "已截断" in preview

    def test_failed_result_is_recorded_with_ok_false(self, manager, window):
        preview = manager.record_tool_result(
            window,
            tool_call_id="c1",
            tool_name="analysis_tool",
            result=ToolResult.failure("字段不存在"),
        )
        assert window.tool_results["c1"].ok is False
        assert "调用失败" in preview

    def test_read_back_returns_full_payload(self, manager, window):
        manager.record_tool_result(
            window, tool_call_id="c1", tool_name="search_tool", result=search_result(7)
        )
        payload = manager.read_tool_result(window, "c1")
        assert len(payload["records"]) == 7

    def test_read_missing_id_returns_none(self, manager, window):
        assert manager.read_tool_result(window, "ghost") is None

    def test_analysis_groups_are_previewed_too(self, manager, window):
        result = ToolResult.success(
            {"group_by": "service", "groups": [{"key": "order-service", "value": 12}]},
            "按 service 统计 count",
        )
        preview = manager.record_tool_result(
            window, tool_call_id="c2", tool_name="analysis_tool", result=result
        )
        assert "order-service" in preview


class TestRecentContent:
    def test_keeps_tool_calls_paired_with_their_answers(self, manager, window):
        """切片不能把 tool 消息与它的 assistant 调用切散，否则端点会 400。"""
        call = ToolCall(id="c1", name="search_tool")
        manager.append_message(window, Message.assistant(tool_calls=[call]))
        manager.append_message(window, Message.tool("结果", tool_call_id="c1"))
        manager.append_message(window, Message.assistant("结论"))

        clipped = window.recent_content(2)
        assert all(
            m.role is not MessageRole.TOOL or m.tool_call_id == "c1"
            for m in clipped
            if m.role is MessageRole.TOOL
        )
        # 被截断到只剩 tool + assistant 时，孤儿 tool 消息必须被丢掉
        assert [m.role for m in clipped] == [MessageRole.ASSISTANT]

    def test_drops_assistant_whose_tool_call_was_never_answered(self, manager, window):
        """工具预算在半途耗尽时会留下没有应答的 tool_call，这条消息不能进下一次请求。"""
        manager.append_message(
            window, Message.assistant(tool_calls=[ToolCall(id="c9", name="search_tool")])
        )
        assert all(not m.tool_calls for m in window.recent_content(10))

    def test_keeps_assistant_whose_tool_call_was_answered(self, manager, window):
        call = ToolCall(id="c1", name="search_tool")
        manager.append_message(window, Message.assistant(tool_calls=[call]))
        manager.append_message(window, Message.tool("结果", tool_call_id="c1"))
        kept = window.recent_content(10)
        assert [m.role for m in kept][-2:] == [MessageRole.ASSISTANT, MessageRole.TOOL]

    def test_system_prompt_is_not_part_of_recent_content(self, manager, window):
        assert all(m.role is not MessageRole.SYSTEM for m in window.recent_content(10))

    def test_limit_zero_returns_nothing(self, window):
        assert window.recent_content(0) == []


class TestStageMessages:
    def test_intent_only_feeds_summary_and_history(self, manager, window):
        history = [{"task_id": "t0", "content": "上一个任务", "status": "completed"}]
        messages = manager.build_intent_messages(window, history)

        assert [m.role for m in messages] == [MessageRole.SYSTEM, MessageRole.USER]
        payload = json.loads(messages[1].content)
        assert payload["current_task"] == {
            "task_id": window.task_id,
            "content": "查一下 order-service 的错误日志",
            "status": "created",
        }
        assert payload["history"] == history

    def test_intent_history_is_capped(self, log_file):
        from agent.config import load_settings

        settings = load_settings(
            env_file=None, tool={"data_file": log_file}, context={"history_summary_limit": 2}
        )
        manager = ContextManager(settings, KnowledgeMemory())
        task = Task.create("q", session_id="s")
        window = manager.create_window(task)
        history = [{"task_id": f"t{i}", "content": str(i)} for i in range(5)]
        payload = json.loads(manager.build_intent_messages(window, history)[1].content)
        assert [item["task_id"] for item in payload["history"]] == ["t3", "t4"]

    def test_plan_prompt_switches_on_intent(self, manager, window):
        manager.update_summary(window, intent=IntentType.ANALYSIS)
        system = manager.build_plan_messages(window)[0].content
        assert "analysis_tool" in system
        assert "tool_call_id" in system

    def test_plan_injects_related_context_as_user_text(self, manager, window):
        related = {
            "task_0": [
                Message.user("查错误日志"),
                Message.assistant("共 12 条 ERROR"),
            ]
        }
        messages = manager.build_plan_messages(window, related)
        joined = "\n".join(m.content for m in messages)
        assert "[关联任务 task_0]" in joined
        assert "共 12 条 ERROR" in joined
        # 注入的是文本，不是伪造的历史 assistant 消息
        assert all(m.role in (MessageRole.SYSTEM, MessageRole.USER) for m in messages)

    def test_plan_without_related_has_no_injection_block(self, manager, window):
        messages = manager.build_plan_messages(window, {})
        assert len(messages) == 2

    def test_execute_instruction_names_step_and_tool(self, manager, window):
        manager.update_summary(
            window,
            intent=IntentType.QUERY,
            operations=[
                Operation(index=0, description="检索错误日志", suggested_tool="search_tool"),
                Operation(index=1, description="统计", suggested_tool="analysis_tool"),
            ],
        )
        messages = manager.build_execute_messages(window, window.summary.operations[0])
        instruction = messages[-1].content
        assert "当前步骤 1/2：检索错误日志" in instruction
        assert "建议工具：search_tool" in instruction
        assert "查一下 order-service 的错误日志" in instruction

    def test_execute_marks_toolless_step(self, manager, window):
        manager.update_summary(
            window, operations=[Operation(index=0, description="直接回答", suggested_tool=None)]
        )
        instruction = manager.build_execute_messages(window, window.summary.operations[0])[-1]
        assert "建议工具：无" in instruction.content

    def test_execute_includes_prior_conversation(self, manager, window):
        manager.update_summary(
            window,
            operations=[Operation(index=0, description="检索", suggested_tool="search_tool")],
        )
        manager.append_message(window, Message.assistant("上一步已完成检索"))
        messages = manager.build_execute_messages(window, window.summary.operations[0])
        joined = [m.content for m in messages]
        assert any("上一步已完成检索" in text for text in joined)

    def test_validate_feeds_summary_and_tool_result_index(self, manager, window):
        manager.update_summary(
            window,
            intent=IntentType.QUERY,
            operations=[
                Operation(
                    index=0,
                    description="检索",
                    status=OperationStatus.SUCCEEDED,
                    result="命中 12 条",
                )
            ],
        )
        manager.record_tool_result(
            window, tool_call_id="c1", tool_name="search_tool", result=search_result()
        )
        payload = json.loads(manager.build_validate_messages(window)[1].content)

        assert payload["user_request"] == "查一下 order-service 的错误日志"
        assert payload["operations"][0]["status"] == "succeeded"
        assert payload["tool_results"] == [
            {"tool_call_id": "c1", "tool_name": "search_tool", "ok": True}
        ]
        # 校验阶段看的是索引而非全量日志，避免把整批数据再塞一遍
        assert "records" not in json.dumps(payload)

    @pytest.mark.parametrize(
        "builder",
        ["build_intent_messages", "build_plan_messages", "build_validate_messages"],
    )
    def test_every_stage_starts_with_exactly_one_system_message(self, manager, window, builder):
        args = ([],) if builder == "build_intent_messages" else ()
        messages = getattr(manager, builder)(window, *args)
        assert messages[0].role is MessageRole.SYSTEM
        assert sum(m.role is MessageRole.SYSTEM for m in messages) == 1


def test_summary_prompt_fields_do_not_leak_internals():
    """意图阶段的投影只给三个字段，新增内部字段不会自动泄漏给模型。"""
    summary = TaskSummary(task_id="t", content="c", result="内部结果")
    assert "result" not in summary.to_prompt_dict(TaskSummary.INTENT_PROMPT_FIELDS)
