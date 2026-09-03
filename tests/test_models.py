"""数据模型测试：黑板上流动结构的不变量。"""

import json

import pytest
from pydantic import ValidationError

from agent.enums import IntentType, MessageRole, OperationStatus, TaskStatus
from agent.errors import TaskStateError
from agent.models import Message, Operation, Task, TaskSummary, ToolCall


class TestToolCall:
    def test_arguments_stay_a_json_string(self):
        """保留原始字符串，"模型吐了非法 JSON"才可复现。"""
        call = ToolCall(name="search_tool", arguments='{"keyword": "timeout"}')
        assert isinstance(call.arguments, str)
        assert call.parsed_arguments() == {"keyword": "timeout"}

    def test_invalid_json_raises_with_tool_name(self):
        call = ToolCall(name="search_tool", arguments="{keyword: timeout}")
        with pytest.raises(ValueError, match="search_tool 的 arguments 不是合法 JSON"):
            call.parsed_arguments()

    def test_non_object_arguments_rejected(self):
        with pytest.raises(ValueError, match="必须是 JSON 对象"):
            ToolCall(name="t", arguments="[1, 2]").parsed_arguments()

    def test_llm_dict_roundtrip(self):
        call = ToolCall(id="call_1", name="search_tool", arguments='{"keyword":"x"}')
        assert ToolCall.from_llm_dict(call.to_llm_dict()) == call

    def test_from_llm_dict_reports_missing_field(self):
        with pytest.raises(ValueError, match="缺少必需字段 'id'"):
            ToolCall.from_llm_dict({"function": {"name": "t", "arguments": "{}"}})

    def test_from_llm_dict_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="不支持的 tool_call 类型"):
            ToolCall.from_llm_dict({"id": "1", "type": "code", "function": {"name": "t"}})

    def test_null_arguments_fall_back_to_empty_object(self):
        """部分端点在无参调用时回 arguments=null。"""
        call = ToolCall.from_llm_dict(
            {"id": "c1", "function": {"name": "t", "arguments": None}},
        )
        assert call.parsed_arguments() == {}


class TestMessage:
    def test_to_llm_dict_only_exposes_protocol_fields(self):
        """message_id / created_at 这些编排字段绝不能进模型载荷。"""
        payload = Message.user("查一下错误日志").to_llm_dict()
        assert payload == {"role": "user", "content": "查一下错误日志"}
        assert "message_id" not in payload
        assert "created_at" not in payload

    def test_assistant_with_tool_calls(self):
        call = ToolCall(id="call_1", name="search_tool")
        payload = Message.assistant(tool_calls=[call]).to_llm_dict()
        assert payload["tool_calls"] == [call.to_llm_dict()]

    def test_tool_calls_only_on_assistant(self):
        with pytest.raises(ValidationError, match="只有 assistant 消息可携带 tool_calls"):
            Message.user("x", tool_calls=[ToolCall(name="t")])

    def test_tool_message_requires_tool_call_id(self):
        with pytest.raises(ValidationError, match="必须提供 tool_call_id"):
            Message(role=MessageRole.TOOL, content="result")

    def test_tool_call_id_only_on_tool_message(self):
        with pytest.raises(ValidationError, match="tool_call_id 仅对 role=tool 有意义"):
            Message.assistant("x", tool_call_id="call_1")

    def test_tool_message_payload_is_closed_on_call_id(self):
        payload = Message.tool("37 条命中", tool_call_id="call_1", name="search_tool").to_llm_dict()
        assert payload["tool_call_id"] == "call_1"
        assert payload["role"] == "tool"

    def test_unknown_field_is_rejected(self):
        """extra=forbid：字段名写错当场报错，而不是被静默丢掉。"""
        with pytest.raises(ValidationError):
            Message.user("x", scope="current")


class TestTaskSummary:
    def test_has_exactly_the_eight_required_fields(self):
        """字段集与 require.md 的 task summary 一一对应。"""
        assert set(TaskSummary.model_fields) == {
            "task_id",
            "content",
            "intent",
            "related_task_ids",
            "operations",
            "result",
            "status",
            "output",
        }

    def test_defaults(self):
        summary = TaskSummary(task_id="t1", content="查错误日志")
        assert summary.status is TaskStatus.CREATED
        assert summary.intent is None
        assert summary.related_task_ids == []
        assert summary.operations == []

    def test_intent_prompt_projection_is_a_whitelist(self):
        """意图识别阶段只投喂 content 与 status，不泄漏 operations/result。"""
        summary = TaskSummary(
            task_id="t1",
            content="查错误日志",
            status=TaskStatus.INTENTING,
            result="不该被投喂",
        )
        payload = summary.to_prompt_dict(TaskSummary.INTENT_PROMPT_FIELDS)
        assert payload == {"task_id": "t1", "content": "查错误日志", "status": "intenting"}

    def test_projection_rejects_unknown_field(self):
        summary = TaskSummary(task_id="t1", content="x")
        with pytest.raises(ValueError, match="未知的 summary 字段"):
            summary.to_prompt_dict(("nope",))

    def test_pending_operations_skips_finished_steps(self):
        summary = TaskSummary(
            task_id="t1",
            content="x",
            operations=[
                Operation(index=0, description="检索", status=OperationStatus.SUCCEEDED),
                Operation(index=1, description="统计"),
            ],
        )
        assert [op.index for op in summary.pending_operations()] == [1]

    def test_serialization_roundtrip(self):
        summary = TaskSummary(
            task_id="t1",
            content="查错误日志",
            intent=IntentType.QUERY,
            related_task_ids=["t0"],
            operations=[Operation(index=0, description="检索", suggested_tool="search_tool")],
            status=TaskStatus.EXECUTING,
        )
        restored = TaskSummary.from_dict(json.loads(json.dumps(summary.to_dict())))
        assert restored == summary


class TestTask:
    def test_create_wires_summary_to_task(self):
        task = Task.create("查错误日志", session_id="s1")
        assert task.summary.task_id == task.task_id
        assert task.summary.content == "查错误日志"
        assert task.status is TaskStatus.CREATED

    def test_transition_syncs_both_status_fields(self):
        """两处状态只在 transition_to 里一起改，避免前端读 summary 时看到旧值。"""
        task = Task.create("x", session_id="s1")
        task.transition_to(TaskStatus.INTENTING)
        assert task.status is TaskStatus.INTENTING
        assert task.summary.status is TaskStatus.INTENTING

    def test_transition_bumps_updated_at(self):
        task = Task.create("x", session_id="s1")
        before = task.updated_at
        task.transition_to(TaskStatus.INTENTING)
        assert task.updated_at >= before

    def test_illegal_transition_rejected(self):
        task = Task.create("x", session_id="s1")
        with pytest.raises(TaskStateError):
            task.transition_to(TaskStatus.COMPLETED)
        # 失败的转移不留下副作用
        assert task.status is TaskStatus.CREATED
        assert task.summary.status is TaskStatus.CREATED

    def test_mismatched_summary_is_rejected(self):
        with pytest.raises(ValidationError, match="不一致"):
            Task(task_id="t1", session_id="s1", summary=TaskSummary(task_id="t2", content="x"))

    def test_desynced_status_is_rejected(self):
        with pytest.raises(ValidationError, match=r"必须与 task\.status 同步"):
            Task(
                task_id="t1",
                session_id="s1",
                status=TaskStatus.EXECUTING,
                summary=TaskSummary(task_id="t1", content="x"),
            )
