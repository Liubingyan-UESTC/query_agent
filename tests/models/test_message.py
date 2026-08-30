"""步骤 5 验收：`to_llm_dict()` 符合 OpenAI messages 规范、JSON 序列化往返一致。"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent.common.enums import ContextScope, MessageRole
from agent.models.message import Message, ToolCall

# 编排字段绝不能进入发给模型的载荷
ORCHESTRATION_FIELDS = [
    "message_id",
    "artifact_refs",
    "scope",
    "source_task_id",
    "created_at",
    "meta",
]

# OpenAI messages 规范允许出现的键
PROTOCOL_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}


# ============================================================ OpenAI 协议一致性


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT])
def test_plain_message_emits_only_role_and_content(role: MessageRole) -> None:
    payload = Message(role=role, content="你好").to_llm_dict()

    assert payload == {"role": role.value, "content": "你好"}


def test_role_is_serialized_as_a_plain_string() -> None:
    """载荷要能直接交给 json.dumps，不能残留枚举对象。"""
    payload = Message.user("hi").to_llm_dict()

    assert type(payload["role"]) is str
    assert json.dumps(payload)


def test_content_key_is_always_present() -> None:
    """OpenAI 要求 content 键必须出现，哪怕是空串。"""
    payload = Message.assistant().to_llm_dict()

    assert "content" in payload
    assert payload["content"] == ""


def test_orchestration_fields_never_reach_the_model() -> None:
    message = Message.user(
        "查一下昨天的错误日志",
        artifact_refs=["art_1", "art_2"],
        scope=ContextScope.RELATED,
        source_task_id="task_upstream",
        meta={"channel": "web"},
    )

    payload = message.to_llm_dict()

    assert set(payload) <= PROTOCOL_KEYS
    for field in ORCHESTRATION_FIELDS:
        assert field not in payload


def test_optional_keys_are_omitted_rather_than_null() -> None:
    """输出 null 占位会白白消耗 token，且部分网关对 null 的处理不一致。"""
    payload = Message.user("hi").to_llm_dict()

    assert "name" not in payload
    assert "tool_calls" not in payload
    assert "tool_call_id" not in payload


def test_name_is_emitted_when_set() -> None:
    payload = Message.user("hi", name="alice").to_llm_dict()

    assert payload["name"] == "alice"


def test_assistant_tool_calls_use_the_openai_nested_shape() -> None:
    message = Message.assistant(
        tool_calls=[ToolCall(id="call_1", name="kibana_query", arguments='{"index":"logs"}')]
    )

    payload = message.to_llm_dict()

    assert payload["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "kibana_query", "arguments": '{"index":"logs"}'},
        }
    ]


def test_tool_result_message_carries_its_call_id() -> None:
    payload = Message.tool_result("共 12 行", tool_call_id="call_1").to_llm_dict()

    assert payload == {"role": "tool", "content": "共 12 行", "tool_call_id": "call_1"}


def test_system_factory_builds_the_first_message_of_a_task() -> None:
    """步骤 13 的 create_window 用它作为窗口首条消息。"""
    message = Message.system("你是数据查询助手")

    assert message.role is MessageRole.SYSTEM
    assert message.to_llm_dict() == {"role": "system", "content": "你是数据查询助手"}


@pytest.mark.parametrize(
    ("factory", "role"),
    [
        (Message.system, MessageRole.SYSTEM),
        (Message.user, MessageRole.USER),
        (Message.assistant, MessageRole.ASSISTANT),
    ],
)
def test_factories_forward_orchestration_kwargs(factory: object, role: MessageRole) -> None:
    message = factory("正文", scope=ContextScope.RELATED, source_task_id="task_1")  # type: ignore[operator]

    assert message.role is role
    assert message.scope is ContextScope.RELATED
    assert message.source_task_id == "task_1"


# ============================================================ 协议不变量


def test_non_current_scope_requires_source_task_id() -> None:
    with pytest.raises(ValidationError, match="source_task_id"):
        Message.user("注入", scope=ContextScope.RELATED)


def test_only_assistant_may_carry_tool_calls() -> None:
    with pytest.raises(ValidationError, match="assistant"):
        Message(
            role=MessageRole.USER,
            tool_calls=[ToolCall(id="c1", name="t")],
        )


def test_tool_message_requires_a_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role=MessageRole.TOOL, content="结果")


def test_call_id_is_rejected_on_non_tool_roles() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role=MessageRole.USER, content="hi", tool_call_id="call_1")


def test_unknown_field_is_rejected() -> None:
    """字段名写错若被静默丢弃，表现出来的现象是「值莫名变成默认值」，极难排查。"""
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="hi", scoope=ContextScope.CURRENT)


def test_invalid_role_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(role="Assist", content="hi")


def test_assignment_is_validated() -> None:
    message = Message.user("hi")

    with pytest.raises(ValidationError):
        message.role = "nope"


# ============================================================ 序列化往返


def test_json_round_trip_is_lossless() -> None:
    original = Message.assistant(
        "已完成查询",
        name="planner",
        tool_calls=[ToolCall(id="c1", name="kibana_query", arguments='{"size":10}')],
        artifact_refs=["art_1"],
        scope=ContextScope.HISTORY,
        source_task_id="task_1",
        meta={"tokens": 42},
    )

    restored = Message.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original


def test_to_dict_is_directly_json_serializable() -> None:
    """datetime 与枚举都要已经转成字符串，调用方无须自备 encoder。"""
    payload = Message.user("hi").to_dict()

    assert isinstance(payload["created_at"], str)
    assert isinstance(payload["scope"], str)
    assert json.dumps(payload)


def test_round_trip_preserves_timezone_aware_timestamp() -> None:
    original = Message.user("hi", created_at=datetime(2026, 8, 30, 1, 2, 3, tzinfo=UTC))

    restored = Message.from_dict(original.to_dict())

    assert restored.created_at == original.created_at
    assert restored.created_at.tzinfo is not None


# ============================================================ 默认值


def test_each_message_gets_a_unique_prefixed_id() -> None:
    ids = {Message.user("hi").message_id for _ in range(100)}

    assert len(ids) == 100
    assert all(value.startswith("msg_") for value in ids)


def test_defaults_are_not_shared_between_instances() -> None:
    """可变默认值若被共享，一条消息的 artifact_refs 会污染所有消息。"""
    first = Message.user("a")
    second = Message.user("b")

    first.artifact_refs.append("art_1")
    first.meta["k"] = "v"

    assert second.artifact_refs == []
    assert second.meta == {}


def test_default_scope_is_current() -> None:
    """默认必须是 CURRENT，否则新消息不会被写回 WorkingMemory。"""
    assert Message.user("hi").scope is ContextScope.CURRENT


def test_created_at_is_timezone_aware_utc() -> None:
    assert Message.user("hi").created_at.tzinfo is not None


# ============================================================ ToolCall


def test_parsed_arguments_returns_the_decoded_object() -> None:
    call = ToolCall(id="c1", name="q", arguments='{"index": "logs", "size": 10}')

    assert call.parsed_arguments() == {"index": "logs", "size": 10}


def test_arguments_default_to_an_empty_object() -> None:
    assert ToolCall(id="c1", name="q").parsed_arguments() == {}


def test_malformed_arguments_raise_with_the_tool_name() -> None:
    call = ToolCall(id="c1", name="kibana_query", arguments="{not json")

    with pytest.raises(ValueError, match="kibana_query"):
        call.parsed_arguments()


@pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "42", "null"])
def test_non_object_arguments_are_rejected(raw: str) -> None:
    call = ToolCall(id="c1", name="q", arguments=raw)

    with pytest.raises(ValueError, match="JSON 对象"):
        call.parsed_arguments()


def test_arguments_are_kept_verbatim() -> None:
    """保留模型原样输出，才能在解析失败时复现问题。"""
    raw = '{"index":"logs",  "size":  10}'
    call = ToolCall(id="c1", name="q", arguments=raw)

    assert call.arguments == raw
    assert call.to_llm_dict()["function"]["arguments"] == raw


# ============================================================ ToolCall 嵌套结构互转


def test_from_llm_dict_flattens_the_openai_shape() -> None:
    call = ToolCall.from_llm_dict(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "kibana_query", "arguments": '{"index":"logs"}'},
        }
    )

    assert call.id == "call_1"
    assert call.name == "kibana_query"
    assert call.arguments == '{"index":"logs"}'


def test_llm_dict_round_trip_is_lossless() -> None:
    original = ToolCall(id="call_1", name="kibana_query", arguments='{"size":10}')

    assert ToolCall.from_llm_dict(original.to_llm_dict()) == original


def test_from_llm_dict_defaults_the_type_to_function() -> None:
    """部分网关会省略 type 字段。"""
    call = ToolCall.from_llm_dict({"id": "c1", "function": {"name": "q"}})

    assert call.name == "q"
    assert call.arguments == "{}"


def test_from_llm_dict_rejects_unknown_call_types() -> None:
    with pytest.raises(ValueError, match="tool_call 类型"):
        ToolCall.from_llm_dict({"id": "c1", "type": "code_interpreter", "function": {"name": "q"}})


@pytest.mark.parametrize("payload", [{"id": "c1"}, {"id": "c1", "function": "not-an-object"}])
def test_from_llm_dict_requires_the_function_object(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="function"):
        ToolCall.from_llm_dict(payload)


@pytest.mark.parametrize(
    ("payload", "missing"),
    [
        ({"function": {"name": "q"}}, "id"),
        ({"id": "c1", "function": {"arguments": "{}"}}, "name"),
    ],
)
def test_from_llm_dict_names_the_missing_field(payload: dict[str, object], missing: str) -> None:
    """裸 KeyError 看不出是模型返回不合规还是解析代码写错。"""
    with pytest.raises(ValueError, match=missing):
        ToolCall.from_llm_dict(payload)


def test_message_tool_calls_survive_a_full_protocol_round_trip() -> None:
    """步骤 9 的实际用法：从响应载荷还原为 Message，再发回给模型。"""
    original = Message.assistant(
        tool_calls=[
            ToolCall(id="c1", name="kibana_query", arguments='{"index":"logs"}'),
            ToolCall(id="c2", name="analysis", arguments='{"artifact_id":"art_1"}'),
        ]
    )

    payload = original.to_llm_dict()
    restored = Message.assistant(
        tool_calls=[ToolCall.from_llm_dict(item) for item in payload["tool_calls"]]
    )

    assert restored.tool_calls == original.tool_calls
