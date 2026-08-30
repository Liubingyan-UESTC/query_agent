"""步骤 10：json_schema 优先、修复重问一次、最终失败抛 LLMResponseFormatError。"""

import pytest
from pydantic import BaseModel, Field

from agent.common.errors import LLMError, LLMResponseFormatError, LLMTimeoutError
from agent.llm.base import LLMRequest
from agent.llm.mock_client import MockLLMClient
from agent.llm.structured import call_structured, extract_json
from agent.models.message import Message
from agent.prompt.assembler import EmptyPlanOutput

SCHEMA = {
    "title": "Intent Result!",
    "type": "object",
    "additionalProperties": False,
    "required": ["intent"],
    "properties": {
        "intent": {"type": "string", "enum": ["chat", "new_query"]},
        "related_task_ids": {"type": "array"},
        "confidence": {"type": "number"},
        "ok": {"type": "boolean"},
        "count": {"type": "integer"},
        "meta": {"type": "object"},
        "note": {"type": ["string", "null"]},
        "flag": {"type": "unknown"},
    },
}


class IntentOut(BaseModel):
    intent: str
    related_task_ids: list[str] = Field(default_factory=list)


def test_json_schema_success_on_first_try() -> None:
    client = MockLLMClient([MockLLMClient.reply('{"intent":"chat"}')])

    data = call_structured(client, [Message.user("q")], SCHEMA)

    assert data == {"intent": "chat"}
    assert client.calls[0].response_format is not None
    assert client.calls[0].response_format["type"] == "json_schema"
    assert client.calls[0].response_format["json_schema"]["name"] == "Intent_Result"


def test_repair_once_then_success() -> None:
    client = MockLLMClient(
        [
            MockLLMClient.reply("not-json"),
            MockLLMClient.reply('{"intent":"new_query"}'),
        ]
    )

    data = call_structured(client, [Message.user("查日志")], SCHEMA)

    assert data["intent"] == "new_query"
    assert len(client.calls) == 2
    assert "无法通过校验" in client.calls[1].messages[-1].content


def test_all_repairs_exhausted_raise_format_error() -> None:
    client = MockLLMClient(
        [MockLLMClient.reply("nope"), MockLLMClient.reply("still nope")],
    )

    with pytest.raises(LLMResponseFormatError) as caught:
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=1)

    assert caught.value.retryable is False
    assert caught.value.detail is not None


def test_max_repair_zero_does_not_reask() -> None:
    client = MockLLMClient([MockLLMClient.reply("nope")])

    with pytest.raises(LLMResponseFormatError):
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=0)

    assert len(client.calls) == 1


def test_unsupported_json_schema_falls_back_to_prompt() -> None:
    def handler(request: LLMRequest) -> object:
        if request.response_format is not None:
            raise LLMError("response_format json_schema is not supported", retryable=False)
        return MockLLMClient.reply('{"intent":"chat"}')

    client = MockLLMClient([handler])  # type: ignore[list-item]
    data = call_structured(client, [Message.user("q")], SCHEMA)

    assert data["intent"] == "chat"
    assert any("JSON Schema" in message.content for message in client.calls[1].messages)


def test_unsupported_then_invalid_then_repair() -> None:
    replies = [
        LLMError("does not support json_object", retryable=False),
        MockLLMClient.reply("oops"),
        MockLLMClient.reply('{"intent":"chat"}'),
    ]
    client = MockLLMClient(replies)  # type: ignore[arg-type]

    data = call_structured(client, [Message.user("q")], SCHEMA, max_repair=1)

    assert data["intent"] == "chat"
    assert len(client.calls) == 3


def test_retryable_error_is_not_treated_as_unsupported() -> None:
    client = MockLLMClient([LLMTimeoutError("timeout")])

    with pytest.raises(LLMTimeoutError):
        call_structured(client, [Message.user("q")], SCHEMA)


def test_non_format_client_error_propagates() -> None:
    client = MockLLMClient([LLMError("bad key", retryable=False)])

    with pytest.raises(LLMError, match="bad key"):
        call_structured(client, [Message.user("q")], SCHEMA)


def test_prompt_mode_skips_json_schema() -> None:
    client = MockLLMClient([MockLLMClient.reply('{"intent":"chat"}')])

    call_structured(client, [Message.user("q")], SCHEMA, prefer_json_schema=False)

    assert client.calls[0].response_format is None
    assert "JSON Schema" in client.calls[0].messages[-1].content


def test_pydantic_schema_and_enum_repair() -> None:
    client = MockLLMClient(
        [
            MockLLMClient.reply('{"intent":1}'),
            MockLLMClient.reply('{"intent":"chat"}'),
        ]
    )

    data = call_structured(client, [Message.user("q")], IntentOut)

    assert data["intent"] == "chat"
    assert data["related_task_ids"] == []


def test_extract_json_from_fence_and_surrounding_text() -> None:
    assert extract_json('prefix {"intent":"chat"} suffix') == {"intent": "chat"}
    assert extract_json('```json\n{"intent":"chat"}\n```') == {"intent": "chat"}


def test_extract_json_rejects_empty_and_non_json() -> None:
    with pytest.raises(ValueError, match="空内容"):
        extract_json("   ")
    with pytest.raises(ValueError, match="没有合法 JSON"):
        extract_json("definitely not")


def test_schema_field_validation() -> None:
    client = MockLLMClient(
        [
            MockLLMClient.reply(
                '{"intent":"chat","related_task_ids":[],"confidence":0.5,'
                '"ok":true,"count":1,"meta":{},"note":null,"flag":1}'
            )
        ]
    )

    data = call_structured(client, [Message.user("q")], SCHEMA)

    assert data["count"] == 1
    assert data["note"] is None


def test_schema_rejects_wrong_enum_type_and_extra() -> None:
    client = MockLLMClient([MockLLMClient.reply('{"intent":"export"}')])
    with pytest.raises(LLMResponseFormatError, match="不在"):
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=0)

    client = MockLLMClient([MockLLMClient.reply('{"intent":"chat","nope":1}')])
    with pytest.raises(LLMResponseFormatError, match="多余"):
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=0)

    client = MockLLMClient([MockLLMClient.reply("[]")])
    with pytest.raises(LLMResponseFormatError, match="JSON 对象"):
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=0)

    client = MockLLMClient([MockLLMClient.reply('{"intent":"chat","count":true}')])
    with pytest.raises(LLMResponseFormatError, match="integer"):
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=0)


def test_non_object_schema_is_rejected() -> None:
    client = MockLLMClient([MockLLMClient.reply("[1]")])
    with pytest.raises(LLMResponseFormatError, match="type=object"):
        call_structured(client, [Message.user("q")], {"type": "array"}, max_repair=0)


def test_missing_required_and_type_mismatch() -> None:
    client = MockLLMClient([MockLLMClient.reply("{}")])
    with pytest.raises(LLMResponseFormatError, match="缺少字段"):
        call_structured(client, [Message.user("q")], SCHEMA, max_repair=0)

    client = MockLLMClient([MockLLMClient.reply('{"intent":1}')])
    with pytest.raises(LLMResponseFormatError, match="string"):
        call_structured(
            client,
            [Message.user("q")],
            {
                "type": "object",
                "properties": {"intent": {"type": "string"}},
                "required": ["intent"],
            },
            max_repair=0,
        )


def test_invalid_schema_type_and_empty_messages() -> None:
    with pytest.raises(TypeError, match="schema"):
        call_structured(MockLLMClient(), [Message.user("q")], "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_repair"):
        call_structured(MockLLMClient(), [Message.user("q")], SCHEMA, max_repair=-1)
    with pytest.raises(ValueError, match="messages"):
        call_structured(MockLLMClient(), [], SCHEMA)


def test_purpose_and_temperature_are_forwarded() -> None:
    client = MockLLMClient([MockLLMClient.reply('{"intent":"chat"}')])

    call_structured(
        client,
        [Message.user("q")],
        SCHEMA,
        purpose="intent",
        model="intent-model",
        max_tokens=64,
    )

    request = client.calls[0]
    assert request.purpose == "intent"
    assert request.model == "intent-model"
    assert request.temperature == 0.0
    assert request.max_tokens == 64


def test_number_accepts_int_and_boolean_array_object() -> None:
    schema = {
        "type": "object",
        "required": ["n", "ok", "items", "obj", "text"],
        "properties": {
            "n": {"type": "number"},
            "ok": {"type": "boolean"},
            "items": {"type": "array"},
            "obj": {"type": "object"},
            "text": {"type": "string"},
        },
    }
    client = MockLLMClient(
        [MockLLMClient.reply('{"n":2,"ok":false,"items":[1],"obj":{},"text":"x"}')]
    )

    data = call_structured(client, [Message.user("q")], schema)

    assert data["n"] == 2
    assert data["ok"] is False


def test_max_items_and_nested_required() -> None:
    schema = {
        "type": "object",
        "required": ["operations"],
        "additionalProperties": False,
        "properties": {
            "operations": {
                "type": "array",
                "maxItems": 0,
                "items": {
                    "type": "object",
                    "required": ["index", "tool"],
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "tool": {"type": "string"},
                    },
                },
            }
        },
    }
    ok = MockLLMClient([MockLLMClient.reply('{"operations":[]}')])
    assert call_structured(ok, [Message.user("q")], schema) == {"operations": []}

    too_many = MockLLMClient([MockLLMClient.reply('{"operations":[{"index":0,"tool":"search"}]}')])
    with pytest.raises(LLMResponseFormatError, match="maxItems"):
        call_structured(too_many, [Message.user("q")], schema, max_repair=0)

    nested = {
        "type": "object",
        "required": ["operations"],
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "tool"],
                    "properties": {
                        "index": {"type": "integer"},
                        "tool": {"type": "string"},
                    },
                },
            }
        },
    }
    missing = MockLLMClient([MockLLMClient.reply('{"operations":[{"index":0}]}')])
    with pytest.raises(LLMResponseFormatError, match="缺少字段"):
        call_structured(missing, [Message.user("q")], nested, max_repair=0)

    chat_ok = MockLLMClient([MockLLMClient.reply('{"operations":[]}')])
    assert call_structured(chat_ok, [Message.user("q")], EmptyPlanOutput) == {"operations": []}
    chat_bad = MockLLMClient(
        [
            MockLLMClient.reply(
                '{"operations":[{"index":0,"name":"x","tool":"search","args":{},"expect":"y"}]}'
            )
        ]
    )
    with pytest.raises(LLMResponseFormatError):
        call_structured(chat_bad, [Message.user("q")], EmptyPlanOutput, max_repair=0)


def test_property_without_type_and_non_dict_spec() -> None:
    schema = {
        "type": "object",
        "properties": {"intent": "just-a-flag", "other": {}},
        "required": ["intent"],
    }
    client = MockLLMClient([MockLLMClient.reply('{"intent":"chat","other":1}')])

    data = call_structured(client, [Message.user("q")], schema)

    assert data["other"] == 1
