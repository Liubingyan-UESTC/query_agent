"""步骤 10：MockLLMClient 按序号/谓词回放，并记录完整 messages。"""

import pytest

from agent.common.errors import LLMError, LLMTimeoutError
from agent.llm.base import LLMRequest
from agent.llm.mock_client import MockLLMClient
from agent.models.message import Message


def test_replays_in_enqueue_order() -> None:
    client = MockLLMClient(
        [MockLLMClient.reply("one"), MockLLMClient.reply("two")],
    )

    first = client.chat(LLMRequest(messages=[Message.user("a")]))
    second = client.chat(LLMRequest(messages=[Message.user("b")]))

    assert first.content == "one"
    assert second.content == "two"
    assert client.recorded_messages() == [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]


def test_when_predicate_takes_priority() -> None:
    client = MockLLMClient()
    client.enqueue(MockLLMClient.reply("fallback"))
    client.enqueue(
        MockLLMClient.reply("intent"),
        when=lambda req: req.purpose == "intent",
    )

    hit = client.chat(LLMRequest(messages=[Message.user("q")], purpose="intent"))
    rest = client.chat(LLMRequest(messages=[Message.user("q")]))

    assert hit.content == "intent"
    assert rest.content == "fallback"


def test_callable_reply_receives_request() -> None:
    def echo(request: LLMRequest) -> object:
        return MockLLMClient.reply(request.messages[0].content)

    client = MockLLMClient([echo])  # type: ignore[list-item]
    response = client.chat(LLMRequest(messages=[Message.user("hello")]))

    assert response.content == "hello"


def test_scripted_exception_is_raised() -> None:
    client = MockLLMClient([LLMTimeoutError("timeout")])

    with pytest.raises(LLMTimeoutError):
        client.chat(LLMRequest(messages=[Message.user("q")]))


def test_exhausted_scripts_raise() -> None:
    client = MockLLMClient([MockLLMClient.reply("only")])
    client.chat(LLMRequest(messages=[Message.user("q")]))

    with pytest.raises(LLMError, match="没有匹配"):
        client.chat(LLMRequest(messages=[Message.user("q")]))


def test_stream_chat_emits_content_then_finish() -> None:
    client = MockLLMClient([MockLLMClient.reply("Hel")])
    chunks = list(client.stream_chat(LLMRequest(messages=[Message.user("q")])))

    assert [item.content for item in chunks] == ["Hel", ""]
    assert chunks[-1].finish_reason == "stop"


def test_stream_empty_content_only_finish_chunk() -> None:
    client = MockLLMClient([MockLLMClient.reply("")])
    chunks = list(client.stream_chat(LLMRequest(messages=[Message.user("q")])))

    assert len(chunks) == 1
    assert chunks[0].finish_reason == "stop"


def test_chat_rejects_stream_flag() -> None:
    client = MockLLMClient([MockLLMClient.reply("x")])
    with pytest.raises(ValueError, match="stream_chat"):
        client.chat(LLMRequest(messages=[Message.user("q")], stream=True))


def test_empty_messages_are_rejected() -> None:
    client = MockLLMClient([MockLLMClient.reply("x")])
    with pytest.raises(ValueError, match="messages"):
        client.chat(LLMRequest(messages=[]))
