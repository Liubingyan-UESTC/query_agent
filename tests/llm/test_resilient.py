"""步骤 10：主模型超时降级、不可重试错误不重试、埋点字段齐全。"""

import pytest

from agent.common.errors import LLMError, LLMTimeoutError
from agent.config.settings import LLMProfile, LLMSettings
from agent.llm.base import LLMRequest
from agent.llm.mock_client import MockLLMClient
from agent.llm.resilient import ResilientLLMClient
from agent.models.message import Message


def _settings(*, max_retries: int = 0, fallbacks: list[LLMProfile] | None = None) -> LLMSettings:
    return LLMSettings(
        use_mock=True,
        primary=LLMProfile(name="primary", model="main-model", max_retries=max_retries),
        fallbacks=fallbacks or [],
        intent_model="intent-model",
    )


def _client(
    *inner: MockLLMClient,
    settings: LLMSettings | None = None,
    sleeps: list[float] | None = None,
) -> ResilientLLMClient:
    return ResilientLLMClient(
        inner,
        settings or _settings(),
        sleeper=(sleeps.append if sleeps is not None else (lambda _: None)),
    )


def test_primary_timeout_falls_back() -> None:
    primary = MockLLMClient([LLMTimeoutError("timeout")])
    backup = MockLLMClient([MockLLMClient.reply("ok", model="backup-model")])
    client = _client(
        primary,
        backup,
        settings=_settings(fallbacks=[LLMProfile(name="backup", model="backup-model")]),
    )

    response = client.chat(LLMRequest(messages=[Message.user("q")]))

    assert response.content == "ok"
    assert client.last_stats is not None
    assert client.last_stats.ok is True
    assert client.last_stats.fallback_index == 1
    assert client.last_stats.profile_name == "backup"
    assert client.last_stats.model == "backup-model"
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1


def test_retryable_error_retries_same_client() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky(_request: LLMRequest) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMTimeoutError("timeout")
        return MockLLMClient.reply("recovered", prompt_tokens=3, completion_tokens=2)

    client = _client(
        MockLLMClient([flaky]),  # type: ignore[list-item]
        settings=_settings(max_retries=1),
        sleeps=sleeps,
    )

    response = client.chat(LLMRequest(messages=[Message.user("q")]))

    assert response.content == "recovered"
    assert sleeps == [0.5]
    assert client.last_stats is not None
    assert client.last_stats.retry_count == 1
    assert client.last_stats.fallback_index == 0
    assert client.last_stats.prompt_tokens == 3
    assert client.last_stats.completion_tokens == 2


def test_non_retryable_error_is_not_retried_or_degraded() -> None:
    sleeps: list[float] = []
    primary = MockLLMClient([LLMError("bad request", retryable=False)])
    backup = MockLLMClient([MockLLMClient.reply("should-not-run")])
    client = _client(
        primary,
        backup,
        settings=_settings(
            max_retries=2,
            fallbacks=[LLMProfile(name="backup", max_retries=0)],
        ),
        sleeps=sleeps,
    )

    with pytest.raises(LLMError, match="bad request") as caught:
        client.chat(LLMRequest(messages=[Message.user("q")]))

    assert caught.value.retryable is False
    assert sleeps == []
    assert backup.calls == []
    assert client.last_stats is not None
    assert client.last_stats.ok is False
    assert client.last_stats.fallback_index == 0


def test_all_profiles_fail() -> None:
    client = _client(
        MockLLMClient([LLMTimeoutError("p1")]),
        MockLLMClient([LLMTimeoutError("p2")]),
        settings=_settings(fallbacks=[LLMProfile(name="backup", max_retries=0)]),
    )

    with pytest.raises(LLMTimeoutError, match="p2"):
        client.chat(LLMRequest(messages=[Message.user("q")]))

    assert client.last_stats is not None
    assert client.last_stats.ok is False
    assert client.last_stats.fallback_index == 1


def test_purpose_routes_model_name() -> None:
    inner = MockLLMClient([MockLLMClient.reply("ok")])
    client = _client(inner, settings=_settings())

    client.chat(LLMRequest(messages=[Message.user("q")], purpose="intent"))

    assert inner.calls[0].model == "intent-model"
    assert client.last_stats is not None
    assert client.last_stats.purpose == "intent"


def test_explicit_model_wins_over_purpose() -> None:
    inner = MockLLMClient([MockLLMClient.reply("ok")])
    client = _client(inner)

    client.chat(LLMRequest(messages=[Message.user("q")], purpose="intent", model="forced"))

    assert inner.calls[0].model == "forced"


def test_purpose_does_not_pin_fallback_to_primary_model() -> None:
    """降级端点必须用自己的 profile.model，不能继续要 intent / 主模型的名字。"""
    primary = MockLLMClient([LLMTimeoutError("timeout")])
    backup = MockLLMClient([MockLLMClient.reply("ok")])
    client = _client(
        primary,
        backup,
        settings=_settings(fallbacks=[LLMProfile(name="backup", model="backup-model")]),
    )

    client.chat(LLMRequest(messages=[Message.user("q")], purpose="intent"))

    assert primary.calls[0].model == "intent-model"
    assert backup.calls[0].model == "backup-model"


def test_explicit_model_is_shared_across_fallback_chain() -> None:
    primary = MockLLMClient([LLMTimeoutError("timeout")])
    backup = MockLLMClient([MockLLMClient.reply("ok")])
    client = _client(
        primary,
        backup,
        settings=_settings(fallbacks=[LLMProfile(name="backup", model="backup-model")]),
    )

    client.chat(LLMRequest(messages=[Message.user("q")], model="forced"))

    assert primary.calls[0].model == "forced"
    assert backup.calls[0].model == "forced"


def test_empty_purpose_route_still_uses_fallback_profile_model() -> None:
    settings = LLMSettings(
        use_mock=True,
        primary=LLMProfile(name="primary", model="main-model", max_retries=0),
        fallbacks=[LLMProfile(name="backup", model="backup-model")],
        intent_model=None,
    )
    primary = MockLLMClient([LLMTimeoutError("timeout")])
    backup = MockLLMClient([MockLLMClient.reply("ok")])
    client = _client(primary, backup, settings=settings)

    client.chat(LLMRequest(messages=[Message.user("q")], purpose="intent"))

    assert primary.calls[0].model == "main-model"
    assert backup.calls[0].model == "backup-model"


def test_backoff_is_capped() -> None:
    sleeps: list[float] = []
    replies: list[object] = [LLMTimeoutError("t") for _ in range(6)]
    client = _client(
        MockLLMClient(replies),  # type: ignore[arg-type]
        settings=_settings(max_retries=5),
        sleeps=sleeps,
    )

    with pytest.raises(LLMTimeoutError):
        client.chat(LLMRequest(messages=[Message.user("q")]))

    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_empty_stream_succeeds() -> None:
    class Silent(MockLLMClient):
        def stream_chat(self, request: LLMRequest):  # type: ignore[override]
            self.calls.append(request)
            if False:
                yield from ()

    client = _client(Silent())
    assert list(client.stream_chat(LLMRequest(messages=[Message.user("q")]))) == []
    assert client.last_stats is not None
    assert client.last_stats.ok is True


def test_stream_chat_uses_same_retry_policy() -> None:
    primary = MockLLMClient([LLMTimeoutError("timeout")])
    backup = MockLLMClient([MockLLMClient.reply("streamed")])
    client = _client(
        primary,
        backup,
        settings=_settings(fallbacks=[LLMProfile(name="backup")]),
    )

    chunks = list(client.stream_chat(LLMRequest(messages=[Message.user("q")])))

    assert [item.content for item in chunks] == ["streamed", ""]
    assert client.last_stats is not None
    assert client.last_stats.ok is True
    assert client.last_stats.fallback_index == 1


def test_rejects_empty_client_list() -> None:
    with pytest.raises(ValueError, match="底层客户端"):
        ResilientLLMClient([], _settings())


def test_chat_rejects_stream_flag() -> None:
    client = _client(MockLLMClient([MockLLMClient.reply("x")]))
    with pytest.raises(ValueError, match="stream_chat"):
        client.chat(LLMRequest(messages=[Message.user("q")], stream=True))


def test_close_closes_inner_clients() -> None:
    closed: list[str] = []

    class _Inner(MockLLMClient):
        def close(self) -> None:
            closed.append("yes")
            super().close()

    client = _client(_Inner([MockLLMClient.reply("x")]))
    client.close()

    assert closed == ["yes"]


def test_extra_clients_fall_back_to_primary_profile() -> None:
    """客户端比 profile 链更长时，用 primary 的 max_retries / name，避免越界。"""
    settings = _settings()
    extra = MockLLMClient([MockLLMClient.reply("from-extra", model="extra")])
    client = _client(MockLLMClient([LLMTimeoutError("p")]), extra, settings=settings)

    response = client.chat(LLMRequest(messages=[Message.user("q")]))

    assert response.content == "from-extra"
    assert client.last_stats is not None
    assert client.last_stats.profile_name == "primary"


def test_default_sleeper_and_clock_are_real() -> None:
    """覆盖构造期缺省分支，不发起会 sleep 的调用。"""
    inner = MockLLMClient([MockLLMClient.reply("ok")])
    client = ResilientLLMClient([inner], _settings())
    response = client.chat(LLMRequest(messages=[Message.user("q")]))
    assert response.content == "ok"
    client.close()
