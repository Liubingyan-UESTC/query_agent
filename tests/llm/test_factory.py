"""步骤 10：`build_llm_client` 按 use_mock / profile 链装配。"""

from pydantic import SecretStr

from agent.config.settings import AppSettings, LLMProfile, LLMSettings
from agent.llm.factory import build_llm_client
from agent.llm.mock_client import MockLLMClient
from agent.llm.openai_client import OpenAICompatClient
from agent.llm.resilient import ResilientLLMClient


def test_use_mock_returns_default_mock() -> None:
    client = build_llm_client(LLMSettings(use_mock=True))

    assert isinstance(client, MockLLMClient)


def test_use_mock_returns_injected_mock() -> None:
    mock = MockLLMClient([MockLLMClient.reply("preset")])
    client = build_llm_client(LLMSettings(use_mock=True), mock=mock)

    assert client is mock


def test_use_mock_from_app_settings() -> None:
    settings = AppSettings(llm=LLMSettings(use_mock=True))
    client = build_llm_client(settings)

    assert isinstance(client, MockLLMClient)


def test_real_settings_wrap_openai_in_resilient() -> None:
    settings = LLMSettings(
        use_mock=False,
        primary=LLMProfile(name="primary", api_key=SecretStr("sk-test")),
        fallbacks=[LLMProfile(name="backup", api_key=SecretStr("sk-fb"))],
    )
    client = build_llm_client(settings, sleeper=lambda _: None)

    assert isinstance(client, ResilientLLMClient)
    assert len(client._clients) == 2
    assert all(isinstance(item, OpenAICompatClient) for item in client._clients)
    client.close()


def test_client_factory_is_used_for_each_profile() -> None:
    seen: list[str] = []

    def factory(profile: LLMProfile) -> MockLLMClient:
        seen.append(profile.name)
        return MockLLMClient([MockLLMClient.reply(profile.name)])

    settings = LLMSettings(
        use_mock=False,
        primary=LLMProfile(name="primary", api_key=SecretStr("sk-test")),
        fallbacks=[LLMProfile(name="backup", api_key=SecretStr("sk-fb"))],
    )
    client = build_llm_client(settings, client_factory=factory)

    assert seen == ["primary", "backup"]
    assert isinstance(client, ResilientLLMClient)
    client.close()
