"""配置测试：统一入口、.env 读取、密钥保护、零配置回落。"""

from pathlib import Path

import pytest

from agent.config import AppSettings, LLMProfile, get_settings, load_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


def write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


class TestDefaults:
    def test_zero_config_is_valid(self):
        """一个环境变量都不设也能构建配置——控制台零配置可跑的前提。"""
        settings = load_settings(env_file=None)
        assert isinstance(settings, AppSettings)
        assert settings.env == "dev"
        assert settings.llm.use_mock_client() is True

    def test_all_groups_are_reachable_from_the_root(self):
        """唯一入口：五个配置组都挂在 AppSettings 上。"""
        settings = load_settings(env_file=None)
        assert settings.llm.primary.model
        assert settings.context.history_summary_limit >= 0
        assert settings.memory.max_tasks >= 1
        assert settings.tool.max_rows >= 1
        assert settings.task.max_steps >= 1


class TestEnvFile:
    def test_reads_grouped_keys(self, tmp_path):
        env_file = write_env(
            tmp_path,
            "\n".join(
                [
                    "APP_ENV=test",
                    "LOG_LEVEL=debug",
                    "LLM_API_KEY=sk-secret",
                    "LLM_MODEL=qwen-max",
                    "LLM_BASE_URL=https://dashscope.example/v1",
                    "CONTEXT_HISTORY_SUMMARY_LIMIT=3",
                    "TOOL_PREVIEW_MAX_ROWS=5",
                    "TASK_MAX_STEPS=4",
                ]
            ),
        )
        settings = load_settings(env_file=env_file)
        assert settings.env == "test"
        assert settings.log.level == "DEBUG"
        assert settings.llm.primary.model == "qwen-max"
        assert settings.llm.primary.base_url == "https://dashscope.example/v1"
        assert settings.context.history_summary_limit == 3
        assert settings.tool.preview_max_rows == 5
        assert settings.task.max_steps == 4

    def test_env_file_none_ignores_real_dotenv(self, tmp_path, monkeypatch):
        """env_file=None 时子配置也不许偷偷去读项目根的 .env，否则测试依赖本地环境。"""
        monkeypatch.delenv("LLM_MODEL", raising=False)
        write_env(tmp_path, "LLM_MODEL=should-not-be-read")
        settings = load_settings(env_file=None)
        assert settings.llm.primary.model == "gpt-4o-mini"

    def test_process_env_overrides(self, monkeypatch):
        monkeypatch.setenv("TASK_MAX_TOOL_CALLS", "3")
        assert load_settings(env_file=None).task.max_tool_calls == 3

    def test_fallback_chain_from_json(self, tmp_path):
        env_file = write_env(
            tmp_path,
            'LLM_FALLBACKS=[{"name":"backup","model":"m2","api_key":"sk-2"}]\nLLM_API_KEY=sk-1\n',
        )
        settings = load_settings(env_file=env_file)
        chain = settings.llm.profile_chain()
        assert [profile.name for profile in chain] == ["primary", "backup"]


class TestSecrets:
    def test_api_key_is_not_leaked_in_repr(self, tmp_path):
        env_file = write_env(tmp_path, "LLM_API_KEY=sk-super-secret\n")
        settings = load_settings(env_file=env_file)
        assert "sk-super-secret" not in repr(settings)
        assert "sk-super-secret" not in str(settings.llm.primary)
        assert settings.llm.primary.api_key.get_secret_value() == "sk-super-secret"

    def test_has_api_key(self):
        assert not LLMProfile().has_api_key()
        assert LLMProfile(api_key="sk-1").has_api_key()


class TestMockFallback:
    def test_missing_key_falls_back_to_mock(self):
        settings = load_settings(env_file=None)
        assert settings.llm.primary.has_api_key() is False
        assert settings.llm.use_mock_client() is True

    def test_key_present_uses_real_client(self, tmp_path):
        env_file = write_env(tmp_path, "LLM_API_KEY=sk-1\n")
        assert load_settings(env_file=env_file).llm.use_mock_client() is False

    def test_use_mock_wins_over_key(self, tmp_path):
        env_file = write_env(tmp_path, "LLM_API_KEY=sk-1\nLLM_USE_MOCK=true\n")
        assert load_settings(env_file=env_file).llm.use_mock_client() is True


class TestSingleton:
    def test_get_settings_is_cached(self):
        assert get_settings() is get_settings()

    def test_reset_cache_rebuilds(self, monkeypatch):
        first = get_settings()
        monkeypatch.setenv("TASK_MAX_STEPS", "2")
        reset_settings_cache()
        second = get_settings()
        assert second is not first
        assert second.task.max_steps == 2


class TestValidation:
    def test_out_of_range_value_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TASK_MAX_STEPS", "0")
        from agent.errors import ConfigError

        with pytest.raises(ConfigError, match="配置校验失败"):
            load_settings(env_file=None)

    def test_data_file_resolves_against_project_root(self):
        settings = load_settings(env_file=None)
        resolved = settings.tool.resolved_data_file()
        assert resolved.is_absolute()
        assert resolved.name == "logs.json"
