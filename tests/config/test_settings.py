"""步骤 2 验收：默认值生效、环境变量覆盖生效、缺失必填项报明确错误。"""

import os
import re
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from agent.config import (
    AppSettings,
    ConfigError,
    ContextSettings,
    ESSettings,
    LLMProfile,
    LLMSettings,
    MemorySettings,
    PrimaryLLMProfile,
    StoreSettings,
    TaskSettings,
    ToolSettings,
    TraceSettings,
    get_settings,
    load_settings,
    reset_settings_cache,
)
from agent.config.settings import _GROUP_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

MANAGED_PREFIXES = (
    "APP_",
    "LLM_",
    "CONTEXT_",
    "MEMORY_",
    "TOOL_",
    "TASK_",
    "STORE_",
    "ES_",
    "TRACE_",
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """清空本项目管辖的全部环境变量，使每个用例从零起步。"""
    for key in list(os.environ):
        if key.startswith(MANAGED_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _load(env: pytest.MonkeyPatch, **values: str) -> AppSettings:
    """在隔离环境下加载配置；env_file=None 确保不受本地 .env 影响。"""
    for key, value in values.items():
        env.setenv(key, value)
    return load_settings(env_file=None)


# ============================================================ 默认值


def test_defaults_apply_when_env_is_absent(env: pytest.MonkeyPatch) -> None:
    settings = _load(env, LLM_USE_MOCK="true")

    assert settings.env == "dev"
    assert settings.debug is True
    assert settings.log_level == "INFO"
    assert settings.allowed_hosts == ["*"]
    assert settings.cors_allowed_origins == []
    assert settings.is_prod is False

    assert settings.llm.primary.model == "gpt-4o-mini"
    assert settings.llm.primary.timeout == 60.0
    assert settings.llm.primary.max_retries == 2
    assert settings.llm.fallbacks == []

    assert settings.context.max_total_tokens == 32_000
    assert settings.context.history_summary_limit == 10
    assert settings.context.related_task_content_limit == 6

    assert settings.memory.session_ttl == 3600
    assert settings.memory.working_memory_max_tasks == 20

    assert settings.tool.default_timeout == 30.0
    assert settings.tool.max_result_rows == 1000
    assert settings.tool.max_retry == 2

    assert settings.task.max_steps == 10
    assert settings.task.max_tool_calls == 20
    assert settings.task.max_replan == 2
    assert settings.task.max_duration_seconds == 300.0

    assert settings.store.backend == "memory"
    assert settings.es.hosts == ["http://localhost:9200"]
    assert settings.es.username is None
    assert settings.trace.enabled is False


# ============================================================ 环境变量覆盖


def test_env_overrides_take_effect(env: pytest.MonkeyPatch) -> None:
    settings = _load(
        env,
        LLM_USE_MOCK="true",
        APP_ENV="test",
        APP_DEBUG="false",
        APP_LOG_LEVEL="DEBUG",
        APP_ALLOWED_HOSTS='["api.internal"]',
        CONTEXT_MAX_TOTAL_TOKENS="8000",
        CONTEXT_HISTORY_SUMMARY_LIMIT="3",
        MEMORY_SESSION_TTL="60",
        TOOL_MAX_RESULT_ROWS="50",
        TASK_MAX_STEPS="7",
        STORE_BACKEND="redis",
        STORE_REDIS_URL="redis://cache:6379/1",
        ES_HOSTS='["http://es-a:9200", "http://es-b:9200"]',
        ES_VERIFY_CERTS="true",
        TRACE_ENABLED="true",
        TRACE_DIR="/var/log/traces",
    )

    assert settings.env == "test"
    assert settings.debug is False
    assert settings.log_level == "DEBUG"
    assert settings.allowed_hosts == ["api.internal"]
    assert settings.context.max_total_tokens == 8000
    assert settings.context.history_summary_limit == 3
    assert settings.memory.session_ttl == 60
    assert settings.tool.max_result_rows == 50
    assert settings.task.max_steps == 7
    assert settings.store.backend == "redis"
    assert settings.store.redis_url == "redis://cache:6379/1"
    assert settings.es.hosts == ["http://es-a:9200", "http://es-b:9200"]
    assert settings.es.verify_certs is True
    assert settings.trace.enabled is True
    assert settings.trace.dir == Path("/var/log/traces")


def test_primary_profile_reads_flat_env_names(env: pytest.MonkeyPatch) -> None:
    """含下划线的字段必须能由扁平变量名注入。

    这是本模块不使用 `env_nested_delimiter="_"` 的原因：那会把
    `LLM_PRIMARY_API_KEY` 贪心切分成 `primary.api.key`，导致 api_key 永远读不到。
    """
    settings = _load(
        env,
        LLM_PRIMARY_NAME="main",
        LLM_PRIMARY_BASE_URL="https://example.test/v1",
        LLM_PRIMARY_API_KEY="sk-unit-test",
        LLM_PRIMARY_MODEL="my-model",
        LLM_PRIMARY_TIMEOUT="12.5",
        LLM_PRIMARY_MAX_RETRIES="4",
        LLM_PRIMARY_TEMPERATURE="0.7",
        LLM_PRIMARY_MAX_TOKENS="256",
    )

    primary = settings.llm.primary
    assert primary.name == "main"
    assert primary.base_url == "https://example.test/v1"
    assert primary.api_key.get_secret_value() == "sk-unit-test"
    assert primary.model == "my-model"
    assert primary.timeout == 12.5
    assert primary.max_retries == 4
    assert primary.temperature == 0.7
    assert primary.max_tokens == 256


def test_fallbacks_parsed_from_json(env: pytest.MonkeyPatch) -> None:
    settings = _load(
        env,
        LLM_PRIMARY_API_KEY="sk-primary",
        LLM_FALLBACKS=(
            '[{"name": "backup-a", "model": "m-a", "api_key": "sk-a"},'
            ' {"name": "backup-b", "model": "m-b", "api_key": "sk-b"}]'
        ),
    )

    assert [p.name for p in settings.llm.fallbacks] == ["backup-a", "backup-b"]
    # 降级链顺序：主模型在前，降级模型按声明顺序
    assert [p.name for p in settings.llm.profile_chain()] == ["primary", "backup-a", "backup-b"]
    # JSON 中未给出的字段回落到 LLMProfile 的默认值
    assert settings.llm.fallbacks[0].max_retries == 2


# ============================================================ 缺失必填项


def test_missing_api_key_raises_error_naming_the_env_var(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(env)

    message = str(excinfo.value)
    assert "LLM_PRIMARY_API_KEY" in message
    assert "LLM_USE_MOCK" in message, "错误信息应给出可行的替代方案"


def test_use_mock_waives_api_key_requirement(env: pytest.MonkeyPatch) -> None:
    settings = _load(env, LLM_USE_MOCK="true")

    assert settings.llm.use_mock is True
    assert settings.llm.primary.has_api_key() is False


def test_prod_requires_secret_key(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(env, LLM_USE_MOCK="true", APP_ENV="prod", APP_DEBUG="false")

    assert "APP_SECRET_KEY" in str(excinfo.value)


def test_prod_requires_debug_off(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(env, LLM_USE_MOCK="true", APP_ENV="prod", APP_SECRET_KEY="s3cret", APP_DEBUG="true")

    assert "APP_DEBUG" in str(excinfo.value)


def test_prod_accepts_hardened_config(env: pytest.MonkeyPatch) -> None:
    settings = _load(
        env,
        LLM_USE_MOCK="true",
        APP_ENV="prod",
        APP_SECRET_KEY="s3cret",
        APP_DEBUG="false",
    )

    assert settings.is_prod is True


# ============================================================ 取值非法


@pytest.mark.parametrize(
    ("env_key", "bad_value"),
    [
        ("CONTEXT_MAX_TOTAL_TOKENS", "not-a-number"),
        ("CONTEXT_MAX_TOTAL_TOKENS", "0"),
        ("MEMORY_SESSION_TTL", "-1"),
        ("TOOL_MAX_RESULT_ROWS", "0"),
        ("TASK_MAX_STEPS", "0"),
        ("APP_LOG_LEVEL", "VERBOSE"),
        ("STORE_BACKEND", "postgres"),
        ("LLM_PRIMARY_TEMPERATURE", "3.5"),
    ],
)
def test_invalid_value_reports_the_offending_env_var(
    env: pytest.MonkeyPatch, env_key: str, bad_value: str
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(env, LLM_USE_MOCK="true", **{env_key: bad_value})

    assert env_key in str(excinfo.value)


def test_budget_ratios_must_sum_to_one(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(
            env,
            LLM_USE_MOCK="true",
            CONTEXT_SUMMARY_BUDGET_RATIO="0.5",
            CONTEXT_CONTENT_BUDGET_RATIO="0.5",
            CONTEXT_ARTIFACT_BUDGET_RATIO="0.5",
        )

    message = str(excinfo.value)
    assert "CONTEXT_SUMMARY_BUDGET_RATIO" in message
    assert "1.5" in message, "错误信息应给出实际求和结果便于排查"


def test_es_credentials_must_be_paired(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(env, LLM_USE_MOCK="true", ES_USERNAME="elastic")

    assert "ES_PASSWORD" in str(excinfo.value)


def test_es_blank_credentials_are_treated_as_absent(env: pytest.MonkeyPatch) -> None:
    settings = _load(env, LLM_USE_MOCK="true", ES_USERNAME="", ES_DEFAULT_INDEX="")

    assert settings.es.username is None
    assert settings.es.default_index is None


def test_redis_backend_requires_a_url(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(env, LLM_USE_MOCK="true", STORE_BACKEND="redis", STORE_REDIS_URL="   ")

    assert "STORE_REDIS_URL" in str(excinfo.value)


def test_error_inside_group_payload_reports_group_prefix(env: pytest.MonkeyPatch) -> None:
    """错误由聚合根报出时，定位路径首段是子配置字段名，须换用该组的环境变量前缀。"""
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=None, llm={"use_mock": "not-a-bool"})

    message = str(excinfo.value)
    assert "LLM_USE_MOCK" in message
    assert "APP_LLM" not in message


def test_fallback_list_error_points_at_the_element(env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError) as excinfo:
        _load(
            env,
            LLM_PRIMARY_API_KEY="sk-primary",
            LLM_FALLBACKS='[{"name": "a", "max_tokens": -1}]',
        )

    assert "LLM_FALLBACKS[0].max_tokens" in str(excinfo.value)


# ============================================================ 按用途路由


def test_purpose_routing_falls_back_to_primary_model(env: pytest.MonkeyPatch) -> None:
    settings = _load(
        env,
        LLM_PRIMARY_API_KEY="sk-x",
        LLM_PRIMARY_MODEL="base-model",
        LLM_PLAN_MODEL="plan-model",
        # 显式留空，等同于未配置
        LLM_INTENT_MODEL="",
    )

    assert settings.llm.intent_model is None
    assert settings.llm.model_for("intent") == "base-model"
    assert settings.llm.model_for("execute") == "base-model"
    assert settings.llm.model_for("plan") == "plan-model"


# ============================================================ 派生计算


def test_token_budget_splits_by_ratio(env: pytest.MonkeyPatch) -> None:
    settings = _load(env, LLM_USE_MOCK="true", CONTEXT_MAX_TOTAL_TOKENS="10000")

    budget = settings.context.token_budget()
    assert budget == {"summary": 1500, "content": 6500, "artifacts": 2000}
    assert sum(budget.values()) == 10000


# ============================================================ 密钥保护


def test_secrets_are_masked_in_repr(env: pytest.MonkeyPatch) -> None:
    settings = _load(
        env,
        LLM_PRIMARY_API_KEY="sk-must-not-leak",
        APP_SECRET_KEY="django-must-not-leak",
    )

    dumped = repr(settings) + str(settings)
    assert "sk-must-not-leak" not in dumped
    assert "django-must-not-leak" not in dumped
    # 但取值路径仍然可用
    assert settings.llm.primary.api_key.get_secret_value() == "sk-must-not-leak"


# ============================================================ env 文件来源


def test_env_file_source_propagates_to_every_group(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """指定 env 文件时，各子配置都应读到它；置 None 时都应忽略它。

    子配置由 default_factory 构建，`_env_file` 不会自动向下传递，
    故 load_settings 显式下传——本用例守护该行为不被回退。
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_USE_MOCK=true",
                "LLM_PRIMARY_MODEL=from-file",
                "CONTEXT_MAX_TOTAL_TOKENS=4321",
                "TASK_MAX_STEPS=3",
                "APP_LOG_LEVEL=ERROR",
            ]
        ),
        encoding="utf-8",
    )

    from_file = load_settings(env_file=env_file)
    assert from_file.llm.primary.model == "from-file"
    assert from_file.context.max_total_tokens == 4321
    assert from_file.task.max_steps == 3
    assert from_file.log_level == "ERROR"

    isolated = load_settings(env_file=None, llm=LLMSettings(use_mock=True))
    assert isolated.llm.primary.model == "gpt-4o-mini"
    assert isolated.context.max_total_tokens == 32_000
    assert isolated.task.max_steps == 10
    assert isolated.log_level == "INFO"


def test_group_override_bypasses_environment(env: pytest.MonkeyPatch) -> None:
    env.setenv("CONTEXT_MAX_TOTAL_TOKENS", "999")

    settings = load_settings(
        env_file=None,
        llm=LLMSettings(use_mock=True),
        context=ContextSettings(max_total_tokens=111),
    )

    assert settings.context.max_total_tokens == 111


# ============================================================ 单例


def test_get_settings_is_cached_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sentinel = AppSettings(llm=LLMSettings(use_mock=True))

    def fake_load(**_: object) -> AppSettings:
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr("agent.config.settings.load_settings", fake_load)
    reset_settings_cache()
    try:
        assert get_settings() is sentinel
        assert get_settings() is sentinel
        assert calls == 1, "lru_cache 应阻止重复加载"

        reset_settings_cache()
        assert get_settings() is sentinel
        assert calls == 2, "重置后应重新加载"
    finally:
        reset_settings_cache()


# ============================================================ .env.example 一致性


def _is_nested_model(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _expected_env_keys() -> set[str]:
    """由配置类反推应当出现在 .env.example 中的全部键名。"""
    keys: set[str] = set()

    for name in AppSettings.model_fields:
        if name in _GROUP_TYPES:
            continue
        keys.add(f"APP_{name.upper()}")

    prefixes: list[tuple[type[BaseModel], str]] = [
        (PrimaryLLMProfile, "LLM_PRIMARY_"),
        (LLMSettings, "LLM_"),
        (ContextSettings, "CONTEXT_"),
        (MemorySettings, "MEMORY_"),
        (ToolSettings, "TOOL_"),
        (TaskSettings, "TASK_"),
        (StoreSettings, "STORE_"),
        (ESSettings, "ES_"),
        (TraceSettings, "TRACE_"),
    ]
    for model, prefix in prefixes:
        for name, field in model.model_fields.items():
            if _is_nested_model(field.annotation):
                continue
            keys.add(f"{prefix}{name.upper()}")

    return keys


def _env_example_keys() -> set[str]:
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)=")
    keys = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            keys.add(match.group(1))
    return keys


def test_env_example_matches_settings_fields() -> None:
    """配置类与 .env.example 双向对齐，避免新增配置项忘记登记。"""
    expected = _expected_env_keys()
    documented = _env_example_keys()

    assert not expected - documented, f".env.example 缺少配置项：{sorted(expected - documented)}"
    assert not documented - expected, (
        f".env.example 存在多余配置项：{sorted(documented - expected)}"
    )


# ============================================================ 类型契约


def test_primary_profile_is_a_plain_profile() -> None:
    """主模型 profile 必须仍是 LLMProfile，才能与 JSON 注入的降级模型同构。"""
    assert issubclass(PrimaryLLMProfile, LLMProfile)
    assert isinstance(PrimaryLLMProfile(_env_file=None), LLMProfile)


def test_profile_api_key_defaults_to_empty_secret() -> None:
    profile = LLMProfile()
    assert isinstance(profile.api_key, SecretStr)
    assert profile.has_api_key() is False
