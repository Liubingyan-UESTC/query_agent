"""配置中心：全部可变参数的唯一声明处。

约定：

- 环境变量名为 ``<组前缀>_<字段名>``（大写下划线），组前缀见各 Settings 类的 ``env_prefix``；
- 布尔值用 ``true/false``，列表与嵌套对象用 JSON（如 ``LLM_FALLBACKS=[{...}]``）；
- 密钥字段一律 ``SecretStr``，避免在日志与异常栈里被打印出来；
- **业务代码只通过 :func:`get_settings` 取配置，禁止直读 ``os.environ``**。

关于嵌套：pydantic-settings 的 ``env_nested_delimiter="_"`` 会把 ``LLM_API_KEY``
贪心切成 ``api.key``，与含下划线的字段名冲突。所以这里不用嵌套分隔符，改为每个配置组
各自持有 ``env_prefix``，环境变量名保持扁平可读。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.errors import ConfigError

__all__ = [
    "AppSettings",
    "ContextSettings",
    "LLMProfile",
    "LLMSettings",
    "MemorySettings",
    "TaskSettings",
    "ToolSettings",
    "get_settings",
    "load_settings",
    "reset_settings_cache",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

AppEnv = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _settings_config(prefix: str) -> SettingsConfigDict:
    """各配置组共用的加载策略，只有环境变量前缀不同。"""
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class _GroupSettings(BaseSettings):
    """各配置组的共同基类，统一提供"可指定 env 文件来源"的构造入口。

    需要 :meth:`from_env` 而不能只靠 ``default_factory``：pydantic-settings 的
    ``_env_file`` 是实例化参数，不会传递给 ``default_factory`` 构建的嵌套配置。
    不显式下传的话，测试里传的 ``env_file=None`` 只对聚合根生效，各子配置仍会去读真实
    的 ``.env``，测试结果就依赖开发者本地环境了。
    """

    @classmethod
    def from_env(cls, env_file: Path | None) -> Self:
        return cls(_env_file=env_file)


# ============================================================ LLM


class LLMProfile(BaseModel):
    """单个模型端点的连接参数。

    保持为纯数据模型，好让它同时承载两条注入路径：主模型由 ``LLM_*`` 逐字段注入，
    降级模型由 ``LLM_FALLBACKS`` 以 JSON 数组整体注入。
    """

    name: str = "primary"
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "gpt-4o-mini"
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)

    def has_api_key(self) -> bool:
        return bool(self.api_key.get_secret_value())


class PrimaryLLMProfile(_GroupSettings, LLMProfile):
    """主模型 profile，字段取值来自 ``LLM_*`` 环境变量。"""

    model_config = _settings_config("LLM_")


class LLMSettings(_GroupSettings):
    """模型层配置：主模型 + 降级链。"""

    model_config = _settings_config("LLM_")

    primary: LLMProfile = Field(default_factory=PrimaryLLMProfile)
    fallbacks: list[LLMProfile] = Field(default_factory=list)

    # 显式指定走 Mock；未配置 api_key 时也会自动回落，见 use_mock_client()
    use_mock: bool = False

    @classmethod
    def from_env(cls, env_file: Path | None) -> Self:
        """必须显式下传 ``env_file`` 给 primary，否则 ``default_factory`` 会绕开它。"""
        return cls(_env_file=env_file, primary=PrimaryLLMProfile(_env_file=env_file))

    def use_mock_client(self) -> bool:
        """是否使用 MockLLM：显式开启，或压根没配 key（零配置也能跑通控制台）。"""
        return self.use_mock or not self.primary.has_api_key()

    def profile_chain(self) -> list[LLMProfile]:
        """主模型在前、降级模型按序在后。"""
        return [self.primary, *self.fallbacks]


# ============================================================ 上下文 / 记忆


class ContextSettings(_GroupSettings):
    """上下文窗口的装配策略。"""

    model_config = _settings_config("CONTEXT_")

    history_summary_limit: int = Field(default=10, ge=0)
    """意图识别阶段投喂的历史 task summary 条数。"""

    related_content_limit: int = Field(default=6, ge=0)
    """单个关联任务注入的历史消息条数上限。"""

    recent_content_limit: int = Field(default=12, ge=1)
    """执行阶段保留的近 K 条本任务消息。"""


class MemorySettings(_GroupSettings):
    model_config = _settings_config("MEMORY_")

    max_tasks: int = Field(default=20, ge=1)
    """WorkingMemory 保留的历史任务数，超出后淘汰最旧的。"""


# ============================================================ 工具 / 任务


class ToolSettings(_GroupSettings):
    model_config = _settings_config("TOOL_")

    data_file: Path = Path("data/logs.json")
    """假数据库文件（Kibana 风格日志）。相对路径按项目根解析。"""

    max_rows: int = Field(default=200, ge=1)
    """单次检索返回的最大记录数。"""

    preview_max_rows: int = Field(default=3, ge=1)
    """回给模型的结果预览最多展示几条记录。"""

    preview_max_chars: int = Field(default=600, ge=100)
    """结果预览的字符上限，超出即截断。"""

    def resolved_data_file(self) -> Path:
        if self.data_file.is_absolute():
            return self.data_file
        return PROJECT_ROOT / self.data_file


class TaskSettings(_GroupSettings):
    """任务护栏。全部阈值集中在这里，编排层不再各自埋常量。"""

    model_config = _settings_config("TASK_")

    max_steps: int = Field(default=8, ge=1)
    """规划允许的最大步骤数，超出即熔断（ABORTED）。"""

    max_tool_calls: int = Field(default=16, ge=1)
    """单个任务允许的工具调用总次数。"""

    max_rounds_per_step: int = Field(default=4, ge=1)
    """执行单个步骤时"模型↔工具"往返的最大轮数。"""

    max_revalidate: int = Field(default=2, ge=0)
    """校验不通过后允许回到执行阶段补充的次数。"""

    max_retry: int = Field(default=1, ge=0)
    """执行阶段遇到可重试错误（如限流）时，任务级重试的次数。

    与 ``LLM_MAX_RETRIES`` 是两个层级：后者重试的是**单次 HTTP 调用**，这里重试的是
    **整个步骤**（重新装配提示词再问一遍）。默认给 1，避免两层相乘把耗时放大。
    """

    retry_backoff_seconds: float = Field(default=1.0, ge=0)
    """任务级重试的退避基数，第 n 次重试等待 ``base * 2^(n-1)`` 秒。"""


# ============================================================ 聚合根


class AppSettings(BaseSettings):
    """全部配置的聚合根，业务代码只认这一个入口。"""

    model_config = _settings_config("APP_")

    env: AppEnv = "dev"
    debug: bool = True
    log_level: LogLevel = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    tool: ToolSettings = Field(default_factory=ToolSettings)
    task: TaskSettings = Field(default_factory=TaskSettings)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


def load_settings(*, env_file: Path | None = ENV_FILE, **overrides: Any) -> AppSettings:
    """构建配置。``env_file=None`` 时只认进程环境变量与显式覆盖（测试用）。

    子配置显式用 ``from_env`` 构建，否则 ``default_factory`` 会绕开 ``env_file``。
    构建整体包在一个 try 里：子配置的校验错误（如 ``TASK_MAX_STEPS=0``）同样要收成
    ``ConfigError``，否则调用方得同时接住两种异常类型。
    """
    try:
        groups: dict[str, Any] = {
            "llm": LLMSettings.from_env(env_file),
            "context": ContextSettings.from_env(env_file),
            "memory": MemorySettings.from_env(env_file),
            "tool": ToolSettings.from_env(env_file),
            "task": TaskSettings.from_env(env_file),
        }
        groups.update(overrides)
        return AppSettings(_env_file=env_file, **groups)
    except ValidationError as exc:
        raise ConfigError("配置校验失败，请检查 .env", detail=exc.errors()) from exc


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """进程级单例。业务代码统一从这里取配置。"""
    return load_settings()


def reset_settings_cache() -> None:
    """清空单例缓存（测试与热加载用）。"""
    get_settings.cache_clear()
