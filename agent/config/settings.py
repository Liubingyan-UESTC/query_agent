"""配置中心：全部可变参数的唯一声明处。

约定：
- 环境变量命名为 `<组前缀>_<字段名>`（大写下划线），组前缀见各 Settings 类的 `env_prefix`；
- 布尔值用 `true/false`，列表与嵌套对象用 JSON（如 `ES_HOSTS=["http://localhost:9200"]`）；
- 密钥类字段统一用 `SecretStr`，避免在日志与异常栈中被打印；
- 业务代码只通过 `get_settings()` 获取配置，禁止直读 `os.environ`。

关于嵌套配置的实现选择：pydantic-settings 的 `env_nested_delimiter="_"` 会把
`LLM_PRIMARY_API_KEY` 贪心切分为 `primary.api.key`，与含下划线的字段名不兼容。
因此这里不使用嵌套分隔符，改为让每个配置组各自持有独立的 `env_prefix`，
环境变量名保持扁平可读（`LLM_PRIMARY_API_KEY` 而非 `LLM__PRIMARY__API_KEY`）。
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.common.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# LLM 的按用途路由维度；与阶段处理器一一对应。
LLMPurpose = Literal["intent", "plan", "execute"]

AppEnv = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
StoreBackend = Literal["memory", "redis"]

# 预算占比之和允许的浮点误差
_RATIO_TOLERANCE = 1e-6


def _settings_config(prefix: str) -> SettingsConfigDict:
    """生成各配置组共用的加载策略，仅环境变量前缀不同。"""
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class _GroupSettings(BaseSettings):
    """各配置组的共同基类，统一提供可指定 env 文件来源的构造入口。

    存在 `from_env` 而非直接依赖 `default_factory` 的原因：pydantic-settings 的
    `_env_file` 是实例化参数，不会向 `default_factory` 构建的嵌套配置传递。
    若不显式下传，测试传入的 `_env_file=None` 只对聚合根生效，各子配置仍会去读
    真实的 `.env`，使测试结果依赖开发者本地环境。
    """

    @classmethod
    def from_env(cls, env_file: Path | None) -> Self:
        return cls(_env_file=env_file)


# ============================================================ LLM


class LLMProfile(BaseModel):
    """单个模型端点的完整连接参数。

    保持为纯数据模型，使其能同时承载两条注入路径：主模型由 `LLM_PRIMARY_*`
    逐字段注入（见 `PrimaryLLMProfile`），降级模型由 `LLM_FALLBACKS` 以 JSON 数组整体注入。
    """

    name: str = "primary"
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "gpt-4o-mini"
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    def has_api_key(self) -> bool:
        return bool(self.api_key.get_secret_value())


class PrimaryLLMProfile(_GroupSettings, LLMProfile):
    """主模型 profile，字段取值来自 `LLM_PRIMARY_*` 环境变量。"""

    model_config = _settings_config("LLM_PRIMARY_")


class LLMSettings(_GroupSettings):
    """模型层配置：主模型、降级链与按用途的模型路由。"""

    model_config = _settings_config("LLM_")

    primary: LLMProfile = Field(default_factory=PrimaryLLMProfile)
    fallbacks: list[LLMProfile] = Field(default_factory=list)

    # 留空则回落到主模型的 model
    intent_model: str | None = None
    plan_model: str | None = None
    execute_model: str | None = None

    # 本地无密钥时置 true，走 MockLLMClient 跑通全链路
    use_mock: bool = False

    @field_validator("intent_model", "plan_model", "execute_model", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """.env 中的 `LLM_INTENT_MODEL=` 会读成空串，语义上等同于未配置。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_api_key(self) -> Self:
        if self.use_mock:
            return self
        if not self.primary.has_api_key():
            raise ValueError(
                "缺少必填环境变量 LLM_PRIMARY_API_KEY；"
                "若要在无密钥环境下跑通链路，请设置 LLM_USE_MOCK=true"
            )
        return self

    def model_for(self, purpose: LLMPurpose) -> str:
        """按用途解析实际使用的模型名，未单独配置时回落到主模型。"""
        routes: dict[LLMPurpose, str | None] = {
            "intent": self.intent_model,
            "plan": self.plan_model,
            "execute": self.execute_model,
        }
        return routes[purpose] or self.primary.model

    def profile_chain(self) -> list[LLMProfile]:
        """主模型优先、降级模型按声明顺序追加，供 ResilientLLMClient 逐个尝试。"""
        return [self.primary, *self.fallbacks]

    @classmethod
    def from_env(cls, env_file: Path | None) -> Self:
        # primary 自身也是 Settings，需把 env 文件来源继续向下传递
        return cls(_env_file=env_file, primary=PrimaryLLMProfile.from_env(env_file))


# ============================================================ 上下文


class ContextSettings(_GroupSettings):
    """Context Window 的 token 预算与历史注入上限。"""

    model_config = _settings_config("CONTEXT_")

    max_total_tokens: int = Field(default=32_000, gt=0)

    # 三者之和须为 1.0，分别对应 task_summary / task_content / task_artifacts
    summary_budget_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    content_budget_ratio: float = Field(default=0.65, gt=0.0, lt=1.0)
    artifact_budget_ratio: float = Field(default=0.20, gt=0.0, lt=1.0)

    # 意图识别阶段投喂的历史 task_summary 条数
    history_summary_limit: int = Field(default=10, ge=0)
    # 单个关联任务注入的 content 条数上限
    related_task_content_limit: int = Field(default=6, ge=0)
    # 执行阶段保留的近 K 轮对话（一轮 = user | assistant | assistant+tools）
    recent_content_limit: int = Field(default=8, ge=0)
    # 规划 / 校验阶段单条 content 摘要的字符上限
    content_digest_chars: int = Field(default=400, gt=0)

    @model_validator(mode="after")
    def _ratios_must_sum_to_one(self) -> Self:
        total = self.summary_budget_ratio + self.content_budget_ratio + self.artifact_budget_ratio
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise ValueError(
                "CONTEXT_SUMMARY_BUDGET_RATIO + CONTEXT_CONTENT_BUDGET_RATIO + "
                f"CONTEXT_ARTIFACT_BUDGET_RATIO 之和须为 1.0，当前为 {total:.6g}"
            )
        return self

    def token_budget(self) -> dict[str, int]:
        """把占比换算为绝对 token 数，供步骤 14 的裁剪策略直接消费。"""
        return {
            "summary": int(self.max_total_tokens * self.summary_budget_ratio),
            "content": int(self.max_total_tokens * self.content_budget_ratio),
            "artifacts": int(self.max_total_tokens * self.artifact_budget_ratio),
        }


# ============================================================ 记忆


class MemorySettings(_GroupSettings):
    """会话级记忆的存活时间与容量上限。"""

    model_config = _settings_config("MEMORY_")

    session_ttl: int = Field(default=3600, gt=0, description="会话空闲过期秒数")
    working_memory_max_tasks: int = Field(default=20, gt=0, description="超限后淘汰最旧任务")


# ============================================================ 工具


class ToolSettings(_GroupSettings):
    """工具调用的默认超时、结果规模与重试上限。"""

    model_config = _settings_config("TOOL_")

    default_timeout: float = Field(default=30.0, gt=0)
    max_result_rows: int = Field(default=1000, gt=0)
    max_retry: int = Field(default=2, ge=0)
    export_dir: Path = Field(default=Path("data/exports"), description="ExportTool 落盘目录")
    use_mock: bool = Field(default=False, description="为 true 时只实例化 is_mock 工具")


# ============================================================ 任务护栏


class TaskSettings(_GroupSettings):
    """任务执行的护栏上限（token 预算部分见 ContextSettings）。"""

    model_config = _settings_config("TASK_")

    max_steps: int = Field(default=10, gt=0, description="单任务 operations 步骤数上限")
    max_tool_calls: int = Field(default=20, gt=0, description="单任务累计工具调用次数上限")
    max_replan: int = Field(default=2, ge=0, description="校验不通过后的重规划次数上限")
    max_duration_seconds: float = Field(default=300.0, gt=0, description="单任务总时长上限")
    intent_min_confidence: float = Field(
        default=0.4, ge=0.0, le=1.0, description="低于此置信度降级为 CHAT"
    )
    archive_failed_content: bool = Field(
        default=False, description="FAILED/CANCELED 时是否归档 content/artifacts"
    )


# ============================================================ 存储


class StoreSettings(_GroupSettings):
    """存储后端选择。业务代码只依赖 store 抽象，切换后端不改调用方。"""

    model_config = _settings_config("STORE_")

    backend: StoreBackend = "memory"
    redis_url: str = "redis://localhost:6379/0"

    @model_validator(mode="after")
    def _redis_requires_url(self) -> Self:
        if self.backend == "redis" and not self.redis_url.strip():
            raise ValueError("STORE_BACKEND=redis 时必须提供 STORE_REDIS_URL")
        return self


# ============================================================ Elasticsearch


class ESSettings(_GroupSettings):
    """Kibana 底层 Elasticsearch 的连接参数。"""

    model_config = _settings_config("ES_")

    hosts: list[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    username: str | None = None
    password: SecretStr | None = None
    default_index: str | None = None
    verify_certs: bool = False

    @field_validator("username", "default_index", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _credentials_must_pair(self) -> Self:
        has_password = self.password is not None and bool(self.password.get_secret_value())
        if bool(self.username) != has_password:
            raise ValueError("ES_USERNAME 与 ES_PASSWORD 必须同时提供或同时留空")
        return self


# ============================================================ 可观测性


class TraceSettings(_GroupSettings):
    """任务轨迹落盘开关。默认关闭，开启后按 trace_id 写 JSONL。"""

    model_config = _settings_config("TRACE_")

    enabled: bool = False
    dir: Path = Path("logs/traces")


# ============================================================ 聚合


class AppSettings(BaseSettings):
    """全局配置聚合根。各子配置独立从自己的前缀读取环境变量。"""

    model_config = _settings_config("APP_")

    env: AppEnv = "dev"
    debug: bool = True
    log_level: LogLevel = "INFO"
    secret_key: SecretStr = SecretStr("")
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])
    cors_allowed_origins: list[str] = Field(default_factory=list)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    tool: ToolSettings = Field(default_factory=ToolSettings)
    task: TaskSettings = Field(default_factory=TaskSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    es: ESSettings = Field(default_factory=ESSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)

    @model_validator(mode="after")
    def _prod_hardening(self) -> Self:
        if self.env != "prod":
            return self
        if not self.secret_key.get_secret_value():
            raise ValueError("APP_ENV=prod 时必须提供 APP_SECRET_KEY")
        if self.debug:
            raise ValueError("APP_ENV=prod 时 APP_DEBUG 必须为 false")
        return self

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


# 校验错误定位所需的两张映射表。
# 子配置由 default_factory 构建，其 ValidationError 的 loc 是组内相对路径，
# 故需借 exc.title（模型类名）才能还原出正确的环境变量前缀。
_MODEL_ENV_PREFIX: dict[str, str] = {
    "AppSettings": "APP_",
    "LLMSettings": "LLM_",
    "PrimaryLLMProfile": "LLM_PRIMARY_",
    "ContextSettings": "CONTEXT_",
    "MemorySettings": "MEMORY_",
    "ToolSettings": "TOOL_",
    "TaskSettings": "TASK_",
    "StoreSettings": "STORE_",
    "ESSettings": "ES_",
    "TraceSettings": "TRACE_",
}

# 当错误由 AppSettings 直接报出且首段命中子配置字段名时（例如以关键字参数注入子配置），
# 应改用子配置的前缀而非 APP_。
_GROUP_ENV_PREFIX: dict[str, str] = {
    "llm": "LLM_",
    "context": "CONTEXT_",
    "memory": "MEMORY_",
    "tool": "TOOL_",
    "task": "TASK_",
    "store": "STORE_",
    "es": "ES_",
    "trace": "TRACE_",
}


def _env_var_name(title: str, loc: tuple[int | str, ...]) -> str:
    """把 pydantic 的错误定位路径还原为对应的环境变量名。"""
    if not loc:
        return ""

    prefix = _MODEL_ENV_PREFIX.get(title, "")
    parts = list(loc)
    if prefix == "APP_" and isinstance(parts[0], str) and parts[0] in _GROUP_ENV_PREFIX:
        prefix = _GROUP_ENV_PREFIX[parts[0]]
        parts = parts[1:]

    # 环境变量名到列表下标处即终止，其后的路径以 [i].field 形式续接，
    # 避免拼出 LLM_FALLBACKS_MAX_TOKENS 这种并不存在的变量名。
    head: list[str] = []
    tail: list[str] = []
    for part in parts:
        if not tail and isinstance(part, str):
            head.append(part)
        else:
            tail.append(f"[{part}]" if isinstance(part, int) else f".{part}")
    return prefix + "_".join(head).upper() + "".join(tail)


def _describe(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        env_var = _env_var_name(exc.title, err["loc"])
        # model_validator 抛出的错误没有字段定位，其 msg 已自带环境变量名
        lines.append(f"  - {env_var}: {err['msg']}" if env_var else f"  - {err['msg']}")
    return "\n".join(lines)


# 聚合根上的各子配置字段 → 对应类型，供 load_settings 统一下传 env 文件来源
_GROUP_TYPES: dict[str, type[_GroupSettings]] = {
    "llm": LLMSettings,
    "context": ContextSettings,
    "memory": MemorySettings,
    "tool": ToolSettings,
    "task": TaskSettings,
    "store": StoreSettings,
    "es": ESSettings,
    "trace": TraceSettings,
}


def load_settings(
    *,
    env_file: Path | None = ENV_FILE,
    **overrides: object,
) -> AppSettings:
    """构造配置对象；校验失败时抛出指明环境变量名的 `ConfigError`。

    各子配置在此显式构建并统一下传 `env_file`，使「是否读取 .env」成为单一开关；
    测试传 `env_file=None` 即可完全隔离开发者本地环境。
    `overrides` 可直接指定某个子配置实例，此时该组不再从环境读取。
    """
    try:
        # _env_file 是 pydantic-settings 的运行期特殊入参，与字段同经 **kwargs 传入，
        # 以便 mypy 只需在单一调用点忽略动态构参
        kwargs: dict[str, object] = {"_env_file": env_file}
        kwargs.update(
            {
                field: cls.from_env(env_file)
                for field, cls in _GROUP_TYPES.items()
                if field not in overrides
            }
        )
        kwargs.update(overrides)
        return AppSettings(**kwargs)  # type: ignore[arg-type]
    except ValidationError as exc:
        raise ConfigError(
            f"配置加载失败，共 {exc.error_count()} 项问题（请检查环境变量或 .env）",
            detail=_describe(exc),
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """进程级单例。配置在启动时一次性确定，运行期不应变化。"""
    return load_settings()


def reset_settings_cache() -> None:
    """清空单例缓存。仅供测试与开发期热加载使用。"""
    get_settings.cache_clear()
