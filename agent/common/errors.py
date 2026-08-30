"""统一异常体系。

所有内部异常继承 `AgentError`，携带四项信息：

- `code`：稳定的机器可读标识。用于 HTTP 错误响应与指标打点，不随文案变化；
- `message`：面向人的简短描述；
- `retryable`：是否值得重试。这是任务状态机在 `RETRYING` 与 `FAILED` 之间分流的唯一依据，
  也是 `ResilientLLMClient` 决定是否退避重试的唯一依据；
- `detail`：补充的可读上下文（校验明细、原始响应片段等），不参与流程判断。

`retryable` 既是类级默认值也可按实例覆盖：同一类异常在不同成因下可重试性不同
（例如网络类 `LLMError` 可重试，而模型明确拒绝的请求不可重试），
无需为每种成因都新增一个异常类。
"""

from typing import Any

__all__ = [
    "AgentError",
    "AgentMemoryError",
    "ConfigError",
    "ContextError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseFormatError",
    "LLMTimeoutError",
    "StoreError",
    "TaskCanceledError",
    "TaskStateError",
    "ToolError",
    "ToolInvocationError",
    "ToolNotFoundError",
    "ToolTimeoutError",
]


class AgentError(Exception):
    """全部内部异常的根。

    `str(exc)` 会把 `detail` 一并展开，便于日志与 CLI 直接打印；
    结构化场景请用 `to_dict()`，避免把多行文本塞进单个 JSON 字段。
    """

    code: str = "agent_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        detail: str | None = None,
    ) -> None:
        self.message = message
        self.detail = detail
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        super().__init__(message if detail is None else f"{message}\n{detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"message={self.message!r}, retryable={self.retryable!r})"
        )


# ============================================================ 配置


class ConfigError(AgentError):
    """配置缺失、类型错误或取值非法。

    发生在进程启动阶段，调用方应直接退出而非降级，故恒不可重试。
    """

    code = "config_error"
    retryable = False


# ============================================================ 模型


class LLMError(AgentError):
    """模型调用失败的基类。

    默认不可重试；网络抖动等可恢复成因请在构造时显式传 `retryable=True`，
    或使用下方语义更明确的子类。
    """

    code = "llm_error"
    retryable = False


class LLMTimeoutError(LLMError):
    """模型请求超时。等价于结果未知，重试是安全的。"""

    code = "llm_timeout"
    retryable = True


class LLMRateLimitError(LLMError):
    """触发供应商限流。退避后重试或降级到 fallback 模型。"""

    code = "llm_rate_limit"
    retryable = True


class LLMResponseFormatError(LLMError):
    """模型输出不满足约定的结构化格式。

    不可重试：`call_structured` 已在内部完成「带错误信息重问」的修复尝试，
    抛出本异常意味着修复也已失败，再原样重试只会重复失败。
    """

    code = "llm_response_format"
    retryable = False


# ============================================================ 工具


class ToolError(AgentError):
    """工具相关失败的基类。"""

    code = "tool_error"
    retryable = False


class ToolNotFoundError(ToolError):
    """模型请求了未注册或未被当前意图授权的工具。

    不可重试：重试同一个不存在的名字不会有不同结果，应交由规划阶段重新选择工具。
    """

    code = "tool_not_found"
    retryable = False


class ToolInvocationError(ToolError):
    """工具执行过程中失败。

    可重试：多数成因（参数取值不当、下游瞬时故障）可通过模型修正入参后重试解决。
    """

    code = "tool_invocation"
    retryable = True


class ToolTimeoutError(ToolError):
    """工具执行超时。"""

    code = "tool_timeout"
    retryable = True


# ============================================================ 上下文与记忆


class ContextError(AgentError):
    """Context Window 操作非法，例如更新了 summary 的非白名单字段。

    不可重试：属于程序或模型输出的契约违规，重试无法改变结果。
    """

    code = "context_error"
    retryable = False


class StoreError(AgentError):
    """存储后端操作失败。

    类级默认不可重试：类型不匹配、键格式非法属于程序契约问题。
    Redis 连接/超时等瞬时故障在抛出时把 `retryable=True` 覆掉。
    """

    code = "store_error"
    retryable = False


class AgentMemoryError(AgentError):
    """记忆读写失败，包括 WorkingMemory 归档与 KnowledgeMemory 资源加载。

    命名说明：开发计划中此类名为 `MemoryError`，与 Python 内置的 `MemoryError`
    （真实内存耗尽）冲突——`except MemoryError` 的含义会随导入顺序漂移，
    且可能把 OOM 误判为记忆模块故障。故此处冠以 `Agent` 前缀加以区分。
    """

    code = "memory_error"
    retryable = False


# ============================================================ 任务


class TaskStateError(AgentError):
    """非法的状态转移，或在终态上继续推进任务。

    不可重试：属于编排逻辑缺陷，需要修代码而非重试。
    """

    code = "task_state_error"
    retryable = False


class TaskCanceledError(AgentError):
    """任务已被取消。

    并非故障而是控制流信号，但沿用异常体系以便在深层调用栈中即时中断。
    """

    code = "task_canceled"
    retryable = False
