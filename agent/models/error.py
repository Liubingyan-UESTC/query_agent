"""结构化错误信息：固化 `AgentError.to_dict()` 的四个键。

`Task.error`、步骤 24 的 finalizer、步骤 26 的 HTTP 错误响应共用这一形状。
若各处各写各的键名，`retryable` 会在落盘与接口之间悄然丢失，状态机便无法分流。
"""

from typing import Any, Self

from agent.common.errors import AgentError
from agent.models.base import AgentModel

__all__ = ["ErrorInfo"]


class ErrorInfo(AgentModel):
    """与 `AgentError.to_dict()` 键集合完全一致。"""

    code: str
    message: str
    retryable: bool = False
    detail: str | None = None

    @classmethod
    def from_error(cls, error: AgentError) -> Self:
        """由异常实例构造。用 `to_dict()` 作为唯一键来源，避免两边各维护一份字段表。"""
        return cls.model_validate(error.to_dict())

    @classmethod
    def coerce(cls, value: AgentError | dict[str, Any] | Self) -> Self:
        """接受异常、字典或已是 `ErrorInfo` 的值。其余类型交给 pydantic 报错。"""
        if isinstance(value, cls):
            return value
        if isinstance(value, AgentError):
            return cls.from_error(value)
        return cls.model_validate(value)
