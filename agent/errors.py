"""异常体系。

分两类，判据是"能不能靠再试一次解决"：

- ``retryable=True``——瞬时故障（限流、网络抖动、模型偶发格式错误），调用方可重试；
- ``retryable=False``——确定性错误（配置缺失、工具未注册、参数非法），重试只会重复失败。

任务编排层据此决定是转 ``RETRYING`` 还是直接进终态，因此每个子类都要把这个语义定死。
"""

from typing import Any

__all__ = [
    "AgentError",
    "ConfigError",
    "LLMError",
    "LLMResponseFormatError",
    "TaskStateError",
    "ToolError",
]


class AgentError(Exception):
    """所有业务异常的基类。"""

    code = "agent_error"
    retryable = False

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        """转为可 json 序列化的结构，用于写入 task summary 的 output。"""
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }


class ConfigError(AgentError):
    """配置缺失或非法。"""

    code = "config_error"


class LLMError(AgentError):
    """调用模型失败。``retryable`` 由构造方按 HTTP 状态码判定。"""

    code = "llm_error"

    def __init__(self, message: str, *, retryable: bool = True, detail: Any = None) -> None:
        super().__init__(message, detail=detail)
        self.retryable = retryable


class LLMResponseFormatError(LLMError):
    """模型输出无法解析为约定的结构。

    固定 ``retryable=False``：调用方已经带着"上次输出不合法"的提示重问过一次
    （见 ``agent.llm.structured.call_structured``），原样再试只会复现同一个错误。
    """

    code = "llm_response_format_error"

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message, retryable=False, detail=detail)


class ToolError(AgentError):
    """工具未注册、参数非法或执行失败。"""

    code = "tool_error"

    def __init__(self, message: str, *, retryable: bool = False, detail: Any = None) -> None:
        super().__init__(message, detail=detail)
        self.retryable = retryable


class TaskStateError(AgentError, ValueError):
    """非法的状态转移，或对终态任务继续推进。

    同时继承 ``ValueError`` 是为了兼容 require.md 中 ``TaskStatus.transition_to``
    的约定（"非法转移将抛 ValueError"），同时又能被 ``except AgentError`` 统一兜住。
    """

    code = "task_state_error"
