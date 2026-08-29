"""配置加载失败的异常类型。

步骤 3 会建立以 `AgentError` 为根的统一异常体系；届时本类改为继承 `AgentError`
并由 `agent.common.errors` 重导出，调用方的 `except ConfigError` 无需改动。
此处不提前引用 `agent.common`，以遵守「步骤 2 只依赖步骤 1」的依赖约束。
"""


class ConfigError(Exception):
    """配置缺失、类型错误或取值非法。

    该异常发生在进程启动阶段，属于不可重试的致命错误：调用方应直接退出而非降级。
    """

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message if detail is None else f"{message}\n{detail}")
