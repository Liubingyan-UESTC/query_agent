"""ContextManager：Context Window 生命周期、装配策略与关联任务注入。

本层依赖 common / config / models / store，不引用 llm / memory / tool / prompt。
步骤 13 交付窗口生命周期；步骤 14 再补装配与裁剪。
"""

from agent.context_manage.context_manager import (
    INDEX_SESSION,
    NS_INDEX,
    NS_WINDOW,
    SUMMARY_WRITABLE_FIELDS,
    ContextManager,
)

__all__ = [
    "INDEX_SESSION",
    "NS_INDEX",
    "NS_WINDOW",
    "SUMMARY_WRITABLE_FIELDS",
    "ContextManager",
]
