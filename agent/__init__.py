"""Agent 系统的顶层包。

分层依赖（单向，禁止反向 import）::

    errors / enums  →  models  →  {config, llm, tools}
                                →  {context, memory, knowledge}
                                →  task_manager  →  cli
"""

__all__ = ["__version__"]

__version__ = "1.0.0.dev0"
