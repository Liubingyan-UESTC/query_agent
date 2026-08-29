"""Query Agent —— 面向 Kibana/ES 数据查询与分析的 Agent 内核。

本包只暴露版本号，不在此处导入任何子模块：
子包之间存在严格单向依赖（common/config → models → store → llm/memory/tool
→ context/prompt → task_manage → runtime），顶层聚合导入会破坏该约束。
"""

__version__ = "1.0.0.dev0"

__all__ = ["__version__"]
