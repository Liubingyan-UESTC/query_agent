"""M2 Demo：四大模块各自独立跑通（MockLLM + MockTool + 内存 Store）。

用法：在仓库根目录执行 `python scripts/demo_m2.py`
不依赖 Django，也不访问真实 LLM / ES。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.common.enums import IntentType, PromptStage
from agent.common.ids import new_session_id, new_task_id
from agent.context_manage.context_manager import ContextManager
from agent.context_manage.serializer import build_messages
from agent.llm.base import LLMRequest
from agent.llm.mock_client import MockLLMClient
from agent.memory_manage.knowledge_memory import KnowledgeMemory
from agent.memory_manage.working_memory import WorkingMemory
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task_summary import TaskSummary
from agent.store.memory_store import MemoryStore
from agent.tool_manage.base import ToolContext
from agent.tool_manage.manager import ToolManager


def demo_llm() -> str:
    client = MockLLMClient([MockLLMClient.reply('{"intent":"new_query"}')])
    response = client.chat(LLMRequest(messages=[Message.user("查昨天的错误")]))
    if "new_query" not in response.content:
        raise RuntimeError(f"LLM Demo 未得到预期回复：{response.content!r}")
    return f"Mock 回复 {response.content!r}"


def demo_memory() -> str:
    knowledge = KnowledgeMemory()
    skill = knowledge.get_skill(IntentType.NEW_QUERY)
    store = MemoryStore()
    session_id = new_session_id()
    task_id = new_task_id()
    window = ContextWindow(
        task_id=task_id,
        session_id=session_id,
        summary=TaskSummary(task_id=task_id, content="demo"),
    )
    WorkingMemory(session_id, store).archive(window)
    ids = WorkingMemory(session_id, store).list_task_ids()
    return f"skill={skill.name}；已归档 {ids}"


def demo_context() -> str:
    knowledge = KnowledgeMemory()
    mgr = ContextManager(MemoryStore())
    task_id = new_task_id()
    window = mgr.create_window(
        task_id,
        new_session_id(),
        "查昨天 query-agent 的 ERROR",
        knowledge.get_system_prompt(PromptStage.INTENT_RECOGNITION),
    )
    messages = build_messages(window, PromptStage.INTENT_RECOGNITION, knowledge)
    if not messages or messages[0]["role"] != "system":
        raise RuntimeError("Context Demo 装配结果非法")
    return f"窗口 {task_id} 装配出 {len(messages)} 条 messages"


def demo_tools() -> str:
    manager = ToolManager(use_mock=True)
    ctx = ToolContext(task_id=new_task_id(), session_id=new_session_id(), trace_id="demo")
    search = manager.invoke("search", {"index": "logs-app", "limit": 2}, ctx)
    if not search.ok or search.artifact is None:
        raise RuntimeError(f"search 失败：{search.text}")
    artifact = search.artifact
    store = {artifact.artifact_id: artifact}
    export = manager.invoke(
        "export",
        {"artifact_id": artifact.artifact_id},
        ToolContext(
            task_id=ctx.task_id,
            session_id=ctx.session_id,
            trace_id=ctx.trace_id,
            artifacts_reader=store.__getitem__,
        ),
    )
    if not export.ok:
        raise RuntimeError(f"export 失败：{export.text}")
    names = sorted(tool.name for tool in manager.list_tools())
    return f"工具 {names}；search 行数={artifact.row_count}；export ok"


def main() -> int:
    print("M2 Demo：四大模块独立跑通")
    for title, fn in (
        ("LLM Layer", demo_llm),
        ("MemoryManager", demo_memory),
        ("ContextManager", demo_context),
        ("ToolManager", demo_tools),
    ):
        print(f"  [ok] {title}: {fn()}")
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
