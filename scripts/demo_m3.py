"""M3 Demo：MockLLM + MockTool 跑通 created → completed，并演示关联任务。

用法：在仓库根目录执行 `python scripts/demo_m3.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.common.enums import ContextScope, TaskStatus
from agent.common.ids import new_session_id
from agent.config.settings import LLMSettings, load_settings
from agent.context_manage.context_manager import ContextManager
from agent.llm.base import LLMResponse, TokenUsage
from agent.llm.mock_client import MockLLMClient
from agent.memory_manage.memory_manager import MemoryManager
from agent.models.message import ToolCall
from agent.prompt.assembler import PromptAssembler
from agent.store.memory_store import MemoryStore
from agent.task_manage.task_manager import TaskManager
from agent.tool_manage.manager import ToolManager


def _tool_reply(limit: int, call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="search",
                arguments=json.dumps({"index": "logs-app", "limit": limit}),
            )
        ],
        finish_reason="tool_calls",
        usage=TokenUsage(),
        model="mock",
        latency_ms=0.0,
    )


def _script(related: list[str] | None = None) -> list[object]:
    return [
        MockLLMClient.reply(
            json.dumps(
                {
                    "intent": "new_query",
                    "related_task_ids": related or [],
                    "reason": "demo",
                    "confidence": 0.9,
                }
            )
        ),
        MockLLMClient.reply(
            '{"operations":[{"index":0,"name":"查日志","tool":"search",'
            '"args":{"index":"logs-app","limit":2},"expect":"表"}]}'
        ),
        _tool_reply(2, "demo_call"),
        MockLLMClient.reply('{"status":"done","conclusion":"已取回日志"}'),
        MockLLMClient.reply(
            '{"satisfied":true,"missing":[],"suggestion":"","final_output":"查询完成"}'
        ),
    ]


def _manager(llm: MockLLMClient, store: MemoryStore, settings: object) -> TaskManager:
    memory = MemoryManager(store)
    context = ContextManager(store, settings)  # type: ignore[arg-type]
    tools = ToolManager(use_mock=True, allowed_tools_lookup=memory.knowledge.get_allowed_tools)
    assembler = PromptAssembler(memory.knowledge, context, settings)  # type: ignore[arg-type]
    return TaskManager(
        store=store,
        settings=settings,  # type: ignore[arg-type]
        context_manager=context,
        memory_manager=memory,
        tool_manager=tools,
        prompt_assembler=assembler,
        llm=llm,
    )


def main() -> int:
    settings = load_settings(env_file=None, llm=LLMSettings(use_mock=True))
    store = MemoryStore()
    session = new_session_id()
    print("M3 Demo：单任务闭环 + 关联任务")

    first = _manager(MockLLMClient(_script()), store, settings)
    one = first.run(first.create_task(session, "查昨天的 ERROR").task_id)
    if one.status is not TaskStatus.COMPLETED:
        raise RuntimeError(f"首个任务未完成：{one.status} {one.error}")
    print(f"  [ok] 任务1 {one.task_id} → {one.status.value}；output={one.summary.output}")

    second = _manager(MockLLMClient(_script(related=[one.task_id])), store, settings)
    two = second.run(second.create_task(session, "继续看刚才的结果").task_id)
    if two.status is not TaskStatus.COMPLETED:
        raise RuntimeError(f"第二个任务未完成：{two.status} {two.error}")
    window = second.context.load_window(two.task_id)
    related = [item for item in window.content if item.scope is ContextScope.RELATED]
    archived = second.memory.working(session).get_content_history(two.task_id)
    if not related:
        raise RuntimeError("第二个任务没有注入关联上下文")
    if any(item.scope is ContextScope.RELATED for item in archived):
        raise RuntimeError("关联上下文被重复归档")
    print(
        f"  [ok] 任务2 {two.task_id} 关联 {two.summary.related_task_ids}；"
        f"注入 {len(related)} 条 RELATED，归档未带入"
    )
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
