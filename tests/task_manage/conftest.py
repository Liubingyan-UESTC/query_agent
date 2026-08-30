"""组装 TaskManager 测试夹具。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.config.settings import AppSettings, LLMSettings, TaskSettings, load_settings
from agent.context_manage.context_manager import ContextManager
from agent.llm.base import BaseLLMClient
from agent.memory_manage.memory_manager import MemoryManager
from agent.prompt.assembler import PromptAssembler
from agent.store.memory_store import MemoryStore
from agent.task_manage.task_manager import TaskManager
from agent.tool_manage.manager import ToolManager


@dataclass
class Harness:
    manager: TaskManager
    store: MemoryStore
    memory: MemoryManager
    context: ContextManager
    settings: AppSettings


def make_settings(**task_fields: object) -> AppSettings:
    task = TaskSettings(**task_fields) if task_fields else TaskSettings.from_env(None)
    return load_settings(env_file=None, llm=LLMSettings(use_mock=True), task=task)


def build_harness(llm: BaseLLMClient, *, settings: AppSettings | None = None) -> Harness:
    settings = settings or make_settings()
    store = MemoryStore()
    memory = MemoryManager(store)
    context = ContextManager(store, settings)
    tools = ToolManager(
        use_mock=True,
        allowed_tools_lookup=memory.knowledge.get_allowed_tools,
        export_dir=settings.tool.export_dir,
    )
    assembler = PromptAssembler(memory.knowledge, context, settings)
    manager = TaskManager(
        store=store,
        settings=settings,
        context_manager=context,
        memory_manager=memory,
        tool_manager=tools,
        prompt_assembler=assembler,
        llm=llm,
    )
    return Harness(manager, store, memory, context, settings)
