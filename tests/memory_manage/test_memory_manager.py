"""步骤 12：MemoryManager 聚合 working(session_id) 与 knowledge。"""

from pathlib import Path

from agent.common.enums import IntentType
from agent.memory_manage.knowledge_memory import KnowledgeMemory
from agent.memory_manage.memory_manager import MemoryManager
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task import Task
from agent.store.memory_store import MemoryStore
from tests.memory_manage.test_knowledge_memory import _clone


def test_facade_exposes_working_and_knowledge() -> None:
    store = MemoryStore()
    manager = MemoryManager(store)
    window = ContextWindow.from_task(Task.create("查错误", session_id="sess_1"))
    window.append_message(Message.user("查错误"))

    manager.working("sess_1").archive(window)

    assert manager.working("sess_1").list_task_ids() == [window.task_id]
    assert manager.knowledge.get_allowed_tools(IntentType.CHAT) == []


def test_injected_knowledge_instance_is_reused(tmp_path: Path) -> None:
    packaged = KnowledgeMemory()
    manager = MemoryManager(MemoryStore(), knowledge=packaged, knowledge_root=tmp_path)
    assert manager.knowledge is packaged


def test_knowledge_root_override(tmp_path: Path) -> None:
    dest = _clone(tmp_path)
    (dest / "system_prompts" / "base.md").write_text("自定义身份\n", encoding="utf-8")

    manager = MemoryManager(MemoryStore(), knowledge_root=dest)
    assert "自定义身份" in manager.knowledge.get_system_prompt("execute")
