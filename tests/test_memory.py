"""记忆测试：三段历史的归档、回投与淘汰。"""

import pytest

from agent.context import ContextManager, ContextWindow
from agent.enums import IntentType, TaskStatus
from agent.knowledge import KIBANA_FIELDS, KnowledgeMemory
from agent.memory import MemoryManager, WorkingMemory
from agent.models import Message, Task, TaskSummary
from agent.tools.base import ToolResult


@pytest.fixture
def manager(settings):
    return ContextManager(settings, KnowledgeMemory())


def finished_window(manager: ContextManager, content: str, *, session: str = "sess_1"):
    """造一个"跑完了"的窗口：有对话、有工具结果、状态是终态。"""
    task = Task.create(content, session_id=session)
    window = manager.create_window(task)
    manager.update_summary(window, intent=IntentType.QUERY, output=f"{content} 的答复")
    manager.record_tool_result(
        window,
        tool_call_id=f"call_{task.task_id[-4:]}",
        tool_name="search_tool",
        result=ToolResult.success({"records": [{"level": "ERROR"}], "total": 1}, "命中 1 条"),
    )
    manager.append_message(window, Message.assistant("已完成检索"))
    for status in (
        TaskStatus.INTENTING,
        TaskStatus.PLANNING,
        TaskStatus.EXECUTING,
        TaskStatus.VALIDATING,
        TaskStatus.COMPLETED,
    ):
        task.transition_to(status)
    return task, window


class TestArchive:
    def test_archives_into_three_histories(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)

        working = memory.working("sess_1")
        assert task.task_id in working.task_summary_history
        assert task.task_id in working.tool_result_history
        assert task.task_id in working.task_context_history

    def test_summary_history_keeps_the_eight_fields(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)

        archived = memory.working("sess_1").task_summary_history[task.task_id]
        assert set(archived) == set(TaskSummary.model_fields)
        assert archived["status"] == "completed"
        assert archived["output"] == "查错误日志 的答复"

    def test_tool_result_history_keeps_full_payload(self, manager):
        """归档的是全量结果，不是给模型看的预览。"""
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)

        archived = memory.working("sess_1").tool_results_of(task.task_id)
        record = next(iter(archived.values()))
        assert record["tool_name"] == "search_tool"
        assert record["tool_result"]["records"] == [{"level": "ERROR"}]

    def test_context_history_keeps_every_message(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)

        archived = memory.working("sess_1").context_of(task.task_id)
        assert len(archived) == len(window.content)
        assert archived[0].role.value == "system"

    def test_re_archiving_updates_in_place(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)
        manager.append_message(window, Message.assistant("补充一句"))
        memory.archive(window)

        working = memory.working("sess_1")
        assert working.task_ids() == [task.task_id]
        assert len(working.context_of(task.task_id)) == len(window.content)


class TestRecall:
    def test_summaries_are_ordered_oldest_first(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        ids = []
        for i in range(3):
            task, window = finished_window(manager, f"任务{i}")
            memory.archive(window)
            ids.append(task.task_id)
        assert [s["task_id"] for s in memory.working("sess_1").summaries()] == ids

    def test_summaries_limit_keeps_the_newest(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        for i in range(5):
            _, window = finished_window(manager, f"任务{i}")
            memory.archive(window)
        recent = memory.working("sess_1").summaries(limit=2)
        assert [s["content"] for s in recent] == ["任务3", "任务4"]

    def test_related_contents_returns_only_known_tasks(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)

        found = memory.working("sess_1").related_contents([task.task_id, "task_hallucinated"])
        assert list(found) == [task.task_id]

    def test_known_task_ids_supports_filtering_model_output(self, manager):
        """模型可能编造 related_task_ids，编排层用这个集合过滤。"""
        memory = MemoryManager(KnowledgeMemory())
        task, window = finished_window(manager, "查错误日志")
        memory.archive(window)
        assert memory.working("sess_1").known_task_ids() == {task.task_id}

    def test_sessions_are_isolated(self, manager):
        memory = MemoryManager(KnowledgeMemory())
        _, window_a = finished_window(manager, "A 会话的任务", session="sess_a")
        _, window_b = finished_window(manager, "B 会话的任务", session="sess_b")
        memory.archive(window_a)
        memory.archive(window_b)

        assert len(memory.working("sess_a").task_ids()) == 1
        assert len(memory.working("sess_b").task_ids()) == 1
        assert memory.sessions() == ["sess_a", "sess_b"]

    def test_reading_an_empty_session_is_safe(self):
        memory = MemoryManager(KnowledgeMemory())
        working = memory.working("brand_new")
        assert working.summaries() == []
        assert working.context_of("whatever") == []
        assert working.related_contents(["whatever"]) == {}


class TestTrim:
    def test_oldest_tasks_are_evicted_from_all_three_histories(self, manager):
        memory = MemoryManager(KnowledgeMemory(), max_tasks=2)
        ids = []
        for i in range(3):
            task, window = finished_window(manager, f"任务{i}")
            memory.archive(window)
            ids.append(task.task_id)

        working = memory.working("sess_1")
        assert working.task_ids() == ids[1:]
        for history in (
            working.task_summary_history,
            working.tool_result_history,
            working.task_context_history,
        ):
            assert ids[0] not in history

    def test_max_tasks_comes_from_settings(self, settings):
        memory = MemoryManager(KnowledgeMemory(), max_tasks=settings.memory.max_tasks)
        assert memory.working("s")._max_tasks == settings.memory.max_tasks


class TestKnowledgeMemory:
    def test_manager_exposes_knowledge(self):
        knowledge = KnowledgeMemory("- search_tool：x")
        assert MemoryManager(knowledge).knowledge is knowledge

    def test_task_system_prompt_includes_fields_for_data_intents(self):
        knowledge = KnowledgeMemory()
        for intent in (IntentType.QUERY, IntentType.ANALYSIS):
            prompt = knowledge.task_system_prompt(intent)
            assert all(field in prompt for field in KIBANA_FIELDS)

    def test_chat_prompt_stays_short(self):
        """闲聊不需要日志字段说明，省 token 也避免模型跑题。"""
        knowledge = KnowledgeMemory()
        assert "latency_ms" not in knowledge.task_system_prompt(IntentType.CHAT)

    def test_tool_catalog_flows_into_plan_prompt(self):
        knowledge = KnowledgeMemory("- search_tool：按关键字检索日志")
        assert "按关键字检索日志" in knowledge.plan_prompt(IntentType.QUERY)

    def test_prompts_demand_bare_json(self):
        knowledge = KnowledgeMemory()
        for prompt in (
            knowledge.intent_prompt(),
            knowledge.plan_prompt(IntentType.QUERY),
            knowledge.validate_prompt(),
        ):
            assert "只输出一个 JSON 对象" in prompt


def test_working_memory_is_created_lazily_per_session():
    memory = MemoryManager(KnowledgeMemory())
    first = memory.working("s1")
    assert memory.working("s1") is first
    assert isinstance(first, WorkingMemory)


def test_archive_uses_the_windows_own_session(manager):
    """归档目标由窗口自带的 session_id 决定，调用方不用再传一次。"""
    memory = MemoryManager(KnowledgeMemory())
    _, window = finished_window(manager, "任务", session="sess_x")
    memory.archive(window)
    assert isinstance(window, ContextWindow)
    assert memory.sessions() == ["sess_x"]
