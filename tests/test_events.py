"""事件总线测试：TaskManager 在关键节点发出的事件序列。"""

import pytest

from agent.enums import TaskStatus
from agent.errors import LLMError
from agent.events import EventKind, TaskEvent, emit_all
from agent.llm import MockLLMClient
from agent.task_manager import build_task_manager


class Recorder:
    """把事件收进列表的监听器。"""

    def __init__(self) -> None:
        self.events: list[TaskEvent] = []

    def __call__(self, event: TaskEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [event.kind.value for event in self.events]

    def transitions(self) -> list[tuple[str, str]]:
        return [
            (event.payload["from"], event.payload["to"])
            for event in self.events
            if event.kind is EventKind.STATUS_CHANGED
        ]

    def of(self, kind: EventKind) -> list[TaskEvent]:
        return [event for event in self.events if event.kind is kind]


HAPPY = [
    MockLLMClient.json_reply(
        {"intent": "query", "related_task_ids": [], "reason": "", "clarification": ""}
    ),
    MockLLMClient.json_reply(
        {"operations": [{"description": "检索错误日志", "suggested_tool": "search_tool"}]}
    ),
    MockLLMClient.tool_reply("search_tool", {"keyword": "ERROR"}, call_id="call_s1"),
    MockLLMClient.reply("检索到 3 条"),
    MockLLMClient.json_reply({"satisfied": True, "output": "共 3 条 ERROR", "reason": ""}),
]


@pytest.fixture
def make_manager(settings):
    def _make(script, listener=None, **task_overrides):
        used = settings
        if task_overrides:
            from agent.config import load_settings

            used = load_settings(
                env_file=None,
                llm={"use_mock": True},
                tool={"data_file": settings.tool.data_file},
                task=task_overrides,
            )
        return build_task_manager(
            used,
            llm=MockLLMClient(script),
            listeners=[listener] if listener else [],
            sleeper=lambda _s: None,
        )

    return _make


class TestEventSequence:
    def test_happy_path_emits_the_whole_lifecycle(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert recorder.kinds() == [
            "task_created",
            "status_changed",  # created → intenting
            "status_changed",  # intenting → planning
            "status_changed",  # planning → executing
            "tool_called",
            "status_changed",  # executing → validating
            "status_changed",  # validating → completed
            "task_finished",
        ]

    def test_every_transition_is_reported(self, make_manager):
        """转移点全部走 _transition，所以主线的每一步都有事件。"""
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert recorder.transitions() == [
            ("created", "intenting"),
            ("intenting", "planning"),
            ("planning", "executing"),
            ("executing", "validating"),
            ("validating", "completed"),
        ]

    def test_events_carry_task_and_session(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert {e.task_id for e in recorder.events} == {task.task_id}
        assert {e.session_id for e in recorder.events} == {"s1"}

    def test_status_on_event_is_the_state_after_the_move(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        for event in recorder.of(EventKind.STATUS_CHANGED):
            assert event.status.value == event.payload["to"]

    def test_window_is_attached_for_persistence(self, make_manager):
        """落库方靠事件上的黑板取消息与摘要。"""
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        last = recorder.events[-1]
        assert last.window is not None
        assert last.window.task_id == task.task_id
        assert last.window.summary.output == "共 3 条 ERROR"

    def test_finish_event_carries_output_and_intent(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        finished = recorder.of(EventKind.TASK_FINISHED)[0]
        assert finished.status is TaskStatus.COMPLETED
        assert finished.payload == {"output": "共 3 条 ERROR", "intent": "query"}

    def test_created_event_carries_the_user_text(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        manager.create_task("查错误日志", session_id="s1")

        created = recorder.of(EventKind.TASK_CREATED)[0]
        assert created.payload["content"] == "查错误日志"
        assert created.status is TaskStatus.CREATED


class TestToolEvents:
    def test_tool_event_has_everything_needed_for_persistence(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        tool = recorder.of(EventKind.TOOL_CALLED)[0]
        assert tool.payload["call_id"] == "call_s1"
        assert tool.payload["tool_name"] == "search_tool"
        assert tool.payload["ok"] is True
        assert tool.payload["internal"] is False
        assert '"keyword": "ERROR"' in tool.payload["arguments"]
        assert tool.payload["result"]["total"] == 3

    def test_failed_tool_call_is_still_reported(self, make_manager):
        recorder = Recorder()
        manager = make_manager(
            [
                HAPPY[0],
                MockLLMClient.json_reply(
                    {"operations": [{"description": "统计", "suggested_tool": "analysis_tool"}]}
                ),
                MockLLMClient.tool_reply(
                    "analysis_tool", {"tool_call_id": "ghost", "group_by": "service"}, call_id="c1"
                ),
                MockLLMClient.reply("换个办法"),
                HAPPY[4],
            ],
            recorder,
        )
        manager.run(manager.create_task("统计", session_id="s1").task_id)

        tool = recorder.of(EventKind.TOOL_CALLED)[0]
        assert tool.payload["ok"] is False
        assert "黑板上没有" in tool.payload["error"]

    def test_internal_tool_is_flagged(self, make_manager):
        """fetch_tool_result 的结果不进黑板，落库方要靠这个标记决定去哪儿取全量。"""
        recorder = Recorder()
        manager = make_manager(
            [
                HAPPY[0],
                HAPPY[1],
                MockLLMClient.tool_reply("search_tool", {"keyword": "ERROR"}, call_id="call_s1"),
                MockLLMClient.tool_reply(
                    "fetch_tool_result", {"tool_call_id": "call_s1"}, call_id="call_f1"
                ),
                MockLLMClient.reply("看到全量了"),
                HAPPY[4],
            ],
            recorder,
        )
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        flags = {
            e.payload["tool_name"]: e.payload["internal"]
            for e in recorder.of(EventKind.TOOL_CALLED)
        }
        assert flags == {"search_tool": False, "fetch_tool_result": True}


class TestFailureEvents:
    def test_retry_emits_retrying_then_executing(self, make_manager):
        recorder = Recorder()
        manager = make_manager(
            [
                HAPPY[0],
                HAPPY[1],
                LLMError("429 限流", retryable=True),
                HAPPY[2],
                HAPPY[3],
                HAPPY[4],
            ],
            recorder,
            max_retry=1,
        )
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert ("executing", "retrying") in recorder.transitions()
        assert ("retrying", "executing") in recorder.transitions()
        retry_event = next(
            e for e in recorder.of(EventKind.STATUS_CHANGED) if e.status is TaskStatus.RETRYING
        )
        assert "429 限流" in retry_event.summary

    def test_failure_emits_finish_with_reason(self, make_manager):
        recorder = Recorder()
        manager = make_manager([MockLLMClient.reply("x"), MockLLMClient.reply("y")], recorder)
        manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        finished = recorder.of(EventKind.TASK_FINISHED)[0]
        assert finished.status is TaskStatus.FAILED
        assert "failed" in finished.payload["output"]

    def test_cancel_emits_finish(self, make_manager):
        recorder = Recorder()
        manager = make_manager(HAPPY, recorder)
        task = manager.create_task("查错误日志", session_id="s1")
        manager.cancel(task.task_id, "用户取消")

        assert recorder.of(EventKind.TASK_FINISHED)[0].status is TaskStatus.CANCELED


class TestIsolation:
    def test_listener_exception_does_not_break_the_task(self, make_manager, caplog):
        """落库挂了不能把用户的查询请求带走。"""

        def broken(_event):
            raise RuntimeError("数据库连接断了")

        manager = make_manager(HAPPY, broken)
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED

    def test_listener_exception_is_logged(self, caplog):
        def broken(_event):
            raise RuntimeError("数据库连接断了")

        with caplog.at_level("ERROR", logger="agent.events"):
            emit_all(
                (broken,),
                TaskEvent(
                    kind=EventKind.TASK_CREATED,
                    task_id="t",
                    session_id="s",
                    status=TaskStatus.CREATED,
                ),
            )
        assert "数据库连接断了" in caplog.text

    def test_one_broken_listener_does_not_block_the_others(self):
        seen: list[str] = []

        def broken(_event):
            raise RuntimeError("boom")

        emit_all(
            (broken, lambda e: seen.append(e.task_id)),
            TaskEvent(
                kind=EventKind.TASK_CREATED,
                task_id="t1",
                session_id="s",
                status=TaskStatus.CREATED,
            ),
        )
        assert seen == ["t1"]

    def test_no_listeners_is_fine(self, settings):
        manager = build_task_manager(settings, llm=MockLLMClient(HAPPY))
        task = manager.run(manager.create_task("查错误日志", session_id="s1").task_id)
        assert task.status is TaskStatus.COMPLETED
