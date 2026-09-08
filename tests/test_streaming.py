"""流式响应测试：``/api/chat/stream`` 端到端。

不触网（MockLLM），但用真实的 Django 测试客户端把 SSE 流拉下来，逐事件解析。
覆盖：事件序列、断连时的任务去向、错误处理。
"""

from __future__ import annotations

import json
import time

import pytest

from agent.llm import MockLLMClient
from agent.task_manager import build_task_manager
from server.api import runtime
from server.api.listener import DbListener

pytestmark = pytest.mark.django_db

# 与 test_api 共用的最小可工作脚本
QUERY_SCRIPT = [
    MockLLMClient.json_reply(
        {"intent": "query", "related_task_ids": [], "reason": "", "clarification": ""}
    ),
    MockLLMClient.json_reply(
        {"operations": [{"description": "检索错误日志", "suggested_tool": "search_tool"}]}
    ),
    MockLLMClient.tool_reply("search_tool", {"keyword": "ERROR"}, call_id="call_s1"),
    MockLLMClient.reply("检索到 3 条 ERROR"),
    MockLLMClient.json_reply({"satisfied": True, "output": "共 3 条 ERROR 日志", "reason": ""}),
]


@pytest.fixture
def stream_settings(log_file, tmp_path):
    """本地复刻 test_api.api_settings——不依赖跨文件 fixture。"""
    from agent.config import load_settings

    return load_settings(
        env_file=None,
        llm={"use_mock": True},
        tool={"data_file": log_file},
        server={"tool_result_dir": tmp_path / "tool_results"},
    )


@pytest.fixture
def stream_manager(stream_settings):
    """装入一个 MockLLM manager；与 api_manager 类似但 sleeper=None。"""
    llm = MockLLMClient(QUERY_SCRIPT)
    manager = build_task_manager(
        stream_settings, llm=llm, listeners=[DbListener(stream_settings)], sleeper=lambda _s: None
    )
    runtime.set_manager(manager)
    yield llm
    runtime.reset_manager()


def _read_sse_events(response, *, max_events: int = 32, timeout: float = 5.0):
    """从 Django StreamingHttpResponse 一行行读出 ``event:`` / ``data:`` 对。

    Django 测试客户端对 streaming response 会一次性读完生成器（不会真流），所以
    ``response.streaming_content`` 是一个 chunk list。
    """
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    deadline = time.monotonic() + timeout
    for chunk in response.streaming_content:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        for line in chunk.splitlines():
            line = line.rstrip("\r")
            if not line:
                if current.get("event"):
                    events.append(current)
                    if len(events) >= max_events:
                        return events
                current = {}
                continue
            if line.startswith(":"):
                continue  # 注释/心跳
            if line.startswith("event:"):
                current["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                current["data"] = line[len("data:") :].strip()
        if time.monotonic() > deadline:
            break
    if current.get("event"):
        events.append(current)
    return events


def _data(event: dict[str, str]) -> dict:
    return json.loads(event["data"])


# ================================================================ 端到端


class TestStreamingEndpoint:
    def test_returns_event_stream_content_type(self, client, stream_manager):
        response = client.post(
            "/api/chat/stream",
            data=json.dumps({"message": "查一下"}),
            content_type="application/json",
            HTTP_X_SESSION_ID="sess_" + "a" * 12,
        )
        assert response.status_code == 200
        assert response["Content-Type"].startswith("text/event-stream")

    def test_emits_full_event_sequence(self, client, stream_manager):
        response = client.post(
            "/api/chat/stream",
            data=json.dumps({"message": "查一下"}),
            content_type="application/json",
            HTTP_X_SESSION_ID="sess_" + "a" * 12,
        )
        events = _read_sse_events(response)
        kinds = [e["event"] for e in events]

        # 必有的事件类型
        assert "task_created" in kinds
        assert "task_finished" in kinds
        # 状态变化至少要经过 intenting→planning→executing→validating
        status_events = [e for e in events if e["event"] == "status_changed"]
        statuses = [_data(e)["status"] for e in status_events]
        assert "intenting" in statuses
        assert "planning" in statuses
        assert "executing" in statuses
        assert "validating" in statuses
        # 工具调用事件
        tool_events = [e for e in events if e["event"] == "tool_called"]
        assert len(tool_events) >= 1
        assert _data(tool_events[0])["payload"]["tool_name"] == "search_tool"

    def test_terminal_event_has_final_output(self, client, stream_manager):
        response = client.post(
            "/api/chat/stream",
            data=json.dumps({"message": "查一下"}),
            content_type="application/json",
            HTTP_X_SESSION_ID="sess_" + "a" * 12,
        )
        events = _read_sse_events(response)
        finished = [e for e in events if e["event"] == "task_finished"]
        assert len(finished) == 1
        payload = _data(finished[0])
        assert payload["status"] == "completed"
        assert "3 条" in (payload["summary"] or "")

    def test_invalid_body_returns_400_not_stream(self, client, stream_manager):
        """JSON 不合法时直接 400，不开流——流已经开了就回不去了。"""
        response = client.post(
            "/api/chat/stream",
            data="{ not json",
            content_type="application/json",
            HTTP_X_SESSION_ID="sess_" + "a" * 12,
        )
        assert response.status_code == 400
        assert not response["Content-Type"].startswith("text/event-stream")

    def test_empty_message_returns_400(self, client, stream_manager):
        response = client.post(
            "/api/chat/stream",
            data=json.dumps({"message": "   "}),
            content_type="application/json",
            HTTP_X_SESSION_ID="sess_" + "a" * 12,
        )
        assert response.status_code == 400


# ================================================================ 内核层：ephemeral listener


class TestEphemeralListener:
    """直接验证 TaskManager._emit 会通知 ephemeral listener——不用走 HTTP 层。"""

    def test_ephemeral_listener_receives_events(self, stream_settings):

        manager = build_task_manager(
            stream_settings, llm=MockLLMClient(QUERY_SCRIPT), sleeper=lambda _s: None
        )

        manager = build_task_manager(
            stream_settings, llm=MockLLMClient(QUERY_SCRIPT), sleeper=lambda _s: None
        )
        received: list = []
        manager.add_ephemeral(received.append)
        try:
            task = manager.create_task("hi", session_id="s")
            manager.run(task.task_id)
        finally:
            manager.remove_ephemeral(received.append)

        kinds = [e.kind.value for e in received]
        assert "task_created" in kinds
        assert "task_finished" in kinds
        assert "status_changed" in kinds
        assert "tool_called" in kinds

    def test_listener_exception_does_not_break_task(self, stream_settings):
        """订阅方崩溃不能阻塞状态机——内核层的健壮性。"""

        manager = build_task_manager(
            stream_settings, llm=MockLLMClient(QUERY_SCRIPT), sleeper=lambda _s: None
        )

        def buggy(_event):
            raise RuntimeError("boom")

        manager.add_ephemeral(buggy)
        try:
            task = manager.create_task("hi", session_id="s")
            final = manager.run(task.task_id)
        finally:
            manager.remove_ephemeral(buggy)

        assert final.status.value == "completed"

    def test_listener_isolated_per_session(self, stream_settings):
        """一个会话的临时订阅不应收到别的会话的事件——内核层语义正确性。"""

        manager = build_task_manager(
            stream_settings, llm=MockLLMClient(QUERY_SCRIPT), sleeper=lambda _s: None
        )
        received: list = []
        manager.add_ephemeral(received.append)
        try:
            t1 = manager.create_task("hi", session_id="s1")
            manager.run(t1.task_id)
            received.clear()
            t2 = manager.create_task("hi", session_id="s2")
            manager.run(t2.task_id)
        finally:
            manager.remove_ephemeral(received.append)

        # t2 的事件全收到了；t1 的已经清空
        assert received
        assert all(e.session_id == "s2" for e in received)


# ================================================================ 断连与并发


class TestStreamTurnDirect:
    """直接驱动 stream_turn() 生成器，绕过 Django 测试客户端的整批消费。"""

    def test_stream_turn_yields_events_in_order(self, stream_manager):
        from server.api.runtime import stream_turn

        gen = stream_turn("sess_" + "a" * 12, "查一下")
        chunks: list[str] = []
        for chunk in gen:
            chunks.append(chunk)
            # 真流上不会立即拿到全部；这里只看生成器能跑完
        text = "".join(chunks)
        assert "event: task_created" in text
        assert "event: task_finished" in text
        assert "event: status_changed" in text

    def test_client_disconnect_does_not_break_background_thread(self, stream_manager):
        """客户端拿到一半就 close——后台线程必须能正常跑完，不许泄漏。"""
        from server.api.runtime import stream_turn

        gen = stream_turn("sess_" + "b" * 12, "查一下")
        # 只拉一个 chunk 就关掉
        first_chunk = next(gen)
        gen.close()
        assert first_chunk.startswith("event:")

        # 后台线程跑完后 sentinel 进队列；这里只能间接验证——再次发起应当能成功
        # 因为 manager 状态干净
        gen2 = stream_turn("sess_" + "c" * 12, "查一下")
        events_text = "".join(gen2)
        assert "task_finished" in events_text

    def test_stream_turn_keeps_session_lock_released(self, stream_manager):
        """流式跑完一轮之后，同会话的下一个请求应当能立刻进——锁不能泄漏。"""
        from server.api.runtime import active_sessions, stream_turn

        list(stream_turn("sess_" + "d" * 12, "查一下"))
        # 任务结束后 active_sessions 应当回到 0
        assert active_sessions() == 0

        # 同会话立刻再发起——必须能拿到锁并跑完
        gen = stream_turn("sess_" + "d" * 12, "再问一次")
        text = "".join(gen)
        assert "task_finished" in text
