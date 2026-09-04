"""网络层测试：会话凭证、四张表落库、四个接口。

全程 MockLLM（在 ``api_manager`` fixture 里注入），不触网、不碰真实模型。
"""

import json

import pytest

from agent.config import load_settings
from agent.llm import MockLLMClient
from agent.task_manager import build_task_manager
from server.api import runtime
from server.api.listener import DbListener
from server.api.models import Message, Session, Task, ToolCall

pytestmark = pytest.mark.django_db


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


UNKNOWN_SCRIPT = [
    MockLLMClient.json_reply(
        {
            "intent": "unknown",
            "related_task_ids": [],
            "reason": "太模糊",
            "clarification": "你想查哪个服务？",
        }
    )
]


@pytest.fixture
def api_settings(log_file, tmp_path):
    """服务端配置：Mock 模型 + 临时数据文件 + 临时结果目录。"""
    return load_settings(
        env_file=None,
        llm={"use_mock": True},
        tool={"data_file": log_file},
        server={"tool_result_dir": tmp_path / "tool_results"},
    )


@pytest.fixture
def api_manager(api_settings):
    """把进程内单例换成 MockLLM 版本，只挂 DbListener（日志行为由 test_logging 覆盖）。

    返回的 helper 只能**往同一个 manager 里追加脚本**，不提供"重建 manager"——重建会
    清空内存态，正等着澄清的任务会凭空消失，那不是测试想验证的东西。
    """
    llm = MockLLMClient(QUERY_SCRIPT)
    manager = build_task_manager(
        api_settings,
        llm=llm,
        listeners=[DbListener(api_settings)],
        sleeper=lambda _s: None,
    )
    runtime.set_manager(manager)

    class Harness:
        manager = None

        def enqueue(self, *replies):
            llm.enqueue(*replies)

    harness = Harness()
    harness.manager = manager
    yield harness
    runtime.reset_manager()


@pytest.fixture
def clarify_manager(api_settings):
    """脚本以"意图不明"开头的 manager，用于澄清回合。"""
    llm = MockLLMClient(UNKNOWN_SCRIPT)
    manager = build_task_manager(
        api_settings, llm=llm, listeners=[DbListener(api_settings)], sleeper=lambda _s: None
    )
    runtime.set_manager(manager)
    yield llm
    runtime.reset_manager()


def fresh_client():
    """一个不带任何 Cookie 的客户端——Django 的测试客户端会自动保存 Set-Cookie，
    想模拟"另一个用户"就必须换一个实例。"""
    from django.test import Client

    return Client()


def post_chat(client, message, **kwargs):
    return client.post(
        "/api/chat",
        data=json.dumps({"message": message}),
        content_type="application/json",
        **kwargs,
    )


# ================================================================ 会话凭证


class TestSessionCredential:
    def test_first_request_issues_a_credential(self, client, api_manager):
        response = client.get("/api/health")

        assert response.status_code == 200
        session_id = response["X-Session-Id"]
        assert session_id.startswith("sess_")
        # header / cookie / body 三处都给，客户端用哪种都行
        assert response.cookies["agent_session"].value == session_id
        assert response.json()["session_id"] == session_id
        assert Session.objects.filter(session_id=session_id).exists()

    def test_header_credential_is_reused(self, client, api_manager):
        first = client.get("/api/health")["X-Session-Id"]
        second = client.get("/api/health", headers={"X-Session-Id": first})["X-Session-Id"]

        assert second == first
        assert Session.objects.count() == 1

    def test_cookie_credential_is_reused(self, client, api_manager):
        """浏览器不会自己带 header，Cookie 这条路必须同样有效。"""
        first = client.get("/api/health")["X-Session-Id"]
        client.cookies["agent_session"] = first

        assert client.get("/api/health")["X-Session-Id"] == first
        assert Session.objects.count() == 1

    def test_header_wins_over_cookie(self, client, api_manager):
        cookie_session = client.get("/api/health")["X-Session-Id"]
        client.cookies["agent_session"] = cookie_session
        other = client.get("/api/health", headers={"X-Session-Id": "sess_deadbeefcafe"})

        assert other["X-Session-Id"] == "sess_deadbeefcafe"

    def test_no_credential_creates_a_new_session_each_time(self, client, api_manager):
        """两个互不相识的客户端各自拿到一个会话。"""
        first = client.get("/api/health")["X-Session-Id"]
        second = fresh_client().get("/api/health")["X-Session-Id"]

        assert first != second
        assert Session.objects.count() == 2

    @pytest.mark.parametrize(
        "junk", ["", "  ", "not-a-session", "'; drop table sessions; --", "x" * 200]
    )
    def test_malformed_credential_is_replaced(self, client, api_manager, junk):
        """放任客户端拿任意字符串当主键，sessions 表会被垃圾键撑满。"""
        response = client.get("/api/health", headers={"X-Session-Id": junk})

        issued = response["X-Session-Id"]
        assert issued.startswith("sess_")
        assert not Session.objects.filter(session_id=junk).exists()

    def test_uuid_credential_is_accepted(self, client, api_manager):
        uuid = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
        assert client.get("/api/health", headers={"X-Session-Id": uuid})["X-Session-Id"] == uuid

    def test_session_records_user_agent_and_ip(self, client, api_manager):
        response = client.get("/api/health", headers={"User-Agent": "curl/8.0"})
        session = Session.objects.get(session_id=response["X-Session-Id"])

        assert session.user_agent == "curl/8.0"
        assert session.ip == "127.0.0.1"


# ================================================================ /api/chat


class TestChat:
    def test_returns_the_answer(self, client, api_manager):
        response = post_chat(client, "查一下错误日志")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["output"] == "共 3 条 ERROR 日志"
        assert body["intent"] == "query"
        assert body["clarifying"] is False
        assert body["task_id"].startswith("task_")

    def test_operations_and_tool_calls_are_reported(self, client, api_manager):
        body = post_chat(client, "查一下错误日志").json()

        assert body["operations"][0]["description"] == "检索错误日志"
        assert body["operations"][0]["status"] == "succeeded"
        assert body["tool_calls"] == [
            {"call_id": "call_s1", "tool_name": "search_tool", "status": "ok"}
        ]

    def test_second_turn_reuses_the_session(self, client, api_manager):
        first = post_chat(client, "查一下错误日志")
        session_id = first["X-Session-Id"]

        api_manager.enqueue(
            *[
                MockLLMClient.json_reply(
                    {
                        "intent": "analysis",
                        "related_task_ids": [first.json()["task_id"]],
                        "reason": "",
                        "clarification": "",
                    }
                ),
                MockLLMClient.json_reply(
                    {"operations": [{"description": "统计", "suggested_tool": None}]}
                ),
                MockLLMClient.reply("order-service 最多"),
                MockLLMClient.json_reply(
                    {"satisfied": True, "output": "order-service 最多", "reason": ""}
                ),
            ]
        )
        second = post_chat(client, "按服务统计刚才那批", headers={"X-Session-Id": session_id})

        assert second["X-Session-Id"] == session_id
        assert Session.objects.count() == 1
        assert Task.objects.filter(session_id=session_id).count() == 2

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("not json", "不是合法 JSON"),
            ("[1, 2]", "必须是 JSON 对象"),
            ('{"message": ""}', "不能为空"),
            ('{"message": "   "}', "不能为空"),
            ("{}", "不能为空"),
        ],
    )
    def test_bad_request_bodies(self, client, api_manager, body, expected):
        response = client.post("/api/chat", data=body, content_type="application/json")

        assert response.status_code == 400
        assert expected in response.json()["error"]
        # 出错也要带上会话凭证，客户端才能接着用
        assert response.json()["session_id"]

    def test_get_is_not_allowed(self, client, api_manager):
        assert client.get("/api/chat").status_code == 405

    def test_failed_task_returns_the_reason(self, client, api_settings):
        runtime.set_manager(
            build_task_manager(
                api_settings,
                llm=MockLLMClient(
                    [MockLLMClient.reply("不是 JSON"), MockLLMClient.reply("还不是")]
                ),
                listeners=[DbListener(api_settings)],
            )
        )
        body = post_chat(client, "查一下错误日志").json()

        assert body["status"] == "failed"
        assert "failed" in body["output"]


# ================================================================ 澄清回合


class TestClarification:
    def test_clarifying_response_asks_the_question(self, client, clarify_manager):
        body = post_chat(client, "嗯").json()

        assert body["status"] == "clarifying"
        assert body["clarifying"] is True
        assert body["output"] == "你想查哪个服务？"

    def test_next_message_continues_the_same_task(self, client, clarify_manager):
        """同一个 /api/chat 续跑：客户端不需要认识第二个端点。"""
        first = post_chat(client, "嗯")
        session_id = first["X-Session-Id"]
        task_id = first.json()["task_id"]

        clarify_manager.enqueue(*QUERY_SCRIPT)
        second = post_chat(client, "order-service 的错误", headers={"X-Session-Id": session_id})

        assert second.json()["task_id"] == task_id
        assert second.json()["status"] == "completed"
        # 全程只有一个任务
        assert Task.objects.count() == 1
        assert Task.objects.get().status == "completed"

    def test_other_sessions_are_unaffected(self, client, clarify_manager):
        """A 会话卡在澄清，不该让 B 会话的新问题被当成补充说明。"""
        stuck = post_chat(client, "嗯")["X-Session-Id"]

        clarify_manager.enqueue(*QUERY_SCRIPT)
        other = post_chat(fresh_client(), "查一下错误日志")

        assert other["X-Session-Id"] != stuck
        assert other.json()["status"] == "completed"
        assert Task.objects.count() == 2


# ================================================================ 落库


class TestPersistence:
    def test_all_four_tables_get_rows(self, client, api_manager):
        response = post_chat(client, "查一下错误日志")
        session_id = response["X-Session-Id"]
        task_id = response.json()["task_id"]

        assert Session.objects.filter(session_id=session_id).exists()
        assert Task.objects.filter(task_id=task_id, session_id=session_id).exists()
        assert Message.objects.filter(task_id=task_id).exists()
        assert ToolCall.objects.filter(task_id=task_id).exists()

    def test_task_row_carries_status_intent_and_summary_snapshot(self, client, api_manager):
        task_id = post_chat(client, "查一下错误日志").json()["task_id"]
        row = Task.objects.get(task_id=task_id)

        assert row.status == "completed"
        assert row.intent == "query"
        assert row.summary_json["output"] == "共 3 条 ERROR 日志"
        assert row.summary_json["task_id"] == task_id
        assert row.created_at <= row.updated_at

    def test_one_row_per_message_with_all_roles(self, client, api_manager):
        task_id = post_chat(client, "查一下错误日志").json()["task_id"]
        rows = list(Message.objects.filter(task_id=task_id))

        assert {row.role for row in rows} == {"system", "user", "assistant", "tool"}
        # 消息 id 唯一，一条消息一行
        assert len({row.message_id for row in rows}) == len(rows)

    def test_tool_message_points_at_its_tool_call(self, client, api_manager):
        """messages.tool_call_id 是真外键，必须指到 tool_calls 里的行。"""
        task_id = post_chat(client, "查一下错误日志").json()["task_id"]
        tool_message = Message.objects.get(task_id=task_id, role="tool")

        assert tool_message.tool_call_id == "call_s1"
        assert tool_message.tool_call.tool_name == "search_tool"

    def test_tool_call_row_has_arguments_and_result(self, client, api_manager):
        task_id = post_chat(client, "查一下错误日志").json()["task_id"]
        row = ToolCall.objects.get(task_id=task_id)

        assert row.tool_name == "search_tool"
        assert row.arguments_json == {"keyword": "ERROR"}
        assert row.status == "ok"
        assert row.result_json["total"] == 3

    def test_failed_tool_call_is_recorded(self, client, api_settings):
        runtime.set_manager(
            build_task_manager(
                api_settings,
                llm=MockLLMClient(
                    [
                        QUERY_SCRIPT[0],
                        MockLLMClient.json_reply(
                            {
                                "operations": [
                                    {"description": "统计", "suggested_tool": "analysis_tool"}
                                ]
                            }
                        ),
                        MockLLMClient.tool_reply(
                            "analysis_tool",
                            {"tool_call_id": "ghost", "group_by": "service"},
                            call_id="c1",
                        ),
                        MockLLMClient.reply("换个办法"),
                        QUERY_SCRIPT[4],
                    ]
                ),
                listeners=[DbListener(api_settings)],
            )
        )
        post_chat(client, "统计一下")

        row = ToolCall.objects.get(call_id="c1")
        assert row.status == "failed"
        assert "黑板上没有" in row.result_json["error"]

    def test_large_result_goes_to_a_file(self, client, api_settings, api_manager):
        """几百条日志塞进 JSONField 会让 select * 卡住，也没法用 sqlite3 命令行读。"""
        api_settings.server.inline_result_max_chars = 50
        post_chat(client, "查一下错误日志")

        row = ToolCall.objects.get(call_id="call_s1")
        assert row.result_ref.endswith("tool_results/call_s1.json")
        assert row.result_json["_truncated"] is True
        stored = api_settings.server.resolved_tool_result_dir() / "call_s1.json"
        assert json.loads(stored.read_text(encoding="utf-8"))["total"] == 3

    def test_status_transitions_are_persisted_as_they_happen(self, client, clarify_manager):
        """任务行随事件更新，不是等跑完才写一次。"""
        post_chat(client, "嗯")

        # 停在澄清态时，库里已经能看到它卡在哪一步
        assert Task.objects.get().status == "clarifying"

    def test_sessions_are_isolated_in_the_db(self, client, api_manager):
        first = post_chat(client, "查一下错误日志")["X-Session-Id"]
        api_manager.enqueue(*QUERY_SCRIPT)
        second = post_chat(fresh_client(), "查一下错误日志")["X-Session-Id"]

        assert Task.objects.filter(session_id=first).count() == 1
        assert Task.objects.filter(session_id=second).count() == 1


# ================================================================ 查询接口


class TestHistory:
    def test_lists_tasks_of_this_session_only(self, client, api_manager):
        mine = post_chat(client, "查一下错误日志")
        session_id = mine["X-Session-Id"]
        api_manager.enqueue(*QUERY_SCRIPT)
        post_chat(fresh_client(), "别人的任务")  # 另一个会话

        body = client.get("/api/history", headers={"X-Session-Id": session_id}).json()

        assert body["task_count"] == 1
        assert body["tasks"][0]["task_id"] == mine.json()["task_id"]

    def test_history_includes_messages_and_tool_calls(self, client, api_manager):
        session_id = post_chat(client, "查一下错误日志")["X-Session-Id"]
        task = client.get("/api/history", headers={"X-Session-Id": session_id}).json()["tasks"][0]

        assert [m["role"] for m in task["messages"]][:2] == ["system", "user"]
        assert task["tool_calls"][0]["tool_name"] == "search_tool"

    def test_empty_session_history(self, client, api_manager):
        body = client.get("/api/history").json()
        assert body["task_count"] == 0
        assert body["tasks"] == []

    def test_history_survives_a_manager_restart(self, client, api_manager, api_settings):
        """内存态丢了，历史照样能查——这正是落库的意义。"""
        session_id = post_chat(client, "查一下错误日志")["X-Session-Id"]
        runtime.reset_manager()  # 内存态全丢
        runtime.set_manager(build_task_manager(api_settings, llm=MockLLMClient(QUERY_SCRIPT)))

        body = client.get("/api/history", headers={"X-Session-Id": session_id}).json()
        assert body["task_count"] == 1


class TestTaskDetail:
    def test_returns_summary_messages_and_tool_calls(self, client, api_manager):
        response = post_chat(client, "查一下错误日志")
        session_id = response["X-Session-Id"]
        task_id = response.json()["task_id"]

        body = client.get(f"/api/tasks/{task_id}", headers={"X-Session-Id": session_id}).json()

        assert body["task_id"] == task_id
        assert body["summary"]["intent"] == "query"
        assert len(body["messages"]) >= 4
        assert body["tool_calls"][0]["arguments"] == {"keyword": "ERROR"}

    def test_unknown_task_is_404(self, client, api_manager):
        assert client.get("/api/tasks/task_ghost").status_code == 404

    def test_other_sessions_task_is_not_visible(self, client, api_manager):
        """换个会话就看不到别人的任务。"""
        task_id = post_chat(client, "查一下错误日志").json()["task_id"]

        response = client.get(
            f"/api/tasks/{task_id}", headers={"X-Session-Id": "sess_ffffffffffff"}
        )
        assert response.status_code == 404


class TestHealth:
    def test_reports_mock_mode_and_tools(self, client, api_manager):
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["llm"]["mock"] is True
        assert body["tools"] == ["analysis_tool", "fetch_tool_result", "search_tool"]
        assert body["log_file"].endswith("log.txt")


# ================================================================ 日志联动


def test_chat_writes_task_transitions_to_the_log_file(
    client, api_settings, tmp_path, isolated_log_settings
):
    """一次 HTTP 请求应当在日志文件里留下状态转移的痕迹。"""
    from agent.logging_setup import LoggingListener, setup_logging

    api_settings.log.dir = tmp_path / "logs"
    setup_logging(api_settings, force=True)
    runtime.set_manager(
        build_task_manager(
            api_settings,
            llm=MockLLMClient(QUERY_SCRIPT),
            listeners=[LoggingListener(), DbListener(api_settings)],
        )
    )
    try:
        post_chat(client, "查一下错误日志")
        body = api_settings.log.resolved_file().read_text(encoding="utf-8")
    finally:
        runtime.reset_manager()
        # 恢复成会话级的隔离配置，别把后续用例的日志指回仓库的 logs/
        setup_logging(isolated_log_settings, force=True)

    assert "状态转移 created→intenting" in body
    assert "状态转移 validating→completed" in body
    assert "调用 search_tool" in body
    # 每个事件一行，不因为答复里有换行就散成多行
    assert all(line.count("|") >= 4 for line in body.strip().splitlines())
