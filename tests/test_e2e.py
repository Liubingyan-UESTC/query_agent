"""端到端：用 MockLLM 跑通 require.md 的完整工作流，再验证控制台。"""

import json

import pytest

from agent.cli import ConsoleAgent, main
from agent.enums import IntentType, MessageRole, TaskStatus
from agent.errors import ToolError
from agent.llm import MockLLMClient
from agent.task_manager import build_task_manager

pytestmark = pytest.mark.e2e


@pytest.fixture
def manager(settings):
    return build_task_manager(settings, llm=MockLLMClient([]))


def enqueue_query_task(llm):
    """任务一：查 ERROR 日志（意图 → 规划 → 工具 → 结论 → 校验）。"""
    llm.enqueue(
        MockLLMClient.json_reply(
            {"intent": "query", "related_task_ids": [], "reason": "检索日志", "clarification": ""}
        ),
        MockLLMClient.json_reply(
            {
                "operations": [
                    {"description": "按关键字 ERROR 检索", "suggested_tool": "search_tool"}
                ]
            }
        ),
        MockLLMClient.tool_reply("search_tool", {"keyword": "ERROR"}, call_id="call_search"),
        MockLLMClient.reply("检索到 3 条 ERROR 日志，集中在 order-service 与 payment-service。"),
        MockLLMClient.json_reply(
            {
                "satisfied": True,
                "output": "共 3 条 ERROR 日志：order-service 2 条，payment-service 1 条。",
                "reason": "已给出具体数字",
            }
        ),
    )


def enqueue_analysis_task(llm, related_task_id):
    """任务二：对上一批结果按服务统计（带关联任务）。"""
    llm.enqueue(
        MockLLMClient.json_reply(
            {
                "intent": "analysis",
                "related_task_ids": [related_task_id],
                "reason": "延续上一个任务",
                "clarification": "",
            }
        ),
        MockLLMClient.json_reply(
            {
                "operations": [
                    {"description": "重新取回错误日志", "suggested_tool": "search_tool"},
                    {"description": "按 service 分组统计", "suggested_tool": "analysis_tool"},
                ]
            }
        ),
        MockLLMClient.tool_reply("search_tool", {"keyword": "ERROR"}, call_id="call_s2"),
        MockLLMClient.reply("已取回错误日志。"),
        MockLLMClient.tool_reply(
            "analysis_tool",
            {"tool_call_id": "call_s2", "group_by": "service", "metric": "count"},
            call_id="call_a2",
        ),
        MockLLMClient.reply("order-service 2 条，payment-service 1 条。"),
        MockLLMClient.json_reply(
            {
                "satisfied": True,
                "output": "按服务统计：order-service 2 条，payment-service 1 条。",
                "reason": "统计完成",
            }
        ),
    )


class TestFullWorkflow:
    def test_query_task_completes(self, manager):
        enqueue_query_task(manager.llm)
        task = manager.run(manager.create_task("查 ERROR 日志", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.intent is IntentType.QUERY
        assert "3 条 ERROR" in task.summary.output

    def test_blackboard_holds_the_three_records(self, manager):
        """require.md 的黑板三件套在真实链路里都被写满。"""
        enqueue_query_task(manager.llm)
        task = manager.run(manager.create_task("查 ERROR 日志", session_id="s1").task_id)
        window = manager.window(task.task_id)

        # 1. task summary：8 个字段都有意义
        assert window.summary.intent and window.summary.operations
        assert window.summary.result and window.summary.output
        # 2. tool_result：键是 tool_call_id，值是 {tool_name, tool_result}
        assert set(window.tool_results) == {"call_search"}
        assert window.tool_results["call_search"].tool_name == "search_tool"
        # 3. task_content：四种角色齐备
        roles = {message.role for message in window.content}
        assert roles == {
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        }

    def test_stages_are_visited_in_order(self, manager):
        enqueue_query_task(manager.llm)
        manager.run(manager.create_task("查 ERROR 日志", session_id="s1").task_id)
        assert [call.stage for call in manager.llm.calls] == [
            "intent",
            "plan",
            "execute",
            "execute",
            "validate",
        ]

    def test_working_memory_gets_all_three_histories(self, manager):
        enqueue_query_task(manager.llm)
        task = manager.run(manager.create_task("查 ERROR 日志", session_id="s1").task_id)

        working = manager.memory.working("s1")
        assert working.task_summary_history[task.task_id]["status"] == "completed"
        tool_history = working.tool_result_history[task.task_id]
        assert tool_history["call_search"]["tool_name"] == "search_tool"
        assert len(working.task_context_history[task.task_id]) == len(
            manager.window(task.task_id).content
        )

    def test_second_task_reuses_the_first_as_context(self, manager):
        enqueue_query_task(manager.llm)
        first = manager.run(manager.create_task("查 ERROR 日志", session_id="s1").task_id)

        enqueue_analysis_task(manager.llm, first.task_id)
        second = manager.run(manager.create_task("按服务统计刚才那批错误", session_id="s1").task_id)

        assert second.status is TaskStatus.COMPLETED
        assert second.summary.related_task_ids == [first.task_id]
        # 关联任务的历史对话被注入了规划阶段
        plan_request = [c for c in manager.llm.calls if c.stage == "plan"][-1]
        assert f"[关联任务 {first.task_id}]" in "\n".join(m.content for m in plan_request.messages)
        # 统计结果来自黑板上第一步的检索结果
        groups = manager.window(second.task_id).tool_results["call_a2"].tool_result["groups"]
        assert {"key": "order-service", "value": 2} in groups

    def test_related_injection_does_not_pollute_the_new_archive(self, manager):
        """注入的是别的任务的上下文，不能被当成本任务的记录再归档一遍。"""
        enqueue_query_task(manager.llm)
        first = manager.run(manager.create_task("查 ERROR 日志", session_id="s1").task_id)
        enqueue_analysis_task(manager.llm, first.task_id)
        second = manager.run(manager.create_task("按服务统计刚才那批", session_id="s1").task_id)

        archived = manager.memory.working("s1").context_of(second.task_id)
        assert all("[关联任务" not in message.content for message in archived)


class TestZeroConfigHeuristic:
    """不写脚本，全靠 MockLLM 的启发式应答——控制台零配置演示的真实路径。"""

    def test_query_runs_end_to_end(self, manager):
        task = manager.run(
            manager.create_task("查一下 order-service 的错误日志", session_id="s1").task_id
        )
        assert task.status is TaskStatus.COMPLETED
        assert manager.window(task.task_id).tool_results

    def test_analysis_runs_end_to_end(self, manager):
        task = manager.run(manager.create_task("按服务统计错误数量", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.intent is IntentType.ANALYSIS
        tool_names = {r.tool_name for r in manager.window(task.task_id).tool_results.values()}
        assert tool_names == {"search_tool", "analysis_tool"}

    def test_chat_runs_without_tools(self, manager):
        task = manager.run(manager.create_task("你好，你能做什么", session_id="s1").task_id)

        assert task.status is TaskStatus.COMPLETED
        assert task.summary.intent is IntentType.CHAT
        assert manager.window(task.task_id).tool_results == {}


class TestConsole:
    def make_console(self, manager, **kwargs):
        printed: list[str] = []
        console = ConsoleAgent(manager, session_id="s1", printer=printed.append, **kwargs)
        return console, printed

    def test_single_turn_prints_the_answer(self, manager):
        enqueue_query_task(manager.llm)
        console, printed = self.make_console(manager)
        console.emit(console.render(console.ask("查 ERROR 日志")))
        assert "✅" in printed[-1]
        assert "3 条 ERROR" in printed[-1]

    def test_unfinished_task_is_reported_with_its_status(self, manager):
        manager.llm.enqueue(MockLLMClient.reply("不是 JSON"), MockLLMClient.reply("还不是"))
        console, printed = self.make_console(manager)
        console.emit(console.render(console.ask("查日志")))
        assert "⚠️" in printed[-1]
        assert "failed" in printed[-1]

    def test_clarification_round_trip(self, manager):
        """意图不明 → 控制台提问 → 用户补充 → 同一个任务接着跑完。"""
        manager.llm.enqueue(
            MockLLMClient.json_reply(
                {
                    "intent": "unknown",
                    "related_task_ids": [],
                    "reason": "太短",
                    "clarification": "你想查哪个服务？",
                }
            )
        )
        console, printed = self.make_console(manager)

        first = console.ask("嗯")
        console.emit(console.render(first))
        assert first.status is TaskStatus.CLARIFYING
        assert "❓ 你想查哪个服务？" in printed[-1]

        enqueue_query_task(manager.llm)
        second = console.ask("order-service 的错误日志")
        assert second.task_id == first.task_id
        assert second.status is TaskStatus.COMPLETED

    def test_repl_consumes_lines_and_exits(self, manager):
        enqueue_query_task(manager.llm)
        console, printed = self.make_console(manager)
        console.run_forever(["查 ERROR 日志", "", "/exit", "永远读不到"])

        transcript = "\n".join(printed)
        assert "日志查询 Agent · 控制台" in transcript
        assert "3 条 ERROR" in transcript
        assert transcript.rstrip().endswith("再见。")

    def test_repl_stops_at_end_of_input(self, manager):
        console, printed = self.make_console(manager)
        console.run_forever([])
        assert printed[-1] == "再见。"

    def test_history_command_lists_finished_tasks(self, manager):
        enqueue_query_task(manager.llm)
        console, printed = self.make_console(manager)
        console.run_forever(["查 ERROR 日志", "/history", "/exit"])
        assert "1. [completed] 查 ERROR 日志" in "\n".join(printed)

    def test_help_and_unknown_commands(self, manager):
        console, printed = self.make_console(manager)
        console.run_forever(["/help", "/nope", "/exit"])
        transcript = "\n".join(printed)
        assert "/history" in transcript
        assert "未知命令：/nope" in transcript

    def test_verbose_toggle_shows_steps_and_tools(self, manager):
        enqueue_query_task(manager.llm)
        console, printed = self.make_console(manager, verbose=True)
        console.emit(console.render(console.ask("查 ERROR 日志")))

        detail = printed[-1]
        assert "意图：query" in detail
        assert "步骤1 [succeeded]" in detail
        assert "工具 search_tool(call_search) → ok" in detail

    def test_mock_notice_is_shown_when_no_api_key(self, manager):
        console, printed = self.make_console(manager)
        console.run_forever(["/exit"])
        assert "MockLLM" in printed[1]

    def test_session_is_shared_across_turns(self, manager):
        """同一个控制台会话里，第二个问题能看到第一个任务的历史摘要。"""
        enqueue_query_task(manager.llm)
        console, _ = self.make_console(manager)
        first = console.ask("查 ERROR 日志")

        enqueue_analysis_task(manager.llm, first.task_id)
        console.ask("按服务统计刚才那批错误")

        intent_request = [c for c in manager.llm.calls if c.stage == "intent"][-1]
        payload = json.loads(intent_request.messages[-1].content)
        assert [item["task_id"] for item in payload["history"]] == [first.task_id]


class TestEntryPoint:
    """`python -m agent.cli` 的命令行入口。"""

    def test_single_query_mode_prints_and_exits(self, settings, monkeypatch, capsys):
        monkeypatch.setattr("agent.cli.get_settings", lambda: settings)
        assert main(["--query", "查一下 ERROR 日志"]) == 0

        out = capsys.readouterr().out
        assert "✅" in out

    def test_verbose_flag_adds_detail(self, settings, monkeypatch, capsys):
        monkeypatch.setattr("agent.cli.get_settings", lambda: settings)
        main(["--query", "查一下 ERROR 日志", "--verbose"])
        assert "意图：query" in capsys.readouterr().out

    def test_repl_mode_reads_stdin(self, settings, monkeypatch, capsys):
        monkeypatch.setattr("agent.cli.get_settings", lambda: settings)
        monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))
        assert main([]) == 0
        assert "再见。" in capsys.readouterr().out

    def test_agent_error_during_a_turn_is_reported_not_raised(self, manager):
        """一轮问答炸了不该把整个 REPL 带走。"""
        printed: list[str] = []
        console = ConsoleAgent(manager, session_id="s1", printer=printed.append)

        def boom(*_args, **_kwargs):
            raise ToolError("数据文件不见了")

        console.manager.create_task = boom
        console.run_forever(["查日志", "/exit"])
        assert any("数据文件不见了" in line for line in printed)
