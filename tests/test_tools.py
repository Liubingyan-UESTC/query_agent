"""工具层测试：检索、统计、全量取回与 ToolManager 的错误收口。"""

import json
from pathlib import Path

import pytest

from agent.errors import ToolError
from agent.models import ToolCall
from agent.tools import ToolContext, build_tool_manager
from agent.tools.analysis_tool import AnalysisTool
from agent.tools.base import ToolResult
from agent.tools.fetch_tool import FetchToolResultTool
from agent.tools.search_tool import SearchArgs, SearchTool


@pytest.fixture
def blackboard() -> dict:
    """模拟黑板上的 tool_result 字典。"""
    return {}


@pytest.fixture
def ctx(blackboard):
    return ToolContext(
        task_id="task_1",
        session_id="sess_1",
        read_tool_result=blackboard.get,
    )


@pytest.fixture
def manager(settings):
    return build_tool_manager(settings)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))


# ================================================================ SearchTool


class TestSearchTool:
    def test_keyword_matches_any_field(self, manager, ctx):
        result = manager.invoke(call("search_tool", keyword="order-service"), ctx)
        assert result.ok
        assert result.data["total"] == 2
        assert {r["service"] for r in result.data["records"]} == {"order-service"}

    def test_keyword_is_case_insensitive(self, manager, ctx):
        lower = manager.invoke(call("search_tool", keyword="error"), ctx)
        upper = manager.invoke(call("search_tool", keyword="ERROR"), ctx)
        assert lower.data["total"] == upper.data["total"] == 3

    def test_matches_inside_message_text(self, manager, ctx):
        result = manager.invoke(call("search_tool", keyword="deadlock"), ctx)
        assert result.data["total"] == 1
        assert "deadlock" in result.data["records"][0]["message"].lower()

    def test_matches_numeric_field(self, manager, ctx):
        """数字字段也参与匹配——模型不必知道 status_code 是数字还是字符串。"""
        result = manager.invoke(call("search_tool", keyword="504"), ctx)
        assert result.data["total"] == 1

    def test_empty_keyword_returns_everything(self, manager, ctx):
        result = manager.invoke(call("search_tool", keyword=""), ctx)
        assert result.data["total"] == 5

    def test_no_hit_is_still_a_success(self, manager, ctx):
        """查不到不是错误，模型需要知道"确实没有"。"""
        result = manager.invoke(call("search_tool", keyword="没有这种日志"), ctx)
        assert result.ok
        assert result.data["total"] == 0
        assert result.data["records"] == []

    def test_limit_truncates_and_flags(self, manager, ctx):
        result = manager.invoke(call("search_tool", keyword="", limit=2), ctx)
        assert result.data["returned"] == 2
        assert result.data["total"] == 5
        assert result.data["truncated"] is True
        assert "已截断" in result.summary

    def test_limit_is_capped_by_settings(self, log_file):
        from agent.config import load_settings

        settings = load_settings(env_file=None, tool={"data_file": log_file, "max_rows": 1})
        manager = build_tool_manager(settings)
        ctx = ToolContext("t", "s", lambda _cid: None)
        result = manager.invoke(call("search_tool", keyword="", limit=100), ctx)
        assert result.data["returned"] == 1

    def test_summary_reports_counts(self, manager, ctx):
        result = manager.invoke(call("search_tool", keyword="ERROR"), ctx)
        assert "命中 3 条" in result.summary

    def test_missing_data_file_is_a_config_error(self, tmp_path, ctx):
        from agent.config import load_settings

        settings = load_settings(env_file=None, tool={"data_file": tmp_path / "nope.json"})
        tool = SearchTool(settings.tool)
        with pytest.raises(ToolError, match="找不到数据文件"):
            tool.run(SearchArgs(keyword="x"), ctx)

    def test_malformed_data_file_is_rejected(self, tmp_path, ctx):
        from agent.config import load_settings

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        settings = load_settings(env_file=None, tool={"data_file": bad})
        with pytest.raises(ToolError, match="不是合法 JSON"):
            SearchTool(settings.tool).run(SearchArgs(keyword="x"), ctx)

    def test_data_file_without_records_is_rejected(self, tmp_path, ctx):
        from agent.config import load_settings

        bad = tmp_path / "bad.json"
        bad.write_text('{"index": "x"}', encoding="utf-8")
        settings = load_settings(env_file=None, tool={"data_file": bad})
        with pytest.raises(ToolError, match="缺少 records"):
            SearchTool(settings.tool).run(SearchArgs(keyword="x"), ctx)

    def test_real_project_data_file_is_loadable(self):
        """仓库自带的 data/logs.json 必须能被真实加载，否则控制台演示直接废掉。"""
        from agent.config import load_settings
        from agent.tools.search_tool import load_log_index

        settings = load_settings(env_file=None)
        index = load_log_index(settings.tool.resolved_data_file())
        assert len(index["records"]) >= 50
        assert {"timestamp", "level", "service", "message"} <= set(index["records"][0])


# ================================================================ AnalysisTool


class TestAnalysisTool:
    @pytest.fixture
    def searched(self, manager, ctx, blackboard):
        """先检索并把结果放上黑板，模拟执行阶段的真实顺序。"""
        result = manager.invoke(call("search_tool", keyword=""), ctx)
        blackboard["call_search"] = result.data
        return "call_search"

    def test_count_by_service(self, manager, ctx, searched):
        result = manager.invoke(
            call("analysis_tool", tool_call_id=searched, group_by="service"), ctx
        )
        assert result.ok
        assert result.data["groups"][0] == {"key": "order-service", "value": 2}
        assert result.data["analyzed_records"] == 5

    def test_count_by_level(self, manager, ctx, searched):
        result = manager.invoke(call("analysis_tool", tool_call_id=searched, group_by="level"), ctx)
        groups = {item["key"]: item["value"] for item in result.data["groups"]}
        assert groups == {"ERROR": 3, "INFO": 1, "WARN": 1}

    def test_groups_are_sorted_desc(self, manager, ctx, searched):
        result = manager.invoke(call("analysis_tool", tool_call_id=searched, group_by="level"), ctx)
        values = [item["value"] for item in result.data["groups"]]
        assert values == sorted(values, reverse=True)

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [("sum", 5000.0), ("avg", 2500.0), ("max", 3200.0), ("min", 1800.0)],
    )
    def test_numeric_metrics(self, manager, ctx, searched, metric, expected):
        result = manager.invoke(
            call(
                "analysis_tool",
                tool_call_id=searched,
                group_by="service",
                metric=metric,
                field="latency_ms",
            ),
            ctx,
        )
        groups = {item["key"]: item["value"] for item in result.data["groups"]}
        assert groups["order-service"] == expected

    def test_top_limits_group_count(self, manager, ctx, searched):
        result = manager.invoke(
            call("analysis_tool", tool_call_id=searched, group_by="service", top=2), ctx
        )
        assert len(result.data["groups"]) == 2

    def test_unknown_tool_call_id_is_a_recoverable_failure(self, manager, ctx):
        """模型填错了 id 应该能自己纠正，所以是失败结果而不是异常。"""
        result = manager.invoke(call("analysis_tool", tool_call_id="nope", group_by="service"), ctx)
        assert result.ok is False
        assert "黑板上没有" in result.error

    def test_unknown_group_field_lists_available_fields(self, manager, ctx, searched):
        result = manager.invoke(
            call("analysis_tool", tool_call_id=searched, group_by="不存在"), ctx
        )
        assert result.ok is False
        assert "可用字段" in result.error
        assert "service" in result.error

    def test_numeric_metric_requires_field(self, manager, ctx, searched):
        result = manager.invoke(
            call("analysis_tool", tool_call_id=searched, group_by="service", metric="avg"), ctx
        )
        assert result.ok is False
        assert "需要同时指定数值字段" in result.error

    def test_non_numeric_field_rejected(self, manager, ctx, searched):
        result = manager.invoke(
            call(
                "analysis_tool",
                tool_call_id=searched,
                group_by="service",
                metric="avg",
                field="message",
            ),
            ctx,
        )
        assert result.ok is False
        assert "非数值" in result.error

    def test_accepts_bare_record_list(self, ctx, blackboard):
        """结果既可能是 search 的完整包，也可能是裸记录数组。"""
        blackboard["c1"] = [{"level": "ERROR"}, {"level": "INFO"}]
        result = AnalysisTool().run(
            AnalysisTool.args_schema.model_validate({"tool_call_id": "c1", "group_by": "level"}),
            ctx,
        )
        assert result.ok
        assert result.data["analyzed_records"] == 2

    def test_summary_is_human_readable(self, manager, ctx, searched):
        result = manager.invoke(
            call("analysis_tool", tool_call_id=searched, group_by="service"), ctx
        )
        assert "按 service 统计 count" in result.summary
        assert "order-service=2" in result.summary


# ================================================================ FetchToolResultTool


class TestFetchTool:
    def test_returns_full_payload(self, manager, ctx, blackboard):
        search = manager.invoke(call("search_tool", keyword=""), ctx)
        blackboard["call_1"] = search.data

        fetched = manager.invoke(call("fetch_tool_result", tool_call_id="call_1"), ctx)
        assert fetched.ok
        assert fetched.data == search.data
        assert len(fetched.data["records"]) == 5

    def test_is_marked_internal(self, manager):
        """internal 决定了它的结果不写回黑板——require.md 的硬性要求。"""
        assert manager.is_internal("fetch_tool_result") is True
        assert manager.is_internal("search_tool") is False

    def test_missing_id_explains_why(self, manager, ctx):
        result = manager.invoke(call("fetch_tool_result", tool_call_id="ghost"), ctx)
        assert result.ok is False
        assert "自身的结果不会存入黑板" in result.error

    def test_fetching_a_fetch_is_impossible(self, manager, ctx, blackboard):
        """套娃 fetch 取不到东西——因为 fetch 的结果从不落黑板。"""
        blackboard["call_1"] = {"records": []}
        first = manager.invoke(call("fetch_tool_result", tool_call_id="call_1"), ctx)
        assert first.ok
        # 编排层不会把 first 写进黑板，所以再 fetch 一次它的 id 必然落空
        again = manager.invoke(call("fetch_tool_result", tool_call_id="call_fetch"), ctx)
        assert again.ok is False


# ================================================================ ToolManager


class TestToolManager:
    def test_registers_default_tools(self, manager):
        assert manager.names() == ["analysis_tool", "fetch_tool_result", "search_tool"]

    def test_unknown_tool_raises(self, manager, ctx):
        """模型捏造工具名属于不可恢复错误，工具清单就在提示词里。"""
        with pytest.raises(ToolError, match="未注册的工具：ghost_tool"):
            manager.invoke(call("ghost_tool"), ctx)

    def test_unknown_tool_error_lists_alternatives(self, manager, ctx):
        with pytest.raises(ToolError) as excinfo:
            manager.invoke(call("ghost_tool"), ctx)
        assert "search_tool" in str(excinfo.value)

    def test_invalid_arguments_are_recoverable(self, manager, ctx):
        """选对工具填错参数 → 失败结果回喂模型，让它自己改。"""
        result = manager.invoke(call("search_tool", limit=0), ctx)
        assert result.ok is False
        assert "参数不符合 search_tool 的定义" in result.error
        assert "keyword" in result.error  # 缺失的必填字段也点名了

    def test_malformed_json_arguments_are_recoverable(self, manager, ctx):
        result = manager.invoke(ToolCall(name="search_tool", arguments="{keyword: x}"), ctx)
        assert result.ok is False
        assert "不是合法 JSON" in result.error

    def test_tool_internal_exception_is_contained(self, manager, ctx, monkeypatch):
        """工具内部炸了不该击穿整个任务。"""

        def boom(*_args, **_kwargs):
            raise RuntimeError("磁盘满了")

        monkeypatch.setattr(manager.get("search_tool"), "run", boom)
        result = manager.invoke(call("search_tool", keyword="x"), ctx)
        assert result.ok is False
        assert "磁盘满了" in result.error

    def test_duplicate_registration_is_rejected(self, manager):
        with pytest.raises(ToolError, match="工具重名"):
            manager.register(FetchToolResultTool())

    def test_openai_schemas_are_well_formed(self, manager):
        schemas = manager.openai_schemas()
        assert [s["function"]["name"] for s in schemas] == manager.names()
        search = next(s for s in schemas if s["function"]["name"] == "search_tool")
        params = search["function"]["parameters"]
        assert params["required"] == ["keyword"]
        assert params["properties"]["limit"]["default"] == 20
        assert "$defs" not in params

    def test_schemas_carry_descriptions_for_the_model(self, manager):
        """描述是模型选对工具的唯一依据，不能为空。"""
        for schema in manager.openai_schemas():
            assert schema["function"]["description"]

    def test_catalog_lists_every_tool(self, manager):
        catalog = manager.catalog()
        for name in manager.names():
            assert name in catalog


def test_tool_result_factories():
    ok = ToolResult.success({"a": 1}, "成功")
    assert ok.ok and ok.error is None
    bad = ToolResult.failure("参数错")
    assert bad.ok is False and "参数错" in bad.summary


def test_project_data_file_exists_in_repo():
    assert Path("data/logs.json").exists(), "假数据库必须随仓库提交，否则控制台跑不起来"
