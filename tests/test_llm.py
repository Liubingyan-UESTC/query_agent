"""LLM 层测试：全部离线，不触网。"""

import json

import pytest
from pydantic import BaseModel

from agent.config import LLMProfile, load_settings
from agent.enums import MessageRole
from agent.errors import LLMError, LLMResponseFormatError
from agent.llm import (
    LLMRequest,
    LLMResponse,
    MockLLMClient,
    OpenAICompatClient,
    ResilientLLMClient,
    build_llm_client,
    call_structured,
    extract_json,
)
from agent.llm.openai_client import strip_reasoning, translate_openai_error
from agent.models import Message


class Plan(BaseModel):
    operations: list[str]


def request(*messages: Message, stage: str | None = None, **kwargs) -> LLMRequest:
    return LLMRequest(messages=list(messages) or [Message.user("hi")], stage=stage, **kwargs)


# ================================================================ MockLLMClient


class TestMockScript:
    def test_replays_in_order(self):
        client = MockLLMClient([MockLLMClient.reply("first"), MockLLMClient.reply("second")])
        assert client.chat(request()).content == "first"
        assert client.chat(request()).content == "second"

    def test_records_requests(self):
        client = MockLLMClient([MockLLMClient.reply("ok")])
        client.chat(request(Message.user("查错误日志"), stage="intent"))
        assert client.calls[0].stage == "intent"
        assert client.recorded_messages()[0][0]["content"] == "查错误日志"

    def test_scripted_exception_is_raised(self):
        client = MockLLMClient([LLMError("429 限流", retryable=True)])
        with pytest.raises(LLMError, match="限流"):
            client.chat(request())

    def test_callable_reply_sees_the_request(self):
        client = MockLLMClient([lambda req: MockLLMClient.reply(f"stage={req.stage}")])
        assert client.chat(request(stage="plan")).content == "stage=plan"

    def test_tool_reply_builds_a_tool_call(self):
        response = MockLLMClient.tool_reply("search_tool", {"keyword": "timeout"}, call_id="c1")
        assert response.wants_tool
        assert response.tool_calls[0].parsed_arguments() == {"keyword": "timeout"}


class TestMockHeuristic:
    """脚本用尽后的兜底应答——控制台零配置演示依赖它。"""

    def test_intent_detects_analysis(self):
        client = MockLLMClient()
        payload = json.dumps(
            {"current_task": {"task_id": "t1", "content": "按服务统计错误数量"}, "history": []},
            ensure_ascii=False,
        )
        response = client.chat(request(Message.user(payload), stage="intent"))
        assert json.loads(response.content)["intent"] == "analysis"

    def test_intent_detects_chat(self):
        client = MockLLMClient()
        payload = json.dumps({"current_task": {"task_id": "t1", "content": "你好，你能做什么"}})
        response = client.chat(request(Message.user(payload), stage="intent"))
        assert json.loads(response.content)["intent"] == "chat"

    def test_intent_marks_short_input_unknown_with_clarification(self):
        client = MockLLMClient()
        payload = json.dumps({"current_task": {"task_id": "t1", "content": "嗯"}})
        body = json.loads(client.chat(request(Message.user(payload), stage="intent")).content)
        assert body["intent"] == "unknown"
        assert body["clarification"]

    def test_intent_links_related_task_on_reference_words(self):
        client = MockLLMClient()
        payload = json.dumps(
            {
                "current_task": {"task_id": "t2", "content": "继续统计刚才的结果"},
                "history": [{"task_id": "t1", "content": "查错误日志"}],
            },
            ensure_ascii=False,
        )
        body = json.loads(client.chat(request(Message.user(payload), stage="intent")).content)
        assert body["related_task_ids"] == ["t1"]

    def test_plan_returns_two_steps_for_analysis(self):
        client = MockLLMClient()
        response = client.chat(request(Message.user("按服务统计错误分布"), stage="plan"))
        operations = json.loads(response.content)["operations"]
        assert [op["suggested_tool"] for op in operations] == ["search_tool", "analysis_tool"]

    def test_execute_calls_the_suggested_tool(self):
        client = MockLLMClient()
        response = client.chat(
            request(
                Message.user("当前步骤 1/1：检索日志\n建议工具：search_tool"),
                stage="execute",
                tools=[{"type": "function", "function": {"name": "search_tool"}}],
            )
        )
        assert response.tool_calls[0].name == "search_tool"

    def test_execute_answers_directly_when_no_tool_suggested(self):
        client = MockLLMClient()
        response = client.chat(
            request(
                Message.user("当前步骤 1/1：回答用户\n建议工具：无"),
                stage="execute",
                tools=[{"type": "function", "function": {"name": "search_tool"}}],
            )
        )
        assert not response.wants_tool
        assert response.content

    def test_execute_concludes_after_a_tool_message(self):
        client = MockLLMClient()
        response = client.chat(
            request(
                Message.user("当前步骤 1/1：检索\n建议工具：search_tool"),
                Message.tool("命中 12 条", tool_call_id="c1", name="search_tool"),
                stage="execute",
                tools=[{"type": "function", "function": {"name": "search_tool"}}],
            )
        )
        assert not response.wants_tool
        assert "12 条" in response.content

    def test_execute_analysis_reuses_previous_tool_call_id(self):
        """分析步骤要指向上一步检索结果的 tool_call_id，否则拿不到数据。"""
        client = MockLLMClient()
        response = client.chat(
            request(
                Message.user("步骤 1：检索\n建议工具：search_tool"),
                Message.tool("命中 12 条", tool_call_id="call_search", name="search_tool"),
                Message.assistant("已完成检索"),
                Message.user("当前步骤 2/2：按服务统计\n建议工具：analysis_tool"),
                stage="execute",
                tools=[
                    {"type": "function", "function": {"name": "search_tool"}},
                    {"type": "function", "function": {"name": "analysis_tool"}},
                ],
            )
        )
        args = response.tool_calls[0].parsed_arguments()
        assert response.tool_calls[0].name == "analysis_tool"
        assert args["tool_call_id"] == "call_search"
        assert args["group_by"] == "service"

    def test_validate_returns_satisfied(self):
        client = MockLLMClient()
        response = client.chat(
            request(Message.assistant("共 12 条 ERROR 日志"), stage="validate"),
        )
        body = json.loads(response.content)
        assert body["satisfied"] is True
        assert "12 条" in body["output"]


# ================================================================ 结构化输出


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('前言\n```json\n{"a": 1}\n```\n后记') == {"a": 1}

    def test_unlabelled_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_embedded_braces(self):
        assert extract_json('好的，结果是 {"a": 1} 请查收') == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="空内容"):
            extract_json("   ")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="找不到 JSON"):
            extract_json("完全没有 JSON")


class TestCallStructured:
    def test_parses_into_schema(self):
        client = MockLLMClient([MockLLMClient.json_reply({"operations": ["a", "b"]})])
        plan = call_structured(client, [Message.user("plan")], Plan, stage="plan")
        assert plan.operations == ["a", "b"]

    def test_requests_json_object_format(self):
        client = MockLLMClient([MockLLMClient.json_reply({"operations": []})])
        call_structured(client, [Message.user("plan")], Plan)
        assert client.calls[0].response_format == {"type": "json_object"}

    def test_repairs_once_then_succeeds(self):
        client = MockLLMClient(
            [
                MockLLMClient.reply("我觉得应该先检索"),  # 不是 JSON
                MockLLMClient.json_reply({"operations": ["a"]}),
            ]
        )
        plan = call_structured(client, [Message.user("plan")], Plan)
        assert plan.operations == ["a"]
        # 修复轮把上次的错误回喂给了模型
        repair_prompt = client.calls[1].messages[-1].content
        assert "无法通过校验" in repair_prompt

    def test_gives_up_after_max_repair(self):
        client = MockLLMClient([MockLLMClient.reply("nope"), MockLLMClient.reply("still nope")])
        with pytest.raises(LLMResponseFormatError, match="仍未给出合法的 Plan"):
            call_structured(client, [Message.user("plan")], Plan)

    def test_schema_violation_triggers_repair(self):
        client = MockLLMClient(
            [
                MockLLMClient.json_reply({"wrong_key": 1}),
                MockLLMClient.json_reply({"operations": ["a"]}),
            ]
        )
        assert call_structured(client, [Message.user("p")], Plan).operations == ["a"]

    def test_falls_back_when_endpoint_rejects_response_format(self):
        """端点不支持 response_format 时去掉它重来，且不消耗 repair 次数。"""
        client = MockLLMClient(
            [
                LLMError("Unsupported parameter: response_format", retryable=False),
                MockLLMClient.json_reply({"operations": ["a"]}),
            ]
        )
        assert call_structured(client, [Message.user("p")], Plan).operations == ["a"]
        assert client.calls[0].response_format is not None
        assert client.calls[1].response_format is None

    def test_other_llm_errors_propagate(self):
        client = MockLLMClient([LLMError("401 unauthorized", retryable=False)])
        with pytest.raises(LLMError, match="401"):
            call_structured(client, [Message.user("p")], Plan)


# ================================================================ OpenAI 兼容层


class FakeCompletions:
    def __init__(self, payloads, errors=None):
        self._payloads = list(payloads)
        self._errors = list(errors or [])
        self.received: list[dict] = []

    def create(self, **kwargs):
        self.received.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return self._payloads.pop(0)


class FakeOpenAI:
    def __init__(self, payloads=None, errors=None):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(payloads or [], errors)


def completion(content="ok", tool_calls=None):
    return {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"content": content, "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestOpenAICompatClient:
    def make(self, payloads=None, errors=None):
        fake = FakeOpenAI(payloads, errors)
        client = OpenAICompatClient(
            LLMProfile(api_key="sk-1", model="gpt-4o-mini", temperature=0.3),
            client_factory=lambda _profile: fake,
        )
        return client, fake

    def test_parses_content_and_usage(self):
        client, _ = self.make([completion("hello")])
        response = client.chat(request(Message.user("hi")))
        assert response.content == "hello"
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "stop"

    def test_parses_tool_calls(self):
        raw = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search_tool", "arguments": '{"keyword":"x"}'},
            }
        ]
        client, _ = self.make([completion("", raw)])
        response = client.chat(request())
        assert response.tool_calls[0].name == "search_tool"

    def test_profile_supplies_defaults(self):
        client, fake = self.make([completion()])
        client.chat(request(Message.user("hi")))
        sent = fake.chat.completions.received[0]
        assert sent["model"] == "gpt-4o-mini"
        assert sent["temperature"] == 0.3
        assert "tools" not in sent

    def test_request_overrides_profile(self):
        client, fake = self.make([completion()])
        client.chat(request(temperature=0.9, model="other", max_tokens=64))
        sent = fake.chat.completions.received[0]
        assert (sent["model"], sent["temperature"], sent["max_tokens"]) == ("other", 0.9, 64)

    def test_tools_and_response_format_are_forwarded(self):
        """function calling 与 JSON 模式都必须原样传到端点，否则整条链路失效。"""
        client, fake = self.make([completion()])
        schemas = [{"type": "function", "function": {"name": "search_tool"}}]
        client.chat(request(tools=schemas, response_format={"type": "json_object"}))
        sent = fake.chat.completions.received[0]
        assert sent["tools"] == schemas
        assert sent["response_format"] == {"type": "json_object"}

    def test_close_is_delegated_to_the_sdk(self):
        closed = []
        fake = FakeOpenAI([completion()])
        fake.close = lambda: closed.append(True)
        OpenAICompatClient(LLMProfile(api_key="sk-1"), client_factory=lambda _p: fake).close()
        assert closed == [True]

    def test_empty_choices_is_retryable(self):
        client, _ = self.make([{"model": "m", "choices": []}])
        with pytest.raises(LLMError) as excinfo:
            client.chat(request())
        assert excinfo.value.retryable is True

    def test_sdk_error_is_translated(self):
        client, _ = self.make(errors=[RuntimeError("boom")])
        with pytest.raises(LLMError, match="模型调用失败"):
            client.chat(request())


class TestStripReasoning:
    """推理模型（DeepSeek-R1 / MiniMax / Qwen-thinking）会把思维链内联在 content 里。"""

    def test_paired_block_is_removed(self):
        assert strip_reasoning("<think>先查一下</think>共 12 条 ERROR") == "共 12 条 ERROR"

    def test_multiple_blocks_are_removed(self):
        assert strip_reasoning("<think>a</think>中间<think>b</think>结论") == "中间结论"

    def test_multiline_block_is_removed(self):
        raw = "<think>\n第一步…\n第二步…\n</think>\n最终答复"
        assert strip_reasoning(raw) == "最终答复"

    def test_orphan_closing_tag_drops_everything_before_it(self):
        """模型偶尔漏掉开标签，前面那一大段仍然是推理。"""
        assert strip_reasoning("啰嗦的推理过程</think>真正的答复") == "真正的答复"

    def test_plain_content_is_untouched(self):
        assert strip_reasoning("共 12 条 ERROR 日志") == "共 12 条 ERROR 日志"

    def test_empty_content(self):
        assert strip_reasoning("") == ""

    def test_reasoning_never_reaches_the_response(self):
        """走完整解析路径：思维链不该进 LLMResponse.content。"""
        fake = FakeOpenAI([completion("<think>内部推测</think>对外答复")])
        client = OpenAICompatClient(LLMProfile(api_key="sk-1"), client_factory=lambda _p: fake)
        response = client.chat(request())

        assert response.content == "对外答复"
        assert "内部推测" not in response.content


class TestErrorTranslation:
    @pytest.mark.parametrize("status", [429, 500, 503, 408])
    def test_transient_statuses_are_retryable(self, status):
        exc = type("APIStatusError", (Exception,), {"status_code": status})("boom")
        assert translate_openai_error(exc).retryable is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_are_not_retryable(self, status):
        exc = type("APIStatusError", (Exception,), {"status_code": status})("bad request")
        assert translate_openai_error(exc).retryable is False

    def test_connection_error_without_status_is_retryable(self):
        assert translate_openai_error(OSError("connection reset")).retryable is True


class TestResilient:
    def test_retries_then_succeeds_on_same_client(self):
        client = MockLLMClient([LLMError("429", retryable=True), MockLLMClient.reply("ok")])
        resilient = ResilientLLMClient([client], max_retries=2, sleeper=lambda _: None)
        assert resilient.chat(request()).content == "ok"

    def test_falls_back_to_next_client(self):
        primary = MockLLMClient([LLMError("500", retryable=True)])
        backup = MockLLMClient([MockLLMClient.reply("from backup")])
        resilient = ResilientLLMClient([primary, backup], max_retries=0, sleeper=lambda _: None)
        assert resilient.chat(request()).content == "from backup"

    def test_non_retryable_error_stops_immediately(self):
        """401 这类错误换端点也没用，不该浪费降级链。"""
        primary = MockLLMClient([LLMError("401", retryable=False)])
        backup = MockLLMClient([MockLLMClient.reply("never reached")])
        resilient = ResilientLLMClient([primary, backup], max_retries=2, sleeper=lambda _: None)
        with pytest.raises(LLMError, match="401"):
            resilient.chat(request())
        assert backup.calls == []

    def test_raises_last_error_when_all_exhausted(self):
        clients = [
            MockLLMClient([LLMError("first down", retryable=True)]),
            MockLLMClient([LLMError("second down", retryable=True)]),
        ]
        resilient = ResilientLLMClient(clients, max_retries=0, sleeper=lambda _: None)
        with pytest.raises(LLMError, match="second down"):
            resilient.chat(request())

    def test_backoff_is_exponential_and_capped(self):
        delays: list[float] = []
        client = MockLLMClient([LLMError("429", retryable=True) for _ in range(5)])
        resilient = ResilientLLMClient([client], max_retries=4, sleeper=delays.append)
        with pytest.raises(LLMError):
            resilient.chat(request())
        assert delays == [0.5, 1.0, 2.0, 4.0]

    def test_requires_at_least_one_client(self):
        with pytest.raises(ValueError, match="至少需要一个"):
            ResilientLLMClient([])


class TestFactory:
    def test_without_api_key_returns_mock(self):
        settings = load_settings(env_file=None)
        assert isinstance(build_llm_client(settings), MockLLMClient)

    def test_injected_mock_is_used(self):
        settings = load_settings(env_file=None)
        mock = MockLLMClient()
        assert build_llm_client(settings, mock=mock) is mock

    def test_with_api_key_builds_resilient_chain(self):
        settings = load_settings(
            env_file=None,
            llm={
                "primary": {"api_key": "sk-1", "model": "m1"},
                "fallbacks": [{"name": "backup", "api_key": "sk-2", "model": "m2"}],
            },
        )
        built: list[str] = []

        def factory(profile):
            built.append(profile.model)
            return MockLLMClient()

        client = build_llm_client(settings, client_factory=factory)
        assert isinstance(client, ResilientLLMClient)
        assert built == ["m1", "m2"]


def test_llm_response_wants_tool_flag():
    assert not LLMResponse().wants_tool
    assert MockLLMClient.tool_reply("search_tool").wants_tool


def test_request_serializes_messages_for_the_wire():
    payload = request(Message.system("s"), Message.user("u")).to_llm_messages()
    assert payload == [
        {"role": MessageRole.SYSTEM.value, "content": "s"},
        {"role": MessageRole.USER.value, "content": "u"},
    ]
