"""步骤 3 验收：日志为合法 JSON 且自动携带上下文字段。"""

import io
import json
import logging
import threading
from typing import Any

import pytest

from agent.common.errors import ConfigError, ToolTimeoutError
from agent.common.logging import (
    _force_utf8,
    clear_log_context,
    get_log_context,
    get_logger,
    log_context,
    set_log_context,
    setup_logging,
)


@pytest.fixture
def sink() -> io.StringIO:
    """把日志重定向到内存流，并在用例结束后卸载 handler、清空上下文。"""
    stream = io.StringIO()
    handler = setup_logging("DEBUG", stream=stream)
    try:
        yield stream
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        clear_log_context()


def _records(stream: io.StringIO) -> list[dict[str, Any]]:
    """解析全部输出行；任何一行不是合法 JSON 都会在此暴露。"""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ============================================================ JSON 合法性


def test_each_line_is_valid_json(sink: io.StringIO) -> None:
    logger = get_logger("test.json")
    logger.info("第一条")
    logger.warning("第二条")

    records = _records(sink)
    assert len(records) == 2
    assert [r["message"] for r in records] == ["第一条", "第二条"]


def test_record_carries_the_standard_envelope(sink: io.StringIO) -> None:
    get_logger("test.envelope").info("查询完成")

    record = _records(sink)[0]
    assert record["level"] == "INFO"
    assert record["logger"] == "test.envelope"
    assert record["message"] == "查询完成"
    assert record["timestamp"].endswith("+00:00"), "时间戳须为 UTC，便于跨时区聚合"
    assert "test_logging" in record["source"]


def test_chinese_is_not_escaped(sink: io.StringIO) -> None:
    get_logger("test.cjk").info("中文日志")

    assert "中文日志" in sink.getvalue()
    assert "\\u4e2d" not in sink.getvalue()


def test_percent_style_args_are_interpolated(sink: io.StringIO) -> None:
    get_logger("test.args").info("任务 %s 用时 %d ms", "task_1", 42)

    assert _records(sink)[0]["message"] == "任务 task_1 用时 42 ms"


# ============================================================ 上下文自动注入


def test_context_ids_are_injected_without_being_passed(sink: io.StringIO) -> None:
    """业务代码不需要层层传递 ID，日志里也不会漏掉链路信息。"""
    with log_context(trace_id="trace_1", session_id="sess_1", task_id="task_1"):
        get_logger("test.ctx").info("执行中")

    record = _records(sink)[0]
    assert record["trace_id"] == "trace_1"
    assert record["session_id"] == "sess_1"
    assert record["task_id"] == "task_1"


def test_unset_context_fields_are_omitted(sink: io.StringIO) -> None:
    with log_context(trace_id="trace_only"):
        get_logger("test.partial").info("执行中")

    record = _records(sink)[0]
    assert record["trace_id"] == "trace_only"
    assert "session_id" not in record, "未设置的字段不应输出为 null 噪声"
    assert "task_id" not in record


def test_context_is_restored_on_exit(sink: io.StringIO) -> None:
    logger = get_logger("test.restore")

    with log_context(task_id="task_outer"):
        with log_context(task_id="task_inner"):
            logger.info("内层")
        logger.info("外层")
    logger.info("已退出")

    inner, outer, outside = _records(sink)
    assert inner["task_id"] == "task_inner"
    assert outer["task_id"] == "task_outer", "嵌套退出后应还原为外层值"
    assert "task_id" not in outside


def test_set_log_context_persists_without_a_block(sink: io.StringIO) -> None:
    """请求入口场景：设置一次即对后续全部日志生效。"""
    set_log_context(trace_id="trace_mw")
    get_logger("test.middleware").info("请求处理")

    assert _records(sink)[0]["trace_id"] == "trace_mw"
    assert get_log_context() == {"trace_id": "trace_mw"}


def test_clear_log_context_removes_everything() -> None:
    set_log_context(trace_id="t", session_id="s", task_id="k")
    assert len(get_log_context()) == 3

    clear_log_context()
    assert get_log_context() == {}


def test_unknown_context_field_is_rejected() -> None:
    """拼错字段名若被静默接受，会造成"日志里死活找不到该字段"的排查黑洞。"""
    with pytest.raises(ConfigError) as excinfo:
        set_log_context(tenant_id="acme")

    assert "tenant_id" in str(excinfo.value)

    with pytest.raises(ConfigError), log_context(traceid="typo"):
        pass


def test_context_is_isolated_across_threads(sink: io.StringIO) -> None:
    """contextvars 而非全局变量：并发任务不得串台。"""
    logger = get_logger("test.threads")
    barrier = threading.Barrier(2)

    def work(task_id: str) -> None:
        with log_context(task_id=task_id):
            barrier.wait()  # 确保两个线程的上下文同时处于活跃状态
            logger.info("并发写入")

    threads = [threading.Thread(target=work, args=(f"task_{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    task_ids = {record["task_id"] for record in _records(sink)}
    assert task_ids == {"task_0", "task_1"}


# ============================================================ extra 字段


def test_extra_fields_are_nested_under_extra(sink: io.StringIO) -> None:
    get_logger("test.extra").info("工具调用", extra={"tool": "search_tool", "elapsed_ms": 12})

    record = _records(sink)[0]
    assert record["extra"] == {"tool": "search_tool", "elapsed_ms": 12}


def test_extra_key_absent_when_nothing_passed(sink: io.StringIO) -> None:
    get_logger("test.noextra").info("普通日志")

    assert "extra" not in _records(sink)[0]


def test_explicit_extra_overrides_context(sink: io.StringIO) -> None:
    with log_context(task_id="task_ctx"):
        get_logger("test.override").info("显式指定", extra={"task_id": "task_explicit"})

    assert _records(sink)[0]["task_id"] == "task_explicit"


def test_unserializable_extra_does_not_break_logging(sink: io.StringIO) -> None:
    """一条日志写不出去不应中断业务，故不可序列化的值回退为 str()。"""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    get_logger("test.opaque").info("含不可序列化字段", extra={"payload": Opaque()})

    assert _records(sink)[0]["extra"]["payload"] == "<opaque>"


# ============================================================ 异常记录


def test_agent_error_is_recorded_with_decision_fields(sink: io.StringIO) -> None:
    """日志里能直接看出 retryable，无需回查代码即可判断任务为何走了 RETRYING。"""
    try:
        raise ToolTimeoutError("search_tool 执行超时", detail="timeout=30s")
    except ToolTimeoutError:
        get_logger("test.err").exception("工具失败")

    record = _records(sink)[0]
    assert record["error"] == {
        "type": "ToolTimeoutError",
        "code": "tool_timeout",
        "message": "search_tool 执行超时",
        "retryable": True,
        "detail": "timeout=30s",
    }
    assert "ToolTimeoutError" in record["traceback"]


def test_plain_exception_is_recorded_too(sink: io.StringIO) -> None:
    try:
        raise ValueError("普通错误")
    except ValueError:
        get_logger("test.err2").exception("非 AgentError")

    record = _records(sink)[0]
    assert record["error"] == {"type": "ValueError", "message": "普通错误"}


# ============================================================ setup_logging


def test_repeated_setup_does_not_duplicate_output() -> None:
    """重复初始化若叠加 handler，日志会成倍输出并干扰排查。"""
    stream = io.StringIO()
    first = setup_logging("INFO", stream=stream)
    second = setup_logging("INFO", stream=stream)
    try:
        get_logger("test.dup").info("只应出现一次")
        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        assert len(lines) == 1
        assert first not in logging.getLogger().handlers
    finally:
        logging.getLogger().removeHandler(second)
        second.close()


def test_level_filters_lower_severity() -> None:
    stream = io.StringIO()
    handler = setup_logging("WARNING", stream=stream)
    try:
        logger = get_logger("test.level")
        logger.info("不应出现")
        logger.warning("应出现")

        records = _records(stream)
        assert [r["message"] for r in records] == ["应出现"]
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_level_accepts_int_and_lowercase_name() -> None:
    stream = io.StringIO()
    handler = setup_logging("debug", stream=stream)
    try:
        assert logging.getLogger().level == logging.DEBUG
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    handler = setup_logging(logging.ERROR, stream=stream)
    try:
        assert logging.getLogger().level == logging.ERROR
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        setup_logging("VERBOSE", stream=io.StringIO())

    assert "VERBOSE" in str(excinfo.value)


def test_default_stream_is_stderr_and_forced_to_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """未显式指定流时应写 stderr，并把它切到 UTF-8。

    Windows 控制台默认非 UTF-8，中文日志会因编码失败而丢失。
    """
    calls: list[dict[str, str]] = []

    class FakeStderr(io.StringIO):
        def reconfigure(self, **kwargs: str) -> None:  # type: ignore[override]
            calls.append(kwargs)

    fake = FakeStderr()
    monkeypatch.setattr("sys.stderr", fake)

    handler = setup_logging("INFO")
    try:
        get_logger("test.stderr").info("默认流")
        assert calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]
        assert "默认流" in fake.getvalue()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_stream_without_reconfigure_is_left_alone() -> None:
    """老式流对象没有 reconfigure，此时应静默跳过而非报错。"""

    class LegacyStream:
        def __init__(self) -> None:
            self.written: list[str] = []

        def write(self, text: str) -> int:
            self.written.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    legacy = LegacyStream()
    _force_utf8(legacy)  # type: ignore[arg-type]

    assert legacy.written == []


def test_reconfigure_failure_is_suppressed() -> None:
    """重配编码失败也不能让日志初始化崩掉。"""

    class HostileStream(io.StringIO):
        def reconfigure(self, **_: str) -> None:  # type: ignore[override]
            raise OSError("不支持重配")

    _force_utf8(HostileStream())


def test_plain_format_is_available_for_local_debugging() -> None:
    stream = io.StringIO()
    handler = setup_logging("INFO", stream=stream, json_format=False)
    try:
        get_logger("test.plain").info("人类可读")
        output = stream.getvalue()
        assert "人类可读" in output
        with pytest.raises(json.JSONDecodeError):
            json.loads(output.splitlines()[0])
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
