"""步骤 3 验收：异常 retryable 分类正确。

`retryable` 是状态机在 RETRYING 与 FAILED 之间分流的唯一依据，
分类错一个就会造成"该重试的直接失败"或"不可能成功的动作被反复重试"，
故此处对每个异常类逐一固定其取值。
"""

import builtins

import pytest

from agent.common import errors
from agent.common.errors import (
    AgentError,
    AgentMemoryError,
    ConfigError,
    ContextError,
    LLMError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTimeoutError,
    TaskCanceledError,
    TaskStateError,
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
    ToolTimeoutError,
)

RETRYABLE_CLASSES = [
    LLMTimeoutError,
    LLMRateLimitError,
    ToolInvocationError,
    ToolTimeoutError,
]

NON_RETRYABLE_CLASSES = [
    AgentError,
    ConfigError,
    LLMError,
    LLMResponseFormatError,
    ToolError,
    ToolNotFoundError,
    ContextError,
    AgentMemoryError,
    TaskStateError,
    TaskCanceledError,
]

ALL_CLASSES = RETRYABLE_CLASSES + NON_RETRYABLE_CLASSES


# ============================================================ retryable 分类


@pytest.mark.parametrize("error_cls", RETRYABLE_CLASSES)
def test_transient_failures_are_retryable(error_cls: type[AgentError]) -> None:
    assert error_cls("boom").retryable is True


@pytest.mark.parametrize("error_cls", NON_RETRYABLE_CLASSES)
def test_deterministic_failures_are_not_retryable(error_cls: type[AgentError]) -> None:
    assert error_cls("boom").retryable is False


def test_retryable_can_be_overridden_per_instance() -> None:
    """同一异常类在不同成因下可重试性不同，无需为每种成因新增一个类。"""
    transient = LLMError("连接被重置", retryable=True)
    permanent = LLMError("请求被拒绝")

    assert transient.retryable is True
    assert permanent.retryable is False
    # 实例覆盖不得污染类级默认值
    assert LLMError.retryable is False


# ============================================================ 类层次


@pytest.mark.parametrize("error_cls", ALL_CLASSES)
def test_every_error_derives_from_agent_error(error_cls: type[AgentError]) -> None:
    assert issubclass(error_cls, AgentError)
    assert issubclass(error_cls, Exception)


@pytest.mark.parametrize(
    ("error_cls", "base"),
    [
        (LLMTimeoutError, LLMError),
        (LLMRateLimitError, LLMError),
        (LLMResponseFormatError, LLMError),
        (ToolNotFoundError, ToolError),
        (ToolInvocationError, ToolError),
        (ToolTimeoutError, ToolError),
    ],
)
def test_subclasses_group_under_their_family(
    error_cls: type[AgentError], base: type[AgentError]
) -> None:
    """按族捕获（except LLMError）必须能兜住该族全部子类。"""
    assert issubclass(error_cls, base)


@pytest.mark.parametrize("error_cls", ALL_CLASSES)
def test_codes_are_unique_and_stable(error_cls: type[AgentError]) -> None:
    assert error_cls.code
    assert error_cls.code == error_cls.code.lower()


def test_all_codes_are_distinct() -> None:
    codes = [cls.code for cls in ALL_CLASSES]
    assert len(codes) == len(set(codes)), f"存在重复的 code：{codes}"


# ============================================================ 字段与序列化


def test_message_and_detail_are_both_accessible() -> None:
    error = ConfigError("配置加载失败", detail="  - APP_LOG_LEVEL: 非法取值")

    assert error.message == "配置加载失败"
    assert error.detail == "  - APP_LOG_LEVEL: 非法取值"
    # str() 展开 detail，便于日志与 CLI 直接打印
    assert "配置加载失败" in str(error)
    assert "APP_LOG_LEVEL" in str(error)


def test_str_without_detail_is_just_the_message() -> None:
    assert str(ContextError("字段不在白名单内")) == "字段不在白名单内"


def test_to_dict_exposes_the_four_decision_fields() -> None:
    error = ToolTimeoutError("search_tool 执行超时", detail="timeout=30s")

    assert error.to_dict() == {
        "code": "tool_timeout",
        "message": "search_tool 执行超时",
        "retryable": True,
        "detail": "timeout=30s",
    }


def test_code_can_be_overridden_for_finer_grained_metrics() -> None:
    error = LLMError("连接失败", code="llm_connection", retryable=True)

    assert error.to_dict()["code"] == "llm_connection"
    assert LLMError.code == "llm_error"


def test_repr_is_diagnostic() -> None:
    text = repr(ToolNotFoundError("工具不存在：unknown_tool"))

    assert "ToolNotFoundError" in text
    assert "tool_not_found" in text
    assert "unknown_tool" in text


# ============================================================ 命名安全


def test_no_error_name_shadows_a_builtin() -> None:
    """遮蔽内置异常名会让 `except X` 的含义随导入顺序漂移。

    开发计划中的 `MemoryError` 即属此类，故改名为 `AgentMemoryError`。
    """
    exported = set(errors.__all__)
    shadowed = {name for name in exported if hasattr(builtins, name)}

    assert not shadowed, f"以下异常名遮蔽了内置名：{sorted(shadowed)}"
    assert "MemoryError" not in exported


def test_all_exports_exist() -> None:
    for name in errors.__all__:
        assert hasattr(errors, name), f"__all__ 中的 {name} 并不存在"


def test_agent_memory_error_is_not_the_builtin() -> None:
    assert AgentMemoryError is not builtins.MemoryError
    assert not issubclass(AgentMemoryError, builtins.MemoryError)
