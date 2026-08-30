"""步骤 4 验收：严格解析（只认规范拼写）、序列化为小写字符串并可往返。"""

import json

import pytest

from agent.common.enums import (
    ACTIVE_TASK_STATUSES,
    FINISHED_OPERATION_STATUSES,
    TERMINAL_TASK_STATUSES,
    AgentEnum,
    ArtifactType,
    ContextScope,
    IntentType,
    MessageRole,
    OperationStatus,
    PromptStage,
    TaskStatus,
)

ALL_ENUMS: list[type[AgentEnum]] = [
    TaskStatus,
    IntentType,
    MessageRole,
    ArtifactType,
    OperationStatus,
    ContextScope,
    PromptStage,
]

# 开发计划规定的取值域，逐一固定以防误增误删
EXPECTED_MEMBERS: dict[type[AgentEnum], set[str]] = {
    TaskStatus: {
        "CREATED",
        "INTENDING",
        "PLANNING",
        "EXECUTING",
        "VALIDATING",
        "COMPLETED",
        "FAILED",
        "CANCELED",
        "WAITING_USER",
        "RETRYING",
    },
    IntentType: {"NEW_QUERY", "ANALYSIS", "EXPORT", "CHAT", "UNKNOWN"},
    MessageRole: {"SYSTEM", "USER", "ASSISTANT", "TOOL"},
    ArtifactType: {"TABLE", "SCALAR", "CHART", "FILE", "TEXT"},
    OperationStatus: {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED"},
    ContextScope: {"CURRENT", "RELATED", "HISTORY"},
    PromptStage: {"INTENT_RECOGNITION", "PLAN", "EXECUTE", "VALIDATE"},
}


# ============================================================ 取值域


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_member_set_matches_the_plan(enum_cls: type[AgentEnum]) -> None:
    assert {member.name for member in enum_cls} == EXPECTED_MEMBERS[enum_cls]


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_values_are_lowercase_snake_case_of_the_name(enum_cls: type[AgentEnum]) -> None:
    for member in enum_cls:
        assert member.value == member.name.lower()


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_values_helper_lists_canonical_values(enum_cls: type[AgentEnum]) -> None:
    assert enum_cls.values() == [member.value for member in enum_cls]


def test_message_role_matches_openai_protocol() -> None:
    """角色取值需可直接投喂模型，不能有自造名称。"""
    assert set(MessageRole.values()) == {"system", "user", "assistant", "tool"}


# ============================================================ 序列化往返


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_json_round_trip(enum_cls: type[AgentEnum]) -> None:
    for member in enum_cls:
        encoded = json.dumps({"v": member})
        assert encoded == f'{{"v": "{member.value}"}}'
        assert enum_cls(json.loads(encoded)["v"]) is member


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_member_equals_its_string_value(enum_cls: type[AgentEnum]) -> None:
    """StrEnum 语义：无需额外转换层即可与字符串比较。"""
    for member in enum_cls:
        assert member == member.value
        assert f"{member}" == member.value


# ============================================================ 严格解析


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_canonical_value_parses(enum_cls: type[AgentEnum]) -> None:
    for member in enum_cls:
        assert enum_cls.from_str(member.value) is member


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_non_canonical_spellings_are_rejected(enum_cls: type[AgentEnum]) -> None:
    """只认规范拼写：大小写变体、成员名、留空、换分隔符一律非法。

    这些写法曾被归一化接受，现已收紧——错误拼写猜成正确成员会掩盖真正的缺陷。
    """
    for member in enum_cls:
        variants = [
            member.value.upper(),
            member.name,
            member.value.capitalize(),
            f"  {member.value}  ",
            member.value.replace("_", "-"),
            member.value.replace("_", " "),
            member.value.replace("_", ""),
        ]
        for variant in variants:
            if variant == member.value:
                continue  # 单词成员的部分变体与规范值相同，不构成反例
            with pytest.raises(ValueError):
                enum_cls.from_str(variant)


@pytest.mark.parametrize(
    "raw",
    [
        "newQuery",
        "NewQuery",
        "newquery",
        "NEWQUERY",  # v0 笔误
        "analisis",  # v0 笔误
        "analyze",  # 同义词
        "query",
        "other",
    ],
)
def test_typos_and_synonyms_are_not_absorbed(raw: str) -> None:
    with pytest.raises(ValueError):
        IntentType.from_str(raw)


@pytest.mark.parametrize("raw", ["cancelled", "intenting", "waitingUser", "done"])
def test_task_status_typos_are_not_absorbed(raw: str) -> None:
    with pytest.raises(ValueError):
        TaskStatus.from_str(raw)


@pytest.mark.parametrize("raw", ["assist", "function", "Human"])
def test_message_role_typos_are_not_absorbed(raw: str) -> None:
    with pytest.raises(ValueError):
        MessageRole.from_str(raw)


def test_direct_construction_is_equally_strict() -> None:
    """`Cls(raw)` 与 `from_str` 必须一致严格，否则两条入口的行为会分叉。"""
    with pytest.raises(ValueError):
        IntentType("NEW-QUERY")
    with pytest.raises(ValueError):
        OperationStatus("OK")


# ============================================================ 解析失败


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_unknown_value_raises_with_actionable_message(enum_cls: type[AgentEnum]) -> None:
    with pytest.raises(ValueError) as excinfo:
        enum_cls.from_str("definitely-not-a-member")

    message = str(excinfo.value)
    assert enum_cls.__name__ in message
    assert enum_cls.values()[0] in message, "错误信息应列出合法取值"


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_default_is_returned_instead_of_raising(enum_cls: type[AgentEnum]) -> None:
    """`default` 是调用方显式选择的兜底，不是对错误拼写的猜测。"""
    fallback = list(enum_cls)[-1]

    assert enum_cls.from_str("garbage", default=fallback) is fallback


@pytest.mark.parametrize("bad", [None, 42, 3.5, [], {}, True])
def test_non_string_input_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        IntentType.from_str(bad)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_input_is_not_silently_accepted(blank: str) -> None:
    with pytest.raises(ValueError):
        IntentType.from_str(blank)

    assert IntentType.from_str(blank, default=IntentType.UNKNOWN) is IntentType.UNKNOWN


def test_member_passes_through_unchanged() -> None:
    assert IntentType.from_str(IntentType.CHAT) is IntentType.CHAT


def test_direct_construction_from_non_string_raises() -> None:
    with pytest.raises(ValueError):
        IntentType(42)


def test_new_enum_needs_no_boilerplate() -> None:
    """继承基类即可获得严格解析与 values()，后续新增枚举无须写样板代码。"""

    class Flavour(AgentEnum):
        SWEET = "sweet"
        SALTY = "salty"

    assert Flavour.from_str("sweet") is Flavour.SWEET
    assert Flavour.values() == ["sweet", "salty"]
    assert Flavour.from_str("nope", default=Flavour.SALTY) is Flavour.SALTY
    with pytest.raises(ValueError):
        Flavour.from_str("SWEET")


def test_no_tolerant_parsing_machinery_remains() -> None:
    """别名表/归一化一旦被重新引入，取值域就不再是单一权威拼写。"""
    own_attributes = set(vars(AgentEnum))

    assert "_aliases" not in own_attributes
    assert "normalize" not in own_attributes
    assert "_missing_" not in own_attributes, "覆盖 _missing_ 会让 Cls(raw) 重新变得容错"


# ============================================================ 状态分区常量


def test_terminal_and_active_partition_all_statuses() -> None:
    assert set(TaskStatus) == TERMINAL_TASK_STATUSES | ACTIVE_TASK_STATUSES
    assert not TERMINAL_TASK_STATUSES & ACTIVE_TASK_STATUSES


def test_terminal_statuses_match_the_plan() -> None:
    assert sorted(TERMINAL_TASK_STATUSES) == sorted(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED}
    )


@pytest.mark.parametrize("status", list(TaskStatus))
def test_is_terminal_agrees_with_the_constant(status: TaskStatus) -> None:
    assert status.is_terminal is (status in TERMINAL_TASK_STATUSES)


@pytest.mark.parametrize(
    ("status", "finished"),
    [
        (OperationStatus.PENDING, False),
        (OperationStatus.RUNNING, False),
        (OperationStatus.SUCCEEDED, True),
        (OperationStatus.FAILED, True),
        (OperationStatus.SKIPPED, True),
    ],
)
def test_operation_is_finished(status: OperationStatus, finished: bool) -> None:
    assert status.is_finished is finished


def test_finished_operation_statuses_partition_the_enum() -> None:
    unfinished = frozenset(OperationStatus) - FINISHED_OPERATION_STATUSES

    assert unfinished == {OperationStatus.PENDING, OperationStatus.RUNNING}


# ============================================================ 兼容重导出


def test_legacy_module_re_exports_the_same_objects() -> None:
    """旧引用路径必须指向同一对象，否则 `is` 比较与 isinstance 判断会失效。"""
    from agent.task_manage import type as legacy

    assert legacy.TaskStatus is TaskStatus
    assert legacy.IntentType is IntentType
    assert legacy.TaskType is IntentType


def test_v0_typo_spellings_are_gone_entirely() -> None:
    """v0 笔误既不作为成员属性存在，也不作为可解析的字符串存在。"""
    from agent.task_manage.type import TaskType

    for typo in ("NEWQUERY", "ANALISIS"):
        assert not hasattr(TaskType, typo)
        with pytest.raises(ValueError):
            TaskType.from_str(typo)
