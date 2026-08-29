"""步骤 4 验收：容错解析（大小写、别名）、序列化为小写字符串并可往返。"""

import json

import pytest

from agent.common.enums import (
    ACTIVE_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    AgentEnum,
    ArtifactType,
    ContextScope,
    IntentType,
    MessageRole,
    OperationStatus,
    TaskStatus,
)

ALL_ENUMS: list[type[AgentEnum]] = [
    TaskStatus,
    IntentType,
    MessageRole,
    ArtifactType,
    OperationStatus,
    ContextScope,
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


# ============================================================ 容错解析


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_canonical_value_parses(enum_cls: type[AgentEnum]) -> None:
    for member in enum_cls:
        assert enum_cls.from_str(member.value) is member


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_case_and_separator_variants_parse(enum_cls: type[AgentEnum]) -> None:
    for member in enum_cls:
        variants = [
            member.value.upper(),
            member.name,
            f"  {member.value}  ",
            member.value.replace("_", "-"),
            member.value.replace("_", " "),
            member.value.replace("_", "."),
        ]
        for variant in variants:
            assert enum_cls.from_str(variant) is member, f"{variant!r} 未能解析"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("newQuery", IntentType.NEW_QUERY),
        ("NewQuery", IntentType.NEW_QUERY),
        ("newquery", IntentType.NEW_QUERY),
        ("NEWQUERY", IntentType.NEW_QUERY),
        ("analisis", IntentType.ANALYSIS),
        ("analyze", IntentType.ANALYSIS),
        ("query", IntentType.NEW_QUERY),
        ("other", IntentType.UNKNOWN),
    ],
)
def test_intent_aliases_absorb_v0_typos_and_model_variance(raw: str, expected: IntentType) -> None:
    assert IntentType.from_str(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cancelled", TaskStatus.CANCELED),
        ("intenting", TaskStatus.INTENDING),
        ("waitingUser", TaskStatus.WAITING_USER),
        ("waiting_for_user", TaskStatus.WAITING_USER),
        ("done", TaskStatus.COMPLETED),
    ],
)
def test_task_status_aliases(raw: str, expected: TaskStatus) -> None:
    assert TaskStatus.from_str(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("assist", MessageRole.ASSISTANT),
        ("function", MessageRole.TOOL),
        ("Human", MessageRole.USER),
    ],
)
def test_message_role_aliases(raw: str, expected: MessageRole) -> None:
    assert MessageRole.from_str(raw) is expected


def test_direct_construction_is_also_tolerant() -> None:
    """`_missing_` 钩子使 `Cls(raw)` 自身具备容错，无需调用方记得改用 from_str。"""
    assert IntentType("NEW-QUERY") is IntentType.NEW_QUERY
    assert OperationStatus("OK") is OperationStatus.SUCCEEDED


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
    """`_missing_` 对非字符串必须放弃解析，否则会掩盖类型错误。"""
    with pytest.raises(ValueError):
        IntentType(42)


def test_enum_without_aliases_still_parses() -> None:
    """基类的空别名表要能正常工作，后续新增枚举无须为此写样板代码。"""

    class Flavour(AgentEnum):
        SWEET = "sweet"
        SALTY = "salty"

    assert Flavour._aliases() == {}
    assert Flavour.from_str("SWEET") is Flavour.SWEET
    assert Flavour.from_str("nope", default=Flavour.SALTY) is Flavour.SALTY


# ============================================================ 别名表自身的正确性


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_every_alias_points_at_a_real_member(enum_cls: type[AgentEnum]) -> None:
    """别名指向不存在的取值时不会报错，只会永远匹配不上——必须静态拦住。"""
    for alias, target in enum_cls._aliases().items():
        assert target in enum_cls.values(), (
            f"{enum_cls.__name__} 的别名 {alias!r} 指向未知取值 {target!r}"
        )


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_alias_keys_are_already_normalized(enum_cls: type[AgentEnum]) -> None:
    """别名查表发生在归一化之后，未归一的键（如 "NewQuery"）永远命中不了。"""
    for alias in enum_cls._aliases():
        assert alias == enum_cls.normalize(alias), (
            f"{enum_cls.__name__} 的别名键 {alias!r} 未归一化"
        )


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_aliases_do_not_shadow_canonical_values(enum_cls: type[AgentEnum]) -> None:
    """与规范取值重名的别名是死代码，且暗示作者对取值域有误解。"""
    overlap = set(enum_cls._aliases()) & set(enum_cls.values())

    assert not overlap, f"{enum_cls.__name__} 的别名与规范取值重名：{sorted(overlap)}"


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


# ============================================================ 兼容重导出


def test_legacy_module_re_exports_the_same_objects() -> None:
    """旧引用路径必须指向同一对象，否则 `is` 比较与 isinstance 判断会失效。"""
    from agent.task_manage import type as legacy

    assert legacy.TaskStatus is TaskStatus
    assert legacy.IntentType is IntentType
    assert legacy.TaskType is IntentType


def test_legacy_task_type_spellings_still_parse() -> None:
    """v0 的成员名不再作为属性提供，但其字符串写法仍可解析。"""
    from agent.task_manage.type import TaskType

    assert TaskType.from_str("NEWQUERY") is IntentType.NEW_QUERY
    assert TaskType.from_str("ANALISIS") is IntentType.ANALYSIS
    assert not hasattr(TaskType, "NEWQUERY")
