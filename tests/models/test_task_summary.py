"""步骤 6 验收：TaskSummary 序列化字段名与需求文档一致，to_prompt_dict 只投喂点名的字段。"""

import json

import pytest
from pydantic import ValidationError

from agent.common.enums import IntentType, OperationStatus, TaskStatus
from agent.models.task_summary import INTENT_PROMPT_FIELDS, Operation, TaskSummary

# 需求文档字段集，命名按开发计划 1.2 节修正
REQUIREMENT_FIELDS = [
    "task_id",
    "content",
    "intent",
    "related_task_ids",
    "operations",
    "result",
    "status",
    "output",
]


def make_summary(**overrides: object) -> TaskSummary:
    payload: dict[str, object] = {
        "task_id": "task_1",
        "content": "查昨天的错误日志",
    }
    payload.update(overrides)
    return TaskSummary.model_validate(payload)


# ============================================================ 字段集


def test_serialized_field_names_match_the_requirement() -> None:
    """验收核心：对外 JSON 的键必须与需求文档（修正后）逐字一致。"""
    payload = make_summary().to_dict()

    assert list(payload) == REQUIREMENT_FIELDS


def test_legacy_requirement_typos_are_rejected() -> None:
    """`related task` / `summery` 是需求文档笔误，不做兼容。"""
    with pytest.raises(ValidationError):
        TaskSummary.model_validate(
            {"task_id": "task_1", "content": "q", "related task": ["task_0"]}
        )


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_summary(related_task=["task_0"])


# ============================================================ to_prompt_dict


def test_intent_stage_projection_matches_the_requirement() -> None:
    """需求文档第 1 步：意图识别只看 content 与 status。"""
    summary = make_summary(
        intent=IntentType.NEW_QUERY,
        related_task_ids=["task_0"],
        operations=[Operation(index=0, name="查询", tool="kibana_query")],
        result="不该出现在意图识别提示词里",
        status=TaskStatus.INTENDING,
        output="也不该出现",
    )

    payload = summary.to_prompt_dict(INTENT_PROMPT_FIELDS)

    assert payload == {"content": "查昨天的错误日志", "status": "intending"}
    assert set(payload) == {"content", "status"}


def test_prompt_dict_preserves_caller_specified_order() -> None:
    payload = make_summary(status=TaskStatus.PLANNING).to_prompt_dict(("status", "content"))

    assert list(payload) == ["status", "content"]


def test_prompt_dict_serializes_enums_as_strings() -> None:
    payload = make_summary(intent=IntentType.EXPORT).to_prompt_dict(("intent", "status"))

    assert payload["intent"] == "export"
    assert type(payload["intent"]) is str
    assert json.dumps(payload)


def test_empty_field_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="必须指定字段"):
        make_summary().to_prompt_dict(())


def test_unknown_prompt_field_is_rejected_with_legal_names() -> None:
    with pytest.raises(ValueError, match="related_task") as excinfo:
        make_summary().to_prompt_dict(("content", "related_task"))

    assert "related_task_ids" in str(excinfo.value)


def test_intent_prompt_fields_constant_is_exactly_content_and_status() -> None:
    """步骤 21 会直接引用该常量，改动必须是有意识的。"""
    assert INTENT_PROMPT_FIELDS == ("content", "status")


# ============================================================ Operation


def test_operation_defaults_to_pending_with_empty_args() -> None:
    op = Operation(index=0, name="查询", tool="kibana_query")

    assert op.status is OperationStatus.PENDING
    assert op.args == {}
    assert op.result_ref is None
    assert op.error is None


def test_operation_indices_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="唯一"):
        make_summary(
            operations=[
                Operation(index=0, name="a", tool="t"),
                Operation(index=0, name="b", tool="t"),
            ]
        )


def test_operation_index_must_be_non_negative() -> None:
    with pytest.raises(ValidationError, match=">= 0"):
        Operation(index=-1, name="a", tool="t")


def test_operation_lookup_by_index() -> None:
    first = Operation(index=0, name="查询", tool="kibana_query")
    second = Operation(index=2, name="导出", tool="export_tool")
    summary = make_summary(operations=[first, second])

    assert summary.operation(2) is second


def test_missing_operation_raises_key_error() -> None:
    with pytest.raises(KeyError, match="index=3"):
        make_summary().operation(3)


def test_defaults_are_not_shared_between_operations() -> None:
    first = Operation(index=0, name="a", tool="t")
    second = Operation(index=1, name="b", tool="t")

    first.args["k"] = "v"

    assert second.args == {}


# ============================================================ 序列化往返


def test_json_round_trip_is_lossless() -> None:
    original = make_summary(
        intent=IntentType.ANALYSIS,
        related_task_ids=["task_0"],
        operations=[
            Operation(
                index=0,
                name="聚合",
                tool="analysis_tool",
                args={"artifact_id": "art_1"},
                status=OperationStatus.SUCCEEDED,
                result_ref="art_2",
            )
        ],
        result="错误数 12",
        status=TaskStatus.COMPLETED,
        output="昨天共 12 条错误",
    )

    restored = TaskSummary.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original


def test_to_dict_is_directly_json_serializable() -> None:
    payload = make_summary(intent=IntentType.CHAT).to_dict()

    assert payload["intent"] == "chat"
    assert payload["status"] == "created"
    assert json.dumps(payload)


def test_invalid_intent_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_summary(intent="analisis")


def test_invalid_status_string_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_summary(status="intenting")
