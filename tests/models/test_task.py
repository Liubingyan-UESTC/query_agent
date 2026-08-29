"""步骤 6 验收：Task 只做数据访问，不裁定状态转移；序列化往返一致。"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent.common.enums import TaskStatus
from agent.common.errors import AgentError
from agent.models.task import StatusRecord, Task
from agent.models.task_summary import TaskSummary


def test_create_aligns_ids_and_seeds_history() -> None:
    task = Task.create("查昨天的错误日志", session_id="sess_1")

    assert task.task_id == task.summary.task_id
    assert task.task_id.startswith("task_")
    assert task.session_id == "sess_1"
    assert task.trace_id.startswith("trace_")
    assert task.status is TaskStatus.CREATED
    assert task.summary.status is TaskStatus.CREATED
    assert task.summary.content == "查昨天的错误日志"
    assert [record.status for record in task.status_history] == [TaskStatus.CREATED]


def test_create_generates_a_session_id_when_omitted() -> None:
    task = Task.create("q")

    assert task.session_id.startswith("sess_")


def test_mismatched_task_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="不一致"):
        Task(
            task_id="task_a",
            session_id="sess_1",
            summary=TaskSummary(task_id="task_b", content="q"),
        )


# ============================================================ record_status / touch


def test_record_status_keeps_the_three_copies_in_sync() -> None:
    """实体状态、摘要状态、历史记录必须同步，否则归档会出现撕裂。"""
    task = Task.create("q", session_id="sess_1")

    task.record_status(TaskStatus.INTENDING, note="开始意图识别")

    assert task.status is TaskStatus.INTENDING
    assert task.summary.status is TaskStatus.INTENDING
    assert [record.status for record in task.status_history] == [
        TaskStatus.CREATED,
        TaskStatus.INTENDING,
    ]
    assert task.status_history[-1].note == "开始意图识别"


def test_record_status_does_not_validate_transitions() -> None:
    """转移合法性属于步骤 19。本层若拦截，状态机与模型会各持一份规则。"""
    task = Task.create("q", session_id="sess_1")

    task.record_status(TaskStatus.COMPLETED)

    assert task.status is TaskStatus.COMPLETED
    assert task.summary.status is TaskStatus.COMPLETED


def test_record_status_refreshes_updated_at() -> None:
    task = Task.create("q", session_id="sess_1")
    before = task.updated_at

    task.record_status(TaskStatus.PLANNING)

    assert task.updated_at >= before


def test_touch_only_refreshes_updated_at() -> None:
    task = Task.create("q", session_id="sess_1")
    original_status = task.status
    original_history_len = len(task.status_history)
    before = task.updated_at

    task.touch()

    assert task.updated_at >= before
    assert task.status is original_status
    assert len(task.status_history) == original_history_len


def test_touch_is_observable_across_a_clock_tick() -> None:
    task = Task.create("q", session_id="sess_1")
    task.updated_at = datetime.now(UTC) - timedelta(seconds=2)

    task.touch()

    assert task.updated_at > datetime.now(UTC) - timedelta(seconds=1)


# ============================================================ 序列化往返


def test_json_round_trip_is_lossless() -> None:
    task = Task.create("查昨天的错误日志", session_id="sess_1")
    task.record_status(TaskStatus.FAILED, note="工具超时")
    task.error = AgentError("kibana 超时", code="tool_timeout", retryable=True).to_dict()
    task.retry_count = 1

    restored = Task.from_dict(json.loads(json.dumps(task.to_dict())))

    assert restored == task


def test_to_dict_is_directly_json_serializable() -> None:
    payload = Task.create("q", session_id="sess_1").to_dict()

    assert isinstance(payload["created_at"], str)
    assert isinstance(payload["status"], str)
    assert payload["summary"]["task_id"] == payload["task_id"]
    assert json.dumps(payload)


def test_error_stores_the_structured_agent_error() -> None:
    """retryable 必须随任务一起落盘，否则步骤 19 无法决定 RETRYING 还是 FAILED。"""
    task = Task.create("q", session_id="sess_1")
    task.error = AgentError("限流", code="llm_rate_limit", retryable=True).to_dict()

    assert task.error["retryable"] is True
    assert task.to_dict()["error"]["code"] == "llm_rate_limit"


def test_negative_retry_count_is_rejected() -> None:
    task = Task.create("q", session_id="sess_1")

    with pytest.raises(ValidationError, match="retry_count"):
        task.retry_count = -1


def test_assignment_is_validated() -> None:
    task = Task.create("q", session_id="sess_1")

    with pytest.raises(ValidationError):
        task.status = "intenting"


def test_status_record_defaults_to_timezone_aware_utc() -> None:
    record = StatusRecord(status=TaskStatus.CREATED)

    assert record.at.tzinfo is not None


# ============================================================ 兼容重导出


def test_legacy_module_re_exports_the_same_objects() -> None:
    """旧引用路径必须指向同一对象，否则 isinstance 判断会失效。"""
    from agent.task_manage import task as legacy

    assert legacy.Task is Task
    assert legacy.StatusRecord is StatusRecord
