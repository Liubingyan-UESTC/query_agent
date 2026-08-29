"""步骤 7 验收：append / 查询 / 按 scope 拆分 / 序列化往返；注入片段不进 CURRENT。"""

import json

import pytest
from pydantic import ValidationError

from agent.common.enums import ArtifactType, ContextScope, MessageRole
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task import Task
from agent.models.task_summary import TaskSummary


def make_window() -> ContextWindow:
    task = Task.create("查昨天的错误日志", session_id="sess_1")
    return ContextWindow.from_task(task)


def make_artifact(
    task_id: str,
    *,
    title: str = "结果",
    scope: ContextScope = ContextScope.CURRENT,
    source_task_id: str | None = None,
) -> Artifact:
    return Artifact(
        task_id=task_id,
        producer="kibana_query",
        artifact_type=ArtifactType.TABLE,
        title=title,
        data=[{"a": 1}],
        scope=scope,
        source_task_id=source_task_id,
    )


# ============================================================ 创建与一致性


def test_from_task_shares_the_summary_object() -> None:
    task = Task.create("查昨天的错误日志", session_id="sess_1")
    window = ContextWindow.from_task(task)

    assert window.task_id == task.task_id
    assert window.session_id == task.session_id
    assert window.summary is task.summary
    assert window.content == []
    assert window.artifacts == {}


def test_mismatched_summary_task_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="不一致"):
        ContextWindow(
            task_id="task_a",
            session_id="sess_1",
            summary=TaskSummary(task_id="task_b", content="q"),
        )


# ============================================================ append / get messages


def test_append_preserves_order() -> None:
    window = make_window()
    first = Message.system("你是数据查询助手")
    second = Message.user("查昨天的错误日志")

    window.append_message(first)
    window.append_message(second)

    assert window.get_messages() == [first, second]


def test_duplicate_message_id_is_rejected() -> None:
    window = make_window()
    message = Message.user("hi")
    window.append_message(message)

    with pytest.raises(ValueError, match="message_id"):
        window.append_message(message)


def test_get_messages_filters_by_scope() -> None:
    window = make_window()
    current = Message.user("当前问题")
    related = Message.assistant(
        "关联任务的结论", scope=ContextScope.RELATED, source_task_id="task_0"
    )
    history = Message.user("更早的一问", scope=ContextScope.HISTORY, source_task_id="task_old")

    window.append_message(current)
    window.append_message(related)
    window.append_message(history)

    assert window.get_messages(scope=ContextScope.CURRENT) == [current]
    assert related not in window.get_messages(scope=ContextScope.CURRENT)
    assert history not in window.get_messages(scope=ContextScope.CURRENT)


def test_get_messages_filters_by_roles() -> None:
    window = make_window()
    system = Message.system("sys")
    user = Message.user("q")
    window.append_message(system)
    window.append_message(user)

    assert window.get_messages(roles={MessageRole.USER}) == [user]


def test_get_messages_combines_scope_and_roles() -> None:
    window = make_window()
    window.append_message(Message.user("当前", scope=ContextScope.CURRENT))
    window.append_message(Message.user("注入", scope=ContextScope.RELATED))
    window.append_message(Message.assistant("当前答", scope=ContextScope.CURRENT))

    selected = window.get_messages(scope=ContextScope.CURRENT, roles={MessageRole.USER})

    assert [item.content for item in selected] == ["当前"]


def test_empty_roles_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="roles"):
        make_window().get_messages(roles=())


# ============================================================ artifacts


def test_put_and_get_artifact() -> None:
    window = make_window()
    artifact = make_artifact(window.task_id, title="错误日志")

    window.put_artifact(artifact)

    assert window.get_artifact(artifact.artifact_id) is artifact


def test_put_artifact_overwrites_same_id() -> None:
    window = make_window()
    first = make_artifact(window.task_id, title="旧")
    replacement = first.model_copy(update={"title": "新"})

    window.put_artifact(first)
    window.put_artifact(replacement)

    assert window.get_artifact(first.artifact_id).title == "新"
    assert len(window.artifacts) == 1


def test_missing_artifact_raises_key_error() -> None:
    window = make_window()

    with pytest.raises(KeyError, match="没有 artifact_id"):
        window.get_artifact("art_missing")


def test_artifact_index_omits_payload_data() -> None:
    window = make_window()
    artifact = make_artifact(window.task_id, title="错误日志")
    window.put_artifact(artifact)

    index = window.list_artifact_index()

    assert len(index) == 1
    assert index[0].artifact_id == artifact.artifact_id
    assert index[0].title == "错误日志"
    assert index[0].row_count == 1
    assert "data" not in index[0].to_dict()


def test_artifact_index_preserves_insertion_order() -> None:
    window = make_window()
    first = make_artifact(window.task_id, title="a")
    second = make_artifact(window.task_id, title="b")
    window.put_artifact(first)
    window.put_artifact(second)

    assert [item.title for item in window.list_artifact_index()] == ["a", "b"]


# ============================================================ split_by_scope


def test_split_by_scope_keeps_injected_fragments_out_of_current() -> None:
    """验收核心：注入片段不得被 CURRENT 命中，否则会重复归档。"""
    window = make_window()
    current_msg = Message.user("当前问题")
    related_msg = Message.assistant("关联结论", scope=ContextScope.RELATED, source_task_id="task_0")
    history_msg = Message.user("旧问", scope=ContextScope.HISTORY, source_task_id="task_old")
    current_art = make_artifact(window.task_id, title="本任务结果")
    related_art = make_artifact(
        "task_0",
        title="关联表",
        scope=ContextScope.RELATED,
        source_task_id="task_0",
    )

    window.append_message(current_msg)
    window.append_message(related_msg)
    window.append_message(history_msg)
    window.put_artifact(current_art)
    window.put_artifact(related_art)

    slices = window.split_by_scope()

    assert slices[ContextScope.CURRENT].messages == [current_msg]
    assert list(slices[ContextScope.CURRENT].artifacts) == [current_art.artifact_id]
    assert related_msg not in slices[ContextScope.CURRENT].messages
    assert history_msg not in slices[ContextScope.CURRENT].messages
    assert related_art.artifact_id not in slices[ContextScope.CURRENT].artifacts

    assert slices[ContextScope.RELATED].messages == [related_msg]
    assert list(slices[ContextScope.RELATED].artifacts) == [related_art.artifact_id]
    assert slices[ContextScope.HISTORY].messages == [history_msg]
    assert slices[ContextScope.HISTORY].artifacts == {}


def test_split_by_scope_always_includes_every_scope() -> None:
    slices = make_window().split_by_scope()

    assert set(slices) == set(ContextScope)
    assert all(slice_.messages == [] and slice_.artifacts == {} for slice_ in slices.values())


def test_mutating_a_slice_does_not_append_into_the_window() -> None:
    """切片持有新列表；往切片里追加不应改变窗口。"""
    window = make_window()
    window.append_message(Message.user("q"))

    slices = window.split_by_scope()
    slices[ContextScope.CURRENT].messages.append(Message.assistant("不该写回"))

    assert len(window.get_messages()) == 1


# ============================================================ 序列化往返


def test_json_round_trip_is_lossless() -> None:
    window = make_window()
    window.append_message(Message.system("你是数据查询助手"))
    window.append_message(Message.user("查昨天的错误日志"))
    window.append_message(
        Message.assistant("关联结论", scope=ContextScope.RELATED, source_task_id="task_0")
    )
    window.put_artifact(make_artifact(window.task_id, title="本任务结果"))
    window.put_artifact(
        make_artifact(
            "task_0",
            title="关联表",
            scope=ContextScope.RELATED,
            source_task_id="task_0",
        )
    )

    restored = ContextWindow.from_dict(json.loads(json.dumps(window.to_dict())))

    assert restored == window


def test_to_dict_is_directly_json_serializable() -> None:
    window = make_window()
    window.append_message(Message.user("q"))
    payload = window.to_dict()

    assert isinstance(payload["summary"]["status"], str)
    assert payload["summary"]["task_id"] == payload["task_id"]
    assert json.dumps(payload)


def test_defaults_are_not_shared_between_windows() -> None:
    first = make_window()
    second = make_window()

    first.append_message(Message.user("q"))
    first.put_artifact(make_artifact(first.task_id))

    assert second.content == []
    assert second.artifacts == {}
