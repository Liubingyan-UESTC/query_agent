"""状态机测试：合法/非法转移与四个辅助属性。"""

import itertools

import pytest

from agent.enums import TaskStatus
from agent.errors import TaskStateError

TERMINALS = [
    TaskStatus.COMPLETED,
    TaskStatus.CANCELED,
    TaskStatus.FAILED,
    TaskStatus.ABORTED,
]


class TestMainline:
    def test_happy_path_transitions_are_legal(self):
        """require.md 的主线：created→intenting→planning→executing→validating→completed。"""
        chain = [
            TaskStatus.CREATED,
            TaskStatus.INTENTING,
            TaskStatus.PLANNING,
            TaskStatus.EXECUTING,
            TaskStatus.VALIDATING,
            TaskStatus.COMPLETED,
        ]
        for current, target in itertools.pairwise(chain):
            assert current.can_transition_to(target)
            assert current.transition_to(target, "task_x") is target

    def test_skipping_a_stage_is_illegal(self):
        assert not TaskStatus.CREATED.can_transition_to(TaskStatus.PLANNING)
        assert not TaskStatus.INTENTING.can_transition_to(TaskStatus.EXECUTING)

    def test_illegal_transition_raises_with_task_id(self):
        with pytest.raises(TaskStateError, match="task_42 非法状态转移: created -> completed"):
            TaskStatus.CREATED.transition_to(TaskStatus.COMPLETED, "task_42")

    def test_task_state_error_is_a_value_error(self):
        """require.md 约定非法转移抛 ValueError；调用方也可按 AgentError 统一兜住。"""
        with pytest.raises(ValueError):
            TaskStatus.CREATED.transition_to(TaskStatus.COMPLETED, "t")


class TestBranches:
    def test_intenting_can_go_to_clarifying_and_back(self):
        """意图不明 → 澄清 → 用户补充后重新识别，是控制台交互的关键闭环。"""
        assert TaskStatus.INTENTING.can_transition_to(TaskStatus.CLARIFYING)
        assert TaskStatus.CLARIFYING.can_transition_to(TaskStatus.INTENTING)

    def test_validating_can_go_back_to_executing(self):
        """校验不通过时补充执行。"""
        assert TaskStatus.VALIDATING.can_transition_to(TaskStatus.EXECUTING)

    def test_executing_can_abort_on_circuit_break(self):
        assert TaskStatus.EXECUTING.can_transition_to(TaskStatus.ABORTED)

    @pytest.mark.parametrize("terminal", TERMINALS)
    def test_terminal_states_accept_no_transition(self, terminal):
        for target in TaskStatus:
            assert not terminal.can_transition_to(target)


class TestHelperProperties:
    @pytest.mark.parametrize("terminal", TERMINALS)
    def test_is_terminal(self, terminal):
        assert terminal.is_terminal

    @pytest.mark.parametrize(
        "status",
        [TaskStatus.CREATED, TaskStatus.EXECUTING, TaskStatus.VALIDATING],
    )
    def test_non_terminal(self, status):
        assert not status.is_terminal

    def test_is_async_waiting(self):
        assert TaskStatus.WAITING.is_async_waiting
        assert TaskStatus.CLARIFYING.is_async_waiting
        assert not TaskStatus.EXECUTING.is_async_waiting

    def test_is_retryable(self):
        assert TaskStatus.RETRYING.is_retryable
        assert not TaskStatus.FAILED.is_retryable

    def test_requires_user_input(self):
        assert TaskStatus.CLARIFYING.requires_user_input
        assert TaskStatus.AWAITING_APPROVAL.requires_user_input
        # WAITING 等的是外部系统而非用户，不需要用户输入
        assert not TaskStatus.WAITING.requires_user_input

    def test_status_is_a_str_enum(self):
        """str 混入让状态可直接 json 序列化，也便于日志打印。"""
        assert TaskStatus.EXECUTING == "executing"
        assert f"{TaskStatus.EXECUTING.value}" == "executing"
