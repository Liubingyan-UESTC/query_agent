"""步骤 14：build_messages 输出 OpenAI 形状，且不含编排字段。"""

from __future__ import annotations

from agent.common.enums import IntentType, PromptStage
from agent.context_manage.serializer import build_message_objects, build_messages
from agent.context_manage.token_counter import CharEstimateCounter
from tests.context_manage.test_window_policy import FakeKnowledge, _legal, _window


def test_build_messages_is_openai_shaped() -> None:
    window = _window()
    payloads = build_messages(
        window,
        PromptStage.INTENT_RECOGNITION,
        FakeKnowledge(),
        counter=CharEstimateCounter(),
    )

    assert payloads[0]["role"] == "system"
    assert payloads[1]["role"] == "user"
    assert set(payloads[0]) == {"role", "content"}
    assert "scope" not in payloads[1]
    assert "message_id" not in payloads[1]
    assert "artifact_refs" not in payloads[1]


def test_build_message_objects_is_legal_for_all_stages() -> None:
    window = _window()
    knowledge = FakeKnowledge()
    for stage, intent in (
        (PromptStage.INTENT_RECOGNITION, None),
        (PromptStage.PLAN, IntentType.NEW_QUERY),
        (PromptStage.EXECUTE, IntentType.NEW_QUERY),
        (PromptStage.VALIDATE, None),
    ):
        messages = build_message_objects(
            window,
            stage,
            knowledge,
            intent=intent,
            counter=CharEstimateCounter(),
        )
        _legal(messages)
