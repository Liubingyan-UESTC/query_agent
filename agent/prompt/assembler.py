"""PromptAssembler：知识 + 窗口视图 → 阶段处理器唯一的 LLM 请求来源。

系统提示词已由 KnowledgeMemory 前置 `base.md`，这里不再拼一次。
各方法同时给出 `output_schema`，供 `call_structured` 使用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from agent.common.enums import IntentType, PromptStage
from agent.config.settings import AppSettings, ContextSettings, LLMPurpose
from agent.context_manage.context_manager import ContextManager
from agent.context_manage.serializer import build_message_objects
from agent.context_manage.token_counter import TokenCounter
from agent.context_manage.window_policy import StageAssembleOptions
from agent.llm.base import LLMRequest
from agent.memory_manage.knowledge_memory import KnowledgeMemory
from agent.models.base import AgentModel
from agent.models.context_window import ContextWindow
from agent.models.task_summary import Operation

__all__ = [
    "CHAT_INTENTS",
    "EMPTY_PLAN_OUTPUT_SCHEMA",
    "EXECUTE_OUTPUT_SCHEMA",
    "INTENT_OUTPUT_SCHEMA",
    "PLAN_OUTPUT_SCHEMA",
    "VALIDATE_OUTPUT_SCHEMA",
    "AssembledPrompt",
    "EmptyPlanOutput",
    "PlanOutput",
    "PlanStepOut",
    "PromptAssembler",
]

CHAT_INTENTS: frozenset[IntentType] = frozenset({IntentType.CHAT, IntentType.UNKNOWN})


class PlanStepOut(AgentModel):
    """规划步骤的结构化输出。与 skill YAML 的 operations.items 对齐。"""

    index: int
    name: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    expect: str


class PlanOutput(AgentModel):
    operations: list[PlanStepOut]


class EmptyPlanOutput(AgentModel):
    """CHAT / UNKNOWN：operations 必须是空列表。"""

    operations: list[PlanStepOut] = Field(default_factory=list, max_length=0)


INTENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["intent", "related_task_ids", "reason", "confidence"],
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "enum": IntentType.values()},
        "related_task_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

EXECUTE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "conclusion"],
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["continue", "done"]},
        "conclusion": {"type": "string"},
    },
}

_PLAN_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["index", "name", "tool", "args", "expect"],
    "additionalProperties": False,
    "properties": {
        "index": {"type": "integer"},
        "name": {"type": "string"},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "expect": {"type": "string"},
    },
}

PLAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["operations"],
    "additionalProperties": False,
    "properties": {"operations": {"type": "array", "items": _PLAN_STEP_SCHEMA}},
}

EMPTY_PLAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["operations"],
    "additionalProperties": False,
    "properties": {
        "operations": {"type": "array", "maxItems": 0, "items": _PLAN_STEP_SCHEMA},
    },
}

VALIDATE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["satisfied", "missing", "suggestion", "final_output"],
    "additionalProperties": False,
    "properties": {
        "satisfied": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
        "suggestion": {"type": "string"},
        "final_output": {"type": "string"},
    },
}


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    """一次阶段调用的完整载荷：请求体 + 结构化输出 schema。

    `output_schema` 是无 `$ref` 的手写 JSON Schema，给提示词与 json_schema 模式用。
    `output_model` 供 `call_structured` 走 pydantic（规划阶段必须有）。
    """

    request: LLMRequest
    output_schema: dict[str, Any]
    output_model: type[BaseModel] | None = None


class PromptAssembler:
    """阶段处理器只从这里取 prompt，不各自拼接知识与窗口。"""

    def __init__(
        self,
        knowledge_memory: KnowledgeMemory,
        context_manager: ContextManager | None = None,
        settings: AppSettings | ContextSettings | None = None,
        *,
        counter: TokenCounter | None = None,
    ) -> None:
        self.knowledge = knowledge_memory
        self.context_manager = context_manager
        if isinstance(settings, AppSettings):
            self.settings = settings.context
        elif isinstance(settings, ContextSettings):
            self.settings = settings
        elif context_manager is not None:
            self.settings = context_manager.settings
        else:
            self.settings = ContextSettings.from_env(None)
        self.counter = counter

    def build_intent_prompt(
        self,
        window: ContextWindow,
        summary_history: Sequence[Mapping[str, Any]],
    ) -> AssembledPrompt:
        return self._assemble(
            window,
            PromptStage.INTENT_RECOGNITION,
            output_schema=INTENT_OUTPUT_SCHEMA,
            purpose="intent",
            options=StageAssembleOptions(summary_history=summary_history),
        )

    def build_plan_prompt(self, window: ContextWindow) -> AssembledPrompt:
        intent = window.summary.intent
        if intent is None:
            raise ValueError("规划阶段窗口必须已填写 intent")
        if intent in CHAT_INTENTS:
            return self._assemble(
                window,
                PromptStage.PLAN,
                intent=intent,
                output_schema=EMPTY_PLAN_OUTPUT_SCHEMA,
                output_model=EmptyPlanOutput,
                purpose="plan",
            )
        return self._assemble(
            window,
            PromptStage.PLAN,
            intent=intent,
            output_schema=PLAN_OUTPUT_SCHEMA,
            output_model=PlanOutput,
            purpose="plan",
        )

    def build_execute_prompt(
        self,
        window: ContextWindow,
        operation: Operation,
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> AssembledPrompt:
        return self._assemble(
            window,
            PromptStage.EXECUTE,
            intent=window.summary.intent,
            output_schema=EXECUTE_OUTPUT_SCHEMA,
            purpose="execute",
            tools=[dict(item) for item in tool_schemas] if tool_schemas else None,
            options=StageAssembleOptions(operation=operation, tool_schemas=tool_schemas),
        )

    def build_validate_prompt(self, window: ContextWindow) -> AssembledPrompt:
        return self._assemble(
            window,
            PromptStage.VALIDATE,
            intent=window.summary.intent,
            output_schema=VALIDATE_OUTPUT_SCHEMA,
            purpose=None,
        )

    def _assemble(
        self,
        window: ContextWindow,
        stage: PromptStage,
        *,
        output_schema: dict[str, Any],
        purpose: LLMPurpose | None,
        intent: IntentType | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: StageAssembleOptions | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> AssembledPrompt:
        messages = build_message_objects(
            window,
            stage,
            self.knowledge,
            intent=intent,
            settings=self.settings,
            counter=self.counter,
            options=options,
        )
        return AssembledPrompt(
            request=LLMRequest(messages=messages, purpose=purpose, tools=tools),
            output_schema=output_schema,
            output_model=output_model,
        )
