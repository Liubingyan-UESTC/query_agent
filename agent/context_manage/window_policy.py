"""阶段视图定义与 token 预算裁剪。

装配器（步骤 15）只消费本模块产出的消息列表。裁剪优先级：
`CURRENT > RELATED > HISTORY`；同优先级按时间由旧到新丢弃；
工具错误消息优先保留。assistant + 对应 tool 必须整组留下或整组丢掉，
否则裁剪后的序列不再是合法 OpenAI 对话。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from agent.common.enums import ContextScope, IntentType, MessageRole, PromptStage
from agent.config.settings import ContextSettings
from agent.context_manage.token_counter import CharEstimateCounter, TokenCounter
from agent.models.artifact import Artifact
from agent.models.context_window import ContextWindow
from agent.models.message import Message
from agent.models.task_summary import INTENT_PROMPT_FIELDS, Operation, TaskSummary

__all__ = [
    "FEWSHOT_SOURCE_TASK_ID",
    "HISTORY_SUMMARY_FIELDS",
    "KnowledgeView",
    "SkillView",
    "StageAssembleOptions",
    "TokenBudget",
    "assemble_stage_messages",
    "clip_history_summaries",
    "clip_to_budget",
    "group_turns",
    "is_tool_error",
    "last_k_turns",
]

HISTORY_SUMMARY_FIELDS: tuple[str, ...] = ("task_id", "content", "status", "intent")
FEWSHOT_SOURCE_TASK_ID = "fewshot"

_SCOPE_DROP_RANK: dict[ContextScope, int] = {
    ContextScope.HISTORY: 0,
    ContextScope.RELATED: 1,
    ContextScope.CURRENT: 2,
}

_TOOL_ERROR_MARKERS = ("失败", "error", "exception", "timeout", "timed out")


class SkillView(Protocol):
    """KnowledgeMemory.Skill 的结构子集，避免 context_manage 引用 memory 层。"""

    name: str
    description: str
    allowed_tools: list[str]
    field_dict_refs: list[str]
    few_shots: list[Message]
    output_schema: dict[str, Any]


class KnowledgeView(Protocol):
    """KnowledgeMemory 的只读面。实现方在 memory 层，这里只按结构调用。"""

    def get_system_prompt(
        self, stage: PromptStage | str, intent: IntentType | str | None = None
    ) -> str: ...

    def get_skill(self, intent: IntentType | str) -> SkillView: ...

    def get_field_dict(self, index: str | None = None) -> str: ...

    def get_few_shots(self, intent: IntentType | str) -> list[Message]: ...

    def get_allowed_tools(self, intent: IntentType | str) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """由 ContextSettings 占比换算出的绝对预算。"""

    max_total: int
    summary: int
    content: int
    artifacts: int

    @classmethod
    def from_settings(cls, settings: ContextSettings) -> TokenBudget:
        parts = settings.token_budget()
        return cls(
            max_total=settings.max_total_tokens,
            summary=parts["summary"],
            content=parts["content"],
            artifacts=parts["artifacts"],
        )


@dataclass(frozen=True, slots=True)
class StageAssembleOptions:
    """装配时的可选输入。意图识别要历史摘要，执行要当前步骤与工具 schema。"""

    summary_history: Sequence[Mapping[str, Any]] = ()
    operation: Operation | None = None
    tool_schemas: Sequence[Mapping[str, Any]] = ()
    recent_content_limit: int | None = None
    content_digest_chars: int | None = None
    history_summary_limit: int | None = None


@dataclass(slots=True)
class _Unit:
    messages: list[Message]
    scope: ContextScope
    created_at: datetime
    keep_preferred: bool
    origin: int


def assemble_stage_messages(
    window: ContextWindow,
    stage: PromptStage | str,
    knowledge: KnowledgeView,
    *,
    intent: IntentType | str | None = None,
    settings: ContextSettings | None = None,
    budget: TokenBudget | None = None,
    counter: TokenCounter | None = None,
    options: StageAssembleOptions | None = None,
) -> list[Message]:
    """按阶段组装 messages，并在超预算时裁剪。保证 system 唯一且在首位。"""
    resolved = stage if isinstance(stage, PromptStage) else PromptStage.from_str(stage)
    ctx = settings if settings is not None else ContextSettings.from_env(None)
    opts = options if options is not None else StageAssembleOptions()
    token_budget = budget if budget is not None else TokenBudget.from_settings(ctx)
    token_counter = counter if counter is not None else CharEstimateCounter()
    resolved_intent = _resolve_intent(intent, window.summary)

    built = _build_stage(
        window, resolved, knowledge, resolved_intent, ctx, opts, token_budget, token_counter
    )
    return clip_to_budget(built, token_budget, token_counter)


def clip_history_summaries(
    history: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    fields: Sequence[str] = HISTORY_SUMMARY_FIELDS,
) -> list[dict[str, Any]]:
    """取最近 N 条并按字段投影。缺字段的键直接跳过，不补 None。"""
    if limit <= 0:
        return []
    selected = list(history)[-limit:]
    return [{key: item[key] for key in fields if key in item} for item in selected]


def is_tool_error(message: Message) -> bool:
    """工具失败消息：显式 meta.error，或正文带失败标记。"""
    if message.role is not MessageRole.TOOL:
        return False
    flag = message.meta.get("error")
    if flag is True or (isinstance(flag, str) and flag.strip()):
        return True
    text = message.content.lower()
    return any(marker in text for marker in _TOOL_ERROR_MARKERS)


def group_turns(messages: Sequence[Message]) -> list[list[Message]]:
    """把对话切成轮次：单条 user / 无工具 assistant / assistant+后续 tool。"""
    turns: list[list[Message]] = []
    index = 0
    items = list(messages)
    while index < len(items):
        current = items[index]
        if current.role is MessageRole.ASSISTANT and current.tool_calls:
            group = [current]
            index += 1
            while index < len(items) and items[index].role is MessageRole.TOOL:
                group.append(items[index])
                index += 1
            turns.append(group)
            continue
        turns.append([current])
        index += 1
    return turns


def last_k_turns(messages: Sequence[Message], limit: int) -> list[Message]:
    """保留最近 K 轮。system 不计入轮次，由调用方另行处理。"""
    if limit <= 0:
        return []
    turns = group_turns(messages)
    kept: list[Message] = []
    for turn in turns[-limit:]:
        kept.extend(turn)
    return kept


def clip_to_budget(
    messages: Sequence[Message],
    budget: TokenBudget,
    counter: TokenCounter,
) -> list[Message]:
    """按优先级丢掉整组，直到总 token 落入 max_total。"""
    items = list(messages)
    if not items:
        return []
    if counter.count_messages(items) <= budget.max_total:
        return _ensure_single_system(items)

    units = _to_units(items)
    protected = units[0] if units and units[0].messages[0].role is MessageRole.SYSTEM else None
    droppable = [unit for unit in units if unit is not protected]
    droppable.sort(
        key=lambda unit: (
            _SCOPE_DROP_RANK[unit.scope],
            1 if unit.keep_preferred else 0,
            unit.created_at,
            unit.origin,
        )
    )

    dropped: set[int] = set()
    for unit in droppable:
        remaining = _units_to_messages(units, dropped)
        if counter.count_messages(remaining) <= budget.max_total:
            break
        dropped.add(id(unit))

    clipped = _ensure_single_system(_units_to_messages(units, dropped))
    if counter.count_messages(clipped) <= budget.max_total:
        return clipped
    return _truncate_bodies(clipped, budget.max_total, counter)


def _build_stage(
    window: ContextWindow,
    stage: PromptStage,
    knowledge: KnowledgeView,
    intent: IntentType | None,
    settings: ContextSettings,
    options: StageAssembleOptions,
    budget: TokenBudget,
    counter: TokenCounter,
) -> list[Message]:
    if stage is PromptStage.INTENT_RECOGNITION:
        return _build_intent(window, knowledge, settings, options, budget, counter)
    if stage is PromptStage.PLAN:
        if intent is None:
            raise ValueError("规划阶段必须提供 intent")
        return _build_plan(window, knowledge, intent, settings, options, budget, counter)
    if stage is PromptStage.EXECUTE:
        return _build_execute(window, knowledge, intent, settings, options, budget, counter)
    if stage is PromptStage.VALIDATE:
        return _build_validate(window, knowledge, settings, options, budget, counter)
    raise ValueError(f"未知装配阶段：{stage}")


def _build_intent(
    window: ContextWindow,
    knowledge: KnowledgeView,
    settings: ContextSettings,
    options: StageAssembleOptions,
    budget: TokenBudget,
    counter: TokenCounter,
) -> list[Message]:
    system = knowledge.get_system_prompt(PromptStage.INTENT_RECOGNITION)
    limit = (
        options.history_summary_limit
        if options.history_summary_limit is not None
        else settings.history_summary_limit
    )
    history = clip_history_summaries(options.summary_history, limit=limit)
    current = window.summary.to_prompt_dict(INTENT_PROMPT_FIELDS)
    body = _join_sections(
        ("当前任务摘要", _dump(current, budget.summary, counter)),
        ("历史任务摘要", _dump(history, budget.summary, counter)),
    )
    return [Message.system(system), Message.user(body)]


def _build_plan(
    window: ContextWindow,
    knowledge: KnowledgeView,
    intent: IntentType,
    settings: ContextSettings,
    options: StageAssembleOptions,
    budget: TokenBudget,
    counter: TokenCounter,
) -> list[Message]:
    skill = knowledge.get_skill(intent)
    system = _join_prompts(
        knowledge.get_system_prompt(PromptStage.PLAN, intent),
        _render_skill(skill),
        _render_field_dicts(knowledge, skill.field_dict_refs),
    )
    messages = [Message.system(system), *_copy_few_shots(knowledge, intent)]
    digest_limit = _digest_limit(settings, options)
    messages.append(
        Message.user(
            _join_sections(
                ("当前任务摘要", _dump(window.summary.to_dict(), budget.summary, counter)),
                ("产物索引", _artifact_index_text(window, budget.artifacts, counter)),
                ("对话摘要", _content_digest(window, digest_limit)),
            )
        )
    )
    return messages


def _build_execute(
    window: ContextWindow,
    knowledge: KnowledgeView,
    intent: IntentType | None,
    settings: ContextSettings,
    options: StageAssembleOptions,
    budget: TokenBudget,
    counter: TokenCounter,
) -> list[Message]:
    parts = [knowledge.get_system_prompt(PromptStage.EXECUTE, intent)]
    if options.operation is not None:
        parts.append("当前步骤：\n" + _dump(options.operation.to_dict(), budget.summary, counter))
    if options.tool_schemas:
        parts.append("可用工具：\n" + _dump(list(options.tool_schemas), budget.summary, counter))
    messages = [Message.system(_join_prompts(*parts))]
    recent = (
        options.recent_content_limit
        if options.recent_content_limit is not None
        else settings.recent_content_limit
    )
    conversation = last_k_turns(_conversation(window), recent)
    messages.extend(conversation)
    previews = _related_previews(window, options.operation, budget.artifacts, counter)
    if previews:
        messages.append(Message.user(previews))
    return messages


def _build_validate(
    window: ContextWindow,
    knowledge: KnowledgeView,
    settings: ContextSettings,
    options: StageAssembleOptions,
    budget: TokenBudget,
    counter: TokenCounter,
) -> list[Message]:
    system = knowledge.get_system_prompt(PromptStage.VALIDATE)
    digest_limit = _digest_limit(settings, options)
    body = _join_sections(
        ("原始请求", window.summary.content),
        ("当前任务摘要", _dump(window.summary.to_dict(), budget.summary, counter)),
        ("对话摘要", _content_digest(window, digest_limit)),
        ("产物索引", _artifact_index_text(window, budget.artifacts, counter)),
    )
    return [Message.system(system), Message.user(body)]


def _resolve_intent(
    intent: IntentType | str | None,
    summary: TaskSummary,
) -> IntentType | None:
    if intent is None:
        return summary.intent
    return intent if isinstance(intent, IntentType) else IntentType.from_str(intent)


def _conversation(window: ContextWindow) -> list[Message]:
    """窗口对话去掉 system。阶段系统提示词会单独放在首位。"""
    return [item for item in window.content if item.role is not MessageRole.SYSTEM]


def _content_digest(window: ContextWindow, max_chars: int) -> str:
    lines: list[str] = []
    for message in _conversation(window):
        text = message.content.replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        extra = ""
        if message.artifact_refs:
            extra = f" refs={message.artifact_refs}"
        lines.append(f"[{message.role.value}/{message.scope.value}] {text}{extra}")
    return "\n".join(lines) if lines else "（空）"


def _artifact_index_text(window: ContextWindow, limit: int, counter: TokenCounter) -> str:
    items = [item.to_dict() for item in window.list_artifact_index(scope=None)]
    return _dump(items, limit, counter)


def _related_previews(
    window: ContextWindow,
    operation: Operation | None,
    limit: int,
    counter: TokenCounter,
) -> str:
    wanted: list[str] = []
    if operation is not None:
        ref = operation.result_ref
        if ref:
            wanted.append(ref)
        raw = operation.args.get("artifact_id")
        if isinstance(raw, str) and raw:
            wanted.append(raw)
    artifacts: list[Artifact] = []
    seen: set[str] = set()
    for artifact_id in wanted:
        if artifact_id in seen or artifact_id not in window.artifacts:
            continue
        seen.add(artifact_id)
        artifacts.append(window.artifacts[artifact_id])
    if not artifacts:
        return ""
    blocks = [f"artifact_id={item.artifact_id}\n{item.preview()}" for item in artifacts]
    return _clip_to_tokens("相关产物预览：\n" + "\n\n".join(blocks), limit, counter)


def _copy_few_shots(knowledge: KnowledgeView, intent: IntentType) -> list[Message]:
    """从 KnowledgeMemory 取副本。就地改 assembled messages 不会污染缓存。"""
    return [
        shot.model_copy(
            update={"scope": ContextScope.HISTORY, "source_task_id": FEWSHOT_SOURCE_TASK_ID}
        )
        for shot in knowledge.get_few_shots(intent)
    ]


def _render_skill(skill: SkillView) -> str:
    tools = ", ".join(skill.allowed_tools) if skill.allowed_tools else "（无）"
    return f"Skill：{skill.name}\n{skill.description}\n允许的工具：{tools}"


def _render_field_dicts(knowledge: KnowledgeView, refs: Sequence[str]) -> str:
    if not refs:
        return ""
    parts = [knowledge.get_field_dict(name) for name in refs]
    return "\n".join(part.strip() for part in parts)


def _digest_limit(settings: ContextSettings, options: StageAssembleOptions) -> int:
    if options.content_digest_chars is not None:
        return options.content_digest_chars
    return settings.content_digest_chars


def _dump(payload: Any, limit: int, counter: TokenCounter) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return _clip_to_tokens(text, limit, counter)


def _clip_to_tokens(text: str, limit: int, counter: TokenCounter) -> str:
    if limit <= 0:
        return ""
    if counter.count_text(text) <= limit:
        return text
    # 按字符二分，直到计数落入预算
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid] + ("…" if mid < len(text) else "")
        if counter.count_text(candidate) <= limit:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _join_prompts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) + "\n"


def _join_sections(*sections: tuple[str, str]) -> str:
    blocks = [f"## {title}\n{body}" for title, body in sections if body]
    return "\n\n".join(blocks)


def _to_units(messages: Sequence[Message]) -> list[_Unit]:
    units: list[_Unit] = []
    for origin, turn in enumerate(group_turns(messages)):
        head = turn[0]
        units.append(
            _Unit(
                messages=list(turn),
                scope=head.scope,
                created_at=head.created_at,
                keep_preferred=any(is_tool_error(item) for item in turn)
                or head.role is MessageRole.SYSTEM,
                origin=origin,
            )
        )
    return units


def _units_to_messages(units: Sequence[_Unit], dropped: set[int]) -> list[Message]:
    kept: list[Message] = []
    for unit in units:
        if id(unit) in dropped:
            continue
        kept.extend(unit.messages)
    return kept


def _ensure_single_system(messages: Sequence[Message]) -> list[Message]:
    """第一段 system 保留；其余 system 降为 user，避免双 system 被网关拒。"""
    items = list(messages)
    if not items:
        return []
    first_system: Message | None = None
    rest: list[Message] = []
    for index, message in enumerate(items):
        if message.role is MessageRole.SYSTEM:
            if first_system is None and index == 0:
                first_system = message
                continue
            rest.append(
                message.model_copy(
                    update={
                        "role": MessageRole.USER,
                        "content": f"【系统摘录】{message.content}",
                    }
                )
            )
            continue
        rest.append(message)
    if first_system is None:
        return list(items)
    return [first_system, *rest]


def _truncate_bodies(
    messages: Sequence[Message],
    limit: int,
    counter: TokenCounter,
) -> list[Message]:
    """整组裁完仍超预算时，从最旧的非 system 消息截断正文。"""
    items = list(messages)
    for index in range(len(items) - 1, -1, -1):
        if counter.count_messages(items) <= limit:
            return items
        message = items[index]
        if message.role is MessageRole.SYSTEM:
            continue
        if not message.content:
            continue
        # 逐步腰斩，直到本条不再是超限主因
        content = message.content
        while content and counter.count_messages(items) > limit:
            content = content[: max(len(content) // 2, 0)]
            if content and not content.endswith("…"):
                content = content[:-1] + "…" if len(content) > 1 else "…"
            items[index] = message.model_copy(update={"content": content})
            message = items[index]
    return items
