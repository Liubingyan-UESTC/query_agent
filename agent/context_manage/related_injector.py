"""关联任务上下文注入：把历史任务「模拟成当前任务已执行过一部分」。

注入内容标 `scope=RELATED`，参与推理但不归档。本模块不 import memory：
历史三部分通过 `TaskBundleSource` Protocol 读入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent.common.enums import ContextScope, MessageRole
from agent.common.errors import AgentMemoryError, ContextError
from agent.config.settings import AppSettings, ContextSettings
from agent.context_manage.context_manager import ContextManager
from agent.context_manage.window_policy import last_k_turns
from agent.models.artifact import Artifact
from agent.models.message import Message
from agent.models.task_summary import TaskSummary

__all__ = ["InjectionReport", "RelatedContextInjector", "TaskBundleSource"]


class TaskBundleSource(Protocol):
    """WorkingMemory.get_task_bundle / list_task_ids 的结构子集。"""

    def get_task_bundle(
        self, task_id: str
    ) -> tuple[TaskSummary, list[Message], list[Artifact]]: ...

    def list_task_ids(self) -> list[str]: ...


@dataclass
class InjectionReport:
    injected: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    message_count: int = 0
    artifact_count: int = 0


class RelatedContextInjector:
    """按 related_task_ids 顺序注入，已注入过的来源跳过，不覆盖同 id 产物。"""

    def __init__(
        self,
        source: TaskBundleSource,
        context_manager: ContextManager,
        settings: ContextSettings | AppSettings | None = None,
    ) -> None:
        self._source = source
        self._context = context_manager
        if isinstance(settings, AppSettings):
            self.settings = settings.context
        elif settings is None:
            self.settings = context_manager.settings
        else:
            self.settings = settings

    def inject(self, task_id: str, related_task_ids: list[str]) -> InjectionReport:
        report = InjectionReport()
        if not related_task_ids:
            return report
        window = self._context.load_window(task_id)
        already = {
            message.source_task_id
            for message in window.content
            if message.scope is ContextScope.RELATED and message.source_task_id
        }
        known = set(self._source.list_task_ids())
        for related_id in related_task_ids:
            if related_id == task_id or related_id in already:
                report.skipped.append(related_id)
                continue
            if related_id not in known:
                report.skipped.append(related_id)
                continue
            try:
                summary, content, artifacts = self._source.get_task_bundle(related_id)
            except (AgentMemoryError, KeyError, ValueError):
                report.skipped.append(related_id)
                continue
            messages = _build_messages(related_id, summary, content, artifacts, self.settings)
            incoming = [item for item in artifacts if item.artifact_id not in window.artifacts]
            if not messages and not incoming:
                report.skipped.append(related_id)
                continue
            try:
                self._context.inject_segment(
                    task_id,
                    messages,
                    incoming,
                    source_task_id=related_id,
                    scope=ContextScope.RELATED,
                )
            except ContextError:
                report.skipped.append(related_id)
                continue
            window = self._context.load_window(task_id)
            already.add(related_id)
            report.injected.append(related_id)
            report.message_count += len(messages)
            report.artifact_count += len(incoming)
        return report


def _build_messages(
    source_id: str,
    summary: TaskSummary,
    content: list[Message],
    artifacts: list[Artifact],
    settings: ContextSettings,
) -> list[Message]:
    intent = summary.intent.value if summary.intent is not None else "（未识别）"
    conclusion = summary.output or summary.result or "（无）"
    separator = Message.system(
        f"【关联任务 {source_id}】\n意图={intent}\n原始请求={summary.content}\n结论={conclusion}",
        source_task_id=source_id,
        scope=ContextScope.RELATED,
    )
    usable = [
        item
        for item in content
        if item.role is not MessageRole.SYSTEM
        and (item.role is not MessageRole.ASSISTANT or item.content or item.tool_calls)
    ]
    clipped = last_k_turns(usable, settings.related_task_content_limit)
    previews = [
        Message.assistant(
            f"[关联产物] artifact_id={item.artifact_id}\n{item.preview()}",
            artifact_refs=[item.artifact_id],
            source_task_id=source_id,
            scope=ContextScope.RELATED,
        )
        for item in artifacts
    ]
    return [separator, *clipped, *previews]
