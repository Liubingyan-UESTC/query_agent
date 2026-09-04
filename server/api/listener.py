"""DbListener：把 :class:`~agent.events.TaskEvent` 落进四张表。

它是 Agent 事件的订阅者，**Agent 完全不知道它的存在**。每次事件做三件事：

1. `tasks` 按 task_id upsert（状态、意图、summary 快照都跟着更新）；
2. `TOOL_CALLED` 事件写一行 `tool_calls`；
3. 把黑板里还没入库的消息补写进 `messages`。

**顺序是有意的**：先写 tool_calls 再写 messages。tool 消息带外键指向工具调用，
先写调用行才能让外键落到实处（工具执行早于 tool 消息入窗，事件顺序天然满足）。

消息用 ``bulk_create(ignore_conflicts=True)`` 幂等补写：消息一旦生成就不再变化，
主键冲突就是"已经写过了"，直接跳过比先查一遍再插更省事。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.db import transaction

from agent.config import AppSettings, get_settings
from agent.enums import MessageRole
from agent.events import EventKind, TaskEvent
from agent.logging_setup import get_logger
from server.api.models import Message, Session, Task, ToolCall

__all__ = ["DbListener"]

logger = get_logger("db")


class DbListener:
    """Agent 事件 → SQLite 四张表。"""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def __call__(self, event: TaskEvent) -> None:
        # 整个事件在一个事务里：要么四张表一起前进，要么这次事件什么都不留
        with transaction.atomic():
            session = self._ensure_session(event.session_id)
            task = self._upsert_task(event, session)
            if event.kind is EventKind.TOOL_CALLED:
                self._write_tool_call(event, task)
            self._sync_messages(event, task)

    # ---------------------------------------------------------------- 各表

    def _ensure_session(self, session_id: str) -> Session:
        """会话行通常由中间件建好；这里兜底，好让 CLI 之类的非 HTTP 入口也能落库。"""
        session, _ = Session.objects.get_or_create(session_id=session_id)
        return session

    def _upsert_task(self, event: TaskEvent, session: Session) -> Task:
        summary = event.window.summary if event.window else None
        defaults: dict[str, Any] = {
            "session": session,
            "status": event.status.value,
        }
        if summary is not None:
            defaults["intent"] = summary.intent.value if summary.intent else None
            defaults["summary_json"] = summary.to_dict()
        task, _ = Task.objects.update_or_create(task_id=event.task_id, defaults=defaults)
        return task

    def _write_tool_call(self, event: TaskEvent, task: Task) -> None:
        payload = event.payload
        call_id = payload.get("call_id")
        if not call_id:
            logger.warning("工具事件缺少 call_id，跳过落库：%s", payload)
            return

        result_json, result_ref = self._store_result(call_id, payload.get("result"))
        ToolCall.objects.update_or_create(
            call_id=call_id,
            defaults={
                "task": task,
                "tool_name": payload.get("tool_name", ""),
                "arguments_json": _parse_arguments(payload.get("arguments")),
                "result_json": result_json
                if payload.get("ok")
                else {"error": payload.get("error")},
                "result_ref": result_ref,
                "status": "ok" if payload.get("ok") else "failed",
            },
        )

    def _sync_messages(self, event: TaskEvent, task: Task) -> None:
        if event.window is None:
            return
        known_calls = set(ToolCall.objects.filter(task=task).values_list("call_id", flat=True))

        rows = []
        for message in event.window.content:
            tool_call_id = message.tool_call_id
            if tool_call_id and tool_call_id not in known_calls:
                # 外键指不到就置空，不让一条消息把整个请求打挂
                logger.warning(
                    "消息 %s 引用的工具调用 %s 尚未入库，tool_call_id 置空",
                    message.message_id,
                    tool_call_id,
                )
                tool_call_id = None
            rows.append(
                Message(
                    message_id=message.message_id,
                    task=task,
                    role=_role_value(message.role),
                    content=message.content,
                    tool_call_id=tool_call_id,
                    created_at=message.created_at,
                )
            )
        if rows:
            # 主键冲突 = 这条消息之前已经写过，消息本身不可变，跳过即正确
            Message.objects.bulk_create(rows, ignore_conflicts=True)

    # ---------------------------------------------------------------- 大结果

    def _store_result(self, call_id: str, result: Any) -> tuple[Any, str | None]:
        """小结果内联，大结果落文件、表里只留引用。

        一次检索可能带回几百条日志，全塞进 JSONField 会让 `select *` 直接卡住，
        也让 sqlite3 命令行没法读。
        """
        if result is None:
            return None, None
        text = json.dumps(result, ensure_ascii=False)
        limit = self._settings.server.inline_result_max_chars
        if len(text) <= limit:
            return result, None

        directory = self._settings.server.resolved_tool_result_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{call_id}.json"
        path.write_text(text, encoding="utf-8")
        reference = str(_relative(path))
        return (
            {
                "_truncated": True,
                "chars": len(text),
                "result_ref": reference,
                "preview": text[:limit],
            },
            reference,
        )


def _relative(path: Path) -> Path:
    from agent.config import PROJECT_ROOT

    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:  # 目录被配到项目外，就存绝对路径
        return path


def _parse_arguments(raw: Any) -> Any:
    """``arguments`` 在事件里是模型原样返回的 JSON 字符串，尽量解析成对象存。"""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 模型吐了非法 JSON——原样留着，排查时要看到它到底写了什么
            return {"_raw": raw}
    return raw or {}


def _role_value(role: MessageRole | str) -> str:
    return role.value if isinstance(role, MessageRole) else str(role)
