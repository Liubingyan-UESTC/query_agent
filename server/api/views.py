"""四个接口。

纯 Django ``JsonResponse``，不引入 DRF——这几个端点用不上序列化器与路由器那套东西。

``@csrf_exempt``：这是给 curl / 脚本用的 JSON API，客户端拿不到也不需要 CSRF token。
本地测试服务的取舍，真要对公网提供服务必须换成正经的鉴权。
"""

from __future__ import annotations

import json
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from agent.errors import AgentError
from agent.logging_setup import get_logger
from agent.models import Task
from server.api.models import Message, ToolCall
from server.api.models import Task as TaskRow
from server.api.runtime import get_manager, run_turn

__all__ = ["chat", "health", "history", "task_detail"]

logger = get_logger("http")


@csrf_exempt
@require_POST
def chat(request: HttpRequest) -> HttpResponse:
    """跑一轮对话。

    body: ``{"message": "查一下 order-service 的错误日志"}``

    上一轮若停在澄清态，本轮输入自动作为补充说明续跑同一个任务。
    """
    session = request.agent_session  # type: ignore[attr-defined]
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError as exc:
        return _error(f"请求体不是合法 JSON：{exc}", status=400, session_id=session.session_id)
    if not isinstance(payload, dict):
        return _error("请求体必须是 JSON 对象", status=400, session_id=session.session_id)

    text = str(payload.get("message") or "").strip()
    if not text:
        return _error("message 不能为空", status=400, session_id=session.session_id)

    logger.info("会话 %s 收到提问：%s", session.session_id, text)
    try:
        task = run_turn(session.session_id, text)
    except AgentError as exc:
        # 内核已把可预期的错误收成 AgentError；到这里说明是编排层拒绝了请求
        logger.error("会话 %s 处理失败：%s", session.session_id, exc.message)
        return _error(exc.message, status=400, session_id=session.session_id)

    return JsonResponse(_task_payload(task, session.session_id))


@require_GET
def history(request: HttpRequest) -> HttpResponse:
    """本会话的全部任务与消息。读的是 SQLite，不依赖进程内存。"""
    session = request.agent_session  # type: ignore[attr-defined]
    rows = TaskRow.objects.filter(session=session).prefetch_related("messages", "tool_calls")
    return JsonResponse(
        {
            "session_id": session.session_id,
            "created": request.agent_session_created,  # type: ignore[attr-defined]
            "task_count": rows.count(),
            "tasks": [_task_row_payload(row) for row in rows],
        }
    )


@require_GET
def task_detail(request: HttpRequest, task_id: str) -> HttpResponse:
    """单个任务的详情：summary 快照 + 消息 + 工具调用。"""
    session = request.agent_session  # type: ignore[attr-defined]
    row = TaskRow.objects.filter(task_id=task_id, session=session).first()
    if row is None:
        # 限定在本会话内查：换个会话就看不到别人的任务
        return _error(f"本会话下找不到任务 {task_id}", status=404, session_id=session.session_id)
    return JsonResponse(_task_row_payload(row, include_messages=True))


@require_GET
def health(request: HttpRequest) -> HttpResponse:
    """存活探测，顺带告诉调用方现在用的是真实模型还是 MockLLM。

    配置从**正在跑的 manager** 上读，而不是再调一次 ``get_settings()``：报告要反映
    这个进程实际装配了什么，否则改了配置没重启时，探测结果会骗人。
    """
    manager = get_manager()
    settings = manager.settings
    return JsonResponse(
        {
            "status": "ok",
            "session_id": request.agent_session.session_id,  # type: ignore[attr-defined]
            "llm": {
                "mock": settings.llm.use_mock_client(),
                "model": settings.llm.primary.model,
                "base_url": settings.llm.primary.base_url,
            },
            "tools": manager.tools.names(),
            "log_file": str(settings.log.resolved_file()),
        }
    )


# ================================================================ 序列化


def _task_payload(task: Task, session_id: str) -> dict[str, Any]:
    """把内存里的 Task 渲染成响应体。"""
    summary = task.summary
    return {
        "session_id": session_id,
        "task_id": task.task_id,
        "status": task.status.value,
        "clarifying": task.status.requires_user_input,
        "intent": summary.intent.value if summary.intent else None,
        "output": summary.output,
        "result": summary.result,
        "related_task_ids": summary.related_task_ids,
        "operations": [
            {
                "step": op.index + 1,
                "description": op.description,
                "suggested_tool": op.suggested_tool,
                "status": op.status.value,
                "result": op.result,
            }
            for op in summary.operations
        ],
        "tool_calls": [
            {"call_id": row.call_id, "tool_name": row.tool_name, "status": row.status}
            for row in ToolCall.objects.filter(task_id=task.task_id)
        ],
    }


def _task_row_payload(row: TaskRow, *, include_messages: bool = True) -> dict[str, Any]:
    """把库里的一行任务渲染成响应体。"""
    payload: dict[str, Any] = {
        "task_id": row.task_id,
        "status": row.status,
        "intent": row.intent,
        "summary": row.summary_json,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
        "tool_calls": [
            {
                "call_id": call.call_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments_json,
                "status": call.status,
                "result_ref": call.result_ref,
                "created_at": call.created_at.isoformat(),
            }
            for call in row.tool_calls.all()
        ],
    }
    if include_messages:
        payload["messages"] = [_message_payload(message) for message in row.messages.all()]
    return payload


def _message_payload(message: Message) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "created_at": message.created_at.isoformat(),
    }


def _error(message: str, *, status: int, session_id: str) -> JsonResponse:
    return JsonResponse({"error": message, "session_id": session_id}, status=status)
