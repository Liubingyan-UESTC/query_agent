"""会话凭证中间件：进来解析、出去回写。

放在中间件而不是每个视图里，是因为"每个响应都带上凭证"这件事不该依赖视图作者记得写。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from django.http import HttpRequest, HttpResponse

from agent.config import get_settings
from agent.logging_setup import get_logger
from server.api.session import resolve_session

__all__ = ["SessionTokenMiddleware"]

logger = get_logger("http")


class SessionTokenMiddleware:
    """给 ``request`` 挂上 ``agent_session``，并在响应里回写凭证。"""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self._get_response = get_response
        self._settings = get_settings()

    def __call__(self, request: HttpRequest) -> HttpResponse:
        server = self._settings.server
        session, created = resolve_session(request, server)
        # 视图通过 request.agent_session 拿会话，不必各自解析凭证
        request.agent_session = session  # type: ignore[attr-defined]
        request.agent_session_created = created  # type: ignore[attr-defined]
        if created:
            logger.info("新会话 %s（%s %s）", session.session_id, request.method, request.path)

        response = self._get_response(request)

        response[server.session_header] = session.session_id
        response.set_cookie(
            server.session_cookie,
            session.session_id,
            max_age=server.session_cookie_max_age,
            samesite="Lax",
            httponly=True,
        )
        return response


def current_session(request: Any) -> Any:
    """给视图用的取值助手，顺带在类型上遮掉动态属性。"""
    return request.agent_session
