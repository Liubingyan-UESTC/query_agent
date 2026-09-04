"""会话凭证：判断一次请求属于新会话还是已有会话。

流程按 X-Session-Id header → Cookie 的顺序找凭证：

1. **带了合法凭证** → 查（或建）对应的会话行，继续用它；
2. **带了非法凭证** → 当作没带，发一个新的。放任客户端拿任意字符串当主键，
   `sessions` 表会被垃圾键撑满，而且没有任何好处；
3. **没带凭证** → 生成新的 session_id 建行。

响应侧三处都回写（header、Cookie、响应体），curl、浏览器、前端框架都能直接取用。
"""

from __future__ import annotations

import re
from typing import Any

from agent.config import ServerSettings
from agent.models import new_session_id
from server.api.models import Session

__all__ = ["is_valid_session_id", "read_credential", "resolve_session"]

# 内核生成的形如 sess_ab12cd34ef56，另外也接受标准 UUID（含或不含连字符）
_SESSION_ID_RE = re.compile(
    r"^(sess_[0-9a-f]{6,32}|[0-9a-fA-F]{32}|[0-9a-fA-F-]{36})$",
)


def is_valid_session_id(value: str | None) -> bool:
    return bool(value) and bool(_SESSION_ID_RE.match(value or ""))


def read_credential(request: Any, settings: ServerSettings) -> str | None:
    """从请求里取会话凭证：header 优先于 Cookie。"""
    header = request.headers.get(settings.session_header)
    if header:
        return header.strip()
    cookie = request.COOKIES.get(settings.session_cookie)
    return cookie.strip() if cookie else None


def resolve_session(request: Any, settings: ServerSettings) -> tuple[Session, bool]:
    """返回 ``(会话, 是否新建)``。"""
    credential = read_credential(request, settings)
    if not is_valid_session_id(credential):
        credential = new_session_id()

    session, created = Session.objects.get_or_create(
        session_id=credential,
        defaults={
            "user_agent": request.headers.get("User-Agent"),
            "ip": _client_ip(request),
        },
    )
    return session, created


def _client_ip(request: Any) -> str | None:
    """取客户端 IP。本地测试服务，X-Forwarded-For 直接信任即可。"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")
