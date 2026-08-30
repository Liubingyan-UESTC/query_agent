"""ToolManager：发现、按意图过滤、校验、超时、异常封装与埋点。"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from pydantic import ValidationError

from agent.common.enums import IntentType
from agent.common.errors import ToolError, ToolInvocationError, ToolNotFoundError, ToolTimeoutError
from agent.common.logging import get_logger
from agent.config.settings import AppSettings, ToolSettings
from agent.tool_manage.base import BaseTool, ToolContext, ToolDeps, ToolResult
from agent.tool_manage.registry import ToolRegistry, default_registry

__all__ = ["ToolManager"]

logger = get_logger(__name__)

AllowedLookup = Callable[[IntentType | str], Sequence[str]]


class ToolManager:
    """执行循环只依赖本管理器，不直接持有具体工具。"""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        settings: ToolSettings | AppSettings | None = None,
        *,
        use_mock: bool | None = None,
        es_client: Any = None,
        field_catalog: Any = None,
        export_dir: Any = None,
        allowed_tools_lookup: AllowedLookup | None = None,
        discover: bool = True,
    ) -> None:
        if isinstance(settings, AppSettings):
            self.settings = settings.tool
        elif settings is None:
            self.settings = ToolSettings.from_env(None)
        else:
            self.settings = settings
        self._lookup = allowed_tools_lookup
        resolved_mock = self.settings.use_mock if use_mock is None else use_mock
        self.use_mock = resolved_mock
        target = registry if registry is not None else default_registry()
        if discover:
            target.discover()
        deps = ToolDeps(
            settings=self.settings,
            es_client=es_client,
            field_catalog=field_catalog,
            export_dir=export_dir or self.settings.export_dir,
        )
        self._tools = target.instantiate(deps, use_mock=resolved_mock)

    def list_tools(
        self,
        intent: IntentType | str | None = None,
        *,
        allowed_tools: Sequence[str] | None = None,
    ) -> list[BaseTool]:
        names = self._resolve_allowed(intent, allowed_tools)
        tools = list(self._tools.values())
        if names is None:
            return tools
        allowed = set(names)
        return [tool for tool in tools if tool.name in allowed]

    def get_schemas(
        self,
        intent: IntentType | str | None = None,
        *,
        allowed_tools: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            tool.to_openai_schema() for tool in self.list_tools(intent, allowed_tools=allowed_tools)
        ]

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"未注册的工具：{name}") from None

    def invoke(
        self,
        name: str,
        raw_args: Any,
        ctx: ToolContext,
        *,
        intent: IntentType | str | None = None,
        allowed_tools: Sequence[str] | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        if self._lookup is not None and intent is None and allowed_tools is None:
            return ToolResult.fail(
                ToolError("invoke 已配置意图白名单，必须传入 intent 或 allowed_tools"),
                elapsed_ms=_elapsed(started),
            )
        try:
            allowed = self._resolve_allowed(intent, allowed_tools)
        except ToolError as exc:
            return ToolResult.fail(exc, elapsed_ms=_elapsed(started))
        if allowed is not None and name not in allowed:
            return ToolResult.fail(
                ToolNotFoundError(f"工具 {name} 不在当前意图的白名单内：{list(allowed)}"),
                elapsed_ms=_elapsed(started),
            )
        try:
            tool = self.get(name)
        except ToolNotFoundError as exc:
            return ToolResult.fail(exc, elapsed_ms=_elapsed(started))

        try:
            args = tool.args_schema.model_validate({} if raw_args is None else raw_args)
        except ValidationError as exc:
            return ToolResult.fail(
                ToolInvocationError("参数校验失败", retryable=False, detail=str(exc)),
                elapsed_ms=_elapsed(started),
            )

        timeout = tool.timeout if tool.timeout is not None else self.settings.default_timeout
        try:
            result = _run_with_timeout(tool, args, ctx, timeout)
        except ToolTimeoutError as exc:
            result = ToolResult.fail(exc, elapsed_ms=_elapsed(started))
        except ToolError as exc:
            result = ToolResult.fail(exc, elapsed_ms=_elapsed(started))
        except Exception as exc:
            result = ToolResult.fail(
                ToolInvocationError(f"工具 {name} 执行异常：{exc}", detail=repr(exc)),
                elapsed_ms=_elapsed(started),
            )

        result.elapsed_ms = result.elapsed_ms or _elapsed(started)
        logger.info(
            "tool_invoked",
            extra={
                "tool": name,
                "ok": result.ok,
                "elapsed_ms": result.elapsed_ms,
                "task_id": ctx.task_id,
            },
        )
        return result

    def _resolve_allowed(
        self,
        intent: IntentType | str | None,
        allowed_tools: Sequence[str] | None,
    ) -> Sequence[str] | None:
        if allowed_tools is not None:
            return allowed_tools
        if intent is None:
            return None
        if self._lookup is None:
            raise ToolError("按意图过滤需要 allowed_tools_lookup 或显式传入 allowed_tools")
        return self._lookup(intent)


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _run_with_timeout(tool: BaseTool, args: Any, ctx: ToolContext, timeout: float) -> ToolResult:
    if timeout <= 0:
        raise ToolTimeoutError(f"工具 {tool.name} 超时上限非法：{timeout}")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tool.run, args, ctx)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            future.cancel()
            raise ToolTimeoutError(f"工具 {tool.name} 执行超时（{timeout}s）") from exc
