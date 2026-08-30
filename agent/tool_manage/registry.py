"""工具注册表：装饰器登记 + pkgutil 扫描 `tool_manage/tools/` + 重名检测。"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from agent.common.errors import ToolError
from agent.tool_manage.base import BaseTool, ToolDeps

__all__ = ["DEFAULT_TOOLS_PACKAGE", "ToolRegistry", "default_registry", "register_tool"]

DEFAULT_TOOLS_PACKAGE = "agent.tool_manage.tools"

_DEFAULT: ToolRegistry | None = None


class ToolRegistry:
    """登记工具类。实例化时按 `is_mock` 过滤，同名不同类立即报错。"""

    def __init__(self) -> None:
        self._classes: list[type[BaseTool]] = []
        self._discovered = False

    def add_class(self, cls: type[BaseTool]) -> type[BaseTool]:
        if not isinstance(cls, type) or not issubclass(cls, BaseTool) or cls is BaseTool:
            raise ToolError(f"register_tool 只能装饰 BaseTool 子类，实际为 {cls!r}")
        if not getattr(cls, "name", None):
            raise ToolError(f"{cls.__name__} 未声明 name")
        if cls not in self._classes:
            self._classes.append(cls)
        return cls

    def discover(self, package: str = DEFAULT_TOOLS_PACKAGE) -> None:
        """扫描包内模块以触发 `@register_tool`。同一 registry 重复扫描是空操作。"""
        if self._discovered and package == DEFAULT_TOOLS_PACKAGE:
            return
        try:
            pkg = importlib.import_module(package)
        except ModuleNotFoundError as exc:
            raise ToolError(f"无法导入工具包 {package}") from exc
        paths = getattr(pkg, "__path__", None)
        if paths is None:
            raise ToolError(f"{package} 不是包，无法扫描工具")
        for module_info in pkgutil.iter_modules(paths, prefix=pkg.__name__ + "."):
            module = importlib.import_module(module_info.name)
            for value in vars(module).values():
                if (
                    isinstance(value, type)
                    and issubclass(value, BaseTool)
                    and value is not BaseTool
                    and getattr(value, "name", None)
                ):
                    self.add_class(value)
        if package == DEFAULT_TOOLS_PACKAGE:
            self._discovered = True

    def instantiate(self, deps: ToolDeps, *, use_mock: bool = False) -> dict[str, BaseTool]:
        selected = [cls for cls in self._classes if bool(cls.is_mock) == use_mock]
        names = [cls.name for cls in selected]
        dup = sorted({name for name in names if names.count(name) > 1})
        if dup:
            raise ToolError(f"工具重名：{dup}")
        return {cls.name: cls.from_deps(deps) for cls in selected}

    @property
    def classes(self) -> tuple[type[BaseTool], ...]:
        return tuple(self._classes)


def default_registry() -> ToolRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ToolRegistry()
    return _DEFAULT


def register_tool(
    cls: type[BaseTool] | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> Any:
    """`@register_tool` 或 `@register_tool(registry=reg)`。"""

    target = registry if registry is not None else default_registry()

    def decorator(item: type[BaseTool]) -> type[BaseTool]:
        return target.add_class(item)

    if cls is None:
        return decorator
    return decorator(cls)
