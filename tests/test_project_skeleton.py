"""步骤 1 验收：包结构完整、可导入，且顶层包不做聚合导入。"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBPACKAGES = [
    "agent.common",
    "agent.config",
    "agent.models",
    "agent.store",
    "agent.llm",
    "agent.memory_manage",
    "agent.knowledge",
    "agent.context_manage",
    "agent.prompt",
    "agent.tool_manage",
    "agent.tool_manage.tools",
    "agent.task_manage",
    "agent.task_manage.stages",
]


def test_agent_package_exposes_version() -> None:
    agent = importlib.import_module("agent")
    assert isinstance(agent.__version__, str)
    assert agent.__version__


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_is_importable(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def _imported_agent_modules(module_name: str) -> set[str]:
    """在干净解释器中导入指定模块，返回它实际拉入的全部 agent 子模块。"""
    code = (
        f"import sys, {module_name};"
        "print(','.join(sorted(m for m in sys.modules if m.startswith('agent.'))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {name for name in completed.stdout.strip().split(",") if name}


def test_top_level_import_does_not_pull_in_layers() -> None:
    """顶层聚合导入会绕开单向依赖约束，故在干净解释器中断言其未发生。"""
    assert _imported_agent_modules("agent") == set()


def test_common_layer_depends_on_nothing_internal() -> None:
    """common 是依赖图的根。

    它一旦反向引用 config（后者已依赖 common 以取得 ConfigError），就会形成导入环。
    """
    pulled = _imported_agent_modules("agent.common")
    foreign = {name for name in pulled if not name.startswith("agent.common")}

    assert not foreign, f"agent.common 不应依赖内部其他子包，实际拉入：{sorted(foreign)}"


def test_config_layer_only_depends_on_common() -> None:
    pulled = _imported_agent_modules("agent.config")
    allowed = ("agent.common", "agent.config")
    foreign = {name for name in pulled if not name.startswith(allowed)}

    assert not foreign, f"agent.config 只应依赖 common，实际额外拉入：{sorted(foreign)}"
