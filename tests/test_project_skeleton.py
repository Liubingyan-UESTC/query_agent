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


def test_top_level_import_does_not_pull_in_layers() -> None:
    """顶层聚合导入会绕开单向依赖约束，故在干净解释器中断言其未发生。"""
    code = (
        "import sys, agent;"
        "leaked = sorted(m for m in sys.modules if m.startswith('agent.'));"
        "print(','.join(leaked))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == ""
