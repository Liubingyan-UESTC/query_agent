"""`.devcontainer/` 已纳入版本管理，故守护「配置可解析且不含硬编码密钥」这条不变式。"""

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVCONTAINER_JSON = PROJECT_ROOT / ".devcontainer" / "devcontainer.json"

# 常见密钥前缀；容器配置中的敏感值必须写成 ${localEnv:VAR}
SECRET_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,})")

pytestmark = pytest.mark.skipif(
    not DEVCONTAINER_JSON.exists(),
    reason="未使用开发容器",
)


def _strip_jsonc_comments(text: str) -> str:
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)


def test_devcontainer_json_is_parseable() -> None:
    config = json.loads(_strip_jsonc_comments(DEVCONTAINER_JSON.read_text(encoding="utf-8")))
    assert config["name"]
    assert "dockerfile" in config["build"]


def test_devcontainer_json_has_no_hardcoded_secret() -> None:
    text = DEVCONTAINER_JSON.read_text(encoding="utf-8")
    found = SECRET_PATTERN.findall(text)
    assert not found, f"devcontainer.json 中出现硬编码密钥，请改用 ${{localEnv:VAR}}：{found}"


def test_devcontainer_workspace_mount_is_portable() -> None:
    text = DEVCONTAINER_JSON.read_text(encoding="utf-8")
    assert not re.search(r'"source=?[A-Za-z]:\\\\', text), "workspaceMount 不应写死宿主机绝对路径"
