"""步骤 12 验收：各意图 skill 完整加载、缺失报明确错误、字段字典稳定、reload 生效。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent.common.enums import IntentType, MessageRole, PromptStage
from agent.common.errors import AgentMemoryError
from agent.memory_manage.knowledge_memory import (
    IndexDef,
    KnowledgeMemory,
    Skill,
    render_indexes,
)

PROJECT_KNOWLEDGE = Path(__file__).resolve().parents[2] / "agent" / "knowledge"


def _clone(tmp_path: Path) -> Path:
    dest = tmp_path / "knowledge"
    shutil.copytree(
        PROJECT_KNOWLEDGE,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.py"),
    )
    return dest


def test_packaged_skills_load_for_every_intent() -> None:
    memory = KnowledgeMemory()

    for intent in IntentType:
        skill = memory.get_skill(intent)
        assert skill.name
        assert memory.get_few_shots(intent.value)
        prompt = memory.get_system_prompt(PromptStage.PLAN, intent)
        assert "查询 Agent" in prompt
        body = (PROJECT_KNOWLEDGE / skill.system_prompt_ref).read_text(encoding="utf-8").strip()
        assert body in prompt

    assert memory.get_allowed_tools(IntentType.NEW_QUERY) == ["search"]
    assert memory.get_allowed_tools(IntentType.ANALYSIS) == ["search", "analysis"]
    assert memory.get_allowed_tools(IntentType.EXPORT) == ["search", "export"]
    assert memory.get_allowed_tools(IntentType.CHAT) == []
    assert memory.get_allowed_tools("unknown") == []


def test_system_prompt_prepends_base_for_non_plan_stages() -> None:
    memory = KnowledgeMemory()
    text = memory.get_system_prompt("intent_recognition")

    assert text.startswith("你是面向 Kibana")
    assert "当前阶段是意图识别" in text
    assert text.endswith("\n")


def test_plan_stage_requires_intent() -> None:
    memory = KnowledgeMemory()
    with pytest.raises(AgentMemoryError, match="必须提供 intent"):
        memory.get_system_prompt(PromptStage.PLAN)


def test_field_dict_render_is_stable() -> None:
    memory = KnowledgeMemory()
    full = memory.get_field_dict()

    assert memory.get_field_dict() == full
    assert full.startswith("# 字段字典\n")
    assert "## logs-app" in full
    assert "## metrics-host" in full
    assert "| @timestamp | date | 事件发生时间 |" in full
    assert "| mem_pct | float | 内存使用率百分比 |  |  |" in full

    one = memory.get_field_dict("metrics-host")
    assert one == memory.get_field_dict("metrics-host")
    assert "## logs-app" not in one
    assert "## metrics-host" in one


def test_catalog_accessors_match_field_dict() -> None:
    memory = KnowledgeMemory()

    assert memory.has_index("logs-app")
    assert not memory.has_index("no-such")
    assert memory.list_indexes() == ["logs-app", "metrics-host"]
    assert "@timestamp" in memory.field_names("logs-app")
    assert memory.get_index("logs-app").name == "logs-app"


def test_unknown_index_is_explicit_error() -> None:
    memory = KnowledgeMemory()
    with pytest.raises(AgentMemoryError, match="没有索引 'no-such'"):
        memory.get_field_dict("no-such")


def test_render_indexes_skips_empty_description() -> None:
    index = IndexDef.model_validate(
        {"name": "only", "fields": [{"name": "id", "type": "keyword", "meaning": "主键"}]}
    )
    assert render_indexes([index]) == (
        "# 字段字典\n"
        "\n"
        "## only\n"
        "\n"
        "| 字段 | 类型 | 含义 | 可选值 | 示例 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| id | keyword | 主键 |  |  |\n"
    )


def test_reload_picks_up_prompt_edits(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    memory = KnowledgeMemory(root)
    before = memory.get_system_prompt(PromptStage.EXECUTE)
    (root / "system_prompts" / "execute.md").write_text("执行阶段已热更新\n", encoding="utf-8")

    assert memory.get_system_prompt(PromptStage.EXECUTE) == before
    memory.reload()
    assert "执行阶段已热更新" in memory.get_system_prompt(PromptStage.EXECUTE)
    assert "查询 Agent" in memory.get_system_prompt(PromptStage.EXECUTE)


def test_reload_keeps_snapshot_when_new_load_fails(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    memory = KnowledgeMemory(root)
    (root / "skills" / "chat.yaml").unlink()

    with pytest.raises(AgentMemoryError, match="缺少知识资源"):
        memory.reload()
    assert memory.get_skill(IntentType.CHAT).name == "日常对话"


def test_missing_skill_file(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "skills" / "chat.yaml").unlink()
    with pytest.raises(AgentMemoryError, match=r"chat\.yaml"):
        KnowledgeMemory(root)


def test_empty_prompt_file(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "system_prompts" / "base.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="知识资源为空"):
        KnowledgeMemory(root)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "skills" / "chat.yaml").write_text("{{broken", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="YAML 无法解析"):
        KnowledgeMemory(root)


def test_yaml_must_be_object(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "skills" / "chat.yaml").write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="YAML 必须是对象"):
        KnowledgeMemory(root)


def test_skill_rejects_unknown_field(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    path = root / "skills" / "chat.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "unexpected: true\n", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="结构不合法"):
        KnowledgeMemory(root)


def test_skill_rejects_duplicate_tools() -> None:
    with pytest.raises(Exception, match="重复条目"):
        Skill.model_validate(
            {
                "name": "x",
                "description": "d",
                "system_prompt_ref": "system_prompts/plan_chat.md",
                "allowed_tools": ["search", "search"],
                "field_dict_refs": [],
                "few_shots": [{"role": "user", "content": "q"}],
                "output_schema": {},
            }
        )


@pytest.mark.parametrize("tools", [("",), ("  ",)])
def test_skill_rejects_blank_tool_entry(tools: tuple[str, ...]) -> None:
    with pytest.raises(Exception, match="不能为空"):
        Skill.model_validate(
            {
                "name": "x",
                "description": "d",
                "system_prompt_ref": "system_prompts/plan_chat.md",
                "allowed_tools": list(tools),
                "field_dict_refs": [],
                "few_shots": [{"role": "user", "content": "q"}],
                "output_schema": {},
            }
        )


def test_missing_field_dict_ref(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    path = root / "skills" / "chat.yaml"
    text = path.read_text(encoding="utf-8").replace(
        "field_dict_refs: []", "field_dict_refs: [missing-idx]"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="不存在的字段字典"):
        KnowledgeMemory(root)


def test_prompt_ref_outside_root(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    path = root / "skills" / "chat.yaml"
    text = path.read_text(encoding="utf-8").replace(
        "system_prompt_ref: system_prompts/plan_chat.md",
        "system_prompt_ref: ../secrets.md",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="越出知识根目录"):
        KnowledgeMemory(root)


def test_prompt_ref_must_be_relative(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    path = root / "skills" / "chat.yaml"
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace(
            "system_prompt_ref: system_prompts/plan_chat.md",
            "system_prompt_ref: /tmp/plan.md",
        ),
        encoding="utf-8",
    )
    with pytest.raises(AgentMemoryError, match="相对路径"):
        KnowledgeMemory(root)

    path.write_text(
        original.replace(
            "system_prompt_ref: system_prompts/plan_chat.md",
            "system_prompt_ref: C:/plan.md",
        ),
        encoding="utf-8",
    )
    with pytest.raises(AgentMemoryError, match="相对路径"):
        KnowledgeMemory(root)


def test_prompt_ref_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    path = root / "skills" / "chat.yaml"
    text = path.read_text(encoding="utf-8").replace(
        "system_prompt_ref: system_prompts/plan_chat.md",
        "system_prompt_ref: ' system_prompts/plan_chat.md'",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="非法资源引用"):
        KnowledgeMemory(root)


def test_prompt_ref_unlisted_file(tmp_path: Path) -> None:
    """`system_prompt_ref` 不必等于 `plan_<intent>.md`，不在必选集合里的文件也会加载。"""
    root = _clone(tmp_path)
    extra = root / "system_prompts" / "custom.md"
    extra.write_text("这是额外规划词\n", encoding="utf-8")
    path = root / "skills" / "chat.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "system_prompt_ref: system_prompts/plan_chat.md",
            "system_prompt_ref: system_prompts/custom.md",
        ),
        encoding="utf-8",
    )
    memory = KnowledgeMemory(root)
    assert "这是额外规划词" in memory.get_system_prompt(PromptStage.PLAN, IntentType.CHAT)


def test_prompt_ref_missing_file(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    path = root / "skills" / "chat.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "system_prompt_ref: system_prompts/plan_chat.md",
            "system_prompt_ref: system_prompts/missing.md",
        ),
        encoding="utf-8",
    )
    with pytest.raises(AgentMemoryError, match="缺少知识资源"):
        KnowledgeMemory(root)


def test_catalog_indexes_must_be_non_empty_list(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "schemas" / "kibana_fields.yaml").write_text("indexes: []\n", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="非空列表"):
        KnowledgeMemory(root)


def test_catalog_indexes_must_be_a_list(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "schemas" / "kibana_fields.yaml").write_text("indexes: not-a-list\n", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="非空列表"):
        KnowledgeMemory(root)


def test_catalog_index_entry_must_be_object(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "schemas" / "kibana_fields.yaml").write_text(
        "indexes:\n  - just-a-string\n", encoding="utf-8"
    )
    with pytest.raises(AgentMemoryError, match="元素必须是对象"):
        KnowledgeMemory(root)


def test_catalog_rejects_invalid_field(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "schemas" / "kibana_fields.yaml").write_text(
        "indexes:\n  - name: x\n    fields:\n      - name: a\n        type: keyword\n",
        encoding="utf-8",
    )
    with pytest.raises(AgentMemoryError, match="结构不合法"):
        KnowledgeMemory(root)


def test_duplicate_index_names(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "schemas" / "kibana_fields.yaml").write_text(
        "indexes:\n"
        "  - name: dup\n    fields:\n      - name: a\n        type: keyword\n        meaning: m\n"
        "  - name: dup\n    fields:\n      - name: b\n        type: keyword\n        meaning: m\n",
        encoding="utf-8",
    )
    with pytest.raises(AgentMemoryError, match="重复索引名"):
        KnowledgeMemory(root)


def test_duplicate_field_names(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    (root / "schemas" / "kibana_fields.yaml").write_text(
        "indexes:\n"
        "  - name: x\n    fields:\n"
        "      - name: a\n        type: keyword\n        meaning: m\n"
        "      - name: a\n        type: keyword\n        meaning: n\n",
        encoding="utf-8",
    )
    with pytest.raises(AgentMemoryError, match="重复字段名"):
        KnowledgeMemory(root)


def test_missing_root_directory(tmp_path: Path) -> None:
    with pytest.raises(AgentMemoryError, match="不是目录"):
        KnowledgeMemory(tmp_path / "no-such")


def test_root_cannot_be_a_file(tmp_path: Path) -> None:
    file_path = tmp_path / "not-dir"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(AgentMemoryError, match="不是目录"):
        KnowledgeMemory(file_path)


def test_resolve_rejects_empty_relative(tmp_path: Path) -> None:
    from agent.memory_manage.knowledge_memory import _resolve

    with pytest.raises(AgentMemoryError, match="非法资源引用"):
        _resolve(tmp_path, "")
    shots = KnowledgeMemory().get_few_shots(IntentType.NEW_QUERY)
    assert [item.role for item in shots] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert "ERROR" in shots[0].content
