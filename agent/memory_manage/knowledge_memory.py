"""外置知识：系统提示词、按意图的 skill、Kibana 字段字典。

资源全部是 Markdown / YAML，改提示词不改代码。构造与 `reload()` 都会做一次
完整校验：缺文件、坏 YAML、skill 引用了不存在的 prompt 或索引，一律抛
`AgentMemoryError`，不静默降级。

`get_system_prompt` 恒把 `base.md` 放在阶段提示词之前。规划阶段用 skill 的
`system_prompt_ref`，因此 `model_for` 式的「写死主模型名」不会发生在这里——
每个意图的规划词条跟自己的文件走。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, field_validator

from agent.common.enums import IntentType, PromptStage
from agent.common.errors import AgentMemoryError
from agent.models.base import AgentModel
from agent.models.message import Message

__all__ = ["FieldEntry", "IndexDef", "KnowledgeMemory", "Skill", "render_indexes"]

BASE_PROMPT = "system_prompts/base.md"
SCHEMA_FILE = "schemas/kibana_fields.yaml"
_STAGE_FILES: dict[PromptStage, str] = {
    PromptStage.INTENT_RECOGNITION: "system_prompts/intent_recognition.md",
    PromptStage.EXECUTE: "system_prompts/execute.md",
    PromptStage.VALIDATE: "system_prompts/validate.md",
}


class FieldEntry(AgentModel):
    """字段字典中的一条字段。YAML 键 `type` 对应 `field_type`，避免遮蔽内置名。"""

    name: str = Field(min_length=1)
    field_type: str = Field(alias="type", min_length=1)
    meaning: str = Field(min_length=1)
    values: list[str] = Field(default_factory=list)
    example: str | None = None


class IndexDef(AgentModel):
    """一个 ES/Kibana 索引的字段集合。"""

    name: str = Field(min_length=1)
    description: str = ""
    fields: list[FieldEntry] = Field(min_length=1)


class Skill(AgentModel):
    """按意图分发的固定知识。字段与 `skills/<intent>.yaml` 一一对应。"""

    name: str = Field(min_length=1)
    description: str
    system_prompt_ref: str = Field(min_length=1)
    allowed_tools: list[str]
    field_dict_refs: list[str]
    few_shots: list[Message]
    output_schema: dict[str, Any]

    @field_validator("system_prompt_ref")
    @classmethod
    def _posix_ref(cls, value: str) -> str:
        return Path(value).as_posix()

    @field_validator("allowed_tools", "field_dict_refs")
    @classmethod
    def _no_blank_and_no_dup(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        for item in value:
            if not item or not item.strip():
                raise ValueError("条目不能为空或纯空白")
            if item in seen:
                raise ValueError(f"重复条目：{item!r}")
            seen.add(item)
        return value


@dataclass(frozen=True, slots=True)
class _Snapshot:
    prompts: dict[str, str]
    skills: dict[IntentType, Skill]
    field_dict_all: str
    field_dict_by_index: dict[str, str]
    indexes: dict[str, IndexDef]


class KnowledgeMemory:
    """从 `root_dir` 加载并校验知识资源。默认指向 `agent/knowledge`。"""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir is not None else _default_root()
        if not self.root_dir.is_dir():
            raise AgentMemoryError(f"知识根目录不存在或不是目录：{self.root_dir}")
        self._snapshot = _load(self.root_dir)

    def get_system_prompt(
        self, stage: PromptStage | str, intent: IntentType | str | None = None
    ) -> str:
        """返回 `base.md` + 阶段提示词。规划阶段必须带 intent。"""
        resolved = _coerce_stage(stage)
        base = self._snapshot.prompts[BASE_PROMPT]
        if resolved is PromptStage.PLAN:
            if intent is None:
                raise AgentMemoryError("规划阶段必须提供 intent，才能选择对应的 system prompt")
            body = self._snapshot.prompts[self.get_skill(intent).system_prompt_ref]
        else:
            body = self._snapshot.prompts[_STAGE_FILES[resolved]]
        return _join_prompts(base, body)

    def get_skill(self, intent: IntentType | str) -> Skill:
        return self._snapshot.skills[_coerce_intent(intent)]

    def get_field_dict(self, index: str | None = None) -> str:
        """渲染为稳定 Markdown。`index=None` 输出全部索引。"""
        if index is None:
            return self._snapshot.field_dict_all
        try:
            return self._snapshot.field_dict_by_index[index]
        except KeyError:
            known = sorted(self._snapshot.field_dict_by_index)
            raise AgentMemoryError(f"字段字典没有索引 {index!r}；已有：{known}") from None

    def get_few_shots(self, intent: IntentType | str) -> list[Message]:
        return list(self.get_skill(intent).few_shots)

    def get_allowed_tools(self, intent: IntentType | str) -> list[str]:
        return list(self.get_skill(intent).allowed_tools)

    def list_indexes(self) -> list[str]:
        return sorted(self._snapshot.indexes)

    def has_index(self, name: str) -> bool:
        return name in self._snapshot.indexes

    def get_index(self, name: str) -> IndexDef:
        try:
            return self._snapshot.indexes[name]
        except KeyError:
            known = self.list_indexes()
            raise AgentMemoryError(f"字段字典没有索引 {name!r}；已有：{known}") from None

    def field_names(self, index: str) -> frozenset[str]:
        return frozenset(field.name for field in self.get_index(index).fields)

    def reload(self) -> None:
        """重新读盘并校验。失败时保留上一份快照，不会半更新。"""
        self._snapshot = _load(self.root_dir)


def _default_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "knowledge").resolve()


def _coerce_intent(intent: IntentType | str) -> IntentType:
    return intent if isinstance(intent, IntentType) else IntentType.from_str(intent)


def _coerce_stage(stage: PromptStage | str) -> PromptStage:
    return stage if isinstance(stage, PromptStage) else PromptStage.from_str(stage)


def _join_prompts(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts) + "\n"


def _relkey(relative: str) -> str:
    return Path(relative).as_posix()


def _resolve(root: Path, relative: str) -> Path:
    """只接受相对 posix 路径。

    不信任 `Path.is_absolute()`：Windows 上无盘符的 `/tmp` 不一定算绝对路径。
    """
    if not relative or relative != relative.strip():
        raise AgentMemoryError(f"非法资源引用：{relative!r}")
    posix = relative.replace("\\", "/")
    if posix.startswith("/") or (len(posix) >= 2 and posix[1] == ":"):
        raise AgentMemoryError(f"资源引用必须是相对路径：{relative!r}")
    path = (root / posix).resolve()
    if not path.is_relative_to(root):
        raise AgentMemoryError(f"资源引用越出知识根目录：{relative!r}")
    return path


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise AgentMemoryError(f"缺少知识资源：{path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise AgentMemoryError(f"知识资源为空：{path}")
    return text


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(_read_text(path))
    except yaml.YAMLError as exc:
        raise AgentMemoryError(f"YAML 无法解析：{path}", detail=str(exc)) from exc
    if not isinstance(payload, dict):
        raise AgentMemoryError(f"YAML 必须是对象：{path}")
    return payload


def _required_prompt_keys() -> list[str]:
    keys = [BASE_PROMPT, *_STAGE_FILES.values()]
    keys.extend(f"system_prompts/plan_{intent.value}.md" for intent in IntentType)
    return keys


def _load_catalog(root: Path) -> list[IndexDef]:
    path = _resolve(root, SCHEMA_FILE)
    payload = _load_yaml(path)
    raw_indexes = payload.get("indexes")
    if not isinstance(raw_indexes, list) or not raw_indexes:
        raise AgentMemoryError(f"字段字典 indexes 必须是非空列表：{path}")
    catalog: list[IndexDef] = []
    for item in raw_indexes:
        if not isinstance(item, dict):
            raise AgentMemoryError(f"字段字典 indexes 的元素必须是对象：{path}")
        try:
            catalog.append(IndexDef.model_validate(item))
        except ValidationError as exc:
            raise AgentMemoryError(f"字段字典结构不合法：{path}", detail=str(exc)) from exc
    names = [index.name for index in catalog]
    if len(names) != len(set(names)):
        raise AgentMemoryError(f"字段字典存在重复索引名：{path}")
    for index in catalog:
        field_names = [field.name for field in index.fields]
        if len(field_names) != len(set(field_names)):
            raise AgentMemoryError(f"索引 {index.name} 存在重复字段名：{path}")
    return catalog


def render_indexes(indexes: list[IndexDef]) -> str:
    """把字段字典渲染为稳定 Markdown，供 `get_field_dict` 与单测共用。"""
    lines: list[str] = ["# 字段字典", ""]
    for index in indexes:
        lines.append(f"## {index.name}")
        if index.description:
            lines.append(index.description)
        lines.append("")
        lines.append("| 字段 | 类型 | 含义 | 可选值 | 示例 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for field in index.fields:
            values = ", ".join(field.values)
            example = field.example or ""
            lines.append(
                f"| {field.name} | {field.field_type} | {field.meaning} | {values} | {example} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_skill(root: Path, intent: IntentType, index_names: set[str]) -> Skill:
    path = root / "skills" / f"{intent.value}.yaml"
    payload = _load_yaml(path)
    try:
        skill = Skill.model_validate(payload)
    except ValidationError as exc:
        raise AgentMemoryError(f"skill {path.name} 结构不合法", detail=str(exc)) from exc
    _resolve(root, skill.system_prompt_ref)
    for ref in skill.field_dict_refs:
        if ref not in index_names:
            raise AgentMemoryError(
                f"skill {intent.value} 引用了不存在的字段字典 {ref!r}；已有：{sorted(index_names)}"
            )
    return skill


def _load(root: Path) -> _Snapshot:
    catalog = _load_catalog(root)
    index_names = {index.name for index in catalog}
    prompts: dict[str, str] = {}
    for relative in _required_prompt_keys():
        prompts[_relkey(relative)] = _read_text(_resolve(root, relative))
    skills: dict[IntentType, Skill] = {}
    for intent in IntentType:
        skill = _load_skill(root, intent, index_names)
        key = _relkey(skill.system_prompt_ref)
        if key not in prompts:
            prompts[key] = _read_text(_resolve(root, skill.system_prompt_ref))
        skills[intent] = skill
    by_index = {index.name: render_indexes([index]) for index in catalog}
    return _Snapshot(
        prompts=prompts,
        skills=skills,
        field_dict_all=render_indexes(catalog),
        field_dict_by_index=by_index,
        indexes={index.name: index for index in catalog},
    )
