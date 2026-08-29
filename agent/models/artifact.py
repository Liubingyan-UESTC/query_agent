"""工具产物：Context Window 三部分中「task_artifacts」的原子结构。

存在的理由是隔离体积：一次 ES 查询可能回来上万行，若直接进 `task_content`，
上下文窗口会被单次结果打满，且每轮对话都要重新投喂一遍。因此约定：

- 全量数据留在 `Artifact.data`（或 `storage_ref` 指向的外部存储）里；
- 进入提示词的只有 `artifact_id` 与 `preview()` 产出的摘要；
- 需要全量时由工具凭 `artifact_id` 回读，而不是让模型「记住」数据。

`preview()` 的输出长度必须有上界，否则上一条约定就形同虚设——所以行数、单元格宽度、
总字符数三个维度都设了硬上限。

`data` 与 `row_count` 是分页语义，二者不保证相等，赋值 `data` 后也不会回写
`row_count`：`data` 是当前页（或截断后的子集），`row_count` 是调用方声明的总数。
步骤 17/18 做 `max_result_rows` 截断时必须显式传入总数，不能按 `len(data)` 理解。
只在构造时未给 `row_count` 且 `data` 为列表的情况下，才用页长补全——那表示「这一页
就是全部」。
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import Field, model_validator

from agent.common.enums import ArtifactType
from agent.common.ids import new_artifact_id
from agent.models.base import AgentModel

__all__ = [
    "DEFAULT_PREVIEW_CELL_CHARS",
    "DEFAULT_PREVIEW_MAX_CHARS",
    "DEFAULT_PREVIEW_ROWS",
    "Artifact",
]

DEFAULT_PREVIEW_ROWS = 5
DEFAULT_PREVIEW_CELL_CHARS = 40
DEFAULT_PREVIEW_MAX_CHARS = 1200

_ELLIPSIS = "…"
_CELL_SEPARATOR = " | "


def _clip(text: str, limit: int) -> str:
    """按字符数截断，超出部分用省略号代替。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + _ELLIPSIS


class Artifact(AgentModel):
    """一份工具产出的结构化数据。

    `data` 需为 JSON 原生类型，否则 `to_dict()` 无法序列化——大体积或非 JSON 数据
    应落到外部存储并只保留 `storage_ref`。
    """

    artifact_id: str = Field(default_factory=new_artifact_id)
    task_id: str
    producer: str
    artifact_type: ArtifactType
    title: str = ""
    # 表格产物的约定：列名 → 列类型，dict 的插入顺序即列顺序
    data_schema: dict[str, Any] = Field(default_factory=dict)
    data: Any = None
    storage_ref: str | None = None
    # 总数，不是 len(data)。分页/截断后二者可以不相等，赋值 data 不会回写本字段
    row_count: int | None = None
    size_bytes: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _derive_row_count(cls, payload: Any) -> Any:
        """`data` 是列表时补全 `row_count`，但不覆盖调用方的显式值。

        分页场景下 `data` 只是一页而 `row_count` 是总数，两者不能互相推导，
        所以只在调用方没给的时候才填。
        """
        if not isinstance(payload, dict) or payload.get("row_count") is not None:
            return payload
        rows = payload.get("data")
        if not isinstance(rows, list):
            return payload
        return {**payload, "row_count": len(rows)}

    @property
    def columns(self) -> list[str]:
        """表格列名。优先取 `data_schema`，否则从首行推断。"""
        if self.data_schema:
            return list(self.data_schema)
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], dict):
            return [str(key) for key in self.data[0]]
        return []

    def preview(
        self,
        max_rows: int = DEFAULT_PREVIEW_ROWS,
        *,
        max_cell_chars: int = DEFAULT_PREVIEW_CELL_CHARS,
        max_chars: int = DEFAULT_PREVIEW_MAX_CHARS,
    ) -> str:
        """给 LLM 看的摘要文本，长度有硬上界。

        `max_chars` 是最后一道闸：即便列数极多或单行极宽，返回值也不会超过它。
        """
        lines = [self._header_line()]
        lines.extend(self._body_lines(max_rows, max_cell_chars))
        return _clip("\n".join(lines), max_chars)

    def _header_line(self) -> str:
        parts = [f"[{self.artifact_type.value}]"]
        if self.title:
            parts.append(self.title)
        parts.append(f"(artifact_id={self.artifact_id})")
        if self.row_count is not None:
            parts.append(f"共 {self.row_count} 行")
        return " ".join(parts)

    def _body_lines(self, max_rows: int, max_cell_chars: int) -> list[str]:
        if self.artifact_type is ArtifactType.TABLE:
            return self._table_lines(max_rows, max_cell_chars)
        if self.artifact_type is ArtifactType.FILE:
            return self._file_lines()
        if self.artifact_type is ArtifactType.CHART and self.data_schema:
            return [f"维度: {', '.join(self.columns)}"]
        if self.data is None:
            return []
        return [f"值: {_clip(str(self.data), max_cell_chars)}"]

    def _table_lines(self, max_rows: int, max_cell_chars: int) -> list[str]:
        rows = self.data if isinstance(self.data, list) else []
        columns = self.columns
        if not columns:
            return ["（空表）"]

        lines = [_CELL_SEPARATOR.join(_clip(column, max_cell_chars) for column in columns)]
        shown = rows[: max(max_rows, 0)]
        for row in shown:
            values = row if isinstance(row, dict) else {}
            lines.append(
                _CELL_SEPARATOR.join(
                    _clip("" if values.get(column) is None else str(values[column]), max_cell_chars)
                    for column in columns
                )
            )

        omitted = len(rows) - len(shown)
        if omitted > 0:
            lines.append(f"（另有 {omitted} 行未显示，凭 artifact_id 可取全量）")
        return lines

    def _file_lines(self) -> list[str]:
        parts = []
        if self.storage_ref:
            parts.append(f"storage_ref={self.storage_ref}")
        if self.size_bytes is not None:
            parts.append(f"size_bytes={self.size_bytes}")
        return [" ".join(parts)] if parts else []
