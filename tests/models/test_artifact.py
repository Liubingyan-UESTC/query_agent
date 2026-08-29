"""步骤 5 验收：`preview()` 在超大结果集下长度可控、JSON 序列化往返一致。"""

import json

import pytest
from pydantic import ValidationError

from agent.common.enums import ArtifactType
from agent.models.artifact import (
    DEFAULT_PREVIEW_CELL_CHARS,
    DEFAULT_PREVIEW_MAX_CHARS,
    DEFAULT_PREVIEW_ROWS,
    Artifact,
)


def make_table(rows: int, *, columns: int = 3) -> Artifact:
    schema = {f"col{index}": "keyword" for index in range(columns)}
    data = [{f"col{index}": f"r{row}c{index}" for index in range(columns)} for row in range(rows)]
    return Artifact(
        task_id="task_1",
        producer="kibana_query",
        artifact_type=ArtifactType.TABLE,
        title="查询结果",
        data_schema=schema,
        data=data,
    )


# ============================================================ preview 长度上界


@pytest.mark.parametrize("rows", [0, 1, 5, 100, 10_000, 200_000])
def test_preview_length_is_bounded_regardless_of_row_count(rows: int) -> None:
    """验收核心：结果集再大，摘要长度都不超过上限。"""
    preview = make_table(rows).preview()

    assert len(preview) <= DEFAULT_PREVIEW_MAX_CHARS


def test_preview_is_bounded_when_columns_are_absurdly_many() -> None:
    """行数不是唯一的膨胀维度：列数与单元格宽度同样要被截住。"""
    preview = make_table(50, columns=500).preview()

    assert len(preview) <= DEFAULT_PREVIEW_MAX_CHARS


def test_preview_is_bounded_when_a_single_cell_is_huge() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="kibana_query",
        artifact_type=ArtifactType.TABLE,
        data_schema={"payload": "text"},
        data=[{"payload": "x" * 1_000_000}],
    )

    preview = artifact.preview()

    assert len(preview) <= DEFAULT_PREVIEW_MAX_CHARS


def test_each_cell_respects_the_cell_limit() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="t",
        artifact_type=ArtifactType.TABLE,
        data_schema={"a": "text", "b": "text"},
        data=[{"a": "y" * 500, "b": "z" * 500}],
    )

    body = artifact.preview(max_cell_chars=10).splitlines()[-1]

    for cell in body.split(" | "):
        assert len(cell) <= 10


def test_preview_shows_at_most_max_rows() -> None:
    preview = make_table(1000).preview(max_rows=3)

    data_lines = [line for line in preview.splitlines() if line.startswith("r")]
    assert len(data_lines) == 3


def test_preview_reports_how_many_rows_were_omitted() -> None:
    """模型必须知道自己只看到了片段，否则会把预览当成全量来下结论。"""
    preview = make_table(1000).preview(max_rows=5)

    assert "995" in preview
    assert "artifact_id" in preview


def test_preview_of_small_table_has_no_omission_notice() -> None:
    preview = make_table(2).preview(max_rows=5)

    assert "未显示" not in preview


# ============================================================ preview 内容


def test_table_preview_contains_header_rows_and_total() -> None:
    """验收要求：表头 + 前 N 行 + 行数统计。"""
    preview = make_table(42).preview(max_rows=2)

    assert "col0 | col1 | col2" in preview
    assert "r0c0" in preview
    assert "r1c0" in preview
    assert "共 42 行" in preview


def test_preview_identifies_the_artifact_for_follow_up_reads() -> None:
    artifact = make_table(10)

    assert artifact.artifact_id in artifact.preview()


def test_preview_states_the_artifact_type() -> None:
    assert make_table(1).preview().startswith("[table]")


def test_missing_cells_render_as_empty_not_none() -> None:
    """把缺失渲染成 "None" 会让模型误以为存在字面值 None。"""
    artifact = Artifact(
        task_id="task_1",
        producer="t",
        artifact_type=ArtifactType.TABLE,
        data_schema={"a": "text", "b": "text"},
        data=[{"a": "v"}],
    )

    body = artifact.preview().splitlines()[-1]

    assert "None" not in body
    assert body.startswith("v")


def test_columns_fall_back_to_the_first_row() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="t",
        artifact_type=ArtifactType.TABLE,
        data=[{"host": "web-01", "cpu": 0.9}],
    )

    assert artifact.columns == ["host", "cpu"]
    assert "host | cpu" in artifact.preview()


def test_schema_order_defines_column_order() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="t",
        artifact_type=ArtifactType.TABLE,
        data_schema={"z": "text", "a": "text"},
        data=[{"a": "1", "z": "2"}],
    )

    assert artifact.columns == ["z", "a"]
    assert "z | a" in artifact.preview()


def test_empty_table_is_stated_explicitly() -> None:
    artifact = Artifact(task_id="task_1", producer="t", artifact_type=ArtifactType.TABLE, data=[])

    preview = artifact.preview()

    assert "共 0 行" in preview
    assert "（空表）" in preview


@pytest.mark.parametrize(
    ("artifact_type", "data", "expected"),
    [
        (ArtifactType.SCALAR, 12345, "12345"),
        (ArtifactType.TEXT, "结论文本", "结论文本"),
    ],
)
def test_non_table_previews_show_the_value(
    artifact_type: ArtifactType, data: object, expected: str
) -> None:
    artifact = Artifact(task_id="task_1", producer="t", artifact_type=artifact_type, data=data)

    assert expected in artifact.preview()


def test_file_preview_points_at_the_storage_reference() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="export_tool",
        artifact_type=ArtifactType.FILE,
        title="导出结果",
        storage_ref="/exports/task_1.xlsx",
        size_bytes=204_800,
    )

    preview = artifact.preview()

    assert "/exports/task_1.xlsx" in preview
    assert "204800" in preview


def test_chart_preview_lists_its_dimensions() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="analysis",
        artifact_type=ArtifactType.CHART,
        title="错误数趋势",
        data_schema={"ts": "date", "count": "long"},
        data={"series": [1, 2, 3]},
    )

    preview = artifact.preview()

    assert "维度: ts, count" in preview
    assert "series" not in preview, "图表数据本身不该进预览"


def test_chart_without_schema_falls_back_to_the_value_line() -> None:
    artifact = Artifact(
        task_id="task_1",
        producer="analysis",
        artifact_type=ArtifactType.CHART,
        data={"series": [1]},
    )

    assert "series" in artifact.preview()


def test_file_preview_omits_the_body_line_when_nothing_is_known() -> None:
    """既无 storage_ref 也无 size_bytes 时，不该输出一行空白。"""
    artifact = Artifact(task_id="task_1", producer="export_tool", artifact_type=ArtifactType.FILE)

    preview = artifact.preview()

    assert preview.splitlines() == [preview]
    assert preview.startswith("[file]")
    assert not preview.endswith("\n")


@pytest.mark.parametrize("artifact_type", [ArtifactType.SCALAR, ArtifactType.TEXT])
def test_preview_of_empty_data_shows_header_only(artifact_type: ArtifactType) -> None:
    """把空产物渲染成「值: None」会让模型误以为存在字面值 None。"""
    artifact = Artifact(
        task_id="task_1", producer="t", artifact_type=artifact_type, data=None, title="无结果"
    )

    preview = artifact.preview()

    assert preview == f"[{artifact_type.value}] 无结果 (artifact_id={artifact.artifact_id})"


def test_zero_cell_width_yields_empty_cells_rather_than_crashing() -> None:
    """max_cell_chars=0 是调用方可能传进来的边界值，必须有确定行为。"""
    preview = make_table(3).preview(max_cell_chars=0)

    body = preview.splitlines()[-1]
    assert set(body) <= {"|", " "}


def test_preview_never_embeds_bulk_data_for_file_artifacts() -> None:
    """文件产物的正文在外部存储，预览里出现全量内容就说明约定被破坏了。"""
    artifact = Artifact(
        task_id="task_1",
        producer="export_tool",
        artifact_type=ArtifactType.FILE,
        storage_ref="/exports/a.xlsx",
        data=None,
    )

    assert "None" not in artifact.preview()


# ============================================================ row_count 推导


def test_row_count_is_derived_from_list_data() -> None:
    assert make_table(37).row_count == 37


def test_explicit_row_count_survives_paging() -> None:
    """`data` 只是一页时，调用方给的总数不能被页长覆盖。"""
    artifact = Artifact(
        task_id="task_1",
        producer="t",
        artifact_type=ArtifactType.TABLE,
        data=[{"a": 1}],
        row_count=9999,
    )

    assert artifact.row_count == 9999
    assert "共 9999 行" in artifact.preview()


def test_row_count_stays_none_for_scalar_data() -> None:
    artifact = Artifact(task_id="task_1", producer="t", artifact_type=ArtifactType.SCALAR, data=7)

    assert artifact.row_count is None
    assert "共" not in artifact.preview()


# ============================================================ 序列化往返


def test_json_round_trip_is_lossless() -> None:
    original = make_table(3)

    restored = Artifact.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original


def test_to_dict_is_directly_json_serializable() -> None:
    payload = make_table(2).to_dict()

    assert isinstance(payload["created_at"], str)
    assert isinstance(payload["artifact_type"], str)
    assert json.dumps(payload)


def test_round_trip_preserves_derived_row_count() -> None:
    restored = Artifact.from_dict(make_table(11).to_dict())

    assert restored.row_count == 11


# ============================================================ 校验


def test_task_id_and_producer_are_required() -> None:
    """产物必须可溯源到任务与工具，否则归档后无法判断它是谁产生的。"""
    with pytest.raises(ValidationError):
        Artifact(artifact_type=ArtifactType.TABLE)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Artifact(
            task_id="task_1",
            producer="t",
            artifact_type=ArtifactType.TABLE,
            schema={"a": "text"},
        )


def test_invalid_artifact_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Artifact(task_id="task_1", producer="t", artifact_type="DataFrame")


def test_each_artifact_gets_a_unique_prefixed_id() -> None:
    ids = {make_table(1).artifact_id for _ in range(100)}

    assert len(ids) == 100
    assert all(value.startswith("art_") for value in ids)


def test_preview_defaults_are_sane() -> None:
    assert DEFAULT_PREVIEW_ROWS > 0
    assert 0 < DEFAULT_PREVIEW_CELL_CHARS < DEFAULT_PREVIEW_MAX_CHARS
