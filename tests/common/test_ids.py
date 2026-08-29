"""步骤 3 验收：ID 三段式格式、时序可排序、无碰撞。"""

from datetime import UTC, datetime

import pytest

from agent.common.ids import (
    ARTIFACT_ID_PREFIX,
    ID_PATTERN,
    SESSION_ID_PREFIX,
    TASK_ID_PREFIX,
    TRACE_ID_PREFIX,
    new_artifact_id,
    new_id,
    new_session_id,
    new_task_id,
    new_trace_id,
)

GENERATORS = [
    (new_task_id, TASK_ID_PREFIX),
    (new_session_id, SESSION_ID_PREFIX),
    (new_artifact_id, ARTIFACT_ID_PREFIX),
    (new_trace_id, TRACE_ID_PREFIX),
]


@pytest.mark.parametrize(("generator", "prefix"), GENERATORS)
def test_id_has_the_expected_three_part_shape(generator, prefix: str) -> None:
    value = generator()
    match = ID_PATTERN.match(value)

    assert match is not None, f"{value} 不符合 <前缀>_<时间戳>_<随机> 格式"
    assert match.group("prefix") == prefix


@pytest.mark.parametrize(("generator", "prefix"), GENERATORS)
def test_prefixes_are_distinguishable_in_logs(generator, prefix: str) -> None:
    assert generator().startswith(f"{prefix}_")


def test_prefixes_are_mutually_distinct() -> None:
    prefixes = [prefix for _, prefix in GENERATORS]
    assert len(prefixes) == len(set(prefixes))


def test_timestamp_is_current_utc_and_millisecond_precise() -> None:
    before = datetime.now(UTC)
    value = new_task_id()
    after = datetime.now(UTC)

    match = ID_PATTERN.match(value)
    assert match is not None
    stamp = match.group("timestamp")
    parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%S%f").replace(
        # strptime 把末 3 位当作微秒读入，需换算回毫秒
        microsecond=int(stamp[-3:]) * 1000,
        tzinfo=UTC,
    )

    assert before.replace(microsecond=0) <= parsed <= after


def test_ids_sort_chronologically() -> None:
    """字典序即生成时序，使日志与存储键天然按时间排列。"""
    earlier = new_task_id()
    later = new_task_id()

    # 同一毫秒内随机段可能反序，故只比较时间戳段
    assert earlier.split("_")[1] <= later.split("_")[1]


def test_ids_are_unique_under_tight_loop() -> None:
    values = {new_task_id() for _ in range(2000)}

    assert len(values) == 2000


def test_new_id_accepts_a_custom_prefix() -> None:
    value = new_id("op")

    assert value.startswith("op_")
    assert ID_PATTERN.match(value) is not None


def test_random_segment_disambiguates_ids_within_one_millisecond() -> None:
    """时间戳段相同的 ID 必须靠随机段区分，否则毫秒级时间戳不足以保证唯一。"""
    counts: dict[str, list[str]] = {}
    for _ in range(500):
        _, timestamp, random_part = new_task_id().split("_")
        counts.setdefault(timestamp, []).append(random_part)

    crowded = {ts: parts for ts, parts in counts.items() if len(parts) > 1}
    assert crowded, "循环过快以致未出现同毫秒 ID，本用例失去意义"

    for timestamp, parts in crowded.items():
        assert len(set(parts)) == len(parts), f"{timestamp} 毫秒内出现重复随机段"
