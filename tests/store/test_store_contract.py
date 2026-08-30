"""步骤 8 验收：任何 Store 实现都必须通过的契约。

本文件不绑定 MemoryStore：后端来自 `STORE_FACTORIES`。步骤 27 的 Redis
实现加入工厂列表后，这些用例自动跑第二遍。
"""

import pytest

from agent.common.errors import StoreError
from agent.store.base import Store
from agent.store.keys import make_key

pytestmark = pytest.mark.contract


# ============================================================ KV


def test_missing_key_returns_none(store: Store) -> None:
    assert store.get("absent") is None
    assert store.exists("absent") is False
    assert store.delete("absent") is False


def test_set_get_round_trip(store: Store) -> None:
    store.set("k", {"n": 1})

    assert store.get("k") == {"n": 1}
    assert store.exists("k") is True


def test_get_returns_a_copy(store: Store) -> None:
    """交出内部引用会让调用方就地修改污染存储。"""
    store.set("k", {"n": 1})

    got = store.get("k")
    assert isinstance(got, dict)
    got["n"] = 99

    assert store.get("k") == {"n": 1}


def test_set_overwrites(store: Store) -> None:
    store.set("k", "old")
    store.set("k", "new")

    assert store.get("k") == "new"


def test_delete_removes_the_key(store: Store) -> None:
    store.set("k", "v")

    assert store.delete("k") is True
    assert store.get("k") is None
    assert store.delete("k") is False


def test_keys_filters_by_prefix(store: Store) -> None:
    session = "sess_1"
    store.set(make_key(session, "task", "a"), "1")
    store.set(make_key(session, "task", "b"), "2")
    store.set(make_key("sess_10", "task", "c"), "3")

    found = set(store.keys(f"agent:{session}:"))

    assert found == {
        make_key(session, "task", "a"),
        make_key(session, "task", "b"),
    }
    assert make_key("sess_10", "task", "c") not in found


def test_expire_zero_deletes_immediately(store: Store) -> None:
    store.set("k", "v")

    assert store.expire("k", 0) is True
    assert store.get("k") is None


def test_set_with_non_positive_ttl_does_not_store(store: Store) -> None:
    store.set("k", "v", ttl=0)

    assert store.get("k") is None
    assert store.exists("k") is False


def test_expire_missing_key_returns_false(store: Store) -> None:
    assert store.expire("absent", 10) is False


def test_set_clears_previous_ttl_when_omitted(store: Store) -> None:
    store.set("k", "v", ttl=0.001)
    store.set("k", "kept")

    assert store.get("k") == "kept"


# ============================================================ List


def test_push_requires_at_least_one_value(store: Store) -> None:
    with pytest.raises(ValueError, match="push"):
        store.push("q")


def test_push_appends_and_returns_length(store: Store) -> None:
    assert store.push("q", "a") == 1
    assert store.push("q", "b", "c") == 3
    assert store.length("q") == 3
    assert store.range("q") == ["a", "b", "c"]


def test_range_is_closed_and_supports_negative_index(store: Store) -> None:
    store.push("q", "a", "b", "c", "d")

    assert store.range("q", 0, 1) == ["a", "b"]
    assert store.range("q", -2, -1) == ["c", "d"]
    assert store.range("q", 10, 20) == []
    assert store.range("q", 2, 1) == []


def test_range_returns_a_copy(store: Store) -> None:
    store.push("q", {"n": 1})

    got = store.range("q")
    got[0]["n"] = 99

    assert store.range("q") == [{"n": 1}]


def test_trim_keeps_the_closed_window(store: Store) -> None:
    store.push("q", "a", "b", "c", "d")
    store.trim("q", 1, -2)

    assert store.range("q") == ["b", "c"]


def test_trim_empty_window_clears_the_list(store: Store) -> None:
    store.push("q", "a", "b")
    store.trim("q", 2, 1)

    assert store.range("q") == []
    assert store.length("q") == 0


def test_missing_list_reads_as_empty(store: Store) -> None:
    assert store.range("absent") == []
    assert store.length("absent") == 0
    store.trim("absent", 0, -1)


def test_push_copies_on_write(store: Store) -> None:
    payload = {"n": 1}
    store.push("q", payload)
    payload["n"] = 99

    assert store.range("q") == [{"n": 1}]


# ============================================================ 类型隔离


def test_kv_and_list_do_not_share_a_key(store: Store) -> None:
    store.set("k", "v")

    with pytest.raises(StoreError):
        store.push("k", "x")

    store.delete("k")
    store.push("k", "x")

    with pytest.raises(StoreError):
        store.get("k")
