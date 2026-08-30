"""MemoryStore 特有行为：可注入时钟的 TTL、并发 push 不丢元素。"""

import threading

import pytest

from agent.store.keys import KEY_ROOT, make_key
from agent.store.memory_store import MemoryStore


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_ttl_expires_lazily_on_read() -> None:
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    store.set("k", "v", ttl=10)

    clock.advance(9)
    assert store.get("k") == "v"

    clock.advance(2)
    assert store.get("k") is None
    assert store.exists("k") is False


def test_expire_then_advance_clock() -> None:
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    store.set("k", "v")
    store.expire("k", 5)

    clock.advance(5)
    assert store.get("k") is None


def test_keys_does_not_list_expired_entries() -> None:
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    store.set("keep", "1")
    store.set("gone", "2", ttl=1)

    clock.advance(2)

    assert store.keys("") == ["keep"]
    assert store.exists("gone") is False


def test_expired_list_reads_as_empty() -> None:
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    store.push("q", "a", "b")
    store.expire("q", 1)
    clock.advance(2)

    assert store.range("q") == []
    assert store.length("q") == 0


def test_expired_key_can_be_reused_as_the_other_kind() -> None:
    clock = FakeClock()
    store = MemoryStore(clock=clock)
    store.set("k", "v", ttl=1)
    clock.advance(2)
    store.push("k", "a")

    assert store.range("k") == ["a"]


def test_concurrent_pushes_are_not_lost() -> None:
    store = MemoryStore()
    n_threads = 8
    n_each = 50
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for index in range(n_each):
            store.push("q", f"{tid}:{index}")

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.length("q") == n_threads * n_each
    assert len(set(store.range("q"))) == n_threads * n_each


def test_concurrent_sets_on_distinct_keys_do_not_corrupt() -> None:
    store = MemoryStore()
    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for index in range(40):
            store.set(f"k:{tid}:{index}", index)
            assert store.get(f"k:{tid}:{index}") == index

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store.keys("k:")) == n_threads * 40


def test_make_key_follows_the_plan_shape() -> None:
    key = make_key("sess_1", "task", "task_abc")

    assert key == f"{KEY_ROOT}:sess_1:task:task_abc"
    assert key.split(":") == [KEY_ROOT, "sess_1", "task", "task_abc"]


def test_make_key_rejects_colons_and_empty_segments() -> None:
    with pytest.raises(ValueError, match="冒号"):
        make_key("sess:1", "task", "a")
    with pytest.raises(ValueError, match="session_id"):
        make_key("", "task", "a")
