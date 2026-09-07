"""并发行为测试：会话之间并发、会话之内串行。

这些用例刻意**不碰数据库**（manager 不挂 DbListener）：pytest-django 把每个用例包在一个
事务里，而新线程会各建一条连接、看不到那个未提交的事务，测出来的失败与并发无关。
DbListener 的落库行为由 test_api 覆盖。

判定并发用两种手段，都不依赖 sleep 的时长巧合：

- 证明"真的并发"用 :class:`threading.Barrier`——N 个线程必须同时到齐才能通过，
  串行执行下必然超时；
- 证明"真的串行"用一个在途计数器，一旦发现重叠就记下来，断言从未重叠。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.knowledge import KnowledgeMemory
from agent.llm import ConcurrencyGate, GatedLLMClient, MockLLMClient
from agent.llm.base import BaseLLMClient, LLMRequest, LLMResponse
from agent.memory import MemoryManager
from agent.task_manager import build_task_manager
from agent.tools.search_tool import SearchArgs, SearchTool
from server.api import runtime
from server.api.runtime import _SessionLocks

# ================================================================ 测试替身


class _OverlapTrackingLLM(BaseLLMClient):
    """记录"同时有几个 chat 在飞"的模型替身。

    ``peak`` 是观察到的最大并发数：串行执行下恒为 1，并发下 > 1。
    """

    def __init__(self, *, delay: float = 0.03) -> None:
        self._inner = MockLLMClient()
        self._delay = delay
        self._lock = threading.Lock()
        self._inflight = 0
        self.peak = 0

    def chat(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self._inflight += 1
            self.peak = max(self.peak, self._inflight)
        try:
            # 拉长每次调用，让"本该重叠"的情况一定重叠——否则线程可能凑巧错开，
            # 断言就成了运气
            time.sleep(self._delay)
            return self._inner.chat(request)
        finally:
            with self._lock:
                self._inflight -= 1


class _BarrierLLM(BaseLLMClient):
    """每个线程首次调用时在栅栏上会合，用来**确定性地**证明并发。

    ``parties`` 个线程必须同时到齐才能继续。如果实现是串行的，第一个线程会一直等到
    超时（``BrokenBarrierError``），``broke`` 被置真。
    """

    def __init__(self, parties: int, *, timeout: float = 5.0) -> None:
        self._inner = MockLLMClient()
        self._barrier = threading.Barrier(parties)
        self._timeout = timeout
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self.broke = False

    def chat(self, request: LLMRequest) -> LLMResponse:
        thread_id = threading.get_ident()
        with self._lock:
            first_call = thread_id not in self._seen
            self._seen.add(thread_id)
        if first_call:
            try:
                self._barrier.wait(timeout=self._timeout)
            except threading.BrokenBarrierError:
                self.broke = True
        return self._inner.chat(request)


@pytest.fixture
def concurrent_manager(settings):
    """一个不挂任何监听器的 manager 工厂，装进进程单例后自动清理。"""

    def build(llm: BaseLLMClient):
        manager = build_task_manager(settings, llm=llm, sleeper=lambda _s: None)
        runtime.set_manager(manager)
        return manager

    yield build
    runtime.reset_manager()


def _run_all(session_ids: list[str], text: str = "查一下 order-service 的错误日志") -> list:
    """并发跑多轮对话，任何线程抛异常都会在 result() 处重新抛出。"""
    with ThreadPoolExecutor(max_workers=len(session_ids)) as pool:
        futures = [pool.submit(runtime.run_turn, sid, text) for sid in session_ids]
        return [future.result(timeout=30) for future in futures]


# ================================================================ run_turn 的并发语义


class TestSessionsRunConcurrently:
    def test_four_sessions_reach_the_barrier_together(self, concurrent_manager):
        """四个会话必须能同时在飞——这是本次改造的核心目标。"""
        llm = _BarrierLLM(parties=4)
        concurrent_manager(llm)

        tasks = _run_all([f"s-{index}" for index in range(4)])

        assert llm.broke is False, "四个会话没能同时到齐栅栏，说明仍在串行执行"
        assert len({task.task_id for task in tasks}) == 4

    def test_peak_concurrency_exceeds_one(self, concurrent_manager):
        llm = _OverlapTrackingLLM()
        concurrent_manager(llm)

        _run_all([f"s-{index}" for index in range(4)])

        assert llm.peak > 1

    def test_each_session_keeps_its_own_task(self, concurrent_manager):
        """并发下会话不能串味：每个任务的 session_id 必须是自己的。"""
        concurrent_manager(_OverlapTrackingLLM(delay=0.01))
        session_ids = [f"s-{index}" for index in range(6)]

        tasks = _run_all(session_ids)

        assert sorted(task.session_id for task in tasks) == sorted(session_ids)


class TestOneSessionIsSerialized:
    def test_same_session_never_overlaps(self, concurrent_manager):
        """同一会话的多个请求必须串行：追问依赖上一轮结果，并发会让上下文错乱。"""
        llm = _OverlapTrackingLLM()
        concurrent_manager(llm)

        _run_all(["same-session"] * 4)

        assert llm.peak == 1

    def test_same_session_requests_all_complete(self, concurrent_manager):
        """串行不等于丢请求——四轮都得跑出各自的任务。"""
        concurrent_manager(_OverlapTrackingLLM(delay=0.01))

        tasks = _run_all(["same-session"] * 4)

        assert len({task.task_id for task in tasks}) == 4

    def test_one_slow_session_does_not_block_another(self, concurrent_manager):
        """一个会话排队时，别的会话照样能过——栅栏证明两者同时在飞。"""
        llm = _BarrierLLM(parties=2)
        concurrent_manager(llm)

        _run_all(["busy", "other"])

        assert llm.broke is False


# ================================================================ _SessionLocks


class TestSessionLocks:
    def test_same_key_is_mutually_exclusive(self):
        locks = _SessionLocks()
        overlap = []
        inflight = 0
        guard = threading.Lock()

        def work():
            nonlocal inflight
            with locks.acquire("k", max_waiters=8):
                with guard:
                    inflight += 1
                    overlap.append(inflight)
                time.sleep(0.02)
                with guard:
                    inflight -= 1

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: work(), range(4)))

        assert max(overlap) == 1

    def test_different_keys_do_not_block(self):
        """不同 key 必须能同时进入，否则会话锁退化成全局锁。"""
        locks = _SessionLocks()
        barrier = threading.Barrier(3)
        broke = []

        def work(key: str):
            with locks.acquire(key, max_waiters=8):
                try:
                    barrier.wait(timeout=5)
                except threading.BrokenBarrierError:
                    broke.append(key)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(work, ["a", "b", "c"]))

        assert broke == []

    def test_locks_are_reclaimed_after_release(self):
        """锁表必须回收：会话 id 由客户端随手生成，只增不减就是内存泄漏。"""
        locks = _SessionLocks()
        for index in range(50):
            with locks.acquire(f"s-{index}", max_waiters=8):
                pass
        assert locks.active() == 0

    def test_lock_survives_a_waiting_thread(self):
        """有线程在等时不能把锁删掉，否则两边会各持一把不同的锁。"""
        locks = _SessionLocks()
        entered = threading.Event()
        release = threading.Event()
        observed = []

        def holder():
            with locks.acquire("k", max_waiters=8):
                entered.set()
                release.wait(timeout=5)

        def waiter():
            entered.wait(timeout=5)
            observed.append(locks.active())  # 持有者 + 等待者共用一个表项
            with locks.acquire("k", max_waiters=8):
                pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(holder)
            second = pool.submit(waiter)
            entered.wait(timeout=5)
            release.set()
            first.result(timeout=5)
            second.result(timeout=5)

        assert observed == [1]
        assert locks.active() == 0

    def test_lock_is_released_when_body_raises(self):
        locks = _SessionLocks()
        with pytest.raises(RuntimeError), locks.acquire("k", max_waiters=8):
            raise RuntimeError("boom")
        assert locks.active() == 0
        # 还能再拿到，说明锁没被泄漏在持有态
        with locks.acquire("k", max_waiters=8):
            pass


# ================================================================ 内核共享状态


class TestTaskRegistryIsThreadSafe:
    def test_pending_clarification_tolerates_concurrent_creation(self, settings):
        """并发建任务时查澄清态不能炸。

        改造前 ``pending_clarification`` 直接迭代共享的 ``_tasks``，另一个会话插入新任务
        就会撞上 "dictionary changed size during iteration"。
        """
        manager = build_task_manager(settings, llm=MockLLMClient(), sleeper=lambda _s: None)
        stop = threading.Event()
        errors: list[BaseException] = []

        def creator():
            index = 0
            while not stop.is_set():
                manager.create_task(f"任务 {index}", session_id=f"s-{index % 5}")
                index += 1

        def reader():
            try:
                while not stop.is_set():
                    for index in range(5):
                        manager.pending_clarification(f"s-{index}")
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(creator) for _ in range(2)]
            futures += [pool.submit(reader) for _ in range(2)]
            time.sleep(0.3)
            stop.set()
            for future in futures:
                future.result(timeout=10)

        assert errors == []

    def test_concurrent_creation_keeps_every_task_addressable(self, settings):
        """槽是一次原子插入：任务查得到，就一定连黑板和计数器一起查得到。"""
        manager = build_task_manager(settings, llm=MockLLMClient(), sleeper=lambda _s: None)

        with ThreadPoolExecutor(max_workers=8) as pool:
            tasks = list(
                pool.map(
                    lambda index: manager.create_task(f"任务 {index}", session_id=f"s-{index}"),
                    range(64),
                )
            )

        assert len({task.task_id for task in tasks}) == 64
        for task in tasks:
            assert manager.task(task.task_id) is task
            assert manager.window(task.task_id).task_id == task.task_id


class TestMemoryManagerIsThreadSafe:
    def test_working_returns_one_bucket_per_session(self):
        """同一会话并发首访必须拿到同一个桶，否则先归档的记忆会丢。"""
        memory = MemoryManager(KnowledgeMemory())

        with ThreadPoolExecutor(max_workers=16) as pool:
            buckets = list(pool.map(lambda _: memory.working("s"), range(64)))

        assert len({id(bucket) for bucket in buckets}) == 1

    def test_sessions_listing_tolerates_concurrent_creation(self):
        memory = MemoryManager(KnowledgeMemory())
        stop = threading.Event()
        errors: list[BaseException] = []

        def creator():
            index = 0
            while not stop.is_set():
                memory.working(f"s-{index}")
                index += 1

        def lister():
            try:
                while not stop.is_set():
                    memory.sessions()
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(creator), pool.submit(lister)]
            time.sleep(0.2)
            stop.set()
            for future in futures:
                future.result(timeout=10)

        assert errors == []


# ================================================================ 并发闸门


class TestConcurrencyGate:
    def test_in_flight_calls_are_capped(self):
        gate = ConcurrencyGate(3)
        client = GatedLLMClient(_OverlapTrackingLLM(delay=0.02), gate)
        request = LLMRequest(messages=[], stage="intent")

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(lambda _: client.chat(request), range(24)))

        assert client._inner.peak <= 3  # type: ignore[attr-defined]

    def test_gate_allows_full_capacity(self):
        """闸门不能过度收紧：limit 个线程必须能同时在飞。"""
        gate = ConcurrencyGate(4)
        inner = _BarrierLLM(parties=4)
        client = GatedLLMClient(inner, gate)
        request = LLMRequest(messages=[], stage="intent")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: client.chat(request), range(4)))

        assert inner.broke is False

    def test_permit_is_released_when_call_raises(self):
        class _Boom(BaseLLMClient):
            def chat(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("boom")

        gate = ConcurrencyGate(1)
        client = GatedLLMClient(_Boom(), gate)
        request = LLMRequest(messages=[], stage="intent")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                client.chat(request)
        # 许可没漏的话，这里不会挂住
        assert gate.limit == 1

    def test_limit_must_be_positive(self):
        with pytest.raises(ValueError, match="并发上限"):
            ConcurrencyGate(0)

    def test_chain_shares_one_gate(self, settings):
        """降级链上每个端点各包一层，但共享一个 gate——否则上限变成端点数 × N。"""
        from agent.config import LLMProfile, load_settings
        from agent.llm import build_llm_client

        configured = load_settings(
            env_file=None,
            llm={
                "primary": LLMProfile(api_key="k", name="p"),
                "fallbacks": [LLMProfile(api_key="k", name="f")],
                "max_concurrency": 2,
            },
        )
        client = build_llm_client(
            configured, client_factory=lambda _profile: _OverlapTrackingLLM(delay=0)
        )

        gates = {id(inner._gate) for inner in client._clients}  # type: ignore[attr-defined]
        assert len(gates) == 1


# ================================================================ 工具


class TestSearchToolLazyLoad:
    def test_concurrent_first_search_is_consistent(self, settings):
        """并发首次检索：每个线程都要拿到完整结果，不能看到半加载的索引。"""
        tool = SearchTool(settings.tool)
        ctx = None

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(
                pool.map(lambda _: tool.run(SearchArgs(keyword="ERROR"), ctx), range(32))
            )

        totals = {result.data["total"] for result in results}
        assert totals == {3}
