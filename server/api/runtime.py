"""进程内的 Agent 单例，以及"会话之间并发、会话之内串行"的并发闸门。

一个进程一个 :class:`~agent.task_manager.TaskManager`：``MemoryManager`` 本来就按
session_id 分桶，所以一个实例服务所有会话，追问「刚才那批」才能找到关联任务。

**并发模型**：并发的天然单位是会话。

- **不同会话可以同时跑**。它们的黑板、工作记忆、任务表互不相干，没有理由互等。
- **同一会话内串行**。同会话的第二个请求会阻塞等前一个跑完，因为追问依赖上一轮的
  结果：澄清态、related_task_ids、工作记忆都是顺序语义，并发推进只会让上下文错乱。

所以这里用 :class:`_SessionLocks`——一把锁一个会话，而不是一把全局锁。之前那把全局锁
不只是让请求排队，更糟的是**慢操作也在锁内**：一次 LLM 调用可以占 60s，任务级退避还要
再 sleep 若干秒，期间所有会话全被冻住。

**流式响应**：``stream_turn`` 用后台线程跑任务，主线程从有界队列拉事件、转 SSE 推给客户
端。客户端不需要等整轮跑完才能看到进度——意图/规划/工具调用/校验的转移一发生就推。
GIL 不影响，因为任务大部分时间在网络 I/O 上等 LLM 响应，Python 字节码执行只占很少。

**为什么不上多 worker**：``TaskManager`` 是 per-process 单例，任务表与黑板都在进程内存
里。跑 ``gunicorn -w 4`` 会得到 4 个互不相识的 manager——用户第一轮进 worker 1 建了澄清
任务，第二轮被路由到 worker 3 就找不到它，上下文凭空消失。这比性能问题更糟，是正确性
问题。要横向扩容，得先把任务状态搬去 Redis 之类的外部存储。

所以部署方式是**单进程多线程**：``gunicorn -w 1 --threads 16 server.wsgi``。瓶颈本来就
是 LLM 的网络等待，GIL 在这里几乎不影响吞吐。

**重启不恢复**：任务与黑板只活在内存里。重启后 SQLite 里的历史照样能查，但"正等着澄清"
的任务会丢，用户重新提问即可。
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from agent.config import AppSettings, get_settings
from agent.errors import AgentError
from agent.events import TaskEvent
from agent.logging_setup import LoggingListener, get_logger, setup_logging
from agent.models import Task
from agent.task_manager import TaskManager, build_task_manager
from server.api.listener import DbListener

__all__ = [
    "QueueFullError",
    "active_sessions",
    "get_manager",
    "reset_manager",
    "run_turn",
    "set_manager",
    "stream_turn",
]

logger = get_logger("http")


class QueueFullError(AgentError):
    """同一会话排队超过上限，请求被拒。

    HTTP 层翻译成 ``503 Service Unavailable`` + ``Retry-After`` 头，让客户端知道
    应当稍后重试，而不是无脑等。
    """

    code = "queue_full"
    retryable = False

    def __init__(self, session_id: str, queue_size: int) -> None:
        super().__init__(f"会话 {session_id} 排队已满（{queue_size}）")
        self.session_id = session_id
        self.queue_size = queue_size


class _SessionLocks:
    """按 key 发锁，并在最后一个持有者离开后把锁回收。

    直接用 ``dict[str, Lock]`` 而不回收是不行的：会话是客户端随手生成的，长期运行下
    这个字典只增不减，等于一条稳定的内存泄漏。所以给每把锁记一个等待计数，归零即删。

    ``_guard`` 只保护"取锁/还锁"这两下字典操作，绝不在持有 ``_guard`` 时去 acquire
    会话锁——那会让一个慢会话通过 ``_guard`` 把所有会话又串起来，正是要消除的问题。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.Lock, int]] = {}

    @contextmanager
    def acquire(self, key: str, *, max_waiters: int) -> Iterator[None]:
        """抢锁；排队超 ``max_waiters`` 直接抛 :class:`QueueFullError`。

        ``max_waiters`` 是**含持有者**的总并发数——同一个会话允许的并行请求数。
        超了立即拒绝（不排队），否则慢会话下锁表会无限堆积。
        """
        with self._guard:
            entry = self._locks.get(key)
            lock, waiters = entry if entry is not None else (threading.Lock(), 0)
            if waiters >= max_waiters:
                raise QueueFullError(key, waiters)
            # 先把自己登记进等待数，再出去抢锁。否则"抢到锁"与"登记"之间有窗口，
            # 另一个线程可能因为计数为 0 而把这把锁删掉，两边就各持一把不同的锁了。
            self._locks[key] = (lock, waiters + 1)

        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._guard:
                held, waiters = self._locks[key]
                if waiters <= 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (held, waiters - 1)

    def active(self) -> int:
        """当前有多少会话正被持有或等待。健康检查用。"""
        with self._guard:
            return len(self._locks)


_session_locks = _SessionLocks()

# 单例的构造也要串行：不加锁时两个并发的首个请求会各建一个 TaskManager，其中一个连着
# 它的黑板一起被覆盖丢弃。这把锁只在进程生命周期里用一次。
_manager_lock = threading.Lock()
_manager: TaskManager | None = None


def get_manager() -> TaskManager:
    """取进程内单例，首次调用时装配。"""
    global _manager
    manager = _manager
    if manager is not None:
        return manager
    with _manager_lock:
        if _manager is None:
            settings: AppSettings = get_settings()
            setup_logging(settings)
            _manager = build_task_manager(
                settings,
                listeners=[LoggingListener(), DbListener(settings)],
            )
        return _manager


def set_manager(manager: TaskManager | None) -> None:
    """替换单例。测试用这个注入 MockLLM 版本，避免真的去调模型。"""
    global _manager
    with _manager_lock:
        _manager = manager


def reset_manager() -> None:
    set_manager(None)


def active_sessions() -> int:
    """正在处理中（或排队中）的会话数。"""
    return _session_locks.active()


def run_turn(session_id: str, text: str) -> Task:
    """跑一轮对话。

    上一轮若停在 ``CLARIFYING``，本轮输入就是用户的补充说明，续跑**同一个任务**；
    否则新建任务。这与控制台的行为一致，客户端只需要一个端点。

    整轮在**本会话的锁**内完成：同会话的下一个请求会等，别的会话不受影响。

    同会话排队超 ``SERVER_MAX_QUEUE_PER_SESSION`` 时抛 :class:`QueueFullError`，
    由 HTTP 层翻译成 503 + ``Retry-After`` 头。
    """
    manager = get_manager()
    cap = manager.settings.server.max_queue_per_session
    with _session_locks.acquire(session_id, max_waiters=cap):
        pending = manager.pending_clarification(session_id)
        if pending is not None:
            return manager.provide_clarification(pending.task_id, text)
        task = manager.create_task(text, session_id=session_id)
        return manager.run(task.task_id)


# ================================================================ 流式响应


_HEARTBEAT_INTERVAL = 15.0
"""SSE 心跳间隔——避免被中间代理（nginx/CDN）当成空闲连接掐掉。"""

_STREAM_QUEUE_MAX = 128
"""事件队列上限。客户端慢到撑爆这个时新事件被丢弃，任务继续推进，
任务结束后客户端可以从 ``/api/tasks/<id>`` 拿完整快照。"""

# 哨兵：分别表示"任务正常结束"和"任务出错"
_DONE = object()
_ERROR = object()


def _format_sse(event: TaskEvent) -> str:
    """把 :class:`TaskEvent` 序列化成 SSE 文本。

    ``TOOL_CALLED`` 的 ``result`` 字段可能很大（搜索工具回几百条记录），流上不带
    详细数据——客户端拿到 ``task_id`` 后去 ``/api/tasks/<id>`` 看完整消息即可。
    """
    data: dict[str, object] = {
        "kind": event.kind.value,
        "task_id": event.task_id,
        "session_id": event.session_id,
        "status": event.status.value,
        "summary": event.summary,
        "payload": event.payload,
        "at": event.at.isoformat(),
    }
    return f"event: {event.kind.value}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_turn(session_id: str, text: str) -> Iterator[str]:
    """以 SSE 格式流式跑一轮对话。

    与 :func:`run_turn` 行为等价：同会话排队、同会话串行、新建 vs. 追问沿用同一个任务。
    不同点是**任务在后台线程跑**，主线程从队列里拉事件、转 SSE 推给客户端。

    事件序列：``task_created`` → ``status_changed`` ×N → ``tool_called`` ×N →
    ``task_finished``。客户端用浏览器 ``EventSource`` 直接订阅即可。

    **断连处理**：客户端断开时本生成器被 close，背景线程**继续跑**到结束（避免引入
    跨线程改任务状态的竞态），但事件会因队列满而被丢弃。任务结束后通过 ``TASK_FINISHED``
    事件告知客户端最终结果——但这条也会被丢，所以客户端断线重连后应当去 ``/api/tasks/<id>``
    拿最终快照。
    """
    manager = get_manager()
    cap = manager.settings.server.max_queue_per_session

    # 有界队列 + drop-new：客户端慢就丢最新事件，比阻塞任务线程安全
    event_queue: queue.Queue[TaskEvent | object] = queue.Queue(maxsize=_STREAM_QUEUE_MAX)

    def listener(event: TaskEvent) -> None:
        try:
            event_queue.put_nowait(event)
        except queue.Full:
            # 队列满说明客户端跟不上——丢这条。下一次状态变化还是会推，UI 上不会
            # 卡死，只是不够细；TASK_FINISHED 一定会进来，因为最后状态变化必触发
            logger.debug("SSE 队列已满，丢弃事件 %s", event.kind.value)

    def runner() -> None:
        """后台线程：跑任务。所有错误都收成哨兵，由生成器翻译给客户端。"""
        try:
            manager.add_ephemeral(listener)
            # 锁必须在 listener 注册**之后**抢，否则队列抢到了也拿不到 TASK_CREATED
            try:
                with _session_locks.acquire(session_id, max_waiters=cap):
                    pending = manager.pending_clarification(session_id)
                    if pending is not None:
                        manager.provide_clarification(pending.task_id, text)
                    else:
                        task = manager.create_task(text, session_id=session_id)
                        manager.run(task.task_id)
            except QueueFullError:
                event_queue.put(_ERROR)
            except AgentError:
                # 业务错误已经在事件流里以 TASK_FINISHED(status=failed) 发出，
                # 这里不需要重复通知
                pass
        except Exception:
            logger.exception("流式任务后台线程崩溃")
            event_queue.put(_ERROR)
        finally:
            manager.remove_ephemeral(listener)
            event_queue.put(_DONE)

    thread = threading.Thread(target=runner, name=f"agent-stream-{session_id[:12]}", daemon=True)
    thread.start()

    last_heartbeat = time.monotonic()
    try:
        while True:
            try:
                item = event_queue.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                continue
            if item is _DONE:
                break
            if item is _ERROR:
                yield 'event: error\ndata: {"error": "任务未完成"}\n\n'
                break
            assert isinstance(item, TaskEvent)
            yield _format_sse(item)
    finally:
        # 客户端断开时本生成器被 close，但后台线程继续跑——它有自己的 _DONE 哨兵兜底
        pass
