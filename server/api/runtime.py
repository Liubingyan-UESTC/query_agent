"""进程内的 Agent 单例。

一个进程一个 :class:`~agent.task_manager.TaskManager`：``MemoryManager`` 本来就按
session_id 分桶，所以一个实例服务所有会话，追问「刚才那批」才能找到关联任务。

**并发**：Django 开发服务器是多线程的，而 TaskManager 的任务表、黑板都是普通 dict。
所以对外只暴露 :func:`run_turn`，内部用一把全局锁把"跑一轮任务"串行化。这是本地测试
服务的取舍——真要并发，得把任务状态搬去外部存储，让每个请求各持一份。

**重启不恢复**：任务与黑板只活在内存里。重启后 SQLite 里的历史照样能查，但"正等着澄清"
的任务会丢，用户重新提问即可。
"""

from __future__ import annotations

import threading

from agent.config import AppSettings, get_settings
from agent.logging_setup import LoggingListener, setup_logging
from agent.models import Task
from agent.task_manager import TaskManager, build_task_manager
from server.api.listener import DbListener

__all__ = ["get_manager", "reset_manager", "run_turn", "set_manager"]

_lock = threading.Lock()
_manager: TaskManager | None = None


def get_manager() -> TaskManager:
    """取进程内单例，首次调用时装配。"""
    global _manager
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
    _manager = manager


def reset_manager() -> None:
    set_manager(None)


def run_turn(session_id: str, text: str) -> Task:
    """跑一轮对话。

    上一轮若停在 ``CLARIFYING``，本轮输入就是用户的补充说明，续跑**同一个任务**；
    否则新建任务。这与控制台的行为一致，客户端只需要一个端点。
    """
    manager = get_manager()
    with _lock:
        pending = manager.pending_clarification(session_id)
        if pending is not None:
            return manager.provide_clarification(pending.task_id, text)
        task = manager.create_task(text, session_id=session_id)
        return manager.run(task.task_id)
