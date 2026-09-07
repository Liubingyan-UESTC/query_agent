"""并发闸门：限制全进程同时在途的模型调用数。

会话之间并发之后，并发请求数直接等于并发模型调用数。供应商的并发/RPM 额度是有限的，
撞上 429 之后 :class:`~agent.llm.openai_client.ResilientLLMClient` 会退避重试，线程挂得
更久——请求越多、退避越久、越容易再撞，是个正反馈。在出口处设一个信号量把它掐住。

**闸门的位置很关键**：包在**单个端点客户端**外侧，而不是包在 ``ResilientLLMClient``
外侧。这样退避 sleep 发生在闸门**之外**，等待中的线程不占许可——否则一个正在退避 8s 的
请求会白占一个并发位，把有效并发降到远低于设定值。
"""

from __future__ import annotations

import threading

from agent.llm.base import BaseLLMClient, LLMRequest, LLMResponse

__all__ = ["ConcurrencyGate", "GatedLLMClient"]


class ConcurrencyGate:
    """一个可被多个客户端共享的信号量。

    降级链上的每个端点各包一层 :class:`GatedLLMClient`，但它们共享**同一个** gate：
    否则每个端点各有 N 个许可，全链路的在途上限就变成了"端点数 × N"，闸门形同虚设。
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"并发上限必须 ≥ 1，收到 {limit}")
        self.limit = limit
        self._semaphore = threading.BoundedSemaphore(limit)

    def __enter__(self) -> ConcurrencyGate:
        self._semaphore.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._semaphore.release()


class GatedLLMClient(BaseLLMClient):
    """把一次 ``chat`` 圈进并发闸门。除此之外完全透传。"""

    def __init__(self, inner: BaseLLMClient, gate: ConcurrencyGate) -> None:
        self._inner = inner
        self._gate = gate

    def chat(self, request: LLMRequest) -> LLMResponse:
        with self._gate:
            return self._inner.chat(request)

    def close(self) -> None:
        self._inner.close()
