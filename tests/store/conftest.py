"""Store 契约测试的后端工厂。

新增实现（Redis）时在 `STORE_FACTORIES` 加一行，整份 `test_store_contract.py`
自动覆盖，无需复制用例。
"""

from collections.abc import Callable

import pytest

from agent.store.base import Store
from agent.store.memory_store import MemoryStore

STORE_FACTORIES: list[Callable[[], Store]] = [MemoryStore]


@pytest.fixture(params=STORE_FACTORIES, ids=lambda factory: factory.__name__)
def store(request: pytest.FixtureRequest) -> Store:
    return request.param()
