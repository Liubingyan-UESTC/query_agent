"""共享 fixture。

统一原则：测试**不读项目根的 .env**（``env_file=None``），也不触网（MockLLM），
数据文件与日志目录指向临时目录，保证结果只取决于用例自身。
"""

import json
from pathlib import Path

import pytest

from agent.config import load_settings
from agent.logging_setup import setup_logging


@pytest.fixture(autouse=True, scope="session")
def isolated_log_settings(tmp_path_factory):
    """把日志重定向到临时目录，并把这份配置暴露给需要重配日志的用例。

    Django 的 settings 在 pytest 启动时就配好了日志，指向仓库真实的 ``logs/log.txt``。
    不隔离的话，测试里那些故意制造的异常（"数据库连接断了"之类）会混进运维日志，
    让真正 tail 这个文件的人白排查一场。

    需要临时改日志配置的用例（如 test_api 里那个断言日志内容的），**必须用这份配置
    恢复**，不能调 ``load_settings()`` 默认值——那会把后续用例的日志又指回仓库。
    """
    settings = load_settings(env_file=None, log={"dir": tmp_path_factory.mktemp("logs")})
    setup_logging(settings, force=True)
    return settings


SAMPLE_LOGS = [
    {
        "timestamp": "2026-08-31T01:00:00Z",
        "level": "ERROR",
        "service": "order-service",
        "host": "node-1",
        "message": "Read timed out after 3000ms",
        "status_code": 504,
        "latency_ms": 3200,
        "user_id": "u_1001",
    },
    {
        "timestamp": "2026-08-31T01:05:00Z",
        "level": "ERROR",
        "service": "order-service",
        "host": "node-2",
        "message": "Database deadlock detected",
        "status_code": 500,
        "latency_ms": 1800,
        "user_id": "u_1002",
    },
    {
        "timestamp": "2026-08-31T01:10:00Z",
        "level": "ERROR",
        "service": "payment-service",
        "host": "node-1",
        "message": "Connection reset by peer",
        "status_code": 502,
        "latency_ms": 900,
        "user_id": "u_1003",
    },
    {
        "timestamp": "2026-08-31T01:15:00Z",
        "level": "INFO",
        "service": "user-service",
        "host": "node-3",
        "message": "User profile loaded",
        "status_code": 200,
        "latency_ms": 40,
        "user_id": "u_1004",
    },
    {
        "timestamp": "2026-08-31T01:20:00Z",
        "level": "WARN",
        "service": "gateway",
        "host": "node-4",
        "message": "Circuit breaker half-open",
        "status_code": 200,
        "latency_ms": 300,
        "user_id": "u_1005",
    },
]


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """一个只有 5 条记录的小型日志库，便于断言精确数字。"""
    path = tmp_path / "logs.json"
    path.write_text(
        json.dumps({"index": "logs-test", "records": SAMPLE_LOGS}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def settings(log_file: Path):
    """测试用配置：不读 .env、强制 Mock、数据文件指向 tmp。"""
    return load_settings(
        env_file=None,
        llm={"use_mock": True},
        tool={"data_file": log_file},
    )
