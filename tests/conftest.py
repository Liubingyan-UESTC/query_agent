"""共享 fixture。

统一原则：测试**不读项目根的 .env**（``env_file=None``），也不触网（MockLLM），
数据文件按需指向临时目录，保证结果只取决于用例自身。
"""

import json
from pathlib import Path

import pytest

from agent.config import load_settings

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
