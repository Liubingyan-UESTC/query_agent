"""日志系统测试：双写、轮转、级别、格式、幂等。"""

import logging
from pathlib import Path

import pytest

from agent.config import load_settings
from agent.enums import TaskStatus
from agent.events import EventKind, TaskEvent
from agent.logging_setup import ROOT_LOGGER_NAME, LoggingListener, get_logger, setup_logging


@pytest.fixture
def log_settings(tmp_path: Path):
    """日志目录指向 tmp，避免测试往仓库的 ./logs 里写东西。"""
    return load_settings(env_file=None, log={"dir": tmp_path / "logs", "level": "DEBUG"})


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """每个用例前后都把 agent logger 上的 handler 清干净，避免互相污染。"""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    saved = list(logger.handlers)
    logger.handlers.clear()
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers[:] = saved


def read_log(settings) -> str:
    return settings.log.resolved_file().read_text(encoding="utf-8")


class TestSetup:
    def test_creates_missing_directory(self, log_settings):
        """Django / CLI 启动时目录可能压根不存在，必须自动建出来。"""
        target = log_settings.log.resolved_dir()
        assert not target.exists()
        setup_logging(log_settings)
        assert target.is_dir()

    def test_creates_nested_directory(self, tmp_path):
        settings = load_settings(env_file=None, log={"dir": tmp_path / "a" / "b" / "logs"})
        setup_logging(settings)
        assert settings.log.resolved_dir().is_dir()

    def test_writes_to_both_console_and_file(self, log_settings, capsys):
        setup_logging(log_settings)
        get_logger("task").info("状态转移测试", extra={"task_status": "executing"})

        assert "状态转移测试" in read_log(log_settings)
        # 控制台走 stderr，与 CLI 的 stdout 分开
        assert "状态转移测试" in capsys.readouterr().err

    def test_console_can_be_disabled(self, tmp_path, capsys):
        settings = load_settings(env_file=None, log={"dir": tmp_path / "logs", "to_console": False})
        setup_logging(settings)
        get_logger("task").info("只进文件")

        assert "只进文件" in read_log(settings)
        assert capsys.readouterr().err == ""

    def test_is_idempotent(self, log_settings):
        """Django autoreload 会把 settings 导入两次，重复配置不能让日志翻倍。"""
        setup_logging(log_settings)
        setup_logging(log_settings)
        get_logger("task").info("只应出现一次")

        assert read_log(log_settings).count("只应出现一次") == 1

    def test_force_reconfigures(self, log_settings):
        setup_logging(log_settings)
        before = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)
        setup_logging(log_settings, force=True)
        assert len(logging.getLogger(ROOT_LOGGER_NAME).handlers) == before

    def test_does_not_propagate_to_root(self, log_settings):
        """向上冒泡会让 pytest / Django 自带的 root handler 把每条日志再印一遍。"""
        setup_logging(log_settings)
        assert logging.getLogger(ROOT_LOGGER_NAME).propagate is False

    def test_rotation_is_configured(self, log_settings):
        from logging.handlers import RotatingFileHandler

        setup_logging(log_settings)
        handlers = [
            h
            for h in logging.getLogger(ROOT_LOGGER_NAME).handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(handlers) == 1
        assert handlers[0].maxBytes == log_settings.log.max_bytes
        assert handlers[0].backupCount == log_settings.log.backup_count

    def test_rotation_actually_rolls_over(self, tmp_path):
        """写满就轮转，日志文件不会无限膨胀。"""
        settings = load_settings(
            env_file=None,
            log={"dir": tmp_path / "logs", "max_bytes": 1024, "backup_count": 2},
        )
        setup_logging(settings)
        logger = get_logger("task")
        for i in range(200):
            logger.info("填充日志行 %d %s", i, "x" * 50)

        rotated = sorted(settings.log.resolved_dir().glob("log.txt*"))
        assert len(rotated) > 1
        assert settings.log.resolved_file().stat().st_size <= 1024 * 3


class TestLevels:
    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_each_level_is_accepted(self, tmp_path, level):
        settings = load_settings(env_file=None, log={"dir": tmp_path / "logs", "level": level})
        setup_logging(settings)
        assert logging.getLogger(ROOT_LOGGER_NAME).level == getattr(logging, level)

    def test_level_filters_lower_records(self, tmp_path):
        settings = load_settings(env_file=None, log={"dir": tmp_path / "logs", "level": "WARNING"})
        setup_logging(settings)
        logger = get_logger("task")
        logger.info("这条不该出现")
        logger.warning("这条应该出现")

        body = settings.log.resolved_file().read_text(encoding="utf-8")
        assert "这条不该出现" not in body
        assert "这条应该出现" in body

    def test_level_is_case_insensitive(self, tmp_path):
        settings = load_settings(env_file=None, log={"dir": tmp_path / "logs", "level": "debug"})
        assert settings.log.level == "DEBUG"


class TestFormat:
    def test_line_has_time_level_logger_status_and_summary(self, log_settings):
        setup_logging(log_settings)
        get_logger("task").warning("退避后重试", extra={"task_status": "retrying"})

        line = read_log(log_settings).strip().splitlines()[-1]
        columns = [part.strip() for part in line.split("|")]
        assert len(columns) == 5
        assert columns[0].startswith("20")  # 时间
        assert columns[1] == "WARNING"  # 级别
        assert columns[2] == "agent.task"  # 来源
        assert columns[3] == "retrying"  # 任务状态
        assert columns[4] == "退避后重试"  # 事件摘要

    def test_records_without_task_status_get_a_placeholder(self, log_settings):
        """第三方库往我们的 handler 里写日志时不会带 task_status，不能因此 KeyError。"""
        setup_logging(log_settings)
        get_logger("django.request").error("GET /api/chat 500")

        line = read_log(log_settings).strip().splitlines()[-1]
        assert [p.strip() for p in line.split("|")][3] == "-"

    def test_exception_traceback_is_captured(self, log_settings):
        setup_logging(log_settings)
        try:
            raise RuntimeError("落库失败")
        except RuntimeError:
            get_logger("events").exception("监听器出错")

        body = read_log(log_settings)
        assert "Traceback" in body
        assert "落库失败" in body


class TestLoggingListener:
    def event(self, **kwargs):
        defaults = {
            "kind": EventKind.STATUS_CHANGED,
            "task_id": "task_1",
            "session_id": "sess_1",
            "status": TaskStatus.EXECUTING,
            "summary": "状态转移 planning→executing",
        }
        return TaskEvent(**{**defaults, **kwargs})

    def test_logs_status_task_and_session(self, log_settings):
        setup_logging(log_settings)
        LoggingListener()(self.event())

        body = read_log(log_settings)
        assert "状态转移 planning→executing" in body
        assert "task=task_1" in body
        assert "session=sess_1" in body
        assert "executing" in body

    def test_tool_events_go_to_the_tool_logger(self, log_settings):
        setup_logging(log_settings)
        LoggingListener()(self.event(kind=EventKind.TOOL_CALLED, summary="调用 search_tool → ok"))
        assert "agent.tool" in read_log(log_settings)

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (TaskStatus.EXECUTING, "INFO"),
            (TaskStatus.RETRYING, "WARNING"),
            (TaskStatus.CANCELED, "WARNING"),
            (TaskStatus.FAILED, "ERROR"),
            (TaskStatus.ABORTED, "ERROR"),
        ],
    )
    def test_level_follows_severity(self, log_settings, status, expected):
        """出问题的行必须自己浮上来，别埋在 INFO 里。"""
        setup_logging(log_settings)
        LoggingListener()(self.event(status=status, summary="事件"))
        assert expected in read_log(log_settings)

    def test_failed_tool_call_is_a_warning(self, log_settings):
        setup_logging(log_settings)
        LoggingListener()(
            self.event(
                kind=EventKind.TOOL_CALLED,
                summary="调用 analysis_tool → 失败",
                payload={"ok": False},
            )
        )
        assert "WARNING" in read_log(log_settings)
