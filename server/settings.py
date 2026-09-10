"""Django 配置。

只做本地测试服务，所以刻意省掉了 admin / auth / staticfiles 这些用不上的 app——
装得越少，启动越快，也越清楚这个服务到底依赖什么。

日志不用 Django 的 ``LOGGING`` 字典，而是把 ``LOGGING_CONFIG`` 设为 ``None`` 后调
:func:`agent.logging_setup.setup_logging`：控制台入口与 HTTP 入口共用同一套日志配置，
格式、轮转、目录创建都只有一处实现。
"""

from agent.config import PROJECT_ROOT, get_settings
from agent.logging_setup import setup_logging

_settings = get_settings()

BASE_DIR = PROJECT_ROOT

# 本地测试服务：密钥固定即可，不参与任何对外加密。真要上线必须换成从环境读。
SECRET_KEY = "django-insecure-local-only-query-agent"
DEBUG = _settings.debug
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "server.api",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
    "server.api.middleware.SessionTokenMiddleware",
]

ROOT_URLCONF = "server.urls"
WSGI_APPLICATION = "server.wsgi.application"

# SQLite 的三个并发参数，缺一个都会在多线程下出问题：
#
# - ``journal_mode=WAL``——默认的 rollback journal 下，一个写事务会阻塞**所有读**，
#   于是一次 /api/chat 落库就能把并发的 /api/history 卡住。WAL 让读写互不阻塞
#   （仍然是单写者）。这个 pragma 会持久化在库文件上，每次连接重设是幂等的。
# - ``timeout``——拿不到写锁时等多久。默认 5s 且 Django 不主动设，写冲突会直接抛
#   ``OperationalError: database is locked``。给到 20s，让它等而不是报错。
# - ``transaction_mode="IMMEDIATE"``——BEGIN 时就取写锁。默认的 DEFERRED 会在事务
#   中途从读升级为写，两个事务同时升级就必然有一方拿不到锁且**无法退避**（已经读过了），
#   立刻 "database is locked"。DbListener 每个事件都是"先读后写"，正是这个形态。
#
# 没设 ``ATOMIC_REQUESTS``：那会把写事务的范围拉长到整个请求，包住几十秒的 LLM 调用，
# 单写者的 SQLite 会被一个慢任务彻底堵死。
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "data" / "agent.sqlite3",
        # 每请求关连接。SQLite 是本地文件，建连极廉价，而复用连接会让 WAL 的读快照
        # 长期挂着，拖住 checkpoint。
        "CONN_MAX_AGE": 0,
        "OPTIONS": {
            "timeout": 20,
            "transaction_mode": "IMMEDIATE",
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
        },
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = True
TIME_ZONE = "UTC"

# 关掉 Django 自己的 dictConfig，改用 agent 的日志配置（两个入口共用一套）
LOGGING_CONFIG = None
setup_logging(_settings)

# SQLite 文件所在目录可能还不存在（首次 migrate 前）
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
