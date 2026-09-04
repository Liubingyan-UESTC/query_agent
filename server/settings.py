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

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "data" / "agent.sqlite3",
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
