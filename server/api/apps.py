"""应用配置。"""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    name = "server.api"
    label = "api"
    verbose_name = "Query Agent API"
