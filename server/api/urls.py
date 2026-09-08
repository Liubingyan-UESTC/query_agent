"""API 路由。"""

from django.urls import path

from server.api import views

urlpatterns = [
    path("chat", views.chat, name="chat"),
    path("chat/stream", views.chat_stream, name="chat-stream"),
    path("history", views.history, name="history"),
    path("tasks/<str:task_id>", views.task_detail, name="task-detail"),
    path("health", views.health, name="health"),
]
