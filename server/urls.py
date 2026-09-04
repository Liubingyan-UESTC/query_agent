"""URL 路由。所有接口都挂在 /api 下。"""

from django.urls import include, path

urlpatterns = [
    path("api/", include("server.api.urls")),
]
