"""四张表：会话、任务、消息、工具调用。

表名用 ``db_table`` 显式钉死为 ``sessions / tasks / messages / tool_calls``，
免得默认前缀（``api_session`` 之类）让人对着 sqlite3 命令行找不到表。

主键一律用业务 id（``session_id`` / ``task_id`` / ``message_id`` / ``call_id``）而不是
自增整数：这些 id 由 Agent 内核生成，本来就全局唯一，再加一列代理键只会多一次映射。
"""

from django.db import models


class Session(models.Model):
    """一次会话。由会话凭证（X-Session-Id / Cookie）标识。"""

    session_id = models.CharField(primary_key=True, max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    user_agent = models.TextField(null=True, blank=True)
    ip = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        db_table = "sessions"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.session_id


class Task(models.Model):
    """一次用户请求。``summary_json`` 是 task_summary 的完整快照。"""

    task_id = models.CharField(primary_key=True, max_length=64)
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="tasks")
    status = models.CharField(max_length=32, db_index=True)
    intent = models.CharField(max_length=32, null=True, blank=True)
    summary_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "tasks"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.task_id}({self.status})"


class ToolCall(models.Model):
    """一次工具调用，一条记录。

    结果可能很大（几百条日志），所以分两处存：``result_json`` 只放能内联的部分，
    超限时全量写文件、``result_ref`` 存路径。见 :mod:`server.api.listener`。
    """

    call_id = models.CharField(primary_key=True, max_length=64)
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="tool_calls")
    tool_name = models.CharField(max_length=64, db_index=True)
    arguments_json = models.JSONField(default=dict)
    result_json = models.JSONField(null=True, blank=True)
    result_ref = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(max_length=16)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "tool_calls"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.call_id}:{self.tool_name}({self.status})"


class Message(models.Model):
    """一条消息，一条记录（system / user / assistant / tool）。"""

    message_id = models.CharField(primary_key=True, max_length=64)
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=16, db_index=True)
    content = models.TextField(blank=True)
    tool_call = models.ForeignKey(
        ToolCall,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="messages",
        db_column="tool_call_id",
    )
    created_at = models.DateTimeField()

    class Meta:
        db_table = "messages"
        ordering = ["created_at", "message_id"]

    def __str__(self) -> str:
        return f"{self.message_id}:{self.role}"
