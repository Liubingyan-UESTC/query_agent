### step2
下一步需要搭建网络服务和日志系统，将这个Agent系统的服务搭建为网络服务，使用python的Django作为后端框架，使用sqlite数据库记录会话的聊天记录，聊天记录分为四张表：至少 4 张表：sessions, tasks, messages, tool_calls。其中 messages 和 tool_calls 每条记录一行。
其中，
会话表（sessions）：session_id, created_at, user_agent（可选）, ip（可选）。

任务表（tasks）：task_id, session_id, status, intent, summary_json（存储 task_summary 的 JSON 快照）, created_at, updated_at。

消息表（messages）：message_id, task_id, role（system/user/assistant/tool）, content, tool_call_id（外键，可为空）, created_at。每条消息一行，便于查询和展示。

工具调用表（tool_calls）：call_id, task_id, tool_name, arguments_json, result_json（可选，但可能很大，建议存文件引用）, status, created_at。

在本地测试运行，监听7080端口，需要判断每一次请求是新的会话还是已有会话：
统一采用会话凭证（Session Token）机制：

第一次请求（无凭证）时，后端生成一个全局唯一的 session_id（如 UUID），并返回给客户端（可放在响应体或 Set-Cookie 中）。

后续请求，客户端必须在 HTTP Header（如 X-Session-Id）或 Cookie 中携带该凭证。

后端根据凭证查找或创建对应的会话记录（存在会话表里）。

加入日志系统，使用logging模块加入日志系统，将任务过程中的状态转换时间和其他重要事件的日志记录打印在屏幕上，并记录在./logs/log.txt下，记录时间、日志级别、任务状态、事件摘要等

支持日志级别（DEBUG/INFO/WARNING/ERROR）。

支持同时输出到控制台和文件（StreamHandler + FileHandler）。

支持自动轮转（RotatingFileHandler），防止日志文件无限膨胀。
日志文件路径 ./logs/log.txt 需确保目录存在，否则 Django 启动时需自动创建。
