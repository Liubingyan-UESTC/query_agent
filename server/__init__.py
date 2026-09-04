"""网络服务层：把 Agent 内核包装成 HTTP 接口。

这一层可以 import :mod:`agent`，**反过来绝对不行**——Agent 是被服务包着的内核，
不是 Django 的一部分。这条边界让控制台与 HTTP 两个入口共用同一份内核。
"""
