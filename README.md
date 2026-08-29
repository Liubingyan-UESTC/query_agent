# Query Agent

面向 Kibana / Elasticsearch 的数据查询、分析与导出 Agent 系统。用户的一次自然语言请求即一个
Task，系统以状态机驱动其经历「意图识别 → 规划 → 执行 → 校验 → 完成」的完整生命周期，
并以黑板模式（Context Window）在各阶段之间传递上下文。

- 需求说明：[`docs/require.md`](docs/require.md)
- 开发计划与里程碑：[`docs/dev_plan.md`](docs/dev_plan.md)

## 核心概念

| 概念 | 说明 |
| --- | --- |
| **Task** | 用户的一次请求，拥有 `TaskStatus` 状态并由 `TaskManager` 驱动状态转移 |
| **Context Window** | 单个任务的全部工作空间，由 `task_summary` + `task_content` + `task_artifacts` 三部分组成 |
| **WorkingMemory** | 会话级历史记忆，与 Context 三部分一一对应，供意图识别与关联任务注入使用 |
| **KnowledgeMemory** | 外置的固定知识（系统提示词、skill、字段字典），以资源文件维护 |
| **Artifact** | 工具产出的结构化数据；大体积数据只进 artifacts，对话中仅保留 `artifact_id` + 预览 |

## 环境要求

- Python >= 3.11（开发容器使用 3.11，本地已验证 3.13）
- 可选：Redis（步骤 33 起用于多进程会话共享）、Elasticsearch（步骤 35 起用于真实数据源）

`.devcontainer/` 已纳入版本管理，用 VS Code / Cursor 打开容器即自动安装 `requirements.txt`。
容器配置中不含任何密钥，敏感值统一以 `${localEnv:VAR}` 从宿主机环境读取（如 `ANTHROPIC_AUTH_TOKEN`）。

## 本地启动

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备环境变量
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
# 编辑 .env 填入 LLM_PRIMARY_API_KEY 等必填项；本地无密钥时可设 LLM_USE_MOCK=true

# 4. 运行测试与静态检查
pytest
ruff check .
mypy
```

Web 服务（`python app/manage.py runserver`）与命令行交互脚本（`python scripts/run_cli.py`）
分别在开发计划的步骤 26 与步骤 25 交付，当前尚未可用。

## 配置

全部可变参数集中声明在 `agent/config/settings.py`，业务代码只通过 `get_settings()` 读取，
禁止直读 `os.environ`：

```python
from agent.config import get_settings

settings = get_settings()  # 进程级单例，启动时一次性确定
settings.llm.model_for("intent")  # 按用途路由模型，未单独配置则回落主模型
settings.context.token_budget()  # 按占比换算出的各部分 token 预算
```

环境变量命名为 `<组前缀>_<字段名>`，前缀与配置组一一对应：`APP_`、`LLM_`、`LLM_PRIMARY_`、
`CONTEXT_`、`MEMORY_`、`TOOL_`、`TASK_`、`STORE_`、`ES_`、`TRACE_`；全部键名见 `.env.example`
（有测试守护二者不漂移）。配置缺失或非法时抛出 `ConfigError`，错误信息直接指明环境变量名。

## 目录说明

```
query_agent/
├── agent/                  # Agent 内核（不依赖 Web 框架，可独立以脚本驱动）
│   ├── common/             # 枚举、异常体系、结构化日志、ID 生成、指标与追踪
│   ├── config/             # 配置中心，全部可变参数的唯一声明处
│   ├── models/             # 纯数据模型：Message / Artifact / TaskSummary / Task / ContextWindow
│   ├── store/              # 存储抽象与内存实现（后续可换 Redis / DB，业务代码不改）
│   ├── llm/                # LLM Layer：OpenAI 兼容客户端、降级重试、结构化输出、Mock
│   ├── memory_manage/      # WorkingMemory 与 KnowledgeMemory
│   ├── knowledge/          # 知识资源文件：系统提示词 / skill / 字段字典
│   ├── context_manage/     # Context Window 生命周期、预算裁剪、关联任务注入
│   ├── prompt/             # 各阶段提示词装配
│   ├── tool_manage/        # 工具抽象、注册发现、调用管理与工具实例
│   └── task_manage/        # 状态机、阶段处理器、终态收口、TaskManager
├── app/                    # Django 网络服务层（步骤 26 起）
├── tests/                  # 单测 / 契约测试 / 端到端 / 接口测试
└── docs/                   # 需求、开发计划、接口契约与部署文档
```

## 分层依赖约束

依赖方向严格单向，反向引用视为架构违规：

```
common / config → models → store → { llm, memory_manage, tool_manage }
    → context_manage ← prompt → task_manage → runtime → app
```

其余需在 Code Review 中守护的关键不变式（黑板单一写入通道、大数据不入对话、注入内容不归档、
状态转移唯一裁判等）见 [`docs/dev_plan.md`](docs/dev_plan.md) 第四节。

## 开发进度

开发计划共 36 个步骤、6 个里程碑。**当前完成到 M1（步骤 1–8）的步骤 2**：
步骤 1「项目骨架与依赖管理」、步骤 2「配置中心」已交付；
步骤 3–8（异常与日志、全局枚举、数据模型、ContextWindow、存储抽象）尚未开始，
因此 M1 的里程碑成果（模型可序列化往返、Store 通过契约测试）目前还不具备。

| 里程碑 | 步骤 | 状态 |
| --- | --- | --- |
| M1 基础设施 | 1 – 8 | 进行中（2/8） |
| M2 四大模块 | 9 – 18 | 未开始 |
| M3 单任务闭环 | 19 – 24 | 未开始 |
| M4 服务可用 | 25 – 28 | 未开始 |
| M5 流式与质量 | 29 – 32 | 未开始 |
| M6 生产化 | 33 – 36 | 未开始 |
