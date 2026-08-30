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

# 4. 运行测试与静态检查（四项须全绿）
pytest
ruff check .
ruff format --check .           # 亦覆盖 README 内的 python 代码块
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

## 错误处理与日志

所有内部异常继承 `AgentError`，携带 `code`（机器可读、用于 API 响应与指标）、`message`、
`retryable`、`detail` 四项。其中 `retryable` 是任务状态机在 `RETRYING` 与 `FAILED` 之间
分流的唯一依据，也是 LLM 客户端决定是否退避重试的唯一依据：

```python
from agent.common import ToolTimeoutError, log_context, setup_logging, get_logger, new_trace_id

setup_logging("INFO")  # 单行 JSON 输出到 stderr，可重复调用不叠加
logger = get_logger(__name__)

with log_context(trace_id=new_trace_id(), task_id="task_..."):
    logger.info("开始执行", extra={"tool": "search_tool"})
```

`trace_id / session_id / task_id` 存放在 `contextvars` 中，由日志过滤器自动注入每条记录，
业务代码无需层层传递；用 `contextvars` 而非 `threading.local` 是为了同时隔离线程与
asyncio 任务，适配后续的 SSE 流式接口。记录 `AgentError` 时，`code` 与 `retryable`
会一并落进日志，排查时无需回查代码即可判断任务为何走了重试分支。

## 枚举取值域

`agent/common/enums.py` 固化 `TaskStatus`、`IntentType`、`MessageRole`、`ArtifactType`、
`OperationStatus`、`ContextScope` 六个取值域。全部继承 `StrEnum`，成员值即小写字符串，
可直接 `json.dumps`，且 `TaskStatus.CREATED == "created"` 成立：

```python
from agent.common import IntentType

IntentType("new_query")  # 只认规范拼写
IntentType.values()  # 供 JSON Schema 的 enum 约束
IntentType.from_str(raw)  # 解析失败抛 ValueError，消息内含合法取值列表
IntentType.from_str(raw, default=IntentType.UNKNOWN)  # 调用方显式选择的兜底
```

取值域是**严格**的：`IntentType("NEW-QUERY")`、`IntentType("newQuery")`、
`IntentType("analisis")` 全部抛 `ValueError`。不做归一化、不设别名表是有意为之——
把错误拼写猜成正确成员会掩盖真正的缺陷（提示词写错、前端传错字段、模型不遵守
schema），而猜测规则本身会长期膨胀并需要维护。

约束模型输出的正确位置是产出端：用 `values()` 生成 JSON Schema 的 enum 约束。
解析失败时的 `ValueError` 已带合法取值列表，可直接回灌重试提示词。
旧的 `agent/task_manage/type.py` 保留为兼容重导出层，只提供 `TaskType` 类别名。

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

## 消息与产物模型

`agent/models/` 是纯数据层，只依赖 `common` / `config`，不含任何业务策略。

`Message` 同时承载协议字段与编排字段，二者读者不同，故分两个出口：`to_llm_dict()`
只输出 OpenAI messages 规范允许的键（`role` / `content` / `name` / `tool_calls` /
`tool_call_id`），而 `message_id`、`artifact_refs`、`scope`、`source_task_id`、
`created_at`、`meta` 只服务于 ContextManager，绝不进入发给模型的载荷——否则既浪费
token，又会诱导模型模仿这些字段作答。

`Artifact` 存在的理由是隔离体积。一次 ES 查询可能回来上万行，若直接进 `task_content`，
上下文窗口会被单次结果打满且每轮都要重投一遍。因此全量数据留在 `data`（或
`storage_ref` 指向的外部存储），进入提示词的只有 `artifact_id` 与 `preview()` 摘要：

```python
artifact.preview(max_rows=5)
# [table] 查询结果 (artifact_id=art_...) 共 200000 行
# col0 | col1 | col2
# r0c0 | r0c1 | r0c2
# ...
# （另有 199995 行未显示，凭 artifact_id 可取全量）
```

`preview()` 在行数、单元格宽度、总字符数三个维度都设了硬上限，否则上述约定形同虚设：
20 万行、3 列的表格摘要实测 217 字符；500 列的极端表格会被总字符上限截断在 1200 字符。

两个模型都继承 `AgentModel`，统一提供 `to_dict()`（`mode="json"`，datetime 与枚举
已转为字符串，可直接 `json.dumps`）与 `from_dict()`，并统一开启 `extra="forbid"`
与 `validate_assignment=True`——未知字段被静默丢弃时，现象是「值莫名变成默认值」，
排查成本远高于当场报错。

## TaskSummary 与 Task

`TaskSummary` 的字段集与需求文档一致，命名按开发计划 1.2 节修正：`related_task_ids`
而不是 `related task`。`to_prompt_dict(fields)` 按白名单投影——意图识别阶段用
`INTENT_PROMPT_FIELDS`（`content` + `status`），避免把尚未发生的 operations 或
失败现场的 `output` 投喂给模型。空列表与未知字段名一律拒绝。

`Task` 只做数据访问：`create()` 要求显式传入 `session_id`（漏传会让每个请求变成
独立会话，WorkingMemory 静默失效）；`touch()` 刷新 `updated_at`；改状态只能走
`record_status()`——直接赋 `Task.status` 或 `summary.status` 都会被拦住，避免归档后
出现「实体已完成、摘要仍显示 intending」。**转移是否合法不在此处裁定**。

`error` 的类型是 `ErrorInfo`（`code` / `message` / `retryable` / `detail`），与
`AgentError.to_dict()` 键集合一致，供落盘、状态机分流和步骤 26 的 HTTP 错误响应共用。
`operations` 的 `index` 必须从 0 按列表顺序连续递增，这样 `operation(i)` 与执行循环
的列表遍历是同一种解读。

规范定义在 `agent/models/task.py`；`agent/task_manage/task.py` 只是兼容重导出。

## Context Window

`ContextWindow` 把 `summary`、`content`、`artifacts` 聚合成一块黑板，只提供容器操作：
`append_message`、`get_messages`、`put_artifact`、`get_artifact`、`list_artifact_index`、
`split_by_scope`。裁剪预算、写回 WorkingMemory 的策略不在这里。

`split_by_scope()` 按 `Message.scope` / `Artifact.scope` 切开。写回只应取
`CURRENT`；`RELATED`（关联任务注入）和 `HISTORY`（历史片段）参与推理但不归档，
否则会话记忆会随轮次指数膨胀。

注入片段必须带 `source_task_id`；`scope=current` 的产物必须属于本窗口的
`task_id`。`list_artifact_index()` 默认只列 `CURRENT`，避免步骤 28 把关联表
暴露给前端。`get_messages()` 始终返回新列表。`record_status()` 之后必须
`bind_summary(task.summary)`，否则窗口里仍是过期摘要。

## 存储

`agent/store` 提供 `KVStore` / `ListStore` 抽象与线程安全的 `MemoryStore`。
业务代码只依赖抽象：`get` / `set` / `delete` / `exists` / `keys` / `expire` 与
`push` / `range` / `trim` / `length`。语义对齐 Redis（闭区间、负下标），
步骤 27 的 Redis 实现加入 `STORE_FACTORIES` 后整份契约测试自动覆盖。

键名由 `make_key(session_id, namespace, id)` 拼成 `agent:{session}:{ns}:{id}`，
段内禁止冒号。TTL 惰性过期；读出的值是拷贝。KV 与 List 不能共用同一键。

## LLM 客户端

`agent/llm` 提供厂商无关的调用面。上层只依赖 `BaseLLMClient.chat` /
`stream_chat`，不感知 OpenAI 或其它供应商。

`LLMRequest.messages` 使用 `Message`：发出去之前统一走 `to_llm_dict()`，
编排字段不会漏进载荷。`model` / `temperature` / `max_tokens` 留空时回落到
`LLMProfile`。`chat()` 拒绝 `stream=True`，流式走 `stream_chat()`。

`OpenAICompatClient` 基于 `openai` SDK 的兼容模式。SDK 自带重试钉死为 0——
退避与降级由 `ResilientLLMClient` 独占，两层各自重试会让次数变成乘法。
SDK 异常翻译为 `LLMTimeoutError` / `LLMRateLimitError` / `LLMError`，并带上
`retryable`。单测打桩 HTTP 层（不是 SDK 方法），以校验真实请求体。

`build_llm_client(settings)` 是装配入口：`LLM_USE_MOCK=true` 走
`MockLLMClient`，否则按 `profile_chain` 包一层 `ResilientLLMClient`。
同一端点只对 `retryable=True` 指数退避；耗尽后再降级。`purpose` 只覆盖
**主模型**的名字；备用端点用各自的 `profile.model`。调用方写死 `model`
则整条链共用。超时在内层客户端，装饰器无总墙钟；流式只对首个 chunk
创建失败降级，流出后中途断流不换端点。`LLM_USE_MOCK=true` 返回裸 Mock，
不包 Resilient。脚本化的 `LLMTimeoutError` 只能被消费一次，同客户端重试须用 callable。

`call_structured` 优先 `response_format=json_schema`，端点不支持则退回
提示词约束 + JSON 提取；校验失败带错误重问一次，再失败抛
`LLMResponseFormatError`。`MockLLMClient` 按序号或 `when` 谓词回放，
并记录完整 messages，供后续阶段断言 prompt。

```python
from agent.config import get_settings
from agent.llm import LLMRequest, build_llm_client, call_structured
from agent.models import Message

client = build_llm_client(get_settings())
response = client.chat(LLMRequest(messages=[Message.user("查昨天的错误日志")]))
data = call_structured(
    client,
    [Message.user("查昨天的错误日志")],
    {"type": "object", "required": ["intent"], "properties": {"intent": {"type": "string"}}},
    purpose="intent",
)
```

## WorkingMemory

`WorkingMemory(session_id, store)` 按会话保存三部分历史：
`task_summary_history` / `task_content_history` / `task_artifact_history`。
`archive(window)` 只落 `scope=CURRENT` 的 content 与 artifacts；写入顺序记在
独立 List 键上，不依赖 `keys()`。`trim(max_tasks)` 淘汰最旧任务。
`get_task_bundle(task_id)` 是关联任务注入的唯一读口。不同 session 互不可见。

```python
from agent.memory_manage import WorkingMemory
from agent.store import MemoryStore

memory = WorkingMemory(session_id, MemoryStore())
memory.archive(window)
memory.get_summary_history(limit=10)
summary, content, artifacts = memory.get_task_bundle(task_id)
```

## 开发进度

开发计划共 36 个步骤、6 个里程碑。**当前完成到 M2 的步骤 11**：
WorkingMemory 已交付，LLM purpose 路由按当前 profile 解析。

| 里程碑 | 步骤 | 状态 |
| --- | --- | --- |
| M1 基础设施 | 1 – 8 | 已完成 |
| M2 四大模块 | 9 – 18 | 进行中（11/18） |
| M3 单任务闭环 | 19 – 24 | 未开始 |
| M4 服务可用 | 25 – 28 | 未开始 |
| M5 流式与质量 | 29 – 32 | 未开始 |
| M6 生产化 | 33 – 36 | 未开始 |
