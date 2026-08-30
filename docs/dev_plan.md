# Query Agent v1 开发计划

> 依据：`docs/require.md` + 架构补充说明（2026-08-29 确认）
> 目标：把 v0 原型重构为可工程化落地的 Agent 系统（Kibana 数据查询 / 分析 / 导出 / 对话）

---

## 一、总体约定

### 1.1 已确认的关键设计决策

| 决策项 | 结论 |
| --- | --- |
| Context Window 组成 | **三部分**：`task_summary` + `task_content` + `task_artifacts` |
| WorkingMemory 组成 | 与 Context 三部分一一对应：`task_summary_history` + `task_content_history` + `task_artifact_history` |
| Artifacts 定位 | 工具产出的结构化数据（结果集 / 统计表 / 导出文件引用）单独存放，`task_content` 中只保留引用与预览，避免上下文膨胀 |
| 存储 | 先实现纯内存 Store + 抽象接口，后续阶段替换为 Redis（会话级）+ DB（归档），业务代码不改 |
| 接口形态 | 先同步 REST（提交即阻塞返回），第二阶段追加 SSE 流式 + 任务查询/取消 |
| Kibana 对接 | 先定义 Tool 抽象 + Mock 数据源跑通全链路，再接真实 ES 客户端 |

### 1.2 命名与拼写规范

需求文档中的笔误在代码中统一修正，仅在对外 JSON 协议上保留兼容别名：

- `task summery` → `task_summary`（内部与对外统一使用 `task_summary`）
- `related task` → `related_task_ids`（对外 JSON 同时接受/输出 `related_task_ids`）
- 枚举沿用已有 `agent/task_manage/type.py` 的 `TaskStatus.INTENDING` 拼写
- 所有字段 snake_case，所有枚举值序列化为小写字符串
- ID 前缀须为纯小写字母（`new_id()` 强校验）：含下划线会破坏 `<前缀>_<时间戳>_<随机>` 的三段式切分

#### 实现期偏离记录

以下为实现过程中相对本计划原文的偏离，**后续步骤请以本表为准**，按原文书写会引用到不存在的符号。

| 计划原文 | 实际实现 | 偏离原因 | 落地于 |
| --- | --- | --- | --- |
| `MemoryError` | `AgentMemoryError` | 原名遮蔽 Python 内置 `MemoryError`，会使 `except MemoryError` 的语义随导入顺序漂移。已启用 ruff `A` 规则在 CI 拦截同类遮蔽 | 步骤 3 |
| 各枚举独立定义 | 统一继承新增的 `AgentEnum(StrEnum)` 基类 | 基类集中提供 `from_str()` 与 `values()`，供 LLM 输出与 HTTP 入参的解析端复用 | 步骤 4 |
| （无） | 新增 `TERMINAL_TASK_STATUSES` / `ACTIVE_TASK_STATUSES` / `FINISHED_OPERATION_STATUSES` 三个 `frozenset` 常量 | 状态机与轮询接口都需判断「是否终态」，若各处自行罗列状态，新增状态时必然漏改 | 步骤 4 |
| `type.py` 保留 v0 成员名以「避免破坏已有引用」 | 仅保留 `TaskType` 类别名，**不保留 v0 成员属性**：`TaskType.NEWQUERY` 抛 `AttributeError` | 与 1.2 节「修正笔误」直接冲突，取修正拼写：当前仓库无任何 v0 引用，且 mypy 会静态报出 `"type[IntentType]" has no attribute "NEWQUERY"` | 步骤 4 |
| `Artifact.schema` | `Artifact.data_schema` | `schema` 会遮蔽 `pydantic.BaseModel` 上已废弃的同名方法并触发 `UserWarning`。内外统一用 `data_schema`，不设 alias——该字段尚未成为对外契约，无需背兼容包袱 | 步骤 5 |
| 各模型自备 `to_dict()` / `from_dict()` | 统一继承新增的 `AgentModel(BaseModel)` 基类 | 步骤 5–7 的五个模型都要在 Store 中序列化往返，逐个手写必然风格分叉。基类同时统一了 `extra="forbid"` 与 `validate_assignment=True` | 步骤 5 |
| 在 `agent/task_manage/task.py` 补全 Task | 规范定义迁到 `agent/models/task.py`，`task_manage/task.py` 仅作兼容重导出 | Task 是纯数据，必须留在 models 层；放进 task_manage 会让 models 与 manager 循环引用，也破坏「Task 无任何对 manager 层的引用」这条验收 | 步骤 6 |
| 仅提供 `touch()` | 额外提供 `record_status()`，并用 validator + 冻结 `summary.status` 禁止绕过 | `Task.status`、`summary.status`、`status_history` 三处必须同步。本方法只记账，**不校验转移方向**；直接赋值在构造期与赋值期都会报错 | 步骤 6 |
| `Task.error: ?`（未规定形状） | `ErrorInfo{code, message, retryable, detail}` | 与 `AgentError.to_dict()` 键集合对齐。裸 `dict` 会让步骤 20/24/26 各写各的键名，`retryable` 悄然丢失 | 步骤 6 |
| `Task.create` 未规定 `session_id` 是否可省 | `session_id` 必填 | 漏传若静默 `new_session_id()`，每个请求变成独立会话，WorkingMemory 失效且无报错 | 步骤 6 |
| `Operation.index` 未规定与列表顺序的关系 | 必须从 0 按列表顺序连续递增 | `operation(index)` 按键查找，执行循环按列表遍历；乱序时两种解读结果不同 | 步骤 6 |
| `Artifact` 无 scope | 增加 `scope` / `source_task_id`，默认 `CURRENT`；非 CURRENT 必须带 `source_task_id` | `split_by_scope()` 必须同时切开产物。漏标来源会带着默认 CURRENT 被当成当前任务归档 | 步骤 7 |
| `list_artifact_index()` 无过滤 | 默认只列 `CURRENT`，`scope=None` 才全量 | 步骤 28 的 `artifacts: [索引]` 若直接调用，会把关联任务的表暴露给前端 | 步骤 7 |
| 存储异常未单列 | 新增 `StoreError` | 类型不匹配、键格式问题需要稳定 `code`，Redis 瞬时故障可按实例把 `retryable` 覆为 True | 步骤 8 |
| （未规定 `usage` / `LLMChunk` 形状） | `TokenUsage{prompt_tokens, completion_tokens, total_tokens}`；`LLMChunk{content, tool_call_deltas, finish_reason, model}` | 裸 dict 会让步骤 10/30/31 各写各的键。流式 `tool_calls` 是增量碎片，不能收成完整 `ToolCall` | 步骤 9 |
| SDK 默认 `max_retries=2` | `OpenAICompatClient` 把 SDK 重试钉死为 0 | 重试由步骤 10 的 `ResilientLLMClient` 独占；两层各自退避会让实际次数变成乘法 | 步骤 9 |
| `chat(request)` 与 `stream` 字段关系未写 | `chat()` 拒绝 `request.stream=True` | SDK 在 `stream=True` 时返回 iterator，当成 `LLMResponse` 用会立刻炸且错误难读 | 步骤 9 |
| （无） | HTTP 408 映射为 `LLMTimeoutError` | 兼容网关常用 408 表示超时，SDK 只把它收成普通 `APIStatusError`，不翻译就会被当成不可重试的请求拒绝 | 步骤 9 |
| （未规定 purpose 放哪） | `LLMRequest.purpose` 可选 `intent / plan / execute` | 路由必须跟着请求走，否则 `call_structured` 与阶段处理器要另开一套签名。显式 `model` 优先于 purpose | 步骤 10 |
| schema 未规定形态 | 接受 JSON Schema dict 或 `BaseModel`；对象级子集校验（required / type / enum / additionalProperties） | 不引入 `jsonschema` 依赖。嵌套结构留给 pydantic 模型；dict 路径只保证顶层字段不被脏数据写进 summary | 步骤 10 |
| 「主模型失败按 fallbacks 降级」 | 仅 `retryable=True` 耗尽后才降级；不可重试立即抛出 | 400 / 鉴权失败换端点通常同样失败，却会把重试次数与费用乘到链长。超时/限流才值得换模型 | 步骤 10 |
| purpose 路由写进整份 request 再共用 | `_bind_model` 按当前 profile 解析：`model_for(purpose)` 只覆盖主模型；备用端点用自己的 `profile.model` | 整条链钉死主模型名会变成「换了 base_url，还在要对方没有的名字」，降级等于没降 | 步骤 10 |
| 装饰器做「超时控制」 | 超时在内层 `OpenAICompatClient(timeout=profile.timeout)`，装饰器无总墙钟 | 各端点 timeout 不同；总墙钟会让备用端点还没调用就被判超时 | 步骤 10 |
| （流式失败如何降级未写清） | 只对首个 chunk 创建失败降级；流出后中途断流不换端点 | 已经吐出的 delta 无法撤回，换端点会让步骤 30 的 SSE 交错两路输出 | 步骤 10 |
| `build_llm_client` 一律包 Resilient | `LLM_USE_MOCK=true` 返回裸 Mock | 本地无密钥只需脚本回放；测健壮性须像单测那样手工再包一层 | 步骤 10 |
| Mock 异常可被重试多次（未写） | 异常实例消费后 `used` 不回滚 | `max_retries>=1` 却只 enqueue 一条超时时，第二次变成不可重试的「没有匹配」，降级链被截断。同客户端重试须用 callable 或每 attempt 一条 | 步骤 10 |
| `WorkingMemory` 只写三个 namespace | 另增 List 键 `wm_index/task_ids` 记录写入顺序 | `keys()` 在 Redis 下无序，`trim` / `get_summary_history` 必须有稳定顺序，不能靠遍历键 | 步骤 11 |

#### 取值域严格性（2026-08-30 定稿）

枚举**只接受规范拼写**。大小写变体、连字符与驼峰写法、同义词、笔误一律抛 `ValueError`，
不做归一化、不设别名表。理由：把错误拼写猜成正确成员会掩盖真正的缺陷——提示词写错、
前端传错字段、模型不遵守 schema——而猜测规则本身会长期膨胀且需要维护。

因此约束模型输出的正确位置在**产出端**而非解析端：用 `Cls.values()` 生成
JSON Schema 的 enum 约束（步骤 15 提示词装配、步骤 9 LLM 结构化输出）。
解析失败时 `from_str()` 抛出的 `ValueError` 已带合法取值列表，可直接回灌重试提示词。
仅当调用方明确需要兜底时才传 `default=`（如 `IntentType.from_str(raw, default=IntentType.UNKNOWN)`），
这是显式选择，与猜测拼写是两回事。

正确写法速查：`IntentType.NEW_QUERY` / `IntentType.ANALYSIS` / `IntentType.UNKNOWN`。

### 1.3 目标目录结构（最终形态）

```
query_agent/
├── agent/
│   ├── common/            # 步骤 3-4：枚举、异常、日志、ID
│   ├── config/            # 步骤 2：配置中心
│   ├── models/            # 步骤 5-7：纯数据模型（无外部依赖）
│   ├── store/             # 步骤 8：存储抽象 + 内存实现（步骤 27 加 Redis）
│   ├── llm/               # 步骤 9-10：LLM Layer
│   ├── memory_manage/     # 步骤 11-12：MemoryManager
│   ├── knowledge/         # 步骤 12：知识资源文件（prompt / skill / 字段字典）
│   ├── context_manage/    # 步骤 13-14：ContextManager
│   ├── prompt/            # 步骤 15：提示词装配
│   ├── tool_manage/       # 步骤 16-18：ToolManager + 工具实例
│   ├── task_manage/       # 步骤 19-24：状态机 + TaskManager + 各阶段处理器
│   └── runtime.py         # 步骤 25：依赖装配容器
├── app/                   # 步骤 26-28：Django 网络服务层
├── tests/                 # 贯穿全程
└── docs/
```

### 1.4 分层依赖方向（严格单向，不允许反向引用）

```
common / config
      ↓
   models
      ↓
   store
      ↓
 llm    memory    tool_manage        （三者互不依赖）
      ↓      ↓
   context_manage  ←  prompt
      ↓
   task_manage
      ↓
    runtime
      ↓
     app
```

> **本计划中所有步骤的依赖编号只会指向更小的编号**，任一步骤完成后其产出物均可独立单测，不需要等待后续步骤。

---

## 二、里程碑

| 里程碑 | 覆盖步骤 | 可验证成果 |
| --- | --- | --- |
| **M1 基础设施** | 1 – 8 | 配置/日志/异常/数据模型/存储抽象就绪，模型可序列化往返，Store 通过契约测试 |
| **M2 四大模块** | 9 – 18 | LLMLayer、MemoryManager、ContextManager、ToolManager 各自可独立跑通单测与 Demo 脚本 |
| **M3 单任务闭环** | 19 – 24 | 用 MockLLM + MockTool 在纯 Python 脚本中跑通 `created → completed` 全流程，含关联任务注入与写回 WorkingMemory |
| **M4 服务可用** | 25 – 28 | Django 同步 REST 接口可用，支持多轮会话、多任务关联，前端可直接对接 |
| **M5 流式与质量** | 29 – 32 | SSE 流式输出、任务取消、可观测性、测试覆盖率达标 |
| **M6 生产化** | 33 – 36 | Redis/DB 落地、真实 ES 接入、容器化部署与文档交付 |

---

## 阶段 M1：工程基础与数据模型（步骤 1 – 8）

### 步骤 1：项目骨架与依赖管理

- **依赖**：无
- **目标**：确定包结构与依赖清单，保证 `import agent` 可用、测试可跑。
- **交付物**：
  - 补全 `agent/` 及各子包的 `__init__.py`
  - `requirements.txt`：`django`、`openai`、`pydantic`、`pydantic-settings`、`python-dotenv`、`tenacity`、`elasticsearch`、`pandas`、`pytest`、`pytest-cov`、`ruff`、`mypy`（版本用安装时最新稳定版锁定）
  - `pyproject.toml`：ruff / mypy / pytest 配置（`pythonpath = ["."]`）
  - `.env.example`：列出全部环境变量键名（不含真实值）
  - `README.md`：项目简介、本地启动步骤、目录说明
- **验收**：`pip install -r requirements.txt` 成功；`pytest` 在零测试用例下正常退出；`ruff check .` 通过。

### 步骤 2：配置中心

- **依赖**：1
- **目标**：所有可变参数集中声明，禁止代码中散落魔法值。
- **交付物**：`agent/config/settings.py`
  - `LLMProfile`：`name / base_url / api_key / model / timeout / max_retries / temperature / max_tokens`
  - `LLMSettings`：`primary: LLMProfile`、`fallbacks: list[LLMProfile]`、`intent_model / plan_model / execute_model` 的按用途路由映射
  - `ContextSettings`：`max_total_tokens`、各部分预算占比、`history_summary_limit`、`related_task_content_limit`
  - `MemorySettings`：`session_ttl`、`working_memory_max_tasks`
  - `ToolSettings`：`default_timeout`、`max_result_rows`、`max_retry`
  - `ESSettings`：`hosts / username / password / default_index / verify_certs`
  - `AppSettings`：聚合上述所有项，`get_settings()` 带 `lru_cache` 单例
- **验收**：单测覆盖「缺失必填环境变量报明确错误」「默认值生效」「环境变量覆盖生效」三类场景。

### 步骤 3：异常体系与日志

- **依赖**：1, 2
- **目标**：统一错误分类（决定后续状态机走 `FAILED` 还是 `RETRYING`）与可追踪日志。
- **交付物**：
  - `agent/common/errors.py`：`AgentError`（基类，含 `code / message / retryable / detail`）→ `ConfigError`、`LLMError`（`LLMTimeoutError`、`LLMRateLimitError`、`LLMResponseFormatError`）、`ToolError`（`ToolNotFoundError`、`ToolInvocationError`、`ToolTimeoutError`）、`ContextError`、`MemoryError`、`TaskStateError`、`TaskCanceledError`
  - `agent/common/logging.py`：JSON 结构化日志；`contextvars` 承载 `trace_id / session_id / task_id`，自动注入每条日志；`setup_logging(level)`
  - `agent/common/ids.py`：`new_task_id()`、`new_session_id()`、`new_artifact_id()`、`new_trace_id()`（前缀 + 时间戳 + 短随机，便于排查）
- **验收**：单测断言异常 `retryable` 分类正确；日志输出包含上下文字段且为合法 JSON。

### 步骤 4：全局枚举与常量

- **依赖**：1
- **目标**：固化状态、意图、角色等取值域。
- **交付物**：`agent/common/enums.py`（`agent/task_manage/type.py` 保留并从此处重导出，避免破坏已有引用）
  - `TaskStatus`：`CREATED / INTENDING / PLANNING / EXECUTING / VALIDATING / COMPLETED / FAILED / CANCELED / WAITING_USER / RETRYING`
  - `IntentType`（原 `TaskType`）：`NEW_QUERY / ANALYSIS / EXPORT / CHAT / UNKNOWN`
  - `MessageRole`：`SYSTEM / USER / ASSISTANT / TOOL`
  - `ArtifactType`：`TABLE / SCALAR / CHART / FILE / TEXT`
  - `OperationStatus`：`PENDING / RUNNING / SUCCEEDED / FAILED / SKIPPED`
  - `ContextScope`：`CURRENT / RELATED / HISTORY`（用于窗口裁剪与写回过滤）
- **验收**：枚举可 `from_str` 容错解析（大小写、别名）；序列化为小写字符串并可往返。

### 步骤 5：消息与产物模型

- **依赖**：1, 4
- **目标**：定义 Context Window 三部分中的「content」与「artifacts」的原子结构。
- **交付物**：
  - `agent/models/message.py`：`Message{ message_id, role, content, name, tool_call_id, tool_calls, artifact_refs: list[str], scope: ContextScope, source_task_id, created_at, meta }`；`to_llm_dict()` 只输出 OpenAI 兼容字段
  - `agent/models/artifact.py`：`Artifact{ artifact_id, task_id, producer, artifact_type, title, schema, data, storage_ref, row_count, size_bytes, created_at, meta }` + `preview(max_rows)` 返回给 LLM 的摘要文本（表头 + 前 N 行 + 行数统计）
- **设计要点**：Artifact 的大体积 `data` 永不进入 `task_content`；LLM 只看到 `artifact_id` + `preview`，需要全量时由工具按 `artifact_id` 读取。
- **验收**：`Message.to_llm_dict()` 输出符合 OpenAI messages 规范；`Artifact.preview()` 在超大结果集下输出长度可控；JSON 序列化往返一致。

### 步骤 6：TaskSummary 与 Task 模型

- **依赖**：1, 4
- **目标**：落地需求文档中的 task summary 字段集与 Task 实体。
- **交付物**：
  - `agent/models/task_summary.py`：
    ```python
    TaskSummary{
        task_id: str
        content: str                      # 用户原始 query
        intent: IntentType | None
        related_task_ids: list[str]
        operations: list[Operation]        # Operation{index, name, tool, args, status, result_ref, error}
        result: str | None                 # 结构化结论
        status: TaskStatus
        output: str | None                 # 面向用户的最终输出 / 失败原因
    }
    ```
    附 `to_dict()` / `from_dict()` / `to_prompt_dict(fields)`（供意图识别阶段只投喂部分字段）
  - `agent/models/task.py`：补全现有骨架 `Task{ task_id, session_id, status, summary, created_at, updated_at, trace_id, error, retry_count, status_history }`；仅提供数据访问与 `touch()`，**状态转移逻辑不在此处**（属于步骤 19 状态机）
- **验收**：`TaskSummary` 序列化字段名与需求文档一致；`Task` 无任何对 manager 层的引用（用 import 检查测试守护）。

### 步骤 7：ContextWindow 容器

- **依赖**：1, 4, 5, 6
- **目标**：把三部分聚合为一个可整体读写、可整体归档的窗口对象（黑板载体）。
- **交付物**：`agent/models/context_window.py`
  ```python
  ContextWindow{
      task_id: str
      session_id: str
      summary: TaskSummary
      content: list[Message]
      artifacts: dict[str, Artifact]
  }
  ```
  方法（**纯数据操作，不含策略**）：`append_message`、`get_messages(scope=None, roles=None)`、`put_artifact`、`get_artifact`、`list_artifact_index()`、`split_by_scope()`、`to_dict()` / `from_dict()`
- **设计要点**：`split_by_scope()` 是任务完成写回 WorkingMemory 的基础——只写回 `scope == CURRENT` 的内容，注入的关联任务/历史片段不重复归档。
- **验收**：单测验证 append/查询/按 scope 拆分/序列化往返；验证注入片段不会被 `CURRENT` 过滤命中。

### 步骤 8：存储抽象与内存实现

- **依赖**：1, 3
- **目标**：为 Task / Context / Memory 提供统一持久化接口，隔离后续 Redis/DB 替换。
- **交付物**：
  - `agent/store/base.py`：`KVStore` 抽象（`get / set / delete / exists / keys(prefix) / expire`）、`ListStore` 抽象（`push / range / trim / length`），键名规范 `agent:{session_id}:{namespace}:{id}`
  - `agent/store/memory_store.py`：线程安全（`RLock`）内存实现，支持 TTL 惰性过期
  - `tests/store/test_store_contract.py`：**参数化契约测试**，任何新实现（Redis）直接复用该测试套件
- **验收**：契约测试全绿；并发读写压测无数据竞争；TTL 过期行为正确。

---

## 阶段 M2：四大核心模块（步骤 9 – 18）

### 步骤 9：LLM 客户端抽象与 OpenAI 兼容实现

- **依赖**：2, 3, 5
- **目标**：可靠地调用模型，且上层不感知具体供应商。
- **交付物**：`agent/llm/base.py`、`agent/llm/openai_client.py`
  - `LLMRequest{ messages, model, temperature, max_tokens, tools, response_format, stream }`
  - `LLMResponse{ content, tool_calls, finish_reason, usage, model, latency_ms, raw }`
  - `BaseLLMClient.chat(request) -> LLMResponse`、`stream_chat(request) -> Iterator[LLMChunk]`
  - `OpenAICompatClient`：基于 `openai` SDK 的兼容模式实现，统一把 SDK 异常翻译为步骤 3 的 `LLMError` 子类
- **验收**：以打桩 HTTP 层的单测验证请求体拼装、响应解析、异常映射；`stream_chat` 分块顺序正确。

### 步骤 10：LLM Layer 健壮性与结构化输出

- **依赖**：2, 3, 9
- **目标**：满足需求中「配置多模型保持健壮性」，并保证阶段处理器能拿到可靠的结构化结果。
- **交付物**：
  - `agent/llm/resilient.py`：`ResilientLLMClient` 装饰 `BaseLLMClient` —— 内层客户端按 `profile.timeout` 超时、指数退避重试（仅对 `retryable=True` 的异常）、主模型失败按 `fallbacks` 顺序降级。按用途路由只覆盖主模型（`_bind_model`：调用方未写死 `model` 时，index=0 用 `model_for(purpose)`，备用端点用各自的 `profile.model`）。调用埋点（模型、耗时、token、重试次数）。流式只对首个 chunk 创建失败降级。
  - `agent/llm/structured.py`：`call_structured(client, messages, schema, max_repair=1)` —— 优先用 `response_format=json_schema`，不支持时回退到「提示词约束 + JSON 提取 + 校验失败带错误信息重问一次」，最终失败抛 `LLMResponseFormatError`
  - `agent/llm/mock_client.py`：`MockLLMClient`，支持按调用序号/匹配规则返回预设响应，并记录收到的完整 messages（后续所有阶段测试的基石）。脚本化异常实例只能消费一次。
  - `agent/llm/factory.py`：`build_llm_client(settings)`。`LLM_USE_MOCK=true` 返回裸 Mock，不包 Resilient。
- **验收**：单测覆盖「主模型超时 → 降级成功」「非 retryable 错误不重试也不降级」「JSON 修复重问一次成功」「全部失败抛错」；MockLLMClient 可断言收到的 prompt 内容；purpose 不把主模型名钉到备用端点。

### 步骤 11：MemoryManager —— WorkingMemory

- **依赖**：4, 5, 6, 7, 8
- **目标**：按会话维护三部分历史记忆，供意图识别与关联任务注入使用。
- **交付物**：`agent/memory_manage/working_memory.py`
  ```python
  WorkingMemory(session_id, store)
    archive(window: ContextWindow)                  # 写入三部分（只接受 CURRENT scope 内容）
    get_summary_history(limit=None) -> list[dict]
    get_content_history(task_id) -> list[Message]
    get_artifact_history(task_id) -> list[Artifact]
    get_task_bundle(task_id) -> (summary, content, artifacts)
    list_task_ids() -> list[str]
    trim(max_tasks)                                 # 超限淘汰最旧任务
  ```
- **设计要点**：`task_summary_history` / `task_content_history` / `task_artifact_history` 三个独立命名空间，均按 `task_id` 索引且保持写入顺序；`get_task_bundle` 是步骤 22 关联任务注入的唯一数据入口。
- **验收**：单测验证归档-读取往返、按 task_id 精确检索、顺序稳定、`trim` 淘汰策略、会话隔离（不同 session 互不可见）。

### 步骤 12：MemoryManager —— KnowledgeMemory 与知识资源

- **依赖**：2, 3, 4
- **目标**：把「模型必须知道的固定知识」外置为可维护资源，按意图分发。
- **交付物**：
  - `agent/knowledge/` 资源目录：
    - `system_prompts/base.md`、`intent_recognition.md`、`plan_<intent>.md`、`execute.md`、`validate.md`
    - `skills/<intent>.yaml`：`{ name, description, system_prompt_ref, allowed_tools, field_dict_refs, few_shots, output_schema }`
    - `schemas/kibana_fields.yaml`：索引名、字段名、类型、含义、可选值、示例查询
  - `agent/memory_manage/knowledge_memory.py`
    ```python
    KnowledgeMemory(root_dir)
      get_system_prompt(stage, intent=None) -> str
      get_skill(intent) -> Skill
      get_field_dict(index=None) -> str
      get_few_shots(intent) -> list[Message]
      get_allowed_tools(intent) -> list[str]
      reload()                                     # 开发期热加载
    ```
  - `agent/memory_manage/memory_manager.py`：`MemoryManager` 门面，聚合 `working(session_id)` 与 `knowledge`，对上层提供单一入口
- **设计要点**：资源文件全部纯文本/YAML，改提示词不改代码；启动时做一次 schema 校验，缺失引用直接报错而非静默降级。
- **验收**：单测验证各意图 skill 完整加载、缺失文件报明确错误、字段字典渲染文本稳定；`reload()` 生效。

### 步骤 13：ContextManager —— 窗口生命周期

- **依赖**：4, 5, 6, 7, 8
- **目标**：管理 Context Window 的创建、读写、持久化，作为黑板的唯一写入通道。
- **交付物**：`agent/context_manage/context_manager.py`
  ```python
  ContextManager(store, settings)
    create_window(task_id, session_id, query, system_prompt) -> ContextWindow
    load_window(task_id) -> ContextWindow
    save_window(window)
    update_summary(task_id, **fields)               # 白名单字段校验
    append_message(task_id, message)
    add_artifact(task_id, artifact) -> str          # 返回 artifact_id，并自动向 content 追加引用+预览消息
    inject_segment(task_id, messages, artifacts, source_task_id, scope)
    snapshot(task_id) -> dict                       # 只读快照，供调试/SSE 事件
  ```
- **设计要点**：`update_summary` 对字段做白名单与类型校验，禁止 LLM 返回的脏字段污染 summary；`add_artifact` 强制「大数据入 artifacts、引用入 content」的不变式。
- **验收**：单测验证窗口创建后 `status=CREATED` 且 system 消息就位；非法字段更新被拒绝；`add_artifact` 后 content 中只出现引用与预览。

### 步骤 14：ContextManager —— 窗口装配与预算裁剪策略

- **依赖**：2, 4, 5, 6, 7, 13
- **目标**：按阶段与意图把三部分内容组装为 LLM 可消费的 messages，并保证不超预算。
- **交付物**：
  - `agent/context_manage/window_policy.py`：
    - 阶段视图定义
      | 阶段 | 投喂内容 |
      | --- | --- |
      | 意图识别 | 意图识别 system prompt + 当前 summary（仅 `content` / `status`）+ `task_summary_history`（最近 N 条，字段裁剪） |
      | 规划 | 意图专属 system prompt + skill + 字段字典 + 当前窗口（summary 全量 + content 摘要 + artifact 索引） |
      | 执行 | 执行 system prompt + 允许工具 schema + 当前 operation + 相关 artifact 预览 + 近 K 轮 content |
      | 校验 | 校验 system prompt + summary 全量 + content 摘要 + artifact 索引 + 原始 query |
    - `TokenBudget`：预算分配与超限裁剪，优先级 `当前任务 > 关联任务(RELATED) > 历史(HISTORY)`；同优先级内按时间由旧到新丢弃，工具错误消息优先保留
  - `agent/context_manage/serializer.py`：`build_messages(window, stage, intent, knowledge) -> list[dict]`
  - `agent/context_manage/token_counter.py`：可插拔计数（`tiktoken` 可用则精确，否则字符数估算）
- **验收**：单测验证四个阶段的 messages 结构与顺序符合预期；超预算时裁剪顺序符合优先级；裁剪后 messages 仍是合法对话序列（system 唯一且在首位、tool 消息必有对应 tool_call）。

### 步骤 15：PromptAssembler

- **依赖**：12, 14
- **目标**：把 KnowledgeMemory 的知识与 ContextManager 的窗口视图合成为最终 payload，成为阶段处理器的唯一 prompt 来源。
- **交付物**：`agent/prompt/assembler.py`
  ```python
  PromptAssembler(knowledge_memory, context_manager, settings)
    build_intent_prompt(window, summary_history) -> LLMRequest
    build_plan_prompt(window) -> LLMRequest
    build_execute_prompt(window, operation, tool_schemas) -> LLMRequest
    build_validate_prompt(window) -> LLMRequest
  ```
  各方法同时返回对应的 `output_schema`（供步骤 10 的结构化调用使用）。
- **验收**：快照测试（golden file）固定四类 prompt 的渲染结果，提示词变更时 diff 可见可评审。

### 步骤 16：ToolManager —— 工具抽象与注册表

- **依赖**：3, 4, 5
- **目标**：统一工具的声明、发现、参数校验与调用协议。
- **交付物**：
  - `agent/tool_manage/base.py`：
    ```python
    class BaseTool(ABC):
        name: str
        description: str
        args_schema: type[BaseModel]
        produces: ArtifactType | None
        timeout: float
        def run(self, args, ctx: ToolContext) -> ToolResult
        def to_openai_schema(self) -> dict
    ToolResult{ ok, artifact: Artifact | None, text: str, error: ToolError | None, elapsed_ms }
    ToolContext{ task_id, session_id, trace_id, artifacts_reader }   # 只读上下文，工具不直接写窗口
    ```
  - `agent/tool_manage/registry.py`：装饰器 `@register_tool` + 包内自动发现（`pkgutil` 扫描 `tool_manage/tools/`）+ 重名检测
- **设计要点**：工具**不允许**直接操作 ContextWindow，只返回 `ToolResult`，由执行阶段统一落盘，保证黑板写入单一入口。
- **验收**：单测验证注册/发现/重名报错/`to_openai_schema()` 合法（可被 OpenAI tools 字段接受）。

### 步骤 17：ToolManager —— 管理器与 Mock 工具

- **依赖**：2, 3, 5, 16
- **目标**：提供可直接被执行循环调用的工具管理器，并用 Mock 工具打通链路。
- **交付物**：
  - `agent/tool_manage/manager.py`
    ```python
    ToolManager(registry, settings)
      list_tools(intent=None) -> list[BaseTool]        # 按 skill 的 allowed_tools 过滤
      get_schemas(intent=None) -> list[dict]
      invoke(name, raw_args, ctx) -> ToolResult        # 参数校验 → 超时 → 异常封装 → 埋点
    ```
  - `agent/tool_manage/tools/mock_search_tool.py`、`mock_analysis_tool.py`：读取 `tests/fixtures/*.json`，产出真实结构的 `Artifact`
- **验收**：单测验证参数校验失败返回 `ok=False` 且携带可读错误（而非抛出中断任务）、超时被正确中断、意图白名单过滤生效。

### 步骤 18：真实工具实例

- **依赖**：2, 3, 5, 12, 16, 17
- **目标**：实现面向 Kibana/ES 的真实工具（此步只保证接口与本地可测，联调在步骤 35）。
- **交付物**：
  - `agent/tool_manage/tools/search_tool.py`：`SearchTool` —— 输入结构化查询条件（索引、时间范围、过滤、聚合、limit），基于字段字典校验字段合法性，输出 `TABLE` 类 Artifact；ES 客户端通过 `ESClient` 接口注入，便于替换为 Fake
  - `agent/tool_manage/tools/analysis_tool.py`：`AnalysisTool` —— 对已有 `artifact_id` 做聚合/排序/过滤/统计（pandas），输出新 Artifact，**不重复查询**
  - `agent/tool_manage/tools/export_tool.py`：`ExportTool` —— 把 Artifact 导出为 CSV/XLSX，返回 `FILE` 类 Artifact（含 `storage_ref`）
  - `agent/tool_manage/es_client.py`：`ESClient` 协议 + `FakeESClient`（本地测试）
- **验收**：单测以 `FakeESClient` 验证 DSL 拼装正确、非法字段被拦截、`max_result_rows` 截断生效；`AnalysisTool` 对空结果集与类型异常有明确错误；导出文件可被正常打开。

---

## 阶段 M3：状态机与任务编排（步骤 19 – 24）

### 步骤 19：任务状态机

- **依赖**：3, 4, 6
- **目标**：把状态转移规则从流程代码中独立出来，可穷举、可测试。
- **交付物**：`agent/task_manage/state_machine.py`
  - 合法转移表：
    ```
    CREATED      → INTENDING | CANCELED | FAILED
    INTENDING    → PLANNING | FAILED | CANCELED
    PLANNING     → EXECUTING | WAITING_USER | FAILED | CANCELED
    EXECUTING    → EXECUTING | VALIDATING | RETRYING | WAITING_USER | FAILED | CANCELED
    RETRYING     → EXECUTING | FAILED | CANCELED
    WAITING_USER → EXECUTING | PLANNING | CANCELED | FAILED
    VALIDATING   → COMPLETED | PLANNING | FAILED | CANCELED
    COMPLETED / FAILED / CANCELED → 终态
    ```
  - `TaskStateMachine.can(from, to)`、`assert_transition(from, to)`（非法转移抛 `TaskStateError`）、`is_terminal(status)`
  - 转移钩子 `on_enter(status)` / `on_exit(status)` 注册机制（供步骤 31 事件流复用）
- **设计要点**：`VALIDATING → PLANNING` 用于校验不通过的重规划，需带最大重规划次数上限，防止死循环。
- **验收**：穷举所有 `(from, to)` 组合的单测；非法转移全部报错；终态不可再转移。

### 步骤 20：阶段处理器基类

- **依赖**：10, 13, 14, 15, 17, 19
- **目标**：统一各阶段处理器的输入输出协议，让 TaskManager 只做调度。
- **交付物**：`agent/task_manage/stages/base.py`
  ```python
  class BaseStage(ABC):
      stage_name: str
      def run(self, task: Task, window: ContextWindow, deps: StageDeps) -> StageResult
  StageResult{ next_status, updated_fields, messages, artifacts, error, should_continue }
  StageDeps{ llm, context_manager, memory_manager, tool_manager, prompt_assembler, settings }
  ```
  基类统一处理：耗时埋点、异常兜底转 `StageResult(error=...)`、取消信号检查。
- **验收**：以一个哑实现验证异常被封装为 `StageResult` 而非向上抛出；取消信号能在阶段入口即时生效。

### 步骤 21：意图识别阶段（工作流第 1 步）

- **依赖**：11, 12, 15, 20
- **目标**：填充 `summary.intent` 与 `summary.related_task_ids`。
- **交付物**：`agent/task_manage/stages/intent.py`
  - 流程：置 `status=INTENDING` → 取 `WorkingMemory.get_summary_history(limit)` → `PromptAssembler.build_intent_prompt` → `call_structured(schema={intent, related_task_ids, reason, confidence})` → 校验 `intent` 在枚举内、`related_task_ids` 均存在于当前会话历史（不存在的静默丢弃并记警告）→ 写回 summary → `next_status=PLANNING`
  - 兜底：解析失败或置信度过低时 `intent=CHAT` 并记录降级原因（不直接失败）
- **验收**：MockLLM 驱动的单测覆盖「识别为各意图」「返回关联任务」「返回不存在的关联任务被过滤」「解析失败降级为 CHAT」；断言投喂的 messages 只含 `content`/`status` 两个 summary 字段 + 历史摘要。脚本化 `LLMTimeoutError` 只能消费一次，同客户端重试用 callable 或每 attempt 一条；`LLM_USE_MOCK=true` 时 factory 不包 Resilient，测健壮性须手工再包一层。

### 步骤 22：关联任务上下文注入

- **依赖**：7, 11, 13, 14, 21
- **目标**：实现「模拟当前任务已执行了一部分」的效果。
- **交付物**：`agent/context_manage/related_injector.py`
  ```python
  RelatedContextInjector(memory_manager, context_manager, settings)
    inject(task_id, related_task_ids) -> InjectionReport
  ```
  - 对每个 `related_task_id`：从 `WorkingMemory.get_task_bundle` 取三部分 → 生成一条分隔说明消息（`role=system`，标注来源 task_id 与其 summary 摘要）→ 追加裁剪后的历史 content（按 `related_task_content_limit`，保留 user/assistant 关键轮次与工具结论）→ 把历史 Artifact 以引用+预览形式并入当前窗口 artifacts（保留原 `artifact_id`，标记 `source_task_id`）
  - 所有注入内容标记 `scope=RELATED` + `source_task_id`
- **设计要点**：注入内容参与 LLM 推理但**不参与归档**（步骤 24 只写回 `scope=CURRENT`），从根本上避免会话轮次增加导致的记忆指数膨胀。
- **验收**：单测验证注入后窗口可被规划阶段直接使用；`split_by_scope()` 能完整剥离注入内容；多个关联任务注入顺序稳定且互不覆盖；不存在的 task_id 不导致异常。

### 步骤 23：规划阶段与执行循环（工作流第 2、3 步）

- **依赖**：14, 15, 17, 20, 22
- **目标**：产出 operations 并在黑板上逐步执行。
- **交付物**：
  - `agent/task_manage/stages/plan.py`：置 `status=PLANNING` → 注入关联任务上下文（若有）→ 按意图装配规划 prompt → 结构化输出 `operations: [{index, name, tool, args, expect}]` → 校验每个 `tool` 在意图白名单内、`args` 通过该工具的 schema 校验（失败则带错误重问一次）→ 写入 `summary.operations` → `next_status=EXECUTING`；`CHAT` 意图允许产出零工具步骤的直答计划
  - `agent/task_manage/stages/execute.py`：置 `status=EXECUTING` → 遍历 `operations`：装配执行 prompt → 模型决定调用工具或直接给出步骤结论 → `ToolManager.invoke` → 成功则 `add_artifact` + 追加 tool 消息 + `operation.status=SUCCEEDED/result_ref` → 失败则按 `retryable` 与 `max_retry` 决定 `RETRYING`（同步骤重试，args 可由模型修正）或整体 `FAILED` → 全部完成后补全 `summary.result` 与 `summary.output` → `next_status=VALIDATING`
  - 循环护栏：最大步骤数、最大总工具调用次数、单步最大重试次数、总时长上限，任一触顶转 `FAILED` 并写明原因
- **验收**：MockLLM + MockTool 单测覆盖「多步骤顺序执行」「某步工具失败后重试成功」「重试耗尽转 FAILED」「护栏触顶」「CHAT 零工具直答」；断言每步执行后窗口三部分状态正确。

### 步骤 24：校验阶段、终态处理与 TaskManager

- **依赖**：11, 13, 19, 20, 21, 22, 23
- **目标**：完成工作流第 4、5 步，串起端到端闭环。
- **交付物**：
  - `agent/task_manage/stages/validate.py`：置 `status=VALIDATING` → 装配校验 prompt（原始 query + summary 全量 + content 摘要 + artifact 索引）→ 结构化输出 `{ satisfied: bool, missing: list[str], suggestion: str, final_output: str }` → 满足则 `next_status=COMPLETED` 并落 `summary.output`；不满足且重规划次数未超上限则 `next_status=PLANNING`（把 `missing` 写入窗口作为补充要求）；超上限则 `FAILED`
  - `agent/task_manage/finalizer.py`：`finalize(task, window)` —— 终态统一收口：`COMPLETED` 时 `window.split_by_scope()` 取 `CURRENT` 三部分 → `WorkingMemory.archive` → `trim`；`FAILED/CANCELED` 时把错误原因写入 `summary.output`，并按配置决定是否归档（默认归档 summary，不归档 content）
  - `agent/task_manage/task_manager.py`
    ```python
    TaskManager(deps, state_machine, stages, finalizer)
      create_task(session_id, query) -> Task          # 建 Task + 建窗口 + 写 system prompt + status=CREATED
      run(task_id) -> Task                            # 驱动状态机直至终态
      step(task_id) -> Task                           # 单步推进（调试/流式用）
      cancel(task_id, reason)                          # 置取消标志，阶段边界生效
      get_task(task_id) / get_result(task_id)
    ```
    每次转移前 `assert_transition`，转移后写 `status_history` 并持久化
- **验收**：**端到端脚本测试**（`tests/e2e/test_single_task_flow.py`）——MockLLM + MockTool 下跑通 `CREATED → COMPLETED` 全流程；跑通「第二个任务关联第一个任务」场景并断言注入内容未被重复归档；跑通失败/取消分支；断言非法状态跳转被状态机拦截。

---

## 阶段 M4：网络服务层（步骤 25 – 28）

### 步骤 25：依赖装配容器

- **依赖**：2, 8, 10, 11, 12, 13, 14, 15, 17, 24
- **目标**：一处完成全部对象图构建，Web 层与脚本层共用同一入口。
- **交付物**：`agent/runtime.py`
  ```python
  class AgentRuntime:
      @classmethod
      def from_settings(cls, settings) -> "AgentRuntime"
      def handle_query(self, session_id, query) -> TaskResult      # 同步执行
      def get_task(self, task_id) -> Task
      def cancel_task(self, task_id, reason)
  get_runtime() -> AgentRuntime                                     # 进程级单例，线程安全懒加载
  ```
  同时提供 `scripts/run_cli.py`：命令行交互式多轮对话（不依赖 Django，便于快速验证）。
- **验收**：CLI 脚本可完成多轮对话，第二轮能命中关联任务；配置切换 MockLLM/真实 LLM 无需改业务代码。

### 步骤 26：Django 工程初始化

- **依赖**：1, 2, 3
- **目标**：搭好可运行的 Web 骨架（此步不含业务接口）。
- **交付物**：
  - `app/manage.py`、`app/server/{settings.py,urls.py,wsgi.py,asgi.py}`
  - Django settings 从 `agent.config.settings` 读取，禁止重复定义配置
  - 中间件：`TraceIdMiddleware`（生成/透传 `X-Trace-Id` 并注入日志 contextvar）、异常中间件（`AgentError` → 结构化 JSON 错误响应，含 `code/message/trace_id`）、CORS
  - `GET /api/health` 健康检查
- **验收**：`python app/manage.py runserver` 启动成功；健康检查返回 200；主动抛 `AgentError` 时返回统一错误结构且日志含 trace_id。

### 步骤 27：会话管理

- **依赖**：8, 11, 26
- **目标**：把 HTTP 请求映射到 Agent 的会话级记忆。
- **交付物**：`app/api/session.py`
  - `resolve_session(request) -> session_id`：请求体/Header 携带则复用，否则新建并在响应返回
  - `SessionRegistry`：会话元信息（创建时间、最后活跃、任务 id 列表）、TTL 过期清理、并发上限
  - 接口：`POST /api/sessions`（新建）、`GET /api/sessions/{id}`（会话内任务列表）、`DELETE /api/sessions/{id}`（清空该会话 WorkingMemory）
- **验收**：单测验证同一 session 的两次请求共享 WorkingMemory、不同 session 严格隔离、过期会话被清理。

### 步骤 28：同步查询接口

- **依赖**：25, 26, 27
- **目标**：交付前端可直接对接的主接口。
- **交付物**：`app/api/{views.py,serializers.py,urls.py}`
  - `POST /api/query`：入参 `{ session_id?, query }`；出参 `{ task_id, session_id, status, intent, related_task_ids, output, result, operations, artifacts: [索引], trace_id, elapsed_ms }`
  - `GET /api/tasks/{task_id}`：任务详情（含 summary 与 operations 进度）
  - `GET /api/tasks/{task_id}/artifacts/{artifact_id}`：分页拉取 Artifact 全量数据
  - 请求校验（query 非空、长度上限）、并发限流、统一响应包装 `{ code, message, data, trace_id }`
- **验收**：`tests/api/test_query_api.py` 用 Django test client 覆盖成功/参数错误/任务失败/关联任务多轮四类场景；接口契约写入 `docs/api.md`。

---

## 阶段 M5：流式、可观测与质量（步骤 29 – 32）

### 步骤 29：任务事件流

- **依赖**：19, 20, 24
- **目标**：为流式输出与可观测性提供统一事件源（先落事件，再落传输）。
- **交付物**：`agent/task_manage/events.py`
  - `TaskEvent{ event_type, task_id, stage, payload, timestamp }`，类型含 `task_created / status_changed / intent_resolved / plan_ready / operation_started / operation_finished / artifact_created / llm_delta / task_completed / task_failed`
  - `EventBus`：进程内发布订阅 + 每任务事件缓冲队列（支持断线重连后补发）
  - TaskManager / 各阶段通过步骤 19 的转移钩子与 `StageDeps` 发布事件
- **验收**：单测断言一次完整任务的事件序列完整、有序、无重复；无订阅者时不阻塞主流程。

### 步骤 30：SSE 流式接口与任务取消

- **依赖**：28, 29
- **目标**：交付渐进式输出与可中断能力。
- **交付物**：
  - `app/api/sse.py`：`GET /api/query/stream`（或 `POST /api/query` + `Accept: text/event-stream`）—— 后台线程/异步任务执行 `TaskManager.run`，主协程消费 `EventBus` 推送 SSE；心跳保活；客户端断开时触发 `cancel`
  - `POST /api/tasks/{task_id}/cancel`：写取消标志，阶段边界与工具调用超时点生效
  - 执行阶段接入 `llm.stream_chat`，把 `llm_delta` 作为事件透传。流式只对首个 chunk 创建失败降级，流出后中途断流不换端点（已经吐出的 delta 无法撤回，换端点会交错两路 SSE）
- **验收**：集成测试验证流式分块顺序与终止事件；取消请求后任务在 3 秒内进入 `CANCELED` 且不再产生新事件；客户端断开不留悬挂线程。中途断流按错误事件结束，不另开备用端点续流。

### 步骤 31：可观测性

- **依赖**：3, 10, 17, 29
- **目标**：线上问题可定位、成本可核算。
- **交付物**：
  - `agent/common/metrics.py`：计数/直方图抽象 + 内存实现（可导出 Prometheus 文本）；指标含任务数与终态分布、各阶段耗时、LLM 调用次数/token/重试/降级次数、工具调用次数与失败率
  - `agent/common/trace.py`：按 `trace_id` 落任务全量轨迹（每次 LLM 请求响应、每次工具调用）到本地 JSONL，可开关，敏感字段脱敏
  - `GET /api/metrics`、`GET /api/tasks/{task_id}/trace`（开发环境限定）
- **验收**：跑一次完整任务后指标数值与轨迹条目符合预期；关闭开关后无性能与磁盘开销。

### 步骤 32：测试体系与质量门禁

- **依赖**：1 – 31（各步骤单测已随步交付，本步做体系化收口）
- **目标**：把质量要求固化为可执行门禁。
- **交付物**：
  - `tests/` 分层：`unit/`（各模块）、`contract/`（Store、Tool、LLM 契约）、`e2e/`（MockLLM 全链路）、`api/`（HTTP）
  - `tests/fixtures/`：ES 响应样例、各意图 LLM 响应脚本、golden prompt 文件
  - `conftest.py`：`runtime_with_mocks`、`fake_es`、`frozen_clock` 等公共 fixture
  - CI 脚本：`ruff` + `mypy`（`agent/` 严格模式）+ `pytest --cov=agent --cov-fail-under=80`
  - 回归清单 `docs/test_matrix.md`：意图 × 状态路径 × 异常分支矩阵
- **验收**：CI 全绿；覆盖率 ≥ 80%；`docs/test_matrix.md` 中每个用例均有对应自动化测试。

---

## 阶段 M6：生产化（步骤 33 – 36）

### 步骤 33：Redis Store 与会话持久化

- **依赖**：8, 11, 13, 27, 32
- **目标**：多进程部署下会话与记忆共享。
- **交付物**：`agent/store/redis_store.py`（复用步骤 8 的契约测试）、连接池与序列化（JSON + 可选压缩）、键 TTL 策略、内存/Redis 配置切换、Redis 不可用时降级为内存并告警
- **验收**：契约测试在 Redis 后端全绿；多进程下同一 session 记忆一致；断连恢复后自动重连。

### 步骤 34：任务归档与数据库落库

- **依赖**：26, 33
- **目标**：历史任务可查询、可统计。
- **交付物**：Django models（`TaskRecord`、`ArtifactRecord`、`SessionRecord`）+ 迁移；`finalize` 时异步归档（失败不阻塞主流程）；`GET /api/history` 分页查询与过滤；Artifact 大数据落对象存储/本地文件并只在库中存引用
- **验收**：终态任务落库完整；历史查询接口分页正确；归档失败仅告警不影响用户响应。

### 步骤 35：真实 ES 联调与知识库补全

- **依赖**：12, 18, 32
- **目标**：把 Mock 换成真实数据源并保证效果。
- **交付物**：真实 `ESClient` 实现（连接池、超时、重试、慢查询日志）；`knowledge/schemas/kibana_fields.yaml` 依据真实索引补全（字段含义、枚举值、常用查询范式）；真实环境冒烟用例集；查询性能基线与 `max_result_rows` / 超时参数调优
- **验收**：冒烟用例集在真实环境全部通过；字段字典与真实 mapping 一致（提供比对脚本）；慢查询有日志与告警。

### 步骤 36：部署与交付文档

- **依赖**：26, 32, 33, 34, 35
- **目标**：可一键部署、可移交运维。
- **交付物**：`Dockerfile` + `docker-compose.yml`（app + redis）；`gunicorn`/`uvicorn` 启动配置与并发参数；`.env.example` 补齐全部生产变量；`docs/deploy.md`（部署、回滚、配置说明）、`docs/architecture.md`（最终架构图与数据流）、`docs/api.md`（接口契约）、`docs/operations.md`（常见故障排查手册）
- **验收**：干净环境按文档一键起服务并通过健康检查与冒烟用例；文档经一名未参与开发者走查可复现部署。

---

## 三、依赖关系速查

| 步骤 | 依赖步骤 | 步骤 | 依赖步骤 |
| --- | --- | --- | --- |
| 1 | — | 19 | 3, 4, 6 |
| 2 | 1 | 20 | 10, 13, 14, 15, 17, 19 |
| 3 | 1, 2 | 21 | 11, 12, 15, 20 |
| 4 | 1 | 22 | 7, 11, 13, 14, 21 |
| 5 | 1, 4 | 23 | 14, 15, 17, 20, 22 |
| 6 | 1, 4 | 24 | 11, 13, 19–23 |
| 7 | 1, 4, 5, 6 | 25 | 2, 8, 10–15, 17, 24 |
| 8 | 1, 3 | 26 | 1, 2, 3 |
| 9 | 2, 3, 5 | 27 | 8, 11, 26 |
| 10 | 2, 3, 9 | 28 | 25, 26, 27 |
| 11 | 4–8 | 29 | 19, 20, 24 |
| 12 | 2, 3, 4 | 30 | 28, 29 |
| 13 | 4–8 | 31 | 3, 10, 17, 29 |
| 14 | 2, 4–7, 13 | 32 | 1–31 |
| 15 | 12, 14 | 33 | 8, 11, 13, 27, 32 |
| 16 | 3, 4, 5 | 34 | 26, 33 |
| 17 | 2, 3, 5, 16 | 35 | 12, 18, 32 |
| 18 | 2, 3, 5, 12, 16, 17 | 36 | 26, 32–35 |

**无前向依赖已校验**：上表中每个步骤的依赖编号均严格小于自身编号。

### 可并行的工作组

- 步骤 5、6 可并行；步骤 9–10（LLM）、11–12（Memory）、16–17（Tool）三组在步骤 8 完成后可并行推进
- 步骤 26 只依赖 1–3，可与 M2 阶段并行开工，不必等 Agent 内核完成
- 步骤 18（真实工具）可与 M3 并行，因为 M3 用 Mock 工具即可闭环

---

## 四、关键设计不变式（评审与 Code Review 检查项）

1. **黑板单一写入通道**：任何模块修改 Context Window 必须经 `ContextManager`，工具与 LLM 层不得直接持有窗口写权限。
2. **大数据不入对话**：结果集/文件只进 `task_artifacts`，`task_content` 中仅有 `artifact_id` + 预览。
3. **注入不归档**：`scope=RELATED/HISTORY` 的内容参与推理但不写回 WorkingMemory，防止会话记忆指数膨胀。
4. **状态转移唯一裁判**：状态变更只能通过 `TaskStateMachine` 校验后由 `TaskManager` 执行，阶段处理器只返回期望的 `next_status`。
5. **LLM 输出必校验**：所有结构化输出经 schema 校验 + 字段白名单后才写入 summary。
6. **知识外置**：提示词、skill、字段字典一律以资源文件维护，改文案不改代码、不重新发版。
7. **失败可分类**：所有异常携带 `retryable` 语义，决定 `RETRYING` 与 `FAILED` 的分流。
8. **护栏必设**：步骤数、工具调用数、重试次数、重规划次数、总时长、token 预算六项上限缺一不可。

---

## 五、风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 关联任务注入导致上下文超限 | LLM 报错或成本飙升 | 步骤 14 的分优先级预算裁剪 + 步骤 22 的注入内容裁剪上限 |
| LLM 结构化输出不稳定 | 意图/规划阶段失败率高 | 步骤 10 的 json_schema + 一次修复重问 + 降级兜底策略 |
| 校验-重规划死循环 | 任务永不结束、成本失控 | 步骤 19/24 的最大重规划次数上限 |
| Kibana 字段字典与真实 mapping 漂移 | 生成非法查询 | 步骤 18 的字段合法性预校验 + 步骤 35 的 mapping 比对脚本 |
| 同步接口长耗时导致网关超时 | 用户体验差 | 步骤 30 的 SSE 流式 + 网关超时对齐配置 |
| 内存 Store 在多进程部署下不一致 | 会话记忆丢失 | 步骤 33 提前定义好抽象，切换 Redis 不改业务代码 |
