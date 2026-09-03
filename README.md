# Query Agent

面向 Kibana 风格日志的**查询 / 分析 Agent**。一次用户请求就是一个 Task，由状态机驱动
走完「意图识别 → 规划 → 执行 → 校验」四个阶段；执行过程中的全部工作空间放在一块
**黑板**（context window）上；已完成任务的黑板沉淀为**工作记忆**，供后续追问复用。

设计与实现严格对齐 [`docs/require.md`](docs/require.md)。

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ TaskManager（状态机）                                            │
│  created → intenting → planning → executing → validating → done │
│                 ↓                   ↑    ↓        ↓             │
│             clarifying ─────────────┘ retrying ───┘             │
│           （意图不明，问用户）      （限流退避） （不满足则补充执行）│
└───────┬───────────────────┬────────────────────┬────────────────┘
        │                   │                    │
        ▼                   ▼                    ▼
  ContextManager        ToolManager         MemoryManager
  （黑板唯一写入口）      search / analysis    WorkingMemory 三段历史
        │               / fetch_tool_result   + KnowledgeMemory（提示词）
        ▼                     │
┌──────────────────────┐      ▼
│ ContextWindow（黑板） │   data/logs.json
│  · task_summary      │   （假数据库：80 条 Kibana 风格日志）
│  · tool_result       │
│  · task_content      │
└──────────────────────┘
```

---

## 快速开始

```bash
pip install -r requirements.txt
python -m agent.cli                 # 零配置即可运行（自动使用 MockLLM）
```

```
你 > 查一下 order-service 的错误日志
✅ 关键字 ERROR 命中 19 条，返回 19 条

你 > 按服务统计刚才那批错误
✅ 按 service 统计 count（共 19 条）：order-service=12，payment-service=5，gateway=1，user-service=1
```

接入真实模型：

```bash
cp .env.example .env      # 填入 LLM_API_KEY（以及可选的 BASE_URL / MODEL）
python -m agent.cli
```

其他用法：

```bash
python -m agent.cli --verbose              # 打印意图、步骤与工具调用
python -m agent.cli --query "查 ERROR 日志"  # 只跑一条然后退出
```

控制台命令：`/help`、`/history`（本会话任务）、`/verbose`、`/exit`。

---

## 六个模块

| 模块 | 文件 | 职责 |
|---|---|---|
| **配置** | `agent/config.py` | 全部可变参数的唯一入口。密钥用 `SecretStr` 从 `.env` 读；业务代码只调 `get_settings()`，禁止直读 `os.environ` |
| **LLM** | `agent/llm/` | OpenAI 兼容客户端 + 重试/降级链 + 结构化输出 + MockLLM |
| **TaskManager** | `agent/task_manager.py` | 状态机推进四个阶段，护栏与终态收尾 |
| **ContextManager** | `agent/context.py` | 黑板（三项记录）的唯一写入通道，兼各阶段的提示词装配 |
| **MemoryManager** | `agent/memory.py`、`agent/knowledge.py` | WorkingMemory 三段历史 + KnowledgeMemory（按意图组装的系统提示词） |
| **ToolManager** | `agent/tools/` | 工具注册、发现、schema 生成与调用 |

依赖方向是单向的：`errors/enums → models → {config, llm, tools} → {context, memory,
knowledge} → task_manager → cli`。

### 黑板：一次任务的全部工作空间

`ContextWindow` 恰好三项记录，与 require.md 一一对应：

| 记录 | 结构 | 说明 |
|---|---|---|
| `summary` | 8 个字段 | `task_id / content / intent / related_task_ids / operations / result / status / output` |
| `tool_results` | `{tool_call_id: {tool_name, tool_result, ok}}` | 工具结果的**全量**存放处 |
| `content` | `list[Message]` | system / user / assistant / tool 四种角色的完整对话 |

**大结果不进上下文**：工具返回的全量数据写进 `tool_results`，回给模型的只是
「一句概括 + 前 3 条样本 + tool_call_id」。模型需要完整数据时自己调
`fetch_tool_result(tool_call_id)`——这个特殊工具的返回值直接进对话，**不再写回黑板**，
否则同一份数据会在窗口里存两遍。

### 工具

| 工具 | 参数 | 说明 |
|---|---|---|
| `search_tool` | `keyword`、`limit` | 关键字在日志所有字段上做大小写不敏感子串匹配；空串返回全部 |
| `analysis_tool` | `tool_call_id`、`group_by`、`metric`、`field`、`top` | 读黑板上某次检索的结果做分组统计（count / sum / avg / max / min） |
| `fetch_tool_result` | `tool_call_id` | 取回某次调用的全量结果（内部工具，不落黑板） |

数据源是 `data/logs.json`（80 条 Kibana 风格日志，字段见 `agent/knowledge.py` 的
`KIBANA_FIELDS`）。换成真实 ES 时，只需替换 `SearchTool.run` 的取数实现，其余不动。

### 记忆

- **WorkingMemory**（会话级）：任务进终态时整窗归档进三段历史
  `task_summary_history / tool_result_history / task_context_history`。
  意图识别阶段回投摘要历史，模型据此判断「这次请求和哪个历史任务相关」；
  规划阶段再按 `related_task_ids` 注入那些任务的对话。
- **KnowledgeMemory**：按「阶段 × 意图」装配系统提示词。查询/统计类会带上日志字段说明，
  闲聊类不带——省 token，也避免模型跑题。

---

## 配置

全部键都是可选的，一个不填也能跑。完整清单见 [`.env.example`](.env.example)。

| 组 | 关键项 |
|---|---|
| `LLM_` | `API_KEY`（**私密**，留空自动用 MockLLM）、`BASE_URL`、`MODEL`、`MAX_RETRIES`、`FALLBACKS`（降级链） |
| `CONTEXT_` | 历史摘要条数、关联任务注入条数、近 K 条对话 |
| `MEMORY_` | `MAX_TASKS`（工作记忆保留的任务数） |
| `TOOL_` | `DATA_FILE`、`MAX_ROWS`、预览行数与字符上限 |
| `TASK_` | `MAX_STEPS`、`MAX_TOOL_CALLS`、`MAX_ROUNDS_PER_STEP`、`MAX_REVALIDATE`、`MAX_RETRY` |

### 出错时的三条去向

- `RETRYING`——执行阶段遇到可重试的临时错误（限流、网络抖动）：指数退避后重跑当前
  步骤，已完成的步骤不会重来；重试次数用尽转 `FAILED`；
- `FAILED`——不可恢复：模型反复给不出合法 JSON、调用了未注册的工具；
- `ABORTED`——系统熔断：任一护栏耗尽（步骤数、工具调用数、单步轮数、重校验次数）。

后两者都会把原因写进 `summary.output`，控制台原样展示。
（注：`planning` 阶段的熔断落 `FAILED`——require.md 的转移表里 planning 没有到
aborted 的边，实现宁可换个语义相近的终态，也不绕过状态机。同理，只有 `executing`
有到 `retrying` 的边，所以其他阶段的临时错误直接判 `FAILED`。）

---

## 开发

```bash
python -m pytest -q                 # 281 个测试，全部离线
python -m pytest -q --cov           # 覆盖率 ~96%
ruff check . && ruff format --check .
mypy agent                          # strict
```

测试分层：

| 文件 | 覆盖 |
|---|---|
| `test_enums.py` | 13 个状态的合法/非法转移与四个辅助属性 |
| `test_models.py` | 黑板结构的不变量、OpenAI 协议约束、序列化往返 |
| `test_config.py` | `.env` 读取、密钥不泄漏、单例、零配置回落 |
| `test_llm.py` | MockLLM、结构化输出的三层降级、错误分类、重试与降级链 |
| `test_tools.py` | 三个工具的行为与 ToolManager 的错误收口 |
| `test_context.py` | 黑板读写、预览策略、四个阶段的提示词装配 |
| `test_memory.py` | 三段历史的归档、回投与淘汰 |
| `test_task_manager.py` | 四个阶段、重试分支与全部异常分支 |
| `test_e2e.py` | MockLLM 跑通 `created → completed`，含关联任务、澄清回合与控制台 |

测试一律不读项目根的 `.env`（`env_file=None`）、不触网（MockLLM）、数据文件指向
临时目录，因此结果只取决于用例本身。
