# Query Agent

面向 Kibana 风格日志的**查询 / 分析 Agent**。一次用户请求就是一个 Task，由状态机驱动
走完「意图识别 → 规划 → 执行 → 校验」四个阶段；执行过程中的全部工作空间放在一块
**黑板**（context window）上；已完成任务的黑板沉淀为**工作记忆**，供后续追问复用。

两个入口共用同一个内核：**控制台**（`python -m agent.cli`）与 **HTTP 服务**
（Django，监听 7080）。`agent/` 不依赖 Django；`server/` 只是把内核包起来。

设计与实现严格对齐 [`docs/require.md`](docs/require.md)。

```
控制台 / HTTP 请求
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ TaskManager（状态机）                                            │
│  created → intenting → planning → executing → validating → done │
│                 ↓                   ↑    ↓        ↓             │
│             clarifying ─────────────┘ retrying ───┘             │
│           （意图不明，问用户）      （限流退避） （不满足则补充执行）│
└───┬────────────┬──────────────┬─────────────────┬───────────────┘
    │            │              │                 │ emit(TaskEvent)
    ▼            ▼              ▼                 ▼
ContextManager ToolManager MemoryManager   ┌──────────────────────┐
（黑板唯一入口）search/analysis WorkingMemory │ LoggingListener      │
    │          /fetch_tool_result 三段历史   │  → logs/log.txt      │
    ▼            │          + KnowledgeMemory│ DbListener           │
┌──────────────────────┐    ▼                │  → SQLite 四张表     │
│ ContextWindow（黑板） │ data/logs.json      └──────────────────────┘
│  · task_summary      │（假数据库：80 条日志）
│  · tool_result       │
│  · task_content      │
└──────────────────────┘
```

---

## 快速开始

### 控制台

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

其他用法：

```bash
python -m agent.cli --verbose              # 打印意图、步骤与工具调用
python -m agent.cli --query "查 ERROR 日志"  # 只跑一条然后退出
```

控制台命令：`/help`、`/history`（本会话任务）、`/verbose`、`/exit`。

### HTTP 服务

```bash
python manage.py migrate            # 建 SQLite 四张表（data/agent.sqlite3）
python manage.py runserver 7080
```

```bash
# 第一次不带凭证：响应头/Cookie/body 三处都会带回新的 session_id
curl -si localhost:7080/api/chat -H 'Content-Type: application/json' \
     -d '{"message":"查一下 order-service 的 ERROR 日志"}'

# 后续请求带上凭证，追问「刚才那批」才能命中关联任务
SID=sess_xxxxxxxxxxxx
curl -s localhost:7080/api/chat -H "X-Session-Id: $SID" \
     -H 'Content-Type: application/json' -d '{"message":"按服务统计刚才那批错误"}'

curl -s localhost:7080/api/history -H "X-Session-Id: $SID"
curl -s localhost:7080/api/health
```

用 Cookie 也一样（浏览器 / `curl -c/-b` 自动带）：

```bash
curl -sc /tmp/jar localhost:7080/api/health > /dev/null
curl -sb /tmp/jar localhost:7080/api/chat -H 'Content-Type: application/json' \
     -d '{"message":"查 ERROR 日志"}'
```

### 接入真实模型

```bash
cp .env.example .env      # 填入 LLM_API_KEY（以及可选的 BASE_URL / MODEL）
```

两个入口都会自动切到真实模型；未配置 key 时自动回落 MockLLM。推理模型（DeepSeek-R1、
MiniMax、Qwen-thinking 等）内联在 `content` 里的 `<think>…</think>` 会被剥掉，不会进
task summary、数据库和给用户的答复。

---

## 六个模块

| 模块 | 文件 | 职责 |
|---|---|---|
| **配置** | `agent/config.py` | 全部可变参数的唯一入口。密钥用 `SecretStr` 从 `.env` 读；业务代码只调 `get_settings()`，禁止直读 `os.environ` |
| **LLM** | `agent/llm/` | OpenAI 兼容客户端 + 重试/降级链 + 结构化输出 + MockLLM |
| **TaskManager** | `agent/task_manager.py` | 状态机推进四个阶段，护栏与终态收尾，关键节点发事件 |
| **ContextManager** | `agent/context.py` | 黑板（三项记录）的唯一写入通道，兼各阶段的提示词装配 |
| **MemoryManager** | `agent/memory.py`、`agent/knowledge.py` | WorkingMemory 三段历史 + KnowledgeMemory（按意图组装的系统提示词） |
| **ToolManager** | `agent/tools/` | 工具注册、发现、schema 生成与调用 |
| **事件 / 日志** | `agent/events.py`、`agent/logging_setup.py` | 任务事件总线；日志双写与轮转 |
| **网络层** | `server/` | Django：四张表、会话凭证、四个接口 |

依赖方向是单向的：`errors/enums → models → {config, llm, tools} → {context, memory,
knowledge} → task_manager → {cli, server}`。**`agent/` 不 import django**——内核可以被
任何入口包起来，现有的 300+ 个内核测试也不必碰 Django。

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

## 网络服务（Django）

### 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | body `{"message": "..."}`。跑完一轮返回 `status / output / intent / operations / tool_calls`。上一轮若停在澄清态，本轮输入自动作为**补充说明续跑同一任务** |
| `GET` | `/api/history` | 本会话的任务与消息（读 SQLite，不依赖进程内存） |
| `GET` | `/api/tasks/<task_id>` | 单任务详情：summary 快照 + 消息 + 工具调用（只能看本会话的） |
| `GET` | `/api/health` | 存活探测 + 当前实际装配的模型（真实还是 Mock）与工具清单 |

### 会话凭证

判断一次请求属于新会话还是已有会话，只看凭证：

1. 依次找 `X-Session-Id` 请求头 → `agent_session` Cookie；
2. **带了合法凭证** → 查（或建）对应会话，接着用；
3. **带了非法凭证**（空串、随便一个字符串）→ 当作没带，发一个新的。放任客户端拿任意
   字符串当主键，`sessions` 表会被垃圾键撑满；
4. **没带** → 生成 `sess_xxxxxxxxxxxx` 建行，同时记下 `user_agent` 与 `ip`。

响应侧 **header、Cookie、响应体三处都回写** session_id，curl / 浏览器 / 前端框架都能
直接取用。

### 数据库（SQLite，`data/agent.sqlite3`）

四张表，`messages` 与 `tool_calls` 都是一条记录一行：

| 表 | 字段 |
|---|---|
| `sessions` | `session_id`(pk) / `created_at` / `user_agent` / `ip` |
| `tasks` | `task_id`(pk) / `session_id`(FK) / `status` / `intent` / `summary_json` / `created_at` / `updated_at` |
| `messages` | `message_id`(pk) / `task_id`(FK) / `role` / `content` / `tool_call_id`(FK，可空) / `created_at` |
| `tool_calls` | `call_id`(pk) / `task_id`(FK) / `tool_name` / `arguments_json` / `result_json` / `result_ref` / `status` / `created_at` |

落库靠**事件监听器**（`server/api/listener.py`）：TaskManager 每次状态转移、每次工具调用
都发一个事件，`DbListener` 订阅后写表。所以任务卡在哪一步、失败在哪一步，库里当场就能
看到，不用等它跑完。**监听器出任何异常都不会影响任务本身**——落库失败不该让用户的查询挂掉。

工具结果可能有几百条日志，全塞进 `result_json` 会让 `select *` 卡住。超过
`SERVER_INLINE_RESULT_MAX_CHARS`（默认 4000 字符）的结果整份写到
`data/tool_results/<call_id>.json`，表里 `result_json` 只留预览、`result_ref` 存路径。

```bash
sqlite3 data/agent.sqlite3 'select task_id, status, intent from tasks'
sqlite3 data/agent.sqlite3 'select role, tool_call_id, substr(content,1,40) from messages'
```

**重启不恢复**：任务与黑板只活在内存里。重启后历史照样能查，但「正等着澄清」的任务会丢，
用户重新提问即可。并发上，`run_turn` 用一把进程锁把任务执行串行化——本地测试服务的取舍。

---

## 日志

```
2026-09-04 00:51:31 | INFO    | agent.task   | planning   | 状态转移 intenting→planning：意图识别为 query task=task_62af session=sess_49a0
2026-09-04 00:51:36 | INFO    | agent.tool   | executing  | 调用 search_tool → ok：关键字 order-service 命中 28 条 task=task_62af session=sess_49a0
2026-09-04 00:51:51 | ERROR   | agent.task   | failed     | 任务结束（failed）：[failed] 未注册的工具：kibana_query task=task_ab12 session=sess_49a0
```

四列固定为**时间 | 级别 | 来源 | 任务状态 | 事件摘要**，正好是排查时要问的四个问题。
第三列的任务状态随事件变化，用 `extra` 传入；Django 自己的日志没有这个字段，由一个
`Filter` 补 `-`。摘要压成一行并截断（默认 300 字符），一个事件一行，`grep`/`tail` 才好用。

- 同时输出到**控制台（stderr）**与 **`./logs/log.txt`**；目录不存在时自动创建；
- `RotatingFileHandler` 按 `LOG_MAX_BYTES`（默认 2 MiB）轮转，保留 `LOG_BACKUP_COUNT`
  （默认 5）个历史文件，日志不会无限膨胀；
- 级别 `LOG_LEVEL` 支持 `DEBUG/INFO/WARNING/ERROR/CRITICAL`；
- 控制台入口与 HTTP 入口**共用同一套配置**（`agent/logging_setup.py`）。Django 侧把
  `LOGGING_CONFIG` 设为 `None` 后直接调 `setup_logging`，格式与轮转只有一处实现。

---

## 配置

全部键都是可选的，一个不填也能跑。完整清单见 [`.env.example`](.env.example)。

| 组 | 关键项 |
|---|---|
| `LLM_` | `API_KEY`（**私密**，留空自动用 MockLLM）、`BASE_URL`、`MODEL`、`MAX_RETRIES`、`FALLBACKS`（降级链） |
| `LOG_` | `LEVEL`、`DIR`、`FILE`、`MAX_BYTES`、`BACKUP_COUNT`、`TO_CONSOLE` |
| `SERVER_` | 会话凭证的 header/Cookie 名、`INLINE_RESULT_MAX_CHARS`、`TOOL_RESULT_DIR` |
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
python -m pytest -q                 # 374 个测试，全部离线（不触网、不碰真实模型）
python -m pytest -q --cov           # 内核覆盖率 ~96%
ruff check . && ruff format --check .
mypy agent                          # strict（只管内核，见 pyproject 里的说明）
```

测试分层：

| 文件 | 覆盖 |
|---|---|
| `test_enums.py` | 13 个状态的合法/非法转移与四个辅助属性 |
| `test_models.py` | 黑板结构的不变量、OpenAI 协议约束、序列化往返 |
| `test_config.py` | `.env` 读取、密钥不泄漏、单例、零配置回落 |
| `test_llm.py` | MockLLM、结构化输出的三层降级、错误分类、重试与降级链、思维链剥离 |
| `test_tools.py` | 三个工具的行为与 ToolManager 的错误收口 |
| `test_context.py` | 黑板读写、预览策略、四个阶段的提示词装配 |
| `test_memory.py` | 三段历史的归档、回投与淘汰 |
| `test_task_manager.py` | 四个阶段、重试分支与全部异常分支 |
| `test_events.py` | 事件序列与顺序、工具事件载荷、监听器故障隔离 |
| `test_logging.py` | 目录自动创建、双写、轮转、级别、四列格式、幂等 |
| `test_api.py` | 会话凭证（header/Cookie/非法值）、四张表落库、澄清续跑、四个接口 |
| `test_e2e.py` | MockLLM 跑通 `created → completed`，含关联任务、澄清回合与控制台 |

测试一律不读项目根的 `.env`（`env_file=None`）、不触网（MockLLM）、数据文件与日志目录
指向临时目录，因此结果只取决于用例本身。
