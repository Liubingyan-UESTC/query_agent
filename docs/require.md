### 目标
我现在正在重构这个agent系统，用于对v0版本的工程化落地。我把v1的仓库放在QueryAgent/v1下。系统的架构如下：

### 系统架构
这个Agent系统主要是用于查询kibana数据库中的数据，或者对查询到的数据进行分析和统计操作。整个系统由以下几个模块组成：
#### 1.网络服务模块
使用Django框架搭建网络服务，提供用户请求接口接入后端Agent系统
#### 2.LLM Layer模块，
模型模块，使用OpenAI API兼容模式的LLM 客户端访问模型，需要配置模型以保持系统的健壮性。

#### 3. TaskManager：
任务管理模块。用户的一次请求叫做一个Task，TaskManger使用状态机的方式管理任务的执行。Task有属性TaskStatus，TaskStatus之间可以执行状态转移，状态转移的实际和方向由TaskManager管理。一般情况下转移的流程为Created->Intenting->Planning->Executing->Validating->Completed。
特殊情况下会转移到特殊分支，例如Failed，Canceled。
TaskManager跟踪任务状态，引导任务执行。
任务的状态如下：
```python
from __future__ import annotations

from enum import Enum
from typing import Set


class TaskStatus(str, Enum):
    """
    任务状态枚举。

    所有状态按职能分为五类：生命周期主线、异步等待、人工介入、异常重试、终态。
    """

    # ==================== 一、核心生命周期（主线流程） ====================
    CREATED = "created"
    """任务已创建，上下文已初始化，等待进入意图识别。"""

    INTENTING = "intenting"
    """正在进行意图识别（LLM 解析用户输入）。"""

    PLANNING = "planning"
    """正在进行任务规划（生成 operations 步骤列表）。"""

    EXECUTING = "executing"
    """正在同步执行操作步骤（含工具调用、循环等）。"""

    VALIDATING = "validating"
    """正在验证执行结果是否满足用户需求。"""

    # ==================== 二、异步与长时操作（解耦阻塞） ====================
    WAITING = "waiting"
    """任务正在等待外部异步操作完成（如后台导出、Kibana 慢查询回调）。此时系统已释放线程，不占用 Agent 资源。"""

    PAUSED = "paused"
    """任务被人为暂停（如管理员审核、用户临时中断），恢复后可继续流转。"""

    # ==================== 三、人工介入（不确定场景兜底） ====================
    AWAITING_APPROVAL = "awaiting_approval"
    """任务在执行敏感操作前（如导出大量数据、删除索引），需等待用户二次确认。"""

    CLARIFYING = "clarifying"
    """任务因意图不明确（Unknown）进入人工澄清环节，等待用户补充查询条件。"""

    # ==================== 四、异常与重试（防死锁） ====================
    RETRYING = "retrying"
    """任务因临时错误（如 LLM API 429 限流、网络抖动）进入重试队列，等待指数退避后再次执行。"""

    # ==================== 五、终态（不可转移） ====================
    COMPLETED = "completed"
    """任务成功完成，结果已返回用户且已归档。"""

    CANCELED = "canceled"
    """用户主动取消任务。"""

    FAILED = "failed"
    """任务因不可恢复的错误（如模型持续输出格式错误、工具权限不足）而终止，不再自动重试。"""

    ABORTED = "aborted"
    """任务因系统级熔断（如全局超时、资源耗尽）被强制终止，区别于用户主动取消。"""

    # ==================== 辅助属性（便于状态机逻辑判断） ====================

    @property
    def is_terminal(self) -> bool:
        """是否为终态（任务生命周期已结束）。"""
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELED,
            TaskStatus.FAILED,
            TaskStatus.ABORTED,
        }

    @property
    def is_async_waiting(self) -> bool:
        """是否需要释放线程等待外部回调。"""
        return self in {
            TaskStatus.WAITING,
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.CLARIFYING,
        }

    @property
    def is_retryable(self) -> bool:
        """是否为可重试的临时异常状态（TaskManager 需配合退避策略）。"""
        return self == TaskStatus.RETRYING

    @property
    def requires_user_input(self) -> bool:
        """是否需要用户提供额外输入才能继续。"""
        return self in {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.CLARIFYING,
        }

    # ==================== 状态转移合法性校验 ====================

    def can_transition_to(self, target: TaskStatus) -> bool:
        """
        判断当前状态是否允许转移到目标状态。

        严格遵循状态机流转规则，非法转移将抛出 ValueError。
        """
        # 定义合法的转移映射表（当前状态 -> 允许的下一个状态集合）
        transitions: dict[TaskStatus, Set[TaskStatus]] = {
            TaskStatus.CREATED: {TaskStatus.INTENTING, TaskStatus.CANCELED},
            TaskStatus.INTENTING: {
                TaskStatus.PLANNING,
                TaskStatus.CLARIFYING,  # 意图不明，转人工澄清
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
            },
            TaskStatus.PLANNING: {
                TaskStatus.EXECUTING,
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
            },
            TaskStatus.EXECUTING: {
                TaskStatus.WAITING,  # 触发异步操作
                TaskStatus.VALIDATING,  # 所有步骤同步执行完成
                TaskStatus.RETRYING,  # 遇到临时错误
                TaskStatus.AWAITING_APPROVAL,  # 遇到敏感操作
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,  # 系统熔断
            },
            TaskStatus.WAITING: {
                TaskStatus.EXECUTING,  # 异步回调完成，继续执行
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            },
            TaskStatus.RETRYING: {
                TaskStatus.EXECUTING,  # 重试成功
                TaskStatus.FAILED,  # 重试次数耗尽
                TaskStatus.CANCELED,
                TaskStatus.ABORTED,
            },
            TaskStatus.AWAITING_APPROVAL: {
                TaskStatus.EXECUTING,  # 用户确认
                TaskStatus.CANCELED,  # 用户拒绝
                TaskStatus.FAILED,
            },
            TaskStatus.CLARIFYING: {
                TaskStatus.INTENTING,  # 用户补充信息后重新意图识别
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
            },
            TaskStatus.PAUSED: {
                TaskStatus.EXECUTING,  # 恢复执行
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            },
            TaskStatus.VALIDATING: {
                TaskStatus.COMPLETED,  # 验证通过
                TaskStatus.EXECUTING,  # 验证不通过，补充执行
                TaskStatus.CANCELED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            },
            # 终态不可再转移
            TaskStatus.COMPLETED: set(),
            TaskStatus.CANCELED: set(),
            TaskStatus.FAILED: set(),
            TaskStatus.ABORTED: set(),
        }
        allowed = transitions.get(self, set())
        return target in allowed

    def transition_to(self, target: TaskStatus, task_id: str) -> TaskStatus:
        """
        执行状态转移，若非法则抛出异常（配合乐观锁使用）。
        """
        if not self.can_transition_to(target):
            raise ValueError(f"Task {task_id} 非法状态转移: {self.value} -> {target.value}")
        return target
```
每完成一个阶段，TaskManager将任务状态更新为新的状态，当任务状态为终点状态时，这次任务结束，将结果返回给用户。
同时，将任务的Context window中的三项分别写入WorkingMemory中的对应字段。

#### 4.ContextManager：
上下文管理模块。context window，指一次任务的全部工作空间。LLM在当前任务中的看到的内容就是context window中的内容。context window使用黑板模式维护三个记录：task summery、tool result和task content。
其中，task summery是指一次任务的总结字段，包括以下字段：
```
{
    "task_id",      # 任务id
    "content",      # 用户传入的请求query
    "intent",       # 本次任务的意图
    "related task", # 本次任务的关联任务id的列表
    "operations",   # 本次任务的操作步骤的列表
    "result",       # 本次任务的结果
    "status",       # 本次任务的当前状态
    "output",       # 本次任务的最终输出
}
```
tool result是该任务中所有工具调用的结果记录，是一个包含工具调用id和工具调用结果的字典，包括以下字段：
```
{
    "tool_call_id": {
        "tool_name": "xxx",
        "tool_result": "xxx"
    }
}
```
工具调用结果存储在tool_result中，并返回给模型tool_call_id，同时，由ToolManager提供一个特殊工具，通过tool_call_id获取结果，这个特殊工具的返回结果不存入tool_result中，而是直接返回给模型。这样，模型可以获取到工具调用的原始结果，同时避免了tool_result中存储大量的工具调用结果，导致context window过大，影响模型的推理能力。

task content包含本次任务的全量记录，是一个包含System，User，Tool，Assist，等角色的聊天记录列表。
用户的每一次请求对应着不同的请求意图，不同意图的任务所携带的上下文是不一样的，ContextManager根据用户的请求意图以及任务执行过程中工具调用和模型推理的结果主动维护上下文。


#### 5.MemoryManager：
记忆管理模块。这个系统中设计的记忆包括WorkingMemory和KnowledgeMemory。其中，WorkingMemory指的是当前会话中的全部聊天记录。WorkingMemory包含三部分：task_summery_history，tool_result_history和task_context_history。
其中，
task_summery_history记录当前会话中所有任务的task summery dict，
tool_result_history记录当前会话中所有任务的tool result记录，
task_context_history记录当前会话中所有任务的task content记录。

WorkingMemory实际上相当于历史任务的上下文窗口的集合。在每次任务的意图识别阶段，系统会将WorkingMemory中task_summery_history的内容传递给模型，模型根据这些历史任务的记录，识别出当前任务的意图，并识别出与当前任务相关的历史任务，关联任务的id会被写入当前任务的task summery的related task字段中。在任务的后续执行过程中，相关的任务的task_context_history的内容会被传递给模型，模型根据这些历史任务的上下文，辅助当前任务的执行。

KnowledgeMemory指的是外部提供的，模型为了完成任务，需要知道的知识。例如，在查询数据任务中，模型需要知道kibana数据库有哪些字段，在chat任务中，模型需要知道自己的身份，功能等。KnowledgeMemory会以skill或者systemprompt的形式传递给模型，例如在执行阶段，系统会针对不同的任务意图，封装不同的系统提示词，这些提示词就是KnowledgeMemory的一部分

#### 6.ToolManager
工具管理模块。这个模块负责工具的注册、工具发现、工具调用、工具配置等系统中与工具相关的功能。这个模块包含工具管理功能和工具实例。工具管理就是上述提到的工具管理的方法，工具实例指的是具体的工具，例如查询工具：SearchTool，数据分析工具AnalysisTool等

### 系统工作流
系统工作流程如下：

0.第零步是用户的请求通过前端调用发送到Agent系统，构建任务对象。将用户的请求构建为Task对象，并创建人物的上下文窗口task window，将任务的状态设置为created，并在task content的"system"角色中填入系统提示词。进入第一步

1.第一步首先会将systemprompt和当前query的task summery（只有content字段和status字段有内容，其中，content字段的内容是用户的请求原文，status字段的文字是内容是"intenting"，表示当前任务处于意图识别环节）以及历史的task_summery_history发送给模型，systemprompt引导模型根据这些信息对用户意图进行意图识别，填写task summery的intent和related id字段。在这一步中，模型识别出用户的意图，并根据用户的意图识别出是否存在关联任务，如果有关联任务，则将关联任务的id填写到related task字段中，进入第二步。

2.第二步中，系统首先修改任务状态为planning，然后根据用户意图组装不同的系统提示词，引导模型对任务进行规划，将组装的系统提示词与当前任务的context window传递给模型，模型在task summary的operations字段中记录任务步骤，进入第三步。

3.第三步为执行任务的循环。和传统的工作流执行流程一样，在这个循环中，Agent通过黑板模式在task window中的task content表中遍历执行任务的每个步骤，直到任务的每一个步骤完成之后，将任务的状态改为validating，在task summary中的result和output的内容补充完整，进入第四步。

4.第四步中，本次任务的所有task window的内容和task summary的内容，验证是否完成了用户提出的请求，如果完成了，修改任务状态为completed，并将task summary和task content的内容写入Working Memory中，然后将结果返回给用户。

5.在任意步骤中，如果任意步骤出现错误，或者被取消，任务状态都切换为对应的终点状态，并在output中填入错误原因。



这应该是一个比较清晰和简单的系统架构和工作流。