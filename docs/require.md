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

#### 4.ContextManager：
上下文管理模块。context window，指一次任务的全部工作空间。LLM在当前任务中的看到的内容就是context window中的内容。context window使用黑板模式维护两个记录：task summery和task content。其中，task summery是指一次任务的状态的总结字段，包括以下字段：
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
task content包含本次任务的全量记录，是一个包含System，User，Tool，Assist，等角色的聊天记录列表。
用户的每一次请求对应着不同的请求意图，不同意图的任务所携带的上下文是不一样的，ContextManager根据用户的请求意图以及任务执行过程中工具调用和模型推理的结果主动维护上下文。

#### 5.MemoryManager：
记忆管理模块。这个系统中设计的记忆包括WorkingMemory和KnowledgeMemory。其中，WorkingMemory指的是当前会话中的全部聊天记录。WorkingMemory包含两部分：task_summery_history和task_context_history。
其中，task_summery_history记录当前会话中所有任务的task summery dict，task_context_history记录当前会话中所有任务的task content记录。

KnowledgeMemory指的是外部提供的，模型为了完成任务，需要知道的知识。例如，在查询数据任务中，模型需要知道数据库有哪些字段，在chat任务中，模型需要知道自己的身份，功能等。KnowledgeMemory会以skill或者systemprompt的形式传递给模型。

#### 6.ToolManager
工具管理模块。这个模块负责工具的注册、工具发现、工具调用、工具配置等系统中与工具相关的功能。这个模块包含工具管理功能和工具实例。工具管理就是上述提到的工具管理的方法，工具实例指的是具体的工具，例如查询工具：SearchTool，数据分析工具AnalysisTool等

### 系统工作流
系统工作流程如下：

0.第零步是用户的请求通过前端调用发送到Agent系统，构建任务对象。将用户的请求构建为Task对象，并创建人物的上下文窗口task window，将任务的状态设置为created，并在task content的"system"角色中填入系统提示词。进入第一步

1.第一步首先会将systemprompt和当前query的task summery（只有content字段和status字段有内容，其中，content字段的内容是用户的请求原文，status字段的文字是内容是"intenting"，表示当前任务处于意图识别环节）以及历史的task_summery_history发送给模型，systemprompt引导模型根据这些信息对用户意图进行意图识别，填写task summery的intent和related id字段。在这一步中，模型识别出用户的意图，并根据用户的意图识别出是否存在关联任务，如果有关联任务，则将关联任务的id填写到related task字段中，进入第二步。

2.第二步中，系统首先修改任务状态为planning，然后根据用户意图组装不同的系统提示词，引导模型对任务进行规划，将组装的系统提示词与当前任务的context window传递给模型，模型在task summary的operations字段中记录任务步骤，进入第三步。

3.第三步为执行任务的循环。和传统的工作流执行流程一样，在这个循环中，Agent通过黑板模式在task window中的task content表中遍历执行任务的每个步骤，直到任务的每一个步骤完成之后，将任务的状态改为validating，在task summary中的result和output的内容补充完整，进入第四步。

4.第四步中，本次任务的所有task window的内容和task summary的内容，验证是否完成了用户提出的请求，如果完成了，修改任务状态为completed，并将task summary和task content的内容写入Working Memory中，然后将结果返回给用户。

5.在任意步骤中，如果任意步骤出现错误，或者被取消，任务状态都切换为faild或者canceled，并在output中填入错误原因。