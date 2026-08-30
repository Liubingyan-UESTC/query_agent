"""知识资源包。

本包只承载资源文件，不含逻辑代码。之所以是包而非普通目录，是为了让
资源随 `agent` 一起分发（源码树、wheel、容器镜像路径一致）。

`KnowledgeMemory` 用包目录的文件系统路径定位资源，不在 import
`agent.memory_manage` 时加载本包，以免记忆层把资源包算进依赖图。

目录约定：
    system_prompts/  各阶段与各意图的系统提示词（Markdown）
    skills/          按意图组织的 skill 定义（YAML）
    schemas/         Kibana/ES 字段字典（YAML）
"""
