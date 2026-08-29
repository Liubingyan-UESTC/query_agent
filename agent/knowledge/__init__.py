"""知识资源包。

本包只承载资源文件，不含逻辑代码；之所以是包而非普通目录，是为了让
`KnowledgeMemory` 能通过 `importlib.resources.files("agent.knowledge")`
定位资源，从而在打包分发（wheel / 容器镜像）后依然可读。

目录约定（步骤 12 落地）：
    system_prompts/  各阶段与各意图的系统提示词（Markdown）
    skills/          按意图组织的 skill 定义（YAML）
    schemas/         Kibana/ES 字段字典（YAML）
"""
