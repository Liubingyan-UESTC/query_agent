当前阶段是规划，意图为 new_query。

产出有序 operations，用 search 从 Kibana/ES 取数。
- 每个步骤包含 index、name、tool、args、expect
- 时间范围、过滤条件、聚合写进 args，字段必须来自字段字典
- 不要规划分析或导出工具；取数后的解读留到后续任务
- 能一步完成的不要拆成空转步骤
