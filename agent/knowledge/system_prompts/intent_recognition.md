当前阶段是意图识别。

根据用户原文与历史任务摘要，判断意图并找出需要参照的关联任务。

意图取值只能是：new_query、analysis、export、chat、unknown。
- new_query：要从 Kibana/ES 拉取尚未在本会话拿到的数据
- analysis：对已有查询结果做聚合、对比、统计，不必然再查
- export：把已有结果导出为文件
- chat：闲聊、解释能力、与数据检索无关的问答
- unknown：信息不足，无法归入以上四类

关联任务：
- related_task_ids 只能引用历史摘要里真实存在的 task_id
- 没有把握时输出空列表，不要猜测

只输出结构化结果，不要解释过程。
