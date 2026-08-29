"""模型基类：统一序列化入口与校验严格度。

步骤 5–7 的五个模型都要在 Store 中序列化往返，`to_dict()` / `from_dict()` 若逐个手写
必然出现风格分叉。这里集中一次，子类只声明字段。

`extra="forbid"` 是刻意的：Store 反序列化与 HTTP 入参都可能带来未知字段，静默丢弃会
让「字段名写错」表现为「值莫名变成默认值」，排查成本极高。
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict

__all__ = ["AgentModel"]


class AgentModel(BaseModel):
    """全部数据模型的基类。"""

    model_config = ConfigDict(
        extra="forbid",
        # 赋值也走校验：ContextWindow 会就地修改模型，否则非法值只在下次序列化时才暴露
        validate_assignment=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """转为可直接 `json.dumps` 的字典。

        用 `mode="json"` 而非默认的 python 模式：datetime 会变成 ISO 字符串、
        枚举会变成其字符串值，否则调用方还得自备 JSON encoder。
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """由 `to_dict()` 的输出还原。字段缺失或多余都会抛 `ValidationError`。"""
        return cls.model_validate(payload)
