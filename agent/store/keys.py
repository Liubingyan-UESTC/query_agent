"""存储键名规范。

格式为 `agent:{session_id}:{namespace}:{id}`。冒号是唯一分隔符，任一段含冒号
都会让 `keys(prefix)` 的切分语义漂移，故在构造时拒绝。
"""

__all__ = ["KEY_ROOT", "make_key"]

KEY_ROOT = "agent"


def make_key(session_id: str, namespace: str, entity_id: str) -> str:
    """拼出规范键。各段不能为空、不能含冒号。"""
    parts = {
        "session_id": session_id,
        "namespace": namespace,
        "id": entity_id,
    }
    for name, part in parts.items():
        if not part or ":" in part:
            raise ValueError(f"存储键的 {name} 不能为空或含冒号，实际为 {part!r}")
    return f"{KEY_ROOT}:{session_id}:{namespace}:{entity_id}"
