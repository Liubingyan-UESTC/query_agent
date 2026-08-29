from enum import Enum, auto


class TaskType(Enum):
    NEWQUERY = auto()
    ANALISIS = auto()
    EXPORT = auto()
    CHAT = auto()


class TaskStatus(Enum):
    CREATED = auto()
    INTENDING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    VALIDATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()
    WAITING_USER = auto()
    RETRYING = auto()
