"""
Event type definitions and enums
"""
from enum import Enum


class EventScope(Enum):
    """Event scope types"""
    GLOBAL = "global"
    CONVERSATION = "conversation"
    PARTICIPANT = "participant"
    CONNECTION = "connection"
    PIPELINE = "pipeline"


class EventCategory(Enum):
    """Event category types"""
    SYSTEM = "system"
    LIFECYCLE = "lifecycle"
    TRANSCRIPTION = "transcription"
    AUDIO = "audio"
    TURN = "turn"
    METRICS = "metrics"
    ERROR = "error"
    STATE = "state"
    CONFIG = "config"


class EventAction(Enum):
    """Common event actions"""
    # Lifecycle
    CREATED = "created"
    STARTED = "started"
    ENDED = "ended"
    ERROR = "error"
    TIMEOUT = "timeout"
    
    # State changes
    CHANGED = "changed"
    UPDATED = "updated"
    
    # Transcription
    INTERIM = "interim"
    FINAL = "final"
    
    # Audio
    USER_STARTED = "user:started"
    USER_STOPPED = "user:stopped"
    ASSISTANT_STARTED = "assistant:started"
    ASSISTANT_STOPPED = "assistant:stopped"
    
    # Connection
    ESTABLISHED = "established"
    CLOSED = "closed"
    HEARTBEAT = "heartbeat"
    
    # Turn
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    
    # Metrics
    LATENCY = "latency"
    USAGE = "usage"
    PERFORMANCE = "performance"


def build_event_name(
    scope: EventScope,
    scope_id: str,
    category: EventCategory,
    action: EventAction
) -> str:
    """Build a properly formatted event name"""
    return f"{scope.value}:{scope_id}:{category.value}:{action.value}"