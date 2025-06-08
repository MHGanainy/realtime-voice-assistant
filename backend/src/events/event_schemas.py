"""
Pydantic models for event payloads
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Base event model"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: Optional[str] = None
    _event_name: Optional[str] = None
    
    class Config:
        extra = "allow"


class ConversationEvent(BaseEvent):
    """Conversation lifecycle event"""
    conversation_id: str
    session_id: Optional[str] = None
    participant_id: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TranscriptionEvent(BaseEvent):
    """Transcription event"""
    conversation_id: str
    text: str
    speaker: str  # "participant" or "assistant"
    is_final: bool = True
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration_ms: Optional[int] = None


class AudioEvent(BaseEvent):
    """Audio flow event"""
    conversation_id: str
    direction: str  # "inbound" or "outbound"
    speaker: Optional[str] = None  # "participant" or "assistant"
    duration_ms: Optional[int] = None
    
    
class TurnEvent(BaseEvent):
    """Conversation turn event"""
    conversation_id: str
    turn_id: str
    speaker: str
    text: str
    duration_ms: Optional[int] = None
    was_interrupted: bool = False
    confidence: Optional[float] = None


class MetricsEvent(BaseEvent):
    """Performance metrics event"""
    conversation_id: Optional[str] = None
    metric_type: str  # "latency", "usage", "performance"
    service: Optional[str] = None  # "stt", "llm", "tts"
    value: float
    unit: str = "ms"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConnectionEvent(BaseEvent):
    """WebSocket connection event"""
    connection_id: str
    session_id: Optional[str] = None
    action: str  # "established", "closed", "error"
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    reason: Optional[str] = None


class ErrorEvent(BaseEvent):
    """Error event"""
    conversation_id: Optional[str] = None
    error_type: str
    error_message: str
    service: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateChangeEvent(BaseEvent):
    """State change event"""
    conversation_id: str
    entity_type: str  # "conversation", "vad", "pipeline"
    old_state: str
    new_state: str
    reason: Optional[str] = None


class ConfigUpdateEvent(BaseEvent):
    """Configuration update event"""
    conversation_id: Optional[str] = None
    config_type: str  # "stt", "llm", "tts", "pipeline"
    old_values: Dict[str, Any] = Field(default_factory=dict)
    new_values: Dict[str, Any] = Field(default_factory=dict)
    updated_by: Optional[str] = None