from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class EventType(str, Enum):
    """Event types for the voice assistant pipeline"""
    # STT Events
    STT_CONNECTED = "stt.connected"
    STT_DISCONNECTED = "stt.disconnected"
    STT_AUDIO_RECEIVED = "stt.audio_received"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    UTTERANCE_END = "utterance.end"
    
    # LLM Events
    LLM_REQUEST_START = "llm.request_start"
    LLM_TOKEN = "llm.token"
    LLM_COMPLETE = "llm.complete"
    
    # TTS Events
    TTS_REQUEST_START = "tts.request_start"
    TTS_CHUNK = "tts.chunk"
    TTS_COMPLETE = "tts.complete"
    
    # Pipeline Events
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    
    # System Events
    ERROR = "system.error"
    WARNING = "system.warning"
    COMMAND = "system.command"
    METRICS = "system.metrics"

class Event(BaseModel):
    """Base event model"""
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None

class TranscriptEvent(Event):
    """Transcript event from STT"""
    transcript: str
    is_final: bool
    speech_final: bool
    confidence: Optional[float] = None
    
    def __init__(self, **kwargs):
        super().__init__(type=EventType.TRANSCRIPT_FINAL if kwargs.get('is_final') else EventType.TRANSCRIPT_PARTIAL, **kwargs)

class LLMTokenEvent(Event):
    """LLM token event"""
    token: str
    token_count: int
    
    def __init__(self, **kwargs):
        super().__init__(type=EventType.LLM_TOKEN, **kwargs)

class AudioChunkEvent(Event):
    """Audio chunk event from TTS"""
    audio_chunk: bytes
    chunk_index: int
    format: str = "mp3"
    
    def __init__(self, **kwargs):
        super().__init__(type=EventType.TTS_CHUNK, **kwargs)

class ErrorEvent(Event):
    """Error event"""
    error_message: str
    error_type: str
    traceback: Optional[str] = None
    retry_attempt: Optional[int] = None
    
    def __init__(self, **kwargs):
        super().__init__(type=EventType.ERROR, **kwargs)

class MetricsEvent(Event):
    """Metrics event"""
    metrics: Dict[str, float]
    
    def __init__(self, **kwargs):
        super().__init__(type=EventType.METRICS, **kwargs)