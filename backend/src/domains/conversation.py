"""
Domain models for realtime voice conversations with AI assistant.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class ConversationState(Enum):
    """State of a conversation"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


class AudioDirection(Enum):
    """Direction of audio flow"""
    INBOUND = "inbound"   # Participant to Assistant
    OUTBOUND = "outbound" # Assistant to Participant


@dataclass
class ConversationConfig:
    """Configuration for a voice conversation"""
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    audio_format: str = "pcm"
    
    # Service configurations
    stt_provider: str = "deepgram"
    stt_model: Optional[str] = "nova-2"
    llm_provider: str = "deepinfra"
    llm_model: Optional[str] = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tts_provider: str = "deepinfra"
    tts_model: Optional[str] = "hexgrad/Kokoro-82M"
    tts_voice: str = "af_bella"
    
    # Behavior settings
    system_prompt: str = "You are a helpful assistant. Keep your responses brief and conversational."
    enable_interruptions: bool = True
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    
    # LLM parameters
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_top_p: float = 1.0
    
    # Advanced settings
    pipeline_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "audio": {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "format": self.audio_format
            },
            "services": {
                "stt": {"provider": self.stt_provider, "model": self.stt_model},
                "llm": {"provider": self.llm_provider, "model": self.llm_model},
                "tts": {"provider": self.tts_provider, "model": self.tts_model, "voice": self.tts_voice}
            },
            "behavior": {
                "system_prompt": self.system_prompt,
                "enable_interruptions": self.enable_interruptions,
                "vad_enabled": self.vad_enabled
            }
        }


@dataclass
class Participant:
    """Represents the human participant in a conversation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connection_id: str = ""
    connected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Client information
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Session info
    session_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    speaker: str = ""  # "participant" or "assistant"
    text: str = ""
    audio_duration_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # For tracking interruptions
    was_interrupted: bool = False
    interrupted_at_ms: Optional[int] = None
    
    # Quality metrics
    confidence: Optional[float] = None
    
    # Metadata (e.g., emotions, intents, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMetrics:
    """Metrics for a conversation"""
    total_duration_ms: int = 0
    participant_speaking_time_ms: int = 0
    assistant_speaking_time_ms: int = 0
    silence_time_ms: int = 0
    
    turn_count: int = 0
    interruption_count: int = 0
    
    # Audio metrics
    total_audio_bytes_in: int = 0
    total_audio_bytes_out: int = 0
    
    # Quality metrics
    average_confidence: float = 0.0
    average_response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "duration": {
                "total_ms": self.total_duration_ms,
                "participant_ms": self.participant_speaking_time_ms,
                "assistant_ms": self.assistant_speaking_time_ms,
                "silence_ms": self.silence_time_ms
            },
            "interaction": {
                "turns": self.turn_count,
                "interruptions": self.interruption_count
            },
            "audio": {
                "bytes_in": self.total_audio_bytes_in,
                "bytes_out": self.total_audio_bytes_out
            },
            "quality": {
                "avg_confidence": self.average_confidence,
                "avg_response_time_ms": self.average_response_time_ms
            }
        }


@dataclass
class Conversation:
    """Represents a voice conversation session"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participant: Participant = field(default_factory=Participant)
    config: ConversationConfig = field(default_factory=ConversationConfig)
    state: ConversationState = ConversationState.INITIALIZING
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Conversation history
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Metrics
    metrics: ConversationMetrics = field(default_factory=ConversationMetrics)
    
    # Pipeline reference
    pipeline_id: Optional[str] = None
    
    # Error tracking
    error: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self):
        """Mark conversation as started"""
        self.started_at = datetime.utcnow()
        self.state = ConversationState.ACTIVE
    
    def add_turn(self, speaker: str, text: str, **kwargs) -> ConversationTurn:
        """Add a turn to the conversation"""
        turn = ConversationTurn(
            conversation_id=self.id,
            speaker=speaker,
            text=text,
            **kwargs
        )
        
        self.turns.append(turn)
        self.metrics.turn_count += 1
        
        # Update speaking time if duration provided
        if turn.audio_duration_ms:
            if speaker == "participant":
                self.metrics.participant_speaking_time_ms += turn.audio_duration_ms
            else:
                self.metrics.assistant_speaking_time_ms += turn.audio_duration_ms
        
        return turn
    
    def handle_interruption(self, interrupted_turn_id: str, timestamp_ms: int):
        """Mark a turn as interrupted"""
        for turn in self.turns:
            if turn.id == interrupted_turn_id:
                turn.was_interrupted = True
                turn.interrupted_at_ms = timestamp_ms
                self.metrics.interruption_count += 1
                break
    
    def end(self, error: Optional[str] = None):
        """End the conversation"""
        self.ended_at = datetime.utcnow()
        if error:
            self.state = ConversationState.ERROR
            self.error = error
        else:
            self.state = ConversationState.ENDED
        
        # Calculate final metrics
        if self.started_at:
            self.metrics.total_duration_ms = int(
                (self.ended_at - self.started_at).total_seconds() * 1000
            )
            self.metrics.silence_time_ms = (
                self.metrics.total_duration_ms - 
                self.metrics.participant_speaking_time_ms - 
                self.metrics.assistant_speaking_time_ms
            )
    
    def get_transcript(self) -> List[Dict[str, Any]]:
        """Get conversation transcript"""
        return [
            {
                "speaker": turn.speaker,
                "text": turn.text,
                "timestamp": turn.timestamp.isoformat(),
                "duration_ms": turn.audio_duration_ms,
                "was_interrupted": turn.was_interrupted
            }
            for turn in self.turns
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "state": self.state.value,
            "participant_id": self.participant.id,
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metrics": self.metrics.to_dict(),
            "turn_count": len(self.turns),
            "error": self.error
        }


@dataclass
class AudioChunk:
    """Represents a chunk of audio data"""
    conversation_id: str
    direction: AudioDirection
    data: bytes
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence_number: int = 0
    
    # Audio properties
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm"