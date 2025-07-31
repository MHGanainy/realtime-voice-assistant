"""
Transcript Storage Service for managing conversation transcripts with correlation tokens
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class TranscriptEntry:
    timestamp: datetime
    speaker: str  # 'student' or 'ai_patient'
    message: str
    audio_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "speaker": self.speaker,
            "message": self.message,
            "audio_duration": self.audio_duration
        }

@dataclass
class ConversationTranscript:
    correlation_token: str
    session_id: str
    conversation_id: str
    started_at: datetime
    entries: List[TranscriptEntry] = field(default_factory=list)
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entry(self, speaker: str, message: str, audio_duration: Optional[float] = None):
        entry = TranscriptEntry(
            timestamp=datetime.utcnow(),
            speaker=speaker,
            message=message,
            audio_duration=audio_duration
        )
        self.entries.append(entry)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_token": self.correlation_token,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": (self.ended_at - self.started_at).total_seconds() if self.ended_at else None,
            "total_messages": len(self.entries),
            "messages": [entry.to_dict() for entry in self.entries],
            "metadata": self.metadata
        }

class TranscriptStorageService:
    def __init__(self):
        self._transcripts: Dict[str, ConversationTranscript] = {}
        self._session_to_correlation: Dict[str, str] = {}
        self._conversation_to_correlation: Dict[str, str] = {}
        self._lock = asyncio.Lock()
    
    async def create_transcript(
        self, 
        session_id: str, 
        conversation_id: str,
        correlation_token: Optional[str] = None
    ) -> ConversationTranscript:
        async with self._lock:
            # If no correlation token provided, use session_id as fallback
            if not correlation_token:
                correlation_token = f"session_{session_id}"
            
            transcript = ConversationTranscript(
                correlation_token=correlation_token,
                session_id=session_id,
                conversation_id=conversation_id,
                started_at=datetime.utcnow()
            )
            
            self._transcripts[correlation_token] = transcript
            self._session_to_correlation[session_id] = correlation_token
            self._conversation_to_correlation[conversation_id] = correlation_token
            
            logger.info(f"Created transcript for session {session_id} with token {correlation_token}")
            
            return transcript
    
    async def add_message(
        self, 
        conversation_id: str, 
        speaker: str, 
        message: str, 
        audio_duration: Optional[float] = None
    ):
        async with self._lock:
            correlation_token = self._conversation_to_correlation.get(conversation_id)
            if not correlation_token:
                logger.warning(f"No correlation token found for conversation {conversation_id}")
                return
            
            transcript = self._transcripts.get(correlation_token)
            if transcript:
                transcript.add_entry(speaker, message, audio_duration)
                logger.debug(f"Added message to transcript {correlation_token}: {speaker} - {message[:50]}...")
    
    async def end_transcript(self, conversation_id: str):
        async with self._lock:
            correlation_token = self._conversation_to_correlation.get(conversation_id)
            if not correlation_token:
                return
            
            transcript = self._transcripts.get(correlation_token)
            if transcript:
                transcript.ended_at = datetime.utcnow()
                logger.info(f"Ended transcript for conversation {conversation_id}")
    
    async def get_transcript_by_correlation(self, correlation_token: str) -> Optional[ConversationTranscript]:
        async with self._lock:
            return self._transcripts.get(correlation_token)
    
    async def get_transcript_by_session(self, session_id: str) -> Optional[ConversationTranscript]:
        async with self._lock:
            correlation_token = self._session_to_correlation.get(session_id)
            if not correlation_token:
                return None
            return self._transcripts.get(correlation_token)
    
    async def get_transcript_by_conversation(self, conversation_id: str) -> Optional[ConversationTranscript]:
        async with self._lock:
            correlation_token = self._conversation_to_correlation.get(conversation_id)
            if not correlation_token:
                return None
            return self._transcripts.get(correlation_token)
    
    async def update_metadata(self, conversation_id: str, metadata: Dict[str, Any]):
        async with self._lock:
            correlation_token = self._conversation_to_correlation.get(conversation_id)
            if not correlation_token:
                return
            
            transcript = self._transcripts.get(correlation_token)
            if transcript:
                transcript.metadata.update(metadata)
    
    async def cleanup_old_transcripts(self, hours: int = 24):
        """Remove transcripts older than specified hours"""
        async with self._lock:
            cutoff_time = datetime.utcnow()
            cutoff_timestamp = cutoff_time.timestamp() - (hours * 3600)
            
            to_remove = []
            for token, transcript in self._transcripts.items():
                if transcript.started_at.timestamp() < cutoff_timestamp:
                    to_remove.append(token)
            
            for token in to_remove:
                transcript = self._transcripts[token]
                del self._session_to_correlation[transcript.session_id]
                del self._conversation_to_correlation[transcript.conversation_id]
                del self._transcripts[token]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old transcripts")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored transcripts"""
        return {
            "total_transcripts": len(self._transcripts),
            "active_transcripts": sum(
                1 for t in self._transcripts.values() 
                if t.ended_at is None
            ),
            "completed_transcripts": sum(
                1 for t in self._transcripts.values() 
                if t.ended_at is not None
            )
        }

# Create a singleton instance
_transcript_storage: Optional[TranscriptStorageService] = None

def get_transcript_storage() -> TranscriptStorageService:
    """Get the singleton transcript storage service"""
    global _transcript_storage
    if _transcript_storage is None:
        _transcript_storage = TranscriptStorageService()
    return _transcript_storage