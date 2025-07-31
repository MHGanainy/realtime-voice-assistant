"""
Conversation management service for voice assistant.
"""
import asyncio
from typing import Dict, Optional, Callable, Any, List
from datetime import datetime, timedelta
from src.services.transcript_storage import get_transcript_storage
import logging

from src.domains.conversation import (
    Conversation, 
    ConversationConfig, 
    ConversationState,
    ConversationTurn,
    Participant,
    AudioDirection,
    AudioChunk,
    ConversationMetrics
)
from src.config.settings import get_settings
from src.events import EventBus, EventStore, get_event_bus, get_event_store
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages voice conversations between participants and AI assistant"""
    
    def __init__(self, pipeline_runner: Optional[PipelineRunner] = None):
        self._conversations: Dict[str, Conversation] = {}
        self._active_pipelines: Dict[str, Dict[str, Any]] = {}
        self._pipeline_runner = pipeline_runner or PipelineRunner()
        self._settings = get_settings()
        
        self._event_bus = get_event_bus()
        self._event_store = get_event_store()
        
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_conversations())
    
    async def create_conversation(
        self,
        participant: Participant,
        config: Optional[ConversationConfig] = None
    ) -> Conversation:
        """Create a new conversation"""
        config = config or ConversationConfig()
        
        conversation = Conversation(
            participant=participant,
            config=config,
            state=ConversationState.INITIALIZING
        )
        
        self._conversations[conversation.id] = conversation
        
        await self._event_bus.emit(
            f"conversation:{conversation.id}:lifecycle:created",
            conversation_id=conversation.id,
            session_id=participant.session_id,
            participant_id=participant.id,
            config=config.to_dict(),
            created_at=conversation.created_at.isoformat()
        )
        
        await self._event_store.store_event(
            f"conversation:{conversation.id}:lifecycle:created",
            {
                "conversation_id": conversation.id,
                "session_id": participant.session_id,
                "participant_id": participant.id,
                "config": config.to_dict()
            }
        )
        
        logger.info(f"Created conversation {conversation.id} for participant {participant.id}")
        return conversation
    
    async def start_conversation(
        self,
        conversation_id: str,
        transport: Any,
        pipeline: Pipeline
    ) -> bool:
        """Start a conversation with the given transport and pipeline"""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return False
        
        if conversation.state != ConversationState.INITIALIZING:
            logger.warning(f"Conversation {conversation_id} in invalid state: {conversation.state}")
            return False
        
        try:
            self._active_pipelines[conversation_id] = {
                "pipeline": pipeline,
                "transport": transport,
                "task": None
            }
            
            conversation.start()
            conversation.pipeline_id = id(pipeline)
            
            await self._event_bus.emit(
                f"conversation:{conversation_id}:lifecycle:started",
                conversation_id=conversation_id,
                session_id=conversation.participant.session_id,
                pipeline_id=conversation.pipeline_id,
                started_at=conversation.started_at.isoformat()
            )
            
            await self._event_store.store_event(
                f"conversation:{conversation_id}:lifecycle:started",
                {
                    "conversation_id": conversation_id,
                    "session_id": conversation.participant.session_id,
                    "pipeline_id": conversation.pipeline_id
                }
            )
            
            logger.info(f"Started conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start conversation {conversation_id}: {e}")
            await self.end_conversation(conversation_id, error=str(e))
            return False
    
    async def run_pipeline_for_conversation(
        self,
        conversation_id: str,
        task: PipelineTask
    ):
        """Run the pipeline task for a conversation"""
        if conversation_id in self._active_pipelines:
            self._active_pipelines[conversation_id]["task"] = task
            
        try:
            await self._pipeline_runner.run(task)
        except Exception as e:
            logger.error(f"Pipeline error for conversation {conversation_id}: {e}")
            await self.end_conversation(conversation_id, error=str(e))
    
    async def end_conversation(
        self,
        conversation_id: str,
        error: Optional[str] = None
    ) -> bool:
        """End an active conversation"""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False
        
        if conversation_id in self._active_pipelines:
            pipeline_info = self._active_pipelines[conversation_id]
            task = pipeline_info.get("task")
            if task:
                await task.cancel()
            
            del self._active_pipelines[conversation_id]
        
        conversation.end(error=error)
        
        if error:
            event_name = f"conversation:{conversation_id}:lifecycle:error"
            event_data = {
                "conversation_id": conversation_id,
                "session_id": conversation.participant.session_id,
                "error": error,
                "ended_at": conversation.ended_at.isoformat(),
                "metrics": conversation.metrics.to_dict()
            }
        else:
            event_name = f"conversation:{conversation_id}:lifecycle:ended"
            event_data = {
                "conversation_id": conversation_id,
                "session_id": conversation.participant.session_id,
                "ended_at": conversation.ended_at.isoformat(),
                "metrics": conversation.metrics.to_dict()
            }
        
        await self._event_bus.emit(event_name, **event_data)
        await self._event_store.store_event(event_name, event_data)
        
        logger.info(f"Ended conversation {conversation_id}")
        return True
    
    async def handle_audio_chunk(
        self,
        conversation_id: str,
        audio_data: bytes,
        direction: AudioDirection = AudioDirection.INBOUND
    ):
        """Handle incoming audio chunk"""
        conversation = self._conversations.get(conversation_id)
        if not conversation or conversation.state != ConversationState.ACTIVE:
            return
        
        if direction == AudioDirection.INBOUND:
            conversation.metrics.total_audio_bytes_in += len(audio_data)
        else:
            conversation.metrics.total_audio_bytes_out += len(audio_data)
        
        total_bytes = (conversation.metrics.total_audio_bytes_in + 
                      conversation.metrics.total_audio_bytes_out)
        if total_bytes % 102400 == 0:
            await self._event_bus.emit(
                f"conversation:{conversation_id}:metrics:audio",
                conversation_id=conversation_id,
                bytes_in=conversation.metrics.total_audio_bytes_in,
                bytes_out=conversation.metrics.total_audio_bytes_out,
                direction=direction.value
            )
    
    async def handle_transcription(
        self,
        conversation_id: str,
        text: str,
        is_final: bool = True,
        speaker: str = "participant",
        confidence: Optional[float] = None
    ) -> Optional[ConversationTurn]:
        """Handle transcription result"""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None
        
        event_type = "final" if is_final else "interim"
        await self._event_bus.emit(
            f"conversation:{conversation_id}:transcription:{event_type}",
            conversation_id=conversation_id,
            session_id=conversation.participant.session_id,
            text=text,
            speaker=speaker,
            is_final=is_final,
            confidence=confidence
        )
        
        if is_final:
            turn = conversation.add_turn(
                speaker=speaker,
                text=text,
                confidence=confidence
            )
            transcript_storage = get_transcript_storage()
            await transcript_storage.add_message(
                conversation_id=conversation_id,
                speaker=speaker,
                message=text,
                audio_duration=turn.audio_duration_ms / 1000 if turn.audio_duration_ms else None
            )
            await self._event_bus.emit(
                f"conversation:{conversation_id}:turn:completed",
                conversation_id=conversation_id,
                session_id=conversation.participant.session_id,
                turn_id=turn.id,
                speaker=speaker,
                text=text,
                confidence=confidence,
                turn_count=conversation.metrics.turn_count
            )
            
            await self._event_store.store_event(
                f"conversation:{conversation_id}:turn:completed",
                {
                    "conversation_id": conversation_id,
                    "session_id": conversation.participant.session_id,
                    "turn_id": turn.id,
                    "speaker": speaker,
                    "text": text
                }
            )
            
            return turn
        
        return None
    
    async def handle_interruption(
        self,
        conversation_id: str,
        interrupted_turn_id: str,
        timestamp_ms: int
    ):
        """Handle an interruption event"""
        conversation = self._conversations.get(conversation_id)
        if conversation:
            conversation.handle_interruption(interrupted_turn_id, timestamp_ms)
            
            await self._event_bus.emit(
                f"conversation:{conversation_id}:turn:interrupted",
                conversation_id=conversation_id,
                session_id=conversation.participant.session_id,
                interrupted_turn_id=interrupted_turn_id,
                timestamp_ms=timestamp_ms,
                interruption_count=conversation.metrics.interruption_count
            )
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self._conversations.get(conversation_id)
    
    def get_conversations_by_session(self, session_id: str) -> List[Conversation]:
        """Get all conversations for a session"""
        return [
            conv for conv in self._conversations.values()
            if conv.participant.session_id == session_id
        ]
    
    def get_active_conversations(self) -> Dict[str, Conversation]:
        """Get all active conversations"""
        return {
            cid: conv for cid, conv in self._conversations.items()
            if conv.state == ConversationState.ACTIVE
        }
    
    def get_all_conversations(self) -> Dict[str, Conversation]:
        """Get all conversations"""
        return self._conversations.copy()
    
    async def cleanup_conversation(self, conversation_id: str):
        """Remove conversation from memory"""
        if conversation_id in self._conversations:
            await self.end_conversation(conversation_id)
            del self._conversations[conversation_id]
            
            await self._event_store.clear_conversation_events(conversation_id)
            
            logger.info(f"Cleaned up conversation {conversation_id}")
    
    async def _cleanup_stale_conversations(self):
        """Background task to clean up stale conversations"""
        while True:
            try:
                await asyncio.sleep(300)
                
                now = datetime.utcnow()
                max_duration = timedelta(
                    milliseconds=self._settings.max_conversation_duration_ms
                )
                
                stale_conversations = []
                for cid, conv in self._conversations.items():
                    if conv.state == ConversationState.ACTIVE:
                        if conv.started_at and (now - conv.started_at) > max_duration:
                            stale_conversations.append(cid)
                            
                            await self._event_bus.emit(
                                f"conversation:{cid}:lifecycle:timeout",
                                conversation_id=cid,
                                session_id=conv.participant.session_id,
                                duration_ms=int((now - conv.started_at).total_seconds() * 1000)
                            )
                    elif conv.state in [ConversationState.ENDED, ConversationState.ERROR]:
                        if conv.ended_at and (now - conv.ended_at) > timedelta(minutes=10):
                            stale_conversations.append(cid)
                
                for cid in stale_conversations:
                    await self.cleanup_conversation(cid)
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def shutdown(self):
        """Shutdown manager and cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        active_convs = list(self.get_active_conversations().keys())
        for conv_id in active_convs:
            await self.end_conversation(conv_id)
        
        logger.info("Conversation manager shut down")


_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the singleton conversation manager"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
