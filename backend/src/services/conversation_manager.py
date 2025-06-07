"""
Conversation management service for voice assistant.
Handles the lifecycle of 1-on-1 conversations.
"""
import asyncio
from typing import Dict, Optional, Callable, Any, List
from datetime import datetime, timedelta
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
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._settings = get_settings()
        
        # Start cleanup task
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
        await self._emit_event("conversation_created", conversation)
        
        logger.info(f"Created conversation {conversation.id} for participant {participant.id}")
        return conversation
    
    async def start_conversation(
        self,
        conversation_id: str,
        transport: Any,  # FastAPIWebsocketTransport
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
            # Store pipeline reference
            self._active_pipelines[conversation_id] = {
                "pipeline": pipeline,
                "transport": transport,
                "task": None
            }
            
            # Update conversation state
            conversation.start()
            conversation.pipeline_id = id(pipeline)
            
            await self._emit_event("conversation_started", conversation)
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
        
        # Clean up pipeline
        if conversation_id in self._active_pipelines:
            pipeline_info = self._active_pipelines[conversation_id]
            task = pipeline_info.get("task")
            if task:
                await task.cancel()
            
            del self._active_pipelines[conversation_id]
        
        # Update conversation
        conversation.end(error=error)
        
        # Emit appropriate event
        if error:
            await self._emit_event("conversation_error", {
                "conversation": conversation,
                "error": error
            })
        else:
            await self._emit_event("conversation_ended", conversation)
        
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
        
        # Update metrics
        if direction == AudioDirection.INBOUND:
            conversation.metrics.total_audio_bytes_in += len(audio_data)
        else:
            conversation.metrics.total_audio_bytes_out += len(audio_data)
        
        chunk = AudioChunk(
            conversation_id=conversation_id,
            direction=direction,
            data=audio_data,
            sample_rate=conversation.config.sample_rate,
            channels=conversation.config.channels,
            format=conversation.config.audio_format
        )
        
        await self._emit_event("audio_chunk", chunk)
    
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
        
        if is_final:
            turn = conversation.add_turn(
                speaker=speaker,
                text=text,
                confidence=confidence
            )
            await self._emit_event("turn_completed", {
                "conversation": conversation,
                "turn": turn
            })
            logger.debug(f"[{conversation_id}] {speaker}: {text}")
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
            await self._emit_event("interruption", {
                "conversation_id": conversation_id,
                "interrupted_turn_id": interrupted_turn_id
            })
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self._conversations.get(conversation_id)
    
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
            logger.info(f"Cleaned up conversation {conversation_id}")
    
    async def _cleanup_stale_conversations(self):
        """Background task to clean up stale conversations"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                now = datetime.utcnow()
                max_duration = timedelta(
                    milliseconds=self._settings.max_conversation_duration_ms
                )
                
                stale_conversations = []
                for cid, conv in self._conversations.items():
                    if conv.state == ConversationState.ACTIVE:
                        if conv.started_at and (now - conv.started_at) > max_duration:
                            stale_conversations.append(cid)
                    elif conv.state in [ConversationState.ENDED, ConversationState.ERROR]:
                        if conv.ended_at and (now - conv.ended_at) > timedelta(minutes=10):
                            stale_conversations.append(cid)
                
                for cid in stale_conversations:
                    await self.cleanup_conversation(cid)
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def on_event(self, event_name: str, handler: Callable):
        """Register event handler"""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        return lambda: self._event_handlers[event_name].remove(handler)
    
    async def _emit_event(self, event_name: str, data: Any):
        """Emit event to registered handlers"""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")
    
    async def shutdown(self):
        """Shutdown manager and cleanup resources"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # End all active conversations
        active_convs = list(self.get_active_conversations().keys())
        for conv_id in active_convs:
            await self.end_conversation(conv_id)
        
        logger.info("Conversation manager shut down")


# Global conversation manager instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the singleton conversation manager"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager