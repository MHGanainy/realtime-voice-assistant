"""
Custom processors for conversation pipeline.
"""
import logging
from typing import Optional
from datetime import datetime

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TextFrame,
    AudioRawFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    MetricsFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from src.services.conversation_manager import get_conversation_manager
from src.domains.conversation import AudioDirection

logger = logging.getLogger(__name__)


class ConversationProcessor(FrameProcessor):
    """
    Processor that integrates with ConversationManager to track
    conversation state, transcriptions, and metrics.
    """
    
    def __init__(self, conversation_id: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.conversation_manager = get_conversation_manager()
        self._current_speaker_start: Optional[datetime] = None
        self._last_assistant_turn_id: Optional[str] = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and update conversation state"""
        # IMPORTANT: Call parent's process_frame first to handle StartFrame/EndFrame
        await super().process_frame(frame, direction)
        
        try:
            # Handle interim transcription frames
            if isinstance(frame, InterimTranscriptionFrame):
                # You can broadcast interim transcriptions to frontend if needed
                logger.debug(f"Interim transcription: {frame.text}")
            
            # Handle final transcription frames
            elif isinstance(frame, TranscriptionFrame):
                await self._handle_transcription(frame)
            
            # Handle text frames (assistant responses)
            elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
                await self._handle_assistant_text(frame)
            
            # Handle audio frames for metrics
            elif isinstance(frame, AudioRawFrame):
                await self._handle_audio_frame(frame, direction)
            
            # Handle interruption events
            elif isinstance(frame, StartInterruptionFrame):
                await self._handle_start_interruption()
            
            elif isinstance(frame, StopInterruptionFrame):
                await self._handle_stop_interruption()
            
            # Handle speaking events
            elif isinstance(frame, UserStartedSpeakingFrame):
                await self._handle_user_started_speaking()
            
            elif isinstance(frame, UserStoppedSpeakingFrame):
                await self._handle_user_stopped_speaking()
            
            elif isinstance(frame, BotStartedSpeakingFrame):
                await self._handle_bot_started_speaking()
            
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self._handle_bot_stopped_speaking()
            
            # Handle metrics
            elif isinstance(frame, MetricsFrame):
                await self._handle_metrics(frame)
                
        except Exception as e:
            logger.error(f"Error in ConversationProcessor: {e}", exc_info=True)
        
        # Always pass the frame through
        await self.push_frame(frame, direction)
    
    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle transcription from user"""
        if frame.text.strip():
            logger.info(f"[{self.conversation_id}] User: {frame.text}")
            
            await self.conversation_manager.handle_transcription(
                conversation_id=self.conversation_id,
                text=frame.text,
                is_final=True,
                speaker="participant",
                confidence=getattr(frame, 'confidence', None)
            )
    
    async def _handle_assistant_text(self, frame: TextFrame):
        """Handle text response from assistant"""
        if frame.text.strip():
            logger.info(f"[{self.conversation_id}] Assistant: {frame.text}")
            
            turn = await self.conversation_manager.handle_transcription(
                conversation_id=self.conversation_id,
                text=frame.text,
                is_final=True,
                speaker="assistant"
            )
            # Store the turn ID for potential interruption tracking
            if turn and hasattr(turn, 'id'):
                self._last_assistant_turn_id = turn.id
    
    async def _handle_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        """Track audio data for metrics"""
        if direction == FrameDirection.UPSTREAM:
            audio_direction = AudioDirection.INBOUND
        else:
            audio_direction = AudioDirection.OUTBOUND
        
        await self.conversation_manager.handle_audio_chunk(
            conversation_id=self.conversation_id,
            audio_data=frame.audio,
            direction=audio_direction
        )
    
    async def _handle_start_interruption(self):
        """Handle interruption start"""
        logger.debug(f"Interruption started in conversation {self.conversation_id}")
        if self._last_assistant_turn_id:
            timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
            await self.conversation_manager.handle_interruption(
                conversation_id=self.conversation_id,
                interrupted_turn_id=self._last_assistant_turn_id,
                timestamp_ms=timestamp_ms
            )
    
    async def _handle_stop_interruption(self):
        """Handle interruption stop"""
        logger.debug(f"Interruption stopped in conversation {self.conversation_id}")
        # Could implement additional logic here if needed
        pass
    
    async def _handle_user_started_speaking(self):
        """Track when user starts speaking"""
        self._current_speaker_start = datetime.utcnow()
        logger.debug(f"User started speaking in conversation {self.conversation_id}")
    
    async def _handle_user_stopped_speaking(self):
        """Track when user stops speaking"""
        if self._current_speaker_start:
            duration_ms = int((datetime.utcnow() - self._current_speaker_start).total_seconds() * 1000)
            # Could update the last turn with duration if needed
            self._current_speaker_start = None
            logger.debug(f"User stopped speaking in conversation {self.conversation_id} (duration: {duration_ms}ms)")
    
    async def _handle_bot_started_speaking(self):
        """Track when bot starts speaking"""
        self._current_speaker_start = datetime.utcnow()
        logger.debug(f"Bot started speaking in conversation {self.conversation_id}")
    
    async def _handle_bot_stopped_speaking(self):
        """Track when bot stops speaking"""
        if self._current_speaker_start:
            duration_ms = int((datetime.utcnow() - self._current_speaker_start).total_seconds() * 1000)
            self._current_speaker_start = None
            logger.debug(f"Bot stopped speaking in conversation {self.conversation_id} (duration: {duration_ms}ms)")
    
    async def _handle_metrics(self, frame: MetricsFrame):
        """Handle metrics frames"""
        # Could aggregate and store metrics here
        logger.debug(f"Metrics for conversation {self.conversation_id}: {frame}")


class AudioMetricsProcessor(FrameProcessor):
    """
    Processor for tracking detailed audio metrics.
    """
    
    def __init__(self, conversation_id: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.conversation_manager = get_conversation_manager()
        self._audio_stats = {
            "inbound": {"bytes": 0, "frames": 0},
            "outbound": {"bytes": 0, "frames": 0}
        }
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio frames for metrics"""
        # IMPORTANT: Call parent's process_frame first
        await super().process_frame(frame, direction)
        
        try:
            if isinstance(frame, AudioRawFrame):
                if direction == FrameDirection.UPSTREAM:
                    self._audio_stats["inbound"]["bytes"] += len(frame.audio)
                    self._audio_stats["inbound"]["frames"] += 1
                else:
                    self._audio_stats["outbound"]["bytes"] += len(frame.audio)
                    self._audio_stats["outbound"]["frames"] += 1
                
                # Periodically log stats
                total_frames = (
                    self._audio_stats["inbound"]["frames"] + 
                    self._audio_stats["outbound"]["frames"]
                )
                if total_frames % 1000 == 0:
                    logger.info(f"Audio stats for {self.conversation_id}: {self._audio_stats}")
                    
        except Exception as e:
            logger.error(f"Error in AudioMetricsProcessor: {e}", exc_info=True)
        
        await self.push_frame(frame, direction)