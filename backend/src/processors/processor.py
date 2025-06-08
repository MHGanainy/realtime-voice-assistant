"""
Unified conversation processors with position-specific implementations.
"""
from pipecat.metrics.metrics import MetricsData, TTFBMetricsData, ProcessingMetricsData
import logging
import asyncio
from typing import Optional, Dict, Any
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
    MetricsFrame,
    LLMTextFrame,
    TTSTextFrame,
    LLMMessagesFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    ErrorFrame,
    CancelFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from src.services.conversation_manager import get_conversation_manager
from src.domains.conversation import AudioDirection
from src.events import get_event_bus

logger = logging.getLogger(__name__)


class BaseConversationProcessor(FrameProcessor):
    """
    Base processor that provides common functionality for all conversation processors.
    Uses shared state across all instances for the same conversation.
    Processes frames asynchronously to avoid blocking the main pipeline.
    """
    
    # Shared state across all processor instances for each conversation
    _shared_state: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, conversation_id: str, position: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.position = position
        self.conversation_manager = get_conversation_manager()
        self._event_bus = get_event_bus()
        self._session_id: Optional[str] = None
        self._background_tasks = set()  # Track background tasks
        
        # Initialize shared state for this conversation if needed
        if conversation_id not in self._shared_state:
            self._shared_state[conversation_id] = {
                "conversation_buffer": {
                    "assistant_text": ""
                }
            }
        
        # Get session_id from conversation
        conv = self.conversation_manager.get_conversation(conversation_id)
        if conv:
            self._session_id = conv.participant.session_id
        
        logger.info(f"Initialized {self.__class__.__name__} at position: {position}")
    
    @property
    def shared_state(self) -> Dict[str, Any]:
        """Get shared state for this conversation"""
        return self._shared_state[self.conversation_id]
    
    def _create_background_task(self, coro):
        """Create a background task and track it"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with common logic and position-specific handling"""
        await super().process_frame(frame, direction)
        
        # Always pass the frame through immediately
        await self.push_frame(frame, direction)
        
        # Skip audio frames to minimize processing
        if isinstance(frame, AudioRawFrame):
            return
        
        # Process the frame asynchronously in the background
        self._create_background_task(self._process_frame_async(frame, direction))
    
    async def _process_frame_async(self, frame: Frame, direction: FrameDirection):
        """Process frame asynchronously in the background"""
        try:
            # Log only essential frames
            if self._should_log_frame(frame):
                self._log_frame(frame, direction)
            
            # Position-specific handling
            await self.handle_position_specific(frame, direction)
            
            # Common frame handling (only for metrics)
            await self._handle_common_frames(frame, direction)
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} at {self.position}: {e}", exc_info=True)
    
    def _should_log_frame(self, frame: Frame) -> bool:
        """Determine if frame should be logged - only essential frames"""
        # Only log the frames we care about
        essential_frames = (
            TranscriptionFrame,  # Final user transcription
            LLMFullResponseEndFrame,  # LLM complete
            MetricsFrame,        # Built-in metrics (both TTFB and processing time)
        )
        return isinstance(frame, essential_frames)
    
    def _log_frame(self, frame: Frame, direction: FrameDirection):
        """Log frame with appropriate detail level"""
        frame_type = frame.__class__.__name__
        
        if isinstance(frame, (TextFrame, LLMTextFrame, TTSTextFrame)):
            text = getattr(frame, 'text', '')
            preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"📋 [{self.position}] {frame_type} | {direction.name} | text: [{preview}]")
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            logger.info(f"📋 [{self.position}] {frame_type} | {direction.name} | text: [{frame.text}]")
        elif isinstance(frame, MetricsFrame):
            # Special handling for MetricsFrame to see its structure
            logger.info(f"📋 [{self.position}] {frame_type} | {direction.name} | data: {getattr(frame, 'data', 'NO DATA')}")
        else:
            logger.info(f"📋 [{self.position}] {frame_type} | {direction.name}")
    
    async def handle_position_specific(self, frame: Frame, direction: FrameDirection):
        """Override in subclasses for position-specific handling"""
        pass
    
    async def _handle_common_frames(self, frame: Frame, direction: FrameDirection):
        """Handle frames that are common across all positions - only metrics"""
        
        # Only handle metrics frames
        if isinstance(frame, MetricsFrame):
            await self._handle_metrics_frame(frame)
    
    async def _handle_metrics_frame(self, frame: MetricsFrame):
        """Handle metrics frames - emit TTFB and processing time metrics"""
        if hasattr(frame, 'data') and frame.data:
            for metric in frame.data:
                if isinstance(metric, TTFBMetricsData):
                    # Skip metrics with zero or None values (initialization metrics)
                    if not metric.value or metric.value == 0:
                        continue
                        
                    # Get the metric data as a dict
                    metric_dict = metric.model_dump(exclude_none=True)
                    
                    # Extract processor name and determine service type
                    processor_name = metric_dict.get('processor', '').lower()
                    
                    # Determine which position should emit this metric
                    should_emit = False
                    service = 'unknown'
                    
                    if 'sttservice' in processor_name:
                        service = 'stt'
                        should_emit = (self.position == 'post-stt')
                    elif 'llmservice' in processor_name:
                        service = 'llm'
                        should_emit = (self.position == 'post-llm')
                    elif 'ttsservice' in processor_name:
                        service = 'tts'
                        should_emit = (self.position == 'post-tts')
                    
                    # Only emit if we're at the correct position
                    if should_emit:
                        await self._event_bus.emit(
                            f"conversation:{self.conversation_id}:metrics:ttfb",
                            conversation_id=self.conversation_id,
                            session_id=self._session_id,
                            position=self.position,
                            service=service,
                            processor=metric_dict.get('processor'),
                            model=metric_dict.get('model'),
                            ttfb=metric_dict.get('value'),  # The actual TTFB value
                            ttfb_ms=int(metric_dict.get('value', 0) * 1000),  # Convert to milliseconds
                            unit='ms'
                        )
                
                elif isinstance(metric, ProcessingMetricsData):
                    # Skip metrics with zero or None values
                    if not metric.value or metric.value == 0:
                        continue
                    
                    # Get the metric data as a dict
                    metric_dict = metric.model_dump(exclude_none=True)
                    
                    # Extract processor name and determine service type
                    processor_name = metric_dict.get('processor', '').lower()
                    
                    # Determine which position should emit this metric
                    should_emit = False
                    service = 'unknown'
                    
                    if 'sttservice' in processor_name:
                        service = 'stt'
                        should_emit = (self.position == 'post-stt')
                    elif 'llmservice' in processor_name:
                        service = 'llm'
                        should_emit = (self.position == 'post-llm')
                    elif 'ttsservice' in processor_name:
                        service = 'tts'
                        should_emit = (self.position == 'post-tts')
                    
                    # Only emit if we're at the correct position
                    if should_emit:
                        await self._event_bus.emit(
                            f"conversation:{self.conversation_id}:metrics:processing_time",
                            conversation_id=self.conversation_id,
                            session_id=self._session_id,
                            position=self.position,
                            service=service,
                            processor=metric_dict.get('processor'),
                            model=metric_dict.get('model'),
                            processing_time=metric_dict.get('value'),  # The actual processing time value
                            processing_time_ms=int(metric_dict.get('value', 0) * 1000),  # Convert to milliseconds
                            unit='ms'
                        )
    
    async def cleanup(self):
        """Clean up background tasks when processor is stopped"""
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    @classmethod
    def cleanup_conversation(cls, conversation_id: str):
        """Clean up shared state for a conversation"""
        if conversation_id in cls._shared_state:
            del cls._shared_state[conversation_id]


class InputProcessor(BaseConversationProcessor):
    """Processor at the input stage - handles raw input frames"""
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "input")
    
    async def handle_position_specific(self, frame: Frame, direction: FrameDirection):
        """Handle input-specific frames"""
        # All processing is now async and non-blocking
        pass


class PostSTTProcessor(BaseConversationProcessor):
    """Processor after STT - handles transcriptions"""
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-stt")
    
    async def handle_position_specific(self, frame: Frame, direction: FrameDirection):
        """Handle post-STT frames"""
        # Only handle final transcriptions
        if isinstance(frame, TranscriptionFrame):
            await self._handle_final_transcription(frame)
    
    async def _handle_final_transcription(self, frame: TranscriptionFrame):
        """Handle final transcription"""
        if frame.text.strip():
            logger.info(f"[{self.conversation_id}] User: {frame.text}")
            
            # Save transcription asynchronously
            await self.conversation_manager.handle_transcription(
                conversation_id=self.conversation_id,
                text=frame.text,
                is_final=True,
                speaker="participant",
                confidence=getattr(frame, 'confidence', None)
            )


class PostLLMProcessor(BaseConversationProcessor):
    """Processor after LLM - handles LLM responses"""
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-llm")
    
    async def handle_position_specific(self, frame: Frame, direction: FrameDirection):
        """Handle post-LLM frames"""
        
        # Handle LLM response start
        if isinstance(frame, LLMFullResponseStartFrame):
            self.shared_state["conversation_buffer"]["assistant_text"] = ""
        
        # Handle LLM text frames
        elif isinstance(frame, (LLMTextFrame, TextFrame)) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_llm_text(frame)
        
        # Handle LLM response end
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_response_end()
    
    async def _handle_llm_text(self, frame: TextFrame):
        """Handle LLM text output"""
        if frame.text.strip():
            # Buffer the text
            self.shared_state["conversation_buffer"]["assistant_text"] += frame.text
    
    async def _handle_llm_response_end(self):
        """Handle end of LLM response"""
        assistant_text = self.shared_state["conversation_buffer"]["assistant_text"]
        
        if assistant_text.strip():
            logger.info(f"[{self.conversation_id}] Assistant: {assistant_text}")
            
            # Save the complete assistant response asynchronously
            await self.conversation_manager.handle_transcription(
                conversation_id=self.conversation_id,
                text=assistant_text,
                is_final=True,
                speaker="assistant"
            )
            
            # Clear buffer
            self.shared_state["conversation_buffer"]["assistant_text"] = ""


class PostTTSProcessor(BaseConversationProcessor):
    """Processor after TTS - handles TTS output"""
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-tts")
    
    async def handle_position_specific(self, frame: Frame, direction: FrameDirection):
        """Handle post-TTS frames"""
        # All metrics are handled by Pipecat's built-in metrics
        pass


# Factory function to create appropriate processor
def create_conversation_processor(conversation_id: str, position: str) -> BaseConversationProcessor:
    """Create the appropriate processor for the given position"""
    processors = {
        "input": InputProcessor,
        "post-stt": PostSTTProcessor,
        "post-llm": PostLLMProcessor,
        "post-tts": PostTTSProcessor
    }
    
    processor_class = processors.get(position)
    if not processor_class:
        raise ValueError(f"Unknown processor position: {position}")
    
    return processor_class(conversation_id)