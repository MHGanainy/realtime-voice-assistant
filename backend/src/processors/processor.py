"""
Optimized conversation processors with minimal overhead.
"""
from pipecat.metrics.metrics import MetricsData, TTFBMetricsData, ProcessingMetricsData
import logging
import asyncio
from typing import Optional, Dict, Any, Set, Type
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


class BaseProcessor(FrameProcessor):
    """
    Optimized base processor with minimal overhead.
    Each subclass defines which frames it cares about.
    """
    
    # Override in subclasses - which frame types this processor handles
    HANDLED_FRAMES: tuple = ()
    
    # Shared state across all processor instances for each conversation
    _shared_state: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, conversation_id: str, position: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.position = position
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Lazy initialization of heavy objects
        self._conversation_manager = None
        self._event_bus = None
        self._session_id = None
        
        # Initialize shared state for this conversation if needed
        if conversation_id not in self._shared_state:
            self._shared_state[conversation_id] = {
                "conversation_buffer": {
                    "assistant_text": ""
                }
            }
        
        logger.info(f"Initialized {self.__class__.__name__} at position: {position}")
    
    @property
    def conversation_manager(self):
        """Lazy load conversation manager"""
        if self._conversation_manager is None:
            self._conversation_manager = get_conversation_manager()
        return self._conversation_manager
    
    @property
    def event_bus(self):
        """Lazy load event bus"""
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus
    
    @property
    def session_id(self):
        """Lazy load session ID"""
        if self._session_id is None:
            conv = self.conversation_manager.get_conversation(self.conversation_id)
            if conv:
                self._session_id = conv.participant.session_id
        return self._session_id
    
    @property
    def shared_state(self) -> Dict[str, Any]:
        """Get shared state for this conversation"""
        return self._shared_state[self.conversation_id]
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with minimal overhead"""
        await super().process_frame(frame, direction)
        
        # Always pass the frame through immediately
        await self.push_frame(frame, direction)
        
        # Quick check if we should handle this frame
        if not self.HANDLED_FRAMES or not isinstance(frame, self.HANDLED_FRAMES):
            return
        
        # Process in background
        task = asyncio.create_task(self._handle_frame_async(frame, direction))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _handle_frame_async(self, frame: Frame, direction: FrameDirection):
        """Override in subclasses for specific handling"""
        pass
    
    async def cleanup(self):
        """Clean up background tasks"""
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    @classmethod
    def cleanup_conversation(cls, conversation_id: str):
        """Clean up shared state for a conversation"""
        if conversation_id in cls._shared_state:
            del cls._shared_state[conversation_id]


class InputProcessor(BaseProcessor):
    """Minimal input processor - currently handles nothing"""
    HANDLED_FRAMES = ()  # No frames handled
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "input")


class PostSTTProcessor(BaseProcessor):
    """Handles only final transcriptions"""
    HANDLED_FRAMES = (TranscriptionFrame,)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-stt")
    
    async def _handle_frame_async(self, frame: Frame, direction: FrameDirection):
        """Handle transcription"""
        try:
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                logger.info(f"[{self.conversation_id}] User: {frame.text}")
                
                await self.conversation_manager.handle_transcription(
                    conversation_id=self.conversation_id,
                    text=frame.text,
                    is_final=True,
                    speaker="participant",
                    confidence=getattr(frame, 'confidence', None)
                )
        except Exception as e:
            logger.error(f"Error in PostSTTProcessor: {e}")


class PostLLMProcessor(BaseProcessor):
    """Handles LLM response frames"""
    HANDLED_FRAMES = (LLMFullResponseStartFrame, LLMTextFrame, TextFrame, LLMFullResponseEndFrame)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-llm")
    
    async def _handle_frame_async(self, frame: Frame, direction: FrameDirection):
        """Handle LLM frames"""
        try:
            if isinstance(frame, LLMFullResponseStartFrame):
                self.shared_state["conversation_buffer"]["assistant_text"] = ""
            
            elif isinstance(frame, (LLMTextFrame, TextFrame)) and direction == FrameDirection.DOWNSTREAM:
                if hasattr(frame, 'text') and frame.text.strip():
                    self.shared_state["conversation_buffer"]["assistant_text"] += frame.text
            
            elif isinstance(frame, LLMFullResponseEndFrame):
                assistant_text = self.shared_state["conversation_buffer"]["assistant_text"]
                if assistant_text.strip():
                    logger.info(f"[{self.conversation_id}] Assistant: {assistant_text}")
                    
                    await self.conversation_manager.handle_transcription(
                        conversation_id=self.conversation_id,
                        text=assistant_text,
                        is_final=True,
                        speaker="assistant"
                    )
                    
                    self.shared_state["conversation_buffer"]["assistant_text"] = ""
                    
        except Exception as e:
            logger.error(f"Error in PostLLMProcessor: {e}")


class PostTTSProcessor(BaseProcessor):
    """Handles only metrics frames"""
    HANDLED_FRAMES = (MetricsFrame,)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-tts")
    
    async def _handle_frame_async(self, frame: Frame, direction: FrameDirection):
        """Handle metrics"""
        try:
            if isinstance(frame, MetricsFrame) and hasattr(frame, 'data') and frame.data:
                # Log metrics frame details
                logger.info(f"📋 [{self.position}] MetricsFrame | DOWNSTREAM | data: {frame.data}")
                
                # Batch process all metrics
                events = []
                
                for metric in frame.data:
                    if not getattr(metric, 'value', 0):
                        continue
                    
                    metric_dict = metric.model_dump(exclude_none=True)
                    processor_name = metric_dict.get('processor', '').lower()
                    
                    # Determine service type
                    if 'stt' in processor_name or 'deepgram' in processor_name:
                        service = 'stt'
                    elif 'llm' in processor_name and 'tts' not in processor_name:
                        service = 'llm'
                    elif 'tts' in processor_name:
                        service = 'tts'
                    else:
                        service = 'unknown'
                    
                    # Prepare event data
                    if isinstance(metric, TTFBMetricsData):
                        event_type = 'ttfb'
                        value_key = 'ttfb'
                    elif isinstance(metric, ProcessingMetricsData):
                        event_type = 'processing_time'
                        value_key = 'processing_time'
                    else:
                        continue
                    
                    events.append({
                        'type': f"conversation:{self.conversation_id}:metrics:{event_type}",
                        'data': {
                            'conversation_id': self.conversation_id,
                            'session_id': self.session_id,
                            'position': self.position,
                            'service': service,
                            'processor': metric_dict.get('processor'),
                            'model': metric_dict.get('model'),
                            value_key: metric_dict.get('value'),
                            f"{value_key}_ms": int(metric_dict.get('value', 0) * 1000),
                            'unit': 'ms'
                        }
                    })
                
                # Emit all events
                for event in events:
                    await self.event_bus.emit(event['type'], **event['data'])
                    
        except Exception as e:
            logger.error(f"Error in PostTTSProcessor: {e}")


# Factory function
def create_conversation_processor(conversation_id: str, position: str) -> BaseProcessor:
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