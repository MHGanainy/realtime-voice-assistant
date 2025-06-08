"""
Optimized processors that collect events per conversation turn.
"""
from pipecat.metrics.metrics import MetricsData, TTFBMetricsData, ProcessingMetricsData
import logging
import asyncio
from typing import Optional, Dict, Any, Set, List
from datetime import datetime
from dataclasses import dataclass

from pipecat.frames.frames import *
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from src.services.conversation_manager import get_conversation_manager
from src.events import get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class PendingEvent:
    """Represents an event to be emitted later"""
    event_type: str
    data: Dict[str, Any]


class EventCollectorProcessor(FrameProcessor):
    """
    Base processor that collects events during a conversation turn.
    Events are emitted at the end of each TTS cycle.
    """
    
    # Override in subclasses
    HANDLED_FRAMES: tuple = ()
    
    # Shared state across all processor instances for each conversation
    _shared_state: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, conversation_id: str, position: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.position = position
        
        # Initialize shared state if needed
        if conversation_id not in self._shared_state:
            self._shared_state[conversation_id] = {
                "current_turn_events": [],  # Events for current turn
                "current_turn_transcriptions": [],  # Transcriptions for current turn
                "conversation_buffer": {
                    "assistant_text": ""
                },
                "turn_active": False,  # Whether we're in an active turn
            }
    
    @property
    def shared_state(self) -> Dict[str, Any]:
        """Get shared state for this conversation"""
        return self._shared_state[self.conversation_id]
    
    def add_event(self, event_type: str, **data):
        """Add an event to be emitted at the end of the turn"""
        self.shared_state["current_turn_events"].append(
            PendingEvent(event_type, data)
        )
    
    def add_transcription(self, text: str, speaker: str, **kwargs):
        """Add a transcription to be saved at the end of the turn"""
        self.shared_state["current_turn_transcriptions"].append({
            "text": text,
            "speaker": speaker,
            **kwargs
        })
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Ultra-minimal processing"""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        
        # Quick check and synchronous processing
        if self.HANDLED_FRAMES and isinstance(frame, self.HANDLED_FRAMES):
            self._handle_frame_sync(frame, direction)
    
    def _handle_frame_sync(self, frame: Frame, direction: FrameDirection):
        """Synchronous frame handling - override in subclasses"""
        pass
    
    @classmethod
    def cleanup_conversation(cls, conversation_id: str):
        """Clean up shared state for a conversation"""
        if conversation_id in cls._shared_state:
            del cls._shared_state[conversation_id]


class InputEventCollector(EventCollectorProcessor):
    """Input processor - detects start of turns"""
    HANDLED_FRAMES = (UserStartedSpeakingFrame,)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "input")
    
    def _handle_frame_sync(self, frame: Frame, direction: FrameDirection):
        """Mark turn as active when user starts speaking"""
        if isinstance(frame, UserStartedSpeakingFrame):
            self.shared_state["turn_active"] = True
            # Clear previous turn data
            self.shared_state["current_turn_events"].clear()
            self.shared_state["current_turn_transcriptions"].clear()
            # logger.debug(f"Turn started for conversation {self.conversation_id}")


class PostSTTEventCollector(EventCollectorProcessor):
    """Handles transcriptions"""
    HANDLED_FRAMES = (TranscriptionFrame,)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-stt")
    
    def _handle_frame_sync(self, frame: Frame, direction: FrameDirection):
        """Collect transcription for the current turn"""
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            # logger.info(f"[{self.conversation_id}] User: {frame.text}")
            
            # Store for this turn
            self.add_transcription(
                text=frame.text,
                speaker="participant",
                is_final=True,
                confidence=getattr(frame, 'confidence', None)
            )


class PostLLMEventCollector(EventCollectorProcessor):
    """Handles LLM responses"""
    HANDLED_FRAMES = (LLMFullResponseStartFrame, LLMTextFrame, TextFrame, LLMFullResponseEndFrame)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-llm")
    
    def _handle_frame_sync(self, frame: Frame, direction: FrameDirection):
        """Collect LLM response for the current turn"""
        if isinstance(frame, LLMFullResponseStartFrame):
            self.shared_state["conversation_buffer"]["assistant_text"] = ""
        
        elif isinstance(frame, (LLMTextFrame, TextFrame)) and direction == FrameDirection.DOWNSTREAM:
            if hasattr(frame, 'text') and frame.text.strip():
                self.shared_state["conversation_buffer"]["assistant_text"] += frame.text
        
        elif isinstance(frame, LLMFullResponseEndFrame):
            assistant_text = self.shared_state["conversation_buffer"]["assistant_text"]
            if assistant_text.strip():
                # logger.info(f"[{self.conversation_id}] Assistant: {assistant_text}")
                
                # Store for this turn
                self.add_transcription(
                    text=assistant_text,
                    speaker="assistant",
                    is_final=True
                )
                
                self.shared_state["conversation_buffer"]["assistant_text"] = ""


class PostTTSEventCollector(EventCollectorProcessor):
    """
    Handles metrics and emits all events at the end of each turn.
    """
    HANDLED_FRAMES = (MetricsFrame, TTSStoppedFrame, BotStoppedSpeakingFrame)
    
    def __init__(self, conversation_id: str):
        super().__init__(conversation_id, "post-tts")
        self._conversation_manager = None
        self._event_bus = None
        self._session_id = None
    
    @property
    def conversation_manager(self):
        if self._conversation_manager is None:
            self._conversation_manager = get_conversation_manager()
        return self._conversation_manager
    
    @property
    def event_bus(self):
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus
    
    @property
    def session_id(self):
        if self._session_id is None:
            conv = self.conversation_manager.get_conversation(self.conversation_id)
            if conv:
                self._session_id = conv.participant.session_id
        return self._session_id
    
    def _handle_frame_sync(self, frame: Frame, direction: FrameDirection):
        """Collect metrics and emit everything when TTS completes"""
        
        # Collect metrics
        if isinstance(frame, MetricsFrame) and hasattr(frame, 'data') and frame.data:
            # logger.info(f"📋 [{self.position}] MetricsFrame | {direction.name} | data: {frame.data}")
            
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
                
                # Store event for this turn
                if isinstance(metric, TTFBMetricsData):
                    self.add_event(
                        f"conversation:{self.conversation_id}:metrics:ttfb",
                        conversation_id=self.conversation_id,
                        session_id=self.session_id,
                        position=self.position,
                        service=service,
                        processor=metric_dict.get('processor'),
                        model=metric_dict.get('model'),
                        ttfb=metric_dict.get('value'),
                        ttfb_ms=int(metric_dict.get('value', 0) * 1000),
                        unit='ms'
                    )
                elif isinstance(metric, ProcessingMetricsData):
                    self.add_event(
                        f"conversation:{self.conversation_id}:metrics:processing_time",
                        conversation_id=self.conversation_id,
                        session_id=self.session_id,
                        position=self.position,
                        service=service,
                        processor=metric_dict.get('processor'),
                        model=metric_dict.get('model'),
                        processing_time=metric_dict.get('value'),
                        processing_time_ms=int(metric_dict.get('value', 0) * 1000),
                        unit='ms'
                    )
        
        # Check if TTS has finished (end of turn)
        elif isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            if self.shared_state["turn_active"]:
                # logger.debug(f"Turn completed for conversation {self.conversation_id}, emitting all events")
                # Schedule emission in background
                asyncio.create_task(self._emit_turn_events())
                # Mark turn as inactive
                self.shared_state["turn_active"] = False
    
    async def _emit_turn_events(self):
        """Emit all events and save transcriptions for the completed turn"""
        try:
            # Get events and transcriptions for this turn
            events = self.shared_state["current_turn_events"][:]
            transcriptions = self.shared_state["current_turn_transcriptions"][:]
            
            # Clear for next turn
            self.shared_state["current_turn_events"].clear()
            self.shared_state["current_turn_transcriptions"].clear()
            
            # Save transcriptions
            for trans in transcriptions:
                await self.conversation_manager.handle_transcription(
                    conversation_id=self.conversation_id,
                    **trans
                )
            
            # Emit events
            for event in events:
                await self.event_bus.emit(event.event_type, **event.data)
                
            # logger.info(f"Turn complete: Emitted {len(events)} events and saved {len(transcriptions)} transcriptions")
            
        except Exception as e:
            logger.error(f"Error emitting turn events: {e}")
    
    async def cleanup(self):
        """Emit any remaining events on cleanup"""
        if self.shared_state["current_turn_events"] or self.shared_state["current_turn_transcriptions"]:
            await self._emit_turn_events()


# Factory function
def create_conversation_processor(conversation_id: str, position: str) -> EventCollectorProcessor:
    """Create the appropriate processor for the given position"""
    processors = {
        "input": InputEventCollector,
        "post-stt": PostSTTEventCollector,
        "post-llm": PostLLMEventCollector,
        "post-tts": PostTTSEventCollector
    }
    
    processor_class = processors.get(position)
    if not processor_class:
        raise ValueError(f"Unknown processor position: {position}")
    
    return processor_class(conversation_id)