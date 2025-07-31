"""
Transcript processor for capturing conversation messages
"""
from typing import AsyncGenerator
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor
from src.services.transcript_storage import get_transcript_storage
import logging

logger = logging.getLogger(__name__)


class TranscriptProcessor(FrameProcessor):
    def __init__(self, conversation_id: str):
        super().__init__()
        self.conversation_id = conversation_id
        self.transcript_storage = get_transcript_storage()
    
    async def process_frame(self, frame: Frame, direction: str) -> AsyncGenerator[Frame, None]:
        """Process frames and capture transcript data"""
        
        # Capture user speech (STT output)
        if isinstance(frame, TranscriptionFrame):
            await self.transcript_storage.add_message(
                conversation_id=self.conversation_id,
                speaker="student",
                message=frame.text,
                audio_duration=None  # Could calculate from audio frames if needed
            )
            logger.debug(f"Captured student message: {frame.text[:50]}...")
        
        # Capture AI responses (before TTS)
        elif isinstance(frame, TextFrame) and direction == "downstream":
            await self.transcript_storage.add_message(
                conversation_id=self.conversation_id,
                speaker="ai_patient",
                message=frame.text
            )
            logger.debug(f"Captured AI message: {frame.text[:50]}...")
        
        # Always pass the frame through
        yield frame