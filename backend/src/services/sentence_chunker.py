"""
Sentence chunking service for streaming TTS
Implements intelligent sentence boundary detection for lower latency
"""
import re
from typing import List, Optional, Tuple
from pipecat.frames.frames import TextFrame, Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
import asyncio


class SentenceChunker(FrameProcessor):
    """
    Intelligently chunks text into sentences for streaming TTS.
    Reduces perceived latency by sending complete sentences as soon as they're ready.
    """
    
    def __init__(self, 
                 min_sentence_length: int = 10,
                 max_buffer_size: int = 500,
                 timeout_ms: int = 500):
        super().__init__()
        self.min_sentence_length = min_sentence_length
        self.max_buffer_size = max_buffer_size
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds
        
        self.buffer = ""
        self.pending_task = None
        
        # Sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?]+(?:\s+|$)')
        
        # Common abbreviations that don't end sentences
        self.abbreviations = {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
            'inc.', 'ltd.', 'co.', 'corp.', 'eg.', 'e.g.', 'ie.', 'i.e.',
            'etc.', 'vs.', 'st.', 'ave.', 'blvd.'
        }
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            # Cancel any pending timeout
            if self.pending_task:
                self.pending_task.cancel()
                self.pending_task = None
            
            # Add new text to buffer
            self.buffer += frame.text
            
            # Try to extract complete sentences
            sentences = await self._extract_sentences()
            
            # Send complete sentences immediately
            for sentence in sentences:
                if sentence.strip():
                    await self.push_frame(TextFrame(text=sentence), direction)
            
            # If buffer is getting too large, flush it
            if len(self.buffer) > self.max_buffer_size:
                await self._flush_buffer(direction)
            elif self.buffer:
                # Set timeout to flush remaining buffer
                self.pending_task = asyncio.create_task(
                    self._timeout_flush(direction)
                )
        else:
            # Pass through non-text frames
            await self.push_frame(frame, direction)
    
    async def _extract_sentences(self) -> List[str]:
        """Extract complete sentences from buffer"""
        sentences = []
        
        # Find all potential sentence boundaries
        matches = list(self.sentence_endings.finditer(self.buffer))
        
        if not matches:
            return sentences
        
        last_end = 0
        for match in matches:
            end_pos = match.end()
            sentence = self.buffer[last_end:end_pos].strip()
            
            # Check if this is a real sentence ending
            if self._is_sentence_boundary(sentence, end_pos):
                sentences.append(sentence)
                last_end = end_pos
        
        # Update buffer with remaining text
        if last_end > 0:
            self.buffer = self.buffer[last_end:].lstrip()
        
        return sentences
    
    def _is_sentence_boundary(self, sentence: str, end_pos: int) -> bool:
        """Check if this is a real sentence boundary"""
        if len(sentence) < self.min_sentence_length:
            return False
        
        # Check for abbreviations
        words = sentence.lower().split()
        if words and words[-1] in self.abbreviations:
            return False
        
        # Check if next character is lowercase (might be continuation)
        if end_pos < len(self.buffer):
            next_char = self.buffer[end_pos:end_pos+1]
            if next_char.islower():
                return False
        
        return True
    
    async def _timeout_flush(self, direction: FrameDirection):
        """Flush buffer after timeout"""
        await asyncio.sleep(self.timeout_ms)
        await self._flush_buffer(direction)
    
    async def _flush_buffer(self, direction: FrameDirection):
        """Flush remaining buffer content"""
        if self.buffer.strip():
            await self.push_frame(TextFrame(text=self.buffer), direction)
            self.buffer = ""