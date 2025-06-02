"""
Enhanced LLM response aggregator with sentence chunking
"""
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator
from pipecat.frames.frames import TextFrame, Frame
from pipecat.processors.frame_processor import FrameDirection
from typing import AsyncIterator


class EnhancedLLMAssistantResponseAggregator(LLMAssistantResponseAggregator):
    """
    Enhanced aggregator that works with sentence chunking for better streaming
    """
    
    def __init__(self, context=None):
        super().__init__(context)
        self.use_sentence_chunking = True
        
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> AsyncIterator[Frame]:
        # Process frame through parent class
        async for output_frame in super().process_frame(frame, direction):
            yield output_frame