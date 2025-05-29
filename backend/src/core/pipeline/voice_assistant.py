import asyncio
import logging
from typing import AsyncIterator, Optional, Dict, Any, List, Callable
from contextlib import AsyncExitStack
from ..interfaces.stt_base import STTProvider
from ..interfaces.llm_base import LLMProvider, Message
from ..interfaces.tts_base import TTSProvider
from ..events.types import *
from ..events.bus import event_bus
from ..commands.websocket_commands import Command
from .middleware import MiddlewarePipeline
from ...utils.decorators import measure_performance

logger = logging.getLogger(__name__)

class VoiceAssistantPipeline:
    """
    Main pipeline orchestrating STT -> LLM -> TTS flow using Template Method pattern.
    The high-level algorithm is defined here, with specific steps delegated to providers.
    """
    
    def __init__(
        self,
        stt_provider: STTProvider,
        llm_provider: LLMProvider,
        tts_provider: TTSProvider,
        middleware_pipeline: Optional[MiddlewarePipeline] = None,
        use_event_bus: bool = True
    ):
        self.stt = stt_provider
        self.llm = llm_provider
        self.tts = tts_provider
        self.middleware = middleware_pipeline or MiddlewarePipeline()
        self.use_event_bus = use_event_bus
        
        # State management
        self.is_processing = False
        self.is_paused = False
        self.transcript_buffer: List[str] = []
        self._tasks: List[asyncio.Task] = []
        self._correlation_id = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    async def initialize(self) -> None:
        """Initialize all providers and start event bus"""
        async with AsyncExitStack() as stack:
            # Initialize providers using context managers
            await stack.enter_async_context(self.stt)
            await stack.enter_async_context(self.llm)
            await stack.enter_async_context(self.tts)
            
            # Transfer ownership to self
            self._exit_stack = stack.pop_all()
            
        # Start event bus if enabled
        if self.use_event_bus:
            await event_bus.start()
            
        await self._publish_event(Event(
            type=EventType.PIPELINE_START,
            source="pipeline",
            data={"status": "initialized"}
        ))
        
        logger.info("Voice assistant pipeline initialized")
        
    async def cleanup(self) -> None:
        """Cleanup all resources"""
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        # Cleanup providers
        if hasattr(self, '_exit_stack'):
            await self._exit_stack.aclose()
            
        # Stop event bus
        if self.use_event_bus:
            await event_bus.stop()
            
        await self._publish_event(Event(
            type=EventType.PIPELINE_END,
            source="pipeline",
            data={"status": "cleaned_up"}
        ))
        
        logger.info("Voice assistant pipeline cleaned up")
        
    async def pause(self):
        """Pause processing"""
        self.is_paused = True
        logger.info("Pipeline paused")
        
    async def resume(self):
        """Resume processing"""
        self.is_paused = False
        logger.info("Pipeline resumed")
        
    async def stop(self):
        """Stop pipeline"""
        await self.cleanup()
        
    @measure_performance
    async def run(
        self,
        audio_source: AsyncIterator[bytes],
        event_handler: Optional[Callable[[Event], None]] = None
    ) -> None:
        """
        Template method defining the high-level pipeline algorithm.
        Specific steps are delegated to providers and middleware.
        """
        self._correlation_id = f"session_{asyncio.get_running_loop().time()}"
        
        try:
            # Process audio through STT
            async for transcript_event in self._process_stt(audio_source):
                if self.is_paused:
                    continue
                    
                # Handle transcript through middleware
                processed_event = await self._process_middleware(transcript_event)
                
                # Emit event
                if event_handler:
                    await event_handler(processed_event)
                    
                # Check for utterance completion
                # Clear the buffer when we get a speech_final event
                if transcript_event.speech_final and not self.is_processing:
                    # Process complete utterance immediately
                    await self._process_utterance(event_handler)
                    
        except Exception as e:
            await self._handle_error(e, "pipeline_run", event_handler)
            
    async def _process_stt(
        self, 
        audio_source: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        """Process audio through STT provider"""
        # Log that we're starting STT processing
        logger.debug("Starting STT processing")
        
        # Stream audio through STT provider
        async for event in self.stt.stream(audio_source):
            event.correlation_id = self._correlation_id
            await self._publish_event(event)
            
            # Only add final transcripts to buffer
            if event.is_final and event.transcript:
                self.transcript_buffer.append(event.transcript)
                logger.info(f"Added to buffer: '{event.transcript}' (buffer size: {len(self.transcript_buffer)})")
                
            yield event
            
    async def _process_middleware(self, event: Event) -> Event:
        """Process event through middleware pipeline"""
        # For now, just return the event
        # Middleware processing would go here
        return event
        
    async def _process_utterance(
        self, 
        event_handler: Optional[Callable[[Event], None]]
    ) -> None:
        """Process complete utterance through LLM and TTS"""
        if not self.transcript_buffer:
            return
            
        # Get the complete utterance and immediately clear the buffer
        utterance = " ".join(self.transcript_buffer)
        self.transcript_buffer.clear()  # Clear immediately to prevent accumulation
        logger.info(f"Processing utterance: '{utterance}'")
        
        # Skip if utterance is empty after stripping
        if not utterance.strip():
            return
            
        self.is_processing = True
        
        try:
            # Emit utterance end event
            await self._publish_event(Event(
                type=EventType.UTTERANCE_END,
                source="pipeline",
                data={"utterance": utterance},
                correlation_id=self._correlation_id
            ))
            
            # Signal pause
            if event_handler:
                from ..commands.websocket_commands import PauseCommand
                pause_cmd = PauseCommand(reason="processing_utterance")
                result = await pause_cmd.execute({"pipeline": self})
                await event_handler(Event(
                    type=EventType.COMMAND,
                    source="pipeline",
                    data=result,
                    correlation_id=self._correlation_id
                ))
            
            # Process through LLM
            response = await self._process_llm(utterance, event_handler)
            
            # Process through TTS
            await self._process_tts(response, event_handler)
            
            # Signal resume
            if event_handler:
                from ..commands.websocket_commands import ResumeCommand
                resume_cmd = ResumeCommand(reason="processing_complete")
                result = await resume_cmd.execute({"pipeline": self})
                await event_handler(Event(
                    type=EventType.COMMAND,
                    source="pipeline",
                    data=result,
                    correlation_id=self._correlation_id
                ))
                
        except Exception as e:
            await self._handle_error(e, "utterance_processing", event_handler)
        finally:
            self.is_processing = False
            
    async def _process_llm(
        self,
        utterance: str,
        event_handler: Optional[Callable[[Event], None]]
    ) -> str:
        """
        Send the user’s utterance to the LLM, stream tokens out,
        and deliver the finished answer both to the event-bus *and*
        to the WebSocket handler (so the React UI can show it).
        """
        # -------- mark request start ---------------------------------
        await self._publish_event(Event(
            type=EventType.LLM_REQUEST_START,
            source="llm",
            data={"utterance": utterance},
            correlation_id=self._correlation_id
        ))

        # -------- conversational history -----------------------------
        self.llm.add_to_history(Message("user", utterance))

        # -------- stream response tokens -----------------------------
        response_tokens: list[str] = []
        token_count = 0

        async for token in self.llm.stream(utterance):
            response_tokens.append(token)
            token_count += 1

            # live token event (optional, but nice for metrics / typing-effect)
            token_event = LLMTokenEvent(
                source="llm",
                token=token,
                token_count=token_count,
                data={"token": token},
                correlation_id=self._correlation_id
            )
            await self._publish_event(token_event)

        complete_response = "".join(response_tokens)

        # -------- update history -------------------------------------
        self.llm.add_to_history(Message("assistant", complete_response))

        # -------- build ONE completion event -------------------------
        llm_done = Event(
            type=EventType.LLM_COMPLETE,
            source="llm",
            data={
                "utterance": utterance,
                "response": complete_response,
                "token_count": token_count
            },
            correlation_id=self._correlation_id
        )

        # 1) event-bus (logging, metrics, etc.)
        await self._publish_event(llm_done)

        # 2) WebSocket → frontend
        if event_handler:
            await event_handler(llm_done)

        return complete_response

        
    async def _process_tts(
        self,
        text: str,
        event_handler: Optional[Callable[[Event], None]]
    ) -> None:
        """Process text through TTS"""
        await self._publish_event(Event(
            type=EventType.TTS_REQUEST_START,
            source="tts",
            data={"text": text},
            correlation_id=self._correlation_id
        ))
        
        # Create text stream
        async def text_stream():
            yield text
            
        # Stream audio
        chunk_count = 0
        async for audio_chunk in self.tts.stream(text_stream()):
            chunk_event = AudioChunkEvent(
                source="tts",
                audio_chunk=audio_chunk,
                chunk_index=chunk_count,
                data={"chunk_size": len(audio_chunk)},
                correlation_id=self._correlation_id
            )
            
            await self._publish_event(chunk_event)
            
            if event_handler:
                await event_handler(chunk_event)
                
            chunk_count += 1
            
        # Emit completion event
        await self._publish_event(Event(
            type=EventType.TTS_COMPLETE,
            source="tts",
            data={"chunk_count": chunk_count},
            correlation_id=self._correlation_id
        ))
        
    async def _publish_event(self, event: Event):
        """Publish event to event bus if enabled"""
        if self.use_event_bus:
            await event_bus.publish(event)
            
    async def _handle_error(
        self,
        error: Exception,
        phase: str,
        event_handler: Optional[Callable[[Event], None]]
    ):
        """Handle errors in the pipeline"""
        error_event = ErrorEvent(
            source="pipeline",
            error_message=str(error),
            error_type=type(error).__name__,
            data={
                "phase": phase,
                "error": str(error)
            },
            correlation_id=self._correlation_id
        )
        
        await self._publish_event(error_event)
        
        if event_handler:
            await event_handler(error_event)
            
        logger.error(f"Pipeline error in {phase}: {error}", exc_info=True)