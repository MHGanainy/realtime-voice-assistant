"""
Pipeline factory for creating conversation pipelines with optional frame processors.
"""
from typing import Optional, Tuple, Any, List
import logging
import asyncio

from src.domains.conversation import ConversationConfig
from src.config.settings import get_settings
from src.services.providers import (
    create_stt_service,
    create_llm_service,
    create_tts_service,
    create_llm_context
)
from src.events import get_event_bus
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, StartFrame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from src.processors.billing_processor import BillingProcessor

logger = logging.getLogger(__name__)


class OpeningLineProcessor(FrameProcessor):
    """Processor that injects an opening line at conversation start"""
    
    def __init__(self, opening_line: str, conversation_id: str):
        super().__init__()
        self.opening_line = opening_line
        self.conversation_id = conversation_id
        self._sent = False
        self._started = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Detect when pipeline has started
        if isinstance(frame, StartFrame) and not self._started:
            self._started = True
            logger.info(f"[OpeningLine] Pipeline started for conversation {self.conversation_id}")
            
            # Schedule the opening line to be sent after a brief delay
            asyncio.create_task(self._send_opening_after_delay())
        
        # Always pass the original frame through
        await self.push_frame(frame, direction)
    
    async def _send_opening_after_delay(self):
        """Send opening line after a brief delay to ensure pipeline is ready"""
        if self._sent:
            return
            
        # Wait for pipeline to stabilize
        if not self._sent:
            self._sent = True
            
            logger.info(f"[OpeningLine] Injecting opening line: {self.opening_line[:100]}...")
            
            # Send frames that mimic LLM response to trigger TTS properly
            # Start frame to indicate assistant is starting to speak
            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            
            # The actual text content
            await self.push_frame(TextFrame(text=self.opening_line), FrameDirection.DOWNSTREAM)
            
            # End frame to indicate assistant has finished
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            
            logger.info(f"[OpeningLine] Opening line frames injected successfully")


class PipelineFactory:
    """Factory for creating conversation pipelines"""
    
    def __init__(self):
        self._settings = get_settings()
        self._event_bus = get_event_bus()
        self._service_cache = {}
    
    async def create_pipeline(
        self,
        config: ConversationConfig,
        transport: Any,
        conversation_id: str,
        aiohttp_session: Any,
        enable_processors: bool = True,
        correlation_token: Optional[str] = None,
        opening_line: Optional[str] = None  # New parameter for opening line
    ) -> Tuple[Pipeline, int]:
        """
        Create a conversation pipeline with optional frame processors.
        
        Args:
            config: Conversation configuration
            transport: Transport implementation
            conversation_id: Unique conversation identifier
            aiohttp_session: HTTP session for services
            enable_processors: Whether to add conversation processors
            correlation_token: Token for billing/tracking
            opening_line: Optional opening line to speak at start
            
        Returns:
            Tuple of (pipeline, output_sample_rate)
        """
        await self._event_bus.emit(
            f"pipeline:{conversation_id}:lifecycle:creating",
            conversation_id=conversation_id,
            config=config.to_dict(),
            processors_enabled=enable_processors,
            has_opening_line=bool(opening_line)
        )
        
        try:
            # Create services
            stt = create_stt_service(
                config.stt_provider,
                model=config.stt_model,
                vad_events=config.vad_enabled
            )
            
            llm = create_llm_service(
                config.llm_provider,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                top_p=config.llm_top_p
            )
            
            tts, output_sample_rate = create_tts_service(
                config.tts_provider,
                model=config.tts_model,
                voice_id=config.tts_voice,
                speed=getattr(config, 'tts_speed', None),
                temperature=getattr(config, 'tts_temperature', None),
                aiohttp_session=aiohttp_session
            )
            
            context_obj = create_llm_context(
                config.llm_provider,
                system_prompt=config.system_prompt
            )
            
            context_aggregator = llm.create_context_aggregator(context_obj)
            
            # Set aggregation timeouts for low latency
            if hasattr(context_aggregator.user(), 'aggregation_timeout'):
                context_aggregator.user().aggregation_timeout = 0.2
                logger.info(f"Set user aggregation timeout to 0.2s for conversation {conversation_id}")
            
            if hasattr(context_aggregator.assistant(), 'bot_interruption_timeout'):
                context_aggregator.assistant().bot_interruption_timeout = 0.2
                logger.info(f"Set assistant interruption timeout to 0.2s for conversation {conversation_id}")
            
            # Build pipeline components
            pipeline_components = []
            
            # Add transport input
            pipeline_components.append(transport.input())
    
            # Add billing processor if correlation token is available
            if correlation_token:
                logger.info(
                    f"[PIPELINE] Adding BillingProcessor | "
                    f"conversation_id={conversation_id} | "
                    f"correlation_token={correlation_token}"
                )
                pipeline_components.append(
                    BillingProcessor(
                        conversation_id=conversation_id,
                        correlation_token=correlation_token,
                        transport=transport
                    )
                )
                logger.info(
                    f"[PIPELINE] BillingProcessor added successfully | "
                    f"conversation_id={conversation_id}"
                )
            
            # Optionally add input processor
            if enable_processors:
                from src.processors.processor import create_conversation_processor
                pipeline_components.append(create_conversation_processor(conversation_id, "input"))
            
            # Add STT
            pipeline_components.append(stt)
            
            # Optionally add post-STT processor
            if enable_processors:
                pipeline_components.append(create_conversation_processor(conversation_id, "post-stt"))
            
            # Add context aggregator user
            pipeline_components.append(context_aggregator.user())
            
            # Add LLM
            pipeline_components.append(llm)
            
            # Optionally add post-LLM processor
            if enable_processors:
                pipeline_components.append(create_conversation_processor(conversation_id, "post-llm"))
            
            # Add opening line processor AFTER LLM, BEFORE TTS
            # This ensures the text flows directly to TTS
            if opening_line:
                logger.info(
                    f"[PIPELINE] Adding OpeningLineProcessor | "
                    f"conversation_id={conversation_id} | "
                    f"opening_line_length={len(opening_line)}"
                )
                pipeline_components.append(
                    OpeningLineProcessor(
                        opening_line=opening_line,
                        conversation_id=conversation_id
                    )
                )
            
            # Add TTS
            pipeline_components.append(tts)
            
            # Optionally add post-TTS processor
            if enable_processors:
                pipeline_components.append(create_conversation_processor(conversation_id, "post-tts"))
            
            # Add transport output
            pipeline_components.append(transport.output())
            
            # Add context aggregator assistant
            pipeline_components.append(context_aggregator.assistant())
            
            # Create pipeline
            pipeline = Pipeline(pipeline_components)
            
            logger.info(
                f"Created pipeline for conversation {conversation_id} "
                f"(processors {'enabled' if enable_processors else 'disabled'}, "
                f"opening_line {'present' if opening_line else 'none'})"
            )
            
            await self._event_bus.emit(
                f"pipeline:{conversation_id}:lifecycle:created",
                conversation_id=conversation_id,
                pipeline_id=id(pipeline),
                output_sample_rate=output_sample_rate,
                processors_enabled=enable_processors,
                has_opening_line=bool(opening_line)
            )
            
            return pipeline, output_sample_rate
            
        except Exception as e:
            await self._event_bus.emit(
                f"pipeline:{conversation_id}:error:creation",
                conversation_id=conversation_id,
                error_type="pipeline_creation_error",
                error_message=str(e)
            )
            raise
    
    def create_pipeline_task(
        self,
        pipeline: Pipeline,
        config: ConversationConfig,
        output_sample_rate: int
    ) -> PipelineTask:
        """Create a pipeline task with appropriate parameters"""
        vad_analyzer = None
        if config.vad_enabled:
            vad_analyzer = SileroVADAnalyzer(
                params=VADParams(threshold=config.vad_threshold)
            )
        
        params = PipelineParams(
            audio_in_sample_rate=config.sample_rate,
            audio_out_sample_rate=output_sample_rate,
            allow_interruptions=config.enable_interruptions,
            enable_metrics=self._settings.enable_metrics,
            vad_enabled=config.vad_enabled,
            vad_analyzer=vad_analyzer
        )
        
        if config.pipeline_params:
            for key, value in config.pipeline_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        
        return PipelineTask(pipeline, params=params)


_pipeline_factory: Optional[PipelineFactory] = None


def get_pipeline_factory() -> PipelineFactory:
    """Get the singleton pipeline factory"""
    global _pipeline_factory
    if _pipeline_factory is None:
        _pipeline_factory = PipelineFactory()
    return _pipeline_factory