"""
Pipeline factory for creating conversation pipelines with optional frame processors.
"""
from typing import Optional, Tuple, Any, List
import logging

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
from pipecat.processors.frame_processor import FrameProcessor
from src.processors.billing_processor import BillingProcessor

logger = logging.getLogger(__name__)


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
        enable_processors: bool = True,  # New parameter to control processors
        correlation_token: Optional[str] = None
    ) -> Tuple[Pipeline, int]:
        """
        Create a conversation pipeline with optional frame processors.
        
        Args:
            config: Conversation configuration
            transport: Transport implementation
            conversation_id: Unique conversation identifier
            aiohttp_session: HTTP session for services
            enable_processors: Whether to add conversation processors
            
        Returns:
            Tuple of (pipeline, output_sample_rate)
        """
        await self._event_bus.emit(
            f"pipeline:{conversation_id}:lifecycle:creating",
            conversation_id=conversation_id,
            config=config.to_dict(),
            processors_enabled=enable_processors
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
                speed=getattr(config, 'tts_speed', None),  # Add speed parameter
                temperature=getattr(config, 'tts_temperature', None),  # Add temperature
                aiohttp_session=aiohttp_session
            )
            
            context_obj = create_llm_context(
                config.llm_provider,
                system_prompt=config.system_prompt
            )
            
            context_aggregator = llm.create_context_aggregator(context_obj)
            
            # Set aggregation timeouts for low latency
            # These reduce the delay before processing user input and assistant responses
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
                f"(processors {'enabled' if enable_processors else 'disabled'})"
            )
            
            await self._event_bus.emit(
                f"pipeline:{conversation_id}:lifecycle:created",
                conversation_id=conversation_id,
                pipeline_id=id(pipeline),
                output_sample_rate=output_sample_rate,
                processors_enabled=enable_processors
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