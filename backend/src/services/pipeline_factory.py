"""
Pipeline factory for creating conversation pipelines.
"""
from typing import Optional, Tuple, Any
import logging

from src.domains.conversation import ConversationConfig
from src.config.settings import get_settings
from src.services.providers import (
    create_stt_service,
    create_llm_service,
    create_tts_service,
    create_llm_context
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams

logger = logging.getLogger(__name__)


class PipelineFactory:
    """Factory for creating conversation pipelines"""
    
    def __init__(self):
        self._settings = get_settings()
        self._service_cache = {}
    
    async def create_pipeline(
        self,
        config: ConversationConfig,
        transport: Any,
        conversation_id: str,
        aiohttp_session: Any
    ) -> Tuple[Pipeline, int]:
        """
        Create a conversation pipeline.
        Returns tuple of (pipeline, output_sample_rate)
        """
        # Create services using the factory functions
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
            aiohttp_session=aiohttp_session
        )
        
        # Create LLM context
        context_obj = create_llm_context(
            config.llm_provider,
            system_prompt=config.system_prompt
        )
        
        # Create context aggregator
        context_aggregator = llm.create_context_aggregator(context_obj)
        
        # Import processor here to avoid circular imports
        from src.processors.conversation_processors import ConversationProcessor
        
        # Create pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            ConversationProcessor(conversation_id),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        
        logger.info(f"Created pipeline for conversation {conversation_id}")
        return pipeline, output_sample_rate
    
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
        
        # Add any custom pipeline params
        if config.pipeline_params:
            for key, value in config.pipeline_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        
        return PipelineTask(pipeline, params=params)


# Global factory instance
_pipeline_factory: Optional[PipelineFactory] = None


def get_pipeline_factory() -> PipelineFactory:
    """Get the singleton pipeline factory"""
    global _pipeline_factory
    if _pipeline_factory is None:
        _pipeline_factory = PipelineFactory()
    return _pipeline_factory