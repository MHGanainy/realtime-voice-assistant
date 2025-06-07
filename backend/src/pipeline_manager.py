import os
from typing import Optional
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
import logging

logger = logging.getLogger(__name__)

# Global pipeline runner
pipeline_runner = None

def get_pipeline_runner():
    """Get or create the global pipeline runner"""
    global pipeline_runner
    if pipeline_runner is None:
        pipeline_runner = PipelineRunner()
    return pipeline_runner

def create_audio_pipeline(
    stt_service,
    llm_service,
    tts_service,
    transport=None,
    context=None
) -> Pipeline:
    """Create a basic audio processing pipeline"""
    
    # Create pipeline stages
    stages = []
    
    # Add transport input if provided
    if transport:
        stages.append(transport.input())
    
    # Add core services
    stages.extend([
        stt_service,
        # We'll add context aggregators later
        llm_service,
        tts_service,
    ])
    
    # Add transport output if provided
    if transport:
        stages.append(transport.output())
    
    return Pipeline(stages)

def create_pipeline_task(pipeline: Pipeline, *, out_rate: int) -> PipelineTask:
    logger.info(f"Creating pipeline task with output rate {out_rate}")
    return PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=out_rate,
            allow_interruptions=True,
            enable_metrics=True,
        )
    )