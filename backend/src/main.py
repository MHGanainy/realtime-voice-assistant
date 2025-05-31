# backend/src/main.py
"""
Pipecat voice assistant using WebsocketServerTransport
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Setup logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator
)
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

if not all([OPENAI_API_KEY, ELEVEN_API_KEY]):
    raise ValueError("Missing required API keys. Please set OPENAI_API_KEY and ELEVEN_API_KEY")


async def main():
    # Create transport
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            host="0.0.0.0",
            port=8765,
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_analyzer=SileroVADAnalyzer(),
            session_timeout=60 * 3,  # 3 minutes
        )
    )
    
    # Create services
    stt = OpenAISTTService(
        api_key=OPENAI_API_KEY,
        model="whisper-1"
    )
    
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo"
    )
    
    tts = ElevenLabsTTSService(
        api_key=ELEVEN_API_KEY,
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model="eleven_flash_v2_5"
    )
    
    # Initial messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don't include special characters in your answers.",
        },
    ]
    
    # Create aggregators
    user_response = LLMUserResponseAggregator(messages)
    assistant_response = LLMAssistantResponseAggregator(messages)
    
    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),      # WebSocket input from client
            stt,                   # Speech-To-Text
            user_response,         # Aggregate user response
            llm,                   # LLM
            tts,                   # Text-To-Speech
            transport.output(),    # WebSocket output to client
            assistant_response,    # Update conversation context
        ]
    )
    
    # Create task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            allow_interruptions=True,
        ),
    )
    
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected from {client.remote_address}")
        # Greet the user
        messages.append({"role": "system", "content": "Greet the user and ask how you can help them today."})
        await task.queue_frames([LLMMessagesFrame(messages)])
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client.remote_address}")
    
    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
    
    # Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    logger.info("Starting WebSocket server on ws://localhost:8765")
    asyncio.run(main())