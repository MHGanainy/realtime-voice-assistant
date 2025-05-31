# backend/src/main.py
"""
Pipecat voice assistant with frontend communication via separate WebSocket
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import websockets
from typing import Set, Optional

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
from pipecat.frames.frames import (
    LLMMessagesFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    AudioRawFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
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


# Global set to track frontend WebSocket connections
frontend_clients: Set[websockets.WebSocketServerProtocol] = set()


class FrameLogger(FrameProcessor):
    """Custom frame processor for logging transcriptions and sending to frontend"""
    
    def __init__(self):
        super().__init__()
        self.assistant_reply = ""
        self.collecting_reply = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process and log frames"""
        # IMPORTANT: Always call parent's process_frame first
        await super().process_frame(frame, direction)
        
        try:
            # Log user transcriptions
            if isinstance(frame, TranscriptionFrame):
                logger.info(f"🎤 User: {frame.text}")
                # Send transcription to all frontend clients
                await self._broadcast_to_frontend({
                    "type": "transcription",
                    "text": frame.text,
                    "final": True
                })
            
            # Log interim transcriptions
            elif isinstance(frame, InterimTranscriptionFrame):
                logger.debug(f"... {frame.text}")
                await self._broadcast_to_frontend({
                    "type": "transcription",
                    "text": frame.text,
                    "final": False
                })
            
            # Capture assistant responses
            elif isinstance(frame, TextFrame):
                # Check if this is going downstream (to TTS)
                if hasattr(frame, 'text') and frame.text:
                    logger.info(f"🤖 Assistant: {frame.text}")
                    
                    if not self.collecting_reply:
                        self.collecting_reply = True
                        self.assistant_reply = ""
                    
                    self.assistant_reply += frame.text + " "
                    
                    # Send partial assistant response
                    await self._broadcast_to_frontend({
                        "type": "assistant_reply",
                        "text": self.assistant_reply.strip(),
                        "final": False
                    })
            
            # When audio starts being sent, mark reply as final
            elif isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
                if self.collecting_reply and self.assistant_reply:
                    await self._broadcast_to_frontend({
                        "type": "assistant_reply",
                        "text": self.assistant_reply.strip(),
                        "final": True
                    })
                    self.collecting_reply = False
                    self.assistant_reply = ""
                    
        except Exception as e:
            logger.error(f"Error in FrameLogger: {e}")
        
        # IMPORTANT: Forward the frame with direction
        await self.push_frame(frame, direction)
    
    async def _broadcast_to_frontend(self, message: dict):
        """Send message to all connected frontend WebSocket clients"""
        if frontend_clients:
            message_str = json.dumps(message)
            disconnected = set()
            
            for client in frontend_clients:
                try:
                    await client.send(message_str)
                    logger.debug(f"Sent to frontend: {message['type']}")
                except Exception as e:
                    logger.error(f"Error sending to frontend client: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            frontend_clients.difference_update(disconnected)
    
class AssistantResponseTracker(FrameProcessor):
    """Track and send assistant responses to frontend"""
    
    def __init__(self):
        super().__init__()
        self.current_response = []
        self.response_timer = None
        self.MIN_SILENCE_DURATION = 0.5  # 500ms of silence to consider response complete
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Collect text frames
        if isinstance(frame, TextFrame):
            # Cancel any existing timer
            if self.response_timer:
                self.response_timer.cancel()
            
            # Add text to current response
            self.current_response.append(frame.text)
            
            # Send partial update with all accumulated text
            current_text = ''.join(self.current_response)
            logger.debug(f"Assistant partial: {current_text}")
            
            await self._broadcast_to_frontend({
                "type": "assistant_reply", 
                "text": current_text,
                "final": False
            })
            
            # Set timer to finalize response after silence
            self.response_timer = asyncio.create_task(self._finalize_response())
        
        await self.push_frame(frame, direction)
    
    async def _finalize_response(self):
        """Finalize response after a period of silence"""
        try:
            # Wait for silence duration
            await asyncio.sleep(self.MIN_SILENCE_DURATION)
            
            # If we get here, no new text came in during the wait
            if self.current_response:
                final_text = ''.join(self.current_response)
                logger.info(f"🤖 Assistant complete: {final_text}")
                
                await self._broadcast_to_frontend({
                    "type": "assistant_reply",
                    "text": final_text,
                    "final": True
                })
                
                # Clear the response buffer
                self.current_response = []
                
        except asyncio.CancelledError:
            # Timer was cancelled because new text arrived
            pass
    
    async def _broadcast_to_frontend(self, message: dict):
        """Send message to all connected frontend WebSocket clients"""
        if frontend_clients:
            message_str = json.dumps(message)
            disconnected = set()
            
            for client in frontend_clients:
                try:
                    await client.send(message_str)
                except Exception as e:
                    logger.error(f"Error sending to frontend client: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            frontend_clients.difference_update(disconnected)


async def handle_frontend_connection(websocket, path):
    """Handle frontend WebSocket connections (separate from audio)"""
    frontend_clients.add(websocket)
    logger.info(f"Frontend client connected: {websocket.remote_address}")
    
    try:
        # Send initial connection message
        await websocket.send(json.dumps({
            "type": "connection",
            "status": "connected"
        }))
        
        # Keep connection alive and handle any messages from frontend
        async for message in websocket:
            logger.debug(f"Received from frontend: {message}")
            # You can handle frontend commands here if needed
            
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Frontend client disconnected: {websocket.remote_address}")
    finally:
        frontend_clients.remove(websocket)


async def run_frontend_server():
    """Run the frontend WebSocket server on port 8766"""
    logger.info("Starting frontend WebSocket server on ws://localhost:8766")
    await websockets.serve(handle_frontend_connection, "localhost", 8766)


async def main():
    # Start the frontend WebSocket server
    frontend_server = asyncio.create_task(run_frontend_server())
    
    # Create transport for audio/voice pipeline
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            host="0.0.0.0",
            port=8765,
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_enabled=True,
            vad_audio_passthrough=True,
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
    
    # Create frame logger
    frame_logger = FrameLogger()
    
    # Create assistant response tracker
    assistant_tracker = AssistantResponseTracker()
    
    # Initial messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don't include special characters in your answers. Start by greeting the user when they connect.",
        },
    ]
    
    # Create aggregators
    user_response = LLMUserResponseAggregator(messages)
    assistant_response = LLMAssistantResponseAggregator(messages)
    
    # Build pipeline with frame logger and assistant tracker
    pipeline = Pipeline(
        [
            transport.input(),      # WebSocket input from client
            stt,                   # Speech-To-Text
            frame_logger,          # Log and send transcriptions to frontend
            user_response,         # Aggregate user response
            llm,                   # LLM
            assistant_tracker,     # Track and send assistant responses
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
        logger.info(f"Audio client connected from {client.remote_address}")
        # Don't add another greeting - let the initial system prompt handle it
        await task.queue_frames([LLMMessagesFrame(messages)])
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Audio client disconnected: {client.remote_address}")
    
    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
    
    # Run the pipeline
    runner = PipelineRunner()
    
    try:
        await runner.run(task)
    finally:
        # Cancel the frontend server when pipeline stops
        frontend_server.cancel()


if __name__ == "__main__":
    logger.info("Starting Voice Assistant Backend")
    logger.info("Audio WebSocket server on ws://localhost:8765")
    logger.info("Frontend WebSocket server on ws://localhost:8766")
    asyncio.run(main())