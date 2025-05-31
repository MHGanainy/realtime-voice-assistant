# backend/src/main.py
"""
Pipecat voice assistant with frontend communication via separate WebSocket
"""
import asyncio
import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import websockets
from typing import Set, Optional, List, Dict
from datetime import datetime
import time

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Setup logging with custom handler
class MetricsLogHandler:
    """Custom log handler to capture metrics from Pipecat logs"""
    def __init__(self):
        self.last_stt_time = None
        self.last_llm_time = None
        self.last_tts_time = None
        
    def handle(self, message):
        global latency_metrics
        
        try:
            # The message parameter is already a string
            # Debug: Log all messages that contain service names and TTFB
            if "Service#" in message and "TTFB:" in message:
                print(f"[METRICS DEBUG] Found TTFB message: {message}")
            
            # Check if this is a metrics log
            if "TTFB:" in message:
                # Extract TTFB values with updated regex pattern
                match = re.search(r'(\w+Service)#\d+ TTFB: ([\d.]+)', message)
                if match:
                    service_type = match.group(1)
                    ttfb_value = float(match.group(2))
                    ttfb_ms = int(ttfb_value * 1000)
                    
                    print(f"[METRICS DEBUG] Parsed service: {service_type}, TTFB: {ttfb_ms}ms")
                    
                    if "STTService" in service_type:  # Changed to catch both OpenAI and Deepgram
                        latency_metrics["stt"] = ttfb_ms
                        self.last_stt_time = time.time()
                        print(f"[METRICS] Captured STT latency: {ttfb_ms}ms")
                        
                    elif "OpenAILLMService" in service_type:
                        latency_metrics["llm"] = ttfb_ms
                        self.last_llm_time = time.time()
                        print(f"[METRICS] Captured LLM latency: {ttfb_ms}ms")
                        
                    elif "TTSService" in service_type:  # Changed to catch both ElevenLabs and Deepgram
                        latency_metrics["tts"] = ttfb_ms
                        self.last_tts_time = time.time()
                        print(f"[METRICS] Captured TTS latency: {ttfb_ms}ms")
                        
                        # Calculate total and send update
                        if latency_metrics["interaction_start"]:
                            total_time = int((time.time() - latency_metrics["interaction_start"]) * 1000)
                            latency_metrics["total"] = total_time
                            print(f"[METRICS] Total interaction time: {total_time}ms")
                            
                            # Send to frontend
                            asyncio.create_task(broadcast_latencies())
                            print(f"[METRICS] Broadcasting latencies: {latency_metrics}")
                            
                            # Reset interaction start for next interaction
                            latency_metrics["interaction_start"] = None
        except Exception as e:
            print(f"[METRICS ERROR] Error in metrics handler: {e}")
            import traceback
            traceback.print_exc()

metrics_handler = MetricsLogHandler()

async def broadcast_latencies():
    """Broadcast latency metrics to all frontend clients"""
    message = {
        "type": "latency_update",
        "latencies": {
            "stt": latency_metrics["stt"],
            "llm": latency_metrics["llm"],
            "tts": latency_metrics["tts"],
            "total": latency_metrics["total"]
        }
    }
    await broadcast_to_all_clients(message)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
logger.add(metrics_handler.handle, level="DEBUG", filter=lambda record: True)

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
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
    AudioRawFrame,
    EndFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not all([OPENAI_API_KEY, ELEVEN_API_KEY, DEEPGRAM_API_KEY]):
    raise ValueError("Missing required API keys. Please set OPENAI_API_KEY, ELEVEN_API_KEY, and DEEPGRAM_API_KEY")


# Global set to track frontend WebSocket connections
frontend_clients: Set[websockets.WebSocketServerProtocol] = set()

# Global conversation history
conversation_history: List[Dict[str, str]] = []

# Global system prompt
current_system_prompt = "You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don't include special characters in your answers."

# Global pipeline components
pipeline_task = None
context = None
context_aggregator = None
current_stt_service = "openai"  # Track current STT service
current_llm_model = "gpt-3.5-turbo"  # Track current LLM model
current_tts_service = "elevenlabs"  # Track current TTS service
stt_service = None  # Global STT service reference
llm_service = None  # Global LLM service reference
tts_service = None  # Global TTS service reference
pipeline_runner = None  # Global pipeline runner reference

# Global latency tracking
latency_metrics = {
    "stt": 0,
    "llm": 0,
    "tts": 0,
    "total": 0,
    "interaction_start": None
}


class TranscriptionCapture(FrameProcessor):
    """Capture user transcriptions immediately after STT"""
    
    def __init__(self):
        super().__init__()
        self.current_transcription = ""
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        global conversation_history, latency_metrics
        
        try:
            # Mark start of interaction on first interim transcription
            if isinstance(frame, InterimTranscriptionFrame):
                if not latency_metrics["interaction_start"]:
                    latency_metrics["interaction_start"] = time.time()
                    logger.info("Interaction started (first interim transcription)")
                    print(f"[METRICS] Interaction started at: {latency_metrics['interaction_start']}")
                
                logger.debug(f"Interim: {frame.text}")
                await self._broadcast_to_frontend({
                    "type": "transcription",
                    "text": frame.text,
                    "final": False
                })
                
            # Capture final transcriptions
            elif isinstance(frame, TranscriptionFrame):
                # If we didn't catch the start from interim, mark it now
                if not latency_metrics["interaction_start"]:
                    # Estimate start time based on typical STT processing time
                    # This is a fallback since OpenAI Whisper doesn't send interim transcriptions
                    latency_metrics["interaction_start"] = time.time() - 2.0  # Assume ~2 seconds of audio
                    logger.info("Interaction start estimated from final transcription")
                    print(f"[METRICS] Interaction start estimated at: {latency_metrics['interaction_start']}")
                
                logger.info(f"🎤 User said: {frame.text}")
                self.current_transcription = frame.text
                
                # Send to frontend
                await self._broadcast_to_frontend({
                    "type": "transcription",
                    "text": frame.text,
                    "final": True
                })
                
                # Add to conversation history
                conversation_history.append({
                    "role": "user",
                    "content": frame.text,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send updated history
                await self._broadcast_to_frontend({
                    "type": "conversation_history",
                    "history": conversation_history
                })
                
        except Exception as e:
            logger.error(f"Error in TranscriptionCapture: {e}")
        
        await self.push_frame(frame, direction)
    
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


class AssistantResponseCapture(FrameProcessor):
    """Capture assistant responses from LLM output"""
    
    def __init__(self):
        super().__init__()
        self.response_buffer = []
        self.collecting = False
        self.finalize_task = None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        global conversation_history
        
        try:
            # Capture TextFrames going downstream (from LLM to TTS)
            if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
                if frame.text:
                    logger.debug(f"Assistant text: {frame.text}")
                    
                    # Cancel any pending finalization
                    if self.finalize_task:
                        self.finalize_task.cancel()
                    
                    # Start collecting if needed
                    if not self.collecting:
                        self.collecting = True
                        self.response_buffer = []
                    
                    # Add to buffer
                    self.response_buffer.append(frame.text)
                    current_text = ''.join(self.response_buffer)
                    
                    # Send partial response
                    await self._broadcast_to_frontend({
                        "type": "assistant_reply",
                        "text": current_text,
                        "final": False
                    })
                    
                    # Schedule finalization
                    self.finalize_task = asyncio.create_task(self._finalize_response())
                    
        except Exception as e:
            logger.error(f"Error in AssistantResponseCapture: {e}")
        
        await self.push_frame(frame, direction)
    
    async def _finalize_response(self):
        """Finalize response after a delay"""
        try:
            await asyncio.sleep(1.0)
            
            if self.collecting and self.response_buffer:
                final_text = ''.join(self.response_buffer)
                logger.info(f"🤖 Assistant said: {final_text}")
                
                # Send final response
                await self._broadcast_to_frontend({
                    "type": "assistant_reply",
                    "text": final_text,
                    "final": True
                })
                
                # Add to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": final_text,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send updated history
                await self._broadcast_to_frontend({
                    "type": "conversation_history",
                    "history": conversation_history
                })
                
                # Reset state
                self.collecting = False
                self.response_buffer = []
                
        except asyncio.CancelledError:
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
    global current_system_prompt, conversation_history, context, context_aggregator, pipeline_task, current_stt_service, current_llm_model, current_tts_service
    
    frontend_clients.add(websocket)
    logger.info(f"Frontend client connected: {websocket.remote_address}")
    
    try:
        # Send initial connection message
        await websocket.send(json.dumps({
            "type": "connection",
            "status": "connected"
        }))
        
        # Send current system prompt
        await websocket.send(json.dumps({
            "type": "system_prompt",
            "prompt": current_system_prompt
        }))
        
        # Send conversation history
        await websocket.send(json.dumps({
            "type": "conversation_history",
            "history": conversation_history
        }))
        
        # Send current latencies
        await websocket.send(json.dumps({
            "type": "latency_update",
            "latencies": {
                "stt": latency_metrics["stt"],
                "llm": latency_metrics["llm"],
                "tts": latency_metrics["tts"],
                "total": latency_metrics["total"]
            }
        }))
        
        # Send current STT service
        await websocket.send(json.dumps({
            "type": "stt_service",
            "service": current_stt_service
        }))
        
        # Send current LLM model
        await websocket.send(json.dumps({
            "type": "llm_model",
            "model": current_llm_model
        }))
        
        # Send current TTS service
        await websocket.send(json.dumps({
            "type": "tts_service",
            "service": current_tts_service
        }))
        
        # Handle messages from frontend
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.debug(f"Received from frontend: {data}")
                
                if data.get("type") == "update_system_prompt":
                    # Update system prompt
                    new_prompt = data.get("prompt", "")
                    if new_prompt:
                        current_system_prompt = new_prompt
                        logger.info(f"System prompt updated: {current_system_prompt[:50]}...")
                        
                        # Update the context with new system prompt
                        if context and context_aggregator:
                            # Reset context to only system message
                            new_messages = [
                                {
                                    "role": "system",
                                    "content": current_system_prompt,
                                }
                            ]
                            context.set_messages(new_messages)
                            
                            logger.info("Reset LLM context with new system prompt")
                        
                        # Broadcast the update to all clients
                        await broadcast_to_all_clients({
                            "type": "system_prompt",
                            "prompt": current_system_prompt
                        })
                        
                        # Clear conversation history when prompt is updated
                        conversation_history.clear()
                        await broadcast_to_all_clients({
                            "type": "conversation_history",
                            "history": conversation_history
                        })
                
                elif data.get("type") == "clear_history":
                    # Clear conversation history
                    conversation_history.clear()
                    logger.info("Conversation history cleared")
                    
                    # Reset context to just system prompt
                    if context:
                        new_messages = [
                            {
                                "role": "system",
                                "content": current_system_prompt,
                            }
                        ]
                        context.set_messages(new_messages)
                        
                        # Send context update
                        if pipeline_task and context_aggregator:
                            await pipeline_task.queue_frames([context_aggregator.user().get_context_frame()])
                    
                    # Broadcast the update
                    await broadcast_to_all_clients({
                        "type": "conversation_history",
                        "history": conversation_history
                    })
                
                elif data.get("type") == "change_stt_service":
                    # Change STT service
                    new_service = data.get("service", "openai")
                    if new_service in ["openai", "deepgram"] and new_service != current_stt_service:
                        current_stt_service = new_service
                        logger.info(f"STT service changed to: {current_stt_service}")
                        
                        # Broadcast the update to all clients
                        await broadcast_to_all_clients({
                            "type": "stt_service",
                            "service": current_stt_service
                        })
                        
                        # Note: The actual STT service will be changed when a new recording starts
                        await broadcast_to_all_clients({
                            "type": "notification",
                            "message": f"STT service changed to {current_stt_service.title()}. This will take effect on the next recording."
                        })
                    else:
                        logger.debug(f"STT service already set to {new_service} or invalid service name")
                
                elif data.get("type") == "change_llm_model":
                    # Change LLM model
                    new_model = data.get("model", "gpt-3.5-turbo")
                    if new_model in ["gpt-3.5-turbo", "gpt-4o-mini"] and new_model != current_llm_model:
                        current_llm_model = new_model
                        logger.info(f"LLM model changed to: {current_llm_model}")
                        
                        # Broadcast the update to all clients
                        await broadcast_to_all_clients({
                            "type": "llm_model",
                            "model": current_llm_model
                        })
                        
                        # Note: The actual LLM service will be changed when a new recording starts
                        model_display_name = "GPT-3.5 Turbo" if new_model == "gpt-3.5-turbo" else "GPT-4o Mini"
                        await broadcast_to_all_clients({
                            "type": "notification",
                            "message": f"LLM model changed to {model_display_name}. This will take effect on the next recording."
                        })
                    else:
                        logger.debug(f"LLM model already set to {new_model} or invalid model name")
                
                elif data.get("type") == "change_tts_service":
                    # Change TTS service
                    new_service = data.get("service", "elevenlabs")
                    if new_service in ["elevenlabs", "deepgram"] and new_service != current_tts_service:
                        current_tts_service = new_service
                        logger.info(f"TTS service changed to: {current_tts_service}")
                        
                        # Broadcast the update to all clients
                        await broadcast_to_all_clients({
                            "type": "tts_service",
                            "service": current_tts_service
                        })
                        
                        # Note: The actual TTS service will be changed when a new recording starts
                        await broadcast_to_all_clients({
                            "type": "notification",
                            "message": f"TTS service changed to {current_tts_service.title()}. This will take effect on the next recording."
                        })
                    else:
                        logger.debug(f"TTS service already set to {new_service} or invalid service name")
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from frontend: {message}")
            except Exception as e:
                logger.error(f"Error handling frontend message: {e}")
            
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Frontend client disconnected: {websocket.remote_address}")
    finally:
        frontend_clients.remove(websocket)


async def broadcast_to_all_clients(message: dict):
    """Broadcast message to all connected frontend clients"""
    if frontend_clients:
        message_str = json.dumps(message)
        disconnected = set()
        
        for client in frontend_clients:
            try:
                await client.send(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        frontend_clients.difference_update(disconnected)


async def run_frontend_server():
    """Run the frontend WebSocket server on port 8766"""
    logger.info("Starting frontend WebSocket server on ws://localhost:8766")
    await websockets.serve(handle_frontend_connection, "localhost", 8766)


def create_stt_service(service_name: str):
    """Create the appropriate STT service based on the service name"""
    if service_name == "openai":
        return OpenAISTTService(
            api_key=OPENAI_API_KEY,
            model="whisper-1"
        )
    elif service_name == "deepgram":
        return DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2"
        )
    else:
        raise ValueError(f"Unknown STT service: {service_name}")


def create_llm_service(model_name: str):
    """Create the appropriate LLM service based on the model name"""
    return OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model=model_name
    )


def create_tts_service(service_name: str):
    """Create the appropriate TTS service based on the service name"""
    if service_name == "elevenlabs":
        return ElevenLabsTTSService(
            api_key=ELEVEN_API_KEY,
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model="eleven_flash_v2_5"
        )
    elif service_name == "deepgram":
        return DeepgramTTSService(
            api_key=DEEPGRAM_API_KEY,
            voice="aura-helios-en",
            sample_rate=16000,
            encoding="linear16"
        )
    else:
        raise ValueError(f"Unknown TTS service: {service_name}")


async def main():
    global current_system_prompt, pipeline_task, context, context_aggregator, latency_metrics, stt_service, llm_service, tts_service, pipeline_runner
    
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
    
    # Create initial context with system prompt
    messages = [
        {
            "role": "system",
            "content": current_system_prompt,
        }
    ]
    context = OpenAILLMContext(messages)
    
    # Create runner
    runner = PipelineRunner()
    
    # Event handler to create new pipeline with selected STT service, LLM model, and TTS service
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        global stt_service, llm_service, tts_service, pipeline_task, context_aggregator
        
        logger.info(f"Audio client connected from {client.remote_address}")
        
        # Create STT service based on current selection
        stt_service = create_stt_service(current_stt_service)
        logger.info(f"Using {current_stt_service.upper()} STT service for new client")
        
        # Create LLM service based on current selection
        llm_service = create_llm_service(current_llm_model)
        logger.info(f"Using {current_llm_model} LLM model for new client")
        
        # Create TTS service based on current selection
        tts_service = create_tts_service(current_tts_service)
        logger.info(f"Using {current_tts_service.upper()} TTS service for new client")
        
        # Create context aggregator for the selected LLM service
        context_aggregator = llm_service.create_context_aggregator(context)
        
        # Create NEW capture processors for each client
        client_transcription_capture = TranscriptionCapture()
        client_assistant_capture = AssistantResponseCapture()
        
        # Build pipeline with selected services
        pipeline = Pipeline(
            [
                transport.input(),                  # WebSocket input from client
                stt_service,                       # Speech-To-Text (selected service)
                client_transcription_capture,      # Capture user transcriptions
                context_aggregator.user(),         # User context aggregator
                llm_service,                       # LLM (selected model)
                client_assistant_capture,          # Capture assistant responses
                tts_service,                       # Text-To-Speech (selected service)
                transport.output(),               # WebSocket output to client
                context_aggregator.assistant(),   # Assistant context aggregator
            ]
        )
        
        # Create task with metrics enabled
        pipeline_task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                allow_interruptions=True,
                enable_metrics=True,           # Enable performance metrics
                report_only_initial_ttfb=False, # Get TTFB for each interaction
            ),
        )
        
        # Reset metrics for new client
        latency_metrics["interaction_start"] = None
        latency_metrics["stt"] = 0
        latency_metrics["llm"] = 0
        latency_metrics["tts"] = 0
        latency_metrics["total"] = 0
        
        # Run the pipeline for this client
        await runner.run(pipeline_task)
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Audio client disconnected: {client.remote_address}")
    
    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
    
    # Create a dummy pipeline just to start the transport
    # The actual pipeline will be created when a client connects
    dummy_pipeline = Pipeline([
        transport.input(),
        transport.output()
    ])
    
    task = PipelineTask(dummy_pipeline)
    
    try:
        # Run the pipeline (this will start the WebSocket server)
        await runner.run(task)
    finally:
        # Cancel the frontend server when pipeline stops
        frontend_server.cancel()


if __name__ == "__main__":
    logger.info("Starting Voice Assistant Backend")
    logger.info("Audio WebSocket server on ws://localhost:8765")
    logger.info("Frontend WebSocket server on ws://localhost:8766")
    logger.info(f"Default STT service: {current_stt_service}")
    logger.info(f"Default LLM model: {current_llm_model}")
    logger.info(f"Default TTS service: {current_tts_service}")
    asyncio.run(main())