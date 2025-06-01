# backend/src/main.py
"""
Pipecat voice assistant with FastAPI and session management
Now with DeepInfra Llama support and many more models
"""
import asyncio
import os
import sys
import json
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Dict, Any, Set
from datetime import datetime
import time

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Import the DeepInfra service (assuming it's in a services folder)
# You'll need to place the deepinfra_llm.py file in your project
from services.deepinfra_llm import DeepInfraLLMService

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
from pipecat.services.openai.tts import OpenAITTSService  # Add OpenAI TTS import
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")  # Add this to your .env file

if not all([OPENAI_API_KEY, ELEVEN_API_KEY, DEEPGRAM_API_KEY]):
    raise ValueError("Missing required API keys. Please set OPENAI_API_KEY, ELEVEN_API_KEY, and DEEPGRAM_API_KEY")

# Global pipeline runner - will be initialized when the app starts
pipeline_runner = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pipeline_runner
    pipeline_runner = PipelineRunner()
    logger.info("Pipeline runner initialized")
    yield
    # Shutdown
    logger.info("Shutting down pipeline runner")


# Create FastAPI app
app = FastAPI(title="Voice Assistant API", lifespan=lifespan)

# Add session middleware - IMPORTANT: Add this before CORS
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "your-secret-key-change-this-in-production"),
    session_cookie="voice_assistant_session",
    max_age=1800,  # 30 minutes
    same_site="lax",
    https_only=False  # Set to True in production with HTTPS
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage - In production, use Redis or a database
sessions_store: Dict[str, Dict[str, Any]] = {}

# Track WebSocket connections per session
session_connections: Dict[str, Set[WebSocket]] = {}

# Global pipeline runner - will be initialized when the app starts
pipeline_runner = None


# Helper functions
def get_or_create_session_id(request: Request) -> str:
    """Get session ID from request or create new one"""
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id
    return session_id


def get_session_data(session_id: str) -> Dict[str, Any]:
    """Get or create session data"""
    if session_id not in sessions_store:
        sessions_store[session_id] = {
            "conversation_history": [],
            "system_prompt": "You are a helpful voice assistant. Keep your responses concise and conversational. Your output will be converted to audio so don't include special characters in your answers.",
            "current_interaction": {"user": "", "assistant": ""},
            "settings": {
                "stt_service": "openai",
                "llm_model": "gpt-3.5-turbo",
                "llm_service": "openai",  # New setting to track which LLM service
                "tts_service": "elevenlabs"
            },
            "latency_metrics": {
                "stt": 0,
                "llm": 0,
                "tts": 0,
                "total": 0,
                "interaction_start": None
            },
            "created_at": datetime.now().isoformat(),
            "pipeline_task": None,
            "context": None,
            "context_aggregator": None
        }
    return sessions_store[session_id]


async def broadcast_to_session(session_id: str, message: dict):
    """Broadcast message to all WebSocket clients in a session"""
    if session_id in session_connections:
        disconnected = set()
        message_str = json.dumps(message)
        
        for websocket in session_connections[session_id]:
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        session_connections[session_id] -= disconnected
        if not session_connections[session_id]:
            del session_connections[session_id]


def create_stt_service(service_name: str):
    """Create the appropriate STT service"""
    if service_name == "openai":
        return OpenAISTTService(api_key=OPENAI_API_KEY, model="whisper-1")
    elif service_name == "deepgram":
        return DeepgramSTTService(api_key=DEEPGRAM_API_KEY, model="nova-2")
    else:
        raise ValueError(f"Unknown STT service: {service_name}")


def create_llm_service(model_name: str, service_name: str):
    """Create the appropriate LLM service"""
    if service_name == "openai":
        return OpenAILLMService(api_key=OPENAI_API_KEY, model=model_name)
    elif service_name == "deepinfra":
        # For DeepInfra, we use the model name directly as it's already in the correct format
        if not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY not set in environment variables")
            
        return DeepInfraLLMService(
            api_key=DEEPINFRA_API_KEY,
            model=model_name
        )
    else:
        raise ValueError(f"Unknown LLM service: {service_name}")


def create_tts_service(service_name: str, session_data: Dict[str, Any]):
    """Create the appropriate TTS service"""
    if service_name == "elevenlabs":
        return ElevenLabsTTSService(
            api_key=ELEVEN_API_KEY,
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model="eleven_flash_v2_5"
        )
    elif service_name == "deepgram":
        return DeepgramTTSService(
            api_key=DEEPGRAM_API_KEY,
            voice="aura-2-thalia-en",
            sample_rate=16000,
            encoding="linear16"
        )
    elif service_name == "openai":
        return OpenAITTSService(
            api_key=OPENAI_API_KEY,
            voice="nova",  # Using alloy as the single voice
            model="gpt-4o-mini-tts",  # Using only the optimized model
            sample_rate=24000,
        )
    else:
        raise ValueError(f"Unknown TTS service: {service_name}")


# Session-aware frame processors
class SessionTranscriptionCapture(FrameProcessor):
    """Capture user transcriptions for a specific session"""
    
    def __init__(self, session_id: str, session_data: Dict[str, Any]):
        super().__init__()
        self.session_id = session_id
        self.session_data = session_data
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        try:
            if isinstance(frame, InterimTranscriptionFrame):
                if not self.session_data["latency_metrics"]["interaction_start"]:
                    self.session_data["latency_metrics"]["interaction_start"] = time.time()
                    logger.info(f"Session {self.session_id}: Interaction started")
                
                await broadcast_to_session(self.session_id, {
                    "type": "transcription",
                    "text": frame.text,
                    "final": False
                })
                
            elif isinstance(frame, TranscriptionFrame):
                if not self.session_data["latency_metrics"]["interaction_start"]:
                    self.session_data["latency_metrics"]["interaction_start"] = time.time() - 2.0
                
                logger.info(f"Session {self.session_id} - User: {frame.text}")
                
                # Update current interaction
                self.session_data["current_interaction"]["user"] = frame.text
                
                # Send to frontend
                await broadcast_to_session(self.session_id, {
                    "type": "transcription",
                    "text": frame.text,
                    "final": True
                })
                
                # Add to conversation history
                self.session_data["conversation_history"].append({
                    "role": "user",
                    "content": frame.text,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send updated history
                await broadcast_to_session(self.session_id, {
                    "type": "conversation_history",
                    "history": self.session_data["conversation_history"]
                })
                
        except Exception as e:
            logger.error(f"Error in SessionTranscriptionCapture: {e}")
        
        await self.push_frame(frame, direction)


class SessionAssistantResponseCapture(FrameProcessor):
    """Capture assistant responses for a specific session"""
    
    def __init__(self, session_id: str, session_data: Dict[str, Any]):
        super().__init__()
        self.session_id = session_id
        self.session_data = session_data
        self.response_buffer = []
        self.collecting = False
        self.finalize_task = None
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        try:
            if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
                if frame.text:
                    if self.finalize_task:
                        self.finalize_task.cancel()
                    
                    if not self.collecting:
                        self.collecting = True
                        self.response_buffer = []
                    
                    self.response_buffer.append(frame.text)
                    current_text = ''.join(self.response_buffer)
                    
                    # Update current interaction
                    self.session_data["current_interaction"]["assistant"] = current_text
                    
                    await broadcast_to_session(self.session_id, {
                        "type": "assistant_reply",
                        "text": current_text,
                        "final": False
                    })
                    
                    self.finalize_task = asyncio.create_task(self._finalize_response())
                    
        except Exception as e:
            logger.error(f"Error in SessionAssistantResponseCapture: {e}")
        
        await self.push_frame(frame, direction)
    
    async def _finalize_response(self):
        """Finalize response after a delay"""
        try:
            await asyncio.sleep(1.0)
            
            if self.collecting and self.response_buffer:
                final_text = ''.join(self.response_buffer)
                logger.info(f"Session {self.session_id} - Assistant: {final_text}")
                
                await broadcast_to_session(self.session_id, {
                    "type": "assistant_reply",
                    "text": final_text,
                    "final": True
                })
                
                self.session_data["conversation_history"].append({
                    "role": "assistant",
                    "content": final_text,
                    "timestamp": datetime.now().isoformat()
                })
                
                await broadcast_to_session(self.session_id, {
                    "type": "conversation_history",
                    "history": self.session_data["conversation_history"]
                })
                
                self.collecting = False
                self.response_buffer = []
                
        except asyncio.CancelledError:
            pass


class SessionMetricsHandler:
    """Handle metrics for a specific session"""
    
    def __init__(self, session_id: str, session_data: Dict[str, Any]):
        self.session_id = session_id
        self.session_data = session_data
        
    def handle(self, message):
        try:
            metrics = self.session_data["latency_metrics"]
            
            # Handle TTFB metrics
            if "Service#" in message and "TTFB:" in message:
                match = re.search(r'(\w+Service)#\d+ TTFB: ([\d.]+)', message)
                if match:
                    service_type = match.group(1)
                    ttfb_value = float(match.group(2))
                    ttfb_ms = int(ttfb_value * 1000)
                    
                    if "STTService" in service_type:
                        metrics["stt"] = ttfb_ms
                    elif "TTSService" in service_type:
                        metrics["tts"] = ttfb_ms
                        
                        # Calculate total and send update when TTS starts
                        if metrics["interaction_start"]:
                            total_time = int((time.time() - metrics["interaction_start"]) * 1000)
                            metrics["total"] = total_time
                            
                            asyncio.create_task(broadcast_to_session(self.session_id, {
                                "type": "latency_update",
                                "latencies": {
                                    "stt": metrics["stt"],
                                    "llm": metrics["llm"],
                                    "tts": metrics["tts"],
                                    "total": metrics["total"]
                                }
                            }))
                            
                            metrics["interaction_start"] = None
            
            # Handle processing time metrics - use this for LLM
            elif "Service#" in message and "processing time:" in message:
                match = re.search(r'(\w+LLMService)#\d+ processing time: ([\d.]+)', message)
                if match:
                    processing_time = float(match.group(2))
                    processing_ms = int(processing_time * 1000)
                    metrics["llm"] = processing_ms
                    
        except Exception as e:
            logger.error(f"Error in metrics handler: {e}")


# WebSocket endpoints
@app.websocket("/ws/data")
async def websocket_data_endpoint(websocket: WebSocket):
    """Handle frontend data connections"""
    await websocket.accept()
    
    # Get session ID from query params or create new one
    session_id = websocket.query_params.get("session")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get session data
    session_data = get_session_data(session_id)
    
    # Track this connection
    if session_id not in session_connections:
        session_connections[session_id] = set()
    session_connections[session_id].add(websocket)
    
    logger.info(f"Data client connected to session {session_id}")
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "session_id": session_id
        })
        
        await websocket.send_json({
            "type": "system_prompt",
            "prompt": session_data["system_prompt"]
        })
        
        await websocket.send_json({
            "type": "conversation_history",
            "history": session_data["conversation_history"]
        })
        
        await websocket.send_json({
            "type": "latency_update",
            "latencies": session_data["latency_metrics"]
        })
        
        # Send current settings
        await websocket.send_json({
            "type": "stt_service",
            "service": session_data["settings"]["stt_service"]
        })
        
        await websocket.send_json({
            "type": "llm_model",
            "model": session_data["settings"]["llm_model"]
        })
        
        await websocket.send_json({
            "type": "llm_service",
            "service": session_data["settings"]["llm_service"]
        })
        
        await websocket.send_json({
            "type": "tts_service",
            "service": session_data["settings"]["tts_service"]
        })
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "update_system_prompt":
                new_prompt = data.get("prompt", "")
                if new_prompt:
                    session_data["system_prompt"] = new_prompt
                    
                    # Update context if it exists
                    if session_data.get("context"):
                        session_data["context"].set_messages([{
                            "role": "system",
                            "content": new_prompt
                        }])
                    
                    # Clear history
                    session_data["conversation_history"] = []
                    
                    await broadcast_to_session(session_id, {
                        "type": "system_prompt",
                        "prompt": new_prompt
                    })
                    
                    await broadcast_to_session(session_id, {
                        "type": "conversation_history",
                        "history": []
                    })
            
            elif data.get("type") == "clear_history":
                session_data["conversation_history"] = []
                
                # Reset context
                if session_data.get("context"):
                    session_data["context"].set_messages([{
                        "role": "system",
                        "content": session_data["system_prompt"]
                    }])
                
                await broadcast_to_session(session_id, {
                    "type": "conversation_history",
                    "history": []
                })
            
            elif data.get("type") == "change_stt_service":
                service = data.get("service", "openai")
                if service in ["openai", "deepgram"]:
                    session_data["settings"]["stt_service"] = service
                    await broadcast_to_session(session_id, {
                        "type": "stt_service",
                        "service": service
                    })
                    await broadcast_to_session(session_id, {
                        "type": "notification",
                        "message": f"STT service changed to {service.title()}. This will take effect on the next recording."
                    })
            
            elif data.get("type") == "change_llm_model":
                model = data.get("model", "gpt-3.5-turbo")
                service = data.get("service", "openai")
                
                # Update model and service
                session_data["settings"]["llm_model"] = model
                session_data["settings"]["llm_service"] = service
                
                await broadcast_to_session(session_id, {
                    "type": "llm_model",
                    "model": model
                })
                await broadcast_to_session(session_id, {
                    "type": "llm_service",
                    "service": service
                })
                
                # For notification, extract a friendly name if possible
                model_display = model.split("/")[-1] if "/" in model else model
                service_name = "DeepInfra" if service == "deepinfra" else "OpenAI"
                
                await broadcast_to_session(session_id, {
                    "type": "notification",
                    "message": f"LLM changed to {model_display} ({service_name}). This will take effect on the next recording."
                })
            
            elif data.get("type") == "change_tts_service":
                service = data.get("service", "elevenlabs")
                if service in ["elevenlabs", "deepgram", "openai"]:
                    session_data["settings"]["tts_service"] = service
                    await broadcast_to_session(session_id, {
                        "type": "tts_service",
                        "service": service
                    })
                    await broadcast_to_session(session_id, {
                        "type": "notification",
                        "message": f"TTS service changed to {service.title()}. This will take effect on the next recording."
                    })
                    
    except WebSocketDisconnect:
        logger.info(f"Data client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"Error in data websocket: {e}")
    finally:
        if session_id in session_connections:
            session_connections[session_id].discard(websocket)


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """Handle audio connections"""
    await websocket.accept()
    
    # Get session ID from query params
    session_id = websocket.query_params.get("session")
    if not session_id:
        await websocket.close(code=1008, reason="Session ID required")
        return
    
    # Get session data
    session_data = get_session_data(session_id)
    
    logger.info(f"Audio client connected to session {session_id}")
    
    try:
        # Create transport
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=True,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=ProtobufFrameSerializer(),
            )
        )
        
        # Create services based on session settings
        stt = create_stt_service(session_data["settings"]["stt_service"])
        llm = create_llm_service(
            session_data["settings"]["llm_model"],
            session_data["settings"]["llm_service"]
        )
        tts = create_tts_service(session_data["settings"]["tts_service"], session_data)
        
        logger.info(f"Session {session_id}: Using {session_data['settings']['stt_service']} STT, "
                   f"{session_data['settings']['llm_model']} LLM ({session_data['settings']['llm_service']}), "
                   f"{session_data['settings']['tts_service']} TTS")
        
        # Create context
        session_data["context"] = OpenAILLMContext([{
            "role": "system",
            "content": session_data["system_prompt"]
        }])
        session_data["context_aggregator"] = llm.create_context_aggregator(session_data["context"])
        
        # Add metrics handler for this session
        metrics_handler = SessionMetricsHandler(session_id, session_data)
        logger.add(metrics_handler.handle, level="DEBUG")
        
        # Reset metrics
        session_data["latency_metrics"] = {
            "stt": 0, "llm": 0, "tts": 0, "total": 0, "interaction_start": None
        }
        
        # Create pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            SessionTranscriptionCapture(session_id, session_data),
            session_data["context_aggregator"].user(),
            llm,
            SessionAssistantResponseCapture(session_id, session_data),
            tts,
            transport.output(),
            session_data["context_aggregator"].assistant(),
        ])
        
        # Create and run task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                allow_interruptions=True,
                enable_metrics=True,
                report_only_initial_ttfb=False,
            )
        )
        
        session_data["pipeline_task"] = task
        
        # Run the pipeline
        await pipeline_runner.run(task)
        
    except WebSocketDisconnect:
        logger.info(f"Audio client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"Error in audio websocket: {e}")
    finally:
        session_data["pipeline_task"] = None


# Regular HTTP endpoints
@app.get("/api/session/new")
async def create_new_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    get_session_data(session_id)  # Initialize session data
    return {"session_id": session_id}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "sessions": len(sessions_store)}


@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            # OpenAI Models
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "service": "openai",
                "category": "OpenAI",
                "description": "Fast and efficient for most tasks"
            },
            {
                "id": "gpt-4o-mini",
                "name": "GPT-4o Mini",
                "service": "openai",
                "category": "OpenAI",
                "description": "More capable than GPT-3.5"
            },
            
            # Meta Llama Models
            {
                "id": "meta-llama/Meta-Llama-3.3-70B-Instruct",
                "name": "Llama 3.3 70B",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Latest Llama model with 128k context"
            },
            {
                "id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "name": "Llama 3.1 405B",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Largest Llama model, exceptional capabilities"
            },
            {
                "id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "name": "Llama 3.1 70B",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Large model with strong capabilities"
            },
            {
                "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "name": "Llama 3.1 8B",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Smaller, faster Llama model"
            },
            {
                "id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "name": "Llama 3.2 11B Vision",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Multimodal model with vision capabilities"
            },
            {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "name": "Llama 3.2 3B",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Lightweight model for quick responses"
            },
            {
                "id": "meta-llama/Llama-3.2-1B-Instruct",
                "name": "Llama 3.2 1B",
                "service": "deepinfra",
                "category": "Meta Llama",
                "description": "Ultra-fast model for simple tasks"
            },
            
            # DeepSeek Models
            {
                "id": "deepseek-ai/DeepSeek-V3",
                "name": "DeepSeek V3",
                "service": "deepinfra",
                "category": "DeepSeek",
                "description": "671B MoE model with 37B active params"
            },
            {
                "id": "deepseek-ai/DeepSeek-R1",
                "name": "DeepSeek R1",
                "service": "deepinfra",
                "category": "DeepSeek",
                "description": "Reasoning model comparable to OpenAI o1"
            },
            {
                "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "name": "DeepSeek R1 Distill 70B",
                "service": "deepinfra",
                "category": "DeepSeek",
                "description": "Distilled reasoning patterns in smaller model"
            },
            
            # Qwen Models
            {
                "id": "Qwen/QwQ-32B",
                "name": "QwQ 32B",
                "service": "deepinfra",
                "category": "Qwen",
                "description": "Reasoning model from Qwen series"
            },
            {
                "id": "Qwen/Qwen2.5-72B-Instruct",
                "name": "Qwen 2.5 72B",
                "service": "deepinfra",
                "category": "Qwen",
                "description": "Large multilingual model"
            },
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen 2.5 7B",
                "service": "deepinfra",
                "category": "Qwen",
                "description": "Efficient multilingual model"
            },
            {
                "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "name": "Qwen 2.5 Coder 32B",
                "service": "deepinfra",
                "category": "Qwen",
                "description": "Specialized for code generation"
            },
            
            # Google Models
            {
                "id": "google/gemma-3-27b-it",
                "name": "Gemma 3 27B",
                "service": "deepinfra",
                "category": "Google",
                "description": "Google's multimodal open source model"
            },
            {
                "id": "google/gemma-3-12b-it",
                "name": "Gemma 3 12B",
                "service": "deepinfra",
                "category": "Google",
                "description": "Multimodal model with function calling"
            },
            {
                "id": "google/gemma-3-4b-it",
                "name": "Gemma 3 4B",
                "service": "deepinfra",
                "category": "Google",
                "description": "Lightweight multimodal model"
            },
            {
                "id": "google/gemini-2.0-flash-001",
                "name": "Gemini 2.0 Flash",
                "service": "deepinfra",
                "category": "Google",
                "description": "Latest Gemini model for fast inference"
            },
            
            # Microsoft Models
            {
                "id": "microsoft/phi-4",
                "name": "Phi 4",
                "service": "deepinfra",
                "category": "Microsoft",
                "description": "Small model with advanced reasoning"
            },
            {
                "id": "microsoft/phi-4-reasoning-plus",
                "name": "Phi 4 Reasoning Plus",
                "service": "deepinfra",
                "category": "Microsoft",
                "description": "Enhanced reasoning capabilities"
            },
            {
                "id": "microsoft/WizardLM-2-8x22B",
                "name": "WizardLM 2 8x22B",
                "service": "deepinfra",
                "category": "Microsoft",
                "description": "Advanced Wizard model with MoE"
            },
            
            # Mistral Models
            {
                "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "name": "Mixtral 8x7B",
                "service": "deepinfra",
                "category": "Mistral",
                "description": "MoE model with excellent performance"
            },
            {
                "id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "name": "Mixtral 8x22B",
                "service": "deepinfra",
                "category": "Mistral",
                "description": "Large MoE model"
            },
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.3",
                "name": "Mistral 7B v0.3",
                "service": "deepinfra",
                "category": "Mistral",
                "description": "Latest Mistral 7B with function calling"
            },
            {
                "id": "mistralai/Mistral-Nemo-Instruct-2407",
                "name": "Mistral Nemo 12B",
                "service": "deepinfra",
                "category": "Mistral",
                "description": "12B model by Mistral & NVIDIA"
            }
        ]
    }


if __name__ == "__main__":
    logger.info("Starting FastAPI Voice Assistant with Session Support")
    logger.info("Now with DeepInfra support for many models!")
    logger.info("Data WebSocket: ws://localhost:8000/ws/data")
    logger.info("Audio WebSocket: ws://localhost:8000/ws/audio")
    
    # Configure logging
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)