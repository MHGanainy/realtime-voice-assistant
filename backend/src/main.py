# backend/src/main.py
"""
Pipecat voice assistant with FastAPI and session management
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
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

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


def create_llm_service(model_name: str):
    """Create the appropriate LLM service"""
    return OpenAILLMService(api_key=OPENAI_API_KEY, model=model_name)


def create_tts_service(service_name: str):
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
            voice="aura-helios-en",
            sample_rate=16000,
            encoding="linear16"
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
            if "Service#" in message and "TTFB:" in message:
                match = re.search(r'(\w+Service)#\d+ TTFB: ([\d.]+)', message)
                if match:
                    service_type = match.group(1)
                    ttfb_value = float(match.group(2))
                    ttfb_ms = int(ttfb_value * 1000)
                    
                    metrics = self.session_data["latency_metrics"]
                    
                    if "STTService" in service_type:
                        metrics["stt"] = ttfb_ms
                    elif "OpenAILLMService" in service_type:
                        metrics["llm"] = ttfb_ms
                    elif "TTSService" in service_type:
                        metrics["tts"] = ttfb_ms
                        
                        # Calculate total
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
        except Exception as e:
            logger.error(f"Error in metrics handler: {e}")


# Startup event to initialize pipeline runner
@app.on_event("startup")
async def startup_event():
    global pipeline_runner
    pipeline_runner = PipelineRunner()
    logger.info("Pipeline runner initialized")


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
                if model in ["gpt-3.5-turbo", "gpt-4o-mini"]:
                    session_data["settings"]["llm_model"] = model
                    await broadcast_to_session(session_id, {
                        "type": "llm_model",
                        "model": model
                    })
                    model_name = "GPT-3.5 Turbo" if model == "gpt-3.5-turbo" else "GPT-4o Mini"
                    await broadcast_to_session(session_id, {
                        "type": "notification",
                        "message": f"LLM model changed to {model_name}. This will take effect on the next recording."
                    })
            
            elif data.get("type") == "change_tts_service":
                service = data.get("service", "elevenlabs")
                if service in ["elevenlabs", "deepgram"]:
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
        llm = create_llm_service(session_data["settings"]["llm_model"])
        tts = create_tts_service(session_data["settings"]["tts_service"])
        
        logger.info(f"Session {session_id}: Using {session_data['settings']['stt_service']} STT, "
                   f"{session_data['settings']['llm_model']} LLM, {session_data['settings']['tts_service']} TTS")
        
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


if __name__ == "__main__":
    logger.info("Starting FastAPI Voice Assistant with Session Support")
    logger.info("Data WebSocket: ws://localhost:8000/ws/data")
    logger.info("Audio WebSocket: ws://localhost:8000/ws/audio")
    
    # Configure logging
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)