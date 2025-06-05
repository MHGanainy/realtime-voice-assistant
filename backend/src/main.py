from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from src.pipeline_manager import (
    get_pipeline_runner, 
    create_audio_pipeline,
    create_pipeline_task
)
from src.audio_pipeline import (
    create_stt_service,
    create_llm_service,
    create_tts_service,
    create_llm_context

)
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.silero import VADParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.pipeline.pipeline import Pipeline
from dotenv import load_dotenv
from pathlib import Path
import logging
import os
import uuid

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify API keys are loaded
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

if not all([DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVEN_API_KEY]):
    logger.warning("Missing API keys. Please check your .env file")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize pipeline runner
pipeline_runner = get_pipeline_runner()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Voice assistant backend is running"}

@app.get("/api/session/new")
async def create_new_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming"""
    await websocket.accept()
    
    session_id = websocket.query_params.get("session", str(uuid.uuid4()))
    logger.info(f"WebSocket connection established for session {session_id}")
    
    try:
        # Create transport with protobuf for audio only
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,  # Send raw audio in protobuf
                vad_analyzer=SileroVADAnalyzer(),
                serializer=ProtobufFrameSerializer(),
            )
        )
        
        # Create services using factory functions
        stt = create_stt_service("deepgram")
        llm = create_llm_service("openai", model="gpt-3.5-turbo")
        tts = create_tts_service("elevenlabs", voice_id="21m00Tcm4TlvDq8ikWAM")
        
        # Create context using factory function
        context = create_llm_context(
            llm_service="openai",
            system_prompt="You are a helpful assistant. Keep your responses brief and conversational."
        )
        context_aggregator = llm.create_context_aggregator(context)
        
        # Create pipeline - audio only
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        
        # Create and run task
        task = create_pipeline_task(pipeline)
        
        logger.info(f"Starting pipeline for session {session_id}")
        await pipeline_runner.run(task)
        
        # IMPORTANT: Keep connection alive for a bit to ensure all frames are sent
        logger.info(f"Pipeline completed, waiting for final frames...")
        await asyncio.sleep(2) 
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        raise
    finally:
        logger.info(f"Cleaning up session {session_id}")