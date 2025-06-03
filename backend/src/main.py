"""
Real-time Voice Assistant Backend
Following state-of-the-art configuration for lowest latency
"""
import os
import json
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .audio_room import room_manager, Participant
from .audio_config import transport_config


# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("APP_DEBUG", "false").lower() == "true" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the app
    Initialize resources on startup, cleanup on shutdown
    """
    # Startup
    logger.info("Starting Voice Assistant Backend...")
    
    # TODO: Initialize model pools here (Whisper, TTS, etc.)
    # This ensures models are loaded once and shared across requests
    
    yield
    
    # Shutdown
    logger.info("Shutting down Voice Assistant Backend...")
    # TODO: Cleanup resources


app = FastAPI(
    title="Voice Assistant API",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "service": "voice-assistant"}


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rooms_active": len(room_manager.rooms),
        "uptime_seconds": 0,  # TODO: Track actual uptime
    }


@app.get("/api/rooms")
async def list_rooms():
    """List all active rooms (for debugging)"""
    return {
        "rooms": [
            {
                "room_id": room_id,
                "participants": room.participant_count,
                "created_at": room.created_at.isoformat()
            }
            for room_id, room in room_manager.rooms.items()
        ]
    }


@app.get("/api/rooms/{room_id}/stats")
async def get_room_stats(room_id: str):
    """Get compression and performance stats for a room"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    compression_stats = room.audio_processor.get_compression_stats()
    
    return {
        "room_id": room_id,
        "participants": room.participant_count,
        "created_at": room.created_at.isoformat(),
        "compression": {
            "ratio": f"{compression_stats['compression_ratio']*100:.1f}%",
            "bandwidth_saved_kb": round(compression_stats['bandwidth_saved_kb'], 2),
            "frames_processed": compression_stats['frames_processed'],
            "errors": compression_stats['errors']
        }
    }


@app.websocket("/ws/room/{room_id}")
async def websocket_room_endpoint(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for voice rooms
    Handles bidirectional audio streaming and room events
    """
    await websocket.accept()
    
    participant: Optional[Participant] = None
    user_id: Optional[str] = None
    
    try:
        # Wait for join message
        join_msg = await websocket.receive_json()
        if join_msg.get("type") != "join":
            await websocket.close(code=4000, reason="Must join first")
            return
        
        user_id = join_msg.get("userId")
        if not user_id:
            await websocket.close(code=4001, reason="userId required")
            return
        
        # Create participant and join room
        participant = Participant(user_id, websocket)
        room = room_manager.get_or_create_room(room_id)
        await room.add_participant(participant)
        
        # Send join confirmation
        await websocket.send_json({
            "type": "joined",
            "roomId": room_id,
            "userId": user_id,
            "participants": list(room.participants.keys())
        })
        
        logger.info(f"User {user_id} joined room {room_id}")
        
        # Main message loop
        while True:
            # WebSocket can receive both binary (audio) and text (JSON) messages
            message = await websocket.receive()
            
            if "bytes" in message:
                # Audio data received
                audio_data = message["bytes"]
                participant.metrics["packets_received"] += 1
                participant.metrics["bytes_received"] += len(audio_data)
                
                # Process audio and get response
                response_audio = await room.process_audio_chunk(audio_data, user_id)
                
                # For now, echo back to sender (later will broadcast to others)
                if response_audio:
                    await participant.send_audio(response_audio)
                    
            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"]) if isinstance(message["text"], str) else message["text"]
                    await handle_control_message(data, participant, room)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {user_id}: {message['text']}")
            
    except WebSocketDisconnect:
        logger.info(f"User {user_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"Error in room {room_id} for user {user_id}: {e}")
    finally:
        # Clean up on disconnect
        if participant and user_id:
            room = room_manager.get_room(room_id)
            if room:
                await room.remove_participant(user_id)
                await room_manager.remove_room_if_empty(room_id)


async def handle_control_message(data: dict, participant: Participant, room) -> None:
    """Handle JSON control messages from participants"""
    msg_type = data.get("type")
    
    if msg_type == "voice_activity":
        # Handle VAD events
        activity = data.get("activity")
        if activity == "start":
            participant.is_speaking = True
            await room._broadcast_event({
                "type": "participant_speaking",
                "userId": participant.user_id,
                "speaking": True
            })
        elif activity == "end":
            participant.is_speaking = False
            await room._broadcast_event({
                "type": "participant_speaking", 
                "userId": participant.user_id,
                "speaking": False
            })
            
    elif msg_type == "interrupt":
        # Handle barge-in during TTS playback
        logger.info(f"Interrupt from {participant.user_id}")
        # TODO: Cancel ongoing TTS generation
        
    elif msg_type == "ping":
        # Respond to ping for latency measurement
        await participant.send_json({
            "type": "pong",
            "timestamp": data.get("timestamp")
        })
    
    else:
        logger.warning(f"Unknown message type from {participant.user_id}: {msg_type}")


@app.post("/api/token")
async def create_auth_token():
    """
    Create authentication token for WebSocket connection
    TODO: Implement proper auth
    """
    return {
        "token": "dummy-token",
        "expires_in": 3600
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        # Enable auto-reload in development
        reload=os.environ.get("APP_DEBUG", "false").lower() == "true"
    )