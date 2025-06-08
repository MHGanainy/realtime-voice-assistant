"""
Voice Assistant API - Main Application
"""
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from fastapi import FastAPI, WebSocket, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.handlers.websocket_handler import get_websocket_handler
from src.handlers.events_websocket_handler import get_events_websocket_handler
from src.services.conversation_manager import get_conversation_manager
from src.events import get_event_bus, get_event_store

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    api_keys_status = settings.validate_api_keys()
    if not any(api_keys_status.values()):
        logger.warning("No API keys configured! Please check your .env file")
    
    event_bus = get_event_bus()
    event_store = get_event_store()
    
    await event_bus.emit(
        "global:system:startup",
        app_name=settings.app_name,
        version=settings.app_version,
        api_keys_configured=api_keys_status
    )
    
    yield
    
    logger.info("Shutting down application...")
    await event_bus.emit("global:system:shutdown")
    
    websocket_handler = get_websocket_handler()
    await websocket_handler.shutdown()
    
    events_handler = get_events_websocket_handler()
    await events_handler.shutdown()
    
    conversation_manager = get_conversation_manager()
    await conversation_manager.shutdown()
    
    await event_store.shutdown()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="Realtime voice conversation with AI assistant",
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

websocket_handler = get_websocket_handler()
events_handler = get_events_websocket_handler()
conversation_manager = get_conversation_manager()
event_bus = get_event_bus()
event_store = get_event_store()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Realtime voice conversation with AI assistant",
        "endpoints": {
            "health": "/api/health",
            "websocket": "/ws/conversation",
            "events": "/ws/events",
            "conversations": {
                "list": "/api/conversations",
                "get": "/api/conversations/{conversation_id}",
                "stats": "/api/stats"
            },
            "events_api": {
                "stats": "/api/events/stats",
                "history": "/api/events/history"
            }
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    active_conversations = conversation_manager.get_active_conversations()
    api_keys_status = settings.validate_api_keys()
    
    return {
        "status": "healthy",
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
        "active_conversations": len(active_conversations),
        "api_keys_configured": api_keys_status,
        "event_system": {
            "bus_stats": event_bus.get_stats(),
            "store_stats": event_store.get_stats(),
            "websocket_clients": events_handler.get_stats()
        },
        "settings": {
            "debug": settings.debug,
            "metrics_enabled": settings.enable_metrics
        }
    }


@app.websocket("/ws/conversation")
async def websocket_conversation(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None, description="Optional session ID"),
    stt_provider: Optional[str] = Query(None, description="Speech-to-text provider"),
    stt_model: Optional[str] = Query(None, description="STT model"),
    llm_provider: Optional[str] = Query(None, description="LLM provider"),
    llm_model: Optional[str] = Query(None, description="LLM model"),
    tts_provider: Optional[str] = Query(None, description="Text-to-speech provider"),
    tts_model: Optional[str] = Query(None, description="TTS model"),
    tts_voice: Optional[str] = Query(None, description="TTS voice"),
    system_prompt: Optional[str] = Query(None, description="System prompt for assistant"),
    enable_interruptions: Optional[bool] = Query(None, description="Allow interruptions"),
    vad_enabled: Optional[bool] = Query(None, description="Enable voice activity detection"),
    vad_threshold: Optional[float] = Query(None, description="VAD threshold"),
    sample_rate: Optional[int] = Query(None, description="Audio sample rate"),
    channels: Optional[int] = Query(None, description="Audio channels")
):
    """WebSocket endpoint for voice conversations."""
    await websocket_handler.handle_connection(websocket, session_id)


@app.websocket("/ws/events")
async def websocket_events(
    websocket: WebSocket,
    session_id: str = Query(..., description="Required session ID for access control")
):
    """WebSocket endpoint for real-time events."""
    await events_handler.handle_connection(websocket, session_id)


@app.get("/api/conversations")
async def list_conversations(
    active_only: bool = Query(False, description="Return only active conversations"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(100, description="Maximum number of conversations to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get list of conversations"""
    if active_only:
        conversations = conversation_manager.get_active_conversations()
    else:
        conversations = conversation_manager.get_all_conversations()
    
    if session_id:
        conversations = {
            cid: conv for cid, conv in conversations.items()
            if conv.participant.session_id == session_id
        }
    
    conv_list = list(conversations.values())
    conv_list.sort(key=lambda c: c.created_at, reverse=True)
    
    total = len(conv_list)
    conv_list = conv_list[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "conversations": [
            {
                "id": conv.id,
                "participant_id": conv.participant.id,
                "session_id": conv.participant.session_id,
                "state": conv.state.value,
                "created_at": conv.created_at.isoformat(),
                "started_at": conv.started_at.isoformat() if conv.started_at else None,
                "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                "turn_count": len(conv.turns),
                "duration_ms": conv.metrics.total_duration_ms,
                "config": {
                    "stt_provider": conv.config.stt_provider,
                    "llm_provider": conv.config.llm_provider,
                    "tts_provider": conv.config.tts_provider
                }
            }
            for conv in conv_list
        ]
    }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    session_id: Optional[str] = Query(None, description="Session ID for access control")
):
    """Get details of a specific conversation"""
    conversation = conversation_manager.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if session_id and conversation.participant.session_id != session_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JSONResponse(content=conversation.to_dict())


@app.get("/api/conversations/{conversation_id}/transcript")
async def get_conversation_transcript(
    conversation_id: str,
    session_id: Optional[str] = Query(None, description="Session ID for access control")
):
    """Get transcript of a specific conversation"""
    conversation = conversation_manager.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if session_id and conversation.participant.session_id != session_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "conversation_id": conversation_id,
        "participant_id": conversation.participant.id,
        "created_at": conversation.created_at.isoformat(),
        "duration_ms": conversation.metrics.total_duration_ms,
        "transcript": conversation.get_transcript()
    }


@app.get("/api/stats")
async def get_statistics():
    """Get application statistics"""
    all_conversations = conversation_manager.get_all_conversations()
    active_conversations = conversation_manager.get_active_conversations()
    
    total_duration_ms = sum(c.metrics.total_duration_ms for c in all_conversations.values())
    total_turns = sum(c.metrics.turn_count for c in all_conversations.values())
    total_interruptions = sum(c.metrics.interruption_count for c in all_conversations.values())
    
    return {
        "conversations": {
            "total": len(all_conversations),
            "active": len(active_conversations),
            "completed": len(all_conversations) - len(active_conversations)
        },
        "metrics": {
            "total_duration_ms": total_duration_ms,
            "total_turns": total_turns,
            "total_interruptions": total_interruptions,
            "average_duration_ms": total_duration_ms / len(all_conversations) if all_conversations else 0,
            "average_turns_per_conversation": total_turns / len(all_conversations) if all_conversations else 0
        },
        "events": {
            "bus": event_bus.get_stats(),
            "store": event_store.get_stats(),
            "websocket_clients": events_handler.get_stats()
        },
        "services": {
            "api_keys_configured": settings.validate_api_keys(),
            "default_providers": {
                "stt": settings.default_stt_provider,
                "llm": settings.default_llm_provider,
                "tts": settings.default_tts_provider
            }
        }
    }


@app.get("/api/events/stats")
async def get_event_statistics():
    """Get event system statistics"""
    return {
        "bus": event_bus.get_stats(),
        "store": event_store.get_stats(),
        "websocket": events_handler.get_stats()
    }


@app.get("/api/events/history")
async def get_event_history(
    session_id: str = Query(..., description="Session ID for access control"),
    conversation_id: Optional[str] = Query(None, description="Filter by conversation"),
    limit: int = Query(100, description="Maximum events to return"),
    since_minutes: int = Query(60, description="Get events from last N minutes")
):
    """Get historical events for a session"""
    from datetime import datetime, timedelta
    
    since = datetime.utcnow() - timedelta(minutes=since_minutes)
    
    if conversation_id:
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation or conversation.participant.session_id != session_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        events = await event_store.get_conversation_events(
            conversation_id,
            since=since,
            limit=limit
        )
    else:
        events = await event_store.get_session_events(
            session_id,
            since=since,
            limit=limit
        )
    
    return {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "since": since.isoformat(),
        "count": len(events),
        "events": events
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    await event_bus.emit(
        "global:error:unhandled",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=str(request.url)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )