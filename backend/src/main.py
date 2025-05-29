from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .core.factories.registry import create_stt, create_llm, create_tts
from .core.pipeline.voice_assistant import VoiceAssistantPipeline
from .core.pipeline.middleware import MiddlewarePipeline
from .core.events.types import Event, EventType
from .core.events.bus import event_bus, on_event
from .core.commands.websocket_commands import CommandFactory, CommandType
from .middleware.logging_middleware import LoggingMiddleware
from .middleware.error_middleware import ErrorHandlingMiddleware
from .config import settings
from .utils.decorators import managed_resource

# Import providers to trigger registration
import src.providers

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Event handlers for monitoring
@on_event(EventType.ERROR)
async def handle_errors(event: Event):
    """Global error handler"""
    logger.error(f"System error: {event.data}")

@on_event(EventType.METRICS)
async def handle_metrics(event: Event):
    """Metrics handler for monitoring"""
    logger.info(f"Metrics: {event.data}")

# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info(f"Starting {settings.app_name}")
    
    # Start event bus
    if settings.enable_event_bus:
        await event_bus.start()
    
    yield
    
    # Stop event bus
    if settings.enable_event_bus:
        await event_bus.stop()
        
    logger.info(f"Shutting down {settings.app_name}")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan
)

class WebSocketHandler:
    """Handler for WebSocket connections"""
    
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.pipeline: Optional[VoiceAssistantPipeline] = None
        self.tasks: List[asyncio.Task] = []
        self.command_context: Dict[str, Any] = {}
        self.audio_queue: Optional[asyncio.Queue] = None
        
    async def handle_connection(self):
        """Handle WebSocket connection lifecycle with proper cleanup"""
        await self.ws.accept()
        
        try:
            # Create pipeline with providers
            async with self._create_pipeline() as pipeline:
                self.pipeline = pipeline
                self.command_context["pipeline"] = pipeline
                
                # Start processing tasks
                await self._start_tasks()
                
                # Wait for completion
                await asyncio.gather(*self.tasks)
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            await self._cleanup()
            
    def _create_pipeline(self) -> VoiceAssistantPipeline:
        """Create and configure the voice assistant pipeline"""
        # Create providers using factory
        stt = create_stt()
        llm = create_llm()
        tts = create_tts()
        
        # Setup middleware pipeline
        middleware = MiddlewarePipeline()
        if settings.enable_middleware:
            middleware.add(LoggingMiddleware())
            middleware.add(ErrorHandlingMiddleware(
                retry_attempts=settings.deepgram.retry_attempts,
                retry_delay=settings.deepgram.retry_delay
            ))
        
        # Create pipeline
        pipeline = VoiceAssistantPipeline(
            stt_provider=stt,
            llm_provider=llm,
            tts_provider=tts,
            middleware_pipeline=middleware,
            use_event_bus=settings.enable_event_bus
        )
        
        return pipeline
        
    async def _start_tasks(self):
        """Start all processing tasks"""
        # Ping task to keep connection alive
        self.tasks.append(
            asyncio.create_task(self._ping_loop())
        )
        
        # Message router task - handles both audio and commands
        self.tasks.append(
            asyncio.create_task(self._message_router())
        )
        
        # Pipeline runner task
        self.tasks.append(
            asyncio.create_task(self._run_pipeline())
        )
        
    async def _ping_loop(self):
        """Keep WebSocket alive with periodic pings"""
        try:
            while True:
                await asyncio.sleep(settings.ws_ping_interval)
                if self.ws.application_state != WebSocketState.CONNECTED:
                    break
                await self.ws.send_bytes(b"")
        except Exception:
            pass
            
    async def _message_router(self):
        """Single coroutine to handle all WebSocket messages"""
        self.audio_queue = asyncio.Queue()
        
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.receive":
                    if "bytes" in msg:
                        # Audio data - put in queue for pipeline
                        await self.audio_queue.put(msg["bytes"])
                    elif "text" in msg:
                        # Command - handle immediately
                        await self._handle_command(json.loads(msg["text"]))
                elif msg["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            # Signal end of audio stream
            await self.audio_queue.put(None)
    
    async def _run_pipeline(self):
        """Run the pipeline with audio from the queue"""
        async def audio_source():
            """Generator for audio chunks from queue"""
            while True:
                chunk = await self.audio_queue.get()
                if chunk is None:
                    break
                yield chunk
                
        # Run pipeline with audio source
        await self.pipeline.run(
            audio_source(),
            self._handle_pipeline_event
        )
            
    async def _handle_command(self, command_data: Dict[str, Any]):
        """Handle WebSocket commands using Command pattern"""
        try:
            # Create command instance
            command = CommandFactory.create(command_data)
            
            # Execute command
            result = await command.execute(self.command_context)
            
            # Send result
            await self.ws.send_json(result)
            
        except ValueError as e:
            await self.ws.send_json({
                "error": str(e),
                "type": "command_error"
            })
        except Exception as e:
            logger.error(f"Command execution error: {e}", exc_info=True)
            await self.ws.send_json({
                "error": "Internal command error",
                "type": "command_error"
            })
            
    async def _handle_pipeline_event(self, event: Event):
        """Handle events from the pipeline"""
        try:
            if event.type == EventType.TRANSCRIPT_PARTIAL:
                # Optional: Send partial transcripts
                await self.ws.send_json({
                    "type": "transcript",
                    "transcript": event.data["transcript"],
                    "final": False
                })
                
            elif event.type == EventType.TRANSCRIPT_FINAL:
                await self.ws.send_json({
                    "type": "transcript",
                    "transcript": event.data["transcript"],
                    "final": True
                })
                
            elif event.type == EventType.TTS_CHUNK:
                # Send audio chunk
                await self.ws.send_bytes(event.audio_chunk)
                
            elif event.type == EventType.LLM_COMPLETE:
                # Send complete interaction data
                await self.ws.send_json({
                    "type": "interaction_complete",
                    "utterance": event.data["utterance"],
                    "response": event.data["response"],
                    "token_count": event.data["token_count"]
                })
                
            elif event.type == EventType.COMMAND:
                # Send control commands
                await self.ws.send_json(event.data)
                
            elif event.type == EventType.ERROR:
                # Send error information
                await self.ws.send_json({
                    "type": "error",
                    "error": event.data["error"],
                    "phase": event.data.get("phase", "unknown"),
                    "retry_attempt": event.data.get("retry_attempt")
                })
                
        except Exception as e:
            logger.error(f"Error handling event: {e}", exc_info=True)
            
    async def _cleanup(self):
        """Cleanup resources"""
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        # Pipeline cleanup is handled by context manager

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice assistant"""
    handler = WebSocketHandler(websocket)
    await handler.handle_connection()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .core.factories.registry import registry
    
    return {
        "status": "healthy",
        "app": settings.app_name,
        "providers": {
            "registered": registry.list_providers(),
            "active": {
                "stt": settings.stt_provider,
                "llm": settings.llm_provider,
                "tts": settings.tts_provider
            }
        }
    }

@app.get("/config")
async def get_config():
    """Get current configuration (sanitized)"""
    return {
        "app_name": settings.app_name,
        "debug": settings.debug,
        "providers": {
            "stt": {
                "provider": settings.stt_provider,
                "model": settings.deepgram.model
            },
            "llm": {
                "provider": settings.llm_provider,
                "model": settings.openai.model
            },
            "tts": {
                "provider": settings.tts_provider,
                "voice_id": settings.elevenlabs.voice_id
            }
        },
        "features": {
            "middleware_enabled": settings.enable_middleware,
            "event_bus_enabled": settings.enable_event_bus
        }
    }

@app.post("/providers/reload")
async def reload_providers():
    """Reload provider registrations (useful for development)"""
    # Re-import providers to re-trigger decorators
    import importlib
    import src.providers
    importlib.reload(src.providers)
    
    from .core.factories.registry import registry
    return {
        "status": "reloaded",
        "providers": registry.list_providers()
    }