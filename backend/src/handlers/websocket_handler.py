"""
WebSocket connection handler for voice conversations.
Manages the connection lifecycle and conversation setup.
"""
import asyncio
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging
import uuid
import aiohttp

from src.domains.conversation import Participant, ConversationConfig
from src.services.conversation_manager import get_conversation_manager
from src.services.pipeline_factory import get_pipeline_factory
from src.config.settings import get_settings
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer

logger = logging.getLogger(__name__)


class WebSocketConnectionHandler:
    """Handles WebSocket connections for voice conversations"""
    
    def __init__(self):
        self.conversation_manager = get_conversation_manager()
        self.pipeline_factory = get_pipeline_factory()
        self._settings = get_settings()
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._aiohttp_sessions: Dict[str, aiohttp.ClientSession] = {}
    
    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None
    ):
        """Handle a new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        session_id = session_id or connection_id
        aiohttp_session = None
        
        try:
            await websocket.accept()
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Create participant
            participant = Participant(
                connection_id=connection_id,
                session_id=session_id,
                user_agent=websocket.headers.get("user-agent"),
                ip_address=websocket.client.host if websocket.client else None,
                metadata={"session_id": session_id}
            )
            
            # Get configuration from query params or use defaults
            config = self._build_config_from_params(websocket)
            
            # Create conversation
            conversation = await self.conversation_manager.create_conversation(
                participant=participant,
                config=config
            )
            
            # Create aiohttp session for TTS
            aiohttp_session = aiohttp.ClientSession()
            self._aiohttp_sessions[connection_id] = aiohttp_session
            
            # Setup transport
            transport = self._create_transport(websocket, config)
            
            # Create pipeline
            pipeline, output_sample_rate = await self.pipeline_factory.create_pipeline(
                config=config,
                transport=transport,
                conversation_id=conversation.id,
                aiohttp_session=aiohttp_session
            )
            
            # Store connection info
            self._active_connections[connection_id] = {
                "websocket": websocket,
                "participant": participant,
                "conversation": conversation,
                "transport": transport,
                "pipeline": pipeline,
                "aiohttp_session": aiohttp_session,
                "connected_at": datetime.utcnow()
            }
            
            # Register event handlers
            self._setup_event_handlers(conversation.id)
            
            # Start conversation
            success = await self.conversation_manager.start_conversation(
                conversation.id,
                transport,
                pipeline
            )
            
            if not success:
                await websocket.close(code=1011, reason="Failed to start conversation")
                return
            
            # Create and run pipeline task
            task = self.pipeline_factory.create_pipeline_task(
                pipeline=pipeline,
                config=config,
                output_sample_rate=output_sample_rate
            )
            
            logger.info(f"Starting pipeline for conversation {conversation.id}")
            
            # Run the pipeline (this blocks until complete)
            await self.conversation_manager.run_pipeline_for_conversation(
                conversation.id,
                task
            )
            
        except WebSocketDisconnect:
            await self._handle_disconnect(connection_id)
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
            await self._handle_error(connection_id, e)
        finally:
            await self._cleanup_connection(connection_id)
    
    def _build_config_from_params(self, websocket: WebSocket) -> ConversationConfig:
        """Build conversation config from query parameters"""
        params = websocket.query_params
        
        # Use settings defaults if not specified in params
        config = ConversationConfig(
            # Audio settings
            sample_rate=int(params.get("sample_rate", self._settings.default_sample_rate)),
            channels=int(params.get("channels", self._settings.default_channels)),
            audio_format=params.get("audio_format", "pcm"),
            
            # Service settings with defaults from settings
            stt_provider=params.get("stt_provider", self._settings.default_stt_provider),
            stt_model=params.get("stt_model", self._settings.default_stt_model),
            llm_provider=params.get("llm_provider", self._settings.default_llm_provider),
            llm_model=params.get("llm_model", self._settings.default_llm_model),
            tts_provider=params.get("tts_provider", self._settings.default_tts_provider),
            tts_model=params.get("tts_model", self._settings.default_tts_model),
            tts_voice=params.get("tts_voice", self._settings.default_tts_voice),
            
            # Behavior settings
            system_prompt=params.get("system_prompt", self._settings.default_system_prompt),
            enable_interruptions=params.get("enable_interruptions", "true").lower() == "true",
            vad_enabled=params.get("vad_enabled", "true").lower() == "true",
            vad_threshold=float(params.get("vad_threshold", 0.5)),
            
            # LLM parameters
            llm_temperature=float(params.get("llm_temperature", 0.7)),
            llm_max_tokens=int(params.get("llm_max_tokens", 4096)),
            llm_top_p=float(params.get("llm_top_p", 1.0))
        )
        
        return config
    
    def _create_transport(
        self,
        websocket: WebSocket,
        config: ConversationConfig
    ) -> FastAPIWebsocketTransport:
        """Create WebSocket transport for the conversation"""
        vad_analyzer = None
        if config.vad_enabled:
            vad_analyzer = SileroVADAnalyzer(
                params=VADParams(threshold=config.vad_threshold)
            )
        
        params = FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=config.vad_enabled,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            serializer=ProtobufFrameSerializer(),
            audio_frame_size=config.sample_rate // 100,  # 10ms frames
            send_silent_audio=False
        )
        
        return FastAPIWebsocketTransport(
            websocket=websocket,
            params=params
        )
    
    def _setup_event_handlers(self, conversation_id: str):
        """Setup event handlers for conversation events"""
        manager = self.conversation_manager
        
        # Log conversation events
        def log_turn(data):
            turn = data.get('turn')
            if turn:
                logger.info(f"[{conversation_id}] {turn.speaker}: {turn.text}")
        
        def log_error(data):
            error = data.get('error')
            logger.error(f"Conversation error in {conversation_id}: {error}")
        
        def log_interruption(data):
            logger.info(f"Interruption in {conversation_id}")
        
        manager.on_event("turn_completed", log_turn)
        manager.on_event("conversation_error", log_error)
        manager.on_event("interruption", log_interruption)
    
    async def _handle_disconnect(self, connection_id: str):
        """Handle client disconnect"""
        logger.info(f"WebSocket disconnected: {connection_id}")
        
        conn_info = self._active_connections.get(connection_id)
        if conn_info:
            conversation = conn_info["conversation"]
            await self.conversation_manager.end_conversation(conversation.id)
    
    async def _handle_error(self, connection_id: str, error: Exception):
        """Handle connection error"""
        logger.error(f"Connection error for {connection_id}: {error}")
        
        conn_info = self._active_connections.get(connection_id)
        if conn_info:
            websocket = conn_info["websocket"]
            conversation = conn_info["conversation"]
            
            try:
                await websocket.close(code=1011, reason=str(error))
            except:
                pass
            
            await self.conversation_manager.end_conversation(
                conversation.id,
                error=str(error)
            )
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up connection resources"""
        if connection_id in self._active_connections:
            conn_info = self._active_connections[connection_id]
            conversation = conn_info["conversation"]
            
            # Ensure conversation is ended
            await self.conversation_manager.end_conversation(conversation.id)
            
            # Close aiohttp session
            if connection_id in self._aiohttp_sessions:
                session = self._aiohttp_sessions[connection_id]
                await session.close()
                del self._aiohttp_sessions[connection_id]
            
            # Log conversation metrics
            logger.info(
                f"Conversation {conversation.id} summary:\n"
                f"  - Duration: {conversation.metrics.total_duration_ms}ms\n"
                f"  - Turns: {conversation.metrics.turn_count}\n"
                f"  - Interruptions: {conversation.metrics.interruption_count}\n"
                f"  - Audio In: {conversation.metrics.total_audio_bytes_in} bytes\n"
                f"  - Audio Out: {conversation.metrics.total_audio_bytes_out} bytes"
            )
            
            # Remove from active connections
            del self._active_connections[connection_id]
    
    async def shutdown(self):
        """Shutdown handler and cleanup all connections"""
        logger.info("Shutting down WebSocket handler")
        
        # Close all active connections
        connection_ids = list(self._active_connections.keys())
        for conn_id in connection_ids:
            await self._cleanup_connection(conn_id)
        
        # Close all aiohttp sessions
        for session in self._aiohttp_sessions.values():
            await session.close()
        self._aiohttp_sessions.clear()


# Global handler instance
_websocket_handler: Optional[WebSocketConnectionHandler] = None


def get_websocket_handler() -> WebSocketConnectionHandler:
    """Get the singleton WebSocket handler"""
    global _websocket_handler
    if _websocket_handler is None:
        _websocket_handler = WebSocketConnectionHandler()
    return _websocket_handler