"""
WebSocket connection handler for voice conversations.
"""
import asyncio
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging
import uuid
import aiohttp
import os

from src.domains.conversation import Participant, ConversationConfig
from src.services.conversation_manager import get_conversation_manager
from src.services.pipeline_factory import get_pipeline_factory
from src.services.transcript_storage import get_transcript_storage
from src.config.settings import get_settings
from src.events import EventBus, get_event_bus
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams
)
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.serializers.protobuf import ProtobufFrameSerializer

# ===== AUTH IMPORTS - REMOVE FOR NO AUTH =====
try:
    from src.auth.token_validator import token_validator
    print("✓ Token validator imported successfully")
except Exception as e:
    print(f"✗ Failed to import token validator: {e}")
    token_validator = None
# ===== END AUTH IMPORTS =====

logger = logging.getLogger(__name__)


class WebSocketConnectionHandler:
    """Handles WebSocket connections for voice conversations"""
    
    def __init__(self):
        self.conversation_manager = get_conversation_manager()
        self.pipeline_factory = get_pipeline_factory()
        self.transcript_storage = get_transcript_storage()
        self._settings = get_settings()
        self._event_bus = get_event_bus()
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._aiohttp_sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # ===== DEV MODE CONFIGURATION =====
        # Set to True to bypass token validation
        # IMPORTANT: Must be False in production!
        self.AUTH_DEV_MODE = os.getenv('AUTH_DEV_MODE', 'false').lower() == 'true'
        if self.AUTH_DEV_MODE:
            logger.warning("⚠️ AUTH DEV MODE ENABLED - Token validation bypassed!")
            logger.warning("⚠️ This should ONLY be used in development!")
        # ===== END DEV MODE CONFIGURATION =====
    
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
            logger.info(f"Query params received: {dict(websocket.query_params)}")
            
            # ===== AUTHENTICATION START - REMOVE FOR NO AUTH =====
            correlation_token = None
            authenticated_data = {}
            
            if not self.AUTH_DEV_MODE:
                # PRODUCTION MODE: Require both tokens
                
                # 1. Check for JWT token (authentication)
                jwt_token = websocket.query_params.get("token")
                if not jwt_token:
                    logger.warning(f"No JWT token provided for connection {connection_id}")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Authentication required. Please provide a valid JWT token."
                    })
                    await websocket.close(code=1008, reason="No authentication token")
                    return
                
                # Validate JWT token
                if token_validator:
                    auth_data = token_validator.validate_token(jwt_token)
                    if not auth_data:
                        logger.warning(f"Invalid JWT token for connection {connection_id}")
                        await websocket.send_json({
                            "type": "error",
                            "error": "Invalid or expired authentication token"
                        })
                        await websocket.close(code=1008, reason="Invalid token")
                        return
                else:
                    logger.error("Token validator not available")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Authentication system unavailable"
                    })
                    await websocket.close(code=1011, reason="Auth system error")
                    return
                
                # 2. Check for correlation token (transcript tracking)
                correlation_token = websocket.query_params.get("correlation_token")
                if not correlation_token:
                    logger.warning(f"No correlation token provided for connection {connection_id}")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Correlation token required for transcript tracking"
                    })
                    await websocket.close(code=1008, reason="No correlation token")
                    return
                
                # Both tokens valid - store auth data
                authenticated_data = {
                    'attempt_id': auth_data.get('attempt_id'),
                    'student_id': auth_data.get('student_id'),
                    'authenticated': True,
                    'auth_method': 'jwt',
                    'jwt_correlation': auth_data.get('correlation_token'),
                    'actual_correlation': correlation_token
                }
                
                # Optional: Log if correlation tokens don't match
                jwt_correlation = auth_data.get('correlation_token')
                if jwt_correlation and jwt_correlation != correlation_token:
                    logger.warning(f"Correlation token mismatch: JWT={jwt_correlation}, Query={correlation_token}")
                
                # session_id = f"session_{auth_data.get('attempt_id', connection_id)}"
                logger.info(f"Authenticated connection: {session_id} with correlation: {correlation_token}")
                
            else:
                # DEV MODE: Accept with minimal requirements
                correlation_token = websocket.query_params.get("correlation_token")
                if not correlation_token:
                    correlation_token = f"dev_token_{datetime.utcnow().timestamp()}"
                    logger.info(f"[DEV MODE] Generated correlation token: {correlation_token}")
                
                jwt_token = websocket.query_params.get("token")
                authenticated_data = {
                    'authenticated': True,
                    'auth_method': 'dev_mode',
                    'dev_mode': True,
                    'dev_jwt': jwt_token if jwt_token else None
                }
                logger.info(f"[DEV MODE] Accepting connection: {connection_id} with correlation: {correlation_token}")
            # ===== AUTHENTICATION END =====
            
            # Extract opening line parameter
            opening_line = websocket.query_params.get("opening_line")
            if opening_line:
                logger.info(f"Opening line configured for connection {connection_id}: {opening_line[:50]}...")
            
            await self._event_bus.emit(
                f"connection:{connection_id}:established",
                connection_id=connection_id,
                session_id=session_id,
                correlation_token=correlation_token,
                opening_line=opening_line,
                has_opening_line=bool(opening_line),
                client_ip=websocket.client.host if websocket.client else None,
                user_agent=websocket.headers.get("user-agent"),
                **authenticated_data
            )
            
            participant = Participant(
                connection_id=connection_id,
                session_id=session_id,
                user_agent=websocket.headers.get("user-agent"),
                ip_address=websocket.client.host if websocket.client else None,
                metadata={
                    "session_id": session_id,
                    "correlation_token": correlation_token,
                    "opening_line": opening_line,
                    "has_opening_line": bool(opening_line),
                    **authenticated_data
                }
            )
            
            config = self._build_config_from_params(websocket)
            
            enable_processors = websocket.query_params.get("enable_processors", "true").lower() == "true"
            
            conversation = await self.conversation_manager.create_conversation(
                participant=participant,
                config=config
            )
            
            transcript = await self.transcript_storage.create_transcript(
                session_id=session_id,
                conversation_id=conversation.id,
                correlation_token=correlation_token
            )
            
            # Store opening line in transcript metadata if present
            metadata_update = {
                "simulation_attempt_id": correlation_token,
                "connected_at": datetime.utcnow().isoformat(),
                "has_opening_line": bool(opening_line),
                **authenticated_data
            }
            if opening_line:
                metadata_update["opening_line"] = opening_line
                
            await self.transcript_storage.update_metadata(
                conversation_id=conversation.id,
                metadata=metadata_update
            )
            
            aiohttp_session = aiohttp.ClientSession()
            self._aiohttp_sessions[connection_id] = aiohttp_session
            
            transport = self._create_transport(websocket, config)
            
            # Pass opening line to pipeline factory
            pipeline, output_sample_rate = await self.pipeline_factory.create_pipeline(
                config=config,
                transport=transport,
                conversation_id=conversation.id,
                aiohttp_session=aiohttp_session,
                enable_processors=enable_processors,
                correlation_token=correlation_token,
                opening_line=opening_line  # Add opening line parameter
            )
            
            self._active_connections[connection_id] = {
                "websocket": websocket,
                "participant": participant,
                "conversation": conversation,
                "transport": transport,
                "pipeline": pipeline,
                "aiohttp_session": aiohttp_session,
                "connected_at": datetime.utcnow(),
                "processors_enabled": enable_processors,
                "correlation_token": correlation_token,
                "opening_line": opening_line,
                "has_opening_line": bool(opening_line),
                **authenticated_data
            }
            
            success = await self.conversation_manager.start_conversation(
                conversation.id,
                transport,
                pipeline
            )
            
            if not success:
                await websocket.close(code=1011, reason="Failed to start conversation")
                return
            
            task = self.pipeline_factory.create_pipeline_task(
                pipeline=pipeline,
                config=config,
                output_sample_rate=output_sample_rate
            )
            
            logger.info(
                f"Starting pipeline for conversation {conversation.id} "
                f"(processors {'enabled' if enable_processors else 'disabled'}, "
                f"correlation: {correlation_token}, "
                f"auth: {authenticated_data.get('auth_method', 'none')}, "
                f"opening_line: {'yes' if opening_line else 'no'})"
            )
            
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
        
        # Helper function to safely convert to float
        def safe_float(value, default=None):
            """Safely convert a value to float, returning default if None or invalid"""
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        
        config = ConversationConfig(
            sample_rate=int(params.get("sample_rate", self._settings.default_sample_rate)),
            channels=int(params.get("channels", self._settings.default_channels)),
            audio_format=params.get("audio_format", "pcm"),
            
            stt_provider=params.get("stt_provider", self._settings.default_stt_provider),
            stt_model=params.get("stt_model", self._settings.default_stt_model),
            llm_provider=params.get("llm_provider", self._settings.default_llm_provider),
            llm_model=params.get("llm_model", self._settings.default_llm_model),
            tts_provider=params.get("tts_provider", self._settings.default_tts_provider),
            tts_model=params.get("tts_model", self._settings.default_tts_model),
            tts_voice=params.get("tts_voice", self._settings.default_tts_voice),
            tts_speed=safe_float(params.get("tts_speed")),  # Will be None if not provided
            tts_temperature=safe_float(params.get("tts_temperature")),  # Will be None if not provided
            system_prompt=params.get("system_prompt", self._settings.default_system_prompt),
            enable_interruptions=params.get("enable_interruptions", "true").lower() == "true",
            vad_enabled=params.get("vad_enabled", "true").lower() == "true",
            vad_threshold=float(params.get("vad_threshold", 0.5)),
            
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
            audio_frame_size=config.sample_rate // 200,
            send_silent_audio=False
        )
        
        return FastAPIWebsocketTransport(
            websocket=websocket,
            params=params
        )
    
    async def _handle_disconnect(self, connection_id: str):
        """Handle client disconnect"""
        logger.info(f"WebSocket disconnected: {connection_id}")
        
        conn_info = self._active_connections.get(connection_id)
        if conn_info:
            conversation = conn_info["conversation"]
            
            await self.transcript_storage.end_transcript(conversation.id)
            await self.conversation_manager.end_conversation(conversation.id)
            
            await self._event_bus.emit(
                f"connection:{connection_id}:closed",
                connection_id=connection_id,
                session_id=conversation.participant.session_id,
                correlation_token=conn_info.get("correlation_token"),
                opening_line=conn_info.get("opening_line"),
                has_opening_line=conn_info.get("has_opening_line", False),
                reason="client_disconnect",
                processors_enabled=conn_info.get("processors_enabled", True)
            )
    
    async def _handle_error(self, connection_id: str, error: Exception):
        """Handle connection error"""
        logger.error(f"Connection error for {connection_id}: {error}")
        
        conn_info = self._active_connections.get(connection_id)
        if conn_info:
            websocket = conn_info["websocket"]
            conversation = conn_info["conversation"]
            
            await self.transcript_storage.end_transcript(conversation.id)
            await self.transcript_storage.update_metadata(
                conversation.id,
                {"error": str(error), "error_type": type(error).__name__}
            )
            
            try:
                await websocket.close(code=1011, reason=str(error))
            except:
                pass
            
            await self.conversation_manager.end_conversation(
                conversation.id,
                error=str(error)
            )
            
            await self._event_bus.emit(
                f"connection:{connection_id}:error",
                connection_id=connection_id,
                session_id=conversation.participant.session_id,
                correlation_token=conn_info.get("correlation_token"),
                opening_line=conn_info.get("opening_line"),
                has_opening_line=conn_info.get("has_opening_line", False),
                error_type=type(error).__name__,
                error_message=str(error)
            )
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up connection resources"""
        if connection_id in self._active_connections:
            conn_info = self._active_connections[connection_id]
            conversation = conn_info["conversation"]
            
            await self.transcript_storage.end_transcript(conversation.id)
            await self.conversation_manager.end_conversation(conversation.id)
            
            if connection_id in self._aiohttp_sessions:
                session = self._aiohttp_sessions[connection_id]
                await session.close()
                del self._aiohttp_sessions[connection_id]
            
            del self._active_connections[connection_id]
    
    async def shutdown(self):
        """Shutdown handler and cleanup all connections"""
        logger.info("Shutting down WebSocket handler")
        
        connection_ids = list(self._active_connections.keys())
        for conn_id in connection_ids:
            await self._cleanup_connection(conn_id)
        
        for session in self._aiohttp_sessions.values():
            await session.close()
        self._aiohttp_sessions.clear()


_websocket_handler: Optional[WebSocketConnectionHandler] = None


def get_websocket_handler() -> WebSocketConnectionHandler:
    """Get the singleton WebSocket handler"""
    global _websocket_handler
    if _websocket_handler is None:
        _websocket_handler = WebSocketConnectionHandler()
    return _websocket_handler