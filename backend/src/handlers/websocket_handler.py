"""
WebSocket connection handler for voice conversations with Logfire debugging and JWT tracking.
"""
import asyncio
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import logging
import uuid
import aiohttp
import os
import jwt  # Added for JWT decoding
import json

from src.domains.conversation import Participant, ConversationConfig
from src.services.conversation_manager import get_conversation_manager
from src.services.pipeline_factory import get_pipeline_factory
from src.services.transcript_storage import get_transcript_storage
from src.config.settings import get_settings
from src.events import EventBus, get_event_bus
from src.services.logfire_service import get_logfire
from src.services.connection_monitor import get_connection_monitor
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
    """Handles WebSocket connections for voice conversations with enhanced debugging"""
    
    def __init__(self):
        self.conversation_manager = get_conversation_manager()
        self.pipeline_factory = get_pipeline_factory()
        self.transcript_storage = get_transcript_storage()
        self._settings = get_settings()
        self._event_bus = get_event_bus()
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._aiohttp_sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # Add connection-specific lock for thread safety
        self._connections_lock = asyncio.Lock()
        
        # Add Logfire and monitoring
        self.logfire = get_logfire()
        self.monitor = get_connection_monitor()
        
        # ===== DEV MODE CONFIGURATION =====
        self.AUTH_DEV_MODE = os.getenv('AUTH_DEV_MODE', 'false').lower() == 'true'
        if self.AUTH_DEV_MODE:
            logger.warning("⚠️ AUTH DEV MODE ENABLED - Token validation bypassed!")
        # ===== END DEV MODE CONFIGURATION =====
    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None
    ):
        """Handle a new WebSocket connection with comprehensive Logfire tracking"""
        connection_id = str(uuid.uuid4())
        session_id = session_id or connection_id
        aiohttp_session = None
        correlation_token = websocket.query_params.get("correlation_token")
        
        # Initialize JWT data for tracking
        jwt_data = {}
        attempt_id = None
        student_id = None
        
        try:
            # ===== DECODE JWT FOR LOGGING (BEFORE AUTH) =====
            jwt_token = websocket.query_params.get("token")
            
            if jwt_token:
                try:
                    # Decode WITHOUT verification just for logging
                    jwt_payload = jwt.decode(
                        jwt_token,
                        options={"verify_signature": False}
                    )
                    
                    # Extract session identifiers from JWT
                    jwt_data = {
                        'attempt_id': jwt_payload.get('attemptId'),
                        'student_id': jwt_payload.get('studentId'),
                        'jwt_correlation': jwt_payload.get('correlationToken'),
                        'token_type': jwt_payload.get('type'),
                        'token_issued_at': jwt_payload.get('iat'),
                        'token_expires_at': jwt_payload.get('exp'),
                    }
                    
                    # Store for easy access
                    attempt_id = jwt_data.get('attempt_id')
                    student_id = jwt_data.get('student_id')
                    
                    # Calculate token timing
                    if jwt_data.get('token_issued_at') and jwt_data.get('token_expires_at'):
                        jwt_data['token_validity_seconds'] = jwt_data['token_expires_at'] - jwt_data['token_issued_at']
                        current_timestamp = int(datetime.utcnow().timestamp())
                        jwt_data['token_expires_in'] = jwt_data['token_expires_at'] - current_timestamp
                        
                        # Check if token is about to expire
                        if jwt_data['token_expires_in'] < 60:  # Less than 1 minute
                            logger.warning(f"Token expiring soon for {attempt_id}: {jwt_data['token_expires_in']}s remaining")
                    
                    # Verify correlation tokens match
                    if jwt_data.get('jwt_correlation') and correlation_token:
                        jwt_data['correlation_match'] = (jwt_data['jwt_correlation'] == correlation_token)
                        if not jwt_data['correlation_match']:
                            logger.warning(
                                f"Correlation mismatch! JWT: {jwt_data['jwt_correlation']}, Query: {correlation_token}"
                            )
                    
                    logger.info(f"JWT decoded - attempt: {attempt_id}, student: {student_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to decode JWT for logging: {e}")
                    jwt_data = {'jwt_decode_error': str(e)}
            # ===== END JWT DECODING =====
        
        except Exception as e:
            logger.error(f"Error in JWT pre-processing: {e}")
            jwt_data = {'jwt_preprocessing_error': str(e)}
        
        # Start Logfire span with JWT context
        async with self.logfire.async_track_connection(
            connection_id=connection_id,
            session_id=session_id,
            correlation_token=correlation_token,
            attempt_id=attempt_id,
            student_id=student_id
        ):
            try:
                # Accept WebSocket connection
                await websocket.accept()
                websocket._connection_id = connection_id  # Store ID on WebSocket object

                # Log client details to identify if it's the same client
                client_info = {
                    "connection_id": connection_id,
                    "client_host": websocket.client.host if websocket.client else None,
                    "client_port": websocket.client.port if websocket.client else None,
                    "headers": dict(websocket.headers),
                }
                logger.info(f"[DEBUG] WebSocket client info: {json.dumps(client_info)}")
                
                # Filter out duplicate keys from jwt_data before using it
                jwt_data_clean = {k: v for k, v in jwt_data.items() 
                                if k not in ['attempt_id', 'student_id']}
                
                # Log initial connection with JWT data
                self.logfire.log_connection_event(
                    connection_id=connection_id,
                    event="accepted",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    client_ip=websocket.client.host if websocket.client else None,
                    user_agent=websocket.headers.get("user-agent"),
                    query_params=dict(websocket.query_params),
                    **jwt_data_clean
                )
                
                logger.info(f"WebSocket connection established: {connection_id}")
                logger.info(f"Session identifiers - attempt: {attempt_id}, student: {student_id}")
                
                # ===== AUTHENTICATION START =====
                authenticated_data = {}
                
                if not self.AUTH_DEV_MODE:
                    # PRODUCTION MODE: Validate token
                    if not jwt_token:
                        self.logfire.log_disconnection(
                            connection_id=connection_id,
                            reason="no_auth_token",
                            attempt_id=attempt_id,
                            student_id=student_id,
                            session_id=session_id
                        )
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
                            self.logfire.log_disconnection(
                                connection_id=connection_id,
                                reason="invalid_auth_token",
                                attempt_id=attempt_id,
                                student_id=student_id,
                                session_id=session_id
                            )
                            logger.warning(f"Invalid JWT token for connection {connection_id}")
                            await websocket.send_json({
                                "type": "error",
                                "error": "Invalid or expired authentication token"
                            })
                            await websocket.close(code=1008, reason="Invalid token")
                            return
                    else:
                        self.logfire.log_disconnection(
                            connection_id=connection_id,
                            reason="auth_system_error",
                            attempt_id=attempt_id,
                            student_id=student_id,
                            session_id=session_id
                        )
                        logger.error("Token validator not available")
                        await websocket.send_json({
                            "type": "error",
                            "error": "Authentication system unavailable"
                        })
                        await websocket.close(code=1011, reason="Auth system error")
                        return
                    
                    # Check for correlation token
                    if not correlation_token:
                        self.logfire.log_disconnection(
                            connection_id=connection_id,
                            reason="no_correlation_token",
                            attempt_id=attempt_id,
                            student_id=student_id,
                            session_id=session_id
                        )
                        logger.warning(f"No correlation token provided for connection {connection_id}")
                        await websocket.send_json({
                            "type": "error",
                            "error": "Correlation token required for transcript tracking"
                        })
                        await websocket.close(code=1008, reason="No correlation token")
                        return
                    
                    # Merge JWT data with auth data
                    authenticated_data = {
                        **jwt_data_clean,
                        'authenticated': True,
                        'auth_method': 'jwt',
                        'actual_correlation': correlation_token
                    }
                    
                    logger.info(f"Authenticated connection: {session_id} with correlation: {correlation_token}")
                    
                else:
                    # ===== CRITICAL FIX: Ensure unique correlation tokens in DEV MODE =====
                    if not correlation_token:
                        # Include connection_id to ensure uniqueness
                        correlation_token = f"dev_token_{connection_id}_{datetime.utcnow().timestamp()}"
                        logger.info(f"[DEV MODE] Generated unique correlation token: {correlation_token}")
                    else:
                        # Even if provided in dev mode, make it unique by appending connection_id
                        # This prevents two dev connections with same token from interfering
                        original_token = correlation_token
                        correlation_token = f"{correlation_token}_{connection_id}"
                        logger.info(f"[DEV MODE] Made correlation token unique: {original_token} -> {correlation_token}")
                    
                    authenticated_data = {
                        **jwt_data_clean,
                        'authenticated': True,
                        'auth_method': 'dev_mode',
                        'dev_mode': True
                    }
                    logger.info(f"[DEV MODE] Accepting connection: {connection_id} with correlation: {correlation_token}")
                # ===== AUTHENTICATION END =====
                
                # Add to connection monitor with JWT context
                self.monitor.add_connection(
                    connection_id=connection_id,
                    websocket=websocket,
                    session_id=session_id,
                    correlation_token=correlation_token,
                    attempt_id=attempt_id,
                    student_id=student_id,
                    client_ip=websocket.client.host if websocket.client else None,
                    **jwt_data_clean
                )
                
                # Check token expiration
                if jwt_data.get('token_expires_in') and jwt_data['token_expires_in'] < 60:
                    self.logfire.log_connection_event(
                        connection_id=connection_id,
                        event="token_expiring_soon",
                        attempt_id=attempt_id,
                        student_id=student_id,
                        expires_in_seconds=jwt_data['token_expires_in']
                    )
                
                # Extract opening line parameter
                opening_line = websocket.query_params.get("opening_line")
                if opening_line:
                    logger.info(f"Opening line configured for connection {connection_id}: {opening_line[:50]}...")
                    self.logfire.log_connection_event(
                        connection_id=connection_id,
                        event="opening_line_configured",
                        attempt_id=attempt_id,
                        student_id=student_id,
                        has_opening_line=True
                    )
                
                # Filter authenticated_data to avoid duplicates
                authenticated_data_clean = {k: v for k, v in authenticated_data.items() 
                                        if k not in ['attempt_id', 'student_id']}
                
                # Emit connection established event with JWT context
                await self._event_bus.emit(
                    f"connection:{connection_id}:established",
                    connection_id=connection_id,
                    session_id=session_id,
                    correlation_token=correlation_token,
                    attempt_id=attempt_id,
                    student_id=student_id,
                    opening_line=opening_line,
                    has_opening_line=bool(opening_line),
                    client_ip=websocket.client.host if websocket.client else None,
                    user_agent=websocket.headers.get("user-agent"),
                    **authenticated_data_clean
                )
                
                # Create participant with JWT metadata
                participant = Participant(
                    connection_id=connection_id,
                    session_id=session_id,
                    user_agent=websocket.headers.get("user-agent"),
                    ip_address=websocket.client.host if websocket.client else None,
                    metadata={
                        "session_id": session_id,
                        "correlation_token": correlation_token,
                        "attempt_id": attempt_id,
                        "student_id": student_id,
                        "opening_line": opening_line,
                        "has_opening_line": bool(opening_line),
                        "jwt_data": jwt_data_clean,
                        **authenticated_data_clean
                    }
                )
                
                # Build configuration
                config = self._build_config_from_params(websocket)
                
                # Log configuration with JWT context
                self.logfire.log_pipeline_event(
                    connection_id=connection_id,
                    event="config_built",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    stt_provider=config.stt_provider,
                    stt_model=config.stt_model,
                    llm_provider=config.llm_provider,
                    llm_model=config.llm_model,
                    tts_provider=config.tts_provider,
                    tts_model=config.tts_model,
                    tts_voice=config.tts_voice
                )
                
                enable_processors = websocket.query_params.get("enable_processors", "true").lower() == "true"
                
                # Create conversation
                conversation = await self.conversation_manager.create_conversation(
                    participant=participant,
                    config=config
                )
                
                self.logfire.log_connection_event(
                    connection_id=connection_id,
                    event="conversation_created",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    conversation_id=conversation.id
                )
                
                # Create transcript
                transcript = await self.transcript_storage.create_transcript(
                    session_id=session_id,
                    conversation_id=conversation.id,
                    correlation_token=correlation_token
                )
                
                # Store all metadata including JWT data in transcript
                metadata_update = {
                    "simulation_attempt_id": correlation_token,
                    "connected_at": datetime.utcnow().isoformat(),
                    "has_opening_line": bool(opening_line),
                    "attempt_id": attempt_id,
                    "student_id": student_id,
                    **jwt_data_clean,
                    **authenticated_data_clean
                }
                if opening_line:
                    metadata_update["opening_line"] = opening_line
                    
                await self.transcript_storage.update_metadata(
                    conversation_id=conversation.id,
                    metadata=metadata_update
                )
                
                # ===== CRITICAL FIX: Create isolated aiohttp session =====
                aiohttp_session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(
                        force_close=True,
                        enable_cleanup_closed=True,
                        limit=10,  # Limit total connections for this session
                        limit_per_host=5  # Limit per host for this session
                    )
                )
                
                # Store with lock to prevent race conditions
                async with self._connections_lock:
                    self._aiohttp_sessions[connection_id] = aiohttp_session
                
                # Create transport
                transport = self._create_transport(websocket, config)
                
                self.logfire.log_pipeline_event(
                    connection_id=connection_id,
                    event="creating_pipeline",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    processors_enabled=enable_processors
                )
                
                # Create pipeline
                pipeline, output_sample_rate = await self.pipeline_factory.create_pipeline(
                    config=config,
                    transport=transport,
                    conversation_id=conversation.id,
                    aiohttp_session=aiohttp_session,
                    enable_processors=enable_processors,
                    correlation_token=correlation_token,
                    opening_line=opening_line
                )
                
                self.logfire.log_pipeline_event(
                    connection_id=connection_id,
                    event="pipeline_created",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    pipeline_id=id(pipeline),
                    output_sample_rate=output_sample_rate
                )
                
                # Store connection info with lock and cleanup flag
                async with self._connections_lock:
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
                        "attempt_id": attempt_id,
                        "student_id": student_id,
                        "opening_line": opening_line,
                        "has_opening_line": bool(opening_line),
                        "cleanup_in_progress": False,  # Add flag to prevent double cleanup
                        **authenticated_data_clean
                    }
                
                # Start conversation
                success = await self.conversation_manager.start_conversation(
                    conversation.id,
                    transport,
                    pipeline
                )
                
                if not success:
                    self.logfire.log_disconnection(
                        connection_id=connection_id,
                        reason="failed_to_start_conversation",
                        attempt_id=attempt_id,
                        student_id=student_id,
                        conversation_id=conversation.id
                    )
                    await websocket.close(code=1011, reason="Failed to start conversation")
                    return
                
                # Create pipeline task
                task = self.pipeline_factory.create_pipeline_task(
                    pipeline=pipeline,
                    config=config,
                    output_sample_rate=output_sample_rate
                )
                
                self.logfire.log_pipeline_event(
                    connection_id=connection_id,
                    event="starting",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    conversation_id=conversation.id
                )
                
                logger.info(
                    f"Starting pipeline for conversation {conversation.id} "
                    f"(attempt: {attempt_id}, student: {student_id}, "
                    f"processors {'enabled' if enable_processors else 'disabled'}, "
                    f"correlation: {correlation_token})"
                )
                
                # Run the pipeline - this blocks until completion
                await self.conversation_manager.run_pipeline_for_conversation(
                    conversation.id,
                    task
                )
                
            except WebSocketDisconnect as e:
                # Normal WebSocket disconnection
                self.logfire.log_disconnection(
                    connection_id=connection_id,
                    reason="websocket_disconnect",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    error=e,
                    code=getattr(e, 'code', None),
                    ws_reason=getattr(e, 'reason', None)
                )
                await self._handle_disconnect(connection_id)
                
            except asyncio.CancelledError as e:
                # Task was cancelled
                self.logfire.log_disconnection(
                    connection_id=connection_id,
                    reason="task_cancelled",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    error=e
                )
                raise
                
            except aiohttp.ClientError as e:
                # HTTP client error (API calls)
                self.logfire.log_disconnection(
                    connection_id=connection_id,
                    reason="aiohttp_client_error",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    error=e,
                    error_details=str(e)
                )
                await self._handle_error(connection_id, e)
                
            except Exception as e:
                # Unexpected error
                self.logfire.log_disconnection(
                    connection_id=connection_id,
                    reason="unexpected_error",
                    attempt_id=attempt_id,
                    student_id=student_id,
                    error=e,
                    error_type=type(e).__name__,
                    error_details=str(e)
                )
                logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
                await self._handle_error(connection_id, e)
                
            finally:
                # ===== CRITICAL: Proper cleanup with isolation =====
                # Mark connection for cleanup to prevent monitor from interfering
                self.monitor.mark_connection_for_closure(connection_id)
                
                # Clean up aiohttp session for THIS connection only
                async with self._connections_lock:
                    if connection_id in self._aiohttp_sessions:
                        try:
                            session = self._aiohttp_sessions[connection_id]
                            await session.close()
                            del self._aiohttp_sessions[connection_id]
                            logger.debug(f"Closed aiohttp session for connection {connection_id}")
                        except Exception as e:
                            logger.error(f"Error closing aiohttp session for {connection_id}: {e}")
                
                # Check if cleanup is needed
                needs_cleanup = False
                async with self._connections_lock:
                    if connection_id in self._active_connections:
                        conn_info = self._active_connections.get(connection_id)
                        if conn_info and not conn_info.get("cleanup_in_progress", False):
                            conn_info["cleanup_in_progress"] = True
                            needs_cleanup = True
                
                # Perform cleanup if needed
                if needs_cleanup:
                    await self._cleanup_single_connection(connection_id)
                
                # Remove from monitor last
                self.monitor.remove_connection(connection_id)
                
                logger.info(f"Connection {connection_id} fully closed and cleaned up")
    def _build_config_from_params(self, websocket: WebSocket) -> ConversationConfig:
        """Build conversation config from query parameters"""
        params = websocket.query_params
        
        def safe_float(value, default=None):
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
            tts_speed=safe_float(params.get("tts_speed")),
            tts_temperature=0.7, #safe_float(params.get("tts_temperature")),
            system_prompt=params.get("system_prompt", self._settings.default_system_prompt),
            enable_interruptions=params.get("enable_interruptions", "true").lower() == "true",
            vad_enabled=params.get("vad_enabled", "true").lower() == "true",
            vad_threshold=float(params.get("vad_threshold", 0.5)),
            
            llm_temperature=0.3, #float(params.get("llm_temperature", 0.7)),
            llm_max_tokens=4096, #int(params.get("llm_max_tokens", 4096)),
            llm_top_p=0.8 #float(params.get("llm_top_p", 1.0))
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
        """Handle client disconnect for a specific connection"""
        logger.info(f"[DISCONNECT] WebSocket disconnected: {connection_id}")
        
        # Get connection info with proper locking
        async with self._connections_lock:
            conn_info = self._active_connections.get(connection_id)
            if not conn_info:
                logger.debug(f"[DISCONNECT] Connection {connection_id} not found")
                return
            
            # Check if already being cleaned up
            if conn_info.get("cleanup_in_progress", False):
                logger.debug(f"[DISCONNECT] Connection {connection_id} already being cleaned up")
                return
            
            # Mark as being cleaned up
            conn_info["cleanup_in_progress"] = True
            
            # Extract what we need
            conversation = conn_info.get("conversation")
            correlation_token = conn_info.get("correlation_token")
            attempt_id = conn_info.get("attempt_id")
            student_id = conn_info.get("student_id")
        
        if conversation:
            # End transcript and conversation
            try:
                await self.transcript_storage.end_transcript(conversation.id)
                await self.conversation_manager.end_conversation(conversation.id)
            except Exception as e:
                logger.error(f"[DISCONNECT] Error ending conversation: {e}")
            
            # Emit disconnection event
            await self._event_bus.emit(
                f"connection:{connection_id}:closed",
                connection_id=connection_id,
                session_id=conversation.participant.session_id,
                correlation_token=correlation_token,
                attempt_id=attempt_id,
                student_id=student_id,
                reason="client_disconnect"
            )

    async def _handle_error(self, connection_id: str, error: Exception):
        """Handle connection error for a specific connection"""
        logger.error(f"[ERROR] Connection error for {connection_id}: {error}")
        
        async with self._connections_lock:
            conn_info = self._active_connections.get(connection_id)
            if not conn_info:
                return
            
            # Check if already being cleaned up
            if conn_info.get("cleanup_in_progress", False):
                return
            
            conn_info["cleanup_in_progress"] = True
            
            websocket = conn_info.get("websocket")
            conversation = conn_info.get("conversation")
            correlation_token = conn_info.get("correlation_token")
        
        if conversation:
            # Update transcript with error
            await self.transcript_storage.end_transcript(conversation.id)
            await self.transcript_storage.update_metadata(
                conversation.id,
                {"error": str(error), "error_type": type(error).__name__}
            )
            
            # Try to close WebSocket
            if websocket:
                try:
                    await websocket.close(code=1011, reason=str(error))
                except:
                    pass
            
            # End conversation with error
            await self.conversation_manager.end_conversation(
                conversation.id,
                error=str(error)
            )
            
            # Emit error event
            await self._event_bus.emit(
                f"connection:{connection_id}:error",
                connection_id=connection_id,
                session_id=conversation.participant.session_id,
                correlation_token=correlation_token,
                error_type=type(error).__name__,
                error_message=str(error)
            )

    async def shutdown(self):
        """Shutdown handler and cleanup all connections"""
        logger.info("Shutting down WebSocket handler")
        
        # Get all connection IDs
        async with self._connections_lock:
            connection_ids = list(self._active_connections.keys())
        
        # Clean up each connection
        for conn_id in connection_ids:
            await self._cleanup_single_connection(conn_id)
        
        # Clean up remaining aiohttp sessions
        async with self._connections_lock:
            for session in self._aiohttp_sessions.values():
                try:
                    await session.close()
                except:
                    pass
            self._aiohttp_sessions.clear()
            self._active_connections.clear()

    def is_connection_valid(self, connection_id: str) -> bool:
        """Check if a connection is valid and active"""
        with self._connections_lock:
            if connection_id not in self._active_connections:
                return False
            
            conn_info = self._active_connections[connection_id]
            websocket = conn_info.get("websocket")
            
            if not websocket:
                return False
            
            # Check WebSocket state
            try:
                return websocket.client_state.name == "CONNECTED"
            except:
                return False
    
    async def _cleanup_single_connection(self, connection_id: str, skip_conversation_end: bool = False):
            """Clean up a single connection's resources without affecting others"""
            logger.info(f"[CLEANUP] Starting cleanup for connection: {connection_id}")
            
            # Variables to track what needs cleanup
            conversation_id = None
            aiohttp_session = None
            correlation_token = None
            
            # Get connection info and mark for cleanup
            async with self._connections_lock:
                if connection_id not in self._active_connections:
                    logger.debug(f"[CLEANUP] Connection {connection_id} not found, already cleaned")
                    return
                
                conn_info = self._active_connections.get(connection_id)
                if not conn_info:
                    return
                
                # Extract needed info
                conversation = conn_info.get("conversation")
                conversation_id = conversation.id if conversation else None
                aiohttp_session = conn_info.get("aiohttp_session")
                correlation_token = conn_info.get("correlation_token")
            
            # Perform cleanup operations outside the lock
            
            # End transcript for THIS connection only
            if conversation_id:
                try:
                    await self.transcript_storage.end_transcript(conversation_id)
                    logger.debug(f"[CLEANUP] Ended transcript for conversation {conversation_id}")
                except Exception as e:
                    logger.error(f"[CLEANUP] Error ending transcript for {conversation_id}: {e}")
            
            # End conversation for THIS connection only (unless skipped)
            if conversation_id and not skip_conversation_end:
                try:
                    await self.conversation_manager.end_conversation(conversation_id)
                    logger.debug(f"[CLEANUP] Ended conversation {conversation_id}")
                except Exception as e:
                    logger.error(f"[CLEANUP] Error ending conversation {conversation_id}: {e}")
            
            # Close aiohttp session for THIS connection only
            if aiohttp_session:
                try:
                    await aiohttp_session.close()
                    logger.debug(f"[CLEANUP] Closed aiohttp session for {connection_id}")
                except Exception as e:
                    logger.error(f"[CLEANUP] Error closing aiohttp session for {connection_id}: {e}")
            
            # Remove from collections
            async with self._connections_lock:
                # Remove from aiohttp sessions
                if connection_id in self._aiohttp_sessions:
                    del self._aiohttp_sessions[connection_id]
                    logger.debug(f"[CLEANUP] Removed aiohttp session reference for {connection_id}")
                
                # Remove from active connections
                if connection_id in self._active_connections:
                    del self._active_connections[connection_id]
                    logger.debug(f"[CLEANUP] Removed from active connections: {connection_id}")
            
            # Remove from monitor (outside lock)
            try:
                self.monitor.remove_connection(connection_id)
            except Exception as e:
                logger.error(f"[CLEANUP] Error removing from monitor: {e}")
            
            logger.info(f"[CLEANUP] Completed cleanup for connection: {connection_id} (correlation: {correlation_token})")

    async def _cleanup_connection(self, connection_id: str):
        """Alias for _cleanup_single_connection for backward compatibility"""
        await self._cleanup_single_connection(connection_id)
        
    async def close_connection_by_correlation(self, correlation_token: str) -> int:
        """Close ONLY connections with the exact correlation token"""
        connections_closed = 0
        connections_to_close = []
        
        logger.info(f"[FORCE_CLOSE] Searching for connections with correlation token: {correlation_token}")
        
        # Get a snapshot of connections with lock
        async with self._connections_lock:
            # Create a deep copy to avoid issues during iteration
            active_connections_snapshot = {}
            for conn_id, conn_info in self._active_connections.items():
                active_connections_snapshot[conn_id] = dict(conn_info)
        
        # Find ONLY connections with exact correlation token match
        for connection_id, conn_info in active_connections_snapshot.items():
            conn_correlation = conn_info.get("correlation_token")
            # CRITICAL: Exact match only
            if conn_correlation and conn_correlation == correlation_token:
                connections_to_close.append(connection_id)
                logger.info(f"[FORCE_CLOSE] Found connection to close: {connection_id} (correlation: {conn_correlation})")
            else:
                logger.debug(f"[FORCE_CLOSE] Skipping connection {connection_id} (correlation: {conn_correlation} != {correlation_token})")
        
        # If no connections found, log it
        if not connections_to_close:
            logger.warning(f"[FORCE_CLOSE] No connections found with correlation token: {correlation_token}")
            return 0
        
        # Process each connection individually
        for connection_id in connections_to_close:
            try:
                # Verify connection still exists and has the same correlation token
                async with self._connections_lock:
                    current_conn_info = self._active_connections.get(connection_id)
                    if not current_conn_info:
                        logger.warning(f"[FORCE_CLOSE] Connection {connection_id} no longer exists")
                        continue
                    
                    # Double-check correlation token hasn't changed
                    if current_conn_info.get("correlation_token") != correlation_token:
                        logger.warning(f"[FORCE_CLOSE] Connection {connection_id} correlation token changed, skipping")
                        continue
                    
                    # Check if already being cleaned
                    if current_conn_info.get("cleanup_in_progress", False):
                        logger.debug(f"[FORCE_CLOSE] Connection {connection_id} already being cleaned")
                        continue
                    
                    # Mark as being cleaned to prevent concurrent cleanup
                    current_conn_info["cleanup_in_progress"] = True
                    
                    # Extract what we need for cleanup
                    websocket = current_conn_info.get("websocket")
                    pipeline = current_conn_info.get("pipeline")
                    conversation = current_conn_info.get("conversation")
                    conversation_id = conversation.id if conversation else None
                
                # Mark in monitor to prevent health checks (outside lock)
                self.monitor.mark_connection_for_closure(connection_id)
                self.monitor.mark_websocket_closed(connection_id)
                
                # Stop billing processor if exists
                if pipeline and hasattr(pipeline, 'processors'):
                    try:
                        for processor in pipeline.processors:
                            if hasattr(processor, '__class__') and processor.__class__.__name__ == 'BillingProcessor':
                                if hasattr(processor, 'stop_billing'):
                                    await processor.stop_billing("api_force_close")
                                    logger.debug(f"[FORCE_CLOSE] Stopped billing for {connection_id}")
                                    break
                    except Exception as e:
                        logger.error(f"[FORCE_CLOSE] Error stopping billing for {connection_id}: {e}")
                
                # Send termination message if WebSocket is still connected
                if websocket:
                    try:
                        if hasattr(websocket, 'client_state') and websocket.client_state.name == "CONNECTED":
                            try:
                                await asyncio.wait_for(
                                    websocket.send_json({
                                        "type": "session_terminated",
                                        "reason": "simulation_ended",
                                        "message": "Your simulation session has been ended by the system",
                                        "timestamp": datetime.utcnow().isoformat()
                                    }),
                                    timeout=1.0
                                )
                            except (asyncio.TimeoutError, Exception):
                                pass
                            
                            # Close WebSocket
                            await websocket.close(code=1000, reason="Simulation ended by system")
                            logger.debug(f"[FORCE_CLOSE] Closed WebSocket for {connection_id}")
                    except Exception as e:
                        logger.debug(f"[FORCE_CLOSE] WebSocket close error for {connection_id}: {e}")
                
                # End conversation if exists
                if conversation_id:
                    try:
                        await self.conversation_manager.end_conversation(conversation_id)
                        logger.debug(f"[FORCE_CLOSE] Ended conversation {conversation_id}")
                    except Exception as e:
                        logger.error(f"[FORCE_CLOSE] Error ending conversation {conversation_id}: {e}")
                
                # Clean up THIS connection's resources only
                await self._cleanup_single_connection(connection_id, skip_conversation_end=True)
                connections_closed += 1
                
                logger.info(f"[FORCE_CLOSE] Successfully closed connection {connection_id}")
                
            except Exception as e:
                logger.error(f"[FORCE_CLOSE] Error processing connection {connection_id}: {e}", exc_info=True)
        
        logger.info(f"[FORCE_CLOSE] Closed {connections_closed} connection(s) for correlation token: {correlation_token}")
        return connections_closed
    

_websocket_handler: Optional[WebSocketConnectionHandler] = None


def get_websocket_handler() -> WebSocketConnectionHandler:
    """Get the singleton WebSocket handler"""
    global _websocket_handler
    if _websocket_handler is None:
        _websocket_handler = WebSocketConnectionHandler()
    return _websocket_handler