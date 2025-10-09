"""
Logfire logging service for debugging WebSocket disconnections
"""
import logfire
from typing import Optional, Dict, Any
import os
from datetime import datetime
import asyncio
from contextlib import contextmanager, asynccontextmanager
import logging

class LogfireService:
    """Centralized Logfire logging service focused on connection debugging"""
    
    _instance: Optional['LogfireService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Initialize Logfire with configuration"""
        # Get token from environment
        token = os.getenv('LOGFIRE_TOKEN')
        
        if token:
            # Production mode with token
            logfire.configure(
                token=token,
                service_name="voice-agent",
                service_version="2.0.0",
                send_to_logfire=True,
                scrubbing=False
            )
            print(f"✅ Logfire configured with token (production mode)")
        else:
            # Development mode - will create temp project
            logfire.configure(
                service_name="voice-agent-debug",
                service_version="2.0.0",
                send_to_logfire=False,  # Only local in dev
                scrubbing=False
            )
            print(f"⚠️ Logfire running in development mode (no token)")
        
        logfire.info("🚀 Logfire service initialized for disconnection debugging")
        
        # ===== COMPREHENSIVE PYTHON LOGGING BRIDGE =====
        self._setup_comprehensive_logging_bridge()
    
    def _setup_comprehensive_logging_bridge(self):
        """Setup comprehensive logging bridge to capture ALL Python logs"""
        try:
            class LogfireHandler(logging.Handler):
                """Custom handler that sends all logs to Logfire"""
                
                def emit(self, record):
                    # Skip logfire's own logs to avoid recursion
                    if record.name.startswith('logfire'):
                        return
                    
                    # Skip certain noisy loggers if needed
                    skip_loggers = ['urllib3.connectionpool', 'asyncio']
                    if any(record.name.startswith(skip) for skip in skip_loggers):
                        if record.levelno < logging.WARNING:
                            return
                    
                    # Map Python log levels to Logfire functions
                    level_map = {
                        logging.DEBUG: logfire.debug,
                        logging.INFO: logfire.info,
                        logging.WARNING: logfire.warn,
                        logging.ERROR: logfire.error,
                        logging.CRITICAL: logfire.error
                    }
                    log_func = level_map.get(record.levelno, logfire.info)
                    
                    # Format the message
                    try:
                        msg = self.format(record)
                    except Exception:
                        msg = record.getMessage()
                    
                    # Prepare extra data
                    extra_data = {
                        'logger_name': record.name,
                        'level': record.levelname,
                        'pathname': record.pathname,
                        'lineno': record.lineno,
                        'funcname': record.funcName,
                        'module': record.module,
                        'thread': record.thread,
                        'thread_name': record.threadName,
                    }
                    
                    # Add exception info if present
                    if record.exc_info:
                        import traceback
                        extra_data['exc_info'] = ''.join(traceback.format_exception(*record.exc_info))
                    
                    # Log to Logfire with full context
                    try:
                        log_func(
                            f"python.{record.name}",
                            message=msg,
                            **extra_data
                        )
                    except Exception as e:
                        # Fallback to print if Logfire fails
                        print(f"Failed to log to Logfire: {e} - Original message: {msg}")
            
            # Create handler with formatter
            handler = LogfireHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter('%(message)s'))
            
            # Get root logger
            root_logger = logging.getLogger()
            
            # Check if we already have a LogfireHandler to avoid duplicates
            has_logfire_handler = any(
                isinstance(h, LogfireHandler) 
                for h in root_logger.handlers
            )
            
            if not has_logfire_handler:
                # Add our Logfire handler to root logger
                root_logger.addHandler(handler)
                
                # Set root logger level to DEBUG to capture everything
                if root_logger.level > logging.DEBUG:
                    root_logger.setLevel(logging.DEBUG)
            
            # Configure specific loggers
            loggers_to_configure = [
                "src",
                "pipecat", 
                "billing",
                "uvicorn",
                "fastapi",
                "aiohttp",
                "httpx",
                "websockets"
            ]
            
            for logger_name in loggers_to_configure:
                specific_logger = logging.getLogger(logger_name)
                # Ensure they propagate to root
                specific_logger.propagate = True
                # Set their level to DEBUG
                if specific_logger.level > logging.DEBUG:
                    specific_logger.setLevel(logging.DEBUG)
            
            print("✅ Python logging fully bridged to Logfire")
            logfire.info(
                "Python logging integration complete", 
                root_logger_configured=True,
                root_logger_level=logging.getLevelName(root_logger.level),
                handlers_count=len(root_logger.handlers),
                configured_loggers=loggers_to_configure
            )
            
        except Exception as e:
            print(f"⚠️ Could not bridge Python logging to Logfire: {e}")
            import traceback
            traceback.print_exc()
    
    @contextmanager
    def track_connection(self, connection_id: str, session_id: str, 
                        correlation_token: Optional[str] = None,
                        attempt_id: Optional[str] = None,
                        student_id: Optional[str] = None):
        """Create a span for tracking entire connection lifecycle"""
        with logfire.span(
            "websocket_connection",
            connection_id=connection_id,
            session_id=session_id,
            correlation_token=correlation_token,
            attempt_id=attempt_id,
            student_id=student_id,
            start_time=datetime.utcnow().isoformat()
        ) as span:
            # Set attributes for filtering
            if attempt_id:
                span.set_attribute("attempt.id", attempt_id)
            if student_id:
                span.set_attribute("student.id", student_id)
            if correlation_token:
                span.set_attribute("correlation.token", correlation_token)
            yield span
    
    @asynccontextmanager
    async def async_track_connection(self, connection_id: str, session_id: str, 
                                    correlation_token: Optional[str] = None,
                                    attempt_id: Optional[str] = None,
                                    student_id: Optional[str] = None):
        """Async version for tracking connection lifecycle with session identifiers"""
        with logfire.span(
            "websocket_connection",
            connection_id=connection_id,
            session_id=session_id,
            correlation_token=correlation_token,
            attempt_id=attempt_id,
            student_id=student_id,
            start_time=datetime.utcnow().isoformat()
        ) as span:
            # Set attributes for filtering/searching
            if attempt_id:
                span.set_attribute("attempt.id", attempt_id)
            if student_id:
                span.set_attribute("student.id", student_id)
            if correlation_token:
                span.set_attribute("correlation.token", correlation_token)
            yield span
    
    def log_connection_event(self, connection_id: str, event: str, 
                            attempt_id: Optional[str] = None,
                            student_id: Optional[str] = None,
                            **kwargs):
        """Log connection lifecycle events with session context"""
        # Filter out duplicate keys if they exist
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['attempt_id', 'student_id', 'connection_id', 'event']}
        
        logfire.info(
            f"connection.{event}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_kwargs
        )
    
    def log_heartbeat(self, connection_id: str, 
                     attempt_id: Optional[str] = None,
                     student_id: Optional[str] = None,
                     **kwargs):
        """Log heartbeat for connection health monitoring"""
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['attempt_id', 'student_id', 'connection_id']}
        
        logfire.debug(
            "heartbeat",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_kwargs
        )
    
    def log_disconnection(self, connection_id: str, reason: str, 
                         attempt_id: Optional[str] = None,
                         student_id: Optional[str] = None,
                         error: Optional[Exception] = None, 
                         **context):
        """Log disconnection with full context including session identifiers"""
        error_data = {}
        if error:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_args': str(error.args) if hasattr(error, 'args') else None,
            }
        
        # Filter out duplicate keys from context
        filtered_context = {k: v for k, v in context.items() 
                           if k not in ['attempt_id', 'student_id', 'connection_id', 'reason', 
                                        'error_type', 'error_message', 'error_args']}
        
        logfire.error(
            f"DISCONNECTION.{reason}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            reason=reason,
            timestamp=datetime.utcnow().isoformat(),
            **error_data,
            **filtered_context
        )
    
    def log_websocket_state(self, connection_id: str, state: str, 
                           attempt_id: Optional[str] = None,
                           student_id: Optional[str] = None,
                           **details):
        """Log WebSocket state changes with session context"""
        filtered_details = {k: v for k, v in details.items() 
                           if k not in ['attempt_id', 'student_id', 'connection_id', 'state']}
        
        logfire.info(
            f"ws_state.{state}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_details
        )
    
    def log_pipeline_event(self, connection_id: str, event: str,
                          attempt_id: Optional[str] = None,
                          student_id: Optional[str] = None,
                          **details):
        """Log pipeline events that might cause disconnections"""
        filtered_details = {k: v for k, v in details.items() 
                           if k not in ['attempt_id', 'student_id', 'connection_id', 'event']}
        
        logfire.info(
            f"pipeline.{event}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_details
        )
    
    def log_audio_event(self, connection_id: str, event: str,
                       attempt_id: Optional[str] = None,
                       student_id: Optional[str] = None,
                       **kwargs):
        """Log audio processing events"""
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['attempt_id', 'student_id', 'connection_id', 'event']}
        
        logfire.debug(
            f"audio.{event}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_kwargs
        )
    
    def log_transcript(self, connection_id: str, speaker: str, text: str,
                      attempt_id: Optional[str] = None,
                      student_id: Optional[str] = None,
                      **kwargs):
        """Log conversation transcripts"""
        # Truncate long text for logging
        display_text = text[:200] + "..." if len(text) > 200 else text
        
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['attempt_id', 'student_id', 'connection_id', 'speaker', 'text']}
        
        logfire.info(
            f"transcript.{speaker}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            speaker=speaker,
            text=display_text,
            text_length=len(text),
            timestamp=datetime.utcnow().isoformat(),
            **filtered_kwargs
        )
    
    def log_error(self, connection_id: str, error: Exception, 
                 context: str = "",
                 attempt_id: Optional[str] = None,
                 student_id: Optional[str] = None,
                 **kwargs):
        """Log errors with full context"""
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['attempt_id', 'student_id', 'connection_id', 'error', 'context']}
        
        logfire.error(
            f"error.{context}" if context else "error",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            context=context,
            error_type=type(error).__name__,
            error_message=str(error),
            error_args=str(error.args) if hasattr(error, 'args') else None,
            timestamp=datetime.utcnow().isoformat(),
            exc_info=True,
            **filtered_kwargs
        )
    
    def log_metrics(self, connection_id: str, metric_type: str,
                   attempt_id: Optional[str] = None,
                   student_id: Optional[str] = None,
                   **metrics):
        """Log performance metrics with session context"""
        filtered_metrics = {k: v for k, v in metrics.items() 
                           if k not in ['attempt_id', 'student_id', 'connection_id', 'metric_type']}
        
        logfire.info(
            f"metrics.{metric_type}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_metrics
        )
    
    def log_billing_event(self, connection_id: str, event: str,
                         attempt_id: Optional[str] = None,
                         student_id: Optional[str] = None,
                         **kwargs):
        """Log billing-related events"""
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['attempt_id', 'student_id', 'connection_id', 'event']}
        
        logfire.info(
            f"billing.{event}",
            connection_id=connection_id,
            attempt_id=attempt_id,
            student_id=student_id,
            timestamp=datetime.utcnow().isoformat(),
            **filtered_kwargs
        )

# Singleton getter
def get_logfire() -> LogfireService:
    """Get the singleton Logfire service instance"""
    return LogfireService()