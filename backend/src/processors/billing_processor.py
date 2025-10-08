"""
Billing processor with comprehensive logging for tracking and debugging.
"""
import asyncio
import aiohttp
import logging
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from pipecat.frames.frames import Frame, AudioRawFrame, EndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from src.config.settings import get_settings
from src.events import get_event_bus

# Configure dedicated billing logger
billing_logger = logging.getLogger("billing")
billing_logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
import os
os.makedirs("logs", exist_ok=True)

# Add rotating file handler for billing logs
from logging.handlers import RotatingFileHandler
billing_file_handler = RotatingFileHandler(
    'logs/billing.log',
    maxBytes=10485760,  # 10MB
    backupCount=10
)
billing_file_handler.setFormatter(
    logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
)
billing_logger.addHandler(billing_file_handler)

# Also add console handler for immediate visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - BILLING - %(levelname)s - %(message)s')
)
billing_logger.addHandler(console_handler)


class BillingState(Enum):
    """Billing state enumeration for tracking"""
    IDLE = "idle"
    WAITING_FOR_AUDIO = "waiting_for_audio"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BillingMetrics:
    """Track billing metrics for monitoring"""
    def __init__(self):
        self.webhook_attempts = 0
        self.webhook_successes = 0
        self.webhook_failures = 0
        self.total_latency_ms = 0
        self.min_latency_ms = float('inf')
        self.max_latency_ms = 0
        self.last_webhook_time = None
        self.state_transitions = []
        
    def to_dict(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_latency_ms / self.webhook_successes 
            if self.webhook_successes > 0 else 0
        )
        return {
            "webhook_attempts": self.webhook_attempts,
            "webhook_successes": self.webhook_successes,
            "webhook_failures": self.webhook_failures,
            "success_rate": (
                self.webhook_successes / self.webhook_attempts * 100 
                if self.webhook_attempts > 0 else 0
            ),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0,
            "max_latency_ms": self.max_latency_ms,
            "last_webhook_time": self.last_webhook_time.isoformat() if self.last_webhook_time else None,
            "state_transitions": self.state_transitions
        }


class BillingProcessor(FrameProcessor):
    """Processor that sends billing webhooks every minute with comprehensive logging"""
    
    def __init__(
        self, 
        conversation_id: str, 
        correlation_token: Optional[str] = None,
        backend_shared_secret: Optional[str] = None,
        transport=None  # Added transport parameter
    ):
        super().__init__()
        self.conversation_id = conversation_id
        self.correlation_token = correlation_token
        self.settings = get_settings()
        self.event_bus = get_event_bus()
        
        # Store transport for WebSocket access
        self._transport = transport
        
        # Configuration
        self.backend_url = self.settings.backend_url
        self.backend_shared_secret = backend_shared_secret or self.settings.backend_shared_secret
        
        # State tracking
        self.state = BillingState.IDLE
        self.start_time = None
        self.minutes_billed = 0
        self._billing_task = None
        self._active = False
        self._first_audio_received = False
        
        # Metrics
        self.metrics = BillingMetrics()
        self.last_webhook_response = None
        
        # Log initialization with full context
        init_data = {
            "conversation_id": conversation_id,
            "correlation_token": correlation_token,
            "backend_url": self.backend_url,
            "has_backend_shared_secret": bool(self.backend_shared_secret),
            "has_transport": bool(self._transport),
            "initial_state": self.state.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        billing_logger.info(
            f"[INIT] BillingProcessor initialized | {json.dumps(init_data)}"
        )
        
        # Emit initialization event
        asyncio.create_task(self._emit_billing_event("processor_initialized", init_data))
        
    def _transition_state(self, new_state: BillingState, reason: str = ""):
        """Track state transitions with logging"""
        old_state = self.state
        self.state = new_state
        
        transition = {
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.metrics.state_transitions.append(transition)
        
        billing_logger.info(
            f"[STATE] Transition | "
            f"conversation_id={self.conversation_id} | "
            f"from={old_state.value} | "
            f"to={new_state.value} | "
            f"reason={reason}"
        )
        
        asyncio.create_task(
            self._emit_billing_event("state_transition", transition)
        )
        
    async def _emit_billing_event(self, event_type: str, data: Dict[str, Any]):
        """Emit billing events for tracking"""
        try:
            event_name = f"conversation:{self.conversation_id}:billing:{event_type}"
            full_data = {
                "conversation_id": self.conversation_id,
                "correlation_token": self.correlation_token,
                "timestamp": datetime.utcnow().isoformat(),
                **data
            }
            await self.event_bus.emit(event_name, **full_data)
            
            billing_logger.debug(
                f"[EVENT] Emitted {event_name} | data={json.dumps(full_data)}"
            )
        except Exception as e:
            billing_logger.error(
                f"[EVENT] Failed to emit {event_type} | error={str(e)}"
            )
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with comprehensive logging"""
        await super().process_frame(frame, direction)
        
        frame_type = frame.__class__.__name__
        
        # Start billing on first audio frame
        if isinstance(frame, AudioRawFrame) and not self._first_audio_received:
            self._first_audio_received = True
            
            billing_logger.info(
                f"[AUDIO] First audio frame received | "
                f"conversation_id={self.conversation_id} | "
                f"correlation_token={self.correlation_token} | "
                f"will_start_billing={not self._billing_task}"
            )
            
            await self._emit_billing_event("first_audio_received", {
                "direction": direction.name,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            if not self._billing_task:
                self._transition_state(BillingState.WAITING_FOR_AUDIO, "first_audio_received")
                await self._start_billing()
        
        # Stop billing on end frame
        elif isinstance(frame, EndFrame):
            billing_logger.info(
                f"[END] End frame received | "
                f"conversation_id={self.conversation_id} | "
                f"minutes_billed={self.minutes_billed} | "
                f"state={self.state.value}"
            )
            
            await self._emit_billing_event("end_frame_received", {
                "minutes_billed": self.minutes_billed,
                "state": self.state.value
            })
            
            await self._stop_billing("end_frame_received")
        
        # Always pass frame through
        await self.push_frame(frame, direction)
    
    async def _start_billing(self):
        """Start the billing timer with detailed logging"""
        if self._active:
            billing_logger.warning(
                f"[START] Already active | "
                f"conversation_id={self.conversation_id} | "
                f"state={self.state.value}"
            )
            return
        
        try:
            self._transition_state(BillingState.ACTIVE, "billing_started")
            self._active = True
            self.start_time = datetime.utcnow()
            
            # IMMEDIATELY BILL THE FIRST MINUTE
            self.minutes_billed = 1
            
            start_data = {
                "conversation_id": self.conversation_id,
                "correlation_token": self.correlation_token,
                "start_time": self.start_time.isoformat(),
                "backend_url": self.backend_url,
                "initial_minute_billed": True
            }
            
            billing_logger.info(
                f"[START] Starting billing timer and billing first minute immediately | {json.dumps(start_data)}"
            )
            
            # Send the first minute billing webhook immediately
            billing_logger.info(
                f"[START] Sending initial billing webhook for minute 1 | "
                f"conversation_id={self.conversation_id}"
            )
            
            webhook_start = time.time()
            response = await self._send_billing_webhook()
            webhook_duration = (time.time() - webhook_start) * 1000
            
            webhook_result = {
                "minute": 1,
                "success": bool(response),
                "latency_ms": webhook_duration,
                "initial_billing": True
            }
            
            billing_logger.info(
                f"[START] Initial webhook result | {json.dumps(webhook_result)}"
            )
            
            if response:
                await self._handle_billing_response(response)
            else:
                await self._emit_billing_event("initial_webhook_failed", {
                    "minute": 1
                })
            
            # Create billing loop task for subsequent minutes
            self._billing_task = asyncio.create_task(
                self._billing_loop(),
                name=f"billing_loop_{self.conversation_id}"
            )
            
            # Log task creation
            billing_logger.debug(
                f"[START] Created billing task | "
                f"task_name={self._billing_task.get_name()} | "
                f"task_id={id(self._billing_task)}"
            )
            
            # Emit start event
            await self._emit_billing_event("started", start_data)
            
        except Exception as e:
            billing_logger.error(
                f"[START] Failed to start billing | "
                f"error={str(e)} | "
                f"error_type={type(e).__name__}",
                exc_info=True
            )
            self._transition_state(BillingState.ERROR, f"start_failed: {str(e)}")
            raise
    
    async def _billing_loop(self):
        """Main billing loop with comprehensive logging - starts from minute 2"""
        loop_id = f"{self.conversation_id}_{int(time.time())}"
        
        billing_logger.info(
            f"[LOOP] Started | "
            f"loop_id={loop_id} | "
            f"conversation_id={self.conversation_id} | "
            f"starting_from_minute=2"
        )
        
        iteration = 0
        
        try:
            while self._active:
                iteration += 1
                loop_start = time.time()
                
                billing_logger.debug(
                    f"[LOOP] Iteration {iteration} starting | "
                    f"loop_id={loop_id} | "
                    f"current_minute={self.minutes_billed} | "
                    f"waiting_60s=true"
                )
                
                # Wait for 1 minute
                await asyncio.sleep(60)
                
                if not self._active:
                    billing_logger.info(
                        f"[LOOP] Deactivated during wait | "
                        f"loop_id={loop_id} | "
                        f"iteration={iteration}"
                    )
                    break
                
                self.minutes_billed += 1
                
                billing_logger.info(
                    f"[LOOP] Minute elapsed | "
                    f"loop_id={loop_id} | "
                    f"iteration={iteration} | "
                    f"minute={self.minutes_billed} | "
                    f"conversation_id={self.conversation_id}"
                )
                
                # Send billing webhook
                webhook_start = time.time()
                response = await self._send_billing_webhook()
                webhook_duration = (time.time() - webhook_start) * 1000
                
                webhook_result = {
                    "minute": self.minutes_billed,
                    "success": bool(response),
                    "latency_ms": webhook_duration,
                    "iteration": iteration
                }
                
                billing_logger.info(
                    f"[LOOP] Webhook result | "
                    f"loop_id={loop_id} | "
                    f"{json.dumps(webhook_result)}"
                )
                
                if response:
                    await self._handle_billing_response(response)
                else:
                    await self._emit_billing_event("webhook_failed", {
                        "minute": self.minutes_billed,
                        "iteration": iteration,
                        "loop_id": loop_id
                    })
                
                loop_duration = (time.time() - loop_start)
                
                billing_logger.debug(
                    f"[LOOP] Iteration complete | "
                    f"loop_id={loop_id} | "
                    f"iteration={iteration} | "
                    f"total_duration_s={loop_duration:.2f} | "
                    f"webhook_duration_ms={webhook_duration:.2f}"
                )
                
        except asyncio.CancelledError:
            billing_logger.info(
                f"[LOOP] Cancelled | "
                f"loop_id={loop_id} | "
                f"total_iterations={iteration} | "
                f"minutes_billed={self.minutes_billed}"
            )
            raise
            
        except Exception as e:
            billing_logger.error(
                f"[LOOP] Error | "
                f"loop_id={loop_id} | "
                f"iteration={iteration} | "
                f"error={str(e)} | "
                f"error_type={type(e).__name__}",
                exc_info=True
            )
            
            self._transition_state(BillingState.ERROR, f"loop_error: {str(e)}")
            
            await self._emit_billing_event("loop_error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "iteration": iteration,
                "minute": self.minutes_billed
            })
            
        finally:
            billing_logger.info(
                f"[LOOP] Ended | "
                f"loop_id={loop_id} | "
                f"total_iterations={iteration} | "
                f"total_minutes={self.minutes_billed} | "
                f"final_state={self.state.value}"
            )
    
    async def _send_billing_webhook(self) -> Optional[dict]:
        """Send billing webhook with retry logic and graceful DNS failure handling"""
        request_id = f"req_{self.minutes_billed}_{int(time.time() * 1000)}"
        self.metrics.webhook_attempts += 1
        
        if not self.correlation_token:
            billing_logger.error(
                f"[WEBHOOK] Missing correlation token | "
                f"request_id={request_id} | "
                f"conversation_id={self.conversation_id}"
            )
            self.metrics.webhook_failures += 1
            return None
        
        webhook_data = {
            "correlation_token": self.correlation_token,
            "conversation_id": self.conversation_id,
            "minute": self.minutes_billed,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        webhook_url = f"{self.backend_url}/api/billing/voice-minute"
        
        billing_logger.info(
            f"[WEBHOOK] Request starting | "
            f"request_id={request_id} | "
            f"url={webhook_url} | "
            f"minute={self.minutes_billed} | "
            f"attempt={self.metrics.webhook_attempts}"
        )
        
        billing_logger.debug(
            f"[WEBHOOK] Request data | "
            f"request_id={request_id} | "
            f"data={json.dumps(webhook_data)}"
        )
        
        # Retry logic for DNS failures
        max_retries = 3
        retry_delay = 2.0
        
        for retry_attempt in range(max_retries):
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Content-Type": "application/json",
                        "X-Internal-Secret": self.backend_shared_secret,
                        "X-Request-ID": request_id
                    }
                    
                    billing_logger.debug(
                        f"[WEBHOOK] Sending POST | "
                        f"request_id={request_id} | "
                        f"retry_attempt={retry_attempt + 1}/{max_retries} | "
                        f"headers={json.dumps({k: v if k != 'X-Internal-Secret' else 'REDACTED' for k, v in headers.items()})}"
                    )
                    
                    async with session.post(
                        webhook_url,
                        json=webhook_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        latency_ms = (time.time() - start_time) * 1000
                        response_text = await response.text()
                        
                        # Update metrics
                        self.metrics.total_latency_ms += latency_ms
                        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
                        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
                        self.metrics.last_webhook_time = datetime.utcnow()
                        
                        response_log = {
                            "request_id": request_id,
                            "status": response.status,
                            "latency_ms": latency_ms,
                            "minute": self.minutes_billed,
                            "response_size": len(response_text)
                        }
                        
                        billing_logger.info(
                            f"[WEBHOOK] Response received | {json.dumps(response_log)}"
                        )
                        
                        if response.status == 200:
                            try:
                                result = json.loads(response_text)
                                self.metrics.webhook_successes += 1
                                self.last_webhook_response = result
                                
                                billing_logger.debug(
                                    f"[WEBHOOK] Success response | "
                                    f"request_id={request_id} | "
                                    f"data={json.dumps(result)}"
                                )
                                
                                await self._emit_billing_event("webhook_success", {
                                    "request_id": request_id,
                                    "minute": self.minutes_billed,
                                    "latency_ms": latency_ms,
                                    "response": result
                                })
                                
                                return result
                                
                            except json.JSONDecodeError as e:
                                billing_logger.error(
                                    f"[WEBHOOK] Invalid JSON response | "
                                    f"request_id={request_id} | "
                                    f"error={str(e)} | "
                                    f"response={response_text[:500]}"
                                )
                                self.metrics.webhook_failures += 1
                                return None
                        else:
                            self.metrics.webhook_failures += 1
                            
                            billing_logger.error(
                                f"[WEBHOOK] Failed | "
                                f"request_id={request_id} | "
                                f"status={response.status} | "
                                f"response={response_text[:500]}"
                            )
                            
                            await self._emit_billing_event("webhook_error", {
                                "request_id": request_id,
                                "status": response.status,
                                "response": response_text[:500]
                            })
                            
                            return None
                            
            except aiohttp.ClientConnectorDNSError as e:
                # DNS resolution failed - this is often transient
                billing_logger.warning(
                    f"[WEBHOOK] DNS resolution failed | "
                    f"request_id={request_id} | "
                    f"retry_attempt={retry_attempt + 1}/{max_retries} | "
                    f"minute={self.minutes_billed} | "
                    f"error={str(e)}"
                )
                
                if retry_attempt < max_retries - 1:
                    # Still have retries left
                    billing_logger.info(
                        f"[WEBHOOK] Retrying after DNS failure | "
                        f"retry_in={retry_delay}s | "
                        f"retry_attempt={retry_attempt + 2}/{max_retries}"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # All retries exhausted - return safe default
                    billing_logger.error(
                        f"[WEBHOOK] DNS resolution failed after all retries | "
                        f"request_id={request_id} | "
                        f"continuing_conversation=true"
                    )
                    
                    await self._emit_billing_event("webhook_dns_failure", {
                        "request_id": request_id,
                        "minute": self.minutes_billed,
                        "retries_attempted": max_retries
                    })
                    
                    # IMPORTANT: Return a safe response that keeps conversation going
                    return {
                        "status": "billing_unavailable",
                        "creditsRemaining": 999,  # Assume credits available
                        "shouldTerminate": False,  # DO NOT TERMINATE
                        "message": "Billing service temporarily unavailable - continuing conversation"
                    }
                    
            except asyncio.TimeoutError:
                latency_ms = 5000  # Timeout was 5 seconds
                self.metrics.webhook_failures += 1
                
                billing_logger.error(
                    f"[WEBHOOK] Timeout | "
                    f"request_id={request_id} | "
                    f"timeout_ms={latency_ms} | "
                    f"minute={self.minutes_billed}"
                )
                
                if retry_attempt < max_retries - 1:
                    billing_logger.info(f"[WEBHOOK] Retrying after timeout | retry_in={retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    continue
                
                await self._emit_billing_event("webhook_timeout", {
                    "request_id": request_id,
                    "minute": self.minutes_billed
                })
                
                # Return safe default for timeout
                return {
                    "status": "timeout",
                    "creditsRemaining": 999,
                    "shouldTerminate": False,
                    "message": "Billing request timeout - continuing conversation"
                }
                
            except Exception as e:
                self.metrics.webhook_failures += 1
                
                billing_logger.error(
                    f"[WEBHOOK] Exception | "
                    f"request_id={request_id} | "
                    f"error={str(e)} | "
                    f"error_type={type(e).__name__}",
                    exc_info=True
                )
                
                await self._emit_billing_event("webhook_exception", {
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                # For unexpected errors, also return safe default
                return {
                    "status": "error",
                    "creditsRemaining": 999,
                    "shouldTerminate": False,
                    "message": f"Billing error: {type(e).__name__}"
                }
        
        # Should not reach here, but if it does, return safe default
        return {
            "status": "fallback",
            "creditsRemaining": 999,
            "shouldTerminate": False,
            "message": "Billing fallback - continuing conversation"
        }
    
    async def _close_websocket(self, reason: str):
        """Close the WebSocket connection directly"""
        try:
            if self._transport and hasattr(self._transport, '_websocket'):
                websocket = self._transport._websocket
                
                billing_logger.info(
                    f"[WEBSOCKET] Closing connection | "
                    f"conversation_id={self.conversation_id} | "
                    f"reason={reason}"
                )
                
                # Send termination message to client
                try:
                    await websocket.send_json({
                        "type": "session_terminated",
                        "reason": "insufficient_credits",
                        "message": reason or "Your session has ended due to insufficient credits",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    billing_logger.debug("[WEBSOCKET] Termination message sent to client")
                except Exception as e:
                    billing_logger.debug(f"[WEBSOCKET] Could not send termination message: {e}")
                
                # Close the WebSocket
                try:
                    await websocket.close(code=1000, reason="Billing terminated - insufficient credits")
                    billing_logger.info("[WEBSOCKET] Connection closed successfully")
                except Exception as e:
                    billing_logger.error(f"[WEBSOCKET] Error closing connection: {e}")
                    
            else:
                billing_logger.warning(
                    f"[WEBSOCKET] No transport available to close | "
                    f"has_transport={bool(self._transport)} | "
                    f"has_websocket={bool(self._transport and hasattr(self._transport, '_websocket'))}"
                )
        except Exception as e:
            billing_logger.error(
                f"[WEBSOCKET] Exception during close | "
                f"error={str(e)} | "
                f"error_type={type(e).__name__}",
                exc_info=True
            )
    
    async def _handle_billing_response(self, response: dict):
        """Handle billing response with detailed logging"""
        if not response:
            # No response means critical failure - but don't terminate
            billing_logger.warning(
                f"[RESPONSE] No response received | "
                f"conversation_id={self.conversation_id} | "
                f"continuing_without_billing=true"
            )
            return
            
        response_id = f"resp_{self.minutes_billed}_{int(time.time() * 1000)}"
        
        status = response.get("status")
        credits_remaining = response.get("creditsRemaining", 0)
        should_terminate = response.get("shouldTerminate", False)
        message = response.get("message")
        grace_period = response.get("gracePeriodSeconds")
        
        # Check if this is a fallback/error status
        if status in ["billing_unavailable", "timeout", "error", "fallback"]:
            billing_logger.warning(
                f"[RESPONSE] Billing unavailable | "
                f"response_id={response_id} | "
                f"status={status} | "
                f"message={message} | "
                f"continuing_conversation=true"
            )
            await self._emit_billing_event("billing_unavailable", {
                "response_id": response_id,
                "status": status,
                "message": message
            })
            return  # Don't process further, just continue conversation
        
        response_data = {
            "response_id": response_id,
            "minute": self.minutes_billed,
            "status": status,
            "credits_remaining": credits_remaining,
            "should_terminate": should_terminate,
            "message": message,
            "grace_period": grace_period
        }
        
        billing_logger.info(
            f"[RESPONSE] Processing | {json.dumps(response_data)}"
        )
        
        # Emit status event
        await self._emit_billing_event("status_update", response_data)
        
        # Handle different statuses
        if status == "warning":
            billing_logger.warning(
                f"[RESPONSE] Low credits warning | "
                f"response_id={response_id} | "
                f"credits_remaining={credits_remaining} | "
                f"message={message}"
            )
            
        elif should_terminate:
            billing_logger.warning(
                f"[TERMINATION] Requested | "
                f"response_id={response_id} | "
                f"conversation_id={self.conversation_id} | "
                f"reason={message} | "
                f"grace_period={grace_period}s"
            )
            
            self._transition_state(BillingState.STOPPING, "termination_requested")
            
            if grace_period:
                billing_logger.info(
                    f"[TERMINATION] Grace period starting | "
                    f"duration={grace_period}s"
                )
                
                await self._emit_billing_event("grace_period_started", {
                    "duration_seconds": grace_period,
                    "reason": message
                })
                
                await asyncio.sleep(grace_period)
                
                billing_logger.info(
                    f"[TERMINATION] Grace period ended"
                )
                
                await self._emit_billing_event("grace_period_ended", {})
            
            await self._emit_billing_event("terminated", {
                "conversation_id": self.conversation_id,
                "reason": "insufficient_credits",
                "minutes_used": self.minutes_billed,
                "final_message": message
            })
            
            await self._stop_billing("insufficient_credits")
            
            # CLOSE WEBSOCKET BEFORE SENDING END FRAME
            await self._close_websocket(message)
            
            # Send end frame to terminate conversation
            billing_logger.info(
                f"[TERMINATION] Sending EndFrame"
            )
            await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    
    async def _stop_billing(self, reason: str = "normal"):
        """Stop billing with detailed logging and metrics"""
        stop_id = f"stop_{int(time.time() * 1000)}"
        
        if not self._active:
            billing_logger.debug(
                f"[STOP] Already inactive | "
                f"stop_id={stop_id} | "
                f"state={self.state.value}"
            )
            return
        
        billing_logger.info(
            f"[STOP] Initiating | "
            f"stop_id={stop_id} | "
            f"conversation_id={self.conversation_id} | "
            f"reason={reason} | "
            f"minutes_billed={self.minutes_billed}"
        )
        
        self._transition_state(BillingState.STOPPING, f"stop_requested: {reason}")
        self._active = False
        
        # Cancel billing task
        if self._billing_task:
            task_name = self._billing_task.get_name()
            billing_logger.debug(
                f"[STOP] Cancelling task | "
                f"stop_id={stop_id} | "
                f"task_name={task_name}"
            )
            
            self._billing_task.cancel()
            try:
                await self._billing_task
            except asyncio.CancelledError:
                billing_logger.debug(
                    f"[STOP] Task cancelled | "
                    f"stop_id={stop_id} | "
                    f"task_name={task_name}"
                )
        
        # Calculate session duration
        if self.start_time:
            elapsed = datetime.utcnow() - self.start_time
            total_seconds = elapsed.total_seconds()
            
            session_data = {
                "stop_id": stop_id,
                "total_seconds": total_seconds,
                "billed_minutes": self.minutes_billed,
            }
            
            billing_logger.info(
                f"[STOP] Session duration | {json.dumps(session_data)}"
            )
        
        # Log final metrics
        final_metrics = self.metrics.to_dict()
        
        billing_logger.info(
            f"[STOP] Final metrics | "
            f"stop_id={stop_id} | "
            f"{json.dumps(final_metrics)}"
        )
        
        self._transition_state(BillingState.STOPPED, f"stop_completed: {reason}")
        
        await self._emit_billing_event("stopped", {
            "stop_id": stop_id,
            "conversation_id": self.conversation_id,
            "reason": reason,
            "total_minutes": self.minutes_billed,
            "metrics": final_metrics
        })
    
    async def cleanup(self):
        """Cleanup with comprehensive logging"""
        cleanup_id = f"cleanup_{int(time.time() * 1000)}"
        
        billing_logger.info(
            f"[CLEANUP] Starting | "
            f"cleanup_id={cleanup_id} | "
            f"conversation_id={self.conversation_id} | "
            f"state={self.state.value}"
        )
        
        await self._stop_billing("cleanup")
        await super().cleanup()
        
        billing_logger.info(
            f"[CLEANUP] Complete | "
            f"cleanup_id={cleanup_id} | "
            f"final_state={self.state.value}"
        )