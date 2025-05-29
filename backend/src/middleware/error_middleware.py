import logging
import traceback
import asyncio
import time  # Added missing import
from typing import AsyncIterator, Callable, Optional, Type, Tuple, Dict, Any, List
from datetime import datetime
from ..core.pipeline.middleware import Middleware
from ..core.events.types import Event, EventType, ErrorEvent

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(Middleware):
    """
    Middleware for handling errors with retry logic and circuit breaker pattern.
    """
    
    def __init__(
        self, 
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        max_retry_delay: float = 30.0,
        propagate_errors: bool = False,
        error_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (
            ConnectionError,
            asyncio.TimeoutError,
        )
    ):
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.max_retry_delay = max_retry_delay
        self.propagate_errors = propagate_errors
        self.retryable_exceptions = retryable_exceptions
        
        # Circuit breaker settings
        self.error_threshold = error_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Circuit breaker state
        self.error_counts: Dict[str, int] = {}
        self.circuit_open: Dict[str, float] = {}
        
    async def process(
        self, 
        event: Event, 
        next_handler: Callable[[Event], AsyncIterator[Event]]
    ) -> AsyncIterator[Event]:
        """Handle errors with retry logic and circuit breaker"""
        
        event_key = f"{event.type}:{event.source}"
        
        # Check circuit breaker
        if self._is_circuit_open(event_key):
            error_event = ErrorEvent(
                source="error_middleware",
                error_message="Circuit breaker is open",
                error_type="CircuitBreakerOpen",
                data={
                    "original_event": event.dict(),
                    "event_key": event_key
                }
            )
            yield error_event
            return
            
        attempt = 0
        last_exception = None
        delay = self.retry_delay
        
        while attempt < self.retry_attempts:
            try:
                # Process event
                event_processed = False
                async for result in next_handler(event):
                    event_processed = True
                    yield result
                    
                # Success - reset error count
                if event_processed:
                    self._reset_error_count(event_key)
                    
                    if attempt > 0:
                        logger.info(
                            f"Event {event.type} succeeded after {attempt + 1} attempts"
                        )
                        
                return  # Success, exit
                
            except self.retryable_exceptions as e:
                last_exception = e
                attempt += 1
                
                logger.warning(
                    f"Retryable error processing {event.type} "
                    f"(attempt {attempt}/{self.retry_attempts}): {e}"
                )
                
                if attempt < self.retry_attempts:
                    # Calculate next delay with backoff
                    delay = min(delay * self.retry_backoff, self.max_retry_delay)
                    
                    logger.info(
                        f"Retrying {event.type} in {delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                # Non-retryable error
                last_exception = e
                logger.error(
                    f"Non-retryable error processing {event.type}: {e}",
                    exc_info=True
                )
                break
                
        # All attempts failed or non-retryable error
        self._increment_error_count(event_key)
        
        # Create error event
        error_event = ErrorEvent(
            source="error_middleware",
            error_message=str(last_exception),
            error_type=type(last_exception).__name__,
            traceback=traceback.format_exc(),
            retry_attempt=attempt,
            data={
                "original_event": event.dict(),
                "attempts": attempt,
                "event_key": event_key,
                "circuit_breaker_triggered": self._is_circuit_open(event_key)
            }
        )
        
        yield error_event
        
        if self.propagate_errors and last_exception:
            raise last_exception
            
    def _is_circuit_open(self, event_key: str) -> bool:
        """Check if circuit breaker is open for this event type"""
        if event_key not in self.circuit_open:
            return False
            
        # Check if timeout has passed
        if time.time() - self.circuit_open[event_key] > self.circuit_breaker_timeout:
            # Reset circuit
            del self.circuit_open[event_key]
            self.error_counts[event_key] = 0
            logger.info(f"Circuit breaker reset for {event_key}")
            return False
            
        return True
        
    def _increment_error_count(self, event_key: str):
        """Increment error count and potentially open circuit"""
        if event_key not in self.error_counts:
            self.error_counts[event_key] = 0
            
        self.error_counts[event_key] += 1
        
        # Check if we should open the circuit
        if self.error_counts[event_key] >= self.error_threshold:
            self.circuit_open[event_key] = time.time()
            logger.warning(
                f"Circuit breaker opened for {event_key} after "
                f"{self.error_counts[event_key]} errors"
            )
            
    def _reset_error_count(self, event_key: str):
        """Reset error count on success"""
        if event_key in self.error_counts:
            self.error_counts[event_key] = 0
            
    def get_circuit_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        status = {
            "error_counts": self.error_counts.copy(),
            "open_circuits": []
        }
        
        for event_key, open_time in self.circuit_open.items():
            remaining = self.circuit_breaker_timeout - (time.time() - open_time)
            status["open_circuits"].append({
                "event_key": event_key,
                "remaining_timeout": max(0, remaining)
            })
            
        return status