import logging
import time
from typing import AsyncIterator, Callable, Dict, Any, List, Optional
from datetime import datetime
from ..core.pipeline.middleware import Middleware
from ..core.events.types import Event, EventType

logger = logging.getLogger(__name__)

class LoggingMiddleware(Middleware):
    """
    Middleware for comprehensive logging of events and performance metrics.
    """
    
    def __init__(
        self, 
        log_level: str = "INFO",
        log_performance: bool = True,
        log_event_data: bool = False,
        excluded_events: Optional[set] = None
    ):
        self.log_level = getattr(logging, log_level.upper())
        self.log_performance = log_performance
        self.log_event_data = log_event_data
        self.excluded_events = excluded_events or {EventType.STT_AUDIO_RECEIVED}
        
        # Performance tracking
        self.event_timings: Dict[str, List[float]] = {}
        
    async def process(
        self, 
        event: Event, 
        next_handler: Callable[[Event], AsyncIterator[Event]]
    ) -> AsyncIterator[Event]:
        """Log events as they pass through the pipeline"""
        
        # Skip excluded events
        if event.type in self.excluded_events:
            async for result in next_handler(event):
                yield result
            return
        
        # Log incoming event
        start_time = time.time()
        correlation_id = event.correlation_id or "no-correlation-id"
        
        log_message = f"[{correlation_id}] Processing {event.type} from {event.source}"
        if self.log_event_data:
            log_message += f" - Data: {event.data}"
            
        logger.log(self.log_level, log_message)
        
        # Track event processing
        processed_count = 0
        errors = []
        
        try:
            # Process event through next handler
            async for result in next_handler(event):
                processed_count += 1
                
                # Log outgoing event
                result_message = f"[{correlation_id}] Produced {result.type}"
                if self.log_event_data:
                    result_message += f" - Data: {result.data}"
                    
                logger.log(self.log_level, result_message)
                
                yield result
                
        except Exception as e:
            errors.append(str(e))
            logger.error(
                f"[{correlation_id}] Error processing {event.type}: {e}",
                exc_info=True
            )
            raise
            
        finally:
            # Log performance metrics
            if self.log_performance:
                elapsed = time.time() - start_time
                self._log_performance_metrics(
                    event, 
                    elapsed, 
                    processed_count,
                    errors,
                    correlation_id
                )
                
    def _log_performance_metrics(
        self, 
        event: Event,
        elapsed: float,
        processed_count: int,
        errors: list,
        correlation_id: str
    ):
        """Log performance metrics for event processing"""
        
        metrics = {
            "event_type": str(event.type),
            "source": event.source,
            "correlation_id": correlation_id,
            "processing_time_ms": round(elapsed * 1000, 2),
            "events_produced": processed_count,
            "errors": len(errors),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to timing history
        event_type_str = str(event.type)
        if event_type_str not in self.event_timings:
            self.event_timings[event_type_str] = []
        self.event_timings[event_type_str].append(elapsed)
        
        # Calculate average timing
        avg_time = sum(self.event_timings[event_type_str]) / len(self.event_timings[event_type_str])
        metrics["avg_processing_time_ms"] = round(avg_time * 1000, 2)
        
        logger.info(f"Performance metrics: {metrics}")
        
        # Emit metrics event if event bus is available
        try:
            from ..core.events.bus import event_bus
            from ..core.events.types import MetricsEvent
            
            metrics_event = MetricsEvent(
                source="logging_middleware",
                metrics={
                    "processing_time": elapsed,
                    "events_produced": processed_count,
                    "event_type": str(event.type)
                },
                data=metrics
            )
            
            # Fire and forget
            import asyncio
            asyncio.create_task(event_bus.publish(metrics_event))
            
        except ImportError:
            pass  # Event bus not available
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulated statistics"""
        stats = {}
        
        for event_type, timings in self.event_timings.items():
            if timings:  # Only process if we have timings
                stats[event_type] = {
                    "count": len(timings),
                    "avg_ms": round(sum(timings) / len(timings) * 1000, 2),
                    "min_ms": round(min(timings) * 1000, 2),
                    "max_ms": round(max(timings) * 1000, 2)
                }
            
        return stats