import asyncio
import logging
from typing import Dict, List, Callable, Optional, Any
from weakref import WeakSet
from .types import Event, EventType

logger = logging.getLogger(__name__)

class EventBus:
    """
    Asynchronous event bus for pub/sub pattern.
    Supports both sync and async subscribers.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, WeakSet[Callable]] = {}
        self._wildcard_subscribers: WeakSet[Callable] = WeakSet()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the event bus worker"""
        if self._running:
            return
            
        self._running = True
        self._worker_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        
    async def stop(self):
        """Stop the event bus worker"""
        self._running = False
        
        # Put sentinel to wake up worker
        await self._event_queue.put(None)
        
        if self._worker_task:
            await self._worker_task
            
        logger.info("Event bus stopped")
        
    def subscribe(self, event_type: Optional[EventType], handler: Callable):
        """
        Subscribe to events of a specific type or all events (if event_type is None)
        """
        if event_type is None:
            self._wildcard_subscribers.add(handler)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = WeakSet()
            self._subscribers[event_type].add(handler)
            
        logger.debug(f"Handler {handler.__name__} subscribed to {event_type or 'all events'}")
        
    def unsubscribe(self, event_type: Optional[EventType], handler: Callable):
        """Unsubscribe from events"""
        if event_type is None:
            self._wildcard_subscribers.discard(handler)
        elif event_type in self._subscribers:
            self._subscribers[event_type].discard(handler)
            
    async def publish(self, event: Event):
        """Publish an event to the bus"""
        if not self._running:
            logger.warning("Event bus not running, event dropped")
            return
            
        await self._event_queue.put(event)
        
    async def _process_events(self):
        """Worker task to process events"""
        while self._running:
            try:
                event = await self._event_queue.get()
                
                if event is None:  # Sentinel value
                    break
                    
                # Get all relevant handlers
                handlers = set()
                
                # Add specific handlers
                if event.type in self._subscribers:
                    handlers.update(self._subscribers[event.type])
                    
                # Add wildcard handlers
                handlers.update(self._wildcard_subscribers)
                
                # Call all handlers
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            # Run sync handlers in executor
                            await asyncio.get_event_loop().run_in_executor(
                                None, handler, event
                            )
                    except Exception as e:
                        logger.error(
                            f"Error in event handler {handler.__name__}: {e}",
                            exc_info=True
                        )
                        
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
                
    def emit_sync(self, event: Event):
        """Synchronously emit an event (creates task if in async context)"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            # No event loop, queue for later
            logger.warning("No event loop running, event queued for later")

# Global event bus instance
event_bus = EventBus()

# Decorator for easy event subscription
def on_event(event_type: Optional[EventType] = None):
    """Decorator to subscribe a function to events"""
    def decorator(func):
        event_bus.subscribe(event_type, func)
        return func
    return decorator