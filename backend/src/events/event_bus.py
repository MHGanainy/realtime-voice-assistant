"""
Centralized event bus using PyEE with pattern matching support
"""
import asyncio
import re
from typing import Dict, List, Callable, Any, Optional, Set
from datetime import datetime
import logging
from pyee import EventEmitter

logger = logging.getLogger(__name__)


class EventBus:
    """Centralized event bus with pattern matching"""
    
    def __init__(self):
        self._ee = EventEmitter()
        self._patterns: Dict[str, List[Callable]] = {}
        self._exact_handlers: Dict[str, List[Callable]] = {}
        self._stats = {
            "total_events": 0,
            "events_by_type": {}
        }
        self._event_store = None
        
    async def emit(self, event_name: str, **data) -> None:
        """Emit an event with data"""
        self._stats["total_events"] += 1
        self._stats["events_by_type"][event_name] = self._stats["events_by_type"].get(event_name, 0) + 1
        
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()
            
        data["_event_name"] = event_name
        
        if self._event_store is None:
            try:
                from src.events.event_store import get_event_store
                self._event_store = get_event_store()
            except ImportError:
                pass
        
        if self._event_store:
            try:
                await self._event_store.store_event(event_name, data)
            except Exception as e:
                logger.error(f"Failed to store event {event_name}: {e}")
        
        tasks = []
        
        if event_name in self._exact_handlers:
            for handler in self._exact_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(data))
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in exact handler for {event_name}: {e}", exc_info=True)
        
        for pattern, handlers in self._patterns.items():
            if self._match_pattern(pattern, event_name):
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            tasks.append(handler(event_name, data))
                        else:
                            handler(event_name, data)
                    except Exception as e:
                        logger.error(f"Error in pattern handler for {pattern}: {e}", exc_info=True)
        
        self._ee.emit(event_name, data)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def on(self, event_name: str):
        """Decorator for exact event matching"""
        def decorator(func: Callable):
            if event_name not in self._exact_handlers:
                self._exact_handlers[event_name] = []
            self._exact_handlers[event_name].append(func)
            self._ee.on(event_name, func)
            return func
        return decorator
    
    def on_pattern(self, pattern: str):
        """Decorator for pattern-based event matching"""
        def decorator(func: Callable):
            if pattern not in self._patterns:
                self._patterns[pattern] = []
            self._patterns[pattern].append(func)
            return func
        return decorator
    
    def remove_listener(self, event_name: str, handler: Callable):
        """Remove a specific event handler"""
        if event_name in self._exact_handlers:
            if handler in self._exact_handlers[event_name]:
                self._exact_handlers[event_name].remove(handler)
                
        self._ee.remove_listener(event_name, handler)
    
    def remove_pattern_listener(self, pattern: str, handler: Callable):
        """Remove a pattern-based handler"""
        if pattern in self._patterns:
            if handler in self._patterns[pattern]:
                self._patterns[pattern].remove(handler)
    
    def _match_pattern(self, pattern: str, event_name: str) -> bool:
        """Check if an event name matches a pattern"""
        regex_pattern = pattern.replace("**", ".*")
        regex_pattern = regex_pattern.replace("*", "[^:]+")
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, event_name))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self._stats,
            "pattern_handlers": len(self._patterns),
            "exact_handlers": len(self._exact_handlers)
        }
    
    async def wait_for(self, event_name: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for a specific event to occur"""
        future = asyncio.Future()
        
        def handler(data):
            if not future.done():
                future.set_result(data)
        
        self.on(event_name)(handler)
        
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.remove_listener(event_name, handler)


_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the singleton event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus