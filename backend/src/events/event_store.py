"""
In-memory event storage with TTL and size limits
"""
import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import uuid
import logging

logger = logging.getLogger(__name__)


class EventStore:
    """In-memory event storage with automatic cleanup"""
    
    def __init__(
        self,
        max_events_per_conversation: int = 1000,
        max_global_events: int = 5000,
        ttl_hours: int = 24,
        cleanup_interval_minutes: int = 30
    ):
        self._conversation_events: Dict[str, deque] = {}
        self._global_events: deque = deque(maxlen=max_global_events)
        self._session_events: Dict[str, List[str]] = {}
        
        self._max_events_per_conversation = max_events_per_conversation
        self._ttl_hours = ttl_hours
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self._cleanup_task = asyncio.create_task(
            self._periodic_cleanup(cleanup_interval_minutes)
        )
    
    async def store_event(
        self,
        event_name: str,
        event_data: Dict[str, Any]
    ) -> str:
        """Store an event and return its ID"""
        event_id = str(uuid.uuid4())
        
        event_record = {
            "id": event_id,
            "name": event_name,
            "data": event_data,
            "timestamp": datetime.utcnow()
        }
        
        if event_name.startswith("conversation:"):
            parts = event_name.split(":")
            if len(parts) >= 2:
                conv_id = parts[1]
                
                if conv_id not in self._conversation_events:
                    self._conversation_events[conv_id] = deque(
                        maxlen=self._max_events_per_conversation
                    )
                
                self._conversation_events[conv_id].append(event_record)
                
                session_id = event_data.get("session_id")
                if session_id:
                    if session_id not in self._session_events:
                        self._session_events[session_id] = []
                    if conv_id not in self._session_events[session_id]:
                        self._session_events[session_id].append(conv_id)
        
        elif event_name.startswith("global:"):
            self._global_events.append(event_record)
        
        return event_id
    
    async def get_conversation_events(
        self,
        conversation_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get events for a specific conversation"""
        if conversation_id not in self._conversation_events:
            return []
        
        events = list(self._conversation_events[conversation_id])
        
        if since:
            events = [e for e in events if e["timestamp"] > since]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    async def get_session_events(
        self,
        session_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all events for a session (across all conversations)"""
        events = []
        
        conv_ids = self._session_events.get(session_id, [])
        
        for conv_id in conv_ids:
            conv_events = await self.get_conversation_events(conv_id, since)
            events.extend(conv_events)
        
        global_events = list(self._global_events)
        if since:
            global_events = [e for e in global_events if e["timestamp"] > since]
        events.extend(global_events)
        
        events.sort(key=lambda e: e["timestamp"])
        
        if limit:
            events = events[-limit:]
        
        return events
    
    async def get_global_events(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get global events"""
        events = list(self._global_events)
        
        if since:
            events = [e for e in events if e["timestamp"] > since]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    async def clear_conversation_events(self, conversation_id: str):
        """Clear events for a specific conversation"""
        if conversation_id in self._conversation_events:
            del self._conversation_events[conversation_id]
    
    async def _periodic_cleanup(self, interval_minutes: int):
        """Periodically clean up old events"""
        while True:
            try:
                await asyncio.sleep(interval_minutes * 60)
                await self._cleanup_old_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event cleanup: {e}", exc_info=True)
    
    async def _cleanup_old_events(self):
        """Remove events older than TTL"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self._ttl_hours)
        cleaned_count = 0
        
        for conv_id, events in list(self._conversation_events.items()):
            original_len = len(events)
            while events and events[0]["timestamp"] < cutoff_time:
                events.popleft()
                cleaned_count += 1
            
            if not events:
                del self._conversation_events[conv_id]
        
        while self._global_events and self._global_events[0]["timestamp"] < cutoff_time:
            self._global_events.popleft()
            cleaned_count += 1
        
        for session_id, conv_ids in list(self._session_events.items()):
            self._session_events[session_id] = [
                cid for cid in conv_ids if cid in self._conversation_events
            ]
            if not self._session_events[session_id]:
                del self._session_events[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_events = sum(len(events) for events in self._conversation_events.values())
        total_events += len(self._global_events)
        
        return {
            "total_events": total_events,
            "conversation_count": len(self._conversation_events),
            "global_events_count": len(self._global_events),
            "session_count": len(self._session_events),
            "events_by_conversation": {
                conv_id: len(events)
                for conv_id, events in self._conversation_events.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown the event store"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


_event_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """Get the singleton event store instance"""
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store