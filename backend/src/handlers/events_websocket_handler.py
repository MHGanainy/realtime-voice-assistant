"""
WebSocket handler for events-only connections.
Provides real-time event streaming with session-based security.
"""
import asyncio
import json
from typing import Dict, Optional, List, Set, Any
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect, Query
import logging
import uuid
from pydantic import BaseModel, Field

from src.events import (
    EventBus, 
    EventStore, 
    EventSecurity,
    get_event_bus, 
    get_event_store,
    get_event_security
)
from src.services.conversation_manager import get_conversation_manager

logger = logging.getLogger(__name__)


class EventSubscription(BaseModel):
    """Client subscription preferences"""
    patterns: List[str] = Field(default=["*"])
    conversation_ids: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    include_historical: bool = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "patterns": [
                    "conversation:*:transcription:*",
                    "conversation:*:turn:*",
                    "global:system:*"
                ],
                "conversation_ids": ["abc123"],
                "exclude_patterns": ["*:*:metrics:*"],
                "include_historical": True
            }
        }


class EventClient:
    """Represents a connected event WebSocket client"""
    
    def __init__(
        self,
        client_id: str,
        websocket: WebSocket,
        session_id: str,
        subscription: EventSubscription
    ):
        self.id = client_id
        self.websocket = websocket
        self.session_id = session_id
        self.subscription = subscription
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.event_count = 0
        
    def matches_subscription(self, event_name: str) -> bool:
        """Check if event matches client subscription patterns"""
        if self.subscription.exclude_patterns:
            for pattern in self.subscription.exclude_patterns:
                if self._match_pattern(pattern, event_name):
                    return False
        
        for pattern in self.subscription.patterns:
            if self._match_pattern(pattern, event_name):
                return True
        
        if self.subscription.conversation_ids and event_name.startswith("conversation:"):
            parts = event_name.split(":")
            if len(parts) >= 2:
                conv_id = parts[1]
                return conv_id in self.subscription.conversation_ids
        
        return False
    
    def _match_pattern(self, pattern: str, event_name: str) -> bool:
        """Simple pattern matching with * wildcard"""
        import re
        
        if pattern == "*":
            return True
            
        regex_pattern = pattern.replace("**", ".*")
        regex_pattern = regex_pattern.replace("*", "[^:]+")
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, event_name))
    
    async def send_event(self, event_name: str, event_data: Dict[str, Any]):
        """Send event to client with JSON serialization handling"""
        try:
            serializable_data = self._make_json_serializable(event_data)
            
            message = {
                "type": "event",
                "event": event_name,
                "data": serializable_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.websocket.send_json(message)
            self.event_count += 1
        except Exception as e:
            logger.error(f"Failed to send event to client {self.id}: {e}")
            raise
    
    def _make_json_serializable(self, obj) -> Any:
        """Convert objects to JSON-serializable format"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            return str(obj)
            
    async def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send a non-event message to client"""
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to client {self.id}: {e}")
            raise


class EventsWebSocketHandler:
    """Handles WebSocket connections for event streaming"""
    
    def __init__(self):
        self._event_bus = get_event_bus()
        self._event_store = get_event_store()
        self._conversation_manager = get_conversation_manager()
        self._event_security = get_event_security(self._conversation_manager)
        
        self._clients: Dict[str, EventClient] = {}
        self._session_clients: Dict[str, Set[str]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self._setup_event_handlers()
        self._cleanup_task = asyncio.create_task(self._cleanup_disconnected_clients())
        
    def _setup_event_handlers(self):
        """Setup event bus handlers for all events using direct registration"""
        
        async def handle_all_events(event_name: str, event_data: Dict[str, Any]):
            """Forward events to subscribed clients"""
            disconnected_clients = []
            
            if not self._clients:
                return
            
            for client_id, client in self._clients.items():
                try:
                    access_granted = self._event_security.can_access_event(
                        client.session_id, 
                        event_name, 
                        event_data
                    )
                    
                    if access_granted and client.matches_subscription(event_name):
                        await client.send_event(event_name, event_data)
                        
                except Exception as e:
                    logger.error(f"Exception sending event {event_name} to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            for client_id in disconnected_clients:
                await self._remove_client(client_id)
                
        if "*" not in self._event_bus._patterns:
            self._event_bus._patterns["*"] = []
        self._event_bus._patterns["*"].append(handle_all_events)
        
        conversation_patterns = [
            "conversation:*:audio:*",
            "conversation:*:transcription:*", 
            "conversation:*:turn:*",
            "conversation:*:text:*",
            "conversation:*:metrics:*"
        ]
        
        for pattern in conversation_patterns:
            if pattern not in self._event_bus._patterns:
                self._event_bus._patterns[pattern] = []
            self._event_bus._patterns[pattern].append(handle_all_events)
        
        def sync_handler(event_data):
            event_name = event_data.get("_event_name", "unknown")
            asyncio.create_task(handle_all_events(event_name, event_data))
        
        self._event_bus._ee.on("conversation:*", sync_handler)
        self._event_bus._ee.on("global:*", sync_handler)
        
    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: str = Query(..., description="Required session ID for access control")
    ):
        """Handle a new events WebSocket connection"""
        client_id = str(uuid.uuid4())
        
        try:
            if not session_id:
                await websocket.close(code=4001, reason="session_id required")
                return
            
            await websocket.accept()
            logger.info(f"Events WebSocket connection established: {client_id} (session: {session_id})")
            
            subscription = EventSubscription(
                patterns=[
                    f"conversation:*:*:*",
                    "global:*:*"
                ]
            )
            
            client = EventClient(
                client_id=client_id,
                websocket=websocket,
                session_id=session_id,
                subscription=subscription
            )
            
            self._clients[client_id] = client
            if session_id not in self._session_clients:
                self._session_clients[session_id] = set()
            self._session_clients[session_id].add(client_id)
            
            await client.send_message("welcome", {
                "client_id": client_id,
                "session_id": session_id,
                "subscription": subscription.dict(),
                "server_time": datetime.utcnow().isoformat()
            })
            
            if subscription.include_historical:
                await self._send_historical_events(client)
            
            await self._event_bus.emit(
                f"connection:{client_id}:events:established",
                connection_id=client_id,
                session_id=session_id,
                client_ip=websocket.client.host if websocket.client else None
            )
            
            await self._handle_client_messages(client)
            
        except WebSocketDisconnect:
            logger.info(f"Events WebSocket disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error in events WebSocket handler: {e}", exc_info=True)
        finally:
            await self._remove_client(client_id)
    
    async def _handle_client_messages(self, client: EventClient):
        """Handle incoming messages from client"""
        try:
            while True:
                data = await client.websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "ping":
                    client.last_heartbeat = datetime.utcnow()
                    await client.send_message("pong", {
                        "client_time": data.get("timestamp"),
                        "server_time": datetime.utcnow().isoformat()
                    })
                    
                elif message_type == "subscribe":
                    new_sub = EventSubscription(**data.get("subscription", {}))
                    client.subscription = new_sub
                    
                    await client.send_message("subscription_updated", {
                        "subscription": new_sub.dict()
                    })
                    
                    if new_sub.include_historical:
                        await self._send_historical_events(client)
                        
                elif message_type == "get_stats":
                    await client.send_message("stats", {
                        "event_count": client.event_count,
                        "connected_duration_seconds": (
                            datetime.utcnow() - client.connected_at
                        ).total_seconds(),
                        "last_heartbeat": client.last_heartbeat.isoformat()
                    })
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error handling client messages: {e}")
    
    async def _send_historical_events(self, client: EventClient):
        """Send historical events to newly connected client"""
        try:
            since = datetime.utcnow() - timedelta(hours=1)
            
            events = await self._event_store.get_session_events(
                client.session_id,
                since=since,
                limit=100
            )
            
            await client.send_message("historical_start", {
                "count": len(events),
                "since": since.isoformat()
            })
            
            for event in events:
                event_name = event["name"]
                event_data = event["data"]
                
                if self._event_security.can_access_event(
                    client.session_id,
                    event_name,
                    event_data
                ):
                    if client.matches_subscription(event_name):
                        await client.send_event(event_name, event_data)
            
            await client.send_message("historical_end", {
                "count": len(events)
            })
            
        except Exception as e:
            logger.error(f"Error sending historical events: {e}")
    
    async def _remove_client(self, client_id: str):
        """Remove a client and clean up"""
        if client_id not in self._clients:
            return
        
        client = self._clients[client_id]
        
        if client.session_id in self._session_clients:
            self._session_clients[client.session_id].discard(client_id)
            if not self._session_clients[client.session_id]:
                del self._session_clients[client.session_id]
        
        del self._clients[client_id]
        
        await self._event_bus.emit(
            f"connection:{client_id}:events:closed",
            connection_id=client_id,
            session_id=client.session_id,
            event_count=client.event_count,
            duration_seconds=(datetime.utcnow() - client.connected_at).total_seconds()
        )
    
    async def _cleanup_disconnected_clients(self):
        """Periodically clean up disconnected clients"""
        while True:
            try:
                await asyncio.sleep(30)
                
                disconnected = []
                for client_id, client in self._clients.items():
                    try:
                        await client.websocket.send_json({"type": "heartbeat"})
                    except:
                        disconnected.append(client_id)
                
                for client_id in disconnected:
                    await self._remove_client(client_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            "total_clients": len(self._clients),
            "sessions_connected": len(self._session_clients),
            "clients_by_session": {
                session_id: len(client_ids)
                for session_id, client_ids in self._session_clients.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown handler and disconnect all clients"""
        logger.info("Shutting down events WebSocket handler")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        client_ids = list(self._clients.keys())
        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client:
                try:
                    await client.websocket.close(code=1001, reason="Server shutdown")
                except:
                    pass
            await self._remove_client(client_id)


_events_handler: Optional[EventsWebSocketHandler] = None


def get_events_websocket_handler() -> EventsWebSocketHandler:
    """Get the singleton events WebSocket handler"""
    global _events_handler
    if _events_handler is None:
        _events_handler = EventsWebSocketHandler()
    return _events_handler