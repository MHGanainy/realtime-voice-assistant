"""
Connection monitoring service for detecting stale/dead WebSocket connections
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging
from fastapi import WebSocket
from src.services.logfire_service import get_logfire

logger = logging.getLogger(__name__)

class ConnectionInfo:
    """Information about a monitored connection"""
    def __init__(self, connection_id: str, websocket: WebSocket, **metadata):
        self.connection_id = connection_id
        self.websocket = websocket
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.last_ping = None
        self.ping_failures = 0
        self.total_pings = 0
        self.metadata = metadata
    
    @property
    def age_seconds(self) -> float:
        """Connection age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def inactive_seconds(self) -> float:
        """Seconds since last activity"""
        return (datetime.utcnow() - self.last_activity).total_seconds()

class ConnectionMonitor:
    """Monitor WebSocket connections for health and detect disconnections"""
    
    def __init__(self, 
                 check_interval: int = 10,
                 stale_threshold: int = 30,
                 max_ping_failures: int = 3):
        """
        Initialize connection monitor
        
        Args:
            check_interval: Seconds between health checks
            stale_threshold: Seconds of inactivity before connection is considered stale
            max_ping_failures: Number of ping failures before considering connection dead
        """
        self.logfire = get_logfire()
        self._connections: Dict[str, ConnectionInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self.check_interval = check_interval
        self.stale_threshold = stale_threshold
        self.max_ping_failures = max_ping_failures
        self._running = False
    
    async def start(self):
        """Start the connection monitor"""
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"Connection monitor started (interval={self.check_interval}s)")
            self.logfire.log_connection_event(
                connection_id="monitor",
                event="started",
                check_interval=self.check_interval,
                stale_threshold=self.stale_threshold
            )
    
    async def stop(self):
        """Stop the connection monitor"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Connection monitor stopped")
            self.logfire.log_connection_event(
                connection_id="monitor",
                event="stopped"
            )
    
    def add_connection(self, connection_id: str, websocket: WebSocket, **metadata):
        """Add a connection to monitor"""
        self._connections[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            **metadata
        )
        
        self.logfire.log_websocket_state(
            connection_id=connection_id,
            state="added_to_monitor",
            total_connections=len(self._connections),
            metadata=metadata
        )
        
        logger.info(f"Added connection {connection_id} to monitor (total: {len(self._connections)})")
    
    def remove_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Remove a connection from monitoring"""
        if connection_id in self._connections:
            conn_info = self._connections[connection_id]
            
            self.logfire.log_websocket_state(
                connection_id=connection_id,
                state="removed_from_monitor",
                duration_seconds=conn_info.age_seconds,
                ping_failures=conn_info.ping_failures,
                total_pings=conn_info.total_pings,
                final_inactive_seconds=conn_info.inactive_seconds
            )
            
            del self._connections[connection_id]
            logger.info(f"Removed connection {connection_id} from monitor (remaining: {len(self._connections)})")
            return conn_info
        return None
    
    def update_activity(self, connection_id: str):
        """Update last activity timestamp for a connection"""
        if connection_id in self._connections:
            self._connections[connection_id].last_activity = datetime.utcnow()
    
    async def _check_connection_health(self, conn_info: ConnectionInfo) -> bool:
        """
        Check if a connection is healthy by sending a ping
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Try to send ping
            await asyncio.wait_for(
                conn_info.websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()}),
                timeout=5.0
            )
            
            conn_info.last_ping = datetime.utcnow()
            conn_info.total_pings += 1
            
            # Reset failures on successful ping
            if conn_info.ping_failures > 0:
                self.logfire.log_websocket_state(
                    conn_info.connection_id,
                    state="connection_recovered",
                    previous_failures=conn_info.ping_failures
                )
                conn_info.ping_failures = 0
            
            return True
            
        except asyncio.TimeoutError:
            conn_info.ping_failures += 1
            self.logfire.log_disconnection(
                conn_info.connection_id,
                reason="ping_timeout",
                ping_failures=conn_info.ping_failures,
                inactive_seconds=conn_info.inactive_seconds
            )
            return False
            
        except Exception as e:
            conn_info.ping_failures += 1
            self.logfire.log_disconnection(
                conn_info.connection_id,
                reason="ping_error",
                error=e,
                ping_failures=conn_info.ping_failures,
                inactive_seconds=conn_info.inactive_seconds
            )
            return False
    
    # In src/services/connection_monitor.py, modify the _monitor_loop method:

    async def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Connection monitor loop started")
        
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                if not self._connections:
                    continue
                
                # Check all connections
                dead_connections = []
                stale_connections = []
                
                for conn_id, conn_info in list(self._connections.items()):
                    try:
                        # Check if connection is stale
                        if conn_info.inactive_seconds > self.stale_threshold:
                            stale_connections.append(conn_id)
                            
                            self.logfire.log_websocket_state(
                                conn_id,
                                state="stale_detected",
                                inactive_seconds=conn_info.inactive_seconds,
                                age_seconds=conn_info.age_seconds
                            )
                            
                            # Try to ping the connection
                            is_healthy = await self._check_connection_health(conn_info)
                            
                            if not is_healthy:
                                # Check if we've exceeded max failures
                                if conn_info.ping_failures >= self.max_ping_failures:
                                    dead_connections.append(conn_id)
                                    
                                    self.logfire.log_disconnection(
                                        conn_id,
                                        reason="connection_dead",
                                        ping_failures=conn_info.ping_failures,
                                        inactive_seconds=conn_info.inactive_seconds,
                                        age_seconds=conn_info.age_seconds
                                    )
                        else:
                            # Log heartbeat for active connections
                            self.logfire.log_heartbeat(
                                conn_id,
                                inactive_seconds=conn_info.inactive_seconds,
                                age_seconds=conn_info.age_seconds,
                                ping_failures=conn_info.ping_failures
                            )
                    
                    except Exception as e:
                        logger.error(f"Error monitoring connection {conn_id}: {e}")
                        self.logfire.log_error(
                            conn_id,
                            error=e,
                            context="monitor_check"
                        )
                
                # CRITICAL: Actually close dead connections, not just remove them
                for conn_id in dead_connections:
                    logger.warning(f"Dead connection detected: {conn_id}")
                    conn_info = self._connections.get(conn_id)
                    
                    if conn_info and conn_info.metadata.get("correlation_token"):
                        correlation_token = conn_info.metadata["correlation_token"]
                        logger.warning(f"Triggering force close for dead connection with correlation: {correlation_token}")
                        
                        # Import and call the websocket handler to properly close the connection
                        try:
                            from src.handlers.websocket_handler import get_websocket_handler
                            ws_handler = get_websocket_handler()
                            closed = await ws_handler.close_connection_by_correlation(correlation_token)
                            logger.info(f"Force closed {closed} connection(s) for dead correlation token: {correlation_token}")
                        except Exception as e:
                            logger.error(f"Failed to force close dead connection: {e}")
                    
                    # Remove from monitoring after closing
                    self.remove_connection(conn_id)
                
                # Log monitoring stats
                if len(self._connections) > 0:
                    self.logfire.log_metrics(
                        connection_id="monitor",
                        metric_type="health_check",
                        total_connections=len(self._connections),
                        stale_connections=len(stale_connections),
                        dead_connections=len(dead_connections)
                    )
                    
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.logfire.log_error(
                    connection_id="monitor",
                    error=e,
                    context="monitor_loop"
                )
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a specific connection"""
        return self._connections.get(connection_id)
    
    def get_all_connections(self) -> Dict[str, ConnectionInfo]:
        """Get all monitored connections"""
        return self._connections.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        total_ping_failures = sum(c.ping_failures for c in self._connections.values())
        avg_inactive = (
            sum(c.inactive_seconds for c in self._connections.values()) / len(self._connections)
            if self._connections else 0
        )
        
        return {
            "total_connections": len(self._connections),
            "total_ping_failures": total_ping_failures,
            "average_inactive_seconds": avg_inactive,
            "monitor_running": self._running,
            "check_interval": self.check_interval,
            "stale_threshold": self.stale_threshold
        }

# Singleton instance
_monitor: Optional[ConnectionMonitor] = None

def get_connection_monitor() -> ConnectionMonitor:
    """Get the singleton connection monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ConnectionMonitor()
    return _monitor