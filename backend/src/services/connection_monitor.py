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
        # Add flag to track if connection is being force-closed
        self.force_closing = False
        self.websocket_closed = False
    
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
                 max_ping_failures: int = 3,
                 auto_close_dead_connections: bool = False):  # DISABLED by default
        """
        Initialize connection monitor
        
        Args:
            check_interval: Seconds between health checks
            stale_threshold: Seconds of inactivity before connection is considered stale
            max_ping_failures: Number of ping failures before considering connection dead
            auto_close_dead_connections: Whether to automatically close dead connections (DISABLED)
        """
        self.logfire = get_logfire()
        self._connections: Dict[str, ConnectionInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self.check_interval = check_interval
        self.stale_threshold = stale_threshold
        self.max_ping_failures = max_ping_failures
        self.auto_close_dead_connections = auto_close_dead_connections  # Store setting
        self._running = False
        self._lock = asyncio.Lock()
    
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
    
    async def _monitor_loop(self):
        """Main monitoring loop - ONLY LOGS, DOES NOT CLOSE CONNECTIONS"""
        logger.info(f"Connection monitor loop started (auto-close: {self.auto_close_dead_connections})")
        
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                if not self._connections:
                    continue
                
                # Get list of connections to check
                async with self._lock:
                    connections_to_check = list(self._connections.keys())
                
                for conn_id in connections_to_check:
                    try:
                        # Check connection with timeout
                        await asyncio.wait_for(
                            self._check_single_connection(conn_id),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout checking connection {conn_id}")
                    except Exception as e:
                        logger.error(f"Error checking connection {conn_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    async def _check_single_connection(self, conn_id: str):
        """Check a single connection's health - ONLY LOGS, DOES NOT CLOSE"""
        async with self._lock:
            conn_info = self._connections.get(conn_id)
            if not conn_info:
                return
            
            # Skip if being force-closed or WebSocket is closed
            if conn_info.force_closing or conn_info.websocket_closed:
                return
            
            # Only check stale connections
            if conn_info.inactive_seconds <= self.stale_threshold:
                return
        
        # Check health outside lock
        is_healthy = await self._check_connection_health(conn_info)
        
        if not is_healthy and conn_info.ping_failures >= self.max_ping_failures:
            logger.warning(f"Connection {conn_id} appears dead (ping failures: {conn_info.ping_failures})")
            
            # Only auto-close if explicitly enabled (DISABLED BY DEFAULT)
            if self.auto_close_dead_connections:
                logger.warning(f"Auto-closing dead connection {conn_id}")
                async with self._lock:
                    if conn_id in self._connections:
                        del self._connections[conn_id]
                logger.info(f"Removed dead connection {conn_id} from monitor")
            else:
                # Just log the dead connection, don't close it
                logger.info(f"Dead connection detected but auto-close disabled: {conn_id}")
                self.logfire.log_connection_event(
                    connection_id=conn_id,
                    event="dead_connection_detected",
                    ping_failures=conn_info.ping_failures,
                    inactive_seconds=conn_info.inactive_seconds,
                    auto_close_disabled=True
                )
    
    async def _check_connection_health(self, conn_info: ConnectionInfo) -> bool:
        """
        Check if a connection is healthy by sending a ping
        
        Returns:
            True if connection is healthy, False otherwise
        """
        # Skip health check if connection is being force-closed
        if conn_info.force_closing:
            logger.debug(f"Skipping health check for {conn_info.connection_id} - being force closed")
            return True
        
        # Skip if WebSocket is already closed
        if conn_info.websocket_closed:
            logger.debug(f"Skipping health check for {conn_info.connection_id} - WebSocket already closed")
            return False
        
        try:
            # Try to send ping with timeout - but don't close on failure
            await asyncio.wait_for(
                conn_info.websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()}),
                timeout=5.0
            )
            
            conn_info.last_ping = datetime.utcnow()
            conn_info.total_pings += 1
            
            # Reset failures on successful ping
            if conn_info.ping_failures > 0:
                logger.info(f"Connection {conn_info.connection_id} recovered after {conn_info.ping_failures} failures")
                conn_info.ping_failures = 0
            
            return True
            
        except asyncio.TimeoutError:
            conn_info.ping_failures += 1
            logger.debug(f"Ping timeout for {conn_info.connection_id} (failures: {conn_info.ping_failures})")
            return False
            
        except Exception as e:
            conn_info.ping_failures += 1
            logger.debug(f"Ping error for {conn_info.connection_id}: {e} (failures: {conn_info.ping_failures})")
            
            # Check if it's a connection error
            error_str = str(e).lower()
            if "closed" in error_str or "disconnected" in error_str:
                conn_info.websocket_closed = True
            
            return False
    
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
    
    def mark_connection_for_closure(self, connection_id: str):
        """Mark a connection as being force-closed to prevent health checks"""
        if connection_id in self._connections:
            self._connections[connection_id].force_closing = True
            logger.info(f"Marked connection {connection_id} for closure - will skip health checks")
    
    def mark_websocket_closed(self, connection_id: str):
        """Mark a WebSocket as closed to prevent ping attempts"""
        if connection_id in self._connections:
            self._connections[connection_id].websocket_closed = True
            logger.info(f"Marked WebSocket {connection_id} as closed")
    
    def update_activity(self, connection_id: str):
        """Update last activity timestamp for a connection"""
        if connection_id in self._connections:
            self._connections[connection_id].last_activity = datetime.utcnow()
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a specific connection"""
        return self._connections.get(connection_id)
    
    def get_all_connections(self) -> Dict[str, ConnectionInfo]:
        """Get all monitored connections - returns a copy"""
        return dict(self._connections)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        total_ping_failures = sum(c.ping_failures for c in self._connections.values())
        active_connections = sum(1 for c in self._connections.values() 
                                if not c.force_closing and not c.websocket_closed)
        avg_inactive = (
            sum(c.inactive_seconds for c in self._connections.values()) / len(self._connections)
            if self._connections else 0
        )
        
        return {
            "total_connections": len(self._connections),
            "active_connections": active_connections,
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