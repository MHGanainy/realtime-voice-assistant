"""
Event security and access control
"""
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EventSecurity:
    """Handle event access control based on session IDs"""
    
    def __init__(self, conversation_manager):
        self._conversation_manager = conversation_manager
    
    def can_access_event(
        self,
        session_id: str,
        event_name: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Check if a session can access an event"""
        if not session_id:
            return False
        
        if event_name.startswith("global:"):
            return True
        
        if event_name.startswith("conversation:"):
            parts = event_name.split(":")
            if len(parts) >= 2:
                conv_id = parts[1]
                conversation = self._conversation_manager.get_conversation(conv_id)
                
                if conversation:
                    return conversation.participant.session_id == session_id
                else:
                    event_session_id = event_data.get("session_id")
                    return event_session_id == session_id
            return False
        
        if event_name.startswith("connection:"):
            event_session_id = event_data.get("session_id")
            return event_session_id == session_id
        
        if event_name.startswith("participant:"):
            event_session_id = event_data.get("session_id")
            return event_session_id == session_id
        
        return True
    
    def filter_event_data(
        self,
        session_id: str,
        event_name: str,
        event_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Filter event data based on session access"""
        if not self.can_access_event(session_id, event_name, event_data):
            return None
        return event_data
    
    def get_session_conversations(self, session_id: str) -> list[str]:
        """Get all conversation IDs accessible by a session"""
        conversations = []
        
        all_convs = self._conversation_manager.get_all_conversations()
        for conv_id, conv in all_convs.items():
            if conv.participant.session_id == session_id:
                conversations.append(conv_id)
        
        return conversations


_event_security: Optional[EventSecurity] = None


def get_event_security(conversation_manager) -> EventSecurity:
    """Get the singleton event security instance"""
    global _event_security
    if _event_security is None:
        _event_security = EventSecurity(conversation_manager)
    return _event_security