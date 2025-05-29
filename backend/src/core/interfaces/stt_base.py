from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
from ..events.types import TranscriptEvent

class STTProvider(ABC):
    """
    Speech-to-Text provider interface with async context manager support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (connect, authenticate, etc.)"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    @abstractmethod
    async def stream(
        self, 
        audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        """
        Stream audio chunks and yield transcript events
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if provider is connected and ready"""
        pass
    
    @abstractmethod
    async def reconnect(self) -> None:
        """Reconnect to the service"""
        pass