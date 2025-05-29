from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any

class TTSProvider(ABC):
    """
    Text-to-Speech provider interface with async context manager support.
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
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    @abstractmethod
    async def stream(
        self, 
        text_iter: AsyncIterator[str],
        voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Stream text to audio"""
        pass
    
    async def synthesize(
        self, 
        text: str,
        voice_id: Optional[str] = None
    ) -> bytes:
        """Synthesize complete text (non-streaming)"""
        chunks = []
        async def text_generator():
            yield text
            
        async for chunk in self.stream(text_generator(), voice_id):
            chunks.append(chunk)
        return b"".join(chunks)