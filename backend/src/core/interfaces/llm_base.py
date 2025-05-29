from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any

class Message:
    """Chat message"""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class LLMProvider(ABC):
    """
    Language Model provider interface with async context manager support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_history: List[Message] = []
        self._system_prompt = config.get('system_prompt', 'You are a helpful assistant.')
        
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
        prompt: str,
        context: Optional[List[Message]] = None
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        pass
    
    async def get_response(
        self, 
        prompt: str,
        context: Optional[List[Message]] = None
    ) -> str:
        """Get complete response (non-streaming)"""
        tokens = []
        async for token in self.stream(prompt, context):
            tokens.append(token)
        return "".join(tokens)
    
    def add_to_history(self, message: Message) -> None:
        """Add message to conversation history"""
        self.conversation_history.append(message)
        
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the assistant"""
        self._system_prompt = prompt
        
    def get_system_prompt(self) -> str:
        """Get the current system prompt"""
        return self._system_prompt