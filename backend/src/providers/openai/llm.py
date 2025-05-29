from typing import AsyncIterator, Optional, List, Dict, Any
import logging
from openai import AsyncOpenAI
from ...core.interfaces.llm_base import LLMProvider, Message
from ...utils.decorators import provider

logger = logging.getLogger(__name__)

@provider("llm", "openai")
class OpenAIChat(LLMProvider):
    """OpenAI Chat completion provider with retry logic"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        self.client = AsyncOpenAI(api_key=self.config["api_key"])
        logger.info("OpenAI client initialized")
        
    async def cleanup(self) -> None:
        """Cleanup OpenAI client"""
        if self.client:
            await self.client.close()
            
    async def stream(
        self, 
        prompt: str,
        context: Optional[List[Message]] = None
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        messages = self._prepare_messages(prompt, context)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.get("model", "gpt-3.5-turbo"),
                stream=True,
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens"),
            )
            
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # For now, yield an error message
            yield f"I apologize, but I encountered an error: {str(e)}"
                
    def _prepare_messages(
        self, 
        prompt: str, 
        context: Optional[List[Message]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        messages = []
        
        # Add system prompt
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        messages.append({"role": "system", "content": system_prompt})
        
        # Add context or history
        if context:
            for msg in context:
                messages.append({"role": msg.role, "content": msg.content})
        else:
            # Use conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
                
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages