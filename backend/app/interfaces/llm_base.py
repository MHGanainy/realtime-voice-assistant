from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLM(ABC):
    @abstractmethod
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield response tokens/strings."""