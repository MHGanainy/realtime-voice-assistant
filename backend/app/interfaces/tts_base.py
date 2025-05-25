from abc import ABC, abstractmethod
from typing import AsyncIterator

class TTS(ABC):
    @abstractmethod
    async def stream(self, text_iter: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Consume text stream and yield audio bytes (PCM or MP3)."""