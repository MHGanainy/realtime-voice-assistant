from abc import ABC, abstractmethod
from typing import AsyncIterator

class STT(ABC):
    """Speech‑to‑Text base interface."""

    @abstractmethod
    async def stream(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Yield partial (or final) transcripts while audio chunks stream in."""