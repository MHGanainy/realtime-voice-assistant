from typing import AsyncIterator, Optional, Dict, Any, List
import aiohttp
import asyncio
import logging
from ...core.interfaces.tts_base import TTSProvider
from ...utils.decorators import provider

logger = logging.getLogger(__name__)

@provider("tts", "elevenlabs")
class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS provider with retry logic"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://api.elevenlabs.io/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.buffer = ""
        
    async def initialize(self) -> None:
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("ElevenLabs TTS initialized")
        
    async def cleanup(self) -> None:
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
            
    async def stream(
        self, 
        text_iter: AsyncIterator[str],
        voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Stream text to audio with sentence-based chunking"""
        voice_id = voice_id or self.config.get("voice_id", "EXAVITQu4vr4xnSDxMaL")
        self.buffer = ""
        
        async for token in text_iter:
            self.buffer += token
            
            # Check for sentence boundaries
            sentences = self._extract_sentences()
            for sentence in sentences:
                if sentence.strip():
                    async for chunk in self._synthesize_chunk(sentence, voice_id):
                        yield chunk
                        
        # Process remaining buffer
        if self.buffer.strip():
            async for chunk in self._synthesize_chunk(self.buffer.strip(), voice_id):
                yield chunk
                
    def _extract_sentences(self) -> List[str]:
        """Extract complete sentences from buffer"""
        sentences = []
        
        while True:
            # Find sentence ending
            end_pos = -1
            for i, char in enumerate(self.buffer):
                if char in '.!?' and i < len(self.buffer) - 1:
                    if i == len(self.buffer) - 1 or self.buffer[i + 1] in ' \n\r':
                        end_pos = i + 1
                        break
                        
            if end_pos > 0:
                sentence = self.buffer[:end_pos].strip()
                sentences.append(sentence)
                self.buffer = self.buffer[end_pos:].lstrip()
            else:
                break
                
        return sentences
        
    async def _synthesize_chunk(
        self, 
        text: str, 
        voice_id: str
    ) -> AsyncIterator[bytes]:
        """Synthesize a text chunk with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
                
                headers = {
                    "xi-api-key": self.config["api_key"],
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text": text,
                    "model_id": self.config.get("model_id", "eleven_monolingual_v1"),
                    "voice_settings": {
                        "stability": self.config.get("stability", 0.5),
                        "similarity_boost": self.config.get("similarity_boost", 0.5)
                    },
                    "optimize_streaming_latency": self.config.get("optimize_streaming_latency", 1)
                }
                
                async with self.session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        async for chunk in resp.content.iter_chunked(2048):
                            if chunk:
                                yield chunk
                        return  # Success, exit retry loop
                    else:
                        error_text = await resp.text()
                        raise aiohttp.ClientError(f"ElevenLabs API error ({resp.status}): {error_text}")
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"TTS attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"TTS failed after {max_retries} attempts: {e}")
                    # Yield empty audio on failure
                    yield b""