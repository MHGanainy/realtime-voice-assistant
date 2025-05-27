import aiohttp
import asyncio
import json
import logging
from typing import AsyncIterator

from ..interfaces.tts_base import TTS
from ..config import settings

# Set up logging
logger = logging.getLogger(__name__)

STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # default voice

class ElevenTTS(TTS):
    def __init__(self):
        self.buffer = ""
        
    async def stream(self, text_iter: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Stream text to TTS with simple sentence-based chunking.
        """
        self.buffer = ""
        
        async for token in text_iter:
            self.buffer += token
            
            # Look for sentence endings
            while True:
                # Find the first sentence ending
                end_pos = -1
                for i, char in enumerate(self.buffer):
                    if char in '.!?' and i < len(self.buffer) - 1:
                        # Make sure it's really a sentence end (followed by space or newline or end)
                        if i == len(self.buffer) - 1 or self.buffer[i + 1] in ' \n\r':
                            end_pos = i + 1
                            break
                
                if end_pos > 0:
                    # Extract and synthesize the sentence
                    sentence = self.buffer[:end_pos].strip()
                    if sentence:
                        logger.debug(f"Synthesizing: '{sentence}'")
                        async for audio_chunk in self._synth(sentence):
                            yield audio_chunk
                    
                    # Remove the synthesized part from buffer
                    self.buffer = self.buffer[end_pos:].lstrip()
                else:
                    # No complete sentence found, wait for more tokens
                    break
        
        # Synthesize any remaining text
        if self.buffer.strip():
            logger.debug(f"Synthesizing final chunk: '{self.buffer.strip()}'")
            async for audio_chunk in self._synth(self.buffer.strip()):
                yield audio_chunk
    
    async def _synth(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesize text using ElevenLabs API.
        """
        if not text:
            return
            
        headers = {
            "xi-api-key": settings.eleven_api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            },
            "optimize_streaming_latency": 1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    STREAM_URL.format(voice_id=VOICE_ID),
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        # Stream the audio chunks
                        async for chunk in resp.content.iter_chunked(2048):
                            if chunk:  # Make sure chunk is not empty
                                yield chunk
                    else:
                        error_text = await resp.text()
                        logger.error(f"ElevenLabs API error ({resp.status}): {error_text}")
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}", exc_info=True)