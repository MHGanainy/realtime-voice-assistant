"""
Speechify TTS Service for Pipecat - Optimized for low latency
"""
from typing import AsyncGenerator, Optional
import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.transcriptions.language import Language

# Audio conversion imports
try:
    from pydub import AudioSegment
    import pydub.exceptions
except ImportError:
    logger.error("pydub is required for audio format conversion. Install with: pip install pydub")
    raise


class SpeechifyTTSService(TTSService):
    """Speechify Text-to-Speech service using HTTP streaming.
    
    Optimized for low latency by streaming and converting audio chunks as they arrive.
    """
    
    class InputParams(BaseModel):
        language: Optional[str] = None
        model: str = "simba-english"
        
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "simba-english",
        base_url: str = "https://api.sws.speechify.com",
        sample_rate: int = 24000,
        params: Optional[InputParams] = None,
        chunk_size: int = 4096,  # Smaller chunks for lower latency
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )
        
        params = params or SpeechifyTTSService.InputParams()
        
        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._params = params
        self._chunk_size = chunk_size
        
        # Thread pool for CPU-intensive audio conversion
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        self.set_model_name(params.model)
        self.set_voice(voice_id)
        
    def can_generate_metrics(self) -> bool:
        return True
        
    def language_to_service_language(self, language: Language) -> Optional[str]:
        if language:
            return str(language.value)
        return None
    
    def _convert_audio_chunk(self, mp3_data: bytes) -> Optional[bytes]:
        """Convert MP3 chunk to PCM in a separate thread to avoid blocking."""
        try:
            # Create AudioSegment from MP3 data
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)
                
            # Resample to target sample rate
            audio = audio.set_frame_rate(self.sample_rate)
            
            # Convert to 16-bit PCM
            audio = audio.set_sample_width(2)
            
            return audio.raw_data
        except Exception as e:
            logger.error(f"Error converting audio chunk: {e}")
            return None
        
    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Speechify streaming API with low latency.
        
        Streams and converts audio chunks as they arrive rather than waiting for all data.
        """
        logger.debug(f"{self}: Generating TTS for text: [{text}]")
        
        url = f"{self._base_url}/v1/audio/stream"
        
        payload = {
            "input": text,
            "voice_id": self._voice_id,
            "model": self.model_name,
        }
        
        if self._params.language:
            payload["language"] = self._params.language
            
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        
        try:
            await self.start_ttfb_metrics()
            
            async with self._session.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self} API error {response.status}: {error_text}")
                    yield ErrorFrame(error=f"Speechify API error: {error_text}")
                    return
                    
                await self.start_tts_usage_metrics(text)
                
                first_chunk_received = False
                mp3_buffer = bytearray()
                min_mp3_chunk_size = 1024  # Minimum size for valid MP3 frame
                
                # Start streaming immediately
                async for chunk in response.content.iter_chunked(self._chunk_size):
                    if chunk:
                        if not first_chunk_received:
                            await self.stop_ttfb_metrics()
                            first_chunk_received = True
                            yield TTSStartedFrame()
                            logger.debug(f"{self} received first audio chunk")
                        
                        # Accumulate MP3 data
                        mp3_buffer.extend(chunk)
                        
                        # Process when we have enough data for a valid MP3 frame
                        if len(mp3_buffer) >= min_mp3_chunk_size:
                            # Convert in thread pool to avoid blocking
                            loop = asyncio.get_event_loop()
                            pcm_data = await loop.run_in_executor(
                                self._executor,
                                self._convert_audio_chunk,
                                bytes(mp3_buffer)
                            )
                            
                            if pcm_data:
                                # Send PCM audio immediately
                                yield TTSAudioRawFrame(
                                    audio=pcm_data,
                                    sample_rate=self.sample_rate,
                                    num_channels=1
                                )
                                
                            # Clear buffer after processing
                            mp3_buffer.clear()
                
                # Process any remaining data
                if mp3_buffer:
                    loop = asyncio.get_event_loop()
                    pcm_data = await loop.run_in_executor(
                        self._executor,
                        self._convert_audio_chunk,
                        bytes(mp3_buffer)
                    )
                    
                    if pcm_data:
                        yield TTSAudioRawFrame(
                            audio=pcm_data,
                            sample_rate=self.sample_rate,
                            num_channels=1
                        )
                
                yield TTSStoppedFrame()
                
        except asyncio.TimeoutError:
            logger.error(f"{self} request timeout")
            yield ErrorFrame(error="Request timeout")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_ttfb_metrics()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup thread pool on exit."""
        self._executor.shutdown(wait=True)
        await super().__aexit__(exc_type, exc_val, exc_tb)