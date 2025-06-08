"""
Speechify TTS Service for Pipecat
"""
from typing import AsyncGenerator, Optional
import json
import io
import asyncio

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
except ImportError:
    logger.error("pydub is required for audio format conversion. Install with: pip install pydub")
    raise


class SpeechifyTTSService(TTSService):
    """Speechify Text-to-Speech service using HTTP streaming.
    
    This service converts text to speech using Speechify's streaming API.
    Audio is returned as MP3 and converted to PCM for Pipecat compatibility.
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
        **kwargs,
    ):
        # Since Speechify doesn't provide word timestamps, we use TTSService
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
        
        self.set_model_name(params.model)
        self.set_voice(voice_id)
        
    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True
        
    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert Pipecat Language to Speechify language format (e.g., en-US)."""
        if language:
            return str(language.value)
        return None
        
    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Speechify streaming API.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Frames containing audio data and control frames
        """
        logger.debug(f"{self}: Generating TTS for text: [{text}]")
        
        url = f"{self._base_url}/v1/audio/stream"
        
        payload = {
            "input": text,
            "voice_id": self._voice_id,
            "model": self.model_name,
        }
        
        # Add language if specified
        if self._params.language:
            payload["language"] = self._params.language
            
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "audio/mpeg",  # We'll request MP3 and convert to PCM
            "Content-Type": "application/json",
        }
        
        try:
            # Start TTFB metrics before making the request
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
                    
                # Start usage metrics after successful response
                await self.start_tts_usage_metrics(text)
                
                # Collect audio chunks for conversion
                audio_buffer = io.BytesIO()
                first_chunk_received = False
                
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        if not first_chunk_received:
                            # Stop TTFB metrics on first chunk
                            await self.stop_ttfb_metrics()
                            first_chunk_received = True
                            logger.debug(f"{self} received first audio chunk")
                            
                        audio_buffer.write(chunk)
                
                # Convert MP3 to PCM
                audio_buffer.seek(0)
                try:
                    # Load MP3 audio
                    audio_segment = AudioSegment.from_mp3(audio_buffer)
                    
                    # Convert to mono if needed
                    if audio_segment.channels > 1:
                        audio_segment = audio_segment.set_channels(1)
                        
                    # Resample to target sample rate
                    audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                    
                    # Convert to 16-bit PCM
                    audio_segment = audio_segment.set_sample_width(2)  # 16-bit
                    
                    # Get raw PCM data
                    pcm_data = audio_segment.raw_data
                    
                    # Signal TTS has started
                    yield TTSStartedFrame()
                    
                    # Send PCM audio in chunks
                    chunk_size = 8192  # Send in 8KB chunks
                    for i in range(0, len(pcm_data), chunk_size):
                        chunk = pcm_data[i:i + chunk_size]
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1
                        )
                        
                except Exception as e:
                    logger.error(f"Error converting audio format: {e}")
                    yield ErrorFrame(error=f"Audio conversion error: {str(e)}")
                    return
                    
                # Signal TTS has stopped
                yield TTSStoppedFrame()
                
        except asyncio.TimeoutError:
            logger.error(f"{self} request timeout")
            yield ErrorFrame(error="Request timeout")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            # Always stop TTFB metrics if not already stopped
            await self.stop_ttfb_metrics()