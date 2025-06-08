"""
Speechify TTS Service for Pipecat - Optimized for low latency with streaming decoder
"""
from typing import AsyncGenerator, Optional
import asyncio
import subprocess

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


class SpeechifyTTSService(TTSService):
    """Speechify Text-to-Speech service using HTTP streaming.
    
    Uses ffmpeg for streaming MP3 to PCM conversion with low latency.
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
        return True
        
    def language_to_service_language(self, language: Language) -> Optional[str]:
        if language:
            return str(language.value)
        return None
        
    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Speechify streaming API.
        
        Uses ffmpeg for real-time MP3 to PCM conversion.
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
        
        # FFmpeg command for streaming MP3 to PCM conversion
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', 'pipe:0',           # Input from stdin
            '-f', 'mp3',              # Input format
            '-f', 's16le',            # Output format: signed 16-bit little-endian
            '-ar', str(self.sample_rate),  # Sample rate
            '-ac', '1',               # Mono
            '-loglevel', 'error',     # Only show errors
            'pipe:1'                  # Output to stdout
        ]
        
        process = None
        
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
                
                # Start ffmpeg process
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                first_chunk_received = False
                
                # Create tasks for reading and writing
                async def write_to_ffmpeg():
                    try:
                        async for chunk in response.content.iter_chunked(4096):
                            if chunk:
                                if not first_chunk_received:
                                    await self.stop_ttfb_metrics()
                                    logger.debug(f"{self} received first audio chunk")
                                
                                process.stdin.write(chunk)
                                await process.stdin.drain()
                    except Exception as e:
                        logger.error(f"Error writing to ffmpeg: {e}")
                    finally:
                        process.stdin.close()
                
                async def read_from_ffmpeg():
                    nonlocal first_chunk_received
                    try:
                        while True:
                            # Read PCM data in chunks
                            pcm_chunk = await process.stdout.read(4096)
                            if not pcm_chunk:
                                break
                                
                            if not first_chunk_received:
                                first_chunk_received = True
                                yield TTSStartedFrame()
                                
                            yield TTSAudioRawFrame(
                                audio=pcm_chunk,
                                sample_rate=self.sample_rate,
                                num_channels=1
                            )
                    except Exception as e:
                        logger.error(f"Error reading from ffmpeg: {e}")
                
                # Run write task in background
                write_task = asyncio.create_task(write_to_ffmpeg())
                
                # Read and yield audio frames
                async for frame in read_from_ffmpeg():
                    yield frame
                
                # Wait for write task to complete
                await write_task
                
                # Wait for ffmpeg to finish
                await process.wait()
                
                if process.returncode != 0:
                    stderr = await process.stderr.read()
                    error_msg = stderr.decode()
                    if error_msg:
                        logger.error(f"FFmpeg error: {error_msg}")
                
                yield TTSStoppedFrame()
                
        except asyncio.TimeoutError:
            logger.error(f"{self} request timeout")
            yield ErrorFrame(error="Request timeout")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            # Ensure process is terminated
            if process and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    
            await self.stop_ttfb_metrics()