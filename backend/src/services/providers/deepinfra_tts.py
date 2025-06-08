"""
Optimized DeepInfra TTS Service with reduced TTFB
"""
from typing import Optional, AsyncGenerator, Dict
import base64
import json
import asyncio
from collections import deque
from time import time

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class DeepInfraHttpTTSService(ElevenLabsHttpTTSService):
    """Optimized HTTP-stream TTS for DeepInfra with connection pooling"""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "hexgrad/Kokoro-82M",
        base_url: str = "https://api.deepinfra.com",
        sample_rate: Optional[int] = None,
        connection_pool_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            api_key=api_key,
            voice_id=voice_id,
            aiohttp_session=aiohttp_session,
            model=model,
            base_url=base_url.rstrip("/"),
            sample_rate=sample_rate,
            **kwargs,
        )
        
        self._connection_pool_size = connection_pool_size
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()
        
        self._last_request_time = 0
        self._request_times = deque(maxlen=100)
        
        self._configure_session()

    def _configure_session(self):
        """Configure aiohttp session for optimal performance"""
        connector = self._session.connector
        if hasattr(connector, '_limit'):
            connector._limit = max(connector._limit, self._connection_pool_size * 2)
        if hasattr(connector, '_limit_per_host'):
            connector._limit_per_host = max(connector._limit_per_host, self._connection_pool_size)
            
    def _build_request(self, text: str, priority: int = 0):
        """Return (url, payload, headers) for a DeepInfra inference call"""
        url = f"{self._base_url}/v1/inference/{self._model_name}"
        payload = {
            "text": text,
            "preset_voice": [self._voice_id],
            "output_format": "pcm",
            "stream": True,
            "return_timestamps": False,
        }
        headers = {
            "Authorization": f"bearer {self._api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "X-Priority": str(priority),
        }
        return url, payload, headers

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Optimized TTS with connection pooling"""
        
        url, payload, headers = self._build_request(text)
        
        request_start = time()
        
        try:
            await self.start_ttfb_metrics()
            
            async with self._session.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30, connect=5)
            ) as response:
                if response.status != 200:
                    err = await response.text()
                    logger.error(f"{self} API error {response.status}: {err}")
                    yield ErrorFrame(error=f"DeepInfra API error: {err}")
                    return

                await self.start_tts_usage_metrics(text)

                first_byte_time = time() - request_start
                self._request_times.append(first_byte_time)

                first_chunk = True
                
                async for raw in response.content.iter_any():
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        self.start_word_timestamps()
                        yield TTSStartedFrame()
                        self._started = True
                        first_chunk = False
                    
                    yield TTSAudioRawFrame(raw, self.sample_rate, 1)

        except asyncio.TimeoutError:
            logger.error(f"{self} request timeout")
            yield ErrorFrame(error="Request timeout")
        except Exception as exc:
            logger.error(f"{self} exception: {exc}")
            yield ErrorFrame(error=str(exc))
        finally:
            await self.stop_ttfb_metrics()

    def get_average_ttfb(self) -> float:
        """Get average time to first byte from recent requests"""
        if self._request_times:
            return sum(self._request_times) / len(self._request_times)
        return 0.0