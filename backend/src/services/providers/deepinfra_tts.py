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
    """Optimized HTTP-stream TTS for DeepInfra with connection pooling and prefetching"""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "hexgrad/Kokoro-82M",
        base_url: str = "https://api.deepinfra.com",
        sample_rate: Optional[int] = None,
        # Optimization parameters
        enable_prefetch: bool = True,
        prefetch_chars: int = 50,  # Start prefetching after this many chars
        connection_pool_size: int = 3,  # Number of concurrent connections
        enable_request_pipelining: bool = True,
        **kwargs,
    ) -> None:
        logger.debug(
            "Initialising OptimizedDeepInfraHttpTTSService (voice_id='{}', model='{}', base_url='{}')",
            voice_id,
            model,
            base_url,
        )

        super().__init__(
            api_key=api_key,
            voice_id=voice_id,
            aiohttp_session=aiohttp_session,
            model=model,
            base_url=base_url.rstrip("/"),
            sample_rate=sample_rate,
            **kwargs,
        )
        
        # Optimization settings
        self._enable_prefetch = enable_prefetch
        self._prefetch_chars = prefetch_chars
        self._enable_pipelining = enable_request_pipelining
        
        # Connection pool management
        self._connection_pool_size = connection_pool_size
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._prefetch_buffer = deque(maxlen=10)
        
        # Performance metrics
        self._last_request_time = 0
        self._request_times = deque(maxlen=100)
        
        # Configure session for better performance
        self._configure_session()

    def _configure_session(self):
        """Configure aiohttp session for optimal performance"""
        # Update connector settings for better connection reuse
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
            "Connection": "keep-alive",  # Ensure connection reuse
            "X-Priority": str(priority),  # Custom priority header
        }
        return url, payload, headers

    async def _prefetch_request(self, text: str):
        """Prefetch TTS for anticipated text"""
        if not self._enable_prefetch:
            return
            
        try:
            url, payload, headers = self._build_request(text, priority=1)
            
            # Start the request but don't await it yet
            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    # Store the audio for later use
                    audio_chunks = []
                    async for chunk in response.content:
                        audio_chunks.append(chunk)
                    
                    self._prefetch_buffer.append({
                        'text': text,
                        'audio': audio_chunks,
                        'timestamp': time()
                    })
                    logger.trace(f"Prefetched audio for: {text[:30]}...")
                    
        except Exception as e:
            logger.warning(f"Prefetch failed: {e}")

    def _check_prefetch_cache(self, text: str) -> Optional[list]:
        """Check if we have prefetched audio for this text"""
        if not self._enable_prefetch:
            return None
            
        # Look for exact match in prefetch buffer
        for item in self._prefetch_buffer:
            if item['text'] == text and (time() - item['timestamp']) < 30:  # 30s cache validity
                logger.trace(f"Using prefetched audio for: {text[:30]}...")
                return item['audio']
        
        return None

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Optimized TTS with prefetch cache and connection pooling"""
        
        # Check prefetch cache first
        cached_audio = self._check_prefetch_cache(text)
        if cached_audio:
            await self.start_ttfb_metrics()
            await self.stop_ttfb_metrics()  # Instant TTFB from cache
            
            yield TTSStartedFrame()
            self._started = True
            
            for chunk in cached_audio:
                yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
            
            return
        
        # If not in cache, make the request
        url, payload, headers = self._build_request(text)
        
        # Track request timing
        request_start = time()
        
        try:
            await self.start_ttfb_metrics()
            
            # Use existing session with connection pooling
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

                # Track time to first byte
                first_byte_time = time() - request_start
                self._request_times.append(first_byte_time)
                logger.trace(f"TTFB: {first_byte_time:.3f}s")

                first_chunk = True
                
                # Stream with optimized chunk processing
                async for raw in response.content.iter_any():
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        self.start_word_timestamps()
                        yield TTSStartedFrame()
                        self._started = True
                        first_chunk = False
                        
                        # Start prefetching next anticipated text if enabled
                        if self._enable_prefetch and len(text) > self._prefetch_chars:
                            # This is a simplified example - in practice you'd predict next text
                            asyncio.create_task(self._prefetch_next_segment(text))
                    
                    # Direct audio streaming
                    yield TTSAudioRawFrame(raw, self.sample_rate, 1)

        except asyncio.TimeoutError:
            logger.error(f"{self} request timeout")
            yield ErrorFrame(error="Request timeout")
        except Exception as exc:
            logger.error(f"{self} exception: {exc}")
            yield ErrorFrame(error=str(exc))
        finally:
            await self.stop_ttfb_metrics()

    async def _prefetch_next_segment(self, current_text: str):
        """Predict and prefetch next likely text segment"""
        # This is a placeholder - implement your prediction logic
        # For example, if you're reading sentences, prefetch the next sentence
        pass

    def get_average_ttfb(self) -> float:
        """Get average time to first byte from recent requests"""
        if self._request_times:
            return sum(self._request_times) / len(self._request_times)
        return 0.0


class WebSocketDeepInfraTTSService(ElevenLabsHttpTTSService):
    """WebSocket-based TTS for DeepInfra (if/when they support it)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._websocket = None
        self._receive_task = None
        
    async def _connect_websocket(self):
        """Connect to WebSocket endpoint"""
        # This is hypothetical - DeepInfra would need to provide a WebSocket endpoint
        ws_url = f"wss://api.deepinfra.com/v1/inference/{self._model_name}/stream"
        
        try:
            import websockets
            self._websocket = await websockets.connect(
                ws_url,
                extra_headers={
                    "Authorization": f"bearer {self._api_key}",
                }
            )
            logger.debug("Connected to DeepInfra WebSocket")
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_messages())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _receive_messages(self):
        """Receive messages from WebSocket"""
        async for message in self._websocket:
            data = json.loads(message)
            if "audio" in data:
                audio = base64.b64decode(data["audio"])
                await self.push_frame(TTSAudioRawFrame(audio, self.sample_rate, 1))
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Send TTS request over WebSocket"""
        if not self._websocket:
            await self._connect_websocket()
        
        await self.start_ttfb_metrics()
        
        # Send request
        request = {
            "text": text,
            "preset_voice": [self._voice_id],
            "output_format": "pcm",
        }
        await self._websocket.send(json.dumps(request))
        
        # The receive task will handle incoming audio
        yield None  # Placeholder since audio comes through receive task