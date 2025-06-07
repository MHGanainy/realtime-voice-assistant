"""
DeepInfra TTS Service
HTTP-stream TTS for DeepInfra's ElevenLabs-compatible endpoint
"""
from typing import Optional, AsyncGenerator
import base64
import json

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
    """HTTP-stream TTS for DeepInfra's ElevenLabs-compatible endpoint"""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "hexgrad/Kokoro-82M",
        base_url: str = "https://api.deepinfra.com",
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> None:
        logger.debug(
            "Initialising DeepInfraHttpTTSService (voice_id='{}', model='{}', base_url='{}')",
            voice_id,
            model,
            base_url,
        )

        # Parent constructor wires metrics, sentence aggregation, etc.
        super().__init__(
            api_key=api_key,
            voice_id=voice_id,
            aiohttp_session=aiohttp_session,
            model=model,
            base_url=base_url.rstrip("/"),
            sample_rate=sample_rate,
            **kwargs,
        )

    def _build_request(self, text: str):
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
        }
        return url, payload, headers

    @traced_tts
    async def run_tts(self, text: str):  # type: ignore[override]
        """Yield frames from DeepInfra"""
        url, payload, headers = self._build_request(text)

        try:
            await self.start_ttfb_metrics()
            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    err = await response.text()
                    logger.error(f"{self} API error {response.status}: {err}")
                    yield ErrorFrame(error=f"DeepInfra API error: {err}")
                    return

                await self.start_tts_usage_metrics(text)

                # First chunk → emit TTSStartedFrame exactly once
                first_chunk = True
                utterance_duration = 0.0

                # Check if we're expecting timestamps (JSON) or raw audio
                expecting_json = payload.get("return_timestamps", False)

                async for raw in response.content:
                    if first_chunk:
                        self.start_word_timestamps()
                        yield TTSStartedFrame()
                        self._started = True
                        first_chunk = False

                    if expecting_json:
                        # We're expecting JSON with timestamps
                        try:
                            txt = raw.decode('utf-8')
                            stripped = txt.strip()
                            if not stripped:
                                continue  # keep-alive blank line
                            
                            data = json.loads(stripped)
                            
                            # Process base64 audio if present
                            if "audio_base64" in data:
                                await self.stop_ttfb_metrics()
                                audio = base64.b64decode(data["audio_base64"])
                                yield TTSAudioRawFrame(audio, self.sample_rate, 1)

                            # Process word timestamps if present
                            if "words" in data and data["words"]:
                                word_times = []
                                for w in data["words"]:
                                    start = float(w.get("start", 0))
                                    end = float(w.get("end", start))
                                    word_times.append((w.get("text", ""), self._cumulative_time + start))
                                    utterance_duration = max(utterance_duration, end)
                                await self.add_word_timestamps(word_times)
                                
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            continue
                    else:
                        # We're expecting raw PCM audio
                        await self.stop_ttfb_metrics()
                        # The raw bytes are already PCM audio, send directly
                        yield TTSAudioRawFrame(raw, self.sample_rate, 1)

                # Bump cursor so next call starts after this audio
                if utterance_duration:
                    self._cumulative_time += utterance_duration

        except Exception as exc:
            logger.error(f"{self} exception: {exc}")
            yield ErrorFrame(error=str(exc))
        finally:
            await self.stop_ttfb_metrics()