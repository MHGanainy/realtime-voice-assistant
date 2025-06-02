# backend/src/services/deepinfra_tts.py
"""
DeepInfra TTS Service for Pipecat
Provides access to multiple TTS models through DeepInfra's OpenAI-compatible API
"""
import base64
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class DeepInfraTTSService(TTSService):
    """DeepInfra Text-to-Speech service supporting multiple models.

    This service uses DeepInfra's OpenAI-compatible TTS API to generate audio.
    Supports Orpheus, Kokoro, and Dia models with their respective voices.

    Args:
        api_key: DeepInfra API key. Required.
        voice: Voice ID to use. Defaults to model-appropriate voice.
        model: TTS model to use. Defaults to "canopylabs/orpheus-3b-0.1-ft".
        sample_rate: Output audio sample rate in Hz. Defaults to 24000.
        response_format: Audio format. Options: pcm, mp3, opus, flac, wav. Defaults to "pcm".
        speed: Speed of speech (0.25-4.0). Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to TTSService.
    """

    # Model-specific voice mappings
    MODEL_VOICES = {
        "canopylabs/orpheus-3b-0.1-ft": {
            "voices": ["tara", "adam", "sarah", "michael", "emma", "olivia"],
            "default": "tara",
            "native_sample_rate": 24000
        },
        "hexgrad/Kokoro-82M": {
            "voices": ["af_bella", "af_nicole", "af_sarah", "af_sky", 
                      "am_adam", "am_michael", "bf_emma", "bf_isabella", 
                      "bm_george", "bm_lewis"],
            "default": "af_bella",
            "native_sample_rate": 24000
        },
        "nari-labs/Dia-1.6B": {
            "voices": ["luna", "sol", "aurora", "nova", "orion", "lyra"],
            "default": "luna",
            "native_sample_rate": 44100  # Dia outputs at 44.1kHz
        }
    }
    
    # Supported audio formats
    VALID_FORMATS = ["pcm", "mp3", "opus", "flac", "wav"]
    
    # Default sample rates for different formats
    FORMAT_SAMPLE_RATES = {
        "pcm": 24000,
        "mp3": 24000,
        "opus": 24000,
        "flac": 24000,
        "wav": 24000,
    }

    def __init__(
        self,
        *,
        api_key: str,
        voice: Optional[str] = None,
        model: str = "canopylabs/orpheus-3b-0.1-ft",
        sample_rate: Optional[int] = None,
        response_format: str = "pcm",
        speed: float = 1.0,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        **kwargs,
    ):
        # Validate model
        if model not in self.MODEL_VOICES:
            logger.warning(f"Unknown model '{model}', defaulting to Orpheus")
            model = "canopylabs/orpheus-3b-0.1-ft"
        
        # Set voice based on model if not provided
        model_info = self.MODEL_VOICES[model]
        if voice is None:
            voice = model_info["default"]
        elif voice not in model_info["voices"]:
            logger.warning(f"Unknown voice '{voice}' for model {model}, defaulting to '{model_info['default']}'")
            voice = model_info["default"]
            
        if response_format not in self.VALID_FORMATS:
            raise ValueError(f"Invalid response_format. Must be one of: {self.VALID_FORMATS}")
            
        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        # Set default sample rate based on format if not provided
        if sample_rate is None:
            sample_rate = self.FORMAT_SAMPLE_RATES.get(response_format, 24000)
            
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._api_key = api_key
        self._base_url = base_url
        self._voice = voice
        self._response_format = response_format
        self._speed = speed
        self._model_info = model_info
        self.set_model_name(model)
        
        # Create HTTP client with appropriate timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0, connect=10.0)
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        """Change the voice for TTS generation."""
        if voice in self._model_info["voices"]:
            logger.info(f"Switching TTS voice to: [{voice}]")
            self._voice = voice
        else:
            logger.warning(f"Unknown voice '{voice}' for model {self.model_name}, keeping current voice '{self._voice}'")

    async def set_speed(self, speed: float):
        """Change the speech speed."""
        if 0.25 <= speed <= 4.0:
            logger.info(f"Setting TTS speed to: {speed}x")
            self._speed = speed
        else:
            logger.warning(f"Speed {speed} out of range (0.25-4.0), keeping current speed {self._speed}")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        model_display = self.model_name.split('/')[-1]
        logger.info(f"DeepInfra TTS started with model '{model_display}', voice '{self._voice}' at {self.sample_rate}Hz")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            await self.start_ttfb_metrics()
            
            # Prepare request parameters
            params = {
                "model": self.model_name,
                "input": text,
                "voice": self._voice,
                "response_format": self._response_format,
                "speed": self._speed,
            }
            
            # Make streaming request
            async with self._client.stream(
                "POST",
                "/audio/speech",
                json=params,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(
                        f"{self} error getting audio (status: {response.status_code}, error: {error_text})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {response.status_code})"
                    )
                    return
                
                await self.start_tts_usage_metrics(text)
                
                yield TTSStartedFrame()
                
                # Process streaming response
                first_chunk = True
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if len(chunk) > 0:
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False
                        
                        # For PCM format, chunk is raw audio data
                        # For other formats, you might need additional processing
                        if self._response_format == "pcm":
                            frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                            yield frame
                        else:
                            # For non-PCM formats, we need to decode the audio
                            # This is a simplified version - you might need more sophisticated handling
                            frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                            yield frame
                
                yield TTSStoppedFrame()
                
        except httpx.TimeoutException:
            logger.error(f"{self} timeout generating TTS")
            yield ErrorFrame("TTS request timed out")
        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            yield ErrorFrame(f"TTS generation error: {str(e)}")

    async def close(self):
        """Clean up resources."""
        await self._client.aclose()
        await super().close()