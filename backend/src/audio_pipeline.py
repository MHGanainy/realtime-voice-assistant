import numpy as np
import os
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import AudioRawFrame
import logging

# Import LiveOptions if we're using Deepgram
try:
    from deepgram import LiveOptions
except ImportError:
    LiveOptions = None

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, sample_width: int = 2):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
    
    def bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw audio bytes to numpy array"""
        # Convert bytes to int16 numpy array
        return np.frombuffer(audio_bytes, dtype=np.int16)


class AudioFrameProcessor:
    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def create_audio_frame(self, audio_bytes: bytes) -> AudioRawFrame:
        """Create a Pipecat AudioRawFrame from raw audio bytes"""
        return AudioRawFrame(
            audio=audio_bytes,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels
        )


class AudioPipelineHandler:
    """Handles audio processing through the pipeline"""
    def __init__(self, stt_service=None):
        self._ready = True
        self._audio_buffer = bytearray()
        self.stt_service = stt_service
        self.frame_processor = AudioFrameProcessor()
        self._audio_received = False
        self._chunk_count = 0
        
        # For MVP, we'll batch audio before sending to STT
        self.batch_size = 16000  # 1 second of audio at 16kHz
    
    async def process_audio(self, audio_bytes: bytes):
        """Process incoming audio bytes"""
        # Buffer the audio
        self._audio_buffer.extend(audio_bytes)
        self._audio_received = True
        self._chunk_count += 1
        
        # Log every 10 chunks (1 second)
        if self._chunk_count % 10 == 0:
            logger.info(f"Received {self._chunk_count} audio chunks, buffer size: {len(self._audio_buffer)} bytes")
            
        # Check if we have enough audio to process (1 second)
        if len(self._audio_buffer) >= self.batch_size:
            logger.info("🎯 Buffer full - would send to STT now")
            # In a real implementation, we would send this to STT
            # For now, just clear part of the buffer
            self._audio_buffer = self._audio_buffer[self.batch_size:]
        
    def is_ready(self) -> bool:
        """Check if handler is ready to process more audio"""
        return self._ready
    
    def has_received_audio(self) -> bool:
        """Check if audio has been received"""
        return self._audio_received


def create_stt_service(service_name: str, **kwargs):
    """Create the appropriate STT service"""
    if service_name == "deepgram":
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        
        # Get model from kwargs or use default
        model = kwargs.get("model", "nova-3")
        
        # Create LiveOptions
        live_options = LiveOptions(
            model=model,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True
        )
        
        return DeepgramSTTService(
            api_key=api_key,
            live_options=live_options
        )
    else:
        raise ValueError(f"Unknown STT service: {service_name}")


def create_llm_service(service_name: str, **kwargs):
    """Create the appropriate LLM service"""
    if service_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Get model from kwargs or use default
        model = kwargs.get("model", "gpt-3.5-turbo")
        
        return OpenAILLMService(
            api_key=api_key,
            model=model
        )
    # Add more LLM services here as needed
    # elif service_name == "anthropic":
    #     api_key = os.getenv("ANTHROPIC_API_KEY")
    #     return AnthropicLLMService(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown LLM service: {service_name}")


def create_tts_service(service_name: str, **kwargs):
    """Create the appropriate TTS service"""
    if service_name == "elevenlabs":
        api_key = os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY not set")
        
        # Get voice and model from kwargs or use defaults
        voice_id = kwargs.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        model = kwargs.get("model", "eleven_flash_v2_5")
        
        return ElevenLabsTTSService(
            api_key=api_key,
            voice_id=voice_id,
            model=model
        )
    # Add more TTS services here as needed
    # elif service_name == "azure":
    #     api_key = os.getenv("AZURE_SPEECH_KEY")
    #     region = os.getenv("AZURE_SPEECH_REGION")
    #     return AzureTTSService(api_key=api_key, region=region, **kwargs)
    else:
        raise ValueError(f"Unknown TTS service: {service_name}")


def create_llm_context(llm_service: str, system_prompt: str = None, **kwargs):
    """Create the appropriate LLM context based on the service"""
    if llm_service == "openai":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add any initial messages from kwargs
        initial_messages = kwargs.get("initial_messages", [])
        messages.extend(initial_messages)
        
        return OpenAILLMContext(messages)
    # Add more context types here as needed
    # elif llm_service == "anthropic":
    #     return AnthropicLLMContext(system_prompt=system_prompt, **kwargs)
    else:
        raise ValueError(f"Unknown LLM service for context: {llm_service}")


# Add at the end of audio_pipeline.py
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_transport import BaseTransport
from pipecat.frames.frames import StartFrame, EndFrame
import asyncio

class SimpleAudioTransport(BaseTransport):
    """Simple transport for testing audio pipeline"""
    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
    async def send_audio(self, audio_bytes: bytes):
        """Send audio to the pipeline"""
        frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=16000,
            num_channels=1
        )
        await self.audio_queue.put(frame)
    
    async def get_output(self, timeout=1.0):
        """Get output from the pipeline"""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None


class SimplePipelineHandler:
    """Simplified pipeline handler for MVP"""
    def __init__(self, stt_service):
        self.stt_service = stt_service
        self.transport = SimpleAudioTransport()
        self.pipeline = None
        self.task = None
        self.runner = PipelineRunner()
        
    async def start(self):
        """Start the pipeline"""
        # Create simple pipeline: transport -> STT
        self.pipeline = Pipeline([
            self.transport.input(),
            self.stt_service,
            # We'll add more stages later
        ])
        
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
            )
        )
        
        # Start pipeline in background
        asyncio.create_task(self.runner.run(self.task))
        
        # Send start frame
        await self.transport.send_audio(StartFrame())
        
    async def process_audio(self, audio_bytes: bytes):
        """Process audio through pipeline"""
        await self.transport.send_audio(audio_bytes)
        
    async def stop(self):
        """Stop the pipeline"""
        if self.task:
            await self.task.cancel()