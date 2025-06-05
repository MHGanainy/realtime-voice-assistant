import numpy as np
import os
from pipecat.services.deepgram.stt import DeepgramSTTService
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


def create_stt_service(service_name: str):
    """Create the appropriate STT service"""
    if service_name == "deepgram":
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        
        # Create LiveOptions with nova-2 model
        live_options = LiveOptions(
            model="nova-2",
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