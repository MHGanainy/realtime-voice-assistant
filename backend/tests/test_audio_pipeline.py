import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

@pytest.mark.asyncio
async def test_audio_processor_converts_bytes_to_numpy():
    """Test that audio processor converts raw bytes to numpy array"""
    from src.audio_pipeline import AudioProcessor
    
    processor = AudioProcessor(sample_rate=16000, sample_width=2)
    
    # Create fake audio data (10 samples of silence)
    audio_bytes = bytes(20)  # 10 samples * 2 bytes per sample
    
    # Convert to numpy
    audio_array = processor.bytes_to_numpy(audio_bytes)
    
    assert audio_array is not None
    assert len(audio_array) == 10
    assert audio_array.dtype.name == 'int16'

@pytest.mark.asyncio
async def test_stt_service_initialization():
    """Test that we can initialize a Deepgram STT service"""
    with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
        from src.audio_pipeline import create_stt_service
        from pipecat.services.deepgram.stt import DeepgramSTTService
        
        stt_service = create_stt_service("deepgram")
        
        assert stt_service is not None
        # Verify it's a Deepgram service
        assert isinstance(stt_service, DeepgramSTTService)
        # Check that it has the expected model in settings
        assert stt_service._settings["model"] == "nova-2"

@pytest.mark.asyncio
async def test_audio_frame_creation():
    """Test creating Pipecat audio frames from raw bytes"""
    from src.audio_pipeline import AudioFrameProcessor
    from pipecat.frames.frames import AudioRawFrame
    
    processor = AudioFrameProcessor()
    
    # Create fake audio data
    audio_bytes = bytes(3200)  # 0.1 seconds at 16kHz
    
    # Create frame
    frame = processor.create_audio_frame(audio_bytes)
    
    assert isinstance(frame, AudioRawFrame)
    assert frame.audio == audio_bytes
    assert frame.sample_rate == 16000
    assert frame.num_channels == 1

@pytest.mark.asyncio
async def test_audio_pipeline_handler():
    """Test that we can create a simple audio pipeline handler"""
    from src.audio_pipeline import AudioPipelineHandler
    
    # Create handler
    handler = AudioPipelineHandler()
    
    # Process some audio
    audio_bytes = bytes(3200)  # 0.1 seconds
    
    # This should not raise an exception
    await handler.process_audio(audio_bytes)
    
    # Handler should be in ready state
    assert handler.is_ready()

@pytest.mark.asyncio
async def test_audio_pipeline_with_stt_service():
    """Test that AudioPipelineHandler can be initialized with STT service"""
    with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
        from src.audio_pipeline import AudioPipelineHandler, create_stt_service
        
        # Create STT service
        stt_service = create_stt_service("deepgram")
        
        # Create handler with STT
        handler = AudioPipelineHandler(stt_service=stt_service)
        
        assert handler.stt_service is not None
        assert handler.is_ready()

@pytest.mark.asyncio
async def test_audio_pipeline_processes_frames_through_stt():
    """Test that audio is buffered for STT processing"""
    with patch.dict('os.environ', {'DEEPGRAM_API_KEY': 'test-key'}):
        from src.audio_pipeline import AudioPipelineHandler
        
        # Create mock STT service
        mock_stt = AsyncMock()
        
        # Create handler with mock STT
        handler = AudioPipelineHandler(stt_service=mock_stt)
        
        # Process audio
        audio_bytes = bytes(3200)
        await handler.process_audio(audio_bytes)
        
        # Verify audio was received
        assert handler.has_received_audio()
        assert len(handler._audio_buffer) == 3200