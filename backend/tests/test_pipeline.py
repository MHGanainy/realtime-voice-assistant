import pytest
from unittest.mock import Mock, AsyncMock, patch
import os

@pytest.mark.asyncio
async def test_pipeline_runner_initialization():
    """Test that we can initialize a pipeline runner"""
    from pipecat.pipeline.runner import PipelineRunner
    
    runner = PipelineRunner()
    assert runner is not None

@pytest.mark.asyncio
async def test_create_basic_pipeline():
    """Test creating a basic audio pipeline"""
    with patch.dict('os.environ', {
        'DEEPGRAM_API_KEY': 'test-key',
        'OPENAI_API_KEY': 'test-key',
        'ELEVEN_API_KEY': 'test-key'
    }):
        from src.pipeline_manager import create_audio_pipeline
        from pipecat.pipeline.pipeline import Pipeline
        
        # Create mock services
        mock_stt = AsyncMock()
        mock_llm = AsyncMock()
        mock_tts = AsyncMock()
        
        pipeline = create_audio_pipeline(
            stt_service=mock_stt,
            llm_service=mock_llm,
            tts_service=mock_tts
        )
        
        assert isinstance(pipeline, Pipeline)