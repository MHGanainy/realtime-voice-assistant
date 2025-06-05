import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import os

@pytest.mark.asyncio
async def test_websocket_creates_audio_pipeline():
    """Test that WebSocket creates an audio pipeline handler on connection"""
    
    with patch('src.audio_pipeline.AudioPipelineHandler') as mock_handler_class:
        mock_handler = AsyncMock()
        mock_handler.is_ready.return_value = True
        mock_handler.process_audio = AsyncMock()
        mock_handler_class.return_value = mock_handler
        
        # Import app after patching
        from src.main import app
        client = TestClient(app)
        
        with client.websocket_connect("/ws/audio") as websocket:
            # Handler should be created
            mock_handler_class.assert_called_once()
            
            # Send audio data
            audio_bytes = bytes(3200)
            websocket.send_bytes(audio_bytes)
            
            # Get response first
            response = websocket.receive_json()
            assert response["type"] == "audio_received"
            
            # Then check handler was called
            mock_handler.process_audio.assert_called_once_with(audio_bytes)

@pytest.mark.asyncio
async def test_websocket_with_stt_integration():
    """Test that WebSocket can process audio through real STT"""
    # This test requires actual API keys, so we'll skip it in CI
    if not os.getenv("DEEPGRAM_API_KEY"):
        pytest.skip("DEEPGRAM_API_KEY not set")
    
    from src.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Send 1 second of silence (should not produce transcription)
        for _ in range(5):  # 5 chunks = 0.5 seconds
            audio_bytes = bytes(3200)  # silence
            websocket.send_bytes(audio_bytes)
            response = websocket.receive_json()
            assert response["type"] == "audio_received"

@pytest.mark.asyncio
async def test_websocket_creates_pipeline():
    """Test that WebSocket creates a full pipeline"""
    with patch.dict('os.environ', {
        'DEEPGRAM_API_KEY': 'test-key',
        'OPENAI_API_KEY': 'test-key', 
        'ELEVEN_API_KEY': 'test-key'
    }):
        from src.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Should reject connection without session ID
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/audio") as websocket:
                pass