import pytest
from fastapi.testclient import TestClient

def test_websocket_endpoint_exists():
    """Test that the WebSocket endpoint /ws/audio exists"""
    from src.main import app
    
    client = TestClient(app)
    
    # This should not raise an exception
    with client.websocket_connect("/ws/audio") as websocket:
        # Just connect and disconnect
        pass

def test_websocket_receives_binary_audio_data():
    """Test that the WebSocket can receive binary PCM audio data"""
    from src.main import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Create fake PCM audio data (16kHz, 16-bit = 2 bytes per sample)
        # 0.1 seconds of silence = 1600 samples * 2 bytes = 3200 bytes
        fake_audio = bytes(3200)
        
        # Send binary data
        websocket.send_bytes(fake_audio)
        
        # Try to send more data - this should work if connection is still open
        websocket.send_bytes(fake_audio)
        
        # Send a close message to gracefully shutdown
        websocket.close()

def test_websocket_echoes_received_data():
    """Test that the WebSocket echoes back the audio data size for confirmation"""
    from src.main import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Send some audio data
        audio_size = 3200
        fake_audio = bytes(audio_size)
        websocket.send_bytes(fake_audio)
        
        # Expect a response with the number of bytes received
        response = websocket.receive_json()
        assert response["type"] == "audio_received"
        assert response["bytes"] == audio_size

def test_websocket_validates_audio_format():
    """Test that the WebSocket validates audio chunk sizes"""
    from src.main import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Send audio data that's not a multiple of 2 bytes (16-bit samples)
        invalid_audio = bytes(3201)  # Odd number of bytes
        websocket.send_bytes(invalid_audio)
        
        # Expect an error response
        response = websocket.receive_json()
        assert response["type"] == "error"
        assert "invalid audio format" in response["message"].lower()

def test_websocket_handles_streaming_audio_chunks():
    """Test that the WebSocket can handle multiple audio chunks in sequence"""
    from src.main import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Simulate streaming audio: 5 chunks of 0.1 seconds each
        chunk_size = 3200  # 0.1 seconds at 16kHz, 16-bit
        num_chunks = 5
        
        responses_received = []
        
        for i in range(num_chunks):
            # Send a chunk
            fake_audio = bytes(chunk_size)
            websocket.send_bytes(fake_audio)
            
            # Get confirmation
            response = websocket.receive_json()
            responses_received.append(response)
        
        # Verify we received all confirmations
        assert len(responses_received) == num_chunks
        assert all(r["type"] == "audio_received" for r in responses_received)
        assert all(r["bytes"] == chunk_size for r in responses_received)