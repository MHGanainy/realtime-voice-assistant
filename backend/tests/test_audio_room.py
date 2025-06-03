import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock
from fastapi.testclient import TestClient
from fastapi import WebSocket
import json

# Import from src package
from src.audio_room import AudioRoom, RoomManager, Participant
from src.main import app


class TestAudioRoom:
    """Test suite for AudioRoom - our voice conversation space"""
    
    def test_room_creation(self):
        """Test that we can create a new audio room with unique ID"""
        room = AudioRoom()
        assert room.room_id is not None
        assert isinstance(room.room_id, str)
        assert len(room.room_id) > 0
    
    def test_room_has_participant_tracking(self):
        """Test that room can track who's in the conversation"""
        room = AudioRoom()
        assert hasattr(room, 'participants')
        assert room.participant_count == 0
    
    @pytest.mark.asyncio
    async def test_room_handles_audio_chunks(self):
        """Test that room can receive and process audio chunks"""
        room = AudioRoom()
        
        # Simulate incoming audio chunk (Opus encoded)
        audio_chunk = b'\x00\x01\x02\x03'  # Mock Opus data
        
        # Room should be able to process audio
        result = await room.process_audio_chunk(audio_chunk, from_user="test-user")
        assert result is not None


class TestRoomManager:
    """Test suite for managing multiple voice rooms"""
    
    def test_manager_creates_rooms(self):
        """Test that manager can create and track rooms"""
        manager = RoomManager()
        room_id = manager.create_room()
        
        assert room_id is not None
        assert manager.get_room(room_id) is not None
    
    @pytest.mark.asyncio
    async def test_manager_removes_empty_rooms(self):
        """Test that manager cleans up empty rooms"""
        manager = RoomManager()
        room_id = manager.create_room()
        
        # Simulate participant leaving
        await manager.remove_room_if_empty(room_id)
        
        assert manager.get_room(room_id) is None


class TestWebSocketAudioFlow:
    """Test the actual WebSocket connection for audio streaming"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_websocket_connection(self, client):
        """Test that we can establish WebSocket connection to a room"""
        with client.websocket_connect("/ws/room/test-room") as websocket:
            # Send initial handshake
            websocket.send_json({"type": "join", "userId": "test-user"})
            
            # Should receive acknowledgment
            data = websocket.receive_json()
            assert data["type"] == "joined"
            assert data["roomId"] == "test-room"
    
    def test_websocket_audio_echo(self, client):
            """Test that audio sent to backend is processed and returned as Opus frame"""
            with client.websocket_connect("/ws/room/test-room") as websocket:
                # Join room first
                websocket.send_json({"type": "join", "userId": "test-user"})
                websocket.receive_json()  # Consume join response
                
                # Send audio chunk
                audio_data = b'\x00\x01\x02\x03'  # Mock Opus audio
                websocket.send_bytes(audio_data)
                
                # Should receive an Opus frame back
                received = websocket.receive_bytes()
                
                # Check that we received a frame (should be larger than original due to header)
                assert len(received) >= 8  # At least header size
                
                # Parse the frame header
                import struct
                seq_num = struct.unpack('>I', received[0:4])[0]
                timestamp = struct.unpack('>I', received[4:8])[0]
                
                # Verify it's a valid frame
                assert seq_num >= 0  # Should be a valid sequence number
                assert timestamp > 0  # Should have a timestamp
                
                # The frame should contain compressed audio data
                opus_data = received[8:]
                assert len(opus_data) > 0  # Should have some audio data
    
    def test_websocket_handles_participant_events(self, client):
        """Test that room broadcasts participant join/leave events"""
        with client.websocket_connect("/ws/room/test-room") as ws1:
            ws1.send_json({"type": "join", "userId": "user1"})
            ws1.receive_json()  # Consume own join
            
            # Second participant joins
            with client.websocket_connect("/ws/room/test-room") as ws2:
                ws2.send_json({"type": "join", "userId": "user2"})
                ws2.receive_json()  # Consume own join
                
                # First participant should be notified
                notification = ws1.receive_json()
                assert notification["type"] == "participant_joined"
                assert notification["userId"] == "user2"

    def test_websocket_handles_participant_events(self, client):
        """Test that room broadcasts participant join/leave events"""
        with client.websocket_connect("/ws/room/test-room") as ws1:
            ws1.send_json({"type": "join", "userId": "user1"})
            ws1.receive_json()  # Consume own join
            
            # Second participant joins
            with client.websocket_connect("/ws/room/test-room") as ws2:
                ws2.send_json({"type": "join", "userId": "user2"})
                ws2.receive_json()  # Consume own join
                
                # First participant should be notified
                notification = ws1.receive_json()
                assert notification["type"] == "participant_joined"
                assert notification["userId"] == "user2"
    
    def test_websocket_audio_compression(self, client):
        """Test that audio is properly compressed with Opus"""
        with client.websocket_connect("/ws/room/test-room") as websocket:
            # Join room first
            websocket.send_json({"type": "join", "userId": "test-user"})
            websocket.receive_json()  # Consume join response
            
            # Send a proper PCM audio chunk (20ms at 48kHz)
            import numpy as np
            samples = np.zeros(960, dtype=np.int16)  # Silence
            pcm_data = samples.tobytes()
            assert len(pcm_data) == 1920  # 960 samples * 2 bytes
            
            websocket.send_bytes(pcm_data)
            
            # Should receive a compressed Opus frame
            received = websocket.receive_bytes()
            
            # Verify compression happened
            assert len(received) < len(pcm_data)  # Should be compressed
            assert len(received) < 100  # Silence compresses to very small size
            
            # Verify frame structure
            import struct
            seq_num = struct.unpack('>I', received[0:4])[0]
            timestamp = struct.unpack('>I', received[4:8])[0]
            
            assert seq_num == 0  # First frame
            assert timestamp > 0  # Valid timestamp
            
            # Opus data should be very small for silence
            opus_data = received[8:]
            assert len(opus_data) < 20  # Silence compresses to ~3 bytes

class TestAudioConfiguration:
    """Test that our audio configuration matches the spec"""
    
    def test_opus_encoder_config(self):
        """Test Opus encoder is configured per spec: 48kHz, 20ms frames"""
        from src.audio_config import OpusConfig
        
        config = OpusConfig()
        assert config.sample_rate == 48000
        assert config.frame_duration_ms == 20
        assert config.bitrate_mode == "voip"  # Optimized for voice
        assert config.complexity == 5  # Balance quality/latency
    
    def test_audio_buffer_config(self):
        """Test audio buffer is configured for low latency"""
        from src.audio_config import BufferConfig
        
        config = BufferConfig()
        assert config.chunk_duration_ms == 20  # Match Opus frames
        assert config.max_buffer_ms == 120  # ~6 frames buffered max