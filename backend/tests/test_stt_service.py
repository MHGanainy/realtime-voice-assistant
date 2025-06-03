"""
Unit tests for STT (Speech-to-Text) service using Deepgram
Tests the behavior we expect from our STT integration
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

from src.stt_service import STTService, DeepgramSTTService, STTConfig
from src.audio_room import AudioFrame


class TestSTTConfig:
    """Test STT configuration"""
    
    def test_default_config(self):
        """Test default STT configuration values"""
        config = STTConfig()
        
        assert config.provider == "deepgram"
        assert config.model == "nova-2"
        assert config.language == "en"
        assert config.sample_rate == 48000
        assert config.channels == 1
        assert config.encoding == "linear16"
        assert config.interim_results == True
        assert config.punctuate == True
        assert config.endpointing == 300  # ms
        assert config.vad_events == True
        assert config.utterance_end_ms == 1000
    
    def test_custom_config(self):
        """Test custom STT configuration"""
        config = STTConfig(
            model="nova-2-phonecall",
            language="es",
            interim_results=False
        )
        
        assert config.model == "nova-2-phonecall"
        assert config.language == "es"
        assert config.interim_results == False


class TestDeepgramSTTService:
    """Test Deepgram STT service implementation"""
    
    @pytest.fixture
    def mock_deepgram(self):
        """Mock Deepgram client and dependencies"""
        with patch('src.stt_service.DeepgramClient') as mock_client:
            # Mock the WebSocket connection
            mock_ws = AsyncMock()
            mock_ws.send = AsyncMock()
            mock_ws.recv = AsyncMock()
            mock_ws.close = AsyncMock()
            
            # Mock the live transcription client
            mock_live = MagicMock()
            mock_live.connect = AsyncMock(return_value=mock_ws)
            
            # Configure mock client
            mock_client_instance = MagicMock()
            mock_client_instance.listen.live = mock_live
            mock_client.return_value = mock_client_instance
            
            yield {
                'client': mock_client,
                'instance': mock_client_instance,
                'live': mock_live,
                'ws': mock_ws
            }
    
    @pytest.fixture
    def stt_service(self, mock_deepgram):
        """Create STT service with mocked dependencies"""
        service = DeepgramSTTService(api_key="test-key")
        return service
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_deepgram):
        """Test STT service initialization"""
        service = DeepgramSTTService(api_key="test-key")
        
        assert service.api_key == "test-key"
        assert service.config.provider == "deepgram"
        assert service._connection is None
        assert service._is_connected == False
        
        # Verify Deepgram client was created
        mock_deepgram['client'].assert_called_once_with("test-key")
    
    @pytest.mark.asyncio
    async def test_connect(self, stt_service, mock_deepgram):
        """Test connecting to Deepgram"""
        # Setup mock connection
        mock_deepgram['ws'].recv.return_value = json.dumps({
            "type": "Metadata",
            "transaction_key": "test-transaction",
            "request_id": "test-request",
            "sha256": "test-sha",
            "channels": 1,
            "created": "2024-01-01T00:00:00Z"
        })
        
        await stt_service.connect()
        
        # Verify connection was established
        assert stt_service._is_connected == True
        assert stt_service._connection == mock_deepgram['ws']
        
        # Verify connection options
        mock_deepgram['live'].connect.assert_called_once()
        call_args = mock_deepgram['live'].connect.call_args[0][0]
        
        assert call_args['model'] == "nova-2"
        assert call_args['language'] == "en"
        assert call_args['punctuate'] == True
        assert call_args['interim_results'] == True
        assert call_args['endpointing'] == 300
        assert call_args['vad_events'] == True
        assert call_args['utterance_end_ms'] == 1000
        assert call_args['sample_rate'] == 48000
        assert call_args['channels'] == 1
        assert call_args['encoding'] == "linear16"
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk(self, stt_service, mock_deepgram):
        """Test processing audio chunks"""
        await stt_service.connect()
        
        # Create test audio frame
        audio_data = b'\x00\x01' * 960  # 20ms of audio
        frame = AudioFrame(
            data=audio_data,
            timestamp_ms=1000,
            sequence_number=1
        )
        
        # Process the audio
        await stt_service.process_audio(frame)
        
        # Verify audio was sent to Deepgram
        mock_deepgram['ws'].send.assert_called_once_with(audio_data)
    
    @pytest.mark.asyncio
    async def test_handle_partial_transcript(self, stt_service, mock_deepgram):
        """Test handling partial transcripts"""
        await stt_service.connect()
        
        # Track emitted transcripts
        transcripts = []
        stt_service.on('partial_transcript', lambda t: transcripts.append(t))
        
        # Simulate receiving partial transcript
        partial_message = {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": 1.2,
            "start": 0.0,
            "is_final": False,
            "speech_final": False,
            "channel": {
                "alternatives": [{
                    "transcript": "hello world",
                    "confidence": 0.95,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.96},
                        {"word": "world", "start": 0.5, "end": 1.2, "confidence": 0.94}
                    ]
                }]
            }
        }
        
        mock_deepgram['ws'].recv.return_value = json.dumps(partial_message)
        
        # Start listening (which processes messages)
        listen_task = asyncio.create_task(stt_service._listen())
        await asyncio.sleep(0.1)  # Let it process one message
        listen_task.cancel()
        
        # Verify partial transcript was emitted
        assert len(transcripts) == 1
        assert transcripts[0]['text'] == "hello world"
        assert transcripts[0]['is_final'] == False
        assert transcripts[0]['confidence'] == 0.95
        assert len(transcripts[0]['words']) == 2
    
    @pytest.mark.asyncio
    async def test_handle_final_transcript(self, stt_service, mock_deepgram):
        """Test handling final transcripts"""
        await stt_service.connect()
        
        # Track emitted transcripts
        finals = []
        stt_service.on('final_transcript', lambda t: finals.append(t))
        
        # Simulate receiving final transcript
        final_message = {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": 2.5,
            "start": 0.0,
            "is_final": True,
            "speech_final": True,
            "channel": {
                "alternatives": [{
                    "transcript": "hello world how are you",
                    "confidence": 0.98,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
                        {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.98},
                        {"word": "how", "start": 1.2, "end": 1.5, "confidence": 0.97},
                        {"word": "are", "start": 1.5, "end": 1.8, "confidence": 0.98},
                        {"word": "you", "start": 1.8, "end": 2.5, "confidence": 0.99}
                    ]
                }]
            }
        }
        
        mock_deepgram['ws'].recv.return_value = json.dumps(final_message)
        
        # Start listening
        listen_task = asyncio.create_task(stt_service._listen())
        await asyncio.sleep(0.1)
        listen_task.cancel()
        
        # Verify final transcript was emitted
        assert len(finals) == 1
        assert finals[0]['text'] == "hello world how are you"
        assert finals[0]['is_final'] == True
        assert finals[0]['confidence'] == 0.98
        assert finals[0]['duration'] == 2.5
    
    @pytest.mark.asyncio
    async def test_handle_utterance_end(self, stt_service, mock_deepgram):
        """Test handling utterance end events"""
        await stt_service.connect()
        
        # Track utterance end events
        utterance_ends = []
        stt_service.on('utterance_end', lambda e: utterance_ends.append(e))
        
        # Simulate utterance end message
        utterance_message = {
            "type": "UtteranceEnd",
            "channel": [0, 1],
            "last_word_end": 2.5
        }
        
        mock_deepgram['ws'].recv.return_value = json.dumps(utterance_message)
        
        # Start listening
        listen_task = asyncio.create_task(stt_service._listen())
        await asyncio.sleep(0.1)
        listen_task.cancel()
        
        # Verify utterance end was emitted
        assert len(utterance_ends) == 1
        assert utterance_ends[0]['last_word_end'] == 2.5
    
    @pytest.mark.asyncio
    async def test_handle_speech_started(self, stt_service, mock_deepgram):
        """Test handling speech started events (VAD)"""
        await stt_service.connect()
        
        # Track speech events
        speech_events = []
        stt_service.on('speech_started', lambda e: speech_events.append(e))
        
        # Simulate speech started message
        speech_message = {
            "type": "SpeechStarted",
            "channel": [0, 1],
            "timestamp": 1.234
        }
        
        mock_deepgram['ws'].recv.return_value = json.dumps(speech_message)
        
        # Start listening
        listen_task = asyncio.create_task(stt_service._listen())
        await asyncio.sleep(0.1)
        listen_task.cancel()
        
        # Verify speech started was emitted
        assert len(speech_events) == 1
        assert speech_events[0]['timestamp'] == 1.234
    
    @pytest.mark.asyncio
    async def test_error_handling(self, stt_service, mock_deepgram):
        """Test error handling in STT service"""
        await stt_service.connect()
        
        # Track errors
        errors = []
        stt_service.on('error', lambda e: errors.append(e))
        
        # Simulate error message
        error_message = {
            "type": "Error",
            "error": "Invalid audio format",
            "code": "INVALID_AUDIO",
            "description": "Audio must be 16-bit PCM"
        }
        
        mock_deepgram['ws'].recv.return_value = json.dumps(error_message)
        
        # Start listening
        listen_task = asyncio.create_task(stt_service._listen())
        await asyncio.sleep(0.1)
        listen_task.cancel()
        
        # Verify error was emitted
        assert len(errors) == 1
        assert errors[0]['error'] == "Invalid audio format"
        assert errors[0]['code'] == "INVALID_AUDIO"
    
    @pytest.mark.asyncio
    async def test_disconnect(self, stt_service, mock_deepgram):
        """Test disconnecting from Deepgram"""
        await stt_service.connect()
        assert stt_service._is_connected == True
        
        await stt_service.disconnect()
        
        # Verify disconnection
        assert stt_service._is_connected == False
        assert stt_service._connection is None
        mock_deepgram['ws'].close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reconnection_on_error(self, stt_service, mock_deepgram):
        """Test automatic reconnection on connection error"""
        await stt_service.connect()
        
        # Simulate connection error
        mock_deepgram['ws'].recv.side_effect = Exception("Connection lost")
        
        # Track reconnection attempts
        reconnections = []
        stt_service.on('reconnecting', lambda: reconnections.append(True))
        
        # Start listening (will encounter error)
        listen_task = asyncio.create_task(stt_service._listen())
        await asyncio.sleep(0.1)
        
        # Should attempt to reconnect
        assert len(reconnections) > 0
        
        listen_task.cancel()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, stt_service, mock_deepgram):
        """Test STT metrics collection"""
        await stt_service.connect()
        
        # Process some audio
        audio_data = b'\x00\x01' * 960
        frame = AudioFrame(audio_data, 1000, 1)
        await stt_service.process_audio(frame)
        
        # Get metrics
        metrics = stt_service.get_metrics()
        
        assert metrics['provider'] == "deepgram"
        assert metrics['model'] == "nova-2"
        assert metrics['connected'] == True
        assert metrics['frames_processed'] == 1
        assert metrics['bytes_processed'] == len(audio_data)
        assert 'connection_uptime' in metrics
        assert 'last_transcript_time' in metrics


class TestSTTServiceIntegration:
    """Integration tests for STT service with audio room"""
    
    @pytest.mark.asyncio
    async def test_stt_integration_with_audio_room(self):
        """Test STT service integration with audio room"""
        from src.audio_room import AudioRoom, Participant
        
        # Create room with STT
        room = AudioRoom(room_id="test-room")
        
        # Mock STT service
        mock_stt = AsyncMock(spec=STTService)
        mock_stt.connect = AsyncMock()
        mock_stt.process_audio = AsyncMock()
        mock_stt.disconnect = AsyncMock()
        
        # Attach STT to room
        room.stt_service = mock_stt
        
        # Add participant
        mock_ws = AsyncMock()
        participant = Participant("user1", mock_ws)
        await room.add_participant(participant)
        
        # Verify STT was connected
        mock_stt.connect.assert_called_once()
        
        # Process audio chunk
        audio_chunk = b'\x00\x01' * 960
        processed = await room.process_audio_chunk(audio_chunk, "user1")
        
        # Verify STT processed the audio
        # Note: process_audio_chunk should be modified to call STT
        # This is what we expect after implementation
        
        # Remove participant
        await room.remove_participant("user1")
        
        # Verify STT was disconnected when room is empty
        mock_stt.disconnect.assert_called_once()