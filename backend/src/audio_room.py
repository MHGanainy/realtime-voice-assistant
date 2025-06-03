"""
Audio Room - Core abstraction for voice conversations
A "Room" represents a voice conversation space where participants can talk
"""
import asyncio
import uuid
import struct
from typing import Dict, Set, Optional
from datetime import datetime
import logging

from fastapi import WebSocket
import json

from .audio_codec import AudioProcessor, AudioFrame

logger = logging.getLogger(__name__)


class AudioRoom:
    """
    A voice conversation room - like a conference call or voice channel
    Handles audio routing, participant management, and conversation state
    """
    
    def __init__(self, room_id: Optional[str] = None):
        self.room_id = room_id or str(uuid.uuid4())
        self.participants: Dict[str, 'Participant'] = {}
        self.created_at = datetime.utcnow()
        self._lock = asyncio.Lock()
        self.audio_processor = AudioProcessor()  # Add Opus processing
        
    @property
    def participant_count(self) -> int:
        """Number of active participants in the room"""
        return len(self.participants)
    
    async def add_participant(self, participant: 'Participant') -> None:
        """Add a participant to the room"""
        async with self._lock:
            self.participants[participant.user_id] = participant
            logger.info(f"Participant {participant.user_id} joined room {self.room_id}")
            
            # Notify other participants
            await self._broadcast_event({
                "type": "participant_joined",
                "userId": participant.user_id,
                "timestamp": datetime.utcnow().isoformat()
            }, exclude=participant.user_id)
    
    async def remove_participant(self, user_id: str) -> None:
        """Remove a participant from the room"""
        async with self._lock:
            if user_id in self.participants:
                del self.participants[user_id]
                logger.info(f"Participant {user_id} left room {self.room_id}")
                
                # Notify remaining participants
                await self._broadcast_event({
                    "type": "participant_left",
                    "userId": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    async def process_audio_chunk(self, chunk: bytes, from_user: str) -> bytes:
        """
        Process incoming audio chunk with Opus compression
        
        For now, this implements a simple echo with compression:
        1. If we receive raw PCM, we encode it to Opus and return the compressed frame
        2. If we receive a compressed frame, we decode it and return raw PCM
        """
        try:
            # Get current timestamp
            timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
            
            # First, check if this looks like one of our serialized frames
            # Our frames have a very specific structure that's unlikely to occur randomly
            is_frame = False
            
            if len(chunk) >= 12:  # Minimum frame size (8 byte header + some data)
                try:
                    # Try to parse header
                    seq_num = struct.unpack('>I', chunk[0:4])[0]
                    ts = struct.unpack('>I', chunk[4:8])[0]
                    
                    # Our timestamps are milliseconds since epoch (truncated to 32 bits)
                    # Current time in ms is ~1748892963000, but we modulo to fit in 32 bits
                    # So valid timestamps are 0 to 2^32-1
                    current_time_32bit = int(datetime.utcnow().timestamp() * 1000) % (2**32)
                    
                    # Check if this could be a valid frame
                    # Sequence numbers should be small (start at 0)
                    # Timestamps should be reasonable (not too far from current time)
                    time_diff = abs(int(ts) - current_time_32bit)
                    
                    if (0 <= seq_num < 100000 and  # Reasonable sequence number
                        (time_diff < 300000 or time_diff > 4290000000)):  # Within 5 minutes or wrapped
                        # This looks like a valid frame
                        is_frame = True
                        logger.debug(f"Detected frame: seq={seq_num}, ts={ts}")
                except:
                    is_frame = False
            
            if is_frame:
                # This is a serialized frame with Opus data
                logger.debug(f"Processing incoming Opus frame")
                frame = AudioFrame.deserialize(chunk)
                # Decode from Opus to PCM
                pcm_data = self.audio_processor.process_incoming_audio(frame)
                # Return raw PCM
                return pcm_data
            else:
                # This is raw PCM data - encode it
                logger.debug(f"Processing incoming raw PCM: {len(chunk)} bytes")
                frame = self.audio_processor.process_outgoing_audio(chunk, timestamp_ms)
                logger.debug(f"Created Opus frame: seq={frame.sequence_number}, size={len(frame.data)} bytes")
                # Return the serialized frame with Opus data
                serialized = frame.serialize()
                logger.debug(f"Returning serialized frame: {len(serialized)} bytes")
                return serialized
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            # Return original chunk on error
            return chunk
    
    async def broadcast_audio(self, audio: bytes, from_user: Optional[str] = None) -> None:
        """Send audio to all participants except sender"""
        tasks = []
        for user_id, participant in self.participants.items():
            if user_id != from_user:
                tasks.append(participant.send_audio(audio))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _broadcast_event(self, event: dict, exclude: Optional[str] = None) -> None:
        """Broadcast JSON event to all participants"""
        tasks = []
        for user_id, participant in self.participants.items():
            if user_id != exclude:
                tasks.append(participant.send_json(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class Participant:
    """Represents a participant in a voice room"""
    
    def __init__(self, user_id: str, websocket: WebSocket):
        self.user_id = user_id
        self.websocket = websocket
        self.joined_at = datetime.utcnow()
        self.is_speaking = False
        self.metrics = {
            "packets_sent": 0,
            "packets_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0
        }
    
    async def send_audio(self, audio: bytes) -> None:
        """Send audio data to this participant"""
        try:
            await self.websocket.send_bytes(audio)
            self.metrics["packets_sent"] += 1
            self.metrics["bytes_sent"] += len(audio)
        except Exception as e:
            logger.error(f"Failed to send audio to {self.user_id}: {e}")
    
    async def send_json(self, data: dict) -> None:
        """Send JSON message to this participant"""
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send JSON to {self.user_id}: {e}")


class RoomManager:
    """Manages all active voice rooms"""
    
    def __init__(self):
        self.rooms: Dict[str, AudioRoom] = {}
        self._lock = asyncio.Lock()
    
    def create_room(self, room_id: Optional[str] = None) -> str:
        """Create a new room and return its ID"""
        room = AudioRoom(room_id)
        self.rooms[room.room_id] = room
        logger.info(f"Created room {room.room_id}")
        return room.room_id
    
    def get_room(self, room_id: str) -> Optional[AudioRoom]:
        """Get a room by ID"""
        return self.rooms.get(room_id)
    
    def get_or_create_room(self, room_id: str) -> AudioRoom:
        """Get existing room or create new one"""
        if room_id not in self.rooms:
            self.create_room(room_id)
        return self.rooms[room_id]
    
    async def remove_room_if_empty(self, room_id: str) -> None:
        """Remove room if it has no participants"""
        async with self._lock:
            room = self.rooms.get(room_id)
            if room and room.participant_count == 0:
                del self.rooms[room_id]
                logger.info(f"Removed empty room {room_id}")


# Global room manager instance
room_manager = RoomManager()