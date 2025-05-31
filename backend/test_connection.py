# backend/test_connection.py
"""
Simple test script to verify WebSocket connection and audio processing
"""
import asyncio
import websockets
import json
import numpy as np
import struct

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")
            
            # Listen for messages
            async def listen():
                try:
                    async for message in websocket:
                        if isinstance(message, bytes):
                            print(f"📦 Received binary data: {len(message)} bytes")
                        else:
                            print(f"📨 Received text: {message}")
                except Exception as e:
                    print(f"❌ Listen error: {e}")
            
            # Send test audio
            async def send_audio():
                try:
                    print("🎤 Sending test audio...")
                    # Generate 1 second of silence
                    sample_rate = 16000
                    duration = 1
                    samples = np.zeros(sample_rate * duration, dtype=np.int16)
                    
                    # Create a simple protobuf frame
                    frame_type = 1  # AUDIO_RAW
                    frame_id = 0
                    audio_bytes = samples.tobytes()
                    
                    # Pack the frame
                    frame = struct.pack('<BII', frame_type, frame_id, len(audio_bytes)) + audio_bytes
                    
                    await websocket.send(frame)
                    print(f"✅ Sent {len(frame)} bytes")
                    
                except Exception as e:
                    print(f"❌ Send error: {e}")
            
            # Run both tasks
            listen_task = asyncio.create_task(listen())
            
            # Send some audio
            await send_audio()
            
            # Wait a bit for responses
            await asyncio.sleep(5)
            
            print("🔌 Closing connection...")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the backend is running on http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(test_websocket())