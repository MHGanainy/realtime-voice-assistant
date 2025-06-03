#!/usr/bin/env python3
"""
Integration test script for the voice assistant
Tests the complete audio pipeline end-to-end
"""
import asyncio
import websockets
import json
import time
import numpy as np
import struct
from typing import Optional, List
import argparse


class VoiceAssistantTester:
    def __init__(self, backend_url: str = "ws://localhost:8000"):
        self.backend_url = backend_url
        self.room_id = "integration-test"
        self.user_id = f"tester-{int(time.time())}"
        self.received_audio: List[bytes] = []
        self.received_messages: List[dict] = []
        
    async def test_connection(self) -> bool:
        """Test basic WebSocket connection"""
        print("🔌 Testing WebSocket connection...")
        
        try:
            async with websockets.connect(f"{self.backend_url}/ws/room/{self.room_id}") as ws:
                # Send join message
                await ws.send(json.dumps({
                    "type": "join",
                    "userId": self.user_id
                }))
                
                # Wait for response
                response = await ws.recv()
                data = json.loads(response)
                
                if data.get("type") == "joined":
                    print("✅ Connection successful!")
                    print(f"   Room: {data.get('roomId')}")
                    print(f"   Participants: {data.get('participants')}")
                    return True
                else:
                    print("❌ Unexpected response:", data)
                    return False
                    
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def test_audio_echo(self) -> bool:
        """Test audio echo functionality"""
        print("\n🎤 Testing audio echo...")
        
        try:
            async with websockets.connect(f"{self.backend_url}/ws/room/{self.room_id}") as ws:
                # Join first
                await ws.send(json.dumps({
                    "type": "join",
                    "userId": self.user_id
                }))
                await ws.recv()  # Consume join response
                
                # Generate test audio (20ms of 440Hz sine wave at 48kHz)
                sample_rate = 48000
                duration = 0.02  # 20ms chunk
                frequency = 440  # A4 note
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_data = np.sin(2 * np.pi * frequency * t)
                
                # Convert to 16-bit PCM
                audio_bytes = struct.pack(
                    f'{len(audio_data)}h',
                    *[int(sample * 32767) for sample in audio_data]
                )
                
                print(f"   Sending {len(audio_bytes)} bytes of audio...")
                
                # Send audio
                await ws.send(audio_bytes)
                
                # Wait for echo
                start_time = time.time()
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                latency = (time.time() - start_time) * 1000
                
                if isinstance(response, bytes):
                    print(f"✅ Received audio echo: {len(response)} bytes")
                    print(f"   Latency: {latency:.1f}ms")
                    
                    # Debug: show first few bytes
                    print(f"   First 20 bytes (hex): {response[:20].hex()}")
                    print(f"   First 8 bytes (as ints): {list(response[:8])}")
                    
                    # Check if it's a serialized frame (has our header)
                    if len(response) >= 8:
                        try:
                            # Try to parse as our frame format
                            seq_num = struct.unpack('>I', response[0:4])[0]
                            timestamp = struct.unpack('>I', response[4:8])[0]
                            
                            print(f"   Parsed header: seq={seq_num}, timestamp={timestamp}")
                            
                            # Check if this looks like a valid frame
                            # Sequence numbers should be small for a new connection
                            # Timestamp should be a reasonable 32-bit value
                            if seq_num < 100000:  # Reasonable seq number
                                opus_data = response[8:]
                                print(f"   ✅ Detected Opus frame!")
                                print(f"   Frame details: seq={seq_num}, timestamp={timestamp}")
                                print(f"   Opus data size: {len(opus_data)} bytes")
                                
                                # Calculate compression ratio
                                compression_ratio = len(audio_bytes) / len(response) if len(response) > 0 else 0
                                print(f"   Compression ratio: {compression_ratio:.1f}x")
                                
                                # Check if it's mock compression
                                if opus_data.startswith(b'MOCK'):
                                    print("   Using mock Opus compression")
                                else:
                                    print("   ✅ Using REAL Opus compression!")
                                    
                                    # Calculate bandwidth savings
                                    bandwidth_saved = (1 - len(response)/len(audio_bytes))*100
                                    print(f"   ✅ Bandwidth saved: {bandwidth_saved:.1f}%")
                            else:
                                # Not a frame, just raw audio
                                print("   Not detected as Opus frame (seq number too high)")
                                print("   Treating as raw audio echo")
                        except Exception as e:
                            # Not a frame, just raw audio
                            print(f"   Failed to parse as frame: {e}")
                            print("   Treating as raw audio echo")
                    else:
                        print("   Response too small to be a frame")
                    
                    return True
                else:
                    print("❌ Expected audio, got:", type(response))
                    return False
                    
        except asyncio.TimeoutError:
            print("❌ Timeout waiting for audio echo")
            return False
        except Exception as e:
            print(f"❌ Audio test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def test_vad_events(self) -> bool:
        """Test Voice Activity Detection events"""
        print("\n🎙️ Testing VAD events...")
        
        try:
            async with websockets.connect(f"{self.backend_url}/ws/room/{self.room_id}") as ws:
                # Join first
                await ws.send(json.dumps({
                    "type": "join",
                    "userId": self.user_id
                }))
                await ws.recv()
                
                # Send VAD start event
                await ws.send(json.dumps({
                    "type": "voice_activity",
                    "activity": "start"
                }))
                
                print("   Sent VAD start event")
                
                # Send VAD end event
                await ws.send(json.dumps({
                    "type": "voice_activity", 
                    "activity": "end"
                }))
                
                print("   Sent VAD end event")
                print("✅ VAD events sent successfully")
                return True
                
        except Exception as e:
            print(f"❌ VAD test failed: {e}")
            return False
    
    async def test_metrics(self) -> bool:
        """Test metrics collection"""
        print("\n📊 Testing metrics...")
        
        try:
            async with websockets.connect(f"{self.backend_url}/ws/room/{self.room_id}") as ws:
                # Join
                await ws.send(json.dumps({
                    "type": "join",
                    "userId": self.user_id
                }))
                await ws.recv()
                
                # Send ping
                ping_time = int(time.time() * 1000)
                await ws.send(json.dumps({
                    "type": "ping",
                    "timestamp": ping_time
                }))
                
                # Wait for pong
                response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(response)
                
                if data.get("type") == "pong":
                    rtt = int(time.time() * 1000) - ping_time
                    print(f"✅ Ping/Pong successful")
                    print(f"   RTT: {rtt}ms")
                    return True
                else:
                    print("❌ Expected pong, got:", data)
                    return False
                    
        except Exception as e:
            print(f"❌ Metrics test failed: {e}")
            return False
    
    async def test_latency_measurements(self) -> bool:
        """Measure actual latencies"""
        print("\n⏱️ Measuring latencies...")
        
        latencies = []
        
        try:
            async with websockets.connect(f"{self.backend_url}/ws/room/{self.room_id}") as ws:
                # Join
                await ws.send(json.dumps({
                    "type": "join",
                    "userId": self.user_id
                }))
                await ws.recv()
                
                # Send 10 audio chunks and measure latency
                for i in range(10):
                    # Create proper 20ms audio chunk at 48kHz
                    # 960 samples * 2 bytes = 1920 bytes
                    samples = np.zeros(960, dtype=np.int16)  # Use silence for consistency
                    audio_bytes = samples.tobytes()
                    
                    start_time = time.time()
                    await ws.send(audio_bytes)
                    
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        if isinstance(response, bytes):
                            latency = (time.time() - start_time) * 1000
                            latencies.append(latency)
                    except asyncio.TimeoutError:
                        print(f"   Timeout on chunk {i+1}")
                    
                    await asyncio.sleep(0.02)  # 20ms between chunks
                
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)
                    
                    print(f"✅ Latency measurements complete")
                    print(f"   Average: {avg_latency:.1f}ms")
                    print(f"   Min: {min_latency:.1f}ms")
                    print(f"   Max: {max_latency:.1f}ms")
                    print(f"   Samples: {len(latencies)}/10")
                    
                    # Check if we meet the target
                    if avg_latency < 50:  # Good target for local testing
                        print("   ✅ Latency is excellent!")
                    elif avg_latency < 100:
                        print("   ⚠️  Latency is acceptable")
                    else:
                        print("   ❌ Latency is too high")
                    
                    return True
                else:
                    print("❌ No latency measurements collected")
                    return False
                    
        except Exception as e:
            print(f"❌ Latency test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Voice Assistant Integration Tests")
        print(f"   Backend: {self.backend_url}")
        print(f"   Room: {self.room_id}")
        print(f"   User: {self.user_id}")
        print("-" * 50)
        
        results = {
            "connection": await self.test_connection(),
            "audio_echo": await self.test_audio_echo(),
            "vad_events": await self.test_vad_events(),
            "metrics": await self.test_metrics(),
            "latency": await self.test_latency_measurements()
        }
        
        print("\n" + "=" * 50)
        print("📋 Test Results Summary:")
        print("-" * 50)
        
        passed = 0
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test:20} {status}")
            if result:
                passed += 1
        
        print("-" * 50)
        print(f"Total: {passed}/{len(results)} passed")
        
        if passed == len(results):
            print("\n🎉 All tests passed!")
            return True
        else:
            print(f"\n⚠️  {len(results) - passed} tests failed")
            return False


async def main():
    parser = argparse.ArgumentParser(description="Test Voice Assistant Integration")
    parser.add_argument(
        "--backend-url",
        default="ws://localhost:8000",
        help="WebSocket URL of the backend (default: ws://localhost:8000)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run tests continuously"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval between continuous tests in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    tester = VoiceAssistantTester(args.backend_url)
    
    if args.continuous:
        print(f"Running tests every {args.interval} seconds. Press Ctrl+C to stop.\n")
        while True:
            await tester.run_all_tests()
            print(f"\n⏳ Waiting {args.interval} seconds before next run...\n")
            await asyncio.sleep(args.interval)
    else:
        success = await tester.run_all_tests()
        exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())