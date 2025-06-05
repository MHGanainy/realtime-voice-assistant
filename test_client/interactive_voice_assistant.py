import asyncio
import pyaudio
import websockets
import json
import sys
import threading
import queue
from datetime import datetime
import struct
import wave
import io
import os
from pipecat.frames.frames import (
    OutputAudioRawFrame,
    InputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    Frame,
    AudioRawFrame
)
from pipecat.frames.protobufs import frames_pb2
from pipecat.serializers.protobuf import ProtobufFrameSerializer

class InteractiveVoiceAssistant:
    def __init__(self, base_url="ws://localhost:8000"):
        self.base_url = base_url
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.conversation_active = True
        
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 320
        self.format = pyaudio.paInt16
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Audio playback queue
        self.playback_queue = queue.Queue()
        self.playback_thread = None
        self.stop_playback = threading.Event()
        
        # Conversation state
        self.is_assistant_speaking = False
        self.should_send_audio = True  # New flag to control audio sending
        self.conversation_history = []
        self.last_speech_time = 0
        
        # VAD for local speech detection
        self.silence_threshold = 500
        self.silence_duration = 0
        self.speaking = False
        
        print("🎙️ Interactive Voice Assistant")
        print("=" * 50)
        print("💡 Tips:")
        print("   - Speak naturally, the assistant will respond")
        print("   - Say 'goodbye' or press Ctrl+C to exit")
        print("   - The conversation will continue automatically")
        print("=" * 50)
    
    def detect_speech(self, audio_data):
        """Simple VAD - detect if user is speaking"""
        samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
        max_amplitude = max(abs(s) for s in samples) if samples else 0
        
        if max_amplitude > self.silence_threshold:
            if not self.speaking and not self.is_assistant_speaking:
                self.speaking = True
                print("\n🎤 Listening...", end='', flush=True)
            self.silence_duration = 0
            self.last_speech_time = 0
            return True
        else:
            if self.speaking:
                self.silence_duration += self.chunk_size / self.sample_rate
                if self.silence_duration > 1.5:  # 1.5 seconds of silence
                    self.speaking = False
                    if not self.is_assistant_speaking:
                        print(" (processing...)")
                    return False
            return False
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream - only queue audio when assistant is not speaking"""
        if self.is_recording:
            # Always detect speech for UI feedback
            self.detect_speech(in_data)
            
            # Only queue audio if we should be sending (assistant not speaking)
            if self.should_send_audio:
                self.audio_queue.put(in_data)
        
        return (in_data, pyaudio.paContinue)
    
    async def send_audio(self, websocket):
        """Send audio data to websocket continuously"""
        print("\n🎤 Ready to listen!")
        chunk_count = 0
        
        while self.conversation_active:
            try:
                if self.should_send_audio:
                    # Get audio from queue
                    try:
                        audio_data = self.audio_queue.get(timeout=0.01)
                        
                        # Create and send frame
                        frame = frames_pb2.Frame()
                        frame.audio.audio = audio_data
                        frame.audio.sample_rate = self.sample_rate
                        frame.audio.num_channels = 1
                        
                        serialized = frame.SerializeToString()
                        if serialized:
                            await websocket.send(serialized)
                            chunk_count += 1
                            
                            # Debug output every 100 chunks
                            if chunk_count % 100 == 0:
                                print(f"\r📤 Sent {chunk_count} audio chunks", end='', flush=True)
                    
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                else:
                    # Assistant is speaking, clear the audio queue to prevent buildup
                    try:
                        while True:
                            self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                if self.conversation_active:
                    print(f"\n❌ Send error: {e}")
                break
    
    async def receive_responses(self, websocket):
        """Receive responses from the server"""
        audio_chunks = 0
        
        while self.conversation_active:
            try:
                # Add timeout to prevent blocking
                message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                
                if isinstance(message, bytes):
                    frame = frames_pb2.Frame()
                    frame.ParseFromString(message)
                    
                    # Check which type of frame we received
                    if frame.HasField('audio'):
                        audio_chunks += 1
                        
                        # First audio chunk - assistant started speaking
                        if audio_chunks == 1:
                            self.is_assistant_speaking = True
                            self.should_send_audio = False  # Stop sending audio
                            self.speaking = False  # Reset user speaking flag
                            print("\n🤖 Assistant speaking...", end='', flush=True)
                            
                            # Clear any pending audio in the queue
                            try:
                                while True:
                                    self.audio_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        # Queue the raw audio bytes for playback
                        audio_bytes = frame.audio.audio
                        self.playback_queue.put(audio_bytes)
                        
                        # Show progress
                        if audio_chunks % 10 == 0:
                            print(".", end='', flush=True)
                            
                    elif frame.HasField('transcription'):
                        # Show what user said
                        if hasattr(frame.transcription, 'text'):
                            text = frame.transcription.text
                            if text:
                                print(f"\n👤 You said: \"{text}\"")
                                self.conversation_history.append(("user", text))
                                
                                # Check for exit commands
                                if any(word in text.lower() for word in ["goodbye", "bye", "exit", "quit"]):
                                    print("\n👋 Goodbye! Thanks for chatting!")
                                    self.conversation_active = False
                                
                    elif frame.HasField('text'):
                        # Assistant's text response
                        if hasattr(frame.text, 'text'):
                            text = frame.text.text
                            if text:
                                # Reset audio chunk counter for next response
                                if audio_chunks > 0:
                                    print(f"\n💬 {text}")
                                    audio_chunks = 0
                                self.conversation_history.append(("assistant", text))
                                
            except asyncio.TimeoutError:
                # This is expected - no message within timeout
                continue
                
            except websockets.exceptions.ConnectionClosed:
                print("\n📡 Connection closed")
                break
                
            except Exception as e:
                if self.conversation_active:
                    print(f"\n❌ Receive error: {e}")
                    import traceback
                    traceback.print_exc()
                break
    
    def playback_worker(self):
        """Worker thread for audio playback"""
        stream = None
        
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            consecutive_empty = 0
            
            while not self.stop_playback.is_set():
                try:
                    audio_data = self.playback_queue.get(timeout=0.01)
                    stream.write(audio_data)
                    consecutive_empty = 0
                    
                except queue.Empty:
                    consecutive_empty += 1
                    # If queue has been empty for a while and we were speaking
                    if consecutive_empty > 50 and self.is_assistant_speaking:
                        self.is_assistant_speaking = False
                        self.should_send_audio = True  # Resume sending audio
                        print("\n✅ Ready for next input")
                    continue
                    
        except Exception as e:
            print(f"\n❌ Playback error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    async def start_conversation(self, device_index=None):
        """Start the interactive conversation"""
        try:
            # Get session ID
            session_id = await self.get_session_id()
            print(f"\n📋 Session ID: {session_id}")
            
            ws_url = f"{self.base_url}/ws/audio?session={session_id}"
            print(f"🔗 Connecting to assistant...")
            
            async with websockets.connect(
                ws_url, 
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,
                compression=None
            ) as websocket:
                print("✅ Connected!")
                
                # Open audio stream
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self.audio_callback
                )
                
                # Start recording
                self.is_recording = True
                stream.start_stream()
                
                # Start playback thread
                self.stop_playback.clear()
                self.playback_thread = threading.Thread(target=self.playback_worker)
                self.playback_thread.daemon = True
                self.playback_thread.start()
                
                # Create tasks
                send_task = asyncio.create_task(self.send_audio(websocket))
                receive_task = asyncio.create_task(self.receive_responses(websocket))
                
                # Wait for conversation to end
                while self.conversation_active:
                    await asyncio.sleep(0.1)
                
                # Cleanup
                print("\n\n🛑 Ending conversation...")
                self.is_recording = False
                stream.stop_stream()
                stream.close()
                
                # Wait a bit for final responses
                await asyncio.sleep(2)
                
                # Stop tasks
                send_task.cancel()
                receive_task.cancel()
                
                try:
                    await asyncio.gather(send_task, receive_task, return_exceptions=True)
                except:
                    pass
                
                # Stop playback
                self.stop_playback.set()
                if self.playback_thread:
                    self.playback_thread.join(timeout=2)
                
                # Show conversation summary
                self.print_conversation_summary()
                
        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted!")
            self.conversation_active = False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.audio.terminate()
    
    async def get_session_id(self):
        """Get a new session ID from the server"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/session/new') as resp:
                data = await resp.json()
                return data['session_id']
    
    def print_conversation_summary(self):
        """Print conversation history"""
        print("\n" + "=" * 60)
        print("💬 CONVERSATION SUMMARY")
        print("=" * 60)
        
        if self.conversation_history:
            for speaker, text in self.conversation_history:
                icon = "👤" if speaker == "user" else "🤖"
                print(f"\n{icon} {speaker.title()}: {text}")
        else:
            print("\nNo conversation recorded.")
        
        print("\n" + "=" * 60)
        print("✅ Session complete!")
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\n🎤 Available audio input devices:")
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")
                devices.append(info)
        return devices

def main():
    assistant = InteractiveVoiceAssistant()
    
    # List devices
    assistant.list_audio_devices()
    
    # Select device
    print("\nEnter device index (or press Enter for default): ", end='')
    device_input = input().strip()
    device_index = int(device_input) if device_input else None
    
    # Start conversation
    asyncio.run(assistant.start_conversation(device_index))

if __name__ == "__main__":
    main()