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
from pipecat.serializers.protobuf import ProtobufFrameSerializer

class DebugAudioRecorder:
    def __init__(self, base_url="ws://localhost:8000"):
        self.base_url = base_url
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1600
        self.format = pyaudio.paInt16
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Audio playback queue
        self.playback_queue = queue.Queue()
        self.playback_thread = None
        self.stop_playback = threading.Event()
        
        # Pipecat's protobuf serializer
        self.serializer = ProtobufFrameSerializer()
        
        # Debug tracking
        self.debug_mode = True
        self.stats = {
            'chunks_sent': 0,
            'chunks_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'frames_serialized': 0,
            'frames_deserialized': 0,
            'transcriptions': [],
            'assistant_messages': [],
            'audio_formats_received': set(),
            'frame_types_sent': {},
            'frame_types_received': {},
            'serialization_times': [],
            'playback_chunks': 0
        }
        
        # Create debug directory
        self.debug_dir = f"debug_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        print("🔍 DEBUG MODE ENABLED")
        print(f"📁 Debug files will be saved to: {self.debug_dir}/")
        print("\n📦 Using Protobuf serialization")
        print("   OutputAudioRawFrame → server (your voice)")
        print("   InputAudioRawFrame ← server (assistant's voice)")
        
        # Print flow diagram
        print("\n🔄 AUDIO FLOW DIAGRAM:")
        print("┌─────────────────────────────────────────────────────────┐")
        print("│ CLIENT SIDE:                                            │")
        print("│                                                         │")
        print("│  🎤 Microphone                                          │")
        print("│   ↓ (PyAudio callback)                                  │")
        print("│  📦 Audio Queue                                         │")
        print("│   ↓ (send_audio task)                                  │")
        print("│  🔧 Create OutputAudioRawFrame                          │")
        print("│   ↓ (Protobuf serialization)                           │")
        print("│  📤 WebSocket.send()                                    │")
        print("│   ↓                                                     │")
        print("│  ═══════════════ NETWORK ═══════════════               │")
        print("│   ↓                                                     │")
        print("│  📥 WebSocket.recv()                                    │")
        print("│   ↓ (receive_responses task)                           │")
        print("│  🔧 Protobuf deserialization                            │")
        print("│   ↓ (InputAudioRawFrame)                               │")
        print("│  📦 Playback Queue                                      │")
        print("│   ↓ (playback_worker thread)                           │")
        print("│  🔊 Speaker                                             │")
        print("│                                                         │")
        print("└─────────────────────────────────────────────────────────┘")
    
    def log_debug(self, category, message, data=None):
        """Log debug information"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{category}] {message}"
        
        if data:
            log_entry += f"\n    Data: {data}"
        
        # Write to debug file
        with open(f"{self.debug_dir}/debug.log", "a") as f:
            f.write(log_entry + "\n")
        
        # Print important events
        if category in ["FRAME", "ERROR", "AUDIO", "CONNECTION"]:
            print(f"\n🔍 {log_entry}")
    
    def analyze_audio_chunk(self, audio_data, label=""):
        """Analyze an audio chunk for debugging"""
        samples = []
        for i in range(0, min(100, len(audio_data)), 2):
            if i + 1 < len(audio_data):
                sample = struct.unpack('<h', audio_data[i:i+2])[0]
                samples.append(sample)
        
        if samples:
            analysis = {
                'label': label,
                'size_bytes': len(audio_data),
                'duration_ms': int((len(audio_data) / 2 / self.sample_rate) * 1000),
                'sample_count': len(audio_data) // 2,
                'max_value': max(samples),
                'min_value': min(samples),
                'avg_value': int(sum(samples) / len(samples)),
                'is_silence': max(samples) < 500 and min(samples) > -500,
                'format': 'WAV' if audio_data[:4] == b'RIFF' else 'PCM'
            }
            return analysis
        return None
    
    def save_audio_sample(self, audio_data, filename_prefix):
        """Save audio data for debugging"""
        filename = f"{self.debug_dir}/{filename_prefix}_{self.stats['chunks_sent']}.wav"
        
        # If it's already WAV, save as-is
        if audio_data[:4] == b'RIFF':
            with open(filename, 'wb') as f:
                f.write(audio_data)
        else:
            # Convert PCM to WAV
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
        
        return filename
    
    def start_playback_thread(self):
        """Start the audio playback thread"""
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.stop_playback.clear()
            self.playback_thread = threading.Thread(target=self.playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            self.log_debug("PLAYBACK", "Started playback thread")
    
    def playback_worker(self):
        """Worker thread for continuous audio playback"""
        stream = None
        chunks_played = 0
        
        try:
            self.log_debug("PLAYBACK_THREAD", "Starting playback worker thread")
            
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.log_debug("PLAYBACK", "Audio output stream opened", {
                'format': 'pyaudio.paInt16',
                'channels': self.channels,
                'rate': self.sample_rate,
                'chunk_size': self.chunk_size
            })
            
            while not self.stop_playback.is_set():
                try:
                    # Debug: Show queue state periodically
                    if chunks_played % 20 == 0:
                        self.log_debug("PLAYBACK_QUEUE", f"Queue state", {
                            'items_in_queue': self.playback_queue.qsize(),
                            'chunks_played_so_far': chunks_played
                        })
                    
                    audio_data = self.playback_queue.get(timeout=0.1)
                    
                    # Debug: Got audio from queue
                    self.log_debug("PLAYBACK_GOT", f"Got audio chunk #{chunks_played + 1} from queue", {
                        'size': len(audio_data),
                        'duration_ms': (len(audio_data) / 2 / self.sample_rate) * 1000
                    })
                    
                    # Analyze before playing
                    analysis = self.analyze_audio_chunk(audio_data, "playback")
                    if analysis and (chunks_played < 5 or chunks_played % 20 == 0):
                        self.log_debug("PLAYBACK", f"Playing chunk {self.stats['playback_chunks']}", analysis)
                    
                    # Actually play the audio
                    self.log_debug("PYAUDIO_WRITE", f"Writing {len(audio_data)} bytes to PyAudio")
                    stream.write(audio_data)
                    self.log_debug("PYAUDIO_WRITTEN", f"Successfully played chunk #{chunks_played + 1}")
                    
                    chunks_played += 1
                    self.stats['playback_chunks'] += 1
                    
                except queue.Empty:
                    # Debug: Queue is empty
                    if getattr(self, '_playback_empty_count', 0) % 20 == 0:
                        self.log_debug("PLAYBACK_EMPTY", "Playback queue empty, waiting...")
                    setattr(self, '_playback_empty_count', getattr(self, '_playback_empty_count', 0) + 1)
                    continue
                except Exception as e:
                    self.log_debug("ERROR", f"Playback error: {e}")
                    import traceback
                    self.log_debug("ERROR", f"Traceback: {traceback.format_exc()}")
                    
        except Exception as e:
            self.log_debug("ERROR", f"Failed to start playback: {e}")
        finally:
            if stream:
                self.log_debug("PLAYBACK_CLEANUP", "Closing audio stream")
                stream.stop_stream()
                stream.close()
            self.log_debug("PLAYBACK", f"Playback thread ended after {chunks_played} chunks")
    
    def queue_audio_for_playback(self, audio_data):
        """Queue audio data for playback"""
        self.playback_queue.put(audio_data)
        self.log_debug("PLAYBACK", f"Queued {len(audio_data)} bytes for playback")
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("\n🎤 Available audio input devices:")
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")
                devices.append(info)
        
        self.log_debug("DEVICES", f"Found {len(devices)} input devices")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        # Debug: Show when callback is triggered
        callback_num = getattr(self, '_callback_count', 0) + 1
        setattr(self, '_callback_count', callback_num)
        
        if callback_num <= 5 or callback_num % 50 == 0:
            self.log_debug("CALLBACK", f"Audio callback #{callback_num}", {
                'frame_count': frame_count,
                'status': status,
                'data_size': len(in_data),
                'is_recording': self.is_recording
            })
        
        if self.is_recording:
            # Debug: Show queue state before adding
            if callback_num <= 5:
                self.log_debug("QUEUE", f"Adding to queue", {
                    'current_queue_size': self.audio_queue.qsize(),
                    'data_size': len(in_data)
                })
            
            self.audio_queue.put(in_data)
            
            # Debug every 50th callback
            if self.stats['chunks_sent'] % 50 == 0:
                analysis = self.analyze_audio_chunk(in_data, "input")
                if analysis:
                    self.log_debug("AUDIO_IN", f"Captured chunk {self.stats['chunks_sent']}", analysis)
        
        return (in_data, pyaudio.paContinue)
    
    async def send_audio(self, websocket):
        """Send audio data to websocket using Pipecat protocol"""
        chunk_count = 0
        
        self.log_debug("SEND", "Started send_audio task")
        self.log_debug("FLOW", "SEND TASK: Waiting for audio from queue...")
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Debug: Show queue state
                queue_size = self.audio_queue.qsize()
                if chunk_count == 0 or chunk_count % 10 == 0:
                    self.log_debug("QUEUE_STATE", f"Queue has {queue_size} items")
                
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Debug: Show what we got from queue
                if chunk_count < 5:
                    self.log_debug("FLOW", f"Got audio from queue: {len(audio_data)} bytes")
                
                # Create an OutputAudioRawFrame
                self.log_debug("FRAME_CREATE", f"Creating OutputAudioRawFrame", {
                    'audio_size': len(audio_data),
                    'sample_rate': self.sample_rate,
                    'channels': self.channels
                })
                
                frame = OutputAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=self.channels
                )
                
                # Debug: Show frame details
                if chunk_count < 5:
                    self.log_debug("FRAME_DETAILS", "Frame created", {
                        'type': frame.__class__.__name__,
                        'has_audio': hasattr(frame, 'audio'),
                        'audio_len': len(frame.audio) if hasattr(frame, 'audio') else 0,
                        'sample_rate': getattr(frame, 'sample_rate', 'N/A'),
                        'num_channels': getattr(frame, 'num_channels', 'N/A')
                    })
                
                # Track frame type
                frame_type = frame.__class__.__name__
                self.stats['frame_types_sent'][frame_type] = self.stats['frame_types_sent'].get(frame_type, 0) + 1
                
                # Serialize the frame
                self.log_debug("SERIALIZE_START", f"Starting serialization of {frame_type}")
                start_time = asyncio.get_event_loop().time()
                serialized = await self.serializer.serialize(frame)
                serialization_time = (asyncio.get_event_loop().time() - start_time) * 1000
                self.stats['serialization_times'].append(serialization_time)
                
                if serialized:
                    # Debug: Show serialization result
                    if chunk_count < 5 or chunk_count % 20 == 0:
                        self.log_debug("SERIALIZE_RESULT", f"Serialization complete", {
                            'success': True,
                            'input_size': len(audio_data),
                            'output_size': len(serialized),
                            'overhead_bytes': len(serialized) - len(audio_data),
                            'time_ms': f"{serialization_time:.2f}",
                            'compression_ratio': f"{len(serialized) / len(audio_data):.2f}"
                        })
                        
                        # Show hex dump of first serialized message
                        if chunk_count == 0:
                            self.log_debug("PROTOBUF_HEX", "First message hex dump", {
                                'first_50_bytes': serialized[:50].hex(),
                                'last_10_bytes': serialized[-10:].hex()
                            })
                        
                        # Save sample
                        self.save_audio_sample(audio_data, "sent")
                    
                    # Send as binary WebSocket message
                    self.log_debug("WEBSOCKET_SEND", f"Sending {len(serialized)} bytes via WebSocket")
                    await websocket.send(serialized)
                    self.log_debug("WEBSOCKET_SENT", f"Successfully sent chunk {chunk_count}")
                    
                    self.stats['chunks_sent'] += 1
                    self.stats['bytes_sent'] += len(audio_data)
                    self.stats['frames_serialized'] += 1
                    
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print(f"\r📤 Sent {chunk_count} chunks ({self.stats['bytes_sent']:,} bytes)", end='', flush=True)
                    
                else:
                    self.log_debug("ERROR", f"Failed to serialize {frame_type}")
                    
            except queue.Empty:
                # Debug: Show we're waiting
                if getattr(self, '_empty_count', 0) % 10 == 0:
                    self.log_debug("QUEUE_EMPTY", "No audio in queue, waiting...")
                setattr(self, '_empty_count', getattr(self, '_empty_count', 0) + 1)
                continue
            except Exception as e:
                self.log_debug("ERROR", f"Send error: {e}")
                import traceback
                self.log_debug("ERROR", f"Traceback: {traceback.format_exc()}")
                break
        
        self.log_debug("SEND", "Ended send_audio task", {
            'total_chunks': self.stats['chunks_sent'],
            'total_bytes': self.stats['bytes_sent']
        })
    
    async def receive_responses(self, websocket):
        """Receive responses from the server"""
        audio_chunks_received = 0
        message_count = 0
        
        self.log_debug("RECEIVE", "Started receive_responses task")
        self.log_debug("FLOW", "RECEIVE TASK: Waiting for server messages...")
        
        try:
            while True:
                # Receive binary message
                self.log_debug("WEBSOCKET_RECV", f"Waiting for message #{message_count + 1}")
                message = await websocket.recv()
                message_count += 1
                
                self.log_debug("WEBSOCKET", f"Received message #{message_count}", {
                    'type': 'binary' if isinstance(message, bytes) else 'text',
                    'size': len(message) if isinstance(message, bytes) else len(message.encode())
                })
                
                if isinstance(message, bytes):
                    try:
                        # Debug: Show raw message details
                        if message_count <= 5:
                            self.log_debug("RAW_MESSAGE", f"Binary message details", {
                                'size': len(message),
                                'first_20_hex': message[:20].hex(),
                                'last_10_hex': message[-10:].hex()
                            })
                        
                        # Save raw message for debugging
                        if self.stats['chunks_received'] < 5:
                            filename = f"{self.debug_dir}/raw_msg_{self.stats['chunks_received']}.bin"
                            with open(filename, 'wb') as f:
                                f.write(message)
                            self.log_debug("DEBUG_FILE", f"Saved raw message to {filename}")
                        
                        # Try to deserialize as protobuf
                        self.log_debug("DESERIALIZE_START", f"Starting deserialization of {len(message)} bytes")
                        frame = await self.serializer.deserialize(message)
                        
                        if frame:
                            frame_type = frame.__class__.__name__
                            self.stats['frame_types_received'][frame_type] = self.stats['frame_types_received'].get(frame_type, 0) + 1
                            self.stats['frames_deserialized'] += 1
                            
                            self.log_debug("DESERIALIZE_SUCCESS", f"Deserialized to {frame_type}")
                            
                            # Debug: Show frame attributes
                            if message_count <= 5:
                                frame_attrs = {}
                                for attr in ['audio', 'text', 'sample_rate', 'num_channels', 'user_id', 'timestamp']:
                                    if hasattr(frame, attr):
                                        value = getattr(frame, attr)
                                        if attr == 'audio':
                                            frame_attrs[attr] = f"{len(value)} bytes" if value else "None"
                                        else:
                                            frame_attrs[attr] = value
                                self.log_debug("FRAME_ATTRS", f"Frame attributes", frame_attrs)
                            
                            # Handle InputAudioRawFrame or AudioRawFrame
                            if frame_type in ["InputAudioRawFrame", "AudioRawFrame"] or hasattr(frame, 'audio'):
                                audio_chunks_received += 1
                                self.stats['chunks_received'] += 1
                                
                                # Get the audio data
                                audio_data = frame.audio
                                self.stats['bytes_received'] += len(audio_data)
                                
                                self.log_debug("AUDIO_FRAME", f"Processing audio frame #{audio_chunks_received}", {
                                    'frame_type': frame_type,
                                    'audio_size': len(audio_data),
                                    'total_received': self.stats['bytes_received']
                                })
                                
                                # Analyze audio
                                analysis = self.analyze_audio_chunk(audio_data, "received")
                                if analysis:
                                    self.stats['audio_formats_received'].add(analysis['format'])
                                    self.log_debug("AUDIO", f"Received audio chunk {audio_chunks_received}", analysis)
                                
                                # Save samples
                                if audio_chunks_received <= 5 or audio_chunks_received % 20 == 0:
                                    filename = self.save_audio_sample(audio_data, f"received_{audio_chunks_received}")
                                    self.log_debug("AUDIO_SAVED", f"Saved audio sample to {filename}")
                                
                                if audio_chunks_received == 1:
                                    print(f"\n🎵 Starting audio playback...")
                                    self.log_debug("PLAYBACK_INIT", "Initializing playback for first audio chunk")
                                    self.start_playback_thread()
                                
                                # Extract PCM if it's WAV
                                if audio_data[:4] == b'RIFF':
                                    self.log_debug("WAV_DETECTED", "Audio is in WAV format, extracting PCM")
                                    wav_buffer = io.BytesIO(audio_data)
                                    with wave.open(wav_buffer, 'rb') as wav_file:
                                        # Debug WAV details
                                        self.log_debug("WAV_INFO", "WAV file details", {
                                            'channels': wav_file.getnchannels(),
                                            'sample_width': wav_file.getsampwidth(),
                                            'framerate': wav_file.getframerate(),
                                            'n_frames': wav_file.getnframes()
                                        })
                                        pcm_data = wav_file.readframes(wav_file.getnframes())
                                        self.log_debug("PCM_EXTRACTED", f"Extracted {len(pcm_data)} bytes of PCM from WAV")
                                        self.queue_audio_for_playback(pcm_data)
                                else:
                                    # Already PCM
                                    self.log_debug("PCM_DIRECT", "Audio is already PCM, queuing directly")
                                    self.queue_audio_for_playback(audio_data)
                                
                                if audio_chunks_received % 10 == 0:
                                    print(f"\r🎵 Received {audio_chunks_received} audio chunks", end='', flush=True)
                                
                            elif frame_type == "TranscriptionFrame":
                                transcription = frame.text
                                self.stats['transcriptions'].append(transcription)
                                print(f"\n✅ You said: \"{transcription}\"")
                                self.log_debug("TRANSCRIPTION", f"Transcription received", {
                                    'text': transcription,
                                    'user_id': getattr(frame, 'user_id', 'N/A'),
                                    'timestamp': getattr(frame, 'timestamp', 'N/A')
                                })
                                
                            elif frame_type == "TextFrame":
                                text = frame.text
                                self.stats['assistant_messages'].append(text)
                                print(f"\n🤖 Assistant: \"{text}\"")
                                self.log_debug("ASSISTANT", f"Assistant text received", {
                                    'text': text,
                                    'length': len(text)
                                })
                                audio_chunks_received = 0
                            
                            else:
                                self.log_debug("FRAME", f"Received other frame type: {frame_type}")
                                # Log all attributes
                                attrs = {attr: getattr(frame, attr) for attr in dir(frame) 
                                        if not attr.startswith('_') and not callable(getattr(frame, attr))}
                                self.log_debug("FRAME_DATA", f"{frame_type} attributes", attrs)
                            
                        else:
                            self.log_debug("DESERIALIZE_FAIL", "Failed to deserialize message", {
                                'size': len(message),
                                'first_bytes': message[:20].hex()
                            })
                            
                    except Exception as e:
                        self.log_debug("ERROR", f"Error processing message: {e}")
                        import traceback
                        self.log_debug("ERROR", f"Traceback: {traceback.format_exc()}")
                        
                else:
                    # Text message
                    self.log_debug("TEXT_MESSAGE", f"Received text message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.log_debug("CONNECTION", "WebSocket connection closed")
        except Exception as e:
            self.log_debug("ERROR", f"Receive error: {e}")
            import traceback
            self.log_debug("ERROR", f"Traceback: {traceback.format_exc()}")
    
    async def get_session_id(self):
        """Get a new session ID from the server"""
        import aiohttp
        self.log_debug("SESSION", "Requesting new session ID")
        
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/session/new') as resp:
                data = await resp.json()
                session_id = data['session_id']
                self.log_debug("SESSION", f"Got session ID: {session_id}")
                return session_id
    
    def print_final_stats(self):
        """Print final statistics"""
        print("\n\n" + "=" * 60)
        print("📊 SESSION STATISTICS")
        print("=" * 60)
        
        print(f"\n📤 SENT:")
        print(f"   Chunks: {self.stats['chunks_sent']}")
        print(f"   Bytes: {self.stats['bytes_sent']:,}")
        print(f"   Frames serialized: {self.stats['frames_serialized']}")
        print(f"   Frame types: {dict(self.stats['frame_types_sent'])}")
        
        if self.stats['serialization_times']:
            avg_time = sum(self.stats['serialization_times']) / len(self.stats['serialization_times'])
            print(f"   Avg serialization time: {avg_time:.2f}ms")
        
        print(f"\n📥 RECEIVED:")
        print(f"   Chunks: {self.stats['chunks_received']}")
        print(f"   Bytes: {self.stats['bytes_received']:,}")
        print(f"   Frames deserialized: {self.stats['frames_deserialized']}")
        print(f"   Frame types: {dict(self.stats['frame_types_received'])}")
        print(f"   Audio formats: {list(self.stats['audio_formats_received'])}")
        
        print(f"\n🔊 PLAYBACK:")
        print(f"   Chunks played: {self.stats['playback_chunks']}")
        
        print(f"\n💬 CONVERSATION:")
        print(f"   Transcriptions: {len(self.stats['transcriptions'])}")
        for i, trans in enumerate(self.stats['transcriptions'], 1):
            print(f"     {i}. \"{trans}\"")
        
        print(f"\n   Assistant messages: {len(self.stats['assistant_messages'])}")
        for i, msg in enumerate(self.stats['assistant_messages'], 1):
            print(f"     {i}. \"{msg}\"")
        
        print(f"\n📁 Debug files saved to: {self.debug_dir}/")
        print("=" * 60)
    
    async def start_recording(self, device_index=None):
        """Start recording and streaming audio"""
        print("\n🔍 DEBUG: Pipecat Voice Assistant Test")
        print("=" * 50)
        
        self.log_debug("MAIN", "Starting application")
        
        # Get session ID
        self.log_debug("SESSION", "Requesting session from server")
        session_id = await self.get_session_id()
        print(f"\n📋 Session ID: {session_id}")
        
        ws_url = f"{self.base_url}/ws/audio?session={session_id}"
        print(f"🔗 Connecting to {ws_url}...")
        
        self.log_debug("CONNECTION", f"Attempting WebSocket connection to {ws_url}")
        
        try:
            # Debug: Show connection parameters
            self.log_debug("WEBSOCKET_PARAMS", "Connection parameters", {
                'url': ws_url,
                'max_size': 10**8,
                'compression': None
            })
            
            async with websockets.connect(
                ws_url, 
                max_size=10**8,
                compression=None
            ) as websocket:
                print("✅ Connected! WebSocket is open")
                
                # Get websocket attributes safely
                ws_info = {
                    'compression': 'None',
                    'state': str(websocket.state) if hasattr(websocket, 'state') else 'connected',
                    'local_address': str(websocket.local_address) if hasattr(websocket, 'local_address') else 'N/A',
                    'remote_address': str(websocket.remote_address) if hasattr(websocket, 'remote_address') else 'N/A'
                }
                
                self.log_debug("CONNECTION", "WebSocket connected", ws_info)
                self.log_debug("FLOW", "=== AUDIO FLOW STARTING ===")
                
                # Open audio stream for recording
                self.log_debug("AUDIO_INIT", "Preparing to open audio input stream")
                
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self.audio_callback
                )
                
                self.log_debug("AUDIO", "Opened input stream", {
                    'device_index': device_index,
                    'format': 'pyaudio.paInt16',
                    'channels': self.channels,
                    'rate': self.sample_rate,
                    'chunk_size': self.chunk_size,
                    'callback_registered': True
                })
                
                self.log_debug("FLOW", "Starting recording flag")
                self.is_recording = True
                
                self.log_debug("AUDIO_START", "Starting audio stream")
                stream.start_stream()
                self.log_debug("AUDIO_STARTED", "Audio stream is now active")
                
                print("\n🎤 Recording started!")
                print("💡 Try saying: 'Hello, how are you today?'")
                print("⏸️  Press Enter to stop\n")
                
                # Create tasks for sending and receiving
                self.log_debug("TASKS", "Creating async tasks")
                
                self.log_debug("TASK_CREATE", "Creating send_audio task")
                send_task = asyncio.create_task(self.send_audio(websocket))
                
                self.log_debug("TASK_CREATE", "Creating receive_responses task")
                receive_task = asyncio.create_task(self.receive_responses(websocket))
                
                self.log_debug("TASKS", "Both tasks created and running")
                
                # Wait for Enter key
                def wait_for_enter():
                    self.log_debug("INPUT", "Waiting for user to press Enter")
                    input()
                    self.log_debug("INPUT", "Enter pressed by user")
                    self.is_recording = False
                    self.log_debug("USER", "Stop requested")
                
                enter_thread = threading.Thread(target=wait_for_enter)
                enter_thread.start()
                self.log_debug("THREAD", "Enter key listener thread started")
                
                # Wait for recording to stop
                self.log_debug("MAIN_LOOP", "Entering main wait loop")
                loop_count = 0
                while self.is_recording:
                    await asyncio.sleep(0.1)
                    loop_count += 1
                    if loop_count % 50 == 0:  # Every 5 seconds
                        self.log_debug("MAIN_LOOP", f"Still recording... (loop #{loop_count})", {
                            'send_task_done': send_task.done(),
                            'receive_task_done': receive_task.done(),
                            'chunks_sent': self.stats['chunks_sent'],
                            'chunks_received': self.stats['chunks_received']
                        })
                
                print("\n\n🛑 Stopping recording...")
                self.log_debug("SHUTDOWN", "Beginning shutdown sequence")
                
                # Stop recording stream
                self.log_debug("AUDIO_STOP", "Stopping audio input stream")
                stream.stop_stream()
                stream.close()
                self.log_debug("AUDIO", "Input stream closed")
                
                # Give time for final responses
                print("⏳ Waiting for final responses...")
                self.log_debug("WAIT", "Waiting 3 seconds for final responses")
                await asyncio.sleep(30)
                
                # Stop playback
                self.log_debug("PLAYBACK_STOP", "Setting stop flag for playback thread")
                self.stop_playback.set()
                if self.playback_thread:
                    self.log_debug("THREAD_JOIN", "Waiting for playback thread to finish")
                    self.playback_thread.join(timeout=1)
                    self.log_debug("THREAD_JOINED", "Playback thread finished")
                
                # Cancel tasks
                self.log_debug("TASKS_CANCEL", "Cancelling async tasks")
                send_task.cancel()
                receive_task.cancel()
                
                try:
                    await asyncio.gather(send_task, receive_task, return_exceptions=True)
                except:
                    pass
                
                self.log_debug("TASKS_CANCELLED", "All tasks cancelled")
                
                # Print statistics
                self.print_final_stats()
                
                print("\n✅ Session complete!")
                self.log_debug("COMPLETE", "Session ended successfully")
                
        except Exception as e:
            self.log_debug("ERROR", f"Fatal error: {e}")
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.log_debug("CLEANUP", "Starting final cleanup")
            self.audio.terminate()
            self.log_debug("CLEANUP", "Audio terminated")

def main():
    print("🔍 Pipecat Voice Assistant - Debug Edition")
    print("=" * 50)
    
    recorder = DebugAudioRecorder()
    
    # List devices
    recorder.list_audio_devices()
    
    # Select device
    print("\nEnter device index (or press Enter for default): ", end='')
    device_input = input().strip()
    device_index = int(device_input) if device_input else None
    
    # Run the recorder
    asyncio.run(recorder.start_recording(device_index))

if __name__ == "__main__":
    main()