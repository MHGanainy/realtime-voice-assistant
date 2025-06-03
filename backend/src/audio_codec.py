"""
Opus audio codec implementation for real-time voice
Provides 70%+ bandwidth reduction while maintaining quality
"""
import os
import sys

# CRITICAL: Set library path BEFORE any imports that might use it
if sys.platform == 'darwin':  # macOS
    # Add homebrew library paths
    lib_paths = ['/opt/homebrew/lib', '/usr/local/lib']
    current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
    new_paths = [p for p in lib_paths if p not in current_dyld]
    if new_paths:
        os.environ['DYLD_LIBRARY_PATH'] = ':'.join(new_paths + [current_dyld]).rstrip(':')

import struct
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

# Force reload of ctypes to pick up new DYLD_LIBRARY_PATH
import ctypes
import importlib
importlib.reload(ctypes)

# Try to import opuslib for Opus support
OPUS_AVAILABLE = False

try:
    # First try to load the opus library directly
    if sys.platform == 'darwin':
        opus_lib_paths = [
            '/opt/homebrew/lib/libopus.0.dylib',
            '/opt/homebrew/lib/libopus.dylib',
            '/usr/local/lib/libopus.dylib'
        ]
        for path in opus_lib_paths:
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    break
                except:
                    pass
    
    import opuslib
    
    # Test that it actually works
    test_encoder = opuslib.Encoder(48000, 1, opuslib.APPLICATION_VOIP)
    del test_encoder
    OPUS_AVAILABLE = True
    logging.info("opuslib loaded successfully")
except ImportError as e:
    logging.warning(f"opuslib not available. Install with: pip install opuslib")
except Exception as e:
    logging.warning(f"Error loading opuslib: {e}")

logger = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    """Audio frame with metadata for streaming"""
    data: bytes
    timestamp_ms: int
    sequence_number: int
    duration_ms: int = 20
    
    def serialize(self) -> bytes:
        """Serialize frame for network transport
        Format: [seq_num:4][timestamp:4][data:N]
        """
        # Ensure values fit in 32 bits
        seq_32bit = self.sequence_number % (2**32)
        timestamp_32bit = self.timestamp_ms % (2**32)
        
        header = struct.pack('>II', seq_32bit, timestamp_32bit)
        return header + self.data
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'AudioFrame':
        """Deserialize frame from network data"""
        if len(data) < 8:
            raise ValueError("Invalid frame data: too short")
        
        seq_num, timestamp = struct.unpack('>II', data[:8])
        audio_data = data[8:]
        
        return cls(
            data=audio_data,
            timestamp_ms=timestamp,
            sequence_number=seq_num
        )


class OpusEncoder:
    """Opus encoder for real-time voice compression"""
    
    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        bitrate: int = 32000,
        complexity: int = 5,
        use_dtx: bool = True,
        use_inband_fec: bool = True
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = 20
        self.bitrate = bitrate
        self.complexity = complexity
        self.use_dtx = use_dtx
        self.use_inband_fec = use_inband_fec
        
        # Calculate frame size in samples
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        if OPUS_AVAILABLE:
            try:
                # opuslib.Encoder expects specific sample rates
                # Valid rates: 8000, 12000, 16000, 24000, 48000
                if sample_rate not in [8000, 12000, 16000, 24000, 48000]:
                    raise ValueError(f"Invalid sample rate {sample_rate} for Opus")
                
                # Initialize opuslib encoder
                self.encoder = opuslib.Encoder(
                    fs=sample_rate,
                    channels=channels,
                    application=opuslib.APPLICATION_VOIP
                )
                
                # Configure encoder with valid ranges
                # Note: Some properties might fail due to ctypes issues
                try:
                    # Bitrate: 6000-510000 for stereo, 6000-256000 for mono
                    self.encoder.bitrate = max(6000, min(bitrate, 256000 if channels == 1 else 510000))
                except Exception as e:
                    logger.debug(f"Could not set bitrate: {e}")
                
                try:
                    # Complexity: 0-10
                    self.encoder.complexity = max(0, min(complexity, 10))
                except Exception as e:
                    logger.debug(f"Could not set complexity: {e}")
                
                # DTX and FEC might have ctypes issues, wrap them individually
                if use_dtx:
                    try:
                        self.encoder.dtx = 1
                    except Exception as e:
                        logger.debug(f"Could not enable DTX: {e}")
                
                if use_inband_fec:
                    try:
                        self.encoder.inband_fec = 1
                    except Exception as e:
                        logger.debug(f"Could not enable inband FEC: {e}")
                    
                    try:
                        self.encoder.packet_loss_perc = 10
                    except Exception as e:
                        logger.debug(f"Could not set packet loss percentage: {e}")
                
                logger.info("Opus encoder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Opus encoder: {e}")
                self.encoder = None
        else:
            self.encoder = None
            logger.warning("Opus encoder not available - will use mock compression")
    
    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM audio to Opus
        
        Args:
            pcm_data: Raw PCM audio (16-bit, mono/stereo)
            
        Returns:
            Compressed Opus data
        """
        if not self.encoder:
            # Mock compression for testing
            # Real Opus would compress ~93%, we'll simulate with simple decimation
            samples = np.frombuffer(pcm_data, dtype=np.int16)
            # Take every 10th sample and keep as int16 to preserve quality
            decimated = samples[::10]
            # Pack as bytes (still int16, but fewer samples)
            compressed = decimated.tobytes()
            # Add a header to distinguish from real Opus
            return b'MOCK' + compressed
        
        try:
            # Encode with opuslib
            opus_data = self.encoder.encode(pcm_data, self.frame_size)
            return opus_data
        except Exception as e:
            logger.error(f"Opus encoding failed: {e}")
            # Fall back to mock compression
            samples = np.frombuffer(pcm_data, dtype=np.int16)
            decimated = samples[::10]
            compressed = decimated.tobytes()
            return b'MOCK' + compressed
    
    def reset(self):
        """Reset encoder state"""
        if self.encoder:
            # Re-initialize to reset state
            self.__init__(
                self.sample_rate,
                self.channels,
                self.bitrate,
                self.complexity,
                self.use_dtx,
                self.use_inband_fec
            )


class OpusDecoder:
    """Opus decoder for real-time voice decompression"""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = int(sample_rate * 20 / 1000)  # 20ms frames
        
        if OPUS_AVAILABLE:
            try:
                # Validate sample rate
                if sample_rate not in [8000, 12000, 16000, 24000, 48000]:
                    raise ValueError(f"Invalid sample rate {sample_rate} for Opus")
                
                # Initialize opuslib decoder
                self.decoder = opuslib.Decoder(fs=sample_rate, channels=channels)
                logger.info("Opus decoder initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Opus decoder: {e}")
                self.decoder = None
        else:
            self.decoder = None
            logger.warning("Opus decoder not available - will use mock decompression")
    
    def decode(self, opus_data: bytes) -> bytes:
        """Decode Opus audio to PCM
        
        Args:
            opus_data: Compressed Opus data
            
        Returns:
            Raw PCM audio (16-bit)
        """
        if not self.decoder:
            # Mock decompression
            if opus_data.startswith(b'MOCK'):
                compressed = opus_data[4:]
                # The mock encoder kept int16 samples, just decimated
                decimated_samples = np.frombuffer(compressed, dtype=np.int16)
                # Interpolate back to original size by repeating each sample 10 times
                expanded = np.repeat(decimated_samples, 10)[:self.frame_size]
                return expanded.tobytes()
            else:
                # Return silence if we can't decode
                return np.zeros(self.frame_size, dtype=np.int16).tobytes()
        
        try:
            # Decode with opuslib
            pcm_data = self.decoder.decode(opus_data, self.frame_size)
            return pcm_data
        except Exception as e:
            logger.error(f"Opus decoding failed: {e}")
            # Return silence on decode error
            return np.zeros(self.frame_size * self.channels, dtype=np.int16).tobytes()
    
    def decode_lost_packet(self) -> bytes:
        """Generate audio for a lost packet using PLC (Packet Loss Concealment)
        
        Returns:
            PCM audio to fill the gap
        """
        if self.decoder:
            try:
                # Opus decoder can generate comfort noise for lost packets
                return self.decoder.decode(b'', self.frame_size)
            except:
                pass
        
        # Fallback: return silence
        return np.zeros(self.frame_size * self.channels, dtype=np.int16).tobytes()
    
    def reset(self):
        """Reset decoder state"""
        if self.decoder:
            self.__init__(self.sample_rate, self.channels)


class AudioProcessor:
    """Handles audio processing pipeline with Opus codec"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from .audio_config import opus_config
        
        self.encoder = OpusEncoder(
            sample_rate=opus_config.sample_rate,
            channels=opus_config.channels,
            bitrate=opus_config.bitrate,
            complexity=opus_config.complexity,
            use_dtx=opus_config.use_dtx,
            use_inband_fec=opus_config.use_inband_fec
        )
        
        self.decoder = OpusDecoder(
            sample_rate=opus_config.sample_rate,
            channels=opus_config.channels
        )
        
        self.tx_sequence = 0
        self.rx_sequence = 0
        self.metrics = {
            'frames_encoded': 0,
            'frames_decoded': 0,
            'encoding_errors': 0,
            'decoding_errors': 0,
            'bytes_saved': 0
        }
    
    def process_outgoing_audio(self, pcm_data: bytes, timestamp_ms: int) -> AudioFrame:
        """Process outgoing audio: PCM → Opus
        
        Args:
            pcm_data: Raw PCM audio
            timestamp_ms: Timestamp in milliseconds
            
        Returns:
            Compressed audio frame ready for transmission
        """
        try:
            # Encode to Opus
            opus_data = self.encoder.encode(pcm_data)
            
            # Track bandwidth savings
            self.metrics['bytes_saved'] += len(pcm_data) - len(opus_data)
            self.metrics['frames_encoded'] += 1
            
            # Create frame with metadata
            frame = AudioFrame(
                data=opus_data,
                timestamp_ms=timestamp_ms,
                sequence_number=self.tx_sequence
            )
            
            self.tx_sequence += 1
            
            logger.debug(f"Encoded frame {frame.sequence_number}: {len(pcm_data)} → {len(opus_data)} bytes")
            
            return frame
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            self.metrics['encoding_errors'] += 1
            raise
    
    def process_incoming_audio(self, frame: AudioFrame) -> bytes:
        """Process incoming audio: Opus → PCM
        
        Args:
            frame: Compressed audio frame from network
            
        Returns:
            Decoded PCM audio
        """
        try:
            # Check for packet loss (only if we've received packets before)
            if self.rx_sequence > 0:
                expected_seq = self.rx_sequence
                if frame.sequence_number > expected_seq:
                    # Packets were lost
                    lost_count = frame.sequence_number - expected_seq
                    if lost_count > 0 and lost_count < 1000:  # Sanity check
                        logger.warning(f"Lost {lost_count} packets")
                        
                        # Generate audio for lost packets
                        for _ in range(min(lost_count, 10)):  # Limit PLC generation
                            self.decoder.decode_lost_packet()
            
            # Decode the frame
            pcm_data = self.decoder.decode(frame.data)
            
            self.metrics['frames_decoded'] += 1
            self.rx_sequence = frame.sequence_number + 1
            
            return pcm_data
            
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            self.metrics['decoding_errors'] += 1
            # Return silence on error
            return self.decoder.decode_lost_packet()
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        total_saved = self.metrics['bytes_saved']
        total_encoded = self.metrics['frames_encoded']
        
        if total_encoded > 0:
            # Each uncompressed frame is 1920 bytes (960 samples * 2 bytes)
            original_size = total_encoded * 1920
            compressed_size = original_size - total_saved
            compression_ratio = 1 - (compressed_size / original_size) if original_size > 0 else 0
        else:
            compression_ratio = 0
        
        return {
            'compression_ratio': compression_ratio,
            'bandwidth_saved_kb': total_saved / 1024,
            'frames_processed': total_encoded,
            'errors': self.metrics['encoding_errors'] + self.metrics['decoding_errors']
        }