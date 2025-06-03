"""
Audio configuration matching the state-of-the-art spec
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpusConfig:
    """Opus encoder configuration for low-latency voice"""
    sample_rate: int = 48000  # Always capture at 48kHz
    frame_duration_ms: int = 20  # 20ms frames for low latency
    channels: int = 1  # Mono for voice
    bitrate: int = 32000  # 32kbps is plenty for voice
    bitrate_mode: str = "voip"  # Optimized for voice
    complexity: int = 5  # Balance between quality and CPU (0-10)
    packet_loss_perc: int = 10  # Expected packet loss for FEC
    use_inband_fec: bool = True  # Forward error correction
    use_dtx: bool = True  # Discontinuous transmission (silence suppression)
    
    @property
    def frame_size(self) -> int:
        """Number of samples per frame"""
        return int(self.sample_rate * self.frame_duration_ms / 1000)


@dataclass
class BufferConfig:
    """Audio buffer configuration for streaming"""
    chunk_duration_ms: int = 20  # Match Opus frame size
    max_buffer_ms: int = 120  # ~6 frames max latency
    jitter_buffer_ms: int = 60  # Adaptive jitter buffer
    
    @property
    def chunk_size_samples(self) -> int:
        """Samples per chunk at 48kHz"""
        return int(48000 * self.chunk_duration_ms / 1000)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    energy_threshold: float = 0.01  # RMS energy threshold
    zero_crossing_threshold: int = 25  # Zero crossings per frame
    speech_start_ms: int = 200  # Time before confirming speech start
    speech_end_ms: int = 800  # Time before confirming speech end
    frame_duration_ms: int = 20  # Analysis frame size


@dataclass
class TransportConfig:
    """Network transport configuration"""
    protocol: str = "websocket"  # Start with WS, upgrade to WebTransport later
    ping_interval_s: int = 20  # Keep-alive ping
    ping_timeout_s: int = 60  # Connection timeout
    reconnect_delay_ms: int = 1000  # Initial reconnect delay
    max_reconnect_delay_ms: int = 30000  # Max backoff
    
    # WebTransport specific (when we upgrade)
    enable_datagrams: bool = True  # Use unreliable datagrams for audio
    enable_streams: bool = True  # Use reliable streams for control


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration"""
    enable_metrics: bool = True
    metrics_interval_s: int = 10  # Collect metrics every 10s
    latency_buckets_ms: list = None  # Histogram buckets
    
    def __post_init__(self):
        if self.latency_buckets_ms is None:
            # Buckets: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s
            self.latency_buckets_ms = [10, 25, 50, 100, 250, 500, 1000, 2500]


# Global configuration instances
opus_config = OpusConfig()
buffer_config = BufferConfig()
vad_config = VADConfig()
transport_config = TransportConfig()
metrics_config = MetricsConfig()


def get_audio_constraints() -> dict:
    """Get media constraints for getUserMedia"""
    return {
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
            "sampleRate": opus_config.sample_rate,
            "channelCount": opus_config.channels,
            "latency": 0,  # Request lowest latency
            "sampleSize": 16  # 16-bit samples
        }
    }