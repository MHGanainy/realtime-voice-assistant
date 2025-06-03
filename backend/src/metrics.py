"""
Performance metrics collection for the voice assistant
Tracks latencies, throughput, and quality metrics
"""
import time
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class LatencyMetrics:
    """Track latency at each stage of the pipeline"""
    # Audio pipeline stages
    audio_receive_to_vad: List[float] = field(default_factory=list)
    vad_to_stt_start: List[float] = field(default_factory=list)
    stt_processing: List[float] = field(default_factory=list)
    stt_to_llm_start: List[float] = field(default_factory=list)
    llm_processing: List[float] = field(default_factory=list)
    llm_to_tts_start: List[float] = field(default_factory=list)
    tts_processing: List[float] = field(default_factory=list)
    tts_to_send: List[float] = field(default_factory=list)
    
    # End-to-end
    total_latency: List[float] = field(default_factory=list)
    
    def add_measurement(self, stage: str, latency_ms: float):
        """Add a latency measurement for a specific stage"""
        if hasattr(self, stage):
            getattr(self, stage).append(latency_ms)
    
    def get_percentiles(self, stage: str) -> Dict[str, float]:
        """Get p50, p95, p99 for a stage"""
        if not hasattr(self, stage):
            return {}
        
        values = getattr(self, stage)
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "p50": sorted_values[int(n * 0.5)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
            "mean": statistics.mean(values),
            "count": n
        }


@dataclass
class AudioMetrics:
    """Track audio quality metrics"""
    packets_sent: int = 0
    packets_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    packet_loss_ratio: float = 0.0
    
    # Audio quality
    vad_activations: int = 0
    audio_glitches: int = 0
    encoding_failures: int = 0
    
    def calculate_packet_loss(self):
        """Calculate packet loss ratio"""
        if self.packets_sent > 0:
            self.packet_loss_ratio = 1 - (self.packets_received / self.packets_sent)


class MetricsCollector:
    """Central metrics collection for the voice assistant"""
    
    def __init__(self):
        self.latency = LatencyMetrics()
        self.audio = AudioMetrics()
        self.start_time = time.time()
        
        # Track active operations
        self.active_operations: Dict[str, float] = {}
    
    def start_operation(self, operation_id: str) -> float:
        """Start timing an operation"""
        start_time = time.time() * 1000  # Convert to ms
        self.active_operations[operation_id] = start_time
        return start_time
    
    def end_operation(self, operation_id: str, stage: str) -> Optional[float]:
        """End timing an operation and record latency"""
        if operation_id not in self.active_operations:
            return None
        
        start_time = self.active_operations.pop(operation_id)
        latency_ms = (time.time() * 1000) - start_time
        self.latency.add_measurement(stage, latency_ms)
        return latency_ms
    
    def record_audio_packet(self, direction: str, size: int):
        """Record audio packet metrics"""
        if direction == "sent":
            self.audio.packets_sent += 1
            self.audio.bytes_sent += size
        elif direction == "received":
            self.audio.packets_received += 1
            self.audio.bytes_received += size
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        uptime = time.time() - self.start_time
        
        # Calculate latency percentiles for each stage
        latency_summary = {}
        for stage in [
            "audio_receive_to_vad", "vad_to_stt_start", "stt_processing",
            "stt_to_llm_start", "llm_processing", "llm_to_tts_start",
            "tts_processing", "tts_to_send", "total_latency"
        ]:
            percentiles = self.latency.get_percentiles(stage)
            if percentiles:
                latency_summary[stage] = percentiles
        
        # Calculate audio metrics
        self.audio.calculate_packet_loss()
        
        return {
            "uptime_seconds": uptime,
            "latency_ms": latency_summary,
            "audio": {
                "packets_sent": self.audio.packets_sent,
                "packets_received": self.audio.packets_received,
                "bytes_sent": self.audio.bytes_sent,
                "bytes_received": self.audio.bytes_received,
                "packet_loss_ratio": self.audio.packet_loss_ratio,
                "vad_activations": self.audio.vad_activations
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def start_reporting(self, interval_seconds: int = 10):
        """Start periodic metrics reporting"""
        while True:
            await asyncio.sleep(interval_seconds)
            summary = self.get_summary()
            # TODO: Send to monitoring system (Prometheus, CloudWatch, etc.)
            print(f"Metrics: {summary}")


# Global metrics instance
metrics = MetricsCollector()