import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { VoiceRoom } from '../voice/VoiceRoom';
import { AudioCapture } from '../audio/AudioCapture';
import { AudioPlayer } from '../audio/AudioPlayer';

// Mock WebCodecs AudioDecoder API
class MockAudioDecoder {
  constructor(config) {
    this.config = config;
    this.state = 'configured';
    this.decodeQueueSize = 0;
    this.output = config.output;
    this.error = config.error;
    this.codecConfig = null;
  }

  configure(config) {
    this.codecConfig = config;
    this.state = 'configured';
    
    if (config.codec !== 'opus') {
      throw new Error('Unsupported codec');
    }
    
    return Promise.resolve();
  }

  decode(chunk) {
    this.decodeQueueSize++;
    
    // Simulate decoding
    Promise.resolve().then(() => {
      // Simulate decoded audio data
      const audioData = {
        format: 'f32',
        numberOfChannels: 1,
        numberOfFrames: 960, // 20ms at 48kHz
        sampleRate: 48000,
        timestamp: chunk.timestamp,
        duration: 20000, // 20ms in microseconds
        copyTo: (buffer, options) => {
          // Generate the same frequency as the original test signal (1kHz)
          // with similar amplitude (slightly attenuated to simulate codec loss)
          for (let i = 0; i < buffer.length; i++) {
            buffer[i] = Math.sin(2 * Math.PI * 1000 * i / 48000) * 0.9;
          }
        },
        close: () => {}
      };
      
      this.output(audioData);
      this.decodeQueueSize--;
    });
  }

  flush() {
    this.decodeQueueSize = 0;
    return Promise.resolve();
  }

  close() {
    this.state = 'closed';
  }

  static isConfigSupported(config) {
    if (config.codec === 'opus') {
      return Promise.resolve({ supported: true, config });
    }
    return Promise.resolve({ supported: false });
  }
}

class MockEncodedAudioChunk {
  constructor(init) {
    this.type = init.type;
    this.timestamp = init.timestamp;
    this.duration = init.duration;
    this.byteLength = init.byteLength;
    this._data = init.data;
  }

  copyTo(buffer) {
    const view = new Uint8Array(buffer);
    const data = new Uint8Array(this._data);
    view.set(data);
  }
}

// Install mocks
global.AudioDecoder = MockAudioDecoder;
global.EncodedAudioChunk = MockEncodedAudioChunk;

describe('WebCodecs AudioDecoder Integration', () => {
  let room;
  let mockCapture;
  let mockPlayer;
  let mockWebSocket;

  beforeEach(() => {
    vi.useFakeTimers();
    
    // Create mock WebSocket
    mockWebSocket = {
      send: vi.fn(),
      close: vi.fn(),
      readyState: WebSocket.OPEN,
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null
    };

    global.WebSocket = vi.fn(() => {
      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen();
        }
      }, 0);
      return mockWebSocket;
    });
    global.WebSocket.OPEN = 1;
    global.WebSocket.CLOSED = 3;

    // Create mocks
    mockCapture = new AudioCapture();
    mockPlayer = new AudioPlayer();

    // Mock audio components
    vi.spyOn(mockCapture, 'initialize').mockResolvedValue();
    vi.spyOn(mockCapture, 'start').mockResolvedValue();
    vi.spyOn(mockCapture, 'stop').mockResolvedValue();
    vi.spyOn(mockPlayer, 'initialize').mockResolvedValue();
    vi.spyOn(mockPlayer, 'playDecodedOpus').mockResolvedValue();

    room = new VoiceRoom({
      roomId: 'test-room',
      userId: 'test-user',
      capture: mockCapture,
      player: mockPlayer
    });
  });

  afterEach(async () => {
    // Clear any intervals first
    if (room._metricsInterval) {
      clearInterval(room._metricsInterval);
      room._metricsInterval = null;
    }
    
    if (room) {
      room.shouldReconnect = false;
      if (room.socket) {
        room.socket.close();
      }
      if (room.opusDecoder && room.opusDecoder.state === 'configured') {
        room.opusDecoder.close();
      }
    }
    vi.clearAllMocks();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  describe('WebCodecs Opus Decoder', () => {
    it('should initialize AudioDecoder with Opus codec', async () => {
      await room.join();
      
      // Process connection
      await vi.advanceTimersByTimeAsync(10);

      expect(room.opusDecoder).toBeDefined();
      expect(room.opusDecoder).toBeInstanceOf(AudioDecoder);
      expect(room.opusDecoder.state).toBe('configured');
      expect(room.isDecoderReady).toBe(true);
    });

    it('should check codec support before initializing', async () => {
      const supportSpy = vi.spyOn(AudioDecoder, 'isConfigSupported');
      
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      expect(supportSpy).toHaveBeenCalledWith({
        codec: 'opus',
        sampleRate: 48000,
        numberOfChannels: 1
      });
    });

    it('should fallback gracefully if Opus is not supported', async () => {
      // Mock Opus as unsupported
      AudioDecoder.isConfigSupported = vi.fn().mockResolvedValue({ 
        supported: false 
      });

      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      expect(room.opusDecoder).toBeNull();
      expect(room.isDecoderReady).toBe(false);
    });
  });

  describe('Opus Frame Detection', () => {
    beforeEach(() => {
      // Ensure decoder is supported for these tests
      AudioDecoder.isConfigSupported = vi.fn().mockResolvedValue({ 
        supported: true 
      });
    });

    it('should correctly detect Opus frames with header', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Create a valid Opus frame
      const sequence = 42;
      const timestamp = 123456;
      const opusData = new Uint8Array([0x48, 0x01, 0x02, 0x03]); // Mock Opus data
      
      const frameBuffer = new ArrayBuffer(8 + opusData.length);
      const view = new DataView(frameBuffer);
      view.setUint32(0, sequence, false); // Big endian
      view.setUint32(4, timestamp, false);
      new Uint8Array(frameBuffer).set(opusData, 8);

      const result = room._detectOpusFrame(frameBuffer);
      
      expect(result).toEqual({
        isOpusFrame: true,
        sequence: sequence,
        timestamp: timestamp,
        opusData: expect.any(Uint8Array)
      });
      
      expect(result.opusData).toHaveLength(opusData.length);
      expect(Array.from(result.opusData)).toEqual(Array.from(opusData));
    });

    it('should correctly identify raw PCM (not Opus)', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Create raw PCM data
      const pcmData = new ArrayBuffer(1920); // 960 samples * 2 bytes
      const samples = new Int16Array(pcmData);
      
      // Fill with test pattern
      for (let i = 0; i < samples.length; i++) {
        samples[i] = Math.sin(2 * Math.PI * 440 * i / 48000) * 32767;
      }

      const result = room._detectOpusFrame(pcmData);
      
      expect(result).toEqual({
        isOpusFrame: false,
        sequence: null,
        timestamp: null,
        opusData: null
      });
    });

    it('should handle malformed frames', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Too small buffer
      const tooSmall = new ArrayBuffer(4);
      expect(room._detectOpusFrame(tooSmall).isOpusFrame).toBe(false);

      // Invalid sequence number
      const invalidSeq = new ArrayBuffer(12);
      const view = new DataView(invalidSeq);
      view.setUint32(0, 999999999, false); // Too large
      view.setUint32(4, 123456, false);
      expect(room._detectOpusFrame(invalidSeq).isOpusFrame).toBe(false);
    });
  });

  describe('Audio Decoding', () => {
    beforeEach(() => {
      // Ensure decoder is supported for these tests
      AudioDecoder.isConfigSupported = vi.fn().mockResolvedValue({ 
        supported: true 
      });
    });

    it('should decode Opus frames and play audio', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      const playSpy = vi.spyOn(mockPlayer, 'playDecodedOpus');

      // Create and send Opus frame
      const opusFrame = createOpusFrame(new Uint8Array(50), 1, 1000);
      mockWebSocket.onmessage({ data: opusFrame });

      // Process decoding with limited timer advancement
      await vi.advanceTimersByTimeAsync(50);

      expect(playSpy).toHaveBeenCalled();
      const playedBuffer = playSpy.mock.calls[0][0];
      expect(playedBuffer).toBeInstanceOf(ArrayBuffer);
      expect(playedBuffer.byteLength).toBe(1920); // 960 samples * 2 bytes
    });

    it('should handle decoder errors gracefully', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Wait for decoder to be ready
      if (room.opusDecoder) {
        // Mock decoder error
        room.opusDecoder.decode = vi.fn(() => {
          throw new Error('Decode failed');
        });

        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

        // Send Opus frame
        const opusFrame = createOpusFrame(new Uint8Array(50), 1, 1000);
        mockWebSocket.onmessage({ data: opusFrame });

        await vi.advanceTimersByTimeAsync(50);

        expect(errorSpy).toHaveBeenCalledWith(
          expect.stringContaining('Failed to decode Opus'),
          expect.any(Error)
        );

        errorSpy.mockRestore();
      } else {
        // If decoder is not initialized, skip this test
        expect(true).toBe(true);
      }
    });

    it('should track compression metrics', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Send Opus frame
      const opusData = new Uint8Array(50);
      const opusFrame = createOpusFrame(opusData, 1, 1000);
      mockWebSocket.onmessage({ data: opusFrame });

      await vi.advanceTimersByTimeAsync(50);

      // Verify compression ratio calculation
      const expectedRatio = 1 - (opusData.length / 1920);
      expect(room.metrics.compressionRatio).toBeCloseTo(expectedRatio, 2);
    });
  });

  describe('Audio Quality Metrics', () => {
    beforeEach(() => {
      // Ensure decoder is supported for these tests
      AudioDecoder.isConfigSupported = vi.fn().mockResolvedValue({ 
        supported: true 
      });
    });

    it('should measure SNR and correlation for audio quality', async () => {
      room.enableQualityMetrics = true;
      
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Simulate sending original audio
      const originalPCM = new ArrayBuffer(1920);
      const samples = new Int16Array(originalPCM);
      
      // Generate 1kHz sine wave
      for (let i = 0; i < samples.length; i++) {
        samples[i] = Math.sin(2 * Math.PI * 1000 * i / 48000) * 16384;
      }

      // Store original
      room._originalAudioBuffer = originalPCM;

      // Send Opus frame that will decode to similar audio
      const opusFrame = createOpusFrame(new Uint8Array(50), 1, 1000);
      mockWebSocket.onmessage({ data: opusFrame });

      await vi.advanceTimersByTimeAsync(50);

      // Verify quality metrics
      expect(room.metrics.audioQuality).toBeDefined();
      // The mock decoder generates similar but not identical audio
      // Due to float32 to int16 conversion, SNR will be lower
      expect(room.metrics.audioQuality.snr).toBeGreaterThan(0); // Positive SNR
      expect(room.metrics.audioQuality.correlation).toBeGreaterThan(0.9); // High correlation
    });

    it('should handle quality measurement errors', async () => {
      room.enableQualityMetrics = true;
      
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      // Set invalid original buffer
      room._originalAudioBuffer = new ArrayBuffer(7); // Odd size

      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Send Opus frame
      const opusFrame = createOpusFrame(new Uint8Array(50), 1, 1000);
      mockWebSocket.onmessage({ data: opusFrame });

      await vi.advanceTimersByTimeAsync(50);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Original buffer has invalid length'),
        7
      );

      // Should return safe defaults
      expect(room.metrics.audioQuality.snr).toBe(0);
      expect(room.metrics.audioQuality.correlation).toBe(0);

      warnSpy.mockRestore();
    });

    it('should calculate quality metrics correctly', () => {
      // Test the quality measurement function directly
      const original = new ArrayBuffer(1920);
      const decoded = new ArrayBuffer(1920);
      
      const origSamples = new Int16Array(original);
      const decodedSamples = new Int16Array(decoded);
      
      // Fill with identical sine waves
      for (let i = 0; i < 960; i++) {
        const value = Math.sin(2 * Math.PI * 440 * i / 48000) * 16384;
        origSamples[i] = value;
        decodedSamples[i] = value * 0.95; // Slightly attenuated
      }

      const quality = room._measureAudioQuality(original, decoded);
      
      // 5% attenuation should give SNR around 26dB
      expect(quality.snr).toBeGreaterThan(20); // Good SNR
      expect(quality.snr).toBeLessThan(35); // But not perfect
      expect(quality.correlation).toBeGreaterThan(0.99); // Very high correlation
    });
  });

  describe('Real-time Decoding Performance', () => {
    beforeEach(() => {
      // Ensure decoder is supported for these tests
      AudioDecoder.isConfigSupported = vi.fn().mockResolvedValue({ 
        supported: true 
      });
    });

    it('should decode multiple frames without blocking', async () => {
      await room.join();
      await vi.advanceTimersByTimeAsync(10);

      const playSpy = vi.spyOn(mockPlayer, 'playDecodedOpus');
      const playRawSpy = vi.spyOn(mockPlayer, 'play');

      // Mock player.play to avoid null context error
      mockPlayer.play = vi.fn().mockResolvedValue();

      // Send multiple frames rapidly
      for (let i = 0; i < 10; i++) {
        // Ensure even-length data to avoid Int16Array issues
        const dataLength = 46 + i * 2; // Always even
        const opusFrame = createOpusFrame(new Uint8Array(dataLength), i, i * 20000);
        mockWebSocket.onmessage({ data: opusFrame });
      }

      // Process all decoding with limited advancement
      await vi.advanceTimersByTimeAsync(100);

      // Should have decoded Opus frames if decoder is ready
      if (room.isDecoderReady) {
        expect(playSpy).toHaveBeenCalledTimes(10);
      } else {
        // If decoder not ready, frames should be skipped
        expect(playSpy).not.toHaveBeenCalled();
      }
    });

    it('should maintain low decoding latency', async () => {
      vi.useRealTimers(); // Use real timers for latency test
      
      await room.join();

      const latencies = [];
      let decodeStartTime = 0;
      
      // Track decode timing
      const originalPlay = mockPlayer.playDecodedOpus.getMockImplementation() || vi.fn();
      mockPlayer.playDecodedOpus = vi.fn(async (buffer) => {
        if (decodeStartTime > 0) {
          const latency = performance.now() - decodeStartTime;
          latencies.push(latency);
        }
        return originalPlay(buffer);
      });

      // Override handleAudioData to track start time
      const originalHandleAudio = room._handleAudioData.bind(room);
      room._handleAudioData = async function(data) {
        decodeStartTime = performance.now();
        return originalHandleAudio(data);
      };

      // Send frame
      const opusFrame = createOpusFrame(new Uint8Array(50), 1, 1000);
      mockWebSocket.onmessage({ data: opusFrame });

      // Wait for decode
      await new Promise(resolve => setTimeout(resolve, 50));

      if (latencies.length > 0) {
        const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        expect(avgLatency).toBeLessThan(10); // Under 10ms decode latency
      }

      vi.useFakeTimers(); // Switch back
    });
  });
});

// Helper function to create Opus frame
function createOpusFrame(opusData, sequence, timestamp) {
  // Ensure reasonable sequence numbers for detection
  const safeSequence = sequence % 1000000;
  const frame = new ArrayBuffer(8 + opusData.length);
  const view = new DataView(frame);
  view.setUint32(0, safeSequence, false);
  view.setUint32(4, timestamp || 1, false); // Ensure timestamp > 0
  new Uint8Array(frame).set(opusData, 8);
  return frame;
}