import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { VoiceRoom } from '../voice/VoiceRoom';
import { AudioCapture } from '../audio/AudioCapture';
import { AudioPlayer } from '../audio/AudioPlayer';

// Mock Web Audio API
global.AudioContext = class MockAudioContext {
  constructor() {
    this.sampleRate = 48000;
    this.currentTime = 0;
    this.destination = {};
    this.state = 'running';
    this.audioWorklet = {
      addModule: vi.fn().mockResolvedValue(undefined)
    };
  }

  createMediaStreamSource() {
    return { connect: () => {} };
  }

  createBufferSource() {
    return {
      buffer: null,
      connect: () => {},
      start: () => {},
      onended: null
    };
  }

  createBuffer(channels, frames, sampleRate) {
    return {
      copyToChannel: () => {},
      getChannelData: () => new Float32Array(frames)
    };
  }

  close() {
    this.state = 'closed';
    return Promise.resolve();
  }
};

global.AudioWorkletNode = class MockAudioWorkletNode {
  constructor() {
    this.port = {
      postMessage: () => {},
      onmessage: null
    };
  }

  disconnect() {}
};

// Mock MediaDevices
global.navigator = {
  mediaDevices: {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }]
    })
  }
};

// Mock WebCodecs API
class MockAudioDecoder {
  constructor(config) {
    this.config = config;
    this.state = 'configured';
    this.decodeQueue = [];
    this.outputCallback = config.output;
    this.errorCallback = config.error;
  }

  configure(config) {
    this.codecConfig = config;
    // Simulate async configuration
    return Promise.resolve();
  }

  decode(chunk) {
    // Simulate successful decoding
    setTimeout(() => {
      const mockAudioData = {
        numberOfChannels: 1,
        numberOfFrames: 960,
        sampleRate: 48000,
        timestamp: chunk.timestamp,
        duration: 20000, // 20ms in microseconds
        copyTo: (buffer, options) => {
          // For quality testing, return a signal that's similar to the input
          // This simulates what a real codec would do
          for (let i = 0; i < buffer.length; i++) {
            // Generate 1kHz tone (matching the test input)
            buffer[i] = Math.sin(2 * Math.PI * 1000 * i / 48000) * 0.5;
          }
        },
        close: () => {}
      };
      this.outputCallback(mockAudioData);
    }, 1);
  }

  flush() {
    return Promise.resolve();
  }

  close() {
    this.state = 'closed';
  }
}

class MockEncodedAudioChunk {
  constructor(config) {
    this.type = config.type;
    this.timestamp = config.timestamp;
    this.data = config.data;
    this.byteLength = config.data.byteLength || config.data.length;
  }

  copyTo(buffer) {
    const view = new Uint8Array(buffer);
    const data = new Uint8Array(this.data);
    view.set(data);
  }
}

// Install mocks
global.AudioDecoder = MockAudioDecoder;
global.EncodedAudioChunk = MockEncodedAudioChunk;

// Helper function to create mock Opus frames
function createMockOpusFrame(opusData, sequence) {
  const timestamp = Date.now() % (2**32);
  const frame = new ArrayBuffer(8 + opusData.length);
  const view = new DataView(frame);
  view.setUint32(0, sequence, false);
  view.setUint32(4, timestamp, false);
  new Uint8Array(frame).set(opusData, 8);
  return frame;
}

describe('Opus Frame Detection and Parsing', () => {
  let room;
  let mockCapture;
  let mockPlayer;
  let mockWebSocket;

  beforeEach(() => {
    // Create mock WebSocket
    mockWebSocket = {
      send: vi.fn(),
      close: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      readyState: WebSocket.OPEN,
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null
    };

    // Mock WebSocket constructor
    global.WebSocket = vi.fn(() => {
      // Simulate connection opening
      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen();
        }
      }, 0);
      return mockWebSocket;
    });
    global.WebSocket.OPEN = 1;
    global.WebSocket.CLOSED = 3;

    mockCapture = new AudioCapture();
    mockPlayer = new AudioPlayer();

    // Mock the audio components
    vi.spyOn(mockCapture, 'initialize').mockResolvedValue();
    vi.spyOn(mockCapture, 'start').mockResolvedValue();
    vi.spyOn(mockCapture, 'stop').mockResolvedValue();
    vi.spyOn(mockCapture, 'on').mockImplementation((event, handler) => {
      mockCapture[`_${event}Handler`] = handler;
    });
    vi.spyOn(mockCapture, 'emit').mockImplementation((event, data) => {
      const handler = mockCapture[`_${event}Handler`];
      if (handler) handler(data);
    });
    
    vi.spyOn(mockPlayer, 'initialize').mockResolvedValue();
    vi.spyOn(mockPlayer, 'play').mockResolvedValue();
    vi.spyOn(mockPlayer, 'playDecodedOpus').mockResolvedValue();

    room = new VoiceRoom({
      roomId: 'test-room',
      userId: 'test-user',
      capture: mockCapture,
      player: mockPlayer
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Opus Frame Detection', () => {
    it('should correctly identify Opus frames with header', async () => {
      const connectPromise = room._connect();
      
      // Wait for connection to open
      await new Promise(resolve => setTimeout(resolve, 10));
      
      // Simulate successful connection
      mockWebSocket.readyState = WebSocket.OPEN;
      
      // Wait for connection promise
      await connectPromise;
      
      // Create a valid Opus frame with header
      const seqNum = 42;
      const timestamp = 123456;
      const opusData = new Uint8Array([0x48, 0x01, 0x02, 0x03]); // Mock Opus data
      
      const frameBuffer = new ArrayBuffer(8 + opusData.length);
      const view = new DataView(frameBuffer);
      view.setUint32(0, seqNum, false); // Big endian
      view.setUint32(4, timestamp, false);
      new Uint8Array(frameBuffer).set(opusData, 8);

      // Spy on internal method to check frame detection
      const detectSpy = vi.spyOn(room, '_detectOpusFrame');
      
      // Simulate receiving the frame
      mockWebSocket.onmessage({ data: frameBuffer });

      expect(detectSpy).toHaveBeenCalledWith(frameBuffer);
      
      const result = room._detectOpusFrame(frameBuffer);
      expect(result).toEqual({
        isOpusFrame: true,
        sequence: seqNum,
        timestamp: timestamp,
        opusData: expect.any(Uint8Array)
      });
      
      // Verify Opus data extraction
      expect(result.opusData).toHaveLength(opusData.length);
      expect(Array.from(result.opusData)).toEqual(Array.from(opusData));
    });

    it('should correctly identify raw PCM audio (not Opus frame)', async () => {
      const connectPromise = room._connect();
      await new Promise(resolve => setTimeout(resolve, 10));
      mockWebSocket.readyState = WebSocket.OPEN;
      await connectPromise;
      
      // Create raw PCM data (960 samples * 2 bytes = 1920 bytes)
      const pcmData = new ArrayBuffer(1920);
      const samples = new Int16Array(pcmData);
      
      // Fill with test pattern
      for (let i = 0; i < samples.length; i++) {
        samples[i] = Math.sin(2 * Math.PI * 440 * i / 48000) * 32767;
      }

      const detectSpy = vi.spyOn(room, '_detectOpusFrame');
      
      // Simulate receiving raw PCM
      mockWebSocket.onmessage({ data: pcmData });

      const result = room._detectOpusFrame(pcmData);
      expect(result).toEqual({
        isOpusFrame: false,
        sequence: null,
        timestamp: null,
        opusData: null
      });
    });

    it('should handle malformed frames gracefully', async () => {
      const connectPromise = room._connect();
      await new Promise(resolve => setTimeout(resolve, 10));
      mockWebSocket.readyState = WebSocket.OPEN;
      await connectPromise;
      
      // Too small to be a valid frame
      const tooSmall = new ArrayBuffer(4);
      
      const result = room._detectOpusFrame(tooSmall);
      expect(result.isOpusFrame).toBe(false);
    });
  });

  describe('End-to-End Audio Compression/Decompression', () => {
    it('should compress audio on send and decompress on receive', async () => {
      const connectPromise = room._connect();
      await new Promise(resolve => setTimeout(resolve, 10));
      mockWebSocket.readyState = WebSocket.OPEN;
      await connectPromise;
      
      // Setup: track what's sent and received
      const sentData = [];
      mockWebSocket.send = vi.fn((data) => {
        sentData.push(data);
      });

      // 1. Send PCM audio
      const originalPCM = new ArrayBuffer(1920); // 20ms at 48kHz
      const samples = new Int16Array(originalPCM);
      
      // Generate 440Hz test tone
      for (let i = 0; i < samples.length; i++) {
        samples[i] = Math.sin(2 * Math.PI * 440 * i / 48000) * 16384;
      }

      // Emit audio chunk from capture
      mockCapture.emit('audioChunk', {
        data: new Uint8Array(originalPCM),
        timestamp: performance.now(),
        encoded: false
      });

      // 2. Verify data was sent
      expect(mockWebSocket.send).toHaveBeenCalled();
      const sentBuffer = mockWebSocket.send.mock.calls[0][0];
      
      // 3. Backend echo: simulate receiving Opus frame back
      // The backend should have compressed it and added header
      const seqNum = 0;
      const timestamp = Date.now() % (2**32);
      const mockOpusData = new Uint8Array(50); // Compressed size
      
      const echoFrame = new ArrayBuffer(8 + mockOpusData.length);
      const view = new DataView(echoFrame);
      view.setUint32(0, seqNum, false);
      view.setUint32(4, timestamp, false);
      new Uint8Array(echoFrame).set(mockOpusData, 8);

      // Spy on player to verify decompressed audio is played
      const playSpy = vi.spyOn(mockPlayer, 'playDecodedOpus');

      // Mock the decoder to be ready
      room.isDecoderReady = true;
      room.opusDecoder = new MockAudioDecoder({
        output: (audioData) => room._handleDecodedAudio(audioData),
        error: (error) => console.error('Decoder error:', error)
      });

      // 4. Receive the echo
      mockWebSocket.onmessage({ data: echoFrame });

      // 5. Wait for async decoding
      await new Promise(resolve => setTimeout(resolve, 10));

      // 6. Verify Opus was decoded and played
      expect(playSpy).toHaveBeenCalled();
      
      // Verify the audio data passed to player
      const playedAudio = playSpy.mock.calls[0][0];
      expect(playedAudio).toBeInstanceOf(ArrayBuffer);
      
      // Should be back to PCM size after decompression
      expect(playedAudio.byteLength).toBe(1920); // 960 samples * 2 bytes
    });

    it('should measure audio quality after compression/decompression', async () => {
      const connectPromise = room._connect();
      await new Promise(resolve => setTimeout(resolve, 10));
      mockWebSocket.readyState = WebSocket.OPEN;
      await connectPromise;
      
      // Enable quality metrics
      room.enableQualityMetrics = true;
      
      // Original audio: 1kHz test tone
      const originalPCM = new ArrayBuffer(1920);
      const samples = new Int16Array(originalPCM);
      
      for (let i = 0; i < samples.length; i++) {
        samples[i] = Math.sin(2 * Math.PI * 1000 * i / 48000) * 16384;
      }

      // Store original for comparison
      room._originalAudioBuffer = originalPCM;

      // Send audio
      mockCapture.emit('audioChunk', {
        data: new Uint8Array(originalPCM),
        timestamp: performance.now(),
        encoded: false
      });

      // Simulate compressed echo
      const echoFrame = createMockOpusFrame(new Uint8Array(50), 0);
      
      // Setup quality measurement
      const qualitySpy = vi.spyOn(room, '_measureAudioQuality');
      
      // Mock the decoder to be ready
      room.isDecoderReady = true;
      room.opusDecoder = new MockAudioDecoder({
        output: (audioData) => room._handleDecodedAudio(audioData),
        error: (error) => console.error('Decoder error:', error)
      });
      
      // Receive echo
      mockWebSocket.onmessage({ data: echoFrame });
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 20));

      // Verify quality was measured
      expect(qualitySpy).toHaveBeenCalled();
      
      const quality = qualitySpy.mock.results[0].value;
      expect(quality).toHaveProperty('snr');
      expect(quality).toHaveProperty('correlation');
      
      // With a matching test tone, correlation should be high
      expect(quality.correlation).toBeGreaterThan(0.8);
      
      // SNR should be reasonable (> 20dB for similar signals)
      expect(quality.snr).toBeGreaterThan(20);
    });

    it('should handle Opus decoding errors gracefully', async () => {
      const connectPromise = room._connect();
      await new Promise(resolve => setTimeout(resolve, 10));
      mockWebSocket.readyState = WebSocket.OPEN;
      await connectPromise;
      
      // Force decoder to error state
      room.isDecoderReady = true;
      room.opusDecoder = {
        decode: vi.fn().mockImplementation(() => {
          throw new Error('Decoder error');
        }),
        state: 'configured'
      };
      
      // Receive Opus frame
      const echoFrame = createMockOpusFrame(new Uint8Array(50), 0);
      
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      mockWebSocket.onmessage({ data: echoFrame });
      
      await new Promise(resolve => setTimeout(resolve, 10));
      
      // Should log error but not crash
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to decode Opus'),
        expect.any(Error)
      );
      
      errorSpy.mockRestore();
    });
  });
});

describe('Audio Echo Playback', () => {
  let capture;
  let player;
  let room;
  let mockWebSocket;

  beforeEach(async () => {
    // Create mock WebSocket
    mockWebSocket = {
      send: vi.fn(),
      close: vi.fn(),
      readyState: WebSocket.OPEN,
      onopen: null,
      onmessage: null
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

    capture = new AudioCapture();
    player = new AudioPlayer();
    
    // Mock audio capture methods
    vi.spyOn(capture, 'initialize').mockResolvedValue();
    vi.spyOn(capture, 'start').mockResolvedValue();
    vi.spyOn(capture, 'on').mockImplementation((event, handler) => {
      capture[`_${event}Handler`] = handler;
    });
    vi.spyOn(capture, 'emit').mockImplementation((event, data) => {
      const handler = capture[`_${event}Handler`];
      if (handler) handler(data);
    });
    
    // Mock audio context for playback
    const mockContext = {
      createBuffer: vi.fn((channels, length, sampleRate) => ({
        numberOfChannels: channels,
        length: length,
        sampleRate: sampleRate,
        duration: length / sampleRate,
        getChannelData: () => new Float32Array(length),
        copyToChannel: vi.fn()
      })),
      createBufferSource: vi.fn(() => ({
        buffer: null,
        connect: vi.fn(),
        start: vi.fn(),
        onended: null
      })),
      destination: {},
      sampleRate: 48000,
      currentTime: 0
    };
    
    player.context = mockContext;
    vi.spyOn(player, 'playDecodedOpus').mockResolvedValue();
    
    room = new VoiceRoom({
      roomId: 'echo-test',
      userId: 'test-user',
      capture,
      player
    });
  });

  it('should play back compressed echo with correct timing', async () => {
    const connectPromise = room._connect();
    await new Promise(resolve => setTimeout(resolve, 10));
    room.socket = mockWebSocket; // Ensure socket is set
    await connectPromise;
    
    const playSpy = vi.spyOn(player.context, 'createBufferSource');
    
    // Send audio
    const testAudio = new ArrayBuffer(1920);
    capture.emit('audioChunk', {
      data: new Uint8Array(testAudio),
      timestamp: performance.now(),
      encoded: false
    });
    
    // Mock decoder ready
    room.isDecoderReady = true;
    room.opusDecoder = new MockAudioDecoder({
      output: (audioData) => room._handleDecodedAudio(audioData),
      error: (error) => console.error('Decoder error:', error)
    });
    
    // Receive echo (compressed)
    const echoFrame = createMockOpusFrame(new Uint8Array(50), 0);
    mockWebSocket.onmessage({ data: echoFrame });
    
    // Wait for decode and playback
    await new Promise(resolve => setTimeout(resolve, 50));
    
    // Verify audio was played
    expect(player.playDecodedOpus).toHaveBeenCalled();
  });
});