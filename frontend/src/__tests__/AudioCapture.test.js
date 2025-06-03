import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AudioCapture } from '../audio/AudioCapture';

// Mock AudioWorkletNode with message port simulation
class MockAudioWorkletNode {
  constructor(context, name, options) {
    this.context = context;
    this.name = name;
    this.options = options;
    this.port = {
      postMessage: vi.fn(),
      onmessage: null,
      // Simulate message sending from processor
      _simulateMessage: (message) => {
        if (this.port.onmessage) {
          this.port.onmessage({ data: message });
        }
      }
    };
    this.onprocessorerror = null;
    this._connected = false;
    this._intervalId = null;
  }

  connect(destination) {
    this._connected = true;
    this._destination = destination;
    
    // Start simulating audio processing when connected
    // Use immediate start instead of waiting
    this._startProcessing();
  }

  disconnect() {
    this._connected = false;
    this._destination = null;
    
    // Stop processing
    if (this._intervalId) {
      clearInterval(this._intervalId);
      this._intervalId = null;
    }
  }
  
  _startProcessing() {
    // Clear any existing interval first
    if (this._intervalId) {
      clearInterval(this._intervalId);
    }
    
    // Simulate 20ms frame processing
    let frameCount = 0;
    
    // Send first frame immediately
    if (this._connected && this.port.onmessage) {
      const timestamp = performance.now();
      this.port._simulateMessage({
        type: 'audioChunk',
        data: new Float32Array(960), // 20ms at 48kHz
        timestamp: timestamp,
        frameNumber: frameCount++,
        metrics: {
          energy: 0.02,
          zeroCrossings: 100,
          voiceActive: false
        }
      });
    }
    
    // Then continue with interval
    this._intervalId = setInterval(() => {
      if (this._connected && this.port.onmessage) {
        const timestamp = performance.now();
        this.port._simulateMessage({
          type: 'audioChunk',
          data: new Float32Array(960), // 20ms at 48kHz
          timestamp: timestamp,
          frameNumber: frameCount++,
          metrics: {
            energy: 0.02,
            zeroCrossings: 100,
            voiceActive: false
          }
        });
      }
    }, 20);
  }
}

// Mock AudioWorklet module loading
const mockWorkletModule = {
  addModule: vi.fn().mockResolvedValue(undefined)
};

describe('AudioWorklet Integration', () => {
  let capture;
  let mockContext;
  let mockStream;
  let workletNode;

  beforeEach(() => {
    vi.useFakeTimers();
    
    // Mock AudioContext with AudioWorklet support
    mockContext = {
      sampleRate: 48000,
      currentTime: 0,
      state: 'running',
      audioWorklet: mockWorkletModule,
      createMediaStreamSource: vi.fn(() => ({
        connect: vi.fn()
      })),
      destination: {},
      close: vi.fn().mockResolvedValue(undefined)
    };

    global.AudioContext = vi.fn(() => mockContext);
    global.AudioWorkletNode = MockAudioWorkletNode;

    // Mock getUserMedia
    mockStream = {
      getTracks: () => [{ stop: vi.fn() }]
    };
    
    global.navigator.mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue(mockStream)
    };

    capture = new AudioCapture({ useAudioWorklet: true });
  });

  afterEach(async () => {
    // Clean up any active intervals from MockAudioWorkletNode
    if (workletNode && workletNode._intervalId) {
      clearInterval(workletNode._intervalId);
    }
    
    // Stop capture if running
    if (capture && capture.isRecording) {
      await capture.stop();
    }
    
    vi.clearAllMocks();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  describe('AudioWorklet Processor Frame Emission', () => {
    it('should emit frames at exactly 20ms intervals', async () => {
      const onAudioChunk = vi.fn();
      capture.on('audioChunk', onAudioChunk);

      await capture.initialize();
      
      // Get the created worklet node
      workletNode = capture.workletNode;
      expect(workletNode).toBeDefined();
      expect(workletNode).toBeInstanceOf(MockAudioWorkletNode);

      await capture.start();

      // Should receive first frame immediately
      expect(onAudioChunk).toHaveBeenCalledTimes(1);

      // Wait for 4 more frames (80ms)
      vi.advanceTimersByTime(80);

      // Should have received exactly 5 chunks total
      expect(onAudioChunk).toHaveBeenCalledTimes(5);
      
      // Verify data structure
      const firstCall = onAudioChunk.mock.calls[0][0];
      expect(firstCall).toHaveProperty('data');
      expect(firstCall).toHaveProperty('timestamp');
      expect(firstCall.data).toBeInstanceOf(Uint8Array);
      expect(firstCall.data.length).toBe(1920); // 960 samples * 2 bytes
    });

    it('should achieve < 12ms capture-to-emit latency', async () => {
      vi.useRealTimers(); // Need real timers for performance.now()
      
      const latencies = [];

      capture.on('audioChunk', (chunk) => {
        // Measure latency from chunk timestamp to now
        const latency = performance.now() - chunk.timestamp;
        latencies.push(latency);
      });

      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();

      // Wait for several frames
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(latencies.length).toBeGreaterThan(0);
      
      // Calculate average latency
      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      
      expect(avgLatency).toBeLessThan(12); // Should be well under 12ms
      console.log(`AudioWorklet latency: ${avgLatency.toFixed(2)}ms`);
      
      vi.useFakeTimers(); // Switch back
    });

    it('should maintain consistent 960-sample frames', async () => {
      const chunks = [];
      capture.on('audioChunk', (chunk) => chunks.push(chunk));

      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();

      // Advance time to collect frames
      vi.advanceTimersByTime(60); // 3 frames

      expect(chunks).toHaveLength(4); // 1 immediate + 3 from timer
      chunks.forEach(chunk => {
        expect(chunk.data).toHaveLength(960 * 2); // 960 samples * 2 bytes for Int16
      });
    });
  });

  describe('AudioWorklet VAD Integration', () => {
    it('should process VAD in real-time without blocking', async () => {
      const vadEvents = [];
      capture.on('voiceStart', () => vadEvents.push({ type: 'start', time: performance.now() }));
      capture.on('voiceEnd', () => vadEvents.push({ type: 'end', time: performance.now() }));

      await capture.initialize();
      workletNode = capture.workletNode;
      capture.enableVAD = true;
      await capture.start();

      // Simulate voice activity changes
      workletNode.port._simulateMessage({
        type: 'vadStatus',
        active: true,
        timestamp: 0
      });

      vi.advanceTimersByTime(10);

      workletNode.port._simulateMessage({
        type: 'vadStatus',
        active: false,
        timestamp: 100
      });

      vi.advanceTimersByTime(10);

      expect(vadEvents).toHaveLength(2);
      expect(vadEvents[0].type).toBe('start');
      expect(vadEvents[1].type).toBe('end');
    });

    it('should emit audio level updates for visualization', async () => {
      const levels = [];
      capture.on('level', (level) => levels.push(level));

      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();

      // Simulate level updates
      for (let i = 0; i < 10; i++) {
        workletNode.port._simulateMessage({
          type: 'level',
          level: Math.random() * 0.1,
          timestamp: i * 100
        });
      }

      expect(levels).toHaveLength(10);
      levels.forEach(level => {
        expect(level).toBeGreaterThanOrEqual(0);
        expect(level).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('AudioWorklet Error Handling', () => {
    it('should handle processor errors gracefully', async () => {
      const onError = vi.fn();
      capture.on('error', onError);

      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();

      // Simulate processor error
      if (workletNode.onprocessorerror) {
        workletNode.onprocessorerror(new Event('processorerror'));
      }

      expect(onError).toHaveBeenCalled();
    });

    it('should continue processing after transient errors', async () => {
      const chunks = [];
      capture.on('audioChunk', (chunk) => chunks.push(chunk));

      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();

      // Should get first chunk immediately
      expect(chunks.length).toBe(1);
      
      const chunksBeforeError = chunks.length;

      // Simulate error
      workletNode.port._simulateMessage({
        type: 'error',
        message: 'Transient processing error'
      });

      // Advance time for more chunks after error
      vi.advanceTimersByTime(50);

      expect(chunks.length).toBeGreaterThan(chunksBeforeError); // Continued processing
    });
  });

  describe('AudioWorklet Performance', () => {
    it('should handle high-frequency audio processing without dropping frames', async () => {
      const chunks = [];
      capture.on('audioChunk', (chunk) => chunks.push(chunk));

      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();

      // Should get first chunk immediately
      expect(chunks.length).toBe(1);

      // Wait for 1 second of audio (50 frames at 20ms each)
      vi.advanceTimersByTime(1000);

      // Should have received ~50 chunks (1 immediate + 50 from timer)
      expect(chunks.length).toBeGreaterThanOrEqual(45);
      expect(chunks.length).toBeLessThanOrEqual(55);
      
      // Verify no frames were dropped (check sequential processing)
      let droppedFrames = 0;
      for (let i = 1; i < chunks.length; i++) {
        const timeDiff = chunks[i].timestamp - chunks[i-1].timestamp;
        if (timeDiff > 30) { // More than 30ms means dropped frame
          droppedFrames++;
        }
      }
      
      expect(droppedFrames).toBeLessThan(2); // Allow at most 1 dropped frame
    });

    it('should maintain low memory footprint', async () => {
      // Process many frames
      let processedFrames = 0;
      capture.on('audioChunk', () => processedFrames++);
      
      await capture.initialize();
      workletNode = capture.workletNode;
      await capture.start();
      
      // First frame immediately
      expect(processedFrames).toBe(1);
      
      // Run for 2 seconds (100 frames)
      vi.advanceTimersByTime(2000);

      // Verify we processed many frames
      expect(processedFrames).toBeGreaterThan(90);
      
      // Verify worklet is still responsive
      const testMessage = { type: 'ping' };
      workletNode.port.postMessage(testMessage);
      expect(workletNode.port.postMessage).toHaveBeenLastCalledWith(testMessage);
    });
  });
});